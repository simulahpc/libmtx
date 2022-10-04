/* This file is part of Libmtx.
 *
 * Copyright (C) 2022 James D. Trotter
 *
 * Libmtx is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Libmtx is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Libmtx.  If not, see <https://www.gnu.org/licenses/>.
 *
 * Authors: James D. Trotter <james@simula.no>
 * Last modified: 2022-10-03
 *
 * Data structures for matrices in coordinate format.
 */

#include <libmtx/libmtx-config.h>

#include <libmtx/error.h>
#include <libmtx/linalg/field.h>
#include <libmtx/linalg/local/matrix.h>
#include <libmtx/linalg/omp/csr.h>
#include <libmtx/mtxfile/mtxfile.h>
#include <libmtx/linalg/precision.h>
#include <libmtx/util/sort.h>
#include <libmtx/linalg/base/vector.h>
#include <libmtx/linalg/omp/vector.h>
#include <libmtx/linalg/local/vector.h>

#include <errno.h>

#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * matrix properties
 */

/**
 * ‘mtxompcsr_field()’ gets the field of a matrix.
 */
enum mtxfield mtxompcsr_field(const struct mtxompcsr * A)
{
    return mtxompvector_field(&A->a);
}

/**
 * ‘mtxompcsr_precision()’ gets the precision of a matrix.
 */
enum mtxprecision mtxompcsr_precision(const struct mtxompcsr * A)
{
    return mtxompvector_precision(&A->a);
}

/**
 * ‘mtxompcsr_symmetry()’ gets the symmetry of a matrix.
 */
enum mtxsymmetry mtxompcsr_symmetry(const struct mtxompcsr * A)
{
    return A->symmetry;
}

/**
 * ‘mtxompcsr_num_rows()’ gets the number of matrix rows.
 */
int mtxompcsr_num_rows(const struct mtxompcsr * A)
{
    return A->num_rows;
}

/**
 * ‘mtxompcsr_num_columns()’ gets the number of matrix columns.
 */
int mtxompcsr_num_columns(const struct mtxompcsr * A)
{
    return A->num_columns;
}

/**
 * ‘mtxompcsr_num_nonzeros()’ gets the number of the number of
 *  nonzero matrix entries, including those represented implicitly due
 *  to symmetry.
 */
int64_t mtxompcsr_num_nonzeros(const struct mtxompcsr * A)
{
    return A->num_nonzeros;
}

/**
 * ‘mtxompcsr_size()’ gets the number of explicitly stored
 * nonzeros of a matrix.
 */
int64_t mtxompcsr_size(const struct mtxompcsr * A)
{
    return A->size;
}

/**
 * ‘mtxompcsr_rowcolidx()’ gets the row and column indices of the
 * explicitly stored matrix nonzeros.
 *
 * The arguments ‘rowidx’ and ‘colidx’ may be ‘NULL’ or must point to
 * an arrays of length ‘size’.
 */
int mtxompcsr_rowcolidx(
    const struct mtxompcsr * A,
    int64_t size,
    int * rowidx,
    int * colidx)
{
    if (size != A->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (rowidx) {
        for (int i = 0; i < A->num_rows; i++) {
            for (int64_t k = A->rowptr[i]; k < A->rowptr[i+1]; k++)
                rowidx[k] = i;
        }
    }
    if (colidx) { for (int64_t k = 0; k < A->size; k++) colidx[k] = A->colidx[k]; }
    return MTX_SUCCESS;
}

/*
 * memory management
 */

/**
 * ‘mtxompcsr_free()’ frees storage allocated for a matrix.
 */
void mtxompcsr_free(
    struct mtxompcsr * A)
{
    mtxompvector_free(&A->diag);
    mtxompvector_free(&A->a);
    free(A->colidx);
    free(A->rowptr);
}

/**
 * ‘mtxompcsr_alloc_copy()’ allocates a copy of a matrix without
 * initialising the values.
 */
int mtxompcsr_alloc_copy(
    struct mtxompcsr * dst,
    const struct mtxompcsr * src)
{
    return mtxompcsr_alloc_rows(
        dst, src->a.base.field, src->a.base.precision, src->symmetry,
        src->num_rows, src->num_columns, src->rowptr, src->colidx);
}

/**
 * ‘mtxompcsr_init_copy()’ allocates a copy of a matrix and also
 * copies the values.
 */
int mtxompcsr_init_copy(
    struct mtxompcsr * dst,
    const struct mtxompcsr * src)
{
    int err = mtxompcsr_alloc_copy(dst, src);
    if (err) return err;
    err = mtxompcsr_copy(dst, src);
    if (err) { mtxompcsr_free(dst); return err; }
    return MTX_SUCCESS;
}

/*
 * initialise matrices from entrywise data in coordinate format
 */

/**
 * ‘mtxompcsr_alloc_entries()’ allocates a matrix from entrywise
 * data in coordinate format.
 *
 * If it is not ‘NULL’, then ‘perm’ must point to an array of length
 * ‘size’. Because the sparse matrix storage may internally reorder
 * the specified nonzero entries, this array is used to store the
 * permutation applied to the specified nonzero entries.
 */
int mtxompcsr_alloc_entries(
    struct mtxompcsr * A,
    enum mtxfield field,
    enum mtxprecision precision,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    int idxstride,
    int idxbase,
    const int * rowidx,
    const int * colidx,
    int64_t * perm)
{
    A->symmetry = symmetry;
    A->num_rows = num_rows;
    A->num_columns = num_columns;
    if (__builtin_mul_overflow(num_rows, num_columns, &A->num_entries)) {
        errno = EOVERFLOW; return MTX_ERR_ERRNO;
    }
    A->num_nonzeros = 0;
    A->size = size;

    /* compute row pointers and sort column indices rowwise */
    A->rowptr = malloc((num_rows+1) * sizeof(int64_t));
    if (!A->rowptr) return MTX_ERR_ERRNO;
    for (int i = 0; i <= num_rows; i++) A->rowptr[i] = 0;
    for (int64_t k = 0; k < size; k++) {
        int i = *(const int *)((const char *) rowidx+k*idxstride)-idxbase;
        A->rowptr[i+1]++;
    }
    for (int i = 1; i <= num_rows; i++) A->rowptr[i] += A->rowptr[i-1];
    A->colidx = malloc(size * sizeof(int));
    if (!A->colidx) { free(A->rowptr); return MTX_ERR_ERRNO; }
    for (int64_t k = 0; k < size; k++) {
        int i = *(const int *)((const char *) rowidx+k*idxstride)-idxbase;
        int j = *(const int *)((const char *) colidx+k*idxstride)-idxbase;
        A->colidx[A->rowptr[i]++] = j;
        if (perm) perm[k] = A->rowptr[i]-1;
    }
    for (int i = num_rows; i > 0; i--) A->rowptr[i] = A->rowptr[i-1];
    A->rowptr[0] = 0;

    /* assign an equal number of rows to each thread */
    int num_threads = omp_get_max_threads();
    int64_t * offsets = malloc((num_threads+1) * sizeof(int64_t));
    if (!offsets) { free(A->colidx); free(A->rowptr); return MTX_ERR_ERRNO; }
    for (int t = 0; t < num_threads; t++) {
        int i = t * ((num_rows+num_threads-1) / num_threads);
        offsets[t] = A->rowptr[i <= num_rows ? i : num_rows];
    }
    offsets[num_threads] = A->rowptr[num_rows];
    int err = mtxompvector_alloc_custom(
        &A->a, num_threads, offsets, omp_sched_static, 0, field, precision, size);
    if (err) { free(offsets); free(A->colidx); free(A->rowptr); return err; }
    free(offsets);

    /* extract diagonals for symmetric and Hermitian matrices */
    if (num_rows == num_columns &&
        (symmetry == mtx_symmetric ||
         symmetry == mtx_skew_symmetric ||
         symmetry == mtx_hermitian))
    {
        int64_t num_diagonals = 0;
        for (int i = 0; i < num_rows; i++) {
            for (int64_t k = A->rowptr[i]; k < A->rowptr[i+1]; k++) {
                if (i == A->colidx[k]) num_diagonals++;
            }
        }
        A->num_nonzeros = 2*A->size-num_diagonals;
        err = mtxompvector_alloc_packed(
            &A->diag, field, precision, A->size, num_diagonals, NULL);
        if (err) {
            mtxompvector_free(&A->a);
            free(A->colidx); free(A->rowptr);
            return err;
        }
        int64_t * diagidx = A->diag.base.idx;
        num_diagonals = 0;
        for (int i = 0; i < num_rows; i++) {
            for (int64_t k = A->rowptr[i]; k < A->rowptr[i+1]; k++) {
                if (i == A->colidx[k]) diagidx[num_diagonals++] = k;
            }
        }
    } else if (symmetry == mtx_unsymmetric) {
        A->num_nonzeros = A->size;
        err = mtxompvector_alloc_packed(
            &A->diag, field, precision, A->size, 0, NULL);
        if (err) {
            mtxompvector_free(&A->a);
            free(A->colidx); free(A->rowptr);
            return err;
        }
    } else { return MTX_ERR_INVALID_SYMMETRY; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxompcsr_init_entries_real_single()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * real, single precision coefficients.
 */
int mtxompcsr_init_entries_real_single(
    struct mtxompcsr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx,
    const float * data)
{
    int64_t * perm = malloc(size * sizeof(perm));
    if (!perm) return MTX_ERR_ERRNO;
    int err = mtxompcsr_alloc_entries(
        A, mtx_field_real, mtx_single, symmetry, num_rows, num_columns,
        size, sizeof(*rowidx), 0, rowidx, colidx, perm);
    if (err) { free(perm); return err; }
    err = mtxompvector_setzero(&A->a);
    if (err) { mtxompcsr_free(A); free(perm); return err; }
    struct mtxompvector x;
    x.num_threads = A->a.num_threads;
    x.offsets = A->a.offsets;
    x.sched = A->a.sched;
    x.chunk_size = A->a.chunk_size;
    x.base.idx = perm;
    x.base.field = mtx_field_real;
    x.base.precision = mtx_single;
    x.base.size = A->size;
    x.base.num_nonzeros = size;
    x.base.data.real_single = (float *) data;
    err = mtxompvector_ussc(&A->a, &x);
    if (err) { mtxompcsr_free(A); free(perm); return err; }
    free(perm);
    return MTX_SUCCESS;
}

/**
 * ‘mtxompcsr_init_entries_real_double()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * real, double precision coefficients.
 */
int mtxompcsr_init_entries_real_double(
    struct mtxompcsr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx,
    const double * data)
{
    int64_t * perm = malloc(size * sizeof(perm));
    if (!perm) return MTX_ERR_ERRNO;
    int err = mtxompcsr_alloc_entries(
        A, mtx_field_real, mtx_double, symmetry, num_rows, num_columns,
        size, sizeof(*rowidx), 0, rowidx, colidx, perm);
    if (err) { free(perm); return err; }
    err = mtxompvector_setzero(&A->a);
    if (err) { mtxompcsr_free(A); free(perm); return err; }
    struct mtxompvector x;
    x.num_threads = A->a.num_threads;
    x.offsets = A->a.offsets;
    x.sched = A->a.sched;
    x.chunk_size = A->a.chunk_size;
    x.base.idx = perm;
    x.base.field = mtx_field_real;
    x.base.precision = mtx_double;
    x.base.size = A->size;
    x.base.num_nonzeros = size;
    x.base.data.real_double = (double *) data;
    err = mtxompvector_ussc(&A->a, &x);
    if (err) { mtxompcsr_free(A); free(perm); return err; }
    free(perm);
    return MTX_SUCCESS;
}

/**
 * ‘mtxompcsr_init_entries_complex_single()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * complex, single precision coefficients.
 */
int mtxompcsr_init_entries_complex_single(
    struct mtxompcsr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx,
    const float (* data)[2])
{
    int64_t * perm = malloc(size * sizeof(perm));
    if (!perm) return MTX_ERR_ERRNO;
    int err = mtxompcsr_alloc_entries(
        A, mtx_field_complex, mtx_single, symmetry, num_rows, num_columns,
        size, sizeof(*rowidx), 0, rowidx, colidx, perm);
    if (err) { free(perm); return err; }
    err = mtxompvector_setzero(&A->a);
    if (err) { mtxompcsr_free(A); free(perm); return err; }
    struct mtxompvector x;
    x.num_threads = A->a.num_threads;
    x.offsets = A->a.offsets;
    x.sched = A->a.sched;
    x.chunk_size = A->a.chunk_size;
    x.base.idx = perm;
    x.base.field = mtx_field_complex;
    x.base.precision = mtx_single;
    x.base.size = A->size;
    x.base.num_nonzeros = size;
    x.base.data.complex_single = (float (*)[2]) data;
    err = mtxompvector_ussc(&A->a, &x);
    if (err) { mtxompcsr_free(A); free(perm); return err; }
    free(perm);
    return MTX_SUCCESS;
}

/**
 * ‘mtxompcsr_init_entries_complex_double()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * complex, double precision coefficients.
 */
int mtxompcsr_init_entries_complex_double(
    struct mtxompcsr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx,
    const double (* data)[2])
{
    int64_t * perm = malloc(size * sizeof(perm));
    if (!perm) return MTX_ERR_ERRNO;
    int err = mtxompcsr_alloc_entries(
        A, mtx_field_complex, mtx_double, symmetry, num_rows, num_columns,
        size, sizeof(*rowidx), 0, rowidx, colidx, perm);
    if (err) { free(perm); return err; }
    err = mtxompvector_setzero(&A->a);
    if (err) { mtxompcsr_free(A); free(perm); return err; }
    struct mtxompvector x;
    x.num_threads = A->a.num_threads;
    x.offsets = A->a.offsets;
    x.sched = A->a.sched;
    x.chunk_size = A->a.chunk_size;
    x.base.idx = perm;
    x.base.field = mtx_field_complex;
    x.base.precision = mtx_double;
    x.base.size = A->size;
    x.base.num_nonzeros = size;
    x.base.data.complex_double = (double (*)[2]) data;
    err = mtxompvector_ussc(&A->a, &x);
    if (err) { mtxompcsr_free(A); free(perm); return err; }
    free(perm);
    return MTX_SUCCESS;
}

/**
 * ‘mtxompcsr_init_entries_integer_single()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * integer, single precision coefficients.
 */
int mtxompcsr_init_entries_integer_single(
    struct mtxompcsr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx,
    const int32_t * data)
{
    int64_t * perm = malloc(size * sizeof(perm));
    if (!perm) return MTX_ERR_ERRNO;
    int err = mtxompcsr_alloc_entries(
        A, mtx_field_integer, mtx_single, symmetry, num_rows, num_columns,
        size, sizeof(*rowidx), 0, rowidx, colidx, perm);
    if (err) { free(perm); return err; }
    err = mtxompvector_setzero(&A->a);
    if (err) { mtxompcsr_free(A); free(perm); return err; }
    struct mtxompvector x;
    x.num_threads = A->a.num_threads;
    x.offsets = A->a.offsets;
    x.sched = A->a.sched;
    x.chunk_size = A->a.chunk_size;
    x.base.idx = perm;
    x.base.field = mtx_field_integer;
    x.base.precision = mtx_single;
    x.base.size = A->size;
    x.base.num_nonzeros = size;
    x.base.data.integer_single = (int32_t *) data;
    err = mtxompvector_ussc(&A->a, &x);
    if (err) { mtxompcsr_free(A); free(perm); return err; }
    free(perm);
    return MTX_SUCCESS;
}

/**
 * ‘mtxompcsr_init_entries_integer_double()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * integer, double precision coefficients.
 */
int mtxompcsr_init_entries_integer_double(
    struct mtxompcsr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx,
    const int64_t * data)
{
    int64_t * perm = malloc(size * sizeof(perm));
    if (!perm) return MTX_ERR_ERRNO;
    int err = mtxompcsr_alloc_entries(
        A, mtx_field_integer, mtx_double, symmetry, num_rows, num_columns,
        size, sizeof(*rowidx), 0, rowidx, colidx, perm);
    if (err) { free(perm); return err; }
    err = mtxompvector_setzero(&A->a);
    if (err) { mtxompcsr_free(A); free(perm); return err; }
    struct mtxompvector x;
    x.num_threads = A->a.num_threads;
    x.offsets = A->a.offsets;
    x.sched = A->a.sched;
    x.chunk_size = A->a.chunk_size;
    x.base.idx = perm;
    x.base.field = mtx_field_integer;
    x.base.precision = mtx_double;
    x.base.size = A->size;
    x.base.num_nonzeros = size;
    x.base.data.integer_double = (int64_t *) data;
    err = mtxompvector_ussc(&A->a, &x);
    if (err) { mtxompcsr_free(A); free(perm); return err; }
    free(perm);
    return MTX_SUCCESS;
}

/**
 * ‘mtxompcsr_init_entries_pattern()’ allocates and initialises a
 * matrix from entrywise data in coordinate format with boolean
 * coefficients.
 */
int mtxompcsr_init_entries_pattern(
    struct mtxompcsr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx)
{
    return mtxompcsr_alloc_entries(
        A, mtx_field_pattern, mtx_single, symmetry, num_rows, num_columns,
        size, sizeof(*rowidx), 0, rowidx, colidx, NULL);
}

/*
 * initialise matrices from entrywise data in coordinate format with
 * specified strides
 */

/**
 * ‘mtxompcsr_init_entries_strided_real_single()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * real, single precision coefficients.
 */
int mtxompcsr_init_entries_strided_real_single(
    struct mtxompcsr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    int idxstride,
    int idxbase,
    const int * rowidx,
    const int * colidx,
    int datastride,
    const float * data);

/**
 * ‘mtxompcsr_init_entries_strided_real_double()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * real, double precision coefficients.
 */
int mtxompcsr_init_entries_strided_real_double(
    struct mtxompcsr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    int idxstride,
    int idxbase,
    const int * rowidx,
    const int * colidx,
    int datastride,
    const double * data);

/**
 * ‘mtxompcsr_init_entries_strided_complex_single()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * complex, single precision coefficients.
 */
int mtxompcsr_init_entries_strided_complex_single(
    struct mtxompcsr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    int idxstride,
    int idxbase,
    const int * rowidx,
    const int * colidx,
    int datastride,
    const float (* data)[2]);

/**
 * ‘mtxompcsr_init_entries_strided_complex_double()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * complex, double precision coefficients.
 */
int mtxompcsr_init_entries_strided_complex_double(
    struct mtxompcsr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    int idxstride,
    int idxbase,
    const int * rowidx,
    const int * colidx,
    int datastride,
    const double (* data)[2]);

/**
 * ‘mtxompcsr_init_entries_strided_integer_single()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * integer, single precision coefficients.
 */
int mtxompcsr_init_entries_strided_integer_single(
    struct mtxompcsr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    int idxstride,
    int idxbase,
    const int * rowidx,
    const int * colidx,
    int datastride,
    const int32_t * data);

/**
 * ‘mtxompcsr_init_entries_strided_integer_double()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * integer, double precision coefficients.
 */
int mtxompcsr_init_entries_strided_integer_double(
    struct mtxompcsr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    int idxstride,
    int idxbase,
    const int * rowidx,
    const int * colidx,
    int datastride,
    const int64_t * data);

/**
 * ‘mtxompcsr_init_entries_strided_pattern()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * boolean coefficients.
 */
int mtxompcsr_init_entries_strided_pattern(
    struct mtxompcsr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    int idxstride,
    int idxbase,
    const int * rowidx,
    const int * colidx);

/*
 * initialise matrices from row-wise data in compressed row format
 */

/**
 * ‘mtxompcsr_alloc_rows()’ allocates a matrix from row-wise data
 * in compressed row format.
 */
int mtxompcsr_alloc_rows(
    struct mtxompcsr * A,
    enum mtxfield field,
    enum mtxprecision precision,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx)
{
    A->symmetry = symmetry;
    A->num_rows = num_rows;
    A->num_columns = num_columns;
    if (__builtin_mul_overflow(num_rows, num_columns, &A->num_entries)) {
        errno = EOVERFLOW; return MTX_ERR_ERRNO;
    }
    A->num_nonzeros = 0;
    A->size = rowptr[num_rows];

    /* copy row pointers and column indices */
    A->rowptr = malloc((num_rows+1) * sizeof(int64_t));
    if (!A->rowptr) return MTX_ERR_ERRNO;
    for (int i = 0; i <= num_rows; i++) A->rowptr[i] = rowptr[i];
    A->colidx = malloc(A->size * sizeof(int));
    if (!A->colidx) { free(A->rowptr); return MTX_ERR_ERRNO; }
    for (int64_t k = 0; k < A->size; k++) A->colidx[k] = colidx[k];

    /* assign an equal number of rows to each thread */
    int num_threads = omp_get_max_threads();
    int64_t * offsets = malloc((num_threads+1) * sizeof(int64_t));
    if (!offsets) { free(A->colidx); free(A->rowptr); return MTX_ERR_ERRNO; }
    for (int t = 0; t < num_threads; t++) {
        int i = t * ((num_rows+num_threads-1) / num_threads);
        offsets[t] = A->rowptr[i <= num_rows ? i : num_rows];
    }
    offsets[num_threads] = A->rowptr[num_rows];
    int err = mtxompvector_alloc_custom(
        &A->a, num_threads, offsets, omp_sched_static, 0, field, precision, A->size);
    if (err) { free(offsets); free(A->colidx); free(A->rowptr); return err; }
    free(offsets);

    /* extract diagonals for symmetric and Hermitian matrices */
    if (num_rows == num_columns &&
        (symmetry == mtx_symmetric ||
         symmetry == mtx_skew_symmetric ||
         symmetry == mtx_hermitian))
    {
        int64_t num_diagonals = 0;
        for (int i = 0; i < num_rows; i++) {
            for (int64_t k = A->rowptr[i]; k < A->rowptr[i+1]; k++) {
                if (i == A->colidx[k]) num_diagonals++;
            }
        }
        A->num_nonzeros = 2*A->size-num_diagonals;
        err = mtxompvector_alloc_packed(
            &A->diag, field, precision, A->size, num_diagonals, NULL);
        if (err) {
            mtxompvector_free(&A->a);
            free(A->colidx); free(A->rowptr);
            return err;
        }
        int64_t * diagidx = A->diag.base.idx;
        num_diagonals = 0;
        for (int i = 0; i < num_rows; i++) {
            for (int64_t k = A->rowptr[i]; k < A->rowptr[i+1]; k++) {
                if (i == A->colidx[k]) diagidx[num_diagonals++] = k;
            }
        }
    } else if (symmetry == mtx_unsymmetric) {
        A->num_nonzeros = A->size;
        err = mtxompvector_alloc_packed(
            &A->diag, field, precision, A->size, 0, NULL);
        if (err) {
            mtxompvector_free(&A->a);
            free(A->colidx); free(A->rowptr);
            return err;
        }
    } else { return MTX_ERR_INVALID_SYMMETRY; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxompcsr_init_rows_real_single()’ allocates and initialises a
 * matrix from row-wise data in compressed row format with real,
 * single precision coefficients.
 */
int mtxompcsr_init_rows_real_single(
    struct mtxompcsr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const float * data)
{
    int err = mtxompcsr_alloc_rows(
        A, mtx_field_real, mtx_single, symmetry,
	num_rows, num_columns, rowptr, colidx);
    if (err) return err;
    err = mtxompvector_set_real_single(&A->a, rowptr[num_rows], sizeof(*data), data);
    if (err) { mtxompcsr_free(A); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxompcsr_init_rows_real_double()’ allocates and initialises a
 * matrix from row-wise data in compressed row format with real,
 * double precision coefficients.
 */
int mtxompcsr_init_rows_real_double(
    struct mtxompcsr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const double * data)
{
    int err = mtxompcsr_alloc_rows(
        A, mtx_field_real, mtx_double, symmetry,
	num_rows, num_columns, rowptr, colidx);
    if (err) return err;
    err = mtxompvector_set_real_double(&A->a, rowptr[num_rows], sizeof(*data), data);
    if (err) { mtxompcsr_free(A); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxompcsr_init_rows_complex_single()’ allocates and
 * initialises a matrix from row-wise data in compressed row format
 * with complex, single precision coefficients.
 */
int mtxompcsr_init_rows_complex_single(
    struct mtxompcsr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const float (* data)[2])
{
    int err = mtxompcsr_alloc_rows(
        A, mtx_field_complex, mtx_single, symmetry,
	num_rows, num_columns, rowptr, colidx);
    if (err) return err;
    err = mtxompvector_set_complex_single(&A->a, rowptr[num_rows], sizeof(*data), data);
    if (err) { mtxompcsr_free(A); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxompcsr_init_rows_complex_double()’ allocates and
 * initialises a matrix from row-wise data in compressed row format
 * with complex, double precision coefficients.
 */
int mtxompcsr_init_rows_complex_double(
    struct mtxompcsr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const double (* data)[2])
{
    int err = mtxompcsr_alloc_rows(
        A, mtx_field_complex, mtx_double, symmetry,
	num_rows, num_columns, rowptr, colidx);
    if (err) return err;
    err = mtxompvector_set_complex_double(&A->a, rowptr[num_rows], sizeof(*data), data);
    if (err) { mtxompcsr_free(A); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxompcsr_init_rows_integer_single()’ allocates and
 * initialises a matrix from row-wise data in compressed row format
 * with integer, single precision coefficients.
 */
int mtxompcsr_init_rows_integer_single(
    struct mtxompcsr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const int32_t * data)
{
    int err = mtxompcsr_alloc_rows(
        A, mtx_field_integer, mtx_single, symmetry,
	num_rows, num_columns, rowptr, colidx);
    if (err) return err;
    err = mtxompvector_set_integer_single(&A->a, rowptr[num_rows], sizeof(*data), data);
    if (err) { mtxompcsr_free(A); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxompcsr_init_rows_integer_double()’ allocates and
 * initialises a matrix from row-wise data in compressed row format
 * with integer, double precision coefficients.
 */
int mtxompcsr_init_rows_integer_double(
    struct mtxompcsr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const int64_t * data)
{
    int err = mtxompcsr_alloc_rows(
        A, mtx_field_integer, mtx_double, symmetry,
	num_rows, num_columns, rowptr, colidx);
    if (err) return err;
    err = mtxompvector_set_integer_double(&A->a, rowptr[num_rows], sizeof(*data), data);
    if (err) { mtxompcsr_free(A); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxompcsr_init_rows_pattern()’ allocates and initialises a
 * matrix from row-wise data in compressed row format with boolean
 * coefficients.
 */
int mtxompcsr_init_rows_pattern(
    struct mtxompcsr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx)
{
    return mtxompcsr_alloc_rows(
        A, mtx_field_pattern, mtx_single, symmetry,
	num_rows, num_columns, rowptr, colidx);
}

/*
 * initialise matrices from column-wise data in compressed column
 * format
 */

/**
 * ‘mtxompcsr_alloc_columns()’ allocates a matrix from column-wise
 * data in compressed column format.
 */
int mtxompcsr_alloc_columns(
    struct mtxompcsr * A,
    enum mtxfield field,
    enum mtxprecision precision,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx);

/**
 * ‘mtxompcsr_init_columns_real_single()’ allocates and
 * initialises a matrix from column-wise data in compressed column
 * format with real, single precision coefficients.
 */
int mtxompcsr_init_columns_real_single(
    struct mtxompcsr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx,
    const float * data);

/**
 * ‘mtxompcsr_init_columns_real_double()’ allocates and
 * initialises a matrix from column-wise data in compressed column
 * format with real, double precision coefficients.
 */
int mtxompcsr_init_columns_real_double(
    struct mtxompcsr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx,
    const double * data);

/**
 * ‘mtxompcsr_init_columns_complex_single()’ allocates and
 * initialises a matrix from column-wise data in compressed column
 * format with complex, single precision coefficients.
 */
int mtxompcsr_init_columns_complex_single(
    struct mtxompcsr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx,
    const float (* data)[2]);

/**
 * ‘mtxompcsr_init_columns_complex_double()’ allocates and
 * initialises a matrix from column-wise data in compressed column
 * format with complex, double precision coefficients.
 */
int mtxompcsr_init_columns_complex_double(
    struct mtxompcsr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx,
    const double (* data)[2]);

/**
 * ‘mtxompcsr_init_columns_integer_single()’ allocates and
 * initialises a matrix from column-wise data in compressed column
 * format with integer, single precision coefficients.
 */
int mtxompcsr_init_columns_integer_single(
    struct mtxompcsr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx,
    const int32_t * data);

/**
 * ‘mtxompcsr_init_columns_integer_double()’ allocates and
 * initialises a matrix from column-wise data in compressed column
 * format with integer, double precision coefficients.
 */
int mtxompcsr_init_columns_integer_double(
    struct mtxompcsr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx,
    const int64_t * data);

/**
 * ‘mtxompcsr_init_columns_pattern()’ allocates and initialises a
 * matrix from column-wise data in compressed column format with
 * boolean coefficients.
 */
int mtxompcsr_init_columns_pattern(
    struct mtxompcsr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx);

/*
 * initialise matrices from a list of dense cliques
 */

/**
 * ‘mtxompcsr_alloc_cliques()’ allocates a matrix from a list of
 * cliques.
 */
int mtxompcsr_alloc_cliques(
    struct mtxompcsr * A,
    enum mtxfield field,
    enum mtxprecision precision,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_cliques,
    const int64_t * cliqueptr,
    const int * rowidx,
    const int * colidx);

/**
 * ‘mtxompcsr_init_cliques_real_single()’ allocates and
 * initialises a matrix from a list of cliques with real, single
 * precision coefficients.
 */
int mtxompcsr_init_cliques_real_single(
    struct mtxompcsr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_cliques,
    const int64_t * cliqueptr,
    const int * rowidx,
    const int * colidx,
    const float * data);

/**
 * ‘mtxompcsr_init_cliques_real_double()’ allocates and
 * initialises a matrix from a list of cliques with real, double
 * precision coefficients.
 */
int mtxompcsr_init_cliques_real_double(
    struct mtxompcsr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_cliques,
    const int64_t * cliqueptr,
    const int * rowidx,
    const int * colidx,
    const double * data);

/**
 * ‘mtxompcsr_init_cliques_complex_single()’ allocates and
 * initialises a matrix from a list of cliques with complex, single
 * precision coefficients.
 */
int mtxompcsr_init_cliques_complex_single(
    struct mtxompcsr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_cliques,
    const int64_t * cliqueptr,
    const int * rowidx,
    const int * colidx,
    const float (* data)[2]);

/**
 * ‘mtxompcsr_init_cliques_complex_double()’ allocates and
 * initialises a matrix from a list of cliques with complex, double
 * precision coefficients.
 */
int mtxompcsr_init_cliques_complex_double(
    struct mtxompcsr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_cliques,
    const int64_t * cliqueptr,
    const int * rowidx,
    const int * colidx,
    const double (* data)[2]);

/**
 * ‘mtxompcsr_init_cliques_integer_single()’ allocates and
 * initialises a matrix from a list of cliques with integer, single
 * precision coefficients.
 */
int mtxompcsr_init_cliques_integer_single(
    struct mtxompcsr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_cliques,
    const int64_t * cliqueptr,
    const int * rowidx,
    const int * colidx,
    const int32_t * data);

/**
 * ‘mtxompcsr_init_cliques_integer_double()’ allocates and
 * initialises a matrix from a list of cliques with integer, double
 * precision coefficients.
 */
int mtxompcsr_init_cliques_integer_double(
    struct mtxompcsr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_cliques,
    const int64_t * cliqueptr,
    const int * rowidx,
    const int * colidx,
    const int64_t * data);

/**
 * ‘mtxompcsr_init_cliques_pattern()’ allocates and initialises a
 * matrix from a list of cliques with boolean coefficients.
 */
int mtxompcsr_init_cliques_pattern(
    struct mtxompcsr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_cliques,
    const int64_t * cliqueptr,
    const int * rowidx,
    const int * colidx);

/*
 * modifying values
 */

/**
 * ‘mtxompcsr_setzero()’ sets every value of a matrix to zero.
 */
int mtxompcsr_setzero(
    struct mtxompcsr * A)
{
    return mtxompvector_setzero(&A->a);
}

/**
 * ‘mtxompcsr_set_real_single()’ sets values of a matrix based on
 * an array of single precision floating point numbers.
 */
int mtxompcsr_set_real_single(
    struct mtxompcsr * A,
    int64_t size,
    int stride,
    const float * a)
{
    return mtxompvector_set_real_single(&A->a, size, stride, a);
}

/**
 * ‘mtxompcsr_set_real_double()’ sets values of a matrix based on
 * an array of double precision floating point numbers.
 */
int mtxompcsr_set_real_double(
    struct mtxompcsr * A,
    int64_t size,
    int stride,
    const double * a)
{
    return mtxompvector_set_real_double(&A->a, size, stride, a);
}

/**
 * ‘mtxompcsr_set_complex_single()’ sets values of a matrix based
 * on an array of single precision floating point complex numbers.
 */
int mtxompcsr_set_complex_single(
    struct mtxompcsr * A,
    int64_t size,
    int stride,
    const float (*a)[2])
{
    return mtxompvector_set_complex_single(&A->a, size, stride, a);
}

/**
 * ‘mtxompcsr_set_complex_double()’ sets values of a matrix based
 * on an array of double precision floating point complex numbers.
 */
int mtxompcsr_set_complex_double(
    struct mtxompcsr * A,
    int64_t size,
    int stride,
    const double (*a)[2])
{
    return mtxompvector_set_complex_double(&A->a, size, stride, a);
}

/**
 * ‘mtxompcsr_set_integer_single()’ sets values of a matrix based
 * on an array of integers.
 */
int mtxompcsr_set_integer_single(
    struct mtxompcsr * A,
    int64_t size,
    int stride,
    const int32_t * a)
{
    return mtxompvector_set_integer_single(&A->a, size, stride, a);
}

/**
 * ‘mtxompcsr_set_integer_double()’ sets values of a matrix based
 * on an array of integers.
 */
int mtxompcsr_set_integer_double(
    struct mtxompcsr * A,
    int64_t size,
    int stride,
    const int64_t * a)
{
    return mtxompvector_set_integer_double(&A->a, size, stride, a);
}

/*
 * row and column vectors
 */

/**
 * ‘mtxompcsr_alloc_row_vector()’ allocates a row vector for a
 * given matrix, where a row vector is a vector whose length equal to
 * a single row of the matrix.
 */
int mtxompcsr_alloc_row_vector(
    const struct mtxompcsr * A,
    struct mtxvector * x,
    enum mtxvectortype vectortype)
{
    return mtxvector_alloc(
        x, vectortype, A->a.base.field, A->a.base.precision, A->num_columns);
}

/**
 * ‘mtxompcsr_alloc_column_vector()’ allocates a column vector for
 * a given matrix, where a column vector is a vector whose length
 * equal to a single column of the matrix.
 */
int mtxompcsr_alloc_column_vector(
    const struct mtxompcsr * A,
    struct mtxvector * y,
    enum mtxvectortype vectortype)
{
    return mtxvector_alloc(
        y, vectortype, A->a.base.field, A->a.base.precision, A->num_rows);
}

/*
 * convert to and from Matrix Market format
 */

/**
 * ‘mtxompcsr_from_mtxfile()’ converts a matrix from Matrix Market
 * format.
 */
int mtxompcsr_from_mtxfile(
    struct mtxompcsr * A,
    const struct mtxfile * mtxfile)
{
    struct mtxbasecoo coo;
    int err = mtxbasecoo_from_mtxfile(&coo, mtxfile);
    if (err) return err;
    if (mtxfile->header.field == mtxfile_real) {
        if (mtxfile->precision == mtx_single) {
            err = mtxompcsr_init_entries_real_single(
                A, coo.symmetry, coo.num_rows, coo.num_columns,
                coo.size, coo.rowidx, coo.colidx, coo.a.data.real_single);
        } else if (mtxfile->precision == mtx_double) {
            err = mtxompcsr_init_entries_real_double(
                A, coo.symmetry, coo.num_rows, coo.num_columns,
                coo.size, coo.rowidx, coo.colidx, coo.a.data.real_double);
        } else { mtxbasecoo_free(&coo); return MTX_ERR_INVALID_PRECISION; }
    } else if (mtxfile->header.field == mtxfile_complex) {
        if (mtxfile->precision == mtx_single) {
            err = mtxompcsr_init_entries_complex_single(
                A, coo.symmetry, coo.num_rows, coo.num_columns,
                coo.size, coo.rowidx, coo.colidx, coo.a.data.complex_single);
        } else if (mtxfile->precision == mtx_double) {
            err = mtxompcsr_init_entries_complex_double(
                A, coo.symmetry, coo.num_rows, coo.num_columns,
                coo.size, coo.rowidx, coo.colidx, coo.a.data.complex_double);
        } else { mtxbasecoo_free(&coo); return MTX_ERR_INVALID_PRECISION; }
    } else if (mtxfile->header.field == mtxfile_integer) {
        if (mtxfile->precision == mtx_single) {
            err = mtxompcsr_init_entries_integer_single(
                A, coo.symmetry, coo.num_rows, coo.num_columns,
                coo.size, coo.rowidx, coo.colidx, coo.a.data.integer_single);
        } else if (mtxfile->precision == mtx_double) {
            err = mtxompcsr_init_entries_integer_double(
                A, coo.symmetry, coo.num_rows, coo.num_columns,
                coo.size, coo.rowidx, coo.colidx, coo.a.data.integer_double);
        } else { mtxbasecoo_free(&coo); return MTX_ERR_INVALID_PRECISION; }
    } else if (mtxfile->header.field == mtxfile_pattern) {
        err = mtxompcsr_init_entries_pattern(
            A, coo.symmetry, coo.num_rows, coo.num_columns,
            coo.size, coo.rowidx, coo.colidx);
    } else { mtxbasecoo_free(&coo); return MTX_ERR_INVALID_MTX_FIELD; }
    if (err) { mtxbasecoo_free(&coo); return err; }
    mtxbasecoo_free(&coo);
    return MTX_SUCCESS;
}

/**
 * ‘mtxompcsr_to_mtxfile()’ converts a matrix to Matrix Market
 * format.
 */
int mtxompcsr_to_mtxfile(
    struct mtxfile * mtxfile,
    const struct mtxompcsr * A,
    int64_t num_rows,
    const int64_t * rowidx,
    int64_t num_columns,
    const int64_t * colidx,
    enum mtxfileformat mtxfmt)
{
    int err;
    if (mtxfmt != mtxfile_coordinate)
        return MTX_ERR_INCOMPATIBLE_MTX_FORMAT;

    enum mtxfield field = mtxompcsr_field(A);
    enum mtxprecision precision = mtxompcsr_precision(A);
    enum mtxfilesymmetry symmetry;
    err = mtxfilesymmetry_from_mtxsymmetry(&symmetry, A->symmetry);
    if (err) return err;

    if (field == mtx_field_real) {
        err = mtxfile_alloc_matrix_coordinate(
            mtxfile, mtxfile_real, symmetry, precision,
            rowidx ? num_rows : A->num_rows,
            colidx ? num_columns : A->num_columns,
            A->size);
        if (err) return err;
        if (precision == mtx_single) {
            struct mtxfile_matrix_coordinate_real_single * data =
                mtxfile->data.matrix_coordinate_real_single;
            for (int i = 0; i < A->num_rows; i++) {
                for (int64_t k = A->rowptr[i]; k < A->rowptr[i+1]; k++) {
                    data[k].i = rowidx ? rowidx[i]+1 : i+1;
                    data[k].j = colidx ? colidx[A->colidx[k]]+1 : A->colidx[k]+1;
                    data[k].a = A->a.base.data.real_single[k];
                }
            }
        } else if (precision == mtx_double) {
            struct mtxfile_matrix_coordinate_real_double * data =
                mtxfile->data.matrix_coordinate_real_double;
            for (int i = 0; i < A->num_rows; i++) {
                for (int64_t k = A->rowptr[i]; k < A->rowptr[i+1]; k++) {
                    data[k].i = rowidx ? rowidx[i]+1 : i+1;
                    data[k].j = colidx ? colidx[A->colidx[k]]+1 : A->colidx[k]+1;
                    data[k].a = A->a.base.data.real_double[k];
                }
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (field == mtx_field_complex) {
        err = mtxfile_alloc_matrix_coordinate(
            mtxfile, mtxfile_complex, symmetry, precision,
            A->num_rows, A->num_columns, A->size);
        if (err) return err;
        if (precision == mtx_single) {
            struct mtxfile_matrix_coordinate_complex_single * data =
                mtxfile->data.matrix_coordinate_complex_single;
            for (int i = 0; i < A->num_rows; i++) {
                for (int64_t k = A->rowptr[i]; k < A->rowptr[i+1]; k++) {
                    data[k].i = rowidx ? rowidx[i]+1 : i+1;
                    data[k].j = colidx ? colidx[A->colidx[k]]+1 : A->colidx[k]+1;
                    data[k].a[0] = A->a.base.data.complex_single[k][0];
                    data[k].a[1] = A->a.base.data.complex_single[k][1];
                }
            }
        } else if (precision == mtx_double) {
            struct mtxfile_matrix_coordinate_complex_double * data =
                mtxfile->data.matrix_coordinate_complex_double;
            for (int i = 0; i < A->num_rows; i++) {
                for (int64_t k = A->rowptr[i]; k < A->rowptr[i+1]; k++) {
                    data[k].i = rowidx ? rowidx[i]+1 : i+1;
                    data[k].j = colidx ? colidx[A->colidx[k]]+1 : A->colidx[k]+1;
                    data[k].a[0] = A->a.base.data.complex_double[k][0];
                    data[k].a[1] = A->a.base.data.complex_double[k][1];
                }
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (field == mtx_field_integer) {
        err = mtxfile_alloc_matrix_coordinate(
            mtxfile, mtxfile_integer, symmetry, precision,
            A->num_rows, A->num_columns, A->size);
        if (err) return err;
        if (precision == mtx_single) {
            struct mtxfile_matrix_coordinate_integer_single * data =
                mtxfile->data.matrix_coordinate_integer_single;
            for (int i = 0; i < A->num_rows; i++) {
                for (int64_t k = A->rowptr[i]; k < A->rowptr[i+1]; k++) {
                    data[k].i = rowidx ? rowidx[i]+1 : i+1;
                    data[k].j = colidx ? colidx[A->colidx[k]]+1 : A->colidx[k]+1;
                    data[k].a = A->a.base.data.integer_single[k];
                }
            }
        } else if (precision == mtx_double) {
            struct mtxfile_matrix_coordinate_integer_double * data =
                mtxfile->data.matrix_coordinate_integer_double;
            for (int i = 0; i < A->num_rows; i++) {
                for (int64_t k = A->rowptr[i]; k < A->rowptr[i+1]; k++) {
                    data[k].i = rowidx ? rowidx[i]+1 : i+1;
                    data[k].j = colidx ? colidx[A->colidx[k]]+1 : A->colidx[k]+1;
                    data[k].a = A->a.base.data.integer_double[k];
                }
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (field == mtx_field_pattern) {
        err = mtxfile_alloc_matrix_coordinate(
            mtxfile, mtxfile_pattern, symmetry, mtx_single,
            A->num_rows, A->num_columns, A->size);
        if (err) return err;
        struct mtxfile_matrix_coordinate_pattern * data =
            mtxfile->data.matrix_coordinate_pattern;
        for (int i = 0; i < A->num_rows; i++) {
            for (int64_t k = A->rowptr[i]; k < A->rowptr[i+1]; k++) {
                data[k].i = rowidx ? rowidx[i]+1 : i+1;
                data[k].j = colidx ? colidx[A->colidx[k]]+1 : A->colidx[k]+1;
            }
        }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/*
 * partitioning
 */

/**
 * ‘mtxompcsr_partition_rowwise()’ partitions the entries of a matrix
 * rowwise.
 *
 * See ‘partition_int()’ for an explanation of the meaning of the
 * arguments ‘parttype’, ‘num_parts’, ‘partsizes’, ‘blksize’ and
 * ‘parts’.
 *
 * The length of the array ‘dstnzpart’ must be at least equal to the
 * number of (nonzero) matrix entries (which can be obtained by
 * calling ‘mtxmatrix_size()’). If successful, ‘dstnzpart’ is used to
 * store the part numbers assigned to the matrix nonzeros.
 *
 * If ‘dstrowpart’ is not ‘NULL’, then it must be an array of length
 * at least equal to the number of matrix rows, which is used to store
 * the part numbers assigned to the rows.
 *
 * If ‘dstnzpartsizes’ is not ‘NULL’, then it must be an array of
 * length ‘num_parts’, which is used to store the number of nonzeros
 * assigned to each part. Similarly, if ‘dstrowpartsizes’ is not
 * ‘NULL’, then it must be an array of length ‘num_parts’, and it is
 * used to store the number of rows assigned to each part
 */
int mtxompcsr_partition_rowwise(
    const struct mtxompcsr * A,
    enum mtxpartitioning parttype,
    int num_parts,
    const int * partsizes,
    int blksize,
    const int * parts,
    int * dstnzpart,
    int64_t * dstnzpartsizes,
    int * dstrowpart,
    int64_t * dstrowpartsizes)
{
    if (dstrowpart) {
        for (int i = 0; i < A->num_rows; i++) dstrowpart[i] = i;
        int err = partition_int(
            parttype, A->num_rows, num_parts, partsizes, blksize, parts,
            A->num_rows, sizeof(*dstrowpart), dstrowpart,
            dstrowpart, dstrowpartsizes);
        if (err) return err;
    }
    int * rowidx = malloc(A->size * sizeof(int));
    if (!rowidx) return MTX_ERR_ERRNO;
    for (int i = 0; i < A->num_rows; i++) {
        for (int64_t k = A->rowptr[i]; k < A->rowptr[i+1]; k++)
            rowidx[k] = i;
    }
    int err = partition_int(
        parttype, A->num_columns, num_parts, partsizes, blksize, parts,
        A->size, sizeof(*rowidx), rowidx, dstnzpart, dstnzpartsizes);
    if (err) { free(rowidx); return err; }
    free(rowidx);
    return MTX_SUCCESS;
}

/**
 * ‘mtxompcsr_partition_columnwise()’ partitions the entries of a
 * matrix columnwise.
 *
 * See ‘partition_int()’ for an explanation of the meaning of the
 * arguments ‘parttype’, ‘num_parts’, ‘partsizes’, ‘blksize’ and
 * ‘parts’.
 *
 * The length of the array ‘dstnzpart’ must be at least equal to the
 * number of (nonzero) matrix entries (which can be obtained by
 * calling ‘mtxmatrix_size()’). If successful, ‘dstnzpart’ is used to
 * store the part numbers assigned to the matrix nonzeros.
 *
 * If ‘dstcolpart’ is not ‘NULL’, then it must be an array of length
 * at least equal to the number of matrix columns, which is used to
 * store the part numbers assigned to the columns.
 *
 * If ‘dstnzpartsizes’ is not ‘NULL’, then it must be an array of
 * length ‘num_parts’, which is used to store the number of nonzeros
 * assigned to each part. Similarly, if ‘dstcolpartsizes’ is not
 * ‘NULL’, then it must be an array of length ‘num_parts’, and it is
 * used to store the number of columns assigned to each part
 */
int mtxompcsr_partition_columnwise(
    const struct mtxompcsr * A,
    enum mtxpartitioning parttype,
    int num_parts,
    const int * partsizes,
    int blksize,
    const int * parts,
    int * dstnzpart,
    int64_t * dstnzpartsizes,
    int * dstcolpart,
    int64_t * dstcolpartsizes)
{
    if (dstcolpart) {
        for (int j = 0; j < A->num_columns; j++) dstcolpart[j] = j;
        int err = partition_int(
            parttype, A->num_columns, num_parts, partsizes, blksize, parts,
            A->num_columns, sizeof(*dstcolpart), dstcolpart,
            dstcolpart, dstcolpartsizes);
        if (err) return err;
    }
    return partition_int(
        parttype, A->num_columns, num_parts, partsizes, blksize, parts,
        A->size, sizeof(*A->colidx), A->colidx, dstnzpart, dstnzpartsizes);
}

/**
 * ‘mtxompcsr_partition_2d()’ partitions the entries of a matrix in a
 * 2D manner.
 *
 * See ‘partition_int()’ for an explanation of the meaning of the
 * arguments ‘rowparttype’, ‘num_row_parts’, ‘rowpartsizes’,
 * ‘rowblksize’, ‘rowparts’, and so on.
 *
 * The length of the array ‘dstnzpart’ must be at least equal to the
 * number of (nonzero) matrix entries (which can be obtained by
 * calling ‘mtxmatrix_size()’). If successful, ‘dstnzpart’ is used to
 * store the part numbers assigned to the matrix nonzeros.
 *
 * The length of the array ‘dstrowpart’ must be at least equal to the
 * number of matrix rows, and it is used to store the part numbers
 * assigned to the rows. Similarly, the length of ‘dstcolpart’ must be
 * at least equal to the number of matrix columns, and it is used to
 * store the part numbers assigned to the columns.
 *
 * If ‘dstrowpartsizes’ or ‘dstcolpartsizes’ are not ‘NULL’, then they
 * must point to arrays of length ‘num_row_parts’ and ‘num_col_parts’,
 * respectively, which are used to store the number of rows and
 * columns assigned to each part.
 */
int mtxompcsr_partition_2d(
    const struct mtxompcsr * A,
    enum mtxpartitioning rowparttype,
    int num_row_parts,
    const int * rowpartsizes,
    int rowblksize,
    const int * rowparts,
    enum mtxpartitioning colparttype,
    int num_col_parts,
    const int * colpartsizes,
    int colblksize,
    const int * colparts,
    int * dstnzpart,
    int64_t * dstnzpartsizes,
    int * dstrowpart,
    int64_t * dstrowpartsizes,
    int * dstcolpart,
    int64_t * dstcolpartsizes)
{
    int num_parts = num_row_parts * num_col_parts;
    int * dstnzrowpart = malloc(A->size * sizeof(int));
    if (!dstnzrowpart) return MTX_ERR_ERRNO;
    int err = mtxompcsr_partition_rowwise(
        A, rowparttype, num_row_parts, rowpartsizes, rowblksize, rowparts,
        dstnzrowpart, NULL, dstrowpart, dstrowpartsizes);
    if (err) { free(dstnzrowpart); return err; }
    err = mtxompcsr_partition_columnwise(
        A, colparttype, num_col_parts, colpartsizes, colblksize, colparts,
        dstnzpart, NULL, dstcolpart, dstcolpartsizes);
    if (err) { free(dstnzrowpart); return err; }
    for (int64_t k = 0; k < A->size; k++)
        dstnzpart[k] = dstnzrowpart[k]*num_col_parts + dstnzpart[k];
    if (dstnzpartsizes) {
        for (int p = 0; p < num_parts; p++) dstnzpartsizes[p] = 0;
        for (int64_t k = 0; k < A->size; k++) dstnzpartsizes[dstnzpart[k]]++;
    }
    free(dstnzrowpart);
    return MTX_SUCCESS;
}

/**
 * ‘mtxompcsr_split()’ splits a matrix into multiple matrices
 * according to a given assignment of parts to each nonzero matrix
 * element.
 *
 * The partitioning of the nonzero matrix elements is specified by the
 * array ‘parts’. The length of the ‘parts’ array is given by ‘size’,
 * which must match the number of explicitly stored nonzero matrix
 * entries in ‘src’. Each entry in the ‘parts’ array is an integer in
 * the range ‘[0, num_parts)’ designating the part to which the
 * corresponding matrix nonzero belongs.
 *
 * The argument ‘dsts’ is an array of ‘num_parts’ pointers to objects
 * of type ‘struct mtxompcsr’. If successful, then ‘dsts[p]’
 * points to a matrix consisting of elements from ‘src’ that belong to
 * the ‘p’th part, as designated by the ‘parts’ array.
 *
 * The caller is responsible for calling ‘mtxompcsr_free()’ to
 * free storage allocated for each matrix in the ‘dsts’ array.
 */
int mtxompcsr_split(
    int num_parts,
    struct mtxompcsr ** dsts,
    const struct mtxompcsr * src,
    int64_t size,
    int * parts)
{
    if (size != src->size) return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    bool sorted = true;
    for (int64_t k = 0; k < size; k++) {
        if (parts[k] < 0 || parts[k] >= num_parts)
            return MTX_ERR_INDEX_OUT_OF_BOUNDS;
        if (k > 0 && parts[k-1] > parts[k]) sorted = false;
    }

    /* sort by part number and invert the sorting permutation */
    int64_t * invperm;
    if (sorted) {
        invperm = malloc(size * sizeof(int64_t));
        if (!invperm) return MTX_ERR_ERRNO;
        for (int64_t k = 0; k < size; k++) invperm[k] = k;
    } else {
        int64_t * perm = malloc(size * sizeof(int64_t));
        if (!perm) return MTX_ERR_ERRNO;
        int err = radix_sort_int(size, parts, perm);
        if (err) { free(perm); return err; }
        invperm = malloc(size * sizeof(int64_t));
        if (!invperm) { free(perm); return MTX_ERR_ERRNO; }
        for (int64_t k = 0; k < size; k++) invperm[perm[k]] = k;
        free(perm);
    }

    int * rowidx = malloc(size * sizeof(int));
    if (!rowidx) { free(invperm); return MTX_ERR_ERRNO; }
    for (int i = 0; i < src->num_rows; i++) {
        for (int64_t k = src->rowptr[i]; k < src->rowptr[i+1]; k++) {
            rowidx[k] = i;
        }
    }

    /*
     * Extract each part by a) counting the number elements in the
     * part, b) allocating storage, and c) gathering row and column
     * offsets, as well as nonzeros, for the part.
     */
    int64_t offset = 0;
    for (int p = 0; p < num_parts; p++) {
        int64_t partsize = 0;
        while (offset+partsize < size && parts[offset+partsize] == p) partsize++;

        dsts[p]->symmetry = src->symmetry;
        dsts[p]->num_rows = src->num_rows;
        dsts[p]->num_columns = src->num_columns;
        dsts[p]->num_entries = src->num_entries;
        dsts[p]->num_nonzeros = 0;
        dsts[p]->size = partsize;
        dsts[p]->rowptr = malloc((dsts[p]->num_rows+1) * sizeof(int64_t));
        if (!dsts[p]->rowptr) {
            for (int q = p-1; q >= 0; q--) mtxompcsr_free(dsts[q]);
            free(rowidx); free(invperm);
            return MTX_ERR_ERRNO;
        }
        dsts[p]->colidx = malloc(partsize * sizeof(int));
        if (!dsts[p]->colidx) {
            free(dsts[p]->rowptr);
            for (int q = p-1; q >= 0; q--) mtxompcsr_free(dsts[q]);
            free(rowidx); free(invperm);
            return MTX_ERR_ERRNO;
        }
        for (int i = 0; i <= dsts[p]->num_rows; i++) dsts[p]->rowptr[i] = 0;
        for (int64_t k = 0; k < partsize; k++) {
            int i = rowidx[invperm[offset+k]];
            dsts[p]->rowptr[i+1]++;
            dsts[p]->colidx[k] = src->colidx[invperm[offset+k]];
            dsts[p]->num_nonzeros +=
                (dsts[p]->symmetry == mtx_unsymmetric ||
                 dsts[p]->rowptr[k] == dsts[p]->colidx[k]) ? 1 : 2;
        }
        for (int i = 0; i < dsts[p]->num_rows; i++)
            dsts[p]->rowptr[i+1] += dsts[p]->rowptr[i];

        int err = mtxompvector_alloc_packed(
            &dsts[p]->a, src->a.base.field, src->a.base.precision, size, partsize, &invperm[offset]);
        if (err) {
            free(dsts[p]->colidx); free(dsts[p]->rowptr);
            for (int q = p-1; q >= 0; q--) mtxompcsr_free(dsts[q]);
            free(rowidx); free(invperm);
            return err;
        }
        err = mtxompvector_usga(&dsts[p]->a, &src->a);
        if (err) {
            mtxompvector_free(&dsts[p]->a);
            free(dsts[p]->colidx); free(dsts[p]->rowptr);
            for (int q = p-1; q >= 0; q--) mtxompcsr_free(dsts[q]);
            free(rowidx); free(invperm);
            return err;
        }

        /* extract diagonals for symmetric and Hermitian matrices */
        if (dsts[p]->num_rows == dsts[p]->num_columns &&
            (dsts[p]->symmetry == mtx_symmetric ||
             dsts[p]->symmetry == mtx_skew_symmetric ||
             dsts[p]->symmetry == mtx_hermitian))
        {
            int64_t num_diagonals = 0;
            for (int i = 0; i < dsts[p]->num_rows; i++) {
                for (int64_t k = dsts[p]->rowptr[i]; k < dsts[p]->rowptr[i+1]; k++) {
                    if (i == dsts[p]->colidx[k]) num_diagonals++;
                }
            }
            dsts[p]->num_nonzeros = 2*dsts[p]->size-num_diagonals;
            err = mtxompvector_alloc_packed(
                &dsts[p]->diag, dsts[p]->a.base.field, dsts[p]->a.base.precision,
                dsts[p]->size, num_diagonals, NULL);
            if (err) {
                mtxompvector_free(&dsts[p]->a);
                free(dsts[p]->colidx); free(dsts[p]->rowptr);
                for (int q = p-1; q >= 0; q--) mtxompcsr_free(dsts[q]);
                free(rowidx); free(invperm);
                return err;
            }
            int64_t * diagidx = dsts[p]->diag.base.idx;
            num_diagonals = 0;
            for (int i = 0; i < dsts[p]->num_rows; i++) {
                for (int64_t k = dsts[p]->rowptr[i]; k < dsts[p]->rowptr[i+1]; k++) {
                    if (i == dsts[p]->colidx[k]) diagidx[num_diagonals++] = k;
                }
            }
        } else if (dsts[p]->symmetry == mtx_unsymmetric) {
            dsts[p]->num_nonzeros = dsts[p]->size;
            err = mtxompvector_alloc_packed(
                &dsts[p]->diag, dsts[p]->a.base.field, dsts[p]->a.base.precision, dsts[p]->size, 0, NULL);
            if (err) {
                mtxompvector_free(&dsts[p]->a);
                free(dsts[p]->colidx); free(dsts[p]->rowptr);
                for (int q = p-1; q >= 0; q--) mtxompcsr_free(dsts[q]);
                free(rowidx); free(invperm);
                return err;
            }
        } else {
            mtxompvector_free(&dsts[p]->a);
            free(dsts[p]->colidx); free(dsts[p]->rowptr);
            for (int q = p-1; q >= 0; q--) mtxompcsr_free(dsts[q]);
            free(rowidx); free(invperm);
            return MTX_ERR_INVALID_SYMMETRY;
        }

        offset += partsize;
    }
    free(rowidx); free(invperm);
    return MTX_SUCCESS;
}

/*
 * Level 1 BLAS operations
 */

/**
 * ‘mtxompcsr_swap()’ swaps values of two matrices, simultaneously
 * performing ‘y <- x’ and ‘x <- y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxompcsr_swap(
    struct mtxompcsr * x,
    struct mtxompcsr * y)
{
    return mtxompvector_swap(&x->a, &y->a);
}

/**
 * ‘mtxompcsr_copy()’ copies values of a matrix, ‘y = x’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxompcsr_copy(
    struct mtxompcsr * y,
    const struct mtxompcsr * x)
{
    return mtxompvector_copy(&y->a, &x->a);
}

/**
 * ‘mtxompcsr_sscal()’ scales a matrix by a single precision
 * floating point scalar, ‘x = a*x’.
 */
int mtxompcsr_sscal(
    float a,
    struct mtxompcsr * x,
    int64_t * num_flops)
{
    return mtxompvector_sscal(a, &x->a, num_flops);
}

/**
 * ‘mtxompcsr_dscal()’ scales a matrix by a double precision
 * floating point scalar, ‘x = a*x’.
 */
int mtxompcsr_dscal(
    double a,
    struct mtxompcsr * x,
    int64_t * num_flops)
{
    return mtxompvector_dscal(a, &x->a, num_flops);
}

/**
 * ‘mtxompcsr_cscal()’ scales a matrix by a complex, single
 * precision floating point scalar, ‘x = (a+b*i)*x’.
 */
int mtxompcsr_cscal(
    float a[2],
    struct mtxompcsr * x,
    int64_t * num_flops)
{
    return mtxompvector_cscal(a, &x->a, num_flops);
}

/**
 * ‘mtxompcsr_zscal()’ scales a matrix by a complex, double
 * precision floating point scalar, ‘x = (a+b*i)*x’.
 */
int mtxompcsr_zscal(
    double a[2],
    struct mtxompcsr * x,
    int64_t * num_flops)
{
    return mtxompvector_zscal(a, &x->a, num_flops);
}

/**
 * ‘mtxompcsr_saxpy()’ adds a matrix to another one multiplied by
 * a single precision floating point value, ‘y = a*x + y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxompcsr_saxpy(
    float a,
    const struct mtxompcsr * x,
    struct mtxompcsr * y,
    int64_t * num_flops)
{
    return mtxompvector_saxpy(a, &x->a, &y->a, num_flops);
}

/**
 * ‘mtxompcsr_daxpy()’ adds a matrix to another one multiplied by
 * a double precision floating point value, ‘y = a*x + y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxompcsr_daxpy(
    double a,
    const struct mtxompcsr * x,
    struct mtxompcsr * y,
    int64_t * num_flops)
{
    return mtxompvector_daxpy(a, &x->a, &y->a, num_flops);
}

/**
 * ‘mtxompcsr_saypx()’ multiplies a matrix by a single precision
 * floating point scalar and adds another matrix, ‘y = a*y + x’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxompcsr_saypx(
    float a,
    struct mtxompcsr * y,
    const struct mtxompcsr * x,
    int64_t * num_flops)
{
    return mtxompvector_saypx(a, &y->a, &x->a, num_flops);
}

/**
 * ‘mtxompcsr_daypx()’ multiplies a matrix by a double precision
 * floating point scalar and adds another matrix, ‘y = a*y + x’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxompcsr_daypx(
    double a,
    struct mtxompcsr * y,
    const struct mtxompcsr * x,
    int64_t * num_flops)
{
    return mtxompvector_daypx(a, &y->a, &x->a, num_flops);
}

/**
 * ‘mtxompcsr_sdot()’ computes the Frobenius inner product of two
 * matrices in single precision floating point.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxompcsr_sdot(
    const struct mtxompcsr * x,
    const struct mtxompcsr * y,
    float * dot,
    int64_t * num_flops)
{
    if (x->symmetry == mtx_unsymmetric && y->symmetry == mtx_unsymmetric) {
        return mtxompvector_sdot(&x->a, &y->a, dot, num_flops);
    } else { return MTX_ERR_INVALID_SYMMETRY; }
}

/**
 * ‘mtxompcsr_ddot()’ computes the Frobenius inner product of two
 * matrices in double precision floating point.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxompcsr_ddot(
    const struct mtxompcsr * x,
    const struct mtxompcsr * y,
    double * dot,
    int64_t * num_flops)
{
    if (x->symmetry == mtx_unsymmetric && y->symmetry == mtx_unsymmetric) {
        return mtxompvector_ddot(&x->a, &y->a, dot, num_flops);
    } else { return MTX_ERR_INVALID_SYMMETRY; }
}

/**
 * ‘mtxompcsr_cdotu()’ computes the product of the transpose of a
 * complex row matrix with another complex row matrix in single
 * precision floating point, ‘dot := x^T*y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxompcsr_cdotu(
    const struct mtxompcsr * x,
    const struct mtxompcsr * y,
    float (* dot)[2],
    int64_t * num_flops)
{
    if (x->symmetry == mtx_unsymmetric && y->symmetry == mtx_unsymmetric) {
        return mtxompvector_cdotu(&x->a, &y->a, dot, num_flops);
    } else { return MTX_ERR_INVALID_SYMMETRY; }
}

/**
 * ‘mtxompcsr_zdotu()’ computes the product of the transpose of a
 * complex row matrix with another complex row matrix in double
 * precision floating point, ‘dot := x^T*y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxompcsr_zdotu(
    const struct mtxompcsr * x,
    const struct mtxompcsr * y,
    double (* dot)[2],
    int64_t * num_flops)
{
    if (x->symmetry == mtx_unsymmetric && y->symmetry == mtx_unsymmetric) {
        return mtxompvector_zdotu(&x->a, &y->a, dot, num_flops);
    } else { return MTX_ERR_INVALID_SYMMETRY; }
}

/**
 * ‘mtxompcsr_cdotc()’ computes the Frobenius inner product of two
 * complex matrices in single precision floating point, ‘dot :=
 * x^H*y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxompcsr_cdotc(
    const struct mtxompcsr * x,
    const struct mtxompcsr * y,
    float (* dot)[2],
    int64_t * num_flops)
{
    if (x->symmetry == mtx_unsymmetric && y->symmetry == mtx_unsymmetric) {
        return mtxompvector_cdotc(&x->a, &y->a, dot, num_flops);
    } else { return MTX_ERR_INVALID_SYMMETRY; }
}

/**
 * ‘mtxompcsr_zdotc()’ computes the Frobenius inner product of two
 * complex matrices in double precision floating point, ‘dot :=
 * x^H*y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxompcsr_zdotc(
    const struct mtxompcsr * x,
    const struct mtxompcsr * y,
    double (* dot)[2],
    int64_t * num_flops)
{
    if (x->symmetry == mtx_unsymmetric && y->symmetry == mtx_unsymmetric) {
        return mtxompvector_zdotc(&x->a, &y->a, dot, num_flops);
    } else { return MTX_ERR_INVALID_SYMMETRY; }
}

/**
 * ‘mtxompcsr_snrm2()’ computes the Frobenius norm of a matrix in
 * single precision floating point.
 */
int mtxompcsr_snrm2(
    const struct mtxompcsr * x,
    float * nrm2,
    int64_t * num_flops)
{
    if (x->symmetry == mtx_unsymmetric) {
        return mtxompvector_snrm2(&x->a, nrm2, num_flops);
    } else { return MTX_ERR_INVALID_SYMMETRY; }
}

/**
 * ‘mtxompcsr_dnrm2()’ computes the Frobenius norm of a matrix in
 * double precision floating point.
 */
int mtxompcsr_dnrm2(
    const struct mtxompcsr * x,
    double * nrm2,
    int64_t * num_flops)
{
    if (x->symmetry == mtx_unsymmetric) {
        return mtxompvector_dnrm2(&x->a, nrm2, num_flops);
    } else { return MTX_ERR_INVALID_SYMMETRY; }
}

/**
 * ‘mtxompcsr_sasum()’ computes the sum of absolute values
 * (1-norm) of a matrix in single precision floating point.  If the
 * matrix is complex-valued, then the sum of the absolute values of
 * the real and imaginary parts is computed.
 */
int mtxompcsr_sasum(
    const struct mtxompcsr * x,
    float * asum,
    int64_t * num_flops)
{
    if (x->symmetry == mtx_unsymmetric) {
        return mtxompvector_sasum(&x->a, asum, num_flops);
    } else { return MTX_ERR_INVALID_SYMMETRY; }
}

/**
 * ‘mtxompcsr_dasum()’ computes the sum of absolute values
 * (1-norm) of a matrix in double precision floating point.  If the
 * matrix is complex-valued, then the sum of the absolute values of
 * the real and imaginary parts is computed.
 */
int mtxompcsr_dasum(
    const struct mtxompcsr * x,
    double * asum,
    int64_t * num_flops)
{
    if (x->symmetry == mtx_unsymmetric) {
        return mtxompvector_dasum(&x->a, asum, num_flops);
    } else { return MTX_ERR_INVALID_SYMMETRY; }
}

/**
 * ‘mtxompcsr_iamax()’ finds the index of the first element having
 * the maximum absolute value.  If the matrix is complex-valued, then
 * the index points to the first element having the maximum sum of the
 * absolute values of the real and imaginary parts.
 */
int mtxompcsr_iamax(
    const struct mtxompcsr * x,
    int * iamax)
{
    return mtxompvector_iamax(&x->a, iamax);
}

/*
 * Level 2 BLAS operations (matrix-vector)
 */

/**
 * ‘mtxompcsr_sgemv()’ multiplies a matrix ‘A’ or its transpose
 * ‘A'’ by a real scalar ‘alpha’ (‘α’) and a vector ‘x’, before adding
 * the result to another vector ‘y’ multiplied by another real scalar
 * ‘beta’ (‘β’). That is, ‘y = α*A*x + β*y’ or ‘y = α*A'*x + β*y’.
 *
 * The scalars ‘alpha’ and ‘beta’ are given as single precision
 * floating point numbers.
 *
 * The vectors ‘x’ and ‘y’ must be of ‘array’ type, and they must both
 * have the same field and precision.  If the field of the matrix ‘A’
 * is ‘real’, ‘integer’ or ‘complex’, then the vectors must have the
 * same field.  Otherwise, if the matrix field is ‘pattern’, then ‘x’
 * and ‘y’ are allowed to be ‘real’, ‘integer’ or ‘complex’, but they
 * must both have the same field.
 *
 * Moreover, if ‘trans’ is ‘mtx_notrans’, then the size of ‘x’ must
 * equal the number of columns of ‘A’ and the size of ‘y’ must equal
 * the number of rows of ‘A’. if ‘trans’ is ‘mtx_trans’ or
 * ‘mtx_conjtrans’, then the size of ‘x’ must equal the number of rows
 * of ‘A’ and the size of ‘y’ must equal the number of columns of ‘A’.
 *
 * For now, the only case that is parallelised with OpenMP is
 * multiplication with non-transposed and unsymmetric matrices.
 */
int mtxompcsr_sgemv(
    enum mtxtransposition trans,
    float alpha,
    const struct mtxompcsr * A,
    const struct mtxvector * x,
    float beta,
    struct mtxvector * y,
    int64_t * num_flops)
{
    int err;
    enum mtxfield field = mtxompcsr_field(A);
    enum mtxprecision precision = mtxompcsr_precision(A);
    const struct mtxompvector * a = &A->a;
    if (x->type != mtxompvector || y->type != mtxompvector)
        return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
    const struct mtxompvector * xomp = &x->storage.omp;
    const struct mtxbasevector * xbase = &xomp->base;
    struct mtxompvector * yomp = &y->storage.omp;
    struct mtxbasevector * ybase = &yomp->base;
    if (xbase->field != field || ybase->field != field)
        return MTX_ERR_INCOMPATIBLE_FIELD;
    if (xbase->precision != precision || ybase->precision != precision)
        return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (trans == mtx_notrans) {
        if (A->num_rows != ybase->size || A->num_columns != xbase->size)
            return MTX_ERR_INCOMPATIBLE_SIZE;
    } else if (trans == mtx_trans || trans == mtx_conjtrans) {
        if (A->num_columns != ybase->size || A->num_rows != xbase->size)
            return MTX_ERR_INCOMPATIBLE_SIZE;
    }

    if (beta != 1) {
        err = mtxvector_sscal(beta, y, num_flops);
        if (err) return err;
    }

    const int64_t * rowptr = A->rowptr;
    const int * j = A->colidx;
    if (A->symmetry == mtx_unsymmetric) {
        if (field == mtx_field_real) {
            if (trans == mtx_notrans) {
                if (precision == mtx_single) {
                    const float * Adata = a->base.data.real_single;
                    const float * xdata = xbase->data.real_single;
                    float * ydata = ybase->data.real_single;
                    #pragma omp parallel for
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++)
                            ydata[i] += alpha*Adata[k]*xdata[j[k]];
                    }
                    if (num_flops) *num_flops += 3*A->num_nonzeros;
                } else if (precision == mtx_double) {
                    const double * Adata = a->base.data.real_double;
                    const double * xdata = xbase->data.real_double;
                    double * ydata = ybase->data.real_double;
                    #pragma omp parallel for
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++)
                            ydata[i] += alpha*Adata[k]*xdata[j[k]];
                    }
                    if (num_flops) *num_flops += 3*A->num_nonzeros;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (trans == mtx_trans || trans == mtx_conjtrans) {
                if (precision == mtx_single) {
                    const float * Adata = a->base.data.real_single;
                    const float * xdata = xbase->data.real_single;
                    float * ydata = ybase->data.real_single;
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++)
                            ydata[j[k]] += alpha*Adata[k]*xdata[i];
                    }
                    if (num_flops) *num_flops += 3*A->num_nonzeros;
                } else if (precision == mtx_double) {
                    const double * Adata = a->base.data.real_double;
                    const double * xdata = xbase->data.real_double;
                    double * ydata = ybase->data.real_double;
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++)
                            ydata[j[k]] += alpha*Adata[k]*xdata[i];
                    }
                    if (num_flops) *num_flops += 3*A->num_nonzeros;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else { return MTX_ERR_INVALID_TRANSPOSITION; }
        } else if (field == mtx_field_complex) {
            if (trans == mtx_notrans) {
                if (precision == mtx_single) {
                    const float (* Adata)[2] = a->base.data.complex_single;
                    const float (* xdata)[2] = xbase->data.complex_single;
                    float (* ydata)[2] = ybase->data.complex_single;
                    #pragma omp parallel for
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++) {
                            ydata[i][0] += alpha*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                            ydata[i][1] += alpha*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_nonzeros;
                } else if (precision == mtx_double) {
                    const double (* Adata)[2] = a->base.data.complex_double;
                    const double (* xdata)[2] = xbase->data.complex_double;
                    double (* ydata)[2] = ybase->data.complex_double;
                    #pragma omp parallel for
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++) {
                            ydata[i][0] += alpha*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                            ydata[i][1] += alpha*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_nonzeros;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (trans == mtx_trans) {
                if (precision == mtx_single) {
                    const float (* Adata)[2] = a->base.data.complex_single;
                    const float (* xdata)[2] = xbase->data.complex_single;
                    float (* ydata)[2] = ybase->data.complex_single;
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++) {
                            ydata[j[k]][0] += alpha*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                            ydata[j[k]][1] += alpha*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_nonzeros;
                } else if (precision == mtx_double) {
                    const double (* Adata)[2] = a->base.data.complex_double;
                    const double (* xdata)[2] = xbase->data.complex_double;
                    double (* ydata)[2] = ybase->data.complex_double;
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++) {
                            ydata[j[k]][0] += alpha*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                            ydata[j[k]][1] += alpha*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_nonzeros;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (trans == mtx_conjtrans) {
                if (precision == mtx_single) {
                    const float (* Adata)[2] = a->base.data.complex_single;
                    const float (* xdata)[2] = xbase->data.complex_single;
                    float (* ydata)[2] = ybase->data.complex_single;
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++) {
                            ydata[j[k]][0] += alpha*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                            ydata[j[k]][1] += alpha*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_nonzeros;
                } else if (precision == mtx_double) {
                    const double (* Adata)[2] = a->base.data.complex_double;
                    const double (* xdata)[2] = xbase->data.complex_double;
                    double (* ydata)[2] = ybase->data.complex_double;
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++) {
                            ydata[j[k]][0] += alpha*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                            ydata[j[k]][1] += alpha*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_nonzeros;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else { return MTX_ERR_INVALID_TRANSPOSITION; }
        } else if (field == mtx_field_integer) {
            if (trans == mtx_notrans) {
                if (precision == mtx_single) {
                    const int32_t * Adata = a->base.data.integer_single;
                    const int32_t * xdata = xbase->data.integer_single;
                    int32_t * ydata = ybase->data.integer_single;
                    #pragma omp parallel for
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++)
                            ydata[i] += alpha*Adata[k]*xdata[j[k]];
                    }
                    if (num_flops) *num_flops += 3*A->num_nonzeros;
                } else if (precision == mtx_double) {
                    const int64_t * Adata = a->base.data.integer_double;
                    const int64_t * xdata = xbase->data.integer_double;
                    int64_t * ydata = ybase->data.integer_double;
                    #pragma omp parallel for
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++)
                            ydata[i] += alpha*Adata[k]*xdata[j[k]];
                    }
                    if (num_flops) *num_flops += 3*A->num_nonzeros;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (trans == mtx_trans || trans == mtx_conjtrans) {
                if (precision == mtx_single) {
                    const int32_t * Adata = a->base.data.integer_single;
                    const int32_t * xdata = xbase->data.integer_single;
                    int32_t * ydata = ybase->data.integer_single;
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++)
                            ydata[j[k]] += alpha*Adata[k]*xdata[i];
                    }
                    if (num_flops) *num_flops += 3*A->num_nonzeros;
                } else if (precision == mtx_double) {
                    const int64_t * Adata = a->base.data.integer_double;
                    const int64_t * xdata = xbase->data.integer_double;
                    int64_t * ydata = ybase->data.integer_double;
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++)
                            ydata[j[k]] += alpha*Adata[k]*xdata[i];
                    }
                    if (num_flops) *num_flops += 3*A->num_nonzeros;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else { return MTX_ERR_INVALID_TRANSPOSITION; }
        } else { return MTX_ERR_INVALID_FIELD; }
    } else if (A->num_rows == A->num_columns && A->symmetry == mtx_symmetric) {
        int err = mtxompvector_usgz(
            (struct mtxompvector *) &A->diag,
            (struct mtxompvector *) &A->a);
        if (err) return err;
        if (field == mtx_field_real) {
            if (precision == mtx_single) {
                const float * Adata = a->base.data.real_single;
                const float * xdata = xbase->data.real_single;
                float * ydata = ybase->data.real_single;
                for (int i = 0; i < A->num_rows; i++) {
                    for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++) {
                        ydata[i] += alpha*Adata[k]*xdata[j[k]];
                        ydata[j[k]] += alpha*Adata[k]*xdata[i];
                    }
                }
                const float * Adiag = A->diag.base.data.real_single;
                for (int k = 0; k < A->diag.base.num_nonzeros; k++) {
                    int i = A->colidx[A->diag.base.idx[k]];
                    ydata[i] += alpha*Adiag[k]*xdata[i];
                }
                if (num_flops) *num_flops += 6*A->size + 3*A->diag.base.num_nonzeros;
            } else if (precision == mtx_double) {
                const double * Adata = a->base.data.real_double;
                const double * xdata = xbase->data.real_double;
                double * ydata = ybase->data.real_double;
                for (int i = 0; i < A->num_rows; i++) {
                    for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++) {
                        ydata[i] += alpha*Adata[k]*xdata[j[k]];
                        ydata[j[k]] += alpha*Adata[k]*xdata[i];
                    }
                }
                const double * Adiag = A->diag.base.data.real_double;
                for (int k = 0; k < A->diag.base.num_nonzeros; k++) {
                    int i = A->colidx[A->diag.base.idx[k]];
                    ydata[i] += alpha*Adiag[k]*xdata[i];
                }
                if (num_flops) *num_flops += 6*A->size + 3*A->diag.base.num_nonzeros;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (field == mtx_field_complex) {
            if (trans == mtx_notrans || trans == mtx_trans) {
                if (precision == mtx_single) {
                    const float (* Adata)[2] = a->base.data.complex_single;
                    const float (* xdata)[2] = xbase->data.complex_single;
                    float (* ydata)[2] = ybase->data.complex_single;
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++) {
                            ydata[i][0] += alpha*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                            ydata[i][1] += alpha*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                            ydata[j[k]][0] += alpha*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                            ydata[j[k]][1] += alpha*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        }
                    }
                    const float (* Adiag)[2] = A->diag.base.data.complex_single;
                    for (int k = 0; k < A->diag.base.num_nonzeros; k++) {
                        int i = A->colidx[A->diag.base.idx[k]];
                        ydata[i][0] += alpha*(Adiag[k][0]*xdata[i][0]-Adiag[k][1]*xdata[i][1]);
                        ydata[i][1] += alpha*(Adiag[k][0]*xdata[i][1]+Adiag[k][1]*xdata[i][0]);
                    }
                    if (num_flops) *num_flops += 20*A->size + 10*A->diag.base.num_nonzeros;
                } else if (precision == mtx_double) {
                    const double (* Adata)[2] = a->base.data.complex_double;
                    const double (* xdata)[2] = xbase->data.complex_double;
                    double (* ydata)[2] = ybase->data.complex_double;
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++) {
                            ydata[i][0] += alpha*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                            ydata[i][1] += alpha*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                            ydata[j[k]][0] += alpha*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                            ydata[j[k]][1] += alpha*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        }
                    }
                    const double (* Adiag)[2] = A->diag.base.data.complex_double;
                    for (int k = 0; k < A->diag.base.num_nonzeros; k++) {
                        int i = A->colidx[A->diag.base.idx[k]];
                        ydata[i][0] += alpha*(Adiag[k][0]*xdata[i][0]-Adiag[k][1]*xdata[i][1]);
                        ydata[i][1] += alpha*(Adiag[k][0]*xdata[i][1]+Adiag[k][1]*xdata[i][0]);
                    }
                    if (num_flops) *num_flops += 20*A->size + 10*A->diag.base.num_nonzeros;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (trans == mtx_conjtrans) {
                if (precision == mtx_single) {
                    const float (* Adata)[2] = a->base.data.complex_single;
                    const float (* xdata)[2] = xbase->data.complex_single;
                    float (* ydata)[2] = ybase->data.complex_single;
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++) {
                            ydata[i][0] += alpha*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]);
                            ydata[i][1] += alpha*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]);
                            ydata[j[k]][0] += alpha*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                            ydata[j[k]][1] += alpha*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        }
                    }
                    const float (* Adiag)[2] = A->diag.base.data.complex_single;
                    for (int k = 0; k < A->diag.base.num_nonzeros; k++) {
                        int i = A->colidx[A->diag.base.idx[k]];
                        ydata[i][0] += alpha*(Adiag[k][0]*xdata[i][0]+Adiag[k][1]*xdata[i][1]);
                        ydata[i][1] += alpha*(Adiag[k][0]*xdata[i][1]-Adiag[k][1]*xdata[i][0]);
                    }
                    if (num_flops) *num_flops += 20*A->size + 10*A->diag.base.num_nonzeros;
                } else if (precision == mtx_double) {
                    const double (* Adata)[2] = a->base.data.complex_double;
                    const double (* xdata)[2] = xbase->data.complex_double;
                    double (* ydata)[2] = ybase->data.complex_double;
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++) {
                            ydata[i][0] += alpha*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]);
                            ydata[i][1] += alpha*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]);
                            ydata[j[k]][0] += alpha*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                            ydata[j[k]][1] += alpha*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        }
                    }
                    const double (* Adiag)[2] = A->diag.base.data.complex_double;
                    for (int k = 0; k < A->diag.base.num_nonzeros; k++) {
                        int i = A->colidx[A->diag.base.idx[k]];
                        ydata[i][0] += alpha*(Adiag[k][0]*xdata[i][0]+Adiag[k][1]*xdata[i][1]);
                        ydata[i][1] += alpha*(Adiag[k][0]*xdata[i][1]-Adiag[k][1]*xdata[i][0]);
                    }
                    if (num_flops) *num_flops += 20*A->size + 10*A->diag.base.num_nonzeros;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else { return MTX_ERR_INVALID_TRANSPOSITION; }
        } else if (field == mtx_field_integer) {
            if (precision == mtx_single) {
                const int32_t * Adata = a->base.data.integer_single;
                const int32_t * xdata = xbase->data.integer_single;
                int32_t * ydata = ybase->data.integer_single;
                for (int i = 0; i < A->num_rows; i++) {
                    for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++) {
                        ydata[i] += alpha*Adata[k]*xdata[j[k]];
                        ydata[j[k]] += alpha*Adata[k]*xdata[i];
                    }
                }
                const int32_t * Adiag = A->diag.base.data.integer_single;
                for (int k = 0; k < A->diag.base.num_nonzeros; k++) {
                    int i = A->colidx[A->diag.base.idx[k]];
                    ydata[i] += alpha*Adiag[k]*xdata[i];
                }
                if (num_flops) *num_flops += 6*A->size + 3*A->diag.base.num_nonzeros;
            } else if (precision == mtx_double) {
                const int64_t * Adata = a->base.data.integer_double;
                const int64_t * xdata = xbase->data.integer_double;
                int64_t * ydata = ybase->data.integer_double;
                for (int i = 0; i < A->num_rows; i++) {
                    for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++) {
                        ydata[i] += alpha*Adata[k]*xdata[j[k]];
                        ydata[j[k]] += alpha*Adata[k]*xdata[i];
                    }
                }
                const int64_t * Adiag = A->diag.base.data.integer_double;
                for (int k = 0; k < A->diag.base.num_nonzeros; k++) {
                    int i = A->colidx[A->diag.base.idx[k]];
                    ydata[i] += alpha*Adiag[k]*xdata[i];
                }
                if (num_flops) *num_flops += 6*A->size + 3*A->diag.base.num_nonzeros;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else { return MTX_ERR_INVALID_FIELD; }
        err = mtxompvector_ussc((struct mtxompvector *) &A->a, &A->diag);
        if (err) return err;
    } else if (A->num_rows == A->num_columns && A->symmetry == mtx_skew_symmetric) {
        /* TODO: allow skew-symmetric matrices */
        return MTX_ERR_INVALID_SYMMETRY;
    } else if (A->num_rows == A->num_columns && A->symmetry == mtx_hermitian) {
        int err = mtxompvector_usgz(
            (struct mtxompvector *) &A->diag,
            (struct mtxompvector *) &A->a);
        if (err) return err;
        if (field == mtx_field_complex) {
            if (trans == mtx_notrans || trans == mtx_conjtrans) {
                if (precision == mtx_single) {
                    const float (* Adata)[2] = a->base.data.complex_single;
                    const float (* xdata)[2] = xbase->data.complex_single;
                    float (* ydata)[2] = ybase->data.complex_single;
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++) {
                            ydata[i][0] += alpha*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                            ydata[i][1] += alpha*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                            ydata[j[k]][0] += alpha*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                            ydata[j[k]][1] += alpha*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        }
                    }
                    const float (* Adiag)[2] = A->diag.base.data.complex_single;
                    for (int k = 0; k < A->diag.base.num_nonzeros; k++) {
                        int i = A->colidx[A->diag.base.idx[k]];
                        ydata[i][0] += alpha*(Adiag[k][0]*xdata[i][0]+Adiag[k][1]*xdata[i][1]);
                        ydata[i][1] += alpha*(Adiag[k][0]*xdata[i][1]-Adiag[k][1]*xdata[i][0]);
                    }
                    if (num_flops) *num_flops += 20*A->size + 10*A->diag.base.num_nonzeros;
                } else if (precision == mtx_double) {
                    const double (* Adata)[2] = a->base.data.complex_double;
                    const double (* xdata)[2] = xbase->data.complex_double;
                    double (* ydata)[2] = ybase->data.complex_double;
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++) {
                            ydata[i][0] += alpha*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                            ydata[i][1] += alpha*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                            ydata[j[k]][0] += alpha*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                            ydata[j[k]][1] += alpha*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        }
                    }
                    const double (* Adiag)[2] = A->diag.base.data.complex_double;
                    for (int k = 0; k < A->diag.base.num_nonzeros; k++) {
                        int i = A->colidx[A->diag.base.idx[k]];
                        ydata[i][0] += alpha*(Adiag[k][0]*xdata[i][0]+Adiag[k][1]*xdata[i][1]);
                        ydata[i][1] += alpha*(Adiag[k][0]*xdata[i][1]-Adiag[k][1]*xdata[i][0]);
                    }
                    if (num_flops) *num_flops += 20*A->size + 10*A->diag.base.num_nonzeros;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (trans == mtx_trans) {
                if (precision == mtx_single) {
                    const float (* Adata)[2] = a->base.data.complex_single;
                    const float (* xdata)[2] = xbase->data.complex_single;
                    float (* ydata)[2] = ybase->data.complex_single;
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++) {
                            ydata[i][0] += alpha*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]);
                            ydata[i][1] += alpha*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]);
                            ydata[j[k]][0] += alpha*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                            ydata[j[k]][1] += alpha*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        }
                    }
                    const float (* Adiag)[2] = A->diag.base.data.complex_single;
                    for (int k = 0; k < A->diag.base.num_nonzeros; k++) {
                        int i = A->colidx[A->diag.base.idx[k]];
                        ydata[i][0] += alpha*(Adiag[k][0]*xdata[i][0]-Adiag[k][1]*xdata[i][1]);
                        ydata[i][1] += alpha*(Adiag[k][0]*xdata[i][1]+Adiag[k][1]*xdata[i][0]);
                    }
                    if (num_flops) *num_flops += 20*A->size + 10*A->diag.base.num_nonzeros;
                } else if (precision == mtx_double) {
                    const double (* Adata)[2] = a->base.data.complex_double;
                    const double (* xdata)[2] = xbase->data.complex_double;
                    double (* ydata)[2] = ybase->data.complex_double;
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++) {
                            ydata[i][0] += alpha*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]);
                            ydata[i][1] += alpha*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]);
                            ydata[j[k]][0] += alpha*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                            ydata[j[k]][1] += alpha*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        }
                    }
                    const double (* Adiag)[2] = A->diag.base.data.complex_double;
                    for (int k = 0; k < A->diag.base.num_nonzeros; k++) {
                        int i = A->colidx[A->diag.base.idx[k]];
                        ydata[i][0] += alpha*(Adiag[k][0]*xdata[i][0]-Adiag[k][1]*xdata[i][1]);
                        ydata[i][1] += alpha*(Adiag[k][0]*xdata[i][1]+Adiag[k][1]*xdata[i][0]);
                    }
                    if (num_flops) *num_flops += 20*A->size + 10*A->diag.base.num_nonzeros;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else { return MTX_ERR_INVALID_TRANSPOSITION; }
        } else { return MTX_ERR_INVALID_FIELD; }
        err = mtxompvector_ussc((struct mtxompvector *) &A->a, &A->diag);
        if (err) return err;
    } else { return MTX_ERR_INVALID_SYMMETRY; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxompcsr_dgemv()’ multiplies a matrix ‘A’ or its transpose
 * ‘A'’ by a real scalar ‘alpha’ (‘α’) and a vector ‘x’, before adding
 * the result to another vector ‘y’ multiplied by another scalar real
 * ‘beta’ (‘β’).  That is, ‘y = α*A*x + β*y’ or ‘y = α*A'*x + β*y’.
 *
 * The scalars ‘alpha’ and ‘beta’ are given as double precision
 * floating point numbers.
 *
 * The vectors ‘x’ and ‘y’ must be of ‘array’ type, and they must both
 * have the same field and precision.  If the field of the matrix ‘A’
 * is ‘real’, ‘integer’ or ‘complex’, then the vectors must have the
 * same field.  Otherwise, if the matrix field is ‘pattern’, then ‘x’
 * and ‘y’ are allowed to be ‘real’, ‘integer’ or ‘complex’, but they
 * must both have the same field.
 *
 * Moreover, if ‘trans’ is ‘mtx_notrans’, then the size of ‘x’ must
 * equal the number of columns of ‘A’ and the size of ‘y’ must equal
 * the number of rows of ‘A’. if ‘trans’ is ‘mtx_trans’ or
 * ‘mtx_conjtrans’, then the size of ‘x’ must equal the number of rows
 * of ‘A’ and the size of ‘y’ must equal the number of columns of ‘A’.
 *
 * For now, the only case that is parallelised with OpenMP is
 * multiplication with non-transposed and unsymmetric matrices.
 */
int mtxompcsr_dgemv(
    enum mtxtransposition trans,
    double alpha,
    const struct mtxompcsr * A,
    const struct mtxvector * x,
    double beta,
    struct mtxvector * y,
    int64_t * num_flops)
{
    int err;
    enum mtxfield field = mtxompcsr_field(A);
    enum mtxprecision precision = mtxompcsr_precision(A);
    const struct mtxompvector * a = &A->a;
    if (x->type != mtxompvector || y->type != mtxompvector)
        return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
    const struct mtxompvector * xomp = &x->storage.omp;
    const struct mtxbasevector * xbase = &xomp->base;
    struct mtxompvector * yomp = &y->storage.omp;
    struct mtxbasevector * ybase = &yomp->base;
    if (xbase->field != field || ybase->field != field)
        return MTX_ERR_INCOMPATIBLE_FIELD;
    if (xbase->precision != precision || ybase->precision != precision)
        return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (trans == mtx_notrans) {
        if (A->num_rows != ybase->size || A->num_columns != xbase->size)
            return MTX_ERR_INCOMPATIBLE_SIZE;
    } else if (trans == mtx_trans || trans == mtx_conjtrans) {
        if (A->num_columns != ybase->size || A->num_rows != xbase->size)
            return MTX_ERR_INCOMPATIBLE_SIZE;
    }

    if (beta != 1) {
        err = mtxvector_dscal(beta, y, num_flops);
        if (err) return err;
    }

    const int64_t * rowptr = A->rowptr;
    const int * j = A->colidx;
    if (A->symmetry == mtx_unsymmetric) {
        if (field == mtx_field_real) {
            if (trans == mtx_notrans) {
                if (precision == mtx_single) {
                    const float * Adata = a->base.data.real_single;
                    const float * xdata = xbase->data.real_single;
                    float * ydata = ybase->data.real_single;
                    #pragma omp parallel for
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++)
                            ydata[i] += alpha*Adata[k]*xdata[j[k]];
                    }
                    if (num_flops) *num_flops += 3*A->num_nonzeros;
                } else if (precision == mtx_double) {
                    const double * Adata = a->base.data.real_double;
                    const double * xdata = xbase->data.real_double;
                    double * ydata = ybase->data.real_double;
                    #pragma omp parallel for
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++)
                            ydata[i] += alpha*Adata[k]*xdata[j[k]];
                    }
                    if (num_flops) *num_flops += 3*A->num_nonzeros;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (trans == mtx_trans || trans == mtx_conjtrans) {
                if (precision == mtx_single) {
                    const float * Adata = a->base.data.real_single;
                    const float * xdata = xbase->data.real_single;
                    float * ydata = ybase->data.real_single;
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++)
                            ydata[j[k]] += alpha*Adata[k]*xdata[i];
                    }
                    if (num_flops) *num_flops += 3*A->num_nonzeros;
                } else if (precision == mtx_double) {
                    const double * Adata = a->base.data.real_double;
                    const double * xdata = xbase->data.real_double;
                    double * ydata = ybase->data.real_double;
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++)
                            ydata[j[k]] += alpha*Adata[k]*xdata[i];
                    }
                    if (num_flops) *num_flops += 3*A->num_nonzeros;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else { return MTX_ERR_INVALID_TRANSPOSITION; }
        } else if (field == mtx_field_complex) {
            if (trans == mtx_notrans) {
                if (precision == mtx_single) {
                    const float (* Adata)[2] = a->base.data.complex_single;
                    const float (* xdata)[2] = xbase->data.complex_single;
                    float (* ydata)[2] = ybase->data.complex_single;
                    #pragma omp parallel for
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++) {
                            ydata[i][0] += alpha*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                            ydata[i][1] += alpha*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_nonzeros;
                } else if (precision == mtx_double) {
                    const double (* Adata)[2] = a->base.data.complex_double;
                    const double (* xdata)[2] = xbase->data.complex_double;
                    double (* ydata)[2] = ybase->data.complex_double;
                    #pragma omp parallel for
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++) {
                            ydata[i][0] += alpha*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                            ydata[i][1] += alpha*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_nonzeros;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (trans == mtx_trans) {
                if (precision == mtx_single) {
                    const float (* Adata)[2] = a->base.data.complex_single;
                    const float (* xdata)[2] = xbase->data.complex_single;
                    float (* ydata)[2] = ybase->data.complex_single;
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++) {
                            ydata[j[k]][0] += alpha*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                            ydata[j[k]][1] += alpha*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_nonzeros;
                } else if (precision == mtx_double) {
                    const double (* Adata)[2] = a->base.data.complex_double;
                    const double (* xdata)[2] = xbase->data.complex_double;
                    double (* ydata)[2] = ybase->data.complex_double;
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++) {
                            ydata[j[k]][0] += alpha*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                            ydata[j[k]][1] += alpha*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_nonzeros;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (trans == mtx_conjtrans) {
                if (precision == mtx_single) {
                    const float (* Adata)[2] = a->base.data.complex_single;
                    const float (* xdata)[2] = xbase->data.complex_single;
                    float (* ydata)[2] = ybase->data.complex_single;
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++) {
                            ydata[j[k]][0] += alpha*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                            ydata[j[k]][1] += alpha*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_nonzeros;
                } else if (precision == mtx_double) {
                    const double (* Adata)[2] = a->base.data.complex_double;
                    const double (* xdata)[2] = xbase->data.complex_double;
                    double (* ydata)[2] = ybase->data.complex_double;
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++) {
                            ydata[j[k]][0] += alpha*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                            ydata[j[k]][1] += alpha*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_nonzeros;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else { return MTX_ERR_INVALID_TRANSPOSITION; }
        } else if (field == mtx_field_integer) {
            if (trans == mtx_notrans) {
                if (precision == mtx_single) {
                    const int32_t * Adata = a->base.data.integer_single;
                    const int32_t * xdata = xbase->data.integer_single;
                    int32_t * ydata = ybase->data.integer_single;
                    #pragma omp parallel for
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++)
                            ydata[i] += alpha*Adata[k]*xdata[j[k]];
                    }
                    if (num_flops) *num_flops += 3*A->num_nonzeros;
                } else if (precision == mtx_double) {
                    const int64_t * Adata = a->base.data.integer_double;
                    const int64_t * xdata = xbase->data.integer_double;
                    int64_t * ydata = ybase->data.integer_double;
                    #pragma omp parallel for
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++)
                            ydata[i] += alpha*Adata[k]*xdata[j[k]];
                    }
                    if (num_flops) *num_flops += 3*A->num_nonzeros;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (trans == mtx_trans || trans == mtx_conjtrans) {
                if (precision == mtx_single) {
                    const int32_t * Adata = a->base.data.integer_single;
                    const int32_t * xdata = xbase->data.integer_single;
                    int32_t * ydata = ybase->data.integer_single;
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++)
                            ydata[j[k]] += alpha*Adata[k]*xdata[i];
                    }
                    if (num_flops) *num_flops += 3*A->num_nonzeros;
                } else if (precision == mtx_double) {
                    const int64_t * Adata = a->base.data.integer_double;
                    const int64_t * xdata = xbase->data.integer_double;
                    int64_t * ydata = ybase->data.integer_double;
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++)
                            ydata[j[k]] += alpha*Adata[k]*xdata[i];
                    }
                    if (num_flops) *num_flops += 3*A->num_nonzeros;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else { return MTX_ERR_INVALID_TRANSPOSITION; }
        } else { return MTX_ERR_INVALID_FIELD; }
    } else if (A->num_rows == A->num_columns && A->symmetry == mtx_symmetric) {
        int err = mtxompvector_usgz(
            (struct mtxompvector *) &A->diag,
            (struct mtxompvector *) &A->a);
        if (err) return err;
        if (field == mtx_field_real) {
            if (precision == mtx_single) {
                const float * Adata = a->base.data.real_single;
                const float * xdata = xbase->data.real_single;
                float * ydata = ybase->data.real_single;
                for (int i = 0; i < A->num_rows; i++) {
                    for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++) {
                        ydata[i] += alpha*Adata[k]*xdata[j[k]];
                        ydata[j[k]] += alpha*Adata[k]*xdata[i];
                    }
                }
                const float * Adiag = A->diag.base.data.real_single;
                for (int k = 0; k < A->diag.base.num_nonzeros; k++) {
                    int i = A->colidx[A->diag.base.idx[k]];
                    ydata[i] += alpha*Adiag[k]*xdata[i];
                }
                if (num_flops) *num_flops += 6*A->size + 3*A->diag.base.num_nonzeros;
            } else if (precision == mtx_double) {
                const double * Adata = a->base.data.real_double;
                const double * xdata = xbase->data.real_double;
                double * ydata = ybase->data.real_double;
                for (int i = 0; i < A->num_rows; i++) {
                    for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++) {
                        ydata[i] += alpha*Adata[k]*xdata[j[k]];
                        ydata[j[k]] += alpha*Adata[k]*xdata[i];
                    }
                }
                const double * Adiag = A->diag.base.data.real_double;
                for (int k = 0; k < A->diag.base.num_nonzeros; k++) {
                    int i = A->colidx[A->diag.base.idx[k]];
                    ydata[i] += alpha*Adiag[k]*xdata[i];
                }
                if (num_flops) *num_flops += 6*A->size + 3*A->diag.base.num_nonzeros;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (field == mtx_field_complex) {
            if (trans == mtx_notrans || trans == mtx_trans) {
                if (precision == mtx_single) {
                    const float (* Adata)[2] = a->base.data.complex_single;
                    const float (* xdata)[2] = xbase->data.complex_single;
                    float (* ydata)[2] = ybase->data.complex_single;
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++) {
                            ydata[i][0] += alpha*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                            ydata[i][1] += alpha*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                            ydata[j[k]][0] += alpha*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                            ydata[j[k]][1] += alpha*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        }
                    }
                    const float (* Adiag)[2] = A->diag.base.data.complex_single;
                    for (int k = 0; k < A->diag.base.num_nonzeros; k++) {
                        int i = A->colidx[A->diag.base.idx[k]];
                        ydata[i][0] += alpha*(Adiag[k][0]*xdata[i][0]-Adiag[k][1]*xdata[i][1]);
                        ydata[i][1] += alpha*(Adiag[k][0]*xdata[i][1]+Adiag[k][1]*xdata[i][0]);
                    }
                    if (num_flops) *num_flops += 20*A->size + 10*A->diag.base.num_nonzeros;
                } else if (precision == mtx_double) {
                    const double (* Adata)[2] = a->base.data.complex_double;
                    const double (* xdata)[2] = xbase->data.complex_double;
                    double (* ydata)[2] = ybase->data.complex_double;
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++) {
                            ydata[i][0] += alpha*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                            ydata[i][1] += alpha*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                            ydata[j[k]][0] += alpha*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                            ydata[j[k]][1] += alpha*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        }
                    }
                    const double (* Adiag)[2] = A->diag.base.data.complex_double;
                    for (int k = 0; k < A->diag.base.num_nonzeros; k++) {
                        int i = A->colidx[A->diag.base.idx[k]];
                        ydata[i][0] += alpha*(Adiag[k][0]*xdata[i][0]-Adiag[k][1]*xdata[i][1]);
                        ydata[i][1] += alpha*(Adiag[k][0]*xdata[i][1]+Adiag[k][1]*xdata[i][0]);
                    }
                    if (num_flops) *num_flops += 20*A->size + 10*A->diag.base.num_nonzeros;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (trans == mtx_conjtrans) {
                if (precision == mtx_single) {
                    const float (* Adata)[2] = a->base.data.complex_single;
                    const float (* xdata)[2] = xbase->data.complex_single;
                    float (* ydata)[2] = ybase->data.complex_single;
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++) {
                            ydata[i][0] += alpha*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]);
                            ydata[i][1] += alpha*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]);
                            ydata[j[k]][0] += alpha*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                            ydata[j[k]][1] += alpha*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        }
                    }
                    const float (* Adiag)[2] = A->diag.base.data.complex_single;
                    for (int k = 0; k < A->diag.base.num_nonzeros; k++) {
                        int i = A->colidx[A->diag.base.idx[k]];
                        ydata[i][0] += alpha*(Adiag[k][0]*xdata[i][0]+Adiag[k][1]*xdata[i][1]);
                        ydata[i][1] += alpha*(Adiag[k][0]*xdata[i][1]-Adiag[k][1]*xdata[i][0]);
                    }
                    if (num_flops) *num_flops += 20*A->size + 10*A->diag.base.num_nonzeros;
                } else if (precision == mtx_double) {
                    const double (* Adata)[2] = a->base.data.complex_double;
                    const double (* xdata)[2] = xbase->data.complex_double;
                    double (* ydata)[2] = ybase->data.complex_double;
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++) {
                            ydata[i][0] += alpha*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]);
                            ydata[i][1] += alpha*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]);
                            ydata[j[k]][0] += alpha*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                            ydata[j[k]][1] += alpha*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        }
                    }
                    const double (* Adiag)[2] = A->diag.base.data.complex_double;
                    for (int k = 0; k < A->diag.base.num_nonzeros; k++) {
                        int i = A->colidx[A->diag.base.idx[k]];
                        ydata[i][0] += alpha*(Adiag[k][0]*xdata[i][0]+Adiag[k][1]*xdata[i][1]);
                        ydata[i][1] += alpha*(Adiag[k][0]*xdata[i][1]-Adiag[k][1]*xdata[i][0]);
                    }
                    if (num_flops) *num_flops += 20*A->size + 10*A->diag.base.num_nonzeros;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else { return MTX_ERR_INVALID_TRANSPOSITION; }
        } else if (field == mtx_field_integer) {
            if (precision == mtx_single) {
                const int32_t * Adata = a->base.data.integer_single;
                const int32_t * xdata = xbase->data.integer_single;
                int32_t * ydata = ybase->data.integer_single;
                for (int i = 0; i < A->num_rows; i++) {
                    for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++) {
                        ydata[i] += alpha*Adata[k]*xdata[j[k]];
                        ydata[j[k]] += alpha*Adata[k]*xdata[i];
                    }
                }
                const int32_t * Adiag = A->diag.base.data.integer_single;
                for (int k = 0; k < A->diag.base.num_nonzeros; k++) {
                    int i = A->colidx[A->diag.base.idx[k]];
                    ydata[i] += alpha*Adiag[k]*xdata[i];
                }
                if (num_flops) *num_flops += 6*A->size + 3*A->diag.base.num_nonzeros;
            } else if (precision == mtx_double) {
                const int64_t * Adata = a->base.data.integer_double;
                const int64_t * xdata = xbase->data.integer_double;
                int64_t * ydata = ybase->data.integer_double;
                for (int i = 0; i < A->num_rows; i++) {
                    for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++) {
                        ydata[i] += alpha*Adata[k]*xdata[j[k]];
                        ydata[j[k]] += alpha*Adata[k]*xdata[i];
                    }
                }
                const int64_t * Adiag = A->diag.base.data.integer_double;
                for (int k = 0; k < A->diag.base.num_nonzeros; k++) {
                    int i = A->colidx[A->diag.base.idx[k]];
                    ydata[i] += alpha*Adiag[k]*xdata[i];
                }
                if (num_flops) *num_flops += 6*A->size + 3*A->diag.base.num_nonzeros;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else { return MTX_ERR_INVALID_FIELD; }
        err = mtxompvector_ussc((struct mtxompvector *) &A->a, &A->diag);
        if (err) return err;
    } else if (A->num_rows == A->num_columns && A->symmetry == mtx_skew_symmetric) {
        /* TODO: allow skew-symmetric matrices */
        return MTX_ERR_INVALID_SYMMETRY;
    } else if (A->num_rows == A->num_columns && A->symmetry == mtx_hermitian) {
        int err = mtxompvector_usgz(
            (struct mtxompvector *) &A->diag,
            (struct mtxompvector *) &A->a);
        if (err) return err;
        if (field == mtx_field_complex) {
            if (trans == mtx_notrans || trans == mtx_conjtrans) {
                if (precision == mtx_single) {
                    const float (* Adata)[2] = a->base.data.complex_single;
                    const float (* xdata)[2] = xbase->data.complex_single;
                    float (* ydata)[2] = ybase->data.complex_single;
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++) {
                            ydata[i][0] += alpha*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                            ydata[i][1] += alpha*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                            ydata[j[k]][0] += alpha*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                            ydata[j[k]][1] += alpha*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        }
                    }
                    const float (* Adiag)[2] = A->diag.base.data.complex_single;
                    for (int k = 0; k < A->diag.base.num_nonzeros; k++) {
                        int i = A->colidx[A->diag.base.idx[k]];
                        ydata[i][0] += alpha*(Adiag[k][0]*xdata[i][0]+Adiag[k][1]*xdata[i][1]);
                        ydata[i][1] += alpha*(Adiag[k][0]*xdata[i][1]-Adiag[k][1]*xdata[i][0]);
                    }
                    if (num_flops) *num_flops += 20*A->size + 10*A->diag.base.num_nonzeros;
                } else if (precision == mtx_double) {
                    const double (* Adata)[2] = a->base.data.complex_double;
                    const double (* xdata)[2] = xbase->data.complex_double;
                    double (* ydata)[2] = ybase->data.complex_double;
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++) {
                            ydata[i][0] += alpha*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                            ydata[i][1] += alpha*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                            ydata[j[k]][0] += alpha*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                            ydata[j[k]][1] += alpha*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        }
                    }
                    const double (* Adiag)[2] = A->diag.base.data.complex_double;
                    for (int k = 0; k < A->diag.base.num_nonzeros; k++) {
                        int i = A->colidx[A->diag.base.idx[k]];
                        ydata[i][0] += alpha*(Adiag[k][0]*xdata[i][0]+Adiag[k][1]*xdata[i][1]);
                        ydata[i][1] += alpha*(Adiag[k][0]*xdata[i][1]-Adiag[k][1]*xdata[i][0]);
                    }
                    if (num_flops) *num_flops += 20*A->size + 10*A->diag.base.num_nonzeros;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (trans == mtx_trans) {
                if (precision == mtx_single) {
                    const float (* Adata)[2] = a->base.data.complex_single;
                    const float (* xdata)[2] = xbase->data.complex_single;
                    float (* ydata)[2] = ybase->data.complex_single;
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++) {
                            ydata[i][0] += alpha*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]);
                            ydata[i][1] += alpha*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]);
                            ydata[j[k]][0] += alpha*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                            ydata[j[k]][1] += alpha*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        }
                    }
                    const float (* Adiag)[2] = A->diag.base.data.complex_single;
                    for (int k = 0; k < A->diag.base.num_nonzeros; k++) {
                        int i = A->colidx[A->diag.base.idx[k]];
                        ydata[i][0] += alpha*(Adiag[k][0]*xdata[i][0]-Adiag[k][1]*xdata[i][1]);
                        ydata[i][1] += alpha*(Adiag[k][0]*xdata[i][1]+Adiag[k][1]*xdata[i][0]);
                    }
                    if (num_flops) *num_flops += 20*A->size + 10*A->diag.base.num_nonzeros;
                } else if (precision == mtx_double) {
                    const double (* Adata)[2] = a->base.data.complex_double;
                    const double (* xdata)[2] = xbase->data.complex_double;
                    double (* ydata)[2] = ybase->data.complex_double;
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++) {
                            ydata[i][0] += alpha*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]);
                            ydata[i][1] += alpha*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]);
                            ydata[j[k]][0] += alpha*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                            ydata[j[k]][1] += alpha*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        }
                    }
                    const double (* Adiag)[2] = A->diag.base.data.complex_double;
                    for (int k = 0; k < A->diag.base.num_nonzeros; k++) {
                        int i = A->colidx[A->diag.base.idx[k]];
                        ydata[i][0] += alpha*(Adiag[k][0]*xdata[i][0]-Adiag[k][1]*xdata[i][1]);
                        ydata[i][1] += alpha*(Adiag[k][0]*xdata[i][1]+Adiag[k][1]*xdata[i][0]);
                    }
                    if (num_flops) *num_flops += 20*A->size + 10*A->diag.base.num_nonzeros;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else { return MTX_ERR_INVALID_TRANSPOSITION; }
        } else { return MTX_ERR_INVALID_FIELD; }
        err = mtxompvector_ussc((struct mtxompvector *) &A->a, &A->diag);
        if (err) return err;
    } else { return MTX_ERR_INVALID_SYMMETRY; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxompcsr_cgemv()’ multiplies a complex-valued matrix ‘A’, its
 * transpose ‘A'’ or its conjugate transpose ‘Aᴴ’ by a complex scalar
 * ‘alpha’ (‘α’) and a vector ‘x’, before adding the result to another
 * vector ‘y’ multiplied by another complex scalar ‘beta’ (‘β’).  That
 * is, ‘y = α*A*x + β*y’, ‘y = α*A'*x + β*y’ or ‘y = α*Aᴴ*x + β*y’.
 *
 * The scalars ‘alpha’ and ‘beta’ are given as single precision
 * floating point numbers.
 *
 * The vectors ‘x’ and ‘y’ must be of ‘array’ type, and they must have
 * the same field and precision as the matrix ‘A’. Moreover, if
 * ‘trans’ is ‘mtx_notrans’, then the size of ‘x’ must equal the
 * number of columns of ‘A’ and the size of ‘y’ must equal the number
 * of rows of ‘A’. if ‘trans’ is ‘mtx_trans’ or ‘mtx_conjtrans’, then
 * the size of ‘x’ must equal the number of rows of ‘A’ and the size
 * of ‘y’ must equal the number of columns of ‘A’.
 *
 * For now, the only case that is parallelised with OpenMP is
 * multiplication with non-transposed and unsymmetric matrices.
 */
int mtxompcsr_cgemv(
    enum mtxtransposition trans,
    float alpha[2],
    const struct mtxompcsr * A,
    const struct mtxvector * x,
    float beta[2],
    struct mtxvector * y,
    int64_t * num_flops)
{
    int err;
    enum mtxfield field = mtxompcsr_field(A);
    enum mtxprecision precision = mtxompcsr_precision(A);
    const struct mtxompvector * a = &A->a;
    if (x->type != mtxompvector || y->type != mtxompvector)
        return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
    const struct mtxompvector * xomp = &x->storage.omp;
    const struct mtxbasevector * xbase = &xomp->base;
    struct mtxompvector * yomp = &y->storage.omp;
    struct mtxbasevector * ybase = &yomp->base;
    if (xbase->field != field || ybase->field != field)
        return MTX_ERR_INCOMPATIBLE_FIELD;
    if (xbase->precision != precision || ybase->precision != precision)
        return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (trans == mtx_notrans) {
        if (A->num_rows != ybase->size || A->num_columns != xbase->size)
            return MTX_ERR_INCOMPATIBLE_SIZE;
    } else if (trans == mtx_trans || trans == mtx_conjtrans) {
        if (A->num_columns != ybase->size || A->num_rows != xbase->size)
            return MTX_ERR_INCOMPATIBLE_SIZE;
    }
    if (field != mtx_field_complex)
        return MTX_ERR_INCOMPATIBLE_FIELD;

    if (beta[0] != 1 || beta[1] != 0) {
        err = mtxvector_cscal(beta, y, num_flops);
        if (err) return err;
    }

    const int64_t * rowptr = A->rowptr;
    const int * j = A->colidx;
    if (A->symmetry == mtx_unsymmetric) {
        if (trans == mtx_notrans) {
            if (precision == mtx_single) {
                const float (* Adata)[2] = a->base.data.complex_single;
                const float (* xdata)[2] = xbase->data.complex_single;
                float (* ydata)[2] = ybase->data.complex_single;
                #pragma omp parallel for
                for (int i = 0; i < A->num_rows; i++) {
                    for (int k = rowptr[i]; k < rowptr[i+1]; k++) {
                        ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1])-alpha[1]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                        ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0])+alpha[1]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                    }
                }
                if (num_flops) *num_flops += 20*A->num_nonzeros;
            } else if (precision == mtx_double) {
                const double (* Adata)[2] = a->base.data.complex_double;
                const double (* xdata)[2] = xbase->data.complex_double;
                double (* ydata)[2] = ybase->data.complex_double;
                #pragma omp parallel for
                for (int i = 0; i < A->num_rows; i++) {
                    for (int k = rowptr[i]; k < rowptr[i+1]; k++) {
                        ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1])-alpha[1]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                        ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0])+alpha[1]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                    }
                }
                if (num_flops) *num_flops += 20*A->num_nonzeros;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (trans == mtx_trans) {
            if (precision == mtx_single) {
                const float (* Adata)[2] = a->base.data.complex_single;
                const float (* xdata)[2] = xbase->data.complex_single;
                float (* ydata)[2] = ybase->data.complex_single;
                for (int i = 0; i < A->num_rows; i++) {
                    for (int k = rowptr[i]; k < rowptr[i+1]; k++) {
                        ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1])-alpha[1]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0])+alpha[1]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                    }
                }
                if (num_flops) *num_flops += 20*A->num_nonzeros;
            } else if (precision == mtx_double) {
                const double (* Adata)[2] = a->base.data.complex_double;
                const double (* xdata)[2] = xbase->data.complex_double;
                double (* ydata)[2] = ybase->data.complex_double;
                for (int i = 0; i < A->num_rows; i++) {
                    for (int k = rowptr[i]; k < rowptr[i+1]; k++) {
                        ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1])-alpha[1]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0])+alpha[1]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                    }
                }
                if (num_flops) *num_flops += 20*A->num_nonzeros;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (trans == mtx_conjtrans) {
            if (precision == mtx_single) {
                const float (* Adata)[2] = a->base.data.complex_single;
                const float (* xdata)[2] = xbase->data.complex_single;
                float (* ydata)[2] = ybase->data.complex_single;
                for (int i = 0; i < A->num_rows; i++) {
                    for (int k = rowptr[i]; k < rowptr[i+1]; k++) {
                        ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1])-alpha[1]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0])+alpha[1]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                    }
                }
                if (num_flops) *num_flops += 20*A->num_nonzeros;
            } else if (precision == mtx_double) {
                const double (* Adata)[2] = a->base.data.complex_double;
                const double (* xdata)[2] = xbase->data.complex_double;
                double (* ydata)[2] = ybase->data.complex_double;
                for (int i = 0; i < A->num_rows; i++) {
                    for (int k = rowptr[i]; k < rowptr[i+1]; k++) {
                        ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1])-alpha[1]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0])+alpha[1]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                    }
                }
                if (num_flops) *num_flops += 20*A->num_nonzeros;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else { return MTX_ERR_INVALID_TRANSPOSITION; }
    } else if (A->num_rows == A->num_columns && A->symmetry == mtx_symmetric) {
        int err = mtxompvector_usgz(
            (struct mtxompvector *) &A->diag,
            (struct mtxompvector *) &A->a);
        if (err) return err;
        if (trans == mtx_notrans || trans == mtx_trans) {
            if (precision == mtx_single) {
                const float (* Adata)[2] = a->base.data.complex_single;
                const float (* xdata)[2] = xbase->data.complex_single;
                float (* ydata)[2] = ybase->data.complex_single;
                for (int i = 0; i < A->num_rows; i++) {
                    for (int k = rowptr[i]; k < rowptr[i+1]; k++) {
                        ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]) - alpha[1]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                        ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]) + alpha[1]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                        ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                    }
                }
                const float (* Adiag)[2] = A->diag.base.data.complex_single;
                for (int k = 0; k < A->diag.base.num_nonzeros; k++) {
                    int i = A->colidx[A->diag.base.idx[k]];
                    ydata[i][0] += alpha[0]*(Adiag[k][0]*xdata[i][0]-Adiag[k][1]*xdata[i][1]) - alpha[1]*(Adiag[k][0]*xdata[i][1]+Adiag[k][1]*xdata[i][0]);
                    ydata[i][1] += alpha[0]*(Adiag[k][0]*xdata[i][1]+Adiag[k][1]*xdata[i][0]) + alpha[1]*(Adiag[k][0]*xdata[i][0]-Adiag[k][1]*xdata[i][1]);
                }
                if (num_flops) *num_flops += 40*A->size + 20*A->diag.base.num_nonzeros;
            } else if (precision == mtx_double) {
                const double (* Adata)[2] = a->base.data.complex_double;
                const double (* xdata)[2] = xbase->data.complex_double;
                double (* ydata)[2] = ybase->data.complex_double;
                for (int i = 0; i < A->num_rows; i++) {
                    for (int k = rowptr[i]; k < rowptr[i+1]; k++) {
                        ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]) - alpha[1]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                        ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]) + alpha[1]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                        ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                    }
                }
                const double (* Adiag)[2] = A->diag.base.data.complex_double;
                for (int k = 0; k < A->diag.base.num_nonzeros; k++) {
                    int i = A->colidx[A->diag.base.idx[k]];
                    ydata[j[k]][0] += alpha[0]*(Adiag[k][0]*xdata[i][0]-Adiag[k][1]*xdata[i][1]) - alpha[1]*(Adiag[k][0]*xdata[i][1]+Adiag[k][1]*xdata[i][0]);
                    ydata[j[k]][1] += alpha[0]*(Adiag[k][0]*xdata[i][1]+Adiag[k][1]*xdata[i][0]) + alpha[1]*(Adiag[k][0]*xdata[i][0]-Adiag[k][1]*xdata[i][1]);
                }
                if (num_flops) *num_flops += 40*A->size + 20*A->diag.base.num_nonzeros;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (trans == mtx_conjtrans) {
            if (precision == mtx_single) {
                const float (* Adata)[2] = a->base.data.complex_single;
                const float (* xdata)[2] = xbase->data.complex_single;
                float (* ydata)[2] = ybase->data.complex_single;
                for (int i = 0; i < A->num_rows; i++) {
                    for (int k = rowptr[i]; k < rowptr[i+1]; k++) {
                        ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]) - alpha[1]*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]);
                        ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]) + alpha[1]*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]);
                        ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                    }
                }
                const float (* Adiag)[2] = A->diag.base.data.complex_single;
                for (int k = 0; k < A->diag.base.num_nonzeros; k++) {
                    int i = A->colidx[A->diag.base.idx[k]];
                    ydata[i][0] += alpha[0]*(Adiag[k][0]*xdata[i][0]+Adiag[k][1]*xdata[i][1]) - alpha[1]*(Adiag[k][0]*xdata[i][1]-Adiag[k][1]*xdata[i][0]);
                    ydata[i][1] += alpha[0]*(Adiag[k][0]*xdata[i][1]-Adiag[k][1]*xdata[i][0]) + alpha[1]*(Adiag[k][0]*xdata[i][0]+Adiag[k][1]*xdata[i][1]);
                }
                if (num_flops) *num_flops += 40*A->size + 20*A->diag.base.num_nonzeros;
            } else if (precision == mtx_double) {
                const double (* Adata)[2] = a->base.data.complex_double;
                const double (* xdata)[2] = xbase->data.complex_double;
                double (* ydata)[2] = ybase->data.complex_double;
                for (int i = 0; i < A->num_rows; i++) {
                    for (int k = rowptr[i]; k < rowptr[i+1]; k++) {
                        ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]) - alpha[1]*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]);
                        ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]) + alpha[1]*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]);
                        ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                    }
                }
                const double (* Adiag)[2] = A->diag.base.data.complex_double;
                for (int k = 0; k < A->diag.base.num_nonzeros; k++) {
                    int i = A->colidx[A->diag.base.idx[k]];
                        ydata[i][0] += alpha[0]*(Adiag[k][0]*xdata[i][0]+Adiag[k][1]*xdata[i][1]) - alpha[1]*(Adiag[k][0]*xdata[i][1]-Adiag[k][1]*xdata[i][0]);
                        ydata[i][1] += alpha[0]*(Adiag[k][0]*xdata[i][1]-Adiag[k][1]*xdata[i][0]) + alpha[1]*(Adiag[k][0]*xdata[i][0]+Adiag[k][1]*xdata[i][1]);
                }
                if (num_flops) *num_flops += 40*A->size + 20*A->diag.base.num_nonzeros;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else { return MTX_ERR_INVALID_TRANSPOSITION; }
        err = mtxompvector_ussc((struct mtxompvector *) &A->a, &A->diag);
        if (err) return err;
    } else if (A->num_rows == A->num_columns && A->symmetry == mtx_hermitian) {
        int err = mtxompvector_usgz(
            (struct mtxompvector *) &A->diag,
            (struct mtxompvector *) &A->a);
        if (err) return err;
        if (trans == mtx_notrans || trans == mtx_conjtrans) {
            if (precision == mtx_single) {
                const float (* Adata)[2] = a->base.data.complex_single;
                const float (* xdata)[2] = xbase->data.complex_single;
                float (* ydata)[2] = ybase->data.complex_single;
                for (int i = 0; i < A->num_rows; i++) {
                    for (int k = rowptr[i]; k < rowptr[i+1]; k++) {
                        ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]) - alpha[1]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                        ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]) + alpha[1]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                        ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                    }
                }
                const float (* Adiag)[2] = A->diag.base.data.complex_single;
                for (int k = 0; k < A->diag.base.num_nonzeros; k++) {
                    int i = A->colidx[A->diag.base.idx[k]];
                    ydata[i][0] += alpha[0]*(Adiag[k][0]*xdata[i][0]+Adiag[k][1]*xdata[i][1]) - alpha[1]*(Adiag[k][0]*xdata[i][1]-Adiag[k][1]*xdata[i][0]);
                    ydata[i][1] += alpha[0]*(Adiag[k][0]*xdata[i][1]-Adiag[k][1]*xdata[i][0]) + alpha[1]*(Adiag[k][0]*xdata[i][0]+Adiag[k][1]*xdata[i][1]);
                }
                if (num_flops) *num_flops += 40*A->size + 20*A->diag.base.num_nonzeros;
            } else if (precision == mtx_double) {
                const double (* Adata)[2] = a->base.data.complex_double;
                const double (* xdata)[2] = xbase->data.complex_double;
                double (* ydata)[2] = ybase->data.complex_double;
                for (int i = 0; i < A->num_rows; i++) {
                    for (int k = rowptr[i]; k < rowptr[i+1]; k++) {
                        ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]) - alpha[1]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                        ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]) + alpha[1]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                        ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                    }
                }
                const double (* Adiag)[2] = A->diag.base.data.complex_double;
                for (int k = 0; k < A->diag.base.num_nonzeros; k++) {
                    int i = A->colidx[A->diag.base.idx[k]];
                    ydata[i][0] += alpha[0]*(Adiag[k][0]*xdata[i][0]+Adiag[k][1]*xdata[i][1]) - alpha[1]*(Adiag[k][0]*xdata[i][1]-Adiag[k][1]*xdata[i][0]);
                    ydata[i][1] += alpha[0]*(Adiag[k][0]*xdata[i][1]-Adiag[k][1]*xdata[i][0]) + alpha[1]*(Adiag[k][0]*xdata[i][0]+Adiag[k][1]*xdata[i][1]);
                }
                if (num_flops) *num_flops += 40*A->size + 20*A->diag.base.num_nonzeros;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (trans == mtx_trans) {
            if (precision == mtx_single) {
                const float (* Adata)[2] = a->base.data.complex_single;
                const float (* xdata)[2] = xbase->data.complex_single;
                float (* ydata)[2] = ybase->data.complex_single;
                for (int i = 0; i < A->num_rows; i++) {
                    for (int k = rowptr[i]; k < rowptr[i+1]; k++) {
                        ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]) - alpha[1]*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]);
                        ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]) + alpha[1]*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]);
                        ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                    }
                }
                const float (* Adiag)[2] = A->diag.base.data.complex_single;
                for (int k = 0; k < A->diag.base.num_nonzeros; k++) {
                    int i = A->colidx[A->diag.base.idx[k]];
                    ydata[i][0] += alpha[0]*(Adiag[k][0]*xdata[i][0]-Adiag[k][1]*xdata[i][1]) - alpha[1]*(Adiag[k][0]*xdata[i][1]+Adiag[k][1]*xdata[i][0]);
                    ydata[i][1] += alpha[0]*(Adiag[k][0]*xdata[i][1]+Adiag[k][1]*xdata[i][0]) + alpha[1]*(Adiag[k][0]*xdata[i][0]-Adiag[k][1]*xdata[i][1]);
                }
                if (num_flops) *num_flops += 40*A->size + 20*A->diag.base.num_nonzeros;
            } else if (precision == mtx_double) {
                const double (* Adata)[2] = a->base.data.complex_double;
                const double (* xdata)[2] = xbase->data.complex_double;
                double (* ydata)[2] = ybase->data.complex_double;
                for (int i = 0; i < A->num_rows; i++) {
                    for (int k = rowptr[i]; k < rowptr[i+1]; k++) {
                        ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]) - alpha[1]*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]);
                        ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]) + alpha[1]*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]);
                        ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                    }
                }
                const double (* Adiag)[2] = A->diag.base.data.complex_double;
                for (int k = 0; k < A->diag.base.num_nonzeros; k++) {
                    int i = A->colidx[A->diag.base.idx[k]];
                    ydata[i][0] += alpha[0]*(Adiag[k][0]*xdata[i][0]-Adiag[k][1]*xdata[i][1]) - alpha[1]*(Adiag[k][0]*xdata[i][1]+Adiag[k][1]*xdata[i][0]);
                    ydata[i][1] += alpha[0]*(Adiag[k][0]*xdata[i][1]+Adiag[k][1]*xdata[i][0]) + alpha[1]*(Adiag[k][0]*xdata[i][0]-Adiag[k][1]*xdata[i][1]);
                }
                if (num_flops) *num_flops += 40*A->size + 20*A->diag.base.num_nonzeros;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else { return MTX_ERR_INVALID_TRANSPOSITION; }
        err = mtxompvector_ussc((struct mtxompvector *) &A->a, &A->diag);
        if (err) return err;
    } else { return MTX_ERR_INVALID_SYMMETRY; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxompcsr_zgemv()’ multiplies a complex-valued matrix ‘A’, its
 * transpose ‘A'’ or its conjugate transpose ‘Aᴴ’ by a complex scalar
 * ‘alpha’ (‘α’) and a vector ‘x’, before adding the result to another
 * vector ‘y’ multiplied by another complex scalar ‘beta’ (‘β’).  That
 * is, ‘y = α*A*x + β*y’, ‘y = α*A'*x + β*y’ or ‘y = α*Aᴴ*x + β*y’.
 *
 * The scalars ‘alpha’ and ‘beta’ are given as double precision
 * floating point numbers.
 *
 * The vectors ‘x’ and ‘y’ must be of ‘array’ type, and they must have
 * the same field and precision as the matrix ‘A’. Moreover, if
 * ‘trans’ is ‘mtx_notrans’, then the size of ‘x’ must equal the
 * number of columns of ‘A’ and the size of ‘y’ must equal the number
 * of rows of ‘A’. if ‘trans’ is ‘mtx_trans’ or ‘mtx_conjtrans’, then
 * the size of ‘x’ must equal the number of rows of ‘A’ and the size
 * of ‘y’ must equal the number of columns of ‘A’.
 *
 * For now, the only case that is parallelised with OpenMP is
 * multiplication with non-transposed and unsymmetric matrices.
 */
int mtxompcsr_zgemv(
    enum mtxtransposition trans,
    double alpha[2],
    const struct mtxompcsr * A,
    const struct mtxvector * x,
    double beta[2],
    struct mtxvector * y,
    int64_t * num_flops)
{
    int err;
    enum mtxfield field = mtxompcsr_field(A);
    enum mtxprecision precision = mtxompcsr_precision(A);
    const struct mtxompvector * a = &A->a;
    if (x->type != mtxompvector || y->type != mtxompvector)
        return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
    const struct mtxompvector * xomp = &x->storage.omp;
    const struct mtxbasevector * xbase = &xomp->base;
    struct mtxompvector * yomp = &y->storage.omp;
    struct mtxbasevector * ybase = &yomp->base;
    if (xbase->field != field || ybase->field != field)
        return MTX_ERR_INCOMPATIBLE_FIELD;
    if (xbase->precision != precision || ybase->precision != precision)
        return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (trans == mtx_notrans) {
        if (A->num_rows != ybase->size || A->num_columns != xbase->size)
            return MTX_ERR_INCOMPATIBLE_SIZE;
    } else if (trans == mtx_trans || trans == mtx_conjtrans) {
        if (A->num_columns != ybase->size || A->num_rows != xbase->size)
            return MTX_ERR_INCOMPATIBLE_SIZE;
    }
    if (field != mtx_field_complex)
        return MTX_ERR_INCOMPATIBLE_FIELD;

    if (beta[0] != 1 || beta[1] != 0) {
        err = mtxvector_zscal(beta, y, num_flops);
        if (err) return err;
    }

    const int64_t * rowptr = A->rowptr;
    const int * j = A->colidx;
    if (A->symmetry == mtx_unsymmetric) {
        if (trans == mtx_notrans) {
            if (precision == mtx_single) {
                const float (* Adata)[2] = a->base.data.complex_single;
                const float (* xdata)[2] = xbase->data.complex_single;
                float (* ydata)[2] = ybase->data.complex_single;
                #pragma omp parallel for
                for (int i = 0; i < A->num_rows; i++) {
                    for (int k = rowptr[i]; k < rowptr[i+1]; k++) {
                        ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1])-alpha[1]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                        ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0])+alpha[1]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                    }
                }
                if (num_flops) *num_flops += 20*A->num_nonzeros;
            } else if (precision == mtx_double) {
                const double (* Adata)[2] = a->base.data.complex_double;
                const double (* xdata)[2] = xbase->data.complex_double;
                double (* ydata)[2] = ybase->data.complex_double;
                #pragma omp parallel for
                for (int i = 0; i < A->num_rows; i++) {
                    for (int k = rowptr[i]; k < rowptr[i+1]; k++) {
                        ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1])-alpha[1]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                        ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0])+alpha[1]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                    }
                }
                if (num_flops) *num_flops += 20*A->num_nonzeros;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (trans == mtx_trans) {
            if (precision == mtx_single) {
                const float (* Adata)[2] = a->base.data.complex_single;
                const float (* xdata)[2] = xbase->data.complex_single;
                float (* ydata)[2] = ybase->data.complex_single;
                for (int i = 0; i < A->num_rows; i++) {
                    for (int k = rowptr[i]; k < rowptr[i+1]; k++) {
                        ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1])-alpha[1]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0])+alpha[1]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                    }
                }
                if (num_flops) *num_flops += 20*A->num_nonzeros;
            } else if (precision == mtx_double) {
                const double (* Adata)[2] = a->base.data.complex_double;
                const double (* xdata)[2] = xbase->data.complex_double;
                double (* ydata)[2] = ybase->data.complex_double;
                for (int i = 0; i < A->num_rows; i++) {
                    for (int k = rowptr[i]; k < rowptr[i+1]; k++) {
                        ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1])-alpha[1]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0])+alpha[1]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                    }
                }
                if (num_flops) *num_flops += 20*A->num_nonzeros;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (trans == mtx_conjtrans) {
            if (precision == mtx_single) {
                const float (* Adata)[2] = a->base.data.complex_single;
                const float (* xdata)[2] = xbase->data.complex_single;
                float (* ydata)[2] = ybase->data.complex_single;
                for (int i = 0; i < A->num_rows; i++) {
                    for (int k = rowptr[i]; k < rowptr[i+1]; k++) {
                        ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1])-alpha[1]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0])+alpha[1]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                    }
                }
                if (num_flops) *num_flops += 20*A->num_nonzeros;
            } else if (precision == mtx_double) {
                const double (* Adata)[2] = a->base.data.complex_double;
                const double (* xdata)[2] = xbase->data.complex_double;
                double (* ydata)[2] = ybase->data.complex_double;
                for (int i = 0; i < A->num_rows; i++) {
                    for (int k = rowptr[i]; k < rowptr[i+1]; k++) {
                        ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1])-alpha[1]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0])+alpha[1]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                    }
                }
                if (num_flops) *num_flops += 20*A->num_nonzeros;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else { return MTX_ERR_INVALID_TRANSPOSITION; }
    } else if (A->num_rows == A->num_columns && A->symmetry == mtx_symmetric) {
        int err = mtxompvector_usgz(
            (struct mtxompvector *) &A->diag,
            (struct mtxompvector *) &A->a);
        if (err) return err;
        if (trans == mtx_notrans || trans == mtx_trans) {
            if (precision == mtx_single) {
                const float (* Adata)[2] = a->base.data.complex_single;
                const float (* xdata)[2] = xbase->data.complex_single;
                float (* ydata)[2] = ybase->data.complex_single;
                for (int i = 0; i < A->num_rows; i++) {
                    for (int k = rowptr[i]; k < rowptr[i+1]; k++) {
                        ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]) - alpha[1]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                        ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]) + alpha[1]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                        ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                    }
                }
                const float (* Adiag)[2] = A->diag.base.data.complex_single;
                for (int k = 0; k < A->diag.base.num_nonzeros; k++) {
                    int i = A->colidx[A->diag.base.idx[k]];
                    ydata[i][0] += alpha[0]*(Adiag[k][0]*xdata[i][0]-Adiag[k][1]*xdata[i][1]) - alpha[1]*(Adiag[k][0]*xdata[i][1]+Adiag[k][1]*xdata[i][0]);
                    ydata[i][1] += alpha[0]*(Adiag[k][0]*xdata[i][1]+Adiag[k][1]*xdata[i][0]) + alpha[1]*(Adiag[k][0]*xdata[i][0]-Adiag[k][1]*xdata[i][1]);
                }
                if (num_flops) *num_flops += 40*A->size + 20*A->diag.base.num_nonzeros;
            } else if (precision == mtx_double) {
                const double (* Adata)[2] = a->base.data.complex_double;
                const double (* xdata)[2] = xbase->data.complex_double;
                double (* ydata)[2] = ybase->data.complex_double;
                for (int i = 0; i < A->num_rows; i++) {
                    for (int k = rowptr[i]; k < rowptr[i+1]; k++) {
                        ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]) - alpha[1]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                        ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]) + alpha[1]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                        ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                    }
                }
                const double (* Adiag)[2] = A->diag.base.data.complex_double;
                for (int k = 0; k < A->diag.base.num_nonzeros; k++) {
                    int i = A->colidx[A->diag.base.idx[k]];
                    ydata[j[k]][0] += alpha[0]*(Adiag[k][0]*xdata[i][0]-Adiag[k][1]*xdata[i][1]) - alpha[1]*(Adiag[k][0]*xdata[i][1]+Adiag[k][1]*xdata[i][0]);
                    ydata[j[k]][1] += alpha[0]*(Adiag[k][0]*xdata[i][1]+Adiag[k][1]*xdata[i][0]) + alpha[1]*(Adiag[k][0]*xdata[i][0]-Adiag[k][1]*xdata[i][1]);
                }
                if (num_flops) *num_flops += 40*A->size + 20*A->diag.base.num_nonzeros;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (trans == mtx_conjtrans) {
            if (precision == mtx_single) {
                const float (* Adata)[2] = a->base.data.complex_single;
                const float (* xdata)[2] = xbase->data.complex_single;
                float (* ydata)[2] = ybase->data.complex_single;
                for (int i = 0; i < A->num_rows; i++) {
                    for (int k = rowptr[i]; k < rowptr[i+1]; k++) {
                        ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]) - alpha[1]*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]);
                        ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]) + alpha[1]*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]);
                        ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                    }
                }
                const float (* Adiag)[2] = A->diag.base.data.complex_single;
                for (int k = 0; k < A->diag.base.num_nonzeros; k++) {
                    int i = A->colidx[A->diag.base.idx[k]];
                    ydata[i][0] += alpha[0]*(Adiag[k][0]*xdata[i][0]+Adiag[k][1]*xdata[i][1]) - alpha[1]*(Adiag[k][0]*xdata[i][1]-Adiag[k][1]*xdata[i][0]);
                    ydata[i][1] += alpha[0]*(Adiag[k][0]*xdata[i][1]-Adiag[k][1]*xdata[i][0]) + alpha[1]*(Adiag[k][0]*xdata[i][0]+Adiag[k][1]*xdata[i][1]);
                }
                if (num_flops) *num_flops += 40*A->size + 20*A->diag.base.num_nonzeros;
            } else if (precision == mtx_double) {
                const double (* Adata)[2] = a->base.data.complex_double;
                const double (* xdata)[2] = xbase->data.complex_double;
                double (* ydata)[2] = ybase->data.complex_double;
                for (int i = 0; i < A->num_rows; i++) {
                    for (int k = rowptr[i]; k < rowptr[i+1]; k++) {
                        ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]) - alpha[1]*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]);
                        ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]) + alpha[1]*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]);
                        ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                    }
                }
                const double (* Adiag)[2] = A->diag.base.data.complex_double;
                for (int k = 0; k < A->diag.base.num_nonzeros; k++) {
                    int i = A->colidx[A->diag.base.idx[k]];
                        ydata[i][0] += alpha[0]*(Adiag[k][0]*xdata[i][0]+Adiag[k][1]*xdata[i][1]) - alpha[1]*(Adiag[k][0]*xdata[i][1]-Adiag[k][1]*xdata[i][0]);
                        ydata[i][1] += alpha[0]*(Adiag[k][0]*xdata[i][1]-Adiag[k][1]*xdata[i][0]) + alpha[1]*(Adiag[k][0]*xdata[i][0]+Adiag[k][1]*xdata[i][1]);
                }
                if (num_flops) *num_flops += 40*A->size + 20*A->diag.base.num_nonzeros;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else { return MTX_ERR_INVALID_TRANSPOSITION; }
        err = mtxompvector_ussc((struct mtxompvector *) &A->a, &A->diag);
        if (err) return err;
    } else if (A->num_rows == A->num_columns && A->symmetry == mtx_hermitian) {
        int err = mtxompvector_usgz(
            (struct mtxompvector *) &A->diag,
            (struct mtxompvector *) &A->a);
        if (err) return err;
        if (trans == mtx_notrans || trans == mtx_conjtrans) {
            if (precision == mtx_single) {
                const float (* Adata)[2] = a->base.data.complex_single;
                const float (* xdata)[2] = xbase->data.complex_single;
                float (* ydata)[2] = ybase->data.complex_single;
                for (int i = 0; i < A->num_rows; i++) {
                    for (int k = rowptr[i]; k < rowptr[i+1]; k++) {
                        ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]) - alpha[1]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                        ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]) + alpha[1]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                        ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                    }
                }
                const float (* Adiag)[2] = A->diag.base.data.complex_single;
                for (int k = 0; k < A->diag.base.num_nonzeros; k++) {
                    int i = A->colidx[A->diag.base.idx[k]];
                    ydata[i][0] += alpha[0]*(Adiag[k][0]*xdata[i][0]+Adiag[k][1]*xdata[i][1]) - alpha[1]*(Adiag[k][0]*xdata[i][1]-Adiag[k][1]*xdata[i][0]);
                    ydata[i][1] += alpha[0]*(Adiag[k][0]*xdata[i][1]-Adiag[k][1]*xdata[i][0]) + alpha[1]*(Adiag[k][0]*xdata[i][0]+Adiag[k][1]*xdata[i][1]);
                }
                if (num_flops) *num_flops += 40*A->size + 20*A->diag.base.num_nonzeros;
            } else if (precision == mtx_double) {
                const double (* Adata)[2] = a->base.data.complex_double;
                const double (* xdata)[2] = xbase->data.complex_double;
                double (* ydata)[2] = ybase->data.complex_double;
                for (int i = 0; i < A->num_rows; i++) {
                    for (int k = rowptr[i]; k < rowptr[i+1]; k++) {
                        ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]) - alpha[1]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                        ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]) + alpha[1]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                        ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                    }
                }
                const double (* Adiag)[2] = A->diag.base.data.complex_double;
                for (int k = 0; k < A->diag.base.num_nonzeros; k++) {
                    int i = A->colidx[A->diag.base.idx[k]];
                    ydata[i][0] += alpha[0]*(Adiag[k][0]*xdata[i][0]+Adiag[k][1]*xdata[i][1]) - alpha[1]*(Adiag[k][0]*xdata[i][1]-Adiag[k][1]*xdata[i][0]);
                    ydata[i][1] += alpha[0]*(Adiag[k][0]*xdata[i][1]-Adiag[k][1]*xdata[i][0]) + alpha[1]*(Adiag[k][0]*xdata[i][0]+Adiag[k][1]*xdata[i][1]);
                }
                if (num_flops) *num_flops += 40*A->size + 20*A->diag.base.num_nonzeros;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (trans == mtx_trans) {
            if (precision == mtx_single) {
                const float (* Adata)[2] = a->base.data.complex_single;
                const float (* xdata)[2] = xbase->data.complex_single;
                float (* ydata)[2] = ybase->data.complex_single;
                for (int i = 0; i < A->num_rows; i++) {
                    for (int k = rowptr[i]; k < rowptr[i+1]; k++) {
                        ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]) - alpha[1]*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]);
                        ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]) + alpha[1]*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]);
                        ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                    }
                }
                const float (* Adiag)[2] = A->diag.base.data.complex_single;
                for (int k = 0; k < A->diag.base.num_nonzeros; k++) {
                    int i = A->colidx[A->diag.base.idx[k]];
                    ydata[i][0] += alpha[0]*(Adiag[k][0]*xdata[i][0]-Adiag[k][1]*xdata[i][1]) - alpha[1]*(Adiag[k][0]*xdata[i][1]+Adiag[k][1]*xdata[i][0]);
                    ydata[i][1] += alpha[0]*(Adiag[k][0]*xdata[i][1]+Adiag[k][1]*xdata[i][0]) + alpha[1]*(Adiag[k][0]*xdata[i][0]-Adiag[k][1]*xdata[i][1]);
                }
                if (num_flops) *num_flops += 40*A->size + 20*A->diag.base.num_nonzeros;
            } else if (precision == mtx_double) {
                const double (* Adata)[2] = a->base.data.complex_double;
                const double (* xdata)[2] = xbase->data.complex_double;
                double (* ydata)[2] = ybase->data.complex_double;
                for (int i = 0; i < A->num_rows; i++) {
                    for (int k = rowptr[i]; k < rowptr[i+1]; k++) {
                        ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]) - alpha[1]*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]);
                        ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]) + alpha[1]*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]);
                        ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                    }
                }
                const double (* Adiag)[2] = A->diag.base.data.complex_double;
                for (int k = 0; k < A->diag.base.num_nonzeros; k++) {
                    int i = A->colidx[A->diag.base.idx[k]];
                    ydata[i][0] += alpha[0]*(Adiag[k][0]*xdata[i][0]-Adiag[k][1]*xdata[i][1]) - alpha[1]*(Adiag[k][0]*xdata[i][1]+Adiag[k][1]*xdata[i][0]);
                    ydata[i][1] += alpha[0]*(Adiag[k][0]*xdata[i][1]+Adiag[k][1]*xdata[i][0]) + alpha[1]*(Adiag[k][0]*xdata[i][0]-Adiag[k][1]*xdata[i][1]);
                }
                if (num_flops) *num_flops += 40*A->size + 20*A->diag.base.num_nonzeros;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else { return MTX_ERR_INVALID_TRANSPOSITION; }
        err = mtxompvector_ussc((struct mtxompvector *) &A->a, &A->diag);
        if (err) return err;
    } else { return MTX_ERR_INVALID_SYMMETRY; }
    return MTX_SUCCESS;
}
