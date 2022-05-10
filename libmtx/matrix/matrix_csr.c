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
 * Last modified: 2022-05-05
 *
 * Data structures for matrices in coordinate format.
 */

#include <libmtx/libmtx-config.h>

#include <libmtx/error.h>
#include <libmtx/field.h>
#include <libmtx/matrix/matrix.h>
#include <libmtx/matrix/matrix_csr.h>
#include <libmtx/mtxfile/mtxfile.h>
#include <libmtx/precision.h>
#include <libmtx/util/sort.h>
#include <libmtx/vector/base.h>
#include <libmtx/vector/vector.h>

#include <errno.h>

#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * memory management
 */

/**
 * ‘mtxmatrix_csr_free()’ frees storage allocated for a matrix.
 */
void mtxmatrix_csr_free(
    struct mtxmatrix_csr * A)
{
    mtxvector_packed_free(&A->diag);
    mtxvector_base_free(&A->a);
    free(A->colidx);
    free(A->rowptr);
}

/**
 * ‘mtxmatrix_csr_alloc_copy()’ allocates a copy of a matrix without
 * initialising the values.
 */
int mtxmatrix_csr_alloc_copy(
    struct mtxmatrix_csr * dst,
    const struct mtxmatrix_csr * src)
{
    /* return mtxmatrix_csr_alloc_rows( */
    /*     dst, src->a.field, src->a.precision, src->symmetry, */
    /*     src->num_rows, src->num_columns, src->rowptr, src->colidx); */
}

/**
 * ‘mtxmatrix_csr_init_copy()’ allocates a copy of a matrix and also
 * copies the values.
 */
int mtxmatrix_csr_init_copy(
    struct mtxmatrix_csr * dst,
    const struct mtxmatrix_csr * src)
{
    int err = mtxmatrix_csr_alloc_copy(dst, src);
    if (err) return err;
    return mtxmatrix_csr_copy(dst, src);
}

/*
 * initialise matrices from entrywise data in coordinate format
 */

/**
 * ‘mtxmatrix_csr_alloc_entries()’ allocates a matrix from entrywise
 * data in coordinate format.
 *
 * If it is not ‘NULL’, then ‘perm’ must point to an array of length
 * ‘size’. Because the sparse matrix storage may internally reorder
 * the specified nonzero entries, this array is used to store the
 * permutation applied to the specified nonzero entries.
 */
int mtxmatrix_csr_alloc_entries(
    struct mtxmatrix_csr * A,
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

    int err = mtxvector_base_alloc(&A->a, field, precision, size);
    if (err) { free(A->colidx); free(A->rowptr); return err; }

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
        err = mtxvector_packed_alloc(
            &A->diag, mtxvector_base, field, precision,
            A->size, num_diagonals, NULL);
        if (err) {
            mtxvector_base_free(&A->a);
            free(A->colidx); free(A->rowptr);
            return err;
        }
        int64_t * diagidx = A->diag.idx;
        num_diagonals = 0;
        for (int i = 0; i < num_rows; i++) {
            for (int64_t k = A->rowptr[i]; k < A->rowptr[i+1]; k++) {
                if (i == A->colidx[k]) diagidx[num_diagonals++] = k;
            }
        }
    } else if (symmetry == mtx_unsymmetric) {
        A->num_nonzeros = A->size;
        err = mtxvector_packed_alloc(
            &A->diag, mtxvector_base, field, precision, A->size, 0, NULL);
        if (err) {
            mtxvector_base_free(&A->a);
            free(A->colidx); free(A->rowptr);
            return err;
        }
    } else { return MTX_ERR_INVALID_SYMMETRY; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_csr_init_entries_real_single()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * real, single precision coefficients.
 */
int mtxmatrix_csr_init_entries_real_single(
    struct mtxmatrix_csr * A,
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
    int err = mtxmatrix_csr_alloc_entries(
        A, mtx_field_real, mtx_single, symmetry, num_rows, num_columns,
        size, sizeof(*rowidx), 0, rowidx, colidx, perm);
    if (err) { free(perm); return err; }
    struct mtxvector_packed x;
    x.size = A->size;
    x.num_nonzeros = size;
    x.idx = perm;
    x.x.type = mtxvector_base;
    x.x.storage.base.field = mtx_field_real;
    x.x.storage.base.precision = mtx_single;
    x.x.storage.base.size = size;
    x.x.storage.base.data.real_single = (float *) data;
    err = mtxvector_base_ussc(&A->a, &x);
    if (err) { mtxmatrix_csr_free(A); free(perm); return err; }
    free(perm);
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_csr_init_entries_real_double()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * real, double precision coefficients.
 */
int mtxmatrix_csr_init_entries_real_double(
    struct mtxmatrix_csr * A,
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
    int err = mtxmatrix_csr_alloc_entries(
        A, mtx_field_real, mtx_double, symmetry, num_rows, num_columns,
        size, sizeof(*rowidx), 0, rowidx, colidx, perm);
    if (err) { free(perm); return err; }
    struct mtxvector_packed x;
    x.size = A->size;
    x.num_nonzeros = size;
    x.idx = perm;
    x.x.type = mtxvector_base;
    x.x.storage.base.field = mtx_field_real;
    x.x.storage.base.precision = mtx_double;
    x.x.storage.base.size = size;
    x.x.storage.base.data.real_double = (double *) data;
    err = mtxvector_base_ussc(&A->a, &x);
    if (err) { mtxmatrix_csr_free(A); free(perm); return err; }
    free(perm);
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_csr_init_entries_complex_single()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * complex, single precision coefficients.
 */
int mtxmatrix_csr_init_entries_complex_single(
    struct mtxmatrix_csr * A,
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
    int err = mtxmatrix_csr_alloc_entries(
        A, mtx_field_complex, mtx_single, symmetry, num_rows, num_columns,
        size, sizeof(*rowidx), 0, rowidx, colidx, perm);
    if (err) { free(perm); return err; }
    struct mtxvector_packed x;
    x.size = A->size;
    x.num_nonzeros = size;
    x.idx = perm;
    x.x.type = mtxvector_base;
    x.x.storage.base.field = mtx_field_complex;
    x.x.storage.base.precision = mtx_single;
    x.x.storage.base.size = size;
    x.x.storage.base.data.complex_single = (float (*)[2]) data;
    err = mtxvector_base_ussc(&A->a, &x);
    if (err) { mtxmatrix_csr_free(A); free(perm); return err; }
    free(perm);
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_csr_init_entries_complex_double()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * complex, double precision coefficients.
 */
int mtxmatrix_csr_init_entries_complex_double(
    struct mtxmatrix_csr * A,
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
    int err = mtxmatrix_csr_alloc_entries(
        A, mtx_field_complex, mtx_double, symmetry, num_rows, num_columns,
        size, sizeof(*rowidx), 0, rowidx, colidx, perm);
    if (err) { free(perm); return err; }
    struct mtxvector_packed x;
    x.size = A->size;
    x.num_nonzeros = size;
    x.idx = perm;
    x.x.type = mtxvector_base;
    x.x.storage.base.field = mtx_field_complex;
    x.x.storage.base.precision = mtx_double;
    x.x.storage.base.size = size;
    x.x.storage.base.data.complex_double = (double (*)[2]) data;
    err = mtxvector_base_ussc(&A->a, &x);
    if (err) { mtxmatrix_csr_free(A); free(perm); return err; }
    free(perm);
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_csr_init_entries_integer_single()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * integer, single precision coefficients.
 */
int mtxmatrix_csr_init_entries_integer_single(
    struct mtxmatrix_csr * A,
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
    int err = mtxmatrix_csr_alloc_entries(
        A, mtx_field_integer, mtx_single, symmetry, num_rows, num_columns,
        size, sizeof(*rowidx), 0, rowidx, colidx, perm);
    if (err) { free(perm); return err; }
    struct mtxvector_packed x;
    x.size = A->size;
    x.num_nonzeros = size;
    x.idx = perm;
    x.x.type = mtxvector_base;
    x.x.storage.base.field = mtx_field_integer;
    x.x.storage.base.precision = mtx_single;
    x.x.storage.base.size = size;
    x.x.storage.base.data.integer_single = (int32_t *) data;
    err = mtxvector_base_ussc(&A->a, &x);
    if (err) { mtxmatrix_csr_free(A); free(perm); return err; }
    free(perm);
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_csr_init_entries_integer_double()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * integer, double precision coefficients.
 */
int mtxmatrix_csr_init_entries_integer_double(
    struct mtxmatrix_csr * A,
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
    int err = mtxmatrix_csr_alloc_entries(
        A, mtx_field_integer, mtx_double, symmetry, num_rows, num_columns,
        size, sizeof(*rowidx), 0, rowidx, colidx, perm);
    if (err) { free(perm); return err; }
    struct mtxvector_packed x;
    x.size = A->size;
    x.num_nonzeros = size;
    x.idx = perm;
    x.x.type = mtxvector_base;
    x.x.storage.base.field = mtx_field_integer;
    x.x.storage.base.precision = mtx_double;
    x.x.storage.base.size = size;
    x.x.storage.base.data.integer_double = (int64_t *) data;
    err = mtxvector_base_ussc(&A->a, &x);
    if (err) { mtxmatrix_csr_free(A); free(perm); return err; }
    free(perm);
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_csr_init_entries_pattern()’ allocates and initialises a
 * matrix from entrywise data in coordinate format with boolean
 * coefficients.
 */
int mtxmatrix_csr_init_entries_pattern(
    struct mtxmatrix_csr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx)
{
    return mtxmatrix_csr_alloc_entries(
        A, mtx_field_pattern, mtx_single, symmetry, num_rows, num_columns,
        size, sizeof(*rowidx), 0, rowidx, colidx, NULL);
}

/*
 * initialise matrices from entrywise data in coordinate format with
 * specified strides
 */

/**
 * ‘mtxmatrix_csr_init_entries_strided_real_single()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * real, single precision coefficients.
 */
int mtxmatrix_csr_init_entries_strided_real_single(
    struct mtxmatrix_csr * A,
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
 * ‘mtxmatrix_csr_init_entries_strided_real_double()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * real, double precision coefficients.
 */
int mtxmatrix_csr_init_entries_strided_real_double(
    struct mtxmatrix_csr * A,
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
 * ‘mtxmatrix_csr_init_entries_strided_complex_single()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * complex, single precision coefficients.
 */
int mtxmatrix_csr_init_entries_strided_complex_single(
    struct mtxmatrix_csr * A,
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
 * ‘mtxmatrix_csr_init_entries_strided_complex_double()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * complex, double precision coefficients.
 */
int mtxmatrix_csr_init_entries_strided_complex_double(
    struct mtxmatrix_csr * A,
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
 * ‘mtxmatrix_csr_init_entries_strided_integer_single()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * integer, single precision coefficients.
 */
int mtxmatrix_csr_init_entries_strided_integer_single(
    struct mtxmatrix_csr * A,
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
 * ‘mtxmatrix_csr_init_entries_strided_integer_double()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * integer, double precision coefficients.
 */
int mtxmatrix_csr_init_entries_strided_integer_double(
    struct mtxmatrix_csr * A,
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
 * ‘mtxmatrix_csr_init_entries_strided_pattern()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * boolean coefficients.
 */
int mtxmatrix_csr_init_entries_strided_pattern(
    struct mtxmatrix_csr * A,
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
 * ‘mtxmatrix_csr_alloc_rows()’ allocates a matrix from row-wise data
 * in compressed row format.
 */
int mtxmatrix_csr_alloc_rows(
    struct mtxmatrix_csr * A,
    enum mtxfield field,
    enum mtxprecision precision,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx);

/**
 * ‘mtxmatrix_csr_init_rows_real_single()’ allocates and initialises a
 * matrix from row-wise data in compressed row format with real,
 * single precision coefficients.
 */
int mtxmatrix_csr_init_rows_real_single(
    struct mtxmatrix_csr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const float * data);

/**
 * ‘mtxmatrix_csr_init_rows_real_double()’ allocates and initialises a
 * matrix from row-wise data in compressed row format with real,
 * double precision coefficients.
 */
int mtxmatrix_csr_init_rows_real_double(
    struct mtxmatrix_csr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const double * data);

/**
 * ‘mtxmatrix_csr_init_rows_complex_single()’ allocates and
 * initialises a matrix from row-wise data in compressed row format
 * with complex, single precision coefficients.
 */
int mtxmatrix_csr_init_rows_complex_single(
    struct mtxmatrix_csr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const float (* data)[2]);

/**
 * ‘mtxmatrix_csr_init_rows_complex_double()’ allocates and
 * initialises a matrix from row-wise data in compressed row format
 * with complex, double precision coefficients.
 */
int mtxmatrix_csr_init_rows_complex_double(
    struct mtxmatrix_csr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const double (* data)[2]);

/**
 * ‘mtxmatrix_csr_init_rows_integer_single()’ allocates and
 * initialises a matrix from row-wise data in compressed row format
 * with integer, single precision coefficients.
 */
int mtxmatrix_csr_init_rows_integer_single(
    struct mtxmatrix_csr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const int32_t * data);

/**
 * ‘mtxmatrix_csr_init_rows_integer_double()’ allocates and
 * initialises a matrix from row-wise data in compressed row format
 * with integer, double precision coefficients.
 */
int mtxmatrix_csr_init_rows_integer_double(
    struct mtxmatrix_csr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const int64_t * data);

/**
 * ‘mtxmatrix_csr_init_rows_pattern()’ allocates and initialises a
 * matrix from row-wise data in compressed row format with boolean
 * coefficients.
 */
int mtxmatrix_csr_init_rows_pattern(
    struct mtxmatrix_csr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx);

/*
 * initialise matrices from column-wise data in compressed column
 * format
 */

/**
 * ‘mtxmatrix_csr_alloc_columns()’ allocates a matrix from column-wise
 * data in compressed column format.
 */
int mtxmatrix_csr_alloc_columns(
    struct mtxmatrix_csr * A,
    enum mtxfield field,
    enum mtxprecision precision,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx);

/**
 * ‘mtxmatrix_csr_init_columns_real_single()’ allocates and
 * initialises a matrix from column-wise data in compressed column
 * format with real, single precision coefficients.
 */
int mtxmatrix_csr_init_columns_real_single(
    struct mtxmatrix_csr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx,
    const float * data);

/**
 * ‘mtxmatrix_csr_init_columns_real_double()’ allocates and
 * initialises a matrix from column-wise data in compressed column
 * format with real, double precision coefficients.
 */
int mtxmatrix_csr_init_columns_real_double(
    struct mtxmatrix_csr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx,
    const double * data);

/**
 * ‘mtxmatrix_csr_init_columns_complex_single()’ allocates and
 * initialises a matrix from column-wise data in compressed column
 * format with complex, single precision coefficients.
 */
int mtxmatrix_csr_init_columns_complex_single(
    struct mtxmatrix_csr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx,
    const float (* data)[2]);

/**
 * ‘mtxmatrix_csr_init_columns_complex_double()’ allocates and
 * initialises a matrix from column-wise data in compressed column
 * format with complex, double precision coefficients.
 */
int mtxmatrix_csr_init_columns_complex_double(
    struct mtxmatrix_csr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx,
    const double (* data)[2]);

/**
 * ‘mtxmatrix_csr_init_columns_integer_single()’ allocates and
 * initialises a matrix from column-wise data in compressed column
 * format with integer, single precision coefficients.
 */
int mtxmatrix_csr_init_columns_integer_single(
    struct mtxmatrix_csr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx,
    const int32_t * data);

/**
 * ‘mtxmatrix_csr_init_columns_integer_double()’ allocates and
 * initialises a matrix from column-wise data in compressed column
 * format with integer, double precision coefficients.
 */
int mtxmatrix_csr_init_columns_integer_double(
    struct mtxmatrix_csr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx,
    const int64_t * data);

/**
 * ‘mtxmatrix_csr_init_columns_pattern()’ allocates and initialises a
 * matrix from column-wise data in compressed column format with
 * boolean coefficients.
 */
int mtxmatrix_csr_init_columns_pattern(
    struct mtxmatrix_csr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx);

/*
 * initialise matrices from a list of dense cliques
 */

/**
 * ‘mtxmatrix_csr_alloc_cliques()’ allocates a matrix from a list of
 * cliques.
 */
int mtxmatrix_csr_alloc_cliques(
    struct mtxmatrix_csr * A,
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
 * ‘mtxmatrix_csr_init_cliques_real_single()’ allocates and
 * initialises a matrix from a list of cliques with real, single
 * precision coefficients.
 */
int mtxmatrix_csr_init_cliques_real_single(
    struct mtxmatrix_csr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_cliques,
    const int64_t * cliqueptr,
    const int * rowidx,
    const int * colidx,
    const float * data);

/**
 * ‘mtxmatrix_csr_init_cliques_real_double()’ allocates and
 * initialises a matrix from a list of cliques with real, double
 * precision coefficients.
 */
int mtxmatrix_csr_init_cliques_real_double(
    struct mtxmatrix_csr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_cliques,
    const int64_t * cliqueptr,
    const int * rowidx,
    const int * colidx,
    const double * data);

/**
 * ‘mtxmatrix_csr_init_cliques_complex_single()’ allocates and
 * initialises a matrix from a list of cliques with complex, single
 * precision coefficients.
 */
int mtxmatrix_csr_init_cliques_complex_single(
    struct mtxmatrix_csr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_cliques,
    const int64_t * cliqueptr,
    const int * rowidx,
    const int * colidx,
    const float (* data)[2]);

/**
 * ‘mtxmatrix_csr_init_cliques_complex_double()’ allocates and
 * initialises a matrix from a list of cliques with complex, double
 * precision coefficients.
 */
int mtxmatrix_csr_init_cliques_complex_double(
    struct mtxmatrix_csr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_cliques,
    const int64_t * cliqueptr,
    const int * rowidx,
    const int * colidx,
    const double (* data)[2]);

/**
 * ‘mtxmatrix_csr_init_cliques_integer_single()’ allocates and
 * initialises a matrix from a list of cliques with integer, single
 * precision coefficients.
 */
int mtxmatrix_csr_init_cliques_integer_single(
    struct mtxmatrix_csr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_cliques,
    const int64_t * cliqueptr,
    const int * rowidx,
    const int * colidx,
    const int32_t * data);

/**
 * ‘mtxmatrix_csr_init_cliques_integer_double()’ allocates and
 * initialises a matrix from a list of cliques with integer, double
 * precision coefficients.
 */
int mtxmatrix_csr_init_cliques_integer_double(
    struct mtxmatrix_csr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_cliques,
    const int64_t * cliqueptr,
    const int * rowidx,
    const int * colidx,
    const int64_t * data);

/**
 * ‘mtxmatrix_csr_init_cliques_pattern()’ allocates and initialises a
 * matrix from a list of cliques with boolean coefficients.
 */
int mtxmatrix_csr_init_cliques_pattern(
    struct mtxmatrix_csr * A,
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
 * ‘mtxmatrix_csr_setzero()’ sets every value of a matrix to zero.
 */
int mtxmatrix_csr_setzero(
    struct mtxmatrix_csr * A)
{
    return mtxvector_base_setzero(&A->a);
}

/**
 * ‘mtxmatrix_csr_set_real_single()’ sets values of a matrix based on
 * an array of single precision floating point numbers.
 */
int mtxmatrix_csr_set_real_single(
    struct mtxmatrix_csr * A,
    int64_t size,
    int stride,
    const float * a)
{
    return mtxvector_base_set_real_single(&A->a, size, stride, a);
}

/**
 * ‘mtxmatrix_csr_set_real_double()’ sets values of a matrix based on
 * an array of double precision floating point numbers.
 */
int mtxmatrix_csr_set_real_double(
    struct mtxmatrix_csr * A,
    int64_t size,
    int stride,
    const double * a)
{
    return mtxvector_base_set_real_double(&A->a, size, stride, a);
}

/**
 * ‘mtxmatrix_csr_set_complex_single()’ sets values of a matrix based
 * on an array of single precision floating point complex numbers.
 */
int mtxmatrix_csr_set_complex_single(
    struct mtxmatrix_csr * A,
    int64_t size,
    int stride,
    const float (*a)[2])
{
    return mtxvector_base_set_complex_single(&A->a, size, stride, a);
}

/**
 * ‘mtxmatrix_csr_set_complex_double()’ sets values of a matrix based
 * on an array of double precision floating point complex numbers.
 */
int mtxmatrix_csr_set_complex_double(
    struct mtxmatrix_csr * A,
    int64_t size,
    int stride,
    const double (*a)[2])
{
    return mtxvector_base_set_complex_double(&A->a, size, stride, a);
}

/**
 * ‘mtxmatrix_csr_set_integer_single()’ sets values of a matrix based
 * on an array of integers.
 */
int mtxmatrix_csr_set_integer_single(
    struct mtxmatrix_csr * A,
    int64_t size,
    int stride,
    const int32_t * a)
{
    return mtxvector_base_set_integer_single(&A->a, size, stride, a);
}

/**
 * ‘mtxmatrix_csr_set_integer_double()’ sets values of a matrix based
 * on an array of integers.
 */
int mtxmatrix_csr_set_integer_double(
    struct mtxmatrix_csr * A,
    int64_t size,
    int stride,
    const int64_t * a)
{
    return mtxvector_base_set_integer_double(&A->a, size, stride, a);
}

/*
 * row and column vectors
 */

/**
 * ‘mtxmatrix_csr_alloc_row_vector()’ allocates a row vector for a
 * given matrix, where a row vector is a vector whose length equal to
 * a single row of the matrix.
 */
int mtxmatrix_csr_alloc_row_vector(
    const struct mtxmatrix_csr * A,
    struct mtxvector * x,
    enum mtxvectortype vectortype)
{
    return mtxvector_alloc(
        x, vectortype, A->a.field, A->a.precision, A->num_columns);
}

/**
 * ‘mtxmatrix_csr_alloc_column_vector()’ allocates a column vector for
 * a given matrix, where a column vector is a vector whose length
 * equal to a single column of the matrix.
 */
int mtxmatrix_csr_alloc_column_vector(
    const struct mtxmatrix_csr * A,
    struct mtxvector * y,
    enum mtxvectortype vectortype)
{
    return mtxvector_alloc(
        y, vectortype, A->a.field, A->a.precision, A->num_rows);
}

/*
 * convert to and from Matrix Market format
 */

/**
 * ‘mtxmatrix_csr_from_mtxfile()’ converts a matrix from Matrix Market
 * format.
 */
int mtxmatrix_csr_from_mtxfile(
    struct mtxmatrix_csr * A,
    const struct mtxfile * mtxfile)
{
    struct mtxmatrix_coordinate coo;
    int err = mtxmatrix_coordinate_from_mtxfile(&coo, mtxfile);
    if (err) return err;
    if (mtxfile->header.field == mtxfile_real) {
        if (mtxfile->precision == mtx_single) {
            err = mtxmatrix_csr_init_entries_real_single(
                A, coo.symmetry, coo.num_rows, coo.num_columns,
                coo.size, coo.rowidx, coo.colidx, coo.a.data.real_single);
        } else if (mtxfile->precision == mtx_double) {
            err = mtxmatrix_csr_init_entries_real_double(
                A, coo.symmetry, coo.num_rows, coo.num_columns,
                coo.size, coo.rowidx, coo.colidx, coo.a.data.real_double);
        } else { mtxmatrix_coordinate_free(&coo); return MTX_ERR_INVALID_PRECISION; }
    } else if (mtxfile->header.field == mtxfile_complex) {
        if (mtxfile->precision == mtx_single) {
            err = mtxmatrix_csr_init_entries_complex_single(
                A, coo.symmetry, coo.num_rows, coo.num_columns,
                coo.size, coo.rowidx, coo.colidx, coo.a.data.complex_single);
        } else if (mtxfile->precision == mtx_double) {
            err = mtxmatrix_csr_init_entries_complex_double(
                A, coo.symmetry, coo.num_rows, coo.num_columns,
                coo.size, coo.rowidx, coo.colidx, coo.a.data.complex_double);
        } else { mtxmatrix_coordinate_free(&coo); return MTX_ERR_INVALID_PRECISION; }
    } else if (mtxfile->header.field == mtxfile_integer) {
        if (mtxfile->precision == mtx_single) {
            err = mtxmatrix_csr_init_entries_integer_single(
                A, coo.symmetry, coo.num_rows, coo.num_columns,
                coo.size, coo.rowidx, coo.colidx, coo.a.data.integer_single);
        } else if (mtxfile->precision == mtx_double) {
            err = mtxmatrix_csr_init_entries_integer_double(
                A, coo.symmetry, coo.num_rows, coo.num_columns,
                coo.size, coo.rowidx, coo.colidx, coo.a.data.integer_double);
        } else { mtxmatrix_coordinate_free(&coo); return MTX_ERR_INVALID_PRECISION; }
    } else if (mtxfile->header.field == mtxfile_pattern) {
        err = mtxmatrix_csr_init_entries_pattern(
            A, coo.symmetry, coo.num_rows, coo.num_columns,
            coo.size, coo.rowidx, coo.colidx);
    } else { mtxmatrix_coordinate_free(&coo); return MTX_ERR_INVALID_MTX_FIELD; }
    if (err) { mtxmatrix_coordinate_free(&coo); return err; }
    mtxmatrix_coordinate_free(&coo);
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_csr_to_mtxfile()’ converts a matrix to Matrix Market
 * format.
 */
int mtxmatrix_csr_to_mtxfile(
    struct mtxfile * mtxfile,
    const struct mtxmatrix_csr * A,
    int64_t num_rows,
    const int64_t * rowidx,
    int64_t num_columns,
    const int64_t * colidx,
    enum mtxfileformat mtxfmt)
{
    int err;
    if (mtxfmt != mtxfile_coordinate)
        return MTX_ERR_INCOMPATIBLE_MTX_FORMAT;

    enum mtxfilesymmetry symmetry;
    err = mtxfilesymmetry_from_mtxsymmetry(&symmetry, A->symmetry);
    if (err) return err;

    if (A->a.field == mtx_field_real) {
        err = mtxfile_alloc_matrix_coordinate(
            mtxfile, mtxfile_real, symmetry, A->a.precision,
            rowidx ? num_rows : A->num_rows,
            colidx ? num_columns : A->num_columns,
            A->size);
        if (err) return err;
        if (A->a.precision == mtx_single) {
            struct mtxfile_matrix_coordinate_real_single * data =
                mtxfile->data.matrix_coordinate_real_single;
            for (int i = 0; i < A->num_rows; i++) {
                for (int64_t k = A->rowptr[i]; k < A->rowptr[i+1]; k++) {
                    data[k].i = rowidx ? rowidx[i]+1 : i+1;
                    data[k].j = colidx ? colidx[A->colidx[k]]+1 : A->colidx[k]+1;
                    data[k].a = A->a.data.real_single[k];
                }
            }
        } else if (A->a.precision == mtx_double) {
            struct mtxfile_matrix_coordinate_real_double * data =
                mtxfile->data.matrix_coordinate_real_double;
            for (int i = 0; i < A->num_rows; i++) {
                for (int64_t k = A->rowptr[i]; k < A->rowptr[i+1]; k++) {
                    data[k].i = rowidx ? rowidx[i]+1 : i+1;
                    data[k].j = colidx ? colidx[A->colidx[k]]+1 : A->colidx[k]+1;
                    data[k].a = A->a.data.real_double[k];
                }
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (A->a.field == mtx_field_complex) {
        err = mtxfile_alloc_matrix_coordinate(
            mtxfile, mtxfile_complex, symmetry, A->a.precision,
            A->num_rows, A->num_columns, A->size);
        if (err) return err;
        if (A->a.precision == mtx_single) {
            struct mtxfile_matrix_coordinate_complex_single * data =
                mtxfile->data.matrix_coordinate_complex_single;
            for (int i = 0; i < A->num_rows; i++) {
                for (int64_t k = A->rowptr[i]; k < A->rowptr[i+1]; k++) {
                    data[k].i = rowidx ? rowidx[i]+1 : i+1;
                    data[k].j = colidx ? colidx[A->colidx[k]]+1 : A->colidx[k]+1;
                    data[k].a[0] = A->a.data.complex_single[k][0];
                    data[k].a[1] = A->a.data.complex_single[k][1];
                }
            }
        } else if (A->a.precision == mtx_double) {
            struct mtxfile_matrix_coordinate_complex_double * data =
                mtxfile->data.matrix_coordinate_complex_double;
            for (int i = 0; i < A->num_rows; i++) {
                for (int64_t k = A->rowptr[i]; k < A->rowptr[i+1]; k++) {
                    data[k].i = rowidx ? rowidx[i]+1 : i+1;
                    data[k].j = colidx ? colidx[A->colidx[k]]+1 : A->colidx[k]+1;
                    data[k].a[0] = A->a.data.complex_double[k][0];
                    data[k].a[1] = A->a.data.complex_double[k][1];
                }
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (A->a.field == mtx_field_integer) {
        err = mtxfile_alloc_matrix_coordinate(
            mtxfile, mtxfile_integer, symmetry, A->a.precision,
            A->num_rows, A->num_columns, A->size);
        if (err) return err;
        if (A->a.precision == mtx_single) {
            struct mtxfile_matrix_coordinate_integer_single * data =
                mtxfile->data.matrix_coordinate_integer_single;
            for (int i = 0; i < A->num_rows; i++) {
                for (int64_t k = A->rowptr[i]; k < A->rowptr[i+1]; k++) {
                    data[k].i = rowidx ? rowidx[i]+1 : i+1;
                    data[k].j = colidx ? colidx[A->colidx[k]]+1 : A->colidx[k]+1;
                    data[k].a = A->a.data.integer_single[k];
                }
            }
        } else if (A->a.precision == mtx_double) {
            struct mtxfile_matrix_coordinate_integer_double * data =
                mtxfile->data.matrix_coordinate_integer_double;
            for (int i = 0; i < A->num_rows; i++) {
                for (int64_t k = A->rowptr[i]; k < A->rowptr[i+1]; k++) {
                    data[k].i = rowidx ? rowidx[i]+1 : i+1;
                    data[k].j = colidx ? colidx[A->colidx[k]]+1 : A->colidx[k]+1;
                    data[k].a = A->a.data.integer_double[k];
                }
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (A->a.field == mtx_field_pattern) {
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
 * Level 1 BLAS operations
 */

/**
 * ‘mtxmatrix_csr_swap()’ swaps values of two matrices, simultaneously
 * performing ‘y <- x’ and ‘x <- y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_csr_swap(
    struct mtxmatrix_csr * x,
    struct mtxmatrix_csr * y)
{
    return mtxvector_base_swap(&x->a, &y->a);
}

/**
 * ‘mtxmatrix_csr_copy()’ copies values of a matrix, ‘y = x’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_csr_copy(
    struct mtxmatrix_csr * y,
    const struct mtxmatrix_csr * x)
{
    return mtxvector_base_copy(&y->a, &x->a);
}

/**
 * ‘mtxmatrix_csr_sscal()’ scales a matrix by a single precision
 * floating point scalar, ‘x = a*x’.
 */
int mtxmatrix_csr_sscal(
    float a,
    struct mtxmatrix_csr * x,
    int64_t * num_flops)
{
    return mtxvector_base_sscal(a, &x->a, num_flops);
}

/**
 * ‘mtxmatrix_csr_dscal()’ scales a matrix by a double precision
 * floating point scalar, ‘x = a*x’.
 */
int mtxmatrix_csr_dscal(
    double a,
    struct mtxmatrix_csr * x,
    int64_t * num_flops)
{
    return mtxvector_base_dscal(a, &x->a, num_flops);
}

/**
 * ‘mtxmatrix_csr_cscal()’ scales a matrix by a complex, single
 * precision floating point scalar, ‘x = (a+b*i)*x’.
 */
int mtxmatrix_csr_cscal(
    float a[2],
    struct mtxmatrix_csr * x,
    int64_t * num_flops)
{
    return mtxvector_base_cscal(a, &x->a, num_flops);
}

/**
 * ‘mtxmatrix_csr_zscal()’ scales a matrix by a complex, double
 * precision floating point scalar, ‘x = (a+b*i)*x’.
 */
int mtxmatrix_csr_zscal(
    double a[2],
    struct mtxmatrix_csr * x,
    int64_t * num_flops)
{
    return mtxvector_base_zscal(a, &x->a, num_flops);
}

/**
 * ‘mtxmatrix_csr_saxpy()’ adds a matrix to another one multiplied by
 * a single precision floating point value, ‘y = a*x + y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_csr_saxpy(
    float a,
    const struct mtxmatrix_csr * x,
    struct mtxmatrix_csr * y,
    int64_t * num_flops)
{
    return mtxvector_base_saxpy(a, &x->a, &y->a, num_flops);
}

/**
 * ‘mtxmatrix_csr_daxpy()’ adds a matrix to another one multiplied by
 * a double precision floating point value, ‘y = a*x + y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_csr_daxpy(
    double a,
    const struct mtxmatrix_csr * x,
    struct mtxmatrix_csr * y,
    int64_t * num_flops)
{
    return mtxvector_base_daxpy(a, &x->a, &y->a, num_flops);
}

/**
 * ‘mtxmatrix_csr_saypx()’ multiplies a matrix by a single precision
 * floating point scalar and adds another matrix, ‘y = a*y + x’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_csr_saypx(
    float a,
    struct mtxmatrix_csr * y,
    const struct mtxmatrix_csr * x,
    int64_t * num_flops)
{
    return mtxvector_base_saypx(a, &y->a, &x->a, num_flops);
}

/**
 * ‘mtxmatrix_csr_daypx()’ multiplies a matrix by a double precision
 * floating point scalar and adds another matrix, ‘y = a*y + x’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_csr_daypx(
    double a,
    struct mtxmatrix_csr * y,
    const struct mtxmatrix_csr * x,
    int64_t * num_flops)
{
    return mtxvector_base_daypx(a, &y->a, &x->a, num_flops);
}

/**
 * ‘mtxmatrix_csr_sdot()’ computes the Frobenius inner product of two
 * matrices in single precision floating point.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_csr_sdot(
    const struct mtxmatrix_csr * x,
    const struct mtxmatrix_csr * y,
    float * dot,
    int64_t * num_flops)
{
    if (x->symmetry == mtx_unsymmetric && y->symmetry == mtx_unsymmetric) {
        return mtxvector_base_sdot(&x->a, &y->a, dot, num_flops);
    } else { return MTX_ERR_INVALID_SYMMETRY; }
}

/**
 * ‘mtxmatrix_csr_ddot()’ computes the Frobenius inner product of two
 * matrices in double precision floating point.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_csr_ddot(
    const struct mtxmatrix_csr * x,
    const struct mtxmatrix_csr * y,
    double * dot,
    int64_t * num_flops)
{
    if (x->symmetry == mtx_unsymmetric && y->symmetry == mtx_unsymmetric) {
        return mtxvector_base_ddot(&x->a, &y->a, dot, num_flops);
    } else { return MTX_ERR_INVALID_SYMMETRY; }
}

/**
 * ‘mtxmatrix_csr_cdotu()’ computes the product of the transpose of a
 * complex row matrix with another complex row matrix in single
 * precision floating point, ‘dot := x^T*y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_csr_cdotu(
    const struct mtxmatrix_csr * x,
    const struct mtxmatrix_csr * y,
    float (* dot)[2],
    int64_t * num_flops)
{
    if (x->symmetry == mtx_unsymmetric && y->symmetry == mtx_unsymmetric) {
        return mtxvector_base_cdotu(&x->a, &y->a, dot, num_flops);
    } else { return MTX_ERR_INVALID_SYMMETRY; }
}

/**
 * ‘mtxmatrix_csr_zdotu()’ computes the product of the transpose of a
 * complex row matrix with another complex row matrix in double
 * precision floating point, ‘dot := x^T*y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_csr_zdotu(
    const struct mtxmatrix_csr * x,
    const struct mtxmatrix_csr * y,
    double (* dot)[2],
    int64_t * num_flops)
{
    if (x->symmetry == mtx_unsymmetric && y->symmetry == mtx_unsymmetric) {
        return mtxvector_base_zdotu(&x->a, &y->a, dot, num_flops);
    } else { return MTX_ERR_INVALID_SYMMETRY; }
}

/**
 * ‘mtxmatrix_csr_cdotc()’ computes the Frobenius inner product of two
 * complex matrices in single precision floating point, ‘dot :=
 * x^H*y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_csr_cdotc(
    const struct mtxmatrix_csr * x,
    const struct mtxmatrix_csr * y,
    float (* dot)[2],
    int64_t * num_flops)
{
    if (x->symmetry == mtx_unsymmetric && y->symmetry == mtx_unsymmetric) {
        return mtxvector_base_cdotc(&x->a, &y->a, dot, num_flops);
    } else { return MTX_ERR_INVALID_SYMMETRY; }
}

/**
 * ‘mtxmatrix_csr_zdotc()’ computes the Frobenius inner product of two
 * complex matrices in double precision floating point, ‘dot :=
 * x^H*y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_csr_zdotc(
    const struct mtxmatrix_csr * x,
    const struct mtxmatrix_csr * y,
    double (* dot)[2],
    int64_t * num_flops)
{
    if (x->symmetry == mtx_unsymmetric && y->symmetry == mtx_unsymmetric) {
        return mtxvector_base_zdotc(&x->a, &y->a, dot, num_flops);
    } else { return MTX_ERR_INVALID_SYMMETRY; }
}

/**
 * ‘mtxmatrix_csr_snrm2()’ computes the Frobenius norm of a matrix in
 * single precision floating point.
 */
int mtxmatrix_csr_snrm2(
    const struct mtxmatrix_csr * x,
    float * nrm2,
    int64_t * num_flops)
{
    if (x->symmetry == mtx_unsymmetric) {
        return mtxvector_base_snrm2(&x->a, nrm2, num_flops);
    } else { return MTX_ERR_INVALID_SYMMETRY; }
}

/**
 * ‘mtxmatrix_csr_dnrm2()’ computes the Frobenius norm of a matrix in
 * double precision floating point.
 */
int mtxmatrix_csr_dnrm2(
    const struct mtxmatrix_csr * x,
    double * nrm2,
    int64_t * num_flops)
{
    if (x->symmetry == mtx_unsymmetric) {
        return mtxvector_base_dnrm2(&x->a, nrm2, num_flops);
    } else { return MTX_ERR_INVALID_SYMMETRY; }
}

/**
 * ‘mtxmatrix_csr_sasum()’ computes the sum of absolute values
 * (1-norm) of a matrix in single precision floating point.  If the
 * matrix is complex-valued, then the sum of the absolute values of
 * the real and imaginary parts is computed.
 */
int mtxmatrix_csr_sasum(
    const struct mtxmatrix_csr * x,
    float * asum,
    int64_t * num_flops)
{
    if (x->symmetry == mtx_unsymmetric) {
        return mtxvector_base_sasum(&x->a, asum, num_flops);
    } else { return MTX_ERR_INVALID_SYMMETRY; }
}

/**
 * ‘mtxmatrix_csr_dasum()’ computes the sum of absolute values
 * (1-norm) of a matrix in double precision floating point.  If the
 * matrix is complex-valued, then the sum of the absolute values of
 * the real and imaginary parts is computed.
 */
int mtxmatrix_csr_dasum(
    const struct mtxmatrix_csr * x,
    double * asum,
    int64_t * num_flops)
{
    if (x->symmetry == mtx_unsymmetric) {
        return mtxvector_base_dasum(&x->a, asum, num_flops);
    } else { return MTX_ERR_INVALID_SYMMETRY; }
}

/**
 * ‘mtxmatrix_csr_iamax()’ finds the index of the first element having
 * the maximum absolute value.  If the matrix is complex-valued, then
 * the index points to the first element having the maximum sum of the
 * absolute values of the real and imaginary parts.
 */
int mtxmatrix_csr_iamax(
    const struct mtxmatrix_csr * x,
    int * iamax)
{
    return mtxvector_base_iamax(&x->a, iamax);
}

/*
 * Level 2 BLAS operations (matrix-vector)
 */

/**
 * ‘mtxmatrix_csr_sgemv()’ multiplies a matrix ‘A’ or its transpose
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
 */
int mtxmatrix_csr_sgemv(
    enum mtxtransposition trans,
    float alpha,
    const struct mtxmatrix_csr * A,
    const struct mtxvector * x,
    float beta,
    struct mtxvector * y,
    int64_t * num_flops)
{
    int err;
    const struct mtxvector_base * a = &A->a;
    if (x->type != mtxvector_base || y->type != mtxvector_base)
        return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
    const struct mtxvector_base * x_ = &x->storage.base;
    struct mtxvector_base * y_ = &y->storage.base;
    if (x_->field != a->field || y_->field != a->field)
        return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x_->precision != a->precision || y_->precision != a->precision)
        return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (trans == mtx_notrans) {
        if (A->num_rows != y_->size || A->num_columns != x_->size)
            return MTX_ERR_INCOMPATIBLE_SIZE;
    } else if (trans == mtx_trans || trans == mtx_conjtrans) {
        if (A->num_columns != y_->size || A->num_rows != x_->size)
            return MTX_ERR_INCOMPATIBLE_SIZE;
    }

    if (beta != 1) {
        err = mtxvector_sscal(beta, y, num_flops);
        if (err) return err;
    }

    const int64_t * rowptr = A->rowptr;
    const int * j = A->colidx;
    if (A->symmetry == mtx_unsymmetric) {
        if (a->field == mtx_field_real) {
            if (trans == mtx_notrans) {
                if (a->precision == mtx_single) {
                    const float * Adata = a->data.real_single;
                    const float * xdata = x_->data.real_single;
                    float * ydata = y_->data.real_single;
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++)
                            ydata[i] += alpha*Adata[k]*xdata[j[k]];
                    }
                    if (num_flops) *num_flops += 3*A->num_nonzeros;
                } else if (a->precision == mtx_double) {
                    const double * Adata = a->data.real_double;
                    const double * xdata = x_->data.real_double;
                    double * ydata = y_->data.real_double;
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++)
                            ydata[i] += alpha*Adata[k]*xdata[j[k]];
                    }
                    if (num_flops) *num_flops += 3*A->num_nonzeros;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (trans == mtx_trans || trans == mtx_conjtrans) {
                if (a->precision == mtx_single) {
                    const float * Adata = a->data.real_single;
                    const float * xdata = x_->data.real_single;
                    float * ydata = y_->data.real_single;
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++)
                            ydata[j[k]] += alpha*Adata[k]*xdata[i];
                    }
                    if (num_flops) *num_flops += 3*A->num_nonzeros;
                } else if (a->precision == mtx_double) {
                    const double * Adata = a->data.real_double;
                    const double * xdata = x_->data.real_double;
                    double * ydata = y_->data.real_double;
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++)
                            ydata[j[k]] += alpha*Adata[k]*xdata[i];
                    }
                    if (num_flops) *num_flops += 3*A->num_nonzeros;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else { return MTX_ERR_INVALID_TRANSPOSITION; }
        } else if (a->field == mtx_field_complex) {
            if (trans == mtx_notrans) {
                if (a->precision == mtx_single) {
                    const float (* Adata)[2] = a->data.complex_single;
                    const float (* xdata)[2] = x_->data.complex_single;
                    float (* ydata)[2] = y_->data.complex_single;
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++) {
                            ydata[i][0] += alpha*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                            ydata[i][1] += alpha*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_nonzeros;
                } else if (a->precision == mtx_double) {
                    const double (* Adata)[2] = a->data.complex_double;
                    const double (* xdata)[2] = x_->data.complex_double;
                    double (* ydata)[2] = y_->data.complex_double;
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++) {
                            ydata[i][0] += alpha*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                            ydata[i][1] += alpha*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_nonzeros;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (trans == mtx_trans) {
                if (a->precision == mtx_single) {
                    const float (* Adata)[2] = a->data.complex_single;
                    const float (* xdata)[2] = x_->data.complex_single;
                    float (* ydata)[2] = y_->data.complex_single;
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++) {
                            ydata[j[k]][0] += alpha*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                            ydata[j[k]][1] += alpha*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_nonzeros;
                } else if (a->precision == mtx_double) {
                    const double (* Adata)[2] = a->data.complex_double;
                    const double (* xdata)[2] = x_->data.complex_double;
                    double (* ydata)[2] = y_->data.complex_double;
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++) {
                            ydata[j[k]][0] += alpha*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                            ydata[j[k]][1] += alpha*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_nonzeros;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (trans == mtx_conjtrans) {
                if (a->precision == mtx_single) {
                    const float (* Adata)[2] = a->data.complex_single;
                    const float (* xdata)[2] = x_->data.complex_single;
                    float (* ydata)[2] = y_->data.complex_single;
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++) {
                            ydata[j[k]][0] += alpha*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                            ydata[j[k]][1] += alpha*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_nonzeros;
                } else if (a->precision == mtx_double) {
                    const double (* Adata)[2] = a->data.complex_double;
                    const double (* xdata)[2] = x_->data.complex_double;
                    double (* ydata)[2] = y_->data.complex_double;
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++) {
                            ydata[j[k]][0] += alpha*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                            ydata[j[k]][1] += alpha*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_nonzeros;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else { return MTX_ERR_INVALID_TRANSPOSITION; }
        } else if (a->field == mtx_field_integer) {
            if (trans == mtx_notrans) {
                if (a->precision == mtx_single) {
                    const int32_t * Adata = a->data.integer_single;
                    const int32_t * xdata = x_->data.integer_single;
                    int32_t * ydata = y_->data.integer_single;
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++)
                            ydata[i] += alpha*Adata[k]*xdata[j[k]];
                    }
                    if (num_flops) *num_flops += 3*A->num_nonzeros;
                } else if (a->precision == mtx_double) {
                    const int64_t * Adata = a->data.integer_double;
                    const int64_t * xdata = x_->data.integer_double;
                    int64_t * ydata = y_->data.integer_double;
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++)
                            ydata[i] += alpha*Adata[k]*xdata[j[k]];
                    }
                    if (num_flops) *num_flops += 3*A->num_nonzeros;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (trans == mtx_trans || trans == mtx_conjtrans) {
                if (a->precision == mtx_single) {
                    const int32_t * Adata = a->data.integer_single;
                    const int32_t * xdata = x_->data.integer_single;
                    int32_t * ydata = y_->data.integer_single;
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++)
                            ydata[j[k]] += alpha*Adata[k]*xdata[i];
                    }
                    if (num_flops) *num_flops += 3*A->num_nonzeros;
                } else if (a->precision == mtx_double) {
                    const int64_t * Adata = a->data.integer_double;
                    const int64_t * xdata = x_->data.integer_double;
                    int64_t * ydata = y_->data.integer_double;
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++)
                            ydata[j[k]] += alpha*Adata[k]*xdata[i];
                    }
                    if (num_flops) *num_flops += 3*A->num_nonzeros;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else { return MTX_ERR_INVALID_TRANSPOSITION; }
        } else { return MTX_ERR_INVALID_FIELD; }
    } else if (A->num_rows == A->num_columns && A->symmetry == mtx_symmetric) {
        int err = mtxvector_base_usgz(
            (struct mtxvector_packed *) &A->diag,
            (struct mtxvector_base *) &A->a);
        if (err) return err;
        if (a->field == mtx_field_real) {
            if (a->precision == mtx_single) {
                const float * Adata = a->data.real_single;
                const float * xdata = x_->data.real_single;
                float * ydata = y_->data.real_single;
                for (int i = 0; i < A->num_rows; i++) {
                    for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++) {
                        ydata[i] += alpha*Adata[k]*xdata[j[k]];
                        ydata[j[k]] += alpha*Adata[k]*xdata[i];
                    }
                }
                const float * Adiag = A->diag.x.storage.base.data.real_single;
                for (int k = 0; k < A->diag.num_nonzeros; k++) {
                    int i = A->colidx[A->diag.idx[k]];
                    ydata[i] += alpha*Adiag[k]*xdata[i];
                }
                if (num_flops) *num_flops += 6*A->size + 3*A->diag.num_nonzeros;
            } else if (a->precision == mtx_double) {
                const double * Adata = a->data.real_double;
                const double * xdata = x_->data.real_double;
                double * ydata = y_->data.real_double;
                for (int i = 0; i < A->num_rows; i++) {
                    for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++) {
                        ydata[i] += alpha*Adata[k]*xdata[j[k]];
                        ydata[j[k]] += alpha*Adata[k]*xdata[i];
                    }
                }
                const double * Adiag = A->diag.x.storage.base.data.real_double;
                for (int k = 0; k < A->diag.num_nonzeros; k++) {
                    int i = A->colidx[A->diag.idx[k]];
                    ydata[i] += alpha*Adiag[k]*xdata[i];
                }
                if (num_flops) *num_flops += 6*A->size + 3*A->diag.num_nonzeros;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (a->field == mtx_field_complex) {
            if (trans == mtx_notrans || trans == mtx_trans) {
                if (a->precision == mtx_single) {
                    const float (* Adata)[2] = a->data.complex_single;
                    const float (* xdata)[2] = x_->data.complex_single;
                    float (* ydata)[2] = y_->data.complex_single;
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++) {
                            ydata[i][0] += alpha*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                            ydata[i][1] += alpha*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                            ydata[j[k]][0] += alpha*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                            ydata[j[k]][1] += alpha*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        }
                    }
                    const float (* Adiag)[2] = A->diag.x.storage.base.data.complex_single;
                    for (int k = 0; k < A->diag.num_nonzeros; k++) {
                        int i = A->colidx[A->diag.idx[k]];
                        ydata[i][0] += alpha*(Adiag[k][0]*xdata[i][0]-Adiag[k][1]*xdata[i][1]);
                        ydata[i][1] += alpha*(Adiag[k][0]*xdata[i][1]+Adiag[k][1]*xdata[i][0]);
                    }
                    if (num_flops) *num_flops += 20*A->size + 10*A->diag.num_nonzeros;
                } else if (a->precision == mtx_double) {
                    const double (* Adata)[2] = a->data.complex_double;
                    const double (* xdata)[2] = x_->data.complex_double;
                    double (* ydata)[2] = y_->data.complex_double;
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++) {
                            ydata[i][0] += alpha*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                            ydata[i][1] += alpha*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                            ydata[j[k]][0] += alpha*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                            ydata[j[k]][1] += alpha*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        }
                    }
                    const double (* Adiag)[2] = A->diag.x.storage.base.data.complex_double;
                    for (int k = 0; k < A->diag.num_nonzeros; k++) {
                        int i = A->colidx[A->diag.idx[k]];
                        ydata[i][0] += alpha*(Adiag[k][0]*xdata[i][0]-Adiag[k][1]*xdata[i][1]);
                        ydata[i][1] += alpha*(Adiag[k][0]*xdata[i][1]+Adiag[k][1]*xdata[i][0]);
                    }
                    if (num_flops) *num_flops += 20*A->size + 10*A->diag.num_nonzeros;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (trans == mtx_conjtrans) {
                if (a->precision == mtx_single) {
                    const float (* Adata)[2] = a->data.complex_single;
                    const float (* xdata)[2] = x_->data.complex_single;
                    float (* ydata)[2] = y_->data.complex_single;
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++) {
                            ydata[i][0] += alpha*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]);
                            ydata[i][1] += alpha*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]);
                            ydata[j[k]][0] += alpha*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                            ydata[j[k]][1] += alpha*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        }
                    }
                    const float (* Adiag)[2] = A->diag.x.storage.base.data.complex_single;
                    for (int k = 0; k < A->diag.num_nonzeros; k++) {
                        int i = A->colidx[A->diag.idx[k]];
                        ydata[i][0] += alpha*(Adiag[k][0]*xdata[i][0]+Adiag[k][1]*xdata[i][1]);
                        ydata[i][1] += alpha*(Adiag[k][0]*xdata[i][1]-Adiag[k][1]*xdata[i][0]);
                    }
                    if (num_flops) *num_flops += 20*A->size + 10*A->diag.num_nonzeros;
                } else if (a->precision == mtx_double) {
                    const double (* Adata)[2] = a->data.complex_double;
                    const double (* xdata)[2] = x_->data.complex_double;
                    double (* ydata)[2] = y_->data.complex_double;
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++) {
                            ydata[i][0] += alpha*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]);
                            ydata[i][1] += alpha*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]);
                            ydata[j[k]][0] += alpha*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                            ydata[j[k]][1] += alpha*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        }
                    }
                    const double (* Adiag)[2] = A->diag.x.storage.base.data.complex_double;
                    for (int k = 0; k < A->diag.num_nonzeros; k++) {
                        int i = A->colidx[A->diag.idx[k]];
                        ydata[i][0] += alpha*(Adiag[k][0]*xdata[i][0]+Adiag[k][1]*xdata[i][1]);
                        ydata[i][1] += alpha*(Adiag[k][0]*xdata[i][1]-Adiag[k][1]*xdata[i][0]);
                    }
                    if (num_flops) *num_flops += 20*A->size + 10*A->diag.num_nonzeros;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else { return MTX_ERR_INVALID_TRANSPOSITION; }
        } else if (a->field == mtx_field_integer) {
            if (a->precision == mtx_single) {
                const int32_t * Adata = a->data.integer_single;
                const int32_t * xdata = x_->data.integer_single;
                int32_t * ydata = y_->data.integer_single;
                for (int i = 0; i < A->num_rows; i++) {
                    for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++) {
                        ydata[i] += alpha*Adata[k]*xdata[j[k]];
                        ydata[j[k]] += alpha*Adata[k]*xdata[i];
                    }
                }
                const int32_t * Adiag = A->diag.x.storage.base.data.integer_single;
                for (int k = 0; k < A->diag.num_nonzeros; k++) {
                    int i = A->colidx[A->diag.idx[k]];
                    ydata[i] += alpha*Adiag[k]*xdata[i];
                }
                if (num_flops) *num_flops += 6*A->size + 3*A->diag.num_nonzeros;
            } else if (a->precision == mtx_double) {
                const int64_t * Adata = a->data.integer_double;
                const int64_t * xdata = x_->data.integer_double;
                int64_t * ydata = y_->data.integer_double;
                for (int i = 0; i < A->num_rows; i++) {
                    for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++) {
                        ydata[i] += alpha*Adata[k]*xdata[j[k]];
                        ydata[j[k]] += alpha*Adata[k]*xdata[i];
                    }
                }
                const int64_t * Adiag = A->diag.x.storage.base.data.integer_double;
                for (int k = 0; k < A->diag.num_nonzeros; k++) {
                    int i = A->colidx[A->diag.idx[k]];
                    ydata[i] += alpha*Adiag[k]*xdata[i];
                }
                if (num_flops) *num_flops += 6*A->size + 3*A->diag.num_nonzeros;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else { return MTX_ERR_INVALID_FIELD; }
        err = mtxvector_base_ussc((struct mtxvector_base *) &A->a, &A->diag);
        if (err) return err;
    } else if (A->num_rows == A->num_columns && A->symmetry == mtx_skew_symmetric) {
        /* TODO: allow skew-symmetric matrices */
        return MTX_ERR_INVALID_SYMMETRY;
    } else if (A->num_rows == A->num_columns && A->symmetry == mtx_hermitian) {
        int err = mtxvector_base_usgz(
            (struct mtxvector_packed *) &A->diag,
            (struct mtxvector_base *) &A->a);
        if (err) return err;
        if (a->field == mtx_field_complex) {
            if (trans == mtx_notrans || trans == mtx_conjtrans) {
                if (a->precision == mtx_single) {
                    const float (* Adata)[2] = a->data.complex_single;
                    const float (* xdata)[2] = x_->data.complex_single;
                    float (* ydata)[2] = y_->data.complex_single;
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++) {
                            ydata[i][0] += alpha*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                            ydata[i][1] += alpha*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                            ydata[j[k]][0] += alpha*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                            ydata[j[k]][1] += alpha*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        }
                    }
                    const float (* Adiag)[2] = A->diag.x.storage.base.data.complex_single;
                    for (int k = 0; k < A->diag.num_nonzeros; k++) {
                        int i = A->colidx[A->diag.idx[k]];
                        ydata[i][0] += alpha*(Adiag[k][0]*xdata[i][0]+Adiag[k][1]*xdata[i][1]);
                        ydata[i][1] += alpha*(Adiag[k][0]*xdata[i][1]-Adiag[k][1]*xdata[i][0]);
                    }
                    if (num_flops) *num_flops += 20*A->size + 10*A->diag.num_nonzeros;
                } else if (a->precision == mtx_double) {
                    const double (* Adata)[2] = a->data.complex_double;
                    const double (* xdata)[2] = x_->data.complex_double;
                    double (* ydata)[2] = y_->data.complex_double;
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++) {
                            ydata[i][0] += alpha*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                            ydata[i][1] += alpha*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                            ydata[j[k]][0] += alpha*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                            ydata[j[k]][1] += alpha*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        }
                    }
                    const double (* Adiag)[2] = A->diag.x.storage.base.data.complex_double;
                    for (int k = 0; k < A->diag.num_nonzeros; k++) {
                        int i = A->colidx[A->diag.idx[k]];
                        ydata[i][0] += alpha*(Adiag[k][0]*xdata[i][0]+Adiag[k][1]*xdata[i][1]);
                        ydata[i][1] += alpha*(Adiag[k][0]*xdata[i][1]-Adiag[k][1]*xdata[i][0]);
                    }
                    if (num_flops) *num_flops += 20*A->size + 10*A->diag.num_nonzeros;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (trans == mtx_trans) {
                if (a->precision == mtx_single) {
                    const float (* Adata)[2] = a->data.complex_single;
                    const float (* xdata)[2] = x_->data.complex_single;
                    float (* ydata)[2] = y_->data.complex_single;
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++) {
                            ydata[i][0] += alpha*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]);
                            ydata[i][1] += alpha*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]);
                            ydata[j[k]][0] += alpha*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                            ydata[j[k]][1] += alpha*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        }
                    }
                    const float (* Adiag)[2] = A->diag.x.storage.base.data.complex_single;
                    for (int k = 0; k < A->diag.num_nonzeros; k++) {
                        int i = A->colidx[A->diag.idx[k]];
                        ydata[i][0] += alpha*(Adiag[k][0]*xdata[i][0]-Adiag[k][1]*xdata[i][1]);
                        ydata[i][1] += alpha*(Adiag[k][0]*xdata[i][1]+Adiag[k][1]*xdata[i][0]);
                    }
                    if (num_flops) *num_flops += 20*A->size + 10*A->diag.num_nonzeros;
                } else if (a->precision == mtx_double) {
                    const double (* Adata)[2] = a->data.complex_double;
                    const double (* xdata)[2] = x_->data.complex_double;
                    double (* ydata)[2] = y_->data.complex_double;
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++) {
                            ydata[i][0] += alpha*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]);
                            ydata[i][1] += alpha*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]);
                            ydata[j[k]][0] += alpha*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                            ydata[j[k]][1] += alpha*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        }
                    }
                    const double (* Adiag)[2] = A->diag.x.storage.base.data.complex_double;
                    for (int k = 0; k < A->diag.num_nonzeros; k++) {
                        int i = A->colidx[A->diag.idx[k]];
                        ydata[i][0] += alpha*(Adiag[k][0]*xdata[i][0]-Adiag[k][1]*xdata[i][1]);
                        ydata[i][1] += alpha*(Adiag[k][0]*xdata[i][1]+Adiag[k][1]*xdata[i][0]);
                    }
                    if (num_flops) *num_flops += 20*A->size + 10*A->diag.num_nonzeros;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else { return MTX_ERR_INVALID_TRANSPOSITION; }
        } else { return MTX_ERR_INVALID_FIELD; }
        err = mtxvector_base_ussc((struct mtxvector_base *) &A->a, &A->diag);
        if (err) return err;
    } else { return MTX_ERR_INVALID_SYMMETRY; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_csr_dgemv()’ multiplies a matrix ‘A’ or its transpose
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
 */
int mtxmatrix_csr_dgemv(
    enum mtxtransposition trans,
    double alpha,
    const struct mtxmatrix_csr * A,
    const struct mtxvector * x,
    double beta,
    struct mtxvector * y,
    int64_t * num_flops)
{
    int err;
    const struct mtxvector_base * a = &A->a;
    if (x->type != mtxvector_base || y->type != mtxvector_base)
        return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
    const struct mtxvector_base * x_ = &x->storage.base;
    struct mtxvector_base * y_ = &y->storage.base;
    if (x_->field != a->field || y_->field != a->field)
        return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x_->precision != a->precision || y_->precision != a->precision)
        return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (trans == mtx_notrans) {
        if (A->num_rows != y_->size || A->num_columns != x_->size)
            return MTX_ERR_INCOMPATIBLE_SIZE;
    } else if (trans == mtx_trans || trans == mtx_conjtrans) {
        if (A->num_columns != y_->size || A->num_rows != x_->size)
            return MTX_ERR_INCOMPATIBLE_SIZE;
    }

    if (beta != 1) {
        err = mtxvector_dscal(beta, y, num_flops);
        if (err) return err;
    }

    const int64_t * rowptr = A->rowptr;
    const int * j = A->colidx;
    if (A->symmetry == mtx_unsymmetric) {
        if (a->field == mtx_field_real) {
            if (trans == mtx_notrans) {
                if (a->precision == mtx_single) {
                    const float * Adata = a->data.real_single;
                    const float * xdata = x_->data.real_single;
                    float * ydata = y_->data.real_single;
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++)
                            ydata[i] += alpha*Adata[k]*xdata[j[k]];
                    }
                    if (num_flops) *num_flops += 3*A->num_nonzeros;
                } else if (a->precision == mtx_double) {
                    const double * Adata = a->data.real_double;
                    const double * xdata = x_->data.real_double;
                    double * ydata = y_->data.real_double;
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++)
                            ydata[i] += alpha*Adata[k]*xdata[j[k]];
                    }
                    if (num_flops) *num_flops += 3*A->num_nonzeros;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (trans == mtx_trans || trans == mtx_conjtrans) {
                if (a->precision == mtx_single) {
                    const float * Adata = a->data.real_single;
                    const float * xdata = x_->data.real_single;
                    float * ydata = y_->data.real_single;
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++)
                            ydata[j[k]] += alpha*Adata[k]*xdata[i];
                    }
                    if (num_flops) *num_flops += 3*A->num_nonzeros;
                } else if (a->precision == mtx_double) {
                    const double * Adata = a->data.real_double;
                    const double * xdata = x_->data.real_double;
                    double * ydata = y_->data.real_double;
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++)
                            ydata[j[k]] += alpha*Adata[k]*xdata[i];
                    }
                    if (num_flops) *num_flops += 3*A->num_nonzeros;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else { return MTX_ERR_INVALID_TRANSPOSITION; }
        } else if (a->field == mtx_field_complex) {
            if (trans == mtx_notrans) {
                if (a->precision == mtx_single) {
                    const float (* Adata)[2] = a->data.complex_single;
                    const float (* xdata)[2] = x_->data.complex_single;
                    float (* ydata)[2] = y_->data.complex_single;
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++) {
                            ydata[i][0] += alpha*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                            ydata[i][1] += alpha*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_nonzeros;
                } else if (a->precision == mtx_double) {
                    const double (* Adata)[2] = a->data.complex_double;
                    const double (* xdata)[2] = x_->data.complex_double;
                    double (* ydata)[2] = y_->data.complex_double;
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++) {
                            ydata[i][0] += alpha*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                            ydata[i][1] += alpha*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_nonzeros;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (trans == mtx_trans) {
                if (a->precision == mtx_single) {
                    const float (* Adata)[2] = a->data.complex_single;
                    const float (* xdata)[2] = x_->data.complex_single;
                    float (* ydata)[2] = y_->data.complex_single;
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++) {
                            ydata[j[k]][0] += alpha*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                            ydata[j[k]][1] += alpha*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_nonzeros;
                } else if (a->precision == mtx_double) {
                    const double (* Adata)[2] = a->data.complex_double;
                    const double (* xdata)[2] = x_->data.complex_double;
                    double (* ydata)[2] = y_->data.complex_double;
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++) {
                            ydata[j[k]][0] += alpha*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                            ydata[j[k]][1] += alpha*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_nonzeros;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (trans == mtx_conjtrans) {
                if (a->precision == mtx_single) {
                    const float (* Adata)[2] = a->data.complex_single;
                    const float (* xdata)[2] = x_->data.complex_single;
                    float (* ydata)[2] = y_->data.complex_single;
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++) {
                            ydata[j[k]][0] += alpha*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                            ydata[j[k]][1] += alpha*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_nonzeros;
                } else if (a->precision == mtx_double) {
                    const double (* Adata)[2] = a->data.complex_double;
                    const double (* xdata)[2] = x_->data.complex_double;
                    double (* ydata)[2] = y_->data.complex_double;
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++) {
                            ydata[j[k]][0] += alpha*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                            ydata[j[k]][1] += alpha*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_nonzeros;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else { return MTX_ERR_INVALID_TRANSPOSITION; }
        } else if (a->field == mtx_field_integer) {
            if (trans == mtx_notrans) {
                if (a->precision == mtx_single) {
                    const int32_t * Adata = a->data.integer_single;
                    const int32_t * xdata = x_->data.integer_single;
                    int32_t * ydata = y_->data.integer_single;
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++)
                            ydata[i] += alpha*Adata[k]*xdata[j[k]];
                    }
                    if (num_flops) *num_flops += 3*A->num_nonzeros;
                } else if (a->precision == mtx_double) {
                    const int64_t * Adata = a->data.integer_double;
                    const int64_t * xdata = x_->data.integer_double;
                    int64_t * ydata = y_->data.integer_double;
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++)
                            ydata[i] += alpha*Adata[k]*xdata[j[k]];
                    }
                    if (num_flops) *num_flops += 3*A->num_nonzeros;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (trans == mtx_trans || trans == mtx_conjtrans) {
                if (a->precision == mtx_single) {
                    const int32_t * Adata = a->data.integer_single;
                    const int32_t * xdata = x_->data.integer_single;
                    int32_t * ydata = y_->data.integer_single;
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++)
                            ydata[j[k]] += alpha*Adata[k]*xdata[i];
                    }
                    if (num_flops) *num_flops += 3*A->num_nonzeros;
                } else if (a->precision == mtx_double) {
                    const int64_t * Adata = a->data.integer_double;
                    const int64_t * xdata = x_->data.integer_double;
                    int64_t * ydata = y_->data.integer_double;
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++)
                            ydata[j[k]] += alpha*Adata[k]*xdata[i];
                    }
                    if (num_flops) *num_flops += 3*A->num_nonzeros;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else { return MTX_ERR_INVALID_TRANSPOSITION; }
        } else { return MTX_ERR_INVALID_FIELD; }
    } else if (A->num_rows == A->num_columns && A->symmetry == mtx_symmetric) {
        int err = mtxvector_base_usgz(
            (struct mtxvector_packed *) &A->diag,
            (struct mtxvector_base *) &A->a);
        if (err) return err;
        if (a->field == mtx_field_real) {
            if (a->precision == mtx_single) {
                const float * Adata = a->data.real_single;
                const float * xdata = x_->data.real_single;
                float * ydata = y_->data.real_single;
                for (int i = 0; i < A->num_rows; i++) {
                    for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++) {
                        ydata[i] += alpha*Adata[k]*xdata[j[k]];
                        ydata[j[k]] += alpha*Adata[k]*xdata[i];
                    }
                }
                const float * Adiag = A->diag.x.storage.base.data.real_single;
                for (int k = 0; k < A->diag.num_nonzeros; k++) {
                    int i = A->colidx[A->diag.idx[k]];
                    ydata[i] += alpha*Adiag[k]*xdata[i];
                }
                if (num_flops) *num_flops += 6*A->size + 3*A->diag.num_nonzeros;
            } else if (a->precision == mtx_double) {
                const double * Adata = a->data.real_double;
                const double * xdata = x_->data.real_double;
                double * ydata = y_->data.real_double;
                for (int i = 0; i < A->num_rows; i++) {
                    for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++) {
                        ydata[i] += alpha*Adata[k]*xdata[j[k]];
                        ydata[j[k]] += alpha*Adata[k]*xdata[i];
                    }
                }
                const double * Adiag = A->diag.x.storage.base.data.real_double;
                for (int k = 0; k < A->diag.num_nonzeros; k++) {
                    int i = A->colidx[A->diag.idx[k]];
                    ydata[i] += alpha*Adiag[k]*xdata[i];
                }
                if (num_flops) *num_flops += 6*A->size + 3*A->diag.num_nonzeros;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (a->field == mtx_field_complex) {
            if (trans == mtx_notrans || trans == mtx_trans) {
                if (a->precision == mtx_single) {
                    const float (* Adata)[2] = a->data.complex_single;
                    const float (* xdata)[2] = x_->data.complex_single;
                    float (* ydata)[2] = y_->data.complex_single;
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++) {
                            ydata[i][0] += alpha*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                            ydata[i][1] += alpha*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                            ydata[j[k]][0] += alpha*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                            ydata[j[k]][1] += alpha*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        }
                    }
                    const float (* Adiag)[2] = A->diag.x.storage.base.data.complex_single;
                    for (int k = 0; k < A->diag.num_nonzeros; k++) {
                        int i = A->colidx[A->diag.idx[k]];
                        ydata[i][0] += alpha*(Adiag[k][0]*xdata[i][0]-Adiag[k][1]*xdata[i][1]);
                        ydata[i][1] += alpha*(Adiag[k][0]*xdata[i][1]+Adiag[k][1]*xdata[i][0]);
                    }
                    if (num_flops) *num_flops += 20*A->size + 10*A->diag.num_nonzeros;
                } else if (a->precision == mtx_double) {
                    const double (* Adata)[2] = a->data.complex_double;
                    const double (* xdata)[2] = x_->data.complex_double;
                    double (* ydata)[2] = y_->data.complex_double;
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++) {
                            ydata[i][0] += alpha*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                            ydata[i][1] += alpha*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                            ydata[j[k]][0] += alpha*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                            ydata[j[k]][1] += alpha*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        }
                    }
                    const double (* Adiag)[2] = A->diag.x.storage.base.data.complex_double;
                    for (int k = 0; k < A->diag.num_nonzeros; k++) {
                        int i = A->colidx[A->diag.idx[k]];
                        ydata[i][0] += alpha*(Adiag[k][0]*xdata[i][0]-Adiag[k][1]*xdata[i][1]);
                        ydata[i][1] += alpha*(Adiag[k][0]*xdata[i][1]+Adiag[k][1]*xdata[i][0]);
                    }
                    if (num_flops) *num_flops += 20*A->size + 10*A->diag.num_nonzeros;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (trans == mtx_conjtrans) {
                if (a->precision == mtx_single) {
                    const float (* Adata)[2] = a->data.complex_single;
                    const float (* xdata)[2] = x_->data.complex_single;
                    float (* ydata)[2] = y_->data.complex_single;
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++) {
                            ydata[i][0] += alpha*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]);
                            ydata[i][1] += alpha*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]);
                            ydata[j[k]][0] += alpha*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                            ydata[j[k]][1] += alpha*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        }
                    }
                    const float (* Adiag)[2] = A->diag.x.storage.base.data.complex_single;
                    for (int k = 0; k < A->diag.num_nonzeros; k++) {
                        int i = A->colidx[A->diag.idx[k]];
                        ydata[i][0] += alpha*(Adiag[k][0]*xdata[i][0]+Adiag[k][1]*xdata[i][1]);
                        ydata[i][1] += alpha*(Adiag[k][0]*xdata[i][1]-Adiag[k][1]*xdata[i][0]);
                    }
                    if (num_flops) *num_flops += 20*A->size + 10*A->diag.num_nonzeros;
                } else if (a->precision == mtx_double) {
                    const double (* Adata)[2] = a->data.complex_double;
                    const double (* xdata)[2] = x_->data.complex_double;
                    double (* ydata)[2] = y_->data.complex_double;
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++) {
                            ydata[i][0] += alpha*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]);
                            ydata[i][1] += alpha*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]);
                            ydata[j[k]][0] += alpha*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                            ydata[j[k]][1] += alpha*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        }
                    }
                    const double (* Adiag)[2] = A->diag.x.storage.base.data.complex_double;
                    for (int k = 0; k < A->diag.num_nonzeros; k++) {
                        int i = A->colidx[A->diag.idx[k]];
                        ydata[i][0] += alpha*(Adiag[k][0]*xdata[i][0]+Adiag[k][1]*xdata[i][1]);
                        ydata[i][1] += alpha*(Adiag[k][0]*xdata[i][1]-Adiag[k][1]*xdata[i][0]);
                    }
                    if (num_flops) *num_flops += 20*A->size + 10*A->diag.num_nonzeros;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else { return MTX_ERR_INVALID_TRANSPOSITION; }
        } else if (a->field == mtx_field_integer) {
            if (a->precision == mtx_single) {
                const int32_t * Adata = a->data.integer_single;
                const int32_t * xdata = x_->data.integer_single;
                int32_t * ydata = y_->data.integer_single;
                for (int i = 0; i < A->num_rows; i++) {
                    for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++) {
                        ydata[i] += alpha*Adata[k]*xdata[j[k]];
                        ydata[j[k]] += alpha*Adata[k]*xdata[i];
                    }
                }
                const int32_t * Adiag = A->diag.x.storage.base.data.integer_single;
                for (int k = 0; k < A->diag.num_nonzeros; k++) {
                    int i = A->colidx[A->diag.idx[k]];
                    ydata[i] += alpha*Adiag[k]*xdata[i];
                }
                if (num_flops) *num_flops += 6*A->size + 3*A->diag.num_nonzeros;
            } else if (a->precision == mtx_double) {
                const int64_t * Adata = a->data.integer_double;
                const int64_t * xdata = x_->data.integer_double;
                int64_t * ydata = y_->data.integer_double;
                for (int i = 0; i < A->num_rows; i++) {
                    for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++) {
                        ydata[i] += alpha*Adata[k]*xdata[j[k]];
                        ydata[j[k]] += alpha*Adata[k]*xdata[i];
                    }
                }
                const int64_t * Adiag = A->diag.x.storage.base.data.integer_double;
                for (int k = 0; k < A->diag.num_nonzeros; k++) {
                    int i = A->colidx[A->diag.idx[k]];
                    ydata[i] += alpha*Adiag[k]*xdata[i];
                }
                if (num_flops) *num_flops += 6*A->size + 3*A->diag.num_nonzeros;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else { return MTX_ERR_INVALID_FIELD; }
        err = mtxvector_base_ussc((struct mtxvector_base *) &A->a, &A->diag);
        if (err) return err;
    } else if (A->num_rows == A->num_columns && A->symmetry == mtx_skew_symmetric) {
        /* TODO: allow skew-symmetric matrices */
        return MTX_ERR_INVALID_SYMMETRY;
    } else if (A->num_rows == A->num_columns && A->symmetry == mtx_hermitian) {
        int err = mtxvector_base_usgz(
            (struct mtxvector_packed *) &A->diag,
            (struct mtxvector_base *) &A->a);
        if (err) return err;
        if (a->field == mtx_field_complex) {
            if (trans == mtx_notrans || trans == mtx_conjtrans) {
                if (a->precision == mtx_single) {
                    const float (* Adata)[2] = a->data.complex_single;
                    const float (* xdata)[2] = x_->data.complex_single;
                    float (* ydata)[2] = y_->data.complex_single;
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++) {
                            ydata[i][0] += alpha*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                            ydata[i][1] += alpha*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                            ydata[j[k]][0] += alpha*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                            ydata[j[k]][1] += alpha*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        }
                    }
                    const float (* Adiag)[2] = A->diag.x.storage.base.data.complex_single;
                    for (int k = 0; k < A->diag.num_nonzeros; k++) {
                        int i = A->colidx[A->diag.idx[k]];
                        ydata[i][0] += alpha*(Adiag[k][0]*xdata[i][0]+Adiag[k][1]*xdata[i][1]);
                        ydata[i][1] += alpha*(Adiag[k][0]*xdata[i][1]-Adiag[k][1]*xdata[i][0]);
                    }
                    if (num_flops) *num_flops += 20*A->size + 10*A->diag.num_nonzeros;
                } else if (a->precision == mtx_double) {
                    const double (* Adata)[2] = a->data.complex_double;
                    const double (* xdata)[2] = x_->data.complex_double;
                    double (* ydata)[2] = y_->data.complex_double;
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++) {
                            ydata[i][0] += alpha*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                            ydata[i][1] += alpha*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                            ydata[j[k]][0] += alpha*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                            ydata[j[k]][1] += alpha*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        }
                    }
                    const double (* Adiag)[2] = A->diag.x.storage.base.data.complex_double;
                    for (int k = 0; k < A->diag.num_nonzeros; k++) {
                        int i = A->colidx[A->diag.idx[k]];
                        ydata[i][0] += alpha*(Adiag[k][0]*xdata[i][0]+Adiag[k][1]*xdata[i][1]);
                        ydata[i][1] += alpha*(Adiag[k][0]*xdata[i][1]-Adiag[k][1]*xdata[i][0]);
                    }
                    if (num_flops) *num_flops += 20*A->size + 10*A->diag.num_nonzeros;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (trans == mtx_trans) {
                if (a->precision == mtx_single) {
                    const float (* Adata)[2] = a->data.complex_single;
                    const float (* xdata)[2] = x_->data.complex_single;
                    float (* ydata)[2] = y_->data.complex_single;
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++) {
                            ydata[i][0] += alpha*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]);
                            ydata[i][1] += alpha*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]);
                            ydata[j[k]][0] += alpha*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                            ydata[j[k]][1] += alpha*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        }
                    }
                    const float (* Adiag)[2] = A->diag.x.storage.base.data.complex_single;
                    for (int k = 0; k < A->diag.num_nonzeros; k++) {
                        int i = A->colidx[A->diag.idx[k]];
                        ydata[i][0] += alpha*(Adiag[k][0]*xdata[i][0]-Adiag[k][1]*xdata[i][1]);
                        ydata[i][1] += alpha*(Adiag[k][0]*xdata[i][1]+Adiag[k][1]*xdata[i][0]);
                    }
                    if (num_flops) *num_flops += 20*A->size + 10*A->diag.num_nonzeros;
                } else if (a->precision == mtx_double) {
                    const double (* Adata)[2] = a->data.complex_double;
                    const double (* xdata)[2] = x_->data.complex_double;
                    double (* ydata)[2] = y_->data.complex_double;
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++) {
                            ydata[i][0] += alpha*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]);
                            ydata[i][1] += alpha*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]);
                            ydata[j[k]][0] += alpha*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                            ydata[j[k]][1] += alpha*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        }
                    }
                    const double (* Adiag)[2] = A->diag.x.storage.base.data.complex_double;
                    for (int k = 0; k < A->diag.num_nonzeros; k++) {
                        int i = A->colidx[A->diag.idx[k]];
                        ydata[i][0] += alpha*(Adiag[k][0]*xdata[i][0]-Adiag[k][1]*xdata[i][1]);
                        ydata[i][1] += alpha*(Adiag[k][0]*xdata[i][1]+Adiag[k][1]*xdata[i][0]);
                    }
                    if (num_flops) *num_flops += 20*A->size + 10*A->diag.num_nonzeros;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else { return MTX_ERR_INVALID_TRANSPOSITION; }
        } else { return MTX_ERR_INVALID_FIELD; }
        err = mtxvector_base_ussc((struct mtxvector_base *) &A->a, &A->diag);
        if (err) return err;
    } else { return MTX_ERR_INVALID_SYMMETRY; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_csr_cgemv()’ multiplies a complex-valued matrix ‘A’, its
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
 */
int mtxmatrix_csr_cgemv(
    enum mtxtransposition trans,
    float alpha[2],
    const struct mtxmatrix_csr * A,
    const struct mtxvector * x,
    float beta[2],
    struct mtxvector * y,
    int64_t * num_flops)
{
    int err;
    const struct mtxvector_base * a = &A->a;
    if (x->type != mtxvector_base || y->type != mtxvector_base)
        return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
    const struct mtxvector_base * x_ = &x->storage.base;
    struct mtxvector_base * y_ = &y->storage.base;
    if (x_->field != a->field || y_->field != a->field)
        return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x_->precision != a->precision || y_->precision != a->precision)
        return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (trans == mtx_notrans) {
        if (A->num_rows != y_->size || A->num_columns != x_->size)
            return MTX_ERR_INCOMPATIBLE_SIZE;
    } else if (trans == mtx_trans || trans == mtx_conjtrans) {
        if (A->num_columns != y_->size || A->num_rows != x_->size)
            return MTX_ERR_INCOMPATIBLE_SIZE;
    }
    if (a->field != mtx_field_complex)
        return MTX_ERR_INCOMPATIBLE_FIELD;

    if (beta[0] != 1 || beta[1] != 0) {
        err = mtxvector_cscal(beta, y, num_flops);
        if (err) return err;
    }

    const int64_t * rowptr = A->rowptr;
    const int * j = A->colidx;
    if (A->symmetry == mtx_unsymmetric) {
        if (trans == mtx_notrans) {
            if (a->precision == mtx_single) {
                const float (* Adata)[2] = a->data.complex_single;
                const float (* xdata)[2] = x_->data.complex_single;
                float (* ydata)[2] = y_->data.complex_single;
                for (int i = 0; i < A->num_rows; i++) {
                    for (int k = rowptr[i]; k < rowptr[i+1]; k++) {
                        ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1])-alpha[1]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                        ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0])+alpha[1]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                    }
                }
                if (num_flops) *num_flops += 20*A->num_nonzeros;
            } else if (a->precision == mtx_double) {
                const double (* Adata)[2] = a->data.complex_double;
                const double (* xdata)[2] = x_->data.complex_double;
                double (* ydata)[2] = y_->data.complex_double;
                for (int i = 0; i < A->num_rows; i++) {
                    for (int k = rowptr[i]; k < rowptr[i+1]; k++) {
                        ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1])-alpha[1]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                        ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0])+alpha[1]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                    }
                }
                if (num_flops) *num_flops += 20*A->num_nonzeros;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (trans == mtx_trans) {
            if (a->precision == mtx_single) {
                const float (* Adata)[2] = a->data.complex_single;
                const float (* xdata)[2] = x_->data.complex_single;
                float (* ydata)[2] = y_->data.complex_single;
                for (int i = 0; i < A->num_rows; i++) {
                    for (int k = rowptr[i]; k < rowptr[i+1]; k++) {
                        ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1])-alpha[1]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0])+alpha[1]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                    }
                }
                if (num_flops) *num_flops += 20*A->num_nonzeros;
            } else if (a->precision == mtx_double) {
                const double (* Adata)[2] = a->data.complex_double;
                const double (* xdata)[2] = x_->data.complex_double;
                double (* ydata)[2] = y_->data.complex_double;
                for (int i = 0; i < A->num_rows; i++) {
                    for (int k = rowptr[i]; k < rowptr[i+1]; k++) {
                        ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1])-alpha[1]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0])+alpha[1]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                    }
                }
                if (num_flops) *num_flops += 20*A->num_nonzeros;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (trans == mtx_conjtrans) {
            if (a->precision == mtx_single) {
                const float (* Adata)[2] = a->data.complex_single;
                const float (* xdata)[2] = x_->data.complex_single;
                float (* ydata)[2] = y_->data.complex_single;
                for (int i = 0; i < A->num_rows; i++) {
                    for (int k = rowptr[i]; k < rowptr[i+1]; k++) {
                        ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1])-alpha[1]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0])+alpha[1]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                    }
                }
                if (num_flops) *num_flops += 20*A->num_nonzeros;
            } else if (a->precision == mtx_double) {
                const double (* Adata)[2] = a->data.complex_double;
                const double (* xdata)[2] = x_->data.complex_double;
                double (* ydata)[2] = y_->data.complex_double;
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
        int err = mtxvector_base_usgz(
            (struct mtxvector_packed *) &A->diag,
            (struct mtxvector_base *) &A->a);
        if (err) return err;
        if (trans == mtx_notrans || trans == mtx_trans) {
            if (a->precision == mtx_single) {
                const float (* Adata)[2] = a->data.complex_single;
                const float (* xdata)[2] = x_->data.complex_single;
                float (* ydata)[2] = y_->data.complex_single;
                for (int i = 0; i < A->num_rows; i++) {
                    for (int k = rowptr[i]; k < rowptr[i+1]; k++) {
                        ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]) - alpha[1]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                        ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]) + alpha[1]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                        ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                    }
                }
                const float (* Adiag)[2] = A->diag.x.storage.base.data.complex_single;
                for (int k = 0; k < A->diag.num_nonzeros; k++) {
                    int i = A->colidx[A->diag.idx[k]];
                    ydata[i][0] += alpha[0]*(Adiag[k][0]*xdata[i][0]-Adiag[k][1]*xdata[i][1]) - alpha[1]*(Adiag[k][0]*xdata[i][1]+Adiag[k][1]*xdata[i][0]);
                    ydata[i][1] += alpha[0]*(Adiag[k][0]*xdata[i][1]+Adiag[k][1]*xdata[i][0]) + alpha[1]*(Adiag[k][0]*xdata[i][0]-Adiag[k][1]*xdata[i][1]);
                }
                if (num_flops) *num_flops += 40*A->size + 20*A->diag.num_nonzeros;
            } else if (a->precision == mtx_double) {
                const double (* Adata)[2] = a->data.complex_double;
                const double (* xdata)[2] = x_->data.complex_double;
                double (* ydata)[2] = y_->data.complex_double;
                for (int i = 0; i < A->num_rows; i++) {
                    for (int k = rowptr[i]; k < rowptr[i+1]; k++) {
                        ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]) - alpha[1]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                        ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]) + alpha[1]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                        ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                    }
                }
                const double (* Adiag)[2] = A->diag.x.storage.base.data.complex_double;
                for (int k = 0; k < A->diag.num_nonzeros; k++) {
                    int i = A->colidx[A->diag.idx[k]];
                    ydata[j[k]][0] += alpha[0]*(Adiag[k][0]*xdata[i][0]-Adiag[k][1]*xdata[i][1]) - alpha[1]*(Adiag[k][0]*xdata[i][1]+Adiag[k][1]*xdata[i][0]);
                    ydata[j[k]][1] += alpha[0]*(Adiag[k][0]*xdata[i][1]+Adiag[k][1]*xdata[i][0]) + alpha[1]*(Adiag[k][0]*xdata[i][0]-Adiag[k][1]*xdata[i][1]);
                }
                if (num_flops) *num_flops += 40*A->size + 20*A->diag.num_nonzeros;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (trans == mtx_conjtrans) {
            if (a->precision == mtx_single) {
                const float (* Adata)[2] = a->data.complex_single;
                const float (* xdata)[2] = x_->data.complex_single;
                float (* ydata)[2] = y_->data.complex_single;
                for (int i = 0; i < A->num_rows; i++) {
                    for (int k = rowptr[i]; k < rowptr[i+1]; k++) {
                        ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]) - alpha[1]*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]);
                        ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]) + alpha[1]*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]);
                        ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                    }
                }
                const float (* Adiag)[2] = A->diag.x.storage.base.data.complex_single;
                for (int k = 0; k < A->diag.num_nonzeros; k++) {
                    int i = A->colidx[A->diag.idx[k]];
                    ydata[i][0] += alpha[0]*(Adiag[k][0]*xdata[i][0]+Adiag[k][1]*xdata[i][1]) - alpha[1]*(Adiag[k][0]*xdata[i][1]-Adiag[k][1]*xdata[i][0]);
                    ydata[i][1] += alpha[0]*(Adiag[k][0]*xdata[i][1]-Adiag[k][1]*xdata[i][0]) + alpha[1]*(Adiag[k][0]*xdata[i][0]+Adiag[k][1]*xdata[i][1]);
                }
                if (num_flops) *num_flops += 40*A->size + 20*A->diag.num_nonzeros;
            } else if (a->precision == mtx_double) {
                const double (* Adata)[2] = a->data.complex_double;
                const double (* xdata)[2] = x_->data.complex_double;
                double (* ydata)[2] = y_->data.complex_double;
                for (int i = 0; i < A->num_rows; i++) {
                    for (int k = rowptr[i]; k < rowptr[i+1]; k++) {
                        ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]) - alpha[1]*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]);
                        ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]) + alpha[1]*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]);
                        ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                    }
                }
                const double (* Adiag)[2] = A->diag.x.storage.base.data.complex_double;
                for (int k = 0; k < A->diag.num_nonzeros; k++) {
                    int i = A->colidx[A->diag.idx[k]];
                        ydata[i][0] += alpha[0]*(Adiag[k][0]*xdata[i][0]+Adiag[k][1]*xdata[i][1]) - alpha[1]*(Adiag[k][0]*xdata[i][1]-Adiag[k][1]*xdata[i][0]);
                        ydata[i][1] += alpha[0]*(Adiag[k][0]*xdata[i][1]-Adiag[k][1]*xdata[i][0]) + alpha[1]*(Adiag[k][0]*xdata[i][0]+Adiag[k][1]*xdata[i][1]);
                }
                if (num_flops) *num_flops += 40*A->size + 20*A->diag.num_nonzeros;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else { return MTX_ERR_INVALID_TRANSPOSITION; }
        err = mtxvector_base_ussc((struct mtxvector_base *) &A->a, &A->diag);
        if (err) return err;
    } else if (A->num_rows == A->num_columns && A->symmetry == mtx_hermitian) {
        int err = mtxvector_base_usgz(
            (struct mtxvector_packed *) &A->diag,
            (struct mtxvector_base *) &A->a);
        if (err) return err;
        if (trans == mtx_notrans || trans == mtx_conjtrans) {
            if (a->precision == mtx_single) {
                const float (* Adata)[2] = a->data.complex_single;
                const float (* xdata)[2] = x_->data.complex_single;
                float (* ydata)[2] = y_->data.complex_single;
                for (int i = 0; i < A->num_rows; i++) {
                    for (int k = rowptr[i]; k < rowptr[i+1]; k++) {
                        ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]) - alpha[1]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                        ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]) + alpha[1]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                        ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                    }
                }
                const float (* Adiag)[2] = A->diag.x.storage.base.data.complex_single;
                for (int k = 0; k < A->diag.num_nonzeros; k++) {
                    int i = A->colidx[A->diag.idx[k]];
                    ydata[i][0] += alpha[0]*(Adiag[k][0]*xdata[i][0]+Adiag[k][1]*xdata[i][1]) - alpha[1]*(Adiag[k][0]*xdata[i][1]-Adiag[k][1]*xdata[i][0]);
                    ydata[i][1] += alpha[0]*(Adiag[k][0]*xdata[i][1]-Adiag[k][1]*xdata[i][0]) + alpha[1]*(Adiag[k][0]*xdata[i][0]+Adiag[k][1]*xdata[i][1]);
                }
                if (num_flops) *num_flops += 40*A->size + 20*A->diag.num_nonzeros;
            } else if (a->precision == mtx_double) {
                const double (* Adata)[2] = a->data.complex_double;
                const double (* xdata)[2] = x_->data.complex_double;
                double (* ydata)[2] = y_->data.complex_double;
                for (int i = 0; i < A->num_rows; i++) {
                    for (int k = rowptr[i]; k < rowptr[i+1]; k++) {
                        ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]) - alpha[1]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                        ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]) + alpha[1]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                        ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                    }
                }
                const double (* Adiag)[2] = A->diag.x.storage.base.data.complex_double;
                for (int k = 0; k < A->diag.num_nonzeros; k++) {
                    int i = A->colidx[A->diag.idx[k]];
                    ydata[i][0] += alpha[0]*(Adiag[k][0]*xdata[i][0]+Adiag[k][1]*xdata[i][1]) - alpha[1]*(Adiag[k][0]*xdata[i][1]-Adiag[k][1]*xdata[i][0]);
                    ydata[i][1] += alpha[0]*(Adiag[k][0]*xdata[i][1]-Adiag[k][1]*xdata[i][0]) + alpha[1]*(Adiag[k][0]*xdata[i][0]+Adiag[k][1]*xdata[i][1]);
                }
                if (num_flops) *num_flops += 40*A->size + 20*A->diag.num_nonzeros;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (trans == mtx_trans) {
            if (a->precision == mtx_single) {
                const float (* Adata)[2] = a->data.complex_single;
                const float (* xdata)[2] = x_->data.complex_single;
                float (* ydata)[2] = y_->data.complex_single;
                for (int i = 0; i < A->num_rows; i++) {
                    for (int k = rowptr[i]; k < rowptr[i+1]; k++) {
                        ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]) - alpha[1]*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]);
                        ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]) + alpha[1]*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]);
                        ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                    }
                }
                const float (* Adiag)[2] = A->diag.x.storage.base.data.complex_single;
                for (int k = 0; k < A->diag.num_nonzeros; k++) {
                    int i = A->colidx[A->diag.idx[k]];
                    ydata[i][0] += alpha[0]*(Adiag[k][0]*xdata[i][0]-Adiag[k][1]*xdata[i][1]) - alpha[1]*(Adiag[k][0]*xdata[i][1]+Adiag[k][1]*xdata[i][0]);
                    ydata[i][1] += alpha[0]*(Adiag[k][0]*xdata[i][1]+Adiag[k][1]*xdata[i][0]) + alpha[1]*(Adiag[k][0]*xdata[i][0]-Adiag[k][1]*xdata[i][1]);
                }
                if (num_flops) *num_flops += 40*A->size + 20*A->diag.num_nonzeros;
            } else if (a->precision == mtx_double) {
                const double (* Adata)[2] = a->data.complex_double;
                const double (* xdata)[2] = x_->data.complex_double;
                double (* ydata)[2] = y_->data.complex_double;
                for (int i = 0; i < A->num_rows; i++) {
                    for (int k = rowptr[i]; k < rowptr[i+1]; k++) {
                        ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]) - alpha[1]*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]);
                        ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]) + alpha[1]*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]);
                        ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                    }
                }
                const double (* Adiag)[2] = A->diag.x.storage.base.data.complex_double;
                for (int k = 0; k < A->diag.num_nonzeros; k++) {
                    int i = A->colidx[A->diag.idx[k]];
                    ydata[i][0] += alpha[0]*(Adiag[k][0]*xdata[i][0]-Adiag[k][1]*xdata[i][1]) - alpha[1]*(Adiag[k][0]*xdata[i][1]+Adiag[k][1]*xdata[i][0]);
                    ydata[i][1] += alpha[0]*(Adiag[k][0]*xdata[i][1]+Adiag[k][1]*xdata[i][0]) + alpha[1]*(Adiag[k][0]*xdata[i][0]-Adiag[k][1]*xdata[i][1]);
                }
                if (num_flops) *num_flops += 40*A->size + 20*A->diag.num_nonzeros;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else { return MTX_ERR_INVALID_TRANSPOSITION; }
        err = mtxvector_base_ussc((struct mtxvector_base *) &A->a, &A->diag);
        if (err) return err;
    } else { return MTX_ERR_INVALID_SYMMETRY; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_csr_zgemv()’ multiplies a complex-valued matrix ‘A’, its
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
 */
int mtxmatrix_csr_zgemv(
    enum mtxtransposition trans,
    double alpha[2],
    const struct mtxmatrix_csr * A,
    const struct mtxvector * x,
    double beta[2],
    struct mtxvector * y,
    int64_t * num_flops)
{
    int err;
    const struct mtxvector_base * a = &A->a;
    if (x->type != mtxvector_base || y->type != mtxvector_base)
        return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
    const struct mtxvector_base * x_ = &x->storage.base;
    struct mtxvector_base * y_ = &y->storage.base;
    if (x_->field != a->field || y_->field != a->field)
        return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x_->precision != a->precision || y_->precision != a->precision)
        return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (trans == mtx_notrans) {
        if (A->num_rows != y_->size || A->num_columns != x_->size)
            return MTX_ERR_INCOMPATIBLE_SIZE;
    } else if (trans == mtx_trans || trans == mtx_conjtrans) {
        if (A->num_columns != y_->size || A->num_rows != x_->size)
            return MTX_ERR_INCOMPATIBLE_SIZE;
    }
    if (a->field != mtx_field_complex)
        return MTX_ERR_INCOMPATIBLE_FIELD;

    if (beta[0] != 1 || beta[1] != 0) {
        err = mtxvector_zscal(beta, y, num_flops);
        if (err) return err;
    }

    const int64_t * rowptr = A->rowptr;
    const int * j = A->colidx;
    if (A->symmetry == mtx_unsymmetric) {
        if (trans == mtx_notrans) {
            if (a->precision == mtx_single) {
                const float (* Adata)[2] = a->data.complex_single;
                const float (* xdata)[2] = x_->data.complex_single;
                float (* ydata)[2] = y_->data.complex_single;
                for (int i = 0; i < A->num_rows; i++) {
                    for (int k = rowptr[i]; k < rowptr[i+1]; k++) {
                        ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1])-alpha[1]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                        ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0])+alpha[1]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                    }
                }
                if (num_flops) *num_flops += 20*A->num_nonzeros;
            } else if (a->precision == mtx_double) {
                const double (* Adata)[2] = a->data.complex_double;
                const double (* xdata)[2] = x_->data.complex_double;
                double (* ydata)[2] = y_->data.complex_double;
                for (int i = 0; i < A->num_rows; i++) {
                    for (int k = rowptr[i]; k < rowptr[i+1]; k++) {
                        ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1])-alpha[1]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                        ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0])+alpha[1]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                    }
                }
                if (num_flops) *num_flops += 20*A->num_nonzeros;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (trans == mtx_trans) {
            if (a->precision == mtx_single) {
                const float (* Adata)[2] = a->data.complex_single;
                const float (* xdata)[2] = x_->data.complex_single;
                float (* ydata)[2] = y_->data.complex_single;
                for (int i = 0; i < A->num_rows; i++) {
                    for (int k = rowptr[i]; k < rowptr[i+1]; k++) {
                        ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1])-alpha[1]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0])+alpha[1]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                    }
                }
                if (num_flops) *num_flops += 20*A->num_nonzeros;
            } else if (a->precision == mtx_double) {
                const double (* Adata)[2] = a->data.complex_double;
                const double (* xdata)[2] = x_->data.complex_double;
                double (* ydata)[2] = y_->data.complex_double;
                for (int i = 0; i < A->num_rows; i++) {
                    for (int k = rowptr[i]; k < rowptr[i+1]; k++) {
                        ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1])-alpha[1]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0])+alpha[1]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                    }
                }
                if (num_flops) *num_flops += 20*A->num_nonzeros;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (trans == mtx_conjtrans) {
            if (a->precision == mtx_single) {
                const float (* Adata)[2] = a->data.complex_single;
                const float (* xdata)[2] = x_->data.complex_single;
                float (* ydata)[2] = y_->data.complex_single;
                for (int i = 0; i < A->num_rows; i++) {
                    for (int k = rowptr[i]; k < rowptr[i+1]; k++) {
                        ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1])-alpha[1]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0])+alpha[1]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                    }
                }
                if (num_flops) *num_flops += 20*A->num_nonzeros;
            } else if (a->precision == mtx_double) {
                const double (* Adata)[2] = a->data.complex_double;
                const double (* xdata)[2] = x_->data.complex_double;
                double (* ydata)[2] = y_->data.complex_double;
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
        int err = mtxvector_base_usgz(
            (struct mtxvector_packed *) &A->diag,
            (struct mtxvector_base *) &A->a);
        if (err) return err;
        if (trans == mtx_notrans || trans == mtx_trans) {
            if (a->precision == mtx_single) {
                const float (* Adata)[2] = a->data.complex_single;
                const float (* xdata)[2] = x_->data.complex_single;
                float (* ydata)[2] = y_->data.complex_single;
                for (int i = 0; i < A->num_rows; i++) {
                    for (int k = rowptr[i]; k < rowptr[i+1]; k++) {
                        ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]) - alpha[1]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                        ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]) + alpha[1]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                        ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                    }
                }
                const float (* Adiag)[2] = A->diag.x.storage.base.data.complex_single;
                for (int k = 0; k < A->diag.num_nonzeros; k++) {
                    int i = A->colidx[A->diag.idx[k]];
                    ydata[i][0] += alpha[0]*(Adiag[k][0]*xdata[i][0]-Adiag[k][1]*xdata[i][1]) - alpha[1]*(Adiag[k][0]*xdata[i][1]+Adiag[k][1]*xdata[i][0]);
                    ydata[i][1] += alpha[0]*(Adiag[k][0]*xdata[i][1]+Adiag[k][1]*xdata[i][0]) + alpha[1]*(Adiag[k][0]*xdata[i][0]-Adiag[k][1]*xdata[i][1]);
                }
                if (num_flops) *num_flops += 40*A->size + 20*A->diag.num_nonzeros;
            } else if (a->precision == mtx_double) {
                const double (* Adata)[2] = a->data.complex_double;
                const double (* xdata)[2] = x_->data.complex_double;
                double (* ydata)[2] = y_->data.complex_double;
                for (int i = 0; i < A->num_rows; i++) {
                    for (int k = rowptr[i]; k < rowptr[i+1]; k++) {
                        ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]) - alpha[1]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                        ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]) + alpha[1]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                        ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                    }
                }
                const double (* Adiag)[2] = A->diag.x.storage.base.data.complex_double;
                for (int k = 0; k < A->diag.num_nonzeros; k++) {
                    int i = A->colidx[A->diag.idx[k]];
                    ydata[j[k]][0] += alpha[0]*(Adiag[k][0]*xdata[i][0]-Adiag[k][1]*xdata[i][1]) - alpha[1]*(Adiag[k][0]*xdata[i][1]+Adiag[k][1]*xdata[i][0]);
                    ydata[j[k]][1] += alpha[0]*(Adiag[k][0]*xdata[i][1]+Adiag[k][1]*xdata[i][0]) + alpha[1]*(Adiag[k][0]*xdata[i][0]-Adiag[k][1]*xdata[i][1]);
                }
                if (num_flops) *num_flops += 40*A->size + 20*A->diag.num_nonzeros;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (trans == mtx_conjtrans) {
            if (a->precision == mtx_single) {
                const float (* Adata)[2] = a->data.complex_single;
                const float (* xdata)[2] = x_->data.complex_single;
                float (* ydata)[2] = y_->data.complex_single;
                for (int i = 0; i < A->num_rows; i++) {
                    for (int k = rowptr[i]; k < rowptr[i+1]; k++) {
                        ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]) - alpha[1]*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]);
                        ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]) + alpha[1]*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]);
                        ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                    }
                }
                const float (* Adiag)[2] = A->diag.x.storage.base.data.complex_single;
                for (int k = 0; k < A->diag.num_nonzeros; k++) {
                    int i = A->colidx[A->diag.idx[k]];
                    ydata[i][0] += alpha[0]*(Adiag[k][0]*xdata[i][0]+Adiag[k][1]*xdata[i][1]) - alpha[1]*(Adiag[k][0]*xdata[i][1]-Adiag[k][1]*xdata[i][0]);
                    ydata[i][1] += alpha[0]*(Adiag[k][0]*xdata[i][1]-Adiag[k][1]*xdata[i][0]) + alpha[1]*(Adiag[k][0]*xdata[i][0]+Adiag[k][1]*xdata[i][1]);
                }
                if (num_flops) *num_flops += 40*A->size + 20*A->diag.num_nonzeros;
            } else if (a->precision == mtx_double) {
                const double (* Adata)[2] = a->data.complex_double;
                const double (* xdata)[2] = x_->data.complex_double;
                double (* ydata)[2] = y_->data.complex_double;
                for (int i = 0; i < A->num_rows; i++) {
                    for (int k = rowptr[i]; k < rowptr[i+1]; k++) {
                        ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]) - alpha[1]*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]);
                        ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]) + alpha[1]*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]);
                        ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                    }
                }
                const double (* Adiag)[2] = A->diag.x.storage.base.data.complex_double;
                for (int k = 0; k < A->diag.num_nonzeros; k++) {
                    int i = A->colidx[A->diag.idx[k]];
                        ydata[i][0] += alpha[0]*(Adiag[k][0]*xdata[i][0]+Adiag[k][1]*xdata[i][1]) - alpha[1]*(Adiag[k][0]*xdata[i][1]-Adiag[k][1]*xdata[i][0]);
                        ydata[i][1] += alpha[0]*(Adiag[k][0]*xdata[i][1]-Adiag[k][1]*xdata[i][0]) + alpha[1]*(Adiag[k][0]*xdata[i][0]+Adiag[k][1]*xdata[i][1]);
                }
                if (num_flops) *num_flops += 40*A->size + 20*A->diag.num_nonzeros;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else { return MTX_ERR_INVALID_TRANSPOSITION; }
        err = mtxvector_base_ussc((struct mtxvector_base *) &A->a, &A->diag);
        if (err) return err;
    } else if (A->num_rows == A->num_columns && A->symmetry == mtx_hermitian) {
        int err = mtxvector_base_usgz(
            (struct mtxvector_packed *) &A->diag,
            (struct mtxvector_base *) &A->a);
        if (err) return err;
        if (trans == mtx_notrans || trans == mtx_conjtrans) {
            if (a->precision == mtx_single) {
                const float (* Adata)[2] = a->data.complex_single;
                const float (* xdata)[2] = x_->data.complex_single;
                float (* ydata)[2] = y_->data.complex_single;
                for (int i = 0; i < A->num_rows; i++) {
                    for (int k = rowptr[i]; k < rowptr[i+1]; k++) {
                        ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]) - alpha[1]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                        ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]) + alpha[1]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                        ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                    }
                }
                const float (* Adiag)[2] = A->diag.x.storage.base.data.complex_single;
                for (int k = 0; k < A->diag.num_nonzeros; k++) {
                    int i = A->colidx[A->diag.idx[k]];
                    ydata[i][0] += alpha[0]*(Adiag[k][0]*xdata[i][0]+Adiag[k][1]*xdata[i][1]) - alpha[1]*(Adiag[k][0]*xdata[i][1]-Adiag[k][1]*xdata[i][0]);
                    ydata[i][1] += alpha[0]*(Adiag[k][0]*xdata[i][1]-Adiag[k][1]*xdata[i][0]) + alpha[1]*(Adiag[k][0]*xdata[i][0]+Adiag[k][1]*xdata[i][1]);
                }
                if (num_flops) *num_flops += 40*A->size + 20*A->diag.num_nonzeros;
            } else if (a->precision == mtx_double) {
                const double (* Adata)[2] = a->data.complex_double;
                const double (* xdata)[2] = x_->data.complex_double;
                double (* ydata)[2] = y_->data.complex_double;
                for (int i = 0; i < A->num_rows; i++) {
                    for (int k = rowptr[i]; k < rowptr[i+1]; k++) {
                        ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]) - alpha[1]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                        ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]) + alpha[1]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                        ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                    }
                }
                const double (* Adiag)[2] = A->diag.x.storage.base.data.complex_double;
                for (int k = 0; k < A->diag.num_nonzeros; k++) {
                    int i = A->colidx[A->diag.idx[k]];
                    ydata[i][0] += alpha[0]*(Adiag[k][0]*xdata[i][0]+Adiag[k][1]*xdata[i][1]) - alpha[1]*(Adiag[k][0]*xdata[i][1]-Adiag[k][1]*xdata[i][0]);
                    ydata[i][1] += alpha[0]*(Adiag[k][0]*xdata[i][1]-Adiag[k][1]*xdata[i][0]) + alpha[1]*(Adiag[k][0]*xdata[i][0]+Adiag[k][1]*xdata[i][1]);
                }
                if (num_flops) *num_flops += 40*A->size + 20*A->diag.num_nonzeros;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (trans == mtx_trans) {
            if (a->precision == mtx_single) {
                const float (* Adata)[2] = a->data.complex_single;
                const float (* xdata)[2] = x_->data.complex_single;
                float (* ydata)[2] = y_->data.complex_single;
                for (int i = 0; i < A->num_rows; i++) {
                    for (int k = rowptr[i]; k < rowptr[i+1]; k++) {
                        ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]) - alpha[1]*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]);
                        ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]) + alpha[1]*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]);
                        ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                    }
                }
                const float (* Adiag)[2] = A->diag.x.storage.base.data.complex_single;
                for (int k = 0; k < A->diag.num_nonzeros; k++) {
                    int i = A->colidx[A->diag.idx[k]];
                    ydata[i][0] += alpha[0]*(Adiag[k][0]*xdata[i][0]-Adiag[k][1]*xdata[i][1]) - alpha[1]*(Adiag[k][0]*xdata[i][1]+Adiag[k][1]*xdata[i][0]);
                    ydata[i][1] += alpha[0]*(Adiag[k][0]*xdata[i][1]+Adiag[k][1]*xdata[i][0]) + alpha[1]*(Adiag[k][0]*xdata[i][0]-Adiag[k][1]*xdata[i][1]);
                }
                if (num_flops) *num_flops += 40*A->size + 20*A->diag.num_nonzeros;
            } else if (a->precision == mtx_double) {
                const double (* Adata)[2] = a->data.complex_double;
                const double (* xdata)[2] = x_->data.complex_double;
                double (* ydata)[2] = y_->data.complex_double;
                for (int i = 0; i < A->num_rows; i++) {
                    for (int k = rowptr[i]; k < rowptr[i+1]; k++) {
                        ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]) - alpha[1]*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]);
                        ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]) + alpha[1]*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]);
                        ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                    }
                }
                const double (* Adiag)[2] = A->diag.x.storage.base.data.complex_double;
                for (int k = 0; k < A->diag.num_nonzeros; k++) {
                    int i = A->colidx[A->diag.idx[k]];
                    ydata[i][0] += alpha[0]*(Adiag[k][0]*xdata[i][0]-Adiag[k][1]*xdata[i][1]) - alpha[1]*(Adiag[k][0]*xdata[i][1]+Adiag[k][1]*xdata[i][0]);
                    ydata[i][1] += alpha[0]*(Adiag[k][0]*xdata[i][1]+Adiag[k][1]*xdata[i][0]) + alpha[1]*(Adiag[k][0]*xdata[i][0]-Adiag[k][1]*xdata[i][1]);
                }
                if (num_flops) *num_flops += 40*A->size + 20*A->diag.num_nonzeros;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else { return MTX_ERR_INVALID_TRANSPOSITION; }
        err = mtxvector_base_ussc((struct mtxvector_base *) &A->a, &A->diag);
        if (err) return err;
    } else { return MTX_ERR_INVALID_SYMMETRY; }
    return MTX_SUCCESS;
}
