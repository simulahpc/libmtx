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
 * Dense matrices with BLAS-accelerated operations.
 */

#include <libmtx/libmtx-config.h>

#include <libmtx/error.h>
#include <libmtx/linalg/field.h>
#include <libmtx/linalg/precision.h>
#include <libmtx/linalg/blas/dense.h>
#include <libmtx/linalg/local/matrix.h>
#include <libmtx/mtxfile/mtxfile.h>
#include <libmtx/linalg/local/vector.h>

#ifdef LIBMTX_HAVE_BLAS
#include <cblas.h>
#endif

#include <errno.h>

#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * ‘touptri()’ converts from the integer Cartesian coordinates of a
 * location ‘(i,j)’ in the upper triangle of an N-by-N matrix to the
 * position of the corresponding entry in a rowwise arrangement of the
 * upper triangular matrix elements.
 *
 * Note that it is required that i≤j.
 */
static inline int touptri(int N, int i, int j)
{
    return N*(N-1)/2 - (N-i)*(N-i-1)/2 + j;
}

/*
 * matrix properties
 */

/**
 * ‘mtxblasdense_field()’ gets the field of a matrix.
 */
enum mtxfield mtxblasdense_field(const struct mtxblasdense * A)
{
    return mtxblasvector_field(&A->a);
}

/**
 * ‘mtxblasdense_precision()’ gets the precision of a matrix.
 */
enum mtxprecision mtxblasdense_precision(const struct mtxblasdense * A)
{
    return mtxblasvector_precision(&A->a);
}

/**
 * ‘mtxblasdense_symmetry()’ gets the symmetry of a matrix.
 */
enum mtxsymmetry mtxblasdense_symmetry(const struct mtxblasdense * A)
{
    return A->symmetry;
}

/**
 * ‘mtxblasdense_num_rows()’ gets the number of matrix rows.
 */
int mtxblasdense_num_rows(const struct mtxblasdense * A)
{
    return A->num_rows;
}

/**
 * ‘mtxblasdense_num_columns()’ gets the number of matrix columns.
 */
int mtxblasdense_num_columns(const struct mtxblasdense * A)
{
    return A->num_columns;
}

/**
 * ‘mtxblasdense_num_nonzeros()’ gets the number of the number of
 *  nonzero matrix entries, including those represented implicitly due
 *  to symmetry.
 */
int64_t mtxblasdense_num_nonzeros(const struct mtxblasdense * A)
{
    return A->num_nonzeros;
}

/**
 * ‘mtxblasdense_size()’ gets the number of explicitly stored
 * nonzeros of a matrix.
 */
int64_t mtxblasdense_size(const struct mtxblasdense * A)
{
    return A->size;
}

/**
 * ‘mtxblasdense_rowcolidx()’ gets the row and column indices of the
 * explicitly stored matrix nonzeros.
 *
 * The arguments ‘rowidx’ and ‘colidx’ may be ‘NULL’ or must point to
 * an arrays of length ‘size’.
 */
int mtxblasdense_rowcolidx(
    const struct mtxblasdense * A,
    int64_t size,
    int * rowidx,
    int * colidx)
{
    if (A->symmetry == mtx_unsymmetric) {
        int64_t k = 0;
        for (int i = 0; i < A->num_rows; i++) {
            for (int j = 0; j < A->num_columns; j++, k++) {
                if (rowidx) rowidx[k] = i;
                if (colidx) colidx[k] = j;
            }
        }
    } else if (A->num_rows == A->num_columns &&
               (A->symmetry == mtx_symmetric || A->symmetry == mtx_hermitian))
    {
        int64_t k = 0;
        for (int i = 0; i < A->num_rows; i++) {
            for (int j = i; j < A->num_columns; j++, k++) {
                if (rowidx) rowidx[k] = i;
                if (colidx) colidx[k] = j;
            }
        }
    } else if (A->num_rows == A->num_columns && A->symmetry == mtx_skew_symmetric) {
        int64_t k = 0;
        for (int i = 0; i < A->num_rows; i++) {
            for (int j = i+1; j < A->num_columns; j++, k++) {
                if (rowidx) rowidx[k] = i;
                if (colidx) colidx[k] = j;
            }
        }
    } else { return MTX_ERR_INVALID_SYMMETRY; }
    return MTX_SUCCESS;
}

/*
 * memory management
 */

/**
 * ‘mtxblasdense_free()’ frees storage allocated for a matrix.
 */
void mtxblasdense_free(
    struct mtxblasdense * A)
{
    mtxblasvector_free(&A->a);
}

/**
 * ‘mtxblasdense_alloc_copy()’ allocates a copy of a matrix without
 * initialising the values.
 */
int mtxblasdense_alloc_copy(
    struct mtxblasdense * dst,
    const struct mtxblasdense * src)
{
    return mtxblasdense_alloc_entries(
        dst, src->a.base.field, src->a.base.precision, src->symmetry,
        src->num_rows, src->num_columns, src->size,
        0, 0, NULL, NULL);
}

/**
 * ‘mtxblasdense_init_copy()’ allocates a copy of a matrix and also
 * copies the values.
 */
int mtxblasdense_init_copy(
    struct mtxblasdense * dst,
    const struct mtxblasdense * src)
{
    int err = mtxblasdense_alloc_copy(dst, src);
    if (err) return err;
    err = mtxblasdense_copy(dst, src);
    if (err) { mtxblasdense_free(dst); return err; }
    return MTX_SUCCESS;
}

/*
 * initialise matrices from entrywise data in coordinate format
 */

/**
 * ‘mtxblasdense_alloc_entries()’ allocates a matrix from entrywise
 * data in coordinate format.
 */
int mtxblasdense_alloc_entries(
    struct mtxblasdense * A,
    enum mtxfield field,
    enum mtxprecision precision,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    int idxstride,
    int idxbase,
    const int * rowidx,
    const int * colidx)
{
    int64_t num_entries;
    if (__builtin_mul_overflow(num_rows, num_columns, &num_entries)) {
        errno = EOVERFLOW;
        return MTX_ERR_ERRNO;
    }
    A->symmetry = symmetry;
    A->num_rows = num_rows;
    A->num_columns = num_columns;
    A->num_entries = num_entries;
    if (symmetry == mtx_unsymmetric) {
        A->num_nonzeros = num_entries;
        A->size = num_rows*num_columns;
    } else if (num_rows == num_columns &&
               (symmetry == mtx_symmetric || symmetry == mtx_hermitian))
    {
        A->num_nonzeros = num_entries;
        A->size = num_rows*(num_columns+1)/2;
    } else if (num_rows == num_columns && symmetry == mtx_skew_symmetric) {
        A->num_nonzeros = num_entries-num_rows;
        A->size = num_rows*(num_columns-1)/2;
    } else { return MTX_ERR_INVALID_SYMMETRY; }
    return mtxblasvector_alloc(&A->a, field, precision, A->size);
}

static int mtxblasdense_init_entries_idx(
    struct mtxblasdense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx,
    int64_t * idx)
{
    for (int64_t k = 0; k < size; k++) {
        if (rowidx[k] < 0 || rowidx[k] >= num_rows ||
            colidx[k] < 0 || colidx[k] >= num_columns)
            return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    }

    if (symmetry == mtx_unsymmetric) {
        for (int64_t k = 0; k < size; k++)
            idx[k] = (int64_t) rowidx[k] * (int64_t) num_columns + (int64_t) colidx[k];
    } else if (num_rows == num_columns &&
               (symmetry == mtx_symmetric || symmetry == mtx_hermitian))
    {
        int64_t N = num_rows;
        for (int64_t k = 0; k < size; k++) {
            int64_t i = rowidx[k] < colidx[k] ? rowidx[k] : colidx[k];
            int64_t j = rowidx[k] < colidx[k] ? colidx[k] : rowidx[k];
            idx[k] = N*(N-1)/2 - (N-i)*(N-i-1)/2 + j;
        }
    } else if (num_rows == num_columns && symmetry == mtx_skew_symmetric) {
        int64_t N = num_rows;
        for (int64_t k = 0; k < size; k++) {
            int64_t i = rowidx[k] < colidx[k] ? rowidx[k] : colidx[k];
            int64_t j = rowidx[k] < colidx[k] ? colidx[k] : rowidx[k];
            idx[k] = N*(N-1)/2 - (N-i)*(N-i-1)/2 + j-i-1;
        }
    } else { return MTX_ERR_INVALID_SYMMETRY; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxblasdense_init_entries_real_single()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * real, single precision coefficients.
 */
int mtxblasdense_init_entries_real_single(
    struct mtxblasdense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx,
    const float * data)
{
    int err = mtxblasdense_alloc_entries(
        A, mtx_field_real, mtx_single, symmetry, num_rows, num_columns,
        size, 0, 0, NULL, NULL);
    if (err) return err;
    err = mtxblasdense_setzero(A);
    if (err) { mtxblasdense_free(A); return err; }
    int64_t * idx = malloc(size * sizeof(int64_t));
    if (!idx) { mtxblasdense_free(A); return MTX_ERR_ERRNO; }
    err = mtxblasdense_init_entries_idx(
        A, symmetry, num_rows, num_columns, size, rowidx, colidx, idx);
    if (err) { free(idx); mtxblasdense_free(A); return err; }
    struct mtxblasvector x;
    err = mtxblasvector_init_packed_real_single(
        &x, A->size, size, idx, data);
    if (err) { free(idx); mtxblasdense_free(A); return err; }
    free(idx);
    err = mtxblasvector_ussc(&A->a, &x);
    if (err) { mtxblasvector_free(&x); mtxblasdense_free(A); return err; }
    mtxblasvector_free(&x);
    return MTX_SUCCESS;
}

/**
 * ‘mtxblasdense_init_entries_real_double()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * real, double precision coefficients.
 */
int mtxblasdense_init_entries_real_double(
    struct mtxblasdense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx,
    const double * data)
{
    int err = mtxblasdense_alloc_entries(
        A, mtx_field_real, mtx_double, symmetry, num_rows, num_columns,
        size, 0, 0, NULL, NULL);
    if (err) return err;
    err = mtxblasdense_setzero(A);
    if (err) { mtxblasdense_free(A); return err; }
    int64_t * idx = malloc(size * sizeof(int64_t));
    if (!idx) { mtxblasdense_free(A); return MTX_ERR_ERRNO; }
    err = mtxblasdense_init_entries_idx(
        A, symmetry, num_rows, num_columns, size, rowidx, colidx, idx);
    if (err) { free(idx); mtxblasdense_free(A); return err; }
    struct mtxblasvector x;
    err = mtxblasvector_init_packed_real_double(
        &x, A->size, size, idx, data);
    if (err) { free(idx); mtxblasdense_free(A); return err; }
    free(idx);
    err = mtxblasvector_ussc(&A->a, &x);
    if (err) { mtxblasvector_free(&x); mtxblasdense_free(A); return err; }
    mtxblasvector_free(&x);
    return MTX_SUCCESS;
}

/**
 * ‘mtxblasdense_init_entries_complex_single()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * complex, single precision coefficients.
 */
int mtxblasdense_init_entries_complex_single(
    struct mtxblasdense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx,
    const float (* data)[2])
{
    int err = mtxblasdense_alloc_entries(
        A, mtx_field_complex, mtx_single, symmetry, num_rows, num_columns,
        size, 0, 0, NULL, NULL);
    if (err) return err;
    err = mtxblasdense_setzero(A);
    if (err) { mtxblasdense_free(A); return err; }
    int64_t * idx = malloc(size * sizeof(int64_t));
    if (!idx) { mtxblasdense_free(A); return MTX_ERR_ERRNO; }
    err = mtxblasdense_init_entries_idx(
        A, symmetry, num_rows, num_columns, size, rowidx, colidx, idx);
    if (err) { free(idx); mtxblasdense_free(A); return err; }
    struct mtxblasvector x;
    err = mtxblasvector_init_packed_complex_single(
        &x, A->size, size, idx, data);
    if (err) { free(idx); mtxblasdense_free(A); return err; }
    free(idx);
    err = mtxblasvector_ussc(&A->a, &x);
    if (err) { mtxblasvector_free(&x); mtxblasdense_free(A); return err; }
    mtxblasvector_free(&x);
    return MTX_SUCCESS;
}

/**
 * ‘mtxblasdense_init_entries_complex_double()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * complex, double precision coefficients.
 */
int mtxblasdense_init_entries_complex_double(
    struct mtxblasdense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx,
    const double (* data)[2])
{
    int err = mtxblasdense_alloc_entries(
        A, mtx_field_complex, mtx_double, symmetry, num_rows, num_columns,
        size, 0, 0, NULL, NULL);
    if (err) return err;
    err = mtxblasdense_setzero(A);
    if (err) { mtxblasdense_free(A); return err; }
    int64_t * idx = malloc(size * sizeof(int64_t));
    if (!idx) { mtxblasdense_free(A); return MTX_ERR_ERRNO; }
    err = mtxblasdense_init_entries_idx(
        A, symmetry, num_rows, num_columns, size, rowidx, colidx, idx);
    if (err) { free(idx); mtxblasdense_free(A); return err; }
    struct mtxblasvector x;
    err = mtxblasvector_init_packed_complex_double(
        &x, A->size, size, idx, data);
    if (err) { free(idx); mtxblasdense_free(A); return err; }
    free(idx);
    err = mtxblasvector_ussc(&A->a, &x);
    if (err) { mtxblasvector_free(&x); mtxblasdense_free(A); return err; }
    mtxblasvector_free(&x);
    return MTX_SUCCESS;
}

/**
 * ‘mtxblasdense_init_entries_integer_single()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * integer, single precision coefficients.
 */
int mtxblasdense_init_entries_integer_single(
    struct mtxblasdense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx,
    const int32_t * data)
{
    int err = mtxblasdense_alloc_entries(
        A, mtx_field_integer, mtx_single, symmetry, num_rows, num_columns,
        size, 0, 0, NULL, NULL);
    if (err) return err;
    err = mtxblasdense_setzero(A);
    if (err) { mtxblasdense_free(A); return err; }
    int64_t * idx = malloc(size * sizeof(int64_t));
    if (!idx) { mtxblasdense_free(A); return MTX_ERR_ERRNO; }
    err = mtxblasdense_init_entries_idx(
        A, symmetry, num_rows, num_columns, size, rowidx, colidx, idx);
    if (err) { free(idx); mtxblasdense_free(A); return err; }
    struct mtxblasvector x;
    err = mtxblasvector_init_packed_integer_single(
        &x, A->size, size, idx, data);
    if (err) { free(idx); mtxblasdense_free(A); return err; }
    free(idx);
    err = mtxblasvector_ussc(&A->a, &x);
    if (err) { mtxblasvector_free(&x); mtxblasdense_free(A); return err; }
    mtxblasvector_free(&x);
    return MTX_SUCCESS;
}

/**
 * ‘mtxblasdense_init_entries_integer_double()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * integer, double precision coefficients.
 */
int mtxblasdense_init_entries_integer_double(
    struct mtxblasdense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx,
    const int64_t * data)
{
    int err = mtxblasdense_alloc_entries(
        A, mtx_field_integer, mtx_double, symmetry,
        num_rows, num_columns, size, 0, 0, NULL, NULL);
    if (err) return err;
    err = mtxblasdense_setzero(A);
    if (err) { mtxblasdense_free(A); return err; }
    int64_t * idx = malloc(size * sizeof(int64_t));
    if (!idx) { mtxblasdense_free(A); return MTX_ERR_ERRNO; }
    err = mtxblasdense_init_entries_idx(
        A, symmetry, num_rows, num_columns, size, rowidx, colidx, idx);
    if (err) { free(idx); mtxblasdense_free(A); return err; }
    struct mtxblasvector x;
    err = mtxblasvector_init_packed_integer_double(
        &x, A->size, size, idx, data);
    if (err) { free(idx); mtxblasdense_free(A); return err; }
    free(idx);
    err = mtxblasvector_ussc(&A->a, &x);
    if (err) { mtxblasvector_free(&x); mtxblasdense_free(A); return err; }
    mtxblasvector_free(&x);
    return MTX_SUCCESS;
}

/**
 * ‘mtxblasdense_init_entries_pattern()’ allocates and initialises
 * a matrix from entrywise data in coordinate format with boolean
 * coefficients.
 */
int mtxblasdense_init_entries_pattern(
    struct mtxblasdense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx)
{
    return MTX_ERR_INVALID_FIELD;
}

/*
 * initialise matrices from entrywise data in coordinate format with
 * specified strides
 */

/**
 * ‘mtxblasdense_init_entries_strided_real_single()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * real, single precision coefficients.
 */
int mtxblasdense_init_entries_strided_real_single(
    struct mtxblasdense * A,
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
 * ‘mtxblasdense_init_entries_strided_real_double()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * real, double precision coefficients.
 */
int mtxblasdense_init_entries_strided_real_double(
    struct mtxblasdense * A,
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
 * ‘mtxblasdense_init_entries_strided_complex_single()’ allocates
 * and initialises a matrix from entrywise data in coordinate format
 * with complex, single precision coefficients.
 */
int mtxblasdense_init_entries_strided_complex_single(
    struct mtxblasdense * A,
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
 * ‘mtxblasdense_init_entries_strided_complex_double()’ allocates
 * and initialises a matrix from entrywise data in coordinate format
 * with complex, double precision coefficients.
 */
int mtxblasdense_init_entries_strided_complex_double(
    struct mtxblasdense * A,
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
 * ‘mtxblasdense_init_entries_strided_integer_single()’ allocates
 * and initialises a matrix from entrywise data in coordinate format
 * with integer, single precision coefficients.
 */
int mtxblasdense_init_entries_strided_integer_single(
    struct mtxblasdense * A,
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
 * ‘mtxblasdense_init_entries_strided_integer_double()’ allocates
 * and initialises a matrix from entrywise data in coordinate format
 * with integer, double precision coefficients.
 */
int mtxblasdense_init_entries_strided_integer_double(
    struct mtxblasdense * A,
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
 * ‘mtxblasdense_init_entries_strided_pattern()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * boolean coefficients.
 */
int mtxblasdense_init_entries_strided_pattern(
    struct mtxblasdense * A,
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
 * ‘mtxblasdense_alloc_rows()’ allocates a matrix from row-wise
 * data in compressed row format.
 */
int mtxblasdense_alloc_rows(
    struct mtxblasdense * A,
    enum mtxfield field,
    enum mtxprecision precision,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx);

/**
 * ‘mtxblasdense_init_rows_real_single()’ allocates and initialises
 * a matrix from row-wise data in compressed row format with real,
 * single precision coefficients.
 */
int mtxblasdense_init_rows_real_single(
    struct mtxblasdense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const float * data);

/**
 * ‘mtxblasdense_init_rows_real_double()’ allocates and initialises
 * a matrix from row-wise data in compressed row format with real,
 * double precision coefficients.
 */
int mtxblasdense_init_rows_real_double(
    struct mtxblasdense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const double * data);

/**
 * ‘mtxblasdense_init_rows_complex_single()’ allocates and
 * initialises a matrix from row-wise data in compressed row format
 * with complex, single precision coefficients.
 */
int mtxblasdense_init_rows_complex_single(
    struct mtxblasdense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const float (* data)[2]);

/**
 * ‘mtxblasdense_init_rows_complex_double()’ allocates and
 * initialises a matrix from row-wise data in compressed row format
 * with complex, double precision coefficients.
 */
int mtxblasdense_init_rows_complex_double(
    struct mtxblasdense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const double (* data)[2]);

/**
 * ‘mtxblasdense_init_rows_integer_single()’ allocates and
 * initialises a matrix from row-wise data in compressed row format
 * with integer, single precision coefficients.
 */
int mtxblasdense_init_rows_integer_single(
    struct mtxblasdense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const int32_t * data);

/**
 * ‘mtxblasdense_init_rows_integer_double()’ allocates and
 * initialises a matrix from row-wise data in compressed row format
 * with integer, double precision coefficients.
 */
int mtxblasdense_init_rows_integer_double(
    struct mtxblasdense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const int64_t * data);

/**
 * ‘mtxblasdense_init_rows_pattern()’ allocates and initialises a
 * matrix from row-wise data in compressed row format with boolean
 * coefficients.
 */
int mtxblasdense_init_rows_pattern(
    struct mtxblasdense * A,
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
 * ‘mtxblasdense_alloc_columns()’ allocates a matrix from
 * column-wise data in compressed column format.
 */
int mtxblasdense_alloc_columns(
    struct mtxblasdense * A,
    enum mtxfield field,
    enum mtxprecision precision,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx);

/**
 * ‘mtxblasdense_init_columns_real_single()’ allocates and
 * initialises a matrix from column-wise data in compressed column
 * format with real, single precision coefficients.
 */
int mtxblasdense_init_columns_real_single(
    struct mtxblasdense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx,
    const float * data);

/**
 * ‘mtxblasdense_init_columns_real_double()’ allocates and
 * initialises a matrix from column-wise data in compressed column
 * format with real, double precision coefficients.
 */
int mtxblasdense_init_columns_real_double(
    struct mtxblasdense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx,
    const double * data);

/**
 * ‘mtxblasdense_init_columns_complex_single()’ allocates and
 * initialises a matrix from column-wise data in compressed column
 * format with complex, single precision coefficients.
 */
int mtxblasdense_init_columns_complex_single(
    struct mtxblasdense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx,
    const float (* data)[2]);

/**
 * ‘mtxblasdense_init_columns_complex_double()’ allocates and
 * initialises a matrix from column-wise data in compressed column
 * format with complex, double precision coefficients.
 */
int mtxblasdense_init_columns_complex_double(
    struct mtxblasdense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx,
    const double (* data)[2]);

/**
 * ‘mtxblasdense_init_columns_integer_single()’ allocates and
 * initialises a matrix from column-wise data in compressed column
 * format with integer, single precision coefficients.
 */
int mtxblasdense_init_columns_integer_single(
    struct mtxblasdense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx,
    const int32_t * data);

/**
 * ‘mtxblasdense_init_columns_integer_double()’ allocates and
 * initialises a matrix from column-wise data in compressed column
 * format with integer, double precision coefficients.
 */
int mtxblasdense_init_columns_integer_double(
    struct mtxblasdense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx,
    const int64_t * data);

/**
 * ‘mtxblasdense_init_columns_pattern()’ allocates and initialises
 * a matrix from column-wise data in compressed column format with
 * boolean coefficients.
 */
int mtxblasdense_init_columns_pattern(
    struct mtxblasdense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx);

/*
 * initialise matrices from a list of dense cliques
 */

/**
 * ‘mtxblasdense_alloc_cliques()’ allocates a matrix from a list of
 * dense cliques.
 */
int mtxblasdense_alloc_cliques(
    struct mtxblasdense * A,
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
 * ‘mtxblasdense_init_cliques_real_single()’ allocates and
 * initialises a matrix from a list of dense cliques with real, single
 * precision coefficients.
 */
int mtxblasdense_init_cliques_real_single(
    struct mtxblasdense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_cliques,
    const int64_t * cliqueptr,
    const int * rowidx,
    const int * colidx,
    const float * data);

/**
 * ‘mtxblasdense_init_cliques_real_double()’ allocates and
 * initialises a matrix from a list of dense cliques with real, double
 * precision coefficients.
 */
int mtxblasdense_init_cliques_real_double(
    struct mtxblasdense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_cliques,
    const int64_t * cliqueptr,
    const int * rowidx,
    const int * colidx,
    const double * data);

/**
 * ‘mtxblasdense_init_cliques_complex_single()’ allocates and
 * initialises a matrix from a list of dense cliques with complex,
 * single precision coefficients.
 */
int mtxblasdense_init_cliques_complex_single(
    struct mtxblasdense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_cliques,
    const int64_t * cliqueptr,
    const int * rowidx,
    const int * colidx,
    const float (* data)[2]);

/**
 * ‘mtxblasdense_init_cliques_complex_double()’ allocates and
 * initialises a matrix from a list of dense cliques with complex,
 * double precision coefficients.
 */
int mtxblasdense_init_cliques_complex_double(
    struct mtxblasdense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_cliques,
    const int64_t * cliqueptr,
    const int * rowidx,
    const int * colidx,
    const double (* data)[2]);

/**
 * ‘mtxblasdense_init_cliques_integer_single()’ allocates and
 * initialises a matrix from a list of dense cliques with integer,
 * single precision coefficients.
 */
int mtxblasdense_init_cliques_integer_single(
    struct mtxblasdense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_cliques,
    const int64_t * cliqueptr,
    const int * rowidx,
    const int * colidx,
    const int32_t * data);

/**
 * ‘mtxblasdense_init_cliques_integer_double()’ allocates and
 * initialises a matrix from a list of dense cliques with integer,
 * double precision coefficients.
 */
int mtxblasdense_init_cliques_integer_double(
    struct mtxblasdense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_cliques,
    const int64_t * cliqueptr,
    const int * rowidx,
    const int * colidx,
    const int64_t * data);

/**
 * ‘mtxblasdense_init_cliques_pattern()’ allocates and initialises
 * a matrix from a list of dense cliques with boolean coefficients.
 */
int mtxblasdense_init_cliques_pattern(
    struct mtxblasdense * A,
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
 * ‘mtxblasdense_setzero()’ sets every value of a matrix to zero.
 */
int mtxblasdense_setzero(
    struct mtxblasdense * A)
{
    return mtxblasvector_setzero(&A->a);
}

/**
 * ‘mtxblasdense_set_real_single()’ sets values of a matrix based
 * on an array of single precision floating point numbers.
 */
int mtxblasdense_set_real_single(
    struct mtxblasdense * A,
    int64_t size,
    int stride,
    const float * a)
{
    return mtxblasvector_set_real_single(&A->a, size, stride, a);
}

/**
 * ‘mtxblasdense_set_real_double()’ sets values of a matrix based
 * on an array of double precision floating point numbers.
 */
int mtxblasdense_set_real_double(
    struct mtxblasdense * A,
    int64_t size,
    int stride,
    const double * a)
{
    return mtxblasvector_set_real_double(&A->a, size, stride, a);
}

/**
 * ‘mtxblasdense_set_complex_single()’ sets values of a matrix
 * based on an array of single precision floating point complex
 * numbers.
 */
int mtxblasdense_set_complex_single(
    struct mtxblasdense * A,
    int64_t size,
    int stride,
    const float (*a)[2])
{
    return mtxblasvector_set_complex_single(&A->a, size, stride, a);
}

/**
 * ‘mtxblasdense_set_complex_double()’ sets values of a matrix
 * based on an array of double precision floating point complex
 * numbers.
 */
int mtxblasdense_set_complex_double(
    struct mtxblasdense * A,
    int64_t size,
    int stride,
    const double (*a)[2])
{
    return mtxblasvector_set_complex_double(&A->a, size, stride, a);
}

/**
 * ‘mtxblasdense_set_integer_single()’ sets values of a matrix
 * based on an array of integers.
 */
int mtxblasdense_set_integer_single(
    struct mtxblasdense * A,
    int64_t size,
    int stride,
    const int32_t * a)
{
    return mtxblasvector_set_integer_single(&A->a, size, stride, a);
}

/**
 * ‘mtxblasdense_set_integer_double()’ sets values of a matrix
 * based on an array of integers.
 */
int mtxblasdense_set_integer_double(
    struct mtxblasdense * A,
    int64_t size,
    int stride,
    const int64_t * a)
{
    return mtxblasvector_set_integer_double(&A->a, size, stride, a);
}

/*
 * row and column vectors
 */

/**
 * ‘mtxblasdense_alloc_row_vector()’ allocates a row vector for a
 * given matrix, where a row vector is a vector whose length equal to
 * a single row of the matrix.
 */
int mtxblasdense_alloc_row_vector(
    const struct mtxblasdense * A,
    struct mtxvector * x,
    enum mtxvectortype vectortype)
{
    return mtxvector_alloc(
        x, vectortype, A->a.base.field, A->a.base.precision, A->num_columns);
}

/**
 * ‘mtxblasdense_alloc_column_vector()’ allocates a column vector
 * for a given matrix, where a column vector is a vector whose length
 * equal to a single column of the matrix.
 */
int mtxblasdense_alloc_column_vector(
    const struct mtxblasdense * A,
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
 * ‘mtxblasdense_from_mtxfile()’ converts a matrix from Matrix
 * Market format.
 */
int mtxblasdense_from_mtxfile(
    struct mtxblasdense * A,
    const struct mtxfile * mtxfile)
{
    int err;
    if (mtxfile->header.object != mtxfile_matrix)
        return MTX_ERR_INCOMPATIBLE_MTX_OBJECT;

    enum mtxfield field;
    err = mtxfilefield_to_mtxfield(&field, mtxfile->header.field);
    if (err) return err;
    enum mtxsymmetry symmetry;
    err = mtxfilesymmetry_to_mtxsymmetry(&symmetry, mtxfile->header.symmetry);
    if (err) return err;
    enum mtxprecision precision = mtxfile->precision;
    int num_rows = mtxfile->size.num_rows;
    int num_columns = mtxfile->size.num_columns;
    int64_t num_nonzeros = mtxfile->size.num_nonzeros;
    int64_t N = num_rows == num_columns ? num_rows : 0;

    err = mtxblasdense_alloc_entries(
        A, field, precision, symmetry, num_rows, num_columns,
        num_nonzeros, 0, 0, NULL, NULL);
    if (err) return err;
    err = mtxblasdense_setzero(A);
    if (err) { mtxblasdense_free(A); return err; }

    if (mtxfile->header.format == mtxfile_coordinate) {
        if (mtxfile->header.field == mtxfile_real) {
            if (mtxfile->precision == mtx_single) {
                const struct mtxfile_matrix_coordinate_real_single * data =
                    mtxfile->data.matrix_coordinate_real_single;
                float * a = A->a.base.data.real_single;
                if (symmetry == mtx_unsymmetric) {
                    for (int64_t k = 0; k < num_nonzeros; k++) {
                        int64_t i = data[k].i-1, j = data[k].j-1;
                        a[i*num_columns+j] = data[k].a;
                    }
                } else if (num_rows == num_columns && (symmetry == mtx_symmetric || symmetry == mtx_hermitian)) {
                    for (int64_t k = 0; k < num_nonzeros; k++) {
                        int64_t i = data[k].i < data[k].j ? data[k].i-1 : data[k].j-1;
                        int64_t j = data[k].i < data[k].j ? data[k].j-1 : data[k].i-1;
                        a[N*(N-1)/2 - (N-i)*(N-i-1)/2 + j] = data[k].a;
                    }
                } else if (num_rows == num_columns && symmetry == mtx_skew_symmetric) {
                    for (int64_t k = 0; k < num_nonzeros; k++) {
                        int64_t i = data[k].i < data[k].j ? data[k].i-1 : data[k].j-1;
                        int64_t j = data[k].i < data[k].j ? data[k].j-1 : data[k].i-1;
                        a[N*(N-1)/2 - (N-i)*(N-i-1)/2 + j-i-1] = data[k].a;
                    }
                } else { mtxblasdense_free(A); return MTX_ERR_INVALID_SYMMETRY; }
            } else if (mtxfile->precision == mtx_double) {
                const struct mtxfile_matrix_coordinate_real_double * data =
                    mtxfile->data.matrix_coordinate_real_double;
                double * a = A->a.base.data.real_double;
                if (symmetry == mtx_unsymmetric) {
                    for (int64_t k = 0; k < num_nonzeros; k++) {
                        int64_t i = data[k].i-1, j = data[k].j-1;
                        a[i*num_columns+j] = data[k].a;
                    }
                } else if (num_rows == num_columns && (symmetry == mtx_symmetric || symmetry == mtx_hermitian)) {
                    for (int64_t k = 0; k < num_nonzeros; k++) {
                        int64_t i = data[k].i < data[k].j ? data[k].i-1 : data[k].j-1;
                        int64_t j = data[k].i < data[k].j ? data[k].j-1 : data[k].i-1;
                        a[N*(N-1)/2 - (N-i)*(N-i-1)/2 + j] = data[k].a;
                    }
                } else if (num_rows == num_columns && symmetry == mtx_skew_symmetric) {
                    for (int64_t k = 0; k < num_nonzeros; k++) {
                        int64_t i = data[k].i < data[k].j ? data[k].i-1 : data[k].j-1;
                        int64_t j = data[k].i < data[k].j ? data[k].j-1 : data[k].i-1;
                        a[N*(N-1)/2 - (N-i)*(N-i-1)/2 + j-i-1] = data[k].a;
                    }
                } else { mtxblasdense_free(A); return MTX_ERR_INVALID_SYMMETRY; }
            } else { mtxblasdense_free(A); return MTX_ERR_INVALID_PRECISION; }
        } else if (mtxfile->header.field == mtxfile_complex) {
            if (mtxfile->precision == mtx_single) {
                const struct mtxfile_matrix_coordinate_complex_single * data =
                    mtxfile->data.matrix_coordinate_complex_single;
                float (*a)[2] = A->a.base.data.complex_single;
                if (symmetry == mtx_unsymmetric) {
                    for (int64_t k = 0; k < num_nonzeros; k++) {
                        int64_t i = data[k].i-1, j = data[k].j-1;
                        a[i*num_columns+j][0] = data[k].a[0];
                        a[i*num_columns+j][1] = data[k].a[1];
                    }
                } else if (num_rows == num_columns && (symmetry == mtx_symmetric || symmetry == mtx_hermitian)) {
                    for (int64_t k = 0; k < num_nonzeros; k++) {
                        int64_t i = data[k].i < data[k].j ? data[k].i-1 : data[k].j-1;
                        int64_t j = data[k].i < data[k].j ? data[k].j-1 : data[k].i-1;
                        a[N*(N-1)/2 - (N-i)*(N-i-1)/2 + j][0] = data[k].a[0];
                        a[N*(N-1)/2 - (N-i)*(N-i-1)/2 + j][1] = data[k].a[1];
                    }
                } else if (num_rows == num_columns && symmetry == mtx_skew_symmetric) {
                    for (int64_t k = 0; k < num_nonzeros; k++) {
                        int64_t i = data[k].i < data[k].j ? data[k].i-1 : data[k].j-1;
                        int64_t j = data[k].i < data[k].j ? data[k].j-1 : data[k].i-1;
                        a[N*(N-1)/2 - (N-i)*(N-i-1)/2 + j-i-1][0] = data[k].a[0];
                        a[N*(N-1)/2 - (N-i)*(N-i-1)/2 + j-i-1][1] = data[k].a[1];
                    }
                } else { mtxblasdense_free(A); return MTX_ERR_INVALID_SYMMETRY; }
            } else if (mtxfile->precision == mtx_double) {
                const struct mtxfile_matrix_coordinate_complex_double * data =
                    mtxfile->data.matrix_coordinate_complex_double;
                double (*a)[2] = A->a.base.data.complex_double;
                if (symmetry == mtx_unsymmetric) {
                    for (int64_t k = 0; k < num_nonzeros; k++) {
                        int64_t i = data[k].i-1, j = data[k].j-1;
                        a[i*num_columns+j][0] = data[k].a[0];
                        a[i*num_columns+j][1] = data[k].a[1];
                    }
                } else if (num_rows == num_columns && (symmetry == mtx_symmetric || symmetry == mtx_hermitian)) {
                    for (int64_t k = 0; k < num_nonzeros; k++) {
                        int64_t i = data[k].i < data[k].j ? data[k].i-1 : data[k].j-1;
                        int64_t j = data[k].i < data[k].j ? data[k].j-1 : data[k].i-1;
                        a[N*(N-1)/2 - (N-i)*(N-i-1)/2 + j][0] = data[k].a[0];
                        a[N*(N-1)/2 - (N-i)*(N-i-1)/2 + j][1] = data[k].a[1];
                    }
                } else if (num_rows == num_columns && symmetry == mtx_skew_symmetric) {
                    for (int64_t k = 0; k < num_nonzeros; k++) {
                        int64_t i = data[k].i < data[k].j ? data[k].i-1 : data[k].j-1;
                        int64_t j = data[k].i < data[k].j ? data[k].j-1 : data[k].i-1;
                        a[N*(N-1)/2 - (N-i)*(N-i-1)/2 + j-i-1][0] = data[k].a[0];
                        a[N*(N-1)/2 - (N-i)*(N-i-1)/2 + j-i-1][1] = data[k].a[1];
                    }
                } else { mtxblasdense_free(A); return MTX_ERR_INVALID_SYMMETRY; }
            } else { mtxblasdense_free(A); return MTX_ERR_INVALID_PRECISION; }
        } else if (mtxfile->header.field == mtxfile_integer) {
            if (mtxfile->precision == mtx_single) {
                const struct mtxfile_matrix_coordinate_integer_single * data =
                    mtxfile->data.matrix_coordinate_integer_single;
                int32_t * a = A->a.base.data.integer_single;
                if (symmetry == mtx_unsymmetric) {
                    for (int64_t k = 0; k < num_nonzeros; k++) {
                        int64_t i = data[k].i-1, j = data[k].j-1;
                        a[i*num_columns+j] = data[k].a;
                    }
                } else if (num_rows == num_columns && (symmetry == mtx_symmetric || symmetry == mtx_hermitian)) {
                    for (int64_t k = 0; k < num_nonzeros; k++) {
                        int64_t i = data[k].i < data[k].j ? data[k].i-1 : data[k].j-1;
                        int64_t j = data[k].i < data[k].j ? data[k].j-1 : data[k].i-1;
                        a[N*(N-1)/2 - (N-i)*(N-i-1)/2 + j] = data[k].a;
                    }
                } else if (num_rows == num_columns && symmetry == mtx_skew_symmetric) {
                    for (int64_t k = 0; k < num_nonzeros; k++) {
                        int64_t i = data[k].i < data[k].j ? data[k].i-1 : data[k].j-1;
                        int64_t j = data[k].i < data[k].j ? data[k].j-1 : data[k].i-1;
                        a[N*(N-1)/2 - (N-i)*(N-i-1)/2 + j-i-1] = data[k].a;
                    }
                } else { mtxblasdense_free(A); return MTX_ERR_INVALID_SYMMETRY; }
            } else if (mtxfile->precision == mtx_double) {
                const struct mtxfile_matrix_coordinate_integer_double * data =
                    mtxfile->data.matrix_coordinate_integer_double;
                int64_t * a = A->a.base.data.integer_double;
                if (symmetry == mtx_unsymmetric) {
                    for (int64_t k = 0; k < num_nonzeros; k++) {
                        int64_t i = data[k].i-1, j = data[k].j-1;
                        a[i*num_columns+j] = data[k].a;
                    }
                } else if (num_rows == num_columns && (symmetry == mtx_symmetric || symmetry == mtx_hermitian)) {
                    for (int64_t k = 0; k < num_nonzeros; k++) {
                        int64_t i = data[k].i < data[k].j ? data[k].i-1 : data[k].j-1;
                        int64_t j = data[k].i < data[k].j ? data[k].j-1 : data[k].i-1;
                        a[N*(N-1)/2 - (N-i)*(N-i-1)/2 + j] = data[k].a;
                    }
                } else if (num_rows == num_columns && symmetry == mtx_skew_symmetric) {
                    for (int64_t k = 0; k < num_nonzeros; k++) {
                        int64_t i = data[k].i < data[k].j ? data[k].i-1 : data[k].j-1;
                        int64_t j = data[k].i < data[k].j ? data[k].j-1 : data[k].i-1;
                        a[N*(N-1)/2 - (N-i)*(N-i-1)/2 + j-i-1] = data[k].a;
                    }
                } else { mtxblasdense_free(A); return MTX_ERR_INVALID_SYMMETRY; }
            } else { mtxblasdense_free(A); return MTX_ERR_INVALID_PRECISION; }
        } else { mtxblasdense_free(A); return MTX_ERR_INVALID_MTX_FIELD; }
    } else if (mtxfile->header.format == mtxfile_array) {
        if (mtxfile->header.field == mtxfile_real) {
            if (mtxfile->precision == mtx_single) {
                const float * data = mtxfile->data.array_real_single;
                float * a = A->a.base.data.real_single;
                if (symmetry == mtx_unsymmetric) {
                    for (int64_t i = 0, k = 0; i < num_rows; i++) {
                        for (int64_t j = 0; j < num_columns; j++, k++)
                            a[i*num_columns+j] = data[k];
                    }
                } else if (num_rows == num_columns && (symmetry == mtx_symmetric || symmetry == mtx_hermitian)) {
                    for (int64_t i = 0, k = 0; i < num_rows; i++) {
                        for (int64_t j = i; j < num_columns; j++, k++)
                            a[N*(N-1)/2 - (N-i)*(N-i-1)/2 + j] = data[k];
                    }
                } else if (num_rows == num_columns && symmetry == mtx_skew_symmetric) {
                    for (int64_t i = 0, k = 0; i < num_rows; i++) {
                        for (int64_t j = i+1; j < num_columns; j++, k++)
                            a[N*(N-1)/2 - (N-i)*(N-i-1)/2 + j-i-1] = data[k];
                    }
                } else { mtxblasdense_free(A); return MTX_ERR_INVALID_SYMMETRY; }
            } else if (mtxfile->precision == mtx_double) {
                const double * data = mtxfile->data.array_real_double;
                double * a = A->a.base.data.real_double;
                if (symmetry == mtx_unsymmetric) {
                    for (int64_t i = 0, k = 0; i < num_rows; i++) {
                        for (int64_t j = 0; j < num_columns; j++, k++)
                            a[i*num_columns+j] = data[k];
                    }
                } else if (num_rows == num_columns && (symmetry == mtx_symmetric || symmetry == mtx_hermitian)) {
                    for (int64_t i = 0, k = 0; i < num_rows; i++) {
                        for (int64_t j = i; j < num_columns; j++, k++)
                            a[N*(N-1)/2 - (N-i)*(N-i-1)/2 + j] = data[k];
                    }
                } else if (num_rows == num_columns && symmetry == mtx_skew_symmetric) {
                    for (int64_t i = 0, k = 0; i < num_rows; i++) {
                        for (int64_t j = i+1; j < num_columns; j++, k++)
                            a[N*(N-1)/2 - (N-i)*(N-i-1)/2 + j-i-1] = data[k];
                    }
                } else { mtxblasdense_free(A); return MTX_ERR_INVALID_SYMMETRY; }
            } else { mtxblasdense_free(A); return MTX_ERR_INVALID_PRECISION; }
        } else if (mtxfile->header.field == mtxfile_complex) {
            if (mtxfile->precision == mtx_single) {
                const float (*data)[2] = mtxfile->data.array_complex_single;
                float (*a)[2] = A->a.base.data.complex_single;
                if (symmetry == mtx_unsymmetric) {
                    for (int64_t i = 0, k = 0; i < num_rows; i++) {
                        for (int64_t j = 0; j < num_columns; j++, k++) {
                            a[i*num_columns+j][0] = data[k][0];
                            a[i*num_columns+j][1] = data[k][1];
                        }
                    }
                } else if (num_rows == num_columns && (symmetry == mtx_symmetric || symmetry == mtx_hermitian)) {
                    for (int64_t i = 0, k = 0; i < num_rows; i++) {
                        for (int64_t j = i; j < num_columns; j++, k++) {
                            a[N*(N-1)/2 - (N-i)*(N-i-1)/2 + j][0] = data[k][0];
                            a[N*(N-1)/2 - (N-i)*(N-i-1)/2 + j][1] = data[k][1];
                        }
                    }
                } else if (num_rows == num_columns && symmetry == mtx_skew_symmetric) {
                    for (int64_t i = 0, k = 0; i < num_rows; i++) {
                        for (int64_t j = i+1; j < num_columns; j++, k++) {
                            a[N*(N-1)/2 - (N-i)*(N-i-1)/2 + j-i-1][0] = data[k][0];
                            a[N*(N-1)/2 - (N-i)*(N-i-1)/2 + j-i-1][1] = data[k][1];
                        }
                    }
                } else { mtxblasdense_free(A); return MTX_ERR_INVALID_SYMMETRY; }
            } else if (mtxfile->precision == mtx_double) {
                const double (*data)[2] = mtxfile->data.array_complex_double;
                double (*a)[2] = A->a.base.data.complex_double;
                if (symmetry == mtx_unsymmetric) {
                    for (int64_t i = 0, k = 0; i < num_rows; i++) {
                        for (int64_t j = 0; j < num_columns; j++, k++) {
                            a[i*num_columns+j][0] = data[k][0];
                            a[i*num_columns+j][1] = data[k][1];
                        }
                    }
                } else if (num_rows == num_columns && (symmetry == mtx_symmetric || symmetry == mtx_hermitian)) {
                    for (int64_t i = 0, k = 0; i < num_rows; i++) {
                        for (int64_t j = i; j < num_columns; j++, k++) {
                            a[N*(N-1)/2 - (N-i)*(N-i-1)/2 + j][0] = data[k][0];
                            a[N*(N-1)/2 - (N-i)*(N-i-1)/2 + j][1] = data[k][1];
                        }
                    }
                } else if (num_rows == num_columns && symmetry == mtx_skew_symmetric) {
                    for (int64_t i = 0, k = 0; i < num_rows; i++) {
                        for (int64_t j = i+1; j < num_columns; j++, k++) {
                            a[N*(N-1)/2 - (N-i)*(N-i-1)/2 + j-i-1][0] = data[k][0];
                            a[N*(N-1)/2 - (N-i)*(N-i-1)/2 + j-i-1][1] = data[k][1];
                        }
                    }
                } else { mtxblasdense_free(A); return MTX_ERR_INVALID_SYMMETRY; }
            } else { mtxblasdense_free(A); return MTX_ERR_INVALID_PRECISION; }
        } else if (mtxfile->header.field == mtxfile_integer) {
            if (mtxfile->precision == mtx_single) {
                const int32_t * data = mtxfile->data.array_integer_single;
                int32_t * a = A->a.base.data.integer_single;
                if (symmetry == mtx_unsymmetric) {
                    for (int64_t i = 0, k = 0; i < num_rows; i++) {
                        for (int64_t j = 0; j < num_columns; j++, k++)
                            a[i*num_columns+j] = data[k];
                    }
                } else if (num_rows == num_columns && (symmetry == mtx_symmetric || symmetry == mtx_hermitian)) {
                    for (int64_t i = 0, k = 0; i < num_rows; i++) {
                        for (int64_t j = i; j < num_columns; j++, k++)
                            a[N*(N-1)/2 - (N-i)*(N-i-1)/2 + j] = data[k];
                    }
                } else if (num_rows == num_columns && symmetry == mtx_skew_symmetric) {
                    for (int64_t i = 0, k = 0; i < num_rows; i++) {
                        for (int64_t j = i+1; j < num_columns; j++, k++)
                            a[N*(N-1)/2 - (N-i)*(N-i-1)/2 + j-i-1] = data[k];
                    }
                } else { mtxblasdense_free(A); return MTX_ERR_INVALID_SYMMETRY; }
            } else if (mtxfile->precision == mtx_double) {
                const int64_t * data = mtxfile->data.array_integer_double;
                int64_t * a = A->a.base.data.integer_double;
                if (symmetry == mtx_unsymmetric) {
                    for (int64_t i = 0, k = 0; i < num_rows; i++) {
                        for (int64_t j = 0; j < num_columns; j++, k++)
                            a[i*num_columns+j] = data[k];
                    }
                } else if (num_rows == num_columns && (symmetry == mtx_symmetric || symmetry == mtx_hermitian)) {
                    for (int64_t i = 0, k = 0; i < num_rows; i++) {
                        for (int64_t j = i; j < num_columns; j++, k++)
                            a[N*(N-1)/2 - (N-i)*(N-i-1)/2 + j] = data[k];
                    }
                } else if (num_rows == num_columns && symmetry == mtx_skew_symmetric) {
                    for (int64_t i = 0, k = 0; i < num_rows; i++) {
                        for (int64_t j = i+1; j < num_columns; j++, k++)
                            a[N*(N-1)/2 - (N-i)*(N-i-1)/2 + j-i-1] = data[k];
                    }
                } else { mtxblasdense_free(A); return MTX_ERR_INVALID_SYMMETRY; }
            } else { mtxblasdense_free(A); return MTX_ERR_INVALID_PRECISION; }
        } else { mtxblasdense_free(A); return MTX_ERR_INVALID_MTX_FIELD; }
    } else { mtxblasdense_free(A); return MTX_ERR_INVALID_MTX_FORMAT; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxblasdense_to_mtxfile()’ converts a matrix to Matrix Market
 * format.
 */
int mtxblasdense_to_mtxfile(
    struct mtxfile * mtxfile,
    const struct mtxblasdense * A,
    int64_t num_rows,
    const int64_t * rowidx,
    int64_t num_columns,
    const int64_t * colidx,
    enum mtxfileformat mtxfmt)
{
    int err;
    enum mtxsymmetry symmetry = A->symmetry;
    enum mtxfilesymmetry mtxsymmetry;
    err = mtxfilesymmetry_from_mtxsymmetry(&mtxsymmetry, symmetry);
    if (err) return err;
    enum mtxfield field = A->a.base.field;
    enum mtxfilefield mtxfield;
    err = mtxfilefield_from_mtxfield(&mtxfield, field);
    if (err) return err;
    enum mtxprecision precision = A->a.base.precision;

    if (mtxfmt == mtxfile_coordinate) {
        err = mtxfile_alloc_matrix_coordinate(
            mtxfile, mtxfield, mtxsymmetry, precision,
            rowidx ? num_rows : A->num_rows,
            colidx ? num_columns : A->num_columns,
            A->size);
        if (err) return err;
        if (field == mtx_field_real) {
            if (precision == mtx_single) {
                struct mtxfile_matrix_coordinate_real_single * data =
                    mtxfile->data.matrix_coordinate_real_single;
                const float * a = A->a.base.data.real_single;
                if (symmetry == mtx_unsymmetric) {
                    for (int64_t i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = 0; j < A->num_columns; j++, k++) {
                            data[k].i = rowidx ? rowidx[i]+1 : i+1;
                            data[k].j = colidx ? colidx[j]+1 : j+1;
                            data[k].a = a[k];
                        }
                    }
                } else if (num_rows == num_columns && (symmetry == mtx_symmetric || symmetry == mtx_hermitian)) {
                    for (int64_t i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = i; j < A->num_columns; j++, k++) {
                            data[k].i = rowidx ? rowidx[i]+1 : i+1;
                            data[k].j = colidx ? colidx[j]+1 : j+1;
                            data[k].a = a[k];
                        }
                    }
                } else if (num_rows == num_columns && symmetry == mtx_skew_symmetric) {
                    for (int64_t i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = i+1; j < A->num_columns; j++, k++) {
                            data[k].i = rowidx ? rowidx[i]+1 : i+1;
                            data[k].j = colidx ? colidx[j]+1 : j+1;
                            data[k].a = a[k];
                        }
                    }
                } else { mtxfile_free(mtxfile); return MTX_ERR_INVALID_SYMMETRY; }
            } else if (precision == mtx_double) {
                struct mtxfile_matrix_coordinate_real_double * data =
                    mtxfile->data.matrix_coordinate_real_double;
                const double * a = A->a.base.data.real_double;
                if (symmetry == mtx_unsymmetric) {
                    for (int64_t i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = 0; j < A->num_columns; j++, k++) {
                            data[k].i = rowidx ? rowidx[i]+1 : i+1;
                            data[k].j = colidx ? colidx[j]+1 : j+1;
                            data[k].a = a[k];
                        }
                    }
                } else if (num_rows == num_columns && (symmetry == mtx_symmetric || symmetry == mtx_hermitian)) {
                    for (int64_t i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = i; j < A->num_columns; j++, k++) {
                            data[k].i = rowidx ? rowidx[i]+1 : i+1;
                            data[k].j = colidx ? colidx[j]+1 : j+1;
                            data[k].a = a[k];
                        }
                    }
                } else if (num_rows == num_columns && symmetry == mtx_skew_symmetric) {
                    for (int64_t i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = i+1; j < A->num_columns; j++, k++) {
                            data[k].i = rowidx ? rowidx[i]+1 : i+1;
                            data[k].j = colidx ? colidx[j]+1 : j+1;
                            data[k].a = a[k];
                        }
                    }
                } else { mtxfile_free(mtxfile); return MTX_ERR_INVALID_SYMMETRY; }
            } else { mtxfile_free(mtxfile); return MTX_ERR_INVALID_PRECISION; }
        } else if (field == mtx_field_complex) {
            if (precision == mtx_single) {
                struct mtxfile_matrix_coordinate_complex_single * data =
                    mtxfile->data.matrix_coordinate_complex_single;
                const float (* a)[2] = A->a.base.data.complex_single;
                if (symmetry == mtx_unsymmetric) {
                    for (int64_t i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = 0; j < A->num_columns; j++, k++) {
                            data[k].i = rowidx ? rowidx[i]+1 : i+1;
                            data[k].j = colidx ? colidx[j]+1 : j+1;
                            data[k].a[0] = a[k][0]; data[k].a[1] = a[k][1];
                        }
                    }
                } else if (num_rows == num_columns && (symmetry == mtx_symmetric || symmetry == mtx_hermitian)) {
                    for (int64_t i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = i; j < A->num_columns; j++, k++) {
                            data[k].i = rowidx ? rowidx[i]+1 : i+1;
                            data[k].j = colidx ? colidx[j]+1 : j+1;
                            data[k].a[0] = a[k][0]; data[k].a[1] = a[k][1];
                        }
                    }
                } else if (num_rows == num_columns && symmetry == mtx_skew_symmetric) {
                    for (int64_t i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = i+1; j < A->num_columns; j++, k++) {
                            data[k].i = rowidx ? rowidx[i]+1 : i+1;
                            data[k].j = colidx ? colidx[j]+1 : j+1;
                            data[k].a[0] = a[k][0]; data[k].a[1] = a[k][1];
                        }
                    }
                } else { mtxfile_free(mtxfile); return MTX_ERR_INVALID_SYMMETRY; }
            } else if (precision == mtx_double) {
                struct mtxfile_matrix_coordinate_complex_double * data =
                    mtxfile->data.matrix_coordinate_complex_double;
                const double (* a)[2] = A->a.base.data.complex_double;
                if (symmetry == mtx_unsymmetric) {
                    for (int64_t i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = 0; j < A->num_columns; j++, k++) {
                            data[k].i = rowidx ? rowidx[i]+1 : i+1;
                            data[k].j = colidx ? colidx[j]+1 : j+1;
                            data[k].a[0] = a[k][0]; data[k].a[1] = a[k][1];
                        }
                    }
                } else if (num_rows == num_columns && (symmetry == mtx_symmetric || symmetry == mtx_hermitian)) {
                    for (int64_t i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = i; j < A->num_columns; j++, k++) {
                            data[k].i = rowidx ? rowidx[i]+1 : i+1;
                            data[k].j = colidx ? colidx[j]+1 : j+1;
                            data[k].a[0] = a[k][0]; data[k].a[1] = a[k][1];
                        }
                    }
                } else if (num_rows == num_columns && symmetry == mtx_skew_symmetric) {
                    for (int64_t i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = i+1; j < A->num_columns; j++, k++) {
                            data[k].i = rowidx ? rowidx[i]+1 : i+1;
                            data[k].j = colidx ? colidx[j]+1 : j+1;
                            data[k].a[0] = a[k][0]; data[k].a[1] = a[k][1];
                        }
                    }
                } else { mtxfile_free(mtxfile); return MTX_ERR_INVALID_SYMMETRY; }
            } else { mtxfile_free(mtxfile); return MTX_ERR_INVALID_PRECISION; }
        } else if (field == mtx_field_integer) {
            if (precision == mtx_single) {
                struct mtxfile_matrix_coordinate_integer_single * data =
                    mtxfile->data.matrix_coordinate_integer_single;
                const int32_t * a = A->a.base.data.integer_single;
                if (symmetry == mtx_unsymmetric) {
                    for (int64_t i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = 0; j < A->num_columns; j++, k++) {
                            data[k].i = rowidx ? rowidx[i]+1 : i+1;
                            data[k].j = colidx ? colidx[j]+1 : j+1;
                            data[k].a = a[k];
                        }
                    }
                } else if (num_rows == num_columns && (symmetry == mtx_symmetric || symmetry == mtx_hermitian)) {
                    for (int64_t i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = i; j < A->num_columns; j++, k++) {
                            data[k].i = rowidx ? rowidx[i]+1 : i+1;
                            data[k].j = colidx ? colidx[j]+1 : j+1;
                            data[k].a = a[k];
                        }
                    }
                } else if (num_rows == num_columns && symmetry == mtx_skew_symmetric) {
                    for (int64_t i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = i+1; j < A->num_columns; j++, k++) {
                            data[k].i = rowidx ? rowidx[i]+1 : i+1;
                            data[k].j = colidx ? colidx[j]+1 : j+1;
                            data[k].a = a[k];
                        }
                    }
                } else { mtxfile_free(mtxfile); return MTX_ERR_INVALID_SYMMETRY; }
            } else if (precision == mtx_double) {
                struct mtxfile_matrix_coordinate_integer_double * data =
                    mtxfile->data.matrix_coordinate_integer_double;
                const int64_t * a = A->a.base.data.integer_double;
                if (symmetry == mtx_unsymmetric) {
                    for (int64_t i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = 0; j < A->num_columns; j++, k++) {
                            data[k].i = rowidx ? rowidx[i]+1 : i+1;
                            data[k].j = colidx ? colidx[j]+1 : j+1;
                            data[k].a = a[k];
                        }
                    }
                } else if (num_rows == num_columns && (symmetry == mtx_symmetric || symmetry == mtx_hermitian)) {
                    for (int64_t i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = i; j < A->num_columns; j++, k++) {
                            data[k].i = rowidx ? rowidx[i]+1 : i+1;
                            data[k].j = colidx ? colidx[j]+1 : j+1;
                            data[k].a = a[k];
                        }
                    }
                } else if (num_rows == num_columns && symmetry == mtx_skew_symmetric) {
                    for (int64_t i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = i+1; j < A->num_columns; j++, k++) {
                            data[k].i = rowidx ? rowidx[i]+1 : i+1;
                            data[k].j = colidx ? colidx[j]+1 : j+1;
                            data[k].a = a[k];
                        }
                    }
                } else { mtxfile_free(mtxfile); return MTX_ERR_INVALID_SYMMETRY; }
            } else { mtxfile_free(mtxfile); return MTX_ERR_INVALID_PRECISION; }
        } else { mtxfile_free(mtxfile); return MTX_ERR_INVALID_FIELD; }
    } else if (mtxfmt == mtxfile_array) {
        err = mtxfile_alloc_matrix_array(
            mtxfile, mtxfield, mtxsymmetry, precision,
            A->num_rows, A->num_columns);
        if (err) return err;
        if (field == mtx_field_real) {
            if (precision == mtx_single) {
                float * data = mtxfile->data.array_real_single;
                const float * a = A->a.base.data.real_single;
                for (int64_t k = 0; k < A->size; k++) data[k] = a[k];
            } else if (precision == mtx_double) {
                double * data = mtxfile->data.array_real_double;
                const double * a = A->a.base.data.real_double;
                for (int64_t k = 0; k < A->size; k++) data[k] = a[k];
            } else { mtxfile_free(mtxfile); return MTX_ERR_INVALID_PRECISION; }
        } else if (field == mtx_field_complex) {
            if (precision == mtx_single) {
                float (* data)[2] = mtxfile->data.array_complex_single;
                const float (* a)[2] = A->a.base.data.complex_single;
                for (int64_t k = 0; k < A->size; k++) { data[k][0] = a[k][0]; data[k][1] = a[k][1]; }
            } else if (precision == mtx_double) {
                double (* data)[2] = mtxfile->data.array_complex_double;
                const double (* a)[2] = A->a.base.data.complex_double;
                for (int64_t k = 0; k < A->size; k++) { data[k][0] = a[k][0]; data[k][1] = a[k][1]; }
            } else { mtxfile_free(mtxfile); return MTX_ERR_INVALID_PRECISION; }
        } else if (field == mtx_field_integer) {
            if (precision == mtx_single) {
                int32_t * data = mtxfile->data.array_integer_single;
                const int32_t * a = A->a.base.data.integer_single;
                for (int64_t k = 0; k < A->size; k++) data[k] = a[k];
            } else if (precision == mtx_double) {
                int64_t * data = mtxfile->data.array_integer_double;
                const int64_t * a = A->a.base.data.integer_double;
                for (int64_t k = 0; k < A->size; k++) data[k] = a[k];
            } else { mtxfile_free(mtxfile); return MTX_ERR_INVALID_PRECISION; }
        } else { mtxfile_free(mtxfile); return MTX_ERR_INVALID_FIELD; }
    } else { return MTX_ERR_INVALID_MTX_FORMAT; }
    return MTX_SUCCESS;
}

/*
 * Level 1 BLAS operations
 */

/**
 * ‘mtxblasdense_swap()’ swaps values of two matrices,
 * simultaneously performing ‘y <- x’ and ‘x <- y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxblasdense_swap(
    struct mtxblasdense * x,
    struct mtxblasdense * y)
{
    return mtxblasvector_swap(&x->a, &y->a);
}

/**
 * ‘mtxblasdense_copy()’ copies values of a matrix, ‘y = x’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxblasdense_copy(
    struct mtxblasdense * y,
    const struct mtxblasdense * x)
{
    return mtxblasvector_copy(&y->a, &x->a);
}

/**
 * ‘mtxblasdense_sscal()’ scales a matrix by a single precision
 * floating point scalar, ‘x = a*x’.
 */
int mtxblasdense_sscal(
    float a,
    struct mtxblasdense * x,
    int64_t * num_flops)
{
    return mtxblasvector_sscal(a, &x->a, num_flops);
}

/**
 * ‘mtxblasdense_dscal()’ scales a matrix by a double precision
 * floating point scalar, ‘x = a*x’.
 */
int mtxblasdense_dscal(
    double a,
    struct mtxblasdense * x,
    int64_t * num_flops)
{
    return mtxblasvector_dscal(a, &x->a, num_flops);
}

/**
 * ‘mtxblasdense_cscal()’ scales a matrix by a complex, single
 * precision floating point scalar, ‘x = (a+b*i)*x’.
 */
int mtxblasdense_cscal(
    float a[2],
    struct mtxblasdense * x,
    int64_t * num_flops)
{
    return mtxblasvector_cscal(a, &x->a, num_flops);
}

/**
 * ‘mtxblasdense_zscal()’ scales a matrix by a complex, double
 * precision floating point scalar, ‘x = (a+b*i)*x’.
 */
int mtxblasdense_zscal(
    double a[2],
    struct mtxblasdense * x,
    int64_t * num_flops)
{
    return mtxblasvector_zscal(a, &x->a, num_flops);
}

/**
 * ‘mtxblasdense_saxpy()’ adds a matrix to another one multiplied
 * by a single precision floating point value, ‘y = a*x + y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxblasdense_saxpy(
    float a,
    const struct mtxblasdense * x,
    struct mtxblasdense * y,
    int64_t * num_flops)
{
    return mtxblasvector_saxpy(a, &x->a, &y->a, num_flops);
}

/**
 * ‘mtxblasdense_daxpy()’ adds a matrix to another one multiplied
 * by a double precision floating point value, ‘y = a*x + y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxblasdense_daxpy(
    double a,
    const struct mtxblasdense * x,
    struct mtxblasdense * y,
    int64_t * num_flops)
{
    return mtxblasvector_daxpy(a, &x->a, &y->a, num_flops);
}

/**
 * ‘mtxblasdense_saypx()’ multiplies a matrix by a single precision
 * floating point scalar and adds another matrix, ‘y = a*y + x’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxblasdense_saypx(
    float a,
    struct mtxblasdense * y,
    const struct mtxblasdense * x,
    int64_t * num_flops)
{
    return mtxblasvector_saypx(a, &y->a, &x->a, num_flops);
}

/**
 * ‘mtxblasdense_daypx()’ multiplies a matrix by a double precision
 * floating point scalar and adds another matrix, ‘y = a*y + x’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxblasdense_daypx(
    double a,
    struct mtxblasdense * y,
    const struct mtxblasdense * x,
    int64_t * num_flops)
{
    return mtxblasvector_daypx(a, &y->a, &x->a, num_flops);
}

/**
 * ‘mtxblasdense_sdot()’ computes the Frobenius inner product of
 * two matrices in single precision floating point.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxblasdense_sdot(
    const struct mtxblasdense * x,
    const struct mtxblasdense * y,
    float * dot,
    int64_t * num_flops)
{
    return mtxblasvector_sdot(&x->a, &y->a, dot, num_flops);
}

/**
 * ‘mtxblasdense_ddot()’ computes the Frobenius inner product of
 * two matrices in double precision floating point.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxblasdense_ddot(
    const struct mtxblasdense * x,
    const struct mtxblasdense * y,
    double * dot,
    int64_t * num_flops)
{
    return mtxblasvector_ddot(&x->a, &y->a, dot, num_flops);
}

/**
 * ‘mtxblasdense_cdotu()’ computes the product of the transpose of
 * a complex row matrix with another complex row matrix in single
 * precision floating point, ‘dot := x^T*y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxblasdense_cdotu(
    const struct mtxblasdense * x,
    const struct mtxblasdense * y,
    float (* dot)[2],
    int64_t * num_flops)
{
    return mtxblasvector_cdotu(&x->a, &y->a, dot, num_flops);
}

/**
 * ‘mtxblasdense_zdotu()’ computes the product of the transpose of
 * a complex row matrix with another complex row matrix in double
 * precision floating point, ‘dot := x^T*y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxblasdense_zdotu(
    const struct mtxblasdense * x,
    const struct mtxblasdense * y,
    double (* dot)[2],
    int64_t * num_flops)
{
    return mtxblasvector_zdotu(&x->a, &y->a, dot, num_flops);
}

/**
 * ‘mtxblasdense_cdotc()’ computes the Frobenius inner product of
 * two complex matrices in single precision floating point, ‘dot :=
 * x^H*y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxblasdense_cdotc(
    const struct mtxblasdense * x,
    const struct mtxblasdense * y,
    float (* dot)[2],
    int64_t * num_flops)
{
    return mtxblasvector_cdotc(&x->a, &y->a, dot, num_flops);
}

/**
 * ‘mtxblasdense_zdotc()’ computes the Frobenius inner product of
 * two complex matrices in double precision floating point, ‘dot :=
 * x^H*y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxblasdense_zdotc(
    const struct mtxblasdense * x,
    const struct mtxblasdense * y,
    double (* dot)[2],
    int64_t * num_flops)
{
    return mtxblasvector_zdotc(&x->a, &y->a, dot, num_flops);
}

/**
 * ‘mtxblasdense_snrm2()’ computes the Frobenius norm of a matrix
 * in single precision floating point.
 */
int mtxblasdense_snrm2(
    const struct mtxblasdense * x,
    float * nrm2,
    int64_t * num_flops)
{
    return mtxblasvector_snrm2(&x->a, nrm2, num_flops);
}

/**
 * ‘mtxblasdense_dnrm2()’ computes the Frobenius norm of a matrix
 * in double precision floating point.
 */
int mtxblasdense_dnrm2(
    const struct mtxblasdense * x,
    double * nrm2,
    int64_t * num_flops)
{
    return mtxblasvector_dnrm2(&x->a, nrm2, num_flops);
}

/**
 * ‘mtxblasdense_sasum()’ computes the sum of absolute values
 * (1-norm) of a matrix in single precision floating point.  If the
 * matrix is complex-valued, then the sum of the absolute values of
 * the real and imaginary parts is computed.
 */
int mtxblasdense_sasum(
    const struct mtxblasdense * x,
    float * asum,
    int64_t * num_flops)
{
    return mtxblasvector_sasum(&x->a, asum, num_flops);
}

/**
 * ‘mtxblasdense_dasum()’ computes the sum of absolute values
 * (1-norm) of a matrix in double precision floating point.  If the
 * matrix is complex-valued, then the sum of the absolute values of
 * the real and imaginary parts is computed.
 */
int mtxblasdense_dasum(
    const struct mtxblasdense * x,
    double * asum,
    int64_t * num_flops)
{
    return mtxblasvector_dasum(&x->a, asum, num_flops);
}

/**
 * ‘mtxblasdense_iamax()’ finds the index of the first element
 * having the maximum absolute value.  If the matrix is
 * complex-valued, then the index points to the first element having
 * the maximum sum of the absolute values of the real and imaginary
 * parts.
 */
int mtxblasdense_iamax(
    const struct mtxblasdense * x,
    int * iamax)
{
    return mtxblasvector_iamax(&x->a, iamax);
}

/*
 * Level 2 BLAS operations (matrix-vector)
 */

#ifdef LIBMTX_HAVE_BLAS
enum CBLAS_TRANSPOSE mtxtransposition_to_cblas(
    enum mtxtransposition trans)
{
    if (trans == mtx_notrans) return CblasNoTrans;
    else if (trans == mtx_trans) return CblasTrans;
    else if (trans == mtx_conjtrans) return CblasConjTrans;
    else return -1;
}

/*
 * The operation counts below are taken from “Appendix C Operation
 * Counts for the BLAS and LAPACK” in Installation Guide for LAPACK by
 * Susan Blackford and Jack Dongarra, UT-CS-92-151, March, 1992.
 * Updated: June 30, 1999 (VERSION 3.0).
 */

static int64_t cblas_sgemv_num_flops(
    int64_t m, int64_t n, float alpha, float beta)
{
    return 2*m*n
        + (alpha == 1 || alpha == -1 ? 0 : m)
        + (beta == 1 || beta == -1 || beta == 0 ? 0 : m);
}

static int64_t cblas_sspmv_num_flops(
    int64_t n, float alpha, float beta)
{
    return 2*n*n
        + (alpha == 1 || alpha == -1 ? 0 : n)
        + (beta == 1 || beta == -1 || beta == 0 ? 0 : n);
}

static int64_t cblas_dgemv_num_flops(
    int64_t m, int64_t n, double alpha, double beta)
{
    return 2*m*n
        + (alpha == 1 || alpha == -1 ? 0 : m)
        + (beta == 1 || beta == -1 || beta == 0 ? 0 : m);
}

static int64_t cblas_dspmv_num_flops(
    int64_t n, double alpha, double beta)
{
    return 2*n*n
        + (alpha == 1 || alpha == -1 ? 0 : n)
        + (beta == 1 || beta == -1 || beta == 0 ? 0 : n);
}

static int64_t cblas_cgemv_num_flops(
    int64_t m, int64_t n, const float alpha[2], const float beta[2])
{
    return 8*m*n
        + ((alpha[0] == 1 && alpha[1] == 0) ||
           (alpha[0] == -1 && alpha[1] == 0) ? 0 : 6*m)
        + ((beta[0] == 1 && beta[1] == 0) ||
           (beta[0] == -1 && beta[1] == 0) ||
           (beta[0] == 0 && beta[1] == 0) ? 0 : 6*m);
}

static int64_t cblas_chpmv_num_flops(
    int64_t n, const float alpha[2], const float beta[2])
{
    return 8*n*n
        + ((alpha[0] == 1 && alpha[1] == 0) ||
           (alpha[0] == -1 && alpha[1] == 0) ? 0 : 6*n)
        + ((beta[0] == 1 && beta[1] == 0) ||
           (beta[0] == -1 && beta[1] == 0) ||
           (beta[0] == 0 && beta[1] == 0) ? 0 : 6*n);
}

static int64_t cblas_zgemv_num_flops(
    int64_t m, int64_t n, const double alpha[2], const double beta[2])
{
    return 8*m*n
        + ((alpha[0] == 1 && alpha[1] == 0) ||
           (alpha[0] == -1 && alpha[1] == 0) ? 0 : 6*m)
        + ((beta[0] == 1 && beta[1] == 0) ||
           (beta[0] == -1 && beta[1] == 0) ||
           (beta[0] == 0 && beta[1] == 0) ? 0 : 6*m);
}

static int64_t cblas_zhpmv_num_flops(
    int64_t n, const double alpha[2], const double beta[2])
{
    return 8*n*n
        + ((alpha[0] == 1 && alpha[1] == 0) ||
           (alpha[0] == -1 && alpha[1] == 0) ? 0 : 6*n)
        + ((beta[0] == 1 && beta[1] == 0) ||
           (beta[0] == -1 && beta[1] == 0) ||
           (beta[0] == 0 && beta[1] == 0) ? 0 : 6*n);
}

static int64_t cblas_sgemm_num_flops(
    int64_t m, int64_t n, int64_t k, float alpha, float beta)
{
    return 2*m*k*n
        + (beta == 1 || beta == -1 || beta == 0 ? 0 : m);
}
#endif

/**
 * ‘mtxblasdense_sgemv()’ multiplies a matrix ‘A’ or its transpose
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
int mtxblasdense_sgemv(
    enum mtxtransposition trans,
    float alpha,
    const struct mtxblasdense * A,
    const struct mtxvector * x,
    float beta,
    struct mtxvector * y,
    int64_t * num_flops)
{
#ifdef LIBMTX_HAVE_BLAS
    const struct mtxbasevector * a = &A->a.base;
    if (x->type != mtxbasevector || y->type != mtxbasevector)
        return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
    const struct mtxbasevector * xbase = &x->storage.base;
    struct mtxbasevector * ybase = &y->storage.base;
    if (xbase->field != a->field || ybase->field != a->field)
        return MTX_ERR_INCOMPATIBLE_FIELD;
    if (xbase->precision != a->precision || ybase->precision != a->precision)
        return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (trans == mtx_notrans &&
        (A->num_rows != ybase->size || A->num_columns != xbase->size))
        return MTX_ERR_INCOMPATIBLE_SIZE;
    if ((trans == mtx_trans || trans == mtx_conjtrans) &&
        (A->num_columns != ybase->size || A->num_rows != xbase->size))
        return MTX_ERR_INCOMPATIBLE_SIZE;

    if (A->symmetry == mtx_unsymmetric) {
        if (a->field == mtx_field_real) {
            if (trans == mtx_notrans) {
                if (a->precision == mtx_single) {
                    const float * Adata = a->data.real_single;
                    const float * xdata = xbase->data.real_single;
                    float * ydata = ybase->data.real_single;
                    cblas_sgemv(
                        CblasRowMajor, CblasNoTrans, A->num_rows, A->num_columns,
                        alpha, Adata, A->num_columns, xdata, 1, beta, ydata, 1);
                    if (mtxblaserror()) return MTX_ERR_BLAS;
                    if (num_flops) *num_flops += cblas_sgemv_num_flops(
                        A->num_rows, A->num_columns, alpha, 1);
                } else if (a->precision == mtx_double) {
                    const double * Adata = a->data.real_double;
                    const double * xdata = xbase->data.real_double;
                    double * ydata = ybase->data.real_double;
                    cblas_dgemv(
                        CblasRowMajor, CblasNoTrans, A->num_rows, A->num_columns,
                        alpha, Adata, A->num_columns, xdata, 1, beta, ydata, 1);
                    if (mtxblaserror()) return MTX_ERR_BLAS;
                    if (num_flops) *num_flops += cblas_dgemv_num_flops(
                        A->num_rows, A->num_columns, alpha, 1);
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (trans == mtx_trans || trans == mtx_conjtrans) {
                if (a->precision == mtx_single) {
                    const float * Adata = a->data.real_single;
                    const float * xdata = xbase->data.real_single;
                    float * ydata = ybase->data.real_single;
                    cblas_sgemv(
                        CblasRowMajor, CblasTrans, A->num_rows, A->num_columns,
                        alpha, Adata, A->num_columns, xdata, 1, beta, ydata, 1);
                    if (mtxblaserror()) return MTX_ERR_BLAS;
                    if (num_flops) *num_flops += cblas_sgemv_num_flops(
                        A->num_columns, A->num_rows, alpha, beta);
                } else if (a->precision == mtx_double) {
                    const double * Adata = a->data.real_double;
                    const double * xdata = xbase->data.real_double;
                    double * ydata = ybase->data.real_double;
                    cblas_dgemv(
                        CblasRowMajor, CblasTrans, A->num_rows, A->num_columns,
                        alpha, Adata, A->num_columns, xdata, 1, beta, ydata, 1);
                    if (mtxblaserror()) return MTX_ERR_BLAS;
                    if (num_flops) *num_flops += cblas_dgemv_num_flops(
                        A->num_columns, A->num_rows, alpha, beta);
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else { return MTX_ERR_INVALID_TRANSPOSITION; }
        } else if (a->field == mtx_field_complex) {
            if (trans == mtx_notrans) {
                if (a->precision == mtx_single) {
                    const float (* Adata)[2] = a->data.complex_single;
                    const float (* xdata)[2] = xbase->data.complex_single;
                    float (* ydata)[2] = ybase->data.complex_single;
                    const float calpha[2] = {alpha, 0};
                    const float cbeta[2] = {beta, 0};
                    cblas_cgemv(
                        CblasRowMajor, CblasNoTrans, A->num_rows, A->num_columns,
                        calpha, (const float *) Adata, A->num_columns,
                        (const float *) xdata, 1, cbeta, (float *) ydata, 1);
                    if (mtxblaserror()) return MTX_ERR_BLAS;
                    if (num_flops) *num_flops += cblas_cgemv_num_flops(
                        A->num_rows, A->num_columns, calpha, cbeta);
                } else if (a->precision == mtx_double) {
                    const double (* Adata)[2] = a->data.complex_double;
                    const double (* xdata)[2] = xbase->data.complex_double;
                    double (* ydata)[2] = ybase->data.complex_double;
                    const double zalpha[2] = {alpha, 0};
                    const double zbeta[2] = {beta, 0};
                    cblas_zgemv(
                        CblasRowMajor, CblasNoTrans, A->num_rows, A->num_columns,
                        zalpha, (const double *) Adata, A->num_columns,
                        (const double *) xdata, 1, zbeta, (double *) ydata, 1);
                    if (mtxblaserror()) return MTX_ERR_BLAS;
                    if (num_flops) *num_flops += cblas_zgemv_num_flops(
                        A->num_rows, A->num_columns, zalpha, zbeta);
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (trans == mtx_trans) {
                if (a->precision == mtx_single) {
                    const float (* Adata)[2] = a->data.complex_single;
                    const float (* xdata)[2] = xbase->data.complex_single;
                    float (* ydata)[2] = ybase->data.complex_single;
                    const float calpha[2] = {alpha, 0};
                    const float cbeta[2] = {beta, 0};
                    cblas_cgemv(
                        CblasRowMajor, CblasTrans, A->num_rows, A->num_columns,
                        calpha, (const float *) Adata, A->num_columns,
                        (const float *) xdata, 1, cbeta, (float *) ydata, 1);
                    if (mtxblaserror()) return MTX_ERR_BLAS;
                    if (num_flops) *num_flops += cblas_cgemv_num_flops(
                        A->num_columns, A->num_rows, calpha, cbeta);
                } else if (a->precision == mtx_double) {
                    const double (* Adata)[2] = a->data.complex_double;
                    const double (* xdata)[2] = xbase->data.complex_double;
                    double (* ydata)[2] = ybase->data.complex_double;
                    const double zalpha[2] = {alpha, 0};
                    const double zbeta[2] = {beta, 0};
                    cblas_zgemv(
                        CblasRowMajor, CblasTrans, A->num_rows, A->num_columns,
                        zalpha, (const double *) Adata, A->num_columns,
                        (const double *) xdata, 1, zbeta, (double *) ydata, 1);
                    if (mtxblaserror()) return MTX_ERR_BLAS;
                    if (num_flops) *num_flops += cblas_zgemv_num_flops(
                        A->num_columns, A->num_rows, zalpha, zbeta);
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (trans == mtx_conjtrans) {
                if (a->precision == mtx_single) {
                    const float (* Adata)[2] = a->data.complex_single;
                    const float (* xdata)[2] = xbase->data.complex_single;
                    float (* ydata)[2] = ybase->data.complex_single;
                    const float calpha[2] = {alpha, 0};
                    const float cbeta[2] = {beta, 0};
                    cblas_cgemv(
                        CblasRowMajor, CblasConjTrans, A->num_rows, A->num_columns,
                        calpha, (const float *) Adata, A->num_columns,
                        (const float *) xdata, 1, cbeta, (float *) ydata, 1);
                    if (mtxblaserror()) return MTX_ERR_BLAS;
                    if (num_flops) *num_flops += cblas_cgemv_num_flops(
                        A->num_columns, A->num_rows, calpha, cbeta);
                } else if (a->precision == mtx_double) {
                    const double (* Adata)[2] = a->data.complex_double;
                    const double (* xdata)[2] = xbase->data.complex_double;
                    double (* ydata)[2] = ybase->data.complex_double;
                    const double zalpha[2] = {alpha, 0};
                    const double zbeta[2] = {beta, 0};
                    cblas_zgemv(
                        CblasRowMajor, CblasConjTrans, A->num_rows, A->num_columns,
                        zalpha, (const double *) Adata, A->num_columns,
                        (const double *) xdata, 1, zbeta, (double *) ydata, 1);
                    if (mtxblaserror()) return MTX_ERR_BLAS;
                    if (num_flops) *num_flops += cblas_zgemv_num_flops(
                        A->num_columns, A->num_rows, zalpha, zbeta);
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else { return MTX_ERR_INVALID_TRANSPOSITION; }
        } else { return MTX_ERR_INVALID_FIELD; }
    } else if (A->num_rows == A->num_columns && A->symmetry == mtx_symmetric) {
        if (a->field == mtx_field_real) {
            if (a->precision == mtx_single) {
                const float * Adata = a->data.real_single;
                const float * xdata = xbase->data.real_single;
                float * ydata = ybase->data.real_single;
                cblas_sspmv(
                    CblasRowMajor, CblasUpper, A->num_rows,
                    alpha, Adata, xdata, 1, beta, ydata, 1);
                if (mtxblaserror()) return MTX_ERR_BLAS;
                if (num_flops) *num_flops += cblas_sspmv_num_flops(
                    A->num_rows, alpha, beta);
            } else if (a->precision == mtx_double) {
                const double * Adata = a->data.real_double;
                const double * xdata = xbase->data.real_double;
                double * ydata = ybase->data.real_double;
                cblas_dspmv(
                    CblasRowMajor, CblasUpper, A->num_rows,
                    alpha, Adata, xdata, 1, beta, ydata, 1);
                if (mtxblaserror()) return MTX_ERR_BLAS;
                if (num_flops) *num_flops += cblas_dspmv_num_flops(
                    A->num_rows, alpha, beta);
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else { return MTX_ERR_INVALID_FIELD; }
    } else if (A->num_rows == A->num_columns && A->symmetry == mtx_skew_symmetric) {
        /* TODO: allow skew-symmetric matrices */
        return MTX_ERR_INVALID_SYMMETRY;
    } else if (A->num_rows == A->num_columns && A->symmetry == mtx_hermitian) {
        if (a->field == mtx_field_complex) {
            if (trans == mtx_notrans || trans == mtx_conjtrans) {
                if (a->precision == mtx_single) {
                    const float (* Adata)[2] = a->data.complex_single;
                    const float (* xdata)[2] = xbase->data.complex_single;
                    float (* ydata)[2] = ybase->data.complex_single;
                    const float calpha[2] = {alpha, 0};
                    const float cbeta[2] = {beta, 0};
                    cblas_chpmv(
                        CblasRowMajor, CblasUpper, A->num_rows, calpha,
                        (const float *) Adata, (const float *) xdata, 1,
                        cbeta, (float *) ydata, 1);
                    if (mtxblaserror()) return MTX_ERR_BLAS;
                    if (num_flops) *num_flops += cblas_chpmv_num_flops(
                        A->num_rows, calpha, cbeta);
                } else if (a->precision == mtx_double) {
                    const double (* Adata)[2] = a->data.complex_double;
                    const double (* xdata)[2] = xbase->data.complex_double;
                    double (* ydata)[2] = ybase->data.complex_double;
                    const double zalpha[2] = {alpha, 0};
                    const double zbeta[2] = {beta, 0};
                    cblas_zhpmv(
                        CblasRowMajor, CblasUpper, A->num_rows, zalpha,
                        (const double *) Adata, (const double *) xdata, 1,
                        zbeta, (double *) ydata, 1);
                    if (mtxblaserror()) return MTX_ERR_BLAS;
                    if (num_flops) *num_flops += cblas_zhpmv_num_flops(
                        A->num_rows, zalpha, zbeta);
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else { return MTX_ERR_INVALID_TRANSPOSITION; }
        } else { return MTX_ERR_INVALID_FIELD; }
    } else { return MTX_ERR_INVALID_SYMMETRY; }
    return MTX_SUCCESS;
#else
    return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
}

/**
 * ‘mtxblasdense_dgemv()’ multiplies a matrix ‘A’ or its transpose
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
int mtxblasdense_dgemv(
    enum mtxtransposition trans,
    double alpha,
    const struct mtxblasdense * A,
    const struct mtxvector * x,
    double beta,
    struct mtxvector * y,
    int64_t * num_flops)
{
#ifdef LIBMTX_HAVE_BLAS
    const struct mtxbasevector * a = &A->a.base;
    if (x->type != mtxbasevector || y->type != mtxbasevector)
        return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
    const struct mtxbasevector * xbase = &x->storage.base;
    struct mtxbasevector * ybase = &y->storage.base;
    if (xbase->field != a->field || ybase->field != a->field)
        return MTX_ERR_INCOMPATIBLE_FIELD;
    if (xbase->precision != a->precision || ybase->precision != a->precision)
        return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (trans == mtx_notrans &&
        (A->num_rows != ybase->size || A->num_columns != xbase->size))
        return MTX_ERR_INCOMPATIBLE_SIZE;
    if ((trans == mtx_trans || trans == mtx_conjtrans) &&
        (A->num_columns != ybase->size || A->num_rows != xbase->size))
        return MTX_ERR_INCOMPATIBLE_SIZE;

    if (A->symmetry == mtx_unsymmetric) {
        if (a->field == mtx_field_real) {
            if (trans == mtx_notrans) {
                if (a->precision == mtx_single) {
                    const float * Adata = a->data.real_single;
                    const float * xdata = xbase->data.real_single;
                    float * ydata = ybase->data.real_single;
                    cblas_sgemv(
                        CblasRowMajor, CblasNoTrans, A->num_rows, A->num_columns,
                        alpha, Adata, A->num_columns, xdata, 1, beta, ydata, 1);
                    if (mtxblaserror()) return MTX_ERR_BLAS;
                    if (num_flops) *num_flops += cblas_sgemv_num_flops(
                        A->num_rows, A->num_columns, alpha, beta);
                } else if (a->precision == mtx_double) {
                    const double * Adata = a->data.real_double;
                    const double * xdata = xbase->data.real_double;
                    double * ydata = ybase->data.real_double;
                    cblas_dgemv(
                        CblasRowMajor, CblasNoTrans, A->num_rows, A->num_columns,
                        alpha, Adata, A->num_columns, xdata, 1, beta, ydata, 1);
                    if (mtxblaserror()) return MTX_ERR_BLAS;
                    if (num_flops) *num_flops += cblas_dgemv_num_flops(
                        A->num_rows, A->num_columns, alpha, beta);
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (trans == mtx_trans || trans == mtx_conjtrans) {
                if (a->precision == mtx_single) {
                    const float * Adata = a->data.real_single;
                    const float * xdata = xbase->data.real_single;
                    float * ydata = ybase->data.real_single;
                    cblas_sgemv(
                        CblasRowMajor, CblasTrans, A->num_rows, A->num_columns,
                        alpha, Adata, A->num_columns, xdata, 1, beta, ydata, 1);
                    if (mtxblaserror()) return MTX_ERR_BLAS;
                    if (num_flops) *num_flops += cblas_sgemv_num_flops(
                        A->num_columns, A->num_rows, alpha, beta);
                } else if (a->precision == mtx_double) {
                    const double * Adata = a->data.real_double;
                    const double * xdata = xbase->data.real_double;
                    double * ydata = ybase->data.real_double;
                    cblas_dgemv(
                        CblasRowMajor, CblasTrans, A->num_rows, A->num_columns,
                        alpha, Adata, A->num_columns, xdata, 1, beta, ydata, 1);
                    if (mtxblaserror()) return MTX_ERR_BLAS;
                    if (num_flops) *num_flops += cblas_dgemv_num_flops(
                        A->num_columns, A->num_rows, alpha, beta);
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else { return MTX_ERR_INVALID_TRANSPOSITION; }
        } else if (a->field == mtx_field_complex) {
            if (trans == mtx_notrans) {
                if (a->precision == mtx_single) {
                    const float (* Adata)[2] = a->data.complex_single;
                    const float (* xdata)[2] = xbase->data.complex_single;
                    float (* ydata)[2] = ybase->data.complex_single;
                    const float calpha[2] = {alpha, 0};
                    const float cbeta[2] = {beta, 0};
                    cblas_cgemv(
                        CblasRowMajor, CblasNoTrans, A->num_rows, A->num_columns,
                        calpha, (const float *) Adata, A->num_columns,
                        (const float *) xdata, 1, cbeta, (float *) ydata, 1);
                    if (mtxblaserror()) return MTX_ERR_BLAS;
                    if (num_flops) *num_flops += cblas_cgemv_num_flops(
                        A->num_rows, A->num_columns, calpha, cbeta);
                } else if (a->precision == mtx_double) {
                    const double (* Adata)[2] = a->data.complex_double;
                    const double (* xdata)[2] = xbase->data.complex_double;
                    double (* ydata)[2] = ybase->data.complex_double;
                    const double zalpha[2] = {alpha, 0};
                    const double zbeta[2] = {beta, 0};
                    cblas_zgemv(
                        CblasRowMajor, CblasNoTrans, A->num_rows, A->num_columns,
                        zalpha, (const double *) Adata, A->num_columns,
                        (const double *) xdata, 1, zbeta, (double *) ydata, 1);
                    if (mtxblaserror()) return MTX_ERR_BLAS;
                    if (num_flops) *num_flops += cblas_zgemv_num_flops(
                        A->num_rows, A->num_columns, zalpha, zbeta);
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (trans == mtx_trans) {
                if (a->precision == mtx_single) {
                    const float (* Adata)[2] = a->data.complex_single;
                    const float (* xdata)[2] = xbase->data.complex_single;
                    float (* ydata)[2] = ybase->data.complex_single;
                    const float calpha[2] = {alpha, 0};
                    const float cbeta[2] = {beta, 0};
                    cblas_cgemv(
                        CblasRowMajor, CblasTrans, A->num_rows, A->num_columns,
                        calpha, (const float *) Adata, A->num_columns,
                        (const float *) xdata, 1, cbeta, (float *) ydata, 1);
                    if (mtxblaserror()) return MTX_ERR_BLAS;
                    if (num_flops) *num_flops += cblas_cgemv_num_flops(
                        A->num_columns, A->num_rows, calpha, cbeta);
                } else if (a->precision == mtx_double) {
                    const double (* Adata)[2] = a->data.complex_double;
                    const double (* xdata)[2] = xbase->data.complex_double;
                    double (* ydata)[2] = ybase->data.complex_double;
                    const double zalpha[2] = {alpha, 0};
                    const double zbeta[2] = {beta, 0};
                    cblas_zgemv(
                        CblasRowMajor, CblasTrans, A->num_rows, A->num_columns,
                        zalpha, (const double *) Adata, A->num_columns,
                        (const double *) xdata, 1, zbeta, (double *) ydata, 1);
                    if (mtxblaserror()) return MTX_ERR_BLAS;
                    if (num_flops) *num_flops += cblas_zgemv_num_flops(
                        A->num_columns, A->num_rows, zalpha, zbeta);
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (trans == mtx_conjtrans) {
                if (a->precision == mtx_single) {
                    const float (* Adata)[2] = a->data.complex_single;
                    const float (* xdata)[2] = xbase->data.complex_single;
                    float (* ydata)[2] = ybase->data.complex_single;
                    const float calpha[2] = {alpha, 0};
                    const float cbeta[2] = {beta, 0};
                    cblas_cgemv(
                        CblasRowMajor, CblasConjTrans, A->num_rows, A->num_columns,
                        calpha, (const float *) Adata, A->num_columns,
                        (const float *) xdata, 1, cbeta, (float *) ydata, 1);
                    if (mtxblaserror()) return MTX_ERR_BLAS;
                    if (num_flops) *num_flops += cblas_cgemv_num_flops(
                        A->num_columns, A->num_rows, calpha, cbeta);
                } else if (a->precision == mtx_double) {
                    const double (* Adata)[2] = a->data.complex_double;
                    const double (* xdata)[2] = xbase->data.complex_double;
                    double (* ydata)[2] = ybase->data.complex_double;
                    const double zalpha[2] = {alpha, 0};
                    const double zbeta[2] = {beta, 0};
                    cblas_zgemv(
                        CblasRowMajor, CblasConjTrans, A->num_rows, A->num_columns,
                        zalpha, (const double *) Adata, A->num_columns,
                        (const double *) xdata, 1, zbeta, (double *) ydata, 1);
                    if (mtxblaserror()) return MTX_ERR_BLAS;
                    if (num_flops) *num_flops += cblas_zgemv_num_flops(
                        A->num_columns, A->num_rows, zalpha, zbeta);
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else { return MTX_ERR_INVALID_TRANSPOSITION; }
        } else { return MTX_ERR_INVALID_FIELD; }
    } else if (A->num_rows == A->num_columns && A->symmetry == mtx_symmetric) {
        if (a->field == mtx_field_real) {
            if (a->precision == mtx_single) {
                const float * Adata = a->data.real_single;
                const float * xdata = xbase->data.real_single;
                float * ydata = ybase->data.real_single;
                cblas_sspmv(
                    CblasRowMajor, CblasUpper, A->num_rows,
                    alpha, Adata, xdata, 1, beta, ydata, 1);
                if (mtxblaserror()) return MTX_ERR_BLAS;
                if (num_flops) *num_flops += cblas_sspmv_num_flops(
                    A->num_rows, alpha, beta);
            } else if (a->precision == mtx_double) {
                const double * Adata = a->data.real_double;
                const double * xdata = xbase->data.real_double;
                double * ydata = ybase->data.real_double;
                cblas_dspmv(
                    CblasRowMajor, CblasUpper, A->num_rows,
                    alpha, Adata, xdata, 1, beta, ydata, 1);
                if (mtxblaserror()) return MTX_ERR_BLAS;
                if (num_flops) *num_flops += cblas_dspmv_num_flops(
                    A->num_rows, alpha, beta);
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else { return MTX_ERR_INVALID_FIELD; }
    } else if (A->num_rows == A->num_columns && A->symmetry == mtx_skew_symmetric) {
        /* TODO: allow skew-symmetric matrices */
        return MTX_ERR_INVALID_SYMMETRY;
    } else if (A->num_rows == A->num_columns && A->symmetry == mtx_hermitian) {
        if (a->field == mtx_field_complex) {
            if (trans == mtx_notrans || trans == mtx_conjtrans) {
                if (a->precision == mtx_single) {
                    const float (* Adata)[2] = a->data.complex_single;
                    const float (* xdata)[2] = xbase->data.complex_single;
                    float (* ydata)[2] = ybase->data.complex_single;
                    const float calpha[2] = {alpha, 0};
                    const float cbeta[2] = {beta, 0};
                    cblas_chpmv(
                        CblasRowMajor, CblasUpper, A->num_rows, calpha,
                        (const float *) Adata, (const float *) xdata, 1,
                        cbeta, (float *) ydata, 1);
                    if (mtxblaserror()) return MTX_ERR_BLAS;
                    if (num_flops) *num_flops += cblas_chpmv_num_flops(
                        A->num_rows, calpha, cbeta);
                } else if (a->precision == mtx_double) {
                    const double (* Adata)[2] = a->data.complex_double;
                    const double (* xdata)[2] = xbase->data.complex_double;
                    double (* ydata)[2] = ybase->data.complex_double;
                    const double zalpha[2] = {alpha, 0};
                    const double zbeta[2] = {beta, 0};
                    cblas_zhpmv(
                        CblasRowMajor, CblasUpper, A->num_rows, zalpha,
                        (const double *) Adata, (const double *) xdata, 1,
                        zbeta, (double *) ydata, 1);
                    if (mtxblaserror()) return MTX_ERR_BLAS;
                    if (num_flops) *num_flops += cblas_zhpmv_num_flops(
                        A->num_rows, zalpha, zbeta);
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else { return MTX_ERR_INVALID_TRANSPOSITION; }
        } else { return MTX_ERR_INVALID_FIELD; }
    } else { return MTX_ERR_INVALID_SYMMETRY; }
    return MTX_SUCCESS;
#else
    return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
}

/**
 * ‘mtxblasdense_cgemv()’ multiplies a complex-valued matrix ‘A’,
 * its transpose ‘A'’ or its conjugate transpose ‘Aᴴ’ by a complex
 * scalar ‘alpha’ (‘α’) and a vector ‘x’, before adding the result to
 * another vector ‘y’ multiplied by another complex scalar ‘beta’
 * (‘β’).  That is, ‘y = α*A*x + β*y’, ‘y = α*A'*x + β*y’ or ‘y =
 * α*Aᴴ*x + β*y’.
 *
 * The scalars ‘alpha’ and ‘beta’ are given as single precision
 * floating point numbers.
 *
 * The vectors ‘x’ and ‘y’ must be of ‘array’ type, and they must both
 * have the same field and precision as the matrix ‘A’.
 *
 * Moreover, if ‘trans’ is ‘mtx_notrans’, then the size of ‘x’ must
 * equal the number of columns of ‘A’ and the size of ‘y’ must equal
 * the number of rows of ‘A’. if ‘trans’ is ‘mtx_trans’ or
 * ‘mtx_conjtrans’, then the size of ‘x’ must equal the number of rows
 * of ‘A’ and the size of ‘y’ must equal the number of columns of ‘A’.
 */
int mtxblasdense_cgemv(
    enum mtxtransposition trans,
    float alpha[2],
    const struct mtxblasdense * A,
    const struct mtxvector * x,
    float beta[2],
    struct mtxvector * y,
    int64_t * num_flops)
{
#ifdef LIBMTX_HAVE_BLAS
    const struct mtxbasevector * a = &A->a.base;
    if (x->type != mtxbasevector || y->type != mtxbasevector)
        return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
    const struct mtxbasevector * xbase = &x->storage.base;
    struct mtxbasevector * ybase = &y->storage.base;
    if (xbase->field != a->field || ybase->field != a->field)
        return MTX_ERR_INCOMPATIBLE_FIELD;
    if (xbase->precision != a->precision || ybase->precision != a->precision)
        return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (trans == mtx_notrans &&
        (A->num_rows != ybase->size || A->num_columns != xbase->size))
        return MTX_ERR_INCOMPATIBLE_SIZE;
    if ((trans == mtx_trans || trans == mtx_conjtrans) &&
        (A->num_columns != ybase->size || A->num_rows != xbase->size))
        return MTX_ERR_INCOMPATIBLE_SIZE;
    if (a->field != mtx_field_complex)
        return MTX_ERR_INCOMPATIBLE_FIELD;

    if (A->symmetry == mtx_unsymmetric) {
        if (trans == mtx_notrans) {
            if (a->precision == mtx_single) {
                const float (* Adata)[2] = a->data.complex_single;
                const float (* xdata)[2] = xbase->data.complex_single;
                float (* ydata)[2] = ybase->data.complex_single;
                cblas_cgemv(
                    CblasRowMajor, CblasNoTrans, A->num_rows, A->num_columns,
                    alpha, (const float *) Adata, A->num_columns,
                    (const float *) xdata, 1, beta, (float *) ydata, 1);
                if (mtxblaserror()) return MTX_ERR_BLAS;
                if (num_flops) *num_flops += cblas_cgemv_num_flops(
                    A->num_rows, A->num_columns, alpha, beta);
            } else if (a->precision == mtx_double) {
                const double (* Adata)[2] = a->data.complex_double;
                const double (* xdata)[2] = xbase->data.complex_double;
                double (* ydata)[2] = ybase->data.complex_double;
                const double zalpha[2] = {alpha[0], alpha[1]};
                const double zbeta[2] = {beta[0], beta[1]};
                cblas_zgemv(
                    CblasRowMajor, CblasNoTrans, A->num_rows, A->num_columns,
                    zalpha, (const double *) Adata, A->num_columns,
                    (const double *) xdata, 1, zbeta, (double *) ydata, 1);
                if (mtxblaserror()) return MTX_ERR_BLAS;
                if (num_flops) *num_flops += cblas_zgemv_num_flops(
                    A->num_rows, A->num_columns, zalpha, zbeta);
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (trans == mtx_trans) {
            if (a->precision == mtx_single) {
                const float (* Adata)[2] = a->data.complex_single;
                const float (* xdata)[2] = xbase->data.complex_single;
                float (* ydata)[2] = ybase->data.complex_single;
                cblas_cgemv(
                    CblasRowMajor, CblasTrans, A->num_rows, A->num_columns,
                    alpha, (const float *) Adata, A->num_columns,
                    (const float *) xdata, 1, beta, (float *) ydata, 1);
                if (mtxblaserror()) return MTX_ERR_BLAS;
                if (num_flops) *num_flops += cblas_cgemv_num_flops(
                    A->num_rows, A->num_columns, alpha, beta);
            } else if (a->precision == mtx_double) {
                const double (* Adata)[2] = a->data.complex_double;
                const double (* xdata)[2] = xbase->data.complex_double;
                double (* ydata)[2] = ybase->data.complex_double;
                const double zalpha[2] = {alpha[0], alpha[1]};
                const double zbeta[2] = {beta[0], beta[1]};
                cblas_zgemv(
                    CblasRowMajor, CblasTrans, A->num_rows, A->num_columns,
                    zalpha, (const double *) Adata, A->num_columns,
                    (const double *) xdata, 1, zbeta, (double *) ydata, 1);
                if (mtxblaserror()) return MTX_ERR_BLAS;
                if (num_flops) *num_flops += cblas_zgemv_num_flops(
                    A->num_rows, A->num_columns, zalpha, zbeta);
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (trans == mtx_conjtrans) {
            if (a->precision == mtx_single) {
                const float (* Adata)[2] = a->data.complex_single;
                const float (* xdata)[2] = xbase->data.complex_single;
                float (* ydata)[2] = ybase->data.complex_single;
                cblas_cgemv(
                    CblasRowMajor, CblasConjTrans, A->num_rows, A->num_columns,
                    alpha, (const float *) Adata, A->num_columns,
                    (const float *) xdata, 1, beta, (float *) ydata, 1);
                if (mtxblaserror()) return MTX_ERR_BLAS;
                if (num_flops) *num_flops += cblas_cgemv_num_flops(
                    A->num_rows, A->num_columns, alpha, beta);
            } else if (a->precision == mtx_double) {
                const double (* Adata)[2] = a->data.complex_double;
                const double (* xdata)[2] = xbase->data.complex_double;
                double (* ydata)[2] = ybase->data.complex_double;
                const double zalpha[2] = {alpha[0], alpha[1]};
                const double zbeta[2] = {beta[0], beta[1]};
                cblas_zgemv(
                    CblasRowMajor, CblasConjTrans, A->num_rows, A->num_columns,
                    zalpha, (const double *) Adata, A->num_columns,
                    (const double *) xdata, 1, zbeta, (double *) ydata, 1);
                if (mtxblaserror()) return MTX_ERR_BLAS;
                if (num_flops) *num_flops += cblas_zgemv_num_flops(
                    A->num_rows, A->num_columns, zalpha, zbeta);
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else { return MTX_ERR_INVALID_TRANSPOSITION; }
    } else if (A->num_rows == A->num_columns && A->symmetry == mtx_hermitian) {
        if (trans == mtx_notrans || trans == mtx_conjtrans) {
            if (a->precision == mtx_single) {
                const float (* Adata)[2] = a->data.complex_single;
                const float (* xdata)[2] = xbase->data.complex_single;
                float (* ydata)[2] = ybase->data.complex_single;
                cblas_chpmv(
                    CblasRowMajor, CblasUpper, A->num_rows, alpha,
                    (const float *) Adata, (const float *) xdata, 1,
                    beta, (float *) ydata, 1);
                if (mtxblaserror()) return MTX_ERR_BLAS;
                if (num_flops) *num_flops += cblas_chpmv_num_flops(
                    A->num_rows, alpha, beta);
            } else if (a->precision == mtx_double) {
                const double (* Adata)[2] = a->data.complex_double;
                const double (* xdata)[2] = xbase->data.complex_double;
                double (* ydata)[2] = ybase->data.complex_double;
                const double zalpha[2] = {alpha[0], alpha[1]};
                const double zbeta[2] = {beta[0], beta[1]};
                cblas_zhpmv(
                    CblasRowMajor, CblasUpper, A->num_rows, zalpha,
                    (const double *) Adata, (const double *) xdata, 1,
                    zbeta, (double *) ydata, 1);
                if (mtxblaserror()) return MTX_ERR_BLAS;
                if (num_flops) *num_flops += cblas_zhpmv_num_flops(
                    A->num_rows, zalpha, zbeta);
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else { return MTX_ERR_INVALID_TRANSPOSITION; }
    } else { return MTX_ERR_INVALID_SYMMETRY; }
    return MTX_SUCCESS;
#else
    return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
}

/**
 * ‘mtxblasdense_zgemv()’ multiplies a complex-valued matrix ‘A’,
 * its transpose ‘A'’ or its conjugate transpose ‘Aᴴ’ by a complex
 * scalar ‘alpha’ (‘α’) and a vector ‘x’, before adding the result to
 * another vector ‘y’ multiplied by another complex scalar ‘beta’
 * (‘β’).  That is, ‘y = α*A*x + β*y’, ‘y = α*A'*x + β*y’ or ‘y =
 * α*Aᴴ*x + β*y’.
 *
 * The scalars ‘alpha’ and ‘beta’ are given as double precision
 * floating point numbers.
 *
 * The vectors ‘x’ and ‘y’ must be of ‘array’ type, and they must both
 * have the same field and precision as the matrix ‘A’.
 *
 * Moreover, if ‘trans’ is ‘mtx_notrans’, then the size of ‘x’ must
 * equal the number of columns of ‘A’ and the size of ‘y’ must equal
 * the number of rows of ‘A’. if ‘trans’ is ‘mtx_trans’ or
 * ‘mtx_conjtrans’, then the size of ‘x’ must equal the number of rows
 * of ‘A’ and the size of ‘y’ must equal the number of columns of ‘A’.
 */
int mtxblasdense_zgemv(
    enum mtxtransposition trans,
    double alpha[2],
    const struct mtxblasdense * A,
    const struct mtxvector * x,
    double beta[2],
    struct mtxvector * y,
    int64_t * num_flops)
{
#ifdef LIBMTX_HAVE_BLAS
    const struct mtxbasevector * a = &A->a.base;
    if (x->type != mtxbasevector || y->type != mtxbasevector)
        return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
    const struct mtxbasevector * xbase = &x->storage.base;
    struct mtxbasevector * ybase = &y->storage.base;
    if (xbase->field != a->field || ybase->field != a->field)
        return MTX_ERR_INCOMPATIBLE_FIELD;
    if (xbase->precision != a->precision || ybase->precision != a->precision)
        return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (trans == mtx_notrans &&
        (A->num_rows != ybase->size || A->num_columns != xbase->size))
        return MTX_ERR_INCOMPATIBLE_SIZE;
    if ((trans == mtx_trans || trans == mtx_conjtrans) &&
        (A->num_columns != ybase->size || A->num_rows != xbase->size))
        return MTX_ERR_INCOMPATIBLE_SIZE;
    if (a->field != mtx_field_complex)
        return MTX_ERR_INCOMPATIBLE_FIELD;

    if (A->symmetry == mtx_unsymmetric) {
        if (trans == mtx_notrans) {
            if (a->precision == mtx_single) {
                const float (* Adata)[2] = a->data.complex_single;
                const float (* xdata)[2] = xbase->data.complex_single;
                float (* ydata)[2] = ybase->data.complex_single;
                const float calpha[2] = {alpha[0], alpha[1]};
                const float cbeta[2] = {beta[0], beta[1]};
                cblas_cgemv(
                    CblasRowMajor, CblasNoTrans, A->num_rows, A->num_columns,
                    calpha, (const float *) Adata, A->num_columns,
                    (const float *) xdata, 1, cbeta, (float *) ydata, 1);
                if (mtxblaserror()) return MTX_ERR_BLAS;
                if (num_flops) *num_flops += cblas_cgemv_num_flops(
                    A->num_rows, A->num_columns, calpha, cbeta);
            } else if (a->precision == mtx_double) {
                const double (* Adata)[2] = a->data.complex_double;
                const double (* xdata)[2] = xbase->data.complex_double;
                double (* ydata)[2] = ybase->data.complex_double;
                cblas_zgemv(
                    CblasRowMajor, CblasNoTrans, A->num_rows, A->num_columns,
                    alpha, (const double *) Adata, A->num_columns,
                    (const double *) xdata, 1, beta, (double *) ydata, 1);
                if (mtxblaserror()) return MTX_ERR_BLAS;
                if (num_flops) *num_flops += cblas_zgemv_num_flops(
                    A->num_rows, A->num_columns, alpha, beta);
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (trans == mtx_trans) {
            if (a->precision == mtx_single) {
                const float (* Adata)[2] = a->data.complex_single;
                const float (* xdata)[2] = xbase->data.complex_single;
                float (* ydata)[2] = ybase->data.complex_single;
                const float calpha[2] = {alpha[0], alpha[1]};
                const float cbeta[2] = {beta[0], beta[1]};
                cblas_cgemv(
                    CblasRowMajor, CblasTrans, A->num_rows, A->num_columns,
                    calpha, (const float *) Adata, A->num_columns,
                    (const float *) xdata, 1, cbeta, (float *) ydata, 1);
                if (mtxblaserror()) return MTX_ERR_BLAS;
                if (num_flops) *num_flops += cblas_cgemv_num_flops(
                    A->num_rows, A->num_columns, calpha, cbeta);
            } else if (a->precision == mtx_double) {
                const double (* Adata)[2] = a->data.complex_double;
                const double (* xdata)[2] = xbase->data.complex_double;
                double (* ydata)[2] = ybase->data.complex_double;
                cblas_zgemv(
                    CblasRowMajor, CblasTrans, A->num_rows, A->num_columns,
                    alpha, (const double *) Adata, A->num_columns,
                    (const double *) xdata, 1, beta, (double *) ydata, 1);
                if (mtxblaserror()) return MTX_ERR_BLAS;
                if (num_flops) *num_flops += cblas_zgemv_num_flops(
                    A->num_rows, A->num_columns, alpha, beta);
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (trans == mtx_conjtrans) {
            if (a->precision == mtx_single) {
                const float (* Adata)[2] = a->data.complex_single;
                const float (* xdata)[2] = xbase->data.complex_single;
                float (* ydata)[2] = ybase->data.complex_single;
                const float calpha[2] = {alpha[0], alpha[1]};
                const float cbeta[2] = {beta[0], beta[1]};
                cblas_cgemv(
                    CblasRowMajor, CblasConjTrans, A->num_rows, A->num_columns,
                    calpha, (const float *) Adata, A->num_columns,
                    (const float *) xdata, 1, cbeta, (float *) ydata, 1);
                if (mtxblaserror()) return MTX_ERR_BLAS;
                if (num_flops) *num_flops += cblas_cgemv_num_flops(
                    A->num_rows, A->num_columns, calpha, cbeta);
            } else if (a->precision == mtx_double) {
                const double (* Adata)[2] = a->data.complex_double;
                const double (* xdata)[2] = xbase->data.complex_double;
                double (* ydata)[2] = ybase->data.complex_double;
                cblas_zgemv(
                    CblasRowMajor, CblasConjTrans, A->num_rows, A->num_columns,
                    alpha, (const double *) Adata, A->num_columns,
                    (const double *) xdata, 1, beta, (double *) ydata, 1);
                if (mtxblaserror()) return MTX_ERR_BLAS;
                if (num_flops) *num_flops += cblas_zgemv_num_flops(
                    A->num_rows, A->num_columns, alpha, beta);
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else { return MTX_ERR_INVALID_TRANSPOSITION; }
    } else if (A->num_rows == A->num_columns && A->symmetry == mtx_hermitian) {
        if (trans == mtx_notrans || trans == mtx_conjtrans) {
            if (a->precision == mtx_single) {
                const float (* Adata)[2] = a->data.complex_single;
                const float (* xdata)[2] = xbase->data.complex_single;
                float (* ydata)[2] = ybase->data.complex_single;
                const float calpha[2] = {alpha[0], alpha[1]};
                const float cbeta[2] = {beta[0], beta[1]};
                cblas_chpmv(
                    CblasRowMajor, CblasUpper, A->num_rows, calpha,
                    (const float *) Adata, (const float *) xdata, 1,
                    cbeta, (float *) ydata, 1);
                if (mtxblaserror()) return MTX_ERR_BLAS;
                if (num_flops) *num_flops += cblas_chpmv_num_flops(
                    A->num_rows, calpha, cbeta);
            } else if (a->precision == mtx_double) {
                const double (* Adata)[2] = a->data.complex_double;
                const double (* xdata)[2] = xbase->data.complex_double;
                double (* ydata)[2] = ybase->data.complex_double;
                cblas_zhpmv(
                    CblasRowMajor, CblasUpper, A->num_rows, alpha,
                    (const double *) Adata, (const double *) xdata, 1,
                    beta, (double *) ydata, 1);
                if (mtxblaserror()) return MTX_ERR_BLAS;
                if (num_flops) *num_flops += cblas_zhpmv_num_flops(
                    A->num_rows, alpha, beta);
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else { return MTX_ERR_INVALID_TRANSPOSITION; }
    } else { return MTX_ERR_INVALID_SYMMETRY; }
    return MTX_SUCCESS;
#else
    return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
}

/*
 * Level 3 BLAS operations
 */

/**
 * ‘mtxblasdense_sgemm()’ multiplies a matrix ‘A’ (or its transpose ‘A'’)
 * by a real scalar ‘alpha’ (‘α’) and a matrix ‘B’ (or its transpose
 * ‘B'’), before adding the result to another matrix ‘C’ multiplied by
 * a real scalar ‘beta’ (‘β’). That is,
 *
 * ‘C = α*op(A)*op(B) + β*C’, where ‘op(X)=X’ or ‘op(X)=X'’.
 *
 * The scalars ‘alpha’ and ‘beta’ are given as single precision
 * floating point numbers.
 */
int mtxblasdense_sgemm(
    enum mtxtransposition Atrans,
    enum mtxtransposition Btrans,
    float alpha,
    const struct mtxblasdense * A,
    const struct mtxblasdense * B,
    float beta,
    struct mtxblasdense * C,
    int64_t * num_flops)
{
#ifdef LIBMTX_HAVE_BLAS
    const struct mtxbasevector * a = &A->a.base;
    const struct mtxbasevector * b = &B->a.base;
    struct mtxbasevector * c = &C->a.base;
    if (b->field != a->field || c->field != a->field)
        return MTX_ERR_INCOMPATIBLE_FIELD;
    if (b->precision != a->precision || c->precision != a->precision)
        return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (Atrans == mtx_notrans && Btrans == mtx_notrans &&
        (A->num_rows != C->num_rows || A->num_columns != B->num_rows || B->num_columns != C->num_columns))
        return MTX_ERR_INCOMPATIBLE_SIZE;
    if (Atrans == mtx_notrans && (Btrans == mtx_trans || Btrans == mtx_conjtrans) &&
        (A->num_rows != C->num_rows || A->num_columns != B->num_columns || B->num_rows != C->num_columns))
        return MTX_ERR_INCOMPATIBLE_SIZE;
    if ((Atrans == mtx_trans || Atrans == mtx_conjtrans) && Btrans == mtx_notrans &&
        (A->num_columns != C->num_rows || A->num_rows != B->num_rows || B->num_columns != C->num_columns))
        return MTX_ERR_INCOMPATIBLE_SIZE;
    if ((Atrans == mtx_trans || Atrans == mtx_conjtrans) && (Btrans == mtx_trans || Btrans == mtx_conjtrans) &&
        (A->num_columns != C->num_rows || A->num_rows != B->num_columns || B->num_rows != C->num_columns))
        return MTX_ERR_INCOMPATIBLE_SIZE;
    if (B->symmetry != mtx_unsymmetric || C->symmetry != mtx_unsymmetric)
        return MTX_ERR_INCOMPATIBLE_SYMMETRY;

    if (beta != 1) {
        int err = mtxblasdense_sscal(beta, C, num_flops);
        if (err) return err;
    }

    if (A->symmetry == mtx_unsymmetric) {
        if (a->field == mtx_field_real) {
            if (Atrans == mtx_notrans && Btrans == mtx_notrans) {
                if (a->precision == mtx_single) {
                    const float * Adata = a->data.real_single;
                    const float * Bdata = b->data.real_single;
                    float * Cdata = c->data.real_single;

                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < A->num_columns; k++)
                                Cdata[i*C->num_columns+j] += alpha*Adata[i*A->num_columns+k]*Bdata[k*B->num_columns+j];
                        }
                    }
                    if (num_flops) *num_flops += 3*C->num_rows*C->num_columns*A->num_columns;
                } else if (a->precision == mtx_double) {
                    const double * Adata = a->data.real_double;
                    const double * Bdata = b->data.real_double;
                    double * Cdata = c->data.real_double;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < A->num_columns; k++)
                                Cdata[i*C->num_columns+j] += alpha*Adata[i*A->num_columns+k]*Bdata[k*B->num_columns+j];
                        }
                    }
                    if (num_flops) *num_flops += 3*C->num_rows*C->num_columns*A->num_columns;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (Atrans == mtx_notrans && (Btrans == mtx_trans || Btrans == mtx_conjtrans)) {
                if (a->precision == mtx_single) {
                    const float * Adata = a->data.real_single;
                    const float * Bdata = b->data.real_single;
                    float * Cdata = c->data.real_single;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < A->num_columns; k++)
                                Cdata[i*C->num_columns+j] += alpha*Adata[i*A->num_columns+k]*Bdata[j*B->num_columns+k];
                        }
                    }
                    if (num_flops) *num_flops += 3*C->num_rows*C->num_columns*A->num_columns;
                } else if (a->precision == mtx_double) {
                    const double * Adata = a->data.real_double;
                    const double * Bdata = b->data.real_double;
                    double * Cdata = c->data.real_double;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < A->num_columns; k++)
                                Cdata[i*C->num_columns+j] += alpha*Adata[i*A->num_columns+k]*Bdata[j*B->num_columns+k];
                        }
                    }
                    if (num_flops) *num_flops += 3*C->num_rows*C->num_columns*A->num_columns;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if ((Atrans == mtx_trans || Atrans == mtx_conjtrans) && Btrans == mtx_notrans) {
                if (a->precision == mtx_single) {
                    const float * Adata = a->data.real_single;
                    const float * Bdata = b->data.real_single;
                    float * Cdata = c->data.real_single;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < A->num_rows; k++)
                                Cdata[i*C->num_columns+j] += alpha*Adata[k*A->num_columns+i]*Bdata[k*B->num_columns+j];
                        }
                    }
                    if (num_flops) *num_flops += 3*C->num_rows*C->num_columns*A->num_rows;
                } else if (a->precision == mtx_double) {
                    const double * Adata = a->data.real_double;
                    const double * Bdata = b->data.real_double;
                    double * Cdata = c->data.real_double;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < A->num_rows; k++)
                                Cdata[i*C->num_columns+j] += alpha*Adata[k*A->num_columns+i]*Bdata[k*B->num_columns+j];
                        }
                    }
                    if (num_flops) *num_flops += 3*C->num_rows*C->num_columns*A->num_rows;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if ((Atrans == mtx_trans || Atrans == mtx_conjtrans) && (Btrans == mtx_trans || Btrans == mtx_conjtrans)) {
                if (a->precision == mtx_single) {
                    const float * Adata = a->data.real_single;
                    const float * Bdata = b->data.real_single;
                    float * Cdata = c->data.real_single;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < A->num_rows; k++)
                                Cdata[i*C->num_columns+j] += alpha*Adata[k*A->num_columns+i]*Bdata[j*B->num_columns+k];
                        }
                    }
                    if (num_flops) *num_flops += 3*C->num_rows*C->num_columns*A->num_rows;
                } else if (a->precision == mtx_double) {
                    const double * Adata = a->data.real_double;
                    const double * Bdata = b->data.real_double;
                    double * Cdata = c->data.real_double;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < A->num_rows; k++)
                                Cdata[i*C->num_columns+j] += alpha*Adata[k*A->num_columns+i]*Bdata[j*B->num_columns+k];
                        }
                    }
                    if (num_flops) *num_flops += 3*C->num_rows*C->num_columns*A->num_rows;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else { return MTX_ERR_INVALID_TRANSPOSITION; }
        } else if (a->field == mtx_field_complex) {
            if (Atrans == mtx_notrans && Btrans == mtx_notrans) {
                if (a->precision == mtx_single) {
                    const float (* Adata)[2] = a->data.complex_single;
                    const float (* Bdata)[2] = b->data.complex_single;
                    float (* Cdata)[2] = c->data.complex_single;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < A->num_columns; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[i*A->num_columns+k][0]*Bdata[k*B->num_columns+j][0]-Adata[i*A->num_columns+k][1]*Bdata[k*B->num_columns+j][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(Adata[i*A->num_columns+k][0]*Bdata[k*B->num_columns+j][1]+Adata[i*A->num_columns+k][1]*Bdata[k*B->num_columns+j][0]);
                            }
                        }
                    }
                    if (num_flops) *num_flops += 10*C->num_rows*C->num_columns*A->num_columns;
                } else if (a->precision == mtx_double) {
                    const double (* Adata)[2] = a->data.complex_double;
                    const double (* Bdata)[2] = b->data.complex_double;
                    double (* Cdata)[2] = c->data.complex_double;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < A->num_columns; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[i*A->num_columns+k][0]*Bdata[k*B->num_columns+j][0]-Adata[i*A->num_columns+k][1]*Bdata[k*B->num_columns+j][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(Adata[i*A->num_columns+k][0]*Bdata[k*B->num_columns+j][1]+Adata[i*A->num_columns+k][1]*Bdata[k*B->num_columns+j][0]);
                            }
                        }
                    }
                    if (num_flops) *num_flops += 10*C->num_rows*C->num_columns*A->num_columns;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (Atrans == mtx_notrans && Btrans == mtx_trans) {
                if (a->precision == mtx_single) {
                    const float (* Adata)[2] = a->data.complex_single;
                    const float (* Bdata)[2] = b->data.complex_single;
                    float (* Cdata)[2] = c->data.complex_single;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < A->num_columns; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[i*A->num_columns+k][0]*Bdata[j*B->num_columns+k][0]-Adata[i*A->num_columns+k][1]*Bdata[j*B->num_columns+k][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(Adata[i*A->num_columns+k][0]*Bdata[j*B->num_columns+k][1]+Adata[i*A->num_columns+k][1]*Bdata[j*B->num_columns+k][0]);
                            }
                        }
                    }
                    if (num_flops) *num_flops += 10*C->num_rows*C->num_columns*A->num_columns;
                } else if (a->precision == mtx_double) {
                    const double (* Adata)[2] = a->data.complex_double;
                    const double (* Bdata)[2] = b->data.complex_double;
                    double (* Cdata)[2] = c->data.complex_double;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < A->num_columns; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[i*A->num_columns+k][0]*Bdata[j*B->num_columns+k][0]-Adata[i*A->num_columns+k][1]*Bdata[j*B->num_columns+k][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(Adata[i*A->num_columns+k][0]*Bdata[j*B->num_columns+k][1]+Adata[i*A->num_columns+k][1]*Bdata[j*B->num_columns+k][0]);
                            }
                        }
                    }
                    if (num_flops) *num_flops += 10*C->num_rows*C->num_columns*A->num_columns;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (Atrans == mtx_notrans && Btrans == mtx_conjtrans) {
                if (a->precision == mtx_single) {
                    const float (* Adata)[2] = a->data.complex_single;
                        const float (* Bdata)[2] = b->data.complex_single;
                    float (* Cdata)[2] = c->data.complex_single;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < A->num_columns; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[i*A->num_columns+k][0]*Bdata[j*B->num_columns+k][0]+Adata[i*A->num_columns+k][1]*Bdata[j*B->num_columns+k][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(-Adata[i*A->num_columns+k][0]*Bdata[j*B->num_columns+k][1]+Adata[i*A->num_columns+k][1]*Bdata[j*B->num_columns+k][0]);
                            }
                        }
                    }
                    if (num_flops) *num_flops += 10*C->num_rows*C->num_columns*A->num_columns;
                } else if (a->precision == mtx_double) {
                    const double (* Adata)[2] = a->data.complex_double;
                    const double (* Bdata)[2] = b->data.complex_double;
                    double (* Cdata)[2] = c->data.complex_double;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < A->num_columns; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[i*A->num_columns+k][0]*Bdata[j*B->num_columns+k][0]+Adata[i*A->num_columns+k][1]*Bdata[j*B->num_columns+k][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(-Adata[i*A->num_columns+k][0]*Bdata[j*B->num_columns+k][1]+Adata[i*A->num_columns+k][1]*Bdata[j*B->num_columns+k][0]);
                            }
                        }
                    }
                    if (num_flops) *num_flops += 10*C->num_rows*C->num_columns*A->num_columns;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (Atrans == mtx_trans && Btrans == mtx_notrans) {
                if (a->precision == mtx_single) {
                    const float (* Adata)[2] = a->data.complex_single;
                    const float (* Bdata)[2] = b->data.complex_single;
                    float (* Cdata)[2] = c->data.complex_single;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < A->num_rows; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[k*A->num_columns+i][0]*Bdata[k*B->num_columns+j][0]-Adata[k*A->num_columns+i][1]*Bdata[k*B->num_columns+j][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(Adata[k*A->num_columns+i][0]*Bdata[k*B->num_columns+j][1]+Adata[k*A->num_columns+i][1]*Bdata[k*B->num_columns+j][0]);
                            }
                        }
                    }
                    if (num_flops) *num_flops += 10*C->num_rows*C->num_columns*A->num_rows;
                } else if (a->precision == mtx_double) {
                    const double (* Adata)[2] = a->data.complex_double;
                    const double (* Bdata)[2] = b->data.complex_double;
                    double (* Cdata)[2] = c->data.complex_double;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < A->num_rows; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[k*A->num_columns+i][0]*Bdata[k*B->num_columns+j][0]-Adata[k*A->num_columns+i][1]*Bdata[k*B->num_columns+j][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(Adata[k*A->num_columns+i][0]*Bdata[k*B->num_columns+j][1]+Adata[k*A->num_columns+i][1]*Bdata[k*B->num_columns+j][0]);
                            }
                        }
                    }
                    if (num_flops) *num_flops += 10*C->num_rows*C->num_columns*A->num_rows;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (Atrans == mtx_trans && Btrans == mtx_trans) {
                if (a->precision == mtx_single) {
                    const float (* Adata)[2] = a->data.complex_single;
                    const float (* Bdata)[2] = b->data.complex_single;
                    float (* Cdata)[2] = c->data.complex_single;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < A->num_rows; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[k*A->num_columns+i][0]*Bdata[j*B->num_columns+k][0]-Adata[k*A->num_columns+i][1]*Bdata[j*B->num_columns+k][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(Adata[k*A->num_columns+i][0]*Bdata[j*B->num_columns+k][1]+Adata[k*A->num_columns+i][1]*Bdata[j*B->num_columns+k][0]);
                            }
                        }
                    }
                    if (num_flops) *num_flops += 10*C->num_rows*C->num_columns*A->num_rows;
                } else if (a->precision == mtx_double) {
                    const double (* Adata)[2] = a->data.complex_double;
                    const double (* Bdata)[2] = b->data.complex_double;
                    double (* Cdata)[2] = c->data.complex_double;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < A->num_rows; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[k*A->num_columns+i][0]*Bdata[j*B->num_columns+k][0]-Adata[k*A->num_columns+i][1]*Bdata[j*B->num_columns+k][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(Adata[k*A->num_columns+i][0]*Bdata[j*B->num_columns+k][1]+Adata[k*A->num_columns+i][1]*Bdata[j*B->num_columns+k][0]);
                            }
                        }
                    }
                    if (num_flops) *num_flops += 10*C->num_rows*C->num_columns*A->num_rows;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (Atrans == mtx_trans && Btrans == mtx_conjtrans) {
                if (a->precision == mtx_single) {
                    const float (* Adata)[2] = a->data.complex_single;
                    const float (* Bdata)[2] = b->data.complex_single;
                    float (* Cdata)[2] = c->data.complex_single;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < A->num_rows; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[k*A->num_columns+i][0]*Bdata[j*B->num_columns+k][0]+Adata[k*A->num_columns+i][1]*Bdata[j*B->num_columns+k][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(-Adata[k*A->num_columns+i][0]*Bdata[j*B->num_columns+k][1]+Adata[k*A->num_columns+i][1]*Bdata[j*B->num_columns+k][0]);
                            }
                        }
                    }
                    if (num_flops) *num_flops += 10*C->num_rows*C->num_columns*A->num_rows;
                } else if (a->precision == mtx_double) {
                    const double (* Adata)[2] = a->data.complex_double;
                    const double (* Bdata)[2] = b->data.complex_double;
                    double (* Cdata)[2] = c->data.complex_double;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < A->num_rows; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[k*A->num_columns+i][0]*Bdata[j*B->num_columns+k][0]+Adata[k*A->num_columns+i][1]*Bdata[j*B->num_columns+k][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(-Adata[k*A->num_columns+i][0]*Bdata[j*B->num_columns+k][1]+Adata[k*A->num_columns+i][1]*Bdata[j*B->num_columns+k][0]);
                            }
                        }
                    }
                    if (num_flops) *num_flops += 10*C->num_rows*C->num_columns*A->num_rows;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (Atrans == mtx_conjtrans && Btrans == mtx_notrans) {
                if (a->precision == mtx_single) {
                    const float (* Adata)[2] = a->data.complex_single;
                    const float (* Bdata)[2] = b->data.complex_single;
                    float (* Cdata)[2] = c->data.complex_single;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < A->num_rows; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[k*A->num_columns+i][0]*Bdata[k*B->num_columns+j][0]+Adata[k*A->num_columns+i][1]*Bdata[k*B->num_columns+j][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(Adata[k*A->num_columns+i][0]*Bdata[k*B->num_columns+j][1]-Adata[k*A->num_columns+i][1]*Bdata[k*B->num_columns+j][0]);
                            }
                        }
                    }
                    if (num_flops) *num_flops += 10*C->num_rows*C->num_columns*A->num_rows;
                } else if (a->precision == mtx_double) {
                    const double (* Adata)[2] = a->data.complex_double;
                    const double (* Bdata)[2] = b->data.complex_double;
                    double (* Cdata)[2] = c->data.complex_double;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < A->num_rows; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[k*A->num_columns+i][0]*Bdata[k*B->num_columns+j][0]+Adata[k*A->num_columns+i][1]*Bdata[k*B->num_columns+j][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(Adata[k*A->num_columns+i][0]*Bdata[k*B->num_columns+j][1]-Adata[k*A->num_columns+i][1]*Bdata[k*B->num_columns+j][0]);
                            }
                        }
                    }
                    if (num_flops) *num_flops += 10*C->num_rows*C->num_columns*A->num_rows;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else { return MTX_ERR_INVALID_TRANSPOSITION; }
        } else if (a->field == mtx_field_integer) {
            if (Atrans == mtx_notrans && Btrans == mtx_notrans) {
                if (a->precision == mtx_single) {
                    const int32_t * Adata = a->data.integer_single;
                    const int32_t * Bdata = b->data.integer_single;
                    int32_t * Cdata = c->data.integer_single;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < A->num_columns; k++)
                                Cdata[i*C->num_columns+j] += alpha*Adata[i*A->num_columns+k]*Bdata[k*B->num_columns+j];
                        }
                    }
                    if (num_flops) *num_flops += 3*C->num_rows*C->num_columns*A->num_columns;
                } else if (a->precision == mtx_double) {
                    const int64_t * Adata = a->data.integer_double;
                    const int64_t * Bdata = b->data.integer_double;
                    int64_t * Cdata = c->data.integer_double;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < A->num_columns; k++)
                                Cdata[i*C->num_columns+j] += alpha*Adata[i*A->num_columns+k]*Bdata[k*B->num_columns+j];
                        }
                    }
                    if (num_flops) *num_flops += 3*C->num_rows*C->num_columns*A->num_columns;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (Atrans == mtx_notrans && (Btrans == mtx_trans || Btrans == mtx_conjtrans)) {
                if (a->precision == mtx_single) {
                    const int32_t * Adata = a->data.integer_single;
                    const int32_t * Bdata = b->data.integer_single;
                    int32_t * Cdata = c->data.integer_single;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < A->num_columns; k++)
                                Cdata[i*C->num_columns+j] += alpha*Adata[i*A->num_columns+k]*Bdata[j*B->num_columns+k];
                        }
                    }
                    if (num_flops) *num_flops += 3*C->num_rows*C->num_columns*A->num_columns;
                } else if (a->precision == mtx_double) {
                    const int64_t * Adata = a->data.integer_double;
                    const int64_t * Bdata = b->data.integer_double;
                    int64_t * Cdata = c->data.integer_double;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < A->num_columns; k++)
                                Cdata[i*C->num_columns+j] += alpha*Adata[i*A->num_columns+k]*Bdata[j*B->num_columns+k];
                        }
                    }
                    if (num_flops) *num_flops += 3*C->num_rows*C->num_columns*A->num_columns;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if ((Atrans == mtx_trans || Atrans == mtx_conjtrans) && Btrans == mtx_notrans) {
                if (a->precision == mtx_single) {
                    const int32_t * Adata = a->data.integer_single;
                    const int32_t * Bdata = b->data.integer_single;
                    int32_t * Cdata = c->data.integer_single;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < A->num_rows; k++)
                                Cdata[i*C->num_columns+j] += alpha*Adata[k*A->num_columns+i]*Bdata[k*B->num_columns+j];
                        }
                    }
                    if (num_flops) *num_flops += 3*C->num_rows*C->num_columns*A->num_rows;
                } else if (a->precision == mtx_double) {
                    const int64_t * Adata = a->data.integer_double;
                    const int64_t * Bdata = b->data.integer_double;
                    int64_t * Cdata = c->data.integer_double;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < A->num_rows; k++)
                                Cdata[i*C->num_columns+j] += alpha*Adata[k*A->num_columns+i]*Bdata[k*B->num_columns+j];
                        }
                    }
                    if (num_flops) *num_flops += 3*C->num_rows*C->num_columns*A->num_rows;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if ((Atrans == mtx_trans || Atrans == mtx_conjtrans) && (Btrans == mtx_trans || Btrans == mtx_conjtrans)) {
                if (a->precision == mtx_single) {
                    const int32_t * Adata = a->data.integer_single;
                    const int32_t * Bdata = b->data.integer_single;
                    int32_t * Cdata = c->data.integer_single;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < A->num_rows; k++)
                                Cdata[i*C->num_columns+j] += alpha*Adata[k*A->num_columns+i]*Bdata[j*B->num_columns+k];
                        }
                    }
                    if (num_flops) *num_flops += 3*C->num_rows*C->num_columns*A->num_rows;
                } else if (a->precision == mtx_double) {
                    const int64_t * Adata = a->data.integer_double;
                    const int64_t * Bdata = b->data.integer_double;
                    int64_t * Cdata = c->data.integer_double;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < A->num_rows; k++)
                                Cdata[i*C->num_columns+j] += alpha*Adata[k*A->num_columns+i]*Bdata[j*B->num_columns+k];
                        }
                    }
                    if (num_flops) *num_flops += 3*C->num_rows*C->num_columns*A->num_rows;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else { return MTX_ERR_INVALID_TRANSPOSITION; }
        } else { return MTX_ERR_INVALID_FIELD; }
    } else if (A->num_rows == A->num_columns && A->symmetry == mtx_symmetric) {
        if (a->field == mtx_field_real) {
            if (Btrans == mtx_notrans) {
                if (a->precision == mtx_single) {
                    const float * Adata = a->data.real_single;
                    const float * Bdata = b->data.real_single;
                    float * Cdata = c->data.real_single;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < i; k++)
                                Cdata[i*C->num_columns+j] += alpha*Adata[touptri(A->num_columns,k,i)]*Bdata[k*B->num_columns+j];
                            for (int k = i; k < A->num_columns; k++)
                                Cdata[i*C->num_columns+j] += alpha*Adata[touptri(A->num_columns,i,k)]*Bdata[k*B->num_columns+j];
                        }
                    }
                    if (num_flops) *num_flops += 3*C->num_rows*C->num_columns*A->num_columns;
                } else if (a->precision == mtx_double) {
                    const double * Adata = a->data.real_double;
                    const double * Bdata = b->data.real_double;
                    double * Cdata = c->data.real_double;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < i; k++)
                                Cdata[i*C->num_columns+j] += alpha*Adata[touptri(A->num_columns,k,i)]*Bdata[k*B->num_columns+j];
                            for (int k = i; k < A->num_columns; k++)
                                Cdata[i*C->num_columns+j] += alpha*Adata[touptri(A->num_columns,i,k)]*Bdata[k*B->num_columns+j];
                        }
                    }
                    if (num_flops) *num_flops += 3*C->num_rows*C->num_columns*A->num_columns;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (Btrans == mtx_trans || Btrans == mtx_conjtrans) {
                if (a->precision == mtx_single) {
                    const float * Adata = a->data.real_single;
                    const float * Bdata = b->data.real_single;
                    float * Cdata = c->data.real_single;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < i; k++)
                                Cdata[i*C->num_columns+j] += alpha*Adata[touptri(A->num_columns,k,i)]*Bdata[j*B->num_columns+k];
                            for (int k = i; k < A->num_columns; k++)
                                Cdata[i*C->num_columns+j] += alpha*Adata[touptri(A->num_columns,i,k)]*Bdata[j*B->num_columns+k];
                        }
                    }
                    if (num_flops) *num_flops += 3*C->num_rows*C->num_columns*A->num_columns;
                } else if (a->precision == mtx_double) {
                    const double * Adata = a->data.real_double;
                    const double * Bdata = b->data.real_double;
                    double * Cdata = c->data.real_double;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < i; k++)
                                Cdata[i*C->num_columns+j] += alpha*Adata[touptri(A->num_columns,k,i)]*Bdata[j*B->num_columns+k];
                            for (int k = i; k < A->num_columns; k++)
                                Cdata[i*C->num_columns+j] += alpha*Adata[touptri(A->num_columns,i,k)]*Bdata[j*B->num_columns+k];
                        }
                    }
                    if (num_flops) *num_flops += 3*C->num_rows*C->num_columns*A->num_columns;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else { return MTX_ERR_INVALID_TRANSPOSITION; }
        } else if (a->field == mtx_field_complex) {
            if ((Atrans == mtx_notrans || Atrans == mtx_trans) && Btrans == mtx_notrans) {
                if (a->precision == mtx_single) {
                    const float (* Adata)[2] = a->data.complex_single;
                    const float (* Bdata)[2] = b->data.complex_single;
                    float (* Cdata)[2] = c->data.complex_single;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < i; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,k,i)][0]*Bdata[k*B->num_columns+j][0]-Adata[touptri(A->num_columns,k,i)][1]*Bdata[k*B->num_columns+j][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(Adata[touptri(A->num_columns,k,i)][0]*Bdata[k*B->num_columns+j][1]+Adata[touptri(A->num_columns,k,i)][1]*Bdata[k*B->num_columns+j][0]);
                            }
                            for (int k = i; k < A->num_columns; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,i,k)][0]*Bdata[k*B->num_columns+j][0]-Adata[touptri(A->num_columns,i,k)][1]*Bdata[k*B->num_columns+j][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(Adata[touptri(A->num_columns,i,k)][0]*Bdata[k*B->num_columns+j][1]+Adata[touptri(A->num_columns,i,k)][1]*Bdata[k*B->num_columns+j][0]);
                            }
                        }
                    }
                    if (num_flops) *num_flops += 10*C->num_rows*C->num_columns*A->num_columns;
                } else if (a->precision == mtx_double) {
                    const double (* Adata)[2] = a->data.complex_double;
                    const double (* Bdata)[2] = b->data.complex_double;
                    double (* Cdata)[2] = c->data.complex_double;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < i; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,k,i)][0]*Bdata[k*B->num_columns+j][0]-Adata[touptri(A->num_columns,k,i)][1]*Bdata[k*B->num_columns+j][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(Adata[touptri(A->num_columns,k,i)][0]*Bdata[k*B->num_columns+j][1]+Adata[touptri(A->num_columns,k,i)][1]*Bdata[k*B->num_columns+j][0]);
                            }
                            for (int k = i; k < A->num_columns; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,i,k)][0]*Bdata[k*B->num_columns+j][0]-Adata[touptri(A->num_columns,i,k)][1]*Bdata[k*B->num_columns+j][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(Adata[touptri(A->num_columns,i,k)][0]*Bdata[k*B->num_columns+j][1]+Adata[touptri(A->num_columns,i,k)][1]*Bdata[k*B->num_columns+j][0]);
                            }
                        }
                    }
                    if (num_flops) *num_flops += 10*C->num_rows*C->num_columns*A->num_columns;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if ((Atrans == mtx_notrans || Atrans == mtx_trans) && Btrans == mtx_trans) {
                if (a->precision == mtx_single) {
                    const float (* Adata)[2] = a->data.complex_single;
                    const float (* Bdata)[2] = b->data.complex_single;
                    float (* Cdata)[2] = c->data.complex_single;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < i; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,k,i)][0]*Bdata[j*B->num_columns+k][0]-Adata[touptri(A->num_columns,k,i)][1]*Bdata[j*B->num_columns+k][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(Adata[touptri(A->num_columns,k,i)][0]*Bdata[j*B->num_columns+k][1]+Adata[touptri(A->num_columns,k,i)][1]*Bdata[j*B->num_columns+k][0]);
                            }
                            for (int k = i; k < A->num_columns; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,i,k)][0]*Bdata[j*B->num_columns+k][0]-Adata[touptri(A->num_columns,i,k)][1]*Bdata[j*B->num_columns+k][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(Adata[touptri(A->num_columns,i,k)][0]*Bdata[j*B->num_columns+k][1]+Adata[touptri(A->num_columns,i,k)][1]*Bdata[j*B->num_columns+k][0]);
                            }
                        }
                    }
                    if (num_flops) *num_flops += 10*C->num_rows*C->num_columns*A->num_columns;
                } else if (a->precision == mtx_double) {
                    const double (* Adata)[2] = a->data.complex_double;
                    const double (* Bdata)[2] = b->data.complex_double;
                    double (* Cdata)[2] = c->data.complex_double;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < i; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,k,i)][0]*Bdata[j*B->num_columns+k][0]-Adata[touptri(A->num_columns,k,i)][1]*Bdata[j*B->num_columns+k][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(Adata[touptri(A->num_columns,k,i)][0]*Bdata[j*B->num_columns+k][1]+Adata[touptri(A->num_columns,k,i)][1]*Bdata[j*B->num_columns+k][0]);
                            }
                            for (int k = i; k < A->num_columns; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,i,k)][0]*Bdata[j*B->num_columns+k][0]-Adata[touptri(A->num_columns,i,k)][1]*Bdata[j*B->num_columns+k][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(Adata[touptri(A->num_columns,i,k)][0]*Bdata[j*B->num_columns+k][1]+Adata[touptri(A->num_columns,i,k)][1]*Bdata[j*B->num_columns+k][0]);
                            }
                        }
                    }
                    if (num_flops) *num_flops += 10*C->num_rows*C->num_columns*A->num_columns;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if ((Atrans == mtx_notrans || Atrans == mtx_trans) && Btrans == mtx_conjtrans) {
                if (a->precision == mtx_single) {
                    const float (* Adata)[2] = a->data.complex_single;
                    const float (* Bdata)[2] = b->data.complex_single;
                    float (* Cdata)[2] = c->data.complex_single;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < i; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,k,i)][0]*Bdata[j*B->num_columns+k][0]+Adata[touptri(A->num_columns,k,i)][1]*Bdata[j*B->num_columns+k][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(-Adata[touptri(A->num_columns,k,i)][0]*Bdata[j*B->num_columns+k][1]+Adata[touptri(A->num_columns,k,i)][1]*Bdata[j*B->num_columns+k][0]);
                            }
                            for (int k = i; k < A->num_columns; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,i,k)][0]*Bdata[j*B->num_columns+k][0]+Adata[touptri(A->num_columns,i,k)][1]*Bdata[j*B->num_columns+k][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(-Adata[touptri(A->num_columns,i,k)][0]*Bdata[j*B->num_columns+k][1]+Adata[touptri(A->num_columns,i,k)][1]*Bdata[j*B->num_columns+k][0]);
                            }
                        }
                    }
                    if (num_flops) *num_flops += 10*C->num_rows*C->num_columns*A->num_columns;
                } else if (a->precision == mtx_double) {
                    const double (* Adata)[2] = a->data.complex_double;
                    const double (* Bdata)[2] = b->data.complex_double;
                    double (* Cdata)[2] = c->data.complex_double;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < i; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,k,i)][0]*Bdata[j*B->num_columns+k][0]+Adata[touptri(A->num_columns,k,i)][1]*Bdata[j*B->num_columns+k][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(-Adata[touptri(A->num_columns,k,i)][0]*Bdata[j*B->num_columns+k][1]+Adata[touptri(A->num_columns,k,i)][1]*Bdata[j*B->num_columns+k][0]);
                            }
                            for (int k = i; k < A->num_columns; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,i,k)][0]*Bdata[j*B->num_columns+k][0]+Adata[touptri(A->num_columns,i,k)][1]*Bdata[j*B->num_columns+k][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(-Adata[touptri(A->num_columns,i,k)][0]*Bdata[j*B->num_columns+k][1]+Adata[touptri(A->num_columns,i,k)][1]*Bdata[j*B->num_columns+k][0]);
                            }
                        }
                    }
                    if (num_flops) *num_flops += 10*C->num_rows*C->num_columns*A->num_columns;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (Atrans == mtx_conjtrans && Btrans == mtx_notrans) {
                if (a->precision == mtx_single) {
                    const float (* Adata)[2] = a->data.complex_single;
                    const float (* Bdata)[2] = b->data.complex_single;
                    float (* Cdata)[2] = c->data.complex_single;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < i; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,k,i)][0]*Bdata[k*B->num_columns+j][0]+Adata[touptri(A->num_columns,k,i)][1]*Bdata[k*B->num_columns+j][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(Adata[touptri(A->num_columns,k,i)][0]*Bdata[k*B->num_columns+j][1]-Adata[touptri(A->num_columns,k,i)][1]*Bdata[k*B->num_columns+j][0]);
                            }
                            for (int k = i; k < A->num_columns; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,i,k)][0]*Bdata[k*B->num_columns+j][0]+Adata[touptri(A->num_columns,i,k)][1]*Bdata[k*B->num_columns+j][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(Adata[touptri(A->num_columns,i,k)][0]*Bdata[k*B->num_columns+j][1]-Adata[touptri(A->num_columns,i,k)][1]*Bdata[k*B->num_columns+j][0]);
                            }
                        }
                    }
                    if (num_flops) *num_flops += 10*C->num_rows*C->num_columns*A->num_columns;
                } else if (a->precision == mtx_double) {
                    const double (* Adata)[2] = a->data.complex_double;
                    const double (* Bdata)[2] = b->data.complex_double;
                    double (* Cdata)[2] = c->data.complex_double;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < i; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,k,i)][0]*Bdata[k*B->num_columns+j][0]+Adata[touptri(A->num_columns,k,i)][1]*Bdata[k*B->num_columns+j][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(Adata[touptri(A->num_columns,k,i)][0]*Bdata[k*B->num_columns+j][1]-Adata[touptri(A->num_columns,k,i)][1]*Bdata[k*B->num_columns+j][0]);
                            }
                            for (int k = i; k < A->num_columns; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,i,k)][0]*Bdata[k*B->num_columns+j][0]+Adata[touptri(A->num_columns,i,k)][1]*Bdata[k*B->num_columns+j][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(Adata[touptri(A->num_columns,i,k)][0]*Bdata[k*B->num_columns+j][1]-Adata[touptri(A->num_columns,i,k)][1]*Bdata[k*B->num_columns+j][0]);
                            }
                        }
                    }
                    if (num_flops) *num_flops += 10*C->num_rows*C->num_columns*A->num_columns;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (Atrans == mtx_conjtrans && Btrans == mtx_trans) {
                if (a->precision == mtx_single) {
                    const float (* Adata)[2] = a->data.complex_single;
                    const float (* Bdata)[2] = b->data.complex_single;
                    float (* Cdata)[2] = c->data.complex_single;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < i; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,k,i)][0]*Bdata[j*B->num_columns+k][0]+Adata[touptri(A->num_columns,k,i)][1]*Bdata[j*B->num_columns+k][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(Adata[touptri(A->num_columns,k,i)][0]*Bdata[j*B->num_columns+k][1]-Adata[touptri(A->num_columns,k,i)][1]*Bdata[j*B->num_columns+k][0]);
                            }
                            for (int k = i; k < A->num_columns; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,i,k)][0]*Bdata[j*B->num_columns+k][0]+Adata[touptri(A->num_columns,i,k)][1]*Bdata[j*B->num_columns+k][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(Adata[touptri(A->num_columns,i,k)][0]*Bdata[j*B->num_columns+k][1]-Adata[touptri(A->num_columns,i,k)][1]*Bdata[j*B->num_columns+k][0]);
                            }
                        }
                    }
                    if (num_flops) *num_flops += 10*C->num_rows*C->num_columns*A->num_columns;
                } else if (a->precision == mtx_double) {
                    const double (* Adata)[2] = a->data.complex_double;
                    const double (* Bdata)[2] = b->data.complex_double;
                    double (* Cdata)[2] = c->data.complex_double;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < i; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,k,i)][0]*Bdata[j*B->num_columns+k][0]+Adata[touptri(A->num_columns,k,i)][1]*Bdata[j*B->num_columns+k][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(Adata[touptri(A->num_columns,k,i)][0]*Bdata[j*B->num_columns+k][1]-Adata[touptri(A->num_columns,k,i)][1]*Bdata[j*B->num_columns+k][0]);
                            }
                            for (int k = i; k < A->num_columns; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,i,k)][0]*Bdata[j*B->num_columns+k][0]+Adata[touptri(A->num_columns,i,k)][1]*Bdata[j*B->num_columns+k][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(Adata[touptri(A->num_columns,i,k)][0]*Bdata[j*B->num_columns+k][1]-Adata[touptri(A->num_columns,i,k)][1]*Bdata[j*B->num_columns+k][0]);
                            }
                        }
                    }
                    if (num_flops) *num_flops += 10*C->num_rows*C->num_columns*A->num_columns;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (Atrans == mtx_conjtrans && Btrans == mtx_conjtrans) {
                if (a->precision == mtx_single) {
                    const float (* Adata)[2] = a->data.complex_single;
                    const float (* Bdata)[2] = b->data.complex_single;
                    float (* Cdata)[2] = c->data.complex_single;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < i; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,k,i)][0]*Bdata[j*B->num_columns+k][0]-Adata[touptri(A->num_columns,k,i)][1]*Bdata[j*B->num_columns+k][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(-Adata[touptri(A->num_columns,k,i)][0]*Bdata[j*B->num_columns+k][1]-Adata[touptri(A->num_columns,k,i)][1]*Bdata[j*B->num_columns+k][0]);
                            }
                            for (int k = i; k < A->num_columns; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,i,k)][0]*Bdata[j*B->num_columns+k][0]-Adata[touptri(A->num_columns,i,k)][1]*Bdata[j*B->num_columns+k][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(-Adata[touptri(A->num_columns,i,k)][0]*Bdata[j*B->num_columns+k][1]-Adata[touptri(A->num_columns,i,k)][1]*Bdata[j*B->num_columns+k][0]);
                            }
                        }
                    }
                    if (num_flops) *num_flops += 10*C->num_rows*C->num_columns*A->num_columns;
                } else if (a->precision == mtx_double) {
                    const double (* Adata)[2] = a->data.complex_double;
                    const double (* Bdata)[2] = b->data.complex_double;
                    double (* Cdata)[2] = c->data.complex_double;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < i; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,k,i)][0]*Bdata[j*B->num_columns+k][0]-Adata[touptri(A->num_columns,k,i)][1]*Bdata[j*B->num_columns+k][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(-Adata[touptri(A->num_columns,k,i)][0]*Bdata[j*B->num_columns+k][1]-Adata[touptri(A->num_columns,k,i)][1]*Bdata[j*B->num_columns+k][0]);
                            }
                            for (int k = i; k < A->num_columns; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,i,k)][0]*Bdata[j*B->num_columns+k][0]-Adata[touptri(A->num_columns,i,k)][1]*Bdata[j*B->num_columns+k][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(-Adata[touptri(A->num_columns,i,k)][0]*Bdata[j*B->num_columns+k][1]-Adata[touptri(A->num_columns,i,k)][1]*Bdata[j*B->num_columns+k][0]);
                            }
                        }
                    }
                    if (num_flops) *num_flops += 10*C->num_rows*C->num_columns*A->num_columns;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else { return MTX_ERR_INVALID_TRANSPOSITION; }
        } else if (a->field == mtx_field_integer) {
            if (Btrans == mtx_notrans) {
                if (a->precision == mtx_single) {
                    const int32_t * Adata = a->data.integer_single;
                    const int32_t * Bdata = b->data.integer_single;
                    int32_t * Cdata = c->data.integer_single;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < i; k++)
                                Cdata[i*C->num_columns+j] += alpha*Adata[touptri(A->num_columns,k,i)]*Bdata[k*B->num_columns+j];
                            for (int k = i; k < A->num_columns; k++)
                                Cdata[i*C->num_columns+j] += alpha*Adata[touptri(A->num_columns,i,k)]*Bdata[k*B->num_columns+j];
                        }
                    }
                    if (num_flops) *num_flops += 3*C->num_rows*C->num_columns*A->num_columns;
                } else if (a->precision == mtx_double) {
                    const int64_t * Adata = a->data.integer_double;
                    const int64_t * Bdata = b->data.integer_double;
                    int64_t * Cdata = c->data.integer_double;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < i; k++)
                                Cdata[i*C->num_columns+j] += alpha*Adata[touptri(A->num_columns,k,i)]*Bdata[k*B->num_columns+j];
                            for (int k = i; k < A->num_columns; k++)
                                Cdata[i*C->num_columns+j] += alpha*Adata[touptri(A->num_columns,i,k)]*Bdata[k*B->num_columns+j];
                        }
                    }
                    if (num_flops) *num_flops += 3*C->num_rows*C->num_columns*A->num_columns;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (Btrans == mtx_trans || Btrans == mtx_conjtrans) {
                if (a->precision == mtx_single) {
                    const int32_t * Adata = a->data.integer_single;
                    const int32_t * Bdata = b->data.integer_single;
                    int32_t * Cdata = c->data.integer_single;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < i; k++)
                                Cdata[i*C->num_columns+j] += alpha*Adata[touptri(A->num_columns,k,i)]*Bdata[j*B->num_columns+k];
                            for (int k = i; k < A->num_columns; k++)
                                Cdata[i*C->num_columns+j] += alpha*Adata[touptri(A->num_columns,i,k)]*Bdata[j*B->num_columns+k];
                        }
                    }
                    if (num_flops) *num_flops += 3*C->num_rows*C->num_columns*A->num_columns;
                } else if (a->precision == mtx_double) {
                    const int64_t * Adata = a->data.integer_double;
                    const int64_t * Bdata = b->data.integer_double;
                    int64_t * Cdata = c->data.integer_double;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < i; k++)
                                Cdata[i*C->num_columns+j] += alpha*Adata[touptri(A->num_columns,k,i)]*Bdata[j*B->num_columns+k];
                            for (int k = i; k < A->num_columns; k++)
                                Cdata[i*C->num_columns+j] += alpha*Adata[touptri(A->num_columns,i,k)]*Bdata[j*B->num_columns+k];
                        }
                    }
                    if (num_flops) *num_flops += 3*C->num_rows*C->num_columns*A->num_columns;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else { return MTX_ERR_INVALID_TRANSPOSITION; }
        } else { return MTX_ERR_INVALID_FIELD; }
    } else if (A->num_rows == A->num_columns && A->symmetry == mtx_skew_symmetric) {
        /* TODO: allow skew-symmetric matrices */
        return MTX_ERR_INVALID_SYMMETRY;
    } else if (A->num_rows == A->num_columns && A->symmetry == mtx_hermitian) {
        if (a->field == mtx_field_complex) {
            if ((Atrans == mtx_notrans || Atrans == mtx_conjtrans) && Btrans == mtx_notrans) {
                if (a->precision == mtx_single) {
                    const float (* Adata)[2] = a->data.complex_single;
                    const float (* Bdata)[2] = b->data.complex_single;
                    float (* Cdata)[2] = c->data.complex_single;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < i; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,k,i)][0]*Bdata[k*B->num_columns+j][0]+Adata[touptri(A->num_columns,k,i)][1]*Bdata[k*B->num_columns+j][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(Adata[touptri(A->num_columns,k,i)][0]*Bdata[k*B->num_columns+j][1]-Adata[touptri(A->num_columns,k,i)][1]*Bdata[k*B->num_columns+j][0]);
                            }
                            for (int k = i; k < A->num_columns; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,i,k)][0]*Bdata[k*B->num_columns+j][0]-Adata[touptri(A->num_columns,i,k)][1]*Bdata[k*B->num_columns+j][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(Adata[touptri(A->num_columns,i,k)][0]*Bdata[k*B->num_columns+j][1]+Adata[touptri(A->num_columns,i,k)][1]*Bdata[k*B->num_columns+j][0]);
                            }
                        }
                    }
                    if (num_flops) *num_flops += 10*C->num_rows*C->num_columns*A->num_columns;
                } else if (a->precision == mtx_double) {
                    const double (* Adata)[2] = a->data.complex_double;
                    const double (* Bdata)[2] = b->data.complex_double;
                    double (* Cdata)[2] = c->data.complex_double;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < i; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,k,i)][0]*Bdata[k*B->num_columns+j][0]+Adata[touptri(A->num_columns,k,i)][1]*Bdata[k*B->num_columns+j][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(Adata[touptri(A->num_columns,k,i)][0]*Bdata[k*B->num_columns+j][1]-Adata[touptri(A->num_columns,k,i)][1]*Bdata[k*B->num_columns+j][0]);
                            }
                            for (int k = i; k < A->num_columns; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,i,k)][0]*Bdata[k*B->num_columns+j][0]-Adata[touptri(A->num_columns,i,k)][1]*Bdata[k*B->num_columns+j][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(Adata[touptri(A->num_columns,i,k)][0]*Bdata[k*B->num_columns+j][1]+Adata[touptri(A->num_columns,i,k)][1]*Bdata[k*B->num_columns+j][0]);
                            }
                        }
                    }
                    if (num_flops) *num_flops += 10*C->num_rows*C->num_columns*A->num_columns;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if ((Atrans == mtx_notrans || Atrans == mtx_conjtrans) && Btrans == mtx_trans) {
                if (a->precision == mtx_single) {
                    const float (* Adata)[2] = a->data.complex_single;
                    const float (* Bdata)[2] = b->data.complex_single;
                    float (* Cdata)[2] = c->data.complex_single;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < i; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,k,i)][0]*Bdata[j*B->num_columns+k][0]+Adata[touptri(A->num_columns,k,i)][1]*Bdata[j*B->num_columns+k][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(Adata[touptri(A->num_columns,k,i)][0]*Bdata[j*B->num_columns+k][1]-Adata[touptri(A->num_columns,k,i)][1]*Bdata[j*B->num_columns+k][0]);
                            }
                            for (int k = i; k < A->num_columns; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,i,k)][0]*Bdata[j*B->num_columns+k][0]-Adata[touptri(A->num_columns,i,k)][1]*Bdata[j*B->num_columns+k][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(Adata[touptri(A->num_columns,i,k)][0]*Bdata[j*B->num_columns+k][1]+Adata[touptri(A->num_columns,i,k)][1]*Bdata[j*B->num_columns+k][0]);
                            }
                        }
                    }
                    if (num_flops) *num_flops += 10*C->num_rows*C->num_columns*A->num_columns;
                } else if (a->precision == mtx_double) {
                    const double (* Adata)[2] = a->data.complex_double;
                    const double (* Bdata)[2] = b->data.complex_double;
                    double (* Cdata)[2] = c->data.complex_double;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < i; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,k,i)][0]*Bdata[j*B->num_columns+k][0]+Adata[touptri(A->num_columns,k,i)][1]*Bdata[j*B->num_columns+k][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(Adata[touptri(A->num_columns,k,i)][0]*Bdata[j*B->num_columns+k][1]-Adata[touptri(A->num_columns,k,i)][1]*Bdata[j*B->num_columns+k][0]);
                            }
                            for (int k = i; k < A->num_columns; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,i,k)][0]*Bdata[j*B->num_columns+k][0]-Adata[touptri(A->num_columns,i,k)][1]*Bdata[j*B->num_columns+k][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(Adata[touptri(A->num_columns,i,k)][0]*Bdata[j*B->num_columns+k][1]+Adata[touptri(A->num_columns,i,k)][1]*Bdata[j*B->num_columns+k][0]);
                            }
                        }
                    }
                    if (num_flops) *num_flops += 10*C->num_rows*C->num_columns*A->num_columns;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if ((Atrans == mtx_notrans || Atrans == mtx_conjtrans) && Btrans == mtx_conjtrans) {
                if (a->precision == mtx_single) {
                    const float (* Adata)[2] = a->data.complex_single;
                    const float (* Bdata)[2] = b->data.complex_single;
                    float (* Cdata)[2] = c->data.complex_single;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < i; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,k,i)][0]*Bdata[j*B->num_columns+k][0]-Adata[touptri(A->num_columns,k,i)][1]*Bdata[j*B->num_columns+k][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(-Adata[touptri(A->num_columns,k,i)][0]*Bdata[j*B->num_columns+k][1]-Adata[touptri(A->num_columns,k,i)][1]*Bdata[j*B->num_columns+k][0]);
                            }
                            for (int k = i; k < A->num_columns; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,i,k)][0]*Bdata[j*B->num_columns+k][0]+Adata[touptri(A->num_columns,i,k)][1]*Bdata[j*B->num_columns+k][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(-Adata[touptri(A->num_columns,i,k)][0]*Bdata[j*B->num_columns+k][1]+Adata[touptri(A->num_columns,i,k)][1]*Bdata[j*B->num_columns+k][0]);
                            }
                        }
                    }
                    if (num_flops) *num_flops += 10*C->num_rows*C->num_columns*A->num_columns;
                } else if (a->precision == mtx_double) {
                    const double (* Adata)[2] = a->data.complex_double;
                    const double (* Bdata)[2] = b->data.complex_double;
                    double (* Cdata)[2] = c->data.complex_double;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < i; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,k,i)][0]*Bdata[j*B->num_columns+k][0]-Adata[touptri(A->num_columns,k,i)][1]*Bdata[j*B->num_columns+k][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(-Adata[touptri(A->num_columns,k,i)][0]*Bdata[j*B->num_columns+k][1]-Adata[touptri(A->num_columns,k,i)][1]*Bdata[j*B->num_columns+k][0]);
                            }
                            for (int k = i; k < A->num_columns; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,i,k)][0]*Bdata[j*B->num_columns+k][0]+Adata[touptri(A->num_columns,i,k)][1]*Bdata[j*B->num_columns+k][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(-Adata[touptri(A->num_columns,i,k)][0]*Bdata[j*B->num_columns+k][1]+Adata[touptri(A->num_columns,i,k)][1]*Bdata[j*B->num_columns+k][0]);
                            }
                        }
                    }
                    if (num_flops) *num_flops += 10*C->num_rows*C->num_columns*A->num_columns;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (Atrans == mtx_trans && Btrans == mtx_notrans) {
                if (a->precision == mtx_single) {
                    const float (* Adata)[2] = a->data.complex_single;
                    const float (* Bdata)[2] = b->data.complex_single;
                    float (* Cdata)[2] = c->data.complex_single;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < i; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,k,i)][0]*Bdata[k*B->num_columns+j][0]-Adata[touptri(A->num_columns,k,i)][1]*Bdata[k*B->num_columns+j][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(Adata[touptri(A->num_columns,k,i)][0]*Bdata[k*B->num_columns+j][1]+Adata[touptri(A->num_columns,k,i)][1]*Bdata[k*B->num_columns+j][0]);
                            }
                            for (int k = i; k < A->num_columns; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,i,k)][0]*Bdata[k*B->num_columns+j][0]+Adata[touptri(A->num_columns,i,k)][1]*Bdata[k*B->num_columns+j][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(Adata[touptri(A->num_columns,i,k)][0]*Bdata[k*B->num_columns+j][1]-Adata[touptri(A->num_columns,i,k)][1]*Bdata[k*B->num_columns+j][0]);
                            }
                        }
                    }
                    if (num_flops) *num_flops += 10*C->num_rows*C->num_columns*A->num_columns;
                } else if (a->precision == mtx_double) {
                    const double (* Adata)[2] = a->data.complex_double;
                    const double (* Bdata)[2] = b->data.complex_double;
                    double (* Cdata)[2] = c->data.complex_double;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < i; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,k,i)][0]*Bdata[k*B->num_columns+j][0]-Adata[touptri(A->num_columns,k,i)][1]*Bdata[k*B->num_columns+j][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(Adata[touptri(A->num_columns,k,i)][0]*Bdata[k*B->num_columns+j][1]+Adata[touptri(A->num_columns,k,i)][1]*Bdata[k*B->num_columns+j][0]);
                            }
                            for (int k = i; k < A->num_columns; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,i,k)][0]*Bdata[k*B->num_columns+j][0]+Adata[touptri(A->num_columns,i,k)][1]*Bdata[k*B->num_columns+j][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(Adata[touptri(A->num_columns,i,k)][0]*Bdata[k*B->num_columns+j][1]-Adata[touptri(A->num_columns,i,k)][1]*Bdata[k*B->num_columns+j][0]);
                            }
                        }
                    }
                    if (num_flops) *num_flops += 10*C->num_rows*C->num_columns*A->num_columns;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (Atrans == mtx_trans && Btrans == mtx_trans) {
                if (a->precision == mtx_single) {
                    const float (* Adata)[2] = a->data.complex_single;
                    const float (* Bdata)[2] = b->data.complex_single;
                    float (* Cdata)[2] = c->data.complex_single;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < i; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,k,i)][0]*Bdata[j*B->num_columns+k][0]-Adata[touptri(A->num_columns,k,i)][1]*Bdata[j*B->num_columns+k][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(Adata[touptri(A->num_columns,k,i)][0]*Bdata[j*B->num_columns+k][1]+Adata[touptri(A->num_columns,k,i)][1]*Bdata[j*B->num_columns+k][0]);
                            }
                            for (int k = i; k < A->num_columns; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,i,k)][0]*Bdata[j*B->num_columns+k][0]+Adata[touptri(A->num_columns,i,k)][1]*Bdata[j*B->num_columns+k][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(Adata[touptri(A->num_columns,i,k)][0]*Bdata[j*B->num_columns+k][1]-Adata[touptri(A->num_columns,i,k)][1]*Bdata[j*B->num_columns+k][0]);
                            }
                        }
                    }
                    if (num_flops) *num_flops += 10*C->num_rows*C->num_columns*A->num_columns;
                } else if (a->precision == mtx_double) {
                    const double (* Adata)[2] = a->data.complex_double;
                    const double (* Bdata)[2] = b->data.complex_double;
                    double (* Cdata)[2] = c->data.complex_double;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < i; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,k,i)][0]*Bdata[j*B->num_columns+k][0]-Adata[touptri(A->num_columns,k,i)][1]*Bdata[j*B->num_columns+k][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(Adata[touptri(A->num_columns,k,i)][0]*Bdata[j*B->num_columns+k][1]+Adata[touptri(A->num_columns,k,i)][1]*Bdata[j*B->num_columns+k][0]);
                            }
                            for (int k = i; k < A->num_columns; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,i,k)][0]*Bdata[j*B->num_columns+k][0]+Adata[touptri(A->num_columns,i,k)][1]*Bdata[j*B->num_columns+k][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(Adata[touptri(A->num_columns,i,k)][0]*Bdata[j*B->num_columns+k][1]-Adata[touptri(A->num_columns,i,k)][1]*Bdata[j*B->num_columns+k][0]);
                            }
                        }
                    }
                    if (num_flops) *num_flops += 10*C->num_rows*C->num_columns*A->num_columns;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (Atrans == mtx_trans && Btrans == mtx_conjtrans) {
                if (a->precision == mtx_single) {
                    const float (* Adata)[2] = a->data.complex_single;
                    const float (* Bdata)[2] = b->data.complex_single;
                    float (* Cdata)[2] = c->data.complex_single;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < i; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,k,i)][0]*Bdata[j*B->num_columns+k][0]+Adata[touptri(A->num_columns,k,i)][1]*Bdata[j*B->num_columns+k][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(-Adata[touptri(A->num_columns,k,i)][0]*Bdata[j*B->num_columns+k][1]+Adata[touptri(A->num_columns,k,i)][1]*Bdata[j*B->num_columns+k][0]);
                            }
                            for (int k = i; k < A->num_columns; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,i,k)][0]*Bdata[j*B->num_columns+k][0]-Adata[touptri(A->num_columns,i,k)][1]*Bdata[j*B->num_columns+k][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(-Adata[touptri(A->num_columns,i,k)][0]*Bdata[j*B->num_columns+k][1]-Adata[touptri(A->num_columns,i,k)][1]*Bdata[j*B->num_columns+k][0]);
                            }
                        }
                    }
                    if (num_flops) *num_flops += 10*C->num_rows*C->num_columns*A->num_columns;
                } else if (a->precision == mtx_double) {
                    const double (* Adata)[2] = a->data.complex_double;
                    const double (* Bdata)[2] = b->data.complex_double;
                    double (* Cdata)[2] = c->data.complex_double;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < i; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,k,i)][0]*Bdata[j*B->num_columns+k][0]+Adata[touptri(A->num_columns,k,i)][1]*Bdata[j*B->num_columns+k][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(-Adata[touptri(A->num_columns,k,i)][0]*Bdata[j*B->num_columns+k][1]+Adata[touptri(A->num_columns,k,i)][1]*Bdata[j*B->num_columns+k][0]);
                            }
                            for (int k = i; k < A->num_columns; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,i,k)][0]*Bdata[j*B->num_columns+k][0]-Adata[touptri(A->num_columns,i,k)][1]*Bdata[j*B->num_columns+k][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(-Adata[touptri(A->num_columns,i,k)][0]*Bdata[j*B->num_columns+k][1]-Adata[touptri(A->num_columns,i,k)][1]*Bdata[j*B->num_columns+k][0]);
                            }
                        }
                    }
                    if (num_flops) *num_flops += 10*C->num_rows*C->num_columns*A->num_columns;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else { return MTX_ERR_INVALID_TRANSPOSITION; }
        } else { return MTX_ERR_INVALID_FIELD; }
    } else { return MTX_ERR_INVALID_SYMMETRY; }
    return MTX_SUCCESS;
#else
    return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
}

/**
 * ‘mtxblasdense_dgemm()’ multiplies a matrix ‘A’ (or its transpose ‘A'’)
 * by a real scalar ‘alpha’ (‘α’) and a matrix ‘B’ (or its transpose
 * ‘B'’), before adding the result to another matrix ‘C’ multiplied by
 * a real scalar ‘beta’ (‘β’). That is,
 *
 * ‘C = α*op(A)*op(B) + β*C’, where ‘op(X)=X’ or ‘op(X)=X'’.
 *
 * The scalars ‘alpha’ and ‘beta’ are given as double precision
 * floating point numbers.
 */
int mtxblasdense_dgemm(
    enum mtxtransposition Atrans,
    enum mtxtransposition Btrans,
    double alpha,
    const struct mtxblasdense * A,
    const struct mtxblasdense * B,
    double beta,
    struct mtxblasdense * C,
    int64_t * num_flops)
{
#ifdef LIBMTX_HAVE_BLAS
    const struct mtxbasevector * a = &A->a.base;
    const struct mtxbasevector * b = &B->a.base;
    struct mtxbasevector * c = &C->a.base;
    if (b->field != a->field || c->field != a->field)
        return MTX_ERR_INCOMPATIBLE_FIELD;
    if (b->precision != a->precision || c->precision != a->precision)
        return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (Atrans == mtx_notrans && Btrans == mtx_notrans &&
        (A->num_rows != C->num_rows || A->num_columns != B->num_rows || B->num_columns != C->num_columns))
        return MTX_ERR_INCOMPATIBLE_SIZE;
    if (Atrans == mtx_notrans && (Btrans == mtx_trans || Btrans == mtx_conjtrans) &&
        (A->num_rows != C->num_rows || A->num_columns != B->num_columns || B->num_rows != C->num_columns))
        return MTX_ERR_INCOMPATIBLE_SIZE;
    if ((Atrans == mtx_trans || Atrans == mtx_conjtrans) && Btrans == mtx_notrans &&
        (A->num_columns != C->num_rows || A->num_rows != B->num_rows || B->num_columns != C->num_columns))
        return MTX_ERR_INCOMPATIBLE_SIZE;
    if ((Atrans == mtx_trans || Atrans == mtx_conjtrans) && (Btrans == mtx_trans || Btrans == mtx_conjtrans) &&
        (A->num_columns != C->num_rows || A->num_rows != B->num_columns || B->num_rows != C->num_columns))
        return MTX_ERR_INCOMPATIBLE_SIZE;
    if (B->symmetry != mtx_unsymmetric || C->symmetry != mtx_unsymmetric)
        return MTX_ERR_INCOMPATIBLE_SYMMETRY;

    if (beta != 1) {
        int err = mtxblasdense_sscal(beta, C, num_flops);
        if (err) return err;
    }

    if (A->symmetry == mtx_unsymmetric) {
        if (a->field == mtx_field_real) {
            if (Atrans == mtx_notrans && Btrans == mtx_notrans) {
                if (a->precision == mtx_single) {
                    const float * Adata = a->data.real_single;
                    const float * Bdata = b->data.real_single;
                    float * Cdata = c->data.real_single;
                    cblas_sgemm(
                        CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        C->num_rows, C->num_columns, A->num_columns,
                        alpha, Adata, A->num_columns, Bdata, B->num_columns,
                        beta, Cdata, C->num_columns);
                    if (mtxblaserror()) return MTX_ERR_BLAS;
                    if (num_flops) *num_flops += cblas_sgemm_num_flops(
                        C->num_rows, C->num_columns, A->num_columns, alpha, beta);
                } else if (a->precision == mtx_double) {
                    const double * Adata = a->data.real_double;
                    const double * Bdata = b->data.real_double;
                    double * Cdata = c->data.real_double;
                    cblas_dgemm(
                        CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        C->num_rows, C->num_columns, A->num_columns,
                        alpha, Adata, A->num_columns, Bdata, B->num_columns,
                        beta, Cdata, C->num_columns);
                    if (mtxblaserror()) return MTX_ERR_BLAS;
                    if (num_flops) *num_flops += cblas_sgemm_num_flops(
                        C->num_rows, C->num_columns, A->num_columns, alpha, beta);
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (Atrans == mtx_notrans && (Btrans == mtx_trans || Btrans == mtx_conjtrans)) {
                if (a->precision == mtx_single) {
                    const float * Adata = a->data.real_single;
                    const float * Bdata = b->data.real_single;
                    float * Cdata = c->data.real_single;
                    cblas_sgemm(
                        CblasRowMajor, CblasNoTrans, CblasTrans,
                        C->num_rows, C->num_columns, A->num_columns,
                        alpha, Adata, A->num_columns, Bdata, B->num_columns,
                        beta, Cdata, C->num_columns);
                    if (mtxblaserror()) return MTX_ERR_BLAS;
                    if (num_flops) *num_flops += cblas_sgemm_num_flops(
                        C->num_rows, C->num_columns, A->num_columns, alpha, beta);
                } else if (a->precision == mtx_double) {
                    const double * Adata = a->data.real_double;
                    const double * Bdata = b->data.real_double;
                    double * Cdata = c->data.real_double;
                    cblas_dgemm(
                        CblasRowMajor, CblasNoTrans, CblasTrans,
                        C->num_rows, C->num_columns, A->num_columns,
                        alpha, Adata, A->num_columns, Bdata, B->num_columns,
                        beta, Cdata, C->num_columns);
                    if (mtxblaserror()) return MTX_ERR_BLAS;
                    if (num_flops) *num_flops += cblas_sgemm_num_flops(
                        C->num_rows, C->num_columns, A->num_columns, alpha, beta);
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if ((Atrans == mtx_trans || Atrans == mtx_conjtrans) && Btrans == mtx_notrans) {
                if (a->precision == mtx_single) {
                    const float * Adata = a->data.real_single;
                    const float * Bdata = b->data.real_single;
                    float * Cdata = c->data.real_single;
                    cblas_sgemm(
                        CblasRowMajor, CblasTrans, CblasNoTrans,
                        C->num_rows, C->num_columns, A->num_columns,
                        alpha, Adata, A->num_columns, Bdata, B->num_columns,
                        beta, Cdata, C->num_columns);
                    if (mtxblaserror()) return MTX_ERR_BLAS;
                    if (num_flops) *num_flops += cblas_sgemm_num_flops(
                        C->num_rows, C->num_columns, A->num_columns, alpha, beta);
                } else if (a->precision == mtx_double) {
                    const double * Adata = a->data.real_double;
                    const double * Bdata = b->data.real_double;
                    double * Cdata = c->data.real_double;
                    cblas_dgemm(
                        CblasRowMajor, CblasTrans, CblasNoTrans,
                        C->num_rows, C->num_columns, A->num_columns,
                        alpha, Adata, A->num_columns, Bdata, B->num_columns,
                        beta, Cdata, C->num_columns);
                    if (mtxblaserror()) return MTX_ERR_BLAS;
                    if (num_flops) *num_flops += cblas_sgemm_num_flops(
                        C->num_rows, C->num_columns, A->num_columns, alpha, beta);
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if ((Atrans == mtx_trans || Atrans == mtx_conjtrans) && (Btrans == mtx_trans || Btrans == mtx_conjtrans)) {
                if (a->precision == mtx_single) {
                    const float * Adata = a->data.real_single;
                    const float * Bdata = b->data.real_single;
                    float * Cdata = c->data.real_single;
                    cblas_sgemm(
                        CblasRowMajor, CblasTrans, CblasTrans,
                        C->num_rows, C->num_columns, A->num_columns,
                        alpha, Adata, A->num_columns, Bdata, B->num_columns,
                        beta, Cdata, C->num_columns);
                    if (mtxblaserror()) return MTX_ERR_BLAS;
                    if (num_flops) *num_flops += cblas_sgemm_num_flops(
                        C->num_rows, C->num_columns, A->num_columns, alpha, beta);
                } else if (a->precision == mtx_double) {
                    const double * Adata = a->data.real_double;
                    const double * Bdata = b->data.real_double;
                    double * Cdata = c->data.real_double;
                    cblas_dgemm(
                        CblasRowMajor, CblasTrans, CblasTrans,
                        C->num_rows, C->num_columns, A->num_columns,
                        alpha, Adata, A->num_columns, Bdata, B->num_columns,
                        beta, Cdata, C->num_columns);
                    if (mtxblaserror()) return MTX_ERR_BLAS;
                    if (num_flops) *num_flops += cblas_sgemm_num_flops(
                        C->num_rows, C->num_columns, A->num_columns, alpha, beta);
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else { return MTX_ERR_INVALID_TRANSPOSITION; }
        } else if (a->field == mtx_field_complex) {
            if (Atrans == mtx_notrans && Btrans == mtx_notrans) {
                if (a->precision == mtx_single) {
                    const float (* Adata)[2] = a->data.complex_single;
                    const float (* Bdata)[2] = b->data.complex_single;
                    float (* Cdata)[2] = c->data.complex_single;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < A->num_columns; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[i*A->num_columns+k][0]*Bdata[k*B->num_columns+j][0]-Adata[i*A->num_columns+k][1]*Bdata[k*B->num_columns+j][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(Adata[i*A->num_columns+k][0]*Bdata[k*B->num_columns+j][1]+Adata[i*A->num_columns+k][1]*Bdata[k*B->num_columns+j][0]);
                            }
                        }
                    }
                    if (num_flops) *num_flops += 10*C->num_rows*C->num_columns*A->num_columns;
                } else if (a->precision == mtx_double) {
                    const double (* Adata)[2] = a->data.complex_double;
                    const double (* Bdata)[2] = b->data.complex_double;
                    double (* Cdata)[2] = c->data.complex_double;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < A->num_columns; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[i*A->num_columns+k][0]*Bdata[k*B->num_columns+j][0]-Adata[i*A->num_columns+k][1]*Bdata[k*B->num_columns+j][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(Adata[i*A->num_columns+k][0]*Bdata[k*B->num_columns+j][1]+Adata[i*A->num_columns+k][1]*Bdata[k*B->num_columns+j][0]);
                            }
                        }
                    }
                    if (num_flops) *num_flops += 10*C->num_rows*C->num_columns*A->num_columns;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (Atrans == mtx_notrans && Btrans == mtx_trans) {
                if (a->precision == mtx_single) {
                    const float (* Adata)[2] = a->data.complex_single;
                    const float (* Bdata)[2] = b->data.complex_single;
                    float (* Cdata)[2] = c->data.complex_single;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < A->num_columns; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[i*A->num_columns+k][0]*Bdata[j*B->num_columns+k][0]-Adata[i*A->num_columns+k][1]*Bdata[j*B->num_columns+k][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(Adata[i*A->num_columns+k][0]*Bdata[j*B->num_columns+k][1]+Adata[i*A->num_columns+k][1]*Bdata[j*B->num_columns+k][0]);
                            }
                        }
                    }
                    if (num_flops) *num_flops += 10*C->num_rows*C->num_columns*A->num_columns;
                } else if (a->precision == mtx_double) {
                    const double (* Adata)[2] = a->data.complex_double;
                    const double (* Bdata)[2] = b->data.complex_double;
                    double (* Cdata)[2] = c->data.complex_double;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < A->num_columns; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[i*A->num_columns+k][0]*Bdata[j*B->num_columns+k][0]-Adata[i*A->num_columns+k][1]*Bdata[j*B->num_columns+k][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(Adata[i*A->num_columns+k][0]*Bdata[j*B->num_columns+k][1]+Adata[i*A->num_columns+k][1]*Bdata[j*B->num_columns+k][0]);
                            }
                        }
                    }
                    if (num_flops) *num_flops += 10*C->num_rows*C->num_columns*A->num_columns;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (Atrans == mtx_notrans && Btrans == mtx_conjtrans) {
                if (a->precision == mtx_single) {
                    const float (* Adata)[2] = a->data.complex_single;
                        const float (* Bdata)[2] = b->data.complex_single;
                    float (* Cdata)[2] = c->data.complex_single;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < A->num_columns; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[i*A->num_columns+k][0]*Bdata[j*B->num_columns+k][0]+Adata[i*A->num_columns+k][1]*Bdata[j*B->num_columns+k][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(-Adata[i*A->num_columns+k][0]*Bdata[j*B->num_columns+k][1]+Adata[i*A->num_columns+k][1]*Bdata[j*B->num_columns+k][0]);
                            }
                        }
                    }
                    if (num_flops) *num_flops += 10*C->num_rows*C->num_columns*A->num_columns;
                } else if (a->precision == mtx_double) {
                    const double (* Adata)[2] = a->data.complex_double;
                    const double (* Bdata)[2] = b->data.complex_double;
                    double (* Cdata)[2] = c->data.complex_double;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < A->num_columns; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[i*A->num_columns+k][0]*Bdata[j*B->num_columns+k][0]+Adata[i*A->num_columns+k][1]*Bdata[j*B->num_columns+k][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(-Adata[i*A->num_columns+k][0]*Bdata[j*B->num_columns+k][1]+Adata[i*A->num_columns+k][1]*Bdata[j*B->num_columns+k][0]);
                            }
                        }
                    }
                    if (num_flops) *num_flops += 10*C->num_rows*C->num_columns*A->num_columns;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (Atrans == mtx_trans && Btrans == mtx_notrans) {
                if (a->precision == mtx_single) {
                    const float (* Adata)[2] = a->data.complex_single;
                    const float (* Bdata)[2] = b->data.complex_single;
                    float (* Cdata)[2] = c->data.complex_single;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < A->num_rows; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[k*A->num_columns+i][0]*Bdata[k*B->num_columns+j][0]-Adata[k*A->num_columns+i][1]*Bdata[k*B->num_columns+j][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(Adata[k*A->num_columns+i][0]*Bdata[k*B->num_columns+j][1]+Adata[k*A->num_columns+i][1]*Bdata[k*B->num_columns+j][0]);
                            }
                        }
                    }
                    if (num_flops) *num_flops += 10*C->num_rows*C->num_columns*A->num_rows;
                } else if (a->precision == mtx_double) {
                    const double (* Adata)[2] = a->data.complex_double;
                    const double (* Bdata)[2] = b->data.complex_double;
                    double (* Cdata)[2] = c->data.complex_double;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < A->num_rows; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[k*A->num_columns+i][0]*Bdata[k*B->num_columns+j][0]-Adata[k*A->num_columns+i][1]*Bdata[k*B->num_columns+j][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(Adata[k*A->num_columns+i][0]*Bdata[k*B->num_columns+j][1]+Adata[k*A->num_columns+i][1]*Bdata[k*B->num_columns+j][0]);
                            }
                        }
                    }
                    if (num_flops) *num_flops += 10*C->num_rows*C->num_columns*A->num_rows;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (Atrans == mtx_trans && Btrans == mtx_trans) {
                if (a->precision == mtx_single) {
                    const float (* Adata)[2] = a->data.complex_single;
                    const float (* Bdata)[2] = b->data.complex_single;
                    float (* Cdata)[2] = c->data.complex_single;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < A->num_rows; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[k*A->num_columns+i][0]*Bdata[j*B->num_columns+k][0]-Adata[k*A->num_columns+i][1]*Bdata[j*B->num_columns+k][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(Adata[k*A->num_columns+i][0]*Bdata[j*B->num_columns+k][1]+Adata[k*A->num_columns+i][1]*Bdata[j*B->num_columns+k][0]);
                            }
                        }
                    }
                    if (num_flops) *num_flops += 10*C->num_rows*C->num_columns*A->num_rows;
                } else if (a->precision == mtx_double) {
                    const double (* Adata)[2] = a->data.complex_double;
                    const double (* Bdata)[2] = b->data.complex_double;
                    double (* Cdata)[2] = c->data.complex_double;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < A->num_rows; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[k*A->num_columns+i][0]*Bdata[j*B->num_columns+k][0]-Adata[k*A->num_columns+i][1]*Bdata[j*B->num_columns+k][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(Adata[k*A->num_columns+i][0]*Bdata[j*B->num_columns+k][1]+Adata[k*A->num_columns+i][1]*Bdata[j*B->num_columns+k][0]);
                            }
                        }
                    }
                    if (num_flops) *num_flops += 10*C->num_rows*C->num_columns*A->num_rows;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (Atrans == mtx_trans && Btrans == mtx_conjtrans) {
                if (a->precision == mtx_single) {
                    const float (* Adata)[2] = a->data.complex_single;
                    const float (* Bdata)[2] = b->data.complex_single;
                    float (* Cdata)[2] = c->data.complex_single;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < A->num_rows; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[k*A->num_columns+i][0]*Bdata[j*B->num_columns+k][0]+Adata[k*A->num_columns+i][1]*Bdata[j*B->num_columns+k][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(-Adata[k*A->num_columns+i][0]*Bdata[j*B->num_columns+k][1]+Adata[k*A->num_columns+i][1]*Bdata[j*B->num_columns+k][0]);
                            }
                        }
                    }
                    if (num_flops) *num_flops += 10*C->num_rows*C->num_columns*A->num_rows;
                } else if (a->precision == mtx_double) {
                    const double (* Adata)[2] = a->data.complex_double;
                    const double (* Bdata)[2] = b->data.complex_double;
                    double (* Cdata)[2] = c->data.complex_double;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < A->num_rows; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[k*A->num_columns+i][0]*Bdata[j*B->num_columns+k][0]+Adata[k*A->num_columns+i][1]*Bdata[j*B->num_columns+k][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(-Adata[k*A->num_columns+i][0]*Bdata[j*B->num_columns+k][1]+Adata[k*A->num_columns+i][1]*Bdata[j*B->num_columns+k][0]);
                            }
                        }
                    }
                    if (num_flops) *num_flops += 10*C->num_rows*C->num_columns*A->num_rows;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (Atrans == mtx_conjtrans && Btrans == mtx_notrans) {
                if (a->precision == mtx_single) {
                    const float (* Adata)[2] = a->data.complex_single;
                    const float (* Bdata)[2] = b->data.complex_single;
                    float (* Cdata)[2] = c->data.complex_single;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < A->num_rows; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[k*A->num_columns+i][0]*Bdata[k*B->num_columns+j][0]+Adata[k*A->num_columns+i][1]*Bdata[k*B->num_columns+j][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(Adata[k*A->num_columns+i][0]*Bdata[k*B->num_columns+j][1]-Adata[k*A->num_columns+i][1]*Bdata[k*B->num_columns+j][0]);
                            }
                        }
                    }
                    if (num_flops) *num_flops += 10*C->num_rows*C->num_columns*A->num_rows;
                } else if (a->precision == mtx_double) {
                    const double (* Adata)[2] = a->data.complex_double;
                    const double (* Bdata)[2] = b->data.complex_double;
                    double (* Cdata)[2] = c->data.complex_double;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < A->num_rows; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[k*A->num_columns+i][0]*Bdata[k*B->num_columns+j][0]+Adata[k*A->num_columns+i][1]*Bdata[k*B->num_columns+j][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(Adata[k*A->num_columns+i][0]*Bdata[k*B->num_columns+j][1]-Adata[k*A->num_columns+i][1]*Bdata[k*B->num_columns+j][0]);
                            }
                        }
                    }
                    if (num_flops) *num_flops += 10*C->num_rows*C->num_columns*A->num_rows;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else { return MTX_ERR_INVALID_TRANSPOSITION; }
        } else if (a->field == mtx_field_integer) {
            if (Atrans == mtx_notrans && Btrans == mtx_notrans) {
                if (a->precision == mtx_single) {
                    const int32_t * Adata = a->data.integer_single;
                    const int32_t * Bdata = b->data.integer_single;
                    int32_t * Cdata = c->data.integer_single;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < A->num_columns; k++)
                                Cdata[i*C->num_columns+j] += alpha*Adata[i*A->num_columns+k]*Bdata[k*B->num_columns+j];
                        }
                    }
                    if (num_flops) *num_flops += 3*C->num_rows*C->num_columns*A->num_columns;
                } else if (a->precision == mtx_double) {
                    const int64_t * Adata = a->data.integer_double;
                    const int64_t * Bdata = b->data.integer_double;
                    int64_t * Cdata = c->data.integer_double;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < A->num_columns; k++)
                                Cdata[i*C->num_columns+j] += alpha*Adata[i*A->num_columns+k]*Bdata[k*B->num_columns+j];
                        }
                    }
                    if (num_flops) *num_flops += 3*C->num_rows*C->num_columns*A->num_columns;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (Atrans == mtx_notrans && (Btrans == mtx_trans || Btrans == mtx_conjtrans)) {
                if (a->precision == mtx_single) {
                    const int32_t * Adata = a->data.integer_single;
                    const int32_t * Bdata = b->data.integer_single;
                    int32_t * Cdata = c->data.integer_single;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < A->num_columns; k++)
                                Cdata[i*C->num_columns+j] += alpha*Adata[i*A->num_columns+k]*Bdata[j*B->num_columns+k];
                        }
                    }
                    if (num_flops) *num_flops += 3*C->num_rows*C->num_columns*A->num_columns;
                } else if (a->precision == mtx_double) {
                    const int64_t * Adata = a->data.integer_double;
                    const int64_t * Bdata = b->data.integer_double;
                    int64_t * Cdata = c->data.integer_double;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < A->num_columns; k++)
                                Cdata[i*C->num_columns+j] += alpha*Adata[i*A->num_columns+k]*Bdata[j*B->num_columns+k];
                        }
                    }
                    if (num_flops) *num_flops += 3*C->num_rows*C->num_columns*A->num_columns;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if ((Atrans == mtx_trans || Atrans == mtx_conjtrans) && Btrans == mtx_notrans) {
                if (a->precision == mtx_single) {
                    const int32_t * Adata = a->data.integer_single;
                    const int32_t * Bdata = b->data.integer_single;
                    int32_t * Cdata = c->data.integer_single;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < A->num_rows; k++)
                                Cdata[i*C->num_columns+j] += alpha*Adata[k*A->num_columns+i]*Bdata[k*B->num_columns+j];
                        }
                    }
                    if (num_flops) *num_flops += 3*C->num_rows*C->num_columns*A->num_rows;
                } else if (a->precision == mtx_double) {
                    const int64_t * Adata = a->data.integer_double;
                    const int64_t * Bdata = b->data.integer_double;
                    int64_t * Cdata = c->data.integer_double;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < A->num_rows; k++)
                                Cdata[i*C->num_columns+j] += alpha*Adata[k*A->num_columns+i]*Bdata[k*B->num_columns+j];
                        }
                    }
                    if (num_flops) *num_flops += 3*C->num_rows*C->num_columns*A->num_rows;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if ((Atrans == mtx_trans || Atrans == mtx_conjtrans) && (Btrans == mtx_trans || Btrans == mtx_conjtrans)) {
                if (a->precision == mtx_single) {
                    const int32_t * Adata = a->data.integer_single;
                    const int32_t * Bdata = b->data.integer_single;
                    int32_t * Cdata = c->data.integer_single;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < A->num_rows; k++)
                                Cdata[i*C->num_columns+j] += alpha*Adata[k*A->num_columns+i]*Bdata[j*B->num_columns+k];
                        }
                    }
                    if (num_flops) *num_flops += 3*C->num_rows*C->num_columns*A->num_rows;
                } else if (a->precision == mtx_double) {
                    const int64_t * Adata = a->data.integer_double;
                    const int64_t * Bdata = b->data.integer_double;
                    int64_t * Cdata = c->data.integer_double;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < A->num_rows; k++)
                                Cdata[i*C->num_columns+j] += alpha*Adata[k*A->num_columns+i]*Bdata[j*B->num_columns+k];
                        }
                    }
                    if (num_flops) *num_flops += 3*C->num_rows*C->num_columns*A->num_rows;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else { return MTX_ERR_INVALID_TRANSPOSITION; }
        } else { return MTX_ERR_INVALID_FIELD; }
    } else if (A->num_rows == A->num_columns && A->symmetry == mtx_symmetric) {
        if (a->field == mtx_field_real) {
            if (Btrans == mtx_notrans) {
                if (a->precision == mtx_single) {
                    const float * Adata = a->data.real_single;
                    const float * Bdata = b->data.real_single;
                    float * Cdata = c->data.real_single;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < i; k++)
                                Cdata[i*C->num_columns+j] += alpha*Adata[touptri(A->num_columns,k,i)]*Bdata[k*B->num_columns+j];
                            for (int k = i; k < A->num_columns; k++)
                                Cdata[i*C->num_columns+j] += alpha*Adata[touptri(A->num_columns,i,k)]*Bdata[k*B->num_columns+j];
                        }
                    }
                    if (num_flops) *num_flops += 3*C->num_rows*C->num_columns*A->num_columns;
                } else if (a->precision == mtx_double) {
                    const double * Adata = a->data.real_double;
                    const double * Bdata = b->data.real_double;
                    double * Cdata = c->data.real_double;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < i; k++)
                                Cdata[i*C->num_columns+j] += alpha*Adata[touptri(A->num_columns,k,i)]*Bdata[k*B->num_columns+j];
                            for (int k = i; k < A->num_columns; k++)
                                Cdata[i*C->num_columns+j] += alpha*Adata[touptri(A->num_columns,i,k)]*Bdata[k*B->num_columns+j];
                        }
                    }
                    if (num_flops) *num_flops += 3*C->num_rows*C->num_columns*A->num_columns;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (Btrans == mtx_trans || Btrans == mtx_conjtrans) {
                if (a->precision == mtx_single) {
                    const float * Adata = a->data.real_single;
                    const float * Bdata = b->data.real_single;
                    float * Cdata = c->data.real_single;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < i; k++)
                                Cdata[i*C->num_columns+j] += alpha*Adata[touptri(A->num_columns,k,i)]*Bdata[j*B->num_columns+k];
                            for (int k = i; k < A->num_columns; k++)
                                Cdata[i*C->num_columns+j] += alpha*Adata[touptri(A->num_columns,i,k)]*Bdata[j*B->num_columns+k];
                        }
                    }
                    if (num_flops) *num_flops += 3*C->num_rows*C->num_columns*A->num_columns;
                } else if (a->precision == mtx_double) {
                    const double * Adata = a->data.real_double;
                    const double * Bdata = b->data.real_double;
                    double * Cdata = c->data.real_double;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < i; k++)
                                Cdata[i*C->num_columns+j] += alpha*Adata[touptri(A->num_columns,k,i)]*Bdata[j*B->num_columns+k];
                            for (int k = i; k < A->num_columns; k++)
                                Cdata[i*C->num_columns+j] += alpha*Adata[touptri(A->num_columns,i,k)]*Bdata[j*B->num_columns+k];
                        }
                    }
                    if (num_flops) *num_flops += 3*C->num_rows*C->num_columns*A->num_columns;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else { return MTX_ERR_INVALID_TRANSPOSITION; }
        } else if (a->field == mtx_field_complex) {
            if ((Atrans == mtx_notrans || Atrans == mtx_trans) && Btrans == mtx_notrans) {
                if (a->precision == mtx_single) {
                    const float (* Adata)[2] = a->data.complex_single;
                    const float (* Bdata)[2] = b->data.complex_single;
                    float (* Cdata)[2] = c->data.complex_single;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < i; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,k,i)][0]*Bdata[k*B->num_columns+j][0]-Adata[touptri(A->num_columns,k,i)][1]*Bdata[k*B->num_columns+j][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(Adata[touptri(A->num_columns,k,i)][0]*Bdata[k*B->num_columns+j][1]+Adata[touptri(A->num_columns,k,i)][1]*Bdata[k*B->num_columns+j][0]);
                            }
                            for (int k = i; k < A->num_columns; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,i,k)][0]*Bdata[k*B->num_columns+j][0]-Adata[touptri(A->num_columns,i,k)][1]*Bdata[k*B->num_columns+j][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(Adata[touptri(A->num_columns,i,k)][0]*Bdata[k*B->num_columns+j][1]+Adata[touptri(A->num_columns,i,k)][1]*Bdata[k*B->num_columns+j][0]);
                            }
                        }
                    }
                    if (num_flops) *num_flops += 10*C->num_rows*C->num_columns*A->num_columns;
                } else if (a->precision == mtx_double) {
                    const double (* Adata)[2] = a->data.complex_double;
                    const double (* Bdata)[2] = b->data.complex_double;
                    double (* Cdata)[2] = c->data.complex_double;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < i; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,k,i)][0]*Bdata[k*B->num_columns+j][0]-Adata[touptri(A->num_columns,k,i)][1]*Bdata[k*B->num_columns+j][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(Adata[touptri(A->num_columns,k,i)][0]*Bdata[k*B->num_columns+j][1]+Adata[touptri(A->num_columns,k,i)][1]*Bdata[k*B->num_columns+j][0]);
                            }
                            for (int k = i; k < A->num_columns; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,i,k)][0]*Bdata[k*B->num_columns+j][0]-Adata[touptri(A->num_columns,i,k)][1]*Bdata[k*B->num_columns+j][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(Adata[touptri(A->num_columns,i,k)][0]*Bdata[k*B->num_columns+j][1]+Adata[touptri(A->num_columns,i,k)][1]*Bdata[k*B->num_columns+j][0]);
                            }
                        }
                    }
                    if (num_flops) *num_flops += 10*C->num_rows*C->num_columns*A->num_columns;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if ((Atrans == mtx_notrans || Atrans == mtx_trans) && Btrans == mtx_trans) {
                if (a->precision == mtx_single) {
                    const float (* Adata)[2] = a->data.complex_single;
                    const float (* Bdata)[2] = b->data.complex_single;
                    float (* Cdata)[2] = c->data.complex_single;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < i; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,k,i)][0]*Bdata[j*B->num_columns+k][0]-Adata[touptri(A->num_columns,k,i)][1]*Bdata[j*B->num_columns+k][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(Adata[touptri(A->num_columns,k,i)][0]*Bdata[j*B->num_columns+k][1]+Adata[touptri(A->num_columns,k,i)][1]*Bdata[j*B->num_columns+k][0]);
                            }
                            for (int k = i; k < A->num_columns; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,i,k)][0]*Bdata[j*B->num_columns+k][0]-Adata[touptri(A->num_columns,i,k)][1]*Bdata[j*B->num_columns+k][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(Adata[touptri(A->num_columns,i,k)][0]*Bdata[j*B->num_columns+k][1]+Adata[touptri(A->num_columns,i,k)][1]*Bdata[j*B->num_columns+k][0]);
                            }
                        }
                    }
                    if (num_flops) *num_flops += 10*C->num_rows*C->num_columns*A->num_columns;
                } else if (a->precision == mtx_double) {
                    const double (* Adata)[2] = a->data.complex_double;
                    const double (* Bdata)[2] = b->data.complex_double;
                    double (* Cdata)[2] = c->data.complex_double;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < i; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,k,i)][0]*Bdata[j*B->num_columns+k][0]-Adata[touptri(A->num_columns,k,i)][1]*Bdata[j*B->num_columns+k][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(Adata[touptri(A->num_columns,k,i)][0]*Bdata[j*B->num_columns+k][1]+Adata[touptri(A->num_columns,k,i)][1]*Bdata[j*B->num_columns+k][0]);
                            }
                            for (int k = i; k < A->num_columns; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,i,k)][0]*Bdata[j*B->num_columns+k][0]-Adata[touptri(A->num_columns,i,k)][1]*Bdata[j*B->num_columns+k][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(Adata[touptri(A->num_columns,i,k)][0]*Bdata[j*B->num_columns+k][1]+Adata[touptri(A->num_columns,i,k)][1]*Bdata[j*B->num_columns+k][0]);
                            }
                        }
                    }
                    if (num_flops) *num_flops += 10*C->num_rows*C->num_columns*A->num_columns;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if ((Atrans == mtx_notrans || Atrans == mtx_trans) && Btrans == mtx_conjtrans) {
                if (a->precision == mtx_single) {
                    const float (* Adata)[2] = a->data.complex_single;
                    const float (* Bdata)[2] = b->data.complex_single;
                    float (* Cdata)[2] = c->data.complex_single;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < i; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,k,i)][0]*Bdata[j*B->num_columns+k][0]+Adata[touptri(A->num_columns,k,i)][1]*Bdata[j*B->num_columns+k][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(-Adata[touptri(A->num_columns,k,i)][0]*Bdata[j*B->num_columns+k][1]+Adata[touptri(A->num_columns,k,i)][1]*Bdata[j*B->num_columns+k][0]);
                            }
                            for (int k = i; k < A->num_columns; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,i,k)][0]*Bdata[j*B->num_columns+k][0]+Adata[touptri(A->num_columns,i,k)][1]*Bdata[j*B->num_columns+k][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(-Adata[touptri(A->num_columns,i,k)][0]*Bdata[j*B->num_columns+k][1]+Adata[touptri(A->num_columns,i,k)][1]*Bdata[j*B->num_columns+k][0]);
                            }
                        }
                    }
                    if (num_flops) *num_flops += 10*C->num_rows*C->num_columns*A->num_columns;
                } else if (a->precision == mtx_double) {
                    const double (* Adata)[2] = a->data.complex_double;
                    const double (* Bdata)[2] = b->data.complex_double;
                    double (* Cdata)[2] = c->data.complex_double;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < i; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,k,i)][0]*Bdata[j*B->num_columns+k][0]+Adata[touptri(A->num_columns,k,i)][1]*Bdata[j*B->num_columns+k][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(-Adata[touptri(A->num_columns,k,i)][0]*Bdata[j*B->num_columns+k][1]+Adata[touptri(A->num_columns,k,i)][1]*Bdata[j*B->num_columns+k][0]);
                            }
                            for (int k = i; k < A->num_columns; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,i,k)][0]*Bdata[j*B->num_columns+k][0]+Adata[touptri(A->num_columns,i,k)][1]*Bdata[j*B->num_columns+k][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(-Adata[touptri(A->num_columns,i,k)][0]*Bdata[j*B->num_columns+k][1]+Adata[touptri(A->num_columns,i,k)][1]*Bdata[j*B->num_columns+k][0]);
                            }
                        }
                    }
                    if (num_flops) *num_flops += 10*C->num_rows*C->num_columns*A->num_columns;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (Atrans == mtx_conjtrans && Btrans == mtx_notrans) {
                if (a->precision == mtx_single) {
                    const float (* Adata)[2] = a->data.complex_single;
                    const float (* Bdata)[2] = b->data.complex_single;
                    float (* Cdata)[2] = c->data.complex_single;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < i; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,k,i)][0]*Bdata[k*B->num_columns+j][0]+Adata[touptri(A->num_columns,k,i)][1]*Bdata[k*B->num_columns+j][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(Adata[touptri(A->num_columns,k,i)][0]*Bdata[k*B->num_columns+j][1]-Adata[touptri(A->num_columns,k,i)][1]*Bdata[k*B->num_columns+j][0]);
                            }
                            for (int k = i; k < A->num_columns; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,i,k)][0]*Bdata[k*B->num_columns+j][0]+Adata[touptri(A->num_columns,i,k)][1]*Bdata[k*B->num_columns+j][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(Adata[touptri(A->num_columns,i,k)][0]*Bdata[k*B->num_columns+j][1]-Adata[touptri(A->num_columns,i,k)][1]*Bdata[k*B->num_columns+j][0]);
                            }
                        }
                    }
                    if (num_flops) *num_flops += 10*C->num_rows*C->num_columns*A->num_columns;
                } else if (a->precision == mtx_double) {
                    const double (* Adata)[2] = a->data.complex_double;
                    const double (* Bdata)[2] = b->data.complex_double;
                    double (* Cdata)[2] = c->data.complex_double;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < i; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,k,i)][0]*Bdata[k*B->num_columns+j][0]+Adata[touptri(A->num_columns,k,i)][1]*Bdata[k*B->num_columns+j][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(Adata[touptri(A->num_columns,k,i)][0]*Bdata[k*B->num_columns+j][1]-Adata[touptri(A->num_columns,k,i)][1]*Bdata[k*B->num_columns+j][0]);
                            }
                            for (int k = i; k < A->num_columns; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,i,k)][0]*Bdata[k*B->num_columns+j][0]+Adata[touptri(A->num_columns,i,k)][1]*Bdata[k*B->num_columns+j][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(Adata[touptri(A->num_columns,i,k)][0]*Bdata[k*B->num_columns+j][1]-Adata[touptri(A->num_columns,i,k)][1]*Bdata[k*B->num_columns+j][0]);
                            }
                        }
                    }
                    if (num_flops) *num_flops += 10*C->num_rows*C->num_columns*A->num_columns;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (Atrans == mtx_conjtrans && Btrans == mtx_trans) {
                if (a->precision == mtx_single) {
                    const float (* Adata)[2] = a->data.complex_single;
                    const float (* Bdata)[2] = b->data.complex_single;
                    float (* Cdata)[2] = c->data.complex_single;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < i; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,k,i)][0]*Bdata[j*B->num_columns+k][0]+Adata[touptri(A->num_columns,k,i)][1]*Bdata[j*B->num_columns+k][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(Adata[touptri(A->num_columns,k,i)][0]*Bdata[j*B->num_columns+k][1]-Adata[touptri(A->num_columns,k,i)][1]*Bdata[j*B->num_columns+k][0]);
                            }
                            for (int k = i; k < A->num_columns; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,i,k)][0]*Bdata[j*B->num_columns+k][0]+Adata[touptri(A->num_columns,i,k)][1]*Bdata[j*B->num_columns+k][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(Adata[touptri(A->num_columns,i,k)][0]*Bdata[j*B->num_columns+k][1]-Adata[touptri(A->num_columns,i,k)][1]*Bdata[j*B->num_columns+k][0]);
                            }
                        }
                    }
                    if (num_flops) *num_flops += 10*C->num_rows*C->num_columns*A->num_columns;
                } else if (a->precision == mtx_double) {
                    const double (* Adata)[2] = a->data.complex_double;
                    const double (* Bdata)[2] = b->data.complex_double;
                    double (* Cdata)[2] = c->data.complex_double;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < i; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,k,i)][0]*Bdata[j*B->num_columns+k][0]+Adata[touptri(A->num_columns,k,i)][1]*Bdata[j*B->num_columns+k][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(Adata[touptri(A->num_columns,k,i)][0]*Bdata[j*B->num_columns+k][1]-Adata[touptri(A->num_columns,k,i)][1]*Bdata[j*B->num_columns+k][0]);
                            }
                            for (int k = i; k < A->num_columns; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,i,k)][0]*Bdata[j*B->num_columns+k][0]+Adata[touptri(A->num_columns,i,k)][1]*Bdata[j*B->num_columns+k][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(Adata[touptri(A->num_columns,i,k)][0]*Bdata[j*B->num_columns+k][1]-Adata[touptri(A->num_columns,i,k)][1]*Bdata[j*B->num_columns+k][0]);
                            }
                        }
                    }
                    if (num_flops) *num_flops += 10*C->num_rows*C->num_columns*A->num_columns;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (Atrans == mtx_conjtrans && Btrans == mtx_conjtrans) {
                if (a->precision == mtx_single) {
                    const float (* Adata)[2] = a->data.complex_single;
                    const float (* Bdata)[2] = b->data.complex_single;
                    float (* Cdata)[2] = c->data.complex_single;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < i; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,k,i)][0]*Bdata[j*B->num_columns+k][0]-Adata[touptri(A->num_columns,k,i)][1]*Bdata[j*B->num_columns+k][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(-Adata[touptri(A->num_columns,k,i)][0]*Bdata[j*B->num_columns+k][1]-Adata[touptri(A->num_columns,k,i)][1]*Bdata[j*B->num_columns+k][0]);
                            }
                            for (int k = i; k < A->num_columns; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,i,k)][0]*Bdata[j*B->num_columns+k][0]-Adata[touptri(A->num_columns,i,k)][1]*Bdata[j*B->num_columns+k][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(-Adata[touptri(A->num_columns,i,k)][0]*Bdata[j*B->num_columns+k][1]-Adata[touptri(A->num_columns,i,k)][1]*Bdata[j*B->num_columns+k][0]);
                            }
                        }
                    }
                    if (num_flops) *num_flops += 10*C->num_rows*C->num_columns*A->num_columns;
                } else if (a->precision == mtx_double) {
                    const double (* Adata)[2] = a->data.complex_double;
                    const double (* Bdata)[2] = b->data.complex_double;
                    double (* Cdata)[2] = c->data.complex_double;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < i; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,k,i)][0]*Bdata[j*B->num_columns+k][0]-Adata[touptri(A->num_columns,k,i)][1]*Bdata[j*B->num_columns+k][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(-Adata[touptri(A->num_columns,k,i)][0]*Bdata[j*B->num_columns+k][1]-Adata[touptri(A->num_columns,k,i)][1]*Bdata[j*B->num_columns+k][0]);
                            }
                            for (int k = i; k < A->num_columns; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,i,k)][0]*Bdata[j*B->num_columns+k][0]-Adata[touptri(A->num_columns,i,k)][1]*Bdata[j*B->num_columns+k][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(-Adata[touptri(A->num_columns,i,k)][0]*Bdata[j*B->num_columns+k][1]-Adata[touptri(A->num_columns,i,k)][1]*Bdata[j*B->num_columns+k][0]);
                            }
                        }
                    }
                    if (num_flops) *num_flops += 10*C->num_rows*C->num_columns*A->num_columns;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else { return MTX_ERR_INVALID_TRANSPOSITION; }
        } else if (a->field == mtx_field_integer) {
            if (Btrans == mtx_notrans) {
                if (a->precision == mtx_single) {
                    const int32_t * Adata = a->data.integer_single;
                    const int32_t * Bdata = b->data.integer_single;
                    int32_t * Cdata = c->data.integer_single;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < i; k++)
                                Cdata[i*C->num_columns+j] += alpha*Adata[touptri(A->num_columns,k,i)]*Bdata[k*B->num_columns+j];
                            for (int k = i; k < A->num_columns; k++)
                                Cdata[i*C->num_columns+j] += alpha*Adata[touptri(A->num_columns,i,k)]*Bdata[k*B->num_columns+j];
                        }
                    }
                    if (num_flops) *num_flops += 3*C->num_rows*C->num_columns*A->num_columns;
                } else if (a->precision == mtx_double) {
                    const int64_t * Adata = a->data.integer_double;
                    const int64_t * Bdata = b->data.integer_double;
                    int64_t * Cdata = c->data.integer_double;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < i; k++)
                                Cdata[i*C->num_columns+j] += alpha*Adata[touptri(A->num_columns,k,i)]*Bdata[k*B->num_columns+j];
                            for (int k = i; k < A->num_columns; k++)
                                Cdata[i*C->num_columns+j] += alpha*Adata[touptri(A->num_columns,i,k)]*Bdata[k*B->num_columns+j];
                        }
                    }
                    if (num_flops) *num_flops += 3*C->num_rows*C->num_columns*A->num_columns;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (Btrans == mtx_trans || Btrans == mtx_conjtrans) {
                if (a->precision == mtx_single) {
                    const int32_t * Adata = a->data.integer_single;
                    const int32_t * Bdata = b->data.integer_single;
                    int32_t * Cdata = c->data.integer_single;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < i; k++)
                                Cdata[i*C->num_columns+j] += alpha*Adata[touptri(A->num_columns,k,i)]*Bdata[j*B->num_columns+k];
                            for (int k = i; k < A->num_columns; k++)
                                Cdata[i*C->num_columns+j] += alpha*Adata[touptri(A->num_columns,i,k)]*Bdata[j*B->num_columns+k];
                        }
                    }
                    if (num_flops) *num_flops += 3*C->num_rows*C->num_columns*A->num_columns;
                } else if (a->precision == mtx_double) {
                    const int64_t * Adata = a->data.integer_double;
                    const int64_t * Bdata = b->data.integer_double;
                    int64_t * Cdata = c->data.integer_double;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < i; k++)
                                Cdata[i*C->num_columns+j] += alpha*Adata[touptri(A->num_columns,k,i)]*Bdata[j*B->num_columns+k];
                            for (int k = i; k < A->num_columns; k++)
                                Cdata[i*C->num_columns+j] += alpha*Adata[touptri(A->num_columns,i,k)]*Bdata[j*B->num_columns+k];
                        }
                    }
                    if (num_flops) *num_flops += 3*C->num_rows*C->num_columns*A->num_columns;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else { return MTX_ERR_INVALID_TRANSPOSITION; }
        } else { return MTX_ERR_INVALID_FIELD; }
    } else if (A->num_rows == A->num_columns && A->symmetry == mtx_skew_symmetric) {
        /* TODO: allow skew-symmetric matrices */
        return MTX_ERR_INVALID_SYMMETRY;
    } else if (A->num_rows == A->num_columns && A->symmetry == mtx_hermitian) {
        if (a->field == mtx_field_complex) {
            if ((Atrans == mtx_notrans || Atrans == mtx_conjtrans) && Btrans == mtx_notrans) {
                if (a->precision == mtx_single) {
                    const float (* Adata)[2] = a->data.complex_single;
                    const float (* Bdata)[2] = b->data.complex_single;
                    float (* Cdata)[2] = c->data.complex_single;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < i; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,k,i)][0]*Bdata[k*B->num_columns+j][0]+Adata[touptri(A->num_columns,k,i)][1]*Bdata[k*B->num_columns+j][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(Adata[touptri(A->num_columns,k,i)][0]*Bdata[k*B->num_columns+j][1]-Adata[touptri(A->num_columns,k,i)][1]*Bdata[k*B->num_columns+j][0]);
                            }
                            for (int k = i; k < A->num_columns; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,i,k)][0]*Bdata[k*B->num_columns+j][0]-Adata[touptri(A->num_columns,i,k)][1]*Bdata[k*B->num_columns+j][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(Adata[touptri(A->num_columns,i,k)][0]*Bdata[k*B->num_columns+j][1]+Adata[touptri(A->num_columns,i,k)][1]*Bdata[k*B->num_columns+j][0]);
                            }
                        }
                    }
                    if (num_flops) *num_flops += 10*C->num_rows*C->num_columns*A->num_columns;
                } else if (a->precision == mtx_double) {
                    const double (* Adata)[2] = a->data.complex_double;
                    const double (* Bdata)[2] = b->data.complex_double;
                    double (* Cdata)[2] = c->data.complex_double;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < i; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,k,i)][0]*Bdata[k*B->num_columns+j][0]+Adata[touptri(A->num_columns,k,i)][1]*Bdata[k*B->num_columns+j][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(Adata[touptri(A->num_columns,k,i)][0]*Bdata[k*B->num_columns+j][1]-Adata[touptri(A->num_columns,k,i)][1]*Bdata[k*B->num_columns+j][0]);
                            }
                            for (int k = i; k < A->num_columns; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,i,k)][0]*Bdata[k*B->num_columns+j][0]-Adata[touptri(A->num_columns,i,k)][1]*Bdata[k*B->num_columns+j][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(Adata[touptri(A->num_columns,i,k)][0]*Bdata[k*B->num_columns+j][1]+Adata[touptri(A->num_columns,i,k)][1]*Bdata[k*B->num_columns+j][0]);
                            }
                        }
                    }
                    if (num_flops) *num_flops += 10*C->num_rows*C->num_columns*A->num_columns;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if ((Atrans == mtx_notrans || Atrans == mtx_conjtrans) && Btrans == mtx_trans) {
                if (a->precision == mtx_single) {
                    const float (* Adata)[2] = a->data.complex_single;
                    const float (* Bdata)[2] = b->data.complex_single;
                    float (* Cdata)[2] = c->data.complex_single;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < i; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,k,i)][0]*Bdata[j*B->num_columns+k][0]+Adata[touptri(A->num_columns,k,i)][1]*Bdata[j*B->num_columns+k][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(Adata[touptri(A->num_columns,k,i)][0]*Bdata[j*B->num_columns+k][1]-Adata[touptri(A->num_columns,k,i)][1]*Bdata[j*B->num_columns+k][0]);
                            }
                            for (int k = i; k < A->num_columns; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,i,k)][0]*Bdata[j*B->num_columns+k][0]-Adata[touptri(A->num_columns,i,k)][1]*Bdata[j*B->num_columns+k][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(Adata[touptri(A->num_columns,i,k)][0]*Bdata[j*B->num_columns+k][1]+Adata[touptri(A->num_columns,i,k)][1]*Bdata[j*B->num_columns+k][0]);
                            }
                        }
                    }
                    if (num_flops) *num_flops += 10*C->num_rows*C->num_columns*A->num_columns;
                } else if (a->precision == mtx_double) {
                    const double (* Adata)[2] = a->data.complex_double;
                    const double (* Bdata)[2] = b->data.complex_double;
                    double (* Cdata)[2] = c->data.complex_double;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < i; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,k,i)][0]*Bdata[j*B->num_columns+k][0]+Adata[touptri(A->num_columns,k,i)][1]*Bdata[j*B->num_columns+k][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(Adata[touptri(A->num_columns,k,i)][0]*Bdata[j*B->num_columns+k][1]-Adata[touptri(A->num_columns,k,i)][1]*Bdata[j*B->num_columns+k][0]);
                            }
                            for (int k = i; k < A->num_columns; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,i,k)][0]*Bdata[j*B->num_columns+k][0]-Adata[touptri(A->num_columns,i,k)][1]*Bdata[j*B->num_columns+k][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(Adata[touptri(A->num_columns,i,k)][0]*Bdata[j*B->num_columns+k][1]+Adata[touptri(A->num_columns,i,k)][1]*Bdata[j*B->num_columns+k][0]);
                            }
                        }
                    }
                    if (num_flops) *num_flops += 10*C->num_rows*C->num_columns*A->num_columns;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if ((Atrans == mtx_notrans || Atrans == mtx_conjtrans) && Btrans == mtx_conjtrans) {
                if (a->precision == mtx_single) {
                    const float (* Adata)[2] = a->data.complex_single;
                    const float (* Bdata)[2] = b->data.complex_single;
                    float (* Cdata)[2] = c->data.complex_single;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < i; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,k,i)][0]*Bdata[j*B->num_columns+k][0]-Adata[touptri(A->num_columns,k,i)][1]*Bdata[j*B->num_columns+k][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(-Adata[touptri(A->num_columns,k,i)][0]*Bdata[j*B->num_columns+k][1]-Adata[touptri(A->num_columns,k,i)][1]*Bdata[j*B->num_columns+k][0]);
                            }
                            for (int k = i; k < A->num_columns; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,i,k)][0]*Bdata[j*B->num_columns+k][0]+Adata[touptri(A->num_columns,i,k)][1]*Bdata[j*B->num_columns+k][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(-Adata[touptri(A->num_columns,i,k)][0]*Bdata[j*B->num_columns+k][1]+Adata[touptri(A->num_columns,i,k)][1]*Bdata[j*B->num_columns+k][0]);
                            }
                        }
                    }
                    if (num_flops) *num_flops += 10*C->num_rows*C->num_columns*A->num_columns;
                } else if (a->precision == mtx_double) {
                    const double (* Adata)[2] = a->data.complex_double;
                    const double (* Bdata)[2] = b->data.complex_double;
                    double (* Cdata)[2] = c->data.complex_double;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < i; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,k,i)][0]*Bdata[j*B->num_columns+k][0]-Adata[touptri(A->num_columns,k,i)][1]*Bdata[j*B->num_columns+k][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(-Adata[touptri(A->num_columns,k,i)][0]*Bdata[j*B->num_columns+k][1]-Adata[touptri(A->num_columns,k,i)][1]*Bdata[j*B->num_columns+k][0]);
                            }
                            for (int k = i; k < A->num_columns; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,i,k)][0]*Bdata[j*B->num_columns+k][0]+Adata[touptri(A->num_columns,i,k)][1]*Bdata[j*B->num_columns+k][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(-Adata[touptri(A->num_columns,i,k)][0]*Bdata[j*B->num_columns+k][1]+Adata[touptri(A->num_columns,i,k)][1]*Bdata[j*B->num_columns+k][0]);
                            }
                        }
                    }
                    if (num_flops) *num_flops += 10*C->num_rows*C->num_columns*A->num_columns;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (Atrans == mtx_trans && Btrans == mtx_notrans) {
                if (a->precision == mtx_single) {
                    const float (* Adata)[2] = a->data.complex_single;
                    const float (* Bdata)[2] = b->data.complex_single;
                    float (* Cdata)[2] = c->data.complex_single;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < i; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,k,i)][0]*Bdata[k*B->num_columns+j][0]-Adata[touptri(A->num_columns,k,i)][1]*Bdata[k*B->num_columns+j][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(Adata[touptri(A->num_columns,k,i)][0]*Bdata[k*B->num_columns+j][1]+Adata[touptri(A->num_columns,k,i)][1]*Bdata[k*B->num_columns+j][0]);
                            }
                            for (int k = i; k < A->num_columns; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,i,k)][0]*Bdata[k*B->num_columns+j][0]+Adata[touptri(A->num_columns,i,k)][1]*Bdata[k*B->num_columns+j][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(Adata[touptri(A->num_columns,i,k)][0]*Bdata[k*B->num_columns+j][1]-Adata[touptri(A->num_columns,i,k)][1]*Bdata[k*B->num_columns+j][0]);
                            }
                        }
                    }
                    if (num_flops) *num_flops += 10*C->num_rows*C->num_columns*A->num_columns;
                } else if (a->precision == mtx_double) {
                    const double (* Adata)[2] = a->data.complex_double;
                    const double (* Bdata)[2] = b->data.complex_double;
                    double (* Cdata)[2] = c->data.complex_double;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < i; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,k,i)][0]*Bdata[k*B->num_columns+j][0]-Adata[touptri(A->num_columns,k,i)][1]*Bdata[k*B->num_columns+j][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(Adata[touptri(A->num_columns,k,i)][0]*Bdata[k*B->num_columns+j][1]+Adata[touptri(A->num_columns,k,i)][1]*Bdata[k*B->num_columns+j][0]);
                            }
                            for (int k = i; k < A->num_columns; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,i,k)][0]*Bdata[k*B->num_columns+j][0]+Adata[touptri(A->num_columns,i,k)][1]*Bdata[k*B->num_columns+j][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(Adata[touptri(A->num_columns,i,k)][0]*Bdata[k*B->num_columns+j][1]-Adata[touptri(A->num_columns,i,k)][1]*Bdata[k*B->num_columns+j][0]);
                            }
                        }
                    }
                    if (num_flops) *num_flops += 10*C->num_rows*C->num_columns*A->num_columns;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (Atrans == mtx_trans && Btrans == mtx_trans) {
                if (a->precision == mtx_single) {
                    const float (* Adata)[2] = a->data.complex_single;
                    const float (* Bdata)[2] = b->data.complex_single;
                    float (* Cdata)[2] = c->data.complex_single;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < i; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,k,i)][0]*Bdata[j*B->num_columns+k][0]-Adata[touptri(A->num_columns,k,i)][1]*Bdata[j*B->num_columns+k][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(Adata[touptri(A->num_columns,k,i)][0]*Bdata[j*B->num_columns+k][1]+Adata[touptri(A->num_columns,k,i)][1]*Bdata[j*B->num_columns+k][0]);
                            }
                            for (int k = i; k < A->num_columns; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,i,k)][0]*Bdata[j*B->num_columns+k][0]+Adata[touptri(A->num_columns,i,k)][1]*Bdata[j*B->num_columns+k][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(Adata[touptri(A->num_columns,i,k)][0]*Bdata[j*B->num_columns+k][1]-Adata[touptri(A->num_columns,i,k)][1]*Bdata[j*B->num_columns+k][0]);
                            }
                        }
                    }
                    if (num_flops) *num_flops += 10*C->num_rows*C->num_columns*A->num_columns;
                } else if (a->precision == mtx_double) {
                    const double (* Adata)[2] = a->data.complex_double;
                    const double (* Bdata)[2] = b->data.complex_double;
                    double (* Cdata)[2] = c->data.complex_double;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < i; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,k,i)][0]*Bdata[j*B->num_columns+k][0]-Adata[touptri(A->num_columns,k,i)][1]*Bdata[j*B->num_columns+k][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(Adata[touptri(A->num_columns,k,i)][0]*Bdata[j*B->num_columns+k][1]+Adata[touptri(A->num_columns,k,i)][1]*Bdata[j*B->num_columns+k][0]);
                            }
                            for (int k = i; k < A->num_columns; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,i,k)][0]*Bdata[j*B->num_columns+k][0]+Adata[touptri(A->num_columns,i,k)][1]*Bdata[j*B->num_columns+k][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(Adata[touptri(A->num_columns,i,k)][0]*Bdata[j*B->num_columns+k][1]-Adata[touptri(A->num_columns,i,k)][1]*Bdata[j*B->num_columns+k][0]);
                            }
                        }
                    }
                    if (num_flops) *num_flops += 10*C->num_rows*C->num_columns*A->num_columns;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (Atrans == mtx_trans && Btrans == mtx_conjtrans) {
                if (a->precision == mtx_single) {
                    const float (* Adata)[2] = a->data.complex_single;
                    const float (* Bdata)[2] = b->data.complex_single;
                    float (* Cdata)[2] = c->data.complex_single;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < i; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,k,i)][0]*Bdata[j*B->num_columns+k][0]+Adata[touptri(A->num_columns,k,i)][1]*Bdata[j*B->num_columns+k][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(-Adata[touptri(A->num_columns,k,i)][0]*Bdata[j*B->num_columns+k][1]+Adata[touptri(A->num_columns,k,i)][1]*Bdata[j*B->num_columns+k][0]);
                            }
                            for (int k = i; k < A->num_columns; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,i,k)][0]*Bdata[j*B->num_columns+k][0]-Adata[touptri(A->num_columns,i,k)][1]*Bdata[j*B->num_columns+k][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(-Adata[touptri(A->num_columns,i,k)][0]*Bdata[j*B->num_columns+k][1]-Adata[touptri(A->num_columns,i,k)][1]*Bdata[j*B->num_columns+k][0]);
                            }
                        }
                    }
                    if (num_flops) *num_flops += 10*C->num_rows*C->num_columns*A->num_columns;
                } else if (a->precision == mtx_double) {
                    const double (* Adata)[2] = a->data.complex_double;
                    const double (* Bdata)[2] = b->data.complex_double;
                    double (* Cdata)[2] = c->data.complex_double;
                    for (int i = 0; i < C->num_rows; i++) {
                        for (int j = 0; j < C->num_columns; j++) {
                            for (int k = 0; k < i; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,k,i)][0]*Bdata[j*B->num_columns+k][0]+Adata[touptri(A->num_columns,k,i)][1]*Bdata[j*B->num_columns+k][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(-Adata[touptri(A->num_columns,k,i)][0]*Bdata[j*B->num_columns+k][1]+Adata[touptri(A->num_columns,k,i)][1]*Bdata[j*B->num_columns+k][0]);
                            }
                            for (int k = i; k < A->num_columns; k++) {
                                Cdata[i*C->num_columns+j][0] += alpha*(Adata[touptri(A->num_columns,i,k)][0]*Bdata[j*B->num_columns+k][0]-Adata[touptri(A->num_columns,i,k)][1]*Bdata[j*B->num_columns+k][1]);
                                Cdata[i*C->num_columns+j][1] += alpha*(-Adata[touptri(A->num_columns,i,k)][0]*Bdata[j*B->num_columns+k][1]-Adata[touptri(A->num_columns,i,k)][1]*Bdata[j*B->num_columns+k][0]);
                            }
                        }
                    }
                    if (num_flops) *num_flops += 10*C->num_rows*C->num_columns*A->num_columns;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else { return MTX_ERR_INVALID_TRANSPOSITION; }
        } else { return MTX_ERR_INVALID_FIELD; }
    } else { return MTX_ERR_INVALID_SYMMETRY; }
    return MTX_SUCCESS;
#else
    return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
}
