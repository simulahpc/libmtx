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
 * Dense matrices with BLAS-accelerated operations.
 */

#include <libmtx/libmtx-config.h>

#include <libmtx/error.h>
#include <libmtx/vector/field.h>
#include <libmtx/vector/precision.h>

#include <libmtx/matrix/blas/dense.h>
#include <libmtx/matrix/matrix.h>
#include <libmtx/mtxfile/mtxfile.h>
#include <libmtx/vector/vector.h>

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

/*
 * memory management
 */

/**
 * ‘mtxmatrix_blas_free()’ frees storage allocated for a matrix.
 */
void mtxmatrix_blas_free(
    struct mtxmatrix_blas * A)
{
    mtxvector_free(&A->a);
}

/**
 * ‘mtxmatrix_blas_alloc_copy()’ allocates a copy of a matrix without
 * initialising the values.
 */
int mtxmatrix_blas_alloc_copy(
    struct mtxmatrix_blas * dst,
    const struct mtxmatrix_blas * src)
{
    enum mtxfield field;
    int err = mtxvector_field(&src->a, &field);
    if (err) return err;
    enum mtxprecision precision;
    err = mtxvector_precision(&src->a, &precision);
    return mtxmatrix_blas_alloc_entries(
        dst, src->a.type, field, precision, src->symmetry,
        src->num_rows, src->num_columns, src->size,
        0, 0, NULL, NULL);
}

/**
 * ‘mtxmatrix_blas_init_copy()’ allocates a copy of a matrix and also
 * copies the values.
 */
int mtxmatrix_blas_init_copy(
    struct mtxmatrix_blas * dst,
    const struct mtxmatrix_blas * src)
{
    int err = mtxmatrix_blas_alloc_copy(dst, src);
    if (err) return err;
    return mtxmatrix_blas_copy(dst, src);
}

/*
 * initialise matrices from entrywise data in coordinate format
 */

/**
 * ‘mtxmatrix_blas_alloc_entries()’ allocates a matrix from entrywise
 * data in coordinate format.
 */
int mtxmatrix_blas_alloc_entries(
    struct mtxmatrix_blas * A,
    enum mtxvectortype vectortype,
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
    return mtxvector_alloc(&A->a, vectortype, field, precision, A->size);
}

static int mtxmatrix_blas_init_entries_idx(
    struct mtxmatrix_blas * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    int idxstride,
    int idxbase,
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
        for (int64_t k = 0; k < size; k++) {
            int64_t i = *(const int *)((const char *)rowidx+k*idxstride)-idxbase;
            int64_t j = *(const int *)((const char *)rowidx+k*idxstride)-idxbase;
            if (i < 0 || i >= num_rows || j < 0 || j >= num_columns)
                return MTX_ERR_INDEX_OUT_OF_BOUNDS;
            idx[k] = rowidx[k]*num_columns+colidx[k];
        }
    } else if (num_rows == num_columns &&
               (symmetry == mtx_symmetric || symmetry == mtx_hermitian))
    {
        int64_t N = num_rows;
        for (int64_t k = 0; k < size; k++) {
            int64_t i = *(const int *)((const char *)rowidx+k*idxstride)-idxbase;
            int64_t j = *(const int *)((const char *)rowidx+k*idxstride)-idxbase;
            if (i < 0 || i >= num_rows || j < 0 || j >= num_columns)
                return MTX_ERR_INDEX_OUT_OF_BOUNDS;
            idx[k] = N*(N-1)/2 - (N-(i<j?i:j))*(N-(i<j?i:j)-1)/2 + (i<j?j:i);
        }
    } else if (num_rows == num_columns && symmetry == mtx_skew_symmetric) {
        int64_t N = num_rows;
        for (int64_t k = 0; k < size; k++) {
            int64_t i = *(const int *)((const char *)rowidx+k*idxstride)-idxbase;
            int64_t j = *(const int *)((const char *)rowidx+k*idxstride)-idxbase;
            if (i < 0 || i >= num_rows || j < 0 || j >= num_columns)
                return MTX_ERR_INDEX_OUT_OF_BOUNDS;
            idx[k] = N*(N-1)/2 - (N-(i<j?i:j))*(N-(i<j?i:j)-1)/2 + (i<j?j-i:i-j)-1;
        }
    } else { return MTX_ERR_INVALID_SYMMETRY; }
    return MTX_SUCCESS;
}

static int mtxmatrix_blas_init_entries_idx64(
    struct mtxmatrix_blas * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    int idxstride,
    int idxbase,
    const int64_t * rowidx,
    const int64_t * colidx,
    int64_t * idx)
{
    for (int64_t k = 0; k < size; k++) {
        if (rowidx[k] < 0 || rowidx[k] >= num_rows ||
            colidx[k] < 0 || colidx[k] >= num_columns)
            return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    }

    if (symmetry == mtx_unsymmetric) {
        for (int64_t k = 0; k < size; k++) {
            int64_t i = *(const int *)((const char *)rowidx+k*idxstride)-idxbase;
            int64_t j = *(const int *)((const char *)rowidx+k*idxstride)-idxbase;
            if (i < 0 || i >= num_rows || j < 0 || j >= num_columns)
                return MTX_ERR_INDEX_OUT_OF_BOUNDS;
            idx[k] = rowidx[k]*num_columns+colidx[k];
        }
    } else if (num_rows == num_columns &&
               (symmetry == mtx_symmetric || symmetry == mtx_hermitian))
    {
        int64_t N = num_rows;
        for (int64_t k = 0; k < size; k++) {
            int64_t i = *(const int *)((const char *)rowidx+k*idxstride)-idxbase;
            int64_t j = *(const int *)((const char *)rowidx+k*idxstride)-idxbase;
            if (i < 0 || i >= num_rows || j < 0 || j >= num_columns)
                return MTX_ERR_INDEX_OUT_OF_BOUNDS;
            idx[k] = N*(N-1)/2 - (N-(i<j?i:j))*(N-(i<j?i:j)-1)/2 + (i<j?j:i);
        }
    } else if (num_rows == num_columns && symmetry == mtx_skew_symmetric) {
        int64_t N = num_rows;
        for (int64_t k = 0; k < size; k++) {
            int64_t i = *(const int *)((const char *)rowidx+k*idxstride)-idxbase;
            int64_t j = *(const int *)((const char *)rowidx+k*idxstride)-idxbase;
            if (i < 0 || i >= num_rows || j < 0 || j >= num_columns)
                return MTX_ERR_INDEX_OUT_OF_BOUNDS;
            idx[k] = N*(N-1)/2 - (N-(i<j?i:j))*(N-(i<j?i:j)-1)/2 + (i<j?j-i:i-j)-1;
        }
    } else { return MTX_ERR_INVALID_SYMMETRY; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_blas_init_entries_real_single()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * real, single precision coefficients.
 */
int mtxmatrix_blas_init_entries_real_single(
    struct mtxmatrix_blas * A,
    enum mtxvectortype vectortype,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx,
    const float * data)
{
    int err = mtxmatrix_blas_alloc_entries(
        A, vectortype, mtx_field_real, mtx_single, symmetry, num_rows, num_columns,
        size, 0, 0, NULL, NULL);
    if (err) return err;
    err = mtxmatrix_blas_setzero(A);
    if (err) { mtxmatrix_blas_free(A); return err; }
    int64_t * idx = malloc(size * sizeof(int64_t));
    if (!idx) { mtxmatrix_blas_free(A); return MTX_ERR_ERRNO; }
    err = mtxmatrix_blas_init_entries_idx(
        A, symmetry, num_rows, num_columns, size,
        sizeof(*rowidx), 0, rowidx, colidx, idx);
    if (err) { free(idx); mtxmatrix_blas_free(A); return err; }
    struct mtxvector_packed x;
    enum mtxvectortype vectortype = mtxvector_blas;
    err = mtxvector_packed_init_real_single(
        &x, vectortype, A->size, size, idx, data);
    if (err) { free(idx); mtxmatrix_blas_free(A); return err; }
    free(idx);
    err = mtxvector_ussc(&A->a, &x);
    if (err) { mtxvector_packed_free(&x); mtxmatrix_blas_free(A); return err; }
    mtxvector_packed_free(&x);
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_blas_init_entries_real_double()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * real, double precision coefficients.
 */
int mtxmatrix_blas_init_entries_real_double(
    struct mtxmatrix_blas * A,
    enum mtxvectortype vectortype,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx,
    const double * data)
{
    int err = mtxmatrix_blas_alloc_entries(
        A, mtx_field_real, mtx_double, symmetry, num_rows, num_columns,
        size, 0, 0, NULL, NULL);
    if (err) return err;
    err = mtxmatrix_blas_setzero(A);
    if (err) { mtxmatrix_blas_free(A); return err; }
    int64_t * idx = malloc(size * sizeof(int64_t));
    if (!idx) { mtxmatrix_blas_free(A); return MTX_ERR_ERRNO; }
    err = mtxmatrix_blas_init_entries_idx(
        A, symmetry, num_rows, num_columns, size,
        sizeof(*rowidx), 0, rowidx, colidx, idx);
    if (err) { free(idx); mtxmatrix_blas_free(A); return err; }
    struct mtxvector_packed x;
    enum mtxvectortype vectortype = mtxvector_blas;
    err = mtxvector_packed_init_real_double(
        &x, vectortype, A->size, size, idx, data);
    if (err) { free(idx); mtxmatrix_blas_free(A); return err; }
    free(idx);
    err = mtxvector_ussc(&A->a, &x);
    if (err) { mtxvector_packed_free(&x); mtxmatrix_blas_free(A); return err; }
    mtxvector_packed_free(&x);
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_blas_init_entries_complex_single()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * complex, single precision coefficients.
 */
int mtxmatrix_blas_init_entries_complex_single(
    struct mtxmatrix_blas * A,
    enum mtxvectortype vectortype,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx,
    const float (* data)[2])
{
    int err = mtxmatrix_blas_alloc_entries(
        A, mtx_field_complex, mtx_single, symmetry, num_rows, num_columns,
        size, 0, 0, NULL, NULL);
    if (err) return err;
    err = mtxmatrix_blas_setzero(A);
    if (err) { mtxmatrix_blas_free(A); return err; }
    int64_t * idx = malloc(size * sizeof(int64_t));
    if (!idx) { mtxmatrix_blas_free(A); return MTX_ERR_ERRNO; }
    err = mtxmatrix_blas_init_entries_idx(
        A, symmetry, num_rows, num_columns, size,
        sizeof(*rowidx), 0, rowidx, colidx, idx);
    if (err) { free(idx); mtxmatrix_blas_free(A); return err; }
    struct mtxvector_packed x;
    enum mtxvectortype vectortype = mtxvector_blas;
    err = mtxvector_packed_init_complex_single(
        &x, vectortype, A->size, size, idx, data);
    if (err) { free(idx); mtxmatrix_blas_free(A); return err; }
    free(idx);
    err = mtxvector_ussc(&A->a, &x);
    if (err) { mtxvector_packed_free(&x); mtxmatrix_blas_free(A); return err; }
    mtxvector_packed_free(&x);
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_blas_init_entries_complex_double()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * complex, double precision coefficients.
 */
int mtxmatrix_blas_init_entries_complex_double(
    struct mtxmatrix_blas * A,
    enum mtxvectortype vectortype,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx,
    const double (* data)[2])
{
    int err = mtxmatrix_blas_alloc_entries(
        A, mtx_field_complex, mtx_double, symmetry, num_rows, num_columns,
        size, 0, 0, NULL, NULL);
    if (err) return err;
    err = mtxmatrix_blas_setzero(A);
    if (err) { mtxmatrix_blas_free(A); return err; }
    int64_t * idx = malloc(size * sizeof(int64_t));
    if (!idx) { mtxmatrix_blas_free(A); return MTX_ERR_ERRNO; }
    err = mtxmatrix_blas_init_entries_idx(
        A, symmetry, num_rows, num_columns, size,
        sizeof(*rowidx), 0, rowidx, colidx, idx);
    if (err) { free(idx); mtxmatrix_blas_free(A); return err; }
    struct mtxvector_packed x;
    enum mtxvectortype vectortype = mtxvector_blas;
    err = mtxvector_packed_init_complex_double(
        &x, vectortype, A->size, size, idx, data);
    if (err) { free(idx); mtxmatrix_blas_free(A); return err; }
    free(idx);
    err = mtxvector_ussc(&A->a, &x);
    if (err) { mtxvector_packed_free(&x); mtxmatrix_blas_free(A); return err; }
    mtxvector_packed_free(&x);
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_blas_init_entries_integer_single()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * integer, single precision coefficients.
 */
int mtxmatrix_blas_init_entries_integer_single(
    struct mtxmatrix_blas * A,
    enum mtxvectortype vectortype,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx,
    const int32_t * data)
{
    int err = mtxmatrix_blas_alloc_entries(
        A, mtx_field_integer, mtx_single, symmetry, num_rows, num_columns,
        size, 0, 0, NULL, NULL);
    if (err) return err;
    err = mtxmatrix_blas_setzero(A);
    if (err) { mtxmatrix_blas_free(A); return err; }
    int64_t * idx = malloc(size * sizeof(int64_t));
    if (!idx) { mtxmatrix_blas_free(A); return MTX_ERR_ERRNO; }
    err = mtxmatrix_blas_init_entries_idx(
        A, symmetry, num_rows, num_columns, size,
        sizeof(*rowidx), 0, rowidx, colidx, idx);
    if (err) { free(idx); mtxmatrix_blas_free(A); return err; }
    struct mtxvector_packed x;
    enum mtxvectortype vectortype = mtxvector_blas;
    err = mtxvector_packed_init_integer_single(
        &x, vectortype, A->size, size, idx, data);
    if (err) { free(idx); mtxmatrix_blas_free(A); return err; }
    free(idx);
    err = mtxvector_ussc(&A->a, &x);
    if (err) { mtxvector_packed_free(&x); mtxmatrix_blas_free(A); return err; }
    mtxvector_packed_free(&x);
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_blas_init_entries_integer_double()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * integer, double precision coefficients.
 */
int mtxmatrix_blas_init_entries_integer_double(
    struct mtxmatrix_blas * A,
    enum mtxvectortype vectortype,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx,
    const int64_t * data)
{
    int err = mtxmatrix_blas_alloc_entries(
        A, mtx_field_integer, mtx_double, symmetry,
        num_rows, num_columns, size, 0, 0, NULL, NULL);
    if (err) return err;
    err = mtxmatrix_blas_setzero(A);
    if (err) { mtxmatrix_blas_free(A); return err; }
    int64_t * idx = malloc(size * sizeof(int64_t));
    if (!idx) { mtxmatrix_blas_free(A); return MTX_ERR_ERRNO; }
    err = mtxmatrix_blas_init_entries_idx(
        A, symmetry, num_rows, num_columns, size,
        sizeof(*rowidx), 0, rowidx, colidx, idx);
    if (err) { free(idx); mtxmatrix_blas_free(A); return err; }
    struct mtxvector_packed x;
    enum mtxvectortype vectortype = mtxvector_blas;
    err = mtxvector_packed_init_integer_double(
        &x, vectortype, A->size, size, idx, data);
    if (err) { free(idx); mtxmatrix_blas_free(A); return err; }
    free(idx);
    err = mtxvector_ussc(&A->a, &x);
    if (err) { mtxvector_packed_free(&x); mtxmatrix_blas_free(A); return err; }
    mtxvector_packed_free(&x);
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_blas_init_entries_pattern()’ allocates and initialises
 * a matrix from entrywise data in coordinate format with boolean
 * coefficients.
 */
int mtxmatrix_blas_init_entries_pattern(
    struct mtxmatrix_blas * A,
    enum mtxvectortype vectortype,
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
 * ‘mtxmatrix_blas_init_entries_strided_real_single()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * real, single precision coefficients.
 */
int mtxmatrix_blas_init_entries_strided_real_single(
    struct mtxmatrix_blas * A,
    enum mtxvectortype vectortype,
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
 * ‘mtxmatrix_blas_init_entries_strided_real_double()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * real, double precision coefficients.
 */
int mtxmatrix_blas_init_entries_strided_real_double(
    struct mtxmatrix_blas * A,
    enum mtxvectortype vectortype,
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
 * ‘mtxmatrix_blas_init_entries_strided_complex_single()’ allocates
 * and initialises a matrix from entrywise data in coordinate format
 * with complex, single precision coefficients.
 */
int mtxmatrix_blas_init_entries_strided_complex_single(
    struct mtxmatrix_blas * A,
    enum mtxvectortype vectortype,
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
 * ‘mtxmatrix_blas_init_entries_strided_complex_double()’ allocates
 * and initialises a matrix from entrywise data in coordinate format
 * with complex, double precision coefficients.
 */
int mtxmatrix_blas_init_entries_strided_complex_double(
    struct mtxmatrix_blas * A,
    enum mtxvectortype vectortype,
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
 * ‘mtxmatrix_blas_init_entries_strided_integer_single()’ allocates
 * and initialises a matrix from entrywise data in coordinate format
 * with integer, single precision coefficients.
 */
int mtxmatrix_blas_init_entries_strided_integer_single(
    struct mtxmatrix_blas * A,
    enum mtxvectortype vectortype,
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
 * ‘mtxmatrix_blas_init_entries_strided_integer_double()’ allocates
 * and initialises a matrix from entrywise data in coordinate format
 * with integer, double precision coefficients.
 */
int mtxmatrix_blas_init_entries_strided_integer_double(
    struct mtxmatrix_blas * A,
    enum mtxvectortype vectortype,
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
 * ‘mtxmatrix_blas_init_entries_strided_pattern()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * boolean coefficients.
 */
int mtxmatrix_blas_init_entries_strided_pattern(
    struct mtxmatrix_blas * A,
    enum mtxvectortype vectortype,
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
 * ‘mtxmatrix_blas_alloc_rows()’ allocates a matrix from row-wise
 * data in compressed row format.
 */
int mtxmatrix_blas_alloc_rows(
    struct mtxmatrix_blas * A,
    enum mtxvectortype vectortype,
    enum mtxfield field,
    enum mtxprecision precision,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx);

/**
 * ‘mtxmatrix_blas_init_rows_real_single()’ allocates and initialises
 * a matrix from row-wise data in compressed row format with real,
 * single precision coefficients.
 */
int mtxmatrix_blas_init_rows_real_single(
    struct mtxmatrix_blas * A,
    enum mtxvectortype vectortype,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const float * data);

/**
 * ‘mtxmatrix_blas_init_rows_real_double()’ allocates and initialises
 * a matrix from row-wise data in compressed row format with real,
 * double precision coefficients.
 */
int mtxmatrix_blas_init_rows_real_double(
    struct mtxmatrix_blas * A,
    enum mtxvectortype vectortype,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const double * data);

/**
 * ‘mtxmatrix_blas_init_rows_complex_single()’ allocates and
 * initialises a matrix from row-wise data in compressed row format
 * with complex, single precision coefficients.
 */
int mtxmatrix_blas_init_rows_complex_single(
    struct mtxmatrix_blas * A,
    enum mtxvectortype vectortype,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const float (* data)[2]);

/**
 * ‘mtxmatrix_blas_init_rows_complex_double()’ allocates and
 * initialises a matrix from row-wise data in compressed row format
 * with complex, double precision coefficients.
 */
int mtxmatrix_blas_init_rows_complex_double(
    struct mtxmatrix_blas * A,
    enum mtxvectortype vectortype,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const double (* data)[2]);

/**
 * ‘mtxmatrix_blas_init_rows_integer_single()’ allocates and
 * initialises a matrix from row-wise data in compressed row format
 * with integer, single precision coefficients.
 */
int mtxmatrix_blas_init_rows_integer_single(
    struct mtxmatrix_blas * A,
    enum mtxvectortype vectortype,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const int32_t * data);

/**
 * ‘mtxmatrix_blas_init_rows_integer_double()’ allocates and
 * initialises a matrix from row-wise data in compressed row format
 * with integer, double precision coefficients.
 */
int mtxmatrix_blas_init_rows_integer_double(
    struct mtxmatrix_blas * A,
    enum mtxvectortype vectortype,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const int64_t * data);

/**
 * ‘mtxmatrix_blas_init_rows_pattern()’ allocates and initialises a
 * matrix from row-wise data in compressed row format with boolean
 * coefficients.
 */
int mtxmatrix_blas_init_rows_pattern(
    struct mtxmatrix_blas * A,
    enum mtxvectortype vectortype,
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
 * ‘mtxmatrix_blas_alloc_columns()’ allocates a matrix from
 * column-wise data in compressed column format.
 */
int mtxmatrix_blas_alloc_columns(
    struct mtxmatrix_blas * A,
    enum mtxvectortype vectortype,
    enum mtxfield field,
    enum mtxprecision precision,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx);

/**
 * ‘mtxmatrix_blas_init_columns_real_single()’ allocates and
 * initialises a matrix from column-wise data in compressed column
 * format with real, single precision coefficients.
 */
int mtxmatrix_blas_init_columns_real_single(
    struct mtxmatrix_blas * A,
    enum mtxvectortype vectortype,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx,
    const float * data);

/**
 * ‘mtxmatrix_blas_init_columns_real_double()’ allocates and
 * initialises a matrix from column-wise data in compressed column
 * format with real, double precision coefficients.
 */
int mtxmatrix_blas_init_columns_real_double(
    struct mtxmatrix_blas * A,
    enum mtxvectortype vectortype,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx,
    const double * data);

/**
 * ‘mtxmatrix_blas_init_columns_complex_single()’ allocates and
 * initialises a matrix from column-wise data in compressed column
 * format with complex, single precision coefficients.
 */
int mtxmatrix_blas_init_columns_complex_single(
    struct mtxmatrix_blas * A,
    enum mtxvectortype vectortype,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx,
    const float (* data)[2]);

/**
 * ‘mtxmatrix_blas_init_columns_complex_double()’ allocates and
 * initialises a matrix from column-wise data in compressed column
 * format with complex, double precision coefficients.
 */
int mtxmatrix_blas_init_columns_complex_double(
    struct mtxmatrix_blas * A,
    enum mtxvectortype vectortype,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx,
    const double (* data)[2]);

/**
 * ‘mtxmatrix_blas_init_columns_integer_single()’ allocates and
 * initialises a matrix from column-wise data in compressed column
 * format with integer, single precision coefficients.
 */
int mtxmatrix_blas_init_columns_integer_single(
    struct mtxmatrix_blas * A,
    enum mtxvectortype vectortype,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx,
    const int32_t * data);

/**
 * ‘mtxmatrix_blas_init_columns_integer_double()’ allocates and
 * initialises a matrix from column-wise data in compressed column
 * format with integer, double precision coefficients.
 */
int mtxmatrix_blas_init_columns_integer_double(
    struct mtxmatrix_blas * A,
    enum mtxvectortype vectortype,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx,
    const int64_t * data);

/**
 * ‘mtxmatrix_blas_init_columns_pattern()’ allocates and initialises
 * a matrix from column-wise data in compressed column format with
 * boolean coefficients.
 */
int mtxmatrix_blas_init_columns_pattern(
    struct mtxmatrix_blas * A,
    enum mtxvectortype vectortype,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx);

/*
 * initialise matrices from a list of dense cliques
 */

/**
 * ‘mtxmatrix_blas_alloc_cliques()’ allocates a matrix from a list of
 * dense cliques.
 */
int mtxmatrix_blas_alloc_cliques(
    struct mtxmatrix_blas * A,
    enum mtxvectortype vectortype,
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
 * ‘mtxmatrix_blas_init_cliques_real_single()’ allocates and
 * initialises a matrix from a list of dense cliques with real, single
 * precision coefficients.
 */
int mtxmatrix_blas_init_cliques_real_single(
    struct mtxmatrix_blas * A,
    enum mtxvectortype vectortype,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_cliques,
    const int64_t * cliqueptr,
    const int * rowidx,
    const int * colidx,
    const float * data);

/**
 * ‘mtxmatrix_blas_init_cliques_real_double()’ allocates and
 * initialises a matrix from a list of dense cliques with real, double
 * precision coefficients.
 */
int mtxmatrix_blas_init_cliques_real_double(
    struct mtxmatrix_blas * A,
    enum mtxvectortype vectortype,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_cliques,
    const int64_t * cliqueptr,
    const int * rowidx,
    const int * colidx,
    const double * data);

/**
 * ‘mtxmatrix_blas_init_cliques_complex_single()’ allocates and
 * initialises a matrix from a list of dense cliques with complex,
 * single precision coefficients.
 */
int mtxmatrix_blas_init_cliques_complex_single(
    struct mtxmatrix_blas * A,
    enum mtxvectortype vectortype,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_cliques,
    const int64_t * cliqueptr,
    const int * rowidx,
    const int * colidx,
    const float (* data)[2]);

/**
 * ‘mtxmatrix_blas_init_cliques_complex_double()’ allocates and
 * initialises a matrix from a list of dense cliques with complex,
 * double precision coefficients.
 */
int mtxmatrix_blas_init_cliques_complex_double(
    struct mtxmatrix_blas * A,
    enum mtxvectortype vectortype,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_cliques,
    const int64_t * cliqueptr,
    const int * rowidx,
    const int * colidx,
    const double (* data)[2]);

/**
 * ‘mtxmatrix_blas_init_cliques_integer_single()’ allocates and
 * initialises a matrix from a list of dense cliques with integer,
 * single precision coefficients.
 */
int mtxmatrix_blas_init_cliques_integer_single(
    struct mtxmatrix_blas * A,
    enum mtxvectortype vectortype,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_cliques,
    const int64_t * cliqueptr,
    const int * rowidx,
    const int * colidx,
    const int32_t * data);

/**
 * ‘mtxmatrix_blas_init_cliques_integer_double()’ allocates and
 * initialises a matrix from a list of dense cliques with integer,
 * double precision coefficients.
 */
int mtxmatrix_blas_init_cliques_integer_double(
    struct mtxmatrix_blas * A,
    enum mtxvectortype vectortype,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_cliques,
    const int64_t * cliqueptr,
    const int * rowidx,
    const int * colidx,
    const int64_t * data);

/**
 * ‘mtxmatrix_blas_init_cliques_pattern()’ allocates and initialises
 * a matrix from a list of dense cliques with boolean coefficients.
 */
int mtxmatrix_blas_init_cliques_pattern(
    struct mtxmatrix_blas * A,
    enum mtxvectortype vectortype,
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
 * ‘mtxmatrix_blas_setzero()’ sets every value of a matrix to zero.
 */
int mtxmatrix_blas_setzero(
    struct mtxmatrix_blas * A)
{
    return mtxvector_setzero(&A->a);
}

/**
 * ‘mtxmatrix_blas_set_real_single()’ sets values of a matrix based
 * on an array of single precision floating point numbers.
 */
int mtxmatrix_blas_set_real_single(
    struct mtxmatrix_blas * A,
    int64_t size,
    int stride,
    const float * a)
{
    return mtxvector_set_real_single(&A->a, size, stride, a);
}

/**
 * ‘mtxmatrix_blas_set_real_double()’ sets values of a matrix based
 * on an array of double precision floating point numbers.
 */
int mtxmatrix_blas_set_real_double(
    struct mtxmatrix_blas * A,
    int64_t size,
    int stride,
    const double * a)
{
    return mtxvector_set_real_double(&A->a, size, stride, a);
}

/**
 * ‘mtxmatrix_blas_set_complex_single()’ sets values of a matrix
 * based on an array of single precision floating point complex
 * numbers.
 */
int mtxmatrix_blas_set_complex_single(
    struct mtxmatrix_blas * A,
    int64_t size,
    int stride,
    const float (*a)[2])
{
    return mtxvector_set_complex_single(&A->a, size, stride, a);
}

/**
 * ‘mtxmatrix_blas_set_complex_double()’ sets values of a matrix
 * based on an array of double precision floating point complex
 * numbers.
 */
int mtxmatrix_blas_set_complex_double(
    struct mtxmatrix_blas * A,
    int64_t size,
    int stride,
    const double (*a)[2])
{
    return mtxvector_set_complex_double(&A->a, size, stride, a);
}

/**
 * ‘mtxmatrix_blas_set_integer_single()’ sets values of a matrix
 * based on an array of integers.
 */
int mtxmatrix_blas_set_integer_single(
    struct mtxmatrix_blas * A,
    int64_t size,
    int stride,
    const int32_t * a)
{
    return mtxvector_set_integer_single(&A->a, size, stride, a);
}

/**
 * ‘mtxmatrix_blas_set_integer_double()’ sets values of a matrix
 * based on an array of integers.
 */
int mtxmatrix_blas_set_integer_double(
    struct mtxmatrix_blas * A,
    int64_t size,
    int stride,
    const int64_t * a)
{
    return mtxvector_set_integer_double(&A->a, size, stride, a);
}

/*
 * row and column vectors
 */

/**
 * ‘mtxmatrix_blas_alloc_row_vector()’ allocates a row vector for a
 * given matrix, where a row vector is a vector whose length equal to
 * a single row of the matrix.
 */
int mtxmatrix_blas_alloc_row_vector(
    const struct mtxmatrix_blas * A,
    struct mtxvector * x,
    enum mtxvectortype vectortype)
{
    enum mtxfield field;
    int err = mtxvector_field(&A->a, &field);
    if (err) return err;
    enum mtxprecision precision;
    err = mtxvector_precision(&A->a, &precision);
    if (err) return err;
    return mtxvector_alloc(
        x, vectortype, field, precision, A->num_columns);
}

/**
 * ‘mtxmatrix_blas_alloc_column_vector()’ allocates a column vector
 * for a given matrix, where a column vector is a vector whose length
 * equal to a single column of the matrix.
 */
int mtxmatrix_blas_alloc_column_vector(
    const struct mtxmatrix_blas * A,
    struct mtxvector * y,
    enum mtxvectortype vectortype)
{
    enum mtxfield field;
    int err = mtxvector_field(&A->a, &field);
    if (err) return err;
    enum mtxprecision precision;
    err = mtxvector_precision(&A->a, &precision);
    if (err) return err;
    return mtxvector_alloc(
        y, vectortype, field, precision, A->num_rows);
}

/*
 * convert to and from Matrix Market format
 */

/**
 * ‘mtxmatrix_blas_from_mtxfile()’ converts a matrix from Matrix
 * Market format.
 */
int mtxmatrix_blas_from_mtxfile(
    struct mtxmatrix_blas * A,
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

    err = mtxmatrix_blas_alloc_entries(
        A, field, precision, symmetry, num_rows, num_columns,
        num_nonzeros, 0, 0, NULL, NULL);
    if (err) return err;
    err = mtxmatrix_blas_setzero(A);
    if (err) { mtxmatrix_blas_free(A); return err; }

    if (mtxfile->header.format == mtxfile_coordinate) {
        if (mtxfile->header.field == mtxfile_real) {
            if (mtxfile->precision == mtx_single) {
                const struct mtxfile_matrix_coordinate_real_single * data =
                    mtxfile->data.matrix_coordinate_real_single;
                int64_t * idx = malloc(num_nonzeros * sizeof(int64_t));
                if (!idx) { mtxmatrix_blas_free(A); return MTX_ERR_ERRNO; }
                err = mtxmatrix_blas_init_entries_idx64(
                    A, symmetry, num_rows, num_columns, num_nonzeros,
                    sizeof(*data), 1, &data[0].i, &data[0].j, idx);
                if (err) { free(idx); mtxmatrix_blas_free(A); return err; }
                struct mtxvector_packed x = {
                    .size = A->size, .num_nonzeros = num_nonzeros, .idx = idx};
                enum mtxvectortype vectortype = mtxvector_blas;
                err = mtxvector_init_strided_real_single(
                    &x.x, vectortype, num_nonzeros, sizeof(*data), &data[0].a);
                if (err) { free(idx); mtxmatrix_blas_free(A); return err; }
                err = mtxvector_ussc(&A->a, &x);
                if (err) { mtxvector_packed_free(&x); mtxmatrix_blas_free(A); return err; }
                mtxvector_packed_free(&x);
            } else if (mtxfile->precision == mtx_double) {
                const struct mtxfile_matrix_coordinate_real_double * data =
                    mtxfile->data.matrix_coordinate_real_double;
                int64_t * idx = malloc(num_nonzeros * sizeof(int64_t));
                if (!idx) { mtxmatrix_blas_free(A); return MTX_ERR_ERRNO; }
                err = mtxmatrix_blas_init_entries_idx64(
                    A, symmetry, num_rows, num_columns, num_nonzeros,
                    sizeof(*data), 1, &data[0].i, &data[0].j, idx);
                if (err) { free(idx); mtxmatrix_blas_free(A); return err; }
                struct mtxvector_packed x = {
                    .size = A->size, .num_nonzeros = num_nonzeros, .idx = idx};
                enum mtxvectortype vectortype = mtxvector_blas;
                err = mtxvector_init_strided_real_double(
                    &x.x, vectortype, num_nonzeros, sizeof(*data), &data[0].a);
                if (err) { free(idx); mtxmatrix_blas_free(A); return err; }
                err = mtxvector_ussc(&A->a, &x);
                if (err) { mtxvector_packed_free(&x); mtxmatrix_blas_free(A); return err; }
                mtxvector_packed_free(&x);
            } else { mtxmatrix_blas_free(A); return MTX_ERR_INVALID_PRECISION; }
        } else if (mtxfile->header.field == mtxfile_complex) {
            if (mtxfile->precision == mtx_single) {
                const struct mtxfile_matrix_coordinate_complex_single * data =
                    mtxfile->data.matrix_coordinate_complex_single;
                int64_t * idx = malloc(num_nonzeros * sizeof(int64_t));
                if (!idx) { mtxmatrix_blas_free(A); return MTX_ERR_ERRNO; }
                err = mtxmatrix_blas_init_entries_idx64(
                    A, symmetry, num_rows, num_columns, num_nonzeros,
                    sizeof(*data), 1, &data[0].i, &data[0].j, idx);
                if (err) { free(idx); mtxmatrix_blas_free(A); return err; }
                struct mtxvector_packed x = {
                    .size = A->size, .num_nonzeros = num_nonzeros, .idx = idx};
                enum mtxvectortype vectortype = mtxvector_blas;
                err = mtxvector_init_strided_complex_single(
                    &x.x, vectortype, num_nonzeros, sizeof(*data), &data[0].a);
                if (err) { free(idx); mtxmatrix_blas_free(A); return err; }
                err = mtxvector_ussc(&A->a, &x);
                if (err) { mtxvector_packed_free(&x); mtxmatrix_blas_free(A); return err; }
                mtxvector_packed_free(&x);
            } else if (mtxfile->precision == mtx_double) {
                const struct mtxfile_matrix_coordinate_complex_double * data =
                    mtxfile->data.matrix_coordinate_complex_double;
                int64_t * idx = malloc(num_nonzeros * sizeof(int64_t));
                if (!idx) { mtxmatrix_blas_free(A); return MTX_ERR_ERRNO; }
                err = mtxmatrix_blas_init_entries_idx64(
                    A, symmetry, num_rows, num_columns, num_nonzeros,
                    sizeof(*data), 1, &data[0].i, &data[0].j, idx);
                if (err) { free(idx); mtxmatrix_blas_free(A); return err; }
                struct mtxvector_packed x = {
                    .size = A->size, .num_nonzeros = num_nonzeros, .idx = idx};
                enum mtxvectortype vectortype = mtxvector_blas;
                err = mtxvector_init_strided_complex_double(
                    &x.x, vectortype, num_nonzeros, sizeof(*data), &data[0].a);
                if (err) { free(idx); mtxmatrix_blas_free(A); return err; }
                err = mtxvector_ussc(&A->a, &x);
                if (err) { mtxvector_packed_free(&x); mtxmatrix_blas_free(A); return err; }
                mtxvector_packed_free(&x);
            } else { mtxmatrix_blas_free(A); return MTX_ERR_INVALID_PRECISION; }
        } else if (mtxfile->header.field == mtxfile_integer) {
            if (mtxfile->precision == mtx_single) {
                const struct mtxfile_matrix_coordinate_integer_single * data =
                    mtxfile->data.matrix_coordinate_integer_single;
                int64_t * idx = malloc(num_nonzeros * sizeof(int64_t));
                if (!idx) { mtxmatrix_blas_free(A); return MTX_ERR_ERRNO; }
                err = mtxmatrix_blas_init_entries_idx64(
                    A, symmetry, num_rows, num_columns, num_nonzeros,
                    sizeof(*data), 1, &data[0].i, &data[0].j, idx);
                if (err) { free(idx); mtxmatrix_blas_free(A); return err; }
                struct mtxvector_packed x = {
                    .size = A->size, .num_nonzeros = num_nonzeros, .idx = idx};
                enum mtxvectortype vectortype = mtxvector_blas;
                err = mtxvector_init_strided_integer_single(
                    &x.x, vectortype, num_nonzeros, sizeof(*data), &data[0].a);
                if (err) { free(idx); mtxmatrix_blas_free(A); return err; }
                err = mtxvector_ussc(&A->a, &x);
                if (err) { mtxvector_packed_free(&x); mtxmatrix_blas_free(A); return err; }
                mtxvector_packed_free(&x);
            } else if (mtxfile->precision == mtx_double) {
                const struct mtxfile_matrix_coordinate_integer_double * data =
                    mtxfile->data.matrix_coordinate_integer_double;
                int64_t * idx = malloc(num_nonzeros * sizeof(int64_t));
                if (!idx) { mtxmatrix_blas_free(A); return MTX_ERR_ERRNO; }
                err = mtxmatrix_blas_init_entries_idx64(
                    A, symmetry, num_rows, num_columns, num_nonzeros,
                    sizeof(*data), 1, &data[0].i, &data[0].j, idx);
                if (err) { free(idx); mtxmatrix_blas_free(A); return err; }
                struct mtxvector_packed x = {
                    .size = A->size, .num_nonzeros = num_nonzeros, .idx = idx};
                enum mtxvectortype vectortype = mtxvector_blas;
                err = mtxvector_init_strided_integer_double(
                    &x.x, vectortype, num_nonzeros, sizeof(*data), &data[0].a);
                if (err) { free(idx); mtxmatrix_blas_free(A); return err; }
                err = mtxvector_ussc(&A->a, &x);
                if (err) { mtxvector_packed_free(&x); mtxmatrix_blas_free(A); return err; }
                mtxvector_packed_free(&x);
            } else { mtxmatrix_blas_free(A); return MTX_ERR_INVALID_PRECISION; }
        } else { mtxmatrix_blas_free(A); return MTX_ERR_INVALID_MTX_FIELD; }
    } else if (mtxfile->header.format == mtxfile_array) {
        if (mtxfile->header.field == mtxfile_real) {
            if (mtxfile->precision == mtx_single) {
                const float * data = mtxfile->data.array_real_single;
                err = mtxvector_set_real_single(
                    &A->a, A->size, sizeof(*data), data);
                if (err) { mtxmatrix_blas_free(A); return err; }
            } else if (mtxfile->precision == mtx_double) {
                const double * data = mtxfile->data.array_real_double;
                err = mtxvector_set_real_double(
                    &A->a, A->size, sizeof(*data), data);
                if (err) { mtxmatrix_blas_free(A); return err; }
            } else { mtxmatrix_blas_free(A); return MTX_ERR_INVALID_PRECISION; }
        } else if (mtxfile->header.field == mtxfile_complex) {
            if (mtxfile->precision == mtx_single) {
                const float (*data)[2] = mtxfile->data.array_complex_single;
                err = mtxvector_set_complex_single(
                    &A->a, A->size, sizeof(*data), data);
                if (err) { mtxmatrix_blas_free(A); return err; }
            } else if (mtxfile->precision == mtx_double) {
                const double (*data)[2] = mtxfile->data.array_complex_double;
                err = mtxvector_set_complex_double(
                    &A->a, A->size, sizeof(*data), data);
                if (err) { mtxmatrix_blas_free(A); return err; }
            } else { mtxmatrix_blas_free(A); return MTX_ERR_INVALID_PRECISION; }
        } else if (mtxfile->header.field == mtxfile_integer) {
            if (mtxfile->precision == mtx_single) {
                const int32_t * data = mtxfile->data.array_integer_single;
                err = mtxvector_set_integer_single(
                    &A->a, A->size, sizeof(*data), data);
                if (err) { mtxmatrix_blas_free(A); return err; }
            } else if (mtxfile->precision == mtx_double) {
                const int64_t * data = mtxfile->data.array_integer_double;
                err = mtxvector_set_integer_double(
                    &A->a, A->size, sizeof(*data), data);
                if (err) { mtxmatrix_blas_free(A); return err; }
            } else { mtxmatrix_blas_free(A); return MTX_ERR_INVALID_PRECISION; }
        } else { mtxmatrix_blas_free(A); return MTX_ERR_INVALID_MTX_FIELD; }
    } else { mtxmatrix_blas_free(A); return MTX_ERR_INVALID_MTX_FORMAT; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_blas_to_mtxfile()’ converts a matrix to Matrix Market
 * format.
 */
int mtxmatrix_blas_to_mtxfile(
    struct mtxfile * mtxfile,
    const struct mtxmatrix_blas * A,
    int64_t num_rows,
    const int64_t * rowidx,
    int64_t num_columns,
    const int64_t * colidx,
    enum mtxfileformat mtxfmt)
{
    enum mtxsymmetry symmetry = A->symmetry;
    enum mtxfilesymmetry mtxsymmetry;
    int err = mtxfilesymmetry_from_mtxsymmetry(&mtxsymmetry, symmetry);
    if (err) return err;

    enum mtxfield field;
    err = mtxvector_field(&A->a, &field);
    if (err) return err;
    enum mtxfilefield mtxfield;
    err = mtxfilefield_from_mtxfield(&mtxfield, field);
    if (err) return err;
    enum mtxprecision precision;
    err = mtxvector_precision(&A->a, &precision);
    if (err) return err;

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
                err = mtxvector_get_real_single(
                    &A->a, A->size, sizeof(*data), &data[0].a);
                if (err) { mtxfile_free(mtxfile); return err; }
                if (symmetry == mtx_unsymmetric) {
                    for (int64_t i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = 0; j < A->num_columns; j++, k++) {
                            data[k].i = rowidx ? rowidx[i]+1 : i+1;
                            data[k].j = colidx ? colidx[j]+1 : j+1;
                        }
                    }
                } else if (num_rows == num_columns && (symmetry == mtx_symmetric || symmetry == mtx_hermitian)) {
                    for (int64_t i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = i; j < A->num_columns; j++, k++) {
                            data[k].i = rowidx ? rowidx[i]+1 : i+1;
                            data[k].j = colidx ? colidx[j]+1 : j+1;
                        }
                    }
                } else if (num_rows == num_columns && symmetry == mtx_skew_symmetric) {
                    for (int64_t i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = i+1; j < A->num_columns; j++, k++) {
                            data[k].i = rowidx ? rowidx[i]+1 : i+1;
                            data[k].j = colidx ? colidx[j]+1 : j+1;
                        }
                    }
                } else { mtxfile_free(mtxfile); return MTX_ERR_INVALID_SYMMETRY; }
            } else if (precision == mtx_double) {
                struct mtxfile_matrix_coordinate_real_double * data =
                    mtxfile->data.matrix_coordinate_real_double;
                err = mtxvector_get_real_double(
                    &A->a, A->size, sizeof(*data), &data[0].a);
                if (err) { mtxfile_free(mtxfile); return err; }
                if (symmetry == mtx_unsymmetric) {
                    for (int64_t i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = 0; j < A->num_columns; j++, k++) {
                            data[k].i = rowidx ? rowidx[i]+1 : i+1;
                            data[k].j = colidx ? colidx[j]+1 : j+1;
                        }
                    }
                } else if (num_rows == num_columns && (symmetry == mtx_symmetric || symmetry == mtx_hermitian)) {
                    for (int64_t i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = i; j < A->num_columns; j++, k++) {
                            data[k].i = rowidx ? rowidx[i]+1 : i+1;
                            data[k].j = colidx ? colidx[j]+1 : j+1;
                        }
                    }
                } else if (num_rows == num_columns && symmetry == mtx_skew_symmetric) {
                    for (int64_t i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = i+1; j < A->num_columns; j++, k++) {
                            data[k].i = rowidx ? rowidx[i]+1 : i+1;
                            data[k].j = colidx ? colidx[j]+1 : j+1;
                        }
                    }
                } else { mtxfile_free(mtxfile); return MTX_ERR_INVALID_SYMMETRY; }
            } else { mtxfile_free(mtxfile); return MTX_ERR_INVALID_PRECISION; }
        } else if (field == mtx_field_complex) {
            if (precision == mtx_single) {
                struct mtxfile_matrix_coordinate_complex_single * data =
                    mtxfile->data.matrix_coordinate_complex_single;
                err = mtxvector_get_complex_single(
                    &A->a, A->size, sizeof(*data), &data[0].a);
                if (err) { mtxfile_free(mtxfile); return err; }
                if (symmetry == mtx_unsymmetric) {
                    for (int64_t i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = 0; j < A->num_columns; j++, k++) {
                            data[k].i = rowidx ? rowidx[i]+1 : i+1;
                            data[k].j = colidx ? colidx[j]+1 : j+1;
                        }
                    }
                } else if (num_rows == num_columns && (symmetry == mtx_symmetric || symmetry == mtx_hermitian)) {
                    for (int64_t i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = i; j < A->num_columns; j++, k++) {
                            data[k].i = rowidx ? rowidx[i]+1 : i+1;
                            data[k].j = colidx ? colidx[j]+1 : j+1;
                        }
                    }
                } else if (num_rows == num_columns && symmetry == mtx_skew_symmetric) {
                    for (int64_t i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = i+1; j < A->num_columns; j++, k++) {
                            data[k].i = rowidx ? rowidx[i]+1 : i+1;
                            data[k].j = colidx ? colidx[j]+1 : j+1;
                        }
                    }
                } else { mtxfile_free(mtxfile); return MTX_ERR_INVALID_SYMMETRY; }
            } else if (precision == mtx_double) {
                struct mtxfile_matrix_coordinate_complex_double * data =
                    mtxfile->data.matrix_coordinate_complex_double;
                err = mtxvector_get_complex_double(
                    &A->a, A->size, sizeof(*data), &data[0].a);
                if (err) { mtxfile_free(mtxfile); return err; }
                if (symmetry == mtx_unsymmetric) {
                    for (int64_t i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = 0; j < A->num_columns; j++, k++) {
                            data[k].i = rowidx ? rowidx[i]+1 : i+1;
                            data[k].j = colidx ? colidx[j]+1 : j+1;
                        }
                    }
                } else if (num_rows == num_columns && (symmetry == mtx_symmetric || symmetry == mtx_hermitian)) {
                    for (int64_t i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = i; j < A->num_columns; j++, k++) {
                            data[k].i = rowidx ? rowidx[i]+1 : i+1;
                            data[k].j = colidx ? colidx[j]+1 : j+1;
                        }
                    }
                } else if (num_rows == num_columns && symmetry == mtx_skew_symmetric) {
                    for (int64_t i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = i+1; j < A->num_columns; j++, k++) {
                            data[k].i = rowidx ? rowidx[i]+1 : i+1;
                            data[k].j = colidx ? colidx[j]+1 : j+1;
                        }
                    }
                } else { mtxfile_free(mtxfile); return MTX_ERR_INVALID_SYMMETRY; }
            } else { mtxfile_free(mtxfile); return MTX_ERR_INVALID_PRECISION; }
        } else if (field == mtx_field_integer) {
            if (precision == mtx_single) {
                struct mtxfile_matrix_coordinate_integer_single * data =
                    mtxfile->data.matrix_coordinate_integer_single;
                err = mtxvector_get_integer_single(
                    &A->a, A->size, sizeof(*data), &data[0].a);
                if (err) { mtxfile_free(mtxfile); return err; }
                if (symmetry == mtx_unsymmetric) {
                    for (int64_t i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = 0; j < A->num_columns; j++, k++) {
                            data[k].i = rowidx ? rowidx[i]+1 : i+1;
                            data[k].j = colidx ? colidx[j]+1 : j+1;
                        }
                    }
                } else if (num_rows == num_columns && (symmetry == mtx_symmetric || symmetry == mtx_hermitian)) {
                    for (int64_t i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = i; j < A->num_columns; j++, k++) {
                            data[k].i = rowidx ? rowidx[i]+1 : i+1;
                            data[k].j = colidx ? colidx[j]+1 : j+1;
                        }
                    }
                } else if (num_rows == num_columns && symmetry == mtx_skew_symmetric) {
                    for (int64_t i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = i+1; j < A->num_columns; j++, k++) {
                            data[k].i = rowidx ? rowidx[i]+1 : i+1;
                            data[k].j = colidx ? colidx[j]+1 : j+1;
                        }
                    }
                } else { mtxfile_free(mtxfile); return MTX_ERR_INVALID_SYMMETRY; }
            } else if (precision == mtx_double) {
                struct mtxfile_matrix_coordinate_integer_double * data =
                    mtxfile->data.matrix_coordinate_integer_double;
                err = mtxvector_get_integer_double(
                    &A->a, A->size, sizeof(*data), &data[0].a);
                if (err) { mtxfile_free(mtxfile); return err; }
                if (symmetry == mtx_unsymmetric) {
                    for (int64_t i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = 0; j < A->num_columns; j++, k++) {
                            data[k].i = rowidx ? rowidx[i]+1 : i+1;
                            data[k].j = colidx ? colidx[j]+1 : j+1;
                        }
                    }
                } else if (num_rows == num_columns && (symmetry == mtx_symmetric || symmetry == mtx_hermitian)) {
                    for (int64_t i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = i; j < A->num_columns; j++, k++) {
                            data[k].i = rowidx ? rowidx[i]+1 : i+1;
                            data[k].j = colidx ? colidx[j]+1 : j+1;
                        }
                    }
                } else if (num_rows == num_columns && symmetry == mtx_skew_symmetric) {
                    for (int64_t i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = i+1; j < A->num_columns; j++, k++) {
                            data[k].i = rowidx ? rowidx[i]+1 : i+1;
                            data[k].j = colidx ? colidx[j]+1 : j+1;
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
                err = mtxvector_get_real_single(&A->a, A->size, sizeof(*data), data);
                if (err) { mtxfile_free(mtxfile); return err; }
            } else if (precision == mtx_double) {
                double * data = mtxfile->data.array_real_double;
                err = mtxvector_get_real_double(&A->a, A->size, sizeof(*data), data);
            } else { mtxfile_free(mtxfile); return MTX_ERR_INVALID_PRECISION; }
        } else if (field == mtx_field_complex) {
            if (precision == mtx_single) {
                float (* data)[2] = mtxfile->data.array_complex_single;
                err = mtxvector_get_complex_single(&A->a, A->size, sizeof(*data), data);
            } else if (precision == mtx_double) {
                double (* data)[2] = mtxfile->data.array_complex_double;
                err = mtxvector_get_complex_double(&A->a, A->size, sizeof(*data), data);
            } else { mtxfile_free(mtxfile); return MTX_ERR_INVALID_PRECISION; }
        } else if (field == mtx_field_integer) {
            if (precision == mtx_single) {
                int32_t * data = mtxfile->data.array_integer_single;
                err = mtxvector_get_integer_single(&A->a, A->size, sizeof(*data), data);
            } else if (precision == mtx_double) {
                int64_t * data = mtxfile->data.array_integer_double;
                err = mtxvector_get_integer_double(&A->a, A->size, sizeof(*data), data);
            } else { mtxfile_free(mtxfile); return MTX_ERR_INVALID_PRECISION; }
        } else { mtxfile_free(mtxfile); return MTX_ERR_INVALID_FIELD; }
    } else { return MTX_ERR_INVALID_MTX_FORMAT; }
    return MTX_SUCCESS;
}

/*
 * Level 1 BLAS operations
 */

/**
 * ‘mtxmatrix_blas_swap()’ swaps values of two matrices,
 * simultaneously performing ‘y <- x’ and ‘x <- y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_blas_swap(
    struct mtxmatrix_blas * x,
    struct mtxmatrix_blas * y)
{
    return mtxvector_swap(&x->a, &y->a);
}

/**
 * ‘mtxmatrix_blas_copy()’ copies values of a matrix, ‘y = x’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_blas_copy(
    struct mtxmatrix_blas * y,
    const struct mtxmatrix_blas * x)
{
    return mtxvector_copy(&y->a, &x->a);
}

/**
 * ‘mtxmatrix_blas_sscal()’ scales a matrix by a single precision
 * floating point scalar, ‘x = a*x’.
 */
int mtxmatrix_blas_sscal(
    float a,
    struct mtxmatrix_blas * x,
    int64_t * num_flops)
{
    return mtxvector_sscal(a, &x->a, num_flops);
}

/**
 * ‘mtxmatrix_blas_dscal()’ scales a matrix by a double precision
 * floating point scalar, ‘x = a*x’.
 */
int mtxmatrix_blas_dscal(
    double a,
    struct mtxmatrix_blas * x,
    int64_t * num_flops)
{
    return mtxvector_dscal(a, &x->a, num_flops);
}

/**
 * ‘mtxmatrix_blas_cscal()’ scales a matrix by a complex, single
 * precision floating point scalar, ‘x = (a+b*i)*x’.
 */
int mtxmatrix_blas_cscal(
    float a[2],
    struct mtxmatrix_blas * x,
    int64_t * num_flops)
{
    return mtxvector_cscal(a, &x->a, num_flops);
}

/**
 * ‘mtxmatrix_blas_zscal()’ scales a matrix by a complex, double
 * precision floating point scalar, ‘x = (a+b*i)*x’.
 */
int mtxmatrix_blas_zscal(
    double a[2],
    struct mtxmatrix_blas * x,
    int64_t * num_flops)
{
    return mtxvector_zscal(a, &x->a, num_flops);
}

/**
 * ‘mtxmatrix_blas_saxpy()’ adds a matrix to another one multiplied
 * by a single precision floating point value, ‘y = a*x + y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_blas_saxpy(
    float a,
    const struct mtxmatrix_blas * x,
    struct mtxmatrix_blas * y,
    int64_t * num_flops)
{
    return mtxvector_saxpy(a, &x->a, &y->a, num_flops);
}

/**
 * ‘mtxmatrix_blas_daxpy()’ adds a matrix to another one multiplied
 * by a double precision floating point value, ‘y = a*x + y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_blas_daxpy(
    double a,
    const struct mtxmatrix_blas * x,
    struct mtxmatrix_blas * y,
    int64_t * num_flops)
{
    return mtxvector_daxpy(a, &x->a, &y->a, num_flops);
}

/**
 * ‘mtxmatrix_blas_saypx()’ multiplies a matrix by a single precision
 * floating point scalar and adds another matrix, ‘y = a*y + x’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_blas_saypx(
    float a,
    struct mtxmatrix_blas * y,
    const struct mtxmatrix_blas * x,
    int64_t * num_flops)
{
    return mtxvector_saypx(a, &y->a, &x->a, num_flops);
}

/**
 * ‘mtxmatrix_blas_daypx()’ multiplies a matrix by a double precision
 * floating point scalar and adds another matrix, ‘y = a*y + x’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_blas_daypx(
    double a,
    struct mtxmatrix_blas * y,
    const struct mtxmatrix_blas * x,
    int64_t * num_flops)
{
    return mtxvector_daypx(a, &y->a, &x->a, num_flops);
}

/**
 * ‘mtxmatrix_blas_sdot()’ computes the Frobenius inner product of
 * two matrices in single precision floating point.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_blas_sdot(
    const struct mtxmatrix_blas * x,
    const struct mtxmatrix_blas * y,
    float * dot,
    int64_t * num_flops)
{
    return mtxvector_sdot(&x->a, &y->a, dot, num_flops);
}

/**
 * ‘mtxmatrix_blas_ddot()’ computes the Frobenius inner product of
 * two matrices in double precision floating point.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_blas_ddot(
    const struct mtxmatrix_blas * x,
    const struct mtxmatrix_blas * y,
    double * dot,
    int64_t * num_flops)
{
    return mtxvector_ddot(&x->a, &y->a, dot, num_flops);
}

/**
 * ‘mtxmatrix_blas_cdotu()’ computes the product of the transpose of
 * a complex row matrix with another complex row matrix in single
 * precision floating point, ‘dot := x^T*y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_blas_cdotu(
    const struct mtxmatrix_blas * x,
    const struct mtxmatrix_blas * y,
    float (* dot)[2],
    int64_t * num_flops)
{
    return mtxvector_cdotu(&x->a, &y->a, dot, num_flops);
}

/**
 * ‘mtxmatrix_blas_zdotu()’ computes the product of the transpose of
 * a complex row matrix with another complex row matrix in double
 * precision floating point, ‘dot := x^T*y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_blas_zdotu(
    const struct mtxmatrix_blas * x,
    const struct mtxmatrix_blas * y,
    double (* dot)[2],
    int64_t * num_flops)
{
    return mtxvector_zdotu(&x->a, &y->a, dot, num_flops);
}

/**
 * ‘mtxmatrix_blas_cdotc()’ computes the Frobenius inner product of
 * two complex matrices in single precision floating point, ‘dot :=
 * x^H*y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_blas_cdotc(
    const struct mtxmatrix_blas * x,
    const struct mtxmatrix_blas * y,
    float (* dot)[2],
    int64_t * num_flops)
{
    return mtxvector_cdotc(&x->a, &y->a, dot, num_flops);
}

/**
 * ‘mtxmatrix_blas_zdotc()’ computes the Frobenius inner product of
 * two complex matrices in double precision floating point, ‘dot :=
 * x^H*y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_blas_zdotc(
    const struct mtxmatrix_blas * x,
    const struct mtxmatrix_blas * y,
    double (* dot)[2],
    int64_t * num_flops)
{
    return mtxvector_zdotc(&x->a, &y->a, dot, num_flops);
}

/**
 * ‘mtxmatrix_blas_snrm2()’ computes the Frobenius norm of a matrix
 * in single precision floating point.
 */
int mtxmatrix_blas_snrm2(
    const struct mtxmatrix_blas * x,
    float * nrm2,
    int64_t * num_flops)
{
    return mtxvector_snrm2(&x->a, nrm2, num_flops);
}

/**
 * ‘mtxmatrix_blas_dnrm2()’ computes the Frobenius norm of a matrix
 * in double precision floating point.
 */
int mtxmatrix_blas_dnrm2(
    const struct mtxmatrix_blas * x,
    double * nrm2,
    int64_t * num_flops)
{
    return mtxvector_dnrm2(&x->a, nrm2, num_flops);
}

/**
 * ‘mtxmatrix_blas_sasum()’ computes the sum of absolute values
 * (1-norm) of a matrix in single precision floating point.  If the
 * matrix is complex-valued, then the sum of the absolute values of
 * the real and imaginary parts is computed.
 */
int mtxmatrix_blas_sasum(
    const struct mtxmatrix_blas * x,
    float * asum,
    int64_t * num_flops)
{
    return mtxvector_sasum(&x->a, asum, num_flops);
}

/**
 * ‘mtxmatrix_blas_dasum()’ computes the sum of absolute values
 * (1-norm) of a matrix in double precision floating point.  If the
 * matrix is complex-valued, then the sum of the absolute values of
 * the real and imaginary parts is computed.
 */
int mtxmatrix_blas_dasum(
    const struct mtxmatrix_blas * x,
    double * asum,
    int64_t * num_flops)
{
    return mtxvector_dasum(&x->a, asum, num_flops);
}

/**
 * ‘mtxmatrix_blas_iamax()’ finds the index of the first element
 * having the maximum absolute value.  If the matrix is
 * complex-valued, then the index points to the first element having
 * the maximum sum of the absolute values of the real and imaginary
 * parts.
 */
int mtxmatrix_blas_iamax(
    const struct mtxmatrix_blas * x,
    int * iamax)
{
    return mtxvector_iamax(&x->a, iamax);
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
#endif

/**
 * ‘mtxmatrix_dense_sgemv_base()’ multiplies a matrix ‘A’ or its
 * transpose ‘A'’ by a real scalar ‘alpha’ (‘α’) and a vector ‘x’,
 * before adding the result to another vector ‘y’ multiplied by
 * another real scalar ‘beta’ (‘β’). That is, ‘y = α*A*x + β*y’ or ‘y
 * = α*A'*x + β*y’.
 *
 * The scalars ‘alpha’ and ‘beta’ are given as single precision
 * floating point numbers.
 *
 * The matrix ‘A’ and vectors ‘x’ and ‘y’ must have the same field and
 * precision.
 *
 * If ‘trans’ is ‘mtx_notrans’, then the size of ‘x’ must equal the
 * number of columns of ‘A’ and the size of ‘y’ must equal the number
 * of rows of ‘A’. if ‘trans’ is ‘mtx_trans’ or ‘mtx_conjtrans’, then
 * the size of ‘x’ must equal the number of rows of ‘A’ and the size
 * of ‘y’ must equal the number of columns of ‘A’.
 */
int mtxmatrix_dense_sgemv_base(
    enum mtxtransposition trans,
    float alpha,
    const struct mtxmatrix_blas * A,
    const struct mtxvector_base * x,
    float beta,
    struct mtxvector_base * y,
    int64_t * num_flops)
{
    if (A->a.type != mtxvector_base) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
    const struct mtxvector_base * a = &A->a.storage.base;
    if (x->field != a->field || y->field != a->field)
        return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != a->precision || y->precision != a->precision)
        return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (trans == mtx_notrans &&
        (A->num_rows != y->size || A->num_columns != x->size))
        return MTX_ERR_INCOMPATIBLE_SIZE;
    if ((trans == mtx_trans || trans == mtx_conjtrans) &&
        (A->num_columns != y->size || A->num_rows != x->size))
        return MTX_ERR_INCOMPATIBLE_SIZE;

    if (beta != 1) {
        int err = mtxvector_base_sscal(beta, y, num_flops);
        if (err) return err;
    }

    if (A->symmetry == mtx_unsymmetric) {
        if (a->field == mtx_field_real) {
            if (trans == mtx_notrans) {
                if (a->precision == mtx_single) {
                    const float * Adata = a->data.real_single;
                    const float * xdata = x->data.real_single;
                    float * ydata = y->data.real_single;
                    for (int i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = 0; j < A->num_columns; j++, k++)
                            ydata[i] += alpha*Adata[k]*xdata[j];
                    }
                    if (num_flops) *num_flops += 3*A->num_entries;
                } else if (a->precision == mtx_double) {
                    const double * Adata = a->data.real_double;
                    const double * xdata = x->data.real_double;
                    double * ydata = y->data.real_double;
                    for (int i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = 0; j < A->num_columns; j++, k++)
                            ydata[i] += alpha*Adata[k]*xdata[j];
                    }
                    if (num_flops) *num_flops += 3*A->num_entries;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (trans == mtx_trans || trans == mtx_conjtrans) {
                if (a->precision == mtx_single) {
                    const float * Adata = a->data.real_single;
                    const float * xdata = x->data.real_single;
                    float * ydata = y->data.real_single;
                    for (int i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = 0; j < A->num_columns; j++, k++)
                            ydata[j] += alpha*Adata[k]*xdata[i];
                    }
                    if (num_flops) *num_flops += 3*A->num_entries;
                } else if (a->precision == mtx_double) {
                    const double * Adata = a->data.real_double;
                    const double * xdata = x->data.real_double;
                    double * ydata = y->data.real_double;
                    for (int i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = 0; j < A->num_columns; j++, k++)
                            ydata[j] += alpha*Adata[k]*xdata[i];
                    }
                    if (num_flops) *num_flops += 3*A->num_entries;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else { return MTX_ERR_INVALID_TRANSPOSITION; }
        } else if (a->field == mtx_field_complex) {
            if (trans == mtx_notrans) {
                if (a->precision == mtx_single) {
                    const float (* Adata)[2] = a->data.complex_single;
                    const float (* xdata)[2] = x->data.complex_single;
                    float (* ydata)[2] = y->data.complex_single;
                    for (int i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = 0; j < A->num_columns; j++, k++) {
                            ydata[i][0] += alpha*(Adata[k][0]*xdata[j][0]-Adata[k][1]*xdata[j][1]);
                            ydata[i][1] += alpha*(Adata[k][0]*xdata[j][1]+Adata[k][1]*xdata[j][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_entries;
                } else if (a->precision == mtx_double) {
                    const double (* Adata)[2] = a->data.complex_double;
                    const double (* xdata)[2] = x->data.complex_double;
                    double (* ydata)[2] = y->data.complex_double;
                    for (int i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = 0; j < A->num_columns; j++, k++) {
                            ydata[i][0] += alpha*(Adata[k][0]*xdata[j][0]-Adata[k][1]*xdata[j][1]);
                            ydata[i][1] += alpha*(Adata[k][0]*xdata[j][1]+Adata[k][1]*xdata[j][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_entries;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (trans == mtx_trans) {
                if (a->precision == mtx_single) {
                    const float (* Adata)[2] = a->data.complex_single;
                    const float (* xdata)[2] = x->data.complex_single;
                    float (* ydata)[2] = y->data.complex_single;
                    for (int i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = 0; j < A->num_columns; j++, k++) {
                            ydata[j][0] += alpha*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                            ydata[j][1] += alpha*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_entries;
                } else if (a->precision == mtx_double) {
                    const double (* Adata)[2] = a->data.complex_double;
                    const double (* xdata)[2] = x->data.complex_double;
                    double (* ydata)[2] = y->data.complex_double;
                    for (int i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = 0; j < A->num_columns; j++, k++) {
                            ydata[j][0] += alpha*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                            ydata[j][1] += alpha*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_entries;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (trans == mtx_conjtrans) {
                if (a->precision == mtx_single) {
                    const float (* Adata)[2] = a->data.complex_single;
                    const float (* xdata)[2] = x->data.complex_single;
                    float (* ydata)[2] = y->data.complex_single;
                    for (int i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = 0; j < A->num_columns; j++, k++) {
                            ydata[j][0] += alpha*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                            ydata[j][1] += alpha*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_entries;
                } else if (a->precision == mtx_double) {
                    const double (* Adata)[2] = a->data.complex_double;
                    const double (* xdata)[2] = x->data.complex_double;
                    double (* ydata)[2] = y->data.complex_double;
                    for (int i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = 0; j < A->num_columns; j++, k++) {
                            ydata[j][0] += alpha*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                            ydata[j][1] += alpha*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_entries;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else { return MTX_ERR_INVALID_TRANSPOSITION; }
        } else if (a->field == mtx_field_integer) {
            if (trans == mtx_notrans) {
                if (a->precision == mtx_single) {
                    const int32_t * Adata = a->data.integer_single;
                    const int32_t * xdata = x->data.integer_single;
                    int32_t * ydata = y->data.integer_single;
                    for (int i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = 0; j < A->num_columns; j++, k++)
                            ydata[i] += alpha*Adata[k]*xdata[j];
                    }
                    if (num_flops) *num_flops += 3*A->num_entries;
                } else if (a->precision == mtx_double) {
                    const int64_t * Adata = a->data.integer_double;
                    const int64_t * xdata = x->data.integer_double;
                    int64_t * ydata = y->data.integer_double;
                    for (int i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = 0; j < A->num_columns; j++, k++)
                            ydata[i] += alpha*Adata[k]*xdata[j];
                    }
                    if (num_flops) *num_flops += 3*A->num_entries;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (trans == mtx_trans || trans == mtx_conjtrans) {
                if (a->precision == mtx_single) {
                    const int32_t * Adata = a->data.integer_single;
                    const int32_t * xdata = x->data.integer_single;
                    int32_t * ydata = y->data.integer_single;
                    for (int i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = 0; j < A->num_columns; j++, k++)
                            ydata[j] += alpha*Adata[k]*xdata[i];
                    }
                    if (num_flops) *num_flops += 3*A->num_entries;
                } else if (a->precision == mtx_double) {
                    const int64_t * Adata = a->data.integer_double;
                    const int64_t * xdata = x->data.integer_double;
                    int64_t * ydata = y->data.integer_double;
                    for (int i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = 0; j < A->num_columns; j++, k++)
                            ydata[j] += alpha*Adata[k]*xdata[i];
                    }
                    if (num_flops) *num_flops += 3*A->num_entries;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else { return MTX_ERR_INVALID_TRANSPOSITION; }
        } else { return MTX_ERR_INVALID_FIELD; }
    } else if (A->num_rows == A->num_columns && A->symmetry == mtx_symmetric) {
        if (a->field == mtx_field_real) {
            if (a->precision == mtx_single) {
                const float * Adata = a->data.real_single;
                const float * xdata = x->data.real_single;
                float * ydata = y->data.real_single;
                for (int i = 0, k = 0; i < A->num_rows; i++) {
                    ydata[i] += alpha*Adata[k++]*xdata[i];
                    for (int j = i+1; j < A->num_columns; j++, k++) {
                        ydata[i] += alpha*Adata[k]*xdata[j];
                        ydata[j] += alpha*Adata[k]*xdata[i];
                    }
                }
                if (num_flops) *num_flops += 3*A->num_entries;
            } else if (a->precision == mtx_double) {
                const double * Adata = a->data.real_double;
                const double * xdata = x->data.real_double;
                double * ydata = y->data.real_double;
                for (int i = 0, k = 0; i < A->num_rows; i++) {
                    ydata[i] += alpha*Adata[k++]*xdata[i];
                    for (int j = i+1; j < A->num_columns; j++, k++) {
                        ydata[i] += alpha*Adata[k]*xdata[j];
                        ydata[j] += alpha*Adata[k]*xdata[i];
                    }
                }
                if (num_flops) *num_flops += 3*A->num_entries;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (a->field == mtx_field_complex) {
            if (trans == mtx_notrans || trans == mtx_trans) {
                if (a->precision == mtx_single) {
                    const float (* Adata)[2] = a->data.complex_single;
                    const float (* xdata)[2] = x->data.complex_single;
                    float (* ydata)[2] = y->data.complex_single;
                    for (int i = 0, k = 0; i < A->num_rows; i++) {
                        ydata[i][0] += alpha*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                        ydata[i][1] += alpha*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        k++;
                        for (int j = i+1; j < A->num_columns; j++, k++) {
                            ydata[i][0] += alpha*(Adata[k][0]*xdata[j][0]-Adata[k][1]*xdata[j][1]);
                            ydata[i][1] += alpha*(Adata[k][0]*xdata[j][1]+Adata[k][1]*xdata[j][0]);
                            ydata[j][0] += alpha*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                            ydata[j][1] += alpha*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_entries;
                } else if (a->precision == mtx_double) {
                    const double (* Adata)[2] = a->data.complex_double;
                    const double (* xdata)[2] = x->data.complex_double;
                    double (* ydata)[2] = y->data.complex_double;
                    for (int i = 0, k = 0; i < A->num_rows; i++) {
                        ydata[i][0] += alpha*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                        ydata[i][1] += alpha*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        k++;
                        for (int j = i+1; j < A->num_columns; j++, k++) {
                            ydata[i][0] += alpha*(Adata[k][0]*xdata[j][0]-Adata[k][1]*xdata[j][1]);
                            ydata[i][1] += alpha*(Adata[k][0]*xdata[j][1]+Adata[k][1]*xdata[j][0]);
                            ydata[j][0] += alpha*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                            ydata[j][1] += alpha*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_entries;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (trans == mtx_conjtrans) {
                if (a->precision == mtx_single) {
                    const float (* Adata)[2] = a->data.complex_single;
                    const float (* xdata)[2] = x->data.complex_single;
                    float (* ydata)[2] = y->data.complex_single;
                    for (int i = 0, k = 0; i < A->num_rows; i++) {
                        ydata[i][0] += alpha*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                        ydata[i][1] += alpha*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        k++;
                        for (int j = i+1; j < A->num_columns; j++, k++) {
                            ydata[i][0] += alpha*(Adata[k][0]*xdata[j][0]+Adata[k][1]*xdata[j][1]);
                            ydata[i][1] += alpha*(Adata[k][0]*xdata[j][1]-Adata[k][1]*xdata[j][0]);
                            ydata[j][0] += alpha*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                            ydata[j][1] += alpha*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_entries;
                } else if (a->precision == mtx_double) {
                    const double (* Adata)[2] = a->data.complex_double;
                    const double (* xdata)[2] = x->data.complex_double;
                    double (* ydata)[2] = y->data.complex_double;
                    for (int i = 0, k = 0; i < A->num_rows; i++) {
                        ydata[i][0] += alpha*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                        ydata[i][1] += alpha*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        k++;
                        for (int j = i+1; j < A->num_columns; j++, k++) {
                            ydata[i][0] += alpha*(Adata[k][0]*xdata[j][0]+Adata[k][1]*xdata[j][1]);
                            ydata[i][1] += alpha*(Adata[k][0]*xdata[j][1]-Adata[k][1]*xdata[j][0]);
                            ydata[j][0] += alpha*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                            ydata[j][1] += alpha*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_entries;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else { return MTX_ERR_INVALID_TRANSPOSITION; }
        } else if (a->field == mtx_field_integer) {
            if (a->precision == mtx_single) {
                const int32_t * Adata = a->data.integer_single;
                const int32_t * xdata = x->data.integer_single;
                int32_t * ydata = y->data.integer_single;
                for (int i = 0, k = 0; i < A->num_rows; i++) {
                    ydata[i] += alpha*Adata[k++]*xdata[i];
                    for (int j = i+1; j < A->num_columns; j++, k++) {
                        ydata[i] += alpha*Adata[k]*xdata[j];
                        ydata[j] += alpha*Adata[k]*xdata[i];
                    }
                }
                if (num_flops) *num_flops += 3*A->num_entries;
            } else if (a->precision == mtx_double) {
                const int64_t * Adata = a->data.integer_double;
                const int64_t * xdata = x->data.integer_double;
                int64_t * ydata = y->data.integer_double;
                for (int i = 0, k = 0; i < A->num_rows; i++) {
                    ydata[i] += alpha*Adata[k++]*xdata[i];
                    for (int j = i+1; j < A->num_columns; j++, k++) {
                        ydata[i] += alpha*Adata[k]*xdata[j];
                        ydata[j] += alpha*Adata[k]*xdata[i];
                    }
                }
                if (num_flops) *num_flops += 3*A->num_entries;
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
                    const float (* xdata)[2] = x->data.complex_single;
                    float (* ydata)[2] = y->data.complex_single;
                    for (int i = 0, k = 0; i < A->num_rows; i++) {
                        ydata[i][0] += alpha*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                        ydata[i][1] += alpha*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        k++;
                        for (int j = i+1; j < A->num_columns; j++, k++) {
                            ydata[i][0] += alpha*(Adata[k][0]*xdata[j][0]-Adata[k][1]*xdata[j][1]);
                            ydata[i][1] += alpha*(Adata[k][0]*xdata[j][1]+Adata[k][1]*xdata[j][0]);
                            ydata[j][0] += alpha*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                            ydata[j][1] += alpha*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_entries;
                } else if (a->precision == mtx_double) {
                    const double (* Adata)[2] = a->data.complex_double;
                    const double (* xdata)[2] = x->data.complex_double;
                    double (* ydata)[2] = y->data.complex_double;
                    for (int i = 0, k = 0; i < A->num_rows; i++) {
                        ydata[i][0] += alpha*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                        ydata[i][1] += alpha*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        k++;
                        for (int j = i+1; j < A->num_columns; j++, k++) {
                            ydata[i][0] += alpha*(Adata[k][0]*xdata[j][0]-Adata[k][1]*xdata[j][1]);
                            ydata[i][1] += alpha*(Adata[k][0]*xdata[j][1]+Adata[k][1]*xdata[j][0]);
                            ydata[j][0] += alpha*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                            ydata[j][1] += alpha*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_entries;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (trans == mtx_trans) {
                if (a->precision == mtx_single) {
                    const float (* Adata)[2] = a->data.complex_single;
                    const float (* xdata)[2] = x->data.complex_single;
                    float (* ydata)[2] = y->data.complex_single;
                    for (int i = 0, k = 0; i < A->num_rows; i++) {
                        ydata[i][0] += alpha*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                        ydata[i][1] += alpha*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        k++;
                        for (int j = i+1; j < A->num_columns; j++, k++) {
                            ydata[i][0] += alpha*(Adata[k][0]*xdata[j][0]+Adata[k][1]*xdata[j][1]);
                            ydata[i][1] += alpha*(Adata[k][0]*xdata[j][1]-Adata[k][1]*xdata[j][0]);
                            ydata[j][0] += alpha*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                            ydata[j][1] += alpha*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_entries;
                } else if (a->precision == mtx_double) {
                    const double (* Adata)[2] = a->data.complex_double;
                    const double (* xdata)[2] = x->data.complex_double;
                    double (* ydata)[2] = y->data.complex_double;
                    for (int i = 0, k = 0; i < A->num_rows; i++) {
                        ydata[i][0] += alpha*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                        ydata[i][1] += alpha*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        k++;
                        for (int j = i+1; j < A->num_columns; j++, k++) {
                            ydata[i][0] += alpha*(Adata[k][0]*xdata[j][0]+Adata[k][1]*xdata[j][1]);
                            ydata[i][1] += alpha*(Adata[k][0]*xdata[j][1]-Adata[k][1]*xdata[j][0]);
                            ydata[j][0] += alpha*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                            ydata[j][1] += alpha*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_entries;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else { return MTX_ERR_INVALID_TRANSPOSITION; }
        } else { return MTX_ERR_INVALID_FIELD; }
    } else { return MTX_ERR_INVALID_SYMMETRY; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_dense_sgemv_blas()’ multiplies a matrix ‘A’ or its
 * transpose ‘A'’ by a real scalar ‘alpha’ (‘α’) and a vector ‘x’,
 * before adding the result to another vector ‘y’ multiplied by
 * another real scalar ‘beta’ (‘β’). That is, ‘y = α*A*x + β*y’ or ‘y
 * = α*A'*x + β*y’.
 *
 * The scalars ‘alpha’ and ‘beta’ are given as single precision
 * floating point numbers.
 *
 * The matrix ‘A’ and vectors ‘x’ and ‘y’ must have the same field and
 * precision.
 *
 * If ‘trans’ is ‘mtx_notrans’, then the size of ‘x’ must equal the
 * number of columns of ‘A’ and the size of ‘y’ must equal the number
 * of rows of ‘A’. if ‘trans’ is ‘mtx_trans’ or ‘mtx_conjtrans’, then
 * the size of ‘x’ must equal the number of rows of ‘A’ and the size
 * of ‘y’ must equal the number of columns of ‘A’.
 */
int mtxmatrix_dense_sgemv_blas(
    enum mtxtransposition trans,
    float alpha,
    const struct mtxmatrix_blas * A,
    const struct mtxvector_blas * x,
    float beta,
    struct mtxvector_blas * y,
    int64_t * num_flops)
{
#ifdef LIBMTX_HAVE_BLAS
    if (A->a.type != mtxvector_blas) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
    const struct mtxvector_blas * a = &A->a.storage.blas;
    if (x->base.field != a->base.field || y->base.field != a->base.field)
        return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->base.precision != a->base.precision || y->base.precision != a->base.precision)
        return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (trans == mtx_notrans &&
        (A->num_rows != y->base.size || A->num_columns != x->base.size))
        return MTX_ERR_INCOMPATIBLE_SIZE;
    if ((trans == mtx_trans || trans == mtx_conjtrans) &&
        (A->num_columns != y->base.size || A->num_rows != x->base.size))
        return MTX_ERR_INCOMPATIBLE_SIZE;

    if (A->symmetry == mtx_unsymmetric) {
        if (a->base.field == mtx_field_real) {
            if (trans == mtx_notrans) {
                if (a->base.precision == mtx_single) {
                    const float * Adata = a->base.data.real_single;
                    const float * xdata = x->base.data.real_single;
                    float * ydata = y->base.data.real_single;
                    cblas_sgemv(
                        CblasRowMajor, CblasNoTrans, A->num_rows, A->num_columns,
                        alpha, Adata, A->num_columns, xdata, 1, beta, ydata, 1);
                    if (mtxblaserror()) return MTX_ERR_BLAS;
                    if (num_flops) *num_flops += cblas_sgemv_num_flops(
                        A->num_rows, A->num_columns, alpha, 1);
                } else if (a->base.precision == mtx_double) {
                    const double * Adata = a->base.data.real_double;
                    const double * xdata = x->base.data.real_double;
                    double * ydata = y->base.data.real_double;
                    cblas_dgemv(
                        CblasRowMajor, CblasNoTrans, A->num_rows, A->num_columns,
                        alpha, Adata, A->num_columns, xdata, 1, beta, ydata, 1);
                    if (mtxblaserror()) return MTX_ERR_BLAS;
                    if (num_flops) *num_flops += cblas_dgemv_num_flops(
                        A->num_rows, A->num_columns, alpha, 1);
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (trans == mtx_trans || trans == mtx_conjtrans) {
                if (a->base.precision == mtx_single) {
                    const float * Adata = a->base.data.real_single;
                    const float * xdata = x->base.data.real_single;
                    float * ydata = y->base.data.real_single;
                    cblas_sgemv(
                        CblasRowMajor, CblasTrans, A->num_rows, A->num_columns,
                        alpha, Adata, A->num_columns, xdata, 1, beta, ydata, 1);
                    if (mtxblaserror()) return MTX_ERR_BLAS;
                    if (num_flops) *num_flops += cblas_sgemv_num_flops(
                        A->num_columns, A->num_rows, alpha, beta);
                } else if (a->base.precision == mtx_double) {
                    const double * Adata = a->base.data.real_double;
                    const double * xdata = x->base.data.real_double;
                    double * ydata = y->base.data.real_double;
                    cblas_dgemv(
                        CblasRowMajor, CblasTrans, A->num_rows, A->num_columns,
                        alpha, Adata, A->num_columns, xdata, 1, beta, ydata, 1);
                    if (mtxblaserror()) return MTX_ERR_BLAS;
                    if (num_flops) *num_flops += cblas_dgemv_num_flops(
                        A->num_columns, A->num_rows, alpha, beta);
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else { return MTX_ERR_INVALID_TRANSPOSITION; }
        } else if (a->base.field == mtx_field_complex) {
            if (trans == mtx_notrans) {
                if (a->base.precision == mtx_single) {
                    const float (* Adata)[2] = a->base.data.complex_single;
                    const float (* xdata)[2] = x->base.data.complex_single;
                    float (* ydata)[2] = y->base.data.complex_single;
                    const float calpha[2] = {alpha, 0};
                    const float cbeta[2] = {beta, 0};
                    cblas_cgemv(
                        CblasRowMajor, CblasNoTrans, A->num_rows, A->num_columns,
                        calpha, (const float *) Adata, A->num_columns,
                        (const float *) xdata, 1, cbeta, (float *) ydata, 1);
                    if (mtxblaserror()) return MTX_ERR_BLAS;
                    if (num_flops) *num_flops += cblas_cgemv_num_flops(
                        A->num_rows, A->num_columns, calpha, cbeta);
                } else if (a->base.precision == mtx_double) {
                    const double (* Adata)[2] = a->base.data.complex_double;
                    const double (* xdata)[2] = x->base.data.complex_double;
                    double (* ydata)[2] = y->base.data.complex_double;
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
                if (a->base.precision == mtx_single) {
                    const float (* Adata)[2] = a->base.data.complex_single;
                    const float (* xdata)[2] = x->base.data.complex_single;
                    float (* ydata)[2] = y->base.data.complex_single;
                    const float calpha[2] = {alpha, 0};
                    const float cbeta[2] = {beta, 0};
                    cblas_cgemv(
                        CblasRowMajor, CblasTrans, A->num_rows, A->num_columns,
                        calpha, (const float *) Adata, A->num_columns,
                        (const float *) xdata, 1, cbeta, (float *) ydata, 1);
                    if (mtxblaserror()) return MTX_ERR_BLAS;
                    if (num_flops) *num_flops += cblas_cgemv_num_flops(
                        A->num_columns, A->num_rows, calpha, cbeta);
                } else if (a->base.precision == mtx_double) {
                    const double (* Adata)[2] = a->base.data.complex_double;
                    const double (* xdata)[2] = x->base.data.complex_double;
                    double (* ydata)[2] = y->base.data.complex_double;
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
                if (a->base.precision == mtx_single) {
                    const float (* Adata)[2] = a->base.data.complex_single;
                    const float (* xdata)[2] = x->base.data.complex_single;
                    float (* ydata)[2] = y->base.data.complex_single;
                    const float calpha[2] = {alpha, 0};
                    const float cbeta[2] = {beta, 0};
                    cblas_cgemv(
                        CblasRowMajor, CblasConjTrans, A->num_rows, A->num_columns,
                        calpha, (const float *) Adata, A->num_columns,
                        (const float *) xdata, 1, cbeta, (float *) ydata, 1);
                    if (mtxblaserror()) return MTX_ERR_BLAS;
                    if (num_flops) *num_flops += cblas_cgemv_num_flops(
                        A->num_columns, A->num_rows, calpha, cbeta);
                } else if (a->base.precision == mtx_double) {
                    const double (* Adata)[2] = a->base.data.complex_double;
                    const double (* xdata)[2] = x->base.data.complex_double;
                    double (* ydata)[2] = y->base.data.complex_double;
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
        if (a->base.field == mtx_field_real) {
            if (a->base.precision == mtx_single) {
                const float * Adata = a->base.data.real_single;
                const float * xdata = x->base.data.real_single;
                float * ydata = y->base.data.real_single;
                cblas_sspmv(
                    CblasRowMajor, CblasUpper, A->num_rows,
                    alpha, Adata, xdata, 1, beta, ydata, 1);
                if (mtxblaserror()) return MTX_ERR_BLAS;
                if (num_flops) *num_flops += cblas_sspmv_num_flops(
                    A->num_rows, alpha, beta);
            } else if (a->base.precision == mtx_double) {
                const double * Adata = a->base.data.real_double;
                const double * xdata = x->base.data.real_double;
                double * ydata = y->base.data.real_double;
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
        if (a->base.field == mtx_field_complex) {
            if (trans == mtx_notrans || trans == mtx_conjtrans) {
                if (a->base.precision == mtx_single) {
                    const float (* Adata)[2] = a->base.data.complex_single;
                    const float (* xdata)[2] = x->base.data.complex_single;
                    float (* ydata)[2] = y->base.data.complex_single;
                    const float calpha[2] = {alpha, 0};
                    const float cbeta[2] = {beta, 0};
                    cblas_chpmv(
                        CblasRowMajor, CblasUpper, A->num_rows, calpha,
                        (const float *) Adata, (const float *) xdata, 1,
                        cbeta, (float *) ydata, 1);
                    if (mtxblaserror()) return MTX_ERR_BLAS;
                    if (num_flops) *num_flops += cblas_chpmv_num_flops(
                        A->num_rows, calpha, cbeta);
                } else if (a->base.precision == mtx_double) {
                    const double (* Adata)[2] = a->base.data.complex_double;
                    const double (* xdata)[2] = x->base.data.complex_double;
                    double (* ydata)[2] = y->base.data.complex_double;
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
 * ‘mtxmatrix_dense_sgemv_omp()’ multiplies a matrix ‘A’ or its
 * transpose ‘A'’ by a real scalar ‘alpha’ (‘α’) and a vector ‘x’,
 * before adding the result to another vector ‘y’ multiplied by
 * another real scalar ‘beta’ (‘β’). That is, ‘y = α*A*x + β*y’ or ‘y
 * = α*A'*x + β*y’.
 *
 * The scalars ‘alpha’ and ‘beta’ are given as single precision
 * floating point numbers.
 *
 * The matrix ‘A’ and vectors ‘x’ and ‘y’ must have the same field and
 * precision.
 *
 * If ‘trans’ is ‘mtx_notrans’, then the size of ‘x’ must equal the
 * number of columns of ‘A’ and the size of ‘y’ must equal the number
 * of rows of ‘A’. if ‘trans’ is ‘mtx_trans’ or ‘mtx_conjtrans’, then
 * the size of ‘x’ must equal the number of rows of ‘A’ and the size
 * of ‘y’ must equal the number of columns of ‘A’.
 */
int mtxmatrix_dense_sgemv_omp(
    enum mtxtransposition trans,
    float alpha,
    const struct mtxmatrix_blas * A,
    const struct mtxvector_omp * x,
    float beta,
    struct mtxvector_omp * y,
    int64_t * num_flops)
{
    if (A->a.type != mtxvector_omp) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
    const struct mtxvector_omp * a = &A->a.storage.omp;
    enum mtxfield field = a->base.field;
    if (field != x->base.field || field != y->base.field)
        return MTX_ERR_INCOMPATIBLE_FIELD;
    enum mtxfield precision = a->base.precision;
    if (precision != x->base.precision || precision != y->base.precision)
        return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (trans == mtx_notrans &&
        (A->num_rows != y->base.size || A->num_columns != x->base.size))
        return MTX_ERR_INCOMPATIBLE_SIZE;
    if ((trans == mtx_trans || trans == mtx_conjtrans) &&
        (A->num_columns != y->base.size || A->num_rows != x->base.size))
        return MTX_ERR_INCOMPATIBLE_SIZE;
    int num_threads = a->num_threads;
    int N = A->num_columns;

    if (beta != 1) {
        int err = mtxvector_omp_sscal(beta, y, num_flops);
        if (err) return err;
    }

    if (A->symmetry == mtx_unsymmetric) {
        if (field == mtx_field_real) {
            if (trans == mtx_notrans) {
                if (precision == mtx_single) {
                    const float * Adata = a->base.data.real_single;
                    const float * xdata = x->base.data.real_single;
                    float * ydata = y->base.data.real_single;
                    #pragma omp parallel for num_threads(num_threads)
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int j = 0; j < A->num_columns; j++)
                            ydata[i] += alpha*Adata[i*N+j]*xdata[j];
                    }
                    if (num_flops) *num_flops += 3*A->num_entries;
                } else if (precision == mtx_double) {
                    const double * Adata = a->base.data.real_double;
                    const double * xdata = x->base.data.real_double;
                    double * ydata = y->base.data.real_double;
                    #pragma omp parallel for num_threads(num_threads)
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int j = 0; j < A->num_columns; j++)
                            ydata[i] += alpha*Adata[i*N+j]*xdata[j];
                    }
                    if (num_flops) *num_flops += 3*A->num_entries;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (trans == mtx_trans || trans == mtx_conjtrans) {
                if (precision == mtx_single) {
                    const float * Adata = a->base.data.real_single;
                    const float * xdata = x->base.data.real_single;
                    float * ydata = y->base.data.real_single;
                    #pragma omp parallel for num_threads(num_threads)
                    for (int j = 0; j < A->num_columns; j++) {
                        for (int i = 0; i < A->num_rows; i++)
                            ydata[j] += alpha*Adata[i*N+j]*xdata[i];
                    }
                    if (num_flops) *num_flops += 3*A->num_entries;
                } else if (precision == mtx_double) {
                    const double * Adata = a->base.data.real_double;
                    const double * xdata = x->base.data.real_double;
                    double * ydata = y->base.data.real_double;
                    #pragma omp parallel for num_threads(num_threads)
                    for (int j = 0; j < A->num_columns; j++) {
                        for (int i = 0; i < A->num_rows; i++)
                            ydata[j] += alpha*Adata[i*N+j]*xdata[i];
                    }
                    if (num_flops) *num_flops += 3*A->num_entries;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else { return MTX_ERR_INVALID_TRANSPOSITION; }
        } else if (field == mtx_field_complex) {
            if (trans == mtx_notrans) {
                if (precision == mtx_single) {
                    const float (* Adata)[2] = a->base.data.complex_single;
                    const float (* xdata)[2] = x->base.data.complex_single;
                    float (* ydata)[2] = y->base.data.complex_single;
                    #pragma omp parallel for num_threads(num_threads)
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int j = 0; j < A->num_columns; j++) {
                            ydata[i][0] += alpha*(Adata[i*N+j][0]*xdata[j][0]-Adata[i*N+j][1]*xdata[j][1]);
                            ydata[i][1] += alpha*(Adata[i*N+j][0]*xdata[j][1]+Adata[i*N+j][1]*xdata[j][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_entries;
                } else if (precision == mtx_double) {
                    const double (* Adata)[2] = a->base.data.complex_double;
                    const double (* xdata)[2] = x->base.data.complex_double;
                    double (* ydata)[2] = y->base.data.complex_double;
                    #pragma omp parallel for num_threads(num_threads)
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int j = 0; j < A->num_columns; j++) {
                            ydata[i][0] += alpha*(Adata[i*N+j][0]*xdata[j][0]-Adata[i*N+j][1]*xdata[j][1]);
                            ydata[i][1] += alpha*(Adata[i*N+j][0]*xdata[j][1]+Adata[i*N+j][1]*xdata[j][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_entries;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (trans == mtx_trans) {
                if (precision == mtx_single) {
                    const float (* Adata)[2] = a->base.data.complex_single;
                    const float (* xdata)[2] = x->base.data.complex_single;
                    float (* ydata)[2] = y->base.data.complex_single;
                    #pragma omp parallel for num_threads(num_threads)
                    for (int j = 0; j < A->num_columns; j++) {
                        for (int i = 0; i < A->num_rows; i++) {
                            ydata[j][0] += alpha*(Adata[i*N+j][0]*xdata[i][0]-Adata[i*N+j][1]*xdata[i][1]);
                            ydata[j][1] += alpha*(Adata[i*N+j][0]*xdata[i][1]+Adata[i*N+j][1]*xdata[i][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_entries;
                } else if (precision == mtx_double) {
                    const double (* Adata)[2] = a->base.data.complex_double;
                    const double (* xdata)[2] = x->base.data.complex_double;
                    double (* ydata)[2] = y->base.data.complex_double;
                    #pragma omp parallel for num_threads(num_threads)
                    for (int j = 0; j < A->num_columns; j++) {
                        for (int i = 0; i < A->num_rows; i++) {
                            ydata[j][0] += alpha*(Adata[i*N+j][0]*xdata[i][0]-Adata[i*N+j][1]*xdata[i][1]);
                            ydata[j][1] += alpha*(Adata[i*N+j][0]*xdata[i][1]+Adata[i*N+j][1]*xdata[i][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_entries;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (trans == mtx_conjtrans) {
                if (precision == mtx_single) {
                    const float (* Adata)[2] = a->base.data.complex_single;
                    const float (* xdata)[2] = x->base.data.complex_single;
                    float (* ydata)[2] = y->base.data.complex_single;
                    #pragma omp parallel for num_threads(num_threads)
                    for (int j = 0; j < A->num_columns; j++) {
                        for (int i = 0; i < A->num_rows; i++) {
                            ydata[j][0] += alpha*(Adata[i*N+j][0]*xdata[i][0]+Adata[i*N+j][1]*xdata[i][1]);
                            ydata[j][1] += alpha*(Adata[i*N+j][0]*xdata[i][1]-Adata[i*N+j][1]*xdata[i][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_entries;
                } else if (precision == mtx_double) {
                    const double (* Adata)[2] = a->base.data.complex_double;
                    const double (* xdata)[2] = x->base.data.complex_double;
                    double (* ydata)[2] = y->base.data.complex_double;
                    #pragma omp parallel for num_threads(num_threads)
                    for (int j = 0; j < A->num_columns; j++) {
                        for (int i = 0; i < A->num_rows; i++) {
                            ydata[j][0] += alpha*(Adata[i*N+j][0]*xdata[i][0]+Adata[i*N+j][1]*xdata[i][1]);
                            ydata[j][1] += alpha*(Adata[i*N+j][0]*xdata[i][1]-Adata[i*N+j][1]*xdata[i][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_entries;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else { return MTX_ERR_INVALID_TRANSPOSITION; }
        } else if (field == mtx_field_integer) {
            if (trans == mtx_notrans) {
                if (precision == mtx_single) {
                    const int32_t * Adata = a->base.data.integer_single;
                    const int32_t * xdata = x->base.data.integer_single;
                    int32_t * ydata = y->base.data.integer_single;
                    #pragma omp parallel for num_threads(num_threads)
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int j = 0; j < A->num_columns; j++)
                            ydata[i] += alpha*Adata[i*N+j]*xdata[j];
                    }
                    if (num_flops) *num_flops += 3*A->num_entries;
                } else if (precision == mtx_double) {
                    const int64_t * Adata = a->base.data.integer_double;
                    const int64_t * xdata = x->base.data.integer_double;
                    int64_t * ydata = y->base.data.integer_double;
                    #pragma omp parallel for num_threads(num_threads)
                    for (int i = 0; i < A->num_rows; i++) {
                        for (int j = 0; j < A->num_columns; j++)
                            ydata[i] += alpha*Adata[i*N+j]*xdata[j];
                    }
                    if (num_flops) *num_flops += 3*A->num_entries;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (trans == mtx_trans || trans == mtx_conjtrans) {
                if (precision == mtx_single) {
                    const int32_t * Adata = a->base.data.integer_single;
                    const int32_t * xdata = x->base.data.integer_single;
                    int32_t * ydata = y->base.data.integer_single;
                    #pragma omp parallel for num_threads(num_threads)
                    for (int j = 0; j < A->num_columns; j++) {
                        for (int i = 0; i < A->num_rows; i++)
                            ydata[j] += alpha*Adata[i*N+j]*xdata[i];
                    }
                    if (num_flops) *num_flops += 3*A->num_entries;
                } else if (precision == mtx_double) {
                    const int64_t * Adata = a->base.data.integer_double;
                    const int64_t * xdata = x->base.data.integer_double;
                    int64_t * ydata = y->base.data.integer_double;
                    #pragma omp parallel for num_threads(num_threads)
                    for (int j = 0; j < A->num_columns; j++) {
                        for (int i = 0; i < A->num_rows; i++)
                            ydata[j] += alpha*Adata[i*N+j]*xdata[i];
                    }
                    if (num_flops) *num_flops += 3*A->num_entries;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else { return MTX_ERR_INVALID_TRANSPOSITION; }
        } else { return MTX_ERR_INVALID_FIELD; }
    } else if (A->num_rows == A->num_columns && A->symmetry == mtx_symmetric) {
        return mtxmatrix_dense_sgemv_base(
            trans, alpha, A, &x->base, beta, &y->base, num_flops);
    } else if (A->num_rows == A->num_columns && A->symmetry == mtx_skew_symmetric) {
        return mtxmatrix_dense_sgemv_base(
            trans, alpha, A, &x->base, beta, &y->base, num_flops);
    } else if (A->num_rows == A->num_columns && A->symmetry == mtx_hermitian) {
        return mtxmatrix_dense_sgemv_base(
            trans, alpha, A, &x->base, beta, &y->base, num_flops);
    } else { return MTX_ERR_INVALID_SYMMETRY; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_blas_sgemv()’ multiplies a matrix ‘A’ or its transpose
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
int mtxmatrix_blas_sgemv(
    enum mtxtransposition trans,
    float alpha,
    const struct mtxmatrix_blas * A,
    const struct mtxvector * x,
    float beta,
    struct mtxvector * y,
    int64_t * num_flops)
{
    if (A->a.type != x->type || A->a.type != y->type)
        return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
    if (A->a.type == mtxvector_base) {
        return mtxmatrix_dense_sgemv_base(
            trans, alpha, A, &x->storage.base, beta, &y->storage.base, num_flops);
    } else if (A->a.type == mtxvector_blas) {
        return mtxmatrix_dense_sgemv_blas(
            trans, alpha, A, &x->storage.blas, beta, &y->storage.blas, num_flops);
    } else if (A->a.type == mtxvector_omp) {
        return mtxmatrix_dense_sgemv_omp(
            trans, alpha, A, &x->storage.omp, beta, &y->storage.omp, num_flops);
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/**
 * ‘mtxmatrix_blas_dgemv()’ multiplies a matrix ‘A’ or its transpose
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
int mtxmatrix_blas_dgemv(
    enum mtxtransposition trans,
    double alpha,
    const struct mtxmatrix_blas * A,
    const struct mtxvector * x,
    double beta,
    struct mtxvector * y,
    int64_t * num_flops)
{
#if 0
    const struct mtxvector_base * a = &A->a.base;
    if (x->type != mtxvector_base || y->type != mtxvector_base)
        return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
    const struct mtxvector_base * xbase = &x->storage.base;
    struct mtxvector_base * ybase = &y->storage.base;
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
#endif
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_blas_cgemv()’ multiplies a complex-valued matrix ‘A’,
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
int mtxmatrix_blas_cgemv(
    enum mtxtransposition trans,
    float alpha[2],
    const struct mtxmatrix_blas * A,
    const struct mtxvector * x,
    float beta[2],
    struct mtxvector * y,
    int64_t * num_flops)
{
#if 0
    const struct mtxvector_base * a = &A->a.base;
    if (x->type != mtxvector_base || y->type != mtxvector_base)
        return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
    const struct mtxvector_base * xbase = &x->storage.base;
    struct mtxvector_base * ybase = &y->storage.base;
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
#endif
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_blas_zgemv()’ multiplies a complex-valued matrix ‘A’,
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
int mtxmatrix_blas_zgemv(
    enum mtxtransposition trans,
    double alpha[2],
    const struct mtxmatrix_blas * A,
    const struct mtxvector * x,
    double beta[2],
    struct mtxvector * y,
    int64_t * num_flops)
{
#if 0
    const struct mtxvector_base * a = &A->a.base;
    if (x->type != mtxvector_base || y->type != mtxvector_base)
        return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
    const struct mtxvector_base * xbase = &x->storage.base;
    struct mtxvector_base * ybase = &y->storage.base;
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
#endif
    return MTX_SUCCESS;
}
