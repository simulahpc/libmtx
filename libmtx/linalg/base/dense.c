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
 * Data structures for dense matrices.
 */

#include <libmtx/libmtx-config.h>

#include <libmtx/error.h>
#include <libmtx/linalg/field.h>
#include <libmtx/linalg/precision.h>

#include <libmtx/linalg/base/dense.h>
#include <libmtx/linalg/local/matrix.h>
#include <libmtx/mtxfile/mtxfile.h>
#include <libmtx/linalg/local/vector.h>

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
 * ‘mtxbasedense_field()’ gets the field of a matrix.
 */
enum mtxfield mtxbasedense_field(const struct mtxbasedense * A)
{
    return mtxbasevector_field(&A->a);
}

/**
 * ‘mtxbasedense_precision()’ gets the precision of a matrix.
 */
enum mtxprecision mtxbasedense_precision(const struct mtxbasedense * A)
{
    return mtxbasevector_precision(&A->a);
}

/**
 * ‘mtxbasedense_symmetry()’ gets the symmetry of a matrix.
 */
enum mtxsymmetry mtxbasedense_symmetry(const struct mtxbasedense * A)
{
    return A->symmetry;
}

/**
 * ‘mtxbasedense_num_rows()’ gets the number of matrix rows.
 */
int mtxbasedense_num_rows(const struct mtxbasedense * A)
{
    return A->num_rows;
}

/**
 * ‘mtxbasedense_num_columns()’ gets the number of matrix columns.
 */
int mtxbasedense_num_columns(const struct mtxbasedense * A)
{
    return A->num_columns;
}

/**
 * ‘mtxbasedense_num_nonzeros()’ gets the number of the number of
 *  nonzero matrix entries, including those represented implicitly due
 *  to symmetry.
 */
int64_t mtxbasedense_num_nonzeros(const struct mtxbasedense * A)
{
    return A->num_nonzeros;
}

/**
 * ‘mtxbasedense_size()’ gets the number of explicitly stored
 * nonzeros of a matrix.
 */
int64_t mtxbasedense_size(const struct mtxbasedense * A)
{
    return A->size;
}

/**
 * ‘mtxbasedense_rowcolidx()’ gets the row and column indices of the
 * explicitly stored matrix nonzeros.
 *
 * The arguments ‘rowidx’ and ‘colidx’ may be ‘NULL’ or must point to
 * an arrays of length ‘size’.
 */
int mtxbasedense_rowcolidx(
    const struct mtxbasedense * A,
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
 * ‘mtxbasedense_free()’ frees storage allocated for a matrix.
 */
void mtxbasedense_free(
    struct mtxbasedense * A)
{
    mtxbasevector_free(&A->a);
}

/**
 * ‘mtxbasedense_alloc_copy()’ allocates a copy of a matrix without
 * initialising the values.
 */
int mtxbasedense_alloc_copy(
    struct mtxbasedense * dst,
    const struct mtxbasedense * src)
{
    return mtxbasedense_alloc_entries(
        dst, src->a.field, src->a.precision, src->symmetry,
        src->num_rows, src->num_columns, src->size,
        0, 0, NULL, NULL);
}

/**
 * ‘mtxbasedense_init_copy()’ allocates a copy of a matrix and also
 * copies the values.
 */
int mtxbasedense_init_copy(
    struct mtxbasedense * dst,
    const struct mtxbasedense * src)
{
    int err = mtxbasedense_alloc_copy(dst, src);
    if (err) return err;
    err = mtxbasedense_copy(dst, src);
    if (err) { mtxbasedense_free(dst); return err; }
    return MTX_SUCCESS;
}

/*
 * initialise matrices from entrywise data in coordinate format
 */

/**
 * ‘mtxbasedense_alloc_entries()’ allocates a matrix from entrywise
 * data in coordinate format.
 */
int mtxbasedense_alloc_entries(
    struct mtxbasedense * A,
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
    return mtxbasevector_alloc(&A->a, field, precision, A->size);
}

/**
 * ‘mtxbasedense_init_entries_real_single()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * real, single precision coefficients.
 */
int mtxbasedense_init_entries_real_single(
    struct mtxbasedense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx,
    const float * data)
{
    for (int64_t k = 0; k < size; k++) {
        if (rowidx[k] < 0 || rowidx[k] >= num_rows ||
            colidx[k] < 0 || colidx[k] >= num_columns)
            return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    }
    int err = mtxbasedense_alloc_entries(
        A, mtx_field_real, mtx_single, symmetry, num_rows, num_columns,
        size, 0, 0, NULL, NULL);
    if (err) return err;
    err = mtxbasedense_setzero(A);
    if (err) return err;
    float * a = A->a.data.real_single;
    if (symmetry == mtx_unsymmetric) {
        for (int64_t k = 0; k < size; k++) {
            int64_t i = rowidx[k], j = colidx[k];
            a[i*num_columns+j] = data[k];
        }
    } else if (num_rows == num_columns &&
               (symmetry == mtx_symmetric || symmetry == mtx_hermitian))
    {
        int64_t N = num_rows;
        for (int64_t k = 0; k < size; k++) {
            int64_t i = rowidx[k] < colidx[k] ? rowidx[k] : colidx[k];
            int64_t j = rowidx[k] < colidx[k] ? colidx[k] : rowidx[k];
            a[N*(N-1)/2 - (N-i)*(N-i-1)/2 + j] = data[k];
        }
    } else if (num_rows == num_columns && symmetry == mtx_skew_symmetric) {
        int64_t N = num_rows;
        for (int64_t k = 0; k < size; k++) {
            int64_t i = rowidx[k] < colidx[k] ? rowidx[k] : colidx[k];
            int64_t j = rowidx[k] < colidx[k] ? colidx[k] : rowidx[k];
            a[N*(N-1)/2 - (N-i)*(N-i-1)/2 + j-i-1] = data[k];
        }
    } else { mtxbasedense_free(A); return MTX_ERR_INVALID_SYMMETRY; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxbasedense_init_entries_real_double()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * real, double precision coefficients.
 */
int mtxbasedense_init_entries_real_double(
    struct mtxbasedense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx,
    const double * data)
{
    for (int64_t k = 0; k < size; k++) {
        if (rowidx[k] < 0 || rowidx[k] >= num_rows ||
            colidx[k] < 0 || colidx[k] >= num_columns)
            return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    }
    int err = mtxbasedense_alloc_entries(
        A, mtx_field_real, mtx_double, symmetry, num_rows, num_columns,
        size, 0, 0, NULL, NULL);
    if (err) return err;
    err = mtxbasedense_setzero(A);
    if (err) return err;
    double * a = A->a.data.real_double;
    if (symmetry == mtx_unsymmetric) {
        for (int64_t k = 0; k < size; k++) {
            int64_t i = rowidx[k], j = colidx[k];
            a[i*num_columns+j] = data[k];
        }
    } else if (num_rows == num_columns &&
               (symmetry == mtx_symmetric || symmetry == mtx_hermitian))
    {
        int64_t N = num_rows;
        for (int64_t k = 0; k < size; k++) {
            int64_t i = rowidx[k] < colidx[k] ? rowidx[k] : colidx[k];
            int64_t j = rowidx[k] < colidx[k] ? colidx[k] : rowidx[k];
            a[N*(N-1)/2 - (N-i)*(N-i-1)/2 + j] = data[k];
        }
    } else if (num_rows == num_columns && symmetry == mtx_skew_symmetric) {
        int64_t N = num_rows;
        for (int64_t k = 0; k < size; k++) {
            int64_t i = rowidx[k] < colidx[k] ? rowidx[k] : colidx[k];
            int64_t j = rowidx[k] < colidx[k] ? colidx[k] : rowidx[k];
            a[N*(N-1)/2 - (N-i)*(N-i-1)/2 + j-i-1] = data[k];
        }
    } else { mtxbasedense_free(A); return MTX_ERR_INVALID_SYMMETRY; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxbasedense_init_entries_complex_single()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * complex, single precision coefficients.
 */
int mtxbasedense_init_entries_complex_single(
    struct mtxbasedense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx,
    const float (* data)[2])
{
    for (int64_t k = 0; k < size; k++) {
        if (rowidx[k] < 0 || rowidx[k] >= num_rows ||
            colidx[k] < 0 || colidx[k] >= num_columns)
            return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    }
    int err = mtxbasedense_alloc_entries(
        A, mtx_field_complex, mtx_single, symmetry, num_rows, num_columns,
        size, 0, 0, NULL, NULL);
    if (err) return err;
    err = mtxbasedense_setzero(A);
    if (err) return err;
    float (* a)[2] = A->a.data.complex_single;
    if (symmetry == mtx_unsymmetric) {
        for (int64_t k = 0; k < size; k++) {
            int64_t i = rowidx[k], j = colidx[k];
            a[i*num_columns+j][0] = data[k][0];
            a[i*num_columns+j][1] = data[k][1];
        }
    } else if (num_rows == num_columns &&
               (symmetry == mtx_symmetric || symmetry == mtx_hermitian))
    {
        int64_t N = num_rows;
        for (int64_t k = 0; k < size; k++) {
            int64_t i = rowidx[k] < colidx[k] ? rowidx[k] : colidx[k];
            int64_t j = rowidx[k] < colidx[k] ? colidx[k] : rowidx[k];
            a[N*(N-1)/2 - (N-i)*(N-i-1)/2 + j][0] = data[k][0];
            a[N*(N-1)/2 - (N-i)*(N-i-1)/2 + j][1] = data[k][1];
        }
    } else if (num_rows == num_columns && symmetry == mtx_skew_symmetric) {
        int64_t N = num_rows;
        for (int64_t k = 0; k < size; k++) {
            int64_t i = rowidx[k] < colidx[k] ? rowidx[k] : colidx[k];
            int64_t j = rowidx[k] < colidx[k] ? colidx[k] : rowidx[k];
            a[N*(N-1)/2 - (N-i)*(N-i-1)/2 + j-i-1][0] = data[k][0];
            a[N*(N-1)/2 - (N-i)*(N-i-1)/2 + j-i-1][1] = data[k][1];
        }
    } else { mtxbasedense_free(A); return MTX_ERR_INVALID_SYMMETRY; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxbasedense_init_entries_complex_double()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * complex, double precision coefficients.
 */
int mtxbasedense_init_entries_complex_double(
    struct mtxbasedense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx,
    const double (* data)[2])
{
    for (int64_t k = 0; k < size; k++) {
        if (rowidx[k] < 0 || rowidx[k] >= num_rows ||
            colidx[k] < 0 || colidx[k] >= num_columns)
            return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    }
    int err = mtxbasedense_alloc_entries(
        A, mtx_field_complex, mtx_double, symmetry, num_rows, num_columns,
        size, 0, 0, NULL, NULL);
    if (err) return err;
    err = mtxbasedense_setzero(A);
    if (err) return err;
    double (* a)[2] = A->a.data.complex_double;
    if (symmetry == mtx_unsymmetric) {
        for (int64_t k = 0; k < size; k++) {
            int64_t i = rowidx[k], j = colidx[k];
            a[i*num_columns+j][0] = data[k][0];
            a[i*num_columns+j][1] = data[k][1];
        }
    } else if (num_rows == num_columns &&
               (symmetry == mtx_symmetric || symmetry == mtx_hermitian))
    {
        int64_t N = num_rows;
        for (int64_t k = 0; k < size; k++) {
            int64_t i = rowidx[k] < colidx[k] ? rowidx[k] : colidx[k];
            int64_t j = rowidx[k] < colidx[k] ? colidx[k] : rowidx[k];
            a[N*(N-1)/2 - (N-i)*(N-i-1)/2 + j][0] = data[k][0];
            a[N*(N-1)/2 - (N-i)*(N-i-1)/2 + j][1] = data[k][1];
        }
    } else if (num_rows == num_columns && symmetry == mtx_skew_symmetric) {
        int64_t N = num_rows;
        for (int64_t k = 0; k < size; k++) {
            int64_t i = rowidx[k] < colidx[k] ? rowidx[k] : colidx[k];
            int64_t j = rowidx[k] < colidx[k] ? colidx[k] : rowidx[k];
            a[N*(N-1)/2 - (N-i)*(N-i-1)/2 + j-i-1][0] = data[k][0];
            a[N*(N-1)/2 - (N-i)*(N-i-1)/2 + j-i-1][1] = data[k][1];
        }
    } else { mtxbasedense_free(A); return MTX_ERR_INVALID_SYMMETRY; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxbasedense_init_entries_integer_single()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * integer, single precision coefficients.
 */
int mtxbasedense_init_entries_integer_single(
    struct mtxbasedense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx,
    const int32_t * data)
{
    for (int64_t k = 0; k < size; k++) {
        if (rowidx[k] < 0 || rowidx[k] >= num_rows ||
            colidx[k] < 0 || colidx[k] >= num_columns)
            return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    }
    int err = mtxbasedense_alloc_entries(
        A, mtx_field_integer, mtx_single, symmetry, num_rows, num_columns,
        size, 0, 0, NULL, NULL);
    if (err) return err;
    err = mtxbasedense_setzero(A);
    if (err) return err;
    int32_t * a = A->a.data.integer_single;
    if (symmetry == mtx_unsymmetric) {
        for (int64_t k = 0; k < size; k++) {
            int64_t i = rowidx[k], j = colidx[k];
            a[i*num_columns+j] = data[k];
        }
    } else if (num_rows == num_columns &&
               (symmetry == mtx_symmetric || symmetry == mtx_hermitian))
    {
        int64_t N = num_rows;
        for (int64_t k = 0; k < size; k++) {
            int64_t i = rowidx[k] < colidx[k] ? rowidx[k] : colidx[k];
            int64_t j = rowidx[k] < colidx[k] ? colidx[k] : rowidx[k];
            a[N*(N-1)/2 - (N-i)*(N-i-1)/2 + j] = data[k];
        }
    } else if (num_rows == num_columns && symmetry == mtx_skew_symmetric) {
        int64_t N = num_rows;
        for (int64_t k = 0; k < size; k++) {
            int64_t i = rowidx[k] < colidx[k] ? rowidx[k] : colidx[k];
            int64_t j = rowidx[k] < colidx[k] ? colidx[k] : rowidx[k];
            a[N*(N-1)/2 - (N-i)*(N-i-1)/2 + j-i-1] = data[k];
        }
    } else { mtxbasedense_free(A); return MTX_ERR_INVALID_SYMMETRY; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxbasedense_init_entries_integer_double()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * integer, double precision coefficients.
 */
int mtxbasedense_init_entries_integer_double(
    struct mtxbasedense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx,
    const int64_t * data)
{
    for (int64_t k = 0; k < size; k++) {
        if (rowidx[k] < 0 || rowidx[k] >= num_rows ||
            colidx[k] < 0 || colidx[k] >= num_columns)
            return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    }
    int err = mtxbasedense_alloc_entries(
        A, mtx_field_integer, mtx_double, symmetry, num_rows, num_columns,
        size, 0, 0, NULL, NULL);
    if (err) return err;
    err = mtxbasedense_setzero(A);
    if (err) return err;
    int64_t * a = A->a.data.integer_double;
    if (symmetry == mtx_unsymmetric) {
        for (int64_t k = 0; k < size; k++) {
            int64_t i = rowidx[k], j = colidx[k];
            a[i*num_columns+j] = data[k];
        }
    } else if (num_rows == num_columns &&
               (symmetry == mtx_symmetric || symmetry == mtx_hermitian))
    {
        int64_t N = num_rows;
        for (int64_t k = 0; k < size; k++) {
            int64_t i = rowidx[k] < colidx[k] ? rowidx[k] : colidx[k];
            int64_t j = rowidx[k] < colidx[k] ? colidx[k] : rowidx[k];
            a[N*(N-1)/2 - (N-i)*(N-i-1)/2 + j] = data[k];
        }
    } else if (num_rows == num_columns && symmetry == mtx_skew_symmetric) {
        int64_t N = num_rows;
        for (int64_t k = 0; k < size; k++) {
            int64_t i = rowidx[k] < colidx[k] ? rowidx[k] : colidx[k];
            int64_t j = rowidx[k] < colidx[k] ? colidx[k] : rowidx[k];
            a[N*(N-1)/2 - (N-i)*(N-i-1)/2 + j-i-1] = data[k];
        }
    } else { mtxbasedense_free(A); return MTX_ERR_INVALID_SYMMETRY; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxbasedense_init_entries_pattern()’ allocates and initialises
 * a matrix from entrywise data in coordinate format with boolean
 * coefficients.
 */
int mtxbasedense_init_entries_pattern(
    struct mtxbasedense * A,
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
 * ‘mtxbasedense_init_entries_strided_real_single()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * real, single precision coefficients.
 */
int mtxbasedense_init_entries_strided_real_single(
    struct mtxbasedense * A,
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
 * ‘mtxbasedense_init_entries_strided_real_double()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * real, double precision coefficients.
 */
int mtxbasedense_init_entries_strided_real_double(
    struct mtxbasedense * A,
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
 * ‘mtxbasedense_init_entries_strided_complex_single()’ allocates
 * and initialises a matrix from entrywise data in coordinate format
 * with complex, single precision coefficients.
 */
int mtxbasedense_init_entries_strided_complex_single(
    struct mtxbasedense * A,
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
 * ‘mtxbasedense_init_entries_strided_complex_double()’ allocates
 * and initialises a matrix from entrywise data in coordinate format
 * with complex, double precision coefficients.
 */
int mtxbasedense_init_entries_strided_complex_double(
    struct mtxbasedense * A,
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
 * ‘mtxbasedense_init_entries_strided_integer_single()’ allocates
 * and initialises a matrix from entrywise data in coordinate format
 * with integer, single precision coefficients.
 */
int mtxbasedense_init_entries_strided_integer_single(
    struct mtxbasedense * A,
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
 * ‘mtxbasedense_init_entries_strided_integer_double()’ allocates
 * and initialises a matrix from entrywise data in coordinate format
 * with integer, double precision coefficients.
 */
int mtxbasedense_init_entries_strided_integer_double(
    struct mtxbasedense * A,
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
 * ‘mtxbasedense_init_entries_strided_pattern()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * boolean coefficients.
 */
int mtxbasedense_init_entries_strided_pattern(
    struct mtxbasedense * A,
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
 * ‘mtxbasedense_alloc_rows()’ allocates a matrix from row-wise
 * data in compressed row format.
 */
int mtxbasedense_alloc_rows(
    struct mtxbasedense * A,
    enum mtxfield field,
    enum mtxprecision precision,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx);

/**
 * ‘mtxbasedense_init_rows_real_single()’ allocates and initialises
 * a matrix from row-wise data in compressed row format with real,
 * single precision coefficients.
 */
int mtxbasedense_init_rows_real_single(
    struct mtxbasedense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const float * data);

/**
 * ‘mtxbasedense_init_rows_real_double()’ allocates and initialises
 * a matrix from row-wise data in compressed row format with real,
 * double precision coefficients.
 */
int mtxbasedense_init_rows_real_double(
    struct mtxbasedense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const double * data);

/**
 * ‘mtxbasedense_init_rows_complex_single()’ allocates and
 * initialises a matrix from row-wise data in compressed row format
 * with complex, single precision coefficients.
 */
int mtxbasedense_init_rows_complex_single(
    struct mtxbasedense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const float (* data)[2]);

/**
 * ‘mtxbasedense_init_rows_complex_double()’ allocates and
 * initialises a matrix from row-wise data in compressed row format
 * with complex, double precision coefficients.
 */
int mtxbasedense_init_rows_complex_double(
    struct mtxbasedense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const double (* data)[2]);

/**
 * ‘mtxbasedense_init_rows_integer_single()’ allocates and
 * initialises a matrix from row-wise data in compressed row format
 * with integer, single precision coefficients.
 */
int mtxbasedense_init_rows_integer_single(
    struct mtxbasedense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const int32_t * data);

/**
 * ‘mtxbasedense_init_rows_integer_double()’ allocates and
 * initialises a matrix from row-wise data in compressed row format
 * with integer, double precision coefficients.
 */
int mtxbasedense_init_rows_integer_double(
    struct mtxbasedense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const int64_t * data);

/**
 * ‘mtxbasedense_init_rows_pattern()’ allocates and initialises a
 * matrix from row-wise data in compressed row format with boolean
 * coefficients.
 */
int mtxbasedense_init_rows_pattern(
    struct mtxbasedense * A,
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
 * ‘mtxbasedense_alloc_columns()’ allocates a matrix from
 * column-wise data in compressed column format.
 */
int mtxbasedense_alloc_columns(
    struct mtxbasedense * A,
    enum mtxfield field,
    enum mtxprecision precision,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx);

/**
 * ‘mtxbasedense_init_columns_real_single()’ allocates and
 * initialises a matrix from column-wise data in compressed column
 * format with real, single precision coefficients.
 */
int mtxbasedense_init_columns_real_single(
    struct mtxbasedense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx,
    const float * data);

/**
 * ‘mtxbasedense_init_columns_real_double()’ allocates and
 * initialises a matrix from column-wise data in compressed column
 * format with real, double precision coefficients.
 */
int mtxbasedense_init_columns_real_double(
    struct mtxbasedense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx,
    const double * data);

/**
 * ‘mtxbasedense_init_columns_complex_single()’ allocates and
 * initialises a matrix from column-wise data in compressed column
 * format with complex, single precision coefficients.
 */
int mtxbasedense_init_columns_complex_single(
    struct mtxbasedense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx,
    const float (* data)[2]);

/**
 * ‘mtxbasedense_init_columns_complex_double()’ allocates and
 * initialises a matrix from column-wise data in compressed column
 * format with complex, double precision coefficients.
 */
int mtxbasedense_init_columns_complex_double(
    struct mtxbasedense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx,
    const double (* data)[2]);

/**
 * ‘mtxbasedense_init_columns_integer_single()’ allocates and
 * initialises a matrix from column-wise data in compressed column
 * format with integer, single precision coefficients.
 */
int mtxbasedense_init_columns_integer_single(
    struct mtxbasedense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx,
    const int32_t * data);

/**
 * ‘mtxbasedense_init_columns_integer_double()’ allocates and
 * initialises a matrix from column-wise data in compressed column
 * format with integer, double precision coefficients.
 */
int mtxbasedense_init_columns_integer_double(
    struct mtxbasedense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx,
    const int64_t * data);

/**
 * ‘mtxbasedense_init_columns_pattern()’ allocates and initialises
 * a matrix from column-wise data in compressed column format with
 * boolean coefficients.
 */
int mtxbasedense_init_columns_pattern(
    struct mtxbasedense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx);

/*
 * initialise matrices from a list of dense cliques
 */

/**
 * ‘mtxbasedense_alloc_cliques()’ allocates a matrix from a list of
 * dense cliques.
 */
int mtxbasedense_alloc_cliques(
    struct mtxbasedense * A,
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
 * ‘mtxbasedense_init_cliques_real_single()’ allocates and
 * initialises a matrix from a list of dense cliques with real, single
 * precision coefficients.
 */
int mtxbasedense_init_cliques_real_single(
    struct mtxbasedense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_cliques,
    const int64_t * cliqueptr,
    const int * rowidx,
    const int * colidx,
    const float * data);

/**
 * ‘mtxbasedense_init_cliques_real_double()’ allocates and
 * initialises a matrix from a list of dense cliques with real, double
 * precision coefficients.
 */
int mtxbasedense_init_cliques_real_double(
    struct mtxbasedense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_cliques,
    const int64_t * cliqueptr,
    const int * rowidx,
    const int * colidx,
    const double * data);

/**
 * ‘mtxbasedense_init_cliques_complex_single()’ allocates and
 * initialises a matrix from a list of dense cliques with complex,
 * single precision coefficients.
 */
int mtxbasedense_init_cliques_complex_single(
    struct mtxbasedense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_cliques,
    const int64_t * cliqueptr,
    const int * rowidx,
    const int * colidx,
    const float (* data)[2]);

/**
 * ‘mtxbasedense_init_cliques_complex_double()’ allocates and
 * initialises a matrix from a list of dense cliques with complex,
 * double precision coefficients.
 */
int mtxbasedense_init_cliques_complex_double(
    struct mtxbasedense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_cliques,
    const int64_t * cliqueptr,
    const int * rowidx,
    const int * colidx,
    const double (* data)[2]);

/**
 * ‘mtxbasedense_init_cliques_integer_single()’ allocates and
 * initialises a matrix from a list of dense cliques with integer,
 * single precision coefficients.
 */
int mtxbasedense_init_cliques_integer_single(
    struct mtxbasedense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_cliques,
    const int64_t * cliqueptr,
    const int * rowidx,
    const int * colidx,
    const int32_t * data);

/**
 * ‘mtxbasedense_init_cliques_integer_double()’ allocates and
 * initialises a matrix from a list of dense cliques with integer,
 * double precision coefficients.
 */
int mtxbasedense_init_cliques_integer_double(
    struct mtxbasedense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_cliques,
    const int64_t * cliqueptr,
    const int * rowidx,
    const int * colidx,
    const int64_t * data);

/**
 * ‘mtxbasedense_init_cliques_pattern()’ allocates and initialises
 * a matrix from a list of dense cliques with boolean coefficients.
 */
int mtxbasedense_init_cliques_pattern(
    struct mtxbasedense * A,
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
 * ‘mtxbasedense_setzero()’ sets every value of a matrix to zero.
 */
int mtxbasedense_setzero(
    struct mtxbasedense * A)
{
    return mtxbasevector_setzero(&A->a);
}

/**
 * ‘mtxbasedense_set_real_single()’ sets values of a matrix based
 * on an array of single precision floating point numbers.
 */
int mtxbasedense_set_real_single(
    struct mtxbasedense * A,
    int64_t size,
    int stride,
    const float * a)
{
    return mtxbasevector_set_real_single(&A->a, size, stride, a);
}

/**
 * ‘mtxbasedense_set_real_double()’ sets values of a matrix based
 * on an array of double precision floating point numbers.
 */
int mtxbasedense_set_real_double(
    struct mtxbasedense * A,
    int64_t size,
    int stride,
    const double * a)
{
    return mtxbasevector_set_real_double(&A->a, size, stride, a);
}

/**
 * ‘mtxbasedense_set_complex_single()’ sets values of a matrix
 * based on an array of single precision floating point complex
 * numbers.
 */
int mtxbasedense_set_complex_single(
    struct mtxbasedense * A,
    int64_t size,
    int stride,
    const float (*a)[2])
{
    return mtxbasevector_set_complex_single(&A->a, size, stride, a);
}

/**
 * ‘mtxbasedense_set_complex_double()’ sets values of a matrix
 * based on an array of double precision floating point complex
 * numbers.
 */
int mtxbasedense_set_complex_double(
    struct mtxbasedense * A,
    int64_t size,
    int stride,
    const double (*a)[2])
{
    return mtxbasevector_set_complex_double(&A->a, size, stride, a);
}

/**
 * ‘mtxbasedense_set_integer_single()’ sets values of a matrix
 * based on an array of integers.
 */
int mtxbasedense_set_integer_single(
    struct mtxbasedense * A,
    int64_t size,
    int stride,
    const int32_t * a)
{
    return mtxbasevector_set_integer_single(&A->a, size, stride, a);
}

/**
 * ‘mtxbasedense_set_integer_double()’ sets values of a matrix
 * based on an array of integers.
 */
int mtxbasedense_set_integer_double(
    struct mtxbasedense * A,
    int64_t size,
    int stride,
    const int64_t * a)
{
    return mtxbasevector_set_integer_double(&A->a, size, stride, a);
}

/*
 * row and column vectors
 */

/**
 * ‘mtxbasedense_alloc_row_vector()’ allocates a row vector for a
 * given matrix, where a row vector is a vector whose length equal to
 * a single row of the matrix.
 */
int mtxbasedense_alloc_row_vector(
    const struct mtxbasedense * A,
    struct mtxvector * x,
    enum mtxvectortype vectortype)
{
    return mtxvector_alloc(
        x, vectortype, A->a.field, A->a.precision, A->num_columns);
}

/**
 * ‘mtxbasedense_alloc_column_vector()’ allocates a column vector
 * for a given matrix, where a column vector is a vector whose length
 * equal to a single column of the matrix.
 */
int mtxbasedense_alloc_column_vector(
    const struct mtxbasedense * A,
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
 * ‘mtxbasedense_from_mtxfile()’ converts a matrix from Matrix
 * Market format.
 */
int mtxbasedense_from_mtxfile(
    struct mtxbasedense * A,
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

    err = mtxbasedense_alloc_entries(
        A, field, precision, symmetry, num_rows, num_columns,
        num_nonzeros, 0, 0, NULL, NULL);
    if (err) return err;
    err = mtxbasedense_setzero(A);
    if (err) { mtxbasedense_free(A); return err; }

    if (mtxfile->header.format == mtxfile_coordinate) {
        if (mtxfile->header.field == mtxfile_real) {
            if (mtxfile->precision == mtx_single) {
                const struct mtxfile_matrix_coordinate_real_single * data =
                    mtxfile->data.matrix_coordinate_real_single;
                float * a = A->a.data.real_single;
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
                } else { mtxbasedense_free(A); return MTX_ERR_INVALID_SYMMETRY; }
            } else if (mtxfile->precision == mtx_double) {
                const struct mtxfile_matrix_coordinate_real_double * data =
                    mtxfile->data.matrix_coordinate_real_double;
                double * a = A->a.data.real_double;
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
                } else { mtxbasedense_free(A); return MTX_ERR_INVALID_SYMMETRY; }
            } else { mtxbasedense_free(A); return MTX_ERR_INVALID_PRECISION; }
        } else if (mtxfile->header.field == mtxfile_complex) {
            if (mtxfile->precision == mtx_single) {
                const struct mtxfile_matrix_coordinate_complex_single * data =
                    mtxfile->data.matrix_coordinate_complex_single;
                float (*a)[2] = A->a.data.complex_single;
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
                } else { mtxbasedense_free(A); return MTX_ERR_INVALID_SYMMETRY; }
            } else if (mtxfile->precision == mtx_double) {
                const struct mtxfile_matrix_coordinate_complex_double * data =
                    mtxfile->data.matrix_coordinate_complex_double;
                double (*a)[2] = A->a.data.complex_double;
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
                } else { mtxbasedense_free(A); return MTX_ERR_INVALID_SYMMETRY; }
            } else { mtxbasedense_free(A); return MTX_ERR_INVALID_PRECISION; }
        } else if (mtxfile->header.field == mtxfile_integer) {
            if (mtxfile->precision == mtx_single) {
                const struct mtxfile_matrix_coordinate_integer_single * data =
                    mtxfile->data.matrix_coordinate_integer_single;
                int32_t * a = A->a.data.integer_single;
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
                } else { mtxbasedense_free(A); return MTX_ERR_INVALID_SYMMETRY; }
            } else if (mtxfile->precision == mtx_double) {
                const struct mtxfile_matrix_coordinate_integer_double * data =
                    mtxfile->data.matrix_coordinate_integer_double;
                int64_t * a = A->a.data.integer_double;
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
                } else { mtxbasedense_free(A); return MTX_ERR_INVALID_SYMMETRY; }
            } else { mtxbasedense_free(A); return MTX_ERR_INVALID_PRECISION; }
        } else { mtxbasedense_free(A); return MTX_ERR_INVALID_MTX_FIELD; }
    } else if (mtxfile->header.format == mtxfile_array) {
        if (mtxfile->header.field == mtxfile_real) {
            if (mtxfile->precision == mtx_single) {
                const float * data = mtxfile->data.array_real_single;
                float * a = A->a.data.real_single;
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
                } else { mtxbasedense_free(A); return MTX_ERR_INVALID_SYMMETRY; }
            } else if (mtxfile->precision == mtx_double) {
                const double * data = mtxfile->data.array_real_double;
                double * a = A->a.data.real_double;
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
                } else { mtxbasedense_free(A); return MTX_ERR_INVALID_SYMMETRY; }
            } else { mtxbasedense_free(A); return MTX_ERR_INVALID_PRECISION; }
        } else if (mtxfile->header.field == mtxfile_complex) {
            if (mtxfile->precision == mtx_single) {
                const float (*data)[2] = mtxfile->data.array_complex_single;
                float (*a)[2] = A->a.data.complex_single;
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
                } else { mtxbasedense_free(A); return MTX_ERR_INVALID_SYMMETRY; }
            } else if (mtxfile->precision == mtx_double) {
                const double (*data)[2] = mtxfile->data.array_complex_double;
                double (*a)[2] = A->a.data.complex_double;
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
                } else { mtxbasedense_free(A); return MTX_ERR_INVALID_SYMMETRY; }
            } else { mtxbasedense_free(A); return MTX_ERR_INVALID_PRECISION; }
        } else if (mtxfile->header.field == mtxfile_integer) {
            if (mtxfile->precision == mtx_single) {
                const int32_t * data = mtxfile->data.array_integer_single;
                int32_t * a = A->a.data.integer_single;
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
                } else { mtxbasedense_free(A); return MTX_ERR_INVALID_SYMMETRY; }
            } else if (mtxfile->precision == mtx_double) {
                const int64_t * data = mtxfile->data.array_integer_double;
                int64_t * a = A->a.data.integer_double;
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
                } else { mtxbasedense_free(A); return MTX_ERR_INVALID_SYMMETRY; }
            } else { mtxbasedense_free(A); return MTX_ERR_INVALID_PRECISION; }
        } else { mtxbasedense_free(A); return MTX_ERR_INVALID_MTX_FIELD; }
    } else { mtxbasedense_free(A); return MTX_ERR_INVALID_MTX_FORMAT; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxbasedense_to_mtxfile()’ converts a matrix to Matrix Market
 * format.
 */
int mtxbasedense_to_mtxfile(
    struct mtxfile * mtxfile,
    const struct mtxbasedense * A,
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
    enum mtxfield field = A->a.field;
    enum mtxfilefield mtxfield;
    err = mtxfilefield_from_mtxfield(&mtxfield, field);
    if (err) return err;
    enum mtxprecision precision = A->a.precision;

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
                const float * a = A->a.data.real_single;
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
                const double * a = A->a.data.real_double;
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
                const float (* a)[2] = A->a.data.complex_single;
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
                const double (* a)[2] = A->a.data.complex_double;
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
                const int32_t * a = A->a.data.integer_single;
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
                const int64_t * a = A->a.data.integer_double;
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
                const float * a = A->a.data.real_single;
                for (int64_t k = 0; k < A->size; k++) data[k] = a[k];
            } else if (precision == mtx_double) {
                double * data = mtxfile->data.array_real_double;
                const double * a = A->a.data.real_double;
                for (int64_t k = 0; k < A->size; k++) data[k] = a[k];
            } else { mtxfile_free(mtxfile); return MTX_ERR_INVALID_PRECISION; }
        } else if (field == mtx_field_complex) {
            if (precision == mtx_single) {
                float (* data)[2] = mtxfile->data.array_complex_single;
                const float (* a)[2] = A->a.data.complex_single;
                for (int64_t k = 0; k < A->size; k++) { data[k][0] = a[k][0]; data[k][1] = a[k][1]; }
            } else if (precision == mtx_double) {
                double (* data)[2] = mtxfile->data.array_complex_double;
                const double (* a)[2] = A->a.data.complex_double;
                for (int64_t k = 0; k < A->size; k++) { data[k][0] = a[k][0]; data[k][1] = a[k][1]; }
            } else { mtxfile_free(mtxfile); return MTX_ERR_INVALID_PRECISION; }
        } else if (field == mtx_field_integer) {
            if (precision == mtx_single) {
                int32_t * data = mtxfile->data.array_integer_single;
                const int32_t * a = A->a.data.integer_single;
                for (int64_t k = 0; k < A->size; k++) data[k] = a[k];
            } else if (precision == mtx_double) {
                int64_t * data = mtxfile->data.array_integer_double;
                const int64_t * a = A->a.data.integer_double;
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
 * ‘mtxbasedense_swap()’ swaps values of two matrices,
 * simultaneously performing ‘y <- x’ and ‘x <- y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxbasedense_swap(
    struct mtxbasedense * x,
    struct mtxbasedense * y)
{
    return mtxbasevector_swap(&x->a, &y->a);
}

/**
 * ‘mtxbasedense_copy()’ copies values of a matrix, ‘y = x’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxbasedense_copy(
    struct mtxbasedense * y,
    const struct mtxbasedense * x)
{
    return mtxbasevector_copy(&y->a, &x->a);
}

/**
 * ‘mtxbasedense_sscal()’ scales a matrix by a single precision
 * floating point scalar, ‘x = a*x’.
 */
int mtxbasedense_sscal(
    float a,
    struct mtxbasedense * x,
    int64_t * num_flops)
{
    return mtxbasevector_sscal(a, &x->a, num_flops);
}

/**
 * ‘mtxbasedense_dscal()’ scales a matrix by a double precision
 * floating point scalar, ‘x = a*x’.
 */
int mtxbasedense_dscal(
    double a,
    struct mtxbasedense * x,
    int64_t * num_flops)
{
    return mtxbasevector_dscal(a, &x->a, num_flops);
}

/**
 * ‘mtxbasedense_cscal()’ scales a matrix by a complex, single
 * precision floating point scalar, ‘x = (a+b*i)*x’.
 */
int mtxbasedense_cscal(
    float a[2],
    struct mtxbasedense * x,
    int64_t * num_flops)
{
    return mtxbasevector_cscal(a, &x->a, num_flops);
}

/**
 * ‘mtxbasedense_zscal()’ scales a matrix by a complex, double
 * precision floating point scalar, ‘x = (a+b*i)*x’.
 */
int mtxbasedense_zscal(
    double a[2],
    struct mtxbasedense * x,
    int64_t * num_flops)
{
    return mtxbasevector_zscal(a, &x->a, num_flops);
}

/**
 * ‘mtxbasedense_saxpy()’ adds a matrix to another one multiplied
 * by a single precision floating point value, ‘y = a*x + y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxbasedense_saxpy(
    float a,
    const struct mtxbasedense * x,
    struct mtxbasedense * y,
    int64_t * num_flops)
{
    return mtxbasevector_saxpy(a, &x->a, &y->a, num_flops);
}

/**
 * ‘mtxbasedense_daxpy()’ adds a matrix to another one multiplied
 * by a double precision floating point value, ‘y = a*x + y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxbasedense_daxpy(
    double a,
    const struct mtxbasedense * x,
    struct mtxbasedense * y,
    int64_t * num_flops)
{
    return mtxbasevector_daxpy(a, &x->a, &y->a, num_flops);
}

/**
 * ‘mtxbasedense_saypx()’ multiplies a matrix by a single precision
 * floating point scalar and adds another matrix, ‘y = a*y + x’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxbasedense_saypx(
    float a,
    struct mtxbasedense * y,
    const struct mtxbasedense * x,
    int64_t * num_flops)
{
    return mtxbasevector_saypx(a, &y->a, &x->a, num_flops);
}

/**
 * ‘mtxbasedense_daypx()’ multiplies a matrix by a double precision
 * floating point scalar and adds another matrix, ‘y = a*y + x’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxbasedense_daypx(
    double a,
    struct mtxbasedense * y,
    const struct mtxbasedense * x,
    int64_t * num_flops)
{
    return mtxbasevector_daypx(a, &y->a, &x->a, num_flops);
}

/**
 * ‘mtxbasedense_sdot()’ computes the Frobenius inner product of
 * two matrices in single precision floating point.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxbasedense_sdot(
    const struct mtxbasedense * x,
    const struct mtxbasedense * y,
    float * dot,
    int64_t * num_flops)
{
    return mtxbasevector_sdot(&x->a, &y->a, dot, num_flops);
}

/**
 * ‘mtxbasedense_ddot()’ computes the Frobenius inner product of
 * two matrices in double precision floating point.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxbasedense_ddot(
    const struct mtxbasedense * x,
    const struct mtxbasedense * y,
    double * dot,
    int64_t * num_flops)
{
    return mtxbasevector_ddot(&x->a, &y->a, dot, num_flops);
}

/**
 * ‘mtxbasedense_cdotu()’ computes the product of the transpose of
 * a complex row matrix with another complex row matrix in single
 * precision floating point, ‘dot := x^T*y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxbasedense_cdotu(
    const struct mtxbasedense * x,
    const struct mtxbasedense * y,
    float (* dot)[2],
    int64_t * num_flops)
{
    return mtxbasevector_cdotu(&x->a, &y->a, dot, num_flops);
}

/**
 * ‘mtxbasedense_zdotu()’ computes the product of the transpose of
 * a complex row matrix with another complex row matrix in double
 * precision floating point, ‘dot := x^T*y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxbasedense_zdotu(
    const struct mtxbasedense * x,
    const struct mtxbasedense * y,
    double (* dot)[2],
    int64_t * num_flops)
{
    return mtxbasevector_zdotu(&x->a, &y->a, dot, num_flops);
}

/**
 * ‘mtxbasedense_cdotc()’ computes the Frobenius inner product of
 * two complex matrices in single precision floating point, ‘dot :=
 * x^H*y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxbasedense_cdotc(
    const struct mtxbasedense * x,
    const struct mtxbasedense * y,
    float (* dot)[2],
    int64_t * num_flops)
{
    return mtxbasevector_cdotc(&x->a, &y->a, dot, num_flops);
}

/**
 * ‘mtxbasedense_zdotc()’ computes the Frobenius inner product of
 * two complex matrices in double precision floating point, ‘dot :=
 * x^H*y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxbasedense_zdotc(
    const struct mtxbasedense * x,
    const struct mtxbasedense * y,
    double (* dot)[2],
    int64_t * num_flops)
{
    return mtxbasevector_zdotc(&x->a, &y->a, dot, num_flops);
}

/**
 * ‘mtxbasedense_snrm2()’ computes the Frobenius norm of a matrix
 * in single precision floating point.
 */
int mtxbasedense_snrm2(
    const struct mtxbasedense * x,
    float * nrm2,
    int64_t * num_flops)
{
    return mtxbasevector_snrm2(&x->a, nrm2, num_flops);
}

/**
 * ‘mtxbasedense_dnrm2()’ computes the Frobenius norm of a matrix
 * in double precision floating point.
 */
int mtxbasedense_dnrm2(
    const struct mtxbasedense * x,
    double * nrm2,
    int64_t * num_flops)
{
    return mtxbasevector_dnrm2(&x->a, nrm2, num_flops);
}

/**
 * ‘mtxbasedense_sasum()’ computes the sum of absolute values
 * (1-norm) of a matrix in single precision floating point.  If the
 * matrix is complex-valued, then the sum of the absolute values of
 * the real and imaginary parts is computed.
 */
int mtxbasedense_sasum(
    const struct mtxbasedense * x,
    float * asum,
    int64_t * num_flops)
{
    return mtxbasevector_sasum(&x->a, asum, num_flops);
}

/**
 * ‘mtxbasedense_dasum()’ computes the sum of absolute values
 * (1-norm) of a matrix in double precision floating point.  If the
 * matrix is complex-valued, then the sum of the absolute values of
 * the real and imaginary parts is computed.
 */
int mtxbasedense_dasum(
    const struct mtxbasedense * x,
    double * asum,
    int64_t * num_flops)
{
    return mtxbasevector_dasum(&x->a, asum, num_flops);
}

/**
 * ‘mtxbasedense_iamax()’ finds the index of the first element
 * having the maximum absolute value.  If the matrix is
 * complex-valued, then the index points to the first element having
 * the maximum sum of the absolute values of the real and imaginary
 * parts.
 */
int mtxbasedense_iamax(
    const struct mtxbasedense * x,
    int * iamax)
{
    return mtxbasevector_iamax(&x->a, iamax);
}

/*
 * Level 2 BLAS operations (matrix-vector)
 */

/**
 * ‘mtxbasedense_sgemv()’ multiplies a matrix ‘A’ or its transpose
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
int mtxbasedense_sgemv(
    enum mtxtransposition trans,
    float alpha,
    const struct mtxbasedense * A,
    const struct mtxvector * x,
    float beta,
    struct mtxvector * y,
    int64_t * num_flops)
{
    const struct mtxbasevector * a = &A->a;
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

    if (beta != 1) {
        int err = mtxvector_sscal(beta, y, num_flops);
        if (err) return err;
    }

    if (A->symmetry == mtx_unsymmetric) {
        if (a->field == mtx_field_real) {
            if (trans == mtx_notrans) {
                if (a->precision == mtx_single) {
                    const float * Adata = a->data.real_single;
                    const float * xdata = xbase->data.real_single;
                    float * ydata = ybase->data.real_single;
                    for (int i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = 0; j < A->num_columns; j++, k++)
                            ydata[i] += alpha*Adata[k]*xdata[j];
                    }
                    if (num_flops) *num_flops += 3*A->num_entries;
                } else if (a->precision == mtx_double) {
                    const double * Adata = a->data.real_double;
                    const double * xdata = xbase->data.real_double;
                    double * ydata = ybase->data.real_double;
                    for (int i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = 0; j < A->num_columns; j++, k++)
                            ydata[i] += alpha*Adata[k]*xdata[j];
                    }
                    if (num_flops) *num_flops += 3*A->num_entries;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (trans == mtx_trans || trans == mtx_conjtrans) {
                if (a->precision == mtx_single) {
                    const float * Adata = a->data.real_single;
                    const float * xdata = xbase->data.real_single;
                    float * ydata = ybase->data.real_single;
                    for (int i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = 0; j < A->num_columns; j++, k++)
                            ydata[j] += alpha*Adata[k]*xdata[i];
                    }
                    if (num_flops) *num_flops += 3*A->num_entries;
                } else if (a->precision == mtx_double) {
                    const double * Adata = a->data.real_double;
                    const double * xdata = xbase->data.real_double;
                    double * ydata = ybase->data.real_double;
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
                    const float (* xdata)[2] = xbase->data.complex_single;
                    float (* ydata)[2] = ybase->data.complex_single;
                    for (int i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = 0; j < A->num_columns; j++, k++) {
                            ydata[i][0] += alpha*(Adata[k][0]*xdata[j][0]-Adata[k][1]*xdata[j][1]);
                            ydata[i][1] += alpha*(Adata[k][0]*xdata[j][1]+Adata[k][1]*xdata[j][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_entries;
                } else if (a->precision == mtx_double) {
                    const double (* Adata)[2] = a->data.complex_double;
                    const double (* xdata)[2] = xbase->data.complex_double;
                    double (* ydata)[2] = ybase->data.complex_double;
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
                    const float (* xdata)[2] = xbase->data.complex_single;
                    float (* ydata)[2] = ybase->data.complex_single;
                    for (int i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = 0; j < A->num_columns; j++, k++) {
                            ydata[j][0] += alpha*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                            ydata[j][1] += alpha*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_entries;
                } else if (a->precision == mtx_double) {
                    const double (* Adata)[2] = a->data.complex_double;
                    const double (* xdata)[2] = xbase->data.complex_double;
                    double (* ydata)[2] = ybase->data.complex_double;
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
                    const float (* xdata)[2] = xbase->data.complex_single;
                    float (* ydata)[2] = ybase->data.complex_single;
                    for (int i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = 0; j < A->num_columns; j++, k++) {
                            ydata[j][0] += alpha*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                            ydata[j][1] += alpha*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_entries;
                } else if (a->precision == mtx_double) {
                    const double (* Adata)[2] = a->data.complex_double;
                    const double (* xdata)[2] = xbase->data.complex_double;
                    double (* ydata)[2] = ybase->data.complex_double;
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
                    const int32_t * xdata = xbase->data.integer_single;
                    int32_t * ydata = ybase->data.integer_single;
                    for (int i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = 0; j < A->num_columns; j++, k++)
                            ydata[i] += alpha*Adata[k]*xdata[j];
                    }
                    if (num_flops) *num_flops += 3*A->num_entries;
                } else if (a->precision == mtx_double) {
                    const int64_t * Adata = a->data.integer_double;
                    const int64_t * xdata = xbase->data.integer_double;
                    int64_t * ydata = ybase->data.integer_double;
                    for (int i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = 0; j < A->num_columns; j++, k++)
                            ydata[i] += alpha*Adata[k]*xdata[j];
                    }
                    if (num_flops) *num_flops += 3*A->num_entries;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (trans == mtx_trans || trans == mtx_conjtrans) {
                if (a->precision == mtx_single) {
                    const int32_t * Adata = a->data.integer_single;
                    const int32_t * xdata = xbase->data.integer_single;
                    int32_t * ydata = ybase->data.integer_single;
                    for (int i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = 0; j < A->num_columns; j++, k++)
                            ydata[j] += alpha*Adata[k]*xdata[i];
                    }
                    if (num_flops) *num_flops += 3*A->num_entries;
                } else if (a->precision == mtx_double) {
                    const int64_t * Adata = a->data.integer_double;
                    const int64_t * xdata = xbase->data.integer_double;
                    int64_t * ydata = ybase->data.integer_double;
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
                const float * xdata = xbase->data.real_single;
                float * ydata = ybase->data.real_single;
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
                const double * xdata = xbase->data.real_double;
                double * ydata = ybase->data.real_double;
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
                    const float (* xdata)[2] = xbase->data.complex_single;
                    float (* ydata)[2] = ybase->data.complex_single;
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
                    const double (* xdata)[2] = xbase->data.complex_double;
                    double (* ydata)[2] = ybase->data.complex_double;
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
                    const float (* xdata)[2] = xbase->data.complex_single;
                    float (* ydata)[2] = ybase->data.complex_single;
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
                    const double (* xdata)[2] = xbase->data.complex_double;
                    double (* ydata)[2] = ybase->data.complex_double;
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
                const int32_t * xdata = xbase->data.integer_single;
                int32_t * ydata = ybase->data.integer_single;
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
                const int64_t * xdata = xbase->data.integer_double;
                int64_t * ydata = ybase->data.integer_double;
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
                    const float (* xdata)[2] = xbase->data.complex_single;
                    float (* ydata)[2] = ybase->data.complex_single;
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
                    const double (* xdata)[2] = xbase->data.complex_double;
                    double (* ydata)[2] = ybase->data.complex_double;
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
                    const float (* xdata)[2] = xbase->data.complex_single;
                    float (* ydata)[2] = ybase->data.complex_single;
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
                    const double (* xdata)[2] = xbase->data.complex_double;
                    double (* ydata)[2] = ybase->data.complex_double;
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
 * ‘mtxbasedense_dgemv()’ multiplies a matrix ‘A’ or its transpose
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
int mtxbasedense_dgemv(
    enum mtxtransposition trans,
    double alpha,
    const struct mtxbasedense * A,
    const struct mtxvector * x,
    double beta,
    struct mtxvector * y,
    int64_t * num_flops)
{
    const struct mtxbasevector * a = &A->a;
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

    if (beta != 1) {
        int err = mtxvector_dscal(beta, y, num_flops);
        if (err) return err;
    }

    if (A->symmetry == mtx_unsymmetric) {
        if (a->field == mtx_field_real) {
            if (trans == mtx_notrans) {
                if (a->precision == mtx_single) {
                    const float * Adata = a->data.real_single;
                    const float * xdata = xbase->data.real_single;
                    float * ydata = ybase->data.real_single;
                    for (int i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = 0; j < A->num_columns; j++, k++)
                            ydata[i] += alpha*Adata[k]*xdata[j];
                    }
                    if (num_flops) *num_flops += 3*A->num_entries;
                } else if (a->precision == mtx_double) {
                    const double * Adata = a->data.real_double;
                    const double * xdata = xbase->data.real_double;
                    double * ydata = ybase->data.real_double;
                    for (int i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = 0; j < A->num_columns; j++, k++)
                            ydata[i] += alpha*Adata[k]*xdata[j];
                    }
                    if (num_flops) *num_flops += 3*A->num_entries;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (trans == mtx_trans || trans == mtx_conjtrans) {
                if (a->precision == mtx_single) {
                    const float * Adata = a->data.real_single;
                    const float * xdata = xbase->data.real_single;
                    float * ydata = ybase->data.real_single;
                    for (int i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = 0; j < A->num_columns; j++, k++)
                            ydata[j] += alpha*Adata[k]*xdata[i];
                    }
                    if (num_flops) *num_flops += 3*A->num_entries;
                } else if (a->precision == mtx_double) {
                    const double * Adata = a->data.real_double;
                    const double * xdata = xbase->data.real_double;
                    double * ydata = ybase->data.real_double;
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
                    const float (* xdata)[2] = xbase->data.complex_single;
                    float (* ydata)[2] = ybase->data.complex_single;
                    for (int i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = 0; j < A->num_columns; j++, k++) {
                            ydata[i][0] += alpha*(Adata[k][0]*xdata[j][0]-Adata[k][1]*xdata[j][1]);
                            ydata[i][1] += alpha*(Adata[k][0]*xdata[j][1]+Adata[k][1]*xdata[j][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_entries;
                } else if (a->precision == mtx_double) {
                    const double (* Adata)[2] = a->data.complex_double;
                    const double (* xdata)[2] = xbase->data.complex_double;
                    double (* ydata)[2] = ybase->data.complex_double;
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
                    const float (* xdata)[2] = xbase->data.complex_single;
                    float (* ydata)[2] = ybase->data.complex_single;
                    for (int i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = 0; j < A->num_columns; j++, k++) {
                            ydata[j][0] += alpha*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                            ydata[j][1] += alpha*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_entries;
                } else if (a->precision == mtx_double) {
                    const double (* Adata)[2] = a->data.complex_double;
                    const double (* xdata)[2] = xbase->data.complex_double;
                    double (* ydata)[2] = ybase->data.complex_double;
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
                    const float (* xdata)[2] = xbase->data.complex_single;
                    float (* ydata)[2] = ybase->data.complex_single;
                    for (int i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = 0; j < A->num_columns; j++, k++) {
                            ydata[j][0] += alpha*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                            ydata[j][1] += alpha*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_entries;
                } else if (a->precision == mtx_double) {
                    const double (* Adata)[2] = a->data.complex_double;
                    const double (* xdata)[2] = xbase->data.complex_double;
                    double (* ydata)[2] = ybase->data.complex_double;
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
                    const int32_t * xdata = xbase->data.integer_single;
                    int32_t * ydata = ybase->data.integer_single;
                    for (int i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = 0; j < A->num_columns; j++, k++)
                            ydata[i] += alpha*Adata[k]*xdata[j];
                    }
                    if (num_flops) *num_flops += 3*A->num_entries;
                } else if (a->precision == mtx_double) {
                    const int64_t * Adata = a->data.integer_double;
                    const int64_t * xdata = xbase->data.integer_double;
                    int64_t * ydata = ybase->data.integer_double;
                    for (int i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = 0; j < A->num_columns; j++, k++)
                            ydata[i] += alpha*Adata[k]*xdata[j];
                    }
                    if (num_flops) *num_flops += 3*A->num_entries;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (trans == mtx_trans || trans == mtx_conjtrans) {
                if (a->precision == mtx_single) {
                    const int32_t * Adata = a->data.integer_single;
                    const int32_t * xdata = xbase->data.integer_single;
                    int32_t * ydata = ybase->data.integer_single;
                    for (int i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = 0; j < A->num_columns; j++, k++)
                            ydata[j] += alpha*Adata[k]*xdata[i];
                    }
                    if (num_flops) *num_flops += 3*A->num_entries;
                } else if (a->precision == mtx_double) {
                    const int64_t * Adata = a->data.integer_double;
                    const int64_t * xdata = xbase->data.integer_double;
                    int64_t * ydata = ybase->data.integer_double;
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
                const float * xdata = xbase->data.real_single;
                float * ydata = ybase->data.real_single;
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
                const double * xdata = xbase->data.real_double;
                double * ydata = ybase->data.real_double;
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
                    const float (* xdata)[2] = xbase->data.complex_single;
                    float (* ydata)[2] = ybase->data.complex_single;
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
                    const double (* xdata)[2] = xbase->data.complex_double;
                    double (* ydata)[2] = ybase->data.complex_double;
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
                    const float (* xdata)[2] = xbase->data.complex_single;
                    float (* ydata)[2] = ybase->data.complex_single;
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
                    const double (* xdata)[2] = xbase->data.complex_double;
                    double (* ydata)[2] = ybase->data.complex_double;
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
                const int32_t * xdata = xbase->data.integer_single;
                int32_t * ydata = ybase->data.integer_single;
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
                const int64_t * xdata = xbase->data.integer_double;
                int64_t * ydata = ybase->data.integer_double;
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
                    const float (* xdata)[2] = xbase->data.complex_single;
                    float (* ydata)[2] = ybase->data.complex_single;
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
                    const double (* xdata)[2] = xbase->data.complex_double;
                    double (* ydata)[2] = ybase->data.complex_double;
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
                    const float (* xdata)[2] = xbase->data.complex_single;
                    float (* ydata)[2] = ybase->data.complex_single;
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
                    const double (* xdata)[2] = xbase->data.complex_double;
                    double (* ydata)[2] = ybase->data.complex_double;
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
 * ‘mtxbasedense_cgemv()’ multiplies a complex-valued matrix ‘A’,
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
int mtxbasedense_cgemv(
    enum mtxtransposition trans,
    float alpha[2],
    const struct mtxbasedense * A,
    const struct mtxvector * x,
    float beta[2],
    struct mtxvector * y,
    int64_t * num_flops)
{
    const struct mtxbasevector * a = &A->a;
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

    if (beta[0] != 1 || beta[1] != 0) {
        int err = mtxvector_cscal(beta, y, num_flops);
        if (err) return err;
    }

    if (A->symmetry == mtx_unsymmetric) {
        if (trans == mtx_notrans) {
            if (a->precision == mtx_single) {
                const float (* Adata)[2] = a->data.complex_single;
                const float (* xdata)[2] = xbase->data.complex_single;
                float (* ydata)[2] = ybase->data.complex_single;
                for (int i = 0, k = 0; i < A->num_rows; i++) {
                    for (int j = 0; j < A->num_columns; j++, k++) {
                        ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[j][0]-Adata[k][1]*xdata[j][1])-alpha[1]*(Adata[k][0]*xdata[j][1]+Adata[k][1]*xdata[j][0]);
                        ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[j][1]+Adata[k][1]*xdata[j][0])+alpha[1]*(Adata[k][0]*xdata[j][0]-Adata[k][1]*xdata[j][1]);
                    }
                }
                if (num_flops) *num_flops += 14*A->num_entries;
            } else if (a->precision == mtx_double) {
                const double (* Adata)[2] = a->data.complex_double;
                const double (* xdata)[2] = xbase->data.complex_double;
                double (* ydata)[2] = ybase->data.complex_double;
                for (int i = 0, k = 0; i < A->num_rows; i++) {
                    for (int j = 0; j < A->num_columns; j++, k++) {
                        ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[j][0]-Adata[k][1]*xdata[j][1])-alpha[1]*(Adata[k][0]*xdata[j][1]+Adata[k][1]*xdata[j][0]);
                        ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[j][1]+Adata[k][1]*xdata[j][0])+alpha[1]*(Adata[k][0]*xdata[j][0]-Adata[k][1]*xdata[j][1]);
                    }
                }
                if (num_flops) *num_flops += 14*A->num_entries;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (trans == mtx_trans) {
            if (a->precision == mtx_single) {
                const float (* Adata)[2] = a->data.complex_single;
                const float (* xdata)[2] = xbase->data.complex_single;
                float (* ydata)[2] = ybase->data.complex_single;
                for (int i = 0, k = 0; i < A->num_rows; i++) {
                    for (int j = 0; j < A->num_columns; j++, k++) {
                        ydata[j][0] += alpha[0]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1])-alpha[1]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        ydata[j][1] += alpha[0]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0])+alpha[1]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                    }
                }
                if (num_flops) *num_flops += 14*A->num_entries;
            } else if (a->precision == mtx_double) {
                const double (* Adata)[2] = a->data.complex_double;
                const double (* xdata)[2] = xbase->data.complex_double;
                double (* ydata)[2] = ybase->data.complex_double;
                for (int i = 0, k = 0; i < A->num_rows; i++) {
                    for (int j = 0; j < A->num_columns; j++, k++) {
                        ydata[j][0] += alpha[0]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1])-alpha[1]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        ydata[j][1] += alpha[0]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0])+alpha[1]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                    }
                }
                if (num_flops) *num_flops += 14*A->num_entries;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (trans == mtx_conjtrans) {
            if (a->precision == mtx_single) {
                const float (* Adata)[2] = a->data.complex_single;
                const float (* xdata)[2] = xbase->data.complex_single;
                float (* ydata)[2] = ybase->data.complex_single;
                for (int i = 0, k = 0; i < A->num_rows; i++) {
                    for (int j = 0; j < A->num_columns; j++, k++) {
                        ydata[j][0] += alpha[0]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1])-alpha[1]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        ydata[j][1] += alpha[0]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0])+alpha[1]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                    }
                }
                if (num_flops) *num_flops += 14*A->num_entries;
            } else if (a->precision == mtx_double) {
                const double (* Adata)[2] = a->data.complex_double;
                const double (* xdata)[2] = xbase->data.complex_double;
                double (* ydata)[2] = ybase->data.complex_double;
                for (int i = 0, k = 0; i < A->num_rows; i++) {
                    for (int j = 0; j < A->num_columns; j++, k++) {
                        ydata[j][0] += alpha[0]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1])-alpha[1]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        ydata[j][1] += alpha[0]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0])+alpha[1]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                    }
                }
                if (num_flops) *num_flops += 14*A->num_entries;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else { return MTX_ERR_INVALID_TRANSPOSITION; }
    } else if (A->num_rows == A->num_columns && A->symmetry == mtx_symmetric) {
        if (trans == mtx_notrans || trans == mtx_trans) {
            if (a->precision == mtx_single) {
                const float (* Adata)[2] = a->data.complex_single;
                const float (* xdata)[2] = xbase->data.complex_single;
                float (* ydata)[2] = ybase->data.complex_single;
                for (int i = 0, k = 0; i < A->num_rows; i++) {
                    ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                    ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                    k++;
                    for (int j = i+1; j < A->num_columns; j++, k++) {
                        ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[j][0]-Adata[k][1]*xdata[j][1]) - alpha[1]*(Adata[k][0]*xdata[j][1]+Adata[k][1]*xdata[j][0]);
                        ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[j][1]+Adata[k][1]*xdata[j][0]) + alpha[1]*(Adata[k][0]*xdata[j][0]-Adata[k][1]*xdata[j][1]);
                        ydata[j][0] += alpha[0]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        ydata[j][1] += alpha[0]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                    }
                }
                if (num_flops) *num_flops += 14*A->num_entries;
            } else if (a->precision == mtx_double) {
                const double (* Adata)[2] = a->data.complex_double;
                const double (* xdata)[2] = xbase->data.complex_double;
                double (* ydata)[2] = ybase->data.complex_double;
                for (int i = 0, k = 0; i < A->num_rows; i++) {
                    ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                    ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                    k++;
                    for (int j = i+1; j < A->num_columns; j++, k++) {
                        ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[j][0]-Adata[k][1]*xdata[j][1]) - alpha[1]*(Adata[k][0]*xdata[j][1]+Adata[k][1]*xdata[j][0]);
                        ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[j][1]+Adata[k][1]*xdata[j][0]) + alpha[1]*(Adata[k][0]*xdata[j][0]-Adata[k][1]*xdata[j][1]);
                        ydata[j][0] += alpha[0]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        ydata[j][1] += alpha[0]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                    }
                }
                if (num_flops) *num_flops += 14*A->num_entries;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (trans == mtx_conjtrans) {
            if (a->precision == mtx_single) {
                const float (* Adata)[2] = a->data.complex_single;
                const float (* xdata)[2] = xbase->data.complex_single;
                float (* ydata)[2] = ybase->data.complex_single;
                for (int i = 0, k = 0; i < A->num_rows; i++) {
                    ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                    ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                    k++;
                    for (int j = i+1; j < A->num_columns; j++, k++) {
                        ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[j][0]+Adata[k][1]*xdata[j][1]) - alpha[1]*(Adata[k][0]*xdata[j][1]-Adata[k][1]*xdata[j][0]);
                        ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[j][1]-Adata[k][1]*xdata[j][0]) + alpha[1]*(Adata[k][0]*xdata[j][0]+Adata[k][1]*xdata[j][1]);
                        ydata[j][0] += alpha[0]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        ydata[j][1] += alpha[0]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                    }
                }
                if (num_flops) *num_flops += 14*A->num_entries;
            } else if (a->precision == mtx_double) {
                const double (* Adata)[2] = a->data.complex_double;
                const double (* xdata)[2] = xbase->data.complex_double;
                double (* ydata)[2] = ybase->data.complex_double;
                for (int i = 0, k = 0; i < A->num_rows; i++) {
                    ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                    ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                    k++;
                    for (int j = i+1; j < A->num_columns; j++, k++) {
                        ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[j][0]+Adata[k][1]*xdata[j][1]) - alpha[1]*(Adata[k][0]*xdata[j][1]-Adata[k][1]*xdata[j][0]);
                        ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[j][1]-Adata[k][1]*xdata[j][0]) + alpha[1]*(Adata[k][0]*xdata[j][0]+Adata[k][1]*xdata[j][1]);
                        ydata[j][0] += alpha[0]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        ydata[j][1] += alpha[0]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                    }
                }
                if (num_flops) *num_flops += 14*A->num_entries;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else { return MTX_ERR_INVALID_TRANSPOSITION; }
    } else if (A->num_rows == A->num_columns && A->symmetry == mtx_skew_symmetric) {
        /* TODO: allow skew-symmetric matrices */
        return MTX_ERR_INVALID_SYMMETRY;
    } else if (A->num_rows == A->num_columns && A->symmetry == mtx_hermitian) {
        if (trans == mtx_notrans || trans == mtx_conjtrans) {
            if (a->precision == mtx_single) {
                const float (* Adata)[2] = a->data.complex_single;
                const float (* xdata)[2] = xbase->data.complex_single;
                float (* ydata)[2] = ybase->data.complex_single;
                for (int i = 0, k = 0; i < A->num_rows; i++) {
                    ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                    ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                    k++;
                    for (int j = i+1; j < A->num_columns; j++, k++) {
                        ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[j][0]-Adata[k][1]*xdata[j][1]) - alpha[1]*(Adata[k][0]*xdata[j][1]+Adata[k][1]*xdata[j][0]);
                        ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[j][1]+Adata[k][1]*xdata[j][0]) + alpha[1]*(Adata[k][0]*xdata[j][0]-Adata[k][1]*xdata[j][1]);
                        ydata[j][0] += alpha[0]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        ydata[j][1] += alpha[0]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                    }
                }
                if (num_flops) *num_flops += 14*A->num_entries;
            } else if (a->precision == mtx_double) {
                const double (* Adata)[2] = a->data.complex_double;
                const double (* xdata)[2] = xbase->data.complex_double;
                double (* ydata)[2] = ybase->data.complex_double;
                for (int i = 0, k = 0; i < A->num_rows; i++) {
                    ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                    ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                    k++;
                    for (int j = i+1; j < A->num_columns; j++, k++) {
                        ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[j][0]-Adata[k][1]*xdata[j][1]) - alpha[1]*(Adata[k][0]*xdata[j][1]+Adata[k][1]*xdata[j][0]);
                        ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[j][1]+Adata[k][1]*xdata[j][0]) + alpha[1]*(Adata[k][0]*xdata[j][0]-Adata[k][1]*xdata[j][1]);
                        ydata[j][0] += alpha[0]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        ydata[j][1] += alpha[0]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                    }
                }
                if (num_flops) *num_flops += 14*A->num_entries;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (trans == mtx_trans) {
            if (a->precision == mtx_single) {
                const float (* Adata)[2] = a->data.complex_single;
                const float (* xdata)[2] = xbase->data.complex_single;
                float (* ydata)[2] = ybase->data.complex_single;
                for (int i = 0, k = 0; i < A->num_rows; i++) {
                    ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                    ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                    k++;
                    for (int j = i+1; j < A->num_columns; j++, k++) {
                        ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[j][0]+Adata[k][1]*xdata[j][1]) - alpha[1]*(Adata[k][0]*xdata[j][1]-Adata[k][1]*xdata[j][0]);
                        ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[j][1]-Adata[k][1]*xdata[j][0]) + alpha[1]*(Adata[k][0]*xdata[j][0]+Adata[k][1]*xdata[j][1]);
                        ydata[j][0] += alpha[0]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        ydata[j][1] += alpha[0]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                    }
                }
            } else if (a->precision == mtx_double) {
                const double (* Adata)[2] = a->data.complex_double;
                const double (* xdata)[2] = xbase->data.complex_double;
                double (* ydata)[2] = ybase->data.complex_double;
                for (int i = 0, k = 0; i < A->num_rows; i++) {
                    ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                    ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                    k++;
                    for (int j = i+1; j < A->num_columns; j++, k++) {
                        ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[j][0]+Adata[k][1]*xdata[j][1]) - alpha[1]*(Adata[k][0]*xdata[j][1]-Adata[k][1]*xdata[j][0]);
                        ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[j][1]-Adata[k][1]*xdata[j][0]) + alpha[1]*(Adata[k][0]*xdata[j][0]+Adata[k][1]*xdata[j][1]);
                        ydata[j][0] += alpha[0]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        ydata[j][1] += alpha[0]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                    }
                }
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else { return MTX_ERR_INVALID_TRANSPOSITION; }
    } else { return MTX_ERR_INVALID_SYMMETRY; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxbasedense_zgemv()’ multiplies a complex-valued matrix ‘A’,
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
int mtxbasedense_zgemv(
    enum mtxtransposition trans,
    double alpha[2],
    const struct mtxbasedense * A,
    const struct mtxvector * x,
    double beta[2],
    struct mtxvector * y,
    int64_t * num_flops)
{
    const struct mtxbasevector * a = &A->a;
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

    if (beta[0] != 1 || beta[1] != 0) {
        int err = mtxvector_zscal(beta, y, num_flops);
        if (err) return err;
    }

    if (A->symmetry == mtx_unsymmetric) {
        if (trans == mtx_notrans) {
            if (a->precision == mtx_single) {
                const float (* Adata)[2] = a->data.complex_single;
                const float (* xdata)[2] = xbase->data.complex_single;
                float (* ydata)[2] = ybase->data.complex_single;
                for (int i = 0, k = 0; i < A->num_rows; i++) {
                    for (int j = 0; j < A->num_columns; j++, k++) {
                        ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[j][0]-Adata[k][1]*xdata[j][1])-alpha[1]*(Adata[k][0]*xdata[j][1]+Adata[k][1]*xdata[j][0]);
                        ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[j][1]+Adata[k][1]*xdata[j][0])+alpha[1]*(Adata[k][0]*xdata[j][0]-Adata[k][1]*xdata[j][1]);
                    }
                }
                if (num_flops) *num_flops += 14*A->num_entries;
            } else if (a->precision == mtx_double) {
                const double (* Adata)[2] = a->data.complex_double;
                const double (* xdata)[2] = xbase->data.complex_double;
                double (* ydata)[2] = ybase->data.complex_double;
                for (int i = 0, k = 0; i < A->num_rows; i++) {
                    for (int j = 0; j < A->num_columns; j++, k++) {
                        ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[j][0]-Adata[k][1]*xdata[j][1])-alpha[1]*(Adata[k][0]*xdata[j][1]+Adata[k][1]*xdata[j][0]);
                        ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[j][1]+Adata[k][1]*xdata[j][0])+alpha[1]*(Adata[k][0]*xdata[j][0]-Adata[k][1]*xdata[j][1]);
                    }
                }
                if (num_flops) *num_flops += 14*A->num_entries;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (trans == mtx_trans) {
            if (a->precision == mtx_single) {
                const float (* Adata)[2] = a->data.complex_single;
                const float (* xdata)[2] = xbase->data.complex_single;
                float (* ydata)[2] = ybase->data.complex_single;
                for (int i = 0, k = 0; i < A->num_rows; i++) {
                    for (int j = 0; j < A->num_columns; j++, k++) {
                        ydata[j][0] += alpha[0]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1])-alpha[1]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        ydata[j][1] += alpha[0]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0])+alpha[1]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                    }
                }
                if (num_flops) *num_flops += 14*A->num_entries;
            } else if (a->precision == mtx_double) {
                const double (* Adata)[2] = a->data.complex_double;
                const double (* xdata)[2] = xbase->data.complex_double;
                double (* ydata)[2] = ybase->data.complex_double;
                for (int i = 0, k = 0; i < A->num_rows; i++) {
                    for (int j = 0; j < A->num_columns; j++, k++) {
                        ydata[j][0] += alpha[0]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1])-alpha[1]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        ydata[j][1] += alpha[0]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0])+alpha[1]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                    }
                }
                if (num_flops) *num_flops += 14*A->num_entries;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (trans == mtx_conjtrans) {
            if (a->precision == mtx_single) {
                const float (* Adata)[2] = a->data.complex_single;
                const float (* xdata)[2] = xbase->data.complex_single;
                float (* ydata)[2] = ybase->data.complex_single;
                for (int i = 0, k = 0; i < A->num_rows; i++) {
                    for (int j = 0; j < A->num_columns; j++, k++) {
                        ydata[j][0] += alpha[0]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1])-alpha[1]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        ydata[j][1] += alpha[0]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0])+alpha[1]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                    }
                }
                if (num_flops) *num_flops += 14*A->num_entries;
            } else if (a->precision == mtx_double) {
                const double (* Adata)[2] = a->data.complex_double;
                const double (* xdata)[2] = xbase->data.complex_double;
                double (* ydata)[2] = ybase->data.complex_double;
                for (int i = 0, k = 0; i < A->num_rows; i++) {
                    for (int j = 0; j < A->num_columns; j++, k++) {
                        ydata[j][0] += alpha[0]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1])-alpha[1]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        ydata[j][1] += alpha[0]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0])+alpha[1]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                    }
                }
                if (num_flops) *num_flops += 14*A->num_entries;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else { return MTX_ERR_INVALID_TRANSPOSITION; }
    } else if (A->num_rows == A->num_columns && A->symmetry == mtx_symmetric) {
        if (trans == mtx_notrans || trans == mtx_trans) {
            if (a->precision == mtx_single) {
                const float (* Adata)[2] = a->data.complex_single;
                const float (* xdata)[2] = xbase->data.complex_single;
                float (* ydata)[2] = ybase->data.complex_single;
                for (int i = 0, k = 0; i < A->num_rows; i++) {
                    ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                    ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                    k++;
                    for (int j = i+1; j < A->num_columns; j++, k++) {
                        ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[j][0]-Adata[k][1]*xdata[j][1]) - alpha[1]*(Adata[k][0]*xdata[j][1]+Adata[k][1]*xdata[j][0]);
                        ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[j][1]+Adata[k][1]*xdata[j][0]) + alpha[1]*(Adata[k][0]*xdata[j][0]-Adata[k][1]*xdata[j][1]);
                        ydata[j][0] += alpha[0]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        ydata[j][1] += alpha[0]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                    }
                }
                if (num_flops) *num_flops += 14*A->num_entries;
            } else if (a->precision == mtx_double) {
                const double (* Adata)[2] = a->data.complex_double;
                const double (* xdata)[2] = xbase->data.complex_double;
                double (* ydata)[2] = ybase->data.complex_double;
                for (int i = 0, k = 0; i < A->num_rows; i++) {
                    ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                    ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                    k++;
                    for (int j = i+1; j < A->num_columns; j++, k++) {
                        ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[j][0]-Adata[k][1]*xdata[j][1]) - alpha[1]*(Adata[k][0]*xdata[j][1]+Adata[k][1]*xdata[j][0]);
                        ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[j][1]+Adata[k][1]*xdata[j][0]) + alpha[1]*(Adata[k][0]*xdata[j][0]-Adata[k][1]*xdata[j][1]);
                        ydata[j][0] += alpha[0]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        ydata[j][1] += alpha[0]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                    }
                }
                if (num_flops) *num_flops += 14*A->num_entries;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (trans == mtx_conjtrans) {
            if (a->precision == mtx_single) {
                const float (* Adata)[2] = a->data.complex_single;
                const float (* xdata)[2] = xbase->data.complex_single;
                float (* ydata)[2] = ybase->data.complex_single;
                for (int i = 0, k = 0; i < A->num_rows; i++) {
                    ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                    ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                    k++;
                    for (int j = i+1; j < A->num_columns; j++, k++) {
                        ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[j][0]+Adata[k][1]*xdata[j][1]) - alpha[1]*(Adata[k][0]*xdata[j][1]-Adata[k][1]*xdata[j][0]);
                        ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[j][1]-Adata[k][1]*xdata[j][0]) + alpha[1]*(Adata[k][0]*xdata[j][0]+Adata[k][1]*xdata[j][1]);
                        ydata[j][0] += alpha[0]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        ydata[j][1] += alpha[0]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                    }
                }
                if (num_flops) *num_flops += 14*A->num_entries;
            } else if (a->precision == mtx_double) {
                const double (* Adata)[2] = a->data.complex_double;
                const double (* xdata)[2] = xbase->data.complex_double;
                double (* ydata)[2] = ybase->data.complex_double;
                for (int i = 0, k = 0; i < A->num_rows; i++) {
                    ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                    ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                    k++;
                    for (int j = i+1; j < A->num_columns; j++, k++) {
                        ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[j][0]+Adata[k][1]*xdata[j][1]) - alpha[1]*(Adata[k][0]*xdata[j][1]-Adata[k][1]*xdata[j][0]);
                        ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[j][1]-Adata[k][1]*xdata[j][0]) + alpha[1]*(Adata[k][0]*xdata[j][0]+Adata[k][1]*xdata[j][1]);
                        ydata[j][0] += alpha[0]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        ydata[j][1] += alpha[0]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                    }
                }
                if (num_flops) *num_flops += 14*A->num_entries;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else { return MTX_ERR_INVALID_TRANSPOSITION; }
    } else if (A->num_rows == A->num_columns && A->symmetry == mtx_skew_symmetric) {
        /* TODO: allow skew-symmetric matrices */
        return MTX_ERR_INVALID_SYMMETRY;
    } else if (A->num_rows == A->num_columns && A->symmetry == mtx_hermitian) {
        if (trans == mtx_notrans || trans == mtx_conjtrans) {
            if (a->precision == mtx_single) {
                const float (* Adata)[2] = a->data.complex_single;
                const float (* xdata)[2] = xbase->data.complex_single;
                float (* ydata)[2] = ybase->data.complex_single;
                for (int i = 0, k = 0; i < A->num_rows; i++) {
                    ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                    ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                    k++;
                    for (int j = i+1; j < A->num_columns; j++, k++) {
                        ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[j][0]-Adata[k][1]*xdata[j][1]) - alpha[1]*(Adata[k][0]*xdata[j][1]+Adata[k][1]*xdata[j][0]);
                        ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[j][1]+Adata[k][1]*xdata[j][0]) + alpha[1]*(Adata[k][0]*xdata[j][0]-Adata[k][1]*xdata[j][1]);
                        ydata[j][0] += alpha[0]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        ydata[j][1] += alpha[0]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                    }
                }
                if (num_flops) *num_flops += 14*A->num_entries;
            } else if (a->precision == mtx_double) {
                const double (* Adata)[2] = a->data.complex_double;
                const double (* xdata)[2] = xbase->data.complex_double;
                double (* ydata)[2] = ybase->data.complex_double;
                for (int i = 0, k = 0; i < A->num_rows; i++) {
                    ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                    ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                    k++;
                    for (int j = i+1; j < A->num_columns; j++, k++) {
                        ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[j][0]-Adata[k][1]*xdata[j][1]) - alpha[1]*(Adata[k][0]*xdata[j][1]+Adata[k][1]*xdata[j][0]);
                        ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[j][1]+Adata[k][1]*xdata[j][0]) + alpha[1]*(Adata[k][0]*xdata[j][0]-Adata[k][1]*xdata[j][1]);
                        ydata[j][0] += alpha[0]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        ydata[j][1] += alpha[0]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                    }
                }
                if (num_flops) *num_flops += 14*A->num_entries;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (trans == mtx_trans) {
            if (a->precision == mtx_single) {
                const float (* Adata)[2] = a->data.complex_single;
                const float (* xdata)[2] = xbase->data.complex_single;
                float (* ydata)[2] = ybase->data.complex_single;
                for (int i = 0, k = 0; i < A->num_rows; i++) {
                    ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                    ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                    k++;
                    for (int j = i+1; j < A->num_columns; j++, k++) {
                        ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[j][0]+Adata[k][1]*xdata[j][1]) - alpha[1]*(Adata[k][0]*xdata[j][1]-Adata[k][1]*xdata[j][0]);
                        ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[j][1]-Adata[k][1]*xdata[j][0]) + alpha[1]*(Adata[k][0]*xdata[j][0]+Adata[k][1]*xdata[j][1]);
                        ydata[j][0] += alpha[0]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        ydata[j][1] += alpha[0]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                    }
                }
            } else if (a->precision == mtx_double) {
                const double (* Adata)[2] = a->data.complex_double;
                const double (* xdata)[2] = xbase->data.complex_double;
                double (* ydata)[2] = ybase->data.complex_double;
                for (int i = 0, k = 0; i < A->num_rows; i++) {
                    ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                    ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                    k++;
                    for (int j = i+1; j < A->num_columns; j++, k++) {
                        ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[j][0]+Adata[k][1]*xdata[j][1]) - alpha[1]*(Adata[k][0]*xdata[j][1]-Adata[k][1]*xdata[j][0]);
                        ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[j][1]-Adata[k][1]*xdata[j][0]) + alpha[1]*(Adata[k][0]*xdata[j][0]+Adata[k][1]*xdata[j][1]);
                        ydata[j][0] += alpha[0]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        ydata[j][1] += alpha[0]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                    }
                }
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else { return MTX_ERR_INVALID_TRANSPOSITION; }
    } else { return MTX_ERR_INVALID_SYMMETRY; }
    return MTX_SUCCESS;
}

/*
 * Level 3 BLAS operations
 */

/**
 * ‘mtxbasedense_sgemm()’ multiplies a matrix ‘A’ (or its transpose ‘A'’)
 * by a real scalar ‘alpha’ (‘α’) and a matrix ‘B’ (or its transpose
 * ‘B'’), before adding the result to another matrix ‘C’ multiplied by
 * a real scalar ‘beta’ (‘β’). That is,
 *
 * ‘C = α*op(A)*op(B) + β*C’, where ‘op(X)=X’ or ‘op(X)=X'’.
 *
 * The scalars ‘alpha’ and ‘beta’ are given as single precision
 * floating point numbers.
 */
int mtxbasedense_sgemm(
    enum mtxtransposition Atrans,
    enum mtxtransposition Btrans,
    float alpha,
    const struct mtxbasedense * A,
    const struct mtxbasedense * B,
    float beta,
    struct mtxbasedense * C,
    int64_t * num_flops)
{
    const struct mtxbasevector * a = &A->a;
    const struct mtxbasevector * b = &B->a;
    struct mtxbasevector * c = &C->a;
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
        int err = mtxbasedense_sscal(beta, C, num_flops);
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
}

/**
 * ‘mtxbasedense_dgemm()’ multiplies a matrix ‘A’ (or its transpose ‘A'’)
 * by a real scalar ‘alpha’ (‘α’) and a matrix ‘B’ (or its transpose
 * ‘B'’), before adding the result to another matrix ‘C’ multiplied by
 * a real scalar ‘beta’ (‘β’). That is,
 *
 * ‘C = α*op(A)*op(B) + β*C’, where ‘op(X)=X’ or ‘op(X)=X'’.
 *
 * The scalars ‘alpha’ and ‘beta’ are given as double precision
 * floating point numbers.
 */
int mtxbasedense_dgemm(
    enum mtxtransposition Atrans,
    enum mtxtransposition Btrans,
    double alpha,
    const struct mtxbasedense * A,
    const struct mtxbasedense * B,
    double beta,
    struct mtxbasedense * C,
    int64_t * num_flops)
{
    int M, N, K;
    const struct mtxbasevector * a = &A->a;
    const struct mtxbasevector * b = &B->a;
    struct mtxbasevector * c = &C->a;
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
        int err = mtxbasedense_sscal(beta, C, num_flops);
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
}
