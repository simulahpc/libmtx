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
 * Last modified: 2022-05-04
 *
 * Data structures for dense matrices.
 */

#include <libmtx/libmtx-config.h>

#include <libmtx/error.h>
#include <libmtx/field.h>
#include <libmtx/precision.h>

#include <libmtx/matrix/dense.h>
#include <libmtx/matrix/matrix.h>
#include <libmtx/mtxfile/mtxfile.h>
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
 * ‘mtxmatrix_dense_free()’ frees storage allocated for a matrix.
 */
void mtxmatrix_dense_free(
    struct mtxmatrix_dense * A)
{
    mtxvector_base_free(&A->a);
}

/**
 * ‘mtxmatrix_dense_alloc_copy()’ allocates a copy of a matrix without
 * initialising the values.
 */
int mtxmatrix_dense_alloc_copy(
    struct mtxmatrix_dense * dst,
    const struct mtxmatrix_dense * src)
{
    return mtxmatrix_dense_alloc_entries(
        dst, src->a.field, src->a.precision, src->symmetry,
        src->num_rows, src->num_columns, src->size,
        0, 0, NULL, NULL);
}

/**
 * ‘mtxmatrix_dense_init_copy()’ allocates a copy of a matrix and also
 * copies the values.
 */
int mtxmatrix_dense_init_copy(
    struct mtxmatrix_dense * dst,
    const struct mtxmatrix_dense * src)
{
    int err = mtxmatrix_dense_alloc_copy(dst, src);
    if (err) return err;
    return mtxmatrix_dense_copy(dst, src);
}

/*
 * initialise matrices from entrywise data in coordinate format
 */

/**
 * ‘mtxmatrix_dense_alloc_entries()’ allocates a matrix from entrywise
 * data in coordinate format.
 */
int mtxmatrix_dense_alloc_entries(
    struct mtxmatrix_dense * A,
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
    return mtxvector_base_alloc(&A->a, field, precision, A->size);
}

/**
 * ‘mtxmatrix_dense_init_entries_real_single()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * real, single precision coefficients.
 */
int mtxmatrix_dense_init_entries_real_single(
    struct mtxmatrix_dense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx,
    const float * data)
{
    int err = mtxmatrix_dense_alloc_entries(
        A, mtx_field_real, mtx_single, symmetry, num_rows, num_columns,
        size, 0, 0, NULL, NULL);
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
    } else { mtxmatrix_dense_free(A); return MTX_ERR_INVALID_SYMMETRY; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_dense_init_entries_real_double()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * real, double precision coefficients.
 */
int mtxmatrix_dense_init_entries_real_double(
    struct mtxmatrix_dense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx,
    const double * data)
{
    int err = mtxmatrix_dense_alloc_entries(
        A, mtx_field_real, mtx_double, symmetry, num_rows, num_columns,
        size, 0, 0, NULL, NULL);
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
    } else { mtxmatrix_dense_free(A); return MTX_ERR_INVALID_SYMMETRY; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_dense_init_entries_complex_single()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * complex, single precision coefficients.
 */
int mtxmatrix_dense_init_entries_complex_single(
    struct mtxmatrix_dense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx,
    const float (* data)[2]);

/**
 * ‘mtxmatrix_dense_init_entries_complex_double()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * complex, double precision coefficients.
 */
int mtxmatrix_dense_init_entries_complex_double(
    struct mtxmatrix_dense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx,
    const double (* data)[2]);

/**
 * ‘mtxmatrix_dense_init_entries_integer_single()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * integer, single precision coefficients.
 */
int mtxmatrix_dense_init_entries_integer_single(
    struct mtxmatrix_dense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx,
    const int32_t * data);

/**
 * ‘mtxmatrix_dense_init_entries_integer_double()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * integer, double precision coefficients.
 */
int mtxmatrix_dense_init_entries_integer_double(
    struct mtxmatrix_dense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx,
    const int64_t * data);

/**
 * ‘mtxmatrix_dense_init_entries_pattern()’ allocates and initialises
 * a matrix from entrywise data in coordinate format with boolean
 * coefficients.
 */
int mtxmatrix_dense_init_entries_pattern(
    struct mtxmatrix_dense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx);

/*
 * initialise matrices from entrywise data in coordinate format with
 * specified strides
 */

/**
 * ‘mtxmatrix_dense_init_entries_strided_real_single()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * real, single precision coefficients.
 */
int mtxmatrix_dense_init_entries_strided_real_single(
    struct mtxmatrix_dense * A,
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
 * ‘mtxmatrix_dense_init_entries_strided_real_double()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * real, double precision coefficients.
 */
int mtxmatrix_dense_init_entries_strided_real_double(
    struct mtxmatrix_dense * A,
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
 * ‘mtxmatrix_dense_init_entries_strided_complex_single()’ allocates
 * and initialises a matrix from entrywise data in coordinate format
 * with complex, single precision coefficients.
 */
int mtxmatrix_dense_init_entries_strided_complex_single(
    struct mtxmatrix_dense * A,
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
 * ‘mtxmatrix_dense_init_entries_strided_complex_double()’ allocates
 * and initialises a matrix from entrywise data in coordinate format
 * with complex, double precision coefficients.
 */
int mtxmatrix_dense_init_entries_strided_complex_double(
    struct mtxmatrix_dense * A,
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
 * ‘mtxmatrix_dense_init_entries_strided_integer_single()’ allocates
 * and initialises a matrix from entrywise data in coordinate format
 * with integer, single precision coefficients.
 */
int mtxmatrix_dense_init_entries_strided_integer_single(
    struct mtxmatrix_dense * A,
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
 * ‘mtxmatrix_dense_init_entries_strided_integer_double()’ allocates
 * and initialises a matrix from entrywise data in coordinate format
 * with integer, double precision coefficients.
 */
int mtxmatrix_dense_init_entries_strided_integer_double(
    struct mtxmatrix_dense * A,
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
 * ‘mtxmatrix_dense_init_entries_strided_pattern()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * boolean coefficients.
 */
int mtxmatrix_dense_init_entries_strided_pattern(
    struct mtxmatrix_dense * A,
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
 * ‘mtxmatrix_dense_alloc_rows()’ allocates a matrix from row-wise
 * data in compressed row format.
 */
int mtxmatrix_dense_alloc_rows(
    struct mtxmatrix_dense * A,
    enum mtxfield field,
    enum mtxprecision precision,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx);

/**
 * ‘mtxmatrix_dense_init_rows_real_single()’ allocates and initialises
 * a matrix from row-wise data in compressed row format with real,
 * single precision coefficients.
 */
int mtxmatrix_dense_init_rows_real_single(
    struct mtxmatrix_dense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const float * data);

/**
 * ‘mtxmatrix_dense_init_rows_real_double()’ allocates and initialises
 * a matrix from row-wise data in compressed row format with real,
 * double precision coefficients.
 */
int mtxmatrix_dense_init_rows_real_double(
    struct mtxmatrix_dense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const double * data);

/**
 * ‘mtxmatrix_dense_init_rows_complex_single()’ allocates and
 * initialises a matrix from row-wise data in compressed row format
 * with complex, single precision coefficients.
 */
int mtxmatrix_dense_init_rows_complex_single(
    struct mtxmatrix_dense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const float (* data)[2]);

/**
 * ‘mtxmatrix_dense_init_rows_complex_double()’ allocates and
 * initialises a matrix from row-wise data in compressed row format
 * with complex, double precision coefficients.
 */
int mtxmatrix_dense_init_rows_complex_double(
    struct mtxmatrix_dense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const double (* data)[2]);

/**
 * ‘mtxmatrix_dense_init_rows_integer_single()’ allocates and
 * initialises a matrix from row-wise data in compressed row format
 * with integer, single precision coefficients.
 */
int mtxmatrix_dense_init_rows_integer_single(
    struct mtxmatrix_dense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const int32_t * data);

/**
 * ‘mtxmatrix_dense_init_rows_integer_double()’ allocates and
 * initialises a matrix from row-wise data in compressed row format
 * with integer, double precision coefficients.
 */
int mtxmatrix_dense_init_rows_integer_double(
    struct mtxmatrix_dense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const int64_t * data);

/**
 * ‘mtxmatrix_dense_init_rows_pattern()’ allocates and initialises a
 * matrix from row-wise data in compressed row format with boolean
 * coefficients.
 */
int mtxmatrix_dense_init_rows_pattern(
    struct mtxmatrix_dense * A,
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
 * ‘mtxmatrix_dense_alloc_columns()’ allocates a matrix from
 * column-wise data in compressed column format.
 */
int mtxmatrix_dense_alloc_columns(
    struct mtxmatrix_dense * A,
    enum mtxfield field,
    enum mtxprecision precision,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx);

/**
 * ‘mtxmatrix_dense_init_columns_real_single()’ allocates and
 * initialises a matrix from column-wise data in compressed column
 * format with real, single precision coefficients.
 */
int mtxmatrix_dense_init_columns_real_single(
    struct mtxmatrix_dense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx,
    const float * data);

/**
 * ‘mtxmatrix_dense_init_columns_real_double()’ allocates and
 * initialises a matrix from column-wise data in compressed column
 * format with real, double precision coefficients.
 */
int mtxmatrix_dense_init_columns_real_double(
    struct mtxmatrix_dense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx,
    const double * data);

/**
 * ‘mtxmatrix_dense_init_columns_complex_single()’ allocates and
 * initialises a matrix from column-wise data in compressed column
 * format with complex, single precision coefficients.
 */
int mtxmatrix_dense_init_columns_complex_single(
    struct mtxmatrix_dense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx,
    const float (* data)[2]);

/**
 * ‘mtxmatrix_dense_init_columns_complex_double()’ allocates and
 * initialises a matrix from column-wise data in compressed column
 * format with complex, double precision coefficients.
 */
int mtxmatrix_dense_init_columns_complex_double(
    struct mtxmatrix_dense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx,
    const double (* data)[2]);

/**
 * ‘mtxmatrix_dense_init_columns_integer_single()’ allocates and
 * initialises a matrix from column-wise data in compressed column
 * format with integer, single precision coefficients.
 */
int mtxmatrix_dense_init_columns_integer_single(
    struct mtxmatrix_dense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx,
    const int32_t * data);

/**
 * ‘mtxmatrix_dense_init_columns_integer_double()’ allocates and
 * initialises a matrix from column-wise data in compressed column
 * format with integer, double precision coefficients.
 */
int mtxmatrix_dense_init_columns_integer_double(
    struct mtxmatrix_dense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx,
    const int64_t * data);

/**
 * ‘mtxmatrix_dense_init_columns_pattern()’ allocates and initialises
 * a matrix from column-wise data in compressed column format with
 * boolean coefficients.
 */
int mtxmatrix_dense_init_columns_pattern(
    struct mtxmatrix_dense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx);

/*
 * initialise matrices from a list of dense cliques
 */

/**
 * ‘mtxmatrix_dense_alloc_cliques()’ allocates a matrix from a list of
 * dense cliques.
 */
int mtxmatrix_dense_alloc_cliques(
    struct mtxmatrix_dense * A,
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
 * ‘mtxmatrix_dense_init_cliques_real_single()’ allocates and
 * initialises a matrix from a list of dense cliques with real, single
 * precision coefficients.
 */
int mtxmatrix_dense_init_cliques_real_single(
    struct mtxmatrix_dense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_cliques,
    const int64_t * cliqueptr,
    const int * rowidx,
    const int * colidx,
    const float * data);

/**
 * ‘mtxmatrix_dense_init_cliques_real_double()’ allocates and
 * initialises a matrix from a list of dense cliques with real, double
 * precision coefficients.
 */
int mtxmatrix_dense_init_cliques_real_double(
    struct mtxmatrix_dense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_cliques,
    const int64_t * cliqueptr,
    const int * rowidx,
    const int * colidx,
    const double * data);

/**
 * ‘mtxmatrix_dense_init_cliques_complex_single()’ allocates and
 * initialises a matrix from a list of dense cliques with complex,
 * single precision coefficients.
 */
int mtxmatrix_dense_init_cliques_complex_single(
    struct mtxmatrix_dense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_cliques,
    const int64_t * cliqueptr,
    const int * rowidx,
    const int * colidx,
    const float (* data)[2]);

/**
 * ‘mtxmatrix_dense_init_cliques_complex_double()’ allocates and
 * initialises a matrix from a list of dense cliques with complex,
 * double precision coefficients.
 */
int mtxmatrix_dense_init_cliques_complex_double(
    struct mtxmatrix_dense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_cliques,
    const int64_t * cliqueptr,
    const int * rowidx,
    const int * colidx,
    const double (* data)[2]);

/**
 * ‘mtxmatrix_dense_init_cliques_integer_single()’ allocates and
 * initialises a matrix from a list of dense cliques with integer,
 * single precision coefficients.
 */
int mtxmatrix_dense_init_cliques_integer_single(
    struct mtxmatrix_dense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_cliques,
    const int64_t * cliqueptr,
    const int * rowidx,
    const int * colidx,
    const int32_t * data);

/**
 * ‘mtxmatrix_dense_init_cliques_integer_double()’ allocates and
 * initialises a matrix from a list of dense cliques with integer,
 * double precision coefficients.
 */
int mtxmatrix_dense_init_cliques_integer_double(
    struct mtxmatrix_dense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_cliques,
    const int64_t * cliqueptr,
    const int * rowidx,
    const int * colidx,
    const int64_t * data);

/**
 * ‘mtxmatrix_dense_init_cliques_pattern()’ allocates and initialises
 * a matrix from a list of dense cliques with boolean coefficients.
 */
int mtxmatrix_dense_init_cliques_pattern(
    struct mtxmatrix_dense * A,
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
 * ‘mtxmatrix_dense_setzero()’ sets every value of a matrix to zero.
 */
int mtxmatrix_dense_setzero(
    struct mtxmatrix_dense * A)
{
    return mtxvector_base_setzero(&A->a);
}

/**
 * ‘mtxmatrix_dense_set_real_single()’ sets values of a matrix based
 * on an array of single precision floating point numbers.
 */
int mtxmatrix_dense_set_real_single(
    struct mtxmatrix_dense * A,
    int64_t size,
    int stride,
    const float * a);

/**
 * ‘mtxmatrix_dense_set_real_double()’ sets values of a matrix based
 * on an array of double precision floating point numbers.
 */
int mtxmatrix_dense_set_real_double(
    struct mtxmatrix_dense * A,
    int64_t size,
    int stride,
    const double * a);

/**
 * ‘mtxmatrix_dense_set_complex_single()’ sets values of a matrix
 * based on an array of single precision floating point complex
 * numbers.
 */
int mtxmatrix_dense_set_complex_single(
    struct mtxmatrix_dense * A,
    int64_t size,
    int stride,
    const float (*a)[2]);

/**
 * ‘mtxmatrix_dense_set_complex_double()’ sets values of a matrix
 * based on an array of double precision floating point complex
 * numbers.
 */
int mtxmatrix_dense_set_complex_double(
    struct mtxmatrix_dense * A,
    int64_t size,
    int stride,
    const double (*a)[2]);

/**
 * ‘mtxmatrix_dense_set_integer_single()’ sets values of a matrix
 * based on an array of integers.
 */
int mtxmatrix_dense_set_integer_single(
    struct mtxmatrix_dense * A,
    int64_t size,
    int stride,
    const int32_t * a);

/**
 * ‘mtxmatrix_dense_set_integer_double()’ sets values of a matrix
 * based on an array of integers.
 */
int mtxmatrix_dense_set_integer_double(
    struct mtxmatrix_dense * A,
    int64_t size,
    int stride,
    const int64_t * a);

/*
 * row and column vectors
 */

/**
 * ‘mtxmatrix_dense_alloc_row_vector()’ allocates a row vector for a
 * given matrix, where a row vector is a vector whose length equal to
 * a single row of the matrix.
 */
int mtxmatrix_dense_alloc_row_vector(
    const struct mtxmatrix_dense * A,
    struct mtxvector * x,
    enum mtxvectortype vectortype);

/**
 * ‘mtxmatrix_dense_alloc_column_vector()’ allocates a column vector
 * for a given matrix, where a column vector is a vector whose length
 * equal to a single column of the matrix.
 */
int mtxmatrix_dense_alloc_column_vector(
    const struct mtxmatrix_dense * A,
    struct mtxvector * y,
    enum mtxvectortype vectortype);

/*
 * convert to and from Matrix Market format
 */

/**
 * ‘mtxmatrix_dense_from_mtxfile()’ converts a matrix from Matrix
 * Market format.
 */
int mtxmatrix_dense_from_mtxfile(
    struct mtxmatrix_dense * A,
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

    err = mtxmatrix_dense_alloc_entries(
        A, field, precision, symmetry, num_rows, num_columns,
        num_nonzeros, 0, 0, NULL, NULL);
    if (err) return err;
    err = mtxmatrix_dense_setzero(A);
    if (err) { mtxmatrix_dense_free(A); return err; }

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
                } else { mtxmatrix_dense_free(A); return MTX_ERR_INVALID_SYMMETRY; }
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
                } else { mtxmatrix_dense_free(A); return MTX_ERR_INVALID_SYMMETRY; }
            } else { mtxmatrix_dense_free(A); return MTX_ERR_INVALID_PRECISION; }
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
                } else { mtxmatrix_dense_free(A); return MTX_ERR_INVALID_SYMMETRY; }
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
                } else { mtxmatrix_dense_free(A); return MTX_ERR_INVALID_SYMMETRY; }
            } else { mtxmatrix_dense_free(A); return MTX_ERR_INVALID_PRECISION; }
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
                } else { mtxmatrix_dense_free(A); return MTX_ERR_INVALID_SYMMETRY; }
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
                } else { mtxmatrix_dense_free(A); return MTX_ERR_INVALID_SYMMETRY; }
            } else { mtxmatrix_dense_free(A); return MTX_ERR_INVALID_PRECISION; }
        } else { mtxmatrix_dense_free(A); return MTX_ERR_INVALID_MTX_FIELD; }
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
                } else { mtxmatrix_dense_free(A); return MTX_ERR_INVALID_SYMMETRY; }
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
                } else { mtxmatrix_dense_free(A); return MTX_ERR_INVALID_SYMMETRY; }
            } else { mtxmatrix_dense_free(A); return MTX_ERR_INVALID_PRECISION; }
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
                } else { mtxmatrix_dense_free(A); return MTX_ERR_INVALID_SYMMETRY; }
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
                } else { mtxmatrix_dense_free(A); return MTX_ERR_INVALID_SYMMETRY; }
            } else { mtxmatrix_dense_free(A); return MTX_ERR_INVALID_PRECISION; }
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
                } else { mtxmatrix_dense_free(A); return MTX_ERR_INVALID_SYMMETRY; }
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
                } else { mtxmatrix_dense_free(A); return MTX_ERR_INVALID_SYMMETRY; }
            } else { mtxmatrix_dense_free(A); return MTX_ERR_INVALID_PRECISION; }
        } else { mtxmatrix_dense_free(A); return MTX_ERR_INVALID_MTX_FIELD; }
    } else { mtxmatrix_dense_free(A); return MTX_ERR_INVALID_MTX_FORMAT; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_dense_to_mtxfile()’ converts a matrix to Matrix Market
 * format.
 */
int mtxmatrix_dense_to_mtxfile(
    struct mtxfile * mtxfile,
    const struct mtxmatrix_dense * A,
    int64_t num_rows,
    const int64_t * rowidx,
    int64_t num_columns,
    const int64_t * colidx,
    enum mtxfileformat mtxfmt)
{
    return MTX_SUCCESS;
}

/*
 * Level 1 BLAS operations
 */

/**
 * ‘mtxmatrix_dense_swap()’ swaps values of two matrices,
 * simultaneously performing ‘y <- x’ and ‘x <- y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_dense_swap(
    struct mtxmatrix_dense * x,
    struct mtxmatrix_dense * y);

/**
 * ‘mtxmatrix_dense_copy()’ copies values of a matrix, ‘y = x’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_dense_copy(
    struct mtxmatrix_dense * y,
    const struct mtxmatrix_dense * x)
{
    return mtxvector_base_copy(&y->a, &x->a);
}

/**
 * ‘mtxmatrix_dense_sscal()’ scales a matrix by a single precision
 * floating point scalar, ‘x = a*x’.
 */
int mtxmatrix_dense_sscal(
    float a,
    struct mtxmatrix_dense * x,
    int64_t * num_flops);

/**
 * ‘mtxmatrix_dense_dscal()’ scales a matrix by a double precision
 * floating point scalar, ‘x = a*x’.
 */
int mtxmatrix_dense_dscal(
    double a,
    struct mtxmatrix_dense * x,
    int64_t * num_flops);

/**
 * ‘mtxmatrix_dense_cscal()’ scales a matrix by a complex, single
 * precision floating point scalar, ‘x = (a+b*i)*x’.
 */
int mtxmatrix_dense_cscal(
    float a[2],
    struct mtxmatrix_dense * x,
    int64_t * num_flops);

/**
 * ‘mtxmatrix_dense_zscal()’ scales a matrix by a complex, double
 * precision floating point scalar, ‘x = (a+b*i)*x’.
 */
int mtxmatrix_dense_zscal(
    double a[2],
    struct mtxmatrix_dense * x,
    int64_t * num_flops);

/**
 * ‘mtxmatrix_dense_saxpy()’ adds a matrix to another one multiplied
 * by a single precision floating point value, ‘y = a*x + y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_dense_saxpy(
    float a,
    const struct mtxmatrix_dense * x,
    struct mtxmatrix_dense * y,
    int64_t * num_flops);

/**
 * ‘mtxmatrix_dense_daxpy()’ adds a matrix to another one multiplied
 * by a double precision floating point value, ‘y = a*x + y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_dense_daxpy(
    double a,
    const struct mtxmatrix_dense * x,
    struct mtxmatrix_dense * y,
    int64_t * num_flops);

/**
 * ‘mtxmatrix_dense_saypx()’ multiplies a matrix by a single precision
 * floating point scalar and adds another matrix, ‘y = a*y + x’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_dense_saypx(
    float a,
    struct mtxmatrix_dense * y,
    const struct mtxmatrix_dense * x,
    int64_t * num_flops);

/**
 * ‘mtxmatrix_dense_daypx()’ multiplies a matrix by a double precision
 * floating point scalar and adds another matrix, ‘y = a*y + x’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_dense_daypx(
    double a,
    struct mtxmatrix_dense * y,
    const struct mtxmatrix_dense * x,
    int64_t * num_flops);

/**
 * ‘mtxmatrix_dense_sdot()’ computes the Frobenius inner product of
 * two matrices in single precision floating point.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_dense_sdot(
    const struct mtxmatrix_dense * x,
    const struct mtxmatrix_dense * y,
    float * dot,
    int64_t * num_flops);

/**
 * ‘mtxmatrix_dense_ddot()’ computes the Frobenius inner product of
 * two matrices in double precision floating point.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_dense_ddot(
    const struct mtxmatrix_dense * x,
    const struct mtxmatrix_dense * y,
    double * dot,
    int64_t * num_flops);

/**
 * ‘mtxmatrix_dense_cdotu()’ computes the product of the transpose of
 * a complex row matrix with another complex row matrix in single
 * precision floating point, ‘dot := x^T*y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_dense_cdotu(
    const struct mtxmatrix_dense * x,
    const struct mtxmatrix_dense * y,
    float (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxmatrix_dense_zdotu()’ computes the product of the transpose of
 * a complex row matrix with another complex row matrix in double
 * precision floating point, ‘dot := x^T*y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_dense_zdotu(
    const struct mtxmatrix_dense * x,
    const struct mtxmatrix_dense * y,
    double (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxmatrix_dense_cdotc()’ computes the Frobenius inner product of
 * two complex matrices in single precision floating point, ‘dot :=
 * x^H*y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_dense_cdotc(
    const struct mtxmatrix_dense * x,
    const struct mtxmatrix_dense * y,
    float (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxmatrix_dense_zdotc()’ computes the Frobenius inner product of
 * two complex matrices in double precision floating point, ‘dot :=
 * x^H*y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_dense_zdotc(
    const struct mtxmatrix_dense * x,
    const struct mtxmatrix_dense * y,
    double (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxmatrix_dense_snrm2()’ computes the Frobenius norm of a matrix
 * in single precision floating point.
 */
int mtxmatrix_dense_snrm2(
    const struct mtxmatrix_dense * x,
    float * nrm2,
    int64_t * num_flops);

/**
 * ‘mtxmatrix_dense_dnrm2()’ computes the Frobenius norm of a matrix
 * in double precision floating point.
 */
int mtxmatrix_dense_dnrm2(
    const struct mtxmatrix_dense * x,
    double * nrm2,
    int64_t * num_flops);

/**
 * ‘mtxmatrix_dense_sasum()’ computes the sum of absolute values
 * (1-norm) of a matrix in single precision floating point.  If the
 * matrix is complex-valued, then the sum of the absolute values of
 * the real and imaginary parts is computed.
 */
int mtxmatrix_dense_sasum(
    const struct mtxmatrix_dense * x,
    float * asum,
    int64_t * num_flops);

/**
 * ‘mtxmatrix_dense_dasum()’ computes the sum of absolute values
 * (1-norm) of a matrix in double precision floating point.  If the
 * matrix is complex-valued, then the sum of the absolute values of
 * the real and imaginary parts is computed.
 */
int mtxmatrix_dense_dasum(
    const struct mtxmatrix_dense * x,
    double * asum,
    int64_t * num_flops);

/**
 * ‘mtxmatrix_dense_iamax()’ finds the index of the first element
 * having the maximum absolute value.  If the matrix is
 * complex-valued, then the index points to the first element having
 * the maximum sum of the absolute values of the real and imaginary
 * parts.
 */
int mtxmatrix_dense_iamax(
    const struct mtxmatrix_dense * x,
    int * iamax);

/*
 * Level 2 BLAS operations (matrix-vector)
 */

/**
 * ‘mtxmatrix_dense_sgemv()’ multiplies a matrix ‘A’ or its transpose
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
int mtxmatrix_dense_sgemv(
    enum mtxtransposition trans,
    float alpha,
    const struct mtxmatrix_dense * A,
    const struct mtxvector * x,
    float beta,
    struct mtxvector * y,
    int64_t * num_flops);

/**
 * ‘mtxmatrix_dense_dgemv()’ multiplies a matrix ‘A’ or its transpose
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
int mtxmatrix_dense_dgemv(
    enum mtxtransposition trans,
    double alpha,
    const struct mtxmatrix_dense * A,
    const struct mtxvector * x,
    double beta,
    struct mtxvector * y,
    int64_t * num_flops);

/**
 * ‘mtxmatrix_dense_cgemv()’ multiplies a complex-valued matrix ‘A’,
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
int mtxmatrix_dense_cgemv(
    enum mtxtransposition trans,
    float alpha[2],
    const struct mtxmatrix_dense * A,
    const struct mtxvector * x,
    float beta[2],
    struct mtxvector * y,
    int64_t * num_flops);

/**
 * ‘mtxmatrix_dense_zgemv()’ multiplies a complex-valued matrix ‘A’,
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
int mtxmatrix_dense_zgemv(
    enum mtxtransposition trans,
    double alpha[2],
    const struct mtxmatrix_dense * A,
    const struct mtxvector * x,
    double beta[2],
    struct mtxvector * y,
    int64_t * num_flops);
