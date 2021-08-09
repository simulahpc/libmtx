/* This file is part of libmtx.
 *
 * Copyright (C) 2021 James D. Trotter
 *
 * libmtx is free software: you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * libmtx is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with libmtx.  If not, see
 * <https://www.gnu.org/licenses/>.
 *
 * Authors: James D. Trotter <james@simula.no>
 * Last modified: 2021-08-09
 *
 * Various operations for matrices in the Matrix Market format.
 */

#ifndef LIBMTX_MATRIX_H
#define LIBMTX_MATRIX_H

#include <libmtx/mtx/header.h>

#include <stdbool.h>
#include <stdint.h>

struct mtx_index_set;
struct mtx;

/**
 * `mtx_matrix_row_index()` retrieves the row index for a given
 * nonzero of a matrix.
 */
int mtx_matrix_row_index(
    const struct mtx * mtx,
    int64_t k,
    int * row);

/**
 * `mtx_matrix_column_index()` retrieves the column index for a given
 * nonzero of a matrix.
 */
int mtx_matrix_column_index(
    const struct mtx * mtx,
    int64_t k,
    int * column);

/**
 * `mtx_matrix_set_zero()' zeroes a matrix.
 */
int mtx_matrix_set_zero(
    struct mtx * mtx);

/**
 * `mtx_matrix_set_constant_real()' sets every (nonzero) value of a
 * matrix equal to a constant, single precision floating point number.
 */
int mtx_matrix_set_constant_real(
    struct mtx * mtx,
    float a);

/**
 * `mtx_matrix_set_constant_double()' sets every (nonzero) value of a
 * matrix equal to a constant, double precision floating point number.
 */
int mtx_matrix_set_constant_double(
    struct mtx * mtx,
    double a);

/**
 * `mtx_matrix_set_constant_complex()' sets every (nonzero) value of a
 * matrix equal to a constant, single precision floating point complex
 * number.
 */
int mtx_matrix_set_constant_complex(
    struct mtx * mtx,
    float a,
    float b);

/**
 * `mtx_matrix_set_constant_integer()' sets every (nonzero) value of a
 * matrix equal to a constant integer.
 */
int mtx_matrix_set_constant_integer(
    struct mtx * mtx,
    int a);

/**
 * `mtx_matrix_num_nonzeros()` computes the number of nonzeros,
 * including, in the case of a matrix, any nonzeros that are not
 * stored explicitly due to symmetry.
 */
int mtx_matrix_num_nonzeros(
    enum mtx_object object,
    enum mtx_format format,
    enum mtx_field field,
    enum mtx_symmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const void * data,
    int64_t * num_nonzeros);

/**
 * `mtx_matrix_num_diagonal_nonzeros()` counts the number of nonzeros
 * on the main diagonal of a matrix in the Matrix Market format.
 */
int mtx_matrix_num_diagonal_nonzeros(
    const struct mtx * matrix,
    int64_t * num_diagonal_nonzeros);

/**
 * `mtx_matrix_size_per_row()' counts the number of entries stored for
 * each row of a matrix.
 *
 * The array `size_per_row' must point to an array containing enough
 * storage for `mtx->num_rows' values of type `int'.
 */
int mtx_matrix_size_per_row(
    const struct mtx * mtx,
    int * size_per_row);

/**
 * `mtx_matrix_column_ptr()' computes column pointers of a matrix.
 *
 * The array `column_ptr' must point to an array containing enough
 * storage for `mtx->num_columns+1' values of type `int64_t'.
 *
 * The matrix is not required to be sorted in column major order.  If
 * the matrix is sorted in column major order, then the `i'-th entry
 * of the `column_ptr' is the location of the first nonzero in the
 * `mtx->data' array that belongs to the `i+1'-th column of the
 * matrix, for `i=0,1,...,mtx->num_columns-1'. The final entry of
 * `column_ptr' indicates the position one place beyond the last
 * nonzero in `mtx->data'.
 */
int mtx_matrix_column_ptr(
    const struct mtx * mtx,
    int64_t * column_ptr);

/**
 * `mtx_matrix_row_ptr()' computes row pointers of a matrix.
 *
 * The array `row_ptr' must point to an array containing enough
 * storage for `mtx->num_rows+1' values of type `int64_t'.
 *
 * The matrix is not required to be sorted in row major order.  If the
 * matrix is sorted in row major order, then the `i'-th entry of the
 * `row_ptr' is the location of the first nonzero in the `mtx->data'
 * array that belongs to the `i+1'-th row of the matrix, for
 * `i=0,1,...,mtx->num_rows-1'. The final entry of `row_ptr' indicates
 * the position one place beyond the last nonzero in `mtx->data'.
 */
int mtx_matrix_row_ptr(
    const struct mtx * mtx,
    int64_t * row_ptr);

/**
 * `mtx_matrix_column_indices()' extracts the column indices of a
 * matrix to a separate array.
 *
 * The array `column_indices' must point to an array containing enough
 * storage for `mtx->size' values of type `int'.
 */
int mtx_matrix_column_indices(
    const struct mtx * mtx,
    int * column_indices);

/**
 * `mtx_matrix_row_indices()' extracts the row indices of a matrix to
 * a separate array.
 *
 * The array `row_indices' must point to an array containing enough
 * storage for `mtx->size' values of type `int'.
 */
int mtx_matrix_row_indices(
    const struct mtx * mtx,
    int * row_indices);

/**
 * `mtx_matrix_data_real()' extracts the nonzero values of a real
 * matrix to a separate array of single precision floating point
 * values.
 *
 * The array `data' must point to an array containing enough storage
 * for `mtx->size' values of type `float'.
 */
int mtx_matrix_data_real(
    const struct mtx * mtx,
    float * data);

/**
 * `mtx_matrix_data_double()' extracts the nonzero values of a double
 * matrix to a separate array of double precision floating point
 * values.
 *
 * The array `data' must point to an array containing enough storage
 * for `mtx->size' values of type `double'.
 */
int mtx_matrix_data_double(
    const struct mtx * mtx,
    double * data);

/**
 * `mtx_matrix_data_complex()' extracts the nonzero values of a
 * complex matrix to a separate array of single precision floating
 * point values.
 *
 * The array `data' must point to an array containing enough storage
 * for `2*mtx->size' values of type `float'.
 */
int mtx_matrix_data_complex(
    const struct mtx * mtx,
    float * data);

/**
 * `mtx_matrix_data_integer()' extracts the nonzero values of an
 * integer matrix to a separate array of integers.
 *
 * The array `data' must point to an array containing enough storage
 * for `mtx->size' values of type `int'.
 */
int mtx_matrix_data_integer(
    const struct mtx * mtx,
    int * data);

/**
 * `mtx_matrix_diagonal_size_per_row()` counts for each row of a
 * matrix the number of stored nonzero entries on the diagonal.
 *
 * The array `diagonal_size_per_row' must point to an array containing
 * enough storage for `mtx->num_rows' values of type `int'.
 */
int mtx_matrix_diagonal_size_per_row(
    const struct mtx * mtx,
    int * diagonal_size_per_row);

/**
 * `mtx_matrix_submatrix()` obtains a submatrix consisting of the
 * given rows and columns.
 */
int mtx_matrix_submatrix(
    const struct mtx * mtx,
    const struct mtx_index_set * rows,
    const struct mtx_index_set * columns,
    struct mtx * submatrix);

/**
 * `mtx_matrix_transpose()` transposes a square matrix.
 */
int mtx_matrix_transpose(
    struct mtx * matrix);

#endif
