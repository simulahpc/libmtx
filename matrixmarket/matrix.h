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
 * Last modified: 2021-06-18
 *
 * Various operations for matrices in the Matrix Market format.
 */

#ifndef MATRIXMARKET_MATRIX_H
#define MATRIXMARKET_MATRIX_H

#include <matrixmarket/header.h>

#include <stdbool.h>
#include <stdint.h>

struct mtx_index_set;
struct mtx;

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
 * `mtx_matrix_nonzeros_per_row()` counts the number of nonzeros in
 * each row of a matrix in the Matrix Market format.
 *
 * If `include_strict_upper_triangular_part` is `true` and `symmetry`
 * is `symmetric`, `skew-symmetric` or `hermitian`, then nonzeros in
 * the strict upper triangular part are also counted. Conversely, if
 * `include_strict_upper_triangular_part` is `false`, then only
 * nonzeros in the lower triangular part of the matrix are counted.
 *
 * `mtx_matrix_nonzeros_per_row()` returns `MTX_ERR_ERRNO' with
 * `errno' set to `EINVAL' if `symmetry` is `general` and
 * `include_strict_upper_triangular_part` is `false`.
 */
int mtx_matrix_nonzeros_per_row(
    const struct mtx * matrix,
    bool include_strict_upper_triangular_part,
    int64_t * nonzeros_per_row);

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
