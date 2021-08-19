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
 * Data structures for representing objects in Matrix Market format.
 */

#ifndef LIBMTX_MTX_MTX_H
#define LIBMTX_MTX_MTX_H

#include <libmtx/mtx/header.h>

#include <libmtx/matrix/array/data.h>
#include <libmtx/matrix/coordinate/data.h>
#include <libmtx/vector/array/data.h>
#include <libmtx/vector/coordinate/data.h>

#include <stdint.h>

/**
 * `mtx' is a data structure for objects (vectors or matrices) in the
 * Matrix Market format.
 */
struct mtx
{
    /**
     * `object` is the type of Matrix Market object: `matrix` or
     * `vector`.
     */
    enum mtx_object object;

    /**
     * `format` is the matrix format: `coordinate` or `array`.
     */
    enum mtx_format format;

    /**
     * `field` is the matrix field: `real`, `double`, `complex`,
     * `integer` or `pattern`.
     */
    enum mtx_field field;

    /**
     * `symmetry` is the matrix symmetry: `general`, `symmetric`,
     * `skew-symmetric`, or `hermitian`.
     *
     * Note that if `symmetry' is `symmetric', `skew-symmetric' or
     * `hermitian', then the matrix must be square, so that `num_rows'
     * is equal to `num_columns'.
     */
    enum mtx_symmetry symmetry;

    /**
     * `num_comment_lines` is the number of comment lines.
     */
    int num_comment_lines;

    /**
     * `comment_lines` is an array containing comment lines.
     */
    char ** comment_lines;

    /**
     * `num_rows` is the number of rows in the matrix or vector.
     */
    int num_rows;

    /**
     * `num_columns` is the number of columns in the matrix if
     * `object' is `matrix'. Otherwise, if `object' is `vector', then
     * `num_columns' is equal to `-1'.
     */
    int num_columns;

    /**
     * `num_nonzeros' is the number of nonzero matrix or vector
     * entries for a sparse matrix or vector.  This only includes
     * entries that are stored explicitly, and not those that are
     * implicitly, for example, due to symmetry.
     *
     * If `format' is `array', then `num_nonzeros' is set to `-1', and
     * it is not used.
     */
    int64_t num_nonzeros;

    /**
     * `storage' contains data for the (nonzero) matrix or vector
     * entries.
     *
     * The storage format of the matrix or vector data depends on
     * `object' and `format'.  Only the member of the `storage' union
     * that corresponds to the matrix (or vector) `object' and
     * `format' should be used to access the data.
     *
     * For example, if `object' is `matrix' and `format' is `array',
     * then `data.matrix_array' is used to store the matrix entries in
     * the array format.
     */
    union {
        struct mtx_matrix_array_data matrix_array;
        struct mtx_matrix_coordinate_data matrix_coordinate;
        struct mtx_vector_array_data vector_array;
        struct mtx_vector_coordinate_data vector_coordinate;
    } storage;
};

/**
 * `mtx_free()` frees resources associated with a Matrix Market
 * object.
 */
void mtx_free(
    struct mtx * mtx);

/**
 * `mtx_copy()' copies a matrix or vector.
 */
int mtx_copy(
    struct mtx * dstmtx,
    const struct mtx * srcmtx);

/**
 * `mtx_set_comment_lines()' copies comment lines to a Matrix Market
 * object.
 *
 * Any storage associated with existing comment lines is freed.
 */
int mtx_set_comment_lines(
    struct mtx * mtx,
    int  num_comment_lines,
    const char ** comment_lines);

/**
 * `mtx_set_zero()' zeroes a matrix or vector.
 */
int mtx_set_zero(
    struct mtx * mtx);

/**
 * `mtx_set_constant_real_single()' sets every (nonzero) value of a
 * matrix or vector equal to a constant, single precision floating
 * point number.
 */
int mtx_set_constant_real_single(
    struct mtx * mtx,
    float a);

/**
 * `mtx_set_constant_real_double()' sets every (nonzero) value of a
 * matrix or vector equal to a constant, double precision floating
 * point number.
 */
int mtx_set_constant_real_double(
    struct mtx * mtx,
    double a);

/**
 * `mtx_set_constant_complex_single()' sets every (nonzero) value of a
 * matrix or vector equal to a constant, single precision floating
 * point complex number.
 */
int mtx_set_constant_complex_single(
    struct mtx * mtx,
    float a[2]);

/**
 * `mtx_set_constant_integer_single()' sets every (nonzero) value of a
 * matrix or vector equal to a constant integer.
 */
int mtx_set_constant_integer_single(
    struct mtx * mtx,
    int32_t a);

#endif
