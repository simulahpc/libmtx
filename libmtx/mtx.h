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

#ifndef LIBMTX_MTX_H
#define LIBMTX_MTX_H

#include <libmtx/assembly.h>
#include <libmtx/mtx/header.h>
#include <libmtx/mtx/reorder.h>
#include <libmtx/mtx/sort.h>
#include <libmtx/triangle.h>

#include <stdbool.h>
#include <stdint.h>

struct mtx_index_set;

/**
 * `mtx` is a data structure for objects (vectors or matrices) in the
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
     * `triangle' specifies triangular properties of a matrix:
     * `nontriangular', `lower-triangular', `upper-triangular' or
     * `diagonal'.
     *
     * For symmetric, dense matrices in array format, `triangle' is
     * `lower-triangular' if the lower triangular part of the matrix
     * is stored, and `upper-triangular' if the upper triangular part
     * of the matrix is stored.
     *
     * Note that the triangular properties of a matrix are not
     * explicitly stored in a Matrix Market file, but it is useful
     * additional data that can be provided by the user.
     */
    enum mtx_triangle triangle;

    /**
     * `sorting' is the sorting of matrix nonzeros: `unsorted',
     * 'row-major' or 'column-major'.
     *
     * Note that the sorting is not explicitly stored in a Matrix
     * Market file, but it is useful additional data that can be
     * provided by the user.
     */
    enum mtx_sorting sorting;

    /**
     * `ordering' is the matrix ordering: `unordered' or `rcm'.
     *
     * Note that the ordering is not explicitly stored in a Matrix
     * Market file, but it is useful additional data that can be
     * provided by the user.
     */
    enum mtx_ordering ordering;

    /**
     * `assembly' is the matrix assembly state: `unassembled' or
     * `assembled'.
     *
     * An unassembled sparse matrix may contain more than one value
     * associated with each nonzero matrix entry. In contrast, there
     * is only one value associated with each nonzero matrix entry of
     * an assembled sparse matrix.
     *
     * Note that the assembly state is not explicitly stored in a
     * Matrix Market file, but it is useful additional data that can
     * be provided by the user.
     */
    enum mtx_assembly assembly;

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
     * `num_nonzeros` is the number of nonzero matrix entries,
     * including entries that are not stored explicitly, for example,
     * due to symmetry.
     *
     * The number of nonzeros depends on the matrix `format':
     *
     * If `format' is `array', then `num_nonzeros' is equal to
     * `num_rows*num_columns'. Otherwise, the value is a non-negative
     * integer less than or equal to `num_rows*num_columns'.
     */
    int64_t num_nonzeros;

    /**
     * `size` is the number of nonzero matrix entries stored in the
     * `data` array.
     *
     * The number of stored nonzeros depends on the matrix `format`
     * and `symmetry`:
     *
     * - If `symmetry` is `general`, then `size` is the number of
     *   nonzero entries. Whenever `format` is `array`, then `size` is
     *   equal to `num_rows*num_columns`.
     *
     * - If `symmetry` is `symmetric` or `hermitian`, then `size` is
     *   the number of nonzero entries on or below the
     *   diagonal. Whenever `format` is `array`, then `size' is equal
     *   to `(n+1)*n/2', where `n' is equal to `num_rows' and
     *   `num_columns'.
     *
     * - If `symmetry` is `skew-symmetric`, then `size` is the number
     *   of nonzero entries below the diagonal. Whenever `format` is
     *   `array`, then `size' is equal to `n*n/2', where `n' is equal
     *   to `num_rows' and `num_columns'.
     */
    int64_t size;

    /**
     * `nonzero_size' is the size (in bytes) of each nonzero entry in
     * the `data' array.
     *
     * The size of each nonzero depends on the `format' and `field':
     *
     *   - If `format` is `array' and `field' is `real', `double',
     *     `complex' or `integer', then `nonzero_size' is
     *     `sizeof(float)', `sizeof(double)', `2*sizeof(float)', or
     *     `sizeof(int)', respectively.
     *
     *   - If `format' is `coordinate', then `nonzero_size' is equal
     *     to the size of the corresponding struct
     *     `mtx_<object>_coordinate_<field>', where `<object>' is
     *     `matrix' or `vector' and `<field>' is `real', `double',
     *     `complex', `integer' or `pattern'.
     */
    int nonzero_size;

    /**
     * `data` contains data for the nonzero matrix entries.
     *
     * The storage format of nonzero values depends on the matrix
     * `format` and `field`:
     *
     *   - If `format` is `array` and `field` is `real`, `double` or
     *     `integer`, then `data` is an array of `size` values of type
     *     `float`, `double` or `int`, respectively.
     *
     *   - If `format` is `array` and `field` is `complex`, then
     *     `data` is an array of `2*size` values of type `float`.
     *
     *   - If `object' is `matrix', `format` is `coordinate` and
     *     `field` is `real`, `double`, `complex`, `integer` or
     *     `pattern`, then `data` is an array of `size` values of type
     *     `mtx_matrix_coordinate_real`,
     *     `mtx_matrix_coordinate_double`,
     *     `mtx_matrix_coordinate_complex`,
     *     `mtx_matrix_coordinate_integer` or
     *     `mtx_matrix_coordinate_pattern`, respectively.
     *
     *   - If `object' is `vector', `format` is `coordinate` and
     *     `field` is `real`, `double`, `complex`, `integer` or
     *     `pattern`, then `data` is an array of `size` values of type
     *     `mtx_vector_coordinate_real`,
     *     `mtx_vector_coordinate_double`,
     *     `mtx_vector_coordinate_complex`,
     *     `mtx_vector_coordinate_integer` or
     *     `mtx_vector_coordinate_pattern`, respectively.
     */
    void * data;
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
 * `mtx_set_constant_real()' sets every (nonzero) value of a matrix or
 * vector equal to a constant, single precision floating point number.
 */
int mtx_set_constant_real(
    struct mtx * mtx,
    float a);

/**
 * `mtx_set_constant_double()' sets every (nonzero) value of a matrix
 * or vector equal to a constant, double precision floating point
 * number.
 */
int mtx_set_constant_double(
    struct mtx * mtx,
    double a);

/**
 * `mtx_set_constant_complex()' sets every (nonzero) value of a matrix
 * or vector equal to a constant, single precision floating point
 * complex number.
 */
int mtx_set_constant_complex(
    struct mtx * mtx,
    float a,
    float b);

/**
 * `mtx_set_constant_integer()' sets every (nonzero) value of a matrix
 * or vector equal to a constant integer.
 */
int mtx_set_constant_integer(
    struct mtx * mtx,
    int a);

#endif
