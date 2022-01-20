/* This file is part of Libmtx.
 *
 * Copyright (C) 2021 James D. Trotter
 *
 * Libmtx is free software: you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * Libmtx is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Libmtx.  If not, see
 * <https://www.gnu.org/licenses/>.
 *
 * Authors: James D. Trotter <james@simula.no>
 * Last modified: 2021-08-16
 *
 * Data structures for matrices in array format.
 */

#ifndef LIBMTX_MATRIX_ARRAY_DATA_H
#define LIBMTX_MATRIX_ARRAY_DATA_H

#include <libmtx/mtx/header.h>
#include <libmtx/precision.h>
#include <libmtx/mtx/sort.h>
#include <libmtx/mtx/triangle.h>

#include <stdint.h>

/**
 * `mtx_matrix_array_data' is a data structure for representing data
 * associated with matrices in array format.
 */
struct mtx_matrix_array_data
{
    /**
     * `field' is the field associated with the matrix values: `real',
     * `complex' or `integer'.
     */
    enum mtx_field field;

    /**
     * `precision' is the precision associated with the matrix values:
     * `single' or `double'.
     */
    enum mtxprecision precision;

    /**
     * `symmetry' is the matrix symmetry: `general', `symmetric',
     * `skew-symmetric', or `hermitian'.
     *
     * Note that if `symmetry' is `symmetric', `skew-symmetric' or
     * `hermitian', then the matrix must be square, so that `num_rows'
     * is equal to `num_columns'.
     */
    enum mtx_symmetry symmetry;

    /**
     * `triangle' specifies triangular properties of a matrix:
     * `mtx_nontriangular', `mtx_lower_triangular',
     * `mtx_upper_triangular', `mtx_strict_lower_triangular' or
     * `mtx_strict_upper_triangular'.
     *
     * For symmetric or Hermitian matrices, `triangle' is
     * `mtx_lower_triangular' if the lower triangular part of the
     * matrix is stored, or `mtx_upper_triangular' if the upper
     * triangular part of the matrix is stored.
     *
     * For skew-symmetric matrices, `triangle' is
     * `mtx_strict_lower_triangular' if the strict lower triangular
     * part of the matrix is stored, or `mtx_strict_upper_triangular'
     * if the strict upper triangular part of the matrix is stored.
     *
     * Otherwise, `triangle' is `mtx_nontriangular'.
     *
     * Note that the triangular properties of a matrix are not
     * explicitly stored in a Matrix Market file, but it is useful
     * additional data that can be provided by the user.
     */
    enum mtx_triangle triangle;

    /**
     * `sorting' is the sorting of matrix nonzeros: 'mtx_row_major' or
     * 'mtx_column_major'.
     *
     * Matrices in array format are stored in row major order by
     * default.
     *
     * Note that the sorting is not explicitly stored in a Matrix
     * Market file, but it is useful additional data that can be
     * provided by the user.
     */
    enum mtx_sorting sorting;

    /**
     * `num_rows' is the number of rows in the matrix.
     */
    int num_rows;

    /**
     * `num_columns' is the number of columns in the matrix.
     */
    int num_columns;

    /**
     * `size' is the number of entries stored in the `data' array.
     *
     * - If `triangle' is `mtx_nontriangular', then `size' is equal to
     *   `num_rows*num_columns'.
     *
     * - If `triangle' is `mtx_lower_triangular', then `size' is the
     *   number of nonzero entries on or below the main diagonal,
     *   which is equal to
     *
     *     `num_rows*(num_rows+1)/2'
     *
     *   if `num_rows <= num_columns', or
     *
     *     `num_columns*(num_columns+1)/2+(num_rows-num_columns)*num_columns'
     *
     *   otherwise.
     *
     * - If `triangle' is `mtx_upper_triangular', then `size' is
     *   instead the number of nonzeros on or above the main diagonal,
     *   which is equal to
     *
     *     `num_columns*(num_columns+1)/2'
     *
     *   if `num_columns <= num_rows', or
     *
     *     `num_rows*(num_rows+1)/2+(num_columns-num_rows)*num_rows'
     *
     *   otherwise.
     *
     * - If `triangle' is `mtx_strict_lower_triangular', then `size'
     *   is the number of nonzero entries below the main diagonal,
     *   which is equal to
     *
     *     `num_rows*(num_rows-1)/2'
     *
     *   if `num_rows <= num_columns', or
     *
     *     `num_columns*(num_columns-1)/2+(num_rows-num_columns)*num_columns'
     *
     *   otherwise.
     *
     * - If `triangle' is `mtx_strict_upper_triangular', then `size'
     *   is instead the number of nonzeros above the main diagonal,
     *   which is equal to
     *
     *     `num_columns*(num_columns-1)/2'
     *
     *   if `num_columns <= num_rows', or
     *
     *     `num_rows*(num_rows-1)/2+(num_columns-num_rows)*num_rows'
     *
     *   otherwise.
     */
    int64_t size;

    /**
     * `data' is used to store the matrix values.
     *
     * The storage format of nonzero values depends on `field' and
     * `precision'.  Only the member of the `data' union that
     * corresponds to the matrix `field' and `precision' should be
     * used to access the underlying data arrays containing the matrix
     * values.
     *
     * For example, if `field' is `real' and `precision' is `single',
     * then `data.real_single' is an array of `size' values of type
     * `float', which contains the values of the matrix entries.
     */
    union {
        float * real_single;
        double * real_double;
        float (* complex_single)[2];
        double (* complex_double)[2];
        int32_t * integer_single;
        int64_t * integer_double;
    } data;
};

/**
 * `mtx_matrix_array_data_free()' frees resources associated with a
 * Matrix Market object.
 */
void mtx_matrix_array_data_free(
    struct mtx_matrix_array_data * mtxdata);

/**
 * `mtx_matrix_array_data_alloc()' allocates data for a matrix in
 * array format.
 */
int mtx_matrix_array_data_alloc(
    struct mtx_matrix_array_data * mtxdata,
    enum mtx_field field,
    enum mtxprecision precision,
    enum mtx_symmetry symmetry,
    enum mtx_triangle triangle,
    int num_rows,
    int num_columns);

/*
 * Array matrix allocation and initialisation.
 */

/**
 * `mtx_matrix_array_data_init_real_single()' creates data for a
 * matrix with real, single-precision floating point coefficients.
 */
int mtx_matrix_array_data_init_real_single(
    struct mtx_matrix_array_data * mtxdata,
    enum mtx_symmetry symmetry,
    enum mtx_triangle triangle,
    int num_rows,
    int num_columns,
    int64_t size,
    const float * data);

/**
 * `mtx_matrix_array_data_init_real_double()' creates data for a
 * matrix with real, double-precision floating point coefficients.
 */
int mtx_matrix_array_data_init_real_double(
    struct mtx_matrix_array_data * mtxdata,
    enum mtx_symmetry symmetry,
    enum mtx_triangle triangle,
    int num_rows,
    int num_columns,
    int64_t size,
    const double * data);

/**
 * `mtx_matrix_array_data_init_complex_single()' creates data for a
 * matrix with complex, single-precision floating point coefficients.
 */
int mtx_matrix_array_data_init_complex_single(
    struct mtx_matrix_array_data * mtxdata,
    enum mtx_symmetry symmetry,
    enum mtx_triangle triangle,
    int num_rows,
    int num_columns,
    int64_t size,
    const float (* data)[2]);

/**
 * `mtx_matrix_array_data_init_complex_double()' creates data for a
 * matrix with complex, double-precision floating point coefficients.
 */
int mtx_matrix_array_data_init_complex_double(
    struct mtx_matrix_array_data * mtxdata,
    enum mtx_symmetry symmetry,
    enum mtx_triangle triangle,
    int num_rows,
    int num_columns,
    int64_t size,
    const double (* data)[2]);

/**
 * `mtx_matrix_array_data_init_integer_single()' creates data for a
 * matrix with integer, single-precision coefficients.
 */
int mtx_matrix_array_data_init_integer_single(
    struct mtx_matrix_array_data * mtxdata,
    enum mtx_symmetry symmetry,
    enum mtx_triangle triangle,
    int num_rows,
    int num_columns,
    int64_t size,
    const int32_t * data);

/**
 * `mtx_matrix_array_data_init_integer_double()' creates data for a
 * matrix with integer, double-precision coefficients.
 */
int mtx_matrix_array_data_init_integer_double(
    struct mtx_matrix_array_data * mtxdata,
    enum mtx_symmetry symmetry,
    enum mtx_triangle triangle,
    int num_rows,
    int num_columns,
    int64_t size,
    const int64_t * data);

/**
 * `mtx_matrix_array_data_copy_alloc()' allocates a copy of a matrix
 * without copying the matrix values.
 */
int mtx_matrix_array_data_copy_alloc(
    struct mtx_matrix_array_data * dst,
    const struct mtx_matrix_array_data * src);

/**
 * `mtx_matrix_array_data_copy_init()' creates a copy of a matrix and
 * also copies matrix values.
 */
int mtx_matrix_array_data_copy_init(
    struct mtx_matrix_array_data * dst,
    const struct mtx_matrix_array_data * src);

/**
 * `mtx_matrix_array_data_set_zero()' zeroes a matrix.
 */
int mtx_matrix_array_data_set_zero(
    struct mtx_matrix_array_data * mtxdata);

/**
 * `mtx_matrix_array_data_set_constant_real_single()' sets every value
 * of a matrix equal to a constant, single precision floating point
 * number.
 */
int mtx_matrix_array_data_set_constant_real_single(
    struct mtx_matrix_array_data * mtxdata,
    float a);

/**
 * `mtx_matrix_array_data_set_constant_real_double()' sets every value
 * of a matrix equal to a constant, double precision floating point
 * number.
 */
int mtx_matrix_array_data_set_constant_real_double(
    struct mtx_matrix_array_data * mtxdata,
    double a);

/**
 * `mtx_matrix_array_data_set_constant_complex_single()' sets every
 * value of a matrix equal to a constant, single precision floating
 * point complex number.
 */
int mtx_matrix_array_data_set_constant_complex_single(
    struct mtx_matrix_array_data * mtxdata,
    float a[2]);

/**
 * `mtx_matrix_array_data_set_constant_complex_double()' sets every
 * value of a matrix equal to a constant, double precision floating
 * point complex number.
 */
int mtx_matrix_array_data_set_constant_complex_double(
    struct mtx_matrix_array_data * mtxdata,
    double a[2]);

/**
 * `mtx_matrix_array_data_set_constant_integer_single()' sets every
 * value of a matrix equal to a constant integer.
 */
int mtx_matrix_array_data_set_constant_integer_single(
    struct mtx_matrix_array_data * mtxdata,
    int32_t a);

/**
 * `mtx_matrix_array_data_set_constant_integer_double()' sets every
 * value of a matrix equal to a constant integer.
 */
int mtx_matrix_array_data_set_constant_integer_double(
    struct mtx_matrix_array_data * mtxdata,
    int64_t a);

#endif
