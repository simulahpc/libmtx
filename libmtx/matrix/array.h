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
 * Dense matrices in Matrix Market array format.
 */

#ifndef LIBMTX_MATRIX_ARRAY_ARRAY_H
#define LIBMTX_MATRIX_ARRAY_ARRAY_H

#include <libmtx/mtx/header.h>
#include <libmtx/mtx/precision.h>
#include <libmtx/mtx/sort.h>
#include <libmtx/mtx/triangle.h>

#include <stdint.h>

struct mtx;

/*
 * Dense (array) matrix allocation.
 */

/**
 * `mtx_alloc_matrix_array()` allocates a dense matrix in array
 * format.
 *
 * If `symmetry' is `mtx_symmetric', `mtx_skew_symmetric' or
 * `mtx_hermitian', then `triangle' must be either
 * `mtx_lower_triangular' or `mtx_upper_triangular' to indicate which
 * triangle of the matrix is stored in `data'.  Otherwise, if
 * `symmetry' is `mtx_general', then `triangle' must be
 * `mtx_nontriangular'.
 *
 * ´sorting' must be `mtx_row_major' or `mtx_column_major'.
 */
int mtx_alloc_matrix_array(
    struct mtx * mtx,
    enum mtx_field field,
    enum mtxprecision precision,
    enum mtx_symmetry symmetry,
    enum mtx_triangle triangle,
    enum mtx_sorting sorting,
    int num_comment_lines,
    const char ** comment_lines,
    int num_rows,
    int num_columns);

/*
 * Array matrix allocation and initialisation.
 */

/**
 * `mtx_init_matrix_array_real_single()` creates a dense matrix with
 * real, single-precision floating point coefficients.
 *
 * If `symmetry' is `mtx_symmetric', `mtx_skew_symmetric' or
 * `mtx_hermitian', then `triangle' must be either
 * `mtx_lower_triangular' or `mtx_upper_triangular' to indicate which
 * triangle of the matrix is stored in `data'.  Otherwise, if
 * `symmetry' is `mtx_general', then `triangle' must be
 * `mtx_nontriangular'.
 *
 * ´sorting' must be `mtx_row_major' or `mtx_column_major'.
 */
int mtx_init_matrix_array_real_single(
    struct mtx * mtx,
    enum mtx_symmetry symmetry,
    enum mtx_triangle triangle,
    enum mtx_sorting sorting,
    int num_comment_lines,
    const char ** comment_lines,
    int num_rows,
    int num_columns,
    int64_t size,
    const float * data);

/**
 * `mtx_init_matrix_array_real_double()` creates a dense matrix with
 * real, double-precision floating point coefficients.
 *
 * If `symmetry' is `mtx_symmetric', `mtx_skew_symmetric' or
 * `mtx_hermitian', then `triangle' must be either
 * `mtx_lower_triangular' or `mtx_upper_triangular' to indicate which
 * triangle of the matrix is stored in `data'.  Otherwise, if
 * `symmetry' is `mtx_general', then `triangle' must be
 * `mtx_nontriangular'.
 *
 * ´sorting' must be `mtx_row_major' or `mtx_column_major'.
 */
int mtx_init_matrix_array_real_double(
    struct mtx * mtx,
    enum mtx_symmetry symmetry,
    enum mtx_triangle triangle,
    enum mtx_sorting sorting,
    int num_comment_lines,
    const char ** comment_lines,
    int num_rows,
    int num_columns,
    int64_t size,
    const double * data);

/**
 * `mtx_init_matrix_array_complex_single()` creates a dense matrix
 * with complex, single-precision floating point coefficients.
 *
 * If `symmetry' is `mtx_symmetric', `mtx_skew_symmetric' or
 * `mtx_hermitian', then `triangle' must be either
 * `mtx_lower_triangular' or `mtx_upper_triangular' to indicate which
 * triangle of the matrix is stored in `data'.  Otherwise, if
 * `symmetry' is `mtx_general', then `triangle' must be
 * `mtx_nontriangular'.
 *
 * ´sorting' must be `mtx_row_major' or `mtx_column_major'.
 */
int mtx_init_matrix_array_complex_single(
    struct mtx * mtx,
    enum mtx_symmetry symmetry,
    enum mtx_triangle triangle,
    enum mtx_sorting sorting,
    int num_comment_lines,
    const char ** comment_lines,
    int num_rows,
    int num_columns,
    int64_t size,
    const float (* data)[2]);

/**
 * `mtx_init_matrix_array_integer_single()` creates a dense matrix
 * with integer coefficients.
 *
 * If `symmetry' is `mtx_symmetric', `mtx_skew_symmetric' or
 * `mtx_hermitian', then `triangle' must be either
 * `mtx_lower_triangular' or `mtx_upper_triangular' to indicate which
 * triangle of the matrix is stored in `data'.  Otherwise, if
 * `symmetry' is `mtx_general', then `triangle' must be
 * `mtx_nontriangular'.
 *
 * ´sorting' must be `mtx_row_major' or `mtx_column_major'.
 */
int mtx_init_matrix_array_integer_single(
    struct mtx * mtx,
    enum mtx_symmetry symmetry,
    enum mtx_triangle triangle,
    enum mtx_sorting sorting,
    int num_comment_lines,
    const char ** comment_lines,
    int num_rows,
    int num_columns,
    int64_t size,
    const int32_t * data);

/*
 * Dense (array) matrix value initialisation.
 */

/**
 * `mtx_matrix_array_set_zero()' zeroes a matrix in array format.
 */
int mtx_matrix_array_set_zero(
    struct mtx * mtx);

/**
 * `mtx_matrix_array_set_constant_real_single()' sets every value of a
 * matrix equal to a constant, single precision floating point number.
 */
int mtx_matrix_array_set_constant_real_single(
    struct mtx * mtx,
    float a);

/**
 * `mtx_matrix_array_set_constant_real_double()' sets every value of a
 * matrix equal to a constant, double precision floating point number.
 */
int mtx_matrix_array_set_constant_real_double(
    struct mtx * mtx,
    double a);

/**
 * `mtx_matrix_array_set_constant_complex_single()' sets every value
 * of a matrix equal to a constant, single precision floating point
 * complex number.
 */
int mtx_matrix_array_set_constant_complex_single(
    struct mtx * mtx,
    float a[2]);

/**
 * `mtx_matrix_array_set_constant_integer_single()' sets every value
 * of a matrix equal to a constant, single precision integer.
 */
int mtx_matrix_array_set_constant_integer_single(
    struct mtx * mtx,
    int32_t a);

#endif
