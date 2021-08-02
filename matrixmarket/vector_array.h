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
 * Last modified: 2021-08-02
 *
 * Dense vectors in Matrix Market format.
 */

#ifndef MATRIXMARKET_VECTOR_ARRAY_H
#define MATRIXMARKET_VECTOR_ARRAY_H

struct mtx;

/*
 * Dense vector constructors.
 */

/**
 * `mtx_alloc_vector_array_real()` allocates a vector with real,
 * single-precision floating point coefficients.
 */
int mtx_alloc_vector_array_real(
    struct mtx * mtx,
    int num_comment_lines,
    const char ** comment_lines,
    int size);

/**
 * `mtx_init_vector_array_real()` creates a vector with real,
 * single-precision floating point coefficients.
 */
int mtx_init_vector_array_real(
    struct mtx * mtx,
    int num_comment_lines,
    const char ** comment_lines,
    int size,
    const float * data);

/**
 * `mtx_init_vector_array_real_zero()` creates a vector of real,
 * single-precision floating point coefficients by filling with zeros.
 */
int mtx_init_vector_array_real_zero(
    struct mtx * mtx,
    int num_comment_lines,
    const char ** comment_lines,
    int size);

/**
 * `mtx_init_vector_array_real_ones()` creates a vector of real,
 * single-precision floating point coefficients by filling with ones.
 */
int mtx_init_vector_array_real_ones(
    struct mtx * mtx,
    int num_comment_lines,
    const char ** comment_lines,
    int size);

/**
 * `mtx_alloc_vector_array_double()` allocates a vector with real,
 * double-precision floating point coefficients.
 */
int mtx_alloc_vector_array_double(
    struct mtx * mtx,
    int num_comment_lines,
    const char ** comment_lines,
    int size);

/**
 * `mtx_init_vector_array_double()` creates a vector with real,
 * double-precision floating point coefficients.
 */
int mtx_init_vector_array_double(
    struct mtx * mtx,
    int num_comment_lines,
    const char ** comment_lines,
    int size,
    const double * data);

/**
 * `mtx_init_vector_array_double_zero()` creates a vector of real,
 * double-precision floating point coefficients by filling with zeros.
 */
int mtx_init_vector_array_double_zero(
    struct mtx * mtx,
    int num_comment_lines,
    const char ** comment_lines,
    int size);

/**
 * `mtx_init_vector_array_double_ones()` creates a vector of real,
 * double-precision floating point coefficients by filling with ones.
 */
int mtx_init_vector_array_double_ones(
    struct mtx * mtx,
    int num_comment_lines,
    const char ** comment_lines,
    int size);

/**
 * `mtx_alloc_vector_array_complex()` allocates a vector with complex,
 * single-precision floating point coefficients.
 */
int mtx_alloc_vector_array_complex(
    struct mtx * mtx,
    int num_comment_lines,
    const char ** comment_lines,
    int size);

/**
 * `mtx_init_vector_array_complex()` creates a vector with complex,
 * single-precision floating point coefficients.
 */
int mtx_init_vector_array_complex(
    struct mtx * mtx,
    int num_comment_lines,
    const char ** comment_lines,
    int size,
    const float * data);

/**
 * `mtx_init_vector_array_complex_zero()` creates a vector of complex,
 * single-precision floating point coefficients by filling with zeros.
 */
int mtx_init_vector_array_complex_zero(
    struct mtx * mtx,
    int num_comment_lines,
    const char ** comment_lines,
    int size);

/**
 * `mtx_init_vector_array_complex_ones()` creates a vector of complex,
 * single-precision floating point coefficients by filling with ones.
 */
int mtx_init_vector_array_complex_ones(
    struct mtx * mtx,
    int num_comment_lines,
    const char ** comment_lines,
    int size);

/**
 * `mtx_alloc_vector_array_integer()` allocates a vector with integer
 * coefficients.
 */
int mtx_alloc_vector_array_integer(
    struct mtx * mtx,
    int num_comment_lines,
    const char ** comment_lines,
    int size);

/**
 * `mtx_init_vector_array_integer()` creates a vector with integer
 * coefficients.
 */
int mtx_init_vector_array_integer(
    struct mtx * mtx,
    int num_comment_lines,
    const char ** comment_lines,
    int size,
    const int * data);

/**
 * `mtx_init_vector_array_integer_zero()` creates a vector of integer,
 * coefficients by filling with zeros.
 */
int mtx_init_vector_array_integer_zero(
    struct mtx * mtx,
    int num_comment_lines,
    const char ** comment_lines,
    int size);

/**
 * `mtx_init_vector_array_integer_ones()` creates a vector of integer,
 * coefficients by filling with ones.
 */
int mtx_init_vector_array_integer_ones(
    struct mtx * mtx,
    int num_comment_lines,
    const char ** comment_lines,
    int size);

/*
 * Other dense vector functions.
 */

/**
 * `mtx_vector_array_set_zero()' zeroes a vector in array format.
 */
int mtx_vector_array_set_zero(
    struct mtx * mtx);

#endif
