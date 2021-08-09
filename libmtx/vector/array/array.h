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
 * Dense vectors in Matrix Market array format.
 */

#ifndef LIBMTX_VECTOR_ARRAY_ARRAY_H
#define LIBMTX_VECTOR_ARRAY_ARRAY_H

struct mtx;

/*
 * Dense (array) vector allocation.
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
 * `mtx_alloc_vector_array_double()` allocates a vector with real,
 * double-precision floating point coefficients.
 */
int mtx_alloc_vector_array_double(
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
 * `mtx_alloc_vector_array_integer()` allocates a vector with integer
 * coefficients.
 */
int mtx_alloc_vector_array_integer(
    struct mtx * mtx,
    int num_comment_lines,
    const char ** comment_lines,
    int size);

/*
 * Dense (array) vector allocation and initialisation.
 */

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
 * `mtx_init_vector_array_integer()` creates a vector with integer
 * coefficients.
 */
int mtx_init_vector_array_integer(
    struct mtx * mtx,
    int num_comment_lines,
    const char ** comment_lines,
    int size,
    const int * data);

/*
 * Dense (array) vector value initialisation.
 */

/**
 * `mtx_vector_array_set_zero()' zeroes a vector in array format.
 */
int mtx_vector_array_set_zero(
    struct mtx * mtx);

/**
 * `mtx_vector_array_set_constant_real()' sets every value of a vector
 * equal to a constant, single precision floating point number.
 */
int mtx_vector_array_set_constant_real(
    struct mtx * mtx,
    float a);

/**
 * `mtx_vector_array_set_constant_double()' sets every value of a
 * vector equal to a constant, double precision floating point number.
 */
int mtx_vector_array_set_constant_double(
    struct mtx * mtx,
    double a);

/**
 * `mtx_vector_array_set_constant_complex()' sets every value of a
 * vector equal to a constant, single precision floating point complex
 * number.
 */
int mtx_vector_array_set_constant_complex(
    struct mtx * mtx,
    float a,
    float b);

/**
 * `mtx_vector_array_set_constant_integer()' sets every value of a
 * vector equal to a constant integer.
 */
int mtx_vector_array_set_constant_integer(
    struct mtx * mtx,
    int a);

#endif
