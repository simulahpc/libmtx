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
 * Last modified: 2021-08-19
 *
 * Vectors in array format.
 */

#ifndef LIBMTX_VECTOR_ARRAY_ARRAY_H
#define LIBMTX_VECTOR_ARRAY_ARRAY_H

#include <libmtx/mtx/header.h>
#include <libmtx/mtx/precision.h>

#include <stdint.h>

struct mtx;

/*
 * Array vector allocation and initialisation.
 */

/**
 * `mtx_alloc_vector_array()` allocates a dense vector in array
 * format.
 */
int mtx_alloc_vector_array(
    struct mtx * mtx,
    enum mtx_field field,
    enum mtxprecision precision,
    int num_comment_lines,
    const char ** comment_lines,
    int size);

/**
 * `mtx_init_vector_array_real_single()' creates a vector with real,
 * single-precision floating point coefficients.
 */
int mtx_init_vector_array_real_single(
    struct mtx * mtx,
    int num_comment_lines,
    const char ** comment_lines,
    int64_t size,
    const float * data);

/**
 * `mtx_init_vector_array_real_double()' creates a vector with real,
 * double-precision floating point coefficients.
 */
int mtx_init_vector_array_real_double(
    struct mtx * mtx,
    int num_comment_lines,
    const char ** comment_lines,
    int64_t size,
    const double * data);

/**
 * `mtx_init_vector_array_complex_single()' creates a vector with
 * complex, single-precision floating point coefficients.
 */
int mtx_init_vector_array_complex_single(
    struct mtx * mtx,
    int num_comment_lines,
    const char ** comment_lines,
    int64_t size,
    const float (* data)[2]);

/**
 * `mtx_init_vector_array_integer_single()` creates a vector with
 * single precision, integer coefficients.
 */
int mtx_init_vector_array_integer_single(
    struct mtx * mtx,
    int num_comment_lines,
    const char ** comment_lines,
    int64_t size,
    const int32_t * data);

/*
 * Array vector value initialisation.
 */

/**
 * `mtx_vector_array_set_zero()' zeroes a vector in array format.
 */
int mtx_vector_array_set_zero(
    struct mtx * mtx);

/**
 * `mtx_vector_array_set_constant_real_single()' sets every value of a
 * vector equal to a constant, single precision floating point number.
 */
int mtx_vector_array_set_constant_real_single(
    struct mtx * mtx,
    float a);

/**
 * `mtx_vector_array_set_constant_real_double()' sets every value of a
 * vector equal to a constant, double precision floating point number.
 */
int mtx_vector_array_set_constant_real_double(
    struct mtx * mtx,
    double a);

/**
 * `mtx_vector_array_set_constant_complex_single()' sets every value
 * of a vector equal to a constant, single precision floating point
 * complex number.
 */
int mtx_vector_array_set_constant_complex_single(
    struct mtx * mtx,
    float a[2]);

/**
 * `mtx_vector_array_set_constant_integer_single()' sets every value
 * of a vector equal to a constant, single precision integer.
 */
int mtx_vector_array_set_constant_integer_single(
    struct mtx * mtx,
    int32_t a);

#endif
