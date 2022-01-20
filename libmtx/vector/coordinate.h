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
 * Last modified: 2021-08-09
 *
 * Sparse vectors in Matrix Market coordinate format.
 */

#ifndef LIBMTX_VECTOR_COORDINATE_COORDINATE_H
#define LIBMTX_VECTOR_COORDINATE_COORDINATE_H

#include <libmtx/mtx/assembly.h>
#include <libmtx/mtx/header.h>
#include <libmtx/precision.h>
#include <libmtx/mtx/reorder.h>
#include <libmtx/mtx/sort.h>

#include <stdint.h>

struct mtx;
struct mtx_vector_coordinate_real_single;
struct mtx_vector_coordinate_real_double;
struct mtx_vector_coordinate_complex_single;
struct mtx_vector_coordinate_complex_double;
struct mtx_vector_coordinate_integer_single;
struct mtx_vector_coordinate_integer_double;
struct mtx_vector_coordinate_pattern;

/*
 * Coordinate vector allocation and initialisation.
 */

/**
 * `mtx_alloc_vector_coordinate()` allocates a sparse vector in
 * coordinate format.
 */
int mtx_alloc_vector_coordinate(
    struct mtx * mtx,
    enum mtx_field field,
    enum mtxprecision precision,
    int num_comment_lines,
    const char ** comment_lines,
    int num_rows,
    int64_t size);

/**
 * `mtx_init_vector_coordinate_real_single()' creates a coordinate
 * vector with real, single-precision floating point coefficients.
 */
int mtx_init_vector_coordinate_real_single(
    struct mtx * mtx,
    enum mtx_sorting sorting,
    enum mtx_assembly assembly,
    int num_comment_lines,
    const char ** comment_lines,
    int num_rows,
    int64_t size,
    const struct mtx_vector_coordinate_real_single * data);

/**
 * `mtx_init_vector_coordinate_real_double()' creates a coordinate
 * vector with real, double-precision floating point coefficients.
 */
int mtx_init_vector_coordinate_real_double(
    struct mtx * mtx,
    enum mtx_sorting sorting,
    enum mtx_assembly assembly,
    int num_comment_lines,
    const char ** comment_lines,
    int num_rows,
    int64_t size,
    const struct mtx_vector_coordinate_real_double * data);

/**
 * `mtx_init_vector_coordinate_complex_single()' creates a coordinate
 * vector with complex, single-precision floating point coefficients.
 */
int mtx_init_vector_coordinate_complex_single(
    struct mtx * mtx,
    enum mtx_sorting sorting,
    enum mtx_assembly assembly,
    int num_comment_lines,
    const char ** comment_lines,
    int num_rows,
    int64_t size,
    const struct mtx_vector_coordinate_complex_single * data);

/**
 * `mtx_init_vector_coordinate_integer_single()' creates a coordinate
 * vector with single precision, integer coefficients.
 */
int mtx_init_vector_coordinate_integer_single(
    struct mtx * mtx,
    enum mtx_sorting sorting,
    enum mtx_assembly assembly,
    int num_comment_lines,
    const char ** comment_lines,
    int num_rows,
    int64_t size,
    const struct mtx_vector_coordinate_integer_single * data);

/**
 * `mtx_init_vector_coordinate_pattern()` creates a coordinate vector
 * with boolean (pattern) coefficients.
 */
int mtx_init_vector_coordinate_pattern(
    struct mtx * mtx,
    enum mtx_sorting sorting,
    enum mtx_assembly assembly,
    int num_comment_lines,
    const char ** comment_lines,
    int num_rows,
    int64_t size,
    const struct mtx_vector_coordinate_pattern * data);

/*
 * Coordinate vector value initialisation.
 */

/**
 * `mtx_vector_coordinate_set_zero()' zeroes a vector in coordinate
 * format.
 */
int mtx_vector_coordinate_set_zero(
    struct mtx * mtx);

/**
 * `mtx_vector_coordinate_set_constant_real_single()' sets every
 * nonzero value of a vector equal to a constant, single precision
 * floating point number.
 */
int mtx_vector_coordinate_set_constant_real_single(
    struct mtx * mtx,
    float a);

/**
 * `mtx_vector_coordinate_set_constant_real_double()' sets every
 * nonzero value of a vector equal to a constant, double precision
 * floating point number.
 */
int mtx_vector_coordinate_set_constant_real_double(
    struct mtx * mtx,
    double a);

/**
 * `mtx_vector_coordinate_set_constant_complex_single()' sets every
 * nonzero value of a vector equal to a constant, single precision
 * floating point complex number.
 */
int mtx_vector_coordinate_set_constant_complex_single(
    struct mtx * mtx,
    float a[2]);

/**
 * `mtx_vector_coordinate_set_constant_integer_single()' sets every
 * nonzero value of a vector equal to a constant, single precision
 * integer.
 */
int mtx_vector_coordinate_set_constant_integer_single(
    struct mtx * mtx,
    int32_t a);

#endif
