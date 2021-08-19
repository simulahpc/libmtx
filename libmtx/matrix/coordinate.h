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
 * Sparse matrices in Matrix Market format.
 */

#ifndef LIBMTX_MATRIX_COORDINATE_H
#define LIBMTX_MATRIX_COORDINATE_H

#include <libmtx/mtx/assembly.h>
#include <libmtx/mtx/header.h>
#include <libmtx/mtx/precision.h>
#include <libmtx/mtx/reorder.h>
#include <libmtx/mtx/sort.h>
#include <libmtx/mtx/triangle.h>

#include <stdint.h>

struct mtx;
struct mtx_matrix_coordinate_real_single;
struct mtx_matrix_coordinate_real_double;
struct mtx_matrix_coordinate_complex_single;
struct mtx_matrix_coordinate_complex_double;
struct mtx_matrix_coordinate_integer_single;
struct mtx_matrix_coordinate_integer_double;
struct mtx_matrix_coordinate_pattern;

/*
 * Coordinate matrix allocation and initialisation.
 */

/**
 * `mtx_alloc_matrix_coordinate()` allocates a sparse matrix in
 * coordinate format.
 */
int mtx_alloc_matrix_coordinate(
    struct mtx * mtx,
    enum mtx_field field,
    enum mtx_precision precision,
    enum mtx_symmetry symmetry,
    int num_comment_lines,
    const char ** comment_lines,
    int num_rows,
    int num_columns,
    int64_t size);

/**
 * `mtx_init_matrix_coordinate_real_single()` creates a sparse matrix
 * with real, single-precision floating point coefficients.
 */
int mtx_init_matrix_coordinate_real_single(
    struct mtx * mtx,
    enum mtx_symmetry symmetry,
    enum mtx_triangle triangle,
    enum mtx_sorting sorting,
    enum mtx_assembly assembly,
    int num_comment_lines,
    const char ** comment_lines,
    int num_rows,
    int num_columns,
    int64_t size,
    const struct mtx_matrix_coordinate_real_single * data);

/**
 * `mtx_init_matrix_coordinate_real_double()` creates a sparse matrix
 * with real, double-precision floating point coefficients.
 */
int mtx_init_matrix_coordinate_real_double(
    struct mtx * mtx,
    enum mtx_symmetry symmetry,
    enum mtx_triangle triangle,
    enum mtx_sorting sorting,
    enum mtx_assembly assembly,
    int num_comment_lines,
    const char ** comment_lines,
    int num_rows,
    int num_columns,
    int64_t size,
    const struct mtx_matrix_coordinate_real_double * data);

/**
 * `mtx_init_matrix_coordinate_complex_single()` creates a sparse
 * matrix with complex, single-precision floating point coefficients.
 */
int mtx_init_matrix_coordinate_complex_single(
    struct mtx * mtx,
    enum mtx_symmetry symmetry,
    enum mtx_triangle triangle,
    enum mtx_sorting sorting,
    enum mtx_assembly assembly,
    int num_comment_lines,
    const char ** comment_lines,
    int num_rows,
    int num_columns,
    int64_t size,
    const struct mtx_matrix_coordinate_complex_single * data);

/**
 * `mtx_init_matrix_coordinate_integer_single()` creates a sparse
 * matrix with single precision, integer coefficients.
 */
int mtx_init_matrix_coordinate_integer_single(
    struct mtx * mtx,
    enum mtx_symmetry symmetry,
    enum mtx_triangle triangle,
    enum mtx_sorting sorting,
    enum mtx_assembly assembly,
    int num_comment_lines,
    const char ** comment_lines,
    int num_rows,
    int num_columns,
    int64_t size,
    const struct mtx_matrix_coordinate_integer_single * data);

/**
 * `mtx_init_matrix_coordinate_pattern()` creates a sparse matrix
 * with boolean coefficients.
 */
int mtx_init_matrix_coordinate_pattern(
    struct mtx * mtx,
    enum mtx_symmetry symmetry,
    enum mtx_triangle triangle,
    enum mtx_sorting sorting,
    enum mtx_assembly assembly,
    int num_comment_lines,
    const char ** comment_lines,
    int num_rows,
    int num_columns,
    int64_t size,
    const struct mtx_matrix_coordinate_pattern * data);

/*
 * Coordinate matrix value initialisation.
 */

/**
 * `mtx_matrix_coordinate_set_zero()' zeroes a matrix in coordinate
 * format.
 */
int mtx_matrix_coordinate_set_zero(
    struct mtx * mtx);

/**
 * `mtx_matrix_coordinate_set_constant_real_single()' sets every
 * nonzero value of a matrix equal to a constant, single precision
 * floating point number.
 */
int mtx_matrix_coordinate_set_constant_real_single(
    struct mtx * mtx,
    float a);

/**
 * `mtx_matrix_coordinate_set_constant_real_double()' sets every
 * nonzero value of a matrix equal to a constant, double precision
 * floating point number.
 */
int mtx_matrix_coordinate_set_constant_real_double(
    struct mtx * mtx,
    double a);

/**
 * `mtx_matrix_coordinate_set_constant_complex_single()' sets every
 * nonzero value of a matrix equal to a constant, single precision
 * floating point complex number.
 */
int mtx_matrix_coordinate_set_constant_complex_single(
    struct mtx * mtx,
    float a[2]);

/**
 * `mtx_matrix_coordinate_set_constant_integer_single()' sets every
 * nonzero value of a matrix equal to a constant, single precision
 * integer.
 */
int mtx_matrix_coordinate_set_constant_integer_single(
    struct mtx * mtx,
    int32_t a);

/*
 * Other, coordinate matrix functions.
 */

/**
 * `mtx_matrix_coordinate_transpose()' transposes a matrix in
 * coordinate format.
 */
int mtx_matrix_coordinate_transpose(
    struct mtx * mtx);

#endif
