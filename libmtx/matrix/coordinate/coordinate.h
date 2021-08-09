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
#include <libmtx/mtx/reorder.h>
#include <libmtx/mtx/sort.h>
#include <libmtx/triangle.h>

#include <stdint.h>

struct mtx;

/*
 * Data types for sparse matrix nonzero values.
 */

/**
 * `mtx_matrix_coordinate_real' represents a nonzero matrix entry in a
 * Matrix Market file with `matrix' object, `coordinate' format and
 * `real' field.
 */
struct mtx_matrix_coordinate_real
{
    int i, j; /* row and column index */
    float a;  /* nonzero value */
};

/**
 * `mtx_matrix_coordinate_double' represents a nonzero matrix entry in
 * a Matrix Market file with `matrix' object, `coordinate' format and
 * `double' field.
 */
struct mtx_matrix_coordinate_double
{
    int i, j; /* row and column index */
    double a; /* nonzero value */
};

/**
 * `mtx_matrix_coordinate_complex' represents a nonzero matrix entry
 * in a Matrix Market file with `matrix' object, `coordinate' format
 * and `complex' field.
 */
struct mtx_matrix_coordinate_complex
{
    int i, j;     /* row and column index */
    float a, b;   /* real and imaginary parts of nonzero value */
};

/**
 * `mtx_matrix_coordinate_integer' represents a nonzero matrix entry
 * in a Matrix Market file with `matrix' object, `coordinate' format
 * and `integer' field.
 */
struct mtx_matrix_coordinate_integer
{
    int i, j; /* row and column index */
    int a;    /* nonzero value */
};

/**
 * `mtx_matrix_coordinate_pattern' represents a nonzero matrix entry
 * in a Matrix Market file with `matrix' object, `coordinate' format
 * and `pattern' field.
 */
struct mtx_matrix_coordinate_pattern
{
    int i, j; /* row and column index */
};

/*
 * Sparse (coordinate) matrix allocation.
 */

/**
 * `mtx_alloc_matrix_coordinate_real()` allocates a sparse matrix with
 * real, single-precision floating point coefficients.
 */
int mtx_alloc_matrix_coordinate_real(
    struct mtx * mtx,
    enum mtx_symmetry symmetry,
    int num_comment_lines,
    const char ** comment_lines,
    int num_rows,
    int num_columns,
    int64_t size);

/**
 * `mtx_alloc_matrix_coordinate_double()` allocates a sparse matrix
 * with real, double-precision floating point coefficients.
 */
int mtx_alloc_matrix_coordinate_double(
    struct mtx * mtx,
    enum mtx_symmetry symmetry,
    int num_comment_lines,
    const char ** comment_lines,
    int num_rows,
    int num_columns,
    int64_t size);

/**
 * `mtx_alloc_matrix_coordinate_complex()` allocates a sparse matrix
 * with complex, single-precision floating point coefficients.
 */
int mtx_alloc_matrix_coordinate_complex(
    struct mtx * mtx,
    enum mtx_symmetry symmetry,
    int num_comment_lines,
    const char ** comment_lines,
    int num_rows,
    int num_columns,
    int64_t size);

/**
 * `mtx_alloc_matrix_coordinate_integer()` allocates a sparse matrix
 * with integer coefficients.
 */
int mtx_alloc_matrix_coordinate_integer(
    struct mtx * mtx,
    enum mtx_symmetry symmetry,
    int num_comment_lines,
    const char ** comment_lines,
    int num_rows,
    int num_columns,
    int64_t size);

/**
 * `mtx_alloc_matrix_coordinate_pattern()` allocates a sparse matrix
 * with boolean coefficients.
 */
int mtx_alloc_matrix_coordinate_pattern(
    struct mtx * mtx,
    enum mtx_symmetry symmetry,
    int num_comment_lines,
    const char ** comment_lines,
    int num_rows,
    int num_columns,
    int64_t size);

/*
 * Sparse (coordinate) matrix initialisation.
 */

/**
 * `mtx_init_matrix_coordinate_real()` creates a sparse matrix with
 * real, single-precision floating point coefficients.
 */
int mtx_init_matrix_coordinate_real(
    struct mtx * mtx,
    enum mtx_symmetry symmetry,
    enum mtx_triangle triangle,
    enum mtx_sorting sorting,
    enum mtx_ordering ordering,
    enum mtx_assembly assembly,
    int num_comment_lines,
    const char ** comment_lines,
    int num_rows,
    int num_columns,
    int64_t size,
    const struct mtx_matrix_coordinate_real * data);

/**
 * `mtx_init_matrix_coordinate_double()` creates a sparse matrix with
 * real, double-precision floating point coefficients.
 */
int mtx_init_matrix_coordinate_double(
    struct mtx * mtx,
    enum mtx_symmetry symmetry,
    enum mtx_triangle triangle,
    enum mtx_sorting sorting,
    enum mtx_ordering ordering,
    enum mtx_assembly assembly,
    int num_comment_lines,
    const char ** comment_lines,
    int num_rows,
    int num_columns,
    int64_t size,
    const struct mtx_matrix_coordinate_double * data);

/**
 * `mtx_init_matrix_coordinate_complex()` creates a sparse matrix with
 * complex, single-precision floating point coefficients.
 */
int mtx_init_matrix_coordinate_complex(
    struct mtx * mtx,
    enum mtx_symmetry symmetry,
    enum mtx_triangle triangle,
    enum mtx_sorting sorting,
    enum mtx_ordering ordering,
    enum mtx_assembly assembly,
    int num_comment_lines,
    const char ** comment_lines,
    int num_rows,
    int num_columns,
    int64_t size,
    const struct mtx_matrix_coordinate_complex * data);

/**
 * `mtx_init_matrix_coordinate_integer()` creates a sparse matrix with
 * integer coefficients.
 */
int mtx_init_matrix_coordinate_integer(
    struct mtx * mtx,
    enum mtx_symmetry symmetry,
    enum mtx_triangle triangle,
    enum mtx_sorting sorting,
    enum mtx_ordering ordering,
    enum mtx_assembly assembly,
    int num_comment_lines,
    const char ** comment_lines,
    int num_rows,
    int num_columns,
    int64_t size,
    const struct mtx_matrix_coordinate_integer * data);

/**
 * `mtx_init_matrix_coordinate_pattern()` creates a sparse matrix with
 * boolean coefficients.
 */
int mtx_init_matrix_coordinate_pattern(
    struct mtx * mtx,
    enum mtx_symmetry symmetry,
    enum mtx_triangle triangle,
    enum mtx_sorting sorting,
    enum mtx_ordering ordering,
    enum mtx_assembly assembly,
    int num_comment_lines,
    const char ** comment_lines,
    int num_rows,
    int num_columns,
    int64_t size,
    const struct mtx_matrix_coordinate_pattern * data);

/*
 * Sparse (coordinate) matrix value initialisation.
 */

/**
 * `mtx_matrix_coordinate_set_zero()' zeroes a matrix in coordinate
 * format.
 */
int mtx_matrix_coordinate_set_zero(
    struct mtx * mtx);

/**
 * `mtx_matrix_coordinate_set_constant_real()' sets every nonzero
 * value of a matrix equal to a constant, single precision floating
 * point number.
 */
int mtx_matrix_coordinate_set_constant_real(
    struct mtx * mtx,
    float a);

/**
 * `mtx_matrix_coordinate_set_constant_double()' sets every nonzero
 * value of a matrix equal to a constant, double precision floating
 * point number.
 */
int mtx_matrix_coordinate_set_constant_double(
    struct mtx * mtx,
    double a);

/**
 * `mtx_matrix_coordinate_set_constant_complex()' sets every nonzero
 * value of a matrix equal to a constant, single precision floating
 * point complex number.
 */
int mtx_matrix_coordinate_set_constant_complex(
    struct mtx * mtx,
    float a,
    float b);

/**
 * `mtx_matrix_coordinate_set_constant_integer()' sets every nonzero
 * value of a matrix equal to a constant integer.
 */
int mtx_matrix_coordinate_set_constant_integer(
    struct mtx * mtx,
    int a);

/*
 * Other, sparse (coordinate) matrix functions.
 */

/**
 * `mtx_matrix_coordinate_num_nonzeros()` computes the number of
 * matrix nonzeros, including those that are not stored explicitly due
 * to symmetry.
 */
int mtx_matrix_coordinate_num_nonzeros(
    enum mtx_field field,
    enum mtx_symmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const void * data,
    int64_t * num_nonzeros);

/**
 * `mtx_matrix_coordinate_num_diagonal_nonzeros()` counts the number
 * of nonzeros on the main diagonal of a sparse matrix in the Matrix
 * Market format.
 */
int mtx_matrix_coordinate_num_diagonal_nonzeros(
    enum mtx_field field,
    int64_t size,
    const void * data,
    int64_t * num_diagonal_nonzeros);

/**
 * `mtx_matrix_coordinate_transpose()` transposes a square sparse
 * matrix.
 */
int mtx_matrix_coordinate_transpose(
    struct mtx * mtx);

#endif
