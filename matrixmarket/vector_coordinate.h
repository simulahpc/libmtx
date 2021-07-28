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
 * Last modified: 2021-07-28
 *
 * Sparse vectors in Matrix Market format.
 */

#ifndef MATRIXMARKET_VECTOR_COORDINATE_H
#define MATRIXMARKET_VECTOR_COORDINATE_H

#include <matrixmarket/header.h>

struct mtx;

/*
 * Data types for sparse vector nonzero values.
 */

/**
 * `mtx_vector_coordinate_real' represents a nonzero vector entry in a
 * Matrix Market file with `vector' object, `coordinate' format and
 * `real' field.
 */
struct mtx_vector_coordinate_real
{
    int i;    /* row index */
    float a;  /* nonzero value */
};

/**
 * `mtx_vector_coordinate_double' represents a nonzero vector entry in
 * a Matrix Market file with `vector' object, `coordinate' format and
 * `double' field.
 */
struct mtx_vector_coordinate_double
{
    int i;    /* row index */
    double a; /* nonzero value */
};

/**
 * `mtx_vector_coordinate_complex' represents a nonzero vector entry
 * in a Matrix Market file with `vector' object, `coordinate' format
 * and `complex' field.
 */
struct mtx_vector_coordinate_complex
{
    int i;        /* row index */
    float a, b;   /* real and imaginary parts of nonzero value */
};

/**
 * `mtx_vector_coordinate_integer' represents a nonzero vector entry
 * in a Matrix Market file with `vector' object, `coordinate' format
 * and `integer' field.
 */
struct mtx_vector_coordinate_integer
{
    int i;    /* row index */
    int a;    /* nonzero value */
};

/**
 * `mtx_vector_coordinate_pattern' represents a nonzero vector entry
 * in a Matrix Market file with `vector' object, `coordinate' format
 * and `pattern' field.
 */
struct mtx_vector_coordinate_pattern
{
    int i; /* row index */
};

/*
 * Sparse vector constructors.
 */

/**
 * `mtx_init_vector_coordinate_real()` creates a sparse vector with
 * real, single-precision floating point coefficients.
 */
int mtx_init_vector_coordinate_real(
    struct mtx * mtx,
    enum mtx_sorting sorting,
    enum mtx_ordering ordering,
    enum mtx_assembly assembly,
    int num_comment_lines,
    const char ** comment_lines,
    int num_rows,
    int size,
    const struct mtx_vector_coordinate_real * data);

/**
 * `mtx_init_vector_coordinate_double()` creates a sparse vector with
 * real, double-precision floating point coefficients.
 */
int mtx_init_vector_coordinate_double(
    struct mtx * mtx,
    enum mtx_sorting sorting,
    enum mtx_ordering ordering,
    enum mtx_assembly assembly,
    int num_comment_lines,
    const char ** comment_lines,
    int num_rows,
    int size,
    const struct mtx_vector_coordinate_double * data);

/**
 * `mtx_init_vector_coordinate_complex()` creates a sparse vector with
 * complex, single-precision floating point coefficients.
 */
int mtx_init_vector_coordinate_complex(
    struct mtx * mtx,
    enum mtx_sorting sorting,
    enum mtx_ordering ordering,
    enum mtx_assembly assembly,
    int num_comment_lines,
    const char ** comment_lines,
    int num_rows,
    int size,
    const struct mtx_vector_coordinate_complex * data);

/**
 * `mtx_init_vector_coordinate_integer()` creates a sparse vector with
 * integer coefficients.
 */
int mtx_init_vector_coordinate_integer(
    struct mtx * mtx,
    enum mtx_sorting sorting,
    enum mtx_ordering ordering,
    enum mtx_assembly assembly,
    int num_comment_lines,
    const char ** comment_lines,
    int num_rows,
    int size,
    const struct mtx_vector_coordinate_integer * data);

/**
 * `mtx_init_vector_coordinate_pattern()` creates a sparse vector with
 * boolean coefficients.
 */
int mtx_init_vector_coordinate_pattern(
    struct mtx * mtx,
    enum mtx_sorting sorting,
    enum mtx_ordering ordering,
    enum mtx_assembly assembly,
    int num_comment_lines,
    const char ** comment_lines,
    int num_rows,
    int size,
    const struct mtx_vector_coordinate_pattern * data);

#endif
