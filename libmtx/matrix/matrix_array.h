/* This file is part of libmtx.
 *
 * Copyright (C) 2021 James D. Trotter
 *
 * libmtx is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * libmtx is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with libmtx.  If not, see <https://www.gnu.org/licenses/>.
 *
 * Authors: James D. Trotter <james@simula.no>
 * Last modified: 2021-10-05
 *
 * Data structures for matrices in array format.
 */

#ifndef LIBMTX_MATRIX_ARRAY_H
#define LIBMTX_MATRIX_ARRAY_H

#include <libmtx/libmtx-config.h>

#include <libmtx/mtx/precision.h>
#include <libmtx/util/field.h>

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

struct mtxfile;

/**
 * `mtxmatrix_array' represents a matrix in array format.
 */
struct mtxmatrix_array
{
    /**
     * `field' is the matrix field: `real', `complex', `integer' or
     * `pattern'.
     */
    enum mtx_field_ field;

    /**
     * `precision' is the precision used to store values.
     */
    enum mtx_precision precision;

    /**
     * `num_rows' is the number of matrix rows.
     */
    int num_rows;

    /**
     * `num_columns' is the number of matrix columns.
     */
    int num_columns;

    /**
     * `size' is the number of matrix elements, which is equal to
     * ‘num_rows*num_columns’.
     */
    int64_t size;

    /**
     * `data' contains values for each matrix entry.
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

/*
 * Memory management
 */

/**
 * `mtxmatrix_array_alloc()' allocates a matrix in array format.
 */
int mtxmatrix_array_alloc(
    struct mtxmatrix_array * matrix,
    enum mtx_field_ field,
    enum mtx_precision precision,
    int num_rows,
    int num_columns);

/**
 * `mtxmatrix_array_free()' frees storage allocated for a matrix.
 */
void mtxmatrix_array_free(
    struct mtxmatrix_array * matrix);

/**
 * `mtxmatrix_array_alloc_copy()' allocates a copy of a matrix without
 * initialising the values.
 */
int mtxmatrix_array_alloc_copy(
    struct mtxmatrix_array * dst,
    const struct mtxmatrix_array * src);

/**
 * `mtxmatrix_array_init_copy()' allocates a copy of a matrix and also
 * copies the values.
 */
int mtxmatrix_array_init_copy(
    struct mtxmatrix_array * dst,
    const struct mtxmatrix_array * src);

/*
 * Matrix initialisation
 */

/**
 * `mtxmatrix_array_init_real_single()' allocates and initialises a
 * matrix in array format with real, single precision coefficients.
 */
int mtxmatrix_array_init_real_single(
    struct mtxmatrix_array * matrix,
    int num_rows,
    int num_columns,
    const float * data);

/**
 * `mtxmatrix_array_init_real_double()' allocates and initialises a
 * matrix in array format with real, double precision coefficients.
 */
int mtxmatrix_array_init_real_double(
    struct mtxmatrix_array * matrix,
    int num_rows,
    int num_columns,
    const double * data);

/**
 * `mtxmatrix_array_init_complex_single()' allocates and initialises a
 * matrix in array format with complex, single precision coefficients.
 */
int mtxmatrix_array_init_complex_single(
    struct mtxmatrix_array * matrix,
    int num_rows,
    int num_columns,
    const float (* data)[2]);

/**
 * `mtxmatrix_array_init_complex_double()' allocates and initialises a
 * matrix in array format with complex, double precision coefficients.
 */
int mtxmatrix_array_init_complex_double(
    struct mtxmatrix_array * matrix,
    int num_rows,
    int num_columns,
    const double (* data)[2]);

/**
 * `mtxmatrix_array_init_integer_single()' allocates and initialises a
 * matrix in array format with integer, single precision coefficients.
 */
int mtxmatrix_array_init_integer_single(
    struct mtxmatrix_array * matrix,
    int num_rows,
    int num_columns,
    const int32_t * data);

/**
 * `mtxmatrix_array_init_integer_double()' allocates and initialises a
 * matrix in array format with integer, double precision coefficients.
 */
int mtxmatrix_array_init_integer_double(
    struct mtxmatrix_array * matrix,
    int num_rows,
    int num_columns,
    const int64_t * data);

/*
 * Convert to and from Matrix Market format
 */

/**
 * `mtxmatrix_array_from_mtxfile()' converts a matrix in Matrix Market
 * format to a matrix.
 */
int mtxmatrix_array_from_mtxfile(
    struct mtxmatrix_array * matrix,
    const struct mtxfile * mtxfile);

/**
 * `mtxmatrix_array_to_mtxfile()' converts a matrix to a matrix in
 * Matrix Market format.
 */
int mtxmatrix_array_to_mtxfile(
    const struct mtxmatrix_array * matrix,
    struct mtxfile * mtxfile);

#endif