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
 * Last modified: 2021-08-16
 *
 * Data structures for vectors in array format.
 */

#ifndef LIBMTX_VECTOR_ARRAY_DATA_H
#define LIBMTX_VECTOR_ARRAY_DATA_H

#include <libmtx/mtx/header.h>
#include <libmtx/mtx/precision.h>

#include <stdint.h>

/**
 * `mtx_vector_array_data' is a data structure for representing data
 * associated with vectors in array format.
 */
struct mtx_vector_array_data
{
    /**
     * `field' is the field associated with the vector values: `real',
     * `complex' or `integer'.
     */
    enum mtx_field field;

    /**
     * `precision' is the precision associated with the vector values:
     * `single' or `double'.
     */
    enum mtx_precision precision;

    /**
     * `num_rows' is the number of rows in the vector.
     */
    int num_rows;

    /**
     * `num_columns' is the number of columns in the vector.
     */
    int num_columns;

    /**
     * `size' is the number of entries stored in the `data' array.
     */
    int64_t size;

    /**
     * `data' is used to store the vector values.
     *
     * The storage format of nonzero values depends on `field' and
     * `precision'.  Only the member of the `data' union that
     * corresponds to the vector's `field' and `precision' should be
     * used to access the underlying data arrays containing the vector
     * values.
     *
     * For example, if `field' is `real' and `precision' is `single',
     * then `data.real_single' is an array of `size' values of type
     * `float', which contains the values of the vector entries.
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
 * `mtx_vector_array_data_free()' frees resources associated with a
 * vector in array format.
 */
void mtx_vector_array_data_free(
    struct mtx_vector_array_data * mtxdata);

/**
 * `mtx_vector_array_data_alloc()' allocates data for a vector in
 * array format.
 */
int mtx_vector_array_data_alloc(
    struct mtx_vector_array_data * mtxdata,
    enum mtx_field field,
    enum mtx_precision precision,
    int64_t size);

/*
 * Array vector allocation and initialisation.
 */

/**
 * `mtx_vector_array_data_init_real_single()' creates data for a
 * vector with real, single-precision floating point coefficients.
 */
int mtx_vector_array_data_init_real_single(
    struct mtx_vector_array_data * mtxdata,
    int64_t size,
    const float * data);

/**
 * `mtx_vector_array_data_init_real_double()' creates data for a
 * vector with real, double-precision floating point coefficients.
 */
int mtx_vector_array_data_init_real_double(
    struct mtx_vector_array_data * mtxdata,
    int64_t size,
    const double * data);

/**
 * `mtx_vector_array_data_init_complex_single()' creates data for a
 * vector with complex, single-precision floating point coefficients.
 */
int mtx_vector_array_data_init_complex_single(
    struct mtx_vector_array_data * mtxdata,
    int64_t size,
    const float (* data)[2]);

/**
 * `mtx_vector_array_data_init_complex_double()' creates data for a
 * vector with complex, double-precision floating point coefficients.
 */
int mtx_vector_array_data_init_complex_double(
    struct mtx_vector_array_data * mtxdata,
    int64_t size,
    const double (* data)[2]);

/**
 * `mtx_vector_array_data_init_integer_single()' creates data for a
 * vector with integer, single-precision coefficients.
 */
int mtx_vector_array_data_init_integer_single(
    struct mtx_vector_array_data * mtxdata,
    int64_t size,
    const int32_t * data);

/**
 * `mtx_vector_array_data_init_integer_double()' creates data for a
 * vector with integer, double-precision coefficients.
 */
int mtx_vector_array_data_init_integer_double(
    struct mtx_vector_array_data * mtxdata,
    int64_t size,
    const int64_t * data);

/**
 * `mtx_vector_array_data_copy_alloc()' allocates a copy of a vector
 * without copying the vector values.
 */
int mtx_vector_array_data_copy_alloc(
    struct mtx_vector_array_data * dst,
    const struct mtx_vector_array_data * src);

/**
 * `mtx_vector_array_data_copy_init()' creates a copy of a vector and
 * also copies vector values.
 */
int mtx_vector_array_data_copy_init(
    struct mtx_vector_array_data * dst,
    const struct mtx_vector_array_data * src);

/**
 * `mtx_vector_array_data_set_zero()' zeros a vector.
 */
int mtx_vector_array_data_set_zero(
    struct mtx_vector_array_data * mtxdata);

/**
 * `mtx_vector_array_data_set_constant_real_single()' sets every value
 * of a vector equal to a constant, single precision floating point
 * number.
 */
int mtx_vector_array_data_set_constant_real_single(
    struct mtx_vector_array_data * mtxdata,
    float a);

/**
 * `mtx_vector_array_data_set_constant_real_double()' sets every value
 * of a vector equal to a constant, double precision floating point
 * number.
 */
int mtx_vector_array_data_set_constant_real_double(
    struct mtx_vector_array_data * mtxdata,
    double a);

/**
 * `mtx_vector_array_data_set_constant_complex_single()' sets every
 * value of a vector equal to a constant, single precision floating
 * point complex number.
 */
int mtx_vector_array_data_set_constant_complex_single(
    struct mtx_vector_array_data * mtxdata,
    float a[2]);

/**
 * `mtx_vector_array_data_set_constant_complex_double()' sets every
 * value of a vector equal to a constant, double precision floating
 * point complex number.
 */
int mtx_vector_array_data_set_constant_complex_double(
    struct mtx_vector_array_data * mtxdata,
    double a[2]);

/**
 * `mtx_vector_array_data_set_constant_integer_single()' sets every
 * value of a vector equal to a constant integer.
 */
int mtx_vector_array_data_set_constant_integer_single(
    struct mtx_vector_array_data * mtxdata,
    int32_t a);

/**
 * `mtx_vector_array_data_set_constant_integer_double()' sets every
 * value of a vector equal to a constant integer.
 */
int mtx_vector_array_data_set_constant_integer_double(
    struct mtx_vector_array_data * mtxdata,
    int64_t a);

#endif
