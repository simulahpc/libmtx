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
 * Last modified: 2021-09-20
 *
 * Data structures for vectors in array format.
 */

#ifndef LIBMTX_VECTOR_ARRAY_H
#define LIBMTX_VECTOR_ARRAY_H

#include <libmtx/libmtx-config.h>

#include <libmtx/mtx/precision.h>
#include <libmtx/util/field.h>

#ifdef LIBMTX_HAVE_LIBZ
#include <zlib.h>
#endif

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

struct mtxfile;

/**
 * `mtxvector_array' represents a vector in array format.
 */
struct mtxvector_array
{
    /**
     * `field' is the vector field: `real', `complex', `integer' or
     * `pattern'.
     */
    enum mtx_field_ field;

    /**
     * `precision' is the precision used to store values.
     */
    enum mtx_precision precision;

    /**
     * `size' is the number of vector elements.
     */
    int size;

    /**
     * `data' contains the data lines of the vector.
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
 * `mtxvector_array_free()' frees storage allocated for a vector.
 */
void mtxvector_array_free(
    struct mtxvector_array * vector);

/**
 * `mtxvector_array_alloc_copy()' allocates a copy of a vector without
 * initialising the values.
 */
int mtxvector_array_alloc_copy(
    struct mtxvector_array * dst,
    const struct mtxvector_array * src);

/**
 * `mtxvector_array_init_copy()' allocates a copy of a vector and also
 * copies the values.
 */
int mtxvector_array_init_copy(
    struct mtxvector_array * dst,
    const struct mtxvector_array * src);

/*
 * Vector array formats
 */

/**
 * `mtxvector_array_alloc()' allocates a vector in array format.
 */
int mtxvector_array_alloc(
    struct mtxvector_array * vector,
    enum mtx_field_ field,
    enum mtx_precision precision,
    int size);

/**
 * `mtxvector_array_init_real_single()' allocates and initialises a
 * vector in array format with real, single precision coefficients.
 */
int mtxvector_array_init_real_single(
    struct mtxvector_array * vector,
    int size,
    const float * data);

/**
 * `mtxvector_array_init_real_double()' allocates and initialises a
 * vector in array format with real, double precision coefficients.
 */
int mtxvector_array_init_real_double(
    struct mtxvector_array * vector,
    int size,
    const double * data);

/**
 * `mtxvector_array_init_complex_single()' allocates and initialises a
 * vector in array format with complex, single precision coefficients.
 */
int mtxvector_array_init_complex_single(
    struct mtxvector_array * vector,
    int size,
    const float (* data)[2]);

/**
 * `mtxvector_array_init_complex_double()' allocates and initialises a
 * vector in array format with complex, double precision coefficients.
 */
int mtxvector_array_init_complex_double(
    struct mtxvector_array * vector,
    int size,
    const double (* data)[2]);

/**
 * `mtxvector_array_init_integer_single()' allocates and initialises a
 * vector in array format with integer, single precision coefficients.
 */
int mtxvector_array_init_integer_single(
    struct mtxvector_array * vector,
    int size,
    const int32_t * data);

/**
 * `mtxvector_array_init_integer_double()' allocates and initialises a
 * vector in array format with integer, double precision coefficients.
 */
int mtxvector_array_init_integer_double(
    struct mtxvector_array * vector,
    int size,
    const int64_t * data);

/*
 * Modifying values
 */

/**
 * `mtxvector_array_set_constant_real_single()' sets every value of a
 * vector equal to a constant, single precision floating point number.
 */
int mtxvector_array_set_constant_real_single(
    struct mtxvector_array * vector,
    float a);

/**
 * `mtxvector_array_set_constant_real_double()' sets every value of a
 * vector equal to a constant, double precision floating point number.
 */
int mtxvector_array_set_constant_real_double(
    struct mtxvector_array * vector,
    double a);

/**
 * `mtxvector_array_set_constant_complex_single()' sets every value of
 * a vector equal to a constant, single precision floating point
 * complex number.
 */
int mtxvector_array_set_constant_complex_single(
    struct mtxvector_array * vector,
    float a[2]);

/**
 * `mtxvector_array_set_constant_complex_double()' sets every value of
 * a vector equal to a constant, double precision floating point
 * complex number.
 */
int mtxvector_array_set_constant_complex_double(
    struct mtxvector_array * vector,
    double a[2]);

/**
 * `mtxvector_array_set_constant_integer_single()' sets every value of
 * a vector equal to a constant integer.
 */
int mtxvector_array_set_constant_integer_single(
    struct mtxvector_array * vector,
    int32_t a);

/**
 * `mtxvector_array_set_constant_integer_double()' sets every value of
 * a vector equal to a constant integer.
 */
int mtxvector_array_set_constant_integer_double(
    struct mtxvector_array * vector,
    int64_t a);

/*
 * Convert to and from Matrix Market format
 */

/**
 * `mtxvector_array_from_mtxfile()' converts a vector in Matrix Market
 * format to a vector.
 */
int mtxvector_array_from_mtxfile(
    struct mtxvector_array * vector,
    const struct mtxfile * mtxfile);

/**
 * `mtxvector_array_to_mtxfile()' converts a vector to a vector in
 * Matrix Market format.
 */
int mtxvector_array_to_mtxfile(
    const struct mtxvector_array * vector,
    struct mtxfile * mtxfile);

/*
 * Level 1 BLAS operations
 */

/**
 * `mtxvector_array_swap()' swaps values of two vectors,
 * simultaneously performing ‘y <- x’ and ‘x <- y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_array_swap(
    struct mtxvector_array * x,
    struct mtxvector_array * y);

/**
 * `mtxvector_array_copy()' copies values of a vector, ‘y = x’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_array_copy(
    struct mtxvector_array * y,
    const struct mtxvector_array * x);

/**
 * `mtxvector_array_sscal()' scales a vector by a single precision
 * floating point scalar, `x = a*x'.
 */
int mtxvector_array_sscal(
    float a,
    struct mtxvector_array * x,
    int64_t * num_flops);

/**
 * `mtxvector_array_dscal()' scales a vector by a double precision
 * floating point scalar, `x = a*x'.
 */
int mtxvector_array_dscal(
    double a,
    struct mtxvector_array * x,
    int64_t * num_flops);

/**
 * `mtxvector_array_saxpy()' adds a vector to another one multiplied
 * by a single precision floating point value, ‘y = a*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_array_saxpy(
    float a,
    const struct mtxvector_array * x,
    struct mtxvector_array * y,
    int64_t * num_flops);

/**
 * `mtxvector_array_daxpy()' adds a vector to another one multiplied
 * by a double precision floating point value, ‘y = a*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_array_daxpy(
    double a,
    const struct mtxvector_array * x,
    struct mtxvector_array * y,
    int64_t * num_flops);

/**
 * `mtxvector_array_saypx()' multiplies a vector by a single precision
 * floating point scalar and adds another vector, ‘y = a*y + x’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_array_saypx(
    float a,
    struct mtxvector_array * y,
    const struct mtxvector_array * x,
    int64_t * num_flops);

/**
 * `mtxvector_array_daypx()' multiplies a vector by a double precision
 * floating point scalar and adds another vector, ‘y = a*y + x’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_array_daypx(
    double a,
    struct mtxvector_array * y,
    const struct mtxvector_array * x,
    int64_t * num_flops);

/**
 * `mtxvector_array_sdot()' computes the Euclidean dot product of two
 * vectors in single precision floating point.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_array_sdot(
    const struct mtxvector_array * x,
    const struct mtxvector_array * y,
    float * dot,
    int64_t * num_flops);

/**
 * `mtxvector_array_ddot()' computes the Euclidean dot product of two
 * vectors in double precision floating point.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_array_ddot(
    const struct mtxvector_array * x,
    const struct mtxvector_array * y,
    double * dot,
    int64_t * num_flops);

/**
 * `mtxvector_array_cdotu()' computes the product of the transpose of
 * a complex row vector with another complex row vector in single
 * precision floating point, ‘dot := x^T*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_array_cdotu(
    const struct mtxvector_array * x,
    const struct mtxvector_array * y,
    float (* dot)[2],
    int64_t * num_flops);

/**
 * `mtxvector_array_zdotu()' computes the product of the transpose of
 * a complex row vector with another complex row vector in double
 * precision floating point, ‘dot := x^T*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_array_zdotu(
    const struct mtxvector_array * x,
    const struct mtxvector_array * y,
    double (* dot)[2],
    int64_t * num_flops);

/**
 * `mtxvector_array_cdotc()' computes the Euclidean dot product of two
 * complex vectors in single precision floating point, ‘dot := x^H*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_array_cdotc(
    const struct mtxvector_array * x,
    const struct mtxvector_array * y,
    float (* dot)[2],
    int64_t * num_flops);

/**
 * `mtxvector_array_zdotc()' computes the Euclidean dot product of two
 * complex vectors in double precision floating point, ‘dot := x^H*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_array_zdotc(
    const struct mtxvector_array * x,
    const struct mtxvector_array * y,
    double (* dot)[2],
    int64_t * num_flops);

/**
 * `mtxvector_array_snrm2()' computes the Euclidean norm of a vector in
 * single precision floating point.
 */
int mtxvector_array_snrm2(
    const struct mtxvector_array * x,
    float * nrm2,
    int64_t * num_flops);

/**
 * `mtxvector_array_dnrm2()' computes the Euclidean norm of a vector in
 * double precision floating point.
 */
int mtxvector_array_dnrm2(
    const struct mtxvector_array * x,
    double * nrm2,
    int64_t * num_flops);

/**
 * `mtxvector_array_sasum()' computes the sum of absolute values
 * (1-norm) of a vector in single precision floating point.  If the
 * vector is complex-valued, then the sum of the absolute values of
 * the real and imaginaty parts is computed.
 */
int mtxvector_array_sasum(
    const struct mtxvector_array * x,
    float * asum,
    int64_t * num_flops);

/**
 * `mtxvector_array_dasum()' computes the sum of absolute values
 * (1-norm) of a vector in double precision floating point.  If the
 * vector is complex-valued, then the sum of the absolute values of
 * the real and imaginaty parts is computed.
 */
int mtxvector_array_dasum(
    const struct mtxvector_array * x,
    double * asum,
    int64_t * num_flops);

/**
 * `mtxvector_array_iamax()' finds the index of the first element
 * having the maximum absolute value.  If the vector is
 * complex-valued, then the index points to the first element having
 * the maximum sum of the absolute values of the real and imaginary
 * parts.
 */
int mtxvector_array_iamax(
    const struct mtxvector_array * x,
    int * iamax);

#endif
