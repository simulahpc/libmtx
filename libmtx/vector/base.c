/* This file is part of Libmtx.
 *
 * Copyright (C) 2022 James D. Trotter
 *
 * Libmtx is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Libmtx is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Libmtx.  If not, see <https://www.gnu.org/licenses/>.
 *
 * Authors: James D. Trotter <james@simula.no>
 * Last modified: 2022-06-06
 *
 * Data structures and routines for basic dense vectors.
 */

#include <libmtx/libmtx-config.h>

#include <libmtx/error.h>
#include <libmtx/vector/precision.h>
#include <libmtx/vector/field.h>
#include <libmtx/mtxfile/header.h>
#include <libmtx/mtxfile/mtxfile.h>
#include <libmtx/util/partition.h>
#include <libmtx/util/sort.h>
#include <libmtx/vector/base.h>
#include <libmtx/vector/packed.h>
#include <libmtx/vector/vector.h>

#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * vector properties
 */

/**
 * ‘mtxvector_base_field()’ gets the field of a vector.
 */
enum mtxfield mtxvector_base_field(const struct mtxvector_base * x)
{
    return x->field;
}

/**
 * ‘mtxvector_base_precision()’ gets the precision of a vector.
 */
enum mtxprecision mtxvector_base_precision(const struct mtxvector_base * x)
{
    return x->precision;
}

/**
 * ‘mtxvector_base_size()’ gets the size of a vector.
 */
int64_t mtxvector_base_size(const struct mtxvector_base * x)
{
    return x->size;
}

/*
 * Memory management
 */

/**
 * ‘mtxvector_base_free()’ frees storage allocated for a vector.
 */
void mtxvector_base_free(
    struct mtxvector_base * x)
{
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            free(x->data.real_single);
        } else if (x->precision == mtx_double) {
            free(x->data.real_double);
        }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            free(x->data.complex_single);
        } else if (x->precision == mtx_double) {
            free(x->data.complex_double);
        }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            free(x->data.integer_single);
        } else if (x->precision == mtx_double) {
            free(x->data.integer_double);
        }
    }
}

/**
 * ‘mtxvector_base_alloc_copy()’ allocates a copy of a vector without
 * initialising the values.
 */
int mtxvector_base_alloc_copy(
    struct mtxvector_base * dst,
    const struct mtxvector_base * src)
{
    return mtxvector_base_alloc(dst, src->field, src->precision, src->size);
}

/**
 * ‘mtxvector_base_init_copy()’ allocates a copy of a vector and also
 * copies the values.
 */
int mtxvector_base_init_copy(
    struct mtxvector_base * dst,
    const struct mtxvector_base * src)
{
    int err = mtxvector_base_alloc_copy(dst, src);
    if (err) return err;
    err = mtxvector_base_copy(dst, src);
    if (err) {
        mtxvector_base_free(dst);
        return err;
    }
    return MTX_SUCCESS;
}

/*
 * Allocation and initialisation
 */

/**
 * ‘mtxvector_base_alloc()’ allocates a vector.
 */
int mtxvector_base_alloc(
    struct mtxvector_base * x,
    enum mtxfield field,
    enum mtxprecision precision,
    int64_t size)
{
    if (field == mtx_field_real) {
        if (precision == mtx_single) {
            x->data.real_single = malloc(size * sizeof(*x->data.real_single));
            if (!x->data.real_single) return MTX_ERR_ERRNO;
        } else if (precision == mtx_double) {
            x->data.real_double = malloc(size * sizeof(*x->data.real_double));
            if (!x->data.real_double) return MTX_ERR_ERRNO;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (field == mtx_field_complex) {
        if (precision == mtx_single) {
            x->data.complex_single = malloc(size * sizeof(*x->data.complex_single));
            if (!x->data.complex_single) return MTX_ERR_ERRNO;
        } else if (precision == mtx_double) {
            x->data.complex_double = malloc(size * sizeof(*x->data.complex_double));
            if (!x->data.complex_double) return MTX_ERR_ERRNO;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (field == mtx_field_integer) {
        if (precision == mtx_single) {
            x->data.integer_single = malloc(size * sizeof(*x->data.integer_single));
            if (!x->data.integer_single) return MTX_ERR_ERRNO;
        } else if (precision == mtx_double) {
            x->data.integer_double = malloc(size * sizeof(*x->data.integer_double));
            if (!x->data.integer_double) return MTX_ERR_ERRNO;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (field == mtx_field_pattern) {
        x->data.pattern = NULL;
    } else { return MTX_ERR_INVALID_FIELD; }
    x->field = field;
    x->precision = precision;
    x->size = size;
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_init_real_single()’ allocates and initialises a
 * vector with real, single precision coefficients.
 */
int mtxvector_base_init_real_single(
    struct mtxvector_base * x,
    int64_t size,
    const float * data)
{
    int err = mtxvector_base_alloc(x, mtx_field_real, mtx_single, size);
    if (err) return err;
    for (int64_t k = 0; k < size; k++)
        x->data.real_single[k] = data[k];
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_init_real_double()’ allocates and initialises a
 * vector with real, double precision coefficients.
 */
int mtxvector_base_init_real_double(
    struct mtxvector_base * x,
    int64_t size,
    const double * data)
{
    int err = mtxvector_base_alloc(x, mtx_field_real, mtx_double, size);
    if (err) return err;
    for (int64_t k = 0; k < size; k++)
        x->data.real_double[k] = data[k];
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_init_complex_single()’ allocates and initialises a
 * vector with complex, single precision coefficients.
 */
int mtxvector_base_init_complex_single(
    struct mtxvector_base * x,
    int64_t size,
    const float (* data)[2])
{
    int err = mtxvector_base_alloc(x, mtx_field_complex, mtx_single, size);
    if (err) return err;
    for (int64_t k = 0; k < size; k++) {
        x->data.complex_single[k][0] = data[k][0];
        x->data.complex_single[k][1] = data[k][1];
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_init_complex_double()’ allocates and initialises a
 * vector with complex, double precision coefficients.
 */
int mtxvector_base_init_complex_double(
    struct mtxvector_base * x,
    int64_t size,
    const double (* data)[2])
{
    int err = mtxvector_base_alloc(x, mtx_field_complex, mtx_double, size);
    if (err) return err;
    for (int64_t k = 0; k < size; k++) {
        x->data.complex_double[k][0] = data[k][0];
        x->data.complex_double[k][1] = data[k][1];
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_init_integer_single()’ allocates and initialises a
 * vector with integer, single precision coefficients.
 */
int mtxvector_base_init_integer_single(
    struct mtxvector_base * x,
    int64_t size,
    const int32_t * data)
{
    int err = mtxvector_base_alloc(x, mtx_field_integer, mtx_single, size);
    if (err) return err;
    for (int64_t k = 0; k < size; k++)
        x->data.integer_single[k] = data[k];
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_init_integer_double()’ allocates and initialises a
 * vector with integer, double precision coefficients.
 */
int mtxvector_base_init_integer_double(
    struct mtxvector_base * x,
    int64_t size,
    const int64_t * data)
{
    int err = mtxvector_base_alloc(x, mtx_field_integer, mtx_double, size);
    if (err) return err;
    for (int64_t k = 0; k < size; k++)
        x->data.integer_double[k] = data[k];
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_init_pattern()’ allocates and initialises a vector
 * of ones.
 */
int mtxvector_base_init_pattern(
    struct mtxvector_base * x,
    int64_t size)
{
    return mtxvector_base_alloc(x, mtx_field_pattern, mtx_single, size);
}

/*
 * initialise vectors from strided arrays
 */

/**
 * ‘mtxvector_base_init_strided_real_single()’ allocates and
 * initialises a vector with real, single precision coefficients.
 */
int mtxvector_base_init_strided_real_single(
    struct mtxvector_base * x,
    int64_t size,
    int64_t stride,
    const float * data)
{
    int err = mtxvector_base_alloc(x, mtx_field_real, mtx_single, size);
    if (err) return err;
    err = mtxvector_base_set_real_single(x, size, stride, data);
    if (err) { mtxvector_base_free(x); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_init_strided_real_double()’ allocates and
 * initialises a vector with real, double precision coefficients.
 */
int mtxvector_base_init_strided_real_double(
    struct mtxvector_base * x,
    int64_t size,
    int64_t stride,
    const double * data)
{
    int err = mtxvector_base_alloc(x, mtx_field_real, mtx_double, size);
    if (err) return err;
    err = mtxvector_base_set_real_double(x, size, stride, data);
    if (err) { mtxvector_base_free(x); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_init_strided_complex_single()’ allocates and
 * initialises a vector with complex, single precision coefficients.
 */
int mtxvector_base_init_strided_complex_single(
    struct mtxvector_base * x,
    int64_t size,
    int64_t stride,
    const float (* data)[2])
{
    int err = mtxvector_base_alloc(x, mtx_field_complex, mtx_single, size);
    if (err) return err;
    err = mtxvector_base_set_complex_single(x, size, stride, data);
    if (err) { mtxvector_base_free(x); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_init_strided_complex_double()’ allocates and
 * initialises a vector with complex, double precision coefficients.
 */
int mtxvector_base_init_strided_complex_double(
    struct mtxvector_base * x,
    int64_t size,
    int64_t stride,
    const double (* data)[2])
{
    int err = mtxvector_base_alloc(x, mtx_field_complex, mtx_double, size);
    if (err) return err;
    err = mtxvector_base_set_complex_double(x, size, stride, data);
    if (err) { mtxvector_base_free(x); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_init_strided_integer_single()’ allocates and
 * initialises a vector with integer, single precision coefficients.
 */
int mtxvector_base_init_strided_integer_single(
    struct mtxvector_base * x,
    int64_t size,
    int64_t stride,
    const int32_t * data)
{
    int err = mtxvector_base_alloc(x, mtx_field_integer, mtx_single, size);
    if (err) return err;
    err = mtxvector_base_set_integer_single(x, size, stride, data);
    if (err) { mtxvector_base_free(x); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_init_strided_integer_double()’ allocates and
 * initialises a vector with integer, double precision coefficients.
 */
int mtxvector_base_init_strided_integer_double(
    struct mtxvector_base * x,
    int64_t size,
    int64_t stride,
    const int64_t * data)
{
    int err = mtxvector_base_alloc(x, mtx_field_integer, mtx_double, size);
    if (err) return err;
    err = mtxvector_base_set_integer_double(x, size, stride, data);
    if (err) { mtxvector_base_free(x); return err; }
    return MTX_SUCCESS;
}

/*
 * accessing values
 */

/**
 * ‘mtxvector_base_get_real_single()’ obtains the values of a vector
 * of single precision floating point numbers.
 *
 * The array ‘a’ must be large enough to store ‘size’ elements
 * separated by the given stride (in bytes), and ‘size’ must be
 * greater than or equal to the number of elements in the vector.
 */
int mtxvector_base_get_real_single(
    const struct mtxvector_base * x,
    int64_t size,
    int stride,
    float * a)
{
    if (x->field != mtx_field_real) return MTX_ERR_INVALID_FIELD;
    if (x->precision != mtx_single) return MTX_ERR_INVALID_PRECISION;
    if (size < x->size) return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    float * b = x->data.real_single;
    for (int64_t i = 0; i < x->size; i++)
        *(float *)((char *) a + i*stride) = b[i];
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_get_real_double()’ obtains the values of a vector
 * of double precision floating point numbers.
 *
 * The array ‘a’ must be large enough to store ‘size’ elements
 * separated by the given stride (in bytes), and ‘size’ must be
 * greater than or equal to the number of elements in the vector.
 */
int mtxvector_base_get_real_double(
    const struct mtxvector_base * x,
    int64_t size,
    int stride,
    double * a)
{
    if (x->field != mtx_field_real) return MTX_ERR_INVALID_FIELD;
    if (x->precision != mtx_double) return MTX_ERR_INVALID_PRECISION;
    if (size < x->size) return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    double * b = x->data.real_double;
    for (int64_t i = 0; i < x->size; i++)
        *(double *)((char *) a + i*stride) = b[i];
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_get_complex_single()’ obtains the values of a
 * vector of single precision floating point complex numbers.
 *
 * The array ‘a’ must be large enough to store ‘size’ elements
 * separated by the given stride (in bytes), and ‘size’ must be
 * greater than or equal to the number of elements in the vector.
 */
int mtxvector_base_get_complex_single(
    struct mtxvector_base * x,
    int64_t size,
    int stride,
    float (* a)[2])
{
    if (x->field != mtx_field_complex) return MTX_ERR_INVALID_FIELD;
    if (x->precision != mtx_single) return MTX_ERR_INVALID_PRECISION;
    if (size < x->size) return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    float (* b)[2] = x->data.complex_single;
    for (int64_t i = 0; i < x->size; i++) {
        (*(float (*)[2])((char *) a + i*stride))[0] = b[i][0];
        (*(float (*)[2])((char *) a + i*stride))[1] = b[i][1];
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_get_complex_double()’ obtains the values of a
 * vector of double precision floating point complex numbers.
 *
 * The array ‘a’ must be large enough to store ‘size’ elements
 * separated by the given stride (in bytes), and ‘size’ must be
 * greater than or equal to the number of elements in the vector.
 */
int mtxvector_base_get_complex_double(
    struct mtxvector_base * x,
    int64_t size,
    int stride,
    double (* a)[2])
{
    if (x->field != mtx_field_complex) return MTX_ERR_INVALID_FIELD;
    if (x->precision != mtx_double) return MTX_ERR_INVALID_PRECISION;
    if (size < x->size) return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    double (* b)[2] = x->data.complex_double;
    for (int64_t i = 0; i < x->size; i++) {
        (*(double (*)[2])((char *) a + i*stride))[0] = b[i][0];
        (*(double (*)[2])((char *) a + i*stride))[1] = b[i][1];
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_get_integer_single()’ obtains the values of a
 * vector of single precision integers.
 *
 * The array ‘a’ must be large enough to store ‘size’ elements
 * separated by the given stride (in bytes), and ‘size’ must be
 * greater than or equal to the number of elements in the vector.
 */
int mtxvector_base_get_integer_single(
    struct mtxvector_base * x,
    int64_t size,
    int stride,
    int32_t * a)
{
    if (x->field != mtx_field_integer) return MTX_ERR_INVALID_FIELD;
    if (x->precision != mtx_single) return MTX_ERR_INVALID_PRECISION;
    if (size < x->size) return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    int32_t * b = x->data.integer_single;
    for (int64_t i = 0; i < x->size; i++)
        *(int32_t *)((char *) a + i*stride) = b[i];
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_get_integer_double()’ obtains the values of a
 * vector of double precision integers.
 *
 * The array ‘a’ must be large enough to store ‘size’ elements
 * separated by the given stride (in bytes), and ‘size’ must be
 * greater than or equal to the number of elements in the vector.
 */
int mtxvector_base_get_integer_double(
    struct mtxvector_base * x,
    int64_t size,
    int stride,
    int64_t * a)
{
    if (x->field != mtx_field_integer) return MTX_ERR_INVALID_FIELD;
    if (x->precision != mtx_double) return MTX_ERR_INVALID_PRECISION;
    if (size < x->size) return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    int64_t * b = x->data.integer_double;
    for (int64_t i = 0; i < x->size; i++)
        *(int64_t *)((char *) a + i*stride) = b[i];
    return MTX_SUCCESS;
}

/*
 * Modifying values
 */

/**
 * ‘mtxvector_base_setzero()’ sets every value of a vector to zero.
 */
int mtxvector_base_setzero(
    struct mtxvector_base * x)
{
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            for (int k = 0; k < x->size; k++)
                x->data.real_single[k] = 0;
        } else if (x->precision == mtx_double) {
            for (int k = 0; k < x->size; k++)
                x->data.real_double[k] = 0;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            for (int k = 0; k < x->size; k++) {
                x->data.complex_single[k][0] = 0;
                x->data.complex_single[k][1] = 0;
            }
        } else if (x->precision == mtx_double) {
            for (int k = 0; k < x->size; k++) {
                x->data.complex_double[k][0] = 0;
                x->data.complex_double[k][1] = 0;
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            for (int k = 0; k < x->size; k++)
                x->data.integer_single[k] = 0;
        } else if (x->precision == mtx_double) {
            for (int k = 0; k < x->size; k++)
                x->data.integer_double[k] = 0;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_set_constant_real_single()’ sets every value of a
 * vector equal to a constant, single precision floating point number.
 */
int mtxvector_base_set_constant_real_single(
    struct mtxvector_base * x,
    float a)
{
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            for (int k = 0; k < x->size; k++)
                x->data.real_single[k] = a;
        } else if (x->precision == mtx_double) {
            for (int k = 0; k < x->size; k++)
                x->data.real_double[k] = a;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            for (int k = 0; k < x->size; k++) {
                x->data.complex_single[k][0] = a;
                x->data.complex_single[k][1] = 0;
            }
        } else if (x->precision == mtx_double) {
            for (int k = 0; k < x->size; k++) {
                x->data.complex_double[k][0] = a;
                x->data.complex_double[k][1] = 0;
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            for (int k = 0; k < x->size; k++)
                x->data.integer_single[k] = a;
        } else if (x->precision == mtx_double) {
            for (int k = 0; k < x->size; k++)
                x->data.integer_double[k] = a;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_set_constant_real_double()’ sets every value of a
 * vector equal to a constant, double precision floating point number.
 */
int mtxvector_base_set_constant_real_double(
    struct mtxvector_base * x,
    double a)
{
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            for (int k = 0; k < x->size; k++)
                x->data.real_single[k] = a;
        } else if (x->precision == mtx_double) {
            for (int k = 0; k < x->size; k++)
                x->data.real_double[k] = a;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            for (int k = 0; k < x->size; k++) {
                x->data.complex_single[k][0] = a;
                x->data.complex_single[k][1] = 0;
            }
        } else if (x->precision == mtx_double) {
            for (int k = 0; k < x->size; k++) {
                x->data.complex_double[k][0] = a;
                x->data.complex_double[k][1] = 0;
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            for (int k = 0; k < x->size; k++)
                x->data.integer_single[k] = a;
        } else if (x->precision == mtx_double) {
            for (int k = 0; k < x->size; k++)
                x->data.integer_double[k] = a;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_set_constant_complex_single()’ sets every value of
 * a vector equal to a constant, single precision floating point
 * complex number.
 */
int mtxvector_base_set_constant_complex_single(
    struct mtxvector_base * x,
    float a[2])
{
    if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            for (int k = 0; k < x->size; k++) {
                x->data.complex_single[k][0] = a[0];
                x->data.complex_single[k][1] = a[1];
            }
        } else if (x->precision == mtx_double) {
            for (int k = 0; k < x->size; k++) {
                x->data.complex_double[k][0] = a[0];
                x->data.complex_double[k][1] = a[1];
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_set_constant_complex_double()’ sets every value of
 * a vector equal to a constant, double precision floating point
 * complex number.
 */
int mtxvector_base_set_constant_complex_double(
    struct mtxvector_base * x,
    double a[2])
{
    if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            for (int k = 0; k < x->size; k++) {
                x->data.complex_single[k][0] = a[0];
                x->data.complex_single[k][1] = a[1];
            }
        } else if (x->precision == mtx_double) {
            for (int k = 0; k < x->size; k++) {
                x->data.complex_double[k][0] = a[0];
                x->data.complex_double[k][1] = a[1];
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_set_constant_integer_single()’ sets every value of
 * a vector equal to a constant integer.
 */
int mtxvector_base_set_constant_integer_single(
    struct mtxvector_base * x,
    int32_t a)
{
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            for (int k = 0; k < x->size; k++)
                x->data.real_single[k] = a;
        } else if (x->precision == mtx_double) {
            for (int k = 0; k < x->size; k++)
                x->data.real_double[k] = a;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            for (int k = 0; k < x->size; k++) {
                x->data.complex_single[k][0] = a;
                x->data.complex_single[k][1] = 0;
            }
        } else if (x->precision == mtx_double) {
            for (int k = 0; k < x->size; k++) {
                x->data.complex_double[k][0] = a;
                x->data.complex_double[k][1] = 0;
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            for (int k = 0; k < x->size; k++)
                x->data.integer_single[k] = a;
        } else if (x->precision == mtx_double) {
            for (int k = 0; k < x->size; k++)
                x->data.integer_double[k] = a;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_set_constant_integer_double()’ sets every value of
 * a vector equal to a constant integer.
 */
int mtxvector_base_set_constant_integer_double(
    struct mtxvector_base * x,
    int64_t a)
{
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            for (int k = 0; k < x->size; k++)
                x->data.real_single[k] = a;
        } else if (x->precision == mtx_double) {
            for (int k = 0; k < x->size; k++)
                x->data.real_double[k] = a;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            for (int k = 0; k < x->size; k++) {
                x->data.complex_single[k][0] = a;
                x->data.complex_single[k][1] = 0;
            }
        } else if (x->precision == mtx_double) {
            for (int k = 0; k < x->size; k++) {
                x->data.complex_double[k][0] = a;
                x->data.complex_double[k][1] = 0;
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            for (int k = 0; k < x->size; k++)
                x->data.integer_single[k] = a;
        } else if (x->precision == mtx_double) {
            for (int k = 0; k < x->size; k++)
                x->data.integer_double[k] = a;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_set_real_single()’ sets values of a vector based on
 * an array of single precision floating point numbers.
 */
int mtxvector_base_set_real_single(
    struct mtxvector_base * x,
    int64_t size,
    int stride,
    const float * a)
{
    if (x->field != mtx_field_real) return MTX_ERR_INVALID_FIELD;
    if (x->precision != mtx_single) return MTX_ERR_INVALID_PRECISION;
    if (x->size != size) return MTX_ERR_INCOMPATIBLE_SIZE;
    float * b = x->data.real_single;
    for (int64_t i = 0; i < size; i++)
        b[i] = *(const float *)((const char *) a + i*stride);
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_set_real_double()’ sets values of a vector based on
 * an array of double precision floating point numbers.
 */
int mtxvector_base_set_real_double(
    struct mtxvector_base * x,
    int64_t size,
    int stride,
    const double * a)
{
    if (x->field != mtx_field_real) return MTX_ERR_INVALID_FIELD;
    if (x->precision != mtx_double) return MTX_ERR_INVALID_PRECISION;
    if (x->size != size) return MTX_ERR_INCOMPATIBLE_SIZE;
    double * b = x->data.real_double;
    for (int64_t i = 0; i < size; i++)
        b[i] = *(const double *)((const char *) a + i*stride);
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_set_complex_single()’ sets values of a vector based
 * on an array of single precision floating point complex numbers.
 */
int mtxvector_base_set_complex_single(
    struct mtxvector_base * x,
    int64_t size,
    int stride,
    const float (*a)[2])
{
    if (x->field != mtx_field_complex) return MTX_ERR_INVALID_FIELD;
    if (x->precision != mtx_single) return MTX_ERR_INVALID_PRECISION;
    if (x->size != size) return MTX_ERR_INCOMPATIBLE_SIZE;
    float (*b)[2] = x->data.complex_single;
    for (int64_t i = 0; i < size; i++) {
        b[i][0] = (*(const float (*)[2])((const char *) a + i*stride))[0];
        b[i][1] = (*(const float (*)[2])((const char *) a + i*stride))[1];
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_set_complex_double()’ sets values of a vector based
 * on an array of double precision floating point complex numbers.
 */
int mtxvector_base_set_complex_double(
    struct mtxvector_base * x,
    int64_t size,
    int stride,
    const double (*a)[2])
{
    if (x->field != mtx_field_complex) return MTX_ERR_INVALID_FIELD;
    if (x->precision != mtx_double) return MTX_ERR_INVALID_PRECISION;
    if (x->size != size) return MTX_ERR_INCOMPATIBLE_SIZE;
    double (*b)[2] = x->data.complex_double;
    for (int64_t i = 0; i < size; i++) {
        b[i][0] = (*(const double (*)[2])((const char *) a + i*stride))[0];
        b[i][1] = (*(const double (*)[2])((const char *) a + i*stride))[1];
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_set_integer_single()’ sets values of a vector based
 * on an array of integers.
 */
int mtxvector_base_set_integer_single(
    struct mtxvector_base * x,
    int64_t size,
    int stride,
    const int32_t * a)
{
    if (x->field != mtx_field_integer) return MTX_ERR_INVALID_FIELD;
    if (x->precision != mtx_single) return MTX_ERR_INVALID_PRECISION;
    if (x->size != size) return MTX_ERR_INCOMPATIBLE_SIZE;
    int32_t * b = x->data.integer_single;
    for (int64_t i = 0; i < size; i++)
        b[i] = *(const int32_t *)((const char *) a + i*stride);
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_set_integer_double()’ sets values of a vector based
 * on an array of integers.
 */
int mtxvector_base_set_integer_double(
    struct mtxvector_base * x,
    int64_t size,
    int stride,
    const int64_t * a)
{
    if (x->field != mtx_field_integer) return MTX_ERR_INVALID_FIELD;
    if (x->precision != mtx_double) return MTX_ERR_INVALID_PRECISION;
    if (x->size != size) return MTX_ERR_INCOMPATIBLE_SIZE;
    int64_t * b = x->data.integer_double;
    for (int64_t i = 0; i < size; i++)
        b[i] = *(const int64_t *)((const char *) a + i*stride);
    return MTX_SUCCESS;
}

/*
 * Convert to and from Matrix Market format
 */

/**
 * ‘mtxvector_base_from_mtxfile()’ converts a vector in Matrix Market
 * format to a vector.
 */
int mtxvector_base_from_mtxfile(
    struct mtxvector_base * x,
    const struct mtxfile * mtxfile)
{
    if (mtxfile->header.object != mtxfile_vector)
        return MTX_ERR_INCOMPATIBLE_MTX_OBJECT;
    if (mtxfile->header.format == mtxfile_array) {
        if (mtxfile->header.field == mtxfile_real) {
            if (mtxfile->precision == mtx_single) {
                return mtxvector_base_init_real_single(
                    x, mtxfile->size.num_rows, mtxfile->data.array_real_single);
            } else if (mtxfile->precision == mtx_double) {
                return mtxvector_base_init_real_double(
                    x, mtxfile->size.num_rows, mtxfile->data.array_real_double);
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (mtxfile->header.field == mtxfile_complex) {
            if (mtxfile->precision == mtx_single) {
                return mtxvector_base_init_complex_single(
                    x, mtxfile->size.num_rows, mtxfile->data.array_complex_single);
            } else if (mtxfile->precision == mtx_double) {
                return mtxvector_base_init_complex_double(
                    x, mtxfile->size.num_rows, mtxfile->data.array_complex_double);
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (mtxfile->header.field == mtxfile_integer) {
            if (mtxfile->precision == mtx_single) {
                return mtxvector_base_init_integer_single(
                    x, mtxfile->size.num_rows, mtxfile->data.array_integer_single);
            } else if (mtxfile->precision == mtx_double) {
                return mtxvector_base_init_integer_double(
                    x, mtxfile->size.num_rows, mtxfile->data.array_integer_double);
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (mtxfile->header.field == mtxfile_pattern) {
                return mtxvector_base_init_pattern(x, mtxfile->size.num_rows);
        } else { return MTX_ERR_INVALID_MTX_FIELD; }
    } else if (mtxfile->header.format == mtxfile_coordinate) {
        struct mtxvector_packed xpacked;
        int err = mtxvector_packed_from_mtxfile(
            &xpacked, mtxfile, mtxvector_base);
        if (err) return err;
        err = mtxvector_base_alloc(
            x, xpacked.x.storage.base.field,
            xpacked.x.storage.base.precision, xpacked.size);
        if (err) {
            mtxvector_packed_free(&xpacked);
            return err;
        }
        err = mtxvector_base_setzero(x);
        if (err) {
            mtxvector_base_free(x);
            mtxvector_packed_free(&xpacked);
            return err;
        }
        err = mtxvector_base_ussc(x, &xpacked);
        if (err) {
            mtxvector_base_free(x);
            mtxvector_packed_free(&xpacked);
            return err;
        }
        mtxvector_packed_free(&xpacked);
    } else { return MTX_ERR_INVALID_MTX_FORMAT; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_to_mtxfile()’ converts a vector to a vector in
 * Matrix Market format.
 */
int mtxvector_base_to_mtxfile(
    struct mtxfile * mtxfile,
    const struct mtxvector_base * x,
    int64_t num_rows,
    const int64_t * idx,
    enum mtxfileformat mtxfmt)
{
    int err;
    if (mtxfmt == mtxfile_array) {
        if (x->field == mtx_field_real) {
            if (x->precision == mtx_single) {
                return mtxfile_init_vector_array_real_single(
                    mtxfile, x->size, x->data.real_single);
            } else if (x->precision == mtx_double) {
                return mtxfile_init_vector_array_real_double(
                    mtxfile, x->size, x->data.real_double);
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (x->field == mtx_field_complex) {
            if (x->precision == mtx_single) {
                return mtxfile_init_vector_array_complex_single(
                    mtxfile, x->size, x->data.complex_single);
            } else if (x->precision == mtx_double) {
                return mtxfile_init_vector_array_complex_double(
                    mtxfile, x->size, x->data.complex_double);
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (x->field == mtx_field_integer) {
            if (x->precision == mtx_single) {
                return mtxfile_init_vector_array_integer_single(
                    mtxfile, x->size, x->data.integer_single);
            } else if (x->precision == mtx_double) {
                return mtxfile_init_vector_array_integer_double(
                    mtxfile, x->size, x->data.integer_double);
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (x->field == mtx_field_pattern) {
            return MTX_ERR_INCOMPATIBLE_FIELD;
        } else { return MTX_ERR_INVALID_FIELD; }
    } else if (mtxfmt == mtxfile_coordinate) {
        if (x->field == mtx_field_real) {
            err = mtxfile_alloc_vector_coordinate(
                mtxfile, mtxfile_real, x->precision,
                idx ? num_rows : x->size, x->size);
            if (err) return err;
            if (x->precision == mtx_single) {
                for (int64_t k = 0; k < x->size; k++) {
                    mtxfile->data.vector_coordinate_real_single[k].i =
                        idx ? idx[k]+1 : k+1;
                    mtxfile->data.vector_coordinate_real_single[k].a =
                        x->data.real_single[k];
                }
            } else if (x->precision == mtx_double) {
                for (int64_t k = 0; k < x->size; k++) {
                    mtxfile->data.vector_coordinate_real_double[k].i =
                        idx ? idx[k]+1 : k+1;
                    mtxfile->data.vector_coordinate_real_double[k].a =
                        x->data.real_double[k];
                }
            } else {
                mtxfile_free(mtxfile);
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (x->field == mtx_field_complex) {
            err = mtxfile_alloc_vector_coordinate(
                mtxfile, mtxfile_complex, x->precision, num_rows, x->size);
            if (err) return err;
            if (x->precision == mtx_single) {
                for (int64_t k = 0; k < x->size; k++) {
                    mtxfile->data.vector_coordinate_complex_single[k].i =
                        idx ? idx[k]+1 : k+1;
                    mtxfile->data.vector_coordinate_complex_single[k].a[0] =
                        x->data.complex_single[k][0];
                    mtxfile->data.vector_coordinate_complex_single[k].a[1] =
                        x->data.complex_single[k][1];
                }
            } else if (x->precision == mtx_double) {
                for (int64_t k = 0; k < x->size; k++) {
                    mtxfile->data.vector_coordinate_complex_double[k].i =
                        idx ? idx[k]+1 : k+1;
                    mtxfile->data.vector_coordinate_complex_double[k].a[0] =
                        x->data.complex_double[k][0];
                    mtxfile->data.vector_coordinate_complex_double[k].a[1] =
                        x->data.complex_double[k][1];
                }
            } else {
                mtxfile_free(mtxfile);
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (x->field == mtx_field_integer) {
            err = mtxfile_alloc_vector_coordinate(
                mtxfile, mtxfile_integer, x->precision, num_rows, x->size);
            if (err) return err;
            if (x->precision == mtx_single) {
                for (int64_t k = 0; k < x->size; k++) {
                    mtxfile->data.vector_coordinate_integer_single[k].i =
                        idx ? idx[k]+1 : k+1;
                    mtxfile->data.vector_coordinate_integer_single[k].a =
                        x->data.integer_single[k];
                }
            } else if (x->precision == mtx_double) {
                for (int64_t k = 0; k < x->size; k++) {
                    mtxfile->data.vector_coordinate_integer_double[k].i =
                        idx ? idx[k]+1 : k+1;
                    mtxfile->data.vector_coordinate_integer_double[k].a =
                        x->data.integer_double[k];
                }
            } else {
                mtxfile_free(mtxfile);
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (x->field == mtx_field_pattern) {
            err = mtxfile_alloc_vector_coordinate(
                mtxfile, mtxfile_pattern, x->precision, num_rows, x->size);
            if (err) return err;
            for (int64_t k = 0; k < x->size; k++)
                mtxfile->data.vector_coordinate_pattern[k].i =
                    idx ? idx[k]+1 : k+1;
        } else { return MTX_ERR_INVALID_FIELD; }
    } else { return MTX_ERR_INVALID_MTX_FORMAT; }
    return MTX_SUCCESS;
}

/*
 * Partitioning
 */

/**
 * ‘mtxvector_base_split()’ splits a vector into multiple vectors
 * according to a given assignment of parts to each vector element.
 *
 * The partitioning of the vector elements is specified by the array
 * ‘parts’. The length of the ‘parts’ array is given by ‘size’, which
 * must match the size of the vector ‘src’. Each entry in the array is
 * an integer in the range ‘[0, num_parts)’ designating the part to
 * which the corresponding vector element belongs.
 *
 * The argument ‘dsts’ is an array of ‘num_parts’ pointers to objects
 * of type ‘struct mtxvector_base’. If successful, then ‘dsts[p]’
 * points to a vector consisting of elements from ‘src’ that belong to
 * the ‘p’th part, as designated by the ‘parts’ array.
 *
 * Finally, the argument ‘invperm’ may either be ‘NULL’, in which case
 * it is ignored, or it must point to an array of length ‘size’, which
 * is used to store the inverse permutation obtained from sorting the
 * vector elements in ascending order according to their assigned
 * parts. That is, ‘invperm[i]’ is the original position (before
 * sorting) of the vector element that now occupies the ‘i’th position
 * among the sorted elements.
 *
 * The caller is responsible for calling ‘mtxvector_base_free()’ to
 * free storage allocated for each vector in the ‘dsts’ array.
 */
int mtxvector_base_split(
    int num_parts,
    struct mtxvector_base ** dsts,
    const struct mtxvector_base * src,
    int64_t size,
    int * parts,
    int64_t * invperm)
{
    if (size != src->size) return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    bool sorted = true;
    for (int64_t k = 0; k < size; k++) {
        if (parts[k] < 0 || parts[k] >= num_parts)
            return MTX_ERR_INDEX_OUT_OF_BOUNDS;
        if (k > 0 && parts[k-1] > parts[k]) sorted = false;
    }

    /* sort by part number and invert the sorting permutation */
    bool free_invperm = !invperm;
    if (sorted) {
        if (!invperm) {
            invperm = malloc(size * sizeof(int64_t));
            if (!invperm) return MTX_ERR_ERRNO;
        }
        for (int64_t k = 0; k < size; k++) invperm[k] = k;
    } else {
        int64_t * perm = malloc(size * sizeof(int64_t));
        if (!perm) return MTX_ERR_ERRNO;
        int err = radix_sort_int(size, parts, perm);
        if (err) { free(perm); return err; }
        if (!invperm) {
            invperm = malloc(size * sizeof(int64_t));
            if (!invperm) { free(perm); return MTX_ERR_ERRNO; }
        }
        for (int64_t k = 0; k < size; k++) invperm[perm[k]] = k;
        free(perm);
    }

    /*
     * Extract each part by a) counting the number elements in the
     * part, b) allocating storage, and c) gathering vector elements
     * for the part.
     */
    int64_t offset = 0;
    for (int p = 0; p < num_parts; p++) {
        int64_t partsize = 0;
        while (offset+partsize < size && parts[offset+partsize] == p) partsize++;

        struct mtxvector_packed dst;
        dst.size = size;
        dst.num_nonzeros = partsize;
        dst.idx = &invperm[offset];
        dst.x.type = mtxvector_base;
        int err = mtxvector_base_alloc(
            &dst.x.storage.base, src->field, src->precision, partsize);
        if (err) {
            for (int q = p-1; q >= 0; q--) mtxvector_base_free(dsts[q]);
            free(invperm);
            return err;
        }

        err = mtxvector_base_usga(&dst, src);
        if (err) {
            mtxvector_base_free(&dst.x.storage.base);
            for (int q = p-1; q >= 0; q--) mtxvector_base_free(dsts[q]);
            if (free_invperm) free(invperm);
            return err;
        }
        *dsts[p] = dst.x.storage.base;
        offset += partsize;
    }
    if (free_invperm) free(invperm);
    return MTX_SUCCESS;
}

/*
 * Level 1 BLAS operations
 */

/**
 * ‘mtxvector_base_swap()’ swaps values of two vectors, simultaneously
 * performing ‘y <- x’ and ‘x <- y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_base_swap(
    struct mtxvector_base * x,
    struct mtxvector_base * y)
{
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            float * xdata = x->data.real_single;
            float * ydata = y->data.real_single;
            for (int64_t k = 0; k < x->size; k++) {
                float z = ydata[k];
                ydata[k] = xdata[k];
                xdata[k] = z;
            }
        } else if (x->precision == mtx_double) {
            double * xdata = x->data.real_double;
            double * ydata = y->data.real_double;
            for (int64_t k = 0; k < x->size; k++) {
                double z = ydata[k];
                ydata[k] = xdata[k];
                xdata[k] = z;
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            float (* xdata)[2] = x->data.complex_single;
            float (* ydata)[2] = y->data.complex_single;
            for (int64_t k = 0; k < x->size; k++) {
                float z[2] = {ydata[k][0], ydata[k][1]};
                ydata[k][0] = xdata[k][0];
                ydata[k][1] = xdata[k][1];
                xdata[k][0] = z[0];
                xdata[k][1] = z[1];
            }
        } else if (x->precision == mtx_double) {
            double (* xdata)[2] = x->data.complex_double;
            double (* ydata)[2] = y->data.complex_double;
            for (int64_t k = 0; k < x->size; k++) {
                double z[2] = {ydata[k][0], ydata[k][1]};
                ydata[k][0] = xdata[k][0];
                ydata[k][1] = xdata[k][1];
                xdata[k][0] = z[0];
                xdata[k][1] = z[1];
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            int32_t * xdata = x->data.integer_single;
            int32_t * ydata = y->data.integer_single;
            for (int64_t k = 0; k < x->size; k++) {
                int32_t z = ydata[k];
                ydata[k] = xdata[k];
                xdata[k] = z;
            }
        } else if (x->precision == mtx_double) {
            int64_t * xdata = x->data.integer_double;
            int64_t * ydata = y->data.integer_double;
            for (int64_t k = 0; k < x->size; k++) {
                int64_t z = ydata[k];
                ydata[k] = xdata[k];
                xdata[k] = z;
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_copy()’ copies values of a vector, ‘y = x’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_base_copy(
    struct mtxvector_base * y,
    const struct mtxvector_base * x)
{
    if (y->field != x->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (y->precision != x->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (y->size != x->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (y->field == mtx_field_real) {
        if (y->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            float * ydata = y->data.real_single;
            for (int64_t k = 0; k < y->size; k++)
                ydata[k] = xdata[k];
        } else if (y->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            double * ydata = y->data.real_double;
            for (int64_t k = 0; k < y->size; k++)
                ydata[k] = xdata[k];
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (y->field == mtx_field_complex) {
        if (y->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            float (* ydata)[2] = y->data.complex_single;
            for (int64_t k = 0; k < y->size; k++) {
                ydata[k][0] = xdata[k][0];
                ydata[k][1] = xdata[k][1];
            }
        } else if (y->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            double (* ydata)[2] = y->data.complex_double;
            for (int64_t k = 0; k < y->size; k++) {
                ydata[k][0] = xdata[k][0];
                ydata[k][1] = xdata[k][1];
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (y->field == mtx_field_integer) {
        if (y->precision == mtx_single) {
            const int32_t * xdata = x->data.integer_single;
            int32_t * ydata = y->data.integer_single;
            for (int64_t k = 0; k < y->size; k++)
                ydata[k] = xdata[k];
        } else if (y->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            int64_t * ydata = y->data.integer_double;
            for (int64_t k = 0; k < y->size; k++)
                ydata[k] = xdata[k];
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_sscal()’ scales a vector by a single precision
 * floating point scalar, ‘x = a*x’.
 */
int mtxvector_base_sscal(
    float a,
    struct mtxvector_base * x,
    int64_t * num_flops)
{
    if (a == 1) return MTX_SUCCESS;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            float * xdata = x->data.real_single;
            for (int64_t k = 0; k < x->size; k++)
                xdata[k] *= a;
            if (num_flops) *num_flops += x->size;
        } else if (x->precision == mtx_double) {
            double * xdata = x->data.real_double;
            for (int64_t k = 0; k < x->size; k++)
                xdata[k] *= a;
            if (num_flops) *num_flops += x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            float (* xdata)[2] = x->data.complex_single;
            for (int64_t k = 0; k < x->size; k++) {
                xdata[k][0] *= a;
                xdata[k][1] *= a;
            }
            if (num_flops) *num_flops += 2*x->size;
        } else if (x->precision == mtx_double) {
            double (* xdata)[2] = x->data.complex_double;
            for (int64_t k = 0; k < x->size; k++) {
                xdata[k][0] *= a;
                xdata[k][1] *= a;
            }
            if (num_flops) *num_flops += 2*x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            int32_t * xdata = x->data.integer_single;
            for (int64_t k = 0; k < x->size; k++)
                xdata[k] *= a;
            if (num_flops) *num_flops += x->size;
        } else if (x->precision == mtx_double) {
            int64_t * xdata = x->data.integer_double;
            for (int64_t k = 0; k < x->size; k++)
                xdata[k] *= a;
            if (num_flops) *num_flops += x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_dscal()’ scales a vector by a double precision
 * floating point scalar, ‘x = a*x’.
 */
int mtxvector_base_dscal(
    double a,
    struct mtxvector_base * x,
    int64_t * num_flops)
{
    if (a == 1) return MTX_SUCCESS;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            float * xdata = x->data.real_single;
            for (int64_t k = 0; k < x->size; k++)
                xdata[k] *= a;
            if (num_flops) *num_flops += x->size;
        } else if (x->precision == mtx_double) {
            double * xdata = x->data.real_double;
            for (int64_t k = 0; k < x->size; k++)
                xdata[k] *= a;
            if (num_flops) *num_flops += x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            float (* xdata)[2] = x->data.complex_single;
            for (int64_t k = 0; k < x->size; k++) {
                xdata[k][0] *= a;
                xdata[k][1] *= a;
            }
            if (num_flops) *num_flops += 2*x->size;
        } else if (x->precision == mtx_double) {
            double (* xdata)[2] = x->data.complex_double;
            for (int64_t k = 0; k < x->size; k++) {
                xdata[k][0] *= a;
                xdata[k][1] *= a;
            }
            if (num_flops) *num_flops += 2*x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            int32_t * xdata = x->data.integer_single;
            for (int64_t k = 0; k < x->size; k++)
                xdata[k] *= a;
            if (num_flops) *num_flops += x->size;
        } else if (x->precision == mtx_double) {
            int64_t * xdata = x->data.integer_double;
            for (int64_t k = 0; k < x->size; k++)
                xdata[k] *= a;
            if (num_flops) *num_flops += x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_cscal()’ scales a vector by a complex, single
 * precision floating point scalar, ‘x = (a+b*i)*x’.
 */
int mtxvector_base_cscal(
    float a[2],
    struct mtxvector_base * x,
    int64_t * num_flops)
{
    if (x->field != mtx_field_complex) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision == mtx_single) {
        float (* xdata)[2] = x->data.complex_single;
        for (int64_t k = 0; k < x->size; k++) {
            float c = xdata[k][0]*a[0] - xdata[k][1]*a[1];
            float d = xdata[k][0]*a[1] + xdata[k][1]*a[0];
            xdata[k][0] = c;
            xdata[k][1] = d;
        }
        if (num_flops) *num_flops += 6*x->size;
    } else if (x->precision == mtx_double) {
        double (* xdata)[2] = x->data.complex_double;
        for (int64_t k = 0; k < x->size; k++) {
            double c = xdata[k][0]*a[0] - xdata[k][1]*a[1];
            double d = xdata[k][0]*a[1] + xdata[k][1]*a[0];
            xdata[k][0] = c;
            xdata[k][1] = d;
        }
        if (num_flops) *num_flops += 6*x->size;
    } else { return MTX_ERR_INVALID_PRECISION; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_zscal()’ scales a vector by a complex, double
 * precision floating point scalar, ‘x = (a+b*i)*x’.
 */
int mtxvector_base_zscal(
    double a[2],
    struct mtxvector_base * x,
    int64_t * num_flops)
{
    if (x->field != mtx_field_complex) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision == mtx_single) {
        float (* xdata)[2] = x->data.complex_single;
        for (int64_t k = 0; k < x->size; k++) {
            float c = xdata[k][0]*a[0] - xdata[k][1]*a[1];
            float d = xdata[k][0]*a[1] + xdata[k][1]*a[0];
            xdata[k][0] = c;
            xdata[k][1] = d;
        }
        if (num_flops) *num_flops += 6*x->size;
    } else if (x->precision == mtx_double) {
        double (* xdata)[2] = x->data.complex_double;
        for (int64_t k = 0; k < x->size; k++) {
            double c = xdata[k][0]*a[0] - xdata[k][1]*a[1];
            double d = xdata[k][0]*a[1] + xdata[k][1]*a[0];
            xdata[k][0] = c;
            xdata[k][1] = d;
        }
        if (num_flops) *num_flops += 6*x->size;
    } else { return MTX_ERR_INVALID_PRECISION; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_saxpy()’ adds a vector to another one multiplied by
 * a single precision floating point value, ‘y = a*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_base_saxpy(
    float a,
    const struct mtxvector_base * x,
    struct mtxvector_base * y,
    int64_t * num_flops)
{
    if (y->field != x->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (y->precision != x->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (y->size != x->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (y->field == mtx_field_real) {
        if (y->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            float * ydata = y->data.real_single;
            for (int64_t k = 0; k < y->size; k++)
                ydata[k] += a*xdata[k];
            if (num_flops) *num_flops += 2*y->size;
        } else if (y->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            double * ydata = y->data.real_double;
            for (int64_t k = 0; k < y->size; k++)
                ydata[k] += a*xdata[k];
            if (num_flops) *num_flops += 2*y->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (y->field == mtx_field_complex) {
        if (y->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            float (* ydata)[2] = y->data.complex_single;
            for (int64_t k = 0; k < y->size; k++) {
                ydata[k][0] += a*xdata[k][0];
                ydata[k][1] += a*xdata[k][1];
            }
            if (num_flops) *num_flops += 4*y->size;
        } else if (y->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            double (* ydata)[2] = y->data.complex_double;
            for (int64_t k = 0; k < y->size; k++) {
                ydata[k][0] += a*xdata[k][0];
                ydata[k][1] += a*xdata[k][1];
            }
            if (num_flops) *num_flops += 4*y->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (y->field == mtx_field_integer) {
        if (y->precision == mtx_single) {
            const int32_t * xdata = x->data.integer_single;
            int32_t * ydata = y->data.integer_single;
            for (int64_t k = 0; k < y->size; k++)
                ydata[k] += a*xdata[k];
            if (num_flops) *num_flops += 2*y->size;
        } else if (y->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            int64_t * ydata = y->data.integer_double;
            for (int64_t k = 0; k < y->size; k++)
                ydata[k] += a*xdata[k];
            if (num_flops) *num_flops += 2*y->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_daxpy()’ adds a vector to another one multiplied by
 * a double precision floating point value, ‘y = a*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_base_daxpy(
    double a,
    const struct mtxvector_base * x,
    struct mtxvector_base * y,
    int64_t * num_flops)
{
    if (y->field != x->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (y->precision != x->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (y->size != x->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (y->field == mtx_field_real) {
        if (y->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            float * ydata = y->data.real_single;
            for (int64_t k = 0; k < y->size; k++)
                ydata[k] += a*xdata[k];
            if (num_flops) *num_flops += 2*y->size;
        } else if (y->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            double * ydata = y->data.real_double;
            for (int64_t k = 0; k < y->size; k++)
                ydata[k] += a*xdata[k];
            if (num_flops) *num_flops += 2*y->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (y->field == mtx_field_complex) {
        if (y->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            float (* ydata)[2] = y->data.complex_single;
            for (int64_t k = 0; k < y->size; k++) {
                ydata[k][0] += a*xdata[k][0];
                ydata[k][1] += a*xdata[k][1];
            }
            if (num_flops) *num_flops += 4*y->size;
        } else if (y->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            double (* ydata)[2] = y->data.complex_double;
            for (int64_t k = 0; k < y->size; k++) {
                ydata[k][0] += a*xdata[k][0];
                ydata[k][1] += a*xdata[k][1];
            }
            if (num_flops) *num_flops += 4*y->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (y->field == mtx_field_integer) {
        if (y->precision == mtx_single) {
            const int32_t * xdata = x->data.integer_single;
            int32_t * ydata = y->data.integer_single;
            for (int64_t k = 0; k < y->size; k++)
                ydata[k] += a*xdata[k];
            if (num_flops) *num_flops += 2*y->size;
        } else if (y->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            int64_t * ydata = y->data.integer_double;
            for (int64_t k = 0; k < y->size; k++)
                ydata[k] += a*xdata[k];
            if (num_flops) *num_flops += 2*y->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_saypx()’ multiplies a vector by a single precision
 * floating point scalar and adds another vector, ‘y = a*y + x’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_base_saypx(
    float a,
    struct mtxvector_base * y,
    const struct mtxvector_base * x,
    int64_t * num_flops)
{
    if (y->field != x->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (y->precision != x->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (y->size != x->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (y->field == mtx_field_real) {
        if (y->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            float * ydata = y->data.real_single;
            for (int64_t k = 0; k < y->size; k++)
                ydata[k] = a*ydata[k]+xdata[k];
            if (num_flops) *num_flops += 2*y->size;
        } else if (y->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            double * ydata = y->data.real_double;
            for (int64_t k = 0; k < y->size; k++)
                ydata[k] = a*ydata[k]+xdata[k];
            if (num_flops) *num_flops += 2*y->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (y->field == mtx_field_complex) {
        if (y->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            float (* ydata)[2] = y->data.complex_single;
            for (int64_t k = 0; k < y->size; k++) {
                ydata[k][0] = a*ydata[k][0]+xdata[k][0];
                ydata[k][1] = a*ydata[k][1]+xdata[k][1];
            }
            if (num_flops) *num_flops += 4*y->size;
        } else if (y->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            double (* ydata)[2] = y->data.complex_double;
            for (int64_t k = 0; k < y->size; k++) {
                ydata[k][0] = a*ydata[k][0]+xdata[k][0];
                ydata[k][1] = a*ydata[k][1]+xdata[k][1];
            }
            if (num_flops) *num_flops += 4*y->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (y->field == mtx_field_integer) {
        if (y->precision == mtx_single) {
            const int32_t * xdata = x->data.integer_single;
            int32_t * ydata = y->data.integer_single;
            for (int64_t k = 0; k < y->size; k++)
                ydata[k] = a*ydata[k]+xdata[k];
            if (num_flops) *num_flops += 2*y->size;
        } else if (y->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            int64_t * ydata = y->data.integer_double;
            for (int64_t k = 0; k < y->size; k++)
                ydata[k] = a*ydata[k]+xdata[k];
            if (num_flops) *num_flops += 2*y->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_daypx()’ multiplies a vector by a double precision
 * floating point scalar and adds another vector, ‘y = a*y + x’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_base_daypx(
    double a,
    struct mtxvector_base * y,
    const struct mtxvector_base * x,
    int64_t * num_flops)
{
    if (y->field != x->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (y->precision != x->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (y->size != x->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (y->field == mtx_field_real) {
        if (y->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            float * ydata = y->data.real_single;
            for (int64_t k = 0; k < y->size; k++)
                ydata[k] = a*ydata[k]+xdata[k];
            if (num_flops) *num_flops += 2*y->size;
        } else if (y->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            double * ydata = y->data.real_double;
            for (int64_t k = 0; k < y->size; k++)
                ydata[k] = a*ydata[k]+xdata[k];
            if (num_flops) *num_flops += 2*y->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (y->field == mtx_field_complex) {
        if (y->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            float (* ydata)[2] = y->data.complex_single;
            for (int64_t k = 0; k < y->size; k++) {
                ydata[k][0] = a*ydata[k][0]+xdata[k][0];
                ydata[k][1] = a*ydata[k][1]+xdata[k][1];
            }
            if (num_flops) *num_flops += 4*y->size;
        } else if (y->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            double (* ydata)[2] = y->data.complex_double;
            for (int64_t k = 0; k < y->size; k++) {
                ydata[k][0] = a*ydata[k][0]+xdata[k][0];
                ydata[k][1] = a*ydata[k][1]+xdata[k][1];
            }
            if (num_flops) *num_flops += 4*y->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (y->field == mtx_field_integer) {
        if (y->precision == mtx_single) {
            const int32_t * xdata = x->data.integer_single;
            int32_t * ydata = y->data.integer_single;
            for (int64_t k = 0; k < y->size; k++)
                ydata[k] = a*ydata[k]+xdata[k];
            if (num_flops) *num_flops += 2*y->size;
        } else if (y->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            int64_t * ydata = y->data.integer_double;
            for (int64_t k = 0; k < y->size; k++)
                ydata[k] = a*ydata[k]+xdata[k];
            if (num_flops) *num_flops += 2*y->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_sdot()’ computes the Euclidean dot product of two
 * vectors in single precision floating point.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_base_sdot(
    const struct mtxvector_base * x,
    const struct mtxvector_base * y,
    float * dot,
    int64_t * num_flops)
{
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            const float * ydata = y->data.real_single;
            float c = 0;
            for (int64_t k = 0; k < x->size; k++)
                c += xdata[k]*ydata[k];
            *dot = c;
            if (num_flops) *num_flops += 2*x->size;
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            const double * ydata = y->data.real_double;
            float c = 0;
            for (int64_t k = 0; k < x->size; k++)
                c += xdata[k]*ydata[k];
            *dot = c;
            if (num_flops) *num_flops += 2*x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            const int32_t * xdata = x->data.integer_single;
            const int32_t * ydata = y->data.integer_single;
            float c = 0;
            for (int64_t k = 0; k < x->size; k++)
                c += xdata[k]*ydata[k];
            *dot = c;
            if (num_flops) *num_flops += 2*x->size;
        } else if (x->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            const int64_t * ydata = y->data.integer_double;
            float c = 0;
            for (int64_t k = 0; k < x->size; k++)
                c += xdata[k]*ydata[k];
            *dot = c;
            if (num_flops) *num_flops += 2*x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_ddot()’ computes the Euclidean dot product of two
 * vectors in double precision floating point.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_base_ddot(
    const struct mtxvector_base * x,
    const struct mtxvector_base * y,
    double * dot,
    int64_t * num_flops)
{
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            const float * ydata = y->data.real_single;
            double c = 0;
            for (int64_t k = 0; k < x->size; k++)
                c += xdata[k]*ydata[k];
            *dot = c;
            if (num_flops) *num_flops += 2*x->size;
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            const double * ydata = y->data.real_double;
            double c = 0;
            for (int64_t k = 0; k < x->size; k++)
                c += xdata[k]*ydata[k];
            *dot = c;
            if (num_flops) *num_flops += 2*x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            const int32_t * xdata = x->data.integer_single;
            const int32_t * ydata = y->data.integer_single;
            double c = 0;
            for (int64_t k = 0; k < x->size; k++)
                c += xdata[k]*ydata[k];
            *dot = c;
            if (num_flops) *num_flops += 2*x->size;
        } else if (x->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            const int64_t * ydata = y->data.integer_double;
            double c = 0;
            for (int64_t k = 0; k < x->size; k++)
                c += xdata[k]*ydata[k];
            *dot = c;
            if (num_flops) *num_flops += 2*x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_cdotu()’ computes the product of the transpose of
 * a complex row vector with another complex row vector in single
 * precision floating point, ‘dot := x^T*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_base_cdotu(
    const struct mtxvector_base * x,
    const struct mtxvector_base * y,
    float (* dot)[2],
    int64_t * num_flops)
{
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            const float (* ydata)[2] = y->data.complex_single;
            float c0 = 0, c1 = 0;
            for (int64_t k = 0; k < x->size; k++) {
                c0 += xdata[k][0]*ydata[k][0] - xdata[k][1]*ydata[k][1];
                c1 += xdata[k][0]*ydata[k][1] + xdata[k][1]*ydata[k][0];
            }
            (*dot)[0] = c0; (*dot)[1] = c1;
            if (num_flops) *num_flops += 8*x->size;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            const double (* ydata)[2] = y->data.complex_double;
            float c0 = 0, c1 = 0;
            for (int64_t k = 0; k < x->size; k++) {
                c0 += xdata[k][0]*ydata[k][0] - xdata[k][1]*ydata[k][1];
                c1 += xdata[k][0]*ydata[k][1] + xdata[k][1]*ydata[k][0];
            }
            (*dot)[0] = c0; (*dot)[1] = c1;
            if (num_flops) *num_flops += 8*x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else {
        (*dot)[1] = 0;
        return mtxvector_base_sdot(x, y, &(*dot)[0], num_flops);
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_zdotu()’ computes the product of the transpose of
 * a complex row vector with another complex row vector in double
 * precision floating point, ‘dot := x^T*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_base_zdotu(
    const struct mtxvector_base * x,
    const struct mtxvector_base * y,
    double (* dot)[2],
    int64_t * num_flops)
{
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            const float (* ydata)[2] = y->data.complex_single;
            double c0 = 0, c1 = 0;
            for (int64_t k = 0; k < x->size; k++) {
                c0 += xdata[k][0]*ydata[k][0] - xdata[k][1]*ydata[k][1];
                c1 += xdata[k][0]*ydata[k][1] + xdata[k][1]*ydata[k][0];
            }
            (*dot)[0] = c0; (*dot)[1] = c1;
            if (num_flops) *num_flops += 8*x->size;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            const double (* ydata)[2] = y->data.complex_double;
            double c0 = 0, c1 = 0;
            for (int64_t k = 0; k < x->size; k++) {
                c0 += xdata[k][0]*ydata[k][0] - xdata[k][1]*ydata[k][1];
                c1 += xdata[k][0]*ydata[k][1] + xdata[k][1]*ydata[k][0];
            }
            (*dot)[0] = c0; (*dot)[1] = c1;
            if (num_flops) *num_flops += 8*x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else {
        (*dot)[1] = 0;
        return mtxvector_base_ddot(x, y, &(*dot)[0], num_flops);
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_cdotc()’ computes the Euclidean dot product of two
 * complex vectors in single precision floating point, ‘dot := x^H*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_base_cdotc(
    const struct mtxvector_base * x,
    const struct mtxvector_base * y,
    float (* dot)[2],
    int64_t * num_flops)
{
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            const float (* ydata)[2] = y->data.complex_single;
            float c0 = 0, c1 = 0;
            for (int64_t k = 0; k < x->size; k++) {
                c0 += xdata[k][0]*ydata[k][0] + xdata[k][1]*ydata[k][1];
                c1 += xdata[k][0]*ydata[k][1] - xdata[k][1]*ydata[k][0];
            }
            (*dot)[0] = c0; (*dot)[1] = c1;
            if (num_flops) *num_flops += 8*x->size;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            const double (* ydata)[2] = y->data.complex_double;
            float c0 = 0, c1 = 0;
            for (int64_t k = 0; k < x->size; k++) {
                c0 += xdata[k][0]*ydata[k][0] + xdata[k][1]*ydata[k][1];
                c1 += xdata[k][0]*ydata[k][1] - xdata[k][1]*ydata[k][0];
            }
            (*dot)[0] = c0; (*dot)[1] = c1;
            if (num_flops) *num_flops += 8*x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else {
        (*dot)[1] = 0;
        return mtxvector_base_sdot(x, y, &(*dot)[0], num_flops);
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_zdotc()’ computes the Euclidean dot product of two
 * complex vectors in double precision floating point, ‘dot := x^H*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_base_zdotc(
    const struct mtxvector_base * x,
    const struct mtxvector_base * y,
    double (* dot)[2],
    int64_t * num_flops)
{
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            const float (* ydata)[2] = y->data.complex_single;
            double c0 = 0, c1 = 0;
            for (int64_t k = 0; k < x->size; k++) {
                c0 += xdata[k][0]*ydata[k][0] + xdata[k][1]*ydata[k][1];
                c1 += xdata[k][0]*ydata[k][1] - xdata[k][1]*ydata[k][0];
            }
            (*dot)[0] = c0; (*dot)[1] = c1;
            if (num_flops) *num_flops += 8*x->size;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            const double (* ydata)[2] = y->data.complex_double;
            double c0 = 0, c1 = 0;
            for (int64_t k = 0; k < x->size; k++) {
                c0 += xdata[k][0]*ydata[k][0] + xdata[k][1]*ydata[k][1];
                c1 += xdata[k][0]*ydata[k][1] - xdata[k][1]*ydata[k][0];
            }
            (*dot)[0] = c0; (*dot)[1] = c1;
            if (num_flops) *num_flops += 8*x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else {
        (*dot)[1] = 0;
        return mtxvector_base_ddot(x, y, &(*dot)[0], num_flops);
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_snrm2()’ computes the Euclidean norm of a vector
 * in single precision floating point.
 */
int mtxvector_base_snrm2(
    const struct mtxvector_base * x,
    float * nrm2,
    int64_t * num_flops)
{
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            float c = 0;
            for (int64_t k = 0; k < x->size; k++)
                c += xdata[k]*xdata[k];
            *nrm2 = sqrtf(c);
            if (num_flops) *num_flops += 2*x->size;
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            float c = 0;
            for (int64_t k = 0; k < x->size; k++)
                c += xdata[k]*xdata[k];
            *nrm2 = sqrtf(c);
            if (num_flops) *num_flops += 2*x->size;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            float c = 0;
            for (int64_t k = 0; k < x->size; k++)
                c += xdata[k][0]*xdata[k][0] + xdata[k][1]*xdata[k][1];
            *nrm2 = sqrtf(c);
            if (num_flops) *num_flops += 4*x->size;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            float c = 0;
            for (int64_t k = 0; k < x->size; k++)
                c += xdata[k][0]*xdata[k][0] + xdata[k][1]*xdata[k][1];
            *nrm2 = sqrtf(c);
            if (num_flops) *num_flops += 4*x->size;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            const int32_t * xdata = x->data.integer_single;
            float c = 0;
            for (int64_t k = 0; k < x->size; k++)
                c += xdata[k]*xdata[k];
            *nrm2 = sqrtf(c);
            if (num_flops) *num_flops += 2*x->size;
        } else if (x->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            float c = 0;
            for (int64_t k = 0; k < x->size; k++)
                c += xdata[k]*xdata[k];
            *nrm2 = sqrtf(c);
            if (num_flops) *num_flops += 2*x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_dnrm2()’ computes the Euclidean norm of a vector
 * in double precision floating point.
 */
int mtxvector_base_dnrm2(
    const struct mtxvector_base * x,
    double * nrm2,
    int64_t * num_flops)
{
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            double c = 0;
            for (int64_t k = 0; k < x->size; k++)
                c += xdata[k]*xdata[k];
            *nrm2 = sqrtf(c);
            if (num_flops) *num_flops += 2*x->size;
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            double c = 0;
            for (int64_t k = 0; k < x->size; k++)
                c += xdata[k]*xdata[k];
            *nrm2 = sqrtf(c);
            if (num_flops) *num_flops += 2*x->size;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            double c = 0;
            for (int64_t k = 0; k < x->size; k++)
                c += xdata[k][0]*xdata[k][0] + xdata[k][1]*xdata[k][1];
            *nrm2 = sqrtf(c);
            if (num_flops) *num_flops += 4*x->size;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            double c = 0;
            for (int64_t k = 0; k < x->size; k++)
                c += xdata[k][0]*xdata[k][0] + xdata[k][1]*xdata[k][1];
            *nrm2 = sqrtf(c);
            if (num_flops) *num_flops += 4*x->size;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            const int32_t * xdata = x->data.integer_single;
            double c = 0;
            for (int64_t k = 0; k < x->size; k++)
                c += xdata[k]*xdata[k];
            *nrm2 = sqrtf(c);
            if (num_flops) *num_flops += 2*x->size;
        } else if (x->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            double c = 0;
            for (int64_t k = 0; k < x->size; k++)
                c += xdata[k]*xdata[k];
            *nrm2 = sqrtf(c);
            if (num_flops) *num_flops += 2*x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_sasum()’ computes the sum of absolute values
 * (1-norm) of a vector in single precision floating point.  If the
 * vector is complex-valued, then the sum of the absolute values of
 * the real and imaginary parts is computed.
 */
int mtxvector_base_sasum(
    const struct mtxvector_base * x,
    float * asum,
    int64_t * num_flops)
{
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            float c = 0;
            for (int64_t k = 0; k < x->size; k++)
                c += fabsf(xdata[k]);
            *asum = c;
            if (num_flops) *num_flops += x->size;
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            float c = 0;
            for (int64_t k = 0; k < x->size; k++)
                c += fabs(xdata[k]);
            *asum = c;
            if (num_flops) *num_flops += x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            float c = 0;
            for (int64_t k = 0; k < x->size; k++)
                c += fabsf(xdata[k][0]) + fabsf(xdata[k][1]);
            *asum = c;
            if (num_flops) *num_flops += 2*x->size;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            float c = 0;
            for (int64_t k = 0; k < x->size; k++)
                c += fabs(xdata[k][0]) + fabs(xdata[k][1]);
            *asum = c;
            if (num_flops) *num_flops += 2*x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            const int32_t * xdata = x->data.integer_single;
            float c = 0;
            for (int64_t k = 0; k < x->size; k++)
                c += abs(xdata[k]);
            *asum = c;
            if (num_flops) *num_flops += x->size;
        } else if (x->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            float c = 0;
            for (int64_t k = 0; k < x->size; k++)
                c += llabs(xdata[k]);
            *asum = c;
            if (num_flops) *num_flops += x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_dasum()’ computes the sum of absolute values
 * (1-norm) of a vector in double precision floating point.  If the
 * vector is complex-valued, then the sum of the absolute values of
 * the real and imaginary parts is computed.
 */
int mtxvector_base_dasum(
    const struct mtxvector_base * x,
    double * asum,
    int64_t * num_flops)
{
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            double c = 0;
            for (int64_t k = 0; k < x->size; k++)
                c += fabsf(xdata[k]);
            *asum = c;
            if (num_flops) *num_flops += x->size;
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            double c = 0;
            for (int64_t k = 0; k < x->size; k++)
                c += fabs(xdata[k]);
            *asum = c;
            if (num_flops) *num_flops += x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            double c = 0;
            for (int64_t k = 0; k < x->size; k++)
                c += fabsf(xdata[k][0]) + fabsf(xdata[k][1]);
            *asum = c;
            if (num_flops) *num_flops += 2*x->size;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            double c = 0;
            for (int64_t k = 0; k < x->size; k++)
                c += fabs(xdata[k][0]) + fabs(xdata[k][1]);
            *asum = c;
            if (num_flops) *num_flops += 2*x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            const int32_t * xdata = x->data.integer_single;
            double c = 0;
            for (int64_t k = 0; k < x->size; k++)
                c += abs(xdata[k]);
            *asum = c;
            if (num_flops) *num_flops += x->size;
        } else if (x->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            double c = 0;
            for (int64_t k = 0; k < x->size; k++)
                c += llabs(xdata[k]);
            *asum = c;
            if (num_flops) *num_flops += x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_iamax()’ finds the index of the first element
 * having the maximum absolute value.  If the vector is
 * complex-valued, then the index points to the first element having
 * the maximum sum of the absolute values of the real and imaginary
 * parts.
 */
int mtxvector_base_iamax(
    const struct mtxvector_base * x,
    int * iamax)
{
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            *iamax = 0;
            float max = x->size > 0 ? fabsf(xdata[0]) : 0;
            for (int64_t k = 1; k < x->size; k++) {
                if (max < fabsf(xdata[k])) {
                    max = fabsf(xdata[k]);
                    *iamax = k;
                }
            }
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            *iamax = 0;
            double max = x->size > 0 ? fabs(xdata[0]) : 0;
            for (int64_t k = 1; k < x->size; k++) {
                if (max < fabs(xdata[k])) {
                    max = fabs(xdata[k]);
                    *iamax = k;
                }
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            *iamax = 0;
            float max = x->size > 0 ? fabsf(xdata[0][0]) + fabsf(xdata[0][1]) : 0;
            for (int64_t k = 1; k < x->size; k++) {
                if (max < fabsf(xdata[k][0]) + fabsf(xdata[k][1])) {
                    max = fabsf(xdata[k][0]) + fabsf(xdata[k][1]);
                    *iamax = k;
                }
            }
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            *iamax = 0;
            double max = x->size > 0 ? fabs(xdata[0][0]) + fabs(xdata[0][1]) : 0;
            for (int64_t k = 1; k < x->size; k++) {
                if (max < fabs(xdata[k][0]) + fabs(xdata[k][1])) {
                    max = fabs(xdata[k][0]) + fabs(xdata[k][1]);
                    *iamax = k;
                }
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            const int32_t * xdata = x->data.integer_single;
            *iamax = 0;
            int32_t max = x->size > 0 ? abs(xdata[0]) : 0;
            for (int64_t k = 1; k < x->size; k++) {
                if (max < abs(xdata[k])) {
                    max = abs(xdata[k]);
                    *iamax = k;
                }
            }
        } else if (x->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            *iamax = 0;
            int64_t max = x->size > 0 ? llabs(xdata[0]) : 0;
            for (int64_t k = 1; k < x->size; k++) {
                if (max < llabs(xdata[k])) {
                    max = llabs(xdata[k]);
                    *iamax = k;
                }
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/*
 * Level 1 Sparse BLAS operations.
 *
 * See I. Duff, M. Heroux and R. Pozo, “An Overview of the Sparse
 * Basic Linear Algebra Subprograms: The New Standard from the BLAS
 * Technical Forum,” ACM TOMS, Vol. 28, No. 2, June 2002, pp. 239-267.
 */

/**
 * ‘mtxvector_base_ussdot()’ computes the Euclidean dot product of two
 * vectors in single precision floating point.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. The vector ‘x’ is a sparse vector in packed form. Repeated
 * indices in the packed vector are not allowed, otherwise the result
 * is undefined.
 */
int mtxvector_base_ussdot(
    const struct mtxvector_packed * xpacked,
    const struct mtxvector_base * y,
    float * dot,
    int64_t * num_flops)
{
    if (xpacked->x.type != mtxvector_base) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
    const struct mtxvector_base * x = &xpacked->x.storage.base;
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (xpacked->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (xpacked->num_nonzeros != x->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    const int64_t * idx = xpacked->idx;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            const float * ydata = y->data.real_single;
            float c = 0;
            for (int64_t k = 0; k < x->size; k++)
                c += xdata[k]*ydata[idx[k]];
            *dot = c;
            if (num_flops) *num_flops += 2*x->size;
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            const double * ydata = y->data.real_double;
            float c = 0;
            for (int64_t k = 0; k < x->size; k++)
                c += xdata[k]*ydata[idx[k]];
            *dot = c;
            if (num_flops) *num_flops += 2*x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            const int32_t * xdata = x->data.integer_single;
            const int32_t * ydata = y->data.integer_single;
            float c = 0;
            for (int64_t k = 0; k < x->size; k++)
                c += xdata[k]*ydata[idx[k]];
            *dot = c;
            if (num_flops) *num_flops += 2*x->size;
        } else if (x->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            const int64_t * ydata = y->data.integer_double;
            float c = 0;
            for (int64_t k = 0; k < x->size; k++)
                c += xdata[k]*ydata[idx[k]];
            *dot = c;
            if (num_flops) *num_flops += 2*x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_usddot()’ computes the Euclidean dot product of two
 * vectors in double precision floating point.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. The vector ‘x’ is a sparse vector in packed form. Repeated
 * indices in the packed vector are not allowed, otherwise the result
 * is undefined.
 */
int mtxvector_base_usddot(
    const struct mtxvector_packed * xpacked,
    const struct mtxvector_base * y,
    double * dot,
    int64_t * num_flops)
{
    if (xpacked->x.type != mtxvector_base) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
    const struct mtxvector_base * x = &xpacked->x.storage.base;
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (xpacked->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (xpacked->num_nonzeros != x->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    const int64_t * idx = xpacked->idx;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            const float * ydata = y->data.real_single;
            double c = 0;
            for (int64_t k = 0; k < x->size; k++)
                c += xdata[k]*ydata[idx[k]];
            *dot = c;
            if (num_flops) *num_flops += 2*x->size;
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            const double * ydata = y->data.real_double;
            double c = 0;
            for (int64_t k = 0; k < x->size; k++)
                c += xdata[k]*ydata[idx[k]];
            *dot = c;
            if (num_flops) *num_flops += 2*x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            const int32_t * xdata = x->data.integer_single;
            const int32_t * ydata = y->data.integer_single;
            double c = 0;
            for (int64_t k = 0; k < x->size; k++)
                c += xdata[k]*ydata[idx[k]];
            *dot = c;
            if (num_flops) *num_flops += 2*x->size;
        } else if (x->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            const int64_t * ydata = y->data.integer_double;
            double c = 0;
            for (int64_t k = 0; k < x->size; k++)
                c += xdata[k]*ydata[idx[k]];
            *dot = c;
            if (num_flops) *num_flops += 2*x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_uscdotu()’ computes the product of the transpose of
 * a complex row vector with another complex row vector in single
 * precision floating point, ‘dot := x^T*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. The vector ‘x’ is a sparse vector in packed form. Repeated
 * indices in the packed vector are not allowed, otherwise the result
 * is undefined.
 */
int mtxvector_base_uscdotu(
    const struct mtxvector_packed * xpacked,
    const struct mtxvector_base * y,
    float (* dot)[2],
    int64_t * num_flops)
{
    if (xpacked->x.type != mtxvector_base) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
    const struct mtxvector_base * x = &xpacked->x.storage.base;
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (xpacked->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (xpacked->num_nonzeros != x->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    const int64_t * idx = xpacked->idx;
    if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            const float (* ydata)[2] = y->data.complex_single;
            float c0 = 0, c1 = 0;
            for (int64_t k = 0; k < x->size; k++) {
                c0 += xdata[k][0]*ydata[idx[k]][0] - xdata[k][1]*ydata[idx[k]][1];
                c1 += xdata[k][0]*ydata[idx[k]][1] + xdata[k][1]*ydata[idx[k]][0];
            }
            (*dot)[0] = c0; (*dot)[1] = c1;
            if (num_flops) *num_flops += 8*x->size;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            const double (* ydata)[2] = y->data.complex_double;
            float c0 = 0, c1 = 0;
            for (int64_t k = 0; k < x->size; k++) {
                c0 += xdata[k][0]*ydata[idx[k]][0] - xdata[k][1]*ydata[idx[k]][1];
                c1 += xdata[k][0]*ydata[idx[k]][1] + xdata[k][1]*ydata[idx[k]][0];
            }
            (*dot)[0] = c0; (*dot)[1] = c1;
            if (num_flops) *num_flops += 8*x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else {
        (*dot)[1] = 0;
        return mtxvector_base_ussdot(xpacked, y, &(*dot)[0], num_flops);
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_uszdotu()’ computes the product of the transpose of
 * a complex row vector with another complex row vector in double
 * precision floating point, ‘dot := x^T*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. The vector ‘x’ is a sparse vector in packed form. Repeated
 * indices in the packed vector are not allowed, otherwise the result
 * is undefined.
 */
int mtxvector_base_uszdotu(
    const struct mtxvector_packed * xpacked,
    const struct mtxvector_base * y,
    double (* dot)[2],
    int64_t * num_flops)
{
    if (xpacked->x.type != mtxvector_base) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
    const struct mtxvector_base * x = &xpacked->x.storage.base;
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (xpacked->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (xpacked->num_nonzeros != x->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    const int64_t * idx = xpacked->idx;
    if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            const float (* ydata)[2] = y->data.complex_single;
            double c0 = 0, c1 = 0;
            for (int64_t k = 0; k < x->size; k++) {
                c0 += xdata[k][0]*ydata[idx[k]][0] - xdata[k][1]*ydata[idx[k]][1];
                c1 += xdata[k][0]*ydata[idx[k]][1] + xdata[k][1]*ydata[idx[k]][0];
            }
            (*dot)[0] = c0; (*dot)[1] = c1;
            if (num_flops) *num_flops += 8*x->size;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            const double (* ydata)[2] = y->data.complex_double;
            double c0 = 0, c1 = 0;
            for (int64_t k = 0; k < x->size; k++) {
                c0 += xdata[k][0]*ydata[idx[k]][0] - xdata[k][1]*ydata[idx[k]][1];
                c1 += xdata[k][0]*ydata[idx[k]][1] + xdata[k][1]*ydata[idx[k]][0];
            }
            (*dot)[0] = c0; (*dot)[1] = c1;
            if (num_flops) *num_flops += 8*x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else {
        (*dot)[1] = 0;
        return mtxvector_base_usddot(xpacked, y, &(*dot)[0], num_flops);
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_uscdotc()’ computes the Euclidean dot product of two
 * complex vectors in single precision floating point, ‘dot := x^H*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. The vector ‘x’ is a sparse vector in packed form. Repeated
 * indices in the packed vector are not allowed, otherwise the result
 * is undefined.
 */
int mtxvector_base_uscdotc(
    const struct mtxvector_packed * xpacked,
    const struct mtxvector_base * y,
    float (* dot)[2],
    int64_t * num_flops)
{
    if (xpacked->x.type != mtxvector_base) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
    const struct mtxvector_base * x = &xpacked->x.storage.base;
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (xpacked->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (xpacked->num_nonzeros != x->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    const int64_t * idx = xpacked->idx;
    if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            const float (* ydata)[2] = y->data.complex_single;
            float c0 = 0, c1 = 0;
            for (int64_t k = 0; k < x->size; k++) {
                c0 += xdata[k][0]*ydata[idx[k]][0] + xdata[k][1]*ydata[idx[k]][1];
                c1 += xdata[k][0]*ydata[idx[k]][1] - xdata[k][1]*ydata[idx[k]][0];
            }
            (*dot)[0] = c0; (*dot)[1] = c1;
            if (num_flops) *num_flops += 8*x->size;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            const double (* ydata)[2] = y->data.complex_double;
            float c0 = 0, c1 = 0;
            for (int64_t k = 0; k < x->size; k++) {
                c0 += xdata[k][0]*ydata[idx[k]][0] + xdata[k][1]*ydata[idx[k]][1];
                c1 += xdata[k][0]*ydata[idx[k]][1] - xdata[k][1]*ydata[idx[k]][0];
            }
            (*dot)[0] = c0; (*dot)[1] = c1;
            if (num_flops) *num_flops += 8*x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else {
        (*dot)[1] = 0;
        return mtxvector_base_ussdot(xpacked, y, &(*dot)[0], num_flops);
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_uszdotc()’ computes the Euclidean dot product of two
 * complex vectors in double precision floating point, ‘dot := x^H*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. The vector ‘x’ is a sparse vector in packed form. Repeated
 * indices in the packed vector are not allowed, otherwise the result
 * is undefined.
 */
int mtxvector_base_uszdotc(
    const struct mtxvector_packed * xpacked,
    const struct mtxvector_base * y,
    double (* dot)[2],
    int64_t * num_flops)
{
    if (xpacked->x.type != mtxvector_base) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
    const struct mtxvector_base * x = &xpacked->x.storage.base;
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (xpacked->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (xpacked->num_nonzeros != x->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    const int64_t * idx = xpacked->idx;
    if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            const float (* ydata)[2] = y->data.complex_single;
            double c0 = 0, c1 = 0;
            for (int64_t k = 0; k < x->size; k++) {
                c0 += xdata[k][0]*ydata[idx[k]][0] + xdata[k][1]*ydata[idx[k]][1];
                c1 += xdata[k][0]*ydata[idx[k]][1] - xdata[k][1]*ydata[idx[k]][0];
            }
            (*dot)[0] = c0; (*dot)[1] = c1;
            if (num_flops) *num_flops += 8*x->size;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            const double (* ydata)[2] = y->data.complex_double;
            double c0 = 0, c1 = 0;
            for (int64_t k = 0; k < x->size; k++) {
                c0 += xdata[k][0]*ydata[idx[k]][0] + xdata[k][1]*ydata[idx[k]][1];
                c1 += xdata[k][0]*ydata[idx[k]][1] - xdata[k][1]*ydata[idx[k]][0];
            }
            (*dot)[0] = c0; (*dot)[1] = c1;
            if (num_flops) *num_flops += 8*x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else {
        (*dot)[1] = 0;
        return mtxvector_base_usddot(xpacked, y, &(*dot)[0], num_flops);
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_ussaxpy()’ performs a sparse vector update,
 * multiplying a sparse vector ‘x’ in packed form by a scalar ‘alpha’
 * and adding the result to a vector ‘y’. That is, ‘y = alpha*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. Repeated indices in the packed vector are not allowed,
 * otherwise the result is undefined.
 */
int mtxvector_base_ussaxpy(
    struct mtxvector_base * y,
    float alpha,
    const struct mtxvector_packed * xpacked,
    int64_t * num_flops)
{
    if (xpacked->x.type != mtxvector_base) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
    const struct mtxvector_base * x = &xpacked->x.storage.base;
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (xpacked->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (xpacked->num_nonzeros != x->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    const int64_t * idx = xpacked->idx;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            float * ydata = y->data.real_single;
            for (int64_t k = 0; k < x->size; k++)
                ydata[idx[k]] += alpha*xdata[k];
            if (num_flops) *num_flops += 2*x->size;
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            double * ydata = y->data.real_double;
            for (int64_t k = 0; k < x->size; k++)
                ydata[idx[k]] += alpha*xdata[k];
            if (num_flops) *num_flops += 2*x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            float (* ydata)[2] = y->data.complex_single;
            for (int64_t k = 0; k < x->size; k++) {
                ydata[idx[k]][0] += alpha*xdata[k][0];
                ydata[idx[k]][1] += alpha*xdata[k][1];
            }
            if (num_flops) *num_flops += 4*x->size;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            double (* ydata)[2] = y->data.complex_double;
            for (int64_t k = 0; k < x->size; k++) {
                ydata[idx[k]][0] += alpha*xdata[k][0];
                ydata[idx[k]][1] += alpha*xdata[k][1];
            }
            if (num_flops) *num_flops += 4*x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            const int32_t * xdata = x->data.integer_single;
            int32_t * ydata = y->data.integer_single;
            for (int64_t k = 0; k < x->size; k++)
                ydata[idx[k]] += alpha*xdata[k];
            if (num_flops) *num_flops += 2*x->size;
        } else if (x->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            int64_t * ydata = y->data.integer_double;
            for (int64_t k = 0; k < x->size; k++)
                ydata[idx[k]] += alpha*xdata[k];
            if (num_flops) *num_flops += 2*x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_usdaxpy()’ performs a sparse vector update,
 * multiplying a sparse vector ‘x’ in packed form by a scalar ‘alpha’
 * and adding the result to a vector ‘y’. That is, ‘y = alpha*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. Repeated indices in the packed vector are not allowed,
 * otherwise the result is undefined.
 */
int mtxvector_base_usdaxpy(
    struct mtxvector_base * y,
    double alpha,
    const struct mtxvector_packed * xpacked,
    int64_t * num_flops)
{
    if (xpacked->x.type != mtxvector_base) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
    const struct mtxvector_base * x = &xpacked->x.storage.base;
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (xpacked->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (xpacked->num_nonzeros != x->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    const int64_t * idx = xpacked->idx;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            float * ydata = y->data.real_single;
            for (int64_t k = 0; k < x->size; k++)
                ydata[idx[k]] += alpha*xdata[k];
            if (num_flops) *num_flops += 2*x->size;
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            double * ydata = y->data.real_double;
            for (int64_t k = 0; k < x->size; k++)
                ydata[idx[k]] += alpha*xdata[k];
            if (num_flops) *num_flops += 2*x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            float (* ydata)[2] = y->data.complex_single;
            for (int64_t k = 0; k < x->size; k++) {
                ydata[idx[k]][0] += alpha*xdata[k][0];
                ydata[idx[k]][1] += alpha*xdata[k][1];
            }
            if (num_flops) *num_flops += 4*x->size;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            double (* ydata)[2] = y->data.complex_double;
            for (int64_t k = 0; k < x->size; k++) {
                ydata[idx[k]][0] += alpha*xdata[k][0];
                ydata[idx[k]][1] += alpha*xdata[k][1];
            }
            if (num_flops) *num_flops += 4*x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            const int32_t * xdata = x->data.integer_single;
            int32_t * ydata = y->data.integer_single;
            for (int64_t k = 0; k < x->size; k++)
                ydata[idx[k]] += alpha*xdata[k];
            if (num_flops) *num_flops += 2*x->size;
        } else if (x->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            int64_t * ydata = y->data.integer_double;
            for (int64_t k = 0; k < x->size; k++)
                ydata[idx[k]] += alpha*xdata[k];
            if (num_flops) *num_flops += 2*x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_uscaxpy()’ performs a sparse vector update,
 * multiplying a sparse vector ‘x’ in packed form by a scalar ‘alpha’
 * and adding the result to a vector ‘y’. That is, ‘y = alpha*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. Repeated indices in the packed vector are not allowed,
 * otherwise the result is undefined.
 */
int mtxvector_base_uscaxpy(
    struct mtxvector_base * y,
    float alpha[2],
    const struct mtxvector_packed * xpacked,
    int64_t * num_flops)
{
    if (xpacked->x.type != mtxvector_base) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
    const struct mtxvector_base * x = &xpacked->x.storage.base;
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (xpacked->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (xpacked->num_nonzeros != x->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    const int64_t * idx = xpacked->idx;
    if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            float (* ydata)[2] = y->data.complex_single;
            for (int64_t k = 0; k < x->size; k++) {
                ydata[idx[k]][0] += alpha[0]*xdata[k][0] - alpha[1]*xdata[k][1];
                ydata[idx[k]][1] += alpha[0]*xdata[k][1] + alpha[1]*xdata[k][0];
            }
            if (num_flops) *num_flops += 8*x->size;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            double (* ydata)[2] = y->data.complex_double;
            for (int64_t k = 0; k < x->size; k++) {
                ydata[idx[k]][0] += alpha[0]*xdata[k][0] - alpha[1]*xdata[k][1];
                ydata[idx[k]][1] += alpha[0]*xdata[k][1] + alpha[1]*xdata[k][0];
            }
            if (num_flops) *num_flops += 8*x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_uszaxpy()’ performs a sparse vector update,
 * multiplying a sparse vector ‘x’ in packed form by a scalar ‘alpha’
 * and adding the result to a vector ‘y’. That is, ‘y = alpha*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. Repeated indices in the packed vector are not allowed,
 * otherwise the result is undefined.
 */
int mtxvector_base_uszaxpy(
    struct mtxvector_base * y,
    double alpha[2],
    const struct mtxvector_packed * xpacked,
    int64_t * num_flops)
{
    if (xpacked->x.type != mtxvector_base) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
    const struct mtxvector_base * x = &xpacked->x.storage.base;
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (xpacked->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (xpacked->num_nonzeros != x->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    const int64_t * idx = xpacked->idx;
    if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            float (* ydata)[2] = y->data.complex_single;
            for (int64_t k = 0; k < x->size; k++) {
                ydata[idx[k]][0] += alpha[0]*xdata[k][0] - alpha[1]*xdata[k][1];
                ydata[idx[k]][1] += alpha[0]*xdata[k][1] + alpha[1]*xdata[k][0];
            }
            if (num_flops) *num_flops += 8*x->size;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            double (* ydata)[2] = y->data.complex_double;
            for (int64_t k = 0; k < x->size; k++) {
                ydata[idx[k]][0] += alpha[0]*xdata[k][0] - alpha[1]*xdata[k][1];
                ydata[idx[k]][1] += alpha[0]*xdata[k][1] + alpha[1]*xdata[k][0];
            }
            if (num_flops) *num_flops += 8*x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_usga()’ performs a gather operation from a vector
 * ‘y’ into a sparse vector ‘x’ in packed form. Repeated indices in
 * the packed vector are allowed.
 */
int mtxvector_base_usga(
    struct mtxvector_packed * xpacked,
    const struct mtxvector_base * y)
{
    if (xpacked->x.type != mtxvector_base) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
    struct mtxvector_base * x = &xpacked->x.storage.base;
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (xpacked->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (xpacked->num_nonzeros != x->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    const int64_t * idx = xpacked->idx;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            float * xdata = x->data.real_single;
            const float * ydata = y->data.real_single;
            for (int64_t k = 0; k < x->size; k++)
                xdata[k] = ydata[idx[k]];
        } else if (x->precision == mtx_double) {
            double * xdata = x->data.real_double;
            const double * ydata = y->data.real_double;
            for (int64_t k = 0; k < x->size; k++)
                xdata[k] = ydata[idx[k]];
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            float (* xdata)[2] = x->data.complex_single;
            const float (* ydata)[2] = y->data.complex_single;
            for (int64_t k = 0; k < x->size; k++) {
                xdata[k][0] = ydata[idx[k]][0];
                xdata[k][1] = ydata[idx[k]][1];
            }
        } else if (x->precision == mtx_double) {
            double (* xdata)[2] = x->data.complex_double;
            const double (* ydata)[2] = y->data.complex_double;
            for (int64_t k = 0; k < x->size; k++) {
                xdata[k][0] = ydata[idx[k]][0];
                xdata[k][1] = ydata[idx[k]][1];
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            int32_t * xdata = x->data.integer_single;
            const int32_t * ydata = y->data.integer_single;
            for (int64_t k = 0; k < x->size; k++)
                xdata[k] = ydata[idx[k]];
        } else if (x->precision == mtx_double) {
            int64_t * xdata = x->data.integer_double;
            const int64_t * ydata = y->data.integer_double;
            for (int64_t k = 0; k < x->size; k++)
                xdata[k] = ydata[idx[k]];
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_usgz()’ performs a gather operation from a vector
 * ‘y’ into a sparse vector ‘x’ in packed form, while zeroing the
 * values of the source vector ‘y’ that were copied to ‘x’. Repeated
 * indices in the packed vector are allowed.
 */
int mtxvector_base_usgz(
    struct mtxvector_packed * xpacked,
    struct mtxvector_base * y)
{
    if (xpacked->x.type != mtxvector_base) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
    struct mtxvector_base * x = &xpacked->x.storage.base;
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (xpacked->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (xpacked->num_nonzeros != x->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    const int64_t * idx = xpacked->idx;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            float * xdata = x->data.real_single;
            float * ydata = y->data.real_single;
            for (int64_t k = 0; k < x->size; k++) {
                xdata[k] = ydata[idx[k]];
                ydata[idx[k]] = 0;
            }
        } else if (x->precision == mtx_double) {
            double * xdata = x->data.real_double;
            double * ydata = y->data.real_double;
            for (int64_t k = 0; k < x->size; k++) {
                xdata[k] = ydata[idx[k]];
                ydata[idx[k]] = 0;
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            float (* xdata)[2] = x->data.complex_single;
            float (* ydata)[2] = y->data.complex_single;
            for (int64_t k = 0; k < x->size; k++) {
                xdata[k][0] = ydata[idx[k]][0];
                xdata[k][1] = ydata[idx[k]][1];
                ydata[idx[k]][0] = 0;
                ydata[idx[k]][1] = 0;
            }
        } else if (x->precision == mtx_double) {
            double (* xdata)[2] = x->data.complex_double;
            double (* ydata)[2] = y->data.complex_double;
            for (int64_t k = 0; k < x->size; k++) {
                xdata[k][0] = ydata[idx[k]][0];
                xdata[k][1] = ydata[idx[k]][1];
                ydata[idx[k]][0] = 0;
                ydata[idx[k]][1] = 0;
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            int32_t * xdata = x->data.integer_single;
            int32_t * ydata = y->data.integer_single;
            for (int64_t k = 0; k < x->size; k++) {
                xdata[k] = ydata[idx[k]];
                ydata[idx[k]] = 0;
            }
        } else if (x->precision == mtx_double) {
            int64_t * xdata = x->data.integer_double;
            int64_t * ydata = y->data.integer_double;
            for (int64_t k = 0; k < x->size; k++) {
                xdata[k] = ydata[idx[k]];
                ydata[idx[k]] = 0;
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_ussc()’ performs a scatter operation to a vector
 * ‘y’ from a sparse vector ‘x’ in packed form. Repeated indices in
 * the packed vector are not allowed, otherwise the result is
 * undefined.
 */
int mtxvector_base_ussc(
    struct mtxvector_base * y,
    const struct mtxvector_packed * xpacked)
{
    if (xpacked->x.type != mtxvector_base) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
    const struct mtxvector_base * x = &xpacked->x.storage.base;
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (xpacked->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (xpacked->num_nonzeros != x->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    const int64_t * idx = xpacked->idx;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            float * ydata = y->data.real_single;
            for (int64_t k = 0; k < x->size; k++)
                ydata[idx[k]] = xdata[k];
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            double * ydata = y->data.real_double;
            for (int64_t k = 0; k < x->size; k++)
                ydata[idx[k]] = xdata[k];
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            float (* ydata)[2] = y->data.complex_single;
            for (int64_t k = 0; k < x->size; k++) {
                ydata[idx[k]][0] = xdata[k][0];
                ydata[idx[k]][1] = xdata[k][1];
            }
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            double (* ydata)[2] = y->data.complex_double;
            for (int64_t k = 0; k < x->size; k++) {
                ydata[idx[k]][0] = xdata[k][0];
                ydata[idx[k]][1] = xdata[k][1];
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            const int32_t * xdata = x->data.integer_single;
            int32_t * ydata = y->data.integer_single;
            for (int64_t k = 0; k < x->size; k++)
                ydata[idx[k]] = xdata[k];
        } else if (x->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            int64_t * ydata = y->data.integer_double;
            for (int64_t k = 0; k < x->size; k++)
                ydata[idx[k]] = xdata[k];
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/*
 * Level 1 BLAS-like extensions
 */

/**
 * ‘mtxvector_base_usscga()’ performs a combined scatter-gather
 * operation from a sparse vector ‘x’ in packed form into another
 * sparse vector ‘z’ in packed form. Repeated indices in the packed
 * vector ‘x’ are not allowed, otherwise the result is undefined. They
 * are, however, allowed in the packed vector ‘z’.
 */
int mtxvector_base_usscga(
    struct mtxvector_packed * zpacked,
    const struct mtxvector_packed * xpacked)
{
    if (xpacked->x.type != mtxvector_base) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
    const struct mtxvector_base * x = &xpacked->x.storage.base;
    if (zpacked->x.type != mtxvector_base) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
    const struct mtxvector_base * z = &zpacked->x.storage.base;
    if (x->field != z->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != z->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (xpacked->size != zpacked->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    struct mtxvector_base y;
    int err = mtxvector_base_alloc(&y, x->field, x->precision, xpacked->size);
    if (err) return err;
    err = mtxvector_base_setzero(&y);
    if (err) { mtxvector_base_free(&y); return err; }
    err = mtxvector_base_ussc(&y, xpacked);
    if (err) { mtxvector_base_free(&y); return err; }
    err = mtxvector_base_usga(zpacked, &y);
    if (err) { mtxvector_base_free(&y); return err; }
    mtxvector_base_free(&y);
    return MTX_SUCCESS;
}

/*
 * MPI functions
 */

#ifdef LIBMTX_HAVE_MPI
/**
 * ‘mtxvector_base_send()’ sends a vector to another MPI process.
 *
 * This is analogous to ‘MPI_Send()’ and requires the receiving
 * process to perform a matching call to ‘mtxvector_base_recv()’.
 */
int mtxvector_base_send(
    const struct mtxvector_base * x,
    int64_t offset,
    int count,
    int recipient,
    int tag,
    MPI_Comm comm,
    int * mpierrcode)
{
    int err;
    if (offset + count > x->size) return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            err = MPI_Send(
                &x->data.real_single[offset], count, MPI_FLOAT,
                recipient, tag, comm);
            if (mpierrcode) *mpierrcode = err;
            if (err) return MTX_ERR_MPI;
        } else if (x->precision == mtx_double) {
            err = MPI_Send(
                &x->data.real_double[offset], count, MPI_DOUBLE,
                recipient, tag, comm);
            if (mpierrcode) *mpierrcode = err;
            if (err) return MTX_ERR_MPI;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            err = MPI_Send(
                &x->data.complex_single[offset], 2*count, MPI_FLOAT,
                recipient, tag, comm);
            if (mpierrcode) *mpierrcode = err;
            if (err) return MTX_ERR_MPI;
        } else if (x->precision == mtx_double) {
            err = MPI_Send(
                &x->data.complex_double[offset], 2*count, MPI_DOUBLE,
                recipient, tag, comm);
            if (mpierrcode) *mpierrcode = err;
            if (err) return MTX_ERR_MPI;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            err = MPI_Send(
                &x->data.integer_single[offset], count, MPI_INT32_T,
                recipient, tag, comm);
            if (mpierrcode) *mpierrcode = err;
            if (err) return MTX_ERR_MPI;
        } else if (x->precision == mtx_double) {
            err = MPI_Send(
                &x->data.integer_double[offset], count, MPI_INT64_T,
                recipient, tag, comm);
            if (mpierrcode) *mpierrcode = err;
            if (err) return MTX_ERR_MPI;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_recv()’ receives a vector from another MPI process.
 *
 * This is analogous to ‘MPI_Recv()’ and requires the sending process
 * to perform a matching call to ‘mtxvector_base_send()’.
 */
int mtxvector_base_recv(
    struct mtxvector_base * x,
    int64_t offset,
    int count,
    int sender,
    int tag,
    MPI_Comm comm,
    MPI_Status * status,
    int * mpierrcode)
{
    int err;
    if (offset + count > x->size) return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            err = MPI_Recv(
                &x->data.real_single[offset], count, MPI_FLOAT,
                sender, tag, comm, status);
            if (mpierrcode) *mpierrcode = err;
            if (err) return MTX_ERR_MPI;
        } else if (x->precision == mtx_double) {
            err = MPI_Recv(
                &x->data.real_double[offset], count, MPI_DOUBLE,
                sender, tag, comm, status);
            if (mpierrcode) *mpierrcode = err;
            if (err) return MTX_ERR_MPI;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            err = MPI_Recv(
                &x->data.complex_single[offset], 2*count, MPI_FLOAT,
                sender, tag, comm, status);
            if (mpierrcode) *mpierrcode = err;
            if (err) return MTX_ERR_MPI;
        } else if (x->precision == mtx_double) {
            err = MPI_Recv(
                &x->data.complex_double[offset], 2*count, MPI_DOUBLE,
                sender, tag, comm, status);
            if (mpierrcode) *mpierrcode = err;
            if (err) return MTX_ERR_MPI;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            err = MPI_Recv(
                &x->data.integer_single[offset], count, MPI_INT32_T,
                sender, tag, comm, status);
            if (mpierrcode) *mpierrcode = err;
            if (err) return MTX_ERR_MPI;
        } else if (x->precision == mtx_double) {
            err = MPI_Recv(
                &x->data.integer_double[offset], count, MPI_INT64_T,
                sender, tag, comm, status);
            if (mpierrcode) *mpierrcode = err;
            if (err) return MTX_ERR_MPI;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_irecv()’ performs a non-blocking receive of a
 * vector from another MPI process.
 *
 * This is analogous to ‘MPI_Irecv()’ and requires the sending process
 * to perform a matching call to ‘mtxvector_base_send()’.
 */
int mtxvector_base_irecv(
    struct mtxvector_base * x,
    int64_t offset,
    int count,
    int sender,
    int tag,
    MPI_Comm comm,
    MPI_Request * request,
    int * mpierrcode)
{
    int err;
    if (offset + count > x->size) return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            err = MPI_Irecv(
                &x->data.real_single[offset], count, MPI_FLOAT,
                sender, tag, comm, request);
            if (mpierrcode) *mpierrcode = err;
            if (err) return MTX_ERR_MPI;
        } else if (x->precision == mtx_double) {
            err = MPI_Irecv(
                &x->data.real_double[offset], count, MPI_DOUBLE,
                sender, tag, comm, request);
            if (mpierrcode) *mpierrcode = err;
            if (err) return MTX_ERR_MPI;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            err = MPI_Irecv(
                &x->data.complex_single[offset], 2*count, MPI_FLOAT,
                sender, tag, comm, request);
            if (mpierrcode) *mpierrcode = err;
            if (err) return MTX_ERR_MPI;
        } else if (x->precision == mtx_double) {
            err = MPI_Irecv(
                &x->data.complex_double[offset], 2*count, MPI_DOUBLE,
                sender, tag, comm, request);
            if (mpierrcode) *mpierrcode = err;
            if (err) return MTX_ERR_MPI;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            err = MPI_Irecv(
                &x->data.integer_single[offset], count, MPI_INT32_T,
                sender, tag, comm, request);
            if (mpierrcode) *mpierrcode = err;
            if (err) return MTX_ERR_MPI;
        } else if (x->precision == mtx_double) {
            err = MPI_Irecv(
                &x->data.integer_double[offset], count, MPI_INT64_T,
                sender, tag, comm, request);
            if (mpierrcode) *mpierrcode = err;
            if (err) return MTX_ERR_MPI;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}
#endif
