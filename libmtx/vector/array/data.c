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

#include <libmtx/vector/array/data.h>

#include <libmtx/error.h>
#include <libmtx/mtx/header.h>
#include <libmtx/mtx/precision.h>

#include <stdint.h>
#include <stdlib.h>

/**
 * `mtx_vector_array_data_free()' frees resources associated with a
 * Matrix Market object.
 */
void mtx_vector_array_data_free(
    struct mtx_vector_array_data * mtxdata)
{
    if (mtxdata->field == mtx_real) {
        if (mtxdata->precision == mtx_single) {
            free(mtxdata->data.real_single);
        } else if (mtxdata->precision == mtx_double) {
            free(mtxdata->data.real_double);
        }
    } else if (mtxdata->field == mtx_complex) {
        if (mtxdata->precision == mtx_single) {
            free(mtxdata->data.complex_single);
        } else if (mtxdata->precision == mtx_double) {
            free(mtxdata->data.complex_double);
        }
    } else if (mtxdata->field == mtx_integer) {
        if (mtxdata->precision == mtx_single) {
            free(mtxdata->data.integer_single);
        } else if (mtxdata->precision == mtx_double) {
            free(mtxdata->data.integer_double);
        }
    }
}

/**
 * `mtx_vector_array_data_alloc()' allocates data for a vector in
 * array format.
 */
int mtx_vector_array_data_alloc(
    struct mtx_vector_array_data * mtxdata,
    enum mtx_field field,
    enum mtx_precision precision,
    int64_t size)
{
    if (field == mtx_real) {
        if (precision == mtx_single) {
            mtxdata->data.real_single = malloc(size * sizeof(float));
            if (!mtxdata->data.real_single)
                return MTX_ERR_ERRNO;
        } else if (precision == mtx_double) {
            mtxdata->data.real_double = malloc(size * sizeof(double));
            if (!mtxdata->data.real_double)
                return MTX_ERR_ERRNO;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (field == mtx_complex) {
        if (precision == mtx_single) {
            mtxdata->data.complex_single = malloc(size * sizeof(float[2]));
            if (!mtxdata->data.complex_single)
                return MTX_ERR_ERRNO;
        } else if (precision == mtx_double) {
            mtxdata->data.complex_double = malloc(size * sizeof(double[2]));
            if (!mtxdata->data.complex_double)
                return MTX_ERR_ERRNO;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (field == mtx_integer) {
        if (precision == mtx_single) {
            mtxdata->data.integer_single = malloc(size * sizeof(int32_t));
            if (!mtxdata->data.integer_single)
                return MTX_ERR_ERRNO;
        } else if (precision == mtx_double) {
            mtxdata->data.integer_double = malloc(size * sizeof(int64_t));
            if (!mtxdata->data.integer_double)
                return MTX_ERR_ERRNO;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else {
        return MTX_ERR_INVALID_MTX_FIELD;
    }

    mtxdata->field = field;
    mtxdata->precision = precision;
    mtxdata->num_rows = size;
    mtxdata->num_columns = -1;
    mtxdata->size = size;
    return MTX_SUCCESS;
}

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
    const float * data)
{
    int err = mtx_vector_array_data_alloc(
        mtxdata, mtx_real, mtx_single, size);
    if (err)
        return err;
    for (int64_t i = 0; i < size; i++)
        mtxdata->data.real_single[i] = data[i];
    return MTX_SUCCESS;
}

/**
 * `mtx_vector_array_data_init_real_double()' creates data for a
 * vector with real, double-precision floating point coefficients.
 */
int mtx_vector_array_data_init_real_double(
    struct mtx_vector_array_data * mtxdata,
    int64_t size,
    const double * data)
{
    int err = mtx_vector_array_data_alloc(
        mtxdata, mtx_real, mtx_double, size);
    if (err)
        return err;
    for (int64_t i = 0; i < size; i++)
        mtxdata->data.real_double[i] = data[i];
    return MTX_SUCCESS;
}

/**
 * `mtx_vector_array_data_init_complex_single()' creates data for a
 * vector with complex, single-precision floating point coefficients.
 */
int mtx_vector_array_data_init_complex_single(
    struct mtx_vector_array_data * mtxdata,
    int64_t size,
    const float (* data)[2])
{
    int err = mtx_vector_array_data_alloc(
        mtxdata, mtx_complex, mtx_single, size);
    if (err)
        return err;
    for (int64_t i = 0; i < size; i++) {
        mtxdata->data.complex_single[i][0] = data[i][0];
        mtxdata->data.complex_single[i][1] = data[i][1];
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_vector_array_data_init_complex_double()' creates data for a
 * vector with complex, double-precision floating point coefficients.
 */
int mtx_vector_array_data_init_complex_double(
    struct mtx_vector_array_data * mtxdata,
    int64_t size,
    const double (* data)[2])
{
    int err = mtx_vector_array_data_alloc(
        mtxdata, mtx_complex, mtx_double, size);
    if (err)
        return err;
    for (int64_t i = 0; i < size; i++) {
        mtxdata->data.complex_double[i][0] = data[i][0];
        mtxdata->data.complex_double[i][1] = data[i][1];
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_vector_array_data_init_integer_single()' creates data for a
 * vector with integer, single-precision coefficients.
 */
int mtx_vector_array_data_init_integer_single(
    struct mtx_vector_array_data * mtxdata,
    int64_t size,
    const int32_t * data)
{
    int err = mtx_vector_array_data_alloc(
        mtxdata, mtx_integer, mtx_single, size);
    if (err)
        return err;
    for (int64_t i = 0; i < size; i++)
        mtxdata->data.integer_single[i] = data[i];
    return MTX_SUCCESS;
}

/**
 * `mtx_vector_array_data_init_integer_double()' creates data for a
 * vector with integer, double-precision coefficients.
 */
int mtx_vector_array_data_init_integer_double(
    struct mtx_vector_array_data * mtxdata,
    int64_t size,
    const int64_t * data)
{
    int err = mtx_vector_array_data_alloc(
        mtxdata, mtx_integer, mtx_double, size);
    if (err)
        return err;
    for (int64_t i = 0; i < size; i++)
        mtxdata->data.integer_double[i] = data[i];
    return MTX_SUCCESS;
}

/**
 * `mtx_vector_array_data_copy_alloc()' allocates a copy of a vector
 * without copying the vector values.
 */
int mtx_vector_array_data_copy_alloc(
    struct mtx_vector_array_data * dst,
    const struct mtx_vector_array_data * src)
{
    return mtx_vector_array_data_alloc(
        dst, src->field, src->precision, src->size);
}

/**
 * `mtx_vector_array_data_copy_init()' creates a copy of a vector and
 * also copies vector values.
 */
int mtx_vector_array_data_copy_init(
    struct mtx_vector_array_data * dst,
    const struct mtx_vector_array_data * src)
{
    if (src->field == mtx_real) {
        if (src->precision == mtx_single) {
            return mtx_vector_array_data_init_real_single(
                dst, src->size, src->data.real_single);
        } else if (src->precision == mtx_double) {
            return mtx_vector_array_data_init_real_double(
                dst, src->size, src->data.real_double);
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (src->field == mtx_complex) {
        if (src->precision == mtx_single) {
            return mtx_vector_array_data_init_complex_single(
                dst, src->size, src->data.complex_single);
        } else if (src->precision == mtx_double) {
            return mtx_vector_array_data_init_complex_double(
                dst, src->size, src->data.complex_double);
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (src->field == mtx_integer) {
        if (src->precision == mtx_single) {
            return mtx_vector_array_data_init_integer_single(
                dst, src->size, src->data.integer_single);
        } else if (src->precision == mtx_double) {
            return mtx_vector_array_data_init_integer_double(
                dst, src->size, src->data.integer_double);
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else {
        return MTX_ERR_INVALID_MTX_FIELD;
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_vector_array_data_set_zero()' zeroes a matrix or vector.
 */
int mtx_vector_array_data_set_zero(
    struct mtx_vector_array_data * mtxdata)
{
    if (mtxdata->field == mtx_real) {
        if (mtxdata->precision == mtx_single) {
            for (int64_t k = 0; k < mtxdata->size; k++)
                mtxdata->data.real_single[k] = 0;
        } else if (mtxdata->precision == mtx_double) {
            for (int64_t k = 0; k < mtxdata->size; k++)
                mtxdata->data.real_double[k] = 0;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (mtxdata->field == mtx_complex) {
        if (mtxdata->precision == mtx_single) {
            for (int64_t k = 0; k < mtxdata->size; k++) {
                mtxdata->data.complex_single[k][0] = 0;
                mtxdata->data.complex_single[k][1] = 0;
            }
        } else if (mtxdata->precision == mtx_double) {
            for (int64_t k = 0; k < mtxdata->size; k++) {
                mtxdata->data.complex_double[k][0] = 0;
                mtxdata->data.complex_double[k][1] = 0;
            }
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (mtxdata->field == mtx_integer) {
        if (mtxdata->precision == mtx_single) {
            for (int64_t k = 0; k < mtxdata->size; k++)
                mtxdata->data.integer_single[k] = 0;
        } else if (mtxdata->precision == mtx_double) {
            for (int64_t k = 0; k < mtxdata->size; k++)
                mtxdata->data.integer_double[k] = 0;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else {
        return MTX_ERR_INVALID_MTX_FIELD;
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_vector_array_data_set_constant_real_single()' sets every value
 * of a vector equal to a constant, single precision floating point
 * number.
 */
int mtx_vector_array_data_set_constant_real_single(
    struct mtx_vector_array_data * mtxdata,
    float a)
{
    if (mtxdata->field != mtx_real)
        return MTX_ERR_INVALID_MTX_FIELD;
    if (mtxdata->precision != mtx_single)
        return MTX_ERR_INVALID_PRECISION;
    for (int64_t k = 0; k < mtxdata->size; k++)
        mtxdata->data.real_single[k] = a;
    return MTX_SUCCESS;
}

/**
 * `mtx_vector_array_data_set_constant_real_double()' sets every value
 * of a vector equal to a constant, double precision floating point
 * number.
 */
int mtx_vector_array_data_set_constant_real_double(
    struct mtx_vector_array_data * mtxdata,
    double a)
{
    if (mtxdata->field != mtx_real)
        return MTX_ERR_INVALID_MTX_FIELD;
    if (mtxdata->precision != mtx_double)
        return MTX_ERR_INVALID_PRECISION;
    for (int64_t k = 0; k < mtxdata->size; k++)
        mtxdata->data.real_double[k] = a;
    return MTX_SUCCESS;
}

/**
 * `mtx_vector_array_data_set_constant_complex_single()' sets every
 * value of a vector equal to a constant, single precision floating
 * point complex number.
 */
int mtx_vector_array_data_set_constant_complex_single(
    struct mtx_vector_array_data * mtxdata,
    float a[2])
{
    if (mtxdata->field != mtx_complex)
        return MTX_ERR_INVALID_MTX_FIELD;
    if (mtxdata->precision != mtx_single)
        return MTX_ERR_INVALID_PRECISION;
    for (int64_t k = 0; k < mtxdata->size; k++) {
        mtxdata->data.complex_single[k][0] = a[0];
        mtxdata->data.complex_single[k][1] = a[1];
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_vector_array_data_set_constant_complex_double()' sets every
 * value of a vector equal to a constant, double precision floating
 * point complex number.
 */
int mtx_vector_array_data_set_constant_complex_double(
    struct mtx_vector_array_data * mtxdata,
    double a[2])
{
    if (mtxdata->field != mtx_complex)
        return MTX_ERR_INVALID_MTX_FIELD;
    if (mtxdata->precision != mtx_double)
        return MTX_ERR_INVALID_PRECISION;
    for (int64_t k = 0; k < mtxdata->size; k++) {
        mtxdata->data.complex_double[k][0] = a[0];
        mtxdata->data.complex_double[k][1] = a[1];
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_vector_array_data_set_constant_integer_single()' sets every
 * value of a vector equal to a constant integer.
 */
int mtx_vector_array_data_set_constant_integer_single(
    struct mtx_vector_array_data * mtxdata,
    int32_t a)
{
    if (mtxdata->field != mtx_integer)
        return MTX_ERR_INVALID_MTX_FIELD;
    if (mtxdata->precision != mtx_single)
        return MTX_ERR_INVALID_PRECISION;
    for (int64_t k = 0; k < mtxdata->size; k++)
        mtxdata->data.integer_single[k] = a;
    return MTX_SUCCESS;
}

/**
 * `mtx_vector_array_data_set_constant_integer_double()' sets every
 * value of a vector equal to a constant integer.
 */
int mtx_vector_array_data_set_constant_integer_double(
    struct mtx_vector_array_data * mtxdata,
    int64_t a)
{
    if (mtxdata->field != mtx_integer)
        return MTX_ERR_INVALID_MTX_FIELD;
    if (mtxdata->precision != mtx_double)
        return MTX_ERR_INVALID_PRECISION;
    for (int64_t k = 0; k < mtxdata->size; k++)
        mtxdata->data.integer_double[k] = a;
    return MTX_SUCCESS;
}
