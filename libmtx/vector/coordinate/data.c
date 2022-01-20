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
 * Last modified: 2021-08-16
 *
 * Data structures for vectors in coordinate format.
 */

#include <libmtx/vector/coordinate/data.h>

#include <libmtx/error.h>
#include <libmtx/mtx/assembly.h>
#include <libmtx/mtx/header.h>
#include <libmtx/precision.h>
#include <libmtx/mtx/sort.h>

#include <stdint.h>
#include <stdlib.h>

/**
 * `mtx_vector_coordinate_data_free()' frees resources associated with
 * a Matrix Market object.
 */
void mtx_vector_coordinate_data_free(
    struct mtx_vector_coordinate_data * mtxdata)
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
    } else if (mtxdata->field == mtx_pattern) {
        free(mtxdata->data.pattern);
    }
}

/**
 * `mtx_vector_coordinate_data_alloc()' allocates data for a vector in
 * coordinate format.
 */
int mtx_vector_coordinate_data_alloc(
    struct mtx_vector_coordinate_data * mtxdata,
    enum mtx_field field,
    enum mtxprecision precision,
    enum mtx_sorting sorting,
    enum mtx_assembly assembly,
    int num_rows,
    int num_columns,
    int64_t size)
{
    if (field == mtx_real) {
        if (precision == mtx_single) {
            mtxdata->data.real_single = malloc(
                size * sizeof(struct mtx_vector_coordinate_real_single));
            if (!mtxdata->data.real_single)
                return MTX_ERR_ERRNO;
        } else if (precision == mtx_double) {
            mtxdata->data.real_double = malloc(
                size * sizeof(struct mtx_vector_coordinate_real_double));
            if (!mtxdata->data.real_double)
                return MTX_ERR_ERRNO;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (field == mtx_complex) {
        if (precision == mtx_single) {
            mtxdata->data.complex_single = malloc(
                size * sizeof(struct mtx_vector_coordinate_complex_single));
            if (!mtxdata->data.complex_single)
                return MTX_ERR_ERRNO;
        } else if (precision == mtx_double) {
            mtxdata->data.complex_double = malloc(
                size * sizeof(struct mtx_vector_coordinate_complex_double));
            if (!mtxdata->data.complex_double)
                return MTX_ERR_ERRNO;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (field == mtx_integer) {
        if (precision == mtx_single) {
            mtxdata->data.integer_single = malloc(
                size * sizeof(struct mtx_vector_coordinate_integer_single));
            if (!mtxdata->data.integer_single)
                return MTX_ERR_ERRNO;
        } else if (precision == mtx_double) {
            mtxdata->data.integer_double = malloc(
                size * sizeof(struct mtx_vector_coordinate_integer_double));
            if (!mtxdata->data.integer_double)
                return MTX_ERR_ERRNO;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (field == mtx_pattern) {
        mtxdata->data.pattern = malloc(
            size * sizeof(struct mtx_vector_coordinate_pattern));
        if (!mtxdata->data.pattern)
            return MTX_ERR_ERRNO;
    } else {
        return MTX_ERR_INVALID_MTX_FIELD;
    }

    mtxdata->field = field;
    mtxdata->precision = precision;
    mtxdata->sorting = sorting;
    mtxdata->assembly = assembly;
    mtxdata->num_rows = num_rows;
    mtxdata->num_columns = num_columns;    
    mtxdata->size = size;
    return MTX_SUCCESS;
}

/*
 * Coordinate vector allocation and initialisation.
 */

/**
 * `mtx_vector_coordinate_data_init_real_single()' creates data for a
 * vector with real, single-precision floating point coefficients.
 */
int mtx_vector_coordinate_data_init_real_single(
    struct mtx_vector_coordinate_data * mtxdata,
    enum mtx_sorting sorting,
    enum mtx_assembly assembly,
    int num_rows,
    int num_columns,
    int64_t size,
    const struct mtx_vector_coordinate_real_single * data)
{
    int err = mtx_vector_coordinate_data_alloc(
        mtxdata, mtx_real, mtx_single, sorting, assembly,
        num_rows, num_columns, size);
    if (err)
        return err;
    for (int64_t i = 0; i < size; i++)
        mtxdata->data.real_single[i] = data[i];
    return MTX_SUCCESS;
}

/**
 * `mtx_vector_coordinate_data_init_real_double()' creates data for a
 * vector with real, double-precision floating point coefficients.
 */
int mtx_vector_coordinate_data_init_real_double(
    struct mtx_vector_coordinate_data * mtxdata,
    enum mtx_sorting sorting,
    enum mtx_assembly assembly,
    int num_rows,
    int num_columns,
    int64_t size,
    const struct mtx_vector_coordinate_real_double * data)
{
    int err = mtx_vector_coordinate_data_alloc(
        mtxdata, mtx_real, mtx_double, sorting, assembly,
        num_rows, num_columns, size);
    if (err)
        return err;
    for (int64_t i = 0; i < size; i++)
        mtxdata->data.real_double[i] = data[i];
    return MTX_SUCCESS;
}

/**
 * `mtx_vector_coordinate_data_init_complex_single()' creates data for
 * a vector with complex, single-precision floating point
 * coefficients.
 */
int mtx_vector_coordinate_data_init_complex_single(
    struct mtx_vector_coordinate_data * mtxdata,
    enum mtx_sorting sorting,
    enum mtx_assembly assembly,
    int num_rows,
    int num_columns,
    int64_t size,
    const struct mtx_vector_coordinate_complex_single * data)
{
    int err = mtx_vector_coordinate_data_alloc(
        mtxdata, mtx_complex, mtx_single, sorting, assembly,
        num_rows, num_columns, size);
    if (err)
        return err;
    for (int64_t i = 0; i < size; i++)
        mtxdata->data.complex_single[i] = data[i];
    return MTX_SUCCESS;
}

/**
 * `mtx_vector_coordinate_data_init_complex_double()' creates data for
 * a vector with complex, double-precision floating point
 * coefficients.
 */
int mtx_vector_coordinate_data_init_complex_double(
    struct mtx_vector_coordinate_data * mtxdata,
    enum mtx_sorting sorting,
    enum mtx_assembly assembly,
    int num_rows,
    int num_columns,
    int64_t size,
    const struct mtx_vector_coordinate_complex_double * data)
{
    int err = mtx_vector_coordinate_data_alloc(
        mtxdata, mtx_complex, mtx_double, sorting, assembly,
        num_rows, num_columns, size);
    if (err)
        return err;
    for (int64_t i = 0; i < size; i++)
        mtxdata->data.complex_double[i] = data[i];
    return MTX_SUCCESS;
}

/**
 * `mtx_vector_coordinate_data_init_integer_single()' creates data for
 * a vector with integer, single-precision coefficients.
 */
int mtx_vector_coordinate_data_init_integer_single(
    struct mtx_vector_coordinate_data * mtxdata,
    enum mtx_sorting sorting,
    enum mtx_assembly assembly,
    int num_rows,
    int num_columns,
    int64_t size,
    const struct mtx_vector_coordinate_integer_single * data)
{
    int err = mtx_vector_coordinate_data_alloc(
        mtxdata, mtx_integer, mtx_single, sorting, assembly,
        num_rows, num_columns, size);
    if (err)
        return err;
    for (int64_t i = 0; i < size; i++)
        mtxdata->data.integer_single[i] = data[i];
    return MTX_SUCCESS;
}

/**
 * `mtx_vector_coordinate_data_init_integer_double()' creates data for
 * a vector with integer, double-precision coefficients.
 */
int mtx_vector_coordinate_data_init_integer_double(
    struct mtx_vector_coordinate_data * mtxdata,
    enum mtx_sorting sorting,
    enum mtx_assembly assembly,
    int num_rows,
    int num_columns,
    int64_t size,
    const struct mtx_vector_coordinate_integer_double * data)
{
    int err = mtx_vector_coordinate_data_alloc(
        mtxdata, mtx_integer, mtx_double, sorting, assembly,
        num_rows, num_columns, size);
    if (err)
        return err;
    for (int64_t i = 0; i < size; i++)
        mtxdata->data.integer_double[i] = data[i];
    return MTX_SUCCESS;
}

/**
 * `mtx_vector_coordinate_data_init_pattern()' creates data for a
 * vector with boolean (pattern) coefficients.
 */
int mtx_vector_coordinate_data_init_pattern(
    struct mtx_vector_coordinate_data * mtxdata,
    enum mtx_sorting sorting,
    enum mtx_assembly assembly,
    int num_rows,
    int num_columns,
    int64_t size,
    const struct mtx_vector_coordinate_pattern * data)
{
    int err = mtx_vector_coordinate_data_alloc(
        mtxdata, mtx_pattern, mtx_single, sorting, assembly,
        num_rows, num_columns, size);
    if (err)
        return err;
    for (int64_t i = 0; i < size; i++)
        mtxdata->data.pattern[i] = data[i];
    return MTX_SUCCESS;
}

/**
 * `mtx_vector_coordinate_data_copy_alloc()' allocates a copy of a vector
 * without copying the vector values.
 */
int mtx_vector_coordinate_data_copy_alloc(
    struct mtx_vector_coordinate_data * dst,
    const struct mtx_vector_coordinate_data * src)
{
    return mtx_vector_coordinate_data_alloc(
        dst, src->field, src->precision,
        src->sorting, src->assembly,
        src->num_rows, src->num_columns, src->size);
}

/**
 * `mtx_vector_coordinate_data_copy_init()' creates a copy of a vector
 * and also copies vector values.
 */
int mtx_vector_coordinate_data_copy_init(
    struct mtx_vector_coordinate_data * dst,
    const struct mtx_vector_coordinate_data * src)
{
    if (src->field == mtx_real) {
        if (src->precision == mtx_single) {
            return mtx_vector_coordinate_data_init_real_single(
                dst, src->sorting, src->assembly,
                src->num_rows, src->num_columns, src->size,
                src->data.real_single);
        } else if (src->precision == mtx_double) {
            return mtx_vector_coordinate_data_init_real_double(
                dst, src->sorting, src->assembly,
                src->num_rows, src->num_columns, src->size,
                src->data.real_double);
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (src->field == mtx_complex) {
        if (src->precision == mtx_single) {
            return mtx_vector_coordinate_data_init_complex_single(
                dst, src->sorting, src->assembly,
                src->num_rows, src->num_columns, src->size,
                src->data.complex_single);
        } else if (src->precision == mtx_double) {
            return mtx_vector_coordinate_data_init_complex_double(
                dst, src->sorting, src->assembly,
                src->num_rows, src->num_columns, src->size,
                src->data.complex_double);
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (src->field == mtx_integer) {
        if (src->precision == mtx_single) {
            return mtx_vector_coordinate_data_init_integer_single(
                dst, src->sorting, src->assembly,
                src->num_rows, src->num_columns, src->size,
                src->data.integer_single);
        } else if (src->precision == mtx_double) {
            return mtx_vector_coordinate_data_init_integer_double(
                dst, src->sorting, src->assembly,
                src->num_rows, src->num_columns, src->size,
                src->data.integer_double);
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (src->field == mtx_pattern) {
        return mtx_vector_coordinate_data_init_pattern(
            dst, src->sorting, src->assembly,
            src->num_rows, src->num_columns, src->size,
            src->data.pattern);
    } else {
        return MTX_ERR_INVALID_MTX_FIELD;
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_vector_coordinate_data_set_zero()' zeroes a vector.
 */
int mtx_vector_coordinate_data_set_zero(
    struct mtx_vector_coordinate_data * mtxdata)
{
    if (mtxdata->field == mtx_real) {
        if (mtxdata->precision == mtx_single) {
            for (int64_t k = 0; k < mtxdata->size; k++)
                mtxdata->data.real_single[k].a = 0;
        } else if (mtxdata->precision == mtx_double) {
            for (int64_t k = 0; k < mtxdata->size; k++)
                mtxdata->data.real_double[k].a = 0;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (mtxdata->field == mtx_complex) {
        if (mtxdata->precision == mtx_single) {
            for (int64_t k = 0; k < mtxdata->size; k++) {
                mtxdata->data.complex_single[k].a[0] = 0;
                mtxdata->data.complex_single[k].a[1] = 0;
            }
        } else if (mtxdata->precision == mtx_double) {
            for (int64_t k = 0; k < mtxdata->size; k++) {
                mtxdata->data.complex_double[k].a[0] = 0;
                mtxdata->data.complex_double[k].a[1] = 0;
            }
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (mtxdata->field == mtx_integer) {
        if (mtxdata->precision == mtx_single) {
            for (int64_t k = 0; k < mtxdata->size; k++)
                mtxdata->data.integer_single[k].a = 0;
        } else if (mtxdata->precision == mtx_double) {
            for (int64_t k = 0; k < mtxdata->size; k++)
                mtxdata->data.integer_double[k].a = 0;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else {
        return MTX_ERR_INVALID_MTX_FIELD;
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_vector_coordinate_data_set_constant_real_single()' sets every
 * (nonzero) value of a vector equal to a constant, single precision
 * floating point number.
 */
int mtx_vector_coordinate_data_set_constant_real_single(
    struct mtx_vector_coordinate_data * mtxdata,
    float a)
{
    if (mtxdata->field != mtx_real)
        return MTX_ERR_INVALID_MTX_FIELD;
    if (mtxdata->precision != mtx_single)
        return MTX_ERR_INVALID_PRECISION;
    for (int64_t k = 0; k < mtxdata->size; k++)
        mtxdata->data.real_single[k].a = a;
    return MTX_SUCCESS;
}

/**
 * `mtx_vector_coordinate_data_set_constant_real_double()' sets every
 * (nonzero) value of a vector equal to a constant, double precision
 * floating point number.
 */
int mtx_vector_coordinate_data_set_constant_real_double(
    struct mtx_vector_coordinate_data * mtxdata,
    double a)
{
    if (mtxdata->field != mtx_real)
        return MTX_ERR_INVALID_MTX_FIELD;
    if (mtxdata->precision != mtx_double)
        return MTX_ERR_INVALID_PRECISION;
    for (int64_t k = 0; k < mtxdata->size; k++)
        mtxdata->data.real_double[k].a = a;
    return MTX_SUCCESS;
}

/**
 * `mtx_vector_coordinate_data_set_constant_complex_single()' sets
 * every (nonzero) value of a vector equal to a constant, single
 * precision floating point complex number.
 */
int mtx_vector_coordinate_data_set_constant_complex_single(
    struct mtx_vector_coordinate_data * mtxdata,
    float a[2])
{
    if (mtxdata->field != mtx_complex)
        return MTX_ERR_INVALID_MTX_FIELD;
    if (mtxdata->precision != mtx_single)
        return MTX_ERR_INVALID_PRECISION;
    for (int64_t k = 0; k < mtxdata->size; k++) {
        mtxdata->data.complex_single[k].a[0] = a[0];
        mtxdata->data.complex_single[k].a[1] = a[1];
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_vector_coordinate_data_set_constant_complex_double()' sets
 * every (nonzero) value of a vector equal to a constant, double
 * precision floating point complex number.
 */
int mtx_vector_coordinate_data_set_constant_complex_double(
    struct mtx_vector_coordinate_data * mtxdata,
    double a[2])
{
    if (mtxdata->field != mtx_complex)
        return MTX_ERR_INVALID_MTX_FIELD;
    if (mtxdata->precision != mtx_double)
        return MTX_ERR_INVALID_PRECISION;
    for (int64_t k = 0; k < mtxdata->size; k++) {
        mtxdata->data.complex_double[k].a[0] = a[0];
        mtxdata->data.complex_double[k].a[1] = a[1];
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_vector_coordinate_data_set_constant_integer_single()' sets
 * every (nonzero) value of a vector equal to a constant integer.
 */
int mtx_vector_coordinate_data_set_constant_integer_single(
    struct mtx_vector_coordinate_data * mtxdata,
    int32_t a)
{
    if (mtxdata->field != mtx_integer)
        return MTX_ERR_INVALID_MTX_FIELD;
    if (mtxdata->precision != mtx_single)
        return MTX_ERR_INVALID_PRECISION;
    for (int64_t k = 0; k < mtxdata->size; k++)
        mtxdata->data.integer_single[k].a = a;
    return MTX_SUCCESS;
}

/**
 * `mtx_vector_coordinate_data_set_constant_integer_double()' sets
 * every (nonzero) value of a vector equal to a constant integer.
 */
int mtx_vector_coordinate_data_set_constant_integer_double(
    struct mtx_vector_coordinate_data * mtxdata,
    int64_t a)
{
    if (mtxdata->field != mtx_integer)
        return MTX_ERR_INVALID_MTX_FIELD;
    if (mtxdata->precision != mtx_double)
        return MTX_ERR_INVALID_PRECISION;
    for (int64_t k = 0; k < mtxdata->size; k++)
        mtxdata->data.integer_double[k].a = a;
    return MTX_SUCCESS;
}
