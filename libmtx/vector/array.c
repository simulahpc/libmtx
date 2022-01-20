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
 * Last modified: 2021-08-19
 *
 * Vectors in array format.
 */

#include <libmtx/vector/array.h>

#include <libmtx/error.h>
#include <libmtx/mtx/header.h>
#include <libmtx/mtx/mtx.h>
#include <libmtx/precision.h>

#include <errno.h>

#include <stdlib.h>
#include <string.h>

/**
 * `mtx_alloc_vector_array()` allocates a dense vector in array
 * format.
 */
int mtx_alloc_vector_array(
    struct mtx * mtx,
    enum mtx_field field,
    enum mtxprecision precision,
    int num_comment_lines,
    const char ** comment_lines,
    int num_rows)
{
    int err;
    mtx->object = mtx_vector;
    mtx->format = mtx_array;
    mtx->field = field;
    mtx->symmetry = mtx_general;

    mtx->num_comment_lines = 0;
    mtx->comment_lines = NULL;
    err = mtx_set_comment_lines(
        mtx, num_comment_lines, comment_lines);
    if (err)
        return err;

    mtx->num_rows = num_rows;
    mtx->num_columns = -1;
    mtx->num_nonzeros = -1;

    /* Allocate storage for vector data. */
    err = mtx_vector_array_data_alloc(
        &mtx->storage.vector_array,
        field, precision, mtx->num_rows);
    if (err) {
        for (int i = 0; i < num_comment_lines; i++)
            free(mtx->comment_lines[i]);
        free(mtx->comment_lines);
        return err;
    }
    return MTX_SUCCESS;
}

/*
 * Array vector allocation and initialisation.
 */

/**
 * `mtx_init_vector_array_real_single()' creates a vector with real,
 * single-precision floating point coefficients.
 */
int mtx_init_vector_array_real_single(
    struct mtx * mtx,
    int num_comment_lines,
    const char ** comment_lines,
    int64_t size,
    const float * data)
{
    struct mtx_vector_array_data * mtxdata =
        &mtx->storage.vector_array;
    int err = mtx_vector_array_data_init_real_single(
        mtxdata, size, data);
    if (err)
        return err;

    mtx->object = mtx_vector;
    mtx->format = mtx_array;
    mtx->field = mtx_real;
    mtx->symmetry = mtx_general;
    mtx->num_comment_lines = 0;
    mtx->comment_lines = NULL;
    err = mtx_set_comment_lines(mtx, num_comment_lines, comment_lines);
    if (err) {
        mtx_vector_array_data_free(mtxdata);
        return err;
    }

    mtx->num_rows = size;
    mtx->num_columns = -1;
    mtx->num_nonzeros = -1;
    return MTX_SUCCESS;
}

/**
 * `mtx_init_vector_array_real_double()' creates a vector with real,
 * double-precision floating point coefficients.
 */
int mtx_init_vector_array_real_double(
    struct mtx * mtx,
    int num_comment_lines,
    const char ** comment_lines,
    int64_t size,
    const double * data)
{
    struct mtx_vector_array_data * mtxdata =
        &mtx->storage.vector_array;
    int err = mtx_vector_array_data_init_real_double(
        mtxdata, size, data);
    if (err)
        return err;

    mtx->object = mtx_vector;
    mtx->format = mtx_array;
    mtx->field = mtx_real;
    mtx->symmetry = mtx_general;
    mtx->num_comment_lines = 0;
    mtx->comment_lines = NULL;
    err = mtx_set_comment_lines(mtx, num_comment_lines, comment_lines);
    if (err) {
        mtx_vector_array_data_free(mtxdata);
        return err;
    }

    mtx->num_rows = size;
    mtx->num_columns = -1;
    mtx->num_nonzeros = -1;
    return MTX_SUCCESS;
}

/**
 * `mtx_init_vector_array_complex_single()' creates a vector with
 * complex, single-precision floating point coefficients.
 */
int mtx_init_vector_array_complex_single(
    struct mtx * mtx,
    int num_comment_lines,
    const char ** comment_lines,
    int64_t size,
    const float (* data)[2])
{
    struct mtx_vector_array_data * mtxdata =
        &mtx->storage.vector_array;
    int err = mtx_vector_array_data_init_complex_single(
        mtxdata, size, data);
    if (err)
        return err;

    mtx->object = mtx_vector;
    mtx->format = mtx_array;
    mtx->field = mtx_complex;
    mtx->symmetry = mtx_general;
    mtx->num_comment_lines = 0;
    mtx->comment_lines = NULL;
    err = mtx_set_comment_lines(mtx, num_comment_lines, comment_lines);
    if (err) {
        mtx_vector_array_data_free(mtxdata);
        return err;
    }

    mtx->num_rows = size;
    mtx->num_columns = -1;
    mtx->num_nonzeros = -1;
    return MTX_SUCCESS;
}

/**
 * `mtx_init_vector_array_integer_single()` creates a vector with
 * single precision, integer coefficients.
 */
int mtx_init_vector_array_integer_single(
    struct mtx * mtx,
    int num_comment_lines,
    const char ** comment_lines,
    int64_t size,
    const int32_t * data)
{
    struct mtx_vector_array_data * mtxdata =
        &mtx->storage.vector_array;
    int err = mtx_vector_array_data_init_integer_single(
        mtxdata, size, data);
    if (err)
        return err;

    mtx->object = mtx_vector;
    mtx->format = mtx_array;
    mtx->field = mtx_integer;
    mtx->symmetry = mtx_general;
    mtx->num_comment_lines = 0;
    mtx->comment_lines = NULL;
    err = mtx_set_comment_lines(mtx, num_comment_lines, comment_lines);
    if (err) {
        mtx_vector_array_data_free(mtxdata);
        return err;
    }

    mtx->num_rows = size;
    mtx->num_columns = -1;
    mtx->num_nonzeros = -1;
    return MTX_SUCCESS;
}

/**
 * `mtx_vector_array_set_zero()' zeroes a vector in array format.
 */
int mtx_vector_array_set_zero(
    struct mtx * mtx)
{
    if (mtx->object != mtx_vector)
        return MTX_ERR_INVALID_MTX_OBJECT;
    if (mtx->format != mtx_array)
        return MTX_ERR_INVALID_MTX_FORMAT;
    struct mtx_vector_array_data * mtxdata =
        &mtx->storage.vector_array;
    return mtx_vector_array_data_set_zero(mtxdata);
}

/**
 * `mtx_vector_array_set_constant_real_single()' sets every value of a
 * vector equal to a constant, single precision floating point number.
 */
int mtx_vector_array_set_constant_real_single(
    struct mtx * mtx,
    float a)
{
    if (mtx->object != mtx_vector)
        return MTX_ERR_INVALID_MTX_OBJECT;
    if (mtx->format != mtx_array)
        return MTX_ERR_INVALID_MTX_FORMAT;
    struct mtx_vector_array_data * mtxdata =
        &mtx->storage.vector_array;
    return mtx_vector_array_data_set_constant_real_single(mtxdata, a);
}

/**
 * `mtx_vector_array_set_constant_real_double()' sets every value of a
 * vector equal to a constant, double precision floating point number.
 */
int mtx_vector_array_set_constant_real_double(
    struct mtx * mtx,
    double a)
{
    if (mtx->object != mtx_vector)
        return MTX_ERR_INVALID_MTX_OBJECT;
    if (mtx->format != mtx_array)
        return MTX_ERR_INVALID_MTX_FORMAT;
    struct mtx_vector_array_data * mtxdata =
        &mtx->storage.vector_array;
    return mtx_vector_array_data_set_constant_real_double(mtxdata, a);
}

/**
 * `mtx_vector_array_set_constant_complex_single()' sets every value
 * of a vector equal to a constant, single precision floating point
 * complex number.
 */
int mtx_vector_array_set_constant_complex_single(
    struct mtx * mtx,
    float a[2])
{
    if (mtx->object != mtx_vector)
        return MTX_ERR_INVALID_MTX_OBJECT;
    if (mtx->format != mtx_array)
        return MTX_ERR_INVALID_MTX_FORMAT;
    struct mtx_vector_array_data * mtxdata =
        &mtx->storage.vector_array;
    return mtx_vector_array_data_set_constant_complex_single(mtxdata, a);
}

/**
 * `mtx_vector_array_set_constant_integer_single()' sets every value
 * of a vector equal to a constant, single precision integer.
 */
int mtx_vector_array_set_constant_integer_single(
    struct mtx * mtx,
    int32_t a)
{
    if (mtx->object != mtx_vector)
        return MTX_ERR_INVALID_MTX_OBJECT;
    if (mtx->format != mtx_array)
        return MTX_ERR_INVALID_MTX_FORMAT;
    struct mtx_vector_array_data * mtxdata =
        &mtx->storage.vector_array;
    return mtx_vector_array_data_set_constant_integer_single(mtxdata, a);
}
