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
 * Dense vectors in Matrix Market format.
 */

#include <libmtx/error.h>
#include <libmtx/header.h>
#include <libmtx/mtx.h>
#include <libmtx/vector_array.h>

#include <errno.h>

#include <stdlib.h>
#include <string.h>

static int mtx_alloc_vector_array_field(
    struct mtx * mtx,
    enum mtx_field field,
    int num_comment_lines,
    const char ** comment_lines,
    int nonzero_size,
    int size)
{
    int err;
    mtx->object = mtx_vector;
    mtx->format = mtx_array;
    mtx->field = field;
    mtx->symmetry = mtx_general;
    mtx->triangle = mtx_nontriangular;
    mtx->sorting = mtx_row_major;
    mtx->ordering = mtx_unordered;
    mtx->assembly = mtx_assembled;

    mtx->num_comment_lines = 0;
    mtx->comment_lines = NULL;
    err = mtx_set_comment_lines(mtx, num_comment_lines, comment_lines);
    if (err)
        return err;

    mtx->num_rows = size;
    mtx->num_columns = -1;
    mtx->num_nonzeros = size;
    mtx->size = size;
    mtx->nonzero_size = nonzero_size;

    /* Allocate storage for vector data. */
    mtx->data = malloc(size * nonzero_size);
    if (!mtx->data) {
        for (int i = 0; i < num_comment_lines; i++)
            free(mtx->comment_lines[i]);
        free(mtx->comment_lines);
        return MTX_ERR_ERRNO;
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_alloc_vector_array_real()` allocates a vector with real,
 * single-precision floating point coefficients.
 */
int mtx_alloc_vector_array_real(
    struct mtx * mtx,
    int num_comment_lines,
    const char ** comment_lines,
    int size)
{
    return mtx_alloc_vector_array_field(
        mtx, mtx_real, num_comment_lines, comment_lines,
        sizeof(float), size);
}

/**
 * `mtx_init_vector_array_real()` creates a vector with real,
 * single-precision floating point coefficients.
 */
int mtx_init_vector_array_real(
    struct mtx * mtx,
    int num_comment_lines,
    const char ** comment_lines,
    int size,
    const float * data)
{
    int err = mtx_alloc_vector_array_real(
        mtx, num_comment_lines, comment_lines, size);
    if (err)
        return err;
    for (int i = 0; i < size; i++)
        ((float *) mtx->data)[i] = data[i];
    return MTX_SUCCESS;
}

/**
 * `mtx_alloc_vector_array_double()` creates a vector with real,
 * double-precision floating point coefficients.
 */
int mtx_alloc_vector_array_double(
    struct mtx * mtx,
    int num_comment_lines,
    const char ** comment_lines,
    int size)
{
    return mtx_alloc_vector_array_field(
        mtx, mtx_double, num_comment_lines, comment_lines,
        sizeof(double), size);
}

/**
 * `mtx_init_vector_array_double()` creates a vector with real,
 * double-precision floating point coefficients.
 */
int mtx_init_vector_array_double(
    struct mtx * mtx,
    int num_comment_lines,
    const char ** comment_lines,
    int size,
    const double * data)
{
    int err = mtx_alloc_vector_array_double(
        mtx, num_comment_lines, comment_lines, size);
    if (err)
        return err;
    for (int i = 0; i < size; i++)
        ((double *) mtx->data)[i] = data[i];
    return MTX_SUCCESS;
}

/**
 * `mtx_alloc_vector_array_complex()` allocates a vector with
 * complex, single-precision floating point coefficients.
 */
int mtx_alloc_vector_array_complex(
    struct mtx * mtx,
    int num_comment_lines,
    const char ** comment_lines,
    int size)
{
    return mtx_alloc_vector_array_field(
        mtx, mtx_complex, num_comment_lines, comment_lines,
        2*sizeof(float), size);
}

/**
 * `mtx_init_vector_array_complex()` creates a vector with complex,
 * single-precision floating point coefficients.
 */
int mtx_init_vector_array_complex(
    struct mtx * mtx,
    int num_comment_lines,
    const char ** comment_lines,
    int size,
    const float * data)
{
    int err = mtx_alloc_vector_array_complex(
        mtx, num_comment_lines, comment_lines, size);
    if (err)
        return err;
    for (int i = 0; i < size; i++) {
        ((float *) mtx->data)[2*i+0] = data[2*i+0];
        ((float *) mtx->data)[2*i+1] = data[2*i+1];
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_alloc_vector_array_integer()` allocates a vector with
 * integer coefficients.
 */
int mtx_alloc_vector_array_integer(
    struct mtx * mtx,
    int num_comment_lines,
    const char ** comment_lines,
    int size)
{
    return mtx_alloc_vector_array_field(
        mtx, mtx_integer, num_comment_lines, comment_lines,
        sizeof(int), size);
}

/**
 * `mtx_init_vector_array_integer()` creates a vector with integer
 * coefficients.
 */
int mtx_init_vector_array_integer(
    struct mtx * mtx,
    int num_comment_lines,
    const char ** comment_lines,
    int size,
    const int * data)
{
    int err = mtx_alloc_vector_array_integer(
        mtx, num_comment_lines, comment_lines, size);
    if (err)
        return err;
    for (int i = 0; i < size; i++)
        ((int *) mtx->data)[i] = data[i];
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

    if (mtx->field == mtx_real) {
        float * data = (float *) mtx->data;
        for (int64_t k = 0; k < mtx->size; k++)
            data[k] = 0;
    } else if (mtx->field == mtx_double) {
        double * data = (double *) mtx->data;
        for (int64_t k = 0; k < mtx->size; k++)
            data[k] = 0;
    } else if (mtx->field == mtx_complex) {
        float * data = (float *) mtx->data;
        for (int64_t k = 0; k < mtx->size; k++) {
            data[2*k+0] = 0;
            data[2*k+1] = 0;
        }
    } else if (mtx->field == mtx_integer) {
        int * data = (int *) mtx->data;
        for (int64_t k = 0; k < mtx->size; k++)
            data[k] = 0;
    } else {
        return MTX_ERR_INVALID_MTX_FIELD;
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_vector_array_set_constant_real()' sets every value of a vector
 * equal to a constant, single precision floating point number.
 */
int mtx_vector_array_set_constant_real(
    struct mtx * mtx,
    float a)
{
    if (mtx->object != mtx_vector)
        return MTX_ERR_INVALID_MTX_OBJECT;
    if (mtx->format != mtx_array)
        return MTX_ERR_INVALID_MTX_FORMAT;
    if (mtx->field != mtx_real)
        return MTX_ERR_INVALID_MTX_FIELD;

    float * data = (float *) mtx->data;
    for (int64_t k = 0; k < mtx->size; k++)
        data[k] = a;
    return MTX_SUCCESS;
}

/**
 * `mtx_vector_array_set_constant_double()' sets every value of a
 * vector equal to a constant, double precision floating point number.
 */
int mtx_vector_array_set_constant_double(
    struct mtx * mtx,
    double a)
{
    if (mtx->object != mtx_vector)
        return MTX_ERR_INVALID_MTX_OBJECT;
    if (mtx->format != mtx_array)
        return MTX_ERR_INVALID_MTX_FORMAT;
    if (mtx->field != mtx_double)
        return MTX_ERR_INVALID_MTX_FIELD;

    double * data = (double *) mtx->data;
    for (int64_t k = 0; k < mtx->size; k++)
        data[k] = a;
    return MTX_SUCCESS;
}

/**
 * `mtx_vector_array_set_constant_complex()' sets every value of a
 * vector equal to a constant, single precision floating point complex
 * number.
 */
int mtx_vector_array_set_constant_complex(
    struct mtx * mtx,
    float a,
    float b)
{
    if (mtx->object != mtx_vector)
        return MTX_ERR_INVALID_MTX_OBJECT;
    if (mtx->format != mtx_array)
        return MTX_ERR_INVALID_MTX_FORMAT;
    if (mtx->field != mtx_complex)
        return MTX_ERR_INVALID_MTX_FIELD;

    float * data = (float *) mtx->data;
    for (int64_t k = 0; k < mtx->size; k++) {
        data[2*k+0] = a;
        data[2*k+1] = b;
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_vector_array_set_constant_integer()' sets every value of a
 * vector equal to a constant integer.
 */
int mtx_vector_array_set_constant_integer(
    struct mtx * mtx,
    int a)
{
    if (mtx->object != mtx_vector)
        return MTX_ERR_INVALID_MTX_OBJECT;
    if (mtx->format != mtx_array)
        return MTX_ERR_INVALID_MTX_FORMAT;
    if (mtx->field != mtx_integer)
        return MTX_ERR_INVALID_MTX_FIELD;

    int * data = (int *) mtx->data;
    for (int64_t k = 0; k < mtx->size; k++)
        data[k] = a;
    return MTX_SUCCESS;
}
