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
 * Last modified: 2021-06-18
 *
 * Dense vectors in Matrix Market format.
 */

#include <matrixmarket/error.h>
#include <matrixmarket/header.h>
#include <matrixmarket/mtx.h>
#include <matrixmarket/vector_array.h>

#ifdef HAVE_BLAS
#include <cblas.h>
#endif

#include <errno.h>

#include <stdlib.h>
#include <string.h>

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
    mtx->object = mtx_vector;
    mtx->format = mtx_array;
    mtx->field = mtx_real;
    mtx->symmetry = mtx_general;
    mtx->num_comment_lines = num_comment_lines;

    /* Allocate storage for and copy comment lines. */
    mtx->comment_lines = malloc(num_comment_lines * sizeof(char *));
    if (!mtx->comment_lines)
        return MTX_ERR_ERRNO;
    for (int i = 0; i < num_comment_lines; i++)
        mtx->comment_lines[i] = strdup(comment_lines[i]);

    mtx->num_rows = size;
    mtx->num_columns = 1;
    mtx->num_nonzeros = size;
    mtx->size = size;

    /* Allocate storage for vector data. */
    mtx->data = malloc(size * sizeof(float));
    if (!mtx->data) {
        for (int i = 0; i < num_comment_lines; i++)
            free(mtx->comment_lines[i]);
        free(mtx->comment_lines);
        return MTX_ERR_ERRNO;
    }
    return MTX_SUCCESS;
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
 * `mtx_init_vector_array_real_zero()` creates a vector of real,
 * single-precision floating point coefficients by filling with zeros.
 */
int mtx_init_vector_array_real_zero(
    struct mtx * mtx,
    int num_comment_lines,
    const char ** comment_lines,
    int size)
{
    int err = mtx_alloc_vector_array_real(
        mtx, num_comment_lines, comment_lines, size);
    if (err)
        return err;
    for (int i = 0; i < size; i++)
        ((float *) mtx->data)[i] = 0.0f;
    return MTX_SUCCESS;
}

/**
 * `mtx_init_vector_array_real_ones()` creates a vector of real,
 * single-precision floating point coefficients by filling with ones.
 */
int mtx_init_vector_array_real_ones(
    struct mtx * mtx,
    int num_comment_lines,
    const char ** comment_lines,
    int size)
{
    int err = mtx_alloc_vector_array_real(
        mtx, num_comment_lines, comment_lines, size);
    if (err)
        return err;
    for (int i = 0; i < size; i++)
        ((float *) mtx->data)[i] = 1.0f;
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
    mtx->object = mtx_vector;
    mtx->format = mtx_array;
    mtx->field = mtx_double;
    mtx->symmetry = mtx_general;
    mtx->num_comment_lines = num_comment_lines;

    /* Allocate storage for and copy comment lines. */
    mtx->comment_lines = malloc(num_comment_lines * sizeof(char *));
    if (!mtx->comment_lines)
        return MTX_ERR_ERRNO;
    for (int i = 0; i < num_comment_lines; i++)
        mtx->comment_lines[i] = strdup(comment_lines[i]);

    mtx->num_rows = size;
    mtx->num_columns = 1;
    mtx->num_nonzeros = size;
    mtx->size = size;

    /* Allocate storage for vector data. */
    mtx->data = malloc(size * sizeof(double));
    if (!mtx->data) {
        for (int i = 0; i < num_comment_lines; i++)
            free(mtx->comment_lines[i]);
        free(mtx->comment_lines);
        return MTX_ERR_ERRNO;
    }
    return MTX_SUCCESS;
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
 * `mtx_init_vector_array_double_zero()` creates a vector of real,
 * double-precision floating point coefficients by filling with zeros.
 */
int mtx_init_vector_array_double_zero(
    struct mtx * mtx,
    int num_comment_lines,
    const char ** comment_lines,
    int size)
{
    int err = mtx_alloc_vector_array_double(
        mtx, num_comment_lines, comment_lines, size);
    if (err)
        return err;
    for (int i = 0; i < size; i++)
        ((double *) mtx->data)[i] = 0.0;
    return MTX_SUCCESS;
}

/**
 * `mtx_init_vector_array_double_ones()` creates a vector of real,
 * double-precision floating point coefficients by filling with ones.
 */
int mtx_init_vector_array_double_ones(
    struct mtx * mtx,
    int num_comment_lines,
    const char ** comment_lines,
    int size)
{
    int err = mtx_alloc_vector_array_double(
        mtx, num_comment_lines, comment_lines, size);
    if (err)
        return err;
    for (int i = 0; i < size; i++)
        ((double *) mtx->data)[i] = 1.0;
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
    mtx->object = mtx_vector;
    mtx->format = mtx_array;
    mtx->field = mtx_complex;
    mtx->symmetry = mtx_general;
    mtx->num_comment_lines = num_comment_lines;

    /* Allocate storage for and copy comment lines. */
    mtx->comment_lines = malloc(num_comment_lines * sizeof(char *));
    if (!mtx->comment_lines)
        return MTX_ERR_ERRNO;
    for (int i = 0; i < num_comment_lines; i++)
        mtx->comment_lines[i] = strdup(comment_lines[i]);

    mtx->num_rows = size;
    mtx->num_columns = 1;
    mtx->num_nonzeros = size;
    mtx->size = size;

    /* Allocate storage for vector data. */
    mtx->data = malloc(size * 2*sizeof(float));
    if (!mtx->data) {
        for (int i = 0; i < num_comment_lines; i++)
            free(mtx->comment_lines[i]);
        free(mtx->comment_lines);
        return MTX_ERR_ERRNO;
    }
    return MTX_SUCCESS;
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
 * `mtx_init_vector_array_complex_zero()` creates a vector of
 * complex, single-precision floating point coefficients by filling
 * with zeros.
 */
int mtx_init_vector_array_complex_zero(
    struct mtx * mtx,
    int num_comment_lines,
    const char ** comment_lines,
    int size)
{
    int err = mtx_alloc_vector_array_complex(
        mtx, num_comment_lines, comment_lines, size);
    if (err)
        return err;
    for (int i = 0; i < size; i++) {
        ((float *) mtx->data)[2*i+0] = 0.0f;
        ((float *) mtx->data)[2*i+1] = 0.0f;
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_init_vector_array_complex_ones()` creates a vector of
 * complex, single-precision floating point coefficients by filling
 * with ones.
 */
int mtx_init_vector_array_complex_ones(
    struct mtx * mtx,
    int num_comment_lines,
    const char ** comment_lines,
    int size)
{
    int err = mtx_alloc_vector_array_complex(
        mtx, num_comment_lines, comment_lines, size);
    if (err)
        return err;
    for (int i = 0; i < size; i++) {
        ((float *) mtx->data)[2*i+0] = 1.0f;
        ((float *) mtx->data)[2*i+1] = 0.0f;
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
    mtx->object = mtx_vector;
    mtx->format = mtx_array;
    mtx->field = mtx_integer;
    mtx->symmetry = mtx_general;
    mtx->num_comment_lines = num_comment_lines;

    /* Allocate storage for and copy comment lines. */
    mtx->comment_lines = malloc(num_comment_lines * sizeof(char *));
    if (!mtx->comment_lines)
        return MTX_ERR_ERRNO;
    for (int i = 0; i < num_comment_lines; i++)
        mtx->comment_lines[i] = strdup(comment_lines[i]);

    mtx->num_rows = size;
    mtx->num_columns = 1;
    mtx->num_nonzeros = size;
    mtx->size = size;

    /* Allocate storage for vector data. */
    mtx->data = malloc(size * sizeof(int));
    if (!mtx->data) {
        for (int i = 0; i < num_comment_lines; i++)
            free(mtx->comment_lines[i]);
        free(mtx->comment_lines);
        return MTX_ERR_ERRNO;
    }
    return MTX_SUCCESS;
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
 * `mtx_init_vector_array_integer_zero()` creates a vector of
 * integer, coefficients by filling with zeros.
 */
int mtx_init_vector_array_integer_zero(
    struct mtx * mtx,
    int num_comment_lines,
    const char ** comment_lines,
    int size)
{
    int err = mtx_alloc_vector_array_integer(
        mtx, num_comment_lines, comment_lines, size);
    if (err)
        return err;
    for (int i = 0; i < size; i++)
        ((int *) mtx->data)[i] = 0;
    return MTX_SUCCESS;
}

/**
 * `mtx_init_vector_array_integer_ones()` creates a vector of
 * integer, coefficients by filling with ones.
 */
int mtx_init_vector_array_integer_ones(
    struct mtx * mtx,
    int num_comment_lines,
    const char ** comment_lines,
    int size)
{
    int err = mtx_alloc_vector_array_integer(
        mtx, num_comment_lines, comment_lines, size);
    if (err)
        return err;
    for (int i = 0; i < size; i++)
        ((int *) mtx->data)[i] = 1;
    return MTX_SUCCESS;
}
