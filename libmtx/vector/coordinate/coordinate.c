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
 * Sparse vectors in Matrix Market coordinate format.
 */

#include <libmtx/vector/coordinate/coordinate.h>

#include <libmtx/error.h>
#include <libmtx/mtx.h>

#include <errno.h>

#include <stdlib.h>

/**
 * `mtx_alloc_vector_coordinate_real()` allocates a sparse vector with
 * real, single-precision floating point coefficients.
 */
int mtx_alloc_vector_coordinate_real(
    struct mtx * mtx,
    int num_comment_lines,
    const char ** comment_lines,
    int num_rows,
    int size)
{
    mtx->object = mtx_vector;
    mtx->format = mtx_coordinate;
    mtx->field = mtx_real;
    mtx->symmetry = mtx_general;
    mtx->triangle = mtx_nontriangular;
    mtx->sorting = mtx_unsorted;
    mtx->ordering = mtx_unordered;
    mtx->assembly = mtx_unassembled;

    mtx->num_comment_lines = 0;
    mtx->comment_lines = NULL;
    int err = mtx_set_comment_lines(
        mtx, num_comment_lines, comment_lines);
    if (err)
        return err;

    mtx->num_rows = num_rows;
    mtx->num_columns = -1;
    mtx->num_nonzeros = size;
    mtx->size = size;
    mtx->nonzero_size = sizeof(struct mtx_vector_coordinate_real);
    mtx->data = malloc(size * mtx->nonzero_size);
    if (!mtx->data) {
        for (int i = 0; i < num_comment_lines; i++)
            free(mtx->comment_lines[i]);
        free(mtx->comment_lines);
        return MTX_ERR_ERRNO;
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_alloc_vector_coordinate_double()` allocates a sparse vector
 * with real, double-precision floating point coefficients.
 */
int mtx_alloc_vector_coordinate_double(
    struct mtx * mtx,
    int num_comment_lines,
    const char ** comment_lines,
    int num_rows,
    int size)
{
    mtx->object = mtx_vector;
    mtx->format = mtx_coordinate;
    mtx->field = mtx_double;
    mtx->symmetry = mtx_general;
    mtx->triangle = mtx_nontriangular;
    mtx->sorting = mtx_unsorted;
    mtx->ordering = mtx_unordered;
    mtx->assembly = mtx_unassembled;

    mtx->num_comment_lines = 0;
    mtx->comment_lines = NULL;
    int err = mtx_set_comment_lines(
        mtx, num_comment_lines, comment_lines);
    if (err)
        return err;

    mtx->num_rows = num_rows;
    mtx->num_columns = -1;
    mtx->num_nonzeros = size;
    mtx->size = size;
    mtx->nonzero_size = sizeof(struct mtx_vector_coordinate_double);
    mtx->data = malloc(size * mtx->nonzero_size);
    if (!mtx->data) {
        for (int i = 0; i < num_comment_lines; i++)
            free(mtx->comment_lines[i]);
        free(mtx->comment_lines);
        return MTX_ERR_ERRNO;
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_alloc_vector_coordinate_complex()` allocates a sparse vector
 * with complex, single-precision floating point coefficients.
 */
int mtx_alloc_vector_coordinate_complex(
    struct mtx * mtx,
    int num_comment_lines,
    const char ** comment_lines,
    int num_rows,
    int size)
{
    mtx->object = mtx_vector;
    mtx->format = mtx_coordinate;
    mtx->field = mtx_complex;
    mtx->symmetry = mtx_general;
    mtx->triangle = mtx_nontriangular;
    mtx->sorting = mtx_unsorted;
    mtx->ordering = mtx_unordered;
    mtx->assembly = mtx_unassembled;

    mtx->num_comment_lines = 0;
    mtx->comment_lines = NULL;
    int err = mtx_set_comment_lines(
        mtx, num_comment_lines, comment_lines);
    if (err)
        return err;

    mtx->num_rows = num_rows;
    mtx->num_columns = -1;
    mtx->num_nonzeros = size;
    mtx->size = size;
    mtx->nonzero_size = sizeof(struct mtx_vector_coordinate_complex);
    mtx->data = malloc(size * mtx->nonzero_size);
    if (!mtx->data) {
        for (int i = 0; i < num_comment_lines; i++)
            free(mtx->comment_lines[i]);
        free(mtx->comment_lines);
        return MTX_ERR_ERRNO;
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_alloc_vector_coordinate_integer()` allocates a sparse vector
 * with integer coefficients.
 */
int mtx_alloc_vector_coordinate_integer(
    struct mtx * mtx,
    int num_comment_lines,
    const char ** comment_lines,
    int num_rows,
    int size)
{
    mtx->object = mtx_vector;
    mtx->format = mtx_coordinate;
    mtx->field = mtx_integer;
    mtx->symmetry = mtx_general;
    mtx->triangle = mtx_nontriangular;
    mtx->sorting = mtx_unsorted;
    mtx->ordering = mtx_unordered;
    mtx->assembly = mtx_unassembled;

    mtx->num_comment_lines = 0;
    mtx->comment_lines = NULL;
    int err = mtx_set_comment_lines(
        mtx, num_comment_lines, comment_lines);
    if (err)
        return err;

    mtx->num_rows = num_rows;
    mtx->num_columns = -1;
    mtx->num_nonzeros = size;
    mtx->size = size;
    mtx->nonzero_size = sizeof(struct mtx_vector_coordinate_integer);
    mtx->data = malloc(size * mtx->nonzero_size);
    if (!mtx->data) {
        for (int i = 0; i < num_comment_lines; i++)
            free(mtx->comment_lines[i]);
        free(mtx->comment_lines);
        return MTX_ERR_ERRNO;
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_alloc_vector_coordinate_pattern()` allocates a sparse vector
 * with boolean coefficients.
 */
int mtx_alloc_vector_coordinate_pattern(
    struct mtx * mtx,
    int num_comment_lines,
    const char ** comment_lines,
    int num_rows,
    int size)
{
    mtx->object = mtx_vector;
    mtx->format = mtx_coordinate;
    mtx->field = mtx_pattern;
    mtx->symmetry = mtx_general;
    mtx->triangle = mtx_nontriangular;
    mtx->sorting = mtx_unsorted;
    mtx->ordering = mtx_unordered;
    mtx->assembly = mtx_unassembled;

    mtx->num_comment_lines = 0;
    mtx->comment_lines = NULL;
    int err = mtx_set_comment_lines(
        mtx, num_comment_lines, comment_lines);
    if (err)
        return err;

    mtx->num_rows = num_rows;
    mtx->num_columns = -1;
    mtx->num_nonzeros = size;
    mtx->size = size;
    mtx->nonzero_size = sizeof(struct mtx_vector_coordinate_pattern);
    mtx->data = malloc(size * mtx->nonzero_size);
    if (!mtx->data) {
        for (int i = 0; i < num_comment_lines; i++)
            free(mtx->comment_lines[i]);
        free(mtx->comment_lines);
        return MTX_ERR_ERRNO;
    }
    return MTX_SUCCESS;
}

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
    const struct mtx_vector_coordinate_real * data)
{
    int err = mtx_alloc_vector_coordinate_real(
        mtx, num_comment_lines, comment_lines, num_rows, size);
    if (err)
        return err;

    mtx->sorting = sorting;
    mtx->ordering = ordering;
    mtx->assembly = assembly;
    struct mtx_vector_coordinate_real * mtxdata =
        (struct mtx_vector_coordinate_real *) mtx->data;
    for (int i = 0; i < size; i++)
        mtxdata[i] = data[i];
    return MTX_SUCCESS;
}

/**
 * `mtx_init_vector_coordinate_double()` creates a sparse vector
 * with real, double-precision floating point coefficients.
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
    const struct mtx_vector_coordinate_double * data)
{
    int err = mtx_alloc_vector_coordinate_double(
        mtx, num_comment_lines, comment_lines, num_rows, size);
    if (err)
        return err;

    mtx->sorting = sorting;
    mtx->ordering = ordering;
    mtx->assembly = assembly;
    struct mtx_vector_coordinate_double * mtxdata =
        (struct mtx_vector_coordinate_double *) mtx->data;
    for (int i = 0; i < size; i++)
        mtxdata[i] = data[i];
    return MTX_SUCCESS;
}

/**
 * `mtx_init_vector_coordinate_complex()` creates a sparse vector
 * with complex, single-precision floating point coefficients.
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
    const struct mtx_vector_coordinate_complex * data)
{
    int err = mtx_alloc_vector_coordinate_complex(
        mtx, num_comment_lines, comment_lines, num_rows, size);
    if (err)
        return err;

    mtx->sorting = sorting;
    mtx->ordering = ordering;
    mtx->assembly = assembly;
    struct mtx_vector_coordinate_complex * mtxdata =
        (struct mtx_vector_coordinate_complex *) mtx->data;
    for (int i = 0; i < size; i++)
        mtxdata[i] = data[i];
    return MTX_SUCCESS;
}

/**
 * `mtx_init_vector_coordinate_integer()` creates a sparse vector
 * with integer coefficients.
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
    const struct mtx_vector_coordinate_integer * data)
{
    int err = mtx_alloc_vector_coordinate_integer(
        mtx, num_comment_lines, comment_lines, num_rows, size);
    if (err)
        return err;

    mtx->sorting = sorting;
    mtx->ordering = ordering;
    mtx->assembly = assembly;
    struct mtx_vector_coordinate_integer * mtxdata =
        (struct mtx_vector_coordinate_integer *) mtx->data;
    for (int i = 0; i < size; i++)
        mtxdata[i] = data[i];
    return MTX_SUCCESS;
}

/**
 * `mtx_init_vector_coordinate_pattern()` creates a sparse vector
 * with boolean coefficients.
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
    const struct mtx_vector_coordinate_pattern * data)
{
    int err = mtx_alloc_vector_coordinate_pattern(
        mtx, num_comment_lines, comment_lines, num_rows, size);
    if (err)
        return err;

    mtx->sorting = sorting;
    mtx->ordering = ordering;
    mtx->assembly = assembly;
    struct mtx_vector_coordinate_pattern * mtxdata =
        (struct mtx_vector_coordinate_pattern *) mtx->data;
    for (int i = 0; i < size; i++)
        mtxdata[i] = data[i];
    return MTX_SUCCESS;
}

/*
 * Other sparse vector functions.
 */

/**
 * `mtx_vector_coordinate_set_zero()' zeroes a vector in coordinate
 * format.
 */
int mtx_vector_coordinate_set_zero(
    struct mtx * mtx)
{
    if (mtx->object != mtx_vector)
        return MTX_ERR_INVALID_MTX_OBJECT;
    if (mtx->format != mtx_coordinate)
        return MTX_ERR_INVALID_MTX_FORMAT;

    if (mtx->field == mtx_real) {
        struct mtx_vector_coordinate_real * data =
            (struct mtx_vector_coordinate_real *) mtx->data;
        for (int64_t k = 0; k < mtx->size; k++)
            data[k].a = 0;
    } else if (mtx->field == mtx_double) {
        struct mtx_vector_coordinate_double * data =
            (struct mtx_vector_coordinate_double *) mtx->data;
        for (int64_t k = 0; k < mtx->size; k++)
            data[k].a = 0;
    } else if (mtx->field == mtx_complex) {
        struct mtx_vector_coordinate_complex * data =
            (struct mtx_vector_coordinate_complex *) mtx->data;
        for (int64_t k = 0; k < mtx->size; k++) {
            data[k].a = 0;
            data[k].b = 0;
        }
    } else if (mtx->field == mtx_integer) {
        struct mtx_vector_coordinate_integer * data =
            (struct mtx_vector_coordinate_integer *) mtx->data;
        for (int64_t k = 0; k < mtx->size; k++)
            data[k].a = 0;
    } else if (mtx->field == mtx_pattern) {
        /* Since no values are stored, there is nothing to do here. */
    } else {
        return MTX_ERR_INVALID_MTX_FIELD;
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_vector_coordinate_set_constant_real()' sets every nonzero
 * value of a vector equal to a constant, single precision floating
 * point number.
 */
int mtx_vector_coordinate_set_constant_real(
    struct mtx * mtx,
    float a)
{
    if (mtx->object != mtx_vector)
        return MTX_ERR_INVALID_MTX_OBJECT;
    if (mtx->format != mtx_coordinate)
        return MTX_ERR_INVALID_MTX_FORMAT;
    if (mtx->field != mtx_real)
        return MTX_ERR_INVALID_MTX_FIELD;

    struct mtx_vector_coordinate_real * data =
        (struct mtx_vector_coordinate_real *) mtx->data;
    for (int64_t k = 0; k < mtx->size; k++)
        data[k].a = a;
    return MTX_SUCCESS;
}

/**
 * `mtx_vector_coordinate_set_constant_double()' sets every nonzero
 * value of a vector equal to a constant, double precision floating
 * point number.
 */
int mtx_vector_coordinate_set_constant_double(
    struct mtx * mtx,
    double a)
{
    if (mtx->object != mtx_vector)
        return MTX_ERR_INVALID_MTX_OBJECT;
    if (mtx->format != mtx_coordinate)
        return MTX_ERR_INVALID_MTX_FORMAT;
    if (mtx->field != mtx_double)
        return MTX_ERR_INVALID_MTX_FIELD;

    struct mtx_vector_coordinate_double * data =
        (struct mtx_vector_coordinate_double *) mtx->data;
    for (int64_t k = 0; k < mtx->size; k++)
        data[k].a = a;
    return MTX_SUCCESS;
}

/**
 * `mtx_vector_coordinate_set_constant_complex()' sets every nonzero
 * value of a vector equal to a constant, single precision floating
 * point complex number.
 */
int mtx_vector_coordinate_set_constant_complex(
    struct mtx * mtx,
    float a,
    float b)
{
    if (mtx->object != mtx_vector)
        return MTX_ERR_INVALID_MTX_OBJECT;
    if (mtx->format != mtx_coordinate)
        return MTX_ERR_INVALID_MTX_FORMAT;
    if (mtx->field != mtx_complex)
        return MTX_ERR_INVALID_MTX_FIELD;

    struct mtx_vector_coordinate_complex * data =
        (struct mtx_vector_coordinate_complex *) mtx->data;
    for (int64_t k = 0; k < mtx->size; k++) {
        data[k].a = a;
        data[k].b = b;
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_vector_coordinate_set_constant_integer()' sets every nonzero
 * value of a vector equal to a constant integer.
 */
int mtx_vector_coordinate_set_constant_integer(
    struct mtx * mtx,
    int a)
{
    if (mtx->object != mtx_vector)
        return MTX_ERR_INVALID_MTX_OBJECT;
    if (mtx->format != mtx_coordinate)
        return MTX_ERR_INVALID_MTX_FORMAT;
    if (mtx->field != mtx_integer)
        return MTX_ERR_INVALID_MTX_FIELD;

    struct mtx_vector_coordinate_integer * data =
        (struct mtx_vector_coordinate_integer *) mtx->data;
    for (int64_t k = 0; k < mtx->size; k++)
        data[k].a = a;
    return MTX_SUCCESS;
}
