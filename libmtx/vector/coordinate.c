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
 * Last modified: 2021-08-19
 *
 * Sparse vectors in Matrix Market coordinate format.
 */

#include <libmtx/vector/coordinate.h>

#include <libmtx/mtx/assembly.h>
#include <libmtx/error.h>
#include <libmtx/mtx/mtx.h>
#include <libmtx/mtx/precision.h>
#include <libmtx/mtx/reorder.h>
#include <libmtx/mtx/sort.h>

#include <errno.h>

#include <stdlib.h>

/**
 * `mtx_alloc_vector_coordinate()` allocates a sparse vector in
 * coordinate format.
 */
int mtx_alloc_vector_coordinate(
    struct mtx * mtx,
    enum mtx_field field,
    enum mtx_precision precision,
    int num_comment_lines,
    const char ** comment_lines,
    int num_rows,
    int64_t num_nonzeros)
{
    mtx->object = mtx_vector;
    mtx->format = mtx_coordinate;
    mtx->field = field;
    mtx->symmetry = mtx_general;

    mtx->num_comment_lines = 0;
    mtx->comment_lines = NULL;
    int err = mtx_set_comment_lines(
        mtx, num_comment_lines, comment_lines);
    if (err)
        return err;

    mtx->num_rows = num_rows;
    mtx->num_columns = -1;
    mtx->num_nonzeros = num_nonzeros;

    err = mtx_vector_coordinate_data_alloc(
        &mtx->storage.vector_coordinate,
        field, precision, mtx_unsorted, mtx_unassembled,
        mtx->num_rows, mtx->num_columns, mtx->num_nonzeros);
    if (err) {
        for (int i = 0; i < num_comment_lines; i++)
            free(mtx->comment_lines[i]);
        free(mtx->comment_lines);
        return err;
    }
    return MTX_SUCCESS;
}

/*
 * Coordinate vector allocation and initialisation.
 */

/**
 * `mtx_init_vector_coordinate_real_single()' creates a coordinate
 * vector with real, single-precision floating point coefficients.
 */
int mtx_init_vector_coordinate_real_single(
    struct mtx * mtx,
    enum mtx_sorting sorting,
    enum mtx_assembly assembly,
    int num_comment_lines,
    const char ** comment_lines,
    int num_rows,
    int64_t size,
    const struct mtx_vector_coordinate_real_single * data)
{
    struct mtx_vector_coordinate_data * mtxdata =
        &mtx->storage.vector_coordinate;
    int err = mtx_vector_coordinate_data_init_real_single(
        mtxdata, sorting, assembly, num_rows, -1, size, data);
    if (err)
        return err;

    mtx->object = mtx_vector;
    mtx->format = mtx_coordinate;
    mtx->field = mtx_real;
    mtx->symmetry = mtx_general;
    mtx->num_comment_lines = 0;
    mtx->comment_lines = NULL;
    err = mtx_set_comment_lines(mtx, num_comment_lines, comment_lines);
    if (err) {
        mtx_vector_coordinate_data_free(mtxdata);
        return err;
    }

    mtx->num_rows = num_rows;
    mtx->num_columns = -1;
    mtx->num_nonzeros = size;
    return MTX_SUCCESS;
}

/**
 * `mtx_init_vector_coordinate_real_double()' creates a coordinate
 * vector with real, double-precision floating point coefficients.
 */
int mtx_init_vector_coordinate_real_double(
    struct mtx * mtx,
    enum mtx_sorting sorting,
    enum mtx_assembly assembly,
    int num_comment_lines,
    const char ** comment_lines,
    int num_rows,
    int64_t size,
    const struct mtx_vector_coordinate_real_double * data)
{
    struct mtx_vector_coordinate_data * mtxdata =
        &mtx->storage.vector_coordinate;
    int err = mtx_vector_coordinate_data_init_real_double(
        mtxdata, sorting, assembly, num_rows, -1, size, data);
    if (err)
        return err;

    mtx->object = mtx_vector;
    mtx->format = mtx_coordinate;
    mtx->field = mtx_real;
    mtx->symmetry = mtx_general;
    mtx->num_comment_lines = 0;
    mtx->comment_lines = NULL;
    err = mtx_set_comment_lines(mtx, num_comment_lines, comment_lines);
    if (err) {
        mtx_vector_coordinate_data_free(mtxdata);
        return err;
    }

    mtx->num_rows = num_rows;
    mtx->num_columns = -1;
    mtx->num_nonzeros = size;
    return MTX_SUCCESS;
}

/**
 * `mtx_init_vector_coordinate_complex_single()' creates a coordinate
 * vector with complex, single-precision floating point coefficients.
 */
int mtx_init_vector_coordinate_complex_single(
    struct mtx * mtx,
    enum mtx_sorting sorting,
    enum mtx_assembly assembly,
    int num_comment_lines,
    const char ** comment_lines,
    int num_rows,
    int64_t size,
    const struct mtx_vector_coordinate_complex_single * data)
{
    struct mtx_vector_coordinate_data * mtxdata =
        &mtx->storage.vector_coordinate;
    int err = mtx_vector_coordinate_data_init_complex_single(
        mtxdata, sorting, assembly, num_rows, -1, size, data);
    if (err)
        return err;

    mtx->object = mtx_vector;
    mtx->format = mtx_coordinate;
    mtx->field = mtx_complex;
    mtx->symmetry = mtx_general;
    mtx->num_comment_lines = 0;
    mtx->comment_lines = NULL;
    err = mtx_set_comment_lines(mtx, num_comment_lines, comment_lines);
    if (err) {
        mtx_vector_coordinate_data_free(mtxdata);
        return err;
    }

    mtx->num_rows = num_rows;
    mtx->num_columns = -1;
    mtx->num_nonzeros = size;
    return MTX_SUCCESS;
}

/**
 * `mtx_init_vector_coordinate_integer_single()' creates a coordinate
 * vector with single precision, integer coefficients.
 */
int mtx_init_vector_coordinate_integer_single(
    struct mtx * mtx,
    enum mtx_sorting sorting,
    enum mtx_assembly assembly,
    int num_comment_lines,
    const char ** comment_lines,
    int num_rows,
    int64_t size,
    const struct mtx_vector_coordinate_integer_single * data)
{
    struct mtx_vector_coordinate_data * mtxdata =
        &mtx->storage.vector_coordinate;
    int err = mtx_vector_coordinate_data_init_integer_single(
        mtxdata, sorting, assembly, num_rows, -1, size, data);
    if (err)
        return err;

    mtx->object = mtx_vector;
    mtx->format = mtx_coordinate;
    mtx->field = mtx_integer;
    mtx->symmetry = mtx_general;
    mtx->num_comment_lines = 0;
    mtx->comment_lines = NULL;
    err = mtx_set_comment_lines(mtx, num_comment_lines, comment_lines);
    if (err) {
        mtx_vector_coordinate_data_free(mtxdata);
        return err;
    }

    mtx->num_rows = num_rows;
    mtx->num_columns = -1;
    mtx->num_nonzeros = size;
    return MTX_SUCCESS;
}

/**
 * `mtx_init_vector_coordinate_pattern()` creates a coordinate vector
 * with boolean (pattern) coefficients.
 */
int mtx_init_vector_coordinate_pattern(
    struct mtx * mtx,
    enum mtx_sorting sorting,
    enum mtx_assembly assembly,
    int num_comment_lines,
    const char ** comment_lines,
    int num_rows,
    int64_t size,
    const struct mtx_vector_coordinate_pattern * data)
{
    struct mtx_vector_coordinate_data * mtxdata =
        &mtx->storage.vector_coordinate;
    int err = mtx_vector_coordinate_data_init_pattern(
        mtxdata, sorting, assembly, num_rows, -1, size, data);
    if (err)
        return err;

    mtx->object = mtx_vector;
    mtx->format = mtx_coordinate;
    mtx->field = mtx_pattern;
    mtx->symmetry = mtx_general;
    mtx->num_comment_lines = 0;
    mtx->comment_lines = NULL;
    err = mtx_set_comment_lines(mtx, num_comment_lines, comment_lines);
    if (err) {
        mtx_vector_coordinate_data_free(mtxdata);
        return err;
    }

    mtx->num_rows = num_rows;
    mtx->num_columns = -1;
    mtx->num_nonzeros = size;
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
    struct mtx_vector_coordinate_data * mtxdata =
        &mtx->storage.vector_coordinate;
    return mtx_vector_coordinate_data_set_zero(mtxdata);
}

/**
 * `mtx_vector_coordinate_set_constant_real_single()' sets every
 * nonzero value of a vector equal to a constant, single precision
 * floating point number.
 */
int mtx_vector_coordinate_set_constant_real_single(
    struct mtx * mtx,
    float a)
{
    if (mtx->object != mtx_vector)
        return MTX_ERR_INVALID_MTX_OBJECT;
    if (mtx->format != mtx_coordinate)
        return MTX_ERR_INVALID_MTX_FORMAT;
    struct mtx_vector_coordinate_data * mtxdata =
        &mtx->storage.vector_coordinate;
    return mtx_vector_coordinate_data_set_constant_real_single(mtxdata, a);
}

/**
 * `mtx_vector_coordinate_set_constant_real_double()' sets every
 * nonzero value of a vector equal to a constant, double precision
 * floating point number.
 */
int mtx_vector_coordinate_set_constant_real_double(
    struct mtx * mtx,
    double a)
{
    if (mtx->object != mtx_vector)
        return MTX_ERR_INVALID_MTX_OBJECT;
    if (mtx->format != mtx_coordinate)
        return MTX_ERR_INVALID_MTX_FORMAT;
    struct mtx_vector_coordinate_data * mtxdata =
        &mtx->storage.vector_coordinate;
    return mtx_vector_coordinate_data_set_constant_real_double(mtxdata, a);
}

/**
 * `mtx_vector_coordinate_set_constant_complex_single()' sets every
 * nonzero value of a vector equal to a constant, single precision
 * floating point complex number.
 */
int mtx_vector_coordinate_set_constant_complex_single(
    struct mtx * mtx,
    float a[2])
{
    if (mtx->object != mtx_vector)
        return MTX_ERR_INVALID_MTX_OBJECT;
    if (mtx->format != mtx_coordinate)
        return MTX_ERR_INVALID_MTX_FORMAT;
    struct mtx_vector_coordinate_data * mtxdata =
        &mtx->storage.vector_coordinate;
    return mtx_vector_coordinate_data_set_constant_complex_single(mtxdata, a);
}

/**
 * `mtx_vector_coordinate_set_constant_integer_single()' sets every
 * nonzero value of a vector equal to a constant, single precision
 * integer.
 */
int mtx_vector_coordinate_set_constant_integer_single(
    struct mtx * mtx,
    int32_t a)
{
    if (mtx->object != mtx_vector)
        return MTX_ERR_INVALID_MTX_OBJECT;
    if (mtx->format != mtx_coordinate)
        return MTX_ERR_INVALID_MTX_FORMAT;
    struct mtx_vector_coordinate_data * mtxdata =
        &mtx->storage.vector_coordinate;
    return mtx_vector_coordinate_data_set_constant_integer_single(mtxdata, a);
}
