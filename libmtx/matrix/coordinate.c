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
 * Sparse matrices in Matrix Market format.
 */

#include <libmtx/matrix/coordinate.h>

#include <libmtx/mtx/assembly.h>
#include <libmtx/error.h>
#include <libmtx/mtx/mtx.h>
#include <libmtx/mtx/header.h>
#include <libmtx/mtx/reorder.h>
#include <libmtx/mtx/sort.h>
#include <libmtx/mtx/triangle.h>

#include <errno.h>

#include <stdlib.h>
#include <string.h>

/*
 * Coordinate matrix allocation.
 */

/**
 * `mtx_alloc_matrix_coordinate()` allocates a sparse matrix in
 * coordinate format.
 */
int mtx_alloc_matrix_coordinate(
    struct mtx * mtx,
    enum mtx_field field,
    enum mtx_precision precision,
    enum mtx_symmetry symmetry,
    int num_comment_lines,
    const char ** comment_lines,
    int num_rows,
    int num_columns,
    int64_t num_nonzeros)
{
    int err;
    mtx->object = mtx_matrix;
    mtx->format = mtx_coordinate;
    mtx->field = field;
    mtx->symmetry = symmetry;

    mtx->num_comment_lines = 0;
    mtx->comment_lines = NULL;
    err = mtx_set_comment_lines(mtx, num_comment_lines, comment_lines);
    if (err)
        return err;

    mtx->num_rows = num_rows;
    mtx->num_columns = num_columns;
    mtx->num_nonzeros = num_nonzeros;

    /* Allocate storage for matrix data. */
    err = mtx_matrix_coordinate_data_alloc(
        &mtx->storage.matrix_coordinate,
        field, precision,
        mtx->num_rows, mtx->num_columns, mtx->num_nonzeros);
    if (err) {
        for (int i = 0; i < num_comment_lines; i++)
            free(mtx->comment_lines[i]);
        free(mtx->comment_lines);
        return err;
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_init_matrix_coordinate_real_single()` creates a sparse matrix
 * with real, single-precision floating point coefficients.
 */
int mtx_init_matrix_coordinate_real_single(
    struct mtx * mtx,
    enum mtx_symmetry symmetry,
    enum mtx_triangle triangle,
    enum mtx_sorting sorting,
    enum mtx_assembly assembly,
    int num_comment_lines,
    const char ** comment_lines,
    int num_rows,
    int num_columns,
    int64_t size,
    const struct mtx_matrix_coordinate_real_single * data)
{
    struct mtx_matrix_coordinate_data * mtxdata =
        &mtx->storage.matrix_coordinate;
    int err = mtx_matrix_coordinate_data_init_real_single(
        mtxdata, symmetry, triangle, sorting, assembly,
        num_rows, num_columns, size, data);
    if (err)
        return err;

    mtx->object = mtx_matrix;
    mtx->format = mtx_coordinate;
    mtx->field = mtx_real;
    mtx->symmetry = symmetry;
    mtx->num_comment_lines = 0;
    mtx->comment_lines = NULL;
    err = mtx_set_comment_lines(mtx, num_comment_lines, comment_lines);
    if (err) {
        mtx_matrix_coordinate_data_free(mtxdata);
        return err;
    }

    mtx->num_rows = num_rows;
    mtx->num_columns = num_columns;
    mtx->num_nonzeros = size;
    return MTX_SUCCESS;
}

/**
 * `mtx_init_matrix_coordinate_real_double()` creates a sparse matrix
 * with real, double-precision floating point coefficients.
 */
int mtx_init_matrix_coordinate_real_double(
    struct mtx * mtx,
    enum mtx_symmetry symmetry,
    enum mtx_triangle triangle,
    enum mtx_sorting sorting,
    enum mtx_assembly assembly,
    int num_comment_lines,
    const char ** comment_lines,
    int num_rows,
    int num_columns,
    int64_t size,
    const struct mtx_matrix_coordinate_real_double * data)
{
    struct mtx_matrix_coordinate_data * mtxdata =
        &mtx->storage.matrix_coordinate;
    int err = mtx_matrix_coordinate_data_init_real_double(
        mtxdata, symmetry, triangle, sorting, assembly,
        num_rows, num_columns, size, data);
    if (err)
        return err;

    mtx->object = mtx_matrix;
    mtx->format = mtx_coordinate;
    mtx->field = mtx_real;
    mtx->symmetry = symmetry;
    mtx->num_comment_lines = 0;
    mtx->comment_lines = NULL;
    err = mtx_set_comment_lines(mtx, num_comment_lines, comment_lines);
    if (err) {
        mtx_matrix_coordinate_data_free(mtxdata);
        return err;
    }

    mtx->num_rows = num_rows;
    mtx->num_columns = num_columns;
    mtx->num_nonzeros = size;
    return MTX_SUCCESS;
}

/**
 * `mtx_init_matrix_coordinate_complex_single()` creates a sparse
 * matrix with complex, single-precision floating point coefficients.
 */
int mtx_init_matrix_coordinate_complex_single(
    struct mtx * mtx,
    enum mtx_symmetry symmetry,
    enum mtx_triangle triangle,
    enum mtx_sorting sorting,
    enum mtx_assembly assembly,
    int num_comment_lines,
    const char ** comment_lines,
    int num_rows,
    int num_columns,
    int64_t size,
    const struct mtx_matrix_coordinate_complex_single * data)
{
    struct mtx_matrix_coordinate_data * mtxdata =
        &mtx->storage.matrix_coordinate;
    int err = mtx_matrix_coordinate_data_init_complex_single(
        mtxdata, symmetry, triangle, sorting, assembly,
        num_rows, num_columns, size, data);
    if (err)
        return err;

    mtx->object = mtx_matrix;
    mtx->format = mtx_coordinate;
    mtx->field = mtx_complex;
    mtx->symmetry = symmetry;
    mtx->num_comment_lines = 0;
    mtx->comment_lines = NULL;
    err = mtx_set_comment_lines(mtx, num_comment_lines, comment_lines);
    if (err) {
        mtx_matrix_coordinate_data_free(mtxdata);
        return err;
    }

    mtx->num_rows = num_rows;
    mtx->num_columns = num_columns;
    mtx->num_nonzeros = size;
    return MTX_SUCCESS;
}

/**
 * `mtx_init_matrix_coordinate_integer_single()` creates a sparse
 * matrix with single precision, integer coefficients.
 */
int mtx_init_matrix_coordinate_integer_single(
    struct mtx * mtx,
    enum mtx_symmetry symmetry,
    enum mtx_triangle triangle,
    enum mtx_sorting sorting,
    enum mtx_assembly assembly,
    int num_comment_lines,
    const char ** comment_lines,
    int num_rows,
    int num_columns,
    int64_t size,
    const struct mtx_matrix_coordinate_integer_single * data)
{
    struct mtx_matrix_coordinate_data * mtxdata =
        &mtx->storage.matrix_coordinate;
    int err = mtx_matrix_coordinate_data_init_integer_single(
        mtxdata, symmetry, triangle, sorting, assembly,
        num_rows, num_columns, size, data);
    if (err)
        return err;

    mtx->object = mtx_matrix;
    mtx->format = mtx_coordinate;
    mtx->field = mtx_integer;
    mtx->symmetry = symmetry;
    mtx->num_comment_lines = 0;
    mtx->comment_lines = NULL;
    err = mtx_set_comment_lines(mtx, num_comment_lines, comment_lines);
    if (err) {
        mtx_matrix_coordinate_data_free(mtxdata);
        return err;
    }

    mtx->num_rows = num_rows;
    mtx->num_columns = num_columns;
    mtx->num_nonzeros = size;
    return MTX_SUCCESS;
}

/**
 * `mtx_init_matrix_coordinate_pattern()` creates a sparse matrix
 * with boolean coefficients.
 */
int mtx_init_matrix_coordinate_pattern(
    struct mtx * mtx,
    enum mtx_symmetry symmetry,
    enum mtx_triangle triangle,
    enum mtx_sorting sorting,
    enum mtx_assembly assembly,
    int num_comment_lines,
    const char ** comment_lines,
    int num_rows,
    int num_columns,
    int64_t size,
    const struct mtx_matrix_coordinate_pattern * data)
{
    struct mtx_matrix_coordinate_data * mtxdata =
        &mtx->storage.matrix_coordinate;
    int err = mtx_matrix_coordinate_data_init_pattern(
        mtxdata, symmetry, triangle, sorting, assembly,
        num_rows, num_columns, size, data);
    if (err)
        return err;

    mtx->object = mtx_matrix;
    mtx->format = mtx_coordinate;
    mtx->field = mtx_pattern;
    mtx->symmetry = symmetry;
    mtx->num_comment_lines = 0;
    mtx->comment_lines = NULL;
    err = mtx_set_comment_lines(mtx, num_comment_lines, comment_lines);
    if (err) {
        mtx_matrix_coordinate_data_free(mtxdata);
        return err;
    }

    mtx->num_rows = num_rows;
    mtx->num_columns = num_columns;
    mtx->num_nonzeros = size;
    return MTX_SUCCESS;
}

/**
 * `mtx_matrix_coordinate_set_zero()' zeroes a matrix in coordinate
 * format.
 */
int mtx_matrix_coordinate_set_zero(
    struct mtx * mtx)
{
    if (mtx->object != mtx_matrix)
        return MTX_ERR_INVALID_MTX_OBJECT;
    if (mtx->format != mtx_coordinate)
        return MTX_ERR_INVALID_MTX_FORMAT;
    struct mtx_matrix_coordinate_data * mtxdata =
        &mtx->storage.matrix_coordinate;
    return mtx_matrix_coordinate_data_set_zero(mtxdata);
}

/**
 * `mtx_matrix_coordinate_set_constant_real_single()' sets every
 * nonzero value of a matrix equal to a constant, single precision
 * floating point number.
 */
int mtx_matrix_coordinate_set_constant_real_single(
    struct mtx * mtx,
    float a)
{
    if (mtx->object != mtx_matrix)
        return MTX_ERR_INVALID_MTX_OBJECT;
    if (mtx->format != mtx_coordinate)
        return MTX_ERR_INVALID_MTX_FORMAT;
    struct mtx_matrix_coordinate_data * mtxdata =
        &mtx->storage.matrix_coordinate;
    return mtx_matrix_coordinate_data_set_constant_real_single(mtxdata, a);
}

/**
 * `mtx_matrix_coordinate_set_constant_real_double()' sets every
 * nonzero value of a matrix equal to a constant, double precision
 * floating point number.
 */
int mtx_matrix_coordinate_set_constant_real_double(
    struct mtx * mtx,
    double a)
{
    if (mtx->object != mtx_matrix)
        return MTX_ERR_INVALID_MTX_OBJECT;
    if (mtx->format != mtx_coordinate)
        return MTX_ERR_INVALID_MTX_FORMAT;
    struct mtx_matrix_coordinate_data * mtxdata =
        &mtx->storage.matrix_coordinate;
    return mtx_matrix_coordinate_data_set_constant_real_double(mtxdata, a);
}

/**
 * `mtx_matrix_coordinate_set_constant_complex_single()' sets every
 * nonzero value of a matrix equal to a constant, single precision
 * floating point complex number.
 */
int mtx_matrix_coordinate_set_constant_complex_single(
    struct mtx * mtx,
    float a[2])
{
    if (mtx->object != mtx_matrix)
        return MTX_ERR_INVALID_MTX_OBJECT;
    if (mtx->format != mtx_coordinate)
        return MTX_ERR_INVALID_MTX_FORMAT;
    struct mtx_matrix_coordinate_data * mtxdata =
        &mtx->storage.matrix_coordinate;
    return mtx_matrix_coordinate_data_set_constant_complex_single(mtxdata, a);
}

/**
 * `mtx_matrix_coordinate_set_constant_integer_single()' sets every
 * nonzero value of a matrix equal to a constant, single precision
 * integer.
 */
int mtx_matrix_coordinate_set_constant_integer_single(
    struct mtx * mtx,
    int32_t a)
{
    if (mtx->object != mtx_matrix)
        return MTX_ERR_INVALID_MTX_OBJECT;
    if (mtx->format != mtx_coordinate)
        return MTX_ERR_INVALID_MTX_FORMAT;
    struct mtx_matrix_coordinate_data * mtxdata =
        &mtx->storage.matrix_coordinate;
    return mtx_matrix_coordinate_data_set_constant_integer_single(mtxdata, a);
}
