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
 * Dense matrices in Matrix Market array format.
 */

#include <libmtx/matrix/array.h>

#include <libmtx/error.h>
#include <libmtx/mtx/mtx.h>
#include <libmtx/mtx/header.h>
#include <libmtx/mtx/precision.h>
#include <libmtx/mtx/sort.h>
#include <libmtx/mtx/triangle.h>

#include <errno.h>

#include <stdlib.h>
#include <string.h>

/*
 * Dense matrix allocation.
 */

/**
 * `mtx_alloc_matrix_array()` allocates a dense matrix in array
 * format.
 *
 * If `symmetry' is `mtx_symmetric', `mtx_skew_symmetric' or
 * `mtx_hermitian', then `triangle' must be either
 * `mtx_lower_triangular' or `mtx_upper_triangular' to indicate which
 * triangle of the matrix is stored in `data'.  Otherwise, if
 * `symmetry' is `mtx_general', then `triangle' must be
 * `mtx_nontriangular'.
 *
 * ´sorting' must be `mtx_row_major' or `mtx_column_major'.
 */
int mtx_alloc_matrix_array(
    struct mtx * mtx,
    enum mtx_field field,
    enum mtxprecision precision,
    enum mtx_symmetry symmetry,
    enum mtx_triangle triangle,
    enum mtx_sorting sorting,
    int num_comment_lines,
    const char ** comment_lines,
    int num_rows,
    int num_columns)
{
    int err;
    if (sorting != mtx_row_major &&
        sorting != mtx_column_major)
    {
        return MTX_ERR_INVALID_MTX_SORTING;
    }

    mtx->object = mtx_matrix;
    mtx->format = mtx_array;
    mtx->field = field;
    mtx->symmetry = symmetry;

    mtx->num_comment_lines = 0;
    mtx->comment_lines = NULL;
    err = mtx_set_comment_lines(mtx, num_comment_lines, comment_lines);
    if (err)
        return err;

    mtx->num_rows = num_rows;
    mtx->num_columns = num_columns;
    mtx->num_nonzeros = -1;

    err = mtx_matrix_array_data_alloc(
        &mtx->storage.matrix_array,
        field, precision, symmetry, triangle,
        num_rows, num_columns);
    if (err) {
        for (int i = 0; i < num_comment_lines; i++)
            free(mtx->comment_lines[i]);
        free(mtx->comment_lines);
        return err;
    }
    return MTX_SUCCESS;
}

/*
 * Array matrix allocation and initialisation.
 */

/**
 * `mtx_init_matrix_array_real_single()` creates a dense matrix with
 * real, single-precision floating point coefficients.
 *
 * If `symmetry' is `mtx_symmetric', `mtx_skew_symmetric' or
 * `mtx_hermitian', then `triangle' must be either
 * `mtx_lower_triangular' or `mtx_upper_triangular' to indicate which
 * triangle of the matrix is stored in `data'.  Otherwise, if
 * `symmetry' is `mtx_general', then `triangle' must be
 * `mtx_nontriangular'.
 *
 * ´sorting' must be `mtx_row_major' or `mtx_column_major'.
 */
int mtx_init_matrix_array_real_single(
    struct mtx * mtx,
    enum mtx_symmetry symmetry,
    enum mtx_triangle triangle,
    enum mtx_sorting sorting,
    int num_comment_lines,
    const char ** comment_lines,
    int num_rows,
    int num_columns,
    int64_t size,
    const float * data)
{
    struct mtx_matrix_array_data * mtxdata =
        &mtx->storage.matrix_array;
    int err = mtx_matrix_array_data_init_real_single(
        mtxdata, symmetry, triangle,
        num_rows, num_columns, size, data);
    if (err)
        return err;

    mtx->object = mtx_matrix;
    mtx->format = mtx_array;
    mtx->field = mtx_real;
    mtx->symmetry = symmetry;
    mtx->num_comment_lines = 0;
    mtx->comment_lines = NULL;
    err = mtx_set_comment_lines(mtx, num_comment_lines, comment_lines);
    if (err) {
        mtx_matrix_array_data_free(mtxdata);
        return err;
    }

    mtx->num_rows = num_rows;
    mtx->num_columns = num_columns;
    mtx->num_nonzeros = -1;
    return MTX_SUCCESS;
}

/**
 * `mtx_init_matrix_array_real_double()` creates a dense matrix with
 * real, double-precision floating point coefficients.
 *
 * If `symmetry' is `mtx_symmetric', `mtx_skew_symmetric' or
 * `mtx_hermitian', then `triangle' must be either
 * `mtx_lower_triangular' or `mtx_upper_triangular' to indicate which
 * triangle of the matrix is stored in `data'.  Otherwise, if
 * `symmetry' is `mtx_general', then `triangle' must be
 * `mtx_nontriangular'.
 *
 * ´sorting' must be `mtx_row_major' or `mtx_column_major'.
 */
int mtx_init_matrix_array_real_double(
    struct mtx * mtx,
    enum mtx_symmetry symmetry,
    enum mtx_triangle triangle,
    enum mtx_sorting sorting,
    int num_comment_lines,
    const char ** comment_lines,
    int num_rows,
    int num_columns,
    int64_t size,
    const double * data)
{
    struct mtx_matrix_array_data * mtxdata =
        &mtx->storage.matrix_array;
    int err = mtx_matrix_array_data_init_real_double(
        mtxdata, symmetry, triangle,
        num_rows, num_columns, size, data);
    if (err)
        return err;

    mtx->object = mtx_matrix;
    mtx->format = mtx_array;
    mtx->field = mtx_real;
    mtx->symmetry = symmetry;
    mtx->num_comment_lines = 0;
    mtx->comment_lines = NULL;
    err = mtx_set_comment_lines(mtx, num_comment_lines, comment_lines);
    if (err) {
        mtx_matrix_array_data_free(mtxdata);
        return err;
    }

    mtx->num_rows = num_rows;
    mtx->num_columns = num_columns;
    mtx->num_nonzeros = -1;
    return MTX_SUCCESS;
}

/**
 * `mtx_init_matrix_array_complex_single()` creates a dense matrix
 * with complex, single-precision floating point coefficients.
 *
 * If `symmetry' is `mtx_symmetric', `mtx_skew_symmetric' or
 * `mtx_hermitian', then `triangle' must be either
 * `mtx_lower_triangular' or `mtx_upper_triangular' to indicate which
 * triangle of the matrix is stored in `data'.  Otherwise, if
 * `symmetry' is `mtx_general', then `triangle' must be
 * `mtx_nontriangular'.
 *
 * ´sorting' must be `mtx_row_major' or `mtx_column_major'.
 */
int mtx_init_matrix_array_complex_single(
    struct mtx * mtx,
    enum mtx_symmetry symmetry,
    enum mtx_triangle triangle,
    enum mtx_sorting sorting,
    int num_comment_lines,
    const char ** comment_lines,
    int num_rows,
    int num_columns,
    int64_t size,
    const float (* data)[2])
{
    struct mtx_matrix_array_data * mtxdata =
        &mtx->storage.matrix_array;
    int err = mtx_matrix_array_data_init_complex_single(
        mtxdata, symmetry, triangle,
        num_rows, num_columns, size, data);
    if (err)
        return err;

    mtx->object = mtx_matrix;
    mtx->format = mtx_array;
    mtx->field = mtx_complex;
    mtx->symmetry = symmetry;
    mtx->num_comment_lines = 0;
    mtx->comment_lines = NULL;
    err = mtx_set_comment_lines(mtx, num_comment_lines, comment_lines);
    if (err) {
        mtx_matrix_array_data_free(mtxdata);
        return err;
    }

    mtx->num_rows = num_rows;
    mtx->num_columns = num_columns;
    mtx->num_nonzeros = -1;
    return MTX_SUCCESS;
}

/**
 * `mtx_init_matrix_array_integer_single()` creates a dense matrix
 * with integer coefficients.
 *
 * If `symmetry' is `mtx_symmetric', `mtx_skew_symmetric' or
 * `mtx_hermitian', then `triangle' must be either
 * `mtx_lower_triangular' or `mtx_upper_triangular' to indicate which
 * triangle of the matrix is stored in `data'.  Otherwise, if
 * `symmetry' is `mtx_general', then `triangle' must be
 * `mtx_nontriangular'.
 *
 * ´sorting' must be `mtx_row_major' or `mtx_column_major'.
 */
int mtx_init_matrix_array_integer_single(
    struct mtx * mtx,
    enum mtx_symmetry symmetry,
    enum mtx_triangle triangle,
    enum mtx_sorting sorting,
    int num_comment_lines,
    const char ** comment_lines,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * data)
{
    struct mtx_matrix_array_data * mtxdata =
        &mtx->storage.matrix_array;
    int err = mtx_matrix_array_data_init_integer_single(
        mtxdata, symmetry, triangle,
        num_rows, num_columns, size, data);
    if (err)
        return err;

    mtx->object = mtx_matrix;
    mtx->format = mtx_array;
    mtx->field = mtx_integer;
    mtx->symmetry = symmetry;
    mtx->num_comment_lines = 0;
    mtx->comment_lines = NULL;
    err = mtx_set_comment_lines(mtx, num_comment_lines, comment_lines);
    if (err) {
        mtx_matrix_array_data_free(mtxdata);
        return err;
    }

    mtx->num_rows = num_rows;
    mtx->num_columns = num_columns;
    mtx->num_nonzeros = -1;
    return MTX_SUCCESS;
}

/*
 * Other dense matrix functions.
 */

/**
 * `mtx_matrix_array_set_zero()' zeroes a matrix in array format.
 */
int mtx_matrix_array_set_zero(
    struct mtx * mtx)
{
    if (mtx->object != mtx_matrix)
        return MTX_ERR_INVALID_MTX_OBJECT;
    if (mtx->format != mtx_array)
        return MTX_ERR_INVALID_MTX_FORMAT;
    struct mtx_matrix_array_data * mtxdata =
        &mtx->storage.matrix_array;
    return mtx_matrix_array_data_set_zero(mtxdata);
}

/**
 * `mtx_matrix_array_set_constant_real_single()' sets every value of a
 * matrix equal to a constant, single precision floating point number.
 */
int mtx_matrix_array_set_constant_real_single(
    struct mtx * mtx,
    float a)
{
    if (mtx->object != mtx_matrix)
        return MTX_ERR_INVALID_MTX_OBJECT;
    if (mtx->format != mtx_array)
        return MTX_ERR_INVALID_MTX_FORMAT;
    struct mtx_matrix_array_data * mtxdata =
        &mtx->storage.matrix_array;
    return mtx_matrix_array_data_set_constant_real_single(
        mtxdata, a);
}

/**
 * `mtx_matrix_array_set_constant_real_double()' sets every value of a
 * matrix equal to a constant, double precision floating point number.
 */
int mtx_matrix_array_set_constant_real_double(
    struct mtx * mtx,
    double a)
{
    if (mtx->object != mtx_matrix)
        return MTX_ERR_INVALID_MTX_OBJECT;
    if (mtx->format != mtx_array)
        return MTX_ERR_INVALID_MTX_FORMAT;
    struct mtx_matrix_array_data * mtxdata =
        &mtx->storage.matrix_array;
    return mtx_matrix_array_data_set_constant_real_double(
        mtxdata, a);
}

/**
 * `mtx_matrix_array_set_constant_complex_single()' sets every value
 * of a matrix equal to a constant, single precision floating point
 * complex number.
 */
int mtx_matrix_array_set_constant_complex_single(
    struct mtx * mtx,
    float a[2])
{
    if (mtx->object != mtx_matrix)
        return MTX_ERR_INVALID_MTX_OBJECT;
    if (mtx->format != mtx_array)
        return MTX_ERR_INVALID_MTX_FORMAT;
    struct mtx_matrix_array_data * mtxdata =
        &mtx->storage.matrix_array;
    return mtx_matrix_array_data_set_constant_complex_single(
        mtxdata, a);
}

/**
 * `mtx_matrix_array_set_constant_integer_single()' sets every value
 * of a matrix equal to a constant, single precision integer.
 */
int mtx_matrix_array_set_constant_integer_single(
    struct mtx * mtx,
    int32_t a)
{
    if (mtx->object != mtx_matrix)
        return MTX_ERR_INVALID_MTX_OBJECT;
    if (mtx->format != mtx_array)
        return MTX_ERR_INVALID_MTX_FORMAT;
    struct mtx_matrix_array_data * mtxdata =
        &mtx->storage.matrix_array;
    return mtx_matrix_array_data_set_constant_integer_single(
        mtxdata, a);
    return MTX_SUCCESS;
}
