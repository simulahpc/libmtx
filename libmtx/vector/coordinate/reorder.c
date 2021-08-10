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
 * Reordering the rows or columns of vectors in coordinate format.
 */

#include <libmtx/vector/array/reorder.h>

#include <libmtx/error.h>
#include <libmtx/mtx/mtx.h>
#include <libmtx/mtx/reorder.h>
#include <libmtx/vector/coordinate/coordinate.h>

static int mtx_vector_coordinate_real_permute(
    struct mtx * mtx,
    const int * row_permutation,
    const int * column_permutation)
{
    const int * permutation;
    if (mtx->num_rows >= 0 && mtx->num_columns == -1) {
        permutation = row_permutation;
    } else if (mtx->num_rows == -1 && mtx->num_columns >= 0) {
        permutation = column_permutation;
    } else {
        return MTX_ERR_INVALID_MTX_SIZE;
    }

    struct mtx_vector_coordinate_real * data =
        (struct mtx_vector_coordinate_real *) mtx->data;
    for (int i = 0; i < mtx->size; i++)
        data[i].i = permutation[data[i].i-1];
    mtx->sorting = mtx_unsorted;
    return MTX_SUCCESS;
}

static int mtx_vector_coordinate_double_permute(
    struct mtx * mtx,
    const int * row_permutation,
    const int * column_permutation)
{
    const int * permutation;
    if (mtx->num_rows >= 0 && mtx->num_columns == -1) {
        permutation = row_permutation;
    } else if (mtx->num_rows == -1 && mtx->num_columns >= 0) {
        permutation = column_permutation;
    } else {
        return MTX_ERR_INVALID_MTX_SIZE;
    }

    struct mtx_vector_coordinate_double * data =
        (struct mtx_vector_coordinate_double *) mtx->data;
    for (int i = 0; i < mtx->size; i++)
        data[i].i = permutation[data[i].i-1];
    mtx->sorting = mtx_unsorted;
    return MTX_SUCCESS;
}

static int mtx_vector_coordinate_complex_permute(
    struct mtx * mtx,
    const int * row_permutation,
    const int * column_permutation)
{
    const int * permutation;
    if (mtx->num_rows >= 0 && mtx->num_columns == -1) {
        permutation = row_permutation;
    } else if (mtx->num_rows == -1 && mtx->num_columns >= 0) {
        permutation = column_permutation;
    } else {
        return MTX_ERR_INVALID_MTX_SIZE;
    }

    struct mtx_vector_coordinate_complex * data =
        (struct mtx_vector_coordinate_complex *) mtx->data;
    for (int i = 0; i < mtx->size; i++)
        data[i].i = permutation[data[i].i-1];
    mtx->sorting = mtx_unsorted;
    return MTX_SUCCESS;
}

static int mtx_vector_coordinate_integer_permute(
    struct mtx * mtx,
    const int * row_permutation,
    const int * column_permutation)
{
    const int * permutation;
    if (mtx->num_rows >= 0 && mtx->num_columns == -1) {
        permutation = row_permutation;
    } else if (mtx->num_rows == -1 && mtx->num_columns >= 0) {
        permutation = column_permutation;
    } else {
        return MTX_ERR_INVALID_MTX_SIZE;
    }

    struct mtx_vector_coordinate_integer * data =
        (struct mtx_vector_coordinate_integer *) mtx->data;
    for (int i = 0; i < mtx->size; i++)
        data[i].i = permutation[data[i].i-1];
    mtx->sorting = mtx_unsorted;
    return MTX_SUCCESS;
}

static int mtx_vector_coordinate_pattern_permute(
    struct mtx * mtx,
    const int * row_permutation,
    const int * column_permutation)
{
    const int * permutation;
    if (mtx->num_rows >= 0 && mtx->num_columns == -1) {
        permutation = row_permutation;
    } else if (mtx->num_rows == -1 && mtx->num_columns >= 0) {
        permutation = column_permutation;
    } else {
        return MTX_ERR_INVALID_MTX_SIZE;
    }

    struct mtx_vector_coordinate_pattern * data =
        (struct mtx_vector_coordinate_pattern *) mtx->data;
    for (int i = 0; i < mtx->size; i++)
        data[i].i = permutation[data[i].i-1];
    mtx->sorting = mtx_unsorted;
    return MTX_SUCCESS;
}

/**
 * `mtx_vector_coordinate_permute()' permutes the elements of a vector
 * based on a given permutation.
 *
 * The array `row_permutation' should be a permutation of the integers
 * `1,2,...,mtx->num_rows', and the array `column_permutation' should
 * be a permutation of the integers `1,2,...,mtx->num_columns'. The
 * elements belonging to row `i' (or column `j') in the permuted
 * vector are then equal to the elements in row `row_permutation[i-1]'
 * (or column `column_permutation[j-1]') in the original vector, for
 * `i=1,2,...,mtx->num_rows' (and `j=1,2,...,mtx->num_columns').
 */
int mtx_vector_coordinate_permute(
    struct mtx * mtx,
    const int * row_permutation,
    const int * column_permutation)
{
    if (mtx->object != mtx_vector)
        return MTX_ERR_INVALID_MTX_OBJECT;
    if (mtx->format != mtx_coordinate)
        return MTX_ERR_INVALID_MTX_FORMAT;

    if (mtx->field == mtx_real) {
        return mtx_vector_coordinate_real_permute(
            mtx, row_permutation, column_permutation);
    } else if (mtx->field == mtx_double) {
        return mtx_vector_coordinate_double_permute(
            mtx, row_permutation, column_permutation);
    } else if (mtx->field == mtx_complex) {
        return mtx_vector_coordinate_complex_permute(
            mtx, row_permutation, column_permutation);
    } else if (mtx->field == mtx_integer) {
        return mtx_vector_coordinate_integer_permute(
            mtx, row_permutation, column_permutation);
    } else if (mtx->field == mtx_pattern) {
        return mtx_vector_coordinate_pattern_permute(
            mtx, row_permutation, column_permutation);
    } else {
        return MTX_ERR_INVALID_MTX_FIELD;
    }
    return MTX_SUCCESS;
}
