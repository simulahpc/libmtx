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
 * Reordering the rows and columns of matrices in coordinate format.
 */

#include <libmtx/matrix/coordinate/reorder.h>

#include <libmtx/error.h>
#include <libmtx/mtx/header.h>
#include <libmtx/matrix/coordinate/coordinate.h>
#include <libmtx/mtx/matrix.h>
#include <libmtx/mtx/mtx.h>
#include <libmtx/mtx/reorder.h>

#include <errno.h>

#include <stdlib.h>

static int mtx_matrix_coordinate_real_permute(
    struct mtx * mtx,
    const int * rowperm,
    const int * colperm)
{
    struct mtx_matrix_coordinate_real * data =
        (struct mtx_matrix_coordinate_real *) mtx->data;
    if (rowperm && colperm) {
        for (int64_t k = 0; k < mtx->size; k++) {
            data[k].i = rowperm[data[k].i-1];
            data[k].j = colperm[data[k].j-1];
        }
    } else if (rowperm) {
        for (int64_t k = 0; k < mtx->size; k++) {
            data[k].i = rowperm[data[k].i-1];
        }
    } else if (colperm) {
        for (int64_t k = 0; k < mtx->size; k++) {
            data[k].j = colperm[data[k].j-1];
        }
    }
    mtx->sorting = mtx_unsorted;
    return MTX_SUCCESS;
}

static int mtx_matrix_coordinate_double_permute(
    struct mtx * mtx,
    const int * rowperm,
    const int * colperm)
{
    struct mtx_matrix_coordinate_double * data =
        (struct mtx_matrix_coordinate_double *) mtx->data;
    if (rowperm && colperm) {
        for (int64_t k = 0; k < mtx->size; k++) {
            data[k].i = rowperm[data[k].i-1];
            data[k].j = colperm[data[k].j-1];
        }
    } else if (rowperm) {
        for (int64_t k = 0; k < mtx->size; k++) {
            data[k].i = rowperm[data[k].i-1];
        }
    } else if (colperm) {
        for (int64_t k = 0; k < mtx->size; k++) {
            data[k].j = colperm[data[k].j-1];
        }
    }
    mtx->sorting = mtx_unsorted;
    return MTX_SUCCESS;
}

static int mtx_matrix_coordinate_complex_permute(
    struct mtx * mtx,
    const int * rowperm,
    const int * colperm)
{
    struct mtx_matrix_coordinate_complex * data =
        (struct mtx_matrix_coordinate_complex *) mtx->data;
    if (rowperm && colperm) {
        for (int64_t k = 0; k < mtx->size; k++) {
            data[k].i = rowperm[data[k].i-1];
            data[k].j = colperm[data[k].j-1];
        }
    } else if (rowperm) {
        for (int64_t k = 0; k < mtx->size; k++) {
            data[k].i = rowperm[data[k].i-1];
        }
    } else if (colperm) {
        for (int64_t k = 0; k < mtx->size; k++) {
            data[k].j = colperm[data[k].j-1];
        }
    }
    mtx->sorting = mtx_unsorted;
    return MTX_SUCCESS;
}

static int mtx_matrix_coordinate_integer_permute(
    struct mtx * mtx,
    const int * rowperm,
    const int * colperm)
{
    struct mtx_matrix_coordinate_integer * data =
        (struct mtx_matrix_coordinate_integer *) mtx->data;
    if (rowperm && colperm) {
        for (int64_t k = 0; k < mtx->size; k++) {
            data[k].i = rowperm[data[k].i-1];
            data[k].j = colperm[data[k].j-1];
        }
    } else if (rowperm) {
        for (int64_t k = 0; k < mtx->size; k++) {
            data[k].i = rowperm[data[k].i-1];
        }
    } else if (colperm) {
        for (int64_t k = 0; k < mtx->size; k++) {
            data[k].j = colperm[data[k].j-1];
        }
    }
    mtx->sorting = mtx_unsorted;
    return MTX_SUCCESS;
}

static int mtx_matrix_coordinate_pattern_permute(
    struct mtx * mtx,
    const int * rowperm,
    const int * colperm)
{
    struct mtx_matrix_coordinate_pattern * data =
        (struct mtx_matrix_coordinate_pattern *) mtx->data;
    if (rowperm && colperm) {
        for (int64_t k = 0; k < mtx->size; k++) {
            data[k].i = rowperm[data[k].i-1];
            data[k].j = colperm[data[k].j-1];
        }
    } else if (rowperm) {
        for (int64_t k = 0; k < mtx->size; k++) {
            data[k].i = rowperm[data[k].i-1];
        }
    } else if (colperm) {
        for (int64_t k = 0; k < mtx->size; k++) {
            data[k].j = colperm[data[k].j-1];
        }
    }
    mtx->sorting = mtx_unsorted;
    return MTX_SUCCESS;
}

/**
 * `mtx_matrix_coordinate_permute()' permutes the elements of a matrix
 * based on a given permutation.
 *
 * The array `row_permutation' should be a permutation of the integers
 * `1,2,...,mtx->num_rows', and the array `column_permutation' should
 * be a permutation of the integers `1,2,...,mtx->num_columns'. The
 * elements belonging to row `i' and column `j' in the permuted matrix
 * are then equal to the elements in row `row_permutation[i-1]' and
 * column `column_permutation[j-1]' in the original matrix, for
 * `i=1,2,...,mtx->num_rows' and `j=1,2,...,mtx->num_columns'.
 */
int mtx_matrix_coordinate_permute(
    struct mtx * mtx,
    const int * row_permutation,
    const int * column_permutation)
{
    if (mtx->field == mtx_real) {
        return mtx_matrix_coordinate_real_permute(
            mtx, row_permutation, column_permutation);
    } else if (mtx->field == mtx_double) {
        return mtx_matrix_coordinate_double_permute(
            mtx, row_permutation, column_permutation);
    } else if (mtx->field == mtx_complex) {
        return mtx_matrix_coordinate_complex_permute(
            mtx, row_permutation, column_permutation);
    } else if (mtx->field == mtx_integer) {
        return mtx_matrix_coordinate_integer_permute(
            mtx, row_permutation, column_permutation);
    } else if (mtx->field == mtx_pattern) {
        return mtx_matrix_coordinate_pattern_permute(
            mtx, row_permutation, column_permutation);
    } else {
        return MTX_ERR_INVALID_MTX_FIELD;
    }
    return MTX_SUCCESS;
}
