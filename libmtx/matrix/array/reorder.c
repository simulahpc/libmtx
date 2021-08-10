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
 * Reordering the rows and columns of matrices in array format.
 */

#include <libmtx/matrix/array/reorder.h>

#include <libmtx/error.h>
#include <libmtx/mtx/mtx.h>
#include <libmtx/mtx/reorder.h>

static int mtx_matrix_array_real_permute(
    struct mtx * mtx,
    const int * rowperm,
    const int * colperm)
{
    struct mtx orig;
    int err = mtx_copy(&orig, mtx);
    if (err)
        return err;

    const float * src = (const float *) orig.data;
    float * dst = (float *) mtx->data;
    if (rowperm && colperm) {
        for (int i = 0; i < mtx->num_rows; i++) {
            for (int j = 0; j < mtx->num_columns; j++) {
                int k = i*mtx->num_columns+j;
                int l = (rowperm[i]-1)*mtx->num_columns + colperm[j]-1;
                dst[k] = src[l];
            }
        }
    } else if (rowperm) {
        for (int i = 0; i < mtx->num_rows; i++) {
            for (int j = 0; j < mtx->num_columns; j++) {
                int k = i*mtx->num_columns+j;
                int l = (rowperm[i]-1)*mtx->num_columns + j;
                dst[k] = src[l];
            }
        }
    } else if (colperm) {
        for (int i = 0; i < mtx->num_rows; i++) {
            for (int j = 0; j < mtx->num_columns; j++) {
                int k = i*mtx->num_columns+j;
                int l = i*mtx->num_columns + colperm[j]-1;
                dst[k] = src[l];
            }
        }
    }

    mtx_free(&orig);
    return MTX_SUCCESS;
}

static int mtx_matrix_array_double_permute(
    struct mtx * mtx,
    const int * rowperm,
    const int * colperm)
{
    struct mtx orig;
    int err = mtx_copy(&orig, mtx);
    if (err)
        return err;

    const double * src = (const double *) orig.data;
    double * dst = (double *) mtx->data;
    if (rowperm && colperm) {
        for (int i = 0; i < mtx->num_rows; i++) {
            for (int j = 0; j < mtx->num_columns; j++) {
                int k = i*mtx->num_columns+j;
                int l = (rowperm[i]-1)*mtx->num_columns + colperm[j]-1;
                dst[k] = src[l];
            }
        }
    } else if (rowperm) {
        for (int i = 0; i < mtx->num_rows; i++) {
            for (int j = 0; j < mtx->num_columns; j++) {
                int k = i*mtx->num_columns+j;
                int l = (rowperm[i]-1)*mtx->num_columns + j;
                dst[k] = src[l];
            }
        }
    } else if (colperm) {
        for (int i = 0; i < mtx->num_rows; i++) {
            for (int j = 0; j < mtx->num_columns; j++) {
                int k = i*mtx->num_columns+j;
                int l = i*mtx->num_columns + colperm[j]-1;
                dst[k] = src[l];
            }
        }
    }

    mtx_free(&orig);
    return MTX_SUCCESS;
}

static int mtx_matrix_array_complex_permute(
    struct mtx * mtx,
    const int * rowperm,
    const int * colperm)
{
    struct mtx orig;
    int err = mtx_copy(&orig, mtx);
    if (err)
        return err;

    const float * src = (const float *) orig.data;
    float * dst = (float *) mtx->data;
    if (rowperm && colperm) {
        for (int i = 0; i < mtx->num_rows; i++) {
            for (int j = 0; j < mtx->num_columns; j++) {
                int k = i*mtx->num_columns+j;
                int l = (rowperm[i]-1)*mtx->num_columns + colperm[j]-1;
                dst[2*k+0] = src[2*l+0];
                dst[2*k+1] = src[2*l+1];
            }
        }
    } else if (rowperm) {
        for (int i = 0; i < mtx->num_rows; i++) {
            for (int j = 0; j < mtx->num_columns; j++) {
                int k = i*mtx->num_columns+j;
                int l = (rowperm[i]-1)*mtx->num_columns + j;
                dst[2*k+0] = src[2*l+0];
                dst[2*k+1] = src[2*l+1];
            }
        }
    } else if (colperm) {
        for (int i = 0; i < mtx->num_rows; i++) {
            for (int j = 0; j < mtx->num_columns; j++) {
                int k = i*mtx->num_columns+j;
                int l = i*mtx->num_columns + colperm[j]-1;
                dst[2*k+0] = src[2*l+0];
                dst[2*k+1] = src[2*l+1];
            }
        }
    }

    mtx_free(&orig);
    return MTX_SUCCESS;
}

static int mtx_matrix_array_integer_permute(
    struct mtx * mtx,
    const int * rowperm,
    const int * colperm)
{
    struct mtx orig;
    int err = mtx_copy(&orig, mtx);
    if (err)
        return err;

    const int * src = (const int *) orig.data;
    int * dst = (int *) mtx->data;
    if (rowperm && colperm) {
        for (int i = 0; i < mtx->num_rows; i++) {
            for (int j = 0; j < mtx->num_columns; j++) {
                int k = i*mtx->num_columns+j;
                int l = (rowperm[i]-1)*mtx->num_columns + colperm[j]-1;
                dst[k] = src[l];
            }
        }
    } else if (rowperm) {
        for (int i = 0; i < mtx->num_rows; i++) {
            for (int j = 0; j < mtx->num_columns; j++) {
                int k = i*mtx->num_columns+j;
                int l = (rowperm[i]-1)*mtx->num_columns + j;
                dst[k] = src[l];
            }
        }
    } else if (colperm) {
        for (int i = 0; i < mtx->num_rows; i++) {
            for (int j = 0; j < mtx->num_columns; j++) {
                int k = i*mtx->num_columns+j;
                int l = i*mtx->num_columns + colperm[j]-1;
                dst[k] = src[l];
            }
        }
    }

    mtx_free(&orig);
    return MTX_SUCCESS;
}

/**
 * `mtx_matrix_array_permute()' permutes the elements of a matrix
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
int mtx_matrix_array_permute(
    struct mtx * mtx,
    const int * row_permutation,
    const int * column_permutation)
{
    if (mtx->field == mtx_real) {
        return mtx_matrix_array_real_permute(
            mtx, row_permutation, column_permutation);
    } else if (mtx->field == mtx_double) {
        return mtx_matrix_array_double_permute(
            mtx, row_permutation, column_permutation);
    } else if (mtx->field == mtx_complex) {
        return mtx_matrix_array_complex_permute(
            mtx, row_permutation, column_permutation);
    } else if (mtx->field == mtx_integer) {
        return mtx_matrix_array_integer_permute(
            mtx, row_permutation, column_permutation);
    } else {
        return MTX_ERR_INVALID_MTX_FIELD;
    }
    return MTX_SUCCESS;
}
