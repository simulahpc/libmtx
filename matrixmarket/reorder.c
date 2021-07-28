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
 * Last modified: 2021-07-28
 *
 * Reordering the rows and columns of sparse matrices.
 */

#include <matrixmarket/error.h>
#include <matrixmarket/mtx.h>
#include <matrixmarket/header.h>
#include <matrixmarket/matrix_coordinate.h>
#include <matrixmarket/vector_array.h>
#include <matrixmarket/vector_coordinate.h>

#include <errno.h>

#include <stdlib.h>

static int mtx_permute_vector_array_real(
    struct mtx * mtx,
    const int * permutation)
{
    struct mtx orig;
    int err = mtx_copy(&orig, mtx);
    if (err)
        return err;

    const float * src = (const float *) orig.data;
    float * dst = (float *) mtx->data;
    for (int i = 0; i < mtx->size; i++)
        dst[i] = src[permutation[i]-1];
    mtx_free(&orig);
    return MTX_SUCCESS;
}

static int mtx_permute_vector_array_double(
    struct mtx * mtx,
    const int * permutation)
{
    struct mtx orig;
    int err = mtx_copy(&orig, mtx);
    if (err)
        return err;

    const double * src = (const double *) orig.data;
    double * dst = (double *) mtx->data;
    for (int i = 0; i < mtx->size; i++)
        dst[i] = src[permutation[i]-1];
    mtx_free(&orig);
    return MTX_SUCCESS;
}

static int mtx_permute_vector_array_complex(
    struct mtx * mtx,
    const int * permutation)
{
    struct mtx orig;
    int err = mtx_copy(&orig, mtx);
    if (err)
        return err;

    const float * src = (const float *) orig.data;
    float * dst = (float *) mtx->data;
    for (int i = 0; i < mtx->size; i++) {
        dst[2*i+0] = src[2*(permutation[i]-1)+0];
        dst[2*i+1] = src[2*(permutation[i]-1)+1];
    }
    mtx_free(&orig);
    return MTX_SUCCESS;
}

static int mtx_permute_vector_array_integer(
    struct mtx * mtx,
    const int * permutation)
{
    struct mtx orig;
    int err = mtx_copy(&orig, mtx);
    if (err)
        return err;

    const int * src = (const int *) orig.data;
    int * dst = (int *) mtx->data;
    for (int i = 0; i < mtx->size; i++)
        dst[i] = src[permutation[i]-1];
    mtx_free(&orig);
    return MTX_SUCCESS;
}

static int mtx_permute_vector_array(
    struct mtx * mtx,
    const int * permutation)
{
    if (mtx->field == mtx_real) {
        return mtx_permute_vector_array_real(mtx, permutation);
    } else if (mtx->field == mtx_double) {
        return mtx_permute_vector_array_double(mtx, permutation);
    } else if (mtx->field == mtx_complex) {
        return mtx_permute_vector_array_complex(mtx, permutation);
    } else if (mtx->field == mtx_integer) {
        return mtx_permute_vector_array_integer(mtx, permutation);
    } else {
        return MTX_ERR_INVALID_MTX_FIELD;
    }
    return MTX_SUCCESS;
}

static int mtx_permute_vector_coordinate_real(
    struct mtx * mtx,
    const int * permutation)
{
    struct mtx_vector_coordinate_real * data =
        (struct mtx_vector_coordinate_real *) mtx->data;
    for (int i = 0; i < mtx->size; i++)
        data[i].i = permutation[data[i].i-1];
    mtx->sorting = mtx_unsorted;
    return MTX_SUCCESS;
}

static int mtx_permute_vector_coordinate_double(
    struct mtx * mtx,
    const int * permutation)
{
    struct mtx_vector_coordinate_double * data =
        (struct mtx_vector_coordinate_double *) mtx->data;
    for (int i = 0; i < mtx->size; i++)
        data[i].i = permutation[data[i].i-1];
    mtx->sorting = mtx_unsorted;
    return MTX_SUCCESS;
}

static int mtx_permute_vector_coordinate_complex(
    struct mtx * mtx,
    const int * permutation)
{
    struct mtx_vector_coordinate_complex * data =
        (struct mtx_vector_coordinate_complex *) mtx->data;
    for (int i = 0; i < mtx->size; i++)
        data[i].i = permutation[data[i].i-1];
    mtx->sorting = mtx_unsorted;
    return MTX_SUCCESS;
}

static int mtx_permute_vector_coordinate_integer(
    struct mtx * mtx,
    const int * permutation)
{
    struct mtx_vector_coordinate_integer * data =
        (struct mtx_vector_coordinate_integer *) mtx->data;
    for (int i = 0; i < mtx->size; i++)
        data[i].i = permutation[data[i].i-1];
    mtx->sorting = mtx_unsorted;
    return MTX_SUCCESS;
}

static int mtx_permute_vector_coordinate_pattern(
    struct mtx * mtx,
    const int * permutation)
{
    struct mtx_vector_coordinate_pattern * data =
        (struct mtx_vector_coordinate_pattern *) mtx->data;
    for (int i = 0; i < mtx->size; i++)
        data[i].i = permutation[data[i].i-1];
    mtx->sorting = mtx_unsorted;
    return MTX_SUCCESS;
}

static int mtx_permute_vector_coordinate(
    struct mtx * mtx,
    const int * permutation)
{
    if (mtx->field == mtx_real) {
        return mtx_permute_vector_coordinate_real(mtx, permutation);
    } else if (mtx->field == mtx_double) {
        return mtx_permute_vector_coordinate_double(mtx, permutation);
    } else if (mtx->field == mtx_complex) {
        return mtx_permute_vector_coordinate_complex(mtx, permutation);
    } else if (mtx->field == mtx_integer) {
        return mtx_permute_vector_coordinate_integer(mtx, permutation);
    } else {
        return MTX_ERR_INVALID_MTX_FIELD;
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_permute_vector()' permutes the elements of a vector based on a
 * given permutation.
 *
 * The array `permutation' should be a permutation of the integers
 * `1,2,...,mtx->num_rows'. The element at position `i' in the
 * permuted vector is then equal to the element at the position
 * `permutation[i-1]' in the original vector, for
 * `i=1,2,...,mtx->num_rows'.
 */
int mtx_permute_vector(
    struct mtx * mtx,
    const int * permutation)
{
    if (mtx->object != mtx_vector)
        return MTX_ERR_INVALID_MTX_OBJECT;

    if (mtx->format == mtx_array) {
        return mtx_permute_vector_array(mtx, permutation);
    } else if (mtx->format == mtx_coordinate) {
        return mtx_permute_vector_coordinate(mtx, permutation);
    } else {
        return MTX_ERR_INVALID_MTX_FORMAT;
    }
    return MTX_SUCCESS;
}

static int mtx_permute_matrix_array_real(
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

static int mtx_permute_matrix_array_double(
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

static int mtx_permute_matrix_array_complex(
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

static int mtx_permute_matrix_array_integer(
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

static int mtx_permute_matrix_array(
    struct mtx * mtx,
    const int * row_permutation,
    const int * column_permutation)
{
    if (mtx->field == mtx_real) {
        return mtx_permute_matrix_array_real(
            mtx, row_permutation, column_permutation);
    } else if (mtx->field == mtx_double) {
        return mtx_permute_matrix_array_double(
            mtx, row_permutation, column_permutation);
    } else if (mtx->field == mtx_complex) {
        return mtx_permute_matrix_array_complex(
            mtx, row_permutation, column_permutation);
    } else if (mtx->field == mtx_integer) {
        return mtx_permute_matrix_array_integer(
            mtx, row_permutation, column_permutation);
    } else {
        return MTX_ERR_INVALID_MTX_FIELD;
    }
    return MTX_SUCCESS;
}

static int mtx_permute_matrix_coordinate_real(
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

static int mtx_permute_matrix_coordinate_double(
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

static int mtx_permute_matrix_coordinate_complex(
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

static int mtx_permute_matrix_coordinate_integer(
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

static int mtx_permute_matrix_coordinate_pattern(
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

static int mtx_permute_matrix_coordinate(
    struct mtx * mtx,
    const int * row_permutation,
    const int * column_permutation)
{
    if (mtx->field == mtx_real) {
        return mtx_permute_matrix_coordinate_real(
            mtx, row_permutation, column_permutation);
    } else if (mtx->field == mtx_double) {
        return mtx_permute_matrix_coordinate_double(
            mtx, row_permutation, column_permutation);
    } else if (mtx->field == mtx_complex) {
        return mtx_permute_matrix_coordinate_complex(
            mtx, row_permutation, column_permutation);
    } else if (mtx->field == mtx_integer) {
        return mtx_permute_matrix_coordinate_integer(
            mtx, row_permutation, column_permutation);
    } else if (mtx->field == mtx_pattern) {
        return mtx_permute_matrix_coordinate_pattern(
            mtx, row_permutation, column_permutation);
    } else {
        return MTX_ERR_INVALID_MTX_FIELD;
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_permute_matrix()' permutes the elements of a matrix based on a
 * given permutation.
 *
 * The array `row_permutation' should be a permutation of the integers
 * `1,2,...,mtx->num_rows', and the array `column_permutation' should
 * be a permutation of the integers `1,2,...,mtx->num_columns'. The
 * elements belonging to row `i' and column `j' in the permuted matrix
 * are then equal to the elements in row `row_permutation[i-1]' and
 * column `column_permutation[j-1]' in the original matrix, for
 * `i=1,2,...,mtx->num_rows' and `j=1,2,...,mtx->num_columns'.
 */
int mtx_permute_matrix(
    struct mtx * mtx,
    const int * row_permutation,
    const int * column_permutation)
{
    if (mtx->object != mtx_matrix)
        return MTX_ERR_INVALID_MTX_OBJECT;

    if (!row_permutation && !column_permutation)
        return MTX_SUCCESS;

    if (mtx->format == mtx_array) {
        return mtx_permute_matrix_array(
            mtx, row_permutation, column_permutation);
    } else if (mtx->format == mtx_coordinate) {
        return mtx_permute_matrix_coordinate(
            mtx, row_permutation, column_permutation);
    } else {
        return MTX_ERR_INVALID_MTX_FORMAT;
    }
    return MTX_SUCCESS;
}
