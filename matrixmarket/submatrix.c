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
 * Extracting submatrices from dense and sparse matrices in the Matrix
 * Market format.
 */

#include <matrixmarket/error.h>
#include <matrixmarket/index_set.h>
#include <matrixmarket/matrix.h>
#include <matrixmarket/matrix_coordinate.h>
#include <matrixmarket/mtx.h>

#include <errno.h>

#include <stdlib.h>
#include <string.h>

/**
 * `mtx_matrix_submatrix()` obtains a submatrix consisting of the
 * given rows and columns.
 */
int mtx_matrix_submatrix(
    const struct mtx * mtx,
    const struct mtx_index_set * rows,
    const struct mtx_index_set * columns,
    struct mtx * submtx)
{
    int err;
    if (mtx->object != mtx_matrix) {
        errno = EINVAL;
        return MTX_ERR_ERRNO;
    }

    if (mtx->symmetry != mtx_general) {
        errno = ENOTSUP;
        return MTX_ERR_ERRNO;
    }

    /* Copy matrix header info. */
    submtx->object = mtx->object;
    submtx->format = mtx->format;
    submtx->field = mtx->field;
    submtx->symmetry = mtx->symmetry;
    submtx->sorting = mtx->sorting;
    submtx->ordering = mtx->ordering;
    submtx->assembly = mtx->assembly;

    /* Allocate storage for comment lines. */
    submtx->num_comment_lines = mtx->num_comment_lines;
    submtx->comment_lines = malloc(submtx->num_comment_lines * sizeof(char *));
    if (!submtx->comment_lines)
        return MTX_ERR_ERRNO;

    /* Copy comment lines. */
    for (int i = 0; i < submtx->num_comment_lines; i++) {
        submtx->comment_lines[i] = strdup(mtx->comment_lines[i]);
        if (!submtx->comment_lines[i]) {
            for (int j = i-1; j >= 0; j--)
                free(submtx->comment_lines[j]);
            free(submtx->comment_lines);
            return MTX_ERR_ERRNO;
        }
    }

    /* Copy size information. */
    submtx->num_rows = mtx->num_rows;
    submtx->num_columns = mtx->num_columns;

    /* Count the number of nonzeros in the submatrix. */
    int64_t size;
    if (mtx->format == mtx_array) {
        int num_submtx_rows;
        err = mtx_index_set_size(rows, &num_submtx_rows);
        int num_submtx_columns;
        err = mtx_index_set_size(columns, &num_submtx_columns);
        size = num_submtx_rows * num_submtx_columns;
    } else if (mtx->format == mtx_coordinate) {
        if (mtx->field == mtx_real) {
            const struct mtx_matrix_coordinate_real * nonzeros =
                (const struct mtx_matrix_coordinate_real *) mtx->data;
            size = 0;
            for (int k = 0; k < mtx->size; k++) {
                bool has_row = mtx_index_set_contains(rows, nonzeros[k].i);
                bool has_column = mtx_index_set_contains(columns, nonzeros[k].j);
                if (has_row && has_column)
                    size++;
            }
        } else if (mtx->field == mtx_double ||
                   mtx->field == mtx_complex ||
                   mtx->field == mtx_integer ||
                   mtx->field == mtx_pattern)
        {
            for (int j = 0; j < submtx->num_comment_lines; j++)
                free(submtx->comment_lines[j]);
            free(submtx->comment_lines);
            errno = ENOTSUP;
            return MTX_ERR_ERRNO;
        } else {
            for (int j = 0; j < submtx->num_comment_lines; j++)
                free(submtx->comment_lines[j]);
            free(submtx->comment_lines);
            errno = EINVAL;
            return MTX_ERR_ERRNO;
        }

    } else {
        for (int j = 0; j < submtx->num_comment_lines; j++)
            free(submtx->comment_lines[j]);
        free(submtx->comment_lines);
        errno = EINVAL;
        return MTX_ERR_ERRNO;
    }

    submtx->size = size;
    submtx->nonzero_size = mtx->nonzero_size;

    /* Allocate storage for data. */
    submtx->data = malloc(submtx->size * submtx->nonzero_size);
    if (!submtx->data) {
        for (int j = 0; j < submtx->num_comment_lines; j++)
            free(submtx->comment_lines[j]);
        free(submtx->comment_lines);
        return MTX_ERR_ERRNO;
    }

    /* Copy nonzeros that belong to the submatrix. */
    if (mtx->format == mtx_array) {
        /* TODO: Add support for dense matrices or vectors. */
        free(submtx->data);
        for (int j = 0; j < submtx->num_comment_lines; j++)
            free(submtx->comment_lines[j]);
        free(submtx->comment_lines);
        errno = ENOTSUP;
        return MTX_ERR_ERRNO;

    } else if (mtx->format == mtx_coordinate) {
        if (mtx->field == mtx_real) {
            const struct mtx_matrix_coordinate_real * nonzeros =
                (const struct mtx_matrix_coordinate_real *) mtx->data;
            struct mtx_matrix_coordinate_real * submtx_nonzeros =
                (struct mtx_matrix_coordinate_real *) submtx->data;
            int64_t l = 0;
            for (int k = 0; k < mtx->size; k++) {
                bool has_row = mtx_index_set_contains(rows, nonzeros[k].i);
                bool has_column = mtx_index_set_contains(columns, nonzeros[k].j);
                if (has_row && has_column) {
                    submtx_nonzeros[l] = nonzeros[k];
                    l++;
                }
            }
        } else if (mtx->field == mtx_double ||
                   mtx->field == mtx_complex ||
                   mtx->field == mtx_integer ||
                   mtx->field == mtx_pattern)
        {
            free(submtx->data);
            for (int j = 0; j < submtx->num_comment_lines; j++)
                free(submtx->comment_lines[j]);
            free(submtx->comment_lines);
            errno = ENOTSUP;
            return MTX_ERR_ERRNO;
        } else {
            free(submtx->data);
            for (int j = 0; j < submtx->num_comment_lines; j++)
                free(submtx->comment_lines[j]);
            free(submtx->comment_lines);
            errno = EINVAL;
            return MTX_ERR_ERRNO;
        }

    } else {
        free(submtx->data);
        for (int j = 0; j < submtx->num_comment_lines; j++)
            free(submtx->comment_lines[j]);
        free(submtx->comment_lines);
        errno = EINVAL;
        return MTX_ERR_ERRNO;
    }

    /* Calculate the number of nonzeros. */
    err = mtx_matrix_num_nonzeros(
        submtx->object, submtx->format, submtx->field, submtx->symmetry,
        submtx->num_rows, submtx->num_columns, submtx->size, submtx->data,
        &submtx->num_nonzeros);
    if (err) {
        for (int i = 0; i < mtx->num_comment_lines; i++)
            free(mtx->comment_lines[i]);
        free(mtx->comment_lines);
        return err;
    }

    return MTX_SUCCESS;
}
