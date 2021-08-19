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
 * Extracting submatrices from dense and sparse matrices in the Matrix
 * Market format.
 */

#include <libmtx/error.h>
#include <libmtx/util/index_set.h>
#include <libmtx/matrix/coordinate.h>
#include <libmtx/mtx/mtx.h>
#include <libmtx/mtx/submatrix.h>

#include <errno.h>

#include <stdlib.h>
#include <string.h>

/**
 * `mtx_matrix_submatrix()` obtains a submatrix consisting of the
 * given rows and columns.
 */
int mtx_matrix_submatrix(
    struct mtx * submtx,
    const struct mtx * mtx,
    const struct mtx_index_set * rows,
    const struct mtx_index_set * columns)
{
    int err;
    if (mtx->object != mtx_matrix)
        return MTX_ERR_INVALID_MTX_OBJECT;
   if (mtx->format == mtx_array)
       return MTX_ERR_INVALID_MTX_FORMAT;
     if (mtx->symmetry != mtx_general) {
        errno = ENOTSUP;
        return MTX_ERR_ERRNO;
    }

    /* Copy matrix header info. */
    submtx->object = mtx->object;
    submtx->format = mtx->format;
    submtx->field = mtx->field;
    submtx->symmetry = mtx->symmetry;
    submtx->num_comment_lines = 0;
    submtx->comment_lines = NULL;
    err = mtx_set_comment_lines(
        submtx, mtx->num_comment_lines,
        (const char **) mtx->comment_lines);
    if (err)
        return err;

    /* Copy size information. */
    submtx->num_rows = mtx->num_rows;
    submtx->num_columns = mtx->num_columns;

    struct mtx_matrix_coordinate_data * submtx_matrix_coordinate =
        &submtx->storage.matrix_coordinate;
    const struct mtx_matrix_coordinate_data * mtx_matrix_coordinate =
        &mtx->storage.matrix_coordinate;
    err = mtx_matrix_coordinate_data_submatrix(
        submtx_matrix_coordinate,
        mtx_matrix_coordinate,
        rows, columns);
    if (err) {
        for (int j = 0; j < submtx->num_comment_lines; j++)
            free(submtx->comment_lines[j]);
        free(submtx->comment_lines);
        return err;
    }
    submtx->num_nonzeros = submtx_matrix_coordinate->size;
    return MTX_SUCCESS;
}
