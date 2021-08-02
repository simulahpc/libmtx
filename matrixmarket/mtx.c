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
 * Last modified: 2021-08-02
 *
 * Data structures for representing objects in Matrix Market format.
 */

#include <matrixmarket/error.h>
#include <matrixmarket/header.h>
#include <matrixmarket/mtx.h>
#include <matrixmarket/matrix.h>
#include <matrixmarket/vector.h>

#include <errno.h>

#include <stdlib.h>
#include <string.h>

/**
 * `mtx_free()` frees resources associated with a Matrix Market
 * object.
 */
void mtx_free(
    struct mtx * mtx)
{
    free(mtx->data);
    for (int i = 0; i < mtx->num_comment_lines; i++)
        free(mtx->comment_lines[i]);
    free(mtx->comment_lines);
}

/**
 * `mtx_copy()' copies a matrix or vector.
 */
int mtx_copy(
    struct mtx * dst,
    const struct mtx * src)
{
    dst->object = src->object;
    dst->format = src->format;
    dst->field = src->field;
    dst->symmetry = src->symmetry;
    dst->sorting = src->sorting;
    dst->ordering = src->ordering;
    dst->assembly = src->assembly;
    dst->num_comment_lines = src->num_comment_lines;
    dst->comment_lines = malloc(dst->num_comment_lines * sizeof(char *));
    if (!dst->comment_lines)
        return MTX_ERR_ERRNO;
    for (int i = 0; i < dst->num_comment_lines; i++) {
        dst->comment_lines[i] = strdup(src->comment_lines[i]);
        if (!dst->comment_lines[i]) {
            for (int j = i-1; j >= 0; j--)
                free(dst->comment_lines[j]);
            free(dst->comment_lines);
            return MTX_ERR_ERRNO;
        }
    }
    dst->num_rows = src->num_rows;
    dst->num_columns = src->num_columns;
    dst->num_nonzeros = src->num_nonzeros;
    dst->size = src->size;
    dst->nonzero_size = src->nonzero_size;
    dst->data = malloc(dst->size * dst->nonzero_size);
    if (!dst->data) {
        for (int i = 0; i < dst->num_comment_lines; i++)
            free(dst->comment_lines[i]);
        free(dst->comment_lines);
        return MTX_ERR_ERRNO;
    }
    memcpy(dst->data, src->data, dst->size * dst->nonzero_size);
    return MTX_SUCCESS;
}

/**
 * `mtx_set_comment_lines()' copies comment lines to a Matrix Market
 * object.
 *
 * Any storage associated with existing comment lines is freed.
 */
int mtx_set_comment_lines(
    struct mtx * mtx,
    int  num_comment_lines,
    const char ** comment_lines)
{
    /* Copy the given comment lines. */
    char ** comment_lines_copy = malloc(num_comment_lines * sizeof(char *));
    if (!comment_lines_copy)
        return MTX_ERR_ERRNO;
    for (int i = 0; i < num_comment_lines; i++) {
        comment_lines_copy[i] = strdup(comment_lines[i]);
        if (!comment_lines_copy[i]) {
            for (int j = i-1; j >= 0; j--)
                free(comment_lines_copy[j]);
            free(comment_lines_copy);
            return MTX_ERR_ERRNO;
        }
    }

    /* Free existing comment lines. */
    if (mtx->comment_lines) {
        for (int i = 0; i < mtx->num_comment_lines; i++)
            free(mtx->comment_lines[i]);
        free(mtx->comment_lines);
    }

    mtx->num_comment_lines = num_comment_lines;
    mtx->comment_lines = comment_lines_copy;
    return MTX_SUCCESS;
}

/**
 * `mtx_set_zero()' zeroes a matrix or vector.
 */
int mtx_set_zero(
    struct mtx * mtx)
{
    if (mtx->object == mtx_matrix) {
        return mtx_matrix_set_zero(mtx);
    } else if (mtx->object == mtx_vector) {
        return mtx_vector_set_zero(mtx);
    } else {
        return MTX_ERR_INVALID_MTX_OBJECT;
    }
    return MTX_SUCCESS;
}
