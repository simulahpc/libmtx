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
 * Data structures for representing objects in Matrix Market format.
 */

#include <matrixmarket/error.h>
#include <matrixmarket/header.h>
#include <matrixmarket/mtx.h>

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
    for (int i = 0; i < dst->num_comment_lines; i++)
        dst->comment_lines[i] = strdup(src->comment_lines[i]);
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
