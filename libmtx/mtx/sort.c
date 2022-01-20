/* This file is part of Libmtx.
 *
 * Copyright (C) 2021 James D. Trotter
 *
 * Libmtx is free software: you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * Libmtx is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Libmtx.  If not, see
 * <https://www.gnu.org/licenses/>.
 *
 * Authors: James D. Trotter <james@simula.no>
 * Last modified: 2021-08-09
 *
 * Sorting matrices and vectors.
 */

#include <libmtx/mtx/sort.h>

#include <libmtx/error.h>
#include <libmtx/matrix/array/sort.h>
#include <libmtx/matrix/coordinate/sort.h>
#include <libmtx/matrix/coordinate/data.h>
#include <libmtx/mtx/header.h>
#include <libmtx/mtx/mtx.h>

#include <errno.h>

#include <stdlib.h>

/**
 * `mtx_sorting_str()` is a string representing the sorting type.
 */
const char * mtx_sorting_str(
    enum mtx_sorting sorting)
{
    switch (sorting) {
    case mtx_unsorted: return "unsorted";
    case mtx_row_major: return "row-major";
    case mtx_column_major: return "column-major";
    default: return "unknown";
    }
}

/**
 * `mtx_sort()' sorts matrix or vector nonzeros in a given order.
 */
int mtx_sort(
    struct mtx * mtx,
    enum mtx_sorting sorting)
{
    if (mtx->object == mtx_matrix) {
        if (mtx->format == mtx_array) {
            return mtx_matrix_array_sort(mtx, sorting);
        } else if (mtx->format == mtx_coordinate) {
            struct mtx_matrix_coordinate_data * matrix_coordinate =
                &mtx->storage.matrix_coordinate;
            return mtx_matrix_coordinate_data_sort(
                matrix_coordinate, sorting);
        } else {
            return MTX_ERR_INVALID_MTX_FORMAT;
        }
    } else if (mtx->object == mtx_vector) {
        if (mtx->format == mtx_array) {
            /* TODO: Implement sorting for vectors in array format. */
            errno = ENOTSUP;
            return MTX_ERR_ERRNO;
        } else if (mtx->format == mtx_coordinate) {
            /* TODO: Implement sorting for vectors in coordinate format. */
            errno = ENOTSUP;
            return MTX_ERR_ERRNO;
        } else {
            return MTX_ERR_INVALID_MTX_FORMAT;
        }
    } else {
        return MTX_ERR_INVALID_MTX_OBJECT;
    }
    return MTX_SUCCESS;
}
