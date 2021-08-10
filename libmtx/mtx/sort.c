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
 * Sorting matrices and vectors.
 */

#include <libmtx/mtx/sort.h>

#include <libmtx/error.h>
#include <libmtx/matrix/array/sort.h>
#include <libmtx/matrix/coordinate/sort.h>
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
 * `mtx_sort_matrix()' sorts matrix nonzeros in a given order.
 */
int mtx_sort_matrix(
    struct mtx * mtx,
    enum mtx_sorting sorting)
{
    int err;
    if (mtx->object != mtx_matrix)
        return MTX_ERR_INVALID_MTX_OBJECT;

    if (mtx->format == mtx_array) {
        return mtx_matrix_array_sort(mtx, sorting);
    } else if (mtx->format == mtx_coordinate) {
        return mtx_matrix_coordinate_sort(mtx, sorting);
    } else {
        return MTX_ERR_INVALID_MTX_FORMAT;
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_sort_vector_coordinate()' sorts nonzeros in a given order
 * for vectors in coordinate format.
 */
int mtx_sort_vector_coordinate(
    struct mtx * mtx,
    enum mtx_sorting sorting)
{
    int err;
    if (mtx->object != mtx_vector)
        return MTX_ERR_INVALID_MTX_OBJECT;
    if (mtx->format != mtx_coordinate)
        return MTX_ERR_INVALID_MTX_FORMAT;

    /* TODO: Implement sorting for sparse vectors. */
    errno = ENOTSUP;
    return MTX_ERR_ERRNO;
}

/**
 * `mtx_sort_vector()' sorts vector nonzeros in a given order.
 */
int mtx_sort_vector(
    struct mtx * mtx,
    enum mtx_sorting sorting)
{
    int err;
    if (mtx->object != mtx_vector)
        return MTX_ERR_INVALID_MTX_OBJECT;

    if (mtx->format == mtx_array) {
        if (mtx->sorting == sorting)
            return MTX_SUCCESS;

        /* TODO: Implement sorting for vectors in array format. */
        errno = ENOTSUP;
        return MTX_ERR_ERRNO;
    } else if (mtx->format == mtx_coordinate) {
        return mtx_sort_vector_coordinate(mtx, sorting);
    } else {
        return MTX_ERR_INVALID_MTX_FORMAT;
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_sort()' sorts matrix or vector nonzeros in a given order.
 */
int mtx_sort(
    struct mtx * mtx,
    enum mtx_sorting sorting)
{
    if (mtx->object == mtx_matrix) {
        return mtx_sort_matrix(mtx, sorting);
    } else if (mtx->object == mtx_vector) {
        return mtx_sort_vector(mtx, sorting);
    } else {
        return MTX_ERR_INVALID_MTX_OBJECT;
    }
    return MTX_SUCCESS;
}
