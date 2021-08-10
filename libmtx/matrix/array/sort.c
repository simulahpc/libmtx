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
 * Sorting dense matrices in array format.
 */

#include <libmtx/matrix/array/sort.h>

#include <libmtx/error.h>
#include <libmtx/mtx/mtx.h>
#include <libmtx/mtx/sort.h>

#include <errno.h>

struct mtx;

/**
 * `mtx_matrix_array_sort()' sorts the entries of a matrix in array
 * format in a given order.
 */
int mtx_matrix_array_sort(
    struct mtx * mtx,
    enum mtx_sorting sorting)
{
    int err;
    if (mtx->object != mtx_matrix)
        return MTX_ERR_INVALID_MTX_OBJECT;
    if (mtx->format != mtx_array)
        return MTX_ERR_INVALID_MTX_FORMAT;

    if (mtx->sorting == sorting)
        return MTX_SUCCESS;

    /* TODO: Implement sorting for matrices in array format. */
    errno = ENOTSUP;
    return MTX_ERR_ERRNO;
}