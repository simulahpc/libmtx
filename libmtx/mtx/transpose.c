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
 * Transpose and complex conjugate of matrices and vectors.
 */

#include <libmtx/mtx/transpose.h>

#include <libmtx/error.h>
#include <libmtx/matrix/coordinate.h>
#include <libmtx/matrix/coordinate/data.h>
#include <libmtx/mtx/mtx.h>
#include <libmtx/mtx/header.h>

#include <errno.h>

/**
 * `mtx_transposition_str()' is a string representing the
 * transposition type.
 */
const char * mtx_transposition_str(
    enum mtx_transposition transposition)
{
    switch (transposition) {
    case mtx_nontransposed: return "nontransposed";
    case mtx_transposed: return "transposed";
    case mtx_conjugated: return "conjugated";
    case mtx_conjugate_transposed: return "conjugate-transposed";
    default: return "unknown";
    }
}

/**
 * `mtx_transpose()' transposes a matrix or vector.
 */
int mtx_transpose(
    struct mtx * mtx)
{
    int err;
    if (mtx->object != mtx_matrix)
        return MTX_ERR_INVALID_MTX_OBJECT;

    if (mtx->format == mtx_array) {
        /* TODO: Implement dense matrix transpose. */
        errno = ENOTSUP;
        return MTX_ERR_ERRNO;
    } else if (mtx->format == mtx_coordinate) {
        struct mtx_matrix_coordinate_data * matrix_coordinate =
            &mtx->storage.matrix_coordinate;
        err = mtx_matrix_coordinate_data_transpose(matrix_coordinate);
        if (err)
            return err;
        int num_rows = mtx->num_rows;
        mtx->num_rows = mtx->num_columns;
        mtx->num_columns = num_rows;
    } else {
        return MTX_ERR_INVALID_MTX_FORMAT;
    }
    return MTX_SUCCESS;
}
