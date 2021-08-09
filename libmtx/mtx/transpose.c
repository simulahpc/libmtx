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
 * Transpose and complex conjugate of matrices and vectors.
 */

#include <libmtx/mtx/transpose.h>

#include <libmtx/error.h>
#include <libmtx/matrix/coordinate/coordinate.h>
#include <libmtx/mtx.h>
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
 * `mtx_matrix_coordinate_transpose()` transposes a square sparse
 * matrix.
 */
int mtx_matrix_coordinate_transpose(
    struct mtx * mtx)
{
    if (mtx->object != mtx_matrix)
        return MTX_ERR_INVALID_MTX_OBJECT;
    if (mtx->format != mtx_coordinate)
        return MTX_ERR_INVALID_MTX_FORMAT;
    if (mtx->num_rows != mtx->num_columns)
        return MTX_ERR_INVALID_MTX_SIZE;

    if (mtx->symmetry == mtx_symmetric) {
        return MTX_SUCCESS;
    } else if (mtx->symmetry == mtx_skew_symmetric) {
        /* TODO: Implement transpose for skew-symmetric matrices. */
        errno = ENOTSUP;
        return MTX_ERR_ERRNO;
    } else if (mtx->symmetry == mtx_hermitian) {
        errno = ENOTSUP;
        return MTX_ERR_ERRNO;
    } else if (mtx->symmetry == mtx_general) {
        if (mtx->field == mtx_real) {
            struct mtx_matrix_coordinate_real * data =
                (struct mtx_matrix_coordinate_real *)
                mtx->data;
            for (int64_t k = 0; k < mtx->size; k++) {
                int i = data[k].i;
                data[k].i = data[k].j;
                data[k].j = i;
            }
        } else if (mtx->field == mtx_double) {
            struct mtx_matrix_coordinate_double * data =
                (struct mtx_matrix_coordinate_double *)
                mtx->data;
            for (int64_t k = 0; k < mtx->size; k++) {
                int i = data[k].i;
                data[k].i = data[k].j;
                data[k].j = i;
            }
        } else if (mtx->field == mtx_complex) {
            struct mtx_matrix_coordinate_complex * data =
                (struct mtx_matrix_coordinate_complex *)
                mtx->data;
            for (int64_t k = 0; k < mtx->size; k++) {
                int i = data[k].i;
                data[k].i = data[k].j;
                data[k].j = i;
            }
        } else if (mtx->field == mtx_integer) {
            struct mtx_matrix_coordinate_integer * data =
                (struct mtx_matrix_coordinate_integer *)
                mtx->data;
            for (int64_t k = 0; k < mtx->size; k++) {
                int i = data[k].i;
                data[k].i = data[k].j;
                data[k].j = i;
            }
        } else if (mtx->field == mtx_pattern) {
            struct mtx_matrix_coordinate_pattern * data =
                (struct mtx_matrix_coordinate_pattern *)
                mtx->data;
            for (int64_t k = 0; k < mtx->size; k++) {
                int i = data[k].i;
                data[k].i = data[k].j;
                data[k].j = i;
            }
        } else {
            return MTX_ERR_INVALID_MTX_FIELD;
        }
    }

    return MTX_SUCCESS;
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
        err = mtx_matrix_coordinate_transpose(mtx);
        if (err)
            return err;
    } else {
        return MTX_ERR_INVALID_MTX_FORMAT;
    }
    return MTX_SUCCESS;
}
