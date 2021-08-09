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
 * Matrix transpose.
 */

#include <libmtx/error.h>
#include <libmtx/mtx.h>
#include <libmtx/header.h>
#include <libmtx/matrix_coordinate.h>

#include <errno.h>

/**
 * `mtx_matrix_coordinate_transpose()` transposes a square sparse
 * matrix.
 */
int mtx_matrix_coordinate_transpose(
    struct mtx * matrix)
{
    if (matrix->object != mtx_matrix)
        return MTX_ERR_INVALID_MTX_OBJECT;
    if (matrix->format != mtx_coordinate)
        return MTX_ERR_INVALID_MTX_FORMAT;
    if (matrix->num_rows != matrix->num_columns)
        return MTX_ERR_INVALID_MTX_SIZE;

    if (matrix->symmetry == mtx_symmetric) {
        return MTX_SUCCESS;
    } else if (matrix->symmetry == mtx_skew_symmetric) {
        /* TODO: Implement transpose for skew-symmetric matrices. */
        errno = ENOTSUP;
        return MTX_ERR_ERRNO;
    } else if (matrix->symmetry == mtx_hermitian) {
        errno = ENOTSUP;
        return MTX_ERR_ERRNO;
    } else if (matrix->symmetry == mtx_general) {
        if (matrix->field == mtx_real) {
            struct mtx_matrix_coordinate_real * data =
                (struct mtx_matrix_coordinate_real *)
                matrix->data;
            for (int64_t k = 0; k < matrix->size; k++) {
                int i = data[k].i;
                data[k].i = data[k].j;
                data[k].j = i;
            }
        } else if (matrix->field == mtx_double) {
            struct mtx_matrix_coordinate_double * data =
                (struct mtx_matrix_coordinate_double *)
                matrix->data;
            for (int64_t k = 0; k < matrix->size; k++) {
                int i = data[k].i;
                data[k].i = data[k].j;
                data[k].j = i;
            }
        } else if (matrix->field == mtx_complex) {
            struct mtx_matrix_coordinate_complex * data =
                (struct mtx_matrix_coordinate_complex *)
                matrix->data;
            for (int64_t k = 0; k < matrix->size; k++) {
                int i = data[k].i;
                data[k].i = data[k].j;
                data[k].j = i;
            }
        } else if (matrix->field == mtx_integer) {
            struct mtx_matrix_coordinate_integer * data =
                (struct mtx_matrix_coordinate_integer *)
                matrix->data;
            for (int64_t k = 0; k < matrix->size; k++) {
                int i = data[k].i;
                data[k].i = data[k].j;
                data[k].j = i;
            }
        } else if (matrix->field == mtx_pattern) {
            struct mtx_matrix_coordinate_pattern * data =
                (struct mtx_matrix_coordinate_pattern *)
                matrix->data;
            for (int64_t k = 0; k < matrix->size; k++) {
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
 * `mtx_matrix_transpose()` transposes a square matrix.
 */
int mtx_matrix_transpose(
    struct mtx * matrix)
{
    int err;
    if (matrix->object != mtx_matrix)
        return MTX_ERR_INVALID_MTX_OBJECT;

    if (matrix->format == mtx_array) {
        /* TODO: Implement dense matrix transpose. */
        errno = ENOTSUP;
        return MTX_ERR_ERRNO;
    } else if (matrix->format == mtx_coordinate) {
        err = mtx_matrix_coordinate_transpose(matrix);
        if (err)
            return err;
    } else {
        return MTX_ERR_INVALID_MTX_FORMAT;
    }
    return MTX_SUCCESS;
}
