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
 * Last modified: 2021-07-28
 *
 * Data structures for representing objects in Matrix Market format.
 */

#include <matrixmarket/error.h>
#include <matrixmarket/header.h>
#include <matrixmarket/matrix_array.h>
#include <matrixmarket/matrix_coordinate.h>
#include <matrixmarket/mtx.h>

#include <errno.h>

#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * `mtx_matrix_row_index()` retrieves the row index for a given
 * nonzero of a matrix.
 */
int mtx_matrix_row_index(
    const struct mtx * mtx,
    int64_t k,
    int * row)
{
    if (mtx->object != mtx_matrix)
        return MTX_ERR_INVALID_MTX_OBJECT;
    if (k < 0 || k > mtx->size) {
        errno = EINVAL;
        return MTX_ERR_ERRNO;
    }

    if (mtx->format == mtx_array) {
        if (mtx->symmetry == mtx_general) {
            *row = k / mtx->num_columns;
        } else if (mtx->symmetry == mtx_symmetric ||
                   mtx->symmetry == mtx_hermitian)
        {
            /* Here we assume that the lower triangular part of the
             * matrix is stored. */
            for (int i = 0; i < mtx->num_rows; i++) {
                if (k < i+1) {
                    *row = i;
                    return MTX_SUCCESS;
                }
                k -= i+1;
            }
        } else if (mtx->symmetry == mtx_skew_symmetric) {
            /* Here we assume that the strict lower triangular part of
             * the matrix is stored. */
            for (int i = 0; i < mtx->num_rows; i++) {
                if (k < i) {
                    *row = i;
                    return MTX_SUCCESS;
                }
                k -= i;
            }
        } else {
            return MTX_ERR_INVALID_MTX_SYMMETRY;
        }

    } else if (mtx->format == mtx_coordinate) {
        if (mtx->field == mtx_real) {
            const struct mtx_matrix_coordinate_real * data =
                (const struct mtx_matrix_coordinate_real *) mtx->data;
            *row = data[k].i;
        } else if (mtx->field == mtx_double) {
            const struct mtx_matrix_coordinate_double * data =
                (const struct mtx_matrix_coordinate_double *) mtx->data;
            *row = data[k].i;
        } else if (mtx->field == mtx_complex) {
            const struct mtx_matrix_coordinate_complex * data =
                (const struct mtx_matrix_coordinate_complex *) mtx->data;
            *row = data[k].i;
        } else if (mtx->field == mtx_integer) {
            const struct mtx_matrix_coordinate_integer * data =
                (const struct mtx_matrix_coordinate_integer *) mtx->data;
            *row = data[k].i;
        } else if (mtx->field == mtx_pattern) {
            const struct mtx_matrix_coordinate_pattern * data =
                (const struct mtx_matrix_coordinate_pattern *) mtx->data;
            *row = data[k].i;
        } else {
            return MTX_ERR_INVALID_MTX_FIELD;
        }
    } else {
        return MTX_ERR_INVALID_MTX_FORMAT;
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_matrix_column_index()` retrieves the column index for a given
 * nonzero of a matrix.
 */
int mtx_matrix_column_index(
    const struct mtx * mtx,
    int64_t k,
    int * column)
{
    if (mtx->object != mtx_matrix)
        return MTX_ERR_INVALID_MTX_OBJECT;
    if (k < 0 || k > mtx->size) {
        errno = EINVAL;
        return MTX_ERR_ERRNO;
    }

    if (mtx->format == mtx_array) {
        if (mtx->symmetry == mtx_general) {
            *column = k % mtx->num_rows;
        } else if (mtx->symmetry == mtx_symmetric ||
                   mtx->symmetry == mtx_hermitian)
        {
            /* Here we assume that the lower triangular part of the
             * matrix is stored. */
            for (int i = 0; i < mtx->num_rows; i++) {
                if (k < i+1) {
                    *column = k;
                    return MTX_SUCCESS;
                }
                k -= i+1;
            }
        } else if (mtx->symmetry == mtx_skew_symmetric) {
            /* Here we assume that the strict lower triangular part of
             * the matrix is stored. */
            for (int i = 0; i < mtx->num_rows; i++) {
                if (k < i) {
                    *column = k;
                    return MTX_SUCCESS;
                }
                k -= i;
            }
        } else {
            return MTX_ERR_INVALID_MTX_SYMMETRY;
        }

    } else if (mtx->format == mtx_coordinate) {
        if (mtx->field == mtx_real) {
            const struct mtx_matrix_coordinate_real * data =
                (const struct mtx_matrix_coordinate_real *) mtx->data;
            *column = data[k].j;
        } else if (mtx->field == mtx_double) {
            const struct mtx_matrix_coordinate_double * data =
                (const struct mtx_matrix_coordinate_double *) mtx->data;
            *column = data[k].j;
        } else if (mtx->field == mtx_complex) {
            const struct mtx_matrix_coordinate_complex * data =
                (const struct mtx_matrix_coordinate_complex *) mtx->data;
            *column = data[k].j;
        } else if (mtx->field == mtx_integer) {
            const struct mtx_matrix_coordinate_integer * data =
                (const struct mtx_matrix_coordinate_integer *) mtx->data;
            *column = data[k].j;
        } else if (mtx->field == mtx_pattern) {
            const struct mtx_matrix_coordinate_pattern * data =
                (const struct mtx_matrix_coordinate_pattern *) mtx->data;
            *column = data[k].j;
        } else {
            return MTX_ERR_INVALID_MTX_FIELD;
        }
    } else {
        return MTX_ERR_INVALID_MTX_FORMAT;
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_matrix_num_nonzeros()` computes the number of nonzeros, including,
 * in the case of a matrix, any nonzeros that are not stored
 * explicitly due to symmetry.
 */
int mtx_matrix_num_nonzeros(
    enum mtx_object object,
    enum mtx_format format,
    enum mtx_field field,
    enum mtx_symmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const void * data,
    int64_t * num_nonzeros)
{
    int err;
    if (object == mtx_matrix) {
        if (format == mtx_array) {
            err = mtx_matrix_array_num_nonzeros(
                num_rows, num_columns, num_nonzeros);
            if (err)
                return err;
        } else if (format == mtx_coordinate) {
            err = mtx_matrix_coordinate_num_nonzeros(
                field, symmetry, num_rows, num_columns,
                size, data, num_nonzeros);
            if (err)
                return err;
        } else {
            return MTX_ERR_INVALID_MTX_FORMAT;
        }

    } else if (object == mtx_vector) {
        errno = ENOTSUP;
        return MTX_ERR_ERRNO;
    } else {
        return MTX_ERR_INVALID_MTX_OBJECT;
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_matrix_num_diagonal_nonzeros()` counts the number of nonzeros
 * on the main diagonal of a matrix in the Matrix Market format.
 */
int mtx_matrix_num_diagonal_nonzeros(
    const struct mtx * matrix,
    int64_t * num_diagonal_nonzeros)
{
    int err;
    if (matrix->object != mtx_matrix)
        return MTX_ERR_INVALID_MTX_OBJECT;

    if (matrix->format == mtx_array) {
        *num_diagonal_nonzeros =
            matrix->num_columns < matrix->num_rows
            ? matrix->num_columns : matrix->num_rows;
    } else if (matrix->format == mtx_coordinate) {
        err = mtx_matrix_coordinate_num_diagonal_nonzeros(
            matrix->field, matrix->size, matrix->data,
            num_diagonal_nonzeros);
        if (err)
            return err;
    } else {
        return MTX_ERR_INVALID_MTX_FORMAT;
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_matrix_nonzeros_per_row()` counts the number of nonzeros in
 * each row of a matrix in the Matrix Market format.
 *
 * If `include_strict_upper_triangular_part` is `true` and `symmetry`
 * is `symmetric`, `skew-symmetric` or `hermitian`, then nonzeros in
 * the strict upper triangular part are also counted. Conversely, if
 * `include_strict_upper_triangular_part` is `false`, then only
 * nonzeros in the lower triangular part of the matrix are counted.
 *
 * `mtx_matrix_nonzeros_per_row()` returns `MTX_ERR_ERRNO' with
 * `errno' set to `EINVAL' if `symmetry` is `general` and
 * `include_strict_upper_triangular_part` is `false`.
 */
int mtx_matrix_nonzeros_per_row(
    const struct mtx * matrix,
    bool include_strict_upper_triangular_part,
    int64_t * nonzeros_per_row)
{
    if (matrix->object != mtx_matrix) {
        errno = EINVAL;
        return MTX_ERR_ERRNO;
    }

    if (matrix->format == mtx_array) {
        for (int i = 0; i < matrix->num_rows; i++)
            nonzeros_per_row[i] += matrix->num_columns;

    } else if (matrix->format == mtx_coordinate) {
        switch (matrix->field) {
        case mtx_real:
            {
                const struct mtx_matrix_coordinate_real * data =
                    (const struct mtx_matrix_coordinate_real *)
                    matrix->data;
                if ((include_strict_upper_triangular_part &&
                     matrix->symmetry == mtx_general) ||
                    (!include_strict_upper_triangular_part &&
                     (matrix->symmetry == mtx_symmetric ||
                      matrix->symmetry == mtx_skew_symmetric ||
                      matrix->symmetry == mtx_hermitian)))
                {
                    for (int64_t k = 0; k < matrix->size; k++)
                        nonzeros_per_row[data[k].i-1]++;
                } else if (include_strict_upper_triangular_part &&
                           (matrix->symmetry == mtx_symmetric ||
                            matrix->symmetry == mtx_skew_symmetric ||
                            matrix->symmetry == mtx_hermitian))
                {
                    for (int64_t k = 0; k < matrix->size; k++) {
                        nonzeros_per_row[data[k].i-1]++;
                        if (data[k].i != data[k].j)
                            nonzeros_per_row[data[k].j-1]++;
                    }
                } else {
                    errno = EINVAL;
                    return MTX_ERR_ERRNO;
                }
            }
            break;
        case mtx_double:
            {
                const struct mtx_matrix_coordinate_double * data =
                    (const struct mtx_matrix_coordinate_double *)
                    matrix->data;
                for (int64_t k = 0; k < matrix->size; k++)
                    nonzeros_per_row[data[k].i-1]++;
            }
            break;
        case mtx_complex:
            {
                const struct mtx_matrix_coordinate_complex * data =
                    (const struct mtx_matrix_coordinate_complex *)
                    matrix->data;
                for (int64_t k = 0; k < matrix->size; k++)
                    nonzeros_per_row[data[k].i-1]++;
            }
            break;
        case mtx_integer:
            {
                const struct mtx_matrix_coordinate_integer * data =
                    (const struct mtx_matrix_coordinate_integer *)
                    matrix->data;
                for (int64_t k = 0; k < matrix->size; k++)
                    nonzeros_per_row[data[k].i-1]++;
            }
            break;
        case mtx_pattern:
            {
                const struct mtx_matrix_coordinate_pattern * data =
                    (const struct mtx_matrix_coordinate_pattern *)
                    matrix->data;
                for (int64_t k = 0; k < matrix->size; k++)
                    nonzeros_per_row[data[k].i-1]++;
            }
            break;
        default:
            errno = EINVAL;
            return MTX_ERR_ERRNO;
        }

    } else {
        errno = EINVAL;
        return MTX_ERR_ERRNO;
    }

    return MTX_SUCCESS;
}

/**
 * `mtx_matrix_size_per_row()' counts the number of entries stored for
 * each row of a matrix.
 *
 * The array `size_per_row' must point to an array containing enough
 * storage for `mtx->num_rows' values of type `int'.
 */
int mtx_matrix_size_per_row(
    const struct mtx * mtx,
    int * size_per_row)
{
    if (mtx->object != mtx_matrix)
        return MTX_ERR_INVALID_MTX_OBJECT;

    if (mtx->format == mtx_array) {
        if (mtx->symmetry == mtx_general) {
            for (int i = 0; i < mtx->num_rows; i++)
                size_per_row[i] = mtx->num_columns;
        } else if (mtx->symmetry == mtx_symmetric ||
                   mtx->symmetry == mtx_hermitian)
        {
            for (int i = 0; i < mtx->num_rows; i++)
                size_per_row[i] = i+1;
        } else if (mtx->symmetry == mtx_skew_symmetric) {
            for (int i = 0; i < mtx->num_rows; i++)
                size_per_row[i] = i;
        } else {
            return MTX_ERR_INVALID_MTX_SYMMETRY;
        }

    } else if (mtx->format == mtx_coordinate) {
        for (int i = 0; i < mtx->num_rows; i++)
            size_per_row[i] = 0;

        if (mtx->field == mtx_real) {
            const struct mtx_matrix_coordinate_real * data =
                (const struct mtx_matrix_coordinate_real *) mtx->data;
            for (int64_t k = 0; k < mtx->size; k++)
                size_per_row[data[k].i-1]++;
        } else if (mtx->field == mtx_double) {
            const struct mtx_matrix_coordinate_double * data =
                (const struct mtx_matrix_coordinate_double *) mtx->data;
            for (int64_t k = 0; k < mtx->size; k++)
                size_per_row[data[k].i-1]++;
        } else if (mtx->field == mtx_complex) {
            const struct mtx_matrix_coordinate_complex * data =
                (const struct mtx_matrix_coordinate_complex *) mtx->data;
            for (int64_t k = 0; k < mtx->size; k++)
                size_per_row[data[k].i-1]++;
        } else if (mtx->field == mtx_integer) {
            const struct mtx_matrix_coordinate_integer * data =
                (const struct mtx_matrix_coordinate_integer *) mtx->data;
            for (int64_t k = 0; k < mtx->size; k++)
                size_per_row[data[k].i-1]++;
        } else if (mtx->field == mtx_pattern) {
            const struct mtx_matrix_coordinate_pattern * data =
                (const struct mtx_matrix_coordinate_pattern *) mtx->data;
            for (int64_t k = 0; k < mtx->size; k++)
                size_per_row[data[k].i-1]++;
        } else {
            return MTX_ERR_INVALID_MTX_FIELD;
        }

    } else {
        return MTX_ERR_INVALID_MTX_FORMAT;
    }

    return MTX_SUCCESS;
}

/**
 * `mtx_matrix_row_ptr()' computes row pointers of a matrix.
 *
 * The array `row_ptr' must point to an array containing enough
 * storage for `mtx->num_rows+1' values of type `int64_t'.
 *
 * The matrix is not required to be sorted in row major order.  If the
 * matrix is sorted in row major order, then the `i'-th entry of the
 * `row_ptr' is the location of the first nonzero in the `mtx->data'
 * array that belongs to the `i+1'-th row of the matrix, for
 * `i=0,1,...,mtx->num_rows-1'. The final entry of `row_ptr' indicates
 * the position one place beyond the last nonzero in `mtx->data'.
 */
int mtx_matrix_row_ptr(
    const struct mtx * mtx,
    int64_t * row_ptr)
{
    if (mtx->object != mtx_matrix)
        return MTX_ERR_INVALID_MTX_OBJECT;

    /* 1. Count the number of entries in each row. */
    if (mtx->format == mtx_array) {
        if (mtx->symmetry == mtx_general) {
            row_ptr[0] = 0;
            for (int i = 0; i < mtx->num_rows; i++)
                row_ptr[i+1] = mtx->num_columns;
        } else if (mtx->symmetry == mtx_symmetric ||
                   mtx->symmetry == mtx_hermitian)
        {
            row_ptr[0] = 0;
            for (int i = 0; i < mtx->num_rows; i++)
                row_ptr[i+1] = i+1;
        } else if (mtx->symmetry == mtx_skew_symmetric) {
            row_ptr[0] = 0;
            for (int i = 0; i < mtx->num_rows; i++)
                row_ptr[i+1] = i;
        } else {
            return MTX_ERR_INVALID_MTX_SYMMETRY;
        }

    } else if (mtx->format == mtx_coordinate) {
        for (int i = 0; i <= mtx->num_rows; i++)
            row_ptr[i] = 0;

        if (mtx->field == mtx_real) {
            const struct mtx_matrix_coordinate_real * data =
                (const struct mtx_matrix_coordinate_real *) mtx->data;
            for (int64_t k = 0; k < mtx->size; k++)
                row_ptr[data[k].i]++;
        } else if (mtx->field == mtx_double) {
            const struct mtx_matrix_coordinate_double * data =
                (const struct mtx_matrix_coordinate_double *) mtx->data;
            for (int64_t k = 0; k < mtx->size; k++)
                row_ptr[data[k].i]++;
        } else if (mtx->field == mtx_complex) {
            const struct mtx_matrix_coordinate_complex * data =
                (const struct mtx_matrix_coordinate_complex *) mtx->data;
            for (int64_t k = 0; k < mtx->size; k++)
                row_ptr[data[k].i]++;
        } else if (mtx->field == mtx_integer) {
            const struct mtx_matrix_coordinate_integer * data =
                (const struct mtx_matrix_coordinate_integer *) mtx->data;
            for (int64_t k = 0; k < mtx->size; k++)
                row_ptr[data[k].i]++;
        } else if (mtx->field == mtx_pattern) {
            const struct mtx_matrix_coordinate_pattern * data =
                (const struct mtx_matrix_coordinate_pattern *) mtx->data;
            for (int64_t k = 0; k < mtx->size; k++)
                row_ptr[data[k].i]++;
        } else {
            return MTX_ERR_INVALID_MTX_FIELD;
        }

    } else {
        return MTX_ERR_INVALID_MTX_FORMAT;
    }

    /* 2. Compute the prefix sum of the row lengths. */
    for (int i = 1; i <= mtx->num_rows; i++)
        row_ptr[i] += row_ptr[i-1];
    return MTX_SUCCESS;
}

/**
 * `mtx_matrix_diagonal_size_per_row()` counts for each row of a
 * matrix the number of nonzero entries on the diagonal.
 *
 * The array `diagonal_size_per_row' must point to an array containing
 * enough storage for `mtx->num_rows' values of type `int'.
 */
int mtx_matrix_diagonal_size_per_row(
    const struct mtx * mtx,
    int * diagonal_size_per_row)
{
    if (mtx->object != mtx_matrix)
        return MTX_ERR_INVALID_MTX_OBJECT;

    if (mtx->format == mtx_array) {
        if (mtx->symmetry == mtx_general ||
            mtx->symmetry == mtx_symmetric ||
            mtx->symmetry == mtx_hermitian) {
            for (int i = 0; i < mtx->num_rows; i++)
                diagonal_size_per_row[i] = 1;
        } else if (mtx->symmetry == mtx_skew_symmetric) {
            for (int i = 0; i < mtx->num_rows; i++)
                diagonal_size_per_row[i] = 0;
        } else {
            return MTX_ERR_INVALID_MTX_SYMMETRY;
        }

    } else if (mtx->format == mtx_coordinate) {
        for (int i = 0; i < mtx->num_rows; i++)
            diagonal_size_per_row[i] = 0;

        if (mtx->field == mtx_real) {
            const struct mtx_matrix_coordinate_real * data =
                (const struct mtx_matrix_coordinate_real *) mtx->data;
            for (int64_t k = 0; k < mtx->size; k++) {
                if (data[k].i == data[k].j)
                    diagonal_size_per_row[data[k].i-1]++;
            }
        } else if (mtx->field == mtx_double) {
            const struct mtx_matrix_coordinate_double * data =
                (const struct mtx_matrix_coordinate_double *) mtx->data;
            for (int64_t k = 0; k < mtx->size; k++) {
                if (data[k].i == data[k].j)
                    diagonal_size_per_row[data[k].i-1]++;
            }
        } else if (mtx->field == mtx_complex) {
            const struct mtx_matrix_coordinate_complex * data =
                (const struct mtx_matrix_coordinate_complex *) mtx->data;
            for (int64_t k = 0; k < mtx->size; k++) {
                if (data[k].i == data[k].j)
                    diagonal_size_per_row[data[k].i-1]++;
            }
        } else if (mtx->field == mtx_integer) {
            const struct mtx_matrix_coordinate_integer * data =
                (const struct mtx_matrix_coordinate_integer *) mtx->data;
            for (int64_t k = 0; k < mtx->size; k++) {
                if (data[k].i == data[k].j)
                    diagonal_size_per_row[data[k].i-1]++;
            }
        } else if (mtx->field == mtx_pattern) {
            const struct mtx_matrix_coordinate_pattern * data =
                (const struct mtx_matrix_coordinate_pattern *) mtx->data;
            for (int64_t k = 0; k < mtx->size; k++) {
                if (data[k].i == data[k].j)
                    diagonal_size_per_row[data[k].i-1]++;
            }
        } else {
            return MTX_ERR_INVALID_MTX_FIELD;
        }

    } else {
        return MTX_ERR_INVALID_MTX_FORMAT;
    }

    return MTX_SUCCESS;
}
