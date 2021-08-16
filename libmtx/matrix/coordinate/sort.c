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
 * Sorting sparse matrices in coordinate format.
 */

#include <libmtx/matrix/coordinate/sort.h>

#include <libmtx/error.h>
#include <libmtx/matrix/coordinate.h>
#include <libmtx/mtx/matrix.h>
#include <libmtx/mtx/mtx.h>
#include <libmtx/mtx/sort.h>

#include <errno.h>

#include <stdlib.h>

struct mtx;

/**
 * `mtx_matrix_coordinate_sort_column_major()' sorts the entries of a
 * matrix in coordinate format in column major order.
 */
int mtx_matrix_coordinate_sort_column_major(
    struct mtx * mtx)
{
    int err;
    if (mtx->object != mtx_matrix)
        return MTX_ERR_INVALID_MTX_OBJECT;
    if (mtx->format != mtx_coordinate)
        return MTX_ERR_INVALID_MTX_FORMAT;
    if (mtx->sorting == mtx_column_major)
        return MTX_SUCCESS;

    /* 1. Allocate storage for column pointers. */
    int64_t * column_ptr = malloc(2*(mtx->num_columns+1) * sizeof(int64_t));
    if (!column_ptr)
        return MTX_ERR_ERRNO;

    /* 2. Count the number of nonzeros stored in each column. */
    err = mtx_matrix_column_ptr(mtx, column_ptr);
    if (err) {
        free(column_ptr);
        return err;
    }
    int64_t * column_endptr = &column_ptr[mtx->num_columns+1];
    for (int j = 0; j <= mtx->num_columns; j++)
        column_endptr[j] = column_ptr[j];

    /*
     * 3. Allocate storage for the sorted data, and sort nonzeros
     *    using an insertion sort within each column.
     */
    if (mtx->field == mtx_real) {
        struct mtx_matrix_coordinate_real * dest =
            malloc(mtx->size * mtx->nonzero_size);
        if (!dest) {
            free(column_ptr);
            return MTX_ERR_ERRNO;
        }

        const struct mtx_matrix_coordinate_real * src =
            (const struct mtx_matrix_coordinate_real *) mtx->data;
        for (int64_t k = 0; k < mtx->size; k++) {
            int j = src[k].j-1;
            int64_t l = column_endptr[j]-1;
            while (l >= column_ptr[j] && dest[l].i > src[k].i) {
                dest[l+1] = dest[l];
                l--;
            }
            dest[l+1] = src[k];
            column_endptr[j]++;
        }
        free(mtx->data);
        mtx->data = dest;

    } else if (mtx->field == mtx_double) {
        struct mtx_matrix_coordinate_double * dest =
            malloc(mtx->size * mtx->nonzero_size);
        if (!dest) {
            free(column_ptr);
            return MTX_ERR_ERRNO;
        }

        const struct mtx_matrix_coordinate_double * src =
            (const struct mtx_matrix_coordinate_double *) mtx->data;
        for (int64_t k = 0; k < mtx->size; k++) {
            int j = src[k].j-1;
            int64_t l = column_endptr[j]-1;
            while (l >= column_ptr[j] && dest[l].i > src[k].i) {
                dest[l+1] = dest[l];
                l--;
            }
            dest[l+1] = src[k];
            column_endptr[j]++;
        }
        free(mtx->data);
        mtx->data = dest;

    } else if (mtx->field == mtx_complex) {
        struct mtx_matrix_coordinate_complex * dest =
            malloc(mtx->size * mtx->nonzero_size);
        if (!dest) {
            free(column_ptr);
            return MTX_ERR_ERRNO;
        }

        const struct mtx_matrix_coordinate_complex * src =
            (const struct mtx_matrix_coordinate_complex *) mtx->data;
        for (int64_t k = 0; k < mtx->size; k++) {
            int j = src[k].j-1;
            int64_t l = column_endptr[j]-1;
            while (l >= column_ptr[j] && dest[l].i > src[k].i) {
                dest[l+1] = dest[l];
                l--;
            }
            dest[l+1] = src[k];
            column_endptr[j]++;
        }
        free(mtx->data);
        mtx->data = dest;

    } else if (mtx->field == mtx_integer) {
        struct mtx_matrix_coordinate_integer * dest =
            malloc(mtx->size * mtx->nonzero_size);
        if (!dest) {
            free(column_ptr);
            return MTX_ERR_ERRNO;
        }

        const struct mtx_matrix_coordinate_integer * src =
            (const struct mtx_matrix_coordinate_integer *) mtx->data;
        for (int64_t k = 0; k < mtx->size; k++) {
            int j = src[k].j-1;
            int64_t l = column_endptr[j]-1;
            while (l >= column_ptr[j] && dest[l].i > src[k].i) {
                dest[l+1] = dest[l];
                l--;
            }
            dest[l+1] = src[k];
            column_endptr[j]++;
        }
        free(mtx->data);
        mtx->data = dest;

    } else if (mtx->field == mtx_pattern) {
        struct mtx_matrix_coordinate_pattern * dest =
            malloc(mtx->size * mtx->nonzero_size);
        if (!dest) {
            free(column_ptr);
            return MTX_ERR_ERRNO;
        }

        const struct mtx_matrix_coordinate_pattern * src =
            (const struct mtx_matrix_coordinate_pattern *) mtx->data;
        for (int64_t k = 0; k < mtx->size; k++) {
            int j = src[k].j-1;
            int64_t l = column_endptr[j]-1;
            while (l >= column_ptr[j] && dest[l].i > src[k].i) {
                dest[l+1] = dest[l];
                l--;
            }
            dest[l+1] = src[k];
            column_endptr[j]++;
        }
        free(mtx->data);
        mtx->data = dest;

    } else {
        free(column_ptr);
        return MTX_ERR_INVALID_MTX_FIELD;
    }

    free(column_ptr);
    mtx->sorting = mtx_column_major;
    return MTX_SUCCESS;
}

/**
 * `mtx_matrix_coordinate_sort_row_major()' sorts the entries of a
 * matrix in coordinate format in row major order.
 */
int mtx_matrix_coordinate_sort_row_major(
    struct mtx * mtx)
{
    int err;
    if (mtx->object != mtx_matrix)
        return MTX_ERR_INVALID_MTX_OBJECT;
    if (mtx->format != mtx_coordinate)
        return MTX_ERR_INVALID_MTX_FORMAT;
    if (mtx->sorting == mtx_row_major)
        return MTX_SUCCESS;

    /* 1. Allocate storage for row pointers. */
    int64_t * row_ptr = malloc(2*(mtx->num_rows+1) * sizeof(int64_t));
    if (!row_ptr)
        return MTX_ERR_ERRNO;

    /* 2. Count the number of nonzeros stored in each row. */
    err = mtx_matrix_row_ptr(mtx, row_ptr);
    if (err) {
        free(row_ptr);
        return err;
    }
    int64_t * row_endptr = &row_ptr[mtx->num_rows+1];
    for (int i = 0; i <= mtx->num_rows; i++)
        row_endptr[i] = row_ptr[i];

    /*
     * 3. Allocate storage for the sorted data, and sort nonzeros
     *    using an insertion sort within each row.
     */
    if (mtx->field == mtx_real) {
        struct mtx_matrix_coordinate_real * dest =
            malloc(mtx->size * mtx->nonzero_size);
        if (!dest) {
            free(row_ptr);
            return MTX_ERR_ERRNO;
        }

        const struct mtx_matrix_coordinate_real * src =
            (const struct mtx_matrix_coordinate_real *) mtx->data;
        for (int64_t k = 0; k < mtx->size; k++) {
            int i = src[k].i-1;
            int64_t l = row_endptr[i]-1;
            while (l >= row_ptr[i] && dest[l].j > src[k].j) {
                dest[l+1] = dest[l];
                l--;
            }
            dest[l+1] = src[k];
            row_endptr[i]++;
        }
        free(mtx->data);
        mtx->data = dest;

    } else if (mtx->field == mtx_double) {
        struct mtx_matrix_coordinate_double * dest =
            malloc(mtx->size * mtx->nonzero_size);
        if (!dest) {
            free(row_ptr);
            return MTX_ERR_ERRNO;
        }

        const struct mtx_matrix_coordinate_double * src =
            (const struct mtx_matrix_coordinate_double *) mtx->data;
        for (int64_t k = 0; k < mtx->size; k++) {
            int i = src[k].i-1;
            int64_t l = row_endptr[i]-1;
            while (l >= row_ptr[i] && dest[l].j > src[k].j) {
                dest[l+1] = dest[l];
                l--;
            }
            dest[l+1] = src[k];
            row_endptr[i]++;
        }
        free(mtx->data);
        mtx->data = dest;

    } else if (mtx->field == mtx_complex) {
        struct mtx_matrix_coordinate_complex * dest =
            malloc(mtx->size * mtx->nonzero_size);
        if (!dest) {
            free(row_ptr);
            return MTX_ERR_ERRNO;
        }

        const struct mtx_matrix_coordinate_complex * src =
            (const struct mtx_matrix_coordinate_complex *) mtx->data;
        for (int64_t k = 0; k < mtx->size; k++) {
            int i = src[k].i-1;
            int64_t l = row_endptr[i]-1;
            while (l >= row_ptr[i] && dest[l].j > src[k].j) {
                dest[l+1] = dest[l];
                l--;
            }
            dest[l+1] = src[k];
            row_endptr[i]++;
        }
        free(mtx->data);
        mtx->data = dest;

    } else if (mtx->field == mtx_integer) {
        struct mtx_matrix_coordinate_integer * dest =
            malloc(mtx->size * mtx->nonzero_size);
        if (!dest) {
            free(row_ptr);
            return MTX_ERR_ERRNO;
        }

        const struct mtx_matrix_coordinate_integer * src =
            (const struct mtx_matrix_coordinate_integer *) mtx->data;
        for (int64_t k = 0; k < mtx->size; k++) {
            int i = src[k].i-1;
            int64_t l = row_endptr[i]-1;
            while (l >= row_ptr[i] && dest[l].j > src[k].j) {
                dest[l+1] = dest[l];
                l--;
            }
            dest[l+1] = src[k];
            row_endptr[i]++;
        }
        free(mtx->data);
        mtx->data = dest;

    } else if (mtx->field == mtx_pattern) {
        struct mtx_matrix_coordinate_pattern * dest =
            malloc(mtx->size * mtx->nonzero_size);
        if (!dest) {
            free(row_ptr);
            return MTX_ERR_ERRNO;
        }

        const struct mtx_matrix_coordinate_pattern * src =
            (const struct mtx_matrix_coordinate_pattern *) mtx->data;
        for (int64_t k = 0; k < mtx->size; k++) {
            int i = src[k].i-1;
            int64_t l = row_endptr[i]-1;
            while (l >= row_ptr[i] && dest[l].j > src[k].j) {
                dest[l+1] = dest[l];
                l--;
            }
            dest[l+1] = src[k];
            row_endptr[i]++;
        }
        free(mtx->data);
        mtx->data = dest;

    } else {
        free(row_ptr);
        return MTX_ERR_INVALID_MTX_FIELD;
    }

    free(row_ptr);
    mtx->sorting = mtx_row_major;
    return MTX_SUCCESS;
}

/**
 * `mtx_matrix_coordinate_sort()' sorts the entries of a matrix in coordinate
 * format in a given order.
 */
int mtx_matrix_coordinate_sort(
    struct mtx * mtx,
    enum mtx_sorting sorting)
{
    int err;
    if (mtx->object != mtx_matrix)
        return MTX_ERR_INVALID_MTX_OBJECT;
    if (mtx->format != mtx_coordinate)
        return MTX_ERR_INVALID_MTX_FORMAT;

    if (sorting == mtx_column_major) {
        return mtx_matrix_coordinate_sort_column_major(mtx);
    } else if (sorting == mtx_row_major) {
        return mtx_matrix_coordinate_sort_row_major(mtx);
    } else {
        return MTX_ERR_INVALID_MTX_SORTING;
    }
    return MTX_SUCCESS;
}
