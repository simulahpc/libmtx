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
 * Sorting matrix and vector nonzeros.
 */

#include <matrixmarket/error.h>
#include <matrixmarket/mtx.h>
#include <matrixmarket/header.h>
#include <matrixmarket/matrix.h>
#include <matrixmarket/matrix_coordinate.h>

#include <errno.h>

#include <stdlib.h>

/**
 * `mtx_sort_matrix_array()' sorts nonzeros in a given order
 * for matrices in array format.
 */
int mtx_sort_matrix_array(
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

/**
 * `mtx_sort_matrix_coordinate_real()' sorts nonzeros in a given
 * order for matrices in coordinate format with real entries.
 */
int mtx_sort_matrix_coordinate_real(
    struct mtx * mtx,
    enum mtx_sorting sorting)
{
    int err;
    if (mtx->object != mtx_matrix)
        return MTX_ERR_INVALID_MTX_OBJECT;
    if (mtx->format != mtx_coordinate)
        return MTX_ERR_INVALID_MTX_FORMAT;
    if (mtx->field != mtx_real)
        return MTX_ERR_INVALID_MTX_FIELD;

    /* Allocate storage for the sorted data. */
    struct mtx_matrix_coordinate_real * dest =
        malloc(mtx->size * mtx->nonzero_size);
    if (!dest)
        return MTX_ERR_ERRNO;

    if (sorting == mtx_column_major) {
        /* 1. Allocate storage for column pointers. */
        int64_t * column_ptr = malloc(2*(mtx->num_columns+1) * sizeof(int64_t));
        if (!column_ptr) {
            free(dest);
            return MTX_ERR_ERRNO;
        }

        /* 2. Count the number of nonzeros stored in each column. */
        err = mtx_matrix_column_ptr(mtx, column_ptr);
        if (err) {
            free(column_ptr);
            free(dest);
            return err;
        }
        int64_t * column_endptr = &column_ptr[mtx->num_columns+1];
        for (int j = 0; j <= mtx->num_columns; j++)
            column_endptr[j] = column_ptr[j];

        /* 3. Sort nonzeros using an insertion sort within each column. */
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

        free(column_ptr);
        free(mtx->data);
        mtx->data = dest;
        mtx->sorting = sorting;
    } else if (sorting == mtx_row_major) {
        /* 1. Allocate storage for row pointers. */
        int64_t * row_ptr = malloc(2*(mtx->num_rows+1) * sizeof(int64_t));
        if (!row_ptr) {
            free(dest);
            return MTX_ERR_ERRNO;
        }

        /* 2. Count the number of nonzeros stored in each row. */
        err = mtx_matrix_row_ptr(mtx, row_ptr);
        if (err) {
            free(row_ptr);
            free(dest);
            return err;
        }
        int64_t * row_endptr = &row_ptr[mtx->num_rows+1];
        for (int i = 0; i <= mtx->num_rows; i++)
            row_endptr[i] = row_ptr[i];

        /* 3. Sort nonzeros using an insertion sort within each row. */
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

        free(row_ptr);
        free(mtx->data);
        mtx->data = dest;
        mtx->sorting = sorting;
    } else {
        return MTX_ERR_INVALID_MTX_SORTING;
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_sort_matrix_coordinate_double()' sorts nonzeros in a given
 * order for matrices in coordinate format with double entries.
 */
int mtx_sort_matrix_coordinate_double(
    struct mtx * mtx,
    enum mtx_sorting sorting)
{
    int err;
    if (mtx->object != mtx_matrix)
        return MTX_ERR_INVALID_MTX_OBJECT;
    if (mtx->format != mtx_coordinate)
        return MTX_ERR_INVALID_MTX_FORMAT;
    if (mtx->field != mtx_double)
        return MTX_ERR_INVALID_MTX_FIELD;

    /* Allocate storage for the sorted data. */
    struct mtx_matrix_coordinate_double * dest =
        malloc(mtx->size * mtx->nonzero_size);
    if (!dest)
        return MTX_ERR_ERRNO;

    if (sorting == mtx_column_major) {
        /* 1. Allocate storage for column pointers. */
        int64_t * column_ptr = malloc(2*(mtx->num_columns+1) * sizeof(int64_t));
        if (!column_ptr) {
            free(dest);
            return MTX_ERR_ERRNO;
        }

        /* 2. Count the number of nonzeros stored in each column. */
        err = mtx_matrix_column_ptr(mtx, column_ptr);
        if (err) {
            free(column_ptr);
            free(dest);
            return err;
        }
        int64_t * column_endptr = &column_ptr[mtx->num_columns+1];
        for (int j = 0; j <= mtx->num_columns; j++)
            column_endptr[j] = column_ptr[j];

        /* 3. Sort nonzeros using an insertion sort within each column. */
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

        free(column_ptr);
        free(mtx->data);
        mtx->data = dest;
        mtx->sorting = sorting;
    } else if (sorting == mtx_row_major) {
        /* 1. Allocate storage for row pointers. */
        int64_t * row_ptr = malloc(2*(mtx->num_rows+1) * sizeof(int64_t));
        if (!row_ptr) {
            free(dest);
            return MTX_ERR_ERRNO;
        }

        /* 2. Count the number of nonzeros stored in each row. */
        err = mtx_matrix_row_ptr(mtx, row_ptr);
        if (err) {
            free(row_ptr);
            free(dest);
            return err;
        }
        int64_t * row_endptr = &row_ptr[mtx->num_rows+1];
        for (int i = 0; i <= mtx->num_rows; i++)
            row_endptr[i] = row_ptr[i];

        /* 3. Sort nonzeros using an insertion sort within each row. */
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

        free(row_ptr);
        free(mtx->data);
        mtx->data = dest;
        mtx->sorting = sorting;
    } else {
        return MTX_ERR_INVALID_MTX_SORTING;
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_sort_matrix_coordinate_complex()' sorts nonzeros in a given
 * order for matrices in coordinate format with complex entries.
 */
int mtx_sort_matrix_coordinate_complex(
    struct mtx * mtx,
    enum mtx_sorting sorting)
{
    int err;
    if (mtx->object != mtx_matrix)
        return MTX_ERR_INVALID_MTX_OBJECT;
    if (mtx->format != mtx_coordinate)
        return MTX_ERR_INVALID_MTX_FORMAT;
    if (mtx->field != mtx_complex)
        return MTX_ERR_INVALID_MTX_FIELD;

    /* Allocate storage for the sorted data. */
    struct mtx_matrix_coordinate_complex * dest =
        malloc(mtx->size * mtx->nonzero_size);
    if (!dest)
        return MTX_ERR_ERRNO;

    if (sorting == mtx_column_major) {
        /* 1. Allocate storage for column pointers. */
        int64_t * column_ptr = malloc(2*(mtx->num_columns+1) * sizeof(int64_t));
        if (!column_ptr) {
            free(dest);
            return MTX_ERR_ERRNO;
        }

        /* 2. Count the number of nonzeros stored in each column. */
        err = mtx_matrix_column_ptr(mtx, column_ptr);
        if (err) {
            free(column_ptr);
            free(dest);
            return err;
        }
        int64_t * column_endptr = &column_ptr[mtx->num_columns+1];
        for (int j = 0; j <= mtx->num_columns; j++)
            column_endptr[j] = column_ptr[j];

        /* 3. Sort nonzeros using an insertion sort within each column. */
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

        free(column_ptr);
        free(mtx->data);
        mtx->data = dest;
        mtx->sorting = sorting;
    } else if (sorting == mtx_row_major) {
        /* 1. Allocate storage for row pointers. */
        int64_t * row_ptr = malloc(2*(mtx->num_rows+1) * sizeof(int64_t));
        if (!row_ptr) {
            free(dest);
            return MTX_ERR_ERRNO;
        }

        /* 2. Count the number of nonzeros stored in each row. */
        err = mtx_matrix_row_ptr(mtx, row_ptr);
        if (err) {
            free(row_ptr);
            free(dest);
            return err;
        }
        int64_t * row_endptr = &row_ptr[mtx->num_rows+1];
        for (int i = 0; i <= mtx->num_rows; i++)
            row_endptr[i] = row_ptr[i];

        /* 3. Sort nonzeros using an insertion sort within each row. */
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

        free(row_ptr);
        free(mtx->data);
        mtx->data = dest;
        mtx->sorting = sorting;
    } else {
        return MTX_ERR_INVALID_MTX_SORTING;
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_sort_matrix_coordinate_integer()' sorts nonzeros in a given
 * order for matrices in coordinate format with integer entries.
 */
int mtx_sort_matrix_coordinate_integer(
    struct mtx * mtx,
    enum mtx_sorting sorting)
{
    int err;
    if (mtx->object != mtx_matrix)
        return MTX_ERR_INVALID_MTX_OBJECT;
    if (mtx->format != mtx_coordinate)
        return MTX_ERR_INVALID_MTX_FORMAT;
    if (mtx->field != mtx_integer)
        return MTX_ERR_INVALID_MTX_FIELD;

    /* Allocate storage for the sorted data. */
    struct mtx_matrix_coordinate_integer * dest =
        malloc(mtx->size * mtx->nonzero_size);
    if (!dest)
        return MTX_ERR_ERRNO;

    if (sorting == mtx_column_major) {
        /* 1. Allocate storage for column pointers. */
        int64_t * column_ptr = malloc(2*(mtx->num_columns+1) * sizeof(int64_t));
        if (!column_ptr) {
            free(dest);
            return MTX_ERR_ERRNO;
        }

        /* 2. Count the number of nonzeros stored in each column. */
        err = mtx_matrix_column_ptr(mtx, column_ptr);
        if (err) {
            free(column_ptr);
            free(dest);
            return err;
        }
        int64_t * column_endptr = &column_ptr[mtx->num_columns+1];
        for (int j = 0; j <= mtx->num_columns; j++)
            column_endptr[j] = column_ptr[j];

        /* 3. Sort nonzeros using an insertion sort within each column. */
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

        free(column_ptr);
        free(mtx->data);
        mtx->data = dest;
        mtx->sorting = sorting;
    } else if (sorting == mtx_row_major) {
        /* 1. Allocate storage for row pointers. */
        int64_t * row_ptr = malloc(2*(mtx->num_rows+1) * sizeof(int64_t));
        if (!row_ptr) {
            free(dest);
            return MTX_ERR_ERRNO;
        }

        /* 2. Count the number of nonzeros stored in each row. */
        err = mtx_matrix_row_ptr(mtx, row_ptr);
        if (err) {
            free(row_ptr);
            free(dest);
            return err;
        }
        int64_t * row_endptr = &row_ptr[mtx->num_rows+1];
        for (int i = 0; i <= mtx->num_rows; i++)
            row_endptr[i] = row_ptr[i];

        /* 3. Sort nonzeros using an insertion sort within each row. */
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

        free(row_ptr);
        free(mtx->data);
        mtx->data = dest;
        mtx->sorting = sorting;
    } else {
        return MTX_ERR_INVALID_MTX_SORTING;
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_sort_matrix_coordinate_pattern()' sorts nonzeros in a given
 * order for matrices in coordinate format with pattern entries.
 */
int mtx_sort_matrix_coordinate_pattern(
    struct mtx * mtx,
    enum mtx_sorting sorting)
{
    int err;
    if (mtx->object != mtx_matrix)
        return MTX_ERR_INVALID_MTX_OBJECT;
    if (mtx->format != mtx_coordinate)
        return MTX_ERR_INVALID_MTX_FORMAT;
    if (mtx->field != mtx_pattern)
        return MTX_ERR_INVALID_MTX_FIELD;

    /* Allocate storage for the sorted data. */
    struct mtx_matrix_coordinate_pattern * dest =
        malloc(mtx->size * mtx->nonzero_size);
    if (!dest)
        return MTX_ERR_ERRNO;

    if (sorting == mtx_column_major) {
        /* 1. Allocate storage for column pointers. */
        int64_t * column_ptr = malloc(2*(mtx->num_columns+1) * sizeof(int64_t));
        if (!column_ptr) {
            free(dest);
            return MTX_ERR_ERRNO;
        }

        /* 2. Count the number of nonzeros stored in each column. */
        err = mtx_matrix_column_ptr(mtx, column_ptr);
        if (err) {
            free(column_ptr);
            free(dest);
            return err;
        }
        int64_t * column_endptr = &column_ptr[mtx->num_columns+1];
        for (int j = 0; j <= mtx->num_columns; j++)
            column_endptr[j] = column_ptr[j];

        /* 3. Sort nonzeros using an insertion sort within each column. */
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

        free(column_ptr);
        free(mtx->data);
        mtx->data = dest;
        mtx->sorting = sorting;
    } else if (sorting == mtx_row_major) {
        /* 1. Allocate storage for row pointers. */
        int64_t * row_ptr = malloc(2*(mtx->num_rows+1) * sizeof(int64_t));
        if (!row_ptr) {
            free(dest);
            return MTX_ERR_ERRNO;
        }

        /* 2. Count the number of nonzeros stored in each row. */
        err = mtx_matrix_row_ptr(mtx, row_ptr);
        if (err) {
            free(row_ptr);
            free(dest);
            return err;
        }
        int64_t * row_endptr = &row_ptr[mtx->num_rows+1];
        for (int i = 0; i <= mtx->num_rows; i++)
            row_endptr[i] = row_ptr[i];

        /* 3. Sort nonzeros using an insertion sort within each row. */
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

        free(row_ptr);
        free(mtx->data);
        mtx->data = dest;
        mtx->sorting = sorting;
    } else {
        return MTX_ERR_INVALID_MTX_SORTING;
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_sort_matrix_coordinate()' sorts nonzeros in a given order
 * for matrices in coordinate format.
 */
int mtx_sort_matrix_coordinate(
    struct mtx * mtx,
    enum mtx_sorting sorting)
{
    int err;
    if (mtx->object != mtx_matrix)
        return MTX_ERR_INVALID_MTX_OBJECT;
    if (mtx->format != mtx_coordinate)
        return MTX_ERR_INVALID_MTX_FORMAT;

    if (mtx->field == mtx_real) {
        return mtx_sort_matrix_coordinate_real(mtx, sorting);
    } else if (mtx->field == mtx_double) {
        return mtx_sort_matrix_coordinate_double(mtx, sorting);
    } else if (mtx->field == mtx_complex) {
        return mtx_sort_matrix_coordinate_complex(mtx, sorting);
    } else if (mtx->field == mtx_integer) {
        return mtx_sort_matrix_coordinate_integer(mtx, sorting);
    } else if (mtx->field == mtx_pattern) {
        return mtx_sort_matrix_coordinate_pattern(mtx, sorting);
    } else {
        return MTX_ERR_INVALID_MTX_FIELD;
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
        return mtx_sort_matrix_array(mtx, sorting);
    } else if (mtx->format == mtx_coordinate) {
        return mtx_sort_matrix_coordinate(mtx, sorting);
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
