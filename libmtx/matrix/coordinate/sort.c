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

#include <libmtx/error.h>
#include <libmtx/matrix/coordinate/data.h>
#include <libmtx/matrix/coordinate/sort.h>
#include <libmtx/mtx/sort.h>

#include <errno.h>

#include <stdlib.h>


/**
 * `mtx_matrix_coordinate_data_sort_column_major()' sorts the entries
 * of a matrix in coordinate format in column major order.
 */
static int mtx_matrix_coordinate_data_sort_column_major(
    struct mtx_matrix_coordinate_data * mtxdata)
{
    int err;
    if (mtxdata->sorting == mtx_column_major)
        return MTX_SUCCESS;

    /* 1. Allocate storage for column pointers. */
    int64_t * column_ptr = malloc(2*(mtxdata->num_columns+1) * sizeof(int64_t));
    if (!column_ptr)
        return MTX_ERR_ERRNO;

    /* 2. Count the number of nonzeros stored in each column. */
    err = mtx_matrix_coordinate_data_column_ptr(
        mtxdata, mtxdata->num_columns+1, column_ptr);
    if (err) {
        free(column_ptr);
        return err;
    }
    int64_t * column_endptr = &column_ptr[mtxdata->num_columns+1];
    for (int j = 0; j <= mtxdata->num_columns; j++)
        column_endptr[j] = column_ptr[j];

    /*
     * 3. Allocate storage for the sorted data, and sort nonzeros
     *    using an insertion sort within each column.
     */
    struct mtx_matrix_coordinate_data srcmtxdata;
    err = mtx_matrix_coordinate_data_copy(&srcmtxdata, mtxdata);
    if (err) {
        free(column_ptr);
        return err;
    }

    if (mtxdata->field == mtx_real) {
        if (mtxdata->precision == mtx_single) {
            struct mtx_matrix_coordinate_real_single * dest =
                mtxdata->data.real_single;
            const struct mtx_matrix_coordinate_real_single * src =
                srcmtxdata.data.real_single;
            for (int64_t k = 0; k < mtxdata->size; k++) {
                int j = src[k].j-1;
                int64_t l = column_endptr[j]-1;
                while (l >= column_ptr[j] && dest[l].i > src[k].i) {
                    dest[l+1] = dest[l];
                    l--;
                }
                dest[l+1] = src[k];
                column_endptr[j]++;
            }
        } else if (mtxdata->precision == mtx_double) {
            struct mtx_matrix_coordinate_real_double * dest =
                mtxdata->data.real_double;
            const struct mtx_matrix_coordinate_real_double * src =
                srcmtxdata.data.real_double;
            for (int64_t k = 0; k < mtxdata->size; k++) {
                int j = src[k].j-1;
                int64_t l = column_endptr[j]-1;
                while (l >= column_ptr[j] && dest[l].i > src[k].i) {
                    dest[l+1] = dest[l];
                    l--;
                }
                dest[l+1] = src[k];
                column_endptr[j]++;
            }
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }

    } else if (mtxdata->field == mtx_complex) {
        if (mtxdata->precision == mtx_single) {
            struct mtx_matrix_coordinate_complex_single * dest =
                mtxdata->data.complex_single;
            const struct mtx_matrix_coordinate_complex_single * src =
                srcmtxdata.data.complex_single;
            for (int64_t k = 0; k < mtxdata->size; k++) {
                int j = src[k].j-1;
                int64_t l = column_endptr[j]-1;
                while (l >= column_ptr[j] && dest[l].i > src[k].i) {
                    dest[l+1] = dest[l];
                    l--;
                }
                dest[l+1] = src[k];
                column_endptr[j]++;
            }

        } else if (mtxdata->precision == mtx_double) {
            struct mtx_matrix_coordinate_complex_double * dest =
                mtxdata->data.complex_double;
            const struct mtx_matrix_coordinate_complex_double * src =
                srcmtxdata.data.complex_double;
            for (int64_t k = 0; k < mtxdata->size; k++) {
                int j = src[k].j-1;
                int64_t l = column_endptr[j]-1;
                while (l >= column_ptr[j] && dest[l].i > src[k].i) {
                    dest[l+1] = dest[l];
                    l--;
                }
                dest[l+1] = src[k];
                column_endptr[j]++;
            }
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }

    } else if (mtxdata->field == mtx_integer) {
        if (mtxdata->precision == mtx_single) {
            struct mtx_matrix_coordinate_integer_single * dest =
                mtxdata->data.integer_single;
            const struct mtx_matrix_coordinate_integer_single * src =
                srcmtxdata.data.integer_single;
            for (int64_t k = 0; k < mtxdata->size; k++) {
                int j = src[k].j-1;
                int64_t l = column_endptr[j]-1;
                while (l >= column_ptr[j] && dest[l].i > src[k].i) {
                    dest[l+1] = dest[l];
                    l--;
                }
                dest[l+1] = src[k];
                column_endptr[j]++;
            }

        } else if (mtxdata->precision == mtx_double) {
            struct mtx_matrix_coordinate_integer_double * dest =
                mtxdata->data.integer_double;
            const struct mtx_matrix_coordinate_integer_double * src =
                srcmtxdata.data.integer_double;
            for (int64_t k = 0; k < mtxdata->size; k++) {
                int j = src[k].j-1;
                int64_t l = column_endptr[j]-1;
                while (l >= column_ptr[j] && dest[l].i > src[k].i) {
                    dest[l+1] = dest[l];
                    l--;
                }
                dest[l+1] = src[k];
                column_endptr[j]++;
            }
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }

    } else if (mtxdata->field == mtx_pattern) {
        struct mtx_matrix_coordinate_pattern * dest =
            mtxdata->data.pattern;
        const struct mtx_matrix_coordinate_pattern * src =
            srcmtxdata.data.pattern;
        for (int64_t k = 0; k < mtxdata->size; k++) {
            int j = src[k].j-1;
            int64_t l = column_endptr[j]-1;
            while (l >= column_ptr[j] && dest[l].i > src[k].i) {
                dest[l+1] = dest[l];
                l--;
            }
            dest[l+1] = src[k];
            column_endptr[j]++;
        }

    } else {
        mtx_matrix_coordinate_data_free(&srcmtxdata);
        free(column_ptr);
        return MTX_ERR_INVALID_MTX_FIELD;
    }

    mtx_matrix_coordinate_data_free(&srcmtxdata);
    free(column_ptr);
    mtxdata->sorting = mtx_column_major;
    return MTX_SUCCESS;
}

/**
 * `mtx_matrix_coordinate_data_sort_row_major()' sorts the entries of
 * a matrix in coordinate format in row major order.
 */
static int mtx_matrix_coordinate_data_sort_row_major(
    struct mtx_matrix_coordinate_data * mtxdata)
{
    int err;
    if (mtxdata->sorting == mtx_row_major)
        return MTX_SUCCESS;

    /* 1. Allocate storage for row pointers. */
    int64_t * row_ptr = malloc(2*(mtxdata->num_rows+1) * sizeof(int64_t));
    if (!row_ptr)
        return MTX_ERR_ERRNO;

    /* 2. Count the number of nonzeros stored in each row. */
    err = mtx_matrix_coordinate_data_row_ptr(
        mtxdata, mtxdata->num_rows+1, row_ptr);
    if (err) {
        free(row_ptr);
        return err;
    }
    int64_t * row_endptr = &row_ptr[mtxdata->num_rows+1];
    for (int j = 0; j <= mtxdata->num_rows; j++)
        row_endptr[j] = row_ptr[j];

    /*
     * 3. Allocate storage for the sorted data, and sort nonzeros
     *    using an insertion sort within each row.
     */
    struct mtx_matrix_coordinate_data srcmtxdata;
    err = mtx_matrix_coordinate_data_copy(&srcmtxdata, mtxdata);
    if (err) {
        free(row_ptr);
        return err;
    }

    if (mtxdata->field == mtx_real) {
        if (mtxdata->precision == mtx_single) {
            struct mtx_matrix_coordinate_real_single * dest =
                mtxdata->data.real_single;
            const struct mtx_matrix_coordinate_real_single * src =
                srcmtxdata.data.real_single;
            for (int64_t k = 0; k < mtxdata->size; k++) {
                int i = src[k].i-1;
                int64_t l = row_endptr[i]-1;
                while (l >= row_ptr[i] && dest[l].j > src[k].j) {
                    dest[l+1] = dest[l];
                    l--;
                }
                dest[l+1] = src[k];
                row_endptr[i]++;
            }
        } else if (mtxdata->precision == mtx_double) {
            struct mtx_matrix_coordinate_real_double * dest =
                mtxdata->data.real_double;
            const struct mtx_matrix_coordinate_real_double * src =
                srcmtxdata.data.real_double;
            for (int64_t k = 0; k < mtxdata->size; k++) {
                int i = src[k].i-1;
                int64_t l = row_endptr[i]-1;
                while (l >= row_ptr[i] && dest[l].j > src[k].j) {
                    dest[l+1] = dest[l];
                    l--;
                }
                dest[l+1] = src[k];
                row_endptr[i]++;
            }
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }

    } else if (mtxdata->field == mtx_complex) {
        if (mtxdata->precision == mtx_single) {
            struct mtx_matrix_coordinate_complex_single * dest =
                mtxdata->data.complex_single;
            const struct mtx_matrix_coordinate_complex_single * src =
                srcmtxdata.data.complex_single;
            for (int64_t k = 0; k < mtxdata->size; k++) {
                int i = src[k].i-1;
                int64_t l = row_endptr[i]-1;
                while (l >= row_ptr[i] && dest[l].j > src[k].j) {
                    dest[l+1] = dest[l];
                    l--;
                }
                dest[l+1] = src[k];
                row_endptr[i]++;
            }

        } else if (mtxdata->precision == mtx_double) {
            struct mtx_matrix_coordinate_complex_double * dest =
                mtxdata->data.complex_double;
            const struct mtx_matrix_coordinate_complex_double * src =
                srcmtxdata.data.complex_double;
            for (int64_t k = 0; k < mtxdata->size; k++) {
                int i = src[k].i-1;
                int64_t l = row_endptr[i]-1;
                while (l >= row_ptr[i] && dest[l].j > src[k].j) {
                    dest[l+1] = dest[l];
                    l--;
                }
                dest[l+1] = src[k];
                row_endptr[i]++;
            }
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }

    } else if (mtxdata->field == mtx_integer) {
        if (mtxdata->precision == mtx_single) {
            struct mtx_matrix_coordinate_integer_single * dest =
                mtxdata->data.integer_single;
            const struct mtx_matrix_coordinate_integer_single * src =
                srcmtxdata.data.integer_single;
            for (int64_t k = 0; k < mtxdata->size; k++) {
                int i = src[k].i-1;
                int64_t l = row_endptr[i]-1;
                while (l >= row_ptr[i] && dest[l].j > src[k].j) {
                    dest[l+1] = dest[l];
                    l--;
                }
                dest[l+1] = src[k];
                row_endptr[i]++;
            }

        } else if (mtxdata->precision == mtx_double) {
            struct mtx_matrix_coordinate_integer_double * dest =
                mtxdata->data.integer_double;
            const struct mtx_matrix_coordinate_integer_double * src =
                srcmtxdata.data.integer_double;
            for (int64_t k = 0; k < mtxdata->size; k++) {
                int i = src[k].i-1;
                int64_t l = row_endptr[i]-1;
                while (l >= row_ptr[i] && dest[l].j > src[k].j) {
                    dest[l+1] = dest[l];
                    l--;
                }
                dest[l+1] = src[k];
                row_endptr[i]++;
            }
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }

    } else if (mtxdata->field == mtx_pattern) {
        struct mtx_matrix_coordinate_pattern * dest =
            mtxdata->data.pattern;
        const struct mtx_matrix_coordinate_pattern * src =
            srcmtxdata.data.pattern;
        for (int64_t k = 0; k < mtxdata->size; k++) {
            int i = src[k].i-1;
            int64_t l = row_endptr[i]-1;
            while (l >= row_ptr[i] && dest[l].j > src[k].j) {
                dest[l+1] = dest[l];
                l--;
            }
            dest[l+1] = src[k];
            row_endptr[i]++;
        }

    } else {
        mtx_matrix_coordinate_data_free(&srcmtxdata);
        free(row_ptr);
        return MTX_ERR_INVALID_MTX_FIELD;
    }

    mtx_matrix_coordinate_data_free(&srcmtxdata);
    free(row_ptr);
    mtxdata->sorting = mtx_row_major;
    return MTX_SUCCESS;
}

/**
 * `mtx_matrix_coordinate_data_sort()' sorts the entries of a matrix
 * in coordinate format in a given order.
 */
int mtx_matrix_coordinate_data_sort(
    struct mtx_matrix_coordinate_data * mtxdata,
    enum mtx_sorting sorting)
{
    if (sorting == mtx_column_major) {
        return mtx_matrix_coordinate_data_sort_column_major(mtxdata);
    } else if (sorting == mtx_row_major) {
        return mtx_matrix_coordinate_data_sort_row_major(mtxdata);
    } else {
        return MTX_ERR_INVALID_MTX_SORTING;
    }
}
