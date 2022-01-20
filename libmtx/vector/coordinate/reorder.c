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
 * Reordering the rows or columns of vectors in coordinate format.
 */

#include <libmtx/vector/coordinate/data.h>
#include <libmtx/vector/coordinate/reorder.h>

#include <libmtx/error.h>

/**
 * `mtx_vector_coordinate_data_permute()' permutes the elements of a
 * vector in coordinate format based on a given permutation.
 *
 * The array `row_permutation' should be a permutation of the integers
 * `1,2,...,mtxdata->num_rows', and the array `column_permutation'
 * should be a permutation of the integers
 * `1,2,...,mtxdata->num_columns'. The elements belonging to row `i'
 * (or column `j') in the permuted vector are then equal to the
 * elements in row `row_permutation[i-1]' (or column
 * `column_permutation[j-1]') in the original vector, for
 * `i=1,2,...,mtxdata->num_rows' (and
 * `j=1,2,...,mtxdata->num_columns').
 */
int mtx_vector_coordinate_data_permute(
    struct mtx_vector_coordinate_data * mtxdata,
    const int * row_permutation,
    const int * column_permutation)
{
    const int * permutation;
    if (mtxdata->num_rows >= 0 && mtxdata->num_columns == -1) {
        permutation = row_permutation;
    } else if (mtxdata->num_rows == -1 && mtxdata->num_columns >= 0) {
        permutation = column_permutation;
    } else {
        return MTX_ERR_INVALID_MTX_SIZE;
    }

    if (mtxdata->field == mtx_real) {
        if (mtxdata->precision == mtx_single) {
            struct mtx_vector_coordinate_real_single * data =
                mtxdata->data.real_single;
            for (int64_t k = 0; k < mtxdata->size; k++)
                data[k].i = permutation[data[k].i-1];
        } else if (mtxdata->precision == mtx_double) {
            struct mtx_vector_coordinate_real_double * data =
                mtxdata->data.real_double;
            for (int64_t k = 0; k < mtxdata->size; k++)
                data[k].i = permutation[data[k].i-1];
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (mtxdata->field == mtx_complex) {
        if (mtxdata->precision == mtx_single) {
            struct mtx_vector_coordinate_complex_single * data =
                mtxdata->data.complex_single;
            for (int64_t k = 0; k < mtxdata->size; k++)
                data[k].i = permutation[data[k].i-1];
        } else if (mtxdata->precision == mtx_double) {
            struct mtx_vector_coordinate_complex_double * data =
                mtxdata->data.complex_double;
            for (int64_t k = 0; k < mtxdata->size; k++)
                data[k].i = permutation[data[k].i-1];
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (mtxdata->field == mtx_integer) {
        if (mtxdata->precision == mtx_single) {
            struct mtx_vector_coordinate_integer_single * data =
                mtxdata->data.integer_single;
            for (int64_t k = 0; k < mtxdata->size; k++)
                data[k].i = permutation[data[k].i-1];
        } else if (mtxdata->precision == mtx_double) {
            struct mtx_vector_coordinate_integer_double * data =
                mtxdata->data.integer_double;
            for (int64_t k = 0; k < mtxdata->size; k++)
                data[k].i = permutation[data[k].i-1];
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (mtxdata->field == mtx_pattern) {
        struct mtx_vector_coordinate_pattern * data =
            mtxdata->data.pattern;
        for (int64_t k = 0; k < mtxdata->size; k++)
            data[k].i = permutation[data[k].i-1];
    } else {
        return MTX_ERR_INVALID_MTX_FIELD;
    }
    return MTX_SUCCESS;
}
