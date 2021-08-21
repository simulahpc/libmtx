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
 * Reordering the rows or columns of vectors in array format.
 */

#include <libmtx/vector/array/data.h>
#include <libmtx/vector/array/reorder.h>

#include <libmtx/error.h>

/**
 * `mtx_vector_array_data_permute()' permutes the elements of a vector
 * in array format based on a given permutation.
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
int mtx_vector_array_data_permute(
    struct mtx_vector_array_data * mtxdata,
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

    struct mtx_vector_array_data srcmtxdata;
    int err = mtx_vector_array_data_copy_init(&srcmtxdata, mtxdata);
    if (err)
        return err;

    if (mtxdata->field == mtx_real) {
        if (mtxdata->precision == mtx_single) {
            const float * src = srcmtxdata.data.real_single;
            float * dst = mtxdata->data.real_single;
            for (int64_t k = 0; k < mtxdata->size; k++)
                dst[k] = src[permutation[k]-1];
        } else if (mtxdata->precision == mtx_double) {
            const double * src = srcmtxdata.data.real_double;
            double * dst = mtxdata->data.real_double;
            for (int64_t k = 0; k < mtxdata->size; k++)
                dst[k] = src[permutation[k]-1];
        } else {
            mtx_vector_array_data_free(&srcmtxdata);
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (mtxdata->field == mtx_complex) {
        if (mtxdata->precision == mtx_single) {
            const float (* src)[2] = srcmtxdata.data.complex_single;
            float (* dst)[2] = mtxdata->data.complex_single;
            for (int64_t k = 0; k < mtxdata->size; k++) {
                dst[k][0] = src[permutation[k]-1][0];
                dst[k][1] = src[permutation[k]-1][1];
            }
        } else if (mtxdata->precision == mtx_double) {
            const double (* src)[2] = srcmtxdata.data.complex_double;
            double (* dst)[2] = mtxdata->data.complex_double;
            for (int64_t k = 0; k < mtxdata->size; k++) {
                dst[k][0] = src[permutation[k]-1][0];
                dst[k][1] = src[permutation[k]-1][1];
            }
        } else {
            mtx_vector_array_data_free(&srcmtxdata);
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (mtxdata->field == mtx_integer) {
        if (mtxdata->precision == mtx_single) {
            const int32_t * src = srcmtxdata.data.integer_single;
            int32_t * dst = mtxdata->data.integer_single;
            for (int64_t k = 0; k < mtxdata->size; k++)
                dst[k] = src[permutation[k]-1];
        } else if (mtxdata->precision == mtx_double) {
            const int64_t * src = srcmtxdata.data.integer_double;
            int64_t * dst = mtxdata->data.integer_double;
            for (int64_t k = 0; k < mtxdata->size; k++)
                dst[k] = src[permutation[k]-1];
        } else {
            mtx_vector_array_data_free(&srcmtxdata);
            return MTX_ERR_INVALID_PRECISION;
        }
    } else {
        mtx_vector_array_data_free(&srcmtxdata);
        return MTX_ERR_INVALID_MTX_FIELD;
    }

    mtx_vector_array_data_free(&srcmtxdata);
    return MTX_SUCCESS;
}
