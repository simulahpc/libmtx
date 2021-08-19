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
 * Reordering the rows and columns of matrices in array format.
 */

#include <libmtx/matrix/array/reorder.h>

#include <libmtx/error.h>
#include <libmtx/mtx/mtx.h>
#include <libmtx/mtx/reorder.h>

#include <errno.h>

/**
 * `mtx_matrix_array_data_permute()' permutes the elements of a matrix
 * in array format based on a given permutation.
 *
 * The array `rowperm' should be a permutation of the integers
 * `1,2,...,mtxdata->num_rows', and the array `colperm' should be a
 * permutation of the integers `1,2,...,mtxdata->num_columns'. The
 * elements belonging to row `i' and column `j' in the permuted matrix
 * are then equal to the elements in row `rowperm[i-1]' and column
 * `colperm[j-1]' in the original matrix, for
 * `i=1,2,...,mtxdata->num_rows' and `j=1,2,...,mtxdata->num_columns'.
 */
int mtx_matrix_array_data_permute(
    struct mtx_matrix_array_data * mtxdata,
    const int * rowperm,
    const int * colperm)
{
    if (mtxdata->triangle != mtx_nontriangular) {
        errno = ENOTSUP;
        return MTX_ERR_ERRNO;
    }

    struct mtx_matrix_array_data srcmtxdata;
    int err = mtx_matrix_array_data_copy(&srcmtxdata, mtxdata);
    if (err)
        return err;

    if (mtxdata->field == mtx_real) {
        if (mtxdata->precision == mtx_single) {
            const float * src = srcmtxdata.data.real_single;
            float * dst = mtxdata->data.real_single;
            if (rowperm && colperm) {
                for (int i = 0; i < mtxdata->num_rows; i++) {
                    for (int j = 0; j < mtxdata->num_columns; j++) {
                        int k = i*mtxdata->num_columns+j;
                        int l = (rowperm[i]-1)*mtxdata->num_columns + colperm[j]-1;
                        dst[k] = src[l];
                    }
                }
            } else if (rowperm) {
                for (int i = 0; i < mtxdata->num_rows; i++) {
                    for (int j = 0; j < mtxdata->num_columns; j++) {
                        int k = i*mtxdata->num_columns+j;
                        int l = (rowperm[i]-1)*mtxdata->num_columns + j;
                        dst[k] = src[l];
                    }
                }
            } else if (colperm) {
                for (int i = 0; i < mtxdata->num_rows; i++) {
                    for (int j = 0; j < mtxdata->num_columns; j++) {
                        int k = i*mtxdata->num_columns+j;
                        int l = i*mtxdata->num_columns + colperm[j]-1;
                        dst[k] = src[l];
                    }
                }
            }

        } else if (mtxdata->precision == mtx_double) {
            const double * src = srcmtxdata.data.real_double;
            double * dst = mtxdata->data.real_double;
            if (rowperm && colperm) {
                for (int i = 0; i < mtxdata->num_rows; i++) {
                    for (int j = 0; j < mtxdata->num_columns; j++) {
                        int k = i*mtxdata->num_columns+j;
                        int l = (rowperm[i]-1)*mtxdata->num_columns + colperm[j]-1;
                        dst[k] = src[l];
                    }
                }
            } else if (rowperm) {
                for (int i = 0; i < mtxdata->num_rows; i++) {
                    for (int j = 0; j < mtxdata->num_columns; j++) {
                        int k = i*mtxdata->num_columns+j;
                        int l = (rowperm[i]-1)*mtxdata->num_columns + j;
                        dst[k] = src[l];
                    }
                }
            } else if (colperm) {
                for (int i = 0; i < mtxdata->num_rows; i++) {
                    for (int j = 0; j < mtxdata->num_columns; j++) {
                        int k = i*mtxdata->num_columns+j;
                        int l = i*mtxdata->num_columns + colperm[j]-1;
                        dst[k] = src[l];
                    }
                }
            }

        } else {
            mtx_matrix_array_data_free(&srcmtxdata);
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (mtxdata->field == mtx_complex) {
        if (mtxdata->precision == mtx_single) {
            const float (* src)[2] = srcmtxdata.data.complex_single;
            float (* dst)[2] = mtxdata->data.complex_single;
            if (rowperm && colperm) {
                for (int i = 0; i < mtxdata->num_rows; i++) {
                    for (int j = 0; j < mtxdata->num_columns; j++) {
                        int k = i*mtxdata->num_columns+j;
                        int l = (rowperm[i]-1)*mtxdata->num_columns + colperm[j]-1;
                        dst[k][0] = src[l][0];
                        dst[k][1] = src[l][1];
                    }
                }
            } else if (rowperm) {
                for (int i = 0; i < mtxdata->num_rows; i++) {
                    for (int j = 0; j < mtxdata->num_columns; j++) {
                        int k = i*mtxdata->num_columns+j;
                        int l = (rowperm[i]-1)*mtxdata->num_columns + j;
                        dst[k][0] = src[l][0];
                        dst[k][1] = src[l][1];
                    }
                }
            } else if (colperm) {
                for (int i = 0; i < mtxdata->num_rows; i++) {
                    for (int j = 0; j < mtxdata->num_columns; j++) {
                        int k = i*mtxdata->num_columns+j;
                        int l = i*mtxdata->num_columns + colperm[j]-1;
                        dst[k][0] = src[l][0];
                        dst[k][1] = src[l][1];
                    }
                }
            }

        } else if (mtxdata->precision == mtx_double) {
            const double (* src)[2] = srcmtxdata.data.complex_double;
            double (* dst)[2] = mtxdata->data.complex_double;
            if (rowperm && colperm) {
                for (int i = 0; i < mtxdata->num_rows; i++) {
                    for (int j = 0; j < mtxdata->num_columns; j++) {
                        int k = i*mtxdata->num_columns+j;
                        int l = (rowperm[i]-1)*mtxdata->num_columns + colperm[j]-1;
                        dst[k][0] = src[l][0];
                        dst[k][1] = src[l][1];
                    }
                }
            } else if (rowperm) {
                for (int i = 0; i < mtxdata->num_rows; i++) {
                    for (int j = 0; j < mtxdata->num_columns; j++) {
                        int k = i*mtxdata->num_columns+j;
                        int l = (rowperm[i]-1)*mtxdata->num_columns + j;
                        dst[k][0] = src[l][0];
                        dst[k][1] = src[l][1];
                    }
                }
            } else if (colperm) {
                for (int i = 0; i < mtxdata->num_rows; i++) {
                    for (int j = 0; j < mtxdata->num_columns; j++) {
                        int k = i*mtxdata->num_columns+j;
                        int l = i*mtxdata->num_columns + colperm[j]-1;
                        dst[k][0] = src[l][0];
                        dst[k][1] = src[l][1];
                    }
                }
            }

        } else {
            mtx_matrix_array_data_free(&srcmtxdata);
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (mtxdata->field == mtx_integer) {
        if (mtxdata->precision == mtx_single) {
            const int32_t * src = srcmtxdata.data.integer_single;
            int32_t * dst = mtxdata->data.integer_single;
            if (rowperm && colperm) {
                for (int i = 0; i < mtxdata->num_rows; i++) {
                    for (int j = 0; j < mtxdata->num_columns; j++) {
                        int k = i*mtxdata->num_columns+j;
                        int l = (rowperm[i]-1)*mtxdata->num_columns + colperm[j]-1;
                        dst[k] = src[l];
                    }
                }
            } else if (rowperm) {
                for (int i = 0; i < mtxdata->num_rows; i++) {
                    for (int j = 0; j < mtxdata->num_columns; j++) {
                        int k = i*mtxdata->num_columns+j;
                        int l = (rowperm[i]-1)*mtxdata->num_columns + j;
                        dst[k] = src[l];
                    }
                }
            } else if (colperm) {
                for (int i = 0; i < mtxdata->num_rows; i++) {
                    for (int j = 0; j < mtxdata->num_columns; j++) {
                        int k = i*mtxdata->num_columns+j;
                        int l = i*mtxdata->num_columns + colperm[j]-1;
                        dst[k] = src[l];
                    }
                }
            }

        } else if (mtxdata->precision == mtx_double) {
            const int64_t * src = srcmtxdata.data.integer_double;
            int64_t * dst = mtxdata->data.integer_double;
            if (rowperm && colperm) {
                for (int i = 0; i < mtxdata->num_rows; i++) {
                    for (int j = 0; j < mtxdata->num_columns; j++) {
                        int k = i*mtxdata->num_columns+j;
                        int l = (rowperm[i]-1)*mtxdata->num_columns + colperm[j]-1;
                        dst[k] = src[l];
                    }
                }
            } else if (rowperm) {
                for (int i = 0; i < mtxdata->num_rows; i++) {
                    for (int j = 0; j < mtxdata->num_columns; j++) {
                        int k = i*mtxdata->num_columns+j;
                        int l = (rowperm[i]-1)*mtxdata->num_columns + j;
                        dst[k] = src[l];
                    }
                }
            } else if (colperm) {
                for (int i = 0; i < mtxdata->num_rows; i++) {
                    for (int j = 0; j < mtxdata->num_columns; j++) {
                        int k = i*mtxdata->num_columns+j;
                        int l = i*mtxdata->num_columns + colperm[j]-1;
                        dst[k] = src[l];
                    }
                }
            }

        } else {
            mtx_matrix_array_data_free(&srcmtxdata);
            return MTX_ERR_INVALID_PRECISION;
        }
    } else {
        mtx_matrix_array_data_free(&srcmtxdata);
        return MTX_ERR_INVALID_MTX_FIELD;
    }

    mtx_matrix_array_data_free(&srcmtxdata);
    return MTX_SUCCESS;
}
