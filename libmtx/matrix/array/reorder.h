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

#ifndef LIBMTX_MTX_MATRIX_ARRAY_REORDER_H
#define LIBMTX_MTX_MATRIX_ARRAY_REORDER_H

struct mtx_matrix_array_data;

/**
 * `mtx_matrix_array_data_permute()' permutes the elements of a matrix
 * in array format based on a given permutation.
 *
 * The array `row_permutation' should be a permutation of the integers
 * `1,2,...,mtxdata->num_rows', and the array `column_permutation'
 * should be a permutation of the integers
 * `1,2,...,mtxdata->num_columns'. The elements belonging to row `i'
 * and column `j' in the permuted matrix are then equal to the
 * elements in row `row_permutation[i-1]' and column
 * `column_permutation[j-1]' in the original matrix, for
 * `i=1,2,...,mtxdata->num_rows' and `j=1,2,...,mtxdata->num_columns'.
 */
int mtx_matrix_array_data_permute(
    struct mtx_matrix_array_data * mtxdata,
    const int * row_permutation,
    const int * column_permutation);

#endif