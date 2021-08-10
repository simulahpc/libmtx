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

#ifndef LIBMTX_MTX_VECTOR_ARRAY_REORDER_H
#define LIBMTX_MTX_VECTOR_ARRAY_REORDER_H

struct mtx;

/**
 * `mtx_vector_array_permute()' permutes the elements of a vector
 * based on a given permutation.
 *
 * The array `row_permutation' should be a permutation of the integers
 * `1,2,...,mtx->num_rows', and the array `column_permutation' should
 * be a permutation of the integers `1,2,...,mtx->num_columns'. The
 * elements belonging to row `i' (or column `j') in the permuted
 * vector are then equal to the elements in row `row_permutation[i-1]'
 * (or column `column_permutation[j-1]') in the original vector, for
 * `i=1,2,...,mtx->num_rows' (and `j=1,2,...,mtx->num_columns').
 */
int mtx_vector_array_permute(
    struct mtx * mtx,
    const int * row_permutation,
    const int * column_permutation);

#endif
