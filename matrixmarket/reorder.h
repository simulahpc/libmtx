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
 * Reordering the rows and columns of sparse matrices.
 */

#ifndef MATRIXMARKET_REORDER_H
#define MATRIXMARKET_REORDER_H

struct mtx;

/**
 * `mtx_permute_vector()' permutes the elements of a vector based on a
 * given permutation.
 *
 * The array `permutation' should be a permutation of the integers
 * `1,2,...,mtx->num_rows'. The element at position `i' in the
 * permuted vector is then equal to the element at the position
 * `permutation[i-1]' in the original vector, for
 * `i=1,2,...,mtx->num_rows'.
 */
int mtx_permute_vector(
    struct mtx * mtx,
    const int * permutation);

/**
 * `mtx_permute_matrix()' permutes the elements of a matrix based on a
 * given permutation.
 *
 * The array `row_permutation' should be a permutation of the integers
 * `1,2,...,mtx->num_rows', and the array `column_permutation' should
 * be a permutation of the integers `1,2,...,mtx->num_columns'. The
 * elements belonging to row `i' and column `j' in the permuted matrix
 * are then equal to the elements in row `row_permutation[i-1]' and
 * column `column_permutation[j-1]' in the original matrix, for
 * `i=1,2,...,mtx->num_rows' and `j=1,2,...,mtx->num_columns'.
 */
int mtx_permute_matrix(
    struct mtx * mtx,
    const int * row_permutation,
    const int * column_permutation);

/**
 * `mtx_matrix_reorder_rcm()` reorders the rows of a symmetric sparse
 * matrix according to the Reverse Cuthill-McKee algorithm.
 *
 * `starting_row' is an integer in the range `[1,mtx->num_rows]',
 * which designates a starting row of the matrix for the Cuthill-McKee
 * algorithm. Alternatively, `starting_row' may be set to `0', in
 * which case a starting row is chosen automatically by selecting a
 * pseudo-peripheral vertex in the graph corresponding to the given
 * matrix.
 *
 * The matrix must be square, in coordinate format and already sorted
 * in row major order (see `mtx_sort'). It is assumed that the matrix
 * sparsity pattern is symmetric. Also, note that if the graph
 * consists of multiple connected components, then only the component
 * to which the starting row belongs is reordered.
 *
 * If successful, `mtx_matrix_reorder_rcm()' returns `MTX_SUCCESS',
 * and the rows and columns of `mtx' have been reordered according to
 * the Reverse Cuthill-McKee algorithm. If `permutation' is not
 * `NULL', then the underlying pointer is set to point to a newly
 * allocated array containing the permutation used to reorder the rows
 * and columns of `mtx'. In this case, the user is responsible for
 * calling `free()' to free the underlying storage.
 */
int mtx_matrix_reorder_rcm(
    struct mtx * mtx,
    int ** permutation,
    int starting_row);

#endif
