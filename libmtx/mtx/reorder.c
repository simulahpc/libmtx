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
 * Reordering the rows and columns of sparse matrices.
 */

#include <libmtx/mtx/reorder.h>

#include <libmtx/error.h>
#include <libmtx/matrix/array/reorder.h>
#include <libmtx/matrix/coordinate/reorder.h>
#include <libmtx/mtx/header.h>
#include <libmtx/mtx/mtx.h>
#include <libmtx/vector/array/reorder.h>
#include <libmtx/vector/coordinate/reorder.h>

#include <errno.h>

#include <stdlib.h>

/**
 * `mtx_ordering_str()` is a string representing the ordering type.
 */
const char * mtx_ordering_str(
    enum mtx_ordering ordering)
{
    switch (ordering) {
    case mtx_unordered: return "unordered";
    case mtx_rcm: return "rcm";
    default: return "unknown";
    }
}

/**
 * `mtx_permute()' permutes the elements of a matrix or vector based
 * on given row and column permutations.
 *
 * The array `row_permutation' should be a permutation of the integers
 * `1,2,...,mtx->num_rows', and the array `column_permutation' should
 * be a permutation of the integers `1,2,...,mtx->num_columns'. The
 * elements belonging to row `i' and column `j' in the permuted matrix
 * are then equal to the elements in row `row_permutation[i-1]' and
 * column `column_permutation[j-1]' in the original matrix, for
 * `i=1,2,...,mtx->num_rows' and `j=1,2,...,mtx->num_columns'.
 */
int mtx_permute(
    struct mtx * mtx,
    const int * row_permutation,
    const int * column_permutation)
{
    if (!row_permutation && !column_permutation)
        return MTX_SUCCESS;

    if (mtx->object == mtx_matrix) {
        if (mtx->format == mtx_array) {
            struct mtx_matrix_array_data * matrix_array =
                &mtx->storage.matrix_array;
            return mtx_matrix_array_data_permute(
                matrix_array, row_permutation, column_permutation);
        } else if (mtx->format == mtx_coordinate) {
            struct mtx_matrix_coordinate_data * matrix_coordinate =
                &mtx->storage.matrix_coordinate;
            return mtx_matrix_coordinate_data_permute(
                matrix_coordinate, row_permutation, column_permutation);
        } else {
            return MTX_ERR_INVALID_MTX_FORMAT;
        }
    } else if (mtx->object == mtx_vector)  {
        if (mtx->format == mtx_array) {
            struct mtx_vector_array_data * vector_array =
                &mtx->storage.vector_array;
            return mtx_vector_array_data_permute(
                vector_array, row_permutation, column_permutation);
        } else if (mtx->format == mtx_coordinate) {
            struct mtx_vector_coordinate_data * vector_coordinate =
                &mtx->storage.vector_coordinate;
            return mtx_vector_coordinate_data_permute(
                vector_coordinate, row_permutation, column_permutation);
        } else {
            return MTX_ERR_INVALID_MTX_FORMAT;
        }
    } else {
        return MTX_ERR_INVALID_MTX_OBJECT;
    }
    return MTX_SUCCESS;
}

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
 * to which the starting vertex belongs is reordered.
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
    int ** out_permutation,
    int starting_row)
{
    int err;
    if (mtx->object != mtx_matrix)
        return MTX_ERR_INVALID_MTX_OBJECT;
    if (mtx->format != mtx_coordinate)
        return MTX_ERR_INVALID_MTX_FORMAT;

    struct mtx_matrix_coordinate_data * mtxdata =
        &mtx->storage.matrix_coordinate;
    return mtx_matrix_coordinate_data_reorder_rcm(
        mtxdata, out_permutation, starting_row);
}

/**
 * `mtx_matrix_reorder()` reorders the rows and columns of a matrix
 * according to the specified algorithm.
 *
 * Some algorithms may pose certain requirements on the matrix. For
 * example, the Reverse Cuthill-McKee ordering requires a matrix to be
 * square and in coordinate format.
 *
 * If successful, `mtx_matrix_reorder()' returns `MTX_SUCCESS', and
 * the rows and columns of mtx have been reordered. If
 * `row_permutation' is not `NULL' and the rows of a matrix were
 * indeed reordered, then `row_permutation' is set to point to a newly
 * allocated array containing the row permutation.  Furthermore, if
 * `column_permutation' is not `NULL', then `column_permutation' may
 * be set to point to an array containing the column
 * permutation. However, this is only done if the columns were also
 * reordered and the permutation is not symmetric. That is, if the row
 * and column permutations are the same, then only `row_permutation'
 * is set and `*column_permutation' is set to `NULL'.
 *
 * If either of the `row_permutation' or `column_permutation' pointers
 * are set, then the user is responsible for calling `free()' to free
 * the underlying storage.
 */
int mtx_matrix_reorder(
    struct mtx * mtx,
    int ** row_permutation,
    int ** column_permutation,
    enum mtx_ordering ordering,
    int rcm_starting_row)
{
    int err;
    if (ordering == mtx_rcm) {
        if (column_permutation)
            *column_permutation = NULL;
        return mtx_matrix_reorder_rcm(
            mtx, row_permutation, rcm_starting_row);
    } else {
        return MTX_ERR_INVALID_MTX_ORDERING;
    }
    return MTX_SUCCESS;
}
