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
 * Reordering the rows and columns of matrices in coordinate format.
 */

#include <libmtx/error.h>
#include <libmtx/matrix/coordinate/data.h>
#include <libmtx/matrix/coordinate/reorder.h>
#include <libmtx/matrix/coordinate/sort.h>
#include <libmtx/util/cuthill_mckee.h>

#include <errno.h>

#include <stdlib.h>

/**
 * `mtx_matrix_coordinate_data_permute()' permutes the elements of a
 * matrix in coordinate format based on a given permutation.
 *
 * The array `rowperm' should be a permutation of the integers
 * `1,2,...,mtxdata->num_rows', and the array `colperm' should be a
 * permutation of the integers `1,2,...,mtxdata->num_columns'. The
 * elements belonging to row `i' and column `j' in the permuted matrix
 * are then equal to the elements in row `rowperm[i-1]' and column
 * `colperm[j-1]' in the original matrix, for
 * `i=1,2,...,mtxdata->num_rows' and `j=1,2,...,mtxdata->num_columns'.
 */
int mtx_matrix_coordinate_data_permute(
    struct mtx_matrix_coordinate_data * mtxdata,
    const int * rowperm,
    const int * colperm)
{
    if (mtxdata->field == mtx_real) {
        if (mtxdata->precision == mtx_single) {
            struct mtx_matrix_coordinate_real_single * data =
                mtxdata->data.real_single;
            if (rowperm && colperm) {
                for (int64_t k = 0; k < mtxdata->size; k++) {
                    data[k].i = rowperm[data[k].i-1];
                    data[k].j = colperm[data[k].j-1];
                }
            } else if (rowperm) {
                for (int64_t k = 0; k < mtxdata->size; k++) {
                    data[k].i = rowperm[data[k].i-1];
                }
            } else if (colperm) {
                for (int64_t k = 0; k < mtxdata->size; k++) {
                    data[k].j = colperm[data[k].j-1];
                }
            }

        } else if (mtxdata->precision == mtx_double) {
            struct mtx_matrix_coordinate_real_double * data =
                mtxdata->data.real_double;
            if (rowperm && colperm) {
                for (int64_t k = 0; k < mtxdata->size; k++) {
                    data[k].i = rowperm[data[k].i-1];
                    data[k].j = colperm[data[k].j-1];
                }
            } else if (rowperm) {
                for (int64_t k = 0; k < mtxdata->size; k++) {
                    data[k].i = rowperm[data[k].i-1];
                }
            } else if (colperm) {
                for (int64_t k = 0; k < mtxdata->size; k++) {
                    data[k].j = colperm[data[k].j-1];
                }
            }

        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (mtxdata->field == mtx_complex) {
        if (mtxdata->precision == mtx_single) {
            struct mtx_matrix_coordinate_complex_single * data =
                mtxdata->data.complex_single;
            if (rowperm && colperm) {
                for (int64_t k = 0; k < mtxdata->size; k++) {
                    data[k].i = rowperm[data[k].i-1];
                    data[k].j = colperm[data[k].j-1];
                }
            } else if (rowperm) {
                for (int64_t k = 0; k < mtxdata->size; k++) {
                    data[k].i = rowperm[data[k].i-1];
                }
            } else if (colperm) {
                for (int64_t k = 0; k < mtxdata->size; k++) {
                    data[k].j = colperm[data[k].j-1];
                }
            }

        } else if (mtxdata->precision == mtx_double) {
            struct mtx_matrix_coordinate_complex_double * data =
                mtxdata->data.complex_double;
            if (rowperm && colperm) {
                for (int64_t k = 0; k < mtxdata->size; k++) {
                    data[k].i = rowperm[data[k].i-1];
                    data[k].j = colperm[data[k].j-1];
                }
            } else if (rowperm) {
                for (int64_t k = 0; k < mtxdata->size; k++) {
                    data[k].i = rowperm[data[k].i-1];
                }
            } else if (colperm) {
                for (int64_t k = 0; k < mtxdata->size; k++) {
                    data[k].j = colperm[data[k].j-1];
                }
            }

        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (mtxdata->field == mtx_integer) {
        if (mtxdata->precision == mtx_single) {
            struct mtx_matrix_coordinate_integer_single * data =
                mtxdata->data.integer_single;
            if (rowperm && colperm) {
                for (int64_t k = 0; k < mtxdata->size; k++) {
                    data[k].i = rowperm[data[k].i-1];
                    data[k].j = colperm[data[k].j-1];
                }
            } else if (rowperm) {
                for (int64_t k = 0; k < mtxdata->size; k++) {
                    data[k].i = rowperm[data[k].i-1];
                }
            } else if (colperm) {
                for (int64_t k = 0; k < mtxdata->size; k++) {
                    data[k].j = colperm[data[k].j-1];
                }
            }

        } else if (mtxdata->precision == mtx_double) {
            struct mtx_matrix_coordinate_integer_double * data =
                mtxdata->data.integer_double;
            if (rowperm && colperm) {
                for (int64_t k = 0; k < mtxdata->size; k++) {
                    data[k].i = rowperm[data[k].i-1];
                    data[k].j = colperm[data[k].j-1];
                }
            } else if (rowperm) {
                for (int64_t k = 0; k < mtxdata->size; k++) {
                    data[k].i = rowperm[data[k].i-1];
                }
            } else if (colperm) {
                for (int64_t k = 0; k < mtxdata->size; k++) {
                    data[k].j = colperm[data[k].j-1];
                }
            }

        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (mtxdata->field == mtx_pattern) {
        struct mtx_matrix_coordinate_pattern * data =
            mtxdata->data.pattern;
        if (rowperm && colperm) {
            for (int64_t k = 0; k < mtxdata->size; k++) {
                data[k].i = rowperm[data[k].i-1];
                data[k].j = colperm[data[k].j-1];
            }
        } else if (rowperm) {
            for (int64_t k = 0; k < mtxdata->size; k++) {
                data[k].i = rowperm[data[k].i-1];
            }
        } else if (colperm) {
            for (int64_t k = 0; k < mtxdata->size; k++) {
                data[k].j = colperm[data[k].j-1];
            }
        }

    } else {
        return MTX_ERR_INVALID_MTX_FIELD;
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_matrix_coordinate_data_reorder_rcm()' reorders the rows of a
 * symmetric sparse matrix according to the Reverse Cuthill-McKee
 * algorithm.
 *
 * `starting_row' is an integer in the range `[1,mtxdata->num_rows]',
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
 * and the rows and columns of `mtxdata' have been reordered according
 * to the Reverse Cuthill-McKee algorithm. If `permutation' is not
 * `NULL', then the underlying pointer is set to point to a newly
 * allocated array containing the permutation used to reorder the rows
 * and columns of `mtxdata'. In this case, the user is responsible for
 * calling `free()' to free the underlying storage.
 */
int mtx_matrix_coordinate_data_reorder_rcm(
    struct mtx_matrix_coordinate_data * mtxdata,
    int ** out_permutation,
    int starting_row)
{
    int err;
    if (mtxdata->num_rows != mtxdata->num_columns)
        return MTX_ERR_INVALID_MTX_SIZE;
    if (starting_row < 0 || starting_row > mtxdata->num_rows)
        return MTX_ERR_INDEX_OUT_OF_BOUNDS;

    if (mtxdata->sorting != mtx_row_major) {
        err = mtx_matrix_coordinate_data_sort(
            mtxdata, mtx_row_major);
        if (err)
            return err;
    }

    /* 1. Allocate storage for and compute row pointers. */
    int64_t * row_ptr = malloc((mtxdata->num_rows+1) * sizeof(int64_t));
    if (!row_ptr)
        return MTX_ERR_ERRNO;
    err = mtx_matrix_coordinate_data_row_ptr(
        mtxdata, mtxdata->num_rows+1, row_ptr);
    if (err) {
        free(row_ptr);
        return err;
    }

    /* 2. Allocate storage for and extract column indices. */
    int * column_indices = malloc(mtxdata->size * sizeof(int));
    if (!column_indices) {
        free(row_ptr);
        return MTX_ERR_ERRNO;
    }
    err = mtx_matrix_coordinate_data_column_indices(
        mtxdata, mtxdata->size, column_indices);
    if (err) {
        free(column_indices);
        free(row_ptr);
        return err;
    }

    /* 1b. Allocate storage for and compute column pointers. */
    int64_t * col_ptr = malloc((mtxdata->num_columns+1) * sizeof(int64_t));
    if (!col_ptr) {
        free(column_indices);
        free(row_ptr);
        return MTX_ERR_ERRNO;
    }
    err = mtx_matrix_coordinate_data_column_ptr(
        mtxdata, mtxdata->num_columns+1, col_ptr);
    if (err) {
        free(col_ptr);
        free(column_indices);
        free(row_ptr);
        return err;
    }

    /* 2b. Allocate storage for and extract row indices. */
    int * row_indices = malloc(mtxdata->size * sizeof(int));
    if (!row_indices) {
        free(col_ptr);
        free(column_indices);
        free(row_ptr);
        return MTX_ERR_ERRNO;
    }
    err = mtx_matrix_coordinate_data_row_indices(
        mtxdata, mtxdata->size, column_indices);
    if (err) {
        free(row_indices);
        free(col_ptr);
        free(column_indices);
        free(row_ptr);
        return err;
    }

    /* 3. Allocate storage for the vertex ordering. */
    int * vertex_order = malloc(mtxdata->num_rows * sizeof(int));
    if (!vertex_order) {
        free(row_indices);
        free(col_ptr);
        free(column_indices);
        free(row_ptr);
        return MTX_ERR_ERRNO;
    }

    /* 5. Compute the Cuthill-McKee ordering. */
    starting_row -= 1;
    err = cuthill_mckee(
        mtxdata->num_rows, mtxdata->num_columns, row_ptr, column_indices,
        col_ptr, row_indices, &starting_row, mtxdata->num_rows, vertex_order);
    if (err) {
        free(vertex_order);
        free(row_indices);
        free(col_ptr);
        free(column_indices);
        free(row_ptr);
        return err;
    }

    free(row_indices);
    free(col_ptr);
    free(column_indices);
    free(row_ptr);

    /* Add one to shift from 0-based to 1-based indexing. */
    for (int i = 0; i < mtxdata->num_rows; i++)
        vertex_order[i]++;

    /* 6. Reverse the ordering. */
    for (int i = 0; i < mtxdata->num_rows/2; i++) {
        int tmp = vertex_order[i];
        vertex_order[i] = vertex_order[mtxdata->num_rows-i-1];
        vertex_order[mtxdata->num_rows-i-1] = tmp;
    }

    int * permutation = malloc(mtxdata->num_rows * sizeof(int));
    if (!permutation) {
        free(vertex_order);
        return err;
    }

    for (int i = 0; i < mtxdata->num_rows; i++)
        permutation[vertex_order[i]-1] = i+1;
    free(vertex_order);

    /* 7. Permute the matrix. */
    err = mtx_matrix_coordinate_data_permute(
        mtxdata, permutation, permutation);
    if (err) {
        free(vertex_order);
        return err;
    }

    if (out_permutation)
        *out_permutation = permutation;
    else
        free(permutation);
    return MTX_SUCCESS;
}
