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
 * Reordering the rows and columns of sparse matrices.
 */

#include <libmtx/mtx/reorder.h>

#include <libmtx/error.h>
#include <libmtx/mtx/header.h>
#include <libmtx/matrix/coordinate/coordinate.h>
#include <libmtx/mtx/matrix.h>
#include <libmtx/mtx/mtx.h>
#include <libmtx/vector/array/array.h>
#include <libmtx/vector/coordinate/coordinate.h>

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

static int mtx_permute_vector_array_real(
    struct mtx * mtx,
    const int * permutation)
{
    struct mtx orig;
    int err = mtx_copy(&orig, mtx);
    if (err)
        return err;

    const float * src = (const float *) orig.data;
    float * dst = (float *) mtx->data;
    for (int i = 0; i < mtx->size; i++)
        dst[i] = src[permutation[i]-1];
    mtx_free(&orig);
    return MTX_SUCCESS;
}

static int mtx_permute_vector_array_double(
    struct mtx * mtx,
    const int * permutation)
{
    struct mtx orig;
    int err = mtx_copy(&orig, mtx);
    if (err)
        return err;

    const double * src = (const double *) orig.data;
    double * dst = (double *) mtx->data;
    for (int i = 0; i < mtx->size; i++)
        dst[i] = src[permutation[i]-1];
    mtx_free(&orig);
    return MTX_SUCCESS;
}

static int mtx_permute_vector_array_complex(
    struct mtx * mtx,
    const int * permutation)
{
    struct mtx orig;
    int err = mtx_copy(&orig, mtx);
    if (err)
        return err;

    const float * src = (const float *) orig.data;
    float * dst = (float *) mtx->data;
    for (int i = 0; i < mtx->size; i++) {
        dst[2*i+0] = src[2*(permutation[i]-1)+0];
        dst[2*i+1] = src[2*(permutation[i]-1)+1];
    }
    mtx_free(&orig);
    return MTX_SUCCESS;
}

static int mtx_permute_vector_array_integer(
    struct mtx * mtx,
    const int * permutation)
{
    struct mtx orig;
    int err = mtx_copy(&orig, mtx);
    if (err)
        return err;

    const int * src = (const int *) orig.data;
    int * dst = (int *) mtx->data;
    for (int i = 0; i < mtx->size; i++)
        dst[i] = src[permutation[i]-1];
    mtx_free(&orig);
    return MTX_SUCCESS;
}

static int mtx_permute_vector_array(
    struct mtx * mtx,
    const int * permutation)
{
    if (mtx->field == mtx_real) {
        return mtx_permute_vector_array_real(mtx, permutation);
    } else if (mtx->field == mtx_double) {
        return mtx_permute_vector_array_double(mtx, permutation);
    } else if (mtx->field == mtx_complex) {
        return mtx_permute_vector_array_complex(mtx, permutation);
    } else if (mtx->field == mtx_integer) {
        return mtx_permute_vector_array_integer(mtx, permutation);
    } else {
        return MTX_ERR_INVALID_MTX_FIELD;
    }
    return MTX_SUCCESS;
}

static int mtx_permute_vector_coordinate_real(
    struct mtx * mtx,
    const int * permutation)
{
    struct mtx_vector_coordinate_real * data =
        (struct mtx_vector_coordinate_real *) mtx->data;
    for (int i = 0; i < mtx->size; i++)
        data[i].i = permutation[data[i].i-1];
    mtx->sorting = mtx_unsorted;
    return MTX_SUCCESS;
}

static int mtx_permute_vector_coordinate_double(
    struct mtx * mtx,
    const int * permutation)
{
    struct mtx_vector_coordinate_double * data =
        (struct mtx_vector_coordinate_double *) mtx->data;
    for (int i = 0; i < mtx->size; i++)
        data[i].i = permutation[data[i].i-1];
    mtx->sorting = mtx_unsorted;
    return MTX_SUCCESS;
}

static int mtx_permute_vector_coordinate_complex(
    struct mtx * mtx,
    const int * permutation)
{
    struct mtx_vector_coordinate_complex * data =
        (struct mtx_vector_coordinate_complex *) mtx->data;
    for (int i = 0; i < mtx->size; i++)
        data[i].i = permutation[data[i].i-1];
    mtx->sorting = mtx_unsorted;
    return MTX_SUCCESS;
}

static int mtx_permute_vector_coordinate_integer(
    struct mtx * mtx,
    const int * permutation)
{
    struct mtx_vector_coordinate_integer * data =
        (struct mtx_vector_coordinate_integer *) mtx->data;
    for (int i = 0; i < mtx->size; i++)
        data[i].i = permutation[data[i].i-1];
    mtx->sorting = mtx_unsorted;
    return MTX_SUCCESS;
}

static int mtx_permute_vector_coordinate_pattern(
    struct mtx * mtx,
    const int * permutation)
{
    struct mtx_vector_coordinate_pattern * data =
        (struct mtx_vector_coordinate_pattern *) mtx->data;
    for (int i = 0; i < mtx->size; i++)
        data[i].i = permutation[data[i].i-1];
    mtx->sorting = mtx_unsorted;
    return MTX_SUCCESS;
}

static int mtx_permute_vector_coordinate(
    struct mtx * mtx,
    const int * permutation)
{
    if (mtx->field == mtx_real) {
        return mtx_permute_vector_coordinate_real(mtx, permutation);
    } else if (mtx->field == mtx_double) {
        return mtx_permute_vector_coordinate_double(mtx, permutation);
    } else if (mtx->field == mtx_complex) {
        return mtx_permute_vector_coordinate_complex(mtx, permutation);
    } else if (mtx->field == mtx_integer) {
        return mtx_permute_vector_coordinate_integer(mtx, permutation);
    } else {
        return MTX_ERR_INVALID_MTX_FIELD;
    }
    return MTX_SUCCESS;
}

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
    const int * permutation)
{
    if (mtx->object != mtx_vector)
        return MTX_ERR_INVALID_MTX_OBJECT;

    if (mtx->format == mtx_array) {
        return mtx_permute_vector_array(mtx, permutation);
    } else if (mtx->format == mtx_coordinate) {
        return mtx_permute_vector_coordinate(mtx, permutation);
    } else {
        return MTX_ERR_INVALID_MTX_FORMAT;
    }
    return MTX_SUCCESS;
}

static int mtx_permute_matrix_array_real(
    struct mtx * mtx,
    const int * rowperm,
    const int * colperm)
{
    struct mtx orig;
    int err = mtx_copy(&orig, mtx);
    if (err)
        return err;

    const float * src = (const float *) orig.data;
    float * dst = (float *) mtx->data;
    if (rowperm && colperm) {
        for (int i = 0; i < mtx->num_rows; i++) {
            for (int j = 0; j < mtx->num_columns; j++) {
                int k = i*mtx->num_columns+j;
                int l = (rowperm[i]-1)*mtx->num_columns + colperm[j]-1;
                dst[k] = src[l];
            }
        }
    } else if (rowperm) {
        for (int i = 0; i < mtx->num_rows; i++) {
            for (int j = 0; j < mtx->num_columns; j++) {
                int k = i*mtx->num_columns+j;
                int l = (rowperm[i]-1)*mtx->num_columns + j;
                dst[k] = src[l];
            }
        }
    } else if (colperm) {
        for (int i = 0; i < mtx->num_rows; i++) {
            for (int j = 0; j < mtx->num_columns; j++) {
                int k = i*mtx->num_columns+j;
                int l = i*mtx->num_columns + colperm[j]-1;
                dst[k] = src[l];
            }
        }
    }

    mtx_free(&orig);
    return MTX_SUCCESS;
}

static int mtx_permute_matrix_array_double(
    struct mtx * mtx,
    const int * rowperm,
    const int * colperm)
{
    struct mtx orig;
    int err = mtx_copy(&orig, mtx);
    if (err)
        return err;

    const double * src = (const double *) orig.data;
    double * dst = (double *) mtx->data;
    if (rowperm && colperm) {
        for (int i = 0; i < mtx->num_rows; i++) {
            for (int j = 0; j < mtx->num_columns; j++) {
                int k = i*mtx->num_columns+j;
                int l = (rowperm[i]-1)*mtx->num_columns + colperm[j]-1;
                dst[k] = src[l];
            }
        }
    } else if (rowperm) {
        for (int i = 0; i < mtx->num_rows; i++) {
            for (int j = 0; j < mtx->num_columns; j++) {
                int k = i*mtx->num_columns+j;
                int l = (rowperm[i]-1)*mtx->num_columns + j;
                dst[k] = src[l];
            }
        }
    } else if (colperm) {
        for (int i = 0; i < mtx->num_rows; i++) {
            for (int j = 0; j < mtx->num_columns; j++) {
                int k = i*mtx->num_columns+j;
                int l = i*mtx->num_columns + colperm[j]-1;
                dst[k] = src[l];
            }
        }
    }

    mtx_free(&orig);
    return MTX_SUCCESS;
}

static int mtx_permute_matrix_array_complex(
    struct mtx * mtx,
    const int * rowperm,
    const int * colperm)
{
    struct mtx orig;
    int err = mtx_copy(&orig, mtx);
    if (err)
        return err;

    const float * src = (const float *) orig.data;
    float * dst = (float *) mtx->data;
    if (rowperm && colperm) {
        for (int i = 0; i < mtx->num_rows; i++) {
            for (int j = 0; j < mtx->num_columns; j++) {
                int k = i*mtx->num_columns+j;
                int l = (rowperm[i]-1)*mtx->num_columns + colperm[j]-1;
                dst[2*k+0] = src[2*l+0];
                dst[2*k+1] = src[2*l+1];
            }
        }
    } else if (rowperm) {
        for (int i = 0; i < mtx->num_rows; i++) {
            for (int j = 0; j < mtx->num_columns; j++) {
                int k = i*mtx->num_columns+j;
                int l = (rowperm[i]-1)*mtx->num_columns + j;
                dst[2*k+0] = src[2*l+0];
                dst[2*k+1] = src[2*l+1];
            }
        }
    } else if (colperm) {
        for (int i = 0; i < mtx->num_rows; i++) {
            for (int j = 0; j < mtx->num_columns; j++) {
                int k = i*mtx->num_columns+j;
                int l = i*mtx->num_columns + colperm[j]-1;
                dst[2*k+0] = src[2*l+0];
                dst[2*k+1] = src[2*l+1];
            }
        }
    }

    mtx_free(&orig);
    return MTX_SUCCESS;
}

static int mtx_permute_matrix_array_integer(
    struct mtx * mtx,
    const int * rowperm,
    const int * colperm)
{
    struct mtx orig;
    int err = mtx_copy(&orig, mtx);
    if (err)
        return err;

    const int * src = (const int *) orig.data;
    int * dst = (int *) mtx->data;
    if (rowperm && colperm) {
        for (int i = 0; i < mtx->num_rows; i++) {
            for (int j = 0; j < mtx->num_columns; j++) {
                int k = i*mtx->num_columns+j;
                int l = (rowperm[i]-1)*mtx->num_columns + colperm[j]-1;
                dst[k] = src[l];
            }
        }
    } else if (rowperm) {
        for (int i = 0; i < mtx->num_rows; i++) {
            for (int j = 0; j < mtx->num_columns; j++) {
                int k = i*mtx->num_columns+j;
                int l = (rowperm[i]-1)*mtx->num_columns + j;
                dst[k] = src[l];
            }
        }
    } else if (colperm) {
        for (int i = 0; i < mtx->num_rows; i++) {
            for (int j = 0; j < mtx->num_columns; j++) {
                int k = i*mtx->num_columns+j;
                int l = i*mtx->num_columns + colperm[j]-1;
                dst[k] = src[l];
            }
        }
    }

    mtx_free(&orig);
    return MTX_SUCCESS;
}

static int mtx_permute_matrix_array(
    struct mtx * mtx,
    const int * row_permutation,
    const int * column_permutation)
{
    if (mtx->field == mtx_real) {
        return mtx_permute_matrix_array_real(
            mtx, row_permutation, column_permutation);
    } else if (mtx->field == mtx_double) {
        return mtx_permute_matrix_array_double(
            mtx, row_permutation, column_permutation);
    } else if (mtx->field == mtx_complex) {
        return mtx_permute_matrix_array_complex(
            mtx, row_permutation, column_permutation);
    } else if (mtx->field == mtx_integer) {
        return mtx_permute_matrix_array_integer(
            mtx, row_permutation, column_permutation);
    } else {
        return MTX_ERR_INVALID_MTX_FIELD;
    }
    return MTX_SUCCESS;
}

static int mtx_permute_matrix_coordinate_real(
    struct mtx * mtx,
    const int * rowperm,
    const int * colperm)
{
    struct mtx_matrix_coordinate_real * data =
        (struct mtx_matrix_coordinate_real *) mtx->data;
    if (rowperm && colperm) {
        for (int64_t k = 0; k < mtx->size; k++) {
            data[k].i = rowperm[data[k].i-1];
            data[k].j = colperm[data[k].j-1];
        }
    } else if (rowperm) {
        for (int64_t k = 0; k < mtx->size; k++) {
            data[k].i = rowperm[data[k].i-1];
        }
    } else if (colperm) {
        for (int64_t k = 0; k < mtx->size; k++) {
            data[k].j = colperm[data[k].j-1];
        }
    }
    mtx->sorting = mtx_unsorted;
    return MTX_SUCCESS;
}

static int mtx_permute_matrix_coordinate_double(
    struct mtx * mtx,
    const int * rowperm,
    const int * colperm)
{
    struct mtx_matrix_coordinate_double * data =
        (struct mtx_matrix_coordinate_double *) mtx->data;
    if (rowperm && colperm) {
        for (int64_t k = 0; k < mtx->size; k++) {
            data[k].i = rowperm[data[k].i-1];
            data[k].j = colperm[data[k].j-1];
        }
    } else if (rowperm) {
        for (int64_t k = 0; k < mtx->size; k++) {
            data[k].i = rowperm[data[k].i-1];
        }
    } else if (colperm) {
        for (int64_t k = 0; k < mtx->size; k++) {
            data[k].j = colperm[data[k].j-1];
        }
    }
    mtx->sorting = mtx_unsorted;
    return MTX_SUCCESS;
}

static int mtx_permute_matrix_coordinate_complex(
    struct mtx * mtx,
    const int * rowperm,
    const int * colperm)
{
    struct mtx_matrix_coordinate_complex * data =
        (struct mtx_matrix_coordinate_complex *) mtx->data;
    if (rowperm && colperm) {
        for (int64_t k = 0; k < mtx->size; k++) {
            data[k].i = rowperm[data[k].i-1];
            data[k].j = colperm[data[k].j-1];
        }
    } else if (rowperm) {
        for (int64_t k = 0; k < mtx->size; k++) {
            data[k].i = rowperm[data[k].i-1];
        }
    } else if (colperm) {
        for (int64_t k = 0; k < mtx->size; k++) {
            data[k].j = colperm[data[k].j-1];
        }
    }
    mtx->sorting = mtx_unsorted;
    return MTX_SUCCESS;
}

static int mtx_permute_matrix_coordinate_integer(
    struct mtx * mtx,
    const int * rowperm,
    const int * colperm)
{
    struct mtx_matrix_coordinate_integer * data =
        (struct mtx_matrix_coordinate_integer *) mtx->data;
    if (rowperm && colperm) {
        for (int64_t k = 0; k < mtx->size; k++) {
            data[k].i = rowperm[data[k].i-1];
            data[k].j = colperm[data[k].j-1];
        }
    } else if (rowperm) {
        for (int64_t k = 0; k < mtx->size; k++) {
            data[k].i = rowperm[data[k].i-1];
        }
    } else if (colperm) {
        for (int64_t k = 0; k < mtx->size; k++) {
            data[k].j = colperm[data[k].j-1];
        }
    }
    mtx->sorting = mtx_unsorted;
    return MTX_SUCCESS;
}

static int mtx_permute_matrix_coordinate_pattern(
    struct mtx * mtx,
    const int * rowperm,
    const int * colperm)
{
    struct mtx_matrix_coordinate_pattern * data =
        (struct mtx_matrix_coordinate_pattern *) mtx->data;
    if (rowperm && colperm) {
        for (int64_t k = 0; k < mtx->size; k++) {
            data[k].i = rowperm[data[k].i-1];
            data[k].j = colperm[data[k].j-1];
        }
    } else if (rowperm) {
        for (int64_t k = 0; k < mtx->size; k++) {
            data[k].i = rowperm[data[k].i-1];
        }
    } else if (colperm) {
        for (int64_t k = 0; k < mtx->size; k++) {
            data[k].j = colperm[data[k].j-1];
        }
    }
    mtx->sorting = mtx_unsorted;
    return MTX_SUCCESS;
}

static int mtx_permute_matrix_coordinate(
    struct mtx * mtx,
    const int * row_permutation,
    const int * column_permutation)
{
    if (mtx->field == mtx_real) {
        return mtx_permute_matrix_coordinate_real(
            mtx, row_permutation, column_permutation);
    } else if (mtx->field == mtx_double) {
        return mtx_permute_matrix_coordinate_double(
            mtx, row_permutation, column_permutation);
    } else if (mtx->field == mtx_complex) {
        return mtx_permute_matrix_coordinate_complex(
            mtx, row_permutation, column_permutation);
    } else if (mtx->field == mtx_integer) {
        return mtx_permute_matrix_coordinate_integer(
            mtx, row_permutation, column_permutation);
    } else if (mtx->field == mtx_pattern) {
        return mtx_permute_matrix_coordinate_pattern(
            mtx, row_permutation, column_permutation);
    } else {
        return MTX_ERR_INVALID_MTX_FIELD;
    }
    return MTX_SUCCESS;
}

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
    const int * column_permutation)
{
    if (mtx->object != mtx_matrix)
        return MTX_ERR_INVALID_MTX_OBJECT;

    if (!row_permutation && !column_permutation)
        return MTX_SUCCESS;

    if (mtx->format == mtx_array) {
        return mtx_permute_matrix_array(
            mtx, row_permutation, column_permutation);
    } else if (mtx->format == mtx_coordinate) {
        return mtx_permute_matrix_coordinate(
            mtx, row_permutation, column_permutation);
    } else {
        return MTX_ERR_INVALID_MTX_FORMAT;
    }
    return MTX_SUCCESS;
}

/**
 * `minimum_degree_vertex()` finds a vertex of minimum degree in a
 * graph.
 */
static int minimum_degree_vertex(
    int num_vertices,
    const int * vertex_degrees,
    int * out_vertex,
    int * out_degree)
{
    int err;
    if (num_vertices == 0) {
        *out_vertex = -1;
        *out_degree = -1;
        return MTX_SUCCESS;
    }

    int min_degree_vertex = 0;
    int min_degree = vertex_degrees[min_degree_vertex];
    for (int vertex = 1; vertex < num_vertices; vertex++) {
        int degree = vertex_degrees[vertex];
        if (min_degree > degree) {
            min_degree = degree;
            min_degree_vertex = vertex;
        }
    }
    if (out_vertex)
        *out_vertex = min_degree_vertex;
    if (out_degree)
        *out_degree = min_degree;
    return MTX_SUCCESS;
}

/**
 * `rooted_level_structure()` computes the rooted level structure at a
 * given vertex.
 *
 * A rooted level structure is a partitioning of the vertex set of a
 * graph into levels, such that each level consists of vertices whose
 * distance from the root is equal to the level number.
 */
static int rooted_level_structure(
    const struct mtx * mtx,
    const int64_t * row_ptr,
    const int * vertex_degrees,
    int root_vertex,
    int * out_num_levels,
    int ** out_vertices_per_level_ptr,
    int ** out_vertices_per_level,
    int ** out_vertex_in_set)
{
    int err;
    int num_vertices = mtx->num_rows;

    /* Reuse arrays that were passed in as function arguments. */
    int * vertices_per_level_ptr = *out_vertices_per_level_ptr;
    int * vertices_per_level = *out_vertices_per_level;
    int * vertex_in_set = *out_vertex_in_set;

    /* Check if we need to allocate storage for any of the arrays. */
    bool alloc_vertices_per_level_ptr = (vertices_per_level_ptr == NULL);
    bool alloc_vertices_per_level = (vertices_per_level == NULL);
    bool alloc_vertex_in_set = (vertex_in_set == NULL);

    /* Allocate storage for offsets to the start of each level of the
     * rooted level structure. */
    if (alloc_vertices_per_level_ptr) {
        vertices_per_level_ptr = (int *) malloc((num_vertices+1) * sizeof(int));
        if (!vertices_per_level_ptr)
            return MTX_ERR_ERRNO;
        for (int i = 0; i < num_vertices+1; i++)
            vertices_per_level_ptr[i] = 0;
    }

    /* Allocate storage for the vertices in each level of the rooted
     * level structure. */
    if (alloc_vertices_per_level) {
        vertices_per_level = (int *) malloc(num_vertices * sizeof(int));
        if (!vertices_per_level) {
            if (alloc_vertices_per_level_ptr)
                free(vertices_per_level_ptr);
            return MTX_ERR_ERRNO;
        }
        for (int i = 0; i < num_vertices; i++)
            vertices_per_level[i] = 0;
    }

    /* Allocate storage to mark whether or not each vertex has
     * been added to the rooted level structure. */
    if (alloc_vertex_in_set) {
        vertex_in_set = (int *) malloc(num_vertices * sizeof(int));
        if (!vertex_in_set) {
            if (alloc_vertices_per_level)
                free(vertices_per_level);
            if (alloc_vertices_per_level_ptr)
                free(vertices_per_level_ptr);
            return MTX_ERR_ERRNO;
        }
    }

    /* Clear the array that is used to test set membership. */
    for (int i = 0; i < num_vertices; i++)
        vertex_in_set[i] = 0;

    /* Handle empty graphs. */
    if (num_vertices == 0) {
        vertices_per_level_ptr[0] = 0;
        *out_num_levels = 0;
        *out_vertices_per_level_ptr = vertices_per_level_ptr;
        *out_vertices_per_level = vertices_per_level;
        *out_vertex_in_set = vertex_in_set;
        return MTX_SUCCESS;
    }

    /* Add the root vertex to the first level. */
    vertex_in_set[root_vertex] = 1;
    vertices_per_level[0] = root_vertex;
    vertices_per_level_ptr[0] = 0;
    vertices_per_level_ptr[1] = 1;

    /* 1. Loop over the levels of the structure. */
    int num_levels = 1;
    for (; num_levels < num_vertices; num_levels++) {

        /* 2. Loop over vertices that belong to the previous level. */
        vertices_per_level_ptr[num_levels+1] = vertices_per_level_ptr[num_levels];
        for (int i = vertices_per_level_ptr[num_levels-1];
             i < vertices_per_level_ptr[num_levels];
             i++)
        {
            int vertex = vertices_per_level[i];

            /* 3. Loop over adjacent vertices. */
            int adjacent_vertices_ptr = vertices_per_level_ptr[num_levels+1];
            for (int k = row_ptr[vertex]; k < row_ptr[vertex+1]; k++) {
                int adjacent_vertex;
                err = mtx_matrix_column_index(mtx, k, &adjacent_vertex);
                if (err) {
                    if (alloc_vertex_in_set)
                        free(vertex_in_set);
                    if (alloc_vertices_per_level)
                        free(vertices_per_level);
                    if (alloc_vertices_per_level_ptr)
                        free(vertices_per_level_ptr);
                    return err;
                }
                /* Subtract one to shift from 1-based column indices
                 * to 0-based numbering of vertices. */
                adjacent_vertex -= 1;

                if (!vertex_in_set[adjacent_vertex]) {

                    /*
                     * Now, we have found a vertex that is adjacent to
                     * a vertex in the current level, but which does
                     * not belong to the rooted level structure
                     * itself.
                     *
                     * Next, use an insertion sort to insert the new
                     * vertex into the list of vertices that are
                     * adjacent to the vertex in the current level,
                     * which is sorted according to degree.
                     */

                    vertex_in_set[adjacent_vertex] = 1;
                    int adjacent_vertex_degree = vertex_degrees[adjacent_vertex];
                    int j = vertices_per_level_ptr[num_levels+1] - 1;
                    while (j >= adjacent_vertices_ptr &&
                           (vertex_degrees[vertices_per_level[j]] <
                            adjacent_vertex_degree))
                    {
                        vertices_per_level[j+1] = vertices_per_level[j];
                        j--;
                    }
                    vertices_per_level[j+1] = adjacent_vertex;
                    vertices_per_level_ptr[num_levels+1]++;
                }
            }
        }

        /* Stop if no new vertices were added. */
        if (vertices_per_level_ptr[num_levels] ==
            vertices_per_level_ptr[num_levels+1])
            break;
    }

    *out_num_levels = num_levels;
    *out_vertices_per_level_ptr = vertices_per_level_ptr;
    *out_vertices_per_level = vertices_per_level;
    *out_vertex_in_set = vertex_in_set;
    return MTX_SUCCESS;
}

/**
 * `find_pseudoperipheral_vertex()` finds a pseudo-peripheral vertex
 * in an undirected graph.
 */
static int find_pseudoperipheral_vertex(
    const struct mtx * mtx,
    const int64_t * row_ptr,
    const int * vertex_degrees,
    int starting_vertex,
    int * out_pseudoperipheral_vertex,
    int * out_num_levels,
    int ** out_vertices_per_level_ptr,
    int ** out_vertices_per_level,
    int ** out_vertex_in_set)
{
    int err;
    int num_vertices = mtx->num_rows;

    /* Reuse arrays that were passed in as function arguments. */
    int * vertices_per_level_ptr = *out_vertices_per_level_ptr;
    int * vertices_per_level = *out_vertices_per_level;
    int * vertex_in_set = *out_vertex_in_set;

    /* Check if we need to allocate storage for any of the arrays. */
    bool alloc_vertices_per_level_ptr = (vertices_per_level_ptr == NULL);
    bool alloc_vertices_per_level = (vertices_per_level == NULL);
    bool alloc_vertex_in_set = (vertex_in_set == NULL);

    /* Allocate storage for offsets to the start of each level of the
     * rooted level structure. */
    if (alloc_vertices_per_level_ptr) {
        vertices_per_level_ptr = (int *) malloc((num_vertices+1) * sizeof(int));
        if (!vertices_per_level_ptr)
            return MTX_ERR_ERRNO;
        for (int i = 0; i < num_vertices+1; i++)
            vertices_per_level_ptr[i] = 0;
    }

    /* Allocate storage for the vertices in each level of the rooted
     * level structure. */
    if (alloc_vertices_per_level) {
        vertices_per_level = (int *) malloc(num_vertices * sizeof(int));
        if (!vertices_per_level) {
            if (alloc_vertices_per_level_ptr)
                free(vertices_per_level_ptr);
            return MTX_ERR_ERRNO;
        }
        for (int i = 0; i < num_vertices; i++)
            vertices_per_level[i] = 0;
    }

    /* Allocate storage to mark whether or not each vertex has
     * been added to the rooted level structure. */
    if (alloc_vertex_in_set) {
        vertex_in_set = (int *) malloc(num_vertices * sizeof(int));
        if (!vertex_in_set) {
            if (alloc_vertices_per_level)
                free(vertices_per_level);
            if (alloc_vertices_per_level_ptr)
                free(vertices_per_level_ptr);
            return MTX_ERR_ERRNO;
        }
        for (int i = 0; i < num_vertices; i++)
            vertex_in_set[i] = 0;
    }

    /* Handle empty graphs. */
    if (num_vertices == 0) {
        *out_vertices_per_level_ptr = vertices_per_level_ptr;
        *out_vertices_per_level = vertices_per_level;
        *out_vertex_in_set = vertex_in_set;
        return 0;
    }

    /* 1. Set the root vertex. */
    int vertex = starting_vertex;
    int num_vertex_levels = 0;
    int num_root_vertex_levels = -1;

    /* 2. Continue while the eccentricity of the current vertex is
     * greater than that of the root vertex.  */
    while (num_vertex_levels > num_root_vertex_levels) {
        num_root_vertex_levels = num_vertex_levels;

        /* 3. Construct the rooted level structure for the current vertex. */
        err = rooted_level_structure(
            mtx, row_ptr, vertex_degrees,
            vertex,
            &num_vertex_levels,
            &vertices_per_level_ptr,
            &vertices_per_level,
            &vertex_in_set);
        if (err) {
            if (alloc_vertex_in_set)
                free(vertex_in_set);
            if (alloc_vertices_per_level)
                free(vertices_per_level);
            if (alloc_vertices_per_level_ptr)
                free(vertices_per_level_ptr);
            return err;
        }

        /* 4. Shrink the last level by selecting a vertex of minimum degree. */
        int min_degree_vertex = vertices_per_level[
            vertices_per_level_ptr[num_vertex_levels-1]];
        int min_degree = vertex_degrees[vertex];
        for (int i = vertices_per_level_ptr[num_vertex_levels-1];
             i < vertices_per_level_ptr[num_vertex_levels]; i++)
        {
            int vertex = vertices_per_level[i];
            int degree = vertex_degrees[vertex];
            if (min_degree > degree) {
                min_degree = degree;
                min_degree_vertex = vertex;
            }
        }
        vertex = min_degree_vertex;
    }

    *out_pseudoperipheral_vertex = vertex;
    *out_num_levels = num_vertex_levels;
    *out_vertices_per_level_ptr = vertices_per_level_ptr;
    *out_vertices_per_level = vertices_per_level;
    *out_vertex_in_set = vertex_in_set;
    return 0;
}

/**
 * `cuthill_mckee()` uses the Cuthill-McKee algorithm to compute a
 * reordering of the vertices of an undirected graph.
 */
static int cuthill_mckee(
    const struct mtx * mtx,
    const int64_t * row_ptr,
    const int * vertex_degrees,
    int starting_vertex,
    int ** out_vertex_order)
{
    int err;
    if (starting_vertex == -1) {

        /*
         * Find a pseudo-peripheral vertex to use as the starting
         * vertex for the Cuthill-McKee algorithm.
         *
         * Note that the procedure for computing a pseudo-peripheral
         * vertex already computes the rooted level structure for the
         * vertex, and this structure is in fact the reordering that
         * results from the Cuthill-McKee algorithm.
         */

        err = minimum_degree_vertex(
            mtx->num_rows, vertex_degrees, &starting_vertex, NULL);
        int pseudoperipheral_vertex = 0;
        int num_levels;
        int * vertices_per_level_ptr = NULL;
        int * vertices_per_level = NULL;
        int * vertex_in_set = NULL;
        err = find_pseudoperipheral_vertex(
            mtx, row_ptr, vertex_degrees,
            starting_vertex,
            &pseudoperipheral_vertex,
            &num_levels,
            &vertices_per_level_ptr,
            &vertices_per_level,
            &vertex_in_set);
        if (err)
            return err;

        starting_vertex = pseudoperipheral_vertex;
        err = rooted_level_structure(
            mtx, row_ptr, vertex_degrees,
            starting_vertex,
            &num_levels,
            &vertices_per_level_ptr,
            &vertices_per_level,
            &vertex_in_set);
        if (err)
            return err;

        free(vertices_per_level_ptr);
        free(vertex_in_set);
        *out_vertex_order = vertices_per_level;
        return MTX_SUCCESS;
    } else {

        /*
         * Compute the rooted level structure for the given starting
         * vertex. The result produces the new vertex ordering.
         */

        int num_levels;
        int * vertices_per_level_ptr = NULL;
        int * vertices_per_level = NULL;
        int * vertex_in_set = NULL;
        err = rooted_level_structure(
            mtx, row_ptr, vertex_degrees,
            starting_vertex,
            &num_levels,
            &vertices_per_level_ptr,
            &vertices_per_level,
            &vertex_in_set);
        if (err)
            return err;

        free(vertex_in_set);
        free(vertices_per_level_ptr);
        *out_vertex_order = vertices_per_level;
        return MTX_SUCCESS;
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
    if (mtx->num_rows != mtx->num_columns)
        return MTX_ERR_INVALID_MTX_SIZE;
    if (mtx->sorting != mtx_row_major)
        return MTX_ERR_INVALID_MTX_SORTING;
    if (starting_row < 0 || starting_row > mtx->num_rows) {
        return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    }

    /* 1. Allocate storage for and compute row pointers. */
    int64_t * row_ptr = malloc((mtx->num_rows+1) * sizeof(int64_t));
    if (!row_ptr)
        return MTX_ERR_ERRNO;
    err = mtx_matrix_row_ptr(mtx, row_ptr);
    if (err) {
        free(row_ptr);
        return err;
    }

    /* 2. Allocate storage for and compute vertex degrees. */
    int * vertex_degrees = malloc(mtx->num_rows * sizeof(int));
    if (!vertex_degrees) {
        free(row_ptr);
        return MTX_ERR_ERRNO;
    }
    err = mtx_matrix_diagonal_size_per_row(mtx, vertex_degrees);
    if (err) {
        free(vertex_degrees);
        free(row_ptr);
        return err;
    }
    for (int i = 0; i < mtx->num_rows; i++)
        vertex_degrees[i] += row_ptr[i+1] - row_ptr[i];

    /* 3. Compute the Cuthill-McKee ordering. */
    int * vertex_order;
    err = cuthill_mckee(
        mtx, row_ptr, vertex_degrees, starting_row-1, &vertex_order);
    if (err) {
        free(vertex_degrees);
        free(row_ptr);
        return err;
    }

    free(vertex_degrees);
    free(row_ptr);

    /* Add one to shift from 0-based to 1-based indexing. */
    for (int i = 0; i < mtx->num_rows; i++)
        vertex_order[i]++;

    /* 4. Reverse the ordering. */
    for (int i = 0; i < mtx->num_rows/2; i++) {
        int tmp = vertex_order[i];
        vertex_order[i] = vertex_order[mtx->num_rows-i-1];
        vertex_order[mtx->num_rows-i-1] = tmp;
    }

    int * permutation = malloc(mtx->num_rows * sizeof(int));
    if (err) {
        free(vertex_order);
        return err;
    }

    for (int i = 0; i < mtx->num_rows; i++)
        permutation[vertex_order[i]-1] = i+1;
    free(vertex_order);

    /* 5. Permute the matrix. */
    err = mtx_permute_matrix(mtx, permutation, permutation);
    if (err) {
        free(vertex_order);
        return err;
    }

    if (out_permutation)
        *out_permutation = permutation;
    else
        free(permutation);
    mtx->ordering = mtx_rcm;
    return MTX_SUCCESS;
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
        if (mtx->sorting != mtx_row_major) {
            err = mtx_sort(mtx, mtx_row_major);
            if (err)
                return err;
        }
        return mtx_matrix_reorder_rcm(
            mtx, row_permutation, rcm_starting_row);
    } else {
        return MTX_ERR_INVALID_MTX_ORDERING;
    }
    return MTX_SUCCESS;
}
