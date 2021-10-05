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
 * Last modified: 2021-08-17
 *
 * Cuthill-McKee algorithm for reordering symmetric, sparse matrices.
 */

#include <libmtx/util/cuthill_mckee.h>
#include <libmtx/error.h>

#include <stdbool.h>
#include <stdlib.h>
#include <stdint.h>

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
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const int64_t * colptr,
    const int * rowidx,
    const int * vertex_degrees,
    int root_vertex,
    int * out_num_levels,
    int ** out_vertices_per_level_ptr,
    int ** out_vertices_per_level,
    int ** out_vertex_in_set)
{
    int err;
    bool square = num_rows == num_columns;
    int num_vertices = square ? num_rows : num_rows + num_columns;

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
    while (vertices_per_level_ptr[num_levels] < num_vertices) {

        /* 2. Loop over vertices that belong to the previous level. */
        vertices_per_level_ptr[num_levels+1] = vertices_per_level_ptr[num_levels];
        for (int i = vertices_per_level_ptr[num_levels-1];
             i < vertices_per_level_ptr[num_levels];
             i++)
        {
            int vertex = vertices_per_level[i];
            int adjacent_vertices_ptr = vertices_per_level_ptr[num_levels+1];

            /* 3. Loop over adjacent vertices. */
            int64_t rowstart = (square || vertex < num_rows) ? rowptr[vertex] : 0;
            int64_t rowend = square || vertex < num_rows ? rowptr[vertex+1] : 0;
            for (int k = rowstart; k < rowend; k++) {
                /* Subtract one to shift from 1-based column indices
                 * to 0-based numbering of vertices. */
                int adjacent_vertex = colidx[k]-1;
                if (!square)
                    adjacent_vertex += num_rows;

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

            int64_t colstart = square ? colptr[vertex]
                : (vertex >= num_rows ? colptr[vertex-num_rows] : 0);
            int64_t colend = square ? colptr[vertex+1]
                : (vertex >= num_rows ? colptr[vertex+1-num_rows] : 0);
            for (int k = colstart; k < colend; k++) {
                /* Subtract one to shift from 1-based column indices
                 * to 0-based numbering of vertices. */
                int adjacent_vertex = rowidx[k]-1;

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

        /* If no new vertices were found, then select a new root
         * vertex among those that have not been visited yet. */
        if (vertices_per_level_ptr[num_levels] ==
            vertices_per_level_ptr[num_levels+1])
        {
            int root_vertex = 0;
            for (int i = 0; i < num_vertices; i++) {
                if (!vertex_in_set[i]) {
                    root_vertex = i;
                    break;
                }
            }
            vertex_in_set[root_vertex] = 1;
            vertices_per_level[vertices_per_level_ptr[num_levels+1]] = root_vertex;
            vertices_per_level_ptr[num_levels+1]++;
        }

        num_levels++;
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
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const int64_t * colptr,
    const int * rowidx,
    const int * vertex_degrees,
    int starting_vertex,
    int * out_pseudoperipheral_vertex,
    int * out_num_levels,
    int ** out_vertices_per_level_ptr,
    int ** out_vertices_per_level,
    int ** out_vertex_in_set)
{
    int err;
    bool square = num_rows == num_columns;
    int num_vertices = square ? num_rows : num_rows + num_columns;

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
            num_rows, num_columns, rowptr, colidx, colptr, rowidx,
            vertex_degrees,
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
 * `cuthill_mckee()' uses the Cuthill-McKee algorithm to compute a
 * reordering of the vertices of an undirected graph.
 *
 * The undirected graph is described in terms of a symmetric adjacency
 * matrix in compressed sparse row (CSR) and compressed sparse column
 * (CSC) format.  The former consists of ‘num_rows+1’ row pointers,
 * ‘rowptr’, and ‘size’ column indices, ‘colidx’, whereas the latter
 * consists of ‘num_columns+1’ column pointers, ‘colptr’, and ‘size’
 * row indices, ‘rowidx’.  Note that the matrix given in CSC format is
 * equivalent to its transpose in CSR format.  Also, note that row and
 * column indices use 1-based indexing.
 *
 * On success, the array `vertex_order' will contain the new ordering
 * of the vertices (i.e., the rows of the matrix).  Therefore, it must
 * hold enough storage for at least `num_rows' values of type `int'.
 */
int cuthill_mckee(
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const int64_t * colptr,
    const int * rowidx,
    int starting_vertex,
    int size,
    int * vertex_order)
{
    int err;
    bool square = num_rows == num_columns;
    int num_vertices = square ? num_rows : num_rows + num_columns;
    if (size < num_vertices)
        return MTX_ERR_INDEX_OUT_OF_BOUNDS;

    int * vertex_degrees = malloc(num_vertices * sizeof(int));
    if (!vertex_degrees)
        return MTX_ERR_ERRNO;
    if (square) {
        for (int i = 0; i < num_rows; i++)
            vertex_degrees[i] = rowptr[i+1] - rowptr[i] + colptr[i+1] - colptr[i];
    } else {
        for (int i = 0; i < num_rows; i++)
            vertex_degrees[i] = rowptr[i+1] - rowptr[i];
        for (int j = 0; j < num_columns; j++)
            vertex_degrees[num_rows+j] = colptr[j+1] - colptr[j];
    }

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
            num_rows, vertex_degrees, &starting_vertex, NULL);
        int pseudoperipheral_vertex = 0;
        int num_levels;
        int * vertices_per_level_ptr = NULL;
        int * vertices_per_level = vertex_order;
        int * vertex_in_set = NULL;
        err = find_pseudoperipheral_vertex(
            num_rows, num_columns,
            rowptr, colidx, colptr, rowidx,
            vertex_degrees,
            starting_vertex,
            &pseudoperipheral_vertex,
            &num_levels,
            &vertices_per_level_ptr,
            &vertices_per_level,
            &vertex_in_set);
        if (err) {
            free(vertex_degrees);
            return err;
        }

        starting_vertex = pseudoperipheral_vertex;
        err = rooted_level_structure(
            num_rows, num_columns,
            rowptr, colidx, colptr, rowidx,
            vertex_degrees,
            starting_vertex,
            &num_levels,
            &vertices_per_level_ptr,
            &vertices_per_level,
            &vertex_in_set);
        if (err) {
            free(vertex_degrees);
            return err;
        }

        free(vertex_in_set);
        free(vertices_per_level_ptr);
        free(vertex_degrees);
        return MTX_SUCCESS;

    } else {

        /*
         * Compute the rooted level structure for the given starting
         * vertex. The result produces the new vertex ordering.
         */

        int num_levels;
        int * vertices_per_level_ptr = NULL;
        int * vertices_per_level = vertex_order;
        int * vertex_in_set = NULL;
        err = rooted_level_structure(
            num_rows, num_columns,
            rowptr, colidx, colptr, rowidx,
            vertex_degrees,
            starting_vertex,
            &num_levels,
            &vertices_per_level_ptr,
            &vertices_per_level,
            &vertex_in_set);
        if (err) {
            free(vertex_degrees);
            return err;
        }

        free(vertex_in_set);
        free(vertices_per_level_ptr);
        free(vertex_degrees);
        return MTX_SUCCESS;
    }
}
