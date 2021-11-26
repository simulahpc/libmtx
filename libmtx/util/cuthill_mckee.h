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

#ifndef LIBMTX_UTIL_CUTHILL_MCKEE
#define LIBMTX_UTIL_CUTHILL_MCKEE

#include <stdint.h>

/**
 * ‘cuthill_mckee()’ uses the Cuthill-McKee algorithm to compute a
 * reordering of the vertices of an undirected graph.
 *
 * The undirected graph is described in terms of a symmetric adjacency
 * matrix in compressed sparse row (CSR) and compressed sparse column
 * (CSC) format.  The former consists of ‘num_rows+1’ row pointers,
 * ‘rowptr’, and ‘size’ column indices, ‘colidx’, whereas the latter
 * consists of ‘num_columns+1’ column pointers, ‘colptr’, and ‘size’
 * row indices, ‘rowidx’.  Note that the matrix given in CSC format is
 * equivalent to its transpose in CSR format.  Also, note that row and
 * column indices use 1-based indexing, as in the Matrix Market
 * format.
 *
 * For a square matrix, the Cuthill-McKee algorithm is carried out on
 * the adjacency matrix of the symmetrisation ‘A+A'’, where ‘A'’
 * denotes the transpose of ‘A’.  For a rectangular matrix, the
 * Cuthill-McKee algorithm is carried out on a bipartite graph formed
 * by the matrix rows and columns.  The adjacency matrix ‘B’ of the
 * bipartite graph is square and symmetric and takes the form of a
 * 2-by-2 block matrix where ‘A’ is placed in the upper right corner
 * and ‘A'’ is placed in the lower left corner:
 *
 *     ⎡  0   A ⎤
 * B = ⎢        ⎥.
 *     ⎣  A'  0 ⎦
 *
 * As a result, the number of vertices in the graph is equal to
 * ‘num_rows’ (and ‘num_columns’) if the matrix is square. Otherwise,
 * if the matrix is rectangular, then there are ‘num_rows+num_columns’
 * vertices.
 *
 * ‘starting_vertex’ is either ‘NULL’, in which case it is ignored, or
 * it is a pointer to an integer that is used to designate a starting
 * vertex for the Cuthill-McKee algorithm.  The designated starting
 * vertex must be in the range ‘1,2,...,N’, where ‘N’ is the number of
 * vertices.  Otherwise, if the starting vertex is set to zero, then a
 * starting vertex is chosen automatically by selecting a
 * pseudo-peripheral vertex, and the value pointed to by
 * ‘starting_vertex’ is updated to reflect the chosen starting vertex.
 *
 * On success, the array ‘vertex_order’ will contain the new ordering
 * of the vertices (i.e., the rows of the matrix for a square matrix
 * or the rows and column of the matrix in the case of a rectangular
 * matrix).  Therefore, it must hold enough storage for at least
 * ‘num_rows’ (or ‘num_rows+num_columns’ if the matrix is rectangular)
 * values of type ‘int’.
 */
int cuthill_mckee(
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const int64_t * colptr,
    const int * rowidx,
    int * starting_vertex,
    int size,
    int * vertex_order);

#endif
