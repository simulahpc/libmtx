/* This file is part of Libmtx.
 *
 * Copyright (C) 2022 James D. Trotter
 *
 * Libmtx is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Libmtx is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Libmtx.  If not, see <https://www.gnu.org/licenses/>.
 *
 * Authors: James D. Trotter <james@simula.no>
 * Last modified: 2022-05-23
 *
 * METIS graph partitioning and sparse matrix reordering algorithms.
 */

#ifndef LIBMTX_UTIL_METIS
#define LIBMTX_UTIL_METIS

#include <stdint.h>

/**
 * ‘metis_partgraph()’ uses the METIS k-way graph partitioner to
 * partition an undirected graph.
 *
 * The undirected graph is described in terms of a symmetric adjacency
 * matrix in compressed sparse row (CSR) and compressed sparse column
 * (CSC) format.  The former consists of ‘num_rows+1’ row pointers,
 * ‘rowptr’, and ‘size’ column indices, ‘colidx’, whereas the latter
 * consists of ‘num_columns+1’ column pointers, ‘colptr’, and ‘size’
 * row indices, ‘rowidx’.  Note that the matrix given in CSC format is
 * equivalent to its transpose in CSR format.  Also, note that row and
 * column indices use 0-based indexing, unlike the Matrix Market
 * format.
 *
 * For a square matrix, the partitioning algorithm is carried out on
 * the adjacency matrix of the symmetrisation ‘A+A'’, where ‘A'’
 * denotes the transpose of ‘A’. For a non-square matrix, the
 * partitioning algorithm is carried out on a bipartite graph formed
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
 * if the matrix is non-square, then there are ‘num_rows+num_columns’
 * vertices.
 *
 * On success, the array ‘parts’ contains the part numbers assigned by
 * the partitioner to the vertices (i.e., the rows of the matrix for a
 * square matrix or the rows and column of the matrix in the case of a
 * non-square matrix). Therefore, ‘parts’ must be an array of length
 * ‘num_rows’ if the matrix is square, or ‘num_rows+num_columns’ if
 * the matrix is non-square.
 */
int metis_partgraph(
    int num_parts,
    int num_rows,
    int num_columns,
    bool symmetric,
    int64_t num_nonzeros,
    const int * rowidx,
    const int * colidx,
    int size,
    int * parts);

#endif
