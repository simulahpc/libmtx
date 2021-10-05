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
    int * vertex_order);

#endif
