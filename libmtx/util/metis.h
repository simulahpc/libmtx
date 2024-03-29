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
 * Last modified: 2022-09-06
 *
 * METIS graph partitioning and sparse matrix reordering algorithms.
 */

#ifndef LIBMTX_UTIL_METIS
#define LIBMTX_UTIL_METIS

#include <stdint.h>

/**
 * ‘metis_partgraphsym()’ uses the METIS k-way graph partitioner to
 * partition an undirected graph given as a square, symmetric matrix
 * in coordinate format.
 *
 * The undirected graph is described in terms of a symmetric adjacency
 * matrix in coordinate (COO) format with ‘N’ rows and columns. There
 * are ‘num_nonzeros’ nonzero matrix entries. The locations of the
 * matrix nonzeros are specified by the arrays ‘rowidx’ and ‘colidx’,
 * both of which are of length ‘num_nonzeros’, and contain offsets in
 * the range ‘[0,N)’. Note that there should not be any duplicate
 * nonzero entries. The nonzeros may be located in the upper or lower
 * triangle of the adjacency matrix. However, if there is a nonzero
 * entry at row ‘i’ and column ‘j’, then there should not be a nonzero
 * entry row ‘j’ and column ‘i’.
 *
 * The values ‘rowidxstride’ and ‘colidxstride’ may be used to specify
 * strides (in bytes) that are used when accessing the row and column
 * offsets in ‘rowidx’ and ‘colidx’, respectively. This is useful for
 * cases where the row and column offsets are not necessarily stored
 * contiguously in memory.
 *
 * On success, the array ‘dstpart’ contains the part numbers assigned
 * by the partitioner to the graph vertices. Therefore, ‘dstpart’ must
 * be an array of length ‘N’.
 *
 * If it is not ‘NULL’, then ‘objval’ is used to store the value of
 * the objective function minimized by the partitioner, which, by
 * default, is the edge-cut of the partitioning solution.
 */
int metis_partgraphsym(
    int num_parts,
    int64_t N,
    int64_t size,
    int rowidxstride,
    int rowidxbase,
    const int64_t * rowidx,
    int colidxstride,
    int colidxbase,
    const int64_t * colidx,
    int * dstpart,
    int64_t * objval,
    int verbose);

/**
 * ‘metis_partgraph()’ uses the METIS k-way graph partitioner to
 * partition an undirected graph derived from a sparse matrix.
 *
 * The sparse matrix is provided in coordinate (COO) format with
 * dimensions given by ‘num_rows’ and ‘num_columns’. Furthermore,
 * there are ‘num_nonzeros’ nonzero matrix entries, whose locations
 * are specified by the arrays ‘rowidx’ and ‘colidx’ (of length
 * ‘num_nonzeros’). The row offsets are in the range ‘[0,num_rows)’,
 * whereas the column offsets are given in the range are in the range
 * ‘[0,num_columns)’.
 *
 * The matrix may be unsymmetric or even non-square. Furthermore,
 * duplicate nonzero matrix entries are allowed, though they will be
 * removed when forming the undirected graph that is passed to the
 * METIS partitioner.
 *
 * If the matrix is square, then the graph to be partitioned is
 * obtained from the symmetrisation ‘A+A'’ of the matrix ‘A’ , where
 * ‘A'’ denotes the transpose of ‘A’.
 *
 * If the matrix is non-square, the partitioning algorithm is carried
 * out on a bipartite graph formed by the matrix rows and columns.
 * The adjacency matrix ‘B’ of the bipartite graph is square and
 * symmetric and takes the form of a 2-by-2 block matrix where ‘A’ is
 * placed in the upper right corner and ‘A'’ is placed in the lower
 * left corner:
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
 * The array ‘dstrowpart’ must be of length ‘num_rows’. This array is
 * used to store the part numbers assigned to the matrix rows. If the
 * matrix is non-square, then ‘dstcolpart’ must be an array of length
 * ‘num_columns’, which is then similarly used to store the part
 * numbers assigned to the matrix columns.
 *
 * If it is not ‘NULL’, then ‘objval’ is used to store the value of
 * the objective function minimized by the partitioner, which, by
 * default, is the edge-cut of the partitioning solution.
 */
int metis_partgraph(
    int num_parts,
    int64_t num_rows,
    int64_t num_columns,
    int64_t num_nonzeros,
    int rowidxstride,
    int rowidxbase,
    const int64_t * rowidx,
    int colidxstride,
    int colidxbase,
    const int64_t * colidx,
    int * dstrowpart,
    int * dstcolpart,
    int64_t * objval,
    int verbose);

/**
 * ‘metis_ndsym()’ uses METIS to compute a multilevel nested
 * dissection ordering of an undirected graph given as a square,
 * symmetric matrix in coordinate format.
 *
 * The undirected graph is described in terms of a symmetric adjacency
 * matrix in coordinate (COO) format with ‘N’ rows and columns. There
 * are ‘num_nonzeros’ nonzero matrix entries. The locations of the
 * matrix nonzeros are specified by the arrays ‘rowidx’ and ‘colidx’,
 * both of which are of length ‘num_nonzeros’, and contain offsets in
 * the range ‘[0,N)’. Note that there should not be any duplicate
 * nonzero entries. The nonzeros may be located in the upper or lower
 * triangle of the adjacency matrix. However, if there is a nonzero
 * entry at row ‘i’ and column ‘j’, then there should not be a nonzero
 * entry row ‘j’ and column ‘i’. (Although both nonzeros are required
 * in the undirected graph data structure passed to METIS, as
 * described in Section 5.5 of the METIS manual, the required nonzeros
 * will be added by ‘metis_ndsym()’ before calling METIS.)
 *
 * The values ‘rowidxstride’ and ‘colidxstride’ may be used to specify
 * strides (in bytes) that are used when accessing the row and column
 * offsets in ‘rowidx’ and ‘colidx’, respectively. This is useful for
 * cases where the row and column offsets are not necessarily stored
 * contiguously in memory.
 *
 * On success, the arrays ‘perm’ and ‘perminv’ contain the permutation
 * and inverse permutation of the graph vertices. Therefore, ‘perm’
 * and ‘perminv’ must be arrays of length ‘N’.
 */
int metis_ndsym(
    int64_t N,
    int64_t size,
    int rowidxstride,
    int rowidxbase,
    const int64_t * rowidx,
    int colidxstride,
    int colidxbase,
    const int64_t * colidx,
    int * perm,
    int * perminv,
    int verbose);

/**
 * ‘metis_nd()’ uses METIS to compute a multilevel nested dissection
 * ordering of an undirected graph derived from a sparse matrix.
 *
 * The sparse matrix is provided in coordinate (COO) format with
 * dimensions given by ‘num_rows’ and ‘num_columns’. Furthermore,
 * there are ‘num_nonzeros’ nonzero matrix entries, whose locations
 * are specified by the arrays ‘rowidx’ and ‘colidx’ (of length
 * ‘num_nonzeros’). The row offsets are in the range ‘[0,num_rows)’,
 * whereas the column offsets are given in the range are in the range
 * ‘[0,num_columns)’.
 *
 * The matrix may be unsymmetric or even non-square. Furthermore,
 * duplicate nonzero matrix entries are allowed, though they will be
 * removed when forming the undirected graph that is passed to METIS.
 *
 * If the matrix is square, then the graph to be reordered is obtained
 * from the symmetrisation ‘A+A'’ of the matrix ‘A’ , where ‘A'’
 * denotes the transpose of ‘A’.
 *
 * If the matrix is non-square, the reordering algorithm is carried
 * out on a bipartite graph formed by the matrix rows and columns.
 * The adjacency matrix ‘B’ of the bipartite graph is square and
 * symmetric and takes the form of a 2-by-2 block matrix where ‘A’ is
 * placed in the upper right corner and ‘A'’ is placed in the lower
 * left corner:
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
 * The arrays ‘rowperm’ and ‘rowperminv’ must be of length
 * ‘num_rows’. These arrays are used to store the permutation and
 * inverse permutation of the matrix rows. If the matrix is
 * non-square, then ‘colperm’ and ‘colperminv’ must be arrays of
 * length ‘num_columns’, which are then similarly used to store the
 * permutation and inverse permutation of the matrix columns.
 */
int metis_nd(
    int64_t num_rows,
    int64_t num_columns,
    int64_t num_nonzeros,
    int rowidxstride,
    int rowidxbase,
    const int64_t * rowidx,
    int colidxstride,
    int colidxbase,
    const int64_t * colidx,
    int * rowperm,
    int * rowperminv,
    int * colperm,
    int * colperminv,
    int verbose);

#endif
