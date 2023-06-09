/* This file is part of Libmtx.
 *
 * Copyright (C) 2023 James D. Trotter
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
 * Last modified: 2023-06-09
 *
 * SCOTCH graph partitioning and sparse matrix reordering algorithms.
 */

#include <libmtx/libmtx-config.h>

#include <libmtx/error.h>
#include <libmtx/util/merge.h>
#include <libmtx/util/scotch.h>

#include <errno.h>
#include <unistd.h>

#include <limits.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

#ifdef LIBMTX_HAVE_SCOTCH
#include <scotch.h>
#endif

/**
 * ‘scotch_partgraphsym()’ uses the SCOTCH k-way graph partitioner to
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
 * On success, the array ‘dstpart’ contains the part numbers assigned
 * by the partitioner to the graph vertices. Therefore, ‘dstpart’ must
 * be an array of length ‘N’.
 *
 * If it is not ‘NULL’, then ‘objval’ is used to store the value of
 * the objective function minimized by the partitioner, which, by
 * default, is the edge-cut of the partitioning solution.
 */
int scotch_partgraphsym(
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
    int64_t * outobjval,
    int verbose)
{
#ifndef LIBMTX_HAVE_SCOTCH
    return MTX_ERR_SCOTCH_NOT_SUPPORTED;
#else
    int err;
    if (N > SCOTCH_NUMMAX-1) return MTX_ERR_SCOTCH_EOVERFLOW;
    if (size > SCOTCH_NUMMAX) return MTX_ERR_SCOTCH_EOVERFLOW;
    if (num_parts <= 0) return MTX_ERR_INDEX_OUT_OF_BOUNDS;

    /* perform some bounds checking on the input graph */
    for (int64_t k = 0; k < size; k++) {
        int64_t i = *(const int64_t *) ((const char *) rowidx + k*rowidxstride)-rowidxbase;
        int64_t j = *(const int64_t *) ((const char *) colidx + k*colidxstride)-colidxbase;
        if (i < 0 || i >= N || j < 0 || j >= N) return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    }

    if (num_parts == 1) {
        for (SCOTCH_Num i = 0; i < N; i++) dstpart[i] = 0;
        return MTX_SUCCESS;
    }

    /* the number of vertices in the graph */
    SCOTCH_Num nvtxs = N;

    /* adjacency structure of the graph (row pointers) */
    SCOTCH_Num * xadj = malloc((nvtxs+1) * sizeof(SCOTCH_Num));
    if (!xadj) return MTX_ERR_ERRNO;
    for (SCOTCH_Num i = 0; i <= nvtxs; i++) xadj[i] = 0;
    for (int64_t k = 0; k < size; k++) {
        int64_t i = *(const int64_t *) ((const char *) rowidx + k*rowidxstride)-rowidxbase;
        int64_t j = *(const int64_t *) ((const char *) colidx + k*colidxstride)-colidxbase;
        if (i != j) { xadj[i+1]++; xadj[j+1]++; }
    }
    for (SCOTCH_Num i = 1; i <= nvtxs; i++) {
        if (xadj[i] > SCOTCH_NUMMAX - xadj[i-1]) { free(xadj); return MTX_ERR_SCOTCH_EOVERFLOW; }
        xadj[i] += xadj[i-1];
    }

    /* adjacency structure of the graph (column offsets) */
    SCOTCH_Num * adjncy = malloc(xadj[nvtxs] * sizeof(SCOTCH_Num));
    if (!adjncy) { free(xadj); return MTX_ERR_ERRNO; }

    for (int64_t k = 0; k < size; k++) {
        int64_t i = *(const int64_t *) ((const char *) rowidx + k*rowidxstride)-rowidxbase;
        int64_t j = *(const int64_t *) ((const char *) colidx + k*colidxstride)-colidxbase;
        if (i != j) { adjncy[xadj[i]++] = j; adjncy[xadj[j]++] = i; }
    }
    for (SCOTCH_Num i = nvtxs; i > 0; i--) xadj[i] = xadj[i-1];
    xadj[0] = 0;

#ifdef DEBUG_SCOTCH
    fprintf(stderr, "nvtxs=%"PRIDX", xadj=[", nvtxs);
    for (SCOTCH_Num i = 0; i <= nvtxs; i++) fprintf(stderr, " %"PRIDX, xadj[i]);
    fprintf(stderr, "]\n");
    fprintf(stderr, "adjncy=[");
    for (SCOTCH_Num i = 0; i < nvtxs; i++) {
        fprintf(stderr, " (%"PRIDX")", i);
        for (SCOTCH_Num k = xadj[i]; k < xadj[i+1]; k++)
            fprintf(stderr, " %"PRIDX, adjncy[k]);
    }
    fprintf(stderr, "]\n");
#endif

    /* edge-cut or the total communication volume of the partitioning
     * solution */
    SCOTCH_Num objval = 0;

    /* This is a vector of size nvtxs that upon successful completion
     * stores the partition vector of the graph. The numbering of this
     * vector starts from either 0 or 1, depending on the value of
     * options[SCOTCH OPTION NUMBERING]. */
    SCOTCH_Num * part = NULL;
    bool free_part = false;
    if (sizeof(*dstpart) == sizeof(*part)) {
        part = (SCOTCH_Num *) dstpart;
    } else {
        part = malloc(nvtxs * sizeof(SCOTCH_Num));
        if (!part) { free(adjncy); free(xadj); return MTX_ERR_ERRNO; }
        free_part = true;
    }

    SCOTCH_Graph graph;
    err = SCOTCH_graphInit(&graph);
    if (err) {
        if (free_part) free(part);
        free(adjncy); free(xadj);
        return MTX_ERR_SCOTCH;
    }

    // int SCOTCH_graphBuild(
    //     SCOTCH_Graph * const grafptr,
    //     const SCOTCH_Num baseval,
    //     const SCOTCH_Num vertnbr,
    //     const SCOTCH_Num * const verttab,
    //     const SCOTCH_Num * const vendtab,
    //     const SCOTCH_Num * const velotab,
    //     const SCOTCH_Num * const vlbltab,
    //     const SCOTCH_Num edgenbr,
    //     const SCOTCH_Num * const edgetab,
    //     const SCOTCH_Num * const edlotab)

    int baseval = 0;
    err = SCOTCH_graphBuild(
        &graph, baseval, nvtxs, xadj, NULL, NULL, NULL,
        xadj[nvtxs], adjncy, NULL);
    if (err) {
        SCOTCH_graphExit(&graph);
        if (free_part) free(part);
        free(adjncy); free(xadj);
        return MTX_ERR_SCOTCH;
    }

    // int SCOTCH_stratInit(SCOTCH_Strat * straptr)
    SCOTCH_Strat strat;
    err = SCOTCH_stratInit(&strat);
    if (err) {
        SCOTCH_graphExit(&graph);
        if (free_part) free(part);
        free(adjncy); free(xadj);
        return MTX_ERR_SCOTCH;
    }

    // int SCOTCH_graphPart(
    //    const SCOTCH_Graph * grafptr,
    //    const SCOTCH_Num partnbr,
    //    const SCOTCH_Strat * straptr,
    //    SCOTCH_Num * parttab)

    err = SCOTCH_graphPart(&graph, num_parts, &strat, part);
    if (err) {
        SCOTCH_stratExit(&strat); SCOTCH_graphExit(&graph);
        if (free_part) free(part);
        free(adjncy); free(xadj);
        return MTX_ERR_SCOTCH;
    }
    SCOTCH_stratExit(&strat); SCOTCH_graphExit(&graph);

    if (outobjval) *outobjval = objval;
    if (sizeof(*dstpart) != sizeof(*part)) {
        for (SCOTCH_Num i = 0; i < N; i++) {
            if (part[i] > INT_MAX) {
                free(part); free(adjncy); free(xadj);
                errno = ERANGE;
                return MTX_ERR_ERRNO;
            }
            dstpart[i] = part[i];
        }
        free(part);
    }
    free(adjncy); free(xadj);
    return MTX_SUCCESS;
#endif
}

/**
 * ‘scotch_partgraph()’ uses the SCOTCH k-way graph partitioner to
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
 * SCOTCH partitioner.
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
int scotch_partgraph(
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
    int verbose)
{
#ifndef LIBMTX_HAVE_SCOTCH
    return MTX_ERR_SCOTCH_NOT_SUPPORTED;
#else
    bool square = num_rows == num_columns;
    int64_t N = square ? num_rows : num_rows + num_columns;

    if (square) {
        /*
         * Handle unsymmetric matrices via symmetrisation: add all
         * nonzeros and their symmetric counterparts, then compact,
         * (i.e., sort and remove duplicates).
         */
        int64_t (* idx)[2] = malloc(num_nonzeros * sizeof(int64_t[2]));
        if (!idx) return MTX_ERR_ERRNO;
        for (int64_t k = 0; k < num_nonzeros; k++) {
            int64_t i = *(const int64_t *) ((const char *) rowidx + k*rowidxstride)-rowidxbase;
            int64_t j = *(const int64_t *) ((const char *) colidx + k*colidxstride)-colidxbase;
            if (i <= j) { idx[k][0] = i; idx[k][1] = j; }
            else        { idx[k][0] = j; idx[k][1] = i; }
        }
        int err = compact_unsorted_int64_pair(
            &num_nonzeros, idx, num_nonzeros, idx, NULL, NULL);
        if (err) { free(idx); return err; }
        err = scotch_partgraphsym(
            num_parts, N, num_nonzeros, sizeof(*idx), 0, &idx[0][0],
            sizeof(*idx), 0, &idx[0][1], dstrowpart, objval, verbose);
        if (err) { free(idx); return err; }
        free(idx);
    } else {
        /*
         * Handle non-square matrices by partitioning the bipartite
         * graph whose vertices are the rows and columns of the
         * matrix. This requires shifting the column indices to the
         * right by an offset equal to the number of matrix rows.
         */
        int * dstpart = malloc(N * sizeof(int));
        if (!dstpart) return MTX_ERR_ERRNO;
        int64_t * tmpcolidx = malloc(num_nonzeros * sizeof(int64_t));
        if (!tmpcolidx) { free(dstpart); return MTX_ERR_ERRNO; }
        for (int64_t k = 0; k < num_nonzeros; k++) {
            int64_t j = *(const int64_t *) ((const char *) colidx + k*colidxstride)-colidxbase;
            tmpcolidx[k] = num_rows+j;
        }
        int err = scotch_partgraphsym(
            num_parts, N, num_nonzeros, rowidxstride, rowidxbase, rowidx,
            sizeof(*tmpcolidx), 0, tmpcolidx, dstpart, objval, verbose);
        if (err) { free(tmpcolidx); free(dstpart); return err; }
        free(tmpcolidx);
        for (int64_t i = 0; i < num_rows; i++) dstrowpart[i] = dstpart[i];
        for (int64_t j = 0; j < num_columns; j++) dstcolpart[j] = dstpart[num_rows+j];
        free(dstpart);
    }
    return MTX_SUCCESS;
#endif
}

/**
 * ‘scotch_ndsym()’ uses SCOTCH to compute a multilevel nested
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
 * in the undirected graph data structure passed to SCOTCH, as
 * described in Section 5.5 of the SCOTCH manual, the required nonzeros
 * will be added by ‘scotch_ndsym()’ before calling SCOTCH.)
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
int scotch_ndsym(
    int64_t N,
    int64_t size,
    int rowidxstride,
    int rowidxbase,
    const int64_t * rowidx,
    int colidxstride,
    int colidxbase,
    const int64_t * colidx,
    int * dstperm,
    int * dstperminv,
    int verbose)
{
#ifndef LIBMTX_HAVE_SCOTCH
    return MTX_ERR_SCOTCH_NOT_SUPPORTED;
#else
    int err;
    if (N > SCOTCH_NUMMAX-1) return MTX_ERR_SCOTCH_EOVERFLOW;
    if (size > SCOTCH_NUMMAX) return MTX_ERR_SCOTCH_EOVERFLOW;

    /* perform some bounds checking on the input graph */
    for (int64_t k = 0; k < size; k++) {
        int64_t i = *(const int64_t *) ((const char *) rowidx + k*rowidxstride)-rowidxbase;
        int64_t j = *(const int64_t *) ((const char *) colidx + k*colidxstride)-colidxbase;
        if (i < 0 || i >= N || j < 0 || j >= N) return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    }

    /* the number of vertices in the graph */
    SCOTCH_Num nvtxs = N;

    /* adjacency structure of the graph (row pointers) */
    SCOTCH_Num * xadj = malloc((nvtxs+1) * sizeof(SCOTCH_Num));
    if (!xadj) return MTX_ERR_ERRNO;
    for (SCOTCH_Num i = 0; i <= nvtxs; i++) xadj[i] = 0;
    for (int64_t k = 0; k < size; k++) {
        int64_t i = *(const int64_t *) ((const char *) rowidx + k*rowidxstride)-rowidxbase;
        int64_t j = *(const int64_t *) ((const char *) colidx + k*colidxstride)-colidxbase;
        if (i != j) { xadj[i+1]++; xadj[j+1]++; }
    }
    for (SCOTCH_Num i = 1; i <= nvtxs; i++) {
        if (xadj[i] > SCOTCH_NUMMAX - xadj[i-1]) { free(xadj); return MTX_ERR_SCOTCH_EOVERFLOW; }
        xadj[i] += xadj[i-1];
    }

    /* adjacency structure of the graph (column offsets) */
    SCOTCH_Num * adjncy = malloc(xadj[nvtxs] * sizeof(SCOTCH_Num));
    if (!adjncy) { free(xadj); return MTX_ERR_ERRNO; }

    for (int64_t k = 0; k < size; k++) {
        int64_t i = *(const int64_t *) ((const char *) rowidx + k*rowidxstride)-rowidxbase;
        int64_t j = *(const int64_t *) ((const char *) colidx + k*colidxstride)-colidxbase;
        if (i != j) { adjncy[xadj[i]++] = j; adjncy[xadj[j]++] = i; }
    }
    for (SCOTCH_Num i = nvtxs; i > 0; i--) xadj[i] = xadj[i-1];
    xadj[0] = 0;

#ifdef DEBUG_SCOTCH
    fprintf(stderr, "nvtxs=%"PRIDX", xadj=[", nvtxs);
    for (SCOTCH_Num i = 0; i <= nvtxs; i++) fprintf(stderr, " %"PRIDX, xadj[i]);
    fprintf(stderr, "]\n");
    fprintf(stderr, "adjncy=[");
    for (SCOTCH_Num i = 0; i < nvtxs; i++) {
        fprintf(stderr, " (%"PRIDX")", i);
        for (SCOTCH_Num k = xadj[i]; k < xadj[i+1]; k++)
            fprintf(stderr, " %"PRIDX, adjncy[k]);
    }
    fprintf(stderr, "]\n");
#endif

    SCOTCH_Num * vwgt = NULL;    /* the weights of the vertices */

    /* These are vectors, each of size nvtxs that upon successful
     * completion store the permutation and inverse permutation
     * vectors of the fill-reducing graph ordering. The numbering of
     * this vector starts from either 0 or 1, depending on the value
     * of options[SCOTCH OPTION NUMBERING]. */
    SCOTCH_Num * perm = NULL;
    SCOTCH_Num * perminv = NULL;
    bool free_perm = false;
    if (sizeof(*dstperm) == sizeof(*perm)) {
        perm = (SCOTCH_Num *) dstperm;
        perminv = (SCOTCH_Num *) dstperminv;
    } else {
        perm = malloc(nvtxs * sizeof(SCOTCH_Num));
        if (!perm) { free(adjncy); free(xadj); return MTX_ERR_ERRNO; }
        perminv = malloc(nvtxs * sizeof(SCOTCH_Num));
        if (!perminv) { free(perm); free(adjncy); free(xadj); return MTX_ERR_ERRNO; }
        free_perm = true;
    }

    SCOTCH_Graph graph;
    err = SCOTCH_graphInit(&graph);
    if (err) {
        if (free_perm) { free(perminv); free(perm); }
        free(adjncy); free(xadj);
        return MTX_ERR_SCOTCH;
    }

    // int SCOTCH_graphBuild(
    //     SCOTCH_Graph * const grafptr,
    //     const SCOTCH_Num baseval,
    //     const SCOTCH_Num vertnbr,
    //     const SCOTCH_Num * const verttab,
    //     const SCOTCH_Num * const vendtab,
    //     const SCOTCH_Num * const velotab,
    //     const SCOTCH_Num * const vlbltab,
    //     const SCOTCH_Num edgenbr,
    //     const SCOTCH_Num * const edgetab,
    //     const SCOTCH_Num * const edlotab)

    int baseval = 0;
    err = SCOTCH_graphBuild(
        &graph, baseval, nvtxs, xadj, NULL, NULL, NULL,
        xadj[nvtxs], adjncy, NULL);
    if (err) {
        SCOTCH_graphExit(&graph);
        if (free_perm) { free(perminv); free(perm); }
        free(adjncy); free(xadj);
        return MTX_ERR_SCOTCH;
    }

    // int SCOTCH_stratInit(SCOTCH_Strat * straptr)
    SCOTCH_Strat strat;
    err = SCOTCH_stratInit(&strat);
    if (err) {
        SCOTCH_graphExit(&graph);
        if (free_perm) { free(perminv); free(perm); }
        free(adjncy); free(xadj);
        return MTX_ERR_SCOTCH;
    }

    // int SCOTCH_graphOrder(
    //     const SCOTCH_Graph * grafptr,
    //     const SCOTCH_Strat * straptr,
    //     SCOTCH_Num * permtab,
    //     SCOTCH_Num * peritab,
    //     SCOTCH_Num * cblkptr,
    //     SCOTCH_Num * rangtab,
    //     SCOTCH_Num * treetab)

    err = SCOTCH_graphOrder(
        &graph, &strat, perm, perminv, NULL, NULL, NULL);
    if (err) {
        SCOTCH_stratExit(&strat); SCOTCH_graphExit(&graph);
        if (free_perm) { free(perminv); free(perm); }
        free(adjncy); free(xadj);
        return MTX_ERR_SCOTCH;
    }
    SCOTCH_stratExit(&strat); SCOTCH_graphExit(&graph);

    if (sizeof(*dstperm) != sizeof(*perm)) {
        for (SCOTCH_Num i = 0; i < N; i++) {
            if (perm[i] > INT_MAX) {
                free(perminv); free(perm); free(adjncy); free(xadj);
                errno = ERANGE;
                return MTX_ERR_ERRNO;
            }
            dstperm[i] = perm[i];
            dstperminv[i] = perminv[i];
        }
        free(perminv); free(perm);
    }
    free(adjncy); free(xadj);
    return MTX_SUCCESS;
#endif
}

/**
 * ‘scotch_nd()’ uses SCOTCH to compute a multilevel nested dissection
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
 * removed when forming the undirected graph that is passed to SCOTCH.
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
int scotch_nd(
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
    int verbose)
{
#ifndef LIBMTX_HAVE_SCOTCH
    return MTX_ERR_SCOTCH_NOT_SUPPORTED;
#else
    bool square = num_rows == num_columns;
    int64_t N = square ? num_rows : num_rows + num_columns;

    if (square) {
        /*
         * Handle unsymmetric matrices via symmetrisation: add all
         * nonzeros and their symmetric counterparts, then compact,
         * (i.e., sort and remove duplicates).
         */
        int64_t (* idx)[2] = malloc(num_nonzeros * sizeof(int64_t[2]));
        if (!idx) return MTX_ERR_ERRNO;
        for (int64_t k = 0; k < num_nonzeros; k++) {
            int64_t i = *(const int64_t *) ((const char *) rowidx + k*rowidxstride)-rowidxbase;
            int64_t j = *(const int64_t *) ((const char *) colidx + k*colidxstride)-colidxbase;
            if (i <= j) { idx[k][0] = i; idx[k][1] = j; }
            else        { idx[k][0] = j; idx[k][1] = i; }
        }
        int err = compact_unsorted_int64_pair(
            &num_nonzeros, idx, num_nonzeros, idx, NULL, NULL);
        if (err) { free(idx); return MTX_ERR_ERRNO; }
        err = scotch_ndsym(
            N, num_nonzeros, sizeof(*idx), 0, &idx[0][0],
            sizeof(*idx), 0, &idx[0][1], rowperm, rowperminv, verbose);
        if (err) { free(idx); return err; }
        free(idx);
    } else {
        /*
         * Handle non-square matrices by reordering the bipartite
         * graph whose vertices are the rows and columns of the
         * matrix. This requires shifting the column indices to the
         * right by an offset equal to the number of matrix rows.
         */
        int * perm = malloc(N * sizeof(int));
        if (!perm) return MTX_ERR_ERRNO;
        int * perminv = malloc(N * sizeof(int));
        if (!perminv) { free(perm); return MTX_ERR_ERRNO; }
        int64_t * tmpcolidx = malloc(num_nonzeros * sizeof(int64_t));
        if (!tmpcolidx) { free(perminv); free(perm); return MTX_ERR_ERRNO; }
        for (int64_t k = 0; k < num_nonzeros; k++) {
            int64_t j = *(const int64_t *) ((const char *) colidx + k*colidxstride)-colidxbase;
            tmpcolidx[k] = num_rows+j;
        }
        int err = scotch_ndsym(
            N, num_nonzeros, rowidxstride, rowidxbase, rowidx,
            sizeof(*tmpcolidx), 0, tmpcolidx, perm, perminv, verbose);
        if (err) { free(tmpcolidx); free(perminv); free(perm); return err; }
        free(tmpcolidx);
        int k = 0, l = 0;
        for (int64_t i = 0; i < N; i++) {
            if (perm[i] < num_rows) { rowperm[k++] = perm[i]; }
            else { colperm[l++] = perm[i]-num_rows; }
        }
        for (int64_t i = 0; i < num_rows; i++) rowperminv[rowperm[i]] = i;
        for (int64_t j = 0; j < num_columns; j++) colperminv[colperm[j]] = j;
        free(perminv); free(perm);
    }
    return MTX_SUCCESS;
#endif
}
