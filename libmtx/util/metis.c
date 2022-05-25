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

#include <libmtx/libmtx-config.h>

#include <libmtx/error.h>
#include <libmtx/util/metis.h>

#ifdef LIBMTX_HAVE_METIS
#include <metis.h>
#endif

#include <stdbool.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

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
 * On success, the array ‘dstpart’ contains the part numbers assigned
 * by the partitioner to the vertices (i.e., the rows of the matrix
 * for a square matrix or the rows and column of the matrix in the
 * case of a non-square matrix). Therefore, ‘dstpart’ must be an array
 * of length ‘num_rows’ if the matrix is square, or
 * ‘num_rows+num_columns’ if the matrix is non-square.
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
    int * dstpart)
{
#ifndef LIBMTX_HAVE_METIS
    return MTX_ERR_METIS_NOT_SUPPORTED;
#else
    bool square = num_rows == num_columns;
    int num_vertices = square ? num_rows : num_rows + num_columns;
    if (!square && symmetric) return MTX_ERR_INVALID_SYMMETRY;
    if (num_nonzeros > IDX_MAX) return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    if (size < num_vertices) return MTX_ERR_INDEX_OUT_OF_BOUNDS;

    /* perform some bounds checking on the input graph */
    /* if (square) { */
    /*     for (int i = 0; i < num_rows; i++) { */
    /*         if (rowptr[i] > rowptr[i+1]) */
    /*             return MTX_ERR_INDEX_OUT_OF_BOUNDS; */
    /*         for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++) { */
    /*             if (colidx[k] < 0 || colidx[k] >= num_columns) */
    /*                 return MTX_ERR_INDEX_OUT_OF_BOUNDS; */
    /*         } */
    /*     } */
    /* } else { */
    /*     for (int j = 0; j < num_columns; j++) { */
    /*         if (colptr[j] > colptr[j+1]) */
    /*             return MTX_ERR_INDEX_OUT_OF_BOUNDS; */
    /*         for (int64_t k = colptr[j]; k < colptr[j+1]; k++) { */
    /*             if (rowidx[k] < 0 || rowidx[k] >= num_rows) */
    /*                 return MTX_ERR_INDEX_OUT_OF_BOUNDS; */
    /*         } */
    /*     } */
    /* } */

    for (int64_t k = 0; k < num_nonzeros; k++) {
        if (rowidx[k] < 0 || rowidx[k] >= num_rows || rowidx[k] > IDX_MAX
            colidx[k] < 0 || colidx[k] >= num_columns || colidx[k] > IDX_MAX)
            return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    }

    /* configure partitioning options */
    idx_t options[METIS_NOPTIONS]; /* array of options */
    int err = METIS_SetDefaultOptions(options);
    if (err == METIS_ERROR_INPUT) { return MTX_ERR_METIS_INPUT; }
    else if (err == METIS_ERROR_MEMORY) { return MTX_ERR_METIS_MEMORY; }
    else if (err == METIS_ERROR) { return MTX_ERR_METIS; }

    /* METIS_OBJTYPE_CUT for edge-cut minimization (default) or
     * METIS_OBJTYPE_VOL for total communication volume
     * minimization */
    // options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_CUT;

    /* METIS_CTYPE_RM for random matching or METIS_CTYPE_SHEM for
     * sorted heavy-edge matching (default) */
    // options[METIS_OPTION_CTYPE] = METIS_CTYPE_SHEM;

    /*
     * METIS_IPTYPE_GROW - grows a bisection using a greedy strategy (default)
     * METIS_IPTYPE_RANDOM - computes a bisection at random followed
     *                       by a refinement
     * METIS_IPTYPE_EDGE - derives a separator from an edge cut
     * METIS_IPTYPE_NODE - grow a bisection using a greedy node-based
     *                     strategy
     */
    // options[METIS_OPTION_IPTYPE] = METIS_IPTYPE_GROW;

    /*
     * METIS_RTYPE_FM - FM-based cut refinement
     * METIS_RTYPE_GREEDY - greedy-based cut and volume refinement
     * METIS_RTYPE_SEP2SIDED - two-sided node FM refinement
     * METIS_RTYPE_SEP1SIDED - One-sided node FM refinement (default)
     */
    // options[METIS_OPTION_RTYPE] = METIS_RTYPE_SEP1SIDED;

    /* 0 (default) performs a 2–hop matching or 1 does not */
    // options[METIS_OPTION_NO2HOP] = 0;

    /* The number of different partitionings to compute. The final
     * partitioning is the one that achieves the best edgecut or
     * communication volume. Default is 1. */
    // options[METIS_OPTION_NCUTS] = 1;

    /* The number of iterations for the refinement algorithms at each
     * stage of the uncoarsening process. Default is 10. */
    // options[METIS_OPTION_NITER] = 10;

    /* Specifies the maximum allowed load imbalance among the
     * partitions. A value of x indicates that the allowed load
     * imbalance is (1 + x)/1000. The load imbalance for the jth
     * constraint is defined to be max_i(w[j,i])/t[j,i]), where w[j,i]
     * is the fraction of the overall weight of the jth constraint
     * that is assigned to the ith partition and t[j, i] is the
     * desired target weight of the jth constraint for the ith
     * partition (i.e., that specified via -tpwgts). For -ptype=rb,
     * the default value is 1 (i.e., load imbalance of 1.001) and for
     * -ptype=kway, the default value is 30 (i.e., load imbalance of
     * 1.03). */
    // options[METIS_OPTION_UFACTOR] = 30;

    /* 0 does not explicitly minimize maximum connectivity (default),
     * or 1 explicitly minimizes maximum connectivity. */
    // options[METIS_OPTION_MINCONN] = 0;

    /* 0 does not force contiguous partitions (default), or 1 forces
     * contiguous partitions. */
    // options[METIS_OPTION_CONTIG] = 0;

    /* seed for the random number generator */
    // options[METIS_OPTION_SEED] = 0;

    /* 0 for C-style numbering that starts from 0 (default), or 1 for
     * Fortran-style numbering that starts from 1 */
    // options[METIS_OPTION_NUMBERING] = 0;

    /*
     * METIS_DBG_INFO (1)
     * METIS_DBG_TIME (2)
     * METIS_DBG_COARSEN (4)
     * METIS_DBG_REFINE (8)
     * METIS_DBG_IPART (16)
     * METIS_DBG_MOVEINFO (32)
     * METIS_DBG_SEPINFO (64)
     * METIS_DBG_CONNINFO (128)
     * METIS_DBG_CONTIGINFO (256)
     */
    options[METIS_OPTION_DBGLVL] = 1;

    /* the number of vertices in the graph */
    idx_t nvtxs = num_vertices;

    /* the number of balancing constraints, should be at least 1 */
    idx_t ncon = 1;

    /* adjacency structure of the graph (row pointers) */
    idx_t * xadj = NULL;
    xadj = malloc((nvtxs+1) * sizeof(idx_t));
    if (!xadj) return MTX_ERR_ERRNO;
    for (idx_t i = 0; i <= nvtxs; i++) xadj[i] = 0;
    if (symmetric) {
        for (int64_t k = 0; k < num_nonzeros; k++) {
            xadj[rowidx[k]+1]++;
            if (rowidx[k] != colidx[k]) xadj[colidx[k]+1]++;
        }
    } else if (square) {
    }
    if (!symmetric) {
        /* sort */
    }

    } else if (square) {
        /*
         * TODO: handle unsymmetric matrices via symmetrisation
         *
         * add all nonzeros and their symmetric counterparts, then
         * sort and remove duplicates
         */

        int64_t num_sym_nonzeros = 0;
        int * symrowidx = malloc(2*num_nonzeros*sizeof(int));
        if (!symrowidx) { free(xadj); return MTX_ERR_ERRNO; }
        int * symcolidx = malloc(2*num_nonzeros*sizeof(int));
        if (!symcolidx) { free(symrowidx); free(xadj); return MTX_ERR_ERRNO; }
        for (int64_t k = 0, l = 0; k < num_nonzeros; k++) {
            symrowidx[l] = rowidx[k]; symcolidx[l] = colidx[k]; l++;
            if (rowidx[k] != colidx[k]) {
                symrowidx[l] = colidx[k]; symcolidx[l] = rowidx[k]; l++;
            }
        }
        err = radix_sort_int64x2(
            2*num_nonzeros, symrowidx, symcolidx, NULL);
        if (err) { free(symcolidx); free(symrowidx); return err; }

        free(symcolidx); free(symrowidx);
    } else {
    }

    for (int j = 0; j < num_columns; j++) {
            for (int64_t k = colptr[j]; k < colptr[j+1]; k++) {
                if (j < rowidx[k]) xadj[rowidx[k]+1]++;
            }
        }
        if (!square) {
            for (int i = 0; i < num_rows; i++) {
                for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++) {
                    if (i < colidx[k]) xadj[num_rows+colidx[k]+1]++;
                }
            }
            for (int j = 0; j < num_columns; j++) {
                for (int64_t k = colptr[j]; k < colptr[j+1]; k++) {
                    if (j < rowidx[k]) xadj[num_rows+j+1]++;
                }
            }
        }
    }
    for (idx_t i = 1; i <= nvtxs; i++) xadj[i] += xadj[i-1];

    /* adjacency structure of the graph (column offsets) */
    idx_t * adjncy = NULL;
    adjncy = malloc(xadj[nvtxs] * sizeof(idx_t));
    if (!adjncy) { free(xadj); return MTX_ERR_ERRNO; }
#if 0
    for (int i = 0; i < num_rows; i++) {
        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++) {
            if (true || i != colidx[k]) adjncy[xadj[i]++] = colidx[k];
        }
    }
#else
    for (int i = 0; i < num_rows; i++) {
        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++) {
            if (i < colidx[k]) adjncy[xadj[i]++] = colidx[k];
        }
    }
    for (int j = 0; j < num_columns; j++) {
        for (int64_t k = colptr[j]; k < colptr[j+1]; k++) {
            if (j < rowidx[k]) adjncy[xadj[rowidx[k]]++] = j;
        }
    }
    if (!square) {
        for (int i = 0; i < num_rows; i++) {
            for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++) {
                if (i < colidx[k]) adjncy[xadj[num_rows+colidx[k]]++] = colidx[k];
            }
        }
        for (int j = 0; j < num_columns; j++) {
            for (int64_t k = colptr[j]; k < colptr[j+1]; k++) {
                if (j < rowidx[k]) adjncy[xadj[num_rows+j]++] = j;
            }
        }
    }
#endif
    for (idx_t i = nvtxs; i > 0; i--) xadj[i] = xadj[i-1];
    xadj[0] = 0;

    /* if (square) { */
    /*     for (int64_t k = 0; k < rowptr[num_rows]; k++) adjncy[k] = colidx[k]; */
    /* } else { */
    /*     for (int64_t k = 0; k < rowptr[num_rows]; k++) */
    /*         adjncy[k] = num_rows+colidx[k]; */
    /*     for (int64_t k = 0; k < colptr[num_columns]; k++) */
    /*         adjncy[rowptr[num_rows]+k] = rowidx[k]; */
    /* } */

/* #ifdef DEBUG_METIS */
    fprintf(stderr, "nvtxs=%"PRIDX", xadj=[", nvtxs);
    for (idx_t i = 0; i <= nvtxs; i++) fprintf(stderr, " %"PRIDX, xadj[i]);
    fprintf(stderr, "]\n");
    fprintf(stderr, "adjncy=[");
    for (idx_t i = 0; i < nvtxs; i++) {
        fprintf(stderr, " (%"PRIDX")", i);
        for (idx_t k = xadj[i]; k < xadj[i+1]; k++)
            fprintf(stderr, " %"PRIDX, adjncy[k]);
    }
    fprintf(stderr, "]\n");
/* #endif */

    idx_t * vwgt = NULL;    /* the weights of the vertices */
    idx_t * vsize = NULL;   /* the size of the vertices for computing
                             * the total communication volume */
    idx_t * adjwgt = NULL;  /* the weights of the edges */

    /* the number of parts to partition the graph */
    idx_t nparts = num_parts;

    /* This is an array of size nparts×ncon that specifies the desired
     * weight for each partition and constraint. The target partition
     * weight for the ith partition and jth constraint is specified at
     * tpwgts[i*ncon+j] (the numbering for both partitions and
     * constraints starts from 0). For each constraint, the sum of the
     * tpwgts[] entries must be 1.0 (i.e., 􏰓i tpwgts[i ∗ ncon + j] =
     * 1.0).  A NULL value can be passed to indicate that the graph
     * should be equally divided among the partitions. */
    real_t * tpwgts = malloc(nparts * sizeof(real_t));
    if (!tpwgts) {
        free(adjncy);
        free(xadj);
        return MTX_ERR_ERRNO;
    }
    for (idx_t p = 0; p < nparts; p++) tpwgts[p] = 1.0 / nparts;

    /* This is an array of size ncon that specifies the allowed load
     * imbalance tolerance for each constraint. For the ith partition
     * and jth constraint the allowed weight is the
     * ubvec[j]*tpwgts[i*ncon+j] fraction of the jth’s constraint
     * total weight. The load imbalances must be greater than 1.0.  A
     * NULL value can be passed indicating that the load imbalance
     * tolerance for each constraint should be 1.001 for ncon=1 or
     * 1.01 otherwise. */
    real_t ubvec[1] = {1};

    /* edge-cut or the total communication volume of the partitioning
     * solution */
    idx_t objval = 0;

    /* This is a vector of size nvtxs that upon successful completion
     * stores the partition vector of the graph. The numbering of this
     * vector starts from either 0 or 1, depending on the value of
     * options[METIS OPTION NUMBERING]. */
    idx_t * part = NULL;
    bool free_part = false;
    if (square && sizeof(*dstpart) == sizeof(*part)) {
        part = (idx_t *) dstpart;
    } else {
        part = malloc(nvtxs * sizeof(idx_t));
        if (!part) {
            free(tpwgts);
            free(adjncy);
            free(xadj);
            return MTX_ERR_ERRNO;
        }
        free_part = true;
    }

    // int METIS_PartGraphKway(
    //     idx_t *nvtxs, idx_t *ncon, idx_t *xadj, idx_t *adjncy,
    //     idx_t *vwgt, idx_t *vsize, idx_t *adjwgt, idx_t *nparts,
    //     real_t *tpwgts, real_t ubvec,
    //     idx_t *options, idx_t *objval, idx_t *part);

    err = METIS_PartGraphKway(
        &nvtxs, &ncon, xadj, adjncy, vwgt, vsize, adjwgt,
        &nparts, tpwgts, ubvec, options, &objval, part);
    if (err == METIS_ERROR_INPUT) { err = MTX_ERR_METIS_INPUT; }
    else if (err == METIS_ERROR_MEMORY) { err = MTX_ERR_METIS_MEMORY; }
    else if (err == METIS_ERROR) { err = MTX_ERR_METIS; }
    else { err = MTX_SUCCESS; }

    /* TODO: fix for non-square matrices */
    if (free_part) {
        for (idx_t i = 0; i < num_vertices; i++) dstpart[i] = part[i];
        free(part);
    }

    free(tpwgts);
    free(adjncy);
    free(xadj);
    return err;
#endif
}