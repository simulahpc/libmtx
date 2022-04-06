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
 * Last modified: 2022-02-23
 *
 * Data structures for distributed matrix-vector multiplication.
 */

#include <libmtx/libmtx-config.h>

#ifdef LIBMTX_HAVE_MPI
#include <libmtx/error.h>
#include <libmtx/field.h>
#include <libmtx/matrix/distmatrix.h>
#include <libmtx/matrix/distmatrixgemv.h>
#include <libmtx/matrix/matrix.h>
#include <libmtx/mtxfile/header.h>
#include <libmtx/mtxfile/mtxdistfile.h>
#include <libmtx/mtxfile/mtxfile.h>
#include <libmtx/mtxfile/size.h>
#include <libmtx/precision.h>
#include <libmtx/util/partition.h>
#include <libmtx/util/transpose.h>
#include <libmtx/vector/distvector.h>
#include <libmtx/vector/vector.h>

#include <mpi.h>

#include <errno.h>

#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

/*
 * Memory management
 */

/**
 * ‘mtxdistmatrixgemv_free()’ frees storage allocated for a
 * distributed matrix-vector multiplication.
 */
void mtxdistmatrixgemv_free(
    struct mtxdistmatrixgemv * gemv)
{
    mtxvector_free(gemv->xe);
    mtxvector_free(gemv->xi);
    free(gemv->xi);
    mtxmatrix_free(gemv->Ae);
    mtxmatrix_free(gemv->Ai);
    free(gemv->Ai);
    mtxpartition_free(&gemv->colexthalo);
}

/**
 * ‘mtxdistmatrixgemv_alloc_copy()’ allocates storage for a copy of a
 * distributed matrix-vector multiplication without initialising the
 * underlying values.
 */
int mtxdistmatrixgemv_alloc_copy(
    struct mtxdistmatrixgemv * dst,
    const struct mtxdistmatrixgemv * src,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistmatrixgemv_init_copy()’ creates a copy of a distributed
 * matrix-vector multiplication.
 */
int mtxdistmatrixgemv_init_copy(
    struct mtxdistmatrixgemv * dst,
    const struct mtxdistmatrixgemv * src,
    struct mtxdisterror * disterr);

/*
 * Initialisation
 */

/**
 * ‘mtxdistmatrixgemv_init()’ allocates and initialises data
 * structures for distributed matrix-vector multiplication.
 */
int mtxdistmatrixgemv_init(
    struct mtxdistmatrixgemv * gemv,
    enum mtxtransposition trans,
    const struct mtxdistmatrix * A,
    const struct mtxdistvector * x,
    struct mtxdistvector * y,
    struct mtxdisterror * disterr)
{
    int err;
    int result;

    /* verify that the matrix and vectors have the same MPI communicator */
    MPI_Comm_compare(A->parent, x->comm, &result);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    err = result != MPI_IDENT ? MTX_ERR_INCOMPATIBLE_MPI_COMM : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    MPI_Comm_compare(A->parent, y->comm, &result);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    err = result != MPI_IDENT ? MTX_ERR_INCOMPATIBLE_MPI_COMM : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    err = (A->comm_size != x->comm_size || A->comm_size != y->comm_size)
        ? MTX_ERR_INCOMPATIBLE_MPI_COMM : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    err = (A->rank != x->rank || A->rank != y->rank)
        ? MTX_ERR_INCOMPATIBLE_MPI_COMM : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;

    MPI_Comm comm = A->parent;
    int comm_size = A->comm_size;
    int rank = A->rank;
    int P = A->num_process_rows;
    int Q = A->num_process_columns;
    int R = A->comm_size;
    int p = A->colrank;
    int q = A->rowrank;

    /* TODO: Implement transposed matrix-vector multiplication */
    err = trans != mtx_notrans ? MTX_ERR_INVALID_TRANSPOSITION : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;

    /*
     * Step 1: partition matrix columns into “interior” and “exterior halo”.
     *
     * On each process, the “interior” consists of matrix columns for
     * which the current process already owns the corresponding source
     * vector element, whereas the “exterior halo” consists of matrix
     * columns where the corresponding source vector elements are
     * owned by other processes. Each process must therefore receive
     * source vector elements for its “exterior halo” from other
     * processes during the “expand” phase of a distributed
     * matrix-vector multiplication.
     *
     * Because the underlying matrix block on each process is often
     * sparse, it is important that we only consider nonzero (i.e.,
     * non-empty) matrix columns.
     */

    /* find nonzero columns in the local matrix block */
    int num_nzcols;
    err = mtxmatrix_nzcols(&A->interior, &num_nzcols, 0, NULL);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    int * nzcols = malloc(num_nzcols * sizeof(int));
    err = !nzcols ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    err = mtxmatrix_nzcols(&A->interior, NULL, num_nzcols, nzcols);
    if (mtxdisterror_allreduce(disterr, err)) {
        free(nzcols);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    /* convert nonzero column numbers to global column numbers */
    int64_t * global_nzcols = malloc(num_nzcols * sizeof(int64_t));
    err = !global_nzcols ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(nzcols);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    for (int j = 0; j < num_nzcols; j++)
        global_nzcols[j] = nzcols[j];
    err = mtxpartition_globalidx(
        &A->colpart, p, num_nzcols, global_nzcols, global_nzcols);
    if (mtxdisterror_allreduce(disterr, err)) {
        free(global_nzcols);
        free(nzcols);
        return MTX_ERR_MPI_COLLECTIVE;
    }

#if 0
    for (int r = 0; r < R; r++) {
        if (r == x->rank) {
            fprintf(stderr, "%s:%d: global_nzcols=[", __FILE__, __LINE__);
            for (int j = 0; j < num_nzcols; j++)
                fprintf(stderr, " %"PRId64, global_nzcols[j]);
            fprintf(stderr, "]\n");
        }
        MPI_Barrier(x->comm);
    }
#endif

    /* For each nonzero matrix column, find the rank of the process
     * that owns the corresponding element in the source vector. The
     * current process will later receive those source vector elements
     * from the processes that own them. */
    int * recvrank = malloc(num_nzcols * sizeof(int));
    err = !recvrank ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(global_nzcols);
        free(nzcols);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    err = mtxpartition_assign(
        &x->rowpart, num_nzcols, global_nzcols, recvrank, NULL);
    if (mtxdisterror_allreduce(disterr, err)) {
        free(recvrank);
        free(global_nzcols);
        free(nzcols);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    /* partition the nonzero matrix columns into “interior” and
     * “exterior halo”. */
    int * exthalo = malloc(num_nzcols * sizeof(int));
    err = !exthalo ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(recvrank);
        free(global_nzcols);
        free(nzcols);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    for (int j = 0; j < num_nzcols; j++)
        exthalo[j] = recvrank[j] == rank ? 0 : 1;
    err = mtxpartition_init_custom(
        &gemv->colexthalo, A->colpart.size, 2, exthalo, NULL, NULL);
    if (mtxdisterror_allreduce(disterr, err)) {
        free(exthalo);
        free(recvrank);
        free(global_nzcols);
        free(nzcols);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    int64_t exthalosize = gemv->colexthalo.part_sizes[1];

    struct mtxmatrix * Ai = malloc(2 * sizeof(struct mtxmatrix));
    err = !Ai ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxpartition_free(&gemv->colexthalo);
        free(exthalo);
        free(recvrank);
        free(global_nzcols);
        free(nzcols);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    struct mtxmatrix * Ae = &Ai[1];
    err = mtxmatrix_partition(Ai, &A->interior, NULL, &gemv->colexthalo);
    if (mtxdisterror_allreduce(disterr, err)) {
        free(Ai);
        mtxpartition_free(&gemv->colexthalo);
        free(exthalo);
        free(recvrank);
        free(global_nzcols);
        free(nzcols);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    /*
     * Step 2: inform owners about which source vector elements to
     * send to other processes.
     *
     * At this point, each process knows the global offset and the
     * rank of the owning process for every source vector element in
     * their “exterior halo”. Thus, we perform an all-to-all exchange
     * to inform each process about source vector elements needed by
     * other processes.
     */
    int * sendcounts = malloc(comm_size * sizeof(int));
    err = !sendcounts ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxmatrix_free(Ae); mtxmatrix_free(Ai); free(Ai);
        mtxpartition_free(&gemv->colexthalo);
        free(exthalo);
        free(recvrank);
        free(global_nzcols);
        free(nzcols);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    for (int j = 0; j < num_nzcols; j++) {
        if (rank != recvrank[j])
            sendcounts[recvrank[j]]++;
    }

    disterr->mpierrcode = MPI_Alltoall(
        sendbuf, 1, MPI_INT, recvbuf, 1, MPI_INT, comm);
    if (mtxdisterror_allreduce(disterr, err)) {
        free(sendcounts);
        mtxmatrix_free(Ae); mtxmatrix_free(Ai); free(Ai);
        mtxpartition_free(&gemv->colexthalo);
        free(exthalo);
        free(recvrank);
        free(global_nzcols);
        free(nzcols);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    free(sendcounts);

    free(exthalo);
    free(recvrank);
    free(global_nzcols);
    free(nzcols);

    /*
     * Step 2: partition source vector into “interior” and “interior halo”.
     *
     * On each process, the “interior” consists of source vector
     * elements for which the current process already owns the
     * corresponding matrix columns, whereas the “interior halo”
     * consists of source vector elements where the corresponding
     * matrix columns are owned by other processes. Each process must
     * therefore send source vector elements belonging to its
     * “interior halo” to other processes during the “expand” phase of
     * a distributed matrix-vector multiplication.
     *
     * Because the underlying matrix block on each process may be
     * sparse, it is important that we only consider nonzero (i.e.,
     * non-empty) matrix columns.
     */

    /* partition the local source vector block */
    struct mtxvector * xi = malloc(2 * sizeof(struct mtxvector));
    err = !xi ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxmatrix_free(Ae); mtxmatrix_free(Ai); free(Ai);
        mtxpartition_free(&gemv->colexthalo);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    struct mtxvector * xe = &xi[1];
    err = mtxvector_partition(xi, &x->interior, &gemv->colexthalo);
    if (mtxdisterror_allreduce(disterr, err)) {
        free(xi);
        mtxmatrix_free(Ae); mtxmatrix_free(Ai); free(Ai);
        mtxpartition_free(&gemv->colexthalo);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    gemv->comm = x->comm;
    gemv->comm_size = x->comm_size;
    gemv->rank = x->rank;
    gemv->A = A;
    gemv->x = x;
    gemv->y = y;
    gemv->Ai = Ai;
    gemv->Ae = Ae;
    gemv->xi = xi;
    gemv->xe = xe;
    return MTX_SUCCESS;
}

/*
 * Matrix-vector multiplication (Level 2 BLAS operations)
 */

static int mtxmatrix_expand(
    struct mtxdistmatrixgemv * gemv)
{
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistmatrixgemv_wait()’ waits for a matrix-vector multiplication
 * operation to complete.
 *
 * Matrix-vector multiplications may be performed asynchronously to
 * overlap communication with computation. Therefore, it may be
 * necessary to explicitly add synchronisation to wait for a
 * particular operation to complete.
 */
int mtxdistmatrixgemv_wait(
    struct mtxdistmatrixgemv * gemv,
    struct mtxdisterror * disterr)
{
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistmatrixgemv_sgemv()’ multiplies a matrix ‘A’ or its
 * transpose ‘A'’ by a real scalar ‘alpha’ (‘α’) and a vector ‘x’,
 * before adding the result to another vector ‘y’ multiplied by
 * another real scalar ‘beta’ (‘β’). That is, ‘y = α*A*x + β*y’ or ‘y
 * = α*A'*x + β*y’.
 *
 * The scalars ‘alpha’ and ‘beta’ are given as single precision
 * floating point numbers.
 */
int mtxdistmatrixgemv_sgemv(
    struct mtxdistmatrixgemv * gemv,
    float alpha,
    float beta,
    int64_t * num_flops,
    struct mtxdisterror * disterr)
{
    int err;

    /* perform the “expand” communication phase to obtain remote
     * source vector elements that are needed to multiply the
     * “exterior” part of the local matrix block. */
    err = mtxmatrix_expand(gemv);

    /* multiply the “exterior” part of the local matrix block */
    err = mtxmatrix_sgemv(
        mtx_notrans, alpha, gemv->Ae, gemv->xe,
        beta, &gemv->y->interior, num_flops);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;

    /* multiply the “interior” part of the local matrix block */
    err = mtxmatrix_sgemv(
        mtx_notrans, alpha, gemv->Ai, gemv->xi,
        beta, &gemv->y->interior, num_flops);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistmatrixgemv_dgemv()’ multiplies a matrix ‘A’ or its
 * transpose ‘A'’ by a real scalar ‘alpha’ (‘α’) and a vector ‘x’,
 * before adding the result to another vector ‘y’ multiplied by
 * another real scalar ‘beta’ (‘β’). That is, ‘y = α*A*x + β*y’ or ‘y
 * = α*A'*x + β*y’.
 *
 * The scalars ‘alpha’ and ‘beta’ are given as double precision
 * floating point numbers.
 */
int mtxdistmatrixgemv_dgemv(
    struct mtxdistmatrixgemv * gemv,
    double alpha,
    double beta,
    int64_t * num_flops,
    struct mtxdisterror * disterr)
{
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistmatrixgemv_cgemv()’ multiplies a complex-valued matrix ‘A’,
 * its transpose ‘A'’ or its conjugate transpose ‘Aᴴ’ by a complex
 * scalar ‘alpha’ (‘α’) and a vector ‘x’, before adding the result to
 * another vector ‘y’ multiplied by another complex scalar ‘beta’
 * (‘β’).  That is, ‘y = α*A*x + β*y’, ‘y = α*A'*x + β*y’ or ‘y =
 * α*Aᴴ*x + β*y’.
 *
 * The complex scalars ‘alpha’ and ‘beta’ are given as pairs of single
 * precision floating point numbers.
 */
int mtxdistmatrixgemv_cgemv(
    struct mtxdistmatrixgemv * gemv,
    float alpha[2],
    float beta[2],
    int64_t * num_flops,
    struct mtxdisterror * disterr)
{
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistmatrixgemv_zgemv()’ multiplies a complex-valued matrix ‘A’,
 * its transpose ‘A'’ or its conjugate transpose ‘Aᴴ’ by a complex
 * scalar ‘alpha’ (‘α’) and a vector ‘x’, before adding the result to
 * another vector ‘y’ multiplied by another complex scalar ‘beta’
 * (‘β’).  That is, ‘y = α*A*x + β*y’, ‘y = α*A'*x + β*y’ or ‘y =
 * α*Aᴴ*x + β*y’.
 *
 * The complex scalars ‘alpha’ and ‘beta’ are given as pairs of double
 * precision floating point numbers.
 */
int mtxdistmatrixgemv_zgemv(
    struct mtxdistmatrixgemv * gemv,
    double alpha[2],
    double beta[2],
    int64_t * num_flops,
    struct mtxdisterror * disterr)
{
    return MTX_SUCCESS;
}
#endif
