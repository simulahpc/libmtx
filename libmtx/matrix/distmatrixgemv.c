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
    for (int r = 0; r < gemv->comm_size; r++) mtxvector_free(&gemv->xr[r]);
    free(gemv->xr);
    for (int r = 0; r < gemv->comm_size; r++) mtxmatrix_free(&gemv->Ar[r]);
    free(gemv->Ar);
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
 * ‘mtxdistmatrixgemv_init()’ allocates and initialises a data
 * structure for distributed matrix-vector multiplication.
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

    /* verify that the vectors use the same MPI communicator */
    disterr->mpierrcode = MPI_Comm_compare(x->comm, y->comm, &result);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    err = result != MPI_IDENT && result != MPI_CONGRUENT
        ? MTX_ERR_INCOMPATIBLE_MPI_COMM : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;

    /* verify that the matrix and vectors come from the same MPI communicator */
    disterr->mpierrcode = MPI_Comm_compare(A->parent, x->comm, &result);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    err = result != MPI_IDENT && result != MPI_CONGRUENT
        ? MTX_ERR_INCOMPATIBLE_MPI_COMM : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;

    int P = A->num_process_rows;
    int Q = A->num_process_columns;
    int R = x->comm_size;

    /* TODO: Implement transposed matrix-vector multiplication */
    err = trans != mtx_notrans ? MTX_ERR_INVALID_TRANSPOSITION : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;

    /* Find the nonzero columns of the matrix block belonging to the
     * current process. */
    int num_nonzero_columns;
    err = mtxmatrix_nzcols(&A->interior, &num_nonzero_columns, 0, NULL);
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;
    int * nonzero_columns = malloc(num_nonzero_columns * sizeof(int));
    err = !nonzero_columns ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;
    err = mtxmatrix_nzcols(&A->interior, NULL, num_nonzero_columns, nonzero_columns);
    if (mtxdisterror_allreduce(disterr, err)) {
        free(nonzero_columns);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    /* convert nonzero column numbers to global column numbers */
    int64_t * global_nonzero_columns = malloc(
        num_nonzero_columns * sizeof(int64_t));
    err = !global_nonzero_columns ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(nonzero_columns);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    for (int j = 0; j < num_nonzero_columns; j++)
        global_nonzero_columns[j] = nonzero_columns[j];

    for (int r = 0; r < R; r++) {
        if (r == x->rank) {
            fprintf(stderr, "%s:%d: global_nonzero_columns=[", __FILE__, __LINE__);
            for (int j = 0; j < num_nonzero_columns; j++)
                fprintf(stderr, " %"PRId64, global_nonzero_columns[j]);
            fprintf(stderr, "]\n");
        }
        MPI_Barrier(x->comm);
    }

    err = mtxpartition_globalidx(
        &A->colpart, A->colrank, num_nonzero_columns,
        global_nonzero_columns, global_nonzero_columns);
    if (mtxdisterror_allreduce(disterr, err)) {
        free(global_nonzero_columns);
        free(nonzero_columns);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    int * parts = nonzero_columns;
    nonzero_columns = NULL;

    /* convert global column indices of nonzero columns to the rank of
     * the process that owns the corresponding entry of the source
     * vector */
    err = mtxpartition_assign(
        &x->rowpart, num_nonzero_columns, global_nonzero_columns, parts, NULL);
    if (mtxdisterror_allreduce(disterr, err)) {
        free(parts);
        free(global_nonzero_columns);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    free(global_nonzero_columns);

    /* Partition the nonzero matrix columns according to the process
     * that owns the corresponding entry of the source vector. */
    struct mtxpartition Acolpart;
    err = mtxpartition_init_custom(
        &Acolpart, A->colpart.size, R, parts, NULL, NULL);
    if (mtxdisterror_allreduce(disterr, err)) {
        free(parts);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    free(parts);

    /* TODO: We should only need to allocate for non-empty blocks
     * below, otherwise it is not going to scale when the number of
     * processes increases. */

    /* partition the local matrix block */
    struct mtxmatrix * Ar = malloc(R * sizeof(struct mtxmatrix));
    err = !Ar ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxpartition_free(&Acolpart);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    err = mtxmatrix_partition(
        Ar, &A->interior, NULL, &Acolpart);
    if (mtxdisterror_allreduce(disterr, err)) {
        free(Ar);
        mtxpartition_free(&Acolpart);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    /* partition the local source vector block */
    struct mtxvector * xr = malloc(R * sizeof(struct mtxvector));
    err = !xr ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        for (int r = 0; r < R; r++) mtxmatrix_free(&Ar[r]);
        free(Ar);
        mtxpartition_free(&Acolpart);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    err = mtxvector_partition(
        xr, &x->interior, &Acolpart);
    if (mtxdisterror_allreduce(disterr, err)) {
        free(xr);
        for (int r = 0; r < R; r++) mtxmatrix_free(&Ar[r]);
        free(Ar);
        mtxpartition_free(&Acolpart);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    mtxpartition_free(&Acolpart);

    gemv->comm = x->comm;
    gemv->comm_size = x->comm_size;
    gemv->rank = x->rank;
    gemv->A = A;
    gemv->x = x;
    gemv->y = y;
    gemv->Ar = Ar;
    gemv->xr = xr;
    return MTX_SUCCESS;
}

/*
 * Matrix-vector multiplication (Level 2 BLAS operations)
 */

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
    struct mtxdisterror * disterr)
{
    int err;
    for (int r = 0; r < gemv->comm_size; r++) {
        err = mtxmatrix_sgemv(
            mtx_notrans, alpha, &gemv->Ar[r], &gemv->xr[r],
            beta, &gemv->y->interior);
        if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    }
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
    struct mtxdisterror * disterr)
{
    int err;
    for (int r = 0; r < gemv->comm_size; r++) {
        err = mtxmatrix_dgemv(
            mtx_notrans, alpha, &gemv->Ar[r], &gemv->xr[r],
            beta, &gemv->y->interior);
        if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    }
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
    struct mtxdisterror * disterr)
{
    int err;
    for (int r = 0; r < gemv->comm_size; r++) {
        err = mtxmatrix_cgemv(
            mtx_notrans, alpha, &gemv->Ar[r], &gemv->xr[r],
            beta, &gemv->y->interior);
        if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    }
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
    struct mtxdisterror * disterr)
{
    int err;
    for (int r = 0; r < gemv->comm_size; r++) {
        err = mtxmatrix_zgemv(
            mtx_notrans, alpha, &gemv->Ar[r], &gemv->xr[r],
            beta, &gemv->y->interior);
        if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    }
    return MTX_SUCCESS;
}
#endif
