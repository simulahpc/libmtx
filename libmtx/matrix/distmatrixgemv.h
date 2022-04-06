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

#ifndef LIBMTX_MATRIX_DISTMATRIXGEMV_H
#define LIBMTX_MATRIX_DISTMATRIXGEMV_H

#include <libmtx/libmtx-config.h>

#ifdef LIBMTX_HAVE_MPI
#include <libmtx/util/transpose.h>

#include <mpi.h>

#ifdef LIBMTX_HAVE_LIBZ
#include <zlib.h>
#endif

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

struct mtxdisterror;
struct mtxdistmatrix;
struct mtxdistvector;

/**
 * ‘mtxdistmatrixgemv’ is a data structure for persistent or repeated
 * matrix-vector multiplication operations with matrices and vectors
 * distributed across multiple processes and MPI being used for
 * communication.
 *
 * Processes are arranged in a two-dimensional grid, and matrices are
 * distributed among processes in rectangular blocks according to
 * specified partitionings of the matrix rows and columns.
 */
struct mtxdistmatrixgemv
{
    /**
     * ‘comm’ is an MPI communicator for processes among which the
     * matrix and vector are distributed. This is equal to
     * ‘A->parent’, ‘x->comm’ and ‘y->comm’.
     */
    MPI_Comm comm;

    /**
     * ‘comm_size’ is the size of the MPI communicator. This is equal
     * to the number of parts in the partitioning of the vectors ‘x’
     * and ‘y’.
     */
    int comm_size;

    /**
     * ‘rank’ is the rank of the current process.
     */
    int rank;

    /**
     * ‘A’ is a pointer to a distributed matrix.
     */
    const struct mtxdistmatrix * A;

    /**
     * ‘x’ is a pointer to a distributed input or source vector.
     */
    const struct mtxdistvector * x;

    /**
     * ‘y’ is a pointer to a distributed output or destination vector.
     */
    struct mtxdistvector * y;

    /**
     * ‘colexthalo’ is the partition used to divide columns of the
     * local matrix block into “interior” and “exterior halo” parts.
     */
    struct mtxpartition colexthalo;

    /**
     * ‘colinthalo’ is the partition used to divide local source
     * vector entries into “interior” and “interior halo” parts.
     */
    struct mtxpartition colinthalo;

    /**
     * ‘Ai’ is the “interior” part of the local matrix block.
     */
    struct mtxmatrix * Ai;

    /**
     * ‘Ae’ is the “exterior” part of the local matrix.
     */
    struct mtxmatrix * Ae;

    /**
     * ‘xi’ is the “interior” part of the local source vector.
     */
    struct mtxvector * xi;

    /**
     * ‘xe’ is the “exterior” part of the local source vector.
     */
    struct mtxvector * xe;
};

/*
 * Memory management
 */

/**
 * ‘mtxdistmatrixgemv_free()’ frees storage allocated for a
 * distributed matrix-vector multiplication.
 */
void mtxdistmatrixgemv_free(
    struct mtxdistmatrixgemv * gemv);

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
    struct mtxdisterror * disterr);

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
    struct mtxdisterror * disterr);

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
    struct mtxdisterror * disterr);

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
    struct mtxdisterror * disterr);

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
    struct mtxdisterror * disterr);

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
    struct mtxdisterror * disterr);
#endif

#endif
