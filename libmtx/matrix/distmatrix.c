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
 * Last modified: 2022-04-27
 *
 * Data structures for distributed matrices.
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
#include <libmtx/util/merge.h>
#include <libmtx/util/partition.h>
#include <libmtx/vector/distvector.h>
#include <libmtx/vector/vector.h>

#include <mpi.h>

#ifdef LIBMTX_HAVE_LIBZ
#include <zlib.h>
#endif

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
 * ‘mtxdistmatrix_free()’ frees storage allocated for a matrix.
 */
void mtxdistmatrix_free(
    struct mtxdistmatrix * distmatrix)
{
    free(distmatrix->colmap);
    free(distmatrix->rowmap);
    mtxmatrix_free(&distmatrix->interior);
    mtxpartition_free(&distmatrix->colpart);
    mtxpartition_free(&distmatrix->rowpart);
}

static int mtxdistmatrix_init_comm(
    struct mtxdistmatrix * A,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    A->comm = comm;
    disterr->mpierrcode = MPI_Comm_size(comm, &A->comm_size);
    int err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    disterr->mpierrcode = MPI_Comm_rank(comm, &A->rank);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

static int mtxdistmatrix_init_partitions(
    struct mtxdistmatrix * distmatrix,
    int num_local_rows,
    int num_local_columns,
    const struct mtxpartition * rowpart,
    const struct mtxpartition * colpart,
    struct mtxdisterror * disterr)
{
    int err = MTX_SUCCESS;
    MPI_Comm comm = distmatrix->comm;
    int comm_size = distmatrix->comm_size;
    int rank = distmatrix->rank;

    int num_row_parts;
    int num_col_parts;
    if (rowpart && colpart) {
        num_row_parts = rowpart->num_parts;
        num_col_parts = colpart->num_parts;
    } else if (rowpart) {
        num_row_parts = rowpart->num_parts;
        num_col_parts = 1;
    } else if (colpart) {
        num_row_parts = 1;
        num_col_parts = colpart->num_parts;
    } else {
        num_row_parts = comm_size;
        num_col_parts = 1;
    }

    int p = rank / num_col_parts;
    int q = rank % num_col_parts;
    int num_parts = num_row_parts * num_col_parts;
    if (num_parts > comm_size)
        err = MTX_ERR_INCOMPATIBLE_PARTITION;
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;

    /* Check that blocks are compatible in size */
    if (num_row_parts > 1) {
        if (p == 0) {
            for (int s = 1; s < num_row_parts && !err; s++) {
                int num_local_columns_s;
                disterr->mpierrcode = MPI_Recv(
                    &num_local_columns_s, 1, MPI_INT,
                    s*num_col_parts+q, 0, comm, MPI_STATUS_IGNORE);
                if (disterr->mpierrcode) err = MTX_ERR_MPI;
                else if (num_local_columns != num_local_columns_s)
                    err = MTX_ERR_INCOMPATIBLE_PARTITION;
            }
        } else {
            disterr->mpierrcode = MPI_Send(
                &num_local_columns, 1, MPI_INT,
                0*num_col_parts+q, 0, comm);
            if (disterr->mpierrcode) err = MTX_ERR_MPI;
        }
    }
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;
    if (num_col_parts > 1) {
        if (q == 0) {
            for (int t = 1; t < num_col_parts && !err; t++) {
                int num_local_rows_t;
                disterr->mpierrcode = MPI_Recv(
                    &num_local_rows_t, 1, MPI_INT,
                    p*num_col_parts+t, 0, comm, MPI_STATUS_IGNORE);
                if (disterr->mpierrcode) err = MTX_ERR_MPI;
                else if (num_local_rows != num_local_rows_t)
                    err = MTX_ERR_INCOMPATIBLE_PARTITION;
            }
        } else {
            disterr->mpierrcode = MPI_Send(
                &num_local_rows, 1, MPI_INT,
                p*num_col_parts+0, 0, comm);
            if (disterr->mpierrcode) err = MTX_ERR_MPI;
        }
    }
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;

    if (rowpart && p < num_row_parts && rowpart->part_sizes[p] != num_local_rows)
        err = MTX_ERR_INCOMPATIBLE_PARTITION;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    if (colpart && q < num_col_parts && colpart->part_sizes[q] != num_local_columns)
        err = MTX_ERR_INCOMPATIBLE_PARTITION;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;

    if (rowpart && colpart) {
        int64_t size = num_local_rows * num_local_columns;
        disterr->mpierrcode = MPI_Allreduce(
            MPI_IN_PLACE, &size, 1, MPI_INT64_T, MPI_SUM, comm);
        if (disterr->mpierrcode) err = MTX_ERR_MPI;
        else if (rowpart->size * colpart->size != size)
            err = MTX_ERR_INCOMPATIBLE_PARTITION;
        if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
        err = mtxpartition_copy(&distmatrix->rowpart, rowpart);
        if (mtxdisterror_allreduce(disterr, err))
            return MTX_ERR_MPI_COLLECTIVE;
        err = mtxpartition_copy(&distmatrix->colpart, colpart);
        if (mtxdisterror_allreduce(disterr, err)) {
            mtxpartition_free(&distmatrix->rowpart);
            return MTX_ERR_MPI_COLLECTIVE;
        }
    } else if (rowpart) {
        err = mtxpartition_copy(&distmatrix->rowpart, rowpart);
        if (mtxdisterror_allreduce(disterr, err))
            return MTX_ERR_MPI_COLLECTIVE;
        err = mtxpartition_init_singleton(
            &distmatrix->colpart, num_local_columns);
        if (mtxdisterror_allreduce(disterr, err)) {
            mtxpartition_free(&distmatrix->rowpart);
            return MTX_ERR_MPI_COLLECTIVE;
        }
    } else if (colpart) {
        err = mtxpartition_init_singleton(
            &distmatrix->rowpart, num_local_rows);
        if (mtxdisterror_allreduce(disterr, err))
            return MTX_ERR_MPI_COLLECTIVE;
        err = mtxpartition_copy(&distmatrix->colpart, colpart);
        if (mtxdisterror_allreduce(disterr, err)) {
            mtxpartition_free(&distmatrix->rowpart);
            return MTX_ERR_MPI_COLLECTIVE;
        }
    } else {
        /* partition rows into blocks by default */
        int64_t * num_rows_per_rank = malloc(comm_size * sizeof(int64_t));
        err = !num_rows_per_rank ? MTX_ERR_ERRNO : MTX_SUCCESS;
        if (mtxdisterror_allreduce(disterr, err))
            return MTX_ERR_MPI_COLLECTIVE;
        num_rows_per_rank[rank] = num_local_rows;
        disterr->mpierrcode = MPI_Allgather(
            MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
            num_rows_per_rank, 1, MPI_INT64_T, comm);
        err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
        if (mtxdisterror_allreduce(disterr, err)) {
            free(num_rows_per_rank);
            return MTX_ERR_MPI_COLLECTIVE;
        }
        int64_t num_rows = 0;
        for (int r = 0; r < comm_size; r++)
            num_rows += num_rows_per_rank[r];
        err = mtxpartition_init_block(
            &distmatrix->rowpart, num_rows, comm_size, num_rows_per_rank);
        if (mtxdisterror_allreduce(disterr, err)) {
            free(num_rows_per_rank);
            return MTX_ERR_MPI_COLLECTIVE;
        }
        free(num_rows_per_rank);
        err = mtxpartition_init_singleton(
            &distmatrix->colpart, num_local_columns);
        if (mtxdisterror_allreduce(disterr, err)) {
            mtxpartition_free(&distmatrix->rowpart);
            return MTX_ERR_MPI_COLLECTIVE;
        }
    }
    return MTX_SUCCESS;
}

static int mtxdistmatrix_init_size(
    struct mtxdistmatrix * A,
    int64_t num_rows,
    int64_t num_columns,
    int64_t size,
    struct mtxdisterror * disterr)
{
    /* check that dimensions are the same on all processes */
    int64_t pdims[4] = {-num_rows, num_rows, -num_columns, num_columns};
    disterr->mpierrcode = MPI_Allreduce(
        MPI_IN_PLACE, pdims, 4, MPI_INT64_T, MPI_MIN, A->comm);
    int err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    if (pdims[0] != -pdims[1]) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (pdims[2] != -pdims[3]) return MTX_ERR_INCOMPATIBLE_SIZE;
    A->num_rows = num_rows;
    A->num_columns = num_columns;

    /* sum the number of explicitly stored nonzeros across all processes */
    A->size = size;
    disterr->mpierrcode = MPI_Allreduce(
        MPI_IN_PLACE, &A->size, 1, MPI_INT64_T, MPI_SUM, A->comm);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;

    /* int64_t M = A->num_rows; */
    /* int comm_size = A->comm_size; */
    /* int rank = A->rank; */
    /* A->rowblksize = M/comm_size + (rank < (M % comm_size) ? 1 : 0); */
    /* A->rowblkstart = rank*(M/comm_size) */
    /*     + (rank < (M % comm_size) ? rank : (M % comm_size)); */
    /* A->rowranks = malloc(A->rowblksize * sizeof(int)); */
    /* err = !A->rowranks ? MTX_ERR_ERRNO : MTX_SUCCESS; */
    /* if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE; */
    /* for (int64_t i = 0; i < A->rowblksize; i++) A->rowranks[i] = -1; */

    /* int64_t N = A->num_columns; */
    /* int comm_size = A->comm_size; */
    /* int rank = A->rank; */
    /* A->colblksize = N/comm_size + (rank < (N % comm_size) ? 1 : 0); */
    /* A->colblkstart = rank*(N/comm_size) */
    /*     + (rank < (N % comm_size) ? rank : (N % comm_size)); */
    /* A->colranks = malloc(A->colblksize * sizeof(int)); */
    /* err = !A->colranks ? MTX_ERR_ERRNO : MTX_SUCCESS; */
    /* if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE; */
    /* for (int64_t i = 0; i < A->colblksize; i++) A->colranks[i] = -1; */
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistmatrix_alloc_copy()’ allocates storage for a copy of a
 * distributed matrix without initialising the underlying values.
 */
int mtxdistmatrix_alloc_copy(
    struct mtxdistmatrix * dst,
    const struct mtxdistmatrix * src,
    struct mtxdisterror * disterr)
{
    int err = mtxdistmatrix_init_comm(dst, src->comm, disterr);
    if (err) return err;
    err = mtxpartition_copy(&dst->rowpart, &src->rowpart);
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;
    err = mtxpartition_copy(&dst->colpart, &src->colpart);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxpartition_free(&dst->rowpart);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    err = mtxmatrix_alloc_copy(&dst->interior, &src->interior);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxpartition_free(&dst->colpart);
        mtxpartition_free(&dst->rowpart);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistmatrix_init_copy()’ creates a copy of a distributed matrix.
 */
int mtxdistmatrix_init_copy(
    struct mtxdistmatrix * dst,
    const struct mtxdistmatrix * src,
    struct mtxdisterror * disterr)
{
    int err = mtxdistmatrix_init_comm(dst, src->comm, disterr);
    if (err) return err;
    err = mtxpartition_copy(&dst->rowpart, &src->rowpart);
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;
    err = mtxpartition_copy(&dst->colpart, &src->colpart);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxpartition_free(&dst->rowpart);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    err = mtxmatrix_init_copy(&dst->interior, &src->interior);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxpartition_free(&dst->colpart);
        mtxpartition_free(&dst->rowpart);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    return MTX_SUCCESS;
}

/*
 * Distributed matrices in array format
 */

/**
 * ‘mtxdistmatrix_alloc_array()’ allocates a distributed matrix in
 * array format.
 */
int mtxdistmatrix_alloc_array(
    struct mtxdistmatrix * distmatrix,
    enum mtxfield field,
    enum mtxprecision precision,
    enum mtxsymmetry symmetry,
    int num_local_rows,
    int num_local_columns,
    const struct mtxpartition * rowpart,
    const struct mtxpartition * colpart,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxdistmatrix_init_comm(distmatrix, comm, disterr);
    if (err) return err;
    err = mtxdistmatrix_init_partitions(
        distmatrix, num_local_rows, num_local_columns, rowpart, colpart, disterr);
    if (err) return err;
    err = mtxmatrix_alloc_array(
        &distmatrix->interior, field, precision, symmetry,
        num_local_rows, num_local_columns);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxpartition_free(&distmatrix->colpart);
        mtxpartition_free(&distmatrix->rowpart);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    distmatrix->rowmap = NULL;
    distmatrix->colmap = NULL;
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistmatrix_init_array_real_single()’ allocates and initialises
 * a distributed matrix in array format with real, single precision
 * coefficients.
 */
int mtxdistmatrix_init_array_real_single(
    struct mtxdistmatrix * distmatrix,
    enum mtxsymmetry symmetry,
    int num_local_rows,
    int num_local_columns,
    const float * data,
    const struct mtxpartition * rowpart,
    const struct mtxpartition * colpart,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxdistmatrix_init_comm(distmatrix, comm, disterr);
    if (err) return err;
    err = mtxdistmatrix_init_partitions(
        distmatrix, num_local_rows, num_local_columns, rowpart, colpart, disterr);
    if (err) return err;
    err = mtxmatrix_init_array_real_single(
        &distmatrix->interior, symmetry, num_local_rows, num_local_columns, data);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxpartition_free(&distmatrix->colpart);
        mtxpartition_free(&distmatrix->rowpart);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    distmatrix->num_rows = rowpart ? rowpart->size : 0;
    distmatrix->num_columns = colpart ? colpart->size : 0;
    distmatrix->rowmapsize = 0;
    distmatrix->rowmap = NULL;
    distmatrix->colmapsize = 0;
    distmatrix->colmap = NULL;
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistmatrix_init_array_real_double()’ allocates and initialises
 * a distributed matrix in array format with real, double precision
 * coefficients.
 */
int mtxdistmatrix_init_array_real_double(
    struct mtxdistmatrix * distmatrix,
    enum mtxsymmetry symmetry,
    int num_local_rows,
    int num_local_columns,
    const double * data,
    const struct mtxpartition * rowpart,
    const struct mtxpartition * colpart,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxdistmatrix_init_comm(distmatrix, comm, disterr);
    if (err) return err;
    err = mtxdistmatrix_init_partitions(
        distmatrix, num_local_rows, num_local_columns, rowpart, colpart, disterr);
    if (err) return err;
    err = mtxmatrix_init_array_real_double(
        &distmatrix->interior, symmetry, num_local_rows, num_local_columns, data);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxpartition_free(&distmatrix->colpart);
        mtxpartition_free(&distmatrix->rowpart);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    distmatrix->rowmap = NULL;
    distmatrix->colmap = NULL;
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistmatrix_init_array_complex_single()’ allocates and
 * initialises a distributed matrix in array format with complex,
 * single precision coefficients.
 */
int mtxdistmatrix_init_array_complex_single(
    struct mtxdistmatrix * distmatrix,
    enum mtxsymmetry symmetry,
    int num_local_rows,
    int num_local_columns,
    const float (* data)[2],
    const struct mtxpartition * rowpart,
    const struct mtxpartition * colpart,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxdistmatrix_init_comm(distmatrix, comm, disterr);
    if (err) return err;
    err = mtxdistmatrix_init_partitions(
        distmatrix, num_local_rows, num_local_columns, rowpart, colpart, disterr);
    if (err) return err;
    err = mtxmatrix_init_array_complex_single(
        &distmatrix->interior, symmetry, num_local_rows, num_local_columns, data);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxpartition_free(&distmatrix->colpart);
        mtxpartition_free(&distmatrix->rowpart);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    distmatrix->rowmap = NULL;
    distmatrix->colmap = NULL;
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistmatrix_init_array_complex_double()’ allocates and
 * initialises a distributed matrix in array format with complex,
 * double precision coefficients.
 */
int mtxdistmatrix_init_array_complex_double(
    struct mtxdistmatrix * distmatrix,
    enum mtxsymmetry symmetry,
    int num_local_rows,
    int num_local_columns,
    const double (* data)[2],
    const struct mtxpartition * rowpart,
    const struct mtxpartition * colpart,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxdistmatrix_init_comm(distmatrix, comm, disterr);
    if (err) return err;
    err = mtxdistmatrix_init_partitions(
        distmatrix, num_local_rows, num_local_columns, rowpart, colpart, disterr);
    if (err) return err;
    err = mtxmatrix_init_array_complex_double(
        &distmatrix->interior, symmetry, num_local_rows, num_local_columns, data);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxpartition_free(&distmatrix->colpart);
        mtxpartition_free(&distmatrix->rowpart);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    distmatrix->rowmap = NULL;
    distmatrix->colmap = NULL;
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistmatrix_init_array_integer_single()’ allocates and
 * initialises a distributed matrix in array format with integer,
 * single precision coefficients.
 */
int mtxdistmatrix_init_array_integer_single(
    struct mtxdistmatrix * distmatrix,
    enum mtxsymmetry symmetry,
    int num_local_rows,
    int num_local_columns,
    const int32_t * data,
    const struct mtxpartition * rowpart,
    const struct mtxpartition * colpart,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxdistmatrix_init_comm(distmatrix, comm, disterr);
    if (err) return err;
    err = mtxdistmatrix_init_partitions(
        distmatrix, num_local_rows, num_local_columns, rowpart, colpart, disterr);
    if (err) return err;
    err = mtxmatrix_init_array_integer_single(
        &distmatrix->interior, symmetry, num_local_rows, num_local_columns, data);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxpartition_free(&distmatrix->colpart);
        mtxpartition_free(&distmatrix->rowpart);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    distmatrix->rowmap = NULL;
    distmatrix->colmap = NULL;
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistmatrix_init_array_integer_double()’ allocates and
 * initialises a distributed matrix in array format with integer,
 * double precision coefficients.
 */
int mtxdistmatrix_init_array_integer_double(
    struct mtxdistmatrix * distmatrix,
    enum mtxsymmetry symmetry,
    int num_local_rows,
    int num_local_columns,
    const int64_t * data,
    const struct mtxpartition * rowpart,
    const struct mtxpartition * colpart,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxdistmatrix_init_comm(distmatrix, comm, disterr);
    if (err) return err;
    err = mtxdistmatrix_init_partitions(
        distmatrix, num_local_rows, num_local_columns, rowpart, colpart, disterr);
    if (err) return err;
    err = mtxmatrix_init_array_integer_double(
        &distmatrix->interior, symmetry, num_local_rows, num_local_columns, data);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxpartition_free(&distmatrix->colpart);
        mtxpartition_free(&distmatrix->rowpart);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    distmatrix->rowmap = NULL;
    distmatrix->colmap = NULL;
    return MTX_SUCCESS;
}

/*
 * distributed matrices in coordinate format from global row and
 * column indices.
 */

static int mtxdistmatrix_init_coordinate_global(
    struct mtxdistmatrix * A,
    enum mtxfield field,
    enum mtxprecision precision,
    enum mtxsymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    int64_t num_nonzeros,
    const int64_t * rowidx,
    const int64_t * colidx,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxdistmatrix_init_comm(A, comm, disterr);
    if (err) return err;
    err = mtxdistmatrix_init_size(A, num_rows, num_columns, num_nonzeros, disterr);
    if (err) return err;

    /* compute the mapping from local to global matrix rows */
    int64_t * localrowidx = malloc(num_nonzeros * sizeof(int64_t));
    err = !localrowidx ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    int64_t * rowperm = malloc(num_nonzeros * sizeof(int64_t));
    err = !rowperm ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(localrowidx);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    int64_t * rowdstidx = malloc(num_nonzeros * sizeof(int64_t));
    err = !rowdstidx ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(rowperm); free(localrowidx);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    for (int64_t k = 0; k < num_nonzeros; k++) localrowidx[k] = rowidx[k];
    err = compact_unsorted_int64(
        &A->rowmapsize, NULL, num_nonzeros, localrowidx, rowperm, rowdstidx);
    if (mtxdisterror_allreduce(disterr, err)) {
        free(rowdstidx); free(rowperm); free(localrowidx);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    A->rowmap = malloc(A->rowmapsize * sizeof(int64_t));
    err = !A->rowmap ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(rowdstidx); free(rowperm); free(localrowidx);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    err = compact_sorted_int64(
        &A->rowmapsize, A->rowmap, num_nonzeros, localrowidx, NULL);
    if (mtxdisterror_allreduce(disterr, err)) {
        free(A->rowmap); free(rowdstidx); free(rowperm); free(localrowidx);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    /* compute the mapping from local to global matrix columns */
    int64_t * localcolidx = malloc(num_nonzeros * sizeof(int64_t));
    err = !localcolidx ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(A->rowmap); free(rowdstidx); free(rowperm); free(localrowidx);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    int64_t * colperm = malloc(num_nonzeros * sizeof(int64_t));
    err = !colperm ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(localcolidx);
        free(A->rowmap); free(rowdstidx); free(rowperm); free(localrowidx);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    int64_t * coldstidx = malloc(num_nonzeros * sizeof(int64_t));
    err = !coldstidx ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(colperm); free(localcolidx);
        free(A->rowmap); free(rowdstidx); free(rowperm); free(localrowidx);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    for (int64_t k = 0; k < num_nonzeros; k++) localcolidx[k] = colidx[k];
    err = compact_unsorted_int64(
        &A->colmapsize, NULL, num_nonzeros, localcolidx, colperm, coldstidx);
    if (mtxdisterror_allreduce(disterr, err)) {
        free(coldstidx); free(colperm); free(localcolidx);
        free(A->rowmap); free(rowdstidx); free(rowperm); free(localrowidx);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    A->colmap = malloc(A->colmapsize * sizeof(int64_t));
    err = !A->colmap ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(coldstidx); free(colperm); free(localcolidx);
        free(A->rowmap); free(rowdstidx); free(rowperm); free(localrowidx);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    err = compact_sorted_int64(
        &A->colmapsize, A->colmap, num_nonzeros, localcolidx, NULL);
    if (mtxdisterror_allreduce(disterr, err)) {
        free(A->colmap); free(coldstidx); free(colperm); free(localcolidx);
        free(A->rowmap); free(rowdstidx); free(rowperm); free(localrowidx);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    /* allocate storage for the local matrix */
    err = mtxmatrix_alloc_coordinate(
        &A->interior, field, precision, symmetry,
        A->rowmapsize, A->colmapsize, num_nonzeros);
    if (mtxdisterror_allreduce(disterr, err)) {
        free(A->colmap); free(coldstidx); free(colperm); free(localcolidx);
        free(A->rowmap); free(rowdstidx); free(rowperm); free(localrowidx);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    /* set the row and column indices of the local matrix */
    A->interior.storage.coordinate.num_nonzeros = 0;
    for (int64_t k = 0; k < num_nonzeros; k++) {
        A->interior.storage.coordinate.rowidx[k] = rowdstidx[rowperm[k]];
        A->interior.storage.coordinate.colidx[k] = coldstidx[colperm[k]];
        A->interior.storage.coordinate.num_nonzeros +=
            (symmetry == mtx_unsymmetric ||
             A->interior.storage.coordinate.rowidx[k] ==
             A->interior.storage.coordinate.colidx[k]) ? 1 : 2;
    }
    free(coldstidx); free(colperm); free(localcolidx);
    free(rowdstidx); free(rowperm); free(localrowidx);

    /* sum the number of nonzeros across all processes */
    A->num_nonzeros = A->interior.storage.coordinate.num_nonzeros;
    disterr->mpierrcode = MPI_Allreduce(
        MPI_IN_PLACE, &A->num_nonzeros, 1, MPI_INT64_T, MPI_SUM, comm);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(A->colmap); free(A->rowmap);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    /* TODO: remove these lines when rowpart and colpart are no
     * longer needed */
    mtxpartition_init_singleton(&A->rowpart, num_rows);
    mtxpartition_init_singleton(&A->colpart, num_columns);
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistmatrix_init_global_coordinate_real_single()’ allocates and
 * initialises a distributed matrix in coordinate format with real,
 * single precision coefficients.
 */
int mtxdistmatrix_init_global_coordinate_real_single(
    struct mtxdistmatrix * A,
    enum mtxsymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    int64_t num_nonzeros,
    const int64_t * rowidx,
    const int64_t * colidx,
    const float * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxdistmatrix_init_coordinate_global(
        A, mtx_field_real, mtx_single, symmetry,
        num_rows, num_columns, num_nonzeros,
        rowidx, colidx, comm, disterr);
    if (err) return err;
    err = mtxvector_base_init_real_single(
        &A->interior.storage.coordinate.a, num_nonzeros, data);
    if (err) {
        mtxmatrix_free(&A->interior); free(A->colmap); free(A->rowmap);
        return err;
    }
    return MTX_SUCCESS;
}

/*
 * distributed matrices in coordinate format from local row and column
 * indices.
 */

static int mtxdistmatrix_init_local_coordinate(
    struct mtxdistmatrix * A,
    int64_t num_rows,
    int64_t num_columns,
    int64_t num_nonzeros,
    int64_t rowmapsize,
    const int64_t * rowmap,
    int64_t colmapsize,
    const int64_t * colmap,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxdistmatrix_init_comm(A, comm, disterr);
    if (err) return err;
    err = mtxdistmatrix_init_size(A, num_rows, num_columns, num_nonzeros, disterr);
    if (err) return err;
    A->rowmapsize = rowmapsize;
    A->rowmap = malloc(A->rowmapsize * sizeof(int64_t));
    err = !A->rowmap ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    for (int64_t i = 0; i < rowmapsize; i++) A->rowmap[i] = rowmap[i];
    A->colmapsize = colmapsize;
    A->colmap = malloc(A->colmapsize * sizeof(int64_t));
    err = !A->colmap ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(A->rowmap);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    for (int64_t j = 0; j < colmapsize; j++) A->colmap[j] = colmap[j];

    /* TODO: remove these lines when rowpart and colpart are no
     * longer needed */
    mtxpartition_init_singleton(&A->rowpart, num_rows);
    mtxpartition_init_singleton(&A->colpart, num_columns);
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistmatrix_init_local_coordinate_real_single()’ allocates and
 * initialises a distributed matrix in coordinate format with real,
 * single precision coefficients.
 */
int mtxdistmatrix_init_local_coordinate_real_single(
    struct mtxdistmatrix * A,
    enum mtxsymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    int64_t num_nonzeros,
    const int * rowidx,
    const int * colidx,
    const float * data,
    int64_t rowmapsize,
    const int64_t * rowmap,
    int64_t colmapsize,
    const int64_t * colmap,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxdistmatrix_init_local_coordinate(
        A, num_rows, num_columns, num_nonzeros,
        rowmapsize, rowmap, colmapsize, colmap, comm, disterr);
    if (err) return err;
    err = mtxmatrix_init_coordinate_real_single(
        &A->interior, symmetry, rowmapsize, colmapsize,
        num_nonzeros, rowidx, colidx, data);
    if (err) { free(A->colmap); free(A->rowmap); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistmatrix_init_local_coordinate_real_double()’ allocates and
 * initialises a distributed matrix in coordinate format with real,
 * double precision coefficients.
 */
int mtxdistmatrix_init_local_coordinate_real_double(
    struct mtxdistmatrix * A,
    enum mtxsymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    int64_t num_nonzeros,
    const int * rowidx,
    const int * colidx,
    const double * data,
    int64_t rowmapsize,
    const int64_t * rowmap,
    int64_t colmapsize,
    const int64_t * colmap,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxdistmatrix_init_local_coordinate(
        A, num_rows, num_columns, num_nonzeros,
        rowmapsize, rowmap, colmapsize, colmap, comm, disterr);
    if (err) return err;
    err = mtxmatrix_init_coordinate_real_double(
        &A->interior, symmetry, rowmapsize, colmapsize,
        num_nonzeros, rowidx, colidx, data);
    if (err) { free(A->colmap); free(A->rowmap); return err; }
    return MTX_SUCCESS;
}

/*
 * distributed matrices in coordinate format
 */

/**
 * ‘mtxdistmatrix_alloc_coordinate()’ allocates a distributed matrix
 * in coordinate format.
 */
int mtxdistmatrix_alloc_coordinate(
    struct mtxdistmatrix * distmatrix,
    enum mtxfield field,
    enum mtxprecision precision,
    enum mtxsymmetry symmetry,
    int num_local_rows,
    int num_local_columns,
    int64_t num_local_nonzeros,
    const struct mtxpartition * rowpart,
    const struct mtxpartition * colpart,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxdistmatrix_init_comm(distmatrix, comm, disterr);
    if (err) return err;
    err = mtxdistmatrix_init_partitions(
        distmatrix, num_local_rows, num_local_columns, rowpart, colpart, disterr);
    if (err) return err;
    err = mtxmatrix_alloc_coordinate(
        &distmatrix->interior, field, precision, symmetry,
        num_local_rows, num_local_columns, num_local_nonzeros);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxpartition_free(&distmatrix->colpart);
        mtxpartition_free(&distmatrix->rowpart);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    distmatrix->rowmap = NULL;
    distmatrix->colmap = NULL;
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistmatrix_init_coordinate_real_single()’ allocates and
 * initialises a distributed matrix in coordinate format with real,
 * single precision coefficients.
 */
int mtxdistmatrix_init_coordinate_real_single(
    struct mtxdistmatrix * distmatrix,
    enum mtxsymmetry symmetry,
    int num_local_rows,
    int num_local_columns,
    int64_t num_local_nonzeros,
    const int * rowidx,
    const int * colidx,
    const float * data,
    const struct mtxpartition * rowpart,
    const struct mtxpartition * colpart,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxdistmatrix_init_comm(distmatrix, comm, disterr);
    if (err) return err;
    err = mtxdistmatrix_init_partitions(
        distmatrix, num_local_rows, num_local_columns, rowpart, colpart, disterr);
    if (err) return err;
    err = mtxmatrix_init_coordinate_real_single(
        &distmatrix->interior, symmetry, num_local_rows, num_local_columns,
        num_local_nonzeros, rowidx, colidx, data);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxpartition_free(&distmatrix->colpart);
        mtxpartition_free(&distmatrix->rowpart);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    distmatrix->rowmap = NULL;
    distmatrix->colmap = NULL;
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistmatrix_init_coordinate_real_double()’ allocates and
 * initialises a distributed matrix in coordinate format with real,
 * double precision coefficients.
 */
int mtxdistmatrix_init_coordinate_real_double(
    struct mtxdistmatrix * distmatrix,
    enum mtxsymmetry symmetry,
    int num_local_rows,
    int num_local_columns,
    int64_t num_local_nonzeros,
    const int * rowidx,
    const int * colidx,
    const double * data,
    const struct mtxpartition * rowpart,
    const struct mtxpartition * colpart,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxdistmatrix_init_comm(distmatrix, comm, disterr);
    if (err) return err;
    err = mtxdistmatrix_init_partitions(
        distmatrix, num_local_rows, num_local_columns, rowpart, colpart, disterr);
    if (err) return err;
    err = mtxmatrix_init_coordinate_real_double(
        &distmatrix->interior, symmetry, num_local_rows, num_local_columns,
        num_local_nonzeros, rowidx, colidx, data);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxpartition_free(&distmatrix->colpart);
        mtxpartition_free(&distmatrix->rowpart);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    distmatrix->rowmap = NULL;
    distmatrix->colmap = NULL;
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistmatrix_init_coordinate_complex_single()’ allocates and
 * initialises a distributed matrix in coordinate format with complex,
 * single precision coefficients.
 */
int mtxdistmatrix_init_coordinate_complex_single(
    struct mtxdistmatrix * distmatrix,
    enum mtxsymmetry symmetry,
    int num_local_rows,
    int num_local_columns,
    int64_t num_local_nonzeros,
    const int * rowidx,
    const int * colidx,
    const float (* data)[2],
    const struct mtxpartition * rowpart,
    const struct mtxpartition * colpart,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxdistmatrix_init_comm(distmatrix, comm, disterr);
    if (err) return err;
    err = mtxdistmatrix_init_partitions(
        distmatrix, num_local_rows, num_local_columns, rowpart, colpart, disterr);
    if (err) return err;
    err = mtxmatrix_init_coordinate_complex_single(
        &distmatrix->interior, symmetry, num_local_rows, num_local_columns,
        num_local_nonzeros, rowidx, colidx, data);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxpartition_free(&distmatrix->colpart);
        mtxpartition_free(&distmatrix->rowpart);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    distmatrix->rowmap = NULL;
    distmatrix->colmap = NULL;
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistmatrix_init_coordinate_complex_double()’ allocates and
 * initialises a distributed matrix in coordinate format with complex,
 * double precision coefficients.
 */
int mtxdistmatrix_init_coordinate_complex_double(
    struct mtxdistmatrix * distmatrix,
    enum mtxsymmetry symmetry,
    int num_local_rows,
    int num_local_columns,
    int64_t num_local_nonzeros,
    const int * rowidx,
    const int * colidx,
    const double (* data)[2],
    const struct mtxpartition * rowpart,
    const struct mtxpartition * colpart,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxdistmatrix_init_comm(distmatrix, comm, disterr);
    if (err) return err;
    err = mtxdistmatrix_init_partitions(
        distmatrix, num_local_rows, num_local_columns, rowpart, colpart, disterr);
    if (err) return err;
    err = mtxmatrix_init_coordinate_complex_double(
        &distmatrix->interior, symmetry, num_local_rows, num_local_columns,
        num_local_nonzeros, rowidx, colidx, data);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxpartition_free(&distmatrix->colpart);
        mtxpartition_free(&distmatrix->rowpart);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    distmatrix->rowmap = NULL;
    distmatrix->colmap = NULL;
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistmatrix_init_coordinate_integer_single()’ allocates and
 * initialises a distributed matrix in coordinate format with integer,
 * single precision coefficients.
 */
int mtxdistmatrix_init_coordinate_integer_single(
    struct mtxdistmatrix * distmatrix,
    enum mtxsymmetry symmetry,
    int num_local_rows,
    int num_local_columns,
    int64_t num_local_nonzeros,
    const int * rowidx,
    const int * colidx,
    const int32_t * data,
    const struct mtxpartition * rowpart,
    const struct mtxpartition * colpart,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxdistmatrix_init_comm(distmatrix, comm, disterr);
    if (err) return err;
    err = mtxdistmatrix_init_partitions(
        distmatrix, num_local_rows, num_local_columns, rowpart, colpart, disterr);
    if (err) return err;
    err = mtxmatrix_init_coordinate_integer_single(
        &distmatrix->interior, symmetry, num_local_rows, num_local_columns,
        num_local_nonzeros, rowidx, colidx, data);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxpartition_free(&distmatrix->colpart);
        mtxpartition_free(&distmatrix->rowpart);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    distmatrix->rowmap = NULL;
    distmatrix->colmap = NULL;
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistmatrix_init_coordinate_integer_double()’ allocates and
 * initialises a distributed matrix in coordinate format with integer,
 * double precision coefficients.
 */
int mtxdistmatrix_init_coordinate_integer_double(
    struct mtxdistmatrix * distmatrix,
    enum mtxsymmetry symmetry,
    int num_local_rows,
    int num_local_columns,
    int64_t num_local_nonzeros,
    const int * rowidx,
    const int * colidx,
    const int64_t * data,
    const struct mtxpartition * rowpart,
    const struct mtxpartition * colpart,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxdistmatrix_init_comm(distmatrix, comm, disterr);
    if (err) return err;
    err = mtxdistmatrix_init_partitions(
        distmatrix, num_local_rows, num_local_columns, rowpart, colpart, disterr);
    if (err) return err;
    err = mtxmatrix_init_coordinate_integer_double(
        &distmatrix->interior, symmetry, num_local_rows, num_local_columns,
        num_local_nonzeros, rowidx, colidx, data);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxpartition_free(&distmatrix->colpart);
        mtxpartition_free(&distmatrix->rowpart);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    distmatrix->rowmap = NULL;
    distmatrix->colmap = NULL;
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistmatrix_init_coordinate_pattern()’ allocates and initialises
 * a distributed matrix in coordinate format with boolean
 * coefficients.
 */
int mtxdistmatrix_init_coordinate_pattern(
    struct mtxdistmatrix * distmatrix,
    enum mtxsymmetry symmetry,
    int num_local_rows,
    int num_local_columns,
    int64_t num_local_nonzeros,
    const int * rowidx,
    const int * colidx,
    const struct mtxpartition * rowpart,
    const struct mtxpartition * colpart,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxdistmatrix_init_comm(distmatrix, comm, disterr);
    if (err) return err;
    err = mtxdistmatrix_init_partitions(
        distmatrix, num_local_rows, num_local_columns, rowpart, colpart, disterr);
    if (err) return err;
    err = mtxmatrix_init_coordinate_pattern(
        &distmatrix->interior, symmetry, num_local_rows, num_local_columns,
        num_local_nonzeros, rowidx, colidx);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxpartition_free(&distmatrix->colpart);
        mtxpartition_free(&distmatrix->rowpart);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    distmatrix->rowmap = NULL;
    distmatrix->colmap = NULL;
    return MTX_SUCCESS;
}

/*
 * Row and column vectors
 */

/**
 * ‘mtxdistmatrix_alloc_row_vector()’ allocates a distributed row
 * vector for a given distributed matrix. A row vector is a vector
 * whose length equal to a single row of the matrix, and it is
 * distributed among processes in a given process row according to the
 * column partitioning of the distributed matrix.
 */
int mtxdistmatrix_alloc_row_vector(
    const struct mtxdistmatrix * distmatrix,
    struct mtxdistvector * distvector,
    enum mtxvectortype vector_type,
    struct mtxdisterror * disterr)
{
    distvector->comm = distmatrix->comm;
    distvector->comm_size = distmatrix->comm_size;
    distvector->rank = distmatrix->rank;
    int err = mtxpartition_copy(&distvector->rowpart, &distmatrix->colpart);
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;
    err = mtxmatrix_alloc_row_vector(
        &distmatrix->interior, &distvector->interior, vector_type);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxpartition_free(&distvector->rowpart);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistmatrix_alloc_column_vector()’ allocates a distributed
 * column vector for a given distributed matrix. A column vector is a
 * vector whose length equal to a single column of the matrix, and it
 * is distributed among processes in a given process column according
 * to the row partitioning of the distributed matrix.
 */
int mtxdistmatrix_alloc_column_vector(
    const struct mtxdistmatrix * distmatrix,
    struct mtxdistvector * distvector,
    enum mtxvectortype vector_type,
    struct mtxdisterror * disterr)
{
    distvector->comm = distmatrix->comm;
    distvector->comm_size = distmatrix->comm_size;
    distvector->rank = distmatrix->rank;
    int err = mtxpartition_copy(&distvector->rowpart, &distmatrix->rowpart);
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;
    err = mtxmatrix_alloc_column_vector(
        &distmatrix->interior, &distvector->interior, vector_type);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxpartition_free(&distvector->rowpart);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    return MTX_SUCCESS;
}

/*
 * Convert to and from Matrix Market format
 */

/**
 * ‘mtxdistmatrix_from_mtxfile()’ converts a matrix in Matrix Market
 * format to a distributed matrix.
 *
 * The ‘type’ argument may be used to specify a desired storage format
 * or implementation for the underlying ‘mtxmatrix’ on each
 * process. If ‘type’ is ‘mtxmatrix_auto’, then the type of
 * ‘mtxmatrix’ is chosen to match the type of ‘mtxfile’. That is,
 * ‘mtxmatrix_array’ is used if ‘mtxfile’ is in array format, and
 * ‘mtxmatrix_coordinate’ is used if ‘mtxfile’ is in coordinate
 * format.
 *
 * Furthermore, ‘rowpart’ and ‘colpart’ must be partitionings of the
 * rows and columns of the global matrix. Therefore, ‘rowpart->size’
 * must be equal to the number of rows and ‘colpart->size’ must be
 * equal to the number of columns in ‘mtxfile’. There must be at least
 * one MPI process in the communicator ‘comm’ for each part in the
 * partitioned matrix (i.e., the number of row parts times the number
 * of column parts).
 *
 * If ‘rowpart’ and ‘colpart’ are both ‘NULL’, then the rows are
 * partitioned into contiguous blocks of equal size by default.
 */
int mtxdistmatrix_from_mtxfile(
    struct mtxdistmatrix * dst,
    const struct mtxfile * src,
    enum mtxmatrixtype matrix_type,
    const struct mtxpartition * rowpart,
    const struct mtxpartition * colpart,
    MPI_Comm comm,
    int root,
    struct mtxdisterror * disterr)
{
    int err = mtxdistmatrix_init_comm(dst, comm, disterr);
    if (err) return err;
    int comm_size = dst->comm_size;
    int rank = dst->rank;

    if (rank == root && src->header.object != mtxfile_matrix)
        err = MTX_ERR_INCOMPATIBLE_MTX_OBJECT;
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;

    int num_local_rows;
    int num_local_columns;
    if (rowpart && colpart) {
        int p = rank / colpart->num_parts;
        int q = rank % colpart->num_parts;
        num_local_rows = p < rowpart->num_parts ? rowpart->part_sizes[p] : 0;
        num_local_columns = q < colpart->num_parts ? colpart->part_sizes[q] : 0;
    } else if (rowpart) {
        /* broadcast the number of columns in the Matrix Market file */
        int num_columns = (rank == root) ? src->size.num_columns : 0;
        disterr->mpierrcode = MPI_Bcast(&num_columns, 1, MPI_INT, root, comm);
        if (disterr->mpierrcode) err = MTX_ERR_MPI;
        if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
        num_local_rows = rank < rowpart->num_parts ? rowpart->part_sizes[rank] : 0;
        num_local_columns = num_columns;
    } else if (colpart) {
        /* broadcast the number of rows in the Matrix Market file */
        int num_rows = (rank == root) ? src->size.num_rows : 0;
        disterr->mpierrcode = MPI_Bcast(&num_rows, 1, MPI_INT, root, comm);
        if (disterr->mpierrcode) err = MTX_ERR_MPI;
        if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
        num_local_rows = num_rows;
        num_local_columns = rank < colpart->num_parts ? colpart->part_sizes[rank] : 0;
    } else {
        /* broadcast the number of rows and columns */
        int num_rows = (rank == root) ? src->size.num_rows : 0;
        disterr->mpierrcode = MPI_Bcast(&num_rows, 1, MPI_INT, root, comm);
        if (disterr->mpierrcode) err = MTX_ERR_MPI;
        if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
        int num_columns = (rank == root) ? src->size.num_columns : 0;
        disterr->mpierrcode = MPI_Bcast(&num_columns, 1, MPI_INT, root, comm);
        if (disterr->mpierrcode) err = MTX_ERR_MPI;
        if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;

        /* divide rows into equal-sized blocks */
        num_local_rows = num_rows / comm_size
            + (rank < (num_rows % comm_size) ? 1 : 0);
        num_local_columns = num_columns;
    }

    err = mtxdistmatrix_init_partitions(
        dst, num_local_rows, num_local_columns, rowpart, colpart, disterr);
    if (err) return err;

    /* 1. Partition the matrix */
    int num_parts = dst->rowpart.num_parts * dst->colpart.num_parts;
    struct mtxfile * sendmtxfiles = (rank == root)
        ? malloc(num_parts * sizeof(struct mtxfile)) : NULL;
    err = (rank == root && !sendmtxfiles) ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxpartition_free(&dst->colpart);
        mtxpartition_free(&dst->rowpart);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    if (rank == root) {
        err = mtxfile_partition(
            sendmtxfiles, src, &dst->rowpart, &dst->colpart);
    }
    if (mtxdisterror_allreduce(disterr, err)) {
        if (rank == root)
            free(sendmtxfiles);
        mtxpartition_free(&dst->colpart);
        mtxpartition_free(&dst->rowpart);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    /* 2. Send each part to the owning process */
    struct mtxfile recvmtxfile;
    err = mtxfile_scatter(sendmtxfiles, &recvmtxfile, root, comm, disterr);
    if (err) {
        if (rank == root) {
            for (int p = 0; p < comm_size; p++)
                mtxfile_free(&sendmtxfiles[p]);
            free(sendmtxfiles);
        }
        mtxpartition_free(&dst->colpart);
        mtxpartition_free(&dst->rowpart);
        return err;
    }

    if (rank == root) {
        for (int p = 0; p < comm_size; p++)
            mtxfile_free(&sendmtxfiles[p]);
        free(sendmtxfiles);
    }

    /* 3. Let each process create its local part of the matrix */
    err = mtxmatrix_from_mtxfile(
        &dst->interior, matrix_type, &recvmtxfile);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxfile_free(&recvmtxfile);
        mtxpartition_free(&dst->colpart);
        mtxpartition_free(&dst->rowpart);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    mtxfile_free(&recvmtxfile);

    dst->rowmap = NULL;
    dst->colmap = NULL;
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistmatrix_to_mtxfile()’ gathers a distributed matrix onto a
 * single, root process and converts it to a (non-distributed) Matrix
 * Market file on that process.
 *
 * This function performs collective communication and therefore
 * requires every process in the communicator to perform matching
 * calls to this function.
 */
int mtxdistmatrix_to_mtxfile(
    struct mtxfile * mtxfile,
    const struct mtxdistmatrix * A,
    enum mtxfileformat mtxfmt,
    int root,
    struct mtxdisterror * disterr)
{
    int err;
    enum mtxfield field;
    err = mtxmatrix_field(&A->interior, &field);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    enum mtxprecision precision;
    err = mtxmatrix_precision(&A->interior, &precision);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    enum mtxsymmetry symmetry;
    err = mtxmatrix_symmetry(&A->interior, &symmetry);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    int64_t size;
    err = mtxmatrix_size(&A->interior, &size);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;

    struct mtxfileheader mtxheader;
    mtxheader.object = mtxfile_matrix;
    mtxheader.format = mtxfmt;
    if (field == mtx_field_real) mtxheader.field = mtxfile_real;
    else if (field == mtx_field_complex) mtxheader.field = mtxfile_complex;
    else if (field == mtx_field_integer) mtxheader.field = mtxfile_integer;
    else if (field == mtx_field_pattern) mtxheader.field = mtxfile_pattern;
    else { return MTX_ERR_INVALID_FIELD; }
    mtxheader.symmetry = symmetry;

    struct mtxfilesize mtxsize;
    mtxsize.num_rows = A->num_rows;
    mtxsize.num_columns = A->num_columns;
    if (mtxfmt == mtxfile_array) {
        mtxsize.num_nonzeros = -1;
    } else if (mtxfmt == mtxfile_coordinate) {
        mtxsize.num_nonzeros = A->size;
    } else { return MTX_ERR_INVALID_MTX_FORMAT; }

    err = mtxfile_alloc(mtxfile, &mtxheader, NULL, &mtxsize, precision);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;

    int64_t offset = 0;
    for (int p = 0; p < A->comm_size; p++) {
        if (A->rank == root && p != root) {
            /* receive from the root process */
            struct mtxfile recvmtxfile;
            err = err ? err : mtxfile_recv(&recvmtxfile, p, 0, A->comm, disterr);
            int64_t recvsize;
            if (mtxfile->header.format == mtxfile_array) {
                recvsize = recvmtxfile.size.num_rows *
                    recvmtxfile.size.num_columns;
            } else if (mtxfile->header.format == mtxfile_coordinate) {
                recvsize = recvmtxfile.size.num_nonzeros;
            } else { err = MTX_ERR_INVALID_MTX_FORMAT; }
            err = err ? err : mtxfiledata_copy(
                &mtxfile->data, &recvmtxfile.data,
                recvmtxfile.header.object, recvmtxfile.header.format,
                recvmtxfile.header.field, recvmtxfile.precision,
                recvsize, offset, 0);
            mtxfile_free(&recvmtxfile);
            offset += recvsize;
        } else if (A->rank != root && A->rank == p) {
            /* send to the root process */
            struct mtxfile sendmtxfile;
            err = mtxmatrix_to_mtxfile(
                &sendmtxfile, &A->interior,
                A->num_rows, A->rowmap, A->num_columns, A->colmap, mtxfmt);
            err = err ? err : mtxfile_send(
                &sendmtxfile, root, 0, A->comm, disterr);
            mtxfile_free(&sendmtxfile);
        } else if (A->rank == root && p == root) {
            struct mtxfile srcmtxfile;
            err = mtxmatrix_to_mtxfile(
                &srcmtxfile, &A->interior,
                A->num_rows, A->rowmap, A->num_columns, A->colmap, mtxfmt);
            err = err ? err : mtxfiledata_copy(
                &mtxfile->data, &srcmtxfile.data,
                srcmtxfile.header.object, srcmtxfile.header.format,
                srcmtxfile.header.field, srcmtxfile.precision,
                size, offset, 0);
            mtxfile_free(&srcmtxfile);
            offset += size;
        }
        if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistmatrix_from_mtxdistfile()’ converts a matrix in distributed
 * Matrix Market format to a distributed matrix.
 */
int mtxdistmatrix_from_mtxdistfile(
    struct mtxdistmatrix * dst,
    const struct mtxdistfile * src,
    enum mtxmatrixtype matrix_type,
    const struct mtxpartition * rowpart,
    const struct mtxpartition * colpart,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxdistmatrix_init_comm(dst, comm, disterr);
    if (err) return err;
    int comm_size = dst->comm_size;
    int rank = dst->rank;

    if (src->header.object != mtxfile_matrix)
        err = MTX_ERR_INCOMPATIBLE_MTX_OBJECT;
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;

    int num_local_rows;
    int num_local_columns;
    if (rowpart && colpart) {
        int p = rank / colpart->num_parts;
        int q = rank % colpart->num_parts;
        num_local_rows = p < rowpart->num_parts ? rowpart->part_sizes[p] : 0;
        num_local_columns = q < colpart->num_parts ? colpart->part_sizes[q] : 0;
    } else if (rowpart) {
        num_local_rows = rank < rowpart->num_parts ? rowpart->part_sizes[rank] : 0;
        num_local_columns = src->size.num_columns;
    } else if (colpart) {
        num_local_rows = src->size.num_rows;
        num_local_columns = rank < colpart->num_parts ? colpart->part_sizes[rank] : 0;
    } else {
        /* divide rows into equal-sized blocks */
        num_local_rows = src->size.num_rows / comm_size
            + (rank < (src->size.num_rows % comm_size) ? 1 : 0);
        num_local_columns = src->size.num_columns;
    }
    err = mtxdistmatrix_init_partitions(
        dst, num_local_rows, num_local_columns, rowpart, colpart, disterr);
    if (err) return err;

    /* 1. Partition the matrix */
    int num_parts = dst->rowpart.num_parts * dst->colpart.num_parts;
    struct mtxdistfile * dsts =
        malloc(num_parts * sizeof(struct mtxdistfile));
    err = !dsts ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxpartition_free(&dst->colpart);
        mtxpartition_free(&dst->rowpart);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    err = mtxdistfile_partition(
        dsts, src, &dst->rowpart, &dst->colpart, disterr);
    if (err) {
        free(dsts);
        mtxpartition_free(&dst->colpart);
        mtxpartition_free(&dst->rowpart);
        return err;
    }

    for (int p = 0; p < num_parts; p++) {
        struct mtxfile mtxfile;
        err = mtxdistfile_to_mtxfile(&mtxfile, &dsts[p], p, disterr);
        if (err) {
            for (int q = 0; q < num_parts; q++)
                mtxdistfile_free(&dsts[q]);
            free(dsts);
            mtxpartition_free(&dst->colpart);
            mtxpartition_free(&dst->rowpart);
            return err;
        }

        err = rank == p
            ? mtxmatrix_from_mtxfile(
                &dst->interior, matrix_type, &mtxfile)
            : MTX_SUCCESS;
        if (mtxdisterror_allreduce(disterr, err)) {
            if (rank == p) mtxfile_free(&mtxfile);
            for (int q = 0; q < num_parts; q++)
                mtxdistfile_free(&dsts[q]);
            free(dsts);
            mtxpartition_free(&dst->colpart);
            mtxpartition_free(&dst->rowpart);
            return MTX_ERR_MPI_COLLECTIVE;
        }
        if (rank == p) mtxfile_free(&mtxfile);
        mtxdistfile_free(&dsts[p]);
    }
    free(dsts);

    dst->rowmap = NULL;
    dst->colmap = NULL;
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistmatrix_to_mtxdistfile()’ converts a distributed matrix to a
 * matrix in a distributed Matrix Market format.
 */
int mtxdistmatrix_to_mtxdistfile(
    struct mtxdistfile * dst,
    const struct mtxdistmatrix * src,
    enum mtxfileformat mtxfmt,
    struct mtxdisterror * disterr)
{
    int err;
    MPI_Comm comm = src->comm;
    int comm_size = src->comm_size;
    int rank = src->rank;
    int num_parts = src->rowpart.num_parts * src->colpart.num_parts;

    /* 1. Each process converts its part of the matrix to Matrix
     * Market format */
    struct mtxfile mtxfile;
    err = (rank < num_parts)
        ? mtxmatrix_to_mtxfile(&mtxfile, &src->interior, 0, NULL, 0, NULL, mtxfmt)
        : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;

    /* 2. Set up partitions for the data of Matrix Market files on
     * each individual process */
    int64_t * part_sizes = malloc(num_parts * sizeof(int64_t));
    err = !part_sizes ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxfile_free(&mtxfile);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    for (int p = 0; p < num_parts; p++)
        part_sizes[p] = 0;
    int64_t num_data_lines;
    err = mtxfilesize_num_data_lines(
        &mtxfile.size, mtxfile.header.symmetry, &num_data_lines);
    if (mtxdisterror_allreduce(disterr, err)) {
        free(part_sizes);
        mtxfile_free(&mtxfile);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    struct mtxdistfile * srcs =
        malloc(num_parts * sizeof(struct mtxdistfile));
    err = !srcs ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(part_sizes);
        mtxfile_free(&mtxfile);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    /* 2. Distribute each Matrix Market file across processes */
    for (int p = 0; p < num_parts; p++) {
        if (rank == p) part_sizes[p] = num_data_lines;
        disterr->mpierrcode = MPI_Bcast(
            &part_sizes[p], 1, MPI_INT64_T, p, comm);
        err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
        if (mtxdisterror_allreduce(disterr, err)) {
            for (int q = p-1; q >= 0; q--)
                mtxdistfile_free(&srcs[q]);
            free(srcs);
            free(part_sizes);
            mtxfile_free(&mtxfile);
            return MTX_ERR_MPI_COLLECTIVE;
        }
        struct mtxpartition datapart;
        err = mtxpartition_init_block(
            &datapart, part_sizes[p], comm_size, part_sizes);
        if (mtxdisterror_allreduce(disterr, err)) {
            for (int q = p-1; q >= 0; q--)
                mtxdistfile_free(&srcs[q]);
            free(srcs);
            free(part_sizes);
            mtxfile_free(&mtxfile);
            return MTX_ERR_MPI_COLLECTIVE;
        }
        part_sizes[p] = 0;

        err = mtxdistfile_from_mtxfile(
            &srcs[p], rank == p ? &mtxfile : NULL,
            &datapart, comm, p, disterr);
        if (err) {
            mtxpartition_free(&datapart);
            for (int q = p-1; q >= 0; q--)
                mtxdistfile_free(&srcs[q]);
            free(srcs);
            free(part_sizes);
            mtxfile_free(&mtxfile);
            return err;
        }
        mtxpartition_free(&datapart);
    }
    free(part_sizes);
    mtxfile_free(&mtxfile);

    /* 3. Join the distributed Matrix Market files together */
    err = mtxdistfile_join(
        dst, srcs, &src->rowpart, &src->colpart, disterr);
    if (err) {
        for (int p = 0; p < num_parts; p++)
            mtxdistfile_free(&srcs[p]);
        free(srcs);
        return err;
    }
    for (int p = 0; p < num_parts; p++)
        mtxdistfile_free(&srcs[p]);
    free(srcs);
    return MTX_SUCCESS;
}

/*
 * I/O functions
 */

/**
 * ‘mtxdistmatrix_read()’ reads a matrix from a Matrix Market file.
 * The file may optionally be compressed by gzip.
 *
 * The ‘precision’ argument specifies which precision to use for
 * storing matrix or matrix values.
 *
 * If ‘path’ is ‘-’, then standard input is used.
 *
 * If an error code is returned, then ‘lines_read’ and ‘bytes_read’
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the matrix.
 */
int mtxdistmatrix_read(
    struct mtxdistmatrix * distmatrix,
    enum mtxprecision precision,
    enum mtxmatrixtype type,
    const char * path,
    bool gzip,
    int * lines_read,
    int64_t * bytes_read);

/**
 * ‘mtxdistmatrix_fread()’ reads a matrix from a stream in Matrix
 * Market format.
 *
 * ‘precision’ is used to determine the precision to use for storing
 * the values of matrix or matrix entries.
 *
 * If an error code is returned, then ‘lines_read’ and ‘bytes_read’
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the matrix.
 */
int mtxdistmatrix_fread(
    struct mtxdistmatrix * distmatrix,
    enum mtxprecision precision,
    enum mtxmatrixtype type,
    FILE * f,
    int * lines_read,
    int64_t * bytes_read,
    size_t line_max,
    char * linebuf);

#ifdef LIBMTX_HAVE_LIBZ
/**
 * ‘mtxdistmatrix_gzread()’ reads a matrix from a gzip-compressed
 * stream.
 *
 * ‘precision’ is used to determine the precision to use for storing
 * the values of matrix or matrix entries.
 *
 * If an error code is returned, then ‘lines_read’ and ‘bytes_read’
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the matrix.
 */
int mtxdistmatrix_gzread(
    struct mtxdistmatrix * distmatrix,
    enum mtxprecision precision,
    enum mtxmatrixtype type,
    gzFile f,
    int * lines_read,
    int64_t * bytes_read,
    size_t line_max,
    char * linebuf);
#endif

/**
 * ‘mtxdistmatrix_write()’ writes a matrix to a Matrix Market
 * file. The file may optionally be compressed by gzip.
 *
 * If ‘path’ is ‘-’, then standard output is used.
 *
 * If ‘fmt’ is ‘NULL’, then the format specifier ‘%g’ is used to print
 * floating point numbers with enough digits to ensure correct
 * round-trip conversion from decimal text and back.  Otherwise, the
 * given format string is used to print numerical values.
 *
 * The format string follows the conventions of ‘printf’. If the field
 * is ‘real’, ‘double’ or ‘complex’, then the format specifiers '%e',
 * '%E', '%f', '%F', '%g' or '%G' may be used. If the field is
 * ‘integer’, then the format specifier must be '%d'. The format
 * string is ignored if the field is ‘pattern’. Field width and
 * precision may be specified (e.g., "%3.1f"), but variable field
 * width and precision (e.g., "%*.*f"), as well as length modifiers
 * (e.g., "%Lf") are not allowed.
 */
int mtxdistmatrix_write(
    const struct mtxdistmatrix * distmatrix,
    enum mtxfileformat mtxfmt,
    const char * path,
    bool gzip,
    const char * fmt,
    int64_t * bytes_written);

/**
 * ‘mtxdistmatrix_fwrite()’ writes a matrix to a stream.
 *
 * If ‘fmt’ is ‘NULL’, then the format specifier ‘%g’ is used to print
 * floating point numbers with enough digits to ensure correct
 * round-trip conversion from decimal text and back.  Otherwise, the
 * given format string is used to print numerical values.
 *
 * The format string follows the conventions of ‘printf’. If the field
 * is ‘real’, ‘double’ or ‘complex’, then the format specifiers '%e',
 * '%E', '%f', '%F', '%g' or '%G' may be used. If the field is
 * ‘integer’, then the format specifier must be '%d'. The format
 * string is ignored if the field is ‘pattern’. Field width and
 * precision may be specified (e.g., "%3.1f"), but variable field
 * width and precision (e.g., "%*.*f"), as well as length modifiers
 * (e.g., "%Lf") are not allowed.
 *
 * If it is not ‘NULL’, then the number of bytes written to the stream
 * is returned in ‘bytes_written’.
 */
int mtxdistmatrix_fwrite(
    const struct mtxdistmatrix * distmatrix,
    enum mtxfileformat mtxfmt,
    FILE * f,
    const char * fmt,
    int64_t * bytes_written);

/**
 * ‘mtxdistmatrix_fwrite_shared()’ writes a distributed matrix as a
 * Matrix Market file to a single stream that is shared by every
 * process in the communicator.
 *
 * If ‘fmt’ is ‘NULL’, then the format specifier ‘%g’ is used to print
 * floating point numbers with enough digits to ensure correct
 * round-trip conversion from decimal text and back.  Otherwise, the
 * given format string is used to print numerical values.
 *
 * The format string follows the conventions of ‘printf’. If the field
 * is ‘real’, ‘double’ or ‘complex’, then the format specifiers '%e',
 * '%E', '%f', '%F', '%g' or '%G' may be used. If the field is
 * ‘integer’, then the format specifier must be '%d'. The format
 * string is ignored if the field is ‘pattern’. Field width and
 * precision may be specified (e.g., "%3.1f"), but variable field
 * width and precision (e.g., "%*.*f"), as well as length modifiers
 * (e.g., "%Lf") are not allowed.
 *
 * If it is not ‘NULL’, then the number of bytes written to the stream
 * is returned in ‘bytes_written’.
 *
 * Note that only the specified ‘root’ process will print anything to
 * the stream. Other processes will therefore send their part of the
 * distributed Matrix Market file to the root process for printing.
 *
 * This function performs collective communication and therefore
 * requires every process in the communicator to perform matching
 * calls to the function.
 */
int mtxdistmatrix_fwrite_shared(
    const struct mtxdistmatrix * mtxdistmatrix,
    enum mtxfileformat mtxfmt,
    FILE * f,
    const char * fmt,
    int64_t * bytes_written,
    int root,
    struct mtxdisterror * disterr);

#ifdef LIBMTX_HAVE_LIBZ
/**
 * ‘mtxdistmatrix_gzwrite()’ writes a matrix to a gzip-compressed
 * stream.
 *
 * If ‘fmt’ is ‘NULL’, then the format specifier ‘%g’ is used to print
 * floating point numbers with enough digits to ensure correct
 * round-trip conversion from decimal text and back.  Otherwise, the
 * given format string is used to print numerical values.
 *
 * The format string follows the conventions of ‘printf’. If the field
 * is ‘real’, ‘double’ or ‘complex’, then the format specifiers '%e',
 * '%E', '%f', '%F', '%g' or '%G' may be used. If the field is
 * ‘integer’, then the format specifier must be '%d'. The format
 * string is ignored if the field is ‘pattern’. Field width and
 * precision may be specified (e.g., "%3.1f"), but variable field
 * width and precision (e.g., "%*.*f"), as well as length modifiers
 * (e.g., "%Lf") are not allowed.
 *
 * If it is not ‘NULL’, then the number of bytes written to the stream
 * is returned in ‘bytes_written’.
 */
int mtxdistmatrix_gzwrite(
    const struct mtxdistmatrix * distmatrix,
    enum mtxfileformat mtxfmt,
    gzFile f,
    const char * fmt,
    int64_t * bytes_written);
#endif

/*
 * Level 1 BLAS operations
 */

/**
 * ‘mtxdistmatrix_swap()’ swaps values of two matrices, simultaneously
 * performing ‘y <- x’ and ‘x <- y’.
 */
int mtxdistmatrix_swap(
    struct mtxdistmatrix * x,
    struct mtxdistmatrix * y,
    struct mtxdisterror * disterr)
{
    int result;
    disterr->mpierrcode = MPI_Comm_compare(x->comm, y->comm, &result);
    int err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    err = result != MPI_IDENT && result != MPI_CONGRUENT
        ? MTX_ERR_INCOMPATIBLE_MPI_COMM : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    err = mtxmatrix_swap(&x->interior, &y->interior);
    return mtxdisterror_allreduce(disterr, err);
}

/**
 * ‘mtxdistmatrix_copy()’ copies values of a matrix, ‘y = x’.
 */
int mtxdistmatrix_copy(
    struct mtxdistmatrix * y,
    const struct mtxdistmatrix * x,
    struct mtxdisterror * disterr)
{
    int result;
    disterr->mpierrcode = MPI_Comm_compare(x->comm, y->comm, &result);
    int err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    err = result != MPI_IDENT && result != MPI_CONGRUENT
        ? MTX_ERR_INCOMPATIBLE_MPI_COMM : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    err = mtxmatrix_copy(&y->interior, &x->interior);
    return mtxdisterror_allreduce(disterr, err);
}

/**
 * ‘mtxdistmatrix_sscal()’ scales a matrix by a single precision
 * floating point scalar, ‘x = a*x’.
 */
int mtxdistmatrix_sscal(
    float a,
    struct mtxdistmatrix * x,
    int64_t * num_flops,
    struct mtxdisterror * disterr)
{
    int err = mtxmatrix_sscal(a, &x->interior, num_flops);
    return mtxdisterror_allreduce(disterr, err);
}

/**
 * ‘mtxdistmatrix_dscal()’ scales a matrix by a double precision
 * floating point scalar, ‘x = a*x’.
 */
int mtxdistmatrix_dscal(
    double a,
    struct mtxdistmatrix * x,
    int64_t * num_flops,
    struct mtxdisterror * disterr)
{
    int err = mtxmatrix_dscal(a, &x->interior, num_flops);
    return mtxdisterror_allreduce(disterr, err);
}

/**
 * ‘mtxdistmatrix_saxpy()’ adds a matrix to another matrix multiplied
 * by a single precision floating point value, ‘y = a*x + y’.
 */
int mtxdistmatrix_saxpy(
    float a,
    const struct mtxdistmatrix * x,
    struct mtxdistmatrix * y,
    int64_t * num_flops,
    struct mtxdisterror * disterr)
{
    int result;
    disterr->mpierrcode = MPI_Comm_compare(x->comm, y->comm, &result);
    int err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    err = result != MPI_IDENT && result != MPI_CONGRUENT
        ? MTX_ERR_INCOMPATIBLE_MPI_COMM : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    err = mtxmatrix_saxpy(a, &x->interior, &y->interior, num_flops);
    return mtxdisterror_allreduce(disterr, err);
}

/**
 * ‘mtxdistmatrix_daxpy()’ adds a matrix to another matrix multiplied
 * by a double precision floating point value, ‘y = a*x + y’.
 */
int mtxdistmatrix_daxpy(
    double a,
    const struct mtxdistmatrix * x,
    struct mtxdistmatrix * y,
    int64_t * num_flops,
    struct mtxdisterror * disterr)
{
    int result;
    disterr->mpierrcode = MPI_Comm_compare(x->comm, y->comm, &result);
    int err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    err = result != MPI_IDENT && result != MPI_CONGRUENT
        ? MTX_ERR_INCOMPATIBLE_MPI_COMM : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    err = mtxmatrix_daxpy(a, &x->interior, &y->interior, num_flops);
    return mtxdisterror_allreduce(disterr, err);
}

/**
 * ‘mtxdistmatrix_saypx()’ multiplies a matrix by a single precision
 * floating point scalar and adds another matrix, ‘y = a*y + x’.
 */
int mtxdistmatrix_saypx(
    float a,
    struct mtxdistmatrix * y,
    const struct mtxdistmatrix * x,
    int64_t * num_flops,
    struct mtxdisterror * disterr)
{
    int result;
    disterr->mpierrcode = MPI_Comm_compare(x->comm, y->comm, &result);
    int err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    err = result != MPI_IDENT && result != MPI_CONGRUENT
        ? MTX_ERR_INCOMPATIBLE_MPI_COMM : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    err = mtxmatrix_saypx(a, &y->interior, &x->interior, num_flops);
    return mtxdisterror_allreduce(disterr, err);
}

/**
 * ‘mtxdistmatrix_daypx()’ multiplies a matrix by a double precision
 * floating point scalar and adds another matrix, ‘y = a*y + x’.
 */
int mtxdistmatrix_daypx(
    double a,
    struct mtxdistmatrix * y,
    const struct mtxdistmatrix * x,
    int64_t * num_flops,
    struct mtxdisterror * disterr)
{
    int result;
    disterr->mpierrcode = MPI_Comm_compare(x->comm, y->comm, &result);
    int err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    err = result != MPI_IDENT && result != MPI_CONGRUENT
        ? MTX_ERR_INCOMPATIBLE_MPI_COMM : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    err = mtxmatrix_daypx(a, &y->interior, &x->interior, num_flops);
    return mtxdisterror_allreduce(disterr, err);
}

/**
 * ‘mtxdistmatrix_sdot()’ computes the Euclidean dot product of two
 * matrices in single precision floating point.
 */
int mtxdistmatrix_sdot(
    const struct mtxdistmatrix * x,
    const struct mtxdistmatrix * y,
    float * dot,
    int64_t * num_flops,
    struct mtxdisterror * disterr)
{
    int result;
    disterr->mpierrcode = MPI_Comm_compare(x->comm, y->comm, &result);
    int err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    err = result != MPI_IDENT && result != MPI_CONGRUENT
        ? MTX_ERR_INCOMPATIBLE_MPI_COMM : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    err = mtxmatrix_sdot(&x->interior, &y->interior, dot, num_flops);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    disterr->mpierrcode = MPI_Allreduce(
        MPI_IN_PLACE, dot, 1, MPI_FLOAT, MPI_SUM, x->comm);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistmatrix_ddot()’ computes the Euclidean dot product of two
 * matrices in double precision floating point.
 */
int mtxdistmatrix_ddot(
    const struct mtxdistmatrix * x,
    const struct mtxdistmatrix * y,
    double * dot,
    int64_t * num_flops,
    struct mtxdisterror * disterr)
{
    int result;
    disterr->mpierrcode = MPI_Comm_compare(x->comm, y->comm, &result);
    int err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    err = result != MPI_IDENT && result != MPI_CONGRUENT
        ? MTX_ERR_INCOMPATIBLE_MPI_COMM : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    err = mtxmatrix_ddot(&x->interior, &y->interior, dot, num_flops);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    disterr->mpierrcode = MPI_Allreduce(
        MPI_IN_PLACE, dot, 1, MPI_DOUBLE, MPI_SUM, x->comm);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistmatrix_cdotu()’ computes the product of the transpose of a
 * complex row matrix with another complex row matrix in single
 * precision floating point, ‘dot := x^T*y’.
 */
int mtxdistmatrix_cdotu(
    const struct mtxdistmatrix * x,
    const struct mtxdistmatrix * y,
    float (* dot)[2],
    int64_t * num_flops,
    struct mtxdisterror * disterr)
{
    int result;
    disterr->mpierrcode = MPI_Comm_compare(x->comm, y->comm, &result);
    int err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    err = result != MPI_IDENT && result != MPI_CONGRUENT
        ? MTX_ERR_INCOMPATIBLE_MPI_COMM : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    err = mtxmatrix_cdotu(&x->interior, &y->interior, dot, num_flops);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    disterr->mpierrcode = MPI_Allreduce(
        MPI_IN_PLACE, dot, 2, MPI_FLOAT, MPI_SUM, x->comm);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistmatrix_zdotu()’ computes the product of the transpose of a
 * complex row matrix with another complex row matrix in double
 * precision floating point, ‘dot := x^T*y’.
 */
int mtxdistmatrix_zdotu(
    const struct mtxdistmatrix * x,
    const struct mtxdistmatrix * y,
    double (* dot)[2],
    int64_t * num_flops,
    struct mtxdisterror * disterr)
{
    int result;
    disterr->mpierrcode = MPI_Comm_compare(x->comm, y->comm, &result);
    int err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    err = result != MPI_IDENT && result != MPI_CONGRUENT
        ? MTX_ERR_INCOMPATIBLE_MPI_COMM : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    err = mtxmatrix_zdotu(&x->interior, &y->interior, dot, num_flops);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    disterr->mpierrcode = MPI_Allreduce(
        MPI_IN_PLACE, dot, 2, MPI_DOUBLE, MPI_SUM, x->comm);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistmatrix_cdotc()’ computes the Euclidean dot product of two
 * complex matrices in single precision floating point, ‘dot := x^H*y’.
 */
int mtxdistmatrix_cdotc(
    const struct mtxdistmatrix * x,
    const struct mtxdistmatrix * y,
    float (* dot)[2],
    int64_t * num_flops,
    struct mtxdisterror * disterr)
{
    int result;
    disterr->mpierrcode = MPI_Comm_compare(x->comm, y->comm, &result);
    int err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    err = result != MPI_IDENT && result != MPI_CONGRUENT
        ? MTX_ERR_INCOMPATIBLE_MPI_COMM : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    err = mtxmatrix_cdotc(&x->interior, &y->interior, dot, num_flops);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    disterr->mpierrcode = MPI_Allreduce(
        MPI_IN_PLACE, dot, 2, MPI_FLOAT, MPI_SUM, x->comm);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistmatrix_zdotc()’ computes the Euclidean dot product of two
 * complex matrices in double precision floating point, ‘dot := x^H*y’.
 */
int mtxdistmatrix_zdotc(
    const struct mtxdistmatrix * x,
    const struct mtxdistmatrix * y,
    double (* dot)[2],
    int64_t * num_flops,
    struct mtxdisterror * disterr)
{
    int result;
    disterr->mpierrcode = MPI_Comm_compare(x->comm, y->comm, &result);
    int err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    err = result != MPI_IDENT && result != MPI_CONGRUENT
        ? MTX_ERR_INCOMPATIBLE_MPI_COMM : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    err = mtxmatrix_zdotc(&x->interior, &y->interior, dot, num_flops);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    disterr->mpierrcode = MPI_Allreduce(
        MPI_IN_PLACE, dot, 2, MPI_DOUBLE, MPI_SUM, x->comm);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistmatrix_snrm2()’ computes the Euclidean norm of a matrix in
 * single precision floating point.
 */
int mtxdistmatrix_snrm2(
    const struct mtxdistmatrix * x,
    float * nrm2,
    int64_t * num_flops,
    struct mtxdisterror * disterr)
{
    int err = mtxmatrix_sdot(&x->interior, &x->interior, nrm2, num_flops);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    disterr->mpierrcode = MPI_Allreduce(
        MPI_IN_PLACE, nrm2, 1, MPI_FLOAT, MPI_SUM, x->comm);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    *nrm2 = sqrtf(*nrm2);
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistmatrix_dnrm2()’ computes the Euclidean norm of a matrix in
 * double precision floating point.
 */
int mtxdistmatrix_dnrm2(
    const struct mtxdistmatrix * x,
    double * nrm2,
    int64_t * num_flops,
    struct mtxdisterror * disterr)
{
    int err = mtxmatrix_ddot(&x->interior, &x->interior, nrm2, num_flops);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    disterr->mpierrcode = MPI_Allreduce(
        MPI_IN_PLACE, nrm2, 1, MPI_DOUBLE, MPI_SUM, x->comm);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    *nrm2 = sqrt(*nrm2);
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistmatrix_sasum()’ computes the sum of absolute values
 * (1-norm) of a matrix in single precision floating point.
 */
int mtxdistmatrix_sasum(
    const struct mtxdistmatrix * x,
    float * asum,
    int64_t * num_flops,
    struct mtxdisterror * disterr)
{
    int err = mtxmatrix_sasum(&x->interior, asum, num_flops);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    disterr->mpierrcode = MPI_Allreduce(
        MPI_IN_PLACE, asum, 1, MPI_FLOAT, MPI_SUM, x->comm);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistmatrix_dasum()’ computes the sum of absolute values
 * (1-norm) of a matrix in double precision floating point.
 */
int mtxdistmatrix_dasum(
    const struct mtxdistmatrix * x,
    double * asum,
    int64_t * num_flops,
    struct mtxdisterror * disterr)
{
    int err = mtxmatrix_dasum(&x->interior, asum, num_flops);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    disterr->mpierrcode = MPI_Allreduce(
        MPI_IN_PLACE, asum, 1, MPI_DOUBLE, MPI_SUM, x->comm);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistmatrix_iamax()’ finds the index of the first element having
 * the maximum absolute value.
 */
int mtxdistmatrix_iamax(
    const struct mtxdistmatrix * x,
    int * max,
    struct mtxdisterror * disterr);

/*
 * Level 2 BLAS operations
 */

/**
 * ‘mtxdistmatrix_sgemv()’ multiplies a matrix ‘A’ or its transpose ‘A'’
 * by a real scalar ‘alpha’ (‘α’) and a vector ‘x’, before adding the
 * result to another vector ‘y’ multiplied by another real scalar
 * ‘beta’ (‘β’). That is, ‘y = α*A*x + β*y’ or ‘y = α*A'*x + β*y’.
 *
 * The scalars ‘alpha’ and ‘beta’ are given as single precision
 * floating point numbers.
 */
int mtxdistmatrix_sgemv(
    enum mtxtransposition trans,
    float alpha,
    const struct mtxdistmatrix * A,
    const struct mtxdistvector * x,
    float beta,
    struct mtxdistvector * y,
    int64_t * num_flops,
    struct mtxdisterror * disterr)
{
    int err;
    struct mtxdistmatrixgemv gemv;
    err = mtxdistmatrixgemv_init(&gemv, trans, A, x, y, disterr);
    if (err) return err;
    err = mtxdistmatrixgemv_sgemv(&gemv, alpha, beta, num_flops, disterr);
    if (err) {
        mtxdistmatrixgemv_free(&gemv);
        return err;
    }
    err = mtxdistmatrixgemv_wait(&gemv, disterr);
    if (err) {
        mtxdistmatrixgemv_free(&gemv);
        return err;
    }
    mtxdistmatrixgemv_free(&gemv);
    return MTX_SUCCESS;
#if 0
    int result;
    if (trans == mtx_notrans) {
        disterr->mpierrcode = MPI_Comm_compare(A->rowcomm, x->comm, &result);
        int err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
        if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
        err = result != MPI_IDENT && result != MPI_CONGRUENT
            ? MTX_ERR_INCOMPATIBLE_MPI_COMM : MTX_SUCCESS;
        if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
        disterr->mpierrcode = MPI_Comm_compare(A->colcomm, y->comm, &result);
        err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
        if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
        err = result != MPI_IDENT && result != MPI_CONGRUENT
            ? MTX_ERR_INCOMPATIBLE_MPI_COMM : MTX_SUCCESS;
        if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;

        MPI_Comm comm = A->comm;
        int comm_size = A->comm_size;
        int P = A->comm_size / A->colpart.num_parts;
        int Q = A->colpart.num_parts;
        int r = A->rank;

        int p = r / A->colpart.num_parts;
        int q = r % A->colpart.num_parts;
        int num_local_rows =
            p < A->rowpart.num_parts ? A->rowpart.part_sizes[p] : 0;
        int num_local_columns =
            q < A->colpart.num_parts ? A->colpart.part_sizes[q] : 0;

        err = mtxpartition_compare(&A->rowpart, &y->rowpart, &result);
        if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
        if (result == 0) {
            /* Rows of the matrix are partitioned in the same way as rows
             * of the destination vector. We can therefore use the destination
             * vector ‘y’ to store the computed results directly. */

            err = mtxpartition_compare(&A->colpart, &x->rowpart, &result);
            if (mtxdisterror_allreduce(disterr, err))
                return MTX_ERR_MPI_COLLECTIVE;

            if (result == 0) {
                /* Columns of the matrix are partitioned in the same way
                 * as columns of the source vector. We can therefore use
                 * the part of the source vector ‘x’ on the current process
                 * to compute results directly. */
                err = mtxmatrix_sgemv(
                    trans, alpha, &A->interior, &x->interior, beta, &y->interior,
                    num_flops);
                if (mtxdisterror_allreduce(disterr, err))
                    return MTX_ERR_MPI_COLLECTIVE;

            } else {
                /* Columns of the matrix are not partitioned in the same
                 * way as columns of the source vector. Thus, additional
                 * storage and communication is needed to prepare the part
                 * of the source vector that is needed by the current
                 * process to compute its part of the destination vector. */

#if 0
                /* allocate temporary source vector */
                struct mtxdistvector xr;
                for (int p = 0; p < rowpart.num_parts; p++) {
                    err = mtxdistmatrix_alloc_row_vector(
                        A, &xr, x->interior.type, disterr);
                    if (err) return err;
                }

                /* Obtain the global column numbers according to the
                 * partitioning of the matrix columns. These are the
                 * global element numbers of the source vector that are
                 * needed to perform the local part of the matrix-vector
                 * multiplication. */
                int64_t * colidx = malloc(num_local_columns * sizeof(int64_t));
                err = !colidx ? MTX_ERR_ERRNO : MTX_SUCCESS;
                if (mtxdisterror_allreduce(disterr, err)) {
                    mtxvector_free(&xr);
                    return MTX_ERR_MPI_COLLECTIVE;
                }
                for (int64_t j = 0; j < num_local_columns; j++)
                    colidx[j] = j;
                err = mtxpartition_globalidx(
                    &A->colpart, q, num_local_columns, colidx, colidx);
                if (mtxdisterror_allreduce(disterr, err)) {
                    free(colidx);
                    mtxvector_free(&xr);
                    return MTX_ERR_MPI_COLLECTIVE;
                }

                free(colidx);

                /* perform halo update to gather source vector */
                err = mtxdistvector_halo_update(&xr, x, MPI_COMM_NULL, disterr);
                if (err) {
                    mtxdistvector_free(&xr);
                    return err;
                }

                /* perform local matrix-vector multiplication */
                err = mtxmatrix_sgemv(
                    trans, alpha, &A->interior, &xr.interior, beta, &y->interior,
                    num_flops);
                if (mtxdisterror_allreduce(disterr, err)) {
                    mtxdistvector_free(&xr);
                    return MTX_ERR_MPI_COLLECTIVE;
                }
                mtxdistvector_free(&xr);
#endif
            }

        } else {
            /* Rows of the matrix are not partitioned in the same way as
             * rows of the destination vector. Thus, additional storage
             * and communication is needed to form the final destination
             * vector. */

            /* TODO: not implemented */
            errno = ENOTSUP;
            return MTX_ERR_ERRNO;
        }

    } else if (trans == mtx_trans || trans == mtx_conjtrans) {
        disterr->mpierrcode = MPI_Comm_compare(A->colcomm, x->comm, &result);
        int err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
        if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
        err = result != MPI_IDENT && result != MPI_CONGRUENT
            ? MTX_ERR_INCOMPATIBLE_MPI_COMM : MTX_SUCCESS;
        if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
        disterr->mpierrcode = MPI_Comm_compare(A->rowcomm, y->comm, &result);
        err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
        if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
        err = result != MPI_IDENT && result != MPI_CONGRUENT
            ? MTX_ERR_INCOMPATIBLE_MPI_COMM : MTX_SUCCESS;
        if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;

        MPI_Comm comm = A->comm;
        int comm_size = A->comm_size;
        int P = A->comm_size / A->colpart.num_parts;
        int Q = A->colpart.num_parts;
        int r = A->rank;

        int p = r / A->colpart.num_parts;
        int q = r % A->colpart.num_parts;
        int num_local_rows =
            p < A->rowpart.num_parts ? A->rowpart.part_sizes[p] : 0;
        int num_local_columns =
            q < A->colpart.num_parts ? A->colpart.part_sizes[q] : 0;

        err = mtxpartition_compare(&A->colpart, &y->rowpart, &result);
        if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
        if (result == 0) {
            /*
             * Rows of the transposed matrix are partitioned in the
             * same way as rows of the destination vector. We can
             * therefore use the destination vector ‘y’ to store the
             * computed results directly.
             */

            err = mtxpartition_compare(&A->rowpart, &x->rowpart, &result);
            if (mtxdisterror_allreduce(disterr, err))
                return MTX_ERR_MPI_COLLECTIVE;

            if (result == 0) {
                /*
                 * Columns of the transposed matrix are partitioned in
                 * the same way as columns of the source vector. We
                 * can therefore use the part of the source vector ‘x’
                 * on the current process to compute results directly.
                 */
                err = mtxmatrix_sgemv(
                    trans, alpha, &A->interior, &x->interior, beta, &y->interior,
                    num_flops);
                if (mtxdisterror_allreduce(disterr, err))
                    return MTX_ERR_MPI_COLLECTIVE;

            } else {
                /*
                 * Columns of the transposed matrix are not
                 * partitioned in the same way as columns of the
                 * source vector. Thus, additional storage and
                 * communication is needed to prepare the part of the
                 * source vector that is needed by the current process
                 * to compute its part of the destination vector.
                 */

                /* allocate temporary source vector */
                struct mtxdistvector xr;
                err = mtxdistmatrix_alloc_column_vector(
                    A, &xr, x->interior.type, disterr);
                if (err) return err;

                /* perform halo update to gather source vector */
                err = mtxdistvector_halo_update(&xr, x, disterr);
                if (err) {
                    mtxdistvector_free(&xr);
                    return err;
                }

                /* perform local matrix-vector multiplication */
                err = mtxmatrix_sgemv(
                    trans, alpha, &A->interior, &xr.interior, beta, &y->interior,
                    num_flops);
                if (mtxdisterror_allreduce(disterr, err)) {
                    mtxdistvector_free(&xr);
                    return MTX_ERR_MPI_COLLECTIVE;
                }
                mtxdistvector_free(&xr);
            }

        } else {
            /* Rows of the transposed matrix are not partitioned in
             * the same way as rows of the destination vector. Thus,
             * additional storage and communication is needed to form
             * the final destination vector. */

            /* TODO: not implemented */
            errno = ENOTSUP;
            return MTX_ERR_ERRNO;
        }
    } else {
        return MTX_ERR_INVALID_TRANSPOSITION;
    }
    return MTX_SUCCESS;
#endif
}

/**
 * ‘mtxdistmatrix_sgemv2()’ multiplies a matrix ‘A’ or its transpose
 * ‘A'’ by a real scalar ‘alpha’ (‘α’) and a vector ‘x’, before adding
 * the result to another vector ‘y’ multiplied by another real scalar
 * ‘beta’ (‘β’). That is, ‘y = α*A*x + β*y’ or ‘y = α*A'*x + β*y’.
 *
 * The scalars ‘alpha’ and ‘beta’ are given as single precision
 * floating point numbers.
 */
int mtxdistmatrix_sgemv2(
    enum mtxtransposition trans,
    float alpha,
    const struct mtxdistmatrix * A,
    const struct mtxvector_dist * x,
    float beta,
    struct mtxvector_dist * y,
    int64_t * num_flops,
    struct mtxdisterror * disterr)
{
    int err;
    struct mtxdistmatrixgemv2 gemv;
    err = mtxdistmatrixgemv2_init(&gemv, trans, A, x, y, disterr);
    if (err) return err;
    err = mtxdistmatrixgemv2_sgemv(&gemv, alpha, beta, num_flops, disterr);
    if (err) {
        mtxdistmatrixgemv2_free(&gemv);
        return err;
    }
    err = mtxdistmatrixgemv2_wait(&gemv, disterr);
    if (err) {
        mtxdistmatrixgemv2_free(&gemv);
        return err;
    }
    mtxdistmatrixgemv2_free(&gemv);
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistmatrix_dgemv()’ multiplies a matrix ‘A’ or its transpose ‘A'’
 * by a real scalar ‘alpha’ (‘α’) and a vector ‘x’, before adding the
 * result to another vector ‘y’ multiplied by another scalar real
 * ‘beta’ (‘β’).  That is, ‘y = α*A*x + β*y’ or ‘y = α*A'*x + β*y’.
 *
 * The scalars ‘alpha’ and ‘beta’ are given as double precision
 * floating point numbers.
 */
int mtxdistmatrix_dgemv(
    enum mtxtransposition trans,
    double alpha,
    const struct mtxdistmatrix * A,
    const struct mtxdistvector * x,
    double beta,
    struct mtxdistvector * y,
    int64_t * num_flops,
    struct mtxdisterror * disterr)
{
    int err;
    struct mtxdistmatrixgemv gemv;
    err = mtxdistmatrixgemv_init(&gemv, trans, A, x, y, disterr);
    if (err)
        return err;
    err = mtxdistmatrixgemv_dgemv(&gemv, alpha, beta, num_flops, disterr);
    if (err) {
        mtxdistmatrixgemv_free(&gemv);
        return err;
    }
    err = mtxdistmatrixgemv_wait(&gemv, disterr);
    if (err) {
        mtxdistmatrixgemv_free(&gemv);
        return err;
    }
    mtxdistmatrixgemv_free(&gemv);
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistmatrix_cgemv()’ multiplies a complex-valued matrix ‘A’, its
 * transpose ‘A'’ or its conjugate transpose ‘Aᴴ’ by a complex scalar
 * ‘alpha’ (‘α’) and a vector ‘x’, before adding the result to another
 * vector ‘y’ multiplied by another complex scalar ‘beta’ (‘β’).  That
 * is, ‘y = α*A*x + β*y’, ‘y = α*A'*x + β*y’ or ‘y = α*Aᴴ*x + β*y’.
 *
 * The scalars ‘alpha’ and ‘beta’ are given as single precision
 * floating point numbers.
 */
int mtxdistmatrix_cgemv(
    enum mtxtransposition trans,
    float alpha[2],
    const struct mtxdistmatrix * A,
    const struct mtxdistvector * x,
    float beta[2],
    struct mtxdistvector * y,
    int64_t * num_flops,
    struct mtxdisterror * disterr)
{
    int err;
    struct mtxdistmatrixgemv gemv;
    err = mtxdistmatrixgemv_init(&gemv, trans, A, x, y, disterr);
    if (err)
        return err;
    err = mtxdistmatrixgemv_cgemv(&gemv, alpha, beta, num_flops, disterr);
    if (err) {
        mtxdistmatrixgemv_free(&gemv);
        return err;
    }
    err = mtxdistmatrixgemv_wait(&gemv, disterr);
    if (err) {
        mtxdistmatrixgemv_free(&gemv);
        return err;
    }
    mtxdistmatrixgemv_free(&gemv);
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistmatrix_zgemv()’ multiplies a complex-valued matrix ‘A’, its
 * transpose ‘A'’ or its conjugate transpose ‘Aᴴ’ by a complex scalar
 * ‘alpha’ (‘α’) and a vector ‘x’, before adding the result to another
 * vector ‘y’ multiplied by another complex scalar ‘beta’ (‘β’).  That
 * is, ‘y = α*A*x + β*y’, ‘y = α*A'*x + β*y’ or ‘y = α*Aᴴ*x + β*y’.
 *
 * The scalars ‘alpha’ and ‘beta’ are given as double precision
 * floating point numbers.
 */
int mtxdistmatrix_zgemv(
    enum mtxtransposition trans,
    double alpha[2],
    const struct mtxdistmatrix * A,
    const struct mtxdistvector * x,
    double beta[2],
    struct mtxdistvector * y,
    int64_t * num_flops,
    struct mtxdisterror * disterr)
{
    int err;
    struct mtxdistmatrixgemv gemv;
    err = mtxdistmatrixgemv_init(&gemv, trans, A, x, y, disterr);
    if (err)
        return err;
    err = mtxdistmatrixgemv_zgemv(&gemv, alpha, beta, num_flops, disterr);
    if (err) {
        mtxdistmatrixgemv_free(&gemv);
        return err;
    }
    err = mtxdistmatrixgemv_wait(&gemv, disterr);
    if (err) {
        mtxdistmatrixgemv_free(&gemv);
        return err;
    }
    mtxdistmatrixgemv_free(&gemv);
    return MTX_SUCCESS;
}
#endif
