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
 * Last modified: 2022-01-20
 *
 * Data structures for distributed matrices.
 */

#include <libmtx/libmtx-config.h>

#ifdef LIBMTX_HAVE_MPI
#include <libmtx/error.h>
#include <libmtx/field.h>
#include <libmtx/matrix/distmatrix.h>
#include <libmtx/matrix/matrix.h>
#include <libmtx/mtxfile/header.h>
#include <libmtx/mtxfile/mtxdistfile.h>
#include <libmtx/mtxfile/mtxfile.h>
#include <libmtx/mtxfile/size.h>
#include <libmtx/precision.h>
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
    mtxpartition_free(&distmatrix->rowpart);
    mtxpartition_free(&distmatrix->colpart);
    mtxmatrix_free(&distmatrix->interior);
}

static int mtxdistmatrix_init_comm(
    struct mtxdistmatrix * distmatrix,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err;
    int comm_size;
    disterr->mpierrcode = MPI_Comm_size(comm, &comm_size);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;
    int rank;
    disterr->mpierrcode = MPI_Comm_rank(comm, &rank);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;
    distmatrix->comm = comm;
    distmatrix->comm_size = comm_size;
    distmatrix->rank = rank;
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
                if (num_local_rows != num_local_rows_t)
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

    if (rowpart) {
        int64_t num_rows = num_local_rows;
        disterr->mpierrcode = MPI_Allreduce(
            MPI_IN_PLACE, &num_rows, 1, MPI_INT64_T, MPI_SUM, comm);
        if (disterr->mpierrcode) err = MTX_ERR_MPI;
        if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
        if (rowpart->size != num_rows ||
            (p < num_row_parts && rowpart->part_sizes[p] != num_local_rows))
            err = MTX_ERR_INCOMPATIBLE_PARTITION;
        if (mtxdisterror_allreduce(disterr, err))
            return MTX_ERR_MPI_COLLECTIVE;
    }

    if (colpart) {
        int64_t num_columns = num_local_columns;
        disterr->mpierrcode = MPI_Allreduce(
            MPI_IN_PLACE, &num_columns, 1, MPI_INT64_T, MPI_SUM, comm);
        if (disterr->mpierrcode) err = MTX_ERR_MPI;
        if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
        if (colpart->size != num_columns ||
            (q < num_col_parts && colpart->part_sizes[q] != num_local_columns))
            err = MTX_ERR_INCOMPATIBLE_PARTITION;
        if (mtxdisterror_allreduce(disterr, err))
            return MTX_ERR_MPI_COLLECTIVE;
    }

    if (rowpart && colpart) {
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
        &distmatrix->interior, field, precision,
        num_local_rows, num_local_columns);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxpartition_free(&distmatrix->colpart);
        mtxpartition_free(&distmatrix->rowpart);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistmatrix_init_array_real_single()’ allocates and initialises
 * a distributed matrix in array format with real, single precision
 * coefficients.
 */
int mtxdistmatrix_init_array_real_single(
    struct mtxdistmatrix * distmatrix,
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
        &distmatrix->interior, num_local_rows, num_local_columns, data);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxpartition_free(&distmatrix->colpart);
        mtxpartition_free(&distmatrix->rowpart);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistmatrix_init_array_real_double()’ allocates and initialises
 * a distributed matrix in array format with real, double precision
 * coefficients.
 */
int mtxdistmatrix_init_array_real_double(
    struct mtxdistmatrix * distmatrix,
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
        &distmatrix->interior, num_local_rows, num_local_columns, data);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxpartition_free(&distmatrix->colpart);
        mtxpartition_free(&distmatrix->rowpart);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistmatrix_init_array_complex_single()’ allocates and
 * initialises a distributed matrix in array format with complex,
 * single precision coefficients.
 */
int mtxdistmatrix_init_array_complex_single(
    struct mtxdistmatrix * distmatrix,
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
        &distmatrix->interior, num_local_rows, num_local_columns, data);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxpartition_free(&distmatrix->colpart);
        mtxpartition_free(&distmatrix->rowpart);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistmatrix_init_array_complex_double()’ allocates and
 * initialises a distributed matrix in array format with complex,
 * double precision coefficients.
 */
int mtxdistmatrix_init_array_complex_double(
    struct mtxdistmatrix * distmatrix,
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
        &distmatrix->interior, num_local_rows, num_local_columns, data);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxpartition_free(&distmatrix->colpart);
        mtxpartition_free(&distmatrix->rowpart);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistmatrix_init_array_integer_single()’ allocates and
 * initialises a distributed matrix in array format with integer,
 * single precision coefficients.
 */
int mtxdistmatrix_init_array_integer_single(
    struct mtxdistmatrix * distmatrix,
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
        &distmatrix->interior, num_local_rows, num_local_columns, data);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxpartition_free(&distmatrix->colpart);
        mtxpartition_free(&distmatrix->rowpart);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistmatrix_init_array_integer_double()’ allocates and
 * initialises a distributed matrix in array format with integer,
 * double precision coefficients.
 */
int mtxdistmatrix_init_array_integer_double(
    struct mtxdistmatrix * distmatrix,
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
        &distmatrix->interior, num_local_rows, num_local_columns, data);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxpartition_free(&distmatrix->colpart);
        mtxpartition_free(&distmatrix->rowpart);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    return MTX_SUCCESS;
}

/*
 * Distributed matrices in coordinate format
 */

/**
 * ‘mtxdistmatrix_alloc_coordinate()’ allocates a distributed matrix
 * in coordinate format.
 */
int mtxdistmatrix_alloc_coordinate(
    struct mtxdistmatrix * distmatrix,
    enum mtxfield field,
    enum mtxprecision precision,
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
        &distmatrix->interior, field, precision,
        num_local_rows, num_local_columns, num_local_nonzeros);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxpartition_free(&distmatrix->colpart);
        mtxpartition_free(&distmatrix->rowpart);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistmatrix_init_coordinate_real_single()’ allocates and
 * initialises a distributed matrix in coordinate format with real,
 * single precision coefficients.
 */
int mtxdistmatrix_init_coordinate_real_single(
    struct mtxdistmatrix * distmatrix,
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
        &distmatrix->interior, num_local_rows, num_local_columns,
        num_local_nonzeros, rowidx, colidx, data);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxpartition_free(&distmatrix->colpart);
        mtxpartition_free(&distmatrix->rowpart);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistmatrix_init_coordinate_real_double()’ allocates and
 * initialises a distributed matrix in coordinate format with real,
 * double precision coefficients.
 */
int mtxdistmatrix_init_coordinate_real_double(
    struct mtxdistmatrix * distmatrix,
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
        &distmatrix->interior, num_local_rows, num_local_columns,
        num_local_nonzeros, rowidx, colidx, data);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxpartition_free(&distmatrix->colpart);
        mtxpartition_free(&distmatrix->rowpart);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistmatrix_init_coordinate_complex_single()’ allocates and
 * initialises a distributed matrix in coordinate format with complex,
 * single precision coefficients.
 */
int mtxdistmatrix_init_coordinate_complex_single(
    struct mtxdistmatrix * distmatrix,
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
        &distmatrix->interior, num_local_rows, num_local_columns,
        num_local_nonzeros, rowidx, colidx, data);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxpartition_free(&distmatrix->colpart);
        mtxpartition_free(&distmatrix->rowpart);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistmatrix_init_coordinate_complex_double()’ allocates and
 * initialises a distributed matrix in coordinate format with complex,
 * double precision coefficients.
 */
int mtxdistmatrix_init_coordinate_complex_double(
    struct mtxdistmatrix * distmatrix,
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
        &distmatrix->interior, num_local_rows, num_local_columns,
        num_local_nonzeros, rowidx, colidx, data);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxpartition_free(&distmatrix->colpart);
        mtxpartition_free(&distmatrix->rowpart);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistmatrix_init_coordinate_integer_single()’ allocates and
 * initialises a distributed matrix in coordinate format with integer,
 * single precision coefficients.
 */
int mtxdistmatrix_init_coordinate_integer_single(
    struct mtxdistmatrix * distmatrix,
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
        &distmatrix->interior, num_local_rows, num_local_columns,
        num_local_nonzeros, rowidx, colidx, data);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxpartition_free(&distmatrix->colpart);
        mtxpartition_free(&distmatrix->rowpart);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistmatrix_init_coordinate_integer_double()’ allocates and
 * initialises a distributed matrix in coordinate format with integer,
 * double precision coefficients.
 */
int mtxdistmatrix_init_coordinate_integer_double(
    struct mtxdistmatrix * distmatrix,
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
        &distmatrix->interior, num_local_rows, num_local_columns,
        num_local_nonzeros, rowidx, colidx, data);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxpartition_free(&distmatrix->colpart);
        mtxpartition_free(&distmatrix->rowpart);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistmatrix_init_coordinate_pattern()’ allocates and initialises
 * a distributed matrix in coordinate format with boolean
 * coefficients.
 */
int mtxdistmatrix_init_coordinate_pattern(
    struct mtxdistmatrix * distmatrix,
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
        &distmatrix->interior, num_local_rows, num_local_columns,
        num_local_nonzeros, rowidx, colidx);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxpartition_free(&distmatrix->colpart);
        mtxpartition_free(&distmatrix->rowpart);
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
        &dst->interior, &recvmtxfile, matrix_type);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxfile_free(&recvmtxfile);
        mtxpartition_free(&dst->colpart);
        mtxpartition_free(&dst->rowpart);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    mtxfile_free(&recvmtxfile);
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
    struct mtxfile * dst,
    const struct mtxdistmatrix * src,
    enum mtxfileformat mtxfmt,
    int root,
    struct mtxdisterror * disterr)
{
    int err;
    MPI_Comm comm = src->comm;
    int comm_size = src->comm_size;
    int rank = src->rank;

    /* 1. Each process converts its part of the matrix to Matrix
     * Market format */
    struct mtxfile sendmtxfile;
    err = mtxmatrix_to_mtxfile(
        &sendmtxfile, &src->interior, mtxfmt);
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;

    /* 2. Gather each part onto the root process */
    struct mtxfile * recvmtxfiles =
        rank == root ? malloc(comm_size * sizeof(struct mtxfile)) : NULL;
    err = rank == root && !recvmtxfiles ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxfile_free(&sendmtxfile);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    err = mtxfile_gather(
        &sendmtxfile, recvmtxfiles, root, comm, disterr);
    if (err) {
        if (rank == root) free(recvmtxfiles);
        mtxfile_free(&sendmtxfile);
        return err;
    }
    mtxfile_free(&sendmtxfile);

    /* 3. Join the Matrix Market files on the root process */
    err = rank == root
        ? mtxfile_join(dst, recvmtxfiles, &src->rowpart, &src->colpart)
        : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        if (rank == root) {
            for (int p = 0; p < comm_size; p++)
                mtxfile_free(&recvmtxfiles[p]);
            free(recvmtxfiles);
        }
        return MTX_ERR_MPI_COLLECTIVE;
    }
    if (rank == root) {
        for (int p = 0; p < comm_size; p++)
            mtxfile_free(&recvmtxfiles[p]);
        free(recvmtxfiles);
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
    int err = mtxdistmatrix_init_comm(
        dst, comm, disterr);
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
            ? mtxmatrix_from_mtxfile(&dst->interior, &mtxfile, matrix_type)
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
        ? mtxmatrix_to_mtxfile(&mtxfile, &src->interior, mtxfmt)
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
    int err = MPI_Comm_compare(x->comm, y->comm, &result);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    err = result != MPI_IDENT ? MTX_ERR_INCOMPATIBLE_MPI_COMM : MTX_SUCCESS;
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
    int err = MPI_Comm_compare(x->comm, y->comm, &result);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    err = result != MPI_IDENT ? MTX_ERR_INCOMPATIBLE_MPI_COMM : MTX_SUCCESS;
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
    int err = MPI_Comm_compare(x->comm, y->comm, &result);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    err = result != MPI_IDENT ? MTX_ERR_INCOMPATIBLE_MPI_COMM : MTX_SUCCESS;
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
    int err = MPI_Comm_compare(x->comm, y->comm, &result);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    err = result != MPI_IDENT ? MTX_ERR_INCOMPATIBLE_MPI_COMM : MTX_SUCCESS;
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
    int err = MPI_Comm_compare(x->comm, y->comm, &result);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    err = result != MPI_IDENT ? MTX_ERR_INCOMPATIBLE_MPI_COMM : MTX_SUCCESS;
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
    int err = MPI_Comm_compare(x->comm, y->comm, &result);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    err = result != MPI_IDENT ? MTX_ERR_INCOMPATIBLE_MPI_COMM : MTX_SUCCESS;
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
    int err = MPI_Comm_compare(x->comm, y->comm, &result);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    err = result != MPI_IDENT ? MTX_ERR_INCOMPATIBLE_MPI_COMM : MTX_SUCCESS;
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
    int err = MPI_Comm_compare(x->comm, y->comm, &result);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    err = result != MPI_IDENT ? MTX_ERR_INCOMPATIBLE_MPI_COMM : MTX_SUCCESS;
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
    int err = MPI_Comm_compare(x->comm, y->comm, &result);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    err = result != MPI_IDENT ? MTX_ERR_INCOMPATIBLE_MPI_COMM : MTX_SUCCESS;
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
    int err = MPI_Comm_compare(x->comm, y->comm, &result);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    err = result != MPI_IDENT ? MTX_ERR_INCOMPATIBLE_MPI_COMM : MTX_SUCCESS;
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
    int err = MPI_Comm_compare(x->comm, y->comm, &result);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    err = result != MPI_IDENT ? MTX_ERR_INCOMPATIBLE_MPI_COMM : MTX_SUCCESS;
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
    int err = MPI_Comm_compare(x->comm, y->comm, &result);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    err = result != MPI_IDENT ? MTX_ERR_INCOMPATIBLE_MPI_COMM : MTX_SUCCESS;
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
#endif
