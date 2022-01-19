/* This file is part of libmtx.
 *
 * Copyright (C) 2022 James D. Trotter
 *
 * libmtx is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * libmtx is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with libmtx.  If not, see <https://www.gnu.org/licenses/>.
 *
 * Authors: James D. Trotter <james@simula.no>
 * Last modified: 2022-01-14
 *
 * Data structures for distributed vectors.
 */

#include <libmtx/libmtx-config.h>

#ifdef LIBMTX_HAVE_MPI
#include <libmtx/error.h>
#include <libmtx/precision.h>
#include <libmtx/mtxfile/mtxdistfile.h>
#include <libmtx/mtxfile/header.h>
#include <libmtx/mtxfile/mtxfile.h>
#include <libmtx/mtxfile/size.h>
#include <libmtx/field.h>
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
 * ‘mtxdistvector_free()’ frees storage allocated for a vector.
 */
void mtxdistvector_free(
    struct mtxdistvector * distvector)
{
    mtxpartition_free(&distvector->partition);
    mtxvector_free(&distvector->interior);
}

/**
 * ‘mtxdistvector_alloc_copy()’ allocates storage for a copy of a
 * distributed vector without initialising the underlying values.
 */
int mtxdistvector_alloc_copy(
    struct mtxdistvector * dst,
    const struct mtxdistvector * src);

/**
 * ‘mtxdistvector_init_copy()’ creates a copy of a distributed vector.
 */
int mtxdistvector_init_copy(
    struct mtxdistvector * dst,
    const struct mtxdistvector * src);

/*
 * Distributed vectors in array format
 */

static int mtxdistvector_init_comm(
    struct mtxdistvector * distvector,
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
    distvector->comm = comm;
    distvector->comm_size = comm_size;
    distvector->rank = rank;
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistvector_alloc_array()’ allocates a distributed vector in
 * array format.
 */
int mtxdistvector_alloc_array(
    struct mtxdistvector * distvector,
    enum mtxfield field,
    enum mtxprecision precision,
    int num_rows,
    const struct mtxpartition * partition,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxdistvector_init_comm(distvector, comm, disterr);
    if (err) return err;
    if (partition->num_parts > distvector->comm_size || partition->size != num_rows)
        err = MTX_ERR_INCOMPATIBLE_PARTITION;
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;
    err = mtxpartition_copy(&distvector->partition, partition);
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;
    int64_t local_size = partition->part_sizes[distvector->rank];
    err = mtxvector_alloc_array(
        &distvector->interior, field, precision, local_size);
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistvector_init_array_real_single()’ allocates and initialises
 * a distributed vector in array format with real, single precision
 * coefficients.
 */
int mtxdistvector_init_array_real_single(
    struct mtxdistvector * distvector,
    int num_rows,
    const float * data,
    const struct mtxpartition * partition,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxdistvector_init_comm(distvector, comm, disterr);
    if (err) return err;
    if (partition->num_parts > distvector->comm_size || partition->size != num_rows)
        err = MTX_ERR_INCOMPATIBLE_PARTITION;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    err = mtxpartition_copy(&distvector->partition, partition);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    int64_t local_size = partition->part_sizes[distvector->rank];
    err = mtxvector_init_array_real_single(
        &distvector->interior, local_size, data);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistvector_init_array_real_double()’ allocates and initialises
 * a distributed vector in array format with real, double precision
 * coefficients.
 */
int mtxdistvector_init_array_real_double(
    struct mtxdistvector * distvector,
    int num_rows,
    const double * data,
    const struct mtxpartition * partition,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxdistvector_init_comm(distvector, comm, disterr);
    if (err) return err;
    if (partition->num_parts > distvector->comm_size || partition->size != num_rows)
        err = MTX_ERR_INCOMPATIBLE_PARTITION;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    err = mtxpartition_copy(&distvector->partition, partition);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    int64_t local_size = partition->part_sizes[distvector->rank];
    err = mtxvector_init_array_real_double(
        &distvector->interior, local_size, data);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistvector_init_array_complex_single()’ allocates and
 * initialises a distributed vector in array format with complex,
 * single precision coefficients.
 */
int mtxdistvector_init_array_complex_single(
    struct mtxdistvector * distvector,
    int num_rows,
    const float (* data)[2],
    const struct mtxpartition * partition,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxdistvector_init_comm(distvector, comm, disterr);
    if (err) return err;
    if (partition->num_parts > distvector->comm_size || partition->size != num_rows)
        err = MTX_ERR_INCOMPATIBLE_PARTITION;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    err = mtxpartition_copy(&distvector->partition, partition);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    int64_t local_size = partition->part_sizes[distvector->rank];
    err = mtxvector_init_array_complex_single(
        &distvector->interior, local_size, data);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistvector_init_array_complex_double()’ allocates and
 * initialises a distributed vector in array format with complex,
 * double precision coefficients.
 */
int mtxdistvector_init_array_complex_double(
    struct mtxdistvector * distvector,
    int num_rows,
    const double (* data)[2],
    const struct mtxpartition * partition,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxdistvector_init_comm(distvector, comm, disterr);
    if (err) return err;
    if (partition->num_parts > distvector->comm_size || partition->size != num_rows)
        err = MTX_ERR_INCOMPATIBLE_PARTITION;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    err = mtxpartition_copy(&distvector->partition, partition);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    int64_t local_size = partition->part_sizes[distvector->rank];
    err = mtxvector_init_array_complex_double(
        &distvector->interior, local_size, data);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistvector_init_array_integer_single()’ allocates and
 * initialises a distributed vector in array format with integer,
 * single precision coefficients.
 */
int mtxdistvector_init_array_integer_single(
    struct mtxdistvector * distvector,
    int num_rows,
    const int32_t * data,
    const struct mtxpartition * partition,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxdistvector_init_comm(distvector, comm, disterr);
    if (err) return err;
    if (partition->num_parts > distvector->comm_size || partition->size != num_rows)
        err = MTX_ERR_INCOMPATIBLE_PARTITION;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    err = mtxpartition_copy(&distvector->partition, partition);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    int64_t local_size = partition->part_sizes[distvector->rank];
    err = mtxvector_init_array_integer_single(
        &distvector->interior, local_size, data);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistvector_init_array_integer_double()’ allocates and
 * initialises a distributed vector in array format with integer,
 * double precision coefficients.
 */
int mtxdistvector_init_array_integer_double(
    struct mtxdistvector * distvector,
    int num_rows,
    const int64_t * data,
    const struct mtxpartition * partition,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxdistvector_init_comm(distvector, comm, disterr);
    if (err) return err;
    if (partition->num_parts > distvector->comm_size || partition->size != num_rows)
        err = MTX_ERR_INCOMPATIBLE_PARTITION;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    err = mtxpartition_copy(&distvector->partition, partition);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    int64_t local_size = partition->part_sizes[distvector->rank];
    err = mtxvector_init_array_integer_double(
        &distvector->interior, local_size, data);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/*
 * Distributed vectors in coordinate format
 */

/**
 * ‘mtxdistvector_alloc_coordinate()’ allocates a distributed vector
 * in coordinate format.
 */
int mtxdistvector_alloc_coordinate(
    struct mtxdistvector * distvector,
    enum mtxfield field,
    enum mtxprecision precision,
    int num_rows,
    int64_t num_nonzeros,
    const struct mtxpartition * partition,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxdistvector_init_comm(distvector, comm, disterr);
    if (err) return err;
    if (partition->num_parts > distvector->comm_size || partition->size != num_rows)
        err = MTX_ERR_INCOMPATIBLE_PARTITION;
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;
    err = mtxpartition_copy(&distvector->partition, partition);
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;
    int64_t local_size = partition->part_sizes[distvector->rank];
    err = mtxvector_alloc_coordinate(
        &distvector->interior, field, precision, local_size, num_nonzeros);
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistvector_init_coordinate_real_single()’ allocates and
 * initialises a distributed vector in coordinate format with real,
 * single precision coefficients.
 */
int mtxdistvector_init_coordinate_real_single(
    struct mtxdistvector * distvector,
    int num_rows,
    int64_t num_nonzeros,
    const int * idx,
    const float * data,
    const struct mtxpartition * partition,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxdistvector_init_comm(distvector, comm, disterr);
    if (err) return err;
    if (partition->num_parts > distvector->comm_size || partition->size != num_rows)
        err = MTX_ERR_INCOMPATIBLE_PARTITION;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    err = mtxpartition_copy(&distvector->partition, partition);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    int64_t local_size = partition->part_sizes[distvector->rank];
    err = mtxvector_init_coordinate_real_single(
        &distvector->interior, local_size, num_nonzeros, idx, data);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistvector_init_coordinate_real_double()’ allocates and
 * initialises a distributed vector in coordinate format with real,
 * double precision coefficients.
 */
int mtxdistvector_init_coordinate_real_double(
    struct mtxdistvector * distvector,
    int num_rows,
    int64_t num_nonzeros,
    const int * idx,
    const double * data,
    const struct mtxpartition * partition,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxdistvector_init_comm(distvector, comm, disterr);
    if (err) return err;
    if (partition->num_parts > distvector->comm_size || partition->size != num_rows)
        err = MTX_ERR_INCOMPATIBLE_PARTITION;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    err = mtxpartition_copy(&distvector->partition, partition);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    int64_t local_size = partition->part_sizes[distvector->rank];
    err = mtxvector_init_coordinate_real_double(
        &distvector->interior, local_size, num_nonzeros, idx, data);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistvector_init_coordinate_complex_single()’ allocates and
 * initialises a distributed vector in coordinate format with complex,
 * single precision coefficients.
 */
int mtxdistvector_init_coordinate_complex_single(
    struct mtxdistvector * distvector,
    int num_rows,
    int64_t num_nonzeros,
    const int * idx,
    const float (* data)[2],
    const struct mtxpartition * partition,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxdistvector_init_comm(distvector, comm, disterr);
    if (err) return err;
    if (partition->num_parts > distvector->comm_size || partition->size != num_rows)
        err = MTX_ERR_INCOMPATIBLE_PARTITION;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    err = mtxpartition_copy(&distvector->partition, partition);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    int64_t local_size = partition->part_sizes[distvector->rank];
    err = mtxvector_init_coordinate_complex_single(
        &distvector->interior, local_size, num_nonzeros, idx, data);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistvector_init_coordinate_complex_double()’ allocates and
 * initialises a distributed vector in coordinate format with complex,
 * double precision coefficients.
 */
int mtxdistvector_init_coordinate_complex_double(
    struct mtxdistvector * distvector,
    int num_rows,
    int64_t num_nonzeros,
    const int * idx,
    const double (* data)[2],
    const struct mtxpartition * partition,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxdistvector_init_comm(distvector, comm, disterr);
    if (err) return err;
    if (partition->num_parts > distvector->comm_size || partition->size != num_rows)
        err = MTX_ERR_INCOMPATIBLE_PARTITION;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    err = mtxpartition_copy(&distvector->partition, partition);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    int64_t local_size = partition->part_sizes[distvector->rank];
    err = mtxvector_init_coordinate_complex_double(
        &distvector->interior, local_size, num_nonzeros, idx, data);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistvector_init_coordinate_integer_single()’ allocates and
 * initialises a distributed vector in coordinate format with integer,
 * single precision coefficients.
 */
int mtxdistvector_init_coordinate_integer_single(
    struct mtxdistvector * distvector,
    int num_rows,
    int64_t num_nonzeros,
    const int * idx,
    const int32_t * data,
    const struct mtxpartition * partition,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxdistvector_init_comm(distvector, comm, disterr);
    if (err) return err;
    if (partition->num_parts > distvector->comm_size || partition->size != num_rows)
        err = MTX_ERR_INCOMPATIBLE_PARTITION;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    err = mtxpartition_copy(&distvector->partition, partition);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    int64_t local_size = partition->part_sizes[distvector->rank];
    err = mtxvector_init_coordinate_integer_single(
        &distvector->interior, local_size, num_nonzeros, idx, data);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistvector_init_coordinate_integer_double()’ allocates and
 * initialises a distributed vector in coordinate format with integer,
 * double precision coefficients.
 */
int mtxdistvector_init_coordinate_integer_double(
    struct mtxdistvector * distvector,
    int num_rows,
    int64_t num_nonzeros,
    const int * idx,
    const int64_t * data,
    const struct mtxpartition * partition,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxdistvector_init_comm(distvector, comm, disterr);
    if (err) return err;
    if (partition->num_parts > distvector->comm_size || partition->size != num_rows)
        err = MTX_ERR_INCOMPATIBLE_PARTITION;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    err = mtxpartition_copy(&distvector->partition, partition);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    int64_t local_size = partition->part_sizes[distvector->rank];
    err = mtxvector_init_coordinate_integer_double(
        &distvector->interior, local_size, num_nonzeros, idx, data);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistvector_init_coordinate_pattern()’ allocates and initialises
 * a distributed vector in coordinate format with boolean
 * coefficients.
 */
int mtxdistvector_init_coordinate_pattern(
    struct mtxdistvector * distvector,
    int num_rows,
    int64_t num_nonzeros,
    const int * idx,
    const struct mtxpartition * partition,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxdistvector_init_comm(distvector, comm, disterr);
    if (err) return err;
    if (partition->num_parts > distvector->comm_size || partition->size != num_rows)
        err = MTX_ERR_INCOMPATIBLE_PARTITION;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    err = mtxpartition_copy(&distvector->partition, partition);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    int64_t local_size = partition->part_sizes[distvector->rank];
    err = mtxvector_init_coordinate_pattern(
        &distvector->interior, local_size, num_nonzeros, idx);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/*
 * Convert to and from Matrix Market format
 */

/**
 * ‘mtxdistvector_from_mtxfile()’ converts a vector in Matrix Market
 * format to a distributed vector.
 */
int mtxdistvector_from_mtxfile(
    struct mtxdistvector * distvector,
    const struct mtxfile * mtxfile,
    enum mtxvectortype vector_type,
    const struct mtxpartition * partition,
    MPI_Comm comm,
    int root,
    struct mtxdisterror * disterr)
{
    int err = mtxdistvector_init_comm(distvector, comm, disterr);
    if (err)
        return err;
    int comm_size = distvector->comm_size;
    int rank = distvector->rank;

    if (rank == root && mtxfile->header.object != mtxfile_vector)
        err = MTX_ERR_INCOMPATIBLE_MTX_OBJECT;
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;

    /* broadcast the number of rows in the Matrix Market file */
    int num_rows = (rank == root) ? mtxfile->size.num_rows : 0;
    disterr->mpierrcode = MPI_Bcast(&num_rows, 1, MPI_INT, root, comm);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;

    if (partition) {
        if (partition->num_parts > comm_size || partition->size != num_rows)
            err = MTX_ERR_INCOMPATIBLE_PARTITION;
        if (mtxdisterror_allreduce(disterr, err))
            return MTX_ERR_MPI_COLLECTIVE;
        err = mtxpartition_copy(&distvector->partition, partition);
        if (mtxdisterror_allreduce(disterr, err))
            return MTX_ERR_MPI_COLLECTIVE;
    } else {
        /* partition vector into equal-sized blocks by default */
        err = mtxpartition_init_block(
            &distvector->partition, num_rows,
            distvector->comm_size, NULL);
        if (mtxdisterror_allreduce(disterr, err))
            return MTX_ERR_MPI_COLLECTIVE;
    }

    /* 1. Partition the vector */
    struct mtxfile * sendmtxfiles = (rank == root) ?
        malloc(distvector->partition.num_parts *
               sizeof(struct mtxfile)) : NULL;
    err = (rank == root && !sendmtxfiles) ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxpartition_free(&distvector->partition);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    if (rank == root) {
        err = mtxfile_partition(
            mtxfile, sendmtxfiles, &distvector->partition, NULL);
    }
    if (mtxdisterror_allreduce(disterr, err)) {
        if (rank == root)
            free(sendmtxfiles);
        mtxpartition_free(&distvector->partition);
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
        mtxpartition_free(&distvector->partition);
        return err;
    }

    if (rank == root) {
        for (int p = 0; p < comm_size; p++)
            mtxfile_free(&sendmtxfiles[p]);
        free(sendmtxfiles);
    }

    /* 3. Let each process create its local part of the vector */
    err = mtxvector_from_mtxfile(
        &distvector->interior, &recvmtxfile, vector_type);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxfile_free(&recvmtxfile);
        mtxpartition_free(&distvector->partition);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    mtxfile_free(&recvmtxfile);
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistvector_to_mtxfile()’ gathers a distributed vector onto a
 * single, root process and converts it to a (non-distributed) Matrix
 * Market file on that process.
 *
 * This function performs collective communication and therefore
 * requires every process in the communicator to perform matching
 * calls to this function.
 */
int mtxdistvector_to_mtxfile(
    struct mtxfile * dst,
    const struct mtxdistvector * src,
    enum mtxfileformat mtxfmt,
    int root,
    struct mtxdisterror * disterr)
{
    int err;
    struct mtxdistfile mtxdistfile;
    err = mtxdistvector_to_mtxdistfile(src, &mtxdistfile, mtxfmt, disterr);
    if (err)
        return err;
    err = mtxdistfile_to_mtxfile(dst, &mtxdistfile, root, disterr);
    if (err) {
        mtxdistfile_free(&mtxdistfile);
        return err;
    }
    mtxdistfile_free(&mtxdistfile);
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistvector_from_mtxdistfile()’ converts a vector in distributed
 * Matrix Market format to a distributed vector.
 */
int mtxdistvector_from_mtxdistfile(
    struct mtxdistvector * distvector,
    const struct mtxdistfile * mtxdistfile,
    enum mtxvectortype vector_type,
    const struct mtxpartition * partition,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err;
    if (mtxdistfile->header.object != mtxfile_vector)
        return MTX_ERR_INCOMPATIBLE_MTX_OBJECT;
    err = mtxdistvector_init_comm(distvector, comm, disterr);
    if (err)
        return err;

    int comm_size = distvector->comm_size;
    int rank = distvector->rank;
    int num_rows = mtxdistfile->size.num_rows;

    if (partition) {
        if (partition->num_parts > comm_size || partition->size != num_rows)
            err = MTX_ERR_INCOMPATIBLE_PARTITION;
        if (mtxdisterror_allreduce(disterr, err))
            return MTX_ERR_MPI_COLLECTIVE;
        err = mtxpartition_copy(&distvector->partition, partition);
        if (mtxdisterror_allreduce(disterr, err))
            return MTX_ERR_MPI_COLLECTIVE;
    } else {
        /* partition vector into equal-sized blocks by default */
        err = mtxpartition_init_block(
            &distvector->partition, num_rows,
            distvector->comm_size, NULL);
        if (mtxdisterror_allreduce(disterr, err))
            return MTX_ERR_MPI_COLLECTIVE;
    }

    /* 1. Partition the vector */
    struct mtxdistfile * dsts =
        malloc(distvector->partition.num_parts * sizeof(struct mtxdistfile));
    err = !dsts ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxpartition_free(&distvector->partition);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    err = mtxdistfile_partition(
        mtxdistfile, dsts,
        &distvector->partition, NULL, disterr);
    if (err) {
        free(dsts);
        mtxpartition_free(&distvector->partition);
        return err;
    }

    for (int p = 0; p < distvector->partition.num_parts; p++) {
        struct mtxfile mtxfile;
        err = mtxdistfile_to_mtxfile(&mtxfile, &dsts[p], p, disterr);
        if (err) {
            for (int q = 0; q < distvector->partition.num_parts; q++)
                mtxdistfile_free(&dsts[q]);
            free(dsts);
            mtxpartition_free(&distvector->partition);
            return err;
        }

        if (rank == p) {
            err = mtxvector_from_mtxfile(
                &distvector->interior, &mtxfile, vector_type);
        }
        if (mtxdisterror_allreduce(disterr, err)) {
            if (rank == p)
                mtxfile_free(&mtxfile);
            for (int q = 0; q < distvector->partition.num_parts; q++)
                mtxdistfile_free(&dsts[q]);
            free(dsts);
            mtxpartition_free(&distvector->partition);
            return MTX_ERR_MPI_COLLECTIVE;
        }

        if (rank == p)
            mtxfile_free(&mtxfile);
    }

    for (int q = 0; q < distvector->partition.num_parts; q++)
        mtxdistfile_free(&dsts[q]);
    free(dsts);
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistvector_to_mtxdistfile()’ converts a distributed vector to a
 * vector in a distributed Matrix Market format.
 */
int mtxdistvector_to_mtxdistfile(
    const struct mtxdistvector * distvector,
    struct mtxdistfile * dst,
    enum mtxfileformat mtxfmt,
    struct mtxdisterror * disterr)
{
    int err;
    MPI_Comm comm = distvector->comm;
    int comm_size = distvector->comm_size;
    int rank = distvector->rank;

    struct mtxfile mtxfile;
    err = mtxvector_to_mtxfile(&distvector->interior, &mtxfile, mtxfmt);
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;

    /* TODO: Map from local numbering of rows on each process to
     * global row numbering? */

    int64_t * part_sizes = malloc(comm_size * sizeof(int64_t));
    err = !part_sizes ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxfile_free(&mtxfile);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    int64_t part_size = mtxfile.header.format == mtxfile_array
        ? mtxfile.size.num_rows : mtxfile.size.num_nonzeros;
    disterr->mpierrcode = MPI_Allgather(
        &part_size, 1, MPI_INT64_T, part_sizes, 1, MPI_INT64_T, comm);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(part_sizes);
        mtxfile_free(&mtxfile);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    int64_t size = 0;
    for (int p = 0; p < comm_size; p++)
        size += part_sizes[p];

    struct mtxpartition partition;
    err = mtxpartition_init_block(
        &partition, size, comm_size, part_sizes);
    if (mtxdisterror_allreduce(disterr, err)) {
        free(part_sizes);
        mtxfile_free(&mtxfile);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    free(part_sizes);

    struct mtxfilesize dstsize;
    if (mtxfile.header.format == mtxfile_array) {
        dstsize.num_rows = size;
        dstsize.num_columns = mtxfile.size.num_columns;
        dstsize.num_nonzeros = mtxfile.size.num_nonzeros;
    } else if (mtxfile.header.format == mtxfile_coordinate) {
        disterr->mpierrcode = MPI_Allreduce(
            MPI_IN_PLACE, &mtxfile.size.num_rows, 1, MPI_INT, MPI_SUM, comm);
        err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
        if (mtxdisterror_allreduce(disterr, err)) {
            free(part_sizes);
            mtxfile_free(&mtxfile);
            return MTX_ERR_MPI_COLLECTIVE;
        }
        dstsize.num_rows = mtxfile.size.num_rows;
        dstsize.num_columns = mtxfile.size.num_columns;
        dstsize.num_nonzeros = size;
    }

    err = mtxdistfile_alloc(
        dst, &mtxfile.header, &mtxfile.comments,
        &dstsize, mtxfile.precision,
        &partition, distvector->comm, disterr);
    if (err) {
        mtxpartition_free(&partition);
        mtxfile_free(&mtxfile);
        return err;
    }
    mtxpartition_free(&partition);

    int64_t num_data_lines;
    err = mtxfilesize_num_data_lines(
        &mtxfile.size, mtxfile.header.symmetry, &num_data_lines);
    if (err) {
        mtxdistfile_free(dst);
        mtxfile_free(&mtxfile);
        return err;
    }
    err = mtxfiledata_copy(
        &dst->data, &mtxfile.data,
        dst->header.object, dst->header.format,
        dst->header.field, dst->precision,
        num_data_lines, 0, 0);
    if (err) {
        mtxdistfile_free(dst);
        mtxfile_free(&mtxfile);
        return err;
    }
    mtxfile_free(&mtxfile);
    return MTX_SUCCESS;
}

/*
 * I/O functions
 */

/**
 * ‘mtxdistvector_read()’ reads a vector from a Matrix Market file.
 * The file may optionally be compressed by gzip.
 *
 * The ‘precision’ argument specifies which precision to use for
 * storing matrix or vector values.
 *
 * If ‘path’ is ‘-’, then standard input is used.
 *
 * If an error code is returned, then ‘lines_read’ and ‘bytes_read’
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the vector.
 */
int mtxdistvector_read(
    struct mtxdistvector * distvector,
    enum mtxprecision precision,
    const char * path,
    bool gzip,
    int * lines_read,
    int64_t * bytes_read);

/**
 * ‘mtxdistvector_fread()’ reads a vector from a stream in Matrix
 * Market format.
 *
 * ‘precision’ is used to determine the precision to use for storing
 * the values of matrix or vector entries.
 *
 * If an error code is returned, then ‘lines_read’ and ‘bytes_read’
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the vector.
 */
int mtxdistvector_fread(
    struct mtxdistvector * distvector,
    enum mtxprecision precision,
    FILE * f,
    int * lines_read,
    int64_t * bytes_read,
    size_t line_max,
    char * linebuf);

#ifdef LIBMTX_HAVE_LIBZ
/**
 * ‘mtxdistvector_gzread()’ reads a vector from a gzip-compressed
 * stream.
 *
 * ‘precision’ is used to determine the precision to use for storing
 * the values of matrix or vector entries.
 *
 * If an error code is returned, then ‘lines_read’ and ‘bytes_read’
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the vector.
 */
int mtxdistvector_gzread(
    struct mtxdistvector * distvector,
    enum mtxprecision precision,
    gzFile f,
    int * lines_read,
    int64_t * bytes_read,
    size_t line_max,
    char * linebuf);
#endif

/**
 * ‘mtxdistvector_write()’ writes a vector to a Matrix Market
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
int mtxdistvector_write(
    const struct mtxdistvector * distvector,
    enum mtxfileformat mtxfmt,
    const char * path,
    bool gzip,
    const char * fmt,
    int64_t * bytes_written);

/**
 * ‘mtxdistvector_fwrite()’ writes a vector to a stream.
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
int mtxdistvector_fwrite(
    const struct mtxdistvector * distvector,
    enum mtxfileformat mtxfmt,
    FILE * f,
    const char * fmt,
    int64_t * bytes_written);

/**
 * `mtxdistvector_fwrite_shared()' writes a distributed vector as a
 * Matrix Market file to a single stream that is shared by every
 * process in the communicator.
 *
 * If ‘fmt’ is ‘NULL’, then the format specifier ‘%g’ is used to print
 * floating point numbers with enough digits to ensure correct
 * round-trip conversion from decimal text and back.  Otherwise, the
 * given format string is used to print numerical values.
 *
 * The format string follows the conventions of `printf'. If the field
 * is `real', `double' or `complex', then the format specifiers '%e',
 * '%E', '%f', '%F', '%g' or '%G' may be used. If the field is
 * `integer', then the format specifier must be '%d'. The format
 * string is ignored if the field is `pattern'. Field width and
 * precision may be specified (e.g., "%3.1f"), but variable field
 * width and precision (e.g., "%*.*f"), as well as length modifiers
 * (e.g., "%Lf") are not allowed.
 *
 * If it is not `NULL', then the number of bytes written to the stream
 * is returned in `bytes_written'.
 *
 * Note that only the specified ‘root’ process will print anything to
 * the stream. Other processes will therefore send their part of the
 * distributed Matrix Market file to the root process for printing.
 *
 * This function performs collective communication and therefore
 * requires every process in the communicator to perform matching
 * calls to the function.
 */
int mtxdistvector_fwrite_shared(
    const struct mtxdistvector * mtxdistvector,
    enum mtxfileformat mtxfmt,
    FILE * f,
    const char * fmt,
    int64_t * bytes_written,
    int root,
    struct mtxdisterror * disterr)
{
    int err;
    struct mtxdistfile mtxdistfile;
    err = mtxdistvector_to_mtxdistfile(
        mtxdistvector, &mtxdistfile, mtxfmt, disterr);
    if (err)
        return err;

    err = mtxdistfile_fwrite_shared(
        &mtxdistfile, f, fmt, bytes_written, root, disterr);
    if (err) {
        mtxdistfile_free(&mtxdistfile);
        return err;
    }
    mtxdistfile_free(&mtxdistfile);
    return MTX_SUCCESS;
}

#ifdef LIBMTX_HAVE_LIBZ
/**
 * ‘mtxdistvector_gzwrite()’ writes a vector to a gzip-compressed
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
int mtxdistvector_gzwrite(
    const struct mtxdistvector * distvector,
    enum mtxfileformat mtxfmt,
    gzFile f,
    const char * fmt,
    int64_t * bytes_written);
#endif

/*
 * Level 1 BLAS operations
 */

/**
 * `mtxdistvector_swap()' swaps values of two vectors, simultaneously
 * performing ‘y <- x’ and ‘x <- y’.
 */
int mtxdistvector_swap(
    struct mtxdistvector * x,
    struct mtxdistvector * y,
    struct mtxdisterror * disterr);

/**
 * `mtxdistvector_copy()' copies values of a vector, ‘y = x’.
 */
int mtxdistvector_copy(
    struct mtxdistvector * y,
    const struct mtxdistvector * x,
    struct mtxdisterror * disterr);

/**
 * `mtxdistvector_sscal()' scales a vector by a single precision
 * floating point scalar, ‘x = a*x’.
 */
int mtxdistvector_sscal(
    float a,
    struct mtxdistvector * x,
    int64_t * num_flops,
    struct mtxdisterror * disterr)
{
    int err;
    err = mtxvector_sscal(a, &x->interior, num_flops);
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * `mtxdistvector_dscal()' scales a vector by a double precision
 * floating point scalar, ‘x = a*x’.
 */
int mtxdistvector_dscal(
    double a,
    struct mtxdistvector * x,
    int64_t * num_flops,
    struct mtxdisterror * disterr)
{
    int err;
    err = mtxvector_dscal(a, &x->interior, num_flops);
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * `mtxdistvector_saxpy()' adds a vector to another vector multiplied
 * by a single precision floating point value, ‘y = a*x + y’.
 */
int mtxdistvector_saxpy(
    float a,
    const struct mtxdistvector * x,
    struct mtxdistvector * y,
    int64_t * num_flops,
    struct mtxdisterror * disterr)
{
    int err;
    err = mtxvector_saxpy(a, &x->interior, &y->interior, num_flops);
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * `mtxdistvector_daxpy()' adds a vector to another vector multiplied
 * by a double precision floating point value, ‘y = a*x + y’.
 */
int mtxdistvector_daxpy(
    double a,
    const struct mtxdistvector * x,
    struct mtxdistvector * y,
    int64_t * num_flops,
    struct mtxdisterror * disterr)
{
    int err;
    err = mtxvector_daxpy(a, &x->interior, &y->interior, num_flops);
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * `mtxdistvector_saypx()' multiplies a vector by a single precision
 * floating point scalar and adds another vector, ‘y = a*y + x’.
 */
int mtxdistvector_saypx(
    float a,
    struct mtxdistvector * y,
    const struct mtxdistvector * x,
    int64_t * num_flops,
    struct mtxdisterror * disterr)
{
    int err;
    err = mtxvector_saypx(a, &y->interior, &x->interior, num_flops);
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * `mtxdistvector_daypx()' multiplies a vector by a double precision
 * floating point scalar and adds another vector, ‘y = a*y + x’.
 */
int mtxdistvector_daypx(
    double a,
    struct mtxdistvector * y,
    const struct mtxdistvector * x,
    int64_t * num_flops,
    struct mtxdisterror * disterr)
{
    int err;
    err = mtxvector_daypx(a, &y->interior, &x->interior, num_flops);
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * `mtxdistvector_sdot()' computes the Euclidean dot product of two
 * vectors in single precision floating point.
 */
int mtxdistvector_sdot(
    const struct mtxdistvector * x,
    const struct mtxdistvector * y,
    float * dot,
    int64_t * num_flops,
    struct mtxdisterror * disterr)
{
    int err;
    float dotp;
    err = mtxvector_sdot(&x->interior, &y->interior, &dotp, num_flops);
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;
    disterr->mpierrcode = MPI_Allreduce(
        &dotp, dot, 1, MPI_FLOAT, MPI_SUM, x->comm);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * `mtxdistvector_ddot()' computes the Euclidean dot product of two
 * vectors in double precision floating point.
 */
int mtxdistvector_ddot(
    const struct mtxdistvector * x,
    const struct mtxdistvector * y,
    double * dot,
    int64_t * num_flops,
    struct mtxdisterror * disterr)
{
    int err;
    double dotp;
    err = mtxvector_ddot(&x->interior, &y->interior, &dotp, num_flops);
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;
    disterr->mpierrcode = MPI_Allreduce(
        &dotp, dot, 1, MPI_DOUBLE, MPI_SUM, x->comm);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * `mtxdistvector_cdotu()' computes the product of the transpose of a
 * complex row vector with another complex row vector in single
 * precision floating point, ‘dot := x^T*y’.
 */
int mtxdistvector_cdotu(
    const struct mtxdistvector * x,
    const struct mtxdistvector * y,
    float (* dot)[2],
    int64_t * num_flops,
    struct mtxdisterror * disterr)
{
    int err;
    float dotp[2];
    err = mtxvector_cdotu(&x->interior, &y->interior, &dotp, num_flops);
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;
    disterr->mpierrcode = MPI_Allreduce(
        dotp, *dot, 2, MPI_FLOAT, MPI_SUM, x->comm);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * `mtxdistvector_zdotu()' computes the product of the transpose of a
 * complex row vector with another complex row vector in double
 * precision floating point, ‘dot := x^T*y’.
 */
int mtxdistvector_zdotu(
    const struct mtxdistvector * x,
    const struct mtxdistvector * y,
    double (* dot)[2],
    int64_t * num_flops,
    struct mtxdisterror * disterr)
{
    int err;
    double dotp[2];
    err = mtxvector_zdotu(&x->interior, &y->interior, &dotp, num_flops);
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;
    disterr->mpierrcode = MPI_Allreduce(
        dotp, *dot, 2, MPI_DOUBLE, MPI_SUM, x->comm);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * `mtxdistvector_cdotc()' computes the Euclidean dot product of two
 * complex vectors in single precision floating point, ‘dot := x^H*y’.
 */
int mtxdistvector_cdotc(
    const struct mtxdistvector * x,
    const struct mtxdistvector * y,
    float (* dot)[2],
    int64_t * num_flops,
    struct mtxdisterror * disterr)
{
    int err;
    float dotp[2];
    err = mtxvector_cdotc(&x->interior, &y->interior, &dotp, num_flops);
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;
    disterr->mpierrcode = MPI_Allreduce(
        dotp, *dot, 2, MPI_FLOAT, MPI_SUM, x->comm);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * `mtxdistvector_zdotc()' computes the Euclidean dot product of two
 * complex vectors in double precision floating point, ‘dot := x^H*y’.
 */
int mtxdistvector_zdotc(
    const struct mtxdistvector * x,
    const struct mtxdistvector * y,
    double (* dot)[2],
    int64_t * num_flops,
    struct mtxdisterror * disterr)
{
    int err;
    double dotp[2];
    err = mtxvector_zdotc(&x->interior, &y->interior, &dotp, num_flops);
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;
    disterr->mpierrcode = MPI_Allreduce(
        dotp, *dot, 2, MPI_DOUBLE, MPI_SUM, x->comm);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * `mtxdistvector_snrm2()' computes the Euclidean norm of a vector in
 * single precision floating point.
 */
int mtxdistvector_snrm2(
    const struct mtxdistvector * x,
    float * nrm2,
    int64_t * num_flops,
    struct mtxdisterror * disterr)
{
    int err;
    float dot[2];
    err = mtxvector_cdotc(&x->interior, &x->interior, &dot, num_flops);
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;
    disterr->mpierrcode = MPI_Allreduce(
        &dot[0], nrm2, 1, MPI_FLOAT, MPI_SUM, x->comm);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;
    *nrm2 = sqrtf(*nrm2);
    return MTX_SUCCESS;
}

/**
 * `mtxdistvector_dnrm2()' computes the Euclidean norm of a vector in
 * double precision floating point.
 */
int mtxdistvector_dnrm2(
    const struct mtxdistvector * x,
    double * nrm2,
    int64_t * num_flops,
    struct mtxdisterror * disterr)
{
    double dot[2];
    int err = mtxvector_zdotc(&x->interior, &x->interior, &dot, num_flops);
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;
    disterr->mpierrcode = MPI_Allreduce(
        &dot[0], nrm2, 1, MPI_DOUBLE, MPI_SUM, x->comm);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;
    *nrm2 = sqrt(*nrm2);
    return MTX_SUCCESS;
}

/**
 * `mtxdistvector_sasum()' computes the sum of absolute values
 * (1-norm) of a vector in single precision floating point.
 */
int mtxdistvector_sasum(
    const struct mtxdistvector * x,
    float * asum,
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * `mtxdistvector_dasum()' computes the sum of absolute values
 * (1-norm) of a vector in double precision floating point.
 */
int mtxdistvector_dasum(
    const struct mtxdistvector * x,
    double * asum,
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * `mtxdistvector_imax()' finds the index of the first element having
 * the maximum absolute value.
 */
int mtxdistvector_imax(
    const struct mtxdistvector * x,
    int * max,
    struct mtxdisterror * disterr);
#endif
