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
 * Last modified: 2022-01-26
 *
 * Data structures for distributed vectors.
 */

#include <libmtx/libmtx-config.h>

#ifdef LIBMTX_HAVE_MPI
#include <libmtx/error.h>
#include <libmtx/field.h>
#include <libmtx/mtxfile/header.h>
#include <libmtx/mtxfile/mtxdistfile.h>
#include <libmtx/mtxfile/mtxfile.h>
#include <libmtx/mtxfile/size.h>
#include <libmtx/precision.h>
#include <libmtx/util/partition.h>
#include <libmtx/util/permute.h>
#include <libmtx/util/sort.h>
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
    mtxpartition_free(&distvector->rowpart);
    mtxvector_free(&distvector->interior);
}

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

static int mtxdistvector_init_partitions(
    struct mtxdistvector * distvector,
    int num_local_rows,
    const struct mtxpartition * rowpart,
    struct mtxdisterror * disterr)
{
    int err = MTX_SUCCESS;
    MPI_Comm comm = distvector->comm;
    int comm_size = distvector->comm_size;
    int rank = distvector->rank;
    int64_t num_rows = num_local_rows;
    disterr->mpierrcode = MPI_Allreduce(
        MPI_IN_PLACE, &num_rows, 1, MPI_INT64_T, MPI_SUM, comm);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;

    if (rowpart) {
        if (rowpart->num_parts > comm_size ||
            rowpart->size != num_rows ||
            (rank < rowpart->num_parts &&
             rowpart->part_sizes[rank] != num_local_rows))
            err = MTX_ERR_INCOMPATIBLE_PARTITION;
        if (mtxdisterror_allreduce(disterr, err))
            return MTX_ERR_MPI_COLLECTIVE;
        err = mtxpartition_copy(&distvector->rowpart, rowpart);
        if (mtxdisterror_allreduce(disterr, err))
            return MTX_ERR_MPI_COLLECTIVE;
    } else {
        /* use a block partitioning by default */
        int64_t * partsizes = malloc(comm_size * sizeof(int64_t));
        err = !partsizes ? MTX_ERR_ERRNO : MTX_SUCCESS;
        if (mtxdisterror_allreduce(disterr, err))
            return MTX_ERR_MPI_COLLECTIVE;
        partsizes[rank] = num_local_rows;
        disterr->mpierrcode = MPI_Allgather(
            MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
            partsizes, 1, MPI_INT64_T, comm);
        err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
        if (mtxdisterror_allreduce(disterr, err)) {
            free(partsizes);
            return MTX_ERR_MPI_COLLECTIVE;
        }
        err = mtxpartition_init_block(
            &distvector->rowpart, num_rows, comm_size, partsizes);
        if (mtxdisterror_allreduce(disterr, err)) {
            free(partsizes);
            return MTX_ERR_MPI_COLLECTIVE;
        }
        free(partsizes);
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistvector_alloc_copy()’ allocates storage for a copy of a
 * distributed vector without initialising the underlying values.
 */
int mtxdistvector_alloc_copy(
    struct mtxdistvector * dst,
    const struct mtxdistvector * src,
    struct mtxdisterror * disterr)
{
    int err = mtxdistvector_init_comm(dst, src->comm, disterr);
    if (err) return err;
    err = mtxpartition_copy(&dst->rowpart, &src->rowpart);
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;
    err = mtxvector_alloc_copy(&dst->interior, &src->interior);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxpartition_free(&dst->rowpart);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistvector_init_copy()’ creates a copy of a distributed vector.
 */
int mtxdistvector_init_copy(
    struct mtxdistvector * dst,
    const struct mtxdistvector * src,
    struct mtxdisterror * disterr)
{
    int err = mtxdistvector_init_comm(dst, src->comm, disterr);
    if (err) return err;
    err = mtxpartition_copy(&dst->rowpart, &src->rowpart);
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;
    err = mtxvector_init_copy(&dst->interior, &src->interior);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxpartition_free(&dst->rowpart);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    return MTX_SUCCESS;
}

/*
 * Distributed vectors in array format
 */

/**
 * ‘mtxdistvector_alloc_array()’ allocates a distributed vector in
 * array format.
 */
int mtxdistvector_alloc_array(
    struct mtxdistvector * distvector,
    enum mtxfield field,
    enum mtxprecision precision,
    int num_local_rows,
    const struct mtxpartition * rowpart,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxdistvector_init_comm(distvector, comm, disterr);
    if (err) return err;
    err = mtxdistvector_init_partitions(distvector, num_local_rows, rowpart, disterr);
    if (err) return err;
    err = mtxvector_alloc_array(
        &distvector->interior, field, precision, num_local_rows);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxpartition_free(&distvector->rowpart);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistvector_init_array_real_single()’ allocates and initialises
 * a distributed vector in array format with real, single precision
 * coefficients.
 */
int mtxdistvector_init_array_real_single(
    struct mtxdistvector * distvector,
    int num_local_rows,
    const float * data,
    const struct mtxpartition * rowpart,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxdistvector_init_comm(distvector, comm, disterr);
    if (err) return err;
    err = mtxdistvector_init_partitions(distvector, num_local_rows, rowpart, disterr);
    if (err) return err;
    err = mtxvector_init_array_real_single(
        &distvector->interior, num_local_rows, data);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxpartition_free(&distvector->rowpart);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistvector_init_array_real_double()’ allocates and initialises
 * a distributed vector in array format with real, double precision
 * coefficients.
 */
int mtxdistvector_init_array_real_double(
    struct mtxdistvector * distvector,
    int num_local_rows,
    const double * data,
    const struct mtxpartition * rowpart,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxdistvector_init_comm(distvector, comm, disterr);
    if (err) return err;
    err = mtxdistvector_init_partitions(distvector, num_local_rows, rowpart, disterr);
    if (err) return err;
    err = mtxvector_init_array_real_double(
        &distvector->interior, num_local_rows, data);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxpartition_free(&distvector->rowpart);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistvector_init_array_complex_single()’ allocates and
 * initialises a distributed vector in array format with complex,
 * single precision coefficients.
 */
int mtxdistvector_init_array_complex_single(
    struct mtxdistvector * distvector,
    int num_local_rows,
    const float (* data)[2],
    const struct mtxpartition * rowpart,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxdistvector_init_comm(distvector, comm, disterr);
    if (err) return err;
    err = mtxdistvector_init_partitions(distvector, num_local_rows, rowpart, disterr);
    if (err) return err;
    err = mtxvector_init_array_complex_single(
        &distvector->interior, num_local_rows, data);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxpartition_free(&distvector->rowpart);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistvector_init_array_complex_double()’ allocates and
 * initialises a distributed vector in array format with complex,
 * double precision coefficients.
 */
int mtxdistvector_init_array_complex_double(
    struct mtxdistvector * distvector,
    int num_local_rows,
    const double (* data)[2],
    const struct mtxpartition * rowpart,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxdistvector_init_comm(distvector, comm, disterr);
    if (err) return err;
    err = mtxdistvector_init_partitions(distvector, num_local_rows, rowpart, disterr);
    if (err) return err;
    err = mtxvector_init_array_complex_double(
        &distvector->interior, num_local_rows, data);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxpartition_free(&distvector->rowpart);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistvector_init_array_integer_single()’ allocates and
 * initialises a distributed vector in array format with integer,
 * single precision coefficients.
 */
int mtxdistvector_init_array_integer_single(
    struct mtxdistvector * distvector,
    int num_local_rows,
    const int32_t * data,
    const struct mtxpartition * rowpart,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxdistvector_init_comm(distvector, comm, disterr);
    if (err) return err;
    err = mtxdistvector_init_partitions(distvector, num_local_rows, rowpart, disterr);
    if (err) return err;
    err = mtxvector_init_array_integer_single(
        &distvector->interior, num_local_rows, data);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxpartition_free(&distvector->rowpart);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistvector_init_array_integer_double()’ allocates and
 * initialises a distributed vector in array format with integer,
 * double precision coefficients.
 */
int mtxdistvector_init_array_integer_double(
    struct mtxdistvector * distvector,
    int num_local_rows,
    const int64_t * data,
    const struct mtxpartition * rowpart,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxdistvector_init_comm(distvector, comm, disterr);
    if (err) return err;
    err = mtxdistvector_init_partitions(distvector, num_local_rows, rowpart, disterr);
    if (err) return err;
    err = mtxvector_init_array_integer_double(
        &distvector->interior, num_local_rows, data);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxpartition_free(&distvector->rowpart);
        return MTX_ERR_MPI_COLLECTIVE;
    }
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
    int num_local_rows,
    int64_t num_local_nonzeros,
    const struct mtxpartition * rowpart,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxdistvector_init_comm(distvector, comm, disterr);
    if (err) return err;
    err = mtxdistvector_init_partitions(distvector, num_local_rows, rowpart, disterr);
    if (err) return err;
    err = mtxvector_alloc_coordinate(
        &distvector->interior, field, precision, num_local_rows, num_local_nonzeros);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxpartition_free(&distvector->rowpart);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistvector_init_coordinate_real_single()’ allocates and
 * initialises a distributed vector in coordinate format with real,
 * single precision coefficients.
 */
int mtxdistvector_init_coordinate_real_single(
    struct mtxdistvector * distvector,
    int num_local_rows,
    int64_t num_local_nonzeros,
    const int * idx,
    const float * data,
    const struct mtxpartition * rowpart,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxdistvector_init_comm(distvector, comm, disterr);
    if (err) return err;
    err = mtxdistvector_init_partitions(distvector, num_local_rows, rowpart, disterr);
    if (err) return err;
    err = mtxvector_init_coordinate_real_single(
        &distvector->interior, num_local_rows, num_local_nonzeros, idx, data);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxpartition_free(&distvector->rowpart);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistvector_init_coordinate_real_double()’ allocates and
 * initialises a distributed vector in coordinate format with real,
 * double precision coefficients.
 */
int mtxdistvector_init_coordinate_real_double(
    struct mtxdistvector * distvector,
    int num_local_rows,
    int64_t num_local_nonzeros,
    const int * idx,
    const double * data,
    const struct mtxpartition * rowpart,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxdistvector_init_comm(distvector, comm, disterr);
    if (err) return err;
    err = mtxdistvector_init_partitions(distvector, num_local_rows, rowpart, disterr);
    if (err) return err;
    err = mtxvector_init_coordinate_real_double(
        &distvector->interior, num_local_rows, num_local_nonzeros, idx, data);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxpartition_free(&distvector->rowpart);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistvector_init_coordinate_complex_single()’ allocates and
 * initialises a distributed vector in coordinate format with complex,
 * single precision coefficients.
 */
int mtxdistvector_init_coordinate_complex_single(
    struct mtxdistvector * distvector,
    int num_local_rows,
    int64_t num_local_nonzeros,
    const int * idx,
    const float (* data)[2],
    const struct mtxpartition * rowpart,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxdistvector_init_comm(distvector, comm, disterr);
    if (err) return err;
    err = mtxdistvector_init_partitions(distvector, num_local_rows, rowpart, disterr);
    if (err) return err;
    err = mtxvector_init_coordinate_complex_single(
        &distvector->interior, num_local_rows, num_local_nonzeros, idx, data);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxpartition_free(&distvector->rowpart);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistvector_init_coordinate_complex_double()’ allocates and
 * initialises a distributed vector in coordinate format with complex,
 * double precision coefficients.
 */
int mtxdistvector_init_coordinate_complex_double(
    struct mtxdistvector * distvector,
    int num_local_rows,
    int64_t num_local_nonzeros,
    const int * idx,
    const double (* data)[2],
    const struct mtxpartition * rowpart,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxdistvector_init_comm(distvector, comm, disterr);
    if (err) return err;
    err = mtxdistvector_init_partitions(distvector, num_local_rows, rowpart, disterr);
    if (err) return err;
    err = mtxvector_init_coordinate_complex_double(
        &distvector->interior, num_local_rows, num_local_nonzeros, idx, data);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxpartition_free(&distvector->rowpart);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistvector_init_coordinate_integer_single()’ allocates and
 * initialises a distributed vector in coordinate format with integer,
 * single precision coefficients.
 */
int mtxdistvector_init_coordinate_integer_single(
    struct mtxdistvector * distvector,
    int num_local_rows,
    int64_t num_local_nonzeros,
    const int * idx,
    const int32_t * data,
    const struct mtxpartition * rowpart,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxdistvector_init_comm(distvector, comm, disterr);
    if (err) return err;
    err = mtxdistvector_init_partitions(distvector, num_local_rows, rowpart, disterr);
    if (err) return err;
    err = mtxvector_init_coordinate_integer_single(
        &distvector->interior, num_local_rows, num_local_nonzeros, idx, data);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxpartition_free(&distvector->rowpart);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistvector_init_coordinate_integer_double()’ allocates and
 * initialises a distributed vector in coordinate format with integer,
 * double precision coefficients.
 */
int mtxdistvector_init_coordinate_integer_double(
    struct mtxdistvector * distvector,
    int num_local_rows,
    int64_t num_local_nonzeros,
    const int * idx,
    const int64_t * data,
    const struct mtxpartition * rowpart,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxdistvector_init_comm(distvector, comm, disterr);
    if (err) return err;
    err = mtxdistvector_init_partitions(distvector, num_local_rows, rowpart, disterr);
    if (err) return err;
    err = mtxvector_init_coordinate_integer_double(
        &distvector->interior, num_local_rows, num_local_nonzeros, idx, data);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxpartition_free(&distvector->rowpart);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistvector_init_coordinate_pattern()’ allocates and initialises
 * a distributed vector in coordinate format with boolean
 * coefficients.
 */
int mtxdistvector_init_coordinate_pattern(
    struct mtxdistvector * distvector,
    int num_local_rows,
    int64_t num_local_nonzeros,
    const int * idx,
    const struct mtxpartition * rowpart,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxdistvector_init_comm(distvector, comm, disterr);
    if (err) return err;
    err = mtxdistvector_init_partitions(distvector, num_local_rows, rowpart, disterr);
    if (err) return err;
    err = mtxvector_init_coordinate_pattern(
        &distvector->interior, num_local_rows, num_local_nonzeros, idx);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxpartition_free(&distvector->rowpart);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    return MTX_SUCCESS;
}

/*
 * Modifying values
 */

/**
 * ‘mtxdistvector_set_constant_real_single()’ sets every (nonzero)
 * value of a vector equal to a constant, single precision floating
 * point number.
 */
int mtxdistvector_set_constant_real_single(
    struct mtxdistvector * mtxdistvector,
    float a, struct mtxdisterror * disterr)
{
    int err = mtxvector_set_constant_real_single(
        &mtxdistvector->interior, a);
    return mtxdisterror_allreduce(disterr, err);
}

/**
 * ‘mtxdistvector_set_constant_real_double()’ sets every (nonzero)
 * value of a vector equal to a constant, double precision floating
 * point number.
 */
int mtxdistvector_set_constant_real_double(
    struct mtxdistvector * mtxdistvector,
    double a, struct mtxdisterror * disterr)
{
    int err = mtxvector_set_constant_real_double(
        &mtxdistvector->interior, a);
    return mtxdisterror_allreduce(disterr, err);
}

/**
 * ‘mtxdistvector_set_constant_complex_single()’ sets every (nonzero)
 * value of a vector equal to a constant, single precision floating
 * point complex number.
 */
int mtxdistvector_set_constant_complex_single(
    struct mtxdistvector * mtxdistvector,
    float a[2], struct mtxdisterror * disterr)
{
    int err = mtxvector_set_constant_complex_single(
        &mtxdistvector->interior, a);
    return mtxdisterror_allreduce(disterr, err);
}

/**
 * ‘mtxdistvector_set_constant_complex_double()’ sets every (nonzero)
 * value of a vector equal to a constant, double precision floating
 * point complex number.
 */
int mtxdistvector_set_constant_complex_double(
    struct mtxdistvector * mtxdistvector,
    double a[2], struct mtxdisterror * disterr)
{
    int err = mtxvector_set_constant_complex_double(
        &mtxdistvector->interior, a);
    return mtxdisterror_allreduce(disterr, err);
}

/**
 * ‘mtxdistvector_set_constant_integer_single()’ sets every (nonzero)
 * value of a vector equal to a constant integer.
 */
int mtxdistvector_set_constant_integer_single(
    struct mtxdistvector * mtxdistvector,
    int32_t a, struct mtxdisterror * disterr)
{
    int err = mtxvector_set_constant_integer_single(
        &mtxdistvector->interior, a);
    return mtxdisterror_allreduce(disterr, err);
}

/**
 * ‘mtxdistvector_set_constant_integer_double()’ sets every (nonzero)
 * value of a vector equal to a constant integer.
 */
int mtxdistvector_set_constant_integer_double(
    struct mtxdistvector * mtxdistvector,
    int64_t a, struct mtxdisterror * disterr)
{
    int err = mtxvector_set_constant_integer_double(
        &mtxdistvector->interior, a);
    return mtxdisterror_allreduce(disterr, err);
}

/*
 * Convert to and from Matrix Market format
 */

/**
 * ‘mtxdistvector_from_mtxfile()’ converts a vector in Matrix Market
 * format to a distributed vector.
 *
 * The ‘type’ argument may be used to specify a desired storage format
 * or implementation for the underlying ‘mtxvector’ on each
 * process. If ‘type’ is ‘mtxvector_auto’, then the type of
 * ‘mtxvector’ is chosen to match the type of ‘mtxfile’. That is,
 * ‘mtxvector_array’ is used if ‘mtxfile’ is in array format, and
 * ‘mtxvector_coordinate’ is used if ‘mtxfile’ is in coordinate
 * format.
 *
 * Furthermore, ‘rowpart’ must be a partitioning of the rows of the
 * global vector. Therefore, ‘rowpart->size’ must be equal to the
 * number of rows in the underlying vector represented by
 * ‘mtxfile’. The partition must consist of at most one part for each
 * MPI process in the communicator ‘comm’. If ‘rowpart’ is ‘NULL’,
 * then the rows are partitioned into contiguous blocks of equal size
 * by default.
 */
int mtxdistvector_from_mtxfile(
    struct mtxdistvector * distvector,
    const struct mtxfile * mtxfile,
    enum mtxvectortype vector_type,
    const struct mtxpartition * rowpart,
    MPI_Comm comm,
    int root,
    struct mtxdisterror * disterr)
{
    int err = mtxdistvector_init_comm(distvector, comm, disterr);
    if (err) return err;
    int comm_size = distvector->comm_size;
    int rank = distvector->rank;

    if (rank == root && mtxfile->header.object != mtxfile_vector)
        err = MTX_ERR_INCOMPATIBLE_MTX_OBJECT;
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;

    int num_local_rows;
    if (rowpart) {
        num_local_rows = rank < rowpart->num_parts ? rowpart->part_sizes[rank] : 0;
    } else {
        /* broadcast the number of rows in the Matrix Market file */
        int64_t num_rows = (rank == root) ? mtxfile->size.num_rows : 0;
        disterr->mpierrcode = MPI_Bcast(&num_rows, 1, MPI_INT64_T, root, comm);
        err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
        if (mtxdisterror_allreduce(disterr, err))
            return MTX_ERR_MPI_COLLECTIVE;

        /* divide rows into equal-sized blocks */
        num_local_rows = num_rows / comm_size
            + (rank < (num_rows % comm_size) ? 1 : 0);
    }
    err = mtxdistvector_init_partitions(
        distvector, num_local_rows, rowpart, disterr);
    if (err) return err;

    /* 1. Partition the vector */
    struct mtxfile * sendmtxfiles = (rank == root) ?
        malloc(distvector->rowpart.num_parts *
               sizeof(struct mtxfile)) : NULL;
    err = (rank == root && !sendmtxfiles) ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxpartition_free(&distvector->rowpart);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    if (rank == root) {
        err = mtxfile_partition(
            sendmtxfiles, mtxfile, &distvector->rowpart, NULL);
    }
    if (mtxdisterror_allreduce(disterr, err)) {
        if (rank == root)
            free(sendmtxfiles);
        mtxpartition_free(&distvector->rowpart);
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
        mtxpartition_free(&distvector->rowpart);
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
        mtxpartition_free(&distvector->rowpart);
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
    MPI_Comm comm = src->comm;
    int comm_size = src->comm_size;
    int rank = src->rank;

    /* 1. Each process converts its part of the vector to Matrix
     * Market format */
    struct mtxfile sendmtxfile;
    err = mtxvector_to_mtxfile(
        &sendmtxfile, &src->interior, 0, NULL, mtxfmt);
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
        ? mtxfile_join(dst, recvmtxfiles, &src->rowpart, NULL)
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
 * ‘mtxdistvector_from_mtxdistfile()’ converts a vector in distributed
 * Matrix Market format to a distributed vector.
 */
int mtxdistvector_from_mtxdistfile(
    struct mtxdistvector * dst,
    const struct mtxdistfile * src,
    enum mtxvectortype vector_type,
    const struct mtxpartition * rowpart,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxdistvector_init_comm(
        dst, comm, disterr);
    if (err) return err;
    int comm_size = dst->comm_size;
    int rank = dst->rank;

    if (src->header.object != mtxfile_vector)
        err = MTX_ERR_INCOMPATIBLE_MTX_OBJECT;
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;

    int num_local_rows;
    if (rowpart) {
        num_local_rows = rank < rowpart->num_parts ? rowpart->part_sizes[rank] : 0;
    } else {
        /* divide rows into equal-sized blocks */
        int num_rows = src->size.num_rows;
        num_local_rows = num_rows / comm_size
            + (rank < (num_rows % comm_size) ? 1 : 0);
    }
    err = mtxdistvector_init_partitions(dst, num_local_rows, rowpart, disterr);
    if (err) return err;

    /* 1. Partition the vector */
    int num_parts = dst->rowpart.num_parts;
    struct mtxdistfile * dsts =
        malloc(num_parts * sizeof(struct mtxdistfile));
    err = !dsts ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxpartition_free(&dst->rowpart);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    err = mtxdistfile_partition(
        dsts, src, &dst->rowpart, NULL, disterr);
    if (err) {
        free(dsts);
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
            mtxpartition_free(&dst->rowpart);
            return err;
        }

        err = rank == p
            ? mtxvector_from_mtxfile(&dst->interior, &mtxfile, vector_type)
            : MTX_SUCCESS;
        if (mtxdisterror_allreduce(disterr, err)) {
            if (rank == p) mtxfile_free(&mtxfile);
            for (int q = 0; q < num_parts; q++)
                mtxdistfile_free(&dsts[q]);
            free(dsts);
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
 * ‘mtxdistvector_to_mtxdistfile()’ converts a distributed vector to a
 * vector in a distributed Matrix Market format.
 */
int mtxdistvector_to_mtxdistfile(
    struct mtxdistfile * dst,
    const struct mtxdistvector * src,
    enum mtxfileformat mtxfmt,
    struct mtxdisterror * disterr)
{
    int err;
    MPI_Comm comm = src->comm;
    int comm_size = src->comm_size;
    int rank = src->rank;
    int num_parts = src->rowpart.num_parts;

    /* 1. Each process converts its part of the vector to Matrix
     * Market format */
    struct mtxfile mtxfile;
    err = (rank < num_parts)
        ? mtxvector_to_mtxfile(&mtxfile, &src->interior, 0, NULL, mtxfmt)
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
        dst, srcs, &src->rowpart, NULL, disterr);
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
    enum mtxvectortype type,
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
    enum mtxvectortype type,
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
    enum mtxvectortype type,
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
 * ‘mtxdistvector_fwrite_shared()’ writes a distributed vector as a
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
        &mtxdistfile, mtxdistvector, mtxfmt, disterr);
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
 * ‘mtxdistvector_swap()’ swaps values of two vectors, simultaneously
 * performing ‘y <- x’ and ‘x <- y’.
 */
int mtxdistvector_swap(
    struct mtxdistvector * x,
    struct mtxdistvector * y,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistvector_copy()’ copies values of a vector, ‘y = x’.
 */
int mtxdistvector_copy(
    struct mtxdistvector * y,
    const struct mtxdistvector * x,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistvector_sscal()’ scales a vector by a single precision
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
 * ‘mtxdistvector_dscal()’ scales a vector by a double precision
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
 * ‘mtxdistvector_saxpy()’ adds a vector to another vector multiplied
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
 * ‘mtxdistvector_daxpy()’ adds a vector to another vector multiplied
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
 * ‘mtxdistvector_saypx()’ multiplies a vector by a single precision
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
 * ‘mtxdistvector_daypx()’ multiplies a vector by a double precision
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
 * ‘mtxdistvector_sdot()’ computes the Euclidean dot product of two
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
    float dotp = 0.0f;
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
 * ‘mtxdistvector_ddot()’ computes the Euclidean dot product of two
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
    double dotp = 0.0;
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
 * ‘mtxdistvector_cdotu()’ computes the product of the transpose of a
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
    float dotp[2] = {0.0f, 0.0f};
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
 * ‘mtxdistvector_zdotu()’ computes the product of the transpose of a
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
    double dotp[2] = {0.0, 0.0};
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
 * ‘mtxdistvector_cdotc()’ computes the Euclidean dot product of two
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
    float dotp[2] = {0.0f, 0.0f};
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
 * ‘mtxdistvector_zdotc()’ computes the Euclidean dot product of two
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
    double dotp[2] = {0.0, 0.0};
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
 * ‘mtxdistvector_snrm2()’ computes the Euclidean norm of a vector in
 * single precision floating point.
 */
int mtxdistvector_snrm2(
    const struct mtxdistvector * x,
    float * nrm2,
    int64_t * num_flops,
    struct mtxdisterror * disterr)
{
    int err;
    float dot[2] = {0.0f, 0.0f};
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
 * ‘mtxdistvector_dnrm2()’ computes the Euclidean norm of a vector in
 * double precision floating point.
 */
int mtxdistvector_dnrm2(
    const struct mtxdistvector * x,
    double * nrm2,
    int64_t * num_flops,
    struct mtxdisterror * disterr)
{
    double dot[2] = {0.0, 0.0};
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
 * ‘mtxdistvector_sasum()’ computes the sum of absolute values
 * (1-norm) of a vector in single precision floating point.
 */
int mtxdistvector_sasum(
    const struct mtxdistvector * x,
    float * asum,
    int64_t * num_flops,
    struct mtxdisterror * disterr)
{
    int err = mtxvector_sasum(&x->interior, asum, num_flops);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    disterr->mpierrcode = MPI_Allreduce(
        MPI_IN_PLACE, asum, 1, MPI_FLOAT, MPI_SUM, x->comm);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistvector_dasum()’ computes the sum of absolute values
 * (1-norm) of a vector in double precision floating point.
 */
int mtxdistvector_dasum(
    const struct mtxdistvector * x,
    double * asum,
    int64_t * num_flops,
    struct mtxdisterror * disterr)
{
    int err = mtxvector_dasum(&x->interior, asum, num_flops);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    disterr->mpierrcode = MPI_Allreduce(
        MPI_IN_PLACE, asum, 1, MPI_DOUBLE, MPI_SUM, x->comm);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistvector_iamax()’ finds the index of the first element having
 * the maximum absolute value.
 */
int mtxdistvector_iamax(
    const struct mtxdistvector * x,
    int * max,
    struct mtxdisterror * disterr);

/*
 * Halo update and exchange
 */

/**
 * ‘mtxdistvector_halo_update()’ performs a halo update of a distributed
 * vector.
 */
int mtxdistvector_halo_update(
    struct mtxdistvector * dst,
    const struct mtxdistvector * src,
    struct mtxdisterror * disterr)
{
    int result;
    int err = MPI_Comm_compare(dst->comm, src->comm, &result);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    err = result != MPI_IDENT ? MTX_ERR_INCOMPATIBLE_MPI_COMM : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    MPI_Comm comm = src->comm;
    int comm_size = src->comm_size;
    int srcrank = src->rank;
    int dstrank = dst->rank;

    if (src->rowpart.size != dst->rowpart.size)
        return MTX_ERR_INCOMPATIBLE_SIZE;

    /* Consider a vector with 6 elements with two different
     * partitionings, src and dst, into two parts a and b.
     *
     *   src: a0 a1 a2 b0 b1 b2
     *   dst: c0 d0 c2 d1 c1 d2
     *
     * On the first process, the destination vector has 3 elements
     * with local element numbers 0, 1 and 2. Translating these to
     * global element numbers yields 0, 4 and 2. The source vector
     * also has 3 elements in part a, but the corresponding global
     * element numbers are 0, 1 and 2.
     *
     * The global element numbers of the source vector, 0, 1 and 2,
     * belong to parts c, d and c with respect to the destination
     * vector partitioning. Thus, sorting the source vector elements
     * by their destination parts rearranges them in the order 0, 2,
     * 1, with respect to their global element numbers.
     *
     * Next, the two first elements, 0 and 2, are sent by the first
     * process to itself in this order, and element 4 is sent to the
     * first process by the second process. Thus, the first proecss
     * now possesses elements with global numbers 0, 2 and 4, in that
     * order.
     *
     * Finally, these elements must be rearranged in ascending order
     * of their local element numbers within the part c.
     *
     * Suppose that we take the destination vector elements of the
     * first process (i.e., global element numbers 0, 4 and 2), and
     * partition them according to the source vector partitioning.
     * Then we find that the elements come from parts a, b and a, with
     * local element numbers 0, 1 and 2, within their respective
     * parts.
     */

    /*
     * TODO: This operation can be optimised if the element numbers to
     * send/receive are already sorted by part numbers (and in the
     * correct order within each part). In this case, it is possible
     * to drop the intermediate send/receive buffers and use the
     * source/destination vector directly.
     */

    /*
     * TODO: Another optimisation or use case is to treat the
     * "interior" separately from the halo elements. The "interior"
     * consists of those elements that lie in the inersection of the
     * source and destination partitionings. These elements will
     * remain on the same process, and so it is not necessary to
     * perform any communication for them. If the numbering of these
     * elements is the same in the source and destination, then there
     * is also no need to perform any reordering.
     */

    /*
     * TODO: Yet another optimisation is to reuse the send and receive
     * buffers for repeated halo updates. To achieve this, we should
     * abstract away the halo update by storing the needed data in a
     * struct and offering some functions to initialise and perform
     * halo updates.
     *
     * Finally, some operations, such as the sorting of the send (and
     * receive) buffer allocate some temporary workspace. There may
     * also an opportunity to reuse storage for the workspace in some
     * cases by adding extra arguments to ‘mtxvector_sort()’ and
     * ‘mtxvector_permute()’.
     */

    /*
     * Step 1: Prepare to send data to other processes.
     *
     * This involves counting the number of elements to send from the
     * current process to each other process. Before sending, we also
     * need to group elements together according to their destination
     * process. In addition, we handle the case where the source
     * vector has been reordered to use a local numbering within each
     * part that differs from the global element numbering.
     */

    /* For each element in the source vector owned by the current
     * process, find its global element number and which part of the
     * destination vector that it belongs to. This tells us which
     * processes we must send data to. */
    int srclocalsize = srcrank < src->rowpart.num_parts
        ? src->rowpart.part_sizes[srcrank] : 0;
    int64_t * srcidx = malloc(srclocalsize * sizeof(int64_t));
    err = !srcidx ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;
    for (int64_t j = 0; j < srclocalsize; j++)
        srcidx[j] = j;
    err = mtxpartition_globalidx(
        &src->rowpart, srcrank, srclocalsize, srcidx, srcidx);
    if (mtxdisterror_allreduce(disterr, err)) {
        free(srcidx);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    int * dstparts = malloc(srclocalsize * sizeof(int));
    err = !dstparts ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(srcidx);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    err = mtxpartition_assign(&dst->rowpart, srclocalsize, srcidx, dstparts, NULL);
    if (mtxdisterror_allreduce(disterr, err)) {
        free(dstparts);
        free(srcidx);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    /* count the number of elements to send to each process
     * (sendcounts), as well as the offset to the first element to
     * send to each process (senddispls). */
    int * sendcounts = malloc(comm_size * sizeof(int));
    err = !sendcounts ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(dstparts);
        free(srcidx);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    int * senddispls = malloc((comm_size+1) * sizeof(int));
    err = !senddispls ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(sendcounts);
        free(dstparts);
        free(srcidx);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    for (int p = 0; p < comm_size; p++)
        sendcounts[p] = 0;
    for (int64_t k = 0; k < srclocalsize; k++) {
        if (dstparts[k] < 0 || dstparts[k] >= dst->rowpart.num_parts) {
            err = MTX_ERR_INDEX_OUT_OF_BOUNDS;
            break;
        }
        sendcounts[dstparts[k]]++;
    }
    if (mtxdisterror_allreduce(disterr, err)) {
        free(senddispls);
        free(sendcounts);
        free(dstparts);
        free(srcidx);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    senddispls[0] = 0;
    for (int p = 0; p < comm_size; p++)
        senddispls[p+1] = senddispls[p] + sendcounts[p];
    if (!err && senddispls[comm_size] != srclocalsize)
        err = MTX_ERR_INCOMPATIBLE_PARTITION;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(senddispls);
        free(sendcounts);
        free(dstparts);
        free(srcidx);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    /* sort data to send first by the part number of the destination
     * vector partitioning. */
    struct mtxpermutation sendpermpart;
    err = mtxpermutation_init_default(&sendpermpart, srclocalsize);
    if (mtxdisterror_allreduce(disterr, err)) {
        free(senddispls);
        free(sendcounts);
        free(dstparts);
        free(srcidx);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    err = radix_sort_uint32(srclocalsize, dstparts, sendpermpart.perm);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxpermutation_free(&sendpermpart);
        free(senddispls);
        free(sendcounts);
        free(dstparts);
        free(srcidx);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    free(dstparts);
    err = mtxpermutation_permute_int64(
        &sendpermpart, srclocalsize, srcidx);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxpermutation_free(&sendpermpart);
        free(senddispls);
        free(sendcounts);
        free(srcidx);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    /* Second, within each destination part, sort elements in
     * ascending order of their global element numbers.
     *
     * In this way, the sender and receiver agree on a common ordering
     * of elements that are sent and received, without having to
     * explicitly communicate the underlying ordering that is used.
     * (The global element order is instead inferred from partitioning
     * information that is already available to both processes.) */
    struct mtxpermutation sendpermidx;
    err = mtxpermutation_init_default(&sendpermidx, srclocalsize);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxpermutation_free(&sendpermpart);
        free(senddispls);
        free(sendcounts);
        free(srcidx);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    for (int p = 0; p < comm_size; p++) {
        err = radix_sort_uint64(
            sendcounts[p], (uint64_t *) &srcidx[senddispls[p]],
            &sendpermidx.perm[senddispls[p]]);
        if (mtxdisterror_allreduce(disterr, err)) {
            mtxpermutation_free(&sendpermidx);
            mtxpermutation_free(&sendpermpart);
            free(senddispls);
            free(sendcounts);
            free(srcidx);
            return MTX_ERR_MPI_COLLECTIVE;
        }
        for (int64_t i = 0; i < sendcounts[p]; i++)
            sendpermidx.perm[senddispls[p]+i] += senddispls[p];
    }
    free(srcidx);

    /* combine the two sorting permutations */
    struct mtxpermutation sendperm;
    err = mtxpermutation_compose(&sendperm, &sendpermpart, &sendpermidx);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxpermutation_free(&sendpermidx);
        mtxpermutation_free(&sendpermpart);
        free(senddispls);
        free(sendcounts);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    mtxpermutation_free(&sendpermidx);
    mtxpermutation_free(&sendpermpart);

    /*
     * Step 2: Prepare to receive data from other processes.
     */

    /* For each element in the destination vector owned by the current
     * process, find its global element number and which part of the
     * source vector that it belongs to. This tells us which processes
     * we will receive data from. */
    int dstlocalsize = dstrank < dst->rowpart.num_parts
        ? dst->rowpart.part_sizes[dstrank] : 0;
    int64_t * dstidx = malloc(dstlocalsize * sizeof(int64_t));
    err = !dstidx ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxpermutation_free(&sendperm);
        free(senddispls);
        free(sendcounts);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    for (int64_t j = 0; j < dstlocalsize; j++)
        dstidx[j] = j;
    err = mtxpartition_globalidx(
        &dst->rowpart, dstrank, dstlocalsize, dstidx, dstidx);
    if (mtxdisterror_allreduce(disterr, err)) {
        free(dstidx);
        mtxpermutation_free(&sendperm);
        free(senddispls);
        free(sendcounts);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    int * srcparts = malloc(dstlocalsize * sizeof(int));
    err = !srcparts ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(dstidx);
        mtxpermutation_free(&sendperm);
        free(senddispls);
        free(sendcounts);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    err = mtxpartition_assign(
        &src->rowpart, dstlocalsize, dstidx, srcparts, NULL);
    if (mtxdisterror_allreduce(disterr, err)) {
        free(srcparts);
        free(dstidx);
        mtxpermutation_free(&sendperm);
        free(senddispls);
        free(sendcounts);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    /* count the number of elements to receive from each process
     * (recvcounts), as well as the offset to the first element to
     * receive from each process (recvdispls). */
    int * recvcounts = malloc(comm_size * sizeof(int));
    err = !recvcounts ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(srcparts);
        free(dstidx);
        mtxpermutation_free(&sendperm);
        free(senddispls);
        free(sendcounts);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    int * recvdispls = malloc((comm_size+1) * sizeof(int));
    err = !recvdispls ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(recvcounts);
        free(srcparts);
        free(dstidx);
        mtxpermutation_free(&sendperm);
        free(senddispls);
        free(sendcounts);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    for (int p = 0; p < comm_size; p++)
        recvcounts[p] = 0;
    for (int64_t k = 0; k < dstlocalsize; k++) {
        if (srcparts[k] < 0 || srcparts[k] >= src->rowpart.num_parts) {
            err = MTX_ERR_INDEX_OUT_OF_BOUNDS;
            break;
        }
        recvcounts[srcparts[k]]++;
    }
    recvdispls[0] = 0;
    for (int p = 0; p < comm_size; p++)
        recvdispls[p+1] = recvdispls[p] + recvcounts[p];
    if (!err && recvdispls[comm_size] != dstlocalsize)
        err = MTX_ERR_INCOMPATIBLE_PARTITION;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(recvdispls);
        free(recvcounts);
        free(srcparts);
        free(dstidx);
        mtxpermutation_free(&sendperm);
        free(senddispls);
        free(sendcounts);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    /* sort destination vector first by the part number of the source
     * vector partitioning. */
    struct mtxpermutation recvpermpart;
    err = mtxpermutation_init_default(&recvpermpart, dstlocalsize);
    if (mtxdisterror_allreduce(disterr, err)) {
        free(recvdispls);
        free(recvcounts);
        free(srcparts);
        free(dstidx);
        mtxpermutation_free(&sendperm);
        free(senddispls);
        free(sendcounts);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    err = radix_sort_uint32(dstlocalsize, srcparts, recvpermpart.perm);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxpermutation_free(&recvpermpart);
        free(recvdispls);
        free(recvcounts);
        free(srcparts);
        free(dstidx);
        mtxpermutation_free(&sendperm);
        free(senddispls);
        free(sendcounts);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    free(srcparts);
    err = mtxpermutation_permute_int64(
        &recvpermpart, dstlocalsize, dstidx);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxpermutation_free(&recvpermpart);
        free(recvdispls);
        free(recvcounts);
        free(dstidx);
        mtxpermutation_free(&sendperm);
        free(senddispls);
        free(sendcounts);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    /* Second, within each source vector part, sort elements in
     * ascending order of their global element numbers.
     *
     * In this way, the sender and receiver agree on a common ordering
     * of elements that are sent and received, without having to
     * explicitly communicate the underlying ordering that is used.
     * (The global element order is instead inferred from partitioning
     * information that is already available to both processes.) */
    struct mtxpermutation recvpermidx;
    err = mtxpermutation_init_default(&recvpermidx, dstlocalsize);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxpermutation_free(&recvpermpart);
        free(recvdispls);
        free(recvcounts);
        free(dstidx);
        mtxpermutation_free(&sendperm);
        free(senddispls);
        free(sendcounts);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    for (int p = 0; p < comm_size; p++) {
        err = radix_sort_uint64(
            recvcounts[p], (uint64_t *) &dstidx[recvdispls[p]],
            &recvpermidx.perm[recvdispls[p]]);
        if (mtxdisterror_allreduce(disterr, err)) {
            mtxpermutation_free(&recvpermidx);
            mtxpermutation_free(&recvpermpart);
            free(recvdispls);
            free(recvcounts);
            free(dstidx);
            mtxpermutation_free(&sendperm);
            free(senddispls);
            free(sendcounts);
            return MTX_ERR_MPI_COLLECTIVE;
        }
        for (int64_t i = 0; i < recvcounts[p]; i++)
            recvpermidx.perm[recvdispls[p]+i] += recvdispls[p];
    }
    free(dstidx);

    /* combine the two sorting permutations */
    struct mtxpermutation recvperm;
    err = mtxpermutation_compose(&recvperm, &recvpermpart, &recvpermidx);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxpermutation_free(&recvpermidx);
        mtxpermutation_free(&recvpermpart);
        free(recvdispls);
        free(recvcounts);
        mtxpermutation_free(&sendperm);
        free(senddispls);
        free(sendcounts);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    mtxpermutation_free(&recvpermidx);
    mtxpermutation_free(&recvpermpart);

    /* invert the permutation */
    err = mtxpermutation_invert(&recvperm);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxpermutation_free(&recvperm);
        free(recvdispls);
        free(recvcounts);
        mtxpermutation_free(&sendperm);
        free(senddispls);
        free(sendcounts);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    /*
     * Step 3: Exchange data between processes.
     */

    /* allocate buffer for sending data to other processes */
    struct mtxvector sendbuf;
    err = mtxvector_init_copy(&sendbuf, &src->interior);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxpermutation_free(&recvperm);
        free(recvdispls);
        free(recvcounts);
        mtxpermutation_free(&sendperm);
        free(senddispls);
        free(sendcounts);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    /* Apply the permutation to the data that will be sent. */
    err = mtxvector_permute(&sendbuf, 0, srclocalsize, sendperm.perm);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxvector_free(&sendbuf);
        mtxpermutation_free(&recvperm);
        free(recvdispls);
        free(recvcounts);
        mtxpermutation_free(&sendperm);
        free(senddispls);
        free(sendcounts);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    mtxpermutation_free(&sendperm);

    /* allocate buffer for receiving data from other processes */
    struct mtxvector recvbuf;
    err = mtxvector_alloc_copy(&recvbuf, &dst->interior);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxvector_free(&sendbuf);
        mtxpermutation_free(&recvperm);
        free(recvdispls);
        free(recvcounts);
        free(senddispls);
        free(sendcounts);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    /* perform all-to-all exchange */
    err = mtxvector_alltoallv(
        &sendbuf, 0, sendcounts, senddispls,
        &recvbuf, 0, recvcounts, recvdispls,
        comm, disterr);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxvector_free(&recvbuf);
        mtxvector_free(&sendbuf);
        mtxpermutation_free(&recvperm);
        free(recvdispls);
        free(recvcounts);
        free(senddispls);
        free(sendcounts);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    mtxvector_free(&sendbuf);
    free(recvdispls);
    free(recvcounts);
    free(senddispls);
    free(sendcounts);

    /* permute the received data */
    err = mtxvector_permute(&recvbuf, 0, dstlocalsize, recvperm.perm);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxvector_free(&recvbuf);
        mtxpermutation_free(&recvperm);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    mtxpermutation_free(&recvperm);

    /* copy data from the receiving buffer */
    err = mtxvector_copy(&dst->interior, &recvbuf);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxvector_free(&recvbuf);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    mtxvector_free(&recvbuf);
    return MTX_SUCCESS;
}
#endif
