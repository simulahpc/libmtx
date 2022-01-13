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
 * Last modified: 2022-01-12
 *
 * Matrix Market files distributed among multiple processes with MPI
 * for inter-process communication.
 */

#include <libmtx/libmtx-config.h>

#include <libmtx/error.h>
#include <libmtx/mtx/precision.h>
#include <libmtx/mtxfile/mtxdistfile.h>
#include <libmtx/mtxfile/comments.h>
#include <libmtx/mtxfile/data.h>
#include <libmtx/mtxfile/header.h>
#include <libmtx/mtxfile/mtxfile.h>
#include <libmtx/mtxfile/size.h>
#include <libmtx/util/partition.h>
#include <libmtx/util/sort.h>

#ifdef LIBMTX_HAVE_MPI
#include <mpi.h>

#include <unistd.h>
#include <errno.h>

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Memory management
 */

/**
 * ‘mtxdistfile_free()’ frees storage allocated for a distributed
 * Matrix Market file.
 */
void mtxdistfile_free(
    struct mtxdistfile * mtxdistfile)
{
    mtxfiledata_free(
        &mtxdistfile->data,
        mtxdistfile->header.object,
        mtxdistfile->header.format,
        mtxdistfile->header.field,
        mtxdistfile->precision);
    mtxfilecomments_free(&mtxdistfile->comments);
    mtxpartition_free(&mtxdistfile->partition);
}

/**
 * ‘mtxdistfile_alloc()’ allocates storage for a distributed Matrix
 * Market file with the given header line, comment lines, size line
 * and precision.
 *
 * ‘comments’ may be ‘NULL’, in which case it is ignored.
 *
 * ‘partition’ must be a partitioning of a finite set whose size
 * equals the number of data lines in the underlying Matrix Market
 * file (i.e., ‘size->num_nonzeros’ if ‘header->format’ is
 * ‘mtxfile_coordinate’, or ‘size->num_rows*size->num_columns’ or
 * ‘size->num_rows’ if ‘header->format’ is ‘mtxfile_array’ and
 * ‘header->object’ is ‘mtxfile_matrix’ or ‘mtxfile_vector’,
 * respectively). Also, the number of parts in the partition is at
 * most the number of MPI processes in the communicator ‘comm’.
 *
 * ‘comm’ must be the same MPI communicator that was used to create
 * ‘disterr’.
 */
int mtxdistfile_alloc(
    struct mtxdistfile * mtxdistfile,
    const struct mtxfileheader * header,
    const struct mtxfilecomments * comments,
    const struct mtxfilesize * size,
    enum mtxprecision precision,
    const struct mtxpartition * partition,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = MTX_SUCCESS;
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

    /* Check that the partition is compatible */
    int64_t num_data_lines;
    err = mtxfilesize_num_data_lines(
        size, header->symmetry, &num_data_lines);
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;
    if (partition->num_parts > comm_size ||
        partition->size != num_data_lines)
        err = MTX_ERR_INCOMPATIBLE_PARTITION;
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;

    mtxdistfile->comm = comm;
    mtxdistfile->comm_size = comm_size;
    mtxdistfile->rank = rank;

    err = mtxpartition_copy(&mtxdistfile->partition, partition);
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;

    err = mtxfileheader_copy(&mtxdistfile->header, header);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxpartition_free(&mtxdistfile->partition);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    if (comments) {
        err = mtxfilecomments_copy(&mtxdistfile->comments, comments);
        if (mtxdisterror_allreduce(disterr, err)) {
            mtxpartition_free(&mtxdistfile->partition);
            return MTX_ERR_MPI_COLLECTIVE;
        }
    } else {
        err = mtxfilecomments_init(&mtxdistfile->comments);
        if (mtxdisterror_allreduce(disterr, err)) {
            mtxpartition_free(&mtxdistfile->partition);
            return MTX_ERR_MPI_COLLECTIVE;
        }
    }
    err = mtxfilesize_copy(&mtxdistfile->size, size);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxfilecomments_free(&mtxdistfile->comments);
        mtxpartition_free(&mtxdistfile->partition);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    mtxdistfile->precision = precision;

    int num_local_data_lines = partition->part_sizes[rank];
    err = mtxfiledata_alloc(
        &mtxdistfile->data, mtxdistfile->header.object, mtxdistfile->header.format,
        mtxdistfile->header.field, mtxdistfile->precision, num_local_data_lines);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxfilecomments_free(&mtxdistfile->comments);
        mtxpartition_free(&mtxdistfile->partition);
        return err;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistfile_alloc_copy()’ allocates storage for a copy of a Matrix
 * Market file without initialising the underlying values.
 */
int mtxdistfile_alloc_copy(
    struct mtxdistfile * dst,
    const struct mtxdistfile * src,
    struct mtxdisterror * disterr)
{
    int err;
    dst->comm = src->comm;
    dst->comm_size = src->comm_size;
    dst->rank = src->rank;
    err = mtxpartition_copy(&dst->partition, &src->partition);
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;
    err = mtxfileheader_copy(&dst->header, &src->header);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxpartition_free(&dst->partition);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    err = mtxfilecomments_copy(&dst->comments, &src->comments);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxpartition_free(&dst->partition);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    err = mtxfilesize_copy(&dst->size, &src->size);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxfilecomments_free(&dst->comments);
        mtxpartition_free(&dst->partition);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    dst->precision = src->precision;

    int num_local_data_lines = dst->partition.part_sizes[dst->rank];
    err = mtxfiledata_alloc(
        &dst->data, dst->header.object, dst->header.format,
        dst->header.field, dst->precision, num_local_data_lines);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxfilecomments_free(&dst->comments);
        mtxpartition_free(&dst->partition);
        return err;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistfile_init_copy()’ creates a copy of a Matrix Market file.
 */
int mtxdistfile_init_copy(
    struct mtxdistfile * dst,
    const struct mtxdistfile * src,
    struct mtxdisterror * disterr);

/*
 * Matrix array formats
 */

/**
 * ‘mtxdistfile_alloc_matrix_array()’ allocates a distributed matrix
 * in array format.
 */
int mtxdistfile_alloc_matrix_array(
    struct mtxdistfile * mtxdistfile,
    enum mtxfilefield field,
    enum mtxfilesymmetry symmetry,
    enum mtxprecision precision,
    int num_rows,
    int num_columns,
    const struct mtxpartition * partition,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err;
    if (field != mtxfile_real &&
        field != mtxfile_complex &&
        field != mtxfile_integer)
        return MTX_ERR_INVALID_MTX_FIELD;
    if (symmetry != mtxfile_general &&
        symmetry != mtxfile_symmetric &&
        symmetry != mtxfile_skew_symmetric &&
        symmetry != mtxfile_hermitian)
        return MTX_ERR_INVALID_MTX_SYMMETRY;
    if (precision != mtx_single &&
        precision != mtx_double)
        return MTX_ERR_INVALID_PRECISION;

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

    mtxdistfile->comm = comm;
    mtxdistfile->comm_size = comm_size;
    mtxdistfile->rank = rank;
    err = mtxpartition_copy(&mtxdistfile->partition, partition);
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;

    mtxdistfile->header.object = mtxfile_matrix;
    mtxdistfile->header.format = mtxfile_array;
    mtxdistfile->header.field = field;
    mtxdistfile->header.symmetry = symmetry;
    mtxfilecomments_init(&mtxdistfile->comments);
    mtxdistfile->size.num_rows = num_rows;
    mtxdistfile->size.num_columns = num_columns;
    mtxdistfile->size.num_nonzeros = -1;

    /* Check that the partition is compatible */
    int64_t num_data_lines;
    err = mtxfilesize_num_data_lines(
        &mtxdistfile->size, symmetry, &num_data_lines);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxfilecomments_free(&mtxdistfile->comments);
        mtxpartition_free(&mtxdistfile->partition);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    if (partition->num_parts > comm_size ||
        partition->size != num_data_lines)
        err = MTX_ERR_INCOMPATIBLE_PARTITION;
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxfilecomments_free(&mtxdistfile->comments);
        mtxpartition_free(&mtxdistfile->partition);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    mtxdistfile->precision = precision;

    int num_local_data_lines =
        mtxdistfile->partition.part_sizes[mtxdistfile->rank];
    err = mtxfiledata_alloc(
        &mtxdistfile->data, mtxdistfile->header.object, mtxdistfile->header.format,
        mtxdistfile->header.field, mtxdistfile->precision, num_local_data_lines);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxfilecomments_free(&mtxdistfile->comments);
        mtxpartition_free(&mtxdistfile->partition);
        return err;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistfile_init_matrix_array_real_single()’ allocates and
 * initialises a distributed matrix in array format with real, single
 * precision coefficients.
 */
int mtxdistfile_init_matrix_array_real_single(
    struct mtxdistfile * mtxdistfile,
    enum mtxfilesymmetry symmetry,
    int num_rows,
    int num_columns,
    const float * data,
    const struct mtxpartition * partition,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err;
    err = mtxdistfile_alloc_matrix_array(
        mtxdistfile, mtxfile_real, symmetry, mtx_single, num_rows, num_columns,
        partition, comm, disterr);
    if (err)
        return err;
    int num_local_data_lines =
        mtxdistfile->partition.part_sizes[mtxdistfile->rank];
    memcpy(mtxdistfile->data.array_real_single, data,
           num_local_data_lines * sizeof(*data));
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistfile_init_matrix_array_real_double()’ allocates and
 * initialises a distributed matrix in array format with real, double
 * precision coefficients.
 */
int mtxdistfile_init_matrix_array_real_double(
    struct mtxdistfile * mtxdistfile,
    enum mtxfilesymmetry symmetry,
    int num_rows,
    int num_columns,
    const double * data,
    const struct mtxpartition * partition,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err;
    err = mtxdistfile_alloc_matrix_array(
        mtxdistfile, mtxfile_real, symmetry, mtx_double, num_rows, num_columns,
        partition, comm, disterr);
    if (err)
        return err;
    int num_local_data_lines =
        mtxdistfile->partition.part_sizes[mtxdistfile->rank];
    memcpy(mtxdistfile->data.array_real_double, data,
           num_local_data_lines * sizeof(*data));
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistfile_init_matrix_array_complex_single()’ allocates and
 * initialises a distributed matrix in array format with complex,
 * single precision coefficients.
 */
int mtxdistfile_init_matrix_array_complex_single(
    struct mtxdistfile * mtxdistfile,
    enum mtxfilesymmetry symmetry,
    int num_rows,
    int num_columns,
    const float (* data)[2],
    const struct mtxpartition * partition,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistfile_init_matrix_array_complex_double()’ allocates and
 * initialises a matrix in array format with complex, double precision
 * coefficients.
 */
int mtxdistfile_init_matrix_array_complex_double(
    struct mtxdistfile * mtxdistfile,
    enum mtxfilesymmetry symmetry,
    int num_rows,
    int num_columns,
    const double (* data)[2],
    const struct mtxpartition * partition,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistfile_init_matrix_array_integer_single()’ allocates and
 * initialises a distributed matrix in array format with integer,
 * single precision coefficients.
 */
int mtxdistfile_init_matrix_array_integer_single(
    struct mtxdistfile * mtxdistfile,
    enum mtxfilesymmetry symmetry,
    int num_rows,
    int num_columns,
    const int32_t * data,
    const struct mtxpartition * partition,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistfile_init_matrix_array_integer_double()’ allocates and
 * initialises a matrix in array format with integer, double precision
 * coefficients.
 */
int mtxdistfile_init_matrix_array_integer_double(
    struct mtxdistfile * mtxdistfile,
    enum mtxfilesymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * data,
    const struct mtxpartition * partition,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/*
 * Vector array formats
 */

/**
 * ‘mtxdistfile_alloc_vector_array()’ allocates a distributed vector
 * in array format.
 */
int mtxdistfile_alloc_vector_array(
    struct mtxdistfile * mtxdistfile,
    enum mtxfilefield field,
    enum mtxprecision precision,
    int num_rows,
    const struct mtxpartition * partition,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err;
    if (field != mtxfile_real &&
        field != mtxfile_complex &&
        field != mtxfile_integer)
        return MTX_ERR_INVALID_MTX_FIELD;
    if (precision != mtx_single &&
        precision != mtx_double)
        return MTX_ERR_INVALID_PRECISION;

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

    mtxdistfile->comm = comm;
    mtxdistfile->comm_size = comm_size;
    mtxdistfile->rank = rank;
    err = mtxpartition_copy(&mtxdistfile->partition, partition);
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;

    mtxdistfile->header.object = mtxfile_vector;
    mtxdistfile->header.format = mtxfile_array;
    mtxdistfile->header.field = field;
    mtxdistfile->header.symmetry = mtxfile_general;
    mtxfilecomments_init(&mtxdistfile->comments);
    mtxdistfile->size.num_rows = num_rows;
    mtxdistfile->size.num_columns = -1;
    mtxdistfile->size.num_nonzeros = -1;

    /* Check that the partition is compatible */
    int64_t num_data_lines;
    err = mtxfilesize_num_data_lines(
        &mtxdistfile->size, mtxfile_general, &num_data_lines);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxfilecomments_free(&mtxdistfile->comments);
        mtxpartition_free(&mtxdistfile->partition);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    if (partition->num_parts > comm_size ||
        partition->size != num_data_lines)
        err = MTX_ERR_INCOMPATIBLE_PARTITION;
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxfilecomments_free(&mtxdistfile->comments);
        mtxpartition_free(&mtxdistfile->partition);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    mtxdistfile->precision = precision;

    int num_local_data_lines =
        mtxdistfile->partition.part_sizes[mtxdistfile->rank];
    err = mtxfiledata_alloc(
        &mtxdistfile->data, mtxdistfile->header.object, mtxdistfile->header.format,
        mtxdistfile->header.field, mtxdistfile->precision, num_local_data_lines);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxfilecomments_free(&mtxdistfile->comments);
        mtxpartition_free(&mtxdistfile->partition);
        return err;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistfile_init_vector_array_real_single()’ allocates and
 * initialises a distributed vector in array format with real, single
 * precision coefficients.
 */
int mtxdistfile_init_vector_array_real_single(
    struct mtxdistfile * mtxdistfile,
    int num_rows,
    const float * data,
    const struct mtxpartition * partition,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err;
    err = mtxdistfile_alloc_vector_array(
        mtxdistfile, mtxfile_real, mtx_single, num_rows,
        partition, comm, disterr);
    if (err)
        return err;
    int num_local_data_lines =
        mtxdistfile->partition.part_sizes[mtxdistfile->rank];
    memcpy(mtxdistfile->data.array_real_single, data,
           num_local_data_lines * sizeof(*data));
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistfile_init_vector_array_real_double()’ allocates and initialises
 * a vector in array format with real, double precision coefficients.
 */
int mtxdistfile_init_vector_array_real_double(
    struct mtxdistfile * mtxdistfile,
    int num_rows,
    const double * data,
    const struct mtxpartition * partition,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err;
    err = mtxdistfile_alloc_vector_array(
        mtxdistfile, mtxfile_real, mtx_double, num_rows,
        partition, comm, disterr);
    if (err)
        return err;
    int num_local_data_lines =
        mtxdistfile->partition.part_sizes[mtxdistfile->rank];
    memcpy(mtxdistfile->data.array_real_double, data,
           num_local_data_lines * sizeof(*data));
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistfile_init_vector_array_complex_single()’ allocates and
 * initialises a distributed vector in array format with complex,
 * single precision coefficients.
 */
int mtxdistfile_init_vector_array_complex_single(
    struct mtxdistfile * mtxdistfile,
    int num_rows,
    const float (* data)[2],
    const struct mtxpartition * partition,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err;
    err = mtxdistfile_alloc_vector_array(
        mtxdistfile, mtxfile_complex, mtx_single, num_rows,
        partition, comm, disterr);
    if (err)
        return err;
    int num_local_data_lines =
        mtxdistfile->partition.part_sizes[mtxdistfile->rank];
    memcpy(mtxdistfile->data.array_complex_single, data,
           num_local_data_lines * sizeof(*data));
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistfile_init_vector_array_complex_double()’ allocates and
 * initialises a vector in array format with complex, double precision
 * coefficients.
 */
int mtxdistfile_init_vector_array_complex_double(
    struct mtxdistfile * mtxdistfile,
    int num_rows,
    const double (* data)[2],
    const struct mtxpartition * partition,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err;
    err = mtxdistfile_alloc_vector_array(
        mtxdistfile, mtxfile_complex, mtx_double, num_rows,
        partition, comm, disterr);
    if (err)
        return err;
    int num_local_data_lines =
        mtxdistfile->partition.part_sizes[mtxdistfile->rank];
    memcpy(mtxdistfile->data.array_complex_double, data,
           num_local_data_lines * sizeof(*data));
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistfile_init_vector_array_integer_single()’ allocates and
 * initialises a distributed vector in array format with integer,
 * single precision coefficients.
 */
int mtxdistfile_init_vector_array_integer_single(
    struct mtxdistfile * mtxdistfile,
    int num_rows,
    const int32_t * data,
    const struct mtxpartition * partition,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err;
    err = mtxdistfile_alloc_vector_array(
        mtxdistfile, mtxfile_integer, mtx_single, num_rows,
        partition, comm, disterr);
    if (err)
        return err;
    int num_local_data_lines =
        mtxdistfile->partition.part_sizes[mtxdistfile->rank];
    memcpy(mtxdistfile->data.array_integer_single, data,
           num_local_data_lines * sizeof(*data));
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistfile_init_vector_array_integer_double()’ allocates and
 * initialises a vector in array format with integer, double precision
 * coefficients.
 */
int mtxdistfile_init_vector_array_integer_double(
    struct mtxdistfile * mtxdistfile,
    int num_rows,
    const int64_t * data,
    const struct mtxpartition * partition,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err;
    err = mtxdistfile_alloc_vector_array(
        mtxdistfile, mtxfile_integer, mtx_double, num_rows,
        partition, comm, disterr);
    if (err)
        return err;
    int num_local_data_lines =
        mtxdistfile->partition.part_sizes[mtxdistfile->rank];
    memcpy(mtxdistfile->data.array_integer_double, data,
           num_local_data_lines * sizeof(*data));
    return MTX_SUCCESS;
}

/*
 * Matrix coordinate formats
 */

/**
 * ‘mtxdistfile_alloc_matrix_coordinate()’ allocates a distributed
 * matrix in coordinate format.
 */
int mtxdistfile_alloc_matrix_coordinate(
    struct mtxdistfile * mtxdistfile,
    enum mtxfilefield field,
    enum mtxfilesymmetry symmetry,
    enum mtxprecision precision,
    int num_rows,
    int num_columns,
    int64_t num_nonzeros,
    const struct mtxpartition * partition,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err;
    if (field != mtxfile_real &&
        field != mtxfile_complex &&
        field != mtxfile_integer &&
        field != mtxfile_pattern)
        return MTX_ERR_INVALID_MTX_FIELD;
    if (symmetry != mtxfile_general &&
        symmetry != mtxfile_symmetric &&
        symmetry != mtxfile_skew_symmetric &&
        symmetry != mtxfile_hermitian)
        return MTX_ERR_INVALID_MTX_SYMMETRY;
    if (precision != mtx_single &&
        precision != mtx_double)
        return MTX_ERR_INVALID_PRECISION;

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

    if (partition->num_parts > comm_size ||
        partition->size != num_nonzeros)
        err = MTX_ERR_INCOMPATIBLE_PARTITION;
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;

    mtxdistfile->comm = comm;
    mtxdistfile->comm_size = comm_size;
    mtxdistfile->rank = rank;
    err = mtxpartition_copy(&mtxdistfile->partition, partition);
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;

    mtxdistfile->header.object = mtxfile_matrix;
    mtxdistfile->header.format = mtxfile_coordinate;
    mtxdistfile->header.field = field;
    mtxdistfile->header.symmetry = symmetry;
    mtxfilecomments_init(&mtxdistfile->comments);
    mtxdistfile->size.num_rows = num_rows;
    mtxdistfile->size.num_columns = num_columns;
    mtxdistfile->size.num_nonzeros = num_nonzeros;
    mtxdistfile->precision = precision;

    int num_local_data_lines =
        mtxdistfile->partition.part_sizes[mtxdistfile->rank];
    err = mtxfiledata_alloc(
        &mtxdistfile->data, mtxdistfile->header.object, mtxdistfile->header.format,
        mtxdistfile->header.field, mtxdistfile->precision, num_local_data_lines);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxfilecomments_free(&mtxdistfile->comments);
        mtxpartition_free(&mtxdistfile->partition);
        return err;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistfile_init_matrix_coordinate_real_single()’ allocates and
 * initialises a distributed matrix in coordinate format with real,
 * single precision coefficients.
 */
int mtxdistfile_init_matrix_coordinate_real_single(
    struct mtxdistfile * mtxdistfile,
    enum mtxfilesymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_nonzeros,
    const struct mtxfile_matrix_coordinate_real_single * data,
    const struct mtxpartition * partition,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err;
    err = mtxdistfile_alloc_matrix_coordinate(
        mtxdistfile, mtxfile_real, symmetry, mtx_single,
        num_rows, num_columns, num_nonzeros,
        partition, comm, disterr);
    if (err)
        return err;
    int num_local_data_lines =
        mtxdistfile->partition.part_sizes[mtxdistfile->rank];
    memcpy(mtxdistfile->data.matrix_coordinate_real_single, data,
           num_local_data_lines * sizeof(*data));
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistfile_init_matrix_coordinate_real_double()’ allocates and
 * initialises a matrix in coordinate format with real, double
 * precision coefficients.
 */
int mtxdistfile_init_matrix_coordinate_real_double(
    struct mtxdistfile * mtxdistfile,
    enum mtxfilesymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_nonzeros,
    const struct mtxfile_matrix_coordinate_real_double * data,
    const struct mtxpartition * partition,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err;
    err = mtxdistfile_alloc_matrix_coordinate(
        mtxdistfile, mtxfile_real, symmetry, mtx_double,
        num_rows, num_columns, num_nonzeros,
        partition, comm, disterr);
    if (err)
        return err;
    int num_local_data_lines =
        mtxdistfile->partition.part_sizes[mtxdistfile->rank];
    memcpy(mtxdistfile->data.matrix_coordinate_real_double, data,
           num_local_data_lines * sizeof(*data));
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistfile_init_matrix_coordinate_complex_single()’ allocates and
 * initialises a distributed matrix in coordinate format with complex,
 * single precision coefficients.
 */
int mtxdistfile_init_matrix_coordinate_complex_single(
    struct mtxdistfile * mtxdistfile,
    enum mtxfilesymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_nonzeros,
    const struct mtxfile_matrix_coordinate_complex_single * data,
    const struct mtxpartition * partition,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistfile_init_matrix_coordinate_complex_double()’ allocates and
 * initialises a matrix in coordinate format with complex, double
 * precision coefficients.
 */
int mtxdistfile_init_matrix_coordinate_complex_double(
    struct mtxdistfile * mtxdistfile,
    enum mtxfilesymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_nonzeros,
    const struct mtxfile_matrix_coordinate_complex_double * data,
    const struct mtxpartition * partition,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistfile_init_matrix_coordinate_integer_single()’ allocates and
 * initialises a distributed matrix in coordinate format with integer,
 * single precision coefficients.
 */
int mtxdistfile_init_matrix_coordinate_integer_single(
    struct mtxdistfile * mtxdistfile,
    enum mtxfilesymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_nonzeros,
    const struct mtxfile_matrix_coordinate_integer_single * data,
    const struct mtxpartition * partition,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistfile_init_matrix_coordinate_integer_double()’ allocates and
 * initialises a matrix in coordinate format with integer, double
 * precision coefficients.
 */
int mtxdistfile_init_matrix_coordinate_integer_double(
    struct mtxdistfile * mtxdistfile,
    enum mtxfilesymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_nonzeros,
    const struct mtxfile_matrix_coordinate_integer_double * data,
    const struct mtxpartition * partition,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistfile_init_matrix_coordinate_pattern()’ allocates and
 * initialises a matrix in coordinate format with boolean (pattern)
 * precision coefficients.
 */
int mtxdistfile_init_matrix_coordinate_pattern(
    struct mtxdistfile * mtxdistfile,
    enum mtxfilesymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_nonzeros,
    const struct mtxfile_matrix_coordinate_pattern * data,
    const struct mtxpartition * partition,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/*
 * Vector coordinate formats
 */

/**
 * ‘mtxdistfile_alloc_vector_coordinate()’ allocates a distributed
 * vector in coordinate format.
 */
int mtxdistfile_alloc_vector_coordinate(
    struct mtxdistfile * mtxdistfile,
    enum mtxfilefield field,
    enum mtxprecision precision,
    int num_rows,
    int64_t num_nonzeros,
    const struct mtxpartition * partition,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err;
    if (field != mtxfile_real &&
        field != mtxfile_complex &&
        field != mtxfile_integer &&
        field != mtxfile_pattern)
        return MTX_ERR_INVALID_MTX_FIELD;
    if (precision != mtx_single &&
        precision != mtx_double)
        return MTX_ERR_INVALID_PRECISION;

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

    if (partition->num_parts > comm_size ||
        partition->size != num_nonzeros)
        err = MTX_ERR_INCOMPATIBLE_PARTITION;
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;

    mtxdistfile->comm = comm;
    mtxdistfile->comm_size = comm_size;
    mtxdistfile->rank = rank;
    err = mtxpartition_copy(&mtxdistfile->partition, partition);
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;

    mtxdistfile->header.object = mtxfile_vector;
    mtxdistfile->header.format = mtxfile_coordinate;
    mtxdistfile->header.field = field;
    mtxdistfile->header.symmetry = mtxfile_general;
    mtxfilecomments_init(&mtxdistfile->comments);
    mtxdistfile->size.num_rows = num_rows;
    mtxdistfile->size.num_columns = -1;
    mtxdistfile->size.num_nonzeros = num_nonzeros;
    mtxdistfile->precision = precision;

    int num_local_data_lines =
        mtxdistfile->partition.part_sizes[mtxdistfile->rank];
    err = mtxfiledata_alloc(
        &mtxdistfile->data, mtxdistfile->header.object, mtxdistfile->header.format,
        mtxdistfile->header.field, mtxdistfile->precision, num_local_data_lines);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxfilecomments_free(&mtxdistfile->comments);
        mtxpartition_free(&mtxdistfile->partition);
        return err;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistfile_init_vector_coordinate_real_single()’ allocates and
 * initialises a distributed vector in coordinate format with real,
 * single precision coefficients.
 */
int mtxdistfile_init_vector_coordinate_real_single(
    struct mtxdistfile * mtxdistfile,
    int num_rows,
    int64_t num_nonzeros,
    const struct mtxfile_vector_coordinate_real_single * data,
    const struct mtxpartition * partition,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err;
    err = mtxdistfile_alloc_vector_coordinate(
        mtxdistfile, mtxfile_real, mtx_single,
        num_rows, num_nonzeros, partition, comm, disterr);
    if (err)
        return err;
    int num_local_data_lines =
        mtxdistfile->partition.part_sizes[mtxdistfile->rank];
    memcpy(mtxdistfile->data.vector_coordinate_real_single, data,
           num_local_data_lines * sizeof(*data));
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistfile_init_vector_coordinate_real_double()’ allocates and
 * initialises a vector in coordinate format with real, double
 * precision coefficients.
 */
int mtxdistfile_init_vector_coordinate_real_double(
    struct mtxdistfile * mtxdistfile,
    int num_rows,
    int64_t num_nonzeros,
    const struct mtxfile_vector_coordinate_real_double * data,
    const struct mtxpartition * partition,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err;
    err = mtxdistfile_alloc_vector_coordinate(
        mtxdistfile, mtxfile_real, mtx_double,
        num_rows, num_nonzeros, partition, comm, disterr);
    if (err)
        return err;
    int num_local_data_lines =
        mtxdistfile->partition.part_sizes[mtxdistfile->rank];
    memcpy(mtxdistfile->data.vector_coordinate_real_double, data,
           num_local_data_lines * sizeof(*data));
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistfile_init_vector_coordinate_complex_single()’ allocates and
 * initialises a distributed vector in coordinate format with complex,
 * single precision coefficients.
 */
int mtxdistfile_init_vector_coordinate_complex_single(
    struct mtxdistfile * mtxdistfile,
    int num_rows,
    int64_t num_nonzeros,
    const struct mtxfile_vector_coordinate_complex_single * data,
    const struct mtxpartition * partition,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistfile_init_vector_coordinate_complex_double()’ allocates and
 * initialises a vector in coordinate format with complex, double
 * precision coefficients.
 */
int mtxdistfile_init_vector_coordinate_complex_double(
    struct mtxdistfile * mtxdistfile,
    int num_rows,
    int64_t num_nonzeros,
    const struct mtxfile_vector_coordinate_complex_double * data,
    const struct mtxpartition * partition,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistfile_init_vector_coordinate_integer_single()’ allocates and
 * initialises a distributed vector in coordinate format with integer,
 * single precision coefficients.
 */
int mtxdistfile_init_vector_coordinate_integer_single(
    struct mtxdistfile * mtxdistfile,
    int num_rows,
    int64_t num_nonzeros,
    const struct mtxfile_vector_coordinate_integer_single * data,
    const struct mtxpartition * partition,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistfile_init_vector_coordinate_integer_double()’ allocates and
 * initialises a vector in coordinate format with integer, double
 * precision coefficients.
 */
int mtxdistfile_init_vector_coordinate_integer_double(
    struct mtxdistfile * mtxdistfile,
    int num_rows,
    int64_t num_nonzeros,
    const struct mtxfile_vector_coordinate_integer_double * data,
    const struct mtxpartition * partition,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistfile_init_vector_coordinate_pattern()’ allocates and
 * initialises a vector in coordinate format with boolean (pattern)
 * precision coefficients.
 */
int mtxdistfile_init_vector_coordinate_pattern(
    struct mtxdistfile * mtxdistfile,
    int num_rows,
    int64_t num_nonzeros,
    const struct mtxfile_vector_coordinate_pattern * data,
    const struct mtxpartition * partition,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/*
 * Modifying values
 */

/**
 * ‘mtxdistfile_set_constant_real_single()’ sets every (nonzero) value
 * of a matrix or vector equal to a constant, single precision
 * floating point number.
 */
int mtxdistfile_set_constant_real_single(
    struct mtxdistfile * mtxdistfile,
    float a,
    struct mtxdisterror * disterr)
{
    int num_local_data_lines =
        mtxdistfile->partition.part_sizes[mtxdistfile->rank];
    int err = mtxfiledata_set_constant_real_single(
        &mtxdistfile->data, mtxdistfile->header.object,
        mtxdistfile->header.format, mtxdistfile->header.field,
        mtxdistfile->precision, num_local_data_lines, 0, a);
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistfile_set_constant_real_double()’ sets every (nonzero) value
 * of a matrix or vector equal to a constant, double precision
 * floating point number.
 */
int mtxdistfile_set_constant_real_double(
    struct mtxdistfile * mtxdistfile,
    double a,
    struct mtxdisterror * disterr)
{
    int num_local_data_lines =
        mtxdistfile->partition.part_sizes[mtxdistfile->rank];
    int err = mtxfiledata_set_constant_real_double(
        &mtxdistfile->data, mtxdistfile->header.object,
        mtxdistfile->header.format, mtxdistfile->header.field,
        mtxdistfile->precision, num_local_data_lines, 0, a);
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistfile_set_constant_complex_single()’ sets every (nonzero)
 * value of a matrix or vector equal to a constant, single precision
 * floating point complex number.
 */
int mtxdistfile_set_constant_complex_single(
    struct mtxdistfile * mtxdistfile,
    float a[2],
    struct mtxdisterror * disterr)
{
    int num_local_data_lines =
        mtxdistfile->partition.part_sizes[mtxdistfile->rank];
    int err = mtxfiledata_set_constant_complex_single(
        &mtxdistfile->data, mtxdistfile->header.object,
        mtxdistfile->header.format, mtxdistfile->header.field,
        mtxdistfile->precision, num_local_data_lines, 0, a);
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistfile_set_constant_complex_double()’ sets every (nonzero)
 * value of a matrix or vector equal to a constant, double precision
 * floating point complex number.
 */
int mtxdistfile_set_constant_complex_double(
    struct mtxdistfile * mtxdistfile,
    double a[2],
    struct mtxdisterror * disterr)
{
    int num_local_data_lines =
        mtxdistfile->partition.part_sizes[mtxdistfile->rank];
    int err = mtxfiledata_set_constant_complex_double(
        &mtxdistfile->data, mtxdistfile->header.object,
        mtxdistfile->header.format, mtxdistfile->header.field,
        mtxdistfile->precision, num_local_data_lines, 0, a);
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistfile_set_constant_integer_single()’ sets every (nonzero)
 * value of a matrix or vector equal to a constant, 32-bit integer.
 */
int mtxdistfile_set_constant_integer_single(
    struct mtxdistfile * mtxdistfile,
    int32_t a,
    struct mtxdisterror * disterr)
{
    int num_local_data_lines =
        mtxdistfile->partition.part_sizes[mtxdistfile->rank];
    int err = mtxfiledata_set_constant_integer_single(
        &mtxdistfile->data, mtxdistfile->header.object,
        mtxdistfile->header.format, mtxdistfile->header.field,
        mtxdistfile->precision, num_local_data_lines, 0, a);
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistfile_set_constant_integer_double()’ sets every (nonzero)
 * value of a matrix or vector equal to a constant, 64-bit integer.
 */
int mtxdistfile_set_constant_integer_double(
    struct mtxdistfile * mtxdistfile,
    int64_t a,
    struct mtxdisterror * disterr)
{
    int num_local_data_lines =
        mtxdistfile->partition.part_sizes[mtxdistfile->rank];
    int err = mtxfiledata_set_constant_integer_double(
        &mtxdistfile->data, mtxdistfile->header.object,
        mtxdistfile->header.format, mtxdistfile->header.field,
        mtxdistfile->precision, num_local_data_lines, 0, a);
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/*
 * Convert to and from (non-distributed) Matrix Market format
 */

/**
 * ‘mtxdistfile_from_mtxfile()’ creates a distributed Matrix Market
 * file from a Matrix Market file stored on a single root process by
 * distributing the data of the underlying matrix or vector among
 * processes in a communicator.
 *
 * This function performs collective communication and therefore
 * requires every process in the communicator to perform matching
 * calls to this function.
 */
int mtxdistfile_from_mtxfile(
    struct mtxdistfile * dst,
    const struct mtxfile * src,
    MPI_Comm comm,
    int root,
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
    if (root < 0 || root > comm_size)
        return MTX_ERR_INDEX_OUT_OF_BOUNDS;

    dst->comm = comm;
    dst->comm_size = comm_size;
    dst->rank = rank;

    /* Broadcast the header, comments, size line and precision. */
    err = (rank == root) ? mtxfileheader_copy(
        &dst->header, &src->header) : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;
    err = mtxfileheader_bcast(&dst->header, root, comm, disterr);
    if (err)
        return err;
    err = (rank == root) ? mtxfilecomments_copy(
        &dst->comments, &src->comments) : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;
    err = mtxfilecomments_bcast(&dst->comments, root, comm, disterr);
    if (err) {
        if (rank == root)
            mtxfilecomments_free(&dst->comments);
        return err;
    }
    err = (rank == root) ? mtxfilesize_copy(
        &dst->size, &src->size) : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxfilecomments_free(&dst->comments);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    err = mtxfilesize_bcast(&dst->size, root, comm, disterr);
    if (err) {
        mtxfilecomments_free(&dst->comments);
        return err;
    }
    if (rank == root)
        dst->precision = src->precision;
    disterr->mpierrcode = MPI_Bcast(
        &dst->precision, 1, MPI_INT, root, comm);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxfilecomments_free(&dst->comments);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    /* Partition the data lines into equal-sized blocks. */
    int64_t num_data_lines;
    err = mtxfilesize_num_data_lines(
        &dst->size, dst->header.symmetry, &num_data_lines);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxfilecomments_free(&dst->comments);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    err = mtxpartition_init_block(
        &dst->partition, num_data_lines, comm_size, NULL);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxfilecomments_free(&dst->comments);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    /* Find the offset to the first data line to send to each part. */
    int recvcount = dst->partition.part_sizes[rank];
    int * sendcounts = (rank == root) ?
        malloc(comm_size * sizeof(int)) : NULL;
    err = (rank == root && !sendcounts) ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxpartition_free(&dst->partition);
        mtxfilecomments_free(&dst->comments);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    if (rank == root) {
        for (int p = 0; p < comm_size; p++) {
            if (dst->partition.part_sizes[p] > INT_MAX) {
                errno = ERANGE;
                err = MTX_ERR_ERRNO;
                break;
            }
            sendcounts[p] = dst->partition.part_sizes[p];
        }
    }
    if (mtxdisterror_allreduce(disterr, err)) {
        if (rank == root) free(sendcounts);
        mtxpartition_free(&dst->partition);
        mtxfilecomments_free(&dst->comments);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    int * displs = (rank == root) ?
        malloc((comm_size+1) * sizeof(int)) : NULL;
    err = (rank == root && !displs) ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        if (rank == root) free(sendcounts);
        mtxpartition_free(&dst->partition);
        mtxfilecomments_free(&dst->comments);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    if (rank == root) {
        for (int p = 0; p < comm_size; p++) {
            if (dst->partition.parts_ptr[p] > INT_MAX) {
                errno = ERANGE;
                err = MTX_ERR_ERRNO;
                break;
            }
            displs[p] = dst->partition.parts_ptr[p];
        }
    }
    if (mtxdisterror_allreduce(disterr, err)) {
        if (rank == root) free(displs);
        if (rank == root) free(sendcounts);
        mtxpartition_free(&dst->partition);
        mtxfilecomments_free(&dst->comments);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    err = mtxfiledata_alloc(
        &dst->data, dst->header.object, dst->header.format,
        dst->header.field, dst->precision, recvcount);
    if (mtxdisterror_allreduce(disterr, err)) {
        if (rank == root) free(displs);
        if (rank == root) free(sendcounts);
        mtxfilecomments_free(&dst->comments);
        mtxpartition_free(&dst->partition);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    /* Scatter the data lines of the Matrix Market file. */
    err = mtxfiledata_scatterv(
        &src->data, dst->header.object, dst->header.format,
        dst->header.field, dst->precision,
        0, sendcounts, displs, &dst->data, 0, recvcount,
        root, comm, disterr);
    if (err) {
        mtxfiledata_free(
            &dst->data, dst->header.object, dst->header.format,
            dst->header.field, dst->precision);
        if (rank == root) free(displs);
        if (rank == root) free(sendcounts);
        mtxfilecomments_free(&dst->comments);
        mtxpartition_free(&dst->partition);
        return err;
    }

    if (rank == root) free(displs);
    if (rank == root) free(sendcounts);
    return MTX_SUCCESS;
}

/*
 * I/O functions
 */

/**
 * ‘mtxdistfile_read_shared()’ reads a Matrix Market file from the given path
 * and distributes the data among MPI processes in a communicator. The
 * file may optionally be compressed by gzip.
 *
 * The ‘precision’ argument specifies which precision to use for
 * storing matrix or vector values.
 *
 * If ‘path’ is ‘-’, then standard input is used.
 *
 * The file is assumed to be gzip-compressed if ‘gzip’ is ‘true’, and
 * uncompressed otherwise.
 *
 * If an error code is returned, then ‘lines_read’ and ‘bytes_read’
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the Matrix Market file.
 *
 * Only a single root process will read from the specified stream.
 * The data is partitioned into equal-sized parts for each process.
 * For matrices and vectors in coordinate format, the total number of
 * data lines is evenly distributed among processes. Otherwise, the
 * rows are evenly distributed among processes.
 *
 * The file is read one part at a time, which is then sent to the
 * owning process. This avoids reading the entire file into the memory
 * of the root process at once, which would severely limit the size of
 * files that could be read.
 *
 * This function performs collective communication and therefore
 * requires every process in the communicator to perform matching
 * calls to the function.
 */
int mtxdistfile_read_shared(
    struct mtxdistfile * mtxdistfile,
    enum mtxprecision precision,
    const char * path,
    bool gzip,
    int * lines_read,
    int64_t * bytes_read,
    MPI_Comm comm,
    int root,
    struct mtxdisterror * disterr)
{
    int err;
    if (lines_read)
        *lines_read = -1;
    if (bytes_read)
        *bytes_read = 0;

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
    if (root < 0 || root > comm_size)
        return MTX_ERR_INDEX_OUT_OF_BOUNDS;

    if (!gzip) {
        FILE * f;
        if (rank == root && strcmp(path, "-") == 0) {
            int fd = dup(STDIN_FILENO);
            if (fd == -1) {
                err = MTX_ERR_ERRNO;
            } else if ((f = fdopen(fd, "r")) == NULL) {
                int olderrno = errno;
                close(fd);
                errno = olderrno;
                err = MTX_ERR_ERRNO;
            } else {
                err = MTX_SUCCESS;
            }
        } else if (rank == root && ((f = fopen(path, "r")) == NULL)) {
            err = MTX_ERR_ERRNO;
        } else {
            err = MTX_SUCCESS;
        }
        if (mtxdisterror_allreduce(disterr, err))
            return MTX_ERR_MPI_COLLECTIVE;

        if (lines_read)
            *lines_read = 0;
        err = mtxdistfile_fread_shared(
            mtxdistfile, precision, f, lines_read, bytes_read, 0, NULL,
            comm, root, disterr);
        if (err) {
            if (rank == root)
                fclose(f);
            return err;
        }
        if (rank == root)
            fclose(f);
    } else {
#ifdef LIBMTX_HAVE_LIBZ
        gzFile f;
        if (rank == root && strcmp(path, "-") == 0) {
            int fd = dup(STDIN_FILENO);
            if (fd == -1)
                err = MTX_ERR_ERRNO;
            if ((f = gzdopen(fd, "r")) == NULL) {
                int olderrno = errno;
                close(fd);
                errno = olderrno;
                err = MTX_ERR_ERRNO;
            } else {
                err = MTX_SUCCESS;
            }
        } else if (rank == root && (f = gzopen(path, "r")) == NULL) {
            err = MTX_ERR_ERRNO;
        } else {
            err = MTX_SUCCESS;
        }
        if (mtxdisterror_allreduce(disterr, err))
            return MTX_ERR_MPI_COLLECTIVE;

        if (lines_read)
            *lines_read = 0;
        err = mtxdistfile_gzread_shared(
            mtxdistfile, precision, f,
            lines_read, bytes_read, 0, NULL,
            comm, root, disterr);
        if (err) {
            if (rank == root)
                gzclose(f);
            return err;
        }
        if (rank == root)
            gzclose(f);
#else
        return MTX_ERR_ZLIB_NOT_SUPPORTED;
#endif
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistfile_fread_shared()’ reads a Matrix Market file from a stream and
 * distributes the data among MPI processes in a communicator.
 *
 * ‘precision’ is used to determine the precision to use for storing
 * the values of matrix or vector entries.
 *
 * If an error code is returned, then ‘lines_read’ and ‘bytes_read’
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the Matrix Market file.
 *
 * If ‘linebuf’ is not ‘NULL’, then it must point to an array that can
 * hold at least ‘line_max’ values of type ‘char’. This buffer is used
 * for reading lines from the stream. Otherwise, if ‘linebuf’ is
 * ‘NULL’, then a temporary buffer is allocated and used, and the
 * maximum line length is determined by calling ‘sysconf()’ with
 * ‘_SC_LINE_MAX’.
 *
 * Only a single root process will read from the specified stream.
 * The data is partitioned into equal-sized parts for each process.
 * For matrices and vectors in coordinate format, the total number of
 * data lines is evenly distributed among processes. Otherwise, the
 * rows are evenly distributed among processes.
 *
 * The file is read one part at a time, which is then sent to the
 * owning process. This avoids reading the entire file into the memory
 * of the root process at once, which would severely limit the size of
 * files that could be read.
 *
 * This function performs collective communication and therefore
 * requires every process in the communicator to perform matching
 * calls to the function.
 */
int mtxdistfile_fread_shared(
    struct mtxdistfile * mtxdistfile,
    enum mtxprecision precision,
    FILE * f,
    int * lines_read,
    int64_t * bytes_read,
    size_t line_max,
    char * linebuf,
    MPI_Comm comm,
    int root,
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
    if (root < 0 || root > comm_size)
        return MTX_ERR_INDEX_OUT_OF_BOUNDS;

    mtxdistfile->comm = comm;
    mtxdistfile->comm_size = comm_size;
    mtxdistfile->rank = rank;

    bool free_linebuf = (rank == root) && !linebuf;
    if (rank == root && !linebuf) {
        line_max = sysconf(_SC_LINE_MAX);
        linebuf = malloc(line_max+1);
    }
    err = (rank == root && !linebuf) ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;

    /* Read the header on the root process and broadcast to others. */
    err = (rank == root) ? mtxfileheader_fread(
        &mtxdistfile->header, f, lines_read, bytes_read, line_max, linebuf)
        : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        if (free_linebuf)
            free(linebuf);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    err = mtxfileheader_bcast(&mtxdistfile->header, root, comm, disterr);
    if (err) {
        if (free_linebuf)
            free(linebuf);
        return err;
    }

    /* Read comments on the root process and broadcast to others. */
    err = (rank == root) ? mtxfile_fread_comments(
        &mtxdistfile->comments, f, lines_read, bytes_read, line_max, linebuf)
        : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        if (free_linebuf)
            free(linebuf);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    err = mtxfilecomments_bcast(&mtxdistfile->comments, root, comm, disterr);
    if (err) {
        if (free_linebuf)
            free(linebuf);
        return err;
    }

    /* Read the size line on the root process and broadcast to others. */
    err = (rank == root) ? mtxfilesize_fread(
        &mtxdistfile->size, f, lines_read, bytes_read, line_max, linebuf,
        mtxdistfile->header.object, mtxdistfile->header.format)
        : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxfilecomments_free(&mtxdistfile->comments);
        if (free_linebuf)
            free(linebuf);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    err = mtxfilesize_bcast(&mtxdistfile->size, root, comm, disterr);
    if (err) {
        mtxfilecomments_free(&mtxdistfile->comments);
        if (free_linebuf)
            free(linebuf);
        return err;
    }
    mtxdistfile->precision = precision;

    /* Partition the data lines into equal-sized blocks. */
    int64_t num_data_lines;
    err = mtxfilesize_num_data_lines(
        &mtxdistfile->size, mtxdistfile->header.symmetry, &num_data_lines);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxfilecomments_free(&mtxdistfile->comments);
        if (free_linebuf)
            free(linebuf);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    err = mtxpartition_init_block(
        &mtxdistfile->partition, num_data_lines, comm_size, NULL);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxfilecomments_free(&mtxdistfile->comments);
        if (free_linebuf)
            free(linebuf);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    /* Allocate storage for data on each process */
    err = mtxfiledata_alloc(
        &mtxdistfile->data, mtxdistfile->header.object, mtxdistfile->header.format,
        mtxdistfile->header.field, mtxdistfile->precision,
        mtxdistfile->partition.part_sizes[rank]);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxpartition_free(&mtxdistfile->partition);
        mtxfilecomments_free(&mtxdistfile->comments);
        if (free_linebuf)
            free(linebuf);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    /* Allocate temporary storage on the root process for reading data
     * lines. */
    int64_t max_part_size = 0;
    for (int p = 0; p < mtxdistfile->partition.num_parts; p++) {
        if (p != root && max_part_size < mtxdistfile->partition.part_sizes[p])
            max_part_size = mtxdistfile->partition.part_sizes[p];
    }
    union mtxfiledata data;
    err = (rank == root) ?
        mtxfiledata_alloc(
            &data, mtxdistfile->header.object, mtxdistfile->header.format,
            mtxdistfile->header.field, mtxdistfile->precision, max_part_size)
        : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxpartition_free(&mtxdistfile->partition);
        mtxfilecomments_free(&mtxdistfile->comments);
        if (free_linebuf)
            free(linebuf);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    /* Read each part of the Matrix Market file and send it to the
     * owning process. */
    for (int p = 0; p < comm_size; p++) {
        int64_t num_data_lines =
            mtxdistfile->partition.part_sizes[p];

        /* On the root process, read data lines directly into their
         * final location. Otherwise, read data into temporary storage
         * before sending it to the owning process. */
        err = (rank == root) ? mtxfiledata_fread(
            (p == root) ? &mtxdistfile->data : &data,
            f, lines_read, bytes_read, line_max, linebuf,
            mtxdistfile->header.object, mtxdistfile->header.format,
            mtxdistfile->header.field, mtxdistfile->precision,
            mtxdistfile->size.num_rows,
            mtxdistfile->size.num_columns,
            num_data_lines, 0)
            : MTX_SUCCESS;
        if (mtxdisterror_allreduce(disterr, err)) {
            if (rank == root) {
                mtxfiledata_free(
                    &data,
                    mtxdistfile->header.object, mtxdistfile->header.format,
                    mtxdistfile->header.field, mtxdistfile->precision);
            }
            mtxfiledata_free(
                &mtxdistfile->data,
                mtxdistfile->header.object, mtxdistfile->header.format,
                mtxdistfile->header.field, mtxdistfile->precision);
            mtxpartition_free(&mtxdistfile->partition);
            mtxfilecomments_free(&mtxdistfile->comments);
            if (free_linebuf)
                free(linebuf);
            return MTX_ERR_MPI_COLLECTIVE;
        }

        /* Send to the owning process. */
        if (p != root && rank == root) {
            err = mtxfiledata_send(
                &data,
                mtxdistfile->header.object, mtxdistfile->header.format,
                mtxdistfile->header.field, mtxdistfile->precision,
                num_data_lines, 0, p, 0, comm, disterr);
        } else if (p != root && rank == p) {
            err = mtxfiledata_recv(
                &mtxdistfile->data,
                mtxdistfile->header.object, mtxdistfile->header.format,
                mtxdistfile->header.field, mtxdistfile->precision,
                num_data_lines, 0, root, 0, comm, disterr);
        } else {
            err = MTX_SUCCESS;
        }
        if (mtxdisterror_allreduce(disterr, err)) {
            if (rank == root) {
                mtxfiledata_free(
                    &data,
                    mtxdistfile->header.object, mtxdistfile->header.format,
                    mtxdistfile->header.field, mtxdistfile->precision);
            }
            mtxfiledata_free(
                &mtxdistfile->data,
                mtxdistfile->header.object, mtxdistfile->header.format,
                mtxdistfile->header.field, mtxdistfile->precision);
            mtxpartition_free(&mtxdistfile->partition);
            mtxfilecomments_free(&mtxdistfile->comments);
            if (free_linebuf)
                free(linebuf);
            return MTX_ERR_MPI_COLLECTIVE;
        }
    }

    if (rank == root) {
        mtxfiledata_free(
            &data,
            mtxdistfile->header.object, mtxdistfile->header.format,
            mtxdistfile->header.field, mtxdistfile->precision);
    }
    if (free_linebuf)
        free(linebuf);

    disterr->mpierrcode = MPI_Bcast(lines_read, 1, MPI_INT, root, comm);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxdistfile_free(mtxdistfile);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    disterr->mpierrcode = MPI_Bcast(bytes_read, 1, MPI_INT64_T, root, comm);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxdistfile_free(mtxdistfile);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    return MTX_SUCCESS;
}

#ifdef LIBMTX_HAVE_LIBZ
/**
 * ‘mtxdistfile_gzread_shared()’ reads a Matrix Market file from a
 * gzip-compressed stream and distributes the data among MPI processes
 * in a communicator.
 *
 * ‘precision’ is used to determine the precision to use for storing
 * the values of matrix or vector entries.
 *
 * If an error code is returned, then ‘lines_read’ and ‘bytes_read’
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the Matrix Market file.
 *
 * If ‘linebuf’ is not ‘NULL’, then it must point to an array that can
 * hold at least ‘line_max’ values of type ‘char’. This buffer is used
 * for reading lines from the stream. Otherwise, if ‘linebuf’ is
 * ‘NULL’, then a temporary buffer is allocated and used, and the
 * maximum line length is determined by calling ‘sysconf()’ with
 * ‘_SC_LINE_MAX’.
 *
 * Only a single root process will read from the specified stream.
 * The data is partitioned into equal-sized parts for each process.
 * For matrices and vectors in coordinate format, the total number of
 * data lines is evenly distributed among processes. Otherwise, the
 * rows are evenly distributed among processes.
 *
 * The file is read one part at a time, which is then sent to the
 * owning process. This avoids reading the entire file into the memory
 * of the root process at once, which would severely limit the size of
 * files that could be read.
 *
 * This function performs collective communication and therefore
 * requires every process in the communicator to perform matching
 * calls to the function.
 */
int mtxdistfile_gzread_shared(
    struct mtxdistfile * mtxdistfile,
    enum mtxprecision precision,
    gzFile f,
    int * lines_read,
    int64_t * bytes_read,
    size_t line_max,
    char * linebuf,
    MPI_Comm comm,
    int root,
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
    if (root < 0 || root > comm_size)
        return MTX_ERR_INDEX_OUT_OF_BOUNDS;

    mtxdistfile->comm = comm;
    mtxdistfile->comm_size = comm_size;
    mtxdistfile->rank = rank;

    bool free_linebuf = (rank == root) && !linebuf;
    if (rank == root && !linebuf) {
        line_max = sysconf(_SC_LINE_MAX);
        linebuf = malloc(line_max+1);
    }
    err = (rank == root && !linebuf) ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;

    /* Read the header on the root process and broadcast to others. */
    err = (rank == root) ? mtxfileheader_gzread(
        &mtxdistfile->header, f, lines_read, bytes_read, line_max, linebuf)
        : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        if (free_linebuf)
            free(linebuf);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    err = mtxfileheader_bcast(&mtxdistfile->header, root, comm, disterr);
    if (err) {
        if (free_linebuf)
            free(linebuf);
        return err;
    }

    /* Read comments on the root process and broadcast to others. */
    err = (rank == root) ? mtxfile_gzread_comments(
        &mtxdistfile->comments, f, lines_read, bytes_read, line_max, linebuf)
        : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        if (free_linebuf)
            free(linebuf);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    err = mtxfilecomments_bcast(&mtxdistfile->comments, root, comm, disterr);
    if (err) {
        if (free_linebuf)
            free(linebuf);
        return err;
    }

    /* Read the size line on the root process and broadcast to others. */
    err = (rank == root) ? mtxfilesize_gzread(
        &mtxdistfile->size, f, lines_read, bytes_read, line_max, linebuf,
        mtxdistfile->header.object, mtxdistfile->header.format)
        : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxfilecomments_free(&mtxdistfile->comments);
        if (free_linebuf)
            free(linebuf);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    err = mtxfilesize_bcast(&mtxdistfile->size, root, comm, disterr);
    if (err) {
        mtxfilecomments_free(&mtxdistfile->comments);
        if (free_linebuf)
            free(linebuf);
        return err;
    }
    mtxdistfile->precision = precision;

    /* Partition the data lines into equal-sized blocks. */
    int64_t num_data_lines;
    err = mtxfilesize_num_data_lines(
        &mtxdistfile->size, mtxdistfile->header.symmetry, &num_data_lines);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxfilecomments_free(&mtxdistfile->comments);
        if (free_linebuf)
            free(linebuf);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    err = mtxpartition_init_block(
        &mtxdistfile->partition, num_data_lines, comm_size, NULL);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxfilecomments_free(&mtxdistfile->comments);
        if (free_linebuf)
            free(linebuf);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    /* Allocate storage for data on each process */
    err = mtxfiledata_alloc(
        &mtxdistfile->data, mtxdistfile->header.object, mtxdistfile->header.format,
        mtxdistfile->header.field, mtxdistfile->precision,
        mtxdistfile->partition.part_sizes[rank]);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxpartition_free(&mtxdistfile->partition);
        mtxfilecomments_free(&mtxdistfile->comments);
        if (free_linebuf)
            free(linebuf);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    /* Allocate temporary storage on the root process for reading data
     * lines. */
    int64_t max_part_size = 0;
    for (int p = 0; p < mtxdistfile->partition.num_parts; p++) {
        if (p != root && max_part_size < mtxdistfile->partition.part_sizes[p])
            max_part_size = mtxdistfile->partition.part_sizes[p];
    }
    union mtxfiledata data;
    err = (rank == root) ?
        mtxfiledata_alloc(
            &data, mtxdistfile->header.object, mtxdistfile->header.format,
            mtxdistfile->header.field, mtxdistfile->precision, max_part_size)
        : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxpartition_free(&mtxdistfile->partition);
        mtxfilecomments_free(&mtxdistfile->comments);
        if (free_linebuf)
            free(linebuf);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    /* Read each part of the Matrix Market file and send it to the
     * owning process. */
    for (int p = 0; p < comm_size; p++) {
        int64_t num_data_lines =
            mtxdistfile->partition.part_sizes[p];

        /* On the root process, read data lines directly into their
         * final location. Otherwise, read data into temporary storage
         * before sending it to the owning process. */
        err = (rank == root) ? mtxfiledata_gzread(
            (p == root) ? &mtxdistfile->data : &data,
            f, lines_read, bytes_read, line_max, linebuf,
            mtxdistfile->header.object, mtxdistfile->header.format,
            mtxdistfile->header.field, mtxdistfile->precision,
            mtxdistfile->size.num_rows,
            mtxdistfile->size.num_columns,
            num_data_lines, 0)
            : MTX_SUCCESS;
        if (mtxdisterror_allreduce(disterr, err)) {
            if (rank == root) {
                mtxfiledata_free(
                    &data,
                    mtxdistfile->header.object, mtxdistfile->header.format,
                    mtxdistfile->header.field, mtxdistfile->precision);
            }
            mtxfiledata_free(
                &mtxdistfile->data,
                mtxdistfile->header.object, mtxdistfile->header.format,
                mtxdistfile->header.field, mtxdistfile->precision);
            mtxpartition_free(&mtxdistfile->partition);
            mtxfilecomments_free(&mtxdistfile->comments);
            if (free_linebuf)
                free(linebuf);
            return MTX_ERR_MPI_COLLECTIVE;
        }

        /* Send to the owning process. */
        if (p != root && rank == root) {
            err = mtxfiledata_send(
                &data,
                mtxdistfile->header.object, mtxdistfile->header.format,
                mtxdistfile->header.field, mtxdistfile->precision,
                num_data_lines, 0, p, 0, comm, disterr);
        } else if (p != root && rank == p) {
            err = mtxfiledata_recv(
                &mtxdistfile->data,
                mtxdistfile->header.object, mtxdistfile->header.format,
                mtxdistfile->header.field, mtxdistfile->precision,
                num_data_lines, 0, root, 0, comm, disterr);
        } else {
            err = MTX_SUCCESS;
        }
        if (mtxdisterror_allreduce(disterr, err)) {
            if (rank == root) {
                mtxfiledata_free(
                    &data,
                    mtxdistfile->header.object, mtxdistfile->header.format,
                    mtxdistfile->header.field, mtxdistfile->precision);
            }
            mtxfiledata_free(
                &mtxdistfile->data,
                mtxdistfile->header.object, mtxdistfile->header.format,
                mtxdistfile->header.field, mtxdistfile->precision);
            mtxpartition_free(&mtxdistfile->partition);
            mtxfilecomments_free(&mtxdistfile->comments);
            if (free_linebuf)
                free(linebuf);
            return MTX_ERR_MPI_COLLECTIVE;
        }
    }

    if (rank == root) {
        mtxfiledata_free(
            &data,
            mtxdistfile->header.object, mtxdistfile->header.format,
            mtxdistfile->header.field, mtxdistfile->precision);
    }
    if (free_linebuf)
        free(linebuf);

    disterr->mpierrcode = MPI_Bcast(lines_read, 1, MPI_INT, root, comm);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxdistfile_free(mtxdistfile);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    disterr->mpierrcode = MPI_Bcast(bytes_read, 1, MPI_INT64_T, root, comm);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxdistfile_free(mtxdistfile);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    return MTX_SUCCESS;
}
#endif

/**
 * ‘mtxdistfile_write_shared()’ writes a distributed Matrix Market
 * file to a single file that is shared by all processes in the
 * communicator.  The file may optionally be compressed by gzip.
 *
 * If ‘path’ is ‘-’, then standard output is used.
 *
 * If ‘fmt’ is ‘NULL’, then the format specifier ‘%g’ is used to print
 * floating point numbers with enough digits to ensure correct
 * round-trip conversion from decimal text and back.  Otherwise, the
 * given format string is used to print numerical values.
 *
 * The format string follows the conventions of ‘printf’. If the field
 * is ‘real’ or ‘complex’, then the format specifiers '%e', '%E',
 * '%f', '%F', '%g' or '%G' may be used. If the field is ‘integer’,
 * then the format specifier must be '%d'. The format string is
 * ignored if the field is ‘pattern’. Field width and precision may be
 * specified (e.g., "%3.1f"), but variable field width and precision
 * (e.g., "%*.*f"), as well as length modifiers (e.g., "%Lf") are not
 * allowed.
 *
 * Note that only the specified ‘root’ process will print anything to
 * the stream. Other processes will therefore send their part of the
 * distributed Matrix Market file to the root process for printing.
 *
 * This function performs collective communication and therefore
 * requires every process in the communicator to perform matching
 * calls to the function.
 */
int mtxdistfile_write_shared(
    const struct mtxdistfile * mtxdistfile,
    const char * path,
    bool gzip,
    const char * fmt,
    int64_t * bytes_written,
    int root,
    struct mtxdisterror * disterr)
{
    int err;
    int rank = mtxdistfile->rank;
    if (root < 0 || root > mtxdistfile->comm_size)
        return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    *bytes_written = 0;
    if (!gzip) {
        FILE * f;
        if (rank == root && strcmp(path, "-") == 0) {
            int fd = dup(STDOUT_FILENO);
            if (fd == -1)
                err = MTX_ERR_ERRNO;
            if ((f = fdopen(fd, "w")) == NULL) {
                int olderrno = errno;
                close(fd);
                errno = olderrno;
                err = MTX_ERR_ERRNO;
            }
        } else if (rank == root && (f = fopen(path, "w")) == NULL) {
            err = MTX_ERR_ERRNO;
        } else {
            err = MTX_SUCCESS;
        }
        if (mtxdisterror_allreduce(disterr, err))
            return MTX_ERR_MPI_COLLECTIVE;
        err = mtxdistfile_fwrite_shared(
            mtxdistfile, f, fmt, bytes_written, root, disterr);
        if (err) {
            fclose(f);
            return err;
        }
        fclose(f);
    } else {
#ifdef LIBMTX_HAVE_LIBZ
        errno = ENOTSUP;
        return MTX_ERR_ERRNO;
#else
        return MTX_ERR_ZLIB_NOT_SUPPORTED;
#endif
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistfile_fwrite_shared()’ writes a distributed Matrix Market
 * file to a single stream that is shared by every process in the
 * communicator.
 *
 * If ‘fmt’ is ‘NULL’, then the format specifier ‘%g’ is used to print
 * floating point numbers with enough digits to ensure correct
 * round-trip conversion from decimal text and back.  Otherwise, the
 * given format string is used to print numerical values.
 *
 * The format string follows the conventions of ‘printf’. If the field
 * is ‘real’ or ‘complex’, then the format specifiers '%e', '%E',
 * '%f', '%F', '%g' or '%G' may be used. If the field is ‘integer’,
 * then the format specifier must be '%d'. The format string is
 * ignored if the field is ‘pattern’. Field width and precision may be
 * specified (e.g., "%3.1f"), but variable field width and precision
 * (e.g., "%*.*f"), as well as length modifiers (e.g., "%Lf") are not
 * allowed.
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
int mtxdistfile_fwrite_shared(
    const struct mtxdistfile * mtxdistfile,
    FILE * f,
    const char * fmt,
    int64_t * bytes_written,
    int root,
    struct mtxdisterror * disterr)
{
    int err;
    MPI_Comm comm = mtxdistfile->comm;
    int comm_size = mtxdistfile->comm_size;
    int rank = mtxdistfile->rank;
    if (root < 0 || root > comm_size)
        return MTX_ERR_INDEX_OUT_OF_BOUNDS;

    err = (rank == root) ? mtxfileheader_fwrite(
        &mtxdistfile->header, f, bytes_written)
        : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;
    err = (rank == root) ? mtxfilecomments_fputs(
        &mtxdistfile->comments, f, bytes_written)
        : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;
    err = (rank == root) ? mtxfilesize_fwrite(
        &mtxdistfile->size, mtxdistfile->header.object,
        mtxdistfile->header.format,
        f, bytes_written)
        : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;

    /* Allocate temporary storage on the root process for receiving
     * data lines. */
    int64_t max_part_size = 0;
    for (int p = 0; p < mtxdistfile->partition.num_parts; p++) {
        if (p != root && max_part_size < mtxdistfile->partition.part_sizes[p])
            max_part_size = mtxdistfile->partition.part_sizes[p];
    }
    union mtxfiledata data;
    err = (rank == root) ?
        mtxfiledata_alloc(
            &data, mtxdistfile->header.object, mtxdistfile->header.format,
            mtxdistfile->header.field, mtxdistfile->precision, max_part_size)
        : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;

    for (int p = 0; p < comm_size; p++) {
        int64_t num_data_lines =
            mtxdistfile->partition.part_sizes[p];

        /* Send to the root process. */
        if (p != root && rank == root) {
            err = mtxfiledata_recv(
                &data,
                mtxdistfile->header.object, mtxdistfile->header.format,
                mtxdistfile->header.field, mtxdistfile->precision,
                num_data_lines, 0, p, 0, comm, disterr);
        } else if (p != root && rank == p) {
            err = mtxfiledata_send(
                &mtxdistfile->data,
                mtxdistfile->header.object, mtxdistfile->header.format,
                mtxdistfile->header.field, mtxdistfile->precision,
                num_data_lines, 0, root, 0, comm, disterr);
        } else {
            err = MTX_SUCCESS;
        }
        if (mtxdisterror_allreduce(disterr, err)) {
            if (rank == root) {
                mtxfiledata_free(
                    &data,
                    mtxdistfile->header.object, mtxdistfile->header.format,
                    mtxdistfile->header.field, mtxdistfile->precision);
            }
            return MTX_ERR_MPI_COLLECTIVE;
        }

        /* On the root process, write data lines directly from their
         * current location. Otherwise, write data from the temporary
         * storage that was used to receive from the owning
         * process. */
        err = (rank == root) ? mtxfiledata_fwrite(
            (p == root) ? &mtxdistfile->data : &data,
            mtxdistfile->header.object, mtxdistfile->header.format,
            mtxdistfile->header.field, mtxdistfile->precision,
            num_data_lines, f, fmt, bytes_written)
            : MTX_SUCCESS;
        if (mtxdisterror_allreduce(disterr, err)) {
            if (rank == root) {
                mtxfiledata_free(
                    &data,
                    mtxdistfile->header.object, mtxdistfile->header.format,
                    mtxdistfile->header.field, mtxdistfile->precision);
            }
            return MTX_ERR_MPI_COLLECTIVE;
        }
    }

    if (rank == root) {
        mtxfiledata_free(
            &data,
            mtxdistfile->header.object, mtxdistfile->header.format,
            mtxdistfile->header.field, mtxdistfile->precision);
    }

    if (bytes_written) {
        int64_t bytes_written_per_process = *bytes_written;
        disterr->mpierrcode = MPI_Allreduce(
            &bytes_written_per_process, bytes_written, 1, MPI_INT64_T, MPI_SUM,
            mtxdistfile->comm);
        err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
        if (mtxdisterror_allreduce(disterr, err))
            return MTX_ERR_MPI_COLLECTIVE;
    }
    return MTX_SUCCESS;
}

/*
 * Transpose and conjugate transpose.
 */

/**
 * ‘mtxdistfile_transpose()’ tranposes a distributed Matrix Market
 * file.
 */
int mtxdistfile_transpose(
    struct mtxdistfile * mtxdistfile,
    struct mtxdisterror * disterr)
{
    int err;
    int rank = mtxdistfile->rank;
    int64_t local_size = rank < mtxdistfile->partition.num_parts
        ? mtxdistfile->partition.part_sizes[rank] : 0;

    if (mtxdistfile->header.object == mtxfile_matrix) {
        if (mtxdistfile->header.format == mtxfile_array) {
            err = mtxdistfile_sort(
                mtxdistfile, mtxfile_column_major,
                mtxdistfile->partition.size, NULL, disterr);
            if (err)
                return err;
        } else if (mtxdistfile->header.format == mtxfile_coordinate) {
            err = mtxfiledata_transpose(
                &mtxdistfile->data,
                mtxdistfile->header.object, mtxdistfile->header.format,
                mtxdistfile->header.field, mtxdistfile->precision,
                mtxdistfile->size.num_rows, mtxdistfile->size.num_columns,
                local_size);
            if (mtxdisterror_allreduce(disterr, err))
                return MTX_ERR_MPI_COLLECTIVE;
        } else {
            return MTX_ERR_INVALID_MTX_FORMAT;
        }
        err = mtxfilesize_transpose(&mtxdistfile->size);
        if (mtxdisterror_allreduce(disterr, err))
            return MTX_ERR_MPI_COLLECTIVE;
    } else if (mtxdistfile->header.object == mtxfile_vector) {
        return MTX_SUCCESS;
    } else if (mtxdistfile->header.object != mtxfile_vector) {
        return MTX_ERR_INVALID_MTX_OBJECT;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistfile_conjugate_transpose()’ tranposes and complex
 * conjugates a distributed Matrix Market file.
 */
int mtxdistfile_conjugate_transpose(
    struct mtxdistfile * mtxfile,
    struct mtxdisterror * disterr);

/*
 * Sorting
 */

/**
 * ‘mtxdistfile_sort_permutation()’ globally permutes the order of
 * data lines in a distributed Matrix Market file according to a given
 * permutation. The data lines are redistributed among the
 * participating processes, if necessary.
 *
 * The array ‘perm’ must be an array of length at least equal to the
 * number of data lines residing on the current MPI process (i.e.,
 * ‘mtxdistfile->partition.part_sizes[p]’, where ‘p’ is the rank of
 * the current process). This array represents a permutation used to
 * map data lines held by the current process to their new global
 * indices. The permutation must use 0-based indexing.
 */
static int mtxdistfile_sort_permutation(
    struct mtxdistfile * mtxdistfile,
    int64_t * perm,
    struct mtxdisterror * disterr)
{
    int err = MTX_SUCCESS;
    MPI_Comm comm = mtxdistfile->comm;
    int comm_size = mtxdistfile->comm_size;
    int rank = mtxdistfile->rank;
    int64_t size = mtxdistfile->partition.size;
    int64_t local_size = rank < mtxdistfile->partition.num_parts
        ? mtxdistfile->partition.part_sizes[rank] : 0;

    /* The current implementation can only sort at most ‘INT_MAX’ keys
     * on each process due to the use of ‘MPI_Alltoallv’. */
    if (local_size > INT_MAX) errno = ERANGE;
    err = local_size > INT_MAX ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;

    size_t element_size;
    err = mtxfiledata_size_per_element(
        &mtxdistfile->data,
        mtxdistfile->header.object, mtxdistfile->header.format,
        mtxdistfile->header.field, mtxdistfile->precision,
        &element_size);
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;
    void * baseptr;
    err = mtxfiledata_dataptr(
        &mtxdistfile->data,
        mtxdistfile->header.object, mtxdistfile->header.format,
        mtxdistfile->header.field, mtxdistfile->precision,
        &baseptr, 0);
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;

    /* 1. Compute the part to which each element belongs. This
     * requires briefly switching from 1-based to 0-based indexing. */
    int * parts = malloc(local_size * sizeof(int));
    err = !parts ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;

    for (int64_t k = 0; k < local_size; k++)
        perm[k]--;
    err = mtxpartition_assign(
        &mtxdistfile->partition, local_size, perm, parts);
    if (mtxdisterror_allreduce(disterr, err)) {
        free(parts);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    /* 2. Count the number of data lines to send to each process. */
    int * sendcounts = malloc(4*comm_size * sizeof(int));
    err = !sendcounts ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(parts);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    int * senddispls = &sendcounts[comm_size];
    int * recvcounts = &senddispls[comm_size];
    int * recvdispls = &recvcounts[comm_size];

    for (int p = 0; p < comm_size; p++)
        sendcounts[p] = 0;
    for (int64_t k = 0; k < local_size; k++)
        sendcounts[parts[k]]++;
    senddispls[0] = 0;
    for (int p = 1; p < comm_size; p++)
        senddispls[p] = senddispls[p-1] + sendcounts[p-1];

    int64_t * srcdispls = malloc(local_size * sizeof(int64_t));
    err = !srcdispls ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(sendcounts);
        free(parts);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    int64_t * sendperm = malloc(local_size * sizeof(int64_t));
    err = !sendperm ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(srcdispls);
        free(sendcounts);
        free(parts);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    /* 3. Copy data to the send buffer. */
    for (int p = 0; p < comm_size; p++)
        sendcounts[p] = 0;
    for (int64_t k = 0; k < local_size; k++) {
        int64_t destidx = perm[k]-1;
        int p = parts[k];
        srcdispls[senddispls[p]+sendcounts[p]] = k;
        sendperm[senddispls[p]+sendcounts[p]] = perm[k];
        sendcounts[p]++;
    }
    free(parts);

    union mtxfiledata senddata;
    err = mtxfiledata_alloc(
        &senddata,
        mtxdistfile->header.object, mtxdistfile->header.format,
        mtxdistfile->header.field, mtxdistfile->precision, local_size);
    if (mtxdisterror_allreduce(disterr, err)) {
        free(sendperm);
        free(srcdispls);
        free(sendcounts);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    err = mtxfiledata_copy_gather(
        &senddata, &mtxdistfile->data,
        mtxdistfile->header.object, mtxdistfile->header.format,
        mtxdistfile->header.field, mtxdistfile->precision,
        local_size, 0, srcdispls);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxfiledata_free(
            &senddata, mtxdistfile->header.object, mtxdistfile->header.format,
            mtxdistfile->header.field, mtxdistfile->precision);
        free(sendperm);
        free(srcdispls);
        free(sendcounts);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    free(srcdispls);

    /* 4. Obtain ‘recvcounts’ from other processes. */
    disterr->mpierrcode = MPI_Alltoall(
        sendcounts, 1, MPI_INT, recvcounts, 1, MPI_INT, comm);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxfiledata_free(
            &senddata, mtxdistfile->header.object, mtxdistfile->header.format,
            mtxdistfile->header.field, mtxdistfile->precision);
        free(sendperm);
        free(sendcounts);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    recvdispls[0] = 0;
    for (int p = 1; p < comm_size; p++)
        recvdispls[p] = recvdispls[p-1] + recvcounts[p-1];

    /* 5. Exchange the data between processes. */
    err = mtxfiledata_alltoallv(
        &senddata,
        mtxdistfile->header.object, mtxdistfile->header.format,
        mtxdistfile->header.field, mtxdistfile->precision,
        0, sendcounts, senddispls,
        &mtxdistfile->data, 0, recvcounts, recvdispls,
        comm, disterr);
    if (err) {
        mtxfiledata_free(
            &senddata, mtxdistfile->header.object, mtxdistfile->header.format,
            mtxdistfile->header.field, mtxdistfile->precision);
        free(sendperm);
        free(sendcounts);
        return err;
    }
    mtxfiledata_free(
        &senddata, mtxdistfile->header.object, mtxdistfile->header.format,
        mtxdistfile->header.field, mtxdistfile->precision);

    disterr->mpierrcode = MPI_Alltoallv(
        sendperm, sendcounts, senddispls, MPI_INT64_T,
        perm, recvcounts, recvdispls, MPI_INT64_T, comm);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(sendperm);
        free(sendcounts);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    free(sendperm);
    free(sendcounts);

    /* 6. Permute the received data lines. */
    err = mtxpartition_localidx(
        &mtxdistfile->partition, rank, local_size, perm, perm);
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;
    for (int64_t k = 0; k < local_size; k++)
        perm[k]++;
    err = mtxfiledata_permute(
        &mtxdistfile->data,
        mtxdistfile->header.object, mtxdistfile->header.format,
        mtxdistfile->header.field, mtxdistfile->precision,
        mtxdistfile->size.num_rows, mtxdistfile->size.num_columns,
        local_size, perm);
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistfile_sort_keys()’ sorts a distributed Matrix Market file
 * according to the given keys. The data lines are redistributed among
 * the participating processes, if necessary.
 *
 * The array ‘keys’ must be an array of length at least equal to the
 * number of data lines residing on the current MPI process (i.e.,
 * ‘mtxdistfile->partition.part_sizes[p]’, where ‘p’ is the rank of
 * the current process). This array contains 64-bit integer keys that
 * are used to determine the sorting order.
 *
 * ‘perm’ is ignored if it is ‘NULL’. Otherwise, it must point to an
 * array of ‘size’ 64-bit integers, and it is used to store the
 * permutation of the vector or matrix nonzeros.
 */
static int mtxdistfile_sort_keys(
    struct mtxdistfile * mtxdistfile,
    int64_t size,
    int64_t * keys,
    int64_t * perm,
    struct mtxdisterror * disterr)
{
    int err;
    MPI_Comm comm = mtxdistfile->comm;

    /* 1. Sort the keys and obtain a sorting permutation. */
    bool alloc_perm = !perm;
    if (alloc_perm) {
        perm = malloc(size * sizeof(int64_t));
        err = !perm ? MTX_ERR_ERRNO : MTX_SUCCESS;
        if (mtxdisterror_allreduce(disterr, err))
            return MTX_ERR_MPI_COLLECTIVE;
    }
    err = distradix_sort_uint64(
        size, keys, perm, comm, disterr);
    if (err) {
        if (alloc_perm)
            free(perm);
        return err;
    }

    /* Switch from 0- to 1-based indexing. */
    for (int64_t k = 0; k < size; k++)
        perm[k]++;

    /* 2. Sort nonzeros according to the sorting permutation. */
    err = mtxdistfile_sort_permutation(
        mtxdistfile, perm, disterr);
    if (err) {
        if (alloc_perm)
            free(perm);
        return err;
    }
    if (alloc_perm)
        free(perm);
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistfile_sort()’ sorts a distributed Matrix Market file in a
 * given order.
 *
 * The sorting order is determined by ‘sorting’. If the sorting order
 * is ‘mtxfile_unsorted’, nothing is done. If the sorting order is
 * ‘mtxfile_permutation’, then ‘perm’ must point to an array of ‘size’
 * integers that specify the sorting permutation. Note that the
 * sorting permutation uses 1-based indexing.
 *
 * For a vector or matrix in coordinate format, the nonzero values are
 * sorted in the specified order. For Matrix Market files in array
 * format, this operation does nothing.
 *
 * ‘size’ is the number of vector or matrix nonzeros to sort.
 *
 * ‘perm’ is ignored if it is ‘NULL’. Otherwise, it must point to an
 * array of ‘size’ 64-bit integers, and it is used to store the
 * permutation of the vector or matrix nonzeros.
 */
int mtxdistfile_sort(
    struct mtxdistfile * mtxdistfile,
    enum mtxfilesorting sorting,
    int64_t permsize,
    int64_t * perm,
    struct mtxdisterror * disterr)
{
    int err;
    MPI_Comm comm = mtxdistfile->comm;
    int comm_size = mtxdistfile->comm_size;
    int rank = mtxdistfile->rank;

    int64_t size = mtxdistfile->partition.size;
    int local_size = rank < mtxdistfile->partition.num_parts
        ? mtxdistfile->partition.part_sizes[rank] : 0;
    int64_t offset = rank < mtxdistfile->partition.num_parts
        ? mtxdistfile->partition.parts_ptr[rank] : 0;

    if (sorting == mtxfile_unsorted) {
        if (!perm)
            return MTX_SUCCESS;

        err = (permsize < local_size)
            ? MTX_ERR_INDEX_OUT_OF_BOUNDS : MTX_SUCCESS;
        if (mtxdisterror_allreduce(disterr, err))
            return MTX_ERR_MPI_COLLECTIVE;
        int64_t global_offset = mtxdistfile->partition.parts_ptr[rank];
        for (int64_t k = 0; k < local_size; k++)
            perm[k] = global_offset+k+1;
        return MTX_SUCCESS;
    } else if (sorting == mtxfile_permutation) {
        return mtxdistfile_sort_permutation(
            mtxdistfile, perm, disterr);
    } else if (sorting == mtxfile_row_major) {
        int64_t * keys = malloc(local_size * sizeof(int64_t));
        err = !keys ? MTX_ERR_ERRNO : MTX_SUCCESS;
        if (mtxdisterror_allreduce(disterr, err))
            return MTX_ERR_MPI_COLLECTIVE;
        err = mtxfiledata_sortkey_row_major(
            &mtxdistfile->data,
            mtxdistfile->header.object, mtxdistfile->header.format,
            mtxdistfile->header.field, mtxdistfile->precision,
            mtxdistfile->size.num_rows, mtxdistfile->size.num_columns,
            offset, local_size, keys);
        if (mtxdisterror_allreduce(disterr, err)) {
            free(keys);
            return MTX_ERR_MPI_COLLECTIVE;
        }
        err = mtxdistfile_sort_keys(
            mtxdistfile, local_size, keys, perm, disterr);
        if (err) {
            free(keys);
            return err;
        }
        free(keys);

    } else if (sorting == mtxfile_column_major) {
        int64_t * keys = malloc(local_size * sizeof(int64_t));
        err = !keys ? MTX_ERR_ERRNO : MTX_SUCCESS;
        if (mtxdisterror_allreduce(disterr, err))
            return MTX_ERR_MPI_COLLECTIVE;
        err = mtxfiledata_sortkey_column_major(
            &mtxdistfile->data,
            mtxdistfile->header.object, mtxdistfile->header.format,
            mtxdistfile->header.field, mtxdistfile->precision,
            mtxdistfile->size.num_rows, mtxdistfile->size.num_columns,
            offset, local_size, keys);
        if (mtxdisterror_allreduce(disterr, err)) {
            free(keys);
            return MTX_ERR_MPI_COLLECTIVE;
        }
        err = mtxdistfile_sort_keys(
            mtxdistfile, local_size, keys, perm, disterr);
        if (err) {
            free(keys);
            return err;
        }
        free(keys);

    } else if (sorting == mtxfile_morton) {
        int64_t * keys = malloc(local_size * sizeof(int64_t));
        err = !keys ? MTX_ERR_ERRNO : MTX_SUCCESS;
        if (mtxdisterror_allreduce(disterr, err))
            return MTX_ERR_MPI_COLLECTIVE;
        err = mtxfiledata_sortkey_morton(
            &mtxdistfile->data,
            mtxdistfile->header.object, mtxdistfile->header.format,
            mtxdistfile->header.field, mtxdistfile->precision,
            mtxdistfile->size.num_rows, mtxdistfile->size.num_columns,
            offset, local_size, keys);
        if (mtxdisterror_allreduce(disterr, err)) {
            free(keys);
            return MTX_ERR_MPI_COLLECTIVE;
        }
        err = mtxdistfile_sort_keys(
            mtxdistfile, local_size, keys, perm, disterr);
        if (err) {
            free(keys);
            return err;
        }
        free(keys);

    } else {
        return MTX_ERR_INVALID_MTX_SORTING;
    }
    return MTX_SUCCESS;
}

/*
 * Partitioning
 */

/**
 * ‘mtxdistfile_partition()’ partitions and redistributes the entries
 * of a distributed Matrix Market file according to the given row and
 * column partitions.
 *
 * The partitions ‘rowpart’ or ‘colpart’ are allowed to be ‘NULL’, in
 * which case a trivial, singleton partition is used for the rows or
 * columns, respectively.
 *
 * Otherwise, ‘rowpart’ and ‘colpart’ must partition the rows and
 * columns of the matrix or vector ‘src’, respectively. That is,
 * ‘rowpart->size’ must be equal to ‘src->size.num_rows’, and
 * ‘colpart->size’ must be equal to ‘src->size.num_columns’.
 *
 * The argument ‘dsts’ is an array that must have enough storage for
 * ‘P*Q’ values of type ‘struct mtxfile’, where ‘P’ is the number of
 * row parts, ‘rowpart->num_parts’, and ‘Q’ is the number of column
 * parts, ‘colpart->num_parts’. Note that the ‘r’th part corresponds
 * to a row part ‘p’ and column part ‘q’, such that ‘r=p*Q+q’. Thus,
 * the ‘r’th entry of ‘dsts’ is the submatrix corresponding to the
 * ‘p’th row and ‘q’th column of the 2D partitioning.
 *
 * The user is responsible for freeing storage allocated for each
 * Matrix Market file in the ‘dsts’ array.
 */
int mtxdistfile_partition(
    const struct mtxdistfile * src,
    struct mtxdistfile * dsts,
    const struct mtxpartition * rowpart,
    const struct mtxpartition * colpart,
    struct mtxdisterror * disterr)
{
    int err;
    int comm_size = src->comm_size;
    int rank = src->rank;
    int num_row_parts = rowpart ? rowpart->num_parts : 1;
    int num_col_parts = colpart ? colpart->num_parts : 1;
    if (rowpart && rowpart->size != src->size.num_rows)
        return MTX_ERR_INCOMPATIBLE_PARTITION;
    if (colpart && colpart->size != src->size.num_columns)
        return MTX_ERR_INCOMPATIBLE_PARTITION;

    int local_size = rank < src->partition.num_parts
        ? src->partition.part_sizes[rank] : 0;
    int64_t offset = rank < src->partition.num_parts
        ? src->partition.parts_ptr[rank] : 0;
    int * part_per_data_line = malloc(local_size * sizeof(int));
    err = !part_per_data_line ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;

    /* 1. Assign each data line to its row and column part */
    err = mtxfiledata_partition(
        &src->data, src->header.object, src->header.format,
        src->header.field, src->precision,
        src->size.num_rows, src->size.num_columns,
        offset, local_size, rowpart, colpart, part_per_data_line);
    if (mtxdisterror_allreduce(disterr, err)) {
        free(part_per_data_line);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    /* 2. Create index arrays of data lines belonging to each part */
    int num_parts = num_row_parts * num_col_parts;
    int64_t * data_lines_per_part_ptr = malloc((num_parts+1) * sizeof(int64_t));
    err = !data_lines_per_part_ptr ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(part_per_data_line);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    int64_t * data_lines_per_part = malloc(local_size * sizeof(int64_t));
    err = !data_lines_per_part ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(data_lines_per_part_ptr);
        free(part_per_data_line);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    for (int p = 0; p <= num_parts; p++)
        data_lines_per_part_ptr[p] = 0;
    for (int64_t k = 0; k < local_size; k++) {
        int part = part_per_data_line[k];
        data_lines_per_part_ptr[part+1]++;
    }
    for (int p = 0; p < num_parts; p++) {
        data_lines_per_part_ptr[p+1] +=
            data_lines_per_part_ptr[p];
    }
    for (int64_t k = 0; k < local_size; k++) {
        int part = part_per_data_line[k];
        int64_t l = data_lines_per_part_ptr[part];
        data_lines_per_part[l] = k;
        data_lines_per_part_ptr[part]++;
    }
    for (int p = num_parts-1; p >= 0; p--) {
        data_lines_per_part_ptr[p+1] =
            data_lines_per_part_ptr[p];
    }
    data_lines_per_part_ptr[0] = 0;
    free(part_per_data_line);

    int64_t * local_part_sizes = malloc(src->comm_size * sizeof(int64_t));
    err = !local_part_sizes ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(data_lines_per_part);
        free(data_lines_per_part_ptr);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    /* 3. Create a submatrix for each part of the partition */
    for (int p = 0; p < num_row_parts; p++) {
        for (int q = 0; q < num_col_parts; q++) {
            int r = p * num_col_parts + q;
            int64_t local_part_size = data_lines_per_part_ptr[r+1] -
                data_lines_per_part_ptr[r];

            /* Gather the number of nonzeros in each part */
            disterr->mpierrcode = MPI_Allgather(
                &local_part_size, 1, MPI_INT64_T,
                local_part_sizes, 1, MPI_INT64_T, src->comm);
            err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
            if (mtxdisterror_allreduce(disterr, err)) {
                for (int s = r-1; s >= 0; s--)
                    mtxdistfile_free(&dsts[s]);
                free(local_part_sizes);
                free(data_lines_per_part);
                free(data_lines_per_part_ptr);
                return MTX_ERR_MPI_COLLECTIVE;
            }
            int64_t N = 0;
            for (int p = 0; p < src->comm_size; p++)
                N += local_part_sizes[p];

            struct mtxfilesize size;
            if (src->size.num_nonzeros >= 0) {
                size.num_rows = src->size.num_rows;
                size.num_columns = src->size.num_columns;
                size.num_nonzeros = N;
            } else if (src->size.num_columns > 0) {
                size.num_rows = rowpart
                    ? rowpart->part_sizes[p] : src->size.num_rows;
                size.num_columns = colpart
                    ? colpart->part_sizes[q] : src->size.num_columns;
                size.num_nonzeros = -1;
            } else if (src->size.num_rows >= 0) {
                size.num_rows = rowpart
                    ? rowpart->part_sizes[p] : src->size.num_rows;
                size.num_columns = -1;
                size.num_nonzeros = -1;
            } else {
                err = MTX_ERR_INVALID_MTX_SIZE;
            }
            if (mtxdisterror_allreduce(disterr, err)) {
                for (int s = r-1; s >= 0; s--)
                    mtxdistfile_free(&dsts[s]);
                free(local_part_sizes);
                free(data_lines_per_part);
                free(data_lines_per_part_ptr);
                return MTX_ERR_MPI_COLLECTIVE;
            }

            struct mtxpartition partition;
            err = mtxpartition_init_block(
                &partition, N, src->partition.num_parts, local_part_sizes);
            if (mtxdisterror_allreduce(disterr, err)) {
                for (int s = r-1; s >= 0; s--)
                    mtxdistfile_free(&dsts[s]);
                free(local_part_sizes);
                free(data_lines_per_part);
                free(data_lines_per_part_ptr);
                return MTX_ERR_MPI_COLLECTIVE;
            }

            err = mtxdistfile_alloc(
                &dsts[r], &src->header, &src->comments,
                &size, src->precision, &partition,
                src->comm, disterr);
            if (mtxdisterror_allreduce(disterr, err)) {
                mtxpartition_free(&partition);
                for (int s = r-1; s >= 0; s--)
                    mtxdistfile_free(&dsts[s]);
                free(local_part_sizes);
                free(data_lines_per_part);
                free(data_lines_per_part_ptr);
                return MTX_ERR_MPI_COLLECTIVE;
            }
            mtxpartition_free(&partition);

            const int64_t * srcdispls =
                &data_lines_per_part[data_lines_per_part_ptr[r]];
            err = mtxfiledata_copy_gather(
                &dsts[r].data, &src->data,
                dsts[r].header.object, dsts[r].header.format,
                dsts[r].header.field, dsts[r].precision,
                local_part_size, 0, srcdispls);
            if (mtxdisterror_allreduce(disterr, err)) {
                for (int s = r; s >= 0; s--)
                    mtxdistfile_free(&dsts[s]);
                free(local_part_sizes);
                free(data_lines_per_part);
                free(data_lines_per_part_ptr);
                return MTX_ERR_MPI_COLLECTIVE;
            }
        }
    }

    free(local_part_sizes);
    free(data_lines_per_part);
    free(data_lines_per_part_ptr);
    return MTX_SUCCESS;
}

#endif
