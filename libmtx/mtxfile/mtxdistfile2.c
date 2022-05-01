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
 * Last modified: 2022-05-01
 *
 * Matrix Market files distributed among multiple processes with MPI
 * for inter-process communication.
 */

#include <libmtx/libmtx-config.h>

#include <libmtx/error.h>
#include <libmtx/precision.h>
#include <libmtx/mtxfile/mtxdistfile2.h>
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

#include <limits.h>
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
 * ‘mtxdistfile2_free()’ frees storage allocated for a distributed
 * Matrix Market file.
 */
void mtxdistfile2_free(
    struct mtxdistfile2 * mtxdistfile2)
{
    mtxfiledata_free(
        &mtxdistfile2->data,
        mtxdistfile2->header.object,
        mtxdistfile2->header.format,
        mtxdistfile2->header.field,
        mtxdistfile2->precision);
    mtxfilecomments_free(&mtxdistfile2->comments);
}

/**
 * ‘mtxdistfile2_alloc()’ allocates storage for a distributed Matrix
 * Market file with the given header line, comment lines, size line
 * and precision.
 *
 * ‘comments’ may be ‘NULL’, in which case it is ignored.
 *
 * ‘localdatasize’ is the number of entries in the underlying Matrix
 * Market file that are stored on the current process.
 *
 * ‘comm’ must be the same MPI communicator that was used to create
 * ‘disterr’.
 */
int mtxdistfile2_alloc(
    struct mtxdistfile2 * mtxdistfile2,
    const struct mtxfileheader * header,
    const struct mtxfilecomments * comments,
    const struct mtxfilesize * size,
    enum mtxprecision precision,
    int64_t localdatasize,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = MTX_SUCCESS;
    int comm_size;
    disterr->mpierrcode = MPI_Comm_size(comm, &comm_size);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    int rank;
    disterr->mpierrcode = MPI_Comm_rank(comm, &rank);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;

    /* check that the partition is compatible */
    int64_t num_data_lines;
    err = mtxfilesize_num_data_lines(size, header->symmetry, &num_data_lines);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    int64_t datasize;
    disterr->mpierrcode = MPI_Allreduce(
        &localdatasize, &datasize, 1, MPI_INT64_T, MPI_SUM, comm);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (datasize != num_data_lines) return MTX_ERR_INCOMPATIBLE_PARTITION;

    mtxdistfile2->comm = comm;
    mtxdistfile2->comm_size = comm_size;
    mtxdistfile2->rank = rank;

    err = mtxfileheader_copy(&mtxdistfile2->header, header);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    if (comments) {
        err = mtxfilecomments_copy(&mtxdistfile2->comments, comments);
        if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    } else {
        err = mtxfilecomments_init(&mtxdistfile2->comments);
        if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    }
    err = mtxfilesize_copy(&mtxdistfile2->size, size);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxfilecomments_free(&mtxdistfile2->comments);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    mtxdistfile2->precision = precision;
    mtxdistfile2->datasize = datasize;
    mtxdistfile2->localdatasize = localdatasize;
    err = mtxfiledata_alloc(
        &mtxdistfile2->data, mtxdistfile2->header.object, mtxdistfile2->header.format,
        mtxdistfile2->header.field, mtxdistfile2->precision, mtxdistfile2->localdatasize);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxfilecomments_free(&mtxdistfile2->comments);
        return err;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistfile2_alloc_copy()’ allocates storage for a copy of a Matrix
 * Market file without initialising the underlying values.
 */
int mtxdistfile2_alloc_copy(
    struct mtxdistfile2 * dst,
    const struct mtxdistfile2 * src,
    struct mtxdisterror * disterr)
{
    int err;
    dst->comm = src->comm;
    dst->comm_size = src->comm_size;
    dst->rank = src->rank;
    err = mtxfileheader_copy(&dst->header, &src->header);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    err = mtxfilecomments_copy(&dst->comments, &src->comments);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    err = mtxfilesize_copy(&dst->size, &src->size);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxfilecomments_free(&dst->comments);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    dst->precision = src->precision;
    dst->datasize = src->datasize;
    dst->localdatasize = src->localdatasize;

    err = mtxfiledata_alloc(
        &dst->data, dst->header.object, dst->header.format,
        dst->header.field, dst->precision, dst->localdatasize);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxfilecomments_free(&dst->comments);
        return err;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistfile2_init_copy()’ creates a copy of a Matrix Market file.
 */
int mtxdistfile2_init_copy(
    struct mtxdistfile2 * dst,
    const struct mtxdistfile2 * src,
    struct mtxdisterror * disterr)
{
    int err = mtxdistfile2_alloc_copy(dst, src, disterr);
    if (err) return err;
    err = mtxfiledata_copy(
        &dst->data, &src->data,
        src->header.object, src->header.format,
        src->header.field, src->precision,
        src->localdatasize, 0, 0);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxdistfile2_free(dst);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    return MTX_SUCCESS;
}

/*
 * Matrix array formats
 */

/**
 * ‘mtxdistfile2_alloc_matrix_array()’ allocates a distributed matrix
 * in array format.
 */
int mtxdistfile2_alloc_matrix_array(
    struct mtxdistfile2 * mtxdistfile2,
    enum mtxfilefield field,
    enum mtxfilesymmetry symmetry,
    enum mtxprecision precision,
    int64_t num_rows,
    int64_t num_columns,
    int64_t localdatasize,
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

    mtxdistfile2->comm = comm;
    mtxdistfile2->comm_size = comm_size;
    mtxdistfile2->rank = rank;
    mtxdistfile2->header.object = mtxfile_matrix;
    mtxdistfile2->header.format = mtxfile_array;
    mtxdistfile2->header.field = field;
    mtxdistfile2->header.symmetry = symmetry;
    mtxfilecomments_init(&mtxdistfile2->comments);
    mtxdistfile2->size.num_rows = num_rows;
    mtxdistfile2->size.num_columns = num_columns;
    mtxdistfile2->size.num_nonzeros = -1;

    /* check that the partition is compatible */
    int64_t num_data_lines;
    err = mtxfilesize_num_data_lines(
        &mtxdistfile2->size, symmetry, &num_data_lines);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    int64_t datasize;
    disterr->mpierrcode = MPI_Allreduce(
        &localdatasize, &datasize, 1, MPI_INT64_T, MPI_SUM, comm);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (datasize != num_data_lines) return MTX_ERR_INCOMPATIBLE_PARTITION;

    mtxdistfile2->precision = precision;
    mtxdistfile2->datasize = datasize;
    mtxdistfile2->localdatasize = localdatasize;
    err = mtxfiledata_alloc(
        &mtxdistfile2->data, mtxdistfile2->header.object, mtxdistfile2->header.format,
        mtxdistfile2->header.field, mtxdistfile2->precision, mtxdistfile2->localdatasize);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxfilecomments_free(&mtxdistfile2->comments);
        return err;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistfile2_init_matrix_array_real_single()’ allocates and
 * initialises a distributed matrix in array format with real, single
 * precision coefficients.
 */
int mtxdistfile2_init_matrix_array_real_single(
    struct mtxdistfile2 * mtxdistfile2,
    enum mtxfilesymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    int64_t localdatasize,
    const float * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxdistfile2_alloc_matrix_array(
        mtxdistfile2, mtxfile_real, symmetry, mtx_single,
        num_rows, num_columns, localdatasize, comm, disterr);
    if (err) return err;
    memcpy(mtxdistfile2->data.array_real_single, data,
           mtxdistfile2->localdatasize * sizeof(*data));
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistfile2_init_matrix_array_real_double()’ allocates and
 * initialises a distributed matrix in array format with real, double
 * precision coefficients.
 */
int mtxdistfile2_init_matrix_array_real_double(
    struct mtxdistfile2 * mtxdistfile2,
    enum mtxfilesymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    int64_t localdatasize,
    const double * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxdistfile2_alloc_matrix_array(
        mtxdistfile2, mtxfile_real, symmetry, mtx_double,
        num_rows, num_columns, localdatasize, comm, disterr);
    if (err) return err;
    memcpy(mtxdistfile2->data.array_real_double, data,
           mtxdistfile2->localdatasize * sizeof(*data));
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistfile2_init_matrix_array_complex_single()’ allocates and
 * initialises a distributed matrix in array format with complex,
 * single precision coefficients.
 */
int mtxdistfile2_init_matrix_array_complex_single(
    struct mtxdistfile2 * mtxdistfile2,
    enum mtxfilesymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    int64_t localdatasize,
    const float (* data)[2],
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistfile2_init_matrix_array_complex_double()’ allocates and
 * initialises a matrix in array format with complex, double precision
 * coefficients.
 */
int mtxdistfile2_init_matrix_array_complex_double(
    struct mtxdistfile2 * mtxdistfile2,
    enum mtxfilesymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    int64_t localdatasize,
    const double (* data)[2],
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistfile2_init_matrix_array_integer_single()’ allocates and
 * initialises a distributed matrix in array format with integer,
 * single precision coefficients.
 */
int mtxdistfile2_init_matrix_array_integer_single(
    struct mtxdistfile2 * mtxdistfile2,
    enum mtxfilesymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    int64_t localdatasize,
    const int32_t * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistfile2_init_matrix_array_integer_double()’ allocates and
 * initialises a matrix in array format with integer, double precision
 * coefficients.
 */
int mtxdistfile2_init_matrix_array_integer_double(
    struct mtxdistfile2 * mtxdistfile2,
    enum mtxfilesymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    int64_t localdatasize,
    const int64_t * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/*
 * Vector array formats
 */

/**
 * ‘mtxdistfile2_alloc_vector_array()’ allocates a distributed vector
 * in array format.
 */
int mtxdistfile2_alloc_vector_array(
    struct mtxdistfile2 * mtxdistfile2,
    enum mtxfilefield field,
    enum mtxprecision precision,
    int64_t num_rows,
    int64_t localdatasize,
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

    mtxdistfile2->comm = comm;
    mtxdistfile2->comm_size = comm_size;
    mtxdistfile2->rank = rank;

    mtxdistfile2->header.object = mtxfile_vector;
    mtxdistfile2->header.format = mtxfile_array;
    mtxdistfile2->header.field = field;
    mtxdistfile2->header.symmetry = mtxfile_general;
    mtxfilecomments_init(&mtxdistfile2->comments);
    mtxdistfile2->size.num_rows = num_rows;
    mtxdistfile2->size.num_columns = -1;
    mtxdistfile2->size.num_nonzeros = -1;

    /* check that the partition is compatible */
    int64_t num_data_lines;
    err = mtxfilesize_num_data_lines(
        &mtxdistfile2->size, mtxfile_general, &num_data_lines);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    int64_t datasize;
    disterr->mpierrcode = MPI_Allreduce(
        &localdatasize, &datasize, 1, MPI_INT64_T, MPI_SUM, comm);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (datasize != num_data_lines) return MTX_ERR_INCOMPATIBLE_PARTITION;

    mtxdistfile2->precision = precision;
    mtxdistfile2->datasize = datasize;
    mtxdistfile2->localdatasize = localdatasize;

    err = mtxfiledata_alloc(
        &mtxdistfile2->data, mtxdistfile2->header.object, mtxdistfile2->header.format,
        mtxdistfile2->header.field, mtxdistfile2->precision, mtxdistfile2->localdatasize);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxfilecomments_free(&mtxdistfile2->comments);
        return err;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistfile2_init_vector_array_real_single()’ allocates and
 * initialises a distributed vector in array format with real, single
 * precision coefficients.
 */
int mtxdistfile2_init_vector_array_real_single(
    struct mtxdistfile2 * mtxdistfile2,
    int64_t num_rows,
    int64_t localdatasize,
    const float * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxdistfile2_alloc_vector_array(
        mtxdistfile2, mtxfile_real, mtx_single, num_rows,
        localdatasize, comm, disterr);
    if (err) return err;
    memcpy(mtxdistfile2->data.array_real_single, data,
           mtxdistfile2->localdatasize * sizeof(*data));
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistfile2_init_vector_array_real_double()’ allocates and initialises
 * a vector in array format with real, double precision coefficients.
 */
int mtxdistfile2_init_vector_array_real_double(
    struct mtxdistfile2 * mtxdistfile2,
    int64_t num_rows,
    int64_t localdatasize,
    const double * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxdistfile2_alloc_vector_array(
        mtxdistfile2, mtxfile_real, mtx_double, num_rows,
        localdatasize, comm, disterr);
    if (err) return err;
    memcpy(mtxdistfile2->data.array_real_double, data,
           mtxdistfile2->localdatasize * sizeof(*data));
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistfile2_init_vector_array_complex_single()’ allocates and
 * initialises a distributed vector in array format with complex,
 * single precision coefficients.
 */
int mtxdistfile2_init_vector_array_complex_single(
    struct mtxdistfile2 * mtxdistfile2,
    int64_t num_rows,
    int64_t localdatasize,
    const float (* data)[2],
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxdistfile2_alloc_vector_array(
        mtxdistfile2, mtxfile_complex, mtx_single, num_rows,
        localdatasize, comm, disterr);
    if (err) return err;
    memcpy(mtxdistfile2->data.array_complex_single, data,
           mtxdistfile2->localdatasize * sizeof(*data));
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistfile2_init_vector_array_complex_double()’ allocates and
 * initialises a vector in array format with complex, double precision
 * coefficients.
 */
int mtxdistfile2_init_vector_array_complex_double(
    struct mtxdistfile2 * mtxdistfile2,
    int64_t num_rows,
    int64_t localdatasize,
    const double (* data)[2],
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxdistfile2_alloc_vector_array(
        mtxdistfile2, mtxfile_complex, mtx_double, num_rows,
        localdatasize, comm, disterr);
    if (err) return err;
    memcpy(mtxdistfile2->data.array_complex_double, data,
           mtxdistfile2->localdatasize * sizeof(*data));
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistfile2_init_vector_array_integer_single()’ allocates and
 * initialises a distributed vector in array format with integer,
 * single precision coefficients.
 */
int mtxdistfile2_init_vector_array_integer_single(
    struct mtxdistfile2 * mtxdistfile2,
    int64_t num_rows,
    int64_t localdatasize,
    const int32_t * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxdistfile2_alloc_vector_array(
        mtxdistfile2, mtxfile_integer, mtx_single, num_rows,
        localdatasize, comm, disterr);
    if (err) return err;
    memcpy(mtxdistfile2->data.array_integer_single, data,
           mtxdistfile2->localdatasize * sizeof(*data));
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistfile2_init_vector_array_integer_double()’ allocates and
 * initialises a vector in array format with integer, double precision
 * coefficients.
 */
int mtxdistfile2_init_vector_array_integer_double(
    struct mtxdistfile2 * mtxdistfile2,
    int64_t num_rows,
    int64_t localdatasize,
    const int64_t * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxdistfile2_alloc_vector_array(
        mtxdistfile2, mtxfile_integer, mtx_double, num_rows,
        localdatasize, comm, disterr);
    if (err) return err;
    memcpy(mtxdistfile2->data.array_integer_double, data,
           mtxdistfile2->localdatasize * sizeof(*data));
    return MTX_SUCCESS;
}

/*
 * Matrix coordinate formats
 */

/**
 * ‘mtxdistfile2_alloc_matrix_coordinate()’ allocates a distributed
 * matrix in coordinate format.
 */
int mtxdistfile2_alloc_matrix_coordinate(
    struct mtxdistfile2 * mtxdistfile2,
    enum mtxfilefield field,
    enum mtxfilesymmetry symmetry,
    enum mtxprecision precision,
    int64_t num_rows,
    int64_t num_columns,
    int64_t num_nonzeros,
    int64_t localdatasize,
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

    mtxdistfile2->comm = comm;
    mtxdistfile2->comm_size = comm_size;
    mtxdistfile2->rank = rank;
    mtxdistfile2->header.object = mtxfile_matrix;
    mtxdistfile2->header.format = mtxfile_coordinate;
    mtxdistfile2->header.field = field;
    mtxdistfile2->header.symmetry = symmetry;
    mtxfilecomments_init(&mtxdistfile2->comments);
    mtxdistfile2->size.num_rows = num_rows;
    mtxdistfile2->size.num_columns = num_columns;
    mtxdistfile2->size.num_nonzeros = num_nonzeros;
    mtxdistfile2->precision = precision;

    /* check that the partition is compatible */
    int64_t num_data_lines;
    err = mtxfilesize_num_data_lines(
        &mtxdistfile2->size, symmetry, &num_data_lines);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    int64_t datasize;
    disterr->mpierrcode = MPI_Allreduce(
        &localdatasize, &datasize, 1, MPI_INT64_T, MPI_SUM, comm);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (datasize != num_data_lines) return MTX_ERR_INCOMPATIBLE_PARTITION;

    mtxdistfile2->precision = precision;
    mtxdistfile2->datasize = datasize;
    mtxdistfile2->localdatasize = localdatasize;

    err = mtxfiledata_alloc(
        &mtxdistfile2->data, mtxdistfile2->header.object, mtxdistfile2->header.format,
        mtxdistfile2->header.field, mtxdistfile2->precision, mtxdistfile2->localdatasize);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxfilecomments_free(&mtxdistfile2->comments);
        return err;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistfile2_init_matrix_coordinate_real_single()’ allocates and
 * initialises a distributed matrix in coordinate format with real,
 * single precision coefficients.
 */
int mtxdistfile2_init_matrix_coordinate_real_single(
    struct mtxdistfile2 * mtxdistfile2,
    enum mtxfilesymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    int64_t num_nonzeros,
    int64_t localdatasize,
    const struct mtxfile_matrix_coordinate_real_single * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxdistfile2_alloc_matrix_coordinate(
        mtxdistfile2, mtxfile_real, symmetry, mtx_single,
        num_rows, num_columns, num_nonzeros,
        localdatasize, comm, disterr);
    if (err) return err;
    memcpy(mtxdistfile2->data.matrix_coordinate_real_single, data,
           mtxdistfile2->localdatasize * sizeof(*data));
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistfile2_init_matrix_coordinate_real_double()’ allocates and
 * initialises a matrix in coordinate format with real, double
 * precision coefficients.
 */
int mtxdistfile2_init_matrix_coordinate_real_double(
    struct mtxdistfile2 * mtxdistfile2,
    enum mtxfilesymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    int64_t num_nonzeros,
    int64_t localdatasize,
    const struct mtxfile_matrix_coordinate_real_double * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxdistfile2_alloc_matrix_coordinate(
        mtxdistfile2, mtxfile_real, symmetry, mtx_double,
        num_rows, num_columns, num_nonzeros,
        localdatasize, comm, disterr);
    if (err) return err;
    memcpy(mtxdistfile2->data.matrix_coordinate_real_double, data,
           mtxdistfile2->localdatasize * sizeof(*data));
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistfile2_init_matrix_coordinate_complex_single()’ allocates and
 * initialises a distributed matrix in coordinate format with complex,
 * single precision coefficients.
 */
int mtxdistfile2_init_matrix_coordinate_complex_single(
    struct mtxdistfile2 * mtxdistfile2,
    enum mtxfilesymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    int64_t num_nonzeros,
    int64_t localdatasize,
    const struct mtxfile_matrix_coordinate_complex_single * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistfile2_init_matrix_coordinate_complex_double()’ allocates and
 * initialises a matrix in coordinate format with complex, double
 * precision coefficients.
 */
int mtxdistfile2_init_matrix_coordinate_complex_double(
    struct mtxdistfile2 * mtxdistfile2,
    enum mtxfilesymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    int64_t num_nonzeros,
    int64_t localdatasize,
    const struct mtxfile_matrix_coordinate_complex_double * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistfile2_init_matrix_coordinate_integer_single()’ allocates and
 * initialises a distributed matrix in coordinate format with integer,
 * single precision coefficients.
 */
int mtxdistfile2_init_matrix_coordinate_integer_single(
    struct mtxdistfile2 * mtxdistfile2,
    enum mtxfilesymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    int64_t num_nonzeros,
    int64_t localdatasize,
    const struct mtxfile_matrix_coordinate_integer_single * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistfile2_init_matrix_coordinate_integer_double()’ allocates and
 * initialises a matrix in coordinate format with integer, double
 * precision coefficients.
 */
int mtxdistfile2_init_matrix_coordinate_integer_double(
    struct mtxdistfile2 * mtxdistfile2,
    enum mtxfilesymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    int64_t num_nonzeros,
    int64_t localdatasize,
    const struct mtxfile_matrix_coordinate_integer_double * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistfile2_init_matrix_coordinate_pattern()’ allocates and
 * initialises a matrix in coordinate format with boolean (pattern)
 * precision coefficients.
 */
int mtxdistfile2_init_matrix_coordinate_pattern(
    struct mtxdistfile2 * mtxdistfile2,
    enum mtxfilesymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    int64_t num_nonzeros,
    int64_t localdatasize,
    const struct mtxfile_matrix_coordinate_pattern * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/*
 * Vector coordinate formats
 */

/**
 * ‘mtxdistfile2_alloc_vector_coordinate()’ allocates a distributed
 * vector in coordinate format.
 */
int mtxdistfile2_alloc_vector_coordinate(
    struct mtxdistfile2 * mtxdistfile2,
    enum mtxfilefield field,
    enum mtxprecision precision,
    int64_t num_rows,
    int64_t num_nonzeros,
    int64_t localdatasize,
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

    mtxdistfile2->comm = comm;
    mtxdistfile2->comm_size = comm_size;
    mtxdistfile2->rank = rank;

    mtxdistfile2->header.object = mtxfile_vector;
    mtxdistfile2->header.format = mtxfile_coordinate;
    mtxdistfile2->header.field = field;
    mtxdistfile2->header.symmetry = mtxfile_general;
    mtxfilecomments_init(&mtxdistfile2->comments);
    mtxdistfile2->size.num_rows = num_rows;
    mtxdistfile2->size.num_columns = -1;
    mtxdistfile2->size.num_nonzeros = num_nonzeros;

    /* check that the partition is compatible */
    int64_t num_data_lines;
    err = mtxfilesize_num_data_lines(
        &mtxdistfile2->size, mtxfile_general, &num_data_lines);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    int64_t datasize;
    disterr->mpierrcode = MPI_Allreduce(
        &localdatasize, &datasize, 1, MPI_INT64_T, MPI_SUM, comm);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (datasize != num_data_lines) return MTX_ERR_INCOMPATIBLE_PARTITION;

    mtxdistfile2->precision = precision;
    mtxdistfile2->datasize = datasize;
    mtxdistfile2->localdatasize = localdatasize;

    err = mtxfiledata_alloc(
        &mtxdistfile2->data, mtxdistfile2->header.object, mtxdistfile2->header.format,
        mtxdistfile2->header.field, mtxdistfile2->precision, mtxdistfile2->localdatasize);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxfilecomments_free(&mtxdistfile2->comments);
        return err;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistfile2_init_vector_coordinate_real_single()’ allocates and
 * initialises a distributed vector in coordinate format with real,
 * single precision coefficients.
 */
int mtxdistfile2_init_vector_coordinate_real_single(
    struct mtxdistfile2 * mtxdistfile2,
    int64_t num_rows,
    int64_t num_nonzeros,
    int64_t localdatasize,
    const struct mtxfile_vector_coordinate_real_single * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err;
    err = mtxdistfile2_alloc_vector_coordinate(
        mtxdistfile2, mtxfile_real, mtx_single,
        num_rows, num_nonzeros, localdatasize, comm, disterr);
    if (err)
        return err;
    memcpy(mtxdistfile2->data.vector_coordinate_real_single, data,
           mtxdistfile2->localdatasize * sizeof(*data));
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistfile2_init_vector_coordinate_real_double()’ allocates and
 * initialises a vector in coordinate format with real, double
 * precision coefficients.
 */
int mtxdistfile2_init_vector_coordinate_real_double(
    struct mtxdistfile2 * mtxdistfile2,
    int64_t num_rows,
    int64_t num_nonzeros,
    int64_t localdatasize,
    const struct mtxfile_vector_coordinate_real_double * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxdistfile2_alloc_vector_coordinate(
        mtxdistfile2, mtxfile_real, mtx_double,
        num_rows, num_nonzeros, localdatasize, comm, disterr);
    if (err) return err;
    memcpy(mtxdistfile2->data.vector_coordinate_real_double, data,
           mtxdistfile2->localdatasize * sizeof(*data));
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistfile2_init_vector_coordinate_complex_single()’ allocates and
 * initialises a distributed vector in coordinate format with complex,
 * single precision coefficients.
 */
int mtxdistfile2_init_vector_coordinate_complex_single(
    struct mtxdistfile2 * mtxdistfile2,
    int64_t num_rows,
    int64_t num_nonzeros,
    int64_t localdatasize,
    const struct mtxfile_vector_coordinate_complex_single * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistfile2_init_vector_coordinate_complex_double()’ allocates and
 * initialises a vector in coordinate format with complex, double
 * precision coefficients.
 */
int mtxdistfile2_init_vector_coordinate_complex_double(
    struct mtxdistfile2 * mtxdistfile2,
    int64_t num_rows,
    int64_t num_nonzeros,
    int64_t localdatasize,
    const struct mtxfile_vector_coordinate_complex_double * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistfile2_init_vector_coordinate_integer_single()’ allocates and
 * initialises a distributed vector in coordinate format with integer,
 * single precision coefficients.
 */
int mtxdistfile2_init_vector_coordinate_integer_single(
    struct mtxdistfile2 * mtxdistfile2,
    int64_t num_rows,
    int64_t num_nonzeros,
    int64_t localdatasize,
    const struct mtxfile_vector_coordinate_integer_single * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistfile2_init_vector_coordinate_integer_double()’ allocates and
 * initialises a vector in coordinate format with integer, double
 * precision coefficients.
 */
int mtxdistfile2_init_vector_coordinate_integer_double(
    struct mtxdistfile2 * mtxdistfile2,
    int64_t num_rows,
    int64_t num_nonzeros,
    int64_t localdatasize,
    const struct mtxfile_vector_coordinate_integer_double * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistfile2_init_vector_coordinate_pattern()’ allocates and
 * initialises a vector in coordinate format with boolean (pattern)
 * precision coefficients.
 */
int mtxdistfile2_init_vector_coordinate_pattern(
    struct mtxdistfile2 * mtxdistfile2,
    int64_t num_rows,
    int64_t num_nonzeros,
    int64_t localdatasize,
    const struct mtxfile_vector_coordinate_pattern * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/*
 * Modifying values
 */

/**
 * ‘mtxdistfile2_set_constant_real_single()’ sets every (nonzero) value
 * of a matrix or vector equal to a constant, single precision
 * floating point number.
 */
int mtxdistfile2_set_constant_real_single(
    struct mtxdistfile2 * mtxdistfile2,
    float a,
    struct mtxdisterror * disterr)
{
    int err = mtxfiledata_set_constant_real_single(
        &mtxdistfile2->data, mtxdistfile2->header.object,
        mtxdistfile2->header.format, mtxdistfile2->header.field,
        mtxdistfile2->precision, mtxdistfile2->localdatasize, 0, a);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistfile2_set_constant_real_double()’ sets every (nonzero) value
 * of a matrix or vector equal to a constant, double precision
 * floating point number.
 */
int mtxdistfile2_set_constant_real_double(
    struct mtxdistfile2 * mtxdistfile2,
    double a,
    struct mtxdisterror * disterr)
{
    int err = mtxfiledata_set_constant_real_double(
        &mtxdistfile2->data, mtxdistfile2->header.object,
        mtxdistfile2->header.format, mtxdistfile2->header.field,
        mtxdistfile2->precision, mtxdistfile2->localdatasize, 0, a);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistfile2_set_constant_complex_single()’ sets every (nonzero)
 * value of a matrix or vector equal to a constant, single precision
 * floating point complex number.
 */
int mtxdistfile2_set_constant_complex_single(
    struct mtxdistfile2 * mtxdistfile2,
    float a[2],
    struct mtxdisterror * disterr)
{
    int err = mtxfiledata_set_constant_complex_single(
        &mtxdistfile2->data, mtxdistfile2->header.object,
        mtxdistfile2->header.format, mtxdistfile2->header.field,
        mtxdistfile2->precision, mtxdistfile2->localdatasize, 0, a);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistfile2_set_constant_complex_double()’ sets every (nonzero)
 * value of a matrix or vector equal to a constant, double precision
 * floating point complex number.
 */
int mtxdistfile2_set_constant_complex_double(
    struct mtxdistfile2 * mtxdistfile2,
    double a[2],
    struct mtxdisterror * disterr)
{
    int err = mtxfiledata_set_constant_complex_double(
        &mtxdistfile2->data, mtxdistfile2->header.object,
        mtxdistfile2->header.format, mtxdistfile2->header.field,
        mtxdistfile2->precision, mtxdistfile2->localdatasize, 0, a);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistfile2_set_constant_integer_single()’ sets every (nonzero)
 * value of a matrix or vector equal to a constant, 32-bit integer.
 */
int mtxdistfile2_set_constant_integer_single(
    struct mtxdistfile2 * mtxdistfile2,
    int32_t a,
    struct mtxdisterror * disterr)
{
    int err = mtxfiledata_set_constant_integer_single(
        &mtxdistfile2->data, mtxdistfile2->header.object,
        mtxdistfile2->header.format, mtxdistfile2->header.field,
        mtxdistfile2->precision, mtxdistfile2->localdatasize, 0, a);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistfile2_set_constant_integer_double()’ sets every (nonzero)
 * value of a matrix or vector equal to a constant, 64-bit integer.
 */
int mtxdistfile2_set_constant_integer_double(
    struct mtxdistfile2 * mtxdistfile2,
    int64_t a,
    struct mtxdisterror * disterr)
{
    int err = mtxfiledata_set_constant_integer_double(
        &mtxdistfile2->data, mtxdistfile2->header.object,
        mtxdistfile2->header.format, mtxdistfile2->header.field,
        mtxdistfile2->precision, mtxdistfile2->localdatasize, 0, a);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/*
 * Convert to and from (non-distributed) Matrix Market format
 */

static int mtxdistfile2_from_mtxfile_distribute(
    struct mtxdistfile2 * dst,
    const struct mtxfile * src,
    int64_t * partsptr,
    MPI_Comm comm,
    int root,
    struct mtxdisterror * disterr)
{
    int comm_size;
    disterr->mpierrcode = MPI_Comm_size(comm, &comm_size);
    int err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    int rank;
    disterr->mpierrcode = MPI_Comm_rank(comm, &rank);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    if (root < 0 || root >= comm_size) return MTX_ERR_INDEX_OUT_OF_BOUNDS;

    dst->comm = comm;
    dst->comm_size = comm_size;
    dst->rank = rank;

    /* Broadcast the header, comments, size line and precision. */
    err = (rank == root) ? mtxfileheader_copy(
        &dst->header, &src->header) : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    err = mtxfileheader_bcast(&dst->header, root, comm, disterr);
    if (err) return err;
    err = (rank == root) ? mtxfilecomments_copy(
        &dst->comments, &src->comments) : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    err = mtxfilecomments_bcast(&dst->comments, root, comm, disterr);
    if (err) {
        if (rank == root) mtxfilecomments_free(&dst->comments);
        return err;
    }
    err = (rank == root) ? mtxfilesize_copy(&dst->size, &src->size) : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxfilecomments_free(&dst->comments);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    err = mtxfilesize_bcast(&dst->size, root, comm, disterr);
    if (err) {
        mtxfilecomments_free(&dst->comments);
        return err;
    }
    if (rank == root) dst->precision = src->precision;
    disterr->mpierrcode = MPI_Bcast(&dst->precision, 1, MPI_INT, root, comm);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxfilecomments_free(&dst->comments);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    if (rank == root) dst->datasize = src->datasize;
    disterr->mpierrcode = MPI_Bcast(&dst->datasize, 1, MPI_INT64_T, root, comm);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxfilecomments_free(&dst->comments);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    /* receive from the root process. */
    for (int p = 0; p < comm_size; p++) {
        if (p != root && rank == root) {
            int64_t localdatasize = partsptr[p+1] - partsptr[p];
            disterr->mpierrcode = MPI_Send(
                &localdatasize, 1, MPI_INT64_T, p, 0, comm);
            err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
            err = err ? err : mtxfiledata_send(
                &src->data, src->header.object, src->header.format,
                src->header.field, src->precision, localdatasize,
                partsptr[p], p, 0, comm, disterr);
        } else if (p != root && rank == p) {
            disterr->mpierrcode = MPI_Recv(
                &dst->localdatasize, 1, MPI_INT64_T, root, 0, comm,
                MPI_STATUS_IGNORE);
            err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
            err = err ? err : mtxfiledata_alloc(
                &dst->data, dst->header.object, dst->header.format,
                dst->header.field, dst->precision, dst->localdatasize);
            err = err ? err : mtxfiledata_recv(
                &dst->data, dst->header.object, dst->header.format,
                dst->header.field, dst->precision, dst->localdatasize,
                0, root, 0, comm, disterr);
        } else if (p == root && rank == root) {
            dst->localdatasize = partsptr[p+1] - partsptr[p];
            err = err ? err : mtxfiledata_alloc(
                &dst->data, dst->header.object, dst->header.format,
                dst->header.field, dst->precision, dst->localdatasize);
            err = err ? err : mtxfiledata_copy(
                &dst->data, &src->data, dst->header.object,
                dst->header.format, dst->header.field, dst->precision,
                dst->localdatasize, 0, partsptr[p]);
        }
        if (mtxdisterror_allreduce(disterr, err)) {
            if (rank < p) {
                mtxfiledata_free(
                    &dst->data, dst->header.object, dst->header.format,
                    dst->header.field, dst->precision);
            }
            return MTX_ERR_MPI_COLLECTIVE;
        }
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistfile2_from_mtxfile_rowwise()’ creates a distributed Matrix
 * Market file from a Matrix Market file stored on a single root
 * process by partitioning the underlying matrix or vector rowwise and
 * distributing the parts among processes.
 *
 * This function performs collective communication and therefore
 * requires every process in the communicator to perform matching
 * calls to this function.
 */
int mtxdistfile2_from_mtxfile_rowwise(
    struct mtxdistfile2 * dst,
    struct mtxfile * src,
    enum mtxpartitioning parttype,
    int64_t partsize,
    int64_t blksize,
    MPI_Comm comm,
    int root,
    struct mtxdisterror * disterr)
{
    int comm_size;
    disterr->mpierrcode = MPI_Comm_size(comm, &comm_size);
    int err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    int rank;
    disterr->mpierrcode = MPI_Comm_rank(comm, &rank);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    if (root < 0 || root >= comm_size) return MTX_ERR_INDEX_OUT_OF_BOUNDS;

    /* TODO: only block and block-cyclic partitioning work for now. */
    if (parttype != mtx_block && parttype != mtx_block_cyclic)
        return MTX_ERR_INVALID_PARTITION_TYPE;

    /* allocate storage for offsets to each part */
    int64_t * partsptr = malloc((comm_size+1) * sizeof(int64_t));
    err = !partsptr ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;

    /* In the case of a block partitioning, gather part sizes onto the
     * root process. */
    int64_t * partsizes = partsptr;
    if (parttype == mtx_block) {
        disterr->mpierrcode = MPI_Allgather(
            &partsize, 1, MPI_INT64_T, partsizes, 1, MPI_INT64_T, comm);
        err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
        if (mtxdisterror_allreduce(disterr, err)) {
            free(partsptr);
            return MTX_ERR_MPI_COLLECTIVE;
        }
    }

    /* allocate storage for permutation */
    int64_t * perm = rank == root ? malloc(src->datasize * sizeof(int64_t)) : NULL;
    err = rank == root && !perm ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(partsptr);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    /* Partition the Matrix Market file on the root process. */
    err = rank == root ? mtxfile_partition_rowwise(
        src, parttype, comm_size, partsizes, blksize, NULL, partsptr, perm)
        : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(perm); free(partsptr);
        return MTX_ERR_MPI_COLLECTIVE;
    }

#if 0
    /* broadcast the size of each part */
    disterr->mpierrcode = MPI_Bcast(
        &partsptr, comm_size+1, MPI_INT64_T, root, comm);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(partsptr);
        return MTX_ERR_MPI_COLLECTIVE;
    }
#endif

    err = mtxdistfile2_from_mtxfile_distribute(
        dst, src, partsptr, comm, root, disterr);
    if (mtxdisterror_allreduce(disterr, err)) {
        free(perm); free(partsptr);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    free(perm); free(partsptr);
    return MTX_SUCCESS;
}

/*
 * I/O functions
 */

/**
 * ‘mtxdistfile2_read_rowwise()’ reads a Matrix Market file from the
 * given path and distributes the data among MPI processes in a
 * communicator. The file may optionally be compressed by gzip.
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
 * Only a single root process reads from the specified stream. The
 * underlying matrix or vector is partitioned rowwise and distributed
 * among processes.
 *
 * This function performs collective communication and therefore
 * requires every process in the communicator to perform matching
 * calls to the function.
 */
int mtxdistfile2_read_rowwise(
    struct mtxdistfile2 * mtxdistfile2,
    enum mtxprecision precision,
    enum mtxpartitioning parttype,
    int64_t partsize,
    int64_t blksize,
    const char * path,
    bool gzip,
    int64_t * lines_read,
    int64_t * bytes_read,
    MPI_Comm comm,
    int root,
    struct mtxdisterror * disterr)
{
    int rank;
    disterr->mpierrcode = MPI_Comm_rank(comm, &rank);
    int err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    struct mtxfile src;
    if (rank == root)
        err = mtxfile_read(&src, precision, path, gzip, lines_read, bytes_read);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    err = mtxdistfile2_from_mtxfile_rowwise(
        mtxdistfile2, &src, parttype, partsize, blksize, comm, root, disterr);
    if (err) { if (rank == root) mtxfile_free(&src); return err; } 
    if (rank == root) mtxfile_free(&src);
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistfile2_fread_rowwise()’ reads a Matrix Market file from a
 * stream and distributes the data among MPI processes in a
 * communicator.
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
 * Only a single root process reads from the specified stream. The
 * underlying matrix or vector is partitioned rowwise and distributed
 * among processes.
 *
 * This function performs collective communication and therefore
 * requires every process in the communicator to perform matching
 * calls to the function.
 */
int mtxdistfile2_fread_rowwise(
    struct mtxdistfile2 * mtxdistfile2,
    enum mtxprecision precision,
    enum mtxpartitioning parttype,
    int64_t partsize,
    int64_t blksize,
    FILE * f,
    int64_t * lines_read,
    int64_t * bytes_read,
    size_t line_max,
    char * linebuf,
    MPI_Comm comm,
    int root,
    struct mtxdisterror * disterr)
{
    int rank;
    disterr->mpierrcode = MPI_Comm_rank(comm, &rank);
    int err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    struct mtxfile src;
    if (rank == root) {
        err = mtxfile_fread(
            &src, precision, f, lines_read, bytes_read, line_max, linebuf);
    }
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    err = mtxdistfile2_from_mtxfile_rowwise(
        mtxdistfile2, &src, parttype, partsize, blksize, comm, root, disterr);
    if (err) { if (rank == root) mtxfile_free(&src); return err; } 
    if (rank == root) mtxfile_free(&src);
    return MTX_SUCCESS;
}

#ifdef LIBMTX_HAVE_LIBZ
/**
 * ‘mtxdistfile2_gzread_rowwise()’ reads a Matrix Market file from a
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
 * Only a single root process reads from the specified stream. The
 * underlying matrix or vector is partitioned rowwise and distributed
 * among processes.
 *
 * This function performs collective communication and therefore
 * requires every process in the communicator to perform matching
 * calls to the function.
 */
int mtxdistfile2_gzread_rowwise(
    struct mtxdistfile2 * mtxdistfile2,
    enum mtxprecision precision,
    enum mtxpartitioning parttype,
    int64_t partsize,
    int64_t blksize,
    gzFile f,
    int64_t * lines_read,
    int64_t * bytes_read,
    size_t line_max,
    char * linebuf,
    MPI_Comm comm,
    int root,
    struct mtxdisterror * disterr)
{
    int rank;
    disterr->mpierrcode = MPI_Comm_rank(comm, &rank);
    int err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    struct mtxfile src;
    if (rank == root) {
        err = mtxfile_gzread(
            &src, precision, f, lines_read, bytes_read, line_max, linebuf);
    }
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    err = mtxdistfile2_from_mtxfile_rowwise(
        mtxdistfile2, &src, parttype, partsize, blksize, comm, root, disterr);
    if (err) { if (rank == root) mtxfile_free(&src); return err; } 
    if (rank == root) mtxfile_free(&src);
    return MTX_SUCCESS;
}
#endif
#endif
