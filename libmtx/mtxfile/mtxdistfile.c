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
#include <libmtx/vector/precision.h>
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
 * ‘mtxdistfile_free()’ frees storage allocated for a distributed
 * Matrix Market file.
 */
void mtxdistfile_free(
    struct mtxdistfile * mtxdistfile)
{
    free(mtxdistfile->idx);
    mtxfiledata_free(
        &mtxdistfile->data,
        mtxdistfile->header.object,
        mtxdistfile->header.format,
        mtxdistfile->header.field,
        mtxdistfile->precision);
    mtxfilecomments_free(&mtxdistfile->comments);
}

static int mtxdistfile_init_idx(
    struct mtxdistfile * mtxdistfile,
    const int64_t * idx,
    struct mtxdisterror * disterr)
{
    MPI_Comm comm = mtxdistfile->comm;
    int64_t localdatasize = mtxdistfile->localdatasize;
    mtxdistfile->idx = malloc(localdatasize * sizeof(int64_t));
    int err = !mtxdistfile->idx ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    if (idx) {
        memcpy(mtxdistfile->idx, idx,
               mtxdistfile->localdatasize * sizeof(int64_t));
    } else {
        int64_t offset = 0;
        disterr->mpierrcode = MPI_Exscan(
            &localdatasize, &offset, 1, MPI_INT64_T, MPI_SUM, comm);
        err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
        if (mtxdisterror_allreduce(disterr, err)) {
            free(mtxdistfile->idx);
            return MTX_ERR_MPI_COLLECTIVE;
        }
        for (int64_t i = 0; i < localdatasize; i++)
            mtxdistfile->idx[i] = offset+i;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistfile_alloc()’ allocates storage for a distributed Matrix
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
int mtxdistfile_alloc(
    struct mtxdistfile * mtxdistfile,
    const struct mtxfileheader * header,
    const struct mtxfilecomments * comments,
    const struct mtxfilesize * size,
    enum mtxprecision precision,
    int64_t localdatasize,
    const int64_t * idx,
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

    mtxdistfile->comm = comm;
    mtxdistfile->comm_size = comm_size;
    mtxdistfile->rank = rank;

    err = mtxfileheader_copy(&mtxdistfile->header, header);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    if (comments) {
        err = mtxfilecomments_copy(&mtxdistfile->comments, comments);
        if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    } else {
        err = mtxfilecomments_init(&mtxdistfile->comments);
        if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    }
    err = mtxfilesize_copy(&mtxdistfile->size, size);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxfilecomments_free(&mtxdistfile->comments);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    mtxdistfile->precision = precision;
    mtxdistfile->datasize = datasize;
    mtxdistfile->localdatasize = localdatasize;
    err = mtxdistfile_init_idx(mtxdistfile, idx, disterr);
    if (err) {
        mtxfilecomments_free(&mtxdistfile->comments);
        return err;
    }
    err = mtxfiledata_alloc(
        &mtxdistfile->data, mtxdistfile->header.object, mtxdistfile->header.format,
        mtxdistfile->header.field, mtxdistfile->precision, mtxdistfile->localdatasize);
    if (mtxdisterror_allreduce(disterr, err)) {
        free(mtxdistfile->idx);
        mtxfilecomments_free(&mtxdistfile->comments);
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
    err = mtxdistfile_init_idx(dst, src->idx, disterr);
    if (err) {
        mtxfilecomments_free(&dst->comments);
        return err;
    }
    err = mtxfiledata_alloc(
        &dst->data, dst->header.object, dst->header.format,
        dst->header.field, dst->precision, dst->localdatasize);
    if (mtxdisterror_allreduce(disterr, err)) {
        free(dst->idx);
        mtxfilecomments_free(&dst->comments);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistfile_init_copy()’ creates a copy of a Matrix Market file.
 */
int mtxdistfile_init_copy(
    struct mtxdistfile * dst,
    const struct mtxdistfile * src,
    struct mtxdisterror * disterr)
{
    int err = mtxdistfile_alloc_copy(dst, src, disterr);
    if (err) return err;
    err = mtxfiledata_copy(
        &dst->data, &src->data,
        src->header.object, src->header.format,
        src->header.field, src->precision,
        src->localdatasize, 0, 0);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxdistfile_free(dst);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    return MTX_SUCCESS;
}

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
    int64_t num_rows,
    int64_t num_columns,
    int64_t localdatasize,
    const int64_t * idx,
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
    mtxdistfile->header.object = mtxfile_matrix;
    mtxdistfile->header.format = mtxfile_array;
    mtxdistfile->header.field = field;
    mtxdistfile->header.symmetry = symmetry;
    mtxfilecomments_init(&mtxdistfile->comments);
    mtxdistfile->size.num_rows = num_rows;
    mtxdistfile->size.num_columns = num_columns;
    mtxdistfile->size.num_nonzeros = -1;

    /* check that the partition is compatible */
    int64_t num_data_lines;
    err = mtxfilesize_num_data_lines(
        &mtxdistfile->size, symmetry, &num_data_lines);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    int64_t datasize;
    disterr->mpierrcode = MPI_Allreduce(
        &localdatasize, &datasize, 1, MPI_INT64_T, MPI_SUM, comm);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (datasize != num_data_lines) return MTX_ERR_INCOMPATIBLE_PARTITION;

    mtxdistfile->precision = precision;
    mtxdistfile->datasize = datasize;
    mtxdistfile->localdatasize = localdatasize;
    err = mtxdistfile_init_idx(mtxdistfile, idx, disterr);
    if (err) {
        mtxfilecomments_free(&mtxdistfile->comments);
        return err;
    }
    err = mtxfiledata_alloc(
        &mtxdistfile->data, mtxdistfile->header.object, mtxdistfile->header.format,
        mtxdistfile->header.field, mtxdistfile->precision, mtxdistfile->localdatasize);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxfilecomments_free(&mtxdistfile->comments);
        return MTX_ERR_MPI_COLLECTIVE;
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
    int64_t num_rows,
    int64_t num_columns,
    int64_t localdatasize,
    const int64_t * idx,
    const float * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxdistfile_alloc_matrix_array(
        mtxdistfile, mtxfile_real, symmetry, mtx_single,
        num_rows, num_columns, localdatasize, idx, comm, disterr);
    if (err) return err;
    memcpy(mtxdistfile->data.array_real_single, data,
           mtxdistfile->localdatasize * sizeof(*data));
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
    int64_t num_rows,
    int64_t num_columns,
    int64_t localdatasize,
    const int64_t * idx,
    const double * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxdistfile_alloc_matrix_array(
        mtxdistfile, mtxfile_real, symmetry, mtx_double,
        num_rows, num_columns, localdatasize, idx, comm, disterr);
    if (err) return err;
    memcpy(mtxdistfile->data.array_real_double, data,
           mtxdistfile->localdatasize * sizeof(*data));
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
    int64_t num_rows,
    int64_t num_columns,
    int64_t localdatasize,
    const int64_t * idx,
    const float (* data)[2],
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
    int64_t num_rows,
    int64_t num_columns,
    int64_t localdatasize,
    const int64_t * idx,
    const double (* data)[2],
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
    int64_t num_rows,
    int64_t num_columns,
    int64_t localdatasize,
    const int64_t * idx,
    const int32_t * data,
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
    int64_t num_rows,
    int64_t num_columns,
    int64_t localdatasize,
    const int64_t * idx,
    const int64_t * data,
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
    int64_t num_rows,
    int64_t localdatasize,
    const int64_t * idx,
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

    mtxdistfile->header.object = mtxfile_vector;
    mtxdistfile->header.format = mtxfile_array;
    mtxdistfile->header.field = field;
    mtxdistfile->header.symmetry = mtxfile_general;
    mtxfilecomments_init(&mtxdistfile->comments);
    mtxdistfile->size.num_rows = num_rows;
    mtxdistfile->size.num_columns = -1;
    mtxdistfile->size.num_nonzeros = -1;

    /* check that the partition is compatible */
    int64_t num_data_lines;
    err = mtxfilesize_num_data_lines(
        &mtxdistfile->size, mtxfile_general, &num_data_lines);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    int64_t datasize;
    disterr->mpierrcode = MPI_Allreduce(
        &localdatasize, &datasize, 1, MPI_INT64_T, MPI_SUM, comm);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (datasize != num_data_lines) return MTX_ERR_INCOMPATIBLE_PARTITION;

    mtxdistfile->precision = precision;
    mtxdistfile->datasize = datasize;
    mtxdistfile->localdatasize = localdatasize;
    err = mtxdistfile_init_idx(mtxdistfile, idx, disterr);
    if (err) {
        mtxfilecomments_free(&mtxdistfile->comments);
        return err;
    }
    err = mtxfiledata_alloc(
        &mtxdistfile->data, mtxdistfile->header.object, mtxdistfile->header.format,
        mtxdistfile->header.field, mtxdistfile->precision, mtxdistfile->localdatasize);
    if (mtxdisterror_allreduce(disterr, err)) {
        free(mtxdistfile->idx);
        mtxfilecomments_free(&mtxdistfile->comments);
        return MTX_ERR_MPI_COLLECTIVE;
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
    int64_t num_rows,
    int64_t localdatasize,
    const int64_t * idx,
    const float * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxdistfile_alloc_vector_array(
        mtxdistfile, mtxfile_real, mtx_single, num_rows,
        localdatasize, idx, comm, disterr);
    if (err) return err;
    memcpy(mtxdistfile->data.array_real_single, data,
           mtxdistfile->localdatasize * sizeof(*data));
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistfile_init_vector_array_real_double()’ allocates and initialises
 * a vector in array format with real, double precision coefficients.
 */
int mtxdistfile_init_vector_array_real_double(
    struct mtxdistfile * mtxdistfile,
    int64_t num_rows,
    int64_t localdatasize,
    const int64_t * idx,
    const double * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxdistfile_alloc_vector_array(
        mtxdistfile, mtxfile_real, mtx_double, num_rows,
        localdatasize, idx, comm, disterr);
    if (err) return err;
    memcpy(mtxdistfile->data.array_real_double, data,
           mtxdistfile->localdatasize * sizeof(*data));
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistfile_init_vector_array_complex_single()’ allocates and
 * initialises a distributed vector in array format with complex,
 * single precision coefficients.
 */
int mtxdistfile_init_vector_array_complex_single(
    struct mtxdistfile * mtxdistfile,
    int64_t num_rows,
    int64_t localdatasize,
    const int64_t * idx,
    const float (* data)[2],
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxdistfile_alloc_vector_array(
        mtxdistfile, mtxfile_complex, mtx_single, num_rows,
        localdatasize, idx, comm, disterr);
    if (err) return err;
    memcpy(mtxdistfile->data.array_complex_single, data,
           mtxdistfile->localdatasize * sizeof(*data));
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistfile_init_vector_array_complex_double()’ allocates and
 * initialises a vector in array format with complex, double precision
 * coefficients.
 */
int mtxdistfile_init_vector_array_complex_double(
    struct mtxdistfile * mtxdistfile,
    int64_t num_rows,
    int64_t localdatasize,
    const int64_t * idx,
    const double (* data)[2],
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxdistfile_alloc_vector_array(
        mtxdistfile, mtxfile_complex, mtx_double, num_rows,
        localdatasize, idx, comm, disterr);
    if (err) return err;
    memcpy(mtxdistfile->data.array_complex_double, data,
           mtxdistfile->localdatasize * sizeof(*data));
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistfile_init_vector_array_integer_single()’ allocates and
 * initialises a distributed vector in array format with integer,
 * single precision coefficients.
 */
int mtxdistfile_init_vector_array_integer_single(
    struct mtxdistfile * mtxdistfile,
    int64_t num_rows,
    int64_t localdatasize,
    const int64_t * idx,
    const int32_t * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxdistfile_alloc_vector_array(
        mtxdistfile, mtxfile_integer, mtx_single, num_rows,
        localdatasize, idx, comm, disterr);
    if (err) return err;
    memcpy(mtxdistfile->data.array_integer_single, data,
           mtxdistfile->localdatasize * sizeof(*data));
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistfile_init_vector_array_integer_double()’ allocates and
 * initialises a vector in array format with integer, double precision
 * coefficients.
 */
int mtxdistfile_init_vector_array_integer_double(
    struct mtxdistfile * mtxdistfile,
    int64_t num_rows,
    int64_t localdatasize,
    const int64_t * idx,
    const int64_t * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxdistfile_alloc_vector_array(
        mtxdistfile, mtxfile_integer, mtx_double, num_rows,
        localdatasize, idx, comm, disterr);
    if (err) return err;
    memcpy(mtxdistfile->data.array_integer_double, data,
           mtxdistfile->localdatasize * sizeof(*data));
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
    int64_t num_rows,
    int64_t num_columns,
    int64_t num_nonzeros,
    int64_t localdatasize,
    const int64_t * idx,
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

    mtxdistfile->comm = comm;
    mtxdistfile->comm_size = comm_size;
    mtxdistfile->rank = rank;
    mtxdistfile->header.object = mtxfile_matrix;
    mtxdistfile->header.format = mtxfile_coordinate;
    mtxdistfile->header.field = field;
    mtxdistfile->header.symmetry = symmetry;
    mtxfilecomments_init(&mtxdistfile->comments);
    mtxdistfile->size.num_rows = num_rows;
    mtxdistfile->size.num_columns = num_columns;
    mtxdistfile->size.num_nonzeros = num_nonzeros;
    mtxdistfile->precision = precision;

    /* check that the partition is compatible */
    int64_t num_data_lines;
    err = mtxfilesize_num_data_lines(
        &mtxdistfile->size, symmetry, &num_data_lines);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    int64_t datasize;
    disterr->mpierrcode = MPI_Allreduce(
        &localdatasize, &datasize, 1, MPI_INT64_T, MPI_SUM, comm);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (datasize != num_data_lines) return MTX_ERR_INCOMPATIBLE_PARTITION;

    mtxdistfile->precision = precision;
    mtxdistfile->datasize = datasize;
    mtxdistfile->localdatasize = localdatasize;
    err = mtxdistfile_init_idx(mtxdistfile, idx, disterr);
    if (err) {
        mtxfilecomments_free(&mtxdistfile->comments);
        return err;
    }
    err = mtxfiledata_alloc(
        &mtxdistfile->data, mtxdistfile->header.object, mtxdistfile->header.format,
        mtxdistfile->header.field, mtxdistfile->precision, mtxdistfile->localdatasize);
    if (mtxdisterror_allreduce(disterr, err)) {
        free(mtxdistfile->idx);
        mtxfilecomments_free(&mtxdistfile->comments);
        return MTX_ERR_MPI_COLLECTIVE;
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
    int64_t num_rows,
    int64_t num_columns,
    int64_t num_nonzeros,
    int64_t localdatasize,
    const int64_t * idx,
    const struct mtxfile_matrix_coordinate_real_single * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxdistfile_alloc_matrix_coordinate(
        mtxdistfile, mtxfile_real, symmetry, mtx_single,
        num_rows, num_columns, num_nonzeros,
        localdatasize, idx, comm, disterr);
    if (err) return err;
    memcpy(mtxdistfile->data.matrix_coordinate_real_single, data,
           mtxdistfile->localdatasize * sizeof(*data));
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
    int64_t num_rows,
    int64_t num_columns,
    int64_t num_nonzeros,
    int64_t localdatasize,
    const int64_t * idx,
    const struct mtxfile_matrix_coordinate_real_double * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxdistfile_alloc_matrix_coordinate(
        mtxdistfile, mtxfile_real, symmetry, mtx_double,
        num_rows, num_columns, num_nonzeros,
        localdatasize, idx, comm, disterr);
    if (err) return err;
    memcpy(mtxdistfile->data.matrix_coordinate_real_double, data,
           mtxdistfile->localdatasize * sizeof(*data));
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
    int64_t num_rows,
    int64_t num_columns,
    int64_t num_nonzeros,
    int64_t localdatasize,
    const int64_t * idx,
    const struct mtxfile_matrix_coordinate_complex_single * data,
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
    int64_t num_rows,
    int64_t num_columns,
    int64_t num_nonzeros,
    int64_t localdatasize,
    const int64_t * idx,
    const struct mtxfile_matrix_coordinate_complex_double * data,
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
    int64_t num_rows,
    int64_t num_columns,
    int64_t num_nonzeros,
    int64_t localdatasize,
    const int64_t * idx,
    const struct mtxfile_matrix_coordinate_integer_single * data,
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
    int64_t num_rows,
    int64_t num_columns,
    int64_t num_nonzeros,
    int64_t localdatasize,
    const int64_t * idx,
    const struct mtxfile_matrix_coordinate_integer_double * data,
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
    int64_t num_rows,
    int64_t num_columns,
    int64_t num_nonzeros,
    int64_t localdatasize,
    const int64_t * idx,
    const struct mtxfile_matrix_coordinate_pattern * data,
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
    int64_t num_rows,
    int64_t num_nonzeros,
    int64_t localdatasize,
    const int64_t * idx,
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

    mtxdistfile->comm = comm;
    mtxdistfile->comm_size = comm_size;
    mtxdistfile->rank = rank;

    mtxdistfile->header.object = mtxfile_vector;
    mtxdistfile->header.format = mtxfile_coordinate;
    mtxdistfile->header.field = field;
    mtxdistfile->header.symmetry = mtxfile_general;
    mtxfilecomments_init(&mtxdistfile->comments);
    mtxdistfile->size.num_rows = num_rows;
    mtxdistfile->size.num_columns = -1;
    mtxdistfile->size.num_nonzeros = num_nonzeros;

    /* check that the partition is compatible */
    int64_t num_data_lines;
    err = mtxfilesize_num_data_lines(
        &mtxdistfile->size, mtxfile_general, &num_data_lines);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    int64_t datasize;
    disterr->mpierrcode = MPI_Allreduce(
        &localdatasize, &datasize, 1, MPI_INT64_T, MPI_SUM, comm);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (datasize != num_data_lines) return MTX_ERR_INCOMPATIBLE_PARTITION;

    mtxdistfile->precision = precision;
    mtxdistfile->datasize = datasize;
    mtxdistfile->localdatasize = localdatasize;
    err = mtxdistfile_init_idx(mtxdistfile, idx, disterr);
    if (err) {
        mtxfilecomments_free(&mtxdistfile->comments);
        return err;
    }
    err = mtxfiledata_alloc(
        &mtxdistfile->data, mtxdistfile->header.object, mtxdistfile->header.format,
        mtxdistfile->header.field, mtxdistfile->precision, mtxdistfile->localdatasize);
    if (mtxdisterror_allreduce(disterr, err)) {
        free(mtxdistfile->idx);
        mtxfilecomments_free(&mtxdistfile->comments);
        return MTX_ERR_MPI_COLLECTIVE;
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
    int64_t num_rows,
    int64_t num_nonzeros,
    int64_t localdatasize,
    const int64_t * idx,
    const struct mtxfile_vector_coordinate_real_single * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err;
    err = mtxdistfile_alloc_vector_coordinate(
        mtxdistfile, mtxfile_real, mtx_single,
        num_rows, num_nonzeros, localdatasize, idx, comm, disterr);
    if (err)
        return err;
    memcpy(mtxdistfile->data.vector_coordinate_real_single, data,
           mtxdistfile->localdatasize * sizeof(*data));
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistfile_init_vector_coordinate_real_double()’ allocates and
 * initialises a vector in coordinate format with real, double
 * precision coefficients.
 */
int mtxdistfile_init_vector_coordinate_real_double(
    struct mtxdistfile * mtxdistfile,
    int64_t num_rows,
    int64_t num_nonzeros,
    int64_t localdatasize,
    const int64_t * idx,
    const struct mtxfile_vector_coordinate_real_double * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxdistfile_alloc_vector_coordinate(
        mtxdistfile, mtxfile_real, mtx_double,
        num_rows, num_nonzeros, localdatasize, idx, comm, disterr);
    if (err) return err;
    memcpy(mtxdistfile->data.vector_coordinate_real_double, data,
           mtxdistfile->localdatasize * sizeof(*data));
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistfile_init_vector_coordinate_complex_single()’ allocates and
 * initialises a distributed vector in coordinate format with complex,
 * single precision coefficients.
 */
int mtxdistfile_init_vector_coordinate_complex_single(
    struct mtxdistfile * mtxdistfile,
    int64_t num_rows,
    int64_t num_nonzeros,
    int64_t localdatasize,
    const int64_t * idx,
    const struct mtxfile_vector_coordinate_complex_single * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistfile_init_vector_coordinate_complex_double()’ allocates and
 * initialises a vector in coordinate format with complex, double
 * precision coefficients.
 */
int mtxdistfile_init_vector_coordinate_complex_double(
    struct mtxdistfile * mtxdistfile,
    int64_t num_rows,
    int64_t num_nonzeros,
    int64_t localdatasize,
    const int64_t * idx,
    const struct mtxfile_vector_coordinate_complex_double * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistfile_init_vector_coordinate_integer_single()’ allocates and
 * initialises a distributed vector in coordinate format with integer,
 * single precision coefficients.
 */
int mtxdistfile_init_vector_coordinate_integer_single(
    struct mtxdistfile * mtxdistfile,
    int64_t num_rows,
    int64_t num_nonzeros,
    int64_t localdatasize,
    const int64_t * idx,
    const struct mtxfile_vector_coordinate_integer_single * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistfile_init_vector_coordinate_integer_double()’ allocates and
 * initialises a vector in coordinate format with integer, double
 * precision coefficients.
 */
int mtxdistfile_init_vector_coordinate_integer_double(
    struct mtxdistfile * mtxdistfile,
    int64_t num_rows,
    int64_t num_nonzeros,
    int64_t localdatasize,
    const int64_t * idx,
    const struct mtxfile_vector_coordinate_integer_double * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistfile_init_vector_coordinate_pattern()’ allocates and
 * initialises a vector in coordinate format with boolean (pattern)
 * precision coefficients.
 */
int mtxdistfile_init_vector_coordinate_pattern(
    struct mtxdistfile * mtxdistfile,
    int64_t num_rows,
    int64_t num_nonzeros,
    int64_t localdatasize,
    const int64_t * idx,
    const struct mtxfile_vector_coordinate_pattern * data,
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
    int err = mtxfiledata_set_constant_real_single(
        &mtxdistfile->data, mtxdistfile->header.object,
        mtxdistfile->header.format, mtxdistfile->header.field,
        mtxdistfile->precision, mtxdistfile->localdatasize, 0, a);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
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
    int err = mtxfiledata_set_constant_real_double(
        &mtxdistfile->data, mtxdistfile->header.object,
        mtxdistfile->header.format, mtxdistfile->header.field,
        mtxdistfile->precision, mtxdistfile->localdatasize, 0, a);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
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
    int err = mtxfiledata_set_constant_complex_single(
        &mtxdistfile->data, mtxdistfile->header.object,
        mtxdistfile->header.format, mtxdistfile->header.field,
        mtxdistfile->precision, mtxdistfile->localdatasize, 0, a);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
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
    int err = mtxfiledata_set_constant_complex_double(
        &mtxdistfile->data, mtxdistfile->header.object,
        mtxdistfile->header.format, mtxdistfile->header.field,
        mtxdistfile->precision, mtxdistfile->localdatasize, 0, a);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
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
    int err = mtxfiledata_set_constant_integer_single(
        &mtxdistfile->data, mtxdistfile->header.object,
        mtxdistfile->header.format, mtxdistfile->header.field,
        mtxdistfile->precision, mtxdistfile->localdatasize, 0, a);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
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
    int err = mtxfiledata_set_constant_integer_double(
        &mtxdistfile->data, mtxdistfile->header.object,
        mtxdistfile->header.format, mtxdistfile->header.field,
        mtxdistfile->precision, mtxdistfile->localdatasize, 0, a);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/*
 * Convert to and from (non-distributed) Matrix Market format
 */

static int mtxdistfile_from_mtxfile_distribute(
    struct mtxdistfile * dst,
    const struct mtxfile * src,
    int64_t * partsptr,
    const int64_t * idx,
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
            if (!err) {
                disterr->mpierrcode = MPI_Send(
                    &idx[partsptr[p]], localdatasize, MPI_INT64_T, p, 1, comm);
                err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
            }
            err = err ? err : mtxfiledata_send(
                &src->data, src->header.object, src->header.format,
                src->header.field, src->precision, localdatasize,
                partsptr[p], p, 0, comm, disterr);
        } else if (p != root && rank == p) {
            disterr->mpierrcode = MPI_Recv(
                &dst->localdatasize, 1, MPI_INT64_T, root, 0, comm,
                MPI_STATUS_IGNORE);
            err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
            if (!err) {
                dst->idx = malloc(dst->localdatasize * sizeof(int64_t));
                if (!dst->idx) err = MTX_ERR_ERRNO;
            }
            if (!err) {
                disterr->mpierrcode = MPI_Recv(
                    dst->idx, dst->localdatasize, MPI_INT64_T,
                    root, 1, comm, MPI_STATUS_IGNORE);
                err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
            }
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
            if (!err) {
                dst->idx = malloc(dst->localdatasize * sizeof(int64_t));
                if (dst->idx) { memcpy(dst->idx, idx, dst->localdatasize * sizeof(int64_t)); }
                else { err = MTX_ERR_ERRNO; }
            }
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
 * ‘mtxdistfile_from_mtxfile_rowwise()’ creates a distributed Matrix
 * Market file from a Matrix Market file stored on a single root
 * process by partitioning the underlying matrix or vector rowwise and
 * distributing the parts among processes.
 *
 * This function performs collective communication and therefore
 * requires every process in the communicator to perform matching
 * calls to this function.
 */
int mtxdistfile_from_mtxfile_rowwise(
    struct mtxdistfile * dst,
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

    /* allocate storage for part numbers */
    int * dstpart = rank == root ? malloc(src->datasize * sizeof(int)) : NULL;
    err = rank == root && !dstpart ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(partsptr);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    /* partition the Matrix Market file on the root process */
    err = rank == root ? mtxfile_partition_rowwise(
        src, parttype, comm_size, partsizes, blksize, dstpart, partsptr)
        : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(dstpart); free(partsptr);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    /* allocate storage for sorting permutation */
    int64_t * perm = rank == root ? malloc(src->datasize * sizeof(int64_t)) : NULL;
    err = rank == root && !perm ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(dstpart); free(partsptr);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    /* sort the nonzeros in ascending order of their part numbers */
    err = rank == root ? mtxfiledata_sort_int(
        &src->data, src->header.object, src->header.format,
        src->header.field, src->precision, src->size.num_rows,
        src->size.num_columns, src->datasize, dstpart, perm)
        : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(perm); free(dstpart); free(partsptr);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    free(dstpart);

    /* invert the sorting permutation */
    int64_t * tmp = rank == root ? malloc(src->datasize * sizeof(int64_t)) : NULL;
    err = rank == root && !perm ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(perm); free(partsptr);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    if (rank == root) {
        for (int64_t i = 0; i < src->datasize; i++) tmp[i] = perm[i];
        for (int64_t i = 0; i < src->datasize; i++) perm[tmp[i]] = i;
    }
    free(tmp);

    err = mtxdistfile_from_mtxfile_distribute(
        dst, src, partsptr, perm, comm, root, disterr);
    if (mtxdisterror_allreduce(disterr, err)) {
        free(perm); free(partsptr);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    free(perm); free(partsptr);
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistfile_to_mtxfile()’ creates a Matrix Market file on a given
 * root process from a distributed Matrix Market file.
 *
 * This function performs collective communication and therefore
 * requires every process in the communicator to perform matching
 * calls to this function.
 */
int mtxdistfile_to_mtxfile(
    struct mtxfile * dst,
    const struct mtxdistfile * src,
    int root,
    struct mtxdisterror * disterr)
{
    MPI_Comm comm = src->comm;
    int comm_size = src->comm_size;
    int rank = src->rank;
    if (root < 0 || root >= comm_size) return MTX_ERR_INDEX_OUT_OF_BOUNDS;

    /* copy the header, comments, size line and precision */
    int err = rank == root ? mtxfileheader_copy(
        &dst->header, &src->header) : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    err = rank == root ? mtxfilecomments_copy(
        &dst->comments, &src->comments) : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    err = rank == root ? mtxfilesize_copy(&dst->size, &src->size) : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxfilecomments_free(&dst->comments);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    if (rank == root) dst->precision = src->precision;
    if (rank == root) dst->datasize = src->datasize;

    int64_t * idx = rank == root
        ? malloc(dst->datasize * sizeof(int64_t)) : NULL;
    if (rank == root && !idx) err = MTX_ERR_ERRNO;
    if (mtxdisterror_allreduce(disterr, err)) {
        if (rank == root) mtxfilecomments_free(&dst->comments);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    if (rank == root) {
        err = mtxfiledata_alloc(
            &dst->data, dst->header.object, dst->header.format,
            dst->header.field, dst->precision, dst->datasize);
    }
    if (mtxdisterror_allreduce(disterr, err)) {
        if (rank == root) free(idx);
        if (rank == root) mtxfilecomments_free(&dst->comments);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    /* send to the root process. */
    int64_t offset = 0;
    for (int p = 0; p < comm_size; p++) {
        if (p != root && rank == p) {
            int64_t sendsize = src->localdatasize;
            disterr->mpierrcode = MPI_Send(
                &sendsize, 1, MPI_INT64_T, root, 0, comm);
            err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
            if (!err) {
                disterr->mpierrcode = MPI_Send(
                    src->idx, sendsize, MPI_INT64_T, root, 1, comm);
                err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
            }
            err = err ? err : mtxfiledata_send(
                &src->data, src->header.object, src->header.format,
                src->header.field, src->precision, sendsize,
                0, root, 2, comm, disterr);
        } else if (p != root && rank == root) {
            int64_t recvsize;
            disterr->mpierrcode = MPI_Recv(
                &recvsize, 1, MPI_INT64_T, p, 0, comm,
                MPI_STATUS_IGNORE);
            err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
            if (!err) {
                disterr->mpierrcode = MPI_Recv(
                    &idx[offset], recvsize, MPI_INT64_T,
                    p, 1, comm, MPI_STATUS_IGNORE);
                err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
            }
            err = err ? err : mtxfiledata_recv(
                &dst->data, dst->header.object, dst->header.format,
                dst->header.field, dst->precision, recvsize,
                offset, p, 2, comm, disterr);
            offset += recvsize;
        } else if (p == root && rank == root) {
            memcpy(&idx[offset], src->idx,
                   src->localdatasize * sizeof(int64_t));
            err = mtxfiledata_copy(
                &dst->data, &src->data, dst->header.object,
                dst->header.format, dst->header.field, dst->precision,
                src->localdatasize, offset, offset);
            offset += src->localdatasize;
        }
        if (mtxdisterror_allreduce(disterr, err)) {
            if (rank == root) mtxfile_free(dst);
            if (rank == root) free(idx);
            return MTX_ERR_MPI_COLLECTIVE;
        }
    }

    if (rank == root) {
        for (int64_t i = 0; i < dst->datasize; i++) idx[i]++;
        err = mtxfiledata_permute(
            &dst->data, dst->header.object,
            dst->header.format, dst->header.field, dst->precision,
            dst->size.num_rows, dst->size.num_columns, dst->datasize, idx);
    }
    if (mtxdisterror_allreduce(disterr, err)) {
        if (rank == root) mtxfile_free(dst);
        if (rank == root) free(idx);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    if (rank == root) free(idx);
    return MTX_SUCCESS;
}

/*
 * I/O functions
 */

/**
 * ‘mtxdistfile_read_rowwise()’ reads a Matrix Market file from the
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
int mtxdistfile_read_rowwise(
    struct mtxdistfile * mtxdistfile,
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
    err = mtxdistfile_from_mtxfile_rowwise(
        mtxdistfile, &src, parttype, partsize, blksize, comm, root, disterr);
    if (err) { if (rank == root) mtxfile_free(&src); return err; } 
    if (rank == root) mtxfile_free(&src);
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistfile_fread_rowwise()’ reads a Matrix Market file from a
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
int mtxdistfile_fread_rowwise(
    struct mtxdistfile * mtxdistfile,
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
    err = mtxdistfile_from_mtxfile_rowwise(
        mtxdistfile, &src, parttype, partsize, blksize, comm, root, disterr);
    if (err) { if (rank == root) mtxfile_free(&src); return err; } 
    if (rank == root) mtxfile_free(&src);
    return MTX_SUCCESS;
}

#ifdef LIBMTX_HAVE_LIBZ
/**
 * ‘mtxdistfile_gzread_rowwise()’ reads a Matrix Market file from a
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
int mtxdistfile_gzread_rowwise(
    struct mtxdistfile * mtxdistfile,
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
    err = mtxdistfile_from_mtxfile_rowwise(
        mtxdistfile, &src, parttype, partsize, blksize, comm, root, disterr);
    if (err) { if (rank == root) mtxfile_free(&src); return err; } 
    if (rank == root) mtxfile_free(&src);
    return MTX_SUCCESS;
}
#endif

/**
 * ‘mtxdistfile_fwrite()’ writes a distributed Matrix Market
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
int mtxdistfile_fwrite(
    const struct mtxdistfile * mtxdistfile,
    FILE * f,
    const char * fmt,
    int64_t * bytes_written,
    int root,
    struct mtxdisterror * disterr)
{
    int rank;
    disterr->mpierrcode = MPI_Comm_rank(mtxdistfile->comm, &rank);
    int err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    struct mtxfile dst;
    err = mtxdistfile_to_mtxfile(&dst, mtxdistfile, root, disterr);
    if (err) return err;
    if (rank == root)
        err = mtxfile_fwrite(&dst, f, fmt, bytes_written);
    if (mtxdisterror_allreduce(disterr, err)) {
        if (rank == root) mtxfile_free(&dst);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    if (rank == root) mtxfile_free(&dst);
    return MTX_SUCCESS;
}
#endif
