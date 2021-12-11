/* This file is part of libmtx.
 *
 * Copyright (C) 2021 James D. Trotter
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
 * Last modified: 2021-09-22
 *
 * Matrix Market files distributed among multiple processes with MPI
 * for inter-process communication.
 */

#include <libmtx/libmtx-config.h>

#include <libmtx/error.h>
#include <libmtx/mtx/precision.h>
#include <libmtx/mtxdistfile/data.h>
#include <libmtx/mtxdistfile/mtxdistfile.h>
#include <libmtx/mtxfile/comments.h>
#include <libmtx/mtxfile/data.h>
#include <libmtx/mtxfile/header.h>
#include <libmtx/mtxfile/mtxfile.h>
#include <libmtx/mtxfile/size.h>
#include <libmtx/util/distpartition.h>
#include <libmtx/util/partition.h>

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
 * `mtxdistfile_free()' frees storage allocated for a distributed
 * Matrix Market file.
 */
void mtxdistfile_free(
    struct mtxdistfile * mtxdistfile)
{
    mtxfile_comments_free(&mtxdistfile->comments);
    mtxfile_free(&mtxdistfile->mtxfile);
}

/**
 * `mtxdistfile_init()' creates a distributed Matrix Market file from
 * Matrix Market files on each process in a communicator.
 *
 * This function performs collective communication and therefore
 * requires every process in the communicator to perform matching
 * calls to the function.
 */
int mtxdistfile_init(
    struct mtxdistfile * mtxdistfile,
    const struct mtxfile * mtxfile,
    MPI_Comm comm,
    struct mtxmpierror * mpierror)
{
    int err;
    int comm_size;
    mpierror->mpierrcode = MPI_Comm_size(comm, &comm_size);
    err = mpierror->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    int rank;
    mpierror->mpierrcode = MPI_Comm_rank(comm, &rank);
    err = mpierror->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;

    mtxdistfile->comm = comm;
    mtxdistfile->comm_size = comm_size;
    mtxdistfile->rank = rank;

    struct mtxfile_header * headers = malloc(
        comm_size * sizeof(struct mtxfile_header));
    err = !headers ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    struct mtxfile_size * sizes = malloc(
        comm_size * sizeof(struct mtxfile_size));
    err = !sizes ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err)) {
        free(headers);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    enum mtx_precision * precisions = malloc(comm_size * sizeof(enum mtx_precision));
    err = !precisions ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err)) {
        free(sizes);
        free(headers);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    /* Gather headers, size lines and precisions. */
    err = mtxfile_header_allgather(&mtxfile->header, headers, comm, mpierror);
    if (err) {
        free(precisions);
        free(sizes);
        free(headers);
        return err;
    }
    err = mtxfile_size_allgather(&mtxfile->size, sizes, comm, mpierror);
    if (err) {
        free(precisions);
        free(sizes);
        free(headers);
        return err;
    }
    mpierror->mpierrcode = MPI_Allgather(
        &mtxfile->precision, 1, MPI_INT, precisions, 1, MPI_INT, comm);
    err = mpierror->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err)) {
        free(precisions);
        free(sizes);
        free(headers);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    /* Check that the headers and precision are the same for all of
     * the underlying Matrix Market files. */
    err = MTX_SUCCESS;
    for (int p = 1; p < comm_size; p++) {
        if (headers[p].object != headers[0].object) {
            err = MTX_ERR_INCOMPATIBLE_MTX_OBJECT;
            break;
        } else if (headers[p].format != headers[0].format) {
            err = MTX_ERR_INCOMPATIBLE_MTX_FORMAT;
            break;
        } else if (headers[p].field != headers[0].field) {
            err = MTX_ERR_INCOMPATIBLE_MTX_FIELD;
            break;
        } else if (headers[p].symmetry != headers[0].symmetry) {
            err = MTX_ERR_INCOMPATIBLE_MTX_SYMMETRY;
            break;
        } else if (precisions[p] != precisions[0]) {
            err = MTX_ERR_INCOMPATIBLE_PRECISION;
            break;
        }
    }
    if (err) {
        free(precisions);
        free(sizes);
        free(headers);
        return err;
    }

    err = mtxfile_header_copy(&mtxdistfile->header, &headers[0]);
    if (mtxmpierror_allreduce(mpierror, err)) {
        free(precisions);
        free(sizes);
        free(headers);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    free(headers);
    mtxdistfile->precision = precisions[0];
    free(precisions);

    err = mtxfile_comments_init(&mtxdistfile->comments);
    if (mtxmpierror_allreduce(mpierror, err)) {
        free(sizes);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    /* Calcalute the size of the distributed Matrix Market file. */
    if (mtxdistfile->header.format == mtxfile_array) {
        int64_t num_rows = sizes[0].num_rows;
        err = MTX_SUCCESS;
        for (int p = 1; p < comm_size; p++) {
            if (sizes[p].num_columns != sizes[0].num_columns) {
                err = MTX_ERR_INCOMPATIBLE_MTX_SIZE;
                break;
            }
            num_rows += sizes[p].num_rows;
        }
        if (err) {
            mtxfile_comments_free(&mtxdistfile->comments);
            free(sizes);
            return err;
        }
        mtxdistfile->size.num_rows = num_rows;
        mtxdistfile->size.num_columns = sizes[0].num_columns;
        mtxdistfile->size.num_nonzeros = sizes[0].num_nonzeros;
    } else if (mtxdistfile->header.format == mtxfile_coordinate) {
        int64_t num_nonzeros = sizes[0].num_nonzeros;
        err = MTX_SUCCESS;
        for (int p = 1; p < comm_size; p++) {
            if (sizes[p].num_rows != sizes[0].num_rows ||
                sizes[p].num_columns != sizes[0].num_columns) {
                err = MTX_ERR_INCOMPATIBLE_MTX_SIZE;
                break;
            }
            num_nonzeros += sizes[p].num_nonzeros;
        }
        if (err) {
            mtxfile_comments_free(&mtxdistfile->comments);
            free(sizes);
            return err;
        }
        mtxdistfile->size.num_rows = sizes[0].num_rows;
        mtxdistfile->size.num_columns = sizes[0].num_columns;
        mtxdistfile->size.num_nonzeros = num_nonzeros;
    } else {
        mtxfile_comments_free(&mtxdistfile->comments);
        free(sizes);
        return MTX_ERR_INVALID_MTX_FORMAT;
    }
    free(sizes);

    err = mtxfile_init_copy(&mtxdistfile->mtxfile, mtxfile);
    if (mtxmpierror_allreduce(mpierror, err)) {
        mtxfile_comments_free(&mtxdistfile->comments);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    return MTX_SUCCESS;
}

/**
 * `mtxdistfile_alloc_copy()' allocates storage for a copy of a Matrix
 * Market file without initialising the underlying values.
 */
int mtxdistfile_alloc_copy(
    struct mtxdistfile * dst,
    const struct mtxdistfile * src,
    struct mtxmpierror * mpierror)
{
    int err;
    dst->comm = src->comm;
    dst->comm_size = src->comm_size;
    dst->rank = src->rank;
    err = mtxfile_header_copy(&dst->header, &src->header);
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    err = mtxfile_comments_copy(&dst->comments, &src->comments);
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    err = mtxfile_size_copy(&dst->size, &src->size);
    if (mtxmpierror_allreduce(mpierror, err)) {
        mtxfile_comments_free(&dst->comments);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    dst->precision = src->precision;
    err = mtxfile_alloc_copy(&dst->mtxfile, &src->mtxfile);
    if (mtxmpierror_allreduce(mpierror, err)) {
        mtxfile_comments_free(&dst->comments);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    return MTX_SUCCESS;
}

/**
 * `mtxdistfile_init_copy()' creates a copy of a Matrix Market file.
 */
int mtxdistfile_init_copy(
    struct mtxdistfile * dst,
    const struct mtxdistfile * src,
    struct mtxmpierror * mpierror);

/*
 * Matrix array formats
 */

/**
 * `mtxdistfile_alloc_matrix_array()' allocates a distributed matrix
 * in array format.
 */
int mtxdistfile_alloc_matrix_array(
    struct mtxdistfile * mtxdistfile,
    enum mtxfile_field field,
    enum mtxfile_symmetry symmetry,
    enum mtx_precision precision,
    int num_rows,
    int num_columns,
    MPI_Comm comm,
    struct mtxmpierror * mpierror)
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
    mpierror->mpierrcode = MPI_Comm_size(comm, &comm_size);
    err = mpierror->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    int rank;
    mpierror->mpierrcode = MPI_Comm_rank(comm, &rank);
    err = mpierror->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;

    /* Check that all processes request the same number of columns. */
    int * num_columns_per_process = malloc(comm_size * sizeof(int));
    err = !num_columns_per_process ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    mpierror->mpierrcode = MPI_Allgather(
        &num_columns, 1, MPI_INT, num_columns_per_process, 1, MPI_INT, comm);
    err = mpierror->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    for (int p = 0; p < comm_size; p++) {
        if (num_columns != num_columns_per_process[p]) {
            free(num_columns_per_process);
            return MTX_ERR_INCOMPATIBLE_MTX_SIZE;
        }
    }
    free(num_columns_per_process);

    mtxdistfile->comm = comm;
    mtxdistfile->comm_size = comm_size;
    mtxdistfile->rank = rank;
    mtxdistfile->header.object = mtxfile_matrix;
    mtxdistfile->header.format = mtxfile_array;
    mtxdistfile->header.field = field;
    mtxdistfile->header.symmetry = symmetry;
    mtxfile_comments_init(&mtxdistfile->comments);
    mpierror->mpierrcode = MPI_Allreduce(
        &num_rows, &mtxdistfile->size.num_rows, 1, MPI_INT, MPI_SUM, comm);
    err = mpierror->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    mtxdistfile->size.num_columns = num_columns;
    mtxdistfile->size.num_nonzeros = -1;
    mtxdistfile->precision = precision;

    err = mtxfile_alloc_matrix_array(
        &mtxdistfile->mtxfile, field, symmetry, precision, num_rows, num_columns);
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * `mtxdistfile_init_matrix_array_real_single()' allocates and
 * initialises a distributed matrix in array format with real, single
 * precision coefficients.
 */
int mtxdistfile_init_matrix_array_real_single(
    struct mtxdistfile * mtxdistfile,
    enum mtxfile_symmetry symmetry,
    int num_rows,
    int num_columns,
    const float * data,
    MPI_Comm comm,
    struct mtxmpierror * mpierror)
{
    int err;
    err = mtxdistfile_alloc_matrix_array(
        mtxdistfile, mtxfile_real, symmetry, mtx_single, num_rows, num_columns,
        comm, mpierror);
    if (err)
        return err;
    struct mtxfile * mtxfile = &mtxdistfile->mtxfile;
    int64_t num_data_lines;
    err = mtxfile_size_num_data_lines(&mtxfile->size, &num_data_lines);
    if (err) {
        mtxdistfile_free(mtxdistfile);
        return err;
    }
    memcpy(mtxfile->data.array_real_single, data, num_data_lines * sizeof(*data));
    return MTX_SUCCESS;
}

/**
 * `mtxdistfile_init_matrix_array_real_double()' allocates and
 * initialises a distributed matrix in array format with real, double
 * precision coefficients.
 */
int mtxdistfile_init_matrix_array_real_double(
    struct mtxdistfile * mtxdistfile,
    enum mtxfile_symmetry symmetry,
    int num_rows,
    int num_columns,
    const double * data,
    MPI_Comm comm,
    struct mtxmpierror * mpierror)
{
    int err;
    err = mtxdistfile_alloc_matrix_array(
        mtxdistfile, mtxfile_real, symmetry, mtx_double, num_rows, num_columns,
        comm, mpierror);
    if (err)
        return err;
    struct mtxfile * mtxfile = &mtxdistfile->mtxfile;
    int64_t num_data_lines;
    err = mtxfile_size_num_data_lines(&mtxfile->size, &num_data_lines);
    if (err) {
        mtxdistfile_free(mtxdistfile);
        return err;
    }
    memcpy(mtxfile->data.array_real_double, data, num_data_lines * sizeof(*data));
    return MTX_SUCCESS;
}

/**
 * `mtxdistfile_init_matrix_array_complex_single()' allocates and
 * initialises a distributed matrix in array format with complex,
 * single precision coefficients.
 */
int mtxdistfile_init_matrix_array_complex_single(
    struct mtxdistfile * mtxdistfile,
    enum mtxfile_symmetry symmetry,
    int num_rows,
    int num_columns,
    const float (* data)[2],
    MPI_Comm comm,
    struct mtxmpierror * mpierror);

/**
 * `mtxdistfile_init_matrix_array_complex_double()' allocates and
 * initialises a matrix in array format with complex, double precision
 * coefficients.
 */
int mtxdistfile_init_matrix_array_complex_double(
    struct mtxdistfile * mtxdistfile,
    enum mtxfile_symmetry symmetry,
    int num_rows,
    int num_columns,
    const double (* data)[2],
    MPI_Comm comm,
    struct mtxmpierror * mpierror);

/**
 * `mtxdistfile_init_matrix_array_integer_single()' allocates and
 * initialises a distributed matrix in array format with integer,
 * single precision coefficients.
 */
int mtxdistfile_init_matrix_array_integer_single(
    struct mtxdistfile * mtxdistfile,
    enum mtxfile_symmetry symmetry,
    int num_rows,
    int num_columns,
    const int32_t * data,
    MPI_Comm comm,
    struct mtxmpierror * mpierror);

/**
 * `mtxdistfile_init_matrix_array_integer_double()' allocates and
 * initialises a matrix in array format with integer, double precision
 * coefficients.
 */
int mtxdistfile_init_matrix_array_integer_double(
    struct mtxdistfile * mtxdistfile,
    enum mtxfile_symmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * data,
    MPI_Comm comm,
    struct mtxmpierror * mpierror);

/*
 * Vector array formats
 */

/**
 * `mtxdistfile_alloc_vector_array()' allocates a distributed vector
 * in array format.
 */
int mtxdistfile_alloc_vector_array(
    struct mtxdistfile * mtxdistfile,
    enum mtxfile_field field,
    enum mtx_precision precision,
    int num_rows,
    MPI_Comm comm,
    struct mtxmpierror * mpierror)
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
    mpierror->mpierrcode = MPI_Comm_size(comm, &comm_size);
    err = mpierror->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    int rank;
    mpierror->mpierrcode = MPI_Comm_rank(comm, &rank);
    err = mpierror->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;

    mtxdistfile->comm = comm;
    mtxdistfile->comm_size = comm_size;
    mtxdistfile->rank = rank;
    mtxdistfile->header.object = mtxfile_vector;
    mtxdistfile->header.format = mtxfile_array;
    mtxdistfile->header.field = field;
    mtxdistfile->header.symmetry = mtxfile_general;
    mtxfile_comments_init(&mtxdistfile->comments);
    mpierror->mpierrcode = MPI_Allreduce(
        &num_rows, &mtxdistfile->size.num_rows, 1, MPI_INT, MPI_SUM, comm);
    err = mpierror->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    mtxdistfile->size.num_columns = -1;
    mtxdistfile->size.num_nonzeros = -1;
    mtxdistfile->precision = precision;

    err = mtxfile_alloc_vector_array(
        &mtxdistfile->mtxfile, field, precision, num_rows);
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * `mtxdistfile_init_vector_array_real_single()' allocates and
 * initialises a distributed vector in array format with real, single
 * precision coefficients.
 */
int mtxdistfile_init_vector_array_real_single(
    struct mtxdistfile * mtxdistfile,
    int num_rows,
    const float * data,
    MPI_Comm comm,
    struct mtxmpierror * mpierror)
{
    int err;
    err = mtxdistfile_alloc_vector_array(
        mtxdistfile, mtxfile_real, mtx_single, num_rows, comm, mpierror);
    if (err)
        return err;
    struct mtxfile * mtxfile = &mtxdistfile->mtxfile;
    int64_t num_data_lines;
    err = mtxfile_size_num_data_lines(&mtxfile->size, &num_data_lines);
    if (err) {
        mtxdistfile_free(mtxdistfile);
        return err;
    }
    memcpy(mtxfile->data.array_real_single, data, num_data_lines * sizeof(*data));
    return MTX_SUCCESS;
}

/**
 * `mtxdistfile_init_vector_array_real_double()' allocates and initialises
 * a vector in array format with real, double precision coefficients.
 */
int mtxdistfile_init_vector_array_real_double(
    struct mtxdistfile * mtxdistfile,
    int num_rows,
    const double * data,
    MPI_Comm comm,
    struct mtxmpierror * mpierror)
{
    int err;
    err = mtxdistfile_alloc_vector_array(
        mtxdistfile, mtxfile_real, mtx_double, num_rows, comm, mpierror);
    if (err)
        return err;
    struct mtxfile * mtxfile = &mtxdistfile->mtxfile;
    int64_t num_data_lines;
    err = mtxfile_size_num_data_lines(&mtxfile->size, &num_data_lines);
    if (err) {
        mtxdistfile_free(mtxdistfile);
        return err;
    }
    memcpy(mtxfile->data.array_real_double, data, num_data_lines * sizeof(*data));
    return MTX_SUCCESS;
}

/**
 * `mtxdistfile_init_vector_array_complex_single()' allocates and
 * initialises a distributed vector in array format with complex,
 * single precision coefficients.
 */
int mtxdistfile_init_vector_array_complex_single(
    struct mtxdistfile * mtxdistfile,
    int num_rows,
    const float (* data)[2],
    MPI_Comm comm,
    struct mtxmpierror * mpierror);

/**
 * `mtxdistfile_init_vector_array_complex_double()' allocates and
 * initialises a vector in array format with complex, double precision
 * coefficients.
 */
int mtxdistfile_init_vector_array_complex_double(
    struct mtxdistfile * mtxdistfile,
    int num_rows,
    const double (* data)[2],
    MPI_Comm comm,
    struct mtxmpierror * mpierror);

/**
 * `mtxdistfile_init_vector_array_integer_single()' allocates and
 * initialises a distributed vector in array format with integer,
 * single precision coefficients.
 */
int mtxdistfile_init_vector_array_integer_single(
    struct mtxdistfile * mtxdistfile,
    int num_rows,
    const int32_t * data,
    MPI_Comm comm,
    struct mtxmpierror * mpierror)
{
    int err;
    err = mtxdistfile_alloc_vector_array(
        mtxdistfile, mtxfile_integer, mtx_single, num_rows, comm, mpierror);
    if (err)
        return err;
    struct mtxfile * mtxfile = &mtxdistfile->mtxfile;
    int64_t num_data_lines;
    err = mtxfile_size_num_data_lines(&mtxfile->size, &num_data_lines);
    if (err) {
        mtxdistfile_free(mtxdistfile);
        return err;
    }
    memcpy(mtxfile->data.array_integer_single, data, num_data_lines * sizeof(*data));
    return MTX_SUCCESS;
}

/**
 * `mtxdistfile_init_vector_array_integer_double()' allocates and
 * initialises a vector in array format with integer, double precision
 * coefficients.
 */
int mtxdistfile_init_vector_array_integer_double(
    struct mtxdistfile * mtxdistfile,
    int num_rows,
    const int64_t * data,
    MPI_Comm comm,
    struct mtxmpierror * mpierror);

/*
 * Matrix coordinate formats
 */

/**
 * `mtxdistfile_alloc_matrix_coordinate()' allocates a distributed
 * matrix in coordinate format.
 */
int mtxdistfile_alloc_matrix_coordinate(
    struct mtxdistfile * mtxdistfile,
    enum mtxfile_field field,
    enum mtxfile_symmetry symmetry,
    enum mtx_precision precision,
    int num_rows,
    int num_columns,
    int64_t num_nonzeros,
    MPI_Comm comm,
    struct mtxmpierror * mpierror)
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
    mpierror->mpierrcode = MPI_Comm_size(comm, &comm_size);
    err = mpierror->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    int rank;
    mpierror->mpierrcode = MPI_Comm_rank(comm, &rank);
    err = mpierror->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;

    /* Check that all processes specify the same number of rows and columns. */
    int * num_rows_per_process = malloc(comm_size * sizeof(int));
    err = !num_rows_per_process ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    mpierror->mpierrcode = MPI_Allgather(
        &num_rows, 1, MPI_INT, num_rows_per_process, 1, MPI_INT, comm);
    err = mpierror->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    for (int p = 0; p < comm_size; p++) {
        if (num_rows != num_rows_per_process[p]) {
            free(num_rows_per_process);
            return MTX_ERR_INCOMPATIBLE_MTX_SIZE;
        }
    }
    free(num_rows_per_process);

    int * num_columns_per_process = malloc(comm_size * sizeof(int));
    err = !num_columns_per_process ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    mpierror->mpierrcode = MPI_Allgather(
        &num_columns, 1, MPI_INT, num_columns_per_process, 1, MPI_INT, comm);
    err = mpierror->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    for (int p = 0; p < comm_size; p++) {
        if (num_columns != num_columns_per_process[p]) {
            free(num_columns_per_process);
            return MTX_ERR_INCOMPATIBLE_MTX_SIZE;
        }
    }
    free(num_columns_per_process);

    mtxdistfile->comm = comm;
    mtxdistfile->comm_size = comm_size;
    mtxdistfile->rank = rank;
    mtxdistfile->header.object = mtxfile_matrix;
    mtxdistfile->header.format = mtxfile_coordinate;
    mtxdistfile->header.field = field;
    mtxdistfile->header.symmetry = symmetry;
    mtxfile_comments_init(&mtxdistfile->comments);
    mtxdistfile->size.num_rows = num_rows;
    mtxdistfile->size.num_columns = num_columns;
    mpierror->mpierrcode = MPI_Allreduce(
        &num_nonzeros, &mtxdistfile->size.num_nonzeros, 1, MPI_INT64_T, MPI_SUM, comm);
    err = mpierror->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    mtxdistfile->precision = precision;

    err = mtxfile_alloc_matrix_coordinate(
        &mtxdistfile->mtxfile, field, symmetry, precision,
        num_rows, num_columns, num_nonzeros);
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * `mtxdistfile_init_matrix_coordinate_real_single()' allocates and
 * initialises a distributed matrix in coordinate format with real,
 * single precision coefficients.
 */
int mtxdistfile_init_matrix_coordinate_real_single(
    struct mtxdistfile * mtxdistfile,
    enum mtxfile_symmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_nonzeros,
    const struct mtxfile_matrix_coordinate_real_single * data,
    MPI_Comm comm,
    struct mtxmpierror * mpierror);

/**
 * `mtxdistfile_init_matrix_coordinate_real_double()' allocates and
 * initialises a matrix in coordinate format with real, double
 * precision coefficients.
 */
int mtxdistfile_init_matrix_coordinate_real_double(
    struct mtxdistfile * mtxdistfile,
    enum mtxfile_symmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_nonzeros,
    const struct mtxfile_matrix_coordinate_real_double * data,
    MPI_Comm comm,
    struct mtxmpierror * mpierror)
{
    int err;
    err = mtxdistfile_alloc_matrix_coordinate(
        mtxdistfile, mtxfile_real, symmetry, mtx_double,
        num_rows, num_columns, num_nonzeros, comm, mpierror);
    if (err)
        return err;
    struct mtxfile * mtxfile = &mtxdistfile->mtxfile;
    int64_t num_data_lines;
    err = mtxfile_size_num_data_lines(&mtxfile->size, &num_data_lines);
    if (err) {
        mtxdistfile_free(mtxdistfile);
        return err;
    }
    memcpy(mtxfile->data.matrix_coordinate_real_double, data,
           num_data_lines * sizeof(*data));
    return MTX_SUCCESS;
}

/**
 * `mtxdistfile_init_matrix_coordinate_complex_single()' allocates and
 * initialises a distributed matrix in coordinate format with complex,
 * single precision coefficients.
 */
int mtxdistfile_init_matrix_coordinate_complex_single(
    struct mtxdistfile * mtxdistfile,
    enum mtxfile_symmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_nonzeros,
    const struct mtxfile_matrix_coordinate_complex_single * data,
    MPI_Comm comm,
    struct mtxmpierror * mpierror);

/**
 * `mtxdistfile_init_matrix_coordinate_complex_double()' allocates and
 * initialises a matrix in coordinate format with complex, double
 * precision coefficients.
 */
int mtxdistfile_init_matrix_coordinate_complex_double(
    struct mtxdistfile * mtxdistfile,
    enum mtxfile_symmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_nonzeros,
    const struct mtxfile_matrix_coordinate_complex_double * data,
    MPI_Comm comm,
    struct mtxmpierror * mpierror);

/**
 * `mtxdistfile_init_matrix_coordinate_integer_single()' allocates and
 * initialises a distributed matrix in coordinate format with integer,
 * single precision coefficients.
 */
int mtxdistfile_init_matrix_coordinate_integer_single(
    struct mtxdistfile * mtxdistfile,
    enum mtxfile_symmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_nonzeros,
    const struct mtxfile_matrix_coordinate_integer_single * data,
    MPI_Comm comm,
    struct mtxmpierror * mpierror);

/**
 * `mtxdistfile_init_matrix_coordinate_integer_double()' allocates and
 * initialises a matrix in coordinate format with integer, double
 * precision coefficients.
 */
int mtxdistfile_init_matrix_coordinate_integer_double(
    struct mtxdistfile * mtxdistfile,
    enum mtxfile_symmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_nonzeros,
    const struct mtxfile_matrix_coordinate_integer_double * data,
    MPI_Comm comm,
    struct mtxmpierror * mpierror);

/**
 * `mtxdistfile_init_matrix_coordinate_pattern()' allocates and
 * initialises a matrix in coordinate format with boolean (pattern)
 * precision coefficients.
 */
int mtxdistfile_init_matrix_coordinate_pattern(
    struct mtxdistfile * mtxdistfile,
    enum mtxfile_symmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_nonzeros,
    const struct mtxfile_matrix_coordinate_pattern * data,
    MPI_Comm comm,
    struct mtxmpierror * mpierror);

/*
 * Vector coordinate formats
 */

/**
 * `mtxdistfile_alloc_vector_coordinate()' allocates a distributed
 * vector in coordinate format.
 */
int mtxdistfile_alloc_vector_coordinate(
    struct mtxdistfile * mtxdistfile,
    enum mtxfile_field field,
    enum mtx_precision precision,
    int num_rows,
    int64_t num_nonzeros,
    MPI_Comm comm,
    struct mtxmpierror * mpierror)
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
    mpierror->mpierrcode = MPI_Comm_size(comm, &comm_size);
    err = mpierror->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    int rank;
    mpierror->mpierrcode = MPI_Comm_rank(comm, &rank);
    err = mpierror->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;

    /* Check that all processes specify the same number of rows. */
    int * num_rows_per_process = malloc(comm_size * sizeof(int));
    err = !num_rows_per_process ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    mpierror->mpierrcode = MPI_Allgather(
        &num_rows, 1, MPI_INT, num_rows_per_process, 1, MPI_INT, comm);
    err = mpierror->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    for (int p = 0; p < comm_size; p++) {
        if (num_rows != num_rows_per_process[p]) {
            free(num_rows_per_process);
            return MTX_ERR_INCOMPATIBLE_MTX_SIZE;
        }
    }
    free(num_rows_per_process);

    mtxdistfile->comm = comm;
    mtxdistfile->comm_size = comm_size;
    mtxdistfile->rank = rank;
    mtxdistfile->header.object = mtxfile_vector;
    mtxdistfile->header.format = mtxfile_coordinate;
    mtxdistfile->header.field = field;
    mtxdistfile->header.symmetry = mtxfile_general;
    mtxfile_comments_init(&mtxdistfile->comments);
    mtxdistfile->size.num_rows = num_rows;
    mtxdistfile->size.num_columns = -1;
    mpierror->mpierrcode = MPI_Allreduce(
        &num_nonzeros, &mtxdistfile->size.num_nonzeros, 1, MPI_INT64_T, MPI_SUM, comm);
    err = mpierror->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    mtxdistfile->precision = precision;

    err = mtxfile_alloc_vector_coordinate(
        &mtxdistfile->mtxfile, field, precision, num_rows, num_nonzeros);
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * `mtxdistfile_init_vector_coordinate_real_single()' allocates and
 * initialises a distributed vector in coordinate format with real,
 * single precision coefficients.
 */
int mtxdistfile_init_vector_coordinate_real_single(
    struct mtxdistfile * mtxdistfile,
    int num_rows,
    int64_t num_nonzeros,
    const struct mtxfile_vector_coordinate_real_single * data,
    MPI_Comm comm,
    struct mtxmpierror * mpierror);

/**
 * `mtxdistfile_init_vector_coordinate_real_double()' allocates and
 * initialises a vector in coordinate format with real, double
 * precision coefficients.
 */
int mtxdistfile_init_vector_coordinate_real_double(
    struct mtxdistfile * mtxdistfile,
    int num_rows,
    int64_t num_nonzeros,
    const struct mtxfile_vector_coordinate_real_double * data,
    MPI_Comm comm,
    struct mtxmpierror * mpierror)
{
    int err;
    err = mtxdistfile_alloc_vector_coordinate(
        mtxdistfile, mtxfile_real, mtx_double,
        num_rows, num_nonzeros, comm, mpierror);
    if (err)
        return err;
    struct mtxfile * mtxfile = &mtxdistfile->mtxfile;
    int64_t num_data_lines;
    err = mtxfile_size_num_data_lines(&mtxfile->size, &num_data_lines);
    if (err) {
        mtxdistfile_free(mtxdistfile);
        return err;
    }
    memcpy(mtxfile->data.vector_coordinate_real_double, data,
           num_data_lines * sizeof(*data));
    return MTX_SUCCESS;
}

/**
 * `mtxdistfile_init_vector_coordinate_complex_single()' allocates and
 * initialises a distributed vector in coordinate format with complex,
 * single precision coefficients.
 */
int mtxdistfile_init_vector_coordinate_complex_single(
    struct mtxdistfile * mtxdistfile,
    int num_rows,
    int64_t num_nonzeros,
    const struct mtxfile_vector_coordinate_complex_single * data,
    MPI_Comm comm,
    struct mtxmpierror * mpierror);

/**
 * `mtxdistfile_init_vector_coordinate_complex_double()' allocates and
 * initialises a vector in coordinate format with complex, double
 * precision coefficients.
 */
int mtxdistfile_init_vector_coordinate_complex_double(
    struct mtxdistfile * mtxdistfile,
    int num_rows,
    int64_t num_nonzeros,
    const struct mtxfile_vector_coordinate_complex_double * data,
    MPI_Comm comm,
    struct mtxmpierror * mpierror);

/**
 * `mtxdistfile_init_vector_coordinate_integer_single()' allocates and
 * initialises a distributed vector in coordinate format with integer,
 * single precision coefficients.
 */
int mtxdistfile_init_vector_coordinate_integer_single(
    struct mtxdistfile * mtxdistfile,
    int num_rows,
    int64_t num_nonzeros,
    const struct mtxfile_vector_coordinate_integer_single * data,
    MPI_Comm comm,
    struct mtxmpierror * mpierror);

/**
 * `mtxdistfile_init_vector_coordinate_integer_double()' allocates and
 * initialises a vector in coordinate format with integer, double
 * precision coefficients.
 */
int mtxdistfile_init_vector_coordinate_integer_double(
    struct mtxdistfile * mtxdistfile,
    int num_rows,
    int64_t num_nonzeros,
    const struct mtxfile_vector_coordinate_integer_double * data,
    MPI_Comm comm,
    struct mtxmpierror * mpierror);

/**
 * `mtxdistfile_init_vector_coordinate_pattern()' allocates and
 * initialises a vector in coordinate format with boolean (pattern)
 * precision coefficients.
 */
int mtxdistfile_init_vector_coordinate_pattern(
    struct mtxdistfile * mtxdistfile,
    int num_rows,
    int64_t num_nonzeros,
    const struct mtxfile_vector_coordinate_pattern * data,
    MPI_Comm comm,
    struct mtxmpierror * mpierror);

/*
 * Modifying values
 */

/**
 * `mtxdistfile_set_constant_real_single()' sets every (nonzero) value
 * of a matrix or vector equal to a constant, single precision
 * floating point number.
 */
int mtxdistfile_set_constant_real_single(
    struct mtxdistfile * mtxdistfile,
    float a,
    struct mtxmpierror * mpierror)
{
    int err = mtxfile_set_constant_real_single(&mtxdistfile->mtxfile, a);
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * `mtxdistfile_set_constant_real_double()' sets every (nonzero) value
 * of a matrix or vector equal to a constant, double precision
 * floating point number.
 */
int mtxdistfile_set_constant_real_double(
    struct mtxdistfile * mtxdistfile,
    double a,
    struct mtxmpierror * mpierror)
{
    int err = mtxfile_set_constant_real_double(&mtxdistfile->mtxfile, a);
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * `mtxdistfile_set_constant_complex_single()' sets every (nonzero)
 * value of a matrix or vector equal to a constant, single precision
 * floating point complex number.
 */
int mtxdistfile_set_constant_complex_single(
    struct mtxdistfile * mtxdistfile,
    float a[2],
    struct mtxmpierror * mpierror)
{
    int err = mtxfile_set_constant_complex_single(&mtxdistfile->mtxfile, a);
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * `mtxdistfile_set_constant_integer_single()' sets every (nonzero)
 * value of a matrix or vector equal to a constant integer.
 */
int mtxdistfile_set_constant_integer_single(
    struct mtxdistfile * mtxdistfile,
    int32_t a,
    struct mtxmpierror * mpierror)
{
    int err = mtxfile_set_constant_integer_single(&mtxdistfile->mtxfile, a);
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/*
 * Convert to and from (non-distributed) Matrix Market format
 */

/**
 * `mtxdistfile_from_mtxfile()' creates a distributed Matrix Market
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
    struct mtxmpierror * mpierror)
{
    int err;
    int comm_size;
    mpierror->mpierrcode = MPI_Comm_size(comm, &comm_size);
    err = mpierror->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    int rank;
    mpierror->mpierrcode = MPI_Comm_rank(comm, &rank);
    err = mpierror->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;

    dst->comm = comm;
    dst->comm_size = comm_size;
    dst->rank = rank;

    /* Broadcast the header, comments, size line and precision. */
    err = (rank == root) ? mtxfile_header_copy(
        &dst->header, &src->header) : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    err = mtxfile_header_bcast(&dst->header, root, comm, mpierror);
    if (err)
        return err;
    err = (rank == root) ? mtxfile_comments_copy(
        &dst->comments, &src->comments) : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    err = mtxfile_comments_bcast(&dst->comments, root, comm, mpierror);
    if (err) {
        if (rank == root)
            mtxfile_comments_free(&dst->comments);
        return err;
    }
    err = (rank == root) ? mtxfile_size_copy(
        &dst->size, &src->size) : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err)) {
        mtxfile_comments_free(&dst->comments);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    err = mtxfile_size_bcast(&dst->size, root, comm, mpierror);
    if (err) {
        mtxfile_comments_free(&dst->comments);
        return err;
    }
    if (rank == root)
        dst->precision = src->precision;
    mpierror->mpierrcode = MPI_Bcast(
        &dst->precision, 1, MPI_INT, root, comm);
    err = mpierror->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err)) {
        mtxfile_comments_free(&dst->comments);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    int * sendcounts = (rank == root) ?
        malloc((2*comm_size+1) * sizeof(int)) : NULL;
    err = (rank == root && !sendcounts) ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err)) {
        mtxfile_comments_free(&dst->comments);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    int * displs = (rank == root) ? &sendcounts[comm_size] : NULL;

    /* Find the number of data lines to send and offsets for each
     * part. Note for matrices in array format, we evenly distribute
     * the number of rows, whereas in all other cases we evenly
     * distribute the total number of data lines. */
    err = MTX_SUCCESS;
    if (rank == root) {
        int64_t size;
        if (src->size.num_nonzeros >= 0) {
            size = src->size.num_nonzeros;
            displs[0] = 0;
            for (int p = 0; p < comm_size; p++) {
                displs[p+1] = displs[p] +
                    size / comm_size + (p < (size % comm_size) ? 1 : 0);
                sendcounts[p] = displs[p+1] - displs[p];
            }
        } else if (src->size.num_rows >= 0 && src->size.num_columns >= 0) {
            int64_t num_rows = src->size.num_rows;
            int64_t num_columns = src->size.num_columns;
            displs[0] = 0;
            for (int p = 0; p < comm_size; p++) {
                displs[p+1] = displs[p] +
                    (num_rows / comm_size + (p < (num_rows % comm_size) ? 1 : 0)) *
                    num_columns;
                sendcounts[p] = displs[p+1] - displs[p];
            }
        } else if (src->size.num_rows >= 0) {
            size = src->size.num_rows;
            displs[0] = 0;
            for (int p = 0; p < comm_size; p++) {
                displs[p+1] = displs[p] +
                    size / comm_size + (p < (size % comm_size) ? 1 : 0);
                sendcounts[p] = displs[p+1] - displs[p];
            }
        } else {
            err = MTX_ERR_INVALID_MTX_SIZE;
        }
    }
    if (mtxmpierror_allreduce(mpierror, err)) {
        if (rank == root)
            free(sendcounts);
        mtxfile_comments_free(&dst->comments);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    int recvcount;
    mpierror->mpierrcode = MPI_Scatter(
        sendcounts, 1, MPI_INT, &recvcount, 1, MPI_INT, root, comm);
    err = mpierror->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err)) {
        if (rank == root)
            free(sendcounts);
        mtxfile_comments_free(&dst->comments);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    /* Scatter the Matrix Market file. */
    err = mtxfile_scatterv(
        src, sendcounts, displs, &dst->mtxfile, recvcount, root, comm, mpierror);
    if (err) {
        if (rank == root)
            free(sendcounts);
        mtxfile_comments_free(&dst->comments);
        return err;
    }

    if (rank == root)
        free(sendcounts);
    return MTX_SUCCESS;
}

/*
 * I/O functions
 */

/**
 * `mtxdistfile_read()' reads a Matrix Market file from the given path
 * and distributes the data among MPI processes in a communicator.
 *
 * `precision' is used to determine the precision to use for storing
 * the values of matrix or vector entries.
 *
 * If an error code is returned, then `lines_read' and `bytes_read'
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
int mtxdistfile_read(
    struct mtxdistfile * mtxdistfile,
    enum mtx_precision precision,
    const char * path,
    int * lines_read,
    int64_t * bytes_read,
    size_t line_max,
    char * linebuf,
    MPI_Comm comm,
    struct mtxmpierror * mpierror)
{
    int err;
    if (lines_read)
        *lines_read = -1;
    if (bytes_read)
        *bytes_read = 0;

    int comm_size;
    mpierror->mpierrcode = MPI_Comm_size(comm, &comm_size);
    err = mpierror->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    int rank;
    mpierror->mpierrcode = MPI_Comm_rank(comm, &rank);
    err = mpierror->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    int root = comm_size-1;

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
        }
        err = MTX_SUCCESS;
    } else if (rank == root && ((f = fopen(path, "r")) == NULL)) {
        err = MTX_ERR_ERRNO;
    } else {
        err = MTX_SUCCESS;
    }
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;

    if (lines_read)
        *lines_read = 0;
    err = mtxdistfile_fread(
        mtxdistfile, precision, f, lines_read, bytes_read, line_max, linebuf,
        comm, mpierror);
    if (err) {
        if (rank == root)
            fclose(f);
        return err;
    }
    if (rank == root)
        fclose(f);
    return MTX_SUCCESS;
}

/**
 * `mtxdistfile_fread()' reads a Matrix Market file from a stream and
 * distributes the data among MPI processes in a communicator.
 *
 * `precision' is used to determine the precision to use for storing
 * the values of matrix or vector entries.
 *
 * If an error code is returned, then `lines_read' and `bytes_read'
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
int mtxdistfile_fread(
    struct mtxdistfile * mtxdistfile,
    enum mtx_precision precision,
    FILE * f,
    int * lines_read,
    int64_t * bytes_read,
    size_t line_max,
    char * linebuf,
    MPI_Comm comm,
    struct mtxmpierror * mpierror)
{
    int err;
    int comm_size;
    mpierror->mpierrcode = MPI_Comm_size(comm, &comm_size);
    err = mpierror->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    int rank;
    mpierror->mpierrcode = MPI_Comm_rank(comm, &rank);
    err = mpierror->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    int root = comm_size-1;
    if (comm_size <= 0)
        return MTX_SUCCESS;

    mtxdistfile->comm = comm;
    mtxdistfile->comm_size = comm_size;
    mtxdistfile->rank = rank;

    bool free_linebuf = (rank == root) && !linebuf;
    if (rank == root && !linebuf) {
        line_max = sysconf(_SC_LINE_MAX);
        linebuf = malloc(line_max+1);
    }
    err = (rank == root && !linebuf) ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;

    /* Read the header on the root process and broadcast to others. */
    err = (rank == root) ? mtxfile_fread_header(
        &mtxdistfile->header, f, lines_read, bytes_read, line_max, linebuf)
        : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err)) {
        if (free_linebuf)
            free(linebuf);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    err = mtxfile_header_bcast(&mtxdistfile->header, root, comm, mpierror);
    if (err) {
        if (free_linebuf)
            free(linebuf);
        return err;
    }

    /* Read comments on the root process and broadcast to others. */
    err = (rank == root) ? mtxfile_fread_comments(
        &mtxdistfile->comments, f, lines_read, bytes_read, line_max, linebuf)
        : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err)) {
        if (free_linebuf)
            free(linebuf);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    err = mtxfile_comments_bcast(&mtxdistfile->comments, root, comm, mpierror);
    if (err) {
        if (free_linebuf)
            free(linebuf);
        return err;
    }

    /* Read the size line on the root process and broadcast to others. */
    err = (rank == root) ? mtxfile_fread_size(
        &mtxdistfile->size, f, lines_read, bytes_read, line_max, linebuf,
        mtxdistfile->header.object, mtxdistfile->header.format)
        : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err)) {
        mtxfile_comments_free(&mtxdistfile->comments);
        if (free_linebuf)
            free(linebuf);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    err = mtxfile_size_bcast(&mtxdistfile->size, root, comm, mpierror);
    if (err) {
        mtxfile_comments_free(&mtxdistfile->comments);
        if (free_linebuf)
            free(linebuf);
        return err;
    }
    mtxdistfile->precision = precision;

    /* Partition the data into equal-sized blocks for each process.
     * For matrices and vectors in coordinate format, the total number
     * of data lines is evenly distributed among processes. Otherwise,
     * the number of rows is evenly distributed among processes. */
    struct mtxfile_size * sizes = (rank == root) ?
        malloc(comm_size * sizeof(struct mtxfile_size)) : NULL;
    err = (rank == root && !sizes) ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err)) {
        mtxfile_comments_free(&mtxdistfile->comments);
        if (free_linebuf)
            free(linebuf);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    if (rank == root) {
        if (mtxdistfile->size.num_nonzeros >= 0) {
            int64_t N = mtxdistfile->size.num_nonzeros;
            for (int p = 0; p < comm_size; p++) {
                sizes[p].num_rows = mtxdistfile->size.num_rows;
                sizes[p].num_columns = mtxdistfile->size.num_columns;
                sizes[p].num_nonzeros = N / comm_size + (p < (N % comm_size) ? 1 : 0);
            }
        } else {
            int64_t N = mtxdistfile->size.num_rows;
            for (int p = 0; p < comm_size; p++) {
                sizes[p].num_rows = (N / comm_size + (p < (N % comm_size) ? 1 : 0));
                sizes[p].num_columns = mtxdistfile->size.num_columns;
                sizes[p].num_nonzeros = mtxdistfile->size.num_nonzeros;
            }
        }
    }
    if (mtxmpierror_allreduce(mpierror, err)) {
        if (rank == root)
            free(sizes);
        mtxfile_comments_free(&mtxdistfile->comments);
        if (free_linebuf)
            free(linebuf);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    /* Allocate storage for a Matrix Market file on the root process. */
    err = (rank == root) ? mtxfile_alloc(
        &mtxdistfile->mtxfile, &mtxdistfile->header, NULL,
        &sizes[0], mtxdistfile->precision)
        : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err)) {
        if (rank == root)
            free(sizes);
        mtxfile_comments_free(&mtxdistfile->comments);
        if (free_linebuf)
            free(linebuf);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    /* Read each part of the Matrix Market file and send it to the
     * owning process. */
    for (int p = 0; p < comm_size-1; p++) {
        err = (rank == root)
            ? mtxfile_size_copy(&mtxdistfile->mtxfile.size, &sizes[p])
            : MTX_SUCCESS;
        if (mtxmpierror_allreduce(mpierror, err)) {
            if (rank == root || rank < p)
                mtxfile_free(&mtxdistfile->mtxfile);
            if (rank == root)
                free(sizes);
            mtxfile_comments_free(&mtxdistfile->comments);
            if (free_linebuf)
                free(linebuf);
            return MTX_ERR_MPI_COLLECTIVE;
        }

        int64_t num_data_lines;
        err = (rank == root) ? mtxfile_size_num_data_lines(
            &mtxdistfile->mtxfile.size, &num_data_lines) : MTX_SUCCESS;
        if (mtxmpierror_allreduce(mpierror, err)) {
            if (rank == root || rank < p)
                mtxfile_free(&mtxdistfile->mtxfile);
            if (rank == root)
                free(sizes);
            mtxfile_comments_free(&mtxdistfile->comments);
            if (free_linebuf)
                free(linebuf);
            return MTX_ERR_MPI_COLLECTIVE;
        }

        /* Read the next set of data lines on the root process. */
        err = (rank == root) ?
            mtxfiledata_fread(
                &mtxdistfile->mtxfile.data,
                f, lines_read, bytes_read, line_max, linebuf,
                mtxdistfile->mtxfile.header.object, mtxdistfile->mtxfile.header.format,
                mtxdistfile->mtxfile.header.field, mtxdistfile->mtxfile.precision,
                mtxdistfile->mtxfile.size.num_rows,
                mtxdistfile->mtxfile.size.num_columns,
                num_data_lines, 0)
            : MTX_SUCCESS;
        if (mtxmpierror_allreduce(mpierror, err)) {
            if (rank == root || rank < p)
                mtxfile_free(&mtxdistfile->mtxfile);
            if (rank == root)
                free(sizes);
            mtxfile_comments_free(&mtxdistfile->comments);
            if (free_linebuf)
                free(linebuf);
            return MTX_ERR_MPI_COLLECTIVE;
        }

        /* Send to the owning process. */
        if (rank == root) {
            err = mtxfile_send(&mtxdistfile->mtxfile, p, 0, comm, mpierror);
        } else if (rank == p) {
            err = mtxfile_recv(&mtxdistfile->mtxfile, root, 0, comm, mpierror);
        } else {
            err = MTX_SUCCESS;
        }
        if (mtxmpierror_allreduce(mpierror, err)) {
            if (rank == root || rank < p)
                mtxfile_free(&mtxdistfile->mtxfile);
            if (rank == root)
                free(sizes);
            mtxfile_comments_free(&mtxdistfile->comments);
            if (free_linebuf)
                free(linebuf);
            return MTX_ERR_MPI_COLLECTIVE;
        }
    }

    /* Read the final set of data lines on the root process. */
    err = (rank == root)
        ? mtxfile_size_copy(&mtxdistfile->mtxfile.size, &sizes[comm_size-1])
        : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err)) {
        mtxfile_free(&mtxdistfile->mtxfile);
        if (rank == root)
            free(sizes);
        mtxfile_comments_free(&mtxdistfile->comments);
        if (free_linebuf)
            free(linebuf);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    int64_t num_data_lines;
    err = (rank == root) ? mtxfile_size_num_data_lines(
        &mtxdistfile->mtxfile.size, &num_data_lines) : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err)) {
        mtxfile_free(&mtxdistfile->mtxfile);
        if (rank == root)
            free(sizes);
        mtxfile_comments_free(&mtxdistfile->comments);
        if (free_linebuf)
            free(linebuf);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    err = (rank == root) ?
        mtxfiledata_fread(
            &mtxdistfile->mtxfile.data,
            f, lines_read, bytes_read, line_max, linebuf,
            mtxdistfile->mtxfile.header.object, mtxdistfile->mtxfile.header.format,
            mtxdistfile->mtxfile.header.field, mtxdistfile->mtxfile.precision,
            mtxdistfile->mtxfile.size.num_rows, mtxdistfile->mtxfile.size.num_columns,
            num_data_lines, 0)
        : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err)) {
        mtxfile_free(&mtxdistfile->mtxfile);
        if (rank == root)
            free(sizes);
        mtxfile_comments_free(&mtxdistfile->comments);
        if (free_linebuf)
            free(linebuf);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    if (rank == root)
        free(sizes);
    if (free_linebuf)
        free(linebuf);

    mpierror->mpierrcode = MPI_Bcast(lines_read, 1, MPI_INT, root, comm);
    err = mpierror->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err)) {
        mtxdistfile_free(mtxdistfile);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    mpierror->mpierrcode = MPI_Bcast(bytes_read, 1, MPI_INT64_T, root, comm);
    err = mpierror->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err)) {
        mtxdistfile_free(mtxdistfile);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    return MTX_SUCCESS;
}

/**
 * `mtxdistfile_write()' writes a distributed Matrix Market file to
 * the given path.  The file may optionally be compressed by gzip.
 *
 * If `path' is `-', then standard output is used.
 *
 * If ‘fmt’ is ‘NULL’, then the format specifier ‘%g’ is used to print
 * floating point numbers with with enough digits to ensure correct
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
 * This function performs collective communication and therefore
 * requires every process in the communicator to perform matching
 * calls to the function.
 */
int mtxdistfile_write(
    const struct mtxdistfile * mtxdistfile,
    const char * path,
    bool gzip,
    const char * fmt,
    int64_t * bytes_written,
    bool sequential,
    struct mtxmpierror * mpierror)
{
    int err;
    *bytes_written = 0;
    if (!gzip) {
        FILE * f;
        if (strcmp(path, "-") == 0) {
            int fd = dup(STDOUT_FILENO);
            if (fd == -1)
                return MTX_ERR_ERRNO;
            if ((f = fdopen(fd, "w")) == NULL) {
                close(fd);
                return MTX_ERR_ERRNO;
            }
        } else if ((f = fopen(path, "w")) == NULL) {
            return MTX_ERR_ERRNO;
        }
        err = mtxdistfile_fwrite(
            mtxdistfile, f, fmt, bytes_written, sequential, mpierror);
        if (err) {
            fclose(f);
            return err;
        }
        fclose(f);
    } else {
        errno = ENOTSUP;
        return MTX_ERR_ERRNO;
    }
    return MTX_SUCCESS;
}

/**
 * `mtxdistfile_write_shared()' writes a distributed Matrix Market
 * file to a single file that is shared by all processes in the
 * communicator.  The file may optionally be compressed by gzip.
 *
 * If `path' is `-', then standard output is used.
 *
 * If ‘fmt’ is ‘NULL’, then the format specifier ‘%g’ is used to print
 * floating point numbers with with enough digits to ensure correct
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
    struct mtxmpierror * mpierror)
{
    int err;
    const char * mode = mtxdistfile->rank == 0 ? "w" : "a";
    *bytes_written = 0;
    if (!gzip) {
        FILE * f;
        if (strcmp(path, "-") == 0) {
            int fd = dup(STDOUT_FILENO);
            if (fd == -1)
                return MTX_ERR_ERRNO;
            if ((f = fdopen(fd, mode)) == NULL) {
                close(fd);
                return MTX_ERR_ERRNO;
            }
        } else if ((f = fopen(path, mode)) == NULL) {
            return MTX_ERR_ERRNO;
        }
        err = mtxdistfile_fwrite_shared(
            mtxdistfile, f, fmt, bytes_written, root, mpierror);
        if (err) {
            fclose(f);
            return err;
        }
        fclose(f);
    } else {
        errno = ENOTSUP;
        return MTX_ERR_ERRNO;
    }
    return MTX_SUCCESS;
}

static int mtxdistfile_fwrite_mtxfile(
    const struct mtxdistfile * mtxdistfile,
    FILE * f,
    const char * fmt,
    int64_t * bytes_written)
{
    int err;
    const struct mtxfile * mtxfile = &mtxdistfile->mtxfile;
    err = mtxfile_header_fwrite(&mtxfile->header, f, bytes_written);
    if (err)
        return err;
    err = mtxfile_comments_fputs(&mtxdistfile->comments, f, bytes_written);
    if (err)
        return err;
    err = mtxfile_comments_fputs(&mtxfile->comments, f, bytes_written);
    if (err)
        return err;
    err = mtxfile_size_fwrite(
        &mtxfile->size, mtxfile->header.object, mtxfile->header.format,
        f, bytes_written);
    if (err)
        return err;
    int64_t num_data_lines;
    err = mtxfile_size_num_data_lines(
        &mtxfile->size, &num_data_lines);
    if (err)
        return err;
    err = mtxfiledata_fwrite(
        &mtxfile->data, mtxfile->header.object, mtxfile->header.format,
        mtxfile->header.field, mtxfile->precision, num_data_lines,
        f, fmt, bytes_written);
    if (err)
        return err;
    return MTX_SUCCESS;
}

/**
 * `mtxdistfile_fwrite()' writes a distributed Matrix Market file to
 * the specified stream on each process.
 *
 * If ‘fmt’ is ‘NULL’, then the format specifier ‘%g’ is used to print
 * floating point numbers with with enough digits to ensure correct
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
 * If `sequential' is true, then output is performed in sequence by
 * MPI processes in the communicator.  This is useful, for example,
 * when writing to a common stream, such as standard output.  In this
 * case, we want to ensure that the processes write their data in the
 * correct order without interfering with each other.
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
    bool sequential,
    struct mtxmpierror * mpierror)
{
    int err;
    if (sequential) {
        for (int p = 0; p < mtxdistfile->comm_size; p++) {
            err = (mtxdistfile->rank == p)
                ? mtxdistfile_fwrite_mtxfile(mtxdistfile, f, fmt, bytes_written)
                : MTX_SUCCESS;
            if (mtxmpierror_allreduce(mpierror, err))
                return MTX_ERR_MPI_COLLECTIVE;
            MPI_Barrier(mtxdistfile->comm);
        }
    } else {
        err = mtxdistfile_fwrite_mtxfile(mtxdistfile, f, fmt, bytes_written);
        if (mtxmpierror_allreduce(mpierror, err))
            return MTX_ERR_MPI_COLLECTIVE;
    }
    return MTX_SUCCESS;
}

/**
 * `mtxdistfile_fwrite_shared()' writes a distributed Matrix Market
 * file to a single stream that is shared by every process in the
 * communicator.
 *
 * If ‘fmt’ is ‘NULL’, then the format specifier ‘%g’ is used to print
 * floating point numbers with with enough digits to ensure correct
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
int mtxdistfile_fwrite_shared(
    const struct mtxdistfile * mtxdistfile,
    FILE * f,
    const char * fmt,
    int64_t * bytes_written,
    int root,
    struct mtxmpierror * mpierror)
{
    int err;
    MPI_Comm comm = mtxdistfile->comm;
    int rank = mtxdistfile->rank;
    err = (rank == 0) ? mtxfile_header_fwrite(&mtxdistfile->header, f, bytes_written)
        : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    err = (rank == 0) ? mtxfile_comments_fputs(
        &mtxdistfile->comments, f, bytes_written)
        : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    err = (rank == 0) ? mtxfile_size_fwrite(
        &mtxdistfile->size, mtxdistfile->header.object, mtxdistfile->header.format,
        f, bytes_written)
        : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;

    for (int p = 0; p < mtxdistfile->comm_size; p++) {
        const struct mtxfile * mtxfile = &mtxdistfile->mtxfile;
        int64_t num_data_lines;

        err = MTX_SUCCESS;
        if (rank == root && p == root) {
            err = mtxfile_size_num_data_lines(&mtxfile->size, &num_data_lines);
            if (!err) {
                err = mtxfiledata_fwrite(
                    &mtxfile->data, mtxfile->header.object, mtxfile->header.format,
                    mtxfile->header.field, mtxfile->precision, num_data_lines,
                    f, fmt, bytes_written);
            }
        } else if (rank == root && p != root) {
            struct mtxfile recvmtxfile;
            err = mtxfile_recv(&recvmtxfile, p, 0, comm, mpierror);
            if (!err) {
                err =  mtxfile_size_num_data_lines(
                    &recvmtxfile.size, &num_data_lines);
                if (!err) {
                    err = mtxfiledata_fwrite(
                        &recvmtxfile.data, recvmtxfile.header.object, recvmtxfile.header.format,
                        recvmtxfile.header.field, recvmtxfile.precision, num_data_lines,
                        f, fmt, bytes_written);
                }
                mtxfile_free(&recvmtxfile);
            }
        } else if (rank == p) {
            err = mtxfile_send(mtxfile, root, 0, comm, mpierror);
        }
        if (mtxmpierror_allreduce(mpierror, err))
            return MTX_ERR_MPI_COLLECTIVE;
    }
    fflush(f);

    if (bytes_written) {
        int64_t bytes_written_per_process = *bytes_written;
        mpierror->mpierrcode = MPI_Allreduce(
            &bytes_written_per_process, bytes_written, 1, MPI_INT64_T, MPI_SUM,
            mtxdistfile->comm);
        err = mpierror->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
        if (mtxmpierror_allreduce(mpierror, err))
            return MTX_ERR_MPI_COLLECTIVE;
    }
    return MTX_SUCCESS;
}

/*
 * Sorting
 */

/**
 * ‘mtxdistfile_sort()’ sorts a distributed Matrix Market file in a
 * given order.
 *
 * The sorting order is determined by ‘sorting’. If the sorting order
 * is ‘mtxfile_unsorted’, nothing is done. If the sorting order is
 * ‘mtxfile_sorting_permutation’, then ‘perm’ must point to an array
 * of ‘size’ integers that specify the sorting permutation. Note that
 * the sorting permutation uses 1-based indexing.
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
    enum mtxfile_sorting sorting,
    int64_t size,
    int64_t * perm,
    struct mtxmpierror * mpierror)
{
    int err;
    struct mtxfile * mtxfile = &mtxdistfile->mtxfile;
    MPI_Comm comm = mtxdistfile->comm;

    int64_t num_data_lines;
    err = mtxfile_size_num_data_lines(&mtxfile->size, &num_data_lines);
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    err = size < 0 || size > num_data_lines
        ? MTX_ERR_INDEX_OUT_OF_BOUNDS : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;

    if (sorting == mtxfile_unsorted) {
        if (!perm)
            return MTX_SUCCESS;
        int64_t global_offset = 0;
        mpierror->mpierrcode = MPI_Exscan(
            &size, &global_offset, 1, MPI_INT64_T, MPI_SUM, comm);
        err = mpierror->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
        if (mtxmpierror_allreduce(mpierror, err))
            return MTX_ERR_MPI_COLLECTIVE;
        for (int64_t k = 0; k < size; k++)
            perm[k] = global_offset+k+1;
        return MTX_SUCCESS;
    } else if (sorting == mtxfile_sorting_permutation) {
        return mtxdistfiledata_permute(
            &mtxfile->data, mtxfile->header.object, mtxfile->header.format,
            mtxfile->header.field, mtxfile->precision, mtxfile->size.num_rows,
            mtxfile->size.num_columns, size, perm, comm, mpierror);
    } else if (sorting == mtxfile_row_major) {
        return mtxdistfiledata_sort_row_major(
            &mtxfile->data, mtxfile->header.object, mtxfile->header.format,
            mtxfile->header.field, mtxfile->precision, mtxfile->size.num_rows,
            mtxfile->size.num_columns, size, perm, comm, mpierror);
    } else if (sorting == mtxfile_column_major) {
        return mtxdistfiledata_sort_column_major(
            &mtxfile->data, mtxfile->header.object, mtxfile->header.format,
            mtxfile->header.field, mtxfile->precision, mtxfile->size.num_rows,
            mtxfile->size.num_columns, size, perm, comm, mpierror);
    } else if (sorting == mtxfile_morton) {
        return mtxdistfiledata_sort_morton(
            &mtxfile->data, mtxfile->header.object, mtxfile->header.format,
            mtxfile->header.field, mtxfile->precision, mtxfile->size.num_rows,
            mtxfile->size.num_columns, size, perm, comm, mpierror);
    } else {
        return MTX_ERR_INVALID_MTX_SORTING;
    }
    return MTX_SUCCESS;
}

/*
 * Partitioning
 */

/**
 * ‘mtxdistfile_init_from_partition()’ creates a distributed Matrix
 * Market file from a partitioning of another distributed Matrix
 * Market file.
 *
 * On each process, a partitioning can be obtained from
 * ‘mtxdistfile_partition_rows()’. This provides the arrays
 * ‘data_lines_per_part_ptr’ and ‘data_lines_per_part’, which together
 * describe the size of each part and the indices to its data lines on
 * the current process. The number of parts in the partitioning must
 * be less than or equal to the number of processes in the MPI
 * communicator.
 *
 * The ‘p’th value of ‘data_lines_per_part_ptr’ must be an offset to
 * the first data line belonging to the ‘p’th part of the partition,
 * while the final value of the array points to one place beyond the
 * final data line.  Moreover for each part ‘p’ of the partitioning,
 * the entries from ‘data_lines_per_part[p]’ up to, but not including,
 * ‘data_lines_per_part[p+1]’, are the indices of the data lines in
 * ‘src’ that are assigned to the ‘p’th part of the partitioning.
 */
int mtxdistfile_init_from_partition(
    struct mtxdistfile * dst,
    const struct mtxdistfile * src,
    int num_parts,
    const int64_t * data_lines_per_part_ptr,
    const int64_t * data_lines_per_part,
    struct mtxmpierror * mpierror)
{
    int err;
    if (num_parts > src->comm_size)
        return MTX_ERR_INDEX_OUT_OF_BOUNDS;

    /* Allocate storage for sending/receiving Matrix Market files. */
    struct mtxfile * mtxfiles = malloc(2*src->comm_size * sizeof(struct mtxfile));
    err = !mtxfiles ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    struct mtxfile * sendmtxfiles = &mtxfiles[0];
    struct mtxfile * recvmtxfiles = &mtxfiles[src->comm_size];

    err = mtxfile_init_from_partition(
        sendmtxfiles, &src->mtxfile, num_parts,
        data_lines_per_part_ptr, data_lines_per_part);
    if (mtxmpierror_allreduce(mpierror, err)) {
        free(mtxfiles);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    /* Send empty files to any remaining processes, if there are more
     * processes than parts in the partitioning. */
    for (int p = num_parts; p < src->comm_size; p++) {
        struct mtxfile_size size;
        if (src->size.num_nonzeros >= 0) {
            size.num_rows = src->size.num_rows;
            size.num_columns = src->size.num_columns;
            size.num_nonzeros = 0;
        } else {
            size.num_rows = 0;
            size.num_columns = src->size.num_columns;
            size.num_nonzeros = src->size.num_nonzeros;
        }
        err = mtxfile_alloc(
            &sendmtxfiles[p],
            &src->mtxfile.header,
            &src->mtxfile.comments,
            &size, src->mtxfile.precision);
        if (mtxmpierror_allreduce(mpierror, err)) {
            for (int q = p-1; q >= 0; q--)
                mtxfile_free(&sendmtxfiles[q]);
            free(mtxfiles);
            return MTX_ERR_MPI_COLLECTIVE;
        }
    }

    /* Exchange Matrix Market files among processes. */
    err = mtxfile_alltoall(sendmtxfiles, recvmtxfiles, src->comm, mpierror);
    if (err) {
        for (int p = 0; p < src->comm_size; p++)
            mtxfile_free(&sendmtxfiles[p]);
        free(mtxfiles);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    /* Concatenate the files that were received. */
    struct mtxfile dstmtxfile;
    err = mtxfile_init_copy(&dstmtxfile, &recvmtxfiles[0]);
    if (mtxmpierror_allreduce(mpierror, err)) {
        for (int p = 0; p < src->comm_size; p++)
            mtxfile_free(&recvmtxfiles[p]);
        for (int p = 0; p < src->comm_size; p++)
            mtxfile_free(&sendmtxfiles[p]);
        free(mtxfiles);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    err = mtxfile_catn(
        &dstmtxfile, src->comm_size-1, &recvmtxfiles[1], false);
    if (mtxmpierror_allreduce(mpierror, err)) {
        mtxfile_free(&dstmtxfile);
        for (int p = 0; p < src->comm_size; p++)
            mtxfile_free(&recvmtxfiles[p]);
        for (int p = 0; p < src->comm_size; p++)
            mtxfile_free(&sendmtxfiles[p]);
        free(mtxfiles);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    for (int p = 0; p < src->comm_size; p++)
        mtxfile_free(&recvmtxfiles[p]);
    for (int p = 0; p < src->comm_size; p++)
        mtxfile_free(&sendmtxfiles[p]);
    free(mtxfiles);

    /* Create the final, distributed Matrix Market file. */
    err = mtxdistfile_init(dst, &dstmtxfile, src->comm, mpierror);
    if (err) {
        mtxfile_free(&dstmtxfile);
        return err;
    }
    mtxfile_free(&dstmtxfile);
    err = mtxfile_comments_copy(&dst->comments, &src->comments);
    if (err) {
        mtxdistfile_free(dst);
        return err;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistfile_partition_rows()’ partitions data lines of a
 * distributed Matrix Market file according to the given row
 * partitioning.
 *
 * If it is not ‘NULL’, the array ‘part_per_data_line’ must contain
 * enough storage to hold one ‘int’ for each data line held by the
 * current process. (The number of data lines is obtained from
 * ‘mtxfile_size_num_data_lines()’). On a successful return, the ‘k’th
 * entry in the array specifies the part number that was assigned to
 * the ‘k’th data line of ‘src’.
 *
 * The array ‘data_lines_per_part_ptr’ must contain at least enough
 * storage for ‘row_partition->num_parts+1’ values of type ‘int64_t’.
 * If successful, the ‘p’th value of ‘data_lines_per_part_ptr’ is an
 * offset to the first data line belonging to the ‘p’th part of the
 * partition, while the final value of the array points to one place
 * beyond the final data line.  Moreover ‘data_lines_per_part’ must
 * contain enough storage to hold one ‘int64_t’ for each data line.
 * For each part ‘p’ of the partitioning, the entries from
 * ‘data_lines_per_part[p]’ up to, but not including,
 * ‘data_lines_per_part[p+1]’, are the indices of the data lines in
 * ‘src’ that are assigned to the ‘p’th part of the partitioning.
 */
int mtxdistfile_partition_rows(
    const struct mtxdistfile * mtxdistfile,
    const struct mtx_partition * row_partition,
    int * part_per_data_line,
    int64_t * data_lines_per_part_ptr,
    int64_t * data_lines_per_part,
    struct mtxmpierror * mpierror)
{
    int err;

    /* For matrices or vectors in array format, compute the offset to
     * the first element of the part of the matrix or vector owned by
     * the current process. */
    int64_t offset = 0;
    if (mtxdistfile->header.format == mtxfile_array) {
        int64_t num_rows = mtxdistfile->mtxfile.size.num_rows;
        mpierror->mpierrcode = MPI_Exscan(
            &num_rows, &offset, 1, MPI_INT64_T, MPI_SUM, mtxdistfile->comm);
        err = mpierror->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
        if (mtxmpierror_allreduce(mpierror, err))
            return MTX_ERR_MPI_COLLECTIVE;
        if (mtxdistfile->mtxfile.size.num_columns >= 0)
            offset *= mtxdistfile->mtxfile.size.num_columns;
    }

    int64_t size;
    err = mtxfile_size_num_data_lines(&mtxdistfile->mtxfile.size, &size);
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;

    err = mtxfile_partition_rows(
        &mtxdistfile->mtxfile, size, offset, row_partition, part_per_data_line,
        data_lines_per_part_ptr, data_lines_per_part);
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}
#endif
