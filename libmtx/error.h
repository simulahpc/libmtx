/* This file is part of libmtx.
 *
 * Copyright (C) 2021 James D. Trotter
 *
 * libmtx is free software: you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * libmtx is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with libmtx.  If not, see
 * <https://www.gnu.org/licenses/>.
 *
 * Authors: James D. Trotter <james@simula.no>
 * Last modified: 2021-09-03
 *
 * Error handling.
 */

#ifndef LIBMTX_ERROR_H
#define LIBMTX_ERROR_H

#ifdef LIBMTX_HAVE_MPI
#include <mpi.h>
#endif

/**
 * `mtx_error' is a type for enumerating different error codes that
 * are used for error handling.
 *
 * There are error codes for errors based on `errno', MPI errors, as
 * well as errors that may arise during parsing of files in the Matrix
 * Market format.
 */
enum mtx_error
{
    MTX_SUCCESS = 0,                        /* no error */
    MTX_ERR_ERRNO = -1,                     /* error code provided by errno */
    MTX_ERR_MPI = -2,                       /* MPI error */
    MTX_ERR_MPI_COLLECTIVE = -3,            /* MPI collective error */
    MTX_ERR_MPI_NOT_INITIALIZED = -4,       /* MPI not initialized */
    MTX_ERR_EOF = -5,                       /* unexpected end-of-file */
    MTX_ERR_LINE_TOO_LONG = -6,             /* line exceeds maximum length */
    MTX_ERR_INVALID_MTX_HEADER = -7,        /* invalid Matrix Market header */
    MTX_ERR_INVALID_MTX_OBJECT = -8,        /* invalid Matrix Market object */
    MTX_ERR_INVALID_MTX_FORMAT = -9,        /* invalid Matrix Market format */
    MTX_ERR_INVALID_MTX_FIELD = -10,         /* invalid Matrix Market field */
    MTX_ERR_INVALID_MTX_SYMMETRY = -11,     /* invalid Matrix Market symmetry */
    MTX_ERR_INVALID_MTX_TRIANGLE = -12,     /* invalid Matrix Market triangle */
    MTX_ERR_INVALID_MTX_SORTING = -13,      /* invalid Matrix Market sorting */
    MTX_ERR_INVALID_MTX_ORDERING = -14,     /* invalid Matrix Market ordering */
    MTX_ERR_INVALID_MTX_ASSEMBLY = -15,     /* invalid Matrix Market assembly */
    MTX_ERR_INVALID_MTX_COMMENT = -16,      /* invalid Matrix Market comment line */
    MTX_ERR_INVALID_MTX_SIZE = -17,         /* invalid Matrix Market size info */
    MTX_ERR_INVALID_MTX_DATA = -18,         /* invalid Matrix Market data */
    MTX_ERR_INVALID_PRECISION = -19,        /* invalid precision */
    MTX_ERR_INVALID_INDEX_SET_TYPE = -20,   /* invalid index set type */
    MTX_ERR_INVALID_PARTITION_TYPE = -22,   /* invalid partition type */
    MTX_ERR_INVALID_STREAM_TYPE = -23,      /* invalid stream type */
    MTX_ERR_INVALID_FORMAT_SPECIFIER = -24, /* invalid format specifier */
    MTX_ERR_INDEX_OUT_OF_BOUNDS = -25,      /* index out of bounds */
    MTX_ERR_NO_BUFFER_SPACE = -26,          /* not enough space in buffer */
    MTX_ERR_NOT_CONVERGED = -27,            /* iterative method did not converge */
    MTX_ERR_INVALID_PATH_FORMAT = -28,      /* invalid path format */
};

/**
 * `mtx_strerror()' is a string describing an error code.
 *
 * The error code `err' must correspond to one of the error codes
 * defined in the `mtx_error' enum type.
 *
 * If `err' is `MTX_ERR_ERRNO', then `mtx_strerror()' will use the
 * current value of `errno' to obtain a description of the error.
 *
 * If `err' may be `MTX_ERR_MPI', then `mtx_strerror_mpi()' should be
 * used instead.
 */
const char * mtx_strerror(int err);

/**
 * `mtx_strerror_mpi()' is a string describing an error code.
 *
 * The error code `err' must correspond to one of the error codes
 * defined in the `mtx_error' enum type.
 *
 * `mtx_strerror_mpi()' should be used in cases where `err' may be
 * `MTX_ERR_MPI', because it provides a more specific error
 * description than `mtx_strerror()'.
 *
 * If `err' is `MTX_ERR_MPI', then the argument `mpierrcode' should be
 * set to the error code that was returned from the MPI function call
 * that failed. In addition, the argument `mpierrstr' must be a char
 * array whose length is at least equal to `MPI_MAX_ERROR_STRING'. In
 * this case, `MPI_Error_string' will be used to obtain a description
 * of the error.
 *
 * Otherwise, `mtx_strerror_mpi()' returns the same error description
 * as `mtx_strerror()' for error codes other than `MTX_ERR_MPI'.
 */
const char * mtx_strerror_mpi(
    int err,
    int mpierrcode,
    char * mpierrstr);

#ifdef LIBMTX_HAVE_MPI
/**
 * `mtxmpierror' is used for error handling when using MPI.
 *
 * In particular, `mtxmpierror' can be used to perform collective
 * error handling in situations where one or more processes encounter
 * an error, see `mtxmpierror_allreduce'.
 */
struct mtxmpierror
{
    MPI_Comm comm;
    int comm_size;
    int rank;
    int err;
    int mpierrcode;
    int (* buf)[2];
};

/**
 * `mtxmpierror_alloc()' allocates storage needed for the MPI error
 * handling data structure `mtxmpierror'.
 */
int mtxmpierror_alloc(
    struct mtxmpierror * mpierror,
    MPI_Comm comm);

/**
 * `mtxmpierror_free()' frees storage held by `struct mtxmpierror'.
 */
void mtxmpierror_free(
    struct mtxmpierror * mpierror);

/**
 * `mtxmpierror_description()' returns a string describing an MPI
 * error.
 *
 * The caller is responsible for freeing the storage required for the
 * string that is returned by calling `free()'.
 */
char * mtxmpierror_description(
    struct mtxmpierror * mpierror);

/**
 * `mtxmpierror_allreduce()' performs a collective reduction on error
 * codes provided by each MPI process in a communicator.
 *
 * This is a collective operations that must be performed by every
 * process in the communicator of the MPI error struct `mpierror'.
 *
 * Each process gathers the error code and rank of every other
 * process.  If the error code of each and every process is
 * `MTX_SUCCESS', then `mtxmpierror_allreduce()' returns
 * `MTX_SUCCESS'. Otherwise, `MTX_ERR_MPI_COLLECTIVE' is returned.
 * Moreover, the `buf' member of `mpierror' will contain the rank and
 * error code of each process.
 *
 * If the error code `err' is `MTX_ERR_MPI_COLLECTIVE', then it is
 * assumed that a reduction has already been performed, and
 * `mtxmpierror_allreduce()' returns immediately with
 * `MTX_ERR_MPI_COLLETIVE'.  As a result, if any process calls
 * `mtxmpierror_allreduce()' with `err' set to
 * `MTX_ERR_MPI_COLLETIVE', then every other process in the
 * communicator must also set `err' to `MTX_ERR_MPI_COLLECTIVE', or
 * else the program may hang indefinitely.
 *
 * Example usage:
 *
 *     int err;
 *     MPI_Comm comm = MPI_COMM_WORLD;
 *     struct mtx_mpierror mpierror;
 *     err = mtxmpierror_alloc(&mpierror, comm);
 *     if (err)
 *         MPI_Abort(comm, EXIT_FAILURE);
 *
 *     // Get the MPI rank of the current process.
 *     // Perform an all-reduction on the error code from
 *     // MPI_Comm_rank, so that if any process fails,
 *     // then we can exit gracefully.
 *     int comm_err, rank;
 *     err = MPI_Comm_rank(comm, &rank);
 *     comm_err = mtxmpierror_allreduce(mpierror, err);
 *     if (comm_err)
 *         return comm_err;
 *
 *     ...
 *
 */
int mtxmpierror_allreduce(
    struct mtxmpierror * mpierror,
    int err);
#endif

#endif
