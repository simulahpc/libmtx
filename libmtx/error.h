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
 * Last modified: 2022-01-04
 *
 * Error handling.
 */

#ifndef LIBMTX_ERROR_H
#define LIBMTX_ERROR_H

#ifdef LIBMTX_HAVE_MPI
#include <mpi.h>
#endif

/**
 * ‘mtxerror’ is a type for enumerating different error codes that are
 * used for error handling.
 *
 * There are error codes for errors based on ‘errno’, MPI errors, as
 * well as errors that may arise during parsing of files in the Matrix
 * Market format.
 */
enum mtxerror
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
    MTX_ERR_INVALID_MTX_FIELD = -10,        /* invalid Matrix Market field */
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
    MTX_ERR_INVALID_FIELD = -29,            /* invalid field */
    MTX_ERR_INVALID_VECTOR_TYPE = -30,      /* invalid vector type */
    MTX_ERR_INCOMPATIBLE_MTX_OBJECT = -31,  /* incompatible Matrix Market object */
    MTX_ERR_INCOMPATIBLE_MTX_FORMAT = -32,  /* incompatible Matrix Market format */
    MTX_ERR_INCOMPATIBLE_MTX_FIELD = -33,   /* incompatible Matrix Market field */
    MTX_ERR_INCOMPATIBLE_MTX_SYMMETRY = -34,/* incompatible Matrix Market symmetry */
    MTX_ERR_INCOMPATIBLE_MTX_SIZE = -35,    /* incompatible Matrix Market size */
    MTX_ERR_INCOMPATIBLE_FIELD = -36,       /* incompatible field */
    MTX_ERR_INCOMPATIBLE_PRECISION = -37,   /* incompatible precision */
    MTX_ERR_INCOMPATIBLE_SIZE = -38,        /* incompatible size */
    MTX_ERR_INCOMPATIBLE_VECTOR_TYPE = -39, /* incompatible vector type */
    MTX_ERR_INCOMPATIBLE_PARTITION = -40,   /* incompatible partition */
    MTX_ERR_INVALID_SORTING = -41,          /* invalid sorting */
    MTX_ERR_INVALID_ORDERING = -42,         /* invalid ordering */
    MTX_ERR_INVALID_MATRIX_TYPE = -43,      /* invalid matrix type */
    MTX_ERR_INCOMPATIBLE_MATRIX_TYPE = -44, /* incompatible matrix type */
    MTX_ERR_INVALID_TRANS_TYPE = -45,       /* invalid transpose type */
};

/**
 * ‘mtxstrerror()’ is a string describing an error code.
 *
 * The error code ‘err’ must correspond to one of the error codes
 * defined in the ‘mtxerror’ enum type.
 *
 * If ‘err’ is ‘MTX_ERR_ERRNO’, then ‘mtxstrerror()’ will use the
 * current value of ‘errno’ to obtain a description of the error.
 *
 * If ‘err’ may be ‘MTX_ERR_MPI’, then ‘mtxdiststrerror()’ should be
 * used instead.
 */
const char * mtxstrerror(int err);

/**
 * ‘mtxdiststrerror()’ is a string describing an error code.
 *
 * The error code ‘err’ must correspond to one of the error codes
 * defined in the ‘mtxerror’ enum type.
 *
 * ‘mtxdiststrerror()’ should be used in cases where ‘err’ may be
 * ‘MTX_ERR_MPI’, because it provides a more specific error
 * description than ‘mtxstrerror()’.
 *
 * If ‘err’ is ‘MTX_ERR_MPI’, then the argument ‘mpierrcode’ should be
 * set to the error code that was returned from the MPI function call
 * that failed. In addition, the argument ‘mpierrstr’ must be a char
 * array whose length is at least equal to ‘MPI_MAX_ERROR_STRING’. In
 * this case, ‘MPI_Error_string’ will be used to obtain a description
 * of the error.
 *
 * Otherwise, ‘mtxdiststrerror()’ returns the same error description
 * as ‘mtxstrerror()’ for error codes other than ‘MTX_ERR_MPI’.
 */
const char * mtxdiststrerror(
    int err,
    int mpierrcode,
    char * mpierrstr);

#ifdef LIBMTX_HAVE_MPI
/**
 * ‘mtxdisterror’ is used for error handling when using MPI.
 *
 * In particular, ‘mtxdisterror’ can be used to perform collective
 * error handling in situations where one or more processes encounter
 * an error, see ‘mtxdisterror_allreduce’.
 */
struct mtxdisterror
{
    MPI_Comm comm;
    int comm_size;
    int rank;
    int err;
    int mpierrcode;
    int (* buf)[3];
    char * description;
};

/**
 * ‘mtxdisterror_alloc()’ allocates storage needed for the MPI error
 * handling data structure ‘mtxdisterror’.
 */
int mtxdisterror_alloc(
    struct mtxdisterror * disterr,
    MPI_Comm comm,
    int * mpierrcode);

/**
 * ‘mtxdisterror_free()’ frees storage held by ‘struct mtxdisterror’.
 */
void mtxdisterror_free(
    struct mtxdisterror * disterr);

/**
 * ‘mtxdisterror_description()’ returns a string describing an MPI
 * error.
 *
 * Note that if ‘mtxdisterror_description()’ is called more than once,
 * the pointer that was returned from the previous call will no longer
 * be valid and using it will result in a use-after-free error.
 */
char * mtxdisterror_description(
    struct mtxdisterror * disterr);

/**
 * ‘mtxdisterror_allreduce()’ performs a collective reduction on error
 * codes provided by each MPI process in a communicator.
 *
 * This is a collective operations that must be performed by every
 * process in the communicator of the MPI error struct ‘disterr’.
 *
 * Each process gathers the error code and rank of every other
 * process.  If the error code of each and every process is
 * ‘MTX_SUCCESS’, then ‘mtxdisterror_allreduce()’ returns
 * ‘MTX_SUCCESS’. Otherwise, ‘MTX_ERR_MPI_COLLECTIVE’ is returned.
 * Moreover, the ‘buf’ member of ‘disterr’ will contain the rank and
 * error code of each process.
 *
 * If the error code ‘err’ is ‘MTX_ERR_MPI_COLLECTIVE’, then it is
 * assumed that a reduction has already been performed, and
 * ‘mtxdisterror_allreduce()’ returns immediately with
 * ‘MTX_ERR_MPI_COLLETIVE’.  As a result, if any process calls
 * ‘mtxdisterror_allreduce()’ with ‘err’ set to
 * ‘MTX_ERR_MPI_COLLETIVE’, then every other process in the
 * communicator must also set ‘err’ to ‘MTX_ERR_MPI_COLLECTIVE’, or
 * else the program may hang indefinitely.
 *
 * Example usage:
 *
 *     int err;
 *     MPI_Comm comm = MPI_COMM_WORLD;
 *     struct mtx_disterr disterr;
 *     err = mtxdisterror_alloc(&disterr, comm);
 *     if (err)
 *         MPI_Abort(comm, EXIT_FAILURE);
 *
 *     // Get the MPI rank of the current process.
 *     // Perform an all-reduction on the error code from
 *     // MPI_Comm_rank, so that if any process fails,
 *     // then we can exit gracefully.
 *     int comm_err, rank;
 *     err = MPI_Comm_rank(comm, &rank);
 *     comm_err = mtxdisterror_allreduce(disterr, err);
 *     if (comm_err)
 *         return comm_err;
 *
 *     ...
 *
 */
int mtxdisterror_allreduce(
    struct mtxdisterror * disterr,
    int err);
#endif

#endif
