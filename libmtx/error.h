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
 * Last modified: 2022-02-23
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
    MTX_SUCCESS = 0,                  /* no error */
    MTX_ERR_ERRNO,                    /* error code provided by errno */
    MTX_ERR_NOT_SUPPORTED,            /* operation not supported */
    MTX_ERR_MPI_NOT_SUPPORTED,        /* MPI not supported */
    MTX_ERR_OPENMP_NOT_SUPPORTED,     /* OpenMP not supported */
    MTX_ERR_ZLIB_NOT_SUPPORTED,       /* zlib not supported */
    MTX_ERR_BLAS_NOT_SUPPORTED,       /* BLAS not supported */
    MTX_ERR_LIBPNG_NOT_SUPPORTED,     /* libpng not supported */
    MTX_ERR_MPI,                      /* MPI error */
    MTX_ERR_MPI_COLLECTIVE,           /* MPI collective error */
    MTX_ERR_MPI_NOT_INITIALIZED,      /* MPI not initialized */
    MTX_ERR_BLAS,                     /* BLAS error */
    MTX_ERR_EOF,                      /* unexpected end-of-file */
    MTX_ERR_LINE_TOO_LONG,            /* line exceeds maximum length */
    MTX_ERR_INVALID_MTX_HEADER,       /* invalid Matrix Market header */
    MTX_ERR_INVALID_MTX_OBJECT,       /* invalid Matrix Market object */
    MTX_ERR_INVALID_MTX_FORMAT,       /* invalid Matrix Market format */
    MTX_ERR_INVALID_MTX_FIELD,        /* invalid Matrix Market field */
    MTX_ERR_INVALID_MTX_SYMMETRY,     /* invalid Matrix Market symmetry */
    MTX_ERR_INVALID_MTX_TRIANGLE,     /* invalid Matrix Market triangle */
    MTX_ERR_INVALID_MTX_SORTING,      /* invalid Matrix Market sorting */
    MTX_ERR_INVALID_MTX_ORDERING,     /* invalid Matrix Market ordering */
    MTX_ERR_INVALID_MTX_ASSEMBLY,     /* invalid Matrix Market assembly */
    MTX_ERR_INVALID_MTX_COMMENT,      /* invalid Matrix Market comment line */
    MTX_ERR_INVALID_MTX_SIZE,         /* invalid Matrix Market size info */
    MTX_ERR_INVALID_MTX_DATA,         /* invalid Matrix Market data */
    MTX_ERR_INVALID_PRECISION,        /* invalid precision */
    MTX_ERR_INVALID_INDEX_SET_TYPE,   /* invalid index set type */
    MTX_ERR_INVALID_PARTITION_TYPE,   /* invalid partition type */
    MTX_ERR_INVALID_STREAM_TYPE,      /* invalid stream type */
    MTX_ERR_INVALID_FORMAT_SPECIFIER, /* invalid format specifier */
    MTX_ERR_INDEX_OUT_OF_BOUNDS,      /* index out of bounds */
    MTX_ERR_NO_BUFFER_SPACE,          /* not enough space in buffer */
    MTX_ERR_NOT_CONVERGED,            /* iterative method did not converge */
    MTX_ERR_INVALID_PATH_FORMAT,      /* invalid path format */
    MTX_ERR_INVALID_FIELD,            /* invalid field */
    MTX_ERR_INVALID_SYMMETRY,         /* invalid symmetry */
    MTX_ERR_INVALID_VECTOR_TYPE,      /* invalid vector type */
    MTX_ERR_INCOMPATIBLE_MTX_OBJECT,  /* incompatible Matrix Market object */
    MTX_ERR_INCOMPATIBLE_MTX_FORMAT,  /* incompatible Matrix Market format */
    MTX_ERR_INCOMPATIBLE_MTX_FIELD,   /* incompatible Matrix Market field */
    MTX_ERR_INCOMPATIBLE_MTX_SYMMETRY,/* incompatible Matrix Market symmetry */
    MTX_ERR_INCOMPATIBLE_MTX_SIZE,    /* incompatible Matrix Market size */
    MTX_ERR_INCOMPATIBLE_FIELD,       /* incompatible field */
    MTX_ERR_INCOMPATIBLE_PRECISION,   /* incompatible precision */
    MTX_ERR_INCOMPATIBLE_SYMMETRY,    /* incompatible symmetry */
    MTX_ERR_INCOMPATIBLE_SIZE,        /* incompatible size */
    MTX_ERR_INCOMPATIBLE_VECTOR_TYPE, /* incompatible vector type */
    MTX_ERR_INCOMPATIBLE_PATTERN,     /* incompatible sparsity pattern */
    MTX_ERR_INVALID_PARTITION,        /* invalid partition */
    MTX_ERR_INCOMPATIBLE_PARTITION,   /* incompatible partition */
    MTX_ERR_INVALID_SORTING,          /* invalid sorting */
    MTX_ERR_INVALID_ORDERING,         /* invalid ordering */
    MTX_ERR_INVALID_MATRIX_TYPE,      /* invalid matrix type */
    MTX_ERR_INCOMPATIBLE_MATRIX_TYPE, /* incompatible matrix type */
    MTX_ERR_INVALID_TRANSPOSITION,    /* invalid transposition */
    MTX_ERR_INVALID_PROCESS_GRID,     /* invalid process grid */
    MTX_ERR_INVALID_MPI_COMM,         /* invalid MPI communicator */
    MTX_ERR_INCOMPATIBLE_MPI_COMM,    /* incompatible MPI communicator */
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
 * ‘mtxblaserror()’ returns ‘MTX_ERR_BLAS’ if an error occurred in a
 * BLAS routine, and ‘MTX_SUCCESS’ otherwise.
 */
int mtxblaserror(void);

/**
 * ‘mtxblaserrorclear()’ clears any error flags that may have been set
 * during error handling in BLAS routines.
 */
int mtxblaserrorclear(void);

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
