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
 * Last modified: 2021-08-02
 *
 * Error handling.
 */

#ifndef MATRIXMARKET_ERROR_H
#define MATRIXMARKET_ERROR_H

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
    MTX_ERR_EOF = -3,                       /* unexpected end-of-file */
    MTX_ERR_LINE_TOO_LONG = -4,             /* line exceeds maximum length */
    MTX_ERR_INVALID_MTX_HEADER = -5,        /* invalid Matrix Market header */
    MTX_ERR_INVALID_MTX_OBJECT = -6,        /* invalid Matrix Market object */
    MTX_ERR_INVALID_MTX_FORMAT = -7,        /* invalid Matrix Market format */
    MTX_ERR_INVALID_MTX_FIELD = -8,         /* invalid Matrix Market field */
    MTX_ERR_INVALID_MTX_SYMMETRY = -9,      /* invalid Matrix Market symmetry */
    MTX_ERR_INVALID_MTX_SORTING = -10,      /* invalid Matrix Market sorting */
    MTX_ERR_INVALID_MTX_ORDERING = -11,     /* invalid Matrix Market ordering */
    MTX_ERR_INVALID_MTX_ASSEMBLY = -12,     /* invalid Matrix Market assembly */
    MTX_ERR_INVALID_MTX_SIZE = -13,         /* invalid Matrix Market size info */
    MTX_ERR_INVALID_MTX_DATA = -14,         /* invalid Matrix Market data */
    MTX_ERR_INVALID_INDEX_SET_TYPE = -15,   /* invalid index set type */
    MTX_ERR_INVALID_STREAM_TYPE = -16,      /* invalid stream type */
    MTX_ERR_INVALID_FORMAT_SPECIFIER = -17, /* invalid format specifier */
    MTX_ERR_INDEX_OUT_OF_BOUNDS = -18,      /* index out of bounds */
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

#endif
