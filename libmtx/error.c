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
 * Last modified: 2021-08-09
 *
 * Error handling.
 */

#include <libmtx/libmtx-config.h>

#include <libmtx/error.h>

#ifdef LIBMTX_HAVE_MPI
#include <mpi.h>
#endif

#include <errno.h>

#include <string.h>

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
const char * mtx_strerror(
    int err)
{
    switch (err) {
    case MTX_SUCCESS:
        return "success";
    case MTX_ERR_ERRNO:
        return strerror(errno);
    case MTX_ERR_MPI:
        return "MPI error";
    case MTX_ERR_EOF:
        if (errno) return strerror(errno);
        else return "unexpected end-of-file";
    case MTX_ERR_LINE_TOO_LONG:
        return "maximum line length exceeded";
    case MTX_ERR_INVALID_MTX_HEADER:
        return "invalid Matrix Market header";
    case MTX_ERR_INVALID_MTX_OBJECT:
        return "invalid Matrix Market object";
    case MTX_ERR_INVALID_MTX_FORMAT:
        return "invalid Matrix Market format";
    case MTX_ERR_INVALID_MTX_FIELD:
        return "invalid Matrix Market field";
    case MTX_ERR_INVALID_MTX_SYMMETRY:
        return "invalid Matrix Market symmetry";
    case MTX_ERR_INVALID_MTX_SORTING:
        return "invalid Matrix Market sorting";
    case MTX_ERR_INVALID_MTX_ORDERING:
        return "invalid Matrix Market ordering";
    case MTX_ERR_INVALID_MTX_ASSEMBLY:
        return "invalid Matrix Market assembly";
    case MTX_ERR_INVALID_MTX_COMMENT:
        return "invalid Matrix Market comment line; comments must begin with `%'";
    case MTX_ERR_INVALID_MTX_SIZE:
        return "invalid Matrix Market size";
    case MTX_ERR_INVALID_MTX_DATA:
        return "invalid Matrix Market data";
    case MTX_ERR_INVALID_PRECISION:
        return "invalid precision";
    case MTX_ERR_INVALID_INDEX_SET_TYPE:
        return "invalid index set type";
    case MTX_ERR_INVALID_STREAM_TYPE:
        return "invalid stream type";
    case MTX_ERR_INVALID_FORMAT_SPECIFIER:
        return "invalid format specifier";
    case MTX_ERR_INDEX_OUT_OF_BOUNDS:
        return "index out of bounds";
    case MTX_ERR_NOT_CONVERGED:
        return "not converged";
    default:
        return "unknown error";
    }
}

/**
 * `mtx_strerror_mpi()' is a string describing an error code.
 *
 * The error code `err' must correspond to one of the error codes
 * defined in the `mtx_error' enum type.
 *
 * `mtx_strerror_mpi()' should be used in cases where `err' may be
 * `MTX_ERR_MPI', because it provides a more specific error description
 * than `mtx_strerror()'.
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
    char * mpierrstr)
{
    if (err == MTX_ERR_MPI) {
#ifdef LIBMTX_HAVE_MPI
        int mpierrstrlen;
        MPI_Error_string(mpierrcode, mpierrstr, &mpierrstrlen);
        return mpierrstr;
#else
        return "unknown MPI error";
#endif
    } else {
        return mtx_strerror(err);
    }
    return mtx_strerror(err);
}
