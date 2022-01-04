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

#include <libmtx/libmtx-config.h>

#include <libmtx/error.h>

#ifdef LIBMTX_HAVE_MPI
#include <mpi.h>
#endif

#include <errno.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * ‘mtx_strerror()’ is a string describing an error code.
 *
 * The error code ‘err’ must correspond to one of the error codes
 * defined in the ‘mtxerror’ enum type.
 *
 * If ‘err’ is ‘MTX_ERR_ERRNO’, then ‘mtx_strerror()’ will use the
 * current value of ‘errno’ to obtain a description of the error.
 *
 * If ‘err’ may be ‘MTX_ERR_MPI’, then ‘mtx_strerror_mpi()’ should be
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
    case MTX_ERR_MPI_COLLECTIVE:
        return "Collective MPI error";
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
        return "invalid Matrix Market comment line; "
            "comments must begin with ‘%’ and end with ‘\\n’";
    case MTX_ERR_INVALID_MTX_SIZE:
        return "invalid Matrix Market size";
    case MTX_ERR_INVALID_MTX_DATA:
        return "invalid Matrix Market data";
    case MTX_ERR_INVALID_PRECISION:
        return "invalid precision";
    case MTX_ERR_INVALID_INDEX_SET_TYPE:
        return "invalid index set type";
    case MTX_ERR_INVALID_PARTITION_TYPE:
        return "invalid partition type";
    case MTX_ERR_INVALID_STREAM_TYPE:
        return "invalid stream type";
    case MTX_ERR_INVALID_FORMAT_SPECIFIER:
        return "invalid format specifier";
    case MTX_ERR_INDEX_OUT_OF_BOUNDS:
        return "index out of bounds";
    case MTX_ERR_NO_BUFFER_SPACE:
        return "not enough space in buffer";
    case MTX_ERR_NOT_CONVERGED:
        return "not converged";
    case MTX_ERR_INVALID_PATH_FORMAT:
        return "invalid path format";
    case MTX_ERR_INVALID_FIELD:
        return "invalid field";
    case MTX_ERR_INVALID_VECTOR_TYPE:
        return "invalid vector type";
    case MTX_ERR_INCOMPATIBLE_MTX_OBJECT:
        return "incompatible Matrix Market object";
    case MTX_ERR_INCOMPATIBLE_MTX_FORMAT:
        return "incompatible Matrix Market format";
    case MTX_ERR_INCOMPATIBLE_MTX_FIELD:
        return "incompatible Matrix Market field";
    case MTX_ERR_INCOMPATIBLE_MTX_SYMMETRY:
        return "incompatible Matrix Market symmetry";
    case MTX_ERR_INCOMPATIBLE_MTX_SIZE:
        return "incompatible Matrix Market size";
    case MTX_ERR_INCOMPATIBLE_FIELD:
        return "incompatible field";
    case MTX_ERR_INCOMPATIBLE_PRECISION:
        return "incompatible precision";
    case MTX_ERR_INCOMPATIBLE_SIZE:
        return "incompatible size";
    case MTX_ERR_INCOMPATIBLE_VECTOR_TYPE:
        return "incompatible vector type";
    case MTX_ERR_INCOMPATIBLE_PARTITION:
        return "incompatible partition";
    case MTX_ERR_INVALID_ORDERING:
        return "invalid ordering";
    case MTX_ERR_INVALID_MATRIX_TYPE:
        return "invalid matrix type";
    case MTX_ERR_INCOMPATIBLE_MATRIX_TYPE:
        return "incompatible matrix type";
    case MTX_ERR_INVALID_TRANS_TYPE:
        return "invalid transpose type";
    default:
        return "unknown error";
    }
}

/**
 * ‘mtx_strerror_mpi()’ is a string describing an error code.
 *
 * The error code ‘err’ must correspond to one of the error codes
 * defined in the ‘mtxerror’ enum type.
 *
 * ‘mtx_strerror_mpi()’ should be used in cases where ‘err’ may be
 * ‘MTX_ERR_MPI’, because it provides a more specific error description
 * than ‘mtx_strerror()’.
 *
 * If ‘err’ is ‘MTX_ERR_MPI’, then the argument ‘mpierrcode’ should be
 * set to the error code that was returned from the MPI function call
 * that failed. In addition, the argument ‘mpierrstr’ must be a char
 * array whose length is at least equal to ‘MPI_MAX_ERROR_STRING’. In
 * this case, ‘MPI_Error_string’ will be used to obtain a description
 * of the error.
 *
 * Otherwise, ‘mtx_strerror_mpi()’ returns the same error description
 * as ‘mtx_strerror()’ for error codes other than ‘MTX_ERR_MPI’.
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

#ifdef LIBMTX_HAVE_MPI
/**
 * ‘mtxmpierror_alloc()’ allocates storage needed for the MPI error
 * handling data structure ‘mtxmpierror’.
 */
int mtxmpierror_alloc(
    struct mtxmpierror * mpierror,
    MPI_Comm comm,
    int * mpierrcode)
{
    int err;
    int comm_size;
    err = MPI_Comm_size(comm, &comm_size);
    if (err) {
        if (mpierrcode) *mpierrcode = err;
        return MTX_ERR_MPI;
    }

    int rank;
    err = MPI_Comm_rank(comm, &rank);
    if (err) {
        if (mpierrcode) *mpierrcode = err;
        return MTX_ERR_MPI;
    }

    int (* buf)[3] = malloc(3*comm_size * sizeof(int));
    if (!buf)
        return MTX_ERR_ERRNO;

    mpierror->comm = comm;
    mpierror->comm_size = comm_size;
    mpierror->rank = rank;
    mpierror->buf = buf;
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpierror_free()’ frees storage held by ‘struct mtxmpierror’.
 */
void mtxmpierror_free(
    struct mtxmpierror * mpierror)
{
    free(mpierror->buf);
}

/**
 * ‘mtxmpierror_description()’ returns a string describing an MPI
 * error.
 *
 * The caller is responsible for freeing the storage required for the
 * string that is returned by calling ‘free()’.
 */
char * mtxmpierror_description(
    struct mtxmpierror * mpierror)
{
    char mpierrstr[MPI_MAX_ERROR_STRING];
    int comm_err = MTX_SUCCESS;
    for (int p = 0; p < mpierror->comm_size; p++) {
        if (mpierror->buf[p][1])
            comm_err = MTX_ERR_MPI_COLLECTIVE;
    }
    if (comm_err == MTX_SUCCESS)
        return strdup(mtx_strerror(MTX_SUCCESS));

    const char * format_header = "%s";
    const char * format_err_first = ": rank %d - %s";
    const char * format_err = ", rank %d - %s";

    int len = snprintf(NULL, 0, format_header, mtx_strerror(MTX_ERR_MPI_COLLECTIVE));
    int num_errors = 0;
    for (int p = 0; p < mpierror->comm_size; p++) {
        if (mpierror->buf[p][1]) {
            if (mpierror->buf[p][1] == MTX_ERR_ERRNO)
                errno = mpierror->buf[p][2];
            len += snprintf(
                NULL, 0, num_errors == 0 ? format_err_first : format_err,
                mpierror->buf[p][0],
                mtx_strerror_mpi(
                    mpierror->buf[p][1], mpierror->mpierrcode, mpierrstr));
            num_errors++;
        }
    }

    char * description = malloc(len+1);
    if (!description)
        return NULL;

    int newlen = snprintf(description, len, format_header, mtx_strerror(comm_err));
    num_errors = 0;
    for (int p = 0; p < mpierror->comm_size; p++) {
        if (mpierror->buf[p][1]) {
            if (mpierror->buf[p][1] == MTX_ERR_ERRNO)
                errno = mpierror->buf[p][2];
            newlen += snprintf(
                &description[newlen], len-newlen+1,
                num_errors == 0 ? format_err_first : format_err,
                mpierror->buf[p][0],
                mtx_strerror_mpi(
                    mpierror->buf[p][1], mpierror->mpierrcode, mpierrstr));
            num_errors++;
        }
    }
    description[len] = '\0';
    return description;
}

/**
 * ‘mtxmpierror_allreduce()’ performs a collective reduction on error
 * codes provided by each MPI process in a communicator.
 *
 * This is a collective operations that must be performed by every
 * process in the communicator of the MPI error struct ‘mpierror’.
 *
 * Each process gathers the error code and rank of every other
 * process.  If the error code of each and every process is
 * ‘MTX_SUCCESS’, then ‘mtxmpierror_allreduce()’ returns
 * ‘MTX_SUCCESS’. Otherwise, ‘MTX_ERR_MPI_COLLECTIVE’ is returned.
 * Moreover, the ‘buf’ member of ‘mpierror’ will contain the rank and
 * error code of each process.
 *
 * If the error code ‘err’ is ‘MTX_ERR_MPI_COLLECTIVE’, then it is
 * assumed that a reduction has already been performed, and
 * ‘mtxmpierror_allreduce()’ returns immediately with
 * ‘MTX_ERR_MPI_COLLECTIVE’.  As a result, if any process calls
 * ‘mtxmpierror_allreduce()’ with ‘err’ set to
 * ‘MTX_ERR_MPI_COLLECTIVE’, then every other process in the
 * communicator must also set ‘err’ to ‘MTX_ERR_MPI_COLLECTIVE’, or
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
    int err)
{
    if (err == MTX_ERR_MPI_COLLECTIVE)
        return err;

    int buf[3] = { mpierror->rank, err, errno };
    int mpierr = MPI_Allgather(
        &buf, 3, MPI_INT, mpierror->buf, 3, MPI_INT,
        mpierror->comm);
    if (mpierr)
        MPI_Abort(mpierror->comm, EXIT_FAILURE);
    for (int p = 0; p < mpierror->comm_size; p++) {
        if (mpierror->buf[p][1])
            return MTX_ERR_MPI_COLLECTIVE;
    }
    return MTX_SUCCESS;
}
#endif
