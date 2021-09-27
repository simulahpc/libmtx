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
 * Last modified: 2021-09-01
 *
 * Matrix Market size lines.
 */

#include <libmtx/libmtx-config.h>

#include <libmtx/error.h>
#include <libmtx/mtxfile/header.h>
#include <libmtx/mtxfile/size.h>

#include <libmtx/util/parse.h>

#ifdef LIBMTX_HAVE_MPI
#include <mpi.h>
#endif

#ifdef LIBMTX_HAVE_LIBZ
#include <zlib.h>
#endif

#include <errno.h>
#include <unistd.h>

#include <inttypes.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * `mtxfile_parse_size_matrix_array()' parse a size line from a Matrix
 * Market file for a matrix in array format.
 */
static int mtxfile_parse_size_matrix_array(
    struct mtxfile_size * size,
    int64_t * bytes_read,
    const char ** endptr,
    const char * s)
{
    int err;
    const char * t = s;
    err = parse_int32(t, " ", &size->num_rows, &t);
    if (err == EINVAL) {
        return MTX_ERR_INVALID_MTX_SIZE;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    }
    err = parse_int32(t, "\n", &size->num_columns, &t);
    if (err == EINVAL) {
        return MTX_ERR_INVALID_MTX_SIZE;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    }
    size->num_nonzeros = -1;
    if (bytes_read)
        (*bytes_read) += t - s;
    if (endptr)
        *endptr = t;
    return MTX_SUCCESS;
}

/**
 * `mtxfile_parse_size_matrix_coordinate()' parse a size line from a
 * Matrix Market file for a matrix in coordinate format.
 */
static int mtxfile_parse_size_matrix_coordinate(
    struct mtxfile_size * size,
    int64_t * bytes_read,
    const char ** endptr,
    const char * s)
{
    int err;
    const char * t = s;
    err = parse_int32(t, " ", &size->num_rows, &t);
    if (err == EINVAL) {
        return MTX_ERR_INVALID_MTX_SIZE;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    }
    err = parse_int32(t, " ", &size->num_columns, &t);
    if (err == EINVAL) {
        return MTX_ERR_INVALID_MTX_SIZE;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    }
    err = parse_int64(t, "\n", &size->num_nonzeros, &t);
    if (err == EINVAL) {
        return MTX_ERR_INVALID_MTX_SIZE;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    }
    if (bytes_read)
        (*bytes_read) += t - s;
    if (endptr)
        *endptr = t;
    return MTX_SUCCESS;
}

/**
 * `mtxfile_parse_size_vector_array()` parse a size line from a Matrix
 * Market file for a vector in array format.
 */
int mtxfile_parse_size_vector_array(
    struct mtxfile_size * size,
    int64_t * bytes_read,
    const char ** endptr,
    const char * s)
{
    int err;
    const char * t = s;
    err = parse_int32(t, "\n", &size->num_rows, &t);
    if (err == EINVAL) {
        return MTX_ERR_INVALID_MTX_SIZE;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    }
    size->num_columns = -1;
    size->num_nonzeros = -1;
    if (bytes_read)
        (*bytes_read) += t - s;
    if (endptr)
        *endptr = t;
    return MTX_SUCCESS;
}

/**
 * `mtxfile_parse_size_vector_coordinate()` parses a size line from a
 * Matrix Market file for a vector in coordinate format.
 */
int mtxfile_parse_size_vector_coordinate(
    struct mtxfile_size * size,
    int64_t * bytes_read,
    const char ** endptr,
    const char * s)
{
    int err;
    const char * t = s;
    err = parse_int32(t, " ", &size->num_rows, &t);
    if (err == EINVAL) {
        return MTX_ERR_INVALID_MTX_SIZE;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    }
    err = parse_int64(t, "\n", &size->num_nonzeros, &t);
    if (err == EINVAL) {
        return MTX_ERR_INVALID_MTX_SIZE;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    }
    size->num_columns = -1;
    if (bytes_read)
        (*bytes_read) += t - s;
    if (endptr)
        *endptr = t;
    return MTX_SUCCESS;
}

/**
 * `mtxfile_parse_size()' parses a string containing the size line for
 * a file in Matrix Market format.
 *
 * If `endptr' is not `NULL', then the address stored in `endptr'
 * points to the first character beyond the characters that were
 * consumed during parsing.
 *
 * On success, `mtxfile_parse_size()' returns `MTX_SUCCESS' and the
 * fields of `size' will be set accordingly.  Otherwise, an
 * appropriate error code is returned.
 */
int mtxfile_parse_size(
    struct mtxfile_size * size,
    int64_t * bytes_read,
    const char ** endptr,
    const char * s,
    enum mtxfile_object object,
    enum mtxfile_format format)
{
    if (object == mtxfile_matrix) {
        if (format == mtxfile_array) {
            return mtxfile_parse_size_matrix_array(
                size, bytes_read, endptr, s);
        } else if (format == mtxfile_coordinate) {
            return mtxfile_parse_size_matrix_coordinate(
                size, bytes_read, endptr, s);
        } else {
            return MTX_ERR_INVALID_MTX_FORMAT;
        }
    } else if (object == mtxfile_vector) {
        if (format == mtxfile_array) {
            return mtxfile_parse_size_vector_array(
                size, bytes_read, endptr, s);
        } else if (format == mtxfile_coordinate) {
            return mtxfile_parse_size_vector_coordinate(
                size, bytes_read, endptr, s);
        } else {
            return MTX_ERR_INVALID_MTX_FORMAT;
        }
    } else {
        return MTX_ERR_INVALID_MTX_OBJECT;
    }
}

/**
 * `mtxfile_size_copy()' copies a size line.
 */
int mtxfile_size_copy(
    struct mtxfile_size * dst,
    const struct mtxfile_size * src)
{
    dst->num_rows = src->num_rows;
    dst->num_columns = src->num_columns;
    dst->num_nonzeros = src->num_nonzeros;
    return MTX_SUCCESS;
}

/**
 * `mtxfile_size_cat()' updates a size line to match with the
 * concatenation of two Matrix Market files.
 */
int mtxfile_size_cat(
    struct mtxfile_size * dst,
    const struct mtxfile_size * src,
    enum mtxfile_object object,
    enum mtxfile_format format)
{
    int err;
    if (format == mtxfile_array) {
        if (object == mtxfile_matrix) {
            if (dst->num_columns != src->num_columns)
                return MTX_ERR_INVALID_MTX_SIZE;
            dst->num_rows += src->num_rows;
        } else if (object == mtxfile_vector) {
            dst->num_rows += src->num_rows;
        } else {
            return MTX_ERR_INVALID_MTX_OBJECT;
        }
    } else if (format == mtxfile_coordinate) {
        if (dst->num_rows != src->num_rows ||
            dst->num_columns != src->num_columns)
            return MTX_ERR_INVALID_MTX_SIZE;
        dst->num_nonzeros += src->num_nonzeros;
    } else {
        return MTX_ERR_INVALID_MTX_FORMAT;
    }
    return MTX_SUCCESS;
}

/**
 * `mtxfile_size_num_data_lines()' computes the number of data lines
 * that are required in a Matrix Market file with the given size line.
 */
int mtxfile_size_num_data_lines(
    const struct mtxfile_size * size,
    int64_t * num_data_lines)
{
    if (size->num_nonzeros >= 0) {
        *num_data_lines = size->num_nonzeros;
    } else if (size->num_rows >= 0 && size->num_columns >= 0) {
        if (__builtin_mul_overflow(size->num_rows, size->num_columns, num_data_lines)) {
            errno = EOVERFLOW;
            return MTX_ERR_ERRNO;
        }
    } else if (size->num_rows >= 0) {
        *num_data_lines = size->num_rows;
    } else {
        return MTX_ERR_INVALID_MTX_SIZE;
    }
    return MTX_SUCCESS;
}

/**
 * `freadline()' reads a single line from a stream.
 */
static int freadline(
    char * linebuf,
    size_t line_max,
    FILE * f)
{
    char * s = fgets(linebuf, line_max+1, f);
    if (!s && feof(f))
        return MTX_ERR_EOF;
    else if (!s)
        return MTX_ERR_ERRNO;
    int n = strlen(s);
    if (n > 0 && n == line_max && s[n-1] != '\n')
        return MTX_ERR_LINE_TOO_LONG;
    return MTX_SUCCESS;
}

/**
 * `mtxfile_fread_size()` reads a Matrix Market size line from a
 * stream.
 *
 * If an error code is returned, then `lines_read' and `bytes_read'
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the Matrix Market file.
 */
int mtxfile_fread_size(
    struct mtxfile_size * size,
    FILE * f,
    int * lines_read,
    int64_t * bytes_read,
    size_t line_max,
    char * linebuf,
    enum mtxfile_object object,
    enum mtxfile_format format)
{
    int err;
    bool free_linebuf = !linebuf;
    if (!linebuf) {
        line_max = sysconf(_SC_LINE_MAX);
        linebuf = malloc(line_max+1);
        if (!linebuf)
            return MTX_ERR_ERRNO;
    }

    err = freadline(linebuf, line_max, f);
    if (err) {
        if (free_linebuf)
            free(linebuf);
        return err;
    }

    err = mtxfile_parse_size(
        size, bytes_read, NULL, linebuf, object, format);
    if (err) {
        if (free_linebuf)
            free(linebuf);
        return err;
    }
    if (lines_read)
        (*lines_read)++;

    if (free_linebuf)
        free(linebuf);
    return MTX_SUCCESS;
}

#ifdef LIBMTX_HAVE_LIBZ
/**
 * `gzreadline()' reads a single line from a gzip-compressed stream.
 */
static int gzreadline(
    char * linebuf,
    size_t line_max,
    gzFile f)
{
    char * s = gzgets(f, linebuf, line_max+1);
    if (!s && gzeof(f))
        return MTX_ERR_EOF;
    else if (!s)
        return MTX_ERR_ERRNO;
    int n = strlen(s);
    if (n > 0 && n == line_max && s[n-1] != '\n')
        return MTX_ERR_LINE_TOO_LONG;
    return MTX_SUCCESS;
}

/**
 * `mtxfile_gzread_size()` reads a Matrix Market size line from a
 * gzip-compressed stream.
 *
 * If an error code is returned, then `lines_read' and `bytes_read'
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the Matrix Market file.
 */
int mtxfile_gzread_size(
    struct mtxfile_size * size,
    gzFile f,
    int * lines_read,
    int64_t * bytes_read,
    size_t line_max,
    char * linebuf,
    enum mtxfile_object object,
    enum mtxfile_format format)
{
    int err;
    bool free_linebuf = !linebuf;
    if (!linebuf) {
        line_max = sysconf(_SC_LINE_MAX);
        linebuf = malloc(line_max+1);
        if (!linebuf)
            return MTX_ERR_ERRNO;
    }

    err = gzreadline(linebuf, line_max, f);
    if (err) {
        if (free_linebuf)
            free(linebuf);
        return err;
    }

    err = mtxfile_parse_size(
        size, bytes_read, NULL, linebuf, object, format);
    if (err) {
        if (free_linebuf)
            free(linebuf);
        return err;
    }
    if (lines_read)
        (*lines_read)++;

    if (free_linebuf)
        free(linebuf);
    return MTX_SUCCESS;
}
#endif

/**
 * `mtxfile_size_fwrite()' writes the size line of a Matrix Market
 * file to a stream.
 *
 * If it is not `NULL', then the number of bytes written to the stream
 * is returned in `bytes_written'.
 */
int mtxfile_size_fwrite(
    const struct mtxfile_size * size,
    enum mtxfile_object object,
    enum mtxfile_format format,
    FILE * f,
    int64_t * bytes_written)
{
    int ret;
    if (object == mtxfile_matrix) {
        if (format == mtxfile_array) {
            ret = fprintf(f, "%d %d\n", size->num_rows, size->num_columns);
        } else if (format == mtxfile_coordinate) {
            ret = fprintf(
                f, "%d %d %"PRId64"\n",
                size->num_rows, size->num_columns, size->num_nonzeros);
        } else {
            return MTX_ERR_INVALID_MTX_FORMAT;
        }
    } else if (object == mtxfile_vector) {
        if (format == mtxfile_array) {
            ret = fprintf(f, "%d\n", size->num_rows);
        } else if (format == mtxfile_coordinate) {
            ret = fprintf(f, "%d %"PRId64"\n", size->num_rows, size->num_nonzeros);
        } else {
            return MTX_ERR_INVALID_MTX_FORMAT;
        }
    } else {
        return MTX_ERR_INVALID_MTX_OBJECT;
    }
    if (ret < 0)
        return MTX_ERR_ERRNO;
    if (bytes_written)
        *bytes_written += ret;
    return MTX_SUCCESS;
}

#ifdef LIBMTX_HAVE_LIBZ
/**
 * `mtxfile_size_gzwrite()' writes the size line of a Matrix Market
 * file to a gzip-compressed stream.
 *
 * If it is not `NULL', then the number of bytes written to the stream
 * is returned in `bytes_written'.
 */
int mtxfile_size_gzwrite(
    const struct mtxfile_size * size,
    enum mtxfile_object object,
    enum mtxfile_format format,
    gzFile f,
    int64_t * bytes_written)
{
    int ret;
    if (object == mtxfile_matrix) {
        if (format == mtxfile_array) {
            ret = gzprintf(f, "%d %d\n", size->num_rows, size->num_columns);
        } else if (format == mtxfile_coordinate) {
            ret = gzprintf(
                f, "%d %d %"PRId64"\n",
                size->num_rows, size->num_columns, size->num_nonzeros);
        } else {
            return MTX_ERR_INVALID_MTX_FORMAT;
        }
    } else if (object == mtxfile_vector) {
        if (format == mtxfile_array) {
            ret = gzprintf(f, "%d\n", size->num_rows);
        } else if (format == mtxfile_coordinate) {
            ret = gzprintf(f, "%d %"PRId64"\n", size->num_rows, size->num_nonzeros);
        } else {
            return MTX_ERR_INVALID_MTX_FORMAT;
        }
    } else {
        return MTX_ERR_INVALID_MTX_OBJECT;
    }
    if (ret < 0)
        return MTX_ERR_ERRNO;
    if (bytes_written)
        *bytes_written += ret;
    return MTX_SUCCESS;
}
#endif

/*
 * Transpose
 */

/**
 * `mtxfile_size_transpose()' tranposes the size line of a Matrix
 * Market file.
 */
int mtxfile_size_transpose(
    struct mtxfile_size * size)
{
    int num_rows = size->num_rows;
    size->num_rows = size->num_columns;
    size->num_columns = num_rows;
    return MTX_SUCCESS;
}

/*
 * MPI functions
 */

#ifdef LIBMTX_HAVE_MPI
/**
 * `mtxfile_size_datatype()' creates a custom MPI data type for
 * sending or receiving Matrix Market size lines.
 *
 * The user is responsible for calling `MPI_Type_free()' on the
 * returned datatype.
 */
static int mtxfile_size_datatype(
    MPI_Datatype * datatype,
    int * mpierrcode)
{
    int num_elements = 2;
    int block_lengths[] = {2, 1};
    MPI_Datatype element_types[] = {MPI_INT, MPI_INT64_T};
    MPI_Aint element_offsets[] = {
        0, offsetof(struct mtxfile_size, num_nonzeros)};
    MPI_Datatype single_datatype;
    *mpierrcode = MPI_Type_create_struct(
        num_elements, block_lengths, element_offsets,
        element_types, &single_datatype);
    if (*mpierrcode)
        return MTX_ERR_MPI;

    /* Enable sending an array of the custom data type. */
    MPI_Aint lb, extent;
    *mpierrcode = MPI_Type_get_extent(single_datatype, &lb, &extent);
    if (*mpierrcode) {
        MPI_Type_free(&single_datatype);
        return MTX_ERR_MPI;
    }
    *mpierrcode = MPI_Type_create_resized(single_datatype, lb, extent, datatype);
    if (*mpierrcode) {
        MPI_Type_free(&single_datatype);
        return MTX_ERR_MPI;
    }
    *mpierrcode = MPI_Type_commit(datatype);
    if (*mpierrcode) {
        MPI_Type_free(datatype);
        MPI_Type_free(&single_datatype);
        return MTX_ERR_MPI;
    }
    MPI_Type_free(&single_datatype);
    return MTX_SUCCESS;
}

/**
 * `mtxfile_size_send()' sends a Matrix Market size line to another
 * MPI process.
 *
 * This is analogous to `MPI_Send()' and requires the receiving
 * process to perform a matching call to `mtxfile_size_recv()'.
 */
int mtxfile_size_send(
    const struct mtxfile_size * size,
    int dest,
    int tag,
    MPI_Comm comm,
    struct mtxmpierror * mpierror)
{
    mpierror->err = MPI_Send(
        &size->num_rows, 1, MPI_INT32_T, dest, tag, comm);
    if (mpierror->err)
        return MTX_ERR_MPI;
    mpierror->err = MPI_Send(
        &size->num_columns, 1, MPI_INT32_T, dest, tag, comm);
    if (mpierror->err)
        return MTX_ERR_MPI;
    mpierror->err = MPI_Send(
        &size->num_nonzeros, 1, MPI_INT64_T, dest, tag, comm);
    if (mpierror->err)
        return MTX_ERR_MPI;
    return MTX_SUCCESS;
}

/**
 * `mtxfile_size_recv()' receives a Matrix Market size line from
 * another MPI process.
 *
 * This is analogous to `MPI_Recv()' and requires the sending process
 * to perform a matching call to `mtxfile_size_send()'.
 */
int mtxfile_size_recv(
    struct mtxfile_size * size,
    int source,
    int tag,
    MPI_Comm comm,
    struct mtxmpierror * mpierror)
{
    mpierror->err = MPI_Recv(
        &size->num_rows, 1, MPI_INT32_T, source, tag, comm, MPI_STATUS_IGNORE);
    if (mpierror->err)
        return MTX_ERR_MPI;
    mpierror->err = MPI_Recv(
        &size->num_columns, 1, MPI_INT32_T, source, tag, comm, MPI_STATUS_IGNORE);
    if (mpierror->err)
        return MTX_ERR_MPI;
    mpierror->err = MPI_Recv(
        &size->num_nonzeros, 1, MPI_INT64_T, source, tag, comm, MPI_STATUS_IGNORE);
    if (mpierror->err)
        return MTX_ERR_MPI;
    return MTX_SUCCESS;
}

/**
 * `mtxfile_size_bcast()' broadcasts a Matrix Market size line from an
 * MPI root process to other processes in a communicator.
 *
 * This is analogous to `MPI_Bcast()' and requires every process in
 * the communicator to perform matching calls to
 * `mtxfile_size_bcast()'.
 */
int mtxfile_size_bcast(
    struct mtxfile_size * size,
    int root,
    MPI_Comm comm,
    struct mtxmpierror * mpierror)
{
    int err;
    err = MPI_Bcast(
        &size->num_rows, 1, MPI_INT32_T, root, comm);
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    err = MPI_Bcast(
        &size->num_columns, 1, MPI_INT32_T, root, comm);
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    err = MPI_Bcast(
        &size->num_nonzeros, 1, MPI_INT64_T, root, comm);
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * `mtxfile_size_gather()' gathers Matrix Market size lines onto an
 * MPI root process from other processes in a communicator.
 *
 * This is analogous to `MPI_Gather()' and requires every process in
 * the communicator to perform matching calls to
 * `mtxfile_size_gather()'.
 */
int mtxfile_size_gather(
    const struct mtxfile_size * sendsize,
    struct mtxfile_size * recvsizes,
    int root,
    MPI_Comm comm,
    struct mtxmpierror * mpierror)
{
    int err;
    MPI_Datatype sizetype;
    err = mtxfile_size_datatype(&sizetype, &mpierror->mpierrcode);
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    mpierror->mpierrcode = MPI_Gather(
        sendsize, 1, sizetype, recvsizes, 1, sizetype, root, comm);
    err = mpierror->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    MPI_Type_free(&sizetype);
    return MTX_SUCCESS;
}

/**
 * `mtxfile_size_scatterv()' scatters a Matrix Market size line from
 * an MPI root process to other processes in a communicator.
 *
 * This is analogous to `MPI_Scatterv()' and requires every process in
 * the communicator to perform matching calls to
 * `mtxfile_size_scatterv()'.
 */
int mtxfile_size_scatterv(
    const struct mtxfile_size * sendsize,
    struct mtxfile_size * recvsize,
    enum mtxfile_object object,
    enum mtxfile_format format,
    int recvcount,
    int root,
    MPI_Comm comm,
    struct mtxmpierror * mpierror)
{
    int err;
    if (format == mtxfile_array) {
        if (object == mtxfile_matrix) {
            int rank;
            err = MPI_Comm_rank(comm, &rank);
            if (mtxmpierror_allreduce(mpierror, err))
                return MTX_ERR_MPI_COLLECTIVE;
            if (rank == root)
                recvsize->num_columns = sendsize->num_columns;
            err = MPI_Bcast(&recvsize->num_columns, 1, MPI_INT32_T, root, comm);
            if (mtxmpierror_allreduce(mpierror, err))
                return MTX_ERR_MPI_COLLECTIVE;
            err = (recvsize->num_columns == 0)
                ? MTX_ERR_INVALID_MTX_SIZE : MTX_SUCCESS;
            if (mtxmpierror_allreduce(mpierror, err))
                return MTX_ERR_MPI_COLLECTIVE;
            recvsize->num_rows =
                (recvcount + recvsize->num_columns-1) / recvsize->num_columns;
            recvsize->num_nonzeros = -1;
        } else if (object == mtxfile_vector) {
            recvsize->num_rows = recvcount;
            recvsize->num_columns = -1;
            recvsize->num_nonzeros = -1;
        } else {
            return MTX_ERR_INVALID_MTX_OBJECT;
        }

    } else if (format == mtxfile_coordinate) {
        int rank;
        err = MPI_Comm_rank(comm, &rank);
        if (mtxmpierror_allreduce(mpierror, err))
            return MTX_ERR_MPI_COLLECTIVE;
        if (rank == root)
            recvsize->num_rows = sendsize->num_rows;
        err = MPI_Bcast(&recvsize->num_rows, 1, MPI_INT32_T, root, comm);
        if (mtxmpierror_allreduce(mpierror, err))
            return MTX_ERR_MPI_COLLECTIVE;
        if (rank == root)
            recvsize->num_columns = sendsize->num_columns;
        err = MPI_Bcast(&recvsize->num_columns, 1, MPI_INT32_T, root, comm);
        if (mtxmpierror_allreduce(mpierror, err))
            return MTX_ERR_MPI_COLLECTIVE;
        recvsize->num_nonzeros = recvcount;
    } else {
        return MTX_ERR_INVALID_MTX_FORMAT;
    }
    return MTX_SUCCESS;
}
#endif
