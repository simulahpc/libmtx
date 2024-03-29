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
 * Last modified: 2022-04-14
 *
 * Matrix Market size lines.
 */

#include <libmtx/libmtx-config.h>

#include <libmtx/error.h>
#include <libmtx/mtxfile/header.h>
#include <libmtx/mtxfile/size.h>

#ifdef LIBMTX_HAVE_MPI
#include <mpi.h>
#endif

#ifdef LIBMTX_HAVE_LIBZ
#include <zlib.h>
#endif

#include <errno.h>
#include <unistd.h>

#include <inttypes.h>
#include <limits.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * ‘parse_long_long_int()’ parses a string to produce a number that
 * may be represented with the type ‘long long int’.
 */
static int parse_long_long_int(
    long long int * outnumber,
    int base,
    const char * s,
    char ** outendptr,
    int64_t * bytes_read)
{
    errno = 0;
    char * endptr;
    long long int number = strtoll(s, &endptr, base);
    if ((errno == ERANGE && (number == LLONG_MAX || number == LLONG_MIN)) ||
        (errno != 0 && number == 0))
        return errno;
    if (s == endptr) return EINVAL;
    if (outendptr) *outendptr = endptr;
    if (bytes_read) *bytes_read += endptr - s;
    *outnumber = number;
    return 0;
}

/**
 * ‘parse_int64()’ parses a string to produce a number that may be
 * represented as a 64-bit integer.
 *
 * The number is parsed using ‘strtoll()’, following the conventions
 * documented in the man page for that function.  In addition, some
 * further error checking is performed to ensure that the number is
 * parsed correctly.  The parsed number is stored in ‘x’.
 *
 * If ‘endptr’ is not ‘NULL’, the address stored in ‘endptr’ points to
 * the first character beyond the characters that were consumed during
 * parsing.
 *
 * On success, ‘0’ is returned. Otherwise, if the input contained
 * invalid characters, ‘EINVAL’ is returned, or if the resulting
 * number cannot be represented as a signed, 64-bit integer, ‘ERANGE’
 * is returned.
 */
static int parse_int64(
    int64_t * x,
    const char * s,
    char ** endptr,
    int64_t * bytes_read)
{
    long long int y;
    int err = parse_long_long_int(&y, 10, s, endptr, bytes_read);
    if (err) return err;
    if (y < INT64_MIN || y > INT64_MAX) return ERANGE;
    *x = y;
    return 0;
}

/**
 * ‘mtxfilesize_parse_matrix_array()’ parse a size line from a Matrix
 * Market file for a matrix in array format.
 */
static int mtxfilesize_parse_matrix_array(
    struct mtxfilesize * size,
    int64_t * bytes_read,
    char ** outendptr,
    const char * s)
{
    char * endptr;
    int err = parse_int64(&size->num_rows, s, &endptr, bytes_read);
    if (err) return err;
    if (s == endptr || *endptr != ' ') return MTX_ERR_INVALID_MTX_SIZE;
    if (bytes_read) (*bytes_read)++;
    if (outendptr) *outendptr = endptr+1;
    s = endptr+1;
    err = parse_int64(&size->num_columns, s, &endptr, bytes_read);
    if (err) return err;
    if (s == endptr) return MTX_ERR_INVALID_MTX_SIZE;
    if (outendptr) *outendptr = endptr;
    size->num_nonzeros = -1;
    return MTX_SUCCESS;
}

/**
 * ‘mtxfilesize_parse_matrix_coordinate()’ parse a size line from a
 * Matrix Market file for a matrix in coordinate format.
 */
static int mtxfilesize_parse_matrix_coordinate(
    struct mtxfilesize * size,
    int64_t * bytes_read,
    char ** outendptr,
    const char * s)
{
    char * endptr;
    int err = parse_int64(&size->num_rows, s, &endptr, bytes_read);
    if (err) return err;
    if (s == endptr || *endptr != ' ') return MTX_ERR_INVALID_MTX_SIZE;
    if (bytes_read) (*bytes_read)++;
    if (outendptr) *outendptr = endptr+1;
    s = endptr+1;
    err = parse_int64(&size->num_columns, s, &endptr, bytes_read);
    if (err) return err;
    if (s == endptr || *endptr != ' ') return MTX_ERR_INVALID_MTX_SIZE;
    if (bytes_read) (*bytes_read)++;
    if (outendptr) *outendptr = endptr+1;
    s = endptr+1;
    err = parse_int64(&size->num_nonzeros, s, &endptr, bytes_read);
    if (err) return err;
    if (s == endptr) return MTX_ERR_INVALID_MTX_SIZE;
    if (outendptr) *outendptr = endptr;
    return MTX_SUCCESS;
}

/**
 * ‘mtxfilesize_parse_vector_array()‘ parse a size line from a Matrix
 * Market file for a vector in array format.
 */
int mtxfilesize_parse_vector_array(
    struct mtxfilesize * size,
    int64_t * bytes_read,
    char ** outendptr,
    const char * s)
{
    char * endptr;
    int err = parse_int64(&size->num_rows, s, &endptr, bytes_read);
    if (err) return err;
    if (s == endptr) return MTX_ERR_INVALID_MTX_SIZE;
    if (outendptr) *outendptr = endptr;
    size->num_columns = -1;
    size->num_nonzeros = -1;
    return MTX_SUCCESS;
}

/**
 * ‘mtxfilesize_parse_vector_coordinate()‘ parses a size line from a
 * Matrix Market file for a vector in coordinate format.
 */
int mtxfilesize_parse_vector_coordinate(
    struct mtxfilesize * size,
    int64_t * bytes_read,
    char ** outendptr,
    const char * s)
{
    char * endptr;
    int err = parse_int64(&size->num_rows, s, &endptr, bytes_read);
    if (err) return err;
    if (s == endptr || *endptr != ' ') return MTX_ERR_INVALID_MTX_SIZE;
    if (bytes_read) (*bytes_read)++;
    if (outendptr) *outendptr = endptr+1;
    s = endptr+1;
    size->num_columns = -1;
    err = parse_int64(&size->num_nonzeros, s, &endptr, bytes_read);
    if (err) return err;
    if (s == endptr) return MTX_ERR_INVALID_MTX_SIZE;
    if (outendptr) *outendptr = endptr;
    size->num_columns = -1;
    return MTX_SUCCESS;
}

/**
 * ‘mtxfilesize_parse()’ parses a string containing the size line for
 * a file in Matrix Market format.
 *
 * If ‘endptr’ is not ‘NULL’, then the address stored in ‘endptr’
 * points to the first character beyond the characters that were
 * consumed during parsing.
 *
 * On success, ‘mtxfilesize_parse()’ returns ‘MTX_SUCCESS’ and the
 * fields of ‘size’ will be set accordingly.  Otherwise, an
 * appropriate error code is returned.
 */
int mtxfilesize_parse(
    struct mtxfilesize * size,
    int64_t * bytes_read,
    char ** endptr,
    const char * s,
    enum mtxfileobject object,
    enum mtxfileformat format)
{
    if (object == mtxfile_matrix) {
        if (format == mtxfile_array) {
            return mtxfilesize_parse_matrix_array(
                size, bytes_read, endptr, s);
        } else if (format == mtxfile_coordinate) {
            return mtxfilesize_parse_matrix_coordinate(
                size, bytes_read, endptr, s);
        } else { return MTX_ERR_INVALID_MTX_FORMAT; }
    } else if (object == mtxfile_vector) {
        if (format == mtxfile_array) {
            return mtxfilesize_parse_vector_array(
                size, bytes_read, endptr, s);
        } else if (format == mtxfile_coordinate) {
            return mtxfilesize_parse_vector_coordinate(
                size, bytes_read, endptr, s);
        } else { return MTX_ERR_INVALID_MTX_FORMAT; }
    } else { return MTX_ERR_INVALID_MTX_OBJECT; }
}

/**
 * ‘mtxfilesize_copy()’ copies a size line.
 */
int mtxfilesize_copy(
    struct mtxfilesize * dst,
    const struct mtxfilesize * src)
{
    dst->num_rows = src->num_rows;
    dst->num_columns = src->num_columns;
    dst->num_nonzeros = src->num_nonzeros;
    return MTX_SUCCESS;
}

/**
 * ‘mtxfilesize_cat()’ updates a size line to match with the
 * concatenation of two Matrix Market files.
 */
int mtxfilesize_cat(
    struct mtxfilesize * dst,
    const struct mtxfilesize * src,
    enum mtxfileobject object,
    enum mtxfileformat format)
{
    if (format == mtxfile_array) {
        if (object == mtxfile_matrix) {
            if (dst->num_columns != src->num_columns)
                return MTX_ERR_INVALID_MTX_SIZE;
            dst->num_rows += src->num_rows;
        } else if (object == mtxfile_vector) {
            dst->num_rows += src->num_rows;
        } else { return MTX_ERR_INVALID_MTX_OBJECT; }
    } else if (format == mtxfile_coordinate) {
        if (dst->num_rows != src->num_rows ||
            dst->num_columns != src->num_columns)
            return MTX_ERR_INVALID_MTX_SIZE;
        dst->num_nonzeros += src->num_nonzeros;
    } else { return MTX_ERR_INVALID_MTX_FORMAT; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxfilesize_num_data_lines()’ computes the number of data lines
 * that are required in a Matrix Market file with the given size line
 * and symmetry.
 */
int mtxfilesize_num_data_lines(
    const struct mtxfilesize * size,
    enum mtxfilesymmetry symmetry,
    int64_t * num_data_lines)
{
    if (size->num_nonzeros >= 0) {
        *num_data_lines = size->num_nonzeros;
    } else if (size->num_rows >= 0 && size->num_columns >= 0) {
        if (symmetry == mtxfile_general) {
            if (__builtin_mul_overflow(
                    size->num_rows, size->num_columns, num_data_lines))
            {
                errno = EOVERFLOW;
                return MTX_ERR_ERRNO;
            }
        } else if (size->num_rows == size->num_columns &&
                   (symmetry == mtxfile_symmetric ||
                    symmetry == mtxfile_hermitian))
        {
            if (__builtin_mul_overflow(
                    size->num_rows, (size->num_rows+1), num_data_lines))
            {
                errno = EOVERFLOW;
                return MTX_ERR_ERRNO;
            }
            *num_data_lines /= 2;
        } else if (size->num_rows == size->num_columns &&
                   symmetry == mtxfile_skew_symmetric)
        {
            if (__builtin_mul_overflow(
                    size->num_rows, (size->num_rows-1), num_data_lines))
            {
                errno = EOVERFLOW;
                return MTX_ERR_ERRNO;
            }
            *num_data_lines /= 2;
        } else {
            return MTX_ERR_INVALID_MTX_SYMMETRY;
        }
    } else if (size->num_rows >= 0) {
        *num_data_lines = size->num_rows;
    } else {
        return MTX_ERR_INVALID_MTX_SIZE;
    }
    return MTX_SUCCESS;
}

/**
 * ‘freadline()’ reads a single line from a stream.
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
 * ‘mtxfilesize_fread()‘ reads a Matrix Market size line from a
 * stream.
 *
 * If an error code is returned, then ‘lines_read’ and ‘bytes_read’
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the Matrix Market file.
 */
int mtxfilesize_fread(
    struct mtxfilesize * size,
    FILE * f,
    int64_t * lines_read,
    int64_t * bytes_read,
    size_t line_max,
    char * linebuf,
    enum mtxfileobject object,
    enum mtxfileformat format)
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

    char * endptr;
    err = mtxfilesize_parse(
        size, bytes_read, &endptr, linebuf, object, format);
    if (err) {
        if (free_linebuf)
            free(linebuf);
        return err;
    }
    if (*endptr != '\n') return MTX_ERR_INVALID_MTX_SIZE;
    if (bytes_read) (*bytes_read)++;
    if (lines_read) (*lines_read)++;

    if (free_linebuf)
        free(linebuf);
    return MTX_SUCCESS;
}

#ifdef LIBMTX_HAVE_LIBZ
/**
 * ‘gzreadline()’ reads a single line from a gzip-compressed stream.
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
 * ‘mtxfilesize_gzread()‘ reads a Matrix Market size line from a
 * gzip-compressed stream.
 *
 * If an error code is returned, then ‘lines_read’ and ‘bytes_read’
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the Matrix Market file.
 */
int mtxfilesize_gzread(
    struct mtxfilesize * size,
    gzFile f,
    int64_t * lines_read,
    int64_t * bytes_read,
    size_t line_max,
    char * linebuf,
    enum mtxfileobject object,
    enum mtxfileformat format)
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

    char * endptr;
    err = mtxfilesize_parse(
        size, bytes_read, &endptr, linebuf, object, format);
    if (err) {
        if (free_linebuf)
            free(linebuf);
        return err;
    }
    if (*endptr != '\n') return MTX_ERR_INVALID_MTX_SIZE;
    if (bytes_read) (*bytes_read)++;
    if (lines_read) (*lines_read)++;

    if (free_linebuf)
        free(linebuf);
    return MTX_SUCCESS;
}
#endif

/**
 * ‘mtxfilesize_fwrite()’ writes the size line of a Matrix Market
 * file to a stream.
 *
 * If it is not ‘NULL’, then the number of bytes written to the stream
 * is returned in ‘bytes_written’.
 */
int mtxfilesize_fwrite(
    const struct mtxfilesize * size,
    enum mtxfileobject object,
    enum mtxfileformat format,
    FILE * f,
    int64_t * bytes_written)
{
    int ret;
    if (object == mtxfile_matrix) {
        if (format == mtxfile_array) {
            ret = fprintf(f, "%"PRId64" %"PRId64"\n",
                          size->num_rows, size->num_columns);
        } else if (format == mtxfile_coordinate) {
            ret = fprintf(
                f, "%"PRId64" %"PRId64" %"PRId64"\n",
                size->num_rows, size->num_columns, size->num_nonzeros);
        } else {
            return MTX_ERR_INVALID_MTX_FORMAT;
        }
    } else if (object == mtxfile_vector) {
        if (format == mtxfile_array) {
            ret = fprintf(f, "%"PRId64"\n", size->num_rows);
        } else if (format == mtxfile_coordinate) {
            ret = fprintf(f, "%"PRId64" %"PRId64"\n",
                          size->num_rows, size->num_nonzeros);
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
 * ‘mtxfilesize_gzwrite()’ writes the size line of a Matrix Market
 * file to a gzip-compressed stream.
 *
 * If it is not ‘NULL’, then the number of bytes written to the stream
 * is returned in ‘bytes_written’.
 */
int mtxfilesize_gzwrite(
    const struct mtxfilesize * size,
    enum mtxfileobject object,
    enum mtxfileformat format,
    gzFile f,
    int64_t * bytes_written)
{
    int ret;
    if (object == mtxfile_matrix) {
        if (format == mtxfile_array) {
            ret = gzprintf(f, "%"PRId64" %"PRId64"\n",
                           size->num_rows, size->num_columns);
        } else if (format == mtxfile_coordinate) {
            ret = gzprintf(
                f, "%"PRId64" %"PRId64" %"PRId64"\n",
                size->num_rows, size->num_columns, size->num_nonzeros);
        } else {
            return MTX_ERR_INVALID_MTX_FORMAT;
        }
    } else if (object == mtxfile_vector) {
        if (format == mtxfile_array) {
            ret = gzprintf(f, "%"PRId64"\n", size->num_rows);
        } else if (format == mtxfile_coordinate) {
            ret = gzprintf(f, "%"PRId64" %"PRId64"\n",
                           size->num_rows, size->num_nonzeros);
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
 * ‘mtxfilesize_transpose()’ tranposes the size line of a Matrix
 * Market file.
 */
int mtxfilesize_transpose(
    struct mtxfilesize * size)
{
    int64_t num_rows = size->num_rows;
    size->num_rows = size->num_columns;
    size->num_columns = num_rows;
    return MTX_SUCCESS;
}

/*
 * MPI functions
 */

#ifdef LIBMTX_HAVE_MPI
/**
 * ‘mtxfilesize_datatype()’ creates a custom MPI data type for
 * sending or receiving Matrix Market size lines.
 *
 * The user is responsible for calling ‘MPI_Type_free()’ on the
 * returned datatype.
 */
static int mtxfilesize_datatype(
    MPI_Datatype * datatype,
    int * mpierrcode)
{
    int num_elements = 1;
    int block_lengths[] = {3};
    MPI_Datatype element_types[] = {MPI_INT64_T};
    MPI_Aint element_offsets[] = {0};
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
 * ‘mtxfilesize_send()’ sends a Matrix Market size line to another
 * MPI process.
 *
 * This is analogous to ‘MPI_Send()’ and requires the receiving
 * process to perform a matching call to ‘mtxfilesize_recv()’.
 */
int mtxfilesize_send(
    const struct mtxfilesize * size,
    int dest,
    int tag,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    disterr->err = MPI_Send(
        &size->num_rows, 1, MPI_INT64_T, dest, tag, comm);
    if (disterr->err)
        return MTX_ERR_MPI;
    disterr->err = MPI_Send(
        &size->num_columns, 1, MPI_INT64_T, dest, tag, comm);
    if (disterr->err)
        return MTX_ERR_MPI;
    disterr->err = MPI_Send(
        &size->num_nonzeros, 1, MPI_INT64_T, dest, tag, comm);
    if (disterr->err)
        return MTX_ERR_MPI;
    return MTX_SUCCESS;
}

/**
 * ‘mtxfilesize_recv()’ receives a Matrix Market size line from
 * another MPI process.
 *
 * This is analogous to ‘MPI_Recv()’ and requires the sending process
 * to perform a matching call to ‘mtxfilesize_send()’.
 */
int mtxfilesize_recv(
    struct mtxfilesize * size,
    int source,
    int tag,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    disterr->err = MPI_Recv(
        &size->num_rows, 1, MPI_INT64_T, source, tag, comm, MPI_STATUS_IGNORE);
    if (disterr->err)
        return MTX_ERR_MPI;
    disterr->err = MPI_Recv(
        &size->num_columns, 1, MPI_INT64_T, source, tag, comm, MPI_STATUS_IGNORE);
    if (disterr->err)
        return MTX_ERR_MPI;
    disterr->err = MPI_Recv(
        &size->num_nonzeros, 1, MPI_INT64_T, source, tag, comm, MPI_STATUS_IGNORE);
    if (disterr->err)
        return MTX_ERR_MPI;
    return MTX_SUCCESS;
}

/**
 * ‘mtxfilesize_bcast()’ broadcasts a Matrix Market size line from an
 * MPI root process to other processes in a communicator.
 *
 * This is analogous to ‘MPI_Bcast()’ and requires every process in
 * the communicator to perform matching calls to
 * ‘mtxfilesize_bcast()’.
 */
int mtxfilesize_bcast(
    struct mtxfilesize * size,
    int root,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err;
    err = MPI_Bcast(
        &size->num_rows, 1, MPI_INT64_T, root, comm);
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;
    err = MPI_Bcast(
        &size->num_columns, 1, MPI_INT64_T, root, comm);
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;
    err = MPI_Bcast(
        &size->num_nonzeros, 1, MPI_INT64_T, root, comm);
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxfilesize_gather()’ gathers Matrix Market size lines onto an
 * MPI root process from other processes in a communicator.
 *
 * This is analogous to ‘MPI_Gather()’ and requires every process in
 * the communicator to perform matching calls to
 * ‘mtxfilesize_gather()’.
 */
int mtxfilesize_gather(
    const struct mtxfilesize * sendsize,
    struct mtxfilesize * recvsizes,
    int root,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err;
    MPI_Datatype sizetype;
    err = mtxfilesize_datatype(&sizetype, &disterr->mpierrcode);
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;
    disterr->mpierrcode = MPI_Gather(
        sendsize, 1, sizetype, recvsizes, 1, sizetype, root, comm);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;
    MPI_Type_free(&sizetype);
    return MTX_SUCCESS;
}

/**
 * ‘mtxfilesize_allgather()’ gathers Matrix Market size lines onto
 * every MPI process from other processes in a communicator.
 *
 * This is analogous to ‘MPI_Allgather()’ and requires every process
 * in the communicator to perform matching calls to this function.
 */
int mtxfilesize_allgather(
    const struct mtxfilesize * sendsize,
    struct mtxfilesize * recvsizes,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err;
    MPI_Datatype sizetype;
    err = mtxfilesize_datatype(&sizetype, &disterr->mpierrcode);
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;
    disterr->mpierrcode = MPI_Allgather(
        sendsize, 1, sizetype, recvsizes, 1, sizetype, comm);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;
    MPI_Type_free(&sizetype);
    return MTX_SUCCESS;
}

/**
 * ‘mtxfilesize_scatterv()’ scatters a Matrix Market size line from
 * an MPI root process to other processes in a communicator.
 *
 * This is analogous to ‘MPI_Scatterv()’ and requires every process in
 * the communicator to perform matching calls to
 * ‘mtxfilesize_scatterv()’.
 */
int mtxfilesize_scatterv(
    const struct mtxfilesize * sendsize,
    struct mtxfilesize * recvsize,
    enum mtxfileobject object,
    enum mtxfileformat format,
    int recvcount,
    int root,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err;
    if (format == mtxfile_array) {
        if (object == mtxfile_matrix) {
            int rank;
            err = MPI_Comm_rank(comm, &rank);
            if (mtxdisterror_allreduce(disterr, err))
                return MTX_ERR_MPI_COLLECTIVE;
            if (rank == root)
                recvsize->num_columns = sendsize->num_columns;
            err = MPI_Bcast(&recvsize->num_columns, 1, MPI_INT64_T, root, comm);
            if (mtxdisterror_allreduce(disterr, err))
                return MTX_ERR_MPI_COLLECTIVE;
            err = (recvsize->num_columns == 0)
                ? MTX_ERR_INVALID_MTX_SIZE : MTX_SUCCESS;
            if (mtxdisterror_allreduce(disterr, err))
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
        if (mtxdisterror_allreduce(disterr, err))
            return MTX_ERR_MPI_COLLECTIVE;
        if (rank == root)
            recvsize->num_rows = sendsize->num_rows;
        err = MPI_Bcast(&recvsize->num_rows, 1, MPI_INT64_T, root, comm);
        if (mtxdisterror_allreduce(disterr, err))
            return MTX_ERR_MPI_COLLECTIVE;
        if (rank == root)
            recvsize->num_columns = sendsize->num_columns;
        err = MPI_Bcast(&recvsize->num_columns, 1, MPI_INT64_T, root, comm);
        if (mtxdisterror_allreduce(disterr, err))
            return MTX_ERR_MPI_COLLECTIVE;
        recvsize->num_nonzeros = recvcount;
    } else {
        return MTX_ERR_INVALID_MTX_FORMAT;
    }
    return MTX_SUCCESS;
}
#endif
