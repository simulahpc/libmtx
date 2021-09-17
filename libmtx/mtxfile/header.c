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
 * Matrix Market file headers.
 */

#include <libmtx/libmtx-config.h>

#include <libmtx/error.h>
#include <libmtx/mtxfile/header.h>

#ifdef LIBMTX_HAVE_MPI
#include <mpi.h>
#endif

#ifdef LIBMTX_HAVE_LIBZ
#include <zlib.h>
#endif

#include <unistd.h>

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Formatting and parsing of Matrix Market header data types.
 */

/**
 * `mtxfile_object_str()' is a string representing the Matrix Market
 * object type.
 */
const char * mtxfile_object_str(
    enum mtxfile_object object)
{
    switch (object) {
    case mtxfile_matrix: return "matrix";
    case mtxfile_vector: return "vector";
    default: return mtx_strerror(MTX_ERR_INVALID_MTX_OBJECT);
    }
}

/**
 * `mtxfile_parse_object()' parses a string containing the `object' of
 * a Matrix Market file format header.
 *
 * `valid_delimiters' is either `NULL', in which case it is ignored,
 * or it is a string of characters considered to be valid delimiters
 * for the parsed string.  That is, if there are any remaining,
 * non-NULL characters after parsing, then then the next character is
 * searched for in `valid_delimiters'.  If the character is found,
 * then the parsing succeeds and the final delimiter character is
 * consumed by the parser. Otherwise, the parsing fails with an error.
 *
 * If `endptr' is not `NULL', then the address stored in `endptr'
 * points to the first character beyond the characters that were
 * consumed during parsing.
 *
 * On success, `mtxfile_parse_object()' returns `MTX_SUCCESS' and
 * `object' is set according to the parsed string and `bytes_read' is
 * set to the number of bytes that were consumed by the parser.
 * Otherwise, an error code is returned.
 */
int mtxfile_parse_object(
    enum mtxfile_object * object,
    int64_t * bytes_read,
    const char ** endptr,
    const char * s,
    const char * valid_delimiters)
{
    const char * t = s;
    if (strncmp("matrix", t, strlen("matrix")) == 0) {
        t += strlen("matrix");
        *object = mtxfile_matrix;
    } else if (strncmp("vector", t, strlen("vector")) == 0) {
        t += strlen("vector");
        *object = mtxfile_vector;
    } else {
        return MTX_ERR_INVALID_MTX_OBJECT;
    }
    if (valid_delimiters && *t != '\0') {
        if (!strchr(valid_delimiters, *t))
            return MTX_ERR_INVALID_MTX_OBJECT;
        t++;
    }
    if (bytes_read)
        *bytes_read += t-s;
    if (endptr)
        *endptr = t;
    return MTX_SUCCESS;
}

/**
 * `mtxfile_format_str()' is a string representing the Matrix Market
 * format type.
 */
const char * mtxfile_format_str(
    enum mtxfile_format format)
{
    switch (format) {
    case mtxfile_array: return "array";
    case mtxfile_coordinate: return "coordinate";
    default: return mtx_strerror(MTX_ERR_INVALID_MTX_FORMAT);
    }
}

/**
 * `mtxfile_parse_format()' parses a string containing the `format' of
 * a Matrix Market file format header.
 *
 * `valid_delimiters' is either `NULL', in which case it is ignored,
 * or it is a string of characters considered to be valid delimiters
 * for the parsed string.  That is, if there are any remaining,
 * non-NULL characters after parsing, then then the next character is
 * searched for in `valid_delimiters'.  If the character is found,
 * then the parsing succeeds and the final delimiter character is
 * consumed by the parser. Otherwise, the parsing fails with an error.
 *
 * If `endptr' is not `NULL', then the address stored in `endptr'
 * points to the first character beyond the characters that were
 * consumed during parsing.
 *
 * On success, `mtxfile_parse_format()' returns `MTX_SUCCESS' and
 * `format' is set according to the parsed string and `bytes_read' is
 * set to the number of bytes that were consumed by the parser.
 * Otherwise, an error code is returned.
 */
int mtxfile_parse_format(
    enum mtxfile_format * format,
    int64_t * bytes_read,
    const char ** endptr,
    const char * s,
    const char * valid_delimiters)
{
    const char * t = s;
    if (strncmp("array", t, strlen("array")) == 0) {
        t += strlen("array");
        *format = mtxfile_array;
    } else if (strncmp("coordinate", t, strlen("coordinate")) == 0) {
        t += strlen("coordinate");
        *format = mtxfile_coordinate;
    } else {
        return MTX_ERR_INVALID_MTX_FORMAT;
    }
    if (valid_delimiters && *t != '\0') {
        if (!strchr(valid_delimiters, *t))
            return MTX_ERR_INVALID_MTX_FORMAT;
        t++;
    }
    if (bytes_read)
        *bytes_read += t-s;
    if (endptr)
        *endptr = t;
    return MTX_SUCCESS;
}

/**
 * `mtxfile_field_str()' is a string representing the Matrix Market
 * field type.
 */
const char * mtxfile_field_str(
    enum mtxfile_field field)
{
    switch (field) {
    case mtxfile_real: return "real";
    case mtxfile_complex: return "complex";
    case mtxfile_integer: return "integer";
    case mtxfile_pattern: return "pattern";
    default: return mtx_strerror(MTX_ERR_INVALID_MTX_FIELD);
    }
}

/**
 * `mtxfile_parse_field()' parses a string containing the `field' of a
 * Matrix Market file format header.
 *
 * `valid_delimiters' is either `NULL', in which case it is ignored,
 * or it is a string of characters considered to be valid delimiters
 * for the parsed string.  That is, if there are any remaining,
 * non-NULL characters after parsing, then then the next character is
 * searched for in `valid_delimiters'.  If the character is found,
 * then the parsing succeeds and the final delimiter character is
 * consumed by the parser. Otherwise, the parsing fails with an error.
 *
 * If `endptr' is not `NULL', then the address stored in `endptr'
 * points to the first character beyond the characters that were
 * consumed during parsing.
 *
 * On success, `mtxfile_parse_field()' returns `MTX_SUCCESS' and
 * `field' is set according to the parsed string and `bytes_read' is
 * set to the number of bytes that were consumed by the parser.
 * Otherwise, an error code is returned.
 */
int mtxfile_parse_field(
    enum mtxfile_field * field,
    int64_t * bytes_read,
    const char ** endptr,
    const char * s,
    const char * valid_delimiters)
{
    const char * t = s;
    if (strncmp("real", t, strlen("real")) == 0) {
        t += strlen("real");
        *field = mtxfile_real;
    } else if (strncmp("complex", t, strlen("complex")) == 0) {
        t += strlen("complex");
        *field = mtxfile_complex;
    } else if (strncmp("integer", t, strlen("integer")) == 0) {
        t += strlen("integer");
        *field = mtxfile_integer;
    } else if (strncmp("pattern", t, strlen("pattern")) == 0) {
        t += strlen("pattern");
        *field = mtxfile_pattern;
    } else {
        return MTX_ERR_INVALID_MTX_FIELD;
    }
    if (valid_delimiters && *t != '\0') {
        if (!strchr(valid_delimiters, *t))
            return MTX_ERR_INVALID_MTX_FIELD;
        t++;
    }
    if (bytes_read)
        *bytes_read += t-s;
    if (endptr)
        *endptr = t;
    return MTX_SUCCESS;
}

/**
 * `mtxfile_symmetry_str()' is a string representing the Matrix Market
 * symmetry type.
 */
const char * mtxfile_symmetry_str(
    enum mtxfile_symmetry symmetry)
{
    switch (symmetry) {
    case mtxfile_general: return "general";
    case mtxfile_symmetric: return "symmetric";
    case mtxfile_skew_symmetric: return "skew-symmetric";
    case mtxfile_hermitian: return "hermitian";
    default: return mtx_strerror(MTX_ERR_INVALID_MTX_SYMMETRY);
    }
}

/**
 * `mtxfile_parse_symmetry()' parses a string containing the
 * `symmetry' of a Matrix Market file format header.
 *
 * `valid_delimiters' is either `NULL', in which case it is ignored,
 * or it is a string of characters considered to be valid delimiters
 * for the parsed string.  That is, if there are any remaining,
 * non-NULL characters after parsing, then then the next character is
 * searched for in `valid_delimiters'.  If the character is found,
 * then the parsing succeeds and the final delimiter character is
 * consumed by the parser. Otherwise, the parsing fails with an error.
 *
 * If `endptr' is not `NULL', then the address stored in `endptr'
 * points to the first character beyond the characters that were
 * consumed during parsing.
 *
 * On success, `mtxfile_parse_symmetry()' returns `MTX_SUCCESS' and
 * `symmetry' is set according to the parsed string and `bytes_read'
 * is set to the number of bytes that were consumed by the parser.
 * Otherwise, an error code is returned.
 */
int mtxfile_parse_symmetry(
    enum mtxfile_symmetry * symmetry,
    int64_t * bytes_read,
    const char ** endptr,
    const char * s,
    const char * valid_delimiters)
{
    const char * t = s;
    if (strncmp("general", t, strlen("general")) == 0) {
        t += strlen("general");
        *symmetry = mtxfile_general;
    } else if (strncmp("symmetric", t, strlen("symmetric")) == 0) {
        t += strlen("symmetric");
        *symmetry = mtxfile_symmetric;
    } else if (strncmp("hermitian", t, strlen("hermitian")) == 0) {
        t += strlen("hermitian");
        *symmetry = mtxfile_hermitian;
    } else if (strncmp("Hermitian", t, strlen("Hermitian")) == 0) {
        t += strlen("Hermitian");
        *symmetry = mtxfile_hermitian;
    } else if (strncmp("skew-symmetric", t, strlen("skew-symmetric")) == 0) {
        t += strlen("skew-symmetric");
        *symmetry = mtxfile_skew_symmetric;
    } else {
        return MTX_ERR_INVALID_MTX_SYMMETRY;
    }
    if (valid_delimiters && *t != '\0') {
        if (!strchr(valid_delimiters, *t))
            return MTX_ERR_INVALID_MTX_SYMMETRY;
        t++;
    }
    if (bytes_read)
        *bytes_read += t-s;
    if (endptr)
        *endptr = t;
    return MTX_SUCCESS;
}

/*
 * Matrix Market header.
 */

static int mtxfile_parse_identifier(
    int64_t * bytes_read,
    const char ** endptr,
    const char * s,
    const char * valid_delimiters)
{
    const char * t = s;
    if (strncmp("%%MatrixMarket", t, strlen("%%MatrixMarket")) == 0) {
        t += strlen("%%MatrixMarket");
    } else {
        return MTX_ERR_INVALID_MTX_HEADER;
    }
    if (valid_delimiters && *t != '\0') {
        if (!strchr(valid_delimiters, *t))
            return MTX_ERR_INVALID_MTX_HEADER;
        t++;
    }
    if (bytes_read)
        *bytes_read += t-s;
    if (endptr)
        *endptr = t;
    return MTX_SUCCESS;
}

/**
 * `mtxfile_parse_header()' parses a string containing the header line
 * for a file in Matrix Market format.
 *
 * If `endptr' is not `NULL', then the address stored in `endptr'
 * points to the first character beyond the characters that were
 * consumed during parsing.
 *
 * On success, `mtxfile_parse_header()' returns `MTX_SUCCESS' and the
 * `object', `format', `field' and `symmetry' fields of the header
 * will be set according to the contents of the parsed Matrix Market
 * header.  Otherwise, an appropriate error code is returned if the
 * input is not a valid Matrix Market header.
 */
int mtxfile_parse_header(
    struct mtxfile_header * header,
    int64_t * bytes_read,
    const char ** endptr,
    const char * s)
{
    int err;
    const char * t = s;
    if (bytes_read)
        *bytes_read = 0;
    err = mtxfile_parse_identifier(bytes_read, &t, t, " ");
    if (err)
        return err;
    err = mtxfile_parse_object(&header->object, bytes_read, &t, t, " ");
    if (err)
        return err;
    err = mtxfile_parse_format(&header->format, bytes_read, &t, t, " ");
    if (err)
        return err;
    err = mtxfile_parse_field(&header->field, bytes_read, &t, t, " ");
    if (err)
        return err;
    err = mtxfile_parse_symmetry(&header->symmetry, bytes_read, &t, t, "\n");
    if (err)
        return err;
    if (endptr)
        *endptr = t;
    return MTX_SUCCESS;
}

/**
 * `mtxfile_header_copy()' copies a Matrix Market header.
 */
int mtxfile_header_copy(
    struct mtxfile_header * dst,
    const struct mtxfile_header * src)
{
    dst->object = src->object;
    dst->format = src->format;
    dst->field = src->field;
    dst->symmetry = src->symmetry;
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
 * `mtxfile_fread_header()` reads a Matrix Market header from a
 * stream.
 *
 * If an error code is returned, then `lines_read' and `bytes_read'
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the Matrix Market file.
 */
int mtxfile_fread_header(
    struct mtxfile_header * header,
    FILE * f,
    int * lines_read,
    int64_t * bytes_read,
    size_t line_max,
    char * linebuf)
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
    err = mtxfile_parse_header(header, bytes_read, NULL, linebuf);
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
 * `mtxfile_gzread_header()` reads a Matrix Market header from a
 * gzip-compressed stream.
 *
 * If an error code is returned, then `lines_read' and `bytes_read'
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the Matrix Market file.
 */
int mtxfile_gzread_header(
    struct mtxfile_header * header,
    gzFile f,
    int * lines_read,
    int64_t * bytes_read,
    size_t line_max,
    char * linebuf)
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
    err = mtxfile_parse_header(header, bytes_read, NULL, linebuf);
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
 * `mtxfile_header_fwrite()' writes the header line of a Matrix Market
 * file to a stream.
 *
 * If it is not `NULL', then the number of bytes written to the stream
 * is returned in `bytes_written'.
 */
int mtxfile_header_fwrite(
    const struct mtxfile_header * header,
    FILE * f,
    int64_t * bytes_written)
{
    int ret = fprintf(
        f, "%%%%MatrixMarket %s %s %s %s\n",
        mtxfile_object_str(header->object),
        mtxfile_format_str(header->format),
        mtxfile_field_str(header->field),
        mtxfile_symmetry_str(header->symmetry));
    if (ret < 0)
        return MTX_ERR_ERRNO;
    if (bytes_written)
        *bytes_written += ret;
    return MTX_SUCCESS;
}

#ifdef LIBMTX_HAVE_LIBZ
/**
 * `mtxfile_header_gzwrite()' writes the header line of a Matrix
 * Market file to a gzip-compressed stream.
 *
 * If it is not `NULL', then the number of bytes written to the stream
 * is returned in `bytes_written'.
 */
int mtxfile_header_gzwrite(
    const struct mtxfile_header * header,
    gzFile f,
    int64_t * bytes_written)
{
    int ret = gzprintf(
        f, "%%%%MatrixMarket %s %s %s %s\n",
        mtxfile_object_str(header->object),
        mtxfile_format_str(header->format),
        mtxfile_field_str(header->field),
        mtxfile_symmetry_str(header->symmetry));
    if (ret < 0)
        return MTX_ERR_ERRNO;
    if (bytes_written)
        *bytes_written += ret;
    return MTX_SUCCESS;
}
#endif

/*
 * MPI functions
 */

#ifdef LIBMTX_HAVE_MPI
/**
 * `mtxfile_header_send()' sends a Matrix Market header to another MPI
 * process.
 *
 * This is analogous to `MPI_Send()' and requires the receiving
 * process to perform a matching call to `mtxfile_header_recv()'.
 */
int mtxfile_header_send(
    const struct mtxfile_header * header,
    int dest,
    int tag,
    MPI_Comm comm,
    struct mtxmpierror * mpierror)
{
    mpierror->err = MPI_Send(
        &header->object, 1, MPI_INT, dest, tag, comm);
    if (mpierror->err)
        return MTX_ERR_MPI;
    mpierror->err = MPI_Send(
        &header->format, 1, MPI_INT, dest, tag, comm);
    if (mpierror->err)
        return MTX_ERR_MPI;
    mpierror->err = MPI_Send(
        &header->field, 1, MPI_INT, dest, tag, comm);
    if (mpierror->err)
        return MTX_ERR_MPI;
    mpierror->err = MPI_Send(
        &header->symmetry, 1, MPI_INT, dest, tag, comm);
    if (mpierror->err)
        return MTX_ERR_MPI;
    return MTX_SUCCESS;
}

/**
 * `mtxfile_header_recv()' receives a Matrix Market header from
 * another MPI process.
 *
 * This is analogous to `MPI_Recv()' and requires the sending process
 * to perform a matching call to `mtxfile_header_send()'.
 */
int mtxfile_header_recv(
    struct mtxfile_header * header,
    int source,
    int tag,
    MPI_Comm comm,
    struct mtxmpierror * mpierror)
{
    mpierror->err = MPI_Recv(
        &header->object, 1, MPI_INT, source, tag, comm, MPI_STATUS_IGNORE);
    if (mpierror->err)
        return MTX_ERR_MPI;
    mpierror->err = MPI_Recv(
        &header->format, 1, MPI_INT, source, tag, comm, MPI_STATUS_IGNORE);
    if (mpierror->err)
        return MTX_ERR_MPI;
    mpierror->err = MPI_Recv(
        &header->field, 1, MPI_INT, source, tag, comm, MPI_STATUS_IGNORE);
    if (mpierror->err)
        return MTX_ERR_MPI;
    mpierror->err = MPI_Recv(
        &header->symmetry, 1, MPI_INT, source, tag, comm, MPI_STATUS_IGNORE);
    if (mpierror->err)
        return MTX_ERR_MPI;
    return MTX_SUCCESS;
}

/**
 * `mtxfile_header_bcast()' broadcasts a Matrix Market header from an
 * MPI root process to other processes in a communicator.
 *
 * This is analogous to `MPI_Bcast()' and requires every process in
 * the communicator to perform matching calls to
 * `mtxfile_header_bcast()'.
 */
int mtxfile_header_bcast(
    struct mtxfile_header * header,
    int root,
    MPI_Comm comm,
    struct mtxmpierror * mpierror)
{
    int err;
    err = MPI_Bcast(
        &header->object, 1, MPI_INT, root, comm);
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    err = MPI_Bcast(
        &header->format, 1, MPI_INT, root, comm);
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    err = MPI_Bcast(
        &header->field, 1, MPI_INT, root, comm);
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    err = MPI_Bcast(
        &header->symmetry, 1, MPI_INT, root, comm);
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}
#endif
