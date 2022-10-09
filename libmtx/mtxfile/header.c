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
 * Last modified: 2022-10-08
 *
 * Matrix Market file headers.
 */

#include <libmtx/libmtx-config.h>

#include <libmtx/error.h>
#include <libmtx/mtxfile/header.h>
#include <libmtx/linalg/symmetry.h>

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
 * ‘mtxfileobjectstr()’ is a string representing the Matrix Market
 * object type.
 */
const char * mtxfileobjectstr(
    enum mtxfileobject object)
{
    switch (object) {
    case mtxfile_matrix: return "matrix";
    case mtxfile_vector: return "vector";
    default: return mtxstrerror(MTX_ERR_INVALID_MTX_OBJECT);
    }
}

/**
 * ‘mtxfileobject_parse()’ parses a string containing the ‘object’ of
 * a Matrix Market file format header.
 *
 * ‘valid_delimiters’ is either ‘NULL’, in which case it is ignored,
 * or it is a string of characters considered to be valid delimiters
 * for the parsed string.  That is, if there are any remaining,
 * non-NULL characters after parsing, then then the next character is
 * searched for in ‘valid_delimiters’.  If the character is found,
 * then the parsing succeeds and the final delimiter character is
 * consumed by the parser. Otherwise, the parsing fails with an error.
 *
 * If ‘endptr’ is not ‘NULL’, then the address stored in ‘endptr’
 * points to the first character beyond the characters that were
 * consumed during parsing.
 *
 * On success, ‘mtxfileobject_parse()’ returns ‘MTX_SUCCESS’ and
 * ‘object’ is set according to the parsed string and ‘bytes_read’ is
 * set to the number of bytes that were consumed by the parser.
 * Otherwise, an error code is returned.
 */
int mtxfileobject_parse(
    enum mtxfileobject * object,
    int64_t * bytes_read,
    char ** endptr,
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
    if (bytes_read) *bytes_read += t-s;
    if (endptr) *endptr = (char *) t;
    return MTX_SUCCESS;
}

/**
 * ‘mtxfileformatstr()’ is a string representing the Matrix Market
 * format type.
 */
const char * mtxfileformatstr(
    enum mtxfileformat format)
{
    switch (format) {
    case mtxfile_array: return "array";
    case mtxfile_coordinate: return "coordinate";
    default: return mtxstrerror(MTX_ERR_INVALID_MTX_FORMAT);
    }
}

/**
 * ‘mtxfileformat_parse()’ parses a string containing the ‘format’ of
 * a Matrix Market file format header.
 *
 * ‘valid_delimiters’ is either ‘NULL’, in which case it is ignored,
 * or it is a string of characters considered to be valid delimiters
 * for the parsed string.  That is, if there are any remaining,
 * non-NULL characters after parsing, then then the next character is
 * searched for in ‘valid_delimiters’.  If the character is found,
 * then the parsing succeeds and the final delimiter character is
 * consumed by the parser. Otherwise, the parsing fails with an error.
 *
 * If ‘endptr’ is not ‘NULL’, then the address stored in ‘endptr’
 * points to the first character beyond the characters that were
 * consumed during parsing.
 *
 * On success, ‘mtxfileformat_parse()’ returns ‘MTX_SUCCESS’ and
 * ‘format’ is set according to the parsed string and ‘bytes_read’ is
 * set to the number of bytes that were consumed by the parser.
 * Otherwise, an error code is returned.
 */
int mtxfileformat_parse(
    enum mtxfileformat * format,
    int64_t * bytes_read,
    char ** endptr,
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
    if (bytes_read) *bytes_read += t-s;
    if (endptr) *endptr = (char *) t;
    return MTX_SUCCESS;
}

/**
 * ‘mtxfilefieldstr()’ is a string representing the Matrix Market
 * field type.
 */
const char * mtxfilefieldstr(
    enum mtxfilefield field)
{
    switch (field) {
    case mtxfile_real: return "real";
    case mtxfile_complex: return "complex";
    case mtxfile_integer: return "integer";
    case mtxfile_pattern: return "pattern";
    default: return mtxstrerror(MTX_ERR_INVALID_MTX_FIELD);
    }
}

/**
 * ‘mtxfilefield_parse()’ parses a string containing the ‘field’ of a
 * Matrix Market file format header.
 *
 * ‘valid_delimiters’ is either ‘NULL’, in which case it is ignored,
 * or it is a string of characters considered to be valid delimiters
 * for the parsed string.  That is, if there are any remaining,
 * non-NULL characters after parsing, then then the next character is
 * searched for in ‘valid_delimiters’.  If the character is found,
 * then the parsing succeeds and the final delimiter character is
 * consumed by the parser. Otherwise, the parsing fails with an error.
 *
 * If ‘endptr’ is not ‘NULL’, then the address stored in ‘endptr’
 * points to the first character beyond the characters that were
 * consumed during parsing.
 *
 * On success, ‘mtxfilefield_parse()’ returns ‘MTX_SUCCESS’ and
 * ‘field’ is set according to the parsed string and ‘bytes_read’ is
 * set to the number of bytes that were consumed by the parser.
 * Otherwise, an error code is returned.
 */
int mtxfilefield_parse(
    enum mtxfilefield * field,
    int64_t * bytes_read,
    char ** endptr,
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
    if (bytes_read) *bytes_read += t-s;
    if (endptr) *endptr = (char *) t;
    return MTX_SUCCESS;
}

/**
 * ‘mtxfilefield_from_mtxfield()’ converts a value to the
 * ‘mtxfilefield’ enum type to a corresponding value from the
 * ‘mtxfield’ enum type.
 */
int mtxfilefield_from_mtxfield(
    enum mtxfilefield * dst,
    enum mtxfield src)
{
    switch (src) {
    case mtx_field_real: *dst = mtxfile_real; break;
    case mtx_field_complex: *dst = mtxfile_complex; break;
    case mtx_field_integer: *dst = mtxfile_integer; break;
    case mtx_field_pattern: *dst = mtxfile_pattern; break;
    default: return MTX_ERR_INVALID_FIELD;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxfilefield_to_mtxfield()’ converts a value of the ‘mtxfilefield’
 * enum type to a corresponding value of the ‘mtxfield’ enum type.
 */
int mtxfilefield_to_mtxfield(
    enum mtxfield * dst,
    enum mtxfilefield src)
{
    switch (src) {
    case mtxfile_real: *dst = mtx_field_real; break;
    case mtxfile_complex: *dst = mtx_field_complex; break;
    case mtxfile_integer: *dst = mtx_field_integer; break;
    case mtxfile_pattern: *dst = mtx_field_pattern; break;
    default: return MTX_ERR_INVALID_MTX_FIELD;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxfilesymmetrystr()’ is a string representing the Matrix Market
 * symmetry type.
 */
const char * mtxfilesymmetrystr(
    enum mtxfilesymmetry symmetry)
{
    switch (symmetry) {
    case mtxfile_general: return "general";
    case mtxfile_symmetric: return "symmetric";
    case mtxfile_skew_symmetric: return "skew-symmetric";
    case mtxfile_hermitian: return "hermitian";
    default: return mtxstrerror(MTX_ERR_INVALID_MTX_SYMMETRY);
    }
}

/**
 * ‘mtxfilesymmetry_parse()’ parses a string containing the
 * ‘symmetry’ of a Matrix Market file format header.
 *
 * ‘valid_delimiters’ is either ‘NULL’, in which case it is ignored,
 * or it is a string of characters considered to be valid delimiters
 * for the parsed string.  That is, if there are any remaining,
 * non-NULL characters after parsing, then then the next character is
 * searched for in ‘valid_delimiters’.  If the character is found,
 * then the parsing succeeds and the final delimiter character is
 * consumed by the parser. Otherwise, the parsing fails with an error.
 *
 * If ‘endptr’ is not ‘NULL’, then the address stored in ‘endptr’
 * points to the first character beyond the characters that were
 * consumed during parsing.
 *
 * On success, ‘mtxfilesymmetry_parse()’ returns ‘MTX_SUCCESS’ and
 * ‘symmetry’ is set according to the parsed string and ‘bytes_read’
 * is set to the number of bytes that were consumed by the parser.
 * Otherwise, an error code is returned.
 */
int mtxfilesymmetry_parse(
    enum mtxfilesymmetry * symmetry,
    int64_t * bytes_read,
    char ** endptr,
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
    if (bytes_read) *bytes_read += t-s;
    if (endptr) *endptr = (char *) t;
    return MTX_SUCCESS;
}

/**
 * ‘mtxfilesymmetry_from_mtxsymmetry()’ converts a value to the
 * ‘mtxfilesymmetry’ enum type to a corresponding value from the
 * ‘mtxsymmetry’ enum type.
 */
int mtxfilesymmetry_from_mtxsymmetry(
    enum mtxfilesymmetry * dst,
    enum mtxsymmetry src)
{
    switch (src) {
    case mtx_unsymmetric: *dst = mtxfile_general; break;
    case mtx_symmetric: *dst = mtxfile_symmetric; break;
    case mtx_skew_symmetric: *dst = mtxfile_skew_symmetric; break;
    case mtx_hermitian: *dst = mtxfile_hermitian; break;
    default: return MTX_ERR_INVALID_SYMMETRY;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxfilesymmetry_to_mtxsymmetry()’ converts a value of the
 * ‘mtxfilesymmetry’ enum type to a corresponding value of the
 * ‘mtxsymmetry’ enum type.
 */
int mtxfilesymmetry_to_mtxsymmetry(
    enum mtxsymmetry * dst,
    enum mtxfilesymmetry src)
{
    switch (src) {
    case mtxfile_general: *dst = mtx_unsymmetric; break;
    case mtxfile_symmetric: *dst = mtx_symmetric; break;
    case mtxfile_skew_symmetric: *dst = mtx_skew_symmetric; break;
    case mtxfile_hermitian: *dst = mtx_hermitian; break;
    default: return MTX_ERR_INVALID_MTX_SYMMETRY;
    }
    return MTX_SUCCESS;
}

/*
 * Matrix Market header.
 */

static int mtxfile_parse_identifier(
    int64_t * bytes_read,
    char ** endptr,
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
    if (bytes_read) *bytes_read += t-s;
    if (endptr) *endptr = (char *) t;
    return MTX_SUCCESS;
}

/**
 * ‘mtxfileheader_parse()’ parses a string containing the header line
 * for a file in Matrix Market format.
 *
 * If ‘endptr’ is not ‘NULL’, then the address stored in ‘endptr’
 * points to the first character beyond the characters that were
 * consumed during parsing.
 *
 * On success, ‘mtxfileheader_parse()’ returns ‘MTX_SUCCESS’ and the
 * ‘object’, ‘format’, ‘field’ and ‘symmetry’ fields of the header
 * will be set according to the contents of the parsed Matrix Market
 * header.  Otherwise, an appropriate error code is returned if the
 * input is not a valid Matrix Market header.
 */
int mtxfileheader_parse(
    struct mtxfileheader * header,
    int64_t * bytes_read,
    char ** outendptr,
    const char * s)
{
    int err;
    char * endptr;
    if (bytes_read) *bytes_read = 0;
    err = mtxfile_parse_identifier(bytes_read, &endptr, s, " ");
    if (err) return err;
    err = mtxfileobject_parse(&header->object, bytes_read, &endptr, endptr, " ");
    if (err) return err;
    err = mtxfileformat_parse(&header->format, bytes_read, &endptr, endptr, " ");
    if (err) return err;
    err = mtxfilefield_parse(&header->field, bytes_read, &endptr, endptr, " ");
    if (err) return err;
    err = mtxfilesymmetry_parse(&header->symmetry, bytes_read, &endptr, endptr, "\n");
    if (err) return err;
    if (outendptr) *outendptr = endptr;
    return MTX_SUCCESS;
}

/**
 * ‘mtxfileheader_copy()’ copies a Matrix Market header.
 */
int mtxfileheader_copy(
    struct mtxfileheader * dst,
    const struct mtxfileheader * src)
{
    dst->object = src->object;
    dst->format = src->format;
    dst->field = src->field;
    dst->symmetry = src->symmetry;
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
 * ‘mtxfileheader_fread()’ reads a Matrix Market header from a
 * stream.
 *
 * If an error code is returned, then ‘lines_read’ and ‘bytes_read’
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the Matrix Market file.
 */
int mtxfileheader_fread(
    struct mtxfileheader * header,
    FILE * f,
    int64_t * lines_read,
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
    err = mtxfileheader_parse(header, bytes_read, NULL, linebuf);
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
 * ‘mtxfileheader_gzread()’ reads a Matrix Market header from a
 * gzip-compressed stream.
 *
 * If an error code is returned, then ‘lines_read’ and ‘bytes_read’
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the Matrix Market file.
 */
int mtxfileheader_gzread(
    struct mtxfileheader * header,
    gzFile f,
    int64_t * lines_read,
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
    err = mtxfileheader_parse(header, bytes_read, NULL, linebuf);
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
 * ‘mtxfileheader_fwrite()’ writes the header line of a Matrix Market
 * file to a stream.
 *
 * If it is not ‘NULL’, then the number of bytes written to the stream
 * is returned in ‘bytes_written’.
 */
int mtxfileheader_fwrite(
    const struct mtxfileheader * header,
    FILE * f,
    int64_t * bytes_written)
{
    int ret = fprintf(
        f, "%%%%MatrixMarket %s %s %s %s\n",
        mtxfileobjectstr(header->object),
        mtxfileformatstr(header->format),
        mtxfilefieldstr(header->field),
        mtxfilesymmetrystr(header->symmetry));
    if (ret < 0)
        return MTX_ERR_ERRNO;
    if (bytes_written)
        *bytes_written += ret;
    return MTX_SUCCESS;
}

#ifdef LIBMTX_HAVE_LIBZ
/**
 * ‘mtxfileheader_gzwrite()’ writes the header line of a Matrix
 * Market file to a gzip-compressed stream.
 *
 * If it is not ‘NULL’, then the number of bytes written to the stream
 * is returned in ‘bytes_written’.
 */
int mtxfileheader_gzwrite(
    const struct mtxfileheader * header,
    gzFile f,
    int64_t * bytes_written)
{
    int ret = gzprintf(
        f, "%%%%MatrixMarket %s %s %s %s\n",
        mtxfileobjectstr(header->object),
        mtxfileformatstr(header->format),
        mtxfilefieldstr(header->field),
        mtxfilesymmetrystr(header->symmetry));
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
 * ‘mtxfileheader_datatype()’ creates a custom MPI data type for
 * sending or receiving Matrix Market headers.
 *
 * The user is responsible for calling ‘MPI_Type_free()’ on the
 * returned datatype.
 */
static int mtxfileheader_datatype(
    MPI_Datatype * datatype,
    int * mpierrcode)
{
    int num_elements = 1;
    int block_lengths[] = {4};
    MPI_Datatype element_types[] = {MPI_INT};
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
 * ‘mtxfileheader_send()’ sends a Matrix Market header to another MPI
 * process.
 *
 * This is analogous to ‘MPI_Send()’ and requires the receiving
 * process to perform a matching call to ‘mtxfileheader_recv()’.
 */
int mtxfileheader_send(
    const struct mtxfileheader * header,
    int dest,
    int tag,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    disterr->err = MPI_Send(
        &header->object, 1, MPI_INT, dest, tag, comm);
    if (disterr->err)
        return MTX_ERR_MPI;
    disterr->err = MPI_Send(
        &header->format, 1, MPI_INT, dest, tag, comm);
    if (disterr->err)
        return MTX_ERR_MPI;
    disterr->err = MPI_Send(
        &header->field, 1, MPI_INT, dest, tag, comm);
    if (disterr->err)
        return MTX_ERR_MPI;
    disterr->err = MPI_Send(
        &header->symmetry, 1, MPI_INT, dest, tag, comm);
    if (disterr->err)
        return MTX_ERR_MPI;
    return MTX_SUCCESS;
}

/**
 * ‘mtxfileheader_recv()’ receives a Matrix Market header from
 * another MPI process.
 *
 * This is analogous to ‘MPI_Recv()’ and requires the sending process
 * to perform a matching call to ‘mtxfileheader_send()’.
 */
int mtxfileheader_recv(
    struct mtxfileheader * header,
    int source,
    int tag,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    disterr->err = MPI_Recv(
        &header->object, 1, MPI_INT, source, tag, comm, MPI_STATUS_IGNORE);
    if (disterr->err)
        return MTX_ERR_MPI;
    disterr->err = MPI_Recv(
        &header->format, 1, MPI_INT, source, tag, comm, MPI_STATUS_IGNORE);
    if (disterr->err)
        return MTX_ERR_MPI;
    disterr->err = MPI_Recv(
        &header->field, 1, MPI_INT, source, tag, comm, MPI_STATUS_IGNORE);
    if (disterr->err)
        return MTX_ERR_MPI;
    disterr->err = MPI_Recv(
        &header->symmetry, 1, MPI_INT, source, tag, comm, MPI_STATUS_IGNORE);
    if (disterr->err)
        return MTX_ERR_MPI;
    return MTX_SUCCESS;
}

/**
 * ‘mtxfileheader_bcast()’ broadcasts a Matrix Market header from an
 * MPI root process to other processes in a communicator.
 *
 * This is analogous to ‘MPI_Bcast()’ and requires every process in
 * the communicator to perform matching calls to
 * ‘mtxfileheader_bcast()’.
 */
int mtxfileheader_bcast(
    struct mtxfileheader * header,
    int root,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err;
    err = MPI_Bcast(
        &header->object, 1, MPI_INT, root, comm);
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;
    err = MPI_Bcast(
        &header->format, 1, MPI_INT, root, comm);
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;
    err = MPI_Bcast(
        &header->field, 1, MPI_INT, root, comm);
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;
    err = MPI_Bcast(
        &header->symmetry, 1, MPI_INT, root, comm);
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxfileheader_gather()’ gathers Matrix Market headers onto an MPI
 * root process from other processes in a communicator.
 *
 * This is analogous to ‘MPI_Gather()’ and requires every process in
 * the communicator to perform matching calls to
 * ‘mtxfileheader_gather()’.
 */
int mtxfileheader_gather(
    const struct mtxfileheader * sendheader,
    struct mtxfileheader * recvheaders,
    int root,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err;
    MPI_Datatype headertype;
    err = mtxfileheader_datatype(&headertype, &disterr->mpierrcode);
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;
    disterr->mpierrcode = MPI_Gather(
        sendheader, 1, headertype, recvheaders, 1, headertype, root, comm);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;
    MPI_Type_free(&headertype);
    return MTX_SUCCESS;
}

/**
 * ‘mtxfileheader_allgather()’ gathers Matrix Market headers onto
 * every MPI process from other processes in a communicator.
 *
 * This is analogous to ‘MPI_Allgather()’ and requires every process
 * in the communicator to perform matching calls to this function.
 */
int mtxfileheader_allgather(
    const struct mtxfileheader * sendheader,
    struct mtxfileheader * recvheaders,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err;
    MPI_Datatype headertype;
    err = mtxfileheader_datatype(&headertype, &disterr->mpierrcode);
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;
    disterr->mpierrcode = MPI_Allgather(
        sendheader, 1, headertype, recvheaders, 1, headertype, comm);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;
    MPI_Type_free(&headertype);
    return MTX_SUCCESS;
}
#endif
