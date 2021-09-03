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

#ifdef LIBMTX_HAVE_LIBZ
#include <zlib.h>
#endif

#include <errno.h>
#include <unistd.h>

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
    int * bytes_read,
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
    int * bytes_read,
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
    int * bytes_read,
    const char ** endptr,
    const char * s)
{
    int err;
    const char * t = s;
    err = parse_int64(t, "\n", &size->num_nonzeros, &t);
    if (err == EINVAL) {
        return MTX_ERR_INVALID_MTX_SIZE;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    }
    size->num_rows = -1;
    size->num_columns = -1;
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
    int * bytes_read,
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
    int * bytes_read,
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
 * `mtxfile_size_num_data_lines()' computes the number of data lines
 * that are required in a Matrix Market file with the given size line.
 */
int mtxfile_size_num_data_lines(
    const struct mtxfile_size * size,
    size_t * num_data_lines)
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
    int * bytes_read,
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
    int * bytes_read,
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
