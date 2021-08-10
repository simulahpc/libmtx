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
 * Input/output for Matrix Market objects.
 */

#include <libmtx/libmtx-config.h>

#include <libmtx/error.h>
#include <libmtx/matrix/array/array.h>
#include <libmtx/matrix/array/io.h>
#include <libmtx/matrix/coordinate/coordinate.h>
#include <libmtx/matrix/coordinate/io.h>
#include <libmtx/matrix/matrix.h>
#include <libmtx/mtx/io.h>
#include <libmtx/mtx/mtx.h>
#include <libmtx/vector/array/array.h>
#include <libmtx/vector/array/io.h>
#include <libmtx/vector/coordinate/coordinate.h>
#include <libmtx/vector/coordinate/io.h>

#include "../util/format.h"
#include "../util/io.h"
#include "../util/parse.h"

#ifdef LIBMTX_HAVE_LIBZ
#include <zlib.h>
#endif

#include <errno.h>
#include <unistd.h>

#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * `mtx_read()' reads a `struct mtx' object from a file in Matrix
 * Market format. The file may optionally be compressed by gzip.
 *
 * If `path' is `-', then standard input is used.
 *
 * If `format' is `NULL', then the format specifier '%d' is used to
 * print integers and '%f' is used to print floating point
 * numbers. Otherwise, the given format string is used when printing
 * numerical values.
 *
 * The format string follows the conventions of `printf'. If the field
 * is `real', `double' or `complex', then the format specifiers '%e',
 * '%E', '%f', '%F', '%g' or '%G' may be used. If the field is
 * `integer', then the format specifier must be '%d'. The format
 * string is ignored if the field is `pattern'. Field width and
 * precision may be specified (e.g., "%3.1f"), but variable field
 * width and precision (e.g., "%*.*f"), as well as length modifiers
 * (e.g., "%Lf") are not allowed.
 */
int mtx_read(
    struct mtx * mtx,
    const char * path,
    bool gzip,
    int * line_number,
    int * column_number)
{
    int err;
    *line_number = -1;
    *column_number = -1;

    if (!gzip) {
        FILE * f;
        if (strcmp(path, "-") == 0) {
            f = stdin;
        } else if ((f = fopen(path, "r")) == NULL) {
            return MTX_ERR_ERRNO;
        }

        err = mtx_fread(mtx, f, line_number, column_number);
        if (err)
            return err;
        fclose(f);
    } else {
#ifdef LIBMTX_HAVE_LIBZ
        gzFile f;
        if (strcmp(path, "-") == 0) {
            f = gzdopen(STDIN_FILENO, "r");
        } else if ((f = gzopen(path, "r")) == NULL) {
            return MTX_ERR_ERRNO;
        }

        err = mtx_gzread(mtx, f, line_number, column_number);
        if (err)
            return err;
        gzclose(f);
#else
        errno = ENOTSUP;
        return MTX_ERR_ERRNO;
#endif
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_write()' writes a `struct mtx' object from a file in Matrix
 * Market format. The output may optionally be compressed by gzip.
 *
 * If `path' is `-', then standard output is used.
 *
 * If `format' is `NULL', then the format specifier '%d' is used to
 * print integers and '%f' is used to print floating point
 * numbers. Otherwise, the given format string is used when printing
 * numerical values.
 *
 * The format string follows the conventions of `printf'. If the field
 * is `real', `double' or `complex', then the format specifiers '%e',
 * '%E', '%f', '%F', '%g' or '%G' may be used. If the field is
 * `integer', then the format specifier must be '%d'. The format
 * string is ignored if the field is `pattern'. Field width and
 * precision may be specified (e.g., "%3.1f"), but variable field
 * width and precision (e.g., "%*.*f"), as well as length modifiers
 * (e.g., "%Lf") are not allowed.
 */
int mtx_write(
    const struct mtx * mtx,
    const char * path,
    bool gzip,
    const char * format)
{
    int err;
    if (!gzip) {
        FILE * f;
        if (strcmp(path, "-") == 0) {
            f = stdout;
        } else if ((f = fopen(path, "w")) == NULL) {
            return MTX_ERR_ERRNO;
        }

        err = mtx_fwrite(mtx, f, format);
        if (err)
            return err;
        fclose(f);
    } else {
#ifdef LIBMTX_HAVE_LIBZ
        gzFile f;
        if (strcmp(path, "-") == 0) {
            f = gzdopen(STDOUT_FILENO, "w");
        } else if ((f = gzopen(path, "w")) == NULL) {
            return MTX_ERR_ERRNO;
        }

        err = mtx_gzwrite(mtx, f, format);
        if (err)
            return err;
        gzclose(f);
#else
        errno = ENOTSUP;
        return MTX_ERR_ERRNO;
#endif
    }
    return MTX_SUCCESS;
}

/**
 * `read_header_line()` reads a header line from a stream in
 * the Matrix Market file format.
 */
static int read_header_line(
    enum mtx_object * object,
    enum mtx_format * format,
    enum mtx_field * field,
    enum mtx_symmetry * symmetry,
    const struct stream * stream,
    size_t line_max,
    char * linebuf,
    int * line_number,
    int * column_number)
{
    int err;
    err = stream_read_line(stream, line_max, linebuf);
    if (err)
        return err;

    int bytes_read;
    err = mtx_header_parse(
        linebuf, &bytes_read, NULL,
        object, format, field, symmetry);
    if (err) {
        *column_number = bytes_read+1;
        return err;
    }
    (*line_number)++; *column_number = 1;
    return MTX_SUCCESS;
}

/**
 * `comment_line_list` is a linked list data structure used for
 * parsing comment lines.
 */
struct comment_line_list
{
    char * comment_line;
    struct comment_line_list * prev;
    struct comment_line_list * next;
};

/**
 * `read_comment_lines()` reads comment lines from a stream in the
 * Matrix Market file format.
 */
static int read_comment_lines(
    int * num_comment_lines,
    char *** comment_lines,
    const struct stream * stream,
    size_t line_max,
    char * linebuf,
    int * line_number,
    int * column_number)
{
    int err;

    /* 1. Read comment lines into a list. */
    struct comment_line_list * root = NULL;
    struct comment_line_list * node = NULL;
    *num_comment_lines = 0;
    while (true) {
        int c = stream_getc(stream);
        if (c == MTX_ERR_INVALID_STREAM_TYPE)
            return MTX_ERR_INVALID_STREAM_TYPE;
        c = stream_ungetc(c, stream);
        if (c == MTX_ERR_INVALID_STREAM_TYPE)
            return MTX_ERR_INVALID_STREAM_TYPE;

        /* Stop parsing comments on end-of-file or if the line does
         * not start with '%'. */
        if (c == EOF || c != '%')
            break;

        /* Allocate a list node. */
        struct comment_line_list * next =
            malloc(sizeof(struct comment_line_list));
        if (!next) {
            while (node) {
                struct comment_line_list * prev = node->prev;
                free(node->comment_line);
                free(node);
                node = prev;
            }
            return MTX_ERR_ERRNO;
        }
        next->comment_line = NULL;
        next->prev = next->next = NULL;

        /* Read the next line as a comment line. */
        err = stream_read_line(stream, line_max, linebuf);
        if (err) {
            free(next);
            while (node) {
                struct comment_line_list * prev = node->prev;
                free(node->comment_line);
                free(node);
                node = prev;
            }
            return err;
        }

        /* Add the new node to the list. */
        next->comment_line = strdup(linebuf);
        if (!node) {
            root = node = next;
        } else {
            next->prev = node;
            node->next = next;
            node = node->next;
        }

        (*line_number)++; *column_number = 1;
        (*num_comment_lines)++;
    }

    /* 2. Allocate storage for comment lines. */
    *comment_lines = malloc(*num_comment_lines * sizeof(char *));
    if (!*comment_lines) {
        while (node) {
            struct comment_line_list * prev = node->prev;
            free(node->comment_line);
            free(node);
            node = prev;
        }
        return MTX_ERR_ERRNO;
    }

    /* 3. Initialise the array of comment lines. */
    for (int i = 0; i < *num_comment_lines; i++) {
        (*comment_lines)[i] = root->comment_line;
        root = root->next;
    }

    /* 4. Clean up the list. */
    while (node) {
        struct comment_line_list * prev = node->prev;
        free(node);
        node = prev;
    }
    return MTX_SUCCESS;
}

/**
 * `read_size_line()` reads a size line from a stream in the Matrix
 * Market file format.
 */
static int read_size_line(
    enum mtx_object object,
    enum mtx_format format,
    enum mtx_field field,
    enum mtx_symmetry symmetry,
    int * num_rows,
    int * num_columns,
    int64_t * num_nonzeros,
    int64_t * size,
    int * nonzero_size,
    const struct stream * stream,
    size_t line_max,
    char * linebuf,
    int * line_number,
    int * column_number)
{
    int err;

    /* Read the size line. */
    err = stream_read_line(stream, line_max, linebuf);
    if (err)
        return err;

    const char * s = linebuf;
    if (object == mtx_matrix) {
        if (format == mtx_array) {
            int bytes_read;
            err = mtx_matrix_array_parse_size(
                s, &bytes_read, &s,
                object, format, field, symmetry,
                num_rows, num_columns, num_nonzeros,
                size, nonzero_size);
            if (err) {
                *column_number += bytes_read;
                return err;
            }
            (*line_number)++; *column_number = 1;

        } else if (format == mtx_coordinate) {
            int bytes_read;
            err = mtx_matrix_coordinate_parse_size(
                s, &bytes_read, &s,
                object, format, field, symmetry,
                num_rows, num_columns, num_nonzeros,
                size, nonzero_size);
            if (err) {
                *column_number += bytes_read;
                return err;
            }
            (*line_number)++; *column_number = 1;

        } else {
            return MTX_ERR_INVALID_MTX_FORMAT;
        }

    } else if (object == mtx_vector) {
        if (format == mtx_array) {
            int bytes_read;
            err = mtx_vector_array_parse_size(
                s, &bytes_read, &s,
                object, format, field, symmetry,
                num_rows, num_columns, num_nonzeros,
                size, nonzero_size);
            if (err) {
                *column_number += bytes_read;
                return err;
            }
            (*line_number)++; *column_number = 1;

        } else if (format == mtx_coordinate) {
            int bytes_read;
            err = mtx_vector_coordinate_parse_size(
                s, &bytes_read, &s,
                object, format, field, symmetry,
                num_rows, num_columns, num_nonzeros,
                size, nonzero_size);
            if (err) {
                *column_number += bytes_read;
                return err;
            }
            (*line_number)++; *column_number = 1;

        } else {
            return MTX_ERR_INVALID_MTX_FORMAT;
        }
    } else {
        return MTX_ERR_INVALID_MTX_OBJECT;
    }

    return MTX_SUCCESS;
}

/**
 * `parse_array_real()` parses a single nonzero for a matrix whose
 * format is `array` and field is `real`.
 */
static int parse_array_real(
    const char * s, float * a)
{
    int err = parse_float(s, "\n", a, NULL);
    if (err == EINVAL) {
        return MTX_ERR_INVALID_MTX_DATA;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    }
    return MTX_SUCCESS;
}

/**
 * `parse_array_double()` parses a single nonzero for a matrix whose
 * format is `array` and field is `double`.
 */
static int parse_array_double(
    const char * s, double * a)
{
    int err = parse_double(s, "\n", a, NULL);
    if (err == EINVAL) {
        return MTX_ERR_INVALID_MTX_DATA;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    }
    return MTX_SUCCESS;
}

/**
 * `parse_array_complex()` parses a single nonzero for a matrix whose
 * format is `array` and field is `complex`.
 */
static int parse_array_complex(
    const char * s, float * a, float * b)
{
    int err = parse_float(s, " ", a, &s);
    if (err == EINVAL) {
        return MTX_ERR_INVALID_MTX_DATA;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    }
    err = parse_float(s, "\n", b, NULL);
    if (err == EINVAL) {
        return MTX_ERR_INVALID_MTX_DATA;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    }
    return MTX_SUCCESS;
}

/**
 * `parse_array_integer()` parses a single nonzero for a matrix whose
 * format is `array` and field is `integer`.
 */
static int parse_array_integer(
    const char * s, int * a)
{
    int err = parse_int32(s, "\n", a, NULL);
    if (err == EINVAL) {
        return MTX_ERR_INVALID_MTX_DATA;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    }
    return MTX_SUCCESS;
}

/**
 * `read_data_array()` reads lines of dense (array) matrix data from a
 * stream in the Matrix Market file format.
 */
static int read_data_array(
    enum mtx_field field,
    int64_t size,
    void ** out_data,
    const struct stream * stream,
    size_t line_max,
    char * linebuf,
    int * line_number,
    int * column_number)
{
    int err;

    if (field == mtx_real) {
        /* 1. Allocate storage for matrix data. */
        float * data = (float *) malloc(size * sizeof(float));
        if (!data)
            return MTX_ERR_ERRNO;

        /* 2. Read each line of data. */
        for (int64_t k = 0; k < size; k++) {
            err = stream_read_line(stream, line_max, linebuf);
            if (err) {
                free(data);
                return err;
            }
            err = parse_array_real(linebuf, &data[k]);
            if (err) {
                free(data);
                return err;
            }
            (*line_number)++; *column_number = 1;
        }
        *out_data = (void *) data;

    } else if (field == mtx_double) {
        /* 1. Allocate storage for matrix data. */
        double * data = (double *) malloc(size * sizeof(double));
        if (!data)
            return MTX_ERR_ERRNO;

        /* 2. Read each line of data. */
        for (int64_t k = 0; k < size; k++) {
            err = stream_read_line(stream, line_max, linebuf);
            if (err) {
                free(data);
                return err;
            }
            err = parse_array_double(linebuf, &data[k]);
            if (err) {
                free(data);
                return err;
            }
            (*line_number)++; *column_number = 1;
        }
        *out_data = (void *) data;

    } else if (field == mtx_complex) {
        /* 1. Allocate storage for matrix data. */
        float * data = (float *) malloc(size * 2 * sizeof(float));
        if (!data)
            return MTX_ERR_ERRNO;

        /* 2. Read each line of data. */
        for (int64_t k = 0; k < size; k++) {
            err = stream_read_line(stream, line_max, linebuf);
            if (err) {
                free(data);
                return err;
            }
            err = parse_array_complex(
                linebuf, &data[2*k+0], &data[2*k+1]);
            if (err) {
                free(data);
                return err;
            }
            (*line_number)++; *column_number = 1;
        }
        *out_data = (void *) data;

    } else if (field == mtx_integer) {
        /* 1. Allocate storage for matrix data. */
        int * data = (int *) malloc(size * sizeof(int));
        if (!data)
            return MTX_ERR_ERRNO;

        /* 2. Read each line of data. */
        for (int64_t k = 0; k < size; k++) {
            err = stream_read_line(stream, line_max, linebuf);
            if (err) {
                free(data);
                return err;
            }
            err = parse_array_integer(linebuf, &data[k]);
            if (err) {
                free(data);
                return err;
            }
            (*line_number)++; *column_number = 1;
        }
        *out_data = (void *) data;

    } else {
        return MTX_ERR_INVALID_MTX_FIELD;
    }

    return MTX_SUCCESS;
}

/**
 * `parse_matrix_coordinate_real()` parses a single nonzero for a matrix
 * whose format is `coordinate` and field is `real`.
 */
static int parse_matrix_coordinate_real(
    const char * s,
    struct mtx_matrix_coordinate_real * data,
    int num_rows,
    int num_columns,
    int * line_number, int * column_number)
{
    const char * start = s;
    int err = parse_int32(s, " ", &data->i, &s);
    if (err == EINVAL || (!err && (data->i < 1 || data->i > num_rows))) {
        return MTX_ERR_INVALID_MTX_DATA;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    }
    *column_number += s-start; start = s;
    err = parse_int32(s, " ", &data->j, &s);
    if (err == EINVAL || (!err && (data->j < 1 || data->j > num_columns))) {
        return MTX_ERR_INVALID_MTX_DATA;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    }
    *column_number += s-start; start = s;
    err = parse_float(s, "\n", &data->a, NULL);
    if (err == EINVAL) {
        return MTX_ERR_INVALID_MTX_DATA;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    }
    (*line_number)++; *column_number = 1;
    return MTX_SUCCESS;
}

/**
 * `parse_matrix_coordinate_double()` parses a single nonzero for a matrix
 * whose format is `coordinate` and field is `double`.
 */
static int parse_matrix_coordinate_double(
    const char * s,
    struct mtx_matrix_coordinate_double * data,
    int num_rows,
    int num_columns,
    int * line_number, int * column_number)
{
    const char * start = s;
    int err = parse_int32(s, " ", &data->i, &s);
    if (err == EINVAL || (!err && (data->i < 1 || data->i > num_rows))) {
        return MTX_ERR_INVALID_MTX_DATA;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    }
    *column_number += s-start; start = s;
    err = parse_int32(s, " ", &data->j, &s);
    if (err == EINVAL || (!err && (data->j < 1 || data->j > num_columns))) {
        return MTX_ERR_INVALID_MTX_DATA;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    }
    *column_number += s-start; start = s;
    err = parse_double(s, "\n", &data->a, NULL);
    if (err == EINVAL) {
        return MTX_ERR_INVALID_MTX_DATA;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    }
    (*line_number)++; *column_number = 1;
    return MTX_SUCCESS;
}

/**
 * `parse_matrix_coordinate_complex()` parses a single nonzero for a matrix
 * whose format is `coordinate` and field is `complex`.
 */
static int parse_matrix_coordinate_complex(
    const char * s,
    struct mtx_matrix_coordinate_complex * data,
    int num_rows,
    int num_columns,
    int * line_number, int * column_number)
{
    const char * start = s;
    int err = parse_int32(s, " ", &data->i, &s);
    if (err == EINVAL || (!err && (data->i < 1 || data->i > num_rows))) {
        return MTX_ERR_INVALID_MTX_DATA;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    }
    *column_number += s-start; start = s;
    err = parse_int32(s, " ", &data->j, &s);
    if (err == EINVAL || (!err && (data->j < 1 || data->j > num_columns))) {
        return MTX_ERR_INVALID_MTX_DATA;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    }
    *column_number += s-start; start = s;
    err = parse_float(s, " ", &data->a, &s);
    if (err == EINVAL) {
        return MTX_ERR_INVALID_MTX_DATA;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    }
    *column_number += s-start; start = s;
    err = parse_float(s, "\n", &data->b, NULL);
    if (err == EINVAL) {
        return MTX_ERR_INVALID_MTX_DATA;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    }
    (*line_number)++; *column_number = 1;
    return MTX_SUCCESS;
}

/**
 * `parse_matrix_coordinate_integer()` parses a single nonzero for a matrix
 * whose format is `coordinate` and field is `integer`.
 */
static int parse_matrix_coordinate_integer(
    const char * s,
    struct mtx_matrix_coordinate_integer * data,
    int num_rows,
    int num_columns,
    int * line_number, int * column_number)
{
    const char * start = s;
    int err = parse_int32(s, " ", &data->i, &s);
    if (err == EINVAL || (!err && (data->i < 1 || data->i > num_rows))) {
        return MTX_ERR_INVALID_MTX_DATA;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    }
    *column_number += s-start; start = s;
    err = parse_int32(s, " ", &data->j, &s);
    if (err == EINVAL || (!err && (data->j < 1 || data->j > num_columns))) {
        return MTX_ERR_INVALID_MTX_DATA;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    }
    *column_number += s-start; start = s;
    err = parse_int32(s, "\n", &data->a, NULL);
    if (err == EINVAL) {
        return MTX_ERR_INVALID_MTX_DATA;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    }
    (*line_number)++; *column_number = 1;
    return MTX_SUCCESS;
}

/**
 * `parse_matrix_coordinate_pattern()` parses a single nonzero for a matrix
 * whose format is `coordinate` and field is `pattern`.
 */
static int parse_matrix_coordinate_pattern(
    const char * s,
    struct mtx_matrix_coordinate_pattern * data,
    int num_rows,
    int num_columns,
    int * line_number, int * column_number)
{
    const char * start = s;
    int err = parse_int32(s, " ", &data->i, &s);
    if (err == EINVAL || (!err && (data->i < 1 || data->i > num_rows))) {
        return MTX_ERR_INVALID_MTX_DATA;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    }
    *column_number += s-start; start = s;
    err = parse_int32(s, "\n", &data->j, NULL);
    if (err == EINVAL || (!err && (data->j < 1 || data->j > num_columns))) {
        return MTX_ERR_INVALID_MTX_DATA;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    }
    (*line_number)++; *column_number = 1;
    return MTX_SUCCESS;
}

/**
 * `read_data_matrix_coordinate()` reads lines of sparse (coordinate)
 * matrix data from a stream in the Matrix Market file format.
 */
static int read_data_matrix_coordinate(
    enum mtx_field field,
    int num_rows,
    int num_columns,
    int64_t size,
    void ** out_data,
    const struct stream * stream,
    size_t line_max,
    char * linebuf,
    int * line_number,
    int * column_number)
{
    int err;

    if (field == mtx_real) {
        /* 1. Allocate storage for matrix data. */
        struct mtx_matrix_coordinate_real * data =
            (struct mtx_matrix_coordinate_real *) malloc(
                size * sizeof(struct mtx_matrix_coordinate_real));
        if (!data)
            return MTX_ERR_ERRNO;

        /* 2. Read each line of data. */
        for (int64_t k = 0; k < size; k++) {
            err = stream_read_line(stream, line_max, linebuf);
            if (err) {
                free(data);
                return err;
            }
            err = parse_matrix_coordinate_real(
                linebuf, &data[k], num_rows, num_columns,
                line_number, column_number);
            if (err) {
                free(data);
                return err;
            }
        }
        *out_data = (void *) data;

    } else if (field == mtx_double) {
        /* 1. Allocate storage for matrix data. */
        struct mtx_matrix_coordinate_double * data =
            (struct mtx_matrix_coordinate_double *) malloc(
                size * sizeof(struct mtx_matrix_coordinate_double));
        if (!data)
            return MTX_ERR_ERRNO;

        /* 2. Read each line of data. */
        for (int64_t k = 0; k < size; k++) {
            err = stream_read_line(stream, line_max, linebuf);
            if (err) {
                free(data);
                return err;
            }
            err = parse_matrix_coordinate_double(
                linebuf, &data[k], num_rows, num_columns, line_number, column_number);
            if (err) {
                free(data);
                return err;
            }
        }
        *out_data = (void *) data;

    } else if (field == mtx_complex) {
        /* 1. Allocate storage for matrix data. */
        struct mtx_matrix_coordinate_complex * data =
            (struct mtx_matrix_coordinate_complex *) malloc(
                size * sizeof(struct mtx_matrix_coordinate_complex));
        if (!data)
            return MTX_ERR_ERRNO;

        /* 2. Read each line of data. */
        for (int64_t k = 0; k < size; k++) {
            err = stream_read_line(stream, line_max, linebuf);
            if (err) {
                free(data);
                return err;
            }
            err = parse_matrix_coordinate_complex(
                linebuf, &data[k], num_rows, num_columns,
                line_number, column_number);
            if (err) {
                free(data);
                return err;
            }
        }
        *out_data = (void *) data;

    } else if (field == mtx_integer) {
        /* 1. Allocate storage for matrix data. */
        struct mtx_matrix_coordinate_integer * data =
            (struct mtx_matrix_coordinate_integer *) malloc(
                size * sizeof(struct mtx_matrix_coordinate_integer));
        if (!data)
            return MTX_ERR_ERRNO;

        /* 2. Read each line of data. */
        for (int64_t k = 0; k < size; k++) {
            err = stream_read_line(stream, line_max, linebuf);
            if (err) {
                free(data);
                return err;
            }
            err = parse_matrix_coordinate_integer(
                linebuf, &data[k], num_rows, num_columns,
                line_number, column_number);
            if (err) {
                free(data);
                return err;
            }
        }
        *out_data = (void *) data;

    } else if (field == mtx_pattern) {
        /* 1. Allocate storage for matrix data. */
        struct mtx_matrix_coordinate_pattern * data =
            (struct mtx_matrix_coordinate_pattern *) malloc(
                size * sizeof(struct mtx_matrix_coordinate_pattern));
        if (!data)
            return MTX_ERR_ERRNO;

        /* 2. Read each line of data. */
        for (int64_t k = 0; k < size; k++) {
            err = stream_read_line(stream, line_max, linebuf);
            if (err) {
                free(data);
                return err;
            }
            err = parse_matrix_coordinate_pattern(
                linebuf, &data[k], num_rows, num_columns,
                line_number, column_number);
            if (err) {
                free(data);
                return err;
            }
        }
        *out_data = (void *) data;

    } else {
        return MTX_ERR_INVALID_MTX_FIELD;
    }
    return MTX_SUCCESS;
}

/**
 * `parse_vector_coordinate_real()` parses a single nonzero for a vector
 * whose format is `coordinate` and field is `real`.
 */
static int parse_vector_coordinate_real(
    const char * s,
    struct mtx_vector_coordinate_real * data,
    int num_rows,
    int * line_number, int * column_number)
{
    const char * start = s;
    int err = parse_int32(s, " ", &data->i, &s);
    if (err == EINVAL || (!err && (data->i < 1 || data->i > num_rows))) {
        return MTX_ERR_INVALID_MTX_DATA;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    }
    *column_number += s-start; start = s;
    err = parse_float(s, "\n", &data->a, NULL);
    if (err == EINVAL) {
        return MTX_ERR_INVALID_MTX_DATA;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    }
    (*line_number)++; *column_number = 1;
    return MTX_SUCCESS;
}

/**
 * `parse_vector_coordinate_double()` parses a single nonzero for a vector
 * whose format is `coordinate` and field is `double`.
 */
static int parse_vector_coordinate_double(
    const char * s,
    struct mtx_vector_coordinate_double * data,
    int num_rows,
    int * line_number, int * column_number)
{
    const char * start = s;
    int err = parse_int32(s, " ", &data->i, &s);
    if (err == EINVAL || (!err && (data->i < 1 || data->i > num_rows))) {
        return MTX_ERR_INVALID_MTX_DATA;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    }
    *column_number += s-start; start = s;
    err = parse_double(s, "\n", &data->a, NULL);
    if (err == EINVAL) {
        return MTX_ERR_INVALID_MTX_DATA;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    }
    (*line_number)++; *column_number = 1;
    return MTX_SUCCESS;
}

/**
 * `parse_vector_coordinate_complex()` parses a single nonzero for a vector
 * whose format is `coordinate` and field is `complex`.
 */
static int parse_vector_coordinate_complex(
    const char * s,
    struct mtx_vector_coordinate_complex * data,
    int num_rows,
    int * line_number, int * column_number)
{
    const char * start = s;
    int err = parse_int32(s, " ", &data->i, &s);
    if (err == EINVAL || (!err && (data->i < 1 || data->i > num_rows))) {
        return MTX_ERR_INVALID_MTX_DATA;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    }
    *column_number += s-start; start = s;
    err = parse_float(s, " ", &data->a, &s);
    if (err == EINVAL) {
        return MTX_ERR_INVALID_MTX_DATA;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    }
    *column_number += s-start; start = s;
    err = parse_float(s, "\n", &data->b, NULL);
    if (err == EINVAL) {
        return MTX_ERR_INVALID_MTX_DATA;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    }
    (*line_number)++; *column_number = 1;
    return MTX_SUCCESS;
}

/**
 * `parse_vector_coordinate_integer()` parses a single nonzero for a vector
 * whose format is `coordinate` and field is `integer`.
 */
static int parse_vector_coordinate_integer(
    const char * s,
    struct mtx_vector_coordinate_integer * data,
    int num_rows,
    int * line_number, int * column_number)
{
    const char * start = s;
    int err = parse_int32(s, " ", &data->i, &s);
    if (err == EINVAL || (!err && (data->i < 1 || data->i > num_rows))) {
        return MTX_ERR_INVALID_MTX_DATA;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    }
    *column_number += s-start; start = s;
    err = parse_int32(s, "\n", &data->a, NULL);
    if (err == EINVAL) {
        return MTX_ERR_INVALID_MTX_DATA;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    }
    (*line_number)++; *column_number = 1;
    return MTX_SUCCESS;
}

/**
 * `parse_vector_coordinate_pattern()` parses a single nonzero for a vector
 * whose format is `coordinate` and field is `pattern`.
 */
static int parse_vector_coordinate_pattern(
    const char * s,
    struct mtx_vector_coordinate_pattern * data,
    int num_rows,
    int * line_number, int * column_number)
{
    int err = parse_int32(s, " ", &data->i, &s);
    if (err == EINVAL || (!err && (data->i < 1 || data->i > num_rows))) {
        return MTX_ERR_INVALID_MTX_DATA;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    }
    (*line_number)++; *column_number = 1;
    return MTX_SUCCESS;
}

/**
 * `read_data_vector_coordinate()` reads lines of sparse (coordinate)
 * vector data from a stream in the Matrix Market file format.
 */
static int read_data_vector_coordinate(
    enum mtx_field field,
    int num_rows,
    int64_t size,
    void ** out_data,
    const struct stream * stream,
    size_t line_max,
    char * linebuf,
    int * line_number,
    int * column_number)
{
    int err;
    if (field == mtx_real) {
        /* 1. Allocate storage for vector data. */
        struct mtx_vector_coordinate_real * data =
            (struct mtx_vector_coordinate_real *) malloc(
                size * sizeof(struct mtx_vector_coordinate_real));
        if (!data)
            return MTX_ERR_ERRNO;

        /* 2. Read each line of data. */
        for (int64_t k = 0; k < size; k++) {
            err = stream_read_line(stream, line_max, linebuf);
            if (err) {
                free(data);
                return err;
            }
            err = parse_vector_coordinate_real(
                linebuf, &data[k], num_rows,
                line_number, column_number);
            if (err) {
                free(data);
                return err;
            }
        }
        *out_data = (void *) data;

    } else if (field == mtx_double) {
        /* 1. Allocate storage for vector data. */
        struct mtx_vector_coordinate_double * data =
            (struct mtx_vector_coordinate_double *) malloc(
                size * sizeof(struct mtx_vector_coordinate_double));
        if (!data)
            return MTX_ERR_ERRNO;

        /* 2. Read each line of data. */
        for (int64_t k = 0; k < size; k++) {
            err = stream_read_line(stream, line_max, linebuf);
            if (err) {
                free(data);
                return err;
            }
            err = parse_vector_coordinate_double(
                linebuf, &data[k], num_rows, line_number, column_number);
            if (err) {
                free(data);
                return err;
            }
        }
        *out_data = (void *) data;

    } else if (field == mtx_complex) {
        /* 1. Allocate storage for vector data. */
        struct mtx_vector_coordinate_complex * data =
            (struct mtx_vector_coordinate_complex *) malloc(
                size * sizeof(struct mtx_vector_coordinate_complex));
        if (!data)
            return MTX_ERR_ERRNO;

        /* 2. Read each line of data. */
        for (int64_t k = 0; k < size; k++) {
            err = stream_read_line(stream, line_max, linebuf);
            if (err) {
                free(data);
                return err;
            }
            err = parse_vector_coordinate_complex(
                linebuf, &data[k], num_rows,
                line_number, column_number);
            if (err) {
                free(data);
                return err;
            }
        }
        *out_data = (void *) data;

    } else if (field == mtx_integer) {
        /* 1. Allocate storage for vector data. */
        struct mtx_vector_coordinate_integer * data =
            (struct mtx_vector_coordinate_integer *) malloc(
                size * sizeof(struct mtx_vector_coordinate_integer));
        if (!data)
            return MTX_ERR_ERRNO;

        /* 2. Read each line of data. */
        for (int64_t k = 0; k < size; k++) {
            err = stream_read_line(stream, line_max, linebuf);
            if (err) {
                free(data);
                return err;
            }
            err = parse_vector_coordinate_integer(
                linebuf, &data[k], num_rows,
                line_number, column_number);
            if (err) {
                free(data);
                return err;
            }
        }
        *out_data = (void *) data;

    } else if (field == mtx_pattern) {
        /* 1. Allocate storage for vector data. */
        struct mtx_vector_coordinate_pattern * data =
            (struct mtx_vector_coordinate_pattern *) malloc(
                size * sizeof(struct mtx_vector_coordinate_pattern));
        if (!data)
            return MTX_ERR_ERRNO;

        /* 2. Read each line of data. */
        for (int64_t k = 0; k < size; k++) {
            err = stream_read_line(stream, line_max, linebuf);
            if (err) {
                free(data);
                return err;
            }
            err = parse_vector_coordinate_pattern(
                linebuf, &data[k], num_rows,
                line_number, column_number);
            if (err) {
                free(data);
                return err;
            }
        }
        *out_data = (void *) data;

    } else {
        return MTX_ERR_INVALID_MTX_FIELD;
    }
    return MTX_SUCCESS;
}

/**
 * `read_data_lines()` reads lines of matrix data from a stream in the
 * Matrix Market file format.
 */
static int read_data_lines(
    enum mtx_object object,
    enum mtx_format format,
    enum mtx_field field,
    int num_rows,
    int num_columns,
    int64_t size,
    void ** data,
    const struct stream * stream,
    size_t line_max,
    char * linebuf,
    int * line_number,
    int * column_number)
{
    int err;
    if (object == mtx_matrix) {
        if (format == mtx_array) {
            err = read_data_array(
                field, size, data, stream, line_max, linebuf,
                line_number, column_number);
            if (err)
                return err;
        } else if (format == mtx_coordinate) {
            err = read_data_matrix_coordinate(
                field, num_rows, num_columns, size, data,
                stream, line_max, linebuf, line_number, column_number);
            if (err)
                return err;
        } else {
            return MTX_ERR_INVALID_MTX_FORMAT;
        }
    } else if (object == mtx_vector) {
        if (format == mtx_array) {
            err = read_data_array(
                field, size, data, stream, line_max, linebuf,
                line_number, column_number);
            if (err)
                return err;
        } else if (format == mtx_coordinate) {
            err = read_data_vector_coordinate(
                field, num_rows, size, data,
                stream, line_max, linebuf, line_number, column_number);
            if (err)
                return err;
        } else {
            return MTX_ERR_INVALID_MTX_FORMAT;
        }
    } else {
        return MTX_ERR_INVALID_MTX_OBJECT;
    }
    return MTX_SUCCESS;
}

/**
 * `read_mtx()` reads a matrix or vector from a stream in Matrix
 * Market format using the given `getline' function to fetch each
 * line.
 */
static int read_mtx(
    struct mtx * matrix,
    const struct stream * stream,
    int * line_number,
    int * column_number)
{
    int err;

    /* Allocate storage for reading lines from file. */
    long int line_max = sysconf(_SC_LINE_MAX);
    char * linebuf = malloc(line_max+1);
    if (!linebuf)
        return MTX_ERR_ERRNO;

    matrix->comment_lines = NULL;
    matrix->data = NULL;

    /* 1. Parse the header line. */
    *line_number = 1;
    *column_number = 1;
    err = read_header_line(
        &matrix->object, &matrix->format,
        &matrix->field, &matrix->symmetry,
        stream, line_max, linebuf,
        line_number, column_number);
    if (err) {
        free(linebuf);
        return err;
    }

    /* Set extra header information. */
    matrix->triangle =
        matrix->object == mtx_matrix &&
        matrix->format == mtx_array &&
        (matrix->symmetry == mtx_symmetric ||
         matrix->symmetry == mtx_skew_symmetric ||
         matrix->symmetry == mtx_hermitian)
        ? mtx_lower_triangular : mtx_nontriangular;
    matrix->sorting = matrix->format == mtx_array ? mtx_row_major : mtx_unsorted;
    matrix->ordering = mtx_unordered;
    matrix->assembly = matrix->format == mtx_array ? mtx_assembled : mtx_unassembled;

    /* 2. Parse comment lines. */
    err = read_comment_lines(
        &matrix->num_comment_lines, &matrix->comment_lines,
        stream, line_max, linebuf, line_number, column_number);
    if (err) {
        free(linebuf);
        return err;
    }

    /* 3. Parse the size line. */
    err = read_size_line(
        matrix->object, matrix->format, matrix->field, matrix->symmetry,
        &matrix->num_rows, &matrix->num_columns,
        &matrix->num_nonzeros, &matrix->size, &matrix->nonzero_size,
        stream, line_max, linebuf, line_number, column_number);
    if (err) {
        for (int i = 0; i < matrix->num_comment_lines; i++)
            free(matrix->comment_lines[i]);
        free(matrix->comment_lines);
        matrix->comment_lines = NULL;
        free(linebuf);
        return err;
    }

    /* 4. Parse the data. */
    err = read_data_lines(
        matrix->object, matrix->format, matrix->field,
        matrix->num_rows, matrix->num_columns, matrix->size,
        &matrix->data, stream, line_max, linebuf,
        line_number, column_number);
    if (err) {
        for (int i = 0; i < matrix->num_comment_lines; i++)
            free(matrix->comment_lines[i]);
        free(matrix->comment_lines);
        matrix->comment_lines = NULL;
        free(linebuf);
        return err;
    }

    /*
     * 5. If the matrix is sparse, then we can now compute the total
     * number of matrix nonzeros.
     */
    if (matrix->object == mtx_matrix &&
        matrix->format == mtx_coordinate)
    {
        err = mtx_matrix_coordinate_num_nonzeros(
            matrix->field, matrix->symmetry,
            matrix->num_rows, matrix->num_columns,
            matrix->size, matrix->data,
            &matrix->num_nonzeros);
        if (err) {
            free(matrix->data);
            for (int i = 0; i < matrix->num_comment_lines; i++)
                free(matrix->comment_lines[i]);
            free(matrix->comment_lines);
            matrix->comment_lines = NULL;
            free(linebuf);
            return err;
        }
    }

    free(linebuf);
    return MTX_SUCCESS;
}

/**
 * `mtx_fread()` reads an object (matrix or vector) from a stream in
 * Matrix Market format.
 */
int mtx_fread(
    struct mtx * mtx,
    FILE * f,
    int * line_number,
    int * column_number)
{
    struct stream * stream = stream_init_stdio(f);
    if (!stream)
        return MTX_ERR_ERRNO;
    int err = read_mtx(mtx, stream, line_number, column_number);
    free(stream);
    return err;
}

#ifdef LIBMTX_HAVE_LIBZ
/**
 * `mtx_gzread()` reads a matrix or vector from a gzip-compressed
 * stream in Matrix Market format.
 */
int mtx_gzread(
    struct mtx * mtx,
    gzFile f,
    int * line_number,
    int * column_number)
{
    struct stream * stream = stream_init_gz(f);
    if (!stream)
        return MTX_ERR_ERRNO;
    int err = read_mtx(mtx, stream, line_number, column_number);
    free(stream);
    return err;
}
#endif

/**
 * `validate_format_string()' parses and validates a format string.
 */
static int validate_format_string(
    const char * format_str,
    enum mtx_field field)
{
    struct format_specifier format;
    const char * endptr;
    int err = parse_format_specifier(format_str, &format, &endptr);
    if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    } else if (*endptr != '\0') {
        return MTX_ERR_INVALID_FORMAT_SPECIFIER;
    }

    if (format.width == format_specifier_width_star ||
        format.precision == format_specifier_precision_star ||
        format.length != format_specifier_length_none ||
        ((field == mtx_real ||
          field == mtx_double ||
          field == mtx_complex) &&
         (format.specifier != format_specifier_e &&
          format.specifier != format_specifier_E &&
          format.specifier != format_specifier_f &&
          format.specifier != format_specifier_F &&
          format.specifier != format_specifier_g &&
          format.specifier != format_specifier_G)) ||
        (field == mtx_integer &&
         (format.specifier != format_specifier_d)))
    {
        return MTX_ERR_INVALID_FORMAT_SPECIFIER;
    }
    return MTX_SUCCESS;
}

/**
 * `write_matrix()` writes a matrix to a stream in Matrix Market
 * format.
 *
 * If `format' is `NULL', then the format specifier '%d' is used to
 * print integers and '%f' is used to print floating point
 * numbers. Otherwise, the given format string is used when printing
 * numerical values.
 *
 * The format string follows the conventions of `printf'. If the field
 * is `real', `double' or `complex', then the format specifiers '%e',
 * '%E', '%f', '%F', '%g' or '%G' may be used. If the field is
 * `integer', then the format specifier must be '%d'. The format
 * string is ignored if the field is `pattern'. Field width and
 * precision may be specified (e.g., "%3.1f"), but variable field
 * width and precision (e.g., "%*.*f"), as well as length modifiers
 * (e.g., "%Lf") are not allowed.
 */
static int write_matrix(
    const struct mtx * matrix,
    const struct stream * stream,
    const char * format)
{
    int err;
    if (matrix->object != mtx_matrix)
        return MTX_ERR_INVALID_MTX_OBJECT;

    /* Parse and validate the format string. */
    if (format) {
        err = validate_format_string(format, matrix->field);
        if (err)
            return err;
    }

    /* 1. Write the header line. */
    stream_printf(
        stream, "%%%%MatrixMarket %s %s %s %s\n",
        mtx_object_str(matrix->object),
        mtx_format_str(matrix->format),
        mtx_field_str(matrix->field),
        mtx_symmetry_str(matrix->symmetry));

    /* 2. Write comment lines. */
    for (int i = 0; i < matrix->num_comment_lines; i++)
        stream_printf(stream, "%s", matrix->comment_lines[i]);

    /* 3. Write the size line. */
    if (matrix->format == mtx_array) {
        stream_printf(stream, "%d %d\n", matrix->num_rows, matrix->num_columns);
    } else if (matrix->format == mtx_coordinate) {
        stream_printf(
            stream, "%d %d %"PRId64"\n",
            matrix->num_rows,
            matrix->num_columns,
            matrix->size);
    } else {
        return MTX_ERR_INVALID_MTX_FORMAT;
    }

    /* 4. Write the data. */
    if (matrix->format == mtx_array) {
        if (matrix->field == mtx_real) {
            const float * a = (const float *) matrix->data;
            for (int i = 0; i < matrix->num_rows; i++) {
                for (int j = 0; j < matrix->num_columns; j++) {
                    stream_printf(stream, format ? format : "%f",
                                  a[i*matrix->num_columns+j]);
                    stream_putc('\n', stream);
                }
            }
        } else if (matrix->field == mtx_double) {
            const double * a = (const double *) matrix->data;
            for (int i = 0; i < matrix->num_rows; i++) {
                for (int j = 0; j < matrix->num_columns; j++) {
                    stream_printf(stream, format ? format : "%f",
                                  a[i*matrix->num_columns+j]);
                    stream_putc('\n', stream);
                }
            }
        } else if (matrix->field == mtx_complex) {
            const float * a = (const float *) matrix->data;
            for (int i = 0; i < matrix->num_rows; i++) {
                for (int j = 0; j < matrix->num_columns; j++) {
                    stream_printf(stream, format ? format : "%f",
                                  a[2*(i*matrix->num_columns+j)+0]);
                    stream_putc(' ', stream);
                    stream_printf(stream, format ? format : "%f",
                                  a[2*(i*matrix->num_columns+j)+1]);
                    stream_putc('\n', stream);
                }
            }
        } else if (matrix->field == mtx_integer) {
            const int * a = (const int *) matrix->data;
            for (int i = 0; i < matrix->num_rows; i++) {
                for (int j = 0; j < matrix->num_columns; j++) {
                    stream_printf(stream, format ? format : "%d",
                                  a[i*matrix->num_columns+j]);
                    stream_putc('\n', stream);
                }
            }
        } else {
            return MTX_ERR_INVALID_MTX_FIELD;
        }

    } else if (matrix->format == mtx_coordinate) {
        if (matrix->field == mtx_real) {
            const struct mtx_matrix_coordinate_real * a =
                (const struct mtx_matrix_coordinate_real *) matrix->data;
            for (int64_t k = 0; k < matrix->size; k++) {
                stream_printf(stream, "%d %d ", a[k].i, a[k].j);
                stream_printf(stream, format ? format : "%f", a[k].a);
                stream_putc('\n', stream);
            }
        } else if (matrix->field == mtx_double) {
            const struct mtx_matrix_coordinate_double * a =
                (const struct mtx_matrix_coordinate_double *) matrix->data;
            for (int64_t k = 0; k < matrix->size; k++) {
                stream_printf(stream, "%d %d ", a[k].i, a[k].j);
                stream_printf(stream, format ? format : "%f", a[k].a);
                stream_putc('\n', stream);
            }
        } else if (matrix->field == mtx_complex) {
            const struct mtx_matrix_coordinate_complex * a =
                (const struct mtx_matrix_coordinate_complex *) matrix->data;
            for (int64_t k = 0; k < matrix->size; k++) {
                stream_printf(stream, "%d %d ", a[k].i, a[k].j);
                stream_printf(stream, format ? format : "%f", a[k].a);
                stream_putc(' ', stream);
                stream_printf(stream, format ? format : "%f", a[k].b);
                stream_putc('\n', stream);
            }
        } else if (matrix->field == mtx_integer) {
            const struct mtx_matrix_coordinate_integer * a =
                (const struct mtx_matrix_coordinate_integer *) matrix->data;
            for (int64_t k = 0; k < matrix->size; k++) {
                stream_printf(stream, "%d %d ", a[k].i, a[k].j);
                stream_printf(stream, format ? format : "%d", a[k].a);
                stream_putc('\n', stream);
            }
        } else if (matrix->field == mtx_pattern) {
            const struct mtx_matrix_coordinate_pattern * a =
                (const struct mtx_matrix_coordinate_pattern *) matrix->data;
            for (int64_t k = 0; k < matrix->size; k++) {
                stream_printf(stream, "%d %d\n", a[k].i, a[k].j);
            }
        } else {
            return MTX_ERR_INVALID_MTX_FIELD;
        }

    } else {
        return MTX_ERR_INVALID_MTX_FORMAT;
    }

    return MTX_SUCCESS;
}

/**
 * `write_vector()` writes a vector to a stream in the Matrix Market
 * format.
 *
 * If `format' is `NULL', then the format specifier '%d' is used to
 * print integers and '%f' is used to print floating point
 * numbers. Otherwise, the given format string is used when printing
 * numerical values.
 *
 * The format string follows the conventions of `printf'. If the field
 * is `real', `double' or `complex', then the format specifiers '%e',
 * '%E', '%f', '%F', '%g' or '%G' may be used. If the field is
 * `integer', then the format specifier must be '%d'. The format
 * string is ignored if the field is `pattern'. Field width and
 * precision may be specified (e.g., "%3.1f"), but variable field
 * width and precision (e.g., "%*.*f"), as well as length modifiers
 * (e.g., "%Lf") are not allowed.
 */
static int write_vector(
    const struct mtx * vector,
    const struct stream * stream,
    const char * format)
{
    int err;
    if (vector->object != mtx_vector)
        return MTX_ERR_INVALID_MTX_OBJECT;

    /* Parse and validate the format string. */
    if (format) {
        err = validate_format_string(format, vector->field);
        if (err)
            return err;
    }

    /* 1. Write the header line. */
    stream_printf(
        stream, "%%%%MatrixMarket %s %s %s %s\n",
        mtx_object_str(vector->object),
        mtx_format_str(vector->format),
        mtx_field_str(vector->field),
        mtx_symmetry_str(vector->symmetry));

    /* 2. Write comment lines. */
    for (int i = 0; i < vector->num_comment_lines; i++)
        stream_printf(stream, "%s", vector->comment_lines[i]);

    /* 3. Write the size line. */
    if (vector->format == mtx_array) {
        stream_printf(stream, "%"PRId64"\n", vector->size);
    } else if (vector->format == mtx_coordinate) {
        stream_printf(stream, "%d %"PRId64"\n", vector->num_rows, vector->size);
    } else {
        return MTX_ERR_INVALID_MTX_FORMAT;
    }

    /* 4. Write the data. */
    if (vector->format == mtx_array) {
        if (vector->field == mtx_real) {
            const float * a = (const float *) vector->data;
            for (int i = 0; i < vector->size; i++) {
                stream_printf(stream, format ? format : "%f", a[i]);
                stream_putc('\n', stream);
            }
        } else if (vector->field == mtx_double) {
            const double * a = (const double *) vector->data;
            for (int i = 0; i < vector->size; i++) {
                stream_printf(stream, format ? format : "%f", a[i]);
                stream_putc('\n', stream);
            }
        } else if (vector->field == mtx_complex) {
            const float * a = (const float *) vector->data;
            for (int i = 0; i < vector->size; i++) {
                stream_printf(stream, format ? format : "%f", a[2*i+0]);
                stream_putc(' ', stream);
                stream_printf(stream, format ? format : "%f", a[2*i+1]);
                stream_putc('\n', stream);
            }
        } else if (vector->field == mtx_integer) {
            const int * a = (const int *) vector->data;
            for (int i = 0; i < vector->size; i++) {
                stream_printf(stream, format ? format : "%d", a[i]);
                stream_putc('\n', stream);
            }
        } else {
            return MTX_ERR_INVALID_MTX_FIELD;
        }

    } else if (vector->format == mtx_coordinate) {
        if (vector->field == mtx_real) {
            const struct mtx_vector_coordinate_real * a =
                (const struct mtx_vector_coordinate_real *) vector->data;
            for (int64_t k = 0; k < vector->size; k++) {
                stream_printf(stream, "%d ", a[k].i);
                stream_printf(stream, format ? format : "%f", a[k].a);
                stream_putc('\n', stream);
            }
        } else if (vector->field == mtx_double) {
            const struct mtx_vector_coordinate_double * a =
                (const struct mtx_vector_coordinate_double *) vector->data;
            for (int64_t k = 0; k < vector->size; k++) {
                stream_printf(stream, "%d ", a[k].i);
                stream_printf(stream, format ? format : "%f", a[k].a);
                stream_putc('\n', stream);
            }
        } else if (vector->field == mtx_complex) {
            const struct mtx_vector_coordinate_complex * a =
                (const struct mtx_vector_coordinate_complex *) vector->data;
            for (int64_t k = 0; k < vector->size; k++) {
                stream_printf(stream, "%d ", a[k].i);
                stream_printf(stream, format ? format : "%f", a[k].a);
                stream_putc(' ', stream);
                stream_printf(stream, format ? format : "%f", a[k].b);
                stream_putc('\n', stream);
            }
        } else if (vector->field == mtx_integer) {
            const struct mtx_vector_coordinate_integer * a =
                (const struct mtx_vector_coordinate_integer *) vector->data;
            for (int64_t k = 0; k < vector->size; k++) {
                stream_printf(stream, "%d ", a[k].i);
                stream_printf(stream, format ? format : "%d", a[k].a);
                stream_putc('\n', stream);
            }
        } else if (vector->field == mtx_pattern) {
            const struct mtx_vector_coordinate_pattern * a =
                (const struct mtx_vector_coordinate_pattern *) vector->data;
            for (int64_t k = 0; k < vector->size; k++) {
                stream_printf(stream, "%d\n", a[k].i);
            }
        } else {
            return MTX_ERR_INVALID_MTX_FIELD;
        }

    } else {
        return MTX_ERR_INVALID_MTX_FORMAT;
    }

    return MTX_SUCCESS;
}

/**
 * `write_mtx()` writes a matrix or vector to a stream in the Matrix
 * Market format.
 */
static int write_mtx(
    const struct mtx * mtx,
    const struct stream * stream,
    const char * format)
{
    int err;
    if (mtx->object == mtx_matrix) {
        return write_matrix(mtx, stream, format);
    } else if (mtx->object == mtx_vector) {
        return write_vector(mtx, stream, format);
    } else {
        return MTX_ERR_INVALID_MTX_OBJECT;
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_fwrite()` writes a matrix to a stream in the Matrix Market
 * format.
 *
 * If `format' is `NULL', then the format specifier '%d' is used to
 * print integers and '%f' is used to print floating point
 * numbers. Otherwise, the given format string is used when printing
 * numerical values.
 *
 * The format string follows the conventions of `printf'. If the field
 * is `real', `double' or `complex', then the format specifiers '%e',
 * '%E', '%f', '%F', '%g' or '%G' may be used. If the field is
 * `integer', then the format specifier must be '%d'. The format
 * string is ignored if the field is `pattern'. Field width and
 * precision may be specified (e.g., "%3.1f"), but variable field
 * width and precision (e.g., "%*.*f"), as well as length modifiers
 * (e.g., "%Lf") are not allowed.
 */
int mtx_fwrite(
    const struct mtx * mtx,
    FILE * f,
    const char * format)
{
    struct stream * stream = stream_init_stdio(f);
    if (!stream)
        return MTX_ERR_ERRNO;
    int err = write_mtx(mtx, stream, format);
    free(stream);
    return err;
}

#ifdef LIBMTX_HAVE_LIBZ
/**
 * `mtx_gzwrite()` writes a matrix or vector to a gzip-compressed
 * stream in Matrix Market format.
 *
 * If `format' is `NULL', then the format specifier '%d' is used to
 * print integers and '%f' is used to print floating point
 * numbers. Otherwise, the given format string is used when printing
 * numerical values.
 *
 * The format string follows the conventions of `printf'. If the field
 * is `real', `double' or `complex', then the format specifiers '%e',
 * '%E', '%f', '%F', '%g' or '%G' may be used. If the field is
 * `integer', then the format specifier must be '%d'. The format
 * string is ignored if the field is `pattern'. Field width and
 * precision may be specified (e.g., "%3.1f"), but variable field
 * width and precision (e.g., "%*.*f"), as well as length modifiers
 * (e.g., "%Lf") are not allowed.
 */
int mtx_gzwrite(
    const struct mtx * mtx,
    gzFile f,
    const char * format)
{
    struct stream * stream = stream_init_gz(f);
    if (!stream)
        return MTX_ERR_ERRNO;
    int err = write_mtx(mtx, stream, format);
    free(stream);
    return err;
}
#endif