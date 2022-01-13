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
#include <libmtx/matrix/array.h>
#include <libmtx/matrix/coordinate.h>
#include <libmtx/matrix/coordinate/io.h>
#include <libmtx/mtx/io.h>
#include <libmtx/mtx/mtx.h>
#include <libmtx/precision.h>
#include <libmtx/vector/array.h>
#include <libmtx/vector/coordinate.h>
#include <libmtx/vector/coordinate/io.h>

#include "../util/format.h"
#include "../util/io.h"
#include "../util/parse.h"

#ifdef LIBMTX_HAVE_LIBZ
#include <zlib.h>
#endif

#include <errno.h>
#include <unistd.h>

#include <float.h>
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
 * If an error code is returned, then `line_number' and
 * `column_number' are used to return the line and column at which the
 * error was encountered during the parsing of the Matrix Market file.
 */
int mtx_read(
    struct mtx * mtx,
    enum mtxprecision precision,
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

        err = mtx_fread(
            mtx, precision, f,
            line_number, column_number);
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

        err = mtx_gzread(
            mtx, precision, f,
            line_number, column_number);
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
 * If ‘fmt’ is ‘NULL’, then the format specifier ‘%g’ is used to print
 * floating point numbers with enough digits to ensure correct
 * round-trip conversion from decimal text and back.  Otherwise, the
 * given format string is used to print numerical values.
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
    const char * fmt)
{
    int err;
    if (!gzip) {
        FILE * f;
        if (strcmp(path, "-") == 0) {
            f = stdout;
        } else if ((f = fopen(path, "w")) == NULL) {
            return MTX_ERR_ERRNO;
        }

        err = mtx_fwrite(mtx, f, fmt);
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

        err = mtx_gzwrite(mtx, f, fmt);
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
    struct mtx_header * header,
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

    *line_number = 1;
    *column_number = 1;

    int bytes_read;
    err = mtx_header_parse(
        header, linebuf, &bytes_read, NULL);
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
    struct mtx_comments * comments,
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
    comments->num_comment_lines = 0;
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
        comments->num_comment_lines++;
    }

    /* 2. Allocate storage for comment lines. */
    comments->comment_lines = malloc(
        comments->num_comment_lines * sizeof(char *));
    if (!comments->comment_lines) {
        while (node) {
            struct comment_line_list * prev = node->prev;
            free(node->comment_line);
            free(node);
            node = prev;
        }
        return MTX_ERR_ERRNO;
    }

    /* 3. Initialise the array of comment lines. */
    for (int i = 0; i < comments->num_comment_lines; i++) {
        comments->comment_lines[i] = root->comment_line;
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
    struct mtx_size * size,
    enum mtx_object object,
    enum mtx_format format,
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
    err = mtx_size_parse(
        size, object, format,
        linebuf, &bytes_read, NULL);
    if (err) {
        *column_number = bytes_read+1;
        return err;
    }
    (*line_number)++; *column_number = 1;
    return MTX_SUCCESS;
}

/**
 * `parse_array_real_single()' parses a single nonzero for a matrix
 * whose format is `array' and field is `real' in single precision.
 */
static int parse_array_real_single(
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
 * `parse_array_real_double()' parses a single nonzero for a matrix
 * whose format is `array' and field is `double' in double precision.
 */
static int parse_array_real_double(
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
 * `parse_array_complex_single()' parses a single nonzero for a matrix
 * whose format is `array' and field is `complex' in single precision.
 */
static int parse_array_complex_single(
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
 * `parse_array_integer_single()' parses a single nonzero for a matrix
 * whose format is `array' and field is `integer' in single precision.
 */
static int parse_array_integer_single(
    const char * s, int32_t * a)
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
 * `read_data_matrix_array()` reads data lines for a matrix in array
 * format from a stream in the Matrix Market file format.
 */
static int read_data_matrix_array(
    struct mtx_matrix_array_data * matrix_array,
    const struct stream * stream,
    size_t line_max,
    char * linebuf,
    int * line_number,
    int * column_number)
{
    int err;

    if (matrix_array->field == mtx_real) {
        if (matrix_array->precision == mtx_single) {
            float * data = matrix_array->data.real_single;
            for (int64_t k = 0; k < matrix_array->size; k++) {
                err = stream_read_line(stream, line_max, linebuf);
                if (err)
                    return err;
                err = parse_array_real_single(linebuf, &data[k]);
                if (err)
                    return err;
                (*line_number)++; *column_number = 1;
            }
        } else if (matrix_array->precision == mtx_double) {
            double * data = matrix_array->data.real_double;
            for (int64_t k = 0; k < matrix_array->size; k++) {
                err = stream_read_line(stream, line_max, linebuf);
                if (err)
                    return err;
                err = parse_array_real_double(linebuf, &data[k]);
                if (err)
                    return err;
                (*line_number)++; *column_number = 1;
            }
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (matrix_array->field == mtx_complex) {
        if (matrix_array->precision == mtx_single) {
            float (* data)[2] = matrix_array->data.complex_single;
            for (int64_t k = 0; k < matrix_array->size; k++) {
                err = stream_read_line(stream, line_max, linebuf);
                if (err)
                    return err;
                err = parse_array_complex_single(
                    linebuf, &data[k][0], &data[k][1]);
                if (err)
                    return err;
                (*line_number)++; *column_number = 1;
            }
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }

    } else if (matrix_array->field == mtx_integer) {
        if (matrix_array->precision == mtx_single) {
            int32_t * data = matrix_array->data.integer_single;
            for (int64_t k = 0; k < matrix_array->size; k++) {
                err = stream_read_line(stream, line_max, linebuf);
                if (err)
                    return err;
                err = parse_array_integer_single(linebuf, &data[k]);
                if (err)
                    return err;
                (*line_number)++; *column_number = 1;
            }
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else {
        return MTX_ERR_INVALID_MTX_FIELD;
    }
    return MTX_SUCCESS;
}

/**
 * `read_data_matrix_coordinate()` reads data lines for a matrix in
 * coordinate format from a stream in the Matrix Market file format.
 */
static int read_data_matrix_coordinate(
    struct mtx_matrix_coordinate_data * matrix_coordinate,
    const struct stream * stream,
    size_t line_max,
    char * linebuf,
    int * line_number,
    int * column_number)
{
    int err;
    for (int64_t k = 0; k < matrix_coordinate->size; k++) {
        err = stream_read_line(stream, line_max, linebuf);
        if (err)
            return err;

        if (matrix_coordinate->field == mtx_real) {
            if (matrix_coordinate->precision == mtx_single) {
                struct mtx_matrix_coordinate_real_single * data =
                    matrix_coordinate->data.real_single;
                int bytes_read;
                err = mtx_matrix_coordinate_parse_data_real_single(
                    linebuf, &bytes_read, NULL, &data[k],
                    matrix_coordinate->num_rows,
                    matrix_coordinate->num_columns);
                if (err) {
                    *column_number += bytes_read;
                    return err;
                }
            } else if (matrix_coordinate->precision == mtx_double) {
                struct mtx_matrix_coordinate_real_double * data =
                    matrix_coordinate->data.real_double;
                int bytes_read;
                err = mtx_matrix_coordinate_parse_data_real_double(
                    linebuf, &bytes_read, NULL, &data[k],
                    matrix_coordinate->num_rows,
                    matrix_coordinate->num_columns);
                if (err) {
                    *column_number += bytes_read;
                    return err;
                }
            } else {
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (matrix_coordinate->field == mtx_complex) {
            if (matrix_coordinate->precision == mtx_single) {
                struct mtx_matrix_coordinate_complex_single * data =
                    matrix_coordinate->data.complex_single;
                int bytes_read;
                err = mtx_matrix_coordinate_parse_data_complex_single(
                    linebuf, &bytes_read, NULL, &data[k],
                    matrix_coordinate->num_rows,
                    matrix_coordinate->num_columns);
                if (err) {
                    *column_number += bytes_read;
                    return err;
                }
            } else {
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (matrix_coordinate->field == mtx_integer) {
            if (matrix_coordinate->precision == mtx_single) {
                struct mtx_matrix_coordinate_integer_single * data =
                    matrix_coordinate->data.integer_single;
                int bytes_read;
                err = mtx_matrix_coordinate_parse_data_integer_single(
                    linebuf, &bytes_read, NULL, &data[k],
                    matrix_coordinate->num_rows,
                    matrix_coordinate->num_columns);
                if (err) {
                    *column_number += bytes_read;
                    return err;
                }
            } else {
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (matrix_coordinate->field == mtx_pattern) {
            struct mtx_matrix_coordinate_pattern * data =
                    matrix_coordinate->data.pattern;
            int bytes_read;
            err = mtx_matrix_coordinate_parse_data_pattern(
                linebuf, &bytes_read, NULL, &data[k],
                matrix_coordinate->num_rows,
                matrix_coordinate->num_columns);
            if (err) {
                *column_number += bytes_read;
                return err;
            }
        } else {
            return MTX_ERR_INVALID_MTX_FIELD;
        }

        (*line_number)++; *column_number = 1;
    }
    return MTX_SUCCESS;
}

/**
 * `read_data_vector_array()` reads data lines for a vector in array
 * format from a stream in the Matrix Market file format.
 */
static int read_data_vector_array(
    struct mtx_vector_array_data * vector_array,
    const struct stream * stream,
    size_t line_max,
    char * linebuf,
    int * line_number,
    int * column_number)
{
    int err;

    if (vector_array->field == mtx_real) {
        if (vector_array->precision == mtx_single) {
            float * data = vector_array->data.real_single;
            for (int64_t k = 0; k < vector_array->size; k++) {
                err = stream_read_line(stream, line_max, linebuf);
                if (err)
                    return err;
                err = parse_array_real_single(linebuf, &data[k]);
                if (err)
                    return err;
                (*line_number)++; *column_number = 1;
            }
        } else if (vector_array->precision == mtx_double) {
            double * data = vector_array->data.real_double;
            for (int64_t k = 0; k < vector_array->size; k++) {
                err = stream_read_line(stream, line_max, linebuf);
                if (err)
                    return err;
                err = parse_array_real_double(linebuf, &data[k]);
                if (err)
                    return err;
                (*line_number)++; *column_number = 1;
            }
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (vector_array->field == mtx_complex) {
        if (vector_array->precision == mtx_single) {
            float (* data)[2] = vector_array->data.complex_single;
            for (int64_t k = 0; k < vector_array->size; k++) {
                err = stream_read_line(stream, line_max, linebuf);
                if (err)
                    return err;
                err = parse_array_complex_single(
                    linebuf, &data[k][0], &data[k][1]);
                if (err)
                    return err;
                (*line_number)++; *column_number = 1;
            }
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }

    } else if (vector_array->field == mtx_integer) {
        if (vector_array->precision == mtx_single) {
            int32_t * data = vector_array->data.integer_single;
            for (int64_t k = 0; k < vector_array->size; k++) {
                err = stream_read_line(stream, line_max, linebuf);
                if (err)
                    return err;
                err = parse_array_integer_single(linebuf, &data[k]);
                if (err)
                    return err;
                (*line_number)++; *column_number = 1;
            }
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else {
        return MTX_ERR_INVALID_MTX_FIELD;
    }
    return MTX_SUCCESS;
}

/**
 * `read_data_vector_coordinate()` reads data lines of a vector in
 * coordinate format from a stream in the Matrix Market file format.
 */
static int read_data_vector_coordinate(
    struct mtx_vector_coordinate_data * vector_coordinate,
    const struct stream * stream,
    size_t line_max,
    char * linebuf,
    int * line_number,
    int * column_number)
{
    int err;
    for (int64_t k = 0; k < vector_coordinate->size; k++) {
        err = stream_read_line(stream, line_max, linebuf);
        if (err)
            return err;

        if (vector_coordinate->field == mtx_real) {
            if (vector_coordinate->precision == mtx_single) {
                struct mtx_vector_coordinate_real_single * data =
                    vector_coordinate->data.real_single;
                int bytes_read;
                err = mtx_vector_coordinate_parse_data_real_single(
                    linebuf, &bytes_read, NULL,
                    &data[k], vector_coordinate->num_rows);
                if (err) {
                    *column_number += bytes_read;
                    return err;
                }
            } else if (vector_coordinate->precision == mtx_double) {
                struct mtx_vector_coordinate_real_double * data =
                    vector_coordinate->data.real_double;
                int bytes_read;
                err = mtx_vector_coordinate_parse_data_real_double(
                    linebuf, &bytes_read, NULL,
                    &data[k], vector_coordinate->num_rows);
                if (err) {
                    *column_number += bytes_read;
                    return err;
                }
            } else {
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (vector_coordinate->field == mtx_complex) {
            if (vector_coordinate->precision == mtx_single) {
                struct mtx_vector_coordinate_complex_single * data =
                    vector_coordinate->data.complex_single;
                int bytes_read;
                err = mtx_vector_coordinate_parse_data_complex_single(
                    linebuf, &bytes_read, NULL,
                    &data[k], vector_coordinate->num_rows);
                if (err) {
                    *column_number += bytes_read;
                    return err;
                }
            } else {
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (vector_coordinate->field == mtx_integer) {
            if (vector_coordinate->precision == mtx_single) {
                struct mtx_vector_coordinate_integer_single * data =
                    vector_coordinate->data.integer_single;
                int bytes_read;
                err = mtx_vector_coordinate_parse_data_integer_single(
                    linebuf, &bytes_read, NULL,
                    &data[k], vector_coordinate->num_rows);
                if (err) {
                    *column_number += bytes_read;
                    return err;
                }
            } else {
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (vector_coordinate->field == mtx_pattern) {
            struct mtx_vector_coordinate_pattern * data =
                    vector_coordinate->data.pattern;
            int bytes_read;
            err = mtx_vector_coordinate_parse_data_pattern(
                linebuf, &bytes_read, NULL,
                &data[k], vector_coordinate->num_rows);
            if (err) {
                *column_number += bytes_read;
                return err;
            }
        } else {
            return MTX_ERR_INVALID_MTX_FIELD;
        }

        (*line_number)++; *column_number = 1;
    }
    return MTX_SUCCESS;
}

/**
 * `read_mtx()' reads a matrix or vector from a stream in Matrix
 * Market format using the given `getline' function to fetch each
 * line.
 *
 * If an error code is returned, then `line_number' and
 * `column_number' are used to return the line and column at which the
 * error was encountered during the parsing of the Matrix Market file.
 */
static int read_mtx(
    struct mtx_header * header,
    struct mtx * mtx,
    enum mtxprecision precision,
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

    /* 1. Parse the header line. */
    err = read_header_line(
        header, stream, line_max, linebuf,
        line_number, column_number);
    if (err) {
        free(linebuf);
        return err;
    }

    /* 2. Parse comment lines. */
    struct mtx_comments comments;
    err = read_comment_lines(
        &comments, stream, line_max, linebuf,
        line_number, column_number);
    if (err) {
        free(linebuf);
        return err;
    }

    /* 3. Parse the size line. */
    struct mtx_size size;
    err = read_size_line(
        &size, header->object, header->format,
        stream, line_max, linebuf,
        line_number, column_number);
    if (err) {
        mtx_comments_free(&comments);
        free(linebuf);
        return err;
    }

    /* 4. Allocate storage for the matrix or vector, and parse the
     *    data lines. */
    if (header->object == mtx_matrix) {
        if (header->format == mtx_array) {
            enum mtx_triangle triangle;
            if (header->symmetry == mtx_general) {
                triangle = mtx_nontriangular;
            } else if (header->symmetry == mtx_symmetric ||
                       header->symmetry == mtx_hermitian)
            {
                triangle = mtx_lower_triangular;
            } else if (header->symmetry == mtx_skew_symmetric) {
                triangle = mtx_strict_lower_triangular;
            } else {
                mtx_comments_free(&comments);
                free(linebuf);
                return MTX_ERR_INVALID_MTX_SYMMETRY;
            }

            err = mtx_alloc_matrix_array(
                mtx, header->field, precision, header->symmetry,
                triangle, mtx_row_major,
                comments.num_comment_lines, (const char **) comments.comment_lines,
                size.num_rows, size.num_columns);
            if (err) {
                mtx_comments_free(&comments);
                free(linebuf);
                return err;
            }
            mtx_comments_free(&comments);

            err = read_data_matrix_array(
                &mtx->storage.matrix_array,
                stream, line_max, linebuf,
                line_number, column_number);
            if (err) {
                mtx_free(mtx);
                free(linebuf);
                return err;
            }

        } else if (header->format == mtx_coordinate) {
            err = mtx_alloc_matrix_coordinate(
                mtx, header->field, precision, header->symmetry,
                comments.num_comment_lines, (const char **) comments.comment_lines,
                size.num_rows, size.num_columns, size.num_nonzeros);
            if (err) {
                mtx_comments_free(&comments);
                free(linebuf);
                return err;
            }
            mtx_comments_free(&comments);

            err = read_data_matrix_coordinate(
                &mtx->storage.matrix_coordinate,
                stream, line_max, linebuf,
                line_number, column_number);
            if (err) {
                mtx_free(mtx);
                free(linebuf);
                return err;
            }

        } else {
            mtx_comments_free(&comments);
            free(linebuf);
            return MTX_ERR_INVALID_MTX_FORMAT;
        }

    } else if (header->object == mtx_vector) {
        if (header->format == mtx_array) {
            err = mtx_alloc_vector_array(
                mtx, header->field, precision,
                comments.num_comment_lines, (const char **) comments.comment_lines,
                size.num_rows);
            if (err) {
                mtx_comments_free(&comments);
                free(linebuf);
                return err;
            }
            mtx_comments_free(&comments);

            err = read_data_vector_array(
                &mtx->storage.vector_array,
                stream, line_max, linebuf,
                line_number, column_number);
            if (err) {
                mtx_free(mtx);
                free(linebuf);
                return err;
            }

        } else if (header->format == mtx_coordinate) {
            err = mtx_alloc_vector_coordinate(
                mtx, header->field, precision,
                comments.num_comment_lines, (const char **) comments.comment_lines,
                size.num_rows, size.num_nonzeros);
            if (err) {
                mtx_comments_free(&comments);
                free(linebuf);
                return err;
            }
            mtx_comments_free(&comments);

            err = read_data_vector_coordinate(
                &mtx->storage.vector_coordinate,
                stream, line_max, linebuf,
                line_number, column_number);
            if (err) {
                mtx_free(mtx);
                free(linebuf);
                return err;
            }
        } else {
            mtx_comments_free(&comments);
            free(linebuf);
            return MTX_ERR_INVALID_MTX_FORMAT;
        }

    } else {
        mtx_comments_free(&comments);
        free(linebuf);
        return MTX_ERR_INVALID_MTX_OBJECT;
    }

    free(linebuf);
    return MTX_SUCCESS;
}

/**
 * `mtx_fread()` reads an object (matrix or vector) from a stream in
 * Matrix Market format.
 *
 * The `precision' argument specifies which precision to use for
 * storing matrix or vector values.
 *
 * If an error code is returned, then `line_number' and
 * `column_number' are used to return the line and column at which the
 * error was encountered during the parsing of the Matrix Market file.
 */
int mtx_fread(
    struct mtx * mtx,
    enum mtxprecision precision,
    FILE * f,
    int * line_number,
    int * column_number)
{
    struct stream * stream = stream_init_stdio(f);
    if (!stream)
        return MTX_ERR_ERRNO;
    struct mtx_header header;
    int err = read_mtx(
        &header, mtx, precision,
        stream, line_number, column_number);
    free(stream);
    return err;
}

#ifdef LIBMTX_HAVE_LIBZ
/**
 * `mtx_gzread()` reads a matrix or vector from a gzip-compressed
 * stream in Matrix Market format.
 *
 * The `precision' argument specifies which precision to use for
 * storing matrix or vector values.
 *
 * If an error code is returned, then `line_number' and
 * `column_number' are used to return the line and column at which the
 * error was encountered during the parsing of the Matrix Market file.
 */
int mtx_gzread(
    struct mtx * mtx,
    enum mtxprecision precision,
    gzFile f,
    int * line_number,
    int * column_number)
{
    struct stream * stream = stream_init_gz(f);
    if (!stream)
        return MTX_ERR_ERRNO;
    struct mtx_header header;
    int err = read_mtx(
        &header, mtx, precision,
        stream, line_number, column_number);
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
 * If ‘fmt’ is ‘NULL’, then the format specifier ‘%g’ is used to print
 * floating point numbers with enough digits to ensure correct
 * round-trip conversion from decimal text and back.  Otherwise, the
 * given format string is used to print numerical values.
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
    const char * fmt)
{
    int err;
    if (matrix->object != mtx_matrix)
        return MTX_ERR_INVALID_MTX_OBJECT;

    /* Parse and validate the format string. */
    if (fmt) {
        err = validate_format_string(fmt, matrix->field);
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
            matrix->num_nonzeros);
    } else {
        return MTX_ERR_INVALID_MTX_FORMAT;
    }

    /* 4. Write the data. */
    if (matrix->format == mtx_array) {
        const struct mtx_matrix_array_data * matrix_array =
            &matrix->storage.matrix_array;
        if (matrix_array->field == mtx_real) {
            if (matrix_array->precision == mtx_single) {
                const float * a = matrix_array->data.real_single;
                for (int i = 0; i < matrix_array->num_rows; i++) {
                    for (int j = 0; j < matrix_array->num_columns; j++) {
                        stream_printf(stream, fmt ? fmt : "%.*g", FLT_DIG,
                                      a[i*matrix_array->num_columns+j]);
                        stream_putc('\n', stream);
                    }
                }
            } else if (matrix_array->precision == mtx_double) {
                const double * a = matrix_array->data.real_double;
                for (int i = 0; i < matrix_array->num_rows; i++) {
                    for (int j = 0; j < matrix_array->num_columns; j++) {
                        stream_printf(stream, fmt ? fmt : "%.*g", DBL_DIG,
                                      a[i*matrix_array->num_columns+j]);
                        stream_putc('\n', stream);
                    }
                }
            } else {
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (matrix_array->field == mtx_complex) {
            if (matrix_array->precision == mtx_single) {
                const float (* a)[2] = matrix_array->data.complex_single;
                for (int i = 0; i < matrix_array->num_rows; i++) {
                    for (int j = 0; j < matrix_array->num_columns; j++) {
                        stream_printf(stream, fmt ? fmt : "%.*g", FLT_DIG,
                                      a[i*matrix_array->num_columns+j][0]);
                        stream_putc(' ', stream);
                        stream_printf(stream, fmt ? fmt : "%.*g", FLT_DIG,
                                      a[i*matrix_array->num_columns+j][1]);
                        stream_putc('\n', stream);
                    }
                }
            } else {
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (matrix_array->field == mtx_integer) {
            if (matrix_array->precision == mtx_single) {
                const int32_t * a = matrix_array->data.integer_single;
                for (int i = 0; i < matrix_array->num_rows; i++) {
                    for (int j = 0; j < matrix_array->num_columns; j++) {
                        stream_printf(stream, fmt ? fmt : "%d",
                                      a[i*matrix_array->num_columns+j]);
                        stream_putc('\n', stream);
                    }
                }
            } else {
                return MTX_ERR_INVALID_PRECISION;
            }
        } else {
            return MTX_ERR_INVALID_MTX_FIELD;
        }

    } else if (matrix->format == mtx_coordinate) {
        const struct mtx_matrix_coordinate_data * matrix_coordinate =
            &matrix->storage.matrix_coordinate;
        if (matrix_coordinate->field == mtx_real) {
            if (matrix_coordinate->precision == mtx_single) {
                const struct mtx_matrix_coordinate_real_single * a =
                    matrix_coordinate->data.real_single;
                for (int64_t k = 0; k < matrix_coordinate->size; k++) {
                    stream_printf(stream, "%d %d ", a[k].i, a[k].j);
                    stream_printf(stream, fmt ? fmt : "%.*g", FLT_DIG, a[k].a);
                    stream_putc('\n', stream);
                }
            } else if (matrix_coordinate->precision == mtx_double) {
                const struct mtx_matrix_coordinate_real_double * a =
                    matrix_coordinate->data.real_double;
                for (int64_t k = 0; k < matrix_coordinate->size; k++) {
                    stream_printf(stream, "%d %d ", a[k].i, a[k].j);
                    stream_printf(stream, fmt ? fmt : "%.*g", DBL_DIG, a[k].a);
                    stream_putc('\n', stream);
                }
            } else {
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (matrix_coordinate->field == mtx_complex) {
            if (matrix_coordinate->precision == mtx_single) {
                const struct mtx_matrix_coordinate_complex_single * a =
                    matrix_coordinate->data.complex_single;
                for (int64_t k = 0; k < matrix_coordinate->size; k++) {
                    stream_printf(stream, "%d %d ", a[k].i, a[k].j);
                    stream_printf(stream, fmt ? fmt : "%.*g", FLT_DIG, a[k].a[0]);
                    stream_putc(' ', stream);
                    stream_printf(stream, fmt ? fmt : "%.*g", FLT_DIG, a[k].a[1]);
                    stream_putc('\n', stream);
                }
            } else {
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (matrix_coordinate->field == mtx_integer) {
            if (matrix_coordinate->precision == mtx_single) {
                const struct mtx_matrix_coordinate_integer_single * a =
                    matrix_coordinate->data.integer_single;
                for (int64_t k = 0; k < matrix_coordinate->size; k++) {
                    stream_printf(stream, "%d %d ", a[k].i, a[k].j);
                    stream_printf(stream, fmt ? fmt : "%d", a[k].a);
                    stream_putc('\n', stream);
                }
            } else {
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (matrix_coordinate->field == mtx_pattern) {
            const struct mtx_matrix_coordinate_pattern * a =
                    matrix_coordinate->data.pattern;
            for (int64_t k = 0; k < matrix_coordinate->size; k++) {
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
 * If ‘fmt’ is ‘NULL’, then the format specifier ‘%g’ is used to print
 * floating point numbers with enough digits to ensure correct
 * round-trip conversion from decimal text and back.  Otherwise, the
 * given format string is used to print numerical values.
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
    const char * fmt)
{
    int err;
    if (vector->object != mtx_vector)
        return MTX_ERR_INVALID_MTX_OBJECT;

    /* Parse and validate the format string. */
    if (fmt) {
        err = validate_format_string(fmt, vector->field);
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
        stream_printf(stream, "%"PRId64"\n", vector->num_rows);
    } else if (vector->format == mtx_coordinate) {
        stream_printf(
            stream, "%d %"PRId64"\n",
            vector->num_rows, vector->num_nonzeros);
    } else {
        return MTX_ERR_INVALID_MTX_FORMAT;
    }

    /* 4. Write the data. */
    if (vector->format == mtx_array) {
        const struct mtx_vector_array_data * vector_array =
            &vector->storage.vector_array;
        if (vector_array->field == mtx_real) {
            if (vector_array->precision == mtx_single) {
                const float * a = vector_array->data.real_single;
                for (int64_t k = 0; k < vector_array->size; k++) {
                    stream_printf(stream, fmt ? fmt : "%.*g", FLT_DIG, a[k]);
                    stream_putc('\n', stream);
                }
            } else if (vector_array->precision == mtx_double) {
                const double * a = vector_array->data.real_double;
                for (int64_t k = 0; k < vector_array->size; k++) {
                    stream_printf(stream, fmt ? fmt : "%.*g", DBL_DIG, a[k]);
                    stream_putc('\n', stream);
                }
            } else {
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (vector_array->field == mtx_complex) {
            if (vector_array->precision == mtx_single) {
                const float (* a)[2] = vector_array->data.complex_single;
                for (int64_t k = 0; k < vector_array->size; k++) {
                    stream_printf(stream, fmt ? fmt : "%.*g", FLT_DIG, a[k][0]);
                    stream_putc(' ', stream);
                    stream_printf(stream, fmt ? fmt : "%.*g", FLT_DIG, a[k][1]);
                    stream_putc('\n', stream);
                }
            } else {
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (vector_array->field == mtx_integer) {
            if (vector_array->precision == mtx_single) {
                const int32_t * a = vector_array->data.integer_single;
                for (int64_t k = 0; k < vector_array->size; k++) {
                    stream_printf(stream, fmt ? fmt : "%d", a[k]);
                    stream_putc('\n', stream);
                }
            } else {
                return MTX_ERR_INVALID_PRECISION;
            }
        } else {
            return MTX_ERR_INVALID_MTX_FIELD;
        }

    } else if (vector->format == mtx_coordinate) {
        const struct mtx_vector_coordinate_data * vector_coordinate =
            &vector->storage.vector_coordinate;
        if (vector_coordinate->field == mtx_real) {
            if (vector_coordinate->precision == mtx_single) {
                const struct mtx_vector_coordinate_real_single * a =
                    vector_coordinate->data.real_single;
                for (int64_t k = 0; k < vector_coordinate->size; k++) {
                    stream_printf(stream, "%d ", a[k].i);
                    stream_printf(stream, fmt ? fmt : "%.*g", FLT_DIG, a[k].a);
                    stream_putc('\n', stream);
                }
            } else if (vector_coordinate->precision == mtx_double) {
                const struct mtx_vector_coordinate_real_double * a =
                    vector_coordinate->data.real_double;
                for (int64_t k = 0; k < vector_coordinate->size; k++) {
                    stream_printf(stream, "%d ", a[k].i);
                    stream_printf(stream, fmt ? fmt : "%.*g", DBL_DIG, a[k].a);
                    stream_putc('\n', stream);
                }
            } else {
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (vector_coordinate->field == mtx_complex) {
            if (vector_coordinate->precision == mtx_single) {
                const struct mtx_vector_coordinate_complex_single * a =
                    vector_coordinate->data.complex_single;
                for (int64_t k = 0; k < vector_coordinate->size; k++) {
                    stream_printf(stream, "%d ", a[k].i);
                    stream_printf(stream, fmt ? fmt : "%.*g", FLT_DIG, a[k].a[0]);
                    stream_putc(' ', stream);
                    stream_printf(stream, fmt ? fmt : "%.*g", FLT_DIG, a[k].a[1]);
                    stream_putc('\n', stream);
                }
            } else {
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (vector_coordinate->field == mtx_integer) {
            if (vector_coordinate->precision == mtx_single) {
                const struct mtx_vector_coordinate_integer_single * a =
                    vector_coordinate->data.integer_single;
                for (int64_t k = 0; k < vector_coordinate->size; k++) {
                    stream_printf(stream, "%d ", a[k].i);
                    stream_printf(stream, fmt ? fmt : "%d", a[k].a);
                    stream_putc('\n', stream);
                }
            } else {
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (vector_coordinate->field == mtx_pattern) {
            const struct mtx_vector_coordinate_pattern * a =
                vector_coordinate->data.pattern;
            for (int64_t k = 0; k < vector_coordinate->size; k++) {
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
 * If ‘fmt’ is ‘NULL’, then the format specifier ‘%g’ is used to print
 * floating point numbers with enough digits to ensure correct
 * round-trip conversion from decimal text and back.  Otherwise, the
 * given format string is used to print numerical values.
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
    const char * fmt)
{
    struct stream * stream = stream_init_stdio(f);
    if (!stream)
        return MTX_ERR_ERRNO;
    int err = write_mtx(mtx, stream, fmt);
    free(stream);
    return err;
}

#ifdef LIBMTX_HAVE_LIBZ
/**
 * `mtx_gzwrite()` writes a matrix or vector to a gzip-compressed
 * stream in Matrix Market format.
 *
 * If ‘fmt’ is ‘NULL’, then the format specifier ‘%g’ is used to print
 * floating point numbers with enough digits to ensure correct
 * round-trip conversion from decimal text and back.  Otherwise, the
 * given format string is used to print numerical values.
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
    const char * fmt)
{
    struct stream * stream = stream_init_gz(f);
    if (!stream)
        return MTX_ERR_ERRNO;
    int err = write_mtx(mtx, stream, fmt);
    free(stream);
    return err;
}
#endif
