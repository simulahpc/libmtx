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
 * Input/output for dense vectors in array format.
 */

#include <libmtx/libmtx-config.h>

#include <libmtx/error.h>
#include <libmtx/mtx/io.h>
#include <libmtx/mtx/mtx.h>
#include <libmtx/vector/array.h>

#include "libmtx/util/format.h"
#include "libmtx/util/parse.h"

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
 * `mtx_vector_array_parse_size()` parse a size line from a Matrix
 * Market file for a vector in array format.
 */
int mtx_vector_array_parse_size(
    const char * line,
    int * bytes_read,
    const char ** endptr,
    enum mtx_object object,
    enum mtx_format format,
    enum mtx_field field,
    enum mtx_symmetry symmetry,
    int * num_rows,
    int * num_columns,
    int64_t * num_nonzeros,
    int64_t * size,
    int * nonzero_size)
{
    int err;
    *bytes_read = 0;
    if (object != mtx_vector)
        return MTX_ERR_INVALID_MTX_OBJECT;
    if (format != mtx_array)
        return MTX_ERR_INVALID_MTX_FORMAT;

    /* Parse the number of rows. */
    err = parse_int32(line, "\n", num_rows, endptr);
    if (err == EINVAL) {
        return MTX_ERR_INVALID_MTX_SIZE;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    }
    *bytes_read = (*endptr) - line;

    *num_columns = -1;
    *num_nonzeros = *num_rows;
    *size = *num_nonzeros;
    if (field == mtx_real) {
        *nonzero_size = sizeof(float);
    } else if (field == mtx_double) {
        *nonzero_size = sizeof(double);
    } else if (field == mtx_complex) {
        *nonzero_size = 2*sizeof(float);
    } else if (field == mtx_integer) {
        *nonzero_size = sizeof(int);
    } else {
        return MTX_ERR_INVALID_MTX_FIELD;
    }
    return MTX_SUCCESS;
}

#if 0
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
 * `mtx_vector_array_read_data()` reads lines of matrix data from a
 * stream in the Matrix Market file format.
 */
static int mtx_vector_array_read_data(
    enum mtx_object object,
    enum mtx_format format,
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
    if (object != mtx_vector)
        return MTX_ERR_INVALID_MTX_OBJECT;
    if (format != mtx_array)
        return MTX_ERR_INVALID_MTX_FORMAT;

    if (field == mtx_real) {
        /* 1. Allocate storage for matrix data. */
        float * data = (float *) malloc(size * sizeof(float));
        if (!data)
            return MTX_ERR_ERRNO;

        /* 2. Read each line of data. */
        for (int64_t k = 0; k < size; k++) {
            err = read_line(stream, line_max, linebuf);
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
            err = read_line(stream, line_max, linebuf);
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
            err = read_line(stream, line_max, linebuf);
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
            err = read_line(stream, line_max, linebuf);
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
 * `read_mtx()` reads a matrix or vector from a stream in Matrix
 * Market format using the given `getline' function to fetch each
 * line.
 */
static int read_mtx(
    struct mtx * mtx,
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

    mtx->comment_lines = NULL;
    mtx->data = NULL;

    /* 1. Parse the header line. */
    *line_number = 1;
    *column_number = 1;
    err = read_header_line(
        &mtx->object, &mtx->format,
        &mtx->field, &mtx->symmetry,
        stream, line_max, linebuf,
        line_number, column_number);
    if (err) {
        free(linebuf);
        return err;
    }

    /* Set extra header information. */
    mtx->triangle =
        mtx->object == mtx_mtx &&
        mtx->format == mtx_array &&
        (mtx->symmetry == mtx_symmetric ||
         mtx->symmetry == mtx_skew_symmetric ||
         mtx->symmetry == mtx_hermitian)
        ? mtx_lower_triangular : mtx_nontriangular;
    mtx->sorting = mtx->format == mtx_array ? mtx_row_major : mtx_unsorted;
    mtx->ordering = mtx_unordered;
    mtx->assembly = mtx->format == mtx_array ? mtx_assembled : mtx_unassembled;

    /* 2. Parse comment lines. */
    err = read_comment_lines(
        &mtx->num_comment_lines, &mtx->comment_lines,
        stream, line_max, linebuf, line_number, column_number);
    if (err) {
        free(linebuf);
        return err;
    }

    /* 3. Parse the size line. */
    err = mtx_vector_array_read_size(
        mtx->object, mtx->format, mtx->field, mtx->symmetry,
        &mtx->num_rows, &mtx->num_columns,
        &mtx->num_nonzeros, &mtx->size, &mtx->nonzero_size,
        stream, line_max, linebuf, line_number, column_number);
    if (err) {
        for (int i = 0; i < mtx->num_comment_lines; i++)
            free(mtx->comment_lines[i]);
        free(mtx->comment_lines);
        mtx->comment_lines = NULL;
        free(linebuf);
        return err;
    }

    /* 4. Parse the data. */
    err = mtx_vector_array_read_data(
        mtx->object, mtx->format, mtx->field,
        mtx->num_rows, mtx->num_columns, mtx->size,
        &mtx->data, stream, line_max, linebuf,
        line_number, column_number);
    if (err) {
        for (int i = 0; i < mtx->num_comment_lines; i++)
            free(mtx->comment_lines[i]);
        free(mtx->comment_lines);
        mtx->comment_lines = NULL;
        free(linebuf);
        return err;
    }

    free(linebuf);
    return MTX_SUCCESS;
}

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
 * `mtx_vector_array_write()` writes a vector to a stream in the
 * Matrix Market format.
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
static int mtx_vector_array_write(
    const struct mtx * mtx,
    const struct stream * stream,
    const char * format)
{
    int err;
    if (mtx->object != mtx_vector)
        return MTX_ERR_INVALID_MTX_OBJECT;
    if (mtx->format != mtx_array)
        return MTX_ERR_INVALID_MTX_FORMAT;

    /* Parse and validate the format string. */
    if (format) {
        err = validate_format_string(format, mtx->field);
        if (err)
            return err;
    }

    /* 1. Write the header line. */
    stream_printf(
        stream, "%%%%MatrixMarket %s %s %s %s\n",
        mtx_object_str(mtx->object),
        mtx_format_str(mtx->format),
        mtx_field_str(mtx->field),
        mtx_symmetry_str(mtx->symmetry));

    /* 2. Write comment lines. */
    for (int i = 0; i < mtx->num_comment_lines; i++)
        stream_printf(stream, "%s", mtx->comment_lines[i]);

    /* 3. Write the size line. */
    stream_printf(stream, "%"PRId64"\n", mtx->size);

    /* 4. Write the data. */
    if (mtx->field == mtx_real) {
        const float * a = (const float *) mtx->data;
        for (int i = 0; i < mtx->size; i++) {
            stream_printf(stream, format ? format : "%f", a[i]);
            stream_putc('\n', stream);
        }
    } else if (mtx->field == mtx_double) {
        const double * a = (const double *) mtx->data;
        for (int i = 0; i < mtx->size; i++) {
            stream_printf(stream, format ? format : "%f", a[i]);
            stream_putc('\n', stream);
        }
    } else if (mtx->field == mtx_complex) {
        const float * a = (const float *) mtx->data;
        for (int i = 0; i < mtx->size; i++) {
            stream_printf(stream, format ? format : "%f", a[2*i+0]);
            stream_putc(' ', stream);
            stream_printf(stream, format ? format : "%f", a[2*i+1]);
            stream_putc('\n', stream);
        }
    } else if (mtx->field == mtx_integer) {
        const int * a = (const int *) mtx->data;
        for (int i = 0; i < mtx->size; i++) {
            stream_printf(stream, format ? format : "%d", a[i]);
            stream_putc('\n', stream);
        }
    } else {
        return MTX_ERR_INVALID_MTX_FIELD;
    }

    return MTX_SUCCESS;
}
#endif
