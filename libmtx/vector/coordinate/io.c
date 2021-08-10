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
 * Input/output for sparse vectors in coordinate format.
 */

#include <libmtx/libmtx-config.h>

#include <libmtx/error.h>
#include <libmtx/mtx/io.h>
#include <libmtx/mtx/mtx.h>
#include <libmtx/vector/coordinate/coordinate.h>

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
 * `mtx_vector_coordinate_parse_size()` parse a size line from a
 * Matrix Market file for a vector in coordinate format.
 */
int mtx_vector_coordinate_parse_size(
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
    if (format != mtx_coordinate)
        return MTX_ERR_INVALID_MTX_FORMAT;

    /* Parse the number of rows. */
    err = parse_int32(line, " ", num_rows, endptr);
    if (err == EINVAL) {
        return MTX_ERR_INVALID_MTX_SIZE;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    }
    *bytes_read = (*endptr) - line;

    /* Parse the number of stored nonzeros. */
    err = parse_int64(*endptr, "\n", size, endptr);
    if (err == EINVAL) {
        return MTX_ERR_INVALID_MTX_SIZE;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    }
    *bytes_read = (*endptr) - line;

    *num_columns = -1;
    *num_nonzeros = *size;
    if (field == mtx_real) {
        *nonzero_size = sizeof(struct mtx_vector_coordinate_real);
    } else if (field == mtx_double) {
        *nonzero_size = sizeof(struct mtx_vector_coordinate_double);
    } else if (field == mtx_complex) {
        *nonzero_size = sizeof(struct mtx_vector_coordinate_complex);
    } else if (field == mtx_integer) {
        *nonzero_size = sizeof(struct mtx_vector_coordinate_integer);
    } else if (field == mtx_pattern) {
        *nonzero_size = sizeof(struct mtx_vector_coordinate_pattern);
    } else {
        return MTX_ERR_INVALID_MTX_FIELD;
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_vector_coordinate_parse_data_real()' parses a data line from a
 * Matrix Market file for a real vector in coordinate format.
 */
int mtx_vector_coordinate_parse_data_real(
    const char * line,
    int * bytes_read,
    const char ** endptr,
    struct mtx_vector_coordinate_real * data,
    int num_rows)
{
    int err;
    const char * tmp;
    if (!endptr)
        endptr = &tmp;
    err = parse_int32(line, " ", &data->i, endptr);
    if (err == EINVAL || (!err && (data->i < 1 || data->i > num_rows))) {
        return MTX_ERR_INVALID_MTX_DATA;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    }
    *bytes_read = *endptr - line;

    err = parse_float(*endptr, "\n", &data->a, endptr);
    if (err == EINVAL) {
        return MTX_ERR_INVALID_MTX_DATA;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    }
    *bytes_read = *endptr - line;
    return MTX_SUCCESS;
}

/**
 * `mtx_vector_coordinate_parse_data_double()' parses a data line from a
 * Matrix Market file for a double vector in coordinate format.
 */
int mtx_vector_coordinate_parse_data_double(
    const char * line,
    int * bytes_read,
    const char ** endptr,
    struct mtx_vector_coordinate_double * data,
    int num_rows)
{
    int err;
    const char * tmp;
    if (!endptr)
        endptr = &tmp;
    err = parse_int32(line, " ", &data->i, endptr);
    if (err == EINVAL || (!err && (data->i < 1 || data->i > num_rows))) {
        return MTX_ERR_INVALID_MTX_DATA;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    }
    *bytes_read = *endptr - line;

    err = parse_double(*endptr, "\n", &data->a, endptr);
    if (err == EINVAL) {
        return MTX_ERR_INVALID_MTX_DATA;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    }
    *bytes_read = *endptr - line;
    return MTX_SUCCESS;
}

/**
 * `mtx_vector_coordinate_parse_data_complex()' parses a data line from a
 * Matrix Market file for a complex vector in coordinate format.
 */
int mtx_vector_coordinate_parse_data_complex(
    const char * line,
    int * bytes_read,
    const char ** endptr,
    struct mtx_vector_coordinate_complex * data,
    int num_rows)
{
    int err;
    const char * tmp;
    if (!endptr)
        endptr = &tmp;
    err = parse_int32(line, " ", &data->i, endptr);
    if (err == EINVAL || (!err && (data->i < 1 || data->i > num_rows))) {
        return MTX_ERR_INVALID_MTX_DATA;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    }
    *bytes_read = *endptr - line;

    err = parse_float(*endptr, " ", &data->a, endptr);
    if (err == EINVAL) {
        return MTX_ERR_INVALID_MTX_DATA;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    }
    *bytes_read = *endptr - line;

    err = parse_float(*endptr, "\n", &data->b, endptr);
    if (err == EINVAL) {
        return MTX_ERR_INVALID_MTX_DATA;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    }
    *bytes_read = *endptr - line;
    return MTX_SUCCESS;
}

/**
 * `mtx_vector_coordinate_parse_data_integer()' parses a data line from a
 * Matrix Market file for a integer vector in coordinate format.
 */
int mtx_vector_coordinate_parse_data_integer(
    const char * line,
    int * bytes_read,
    const char ** endptr,
    struct mtx_vector_coordinate_integer * data,
    int num_rows)
{
    int err;
    const char * tmp;
    if (!endptr)
        endptr = &tmp;
    err = parse_int32(line, " ", &data->i, endptr);
    if (err == EINVAL || (!err && (data->i < 1 || data->i > num_rows))) {
        return MTX_ERR_INVALID_MTX_DATA;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    }
    *bytes_read = *endptr - line;

    err = parse_int32(*endptr, "\n", &data->a, endptr);
    if (err == EINVAL) {
        return MTX_ERR_INVALID_MTX_DATA;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    }
    *bytes_read = *endptr - line;
    return MTX_SUCCESS;
}

/**
 * `mtx_vector_coordinate_parse_data_pattern()' parses a data line from a
 * Matrix Market file for a pattern vector in coordinate format.
 */
int mtx_vector_coordinate_parse_data_pattern(
    const char * line,
    int * bytes_read,
    const char ** endptr,
    struct mtx_vector_coordinate_pattern * data,
    int num_rows)
{
    int err;
    const char * tmp;
    if (!endptr)
        endptr = &tmp;
    err = parse_int32(line, "\n", &data->i, endptr);
    if (err == EINVAL || (!err && (data->i < 1 || data->i > num_rows))) {
        return MTX_ERR_INVALID_MTX_DATA;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    }
    *bytes_read = *endptr - line;
    return MTX_SUCCESS;
}
