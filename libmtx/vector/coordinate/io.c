/* This file is part of libmtx.
 *
 * Copyright (C) 2022 James D. Trotter
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
 * Last modified: 2022-01-16
 *
 * Input/output for sparse vectors in coordinate format.
 */

#include <libmtx/libmtx-config.h>

#include <libmtx/error.h>
#include <libmtx/vector/coordinate/data.h>
#include <libmtx/vector/coordinate/io.h>

#include "libmtx/util/fmtspec.h"
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
 * `mtx_vector_coordinate_parse_data_real_single()' parses a data line
 * from a Matrix Market file for a single precision, real vector in
 * coordinate format.
 */
int mtx_vector_coordinate_parse_data_real_single(
    const char * line,
    int * bytes_read,
    const char ** endptr,
    struct mtx_vector_coordinate_real_single * data,
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
 * `mtx_vector_coordinate_parse_data_real_double()' parses a data line
 * from a Matrix Market file for a double precision, real vector in
 * coordinate format.
 */
int mtx_vector_coordinate_parse_data_real_double(
    const char * line,
    int * bytes_read,
    const char ** endptr,
    struct mtx_vector_coordinate_real_double * data,
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
 * `mtx_vector_coordinate_parse_data_complex_single()' parses a data
 * line from a Matrix Market file for a single precision, complex
 * vector in coordinate format.
 */
int mtx_vector_coordinate_parse_data_complex_single(
    const char * line,
    int * bytes_read,
    const char ** endptr,
    struct mtx_vector_coordinate_complex_single * data,
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

    err = parse_float(*endptr, " ", &data->a[0], endptr);
    if (err == EINVAL) {
        return MTX_ERR_INVALID_MTX_DATA;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    }
    *bytes_read = *endptr - line;

    err = parse_float(*endptr, "\n", &data->a[1], endptr);
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
 * `mtx_vector_coordinate_parse_data_complex_double()' parses a data
 * line from a Matrix Market file for a double precision, complex
 * vector in coordinate format.
 */
int mtx_vector_coordinate_parse_data_complex_double(
    const char * line,
    int * bytes_read,
    const char ** endptr,
    struct mtx_vector_coordinate_complex_double * data,
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

    err = parse_double(*endptr, " ", &data->a[0], endptr);
    if (err == EINVAL) {
        return MTX_ERR_INVALID_MTX_DATA;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    }
    *bytes_read = *endptr - line;

    err = parse_double(*endptr, "\n", &data->a[1], endptr);
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
 * `mtx_vector_coordinate_parse_data_integer_single()' parses a data
 * line from a Matrix Market file for a single precision, integer
 * vector in coordinate format.
 */
int mtx_vector_coordinate_parse_data_integer_single(
    const char * line,
    int * bytes_read,
    const char ** endptr,
    struct mtx_vector_coordinate_integer_single * data,
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
 * `mtx_vector_coordinate_parse_data_integer_double()' parses a data
 * line from a Matrix Market file for a double precision, integer
 * vector in coordinate format.
 */
int mtx_vector_coordinate_parse_data_integer_double(
    const char * line,
    int * bytes_read,
    const char ** endptr,
    struct mtx_vector_coordinate_integer_double * data,
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

    err = parse_int64(*endptr, "\n", &data->a, endptr);
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
 * `mtx_vector_coordinate_parse_data_pattern()' parses a data line
 * from a Matrix Market file for a pattern vector in coordinate
 * format.
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
