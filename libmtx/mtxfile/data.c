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
 * Matrix Market data lines.
 */

#include <libmtx/libmtx-config.h>

#include <libmtx/error.h>
#include <libmtx/mtx/precision.h>
#include <libmtx/mtxfile/coordinate.h>
#include <libmtx/mtxfile/data.h>
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

/*
 * Array formats
 */

/**
 * `mtxfile_parse_data_array_real_single()' parses a string containing
 * a data line for a Matrix Market file in array format with real
 * values.
 *
 * If `bytes_read' is not `NULL', then it is set to the number of
 * bytes consumed by the parser, if successful.  Similarly, if
 * `endptr' is not `NULL', then the address stored in `endptr' points
 * to the first character beyond the characters that were consumed
 * during parsing.
 */
int mtxfile_parse_data_array_real_single(
    float * data,
    int * bytes_read,
    const char ** endptr,
    const char * s)
{
    const char * t = s;
    int err = parse_float(t, "\n", data, &t);
    if (err == EINVAL) {
        return MTX_ERR_INVALID_MTX_DATA;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    }
    if (bytes_read)
        (*bytes_read) += t-s;
    if (endptr)
        *endptr = t;
    return MTX_SUCCESS;
}

/**
 * `mtxfile_parse_data_array_real_double()' parses a string containing
 * a data line for a Matrix Market file in array format with real
 * values.
 *
 * If `bytes_read' is not `NULL', then it is set to the number of
 * bytes consumed by the parser, if successful.  Similarly, if
 * `endptr' is not `NULL', then the address stored in `endptr' points
 * to the first character beyond the characters that were consumed
 * during parsing.
 */
int mtxfile_parse_data_array_real_double(
    double * data,
    int * bytes_read,
    const char ** endptr,
    const char * s)
{
    const char * t = s;
    int err = parse_double(t, "\n", data, &t);
    if (err == EINVAL) {
        return MTX_ERR_INVALID_MTX_DATA;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    }
    if (bytes_read)
        (*bytes_read) += t-s;
    if (endptr)
        *endptr = t;
    return MTX_SUCCESS;
}

/**
 * `mtxfile_parse_data_array_complex_single()' parses a string
 * containing a data line for a Matrix Market file in array format
 * with complex values.
 *
 * If `bytes_read' is not `NULL', then it is set to the number of
 * bytes consumed by the parser, if successful.  Similarly, if
 * `endptr' is not `NULL', then the address stored in `endptr' points
 * to the first character beyond the characters that were consumed
 * during parsing.
 */
int mtxfile_parse_data_array_complex_single(
    float (* data)[2],
    int * bytes_read,
    const char ** endptr,
    const char * s)
{
    int err;
    const char * t = s;
    err = parse_float(t, " ", &(*data)[0], &t);
    if (err == EINVAL) {
        return MTX_ERR_INVALID_MTX_DATA;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    }
    err = parse_float(t, "\n", &(*data)[1], &t);
    if (err == EINVAL) {
        return MTX_ERR_INVALID_MTX_DATA;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    }
    if (bytes_read)
        (*bytes_read) += t-s;
    if (endptr)
        *endptr = t;
    return MTX_SUCCESS;
}

/**
 * `mtxfile_parse_data_array_complex_double()' parses a string
 * containing a data line for a Matrix Market file in array format
 * with complex values.
 *
 * If `bytes_read' is not `NULL', then it is set to the number of
 * bytes consumed by the parser, if successful.  Similarly, if
 * `endptr' is not `NULL', then the address stored in `endptr' points
 * to the first character beyond the characters that were consumed
 * during parsing.
 */
int mtxfile_parse_data_array_complex_double(
    double (* data)[2],
    int * bytes_read,
    const char ** endptr,
    const char * s)
{
    int err;
    const char * t = s;
    err = parse_double(t, " ", &((*data)[0]), &t);
    if (err == EINVAL) {
        return MTX_ERR_INVALID_MTX_DATA;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    }
    err = parse_double(t, "\n", &((*data)[1]), &t);
    if (err == EINVAL) {
        return MTX_ERR_INVALID_MTX_DATA;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    }
    if (bytes_read)
        (*bytes_read) += t-s;
    if (endptr)
        *endptr = t;
    return MTX_SUCCESS;
}

/**
 * `mtxfile_parse_data_array_integer_single()' parses a string
 * containing a data line for a Matrix Market file in array format
 * with integer values.
 *
 * If `bytes_read' is not `NULL', then it is set to the number of
 * bytes consumed by the parser, if successful.  Similarly, if
 * `endptr' is not `NULL', then the address stored in `endptr' points
 * to the first character beyond the characters that were consumed
 * during parsing.
 */
int mtxfile_parse_data_array_integer_single(
    int32_t * data,
    int * bytes_read,
    const char ** endptr,
    const char * s)
{
    const char * t = s;
    int err = parse_int32(t, "\n", data, &t);
    if (err == EINVAL) {
        return MTX_ERR_INVALID_MTX_DATA;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    }
    if (bytes_read)
        (*bytes_read) += t-s;
    if (endptr)
        *endptr = t;
    return MTX_SUCCESS;
}

/**
 * `mtxfile_parse_data_array_integer_double()' parses a string
 * containing a data line for a Matrix Market file in array format
 * with integer values.
 *
 * If `bytes_read' is not `NULL', then it is set to the number of
 * bytes consumed by the parser, if successful.  Similarly, if
 * `endptr' is not `NULL', then the address stored in `endptr' points
 * to the first character beyond the characters that were consumed
 * during parsing.
 */
int mtxfile_parse_data_array_integer_double(
    int64_t * data,
    int * bytes_read,
    const char ** endptr,
    const char * s)
{
    const char * t = s;
    int err = parse_int64(t, "\n", data, &t);
    if (err == EINVAL) {
        return MTX_ERR_INVALID_MTX_DATA;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    }
    if (bytes_read)
        (*bytes_read) += t-s;
    if (endptr)
        *endptr = t;
    return MTX_SUCCESS;
}

/*
 * Matrix coordinate formats
 */

/**
 * `mtxfile_parse_data_matrix_coordinate_real_single()' parses a
 * string containing a data line for a Matrix Market file in matrix
 * coordinate format with real values.
 *
 * If `bytes_read' is not `NULL', then it is set to the number of
 * bytes consumed by the parser, if successful.  Similarly, if
 * `endptr' is not `NULL', then the address stored in `endptr' points
 * to the first character beyond the characters that were consumed
 * during parsing.
 */
int mtxfile_parse_data_matrix_coordinate_real_single(
    struct mtxfile_matrix_coordinate_real_single * data,
    int * bytes_read,
    const char ** endptr,
    const char * s,
    int num_rows,
    int num_columns)
{
    int err;
    const char * t = s;
    err = parse_int32(t, " ", &data->i, &t);
    if (err == EINVAL) {
        return MTX_ERR_INVALID_MTX_DATA;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    } else if (data->i <= 0 || data->i > num_rows) {
        return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    }
    err = parse_int32(t, " ", &data->j, &t);
    if (err == EINVAL) {
        return MTX_ERR_INVALID_MTX_DATA;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    } else if (data->j <= 0 || data->j > num_columns) {
        return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    }
    err = parse_float(t, "\n", &data->a, &t);
    if (err == EINVAL) {
        return MTX_ERR_INVALID_MTX_DATA;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    }
    if (bytes_read)
        (*bytes_read) += t-s;
    if (endptr)
        *endptr = t;
    return MTX_SUCCESS;
}

/**
 * `mtxfile_parse_data_matrix_coordinate_real_double()' parses a
 * string containing a data line for a Matrix Market file in matrix
 * coordinate format with real values.
 *
 * If `bytes_read' is not `NULL', then it is set to the number of
 * bytes consumed by the parser, if successful.  Similarly, if
 * `endptr' is not `NULL', then the address stored in `endptr' points
 * to the first character beyond the characters that were consumed
 * during parsing.
 */
int mtxfile_parse_data_matrix_coordinate_real_double(
    struct mtxfile_matrix_coordinate_real_double * data,
    int * bytes_read,
    const char ** endptr,
    const char * s,
    int num_rows,
    int num_columns)
{
    int err;
    const char * t = s;
    err = parse_int32(t, " ", &data->i, &t);
    if (err == EINVAL) {
        return MTX_ERR_INVALID_MTX_DATA;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    } else if (data->i <= 0 || data->i > num_rows) {
        return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    }
    err = parse_int32(t, " ", &data->j, &t);
    if (err == EINVAL) {
        return MTX_ERR_INVALID_MTX_DATA;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    } else if (data->j <= 0 || data->j > num_columns) {
        return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    }
    err = parse_double(t, "\n", &data->a, &t);
    if (err == EINVAL) {
        return MTX_ERR_INVALID_MTX_DATA;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    }
    if (bytes_read)
        (*bytes_read) += t-s;
    if (endptr)
        *endptr = t;
    return MTX_SUCCESS;
}

/**
 * `mtxfile_parse_data_matrix_coordinate_complex_single()' parses a
 * string containing a data line for a Matrix Market file in matrix
 * coordinate format with complex values.
 *
 * If `bytes_read' is not `NULL', then it is set to the number of
 * bytes consumed by the parser, if successful.  Similarly, if
 * `endptr' is not `NULL', then the address stored in `endptr' points
 * to the first character beyond the characters that were consumed
 * during parsing.
 */
int mtxfile_parse_data_matrix_coordinate_complex_single(
    struct mtxfile_matrix_coordinate_complex_single * data,
    int * bytes_read,
    const char ** endptr,
    const char * s,
    int num_rows,
    int num_columns)
{
    int err;
    const char * t = s;
    err = parse_int32(t, " ", &data->i, &t);
    if (err == EINVAL) {
        return MTX_ERR_INVALID_MTX_DATA;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    } else if (data->i <= 0 || data->i > num_rows) {
        return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    }
    err = parse_int32(t, " ", &data->j, &t);
    if (err == EINVAL) {
        return MTX_ERR_INVALID_MTX_DATA;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    } else if (data->j <= 0 || data->j > num_columns) {
        return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    }
    err = parse_float(t, " ", &data->a[0], &t);
    if (err == EINVAL) {
        return MTX_ERR_INVALID_MTX_DATA;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    }
    err = parse_float(t, "\n", &data->a[1], &t);
    if (err == EINVAL) {
        return MTX_ERR_INVALID_MTX_DATA;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    }
    if (bytes_read)
        (*bytes_read) += t-s;
    if (endptr)
        *endptr = t;
    return MTX_SUCCESS;
}

/**
 * `mtxfile_parse_data_matrix_coordinate_complex_double()' parses a
 * string containing a data line for a Matrix Market file in matrix
 * coordinate format with complex values.
 *
 * If `bytes_read' is not `NULL', then it is set to the number of
 * bytes consumed by the parser, if successful.  Similarly, if
 * `endptr' is not `NULL', then the address stored in `endptr' points
 * to the first character beyond the characters that were consumed
 * during parsing.
 */
int mtxfile_parse_data_matrix_coordinate_complex_double(
    struct mtxfile_matrix_coordinate_complex_double * data,
    int * bytes_read,
    const char ** endptr,
    const char * s,
    int num_rows,
    int num_columns)
{
    int err;
    const char * t = s;
    err = parse_int32(t, " ", &data->i, &t);
    if (err == EINVAL) {
        return MTX_ERR_INVALID_MTX_DATA;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    } else if (data->i <= 0 || data->i > num_rows) {
        return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    }
    err = parse_int32(t, " ", &data->j, &t);
    if (err == EINVAL) {
        return MTX_ERR_INVALID_MTX_DATA;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    }
    err = parse_double(t, " ", &data->a[0], &t);
    if (err == EINVAL) {
        return MTX_ERR_INVALID_MTX_DATA;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    } else if (data->j <= 0 || data->j > num_columns) {
        return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    }
    err = parse_double(t, "\n", &data->a[1], &t);
    if (err == EINVAL) {
        return MTX_ERR_INVALID_MTX_DATA;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    }
    if (bytes_read)
        (*bytes_read) += t-s;
    if (endptr)
        *endptr = t;
    return MTX_SUCCESS;
}

/**
 * `mtxfile_parse_data_matrix_coordinate_integer_single()' parses a
 * string containing a data line for a Matrix Market file in matrix
 * coordinate format with integer values.
 *
 * If `bytes_read' is not `NULL', then it is set to the number of
 * bytes consumed by the parser, if successful.  Similarly, if
 * `endptr' is not `NULL', then the address stored in `endptr' points
 * to the first character beyond the characters that were consumed
 * during parsing.
 */
int mtxfile_parse_data_matrix_coordinate_integer_single(
    struct mtxfile_matrix_coordinate_integer_single * data,
    int * bytes_read,
    const char ** endptr,
    const char * s,
    int num_rows,
    int num_columns)
{
    int err;
    const char * t = s;
    err = parse_int32(t, " ", &data->i, &t);
    if (err == EINVAL) {
        return MTX_ERR_INVALID_MTX_DATA;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    } else if (data->i <= 0 || data->i > num_rows) {
        return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    }
    err = parse_int32(t, " ", &data->j, &t);
    if (err == EINVAL) {
        return MTX_ERR_INVALID_MTX_DATA;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    } else if (data->j <= 0 || data->j > num_columns) {
        return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    }
    err = parse_int32(t, "\n", &data->a, &t);
    if (err == EINVAL) {
        return MTX_ERR_INVALID_MTX_DATA;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    }
    if (bytes_read)
        (*bytes_read) += t-s;
    if (endptr)
        *endptr = t;
    return MTX_SUCCESS;
}

/**
 * `mtxfile_parse_data_matrix_coordinate_integer_double()' parses a
 * string containing a data line for a Matrix Market file in matrix
 * coordinate format with integer values.
 *
 * If `bytes_read' is not `NULL', then it is set to the number of
 * bytes consumed by the parser, if successful.  Similarly, if
 * `endptr' is not `NULL', then the address stored in `endptr' points
 * to the first character beyond the characters that were consumed
 * during parsing.
 */
int mtxfile_parse_data_matrix_coordinate_integer_double(
    struct mtxfile_matrix_coordinate_integer_double * data,
    int * bytes_read,
    const char ** endptr,
    const char * s,
    int num_rows,
    int num_columns)
{
    int err;
    const char * t = s;
    err = parse_int32(t, " ", &data->i, &t);
    if (err == EINVAL) {
        return MTX_ERR_INVALID_MTX_DATA;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    } else if (data->i <= 0 || data->i > num_rows) {
        return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    }
    err = parse_int32(t, " ", &data->j, &t);
    if (err == EINVAL) {
        return MTX_ERR_INVALID_MTX_DATA;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    } else if (data->j <= 0 || data->j > num_columns) {
        return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    }
    err = parse_int64(t, "\n", &data->a, &t);
    if (err == EINVAL) {
        return MTX_ERR_INVALID_MTX_DATA;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    }
    if (bytes_read)
        (*bytes_read) += t-s;
    if (endptr)
        *endptr = t;
    return MTX_SUCCESS;
}

/**
 * `mtxfile_parse_data_matrix_coordinate_pattern()' parses a string
 * containing a data line for a Matrix Market file in matrix
 * coordinate format with pattern (boolean) values.
 *
 * If `bytes_read' is not `NULL', then it is set to the number of
 * bytes consumed by the parser, if successful.  Similarly, if
 * `endptr' is not `NULL', then the address stored in `endptr' points
 * to the first character beyond the characters that were consumed
 * during parsing.
 */
int mtxfile_parse_data_matrix_coordinate_pattern(
    struct mtxfile_matrix_coordinate_pattern * data,
    int * bytes_read,
    const char ** endptr,
    const char * s,
    int num_rows,
    int num_columns)
{
    int err;
    const char * t = s;
    err = parse_int32(t, " ", &data->i, &t);
    if (err == EINVAL) {
        return MTX_ERR_INVALID_MTX_DATA;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    } else if (data->i <= 0 || data->i > num_rows) {
        return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    }
    err = parse_int32(t, "\n", &data->j, &t);
    if (err == EINVAL) {
        return MTX_ERR_INVALID_MTX_DATA;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    } else if (data->j <= 0 || data->j > num_columns) {
        return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    }
    if (bytes_read)
        (*bytes_read) += t-s;
    if (endptr)
        *endptr = t;
    return MTX_SUCCESS;
}

/*
 * Vector coordinate formats
 */

/**
 * `mtxfile_parse_data_vector_coordinate_real_single()' parses a
 * string containing a data line for a Matrix Market file in vector
 * coordinate format with real values.
 *
 * If `bytes_read' is not `NULL', then it is set to the number of
 * bytes consumed by the parser, if successful.  Similarly, if
 * `endptr' is not `NULL', then the address stored in `endptr' points
 * to the first character beyond the characters that were consumed
 * during parsing.
 */
int mtxfile_parse_data_vector_coordinate_real_single(
    struct mtxfile_vector_coordinate_real_single * data,
    int * bytes_read,
    const char ** endptr,
    const char * s,
    int num_rows)
{
    int err;
    const char * t = s;
    err = parse_int32(t, " ", &data->i, &t);
    if (err == EINVAL) {
        return MTX_ERR_INVALID_MTX_DATA;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    } else if (data->i <= 0 || data->i > num_rows) {
        return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    }
    err = parse_float(t, "\n", &data->a, &t);
    if (err == EINVAL) {
        return MTX_ERR_INVALID_MTX_DATA;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    }
    if (bytes_read)
        (*bytes_read) += t-s;
    if (endptr)
        *endptr = t;
    return MTX_SUCCESS;
}

/**
 * `mtxfile_parse_data_vector_coordinate_real_double()' parses a
 * string containing a data line for a Matrix Market file in vector
 * coordinate format with real values.
 *
 * If `bytes_read' is not `NULL', then it is set to the number of
 * bytes consumed by the parser, if successful.  Similarly, if
 * `endptr' is not `NULL', then the address stored in `endptr' points
 * to the first character beyond the characters that were consumed
 * during parsing.
 */
int mtxfile_parse_data_vector_coordinate_real_double(
    struct mtxfile_vector_coordinate_real_double * data,
    int * bytes_read,
    const char ** endptr,
    const char * s,
    int num_rows)
{
    int err;
    const char * t = s;
    err = parse_int32(t, " ", &data->i, &t);
    if (err == EINVAL) {
        return MTX_ERR_INVALID_MTX_DATA;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    } else if (data->i <= 0 || data->i > num_rows) {
        return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    }
    err = parse_double(t, "\n", &data->a, &t);
    if (err == EINVAL) {
        return MTX_ERR_INVALID_MTX_DATA;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    }
    if (bytes_read)
        (*bytes_read) += t-s;
    if (endptr)
        *endptr = t;
    return MTX_SUCCESS;
}

/**
 * `mtxfile_parse_data_vector_coordinate_complex_single()' parses a
 * string containing a data line for a Matrix Market file in vector
 * coordinate format with complex values.
 *
 * If `bytes_read' is not `NULL', then it is set to the number of
 * bytes consumed by the parser, if successful.  Similarly, if
 * `endptr' is not `NULL', then the address stored in `endptr' points
 * to the first character beyond the characters that were consumed
 * during parsing.
 */
int mtxfile_parse_data_vector_coordinate_complex_single(
    struct mtxfile_vector_coordinate_complex_single * data,
    int * bytes_read,
    const char ** endptr,
    const char * s,
    int num_rows)
{
    int err;
    const char * t = s;
    err = parse_int32(t, " ", &data->i, &t);
    if (err == EINVAL) {
        return MTX_ERR_INVALID_MTX_DATA;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    } else if (data->i <= 0 || data->i > num_rows) {
        return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    }
    err = parse_float(t, " ", &data->a[0], &t);
    if (err == EINVAL) {
        return MTX_ERR_INVALID_MTX_DATA;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    }
    err = parse_float(t, "\n", &data->a[1], &t);
    if (err == EINVAL) {
        return MTX_ERR_INVALID_MTX_DATA;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    }
    if (bytes_read)
        (*bytes_read) += t-s;
    if (endptr)
        *endptr = t;
    return MTX_SUCCESS;
}

/**
 * `mtxfile_parse_data_vector_coordinate_complex_double()' parses a
 * string containing a data line for a Matrix Market file in vector
 * coordinate format with complex values.
 *
 * If `bytes_read' is not `NULL', then it is set to the number of
 * bytes consumed by the parser, if successful.  Similarly, if
 * `endptr' is not `NULL', then the address stored in `endptr' points
 * to the first character beyond the characters that were consumed
 * during parsing.
 */
int mtxfile_parse_data_vector_coordinate_complex_double(
    struct mtxfile_vector_coordinate_complex_double * data,
    int * bytes_read,
    const char ** endptr,
    const char * s,
    int num_rows)
{
    int err;
    const char * t = s;
    err = parse_int32(t, " ", &data->i, &t);
    if (err == EINVAL) {
        return MTX_ERR_INVALID_MTX_DATA;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    } else if (data->i <= 0 || data->i > num_rows) {
        return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    }
    err = parse_double(t, " ", &data->a[0], &t);
    if (err == EINVAL) {
        return MTX_ERR_INVALID_MTX_DATA;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    }
    err = parse_double(t, "\n", &data->a[1], &t);
    if (err == EINVAL) {
        return MTX_ERR_INVALID_MTX_DATA;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    }
    if (bytes_read)
        (*bytes_read) += t-s;
    if (endptr)
        *endptr = t;
    return MTX_SUCCESS;
}

/**
 * `mtxfile_parse_data_vector_coordinate_integer_single()' parses a
 * string containing a data line for a Matrix Market file in vector
 * coordinate format with integer values.
 *
 * If `bytes_read' is not `NULL', then it is set to the number of
 * bytes consumed by the parser, if successful.  Similarly, if
 * `endptr' is not `NULL', then the address stored in `endptr' points
 * to the first character beyond the characters that were consumed
 * during parsing.
 */
int mtxfile_parse_data_vector_coordinate_integer_single(
    struct mtxfile_vector_coordinate_integer_single * data,
    int * bytes_read,
    const char ** endptr,
    const char * s,
    int num_rows)
{
    int err;
    const char * t = s;
    err = parse_int32(t, " ", &data->i, &t);
    if (err == EINVAL) {
        return MTX_ERR_INVALID_MTX_DATA;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    } else if (data->i <= 0 || data->i > num_rows) {
        return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    }
    err = parse_int32(t, "\n", &data->a, &t);
    if (err == EINVAL) {
        return MTX_ERR_INVALID_MTX_DATA;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    }
    if (bytes_read)
        (*bytes_read) += t-s;
    if (endptr)
        *endptr = t;
    return MTX_SUCCESS;
}

/**
 * `mtxfile_parse_data_vector_coordinate_integer_double()' parses a
 * string containing a data line for a Matrix Market file in vector
 * coordinate format with integer values.
 *
 * If `bytes_read' is not `NULL', then it is set to the number of
 * bytes consumed by the parser, if successful.  Similarly, if
 * `endptr' is not `NULL', then the address stored in `endptr' points
 * to the first character beyond the characters that were consumed
 * during parsing.
 */
int mtxfile_parse_data_vector_coordinate_integer_double(
    struct mtxfile_vector_coordinate_integer_double * data,
    int * bytes_read,
    const char ** endptr,
    const char * s,
    int num_rows)
{
    int err;
    const char * t = s;
    err = parse_int32(t, " ", &data->i, &t);
    if (err == EINVAL) {
        return MTX_ERR_INVALID_MTX_DATA;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    } else if (data->i <= 0 || data->i > num_rows) {
        return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    }
    err = parse_int64(t, "\n", &data->a, &t);
    if (err == EINVAL) {
        return MTX_ERR_INVALID_MTX_DATA;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    }
    if (bytes_read)
        (*bytes_read) += t-s;
    if (endptr)
        *endptr = t;
    return MTX_SUCCESS;
}

/**
 * `mtxfile_parse_data_vector_coordinate_pattern()' parses a string
 * containing a data line for a Matrix Market file in vector
 * coordinate format with pattern (boolean) values.
 *
 * If `bytes_read' is not `NULL', then it is set to the number of
 * bytes consumed by the parser, if successful.  Similarly, if
 * `endptr' is not `NULL', then the address stored in `endptr' points
 * to the first character beyond the characters that were consumed
 * during parsing.
 */
int mtxfile_parse_data_vector_coordinate_pattern(
    struct mtxfile_vector_coordinate_pattern * data,
    int * bytes_read,
    const char ** endptr,
    const char * s,
    int num_rows)
{
    int err;
    const char * t = s;
    err = parse_int32(t, "\n", &data->i, &t);
    if (err == EINVAL) {
        return MTX_ERR_INVALID_MTX_DATA;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    } else if (data->i <= 0 || data->i > num_rows) {
        return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    }
    if (bytes_read)
        (*bytes_read) += t-s;
    if (endptr)
        *endptr = t;
    return MTX_SUCCESS;
}

/*
 * Memory management
 */

/**
 * `mtxfile_data_alloc()' allocates storage for a given number of data
 * lines for a given type of matrix or vector.
 */
int mtxfile_data_alloc(
    union mtxfile_data * data,
    enum mtxfile_object object,
    enum mtxfile_format format,
    enum mtxfile_field field,
    enum mtx_precision precision,
    size_t size)
{
    if (format == mtxfile_array) {
        if (field == mtxfile_real) {
            if (precision == mtx_single) {
                data->array_real_single =
                    malloc(size * sizeof(*data->array_real_single));
            } else if (precision == mtx_double) {
                data->array_real_double =
                    malloc(size * sizeof(*data->array_real_double));
            } else {
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_complex) {
            if (precision == mtx_single) {
                data->array_complex_single =
                    malloc(size * sizeof(*data->array_complex_single));
            } else if (precision == mtx_double) {
                data->array_complex_double =
                    malloc(size * sizeof(*data->array_complex_double));
            } else {
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_integer) {
            if (precision == mtx_single) {
                data->array_integer_single =
                    malloc(size * sizeof(*data->array_integer_single));
            } else if (precision == mtx_double) {
                data->array_integer_double =
                    malloc(size * sizeof(*data->array_integer_double));
            } else {
                return MTX_ERR_INVALID_PRECISION;
            }
        } else {
            return MTX_ERR_INVALID_MTX_FIELD;
        }

    } else if (format == mtxfile_coordinate) {
        if (object == mtxfile_matrix) {
            if (field == mtxfile_real) {
                if (precision == mtx_single) {
                    data->matrix_coordinate_real_single =
                        malloc(size * sizeof(*data->matrix_coordinate_real_single));
                } else if (precision == mtx_double) {
                    data->matrix_coordinate_real_double =
                        malloc(size * sizeof(*data->matrix_coordinate_real_double));
                } else {
                    return MTX_ERR_INVALID_PRECISION;
                }
            } else if (field == mtxfile_complex) {
                if (precision == mtx_single) {
                    data->matrix_coordinate_complex_single =
                        malloc(size * sizeof(*data->matrix_coordinate_complex_single));
                } else if (precision == mtx_double) {
                    data->matrix_coordinate_complex_double =
                        malloc(size * sizeof(*data->matrix_coordinate_complex_double));
                } else {
                    return MTX_ERR_INVALID_PRECISION;
                }
            } else if (field == mtxfile_integer) {
                if (precision == mtx_single) {
                    data->matrix_coordinate_integer_single =
                        malloc(size * sizeof(*data->matrix_coordinate_integer_single));
                } else if (precision == mtx_double) {
                    data->matrix_coordinate_integer_double =
                        malloc(size * sizeof(*data->matrix_coordinate_integer_double));
                } else {
                    return MTX_ERR_INVALID_PRECISION;
                }
            } else if (field == mtxfile_pattern) {
                data->matrix_coordinate_pattern =
                    malloc(size * sizeof(*data->matrix_coordinate_pattern));
            } else {
                return MTX_ERR_INVALID_MTX_FIELD;
            }

        } else if (object == mtxfile_vector) {
            if (field == mtxfile_real) {
                if (precision == mtx_single) {
                    data->vector_coordinate_real_single =
                        malloc(size * sizeof(*data->vector_coordinate_real_single));
                } else if (precision == mtx_double) {
                    data->vector_coordinate_real_double =
                        malloc(size * sizeof(*data->vector_coordinate_real_double));
                } else {
                    return MTX_ERR_INVALID_PRECISION;
                }
            } else if (field == mtxfile_complex) {
                if (precision == mtx_single) {
                    data->vector_coordinate_complex_single =
                        malloc(size * sizeof(*data->vector_coordinate_complex_single));
                } else if (precision == mtx_double) {
                    data->vector_coordinate_complex_double =
                        malloc(size * sizeof(*data->vector_coordinate_complex_double));
                } else {
                    return MTX_ERR_INVALID_PRECISION;
                }
            } else if (field == mtxfile_integer) {
                if (precision == mtx_single) {
                    data->vector_coordinate_integer_single =
                        malloc(size * sizeof(*data->vector_coordinate_integer_single));
                } else if (precision == mtx_double) {
                    data->vector_coordinate_integer_double =
                        malloc(size * sizeof(*data->vector_coordinate_integer_double));
                } else {
                    return MTX_ERR_INVALID_PRECISION;
                }
            } else if (field == mtxfile_pattern) {
                data->vector_coordinate_pattern =
                    malloc(size * sizeof(*data->vector_coordinate_pattern));
            } else {
                return MTX_ERR_INVALID_MTX_FIELD;
            }

        } else {
            return MTX_ERR_INVALID_MTX_OBJECT;
        }
    } else {
        return MTX_ERR_INVALID_MTX_FORMAT;
    }
    return MTX_SUCCESS;
}

/**
 * `mtxfile_data_free()' frees allocaed storage for data lines.
 */
int mtxfile_data_free(
    union mtxfile_data * data,
    enum mtxfile_object object,
    enum mtxfile_format format,
    enum mtxfile_field field,
    enum mtx_precision precision)
{
    if (format == mtxfile_array) {
        if (field == mtxfile_real) {
            if (precision == mtx_single) {
                free(data->array_real_single);
            } else if (precision == mtx_double) {
                free(data->array_real_double);
            } else {
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_complex) {
            if (precision == mtx_single) {
                free(data->array_complex_single);
            } else if (precision == mtx_double) {
                free(data->array_complex_double);
            } else {
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_integer) {
            if (precision == mtx_single) {
                free(data->array_integer_single);
            } else if (precision == mtx_double) {
                free(data->array_integer_double);
            } else {
                return MTX_ERR_INVALID_PRECISION;
            }
        } else {
            return MTX_ERR_INVALID_MTX_FIELD;
        }

    } else if (format == mtxfile_coordinate) {
        if (object == mtxfile_matrix) {
            if (field == mtxfile_real) {
                if (precision == mtx_single) {
                    free(data->matrix_coordinate_real_single);
                } else if (precision == mtx_double) {
                    free(data->matrix_coordinate_real_double);
                } else {
                    return MTX_ERR_INVALID_PRECISION;
                }
            } else if (field == mtxfile_complex) {
                if (precision == mtx_single) {
                    free(data->matrix_coordinate_complex_single);
                } else if (precision == mtx_double) {
                    free(data->matrix_coordinate_complex_double);
                } else {
                    return MTX_ERR_INVALID_PRECISION;
                }
            } else if (field == mtxfile_integer) {
                if (precision == mtx_single) {
                    free(data->matrix_coordinate_integer_single);
                } else if (precision == mtx_double) {
                    free(data->matrix_coordinate_integer_double);
                } else {
                    return MTX_ERR_INVALID_PRECISION;
                }
            } else if (field == mtxfile_pattern) {
                free(data->matrix_coordinate_pattern);
            } else {
                return MTX_ERR_INVALID_MTX_FIELD;
            }

        } else if (object == mtxfile_vector) {
            if (field == mtxfile_real) {
                if (precision == mtx_single) {
                    free(data->vector_coordinate_real_single);
                } else if (precision == mtx_double) {
                    free(data->vector_coordinate_real_double);
                } else {
                    return MTX_ERR_INVALID_PRECISION;
                }
            } else if (field == mtxfile_complex) {
                if (precision == mtx_single) {
                    free(data->vector_coordinate_complex_single);
                } else if (precision == mtx_double) {
                    free(data->vector_coordinate_complex_double);
                } else {
                    return MTX_ERR_INVALID_PRECISION;
                }
            } else if (field == mtxfile_integer) {
                if (precision == mtx_single) {
                    free(data->vector_coordinate_integer_single);
                } else if (precision == mtx_double) {
                    free(data->vector_coordinate_integer_double);
                } else {
                    return MTX_ERR_INVALID_PRECISION;
                }
            } else if (field == mtxfile_pattern) {
                free(data->vector_coordinate_pattern);
            } else {
                return MTX_ERR_INVALID_MTX_FIELD;
            }

        } else {
            return MTX_ERR_INVALID_MTX_OBJECT;
        }
    } else {
        return MTX_ERR_INVALID_MTX_FORMAT;
    }
    return MTX_SUCCESS;
}

/**
 * `mtxfile_data_copy()' copies data lines.
 */
int mtxfile_data_copy(
    union mtxfile_data * dst,
    const union mtxfile_data * src,
    enum mtxfile_object object,
    enum mtxfile_format format,
    enum mtxfile_field field,
    enum mtx_precision precision,
    size_t size)
{
    if (format == mtxfile_array) {
        if (field == mtxfile_real) {
            if (precision == mtx_single) {
                memcpy(dst->array_real_single,
                       src->array_real_single,
                       size * sizeof(*src->array_real_single));
            } else if (precision == mtx_double) {
                memcpy(dst->array_real_double,
                       src->array_real_double,
                       size * sizeof(*src->array_real_double));
            } else {
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_complex) {
            if (precision == mtx_single) {
                memcpy(dst->array_complex_single,
                       src->array_complex_single,
                       size * sizeof(*src->array_complex_single));
            } else if (precision == mtx_double) {
                memcpy(dst->array_complex_double,
                       src->array_complex_double,
                       size * sizeof(*src->array_complex_double));
            } else {
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_integer) {
            if (precision == mtx_single) {
                memcpy(dst->array_integer_single,
                       src->array_integer_single,
                       size * sizeof(*src->array_integer_single));
            } else if (precision == mtx_double) {
                memcpy(dst->array_integer_double,
                       src->array_integer_double,
                       size * sizeof(*src->array_integer_double));
            } else {
                return MTX_ERR_INVALID_PRECISION;
            }
        } else {
            return MTX_ERR_INVALID_MTX_FIELD;
        }

    } else if (format == mtxfile_coordinate) {
        if (object == mtxfile_matrix) {
            if (field == mtxfile_real) {
                if (precision == mtx_single) {
                    memcpy(dst->matrix_coordinate_real_single,
                           src->matrix_coordinate_real_single,
                           size * sizeof(*src->matrix_coordinate_real_single));
                } else if (precision == mtx_double) {
                    memcpy(dst->matrix_coordinate_real_double,
                           src->matrix_coordinate_real_double,
                           size * sizeof(*src->matrix_coordinate_real_double));
                } else {
                    return MTX_ERR_INVALID_PRECISION;
                }
            } else if (field == mtxfile_complex) {
                if (precision == mtx_single) {
                    memcpy(dst->matrix_coordinate_complex_single,
                           src->matrix_coordinate_complex_single,
                           size * sizeof(*src->matrix_coordinate_complex_single));
                } else if (precision == mtx_double) {
                    memcpy(dst->matrix_coordinate_complex_double,
                           src->matrix_coordinate_complex_double,
                           size * sizeof(*src->matrix_coordinate_complex_double));
                } else {
                    return MTX_ERR_INVALID_PRECISION;
                }
            } else if (field == mtxfile_integer) {
                if (precision == mtx_single) {
                    memcpy(dst->matrix_coordinate_integer_single,
                           src->matrix_coordinate_integer_single,
                           size * sizeof(*src->matrix_coordinate_integer_single));
                } else if (precision == mtx_double) {
                    memcpy(dst->matrix_coordinate_integer_double,
                           src->matrix_coordinate_integer_double,
                           size * sizeof(*src->matrix_coordinate_integer_double));
                } else {
                    return MTX_ERR_INVALID_PRECISION;
                }
            } else if (field == mtxfile_pattern) {
                memcpy(dst->matrix_coordinate_pattern,
                       src->matrix_coordinate_pattern,
                       size * sizeof(*src->matrix_coordinate_pattern));
            } else {
                return MTX_ERR_INVALID_MTX_FIELD;
            }

        } else if (object == mtxfile_vector) {
            if (field == mtxfile_real) {
                if (precision == mtx_single) {
                    memcpy(dst->vector_coordinate_real_single,
                           src->vector_coordinate_real_single,
                           size * sizeof(*src->vector_coordinate_real_single));
                } else if (precision == mtx_double) {
                    memcpy(dst->vector_coordinate_real_double,
                           src->vector_coordinate_real_double,
                           size * sizeof(*src->vector_coordinate_real_double));
                } else {
                    return MTX_ERR_INVALID_PRECISION;
                }
            } else if (field == mtxfile_complex) {
                if (precision == mtx_single) {
                    memcpy(dst->vector_coordinate_complex_single,
                           src->vector_coordinate_complex_single,
                           size * sizeof(*src->vector_coordinate_complex_single));
                } else if (precision == mtx_double) {
                    memcpy(dst->vector_coordinate_complex_double,
                           src->vector_coordinate_complex_double,
                           size * sizeof(*src->vector_coordinate_complex_double));
                } else {
                    return MTX_ERR_INVALID_PRECISION;
                }
            } else if (field == mtxfile_integer) {
                if (precision == mtx_single) {
                    memcpy(dst->vector_coordinate_integer_single,
                           src->vector_coordinate_integer_single,
                           size * sizeof(*src->vector_coordinate_integer_single));
                } else if (precision == mtx_double) {
                    memcpy(dst->vector_coordinate_integer_double,
                           src->vector_coordinate_integer_double,
                           size * sizeof(*src->vector_coordinate_integer_double));
                } else {
                    return MTX_ERR_INVALID_PRECISION;
                }
            } else if (field == mtxfile_pattern) {
                memcpy(dst->vector_coordinate_pattern,
                       src->vector_coordinate_pattern,
                       size * sizeof(src->vector_coordinate_pattern));
            } else {
                return MTX_ERR_INVALID_MTX_FIELD;
            }
        } else {
            return MTX_ERR_INVALID_MTX_OBJECT;
        }
    } else {
        return MTX_ERR_INVALID_MTX_FORMAT;
    }
    return MTX_SUCCESS;
}

/*
 * I/O functions
 */

static int mtxfile_parse_data(
    union mtxfile_data * data,
    int * bytes_read,
    const char ** endptr,
    const char * s,
    enum mtxfile_object object,
    enum mtxfile_format format,
    enum mtxfile_field field,
    enum mtx_precision precision,
    int num_rows,
    int num_columns,
    size_t i)
{
    if (format == mtxfile_array) {
        if (field == mtxfile_real) {
            if (precision == mtx_single) {
                return mtxfile_parse_data_array_real_single(
                    &data->array_real_single[i],
                    bytes_read, endptr, s);
            } else if (precision == mtx_double) {
                return mtxfile_parse_data_array_real_double(
                    &data->array_real_double[i],
                    bytes_read, endptr, s);
            } else {
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_complex) {
            if (precision == mtx_single) {
                return mtxfile_parse_data_array_complex_single(
                    &data->array_complex_single[i],
                    bytes_read, endptr, s);
            } else if (precision == mtx_double) {
                return mtxfile_parse_data_array_complex_double(
                    &data->array_complex_double[i],
                    bytes_read, endptr, s);
            } else {
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_integer) {
            if (precision == mtx_single) {
                return mtxfile_parse_data_array_integer_single(
                    &data->array_integer_single[i],
                    bytes_read, endptr, s);
            } else if (precision == mtx_double) {
                return mtxfile_parse_data_array_integer_double(
                    &data->array_integer_double[i],
                    bytes_read, endptr, s);
            } else {
                return MTX_ERR_INVALID_PRECISION;
            }
        } else {
            return MTX_ERR_INVALID_MTX_FIELD;
        }

    } else if (format == mtxfile_coordinate) {
        if (object == mtxfile_matrix) {
            if (field == mtxfile_real) {
                if (precision == mtx_single) {
                    return mtxfile_parse_data_matrix_coordinate_real_single(
                        &data->matrix_coordinate_real_single[i],
                        bytes_read, endptr, s, num_rows, num_columns);
                } else if (precision == mtx_double) {
                    return mtxfile_parse_data_matrix_coordinate_real_double(
                        &data->matrix_coordinate_real_double[i],
                        bytes_read, endptr, s, num_rows, num_columns);
                } else {
                    return MTX_ERR_INVALID_PRECISION;
                }
            } else if (field == mtxfile_complex) {
                if (precision == mtx_single) {
                    return mtxfile_parse_data_matrix_coordinate_complex_single(
                        &data->matrix_coordinate_complex_single[i],
                        bytes_read, endptr, s, num_rows, num_columns);
                } else if (precision == mtx_double) {
                    return mtxfile_parse_data_matrix_coordinate_complex_double(
                        &data->matrix_coordinate_complex_double[i],
                        bytes_read, endptr, s, num_rows, num_columns);
                } else {
                    return MTX_ERR_INVALID_PRECISION;
                }
            } else if (field == mtxfile_integer) {
                if (precision == mtx_single) {
                    return mtxfile_parse_data_matrix_coordinate_integer_single(
                        &data->matrix_coordinate_integer_single[i],
                        bytes_read, endptr, s, num_rows, num_columns);
                } else if (precision == mtx_double) {
                    return mtxfile_parse_data_matrix_coordinate_integer_double(
                        &data->matrix_coordinate_integer_double[i],
                        bytes_read, endptr, s, num_rows, num_columns);
                } else {
                    return MTX_ERR_INVALID_PRECISION;
                }
            } else if (field == mtxfile_pattern) {
                return mtxfile_parse_data_matrix_coordinate_pattern(
                    &data->matrix_coordinate_pattern[i],
                    bytes_read, endptr, s, num_rows, num_columns);
            } else {
                return MTX_ERR_INVALID_MTX_FIELD;
            }

        } else if (object == mtxfile_vector) {
            if (field == mtxfile_real) {
                if (precision == mtx_single) {
                    return mtxfile_parse_data_vector_coordinate_real_single(
                        &data->vector_coordinate_real_single[i],
                        bytes_read, endptr, s, num_rows);
                } else if (precision == mtx_double) {
                    return mtxfile_parse_data_vector_coordinate_real_double(
                        &data->vector_coordinate_real_double[i],
                        bytes_read, endptr, s, num_rows);
                } else {
                    return MTX_ERR_INVALID_PRECISION;
                }
            } else if (field == mtxfile_complex) {
                if (precision == mtx_single) {
                    return mtxfile_parse_data_vector_coordinate_complex_single(
                        &data->vector_coordinate_complex_single[i],
                        bytes_read, endptr, s, num_rows);
                } else if (precision == mtx_double) {
                    return mtxfile_parse_data_vector_coordinate_complex_double(
                        &data->vector_coordinate_complex_double[i],
                        bytes_read, endptr, s, num_rows);
                } else {
                    return MTX_ERR_INVALID_PRECISION;
                }
            } else if (field == mtxfile_integer) {
                if (precision == mtx_single) {
                    return mtxfile_parse_data_vector_coordinate_integer_single(
                        &data->vector_coordinate_integer_single[i],
                        bytes_read, endptr, s, num_rows);
                } else if (precision == mtx_double) {
                    return mtxfile_parse_data_vector_coordinate_integer_double(
                        &data->vector_coordinate_integer_double[i],
                        bytes_read, endptr, s, num_rows);
                } else {
                    return MTX_ERR_INVALID_PRECISION;
                }
            } else if (field == mtxfile_pattern) {
                return mtxfile_parse_data_vector_coordinate_pattern(
                    &data->vector_coordinate_pattern[i],
                    bytes_read, endptr, s, num_rows);
            } else {
                return MTX_ERR_INVALID_MTX_FIELD;
            }

        } else {
            return MTX_ERR_INVALID_MTX_OBJECT;
        }
    } else {
        return MTX_ERR_INVALID_MTX_FORMAT;
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
 * `mtxfile_fread_data()` reads Matrix Market data lines from a
 * stream.
 *
 * Storage for the corresponding array of the `data' union, according
 * to the given `object', `format', `field' and `precision' variables,
 * must already be allocated with enough storage to hold at least
 * `size' elements.
 *
 * At most `size' lines are read from the stream.
 *
 * If an error code is returned, then `lines_read' and `bytes_read'
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the Matrix Market file.
 */
int mtxfile_fread_data(
    union mtxfile_data * data,
    FILE * f,
    int * lines_read,
    int * bytes_read,
    size_t line_max,
    char * linebuf,
    enum mtxfile_object object,
    enum mtxfile_format format,
    enum mtxfile_field field,
    enum mtx_precision precision,
    int num_rows,
    int num_columns,
    size_t size)
{
    int err;
    bool free_linebuf = !linebuf;
    if (!linebuf) {
        line_max = sysconf(_SC_LINE_MAX);
        linebuf = malloc(line_max+1);
        if (!linebuf)
            return MTX_ERR_ERRNO;
    }

    for (size_t i = 0; i < size; i++) {
        err = freadline(linebuf, line_max, f);
        if (err) {
            if (free_linebuf)
                free(linebuf);
            return err;
        }

        err = mtxfile_parse_data(
            data, bytes_read, NULL, linebuf,
            object, format, field, precision,
            num_rows, num_columns, i);
        if (err) {
            if (free_linebuf)
                free(linebuf);
            return err;
        }
        if (lines_read)
            (*lines_read)++;
    }

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
 * `mtxfile_gzread_data()' reads Matrix Market data lines from a
 * gzip-compressed stream.
 *
 * Storage for the corresponding array of the `data' union, according
 * to the given `object', `format', `field' and `precision' variables,
 * must already be allocated with enough storage to hold at least
 * `size' elements.
 *
 * At most `size' lines are read from the stream.
 *
 * If an error code is returned, then `lines_read' and `bytes_read'
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the Matrix Market file.
 */
int mtxfile_gzread_data(
    union mtxfile_data * data,
    gzFile f,
    int * lines_read,
    int * bytes_read,
    size_t line_max,
    char * linebuf,
    enum mtxfile_object object,
    enum mtxfile_format format,
    enum mtxfile_field field,
    enum mtx_precision precision,
    int num_rows,
    int num_columns,
    size_t size)
{
    int err;
    bool free_linebuf = !linebuf;
    if (!linebuf) {
        line_max = sysconf(_SC_LINE_MAX);
        linebuf = malloc(line_max+1);
        if (!linebuf)
            return MTX_ERR_ERRNO;
    }

    for (size_t i = 0; i < size; i++) {
        err = gzreadline(linebuf, line_max, f);
        if (err) {
            if (free_linebuf)
                free(linebuf);
            return err;
        }

        err = mtxfile_parse_data(
            data, bytes_read, NULL, linebuf,
            object, format, field, precision,
            num_rows, num_columns, i);
        if (err) {
            if (free_linebuf)
                free(linebuf);
            return err;
        }
        if (lines_read)
            (*lines_read)++;
    }

    if (free_linebuf)
        free(linebuf);
    return MTX_SUCCESS;
}
#endif
