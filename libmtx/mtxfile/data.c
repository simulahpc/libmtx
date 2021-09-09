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
#include <libmtx/util/partition.h>

#ifdef LIBMTX_HAVE_MPI
#include <mpi.h>
#endif

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
 * `mtxfile_data_size_per_element()' calculates the size of each
 * element in an array of Matrix Market data corresponding to the
 * given `object', `format', `field' and `precision'.
 */
int mtxfile_data_size_per_element(
    size_t * size,
    enum mtxfile_object object,
    enum mtxfile_format format,
    enum mtxfile_field field,
    enum mtx_precision precision)
{
    union mtxfile_data data;
    if (format == mtxfile_array) {
        if (field == mtxfile_real) {
            if (precision == mtx_single) {
                *size = sizeof(*data.array_real_single);
            } else if (precision == mtx_double) {
                *size = sizeof(*data.array_real_double);
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (field == mtxfile_complex) {
            if (precision == mtx_single) {
                *size = sizeof(*data.array_complex_single);
            } else if (precision == mtx_double) {
                *size = sizeof(*data.array_complex_double);
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (field == mtxfile_integer) {
            if (precision == mtx_single) {
                *size = sizeof(*data.array_integer_single);
            } else if (precision == mtx_double) {
                *size = sizeof(*data.array_integer_double);
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else { return MTX_ERR_INVALID_MTX_FIELD; }
    } else if (format == mtxfile_coordinate) {
        if (object == mtxfile_matrix) {
            if (field == mtxfile_real) {
                if (precision == mtx_single) {
                    *size = sizeof(*data.matrix_coordinate_real_single);
                } else if (precision == mtx_double) {
                    *size = sizeof(*data.matrix_coordinate_real_double);
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_complex) {
                if (precision == mtx_single) {
                    *size = sizeof(*data.matrix_coordinate_complex_single);
                } else if (precision == mtx_double) {
                    *size = sizeof(*data.matrix_coordinate_complex_double);
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_integer) {
                if (precision == mtx_single) {
                    *size = sizeof(*data.matrix_coordinate_integer_single);
                } else if (precision == mtx_double) {
                    *size = sizeof(*data.matrix_coordinate_integer_double);
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_pattern) {
                *size = sizeof(*data.matrix_coordinate_pattern);
            } else { return MTX_ERR_INVALID_MTX_FIELD; }
        } else if (object == mtxfile_vector) {
            if (field == mtxfile_real) {
                if (precision == mtx_single) {
                    *size = sizeof(*data.vector_coordinate_real_single);
                } else if (precision == mtx_double) {
                    *size = sizeof(*data.vector_coordinate_real_double);
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_complex) {
                if (precision == mtx_single) {
                    *size = sizeof(*data.vector_coordinate_complex_single);
                } else if (precision == mtx_double) {
                    *size = sizeof(*data.vector_coordinate_complex_double);
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_integer) {
                if (precision == mtx_single) {
                    *size = sizeof(*data.vector_coordinate_integer_single);
                } else if (precision == mtx_double) {
                    *size = sizeof(*data.vector_coordinate_integer_double);
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_pattern) {
                *size = sizeof(*data.vector_coordinate_pattern);
            } else { return MTX_ERR_INVALID_MTX_FIELD; }
        } else { return MTX_ERR_INVALID_MTX_OBJECT; }
    } else { return MTX_ERR_INVALID_MTX_FORMAT; }
    return MTX_SUCCESS;
}

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
    int64_t * bytes_read,
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
    int64_t * bytes_read,
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
    int64_t * bytes_read,
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
    int64_t * bytes_read,
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
    int64_t * bytes_read,
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
    int64_t * bytes_read,
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
    int64_t * bytes_read,
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
    int64_t * bytes_read,
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
    int64_t * bytes_read,
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
    int64_t * bytes_read,
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
    int64_t * bytes_read,
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
    int64_t * bytes_read,
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
    int64_t * bytes_read,
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
    int64_t * bytes_read,
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
    int64_t * bytes_read,
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
    int64_t * bytes_read,
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
    int64_t * bytes_read,
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
    int64_t * bytes_read,
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
    int64_t * bytes_read,
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
    int64_t * bytes_read,
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
    size_t size,
    size_t dst_offset,
    size_t src_offset)
{
    if (format == mtxfile_array) {
        if (field == mtxfile_real) {
            if (precision == mtx_single) {
                memcpy(&dst->array_real_single[dst_offset],
                       &src->array_real_single[src_offset],
                       size * sizeof(*src->array_real_single));
            } else if (precision == mtx_double) {
                memcpy(&dst->array_real_double[dst_offset],
                       &src->array_real_double[src_offset],
                       size * sizeof(*src->array_real_double));
            } else {
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_complex) {
            if (precision == mtx_single) {
                memcpy(&dst->array_complex_single[dst_offset],
                       &src->array_complex_single[src_offset],
                       size * sizeof(*src->array_complex_single));
            } else if (precision == mtx_double) {
                memcpy(&dst->array_complex_double[dst_offset],
                       &src->array_complex_double[src_offset],
                       size * sizeof(*src->array_complex_double));
            } else {
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_integer) {
            if (precision == mtx_single) {
                memcpy(&dst->array_integer_single[dst_offset],
                       &src->array_integer_single[src_offset],
                       size * sizeof(*src->array_integer_single));
            } else if (precision == mtx_double) {
                memcpy(&dst->array_integer_double[dst_offset],
                       &src->array_integer_double[src_offset],
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
                    memcpy(&dst->matrix_coordinate_real_single[dst_offset],
                           &src->matrix_coordinate_real_single[src_offset],
                           size * sizeof(*src->matrix_coordinate_real_single));
                } else if (precision == mtx_double) {
                    memcpy(&dst->matrix_coordinate_real_double[dst_offset],
                           &src->matrix_coordinate_real_double[src_offset],
                           size * sizeof(*src->matrix_coordinate_real_double));
                } else {
                    return MTX_ERR_INVALID_PRECISION;
                }
            } else if (field == mtxfile_complex) {
                if (precision == mtx_single) {
                    memcpy(&dst->matrix_coordinate_complex_single[dst_offset],
                           &src->matrix_coordinate_complex_single[src_offset],
                           size * sizeof(*src->matrix_coordinate_complex_single));
                } else if (precision == mtx_double) {
                    memcpy(&dst->matrix_coordinate_complex_double[dst_offset],
                           &src->matrix_coordinate_complex_double[src_offset],
                           size * sizeof(*src->matrix_coordinate_complex_double));
                } else {
                    return MTX_ERR_INVALID_PRECISION;
                }
            } else if (field == mtxfile_integer) {
                if (precision == mtx_single) {
                    memcpy(&dst->matrix_coordinate_integer_single[dst_offset],
                           &src->matrix_coordinate_integer_single[src_offset],
                           size * sizeof(*src->matrix_coordinate_integer_single));
                } else if (precision == mtx_double) {
                    memcpy(&dst->matrix_coordinate_integer_double[dst_offset],
                           &src->matrix_coordinate_integer_double[src_offset],
                           size * sizeof(*src->matrix_coordinate_integer_double));
                } else {
                    return MTX_ERR_INVALID_PRECISION;
                }
            } else if (field == mtxfile_pattern) {
                memcpy(&dst->matrix_coordinate_pattern[dst_offset],
                       &src->matrix_coordinate_pattern[src_offset],
                       size * sizeof(*src->matrix_coordinate_pattern));
            } else {
                return MTX_ERR_INVALID_MTX_FIELD;
            }

        } else if (object == mtxfile_vector) {
            if (field == mtxfile_real) {
                if (precision == mtx_single) {
                    memcpy(&dst->vector_coordinate_real_single[dst_offset],
                           &src->vector_coordinate_real_single[src_offset],
                           size * sizeof(*src->vector_coordinate_real_single));
                } else if (precision == mtx_double) {
                    memcpy(&dst->vector_coordinate_real_double[dst_offset],
                           &src->vector_coordinate_real_double[src_offset],
                           size * sizeof(*src->vector_coordinate_real_double));
                } else {
                    return MTX_ERR_INVALID_PRECISION;
                }
            } else if (field == mtxfile_complex) {
                if (precision == mtx_single) {
                    memcpy(&dst->vector_coordinate_complex_single[dst_offset],
                           &src->vector_coordinate_complex_single[src_offset],
                           size * sizeof(*src->vector_coordinate_complex_single));
                } else if (precision == mtx_double) {
                    memcpy(&dst->vector_coordinate_complex_double[dst_offset],
                           &src->vector_coordinate_complex_double[src_offset],
                           size * sizeof(*src->vector_coordinate_complex_double));
                } else {
                    return MTX_ERR_INVALID_PRECISION;
                }
            } else if (field == mtxfile_integer) {
                if (precision == mtx_single) {
                    memcpy(&dst->vector_coordinate_integer_single[dst_offset],
                           &src->vector_coordinate_integer_single[src_offset],
                           size * sizeof(*src->vector_coordinate_integer_single));
                } else if (precision == mtx_double) {
                    memcpy(&dst->vector_coordinate_integer_double[dst_offset],
                           &src->vector_coordinate_integer_double[src_offset],
                           size * sizeof(*src->vector_coordinate_integer_double));
                } else {
                    return MTX_ERR_INVALID_PRECISION;
                }
            } else if (field == mtxfile_pattern) {
                memcpy(&dst->vector_coordinate_pattern[dst_offset],
                       &src->vector_coordinate_pattern[src_offset],
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
    int64_t * bytes_read,
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
    int64_t * bytes_read,
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
    int64_t * bytes_read,
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

/*
 * Partitioning
 */

/**
 * `mtxfile_data_partition_rows()' partitions data lines according to
 * a given row partitioning.
 *
 * The array `row_parts' must contain enough storage for an array of
 * `size' values of type `int'.  If successful, the `k'-th value of
 * `row_parts' is equal to the part to which the `k'-th data line
 * belongs.
 */
int mtxfile_data_partition_rows(
    const union mtxfile_data * data,
    enum mtxfile_object object,
    enum mtxfile_format format,
    enum mtxfile_field field,
    enum mtx_precision precision,
    int num_rows,
    int num_columns,
    size_t size,
    size_t offset,
    const struct mtx_partition * row_partition,
    int * row_parts)
{
    int err;
    if (format == mtxfile_array) {
        if (object == mtxfile_matrix) {
            for (int64_t l = 0; l < size; l++) {
                int64_t k = offset + l;
                int i = k / num_columns;
                err = mtx_partition_part(row_partition, &row_parts[l], i);
                if (err)
                    return err;
            }
        } else if (object == mtxfile_vector) {
            for (int64_t l = 0; l < size; l++) {
                int64_t k = offset + l;
                int i = k;
                err = mtx_partition_part(row_partition, &row_parts[l], i);
                if (err)
                    return err;
            }
        } else {
            return MTX_ERR_INVALID_MTX_OBJECT;
        }
    } else if (format == mtxfile_coordinate) {
        if (object == mtxfile_matrix) {
            if (field == mtxfile_real) {
                if (precision == mtx_single) {
                    for (int64_t l = 0; l < size; l++) {
                        int64_t k = offset + l;
                        int i = data->matrix_coordinate_real_single[k].i-1;
                        err = mtx_partition_part(row_partition, &row_parts[l], i);
                        if (err)
                            return err;
                    }
                } else if (precision == mtx_double) {
                    for (int64_t l = 0; l < size; l++) {
                        int64_t k = offset + l;
                        int i = data->matrix_coordinate_real_double[k].i-1;
                        err = mtx_partition_part(row_partition, &row_parts[l], i);
                        if (err)
                            return err;
                    }
                } else {
                    return MTX_ERR_INVALID_PRECISION;
                }
            } else if (field == mtxfile_complex) {
                if (precision == mtx_single) {
                    for (int64_t l = 0; l < size; l++) {
                        int64_t k = offset + l;
                        int i = data->matrix_coordinate_complex_single[k].i-1;
                        err = mtx_partition_part(row_partition, &row_parts[l], i);
                        if (err)
                            return err;
                    }
                } else if (precision == mtx_double) {
                    for (int64_t l = 0; l < size; l++) {
                        int64_t k = offset + l;
                        int i = data->matrix_coordinate_complex_double[k].i-1;
                        err = mtx_partition_part(row_partition, &row_parts[l], i);
                        if (err)
                            return err;
                    }
                } else {
                    return MTX_ERR_INVALID_PRECISION;
                }
            } else if (field == mtxfile_integer) {
                if (precision == mtx_single) {
                    for (int64_t l = 0; l < size; l++) {
                        int64_t k = offset + l;
                        int i = data->matrix_coordinate_integer_single[k].i-1;
                        err = mtx_partition_part(row_partition, &row_parts[l], i);
                        if (err)
                            return err;
                    }
                } else if (precision == mtx_double) {
                    for (int64_t l = 0; l < size; l++) {
                        int64_t k = offset + l;
                        int i = data->matrix_coordinate_integer_double[k].i-1;
                        err = mtx_partition_part(row_partition, &row_parts[l], i);
                        if (err)
                            return err;
                    }
                } else {
                    return MTX_ERR_INVALID_PRECISION;
                }
            } else if (field == mtxfile_pattern) {
                for (int64_t l = 0; l < size; l++) {
                    int64_t k = offset + l;
                    int i = data->matrix_coordinate_pattern[k].i-1;
                    err = mtx_partition_part(row_partition, &row_parts[l], i);
                    if (err)
                        return err;
                }
            } else {
                return MTX_ERR_INVALID_MTX_FIELD;
            }
        } else if (object == mtxfile_vector) {
            if (field == mtxfile_real) {
                if (precision == mtx_single) {
                    for (int64_t l = 0; l < size; l++) {
                        int64_t k = offset + l;
                        int i = data->matrix_coordinate_real_single[k].i-1;
                        err = mtx_partition_part(row_partition, &row_parts[l], i);
                        if (err)
                            return err;
                    }
                } else if (precision == mtx_double) {
                    for (int64_t l = 0; l < size; l++) {
                        int64_t k = offset + l;
                        int i = data->matrix_coordinate_real_double[k].i-1;
                        err = mtx_partition_part(row_partition, &row_parts[l], i);
                        if (err)
                            return err;
                    }
                } else {
                    return MTX_ERR_INVALID_PRECISION;
                }
            } else if (field == mtxfile_complex) {
                if (precision == mtx_single) {
                    for (int64_t l = 0; l < size; l++) {
                        int64_t k = offset + l;
                        int i = data->matrix_coordinate_complex_single[k].i-1;
                        err = mtx_partition_part(row_partition, &row_parts[l], i);
                        if (err)
                            return err;
                    }
                } else if (precision == mtx_double) {
                    for (int64_t l = 0; l < size; l++) {
                        int64_t k = offset + l;
                        int i = data->matrix_coordinate_complex_double[k].i-1;
                        err = mtx_partition_part(row_partition, &row_parts[l], i);
                        if (err)
                            return err;
                    }
                } else {
                    return MTX_ERR_INVALID_PRECISION;
                }
            } else if (field == mtxfile_integer) {
                if (precision == mtx_single) {
                    for (int64_t l = 0; l < size; l++) {
                        int64_t k = offset + l;
                        int i = data->matrix_coordinate_integer_single[k].i-1;
                        err = mtx_partition_part(row_partition, &row_parts[l], i);
                        if (err)
                            return err;
                    }
                } else if (precision == mtx_double) {
                    for (int64_t l = 0; l < size; l++) {
                        int64_t k = offset + l;
                        int i = data->matrix_coordinate_integer_double[k].i-1;
                        err = mtx_partition_part(row_partition, &row_parts[l], i);
                        if (err)
                            return err;
                    }
                } else {
                    return MTX_ERR_INVALID_PRECISION;
                }
            } else if (field == mtxfile_pattern) {
                for (int64_t l = 0; l < size; l++) {
                    int64_t k = offset + l;
                    int i = data->matrix_coordinate_pattern[k].i-1;
                    err = mtx_partition_part(row_partition, &row_parts[l], i);
                    if (err)
                        return err;
                }
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
 * Sorting
 */

/**
 * `mtxfile_data_sort_by_key()' sorts data lines according to the
 * given keys using a stable, in-place insertion sort algorihtm.
 */
int mtxfile_data_sort_by_key(
    union mtxfile_data * data,
    enum mtxfile_object object,
    enum mtxfile_format format,
    enum mtxfile_field field,
    enum mtx_precision precision,
    size_t size,
    size_t offset,
    int * keys)
{
    int err;

    /* Allocate storage for a single data line. */
    union mtxfile_data x;
    err = mtxfile_data_alloc(&x, object, format, field, precision, 1);
    if (err)
        return err;

    for (int64_t i = offset; i < offset+size; i++) {
        int xkey = keys[i];
        err = mtxfile_data_copy(
            &x, data, object, format, field, precision, 1, 0, i);
        if (err) {
            mtxfile_data_free(&x, object, format, field, precision);
            return err;
        }

        int64_t j = i-1;
        while (j >= 0 && keys[j] > xkey) {
            keys[j+1] = keys[j];
            err = mtxfile_data_copy(
                data, data, object, format, field, precision, 1, j+1, j);
            if (err) {
                mtxfile_data_free(&x, object, format, field, precision);
                return err;
            }
            j--;
        }

        keys[j+1] = xkey;
        err = mtxfile_data_copy(
            data, &x, object, format, field, precision, 1, j+1, 0);
        if (err) {
            mtxfile_data_free(&x, object, format, field, precision);
            return err;
        }
    }

    mtxfile_data_free(&x, object, format, field, precision);
    return MTX_SUCCESS;
}

/*
 * MPI functions
 */

#ifdef LIBMTX_HAVE_MPI
static int mtxfile_data_send_array(
    const union mtxfile_data * data,
    enum mtxfile_field field,
    enum mtx_precision precision,
    size_t size,
    size_t offset,
    int dest,
    int tag,
    MPI_Comm comm,
    int * mpierrcode)
{
    if (field == mtxfile_real) {
        if (precision == mtx_single) {
            *mpierrcode = MPI_Send(
                &data->array_real_single[offset],
                size, MPI_FLOAT, dest, tag, comm);
            if (*mpierrcode)
                return MTX_ERR_MPI;
        } else if (precision == mtx_double) {
            *mpierrcode = MPI_Send(
                &data->array_real_double[offset],
                size, MPI_DOUBLE, dest, tag, comm);
            if (*mpierrcode)
                return MTX_ERR_MPI;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (field == mtxfile_complex) {
        if (precision == mtx_single) {
            *mpierrcode = MPI_Send(
                &data->array_complex_single[offset],
                2*size, MPI_FLOAT, dest, tag, comm);
            if (*mpierrcode)
                return MTX_ERR_MPI;
        } else if (precision == mtx_double) {
            *mpierrcode = MPI_Send(
                &data->array_complex_double[offset],
                2*size, MPI_DOUBLE, dest, tag, comm);
            if (*mpierrcode)
                return MTX_ERR_MPI;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (field == mtxfile_integer) {
        if (precision == mtx_single) {
            *mpierrcode = MPI_Send(
                &data->array_integer_single[offset],
                size, MPI_INT32_T, dest, tag, comm);
            if (*mpierrcode)
                return MTX_ERR_MPI;
        } else if (precision == mtx_double) {
            *mpierrcode = MPI_Send(
                &data->array_integer_double[offset],
                size, MPI_INT64_T, dest, tag, comm);
            if (*mpierrcode)
                return MTX_ERR_MPI;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else {
        return MTX_ERR_INVALID_MTX_FIELD;
    }
    return MTX_SUCCESS;
}

/**
 * `mtxfile_coordinate_datatype()' creates a custom MPI data type for
 * sending or receiving data in coordinate format.
 *
 * The user is responsible for calling `MPI_Type_free()' on the
 * returned datatype.
 */
static int mtxfile_coordinate_datatype(
    enum mtxfile_object object,
    enum mtxfile_field field,
    enum mtx_precision precision,
    MPI_Datatype * datatype,
    int * mpierrcode)
{
    int num_elements;
    int block_lengths[3];
    MPI_Datatype element_types[3];
    MPI_Aint element_offsets[3];
    if (object == mtxfile_matrix) {
        if (field == mtxfile_real) {
            if (precision == mtx_single) {
                num_elements = 3;
                element_types[0] = MPI_INT;
                block_lengths[0] = 1;
                element_offsets[0] =
                    offsetof(struct mtxfile_matrix_coordinate_real_single, i);
                element_types[1] = MPI_INT;
                block_lengths[1] = 1;
                element_offsets[1] =
                    offsetof(struct mtxfile_matrix_coordinate_real_single, j);
                element_types[2] = MPI_FLOAT;
                block_lengths[2] = 1;
                element_offsets[2] =
                    offsetof(struct mtxfile_matrix_coordinate_real_single, a);
            } else if (precision == mtx_double) {
                num_elements = 3;
                element_types[0] = MPI_INT;
                block_lengths[0] = 1;
                element_offsets[0] =
                    offsetof(struct mtxfile_matrix_coordinate_real_double, i);
                element_types[1] = MPI_INT;
                block_lengths[1] = 1;
                element_offsets[1] =
                    offsetof(struct mtxfile_matrix_coordinate_real_double, j);
                element_types[2] = MPI_DOUBLE;
                block_lengths[2] = 1;
                element_offsets[2] =
                    offsetof(struct mtxfile_matrix_coordinate_real_double, a);
            } else {
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_complex) {
            if (precision == mtx_single) {
                num_elements = 3;
                element_types[0] = MPI_INT;
                block_lengths[0] = 1;
                element_offsets[0] =
                    offsetof(struct mtxfile_matrix_coordinate_complex_single, i);
                element_types[1] = MPI_INT;
                block_lengths[1] = 1;
                element_offsets[1] =
                    offsetof(struct mtxfile_matrix_coordinate_complex_single, j);
                element_types[2] = MPI_FLOAT;
                block_lengths[2] = 2;
                element_offsets[2] =
                    offsetof(struct mtxfile_matrix_coordinate_complex_single, a);
            } else if (precision == mtx_double) {
                num_elements = 3;
                element_types[0] = MPI_INT;
                block_lengths[0] = 1;
                element_offsets[0] =
                    offsetof(struct mtxfile_matrix_coordinate_complex_double, i);
                element_types[1] = MPI_INT;
                block_lengths[1] = 1;
                element_offsets[1] =
                    offsetof(struct mtxfile_matrix_coordinate_complex_double, j);
                element_types[2] = MPI_DOUBLE;
                block_lengths[2] = 2;
                element_offsets[2] =
                    offsetof(struct mtxfile_matrix_coordinate_complex_double, a);
            } else {
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_integer) {
            if (precision == mtx_single) {
                num_elements = 3;
                element_types[0] = MPI_INT;
                block_lengths[0] = 1;
                element_offsets[0] =
                    offsetof(struct mtxfile_matrix_coordinate_integer_single, i);
                element_types[1] = MPI_INT;
                block_lengths[1] = 1;
                element_offsets[1] =
                    offsetof(struct mtxfile_matrix_coordinate_integer_single, j);
                element_types[2] = MPI_INT32_T;
                block_lengths[2] = 1;
                element_offsets[2] =
                    offsetof(struct mtxfile_matrix_coordinate_integer_single, a);
            } else if (precision == mtx_double) {
                num_elements = 3;
                element_types[0] = MPI_INT;
                block_lengths[0] = 1;
                element_offsets[0] =
                    offsetof(struct mtxfile_matrix_coordinate_integer_double, i);
                element_types[1] = MPI_INT;
                block_lengths[1] = 1;
                element_offsets[1] =
                    offsetof(struct mtxfile_matrix_coordinate_integer_double, j);
                element_types[2] = MPI_INT64_T;
                block_lengths[2] = 1;
                element_offsets[2] =
                    offsetof(struct mtxfile_matrix_coordinate_integer_double, a);
            } else {
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_pattern) {
            num_elements = 2;
            element_types[0] = MPI_INT;
            block_lengths[0] = 1;
            element_offsets[0] =
                offsetof(struct mtxfile_matrix_coordinate_pattern, i);
            element_types[1] = MPI_INT;
            block_lengths[1] = 1;
            element_offsets[1] =
                offsetof(struct mtxfile_matrix_coordinate_pattern, j);
        } else {
            return MTX_ERR_INVALID_MTX_FIELD;
        }
    } else if (object == mtxfile_vector) {
        if (field == mtxfile_real) {
            if (precision == mtx_single) {
                num_elements = 2;
                element_types[0] = MPI_INT;
                block_lengths[0] = 1;
                element_offsets[0] =
                    offsetof(struct mtxfile_vector_coordinate_real_single, i);
                element_types[1] = MPI_FLOAT;
                block_lengths[1] = 1;
                element_offsets[1] =
                    offsetof(struct mtxfile_vector_coordinate_real_single, a);
            } else if (precision == mtx_double) {
                num_elements = 2;
                element_types[0] = MPI_INT;
                block_lengths[0] = 1;
                element_offsets[0] =
                    offsetof(struct mtxfile_vector_coordinate_real_double, i);
                element_types[1] = MPI_DOUBLE;
                block_lengths[1] = 1;
                element_offsets[1] =
                    offsetof(struct mtxfile_vector_coordinate_real_double, a);
            } else {
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_complex) {
            if (precision == mtx_single) {
                num_elements = 2;
                element_types[0] = MPI_INT;
                block_lengths[0] = 1;
                element_offsets[0] =
                    offsetof(struct mtxfile_vector_coordinate_complex_single, i);
                element_types[1] = MPI_FLOAT;
                block_lengths[1] = 2;
                element_offsets[1] =
                    offsetof(struct mtxfile_vector_coordinate_complex_single, a);
            } else if (precision == mtx_double) {
                num_elements = 2;
                element_types[0] = MPI_INT;
                block_lengths[0] = 1;
                element_offsets[0] =
                    offsetof(struct mtxfile_vector_coordinate_complex_double, i);
                element_types[1] = MPI_DOUBLE;
                block_lengths[1] = 2;
                element_offsets[1] =
                    offsetof(struct mtxfile_vector_coordinate_complex_double, a);
            } else {
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_integer) {
            if (precision == mtx_single) {
                num_elements = 2;
                element_types[0] = MPI_INT;
                block_lengths[0] = 1;
                element_offsets[0] =
                    offsetof(struct mtxfile_vector_coordinate_integer_single, i);
                element_types[1] = MPI_INT32_T;
                block_lengths[1] = 1;
                element_offsets[1] =
                    offsetof(struct mtxfile_vector_coordinate_integer_single, a);
            } else if (precision == mtx_double) {
                num_elements = 2;
                element_types[0] = MPI_INT;
                block_lengths[0] = 1;
                element_offsets[0] =
                    offsetof(struct mtxfile_vector_coordinate_integer_double, i);
                element_types[1] = MPI_INT64_T;
                block_lengths[1] = 1;
                element_offsets[1] =
                    offsetof(struct mtxfile_vector_coordinate_integer_double, a);
            } else {
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_pattern) {
            num_elements = 1;
            element_types[0] = MPI_INT;
            block_lengths[0] = 1;
            element_offsets[0] =
                offsetof(struct mtxfile_vector_coordinate_pattern, i);
        } else {
            return MTX_ERR_INVALID_MTX_FIELD;
        }
    } else {
        return MTX_ERR_INVALID_MTX_OBJECT;
    }

    /* Create an MPI data type for receiving nonzero data. */
    MPI_Datatype tmp_datatype;
    *mpierrcode = MPI_Type_create_struct(
        num_elements, block_lengths, element_offsets,
        element_types, &tmp_datatype);
    if (*mpierrcode)
        return MTX_ERR_MPI;

    /* Enable sending an array of the custom data type. */
    MPI_Aint lb, extent;
    *mpierrcode = MPI_Type_get_extent(tmp_datatype, &lb, &extent);
    if (*mpierrcode) {
        MPI_Type_free(&tmp_datatype);
        return MTX_ERR_MPI;
    }
    *mpierrcode = MPI_Type_create_resized(tmp_datatype, lb, extent, datatype);
    if (*mpierrcode) {
        MPI_Type_free(&tmp_datatype);
        return MTX_ERR_MPI;
    }
    *mpierrcode = MPI_Type_commit(datatype);
    if (*mpierrcode) {
        MPI_Type_free(datatype);
        MPI_Type_free(&tmp_datatype);
        return MTX_ERR_MPI;
    }

    MPI_Type_free(&tmp_datatype);
    return MTX_SUCCESS;
}

static int mtxfile_data_send_coordinate(
    const union mtxfile_data * data,
    enum mtxfile_object object,
    enum mtxfile_field field,
    enum mtx_precision precision,
    size_t size,
    size_t offset,
    int dest,
    int tag,
    MPI_Comm comm,
    int * mpierrcode)
{
    int err;
    MPI_Datatype datatype;
    err = mtxfile_coordinate_datatype(
        object, field, precision, &datatype, mpierrcode);
    if (err)
        return err;

    if (object == mtxfile_matrix) {
        if (field == mtxfile_real) {
            if (precision == mtx_single) {
                *mpierrcode = MPI_Send(
                    &data->matrix_coordinate_real_single[offset],
                    size, datatype, dest, tag, comm);
                if (*mpierrcode)
                    return MTX_ERR_MPI;
            } else if (precision == mtx_double) {
                *mpierrcode = MPI_Send(
                    &data->matrix_coordinate_real_double[offset],
                    size, datatype, dest, tag, comm);
                if (*mpierrcode)
                    return MTX_ERR_MPI;
            } else {
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_complex) {
            if (precision == mtx_single) {
                *mpierrcode = MPI_Send(
                    &data->matrix_coordinate_complex_single[offset],
                    size, datatype, dest, tag, comm);
                if (*mpierrcode)
                    return MTX_ERR_MPI;
            } else if (precision == mtx_double) {
                *mpierrcode = MPI_Send(
                    &data->matrix_coordinate_complex_double[offset],
                    size, datatype, dest, tag, comm);
                if (*mpierrcode)
                    return MTX_ERR_MPI;
            } else {
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_integer) {
            if (precision == mtx_single) {
                *mpierrcode = MPI_Send(
                    &data->matrix_coordinate_integer_single[offset],
                    size, datatype, dest, tag, comm);
                if (*mpierrcode)
                    return MTX_ERR_MPI;
            } else if (precision == mtx_double) {
                *mpierrcode = MPI_Send(
                    &data->matrix_coordinate_integer_double[offset],
                    size, datatype, dest, tag, comm);
                if (*mpierrcode)
                    return MTX_ERR_MPI;
            } else {
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_pattern) {
            *mpierrcode = MPI_Send(
                &data->matrix_coordinate_pattern[offset],
                size, datatype, dest, tag, comm);
            if (*mpierrcode)
                return MTX_ERR_MPI;
        } else {
            return MTX_ERR_INVALID_MTX_FIELD;
        }

    } else if (object == mtxfile_vector) {
        if (field == mtxfile_real) {
            if (precision == mtx_single) {
                *mpierrcode = MPI_Send(
                    &data->vector_coordinate_real_single[offset],
                    size, datatype, dest, tag, comm);
                if (*mpierrcode)
                    return MTX_ERR_MPI;
            } else if (precision == mtx_double) {
                *mpierrcode = MPI_Send(
                    &data->vector_coordinate_real_double[offset],
                    size, datatype, dest, tag, comm);
                if (*mpierrcode)
                    return MTX_ERR_MPI;
            } else {
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_complex) {
            if (precision == mtx_single) {
                *mpierrcode = MPI_Send(
                    &data->vector_coordinate_complex_single[offset],
                    size, datatype, dest, tag, comm);
                if (*mpierrcode)
                    return MTX_ERR_MPI;
            } else if (precision == mtx_double) {
                *mpierrcode = MPI_Send(
                    &data->vector_coordinate_complex_double[offset],
                    size, datatype, dest, tag, comm);
                if (*mpierrcode)
                    return MTX_ERR_MPI;
            } else {
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_integer) {
            if (precision == mtx_single) {
                *mpierrcode = MPI_Send(
                    &data->vector_coordinate_integer_single[offset],
                    size, datatype, dest, tag, comm);
                if (*mpierrcode)
                    return MTX_ERR_MPI;
            } else if (precision == mtx_double) {
                *mpierrcode = MPI_Send(
                    &data->vector_coordinate_integer_double[offset],
                    size, datatype, dest, tag, comm);
                if (*mpierrcode)
                    return MTX_ERR_MPI;
            } else {
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_pattern) {
            *mpierrcode = MPI_Send(
                &data->vector_coordinate_pattern[offset],
                size, datatype, dest, tag, comm);
            if (*mpierrcode)
                return MTX_ERR_MPI;
        } else {
            return MTX_ERR_INVALID_MTX_FIELD;
        }
    } else {
        return MTX_ERR_INVALID_MTX_OBJECT;
    }
    return MTX_SUCCESS;
}

/**
 * `mtxfile_data_send()' sends Matrix Market data lines to another MPI
 * process.
 *
 * This is analogous to `MPI_Send()' and requires the receiving
 * process to perform a matching call to `mtxfile_data_recv()'.
 */
int mtxfile_data_send(
    const union mtxfile_data * data,
    enum mtxfile_object object,
    enum mtxfile_format format,
    enum mtxfile_field field,
    enum mtx_precision precision,
    size_t size,
    size_t offset,
    int dest,
    int tag,
    MPI_Comm comm,
    struct mtxmpierror * mpierror)
{
    if (format == mtxfile_array) {
        return mtxfile_data_send_array(
            data, field, precision, size, offset,
            dest, tag, comm, &mpierror->err);
    } else if (format == mtxfile_coordinate) {
        return mtxfile_data_send_coordinate(
            data, object, field, precision, size, offset,
            dest, tag, comm, &mpierror->err);
    } else {
        return MTX_ERR_INVALID_MTX_FORMAT;
    }
    return MTX_SUCCESS;
}

static int mtxfile_data_recv_array(
    const union mtxfile_data * data,
    enum mtxfile_field field,
    enum mtx_precision precision,
    size_t size,
    size_t offset,
    int source,
    int tag,
    MPI_Comm comm,
    int * mpierrcode)
{
    if (field == mtxfile_real) {
        if (precision == mtx_single) {
            *mpierrcode = MPI_Recv(
                &data->array_real_single[offset],
                size, MPI_FLOAT, source, tag, comm,
                MPI_STATUS_IGNORE);
            if (*mpierrcode)
                return MTX_ERR_MPI;
        } else if (precision == mtx_double) {
            *mpierrcode = MPI_Recv(
                &data->array_real_double[offset],
                size, MPI_DOUBLE, source, tag, comm,
                MPI_STATUS_IGNORE);
            if (*mpierrcode)
                return MTX_ERR_MPI;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (field == mtxfile_complex) {
        if (precision == mtx_single) {
            *mpierrcode = MPI_Recv(
                &data->array_complex_single[offset],
                2*size, MPI_FLOAT, source, tag, comm,
                MPI_STATUS_IGNORE);
            if (*mpierrcode)
                return MTX_ERR_MPI;
        } else if (precision == mtx_double) {
            *mpierrcode = MPI_Recv(
                &data->array_complex_double[offset],
                2*size, MPI_DOUBLE, source, tag, comm,
                MPI_STATUS_IGNORE);
            if (*mpierrcode)
                return MTX_ERR_MPI;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (field == mtxfile_integer) {
        if (precision == mtx_single) {
            *mpierrcode = MPI_Recv(
                &data->array_integer_single[offset],
                size, MPI_INT32_T, source, tag, comm,
                MPI_STATUS_IGNORE);
            if (*mpierrcode)
                return MTX_ERR_MPI;
        } else if (precision == mtx_double) {
            *mpierrcode = MPI_Recv(
                &data->array_integer_double[offset],
                size, MPI_INT64_T, source, tag, comm,
                MPI_STATUS_IGNORE);
            if (*mpierrcode)
                return MTX_ERR_MPI;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else {
        return MTX_ERR_INVALID_MTX_FIELD;
    }
    return MTX_SUCCESS;
}

static int mtxfile_data_recv_coordinate(
    const union mtxfile_data * data,
    enum mtxfile_object object,
    enum mtxfile_field field,
    enum mtx_precision precision,
    size_t size,
    size_t offset,
    int source,
    int tag,
    MPI_Comm comm,
    int * mpierrcode)
{
    int err;
    MPI_Datatype datatype;
    err = mtxfile_coordinate_datatype(
        object, field, precision, &datatype, mpierrcode);
    if (err)
        return err;

    if (object == mtxfile_matrix) {
        if (field == mtxfile_real) {
            if (precision == mtx_single) {
                *mpierrcode = MPI_Recv(
                    &data->matrix_coordinate_real_single[offset],
                    size, datatype, source, tag, comm, MPI_STATUS_IGNORE);
                if (*mpierrcode)
                    return MTX_ERR_MPI;
            } else if (precision == mtx_double) {
                *mpierrcode = MPI_Recv(
                    &data->matrix_coordinate_real_double[offset],
                    size, datatype, source, tag, comm, MPI_STATUS_IGNORE);
                if (*mpierrcode)
                    return MTX_ERR_MPI;
            } else {
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_complex) {
            if (precision == mtx_single) {
                *mpierrcode = MPI_Recv(
                    &data->matrix_coordinate_complex_single[offset],
                    size, datatype, source, tag, comm, MPI_STATUS_IGNORE);
                if (*mpierrcode)
                    return MTX_ERR_MPI;
            } else if (precision == mtx_double) {
                *mpierrcode = MPI_Recv(
                    &data->matrix_coordinate_complex_double[offset],
                    size, datatype, source, tag, comm, MPI_STATUS_IGNORE);
                if (*mpierrcode)
                    return MTX_ERR_MPI;
            } else {
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_integer) {
            if (precision == mtx_single) {
                *mpierrcode = MPI_Recv(
                    &data->matrix_coordinate_integer_single[offset],
                    size, datatype, source, tag, comm, MPI_STATUS_IGNORE);
                if (*mpierrcode)
                    return MTX_ERR_MPI;
            } else if (precision == mtx_double) {
                *mpierrcode = MPI_Recv(
                    &data->matrix_coordinate_integer_double[offset],
                    size, datatype, source, tag, comm, MPI_STATUS_IGNORE);
                if (*mpierrcode)
                    return MTX_ERR_MPI;
            } else {
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_pattern) {
            *mpierrcode = MPI_Recv(
                &data->matrix_coordinate_pattern[offset],
                size, datatype, source, tag, comm, MPI_STATUS_IGNORE);
            if (*mpierrcode)
                return MTX_ERR_MPI;
        } else {
            return MTX_ERR_INVALID_MTX_FIELD;
        }

    } else if (object == mtxfile_vector) {
        if (field == mtxfile_real) {
            if (precision == mtx_single) {
                *mpierrcode = MPI_Recv(
                    &data->vector_coordinate_real_single[offset],
                    size, datatype, source, tag, comm, MPI_STATUS_IGNORE);
                if (*mpierrcode)
                    return MTX_ERR_MPI;
            } else if (precision == mtx_double) {
                *mpierrcode = MPI_Recv(
                    &data->vector_coordinate_real_double[offset],
                    size, datatype, source, tag, comm, MPI_STATUS_IGNORE);
                if (*mpierrcode)
                    return MTX_ERR_MPI;
            } else {
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_complex) {
            if (precision == mtx_single) {
                *mpierrcode = MPI_Recv(
                    &data->vector_coordinate_complex_single[offset],
                    size, datatype, source, tag, comm, MPI_STATUS_IGNORE);
                if (*mpierrcode)
                    return MTX_ERR_MPI;
            } else if (precision == mtx_double) {
                *mpierrcode = MPI_Recv(
                    &data->vector_coordinate_complex_double[offset],
                    size, datatype, source, tag, comm, MPI_STATUS_IGNORE);
                if (*mpierrcode)
                    return MTX_ERR_MPI;
            } else {
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_integer) {
            if (precision == mtx_single) {
                *mpierrcode = MPI_Recv(
                    &data->vector_coordinate_integer_single[offset],
                    size, datatype, source, tag, comm, MPI_STATUS_IGNORE);
                if (*mpierrcode)
                    return MTX_ERR_MPI;
            } else if (precision == mtx_double) {
                *mpierrcode = MPI_Recv(
                    &data->vector_coordinate_integer_double[offset],
                    size, datatype, source, tag, comm, MPI_STATUS_IGNORE);
                if (*mpierrcode)
                    return MTX_ERR_MPI;
            } else {
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_pattern) {
            *mpierrcode = MPI_Recv(
                &data->vector_coordinate_pattern[offset],
                size, datatype, source, tag, comm, MPI_STATUS_IGNORE);
            if (*mpierrcode)
                return MTX_ERR_MPI;
        } else {
            return MTX_ERR_INVALID_MTX_FIELD;
        }
    } else {
        return MTX_ERR_INVALID_MTX_OBJECT;
    }
    return MTX_SUCCESS;
}

/**
 * `mtxfile_data_recv()' receives Matrix Market data lines from
 * another MPI process.
 *
 * This is analogous to `MPI_Recv()' and requires the sending process
 * to perform a matching call to `mtxfile_data_send()'.
 */
int mtxfile_data_recv(
    union mtxfile_data * data,
    enum mtxfile_object object,
    enum mtxfile_format format,
    enum mtxfile_field field,
    enum mtx_precision precision,
    size_t size,
    size_t offset,
    int source,
    int tag,
    MPI_Comm comm,
    struct mtxmpierror * mpierror)
{
    if (format == mtxfile_array) {
        return mtxfile_data_recv_array(
            data, field, precision, size, offset,
            source, tag, comm, &mpierror->err);
    } else if (format == mtxfile_coordinate) {
        return mtxfile_data_recv_coordinate(
            data, object, field, precision, size, offset,
            source, tag, comm, &mpierror->err);
    } else {
        return MTX_ERR_INVALID_MTX_FORMAT;
    }
    return MTX_SUCCESS;
}

static int mtxfile_data_bcast_array(
    const union mtxfile_data * data,
    enum mtxfile_field field,
    enum mtx_precision precision,
    size_t size,
    size_t offset,
    int root,
    MPI_Comm comm,
    int * mpierrcode)
{
    if (field == mtxfile_real) {
        if (precision == mtx_single) {
            *mpierrcode = MPI_Bcast(
                &data->array_real_single[offset], size, MPI_FLOAT, root, comm);
            if (*mpierrcode)
                return MTX_ERR_MPI;
        } else if (precision == mtx_double) {
            *mpierrcode = MPI_Bcast(
                &data->array_real_double[offset], size, MPI_DOUBLE, root, comm);
            if (*mpierrcode)
                return MTX_ERR_MPI;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (field == mtxfile_complex) {
        if (precision == mtx_single) {
            *mpierrcode = MPI_Bcast(
                &data->array_complex_single[offset], 2*size, MPI_FLOAT, root, comm);
            if (*mpierrcode)
                return MTX_ERR_MPI;
        } else if (precision == mtx_double) {
            *mpierrcode = MPI_Bcast(
                &data->array_complex_double[offset], 2*size, MPI_DOUBLE, root, comm);
            if (*mpierrcode)
                return MTX_ERR_MPI;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (field == mtxfile_integer) {
        if (precision == mtx_single) {
            *mpierrcode = MPI_Bcast(
                &data->array_integer_single[offset], size, MPI_INT32_T, root, comm);
            if (*mpierrcode)
                return MTX_ERR_MPI;
        } else if (precision == mtx_double) {
            *mpierrcode = MPI_Bcast(
                &data->array_integer_double[offset], size, MPI_INT64_T, root, comm);
            if (*mpierrcode)
                return MTX_ERR_MPI;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else {
        return MTX_ERR_INVALID_MTX_FIELD;
    }
    return MTX_SUCCESS;
}

static int mtxfile_data_bcast_coordinate(
    const union mtxfile_data * data,
    enum mtxfile_object object,
    enum mtxfile_field field,
    enum mtx_precision precision,
    size_t size,
    size_t offset,
    int root,
    MPI_Comm comm,
    int * mpierrcode)
{
    int err;
    MPI_Datatype datatype;
    err = mtxfile_coordinate_datatype(
        object, field, precision, &datatype, mpierrcode);
    if (err)
        return err;

    if (object == mtxfile_matrix) {
        if (field == mtxfile_real) {
            if (precision == mtx_single) {
                *mpierrcode = MPI_Bcast(
                    &data->matrix_coordinate_real_single[offset],
                    size, datatype, root, comm);
                if (*mpierrcode)
                    return MTX_ERR_MPI;
            } else if (precision == mtx_double) {
                *mpierrcode = MPI_Bcast(
                    &data->matrix_coordinate_real_double[offset],
                    size, datatype, root, comm);
                if (*mpierrcode)
                    return MTX_ERR_MPI;
            } else {
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_complex) {
            if (precision == mtx_single) {
                *mpierrcode = MPI_Bcast(
                    &data->matrix_coordinate_complex_single[offset],
                    size, datatype, root, comm);
                if (*mpierrcode)
                    return MTX_ERR_MPI;
            } else if (precision == mtx_double) {
                *mpierrcode = MPI_Bcast(
                    &data->matrix_coordinate_complex_double[offset],
                    size, datatype, root, comm);
                if (*mpierrcode)
                    return MTX_ERR_MPI;
            } else {
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_integer) {
            if (precision == mtx_single) {
                *mpierrcode = MPI_Bcast(
                    &data->matrix_coordinate_integer_single[offset],
                    size, datatype, root, comm);
                if (*mpierrcode)
                    return MTX_ERR_MPI;
            } else if (precision == mtx_double) {
                *mpierrcode = MPI_Bcast(
                    &data->matrix_coordinate_integer_double[offset],
                    size, datatype, root, comm);
                if (*mpierrcode)
                    return MTX_ERR_MPI;
            } else {
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_pattern) {
            *mpierrcode = MPI_Bcast(
                &data->matrix_coordinate_pattern[offset],
                size, datatype, root, comm);
            if (*mpierrcode)
                return MTX_ERR_MPI;
        } else {
            return MTX_ERR_INVALID_MTX_FIELD;
        }

    } else if (object == mtxfile_vector) {
        if (field == mtxfile_real) {
            if (precision == mtx_single) {
                *mpierrcode = MPI_Bcast(
                    &data->vector_coordinate_real_single[offset],
                    size, datatype, root, comm);
                if (*mpierrcode)
                    return MTX_ERR_MPI;
            } else if (precision == mtx_double) {
                *mpierrcode = MPI_Bcast(
                    &data->vector_coordinate_real_double[offset],
                    size, datatype, root, comm);
                if (*mpierrcode)
                    return MTX_ERR_MPI;
            } else {
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_complex) {
            if (precision == mtx_single) {
                *mpierrcode = MPI_Bcast(
                    &data->vector_coordinate_complex_single[offset],
                    size, datatype, root, comm);
                if (*mpierrcode)
                    return MTX_ERR_MPI;
            } else if (precision == mtx_double) {
                *mpierrcode = MPI_Bcast(
                    &data->vector_coordinate_complex_double[offset],
                    size, datatype, root, comm);
                if (*mpierrcode)
                    return MTX_ERR_MPI;
            } else {
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_integer) {
            if (precision == mtx_single) {
                *mpierrcode = MPI_Bcast(
                    &data->vector_coordinate_integer_single[offset],
                    size, datatype, root, comm);
                if (*mpierrcode)
                    return MTX_ERR_MPI;
            } else if (precision == mtx_double) {
                *mpierrcode = MPI_Bcast(
                    &data->vector_coordinate_integer_double[offset],
                    size, datatype, root, comm);
                if (*mpierrcode)
                    return MTX_ERR_MPI;
            } else {
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_pattern) {
            *mpierrcode = MPI_Bcast(
                &data->vector_coordinate_pattern[offset],
                size, datatype, root, comm);
            if (*mpierrcode)
                return MTX_ERR_MPI;
        } else {
            return MTX_ERR_INVALID_MTX_FIELD;
        }
    } else {
        return MTX_ERR_INVALID_MTX_OBJECT;
    }
    return MTX_SUCCESS;
}

/**
 * `mtxfile_data_bcast()' broadcasts Matrix Market data lines from an
 * MPI root process to other processes in a communicator.
 *
 * This is analogous to `MPI_Bcast()' and requires every process in
 * the communicator to perform matching calls to
 * `mtxfile_data_bcast()'.
 */
int mtxfile_data_bcast(
    union mtxfile_data * data,
    enum mtxfile_object object,
    enum mtxfile_format format,
    enum mtxfile_field field,
    enum mtx_precision precision,
    size_t size,
    size_t offset,
    int root,
    MPI_Comm comm,
    struct mtxmpierror * mpierror)
{
    if (format == mtxfile_array) {
        return mtxfile_data_bcast_array(
            data, field, precision, size, offset,
            root, comm, &mpierror->err);
    } else if (format == mtxfile_coordinate) {
        return mtxfile_data_bcast_coordinate(
            data, object, field, precision, size, offset,
            root, comm, &mpierror->err);
    } else {
        return MTX_ERR_INVALID_MTX_FORMAT;
    }
    return MTX_SUCCESS;
}

static int mtxfile_data_scatterv_array(
    const union mtxfile_data * sendbuf,
    enum mtxfile_field field,
    enum mtx_precision precision,
    size_t sendoffset,
    int * sendcounts,
    int * displs,
    union mtxfile_data * recvbuf,
    size_t recvoffset,
    int recvcount,
    int root,
    MPI_Comm comm,
    int * mpierrcode)
{
    if (field == mtxfile_real) {
        if (precision == mtx_single) {
            *mpierrcode = MPI_Scatterv(
                &sendbuf->array_real_single[sendoffset], sendcounts, displs, MPI_FLOAT,
                &recvbuf->array_real_single[recvoffset], recvcount, MPI_FLOAT,
                root, comm);
            if (*mpierrcode)
                return MTX_ERR_MPI;
        } else if (precision == mtx_double) {
            *mpierrcode = MPI_Scatterv(
                &sendbuf->array_real_double[sendoffset], sendcounts, displs, MPI_DOUBLE,
                &recvbuf->array_real_double[recvoffset], recvcount, MPI_DOUBLE,
                root, comm);
            if (*mpierrcode)
                return MTX_ERR_MPI;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (field == mtxfile_complex) {
        int comm_size;
        *mpierrcode = MPI_Comm_size(comm, &comm_size);
        if (*mpierrcode)
            MPI_Abort(comm, EXIT_FAILURE);
        for (int p = 0; p < comm_size; p++) {
            sendcounts[p] *= 2;
            displs[p] *= 2;
        }
        if (precision == mtx_single) {
            *mpierrcode = MPI_Scatterv(
                &sendbuf->array_complex_single[sendoffset], sendcounts, displs, MPI_FLOAT,
                &recvbuf->array_complex_single[recvoffset], recvcount, MPI_FLOAT,
                root, comm);
            if (*mpierrcode)
                return MTX_ERR_MPI;
        } else if (precision == mtx_double) {
            *mpierrcode = MPI_Scatterv(
                &sendbuf->array_complex_double[sendoffset], sendcounts, displs, MPI_DOUBLE,
                &recvbuf->array_complex_double[recvoffset], recvcount, MPI_DOUBLE,
                root, comm);
            if (*mpierrcode)
                return MTX_ERR_MPI;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (field == mtxfile_integer) {
        if (precision == mtx_single) {
            *mpierrcode = MPI_Scatterv(
                &sendbuf->array_integer_single[sendoffset], sendcounts, displs, MPI_INT32_T,
                &recvbuf->array_integer_single[recvoffset], recvcount, MPI_INT32_T,
                root, comm);
            if (*mpierrcode)
                return MTX_ERR_MPI;
        } else if (precision == mtx_double) {
            *mpierrcode = MPI_Scatterv(
                &sendbuf->array_integer_double[sendoffset], sendcounts, displs, MPI_INT64_T,
                &recvbuf->array_integer_double[recvoffset], recvcount, MPI_INT64_T,
                root, comm);
            if (*mpierrcode)
                return MTX_ERR_MPI;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else {
        return MTX_ERR_INVALID_MTX_FIELD;
    }
    return MTX_SUCCESS;
}

static int mtxfile_data_scatterv_coordinate(
    const union mtxfile_data * sendbuf,
    enum mtxfile_object object,
    enum mtxfile_field field,
    enum mtx_precision precision,
    size_t sendoffset,
    int * sendcounts,
    int * displs,
    union mtxfile_data * recvbuf,
    size_t recvoffset,
    int recvcount,
    int root,
    MPI_Comm comm,
    int * mpierrcode)
{
    int err;
    MPI_Datatype datatype;
    err = mtxfile_coordinate_datatype(
        object, field, precision, &datatype, mpierrcode);
    if (err)
        return err;

    if (object == mtxfile_matrix) {
        if (field == mtxfile_real) {
            if (precision == mtx_single) {
                *mpierrcode = MPI_Scatterv(
                    &sendbuf->matrix_coordinate_real_single[sendoffset],
                    sendcounts, displs, datatype,
                    &recvbuf->matrix_coordinate_real_single[recvoffset],
                    recvcount, datatype, root, comm);
                if (*mpierrcode)
                    return MTX_ERR_MPI;
            } else if (precision == mtx_double) {
                *mpierrcode = MPI_Scatterv(
                    &sendbuf->matrix_coordinate_real_double[sendoffset],
                    sendcounts, displs, datatype,
                    &recvbuf->matrix_coordinate_real_double[recvoffset],
                    recvcount, datatype, root, comm);
                if (*mpierrcode)
                    return MTX_ERR_MPI;
            } else {
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_complex) {
            if (precision == mtx_single) {
                *mpierrcode = MPI_Scatterv(
                    &sendbuf->matrix_coordinate_complex_single[sendoffset],
                    sendcounts, displs, datatype,
                    &recvbuf->matrix_coordinate_complex_single[recvoffset],
                    recvcount, datatype, root, comm);
                if (*mpierrcode)
                    return MTX_ERR_MPI;
            } else if (precision == mtx_double) {
                *mpierrcode = MPI_Scatterv(
                    &sendbuf->matrix_coordinate_complex_double[sendoffset],
                    sendcounts, displs, datatype,
                    &recvbuf->matrix_coordinate_complex_double[recvoffset],
                    recvcount, datatype, root, comm);
                if (*mpierrcode)
                    return MTX_ERR_MPI;
            } else {
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_integer) {
            if (precision == mtx_single) {
                *mpierrcode = MPI_Scatterv(
                    &sendbuf->matrix_coordinate_integer_single[sendoffset],
                    sendcounts, displs, datatype,
                    &recvbuf->matrix_coordinate_integer_single[recvoffset],
                    recvcount, datatype, root, comm);
                if (*mpierrcode)
                    return MTX_ERR_MPI;
            } else if (precision == mtx_double) {
                *mpierrcode = MPI_Scatterv(
                    &sendbuf->matrix_coordinate_integer_double[sendoffset],
                    sendcounts, displs, datatype,
                    &recvbuf->matrix_coordinate_integer_double[recvoffset],
                    recvcount, datatype, root, comm);
                if (*mpierrcode)
                    return MTX_ERR_MPI;
            } else {
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_pattern) {
            *mpierrcode = MPI_Scatterv(
                &sendbuf->matrix_coordinate_pattern[sendoffset],
                sendcounts, displs, datatype,
                &recvbuf->matrix_coordinate_pattern[recvoffset],
                recvcount, datatype, root, comm);
            if (*mpierrcode)
                return MTX_ERR_MPI;
        } else {
            return MTX_ERR_INVALID_MTX_FIELD;
        }

    } else if (object == mtxfile_vector) {
        if (field == mtxfile_real) {
            if (precision == mtx_single) {
                *mpierrcode = MPI_Scatterv(
                    &sendbuf->vector_coordinate_real_single[sendoffset],
                    sendcounts, displs, datatype,
                    &recvbuf->vector_coordinate_real_single[recvoffset],
                    recvcount, datatype, root, comm);
                if (*mpierrcode)
                    return MTX_ERR_MPI;
            } else if (precision == mtx_double) {
                *mpierrcode = MPI_Scatterv(
                    &sendbuf->vector_coordinate_real_double[sendoffset],
                    sendcounts, displs, datatype,
                    &recvbuf->vector_coordinate_real_double[recvoffset],
                    recvcount, datatype, root, comm);
                if (*mpierrcode)
                    return MTX_ERR_MPI;
            } else {
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_complex) {
            if (precision == mtx_single) {
                *mpierrcode = MPI_Scatterv(
                    &sendbuf->vector_coordinate_complex_single[sendoffset],
                    sendcounts, displs, datatype,
                    &recvbuf->vector_coordinate_complex_single[recvoffset],
                    recvcount, datatype, root, comm);
                if (*mpierrcode)
                    return MTX_ERR_MPI;
            } else if (precision == mtx_double) {
                *mpierrcode = MPI_Scatterv(
                    &sendbuf->vector_coordinate_complex_double[sendoffset],
                    sendcounts, displs, datatype,
                    &recvbuf->vector_coordinate_complex_double[recvoffset],
                    recvcount, datatype, root, comm);
                if (*mpierrcode)
                    return MTX_ERR_MPI;
            } else {
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_integer) {
            if (precision == mtx_single) {
                *mpierrcode = MPI_Scatterv(
                    &sendbuf->vector_coordinate_integer_single[sendoffset],
                    sendcounts, displs, datatype,
                    &recvbuf->vector_coordinate_integer_single[recvoffset],
                    recvcount, datatype, root, comm);
                if (*mpierrcode)
                    return MTX_ERR_MPI;
            } else if (precision == mtx_double) {
                *mpierrcode = MPI_Scatterv(
                    &sendbuf->vector_coordinate_integer_double[sendoffset],
                    sendcounts, displs, datatype,
                    &recvbuf->vector_coordinate_integer_double[recvoffset],
                    recvcount, datatype, root, comm);
                if (*mpierrcode)
                    return MTX_ERR_MPI;
            } else {
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_pattern) {
            *mpierrcode = MPI_Scatterv(
                &sendbuf->vector_coordinate_pattern[sendoffset],
                sendcounts, displs, datatype,
                &recvbuf->vector_coordinate_pattern[recvoffset],
                recvcount, datatype, root, comm);
            if (*mpierrcode)
                return MTX_ERR_MPI;
        } else {
            return MTX_ERR_INVALID_MTX_FIELD;
        }
    } else {
        return MTX_ERR_INVALID_MTX_OBJECT;
    }
    return MTX_SUCCESS;
}

/**
 * `mtxfile_data_scatterv()' scatters Matrix Market data lines from an
 * MPI root process to other processes in a communicator.
 *
 * This is analogous to `MPI_Scatterv()' and requires every process in
 * the communicator to perform matching calls to
 * `mtxfile_data_scatterv()'.
 */
int mtxfile_data_scatterv(
    const union mtxfile_data * sendbuf,
    enum mtxfile_object object,
    enum mtxfile_format format,
    enum mtxfile_field field,
    enum mtx_precision precision,
    size_t sendoffset,
    int * sendcounts,
    int * displs,
    union mtxfile_data * recvbuf,
    size_t recvoffset,
    int recvcount,
    int root,
    MPI_Comm comm,
    struct mtxmpierror * mpierror)
{
    if (format == mtxfile_array) {
        return mtxfile_data_scatterv_array(
            sendbuf, field, precision, sendoffset, sendcounts, displs,
            recvbuf, recvoffset, recvcount, root, comm, &mpierror->err);
    } else if (format == mtxfile_coordinate) {
        return mtxfile_data_scatterv_coordinate(
            sendbuf, object, field, precision, sendoffset, sendcounts, displs,
            recvbuf, recvoffset, recvcount, root, comm, &mpierror->err);
    } else {
        return MTX_ERR_INVALID_MTX_FORMAT;
    }
    return MTX_SUCCESS;
}
#endif
