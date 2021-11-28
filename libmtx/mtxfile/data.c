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
#include <libmtx/mtxfile/data.h>
#include <libmtx/mtxfile/header.h>
#include <libmtx/mtxfile/size.h>

#include <libmtx/util/parse.h>
#include <libmtx/util/format.h>
#include <libmtx/util/partition.h>

#ifdef LIBMTX_HAVE_MPI
#include <mpi.h>
#endif

#ifdef LIBMTX_HAVE_LIBZ
#include <zlib.h>
#endif

#include <errno.h>
#include <unistd.h>

#include <float.h>
#include <inttypes.h>
#include <locale.h>
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
    int64_t size)
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
    int64_t size,
    int64_t dst_offset,
    int64_t src_offset)
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
                       size * sizeof(*src->vector_coordinate_pattern));
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
 * ‘mtxfile_data_copy_gather()’ performs an irregular copying (gather)
 * of data lines from specified locations to a contiguous array.
 */
int mtxfile_data_copy_gather(
    union mtxfile_data * dst,
    const union mtxfile_data * src,
    enum mtxfile_object object,
    enum mtxfile_format format,
    enum mtxfile_field field,
    enum mtx_precision precision,
    int64_t size,
    int64_t dstoffset,
    const int64_t * srcdispls)
{
    if (format == mtxfile_array) {
        if (field == mtxfile_real) {
            if (precision == mtx_single) {
                for (int64_t k = 0; k < size; k++) {
                    dst->array_real_single[dstoffset+k] =
                        src->array_real_single[srcdispls[k]];
                }
            } else if (precision == mtx_double) {
                for (int64_t k = 0; k < size; k++) {
                    dst->array_real_double[dstoffset+k] =
                        src->array_real_double[srcdispls[k]];
                }
            } else {
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_complex) {
            if (precision == mtx_single) {
                for (int64_t k = 0; k < size; k++) {
                    dst->array_complex_single[dstoffset+k][0] =
                        src->array_complex_single[srcdispls[k]][0];
                    dst->array_complex_single[dstoffset+k][1] =
                        src->array_complex_single[srcdispls[k]][1];
                }
            } else if (precision == mtx_double) {
                for (int64_t k = 0; k < size; k++) {
                    dst->array_complex_double[dstoffset+k][0] =
                        src->array_complex_double[srcdispls[k]][0];
                    dst->array_complex_double[dstoffset+k][1] =
                        src->array_complex_double[srcdispls[k]][1];
                }
            } else {
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_integer) {
            if (precision == mtx_single) {
                for (int64_t k = 0; k < size; k++) {
                    dst->array_integer_single[dstoffset+k] =
                        src->array_integer_single[srcdispls[k]];
                }
            } else if (precision == mtx_double) {
                for (int64_t k = 0; k < size; k++) {
                    dst->array_integer_double[dstoffset+k] =
                        src->array_integer_double[srcdispls[k]];
                }
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
                    for (int64_t k = 0; k < size; k++) {
                        dst->matrix_coordinate_real_single[dstoffset+k] =
                            src->matrix_coordinate_real_single[srcdispls[k]];
                    }
                } else if (precision == mtx_double) {
                    for (int64_t k = 0; k < size; k++) {
                        dst->matrix_coordinate_real_double[dstoffset+k] =
                            src->matrix_coordinate_real_double[srcdispls[k]];
                    }
                } else {
                    return MTX_ERR_INVALID_PRECISION;
                }
            } else if (field == mtxfile_complex) {
                if (precision == mtx_single) {
                    for (int64_t k = 0; k < size; k++) {
                        dst->matrix_coordinate_complex_single[dstoffset+k] =
                            src->matrix_coordinate_complex_single[srcdispls[k]];
                    }
                } else if (precision == mtx_double) {
                    for (int64_t k = 0; k < size; k++) {
                        dst->matrix_coordinate_complex_double[dstoffset+k] =
                            src->matrix_coordinate_complex_double[srcdispls[k]];
                    }
                } else {
                    return MTX_ERR_INVALID_PRECISION;
                }
            } else if (field == mtxfile_integer) {
                if (precision == mtx_single) {
                    for (int64_t k = 0; k < size; k++) {
                        dst->matrix_coordinate_integer_single[dstoffset+k] =
                            src->matrix_coordinate_integer_single[srcdispls[k]];
                    }
                } else if (precision == mtx_double) {
                    for (int64_t k = 0; k < size; k++) {
                        dst->matrix_coordinate_integer_double[dstoffset+k] =
                            src->matrix_coordinate_integer_double[srcdispls[k]];
                    }
                } else {
                    return MTX_ERR_INVALID_PRECISION;
                }
            } else if (field == mtxfile_pattern) {
                for (int64_t k = 0; k < size; k++) {
                    dst->matrix_coordinate_pattern[dstoffset+k] =
                        src->matrix_coordinate_pattern[srcdispls[k]];
                }
            } else {
                return MTX_ERR_INVALID_MTX_FIELD;
            }
        } else if (object == mtxfile_vector) {
            if (field == mtxfile_real) {
                if (precision == mtx_single) {
                    for (int64_t k = 0; k < size; k++) {
                        dst->vector_coordinate_real_single[dstoffset+k] =
                            src->vector_coordinate_real_single[srcdispls[k]];
                    }
                } else if (precision == mtx_double) {
                    for (int64_t k = 0; k < size; k++) {
                        dst->vector_coordinate_real_double[dstoffset+k] =
                            src->vector_coordinate_real_double[srcdispls[k]];
                    }
                } else {
                    return MTX_ERR_INVALID_PRECISION;
                }
            } else if (field == mtxfile_complex) {
                if (precision == mtx_single) {
                    for (int64_t k = 0; k < size; k++) {
                        dst->vector_coordinate_complex_single[dstoffset+k] =
                            src->vector_coordinate_complex_single[srcdispls[k]];
                    }
                } else if (precision == mtx_double) {
                    for (int64_t k = 0; k < size; k++) {
                        dst->vector_coordinate_complex_double[dstoffset+k] =
                            src->vector_coordinate_complex_double[srcdispls[k]];
                    }
                } else {
                    return MTX_ERR_INVALID_PRECISION;
                }
            } else if (field == mtxfile_integer) {
                if (precision == mtx_single) {
                    for (int64_t k = 0; k < size; k++) {
                        dst->vector_coordinate_integer_single[dstoffset+k] =
                            src->vector_coordinate_integer_single[srcdispls[k]];
                    }
                } else if (precision == mtx_double) {
                    for (int64_t k = 0; k < size; k++) {
                        dst->vector_coordinate_integer_double[dstoffset+k] =
                            src->vector_coordinate_integer_double[srcdispls[k]];
                    }
                } else {
                    return MTX_ERR_INVALID_PRECISION;
                }
            } else if (field == mtxfile_pattern) {
                for (int64_t k = 0; k < size; k++) {
                    dst->vector_coordinate_pattern[dstoffset+k] =
                        src->vector_coordinate_pattern[srcdispls[k]];
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
 * Extracting row/column pointers and indices
 */

/**
 * `mtxfile_data_rowptr()' computes row pointers for a matrix in
 * coordinate format.
 *
 * ‘rowptr’ must point to an array containing enough storage for
 * ‘num_rows+1’ values of type ‘int64_t’.
 *
 * ‘colidx’ may be ‘NULL’, in which case it is ignored. Otherwise, it
 * must point to an array containing enough storage for ‘size’ values
 * of type ‘int’.  On successful completion, this array will contain
 * the column indices of the nonzero matrix entries arranged in row
 * major order.  The ‘i’-th entry of ‘rowptr’ is the location in the
 * ‘colidx’ array of the first nonzero that belongs to the ‘i+1’-th
 * row of the matrix, for ‘i=0,1,...,num_rows-1’.  The final entry of
 * ‘rowptr’ indicates the position one place beyond the last nonzero.
 *
 * The matrix data is not required to be sorted in any particular
 * order.
 */
int mtxfile_data_rowptr(
    const union mtxfile_data * data,
    enum mtxfile_object object,
    enum mtxfile_format format,
    enum mtxfile_field field,
    enum mtx_precision precision,
    int num_rows,
    int64_t size,
    int64_t * rowptr,
    int * colidx)
{
    if (object != mtxfile_matrix)
        return MTX_ERR_INCOMPATIBLE_MTX_OBJECT;
    if (format != mtxfile_coordinate)
        return MTX_ERR_INCOMPATIBLE_MTX_FORMAT;
    for (int i = 0; i <= num_rows; i++)
        rowptr[i] = 0;
    if (field == mtxfile_real) {
        if (precision == mtx_single) {
            for (int64_t k = 0; k < size; k++)
                rowptr[data->matrix_coordinate_real_single[k].i]++;
        } else if (precision == mtx_double) {
            for (int64_t k = 0; k < size; k++)
                rowptr[data->matrix_coordinate_real_double[k].i]++;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (field == mtxfile_complex) {
        if (precision == mtx_single) {
            for (int64_t k = 0; k < size; k++)
                rowptr[data->matrix_coordinate_complex_single[k].i]++;
        } else if (precision == mtx_double) {
            for (int64_t k = 0; k < size; k++)
                rowptr[data->matrix_coordinate_complex_double[k].i]++;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (field == mtxfile_integer) {
        if (precision == mtx_single) {
            for (int64_t k = 0; k < size; k++)
                rowptr[data->matrix_coordinate_integer_single[k].i]++;
        } else if (precision == mtx_double) {
            for (int64_t k = 0; k < size; k++)
                rowptr[data->matrix_coordinate_integer_double[k].i]++;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (field == mtxfile_pattern) {
        for (int64_t k = 0; k < size; k++)
            rowptr[data->matrix_coordinate_pattern[k].i]++;
    } else {
        return MTX_ERR_INVALID_MTX_FIELD;
    }
    for (int i = 1; i <= num_rows; i++)
        rowptr[i] += rowptr[i-1];

    if (!colidx)
        return MTX_SUCCESS;

    /* Allocate storage for auxiliary row pointers. */
    int64_t * rowendptr = malloc((num_rows+1) * sizeof(int64_t));
    if (!rowendptr)
        return MTX_ERR_ERRNO;
    for (int i = 0; i <= num_rows; i++)
        rowendptr[i] = rowptr[i];

    /* Sort nonzeros in row major order, keeping only the column indices. */
    if (field == mtxfile_real) {
        if (precision == mtx_single) {
            const struct mtxfile_matrix_coordinate_real_single * src =
                data->matrix_coordinate_real_single;
            for (int64_t k = 0; k < size; k++)
                colidx[k] = src[k].j;
            for (int64_t k = 0; k < size; k++) {
                int i = src[k].i-1;
                int64_t l = rowendptr[i]-1;
                while (l >= rowptr[i] && colidx[l] > src[k].j) {
                    colidx[l+1] = colidx[l];
                    l--;
                }
                colidx[l+1] = src[k].j;
                rowendptr[i]++;
            }
        } else if (precision == mtx_double) {
            const struct mtxfile_matrix_coordinate_real_double * src =
                data->matrix_coordinate_real_double;
            for (int64_t k = 0; k < size; k++)
                colidx[k] = src[k].j;
            for (int64_t k = 0; k < size; k++) {
                int i = src[k].i-1;
                int64_t l = rowendptr[i]-1;
                while (l >= rowptr[i] && colidx[l] > src[k].j) {
                    colidx[l+1] = colidx[l];
                    l--;
                }
                colidx[l+1] = src[k].j;
                rowendptr[i]++;
            }
        } else {
            free(rowendptr);
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (field == mtxfile_complex) {
        if (precision == mtx_single) {
            const struct mtxfile_matrix_coordinate_complex_single * src =
                data->matrix_coordinate_complex_single;
            for (int64_t k = 0; k < size; k++)
                colidx[k] = src[k].j;
            for (int64_t k = 0; k < size; k++) {
                int i = src[k].i-1;
                int64_t l = rowendptr[i]-1;
                while (l >= rowptr[i] && colidx[l] > src[k].j) {
                    colidx[l+1] = colidx[l];
                    l--;
                }
                colidx[l+1] = src[k].j;
                rowendptr[i]++;
            }
        } else if (precision == mtx_double) {
            const struct mtxfile_matrix_coordinate_complex_double * src =
                data->matrix_coordinate_complex_double;
            for (int64_t k = 0; k < size; k++)
                colidx[k] = src[k].j;
            for (int64_t k = 0; k < size; k++) {
                int i = src[k].i-1;
                int64_t l = rowendptr[i]-1;
                while (l >= rowptr[i] && colidx[l] > src[k].j) {
                    colidx[l+1] = colidx[l];
                    l--;
                }
                colidx[l+1] = src[k].j;
                rowendptr[i]++;
            }
        } else {
            free(rowendptr);
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (field == mtxfile_integer) {
        if (precision == mtx_single) {
            const struct mtxfile_matrix_coordinate_integer_single * src =
                data->matrix_coordinate_integer_single;
            for (int64_t k = 0; k < size; k++)
                colidx[k] = src[k].j;
            for (int64_t k = 0; k < size; k++) {
                int i = src[k].i-1;
                int64_t l = rowendptr[i]-1;
                while (l >= rowptr[i] && colidx[l] > src[k].j) {
                    colidx[l+1] = colidx[l];
                    l--;
                }
                colidx[l+1] = src[k].j;
                rowendptr[i]++;
            }
        } else if (precision == mtx_double) {
            const struct mtxfile_matrix_coordinate_integer_double * src =
                data->matrix_coordinate_integer_double;
            for (int64_t k = 0; k < size; k++)
                colidx[k] = src[k].j;
            for (int64_t k = 0; k < size; k++) {
                int i = src[k].i-1;
                int64_t l = rowendptr[i]-1;
                while (l >= rowptr[i] && colidx[l] > src[k].j) {
                    colidx[l+1] = colidx[l];
                    l--;
                }
                colidx[l+1] = src[k].j;
                rowendptr[i]++;
            }
        } else {
            free(rowendptr);
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (field == mtxfile_pattern) {
        const struct mtxfile_matrix_coordinate_pattern * src =
            data->matrix_coordinate_pattern;
        for (int64_t k = 0; k < size; k++)
            colidx[k] = src[k].j;
        for (int64_t k = 0; k < size; k++) {
            int i = src[k].i-1;
            int64_t l = rowendptr[i]-1;
            while (l >= rowptr[i] && colidx[l] > src[k].j) {
                colidx[l+1] = colidx[l];
                l--;
            }
            colidx[l+1] = src[k].j;
            rowendptr[i]++;
        }
    } else {
        free(rowendptr);
        return MTX_ERR_INVALID_MTX_FIELD;
    }
    free(rowendptr);
    return MTX_SUCCESS;
}

/**
 * `mtxfile_data_colptr()' computes column pointers for a matrix in
 * coordinate format.
 *
 * ‘colptr’ must point to an array containing enough storage for
 * ‘num_columns+1’ values of type ‘int64_t’.
 *
 * ‘colidx’ may be ‘NULL’, in which case it is ignored. Otherwise, it
 * must point to an array containing enough storage for ‘size’ values
 * of type ‘int’.  On successful completion, this array will contain
 * the column indices of the nonzero matrix entries arranged in column
 * major order.  The ‘i’-th entry of ‘colptr’ is the location in the
 * ‘colidx’ array of the first nonzero that belongs to the ‘i+1’-th
 * column of the matrix, for ‘i=0,1,...,num_columns-1’.  The final
 * entry of ‘colptr’ indicates the position one place beyond the last
 * nonzero.
 *
 * The matrix data is not required to be sorted in any particular
 * order.
 */
int mtxfile_data_colptr(
    const union mtxfile_data * data,
    enum mtxfile_object object,
    enum mtxfile_format format,
    enum mtxfile_field field,
    enum mtx_precision precision,
    int num_columns,
    int64_t size,
    int64_t * colptr,
    int * rowidx)
{
    if (object != mtxfile_matrix)
        return MTX_ERR_INCOMPATIBLE_MTX_OBJECT;
    if (format != mtxfile_coordinate)
        return MTX_ERR_INCOMPATIBLE_MTX_FORMAT;
    for (int j = 0; j <= num_columns; j++)
        colptr[j] = 0;
    if (field == mtxfile_real) {
        if (precision == mtx_single) {
            for (int64_t k = 0; k < size; k++)
                colptr[data->matrix_coordinate_real_single[k].j]++;
        } else if (precision == mtx_double) {
            for (int64_t k = 0; k < size; k++)
                colptr[data->matrix_coordinate_real_double[k].j]++;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (field == mtxfile_complex) {
        if (precision == mtx_single) {
            for (int64_t k = 0; k < size; k++)
                colptr[data->matrix_coordinate_complex_single[k].j]++;
        } else if (precision == mtx_double) {
            for (int64_t k = 0; k < size; k++)
                colptr[data->matrix_coordinate_complex_double[k].j]++;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (field == mtxfile_integer) {
        if (precision == mtx_single) {
            for (int64_t k = 0; k < size; k++)
                colptr[data->matrix_coordinate_integer_single[k].j]++;
        } else if (precision == mtx_double) {
            for (int64_t k = 0; k < size; k++)
                colptr[data->matrix_coordinate_integer_double[k].j]++;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (field == mtxfile_pattern) {
        for (int64_t k = 0; k < size; k++)
            colptr[data->matrix_coordinate_pattern[k].j]++;
    } else {
        return MTX_ERR_INVALID_MTX_FIELD;
    }
    for (int j = 1; j <= num_columns; j++)
        colptr[j] += colptr[j-1];

    if (!rowidx)
        return MTX_SUCCESS;

    /* Allocate storage for auxiliary row pointers. */
    int64_t * colendptr = malloc((num_columns+1) * sizeof(int64_t));
    if (!colendptr)
        return MTX_ERR_ERRNO;
    for (int j = 0; j <= num_columns; j++)
        colendptr[j] = colptr[j];

    /* Sort nonzeros in column major order, keeping only the row indices. */
    if (field == mtxfile_real) {
        if (precision == mtx_single) {
            const struct mtxfile_matrix_coordinate_real_single * src =
                data->matrix_coordinate_real_single;
            for (int64_t k = 0; k < size; k++)
                rowidx[k] = src[k].i;
            for (int64_t k = 0; k < size; k++) {
                int j = src[k].j-1;
                int64_t l = colendptr[j]-1;
                while (l >= colptr[j] && rowidx[l] > src[k].i) {
                    rowidx[l+1] = rowidx[l];
                    l--;
                }
                rowidx[l+1] = src[k].i;
                colendptr[j]++;
            }
        } else if (precision == mtx_double) {
            const struct mtxfile_matrix_coordinate_real_double * src =
                data->matrix_coordinate_real_double;
            for (int64_t k = 0; k < size; k++)
                rowidx[k] = src[k].i;
            for (int64_t k = 0; k < size; k++) {
                int j = src[k].j-1;
                int64_t l = colendptr[j]-1;
                while (l >= colptr[j] && rowidx[l] > src[k].i) {
                    rowidx[l+1] = rowidx[l];
                    l--;
                }
                rowidx[l+1] = src[k].i;
                colendptr[j]++;
            }
        } else {
            free(colendptr);
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (field == mtxfile_complex) {
        if (precision == mtx_single) {
            const struct mtxfile_matrix_coordinate_complex_single * src =
                data->matrix_coordinate_complex_single;
            for (int64_t k = 0; k < size; k++)
                rowidx[k] = src[k].i;
            for (int64_t k = 0; k < size; k++) {
                int j = src[k].j-1;
                int64_t l = colendptr[j]-1;
                while (l >= colptr[j] && rowidx[l] > src[k].i) {
                    rowidx[l+1] = rowidx[l];
                    l--;
                }
                rowidx[l+1] = src[k].i;
                colendptr[j]++;
            }
        } else if (precision == mtx_double) {
            const struct mtxfile_matrix_coordinate_complex_double * src =
                data->matrix_coordinate_complex_double;
            for (int64_t k = 0; k < size; k++)
                rowidx[k] = src[k].i;
            for (int64_t k = 0; k < size; k++) {
                int j = src[k].j-1;
                int64_t l = colendptr[j]-1;
                while (l >= colptr[j] && rowidx[l] > src[k].i) {
                    rowidx[l+1] = rowidx[l];
                    l--;
                }
                rowidx[l+1] = src[k].i;
                colendptr[j]++;
            }
        } else {
            free(colendptr);
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (field == mtxfile_integer) {
        if (precision == mtx_single) {
            const struct mtxfile_matrix_coordinate_integer_single * src =
                data->matrix_coordinate_integer_single;
            for (int64_t k = 0; k < size; k++)
                rowidx[k] = src[k].i;
            for (int64_t k = 0; k < size; k++) {
                int j = src[k].j-1;
                int64_t l = colendptr[j]-1;
                while (l >= colptr[j] && rowidx[l] > src[k].i) {
                    rowidx[l+1] = rowidx[l];
                    l--;
                }
                rowidx[l+1] = src[k].i;
                colendptr[j]++;
            }
        } else if (precision == mtx_double) {
            const struct mtxfile_matrix_coordinate_integer_double * src =
                data->matrix_coordinate_integer_double;
            for (int64_t k = 0; k < size; k++)
                rowidx[k] = src[k].i;
            for (int64_t k = 0; k < size; k++) {
                int j = src[k].j-1;
                int64_t l = colendptr[j]-1;
                while (l >= colptr[j] && rowidx[l] > src[k].i) {
                    rowidx[l+1] = rowidx[l];
                    l--;
                }
                rowidx[l+1] = src[k].i;
                colendptr[j]++;
            }
        } else {
            free(colendptr);
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (field == mtxfile_pattern) {
        const struct mtxfile_matrix_coordinate_pattern * src =
            data->matrix_coordinate_pattern;
        for (int64_t k = 0; k < size; k++)
            rowidx[k] = src[k].i;
        for (int64_t k = 0; k < size; k++) {
            int j = src[k].j-1;
            int64_t l = colendptr[j]-1;
            while (l >= colptr[j] && rowidx[l] > src[k].i) {
                rowidx[l+1] = rowidx[l];
                l--;
            }
            rowidx[l+1] = src[k].i;
            colendptr[j]++;
        }
    } else {
        free(colendptr);
        return MTX_ERR_INVALID_MTX_FIELD;
    }
    free(colendptr);
    return MTX_SUCCESS;
}

/*
 * Modifying values
 */

/**
 * `mtxfile_data_set_constant_real_single()' sets every (nonzero)
 * value of a matrix or vector equal to a constant, single precision
 * floating point number.
 */
int mtxfile_data_set_constant_real_single(
    union mtxfile_data * data,
    enum mtxfile_object object,
    enum mtxfile_format format,
    enum mtxfile_field field,
    enum mtx_precision precision,
    int64_t size,
    int64_t offset,
    float a)
{
    if (format == mtxfile_array) {
        if (field == mtxfile_real) {
            if (precision == mtx_single) {
                for (int64_t k = 0; k < size; k++)
                    data->array_real_single[offset+k] = a;
            } else if (precision == mtx_double) {
                for (int64_t k = 0; k < size; k++)
                    data->array_real_double[offset+k] = a;
            } else {
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_complex) {
            if (precision == mtx_single) {
                for (int64_t k = 0; k < size; k++) {
                    data->array_complex_single[offset+k][0] = a;
                    data->array_complex_single[offset+k][1] = 0;
                }
            } else if (precision == mtx_double) {
                for (int64_t k = 0; k < size; k++) {
                    data->array_complex_double[offset+k][0] = a;
                    data->array_complex_double[offset+k][1] = 0;
                }
            } else {
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_integer) {
            if (precision == mtx_single) {
                for (int64_t k = 0; k < size; k++)
                    data->array_integer_single[offset+k] = a;
            } else if (precision == mtx_double) {
                for (int64_t k = 0; k < size; k++)
                    data->array_integer_double[offset+k] = a;
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
                    for (int64_t k = 0; k < size; k++)
                        data->matrix_coordinate_real_single[offset+k].a = a;
                } else if (precision == mtx_double) {
                    for (int64_t k = 0; k < size; k++)
                        data->matrix_coordinate_real_double[offset+k].a = a;
                } else {
                    return MTX_ERR_INVALID_PRECISION;
                }
            } else if (field == mtxfile_complex) {
                if (precision == mtx_single) {
                    for (int64_t k = 0; k < size; k++) {
                        data->matrix_coordinate_complex_single[offset+k].a[0] = a;
                        data->matrix_coordinate_complex_single[offset+k].a[1] = 0;
                    }
                } else if (precision == mtx_double) {
                    for (int64_t k = 0; k < size; k++) {
                        data->matrix_coordinate_complex_double[offset+k].a[0] = a;
                        data->matrix_coordinate_complex_double[offset+k].a[1] = 0;
                    }
                } else {
                    return MTX_ERR_INVALID_PRECISION;
                }
            } else if (field == mtxfile_integer) {
                if (precision == mtx_single) {
                    for (int64_t k = 0; k < size; k++)
                        data->matrix_coordinate_integer_single[offset+k].a = a;
                } else if (precision == mtx_double) {
                    for (int64_t k = 0; k < size; k++)
                        data->matrix_coordinate_integer_double[offset+k].a = a;
                } else {
                    return MTX_ERR_INVALID_PRECISION;
                }
            } else if (field == mtxfile_pattern) {
                /* Nothing to be done */
            } else {
                return MTX_ERR_INVALID_MTX_FIELD;
            }
        } else if (object == mtxfile_vector) {
            if (field == mtxfile_real) {
                if (precision == mtx_single) {
                    for (int64_t k = 0; k < size; k++)
                        data->vector_coordinate_real_single[offset+k].a = a;
                } else if (precision == mtx_double) {
                    for (int64_t k = 0; k < size; k++)
                        data->vector_coordinate_real_double[offset+k].a = a;
                } else {
                    return MTX_ERR_INVALID_PRECISION;
                }
            } else if (field == mtxfile_complex) {
                if (precision == mtx_single) {
                    for (int64_t k = 0; k < size; k++) {
                        data->vector_coordinate_complex_single[offset+k].a[0] = a;
                        data->vector_coordinate_complex_single[offset+k].a[1] = 0;
                    }
                } else if (precision == mtx_double) {
                    for (int64_t k = 0; k < size; k++) {
                        data->vector_coordinate_complex_double[offset+k].a[0] = a;
                        data->vector_coordinate_complex_double[offset+k].a[1] = a;
                    }
                } else {
                    return MTX_ERR_INVALID_PRECISION;
                }
            } else if (field == mtxfile_integer) {
                if (precision == mtx_single) {
                    for (int64_t k = 0; k < size; k++)
                        data->vector_coordinate_integer_single[offset+k].a = a;
                } else if (precision == mtx_double) {
                    for (int64_t k = 0; k < size; k++)
                        data->vector_coordinate_integer_double[offset+k].a = a;
                } else {
                    return MTX_ERR_INVALID_PRECISION;
                }
            } else if (field == mtxfile_pattern) {
                /* Nothing to be done */
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
 * `mtxfile_data_set_constant_real_double()' sets every (nonzero)
 * value of a matrix or vector equal to a constant, double precision
 * floating point number.
 */
int mtxfile_data_set_constant_real_double(
    union mtxfile_data * data,
    enum mtxfile_object object,
    enum mtxfile_format format,
    enum mtxfile_field field,
    enum mtx_precision precision,
    int64_t size,
    int64_t offset,
    double a);

/**
 * `mtxfile_data_set_constant_complex_single()' sets every (nonzero)
 * value of a matrix or vector equal to a constant, single precision
 * floating point complex number.
 */
int mtxfile_data_set_constant_complex_single(
    union mtxfile_data * data,
    enum mtxfile_object object,
    enum mtxfile_format format,
    enum mtxfile_field field,
    enum mtx_precision precision,
    int64_t size,
    int64_t offset,
    float a[2]);

/**
 * `mtxfile_data_set_constant_integer_single()' sets every (nonzero)
 * value of a matrix or vector equal to a constant integer.
 */
int mtxfile_data_set_constant_integer_single(
    union mtxfile_data * data,
    enum mtxfile_object object,
    enum mtxfile_format format,
    enum mtxfile_field field,
    enum mtx_precision precision,
    int64_t size,
    int64_t offset,
    int32_t a);

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
    int64_t i)
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
 * `offset+size' elements.
 *
 * At most `size' lines are read from the stream and written to the
 * appropriate array of the `data' union, starting `offset' elements
 * from the beginning of the array.
 *
 * If an error code is returned, then `lines_read' and `bytes_read'
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the Matrix Market file.
 *
 * During parsing, the locale is temporarily changed to "C" to ensure
 * that locale-specific settings, such as the type of decimal point,
 * do not affect parsing.
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
    int64_t size,
    int64_t offset)
{
    int err;
    bool free_linebuf = !linebuf;
    if (!linebuf) {
        line_max = sysconf(_SC_LINE_MAX);
        linebuf = malloc(line_max+1);
        if (!linebuf)
            return MTX_ERR_ERRNO;
    }

    /* Set the locale to "C" to ensure that locale-specific settings,
     * such as the type of decimal point, does not affect parsing. */
    char * locale;
    locale = strdup(setlocale(LC_ALL, NULL));
    if (!locale) {
        if (free_linebuf)
            free(linebuf);
        return MTX_ERR_ERRNO;
    }
    setlocale(LC_ALL, "C");

    for (int64_t i = 0; i < size; i++) {
        err = freadline(linebuf, line_max, f);
        if (err) {
            int olderrno = errno;
            setlocale(LC_ALL, locale);
            errno = olderrno;
            free(locale);
            if (free_linebuf)
                free(linebuf);
            return err;
        }

        err = mtxfile_parse_data(
            data, bytes_read, NULL, linebuf,
            object, format, field, precision,
            num_rows, num_columns, offset+i);
        if (err) {
            int olderrno = errno;
            setlocale(LC_ALL, locale);
            errno = olderrno;
            free(locale);
            if (free_linebuf)
                free(linebuf);
            return err;
        }
        if (lines_read)
            (*lines_read)++;
    }

    setlocale(LC_ALL, locale);
    free(locale);
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
 * `offset+size' elements.
 *
 * At most `size' lines are read from the stream and written to the
 * appropriate array of the `data' union, starting `offset' elements
 * from the beginning of the array.
 *
 * If an error code is returned, then `lines_read' and `bytes_read'
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the Matrix Market file.
 *
 * During parsing, the locale is temporarily changed to "C" to ensure
 * that locale-specific settings, such as the type of decimal point,
 * do not affect parsing.
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
    int64_t size,
    int64_t offset)
{
    int err;
    bool free_linebuf = !linebuf;
    if (!linebuf) {
        line_max = sysconf(_SC_LINE_MAX);
        linebuf = malloc(line_max+1);
        if (!linebuf)
            return MTX_ERR_ERRNO;
    }

    /* Set the locale to "C" to ensure that locale-specific settings,
     * such as the type of decimal point, does not affect parsing. */
    char * locale;
    locale = strdup(setlocale(LC_ALL, NULL));
    if (!locale) {
        if (free_linebuf)
            free(linebuf);
        return MTX_ERR_ERRNO;
    }
    setlocale(LC_ALL, "C");

    for (int64_t i = 0; i < size; i++) {
        err = gzreadline(linebuf, line_max, f);
        if (err) {
            int olderrno = errno;
            setlocale(LC_ALL, locale);
            errno = olderrno;
            free(locale);
            if (free_linebuf)
                free(linebuf);
            return err;
        }

        err = mtxfile_parse_data(
            data, bytes_read, NULL, linebuf,
            object, format, field, precision,
            num_rows, num_columns, offset+i);
        if (err) {
            int olderrno = errno;
            setlocale(LC_ALL, locale);
            errno = olderrno;
            free(locale);
            if (free_linebuf)
                free(linebuf);
            return err;
        }
        if (lines_read)
            (*lines_read)++;
    }

    setlocale(LC_ALL, locale);
    free(locale);
    if (free_linebuf)
        free(linebuf);
    return MTX_SUCCESS;
}
#endif

/**
 * `validate_format_string()' parses and validates a format string to
 * be used for outputting numerical values of a Matrix Market file.
 */
static int validate_format_string(
    const char * format_str,
    enum mtxfile_field field)
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
        ((field == mtxfile_real ||
          field == mtxfile_complex) &&
         (format.specifier != format_specifier_e &&
          format.specifier != format_specifier_E &&
          format.specifier != format_specifier_f &&
          format.specifier != format_specifier_F &&
          format.specifier != format_specifier_g &&
          format.specifier != format_specifier_G)) ||
        (field == mtxfile_integer &&
         (format.specifier != format_specifier_d)))
    {
        return MTX_ERR_INVALID_FORMAT_SPECIFIER;
    }
    return MTX_SUCCESS;
}

/**
 * `mtxfile_data_fwrite()' writes data lines of a Matrix Market file
 * to a stream.
 *
 * If ‘fmt’ is ‘NULL’, then the format specifier ‘%g’ is used to print
 * floating point numbers with with enough digits to ensure correct
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
 *
 * If it is not `NULL', then the number of bytes written to the stream
 * is returned in `bytes_written'.
 *
 * The locale is temporarily changed to "C" to ensure that
 * locale-specific settings, such as the type of decimal point, do not
 * affect output.
 */
int mtxfile_data_fwrite(
    const union mtxfile_data * data,
    enum mtxfile_object object,
    enum mtxfile_format format,
    enum mtxfile_field field,
    enum mtx_precision precision,
    int64_t size,
    FILE * f,
    const char * fmt,
    int64_t * bytes_written)
{
    int err, olderrno;
    if (fmt) {
        err = validate_format_string(fmt, field);
        if (err)
            return err;
    }

    /* Set the locale to "C" to ensure that locale-specific settings,
     * such as the type of decimal point, do not affect output. */
    char * locale;
    locale = strdup(setlocale(LC_ALL, NULL));
    if (!locale)
        return MTX_ERR_ERRNO;
    setlocale(LC_ALL, "C");

    err = MTX_SUCCESS;
    int ret;
    if (format == mtxfile_array) {
        if (field == mtxfile_real) {
            if (precision == mtx_single) {
                for (int64_t k = 0; k < size; k++) {
                    ret = fprintf(f, fmt ? fmt : "%.*g", FLT_DIG, data->array_real_single[k]);
                    if (ret < 0) { err = MTX_ERR_ERRNO; goto fwrite_exit; }
                    if (bytes_written) *bytes_written += ret;
                    ret = fputc('\n', f);
                    if (ret == EOF) { err = MTX_ERR_ERRNO; goto fwrite_exit; }
                    if (bytes_written) (*bytes_written)++;
                }
            } else if (precision == mtx_double) {
                for (int64_t k = 0; k < size; k++) {
                    ret = fprintf(f, fmt ? fmt : "%.*g", DBL_DIG, data->array_real_double[k]);
                    if (ret < 0) { err = MTX_ERR_ERRNO; goto fwrite_exit; }
                    if (bytes_written) *bytes_written += ret;
                    ret = fputc('\n', f);
                    if (ret == EOF) { err = MTX_ERR_ERRNO; goto fwrite_exit; }
                    if (bytes_written) (*bytes_written)++;
                }
            } else {
                err = MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_complex) {
            if (precision == mtx_single) {
                for (int64_t k = 0; k < size; k++) {
                    ret = fprintf(
                        f, fmt ? fmt : "%.*g", FLT_DIG, data->array_complex_single[k][0]);
                    if (ret < 0) { err = MTX_ERR_ERRNO; goto fwrite_exit; }
                    if (bytes_written) *bytes_written += ret;
                    ret = fputc(' ', f);
                    if (ret == EOF) { err = MTX_ERR_ERRNO; goto fwrite_exit; }
                    if (bytes_written) (*bytes_written)++;
                    ret = fprintf(
                        f, fmt ? fmt : "%.*g", FLT_DIG, data->array_complex_single[k][1]);
                    if (ret < 0) { err = MTX_ERR_ERRNO; goto fwrite_exit; }
                    if (bytes_written) *bytes_written += ret;
                    ret = fputc('\n', f);
                    if (ret < 0) { err = MTX_ERR_ERRNO; goto fwrite_exit; }
                    if (bytes_written) (*bytes_written)++;
                }
            } else if (precision == mtx_double) {
                for (int64_t k = 0; k < size; k++) {
                    ret = fprintf(
                        f, fmt ? fmt : "%.*g", DBL_DIG, data->array_complex_double[k][0]);
                    if (ret < 0) { err = MTX_ERR_ERRNO; goto fwrite_exit; }
                    if (bytes_written) *bytes_written += ret;
                    ret = fputc(' ', f);
                    if (ret == EOF) { err = MTX_ERR_ERRNO; goto fwrite_exit; }
                    if (bytes_written) (*bytes_written)++;
                    ret = fprintf(
                        f, fmt ? fmt : "%.*g", DBL_DIG, data->array_complex_double[k][1]);
                    if (ret < 0) { err = MTX_ERR_ERRNO; goto fwrite_exit; }
                    if (bytes_written) *bytes_written += ret;
                    ret = fputc('\n', f);
                    if (ret < 0) { err = MTX_ERR_ERRNO; goto fwrite_exit; }
                    if (bytes_written) (*bytes_written)++;
                }
            } else {
                err = MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_integer) {
            if (precision == mtx_single) {
                for (int64_t k = 0; k < size; k++) {
                    ret = fprintf(f, fmt ? fmt : "%d", data->array_integer_single[k]);
                    if (ret < 0) { err = MTX_ERR_ERRNO; goto fwrite_exit; }
                    if (bytes_written) *bytes_written += ret;
                    ret = fputc('\n', f);
                    if (ret == EOF) { err = MTX_ERR_ERRNO; goto fwrite_exit; }
                    if (bytes_written) (*bytes_written)++;
                }
            } else if (precision == mtx_double) {
                for (int64_t k = 0; k < size; k++) {
                    ret = fprintf(f, fmt ? fmt : "%"PRId64, data->array_integer_double[k]);
                    if (ret < 0) { err = MTX_ERR_ERRNO; goto fwrite_exit; }
                    if (bytes_written) *bytes_written += ret;
                    ret = fputc('\n', f);
                    if (ret == EOF) { err = MTX_ERR_ERRNO; goto fwrite_exit; }
                    if (bytes_written) (*bytes_written)++;
                }
            } else {
                err = MTX_ERR_INVALID_PRECISION;
            }
        } else {
            err = MTX_ERR_INVALID_MTX_FIELD;
        }

    } else if (format == mtxfile_coordinate) {
        if (object == mtxfile_matrix) {
            if (field == mtxfile_real) {
                if (precision == mtx_single) {
                    for (int64_t k = 0; k < size; k++) {
                        ret = fprintf(
                            f, "%d %d ",
                            data->matrix_coordinate_real_single[k].i,
                            data->matrix_coordinate_real_single[k].j);
                        if (ret < 0) { err = MTX_ERR_ERRNO; goto fwrite_exit; }
                        if (bytes_written) *bytes_written += ret;
                        ret = fprintf(
                            f, fmt ? fmt : "%.*g", FLT_DIG,
                            data->matrix_coordinate_real_single[k].a);
                        if (ret < 0) { err = MTX_ERR_ERRNO; goto fwrite_exit; }
                        if (bytes_written) *bytes_written += ret;
                        ret = fputc('\n', f);
                        if (ret == EOF) { err = MTX_ERR_ERRNO; goto fwrite_exit; }
                        if (bytes_written) (*bytes_written)++;
                    }
                } else if (precision == mtx_double) {
                    for (int64_t k = 0; k < size; k++) {
                        ret = fprintf(
                            f, "%d %d ",
                            data->matrix_coordinate_real_double[k].i,
                            data->matrix_coordinate_real_double[k].j);
                        if (ret < 0) { err = MTX_ERR_ERRNO; goto fwrite_exit; }
                        if (bytes_written) *bytes_written += ret;
                        ret = fprintf(
                            f, fmt ? fmt : "%.*g", DBL_DIG,
                            data->matrix_coordinate_real_double[k].a);
                        if (ret < 0) { err = MTX_ERR_ERRNO; goto fwrite_exit; }
                        if (bytes_written) *bytes_written += ret;
                        ret = fputc('\n', f);
                        if (ret == EOF) { err = MTX_ERR_ERRNO; goto fwrite_exit; }
                        if (bytes_written) (*bytes_written)++;
                    }
                } else {
                    err = MTX_ERR_INVALID_PRECISION;
                }
            } else if (field == mtxfile_complex) {
                if (precision == mtx_single) {
                    for (int64_t k = 0; k < size; k++) {
                        ret = fprintf(
                            f, "%d %d ",
                            data->matrix_coordinate_complex_single[k].i,
                            data->matrix_coordinate_complex_single[k].j);
                        if (ret < 0) { err = MTX_ERR_ERRNO; goto fwrite_exit; }
                        if (bytes_written) *bytes_written += ret;
                        ret = fprintf(
                            f, fmt ? fmt : "%.*g", FLT_DIG,
                            data->matrix_coordinate_complex_single[k].a[0]);
                        if (ret < 0) { err = MTX_ERR_ERRNO; goto fwrite_exit; }
                        if (bytes_written) *bytes_written += ret;
                        ret = fputc(' ', f);
                        if (ret == EOF) { err = MTX_ERR_ERRNO; goto fwrite_exit; }
                        if (bytes_written) (*bytes_written)++;
                        ret = fprintf(
                            f, fmt ? fmt : "%.*g", FLT_DIG,
                            data->matrix_coordinate_complex_single[k].a[1]);
                        if (ret < 0) { err = MTX_ERR_ERRNO; goto fwrite_exit; }
                        if (bytes_written) *bytes_written += ret;
                        ret = fputc('\n', f);
                        if (ret == EOF) { err = MTX_ERR_ERRNO; goto fwrite_exit; }
                        if (bytes_written) (*bytes_written)++;
                    }
                } else if (precision == mtx_double) {
                    for (int64_t k = 0; k < size; k++) {
                        ret = fprintf(
                            f, "%d %d ",
                            data->matrix_coordinate_complex_double[k].i,
                            data->matrix_coordinate_complex_double[k].j);
                        if (ret < 0) { err = MTX_ERR_ERRNO; goto fwrite_exit; }
                        if (bytes_written) *bytes_written += ret;
                        ret = fprintf(
                            f, fmt ? fmt : "%.*g", DBL_DIG,
                            data->matrix_coordinate_complex_double[k].a[0]);
                        if (ret < 0) { err = MTX_ERR_ERRNO; goto fwrite_exit; }
                        if (bytes_written) *bytes_written += ret;
                        ret = fputc(' ', f);
                        if (ret == EOF) { err = MTX_ERR_ERRNO; goto fwrite_exit; }
                        if (bytes_written) (*bytes_written)++;
                        ret = fprintf(
                            f, fmt ? fmt : "%.*g", DBL_DIG,
                            data->matrix_coordinate_complex_double[k].a[1]);
                        if (ret < 0) { err = MTX_ERR_ERRNO; goto fwrite_exit; }
                        if (bytes_written) *bytes_written += ret;
                        ret = fputc('\n', f);
                        if (ret == EOF) { err = MTX_ERR_ERRNO; goto fwrite_exit; }
                        if (bytes_written) (*bytes_written)++;
                    }
                } else {
                    err = MTX_ERR_INVALID_PRECISION;
                }
            } else if (field == mtxfile_integer) {
                if (precision == mtx_single) {
                    for (int64_t k = 0; k < size; k++) {
                        ret = fprintf(
                            f, "%d %d ",
                            data->matrix_coordinate_integer_single[k].i,
                            data->matrix_coordinate_integer_single[k].j);
                        if (ret < 0) { err = MTX_ERR_ERRNO; goto fwrite_exit; }
                        if (bytes_written) *bytes_written += ret;
                        ret = fprintf(
                            f, fmt ? fmt : "%d",
                            data->matrix_coordinate_integer_single[k].a);
                        if (ret < 0) { err = MTX_ERR_ERRNO; goto fwrite_exit; }
                        if (bytes_written) *bytes_written += ret;
                        ret = fputc('\n', f);
                        if (ret == EOF) { err = MTX_ERR_ERRNO; goto fwrite_exit; }
                        if (bytes_written) (*bytes_written)++;
                    }
                } else if (precision == mtx_double) {
                    for (int64_t k = 0; k < size; k++) {
                        ret = fprintf(
                            f, "%d %d ",
                            data->matrix_coordinate_integer_double[k].i,
                            data->matrix_coordinate_integer_double[k].j);
                        if (ret < 0) { err = MTX_ERR_ERRNO; goto fwrite_exit; }
                        if (bytes_written) *bytes_written += ret;
                        ret = fprintf(
                            f, fmt ? fmt : "%"PRId64,
                            data->matrix_coordinate_integer_double[k].a);
                        if (ret < 0) { err = MTX_ERR_ERRNO; goto fwrite_exit; }
                        if (bytes_written) *bytes_written += ret;
                        ret = fputc('\n', f);
                        if (ret == EOF) { err = MTX_ERR_ERRNO; goto fwrite_exit; }
                        if (bytes_written) (*bytes_written)++;
                    }
                } else {
                    err = MTX_ERR_INVALID_PRECISION;
                }
            } else if (field == mtxfile_pattern) {
                for (int64_t k = 0; k < size; k++) {
                    ret = fprintf(
                        f, "%d %d\n",
                        data->matrix_coordinate_pattern[k].i,
                        data->matrix_coordinate_pattern[k].j);
                    if (ret < 0) { err = MTX_ERR_ERRNO; goto fwrite_exit; }
                    if (bytes_written) *bytes_written += ret;
                }
            } else {
                err = MTX_ERR_INVALID_MTX_FIELD;
            }

        } else if (object == mtxfile_vector) {
            if (field == mtxfile_real) {
                if (precision == mtx_single) {
                    for (int64_t k = 0; k < size; k++) {
                        ret = fprintf(
                            f, "%d ", data->vector_coordinate_real_single[k].i);
                        if (ret < 0) { err = MTX_ERR_ERRNO; goto fwrite_exit; }
                        if (bytes_written) *bytes_written += ret;
                        ret = fprintf(
                            f, fmt ? fmt : "%.*g", FLT_DIG,
                            data->vector_coordinate_real_single[k].a);
                        if (ret < 0) { err = MTX_ERR_ERRNO; goto fwrite_exit; }
                        if (bytes_written) *bytes_written += ret;
                        ret = fputc('\n', f);
                        if (ret == EOF) { err = MTX_ERR_ERRNO; goto fwrite_exit; }
                        if (bytes_written) (*bytes_written)++;
                    }
                } else if (precision == mtx_double) {
                    for (int64_t k = 0; k < size; k++) {
                        ret = fprintf(
                            f, "%d ", data->vector_coordinate_real_double[k].i);
                        if (ret < 0) { err = MTX_ERR_ERRNO; goto fwrite_exit; }
                        if (bytes_written) *bytes_written += ret;
                        ret = fprintf(
                            f, fmt ? fmt : "%.*g", DBL_DIG,
                            data->vector_coordinate_real_double[k].a);
                        if (ret < 0) { err = MTX_ERR_ERRNO; goto fwrite_exit; }
                        if (bytes_written) *bytes_written += ret;
                        ret = fputc('\n', f);
                        if (ret == EOF) { err = MTX_ERR_ERRNO; goto fwrite_exit; }
                        if (bytes_written) (*bytes_written)++;
                    }
                } else {
                    err = MTX_ERR_INVALID_PRECISION;
                }
            } else if (field == mtxfile_complex) {
                if (precision == mtx_single) {
                    for (int64_t k = 0; k < size; k++) {
                        ret = fprintf(
                            f, "%d ", data->vector_coordinate_complex_single[k].i);
                        if (ret < 0) { err = MTX_ERR_ERRNO; goto fwrite_exit; }
                        if (bytes_written) *bytes_written += ret;
                        ret = fprintf(
                            f, fmt ? fmt : "%.*g", FLT_DIG,
                            data->vector_coordinate_complex_single[k].a[0]);
                        if (ret < 0) { err = MTX_ERR_ERRNO; goto fwrite_exit; }
                        if (bytes_written) *bytes_written += ret;
                        ret = fputc(' ', f);
                        if (ret == EOF) { err = MTX_ERR_ERRNO; goto fwrite_exit; }
                        if (bytes_written) (*bytes_written)++;
                        ret = fprintf(
                            f, fmt ? fmt : "%.*g", FLT_DIG,
                            data->vector_coordinate_complex_single[k].a[1]);
                        if (ret < 0) { err = MTX_ERR_ERRNO; goto fwrite_exit; }
                        if (bytes_written) *bytes_written += ret;
                        ret = fputc('\n', f);
                        if (ret == EOF) { err = MTX_ERR_ERRNO; goto fwrite_exit; }
                        if (bytes_written) (*bytes_written)++;
                    }
                } else if (precision == mtx_double) {
                    for (int64_t k = 0; k < size; k++) {
                        ret = fprintf(
                            f, "%d ", data->vector_coordinate_complex_double[k].i);
                        if (ret < 0) { err = MTX_ERR_ERRNO; goto fwrite_exit; }
                        if (bytes_written) *bytes_written += ret;
                        ret = fprintf(
                            f, fmt ? fmt : "%.*g", DBL_DIG,
                            data->vector_coordinate_complex_double[k].a[0]);
                        if (ret < 0) { err = MTX_ERR_ERRNO; goto fwrite_exit; }
                        if (bytes_written) *bytes_written += ret;
                        ret = fputc(' ', f);
                        if (ret == EOF) { err = MTX_ERR_ERRNO; goto fwrite_exit; }
                        if (bytes_written) (*bytes_written)++;
                        ret = fprintf(
                            f, fmt ? fmt : "%.*g", DBL_DIG,
                            data->vector_coordinate_complex_double[k].a[1]);
                        if (ret < 0) { err = MTX_ERR_ERRNO; goto fwrite_exit; }
                        if (bytes_written) *bytes_written += ret;
                        ret = fputc('\n', f);
                        if (ret == EOF) { err = MTX_ERR_ERRNO; goto fwrite_exit; }
                        if (bytes_written) (*bytes_written)++;
                    }
                } else {
                    err = MTX_ERR_INVALID_PRECISION;
                }
            } else if (field == mtxfile_integer) {
                if (precision == mtx_single) {
                    for (int64_t k = 0; k < size; k++) {
                        ret = fprintf(
                            f, "%d ", data->vector_coordinate_integer_single[k].i);
                        if (ret < 0) { err = MTX_ERR_ERRNO; goto fwrite_exit; }
                        if (bytes_written) *bytes_written += ret;
                        ret = fprintf(
                            f, fmt ? fmt : "%d",
                            data->vector_coordinate_integer_single[k].a);
                        if (ret < 0) { err = MTX_ERR_ERRNO; goto fwrite_exit; }
                        if (bytes_written) *bytes_written += ret;
                        ret = fputc('\n', f);
                        if (ret == EOF) { err = MTX_ERR_ERRNO; goto fwrite_exit; }
                        if (bytes_written) (*bytes_written)++;
                    }
                } else if (precision == mtx_double) {
                    for (int64_t k = 0; k < size; k++) {
                        ret = fprintf(
                            f, "%d ", data->vector_coordinate_integer_double[k].i);
                        if (ret < 0) { err = MTX_ERR_ERRNO; goto fwrite_exit; }
                        if (bytes_written) *bytes_written += ret;
                        ret = fprintf(
                            f, fmt ? fmt : "%"PRId64,
                            data->vector_coordinate_integer_double[k].a);
                        if (ret < 0) { err = MTX_ERR_ERRNO; goto fwrite_exit; }
                        if (bytes_written) *bytes_written += ret;
                        ret = fputc('\n', f);
                        if (ret == EOF) { err = MTX_ERR_ERRNO; goto fwrite_exit; }
                        if (bytes_written) (*bytes_written)++;
                    }
                } else {
                    err = MTX_ERR_INVALID_PRECISION;
                }
            } else if (field == mtxfile_pattern) {
                for (int64_t k = 0; k < size; k++) {
                    ret = fprintf(
                        f, "%d\n", data->vector_coordinate_pattern[k].i);
                    if (ret < 0) { err = MTX_ERR_ERRNO; goto fwrite_exit; }
                    if (bytes_written) *bytes_written += ret;
                }
            } else {
                err = MTX_ERR_INVALID_MTX_FIELD;
            }
        } else {
            err = MTX_ERR_INVALID_MTX_OBJECT;
        }
    } else {
        err = MTX_ERR_INVALID_MTX_FORMAT;
    }

fwrite_exit:
    olderrno = errno;
    setlocale(LC_ALL, locale);
    errno = olderrno;
    free(locale);
    return err;
}

#ifdef LIBMTX_HAVE_LIBZ
/**
 * `mtxfile_data_gzwrite()' writes data lines of a Matrix Market file
 * to a gzip-compressed stream.
 *
 * If ‘fmt’ is ‘NULL’, then the format specifier ‘%g’ is used to print
 * floating point numbers with with enough digits to ensure correct
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
 *
 * If it is not `NULL', then the number of bytes written to the stream
 * is returned in `bytes_written'.
 *
 * The locale is temporarily changed to "C" to ensure that
 * locale-specific settings, such as the type of decimal point, do not
 * affect output.
 */
int mtxfile_data_gzwrite(
    const union mtxfile_data * data,
    enum mtxfile_object object,
    enum mtxfile_format format,
    enum mtxfile_field field,
    enum mtx_precision precision,
    int64_t size,
    gzFile f,
    const char * fmt,
    int64_t * bytes_written)
{
    int err, olderrno;
    if (fmt) {
        err = validate_format_string(fmt, field);
        if (err)
            return err;
    }

    /* Set the locale to "C" to ensure that locale-specific settings,
     * such as the type of decimal point, do not affect output. */
    char * locale;
    locale = strdup(setlocale(LC_ALL, NULL));
    if (!locale)
        return MTX_ERR_ERRNO;
    setlocale(LC_ALL, "C");

    err = MTX_SUCCESS;
    int ret;
    if (format == mtxfile_array) {
        if (field == mtxfile_real) {
            if (precision == mtx_single) {
                for (int64_t k = 0; k < size; k++) {
                    ret = gzprintf(f, fmt ? fmt : "%.*g", FLT_DIG, data->array_real_single[k]);
                    if (ret < 0) { err = MTX_ERR_ERRNO; goto gzwrite_exit; }
                    if (bytes_written) *bytes_written += ret;
                    ret = gzputc(f, '\n');
                    if (ret == EOF) { err = MTX_ERR_ERRNO; goto gzwrite_exit; }
                    if (bytes_written) (*bytes_written)++;
                }
            } else if (precision == mtx_double) {
                for (int64_t k = 0; k < size; k++) {
                    ret = gzprintf(f, fmt ? fmt : "%.*g", DBL_DIG, data->array_real_double[k]);
                    if (ret < 0) { err = MTX_ERR_ERRNO; goto gzwrite_exit; }
                    if (bytes_written) *bytes_written += ret;
                    ret = gzputc(f, '\n');
                    if (ret == EOF) { err = MTX_ERR_ERRNO; goto gzwrite_exit; }
                    if (bytes_written) (*bytes_written)++;
                }
            } else {
                err = MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_complex) {
            if (precision == mtx_single) {
                for (int64_t k = 0; k < size; k++) {
                    ret = gzprintf(
                        f, fmt ? fmt : "%.*g", FLT_DIG, data->array_complex_single[k][0]);
                    if (ret < 0) { err = MTX_ERR_ERRNO; goto gzwrite_exit; }
                    if (bytes_written) *bytes_written += ret;
                    ret = gzputc(f, ' ');
                    if (ret == EOF) { err = MTX_ERR_ERRNO; goto gzwrite_exit; }
                    if (bytes_written) (*bytes_written)++;
                    ret = gzprintf(
                        f, fmt ? fmt : "%.*g", FLT_DIG, data->array_complex_single[k][1]);
                    if (ret < 0) { err = MTX_ERR_ERRNO; goto gzwrite_exit; }
                    if (bytes_written) *bytes_written += ret;
                    ret = gzputc(f, '\n');
                    if (ret < 0) { err = MTX_ERR_ERRNO; goto gzwrite_exit; }
                    if (bytes_written) (*bytes_written)++;
                }
            } else if (precision == mtx_double) {
                for (int64_t k = 0; k < size; k++) {
                    ret = gzprintf(
                        f, fmt ? fmt : "%.*g", DBL_DIG, data->array_complex_double[k][0]);
                    if (ret < 0) { err = MTX_ERR_ERRNO; goto gzwrite_exit; }
                    if (bytes_written) *bytes_written += ret;
                    ret = gzputc(f, ' ');
                    if (ret == EOF) { err = MTX_ERR_ERRNO; goto gzwrite_exit; }
                    if (bytes_written) (*bytes_written)++;
                    ret = gzprintf(
                        f, fmt ? fmt : "%.*g", DBL_DIG, data->array_complex_double[k][1]);
                    if (ret < 0) { err = MTX_ERR_ERRNO; goto gzwrite_exit; }
                    if (bytes_written) *bytes_written += ret;
                    ret = gzputc(f, '\n');
                    if (ret < 0) { err = MTX_ERR_ERRNO; goto gzwrite_exit; }
                    if (bytes_written) (*bytes_written)++;
                }
            } else {
                err = MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_integer) {
            if (precision == mtx_single) {
                for (int64_t k = 0; k < size; k++) {
                    ret = gzprintf(f, fmt ? fmt : "%d", data->array_integer_single[k]);
                    if (ret < 0) { err = MTX_ERR_ERRNO; goto gzwrite_exit; }
                    if (bytes_written) *bytes_written += ret;
                    ret = gzputc(f, '\n');
                    if (ret == EOF) { err = MTX_ERR_ERRNO; goto gzwrite_exit; }
                    if (bytes_written) (*bytes_written)++;
                }
            } else if (precision == mtx_double) {
                for (int64_t k = 0; k < size; k++) {
                    ret = gzprintf(f, fmt ? fmt : "%d", data->array_integer_double[k]);
                    if (ret < 0) { err = MTX_ERR_ERRNO; goto gzwrite_exit; }
                    if (bytes_written) *bytes_written += ret;
                    ret = gzputc(f, '\n');
                    if (ret == EOF) { err = MTX_ERR_ERRNO; goto gzwrite_exit; }
                    if (bytes_written) (*bytes_written)++;
                }
            } else {
                err = MTX_ERR_INVALID_PRECISION;
            }
        } else {
            err = MTX_ERR_INVALID_MTX_FIELD;
        }

    } else if (format == mtxfile_coordinate) {
        if (object == mtxfile_matrix) {
            if (field == mtxfile_real) {
                if (precision == mtx_single) {
                    for (int64_t k = 0; k < size; k++) {
                        ret = gzprintf(
                            f, "%d %d ",
                            data->matrix_coordinate_real_single[k].i,
                            data->matrix_coordinate_real_single[k].j);
                        if (ret < 0) { err = MTX_ERR_ERRNO; goto gzwrite_exit; }
                        if (bytes_written) *bytes_written += ret;
                        ret = gzprintf(
                            f, fmt ? fmt : "%.*g", FLT_DIG,
                            data->matrix_coordinate_real_single[k].a);
                        if (ret < 0) { err = MTX_ERR_ERRNO; goto gzwrite_exit; }
                        if (bytes_written) *bytes_written += ret;
                        ret = gzputc(f, '\n');
                        if (ret == EOF) { err = MTX_ERR_ERRNO; goto gzwrite_exit; }
                        if (bytes_written) (*bytes_written)++;
                    }
                } else if (precision == mtx_double) {
                    for (int64_t k = 0; k < size; k++) {
                        ret = gzprintf(
                            f, "%d %d ",
                            data->matrix_coordinate_real_double[k].i,
                            data->matrix_coordinate_real_double[k].j);
                        if (ret < 0) { err = MTX_ERR_ERRNO; goto gzwrite_exit; }
                        if (bytes_written) *bytes_written += ret;
                        ret = gzprintf(
                            f, fmt ? fmt : "%.*g", DBL_DIG,
                            data->matrix_coordinate_real_double[k].a);
                        if (ret < 0) { err = MTX_ERR_ERRNO; goto gzwrite_exit; }
                        if (bytes_written) *bytes_written += ret;
                        ret = gzputc(f, '\n');
                        if (ret == EOF) { err = MTX_ERR_ERRNO; goto gzwrite_exit; }
                        if (bytes_written) (*bytes_written)++;
                    }
                } else {
                    err = MTX_ERR_INVALID_PRECISION;
                }
            } else if (field == mtxfile_complex) {
                if (precision == mtx_single) {
                    for (int64_t k = 0; k < size; k++) {
                        ret = gzprintf(
                            f, "%d %d ",
                            data->matrix_coordinate_complex_single[k].i,
                            data->matrix_coordinate_complex_single[k].j);
                        if (ret < 0) { err = MTX_ERR_ERRNO; goto gzwrite_exit; }
                        if (bytes_written) *bytes_written += ret;
                        ret = gzprintf(
                            f, fmt ? fmt : "%.*g", FLT_DIG,
                            data->matrix_coordinate_complex_single[k].a[0]);
                        if (ret < 0) { err = MTX_ERR_ERRNO; goto gzwrite_exit; }
                        if (bytes_written) *bytes_written += ret;
                        ret = gzputc(f, ' ');
                        if (ret == EOF) { err = MTX_ERR_ERRNO; goto gzwrite_exit; }
                        if (bytes_written) (*bytes_written)++;
                        ret = gzprintf(
                            f, fmt ? fmt : "%.*g", FLT_DIG,
                            data->matrix_coordinate_complex_single[k].a[1]);
                        if (ret < 0) { err = MTX_ERR_ERRNO; goto gzwrite_exit; }
                        if (bytes_written) *bytes_written += ret;
                        ret = gzputc(f, '\n');
                        if (ret == EOF) { err = MTX_ERR_ERRNO; goto gzwrite_exit; }
                        if (bytes_written) (*bytes_written)++;
                    }
                } else if (precision == mtx_double) {
                    for (int64_t k = 0; k < size; k++) {
                        ret = gzprintf(
                            f, "%d %d ",
                            data->matrix_coordinate_complex_double[k].i,
                            data->matrix_coordinate_complex_double[k].j);
                        if (ret < 0) { err = MTX_ERR_ERRNO; goto gzwrite_exit; }
                        if (bytes_written) *bytes_written += ret;
                        ret = gzprintf(
                            f, fmt ? fmt : "%.*g", DBL_DIG,
                            data->matrix_coordinate_complex_double[k].a[0]);
                        if (ret < 0) { err = MTX_ERR_ERRNO; goto gzwrite_exit; }
                        if (bytes_written) *bytes_written += ret;
                        ret = gzputc(f, ' ');
                        if (ret == EOF) { err = MTX_ERR_ERRNO; goto gzwrite_exit; }
                        if (bytes_written) (*bytes_written)++;
                        ret = gzprintf(
                            f, fmt ? fmt : "%.*g", DBL_DIG,
                            data->matrix_coordinate_complex_double[k].a[1]);
                        if (ret < 0) { err = MTX_ERR_ERRNO; goto gzwrite_exit; }
                        if (bytes_written) *bytes_written += ret;
                        ret = gzputc(f, '\n');
                        if (ret == EOF) { err = MTX_ERR_ERRNO; goto gzwrite_exit; }
                        if (bytes_written) (*bytes_written)++;
                    }
                } else {
                    err = MTX_ERR_INVALID_PRECISION;
                }
            } else if (field == mtxfile_integer) {
                if (precision == mtx_single) {
                    for (int64_t k = 0; k < size; k++) {
                        ret = gzprintf(
                            f, "%d %d ",
                            data->matrix_coordinate_integer_single[k].i,
                            data->matrix_coordinate_integer_single[k].j);
                        if (ret < 0) { err = MTX_ERR_ERRNO; goto gzwrite_exit; }
                        if (bytes_written) *bytes_written += ret;
                        ret = gzprintf(
                            f, fmt ? fmt : "%d",
                            data->matrix_coordinate_integer_single[k].a);
                        if (ret < 0) { err = MTX_ERR_ERRNO; goto gzwrite_exit; }
                        if (bytes_written) *bytes_written += ret;
                        ret = gzputc(f, '\n');
                        if (ret == EOF) { err = MTX_ERR_ERRNO; goto gzwrite_exit; }
                        if (bytes_written) (*bytes_written)++;
                    }
                } else if (precision == mtx_double) {
                    for (int64_t k = 0; k < size; k++) {
                        ret = gzprintf(
                            f, "%d %d ",
                            data->matrix_coordinate_integer_double[k].i,
                            data->matrix_coordinate_integer_double[k].j);
                        if (ret < 0) { err = MTX_ERR_ERRNO; goto gzwrite_exit; }
                        if (bytes_written) *bytes_written += ret;
                        ret = gzprintf(
                            f, fmt ? fmt : "%d",
                            data->matrix_coordinate_integer_double[k].a);
                        if (ret < 0) { err = MTX_ERR_ERRNO; goto gzwrite_exit; }
                        if (bytes_written) *bytes_written += ret;
                        ret = gzputc(f, '\n');
                        if (ret == EOF) { err = MTX_ERR_ERRNO; goto gzwrite_exit; }
                        if (bytes_written) (*bytes_written)++;
                    }
                } else {
                    err = MTX_ERR_INVALID_PRECISION;
                }
            } else if (field == mtxfile_pattern) {
                for (int64_t k = 0; k < size; k++) {
                    ret = gzprintf(
                        f, "%d %d\n",
                        data->matrix_coordinate_pattern[k].i,
                        data->matrix_coordinate_pattern[k].j);
                    if (ret < 0) { err = MTX_ERR_ERRNO; goto gzwrite_exit; }
                    if (bytes_written) *bytes_written += ret;
                }
            } else {
                err = MTX_ERR_INVALID_MTX_FIELD;
            }

        } else if (object == mtxfile_vector) {
            if (field == mtxfile_real) {
                if (precision == mtx_single) {
                    for (int64_t k = 0; k < size; k++) {
                        ret = gzprintf(
                            f, "%d ", data->vector_coordinate_real_single[k].i);
                        if (ret < 0) { err = MTX_ERR_ERRNO; goto gzwrite_exit; }
                        if (bytes_written) *bytes_written += ret;
                        ret = gzprintf(
                            f, fmt ? fmt : "%.*g", FLT_DIG,
                            data->vector_coordinate_real_single[k].a);
                        if (ret < 0) { err = MTX_ERR_ERRNO; goto gzwrite_exit; }
                        if (bytes_written) *bytes_written += ret;
                        ret = gzputc(f, '\n');
                        if (ret == EOF) { err = MTX_ERR_ERRNO; goto gzwrite_exit; }
                        if (bytes_written) (*bytes_written)++;
                    }
                } else if (precision == mtx_double) {
                    for (int64_t k = 0; k < size; k++) {
                        ret = gzprintf(
                            f, "%d ", data->vector_coordinate_real_double[k].i);
                        if (ret < 0) { err = MTX_ERR_ERRNO; goto gzwrite_exit; }
                        if (bytes_written) *bytes_written += ret;
                        ret = gzprintf(
                            f, fmt ? fmt : "%.*g", DBL_DIG,
                            data->vector_coordinate_real_double[k].a);
                        if (ret < 0) { err = MTX_ERR_ERRNO; goto gzwrite_exit; }
                        if (bytes_written) *bytes_written += ret;
                        ret = gzputc(f, '\n');
                        if (ret == EOF) { err = MTX_ERR_ERRNO; goto gzwrite_exit; }
                        if (bytes_written) (*bytes_written)++;
                    }
                } else {
                    err = MTX_ERR_INVALID_PRECISION;
                }
            } else if (field == mtxfile_complex) {
                if (precision == mtx_single) {
                    for (int64_t k = 0; k < size; k++) {
                        ret = gzprintf(
                            f, "%d ", data->vector_coordinate_complex_single[k].i);
                        if (ret < 0) { err = MTX_ERR_ERRNO; goto gzwrite_exit; }
                        if (bytes_written) *bytes_written += ret;
                        ret = gzprintf(
                            f, fmt ? fmt : "%.*g", FLT_DIG,
                            data->vector_coordinate_complex_single[k].a[0]);
                        if (ret < 0) { err = MTX_ERR_ERRNO; goto gzwrite_exit; }
                        if (bytes_written) *bytes_written += ret;
                        ret = gzprintf(
                            f, fmt ? fmt : "%.*g", FLT_DIG,
                            data->vector_coordinate_complex_single[k].a[1]);
                        if (ret < 0) { err = MTX_ERR_ERRNO; goto gzwrite_exit; }
                        if (bytes_written) *bytes_written += ret;
                        ret = gzputc(f, '\n');
                        if (ret == EOF) { err = MTX_ERR_ERRNO; goto gzwrite_exit; }
                        if (bytes_written) (*bytes_written)++;
                    }
                } else if (precision == mtx_double) {
                    for (int64_t k = 0; k < size; k++) {
                        ret = gzprintf(
                            f, "%d ", data->vector_coordinate_complex_double[k].i);
                        if (ret < 0) { err = MTX_ERR_ERRNO; goto gzwrite_exit; }
                        if (bytes_written) *bytes_written += ret;
                        ret = gzprintf(
                            f, fmt ? fmt : "%.*g", DBL_DIG,
                            data->vector_coordinate_complex_double[k].a[0]);
                        if (ret < 0) { err = MTX_ERR_ERRNO; goto gzwrite_exit; }
                        if (bytes_written) *bytes_written += ret;
                        ret = gzprintf(
                            f, fmt ? fmt : "%.*g", DBL_DIG,
                            data->vector_coordinate_complex_double[k].a[1]);
                        if (ret < 0) { err = MTX_ERR_ERRNO; goto gzwrite_exit; }
                        if (bytes_written) *bytes_written += ret;
                        ret = gzputc(f, '\n');
                        if (ret == EOF) { err = MTX_ERR_ERRNO; goto gzwrite_exit; }
                        if (bytes_written) (*bytes_written)++;
                    }
                } else {
                    err = MTX_ERR_INVALID_PRECISION;
                }
            } else if (field == mtxfile_integer) {
                if (precision == mtx_single) {
                    for (int64_t k = 0; k < size; k++) {
                        ret = gzprintf(
                            f, "%d ", data->vector_coordinate_integer_single[k].i);
                        if (ret < 0) { err = MTX_ERR_ERRNO; goto gzwrite_exit; }
                        if (bytes_written) *bytes_written += ret;
                        ret = gzprintf(
                            f, fmt ? fmt : "%d",
                            data->vector_coordinate_integer_single[k].a);
                        if (ret < 0) { err = MTX_ERR_ERRNO; goto gzwrite_exit; }
                        if (bytes_written) *bytes_written += ret;
                        ret = gzputc(f, '\n');
                        if (ret == EOF) { err = MTX_ERR_ERRNO; goto gzwrite_exit; }
                        if (bytes_written) (*bytes_written)++;
                    }
                } else if (precision == mtx_double) {
                    for (int64_t k = 0; k < size; k++) {
                        ret = gzprintf(
                            f, "%d ", data->vector_coordinate_integer_double[k].i);
                        if (ret < 0) { err = MTX_ERR_ERRNO; goto gzwrite_exit; }
                        if (bytes_written) *bytes_written += ret;
                        ret = gzprintf(
                            f, fmt ? fmt : "%d",
                            data->vector_coordinate_integer_double[k].a);
                        if (ret < 0) { err = MTX_ERR_ERRNO; goto gzwrite_exit; }
                        if (bytes_written) *bytes_written += ret;
                        ret = gzputc(f, '\n');
                        if (ret == EOF) { err = MTX_ERR_ERRNO; goto gzwrite_exit; }
                        if (bytes_written) (*bytes_written)++;
                    }
                } else {
                    err = MTX_ERR_INVALID_PRECISION;
                }
            } else if (field == mtxfile_pattern) {
                for (int64_t k = 0; k < size; k++) {
                    ret = gzprintf(
                        f, "%d\n", data->vector_coordinate_pattern[k].i);
                    if (ret < 0) { err = MTX_ERR_ERRNO; goto gzwrite_exit; }
                    if (bytes_written) *bytes_written += ret;
                }
            } else {
                err = MTX_ERR_INVALID_MTX_FIELD;
            }
        } else {
            err = MTX_ERR_INVALID_MTX_OBJECT;
        }
    } else {
        err = MTX_ERR_INVALID_MTX_FORMAT;
    }

gzwrite_exit:
    olderrno = errno;
    setlocale(LC_ALL, locale);
    errno = olderrno;
    free(locale);
    return err;
}
#endif

/*
 * Transpose and conjugate transpose.
 */

/**
 * `mtxfile_data_transpose()' tranposes the data lines of a Matrix
 * Market file.
 */
int mtxfile_data_transpose(
    union mtxfile_data * data,
    enum mtxfile_object object,
    enum mtxfile_format format,
    enum mtxfile_field field,
    enum mtx_precision precision,
    int num_rows,
    int num_columns,
    int64_t size)
{
    int err;
    if (object == mtxfile_matrix) {
        if (format == mtxfile_coordinate) {
            if (field == mtxfile_real) {
                if (precision == mtx_single) {
                    for (int64_t k = 0; k < size; k++) {
                        int i = data->matrix_coordinate_real_single[k].i;
                        int j = data->matrix_coordinate_real_single[k].j;
                        data->matrix_coordinate_real_single[k].i = j;
                        data->matrix_coordinate_real_single[k].j = i;
                    }
                } else if (precision == mtx_double) {
                    for (int64_t k = 0; k < size; k++) {
                        int i = data->matrix_coordinate_real_double[k].i;
                        int j = data->matrix_coordinate_real_double[k].j;
                        data->matrix_coordinate_real_double[k].i = j;
                        data->matrix_coordinate_real_double[k].j = i;
                    }
                } else {
                    return MTX_ERR_INVALID_PRECISION;
                }
            } else if (field == mtxfile_complex) {
                if (precision == mtx_single) {
                    for (int64_t k = 0; k < size; k++) {
                        int i = data->matrix_coordinate_complex_single[k].i;
                        int j = data->matrix_coordinate_complex_single[k].j;
                        data->matrix_coordinate_complex_single[k].i = j;
                        data->matrix_coordinate_complex_single[k].j = i;
                    }
                } else if (precision == mtx_double) {
                    for (int64_t k = 0; k < size; k++) {
                        int i = data->matrix_coordinate_complex_double[k].i;
                        int j = data->matrix_coordinate_complex_double[k].j;
                        data->matrix_coordinate_complex_double[k].i = j;
                        data->matrix_coordinate_complex_double[k].j = i;
                    }
                } else {
                    return MTX_ERR_INVALID_PRECISION;
                }
            } else if (field == mtxfile_integer) {
                if (precision == mtx_single) {
                    for (int64_t k = 0; k < size; k++) {
                        int i = data->matrix_coordinate_integer_single[k].i;
                        int j = data->matrix_coordinate_integer_single[k].j;
                        data->matrix_coordinate_integer_single[k].i = j;
                        data->matrix_coordinate_integer_single[k].j = i;
                    }
                } else if (precision == mtx_double) {
                    for (int64_t k = 0; k < size; k++) {
                        int i = data->matrix_coordinate_integer_double[k].i;
                        int j = data->matrix_coordinate_integer_double[k].j;
                        data->matrix_coordinate_integer_double[k].i = j;
                        data->matrix_coordinate_integer_double[k].j = i;
                    }
                } else {
                    return MTX_ERR_INVALID_PRECISION;
                }
            } else if (field == mtxfile_pattern) {
                for (int64_t k = 0; k < size; k++) {
                    int i = data->matrix_coordinate_pattern[k].i;
                    int j = data->matrix_coordinate_pattern[k].j;
                    data->matrix_coordinate_pattern[k].i = j;
                    data->matrix_coordinate_pattern[k].j = i;
                }
            } else {
                return MTX_ERR_INVALID_MTX_FIELD;
            }

        } else if (format == mtxfile_array) {
            union mtxfile_data copy;
            err = mtxfile_data_alloc(&copy, object, format, field, precision, size);
            if (err)
                return err;
            err = mtxfile_data_copy(
                &copy, data, object, format, field, precision, size, 0, 0);
            if (err) {
                mtxfile_data_free(&copy, object, format, field, precision);
                return err;
            }

            int64_t k, l;
            if (field == mtxfile_real) {
                if (precision == mtx_single) {
                    for (int i = 0; i < num_rows; i++) {
                        for (int j = 0; j < num_columns; j++) {
                            k = (int64_t) i * (int64_t) num_columns + (int64_t) j;
                            l = (int64_t) j * (int64_t) num_rows + (int64_t) i;
                            data->array_real_single[l] = copy.array_real_single[k];
                        }
                    }
                } else if (precision == mtx_double) {
                    for (int i = 0; i < num_rows; i++) {
                        for (int j = 0; j < num_columns; j++) {
                            k = (int64_t) i * (int64_t) num_columns + (int64_t) j;
                            l = (int64_t) j * (int64_t) num_rows + (int64_t) i;
                            data->array_real_double[l] = copy.array_real_double[k];
                        }
                    }
                } else {
                    return MTX_ERR_INVALID_PRECISION;
                }
            } else if (field == mtxfile_complex) {
                if (precision == mtx_single) {
                    for (int i = 0; i < num_rows; i++) {
                        for (int j = 0; j < num_columns; j++) {
                            k = (int64_t) i * (int64_t) num_columns + (int64_t) j;
                            l = (int64_t) j * (int64_t) num_rows + (int64_t) i;
                            data->array_complex_single[l][0] =
                                copy.array_complex_single[k][0];
                            data->array_complex_single[l][1] =
                                copy.array_complex_single[k][1];
                        }
                    }
                } else if (precision == mtx_double) {
                    for (int i = 0; i < num_rows; i++) {
                        for (int j = 0; j < num_columns; j++) {
                            k = (int64_t) i * (int64_t) num_columns + (int64_t) j;
                            l = (int64_t) j * (int64_t) num_rows + (int64_t) i;
                            data->array_complex_double[l][0] =
                                copy.array_complex_double[k][0];
                            data->array_complex_double[l][1] =
                                copy.array_complex_double[k][1];
                        }
                    }
                } else {
                    return MTX_ERR_INVALID_PRECISION;
                }
            } else if (field == mtxfile_integer) {
                if (precision == mtx_single) {
                    for (int i = 0; i < num_rows; i++) {
                        for (int j = 0; j < num_columns; j++) {
                            k = (int64_t) i * (int64_t) num_columns + (int64_t) j;
                            l = (int64_t) j * (int64_t) num_rows + (int64_t) i;
                            data->array_integer_single[l] =
                                copy.array_integer_single[k];
                            data->array_integer_single[l] =
                                copy.array_integer_single[k];
                        }
                    }
                } else if (precision == mtx_double) {
                    for (int i = 0; i < num_rows; i++) {
                        for (int j = 0; j < num_columns; j++) {
                            k = (int64_t) i * (int64_t) num_columns + (int64_t) j;
                            l = (int64_t) j * (int64_t) num_rows + (int64_t) i;
                            data->array_integer_double[l] =
                                copy.array_integer_double[k];
                        }
                    }
                } else {
                    return MTX_ERR_INVALID_PRECISION;
                }
            } else {
                return MTX_ERR_INVALID_MTX_FIELD;
            }

            mtxfile_data_free(&copy, object, format, field, precision);
        } else {
            return MTX_ERR_INVALID_MTX_FORMAT;
        }

    } else if (object == mtxfile_vector) {
        return MTX_SUCCESS;
    } else {
        return MTX_ERR_INVALID_MTX_OBJECT;
    }
    return MTX_SUCCESS;
}

/*
 * Sorting
 */

/**
 * `mtxfile_data_sort_permute()' sorts data lines of a Matrix Market
 * file according to a given sorting permutation.
 */
int mtxfile_data_sort_permute(
    union mtxfile_data * data,
    enum mtxfile_object object,
    enum mtxfile_format format,
    enum mtxfile_field field,
    enum mtx_precision precision,
    int num_rows,
    int num_columns,
    int64_t size,
    int64_t * perm)
{
    int err;
    for (int64_t k = 0; k < size; k++) {
        if (perm[k] <= 0 || perm[k] > size)
            return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    }

    if (format == mtxfile_coordinate) {
        /* 1. Copy the original, unsorted data. */
        union mtxfile_data srcdata;
        err = mtxfile_data_alloc(
            &srcdata, object, mtxfile_coordinate, field, precision, size);
        if (err)
            return err;
        err = mtxfile_data_copy(
            &srcdata, data, object, mtxfile_coordinate, field, precision,
            size, 0, 0);
        if (err) {
            mtxfile_data_free(
                &srcdata, object, mtxfile_coordinate, field, precision);
            return err;
        }

        /* 2. Permute the nonzeros. */
        if (object == mtxfile_matrix) {
            if (field == mtxfile_real) {
                if (precision == mtx_single) {
                    struct mtxfile_matrix_coordinate_real_single * dst =
                        data->matrix_coordinate_real_single;
                    const struct mtxfile_matrix_coordinate_real_single * src =
                        srcdata.matrix_coordinate_real_single;
                    for (int64_t k = 0; k < size; k++)
                        dst[perm[k]-1] = src[k];
                } else if (precision == mtx_double) {
                    struct mtxfile_matrix_coordinate_real_double * dst =
                        data->matrix_coordinate_real_double;
                    const struct mtxfile_matrix_coordinate_real_double * src =
                        srcdata.matrix_coordinate_real_double;
                    for (int64_t k = 0; k < size; k++)
                        dst[perm[k]-1] = src[k];
                } else {
                    mtxfile_data_free(
                        &srcdata, object, mtxfile_coordinate, field, precision);
                    return MTX_ERR_INVALID_PRECISION;
                }
            } else if (field == mtxfile_complex) {
                if (precision == mtx_single) {
                    struct mtxfile_matrix_coordinate_complex_single * dst =
                        data->matrix_coordinate_complex_single;
                    const struct mtxfile_matrix_coordinate_complex_single * src =
                        srcdata.matrix_coordinate_complex_single;
                    for (int64_t k = 0; k < size; k++)
                        dst[perm[k]-1] = src[k];
                } else if (precision == mtx_double) {
                    struct mtxfile_matrix_coordinate_complex_double * dst =
                        data->matrix_coordinate_complex_double;
                    const struct mtxfile_matrix_coordinate_complex_double * src =
                        srcdata.matrix_coordinate_complex_double;
                    for (int64_t k = 0; k < size; k++)
                        dst[perm[k]-1] = src[k];
                } else {
                    mtxfile_data_free(
                        &srcdata, object, mtxfile_coordinate, field, precision);
                    return MTX_ERR_INVALID_PRECISION;
                }
            } else if (field == mtxfile_integer) {
                if (precision == mtx_single) {
                    struct mtxfile_matrix_coordinate_integer_single * dst =
                        data->matrix_coordinate_integer_single;
                    const struct mtxfile_matrix_coordinate_integer_single * src =
                        srcdata.matrix_coordinate_integer_single;
                    for (int64_t k = 0; k < size; k++)
                        dst[perm[k]-1] = src[k];
                } else if (precision == mtx_double) {
                    struct mtxfile_matrix_coordinate_integer_double * dst =
                        data->matrix_coordinate_integer_double;
                    const struct mtxfile_matrix_coordinate_integer_double * src =
                        srcdata.matrix_coordinate_integer_double;
                    for (int64_t k = 0; k < size; k++)
                        dst[perm[k]-1] = src[k];
                } else {
                    mtxfile_data_free(
                        &srcdata, object, mtxfile_coordinate, field, precision);
                    return MTX_ERR_INVALID_PRECISION;
                }
            } else if (field == mtxfile_pattern) {
                struct mtxfile_matrix_coordinate_pattern * dst =
                    data->matrix_coordinate_pattern;
                const struct mtxfile_matrix_coordinate_pattern * src =
                    srcdata.matrix_coordinate_pattern;
                for (int64_t k = 0; k < size; k++)
                    dst[perm[k]-1] = src[k];
            } else {
                mtxfile_data_free(
                    &srcdata, mtxfile_matrix, mtxfile_coordinate, field, precision);
                return MTX_ERR_INVALID_MTX_FIELD;
            }
        } else if (object == mtxfile_vector) {
            if (field == mtxfile_real) {
                if (precision == mtx_single) {
                    struct mtxfile_vector_coordinate_real_single * dst =
                        data->vector_coordinate_real_single;
                    const struct mtxfile_vector_coordinate_real_single * src =
                        srcdata.vector_coordinate_real_single;
                    for (int64_t k = 0; k < size; k++)
                        dst[perm[k]-1] = src[k];
                } else if (precision == mtx_double) {
                    struct mtxfile_vector_coordinate_real_double * dst =
                        data->vector_coordinate_real_double;
                    const struct mtxfile_vector_coordinate_real_double * src =
                        srcdata.vector_coordinate_real_double;
                    for (int64_t k = 0; k < size; k++)
                        dst[perm[k]-1] = src[k];
                } else {
                    mtxfile_data_free(
                        &srcdata, mtxfile_vector, mtxfile_coordinate, field, precision);
                    return MTX_ERR_INVALID_PRECISION;
                }
            } else if (field == mtxfile_complex) {
                if (precision == mtx_single) {
                    struct mtxfile_vector_coordinate_complex_single * dst =
                        data->vector_coordinate_complex_single;
                    const struct mtxfile_vector_coordinate_complex_single * src =
                        srcdata.vector_coordinate_complex_single;
                    for (int64_t k = 0; k < size; k++)
                        dst[perm[k]-1] = src[k];
                } else if (precision == mtx_double) {
                    struct mtxfile_vector_coordinate_complex_double * dst =
                        data->vector_coordinate_complex_double;
                    const struct mtxfile_vector_coordinate_complex_double * src =
                        srcdata.vector_coordinate_complex_double;
                    for (int64_t k = 0; k < size; k++)
                        dst[perm[k]-1] = src[k];
                } else {
                    mtxfile_data_free(
                        &srcdata, mtxfile_vector, mtxfile_coordinate, field, precision);
                    return MTX_ERR_INVALID_PRECISION;
                }
            } else if (field == mtxfile_integer) {
                if (precision == mtx_single) {
                    struct mtxfile_vector_coordinate_integer_single * dst =
                        data->vector_coordinate_integer_single;
                    const struct mtxfile_vector_coordinate_integer_single * src =
                        srcdata.vector_coordinate_integer_single;
                    for (int64_t k = 0; k < size; k++)
                        dst[perm[k]-1] = src[k];
                } else if (precision == mtx_double) {
                    struct mtxfile_vector_coordinate_integer_double * dst =
                        data->vector_coordinate_integer_double;
                    const struct mtxfile_vector_coordinate_integer_double * src =
                        srcdata.vector_coordinate_integer_double;
                    for (int64_t k = 0; k < size; k++)
                        dst[perm[k]-1] = src[k];
                } else {
                    mtxfile_data_free(
                        &srcdata, mtxfile_vector, mtxfile_coordinate, field, precision);
                    return MTX_ERR_INVALID_PRECISION;
                }
            } else if (field == mtxfile_pattern) {
                struct mtxfile_vector_coordinate_pattern * dst =
                    data->vector_coordinate_pattern;
                const struct mtxfile_vector_coordinate_pattern * src =
                    srcdata.vector_coordinate_pattern;
                for (int64_t k = 0; k < size; k++)
                    dst[perm[k]-1] = src[k];
            } else {
                mtxfile_data_free(
                    &srcdata, object, mtxfile_coordinate, field, precision);
                return MTX_ERR_INVALID_MTX_FIELD;
            }
        } else {
            mtxfile_data_free(
                &srcdata, object, mtxfile_coordinate, field, precision);
            return MTX_ERR_INVALID_MTX_OBJECT;
        }
        mtxfile_data_free(&srcdata, object, mtxfile_coordinate, field, precision);
        return MTX_SUCCESS;
    } else {
        return MTX_ERR_INVALID_MTX_FORMAT;
    }
    return MTX_SUCCESS;
}

#define mtxfile_data_sort_matrix_coordinate_row_major_sort()            \
    if (perm) {                                                         \
        for (int64_t k = 0; k < size; k++) {                            \
            int i = src[k].i-1;                                         \
            int64_t l = row_endptr[i]-1;                                \
            while (l >= row_ptr[i] && dst[l].j > src[k].j) {            \
                dst[l+1] = dst[l];                                      \
                perm[l+1] = perm[l];                                    \
                l--;                                                    \
            }                                                           \
            dst[l+1] = src[k];                                          \
            perm[l+1] = k+1;                                            \
            row_endptr[i]++;                                            \
        }                                                               \
    } else {                                                            \
        for (int64_t k = 0; k < size; k++) {                            \
            int i = src[k].i-1;                                         \
            int64_t l = row_endptr[i]-1;                                \
            while (l >= row_ptr[i] && dst[l].j > src[k].j) {            \
                dst[l+1] = dst[l];                                      \
                l--;                                                    \
            }                                                           \
            dst[l+1] = src[k];                                          \
            row_endptr[i]++;                                            \
        }                                                               \
    }

static int mtxfile_data_sort_matrix_coordinate_row_major(
    union mtxfile_data * data,
    enum mtxfile_field field,
    enum mtx_precision precision,
    int num_rows,
    int num_columns,
    int64_t size,
    int64_t * perm)
{
    int err;

    /* 1. Allocate storage for row pointers. */
    int64_t * row_ptr = malloc(2*(num_rows+1) * sizeof(int64_t));
    if (!row_ptr)
        return MTX_ERR_ERRNO;

    /* 2. Count the number of nonzeros stored in each row. */
    err = mtxfile_data_rowptr(
        data, mtxfile_matrix, mtxfile_coordinate, field, precision,
        num_rows, size, row_ptr, NULL);
    if (err) {
        free(row_ptr);
        return err;
    }
    int64_t * row_endptr = &row_ptr[num_rows+1];
    for (int j = 0; j <= num_rows; j++)
        row_endptr[j] = row_ptr[j];

    /* 3. Copy the original, unsorted data. */
    union mtxfile_data srcdata;
    err = mtxfile_data_alloc(
        &srcdata, mtxfile_matrix, mtxfile_coordinate, field, precision, size);
    if (err) {
        free(row_ptr);
        return err;
    }
    err = mtxfile_data_copy(
        &srcdata, data, mtxfile_matrix, mtxfile_coordinate, field, precision,
        size, 0, 0);
    if (err) {
        mtxfile_data_free(
            &srcdata, mtxfile_matrix, mtxfile_coordinate, field, precision);
        free(row_ptr);
        return err;
    }

    /* 4. Sort nonzeros using an insertion sort within each row. */
    if (field == mtxfile_real) {
        if (precision == mtx_single) {
            struct mtxfile_matrix_coordinate_real_single * dst =
                data->matrix_coordinate_real_single;
            const struct mtxfile_matrix_coordinate_real_single * src =
                srcdata.matrix_coordinate_real_single;
            mtxfile_data_sort_matrix_coordinate_row_major_sort();
        } else if (precision == mtx_double) {
            struct mtxfile_matrix_coordinate_real_double * dst =
                data->matrix_coordinate_real_double;
            const struct mtxfile_matrix_coordinate_real_double * src =
                srcdata.matrix_coordinate_real_double;
            mtxfile_data_sort_matrix_coordinate_row_major_sort();
        } else {
            mtxfile_data_free(
                &srcdata, mtxfile_matrix, mtxfile_coordinate, field, precision);
            free(row_ptr);
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (field == mtxfile_complex) {
        if (precision == mtx_single) {
            struct mtxfile_matrix_coordinate_complex_single * dst =
                data->matrix_coordinate_complex_single;
            const struct mtxfile_matrix_coordinate_complex_single * src =
                srcdata.matrix_coordinate_complex_single;
            mtxfile_data_sort_matrix_coordinate_row_major_sort();
        } else if (precision == mtx_double) {
            struct mtxfile_matrix_coordinate_complex_double * dst =
                data->matrix_coordinate_complex_double;
            const struct mtxfile_matrix_coordinate_complex_double * src =
                srcdata.matrix_coordinate_complex_double;
            mtxfile_data_sort_matrix_coordinate_row_major_sort();
        } else {
            mtxfile_data_free(
                &srcdata, mtxfile_matrix, mtxfile_coordinate, field, precision);
            free(row_ptr);
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (field == mtxfile_integer) {
        if (precision == mtx_single) {
            struct mtxfile_matrix_coordinate_integer_single * dst =
                data->matrix_coordinate_integer_single;
            const struct mtxfile_matrix_coordinate_integer_single * src =
                srcdata.matrix_coordinate_integer_single;
            mtxfile_data_sort_matrix_coordinate_row_major_sort();
        } else if (precision == mtx_double) {
            struct mtxfile_matrix_coordinate_integer_double * dst =
                data->matrix_coordinate_integer_double;
            const struct mtxfile_matrix_coordinate_integer_double * src =
                srcdata.matrix_coordinate_integer_double;
            mtxfile_data_sort_matrix_coordinate_row_major_sort();
        } else {
            mtxfile_data_free(
                &srcdata, mtxfile_matrix, mtxfile_coordinate, field, precision);
            free(row_ptr);
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (field == mtxfile_pattern) {
        struct mtxfile_matrix_coordinate_pattern * dst =
            data->matrix_coordinate_pattern;
        const struct mtxfile_matrix_coordinate_pattern * src =
            srcdata.matrix_coordinate_pattern;
        mtxfile_data_sort_matrix_coordinate_row_major_sort();
    } else {
        mtxfile_data_free(
            &srcdata, mtxfile_matrix, mtxfile_coordinate, field, precision);
        free(row_ptr);
        return MTX_ERR_INVALID_MTX_FIELD;
    }
    mtxfile_data_free(&srcdata, mtxfile_matrix, mtxfile_coordinate, field, precision);
    free(row_ptr);

    if (perm) {
        int64_t * invperm = malloc(size * sizeof(int64_t));
        if (!invperm)
            return MTX_ERR_ERRNO;
        for (int64_t k = 0; k < size; k++)
            invperm[k] = perm[k];
        for (int64_t k = 0; k < size; k++)
            perm[invperm[k]-1] = k+1;
        free(invperm);
    }
    return MTX_SUCCESS;
}

static int mtxfile_data_sort_vector_coordinate_row_major(
    union mtxfile_data * data,
    enum mtxfile_field field,
    enum mtx_precision precision,
    int64_t size,
    int64_t * perm)
{
    if (field == mtxfile_real) {
        if (precision == mtx_single) {
            struct mtxfile_vector_coordinate_real_single * dst =
                data->vector_coordinate_real_single;
            if (perm) {
                for (int64_t k = 0; k < size; k++) {
                    struct mtxfile_vector_coordinate_real_single x = dst[k];
                    int64_t l = k-1;
                    while (l >= 0 && dst[l].i > x.i) {
                        dst[l+1] = dst[l];
                        perm[l+1] = perm[l];
                        l--;
                    }
                    dst[l+1] = x;
                    perm[l+1] = k+1;
                }
            } else {
                for (int64_t k = 0; k < size; k++) {
                    struct mtxfile_vector_coordinate_real_single x = dst[k];
                    int64_t l = k-1;
                    while (l >= 0 && dst[l].i > x.i) {
                        dst[l+1] = dst[l];
                        l--;
                    }
                    dst[l+1] = x;
                }
            }
        } else if (precision == mtx_double) {
            struct mtxfile_vector_coordinate_real_double * dst =
                data->vector_coordinate_real_double;
            if (perm) {
                for (int64_t k = 0; k < size; k++) {
                    struct mtxfile_vector_coordinate_real_double x = dst[k];
                    int64_t l = k-1;
                    while (l >= 0 && dst[l].i > x.i) {
                        dst[l+1] = dst[l];
                        perm[l+1] = perm[l];
                        l--;
                    }
                    dst[l+1] = x;
                    perm[l+1] = k+1;
                }
                int64_t * invperm = malloc(size * sizeof(int64_t));
                if (!invperm)
                    return MTX_ERR_ERRNO;
                for (int64_t k = 0; k < size; k++)
                    invperm[k] = perm[k];
                for (int64_t k = 0; k < size; k++)
                    perm[invperm[k]-1] = k+1;
                free(invperm);
            } else {
                for (int64_t k = 0; k < size; k++) {
                    struct mtxfile_vector_coordinate_real_double x = dst[k];
                    int64_t l = k-1;
                    while (l >= 0 && dst[l].i > x.i) {
                        dst[l+1] = dst[l];
                        l--;
                    }
                    dst[l+1] = x;
                }
            }
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (field == mtxfile_complex) {
        if (precision == mtx_single) {
            struct mtxfile_vector_coordinate_complex_single * dst =
                data->vector_coordinate_complex_single;
            if (perm) {
                for (int64_t k = 0; k < size; k++) {
                    struct mtxfile_vector_coordinate_complex_single x = dst[k];
                    int64_t l = k-1;
                    while (l >= 0 && dst[l].i > x.i) {
                        dst[l+1] = dst[l];
                        perm[l+1] = perm[l];
                        l--;
                    }
                    dst[l+1] = x;
                    perm[l+1] = k+1;
                }
            } else {
                for (int64_t k = 0; k < size; k++) {
                    struct mtxfile_vector_coordinate_complex_single x = dst[k];
                    int64_t l = k-1;
                    while (l >= 0 && dst[l].i > x.i) {
                        dst[l+1] = dst[l];
                        l--;
                    }
                    dst[l+1] = x;
                }
            }
        } else if (precision == mtx_double) {
            struct mtxfile_vector_coordinate_complex_double * dst =
                data->vector_coordinate_complex_double;
            if (perm) {
                for (int64_t k = 0; k < size; k++) {
                    struct mtxfile_vector_coordinate_complex_double x = dst[k];
                    int64_t l = k-1;
                    while (l >= 0 && dst[l].i > x.i) {
                        dst[l+1] = dst[l];
                        perm[l+1] = perm[l];
                        l--;
                    }
                    dst[l+1] = x;
                    perm[l+1] = k+1;
                }
            } else {
                for (int64_t k = 0; k < size; k++) {
                    struct mtxfile_vector_coordinate_complex_double x = dst[k];
                    int64_t l = k-1;
                    while (l >= 0 && dst[l].i > x.i) {
                        dst[l+1] = dst[l];
                        l--;
                    }
                    dst[l+1] = x;
                }
            }
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (field == mtxfile_integer) {
        if (precision == mtx_single) {
            struct mtxfile_vector_coordinate_integer_single * dst =
                data->vector_coordinate_integer_single;
            if (perm) {
                for (int64_t k = 0; k < size; k++) {
                    struct mtxfile_vector_coordinate_integer_single x = dst[k];
                    int64_t l = k-1;
                    while (l >= 0 && dst[l].i > x.i) {
                        dst[l+1] = dst[l];
                        perm[l+1] = perm[l];
                        l--;
                    }
                    dst[l+1] = x;
                    perm[l+1] = k+1;
                }
            } else {
                for (int64_t k = 0; k < size; k++) {
                    struct mtxfile_vector_coordinate_integer_single x = dst[k];
                    int64_t l = k-1;
                    while (l >= 0 && dst[l].i > x.i) {
                        dst[l+1] = dst[l];
                        l--;
                    }
                    dst[l+1] = x;
                }
            }
        } else if (precision == mtx_double) {
            struct mtxfile_vector_coordinate_integer_double * dst =
                data->vector_coordinate_integer_double;
            if (perm) {
                for (int64_t k = 0; k < size; k++) {
                    struct mtxfile_vector_coordinate_integer_double x = dst[k];
                    int64_t l = k-1;
                    while (l >= 0 && dst[l].i > x.i) {
                        dst[l+1] = dst[l];
                        perm[l+1] = perm[l];
                        l--;
                    }
                    dst[l+1] = x;
                    perm[l+1] = k+1;
                }
            } else {
                for (int64_t k = 0; k < size; k++) {
                    struct mtxfile_vector_coordinate_integer_double x = dst[k];
                    int64_t l = k-1;
                    while (l >= 0 && dst[l].i > x.i) {
                        dst[l+1] = dst[l];
                        l--;
                    }
                    dst[l+1] = x;
                }
            }
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (field == mtxfile_pattern) {
        struct mtxfile_vector_coordinate_pattern * dst =
            data->vector_coordinate_pattern;
        if (perm) {
            for (int64_t k = 0; k < size; k++) {
                struct mtxfile_vector_coordinate_pattern x = dst[k];
                int64_t l = k-1;
                while (l >= 0 && dst[l].i > x.i) {
                    dst[l+1] = dst[l];
                    perm[l+1] = perm[l];
                    l--;
                }
                dst[l+1] = x;
                perm[l+1] = k+1;
            }
        } else {
            for (int64_t k = 0; k < size; k++) {
                struct mtxfile_vector_coordinate_pattern x = dst[k];
                int64_t l = k-1;
                while (l >= 0 && dst[l].i > x.i) {
                    dst[l+1] = dst[l];
                    l--;
                }
                dst[l+1] = x;
            }
        }
    } else {
        return MTX_ERR_INVALID_MTX_FIELD;
    }

    if (perm) {
        int64_t * invperm = malloc(size * sizeof(int64_t));
        if (!invperm)
            return MTX_ERR_ERRNO;
        for (int64_t k = 0; k < size; k++)
            invperm[k] = perm[k];
        for (int64_t k = 0; k < size; k++)
            perm[invperm[k]-1] = k+1;
        free(invperm);
    }
    return MTX_SUCCESS;
}

/**
 * `mtxfile_data_sort_row_major()' sorts data lines of a Matrix Market
 * file in row major order.
 */
int mtxfile_data_sort_row_major(
    union mtxfile_data * data,
    enum mtxfile_object object,
    enum mtxfile_format format,
    enum mtxfile_field field,
    enum mtx_precision precision,
    int num_rows,
    int num_columns,
    int64_t size,
    int64_t * perm)
{
    if (format == mtxfile_array) {
        if (perm) {
            for (int64_t k = 0; k < size; k++)
                perm[k] = k+1;
        }
        return MTX_SUCCESS;
    } else if (format == mtxfile_coordinate) {
        if (object == mtxfile_matrix) {
            return mtxfile_data_sort_matrix_coordinate_row_major(
                data, field, precision, num_rows, num_columns, size, perm);
        } else if (object == mtxfile_vector) {
            return mtxfile_data_sort_vector_coordinate_row_major(
                data, field, precision, size, perm);
        } else {
            return MTX_ERR_INVALID_MTX_OBJECT;
        }
    } else {
        return MTX_ERR_INVALID_MTX_FORMAT;
    }
    return MTX_SUCCESS;
}

#define mtxfile_data_sort_matrix_coordinate_column_major_sort() \
    if (perm) {                                                 \
        for (int64_t k = 0; k < size; k++) {                    \
            int j = src[k].j-1;                                 \
            int64_t l = column_endptr[j]-1;                     \
            while (l >= column_ptr[j] && dst[l].i > src[k].i) { \
                dst[l+1] = dst[l];                              \
                perm[l+1] = perm[l];                            \
                l--;                                            \
            }                                                   \
            dst[l+1] = src[k];                                  \
            perm[l+1] = k+1;                                    \
            column_endptr[j]++;                                 \
        }                                                       \
    } else {                                                    \
        for (int64_t k = 0; k < size; k++) {                    \
            int j = src[k].j-1;                                 \
            int64_t l = column_endptr[j]-1;                     \
            while (l >= column_ptr[j] && dst[l].i > src[k].i) { \
                dst[l+1] = dst[l];                              \
                l--;                                            \
            }                                                   \
            dst[l+1] = src[k];                                  \
            column_endptr[j]++;                                 \
        }                                                       \
    }

static int mtxfile_data_sort_matrix_coordinate_column_major(
    union mtxfile_data * data,
    enum mtxfile_field field,
    enum mtx_precision precision,
    int num_rows,
    int num_columns,
    int64_t size,
    int64_t * perm)
{
    int err;

    /* 1. Allocate storage for column pointers. */
    int64_t * column_ptr = malloc(2*(num_columns+1) * sizeof(int64_t));
    if (!column_ptr)
        return MTX_ERR_ERRNO;

    /* 2. Count the number of nonzeros stored in each column. */
    err = mtxfile_data_colptr(
        data, mtxfile_matrix, mtxfile_coordinate, field, precision,
        num_columns, size, column_ptr, NULL);
    if (err) {
        free(column_ptr);
        return err;
    }
    int64_t * column_endptr = &column_ptr[num_columns+1];
    for (int j = 0; j <= num_columns; j++)
        column_endptr[j] = column_ptr[j];

    /* 3. Copy the original, unsorted data. */
    union mtxfile_data srcdata;
    err = mtxfile_data_alloc(
        &srcdata, mtxfile_matrix, mtxfile_coordinate, field, precision, size);
    if (err) {
        free(column_ptr);
        return err;
    }
    err = mtxfile_data_copy(
        &srcdata, data, mtxfile_matrix, mtxfile_coordinate, field, precision,
        size, 0, 0);
    if (err) {
        mtxfile_data_free(
            &srcdata, mtxfile_matrix, mtxfile_coordinate, field, precision);
        free(column_ptr);
        return err;
    }

    /* 4. Sort nonzeros using an insertion sort within each column. */
    if (field == mtxfile_real) {
        if (precision == mtx_single) {
            struct mtxfile_matrix_coordinate_real_single * dst =
                data->matrix_coordinate_real_single;
            const struct mtxfile_matrix_coordinate_real_single * src =
                srcdata.matrix_coordinate_real_single;
            mtxfile_data_sort_matrix_coordinate_column_major_sort();
        } else if (precision == mtx_double) {
            struct mtxfile_matrix_coordinate_real_double * dst =
                data->matrix_coordinate_real_double;
            const struct mtxfile_matrix_coordinate_real_double * src =
                srcdata.matrix_coordinate_real_double;
            mtxfile_data_sort_matrix_coordinate_column_major_sort();
        } else {
            mtxfile_data_free(
                &srcdata, mtxfile_matrix, mtxfile_coordinate, field, precision);
            free(column_ptr);
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (field == mtxfile_complex) {
        if (precision == mtx_single) {
            struct mtxfile_matrix_coordinate_complex_single * dst =
                data->matrix_coordinate_complex_single;
            const struct mtxfile_matrix_coordinate_complex_single * src =
                srcdata.matrix_coordinate_complex_single;
            mtxfile_data_sort_matrix_coordinate_column_major_sort();
        } else if (precision == mtx_double) {
            struct mtxfile_matrix_coordinate_complex_double * dst =
                data->matrix_coordinate_complex_double;
            const struct mtxfile_matrix_coordinate_complex_double * src =
                srcdata.matrix_coordinate_complex_double;
            mtxfile_data_sort_matrix_coordinate_column_major_sort();
        } else {
            mtxfile_data_free(
                &srcdata, mtxfile_matrix, mtxfile_coordinate, field, precision);
            free(column_ptr);
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (field == mtxfile_integer) {
        if (precision == mtx_single) {
            struct mtxfile_matrix_coordinate_integer_single * dst =
                data->matrix_coordinate_integer_single;
            const struct mtxfile_matrix_coordinate_integer_single * src =
                srcdata.matrix_coordinate_integer_single;
            mtxfile_data_sort_matrix_coordinate_column_major_sort();
        } else if (precision == mtx_double) {
            struct mtxfile_matrix_coordinate_integer_double * dst =
                data->matrix_coordinate_integer_double;
            const struct mtxfile_matrix_coordinate_integer_double * src =
                srcdata.matrix_coordinate_integer_double;
            mtxfile_data_sort_matrix_coordinate_column_major_sort();
        } else {
            mtxfile_data_free(
                &srcdata, mtxfile_matrix, mtxfile_coordinate, field, precision);
            free(column_ptr);
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (field == mtxfile_pattern) {
        struct mtxfile_matrix_coordinate_pattern * dst =
            data->matrix_coordinate_pattern;
        const struct mtxfile_matrix_coordinate_pattern * src =
            srcdata.matrix_coordinate_pattern;
        mtxfile_data_sort_matrix_coordinate_column_major_sort();
    } else {
        mtxfile_data_free(
            &srcdata, mtxfile_matrix, mtxfile_coordinate, field, precision);
        free(column_ptr);
        return MTX_ERR_INVALID_MTX_FIELD;
    }
    mtxfile_data_free(&srcdata, mtxfile_matrix, mtxfile_coordinate, field, precision);
    free(column_ptr);

    if (perm) {
        int64_t * invperm = malloc(size * sizeof(int64_t));
        if (!invperm)
            return MTX_ERR_ERRNO;
        for (int64_t k = 0; k < size; k++)
            invperm[k] = perm[k];
        for (int64_t k = 0; k < size; k++)
            perm[invperm[k]-1] = k+1;
        free(invperm);
    }
    return MTX_SUCCESS;
}

/**
 * `mtxfile_data_sort_column_major()' sorts data lines of a Matrix
 * Market file in column major order.
 *
 * This operation is not supported for non-square matrices in array
 * format, since they are always stored in row major order.  In this
 * case, one might want to transpose the matrix, which will rearrange
 * the elements to correspond with a column major ordering of the
 * original matrix, but the dimensions of the matrix are also
 * exchanged.
 */
int mtxfile_data_sort_column_major(
    union mtxfile_data * data,
    enum mtxfile_object object,
    enum mtxfile_format format,
    enum mtxfile_field field,
    enum mtx_precision precision,
    int num_rows,
    int num_columns,
    int64_t size,
    int64_t * perm)
{
    if (format == mtxfile_array) {
        if (object == mtxfile_matrix) {
            if (num_rows == num_columns && !perm) {
                return mtxfile_data_transpose(
                    data, object, format, field, precision,
                    num_rows, num_columns, size);
            } else {
                errno = ENOTSUP;
                return MTX_ERR_ERRNO;
            }
        } else if (object == mtxfile_vector) {
            if (perm) {
                for (int64_t k = 0; k < size; k++)
                    perm[k] = k+1;
            }
            return MTX_SUCCESS;
        } else {
            return MTX_ERR_INVALID_MTX_OBJECT;
        }
    } else if (format == mtxfile_coordinate) {
        if (object == mtxfile_matrix) {
            return mtxfile_data_sort_matrix_coordinate_column_major(
                data, field, precision, num_rows, num_columns, size, perm);
        } else if (object == mtxfile_vector) {
            /* Row and column major order are the same for vectors. */
            return mtxfile_data_sort_vector_coordinate_row_major(
                data, field, precision, size, perm);
        } else {
            return MTX_ERR_INVALID_MTX_OBJECT;
        }
    } else {
        return MTX_ERR_INVALID_MTX_FORMAT;
    }
    return MTX_SUCCESS;
}

/*
 * Partitioning
 */

/**
 * `mtxfile_data_sort_by_part()' sorts data lines according to a given
 * partitioning using a stable counting sort algorihtm.
 *
 * The array `parts_per_data_line' must contain `size' integers with
 * values in the range `[0,num_parts-1]', specifying which part of the
 * partition that each data line belongs to.
 *
 * If it is not `NULL', the array `data_lines_per_part_ptr' must
 * contain enough storage for `num_parts+1' values of type
 * `int64_t'. On a successful return, the array will contain offsets
 * to the first data line belonging to each part.
 */
int mtxfile_data_sort_by_part(
    union mtxfile_data * data,
    enum mtxfile_object object,
    enum mtxfile_format format,
    enum mtxfile_field field,
    enum mtx_precision precision,
    int64_t size,
    int64_t offset,
    int num_parts,
    int * parts_per_data_line,
    int64_t * data_lines_per_part_ptr)
{
    int err;

    /* Create a temporary copy of the data lines to be sorted. */
    union mtxfile_data original;
    err = mtxfile_data_alloc(
        &original, object, format, field, precision, size);
    if (err)
        return err;
    err = mtxfile_data_copy(
        &original, data, object, format, field, precision, size, 0, offset);
    if (err) {
        mtxfile_data_free(&original, object, format, field, precision);
        return err;
    }

    /* Count the number of data lines in each part and the offset to
     * the first data line of each part. */
    bool alloc_data_lines_per_part_ptr = !data_lines_per_part_ptr;
    if (alloc_data_lines_per_part_ptr) {
        data_lines_per_part_ptr = malloc((num_parts+1) * sizeof(int64_t));
        if (!data_lines_per_part_ptr) {
            mtxfile_data_free(&original, object, format, field, precision);
            return err;
        }
    }
    for (int p = 0; p <= num_parts; p++)
        data_lines_per_part_ptr[p] = 0;
    for (int64_t l = 0; l < size; l++) {
        int part = parts_per_data_line[l];
        data_lines_per_part_ptr[part+1]++;
    }
    for (int p = 0; p < num_parts; p++) {
        data_lines_per_part_ptr[p+1] +=
            data_lines_per_part_ptr[p];
    }

    /* Sort elements into their respective parts. */
    for (int64_t l = 0; l < size; l++) {
        int part = parts_per_data_line[l];
        int dstidx = data_lines_per_part_ptr[part];
        err = mtxfile_data_copy(
            data, &original, object, format, field, precision, 1, dstidx, l);
        data_lines_per_part_ptr[part]++;
    }

    /* If needed, adjust offsets to each part. */
    if (alloc_data_lines_per_part_ptr) {
        free(data_lines_per_part_ptr);
    } else {
        for (int p = num_parts; p > 0; p--)
            data_lines_per_part_ptr[p] = data_lines_per_part_ptr[p-1];
        data_lines_per_part_ptr[0] = 0;
    }
    mtxfile_data_free(&original, object, format, field, precision);
    return MTX_SUCCESS;
}

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
    int64_t size,
    int64_t offset,
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
                        int i = data->vector_coordinate_real_single[k].i-1;
                        err = mtx_partition_part(row_partition, &row_parts[l], i);
                        if (err)
                            return err;
                    }
                } else if (precision == mtx_double) {
                    for (int64_t l = 0; l < size; l++) {
                        int64_t k = offset + l;
                        int i = data->vector_coordinate_real_double[k].i-1;
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
                        int i = data->vector_coordinate_complex_single[k].i-1;
                        err = mtx_partition_part(row_partition, &row_parts[l], i);
                        if (err)
                            return err;
                    }
                } else if (precision == mtx_double) {
                    for (int64_t l = 0; l < size; l++) {
                        int64_t k = offset + l;
                        int i = data->vector_coordinate_complex_double[k].i-1;
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
                        int i = data->vector_coordinate_integer_single[k].i-1;
                        err = mtx_partition_part(row_partition, &row_parts[l], i);
                        if (err)
                            return err;
                    }
                } else if (precision == mtx_double) {
                    for (int64_t l = 0; l < size; l++) {
                        int64_t k = offset + l;
                        int i = data->vector_coordinate_integer_double[k].i-1;
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
                    int i = data->vector_coordinate_pattern[k].i-1;
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
 * Reordering
 */

/**
 * `mtxfile_data_permute()' permutes the elements of a matrix or
 * vector in Matrix Market format based on given row and column
 * permutations.
 *
 * The array ‘rowperm’ should be a permutation of the integers
 * ‘1,2,...,num_rows’.  For a matrix, the array ‘colperm’ should be a
 * permutation of the integers ‘1,2,...,num_columns’.  The elements
 * belonging to row ‘i’ and column ‘j’ in the permuted matrix are then
 * equal to the elements in row ‘rowperm[i-1]’ and column
 * ‘colperm[j-1]’ in the original matrix, for ‘i=1,2,...,num_rows’ and
 * ‘j=1,2,...,num_columns’.
 */
int mtxfile_data_permute(
    const union mtxfile_data * data,
    enum mtxfile_object object,
    enum mtxfile_format format,
    enum mtxfile_field field,
    enum mtx_precision precision,
    int64_t size,
    int64_t offset,
    int num_rows,
    const int * rowperm,
    int num_columns,
    const int * colperm)
{
    int err;
    if (format == mtxfile_array) {
        /* Create a temporary copy of the data to be permuted. */
        union mtxfile_data original;
        err = mtxfile_data_alloc(
            &original, object, format, field, precision, size);
        if (err)
            return err;
        err = mtxfile_data_copy(
            &original, data, object, format, field, precision, size, 0, offset);
        if (err) {
            mtxfile_data_free(&original, object, format, field, precision);
            return err;
        }

        if (object == mtxfile_matrix) {
            if (field == mtxfile_real) {
                if (precision == mtx_single) {
                    const float * src = original.array_real_single;
                    float * dst = data->array_real_single;
                    if (rowperm && colperm) {
                        for (int64_t i = 0; i < num_rows; i++) {
                            for (int64_t j = 0; j < num_columns; j++) {
                                int64_t k = (int64_t) i*num_columns+j;
                                int64_t l = (int64_t) (rowperm[i]-1)*num_columns + colperm[j]-1;
                                dst[k] = src[l];
                            }
                        }
                    } else if (rowperm) {
                        for (int64_t i = 0; i < num_rows; i++) {
                            for (int64_t j = 0; j < num_columns; j++) {
                                int64_t k = (int64_t) i*num_columns+j;
                                int64_t l = (int64_t) (rowperm[i]-1)*num_columns + j;
                                dst[k] = src[l];
                            }
                        }
                    } else if (colperm) {
                        for (int64_t i = 0; i < num_rows; i++) {
                            for (int64_t j = 0; j < num_columns; j++) {
                                int64_t k = (int64_t) i*num_columns+j;
                                int64_t l = (int64_t) i*num_columns + colperm[j]-1;
                                dst[k] = src[l];
                            }
                        }
                    }
                } else if (precision == mtx_double) {
                    const double * src = original.array_real_double;
                    double * dst = data->array_real_double;
                    if (rowperm && colperm) {
                        for (int64_t i = 0; i < num_rows; i++) {
                            for (int64_t j = 0; j < num_columns; j++) {
                                int64_t k = (int64_t) i*num_columns+j;
                                int64_t l = (int64_t) (rowperm[i]-1)*num_columns + colperm[j]-1;
                                dst[k] = src[l];
                            }
                        }
                    } else if (rowperm) {
                        for (int64_t i = 0; i < num_rows; i++) {
                            for (int64_t j = 0; j < num_columns; j++) {
                                int64_t k = (int64_t) i*num_columns+j;
                                int64_t l = (int64_t) (rowperm[i]-1)*num_columns + j;
                                dst[k] = src[l];
                            }
                        }
                    } else if (colperm) {
                        for (int64_t i = 0; i < num_rows; i++) {
                            for (int64_t j = 0; j < num_columns; j++) {
                                int64_t k = (int64_t) i*num_columns+j;
                                int64_t l = (int64_t) i*num_columns + colperm[j]-1;
                                dst[k] = src[l];
                            }
                        }
                    }
                } else {
                    mtxfile_data_free(&original, object, format, field, precision);
                    return MTX_ERR_INVALID_PRECISION;
                }
            } else if (field == mtxfile_complex) {
                if (precision == mtx_single) {
                    const float (* src)[2] = original.array_complex_single;
                    float (* dst)[2] = data->array_complex_single;
                    if (rowperm && colperm) {
                        for (int64_t i = 0; i < num_rows; i++) {
                            for (int64_t j = 0; j < num_columns; j++) {
                                int64_t k = (int64_t) i*num_columns+j;
                                int64_t l = (int64_t) (rowperm[i]-1)*num_columns + colperm[j]-1;
                                dst[k][0] = src[l][0];
                                dst[k][1] = src[l][1];
                            }
                        }
                    } else if (rowperm) {
                        for (int64_t i = 0; i < num_rows; i++) {
                            for (int64_t j = 0; j < num_columns; j++) {
                                int64_t k = (int64_t) i*num_columns+j;
                                int64_t l = (int64_t) (rowperm[i]-1)*num_columns + j;
                                dst[k][0] = src[l][0];
                                dst[k][1] = src[l][1];
                            }
                        }
                    } else if (colperm) {
                        for (int64_t i = 0; i < num_rows; i++) {
                            for (int64_t j = 0; j < num_columns; j++) {
                                int64_t k = (int64_t) i*num_columns+j;
                                int64_t l = (int64_t) i*num_columns + colperm[j]-1;
                                dst[k][0] = src[l][0];
                                dst[k][1] = src[l][1];
                            }
                        }
                    }
                } else if (precision == mtx_double) {
                    const double (* src)[2] = original.array_complex_double;
                    double (* dst)[2] = data->array_complex_double;
                    if (rowperm && colperm) {
                        for (int64_t i = 0; i < num_rows; i++) {
                            for (int64_t j = 0; j < num_columns; j++) {
                                int64_t k = (int64_t) i*num_columns+j;
                                int64_t l = (int64_t) (rowperm[i]-1)*num_columns + colperm[j]-1;
                                dst[k][0] = src[l][0];
                                dst[k][1] = src[l][1];
                            }
                        }
                    } else if (rowperm) {
                        for (int64_t i = 0; i < num_rows; i++) {
                            for (int64_t j = 0; j < num_columns; j++) {
                                int64_t k = (int64_t) i*num_columns+j;
                                int64_t l = (int64_t) (rowperm[i]-1)*num_columns + j;
                                dst[k][0] = src[l][0];
                                dst[k][1] = src[l][1];
                            }
                        }
                    } else if (colperm) {
                        for (int64_t i = 0; i < num_rows; i++) {
                            for (int64_t j = 0; j < num_columns; j++) {
                                int64_t k = (int64_t) i*num_columns+j;
                                int64_t l = (int64_t) i*num_columns + colperm[j]-1;
                                dst[k][0] = src[l][0];
                                dst[k][1] = src[l][1];
                            }
                        }
                    }
                } else {
                    mtxfile_data_free(&original, object, format, field, precision);
                    return MTX_ERR_INVALID_PRECISION;
                }
            } else if (field == mtxfile_integer) {
                if (precision == mtx_single) {
                    const int32_t * src = original.array_integer_single;
                    int32_t * dst = data->array_integer_single;
                    if (rowperm && colperm) {
                        for (int64_t i = 0; i < num_rows; i++) {
                            for (int64_t j = 0; j < num_columns; j++) {
                                int64_t k = (int64_t) i*num_columns+j;
                                int64_t l = (int64_t) (rowperm[i]-1)*num_columns + colperm[j]-1;
                                dst[k] = src[l];
                            }
                        }
                    } else if (rowperm) {
                        for (int64_t i = 0; i < num_rows; i++) {
                            for (int64_t j = 0; j < num_columns; j++) {
                                int64_t k = (int64_t) i*num_columns+j;
                                int64_t l = (int64_t) (rowperm[i]-1)*num_columns + j;
                                dst[k] = src[l];
                            }
                        }
                    } else if (colperm) {
                        for (int64_t i = 0; i < num_rows; i++) {
                            for (int64_t j = 0; j < num_columns; j++) {
                                int64_t k = (int64_t) i*num_columns+j;
                                int64_t l = (int64_t) i*num_columns + colperm[j]-1;
                                dst[k] = src[l];
                            }
                        }
                    }
                } else if (precision == mtx_double) {
                    const int64_t * src = original.array_integer_double;
                    int64_t * dst = data->array_integer_double;
                    if (rowperm && colperm) {
                        for (int64_t i = 0; i < num_rows; i++) {
                            for (int64_t j = 0; j < num_columns; j++) {
                                int64_t k = (int64_t) i*num_columns+j;
                                int64_t l = (int64_t) (rowperm[i]-1)*num_columns + colperm[j]-1;
                                dst[k] = src[l];
                            }
                        }
                    } else if (rowperm) {
                        for (int64_t i = 0; i < num_rows; i++) {
                            for (int64_t j = 0; j < num_columns; j++) {
                                int64_t k = (int64_t) i*num_columns+j;
                                int64_t l = (int64_t) (rowperm[i]-1)*num_columns + j;
                                dst[k] = src[l];
                            }
                        }
                    } else if (colperm) {
                        for (int64_t i = 0; i < num_rows; i++) {
                            for (int64_t j = 0; j < num_columns; j++) {
                                int64_t k = (int64_t) i*num_columns+j;
                                int64_t l = (int64_t) i*num_columns + colperm[j]-1;
                                dst[k] = src[l];
                            }
                        }
                    }
                } else {
                    mtxfile_data_free(&original, object, format, field, precision);
                    return MTX_ERR_INVALID_PRECISION;
                }
            } else {
                mtxfile_data_free(&original, object, format, field, precision);
                return MTX_ERR_INVALID_MTX_FIELD;
            }

        } else if (object == mtxfile_vector) {
            if (field == mtxfile_real) {
                if (precision == mtx_single) {
                    const float * src = original.array_real_single;
                    float * dst = data->array_real_single;
                    if (rowperm) {
                        for (int i = 0; i < num_rows; i++)
                            dst[i] = src[rowperm[i]-1];
                    }
                } else if (precision == mtx_double) {
                    const double * src = original.array_real_double;
                    double * dst = data->array_real_double;
                    if (rowperm) {
                        for (int i = 0; i < num_rows; i++)
                            dst[i] = src[rowperm[i]-1];
                    }
                } else {
                    mtxfile_data_free(&original, object, format, field, precision);
                    return MTX_ERR_INVALID_PRECISION;
                }
            } else if (field == mtxfile_complex) {
                if (precision == mtx_single) {
                    const float (* src)[2] = original.array_complex_single;
                    float (* dst)[2] = data->array_complex_single;
                    if (rowperm) {
                        for (int i = 0; i < num_rows; i++) {
                            dst[i][0] = src[rowperm[i]-1][0];
                            dst[i][1] = src[rowperm[i]-1][1];
                        }
                    }
                } else if (precision == mtx_double) {
                    const double (* src)[2] = original.array_complex_double;
                    double (* dst)[2] = data->array_complex_double;
                    if (rowperm) {
                        for (int i = 0; i < num_rows; i++) {
                            dst[i][0] = src[rowperm[i]-1][0];
                            dst[i][1] = src[rowperm[i]-1][1];
                        }
                    }
                } else {
                    mtxfile_data_free(&original, object, format, field, precision);
                    return MTX_ERR_INVALID_PRECISION;
                }
            } else if (field == mtxfile_integer) {
                if (precision == mtx_single) {
                    const int32_t * src = original.array_integer_single;
                    int32_t * dst = data->array_integer_single;
                    if (rowperm) {
                        for (int i = 0; i < num_rows; i++)
                            dst[i] = src[rowperm[i]-1];
                    }
                } else if (precision == mtx_double) {
                    const int64_t * src = original.array_integer_double;
                    int64_t * dst = data->array_integer_double;
                    if (rowperm) {
                        for (int i = 0; i < num_rows; i++)
                            dst[i] = src[rowperm[i]-1];
                    }
                } else {
                    mtxfile_data_free(&original, object, format, field, precision);
                    return MTX_ERR_INVALID_PRECISION;
                }
            } else {
                mtxfile_data_free(&original, object, format, field, precision);
                return MTX_ERR_INVALID_MTX_FIELD;
            }
        } else {
            mtxfile_data_free(&original, object, format, field, precision);
            return MTX_ERR_INVALID_MTX_OBJECT;
        }
        mtxfile_data_free(&original, object, format, field, precision);

    } else if (format == mtxfile_coordinate) {
        if (object == mtxfile_matrix) {
            if (field == mtxfile_real) {
                if (precision == mtx_single) {
                    struct mtxfile_matrix_coordinate_real_single * dst =
                        data->matrix_coordinate_real_single;
                    if (rowperm && colperm) {
                        for (int64_t k = 0; k < size; k++) {
                            dst[k+offset].i = rowperm[dst[k+offset].i-1];
                            dst[k+offset].j = colperm[dst[k+offset].j-1];
                        }
                    } else if (rowperm) {
                        for (int64_t k = 0; k < size; k++)
                            dst[k+offset].i = rowperm[dst[k+offset].i-1];
                    } else if (colperm) {
                        for (int64_t k = 0; k < size; k++)
                            dst[k+offset].j = colperm[dst[k+offset].j-1];
                    }
                } else if (precision == mtx_double) {
                    struct mtxfile_matrix_coordinate_real_double * dst =
                        data->matrix_coordinate_real_double;
                    if (rowperm && colperm) {
                        for (int64_t k = 0; k < size; k++) {
                            dst[k+offset].i = rowperm[dst[k+offset].i-1];
                            dst[k+offset].j = colperm[dst[k+offset].j-1];
                        }
                    } else if (rowperm) {
                        for (int64_t k = 0; k < size; k++)
                            dst[k+offset].i = rowperm[dst[k+offset].i-1];
                    } else if (colperm) {
                        for (int64_t k = 0; k < size; k++)
                            dst[k+offset].j = colperm[dst[k+offset].j-1];
                    }
                } else {
                    return MTX_ERR_INVALID_PRECISION;
                }
            } else if (field == mtxfile_complex) {
                if (precision == mtx_single) {
                    struct mtxfile_matrix_coordinate_complex_single * dst =
                        data->matrix_coordinate_complex_single;
                    if (rowperm && colperm) {
                        for (int64_t k = 0; k < size; k++) {
                            dst[k+offset].i = rowperm[dst[k+offset].i-1];
                            dst[k+offset].j = colperm[dst[k+offset].j-1];
                        }
                    } else if (rowperm) {
                        for (int64_t k = 0; k < size; k++)
                            dst[k+offset].i = rowperm[dst[k+offset].i-1];
                    } else if (colperm) {
                        for (int64_t k = 0; k < size; k++)
                            dst[k+offset].j = colperm[dst[k+offset].j-1];
                    }
                } else if (precision == mtx_double) {
                    struct mtxfile_matrix_coordinate_complex_double * dst =
                        data->matrix_coordinate_complex_double;
                    if (rowperm && colperm) {
                        for (int64_t k = 0; k < size; k++) {
                            dst[k+offset].i = rowperm[dst[k+offset].i-1];
                            dst[k+offset].j = colperm[dst[k+offset].j-1];
                        }
                    } else if (rowperm) {
                        for (int64_t k = 0; k < size; k++)
                            dst[k+offset].i = rowperm[dst[k+offset].i-1];
                    } else if (colperm) {
                        for (int64_t k = 0; k < size; k++)
                            dst[k+offset].j = colperm[dst[k+offset].j-1];
                    }
                } else {
                    return MTX_ERR_INVALID_PRECISION;
                }
            } else if (field == mtxfile_integer) {
                if (precision == mtx_single) {
                    struct mtxfile_matrix_coordinate_integer_single * dst =
                        data->matrix_coordinate_integer_single;
                    if (rowperm && colperm) {
                        for (int64_t k = 0; k < size; k++) {
                            dst[k+offset].i = rowperm[dst[k+offset].i-1];
                            dst[k+offset].j = colperm[dst[k+offset].j-1];
                        }
                    } else if (rowperm) {
                        for (int64_t k = 0; k < size; k++)
                            dst[k+offset].i = rowperm[dst[k+offset].i-1];
                    } else if (colperm) {
                        for (int64_t k = 0; k < size; k++)
                            dst[k+offset].j = colperm[dst[k+offset].j-1];
                    }
                } else if (precision == mtx_double) {
                    struct mtxfile_matrix_coordinate_integer_double * dst =
                        data->matrix_coordinate_integer_double;
                    if (rowperm && colperm) {
                        for (int64_t k = 0; k < size; k++) {
                            dst[k+offset].i = rowperm[dst[k+offset].i-1];
                            dst[k+offset].j = colperm[dst[k+offset].j-1];
                        }
                    } else if (rowperm) {
                        for (int64_t k = 0; k < size; k++)
                            dst[k+offset].i = rowperm[dst[k+offset].i-1];
                    } else if (colperm) {
                        for (int64_t k = 0; k < size; k++)
                            dst[k+offset].j = colperm[dst[k+offset].j-1];
                    }
                } else {
                    return MTX_ERR_INVALID_PRECISION;
                }
            } else if (field == mtxfile_pattern) {
                struct mtxfile_matrix_coordinate_pattern * dst =
                    data->matrix_coordinate_pattern;
                if (rowperm && colperm) {
                    for (int64_t k = 0; k < size; k++) {
                        dst[k+offset].i = rowperm[dst[k+offset].i-1];
                        dst[k+offset].j = colperm[dst[k+offset].j-1];
                    }
                } else if (rowperm) {
                    for (int64_t k = 0; k < size; k++)
                        dst[k+offset].i = rowperm[dst[k+offset].i-1];
                } else if (colperm) {
                    for (int64_t k = 0; k < size; k++)
                        dst[k+offset].j = colperm[dst[k+offset].j-1];
                }
            } else {
                return MTX_ERR_INVALID_MTX_FIELD;
            }

        } else if (object == mtxfile_vector) {
            if (field == mtxfile_real) {
                if (precision == mtx_single) {
                    struct mtxfile_vector_coordinate_real_single * dst =
                        data->vector_coordinate_real_single;
                    if (rowperm) {
                        for (int64_t k = 0; k < size; k++)
                            dst[k+offset].i = rowperm[dst[k+offset].i-1];
                    }
                } else if (precision == mtx_double) {
                    struct mtxfile_vector_coordinate_real_double * dst =
                        data->vector_coordinate_real_double;
                    if (rowperm) {
                        for (int64_t k = 0; k < size; k++)
                            dst[k+offset].i = rowperm[dst[k+offset].i-1];
                    }
                } else {
                    return MTX_ERR_INVALID_PRECISION;
                }
            } else if (field == mtxfile_complex) {
                if (precision == mtx_single) {
                    struct mtxfile_vector_coordinate_complex_single * dst =
                        data->vector_coordinate_complex_single;
                    if (rowperm) {
                        for (int64_t k = 0; k < size; k++)
                            dst[k+offset].i = rowperm[dst[k+offset].i-1];
                    }
                } else if (precision == mtx_double) {
                    struct mtxfile_vector_coordinate_complex_double * dst =
                        data->vector_coordinate_complex_double;
                    if (rowperm) {
                        for (int64_t k = 0; k < size; k++)
                            dst[k+offset].i = rowperm[dst[k+offset].i-1];
                    }
                } else {
                    return MTX_ERR_INVALID_PRECISION;
                }
            } else if (field == mtxfile_integer) {
                if (precision == mtx_single) {
                    struct mtxfile_vector_coordinate_integer_single * dst =
                        data->vector_coordinate_integer_single;
                    if (rowperm) {
                        for (int64_t k = 0; k < size; k++)
                            dst[k+offset].i = rowperm[dst[k+offset].i-1];
                    }
                } else if (precision == mtx_double) {
                    struct mtxfile_vector_coordinate_integer_double * dst =
                        data->vector_coordinate_integer_double;
                    if (rowperm) {
                        for (int64_t k = 0; k < size; k++)
                            dst[k+offset].i = rowperm[dst[k+offset].i-1];
                    }
                } else {
                    return MTX_ERR_INVALID_PRECISION;
                }
            } else if (field == mtxfile_pattern) {
                struct mtxfile_vector_coordinate_pattern * dst =
                    data->vector_coordinate_pattern;
                if (rowperm) {
                    for (int64_t k = 0; k < size; k++)
                        dst[k+offset].i = rowperm[dst[k+offset].i-1];
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
 * MPI functions
 */

#ifdef LIBMTX_HAVE_MPI
static int mtxfile_array_complex_datatype(
    MPI_Datatype * datatype,
    enum mtx_precision precision,
    int * mpierrcode)
{
    int num_elements = 1;
    MPI_Datatype element_type;
    if (precision == mtx_single)
        element_type = MPI_FLOAT;
    else if (precision == mtx_double)
        element_type = MPI_DOUBLE;
    else
        return MTX_ERR_INVALID_PRECISION;
    int block_length = 2;
    MPI_Aint element_offset = 0;
    MPI_Datatype single_datatype;
    *mpierrcode = MPI_Type_create_struct(
        num_elements, &block_length, &element_offset, &element_type, &single_datatype);
    if (*mpierrcode)
        return MTX_ERR_MPI;
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

static int mtxfile_data_send_array(
    const union mtxfile_data * data,
    enum mtxfile_field field,
    enum mtx_precision precision,
    int64_t size,
    int64_t offset,
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
        MPI_Datatype complex;
        int err = mtxfile_array_complex_datatype(&complex, precision, mpierrcode);
        if (err)
            return err;
        if (precision == mtx_single) {
            *mpierrcode = MPI_Send(
                &data->array_complex_single[offset],
                size, complex, dest, tag, comm);
            if (*mpierrcode) {
                MPI_Type_free(&complex);
                return MTX_ERR_MPI;
            }
        } else if (precision == mtx_double) {
            *mpierrcode = MPI_Send(
                &data->array_complex_double[offset],
                size, complex, dest, tag, comm);
            if (*mpierrcode) {
                MPI_Type_free(&complex);
                return MTX_ERR_MPI;
            }
        } else {
            MPI_Type_free(&complex);
            return MTX_ERR_INVALID_PRECISION;
        }
        MPI_Type_free(&complex);
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
    int64_t size,
    int64_t offset,
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
    int64_t size,
    int64_t offset,
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
    int64_t size,
    int64_t offset,
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
        MPI_Datatype complex;
        int err = mtxfile_array_complex_datatype(&complex, precision, mpierrcode);
        if (err)
            return err;
        if (precision == mtx_single) {
            *mpierrcode = MPI_Recv(
                &data->array_complex_single[offset],
                size, complex, source, tag, comm,
                MPI_STATUS_IGNORE);
            if (*mpierrcode) {
                MPI_Type_free(&complex);
                return MTX_ERR_MPI;
            }
        } else if (precision == mtx_double) {
            *mpierrcode = MPI_Recv(
                &data->array_complex_double[offset],
                size, complex, source, tag, comm,
                MPI_STATUS_IGNORE);
            if (*mpierrcode) {
                MPI_Type_free(&complex);
                return MTX_ERR_MPI;
            }
        } else {
            MPI_Type_free(&complex);
            return MTX_ERR_INVALID_PRECISION;
        }
        MPI_Type_free(&complex);
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
    int64_t size,
    int64_t offset,
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
    int64_t size,
    int64_t offset,
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
    int64_t size,
    int64_t offset,
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
        MPI_Datatype complex;
        int err = mtxfile_array_complex_datatype(&complex, precision, mpierrcode);
        if (err)
            return err;
        if (precision == mtx_single) {
            *mpierrcode = MPI_Bcast(
                &data->array_complex_single[offset], size, complex, root, comm);
            if (*mpierrcode) {
                MPI_Type_free(&complex);
                return MTX_ERR_MPI;
            }
        } else if (precision == mtx_double) {
            *mpierrcode = MPI_Bcast(
                &data->array_complex_double[offset], size, complex, root, comm);
            if (*mpierrcode) {
                MPI_Type_free(&complex);
                return MTX_ERR_MPI;
            }
        } else {
            MPI_Type_free(&complex);
            return MTX_ERR_INVALID_PRECISION;
        }
        MPI_Type_free(&complex);
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
    int64_t size,
    int64_t offset,
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
    int64_t size,
    int64_t offset,
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

static int mtxfile_data_gatherv_array(
    const union mtxfile_data * sendbuf,
    enum mtxfile_field field,
    enum mtx_precision precision,
    int64_t sendoffset,
    int sendcount,
    union mtxfile_data * recvbuf,
    int64_t recvoffset,
    const int * recvcounts,
    const int * recvdispls,
    int root,
    MPI_Comm comm,
    int * mpierrcode)
{
    if (field == mtxfile_real) {
        if (precision == mtx_single) {
            *mpierrcode = MPI_Gatherv(
                &sendbuf->array_real_single[sendoffset], sendcount, MPI_FLOAT,
                &recvbuf->array_real_single[recvoffset], recvcounts, recvdispls, MPI_FLOAT,
                root, comm);
            if (*mpierrcode)
                return MTX_ERR_MPI;
        } else if (precision == mtx_double) {
            *mpierrcode = MPI_Gatherv(
                &sendbuf->array_real_double[sendoffset], sendcount, MPI_DOUBLE,
                &recvbuf->array_real_double[recvoffset], recvcounts, recvdispls, MPI_DOUBLE,
                root, comm);
            if (*mpierrcode)
                return MTX_ERR_MPI;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (field == mtxfile_complex) {
        MPI_Datatype complex;
        int err = mtxfile_array_complex_datatype(&complex, precision, mpierrcode);
        if (err)
            return err;
        if (precision == mtx_single) {
            *mpierrcode = MPI_Gatherv(
                &sendbuf->array_complex_single[sendoffset], sendcount, complex,
                &recvbuf->array_complex_single[recvoffset], recvcounts, recvdispls, complex,
                root, comm);
            if (*mpierrcode) {
                MPI_Type_free(&complex);
                return MTX_ERR_MPI;
            }
        } else if (precision == mtx_double) {
            *mpierrcode = MPI_Gatherv(
                &sendbuf->array_complex_double[sendoffset], sendcount, complex,
                &recvbuf->array_complex_double[recvoffset], recvcounts, recvdispls, complex,
                root, comm);
            if (*mpierrcode) {
                MPI_Type_free(&complex);
                return MTX_ERR_MPI;
            }
        } else {
            MPI_Type_free(&complex);
            return MTX_ERR_INVALID_PRECISION;
        }
        MPI_Type_free(&complex);
    } else if (field == mtxfile_integer) {
        if (precision == mtx_single) {
            *mpierrcode = MPI_Gatherv(
                &sendbuf->array_integer_single[sendoffset], sendcount, MPI_INT32_T,
                &recvbuf->array_integer_single[recvoffset], recvcounts, recvdispls, MPI_INT32_T,
                root, comm);
            if (*mpierrcode)
                return MTX_ERR_MPI;
        } else if (precision == mtx_double) {
            *mpierrcode = MPI_Gatherv(
                &sendbuf->array_integer_double[sendoffset], sendcount, MPI_INT64_T,
                &recvbuf->array_integer_double[recvoffset], recvcounts, recvdispls, MPI_INT64_T,
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

static int mtxfile_data_gatherv_coordinate(
    const union mtxfile_data * sendbuf,
    enum mtxfile_object object,
    enum mtxfile_field field,
    enum mtx_precision precision,
    int64_t sendoffset,
    int sendcount,
    union mtxfile_data * recvbuf,
    int64_t recvoffset,
    const int * recvcounts,
    const int * recvdispls,
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
                *mpierrcode = MPI_Gatherv(
                    &sendbuf->matrix_coordinate_real_single[sendoffset],
                    sendcount, datatype,
                    &recvbuf->matrix_coordinate_real_single[recvoffset],
                    recvcounts, recvdispls, datatype, root, comm);
                if (*mpierrcode)
                    return MTX_ERR_MPI;
            } else if (precision == mtx_double) {
                *mpierrcode = MPI_Gatherv(
                    &sendbuf->matrix_coordinate_real_double[sendoffset],
                    sendcount, datatype,
                    &recvbuf->matrix_coordinate_real_double[recvoffset],
                    recvcounts, recvdispls, datatype, root, comm);
                if (*mpierrcode)
                    return MTX_ERR_MPI;
            } else {
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_complex) {
            if (precision == mtx_single) {
                *mpierrcode = MPI_Gatherv(
                    &sendbuf->matrix_coordinate_complex_single[sendoffset],
                    sendcount, datatype,
                    &recvbuf->matrix_coordinate_complex_single[recvoffset],
                    recvcounts, recvdispls, datatype, root, comm);
                if (*mpierrcode)
                    return MTX_ERR_MPI;
            } else if (precision == mtx_double) {
                *mpierrcode = MPI_Gatherv(
                    &sendbuf->matrix_coordinate_complex_double[sendoffset],
                    sendcount, datatype,
                    &recvbuf->matrix_coordinate_complex_double[recvoffset],
                    recvcounts, recvdispls, datatype, root, comm);
                if (*mpierrcode)
                    return MTX_ERR_MPI;
            } else {
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_integer) {
            if (precision == mtx_single) {
                *mpierrcode = MPI_Gatherv(
                    &sendbuf->matrix_coordinate_integer_single[sendoffset],
                    sendcount, datatype,
                    &recvbuf->matrix_coordinate_integer_single[recvoffset],
                    recvcounts, recvdispls, datatype, root, comm);
                if (*mpierrcode)
                    return MTX_ERR_MPI;
            } else if (precision == mtx_double) {
                *mpierrcode = MPI_Gatherv(
                    &sendbuf->matrix_coordinate_integer_double[sendoffset],
                    sendcount, datatype,
                    &recvbuf->matrix_coordinate_integer_double[recvoffset],
                    recvcounts, recvdispls, datatype, root, comm);
                if (*mpierrcode)
                    return MTX_ERR_MPI;
            } else {
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_pattern) {
            *mpierrcode = MPI_Gatherv(
                &sendbuf->matrix_coordinate_pattern[sendoffset],
                sendcount, datatype,
                &recvbuf->matrix_coordinate_pattern[recvoffset],
                recvcounts, recvdispls, datatype, root, comm);
            if (*mpierrcode)
                return MTX_ERR_MPI;
        } else {
            return MTX_ERR_INVALID_MTX_FIELD;
        }

    } else if (object == mtxfile_vector) {
        if (field == mtxfile_real) {
            if (precision == mtx_single) {
                *mpierrcode = MPI_Gatherv(
                    &sendbuf->vector_coordinate_real_single[sendoffset],
                    sendcount, datatype,
                    &recvbuf->vector_coordinate_real_single[recvoffset],
                    recvcounts, recvdispls, datatype, root, comm);
                if (*mpierrcode)
                    return MTX_ERR_MPI;
            } else if (precision == mtx_double) {
                *mpierrcode = MPI_Gatherv(
                    &sendbuf->vector_coordinate_real_double[sendoffset],
                    sendcount, datatype,
                    &recvbuf->vector_coordinate_real_double[recvoffset],
                    recvcounts, recvdispls, datatype, root, comm);
                if (*mpierrcode)
                    return MTX_ERR_MPI;
            } else {
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_complex) {
            if (precision == mtx_single) {
                *mpierrcode = MPI_Gatherv(
                    &sendbuf->vector_coordinate_complex_single[sendoffset],
                    sendcount, datatype,
                    &recvbuf->vector_coordinate_complex_single[recvoffset],
                    recvcounts, recvdispls, datatype, root, comm);
                if (*mpierrcode)
                    return MTX_ERR_MPI;
            } else if (precision == mtx_double) {
                *mpierrcode = MPI_Gatherv(
                    &sendbuf->vector_coordinate_complex_double[sendoffset],
                    sendcount, datatype,
                    &recvbuf->vector_coordinate_complex_double[recvoffset],
                    recvcounts, recvdispls, datatype, root, comm);
                if (*mpierrcode)
                    return MTX_ERR_MPI;
            } else {
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_integer) {
            if (precision == mtx_single) {
                *mpierrcode = MPI_Gatherv(
                    &sendbuf->vector_coordinate_integer_single[sendoffset],
                    sendcount, datatype,
                    &recvbuf->vector_coordinate_integer_single[recvoffset],
                    recvcounts, recvdispls, datatype, root, comm);
                if (*mpierrcode)
                    return MTX_ERR_MPI;
            } else if (precision == mtx_double) {
                *mpierrcode = MPI_Gatherv(
                    &sendbuf->vector_coordinate_integer_double[sendoffset],
                    sendcount, datatype,
                    &recvbuf->vector_coordinate_integer_double[recvoffset],
                    recvcounts, recvdispls, datatype, root, comm);
                if (*mpierrcode)
                    return MTX_ERR_MPI;
            } else {
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_pattern) {
            *mpierrcode = MPI_Gatherv(
                &sendbuf->vector_coordinate_pattern[sendoffset],
                sendcount, datatype,
                &recvbuf->vector_coordinate_pattern[recvoffset],
                recvcounts, recvdispls, datatype, root, comm);
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
 * `mtxfile_data_gatherv()' gathers Matrix Market data lines onto an
 * MPI root process from other processes in a communicator.
 *
 * This is analogous to `MPI_Gatherv()' and requires every process in
 * the communicator to perform matching calls to
 * `mtxfile_data_gatherv()'.
 */
int mtxfile_data_gatherv(
    const union mtxfile_data * sendbuf,
    enum mtxfile_object object,
    enum mtxfile_format format,
    enum mtxfile_field field,
    enum mtx_precision precision,
    int64_t sendoffset,
    int sendcount,
    union mtxfile_data * recvbuf,
    int64_t recvoffset,
    const int * recvcounts,
    const int * recvdispls,
    int root,
    MPI_Comm comm,
    struct mtxmpierror * mpierror)
{
    if (format == mtxfile_array) {
        return mtxfile_data_gatherv_array(
            sendbuf, field, precision, sendoffset, sendcount,
            recvbuf, recvoffset, recvcounts, recvdispls,
            root, comm, &mpierror->err);
    } else if (format == mtxfile_coordinate) {
        return mtxfile_data_gatherv_coordinate(
            sendbuf, object, field, precision, sendoffset, sendcount,
            recvbuf, recvoffset, recvcounts, recvdispls,
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
    int64_t sendoffset,
    const int * sendcounts,
    const int * displs,
    union mtxfile_data * recvbuf,
    int64_t recvoffset,
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
        MPI_Datatype complex;
        int err = mtxfile_array_complex_datatype(&complex, precision, mpierrcode);
        if (err)
            return err;
        if (precision == mtx_single) {
            *mpierrcode = MPI_Scatterv(
                &sendbuf->array_complex_single[sendoffset], sendcounts, displs, complex,
                &recvbuf->array_complex_single[recvoffset], recvcount, complex,
                root, comm);
            if (*mpierrcode) {
                MPI_Type_free(&complex);
                return MTX_ERR_MPI;
            }
        } else if (precision == mtx_double) {
            *mpierrcode = MPI_Scatterv(
                &sendbuf->array_complex_double[sendoffset], sendcounts, displs, complex,
                &recvbuf->array_complex_double[recvoffset], recvcount, complex,
                root, comm);
            if (*mpierrcode) {
                MPI_Type_free(&complex);
                return MTX_ERR_MPI;
            }
        } else {
            MPI_Type_free(&complex);
            return MTX_ERR_INVALID_PRECISION;
        }
        MPI_Type_free(&complex);
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
    int64_t sendoffset,
    const int * sendcounts,
    const int * displs,
    union mtxfile_data * recvbuf,
    int64_t recvoffset,
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
    int64_t sendoffset,
    const int * sendcounts,
    const int * displs,
    union mtxfile_data * recvbuf,
    int64_t recvoffset,
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

static int mtxfile_data_alltoallv_array(
    const union mtxfile_data * sendbuf,
    enum mtxfile_field field,
    enum mtx_precision precision,
    int64_t sendoffset,
    const int * sendcounts,
    const int * senddispls,
    union mtxfile_data * recvbuf,
    int64_t recvoffset,
    const int * recvcounts,
    const int * recvdispls,
    MPI_Comm comm,
    int * mpierrcode)
{
    if (field == mtxfile_real) {
        if (precision == mtx_single) {
            *mpierrcode = MPI_Alltoallv(
                &sendbuf->array_real_single[sendoffset], sendcounts, senddispls, MPI_FLOAT,
                &recvbuf->array_real_single[recvoffset], recvcounts, recvdispls, MPI_FLOAT,
                comm);
            if (*mpierrcode)
                return MTX_ERR_MPI;
        } else if (precision == mtx_double) {
            *mpierrcode = MPI_Alltoallv(
                &sendbuf->array_real_double[sendoffset], sendcounts, senddispls, MPI_DOUBLE,
                &recvbuf->array_real_double[recvoffset], recvcounts, recvdispls, MPI_DOUBLE,
                comm);
            if (*mpierrcode)
                return MTX_ERR_MPI;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (field == mtxfile_complex) {
        MPI_Datatype complex;
        int err = mtxfile_array_complex_datatype(&complex, precision, mpierrcode);
        if (err)
            return err;
        if (precision == mtx_single) {
            *mpierrcode = MPI_Alltoallv(
                &sendbuf->array_complex_single[sendoffset], sendcounts, senddispls, complex,
                &recvbuf->array_complex_single[recvoffset], recvcounts, recvdispls, complex,
                comm);
            if (*mpierrcode) {
                MPI_Type_free(&complex);
                return MTX_ERR_MPI;
            }
        } else if (precision == mtx_double) {
            *mpierrcode = MPI_Alltoallv(
                &sendbuf->array_complex_double[sendoffset], sendcounts, senddispls, complex,
                &recvbuf->array_complex_double[recvoffset], recvcounts, recvdispls, complex,
                comm);
            if (*mpierrcode) {
                MPI_Type_free(&complex);
                return MTX_ERR_MPI;
            }
        } else {
            MPI_Type_free(&complex);
            return MTX_ERR_INVALID_PRECISION;
        }
        MPI_Type_free(&complex);
    } else if (field == mtxfile_integer) {
        if (precision == mtx_single) {
            *mpierrcode = MPI_Alltoallv(
                &sendbuf->array_integer_single[sendoffset], sendcounts, senddispls, MPI_INT32_T,
                &recvbuf->array_integer_single[recvoffset], recvcounts, recvdispls, MPI_INT32_T,
                comm);
            if (*mpierrcode)
                return MTX_ERR_MPI;
        } else if (precision == mtx_double) {
            *mpierrcode = MPI_Alltoallv(
                &sendbuf->array_integer_double[sendoffset], sendcounts, senddispls, MPI_INT64_T,
                &recvbuf->array_integer_double[recvoffset], recvcounts, recvdispls, MPI_INT64_T,
                comm);
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

static int mtxfile_data_alltoallv_coordinate(
    const union mtxfile_data * sendbuf,
    enum mtxfile_object object,
    enum mtxfile_field field,
    enum mtx_precision precision,
    int64_t sendoffset,
    const int * sendcounts,
    const int * senddispls,
    union mtxfile_data * recvbuf,
    int64_t recvoffset,
    const int * recvcounts,
    const int * recvdispls,
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
                *mpierrcode = MPI_Alltoallv(
                    &sendbuf->matrix_coordinate_real_single[sendoffset],
                    sendcounts, senddispls, datatype,
                    &recvbuf->matrix_coordinate_real_single[recvoffset],
                    recvcounts, recvdispls, datatype, comm);
                if (*mpierrcode)
                    return MTX_ERR_MPI;
            } else if (precision == mtx_double) {
                *mpierrcode = MPI_Alltoallv(
                    &sendbuf->matrix_coordinate_real_double[sendoffset],
                    sendcounts, senddispls, datatype,
                    &recvbuf->matrix_coordinate_real_double[recvoffset],
                    recvcounts, recvdispls, datatype, comm);
                if (*mpierrcode)
                    return MTX_ERR_MPI;
            } else {
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_complex) {
            if (precision == mtx_single) {
                *mpierrcode = MPI_Alltoallv(
                    &sendbuf->matrix_coordinate_complex_single[sendoffset],
                    sendcounts, senddispls, datatype,
                    &recvbuf->matrix_coordinate_complex_single[recvoffset],
                    recvcounts, recvdispls, datatype, comm);
                if (*mpierrcode)
                    return MTX_ERR_MPI;
            } else if (precision == mtx_double) {
                *mpierrcode = MPI_Alltoallv(
                    &sendbuf->matrix_coordinate_complex_double[sendoffset],
                    sendcounts, senddispls, datatype,
                    &recvbuf->matrix_coordinate_complex_double[recvoffset],
                    recvcounts, recvdispls, datatype, comm);
                if (*mpierrcode)
                    return MTX_ERR_MPI;
            } else {
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_integer) {
            if (precision == mtx_single) {
                *mpierrcode = MPI_Alltoallv(
                    &sendbuf->matrix_coordinate_integer_single[sendoffset],
                    sendcounts, senddispls, datatype,
                    &recvbuf->matrix_coordinate_integer_single[recvoffset],
                    recvcounts, recvdispls, datatype, comm);
                if (*mpierrcode)
                    return MTX_ERR_MPI;
            } else if (precision == mtx_double) {
                *mpierrcode = MPI_Alltoallv(
                    &sendbuf->matrix_coordinate_integer_double[sendoffset],
                    sendcounts, senddispls, datatype,
                    &recvbuf->matrix_coordinate_integer_double[recvoffset],
                    recvcounts, recvdispls, datatype, comm);
                if (*mpierrcode)
                    return MTX_ERR_MPI;
            } else {
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_pattern) {
            *mpierrcode = MPI_Alltoallv(
                &sendbuf->matrix_coordinate_pattern[sendoffset],
                sendcounts, senddispls, datatype,
                &recvbuf->matrix_coordinate_pattern[recvoffset],
                recvcounts, recvdispls, datatype, comm);
            if (*mpierrcode)
                return MTX_ERR_MPI;
        } else {
            return MTX_ERR_INVALID_MTX_FIELD;
        }

    } else if (object == mtxfile_vector) {
        if (field == mtxfile_real) {
            if (precision == mtx_single) {
                *mpierrcode = MPI_Alltoallv(
                    &sendbuf->vector_coordinate_real_single[sendoffset],
                    sendcounts, senddispls, datatype,
                    &recvbuf->vector_coordinate_real_single[recvoffset],
                    recvcounts, recvdispls, datatype, comm);
                if (*mpierrcode)
                    return MTX_ERR_MPI;
            } else if (precision == mtx_double) {
                *mpierrcode = MPI_Alltoallv(
                    &sendbuf->vector_coordinate_real_double[sendoffset],
                    sendcounts, senddispls, datatype,
                    &recvbuf->vector_coordinate_real_double[recvoffset],
                    recvcounts, recvdispls, datatype, comm);
                if (*mpierrcode)
                    return MTX_ERR_MPI;
            } else {
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_complex) {
            if (precision == mtx_single) {
                *mpierrcode = MPI_Alltoallv(
                    &sendbuf->vector_coordinate_complex_single[sendoffset],
                    sendcounts, senddispls, datatype,
                    &recvbuf->vector_coordinate_complex_single[recvoffset],
                    recvcounts, recvdispls, datatype, comm);
                if (*mpierrcode)
                    return MTX_ERR_MPI;
            } else if (precision == mtx_double) {
                *mpierrcode = MPI_Alltoallv(
                    &sendbuf->vector_coordinate_complex_double[sendoffset],
                    sendcounts, senddispls, datatype,
                    &recvbuf->vector_coordinate_complex_double[recvoffset],
                    recvcounts, recvdispls, datatype, comm);
                if (*mpierrcode)
                    return MTX_ERR_MPI;
            } else {
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_integer) {
            if (precision == mtx_single) {
                *mpierrcode = MPI_Alltoallv(
                    &sendbuf->vector_coordinate_integer_single[sendoffset],
                    sendcounts, senddispls, datatype,
                    &recvbuf->vector_coordinate_integer_single[recvoffset],
                    recvcounts, recvdispls, datatype, comm);
                if (*mpierrcode)
                    return MTX_ERR_MPI;
            } else if (precision == mtx_double) {
                *mpierrcode = MPI_Alltoallv(
                    &sendbuf->vector_coordinate_integer_double[sendoffset],
                    sendcounts, senddispls, datatype,
                    &recvbuf->vector_coordinate_integer_double[recvoffset],
                    recvcounts, recvdispls, datatype, comm);
                if (*mpierrcode)
                    return MTX_ERR_MPI;
            } else {
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_pattern) {
            *mpierrcode = MPI_Alltoallv(
                &sendbuf->vector_coordinate_pattern[sendoffset],
                sendcounts, senddispls, datatype,
                &recvbuf->vector_coordinate_pattern[recvoffset],
                recvcounts, recvdispls, datatype, comm);
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
 * `mtxfile_data_alltoallv()' performs an all-to-all exchange of
 * Matrix Market data lines between MPI processes in a communicator.
 *
 * This is analogous to `MPI_Alltoallv()' and requires every process
 * in the communicator to perform matching calls to
 * `mtxfile_data_alltoallv()'.
 */
int mtxfile_data_alltoallv(
    const union mtxfile_data * sendbuf,
    enum mtxfile_object object,
    enum mtxfile_format format,
    enum mtxfile_field field,
    enum mtx_precision precision,
    int64_t sendoffset,
    const int * sendcounts,
    const int * senddispls,
    union mtxfile_data * recvbuf,
    int64_t recvoffset,
    const int * recvcounts,
    const int * recvdispls,
    MPI_Comm comm,
    struct mtxmpierror * mpierror)
{
    if (format == mtxfile_array) {
        return mtxfile_data_alltoallv_array(
            sendbuf, field, precision, sendoffset, sendcounts, senddispls,
            recvbuf, recvoffset, recvcounts, recvdispls,
            comm, &mpierror->err);
    } else if (format == mtxfile_coordinate) {
        return mtxfile_data_alltoallv_coordinate(
            sendbuf, object, field, precision, sendoffset, sendcounts, senddispls,
            recvbuf, recvoffset, recvcounts, recvdispls,
            comm, &mpierror->err);
    } else {
        return MTX_ERR_INVALID_MTX_FORMAT;
    }
    return MTX_SUCCESS;
}
#endif
