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
 * Last modified: 2022-04-14
 *
 * Matrix Market data lines.
 */

#include "config.h"

#include <libmtx/libmtx-config.h>

#include <libmtx/error.h>
#include <libmtx/precision.h>
#include <libmtx/mtxfile/data.h>
#include <libmtx/mtxfile/header.h>
#include <libmtx/mtxfile/size.h>

#include <libmtx/util/parse.h>
#include <libmtx/util/fmtspec.h>
#include <libmtx/util/partition.h>
#include <libmtx/util/sort.h>

#ifdef LIBMTX_HAVE_MPI
#include <mpi.h>
#endif

#ifdef LIBMTX_HAVE_LIBZ
#include <zlib.h>
#endif

#if defined(HAVE_IMMINTRIN_H) && defined(HAVE_BMI2_INSTRUCTIONS) && defined(LIBMTX_USE_BMI2)
#include <immintrin.h>
#else
static inline uint32_t _pdep_u32(uint32_t val, uint32_t mask)
{
    uint32_t res = 0;
    for (uint32_t bb = 1; mask; bb += bb) {
        if (val & bb)
            res |= mask & -mask;
        mask &= mask - 1;
    }
    return res;
}
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
 * ‘mtxfiledata_dataptr()’ returns a pointer to the ‘k’-th data
 * line. This is done by using the correct member of the underlying
 * ‘mtxfiledata’ union containing the data lines.
 */
int mtxfiledata_dataptr(
    const union mtxfiledata * data,
    enum mtxfileobject object,
    enum mtxfileformat format,
    enum mtxfilefield field,
    enum mtxprecision precision,
    void ** p,
    int64_t k)
{
    if (format == mtxfile_array) {
        if (field == mtxfile_real) {
            if (precision == mtx_single) {
                *p = &data->array_real_single[k];
            } else if (precision == mtx_double) {
                *p = &data->array_real_double[k];
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (field == mtxfile_complex) {
            if (precision == mtx_single) {
                *p = &data->array_complex_single[k];
            } else if (precision == mtx_double) {
                *p = &data->array_complex_double[k];
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (field == mtxfile_integer) {
            if (precision == mtx_single) {
                *p = &data->array_integer_single[k];
            } else if (precision == mtx_double) {
                *p = &data->array_integer_double[k];
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else { return MTX_ERR_INVALID_MTX_FIELD; }
    } else if (format == mtxfile_coordinate) {
        if (object == mtxfile_matrix) {
            if (field == mtxfile_real) {
                if (precision == mtx_single) {
                    *p = &data->matrix_coordinate_real_single[k];
                } else if (precision == mtx_double) {
                    *p = &data->matrix_coordinate_real_double[k];
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_complex) {
                if (precision == mtx_single) {
                    *p = &data->matrix_coordinate_complex_single[k];
                } else if (precision == mtx_double) {
                    *p = &data->matrix_coordinate_complex_double[k];
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_integer) {
                if (precision == mtx_single) {
                    *p = &data->matrix_coordinate_integer_single[k];
                } else if (precision == mtx_double) {
                    *p = &data->matrix_coordinate_integer_double[k];
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_pattern) {
                *p = &data->matrix_coordinate_pattern[k];
            } else { return MTX_ERR_INVALID_MTX_FIELD; }
        } else if (object == mtxfile_vector) {
            if (field == mtxfile_real) {
                if (precision == mtx_single) {
                    *p = &data->vector_coordinate_real_single[k];
                } else if (precision == mtx_double) {
                    *p = &data->vector_coordinate_real_double[k];
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_complex) {
                if (precision == mtx_single) {
                    *p = &data->vector_coordinate_complex_single[k];
                } else if (precision == mtx_double) {
                    *p = &data->vector_coordinate_complex_double[k];
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_integer) {
                if (precision == mtx_single) {
                    *p = &data->vector_coordinate_integer_single[k];
                } else if (precision == mtx_double) {
                    *p = &data->vector_coordinate_integer_double[k];
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_pattern) {
                *p = &data->vector_coordinate_pattern[k];
            } else { return MTX_ERR_INVALID_MTX_FIELD; }
        } else { return MTX_ERR_INVALID_MTX_OBJECT; }
    } else { return MTX_ERR_INVALID_MTX_FORMAT; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxfiledata_size_per_element()’ calculates the size of each
 * element in an array of Matrix Market data corresponding to the
 * given ‘object’, ‘format’, ‘field’ and ‘precision’.
 */
int mtxfiledata_size_per_element(
    const union mtxfiledata * data,
    enum mtxfileobject object,
    enum mtxfileformat format,
    enum mtxfilefield field,
    enum mtxprecision precision,
    size_t * size)
{
    if (format == mtxfile_array) {
        if (field == mtxfile_real) {
            if (precision == mtx_single) {
                *size = sizeof(*data->array_real_single);
            } else if (precision == mtx_double) {
                *size = sizeof(*data->array_real_double);
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (field == mtxfile_complex) {
            if (precision == mtx_single) {
                *size = sizeof(*data->array_complex_single);
            } else if (precision == mtx_double) {
                *size = sizeof(*data->array_complex_double);
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (field == mtxfile_integer) {
            if (precision == mtx_single) {
                *size = sizeof(*data->array_integer_single);
            } else if (precision == mtx_double) {
                *size = sizeof(*data->array_integer_double);
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else { return MTX_ERR_INVALID_MTX_FIELD; }
    } else if (format == mtxfile_coordinate) {
        if (object == mtxfile_matrix) {
            if (field == mtxfile_real) {
                if (precision == mtx_single) {
                    *size = sizeof(*data->matrix_coordinate_real_single);
                } else if (precision == mtx_double) {
                    *size = sizeof(*data->matrix_coordinate_real_double);
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_complex) {
                if (precision == mtx_single) {
                    *size = sizeof(*data->matrix_coordinate_complex_single);
                } else if (precision == mtx_double) {
                    *size = sizeof(*data->matrix_coordinate_complex_double);
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_integer) {
                if (precision == mtx_single) {
                    *size = sizeof(*data->matrix_coordinate_integer_single);
                } else if (precision == mtx_double) {
                    *size = sizeof(*data->matrix_coordinate_integer_double);
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_pattern) {
                *size = sizeof(*data->matrix_coordinate_pattern);
            } else { return MTX_ERR_INVALID_MTX_FIELD; }
        } else if (object == mtxfile_vector) {
            if (field == mtxfile_real) {
                if (precision == mtx_single) {
                    *size = sizeof(*data->vector_coordinate_real_single);
                } else if (precision == mtx_double) {
                    *size = sizeof(*data->vector_coordinate_real_double);
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_complex) {
                if (precision == mtx_single) {
                    *size = sizeof(*data->vector_coordinate_complex_single);
                } else if (precision == mtx_double) {
                    *size = sizeof(*data->vector_coordinate_complex_double);
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_integer) {
                if (precision == mtx_single) {
                    *size = sizeof(*data->vector_coordinate_integer_single);
                } else if (precision == mtx_double) {
                    *size = sizeof(*data->vector_coordinate_integer_double);
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_pattern) {
                *size = sizeof(*data->vector_coordinate_pattern);
            } else { return MTX_ERR_INVALID_MTX_FIELD; }
        } else { return MTX_ERR_INVALID_MTX_OBJECT; }
    } else { return MTX_ERR_INVALID_MTX_FORMAT; }
    return MTX_SUCCESS;
}

/*
 * Array formats
 */

/**
 * ‘mtxfiledata_parse_array_real_single()’ parses a string containing
 * a data line for a Matrix Market file in array format with real
 * values.
 *
 * If ‘bytes_read’ is not ‘NULL’, then it is set to the number of
 * bytes consumed by the parser, if successful.  Similarly, if
 * ‘endptr’ is not ‘NULL’, then the address stored in ‘endptr’ points
 * to the first character beyond the characters that were consumed
 * during parsing.
 */
int mtxfiledata_parse_array_real_single(
    float * data,
    int64_t * bytes_read,
    char ** outendptr,
    const char * s)
{
    char * endptr;
    int err = parse_float(data, s, &endptr, bytes_read);
    if (err) return err;
    if (s == endptr) return MTX_ERR_INVALID_MTX_DATA;
    if (outendptr) *outendptr = endptr;
    return MTX_SUCCESS;
}

/**
 * ‘mtxfiledata_parse_array_real_double()’ parses a string containing
 * a data line for a Matrix Market file in array format with real
 * values.
 *
 * If ‘bytes_read’ is not ‘NULL’, then it is set to the number of
 * bytes consumed by the parser, if successful.  Similarly, if
 * ‘endptr’ is not ‘NULL’, then the address stored in ‘endptr’ points
 * to the first character beyond the characters that were consumed
 * during parsing.
 */
int mtxfiledata_parse_array_real_double(
    double * data,
    int64_t * bytes_read,
    char ** outendptr,
    const char * s)
{
    char * endptr;
    int err = parse_double(data, s, &endptr, bytes_read);
    if (err) return err;
    if (s == endptr) return MTX_ERR_INVALID_MTX_DATA;
    if (outendptr) *outendptr = endptr;
    return MTX_SUCCESS;
}

/**
 * ‘mtxfiledata_parse_array_complex_single()’ parses a string
 * containing a data line for a Matrix Market file in array format
 * with complex values.
 *
 * If ‘bytes_read’ is not ‘NULL’, then it is set to the number of
 * bytes consumed by the parser, if successful.  Similarly, if
 * ‘endptr’ is not ‘NULL’, then the address stored in ‘endptr’ points
 * to the first character beyond the characters that were consumed
 * during parsing.
 */
int mtxfiledata_parse_array_complex_single(
    float (* data)[2],
    int64_t * bytes_read,
    char ** outendptr,
    const char * s)
{
    char * endptr;
    int err = parse_float(&(*data)[0], s, &endptr, bytes_read);
    if (err) return err;
    if (s == endptr || *endptr != ' ') return MTX_ERR_INVALID_MTX_DATA;
    if (bytes_read) (*bytes_read)++;
    if (outendptr) *outendptr = endptr+1;
    s = endptr+1;
    err = parse_float(&(*data)[1], s, &endptr, bytes_read);
    if (err) return err;
    if (s == endptr) return MTX_ERR_INVALID_MTX_DATA;
    if (outendptr) *outendptr = endptr;
    return MTX_SUCCESS;
}

/**
 * ‘mtxfiledata_parse_array_complex_double()’ parses a string
 * containing a data line for a Matrix Market file in array format
 * with complex values.
 *
 * If ‘bytes_read’ is not ‘NULL’, then it is set to the number of
 * bytes consumed by the parser, if successful.  Similarly, if
 * ‘endptr’ is not ‘NULL’, then the address stored in ‘endptr’ points
 * to the first character beyond the characters that were consumed
 * during parsing.
 */
int mtxfiledata_parse_array_complex_double(
    double (* data)[2],
    int64_t * bytes_read,
    char ** outendptr,
    const char * s)
{
    char * endptr;
    int err = parse_double(&(*data)[0], s, &endptr, bytes_read);
    if (err) return err;
    if (s == endptr || *endptr != ' ') return MTX_ERR_INVALID_MTX_DATA;
    if (bytes_read) (*bytes_read)++;
    if (outendptr) *outendptr = endptr+1;
    s = endptr+1;
    err = parse_double(&(*data)[1], s, &endptr, bytes_read);
    if (err) return err;
    if (s == endptr) return MTX_ERR_INVALID_MTX_DATA;
    if (outendptr) *outendptr = endptr;
    return MTX_SUCCESS;
}

/**
 * ‘mtxfiledata_parse_array_integer_single()’ parses a string
 * containing a data line for a Matrix Market file in array format
 * with integer values.
 *
 * If ‘bytes_read’ is not ‘NULL’, then it is set to the number of
 * bytes consumed by the parser, if successful.  Similarly, if
 * ‘endptr’ is not ‘NULL’, then the address stored in ‘endptr’ points
 * to the first character beyond the characters that were consumed
 * during parsing.
 */
int mtxfiledata_parse_array_integer_single(
    int32_t * data,
    int64_t * bytes_read,
    char ** outendptr,
    const char * s)
{
    char * endptr;
    int err = parse_int32(data, s, &endptr, bytes_read);
    if (err) return err;
    if (s == endptr) return MTX_ERR_INVALID_MTX_DATA;
    if (outendptr) *outendptr = endptr;
    return MTX_SUCCESS;
}

/**
 * ‘mtxfiledata_parse_array_integer_double()’ parses a string
 * containing a data line for a Matrix Market file in array format
 * with integer values.
 *
 * If ‘bytes_read’ is not ‘NULL’, then it is set to the number of
 * bytes consumed by the parser, if successful.  Similarly, if
 * ‘endptr’ is not ‘NULL’, then the address stored in ‘endptr’ points
 * to the first character beyond the characters that were consumed
 * during parsing.
 */
int mtxfiledata_parse_array_integer_double(
    int64_t * data,
    int64_t * bytes_read,
    char ** outendptr,
    const char * s)
{
    char * endptr;
    int err = parse_int64(data, s, &endptr, bytes_read);
    if (err) return err;
    if (s == endptr) return MTX_ERR_INVALID_MTX_DATA;
    if (outendptr) *outendptr = endptr;
    return MTX_SUCCESS;
}

/*
 * Matrix coordinate formats
 */

/**
 * ‘mtxfiledata_parse_matrix_coordinate_real_single()’ parses a
 * string containing a data line for a Matrix Market file in matrix
 * coordinate format with real values.
 *
 * If ‘bytes_read’ is not ‘NULL’, then it is set to the number of
 * bytes consumed by the parser, if successful.  Similarly, if
 * ‘endptr’ is not ‘NULL’, then the address stored in ‘endptr’ points
 * to the first character beyond the characters that were consumed
 * during parsing.
 */
int mtxfiledata_parse_matrix_coordinate_real_single(
    struct mtxfile_matrix_coordinate_real_single * data,
    int64_t * bytes_read,
    char ** outendptr,
    const char * s,
    int64_t num_rows,
    int64_t num_columns)
{
    char * endptr;
    int err = parse_int64(&data->i, s, &endptr, bytes_read);
    if (err) return err;
    if (s == endptr || *endptr != ' ') return MTX_ERR_INVALID_MTX_DATA;
    else if (data->i <= 0 || data->i > num_rows) return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    if (bytes_read) (*bytes_read)++;
    if (outendptr) *outendptr = endptr+1;
    s = endptr+1;
    err = parse_int64(&data->j, s, &endptr, bytes_read);
    if (err) return err;
    if (s == endptr || *endptr != ' ') return MTX_ERR_INVALID_MTX_DATA;
    else if (data->j <= 0 || data->j > num_columns) return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    if (bytes_read) (*bytes_read)++;
    if (outendptr) *outendptr = endptr+1;
    s = endptr+1;
    err = parse_float(&data->a, s, &endptr, bytes_read);
    if (err) return err;
    if (s == endptr) return MTX_ERR_INVALID_MTX_DATA;
    if (outendptr) *outendptr = endptr;
    return MTX_SUCCESS;
}

/**
 * ‘mtxfiledata_parse_matrix_coordinate_real_double()’ parses a
 * string containing a data line for a Matrix Market file in matrix
 * coordinate format with real values.
 *
 * If ‘bytes_read’ is not ‘NULL’, then it is set to the number of
 * bytes consumed by the parser, if successful.  Similarly, if
 * ‘endptr’ is not ‘NULL’, then the address stored in ‘endptr’ points
 * to the first character beyond the characters that were consumed
 * during parsing.
 */
int mtxfiledata_parse_matrix_coordinate_real_double(
    struct mtxfile_matrix_coordinate_real_double * data,
    int64_t * bytes_read,
    char ** outendptr,
    const char * s,
    int64_t num_rows,
    int64_t num_columns)
{
    char * endptr;
    int err = parse_int64(&data->i, s, &endptr, bytes_read);
    if (err) return err;
    if (s == endptr || *endptr != ' ') return MTX_ERR_INVALID_MTX_DATA;
    else if (data->i <= 0 || data->i > num_rows) return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    if (bytes_read) (*bytes_read)++;
    if (outendptr) *outendptr = endptr+1;
    s = endptr+1;
    err = parse_int64(&data->j, s, &endptr, bytes_read);
    if (err) return err;
    if (s == endptr || *endptr != ' ') return MTX_ERR_INVALID_MTX_DATA;
    else if (data->j <= 0 || data->j > num_columns) return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    if (bytes_read) (*bytes_read)++;
    if (outendptr) *outendptr = endptr+1;
    s = endptr+1;
    err = parse_double(&data->a, s, &endptr, bytes_read);
    if (err) return err;
    if (s == endptr) return MTX_ERR_INVALID_MTX_DATA;
    if (outendptr) *outendptr = endptr;
    return MTX_SUCCESS;
}

/**
 * ‘mtxfiledata_parse_matrix_coordinate_complex_single()’ parses a
 * string containing a data line for a Matrix Market file in matrix
 * coordinate format with complex values.
 *
 * If ‘bytes_read’ is not ‘NULL’, then it is set to the number of
 * bytes consumed by the parser, if successful.  Similarly, if
 * ‘endptr’ is not ‘NULL’, then the address stored in ‘endptr’ points
 * to the first character beyond the characters that were consumed
 * during parsing.
 */
int mtxfiledata_parse_matrix_coordinate_complex_single(
    struct mtxfile_matrix_coordinate_complex_single * data,
    int64_t * bytes_read,
    char ** outendptr,
    const char * s,
    int64_t num_rows,
    int64_t num_columns)
{
    char * endptr;
    int err = parse_int64(&data->i, s, &endptr, bytes_read);
    if (err) return err;
    if (s == endptr || *endptr != ' ') return MTX_ERR_INVALID_MTX_DATA;
    else if (data->i <= 0 || data->i > num_rows) return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    if (bytes_read) (*bytes_read)++;
    if (outendptr) *outendptr = endptr+1;
    s = endptr+1;
    err = parse_int64(&data->j, s, &endptr, bytes_read);
    if (err) return err;
    if (s == endptr || *endptr != ' ') return MTX_ERR_INVALID_MTX_DATA;
    else if (data->j <= 0 || data->j > num_columns) return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    if (bytes_read) (*bytes_read)++;
    if (outendptr) *outendptr = endptr+1;
    s = endptr+1;
    err = parse_float(&data->a[0], s, &endptr, bytes_read);
    if (err) return err;
    if (s == endptr || *endptr != ' ') return MTX_ERR_INVALID_MTX_DATA;
    if (bytes_read) (*bytes_read)++;
    if (outendptr) *outendptr = endptr+1;
    s = endptr+1;
    err = parse_float(&data->a[1], s, &endptr, bytes_read);
    if (err) return err;
    if (s == endptr) return MTX_ERR_INVALID_MTX_DATA;
    if (outendptr) *outendptr = endptr;
    return MTX_SUCCESS;
}

/**
 * ‘mtxfiledata_parse_matrix_coordinate_complex_double()’ parses a
 * string containing a data line for a Matrix Market file in matrix
 * coordinate format with complex values.
 *
 * If ‘bytes_read’ is not ‘NULL’, then it is set to the number of
 * bytes consumed by the parser, if successful.  Similarly, if
 * ‘endptr’ is not ‘NULL’, then the address stored in ‘endptr’ points
 * to the first character beyond the characters that were consumed
 * during parsing.
 */
int mtxfiledata_parse_matrix_coordinate_complex_double(
    struct mtxfile_matrix_coordinate_complex_double * data,
    int64_t * bytes_read,
    char ** outendptr,
    const char * s,
    int64_t num_rows,
    int64_t num_columns)
{
    char * endptr;
    int err = parse_int64(&data->i, s, &endptr, bytes_read);
    if (err) return err;
    if (s == endptr || *endptr != ' ') return MTX_ERR_INVALID_MTX_DATA;
    else if (data->i <= 0 || data->i > num_rows) return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    if (bytes_read) (*bytes_read)++;
    if (outendptr) *outendptr = endptr+1;
    s = endptr+1;
    err = parse_int64(&data->j, s, &endptr, bytes_read);
    if (err) return err;
    if (s == endptr || *endptr != ' ') return MTX_ERR_INVALID_MTX_DATA;
    else if (data->j <= 0 || data->j > num_columns) return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    if (bytes_read) (*bytes_read)++;
    if (outendptr) *outendptr = endptr+1;
    s = endptr+1;
    err = parse_double(&data->a[0], s, &endptr, bytes_read);
    if (err) return err;
    if (s == endptr || *endptr != ' ') return MTX_ERR_INVALID_MTX_DATA;
    if (bytes_read) (*bytes_read)++;
    if (outendptr) *outendptr = endptr+1;
    s = endptr+1;
    err = parse_double(&data->a[1], s, &endptr, bytes_read);
    if (err) return err;
    if (s == endptr) return MTX_ERR_INVALID_MTX_DATA;
    if (outendptr) *outendptr = endptr;
    return MTX_SUCCESS;
}

/**
 * ‘mtxfiledata_parse_matrix_coordinate_integer_single()’ parses a
 * string containing a data line for a Matrix Market file in matrix
 * coordinate format with integer values.
 *
 * If ‘bytes_read’ is not ‘NULL’, then it is set to the number of
 * bytes consumed by the parser, if successful.  Similarly, if
 * ‘endptr’ is not ‘NULL’, then the address stored in ‘endptr’ points
 * to the first character beyond the characters that were consumed
 * during parsing.
 */
int mtxfiledata_parse_matrix_coordinate_integer_single(
    struct mtxfile_matrix_coordinate_integer_single * data,
    int64_t * bytes_read,
    char ** outendptr,
    const char * s,
    int64_t num_rows,
    int64_t num_columns)
{
    char * endptr;
    int err = parse_int64(&data->i, s, &endptr, bytes_read);
    if (err) return err;
    if (s == endptr || *endptr != ' ') return MTX_ERR_INVALID_MTX_DATA;
    else if (data->i <= 0 || data->i > num_rows) return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    if (bytes_read) (*bytes_read)++;
    if (outendptr) *outendptr = endptr+1;
    s = endptr+1;
    err = parse_int64(&data->j, s, &endptr, bytes_read);
    if (err) return err;
    if (s == endptr || *endptr != ' ') return MTX_ERR_INVALID_MTX_DATA;
    else if (data->j <= 0 || data->j > num_columns) return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    if (bytes_read) (*bytes_read)++;
    if (outendptr) *outendptr = endptr+1;
    s = endptr+1;
    err = parse_int32(&data->a, s, &endptr, bytes_read);
    if (err) return err;
    if (s == endptr) return MTX_ERR_INVALID_MTX_DATA;
    if (outendptr) *outendptr = endptr;
    return MTX_SUCCESS;
}

/**
 * ‘mtxfiledata_parse_matrix_coordinate_integer_double()’ parses a
 * string containing a data line for a Matrix Market file in matrix
 * coordinate format with integer values.
 *
 * If ‘bytes_read’ is not ‘NULL’, then it is set to the number of
 * bytes consumed by the parser, if successful.  Similarly, if
 * ‘endptr’ is not ‘NULL’, then the address stored in ‘endptr’ points
 * to the first character beyond the characters that were consumed
 * during parsing.
 */
int mtxfiledata_parse_matrix_coordinate_integer_double(
    struct mtxfile_matrix_coordinate_integer_double * data,
    int64_t * bytes_read,
    char ** outendptr,
    const char * s,
    int64_t num_rows,
    int64_t num_columns)
{
    char * endptr;
    int err = parse_int64(&data->i, s, &endptr, bytes_read);
    if (err) return err;
    if (s == endptr || *endptr != ' ') return MTX_ERR_INVALID_MTX_DATA;
    else if (data->i <= 0 || data->i > num_rows) return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    if (bytes_read) (*bytes_read)++;
    if (outendptr) *outendptr = endptr+1;
    s = endptr+1;
    err = parse_int64(&data->j, s, &endptr, bytes_read);
    if (err) return err;
    if (s == endptr || *endptr != ' ') return MTX_ERR_INVALID_MTX_DATA;
    else if (data->j <= 0 || data->j > num_columns) return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    if (bytes_read) (*bytes_read)++;
    if (outendptr) *outendptr = endptr+1;
    s = endptr+1;
    err = parse_int64(&data->a, s, &endptr, bytes_read);
    if (err) return err;
    if (s == endptr) return MTX_ERR_INVALID_MTX_DATA;
    if (outendptr) *outendptr = endptr;
    return MTX_SUCCESS;
}

/**
 * ‘mtxfiledata_parse_matrix_coordinate_pattern()’ parses a string
 * containing a data line for a Matrix Market file in matrix
 * coordinate format with pattern (boolean) values.
 *
 * If ‘bytes_read’ is not ‘NULL’, then it is set to the number of
 * bytes consumed by the parser, if successful.  Similarly, if
 * ‘endptr’ is not ‘NULL’, then the address stored in ‘endptr’ points
 * to the first character beyond the characters that were consumed
 * during parsing.
 */
int mtxfiledata_parse_matrix_coordinate_pattern(
    struct mtxfile_matrix_coordinate_pattern * data,
    int64_t * bytes_read,
    char ** outendptr,
    const char * s,
    int64_t num_rows,
    int64_t num_columns)
{
    char * endptr;
    int err = parse_int64(&data->i, s, &endptr, bytes_read);
    if (err) return err;
    if (s == endptr || *endptr != ' ') return MTX_ERR_INVALID_MTX_DATA;
    else if (data->i <= 0 || data->i > num_rows) return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    if (bytes_read) (*bytes_read)++;
    if (outendptr) *outendptr = endptr+1;
    s = endptr+1;
    err = parse_int64(&data->j, s, &endptr, bytes_read);
    if (err) return err;
    if (s == endptr) return MTX_ERR_INVALID_MTX_DATA;
    else if (data->j <= 0 || data->j > num_columns) return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    if (outendptr) *outendptr = endptr;
    return MTX_SUCCESS;
}

/*
 * Vector coordinate formats
 */

/**
 * ‘mtxfiledata_parse_vector_coordinate_real_single()’ parses a
 * string containing a data line for a Matrix Market file in vector
 * coordinate format with real values.
 *
 * If ‘bytes_read’ is not ‘NULL’, then it is set to the number of
 * bytes consumed by the parser, if successful.  Similarly, if
 * ‘endptr’ is not ‘NULL’, then the address stored in ‘endptr’ points
 * to the first character beyond the characters that were consumed
 * during parsing.
 */
int mtxfiledata_parse_vector_coordinate_real_single(
    struct mtxfile_vector_coordinate_real_single * data,
    int64_t * bytes_read,
    char ** outendptr,
    const char * s,
    int64_t num_rows)
{
    char * endptr;
    int err = parse_int64(&data->i, s, &endptr, bytes_read);
    if (err) return err;
    if (s == endptr || *endptr != ' ') return MTX_ERR_INVALID_MTX_DATA;
    else if (data->i <= 0 || data->i > num_rows) return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    if (bytes_read) (*bytes_read)++;
    if (outendptr) *outendptr = endptr+1;
    s = endptr+1;
    err = parse_float(&data->a, s, &endptr, bytes_read);
    if (err) return err;
    if (s == endptr) return MTX_ERR_INVALID_MTX_DATA;
    if (outendptr) *outendptr = endptr;
    return MTX_SUCCESS;
}

/**
 * ‘mtxfiledata_parse_vector_coordinate_real_double()’ parses a
 * string containing a data line for a Matrix Market file in vector
 * coordinate format with real values.
 *
 * If ‘bytes_read’ is not ‘NULL’, then it is set to the number of
 * bytes consumed by the parser, if successful.  Similarly, if
 * ‘endptr’ is not ‘NULL’, then the address stored in ‘endptr’ points
 * to the first character beyond the characters that were consumed
 * during parsing.
 */
int mtxfiledata_parse_vector_coordinate_real_double(
    struct mtxfile_vector_coordinate_real_double * data,
    int64_t * bytes_read,
    char ** outendptr,
    const char * s,
    int64_t num_rows)
{
    char * endptr;
    int err = parse_int64(&data->i, s, &endptr, bytes_read);
    if (err) return err;
    if (s == endptr || *endptr != ' ') return MTX_ERR_INVALID_MTX_DATA;
    else if (data->i <= 0 || data->i > num_rows) return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    if (bytes_read) (*bytes_read)++;
    if (outendptr) *outendptr = endptr+1;
    s = endptr+1;
    err = parse_double(&data->a, s, &endptr, bytes_read);
    if (err) return err;
    if (s == endptr) return MTX_ERR_INVALID_MTX_DATA;
    if (outendptr) *outendptr = endptr;
    return MTX_SUCCESS;
}

/**
 * ‘mtxfiledata_parse_vector_coordinate_complex_single()’ parses a
 * string containing a data line for a Matrix Market file in vector
 * coordinate format with complex values.
 *
 * If ‘bytes_read’ is not ‘NULL’, then it is set to the number of
 * bytes consumed by the parser, if successful.  Similarly, if
 * ‘endptr’ is not ‘NULL’, then the address stored in ‘endptr’ points
 * to the first character beyond the characters that were consumed
 * during parsing.
 */
int mtxfiledata_parse_vector_coordinate_complex_single(
    struct mtxfile_vector_coordinate_complex_single * data,
    int64_t * bytes_read,
    char ** outendptr,
    const char * s,
    int64_t num_rows)
{
    char * endptr;
    int err = parse_int64(&data->i, s, &endptr, bytes_read);
    if (err) return err;
    if (s == endptr || *endptr != ' ') return MTX_ERR_INVALID_MTX_DATA;
    else if (data->i <= 0 || data->i > num_rows) return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    if (bytes_read) (*bytes_read)++;
    if (outendptr) *outendptr = endptr+1;
    s = endptr+1;
    err = parse_float(&data->a[0], s, &endptr, bytes_read);
    if (err) return err;
    if (s == endptr || *endptr != ' ') return MTX_ERR_INVALID_MTX_DATA;
    if (bytes_read) (*bytes_read)++;
    if (outendptr) *outendptr = endptr+1;
    s = endptr+1;
    err = parse_float(&data->a[1], s, &endptr, bytes_read);
    if (err) return err;
    if (s == endptr) return MTX_ERR_INVALID_MTX_DATA;
    if (outendptr) *outendptr = endptr;
    return MTX_SUCCESS;
}

/**
 * ‘mtxfiledata_parse_vector_coordinate_complex_double()’ parses a
 * string containing a data line for a Matrix Market file in vector
 * coordinate format with complex values.
 *
 * If ‘bytes_read’ is not ‘NULL’, then it is set to the number of
 * bytes consumed by the parser, if successful.  Similarly, if
 * ‘endptr’ is not ‘NULL’, then the address stored in ‘endptr’ points
 * to the first character beyond the characters that were consumed
 * during parsing.
 */
int mtxfiledata_parse_vector_coordinate_complex_double(
    struct mtxfile_vector_coordinate_complex_double * data,
    int64_t * bytes_read,
    char ** outendptr,
    const char * s,
    int64_t num_rows)
{
    char * endptr;
    int err = parse_int64(&data->i, s, &endptr, bytes_read);
    if (err) return err;
    if (s == endptr || *endptr != ' ') return MTX_ERR_INVALID_MTX_DATA;
    else if (data->i <= 0 || data->i > num_rows) return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    if (bytes_read) (*bytes_read)++;
    if (outendptr) *outendptr = endptr+1;
    s = endptr+1;
    err = parse_double(&data->a[0], s, &endptr, bytes_read);
    if (err) return err;
    if (s == endptr || *endptr != ' ') return MTX_ERR_INVALID_MTX_DATA;
    if (bytes_read) (*bytes_read)++;
    if (outendptr) *outendptr = endptr+1;
    s = endptr+1;
    err = parse_double(&data->a[1], s, &endptr, bytes_read);
    if (err) return err;
    if (s == endptr) return MTX_ERR_INVALID_MTX_DATA;
    if (outendptr) *outendptr = endptr;
    return MTX_SUCCESS;
}

/**
 * ‘mtxfiledata_parse_vector_coordinate_integer_single()’ parses a
 * string containing a data line for a Matrix Market file in vector
 * coordinate format with integer values.
 *
 * If ‘bytes_read’ is not ‘NULL’, then it is set to the number of
 * bytes consumed by the parser, if successful.  Similarly, if
 * ‘endptr’ is not ‘NULL’, then the address stored in ‘endptr’ points
 * to the first character beyond the characters that were consumed
 * during parsing.
 */
int mtxfiledata_parse_vector_coordinate_integer_single(
    struct mtxfile_vector_coordinate_integer_single * data,
    int64_t * bytes_read,
    char ** outendptr,
    const char * s,
    int64_t num_rows)
{
    char * endptr;
    int err = parse_int64(&data->i, s, &endptr, bytes_read);
    if (err) return err;
    if (s == endptr || *endptr != ' ') return MTX_ERR_INVALID_MTX_DATA;
    else if (data->i <= 0 || data->i > num_rows) return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    if (bytes_read) (*bytes_read)++;
    if (outendptr) *outendptr = endptr+1;
    s = endptr+1;
    err = parse_int32(&data->a, s, &endptr, bytes_read);
    if (err) return err;
    if (s == endptr) return MTX_ERR_INVALID_MTX_DATA;
    if (outendptr) *outendptr = endptr;
    return MTX_SUCCESS;
}

/**
 * ‘mtxfiledata_parse_vector_coordinate_integer_double()’ parses a
 * string containing a data line for a Matrix Market file in vector
 * coordinate format with integer values.
 *
 * If ‘bytes_read’ is not ‘NULL’, then it is set to the number of
 * bytes consumed by the parser, if successful.  Similarly, if
 * ‘endptr’ is not ‘NULL’, then the address stored in ‘endptr’ points
 * to the first character beyond the characters that were consumed
 * during parsing.
 */
int mtxfiledata_parse_vector_coordinate_integer_double(
    struct mtxfile_vector_coordinate_integer_double * data,
    int64_t * bytes_read,
    char ** outendptr,
    const char * s,
    int64_t num_rows)
{
    char * endptr;
    int err = parse_int64(&data->i, s, &endptr, bytes_read);
    if (err) return err;
    if (s == endptr || *endptr != ' ') return MTX_ERR_INVALID_MTX_DATA;
    else if (data->i <= 0 || data->i > num_rows) return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    if (bytes_read) (*bytes_read)++;
    if (outendptr) *outendptr = endptr+1;
    s = endptr+1;
    err = parse_int64(&data->a, s, &endptr, bytes_read);
    if (err) return err;
    if (s == endptr) return MTX_ERR_INVALID_MTX_DATA;
    if (outendptr) *outendptr = endptr;
    return MTX_SUCCESS;
}

/**
 * ‘mtxfiledata_parse_vector_coordinate_pattern()’ parses a string
 * containing a data line for a Matrix Market file in vector
 * coordinate format with pattern (boolean) values.
 *
 * If ‘bytes_read’ is not ‘NULL’, then it is set to the number of
 * bytes consumed by the parser, if successful.  Similarly, if
 * ‘endptr’ is not ‘NULL’, then the address stored in ‘endptr’ points
 * to the first character beyond the characters that were consumed
 * during parsing.
 */
int mtxfiledata_parse_vector_coordinate_pattern(
    struct mtxfile_vector_coordinate_pattern * data,
    int64_t * bytes_read,
    char ** outendptr,
    const char * s,
    int64_t num_rows)
{
    char * endptr;
    int err = parse_int64(&data->i, s, &endptr, bytes_read);
    if (err) return err;
    if (s == endptr) return MTX_ERR_INVALID_MTX_DATA;
    else if (data->i <= 0 || data->i > num_rows) return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    if (outendptr) *outendptr = endptr;
    return MTX_SUCCESS;
}

/*
 * Memory management
 */

/**
 * ‘mtxfiledata_alloc()’ allocates storage for a given number of data
 * lines for a given type of matrix or vector.
 */
int mtxfiledata_alloc(
    union mtxfiledata * data,
    enum mtxfileobject object,
    enum mtxfileformat format,
    enum mtxfilefield field,
    enum mtxprecision precision,
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
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (field == mtxfile_complex) {
            if (precision == mtx_single) {
                data->array_complex_single =
                    malloc(size * sizeof(*data->array_complex_single));
            } else if (precision == mtx_double) {
                data->array_complex_double =
                    malloc(size * sizeof(*data->array_complex_double));
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (field == mtxfile_integer) {
            if (precision == mtx_single) {
                data->array_integer_single =
                    malloc(size * sizeof(*data->array_integer_single));
            } else if (precision == mtx_double) {
                data->array_integer_double =
                    malloc(size * sizeof(*data->array_integer_double));
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else { return MTX_ERR_INVALID_MTX_FIELD; }
    } else if (format == mtxfile_coordinate) {
        if (object == mtxfile_matrix) {
            if (field == mtxfile_real) {
                if (precision == mtx_single) {
                    data->matrix_coordinate_real_single =
                        malloc(size * sizeof(*data->matrix_coordinate_real_single));
                } else if (precision == mtx_double) {
                    data->matrix_coordinate_real_double =
                        malloc(size * sizeof(*data->matrix_coordinate_real_double));
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_complex) {
                if (precision == mtx_single) {
                    data->matrix_coordinate_complex_single =
                        malloc(size * sizeof(*data->matrix_coordinate_complex_single));
                } else if (precision == mtx_double) {
                    data->matrix_coordinate_complex_double =
                        malloc(size * sizeof(*data->matrix_coordinate_complex_double));
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_integer) {
                if (precision == mtx_single) {
                    data->matrix_coordinate_integer_single =
                        malloc(size * sizeof(*data->matrix_coordinate_integer_single));
                } else if (precision == mtx_double) {
                    data->matrix_coordinate_integer_double =
                        malloc(size * sizeof(*data->matrix_coordinate_integer_double));
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_pattern) {
                data->matrix_coordinate_pattern =
                    malloc(size * sizeof(*data->matrix_coordinate_pattern));
            } else { return MTX_ERR_INVALID_MTX_FIELD; }
        } else if (object == mtxfile_vector) {
            if (field == mtxfile_real) {
                if (precision == mtx_single) {
                    data->vector_coordinate_real_single =
                        malloc(size * sizeof(*data->vector_coordinate_real_single));
                } else if (precision == mtx_double) {
                    data->vector_coordinate_real_double =
                        malloc(size * sizeof(*data->vector_coordinate_real_double));
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_complex) {
                if (precision == mtx_single) {
                    data->vector_coordinate_complex_single =
                        malloc(size * sizeof(*data->vector_coordinate_complex_single));
                } else if (precision == mtx_double) {
                    data->vector_coordinate_complex_double =
                        malloc(size * sizeof(*data->vector_coordinate_complex_double));
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_integer) {
                if (precision == mtx_single) {
                    data->vector_coordinate_integer_single =
                        malloc(size * sizeof(*data->vector_coordinate_integer_single));
                } else if (precision == mtx_double) {
                    data->vector_coordinate_integer_double =
                        malloc(size * sizeof(*data->vector_coordinate_integer_double));
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_pattern) {
                data->vector_coordinate_pattern =
                    malloc(size * sizeof(*data->vector_coordinate_pattern));
            } else { return MTX_ERR_INVALID_MTX_FIELD; }
        } else { return MTX_ERR_INVALID_MTX_OBJECT; }
    } else { return MTX_ERR_INVALID_MTX_FORMAT; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxfiledata_free()’ frees allocaed storage for data lines.
 */
int mtxfiledata_free(
    union mtxfiledata * data,
    enum mtxfileobject object,
    enum mtxfileformat format,
    enum mtxfilefield field,
    enum mtxprecision precision)
{
    if (format == mtxfile_array) {
        if (field == mtxfile_real) {
            if (precision == mtx_single) {
                free(data->array_real_single);
            } else if (precision == mtx_double) {
                free(data->array_real_double);
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (field == mtxfile_complex) {
            if (precision == mtx_single) {
                free(data->array_complex_single);
            } else if (precision == mtx_double) {
                free(data->array_complex_double);
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (field == mtxfile_integer) {
            if (precision == mtx_single) {
                free(data->array_integer_single);
            } else if (precision == mtx_double) {
                free(data->array_integer_double);
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else { return MTX_ERR_INVALID_MTX_FIELD; }
    } else if (format == mtxfile_coordinate) {
        if (object == mtxfile_matrix) {
            if (field == mtxfile_real) {
                if (precision == mtx_single) {
                    free(data->matrix_coordinate_real_single);
                } else if (precision == mtx_double) {
                    free(data->matrix_coordinate_real_double);
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_complex) {
                if (precision == mtx_single) {
                    free(data->matrix_coordinate_complex_single);
                } else if (precision == mtx_double) {
                    free(data->matrix_coordinate_complex_double);
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_integer) {
                if (precision == mtx_single) {
                    free(data->matrix_coordinate_integer_single);
                } else if (precision == mtx_double) {
                    free(data->matrix_coordinate_integer_double);
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_pattern) {
                free(data->matrix_coordinate_pattern);
            } else { return MTX_ERR_INVALID_MTX_FIELD; }
        } else if (object == mtxfile_vector) {
            if (field == mtxfile_real) {
                if (precision == mtx_single) {
                    free(data->vector_coordinate_real_single);
                } else if (precision == mtx_double) {
                    free(data->vector_coordinate_real_double);
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_complex) {
                if (precision == mtx_single) {
                    free(data->vector_coordinate_complex_single);
                } else if (precision == mtx_double) {
                    free(data->vector_coordinate_complex_double);
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_integer) {
                if (precision == mtx_single) {
                    free(data->vector_coordinate_integer_single);
                } else if (precision == mtx_double) {
                    free(data->vector_coordinate_integer_double);
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_pattern) {
                free(data->vector_coordinate_pattern);
            } else { return MTX_ERR_INVALID_MTX_FIELD; }
        } else { return MTX_ERR_INVALID_MTX_OBJECT; }
    } else { return MTX_ERR_INVALID_MTX_FORMAT; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxfiledata_copy()’ copies data lines.
 */
int mtxfiledata_copy(
    union mtxfiledata * dst,
    const union mtxfiledata * src,
    enum mtxfileobject object,
    enum mtxfileformat format,
    enum mtxfilefield field,
    enum mtxprecision precision,
    int64_t size,
    int64_t dstoffset,
    int64_t srcoffset)
{
    if (format == mtxfile_array) {
        if (field == mtxfile_real) {
            if (precision == mtx_single) {
                memcpy(&dst->array_real_single[dstoffset],
                       &src->array_real_single[srcoffset],
                       size * sizeof(*src->array_real_single));
            } else if (precision == mtx_double) {
                memcpy(&dst->array_real_double[dstoffset],
                       &src->array_real_double[srcoffset],
                       size * sizeof(*src->array_real_double));
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (field == mtxfile_complex) {
            if (precision == mtx_single) {
                memcpy(&dst->array_complex_single[dstoffset],
                       &src->array_complex_single[srcoffset],
                       size * sizeof(*src->array_complex_single));
            } else if (precision == mtx_double) {
                memcpy(&dst->array_complex_double[dstoffset],
                       &src->array_complex_double[srcoffset],
                       size * sizeof(*src->array_complex_double));
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (field == mtxfile_integer) {
            if (precision == mtx_single) {
                memcpy(&dst->array_integer_single[dstoffset],
                       &src->array_integer_single[srcoffset],
                       size * sizeof(*src->array_integer_single));
            } else if (precision == mtx_double) {
                memcpy(&dst->array_integer_double[dstoffset],
                       &src->array_integer_double[srcoffset],
                       size * sizeof(*src->array_integer_double));
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else { return MTX_ERR_INVALID_MTX_FIELD; }
    } else if (format == mtxfile_coordinate) {
        if (object == mtxfile_matrix) {
            if (field == mtxfile_real) {
                if (precision == mtx_single) {
                    memcpy(&dst->matrix_coordinate_real_single[dstoffset],
                           &src->matrix_coordinate_real_single[srcoffset],
                           size * sizeof(*src->matrix_coordinate_real_single));
                } else if (precision == mtx_double) {
                    memcpy(&dst->matrix_coordinate_real_double[dstoffset],
                           &src->matrix_coordinate_real_double[srcoffset],
                           size * sizeof(*src->matrix_coordinate_real_double));
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_complex) {
                if (precision == mtx_single) {
                    memcpy(&dst->matrix_coordinate_complex_single[dstoffset],
                           &src->matrix_coordinate_complex_single[srcoffset],
                           size * sizeof(*src->matrix_coordinate_complex_single));
                } else if (precision == mtx_double) {
                    memcpy(&dst->matrix_coordinate_complex_double[dstoffset],
                           &src->matrix_coordinate_complex_double[srcoffset],
                           size * sizeof(*src->matrix_coordinate_complex_double));
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_integer) {
                if (precision == mtx_single) {
                    memcpy(&dst->matrix_coordinate_integer_single[dstoffset],
                           &src->matrix_coordinate_integer_single[srcoffset],
                           size * sizeof(*src->matrix_coordinate_integer_single));
                } else if (precision == mtx_double) {
                    memcpy(&dst->matrix_coordinate_integer_double[dstoffset],
                           &src->matrix_coordinate_integer_double[srcoffset],
                           size * sizeof(*src->matrix_coordinate_integer_double));
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_pattern) {
                memcpy(&dst->matrix_coordinate_pattern[dstoffset],
                       &src->matrix_coordinate_pattern[srcoffset],
                       size * sizeof(*src->matrix_coordinate_pattern));
            } else { return MTX_ERR_INVALID_MTX_FIELD; }
        } else if (object == mtxfile_vector) {
            if (field == mtxfile_real) {
                if (precision == mtx_single) {
                    memcpy(&dst->vector_coordinate_real_single[dstoffset],
                           &src->vector_coordinate_real_single[srcoffset],
                           size * sizeof(*src->vector_coordinate_real_single));
                } else if (precision == mtx_double) {
                    memcpy(&dst->vector_coordinate_real_double[dstoffset],
                           &src->vector_coordinate_real_double[srcoffset],
                           size * sizeof(*src->vector_coordinate_real_double));
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_complex) {
                if (precision == mtx_single) {
                    memcpy(&dst->vector_coordinate_complex_single[dstoffset],
                           &src->vector_coordinate_complex_single[srcoffset],
                           size * sizeof(*src->vector_coordinate_complex_single));
                } else if (precision == mtx_double) {
                    memcpy(&dst->vector_coordinate_complex_double[dstoffset],
                           &src->vector_coordinate_complex_double[srcoffset],
                           size * sizeof(*src->vector_coordinate_complex_double));
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_integer) {
                if (precision == mtx_single) {
                    memcpy(&dst->vector_coordinate_integer_single[dstoffset],
                           &src->vector_coordinate_integer_single[srcoffset],
                           size * sizeof(*src->vector_coordinate_integer_single));
                } else if (precision == mtx_double) {
                    memcpy(&dst->vector_coordinate_integer_double[dstoffset],
                           &src->vector_coordinate_integer_double[srcoffset],
                           size * sizeof(*src->vector_coordinate_integer_double));
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_pattern) {
                memcpy(&dst->vector_coordinate_pattern[dstoffset],
                       &src->vector_coordinate_pattern[srcoffset],
                       size * sizeof(*src->vector_coordinate_pattern));
            } else { return MTX_ERR_INVALID_MTX_FIELD; }
        } else { return MTX_ERR_INVALID_MTX_OBJECT; }
    } else { return MTX_ERR_INVALID_MTX_FORMAT; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxfiledata_copy_gather()’ performs an irregular copying (gather)
 * of data lines from specified locations to a contiguous array.
 */
int mtxfiledata_copy_gather(
    union mtxfiledata * dst,
    const union mtxfiledata * src,
    enum mtxfileobject object,
    enum mtxfileformat format,
    enum mtxfilefield field,
    enum mtxprecision precision,
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
            } else { return MTX_ERR_INVALID_PRECISION; }
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
            } else { return MTX_ERR_INVALID_PRECISION; }
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
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else { return MTX_ERR_INVALID_MTX_FIELD; }
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
                } else { return MTX_ERR_INVALID_PRECISION; }
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
                } else { return MTX_ERR_INVALID_PRECISION; }
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
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_pattern) {
                for (int64_t k = 0; k < size; k++) {
                    dst->matrix_coordinate_pattern[dstoffset+k] =
                        src->matrix_coordinate_pattern[srcdispls[k]];
                }
            } else { return MTX_ERR_INVALID_MTX_FIELD; }
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
                } else { return MTX_ERR_INVALID_PRECISION; }
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
                } else { return MTX_ERR_INVALID_PRECISION; }
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
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_pattern) {
                for (int64_t k = 0; k < size; k++) {
                    dst->vector_coordinate_pattern[dstoffset+k] =
                        src->vector_coordinate_pattern[srcdispls[k]];
                }
            } else { return MTX_ERR_INVALID_MTX_FIELD; }
        } else { return MTX_ERR_INVALID_MTX_OBJECT; }
    } else { return MTX_ERR_INVALID_MTX_FORMAT; }
    return MTX_SUCCESS;
}

/*
 * Extracting row/column pointers and indices
 */

/**
 * ‘mtxfiledata_rowcolidx64()’ extracts row and/or column indices for
 * a matrix or vector in Matrix Market format.
 *
 * ‘rowidx’ may be ‘NULL’, in which case it is ignored. Otherwise, it
 * must point to an array containing enough storage for ‘size’ values
 * of type ‘int’.  If successful, this array will contain the row
 * index of each data line.
 *
 * Similarly, ‘colidx’ may be ‘NULL’, or it must point to an array of
 * the same size, which will be used to store the column index of each
 * data line.
 *
 * Note that indexing is 1-based, meaning that rows are numbered
 * ‘1,2,...,num_rows’, whereas columns are numbered
 * ‘1,2,...,num_columns’.
 *
 * If ‘format’ is ‘mtxfile_array’, then a non-negative ‘offset’ value
 * can be used to obtain row and column indices for matrix or vector
 * entries starting from the specified offset, instead of beginning
 * with the first entry of the matrix or vector.
 */
static int mtxfiledata_rowcolidx64(
    const union mtxfiledata * data,
    enum mtxfileobject object,
    enum mtxfileformat format,
    enum mtxfilefield field,
    enum mtxprecision precision,
    int64_t num_rows,
    int64_t num_columns,
    int64_t offset,
    int64_t size,
    int64_t * rowidx,
    int64_t * colidx)
{
    int err;
    if (rowidx && colidx) {
        if (object == mtxfile_matrix) {
            if (format == mtxfile_coordinate) {
                if (field == mtxfile_real) {
                    if (precision == mtx_single) {
                        for (int64_t k = 0; k < size; k++) {
                            rowidx[k] = data->matrix_coordinate_real_single[k].i;
                            colidx[k] = data->matrix_coordinate_real_single[k].j;
                        }
                    } else if (precision == mtx_double) {
                        for (int64_t k = 0; k < size; k++) {
                            rowidx[k] = data->matrix_coordinate_real_double[k].i;
                            colidx[k] = data->matrix_coordinate_real_double[k].j;
                        }
                    } else { return MTX_ERR_INVALID_PRECISION; }
                } else if (field == mtxfile_complex) {
                    if (precision == mtx_single) {
                        for (int64_t k = 0; k < size; k++) {
                            rowidx[k] = data->matrix_coordinate_complex_single[k].i;
                            colidx[k] = data->matrix_coordinate_complex_single[k].j;
                        }
                    } else if (precision == mtx_double) {
                        for (int64_t k = 0; k < size; k++) {
                            rowidx[k] = data->matrix_coordinate_complex_double[k].i;
                            colidx[k] = data->matrix_coordinate_complex_double[k].j;
                        }
                    } else { return MTX_ERR_INVALID_PRECISION; }
                } else if (field == mtxfile_integer) {
                    if (precision == mtx_single) {
                        for (int64_t k = 0; k < size; k++) {
                            rowidx[k] = data->matrix_coordinate_integer_single[k].i;
                            colidx[k] = data->matrix_coordinate_integer_single[k].j;
                        }
                    } else if (precision == mtx_double) {
                        for (int64_t k = 0; k < size; k++) {
                            rowidx[k] = data->matrix_coordinate_integer_double[k].i;
                            colidx[k] = data->matrix_coordinate_integer_double[k].j;
                        }
                    } else { return MTX_ERR_INVALID_PRECISION; }
                } else if (field == mtxfile_pattern) {
                    for (int64_t k = 0; k < size; k++) {
                        rowidx[k] = data->matrix_coordinate_pattern[k].i;
                        colidx[k] = data->matrix_coordinate_pattern[k].j;
                    }
                } else { return MTX_ERR_INVALID_MTX_FIELD; }
            } else if (format == mtxfile_array) {
                if (offset < 0 || (offset + size) / num_rows > num_columns)
                    return MTX_ERR_INDEX_OUT_OF_BOUNDS;
                for (int64_t k = offset, l = 0; l < size; k++, l++) {
                    int64_t i = k / num_columns;
                    int64_t j = k % num_columns;
                    rowidx[l] = i+1;
                    colidx[l] = j+1;
                }
            } else { return MTX_ERR_INVALID_MTX_FORMAT; }
        } else if (object == mtxfile_vector) {
            if (format == mtxfile_coordinate) {
                if (field == mtxfile_real) {
                    if (precision == mtx_single) {
                        for (int64_t k = 0; k < size; k++) {
                            rowidx[k] = data->vector_coordinate_real_single[k].i;
                            colidx[k] = 1;
                        }
                    } else if (precision == mtx_double) {
                        for (int64_t k = 0; k < size; k++) {
                            rowidx[k] = data->vector_coordinate_real_double[k].i;
                            colidx[k] = 1;
                        }
                    } else { return MTX_ERR_INVALID_PRECISION; }
                } else if (field == mtxfile_complex) {
                    if (precision == mtx_single) {
                        for (int64_t k = 0; k < size; k++) {
                            rowidx[k] = data->vector_coordinate_complex_single[k].i;
                            colidx[k] = 1;
                        }
                    } else if (precision == mtx_double) {
                        for (int64_t k = 0; k < size; k++) {
                            rowidx[k] = data->vector_coordinate_complex_double[k].i;
                            colidx[k] = 1;
                        }
                    } else { return MTX_ERR_INVALID_PRECISION; }
                } else if (field == mtxfile_integer) {
                    if (precision == mtx_single) {
                        for (int64_t k = 0; k < size; k++) {
                            rowidx[k] = data->vector_coordinate_integer_single[k].i;
                            colidx[k] = 1;
                        }
                    } else if (precision == mtx_double) {
                        for (int64_t k = 0; k < size; k++) {
                            rowidx[k] = data->vector_coordinate_integer_double[k].i;
                            colidx[k] = 1;
                        }
                    } else { return MTX_ERR_INVALID_PRECISION; }
                } else if (field == mtxfile_pattern) {
                    for (int64_t k = 0; k < size; k++) {
                        rowidx[k] = data->vector_coordinate_pattern[k].i;
                        colidx[k] = 1;
                    }
                } else { return MTX_ERR_INVALID_MTX_FIELD; }
            } else if (format == mtxfile_array) {
                if (offset < 0 || offset + size > num_rows)
                    return MTX_ERR_INDEX_OUT_OF_BOUNDS;
                for (int64_t k = offset, l = 0; l < size; k++, l++) {
                    int64_t i = k;
                    rowidx[l] = i+1;
                    colidx[l] = 1;
                }
            } else { return MTX_ERR_INVALID_MTX_FORMAT; }
        } else { return MTX_ERR_INVALID_MTX_OBJECT; }
    } else if (rowidx) {
        if (object == mtxfile_matrix) {
            if (format == mtxfile_coordinate) {
                if (field == mtxfile_real) {
                    if (precision == mtx_single) {
                        for (int64_t k = 0; k < size; k++)
                            rowidx[k] = data->matrix_coordinate_real_single[k].i;
                    } else if (precision == mtx_double) {
                        for (int64_t k = 0; k < size; k++)
                            rowidx[k] = data->matrix_coordinate_real_double[k].i;
                    } else { return MTX_ERR_INVALID_PRECISION; }
                } else if (field == mtxfile_complex) {
                    if (precision == mtx_single) {
                        for (int64_t k = 0; k < size; k++)
                            rowidx[k] = data->matrix_coordinate_complex_single[k].i;
                    } else if (precision == mtx_double) {
                        for (int64_t k = 0; k < size; k++)
                            rowidx[k] = data->matrix_coordinate_complex_double[k].i;
                    } else { return MTX_ERR_INVALID_PRECISION; }
                } else if (field == mtxfile_integer) {
                    if (precision == mtx_single) {
                        for (int64_t k = 0; k < size; k++)
                            rowidx[k] = data->matrix_coordinate_integer_single[k].i;
                    } else if (precision == mtx_double) {
                        for (int64_t k = 0; k < size; k++)
                            rowidx[k] = data->matrix_coordinate_integer_double[k].i;
                    } else { return MTX_ERR_INVALID_PRECISION; }
                } else if (field == mtxfile_pattern) {
                    for (int64_t k = 0; k < size; k++)
                        rowidx[k] = data->matrix_coordinate_pattern[k].i;
                } else { return MTX_ERR_INVALID_MTX_FIELD; }
            } else if (format == mtxfile_array) {
                if (offset < 0 || (offset + size) / num_rows > num_columns)
                    return MTX_ERR_INDEX_OUT_OF_BOUNDS;
                for (int64_t k = offset, l = 0; l < size; k++, l++) {
                    int64_t i = k / num_columns;
                    rowidx[l] = i+1;
                }
            } else { return MTX_ERR_INVALID_MTX_FORMAT; }
        } else if (object == mtxfile_vector) {
            if (format == mtxfile_coordinate) {
                if (field == mtxfile_real) {
                    if (precision == mtx_single) {
                        for (int64_t k = 0; k < size; k++)
                            rowidx[k] = data->vector_coordinate_real_single[k].i;
                    } else if (precision == mtx_double) {
                        for (int64_t k = 0; k < size; k++)
                            rowidx[k] = data->vector_coordinate_real_double[k].i;
                    } else { return MTX_ERR_INVALID_PRECISION; }
                } else if (field == mtxfile_complex) {
                    if (precision == mtx_single) {
                        for (int64_t k = 0; k < size; k++)
                            rowidx[k] = data->vector_coordinate_complex_single[k].i;
                    } else if (precision == mtx_double) {
                        for (int64_t k = 0; k < size; k++)
                            rowidx[k] = data->vector_coordinate_complex_double[k].i;
                    } else { return MTX_ERR_INVALID_PRECISION; }
                } else if (field == mtxfile_integer) {
                    if (precision == mtx_single) {
                        for (int64_t k = 0; k < size; k++)
                            rowidx[k] = data->vector_coordinate_integer_single[k].i;
                    } else if (precision == mtx_double) {
                        for (int64_t k = 0; k < size; k++)
                            rowidx[k] = data->vector_coordinate_integer_double[k].i;
                    } else { return MTX_ERR_INVALID_PRECISION; }
                } else if (field == mtxfile_pattern) {
                    for (int64_t k = 0; k < size; k++)
                        rowidx[k] = data->vector_coordinate_pattern[k].i;
                } else { return MTX_ERR_INVALID_MTX_FIELD; }
            } else if (format == mtxfile_array) {
                if (offset < 0 || offset + size > num_rows)
                    return MTX_ERR_INDEX_OUT_OF_BOUNDS;
                for (int64_t k = offset, l = 0; l < size; k++, l++) {
                    int64_t i = k;
                    rowidx[l] = i+1;
                }
            } else { return MTX_ERR_INVALID_MTX_FORMAT; }
        } else { return MTX_ERR_INVALID_MTX_OBJECT; }
    } else if (colidx) {
        if (object == mtxfile_matrix) {
            if (format == mtxfile_coordinate) {
                if (field == mtxfile_real) {
                    if (precision == mtx_single) {
                        for (int64_t k = 0; k < size; k++)
                            colidx[k] = data->matrix_coordinate_real_single[k].j;
                    } else if (precision == mtx_double) {
                        for (int64_t k = 0; k < size; k++)
                            colidx[k] = data->matrix_coordinate_real_double[k].j;
                    } else { return MTX_ERR_INVALID_PRECISION; }
                } else if (field == mtxfile_complex) {
                    if (precision == mtx_single) {
                        for (int64_t k = 0; k < size; k++)
                            colidx[k] = data->matrix_coordinate_complex_single[k].j;
                    } else if (precision == mtx_double) {
                        for (int64_t k = 0; k < size; k++)
                            colidx[k] = data->matrix_coordinate_complex_double[k].j;
                    } else { return MTX_ERR_INVALID_PRECISION; }
                } else if (field == mtxfile_integer) {
                    if (precision == mtx_single) {
                        for (int64_t k = 0; k < size; k++)
                            colidx[k] = data->matrix_coordinate_integer_single[k].j;
                    } else if (precision == mtx_double) {
                        for (int64_t k = 0; k < size; k++)
                            colidx[k] = data->matrix_coordinate_integer_double[k].j;
                    } else { return MTX_ERR_INVALID_PRECISION; }
                } else if (field == mtxfile_pattern) {
                    for (int64_t k = 0; k < size; k++)
                        colidx[k] = data->matrix_coordinate_pattern[k].j;
                } else { return MTX_ERR_INVALID_MTX_FIELD; }
            } else if (format == mtxfile_array) {
                if (offset < 0 || (offset + size) / num_rows > num_columns)
                    return MTX_ERR_INDEX_OUT_OF_BOUNDS;
                for (int64_t k = offset, l = 0; l < size; k++, l++) {
                    int64_t j = k % num_columns;
                    colidx[l] = j+1;
                }
            } else { return MTX_ERR_INVALID_MTX_FORMAT; }
        } else if (object == mtxfile_vector) {
            if (format == mtxfile_coordinate) {
                if (field == mtxfile_real ||
                    field == mtxfile_complex ||
                    field == mtxfile_integer)
                {
                    if (precision == mtx_single || precision == mtx_double) {
                        for (int64_t k = 0; k < size; k++)
                            colidx[k] = 1;
                    } else { return MTX_ERR_INVALID_PRECISION; }
                } else if (field == mtxfile_pattern) {
                    for (int64_t k = 0; k < size; k++)
                        colidx[k] = 1;
                } else { return MTX_ERR_INVALID_MTX_FIELD; }
            } else if (format == mtxfile_array) {
                if (offset < 0 || offset + size > num_rows)
                    return MTX_ERR_INDEX_OUT_OF_BOUNDS;
                for (int64_t k = offset, l = 0; l < size; k++, l++)
                    colidx[l] = 1;
            } else { return MTX_ERR_INVALID_MTX_FORMAT; }
        } else { return MTX_ERR_INVALID_MTX_OBJECT; }
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxfiledata_rowcolidx()’ extracts row and/or column indices for a
 * matrix or vector in Matrix Market format.
 *
 * ‘rowidx’ may be ‘NULL’, in which case it is ignored. Otherwise, it
 * must point to an array containing enough storage for ‘size’ values
 * of type ‘int’.  If successful, this array will contain the row
 * index of each data line.
 *
 * Similarly, ‘colidx’ may be ‘NULL’, or it must point to an array of
 * the same size, which will be used to store the column index of each
 * data line.
 *
 * Note that indexing is 1-based, meaning that rows are numbered
 * ‘1,2,...,num_rows’, whereas columns are numbered
 * ‘1,2,...,num_columns’.
 *
 * If ‘format’ is ‘mtxfile_array’, then a non-negative ‘offset’ value
 * can be used to obtain row and column indices for matrix or vector
 * entries starting from the specified offset, instead of beginning
 * with the first entry of the matrix or vector.
 */
int mtxfiledata_rowcolidx(
    const union mtxfiledata * data,
    enum mtxfileobject object,
    enum mtxfileformat format,
    enum mtxfilefield field,
    enum mtxprecision precision,
    int64_t num_rows,
    int64_t num_columns,
    int64_t offset,
    int64_t size,
    int * rowidx,
    int * colidx)
{
    int err;
    if (rowidx && colidx) {
        if (object == mtxfile_matrix) {
            if (format == mtxfile_coordinate) {
                if (field == mtxfile_real) {
                    if (precision == mtx_single) {
                        for (int64_t k = 0; k < size; k++) {
                            rowidx[k] = data->matrix_coordinate_real_single[k].i;
                            colidx[k] = data->matrix_coordinate_real_single[k].j;
                        }
                    } else if (precision == mtx_double) {
                        for (int64_t k = 0; k < size; k++) {
                            rowidx[k] = data->matrix_coordinate_real_double[k].i;
                            colidx[k] = data->matrix_coordinate_real_double[k].j;
                        }
                    } else { return MTX_ERR_INVALID_PRECISION; }
                } else if (field == mtxfile_complex) {
                    if (precision == mtx_single) {
                        for (int64_t k = 0; k < size; k++) {
                            rowidx[k] = data->matrix_coordinate_complex_single[k].i;
                            colidx[k] = data->matrix_coordinate_complex_single[k].j;
                        }
                    } else if (precision == mtx_double) {
                        for (int64_t k = 0; k < size; k++) {
                            rowidx[k] = data->matrix_coordinate_complex_double[k].i;
                            colidx[k] = data->matrix_coordinate_complex_double[k].j;
                        }
                    } else { return MTX_ERR_INVALID_PRECISION; }
                } else if (field == mtxfile_integer) {
                    if (precision == mtx_single) {
                        for (int64_t k = 0; k < size; k++) {
                            rowidx[k] = data->matrix_coordinate_integer_single[k].i;
                            colidx[k] = data->matrix_coordinate_integer_single[k].j;
                        }
                    } else if (precision == mtx_double) {
                        for (int64_t k = 0; k < size; k++) {
                            rowidx[k] = data->matrix_coordinate_integer_double[k].i;
                            colidx[k] = data->matrix_coordinate_integer_double[k].j;
                        }
                    } else { return MTX_ERR_INVALID_PRECISION; }
                } else if (field == mtxfile_pattern) {
                    for (int64_t k = 0; k < size; k++) {
                        rowidx[k] = data->matrix_coordinate_pattern[k].i;
                        colidx[k] = data->matrix_coordinate_pattern[k].j;
                    }
                } else { return MTX_ERR_INVALID_MTX_FIELD; }
            } else if (format == mtxfile_array) {
                if (offset < 0 || (offset + size) / num_rows > num_columns)
                    return MTX_ERR_INDEX_OUT_OF_BOUNDS;
                for (int64_t k = offset, l = 0; l < size; k++, l++) {
                    int64_t i = k / num_columns;
                    int64_t j = k % num_columns;
                    rowidx[l] = i+1;
                    colidx[l] = j+1;
                }
            } else { return MTX_ERR_INVALID_MTX_FORMAT; }
        } else if (object == mtxfile_vector) {
            if (format == mtxfile_coordinate) {
                if (field == mtxfile_real) {
                    if (precision == mtx_single) {
                        for (int64_t k = 0; k < size; k++) {
                            rowidx[k] = data->vector_coordinate_real_single[k].i;
                            colidx[k] = 1;
                        }
                    } else if (precision == mtx_double) {
                        for (int64_t k = 0; k < size; k++) {
                            rowidx[k] = data->vector_coordinate_real_double[k].i;
                            colidx[k] = 1;
                        }
                    } else { return MTX_ERR_INVALID_PRECISION; }
                } else if (field == mtxfile_complex) {
                    if (precision == mtx_single) {
                        for (int64_t k = 0; k < size; k++) {
                            rowidx[k] = data->vector_coordinate_complex_single[k].i;
                            colidx[k] = 1;
                        }
                    } else if (precision == mtx_double) {
                        for (int64_t k = 0; k < size; k++) {
                            rowidx[k] = data->vector_coordinate_complex_double[k].i;
                            colidx[k] = 1;
                        }
                    } else { return MTX_ERR_INVALID_PRECISION; }
                } else if (field == mtxfile_integer) {
                    if (precision == mtx_single) {
                        for (int64_t k = 0; k < size; k++) {
                            rowidx[k] = data->vector_coordinate_integer_single[k].i;
                            colidx[k] = 1;
                        }
                    } else if (precision == mtx_double) {
                        for (int64_t k = 0; k < size; k++) {
                            rowidx[k] = data->vector_coordinate_integer_double[k].i;
                            colidx[k] = 1;
                        }
                    } else { return MTX_ERR_INVALID_PRECISION; }
                } else if (field == mtxfile_pattern) {
                    for (int64_t k = 0; k < size; k++) {
                        rowidx[k] = data->vector_coordinate_pattern[k].i;
                        colidx[k] = 1;
                    }
                } else { return MTX_ERR_INVALID_MTX_FIELD; }
            } else if (format == mtxfile_array) {
                if (offset < 0 || offset + size > num_rows)
                    return MTX_ERR_INDEX_OUT_OF_BOUNDS;
                for (int64_t k = offset, l = 0; l < size; k++, l++) {
                    int i = k;
                    rowidx[l] = i+1;
                    colidx[l] = 1;
                }
            } else { return MTX_ERR_INVALID_MTX_FORMAT; }
        } else { return MTX_ERR_INVALID_MTX_OBJECT; }
    } else if (rowidx) {
        if (object == mtxfile_matrix) {
            if (format == mtxfile_coordinate) {
                if (field == mtxfile_real) {
                    if (precision == mtx_single) {
                        for (int64_t k = 0; k < size; k++)
                            rowidx[k] = data->matrix_coordinate_real_single[k].i;
                    } else if (precision == mtx_double) {
                        for (int64_t k = 0; k < size; k++)
                            rowidx[k] = data->matrix_coordinate_real_double[k].i;
                    } else { return MTX_ERR_INVALID_PRECISION; }
                } else if (field == mtxfile_complex) {
                    if (precision == mtx_single) {
                        for (int64_t k = 0; k < size; k++)
                            rowidx[k] = data->matrix_coordinate_complex_single[k].i;
                    } else if (precision == mtx_double) {
                        for (int64_t k = 0; k < size; k++)
                            rowidx[k] = data->matrix_coordinate_complex_double[k].i;
                    } else { return MTX_ERR_INVALID_PRECISION; }
                } else if (field == mtxfile_integer) {
                    if (precision == mtx_single) {
                        for (int64_t k = 0; k < size; k++)
                            rowidx[k] = data->matrix_coordinate_integer_single[k].i;
                    } else if (precision == mtx_double) {
                        for (int64_t k = 0; k < size; k++)
                            rowidx[k] = data->matrix_coordinate_integer_double[k].i;
                    } else { return MTX_ERR_INVALID_PRECISION; }
                } else if (field == mtxfile_pattern) {
                    for (int64_t k = 0; k < size; k++)
                        rowidx[k] = data->matrix_coordinate_pattern[k].i;
                } else { return MTX_ERR_INVALID_MTX_FIELD; }
            } else if (format == mtxfile_array) {
                if (offset < 0 || (offset + size) / num_rows > num_columns)
                    return MTX_ERR_INDEX_OUT_OF_BOUNDS;
                for (int64_t k = offset, l = 0; l < size; k++, l++) {
                    int64_t i = k / num_columns;
                    rowidx[l] = i+1;
                }
            } else { return MTX_ERR_INVALID_MTX_FORMAT; }
        } else if (object == mtxfile_vector) {
            if (format == mtxfile_coordinate) {
                if (field == mtxfile_real) {
                    if (precision == mtx_single) {
                        for (int64_t k = 0; k < size; k++)
                            rowidx[k] = data->vector_coordinate_real_single[k].i;
                    } else if (precision == mtx_double) {
                        for (int64_t k = 0; k < size; k++)
                            rowidx[k] = data->vector_coordinate_real_double[k].i;
                    } else { return MTX_ERR_INVALID_PRECISION; }
                } else if (field == mtxfile_complex) {
                    if (precision == mtx_single) {
                        for (int64_t k = 0; k < size; k++)
                            rowidx[k] = data->vector_coordinate_complex_single[k].i;
                    } else if (precision == mtx_double) {
                        for (int64_t k = 0; k < size; k++)
                            rowidx[k] = data->vector_coordinate_complex_double[k].i;
                    } else { return MTX_ERR_INVALID_PRECISION; }
                } else if (field == mtxfile_integer) {
                    if (precision == mtx_single) {
                        for (int64_t k = 0; k < size; k++)
                            rowidx[k] = data->vector_coordinate_integer_single[k].i;
                    } else if (precision == mtx_double) {
                        for (int64_t k = 0; k < size; k++)
                            rowidx[k] = data->vector_coordinate_integer_double[k].i;
                    } else { return MTX_ERR_INVALID_PRECISION; }
                } else if (field == mtxfile_pattern) {
                    for (int64_t k = 0; k < size; k++)
                        rowidx[k] = data->vector_coordinate_pattern[k].i;
                } else { return MTX_ERR_INVALID_MTX_FIELD; }
            } else if (format == mtxfile_array) {
                if (offset < 0 || offset + size > num_rows)
                    return MTX_ERR_INDEX_OUT_OF_BOUNDS;
                for (int64_t k = offset, l = 0; l < size; k++, l++) {
                    int i = k;
                    rowidx[l] = i+1;
                }
            } else { return MTX_ERR_INVALID_MTX_FORMAT; }
        } else { return MTX_ERR_INVALID_MTX_OBJECT; }
    } else if (colidx) {
        if (object == mtxfile_matrix) {
            if (format == mtxfile_coordinate) {
                if (field == mtxfile_real) {
                    if (precision == mtx_single) {
                        for (int64_t k = 0; k < size; k++)
                            colidx[k] = data->matrix_coordinate_real_single[k].j;
                    } else if (precision == mtx_double) {
                        for (int64_t k = 0; k < size; k++)
                            colidx[k] = data->matrix_coordinate_real_double[k].j;
                    } else { return MTX_ERR_INVALID_PRECISION; }
                } else if (field == mtxfile_complex) {
                    if (precision == mtx_single) {
                        for (int64_t k = 0; k < size; k++)
                            colidx[k] = data->matrix_coordinate_complex_single[k].j;
                    } else if (precision == mtx_double) {
                        for (int64_t k = 0; k < size; k++)
                            colidx[k] = data->matrix_coordinate_complex_double[k].j;
                    } else { return MTX_ERR_INVALID_PRECISION; }
                } else if (field == mtxfile_integer) {
                    if (precision == mtx_single) {
                        for (int64_t k = 0; k < size; k++)
                            colidx[k] = data->matrix_coordinate_integer_single[k].j;
                    } else if (precision == mtx_double) {
                        for (int64_t k = 0; k < size; k++)
                            colidx[k] = data->matrix_coordinate_integer_double[k].j;
                    } else { return MTX_ERR_INVALID_PRECISION; }
                } else if (field == mtxfile_pattern) {
                    for (int64_t k = 0; k < size; k++)
                        colidx[k] = data->matrix_coordinate_pattern[k].j;
                } else { return MTX_ERR_INVALID_MTX_FIELD; }
            } else if (format == mtxfile_array) {
                if (offset < 0 || (offset + size) / num_rows > num_columns)
                    return MTX_ERR_INDEX_OUT_OF_BOUNDS;
                for (int64_t k = offset, l = 0; l < size; k++, l++) {
                    int64_t j = k % num_columns;
                    colidx[l] = j+1;
                }
            } else { return MTX_ERR_INVALID_MTX_FORMAT; }
        } else if (object == mtxfile_vector) {
            if (format == mtxfile_coordinate) {
                if (field == mtxfile_real ||
                    field == mtxfile_complex ||
                    field == mtxfile_integer)
                {
                    if (precision == mtx_single || precision == mtx_double) {
                        for (int64_t k = 0; k < size; k++)
                            colidx[k] = 1;
                    } else { return MTX_ERR_INVALID_PRECISION; }
                } else if (field == mtxfile_pattern) {
                    for (int64_t k = 0; k < size; k++)
                        colidx[k] = 1;
                } else { return MTX_ERR_INVALID_MTX_FIELD; }
            } else if (format == mtxfile_array) {
                if (offset < 0 || offset + size > num_rows)
                    return MTX_ERR_INDEX_OUT_OF_BOUNDS;
                for (int64_t k = offset, l = 0; l < size; k++, l++)
                    colidx[l] = 1;
            } else { return MTX_ERR_INVALID_MTX_FORMAT; }
        } else { return MTX_ERR_INVALID_MTX_OBJECT; }
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxfiledata_rowptr()’ computes row pointers for a matrix in
 * coordinate format.
 *
 * ‘rowptr’ must point to an array containing enough storage for
 * ‘num_rows+1’ values of type ‘int64_t’.
 *
 * ‘colidx’ may be ‘NULL’, in which case it is ignored. Otherwise, it
 * must point to an array containing enough storage for ‘size’ values
 * of type ‘int’.  On successful completion, this array will contain
 * the column indices of the nonzero matrix entries arranged rowwise.
 * The order of nonzeros within each row remains unchanged. The ‘i’-th
 * entry of ‘rowptr’ is the location in the ‘colidx’ array of the
 * first nonzero that belongs to the ‘i+1’-th row of the matrix, for
 * ‘i=0,1,...,num_rows-1’.  The final entry of ‘rowptr’ indicates the
 * position one place beyond the last nonzero.
 *
 * This function does not require the matrix data to be sorted in any
 * particular order beforehand.
 */
int mtxfiledata_rowptr(
    const union mtxfiledata * srcdata,
    enum mtxfileobject object,
    enum mtxfileformat format,
    enum mtxfilefield field,
    enum mtxprecision precision,
    int64_t num_rows,
    int64_t size,
    int64_t * rowptr,
    int * colidx,
    void * dstdata)
{
    if (object != mtxfile_matrix)
        return MTX_ERR_INCOMPATIBLE_MTX_OBJECT;
    if (format != mtxfile_coordinate)
        return MTX_ERR_INCOMPATIBLE_MTX_FORMAT;

    for (int64_t i = 0; i <= num_rows; i++)
        rowptr[i] = 0;
    if (field == mtxfile_real) {
        if (precision == mtx_single) {
            for (int64_t k = 0; k < size; k++)
                rowptr[srcdata->matrix_coordinate_real_single[k].i]++;
        } else if (precision == mtx_double) {
            for (int64_t k = 0; k < size; k++)
                rowptr[srcdata->matrix_coordinate_real_double[k].i]++;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (field == mtxfile_complex) {
        if (precision == mtx_single) {
            for (int64_t k = 0; k < size; k++)
                rowptr[srcdata->matrix_coordinate_complex_single[k].i]++;
        } else if (precision == mtx_double) {
            for (int64_t k = 0; k < size; k++)
                rowptr[srcdata->matrix_coordinate_complex_double[k].i]++;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (field == mtxfile_integer) {
        if (precision == mtx_single) {
            for (int64_t k = 0; k < size; k++)
                rowptr[srcdata->matrix_coordinate_integer_single[k].i]++;
        } else if (precision == mtx_double) {
            for (int64_t k = 0; k < size; k++)
                rowptr[srcdata->matrix_coordinate_integer_double[k].i]++;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (field == mtxfile_pattern) {
        for (int64_t k = 0; k < size; k++)
            rowptr[srcdata->matrix_coordinate_pattern[k].i]++;
    } else { return MTX_ERR_INVALID_MTX_FIELD; }
    for (int64_t i = 1; i <= num_rows; i++)
        rowptr[i] += rowptr[i-1];

    /* sort column indices rowwise */
    if (colidx) {
        if (field == mtxfile_real) {
            if (precision == mtx_single) {
                const struct mtxfile_matrix_coordinate_real_single * src =
                    srcdata->matrix_coordinate_real_single;
                for (int64_t k = 0; k < size; k++)
                    colidx[rowptr[src[k].i-1]++] = src[k].j;
            } else if (precision == mtx_double) {
                const struct mtxfile_matrix_coordinate_real_double * src =
                    srcdata->matrix_coordinate_real_double;
                for (int64_t k = 0; k < size; k++)
                    colidx[rowptr[src[k].i-1]++] = src[k].j;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (field == mtxfile_complex) {
            if (precision == mtx_single) {
                const struct mtxfile_matrix_coordinate_complex_single * src =
                    srcdata->matrix_coordinate_complex_single;
                for (int64_t k = 0; k < size; k++)
                    colidx[rowptr[src[k].i-1]++] = src[k].j;
            } else if (precision == mtx_double) {
                const struct mtxfile_matrix_coordinate_complex_double * src =
                    srcdata->matrix_coordinate_complex_double;
                for (int64_t k = 0; k < size; k++)
                    colidx[rowptr[src[k].i-1]++] = src[k].j;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (field == mtxfile_integer) {
            if (precision == mtx_single) {
                const struct mtxfile_matrix_coordinate_integer_single * src =
                    srcdata->matrix_coordinate_integer_single;
                for (int64_t k = 0; k < size; k++)
                    colidx[rowptr[src[k].i-1]++] = src[k].j;
            } else if (precision == mtx_double) {
                const struct mtxfile_matrix_coordinate_integer_double * src =
                    srcdata->matrix_coordinate_integer_double;
                for (int64_t k = 0; k < size; k++)
                    colidx[rowptr[src[k].i-1]++] = src[k].j;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (field == mtxfile_pattern) {
            const struct mtxfile_matrix_coordinate_pattern * src =
                srcdata->matrix_coordinate_pattern;
            for (int64_t k = 0; k < size; k++)
                colidx[rowptr[src[k].i-1]++] = src[k].j;
        } else { return MTX_ERR_INVALID_MTX_FIELD; }
        for (int64_t i = num_rows; i > 0; i--)
            rowptr[i] = rowptr[i-1];
        rowptr[0] = 0;
    }

    /* sort nonzero data rowwise */
    if (dstdata) {
        if (field == mtxfile_real) {
            if (precision == mtx_single) {
                const struct mtxfile_matrix_coordinate_real_single * src =
                    srcdata->matrix_coordinate_real_single;
                float * dst = dstdata;
                for (int64_t k = 0; k < size; k++)
                    dst[rowptr[src[k].i-1]++] = src[k].a;
            } else if (precision == mtx_double) {
                const struct mtxfile_matrix_coordinate_real_double * src =
                    srcdata->matrix_coordinate_real_double;
                double * dst = dstdata;
                for (int64_t k = 0; k < size; k++)
                    dst[rowptr[src[k].i-1]++] = src[k].a;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (field == mtxfile_complex) {
            if (precision == mtx_single) {
                const struct mtxfile_matrix_coordinate_complex_single * src =
                    srcdata->matrix_coordinate_complex_single;
                float (* dst)[2] = dstdata;
                for (int64_t k = 0; k < size; k++) {
                    dst[rowptr[src[k].i-1]  ][0] = src[k].a[0];
                    dst[rowptr[src[k].i-1]++][1] = src[k].a[1];
                }
            } else if (precision == mtx_double) {
                const struct mtxfile_matrix_coordinate_complex_double * src =
                    srcdata->matrix_coordinate_complex_double;
                double (* dst)[2] = dstdata;
                for (int64_t k = 0; k < size; k++) {
                    dst[rowptr[src[k].i-1]  ][0] = src[k].a[0];
                    dst[rowptr[src[k].i-1]++][1] = src[k].a[1];
                }
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (field == mtxfile_integer) {
            if (precision == mtx_single) {
                const struct mtxfile_matrix_coordinate_integer_single * src =
                    srcdata->matrix_coordinate_integer_single;
                int32_t * dst = dstdata;
                for (int64_t k = 0; k < size; k++)
                    dst[rowptr[src[k].i-1]++] = src[k].a;
            } else if (precision == mtx_double) {
                const struct mtxfile_matrix_coordinate_integer_double * src =
                    srcdata->matrix_coordinate_integer_double;
                int64_t * dst = dstdata;
                for (int64_t k = 0; k < size; k++)
                    dst[rowptr[src[k].i-1]++] = src[k].a;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (field == mtxfile_pattern) {
            /* nothing to be done */
        } else { return MTX_ERR_INVALID_MTX_FIELD; }
        if (field != mtxfile_pattern) {
            for (int64_t i = num_rows; i > 0; i--)
                rowptr[i] = rowptr[i-1];
            rowptr[0] = 0;
        }
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxfiledata_colptr()’ computes column pointers for a matrix in
 * coordinate format.
 *
 * ‘colptr’ must point to an array containing enough storage for
 * ‘num_columns+1’ values of type ‘int64_t’.
 *
 * ‘rowidx’ may be ‘NULL’, in which case it is ignored. Otherwise, it
 * must point to an array containing enough storage for ‘size’ values
 * of type ‘int’. On successful completion, this array will contain
 * the row indices of the nonzero matrix entries arranged
 * columnwise. The order of nonzeros within each row remains
 * unchanged. The ‘j’-th entry of ‘colptr’ is the location in the
 * ‘rowidx’ array of the first nonzero that belongs to the ‘j+1’-th
 * column of the matrix, for ‘i=0,1,...,num_columns-1’. The final
 * entry of ‘colptr’ indicates the position one place beyond the last
 * nonzero.
 *
 * The matrix data is not required to be sorted in any particular
 * order.
 */
int mtxfiledata_colptr(
    const union mtxfiledata * srcdata,
    enum mtxfileobject object,
    enum mtxfileformat format,
    enum mtxfilefield field,
    enum mtxprecision precision,
    int64_t num_columns,
    int64_t size,
    int64_t * colptr,
    int * rowidx,
    void * dstdata)
{
    if (object != mtxfile_matrix)
        return MTX_ERR_INCOMPATIBLE_MTX_OBJECT;
    if (format != mtxfile_coordinate)
        return MTX_ERR_INCOMPATIBLE_MTX_FORMAT;
    for (int64_t j = 0; j <= num_columns; j++)
        colptr[j] = 0;
    if (field == mtxfile_real) {
        if (precision == mtx_single) {
            for (int64_t k = 0; k < size; k++)
                colptr[srcdata->matrix_coordinate_real_single[k].j]++;
        } else if (precision == mtx_double) {
            for (int64_t k = 0; k < size; k++)
                colptr[srcdata->matrix_coordinate_real_double[k].j]++;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (field == mtxfile_complex) {
        if (precision == mtx_single) {
            for (int64_t k = 0; k < size; k++)
                colptr[srcdata->matrix_coordinate_complex_single[k].j]++;
        } else if (precision == mtx_double) {
            for (int64_t k = 0; k < size; k++)
                colptr[srcdata->matrix_coordinate_complex_double[k].j]++;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (field == mtxfile_integer) {
        if (precision == mtx_single) {
            for (int64_t k = 0; k < size; k++)
                colptr[srcdata->matrix_coordinate_integer_single[k].j]++;
        } else if (precision == mtx_double) {
            for (int64_t k = 0; k < size; k++)
                colptr[srcdata->matrix_coordinate_integer_double[k].j]++;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (field == mtxfile_pattern) {
        for (int64_t k = 0; k < size; k++)
            colptr[srcdata->matrix_coordinate_pattern[k].j]++;
    } else { return MTX_ERR_INVALID_MTX_FIELD; }
    for (int64_t j = 1; j <= num_columns; j++)
        colptr[j] += colptr[j-1];

    /* sort row indices columnwise */
    if (rowidx) {
        if (field == mtxfile_real) {
            if (precision == mtx_single) {
                const struct mtxfile_matrix_coordinate_real_single * src =
                    srcdata->matrix_coordinate_real_single;
                for (int64_t k = 0; k < size; k++)
                    rowidx[colptr[src[k].j-1]++] = src[k].i;
            } else if (precision == mtx_double) {
                const struct mtxfile_matrix_coordinate_real_double * src =
                    srcdata->matrix_coordinate_real_double;
                for (int64_t k = 0; k < size; k++)
                    rowidx[colptr[src[k].j-1]++] = src[k].i;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (field == mtxfile_complex) {
            if (precision == mtx_single) {
                const struct mtxfile_matrix_coordinate_complex_single * src =
                    srcdata->matrix_coordinate_complex_single;
                for (int64_t k = 0; k < size; k++)
                    rowidx[colptr[src[k].j-1]++] = src[k].i;
            } else if (precision == mtx_double) {
                const struct mtxfile_matrix_coordinate_complex_double * src =
                    srcdata->matrix_coordinate_complex_double;
                for (int64_t k = 0; k < size; k++)
                    rowidx[colptr[src[k].j-1]++] = src[k].i;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (field == mtxfile_integer) {
            if (precision == mtx_single) {
                const struct mtxfile_matrix_coordinate_integer_single * src =
                    srcdata->matrix_coordinate_integer_single;
                for (int64_t k = 0; k < size; k++)
                    rowidx[colptr[src[k].j-1]++] = src[k].i;
            } else if (precision == mtx_double) {
                const struct mtxfile_matrix_coordinate_integer_double * src =
                    srcdata->matrix_coordinate_integer_double;
                for (int64_t k = 0; k < size; k++)
                    rowidx[colptr[src[k].j-1]++] = src[k].i;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (field == mtxfile_pattern) {
            const struct mtxfile_matrix_coordinate_pattern * src =
                srcdata->matrix_coordinate_pattern;
            for (int64_t k = 0; k < size; k++)
                rowidx[colptr[src[k].j-1]++] = src[k].i;
        } else { return MTX_ERR_INVALID_MTX_FIELD; }
        for (int64_t i = num_columns; i > 0; i--)
            colptr[i] = colptr[i-1];
        colptr[0] = 0;
    }

    /* sort nonzero data rowwise */
    if (dstdata) {
        if (field == mtxfile_real) {
            if (precision == mtx_single) {
                const struct mtxfile_matrix_coordinate_real_single * src =
                    srcdata->matrix_coordinate_real_single;
                float * dst = dstdata;
                for (int64_t k = 0; k < size; k++)
                    dst[colptr[src[k].j-1]++] = src[k].a;
            } else if (precision == mtx_double) {
                const struct mtxfile_matrix_coordinate_real_double * src =
                    srcdata->matrix_coordinate_real_double;
                double * dst = dstdata;
                for (int64_t k = 0; k < size; k++)
                    dst[colptr[src[k].j-1]++] = src[k].a;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (field == mtxfile_complex) {
            if (precision == mtx_single) {
                const struct mtxfile_matrix_coordinate_complex_single * src =
                    srcdata->matrix_coordinate_complex_single;
                float (* dst)[2] = dstdata;
                for (int64_t k = 0; k < size; k++) {
                    dst[colptr[src[k].j-1]++][0] = src[k].a[0];
                    dst[colptr[src[k].j-1]++][1] = src[k].a[1];
                }
            } else if (precision == mtx_double) {
                const struct mtxfile_matrix_coordinate_complex_double * src =
                    srcdata->matrix_coordinate_complex_double;
                double (* dst)[2] = dstdata;
                for (int64_t k = 0; k < size; k++) {
                    dst[colptr[src[k].j-1]++][0] = src[k].a[0];
                    dst[colptr[src[k].j-1]++][1] = src[k].a[1];
                }
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (field == mtxfile_integer) {
            if (precision == mtx_single) {
                const struct mtxfile_matrix_coordinate_integer_single * src =
                    srcdata->matrix_coordinate_integer_single;
                int32_t * dst = dstdata;
                for (int64_t k = 0; k < size; k++)
                    dst[colptr[src[k].j-1]++] = src[k].a;
            } else if (precision == mtx_double) {
                const struct mtxfile_matrix_coordinate_integer_double * src =
                    srcdata->matrix_coordinate_integer_double;
                int64_t * dst = dstdata;
                for (int64_t k = 0; k < size; k++)
                    dst[colptr[src[k].j-1]++] = src[k].a;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (field == mtxfile_pattern) {
            /* nothing to be done */
        } else { return MTX_ERR_INVALID_MTX_FIELD; }
        if (field != mtxfile_pattern) {
            for (int64_t i = num_columns; i > 0; i--)
                colptr[i] = colptr[i-1];
            colptr[0] = 0;
        }
    }
    return MTX_SUCCESS;
}

/*
 * Modifying values
 */

/**
 * ‘mtxfiledata_set_constant_real_single()’ sets every (nonzero)
 * value of a matrix or vector equal to a constant, single precision
 * floating point number.
 */
int mtxfiledata_set_constant_real_single(
    union mtxfiledata * data,
    enum mtxfileobject object,
    enum mtxfileformat format,
    enum mtxfilefield field,
    enum mtxprecision precision,
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
            } else { return MTX_ERR_INVALID_PRECISION; }
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
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (field == mtxfile_integer) {
            if (precision == mtx_single) {
                for (int64_t k = 0; k < size; k++)
                    data->array_integer_single[offset+k] = a;
            } else if (precision == mtx_double) {
                for (int64_t k = 0; k < size; k++)
                    data->array_integer_double[offset+k] = a;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else { return MTX_ERR_INVALID_MTX_FIELD; }
    } else if (format == mtxfile_coordinate) {
        if (object == mtxfile_matrix) {
            if (field == mtxfile_real) {
                if (precision == mtx_single) {
                    for (int64_t k = 0; k < size; k++)
                        data->matrix_coordinate_real_single[offset+k].a = a;
                } else if (precision == mtx_double) {
                    for (int64_t k = 0; k < size; k++)
                        data->matrix_coordinate_real_double[offset+k].a = a;
                } else { return MTX_ERR_INVALID_PRECISION; }
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
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_integer) {
                if (precision == mtx_single) {
                    for (int64_t k = 0; k < size; k++)
                        data->matrix_coordinate_integer_single[offset+k].a = a;
                } else if (precision == mtx_double) {
                    for (int64_t k = 0; k < size; k++)
                        data->matrix_coordinate_integer_double[offset+k].a = a;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_pattern) {
                /* Nothing to be done */
            } else { return MTX_ERR_INVALID_MTX_FIELD; }
        } else if (object == mtxfile_vector) {
            if (field == mtxfile_real) {
                if (precision == mtx_single) {
                    for (int64_t k = 0; k < size; k++)
                        data->vector_coordinate_real_single[offset+k].a = a;
                } else if (precision == mtx_double) {
                    for (int64_t k = 0; k < size; k++)
                        data->vector_coordinate_real_double[offset+k].a = a;
                } else { return MTX_ERR_INVALID_PRECISION; }
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
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_integer) {
                if (precision == mtx_single) {
                    for (int64_t k = 0; k < size; k++)
                        data->vector_coordinate_integer_single[offset+k].a = a;
                } else if (precision == mtx_double) {
                    for (int64_t k = 0; k < size; k++)
                        data->vector_coordinate_integer_double[offset+k].a = a;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_pattern) {
                /* Nothing to be done */
            } else { return MTX_ERR_INVALID_MTX_FIELD; }
        } else { return MTX_ERR_INVALID_MTX_OBJECT; }
    } else { return MTX_ERR_INVALID_MTX_FORMAT; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxfiledata_set_constant_real_double()’ sets every (nonzero)
 * value of a matrix or vector equal to a constant, double precision
 * floating point number.
 */
int mtxfiledata_set_constant_real_double(
    union mtxfiledata * data,
    enum mtxfileobject object,
    enum mtxfileformat format,
    enum mtxfilefield field,
    enum mtxprecision precision,
    int64_t size,
    int64_t offset,
    double a)
{
    if (format == mtxfile_array) {
        if (field == mtxfile_real) {
            if (precision == mtx_single) {
                for (int64_t k = 0; k < size; k++)
                    data->array_real_single[offset+k] = a;
            } else if (precision == mtx_double) {
                for (int64_t k = 0; k < size; k++)
                    data->array_real_double[offset+k] = a;
            } else { return MTX_ERR_INVALID_PRECISION; }
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
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (field == mtxfile_integer) {
            if (precision == mtx_single) {
                for (int64_t k = 0; k < size; k++)
                    data->array_integer_single[offset+k] = a;
            } else if (precision == mtx_double) {
                for (int64_t k = 0; k < size; k++)
                    data->array_integer_double[offset+k] = a;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else { return MTX_ERR_INVALID_MTX_FIELD; }
    } else if (format == mtxfile_coordinate) {
        if (object == mtxfile_matrix) {
            if (field == mtxfile_real) {
                if (precision == mtx_single) {
                    for (int64_t k = 0; k < size; k++)
                        data->matrix_coordinate_real_single[offset+k].a = a;
                } else if (precision == mtx_double) {
                    for (int64_t k = 0; k < size; k++)
                        data->matrix_coordinate_real_double[offset+k].a = a;
                } else { return MTX_ERR_INVALID_PRECISION; }
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
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_integer) {
                if (precision == mtx_single) {
                    for (int64_t k = 0; k < size; k++)
                        data->matrix_coordinate_integer_single[offset+k].a = a;
                } else if (precision == mtx_double) {
                    for (int64_t k = 0; k < size; k++)
                        data->matrix_coordinate_integer_double[offset+k].a = a;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_pattern) {
                return MTX_ERR_INCOMPATIBLE_FIELD;
            } else { return MTX_ERR_INVALID_MTX_FIELD; }
        } else if (object == mtxfile_vector) {
            if (field == mtxfile_real) {
                if (precision == mtx_single) {
                    for (int64_t k = 0; k < size; k++)
                        data->vector_coordinate_real_single[offset+k].a = a;
                } else if (precision == mtx_double) {
                    for (int64_t k = 0; k < size; k++)
                        data->vector_coordinate_real_double[offset+k].a = a;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_complex) {
                if (precision == mtx_single) {
                    for (int64_t k = 0; k < size; k++) {
                        data->vector_coordinate_complex_single[offset+k].a[0] = a;
                        data->vector_coordinate_complex_single[offset+k].a[1] = 0;
                    }
                } else if (precision == mtx_double) {
                    for (int64_t k = 0; k < size; k++) {
                        data->vector_coordinate_complex_double[offset+k].a[0] = a;
                        data->vector_coordinate_complex_double[offset+k].a[1] = 0;
                    }
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_integer) {
                if (precision == mtx_single) {
                    for (int64_t k = 0; k < size; k++)
                        data->vector_coordinate_integer_single[offset+k].a = a;
                } else if (precision == mtx_double) {
                    for (int64_t k = 0; k < size; k++)
                        data->vector_coordinate_integer_double[offset+k].a = a;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_pattern) {
                return MTX_ERR_INCOMPATIBLE_FIELD;
            } else { return MTX_ERR_INVALID_MTX_FIELD; }
        } else { return MTX_ERR_INVALID_MTX_OBJECT; }
    } else { return MTX_ERR_INVALID_MTX_FORMAT; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxfiledata_set_constant_complex_single()’ sets every (nonzero)
 * value of a matrix or vector equal to a constant, single precision
 * floating point complex number.
 */
int mtxfiledata_set_constant_complex_single(
    union mtxfiledata * data,
    enum mtxfileobject object,
    enum mtxfileformat format,
    enum mtxfilefield field,
    enum mtxprecision precision,
    int64_t size,
    int64_t offset,
    float a[2])
{
    if (format == mtxfile_array) {
        if (field == mtxfile_real || field == mtxfile_integer) {
            return MTX_ERR_INCOMPATIBLE_FIELD;
        } else if (field == mtxfile_complex) {
            if (precision == mtx_single) {
                for (int64_t k = 0; k < size; k++) {
                    data->array_complex_single[offset+k][0] = a[0];
                    data->array_complex_single[offset+k][1] = a[1];
                }
            } else if (precision == mtx_double) {
                for (int64_t k = 0; k < size; k++) {
                    data->array_complex_double[offset+k][0] = a[0];
                    data->array_complex_double[offset+k][1] = a[1];
                }
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else { return MTX_ERR_INVALID_MTX_FIELD; }
    } else if (format == mtxfile_coordinate) {
        if (object == mtxfile_matrix) {
            if (field == mtxfile_real ||
                field == mtxfile_integer ||
                field == mtxfile_pattern)
            {
                return MTX_ERR_INCOMPATIBLE_FIELD;
            } else if (field == mtxfile_complex) {
                if (precision == mtx_single) {
                    for (int64_t k = 0; k < size; k++) {
                        data->matrix_coordinate_complex_single[offset+k].a[0] = a[0];
                        data->matrix_coordinate_complex_single[offset+k].a[1] = a[1];
                    }
                } else if (precision == mtx_double) {
                    for (int64_t k = 0; k < size; k++) {
                        data->matrix_coordinate_complex_double[offset+k].a[0] = a[0];
                        data->matrix_coordinate_complex_double[offset+k].a[1] = a[1];
                    }
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else { return MTX_ERR_INVALID_MTX_FIELD; }
        } else if (object == mtxfile_vector) {
            if (field == mtxfile_real ||
                field == mtxfile_integer ||
                field == mtxfile_pattern)
            {
                return MTX_ERR_INCOMPATIBLE_FIELD;
            } else if (field == mtxfile_complex) {
                if (precision == mtx_single) {
                    for (int64_t k = 0; k < size; k++) {
                        data->vector_coordinate_complex_single[offset+k].a[0] = a[0];
                        data->vector_coordinate_complex_single[offset+k].a[1] = a[1];
                    }
                } else if (precision == mtx_double) {
                    for (int64_t k = 0; k < size; k++) {
                        data->vector_coordinate_complex_double[offset+k].a[0] = a[0];
                        data->vector_coordinate_complex_double[offset+k].a[1] = a[1];
                    }
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else { return MTX_ERR_INVALID_MTX_FIELD; }
        } else { return MTX_ERR_INVALID_MTX_OBJECT; }
    } else { return MTX_ERR_INVALID_MTX_FORMAT; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxfiledata_set_constant_complex_double()’ sets every (nonzero)
 * value of a matrix or vector equal to a constant, double precision
 * floating point complex number.
 */
int mtxfiledata_set_constant_complex_double(
    union mtxfiledata * data,
    enum mtxfileobject object,
    enum mtxfileformat format,
    enum mtxfilefield field,
    enum mtxprecision precision,
    int64_t size,
    int64_t offset,
    double a[2])
{
    if (format == mtxfile_array) {
        if (field == mtxfile_real || field == mtxfile_integer) {
            return MTX_ERR_INCOMPATIBLE_FIELD;
        } else if (field == mtxfile_complex) {
            if (precision == mtx_single) {
                for (int64_t k = 0; k < size; k++) {
                    data->array_complex_single[offset+k][0] = a[0];
                    data->array_complex_single[offset+k][1] = a[1];
                }
            } else if (precision == mtx_double) {
                for (int64_t k = 0; k < size; k++) {
                    data->array_complex_double[offset+k][0] = a[0];
                    data->array_complex_double[offset+k][1] = a[1];
                }
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else { return MTX_ERR_INVALID_MTX_FIELD; }
    } else if (format == mtxfile_coordinate) {
        if (object == mtxfile_matrix) {
            if (field == mtxfile_real ||
                field == mtxfile_integer ||
                field == mtxfile_pattern)
            {
                return MTX_ERR_INCOMPATIBLE_FIELD;
            } else if (field == mtxfile_complex) {
                if (precision == mtx_single) {
                    for (int64_t k = 0; k < size; k++) {
                        data->matrix_coordinate_complex_single[offset+k].a[0] = a[0];
                        data->matrix_coordinate_complex_single[offset+k].a[1] = a[1];
                    }
                } else if (precision == mtx_double) {
                    for (int64_t k = 0; k < size; k++) {
                        data->matrix_coordinate_complex_double[offset+k].a[0] = a[0];
                        data->matrix_coordinate_complex_double[offset+k].a[1] = a[1];
                    }
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else { return MTX_ERR_INVALID_MTX_FIELD; }
        } else if (object == mtxfile_vector) {
            if (field == mtxfile_real ||
                field == mtxfile_integer ||
                field == mtxfile_pattern)
            {
                return MTX_ERR_INCOMPATIBLE_FIELD;
            } else if (field == mtxfile_complex) {
                if (precision == mtx_single) {
                    for (int64_t k = 0; k < size; k++) {
                        data->vector_coordinate_complex_single[offset+k].a[0] = a[0];
                        data->vector_coordinate_complex_single[offset+k].a[1] = a[1];
                    }
                } else if (precision == mtx_double) {
                    for (int64_t k = 0; k < size; k++) {
                        data->vector_coordinate_complex_double[offset+k].a[0] = a[0];
                        data->vector_coordinate_complex_double[offset+k].a[1] = a[1];
                    }
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else { return MTX_ERR_INVALID_MTX_FIELD; }
        } else { return MTX_ERR_INVALID_MTX_OBJECT; }
    } else { return MTX_ERR_INVALID_MTX_FORMAT; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxfiledata_set_constant_integer_single()’ sets every (nonzero)
 * value of a matrix or vector equal to a constant, 32-bit integer.
 */
int mtxfiledata_set_constant_integer_single(
    union mtxfiledata * data,
    enum mtxfileobject object,
    enum mtxfileformat format,
    enum mtxfilefield field,
    enum mtxprecision precision,
    int64_t size,
    int64_t offset,
    int32_t a)
{
    if (format == mtxfile_array) {
        if (field == mtxfile_real) {
            if (precision == mtx_single) {
                for (int64_t k = 0; k < size; k++)
                    data->array_real_single[offset+k] = a;
            } else if (precision == mtx_double) {
                for (int64_t k = 0; k < size; k++)
                    data->array_real_double[offset+k] = a;
            } else { return MTX_ERR_INVALID_PRECISION; }
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
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (field == mtxfile_integer) {
            if (precision == mtx_single) {
                for (int64_t k = 0; k < size; k++)
                    data->array_integer_single[offset+k] = a;
            } else if (precision == mtx_double) {
                for (int64_t k = 0; k < size; k++)
                    data->array_integer_double[offset+k] = a;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else { return MTX_ERR_INVALID_MTX_FIELD; }
    } else if (format == mtxfile_coordinate) {
        if (object == mtxfile_matrix) {
            if (field == mtxfile_real) {
                if (precision == mtx_single) {
                    for (int64_t k = 0; k < size; k++)
                        data->matrix_coordinate_real_single[offset+k].a = a;
                } else if (precision == mtx_double) {
                    for (int64_t k = 0; k < size; k++)
                        data->matrix_coordinate_real_double[offset+k].a = a;
                } else { return MTX_ERR_INVALID_PRECISION; }
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
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_integer) {
                if (precision == mtx_single) {
                    for (int64_t k = 0; k < size; k++)
                        data->matrix_coordinate_integer_single[offset+k].a = a;
                } else if (precision == mtx_double) {
                    for (int64_t k = 0; k < size; k++)
                        data->matrix_coordinate_integer_double[offset+k].a = a;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_pattern) {
                return MTX_ERR_INCOMPATIBLE_FIELD;
            } else { return MTX_ERR_INVALID_MTX_FIELD; }
        } else if (object == mtxfile_vector) {
            if (field == mtxfile_real) {
                if (precision == mtx_single) {
                    for (int64_t k = 0; k < size; k++)
                        data->vector_coordinate_real_single[offset+k].a = a;
                } else if (precision == mtx_double) {
                    for (int64_t k = 0; k < size; k++)
                        data->vector_coordinate_real_double[offset+k].a = a;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_complex) {
                if (precision == mtx_single) {
                    for (int64_t k = 0; k < size; k++) {
                        data->vector_coordinate_complex_single[offset+k].a[0] = a;
                        data->vector_coordinate_complex_single[offset+k].a[1] = 0;
                    }
                } else if (precision == mtx_double) {
                    for (int64_t k = 0; k < size; k++) {
                        data->vector_coordinate_complex_double[offset+k].a[0] = a;
                        data->vector_coordinate_complex_double[offset+k].a[1] = 0;
                    }
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_integer) {
                if (precision == mtx_single) {
                    for (int64_t k = 0; k < size; k++)
                        data->vector_coordinate_integer_single[offset+k].a = a;
                } else if (precision == mtx_double) {
                    for (int64_t k = 0; k < size; k++)
                        data->vector_coordinate_integer_double[offset+k].a = a;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_pattern) {
                return MTX_ERR_INCOMPATIBLE_FIELD;
            } else { return MTX_ERR_INVALID_MTX_FIELD; }
        } else { return MTX_ERR_INVALID_MTX_OBJECT; }
    } else { return MTX_ERR_INVALID_MTX_FORMAT; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxfiledata_set_constant_integer_double()’ sets every (nonzero)
 * value of a matrix or vector equal to a constant, 64-bit integer.
 */
int mtxfiledata_set_constant_integer_double(
    union mtxfiledata * data,
    enum mtxfileobject object,
    enum mtxfileformat format,
    enum mtxfilefield field,
    enum mtxprecision precision,
    int64_t size,
    int64_t offset,
    int64_t a)
{
    if (format == mtxfile_array) {
        if (field == mtxfile_real) {
            if (precision == mtx_single) {
                for (int64_t k = 0; k < size; k++)
                    data->array_real_single[offset+k] = a;
            } else if (precision == mtx_double) {
                for (int64_t k = 0; k < size; k++)
                    data->array_real_double[offset+k] = a;
            } else { return MTX_ERR_INVALID_PRECISION; }
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
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (field == mtxfile_integer) {
            if (precision == mtx_single) {
                for (int64_t k = 0; k < size; k++)
                    data->array_integer_single[offset+k] = a;
            } else if (precision == mtx_double) {
                for (int64_t k = 0; k < size; k++)
                    data->array_integer_double[offset+k] = a;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else { return MTX_ERR_INVALID_MTX_FIELD; }
    } else if (format == mtxfile_coordinate) {
        if (object == mtxfile_matrix) {
            if (field == mtxfile_real) {
                if (precision == mtx_single) {
                    for (int64_t k = 0; k < size; k++)
                        data->matrix_coordinate_real_single[offset+k].a = a;
                } else if (precision == mtx_double) {
                    for (int64_t k = 0; k < size; k++)
                        data->matrix_coordinate_real_double[offset+k].a = a;
                } else { return MTX_ERR_INVALID_PRECISION; }
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
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_integer) {
                if (precision == mtx_single) {
                    for (int64_t k = 0; k < size; k++)
                        data->matrix_coordinate_integer_single[offset+k].a = a;
                } else if (precision == mtx_double) {
                    for (int64_t k = 0; k < size; k++)
                        data->matrix_coordinate_integer_double[offset+k].a = a;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_pattern) {
                return MTX_ERR_INCOMPATIBLE_FIELD;
            } else { return MTX_ERR_INVALID_MTX_FIELD; }
        } else if (object == mtxfile_vector) {
            if (field == mtxfile_real) {
                if (precision == mtx_single) {
                    for (int64_t k = 0; k < size; k++)
                        data->vector_coordinate_real_single[offset+k].a = a;
                } else if (precision == mtx_double) {
                    for (int64_t k = 0; k < size; k++)
                        data->vector_coordinate_real_double[offset+k].a = a;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_complex) {
                if (precision == mtx_single) {
                    for (int64_t k = 0; k < size; k++) {
                        data->vector_coordinate_complex_single[offset+k].a[0] = a;
                        data->vector_coordinate_complex_single[offset+k].a[1] = 0;
                    }
                } else if (precision == mtx_double) {
                    for (int64_t k = 0; k < size; k++) {
                        data->vector_coordinate_complex_double[offset+k].a[0] = a;
                        data->vector_coordinate_complex_double[offset+k].a[1] = 0;
                    }
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_integer) {
                if (precision == mtx_single) {
                    for (int64_t k = 0; k < size; k++)
                        data->vector_coordinate_integer_single[offset+k].a = a;
                } else if (precision == mtx_double) {
                    for (int64_t k = 0; k < size; k++)
                        data->vector_coordinate_integer_double[offset+k].a = a;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_pattern) {
                return MTX_ERR_INCOMPATIBLE_FIELD;
            } else { return MTX_ERR_INVALID_MTX_FIELD; }
        } else { return MTX_ERR_INVALID_MTX_OBJECT; }
    } else { return MTX_ERR_INVALID_MTX_FORMAT; }
    return MTX_SUCCESS;
}

/*
 * I/O functions
 */

static int mtxfiledata_parse(
    union mtxfiledata * data,
    int64_t * bytes_read,
    char ** endptr,
    const char * s,
    enum mtxfileobject object,
    enum mtxfileformat format,
    enum mtxfilefield field,
    enum mtxprecision precision,
    int64_t num_rows,
    int64_t num_columns,
    int64_t i)
{
    if (format == mtxfile_array) {
        if (field == mtxfile_real) {
            if (precision == mtx_single) {
                return mtxfiledata_parse_array_real_single(
                    &data->array_real_single[i],
                    bytes_read, endptr, s);
            } else if (precision == mtx_double) {
                return mtxfiledata_parse_array_real_double(
                    &data->array_real_double[i],
                    bytes_read, endptr, s);
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (field == mtxfile_complex) {
            if (precision == mtx_single) {
                return mtxfiledata_parse_array_complex_single(
                    &data->array_complex_single[i],
                    bytes_read, endptr, s);
            } else if (precision == mtx_double) {
                return mtxfiledata_parse_array_complex_double(
                    &data->array_complex_double[i],
                    bytes_read, endptr, s);
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (field == mtxfile_integer) {
            if (precision == mtx_single) {
                return mtxfiledata_parse_array_integer_single(
                    &data->array_integer_single[i],
                    bytes_read, endptr, s);
            } else if (precision == mtx_double) {
                return mtxfiledata_parse_array_integer_double(
                    &data->array_integer_double[i],
                    bytes_read, endptr, s);
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else { return MTX_ERR_INVALID_MTX_FIELD; }
    } else if (format == mtxfile_coordinate) {
        if (object == mtxfile_matrix) {
            if (field == mtxfile_real) {
                if (precision == mtx_single) {
                    return mtxfiledata_parse_matrix_coordinate_real_single(
                        &data->matrix_coordinate_real_single[i],
                        bytes_read, endptr, s, num_rows, num_columns);
                } else if (precision == mtx_double) {
                    return mtxfiledata_parse_matrix_coordinate_real_double(
                        &data->matrix_coordinate_real_double[i],
                        bytes_read, endptr, s, num_rows, num_columns);
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_complex) {
                if (precision == mtx_single) {
                    return mtxfiledata_parse_matrix_coordinate_complex_single(
                        &data->matrix_coordinate_complex_single[i],
                        bytes_read, endptr, s, num_rows, num_columns);
                } else if (precision == mtx_double) {
                    return mtxfiledata_parse_matrix_coordinate_complex_double(
                        &data->matrix_coordinate_complex_double[i],
                        bytes_read, endptr, s, num_rows, num_columns);
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_integer) {
                if (precision == mtx_single) {
                    return mtxfiledata_parse_matrix_coordinate_integer_single(
                        &data->matrix_coordinate_integer_single[i],
                        bytes_read, endptr, s, num_rows, num_columns);
                } else if (precision == mtx_double) {
                    return mtxfiledata_parse_matrix_coordinate_integer_double(
                        &data->matrix_coordinate_integer_double[i],
                        bytes_read, endptr, s, num_rows, num_columns);
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_pattern) {
                return mtxfiledata_parse_matrix_coordinate_pattern(
                    &data->matrix_coordinate_pattern[i],
                    bytes_read, endptr, s, num_rows, num_columns);
            } else { return MTX_ERR_INVALID_MTX_FIELD; }
        } else if (object == mtxfile_vector) {
            if (field == mtxfile_real) {
                if (precision == mtx_single) {
                    return mtxfiledata_parse_vector_coordinate_real_single(
                        &data->vector_coordinate_real_single[i],
                        bytes_read, endptr, s, num_rows);
                } else if (precision == mtx_double) {
                    return mtxfiledata_parse_vector_coordinate_real_double(
                        &data->vector_coordinate_real_double[i],
                        bytes_read, endptr, s, num_rows);
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_complex) {
                if (precision == mtx_single) {
                    return mtxfiledata_parse_vector_coordinate_complex_single(
                        &data->vector_coordinate_complex_single[i],
                        bytes_read, endptr, s, num_rows);
                } else if (precision == mtx_double) {
                    return mtxfiledata_parse_vector_coordinate_complex_double(
                        &data->vector_coordinate_complex_double[i],
                        bytes_read, endptr, s, num_rows);
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_integer) {
                if (precision == mtx_single) {
                    return mtxfiledata_parse_vector_coordinate_integer_single(
                        &data->vector_coordinate_integer_single[i],
                        bytes_read, endptr, s, num_rows);
                } else if (precision == mtx_double) {
                    return mtxfiledata_parse_vector_coordinate_integer_double(
                        &data->vector_coordinate_integer_double[i],
                        bytes_read, endptr, s, num_rows);
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_pattern) {
                return mtxfiledata_parse_vector_coordinate_pattern(
                    &data->vector_coordinate_pattern[i],
                    bytes_read, endptr, s, num_rows);
            } else { return MTX_ERR_INVALID_MTX_FIELD; }
        } else { return MTX_ERR_INVALID_MTX_OBJECT; }
    } else { return MTX_ERR_INVALID_MTX_FORMAT; }
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
 * ‘mtxfiledata_fread()‘ reads Matrix Market data lines from a
 * stream.
 *
 * Storage for the corresponding array of the ‘data’ union, according
 * to the given ‘object’, ‘format’, ‘field’ and ‘precision’ variables,
 * must already be allocated with enough storage to hold at least
 * ‘offset+size’ elements.
 *
 * At most ‘size’ lines are read from the stream and written to the
 * appropriate array of the ‘data’ union, starting ‘offset’ elements
 * from the beginning of the array.
 *
 * If an error code is returned, then ‘lines_read’ and ‘bytes_read’
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the Matrix Market file.
 *
 * During parsing, the locale is temporarily changed to "C" to ensure
 * that locale-specific settings, such as the type of decimal point,
 * do not affect parsing.
 */
int mtxfiledata_fread(
    union mtxfiledata * data,
    FILE * f,
    int64_t * lines_read,
    int64_t * bytes_read,
    size_t line_max,
    char * linebuf,
    enum mtxfileobject object,
    enum mtxfileformat format,
    enum mtxfilefield field,
    enum mtxprecision precision,
    int64_t num_rows,
    int64_t num_columns,
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

        char * endptr;
        err = mtxfiledata_parse(
            data, bytes_read, &endptr, linebuf,
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
        if (i < size-1 && *endptr != '\n') return MTX_ERR_INVALID_MTX_DATA;
        if (*endptr == '\n' && bytes_read) (*bytes_read)++;
        if (lines_read) (*lines_read)++;
    }

    setlocale(LC_ALL, locale);
    free(locale);
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
 * ‘mtxfiledata_gzread()’ reads Matrix Market data lines from a
 * gzip-compressed stream.
 *
 * Storage for the corresponding array of the ‘data’ union, according
 * to the given ‘object’, ‘format’, ‘field’ and ‘precision’ variables,
 * must already be allocated with enough storage to hold at least
 * ‘offset+size’ elements.
 *
 * At most ‘size’ lines are read from the stream and written to the
 * appropriate array of the ‘data’ union, starting ‘offset’ elements
 * from the beginning of the array.
 *
 * If an error code is returned, then ‘lines_read’ and ‘bytes_read’
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the Matrix Market file.
 *
 * During parsing, the locale is temporarily changed to "C" to ensure
 * that locale-specific settings, such as the type of decimal point,
 * do not affect parsing.
 */
int mtxfiledata_gzread(
    union mtxfiledata * data,
    gzFile f,
    int64_t * lines_read,
    int64_t * bytes_read,
    size_t line_max,
    char * linebuf,
    enum mtxfileobject object,
    enum mtxfileformat format,
    enum mtxfilefield field,
    enum mtxprecision precision,
    int64_t num_rows,
    int64_t num_columns,
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

        char * endptr;
        err = mtxfiledata_parse(
            data, bytes_read, &endptr, linebuf,
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
        if (i < size-1 && *endptr != '\n') return MTX_ERR_INVALID_MTX_DATA;
        if (*endptr == '\n' && bytes_read) (*bytes_read)++;
        if (lines_read) (*lines_read)++;
    }

    setlocale(LC_ALL, locale);
    free(locale);
    if (free_linebuf)
        free(linebuf);
    return MTX_SUCCESS;
}
#endif

/**
 * ‘validate_format_string()’ parses and validates a format string to
 * be used for outputting numerical values of a Matrix Market file.
 */
static int validate_format_string(
    const char * format_str,
    enum mtxfilefield field)
{
    struct fmtspec format;
    const char * endptr;
    int err = parse_fmtspec(format_str, &format, &endptr);
    if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    } else if (*endptr != '\0') {
        return MTX_ERR_INVALID_FORMAT_SPECIFIER;
    }

    if (format.width == fmtspec_width_star ||
        format.precision == fmtspec_precision_star ||
        format.length != fmtspec_length_none ||
        ((field == mtxfile_real ||
          field == mtxfile_complex) &&
         (format.specifier != fmtspec_e &&
          format.specifier != fmtspec_E &&
          format.specifier != fmtspec_f &&
          format.specifier != fmtspec_F &&
          format.specifier != fmtspec_g &&
          format.specifier != fmtspec_G)) ||
        (field == mtxfile_integer &&
         (format.specifier != fmtspec_d)))
    {
        return MTX_ERR_INVALID_FORMAT_SPECIFIER;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxfiledata_fwrite()’ writes data lines of a Matrix Market file
 * to a stream.
 *
 * If ‘fmt’ is ‘NULL’, then the format specifier ‘%g’ is used to print
 * floating point numbers with enough digits to ensure correct
 * round-trip conversion from decimal text and back.  Otherwise, the
 * given format string is used to print numerical values.
 *
 * The format string follows the conventions of ‘printf’. If the field
 * is ‘real’, ‘double’ or ‘complex’, then the format specifiers '%e',
 * '%E', '%f', '%F', '%g' or '%G' may be used. If the field is
 * ‘integer’, then the format specifier must be '%d'. The format
 * string is ignored if the field is ‘pattern’. Field width and
 * precision may be specified (e.g., "%3.1f"), but variable field
 * width and precision (e.g., "%*.*f"), as well as length modifiers
 * (e.g., "%Lf") are not allowed.
 *
 * If it is not ‘NULL’, then the number of bytes written to the stream
 * is returned in ‘bytes_written’.
 *
 * The locale is temporarily changed to "C" to ensure that
 * locale-specific settings, such as the type of decimal point, do not
 * affect output.
 */
int mtxfiledata_fwrite(
    const union mtxfiledata * data,
    enum mtxfileobject object,
    enum mtxfileformat format,
    enum mtxfilefield field,
    enum mtxprecision precision,
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
 * ‘mtxfiledata_gzwrite()’ writes data lines of a Matrix Market file
 * to a gzip-compressed stream.
 *
 * If ‘fmt’ is ‘NULL’, then the format specifier ‘%g’ is used to print
 * floating point numbers with enough digits to ensure correct
 * round-trip conversion from decimal text and back.  Otherwise, the
 * given format string is used to print numerical values.
 *
 * The format string follows the conventions of ‘printf’. If the field
 * is ‘real’, ‘double’ or ‘complex’, then the format specifiers '%e',
 * '%E', '%f', '%F', '%g' or '%G' may be used. If the field is
 * ‘integer’, then the format specifier must be '%d'. The format
 * string is ignored if the field is ‘pattern’. Field width and
 * precision may be specified (e.g., "%3.1f"), but variable field
 * width and precision (e.g., "%*.*f"), as well as length modifiers
 * (e.g., "%Lf") are not allowed.
 *
 * If it is not ‘NULL’, then the number of bytes written to the stream
 * is returned in ‘bytes_written’.
 *
 * The locale is temporarily changed to "C" to ensure that
 * locale-specific settings, such as the type of decimal point, do not
 * affect output.
 */
int mtxfiledata_gzwrite(
    const union mtxfiledata * data,
    enum mtxfileobject object,
    enum mtxfileformat format,
    enum mtxfilefield field,
    enum mtxprecision precision,
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
 * ‘mtxfiledata_transpose()’ tranposes the data lines of a Matrix
 * Market file.
 */
int mtxfiledata_transpose(
    union mtxfiledata * data,
    enum mtxfileobject object,
    enum mtxfileformat format,
    enum mtxfilefield field,
    enum mtxprecision precision,
    int64_t num_rows,
    int64_t num_columns,
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
                } else { return MTX_ERR_INVALID_PRECISION; }
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
                } else { return MTX_ERR_INVALID_PRECISION; }
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
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_pattern) {
                for (int64_t k = 0; k < size; k++) {
                    int i = data->matrix_coordinate_pattern[k].i;
                    int j = data->matrix_coordinate_pattern[k].j;
                    data->matrix_coordinate_pattern[k].i = j;
                    data->matrix_coordinate_pattern[k].j = i;
                }
            } else { return MTX_ERR_INVALID_MTX_FIELD; }

        } else if (format == mtxfile_array) {
            union mtxfiledata copy;
            err = mtxfiledata_alloc(&copy, object, format, field, precision, size);
            if (err)
                return err;
            err = mtxfiledata_copy(
                &copy, data, object, format, field, precision, size, 0, 0);
            if (err) {
                mtxfiledata_free(&copy, object, format, field, precision);
                return err;
            }

            int64_t k, l;
            if (field == mtxfile_real) {
                if (precision == mtx_single) {
                    for (int64_t i = 0; i < num_rows; i++) {
                        for (int64_t j = 0; j < num_columns; j++) {
                            k = i * num_columns + j;
                            l = j * num_rows + i;
                            data->array_real_single[l] = copy.array_real_single[k];
                        }
                    }
                } else if (precision == mtx_double) {
                    for (int64_t i = 0; i < num_rows; i++) {
                        for (int64_t j = 0; j < num_columns; j++) {
                            k = i * num_columns + j;
                            l = j * num_rows + i;
                            data->array_real_double[l] = copy.array_real_double[k];
                        }
                    }
                } else {
                    mtxfiledata_free(&copy, object, format, field, precision);
                    return MTX_ERR_INVALID_PRECISION;
                }
            } else if (field == mtxfile_complex) {
                if (precision == mtx_single) {
                    for (int64_t i = 0; i < num_rows; i++) {
                        for (int64_t j = 0; j < num_columns; j++) {
                            k = i * num_columns + j;
                            l = j * num_rows + i;
                            data->array_complex_single[l][0] =
                                copy.array_complex_single[k][0];
                            data->array_complex_single[l][1] =
                                copy.array_complex_single[k][1];
                        }
                    }
                } else if (precision == mtx_double) {
                    for (int64_t i = 0; i < num_rows; i++) {
                        for (int64_t j = 0; j < num_columns; j++) {
                            k = i * num_columns + j;
                            l = j * num_rows + i;
                            data->array_complex_double[l][0] =
                                copy.array_complex_double[k][0];
                            data->array_complex_double[l][1] =
                                copy.array_complex_double[k][1];
                        }
                    }
                } else {
                    mtxfiledata_free(&copy, object, format, field, precision);
                    return MTX_ERR_INVALID_PRECISION;
                }
            } else if (field == mtxfile_integer) {
                if (precision == mtx_single) {
                    for (int64_t i = 0; i < num_rows; i++) {
                        for (int64_t j = 0; j < num_columns; j++) {
                            k = i * num_columns + j;
                            l = j * num_rows + i;
                            data->array_integer_single[l] =
                                copy.array_integer_single[k];
                            data->array_integer_single[l] =
                                copy.array_integer_single[k];
                        }
                    }
                } else if (precision == mtx_double) {
                    for (int64_t i = 0; i < num_rows; i++) {
                        for (int64_t j = 0; j < num_columns; j++) {
                            k = i * num_columns + j;
                            l = j * num_rows + i;
                            data->array_integer_double[l] =
                                copy.array_integer_double[k];
                        }
                    }
                } else {
                    mtxfiledata_free(&copy, object, format, field, precision);
                    return MTX_ERR_INVALID_PRECISION;
                }
            } else {
                mtxfiledata_free(&copy, object, format, field, precision);
                return MTX_ERR_INVALID_MTX_FIELD;
            }

            mtxfiledata_free(&copy, object, format, field, precision);
        } else { return MTX_ERR_INVALID_MTX_FORMAT; }

    } else if (object == mtxfile_vector) {
        return MTX_SUCCESS;
    } else { return MTX_ERR_INVALID_MTX_OBJECT; }
    return MTX_SUCCESS;
}

/*
 * Sorting
 */

/**
 * ‘mtxfiledata_permute()’ permutes the order of data lines in a
 * Matrix Market file according to a given permutation.
 *
 * The array ‘perm’ should be a permutation of the integers
 * ‘1,2,...,N’, where ‘N’ is the number of data lines in the matrix or
 * vector.
 */
int mtxfiledata_permute(
    union mtxfiledata * data,
    enum mtxfileobject object,
    enum mtxfileformat format,
    enum mtxfilefield field,
    enum mtxprecision precision,
    int64_t num_rows,
    int64_t num_columns,
    int64_t size,
    int64_t * perm)
{
    int err;
    for (int64_t k = 0; k < size; k++) {
        if (perm[k] <= 0 || perm[k] > size)
            return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    }

    /* 1. Copy the original, unsorted data. */
    union mtxfiledata srcdata;
    err = mtxfiledata_alloc(
        &srcdata, object, format, field, precision, size);
    if (err)
        return err;
    err = mtxfiledata_copy(
        &srcdata, data, object, format, field, precision,
        size, 0, 0);
    if (err) {
        mtxfiledata_free(&srcdata, object, format, field, precision);
        return err;
    }

    /* 2. Permute the nonzeros. */
    if (format == mtxfile_array) {
        if (field == mtxfile_real) {
            if (precision == mtx_single) {
                float * dst = data->array_real_single;
                const float * src = srcdata.array_real_single;
                for (int64_t k = 0; k < size; k++)
                    dst[perm[k]-1] = src[k];
            } else if (precision == mtx_double) {
                double * dst = data->array_real_double;
                const double * src = srcdata.array_real_double;
                for (int64_t k = 0; k < size; k++)
                    dst[perm[k]-1] = src[k];
            } else {
                mtxfiledata_free(&srcdata, object, format, field, precision);
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_complex) {
            if (precision == mtx_single) {
                float (* dst)[2] = data->array_complex_single;
                const float (* src)[2] = srcdata.array_complex_single;
                for (int64_t k = 0; k < size; k++) {
                    dst[perm[k]-1][0] = src[k][0];
                    dst[perm[k]-1][1] = src[k][1];
                }
            } else if (precision == mtx_double) {
                double (* dst)[2] = data->array_complex_double;
                const double (* src)[2] = srcdata.array_complex_double;
                for (int64_t k = 0; k < size; k++) {
                    dst[perm[k]-1][0] = src[k][0];
                    dst[perm[k]-1][1] = src[k][1];
                }
            } else {
                mtxfiledata_free(&srcdata, object, format, field, precision);
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_integer) {
            if (precision == mtx_single) {
                int32_t * dst = data->array_integer_single;
                const int32_t * src = srcdata.array_integer_single;
                for (int64_t k = 0; k < size; k++)
                    dst[perm[k]-1] = src[k];
            } else if (precision == mtx_double) {
                int64_t * dst = data->array_integer_double;
                const int64_t * src = srcdata.array_integer_double;
                for (int64_t k = 0; k < size; k++)
                    dst[perm[k]-1] = src[k];
            } else {
                mtxfiledata_free(&srcdata, object, format, field, precision);
                return MTX_ERR_INVALID_PRECISION;
            }
        } else {
            mtxfiledata_free(&srcdata, object, format, field, precision);
            return MTX_ERR_INVALID_MTX_FIELD;
        }
    } else if (format == mtxfile_coordinate) {
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
                    mtxfiledata_free(&srcdata, object, format, field, precision);
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
                    mtxfiledata_free(&srcdata, object, format, field, precision);
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
                    mtxfiledata_free(&srcdata, object, format, field, precision);
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
                mtxfiledata_free(&srcdata, object, format, field, precision);
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
                    mtxfiledata_free(&srcdata, object, format, field, precision);
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
                    mtxfiledata_free(&srcdata, object, format, field, precision);
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
                    mtxfiledata_free(&srcdata, object, format, field, precision);
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
                mtxfiledata_free(&srcdata, object, format, field, precision);
                return MTX_ERR_INVALID_MTX_FIELD;
            }
        } else {
            mtxfiledata_free(&srcdata, object, format, field, precision);
            return MTX_ERR_INVALID_MTX_OBJECT;
        }
    } else {
        mtxfiledata_free(&srcdata, object, format, field, precision);
        return MTX_ERR_INVALID_MTX_FORMAT;
    }
    mtxfiledata_free(&srcdata, object, format, field, precision);
    return MTX_SUCCESS;
}

/**
 * ‘mtxfiledata_sortkey_row_major()’ provides an array of keys that
 * can be used to sort the data lines of the given Matrix Market file
 * in row major order.
 *
 * The array ‘keys’ must contain enough storage for an array of ‘size’
 * values of type ‘int64_t’.  If successful, the ‘k’-th value of
 * ‘keys’ is the sorting key for the ‘k’-th data line.
 *
 * If ‘format’ is ‘mtxfile_array’, then a non-negative ‘offset’ value
 * can be used to obtain sorting keys for matrix or vector entries
 * starting from the specified offset, instead of beginning with the
 * first entry of the matrix or vector.
 */
int mtxfiledata_sortkey_row_major(
    union mtxfiledata * data,
    enum mtxfileobject object,
    enum mtxfileformat format,
    enum mtxfilefield field,
    enum mtxprecision precision,
    int64_t num_rows,
    int64_t num_columns,
    int64_t offset,
    int64_t size,
    uint64_t * keys)
{
    int err;

    if (object == mtxfile_matrix) {
        if (format == mtxfile_array) {
            if ((offset + size) > num_rows * num_columns)
                return MTX_ERR_INDEX_OUT_OF_BOUNDS;
            for (int64_t k = offset, l = 0; l < size; k++, l++) {
                int64_t i = k / num_columns;
                int64_t j = k % num_columns;
                keys[l] = ((uint64_t) i << 32) | j;
            }
        } else if (format == mtxfile_coordinate) {
            if (field == mtxfile_real) {
                if (precision == mtx_single) {
                    const struct mtxfile_matrix_coordinate_real_single * src =
                        data->matrix_coordinate_real_single;
                    for (int64_t k = 0; k < size; k++)
                        keys[k] = ((uint64_t) src[k].i << 32) | src[k].j;
                } else if (precision == mtx_double) {
                    const struct mtxfile_matrix_coordinate_real_double * src =
                        data->matrix_coordinate_real_double;
                    for (int64_t k = 0; k < size; k++)
                        keys[k] = ((uint64_t) src[k].i << 32) | src[k].j;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_complex) {
                if (precision == mtx_single) {
                    const struct mtxfile_matrix_coordinate_complex_single * src =
                        data->matrix_coordinate_complex_single;
                    for (int64_t k = 0; k < size; k++)
                        keys[k] = ((uint64_t) src[k].i << 32) | src[k].j;
                } else if (precision == mtx_double) {
                    const struct mtxfile_matrix_coordinate_complex_double * src =
                        data->matrix_coordinate_complex_double;
                    for (int64_t k = 0; k < size; k++)
                        keys[k] = ((uint64_t) src[k].i << 32) | src[k].j;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_integer) {
                if (precision == mtx_single) {
                    const struct mtxfile_matrix_coordinate_integer_single * src =
                        data->matrix_coordinate_integer_single;
                    for (int64_t k = 0; k < size; k++)
                        keys[k] = ((uint64_t) src[k].i << 32) | src[k].j;
                } else if (precision == mtx_double) {
                    const struct mtxfile_matrix_coordinate_integer_double * src =
                        data->matrix_coordinate_integer_double;
                    for (int64_t k = 0; k < size; k++)
                        keys[k] = ((uint64_t) src[k].i << 32) | src[k].j;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_pattern) {
                const struct mtxfile_matrix_coordinate_pattern * src =
                    data->matrix_coordinate_pattern;
                for (int64_t k = 0; k < size; k++)
                    keys[k] = ((uint64_t) src[k].i << 32) | src[k].j;
            } else { return MTX_ERR_INVALID_MTX_FIELD; }
        } else { return MTX_ERR_INVALID_MTX_FORMAT; }
    } else if (object == mtxfile_vector) {
        if (format == mtxfile_array) {
            if (num_columns != -1)
                return MTX_ERR_INVALID_MTX_SIZE;
            if (offset + size > num_rows)
                return MTX_ERR_INDEX_OUT_OF_BOUNDS;
            for (int64_t k = offset, l = 0; l < size; k++, l++)
                keys[l] = k;
        } else if (format == mtxfile_coordinate) {
            if (field == mtxfile_real) {
                if (precision == mtx_single) {
                    const struct mtxfile_vector_coordinate_real_single * src =
                        data->vector_coordinate_real_single;
                    for (int64_t k = 0; k < size; k++)
                        keys[k] = src[k].i;
                } else if (precision == mtx_double) {
                    const struct mtxfile_vector_coordinate_real_double * src =
                        data->vector_coordinate_real_double;
                    for (int64_t k = 0; k < size; k++)
                        keys[k] = src[k].i;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_complex) {
                if (precision == mtx_single) {
                    const struct mtxfile_vector_coordinate_complex_single * src =
                        data->vector_coordinate_complex_single;
                    for (int64_t k = 0; k < size; k++)
                        keys[k] = src[k].i;
                } else if (precision == mtx_double) {
                    const struct mtxfile_vector_coordinate_complex_double * src =
                        data->vector_coordinate_complex_double;
                    for (int64_t k = 0; k < size; k++)
                        keys[k] = src[k].i;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_integer) {
                if (precision == mtx_single) {
                    const struct mtxfile_vector_coordinate_integer_single * src =
                        data->vector_coordinate_integer_single;
                    for (int64_t k = 0; k < size; k++)
                        keys[k] = src[k].i;
                } else if (precision == mtx_double) {
                    const struct mtxfile_vector_coordinate_integer_double * src =
                        data->vector_coordinate_integer_double;
                    for (int64_t k = 0; k < size; k++)
                        keys[k] = src[k].i;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_pattern) {
                const struct mtxfile_vector_coordinate_pattern * src =
                    data->vector_coordinate_pattern;
                for (int64_t k = 0; k < size; k++)
                    keys[k] = src[k].i;
            } else { return MTX_ERR_INVALID_MTX_FIELD; }
        } else { return MTX_ERR_INVALID_MTX_FORMAT; }
    } else { return MTX_ERR_INVALID_MTX_OBJECT; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxfiledata_sortkey_column_major()’ provides an array of keys
 * that can be used to sort the data lines of the given Matrix Market
 * file in column major order.
 *
 * The array ‘keys’ must contain enough storage for an array of ‘size’
 * values of type ‘int64_t’.  If successful, the ‘k’-th value of
 * ‘keys’ is the sorting key for the ‘k’-th data line.
 *
 * If ‘format’ is ‘mtxfile_array’, then a non-negative ‘offset’ value
 * can be used to obtain sorting keys for matrix or vector entries
 * starting from the specified offset, instead of beginning with the
 * first entry of the matrix or vector.
 */
int mtxfiledata_sortkey_column_major(
    union mtxfiledata * data,
    enum mtxfileobject object,
    enum mtxfileformat format,
    enum mtxfilefield field,
    enum mtxprecision precision,
    int64_t num_rows,
    int64_t num_columns,
    int64_t offset,
    int64_t size,
    uint64_t * keys)
{
    int err;

    if (object == mtxfile_matrix) {
        if (format == mtxfile_array) {
            if ((offset + size) > num_rows * num_columns)
                return MTX_ERR_INDEX_OUT_OF_BOUNDS;
            for (int64_t k = offset, l = 0; l < size; k++, l++) {
                int64_t i = k / num_columns;
                int64_t j = k % num_columns;
                keys[l] = ((uint64_t) j << 32) | i;
            }
        } else if (format == mtxfile_coordinate) {
            if (field == mtxfile_real) {
                if (precision == mtx_single) {
                    const struct mtxfile_matrix_coordinate_real_single * src =
                        data->matrix_coordinate_real_single;
                    for (int64_t k = 0; k < size; k++)
                        keys[k] = ((uint64_t) src[k].j << 32) | src[k].i;
                } else if (precision == mtx_double) {
                    const struct mtxfile_matrix_coordinate_real_double * src =
                        data->matrix_coordinate_real_double;
                    for (int64_t k = 0; k < size; k++)
                        keys[k] = ((uint64_t) src[k].j << 32) | src[k].i;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_complex) {
                if (precision == mtx_single) {
                    const struct mtxfile_matrix_coordinate_complex_single * src =
                        data->matrix_coordinate_complex_single;
                    for (int64_t k = 0; k < size; k++)
                        keys[k] = ((uint64_t) src[k].j << 32) | src[k].i;
                } else if (precision == mtx_double) {
                    const struct mtxfile_matrix_coordinate_complex_double * src =
                        data->matrix_coordinate_complex_double;
                    for (int64_t k = 0; k < size; k++)
                        keys[k] = ((uint64_t) src[k].j << 32) | src[k].i;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_integer) {
                if (precision == mtx_single) {
                    const struct mtxfile_matrix_coordinate_integer_single * src =
                        data->matrix_coordinate_integer_single;
                    for (int64_t k = 0; k < size; k++)
                        keys[k] = ((uint64_t) src[k].j << 32) | src[k].i;
                } else if (precision == mtx_double) {
                    const struct mtxfile_matrix_coordinate_integer_double * src =
                        data->matrix_coordinate_integer_double;
                    for (int64_t k = 0; k < size; k++)
                        keys[k] = ((uint64_t) src[k].j << 32) | src[k].i;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_pattern) {
                const struct mtxfile_matrix_coordinate_pattern * src =
                    data->matrix_coordinate_pattern;
                for (int64_t k = 0; k < size; k++)
                    keys[k] = ((uint64_t) src[k].j << 32) | src[k].i;
            } else { return MTX_ERR_INVALID_MTX_FIELD; }
        } else { return MTX_ERR_INVALID_MTX_FORMAT; }
    } else if (object == mtxfile_vector) {
        /* For vectors, column major is the same as row major. */
        return mtxfiledata_sortkey_row_major(
            data, object, format, field, precision,
            num_rows, num_columns, offset, size, keys);
    } else { return MTX_ERR_INVALID_MTX_OBJECT; }
    return MTX_SUCCESS;
}

static inline uint64_t xy_to_morton(uint32_t x, uint32_t y)
{
    return _pdep_u32(x, 0x55555555) | _pdep_u32(y, 0xaaaaaaaa);
}

/**
 * ‘mtxfiledata_sortkey_morton()’ provides an array of keys that can
 * be used to sort the data lines of the given Matrix Market file in
 * Morton order (Z-order).
 *
 * The array ‘keys’ must contain enough storage for an array of ‘size’
 * values of type ‘int64_t’.  If successful, the ‘k’-th value of
 * ‘keys’ is the sorting key for the ‘k’-th data line.
 *
 * If ‘format’ is ‘mtxfile_array’, then a non-negative ‘offset’ value
 * can be used to obtain sorting keys for matrix or vector entries
 * starting from the specified offset, instead of beginning with the
 * first entry of the matrix or vector.
 */
int mtxfiledata_sortkey_morton(
    union mtxfiledata * data,
    enum mtxfileobject object,
    enum mtxfileformat format,
    enum mtxfilefield field,
    enum mtxprecision precision,
    int64_t num_rows,
    int64_t num_columns,
    int64_t offset,
    int64_t size,
    uint64_t * keys)
{
    int err;

    if (object == mtxfile_matrix) {
        if (format == mtxfile_array) {
            if ((offset + size) > num_rows * num_columns)
                return MTX_ERR_INDEX_OUT_OF_BOUNDS;
            for (int64_t k = offset, l = 0; l < size; k++, l++) {
                int64_t i = k / num_columns;
                int64_t j = k % num_columns;
                keys[l] = xy_to_morton(j, i);
            }
        } else if (format == mtxfile_coordinate) {
            if (field == mtxfile_real) {
                if (precision == mtx_single) {
                    const struct mtxfile_matrix_coordinate_real_single * src =
                        data->matrix_coordinate_real_single;
                    for (int64_t k = 0; k < size; k++)
                        keys[k] = xy_to_morton(src[k].j-1, src[k].i-1);
                } else if (precision == mtx_double) {
                    const struct mtxfile_matrix_coordinate_real_double * src =
                        data->matrix_coordinate_real_double;
                    for (int64_t k = 0; k < size; k++)
                        keys[k] = xy_to_morton(src[k].j-1, src[k].i-1);
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_complex) {
                if (precision == mtx_single) {
                    const struct mtxfile_matrix_coordinate_complex_single * src =
                        data->matrix_coordinate_complex_single;
                    for (int64_t k = 0; k < size; k++)
                        keys[k] = xy_to_morton(src[k].j-1, src[k].i-1);
                } else if (precision == mtx_double) {
                    const struct mtxfile_matrix_coordinate_complex_double * src =
                        data->matrix_coordinate_complex_double;
                    for (int64_t k = 0; k < size; k++)
                        keys[k] = xy_to_morton(src[k].j-1, src[k].i-1);
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_integer) {
                if (precision == mtx_single) {
                    const struct mtxfile_matrix_coordinate_integer_single * src =
                        data->matrix_coordinate_integer_single;
                    for (int64_t k = 0; k < size; k++)
                        keys[k] = xy_to_morton(src[k].j-1, src[k].i-1);
                } else if (precision == mtx_double) {
                    const struct mtxfile_matrix_coordinate_integer_double * src =
                        data->matrix_coordinate_integer_double;
                    for (int64_t k = 0; k < size; k++)
                        keys[k] = xy_to_morton(src[k].j-1, src[k].i-1);
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_pattern) {
                const struct mtxfile_matrix_coordinate_pattern * src =
                    data->matrix_coordinate_pattern;
                for (int64_t k = 0; k < size; k++)
                    keys[k] = xy_to_morton(src[k].j-1, src[k].i-1);
            } else { return MTX_ERR_INVALID_MTX_FIELD; }
        } else { return MTX_ERR_INVALID_MTX_FORMAT; }
    } else if (object == mtxfile_vector) {
        /* For vectors, Morton order is the same as row major. */
        return mtxfiledata_sortkey_row_major(
            data, object, format, field, precision,
            num_rows, num_columns, offset, size, keys);
    } else { return MTX_ERR_INVALID_MTX_OBJECT; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxfiledata_sort_keys()’ sorts data lines of a Matrix Market file
 * by the given keys.
 */
int mtxfiledata_sort_keys(
    union mtxfiledata * data,
    enum mtxfileobject object,
    enum mtxfileformat format,
    enum mtxfilefield field,
    enum mtxprecision precision,
    int64_t num_rows,
    int64_t num_columns,
    int64_t size,
    uint64_t * keys,
    int64_t * sorting_permutation)
{
    int err;

    /* 1. Sort the keys and obtain a sorting permutation. */
    bool alloc_sorting_permutation = !sorting_permutation;
    if (alloc_sorting_permutation) {
        sorting_permutation = malloc(size * sizeof(int64_t));
        if (!sorting_permutation)
            return MTX_ERR_ERRNO;
    }
    err = radix_sort_uint64(size, keys, sorting_permutation);
    if (err) {
        if (alloc_sorting_permutation)
            free(sorting_permutation);
        return err;
    }

    /* Adjust from 0-based to 1-based indexing. */
    for (int64_t i = 0; i < size; i++)
        sorting_permutation[i]++;

    /* 2. Sort nonzeros according to the sorting permutation. */
    err = mtxfiledata_permute(
        data, object, format, field, precision,
        num_rows, num_columns, size, sorting_permutation);
    if (err) {
        if (alloc_sorting_permutation)
            free(sorting_permutation);
        return err;
    }
    if (alloc_sorting_permutation)
        free(sorting_permutation);
    return MTX_SUCCESS;
}

/**
 * ‘mtxfiledata_sort_int()’ sorts data lines of a Matrix Market file
 * by the given integer keys.
 */
int mtxfiledata_sort_int(
    union mtxfiledata * data,
    enum mtxfileobject object,
    enum mtxfileformat format,
    enum mtxfilefield field,
    enum mtxprecision precision,
    int64_t num_rows,
    int64_t num_columns,
    int64_t size,
    int * keys,
    int64_t * perm)
{
    /* 1. Sort the keys and obtain a sorting permutation. */
    bool alloc_perm = !perm;
    if (alloc_perm) {
        perm = malloc(size * sizeof(int64_t));
        if (!perm) return MTX_ERR_ERRNO;
    }
    int err = radix_sort_int(size, keys, perm);
    if (err) { if (alloc_perm) free(perm); return err; }

    /* 2. Sort nonzeros according to the sorting permutation. */
    for (int64_t i = 0; i < size; i++) perm[i]++;
    err = mtxfiledata_permute(
        data, object, format, field, precision,
        num_rows, num_columns, size, perm);
    if (err) { if (alloc_perm) free(perm); return err; }
    if (alloc_perm) free(perm);
    else { for (int64_t i = 0; i < size; i++) perm[i]--; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxfiledata_sort_row_major()’ sorts data lines of a Matrix Market
 * file in row major order.
 *
 * Matrices and vectors in ‘array’ format are already in row major
 * order, which means that nothing is done in this case. Otherwise,
 */
int mtxfiledata_sort_row_major(
    union mtxfiledata * data,
    enum mtxfileobject object,
    enum mtxfileformat format,
    enum mtxfilefield field,
    enum mtxprecision precision,
    int64_t num_rows,
    int64_t num_columns,
    int64_t size,
    int64_t * sorting_permutation)
{
    int err;
    if (format == mtxfile_array) {
        if (sorting_permutation) {
            for (int64_t k = 0; k < size; k++)
                sorting_permutation[k] = k+1;
        }
        return MTX_SUCCESS;
    } else if (format == mtxfile_coordinate) {
        uint64_t * keys = malloc(size * sizeof(uint64_t));
        if (!keys)
            return MTX_ERR_ERRNO;
        err = mtxfiledata_sortkey_row_major(
            data, object, format, field, precision,
            num_rows, num_columns, 0, size, keys);
        if (err) {
            free(keys);
            return err;
        }
        err = mtxfiledata_sort_keys(
            data, object, format, field, precision,
            num_rows, num_columns, size,
            keys, sorting_permutation);
        if (err) {
            free(keys);
            return err;
        }
        free(keys);
    } else { return MTX_ERR_INVALID_MTX_FORMAT; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxfiledata_sort_column_major()’ sorts data lines of a Matrix
 * Market file in column major order.
 *
 * Matrices and vectors in ‘array’ format are already in column major
 * order, which means that nothing is done in this case. Otherwise,
 */
int mtxfiledata_sort_column_major(
    union mtxfiledata * data,
    enum mtxfileobject object,
    enum mtxfileformat format,
    enum mtxfilefield field,
    enum mtxprecision precision,
    int64_t num_rows,
    int64_t num_columns,
    int64_t size,
    int64_t * sorting_permutation)
{
    int err;

    uint64_t * keys = malloc(size * sizeof(uint64_t));
    if (!keys)
        return MTX_ERR_ERRNO;

    err = mtxfiledata_sortkey_column_major(
        data, object, format, field, precision,
        num_rows, num_columns, 0, size, keys);
    if (err) {
        free(keys);
        return err;
    }

    err = mtxfiledata_sort_keys(
        data, object, format, field, precision,
        num_rows, num_columns, size,
        keys, sorting_permutation);
    if (err) {
        free(keys);
        return err;
    }
    free(keys);
    return MTX_SUCCESS;
}

/**
 * ‘mtxfiledata_sort_morton()’ sorts data lines of a Matrix Market
 * file in Morton order, also known as Z-order.
 *
 * This operation is only supported for matrices in coordinate format.
 */
int mtxfiledata_sort_morton(
    union mtxfiledata * data,
    enum mtxfileobject object,
    enum mtxfileformat format,
    enum mtxfilefield field,
    enum mtxprecision precision,
    int64_t num_rows,
    int64_t num_columns,
    int64_t size,
    int64_t * sorting_permutation)
{
    int err;

    /* 1. Allocate storage for and extract the sorting keys. */
    uint64_t * keys = malloc(size * sizeof(uint64_t));
    if (!keys)
        return MTX_ERR_ERRNO;

    err = mtxfiledata_sortkey_morton(
        data, object, format, field, precision,
        num_rows, num_columns, 0, size, keys);
    if (err) {
        free(keys);
        return err;
    }

    err = mtxfiledata_sort_keys(
        data, object, format, field, precision,
        num_rows, num_columns, size,
        keys, sorting_permutation);
    if (err) {
        free(keys);
        return err;
    }
    free(keys);
    return MTX_SUCCESS;
}

/**
 * ‘mtxfiledata_compact()’ compacts a Matrix Market file in coordinate
 * format by merging adjacent, duplicate data lines.
 *
 * For a matrix or vector in array format, this does nothing.
 *
 * The number of nonzero matrix or vector entries after compaction is
 * returned in ‘outsize’. This can be used to determine the number of
 * entries that were removed as a result of compacting. However, note
 * that the underlying storage for the Matrix Market data is not
 * changed or reallocated. This may result in large amounts of unused
 * memory, if a large number of entries were removed. If necessary, it
 * is possible to allocate new storage, copy the compacted data, and,
 * finally, free the old storage.
 *
 * If ‘perm’ is not ‘NULL’, then it must point to an array of length
 * ‘size’. The ‘i’th entry of ‘perm’ is used to store the index of the
 * corresponding data line in the compacted array that the ‘i’th data
 * line was moved to or merged with. Note that the indexing is
 * 1-based.
 */
int mtxfiledata_compact(
    union mtxfiledata * data,
    enum mtxfileobject object,
    enum mtxfileformat format,
    enum mtxfilefield field,
    enum mtxprecision precision,
    int64_t num_rows,
    int64_t num_columns,
    int64_t size,
    int64_t * perm,
    int64_t * outsize)
{
    if (object == mtxfile_matrix) {
        if (format == mtxfile_array) {
            *outsize = size;
            if (perm) {
                for (int64_t k = 0; k < size; k++)
                    perm[k] = k+1;
            }
            return MTX_SUCCESS;
        } else if (format == mtxfile_coordinate) {
            if (field == mtxfile_real) {
                if (precision == mtx_single) {
                    if (perm && size > 0) perm[0] = 0;
                    int64_t l = 0;
                    for (int64_t k = 1; k < size; k++) {
                        int ip = data->matrix_coordinate_real_single[l].i;
                        int jp = data->matrix_coordinate_real_single[l].j;
                        int i = data->matrix_coordinate_real_single[k].i;
                        int j = data->matrix_coordinate_real_single[k].j;
                        if (i == ip && j == jp) {
                            data->matrix_coordinate_real_single[l].a +=
                                data->matrix_coordinate_real_single[k].a;
                        } else {
                            l++;
                            data->matrix_coordinate_real_single[l].i = i;
                            data->matrix_coordinate_real_single[l].j = j;
                            data->matrix_coordinate_real_single[l].a =
                                data->matrix_coordinate_real_single[k].a;
                        }
                        if (perm) perm[k] = l+1;
                    }
                    *outsize = size > 0 ? l+1 : 0;
                } else if (precision == mtx_double) {
                    if (perm && size > 0) perm[0] = 0;
                    int64_t l = 0;
                    for (int64_t k = 1; k < size; k++) {
                        int ip = data->matrix_coordinate_real_double[l].i;
                        int jp = data->matrix_coordinate_real_double[l].j;
                        int i = data->matrix_coordinate_real_double[k].i;
                        int j = data->matrix_coordinate_real_double[k].j;
                        if (i == ip && j == jp) {
                            data->matrix_coordinate_real_double[l].a +=
                                data->matrix_coordinate_real_double[k].a;
                        } else {
                            l++;
                            data->matrix_coordinate_real_double[l].i = i;
                            data->matrix_coordinate_real_double[l].j = j;
                            data->matrix_coordinate_real_double[l].a =
                                data->matrix_coordinate_real_double[k].a;
                        }
                        if (perm) perm[k] = l+1;
                    }
                    *outsize = size > 0 ? l+1 : 0;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_complex) {
                if (precision == mtx_single) {
                    if (perm && size > 0) perm[0] = 0;
                    int64_t l = 0;
                    for (int64_t k = 1; k < size; k++) {
                        int ip = data->matrix_coordinate_complex_single[l].i;
                        int jp = data->matrix_coordinate_complex_single[l].j;
                        int i = data->matrix_coordinate_complex_single[k].i;
                        int j = data->matrix_coordinate_complex_single[k].j;
                        if (i == ip && j == jp) {
                            data->matrix_coordinate_complex_single[l].a[0] +=
                                data->matrix_coordinate_complex_single[k].a[0];
                            data->matrix_coordinate_complex_single[l].a[1] +=
                                data->matrix_coordinate_complex_single[k].a[1];
                        } else {
                            l++;
                            data->matrix_coordinate_complex_single[l].i = i;
                            data->matrix_coordinate_complex_single[l].j = j;
                            data->matrix_coordinate_complex_single[l].a[0] =
                                data->matrix_coordinate_complex_single[k].a[0];
                            data->matrix_coordinate_complex_single[l].a[1] =
                                data->matrix_coordinate_complex_single[k].a[1];
                        }
                        if (perm) perm[k] = l+1;
                    }
                    *outsize = size > 0 ? l+1 : 0;
                } else if (precision == mtx_double) {
                    if (perm && size > 0) perm[0] = 0;
                    int64_t l = 0;
                    for (int64_t k = 1; k < size; k++) {
                        int ip = data->matrix_coordinate_complex_double[l].i;
                        int jp = data->matrix_coordinate_complex_double[l].j;
                        int i = data->matrix_coordinate_complex_double[k].i;
                        int j = data->matrix_coordinate_complex_double[k].j;
                        if (i == ip && j == jp) {
                            data->matrix_coordinate_complex_double[l].a[0] +=
                                data->matrix_coordinate_complex_double[k].a[0];
                            data->matrix_coordinate_complex_double[l].a[1] +=
                                data->matrix_coordinate_complex_double[k].a[1];
                        } else {
                            l++;
                            data->matrix_coordinate_complex_double[l].i = i;
                            data->matrix_coordinate_complex_double[l].j = j;
                            data->matrix_coordinate_complex_double[l].a[0] =
                                data->matrix_coordinate_complex_double[k].a[0];
                            data->matrix_coordinate_complex_double[l].a[1] =
                                data->matrix_coordinate_complex_double[k].a[1];
                        }
                        if (perm) perm[k] = l+1;
                    }
                    *outsize = size > 0 ? l+1 : 0;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_integer) {
                if (precision == mtx_single) {
                    if (perm && size > 0) perm[0] = 0;
                    int64_t l = 0;
                    for (int64_t k = 1; k < size; k++) {
                        int ip = data->matrix_coordinate_integer_single[l].i;
                        int jp = data->matrix_coordinate_integer_single[l].j;
                        int i = data->matrix_coordinate_integer_single[k].i;
                        int j = data->matrix_coordinate_integer_single[k].j;
                        if (i == ip && j == jp) {
                            data->matrix_coordinate_integer_single[l].a +=
                                data->matrix_coordinate_integer_single[k].a;
                        } else {
                            l++;
                            data->matrix_coordinate_integer_single[l].i = i;
                            data->matrix_coordinate_integer_single[l].j = j;
                            data->matrix_coordinate_integer_single[l].a =
                                data->matrix_coordinate_integer_single[k].a;
                        }
                        if (perm) perm[k] = l+1;
                    }
                    *outsize = size > 0 ? l+1 : 0;
                } else if (precision == mtx_double) {
                    if (perm && size > 0) perm[0] = 0;
                    int64_t l = 0;
                    for (int64_t k = 1; k < size; k++) {
                        int ip = data->matrix_coordinate_integer_double[l].i;
                        int jp = data->matrix_coordinate_integer_double[l].j;
                        int i = data->matrix_coordinate_integer_double[k].i;
                        int j = data->matrix_coordinate_integer_double[k].j;
                        if (i == ip && j == jp) {
                            data->matrix_coordinate_integer_double[l].a +=
                                data->matrix_coordinate_integer_double[k].a;
                        } else {
                            l++;
                            data->matrix_coordinate_integer_double[l].i = i;
                            data->matrix_coordinate_integer_double[l].j = j;
                            data->matrix_coordinate_integer_double[l].a =
                                data->matrix_coordinate_integer_double[k].a;
                        }
                        if (perm) perm[k] = l+1;
                    }
                    *outsize = size > 0 ? l+1 : 0;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_pattern) {
                if (perm && size > 0) perm[0] = 0;
                int64_t l = 0;
                for (int64_t k = 1; k < size; k++) {
                    int ip = data->matrix_coordinate_pattern[l].i;
                    int jp = data->matrix_coordinate_pattern[l].j;
                    int i = data->matrix_coordinate_pattern[k].i;
                    int j = data->matrix_coordinate_pattern[k].j;
                    if (i == ip && j == jp) {
                        /* Nothing to be done */
                    } else {
                        l++;
                        data->matrix_coordinate_pattern[l].i = i;
                        data->matrix_coordinate_pattern[l].j = j;
                    }
                    if (perm) perm[k] = l+1;
                }
                *outsize = size > 0 ? l+1 : 0;
            } else { return MTX_ERR_INVALID_MTX_FIELD; }
        } else { return MTX_ERR_INVALID_MTX_FORMAT; }
    } else if (object == mtxfile_vector) {
        if (format == mtxfile_array) {
            *outsize = size;
            if (perm) {
                for (int64_t k = 0; k < size; k++)
                    perm[k] = k+1;
            }
            return MTX_SUCCESS;
        } else if (format == mtxfile_coordinate) {
            if (field == mtxfile_real) {
                if (precision == mtx_single) {
                    if (perm && size > 0) perm[0] = 0;
                    int64_t l = 0;
                    for (int64_t k = 1; k < size; k++) {
                        int ip = data->vector_coordinate_real_single[l].i;
                        int i = data->vector_coordinate_real_single[k].i;
                        if (i == ip) {
                            data->vector_coordinate_real_single[l].a +=
                                data->vector_coordinate_real_single[k].a;
                        } else {
                            l++;
                            data->vector_coordinate_real_single[l].i = i;
                            data->vector_coordinate_real_single[l].a =
                                data->vector_coordinate_real_single[k].a;
                        }
                        if (perm) perm[k] = l+1;
                    }
                    *outsize = size > 0 ? l+1 : 0;
                } else if (precision == mtx_double) {
                    if (perm && size > 0) perm[0] = 0;
                    int64_t l = 0;
                    for (int64_t k = 1; k < size; k++) {
                        int ip = data->vector_coordinate_real_double[l].i;
                        int i = data->vector_coordinate_real_double[k].i;
                        if (i == ip) {
                            data->vector_coordinate_real_double[l].a +=
                                data->vector_coordinate_real_double[k].a;
                        } else {
                            l++;
                            data->vector_coordinate_real_double[l].i = i;
                            data->vector_coordinate_real_double[l].a =
                                data->vector_coordinate_real_double[k].a;
                        }
                        if (perm) perm[k] = l+1;
                    }
                    *outsize = size > 0 ? l+1 : 0;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_complex) {
                if (precision == mtx_single) {
                    if (perm && size > 0) perm[0] = 0;
                    int64_t l = 0;
                    for (int64_t k = 1; k < size; k++) {
                        int ip = data->vector_coordinate_complex_single[l].i;
                        int i = data->vector_coordinate_complex_single[k].i;
                        if (i == ip) {
                            data->vector_coordinate_complex_single[l].a[0] +=
                                data->vector_coordinate_complex_single[k].a[0];
                            data->vector_coordinate_complex_single[l].a[1] +=
                                data->vector_coordinate_complex_single[k].a[1];
                        } else {
                            l++;
                            data->vector_coordinate_complex_single[l].i = i;
                            data->vector_coordinate_complex_single[l].a[0] =
                                data->vector_coordinate_complex_single[k].a[0];
                            data->vector_coordinate_complex_single[l].a[1] =
                                data->vector_coordinate_complex_single[k].a[1];
                        }
                        if (perm) perm[k] = l+1;
                    }
                    *outsize = size > 0 ? l+1 : 0;
                } else if (precision == mtx_double) {
                    if (perm && size > 0) perm[0] = 0;
                    int64_t l = 0;
                    for (int64_t k = 1; k < size; k++) {
                        int ip = data->vector_coordinate_complex_double[l].i;
                        int i = data->vector_coordinate_complex_double[k].i;
                        if (i == ip) {
                            data->vector_coordinate_complex_double[l].a[0] +=
                                data->vector_coordinate_complex_double[k].a[0];
                            data->vector_coordinate_complex_double[l].a[1] +=
                                data->vector_coordinate_complex_double[k].a[1];
                        } else {
                            l++;
                            data->vector_coordinate_complex_double[l].i = i;
                            data->vector_coordinate_complex_double[l].a[0] =
                                data->vector_coordinate_complex_double[k].a[0];
                            data->vector_coordinate_complex_double[l].a[1] =
                                data->vector_coordinate_complex_double[k].a[1];
                        }
                        if (perm) perm[k] = l+1;
                    }
                    *outsize = size > 0 ? l+1 : 0;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_integer) {
                if (precision == mtx_single) {
                    if (perm && size > 0) perm[0] = 0;
                    int64_t l = 0;
                    for (int64_t k = 1; k < size; k++) {
                        int ip = data->vector_coordinate_integer_single[l].i;
                        int i = data->vector_coordinate_integer_single[k].i;
                        if (i == ip) {
                            data->vector_coordinate_integer_single[l].a +=
                                data->vector_coordinate_integer_single[k].a;
                        } else {
                            l++;
                            data->vector_coordinate_integer_single[l].i = i;
                            data->vector_coordinate_integer_single[l].a =
                                data->vector_coordinate_integer_single[k].a;
                        }
                        if (perm) perm[k] = l+1;
                    }
                    *outsize = size > 0 ? l+1 : 0;
                } else if (precision == mtx_double) {
                    if (perm && size > 0) perm[0] = 0;
                    int64_t l = 0;
                    for (int64_t k = 1; k < size; k++) {
                        int ip = data->vector_coordinate_integer_double[l].i;
                        int i = data->vector_coordinate_integer_double[k].i;
                        if (i == ip) {
                            data->vector_coordinate_integer_double[l].a +=
                                data->vector_coordinate_integer_double[k].a;
                        } else {
                            l++;
                            data->vector_coordinate_integer_double[l].i = i;
                            data->vector_coordinate_integer_double[l].a =
                                data->vector_coordinate_integer_double[k].a;
                        }
                        if (perm) perm[k] = l+1;
                    }
                    *outsize = size > 0 ? l+1 : 0;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_pattern) {
                if (perm && size > 0) perm[0] = 0;
                int64_t l = 0;
                for (int64_t k = 1; k < size; k++) {
                    int ip = data->vector_coordinate_pattern[l].i;
                    int i = data->vector_coordinate_pattern[k].i;
                    if (i == ip) {
                        /* Nothing to be done */
                    } else {
                        l++;
                        data->vector_coordinate_pattern[l].i = i;
                    }
                    if (perm) perm[k] = l+1;
                }
                *outsize = size > 0 ? l+1 : 0;
            } else { return MTX_ERR_INVALID_MTX_FIELD; }
        } else { return MTX_ERR_INVALID_MTX_FORMAT; }
    } else { return MTX_ERR_INVALID_MTX_OBJECT; }
    return MTX_SUCCESS;
}

/*
 * Partitioning
 */

/**
 * ‘mtxfiledata_partition_rowwise()’ partitions data lines according
 * to a given row partition.
 *
 * The array ‘parts’ must contain enough storage for ‘size’ values of
 * type ‘int’. If successful, ‘parts’ will contain the part number of
 * each data line in the partitioning.
 *
 * If ‘format’ is ‘mtxfile_array’, then a non-negative ‘offset’ value
 * can be used to partition matrix or vector entries starting from the
 * specified offset, instead of beginning with the first entry of the
 * matrix or vector.
 */
int mtxfiledata_partition_rowwise(
    union mtxfiledata * data,
    enum mtxfileobject object,
    enum mtxfileformat format,
    enum mtxfilefield field,
    enum mtxprecision precision,
    int64_t num_rows,
    int64_t num_columns,
    int64_t offset,
    int64_t size,
    enum mtxpartitioning type,
    int num_parts,
    const int64_t * partsizes,
    int64_t blksize,
    const int64_t * parts,
    int64_t * partsptr,
    int64_t * perm)
{
    int64_t * rowidx = malloc(size * sizeof(int64_t));
    if (!rowidx) return MTX_ERR_ERRNO;
    int err = mtxfiledata_rowcolidx64(
        data, object, format, field, precision,
        num_rows, num_columns, offset, size, rowidx, NULL);
    if (err) { free(rowidx); return err; }
    for (int64_t k = 0; k < size; k++) rowidx[k]--;
    int * dstpart = malloc(size * sizeof(int));
    if (!dstpart) { free(rowidx); return MTX_ERR_ERRNO; }
    err = partition_int64(
        type, num_rows, num_parts, partsizes, blksize, parts,
        size, sizeof(int64_t), rowidx, dstpart);
    if (err) { free(rowidx); return err; }
    for (int p = 0; p <= num_parts; p++) partsptr[p] = 0;
    for (int64_t k = 0; k < size; k++) partsptr[dstpart[k]+1]++;
    for (int p = 1; p <= num_parts; p++) partsptr[p] += partsptr[p-1];
    err = mtxfiledata_sort_int(
        data, object, format, field, precision,
        num_rows, num_columns, size, dstpart, perm);
    if (err) { free(dstpart); free(rowidx); return err; }
    free(dstpart); free(rowidx);
    return MTX_SUCCESS;
}

/**
 * ‘mtxfiledata_partition()’ partitions data lines according to given
 * row and column partitions.
 *
 * The array ‘parts’ must contain enough storage for ‘size’ values of
 * type ‘int’. If successful, ‘parts’ will contain the part number of
 * each data line in the partitioning.
 *
 * The partitions ‘rowpart’ or ‘colpart’ are allowed to be ‘NULL’, in
 * which case a trivial, singleton partition is used for the rows or
 * columns, respectively.
 *
 * If ‘format’ is ‘mtxfile_array’, then a non-negative ‘offset’ value
 * can be used to partition matrix or vector entries starting from the
 * specified offset, instead of beginning with the first entry of the
 * matrix or vector.
 */
int mtxfiledata_partition(
    const union mtxfiledata * data,
    enum mtxfileobject object,
    enum mtxfileformat format,
    enum mtxfilefield field,
    enum mtxprecision precision,
    int64_t num_rows,
    int64_t num_columns,
    int64_t offset,
    int64_t size,
    const struct mtxpartition * rowpart,
    const struct mtxpartition * colpart,
    int * parts,
    int64_t * localrowidx,
    int64_t * localcolidx)
{
    int err;
    int num_col_parts = colpart ? colpart->num_parts : 1;

    /* Extract row and column indices */
    int * rowidx = (rowpart || localrowidx) ? malloc(size * sizeof(int)) : NULL;
    if ((rowpart || localrowidx) && !rowidx)
        return MTX_ERR_ERRNO;
    int * colidx = (colpart || localcolidx) ? malloc(size * sizeof(int)) : NULL;
    if ((colpart || localcolidx) && !colidx) {
        if (rowpart || localrowidx) free(rowidx);
        return MTX_ERR_ERRNO;
    }

    err = mtxfiledata_rowcolidx(
        data, object, format, field, precision,
        num_rows, num_columns, offset, size,
        rowidx, colidx);
    if (err) {
        if (colpart || localcolidx) free(colidx);
        if (rowpart || localrowidx) free(rowidx);
        return err;
    }

    int64_t * elements = malloc(size * sizeof(int64_t));
    if (!elements) {
        if (colpart || localcolidx) free(colidx);
        if (rowpart || localrowidx) free(rowidx);
        return MTX_ERR_ERRNO;
    }

    /* Assign part numbers to each data line and compute the final
     * part number of the product partition. Note that row and column
     * indices are adjusted from 1-based to 0-based indexing. */
    if (rowpart && colpart) {
        for (int64_t k = 0; k < size; k++)
            elements[k] = rowidx[k]-1;
        int * rowparts = rowidx;
        err = mtxpartition_assign(
            rowpart, size, elements, rowparts, localrowidx);
        if (err) {
            free(elements);
            if (colpart || localcolidx) free(colidx);
            if (rowpart || localrowidx) free(rowidx);
            return err;
        }
        for (int64_t k = 0; k < size; k++)
            elements[k] = colidx[k]-1;
        int * colparts = colidx;
        err = mtxpartition_assign(
            colpart, size, elements, colparts, localcolidx);
        if (err) {
            free(elements);
            if (colpart || localcolidx) free(colidx);
            if (rowpart || localrowidx) free(rowidx);
            return err;
        }
        free(elements);
        for (int64_t k = 0; k < size; k++)
            parts[k] = rowparts[k] * num_col_parts + colparts[k];
    } else if (rowpart) {
        for (int64_t k = 0; k < size; k++)
            elements[k] = rowidx[k]-1;
        err = mtxpartition_assign(
            rowpart, size, elements, parts, localrowidx);
        if (err) {
            free(elements);
            if (colpart || localcolidx) free(colidx);
            if (rowpart || localrowidx) free(rowidx);
            return err;
        }
        free(elements);
        if (localcolidx) {
            for (int64_t k = 0; k < size; k++)
                localcolidx[k] = colidx[k]-1;
        }
    } else if (colpart) {
        for (int64_t k = 0; k < size; k++)
            elements[k] = colidx[k]-1;
        err = mtxpartition_assign(
            colpart, size, elements, parts, localcolidx);
        if (err) {
            free(elements);
            if (colpart || localcolidx) free(colidx);
            if (rowpart || localrowidx) free(rowidx);
            return err;
        }
        free(elements);
        if (localrowidx) {
            for (int64_t k = 0; k < size; k++)
                localrowidx[k] = rowidx[k]-1;
        }
    }
    if (colpart || localcolidx) free(colidx);
    if (rowpart || localrowidx) free(rowidx);
    return MTX_SUCCESS;
}

/*
 * Reordering
 */

/**
 * ‘mtxfiledata_reorder()’ reorders the elements of a matrix or
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
int mtxfiledata_reorder(
    const union mtxfiledata * data,
    enum mtxfileobject object,
    enum mtxfileformat format,
    enum mtxfilefield field,
    enum mtxprecision precision,
    int64_t size,
    int64_t offset,
    int64_t num_rows,
    const int * rowperm,
    int64_t num_columns,
    const int * colperm)
{
    int err;
    if (format == mtxfile_array) {
        if (rowperm) {
            for (int64_t i = 0; i < num_rows; i++) {
                if (rowperm[i] <= 0 || rowperm[i] > num_rows)
                    return MTX_ERR_INDEX_OUT_OF_BOUNDS;
            }
        }
        if (colperm) {
            for (int64_t i = 0; i < num_columns; i++) {
                if (colperm[i] <= 0 || colperm[i] > num_columns)
                    return MTX_ERR_INDEX_OUT_OF_BOUNDS;
            }
        }

        /* Create a temporary copy of the data to be permuted. */
        union mtxfiledata original;
        err = mtxfiledata_alloc(
            &original, object, format, field, precision, size);
        if (err)
            return err;
        err = mtxfiledata_copy(
            &original, data, object, format, field, precision, size, 0, offset);
        if (err) {
            mtxfiledata_free(&original, object, format, field, precision);
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
                                int64_t k = i*num_columns+j;
                                int64_t l = (int64_t) (rowperm[i]-1)*num_columns + colperm[j]-1;
                                dst[k] = src[l];
                            }
                        }
                    } else if (rowperm) {
                        for (int64_t i = 0; i < num_rows; i++) {
                            for (int64_t j = 0; j < num_columns; j++) {
                                int64_t k = i*num_columns+j;
                                int64_t l = (int64_t) (rowperm[i]-1)*num_columns + j;
                                dst[k] = src[l];
                            }
                        }
                    } else if (colperm) {
                        for (int64_t i = 0; i < num_rows; i++) {
                            for (int64_t j = 0; j < num_columns; j++) {
                                int64_t k = i*num_columns+j;
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
                                int64_t k = i*num_columns+j;
                                int64_t l = (int64_t) (rowperm[i]-1)*num_columns + colperm[j]-1;
                                dst[k] = src[l];
                            }
                        }
                    } else if (rowperm) {
                        for (int64_t i = 0; i < num_rows; i++) {
                            for (int64_t j = 0; j < num_columns; j++) {
                                int64_t k = i*num_columns+j;
                                int64_t l = (int64_t) (rowperm[i]-1)*num_columns + j;
                                dst[k] = src[l];
                            }
                        }
                    } else if (colperm) {
                        for (int64_t i = 0; i < num_rows; i++) {
                            for (int64_t j = 0; j < num_columns; j++) {
                                int64_t k = i*num_columns+j;
                                int64_t l = (int64_t) i*num_columns + colperm[j]-1;
                                dst[k] = src[l];
                            }
                        }
                    }
                } else {
                    mtxfiledata_free(&original, object, format, field, precision);
                    return MTX_ERR_INVALID_PRECISION;
                }
            } else if (field == mtxfile_complex) {
                if (precision == mtx_single) {
                    const float (* src)[2] = original.array_complex_single;
                    float (* dst)[2] = data->array_complex_single;
                    if (rowperm && colperm) {
                        for (int64_t i = 0; i < num_rows; i++) {
                            for (int64_t j = 0; j < num_columns; j++) {
                                int64_t k = i*num_columns+j;
                                int64_t l = (int64_t) (rowperm[i]-1)*num_columns + colperm[j]-1;
                                dst[k][0] = src[l][0];
                                dst[k][1] = src[l][1];
                            }
                        }
                    } else if (rowperm) {
                        for (int64_t i = 0; i < num_rows; i++) {
                            for (int64_t j = 0; j < num_columns; j++) {
                                int64_t k = i*num_columns+j;
                                int64_t l = (int64_t) (rowperm[i]-1)*num_columns + j;
                                dst[k][0] = src[l][0];
                                dst[k][1] = src[l][1];
                            }
                        }
                    } else if (colperm) {
                        for (int64_t i = 0; i < num_rows; i++) {
                            for (int64_t j = 0; j < num_columns; j++) {
                                int64_t k = i*num_columns+j;
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
                                int64_t k = i*num_columns+j;
                                int64_t l = (int64_t) (rowperm[i]-1)*num_columns + colperm[j]-1;
                                dst[k][0] = src[l][0];
                                dst[k][1] = src[l][1];
                            }
                        }
                    } else if (rowperm) {
                        for (int64_t i = 0; i < num_rows; i++) {
                            for (int64_t j = 0; j < num_columns; j++) {
                                int64_t k = i*num_columns+j;
                                int64_t l = (int64_t) (rowperm[i]-1)*num_columns + j;
                                dst[k][0] = src[l][0];
                                dst[k][1] = src[l][1];
                            }
                        }
                    } else if (colperm) {
                        for (int64_t i = 0; i < num_rows; i++) {
                            for (int64_t j = 0; j < num_columns; j++) {
                                int64_t k = i*num_columns+j;
                                int64_t l = (int64_t) i*num_columns + colperm[j]-1;
                                dst[k][0] = src[l][0];
                                dst[k][1] = src[l][1];
                            }
                        }
                    }
                } else {
                    mtxfiledata_free(&original, object, format, field, precision);
                    return MTX_ERR_INVALID_PRECISION;
                }
            } else if (field == mtxfile_integer) {
                if (precision == mtx_single) {
                    const int32_t * src = original.array_integer_single;
                    int32_t * dst = data->array_integer_single;
                    if (rowperm && colperm) {
                        for (int64_t i = 0; i < num_rows; i++) {
                            for (int64_t j = 0; j < num_columns; j++) {
                                int64_t k = i*num_columns+j;
                                int64_t l = (int64_t) (rowperm[i]-1)*num_columns + colperm[j]-1;
                                dst[k] = src[l];
                            }
                        }
                    } else if (rowperm) {
                        for (int64_t i = 0; i < num_rows; i++) {
                            for (int64_t j = 0; j < num_columns; j++) {
                                int64_t k = i*num_columns+j;
                                int64_t l = (int64_t) (rowperm[i]-1)*num_columns + j;
                                dst[k] = src[l];
                            }
                        }
                    } else if (colperm) {
                        for (int64_t i = 0; i < num_rows; i++) {
                            for (int64_t j = 0; j < num_columns; j++) {
                                int64_t k = i*num_columns+j;
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
                                int64_t k = i*num_columns+j;
                                int64_t l = (int64_t) (rowperm[i]-1)*num_columns + colperm[j]-1;
                                dst[k] = src[l];
                            }
                        }
                    } else if (rowperm) {
                        for (int64_t i = 0; i < num_rows; i++) {
                            for (int64_t j = 0; j < num_columns; j++) {
                                int64_t k = i*num_columns+j;
                                int64_t l = (int64_t) (rowperm[i]-1)*num_columns + j;
                                dst[k] = src[l];
                            }
                        }
                    } else if (colperm) {
                        for (int64_t i = 0; i < num_rows; i++) {
                            for (int64_t j = 0; j < num_columns; j++) {
                                int64_t k = i*num_columns+j;
                                int64_t l = (int64_t) i*num_columns + colperm[j]-1;
                                dst[k] = src[l];
                            }
                        }
                    }
                } else {
                    mtxfiledata_free(&original, object, format, field, precision);
                    return MTX_ERR_INVALID_PRECISION;
                }
            } else {
                mtxfiledata_free(&original, object, format, field, precision);
                return MTX_ERR_INVALID_MTX_FIELD;
            }

        } else if (object == mtxfile_vector) {
            if (field == mtxfile_real) {
                if (precision == mtx_single) {
                    const float * src = original.array_real_single;
                    float * dst = data->array_real_single;
                    if (rowperm) {
                        for (int64_t i = 0; i < num_rows; i++)
                            dst[i] = src[rowperm[i]-1];
                    }
                } else if (precision == mtx_double) {
                    const double * src = original.array_real_double;
                    double * dst = data->array_real_double;
                    if (rowperm) {
                        for (int64_t i = 0; i < num_rows; i++)
                            dst[i] = src[rowperm[i]-1];
                    }
                } else {
                    mtxfiledata_free(&original, object, format, field, precision);
                    return MTX_ERR_INVALID_PRECISION;
                }
            } else if (field == mtxfile_complex) {
                if (precision == mtx_single) {
                    const float (* src)[2] = original.array_complex_single;
                    float (* dst)[2] = data->array_complex_single;
                    if (rowperm) {
                        for (int64_t i = 0; i < num_rows; i++) {
                            dst[i][0] = src[rowperm[i]-1][0];
                            dst[i][1] = src[rowperm[i]-1][1];
                        }
                    }
                } else if (precision == mtx_double) {
                    const double (* src)[2] = original.array_complex_double;
                    double (* dst)[2] = data->array_complex_double;
                    if (rowperm) {
                        for (int64_t i = 0; i < num_rows; i++) {
                            dst[i][0] = src[rowperm[i]-1][0];
                            dst[i][1] = src[rowperm[i]-1][1];
                        }
                    }
                } else {
                    mtxfiledata_free(&original, object, format, field, precision);
                    return MTX_ERR_INVALID_PRECISION;
                }
            } else if (field == mtxfile_integer) {
                if (precision == mtx_single) {
                    const int32_t * src = original.array_integer_single;
                    int32_t * dst = data->array_integer_single;
                    if (rowperm) {
                        for (int64_t i = 0; i < num_rows; i++)
                            dst[i] = src[rowperm[i]-1];
                    }
                } else if (precision == mtx_double) {
                    const int64_t * src = original.array_integer_double;
                    int64_t * dst = data->array_integer_double;
                    if (rowperm) {
                        for (int64_t i = 0; i < num_rows; i++)
                            dst[i] = src[rowperm[i]-1];
                    }
                } else {
                    mtxfiledata_free(&original, object, format, field, precision);
                    return MTX_ERR_INVALID_PRECISION;
                }
            } else {
                mtxfiledata_free(&original, object, format, field, precision);
                return MTX_ERR_INVALID_MTX_FIELD;
            }
        } else {
            mtxfiledata_free(&original, object, format, field, precision);
            return MTX_ERR_INVALID_MTX_OBJECT;
        }
        mtxfiledata_free(&original, object, format, field, precision);

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
                } else { return MTX_ERR_INVALID_PRECISION; }
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
                } else { return MTX_ERR_INVALID_PRECISION; }
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
                } else { return MTX_ERR_INVALID_PRECISION; }
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
            } else { return MTX_ERR_INVALID_MTX_FIELD; }

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
                } else { return MTX_ERR_INVALID_PRECISION; }
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
                } else { return MTX_ERR_INVALID_PRECISION; }
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
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_pattern) {
                struct mtxfile_vector_coordinate_pattern * dst =
                    data->vector_coordinate_pattern;
                if (rowperm) {
                    for (int64_t k = 0; k < size; k++)
                        dst[k+offset].i = rowperm[dst[k+offset].i-1];
                }
            } else { return MTX_ERR_INVALID_MTX_FIELD; }
        } else { return MTX_ERR_INVALID_MTX_OBJECT; }
    } else { return MTX_ERR_INVALID_MTX_FORMAT; }
    return MTX_SUCCESS;
}

/*
 * MPI functions
 */

#ifdef LIBMTX_HAVE_MPI
/**
 * ‘mtxfiledata_mpidatatype()’ creates a custom MPI data type for
 * sending or receiving data lines.
 *
 * The user is responsible for calling ‘MPI_Type_free()’ on the
 * returned datatype.
 */
int mtxfiledata_mpidatatype(
    const union mtxfiledata * data,
    enum mtxfileobject object,
    enum mtxfileformat format,
    enum mtxfilefield field,
    enum mtxprecision precision,
    MPI_Datatype * datatype,
    int * mpierrcode)
{
    int num_elements;
    int block_lengths[3];
    MPI_Datatype element_types[3];
    MPI_Aint element_offsets[3];
    if (format == mtxfile_array) {
        if (field == mtxfile_real) {
            if (precision == mtx_single) {
                num_elements = 1;
                element_types[0] = MPI_FLOAT;
                block_lengths[0] = 1;
                element_offsets[0] = 0;
            } else if (precision == mtx_double) {
                num_elements = 1;
                element_types[0] = MPI_DOUBLE;
                block_lengths[0] = 1;
                element_offsets[0] = 0;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (field == mtxfile_complex) {
            if (precision == mtx_single) {
                num_elements = 1;
                element_types[0] = MPI_FLOAT;
                block_lengths[0] = 2;
                element_offsets[0] = 0;
            } else if (precision == mtx_double) {
                num_elements = 1;
                element_types[0] = MPI_DOUBLE;
                block_lengths[0] = 1;
                element_offsets[0] = 0;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (field == mtxfile_integer) {
            if (precision == mtx_single) {
                num_elements = 1;
                element_types[0] = MPI_INT32_T;
                block_lengths[0] = 1;
                element_offsets[0] = 0;
            } else if (precision == mtx_double) {
                num_elements = 1;
                element_types[0] = MPI_INT64_T;
                block_lengths[0] = 1;
                element_offsets[0] = 0;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else { return MTX_ERR_INVALID_MTX_FIELD; }
    } else if (format == mtxfile_coordinate) {
        if (object == mtxfile_matrix) {
            if (field == mtxfile_real) {
                if (precision == mtx_single) {
                    num_elements = 3;
                    element_types[0] = MPI_INT64_T;
                    block_lengths[0] = 1;
                    element_offsets[0] =
                        offsetof(struct mtxfile_matrix_coordinate_real_single, i);
                    element_types[1] = MPI_INT64_T;
                    block_lengths[1] = 1;
                    element_offsets[1] =
                        offsetof(struct mtxfile_matrix_coordinate_real_single, j);
                    element_types[2] = MPI_FLOAT;
                    block_lengths[2] = 1;
                    element_offsets[2] =
                        offsetof(struct mtxfile_matrix_coordinate_real_single, a);
                } else if (precision == mtx_double) {
                    num_elements = 3;
                    element_types[0] = MPI_INT64_T;
                    block_lengths[0] = 1;
                    element_offsets[0] =
                        offsetof(struct mtxfile_matrix_coordinate_real_double, i);
                    element_types[1] = MPI_INT64_T;
                    block_lengths[1] = 1;
                    element_offsets[1] =
                        offsetof(struct mtxfile_matrix_coordinate_real_double, j);
                    element_types[2] = MPI_DOUBLE;
                    block_lengths[2] = 1;
                    element_offsets[2] =
                        offsetof(struct mtxfile_matrix_coordinate_real_double, a);
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_complex) {
                if (precision == mtx_single) {
                    num_elements = 3;
                    element_types[0] = MPI_INT64_T;
                    block_lengths[0] = 1;
                    element_offsets[0] =
                        offsetof(struct mtxfile_matrix_coordinate_complex_single, i);
                    element_types[1] = MPI_INT64_T;
                    block_lengths[1] = 1;
                    element_offsets[1] =
                        offsetof(struct mtxfile_matrix_coordinate_complex_single, j);
                    element_types[2] = MPI_FLOAT;
                    block_lengths[2] = 2;
                    element_offsets[2] =
                        offsetof(struct mtxfile_matrix_coordinate_complex_single, a);
                } else if (precision == mtx_double) {
                    num_elements = 3;
                    element_types[0] = MPI_INT64_T;
                    block_lengths[0] = 1;
                    element_offsets[0] =
                        offsetof(struct mtxfile_matrix_coordinate_complex_double, i);
                    element_types[1] = MPI_INT64_T;
                    block_lengths[1] = 1;
                    element_offsets[1] =
                        offsetof(struct mtxfile_matrix_coordinate_complex_double, j);
                    element_types[2] = MPI_DOUBLE;
                    block_lengths[2] = 2;
                    element_offsets[2] =
                        offsetof(struct mtxfile_matrix_coordinate_complex_double, a);
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_integer) {
                if (precision == mtx_single) {
                    num_elements = 3;
                    element_types[0] = MPI_INT64_T;
                    block_lengths[0] = 1;
                    element_offsets[0] =
                        offsetof(struct mtxfile_matrix_coordinate_integer_single, i);
                    element_types[1] = MPI_INT64_T;
                    block_lengths[1] = 1;
                    element_offsets[1] =
                        offsetof(struct mtxfile_matrix_coordinate_integer_single, j);
                    element_types[2] = MPI_INT32_T;
                    block_lengths[2] = 1;
                    element_offsets[2] =
                        offsetof(struct mtxfile_matrix_coordinate_integer_single, a);
                } else if (precision == mtx_double) {
                    num_elements = 3;
                    element_types[0] = MPI_INT64_T;
                    block_lengths[0] = 1;
                    element_offsets[0] =
                        offsetof(struct mtxfile_matrix_coordinate_integer_double, i);
                    element_types[1] = MPI_INT64_T;
                    block_lengths[1] = 1;
                    element_offsets[1] =
                        offsetof(struct mtxfile_matrix_coordinate_integer_double, j);
                    element_types[2] = MPI_INT64_T;
                    block_lengths[2] = 1;
                    element_offsets[2] =
                        offsetof(struct mtxfile_matrix_coordinate_integer_double, a);
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_pattern) {
                num_elements = 2;
                element_types[0] = MPI_INT64_T;
                block_lengths[0] = 1;
                element_offsets[0] =
                    offsetof(struct mtxfile_matrix_coordinate_pattern, i);
                element_types[1] = MPI_INT64_T;
                block_lengths[1] = 1;
                element_offsets[1] =
                    offsetof(struct mtxfile_matrix_coordinate_pattern, j);
            } else { return MTX_ERR_INVALID_MTX_FIELD; }
        } else if (object == mtxfile_vector) {
            if (field == mtxfile_real) {
                if (precision == mtx_single) {
                    num_elements = 2;
                    element_types[0] = MPI_INT64_T;
                    block_lengths[0] = 1;
                    element_offsets[0] =
                        offsetof(struct mtxfile_vector_coordinate_real_single, i);
                    element_types[1] = MPI_FLOAT;
                    block_lengths[1] = 1;
                    element_offsets[1] =
                        offsetof(struct mtxfile_vector_coordinate_real_single, a);
                } else if (precision == mtx_double) {
                    num_elements = 2;
                    element_types[0] = MPI_INT64_T;
                    block_lengths[0] = 1;
                    element_offsets[0] =
                        offsetof(struct mtxfile_vector_coordinate_real_double, i);
                    element_types[1] = MPI_DOUBLE;
                    block_lengths[1] = 1;
                    element_offsets[1] =
                        offsetof(struct mtxfile_vector_coordinate_real_double, a);
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_complex) {
                if (precision == mtx_single) {
                    num_elements = 2;
                    element_types[0] = MPI_INT64_T;
                    block_lengths[0] = 1;
                    element_offsets[0] =
                        offsetof(struct mtxfile_vector_coordinate_complex_single, i);
                    element_types[1] = MPI_FLOAT;
                    block_lengths[1] = 2;
                    element_offsets[1] =
                        offsetof(struct mtxfile_vector_coordinate_complex_single, a);
                } else if (precision == mtx_double) {
                    num_elements = 2;
                    element_types[0] = MPI_INT64_T;
                    block_lengths[0] = 1;
                    element_offsets[0] =
                        offsetof(struct mtxfile_vector_coordinate_complex_double, i);
                    element_types[1] = MPI_DOUBLE;
                    block_lengths[1] = 2;
                    element_offsets[1] =
                        offsetof(struct mtxfile_vector_coordinate_complex_double, a);
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_integer) {
                if (precision == mtx_single) {
                    num_elements = 2;
                    element_types[0] = MPI_INT64_T;
                    block_lengths[0] = 1;
                    element_offsets[0] =
                        offsetof(struct mtxfile_vector_coordinate_integer_single, i);
                    element_types[1] = MPI_INT32_T;
                    block_lengths[1] = 1;
                    element_offsets[1] =
                        offsetof(struct mtxfile_vector_coordinate_integer_single, a);
                } else if (precision == mtx_double) {
                    num_elements = 2;
                    element_types[0] = MPI_INT64_T;
                    block_lengths[0] = 1;
                    element_offsets[0] =
                        offsetof(struct mtxfile_vector_coordinate_integer_double, i);
                    element_types[1] = MPI_INT64_T;
                    block_lengths[1] = 1;
                    element_offsets[1] =
                        offsetof(struct mtxfile_vector_coordinate_integer_double, a);
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_pattern) {
                num_elements = 1;
                element_types[0] = MPI_INT64_T;
                block_lengths[0] = 1;
                element_offsets[0] =
                    offsetof(struct mtxfile_vector_coordinate_pattern, i);
            } else { return MTX_ERR_INVALID_MTX_FIELD; }
        } else { return MTX_ERR_INVALID_MTX_OBJECT; }
    } else { return MTX_ERR_INVALID_MTX_FORMAT; }

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

static int mtxfile_array_complex_datatype(
    MPI_Datatype * datatype,
    enum mtxprecision precision,
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

static int mtxfiledata_send_array(
    const union mtxfiledata * data,
    enum mtxfilefield field,
    enum mtxprecision precision,
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
        } else { return MTX_ERR_INVALID_PRECISION; }
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
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_MTX_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxfile_coordinate_datatype()’ creates a custom MPI data type for
 * sending or receiving data in coordinate format.
 *
 * The user is responsible for calling ‘MPI_Type_free()’ on the
 * returned datatype.
 */
static int mtxfile_coordinate_datatype(
    enum mtxfileobject object,
    enum mtxfilefield field,
    enum mtxprecision precision,
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
                element_types[0] = MPI_INT64_T;
                block_lengths[0] = 1;
                element_offsets[0] =
                    offsetof(struct mtxfile_matrix_coordinate_real_single, i);
                element_types[1] = MPI_INT64_T;
                block_lengths[1] = 1;
                element_offsets[1] =
                    offsetof(struct mtxfile_matrix_coordinate_real_single, j);
                element_types[2] = MPI_FLOAT;
                block_lengths[2] = 1;
                element_offsets[2] =
                    offsetof(struct mtxfile_matrix_coordinate_real_single, a);
            } else if (precision == mtx_double) {
                num_elements = 3;
                element_types[0] = MPI_INT64_T;
                block_lengths[0] = 1;
                element_offsets[0] =
                    offsetof(struct mtxfile_matrix_coordinate_real_double, i);
                element_types[1] = MPI_INT64_T;
                block_lengths[1] = 1;
                element_offsets[1] =
                    offsetof(struct mtxfile_matrix_coordinate_real_double, j);
                element_types[2] = MPI_DOUBLE;
                block_lengths[2] = 1;
                element_offsets[2] =
                    offsetof(struct mtxfile_matrix_coordinate_real_double, a);
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (field == mtxfile_complex) {
            if (precision == mtx_single) {
                num_elements = 3;
                element_types[0] = MPI_INT64_T;
                block_lengths[0] = 1;
                element_offsets[0] =
                    offsetof(struct mtxfile_matrix_coordinate_complex_single, i);
                element_types[1] = MPI_INT64_T;
                block_lengths[1] = 1;
                element_offsets[1] =
                    offsetof(struct mtxfile_matrix_coordinate_complex_single, j);
                element_types[2] = MPI_FLOAT;
                block_lengths[2] = 2;
                element_offsets[2] =
                    offsetof(struct mtxfile_matrix_coordinate_complex_single, a);
            } else if (precision == mtx_double) {
                num_elements = 3;
                element_types[0] = MPI_INT64_T;
                block_lengths[0] = 1;
                element_offsets[0] =
                    offsetof(struct mtxfile_matrix_coordinate_complex_double, i);
                element_types[1] = MPI_INT64_T;
                block_lengths[1] = 1;
                element_offsets[1] =
                    offsetof(struct mtxfile_matrix_coordinate_complex_double, j);
                element_types[2] = MPI_DOUBLE;
                block_lengths[2] = 2;
                element_offsets[2] =
                    offsetof(struct mtxfile_matrix_coordinate_complex_double, a);
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (field == mtxfile_integer) {
            if (precision == mtx_single) {
                num_elements = 3;
                element_types[0] = MPI_INT64_T;
                block_lengths[0] = 1;
                element_offsets[0] =
                    offsetof(struct mtxfile_matrix_coordinate_integer_single, i);
                element_types[1] = MPI_INT64_T;
                block_lengths[1] = 1;
                element_offsets[1] =
                    offsetof(struct mtxfile_matrix_coordinate_integer_single, j);
                element_types[2] = MPI_INT32_T;
                block_lengths[2] = 1;
                element_offsets[2] =
                    offsetof(struct mtxfile_matrix_coordinate_integer_single, a);
            } else if (precision == mtx_double) {
                num_elements = 3;
                element_types[0] = MPI_INT64_T;
                block_lengths[0] = 1;
                element_offsets[0] =
                    offsetof(struct mtxfile_matrix_coordinate_integer_double, i);
                element_types[1] = MPI_INT64_T;
                block_lengths[1] = 1;
                element_offsets[1] =
                    offsetof(struct mtxfile_matrix_coordinate_integer_double, j);
                element_types[2] = MPI_INT64_T;
                block_lengths[2] = 1;
                element_offsets[2] =
                    offsetof(struct mtxfile_matrix_coordinate_integer_double, a);
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (field == mtxfile_pattern) {
            num_elements = 2;
            element_types[0] = MPI_INT64_T;
            block_lengths[0] = 1;
            element_offsets[0] =
                offsetof(struct mtxfile_matrix_coordinate_pattern, i);
            element_types[1] = MPI_INT64_T;
            block_lengths[1] = 1;
            element_offsets[1] =
                offsetof(struct mtxfile_matrix_coordinate_pattern, j);
        } else { return MTX_ERR_INVALID_MTX_FIELD; }
    } else if (object == mtxfile_vector) {
        if (field == mtxfile_real) {
            if (precision == mtx_single) {
                num_elements = 2;
                element_types[0] = MPI_INT64_T;
                block_lengths[0] = 1;
                element_offsets[0] =
                    offsetof(struct mtxfile_vector_coordinate_real_single, i);
                element_types[1] = MPI_FLOAT;
                block_lengths[1] = 1;
                element_offsets[1] =
                    offsetof(struct mtxfile_vector_coordinate_real_single, a);
            } else if (precision == mtx_double) {
                num_elements = 2;
                element_types[0] = MPI_INT64_T;
                block_lengths[0] = 1;
                element_offsets[0] =
                    offsetof(struct mtxfile_vector_coordinate_real_double, i);
                element_types[1] = MPI_DOUBLE;
                block_lengths[1] = 1;
                element_offsets[1] =
                    offsetof(struct mtxfile_vector_coordinate_real_double, a);
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (field == mtxfile_complex) {
            if (precision == mtx_single) {
                num_elements = 2;
                element_types[0] = MPI_INT64_T;
                block_lengths[0] = 1;
                element_offsets[0] =
                    offsetof(struct mtxfile_vector_coordinate_complex_single, i);
                element_types[1] = MPI_FLOAT;
                block_lengths[1] = 2;
                element_offsets[1] =
                    offsetof(struct mtxfile_vector_coordinate_complex_single, a);
            } else if (precision == mtx_double) {
                num_elements = 2;
                element_types[0] = MPI_INT64_T;
                block_lengths[0] = 1;
                element_offsets[0] =
                    offsetof(struct mtxfile_vector_coordinate_complex_double, i);
                element_types[1] = MPI_DOUBLE;
                block_lengths[1] = 2;
                element_offsets[1] =
                    offsetof(struct mtxfile_vector_coordinate_complex_double, a);
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (field == mtxfile_integer) {
            if (precision == mtx_single) {
                num_elements = 2;
                element_types[0] = MPI_INT64_T;
                block_lengths[0] = 1;
                element_offsets[0] =
                    offsetof(struct mtxfile_vector_coordinate_integer_single, i);
                element_types[1] = MPI_INT32_T;
                block_lengths[1] = 1;
                element_offsets[1] =
                    offsetof(struct mtxfile_vector_coordinate_integer_single, a);
            } else if (precision == mtx_double) {
                num_elements = 2;
                element_types[0] = MPI_INT64_T;
                block_lengths[0] = 1;
                element_offsets[0] =
                    offsetof(struct mtxfile_vector_coordinate_integer_double, i);
                element_types[1] = MPI_INT64_T;
                block_lengths[1] = 1;
                element_offsets[1] =
                    offsetof(struct mtxfile_vector_coordinate_integer_double, a);
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (field == mtxfile_pattern) {
            num_elements = 1;
            element_types[0] = MPI_INT64_T;
            block_lengths[0] = 1;
            element_offsets[0] =
                offsetof(struct mtxfile_vector_coordinate_pattern, i);
        } else { return MTX_ERR_INVALID_MTX_FIELD; }
    } else { return MTX_ERR_INVALID_MTX_OBJECT; }

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

static int mtxfiledata_send_coordinate(
    const union mtxfiledata * data,
    enum mtxfileobject object,
    enum mtxfilefield field,
    enum mtxprecision precision,
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
                if (*mpierrcode) {
                    MPI_Type_free(&datatype);
                    return MTX_ERR_MPI;
                }
            } else if (precision == mtx_double) {
                *mpierrcode = MPI_Send(
                    &data->matrix_coordinate_real_double[offset],
                    size, datatype, dest, tag, comm);
                if (*mpierrcode) {
                    MPI_Type_free(&datatype);
                    return MTX_ERR_MPI;
                }
            } else {
                MPI_Type_free(&datatype);
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_complex) {
            if (precision == mtx_single) {
                *mpierrcode = MPI_Send(
                    &data->matrix_coordinate_complex_single[offset],
                    size, datatype, dest, tag, comm);
                if (*mpierrcode) {
                    MPI_Type_free(&datatype);
                    return MTX_ERR_MPI;
                }
            } else if (precision == mtx_double) {
                *mpierrcode = MPI_Send(
                    &data->matrix_coordinate_complex_double[offset],
                    size, datatype, dest, tag, comm);
                if (*mpierrcode) {
                    MPI_Type_free(&datatype);
                    return MTX_ERR_MPI;
                }
            } else {
                MPI_Type_free(&datatype);
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_integer) {
            if (precision == mtx_single) {
                *mpierrcode = MPI_Send(
                    &data->matrix_coordinate_integer_single[offset],
                    size, datatype, dest, tag, comm);
                if (*mpierrcode) {
                    MPI_Type_free(&datatype);
                    return MTX_ERR_MPI;
                }
            } else if (precision == mtx_double) {
                *mpierrcode = MPI_Send(
                    &data->matrix_coordinate_integer_double[offset],
                    size, datatype, dest, tag, comm);
                if (*mpierrcode) {
                    MPI_Type_free(&datatype);
                    return MTX_ERR_MPI;
                }
            } else {
                MPI_Type_free(&datatype);
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_pattern) {
            *mpierrcode = MPI_Send(
                &data->matrix_coordinate_pattern[offset],
                size, datatype, dest, tag, comm);
                if (*mpierrcode) {
                    MPI_Type_free(&datatype);
                    return MTX_ERR_MPI;
                }
        } else {
            MPI_Type_free(&datatype);
            return MTX_ERR_INVALID_MTX_FIELD;
        }

    } else if (object == mtxfile_vector) {
        if (field == mtxfile_real) {
            if (precision == mtx_single) {
                *mpierrcode = MPI_Send(
                    &data->vector_coordinate_real_single[offset],
                    size, datatype, dest, tag, comm);
                if (*mpierrcode) {
                    MPI_Type_free(&datatype);
                    return MTX_ERR_MPI;
                }
            } else if (precision == mtx_double) {
                *mpierrcode = MPI_Send(
                    &data->vector_coordinate_real_double[offset],
                    size, datatype, dest, tag, comm);
                if (*mpierrcode) {
                    MPI_Type_free(&datatype);
                    return MTX_ERR_MPI;
                }
            } else {
                MPI_Type_free(&datatype);
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_complex) {
            if (precision == mtx_single) {
                *mpierrcode = MPI_Send(
                    &data->vector_coordinate_complex_single[offset],
                    size, datatype, dest, tag, comm);
                if (*mpierrcode) {
                    MPI_Type_free(&datatype);
                    return MTX_ERR_MPI;
                }
            } else if (precision == mtx_double) {
                *mpierrcode = MPI_Send(
                    &data->vector_coordinate_complex_double[offset],
                    size, datatype, dest, tag, comm);
                if (*mpierrcode) {
                    MPI_Type_free(&datatype);
                    return MTX_ERR_MPI;
                }
            } else {
                MPI_Type_free(&datatype);
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_integer) {
            if (precision == mtx_single) {
                *mpierrcode = MPI_Send(
                    &data->vector_coordinate_integer_single[offset],
                    size, datatype, dest, tag, comm);
                if (*mpierrcode) {
                    MPI_Type_free(&datatype);
                    return MTX_ERR_MPI;
                }
            } else if (precision == mtx_double) {
                *mpierrcode = MPI_Send(
                    &data->vector_coordinate_integer_double[offset],
                    size, datatype, dest, tag, comm);
                if (*mpierrcode) {
                    MPI_Type_free(&datatype);
                    return MTX_ERR_MPI;
                }
            } else {
                MPI_Type_free(&datatype);
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_pattern) {
            *mpierrcode = MPI_Send(
                &data->vector_coordinate_pattern[offset],
                size, datatype, dest, tag, comm);
                if (*mpierrcode) {
                    MPI_Type_free(&datatype);
                    return MTX_ERR_MPI;
                }
        } else {
            MPI_Type_free(&datatype);
            return MTX_ERR_INVALID_MTX_FIELD;
        }
    } else {
        MPI_Type_free(&datatype);
        return MTX_ERR_INVALID_MTX_OBJECT;
    }
    MPI_Type_free(&datatype);
    return MTX_SUCCESS;
}

/**
 * ‘mtxfiledata_send()’ sends Matrix Market data lines to another MPI
 * process.
 *
 * This is analogous to ‘MPI_Send()’ and requires the receiving
 * process to perform a matching call to ‘mtxfiledata_recv()’.
 */
int mtxfiledata_send(
    const union mtxfiledata * data,
    enum mtxfileobject object,
    enum mtxfileformat format,
    enum mtxfilefield field,
    enum mtxprecision precision,
    int64_t size,
    int64_t offset,
    int dest,
    int tag,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    if (format == mtxfile_array) {
        return mtxfiledata_send_array(
            data, field, precision, size, offset,
            dest, tag, comm, &disterr->err);
    } else if (format == mtxfile_coordinate) {
        return mtxfiledata_send_coordinate(
            data, object, field, precision, size, offset,
            dest, tag, comm, &disterr->err);
    } else { return MTX_ERR_INVALID_MTX_FORMAT; }
    return MTX_SUCCESS;
}

static int mtxfiledata_recv_array(
    const union mtxfiledata * data,
    enum mtxfilefield field,
    enum mtxprecision precision,
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
        } else { return MTX_ERR_INVALID_PRECISION; }
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
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_MTX_FIELD; }
    return MTX_SUCCESS;
}

static int mtxfiledata_recv_coordinate(
    const union mtxfiledata * data,
    enum mtxfileobject object,
    enum mtxfilefield field,
    enum mtxprecision precision,
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
                if (*mpierrcode) {
                    MPI_Type_free(&datatype);
                    return MTX_ERR_MPI;
                }
            } else if (precision == mtx_double) {
                *mpierrcode = MPI_Recv(
                    &data->matrix_coordinate_real_double[offset],
                    size, datatype, source, tag, comm, MPI_STATUS_IGNORE);
                if (*mpierrcode) {
                    MPI_Type_free(&datatype);
                    return MTX_ERR_MPI;
                }
            } else {
                MPI_Type_free(&datatype);
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_complex) {
            if (precision == mtx_single) {
                *mpierrcode = MPI_Recv(
                    &data->matrix_coordinate_complex_single[offset],
                    size, datatype, source, tag, comm, MPI_STATUS_IGNORE);
                if (*mpierrcode) {
                    MPI_Type_free(&datatype);
                    return MTX_ERR_MPI;
                }
            } else if (precision == mtx_double) {
                *mpierrcode = MPI_Recv(
                    &data->matrix_coordinate_complex_double[offset],
                    size, datatype, source, tag, comm, MPI_STATUS_IGNORE);
                if (*mpierrcode) {
                    MPI_Type_free(&datatype);
                    return MTX_ERR_MPI;
                }
            } else {
                MPI_Type_free(&datatype);
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_integer) {
            if (precision == mtx_single) {
                *mpierrcode = MPI_Recv(
                    &data->matrix_coordinate_integer_single[offset],
                    size, datatype, source, tag, comm, MPI_STATUS_IGNORE);
                if (*mpierrcode) {
                    MPI_Type_free(&datatype);
                    return MTX_ERR_MPI;
                }
            } else if (precision == mtx_double) {
                *mpierrcode = MPI_Recv(
                    &data->matrix_coordinate_integer_double[offset],
                    size, datatype, source, tag, comm, MPI_STATUS_IGNORE);
                if (*mpierrcode) {
                    MPI_Type_free(&datatype);
                    return MTX_ERR_MPI;
                }
            } else {
                MPI_Type_free(&datatype);
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_pattern) {
            *mpierrcode = MPI_Recv(
                &data->matrix_coordinate_pattern[offset],
                size, datatype, source, tag, comm, MPI_STATUS_IGNORE);
                if (*mpierrcode) {
                    MPI_Type_free(&datatype);
                    return MTX_ERR_MPI;
                }
        } else {
            MPI_Type_free(&datatype);
            return MTX_ERR_INVALID_MTX_FIELD;
        }

    } else if (object == mtxfile_vector) {
        if (field == mtxfile_real) {
            if (precision == mtx_single) {
                *mpierrcode = MPI_Recv(
                    &data->vector_coordinate_real_single[offset],
                    size, datatype, source, tag, comm, MPI_STATUS_IGNORE);
                if (*mpierrcode) {
                    MPI_Type_free(&datatype);
                    return MTX_ERR_MPI;
                }
            } else if (precision == mtx_double) {
                *mpierrcode = MPI_Recv(
                    &data->vector_coordinate_real_double[offset],
                    size, datatype, source, tag, comm, MPI_STATUS_IGNORE);
                if (*mpierrcode) {
                    MPI_Type_free(&datatype);
                    return MTX_ERR_MPI;
                }
            } else {
                MPI_Type_free(&datatype);
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_complex) {
            if (precision == mtx_single) {
                *mpierrcode = MPI_Recv(
                    &data->vector_coordinate_complex_single[offset],
                    size, datatype, source, tag, comm, MPI_STATUS_IGNORE);
                if (*mpierrcode) {
                    MPI_Type_free(&datatype);
                    return MTX_ERR_MPI;
                }
            } else if (precision == mtx_double) {
                *mpierrcode = MPI_Recv(
                    &data->vector_coordinate_complex_double[offset],
                    size, datatype, source, tag, comm, MPI_STATUS_IGNORE);
                if (*mpierrcode) {
                    MPI_Type_free(&datatype);
                    return MTX_ERR_MPI;
                }
            } else {
                MPI_Type_free(&datatype);
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_integer) {
            if (precision == mtx_single) {
                *mpierrcode = MPI_Recv(
                    &data->vector_coordinate_integer_single[offset],
                    size, datatype, source, tag, comm, MPI_STATUS_IGNORE);
                if (*mpierrcode) {
                    MPI_Type_free(&datatype);
                    return MTX_ERR_MPI;
                }
            } else if (precision == mtx_double) {
                *mpierrcode = MPI_Recv(
                    &data->vector_coordinate_integer_double[offset],
                    size, datatype, source, tag, comm, MPI_STATUS_IGNORE);
                if (*mpierrcode) {
                    MPI_Type_free(&datatype);
                    return MTX_ERR_MPI;
                }
            } else {
                MPI_Type_free(&datatype);
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_pattern) {
            *mpierrcode = MPI_Recv(
                &data->vector_coordinate_pattern[offset],
                size, datatype, source, tag, comm, MPI_STATUS_IGNORE);
            if (*mpierrcode) {
                MPI_Type_free(&datatype);
                return MTX_ERR_MPI;
            }
        } else {
            MPI_Type_free(&datatype);
            return MTX_ERR_INVALID_MTX_FIELD;
        }
    } else {
        MPI_Type_free(&datatype);
        return MTX_ERR_INVALID_MTX_OBJECT;
    }
    MPI_Type_free(&datatype);
    return MTX_SUCCESS;
}

/**
 * ‘mtxfiledata_recv()’ receives Matrix Market data lines from
 * another MPI process.
 *
 * This is analogous to ‘MPI_Recv()’ and requires the sending process
 * to perform a matching call to ‘mtxfiledata_send()’.
 */
int mtxfiledata_recv(
    union mtxfiledata * data,
    enum mtxfileobject object,
    enum mtxfileformat format,
    enum mtxfilefield field,
    enum mtxprecision precision,
    int64_t size,
    int64_t offset,
    int source,
    int tag,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    if (format == mtxfile_array) {
        return mtxfiledata_recv_array(
            data, field, precision, size, offset,
            source, tag, comm, &disterr->err);
    } else if (format == mtxfile_coordinate) {
        return mtxfiledata_recv_coordinate(
            data, object, field, precision, size, offset,
            source, tag, comm, &disterr->err);
    } else { return MTX_ERR_INVALID_MTX_FORMAT; }
    return MTX_SUCCESS;
}

static int mtxfiledata_bcast_array(
    const union mtxfiledata * data,
    enum mtxfilefield field,
    enum mtxprecision precision,
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
        } else { return MTX_ERR_INVALID_PRECISION; }
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
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_MTX_FIELD; }
    return MTX_SUCCESS;
}

static int mtxfiledata_bcast_coordinate(
    const union mtxfiledata * data,
    enum mtxfileobject object,
    enum mtxfilefield field,
    enum mtxprecision precision,
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
                if (*mpierrcode) {
                    MPI_Type_free(&datatype);
                    return MTX_ERR_MPI;
                }
            } else if (precision == mtx_double) {
                *mpierrcode = MPI_Bcast(
                    &data->matrix_coordinate_real_double[offset],
                    size, datatype, root, comm);
                if (*mpierrcode) {
                    MPI_Type_free(&datatype);
                    return MTX_ERR_MPI;
                }
            } else {
                MPI_Type_free(&datatype);
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_complex) {
            if (precision == mtx_single) {
                *mpierrcode = MPI_Bcast(
                    &data->matrix_coordinate_complex_single[offset],
                    size, datatype, root, comm);
                if (*mpierrcode) {
                    MPI_Type_free(&datatype);
                    return MTX_ERR_MPI;
                }
            } else if (precision == mtx_double) {
                *mpierrcode = MPI_Bcast(
                    &data->matrix_coordinate_complex_double[offset],
                    size, datatype, root, comm);
                if (*mpierrcode) {
                    MPI_Type_free(&datatype);
                    return MTX_ERR_MPI;
                }
            } else {
                MPI_Type_free(&datatype);
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_integer) {
            if (precision == mtx_single) {
                *mpierrcode = MPI_Bcast(
                    &data->matrix_coordinate_integer_single[offset],
                    size, datatype, root, comm);
                if (*mpierrcode) {
                    MPI_Type_free(&datatype);
                    return MTX_ERR_MPI;
                }
            } else if (precision == mtx_double) {
                *mpierrcode = MPI_Bcast(
                    &data->matrix_coordinate_integer_double[offset],
                    size, datatype, root, comm);
                if (*mpierrcode) {
                    MPI_Type_free(&datatype);
                    return MTX_ERR_MPI;
                }
            } else {
                MPI_Type_free(&datatype);
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_pattern) {
            *mpierrcode = MPI_Bcast(
                &data->matrix_coordinate_pattern[offset],
                size, datatype, root, comm);
                if (*mpierrcode) {
                    MPI_Type_free(&datatype);
                    return MTX_ERR_MPI;
                }
        } else {
            MPI_Type_free(&datatype);
            return MTX_ERR_INVALID_MTX_FIELD;
        }

    } else if (object == mtxfile_vector) {
        if (field == mtxfile_real) {
            if (precision == mtx_single) {
                *mpierrcode = MPI_Bcast(
                    &data->vector_coordinate_real_single[offset],
                    size, datatype, root, comm);
                if (*mpierrcode) {
                    MPI_Type_free(&datatype);
                    return MTX_ERR_MPI;
                }
            } else if (precision == mtx_double) {
                *mpierrcode = MPI_Bcast(
                    &data->vector_coordinate_real_double[offset],
                    size, datatype, root, comm);
                if (*mpierrcode) {
                    MPI_Type_free(&datatype);
                    return MTX_ERR_MPI;
                }
            } else {
                MPI_Type_free(&datatype);
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_complex) {
            if (precision == mtx_single) {
                *mpierrcode = MPI_Bcast(
                    &data->vector_coordinate_complex_single[offset],
                    size, datatype, root, comm);
                if (*mpierrcode) {
                    MPI_Type_free(&datatype);
                    return MTX_ERR_MPI;
                }
            } else if (precision == mtx_double) {
                *mpierrcode = MPI_Bcast(
                    &data->vector_coordinate_complex_double[offset],
                    size, datatype, root, comm);
                if (*mpierrcode) {
                    MPI_Type_free(&datatype);
                    return MTX_ERR_MPI;
                }
            } else {
                MPI_Type_free(&datatype);
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_integer) {
            if (precision == mtx_single) {
                *mpierrcode = MPI_Bcast(
                    &data->vector_coordinate_integer_single[offset],
                    size, datatype, root, comm);
                if (*mpierrcode) {
                    MPI_Type_free(&datatype);
                    return MTX_ERR_MPI;
                }
            } else if (precision == mtx_double) {
                *mpierrcode = MPI_Bcast(
                    &data->vector_coordinate_integer_double[offset],
                    size, datatype, root, comm);
                if (*mpierrcode) {
                    MPI_Type_free(&datatype);
                    return MTX_ERR_MPI;
                }
            } else {
                MPI_Type_free(&datatype);
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_pattern) {
            *mpierrcode = MPI_Bcast(
                &data->vector_coordinate_pattern[offset],
                size, datatype, root, comm);
            if (*mpierrcode) {
                MPI_Type_free(&datatype);
                return MTX_ERR_MPI;
            }
        } else {
            MPI_Type_free(&datatype);
            return MTX_ERR_INVALID_MTX_FIELD;
        }
    } else {
        MPI_Type_free(&datatype);
        return MTX_ERR_INVALID_MTX_OBJECT;
    }
    MPI_Type_free(&datatype);
    return MTX_SUCCESS;
}

/**
 * ‘mtxfiledata_bcast()’ broadcasts Matrix Market data lines from an
 * MPI root process to other processes in a communicator.
 *
 * This is analogous to ‘MPI_Bcast()’ and requires every process in
 * the communicator to perform matching calls to
 * ‘mtxfiledata_bcast()’.
 */
int mtxfiledata_bcast(
    union mtxfiledata * data,
    enum mtxfileobject object,
    enum mtxfileformat format,
    enum mtxfilefield field,
    enum mtxprecision precision,
    int64_t size,
    int64_t offset,
    int root,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    if (format == mtxfile_array) {
        return mtxfiledata_bcast_array(
            data, field, precision, size, offset,
            root, comm, &disterr->err);
    } else if (format == mtxfile_coordinate) {
        return mtxfiledata_bcast_coordinate(
            data, object, field, precision, size, offset,
            root, comm, &disterr->err);
    } else { return MTX_ERR_INVALID_MTX_FORMAT; }
    return MTX_SUCCESS;
}

static int mtxfiledata_gatherv_array(
    const union mtxfiledata * sendbuf,
    enum mtxfilefield field,
    enum mtxprecision precision,
    int64_t sendoffset,
    int sendcount,
    union mtxfiledata * recvbuf,
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
        } else { return MTX_ERR_INVALID_PRECISION; }
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
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_MTX_FIELD; }
    return MTX_SUCCESS;
}

static int mtxfiledata_gatherv_coordinate(
    const union mtxfiledata * sendbuf,
    enum mtxfileobject object,
    enum mtxfilefield field,
    enum mtxprecision precision,
    int64_t sendoffset,
    int sendcount,
    union mtxfiledata * recvbuf,
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
                if (*mpierrcode) {
                    MPI_Type_free(&datatype);
                    return MTX_ERR_MPI;
                }
            } else if (precision == mtx_double) {
                *mpierrcode = MPI_Gatherv(
                    &sendbuf->matrix_coordinate_real_double[sendoffset],
                    sendcount, datatype,
                    &recvbuf->matrix_coordinate_real_double[recvoffset],
                    recvcounts, recvdispls, datatype, root, comm);
                if (*mpierrcode) {
                    MPI_Type_free(&datatype);
                    return MTX_ERR_MPI;
                }
            } else {
                MPI_Type_free(&datatype);
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_complex) {
            if (precision == mtx_single) {
                *mpierrcode = MPI_Gatherv(
                    &sendbuf->matrix_coordinate_complex_single[sendoffset],
                    sendcount, datatype,
                    &recvbuf->matrix_coordinate_complex_single[recvoffset],
                    recvcounts, recvdispls, datatype, root, comm);
                if (*mpierrcode) {
                    MPI_Type_free(&datatype);
                    return MTX_ERR_MPI;
                }
            } else if (precision == mtx_double) {
                *mpierrcode = MPI_Gatherv(
                    &sendbuf->matrix_coordinate_complex_double[sendoffset],
                    sendcount, datatype,
                    &recvbuf->matrix_coordinate_complex_double[recvoffset],
                    recvcounts, recvdispls, datatype, root, comm);
                if (*mpierrcode) {
                    MPI_Type_free(&datatype);
                    return MTX_ERR_MPI;
                }
            } else {
                MPI_Type_free(&datatype);
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_integer) {
            if (precision == mtx_single) {
                *mpierrcode = MPI_Gatherv(
                    &sendbuf->matrix_coordinate_integer_single[sendoffset],
                    sendcount, datatype,
                    &recvbuf->matrix_coordinate_integer_single[recvoffset],
                    recvcounts, recvdispls, datatype, root, comm);
                if (*mpierrcode) {
                    MPI_Type_free(&datatype);
                    return MTX_ERR_MPI;
                }
            } else if (precision == mtx_double) {
                *mpierrcode = MPI_Gatherv(
                    &sendbuf->matrix_coordinate_integer_double[sendoffset],
                    sendcount, datatype,
                    &recvbuf->matrix_coordinate_integer_double[recvoffset],
                    recvcounts, recvdispls, datatype, root, comm);
                if (*mpierrcode) {
                    MPI_Type_free(&datatype);
                    return MTX_ERR_MPI;
                }
            } else {
                MPI_Type_free(&datatype);
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_pattern) {
            *mpierrcode = MPI_Gatherv(
                &sendbuf->matrix_coordinate_pattern[sendoffset],
                sendcount, datatype,
                &recvbuf->matrix_coordinate_pattern[recvoffset],
                recvcounts, recvdispls, datatype, root, comm);
            if (*mpierrcode) {
                MPI_Type_free(&datatype);
                return MTX_ERR_MPI;
            }
        } else {
            MPI_Type_free(&datatype);
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
                if (*mpierrcode) {
                    MPI_Type_free(&datatype);
                    return MTX_ERR_MPI;
                }
            } else if (precision == mtx_double) {
                *mpierrcode = MPI_Gatherv(
                    &sendbuf->vector_coordinate_real_double[sendoffset],
                    sendcount, datatype,
                    &recvbuf->vector_coordinate_real_double[recvoffset],
                    recvcounts, recvdispls, datatype, root, comm);
                if (*mpierrcode) {
                    MPI_Type_free(&datatype);
                    return MTX_ERR_MPI;
                }
            } else {
                MPI_Type_free(&datatype);
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_complex) {
            if (precision == mtx_single) {
                *mpierrcode = MPI_Gatherv(
                    &sendbuf->vector_coordinate_complex_single[sendoffset],
                    sendcount, datatype,
                    &recvbuf->vector_coordinate_complex_single[recvoffset],
                    recvcounts, recvdispls, datatype, root, comm);
                if (*mpierrcode) {
                    MPI_Type_free(&datatype);
                    return MTX_ERR_MPI;
                }
            } else if (precision == mtx_double) {
                *mpierrcode = MPI_Gatherv(
                    &sendbuf->vector_coordinate_complex_double[sendoffset],
                    sendcount, datatype,
                    &recvbuf->vector_coordinate_complex_double[recvoffset],
                    recvcounts, recvdispls, datatype, root, comm);
                if (*mpierrcode) {
                    MPI_Type_free(&datatype);
                    return MTX_ERR_MPI;
                }
            } else {
                MPI_Type_free(&datatype);
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_integer) {
            if (precision == mtx_single) {
                *mpierrcode = MPI_Gatherv(
                    &sendbuf->vector_coordinate_integer_single[sendoffset],
                    sendcount, datatype,
                    &recvbuf->vector_coordinate_integer_single[recvoffset],
                    recvcounts, recvdispls, datatype, root, comm);
                if (*mpierrcode) {
                    MPI_Type_free(&datatype);
                    return MTX_ERR_MPI;
                }
            } else if (precision == mtx_double) {
                *mpierrcode = MPI_Gatherv(
                    &sendbuf->vector_coordinate_integer_double[sendoffset],
                    sendcount, datatype,
                    &recvbuf->vector_coordinate_integer_double[recvoffset],
                    recvcounts, recvdispls, datatype, root, comm);
                if (*mpierrcode) {
                    MPI_Type_free(&datatype);
                    return MTX_ERR_MPI;
                }
            } else {
                MPI_Type_free(&datatype);
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_pattern) {
            *mpierrcode = MPI_Gatherv(
                &sendbuf->vector_coordinate_pattern[sendoffset],
                sendcount, datatype,
                &recvbuf->vector_coordinate_pattern[recvoffset],
                recvcounts, recvdispls, datatype, root, comm);
            if (*mpierrcode) {
                MPI_Type_free(&datatype);
                return MTX_ERR_MPI;
            }
        } else {
            MPI_Type_free(&datatype);
            return MTX_ERR_INVALID_MTX_FIELD;
        }
    } else {
        MPI_Type_free(&datatype);
        return MTX_ERR_INVALID_MTX_OBJECT;
    }
    MPI_Type_free(&datatype);
    return MTX_SUCCESS;
}

/**
 * ‘mtxfiledata_gatherv()’ gathers Matrix Market data lines onto an
 * MPI root process from other processes in a communicator.
 *
 * This is analogous to ‘MPI_Gatherv()’ and requires every process in
 * the communicator to perform matching calls to
 * ‘mtxfiledata_gatherv()’.
 */
int mtxfiledata_gatherv(
    const union mtxfiledata * sendbuf,
    enum mtxfileobject object,
    enum mtxfileformat format,
    enum mtxfilefield field,
    enum mtxprecision precision,
    int64_t sendoffset,
    int sendcount,
    union mtxfiledata * recvbuf,
    int64_t recvoffset,
    const int * recvcounts,
    const int * recvdispls,
    int root,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    if (format == mtxfile_array) {
        return mtxfiledata_gatherv_array(
            sendbuf, field, precision, sendoffset, sendcount,
            recvbuf, recvoffset, recvcounts, recvdispls,
            root, comm, &disterr->err);
    } else if (format == mtxfile_coordinate) {
        return mtxfiledata_gatherv_coordinate(
            sendbuf, object, field, precision, sendoffset, sendcount,
            recvbuf, recvoffset, recvcounts, recvdispls,
            root, comm, &disterr->err);
    } else { return MTX_ERR_INVALID_MTX_FORMAT; }
    return MTX_SUCCESS;
}

static int mtxfiledata_scatterv_array(
    const union mtxfiledata * sendbuf,
    enum mtxfilefield field,
    enum mtxprecision precision,
    int64_t sendoffset,
    const int * sendcounts,
    const int * displs,
    union mtxfiledata * recvbuf,
    int64_t recvoffset,
    int recvcount,
    int root,
    MPI_Comm comm,
    int * mpierrcode)
{
    int rank;
    *mpierrcode = MPI_Comm_rank(comm, &rank);
    if (*mpierrcode) return MTX_ERR_MPI;
    if (field == mtxfile_real) {
        if (precision == mtx_single) {
            *mpierrcode = MPI_Scatterv(
                rank != root ? NULL :
                &sendbuf->array_real_single[sendoffset], sendcounts, displs, MPI_FLOAT,
                &recvbuf->array_real_single[recvoffset], recvcount, MPI_FLOAT,
                root, comm);
            if (*mpierrcode)
                return MTX_ERR_MPI;
        } else if (precision == mtx_double) {
            *mpierrcode = MPI_Scatterv(
                rank != root ? NULL :
                &sendbuf->array_real_double[sendoffset], sendcounts, displs, MPI_DOUBLE,
                &recvbuf->array_real_double[recvoffset], recvcount, MPI_DOUBLE,
                root, comm);
            if (*mpierrcode)
                return MTX_ERR_MPI;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (field == mtxfile_complex) {
        MPI_Datatype complex;
        int err = mtxfile_array_complex_datatype(&complex, precision, mpierrcode);
        if (err)
            return err;
        if (precision == mtx_single) {
            *mpierrcode = MPI_Scatterv(
                rank != root ? NULL :
                &sendbuf->array_complex_single[sendoffset], sendcounts, displs, complex,
                &recvbuf->array_complex_single[recvoffset], recvcount, complex,
                root, comm);
            if (*mpierrcode) {
                MPI_Type_free(&complex);
                return MTX_ERR_MPI;
            }
        } else if (precision == mtx_double) {
            *mpierrcode = MPI_Scatterv(
                rank != root ? NULL :
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
                rank != root ? NULL :
                &sendbuf->array_integer_single[sendoffset], sendcounts, displs, MPI_INT32_T,
                &recvbuf->array_integer_single[recvoffset], recvcount, MPI_INT32_T,
                root, comm);
            if (*mpierrcode)
                return MTX_ERR_MPI;
        } else if (precision == mtx_double) {
            *mpierrcode = MPI_Scatterv(
                rank != root ? NULL :
                &sendbuf->array_integer_double[sendoffset], sendcounts, displs, MPI_INT64_T,
                &recvbuf->array_integer_double[recvoffset], recvcount, MPI_INT64_T,
                root, comm);
            if (*mpierrcode)
                return MTX_ERR_MPI;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_MTX_FIELD; }
    return MTX_SUCCESS;
}

static int mtxfiledata_scatterv_coordinate(
    const union mtxfiledata * sendbuf,
    enum mtxfileobject object,
    enum mtxfilefield field,
    enum mtxprecision precision,
    int64_t sendoffset,
    const int * sendcounts,
    const int * displs,
    union mtxfiledata * recvbuf,
    int64_t recvoffset,
    int recvcount,
    int root,
    MPI_Comm comm,
    int * mpierrcode)
{
    int rank;
    *mpierrcode = MPI_Comm_rank(comm, &rank);
    if (*mpierrcode) return MTX_ERR_MPI;
    MPI_Datatype datatype;
    int err = mtxfile_coordinate_datatype(
        object, field, precision, &datatype, mpierrcode);
    if (err)
        return err;

    if (object == mtxfile_matrix) {
        if (field == mtxfile_real) {
            if (precision == mtx_single) {
                *mpierrcode = MPI_Scatterv(
                    rank != root ? NULL :
                    &sendbuf->matrix_coordinate_real_single[sendoffset],
                    sendcounts, displs, datatype,
                    &recvbuf->matrix_coordinate_real_single[recvoffset],
                    recvcount, datatype, root, comm);
                if (*mpierrcode) {
                    MPI_Type_free(&datatype);
                    return MTX_ERR_MPI;
                }
            } else if (precision == mtx_double) {
                *mpierrcode = MPI_Scatterv(
                    rank != root ? NULL :
                    &sendbuf->matrix_coordinate_real_double[sendoffset],
                    sendcounts, displs, datatype,
                    &recvbuf->matrix_coordinate_real_double[recvoffset],
                    recvcount, datatype, root, comm);
                if (*mpierrcode) {
                    MPI_Type_free(&datatype);
                    return MTX_ERR_MPI;
                }
            } else {
                MPI_Type_free(&datatype);
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_complex) {
            if (precision == mtx_single) {
                *mpierrcode = MPI_Scatterv(
                    rank != root ? NULL :
                    &sendbuf->matrix_coordinate_complex_single[sendoffset],
                    sendcounts, displs, datatype,
                    &recvbuf->matrix_coordinate_complex_single[recvoffset],
                    recvcount, datatype, root, comm);
                if (*mpierrcode) {
                    MPI_Type_free(&datatype);
                    return MTX_ERR_MPI;
                }
            } else if (precision == mtx_double) {
                *mpierrcode = MPI_Scatterv(
                    rank != root ? NULL :
                    &sendbuf->matrix_coordinate_complex_double[sendoffset],
                    sendcounts, displs, datatype,
                    &recvbuf->matrix_coordinate_complex_double[recvoffset],
                    recvcount, datatype, root, comm);
                if (*mpierrcode) {
                    MPI_Type_free(&datatype);
                    return MTX_ERR_MPI;
                }
            } else {
                MPI_Type_free(&datatype);
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_integer) {
            if (precision == mtx_single) {
                *mpierrcode = MPI_Scatterv(
                    rank != root ? NULL :
                    &sendbuf->matrix_coordinate_integer_single[sendoffset],
                    sendcounts, displs, datatype,
                    &recvbuf->matrix_coordinate_integer_single[recvoffset],
                    recvcount, datatype, root, comm);
                if (*mpierrcode) {
                    MPI_Type_free(&datatype);
                    return MTX_ERR_MPI;
                }
            } else if (precision == mtx_double) {
                *mpierrcode = MPI_Scatterv(
                    rank != root ? NULL :
                    &sendbuf->matrix_coordinate_integer_double[sendoffset],
                    sendcounts, displs, datatype,
                    &recvbuf->matrix_coordinate_integer_double[recvoffset],
                    recvcount, datatype, root, comm);
                if (*mpierrcode) {
                    MPI_Type_free(&datatype);
                    return MTX_ERR_MPI;
                }
            } else {
                MPI_Type_free(&datatype);
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_pattern) {
            *mpierrcode = MPI_Scatterv(
                rank != root ? NULL :
                &sendbuf->matrix_coordinate_pattern[sendoffset],
                sendcounts, displs, datatype,
                &recvbuf->matrix_coordinate_pattern[recvoffset],
                recvcount, datatype, root, comm);
                if (*mpierrcode) {
                    MPI_Type_free(&datatype);
                    return MTX_ERR_MPI;
                }
        } else {
            MPI_Type_free(&datatype);
            return MTX_ERR_INVALID_MTX_FIELD;
        }

    } else if (object == mtxfile_vector) {
        if (field == mtxfile_real) {
            if (precision == mtx_single) {
                *mpierrcode = MPI_Scatterv(
                    rank != root ? NULL :
                    &sendbuf->vector_coordinate_real_single[sendoffset],
                    sendcounts, displs, datatype,
                    &recvbuf->vector_coordinate_real_single[recvoffset],
                    recvcount, datatype, root, comm);
                if (*mpierrcode) {
                    MPI_Type_free(&datatype);
                    return MTX_ERR_MPI;
                }
            } else if (precision == mtx_double) {
                *mpierrcode = MPI_Scatterv(
                    rank != root ? NULL :
                    &sendbuf->vector_coordinate_real_double[sendoffset],
                    sendcounts, displs, datatype,
                    &recvbuf->vector_coordinate_real_double[recvoffset],
                    recvcount, datatype, root, comm);
                if (*mpierrcode) {
                    MPI_Type_free(&datatype);
                    return MTX_ERR_MPI;
                }
            } else {
                MPI_Type_free(&datatype);
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_complex) {
            if (precision == mtx_single) {
                *mpierrcode = MPI_Scatterv(
                    rank != root ? NULL :
                    &sendbuf->vector_coordinate_complex_single[sendoffset],
                    sendcounts, displs, datatype,
                    &recvbuf->vector_coordinate_complex_single[recvoffset],
                    recvcount, datatype, root, comm);
                if (*mpierrcode) {
                    MPI_Type_free(&datatype);
                    return MTX_ERR_MPI;
                }
            } else if (precision == mtx_double) {
                *mpierrcode = MPI_Scatterv(
                    rank != root ? NULL :
                    &sendbuf->vector_coordinate_complex_double[sendoffset],
                    sendcounts, displs, datatype,
                    &recvbuf->vector_coordinate_complex_double[recvoffset],
                    recvcount, datatype, root, comm);
                if (*mpierrcode) {
                    MPI_Type_free(&datatype);
                    return MTX_ERR_MPI;
                }
            } else {
                MPI_Type_free(&datatype);
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_integer) {
            if (precision == mtx_single) {
                *mpierrcode = MPI_Scatterv(
                    rank != root ? NULL :
                    &sendbuf->vector_coordinate_integer_single[sendoffset],
                    sendcounts, displs, datatype,
                    &recvbuf->vector_coordinate_integer_single[recvoffset],
                    recvcount, datatype, root, comm);
                if (*mpierrcode) {
                    MPI_Type_free(&datatype);
                    return MTX_ERR_MPI;
                }
            } else if (precision == mtx_double) {
                *mpierrcode = MPI_Scatterv(
                    rank != root ? NULL :
                    &sendbuf->vector_coordinate_integer_double[sendoffset],
                    sendcounts, displs, datatype,
                    &recvbuf->vector_coordinate_integer_double[recvoffset],
                    recvcount, datatype, root, comm);
                if (*mpierrcode) {
                    MPI_Type_free(&datatype);
                    return MTX_ERR_MPI;
                }
            } else {
                MPI_Type_free(&datatype);
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_pattern) {
            *mpierrcode = MPI_Scatterv(
                rank != root ? NULL :
                &sendbuf->vector_coordinate_pattern[sendoffset],
                sendcounts, displs, datatype,
                &recvbuf->vector_coordinate_pattern[recvoffset],
                recvcount, datatype, root, comm);
            if (*mpierrcode) {
                MPI_Type_free(&datatype);
                return MTX_ERR_MPI;
            }
        } else {
            MPI_Type_free(&datatype);
            return MTX_ERR_INVALID_MTX_FIELD;
        }
    } else {
        MPI_Type_free(&datatype);
        return MTX_ERR_INVALID_MTX_OBJECT;
    }
    MPI_Type_free(&datatype);
    return MTX_SUCCESS;
}

/**
 * ‘mtxfiledata_scatterv()’ scatters Matrix Market data lines from an
 * MPI root process to other processes in a communicator.
 *
 * This is analogous to ‘MPI_Scatterv()’ and requires every process in
 * the communicator to perform matching calls to
 * ‘mtxfiledata_scatterv()’.
 */
int mtxfiledata_scatterv(
    const union mtxfiledata * sendbuf,
    enum mtxfileobject object,
    enum mtxfileformat format,
    enum mtxfilefield field,
    enum mtxprecision precision,
    int64_t sendoffset,
    const int * sendcounts,
    const int * displs,
    union mtxfiledata * recvbuf,
    int64_t recvoffset,
    int recvcount,
    int root,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    if (format == mtxfile_array) {
        return mtxfiledata_scatterv_array(
            sendbuf, field, precision, sendoffset, sendcounts, displs,
            recvbuf, recvoffset, recvcount, root, comm, &disterr->err);
    } else if (format == mtxfile_coordinate) {
        return mtxfiledata_scatterv_coordinate(
            sendbuf, object, field, precision, sendoffset, sendcounts, displs,
            recvbuf, recvoffset, recvcount, root, comm, &disterr->err);
    } else { return MTX_ERR_INVALID_MTX_FORMAT; }
    return MTX_SUCCESS;
}

static int mtxfiledata_alltoallv_array(
    const union mtxfiledata * sendbuf,
    enum mtxfilefield field,
    enum mtxprecision precision,
    int64_t sendoffset,
    const int * sendcounts,
    const int * senddispls,
    union mtxfiledata * recvbuf,
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
        } else { return MTX_ERR_INVALID_PRECISION; }
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
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_MTX_FIELD; }
    return MTX_SUCCESS;
}

static int mtxfiledata_alltoallv_coordinate(
    const union mtxfiledata * sendbuf,
    enum mtxfileobject object,
    enum mtxfilefield field,
    enum mtxprecision precision,
    int64_t sendoffset,
    const int * sendcounts,
    const int * senddispls,
    union mtxfiledata * recvbuf,
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
                if (*mpierrcode) {
                    MPI_Type_free(&datatype);
                    return MTX_ERR_MPI;
                }
            } else if (precision == mtx_double) {
                *mpierrcode = MPI_Alltoallv(
                    &sendbuf->matrix_coordinate_real_double[sendoffset],
                    sendcounts, senddispls, datatype,
                    &recvbuf->matrix_coordinate_real_double[recvoffset],
                    recvcounts, recvdispls, datatype, comm);
                if (*mpierrcode) {
                    MPI_Type_free(&datatype);
                    return MTX_ERR_MPI;
                }
            } else {
                MPI_Type_free(&datatype);
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_complex) {
            if (precision == mtx_single) {
                *mpierrcode = MPI_Alltoallv(
                    &sendbuf->matrix_coordinate_complex_single[sendoffset],
                    sendcounts, senddispls, datatype,
                    &recvbuf->matrix_coordinate_complex_single[recvoffset],
                    recvcounts, recvdispls, datatype, comm);
                if (*mpierrcode) {
                    MPI_Type_free(&datatype);
                    return MTX_ERR_MPI;
                }
            } else if (precision == mtx_double) {
                *mpierrcode = MPI_Alltoallv(
                    &sendbuf->matrix_coordinate_complex_double[sendoffset],
                    sendcounts, senddispls, datatype,
                    &recvbuf->matrix_coordinate_complex_double[recvoffset],
                    recvcounts, recvdispls, datatype, comm);
                if (*mpierrcode) {
                    MPI_Type_free(&datatype);
                    return MTX_ERR_MPI;
                }
            } else {
                MPI_Type_free(&datatype);
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_integer) {
            if (precision == mtx_single) {
                *mpierrcode = MPI_Alltoallv(
                    &sendbuf->matrix_coordinate_integer_single[sendoffset],
                    sendcounts, senddispls, datatype,
                    &recvbuf->matrix_coordinate_integer_single[recvoffset],
                    recvcounts, recvdispls, datatype, comm);
                if (*mpierrcode) {
                    MPI_Type_free(&datatype);
                    return MTX_ERR_MPI;
                }
            } else if (precision == mtx_double) {
                *mpierrcode = MPI_Alltoallv(
                    &sendbuf->matrix_coordinate_integer_double[sendoffset],
                    sendcounts, senddispls, datatype,
                    &recvbuf->matrix_coordinate_integer_double[recvoffset],
                    recvcounts, recvdispls, datatype, comm);
                if (*mpierrcode) {
                    MPI_Type_free(&datatype);
                    return MTX_ERR_MPI;
                }
            } else {
                MPI_Type_free(&datatype);
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_pattern) {
            *mpierrcode = MPI_Alltoallv(
                &sendbuf->matrix_coordinate_pattern[sendoffset],
                sendcounts, senddispls, datatype,
                &recvbuf->matrix_coordinate_pattern[recvoffset],
                recvcounts, recvdispls, datatype, comm);
            if (*mpierrcode) {
                MPI_Type_free(&datatype);
                return MTX_ERR_MPI;
            }
        } else {
            MPI_Type_free(&datatype);
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
                if (*mpierrcode) {
                    MPI_Type_free(&datatype);
                    return MTX_ERR_MPI;
                }
            } else if (precision == mtx_double) {
                *mpierrcode = MPI_Alltoallv(
                    &sendbuf->vector_coordinate_real_double[sendoffset],
                    sendcounts, senddispls, datatype,
                    &recvbuf->vector_coordinate_real_double[recvoffset],
                    recvcounts, recvdispls, datatype, comm);
                if (*mpierrcode) {
                    MPI_Type_free(&datatype);
                    return MTX_ERR_MPI;
                }
            } else {
                MPI_Type_free(&datatype);
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_complex) {
            if (precision == mtx_single) {
                *mpierrcode = MPI_Alltoallv(
                    &sendbuf->vector_coordinate_complex_single[sendoffset],
                    sendcounts, senddispls, datatype,
                    &recvbuf->vector_coordinate_complex_single[recvoffset],
                    recvcounts, recvdispls, datatype, comm);
                if (*mpierrcode) {
                    MPI_Type_free(&datatype);
                    return MTX_ERR_MPI;
                }
            } else if (precision == mtx_double) {
                *mpierrcode = MPI_Alltoallv(
                    &sendbuf->vector_coordinate_complex_double[sendoffset],
                    sendcounts, senddispls, datatype,
                    &recvbuf->vector_coordinate_complex_double[recvoffset],
                    recvcounts, recvdispls, datatype, comm);
                if (*mpierrcode) {
                    MPI_Type_free(&datatype);
                    return MTX_ERR_MPI;
                }
            } else {
                MPI_Type_free(&datatype);
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_integer) {
            if (precision == mtx_single) {
                *mpierrcode = MPI_Alltoallv(
                    &sendbuf->vector_coordinate_integer_single[sendoffset],
                    sendcounts, senddispls, datatype,
                    &recvbuf->vector_coordinate_integer_single[recvoffset],
                    recvcounts, recvdispls, datatype, comm);
                if (*mpierrcode) {
                    MPI_Type_free(&datatype);
                    return MTX_ERR_MPI;
                }
            } else if (precision == mtx_double) {
                *mpierrcode = MPI_Alltoallv(
                    &sendbuf->vector_coordinate_integer_double[sendoffset],
                    sendcounts, senddispls, datatype,
                    &recvbuf->vector_coordinate_integer_double[recvoffset],
                    recvcounts, recvdispls, datatype, comm);
                if (*mpierrcode) {
                    MPI_Type_free(&datatype);
                    return MTX_ERR_MPI;
                }
            } else {
                MPI_Type_free(&datatype);
                return MTX_ERR_INVALID_PRECISION;
            }
        } else if (field == mtxfile_pattern) {
            *mpierrcode = MPI_Alltoallv(
                &sendbuf->vector_coordinate_pattern[sendoffset],
                sendcounts, senddispls, datatype,
                &recvbuf->vector_coordinate_pattern[recvoffset],
                recvcounts, recvdispls, datatype, comm);
            if (*mpierrcode) {
                MPI_Type_free(&datatype);
                return MTX_ERR_MPI;
            }
        } else {
            MPI_Type_free(&datatype);
            return MTX_ERR_INVALID_MTX_FIELD;
        }
    } else {
        MPI_Type_free(&datatype);
        return MTX_ERR_INVALID_MTX_OBJECT;
    }
    MPI_Type_free(&datatype);
    return MTX_SUCCESS;
}

/**
 * ‘mtxfiledata_alltoallv()’ performs an all-to-all exchange of
 * Matrix Market data lines between MPI processes in a communicator.
 *
 * This is analogous to ‘MPI_Alltoallv()’ and requires every process
 * in the communicator to perform matching calls to
 * ‘mtxfiledata_alltoallv()’.
 */
int mtxfiledata_alltoallv(
    const union mtxfiledata * sendbuf,
    enum mtxfileobject object,
    enum mtxfileformat format,
    enum mtxfilefield field,
    enum mtxprecision precision,
    int64_t sendoffset,
    const int * sendcounts,
    const int * senddispls,
    union mtxfiledata * recvbuf,
    int64_t recvoffset,
    const int * recvcounts,
    const int * recvdispls,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    if (format == mtxfile_array) {
        return mtxfiledata_alltoallv_array(
            sendbuf, field, precision, sendoffset, sendcounts, senddispls,
            recvbuf, recvoffset, recvcounts, recvdispls,
            comm, &disterr->err);
    } else if (format == mtxfile_coordinate) {
        return mtxfiledata_alltoallv_coordinate(
            sendbuf, object, field, precision, sendoffset, sendcounts, senddispls,
            recvbuf, recvoffset, recvcounts, recvdispls,
            comm, &disterr->err);
    } else { return MTX_ERR_INVALID_MTX_FORMAT; }
    return MTX_SUCCESS;
}
#endif
