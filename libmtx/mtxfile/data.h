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

#ifndef LIBMTX_MTXFILE_DATA_H
#define LIBMTX_MTXFILE_DATA_H

#include <libmtx/libmtx-config.h>

#include <libmtx/mtxfile/header.h>
#include <libmtx/mtx/precision.h>

#ifdef LIBMTX_HAVE_MPI
#include <mpi.h>
#endif

#ifdef LIBMTX_HAVE_LIBZ
#include <zlib.h>
#endif

#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>

struct mtxmpierror;
struct mtx_partition;

/*
 * Matrix coordinate formats
 */

/**
 * `mtxfile_matrix_coordinate_real_single' represents a nonzero matrix
 * entry in a Matrix Market file with `matrix' object, `coordinate'
 * format and `real' field, when using single precision data types.
 */
struct mtxfile_matrix_coordinate_real_single
{
    int i;    /* row index */
    int j;    /* column index */
    float a;  /* nonzero value */
};

/**
 * `mtxfile_matrix_coordinate_double' represents a nonzero matrix
 * entry in a Matrix Market file with `matrix' object, `coordinate'
 * format and `real' field, when using double precision data types.
 */
struct mtxfile_matrix_coordinate_real_double
{
    int i;     /* row index */
    int j;     /* column index */
    double a;  /* nonzero value */
};

/**
 * `mtxfile_matrix_coordinate_complex_single' represents a nonzero
 * matrix entry in a Matrix Market file with `matrix' object,
 * `coordinate' format and `complex' field, when using single
 * precision data types.
 */
struct mtxfile_matrix_coordinate_complex_single
{
    int i;       /* row index */
    int j;       /* column index */
    float a[2];  /* real and imaginary parts of nonzero value */
};

/**
 * `mtxfile_matrix_coordinate_complex_double' represents a nonzero
 * matrix entry in a Matrix Market file with `matrix' object,
 * `coordinate' format and `complex' field, when using double
 * precision data types.
 */
struct mtxfile_matrix_coordinate_complex_double
{
    int i;       /* row index */
    int j;       /* column index */
    double a[2];  /* real and imaginary parts of nonzero value */
};

/**
 * `mtxfile_matrix_coordinate_integer_single' represents a nonzero
 * matrix entry in a Matrix Market file with `matrix' object,
 * `coordinate' format and `integer' field, when using single
 * precision data types.
 */
struct mtxfile_matrix_coordinate_integer_single
{
    int i;      /* row index */
    int j;      /* column index */
    int32_t a;  /* nonzero value */
};

/**
 * `mtxfile_matrix_coordinate_integer_double' represents a nonzero
 * matrix entry in a Matrix Market file with `matrix' object,
 * `coordinate' format and `integer' field, when using double
 * precision data types.
 */
struct mtxfile_matrix_coordinate_integer_double
{
    int i;      /* row index */
    int j;      /* column index */
    int64_t a;  /* nonzero value */
};

/**
 * `mtxfile_matrix_coordinate_pattern' represents a nonzero matrix
 * entry in a Matrix Market file with `matrix' object, `coordinate'
 * format and `pattern' field.
 */
struct mtxfile_matrix_coordinate_pattern
{
    int i;  /* row index */
    int j;  /* column index */
};

/*
 * Vector coordinate formats
 */

/**
 * `mtxfile_vector_coordinate_real_single' represents a nonzero vector
 * entry in a Matrix Market file with `vector' object, `coordinate'
 * format and `real' field, when using single precision data types.
 */
struct mtxfile_vector_coordinate_real_single
{
    int i;    /* row index */
    float a;  /* nonzero value */
};

/**
 * `mtxfile_vector_coordinate_double' represents a nonzero vector
 * entry in a Matrix Market file with `vector' object, `coordinate'
 * format and `real' field, when using double precision data types.
 */
struct mtxfile_vector_coordinate_real_double
{
    int i;    /* row index */
    double a; /* nonzero value */
};

/**
 * `mtxfile_vector_coordinate_complex_single' represents a nonzero
 * vector entry in a Matrix Market file with `vector' object,
 * `coordinate' format and `complex' field, when using single
 * precision data types.
 */
struct mtxfile_vector_coordinate_complex_single
{
    int i;        /* row index */
    float a[2];   /* real and imaginary parts of nonzero value */
};

/**
 * `mtxfile_vector_coordinate_complex_double' represents a nonzero
 * vector entry in a Matrix Market file with `vector' object,
 * `coordinate' format and `complex' field, when using double
 * precision data types.
 */
struct mtxfile_vector_coordinate_complex_double
{
    int i;        /* row index */
    double a[2];  /* real and imaginary parts of nonzero value */
};

/**
 * `mtxfile_vector_coordinate_integer_single' represents a nonzero
 * vector entry in a Matrix Market file with `vector' object,
 * `coordinate' format and `integer' field, when using single
 * precision data types.
 */
struct mtxfile_vector_coordinate_integer_single
{
    int i;      /* row index */
    int32_t a;  /* nonzero value */
};

/**
 * `mtxfile_vector_coordinate_integer_double' represents a nonzero
 * vector entry in a Matrix Market file with `vector' object,
 * `coordinate' format and `integer' field, when using double
 * precision data types.
 */
struct mtxfile_vector_coordinate_integer_double
{
    int i;      /* row index */
    int64_t a;  /* nonzero value */
};

/**
 * `mtxfile_vector_coordinate_pattern' represents a nonzero vector
 * entry in a Matrix Market file with `vector' object, `coordinate'
 * format and `pattern' field.
 */
struct mtxfile_vector_coordinate_pattern
{
    int i; /* row index */
};

/*
 * Data structures for Matrix Market data lines.
 */

/**
 * `mtxfile_data' represents an array of data lines from a Matrix
 * Market file.
 */
union mtxfile_data
{
    /* Array formats */
    float * array_real_single;
    double * array_real_double;
    float (* array_complex_single)[2];
    double (* array_complex_double)[2];
    int32_t * array_integer_single;
    int64_t * array_integer_double;

    /* Matrix coordinate formats */
    struct mtxfile_matrix_coordinate_real_single * matrix_coordinate_real_single;
    struct mtxfile_matrix_coordinate_real_double * matrix_coordinate_real_double;
    struct mtxfile_matrix_coordinate_complex_single * matrix_coordinate_complex_single;
    struct mtxfile_matrix_coordinate_complex_double * matrix_coordinate_complex_double;
    struct mtxfile_matrix_coordinate_integer_single * matrix_coordinate_integer_single;
    struct mtxfile_matrix_coordinate_integer_double * matrix_coordinate_integer_double;
    struct mtxfile_matrix_coordinate_pattern * matrix_coordinate_pattern;

    /* Vector coordinate formats */
    struct mtxfile_vector_coordinate_real_single * vector_coordinate_real_single;
    struct mtxfile_vector_coordinate_real_double * vector_coordinate_real_double;
    struct mtxfile_vector_coordinate_complex_single * vector_coordinate_complex_single;
    struct mtxfile_vector_coordinate_complex_double * vector_coordinate_complex_double;
    struct mtxfile_vector_coordinate_integer_single * vector_coordinate_integer_single;
    struct mtxfile_vector_coordinate_integer_double * vector_coordinate_integer_double;
    struct mtxfile_vector_coordinate_pattern * vector_coordinate_pattern;
};

/**
 * `mtxfile_data_size_per_element()' calculates the size of each
 * element in an array of Matrix Market data corresponding to the
 * given `object', `format', `field' and `precision'.
 */
int mtxfile_data_size_per_element(
    size_t * size_per_element,
    enum mtxfile_object object,
    enum mtxfile_format format,
    enum mtxfile_field field,
    enum mtx_precision precision);

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
    const char * s);

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
    const char * s);

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
    const char * s);

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
    const char * s);

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
    const char * s);

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
    const char * s);

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
    int num_columns);

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
    int num_columns);

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
    int num_columns);

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
    int num_columns);

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
    int num_columns);

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
    int num_columns);

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
    int num_columns);

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
    int num_rows);

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
    int num_rows);

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
    int num_rows);

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
    int num_rows);

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
    int num_rows);

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
    int num_rows);

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
    int num_rows);

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
    int64_t size);

/**
 * `mtxfile_data_free()' frees allocaed storage for data lines.
 */
int mtxfile_data_free(
    union mtxfile_data * data,
    enum mtxfile_object object,
    enum mtxfile_format format,
    enum mtxfile_field field,
    enum mtx_precision precision);

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
    int64_t src_offset);

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
    const int64_t * srcdispls);

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
    float a);

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

/**
 * `mtxfile_fread_data()' reads Matrix Market data lines from a
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
    int64_t offset);

#ifdef LIBMTX_HAVE_LIBZ
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
    int64_t offset);
#endif

/**
 * `mtxfile_data_fwrite()' writes data lines of a Matrix Market file
 * to a stream.
 *
 * If `fmt' is `NULL', then the format specifier '%d' is used to print
 * integers and '%f' is used to print floating point
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
    int64_t * bytes_written);

#ifdef LIBMTX_HAVE_LIBZ
/**
 * `mtxfile_data_gzwrite()' writes data lines of a Matrix Market file
 * to a gzip-compressed stream.
 *
 * If `fmt' is `NULL', then the format specifier '%d' is used to print
 * integers and '%f' is used to print floating point
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
    int64_t * bytes_written);
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
    int64_t size);

/*
 * Sorting
 */

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
    int64_t size);

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
    int64_t size);

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
    int64_t size,
    int64_t offset,
    int * keys);

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
    int64_t * data_lines_per_part_ptr);

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
    int * row_parts);

/**
 * `mtxfile_data_partition_columns()' partitions data lines according
 * to a given column partitioning.
 *
 * The array `column_parts' must contain enough storage for an array
 * of `size' values of type `int'.  If successful, the `k'-th value of
 * `column_parts' is equal to the part to which the `k'-th data line
 * belongs.
 */
int mtxfile_data_partition_columns(
    const union mtxfile_data * data,
    enum mtxfile_object object,
    enum mtxfile_format format,
    enum mtxfile_field field,
    enum mtx_precision precision,
    int num_rows,
    int num_columns,
    int64_t size,
    int64_t offset,
    const struct mtx_partition * column_partition,
    int * column_parts);

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
    const int * row_permutation,
    int num_columns,
    const int * column_permutation);

/*
 * MPI functions
 */

#ifdef LIBMTX_HAVE_MPI
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
    struct mtxmpierror * mpierror);

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
    struct mtxmpierror * mpierror);

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
    struct mtxmpierror * mpierror);

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
    struct mtxmpierror * mpierror);

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
    struct mtxmpierror * mpierror);

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
    struct mtxmpierror * mpierror);
#endif

#endif
