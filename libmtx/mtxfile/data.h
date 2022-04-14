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

#ifndef LIBMTX_MTXFILE_DATA_H
#define LIBMTX_MTXFILE_DATA_H

#include <libmtx/libmtx-config.h>

#include <libmtx/mtxfile/header.h>
#include <libmtx/precision.h>

#ifdef LIBMTX_HAVE_MPI
#include <mpi.h>
#endif

#ifdef LIBMTX_HAVE_LIBZ
#include <zlib.h>
#endif

#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>

struct mtxdisterror;
struct mtxpartition;

/*
 * Matrix coordinate formats
 */

/**
 * ‘mtxfile_matrix_coordinate_real_single’ represents a nonzero matrix
 * entry in a Matrix Market file with ‘matrix’ object, ‘coordinate’
 * format and ‘real’ field, when using single precision data types.
 */
struct mtxfile_matrix_coordinate_real_single
{
    int64_t i; /* row index */
    int64_t j; /* column index */
    float a;   /* nonzero value */
};

/**
 * ‘mtxfile_matrix_coordinate_double’ represents a nonzero matrix
 * entry in a Matrix Market file with ‘matrix’ object, ‘coordinate’
 * format and ‘real’ field, when using double precision data types.
 */
struct mtxfile_matrix_coordinate_real_double
{
    int64_t i; /* row index */
    int64_t j; /* column index */
    double a;  /* nonzero value */
};

/**
 * ‘mtxfile_matrix_coordinate_complex_single’ represents a nonzero
 * matrix entry in a Matrix Market file with ‘matrix’ object,
 * ‘coordinate’ format and ‘complex’ field, when using single
 * precision data types.
 */
struct mtxfile_matrix_coordinate_complex_single
{
    int64_t i;   /* row index */
    int64_t j;   /* column index */
    float a[2];  /* real and imaginary parts of nonzero value */
};

/**
 * ‘mtxfile_matrix_coordinate_complex_double’ represents a nonzero
 * matrix entry in a Matrix Market file with ‘matrix’ object,
 * ‘coordinate’ format and ‘complex’ field, when using double
 * precision data types.
 */
struct mtxfile_matrix_coordinate_complex_double
{
    int64_t i;    /* row index */
    int64_t j;    /* column index */
    double a[2];  /* real and imaginary parts of nonzero value */
};

/**
 * ‘mtxfile_matrix_coordinate_integer_single’ represents a nonzero
 * matrix entry in a Matrix Market file with ‘matrix’ object,
 * ‘coordinate’ format and ‘integer’ field, when using single
 * precision data types.
 */
struct mtxfile_matrix_coordinate_integer_single
{
    int64_t i;  /* row index */
    int64_t j;  /* column index */
    int32_t a;  /* nonzero value */
};

/**
 * ‘mtxfile_matrix_coordinate_integer_double’ represents a nonzero
 * matrix entry in a Matrix Market file with ‘matrix’ object,
 * ‘coordinate’ format and ‘integer’ field, when using double
 * precision data types.
 */
struct mtxfile_matrix_coordinate_integer_double
{
    int64_t i;  /* row index */
    int64_t j;  /* column index */
    int64_t a;  /* nonzero value */
};

/**
 * ‘mtxfile_matrix_coordinate_pattern’ represents a nonzero matrix
 * entry in a Matrix Market file with ‘matrix’ object, ‘coordinate’
 * format and ‘pattern’ field.
 */
struct mtxfile_matrix_coordinate_pattern
{
    int64_t i;  /* row index */
    int64_t j;  /* column index */
};

/*
 * Vector coordinate formats
 */

/**
 * ‘mtxfile_vector_coordinate_real_single’ represents a nonzero vector
 * entry in a Matrix Market file with ‘vector’ object, ‘coordinate’
 * format and ‘real’ field, when using single precision data types.
 */
struct mtxfile_vector_coordinate_real_single
{
    int64_t i;  /* offset */
    float a;    /* nonzero value */
};

/**
 * ‘mtxfile_vector_coordinate_double’ represents a nonzero vector
 * entry in a Matrix Market file with ‘vector’ object, ‘coordinate’
 * format and ‘real’ field, when using double precision data types.
 */
struct mtxfile_vector_coordinate_real_double
{
    int64_t i;  /* offset */
    double a;   /* nonzero value */
};

/**
 * ‘mtxfile_vector_coordinate_complex_single’ represents a nonzero
 * vector entry in a Matrix Market file with ‘vector’ object,
 * ‘coordinate’ format and ‘complex’ field, when using single
 * precision data types.
 */
struct mtxfile_vector_coordinate_complex_single
{
    int64_t i;   /* offset */
    float a[2];  /* real and imaginary parts of nonzero value */
};

/**
 * ‘mtxfile_vector_coordinate_complex_double’ represents a nonzero
 * vector entry in a Matrix Market file with ‘vector’ object,
 * ‘coordinate’ format and ‘complex’ field, when using double
 * precision data types.
 */
struct mtxfile_vector_coordinate_complex_double
{
    int64_t i;   /* offset */
    double a[2]; /* real and imaginary parts of nonzero value */
};

/**
 * ‘mtxfile_vector_coordinate_integer_single’ represents a nonzero
 * vector entry in a Matrix Market file with ‘vector’ object,
 * ‘coordinate’ format and ‘integer’ field, when using single
 * precision data types.
 */
struct mtxfile_vector_coordinate_integer_single
{
    int64_t i;   /* offset */
    int32_t a;   /* nonzero value */
};

/**
 * ‘mtxfile_vector_coordinate_integer_double’ represents a nonzero
 * vector entry in a Matrix Market file with ‘vector’ object,
 * ‘coordinate’ format and ‘integer’ field, when using double
 * precision data types.
 */
struct mtxfile_vector_coordinate_integer_double
{
    int64_t i;   /* offset */
    int64_t a;   /* nonzero value */
};

/**
 * ‘mtxfile_vector_coordinate_pattern’ represents a nonzero vector
 * entry in a Matrix Market file with ‘vector’ object, ‘coordinate’
 * format and ‘pattern’ field.
 */
struct mtxfile_vector_coordinate_pattern
{
    int64_t i;   /* offset */
};

/*
 * Data structures for Matrix Market data lines.
 */

/**
 * ‘mtxfiledata’ represents an array of data lines from a Matrix
 * Market file.
 */
union mtxfiledata
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
    int64_t k);

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
    size_t * size_per_element);

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
    char ** endptr,
    const char * s);

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
    char ** endptr,
    const char * s);

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
    char ** endptr,
    const char * s);

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
    char ** endptr,
    const char * s);

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
    char ** endptr,
    const char * s);

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
    char ** endptr,
    const char * s);

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
    char ** endptr,
    const char * s,
    int64_t num_rows,
    int64_t num_columns);

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
    char ** endptr,
    const char * s,
    int64_t num_rows,
    int64_t num_columns);

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
    char ** endptr,
    const char * s,
    int64_t num_rows,
    int64_t num_columns);

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
    char ** endptr,
    const char * s,
    int64_t num_rows,
    int64_t num_columns);

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
    char ** endptr,
    const char * s,
    int64_t num_rows,
    int64_t num_columns);

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
    char ** endptr,
    const char * s,
    int64_t num_rows,
    int64_t num_columns);

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
    char ** endptr,
    const char * s,
    int64_t num_rows,
    int64_t num_columns);

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
    char ** endptr,
    const char * s,
    int64_t num_rows);

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
    char ** endptr,
    const char * s,
    int64_t num_rows);

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
    char ** endptr,
    const char * s,
    int64_t num_rows);

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
    char ** endptr,
    const char * s,
    int64_t num_rows);

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
    char ** endptr,
    const char * s,
    int64_t num_rows);

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
    char ** endptr,
    const char * s,
    int64_t num_rows);

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
    char ** endptr,
    const char * s,
    int64_t num_rows);

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
    int64_t size);

/**
 * ‘mtxfiledata_free()’ frees allocaed storage for data lines.
 */
int mtxfiledata_free(
    union mtxfiledata * data,
    enum mtxfileobject object,
    enum mtxfileformat format,
    enum mtxfilefield field,
    enum mtxprecision precision);

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
    int64_t srcoffset);

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
    const int64_t * srcdispls);

/*
 * Extracting row/column pointers and indices
 */

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
    int * colidx);

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
    void * dstdata);

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
    void * dstdata);

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
    float a);

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
    double a);

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
    float a[2]);

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
    double a[2]);

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
    int32_t a);

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
    int64_t a);

/*
 * I/O functions
 */

/**
 * ‘mtxfiledata_fread()’ reads Matrix Market data lines from a
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
    int64_t offset);

#ifdef LIBMTX_HAVE_LIBZ
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
    int64_t offset);
#endif

/**
 * ‘mtxfiledata_fwrite()’ writes data lines of a Matrix Market file
 * to a stream.
 *
 * If ‘fmt’ is ‘NULL’, then the format specifier ‘%g’ is used to print
 * floating point numbers with enough digits to ensure correct
 * round-trip conversion from decimal text and back.  Otherwise, the
 * given format string is used to print numerical values.
 *
 * The format string ‘fmt’ follows the conventions of ‘printf’. If
 * ‘field’ is ‘mtxfile_real’ or ‘mtxfile_complex’, then the format
 * specifiers '%e', '%E', '%f', '%F', '%g' or '%G' may be used. If
 * ‘field’ is ‘mtxfile_integer’, then the format specifier must be
 * '%d'. The format string is ignored if ‘field’ is
 * ‘mtxfile_pattern’. Field width and precision may be specified
 * (e.g., "%3.1f"), but variable field width and precision (e.g.,
 * "%*.*f"), as well as length modifiers (e.g., "%Lf") are not
 * allowed.
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
    int64_t * bytes_written);

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
 * The format string ‘fmt’ follows the conventions of ‘printf’. If
 * ‘field’ is ‘mtxfile_real’ or ‘mtxfile_complex’, then the format
 * specifiers '%e', '%E', '%f', '%F', '%g' or '%G' may be used. If
 * ‘field’ is ‘mtxfile_integer’, then the format specifier must be
 * '%d'. The format string is ignored if ‘field’ is
 * ‘mtxfile_pattern’. Field width and precision may be specified
 * (e.g., "%3.1f"), but variable field width and precision (e.g.,
 * "%*.*f"), as well as length modifiers (e.g., "%Lf") are not
 * allowed.
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
    int64_t * bytes_written);
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
    int64_t size);

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
    int64_t * perm);

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
    uint64_t * keys);

/**
 * ‘mtxfiledata_sortkey_column_major()’ provides an array of keys that
 * can be used to sort the data lines of the given Matrix Market file
 * in column major order.
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
    uint64_t * keys);

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
    uint64_t * keys);

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
    int64_t * sorting_permutation);

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
    int64_t * perm);

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
    int64_t * perm);

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
    int64_t * perm);

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
    int64_t * outsize);

/*
 * Partitioning
 */

/**
 * ‘mtxfiledata_partition()’ partitions data lines according to given
 * row and column partitions.
 *
 * The array ‘parts’ must contain enough storage for ‘size’ values of
 * type ‘int’. If successful, ‘parts’ will contain the part number of
 * each data line in the partitioning.
 *
 * If ‘rowidx’ and ‘colidx’ are not ‘NULL’, then they must point to
 * arrays of length ‘size’, which are then used to store the row and
 * column numbers, respectively, of each data line according to the
 * local numbering of rows and columns within each part.
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
    int64_t * rowidx,
    int64_t * colidx);

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
    const int * row_permutation,
    int64_t num_columns,
    const int * column_permutation);

/*
 * MPI functions
 */

#ifdef LIBMTX_HAVE_MPI
/**
 * ‘mtxfiledata_mpi_datatype()’ creates a custom MPI data type for
 * sending or receiving data lines.
 *
 * The user is responsible for calling ‘MPI_Type_free()’ on the
 * returned datatype.
 */
int mtxfiledata_mpi_datatype(
    const union mtxfiledata * data,
    enum mtxfileobject object,
    enum mtxfileformat format,
    enum mtxfilefield field,
    enum mtxprecision precision,
    MPI_Datatype * datatype,
    int * mpierrcode);

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
    struct mtxdisterror * disterr);

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
    struct mtxdisterror * disterr);

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
    struct mtxdisterror * disterr);

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
    struct mtxdisterror * disterr);

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
    struct mtxdisterror * disterr);

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
    struct mtxdisterror * disterr);
#endif

#endif
