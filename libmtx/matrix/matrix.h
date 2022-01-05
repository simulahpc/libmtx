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
 * Last modified: 2021-10-05
 *
 * Data structures for matrices.
 */

#ifndef LIBMTX_MATRIX_MATRIX_H
#define LIBMTX_MATRIX_MATRIX_H

#include <libmtx/libmtx-config.h>

#include <libmtx/mtx/precision.h>
#include <libmtx/util/field.h>
#include <libmtx/util/transpose.h>
#include <libmtx/matrix/matrix_array.h>
#include <libmtx/matrix/matrix_coordinate.h>
#include <libmtx/vector/vector.h>

#ifdef LIBMTX_HAVE_LIBZ
#include <zlib.h>
#endif

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

struct mtxfile;
struct mtxvector;

/*
 * Matrix types
 */

/**
 * `mtxmatrix_type' is used to enumerate different matrix formats.
 */
enum mtxmatrix_type
{
    mtxmatrix_auto,       /* automatic selection of matrix type */
    mtxmatrix_array,      /* array format for dense matrices */
    mtxmatrix_coordinate, /* coordinate format for sparse matrices */
};

/**
 * `mtxmatrix_type_str()' is a string representing the matrix type.
 */
const char * mtxmatrix_type_str(
    enum mtxmatrix_type type);

/**
 * `mtxmatrix_type_parse()' parses a string to obtain one of the
 * matrix types of `enum mtxmatrix_type'.
 *
 * `valid_delimiters' is either `NULL', in which case it is ignored,
 * or it is a string of characters considered to be valid delimiters
 * for the parsed string.  That is, if there are any remaining,
 * non-NULL characters after parsing, then then the next character is
 * searched for in `valid_delimiters'.  If the character is found,
 * then the parsing succeeds and the final delimiter character is
 * consumed by the parser. Otherwise, the parsing fails with an error.
 *
 * If `endptr' is not `NULL', then the address stored in `endptr'
 * points to the first character beyond the characters that were
 * consumed during parsing.
 *
 * On success, `mtxmatrix_type_parse()' returns `MTX_SUCCESS' and
 * `matrix_type' is set according to the parsed string and
 * `bytes_read' is set to the number of bytes that were consumed by
 * the parser.  Otherwise, an error code is returned.
 */
int mtxmatrix_type_parse(
    enum mtxmatrix_type * matrix_type,
    int64_t * bytes_read,
    const char ** endptr,
    const char * s,
    const char * valid_delimiters);

/*
 * Abstract matrix data structure
 */

/**
 * `mtxmatrix' represents a matrix with various options available for
 * the underlying storage and implementation of matrix operations.
 */
struct mtxmatrix
{
    /**
     * `format' is the matrix format: ‘array’ or ‘coordinate’.
     */
    enum mtxmatrix_type type;

    /**
     * `matrix' is a union of different data types for the underlying
     * storage of the matrix.
     */
    union
    {
        struct mtxmatrix_array array;
        struct mtxmatrix_coordinate coordinate;
    } storage;
};

/*
 * Memory management
 */

/**
 * `mtxmatrix_free()' frees storage allocated for a matrix.
 */
void mtxmatrix_free(
    struct mtxmatrix * matrix);

/**
 * `mtxmatrix_alloc_copy()' allocates a copy of a matrix without
 * initialising the values.
 */
int mtxmatrix_alloc_copy(
    struct mtxmatrix * dst,
    const struct mtxmatrix * src);

/**
 * `mtxmatrix_init_copy()' allocates a copy of a matrix and also
 * copies the values.
 */
int mtxmatrix_init_copy(
    struct mtxmatrix * dst,
    const struct mtxmatrix * src);

/*
 * Matrix initialisation for array formats
 */

/**
 * `mtxmatrix_alloc_array()' allocates a matrix in array format.
 */
int mtxmatrix_alloc_array(
    struct mtxmatrix * matrix,
    enum mtx_field_ field,
    enum mtxprecision precision,
    int num_rows,
    int num_columns);

/**
 * `mtxmatrix_init_array_real_single()' allocates and initialises a
 * matrix in array format with real, single precision coefficients.
 */
int mtxmatrix_init_array_real_single(
    struct mtxmatrix * matrix,
    int num_rows,
    int num_columns,
    const float * data);

/**
 * `mtxmatrix_init_array_real_double()' allocates and initialises a
 * matrix in array format with real, double precision coefficients.
 */
int mtxmatrix_init_array_real_double(
    struct mtxmatrix * matrix,
    int num_rows,
    int num_columns,
    const double * data);

/**
 * `mtxmatrix_init_array_complex_single()' allocates and initialises a
 * matrix in array format with complex, single precision coefficients.
 */
int mtxmatrix_init_array_complex_single(
    struct mtxmatrix * matrix,
    int num_rows,
    int num_columns,
    const float (* data)[2]);

/**
 * `mtxmatrix_init_array_complex_double()' allocates and initialises a
 * matrix in array format with complex, double precision coefficients.
 */
int mtxmatrix_init_array_complex_double(
    struct mtxmatrix * matrix,
    int num_rows,
    int num_columns,
    const double (* data)[2]);

/**
 * `mtxmatrix_init_array_integer_single()' allocates and initialises a
 * matrix in array format with integer, single precision coefficients.
 */
int mtxmatrix_init_array_integer_single(
    struct mtxmatrix * matrix,
    int num_rows,
    int num_columns,
    const int32_t * data);

/**
 * `mtxmatrix_init_array_integer_double()' allocates and initialises a
 * matrix in array format with integer, double precision coefficients.
 */
int mtxmatrix_init_array_integer_double(
    struct mtxmatrix * matrix,
    int num_rows,
    int num_columns,
    const int64_t * data);

/*
 * Matrix coordinate formats
 */

/**
 * `mtxmatrix_alloc_coordinate()' allocates a matrix in
 * coordinate format.
 */
int mtxmatrix_alloc_coordinate(
    struct mtxmatrix * matrix,
    enum mtx_field_ field,
    enum mtxprecision precision,
    int num_rows,
    int num_columns,
    int64_t num_nonzeros);

/**
 * `mtxmatrix_init_coordinate_real_single()' allocates and initialises
 * a matrix in coordinate format with real, single precision
 * coefficients.
 */
int mtxmatrix_init_coordinate_real_single(
    struct mtxmatrix * matrix,
    int num_rows,
    int num_columns,
    int64_t num_nonzeros,
    const int * rowidx,
    const int * colidx,
    const float * values);

/**
 * `mtxmatrix_init_coordinate_real_double()' allocates and initialises
 * a matrix in coordinate format with real, double precision
 * coefficients.
 */
int mtxmatrix_init_coordinate_real_double(
    struct mtxmatrix * matrix,
    int num_rows,
    int num_columns,
    int64_t num_nonzeros,
    const int * rowidx,
    const int * colidx,
    const double * values);

/**
 * `mtxmatrix_init_coordinate_complex_single()' allocates and
 * initialises a matrix in coordinate format with complex, single
 * precision coefficients.
 */
int mtxmatrix_init_coordinate_complex_single(
    struct mtxmatrix * matrix,
    int num_rows,
    int num_columns,
    int64_t num_nonzeros,
    const int * rowidx,
    const int * colidx,
    const float (* values)[2]);

/**
 * `mtxmatrix_init_coordinate_complex_double()' allocates and
 * initialises a matrix in coordinate format with complex, double
 * precision coefficients.
 */
int mtxmatrix_init_coordinate_complex_double(
    struct mtxmatrix * matrix,
    int num_rows,
    int num_columns,
    int64_t num_nonzeros,
    const int * rowidx,
    const int * colidx,
    const double (* values)[2]);

/**
 * `mtxmatrix_init_coordinate_integer_single()' allocates and
 * initialises a matrix in coordinate format with integer, single
 * precision coefficients.
 */
int mtxmatrix_init_coordinate_integer_single(
    struct mtxmatrix * matrix,
    int num_rows,
    int num_columns,
    int64_t num_nonzeros,
    const int * rowidx,
    const int * colidx,
    const int32_t * values);

/**
 * `mtxmatrix_init_coordinate_integer_double()' allocates and
 * initialises a matrix in coordinate format with integer, double
 * precision coefficients.
 */
int mtxmatrix_init_coordinate_integer_double(
    struct mtxmatrix * matrix,
    int num_rows,
    int num_columns,
    int64_t num_nonzeros,
    const int * rowidx,
    const int * colidx,
    const int64_t * values);

/**
 * `mtxmatrix_init_coordinate_pattern()' allocates and initialises a
 * matrix in coordinate format with integer, double precision
 * coefficients.
 */
int mtxmatrix_init_coordinate_pattern(
    struct mtxmatrix * matrix,
    int num_rows,
    int num_columns,
    int64_t num_nonzeros,
    const int * rowidx,
    const int * colidx);

/*
 * Row and column vectors
 */

/**
 * `mtxmatrix_alloc_row_vector()' allocates a row vector for a given
 * matrix, where a row vector is a vector whose length equal to a
 * single row of the matrix.
 */
int mtxmatrix_alloc_row_vector(
    const struct mtxmatrix * matrix,
    struct mtxvector * vector,
    enum mtxvector_type vector_type);

/**
 * `mtxmatrix_alloc_column_vector()' allocates a column vector for a
 * given matrix, where a column vector is a vector whose length equal
 * to a single column of the matrix.
 */
int mtxmatrix_alloc_column_vector(
    const struct mtxmatrix * matrix,
    struct mtxvector * vector,
    enum mtxvector_type vector_type);

/*
 * Convert to and from Matrix Market format
 */

/**
 * `mtxmatrix_from_mtxfile()' converts a matrix in Matrix Market
 * format to a matrix.
 */
int mtxmatrix_from_mtxfile(
    struct mtxmatrix * matrix,
    const struct mtxfile * mtxfile,
    enum mtxmatrix_type type);

/**
 * `mtxmatrix_to_mtxfile()' converts a matrix to a matrix in Matrix
 * Market format.
 */
int mtxmatrix_to_mtxfile(
    const struct mtxmatrix * matrix,
    struct mtxfile * mtxfile);

/*
 * I/O functions
 */

/**
 * `mtxmatrix_read()' reads a matrix from a Matrix Market file.  The
 * file may optionally be compressed by gzip.
 *
 * The `precision' argument specifies which precision to use for
 * storing matrix or matrix values.
 *
 * The `type' argument specifies which format to use for representing
 * the matrix.  If `type' is `mtxmatrix_auto', then the underlying
 * matrix is stored in array format or coordinate format according to
 * the format of the Matrix Market file.  Otherwise, an attempt is
 * made to convert the matrix to the desired type.
 *
 * If `path' is `-', then standard input is used.
 *
 * If an error code is returned, then `lines_read' and `bytes_read'
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the matrix.
 */
int mtxmatrix_read(
    struct mtxmatrix * matrix,
    enum mtxprecision precision,
    enum mtxmatrix_type type,
    const char * path,
    bool gzip,
    int * lines_read,
    int64_t * bytes_read);

/**
 * `mtxmatrix_fread()' reads a matrix from a stream in Matrix Market
 * format.
 *
 * `precision' is used to determine the precision to use for storing
 * the values of matrix or matrix entries.
 *
 * The `type' argument specifies which format to use for representing
 * the matrix.  If `type' is `mtxmatrix_auto', then the underlying
 * matrix is stored in array format or coordinate format according to
 * the format of the Matrix Market file.  Otherwise, an attempt is
 * made to convert the matrix to the desired type.
 *
 * If an error code is returned, then `lines_read' and `bytes_read'
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the matrix.
 */
int mtxmatrix_fread(
    struct mtxmatrix * matrix,
    enum mtxprecision precision,
    enum mtxmatrix_type type,
    FILE * f,
    int * lines_read,
    int64_t * bytes_read,
    size_t line_max,
    char * linebuf);

#ifdef LIBMTX_HAVE_LIBZ
/**
 * `mtxmatrix_gzread()' reads a matrix from a gzip-compressed stream.
 *
 * `precision' is used to determine the precision to use for storing
 * the values of matrix or matrix entries.
 *
 * The `type' argument specifies which format to use for representing
 * the matrix.  If `type' is `mtxmatrix_auto', then the underlying
 * matrix is stored in array format or coordinate format according to
 * the format of the Matrix Market file.  Otherwise, an attempt is
 * made to convert the matrix to the desired type.
 *
 * If an error code is returned, then `lines_read' and `bytes_read'
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the matrix.
 */
int mtxmatrix_gzread(
    struct mtxmatrix * matrix,
    enum mtxprecision precision,
    enum mtxmatrix_type type,
    gzFile f,
    int * lines_read,
    int64_t * bytes_read,
    size_t line_max,
    char * linebuf);
#endif

/**
 * `mtxmatrix_write()' writes a matrix to a Matrix Market file. The
 * file may optionally be compressed by gzip.
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
int mtxmatrix_write(
    const struct mtxmatrix * matrix,
    const char * path,
    bool gzip,
    const char * fmt,
    int64_t * bytes_written);

/**
 * `mtxmatrix_fwrite()' writes a matrix to a stream.
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
 *
 * If it is not `NULL', then the number of bytes written to the stream
 * is returned in `bytes_written'.
 */
int mtxmatrix_fwrite(
    const struct mtxmatrix * matrix,
    FILE * f,
    const char * fmt,
    int64_t * bytes_written);

#ifdef LIBMTX_HAVE_LIBZ
/**
 * `mtxmatrix_gzwrite()' writes a matrix to a gzip-compressed stream.
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
 *
 * If it is not `NULL', then the number of bytes written to the stream
 * is returned in `bytes_written'.
 */
int mtxmatrix_gzwrite(
    const struct mtxmatrix * matrix,
    gzFile f,
    const char * fmt,
    int64_t * bytes_written);
#endif

/*
 * Level 2 BLAS operations
 */

/**
 * ‘mtxmatrix_sgemv()’ multiplies a matrix ‘A’ or its transpose ‘A'’
 * by a real scalar ‘alpha’ (‘α’) and a vector ‘x’, before adding the
 * result to another vector ‘y’ multiplied by another real scalar
 * ‘beta’ (‘β’). That is, ‘y = α*A*x + β*y’ or ‘y = α*A'*x + β*y’.
 *
 * The scalars ‘alpha’ and ‘beta’ are given as single precision
 * floating point numbers.
 */
int mtxmatrix_sgemv(
    enum mtx_trans_type trans,
    float alpha,
    const struct mtxmatrix * A,
    const struct mtxvector * x,
    float beta,
    struct mtxvector * y);

/**
 * ‘mtxmatrix_dgemv()’ multiplies a matrix ‘A’ or its transpose ‘A'’
 * by a real scalar ‘alpha’ (‘α’) and a vector ‘x’, before adding the
 * result to another vector ‘y’ multiplied by another scalar real
 * ‘beta’ (‘β’).  That is, ‘y = α*A*x + β*y’ or ‘y = α*A'*x + β*y’.
 *
 * The scalars ‘alpha’ and ‘beta’ are given as double precision
 * floating point numbers.
 */
int mtxmatrix_dgemv(
    enum mtx_trans_type trans,
    double alpha,
    const struct mtxmatrix * A,
    const struct mtxvector * x,
    double beta,
    struct mtxvector * y);

/**
 * ‘mtxmatrix_cgemv()’ multiplies a complex-valued matrix ‘A’, its
 * transpose ‘A'’ or its conjugate transpose ‘Aᴴ’ by a complex scalar
 * ‘alpha’ (‘α’) and a vector ‘x’, before adding the result to another
 * vector ‘y’ multiplied by another complex scalar ‘beta’ (‘β’).  That
 * is, ‘y = α*A*x + β*y’, ‘y = α*A'*x + β*y’ or ‘y = α*Aᴴ*x + β*y’.
 *
 * The scalars ‘alpha’ and ‘beta’ are given as single precision
 * floating point numbers.
 */
int mtxmatrix_cgemv(
    enum mtx_trans_type trans,
    float alpha[2],
    const struct mtxmatrix * A,
    const struct mtxvector * x,
    float beta[2],
    struct mtxvector * y);

/**
 * ‘mtxmatrix_zgemv()’ multiplies a complex-valued matrix ‘A’, its
 * transpose ‘A'’ or its conjugate transpose ‘Aᴴ’ by a complex scalar
 * ‘alpha’ (‘α’) and a vector ‘x’, before adding the result to another
 * vector ‘y’ multiplied by another complex scalar ‘beta’ (‘β’).  That
 * is, ‘y = α*A*x + β*y’, ‘y = α*A'*x + β*y’ or ‘y = α*Aᴴ*x + β*y’.
 *
 * The scalars ‘alpha’ and ‘beta’ are given as double precision
 * floating point numbers.
 */
int mtxmatrix_zgemv(
    enum mtx_trans_type trans,
    double alpha[2],
    const struct mtxmatrix * A,
    const struct mtxvector * x,
    double beta[2],
    struct mtxvector * y);

#endif
