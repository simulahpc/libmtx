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
 * Last modified: 2022-03-09
 *
 * Data structures for matrices.
 */

#ifndef LIBMTX_MATRIX_MATRIX_H
#define LIBMTX_MATRIX_MATRIX_H

#include <libmtx/libmtx-config.h>

#include <libmtx/precision.h>
#include <libmtx/field.h>
#include <libmtx/util/transpose.h>
#include <libmtx/matrix/matrix_array.h>
#include <libmtx/matrix/matrix_coordinate.h>
#include <libmtx/matrix/matrix_csr.h>
#include <libmtx/vector/vector.h>

#ifdef LIBMTX_HAVE_LIBZ
#include <zlib.h>
#endif

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

struct mtxfile;
struct mtxpartition;
struct mtxvector;

/*
 * Matrix types
 */

/**
 * ‘mtxmatrixtype’ is used to enumerate different matrix formats.
 */
enum mtxmatrixtype
{
    mtxmatrix_auto,       /* automatic selection of matrix type */
    mtxmatrix_array,      /* array format for dense matrices */
    mtxmatrix_coordinate, /* coordinate format for sparse matrices */
    mtxmatrix_csr,        /* compressed sparse row */
};

/**
 * ‘mtxmatrixtype_str()’ is a string representing the matrix type.
 */
const char * mtxmatrixtype_str(
    enum mtxmatrixtype type);

/**
 * ‘mtxmatrixtype_parse()’ parses a string to obtain one of the
 * matrix types of ‘enum mtxmatrixtype’.
 *
 * ‘valid_delimiters’ is either ‘NULL’, in which case it is ignored,
 * or it is a string of characters considered to be valid delimiters
 * for the parsed string.  That is, if there are any remaining,
 * non-NULL characters after parsing, then then the next character is
 * searched for in ‘valid_delimiters’.  If the character is found,
 * then the parsing succeeds and the final delimiter character is
 * consumed by the parser. Otherwise, the parsing fails with an error.
 *
 * If ‘endptr’ is not ‘NULL’, then the address stored in ‘endptr’
 * points to the first character beyond the characters that were
 * consumed during parsing.
 *
 * On success, ‘mtxmatrixtype_parse()’ returns ‘MTX_SUCCESS’ and
 * ‘matrix_type’ is set according to the parsed string and
 * ‘bytes_read’ is set to the number of bytes that were consumed by
 * the parser.  Otherwise, an error code is returned.
 */
int mtxmatrixtype_parse(
    enum mtxmatrixtype * matrix_type,
    int64_t * bytes_read,
    const char ** endptr,
    const char * s,
    const char * valid_delimiters);

/*
 * Abstract matrix data structure
 */

/**
 * ‘mtxmatrix’ represents a matrix with various options available for
 * the underlying storage and implementation of matrix operations.
 */
struct mtxmatrix
{
    /**
     * ‘type’ denotes the matrix type or storage format: ‘array’,
     * ‘coordinate’ or ‘csr’.
     */
    enum mtxmatrixtype type;

    /**
     * ‘storage’ is a union of different data types for the underlying
     * storage of the matrix.
     */
    union
    {
        struct mtxmatrix_array array;
        struct mtxmatrix_coordinate coordinate;
        struct mtxmatrix_csr csr;
    } storage;
};

/*
 * Memory management
 */

/**
 * ‘mtxmatrix_free()’ frees storage allocated for a matrix.
 */
void mtxmatrix_free(
    struct mtxmatrix * matrix);

/**
 * ‘mtxmatrix_alloc_copy()’ allocates a copy of a matrix without
 * initialising the values.
 */
int mtxmatrix_alloc_copy(
    struct mtxmatrix * dst,
    const struct mtxmatrix * src);

/**
 * ‘mtxmatrix_init_copy()’ allocates a copy of a matrix and also
 * copies the values.
 */
int mtxmatrix_init_copy(
    struct mtxmatrix * dst,
    const struct mtxmatrix * src);

/*
 * Matrix initialisation for array formats
 */

/**
 * ‘mtxmatrix_alloc_array()’ allocates a matrix in array format.
 */
int mtxmatrix_alloc_array(
    struct mtxmatrix * matrix,
    enum mtxfield field,
    enum mtxprecision precision,
    int num_rows,
    int num_columns);

/**
 * ‘mtxmatrix_init_array_real_single()’ allocates and initialises a
 * matrix in array format with real, single precision coefficients.
 */
int mtxmatrix_init_array_real_single(
    struct mtxmatrix * matrix,
    int num_rows,
    int num_columns,
    const float * data);

/**
 * ‘mtxmatrix_init_array_real_double()’ allocates and initialises a
 * matrix in array format with real, double precision coefficients.
 */
int mtxmatrix_init_array_real_double(
    struct mtxmatrix * matrix,
    int num_rows,
    int num_columns,
    const double * data);

/**
 * ‘mtxmatrix_init_array_complex_single()’ allocates and initialises a
 * matrix in array format with complex, single precision coefficients.
 */
int mtxmatrix_init_array_complex_single(
    struct mtxmatrix * matrix,
    int num_rows,
    int num_columns,
    const float (* data)[2]);

/**
 * ‘mtxmatrix_init_array_complex_double()’ allocates and initialises a
 * matrix in array format with complex, double precision coefficients.
 */
int mtxmatrix_init_array_complex_double(
    struct mtxmatrix * matrix,
    int num_rows,
    int num_columns,
    const double (* data)[2]);

/**
 * ‘mtxmatrix_init_array_integer_single()’ allocates and initialises a
 * matrix in array format with integer, single precision coefficients.
 */
int mtxmatrix_init_array_integer_single(
    struct mtxmatrix * matrix,
    int num_rows,
    int num_columns,
    const int32_t * data);

/**
 * ‘mtxmatrix_init_array_integer_double()’ allocates and initialises a
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
 * ‘mtxmatrix_alloc_coordinate()’ allocates a matrix in
 * coordinate format.
 */
int mtxmatrix_alloc_coordinate(
    struct mtxmatrix * matrix,
    enum mtxfield field,
    enum mtxprecision precision,
    int num_rows,
    int num_columns,
    int64_t num_nonzeros);

/**
 * ‘mtxmatrix_init_coordinate_real_single()’ allocates and initialises
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
    const float * data);

/**
 * ‘mtxmatrix_init_coordinate_real_double()’ allocates and initialises
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
    const double * data);

/**
 * ‘mtxmatrix_init_coordinate_complex_single()’ allocates and
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
    const float (* data)[2]);

/**
 * ‘mtxmatrix_init_coordinate_complex_double()’ allocates and
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
    const double (* data)[2]);

/**
 * ‘mtxmatrix_init_coordinate_integer_single()’ allocates and
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
    const int32_t * data);

/**
 * ‘mtxmatrix_init_coordinate_integer_double()’ allocates and
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
    const int64_t * data);

/**
 * ‘mtxmatrix_init_coordinate_pattern()’ allocates and initialises a
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
 * Compressed sparse row (CSR)
 */

/**
 * ‘mtxmatrix_alloc_csr()’ allocates a matrix in CSR format.
 */
int mtxmatrix_alloc_csr(
    struct mtxmatrix * matrix,
    enum mtxfield field,
    enum mtxprecision precision,
    int num_rows,
    int num_columns,
    int64_t num_nonzeros);

/**
 * ‘mtxmatrix_init_csr_real_single()’ allocates and initialises a
 * matrix in CSR format with real, single precision coefficients.
 */
int mtxmatrix_init_csr_real_single(
    struct mtxmatrix * matrix,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const float * data);

/**
 * ‘mtxmatrix_init_csr_real_double()’ allocates and initialises a
 * matrix in CSR format with real, double precision coefficients.
 */
int mtxmatrix_init_csr_real_double(
    struct mtxmatrix * matrix,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const double * data);

/**
 * ‘mtxmatrix_init_csr_complex_single()’ allocates and initialises a
 * matrix in CSR format with complex, single precision coefficients.
 */
int mtxmatrix_init_csr_complex_single(
    struct mtxmatrix * matrix,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const float (* data)[2]);

/**
 * ‘mtxmatrix_init_csr_complex_double()’ allocates and initialises a
 * matrix in CSR format with complex, double precision coefficients.
 */
int mtxmatrix_init_csr_complex_double(
    struct mtxmatrix * matrix,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const double (* data)[2]);

/**
 * ‘mtxmatrix_init_csr_integer_single()’ allocates and initialises a
 * matrix in CSR format with integer, single precision coefficients.
 */
int mtxmatrix_init_csr_integer_single(
    struct mtxmatrix * matrix,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const int32_t * data);

/**
 * ‘mtxmatrix_init_csr_integer_double()’ allocates and initialises a
 * matrix in CSR format with integer, double precision coefficients.
 */
int mtxmatrix_init_csr_integer_double(
    struct mtxmatrix * matrix,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const int64_t * data);

/**
 * ‘mtxmatrix_init_csr_pattern()’ allocates and initialises a matrix
 * in CSR format with integer, double precision coefficients.
 */
int mtxmatrix_init_csr_pattern(
    struct mtxmatrix * matrix,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx);

/*
 * Row and column vectors
 */

/**
 * ‘mtxmatrix_alloc_row_vector()’ allocates a row vector for a given
 * matrix, where a row vector is a vector whose length equal to a
 * single row of the matrix.
 */
int mtxmatrix_alloc_row_vector(
    const struct mtxmatrix * matrix,
    struct mtxvector * vector,
    enum mtxvectortype vector_type);

/**
 * ‘mtxmatrix_alloc_column_vector()’ allocates a column vector for a
 * given matrix, where a column vector is a vector whose length equal
 * to a single column of the matrix.
 */
int mtxmatrix_alloc_column_vector(
    const struct mtxmatrix * matrix,
    struct mtxvector * vector,
    enum mtxvectortype vector_type);


/*
 * Convert to and from Matrix Market format
 */

/**
 * ‘mtxmatrix_from_mtxfile()’ converts a matrix in Matrix Market
 * format to a matrix.
 */
int mtxmatrix_from_mtxfile(
    struct mtxmatrix * dst,
    const struct mtxfile * src,
    enum mtxmatrixtype type);

/**
 * ‘mtxmatrix_to_mtxfile()’ converts a matrix to a matrix in Matrix
 * Market format.
 */
int mtxmatrix_to_mtxfile(
    struct mtxfile * dst,
    const struct mtxmatrix * src,
    enum mtxfileformat mtxfmt);

/*
 * I/O functions
 */

/**
 * ‘mtxmatrix_read()’ reads a matrix from a Matrix Market file.  The
 * file may optionally be compressed by gzip.
 *
 * The ‘precision’ argument specifies which precision to use for
 * storing matrix or matrix values.
 *
 * The ‘type’ argument specifies which format to use for representing
 * the matrix.  If ‘type’ is ‘mtxmatrix_auto’, then the underlying
 * matrix is stored in array format or coordinate format according to
 * the format of the Matrix Market file.  Otherwise, an attempt is
 * made to convert the matrix to the desired type.
 *
 * If ‘path’ is ‘-’, then standard input is used.
 *
 * The file is assumed to be gzip-compressed if ‘gzip’ is ‘true’, and
 * uncompressed otherwise.
 *
 * If an error code is returned, then ‘lines_read’ and ‘bytes_read’
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the matrix.
 */
int mtxmatrix_read(
    struct mtxmatrix * matrix,
    enum mtxprecision precision,
    enum mtxmatrixtype type,
    const char * path,
    bool gzip,
    int * lines_read,
    int64_t * bytes_read);

/**
 * ‘mtxmatrix_fread()’ reads a matrix from a stream in Matrix Market
 * format.
 *
 * ‘precision’ is used to determine the precision to use for storing
 * the values of matrix or matrix entries.
 *
 * The ‘type’ argument specifies which format to use for representing
 * the matrix.  If ‘type’ is ‘mtxmatrix_auto’, then the underlying
 * matrix is stored in array format or coordinate format according to
 * the format of the Matrix Market file.  Otherwise, an attempt is
 * made to convert the matrix to the desired type.
 *
 * If an error code is returned, then ‘lines_read’ and ‘bytes_read’
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the matrix.
 */
int mtxmatrix_fread(
    struct mtxmatrix * matrix,
    enum mtxprecision precision,
    enum mtxmatrixtype type,
    FILE * f,
    int * lines_read,
    int64_t * bytes_read,
    size_t line_max,
    char * linebuf);

#ifdef LIBMTX_HAVE_LIBZ
/**
 * ‘mtxmatrix_gzread()’ reads a matrix from a gzip-compressed stream.
 *
 * ‘precision’ is used to determine the precision to use for storing
 * the values of matrix or matrix entries.
 *
 * The ‘type’ argument specifies which format to use for representing
 * the matrix.  If ‘type’ is ‘mtxmatrix_auto’, then the underlying
 * matrix is stored in array format or coordinate format according to
 * the format of the Matrix Market file.  Otherwise, an attempt is
 * made to convert the matrix to the desired type.
 *
 * If an error code is returned, then ‘lines_read’ and ‘bytes_read’
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the matrix.
 */
int mtxmatrix_gzread(
    struct mtxmatrix * matrix,
    enum mtxprecision precision,
    enum mtxmatrixtype type,
    gzFile f,
    int * lines_read,
    int64_t * bytes_read,
    size_t line_max,
    char * linebuf);
#endif

/**
 * ‘mtxmatrix_write()’ writes a matrix to a Matrix Market file. The
 * file may optionally be compressed by gzip.
 *
 * If ‘path’ is ‘-’, then standard output is used.
 *
 * If ‘fmt’ is ‘NULL’, then the format specifier ‘%g’ is used to print
 * floating point numbers with enough digits to ensure correct
 * round-trip conversion from decimal text and back.  Otherwise, the
 * given format string is used to print numerical values.
 *
 * The format string follows the conventions of ‘printf’. If the field
 * is ‘real’ or ‘complex’, then the format specifiers '%e', '%E',
 * '%f', '%F', '%g' or '%G' may be used. If the field is ‘integer’,
 * then the format specifier must be '%d'. The format string is
 * ignored if the field is ‘pattern’. Field width and precision may be
 * specified (e.g., "%3.1f"), but variable field width and precision
 * (e.g., "%*.*f"), as well as length modifiers (e.g., "%Lf") are not
 * allowed.
 */
int mtxmatrix_write(
    const struct mtxmatrix * matrix,
    enum mtxfileformat mtxfmt,
    const char * path,
    bool gzip,
    const char * fmt,
    int64_t * bytes_written);

/**
 * ‘mtxmatrix_fwrite()’ writes a matrix to a stream.
 *
 * If ‘fmt’ is ‘NULL’, then the format specifier ‘%g’ is used to print
 * floating point numbers with enough digits to ensure correct
 * round-trip conversion from decimal text and back.  Otherwise, the
 * given format string is used to print numerical values.
 *
 * The format string follows the conventions of ‘printf’. If the field
 * is ‘real’ or ‘complex’, then the format specifiers '%e',
 * '%E', '%f', '%F', '%g' or '%G' may be used. If the field is
 * ‘integer’, then the format specifier must be '%d'. The format
 * string is ignored if the field is ‘pattern’. Field width and
 * precision may be specified (e.g., "%3.1f"), but variable field
 * width and precision (e.g., "%*.*f"), as well as length modifiers
 * (e.g., "%Lf") are not allowed.
 *
 * If it is not ‘NULL’, then the number of bytes written to the stream
 * is returned in ‘bytes_written’.
 */
int mtxmatrix_fwrite(
    const struct mtxmatrix * matrix,
    enum mtxfileformat mtxfmt,
    FILE * f,
    const char * fmt,
    int64_t * bytes_written);

#ifdef LIBMTX_HAVE_LIBZ
/**
 * ‘mtxmatrix_gzwrite()’ writes a matrix to a gzip-compressed stream.
 *
 * If ‘fmt’ is ‘NULL’, then the format specifier ‘%g’ is used to print
 * floating point numbers with enough digits to ensure correct
 * round-trip conversion from decimal text and back.  Otherwise, the
 * given format string is used to print numerical values.
 *
 * The format string follows the conventions of ‘printf’. If the field
 * is ‘real’ or ‘complex’, then the format specifiers '%e', '%E',
 * '%f', '%F', '%g' or '%G' may be used. If the field is ‘integer’,
 * then the format specifier must be '%d'. The format string is
 * ignored if the field is ‘pattern’. Field width and precision may be
 * specified (e.g., "%3.1f"), but variable field width and precision
 * (e.g., "%*.*f"), as well as length modifiers (e.g., "%Lf") are not
 * allowed.
 *
 * If it is not ‘NULL’, then the number of bytes written to the stream
 * is returned in ‘bytes_written’.
 */
int mtxmatrix_gzwrite(
    const struct mtxmatrix * matrix,
    enum mtxfileformat mtxfmt,
    gzFile f,
    const char * fmt,
    int64_t * bytes_written);
#endif

/*
 * Nonzero rows and columns
 */

/**
 * ‘mtxmatrix_nzrows()’ counts the number of nonzero (non-empty)
 * matrix rows, and, optionally, fills an array with the row indices
 * of the nonzero (non-empty) matrix rows.
 *
 * If ‘num_nonzero_rows’ is ‘NULL’, then it is ignored, or else it
 * must point to an integer that is used to store the number of
 * nonzero matrix rows.
 *
 * ‘nonzero_rows’ may be ‘NULL’, in which case it is ignored.
 * Otherwise, it must point to an array of length at least equal to
 * ‘size’. On successful completion, this array contains the row
 * indices of the nonzero matrix rows. Note that ‘size’ must be at
 * least equal to the number of non-zero rows.
 */
int mtxmatrix_nzrows(
    const struct mtxmatrix * matrix,
    int * num_nonzero_rows,
    int size,
    int * nonzero_rows);

/**
 * ‘mtxmatrix_nzcols()’ counts the number of nonzero (non-empty)
 * matrix columns, and, optionally, fills an array with the column
 * indices of the nonzero (non-empty) matrix columns.
 *
 * If ‘num_nonzero_columns’ is ‘NULL’, then it is ignored, or else it
 * must point to an integer that is used to store the number of
 * nonzero matrix columns.
 *
 * ‘nonzero_columns’ may be ‘NULL’, in which case it is ignored.
 * Otherwise, it must point to an array of length at least equal to
 * ‘size’. On successful completion, this array contains the column
 * indices of the nonzero matrix columns. Note that ‘size’ must be at
 * least equal to the number of non-zero columns.
 */
int mtxmatrix_nzcols(
    const struct mtxmatrix * matrix,
    int * num_nonzero_columns,
    int size,
    int * nonzero_columns);

/*
 * Partitioning
 */

/**
 * ‘mtxmatrix_partition()’ partitions a matrix into blocks according
 * to the given row and column partitions.
 *
 * The partitions ‘rowpart’ or ‘colpart’ are allowed to be ‘NULL’, in
 * which case a trivial, singleton partition is used for the rows or
 * columns, respectively.
 *
 * Otherwise, ‘rowpart’ and ‘colpart’ must partition the rows and
 * columns of the matrix ‘src’, respectively. That is, ‘rowpart->size’
 * must be equal to the number of matrix rows, and ‘colpart->size’
 * must be equal to the number of matrix columns.
 *
 * The argument ‘dsts’ is an array that must have enough storage for
 * ‘P*Q’ values of type ‘struct mtxmatrix’, where ‘P’ is the number of
 * row parts, ‘rowpart->num_parts’, and ‘Q’ is the number of column
 * parts, ‘colpart->num_parts’. Note that the ‘r’th part corresponds
 * to a row part ‘p’ and column part ‘q’, such that ‘r=p*Q+q’. Thus,
 * the ‘r’th entry of ‘dsts’ is the submatrix corresponding to the
 * ‘p’th row and ‘q’th column of the 2D partitioning.
 *
 * The user is responsible for freeing storage allocated for each
 * matrix in the ‘dsts’ array.
 */
int mtxmatrix_partition(
    struct mtxmatrix * dsts,
    const struct mtxmatrix * src,
    const struct mtxpartition * rowpart,
    const struct mtxpartition * colpart);

/**
 * ‘mtxmatrix_join()’ joins together matrices representing compatible
 * blocks of a partitioned matrix to form a larger matrix.
 *
 * The argument ‘srcs’ is logically arranged as a two-dimensional
 * array of size ‘P*Q’, where ‘P’ is the number of row parts
 * (‘rowpart->num_parts’) and ‘Q’ is the number of column parts
 * (‘colpart->num_parts’).  Note that the ‘r’th part corresponds to a
 * row part ‘p’ and column part ‘q’, such that ‘r=p*Q+q’. Thus, the
 * ‘r’th entry of ‘srcs’ is the submatrix corresponding to the ‘p’th
 * row and ‘q’th column of the 2D partitioning.
 *
 * Moreover, the blocks must be compatible, which means that each part
 * in the same block row ‘p’, must have the same number of rows.
 * Similarly, each part in the same block column ‘q’ must have the
 * same number of columns. Finally, for each block column ‘q’, the sum
 * of the number of rows of ‘srcs[p*Q+q]’ for ‘p=0,1,...,P-1’ must be
 * equal to ‘rowpart->size’. Likewise, for each block row ‘p’, the sum
 * of the number of columns of ‘srcs[p*Q+q]’ for ‘q=0,1,...,Q-1’ must
 * be equal to ‘colpart->size’.
 */
int mtxmatrix_join(
    struct mtxmatrix * dst,
    const struct mtxmatrix * srcs,
    const struct mtxpartition * rowpart,
    const struct mtxpartition * colpart);

/*
 * Level 1 BLAS operations
 */

/**
 * ‘mtxmatrix_swap()’ swaps values of two matrices, simultaneously
 * performing ‘Y <- X’ and ‘X <- Y’.
 */
int mtxmatrix_swap(
    struct mtxmatrix * X,
    struct mtxmatrix * Y);

/**
 * ‘mtxmatrix_copy()’ copies values of a matrix, ‘Y = X’.
 */
int mtxmatrix_copy(
    struct mtxmatrix * Y,
    const struct mtxmatrix * X);

/**
 * ‘mtxmatrix_sscal()’ scales a matrix by a single precision floating
 * point scalar, ‘X = a*X’.
 */
int mtxmatrix_sscal(
    float a,
    struct mtxmatrix * X,
    int64_t * num_flops);

/**
 * ‘mtxmatrix_dscal()’ scales a matrix by a double precision floating
 * point scalar, ‘X = a*X’.
 */
int mtxmatrix_dscal(
    double a,
    struct mtxmatrix * X,
    int64_t * num_flops);

/**
 * ‘mtxmatrix_saxpy()’ adds a matrix to another matrix multiplied by a
 * single precision floating point value, ‘Y = a*X + Y’.
 */
int mtxmatrix_saxpy(
    float a,
    const struct mtxmatrix * X,
    struct mtxmatrix * Y,
    int64_t * num_flops);

/**
 * ‘mtxmatrix_daxpy()’ adds a matrix to another matrix multiplied by a
 * double precision floating point value, ‘Y = a*X + Y’.
 */
int mtxmatrix_daxpy(
    double a,
    const struct mtxmatrix * X,
    struct mtxmatrix * Y,
    int64_t * num_flops);

/**
 * ‘mtxmatrix_saypx()’ multiplies a matrix by a single precision
 * floating point scalar and adds another matrix, ‘Y = a*Y + X’.
 */
int mtxmatrix_saypx(
    float a,
    struct mtxmatrix * Y,
    const struct mtxmatrix * X,
    int64_t * num_flops);

/**
 * ‘mtxmatrix_daypx()’ multiplies a matrix by a double precision
 * floating point scalar and adds another matrix, ‘Y = a*Y + X’.
 */
int mtxmatrix_daypx(
    double a,
    struct mtxmatrix * Y,
    const struct mtxmatrix * X,
    int64_t * num_flops);

/**
 * ‘mtxmatrix_sdot()’ computes the Frobenius dot product of two
 * matrices in single precision floating point.
 */
int mtxmatrix_sdot(
    const struct mtxmatrix * X,
    const struct mtxmatrix * Y,
    float * dot,
    int64_t * num_flops);

/**
 * ‘mtxmatrix_ddot()’ computes the Frobenius dot product of two
 * matrices in double precision floating point.
 */
int mtxmatrix_ddot(
    const struct mtxmatrix * X,
    const struct mtxmatrix * Y,
    double * dot,
    int64_t * num_flops);

/**
 * ‘mtxmatrix_cdotu()’ computes the product of the dot product of the
 * vectorisation of a complex matrix with the vectorisation of another
 * complex matrix in single precision floating point, ‘dot :=
 * vec(X)^T*vec(Y)’, where ‘vec(X)’ is the vectorisation of ‘X’.
 */
int mtxmatrix_cdotu(
    const struct mtxmatrix * X,
    const struct mtxmatrix * Y,
    float (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxmatrix_zdotu()’ computes the product of the dot product of the
 * vectorisation of a complex matrix with the vectorisation of another
 * complex matrix in double precision floating point, ‘dot :=
 * vec(X)^T*vec(Y)’, where ‘vec(X)’ is the vectorisation of ‘X’.
 */
int mtxmatrix_zdotu(
    const struct mtxmatrix * X,
    const struct mtxmatrix * Y,
    double (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxmatrix_cdotc()’ computes the Frobenius dot product of two
 * complex matrices in single precision floating point, ‘dot :=
 * vec(X)^H*vec(Y)’, where ‘vec(X)’ is the vectorisation of ‘X’.
 */
int mtxmatrix_cdotc(
    const struct mtxmatrix * X,
    const struct mtxmatrix * Y,
    float (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxmatrix_zdotc()’ computes the Frobenius dot product of two
 * complex matrices in double precision floating point, ‘dot :=
 * vec(X)^H*vec(Y)’, where ‘vec(X)’ is the vectorisation of ‘X’.
 */
int mtxmatrix_zdotc(
    const struct mtxmatrix * X,
    const struct mtxmatrix * Y,
    double (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxmatrix_snrm2()’ computes the Frobenius norm of a matrix in
 * single precision floating point.
 */
int mtxmatrix_snrm2(
    const struct mtxmatrix * X,
    float * nrm2,
    int64_t * num_flops);

/**
 * ‘mtxmatrix_dnrm2()’ computes the Frobenius norm of a matrix in
 * double precision floating point.
 */
int mtxmatrix_dnrm2(
    const struct mtxmatrix * X,
    double * nrm2,
    int64_t * num_flops);

/**
 * ‘mtxmatrix_sasum()’ computes the sum of absolute values of a matrix
 * in single precision floating point.  If the matrix is
 * complex-valued, then the sum of the absolute values of the real and
 * imaginary parts is computed.
 */
int mtxmatrix_sasum(
    const struct mtxmatrix * X,
    float * asum,
    int64_t * num_flops);

/**
 * ‘mtxmatrix_dasum()’ computes the sum of absolute values of a matrix
 * in double precision floating point.  If the matrix is
 * complex-valued, then the sum of the absolute values of the real and
 * imaginary parts is computed.
 */
int mtxmatrix_dasum(
    const struct mtxmatrix * X,
    double * asum,
    int64_t * num_flops);

/**
 * ‘mtxmatrix_iamax()’ finds the index of the first element having the
 * maximum absolute value.  If the matrix is complex-valued, then the
 * index points to the first element having the maximum sum of the
 * absolute values of the real and imaginary parts.
 */
int mtxmatrix_iamax(
    const struct mtxmatrix * X,
    int * iamax);

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
    enum mtxtransposition trans,
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
    enum mtxtransposition trans,
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
    enum mtxtransposition trans,
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
    enum mtxtransposition trans,
    double alpha[2],
    const struct mtxmatrix * A,
    const struct mtxvector * x,
    double beta[2],
    struct mtxvector * y);

#endif
