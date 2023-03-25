/* This file is part of Libmtx.
 *
 * Copyright (C) 2023 James D. Trotter
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
 * Last modified: 2023-03-25
 *
 * Matrix Market files.
 */

#ifndef LIBMTX_MTXFILE_MTXFILE_H
#define LIBMTX_MTXFILE_MTXFILE_H

#include <libmtx/libmtx-config.h>

#include <libmtx/linalg/precision.h>
#include <libmtx/mtxfile/comments.h>
#include <libmtx/mtxfile/data.h>
#include <libmtx/mtxfile/header.h>
#include <libmtx/mtxfile/size.h>
#include <libmtx/linalg/partition.h>
#include <libmtx/util/partition.h>

#ifdef LIBMTX_HAVE_MPI
#include <mpi.h>
#endif

#ifdef LIBMTX_HAVE_LIBZ
#include <zlib.h>
#endif

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

struct mtxdisterror;

/**
 * ‘mtxfile’ represents a file in the Matrix Market file format.
 */
struct mtxfile
{
    /**
     * ‘header’ is the Matrix Market file header.
     */
    struct mtxfileheader header;

    /**
     * ‘comments’ is the Matrix Market comment lines.
     */
    struct mtxfilecomments comments;

    /**
     * ‘size’ is the Matrix Market size line.
     */
    struct mtxfilesize size;

    /**
     * ‘precision’ is the precision used to store the values of the
     * Matrix Market data lines.
     */
    enum mtxprecision precision;

    /**
     * ‘datasize’ is the number of explicitly stored data lines in the
     * Matrix Market file.
     */
    int64_t datasize;

    /**
     * ‘data’ contains the data lines of the Matrix Market file.
     */
    union mtxfiledata data;
};

/*
 * Memory management
 */

/**
 * ‘mtxfile_alloc()’ allocates storage for a Matrix Market file with
 * the given header line, comment lines, size line and precision.
 *
 * ‘comments’ may be ‘NULL’, in which case it is ignored.
 */
int LIBMTX_API mtxfile_alloc(
    struct mtxfile * mtxfile,
    const struct mtxfileheader * header,
    const struct mtxfilecomments * comments,
    const struct mtxfilesize * size,
    enum mtxprecision precision);

/**
 * ‘mtxfile_free()’ frees storage allocated for a Matrix Market file.
 */
void LIBMTX_API mtxfile_free(
    struct mtxfile * mtxfile);

/**
 * ‘mtxfile_alloc_copy()’ allocates storage for a copy of a Matrix
 * Market file without initialising the underlying values.
 */
int LIBMTX_API mtxfile_alloc_copy(
    struct mtxfile * dst,
    const struct mtxfile * src);

/**
 * ‘mtxfile_init_copy()’ creates a copy of a Matrix Market file.
 */
int LIBMTX_API mtxfile_init_copy(
    struct mtxfile * dst,
    const struct mtxfile * src);

/*
 * Matrix array formats
 */

/**
 * ‘mtxfile_alloc_matrix_array()’ allocates a matrix in array format.
 */
int LIBMTX_API mtxfile_alloc_matrix_array(
    struct mtxfile * mtxfile,
    enum mtxfilefield field,
    enum mtxfilesymmetry symmetry,
    enum mtxprecision precision,
    int64_t num_rows,
    int64_t num_columns);

/**
 * ‘mtxfile_init_matrix_array_real_single()’ allocates and initialises
 * a matrix in array format with real, single precision coefficients.
 */
int LIBMTX_API mtxfile_init_matrix_array_real_single(
    struct mtxfile * mtxfile,
    enum mtxfilesymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    const float * data);

/**
 * ‘mtxfile_init_matrix_array_real_double()’ allocates and initialises
 * a matrix in array format with real, double precision coefficients.
 */
int LIBMTX_API mtxfile_init_matrix_array_real_double(
    struct mtxfile * mtxfile,
    enum mtxfilesymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    const double * data);

/**
 * ‘mtxfile_init_matrix_array_complex_single()’ allocates and
 * initialises a matrix in array format with complex, single precision
 * coefficients.
 */
int LIBMTX_API mtxfile_init_matrix_array_complex_single(
    struct mtxfile * mtxfile,
    enum mtxfilesymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    const float (* data)[2]);

/**
 * ‘mtxfile_init_matrix_array_complex_double()’ allocates and
 * initialises a matrix in array format with complex, double precision
 * coefficients.
 */
int LIBMTX_API mtxfile_init_matrix_array_complex_double(
    struct mtxfile * mtxfile,
    enum mtxfilesymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    const double (* data)[2]);

/**
 * ‘mtxfile_init_matrix_array_integer_single()’ allocates and
 * initialises a matrix in array format with integer, single precision
 * coefficients.
 */
int LIBMTX_API mtxfile_init_matrix_array_integer_single(
    struct mtxfile * mtxfile,
    enum mtxfilesymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    const int32_t * data);

/**
 * ‘mtxfile_init_matrix_array_integer_double()’ allocates and
 * initialises a matrix in array format with integer, double precision
 * coefficients.
 */
int LIBMTX_API mtxfile_init_matrix_array_integer_double(
    struct mtxfile * mtxfile,
    enum mtxfilesymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    const int64_t * data);

/*
 * Vector array formats
 */

/**
 * ‘mtxfile_alloc_vector_array()’ allocates a vector in array format.
 */
int LIBMTX_API mtxfile_alloc_vector_array(
    struct mtxfile * mtxfile,
    enum mtxfilefield field,
    enum mtxprecision precision,
    int64_t num_rows);

/**
 * ‘mtxfile_init_vector_array_real_single()’ allocates and initialises
 * a vector in array format with real, single precision coefficients.
 */
int LIBMTX_API mtxfile_init_vector_array_real_single(
    struct mtxfile * mtxfile,
    int64_t num_rows,
    const float * data);

/**
 * ‘mtxfile_init_vector_array_real_double()’ allocates and initialises
 * a vector in array format with real, double precision coefficients.
 */
int LIBMTX_API mtxfile_init_vector_array_real_double(
    struct mtxfile * mtxfile,
    int64_t num_rows,
    const double * data);

/**
 * ‘mtxfile_init_vector_array_complex_single()’ allocates and
 * initialises a vector in array format with complex, single precision
 * coefficients.
 */
int LIBMTX_API mtxfile_init_vector_array_complex_single(
    struct mtxfile * mtxfile,
    int64_t num_rows,
    const float (* data)[2]);

/**
 * ‘mtxfile_init_vector_array_complex_double()’ allocates and
 * initialises a vector in array format with complex, double precision
 * coefficients.
 */
int LIBMTX_API mtxfile_init_vector_array_complex_double(
    struct mtxfile * mtxfile,
    int64_t num_rows,
    const double (* data)[2]);

/**
 * ‘mtxfile_init_vector_array_integer_single()’ allocates and
 * initialises a vector in array format with integer, single precision
 * coefficients.
 */
int LIBMTX_API mtxfile_init_vector_array_integer_single(
    struct mtxfile * mtxfile,
    int64_t num_rows,
    const int32_t * data);

/**
 * ‘mtxfile_init_vector_array_integer_double()’ allocates and
 * initialises a vector in array format with integer, double precision
 * coefficients.
 */
int LIBMTX_API mtxfile_init_vector_array_integer_double(
    struct mtxfile * mtxfile,
    int64_t num_rows,
    const int64_t * data);

/*
 * Matrix coordinate formats
 */

/**
 * ‘mtxfile_alloc_matrix_coordinate()’ allocates a matrix in
 * coordinate format.
 */
int LIBMTX_API mtxfile_alloc_matrix_coordinate(
    struct mtxfile * mtxfile,
    enum mtxfilefield field,
    enum mtxfilesymmetry symmetry,
    enum mtxprecision precision,
    int64_t num_rows,
    int64_t num_columns,
    int64_t num_nonzeros);

/**
 * ‘mtxfile_init_matrix_coordinate_real_single()’ allocates and initialises
 * a matrix in coordinate format with real, single precision coefficients.
 */
int LIBMTX_API mtxfile_init_matrix_coordinate_real_single(
    struct mtxfile * mtxfile,
    enum mtxfilesymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    int64_t num_nonzeros,
    const struct mtxfile_matrix_coordinate_real_single * data);

/**
 * ‘mtxfile_init_matrix_coordinate_real_double()’ allocates and initialises
 * a matrix in coordinate format with real, double precision coefficients.
 */
int LIBMTX_API mtxfile_init_matrix_coordinate_real_double(
    struct mtxfile * mtxfile,
    enum mtxfilesymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    int64_t num_nonzeros,
    const struct mtxfile_matrix_coordinate_real_double * data);

/**
 * ‘mtxfile_init_matrix_coordinate_complex_single()’ allocates and
 * initialises a matrix in coordinate format with complex, single precision
 * coefficients.
 */
int LIBMTX_API mtxfile_init_matrix_coordinate_complex_single(
    struct mtxfile * mtxfile,
    enum mtxfilesymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    int64_t num_nonzeros,
    const struct mtxfile_matrix_coordinate_complex_single * data);

/**
 * ‘mtxfile_init_matrix_coordinate_complex_double()’ allocates and
 * initialises a matrix in coordinate format with complex, double precision
 * coefficients.
 */
int LIBMTX_API mtxfile_init_matrix_coordinate_complex_double(
    struct mtxfile * mtxfile,
    enum mtxfilesymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    int64_t num_nonzeros,
    const struct mtxfile_matrix_coordinate_complex_double * data);

/**
 * ‘mtxfile_init_matrix_coordinate_integer_single()’ allocates and
 * initialises a matrix in coordinate format with integer, single precision
 * coefficients.
 */
int LIBMTX_API mtxfile_init_matrix_coordinate_integer_single(
    struct mtxfile * mtxfile,
    enum mtxfilesymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    int64_t num_nonzeros,
    const struct mtxfile_matrix_coordinate_integer_single * data);

/**
 * ‘mtxfile_init_matrix_coordinate_integer_double()’ allocates and
 * initialises a matrix in coordinate format with integer, double precision
 * coefficients.
 */
int LIBMTX_API mtxfile_init_matrix_coordinate_integer_double(
    struct mtxfile * mtxfile,
    enum mtxfilesymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    int64_t num_nonzeros,
    const struct mtxfile_matrix_coordinate_integer_double * data);

/**
 * ‘mtxfile_init_matrix_coordinate_pattern()’ allocates and
 * initialises a matrix in coordinate format with boolean (pattern)
 * coefficients.
 */
int LIBMTX_API mtxfile_init_matrix_coordinate_pattern(
    struct mtxfile * mtxfile,
    enum mtxfilesymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    int64_t num_nonzeros,
    const struct mtxfile_matrix_coordinate_pattern * data);

/*
 * Vector coordinate formats
 */

/**
 * ‘mtxfile_alloc_vector_coordinate()’ allocates a vector in
 * coordinate format.
 */
int LIBMTX_API mtxfile_alloc_vector_coordinate(
    struct mtxfile * mtxfile,
    enum mtxfilefield field,
    enum mtxprecision precision,
    int64_t num_rows,
    int64_t num_nonzeros);

/**
 * ‘mtxfile_init_vector_coordinate_real_single()’ allocates and initialises
 * a vector in coordinate format with real, single precision coefficients.
 */
int LIBMTX_API mtxfile_init_vector_coordinate_real_single(
    struct mtxfile * mtxfile,
    int64_t num_rows,
    int64_t num_nonzeros,
    const struct mtxfile_vector_coordinate_real_single * data);

/**
 * ‘mtxfile_init_vector_coordinate_real_double()’ allocates and initialises
 * a vector in coordinate format with real, double precision coefficients.
 */
int LIBMTX_API mtxfile_init_vector_coordinate_real_double(
    struct mtxfile * mtxfile,
    int64_t num_rows,
    int64_t num_nonzeros,
    const struct mtxfile_vector_coordinate_real_double * data);

/**
 * ‘mtxfile_init_vector_coordinate_complex_single()’ allocates and
 * initialises a vector in coordinate format with complex, single precision
 * coefficients.
 */
int LIBMTX_API mtxfile_init_vector_coordinate_complex_single(
    struct mtxfile * mtxfile,
    int64_t num_rows,
    int64_t num_nonzeros,
    const struct mtxfile_vector_coordinate_complex_single * data);

/**
 * ‘mtxfile_init_vector_coordinate_complex_double()’ allocates and
 * initialises a vector in coordinate format with complex, double precision
 * coefficients.
 */
int LIBMTX_API mtxfile_init_vector_coordinate_complex_double(
    struct mtxfile * mtxfile,
    int64_t num_rows,
    int64_t num_nonzeros,
    const struct mtxfile_vector_coordinate_complex_double * data);

/**
 * ‘mtxfile_init_vector_coordinate_integer_single()’ allocates and
 * initialises a vector in coordinate format with integer, single precision
 * coefficients.
 */
int LIBMTX_API mtxfile_init_vector_coordinate_integer_single(
    struct mtxfile * mtxfile,
    int64_t num_rows,
    int64_t num_nonzeros,
    const struct mtxfile_vector_coordinate_integer_single * data);

/**
 * ‘mtxfile_init_vector_coordinate_integer_double()’ allocates and
 * initialises a vector in coordinate format with integer, double precision
 * coefficients.
 */
int LIBMTX_API mtxfile_init_vector_coordinate_integer_double(
    struct mtxfile * mtxfile,
    int64_t num_rows,
    int64_t num_nonzeros,
    const struct mtxfile_vector_coordinate_integer_double * data);

/**
 * ‘mtxfile_init_vector_coordinate_pattern()’ allocates and
 * initialises a vector in coordinate format with boolean (pattern)
 * precision coefficients.
 */
int LIBMTX_API mtxfile_init_vector_coordinate_pattern(
    struct mtxfile * mtxfile,
    int64_t num_rows,
    int64_t num_nonzeros,
    const struct mtxfile_vector_coordinate_pattern * data);

/*
 * Modifying values
 */

/**
 * ‘mtxfile_set_constant_real_single()’ sets every (nonzero) value of
 * a matrix or vector equal to a constant, single precision floating
 * point number.
 */
int LIBMTX_API mtxfile_set_constant_real_single(
    struct mtxfile * mtxfile,
    float a);

/**
 * ‘mtxfile_set_constant_real_double()’ sets every (nonzero) value of
 * a matrix or vector equal to a constant, double precision floating
 * point number.
 */
int LIBMTX_API mtxfile_set_constant_real_double(
    struct mtxfile * mtxfile,
    double a);

/**
 * ‘mtxfile_set_constant_complex_single()’ sets every (nonzero) value
 * of a matrix or vector equal to a constant, single precision
 * floating point complex number.
 */
int LIBMTX_API mtxfile_set_constant_complex_single(
    struct mtxfile * mtxfile,
    float a[2]);

/**
 * ‘mtxfile_set_constant_complex_double()’ sets every (nonzero) value
 * of a matrix or vector equal to a constant, double precision
 * floating point complex number.
 */
int LIBMTX_API mtxfile_set_constant_complex_double(
    struct mtxfile * mtxfile,
    double a[2]);

/**
 * ‘mtxfile_set_constant_integer_single()’ sets every (nonzero) value
 * of a matrix or vector equal to a constant integer.
 */
int LIBMTX_API mtxfile_set_constant_integer_single(
    struct mtxfile * mtxfile,
    int32_t a);

/**
 * ‘mtxfile_set_constant_integer_double()’ sets every (nonzero) value
 * of a matrix or vector equal to a constant integer.
 */
int LIBMTX_API mtxfile_set_constant_integer_double(
    struct mtxfile * mtxfile,
    int64_t a);

/*
 * I/O functions
 */

/**
 * ‘mtxfile_read()’ reads a Matrix Market file from the given path.
 * The file may optionally be compressed by gzip.
 *
 * The ‘precision’ argument specifies which precision to use for
 * storing matrix or vector values.
 *
 * If ‘path’ is ‘-’, then standard input is used.
 *
 * The file is assumed to be gzip-compressed if ‘gzip’ is ‘true’, and
 * uncompressed otherwise.
 *
 * If an error code is returned, then ‘lines_read’ and ‘bytes_read’
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the Matrix Market file.
 */
int LIBMTX_API mtxfile_read(
    struct mtxfile * mtxfile,
    enum mtxprecision precision,
    const char * path,
    bool gzip,
    int64_t * lines_read,
    int64_t * bytes_read);

/**
 * ‘mtxfile_fread()’ reads a Matrix Market file from a stream.
 *
 * ‘precision’ is used to determine the precision to use for storing
 * the values of matrix or vector entries.
 *
 * If an error code is returned, then ‘lines_read’ and ‘bytes_read’
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the Matrix Market file.
 *
 * If ‘linebuf’ is not ‘NULL’, then it must point to an array that can
 * hold at least ‘line_max’ values of type ‘char’. This buffer is used
 * for reading lines from the stream. Otherwise, if ‘linebuf’ is
 * ‘NULL’, then a temporary buffer is allocated and used, and the
 * maximum line length is determined by calling ‘sysconf()’ with
 * ‘_SC_LINE_MAX’.
 */
int LIBMTX_API mtxfile_fread(
    struct mtxfile * mtxfile,
    enum mtxprecision precision,
    FILE * f,
    int64_t * lines_read,
    int64_t * bytes_read,
    size_t line_max,
    char * linebuf);

#ifdef LIBMTX_HAVE_LIBZ
/**
 * ‘mtxfile_gzread()’ reads a Matrix Market file from a
 * gzip-compressed stream.
 *
 * ‘precision’ is used to determine the precision to use for storing
 * the values of matrix or vector entries.
 *
 * If an error code is returned, then ‘lines_read’ and ‘bytes_read’
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the Matrix Market file.
 *
 * If ‘linebuf’ is not ‘NULL’, then it must point to an array that can
 * hold at least ‘line_max’ values of type ‘char’. This buffer is used
 * for reading lines from the stream. Otherwise, if ‘linebuf’ is
 * ‘NULL’, then a temporary buffer is allocated and used, and the
 * maximum line length is determined by calling ‘sysconf()’ with
 * ‘_SC_LINE_MAX’.
 */
int LIBMTX_API mtxfile_gzread(
    struct mtxfile * mtxfile,
    enum mtxprecision precision,
    gzFile f,
    int64_t * lines_read,
    int64_t * bytes_read,
    size_t line_max,
    char * linebuf);
#endif

/**
 * ‘mtxfile_write()’ writes a Matrix Market file to the given path.
 * The file may optionally be compressed by gzip.
 *
 * If ‘path’ is ‘-’, then standard output is used.
 *
 * If ‘fmt’ is ‘NULL’, then the format specifier ‘%g’ is used to print
 * floating point numbers with enough digits to ensure correct
 * round-trip conversion from decimal text and back.  Otherwise, the
 * given format string is used to print numerical values.
 *
 * The format string ‘fmt’ follows the conventions of ‘printf’. If the
 * field of ‘mtxfile’ is ‘mtxfile_real’ or ‘mtxfile_complex’, then the
 * format specifiers '%e', '%E', '%f', '%F', '%g' or '%G' may be
 * used. If the field is ‘mtxfile_integer’, then the format specifier
 * must be '%d'. The format string is ignored if the field is
 * ‘mtxfile_pattern’. Field width and precision may be specified
 * (e.g., "%3.1f"), but variable field width and precision (e.g.,
 * "%*.*f") or length modifiers (e.g., "%Lf") are not allowed.
 *
 * The locale is temporarily changed to "C" to ensure that
 * locale-specific settings, such as the type of decimal point, do not
 * affect output.
 */
int LIBMTX_API mtxfile_write(
    const struct mtxfile * mtxfile,
    const char * path,
    bool gzip,
    const char * fmt,
    int64_t * bytes_written);

/**
 * ‘mtxfile_fwrite()’ writes a Matrix Market file to a stream.
 *
 * If ‘fmt’ is ‘NULL’, then the format specifier ‘%g’ is used to print
 * floating point numbers with enough digits to ensure correct
 * round-trip conversion from decimal text and back.  Otherwise, the
 * given format string is used to print numerical values.
 *
 * The format string ‘fmt’ follows the conventions of ‘printf’. If the
 * field of ‘mtxfile’ is ‘mtxfile_real’ or ‘mtxfile_complex’, then the
 * format specifiers '%e', '%E', '%f', '%F', '%g' or '%G' may be
 * used. If the field is ‘mtxfile_integer’, then the format specifier
 * must be '%d'. The format string is ignored if the field is
 * ‘mtxfile_pattern’. Field width and precision may be specified
 * (e.g., "%3.1f"), but variable field width and precision (e.g.,
 * "%*.*f") or length modifiers (e.g., "%Lf") are not allowed.
 *
 * If it is not ‘NULL’, then the number of bytes written to the stream
 * is returned in ‘bytes_written’.
 *
 * The locale is temporarily changed to "C" to ensure that
 * locale-specific settings, such as the type of decimal point, do not
 * affect output.
 */
int LIBMTX_API mtxfile_fwrite(
    const struct mtxfile * mtxfile,
    FILE * f,
    const char * fmt,
    int64_t * bytes_written);

#ifdef LIBMTX_HAVE_LIBZ
/**
 * ‘mtxfile_gzwrite()’ writes a Matrix Market file to a
 * gzip-compressed stream.
 *
 * If ‘fmt’ is ‘NULL’, then the format specifier ‘%g’ is used to print
 * floating point numbers with enough digits to ensure correct
 * round-trip conversion from decimal text and back.  Otherwise, the
 * given format string is used to print numerical values.
 *
 * The format string ‘fmt’ follows the conventions of ‘printf’. If the
 * field of ‘mtxfile’ is ‘mtxfile_real’ or ‘mtxfile_complex’, then the
 * format specifiers '%e', '%E', '%f', '%F', '%g' or '%G' may be
 * used. If the field is ‘mtxfile_integer’, then the format specifier
 * must be '%d'. The format string is ignored if the field is
 * ‘mtxfile_pattern’. Field width and precision may be specified
 * (e.g., "%3.1f"), but variable field width and precision (e.g.,
 * "%*.*f") or length modifiers (e.g., "%Lf") are not allowed.
 *
 * If it is not ‘NULL’, then the number of bytes written to the stream
 * is returned in ‘bytes_written’.
 *
 * The locale is temporarily changed to "C" to ensure that
 * locale-specific settings, such as the type of decimal point, do not
 * affect output.
 */
int LIBMTX_API mtxfile_gzwrite(
    const struct mtxfile * mtxfile,
    gzFile f,
    const char * fmt,
    int64_t * bytes_written);
#endif

/*
 * Transpose and conjugate transpose.
 */

/**
 * ‘mtxfile_transpose()’ tranposes a Matrix Market file.
 */
int LIBMTX_API mtxfile_transpose(
    struct mtxfile * mtxfile);

/**
 * ‘mtxfile_conjugate_transpose()’ tranposes and complex conjugates a
 * Matrix Market file.
 */
int mtxfile_conjugate_transpose(
    struct mtxfile * mtxfile);

/*
 * Sorting
 */

/**
 * ‘mtxfilesorting’ is used to enumerate different ways of sorting
 * Matrix Market files.
 */
enum mtxfilesorting
{
    mtxfile_unsorted,            /* unsorted (default ordering) */
    mtxfile_permutation,         /* user-defined sorting permutation */
    mtxfile_row_major,           /* row major ordering */
    mtxfile_column_major,        /* column major ordering */
    mtxfile_morton,              /* Morton (Z-order curve) ordering */
};

/**
 * ‘mtxfilesortingstr()’ is a string representing the sorting of a
 * matrix or vector in Matrix Market format.
 */
LIBMTX_API const char * mtxfilesortingstr(
    enum mtxfilesorting sorting);

/**
 * ‘mtxfilesorting_parse()’ parses a string corresponding to a value
 * of the enum type ‘mtxfilesorting’.
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
 * On success, ‘mtxfilesorting_parse()’ returns ‘MTX_SUCCESS’ and
 * ‘sorting’ is set according to the parsed string and ‘bytes_read’ is
 * set to the number of bytes that were consumed by the parser.
 * Otherwise, an error code is returned.
 */
int mtxfilesorting_parse(
    enum mtxfilesorting * sorting,
    int64_t * bytes_read,
    const char ** endptr,
    const char * s,
    const char * valid_delimiters);

/**
 * ‘mtxfile_sort()’ sorts a Matrix Market file in a given order.
 *
 * The sorting order is determined by ‘sorting’. If the sorting order
 * is ‘mtxfile_unsorted’, nothing is done. If the sorting order is
 * ‘mtxfile_permutation’, then ‘perm’ must point to an array of ‘size’
 * integers that specify the sorting permutation. Note that the
 * sorting permutation uses 1-based indexing.
 *
 * For a vector or matrix in coordinate format, the nonzero values are
 * sorted in the specified order. For Matrix Market files in array
 * format, this operation does nothing.
 *
 * ‘perm’ is ignored if it is ‘NULL’. Otherwise, it must point to an
 * array of length ‘size’, which is used to store the permutation of
 * the Matrix Market entries. ‘size’ must therefore be at least equal
 * to the number of data lines in the Matrix Market file ‘mtxfile’.
 */
int LIBMTX_API mtxfile_sort(
    struct mtxfile * mtxfile,
    enum mtxfilesorting sorting,
    int64_t size,
    int64_t * perm);

/**
 * ‘mtxfile_compact()’ compacts a Matrix Market file in coordinate
 * format by merging adjacent, duplicate entries.
 *
 * For a matrix or vector in array format, this does nothing.
 *
 * The number of nonzero matrix or vector entries,
 * ‘mtxfile->size.num_nonzeros’, is updated to reflect entries that
 * were removed as a result of compacting. However, the underlying
 * storage is not changed or reallocated. This may result in large
 * amounts of unused memory, if a large number of entries were
 * removed. If necessary, it is possible to allocate new storage, copy
 * the compacted data, and, finally, free the old storage.
 *
 * If ‘perm’ is not ‘NULL’, then it must point to an array of length
 * ‘size’. The ‘i’th entry of ‘perm’ is used to store the index of the
 * corresponding data line in the compacted array that the ‘i’th data
 * line was moved to or merged with. Note that the indexing is
 * 1-based.
 */
int LIBMTX_API mtxfile_compact(
    struct mtxfile * mtxfile,
    int64_t size,
    int64_t * perm);

/**
 * ‘mtxfile_assemble()’ assembles a Matrix Market file in coordinate
 * format by merging duplicate entries. The file may optionally be
 * sorted at the same time.
 *
 * For a matrix or vector in array format, this does nothing.
 *
 * The number of nonzero matrix or vector entries,
 * ‘mtxfile->size.num_nonzeros’, is updated to reflect entries that
 * were removed as a result of compacting. However, the underlying
 * storage is not changed or reallocated. This may result in large
 * amounts of unused memory, if a large number of entries were
 * removed. If necessary, it is possible to allocate new storage, copy
 * the compacted data, and, finally, free the old storage.
 *
 * If ‘perm’ is not ‘NULL’, then it must point to an array of length
 * ‘size’. The ‘i’th entry of ‘perm’ is used to store the index of the
 * corresponding data line in the sorted and compacted array that the
 * ‘i’th data line was moved to or merged with. Note that the indexing
 * is 1-based.
 */
int LIBMTX_API mtxfile_assemble(
    struct mtxfile * mtxfile,
    enum mtxfilesorting sorting,
    int64_t size,
    int64_t * perm);

/*
 * Partitioning
 */

/**
 * ‘mtxfile_partition_nonzeros()’ partitions the nonzeros of a Matrix
 * Market file.
 *
 * See ‘partition_int64()’ for an explanation of the meaning of the
 * arguments ‘parttype’, ‘num_parts’, ‘partsizes’, ‘blksize’ and
 * ‘parts’.
 *
 * The array ‘dstpart’ must contain enough storage for
 * ‘mtxfile->datasize’ values of type ‘int’. If successful, ‘dstpart’
 * is used to store the part number assigned to the matrix or vector
 * nonzeros.
 *
 * If ‘dstpartsizes’ is not ‘NULL’, then it must be an array of length
 * ‘num_parts’, which is used to store the number of items assigned to
 * each part.
 */
int LIBMTX_API mtxfile_partition_nonzeros(
    const struct mtxfile * mtxfile,
    enum mtxpartitioning parttype,
    int num_parts,
    const int64_t * partsizes,
    int64_t blksize,
    const int * parts,
    int * dstpart,
    int64_t * dstpartsizes);

/**
 * ‘mtxfile_partition_rowwise()’ partitions the entries of a Matrix
 * Market file according to a given row partitioning.
 *
 * See ‘partition_int64()’ for an explanation of the meaning of the
 * arguments ‘parttype’, ‘num_parts’, ‘partsizes’, ‘blksize’ and
 * ‘parts’.
 *
 * The array ‘dstpart’ must contain enough storage for
 * ‘mtxfile->datasize’ values of type ‘int’. If successful, ‘dstpart’
 * is used to store the part number assigned to the matrix or vector
 * nonzeros.
 *
 * If ‘dstpartsizes’ is not ‘NULL’, then it must be an array of length
 * ‘num_parts’, which is used to store the number of items assigned to
 * each part.
 */
int LIBMTX_API mtxfile_partition_rowwise(
    const struct mtxfile * mtxfile,
    enum mtxpartitioning parttype,
    int num_parts,
    const int64_t * partsizes,
    int64_t blksize,
    const int * parts,
    int * dstpart,
    int64_t * dstpartsizes);

/**
 * ‘mtxfile_partition_columnwise()’ partitions the entries of a Matrix
 * Market file according to a given column partitioning.
 *
 * See ‘partition_int64()’ for an explanation of the meaning of the
 * arguments ‘parttype’, ‘num_parts’, ‘partsizes’, ‘blksize’ and
 * ‘parts’.
 *
 * The array ‘dstpart’ must contain enough storage for
 * ‘mtxfile->datasize’ values of type ‘int’. If successful, ‘dstpart’
 * is used to store the part number assigned to the matrix or vector
 * nonzeros.
 *
 * If ‘dstpartsizes’ is not ‘NULL’, then it must be an array of length
 * ‘num_parts’, which is used to store the number of items assigned to
 * each part.
 */
int LIBMTX_API mtxfile_partition_columnwise(
    const struct mtxfile * mtxfile,
    enum mtxpartitioning parttype,
    int num_parts,
    const int64_t * partsizes,
    int64_t blksize,
    const int * parts,
    int * dstpart,
    int64_t * dstpartsizes);

/**
 * ‘mtxfile_partition_2d()’ partitions a matrix in Matrix Market
 * format according to given row and column partitionings.
 *
 * The number of parts is equal to the product of ‘num_row_parts’ and
 * ‘num_column_parts’.
 *
 * See ‘partition_int64()’ for an explanation of the meaning of the
 * arguments ‘rowparttype’, ‘num_row_parts’, ‘rowpartsizes’ and
 * ‘rowblksize’, ‘rowparts’, and so on.
 *
 * The array ‘dstpart’ must contain enough storage for
 * ‘mtxfile->datasize’ values of type ‘int’. If successful, ‘dstpart’
 * is used to store the part number assigned to the matrix or vector
 * nonzeros.
 *
 * If ‘dstpartsizes’ is not ‘NULL’, then it must be an array of length
 * ‘num_parts’, which is used to store the number of items assigned to
 * each part.
 */
int LIBMTX_API mtxfile_partition_2d(
    const struct mtxfile * mtxfile,
    enum mtxpartitioning rowparttype,
    int num_row_parts,
    const int64_t * rowpartsizes,
    int64_t rowblksize,
    const int * rowparts,
    enum mtxpartitioning colparttype,
    int num_column_parts,
    const int64_t * colpartsizes,
    int64_t colblksize,
    const int * colparts,
    int * dstpart,
    int64_t * dstpartsizes);

/**
 * ‘mtxfile_partition()’ partitions the entries of a Matrix Market
 * file, and, optionally, also partitions the rows and columns of the
 * underlying matrix or vector.
 *
 * The type of partitioning to perform is determined by
 * ‘matrixparttype’.
 *
 *  - If ‘matrixparttype’ is ‘mtx_matrixparttype_nonzeros’, the
 *    nonzeros of the underlying matrix or vector are partitioned as a
 *    one-dimensional array. The nonzeros are partitioned into
 *    ‘num_nz_parts’ parts according to the partitioning
 *    ‘nzparttype’. If ‘nzparttype’ is ‘mtx_block’, then ‘nzpartsizes’
 *    may be used to specify the size of each part. If ‘nzparttype’ is
 *    ‘mtx_block_cyclic’, then ‘nzblksize’ is used to specify the
 *    block size.
 *
 *  - If ‘matrixparttype’ is ‘mtx_matrixparttype_rows’, the nonzeros
 *    of the underlying matrix or vector are partitioned rowwise.
 *
 *  - If ‘matrixparttype’ is ‘mtx_matrixparttype_columns’, the
 *    nonzeros of the underlying matrix are partitioned columnwise.
 *
 *  - If ‘matrixparttype’ is ‘mtx_matrixparttype_2d’, the nonzeros of
 *    the underlying matrix are partitioned in rectangular blocks
 *    according to the partitioning of the rows and columns.
 *
 *  - If ‘matrixparttype’ is ‘mtx_matrixparttype_metis’, then the rows
 *    and columns of the underlying matrix are partitioned by the
 *    METIS graph partitioner, and the matrix nonzeros are partitioned
 *    accordingly.
 *
 *
 * In any case, the array ‘dstnzpart’ is used to store the part
 * numbers assigned to the matrix nonzeros and must therefore be of
 * length ‘mtxfile->datasize’.
 *
 * If the rows are partitioned, then the array ‘dstrowpart’ must be of
 * length ‘mtxfile->size.num_rows’. This array is used to store the
 * part numbers assigned to the matrix rows. In this case, ‘*rowpart’
 * is also set to ‘true’, whereas it is ‘false’ otherwise.
 *
 * Similarly, if the columns are partitioned (e.g., when partitioning
 * columnwise, 2d or a graph-based partitioning of a non-square
 * matrix), then ‘dstcolpart’ is used to store the part numbers
 * assigned to the matrix columns, and it must therefore be an array
 * of length ‘mtxfile->size.num_columns’. Moreover, ‘*colpart’ is set
 * to ‘true’, whereas it is ‘false’ otherwise.
 *
 * Unless they are set to ‘NULL’, then ‘dstnzpartsizes’,
 * ‘dstrowpartsizes’ and ‘dstcolpartsizes’ must be arrays of length
 * ‘num_parts’, which are then used to store the number of nonzeros,
 * rows and columns assigned to each part, respectively.
 *
 * If ‘matrixparttype’ is ‘matrixparttype_metis’ and ‘objval’ is not
 * ‘NULL’, then it is used to store the value of the objective
 * function minimized by the partitioner, which, by default, is the
 * edge-cut of the partitioning solution.
 */
int LIBMTX_API mtxfile_partition(
    struct mtxfile * mtxfile,
    enum mtxmatrixparttype matrixparttype,
    enum mtxpartitioning nzparttype,
    int num_nz_parts,
    const int64_t * nzpartsizes,
    int64_t nzblksize,
    const int * nzparts,
    enum mtxpartitioning rowparttype,
    int num_row_parts,
    const int64_t * rowpartsizes,
    int64_t rowblksize,
    const int * rowparts,
    enum mtxpartitioning colparttype,
    int num_column_parts,
    const int64_t * colpartsizes,
    int64_t colblksize,
    const int * colparts,
    int * dstnzpart,
    int64_t * dstnzpartsizes,
    bool * rowpart,
    int * dstrowpart,
    int64_t * dstrowpartsizes,
    bool * colpart,
    int * dstcolpart,
    int64_t * dstcolpartsizes,
    int64_t * objval,
    int verbose);

/**
 * ‘mtxfile_split()’ splits a Matrix Market file into several files
 * according to a given partition of the underlying (nonzero) matrix
 * or vector elements.
 *
 * The partitioning of the matrix or vector elements is specified by
 * the array ‘parts’. The length of the ‘parts’ array is given by
 * ‘size’, which must match the number of (nonzero) matrix or vector
 * elements in ‘src’. Each entry in the array is an integer in the
 * range ‘[0, num_parts)’ designating the part to which the
 * corresponding nonzero element belongs.
 *
 * The argument ‘dsts’ is an array of ‘num_parts’ pointers to objects
 * of type ‘struct mtxfile’. If successful, then ‘dsts[p]’ points to a
 * matrix market file consisting of (nonzero) elements from ‘src’ that
 * belong to the ‘p’th part, according to the ‘parts’ array.
 *
 * If ‘src’ is a matrix (or vector) in coordinate format, then each of
 * the matrices or vectors in ‘dsts’ is also a matrix (or vector) in
 * coordinate format with the same number of rows and columns as
 * ‘src’. In this case, the arguments ‘num_rows_per_part’ and
 * ‘num_columns_per_part’ are not used and may be set to ‘NULL’.
 *
 * Otherwise, if ‘src’ is a matrix (or vector) in array format, then
 * the arrays ‘num_rows_per_part’ and ‘num_columns_per_part’ (both of
 * length ‘num_parts’) are used to specify the dimensions of each
 * matrix (or vector) in ‘dsts’. For a given part ‘p’, the number of
 * matrix (or vector) elements assigned to that part must be equal to
 * the product of ‘num_rows_per_part[p]’ and
 * ‘num_columns_per_part[p]’.
 *
 * The user is responsible for freeing storage allocated for each
 * Matrix Market file in the ‘dsts’ array.
 */
int LIBMTX_API mtxfile_split(
    int num_parts,
    struct mtxfile ** dsts,
    const struct mtxfile * src,
    int64_t size,
    int * parts,
    const int64_t * num_rows_per_part,
    const int64_t * num_columns_per_part);

/*
 * Reordering
 */

/**
 * ‘mtxfile_permute()’ reorders the rows of a vector or the rows and
 *  columns of a matrix in Matrix Market format based on given row and
 *  column permutations.
 *
 * The array ‘row_permutation’ should be a permutation of the integers
 * ‘1,2,...,M’, where ‘M’ is the number of rows in the matrix or
 * vector.  If the Matrix Market file represents a matrix, then the
 * array ‘column_permutation’ should be a permutation of the integers
 * ‘1,2,...,N’, where ‘N’ is the number of columns in the matrix.  The
 * elements belonging to row ‘i’ and column ‘j’ in the permuted matrix
 * are then equal to the elements in row ‘row_permutation[i-1]’ and
 * column ‘column_permutation[j-1]’ in the original matrix, for
 * ‘i=1,2,...,M’ and ‘j=1,2,...,N’.
 */
int LIBMTX_API mtxfile_permute(
    struct mtxfile * mtxfile,
    const int * row_permutation,
    const int * column_permutation);

/**
 * ‘mtxfileordering’ is used to enumerate different kinds of orderings
 * for matrices in Matrix Market format.
 */
enum mtxfileordering
{
    mtxfile_default_order, /* default ordering of the Matrix Market file */
    mtxfile_custom_order,  /* general, user-defined ordering */
    mtxfile_rcm,           /* Reverse Cuthill-McKee ordering */
    mtxfile_nd,            /* nested dissection ordering */
    mtxfile_metis,         /* graph partitioning reordering with METIS */
};

/**
 * ‘mtxfileorderingstr()’ is a string representing the ordering of a
 *  Matrix Market file.
 */
LIBMTX_API const char * mtxfileorderingstr(
    enum mtxfileordering ordering);

/**
 * ‘mtxfileordering_parse()’ parses a string corresponding to a value
 *  of the enum type ‘mtxfileordering’.
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
 * On success, ‘mtxfileordering_parse()’ returns ‘MTX_SUCCESS’ and
 * ‘ordering’ is set according to the parsed string and ‘bytes_read’
 * is set to the number of bytes that were consumed by the parser.
 * Otherwise, an error code is returned.
 */
int mtxfileordering_parse(
    enum mtxfileordering * ordering,
    int64_t * bytes_read,
    const char ** endptr,
    const char * s,
    const char * valid_delimiters);

/**
 * ‘mtxfile_reorder_metis()’ reorders the rows and columns of a sparse
 * matrix based on a graph partitioning performed with METIS.
 *
 * For a square matrix, the reordering algorithm is carried out on the
 * adjacency matrix of the symmetrisation ‘A+A'’, where ‘A'’ denotes
 * the transpose of ‘A’. For a rectangular matrix, the reordering is
 * carried out on a bipartite graph formed by the matrix rows and
 * columns. The adjacency matrix ‘B’ of the bipartite graph is square
 * and symmetric and takes the form of a 2-by-2 block matrix where ‘A’
 * is placed in the upper right corner and ‘A'’ is placed in the lower
 * left corner:
 *
 *     ⎡  0   A ⎤
 * B = ⎢        ⎥.
 *     ⎣  A'  0 ⎦
 *
 * The reordering is symmetric if the matrix is square, and
 * unsymmetric otherwise.
 *
 * If successful, this function returns ‘MTX_SUCCESS’, and the rows
 * and columns of ‘mtxfile’ have been reordered. If ‘rowperm’ is not
 * ‘NULL’, then it must point to an array that is large enough to hold
 * one ‘int’ for each row of the matrix. In this case, the array is
 * used to store the permutation for reordering the matrix
 * rows. Similarly, ‘colperm’ is used to store the permutation for
 * reordering the matrix columns.
 *
 * If it is not ‘NULL’, then ‘objval’ is used to store the value of
 * the objective function minimized by the partitioner, which, by
 * default, is the edge-cut of the partitioning solution.
 */
int LIBMTX_API mtxfile_reorder_metis(
    struct mtxfile * mtxfile,
    int * rowperm,
    int * rowperminv,
    int * colperm,
    int * colperminv,
    bool permute,
    bool * symmetric,
    int nparts,
    int * rowpartsizes,
    int * colpartsizes,
    int64_t * objval,
    int verbose);

/**
 * ‘mtxfile_reorder_nd()’ reorders the rows and columns of a sparse
 * matrix according to the nested dissection algorithm.
 *
 * For a square matrix, the reordering algorithm is carried out on the
 * adjacency matrix of the symmetrisation ‘A+A'’, where ‘A'’ denotes
 * the transpose of ‘A’. For a rectangular matrix, the reordering is
 * carried out on a bipartite graph formed by the matrix rows and
 * columns. The adjacency matrix ‘B’ of the bipartite graph is square
 * and symmetric and takes the form of a 2-by-2 block matrix where ‘A’
 * is placed in the upper right corner and ‘A'’ is placed in the lower
 * left corner:
 *
 *     ⎡  0   A ⎤
 * B = ⎢        ⎥.
 *     ⎣  A'  0 ⎦
 *
 * The reordering is symmetric if the matrix is square, and
 * unsymmetric otherwise.
 *
 * If successful, this function returns ‘MTX_SUCCESS’, and the rows
 * and columns of ‘mtxfile’ have been reordered. If ‘rowperm’ is not
 * ‘NULL’, then it must point to an array that is large enough to hold
 * one ‘int’ for each row of the matrix. In this case, the array is
 * used to store the permutation for reordering the matrix
 * rows. Similarly, ‘colperm’ is used to store the permutation for
 * reordering the matrix columns.
 */
int LIBMTX_API mtxfile_reorder_nd(
    struct mtxfile * mtxfile,
    int * rowperm,
    int * rowperminv,
    int * colperm,
    int * colperminv,
    bool permute,
    bool * symmetric,
    int verbose);

/**
 * ‘mtxfile_reorder_rcm()’ reorders the rows of a sparse matrix
 * according to the Reverse Cuthill-McKee algorithm.
 *
 * For a square matrix, the Cuthill-McKee algorithm is carried out on
 * the adjacency matrix of the symmetrisation ‘A+A'’, where ‘A'’
 * denotes the transpose of ‘A’.  For a rectangular matrix, the
 * Cuthill-McKee algorithm is carried out on a bipartite graph formed
 * by the matrix rows and columns.  The adjacency matrix ‘B’ of the
 * bipartite graph is square and symmetric and takes the form of a
 * 2-by-2 block matrix where ‘A’ is placed in the upper right corner
 * and ‘A'’ is placed in the lower left corner:
 *
 *     ⎡  0   A ⎤
 * B = ⎢        ⎥.
 *     ⎣  A'  0 ⎦
 *
 * ‘starting_vertex’ is a pointer to an integer which can be used to
 * designate a starting vertex for the Cuthill-McKee algorithm.
 * Alternatively, setting the starting_vertex to zero causes a
 * starting vertex to be chosen automatically by selecting a
 * pseudo-peripheral vertex.
 *
 * In the case of a square matrix, the starting vertex must be in the
 * range ‘[1,M]’, where ‘M’ is the number of rows (and columns) of the
 * matrix.  Otherwise, if the matrix is rectangular, a starting vertex
 * in the range ‘[1,M]’ selects a vertex corresponding to a row of the
 * matrix, whereas a starting vertex in the range ‘[M+1,M+N]’, where
 * ‘N’ is the number of matrix columns, selects a vertex corresponding
 * to a column of the matrix.
 *
 * The reordering is symmetric if the matrix is square, and
 * unsymmetric otherwise.
 *
 * If successful, this function returns ‘MTX_SUCCESS’, and the rows
 * and columns of ‘mtxfile’ have been reordered according to the
 * Reverse Cuthill-McKee algorithm. If ‘rowperm’ is not ‘NULL’, then
 * it must point to an array that is large enough to hold one ‘int’
 * for each row of the matrix. In this case, the array is used to
 * store the permutation for reordering the matrix rows. Similarly,
 * ‘colperm’ is used to store the permutation for reordering the
 * matrix columns.
 */
int LIBMTX_API mtxfile_reorder_rcm(
    struct mtxfile * mtxfile,
    int * rowperm,
    int * colperm,
    bool permute,
    bool * symmetric,
    int * starting_vertex);

/**
 * ‘mtxfile_reorder()’ reorders the rows and columns of a matrix
 * according to the specified algorithm.
 *
 * If successful, this function returns ‘MTX_SUCCESS’, and the rows
 * and columns of ‘mtxfile’ have been reordered according to the
 * specified method. If ‘rowperm’ and ‘rowperminv’ are not ‘NULL’,
 * then they must point to arrays large enough to hold one ‘int’ for
 * each row of the matrix. In this case, the arrays are used to store
 * the permutation and inverse permutation, respectively, for
 * reordering the matrix rows. Similarly, ‘colperm’ and ‘colperminv’
 * are used to store the permutation and inverse permutation for
 * reordering the matrix columns.
 *
 * If ‘symmetric’ is not ‘NULL’, then it is used to return whether or
 * not the reordering is symmetric. That is, if the value returned in
 * ‘symmetric’ is ‘true’ then ‘rowperm’ and ‘colperm’ are identical,
 * and only one of them is needed.
 *
 * If ‘ordering’ is ‘mtxfile_metis’ and ‘objval’ is not ‘NULL’, then
 * it is used to store the value of the objective function minimized
 * by the partitioner, which, by default, is the edge-cut of the
 * partitioning solution.
 */
int LIBMTX_API mtxfile_reorder(
    struct mtxfile * mtxfile,
    enum mtxfileordering ordering,
    int * rowperm,
    int * rowperminv,
    int * colperm,
    int * colperminv,
    bool permute,
    bool * symmetric,
    int * rcm_starting_vertex,
    int nparts,
    int * rowpartsizes,
    int * colpartsizes,
    int64_t * objval,
    int verbose);

/*
 * MPI functions
 */

#ifdef LIBMTX_HAVE_MPI
/**
 * ‘mtxfile_send()’ sends a Matrix Market file to another MPI process.
 *
 * This is analogous to ‘MPI_Send()’ and requires the receiving
 * process to perform a matching call to ‘mtxfile_recv()’.
 */
int LIBMTX_API mtxfile_send(
    const struct mtxfile * mtxfile,
    int dest,
    int tag,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxfile_recv()’ receives a Matrix Market file from another MPI
 * process.
 *
 * This is analogous to ‘MPI_Recv()’ and requires the sending process
 * to perform a matching call to ‘mtxfile_send()’.
 */
int LIBMTX_API mtxfile_recv(
    struct mtxfile * mtxfile,
    int source,
    int tag,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxfile_bcast()’ broadcasts a Matrix Market file from an MPI root
 * process to other processes in a communicator.
 *
 * This is analogous to ‘MPI_Bcast()’ and requires every process in
 * the communicator to perform matching calls to ‘mtxfile_bcast()’.
 */
int LIBMTX_API mtxfile_bcast(
    struct mtxfile * mtxfile,
    int root,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxfile_gather()’ gathers Matrix Market files onto an MPI root
 * process from other processes in a communicator.
 *
 * This is analogous to ‘MPI_Gather()’ and requires every process in
 * the communicator to perform matching calls to ‘mtxfile_gather()’.
 */
int LIBMTX_API mtxfile_gather(
    const struct mtxfile * sendmtxfile,
    struct mtxfile * recvmtxfiles,
    int root,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxfile_allgather()’ gathers Matrix Market files onto every MPI
 * process from other processes in a communicator.
 *
 * This is analogous to ‘MPI_Allgather()’ and requires every process
 * in the communicator to perform matching calls to
 * ‘mtxfile_allgather()’.
 */
int LIBMTX_API mtxfile_allgather(
    const struct mtxfile * sendmtxfile,
    struct mtxfile * recvmtxfiles,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxfile_scatter()’ scatters Matrix Market files from an MPI root
 * process to other processes in a communicator.
 *
 * This is analogous to ‘MPI_Scatter()’ and requires every process in
 * the communicator to perform matching calls to ‘mtxfile_scatter()’.
 */
int LIBMTX_API mtxfile_scatter(
    const struct mtxfile * sendmtxfiles,
    struct mtxfile * recvmtxfile,
    int root,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxfile_alltoall()’ performs an all-to-all exchange of Matrix
 * Market files between MPI process in a communicator.
 *
 * This is analogous to ‘MPI_Alltoall()’ and requires every process in
 * the communicator to perform matching calls to ‘mtxfile_alltoall()’.
 */
int LIBMTX_API mtxfile_alltoall(
    const struct mtxfile * sendmtxfiles,
    struct mtxfile * recvmtxfiles,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxfile_scatterv()’ scatters a Matrix Market file from an MPI root
 * process to other processes in a communicator.
 *
 * This is analogous to ‘MPI_Scatterv()’ and requires every process in
 * the communicator to perform matching calls to ‘mtxfile_scatterv()’.
 *
 * For a matrix in ‘array’ format, entire rows are scattered, which
 * means that the send and receive counts must be multiples of the
 * number of matrix columns.
 */
int LIBMTX_API mtxfile_scatterv(
    const struct mtxfile * sendmtxfile,
    const int * sendcounts,
    const int * displs,
    struct mtxfile * recvmtxfile,
    int recvcount,
    int root,
    MPI_Comm comm,
    struct mtxdisterror * disterr);
#endif

#endif
