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
 * Last modified: 2022-04-08
 *
 * Data structures for vectors.
 */

#ifndef LIBMTX_VECTOR_VECTOR_H
#define LIBMTX_VECTOR_VECTOR_H

#include <libmtx/libmtx-config.h>

#include <libmtx/precision.h>
#include <libmtx/field.h>
#include <libmtx/vector/base.h>
#include <libmtx/vector/blas.h>
#include <libmtx/vector/omp.h>
#include <libmtx/vector/vector_array.h>
#include <libmtx/vector/vector_coordinate.h>

#ifdef LIBMTX_HAVE_LIBZ
#include <zlib.h>
#endif

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

struct mtxpartition;
struct mtxvector_packed;

/*
 * Vector types
 */

/**
 * ‘mtxvectortype’ is used to enumerate different vector formats.
 */
enum mtxvectortype
{
    mtxvector_auto,       /* automatic selection of vector type */
    mtxvector_array,      /* array format for dense vectors */
    mtxvector_base,       /* basic dense vectors */
    mtxvector_blas,       /* dense vectors with vector operations
                           * performed by an external BLAS library */
    mtxvector_omp,        /* dense vectors using OpenMP for shared
                           * memory parallel operations */
    mtxvector_coordinate, /* coordinate format for sparse vectors */
};

/**
 * ‘mtxvectortype_str()’ is a string representing the vector type.
 */
const char * mtxvectortype_str(
    enum mtxvectortype type);

/**
 * ‘mtxvectortype_parse()’ parses a string to obtain one of the
 * vector types of ‘enum mtxvectortype’.
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
 * On success, ‘mtxvectortype_parse()’ returns ‘MTX_SUCCESS’ and
 * ‘vector_type’ is set according to the parsed string and
 * ‘bytes_read’ is set to the number of bytes that were consumed by
 * the parser.  Otherwise, an error code is returned.
 */
int mtxvectortype_parse(
    enum mtxvectortype * vector_type,
    int64_t * bytes_read,
    const char ** endptr,
    const char * s,
    const char * valid_delimiters);

/*
 * Abstract vector data structure
 */

/**
 * ‘mtxvector’ represents a vector with various options available for
 * the underlying storage and implementation of vector operations.
 */
struct mtxvector
{
    /**
     * ‘type’ is the type of vector.
     */
    enum mtxvectortype type;

    /**
     * ‘storage’ is a union of different data types for the underlying
     * storage of the vector.
     */
    union
    {
        struct mtxvector_array array;
        struct mtxvector_base base;
#ifdef LIBMTX_HAVE_BLAS
        struct mtxvector_blas blas;
#endif
#ifdef LIBMTX_HAVE_OPENMP
        struct mtxvector_omp omp;
#endif
        struct mtxvector_coordinate coordinate;
    } storage;
};

/*
 * Memory management
 */

/**
 * ‘mtxvector_free()’ frees storage allocated for a vector.
 */
void mtxvector_free(
    struct mtxvector * x);

/**
 * ‘mtxvector_alloc_copy()’ allocates a copy of a vector without
 * initialising the values.
 */
int mtxvector_alloc_copy(
    struct mtxvector * dst,
    const struct mtxvector * src);

/**
 * ‘mtxvector_init_copy()’ allocates a copy of a vector and also
 * copies the values.
 */
int mtxvector_init_copy(
    struct mtxvector * dst,
    const struct mtxvector * src);

/*
 * Dense vectors
 */

/**
 * ‘mtxvector_alloc()’ allocates a vector of the given type.
 */
int mtxvector_alloc(
    struct mtxvector * x,
    enum mtxvectortype type,
    enum mtxfield field,
    enum mtxprecision precision,
    int size);

/**
 * ‘mtxvector_init_real_single()’ allocates and initialises a vector
 * with real, single precision coefficients.
 */
int mtxvector_init_real_single(
    struct mtxvector * x,
    enum mtxvectortype type,
    int size,
    const float * data);

/**
 * ‘mtxvector_init_real_double()’ allocates and initialises a vector
 * with real, double precision coefficients.
 */
int mtxvector_init_real_double(
    struct mtxvector * x,
    enum mtxvectortype type,
    int size,
    const double * data);

/**
 * ‘mtxvector_init_complex_single()’ allocates and initialises a
 * vector with complex, single precision coefficients.
 */
int mtxvector_init_complex_single(
    struct mtxvector * x,
    enum mtxvectortype type,
    int size,
    const float (* data)[2]);

/**
 * ‘mtxvector_init_complex_double()’ allocates and initialises a
 * vector with complex, double precision coefficients.
 */
int mtxvector_init_complex_double(
    struct mtxvector * x,
    enum mtxvectortype type,
    int size,
    const double (* data)[2]);

/**
 * ‘mtxvector_init_integer_single()’ allocates and initialises a
 * vector with integer, single precision coefficients.
 */
int mtxvector_init_integer_single(
    struct mtxvector * x,
    enum mtxvectortype type,
    int size,
    const int32_t * data);

/**
 * ‘mtxvector_init_integer_double()’ allocates and initialises a
 * vector with integer, double precision coefficients.
 */
int mtxvector_init_integer_double(
    struct mtxvector * x,
    enum mtxvectortype type,
    int size,
    const int64_t * data);

/**
 * ‘mtxvector_init_pattern()’ allocates and initialises a vector of
 * ones.
 */
int mtxvector_init_pattern(
    struct mtxvector * x,
    enum mtxvectortype type,
    int size);

/*
 * Basic, dense vectors
 */

/**
 * ‘mtxvector_alloc_base()’ allocates a dense vector.
 */
int mtxvector_alloc_base(
    struct mtxvector * x,
    enum mtxfield field,
    enum mtxprecision precision,
    int64_t size);

/**
 * ‘mtxvector_init_base_real_single()’ allocates and initialises a
 *  dense vector with real, single precision coefficients.
 */
int mtxvector_init_base_real_single(
    struct mtxvector * x,
    int64_t size,
    const float * data);

/**
 * ‘mtxvector_init_base_real_double()’ allocates and initialises a
 *  dense vector with real, double precision coefficients.
 */
int mtxvector_init_base_real_double(
    struct mtxvector * x,
    int64_t size,
    const double * data);

/**
 * ‘mtxvector_init_base_complex_single()’ allocates and initialises a
 *  dense vector with complex, single precision coefficients.
 */
int mtxvector_init_base_complex_single(
    struct mtxvector * x,
    int64_t size,
    const float (* data)[2]);

/**
 * ‘mtxvector_init_base_complex_double()’ allocates and initialises a
 *  dense vector with complex, double precision coefficients.
 */
int mtxvector_init_base_complex_double(
    struct mtxvector * x,
    int64_t size,
    const double (* data)[2]);

/**
 * ‘mtxvector_init_base_integer_single()’ allocates and initialises a
 *  dense vector with integer, single precision coefficients.
 */
int mtxvector_init_base_integer_single(
    struct mtxvector * x,
    int64_t size,
    const int32_t * data);

/**
 * ‘mtxvector_init_base_integer_double()’ allocates and initialises a
 *  dense vector with integer, double precision coefficients.
 */
int mtxvector_init_base_integer_double(
    struct mtxvector * x,
    int64_t size,
    const int64_t * data);

/*
 * Dense vectors in array format
 */

/**
 * ‘mtxvector_alloc_array()’ allocates a vector in array format.
 */
int mtxvector_alloc_array(
    struct mtxvector * x,
    enum mtxfield field,
    enum mtxprecision precision,
    int size);

/**
 * ‘mtxvector_init_array_real_single()’ allocates and initialises a
 * vector in array format with real, single precision coefficients.
 */
int mtxvector_init_array_real_single(
    struct mtxvector * x,
    int size,
    const float * data);

/**
 * ‘mtxvector_init_array_real_double()’ allocates and initialises a
 * vector in array format with real, double precision coefficients.
 */
int mtxvector_init_array_real_double(
    struct mtxvector * x,
    int size,
    const double * data);

/**
 * ‘mtxvector_init_array_complex_single()’ allocates and initialises a
 * vector in array format with complex, single precision coefficients.
 */
int mtxvector_init_array_complex_single(
    struct mtxvector * x,
    int size,
    const float (* data)[2]);

/**
 * ‘mtxvector_init_array_complex_double()’ allocates and initialises a
 * vector in array format with complex, double precision coefficients.
 */
int mtxvector_init_array_complex_double(
    struct mtxvector * x,
    int size,
    const double (* data)[2]);

/**
 * ‘mtxvector_init_array_integer_single()’ allocates and initialises a
 * vector in array format with integer, single precision coefficients.
 */
int mtxvector_init_array_integer_single(
    struct mtxvector * x,
    int size,
    const int32_t * data);

/**
 * ‘mtxvector_init_array_integer_double()’ allocates and initialises a
 * vector in array format with integer, double precision coefficients.
 */
int mtxvector_init_array_integer_double(
    struct mtxvector * x,
    int size,
    const int64_t * data);

/*
 * Basic, dense vectors
 */

/**
 * ‘mtxvector_alloc_base()’ allocates a dense vector.
 */
int mtxvector_alloc_base(
    struct mtxvector * x,
    enum mtxfield field,
    enum mtxprecision precision,
    int64_t size);

/**
 * ‘mtxvector_init_base_real_single()’ allocates and initialises a
 *  dense vector with real, single precision coefficients.
 */
int mtxvector_init_base_real_single(
    struct mtxvector * x,
    int64_t size,
    const float * data);

/**
 * ‘mtxvector_init_base_real_double()’ allocates and initialises a
 *  dense vector with real, double precision coefficients.
 */
int mtxvector_init_base_real_double(
    struct mtxvector * x,
    int64_t size,
    const double * data);

/**
 * ‘mtxvector_init_base_complex_single()’ allocates and initialises a
 *  dense vector with complex, single precision coefficients.
 */
int mtxvector_init_base_complex_single(
    struct mtxvector * x,
    int64_t size,
    const float (* data)[2]);

/**
 * ‘mtxvector_init_base_complex_double()’ allocates and initialises a
 *  dense vector with complex, double precision coefficients.
 */
int mtxvector_init_base_complex_double(
    struct mtxvector * x,
    int64_t size,
    const double (* data)[2]);

/**
 * ‘mtxvector_init_base_integer_single()’ allocates and initialises a
 *  dense vector with integer, single precision coefficients.
 */
int mtxvector_init_base_integer_single(
    struct mtxvector * x,
    int64_t size,
    const int32_t * data);

/**
 * ‘mtxvector_init_base_integer_double()’ allocates and initialises a
 *  dense vector with integer, double precision coefficients.
 */
int mtxvector_init_base_integer_double(
    struct mtxvector * x,
    int64_t size,
    const int64_t * data);

/*
 * Dense vectors with vector operations accelerated by an external
 * BLAS library.
 */

/**
 * ‘mtxvector_alloc_blas()’ allocates a dense vector.
 */
int mtxvector_alloc_blas(
    struct mtxvector * x,
    enum mtxfield field,
    enum mtxprecision precision,
    int64_t size);

/**
 * ‘mtxvector_init_blas_real_single()’ allocates and initialises a
 *  dense vector with real, single precision coefficients.
 */
int mtxvector_init_blas_real_single(
    struct mtxvector * x,
    int64_t size,
    const float * data);

/**
 * ‘mtxvector_init_blas_real_double()’ allocates and initialises a
 *  dense vector with real, double precision coefficients.
 */
int mtxvector_init_blas_real_double(
    struct mtxvector * x,
    int64_t size,
    const double * data);

/**
 * ‘mtxvector_init_blas_complex_single()’ allocates and initialises a
 *  dense vector with complex, single precision coefficients.
 */
int mtxvector_init_blas_complex_single(
    struct mtxvector * x,
    int64_t size,
    const float (* data)[2]);

/**
 * ‘mtxvector_init_blas_complex_double()’ allocates and initialises a
 *  dense vector with complex, double precision coefficients.
 */
int mtxvector_init_blas_complex_double(
    struct mtxvector * x,
    int64_t size,
    const double (* data)[2]);

/**
 * ‘mtxvector_init_blas_integer_single()’ allocates and initialises a
 *  dense vector with integer, single precision coefficients.
 */
int mtxvector_init_blas_integer_single(
    struct mtxvector * x,
    int64_t size,
    const int32_t * data);

/**
 * ‘mtxvector_init_blas_integer_double()’ allocates and initialises a
 *  dense vector with integer, double precision coefficients.
 */
int mtxvector_init_blas_integer_double(
    struct mtxvector * x,
    int64_t size,
    const int64_t * data);

/*
 * OpenMP shared-memory parallel vectors
 */

/**
 * ‘mtxvector_alloc_omp()’ allocates an OpenMP shared-memory parallel
 * vector.
 */
int mtxvector_alloc_omp(
    struct mtxvector * x,
    enum mtxfield field,
    enum mtxprecision precision,
    int size,
    int num_threads);

/**
 * ‘mtxvector_init_omp_real_single()’ allocates and initialises an
 *  OpenMP shared-memory parallel vector with real, single precision
 *  coefficients.
 */
int mtxvector_init_omp_real_single(
    struct mtxvector * x,
    int size,
    const float * data,
    int num_threads);

/**
 * ‘mtxvector_init_omp_real_double()’ allocates and initialises an
 *  OpenMP shared-memory parallel vector with real, double precision
 *  coefficients.
 */
int mtxvector_init_omp_real_double(
    struct mtxvector * x,
    int size,
    const double * data,
    int num_threads);

/**
 * ‘mtxvector_init_omp_complex_single()’ allocates and initialises an
 *  OpenMP shared-memory parallel vector with complex, single
 *  precision coefficients.
 */
int mtxvector_init_omp_complex_single(
    struct mtxvector * x,
    int size,
    const float (* data)[2],
    int num_threads);

/**
 * ‘mtxvector_init_omp_complex_double()’ allocates and initialises an
 *  OpenMP shared-memory parallel vector with complex, double
 *  precision coefficients.
 */
int mtxvector_init_omp_complex_double(
    struct mtxvector * x,
    int size,
    const double (* data)[2],
    int num_threads);

/**
 * ‘mtxvector_init_omp_integer_single()’ allocates and initialises an
 *  OpenMP shared-memory parallel vector with integer, single
 *  precision coefficients.
 */
int mtxvector_init_omp_integer_single(
    struct mtxvector * x,
    int size,
    const int32_t * data,
    int num_threads);

/**
 * ‘mtxvector_init_omp_integer_double()’ allocates and initialises an
 *  OpenMP shared-memory parallel vector with integer, double
 *  precision coefficients.
 */
int mtxvector_init_omp_integer_double(
    struct mtxvector * x,
    int size,
    const int64_t * data,
    int num_threads);

/*
 * Vector coordinate formats
 */

/**
 * ‘mtxvector_alloc_coordinate()’ allocates a vector in
 * coordinate format.
 */
int mtxvector_alloc_coordinate(
    struct mtxvector * x,
    enum mtxfield field,
    enum mtxprecision precision,
    int size,
    int64_t num_nonzeros);

/**
 * ‘mtxvector_init_coordinate_real_single()’ allocates and initialises
 * a vector in coordinate format with real, single precision
 * coefficients.
 */
int mtxvector_init_coordinate_real_single(
    struct mtxvector * x,
    int size,
    int64_t num_nonzeros,
    const int * indices,
    const float * values);

/**
 * ‘mtxvector_init_coordinate_real_double()’ allocates and initialises
 * a vector in coordinate format with real, double precision
 * coefficients.
 */
int mtxvector_init_coordinate_real_double(
    struct mtxvector * x,
    int size,
    int64_t num_nonzeros,
    const int * indices,
    const double * values);

/**
 * ‘mtxvector_init_coordinate_complex_single()’ allocates and
 * initialises a vector in coordinate format with complex, single
 * precision coefficients.
 */
int mtxvector_init_coordinate_complex_single(
    struct mtxvector * x,
    int size,
    int64_t num_nonzeros,
    const int * indices,
    const float (* values)[2]);

/**
 * ‘mtxvector_init_coordinate_complex_double()’ allocates and
 * initialises a vector in coordinate format with complex, double
 * precision coefficients.
 */
int mtxvector_init_coordinate_complex_double(
    struct mtxvector * x,
    int size,
    int64_t num_nonzeros,
    const int * indices,
    const double (* values)[2]);

/**
 * ‘mtxvector_init_coordinate_integer_single()’ allocates and
 * initialises a vector in coordinate format with integer, single
 * precision coefficients.
 */
int mtxvector_init_coordinate_integer_single(
    struct mtxvector * x,
    int size,
    int64_t num_nonzeros,
    const int * indices,
    const int32_t * values);

/**
 * ‘mtxvector_init_coordinate_integer_double()’ allocates and
 * initialises a vector in coordinate format with integer, double
 * precision coefficients.
 */
int mtxvector_init_coordinate_integer_double(
    struct mtxvector * x,
    int size,
    int64_t num_nonzeros,
    const int * indices,
    const int64_t * values);

/**
 * ‘mtxvector_init_coordinate_pattern()’ allocates and initialises a
 * vector in coordinate format with integer, double precision
 * coefficients.
 */
int mtxvector_init_coordinate_pattern(
    struct mtxvector * x,
    int size,
    int64_t num_nonzeros,
    const int * indices);

/*
 * Modifying values
 */

/**
 * ‘mtxvector_set_constant_real_single()’ sets every (nonzero) value
 * of a vector equal to a constant, single precision floating point
 * number.
 */
int mtxvector_set_constant_real_single(
    struct mtxvector * x,
    float a);

/**
 * ‘mtxvector_set_constant_real_double()’ sets every (nonzero) value
 * of a vector equal to a constant, double precision floating point
 * number.
 */
int mtxvector_set_constant_real_double(
    struct mtxvector * x,
    double a);

/**
 * ‘mtxvector_set_constant_complex_single()’ sets every (nonzero)
 * value of a vector equal to a constant, single precision floating
 * point complex number.
 */
int mtxvector_set_constant_complex_single(
    struct mtxvector * x,
    float a[2]);

/**
 * ‘mtxvector_set_constant_complex_double()’ sets every (nonzero)
 * value of a vector equal to a constant, double precision floating
 * point complex number.
 */
int mtxvector_set_constant_complex_double(
    struct mtxvector * x,
    double a[2]);

/**
 * ‘mtxvector_set_constant_integer_single()’ sets every (nonzero)
 * value of a vector equal to a constant integer.
 */
int mtxvector_set_constant_integer_single(
    struct mtxvector * x,
    int32_t a);

/**
 * ‘mtxvector_set_constant_integer_double()’ sets every (nonzero)
 * value of a vector equal to a constant integer.
 */
int mtxvector_set_constant_integer_double(
    struct mtxvector * x,
    int64_t a);

/*
 * Convert to and from Matrix Market format
 */

/**
 * ‘mtxvector_from_mtxfile()’ converts a vector in Matrix Market
 * format to a vector.
 */
int mtxvector_from_mtxfile(
    struct mtxvector * dst,
    const struct mtxfile * src,
    enum mtxvectortype type);

/**
 * ‘mtxvector_to_mtxfile()’ converts a vector to a vector in Matrix
 * Market format.
 */
int mtxvector_to_mtxfile(
    struct mtxfile * dst,
    const struct mtxvector * src,
    enum mtxfileformat mtxfmt);

/*
 * I/O functions
 */

/**
 * ‘mtxvector_read()’ reads a vector from a Matrix Market file.  The
 * file may optionally be compressed by gzip.
 *
 * The ‘precision’ argument specifies which precision to use for
 * storing vector values.
 *
 * The ‘type’ argument specifies which format to use for representing
 * the vector.  If ‘type’ is ‘mtxvector_auto’, then the underlying
 * vector is stored in array format or coordinate format according to
 * the format of the Matrix Market file.  Otherwise, an attempt is
 * made to convert the vector to the desired type.
 *
 * If ‘path’ is ‘-’, then standard input is used.
 *
 * The file is assumed to be gzip-compressed if ‘gzip’ is ‘true’, and
 * uncompressed otherwise.
 *
 * If an error code is returned, then ‘lines_read’ and ‘bytes_read’
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the vector.
 */
int mtxvector_read(
    struct mtxvector * x,
    enum mtxprecision precision,
    enum mtxvectortype type,
    const char * path,
    bool gzip,
    int * lines_read,
    int64_t * bytes_read);

/**
 * ‘mtxvector_fread()’ reads a vector from a stream in Matrix Market
 * format.
 *
 * ‘precision’ is used to determine the precision to use for storing
 * the values of vector entries.
 *
 * The ‘type’ argument specifies which format to use for representing
 * the vector.  If ‘type’ is ‘mtxvector_auto’, then the underlying
 * vector is stored in array format or coordinate format according to
 * the format of the Matrix Market file.  Otherwise, an attempt is
 * made to convert the vector to the desired type.
 *
 * If an error code is returned, then ‘lines_read’ and ‘bytes_read’
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the vector.
 */
int mtxvector_fread(
    struct mtxvector * x,
    enum mtxprecision precision,
    enum mtxvectortype type,
    FILE * f,
    int * lines_read,
    int64_t * bytes_read,
    size_t line_max,
    char * linebuf);

#ifdef LIBMTX_HAVE_LIBZ
/**
 * ‘mtxvector_gzread()’ reads a vector from a gzip-compressed stream.
 *
 * ‘precision’ is used to determine the precision to use for storing
 * the values of vector entries.
 *
 * The ‘type’ argument specifies which format to use for representing
 * the vector.  If ‘type’ is ‘mtxvector_auto’, then the underlying
 * vector is stored in array format or coordinate format according to
 * the format of the Matrix Market file.  Otherwise, an attempt is
 * made to convert the vector to the desired type.
 *
 * If an error code is returned, then ‘lines_read’ and ‘bytes_read’
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the vector.
 */
int mtxvector_gzread(
    struct mtxvector * x,
    enum mtxprecision precision,
    enum mtxvectortype type,
    gzFile f,
    int * lines_read,
    int64_t * bytes_read,
    size_t line_max,
    char * linebuf);
#endif

/**
 * ‘mtxvector_write()’ writes a vector to a Matrix Market file. The
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
int mtxvector_write(
    const struct mtxvector * x,
    enum mtxfileformat mtxfmt,
    const char * path,
    bool gzip,
    const char * fmt,
    int64_t * bytes_written);

/**
 * ‘mtxvector_fwrite()’ writes a vector to a stream.
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
int mtxvector_fwrite(
    const struct mtxvector * x,
    enum mtxfileformat mtxfmt,
    FILE * f,
    const char * fmt,
    int64_t * bytes_written);

#ifdef LIBMTX_HAVE_LIBZ
/**
 * ‘mtxvector_gzwrite()’ writes a vector to a gzip-compressed stream.
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
int mtxvector_gzwrite(
    const struct mtxvector * x,
    enum mtxfileformat mtxfmt,
    gzFile f,
    const char * fmt,
    int64_t * bytes_written);
#endif

/*
 * Partitioning
 */

/**
 * ‘mtxvector_partition()’ partitions a vector into blocks according
 * to the given partitioning.
 *
 * The partition ‘part’ is allowed to be ‘NULL’, in which case a
 * trivial, singleton partition is used to partition the entries of
 * the vector. Otherwise, ‘part’ must partition the entries of the
 * vector ‘src’. That is, ‘part->size’ must be equal to the size of
 * the vector.
 *
 * The argument ‘dsts’ is an array that must have enough storage for
 * ‘P’ values of type ‘struct mtxvector’, where ‘P’ is the number of
 * parts, ‘part->num_parts’.
 *
 * The user is responsible for freeing storage allocated for each
 * vector in the ‘dsts’ array.
 */
int mtxvector_partition(
    struct mtxvector * dsts,
    const struct mtxvector * src,
    const struct mtxpartition * part);

/**
 * ‘mtxvector_join()’ joins together block vectors to form a larger
 * vector.
 *
 * The argument ‘srcs’ is an array of size ‘P’, where ‘P’ is the
 * number of parts in the partitioning (i.e, ‘part->num_parts’).
 */
int mtxvector_join(
    struct mtxvector * dst,
    const struct mtxvector * srcs,
    const struct mtxpartition * part);

/*
 * Level 1 BLAS operations
 */

/**
 * ‘mtxvector_swap()’ swaps values of two vectors, simultaneously
 * performing ‘y <- x’ and ‘x <- y’.
 */
int mtxvector_swap(
    struct mtxvector * x,
    struct mtxvector * y);

/**
 * ‘mtxvector_copy()’ copies values of a vector, ‘y = x’.
 */
int mtxvector_copy(
    struct mtxvector * y,
    const struct mtxvector * x);

/**
 * ‘mtxvector_sscal()’ scales a vector by a single precision floating
 * point scalar, ‘x = a*x’.
 */
int mtxvector_sscal(
    float a,
    struct mtxvector * x,
    int64_t * num_flops);

/**
 * ‘mtxvector_dscal()’ scales a vector by a double precision floating
 * point scalar, ‘x = a*x’.
 */
int mtxvector_dscal(
    double a,
    struct mtxvector * x,
    int64_t * num_flops);

/**
 * ‘mtxvector_cscal()’ scales a vector by a complex, single precision
 * floating point scalar, ‘x = (a+b*i)*x’.
 */
int mtxvector_cscal(
    float a[2],
    struct mtxvector * x,
    int64_t * num_flops);

/**
 * ‘mtxvector_zscal()’ scales a vector by a complex, double precision
 * floating point scalar, ‘x = (a+b*i)*x’.
 */
int mtxvector_zscal(
    double a[2],
    struct mtxvector * x,
    int64_t * num_flops);

/**
 * ‘mtxvector_saxpy()’ adds a vector to another vector multiplied by a
 * single precision floating point value, ‘y = a*x + y’.
 */
int mtxvector_saxpy(
    float a,
    const struct mtxvector * x,
    struct mtxvector * y,
    int64_t * num_flops);

/**
 * ‘mtxvector_daxpy()’ adds a vector to another vector multiplied by a
 * double precision floating point value, ‘y = a*x + y’.
 */
int mtxvector_daxpy(
    double a,
    const struct mtxvector * x,
    struct mtxvector * y,
    int64_t * num_flops);

/**
 * ‘mtxvector_saypx()’ multiplies a vector by a single precision
 * floating point scalar and adds another vector, ‘y = a*y + x’.
 */
int mtxvector_saypx(
    float a,
    struct mtxvector * y,
    const struct mtxvector * x,
    int64_t * num_flops);

/**
 * ‘mtxvector_daypx()’ multiplies a vector by a double precision
 * floating point scalar and adds another vector, ‘y = a*y + x’.
 */
int mtxvector_daypx(
    double a,
    struct mtxvector * y,
    const struct mtxvector * x,
    int64_t * num_flops);

/**
 * ‘mtxvector_sdot()’ computes the Euclidean dot product of two
 * vectors in single precision floating point.
 */
int mtxvector_sdot(
    const struct mtxvector * x,
    const struct mtxvector * y,
    float * dot,
    int64_t * num_flops);

/**
 * ‘mtxvector_ddot()’ computes the Euclidean dot product of two
 * vectors in double precision floating point.
 */
int mtxvector_ddot(
    const struct mtxvector * x,
    const struct mtxvector * y,
    double * dot,
    int64_t * num_flops);

/**
 * ‘mtxvector_cdotu()’ computes the product of the transpose of a
 * complex row vector with another complex row vector in single
 * precision floating point, ‘dot := x^T*y’.
 */
int mtxvector_cdotu(
    const struct mtxvector * x,
    const struct mtxvector * y,
    float (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxvector_zdotu()’ computes the product of the transpose of a
 * complex row vector with another complex row vector in double
 * precision floating point, ‘dot := x^T*y’.
 */
int mtxvector_zdotu(
    const struct mtxvector * x,
    const struct mtxvector * y,
    double (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxvector_cdotc()’ computes the Euclidean dot product of two
 * complex vectors in single precision floating point, ‘dot := x^H*y’.
 */
int mtxvector_cdotc(
    const struct mtxvector * x,
    const struct mtxvector * y,
    float (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxvector_zdotc()’ computes the Euclidean dot product of two
 * complex vectors in double precision floating point, ‘dot := x^H*y’.
 */
int mtxvector_zdotc(
    const struct mtxvector * x,
    const struct mtxvector * y,
    double (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxvector_snrm2()’ computes the Euclidean norm of a vector in
 * single precision floating point.
 */
int mtxvector_snrm2(
    const struct mtxvector * x,
    float * nrm2,
    int64_t * num_flops);

/**
 * ‘mtxvector_dnrm2()’ computes the Euclidean norm of a vector in
 * double precision floating point.
 */
int mtxvector_dnrm2(
    const struct mtxvector * x,
    double * nrm2,
    int64_t * num_flops);

/**
 * ‘mtxvector_sasum()’ computes the sum of absolute values (1-norm) of
 * a vector in single precision floating point.  If the vector is
 * complex-valued, then the sum of the absolute values of the real and
 * imaginary parts is computed.
 */
int mtxvector_sasum(
    const struct mtxvector * x,
    float * asum,
    int64_t * num_flops);

/**
 * ‘mtxvector_dasum()’ computes the sum of absolute values (1-norm) of
 * a vector in double precision floating point.  If the vector is
 * complex-valued, then the sum of the absolute values of the real and
 * imaginary parts is computed.
 */
int mtxvector_dasum(
    const struct mtxvector * x,
    double * asum,
    int64_t * num_flops);

/**
 * ‘mtxvector_iamax()’ finds the index of the first element having the
 * maximum absolute value.  If the vector is complex-valued, then the
 * index points to the first element having the maximum sum of the
 * absolute values of the real and imaginary parts.
 */
int mtxvector_iamax(
    const struct mtxvector * x,
    int * iamax);

/*
 * Level 1 Sparse BLAS operations.
 *
 * See I. Duff, M. Heroux and R. Pozo, "An Overview of the Sparse
 * Basic Linear Algebra Subprograms: The New Standard from the BLAS
 * Technical Forum," ACM TOMS, Vol. 28, No. 2, June 2002, pp. 239-267.
 */

/**
 * ‘mtxvector_usga()’ performs a (sparse) gather from a vector ‘y’
 * into another vector ‘x’.
 */
int mtxvector_usga(
    const struct mtxvector * y,
    struct mtxvector * x);

/**
 * ‘mtxvector_usga2()’ performs a gather operation from a vector ‘y’
 * into a sparse vector ‘x’ in packed storage format.
 */
int mtxvector_usga2(
    struct mtxvector_packed * x,
    const struct mtxvector * y);

/**
 * ‘mtxvector_usgz()’ performs a (sparse) gather from a vector ‘y’
 * into another vector ‘x’, while zeroing the corresponding elements
 * of ‘y’ that were copied to ‘x’.
 */
int mtxvector_usgz(
    const struct mtxvector * y,
    struct mtxvector * x);

/**
 * ‘mtxvector_ussc()’ performs a (sparse) scatter from a vector ‘x’
 * into another vector ‘y’.
 */
int mtxvector_ussc(
    const struct mtxvector * x,
    struct mtxvector * y);

/**
 * ‘mtxvector_ussc2()’ performs a scatter operation to a vector ‘y’
 * from a sparse vector ‘x’ in packed storage format.
 */
int mtxvector_ussc2(
    struct mtxvector * y,
    const struct mtxvector_packed * x);

/*
 * Sorting
 */

/**
 * ‘mtxvector_permute()’ permutes the elements of a vector according
 * to a given permutation.
 *
 * The array ‘perm’ should be an array of length ‘size’ that stores a
 * permutation of the integers ‘0,1,...,N-1’, where ‘N’ is the number
 * of vector elements.
 *
 * After permuting, the 1st vector element of the original vector is
 * now located at position ‘perm[0]’ in the sorted vector ‘x’, the 2nd
 * element is now at position ‘perm[1]’, and so on.
 */
int mtxvector_permute(
    struct mtxvector * x,
    int64_t offset,
    int64_t size,
    int64_t * perm);

/**
 * ‘mtxvector_sort()’ sorts elements of a vector by the given keys.
 *
 * The array ‘keys’ must be an array of length ‘size’ that stores a
 * 64-bit unsigned integer sorting key that is used to define the
 * order in which to sort the vector elements..
 *
 * If it is not ‘NULL’, then ‘perm’ must point to an array of length
 * ‘size’, which is then used to store the sorting permutation. That
 * is, ‘perm’ is a permutation of the integers ‘0,1,...,N-1’, where
 * ‘N’ is the number of vector elements, such that the 1st vector
 * element in the original vector is now located at position ‘perm[0]’
 * in the sorted vector ‘x’, the 2nd element is now at position
 * ‘perm[1]’, and so on.
 */
int mtxvector_sort(
    struct mtxvector * x,
    int64_t size,
    uint64_t * keys,
    int64_t * perm);

/*
 * MPI functions
 */

#ifdef LIBMTX_HAVE_MPI
/**
 * ‘mtxvector_send()’ sends a vector to another MPI process.
 *
 * This is analogous to ‘MPI_Send()’ and requires the receiving
 * process to perform a matching call to ‘mtxvector_recv()’.
 */
int mtxvector_send(
    const struct mtxvector * x,
    int64_t size,
    int64_t offset,
    int dest,
    int tag,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxvector_recv()’ receives a vector from another MPI process.
 *
 * This is analogous to ‘MPI_Recv()’ and requires the sending process
 * to perform a matching call to ‘mtxvector_send()’.
 */
int mtxvector_recv(
    struct mtxvector * x,
    int64_t size,
    int64_t offset,
    int source,
    int tag,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxvector_bcast()’ broadcasts a vector from an MPI root process to
 * other processes in a communicator.
 *
 * This is analogous to ‘MPI_Bcast()’ and requires every process in
 * the communicator to perform matching calls to
 * ‘mtxvector_bcast()’.
 */
int mtxvector_bcast(
    struct mtxvector * x,
    int64_t size,
    int64_t offset,
    int root,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxvector_gatherv()’ gathers a vector onto an MPI root process
 * from other processes in a communicator.
 *
 * This is analogous to ‘MPI_Gatherv()’ and requires every process in
 * the communicator to perform matching calls to
 * ‘mtxvector_gatherv()’.
 */
int mtxvector_gatherv(
    const struct mtxvector * sendbuf,
    int64_t sendoffset,
    int sendcount,
    struct mtxvector * recvbuf,
    int64_t recvoffset,
    const int * recvcounts,
    const int * recvdispls,
    int root,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxvector_scatterv()’ scatters a vector from an MPI root process
 * to other processes in a communicator.
 *
 * This is analogous to ‘MPI_Scatterv()’ and requires every process in
 * the communicator to perform matching calls to
 * ‘mtxvector_scatterv()’.
 */
int mtxvector_scatterv(
    const struct mtxvector * sendbuf,
    int64_t sendoffset,
    const int * sendcounts,
    const int * displs,
    struct mtxvector * recvbuf,
    int64_t recvoffset,
    int recvcount,
    int root,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxvector_alltoallv()’ performs an all-to-all exchange of a vector
 * between MPI processes in a communicator.
 *
 * This is analogous to ‘MPI_Alltoallv()’ and requires every process
 * in the communicator to perform matching calls to
 * ‘mtxvector_alltoallv()’.
 */
int mtxvector_alltoallv(
    const struct mtxvector * sendbuf,
    int64_t sendoffset,
    const int * sendcounts,
    const int * senddispls,
    struct mtxvector * recvbuf,
    int64_t recvoffset,
    const int * recvcounts,
    const int * recvdispls,
    MPI_Comm comm,
    struct mtxdisterror * disterr);
#endif

#endif
