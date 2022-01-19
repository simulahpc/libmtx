/* This file is part of libmtx.
 *
 * Copyright (C) 2022 James D. Trotter
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
 * Last modified: 2022-01-19
 *
 * Data structures for distributed vectors.
 */

#ifndef LIBMTX_VECTOR_DISTVECTOR_H
#define LIBMTX_VECTOR_DISTVECTOR_H

#include <libmtx/libmtx-config.h>

#ifdef LIBMTX_HAVE_MPI
#include <libmtx/precision.h>
#include <libmtx/field.h>
#include <libmtx/vector/vector.h>

#include <mpi.h>

#ifdef LIBMTX_HAVE_LIBZ
#include <zlib.h>
#endif

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

struct mtxfile;
struct mtxdistfile;
struct mtxdisterror;
struct mtxpartition;

/**
 * ‘mtxdistvector’ is a vector distributed across multiple processes,
 * where MPI is used for communicating between processes.
 *
 * Processes are arranged as a one-dimensional linear array, and
 * vector elements (i.e., the rows of a row vector) are distributed
 * among processes according to a specified partitioning.
 */
struct mtxdistvector
{
    /**
     * ‘comm’ is an MPI communicator for processes among which the
     * Matrix Market file is distributed.
     */
    MPI_Comm comm;

    /**
     * ‘comm_size’ is the size of the MPI communicator.  This is equal
     * to the number of parts of the row partitioning of the matrix or
     * vector.
     */
    int comm_size;

    /**
     * ‘rank’ is the rank of the current process.
     */
    int rank;

    /**
     * ‘rowpart’ is a partitioning of the rows of the vector.
     */
    struct mtxpartition rowpart;

    struct mtxvector interior;

    struct mtxvector_coordinate interior_halo;

    struct mtxvector_coordinate exterior_halo;
};

/*
 * Memory management
 */

/**
 * ‘mtxdistvector_free()’ frees storage allocated for a vector.
 */
void mtxdistvector_free(
    struct mtxdistvector * distvector);

/**
 * ‘mtxdistvector_alloc_copy()’ allocates storage for a copy of a
 * distributed vector without initialising the underlying values.
 */
int mtxdistvector_alloc_copy(
    struct mtxdistvector * dst,
    const struct mtxdistvector * src);

/**
 * ‘mtxdistvector_init_copy()’ creates a copy of a distributed vector.
 */
int mtxdistvector_init_copy(
    struct mtxdistvector * dst,
    const struct mtxdistvector * src);

/*
 * Distributed vectors in array format
 */

/**
 * ‘mtxdistvector_alloc_array()’ allocates a distributed vector in
 * array format.
 */
int mtxdistvector_alloc_array(
    struct mtxdistvector * distvector,
    enum mtxfield field,
    enum mtxprecision precision,
    int num_rows,
    const struct mtxpartition * rowpart,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistvector_init_array_real_single()’ allocates and initialises
 * a distributed vector in array format with real, single precision
 * coefficients.
 */
int mtxdistvector_init_array_real_single(
    struct mtxdistvector * distvector,
    int num_rows,
    const float * data,
    const struct mtxpartition * rowpart,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistvector_init_array_real_double()’ allocates and initialises
 * a distributed vector in array format with real, double precision
 * coefficients.
 */
int mtxdistvector_init_array_real_double(
    struct mtxdistvector * distvector,
    int num_rows,
    const double * data,
    const struct mtxpartition * rowpart,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistvector_init_array_complex_single()’ allocates and
 * initialises a distributed vector in array format with complex,
 * single precision coefficients.
 */
int mtxdistvector_init_array_complex_single(
    struct mtxdistvector * distvector,
    int num_rows,
    const float (* data)[2],
    const struct mtxpartition * rowpart,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistvector_init_array_complex_double()’ allocates and
 * initialises a distributed vector in array format with complex,
 * double precision coefficients.
 */
int mtxdistvector_init_array_complex_double(
    struct mtxdistvector * distvector,
    int num_rows,
    const double (* data)[2],
    const struct mtxpartition * rowpart,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistvector_init_array_integer_single()’ allocates and
 * initialises a distributed vector in array format with integer,
 * single precision coefficients.
 */
int mtxdistvector_init_array_integer_single(
    struct mtxdistvector * distvector,
    int num_rows,
    const int32_t * data,
    const struct mtxpartition * rowpart,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistvector_init_array_integer_double()’ allocates and
 * initialises a distributed vector in array format with integer,
 * double precision coefficients.
 */
int mtxdistvector_init_array_integer_double(
    struct mtxdistvector * distvector,
    int num_rows,
    const int64_t * data,
    const struct mtxpartition * rowpart,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/*
 * Distributed vectors in coordinate format
 */

/**
 * ‘mtxdistvector_alloc_coordinate()’ allocates a distributed vector
 * in coordinate format.
 */
int mtxdistvector_alloc_coordinate(
    struct mtxdistvector * distvector,
    enum mtxfield field,
    enum mtxprecision precision,
    int num_rows,
    int64_t num_nonzeros,
    const struct mtxpartition * rowpart,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistvector_init_coordinate_real_single()’ allocates and
 * initialises a distributed vector in coordinate format with real,
 * single precision coefficients.
 */
int mtxdistvector_init_coordinate_real_single(
    struct mtxdistvector * distvector,
    int num_rows,
    int64_t num_nonzeros,
    const int * idx,
    const float * data,
    const struct mtxpartition * rowpart,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistvector_init_coordinate_real_double()’ allocates and
 * initialises a distributed vector in coordinate format with real,
 * double precision coefficients.
 */
int mtxdistvector_init_coordinate_real_double(
    struct mtxdistvector * distvector,
    int num_rows,
    int64_t num_nonzeros,
    const int * idx,
    const double * data,
    const struct mtxpartition * rowpart,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistvector_init_coordinate_complex_single()’ allocates and
 * initialises a distributed vector in coordinate format with complex,
 * single precision coefficients.
 */
int mtxdistvector_init_coordinate_complex_single(
    struct mtxdistvector * distvector,
    int num_rows,
    int64_t num_nonzeros,
    const int * idx,
    const float (* data)[2],
    const struct mtxpartition * rowpart,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistvector_init_coordinate_complex_double()’ allocates and
 * initialises a distributed vector in coordinate format with complex,
 * double precision coefficients.
 */
int mtxdistvector_init_coordinate_complex_double(
    struct mtxdistvector * distvector,
    int num_rows,
    int64_t num_nonzeros,
    const int * idx,
    const double (* data)[2],
    const struct mtxpartition * rowpart,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistvector_init_coordinate_integer_single()’ allocates and
 * initialises a distributed vector in coordinate format with integer,
 * single precision coefficients.
 */
int mtxdistvector_init_coordinate_integer_single(
    struct mtxdistvector * distvector,
    int num_rows,
    int64_t num_nonzeros,
    const int * idx,
    const int32_t * data,
    const struct mtxpartition * rowpart,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistvector_init_coordinate_integer_double()’ allocates and
 * initialises a distributed vector in coordinate format with integer,
 * double precision coefficients.
 */
int mtxdistvector_init_coordinate_integer_double(
    struct mtxdistvector * distvector,
    int num_rows,
    int64_t num_nonzeros,
    const int * idx,
    const int64_t * data,
    const struct mtxpartition * rowpart,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistvector_init_coordinate_pattern()’ allocates and initialises
 * a distributed vector in coordinate format with boolean
 * coefficients.
 */
int mtxdistvector_init_coordinate_pattern(
    struct mtxdistvector * distvector,
    int num_rows,
    int64_t num_nonzeros,
    const int * idx,
    const struct mtxpartition * rowpart,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/*
 * Convert to and from Matrix Market format
 */

/**
 * ‘mtxdistvector_from_mtxfile()’ converts a vector in Matrix Market
 * format to a distributed vector.
 *
 * The ‘type’ argument may be used to specify a desired storage format
 * or implementation for the underlying ‘mtxvector’ on each
 * process. If ‘type’ is ‘mtxvector_auto’, then the type of
 * ‘mtxvector’ is chosen to match the type of ‘mtxfile’. That is,
 * ‘mtxvector_array’ is used if ‘mtxfile’ is in array format, and
 * ‘mtxvector_coordinate’ is used if ‘mtxfile’ is in coordinate
 * format.
 *
 * Furthermore, ‘rowpart’ must be a partitioning of the rows of the
 * global vector. Therefore, ‘rowpart->size’ must be equal to the
 * number of rows in the underlying vector represented by
 * ‘mtxfile’. The partition must consist of at most one part for each
 * MPI process in the communicator ‘comm’. If ‘rowpart’ is ‘NULL’,
 * then the rows are partitioned into contiguous blocks of equal size
 * by default.
 */
int mtxdistvector_from_mtxfile(
    struct mtxdistvector * dst,
    const struct mtxfile * src,
    enum mtxvectortype vector_type,
    const struct mtxpartition * rowpart,
    MPI_Comm comm,
    int root,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistvector_to_mtxfile()’ gathers a distributed vector onto a
 * single, root process and converts it to a (non-distributed) Matrix
 * Market file on that process.
 *
 * This function performs collective communication and therefore
 * requires every process in the communicator to perform matching
 * calls to this function.
 */
int mtxdistvector_to_mtxfile(
    struct mtxfile * dst,
    const struct mtxdistvector * src,
    enum mtxfileformat mtxfmt,
    int root,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistvector_from_mtxdistfile()’ converts a vector in distributed
 * Matrix Market format to a distributed vector.
 */
int mtxdistvector_from_mtxdistfile(
    struct mtxdistvector * dst,
    const struct mtxdistfile * src,
    enum mtxvectortype vector_type,
    const struct mtxpartition * rowpart,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistvector_to_mtxdistfile()’ converts a distributed vector to a
 * vector in a distributed Matrix Market format.
 */
int mtxdistvector_to_mtxdistfile(
    struct mtxdistfile * dst,
    const struct mtxdistvector * src,
    enum mtxfileformat mtxfmt,
    struct mtxdisterror * disterr);

/*
 * I/O functions
 */

/**
 * ‘mtxdistvector_read()’ reads a vector from a Matrix Market file.
 * The file may optionally be compressed by gzip.
 *
 * The ‘precision’ argument specifies which precision to use for
 * storing matrix or vector values.
 *
 * If ‘path’ is ‘-’, then standard input is used.
 *
 * If an error code is returned, then ‘lines_read’ and ‘bytes_read’
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the vector.
 */
int mtxdistvector_read(
    struct mtxdistvector * distvector,
    enum mtxprecision precision,
    const char * path,
    bool gzip,
    int * lines_read,
    int64_t * bytes_read);

/**
 * ‘mtxdistvector_fread()’ reads a vector from a stream in Matrix
 * Market format.
 *
 * ‘precision’ is used to determine the precision to use for storing
 * the values of matrix or vector entries.
 *
 * If an error code is returned, then ‘lines_read’ and ‘bytes_read’
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the vector.
 */
int mtxdistvector_fread(
    struct mtxdistvector * distvector,
    enum mtxprecision precision,
    FILE * f,
    int * lines_read,
    int64_t * bytes_read,
    size_t line_max,
    char * linebuf);

#ifdef LIBMTX_HAVE_LIBZ
/**
 * ‘mtxdistvector_gzread()’ reads a vector from a gzip-compressed
 * stream.
 *
 * ‘precision’ is used to determine the precision to use for storing
 * the values of matrix or vector entries.
 *
 * If an error code is returned, then ‘lines_read’ and ‘bytes_read’
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the vector.
 */
int mtxdistvector_gzread(
    struct mtxdistvector * distvector,
    enum mtxprecision precision,
    gzFile f,
    int * lines_read,
    int64_t * bytes_read,
    size_t line_max,
    char * linebuf);
#endif

/**
 * ‘mtxdistvector_write()’ writes a vector to a Matrix Market
 * file. The file may optionally be compressed by gzip.
 *
 * If ‘path’ is ‘-’, then standard output is used.
 *
 * If ‘fmt’ is ‘NULL’, then the format specifier ‘%g’ is used to print
 * floating point numbers with enough digits to ensure correct
 * round-trip conversion from decimal text and back.  Otherwise, the
 * given format string is used to print numerical values.
 *
 * The format string follows the conventions of ‘printf’. If the field
 * is ‘real’, ‘double’ or ‘complex’, then the format specifiers '%e',
 * '%E', '%f', '%F', '%g' or '%G' may be used. If the field is
 * ‘integer', then the format specifier must be '%d'. The format
 * string is ignored if the field is ‘pattern’. Field width and
 * precision may be specified (e.g., "%3.1f"), but variable field
 * width and precision (e.g., "%*.*f"), as well as length modifiers
 * (e.g., "%Lf") are not allowed.
 */
int mtxdistvector_write(
    const struct mtxdistvector * distvector,
    enum mtxfileformat mtxfmt,
    const char * path,
    bool gzip,
    const char * fmt,
    int64_t * bytes_written);

/**
 * ‘mtxdistvector_fwrite()’ writes a vector to a stream.
 *
 * If ‘fmt’ is ‘NULL’, then the format specifier ‘%g’ is used to print
 * floating point numbers with enough digits to ensure correct
 * round-trip conversion from decimal text and back.  Otherwise, the
 * given format string is used to print numerical values.
 *
 * The format string follows the conventions of ‘printf’. If the field
 * is ‘real’, ‘double’ or ‘complex’, then the format specifiers '%e',
 * '%E', '%f', '%F', '%g' or '%G' may be used. If the field is
 * ‘integer', then the format specifier must be '%d'. The format
 * string is ignored if the field is ‘pattern’. Field width and
 * precision may be specified (e.g., "%3.1f"), but variable field
 * width and precision (e.g., "%*.*f"), as well as length modifiers
 * (e.g., "%Lf") are not allowed.
 *
 * If it is not ‘NULL’, then the number of bytes written to the stream
 * is returned in ‘bytes_written’.
 */
int mtxdistvector_fwrite(
    const struct mtxdistvector * distvector,
    enum mtxfileformat mtxfmt,
    FILE * f,
    const char * fmt,
    int64_t * bytes_written);

/**
 * `mtxdistvector_fwrite_shared()' writes a distributed vector as a
 * Matrix Market file to a single stream that is shared by every
 * process in the communicator.
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
 *
 * Note that only the specified ‘root’ process will print anything to
 * the stream. Other processes will therefore send their part of the
 * distributed Matrix Market file to the root process for printing.
 *
 * This function performs collective communication and therefore
 * requires every process in the communicator to perform matching
 * calls to the function.
 */
int mtxdistvector_fwrite_shared(
    const struct mtxdistvector * mtxdistvector,
    enum mtxfileformat mtxfmt,
    FILE * f,
    const char * fmt,
    int64_t * bytes_written,
    int root,
    struct mtxdisterror * disterr);

#ifdef LIBMTX_HAVE_LIBZ
/**
 * ‘mtxdistvector_gzwrite()’ writes a vector to a gzip-compressed
 * stream.
 *
 * If ‘fmt’ is ‘NULL’, then the format specifier ‘%g’ is used to print
 * floating point numbers with enough digits to ensure correct
 * round-trip conversion from decimal text and back.  Otherwise, the
 * given format string is used to print numerical values.
 *
 * The format string follows the conventions of ‘printf’. If the field
 * is ‘real’, ‘double’ or ‘complex’, then the format specifiers '%e',
 * '%E', '%f', '%F', '%g' or '%G' may be used. If the field is
 * ‘integer', then the format specifier must be '%d'. The format
 * string is ignored if the field is ‘pattern’. Field width and
 * precision may be specified (e.g., "%3.1f"), but variable field
 * width and precision (e.g., "%*.*f"), as well as length modifiers
 * (e.g., "%Lf") are not allowed.
 *
 * If it is not ‘NULL’, then the number of bytes written to the stream
 * is returned in ‘bytes_written’.
 */
int mtxdistvector_gzwrite(
    const struct mtxdistvector * distvector,
    enum mtxfileformat mtxfmt,
    gzFile f,
    const char * fmt,
    int64_t * bytes_written);
#endif

/*
 * Level 1 BLAS operations
 */

/**
 * `mtxdistvector_swap()' swaps values of two vectors, simultaneously
 * performing ‘y <- x’ and ‘x <- y’.
 */
int mtxdistvector_swap(
    struct mtxdistvector * x,
    struct mtxdistvector * y,
    struct mtxdisterror * disterr);

/**
 * `mtxdistvector_copy()' copies values of a vector, ‘y = x’.
 */
int mtxdistvector_copy(
    struct mtxdistvector * y,
    const struct mtxdistvector * x,
    struct mtxdisterror * disterr);

/**
 * `mtxdistvector_sscal()' scales a vector by a single precision
 * floating point scalar, ‘x = a*x’.
 */
int mtxdistvector_sscal(
    float a,
    struct mtxdistvector * x,
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * `mtxdistvector_dscal()' scales a vector by a double precision
 * floating point scalar, ‘x = a*x’.
 */
int mtxdistvector_dscal(
    double a,
    struct mtxdistvector * x,
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * `mtxdistvector_saxpy()' adds a vector to another vector multiplied
 * by a single precision floating point value, ‘y = a*x + y’.
 */
int mtxdistvector_saxpy(
    float a,
    const struct mtxdistvector * x,
    struct mtxdistvector * y,
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * `mtxdistvector_daxpy()' adds a vector to another vector multiplied
 * by a double precision floating point value, ‘y = a*x + y’.
 */
int mtxdistvector_daxpy(
    double a,
    const struct mtxdistvector * x,
    struct mtxdistvector * y,
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * `mtxdistvector_saypx()' multiplies a vector by a single precision
 * floating point scalar and adds another vector, ‘y = a*y + x’.
 */
int mtxdistvector_saypx(
    float a,
    struct mtxdistvector * y,
    const struct mtxdistvector * x,
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * `mtxdistvector_daypx()' multiplies a vector by a double precision
 * floating point scalar and adds another vector, ‘y = a*y + x’.
 */
int mtxdistvector_daypx(
    double a,
    struct mtxdistvector * y,
    const struct mtxdistvector * x,
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * `mtxdistvector_sdot()' computes the Euclidean dot product of two
 * vectors in single precision floating point.
 */
int mtxdistvector_sdot(
    const struct mtxdistvector * x,
    const struct mtxdistvector * y,
    float * dot,
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * `mtxdistvector_ddot()' computes the Euclidean dot product of two
 * vectors in double precision floating point.
 */
int mtxdistvector_ddot(
    const struct mtxdistvector * x,
    const struct mtxdistvector * y,
    double * dot,
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * `mtxdistvector_cdotu()' computes the product of the transpose of a
 * complex row vector with another complex row vector in single
 * precision floating point, ‘dot := x^T*y’.
 */
int mtxdistvector_cdotu(
    const struct mtxdistvector * x,
    const struct mtxdistvector * y,
    float (* dot)[2],
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * `mtxdistvector_zdotu()' computes the product of the transpose of a
 * complex row vector with another complex row vector in double
 * precision floating point, ‘dot := x^T*y’.
 */
int mtxdistvector_zdotu(
    const struct mtxdistvector * x,
    const struct mtxdistvector * y,
    double (* dot)[2],
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * `mtxdistvector_cdotc()' computes the Euclidean dot product of two
 * complex vectors in single precision floating point, ‘dot := x^H*y’.
 */
int mtxdistvector_cdotc(
    const struct mtxdistvector * x,
    const struct mtxdistvector * y,
    float (* dot)[2],
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * `mtxdistvector_zdotc()' computes the Euclidean dot product of two
 * complex vectors in double precision floating point, ‘dot := x^H*y’.
 */
int mtxdistvector_zdotc(
    const struct mtxdistvector * x,
    const struct mtxdistvector * y,
    double (* dot)[2],
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * `mtxdistvector_snrm2()' computes the Euclidean norm of a vector in
 * single precision floating point.
 */
int mtxdistvector_snrm2(
    const struct mtxdistvector * x,
    float * nrm2,
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * `mtxdistvector_dnrm2()' computes the Euclidean norm of a vector in
 * double precision floating point.
 */
int mtxdistvector_dnrm2(
    const struct mtxdistvector * x,
    double * nrm2,
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * `mtxdistvector_sasum()' computes the sum of absolute values
 * (1-norm) of a vector in single precision floating point.
 */
int mtxdistvector_sasum(
    const struct mtxdistvector * x,
    float * asum,
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * `mtxdistvector_dasum()' computes the sum of absolute values
 * (1-norm) of a vector in double precision floating point.
 */
int mtxdistvector_dasum(
    const struct mtxdistvector * x,
    double * asum,
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * `mtxdistvector_imax()' finds the index of the first element having
 * the maximum absolute value.
 */
int mtxdistvector_imax(
    const struct mtxdistvector * x,
    int * max,
    struct mtxdisterror * disterr);
#endif

#endif
