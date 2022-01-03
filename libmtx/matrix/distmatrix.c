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
 * Authors: James D. Trotter <james@simula.no> Last modified:
 * 2022-01-03
 *
 * Data structures for distributed matrices.
 */

#include <libmtx/libmtx-config.h>

#ifdef LIBMTX_HAVE_MPI
#include <libmtx/error.h>
#include <libmtx/matrix/distmatrix.h>
#include <libmtx/matrix/matrix.h>
#include <libmtx/mtx/precision.h>
#include <libmtx/mtxdistfile/mtxdistfile.h>
#include <libmtx/mtxfile/header.h>
#include <libmtx/mtxfile/mtxfile.h>
#include <libmtx/mtxfile/size.h>
#include <libmtx/util/field.h>
#include <libmtx/util/partition.h>
#include <libmtx/vector/distvector.h>
#include <libmtx/vector/vector.h>

#include <mpi.h>

#ifdef LIBMTX_HAVE_LIBZ
#include <zlib.h>
#endif

#include <errno.h>

#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

/*
 * Memory management
 */

/**
 * ‘mtxdistmatrix_free()’ frees storage allocated for a distributed
 * matrix.
 */
void mtxdistmatrix_free(
    struct mtxdistmatrix * distmatrix);

/**
 * ‘mtxdistmatrix_alloc_copy()’ allocates storage for a copy of a
 * distributed matrix without initialising the underlying values.
 */
int mtxdistmatrix_alloc_copy(
    struct mtxdistmatrix * dst,
    const struct mtxdistmatrix * src);

/**
 * ‘mtxdistmatrix_init_copy()’ creates a copy of a distributed matrix.
 */
int mtxdistmatrix_init_copy(
    struct mtxdistmatrix * dst,
    const struct mtxdistmatrix * src);

/*
 * Distributed matrices in array format
 */

/**
 * ‘mtxdistmatrix_alloc_array()’ allocates a distributed matrix in
 * array format.
 */
int mtxdistmatrix_alloc_array(
    struct mtxdistmatrix * distmatrix,
    enum mtx_field_ field,
    enum mtx_precision precision,
    int num_rows,
    int num_columns,
    MPI_Comm comm,
    struct mtxmpierror * mpierror);

/**
 * ‘mtxdistmatrix_init_array_real_single()’ allocates and initialises
 * a distributed matrix in array format with real, single precision
 * coefficients.
 */
int mtxdistmatrix_init_array_real_single(
    struct mtxdistmatrix * distmatrix,
    int num_rows,
    int num_columns,
    const float * data,
    MPI_Comm comm,
    struct mtxmpierror * mpierror);

/**
 * ‘mtxdistmatrix_init_array_real_double()’ allocates and initialises
 * a distributed matrix in array format with real, double precision
 * coefficients.
 */
int mtxdistmatrix_init_array_real_double(
    struct mtxdistmatrix * distmatrix,
    int num_rows,
    int num_columns,
    const double * data,
    MPI_Comm comm,
    struct mtxmpierror * mpierror);

/**
 * ‘mtxdistmatrix_init_array_complex_single()’ allocates and
 * initialises a distributed matrix in array format with complex,
 * single precision coefficients.
 */
int mtxdistmatrix_init_array_complex_single(
    struct mtxdistmatrix * distmatrix,
    int num_rows,
    int num_columns,
    const float (* data)[2],
    MPI_Comm comm,
    struct mtxmpierror * mpierror);

/**
 * ‘mtxdistmatrix_init_array_complex_double()’ allocates and
 * initialises a distributed matrix in array format with complex,
 * double precision coefficients.
 */
int mtxdistmatrix_init_array_complex_double(
    struct mtxdistmatrix * distmatrix,
    int num_rows,
    int num_columns,
    const double (* data)[2],
    MPI_Comm comm,
    struct mtxmpierror * mpierror);

/**
 * ‘mtxdistmatrix_init_array_integer_single()’ allocates and
 * initialises a distributed matrix in array format with integer,
 * single precision coefficients.
 */
int mtxdistmatrix_init_array_integer_single(
    struct mtxdistmatrix * distmatrix,
    int num_rows,
    int num_columns,
    const int32_t * data,
    MPI_Comm comm,
    struct mtxmpierror * mpierror);

/**
 * ‘mtxdistmatrix_init_array_integer_double()’ allocates and
 * initialises a distributed matrix in array format with integer,
 * double precision coefficients.
 */
int mtxdistmatrix_init_array_integer_double(
    struct mtxdistmatrix * distmatrix,
    int num_rows,
    int num_columns,
    const int64_t * data,
    MPI_Comm comm,
    struct mtxmpierror * mpierror);

/*
 * Distributed matrices in coordinate format
 */

/**
 * ‘mtxdistmatrix_alloc_coordinate()’ allocates a distributed matrix
 * in coordinate format.
 */
int mtxdistmatrix_alloc_coordinate(
    struct mtxdistmatrix * distmatrix,
    enum mtx_field_ field,
    enum mtx_precision precision,
    int num_rows,
    int num_columns,
    int64_t num_nonzeros,
    MPI_Comm comm,
    struct mtxmpierror * mpierror);

/**
 * ‘mtxdistmatrix_init_coordinate_real_single()’ allocates and
 * initialises a distributed matrix in coordinate format with real,
 * single precision coefficients.
 */
int mtxdistmatrix_init_coordinate_real_single(
    struct mtxdistmatrix * distmatrix,
    int num_rows,
    int num_columns,
    int64_t num_nonzeros,
    const int * rowidx,
    const int * colidx,
    const float * data,
    MPI_Comm comm,
    struct mtxmpierror * mpierror);

/**
 * ‘mtxdistmatrix_init_coordinate_real_double()’ allocates and
 * initialises a distributed matrix in coordinate format with real,
 * double precision coefficients.
 */
int mtxdistmatrix_init_coordinate_real_double(
    struct mtxdistmatrix * distmatrix,
    int num_rows,
    int num_columns,
    int64_t num_nonzeros,
    const int * rowidx,
    const int * colidx,
    const double * data,
    MPI_Comm comm,
    struct mtxmpierror * mpierror);

/**
 * ‘mtxdistmatrix_init_coordinate_complex_single()’ allocates and
 * initialises a distributed matrix in coordinate format with complex,
 * single precision coefficients.
 */
int mtxdistmatrix_init_coordinate_complex_single(
    struct mtxdistmatrix * distmatrix,
    int num_rows,
    int num_columns,
    int64_t num_nonzeros,
    const int * rowidx,
    const int * colidx,
    const float (* data)[2],
    MPI_Comm comm,
    struct mtxmpierror * mpierror);

/**
 * ‘mtxdistmatrix_init_coordinate_complex_double()’ allocates and
 * initialises a distributed matrix in coordinate format with complex,
 * double precision coefficients.
 */
int mtxdistmatrix_init_coordinate_complex_double(
    struct mtxdistmatrix * distmatrix,
    int num_rows,
    int num_columns,
    int64_t num_nonzeros,
    const int * rowidx,
    const int * colidx,
    const double (* data)[2],
    MPI_Comm comm,
    struct mtxmpierror * mpierror);

/**
 * ‘mtxdistmatrix_init_coordinate_integer_single()’ allocates and
 * initialises a distributed matrix in coordinate format with integer,
 * single precision coefficients.
 */
int mtxdistmatrix_init_coordinate_integer_single(
    struct mtxdistmatrix * distmatrix,
    int num_rows,
    int num_columns,
    int64_t num_nonzeros,
    const int * rowidx,
    const int * colidx,
    const int32_t * data,
    MPI_Comm comm,
    struct mtxmpierror * mpierror);

/**
 * ‘mtxdistmatrix_init_coordinate_integer_double()’ allocates and
 * initialises a distributed matrix in coordinate format with integer,
 * double precision coefficients.
 */
int mtxdistmatrix_init_coordinate_integer_double(
    struct mtxdistmatrix * distmatrix,
    int num_rows,
    int num_columns,
    int64_t num_nonzeros,
    const int * rowidx,
    const int * colidx,
    const int64_t * data,
    MPI_Comm comm,
    struct mtxmpierror * mpierror);

/**
 * ‘mtxdistmatrix_init_coordinate_pattern()’ allocates and initialises
 * a distributed matrix in coordinate format with boolean
 * coefficients.
 */
int mtxdistmatrix_init_coordinate_pattern(
    struct mtxdistmatrix * distmatrix,
    int num_rows,
    int num_columns,
    int64_t num_nonzeros,
    const int * rowidx,
    const int * colidx,
    MPI_Comm comm,
    struct mtxmpierror * mpierror);

/*
 * Row and column vectors
 */

/**
 * ‘mtxdistmatrix_alloc_row_vector()’ allocates a distributed row
 * vector for a given distributed matrix, where a row vector is a
 * vector whose length equal to a single row of the matrix.
 */
int mtxdistmatrix_alloc_row_vector(
    const struct mtxdistmatrix * matrix,
    struct mtxdistvector * vector,
    enum mtxvector_type vector_type);

/**
 * ‘mtxdistmatrix_alloc_column_vector()’ allocates a distributed
 * column vector for a given distributed matrix, where a column vector
 * is a vector whose length equal to a single column of the matrix.
 */
int mtxdistmatrix_alloc_column_vector(
    const struct mtxdistmatrix * matrix,
    struct mtxdistvector * vector,
    enum mtxvector_type vector_type);

/*
 * Convert to and from Matrix Market format
 */

/**
 * ‘mtxdistmatrix_from_mtxfile()’ converts a matrix in Matrix Market
 * format to a distributed matrix.
 */
int mtxdistmatrix_from_mtxfile(
    struct mtxdistmatrix * distmatrix,
    const struct mtxfile * mtxfile,
    enum mtxmatrix_type matrix_type,
    MPI_Comm comm,
    int root,
    struct mtxmpierror * mpierror);

/**
 * ‘mtxdistmatrix_from_mtxdistfile()’ converts a matrix in distributed
 * Matrix Market format to a distributed matrix.
 *
 * TODO: This function should also ensure that the distributed matrix
 * is partitioned (that is, each location in the matrix is assigned to
 * a single process.)  Furthermore, the array and coordinate matrices
 * on each process should ensure that there are no duplicate entries.
 */
int mtxdistmatrix_from_mtxdistfile(
    struct mtxdistmatrix * distmatrix,
    const struct mtxdistfile * mtxdistfile,
    enum mtxmatrix_type matrix_type,
    MPI_Comm comm,
    struct mtxmpierror * mpierror);

/**
 * ‘mtxdistmatrix_to_mtxdistfile()’ converts a distributed matrix to a
 * matrix in a distributed Matrix Market format.
 */
int mtxdistmatrix_to_mtxdistfile(
    const struct mtxdistmatrix * distmatrix,
    struct mtxdistfile * mtxdistfile,
    struct mtxmpierror * mpierror);

/*
 * I/O functions
 */

/**
 * ‘mtxdistmatrix_read()’ reads a distributed matrix from a Matrix
 * Market file.  The file may optionally be compressed by gzip.
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
 * If an error code is returned, then ‘lines_read’ and ‘bytes_read’
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the matrix.
 */
int mtxdistmatrix_read(
    struct mtxdistmatrix * distmatrix,
    enum mtx_precision precision,
    enum mtxmatrix_type type,
    const char * path,
    bool gzip,
    int * lines_read,
    int64_t * bytes_read);

/**
 * ‘mtxdistmatrix_fread()’ reads a matrix from a stream in Matrix
 * Market format.
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
int mtxdistmatrix_fread(
    struct mtxdistmatrix * distmatrix,
    enum mtx_precision precision,
    enum mtxmatrix_type type,
    FILE * f,
    int * lines_read,
    int64_t * bytes_read,
    size_t line_max,
    char * linebuf);

#ifdef LIBMTX_HAVE_LIBZ
/**
 * ‘mtxdistmatrix_gzread()’ reads a matrix from a gzip-compressed
 * stream.
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
int mtxdistmatrix_gzread(
    struct mtxdistmatrix * distmatrix,
    enum mtx_precision precision,
    enum mtxmatrix_type type,
    gzFile f,
    int * lines_read,
    int64_t * bytes_read,
    size_t line_max,
    char * linebuf);
#endif

/**
 * ‘mtxdistmatrix_write()’ writes a matrix to a Matrix Market
 * file. The file may optionally be compressed by gzip.
 *
 * If ‘path’ is ‘-’, then standard output is used.
 *
 * If ‘fmt’ is ‘NULL’, then the format specifier ‘%g’ is used to print
 * floating point numbers with with enough digits to ensure correct
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
int mtxdistmatrix_write(
    const struct mtxdistmatrix * distmatrix,
    const char * path,
    bool gzip,
    const char * fmt,
    int64_t * bytes_written);

/**
 * ‘mtxdistmatrix_fwrite()’ writes a matrix to a stream.
 *
 * If ‘fmt’ is ‘NULL’, then the format specifier ‘%g’ is used to print
 * floating point numbers with with enough digits to ensure correct
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
int mtxdistmatrix_fwrite(
    const struct mtxdistmatrix * distmatrix,
    FILE * f,
    const char * fmt,
    int64_t * bytes_written);

/**
 * `mtxdistmatrix_fwrite_shared()' writes a distributed matrix as a
 * Matrix Market file to a single stream that is shared by every
 * process in the communicator.
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
 * Note that only the specified ‘root’ process will print anything to
 * the stream. Other processes will therefore send their part of the
 * distributed Matrix Market file to the root process for printing.
 *
 * This function performs collective communication and therefore
 * requires every process in the communicator to perform matching
 * calls to the function.
 */
int mtxdistmatrix_fwrite_shared(
    const struct mtxdistmatrix * mtxdistmatrix,
    FILE * f,
    const char * fmt,
    int64_t * bytes_written,
    int root,
    struct mtxmpierror * mpierror);

#ifdef LIBMTX_HAVE_LIBZ
/**
 * ‘mtxdistmatrix_gzwrite()’ writes a matrix to a gzip-compressed
 * stream.
 *
 * If ‘fmt’ is ‘NULL’, then the format specifier ‘%g’ is used to print
 * floating point numbers with with enough digits to ensure correct
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
int mtxdistmatrix_gzwrite(
    const struct mtxdistmatrix * distmatrix,
    gzFile f,
    const char * fmt,
    int64_t * bytes_written);
#endif

/*
 * Level 2 BLAS operations
 */

/**
 * ‘mtxdistmatrix_sgemv()’ multiplies a matrix ‘A’ or its transpose ‘A'’
 * by a real scalar ‘alpha’ (‘α’) and a vector ‘x’, before adding the
 * result to another vector ‘y’ multiplied by another real scalar
 * ‘beta’ (‘β’). That is, ‘y = α*A*x + β*y’ or ‘y = α*A'*x + β*y’.
 *
 * The scalars ‘alpha’ and ‘beta’ are given as single precision
 * floating point numbers.
 */
int mtxdistmatrix_sgemv(
    enum mtx_trans_type trans,
    float alpha,
    const struct mtxdistmatrix * A,
    const struct mtxdistvector * x,
    float beta,
    struct mtxdistvector * y,
    struct mtxmpierror * mpierror);

/**
 * ‘mtxdistmatrix_dgemv()’ multiplies a matrix ‘A’ or its transpose ‘A'’
 * by a real scalar ‘alpha’ (‘α’) and a vector ‘x’, before adding the
 * result to another vector ‘y’ multiplied by another scalar real
 * ‘beta’ (‘β’).  That is, ‘y = α*A*x + β*y’ or ‘y = α*A'*x + β*y’.
 *
 * The scalars ‘alpha’ and ‘beta’ are given as double precision
 * floating point numbers.
 */
int mtxdistmatrix_dgemv(
    enum mtx_trans_type trans,
    double alpha,
    const struct mtxdistmatrix * A,
    const struct mtxdistvector * x,
    double beta,
    struct mtxdistvector * y,
    struct mtxmpierror * mpierror);

/**
 * ‘mtxdistmatrix_cgemv()’ multiplies a complex-valued matrix ‘A’, its
 * transpose ‘A'’ or its conjugate transpose ‘Aᴴ’ by a complex scalar
 * ‘alpha’ (‘α’) and a vector ‘x’, before adding the result to another
 * vector ‘y’ multiplied by another complex scalar ‘beta’ (‘β’).  That
 * is, ‘y = α*A*x + β*y’, ‘y = α*A'*x + β*y’ or ‘y = α*Aᴴ*x + β*y’.
 *
 * The scalars ‘alpha’ and ‘beta’ are given as single precision
 * floating point numbers.
 */
int mtxdistmatrix_cgemv(
    enum mtx_trans_type trans,
    float alpha[2],
    const struct mtxdistmatrix * A,
    const struct mtxdistvector * x,
    float beta[2],
    struct mtxdistvector * y,
    struct mtxmpierror * mpierror);

/**
 * ‘mtxdistmatrix_zgemv()’ multiplies a complex-valued matrix ‘A’, its
 * transpose ‘A'’ or its conjugate transpose ‘Aᴴ’ by a complex scalar
 * ‘alpha’ (‘α’) and a vector ‘x’, before adding the result to another
 * vector ‘y’ multiplied by another complex scalar ‘beta’ (‘β’).  That
 * is, ‘y = α*A*x + β*y’, ‘y = α*A'*x + β*y’ or ‘y = α*Aᴴ*x + β*y’.
 *
 * The scalars ‘alpha’ and ‘beta’ are given as double precision
 * floating point numbers.
 */
int mtxdistmatrix_zgemv(
    enum mtx_trans_type trans,
    double alpha[2],
    const struct mtxdistmatrix * A,
    const struct mtxdistvector * x,
    double beta[2],
    struct mtxdistvector * y,
    struct mtxmpierror * mpierror);
#endif
