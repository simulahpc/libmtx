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
 * Last modified: 2022-04-27
 *
 * Data structures for distributed matrices.
 */

#ifndef LIBMTX_MATRIX_DISTMATRIX_H
#define LIBMTX_MATRIX_DISTMATRIX_H

#include <libmtx/libmtx-config.h>

#ifdef LIBMTX_HAVE_MPI
#include <libmtx/precision.h>
#include <libmtx/field.h>
#include <libmtx/matrix/matrix.h>
#include <libmtx/util/partition.h>
#include <libmtx/util/transpose.h>

#include <mpi.h>

#ifdef LIBMTX_HAVE_LIBZ
#include <zlib.h>
#endif

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

struct mtxdisterror;
struct mtxdistfile;
struct mtxdistvector;
struct mtxfile;
struct mtxpartition;
struct mtxvector_dist;

/**
 * ‘mtxdistmatrix’ is a matrix distributed across multiple processes,
 * where MPI is used for communicating between processes.
 *
 * Processes are arranged in a two-dimensional grid, and matrices are
 * distributed among processes in rectangular blocks according to
 * specified partitionings of the matrix rows and columns.
 */
struct mtxdistmatrix
{
    /**
     * ‘comm’ is a two-dimensional Cartesian communicator for
     * processes among which the matrix is distributed.
     */
    MPI_Comm comm;

    /**
     * ‘comm_size’ is the size (number of processes) of the MPI
     * communicator ‘comm’. This is equal to the number of parts in
     * the two-dimensional matrix partitioning, or, equivalently, the
     * number of parts in the partitioning of the matrix rows times
     * the number of parts in the partitioning of the columns.
     */
    int comm_size;

    /**
     * ‘rank’ is the rank of the current process within the process
     * group of the communicator ‘comm’.
     */
    int rank;

    /**
     * ‘rowpart’ is a partitioning of the rows of the matrix.
     */
    struct mtxpartition rowpart;

    /**
     * ‘colpart’ is a partitioning of the columns of the matrix.
     */
    struct mtxpartition colpart;

    /**
     * ‘num_rows’ is the number of matrix rows in the entire,
     * distributed matrix. This number must be the same across all
     * processes in the communicator ‘comm’.
     */
    int64_t num_rows;

    /**
     * ‘num_columns’ is the number of matrix columns in the entire,
     * distributed matrix. This number must be the same across all
     * processes in the communicator ‘comm’.
     */
    int64_t num_columns;

    /**
     * ‘num_nonzeros’ is the total number of nonzero matrix entries of
     * the distributed sparse matrix, including those represented
     * implicitly due to symmetry. This is equal to the sum of the
     * number of nonzero matrix entries on each process
     * (‘interior.num_nonzeros’).
     */
    int64_t num_nonzeros;

    /**
     * ‘size’ is the total number of explicitly stored matrix entries
     * for the distributed sparse matrix. This is equal to the sum of
     * the number of explicitly stored matrix entries on each process
     * (‘interior.size’).
     */
    int64_t size;

    /**
     * ‘interior’ is the local, rectangular block of the distributed
     * matrix that resides on the current process.
     */
    struct mtxmatrix interior;

    /**
     * ‘rowmapsize’ is the number of matrix rows in the local part of
     * the distributed matrix that resides on the current process. It
     * is also equal to the size of the ‘rowmap’ array.
     */
    int64_t rowmapsize;

    /**
     * ‘rowmap’ is an array that maps rows of the local matrix on the
     * current process to rows of the global matrix. The size of the
     * array is equal to the number of rows in the local matrix.
     */
    int64_t * rowmap;

    /**
     * ‘colmapsize’ is the number of matrix columns in the local part
     * of the distributed matrix that resides on the current
     * process. It is also equal to the size of the ‘colmap’ array.
     */
    int64_t colmapsize;

    /**
     * ‘colmap’ is an array that maps columns of the local matrix on
     * the current process to columns of the global matrix. The size
     * of the array is equal to the number of columns in the local
     * matrix.
     */
    int64_t * colmap;
};

/*
 * Memory management
 */

/**
 * ‘mtxdistmatrix_free()’ frees storage allocated for a matrix.
 */
void mtxdistmatrix_free(
    struct mtxdistmatrix * distmatrix);

/**
 * ‘mtxdistmatrix_alloc_copy()’ allocates storage for a copy of a
 * distributed matrix without initialising the underlying values.
 */
int mtxdistmatrix_alloc_copy(
    struct mtxdistmatrix * dst,
    const struct mtxdistmatrix * src,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistmatrix_init_copy()’ creates a copy of a distributed matrix.
 */
int mtxdistmatrix_init_copy(
    struct mtxdistmatrix * dst,
    const struct mtxdistmatrix * src,
    struct mtxdisterror * disterr);

/*
 * Distributed matrices in array format
 */

/**
 * ‘mtxdistmatrix_alloc_array()’ allocates a distributed matrix in
 * array format.
 */
int mtxdistmatrix_alloc_array(
    struct mtxdistmatrix * distmatrix,
    enum mtxfield field,
    enum mtxprecision precision,
    enum mtxsymmetry symmetry,
    int num_local_rows,
    int num_local_columns,
    const struct mtxpartition * rowpart,
    const struct mtxpartition * colpart,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistmatrix_init_array_real_single()’ allocates and initialises
 * a distributed matrix in array format with real, single precision
 * coefficients.
 */
int mtxdistmatrix_init_array_real_single(
    struct mtxdistmatrix * distmatrix,
    enum mtxsymmetry symmetry,
    int num_local_rows,
    int num_local_columns,
    const float * data,
    const struct mtxpartition * rowpart,
    const struct mtxpartition * colpart,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistmatrix_init_array_real_double()’ allocates and initialises
 * a distributed matrix in array format with real, double precision
 * coefficients.
 */
int mtxdistmatrix_init_array_real_double(
    struct mtxdistmatrix * distmatrix,
    enum mtxsymmetry symmetry,
    int num_local_rows,
    int num_local_columns,
    const double * data,
    const struct mtxpartition * rowpart,
    const struct mtxpartition * colpart,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistmatrix_init_array_complex_single()’ allocates and
 * initialises a distributed matrix in array format with complex,
 * single precision coefficients.
 */
int mtxdistmatrix_init_array_complex_single(
    struct mtxdistmatrix * distmatrix,
    enum mtxsymmetry symmetry,
    int num_local_rows,
    int num_local_columns,
    const float (* data)[2],
    const struct mtxpartition * rowpart,
    const struct mtxpartition * colpart,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistmatrix_init_array_complex_double()’ allocates and
 * initialises a distributed matrix in array format with complex,
 * double precision coefficients.
 */
int mtxdistmatrix_init_array_complex_double(
    struct mtxdistmatrix * distmatrix,
    enum mtxsymmetry symmetry,
    int num_local_rows,
    int num_local_columns,
    const double (* data)[2],
    const struct mtxpartition * rowpart,
    const struct mtxpartition * colpart,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistmatrix_init_array_integer_single()’ allocates and
 * initialises a distributed matrix in array format with integer,
 * single precision coefficients.
 */
int mtxdistmatrix_init_array_integer_single(
    struct mtxdistmatrix * distmatrix,
    enum mtxsymmetry symmetry,
    int num_local_rows,
    int num_local_columns,
    const int32_t * data,
    const struct mtxpartition * rowpart,
    const struct mtxpartition * colpart,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistmatrix_init_array_integer_double()’ allocates and
 * initialises a distributed matrix in array format with integer,
 * double precision coefficients.
 */
int mtxdistmatrix_init_array_integer_double(
    struct mtxdistmatrix * distmatrix,
    enum mtxsymmetry symmetry,
    int num_local_rows,
    int num_local_columns,
    const int64_t * data,
    const struct mtxpartition * rowpart,
    const struct mtxpartition * colpart,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/*
 * distributed matrices in coordinate format from global row and
 * column indices.
 */

/**
 * ‘mtxdistmatrix_init_global_coordinate_real_single()’ allocates and
 * initialises a distributed matrix in coordinate format with real,
 * single precision coefficients.
 */
int mtxdistmatrix_init_global_coordinate_real_single(
    struct mtxdistmatrix * A,
    enum mtxsymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    int64_t num_nonzeros,
    const int64_t * rowidx,
    const int64_t * colidx,
    const float * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/*
 * distributed matrices in coordinate format from local row and column
 * indices.
 */

/**
 * ‘mtxdistmatrix_init_local_coordinate_real_single()’ allocates and
 * initialises a distributed matrix in coordinate format with real,
 * single precision coefficients.
 */
int mtxdistmatrix_init_local_coordinate_real_single(
    struct mtxdistmatrix * A,
    enum mtxsymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    int64_t num_nonzeros,
    const int * rowidx,
    const int * colidx,
    const float * data,
    int64_t rowmapsize,
    const int64_t * rowmap,
    int64_t colmapsize,
    const int64_t * colmap,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistmatrix_init_local_coordinate_real_double()’ allocates and
 * initialises a distributed matrix in coordinate format with real,
 * double precision coefficients.
 */
int mtxdistmatrix_init_local_coordinate_real_double(
    struct mtxdistmatrix * A,
    enum mtxsymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    int64_t num_nonzeros,
    const int * rowidx,
    const int * colidx,
    const double * data,
    int64_t rowmapsize,
    const int64_t * rowmap,
    int64_t colmapsize,
    const int64_t * colmap,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/*
 * distributed matrices in coordinate format
 */

/**
 * ‘mtxdistmatrix_alloc_coordinate()’ allocates a distributed matrix
 * in coordinate format.
 */
int mtxdistmatrix_alloc_coordinate(
    struct mtxdistmatrix * distmatrix,
    enum mtxfield field,
    enum mtxprecision precision,
    enum mtxsymmetry symmetry,
    int num_local_rows,
    int num_local_columns,
    int64_t num_local_nonzeros,
    const struct mtxpartition * rowpart,
    const struct mtxpartition * colpart,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistmatrix_init_global_coordinate_real_single()’ allocates and
 * initialises a distributed matrix in coordinate format with real,
 * single precision coefficients.
 */
int mtxdistmatrix_init_global_coordinate_real_single(
    struct mtxdistmatrix * distmatrix,
    enum mtxsymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    int64_t num_nonzeros,
    const int64_t * rowidx,
    const int64_t * colidx,
    const float * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistmatrix_init_coordinate_real_single()’ allocates and
 * initialises a distributed matrix in coordinate format with real,
 * single precision coefficients.
 */
int mtxdistmatrix_init_coordinate_real_single(
    struct mtxdistmatrix * distmatrix,
    enum mtxsymmetry symmetry,
    int num_local_rows,
    int num_local_columns,
    int64_t num_local_nonzeros,
    const int * rowidx,
    const int * colidx,
    const float * data,
    const struct mtxpartition * rowpart,
    const struct mtxpartition * colpart,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistmatrix_init_coordinate_real_double()’ allocates and
 * initialises a distributed matrix in coordinate format with real,
 * double precision coefficients.
 */
int mtxdistmatrix_init_coordinate_real_double(
    struct mtxdistmatrix * distmatrix,
    enum mtxsymmetry symmetry,
    int num_local_rows,
    int num_local_columns,
    int64_t num_local_nonzeros,
    const int * rowidx,
    const int * colidx,
    const double * data,
    const struct mtxpartition * rowpart,
    const struct mtxpartition * colpart,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistmatrix_init_coordinate_complex_single()’ allocates and
 * initialises a distributed matrix in coordinate format with complex,
 * single precision coefficients.
 */
int mtxdistmatrix_init_coordinate_complex_single(
    struct mtxdistmatrix * distmatrix,
    enum mtxsymmetry symmetry,
    int num_local_rows,
    int num_local_columns,
    int64_t num_local_nonzeros,
    const int * rowidx,
    const int * colidx,
    const float (* data)[2],
    const struct mtxpartition * rowpart,
    const struct mtxpartition * colpart,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistmatrix_init_coordinate_complex_double()’ allocates and
 * initialises a distributed matrix in coordinate format with complex,
 * double precision coefficients.
 */
int mtxdistmatrix_init_coordinate_complex_double(
    struct mtxdistmatrix * distmatrix,
    enum mtxsymmetry symmetry,
    int num_local_rows,
    int num_local_columns,
    int64_t num_local_nonzeros,
    const int * rowidx,
    const int * colidx,
    const double (* data)[2],
    const struct mtxpartition * rowpart,
    const struct mtxpartition * colpart,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistmatrix_init_coordinate_integer_single()’ allocates and
 * initialises a distributed matrix in coordinate format with integer,
 * single precision coefficients.
 */
int mtxdistmatrix_init_coordinate_integer_single(
    struct mtxdistmatrix * distmatrix,
    enum mtxsymmetry symmetry,
    int num_local_rows,
    int num_local_columns,
    int64_t num_local_nonzeros,
    const int * rowidx,
    const int * colidx,
    const int32_t * data,
    const struct mtxpartition * rowpart,
    const struct mtxpartition * colpart,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistmatrix_init_coordinate_integer_double()’ allocates and
 * initialises a distributed matrix in coordinate format with integer,
 * double precision coefficients.
 */
int mtxdistmatrix_init_coordinate_integer_double(
    struct mtxdistmatrix * distmatrix,
    enum mtxsymmetry symmetry,
    int num_local_rows,
    int num_local_columns,
    int64_t num_local_nonzeros,
    const int * rowidx,
    const int * colidx,
    const int64_t * data,
    const struct mtxpartition * rowpart,
    const struct mtxpartition * colpart,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistmatrix_init_coordinate_pattern()’ allocates and initialises
 * a distributed matrix in coordinate format with boolean
 * coefficients.
 */
int mtxdistmatrix_init_coordinate_pattern(
    struct mtxdistmatrix * distmatrix,
    enum mtxsymmetry symmetry,
    int num_local_rows,
    int num_local_columns,
    int64_t num_local_nonzeros,
    const int * rowidx,
    const int * colidx,
    const struct mtxpartition * rowpart,
    const struct mtxpartition * colpart,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/*
 * Row and column vectors
 */

/**
 * ‘mtxdistmatrix_alloc_row_vector()’ allocates a distributed row
 * vector for a given distributed matrix. A row vector is a vector
 * whose length equal to a single row of the matrix, and it is
 * distributed among processes in a given process row according to the
 * column partitioning of the distributed matrix.
 */
int mtxdistmatrix_alloc_row_vector(
    const struct mtxdistmatrix * distmatrix,
    struct mtxdistvector * distvector,
    enum mtxvectortype vector_type,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistmatrix_alloc_column_vector()’ allocates a distributed
 * column vector for a given distributed matrix. A column vector is a
 * vector whose length equal to a single column of the matrix, and it
 * is distributed among processes in a given process column according
 * to the row partitioning of the distributed matrix.
 */
int mtxdistmatrix_alloc_column_vector(
    const struct mtxdistmatrix * distmatrix,
    struct mtxdistvector * distvector,
    enum mtxvectortype vector_type,
    struct mtxdisterror * disterr);

/*
 * Convert to and from Matrix Market format
 */

/**
 * ‘mtxdistmatrix_from_mtxfile()’ converts a matrix in Matrix Market
 * format to a distributed matrix.
 *
 * The ‘type’ argument may be used to specify a desired storage format
 * or implementation for the underlying ‘mtxmatrix’ on each
 * process. If ‘type’ is ‘mtxmatrix_auto’, then the type of
 * ‘mtxmatrix’ is chosen to match the type of ‘mtxfile’. That is,
 * ‘mtxmatrix_array’ is used if ‘mtxfile’ is in array format, and
 * ‘mtxmatrix_coordinate’ is used if ‘mtxfile’ is in coordinate
 * format.
 *
 * Furthermore, ‘rowpart’ and ‘colpart’ must be partitionings of the
 * rows and columns of the global matrix. Therefore, ‘rowpart->size’
 * must be equal to the number of rows and ‘colpart->size’ must be
 * equal to the number of columns in ‘mtxfile’. There must be at least
 * one MPI process in the communicator ‘comm’ for each part in the
 * partitioned matrix (i.e., the number of row parts times the number
 * of column parts).
 *
 * If ‘rowpart’ and ‘colpart’ are both ‘NULL’, then the rows are
 * partitioned into contiguous blocks of equal size by default.
 */
int mtxdistmatrix_from_mtxfile(
    struct mtxdistmatrix * dst,
    const struct mtxfile * src,
    enum mtxmatrixtype matrix_type,
    const struct mtxpartition * rowpart,
    const struct mtxpartition * colpart,
    MPI_Comm comm,
    int root,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistmatrix_to_mtxfile()’ gathers a distributed matrix onto a
 * single, root process and converts it to a (non-distributed) Matrix
 * Market file on that process.
 *
 * This function performs collective communication and therefore
 * requires every process in the communicator to perform matching
 * calls to this function.
 */
int mtxdistmatrix_to_mtxfile(
    struct mtxfile * dst,
    const struct mtxdistmatrix * src,
    enum mtxfileformat mtxfmt,
    int root,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistmatrix_from_mtxdistfile()’ converts a matrix in distributed
 * Matrix Market format to a distributed matrix.
 */
int mtxdistmatrix_from_mtxdistfile(
    struct mtxdistmatrix * dst,
    const struct mtxdistfile * src,
    enum mtxmatrixtype matrix_type,
    const struct mtxpartition * rowpart,
    const struct mtxpartition * colpart,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistmatrix_to_mtxdistfile()’ converts a distributed matrix to a
 * matrix in a distributed Matrix Market format.
 */
int mtxdistmatrix_to_mtxdistfile(
    struct mtxdistfile * dst,
    const struct mtxdistmatrix * src,
    enum mtxfileformat mtxfmt,
    struct mtxdisterror * disterr);

/*
 * I/O functions
 */

/**
 * ‘mtxdistmatrix_read()’ reads a matrix from a Matrix Market file.
 * The file may optionally be compressed by gzip.
 *
 * The ‘precision’ argument specifies which precision to use for
 * storing matrix or matrix values.
 *
 * If ‘path’ is ‘-’, then standard input is used.
 *
 * If an error code is returned, then ‘lines_read’ and ‘bytes_read’
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the matrix.
 */
int mtxdistmatrix_read(
    struct mtxdistmatrix * distmatrix,
    enum mtxprecision precision,
    enum mtxmatrixtype type,
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
 * If an error code is returned, then ‘lines_read’ and ‘bytes_read’
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the matrix.
 */
int mtxdistmatrix_fread(
    struct mtxdistmatrix * distmatrix,
    enum mtxprecision precision,
    enum mtxmatrixtype type,
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
 * If an error code is returned, then ‘lines_read’ and ‘bytes_read’
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the matrix.
 */
int mtxdistmatrix_gzread(
    struct mtxdistmatrix * distmatrix,
    enum mtxprecision precision,
    enum mtxmatrixtype type,
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
 */
int mtxdistmatrix_write(
    const struct mtxdistmatrix * distmatrix,
    enum mtxfileformat mtxfmt,
    const char * path,
    bool gzip,
    const char * fmt,
    int64_t * bytes_written);

/**
 * ‘mtxdistmatrix_fwrite()’ writes a matrix to a stream.
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
 */
int mtxdistmatrix_fwrite(
    const struct mtxdistmatrix * distmatrix,
    enum mtxfileformat mtxfmt,
    FILE * f,
    const char * fmt,
    int64_t * bytes_written);

/**
 * ‘mtxdistmatrix_fwrite_shared()’ writes a distributed matrix as a
 * Matrix Market file to a single stream that is shared by every
 * process in the communicator.
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
    enum mtxfileformat mtxfmt,
    FILE * f,
    const char * fmt,
    int64_t * bytes_written,
    int root,
    struct mtxdisterror * disterr);

#ifdef LIBMTX_HAVE_LIBZ
/**
 * ‘mtxdistmatrix_gzwrite()’ writes a matrix to a gzip-compressed
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
 * ‘integer’, then the format specifier must be '%d'. The format
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
    enum mtxfileformat mtxfmt,
    gzFile f,
    const char * fmt,
    int64_t * bytes_written);
#endif

/*
 * Level 1 BLAS operations
 */

/**
 * ‘mtxdistmatrix_swap()’ swaps values of two matrices, simultaneously
 * performing ‘y <- x’ and ‘x <- y’.
 */
int mtxdistmatrix_swap(
    struct mtxdistmatrix * x,
    struct mtxdistmatrix * y,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistmatrix_copy()’ copies values of a matrix, ‘y = x’.
 */
int mtxdistmatrix_copy(
    struct mtxdistmatrix * y,
    const struct mtxdistmatrix * x,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistmatrix_sscal()’ scales a matrix by a single precision
 * floating point scalar, ‘x = a*x’.
 */
int mtxdistmatrix_sscal(
    float a,
    struct mtxdistmatrix * x,
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistmatrix_dscal()’ scales a matrix by a double precision
 * floating point scalar, ‘x = a*x’.
 */
int mtxdistmatrix_dscal(
    double a,
    struct mtxdistmatrix * x,
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistmatrix_saxpy()’ adds a matrix to another matrix multiplied
 * by a single precision floating point value, ‘y = a*x + y’.
 */
int mtxdistmatrix_saxpy(
    float a,
    const struct mtxdistmatrix * x,
    struct mtxdistmatrix * y,
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistmatrix_daxpy()’ adds a matrix to another matrix multiplied
 * by a double precision floating point value, ‘y = a*x + y’.
 */
int mtxdistmatrix_daxpy(
    double a,
    const struct mtxdistmatrix * x,
    struct mtxdistmatrix * y,
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistmatrix_saypx()’ multiplies a matrix by a single precision
 * floating point scalar and adds another matrix, ‘y = a*y + x’.
 */
int mtxdistmatrix_saypx(
    float a,
    struct mtxdistmatrix * y,
    const struct mtxdistmatrix * x,
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistmatrix_daypx()’ multiplies a matrix by a double precision
 * floating point scalar and adds another matrix, ‘y = a*y + x’.
 */
int mtxdistmatrix_daypx(
    double a,
    struct mtxdistmatrix * y,
    const struct mtxdistmatrix * x,
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistmatrix_sdot()’ computes the Euclidean dot product of two
 * matrices in single precision floating point.
 */
int mtxdistmatrix_sdot(
    const struct mtxdistmatrix * x,
    const struct mtxdistmatrix * y,
    float * dot,
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistmatrix_ddot()’ computes the Euclidean dot product of two
 * matrices in double precision floating point.
 */
int mtxdistmatrix_ddot(
    const struct mtxdistmatrix * x,
    const struct mtxdistmatrix * y,
    double * dot,
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistmatrix_cdotu()’ computes the product of the transpose of a
 * complex row matrix with another complex row matrix in single
 * precision floating point, ‘dot := x^T*y’.
 */
int mtxdistmatrix_cdotu(
    const struct mtxdistmatrix * x,
    const struct mtxdistmatrix * y,
    float (* dot)[2],
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistmatrix_zdotu()’ computes the product of the transpose of a
 * complex row matrix with another complex row matrix in double
 * precision floating point, ‘dot := x^T*y’.
 */
int mtxdistmatrix_zdotu(
    const struct mtxdistmatrix * x,
    const struct mtxdistmatrix * y,
    double (* dot)[2],
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistmatrix_cdotc()’ computes the Euclidean dot product of two
 * complex matrices in single precision floating point, ‘dot := x^H*y’.
 */
int mtxdistmatrix_cdotc(
    const struct mtxdistmatrix * x,
    const struct mtxdistmatrix * y,
    float (* dot)[2],
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistmatrix_zdotc()’ computes the Euclidean dot product of two
 * complex matrices in double precision floating point, ‘dot := x^H*y’.
 */
int mtxdistmatrix_zdotc(
    const struct mtxdistmatrix * x,
    const struct mtxdistmatrix * y,
    double (* dot)[2],
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistmatrix_snrm2()’ computes the Euclidean norm of a matrix in
 * single precision floating point.
 */
int mtxdistmatrix_snrm2(
    const struct mtxdistmatrix * x,
    float * nrm2,
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistmatrix_dnrm2()’ computes the Euclidean norm of a matrix in
 * double precision floating point.
 */
int mtxdistmatrix_dnrm2(
    const struct mtxdistmatrix * x,
    double * nrm2,
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistmatrix_sasum()’ computes the sum of absolute values
 * (1-norm) of a matrix in single precision floating point.
 */
int mtxdistmatrix_sasum(
    const struct mtxdistmatrix * x,
    float * asum,
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistmatrix_dasum()’ computes the sum of absolute values
 * (1-norm) of a matrix in double precision floating point.
 */
int mtxdistmatrix_dasum(
    const struct mtxdistmatrix * x,
    double * asum,
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistmatrix_iamax()’ finds the index of the first element having
 * the maximum absolute value.
 */
int mtxdistmatrix_iamax(
    const struct mtxdistmatrix * x,
    int * max,
    struct mtxdisterror * disterr);

/*
 * Level 2 BLAS operations
 */

/**
 * ‘mtxdistmatrix_sgemv()’ multiplies a matrix ‘A’ or its transpose
 * ‘A'’ by a real scalar ‘alpha’ (‘α’) and a vector ‘x’, before adding
 * the result to another vector ‘y’ multiplied by another real scalar
 * ‘beta’ (‘β’). That is, ‘y = α*A*x + β*y’ or ‘y = α*A'*x + β*y’.
 *
 * The scalars ‘alpha’ and ‘beta’ are given as single precision
 * floating point numbers.
 */
int mtxdistmatrix_sgemv(
    enum mtxtransposition trans,
    float alpha,
    const struct mtxdistmatrix * A,
    const struct mtxdistvector * x,
    float beta,
    struct mtxdistvector * y,
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistmatrix_sgemv2()’ multiplies a matrix ‘A’ or its transpose
 * ‘A'’ by a real scalar ‘alpha’ (‘α’) and a vector ‘x’, before adding
 * the result to another vector ‘y’ multiplied by another real scalar
 * ‘beta’ (‘β’). That is, ‘y = α*A*x + β*y’ or ‘y = α*A'*x + β*y’.
 *
 * The scalars ‘alpha’ and ‘beta’ are given as single precision
 * floating point numbers.
 */
int mtxdistmatrix_sgemv2(
    enum mtxtransposition trans,
    float alpha,
    const struct mtxdistmatrix * A,
    const struct mtxvector_dist * x,
    float beta,
    struct mtxvector_dist * y,
    int64_t * num_flops,
    struct mtxdisterror * disterr);

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
    enum mtxtransposition trans,
    double alpha,
    const struct mtxdistmatrix * A,
    const struct mtxdistvector * x,
    double beta,
    struct mtxdistvector * y,
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistmatrix_cgemv()’ multiplies a complex-valued matrix ‘A’, its
 * transpose ‘A'’ or its conjugate transpose ‘Aᴴ’ by a complex scalar
 * ‘alpha’ (‘α’) and a vector ‘x’, before adding the result to another
 * vector ‘y’ multiplied by another complex scalar ‘beta’ (‘β’).  That
 * is, ‘y = α*A*x + β*y’, ‘y = α*A'*x + β*y’ or ‘y = α*Aᴴ*x + β*y’.
 *
 * The complex scalars ‘alpha’ and ‘beta’ are given as pairs of single
 * precision floating point numbers.
 */
int mtxdistmatrix_cgemv(
    enum mtxtransposition trans,
    float alpha[2],
    const struct mtxdistmatrix * A,
    const struct mtxdistvector * x,
    float beta[2],
    struct mtxdistvector * y,
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistmatrix_zgemv()’ multiplies a complex-valued matrix ‘A’, its
 * transpose ‘A'’ or its conjugate transpose ‘Aᴴ’ by a complex scalar
 * ‘alpha’ (‘α’) and a vector ‘x’, before adding the result to another
 * vector ‘y’ multiplied by another complex scalar ‘beta’ (‘β’).  That
 * is, ‘y = α*A*x + β*y’, ‘y = α*A'*x + β*y’ or ‘y = α*Aᴴ*x + β*y’.
 *
 * The complex scalars ‘alpha’ and ‘beta’ are given as pairs of double
 * precision floating point numbers.
 */
int mtxdistmatrix_zgemv(
    enum mtxtransposition trans,
    double alpha[2],
    const struct mtxdistmatrix * A,
    const struct mtxdistvector * x,
    double beta[2],
    struct mtxdistvector * y,
    int64_t * num_flops,
    struct mtxdisterror * disterr);
#endif

#endif
