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
 * Last modified: 2022-01-11
 *
 * Matrix Market files distributed among multiple processes with MPI
 * for inter-process communication.
 */

#ifndef LIBMTX_MTXDISTFILE_MTXDISTFILE_H
#define LIBMTX_MTXDISTFILE_MTXDISTFILE_H

#include <libmtx/libmtx-config.h>

#include <libmtx/mtx/precision.h>
#include <libmtx/mtxfile/comments.h>
#include <libmtx/mtxfile/data.h>
#include <libmtx/mtxfile/header.h>
#include <libmtx/mtxfile/mtxfile.h>
#include <libmtx/mtxfile/size.h>

#ifdef LIBMTX_HAVE_MPI
#include <mpi.h>
#endif

#ifdef LIBMTX_HAVE_LIBZ
#include <zlib.h>
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

struct mtxdisterror;
struct mtxpartition;

#ifdef LIBMTX_HAVE_MPI
/**
 * ‘mtxdistfile’ represents a file in the Matrix Market file format
 * distributed among multiple processes, where MPI is used for
 * communicating between processes.
 */
struct mtxdistfile
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
     * ‘mtxfile’ is the part of the Matrix Market file owned by the
     * current process.
     */
    struct mtxfile mtxfile;
};

/*
 * Memory management
 */

/**
 * ‘mtxdistfile_init()’ creates a distributed Matrix Market file from
 * Matrix Market files on each process in a communicator.
 *
 * This function performs collective communication and therefore
 * requires every process in the communicator to perform matching
 * calls to the function.
 */
int mtxdistfile_init(
    struct mtxdistfile * mtxdistfile,
    const struct mtxfile * mtxfile,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistfile_free()’ frees storage allocated for a distributed
 * Matrix Market file.
 */
void mtxdistfile_free(
    struct mtxdistfile * mtxdistfile);

/**
 * ‘mtxdistfile_alloc_copy()’ allocates storage for a copy of a
 * distributed Matrix Market file without initialising the underlying
 * values.
 */
int mtxdistfile_alloc_copy(
    struct mtxdistfile * dst,
    const struct mtxdistfile * src,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistfile_init_copy()’ creates a copy of a distributed Matrix
 * Market file.
 */
int mtxdistfile_init_copy(
    struct mtxdistfile * dst,
    const struct mtxdistfile * src,
    struct mtxdisterror * disterr);

/*
 * Matrix array formats
 */

/**
 * ‘mtxdistfile_alloc_matrix_array()’ allocates a distributed matrix
 * in array format.
 */
int mtxdistfile_alloc_matrix_array(
    struct mtxdistfile * mtxdistfile,
    enum mtxfilefield field,
    enum mtxfilesymmetry symmetry,
    enum mtxprecision precision,
    int num_rows,
    int num_columns,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistfile_init_matrix_array_real_single()’ allocates and
 * initialises a distributed matrix in array format with real, single
 * precision coefficients.
 */
int mtxdistfile_init_matrix_array_real_single(
    struct mtxdistfile * mtxdistfile,
    enum mtxfilesymmetry symmetry,
    int num_rows,
    int num_columns,
    const float * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistfile_init_matrix_array_real_double()’ allocates and initialises
 * a matrix in array format with real, double precision coefficients.
 */
int mtxdistfile_init_matrix_array_real_double(
    struct mtxdistfile * mtxdistfile,
    enum mtxfilesymmetry symmetry,
    int num_rows,
    int num_columns,
    const double * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistfile_init_matrix_array_complex_single()’ allocates and
 * initialises a distributed matrix in array format with complex,
 * single precision coefficients.
 */
int mtxdistfile_init_matrix_array_complex_single(
    struct mtxdistfile * mtxdistfile,
    enum mtxfilesymmetry symmetry,
    int num_rows,
    int num_columns,
    const float (* data)[2],
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistfile_init_matrix_array_complex_double()’ allocates and
 * initialises a matrix in array format with complex, double precision
 * coefficients.
 */
int mtxdistfile_init_matrix_array_complex_double(
    struct mtxdistfile * mtxdistfile,
    enum mtxfilesymmetry symmetry,
    int num_rows,
    int num_columns,
    const double (* data)[2],
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistfile_init_matrix_array_integer_single()’ allocates and
 * initialises a distributed matrix in array format with integer,
 * single precision coefficients.
 */
int mtxdistfile_init_matrix_array_integer_single(
    struct mtxdistfile * mtxdistfile,
    enum mtxfilesymmetry symmetry,
    int num_rows,
    int num_columns,
    const int32_t * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistfile_init_matrix_array_integer_double()’ allocates and
 * initialises a matrix in array format with integer, double precision
 * coefficients.
 */
int mtxdistfile_init_matrix_array_integer_double(
    struct mtxdistfile * mtxdistfile,
    enum mtxfilesymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/*
 * Vector array formats
 */

/**
 * ‘mtxdistfile_alloc_vector_array()’ allocates a distributed vector
 * in array format.
 */
int mtxdistfile_alloc_vector_array(
    struct mtxdistfile * mtxdistfile,
    enum mtxfilefield field,
    enum mtxprecision precision,
    int num_rows,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistfile_init_vector_array_real_single()’ allocates and
 * initialises a distributed vector in array format with real, single
 * precision coefficients.
 */
int mtxdistfile_init_vector_array_real_single(
    struct mtxdistfile * mtxdistfile,
    int num_rows,
    const float * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistfile_init_vector_array_real_double()’ allocates and initialises
 * a vector in array format with real, double precision coefficients.
 */
int mtxdistfile_init_vector_array_real_double(
    struct mtxdistfile * mtxdistfile,
    int num_rows,
    const double * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistfile_init_vector_array_complex_single()’ allocates and
 * initialises a distributed vector in array format with complex,
 * single precision coefficients.
 */
int mtxdistfile_init_vector_array_complex_single(
    struct mtxdistfile * mtxdistfile,
    int num_rows,
    const float (* data)[2],
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistfile_init_vector_array_complex_double()’ allocates and
 * initialises a vector in array format with complex, double precision
 * coefficients.
 */
int mtxdistfile_init_vector_array_complex_double(
    struct mtxdistfile * mtxdistfile,
    int num_rows,
    const double (* data)[2],
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistfile_init_vector_array_integer_single()’ allocates and
 * initialises a distributed vector in array format with integer,
 * single precision coefficients.
 */
int mtxdistfile_init_vector_array_integer_single(
    struct mtxdistfile * mtxdistfile,
    int num_rows,
    const int32_t * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistfile_init_vector_array_integer_double()’ allocates and
 * initialises a vector in array format with integer, double precision
 * coefficients.
 */
int mtxdistfile_init_vector_array_integer_double(
    struct mtxdistfile * mtxdistfile,
    int num_rows,
    const int64_t * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/*
 * Matrix coordinate formats
 */

/**
 * ‘mtxdistfile_alloc_matrix_coordinate()’ allocates a distributed
 * matrix in coordinate format.
 */
int mtxdistfile_alloc_matrix_coordinate(
    struct mtxdistfile * mtxdistfile,
    enum mtxfilefield field,
    enum mtxfilesymmetry symmetry,
    enum mtxprecision precision,
    int num_rows,
    int num_columns,
    int64_t num_nonzeros,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistfile_init_matrix_coordinate_real_single()’ allocates and
 * initialises a distributed matrix in coordinate format with real,
 * single precision coefficients.
 */
int mtxdistfile_init_matrix_coordinate_real_single(
    struct mtxdistfile * mtxdistfile,
    enum mtxfilesymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_nonzeros,
    const struct mtxfile_matrix_coordinate_real_single * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistfile_init_matrix_coordinate_real_double()’ allocates and
 * initialises a matrix in coordinate format with real, double
 * precision coefficients.
 */
int mtxdistfile_init_matrix_coordinate_real_double(
    struct mtxdistfile * mtxdistfile,
    enum mtxfilesymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_nonzeros,
    const struct mtxfile_matrix_coordinate_real_double * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistfile_init_matrix_coordinate_complex_single()’ allocates and
 * initialises a distributed matrix in coordinate format with complex,
 * single precision coefficients.
 */
int mtxdistfile_init_matrix_coordinate_complex_single(
    struct mtxdistfile * mtxdistfile,
    enum mtxfilesymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_nonzeros,
    const struct mtxfile_matrix_coordinate_complex_single * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistfile_init_matrix_coordinate_complex_double()’ allocates and
 * initialises a matrix in coordinate format with complex, double
 * precision coefficients.
 */
int mtxdistfile_init_matrix_coordinate_complex_double(
    struct mtxdistfile * mtxdistfile,
    enum mtxfilesymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_nonzeros,
    const struct mtxfile_matrix_coordinate_complex_double * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistfile_init_matrix_coordinate_integer_single()’ allocates and
 * initialises a distributed matrix in coordinate format with integer,
 * single precision coefficients.
 */
int mtxdistfile_init_matrix_coordinate_integer_single(
    struct mtxdistfile * mtxdistfile,
    enum mtxfilesymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_nonzeros,
    const struct mtxfile_matrix_coordinate_integer_single * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistfile_init_matrix_coordinate_integer_double()’ allocates and
 * initialises a matrix in coordinate format with integer, double
 * precision coefficients.
 */
int mtxdistfile_init_matrix_coordinate_integer_double(
    struct mtxdistfile * mtxdistfile,
    enum mtxfilesymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_nonzeros,
    const struct mtxfile_matrix_coordinate_integer_double * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistfile_init_matrix_coordinate_pattern()’ allocates and
 * initialises a matrix in coordinate format with boolean (pattern)
 * precision coefficients.
 */
int mtxdistfile_init_matrix_coordinate_pattern(
    struct mtxdistfile * mtxdistfile,
    enum mtxfilesymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_nonzeros,
    const struct mtxfile_matrix_coordinate_pattern * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/*
 * Vector coordinate formats
 */

/**
 * ‘mtxdistfile_alloc_vector_coordinate()’ allocates a distributed
 * vector in coordinate format.
 */
int mtxdistfile_alloc_vector_coordinate(
    struct mtxdistfile * mtxdistfile,
    enum mtxfilefield field,
    enum mtxprecision precision,
    int num_rows,
    int64_t num_nonzeros,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistfile_init_vector_coordinate_real_single()’ allocates and
 * initialises a distributed vector in coordinate format with real,
 * single precision coefficients.
 */
int mtxdistfile_init_vector_coordinate_real_single(
    struct mtxdistfile * mtxdistfile,
    int num_rows,
    int64_t num_nonzeros,
    const struct mtxfile_vector_coordinate_real_single * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistfile_init_vector_coordinate_real_double()’ allocates and
 * initialises a vector in coordinate format with real, double
 * precision coefficients.
 */
int mtxdistfile_init_vector_coordinate_real_double(
    struct mtxdistfile * mtxdistfile,
    int num_rows,
    int64_t num_nonzeros,
    const struct mtxfile_vector_coordinate_real_double * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistfile_init_vector_coordinate_complex_single()’ allocates and
 * initialises a distributed vector in coordinate format with complex,
 * single precision coefficients.
 */
int mtxdistfile_init_vector_coordinate_complex_single(
    struct mtxdistfile * mtxdistfile,
    int num_rows,
    int64_t num_nonzeros,
    const struct mtxfile_vector_coordinate_complex_single * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistfile_init_vector_coordinate_complex_double()’ allocates and
 * initialises a vector in coordinate format with complex, double
 * precision coefficients.
 */
int mtxdistfile_init_vector_coordinate_complex_double(
    struct mtxdistfile * mtxdistfile,
    int num_rows,
    int64_t num_nonzeros,
    const struct mtxfile_vector_coordinate_complex_double * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistfile_init_vector_coordinate_integer_single()’ allocates and
 * initialises a distributed vector in coordinate format with integer,
 * single precision coefficients.
 */
int mtxdistfile_init_vector_coordinate_integer_single(
    struct mtxdistfile * mtxdistfile,
    int num_rows,
    int64_t num_nonzeros,
    const struct mtxfile_vector_coordinate_integer_single * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistfile_init_vector_coordinate_integer_double()’ allocates and
 * initialises a vector in coordinate format with integer, double
 * precision coefficients.
 */
int mtxdistfile_init_vector_coordinate_integer_double(
    struct mtxdistfile * mtxdistfile,
    int num_rows,
    int64_t num_nonzeros,
    const struct mtxfile_vector_coordinate_integer_double * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistfile_init_vector_coordinate_pattern()’ allocates and
 * initialises a vector in coordinate format with boolean (pattern)
 * precision coefficients.
 */
int mtxdistfile_init_vector_coordinate_pattern(
    struct mtxdistfile * mtxdistfile,
    int num_rows,
    int64_t num_nonzeros,
    const struct mtxfile_vector_coordinate_pattern * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/*
 * Modifying values
 */

/**
 * ‘mtxdistfile_set_constant_real_single()’ sets every (nonzero) value
 * of a matrix or vector equal to a constant, single precision
 * floating point number.
 */
int mtxdistfile_set_constant_real_single(
    struct mtxdistfile * mtxdistfile,
    float a,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistfile_set_constant_real_double()’ sets every (nonzero) value
 * of a matrix or vector equal to a constant, double precision
 * floating point number.
 */
int mtxdistfile_set_constant_real_double(
    struct mtxdistfile * mtxdistfile,
    double a,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistfile_set_constant_complex_single()’ sets every (nonzero)
 * value of a matrix or vector equal to a constant, single precision
 * floating point complex number.
 */
int mtxdistfile_set_constant_complex_single(
    struct mtxdistfile * mtxdistfile,
    float a[2],
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistfile_set_constant_integer_single()’ sets every (nonzero)
 * value of a matrix or vector equal to a constant integer.
 */
int mtxdistfile_set_constant_integer_single(
    struct mtxdistfile * mtxdistfile,
    int32_t a,
    struct mtxdisterror * disterr);

/*
 * Convert to and from (non-distributed) Matrix Market format
 */

/**
 * ‘mtxdistfile_from_mtxfile()’ creates a distributed Matrix Market
 * file from a Matrix Market file stored on a single root process by
 * distributing the data of the underlying matrix or vector among
 * processes in a communicator.
 *
 * This function performs collective communication and therefore
 * requires every process in the communicator to perform matching
 * calls to this function.
 */
int mtxdistfile_from_mtxfile(
    struct mtxdistfile * dst,
    const struct mtxfile * src,
    MPI_Comm comm,
    int root,
    struct mtxdisterror * disterr);

/*
 * I/O functions
 */

/**
 * ‘mtxdistfile_read_shared()’ reads a Matrix Market file from the given path
 * and distributes the data among MPI processes in a communicator. The
 * file may optionally be compressed by gzip.
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
 *
 * Only a single root process will read from the specified stream.
 * The data is partitioned into equal-sized parts for each process.
 * For matrices and vectors in coordinate format, the total number of
 * data lines is evenly distributed among processes. Otherwise, the
 * rows are evenly distributed among processes.
 *
 * The file is read one part at a time, which is then sent to the
 * owning process. This avoids reading the entire file into the memory
 * of the root process at once, which would severely limit the size of
 * files that could be read.
 *
 * This function performs collective communication and therefore
 * requires every process in the communicator to perform matching
 * calls to the function.
 */
int mtxdistfile_read_shared(
    struct mtxdistfile * mtxdistfile,
    enum mtxprecision precision,
    const char * path,
    bool gzip,
    int * lines_read,
    int64_t * bytes_read,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistfile_fread_shared()’ reads a Matrix Market file from a stream and
 * distributes the data among MPI processes in a communicator.
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
 *
 * Only a single root process will read from the specified stream.
 * The data is partitioned into equal-sized parts for each process.
 * For matrices and vectors in coordinate format, the total number of
 * data lines is evenly distributed among processes. Otherwise, the
 * rows are evenly distributed among processes.
 *
 * The file is read one part at a time, which is then sent to the
 * owning process. This avoids reading the entire file into the memory
 * of the root process at once, which would severely limit the size of
 * files that could be read.
 *
 * This function performs collective communication and therefore
 * requires every process in the communicator to perform matching
 * calls to the function.
 */
int mtxdistfile_fread_shared(
    struct mtxdistfile * mtxdistfile,
    enum mtxprecision precision,
    FILE * f,
    int * lines_read,
    int64_t * bytes_read,
    size_t line_max,
    char * linebuf,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistfile_fread_shared()’ reads a Matrix Market file from a stream and
 * distributes the data among MPI processes in a communicator.
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
 *
 * Only a single root process will read from the specified stream.
 * The data is partitioned into equal-sized parts for each process.
 * For matrices and vectors in coordinate format, the total number of
 * data lines is evenly distributed among processes. Otherwise, the
 * rows are evenly distributed among processes.
 *
 * The file is read one part at a time, which is then sent to the
 * owning process. This avoids reading the entire file into the memory
 * of the root process at once, which would severely limit the size of
 * files that could be read.
 *
 * This function performs collective communication and therefore
 * requires every process in the communicator to perform matching
 * calls to the function.
 */
int mtxdistfile_fread_shared(
    struct mtxdistfile * mtxdistfile,
    enum mtxprecision precision,
    FILE * f,
    int * lines_read,
    int64_t * bytes_read,
    size_t line_max,
    char * linebuf,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

#ifdef LIBMTX_HAVE_LIBZ
/**
 * ‘mtxdistfile_gzread_shared()’ reads a Matrix Market file from a
 * gzip-compressed stream and distributes the data among MPI processes
 * in a communicator.
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
 *
 * Only a single root process will read from the specified stream.
 * The data is partitioned into equal-sized parts for each process.
 * For matrices and vectors in coordinate format, the total number of
 * data lines is evenly distributed among processes. Otherwise, the
 * rows are evenly distributed among processes.
 *
 * The file is read one part at a time, which is then sent to the
 * owning process. This avoids reading the entire file into the memory
 * of the root process at once, which would severely limit the size of
 * files that could be read.
 *
 * This function performs collective communication and therefore
 * requires every process in the communicator to perform matching
 * calls to the function.
 */
int mtxdistfile_gzread_shared(
    struct mtxdistfile * mtxdistfile,
    enum mtxprecision precision,
    gzFile f,
    int * lines_read,
    int64_t * bytes_read,
    size_t line_max,
    char * linebuf,
    MPI_Comm comm,
    struct mtxdisterror * disterr);
#endif

/**
 * ‘mtxdistfile_write_shared()’ writes a distributed Matrix Market
 * file to a single file that is shared by all processes in the
 * communicator.  The file may optionally be compressed by gzip.
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
 *
 * Note that only the specified ‘root’ process will print anything to
 * the stream. Other processes will therefore send their part of the
 * distributed Matrix Market file to the root process for printing.
 *
 * This function performs collective communication and therefore
 * requires every process in the communicator to perform matching
 * calls to the function.
 */
int mtxdistfile_write_shared(
    const struct mtxdistfile * mtxdistfile,
    const char * path,
    bool gzip,
    const char * fmt,
    int64_t * bytes_written,
    int root,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistfile_write()’ writes a distributed Matrix Market file to
 * the given path.  The file may optionally be compressed by gzip.
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
 *
 * This function performs collective communication and therefore
 * requires every process in the communicator to perform matching
 * calls to the function.
 */
int mtxdistfile_write(
    const struct mtxdistfile * mtxdistfile,
    const char * path,
    bool gzip,
    const char * fmt,
    int64_t * bytes_written,
    bool sequential,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistfile_fwrite()’ writes a distributed Matrix Market file to
 * the specified stream on each process.
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
 *
 * If ‘sequential’ is ‘true’, then output is performed in sequence by
 * MPI processes in the communicator.  This is useful, for example,
 * when writing to a common stream, such as standard output.  In this
 * case, we want to ensure that the processes write their data in the
 * correct order without interfering with each other.
 *
 * This function performs collective communication and therefore
 * requires every process in the communicator to perform matching
 * calls to the function.
 */
int mtxdistfile_fwrite(
    const struct mtxdistfile * mtxdistfile,
    FILE * f,
    const char * fmt,
    int64_t * bytes_written,
    bool sequential,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistfile_fwrite_shared()’ writes a distributed Matrix Market
 * file to a single stream that is shared by every process in the
 * communicator.
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
 *
 * Note that only the specified ‘root’ process will print anything to
 * the stream. Other processes will therefore send their part of the
 * distributed Matrix Market file to the root process for printing.
 *
 * This function performs collective communication and therefore
 * requires every process in the communicator to perform matching
 * calls to the function.
 */
int mtxdistfile_fwrite_shared(
    const struct mtxdistfile * mtxdistfile,
    FILE * f,
    const char * fmt,
    int64_t * bytes_written,
    int root,
    struct mtxdisterror * disterr);

/*
 * Transpose and conjugate transpose.
 */

/**
 * ‘mtxdistfile_transpose()’ tranposes a distributed Matrix Market
 * file.
 */
int mtxdistfile_transpose(
    struct mtxdistfile * mtxdistfile,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistfile_conjugate_transpose()’ tranposes and complex
 * conjugates a distributed Matrix Market file.
 */
int mtxdistfile_conjugate_transpose(
    struct mtxdistfile * mtxdistfile,
    struct mtxdisterror * disterr);

/*
 * Sorting
 */

/**
 * ‘mtxdistfile_sort()’ sorts a distributed Matrix Market file in a
 * given order.
 *
 * The sorting order is determined by ‘sorting’. If the sorting order
 * is ‘mtxfile_unsorted’, nothing is done. If the sorting order is
 * ‘mtxfile_permutation’, then ‘perm’ must point to an array
 * of ‘size’ integers that specify the sorting permutation. Note that
 * the sorting permutation uses 1-based indexing.
 *
 * For a vector or matrix in coordinate format, the nonzero values are
 * sorted in the specified order. For Matrix Market files in array
 * format, this operation does nothing.
 *
 * ‘size’ is the number of vector or matrix nonzeros to sort.
 *
 * ‘perm’ is ignored if it is ‘NULL’. Otherwise, it must point to an
 * array of ‘size’ 64-bit integers, and it is used to store the
 * permutation of the vector or matrix nonzeros.
 */
int mtxdistfile_sort(
    struct mtxdistfile * mtxdistfile,
    enum mtxfilesorting sorting,
    int64_t size,
    int64_t * perm,
    struct mtxdisterror * disterr);

/*
 * Partitioning
 */

/**
 * ‘mtxdistfile_partition()’ partitions and redistributes the entries
 * of a distributed Matrix Market file according to the given row and
 * column partitions.
 *
 * The partitions ‘rowpart’ or ‘colpart’ are allowed to be ‘NULL’, in
 * which case a trivial, singleton partition is used for the rows or
 * columns, respectively.
 *
 * Otherwise, ‘rowpart’ and ‘colpart’ must partition the rows and
 * columns of the matrix or vector ‘src’, respectively. That is,
 * ‘rowpart->size’ must be equal to ‘src->size.num_rows’, and
 * ‘colpart->size’ must be equal to ‘src->size.num_columns’.
 *
 * The distributed Matrix Market file ‘dst’ is used to store the
 * results of the partitioning and redistribution. The user is
 * responsible for calling ‘mtxdistfile_free’ to free storage
 * allocated for ‘dst’.
 */
int mtxdistfile_partition(
    const struct mtxdistfile * src,
    struct mtxdistfile * dsts,
    const struct mtxpartition * rowpart,
    const struct mtxpartition * colpart,
    struct mtxdisterror * disterr);

#if 0
/**
 * ‘mtxdistfile_init_from_partition()’ creates a distributed Matrix
 * Market file from a partitioning of another distributed Matrix
 * Market file.
 *
 * On each process, a partitioning can be obtained from
 * ‘mtxdistfile_partition_rows()’. This provides the arrays
 * ‘data_lines_per_part_ptr’ and ‘data_lines_per_part’, which together
 * describe the size of each part and the indices to its data lines on
 * the current process. The number of parts in the partitioning must
 * be less than or equal to the number of processes in the MPI
 * communicator.
 *
 * The ‘p’th value of ‘data_lines_per_part_ptr’ must be an offset to
 * the first data line belonging to the ‘p’th part of the partition,
 * while the final value of the array points to one place beyond the
 * final data line.  Moreover for each part ‘p’ of the partitioning,
 * the entries from ‘data_lines_per_part[p]’ up to, but not including,
 * ‘data_lines_per_part[p+1]’, are the indices of the data lines in
 * ‘src’ that are assigned to the ‘p’th part of the partitioning.
 */
int mtxdistfile_init_from_partition(
    struct mtxdistfile * dst,
    const struct mtxdistfile * src,
    int num_parts,
    const int64_t * data_lines_per_part_ptr,
    const int64_t * data_lines_per_part,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistfile_partition_rows()’ partitions data lines of a
 * distributed Matrix Market file according to the given row
 * partitioning.
 *
 * If it is not ‘NULL’, the array ‘part_per_data_line’ must contain
 * enough storage to hold one ‘int’ for each data line held by the
 * current process. (The number of data lines is obtained from
 * ‘mtxfilesize_num_data_lines()’). On a successful return, the ‘k’th
 * entry in the array specifies the part number that was assigned to
 * the ‘k’th data line of ‘src’.
 *
 * The array ‘data_lines_per_part_ptr’ must contain at least enough
 * storage for ‘row_partition->num_parts+1’ values of type ‘int64_t’.
 * If successful, the ‘p’th value of ‘data_lines_per_part_ptr’ is an
 * offset to the first data line belonging to the ‘p’th part of the
 * partition, while the final value of the array points to one place
 * beyond the final data line.  Moreover ‘data_lines_per_part’ must
 * contain enough storage to hold one ‘int64_t’ for each data line.
 * For each part ‘p’ of the partitioning, the entries from
 * ‘data_lines_per_part[p]’ up to, but not including,
 * ‘data_lines_per_part[p+1]’, are the indices of the data lines in
 * ‘src’ that are assigned to the ‘p’th part of the partitioning.
 */
int mtxdistfile_partition_rows(
    const struct mtxdistfile * mtxdistfile,
    const struct mtxpartition * row_partition,
    int * part_per_data_line,
    int64_t * data_lines_per_part_ptr,
    int64_t * data_lines_per_part,
    struct mtxdisterror * disterr);
#endif
#endif

#endif
