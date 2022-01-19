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
 * Matrix Market files distributed among multiple processes with MPI
 * for inter-process communication.
 */

#ifndef LIBMTX_MTXDISTFILE_MTXDISTFILE_H
#define LIBMTX_MTXDISTFILE_MTXDISTFILE_H

#include <libmtx/libmtx-config.h>

#include <libmtx/precision.h>
#include <libmtx/mtxfile/comments.h>
#include <libmtx/mtxfile/data.h>
#include <libmtx/mtxfile/header.h>
#include <libmtx/mtxfile/mtxfile.h>
#include <libmtx/mtxfile/size.h>
#include <libmtx/util/partition.h>

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
 *
 * Processes are arranged as a one-dimensional linear array, and data
 * lines of the underlying Matrix Market file are distributed among
 * processes according to a specified partitioning.
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
     * ‘partition’ is a partitioning of the data lines in the
     * underlying Matrix Market file.
     */
    struct mtxpartition partition;

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
     * ‘data’ contains the data lines of the Matrix Market file that
     * are owned by the current process.
     */
    union mtxfiledata data;
};

/*
 * Memory management
 */

/**
 * ‘mtxdistfile_alloc()’ allocates storage for a distributed Matrix
 * Market file with the given header line, comment lines, size line
 * and precision.
 *
 * ‘comments’ may be ‘NULL’, in which case it is ignored.
 *
 * ‘partition’ must be a partitioning of a finite set whose size
 * equals the number of data lines in the underlying Matrix Market
 * file (i.e., ‘size->num_nonzeros’ if ‘header->format’ is
 * ‘mtxfile_coordinate’, or ‘size->num_rows*size->num_columns’ or
 * ‘size->num_rows’ if ‘header->format’ is ‘mtxfile_array’ and
 * ‘header->object’ is ‘mtxfile_matrix’ or ‘mtxfile_vector’,
 * respectively). Also, the number of parts in the partition is at
 * most the number of MPI processes in the communicator ‘comm’.
 *
 * ‘comm’ must be the same MPI communicator that was used to create
 * ‘disterr’.
 */
int mtxdistfile_alloc(
    struct mtxdistfile * mtxdistfile,
    const struct mtxfileheader * header,
    const struct mtxfilecomments * comments,
    const struct mtxfilesize * size,
    enum mtxprecision precision,
    const struct mtxpartition * partition,
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
    const struct mtxpartition * partition,
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
    const struct mtxpartition * partition,
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
    const struct mtxpartition * partition,
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
    const struct mtxpartition * partition,
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
    const struct mtxpartition * partition,
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
    const struct mtxpartition * partition,
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
    const struct mtxpartition * partition,
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
    const struct mtxpartition * partition,
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
    const struct mtxpartition * partition,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistfile_init_vector_array_real_double()’ allocates and
 * initialises a vector in array format with real, double precision
 * coefficients.
 */
int mtxdistfile_init_vector_array_real_double(
    struct mtxdistfile * mtxdistfile,
    int num_rows,
    const double * data,
    const struct mtxpartition * partition,
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
    const struct mtxpartition * partition,
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
    const struct mtxpartition * partition,
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
    const struct mtxpartition * partition,
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
    const struct mtxpartition * partition,
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
    const struct mtxpartition * partition,
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
    const struct mtxpartition * partition,
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
    const struct mtxpartition * partition,
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
    const struct mtxpartition * partition,
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
    const struct mtxpartition * partition,
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
    const struct mtxpartition * partition,
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
    const struct mtxpartition * partition,
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
    const struct mtxpartition * partition,
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
    const struct mtxpartition * partition,
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
    const struct mtxpartition * partition,
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
    const struct mtxpartition * partition,
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
    const struct mtxpartition * partition,
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
    const struct mtxpartition * partition,
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
    const struct mtxpartition * partition,
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
    const struct mtxpartition * partition,
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
    const struct mtxpartition * partition,
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
 * ‘mtxdistfile_set_constant_complex_double()’ sets every (nonzero)
 * value of a matrix or vector equal to a constant, double precision
 * floating point complex number.
 */
int mtxdistfile_set_constant_complex_double(
    struct mtxdistfile * mtxdistfile,
    double a[2],
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistfile_set_constant_integer_single()’ sets every (nonzero)
 * value of a matrix or vector equal to a constant, 32-bit integer.
 */
int mtxdistfile_set_constant_integer_single(
    struct mtxdistfile * mtxdistfile,
    int32_t a,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistfile_set_constant_integer_double()’ sets every (nonzero)
 * value of a matrix or vector equal to a constant, 64-bit integer.
 */
int mtxdistfile_set_constant_integer_double(
    struct mtxdistfile * mtxdistfile,
    int64_t a,
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
    const struct mtxpartition * datapart,
    MPI_Comm comm,
    int root,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistfile_to_mtxfile()’ gathers a distributed Matrix Market file
 * onto a single, root process and creates a non-distributed Matrix
 * Market file on that process.
 *
 * This function performs collective communication and therefore
 * requires every process in the communicator to perform matching
 * calls to this function.
 */
int mtxdistfile_to_mtxfile(
    struct mtxfile * dst,
    const struct mtxdistfile * src,
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
    int root,
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
    int root,
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
    int root,
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
    int root,
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
    int root,
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
 * The argument ‘dsts’ is an array that must have enough storage for
 * ‘P*Q’ values of type ‘struct mtxdistfile’, where ‘P’ is the number
 * of row parts, ‘rowpart->num_parts’, and ‘Q’ is the number of column
 * parts, ‘colpart->num_parts’. Note that the ‘r’th part corresponds
 * to a row part ‘p’ and column part ‘q’, such that ‘r=p*Q+q’. Thus,
 * the ‘r’th entry of ‘dsts’ is the submatrix corresponding to the
 * ‘p’th row and ‘q’th column of the 2D partitioning.
 *
 * The user is responsible for freeing storage allocated for each
 * Matrix Market file in the ‘dsts’ array.
 */
int mtxdistfile_partition(
    struct mtxdistfile * dsts,
    const struct mtxdistfile * src,
    const struct mtxpartition * rowpart,
    const struct mtxpartition * colpart,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistfile_join()’ joins together distributed Matrix Market files
 * representing compatible blocks of a partitioned matrix or vector to
 * form a larger matrix or vector.
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
 * of ‘srcs[p*Q+q]->size.num_rows’ for ‘p=0,1,...,P-1’ must be equal
 * to ‘rowpart->size’. Likewise, for each block row ‘p’, the sum of
 * ‘srcs[p*Q+q]->size.num_rows’ for ‘q=0,1,...,Q-1’ must be equal to
 * ‘colpart->size’.
 */
int mtxdistfile_join(
    struct mtxdistfile * dst,
    const struct mtxdistfile * srcs,
    const struct mtxpartition * rowpart,
    const struct mtxpartition * colpart,
    struct mtxdisterror * disterr);
#endif

#endif
