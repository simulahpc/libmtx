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
 * Last modified: 2022-05-01
 *
 * Matrix Market files distributed among multiple processes with MPI
 * for inter-process communication.
 */

#ifndef LIBMTX_MTXFILE_MTXDISTFILE_H
#define LIBMTX_MTXFILE_MTXDISTFILE_H

#include <libmtx/libmtx-config.h>

#include <libmtx/vector/precision.h>
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
     * ‘datasize’ is the total number of explicitly stored data lines
     * in the entire distributed Matrix Market file.
     */
    int64_t datasize;

    /**
     * ‘localdatasize’ is the number of explicitly stored data lines
     * of the distributed Matrix Market file that are stored on the
     * current process.
     */
    int64_t localdatasize;

    /**
     * ‘idx’ is an array of length ‘localdatasize’, containing the
     * global offset to each entry of the Matrix Market file stored on
     * the current process. Note that offsets are 0-based, unlike the
     * Matrix Market format, where indices are 1-based.
     */
    int64_t * idx;

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
 * ‘localdatasize’ is the number of entries in the underlying Matrix
 * Market file that are stored on the current process.
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
    int64_t localdatasize,
    const int64_t * idx,
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
    int64_t num_rows,
    int64_t num_columns,
    int64_t localdatasize,
    const int64_t * idx,
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
    int64_t num_rows,
    int64_t num_columns,
    int64_t localdatasize,
    const int64_t * idx,
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
    int64_t num_rows,
    int64_t num_columns,
    int64_t localdatasize,
    const int64_t * idx,
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
    int64_t num_rows,
    int64_t num_columns,
    int64_t localdatasize,
    const int64_t * idx,
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
    int64_t num_rows,
    int64_t num_columns,
    int64_t localdatasize,
    const int64_t * idx,
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
    int64_t num_rows,
    int64_t num_columns,
    int64_t localdatasize,
    const int64_t * idx,
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
    int64_t num_rows,
    int64_t num_columns,
    int64_t localdatasize,
    const int64_t * idx,
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
    int64_t num_rows,
    int64_t localdatasize,
    const int64_t * idx,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistfile_init_vector_array_real_single()’ allocates and
 * initialises a distributed vector in array format with real, single
 * precision coefficients.
 */
int mtxdistfile_init_vector_array_real_single(
    struct mtxdistfile * mtxdistfile,
    int64_t num_rows,
    int64_t localdatasize,
    const int64_t * idx,
    const float * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistfile_init_vector_array_real_double()’ allocates and
 * initialises a vector in array format with real, double precision
 * coefficients.
 */
int mtxdistfile_init_vector_array_real_double(
    struct mtxdistfile * mtxdistfile,
    int64_t num_rows,
    int64_t localdatasize,
    const int64_t * idx,
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
    int64_t num_rows,
    int64_t localdatasize,
    const int64_t * idx,
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
    int64_t num_rows,
    int64_t localdatasize,
    const int64_t * idx,
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
    int64_t num_rows,
    int64_t localdatasize,
    const int64_t * idx,
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
    int64_t num_rows,
    int64_t localdatasize,
    const int64_t * idx,
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
    int64_t num_rows,
    int64_t num_columns,
    int64_t num_nonzeros,
    int64_t localdatasize,
    const int64_t * idx,
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
    int64_t num_rows,
    int64_t num_columns,
    int64_t num_nonzeros,
    int64_t localdatasize,
    const int64_t * idx,
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
    int64_t num_rows,
    int64_t num_columns,
    int64_t num_nonzeros,
    int64_t localdatasize,
    const int64_t * idx,
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
    int64_t num_rows,
    int64_t num_columns,
    int64_t num_nonzeros,
    int64_t localdatasize,
    const int64_t * idx,
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
    int64_t num_rows,
    int64_t num_columns,
    int64_t num_nonzeros,
    int64_t localdatasize,
    const int64_t * idx,
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
    int64_t num_rows,
    int64_t num_columns,
    int64_t num_nonzeros,
    int64_t localdatasize,
    const int64_t * idx,
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
    int64_t num_rows,
    int64_t num_columns,
    int64_t num_nonzeros,
    int64_t localdatasize,
    const int64_t * idx,
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
    int64_t num_rows,
    int64_t num_columns,
    int64_t num_nonzeros,
    int64_t localdatasize,
    const int64_t * idx,
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
    int64_t num_rows,
    int64_t num_nonzeros,
    int64_t localdatasize,
    const int64_t * idx,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistfile_init_vector_coordinate_real_single()’ allocates and
 * initialises a distributed vector in coordinate format with real,
 * single precision coefficients.
 */
int mtxdistfile_init_vector_coordinate_real_single(
    struct mtxdistfile * mtxdistfile,
    int64_t num_rows,
    int64_t num_nonzeros,
    int64_t localdatasize,
    const int64_t * idx,
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
    int64_t num_rows,
    int64_t num_nonzeros,
    int64_t localdatasize,
    const int64_t * idx,
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
    int64_t num_rows,
    int64_t num_nonzeros,
    int64_t localdatasize,
    const int64_t * idx,
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
    int64_t num_rows,
    int64_t num_nonzeros,
    int64_t localdatasize,
    const int64_t * idx,
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
    int64_t num_rows,
    int64_t num_nonzeros,
    int64_t localdatasize,
    const int64_t * idx,
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
    int64_t num_rows,
    int64_t num_nonzeros,
    int64_t localdatasize,
    const int64_t * idx,
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
    int64_t num_rows,
    int64_t num_nonzeros,
    int64_t localdatasize,
    const int64_t * idx,
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
 * ‘mtxdistfile_from_mtxfile_rowwise()’ creates a distributed Matrix
 * Market file from a Matrix Market file stored on a single root
 * process by partitioning the underlying matrix or vector rowwise and
 * distributing the parts among processes.
 *
 * This function performs collective communication and therefore
 * requires every process in the communicator to perform matching
 * calls to this function.
 */
int mtxdistfile_from_mtxfile_rowwise(
    struct mtxdistfile * dst,
    struct mtxfile * src,
    enum mtxpartitioning parttype,
    int64_t partsize,
    int64_t blksize,
    const int * parts,
    MPI_Comm comm,
    int root,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistfile_to_mtxfile()’ creates a Matrix Market file on a given
 * root process from a distributed Matrix Market file.
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
 * ‘mtxdistfile_read_rowwise()’ reads a Matrix Market file from the
 * given path and distributes the data among MPI processes in a
 * communicator. The file may optionally be compressed by gzip.
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
 * Only a single root process reads from the specified stream. The
 * underlying matrix or vector is partitioned rowwise and distributed
 * among processes.
 *
 * This function performs collective communication and therefore
 * requires every process in the communicator to perform matching
 * calls to the function.
 */
int mtxdistfile_read_rowwise(
    struct mtxdistfile * mtxdistfile,
    enum mtxprecision precision,
    enum mtxpartitioning parttype,
    int64_t partsize,
    int64_t blksize,
    const int * parts,
    const char * path,
    bool gzip,
    int64_t * lines_read,
    int64_t * bytes_read,
    MPI_Comm comm,
    int root,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistfile_fread_rowwise()’ reads a Matrix Market file from a
 * stream and distributes the data among MPI processes in a
 * communicator.
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
 * Only a single root process reads from the specified stream. The
 * underlying matrix or vector is partitioned rowwise and distributed
 * among processes.
 *
 * This function performs collective communication and therefore
 * requires every process in the communicator to perform matching
 * calls to the function.
 */
int mtxdistfile_fread_rowwise(
    struct mtxdistfile * mtxdistfile,
    enum mtxprecision precision,
    enum mtxpartitioning parttype,
    int64_t partsize,
    int64_t blksize,
    const int * parts,
    FILE * f,
    int64_t * lines_read,
    int64_t * bytes_read,
    size_t line_max,
    char * linebuf,
    MPI_Comm comm,
    int root,
    struct mtxdisterror * disterr);

#ifdef LIBMTX_HAVE_LIBZ
/**
 * ‘mtxdistfile_gzread_rowwise()’ reads a Matrix Market file from a
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
 * Only a single root process reads from the specified stream. The
 * underlying matrix or vector is partitioned rowwise and distributed
 * among processes.
 *
 * This function performs collective communication and therefore
 * requires every process in the communicator to perform matching
 * calls to the function.
 */
int mtxdistfile_gzread_rowwise(
    struct mtxdistfile * mtxdistfile,
    enum mtxprecision precision,
    enum mtxpartitioning parttype,
    int64_t partsize,
    int64_t blksize,
    const int * parts,
    gzFile f,
    int64_t * lines_read,
    int64_t * bytes_read,
    size_t line_max,
    char * linebuf,
    MPI_Comm comm,
    int root,
    struct mtxdisterror * disterr);
#endif

/**
 * ‘mtxdistfile_fwrite()’ writes a distributed Matrix Market
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
int mtxdistfile_fwrite(
    const struct mtxdistfile * mtxdistfile,
    FILE * f,
    const char * fmt,
    int64_t * bytes_written,
    int root,
    struct mtxdisterror * disterr);
#endif

#endif
