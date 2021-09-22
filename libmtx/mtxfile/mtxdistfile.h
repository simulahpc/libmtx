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
 * Last modified: 2021-09-22
 *
 * Matrix Market files distributed among multiple processes with MPI
 * for inter-process communication.
 */

#ifndef LIBMTX_MTXFILE_MTXDISTFILE_H
#define LIBMTX_MTXFILE_MTXDISTFILE_H

#include <libmtx/libmtx-config.h>

#include <libmtx/mtx/precision.h>
#include <libmtx/mtxfile/comments.h>
#include <libmtx/mtxfile/data.h>
#include <libmtx/mtxfile/header.h>
#include <libmtx/mtxfile/mtxfile.h>
#include <libmtx/mtxfile/size.h>
#include <libmtx/util/distpartition.h>
#include <libmtx/util/partition.h>

#ifdef LIBMTX_HAVE_MPI
#include <mpi.h>
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

struct mtxmpierror;
struct mtx_partition;

#ifdef LIBMTX_HAVE_MPI
/**
 * `mtxdistfile' represents a file in the Matrix Market file format
 * distributed among multiple processes, where MPI is used for
 * communicating between processes.
 */
struct mtxdistfile
{
    /**
     * `comm' is an MPI communicator for processes among which the
     * Matrix Market file is distributed.
     */
    MPI_Comm comm;

    /**
     * `comm_size' is the size of the MPI communicator.  This is equal
     * to the number of parts of the row partitioning of the matrix or
     * vector.
     */
    int comm_size;

    /**
     * `rank' is the rank of the current process.
     */
    int rank;

    /**
     * `row_partition' is a partitioning of the rows of the underlying
     * matrix or vector.  The partitioning information is also
     * distributed among processes, so the current process stores the
     * part of the partitioning that it owns.
     */
    struct mtxdistpartition row_partition;

    /**
     * `header' is the Matrix Market file header.
     */
    struct mtxfile_header header;

    /**
     * `comments' is the Matrix Market comment lines.
     */
    struct mtxfile_comments comments;

    /**
     * `size' is the Matrix Market size line.
     */
    struct mtxfile_size size;

    /**
     * `precision' is the precision used to store the values of the
     * Matrix Market data lines.
     */
    enum mtx_precision precision;

    /**
     * `mtxfile' is the part of the Matrix Market file owned by the
     * current process.
     */
    struct mtxfile mtxfile;
};

/*
 * Memory management
 */

/**
 * `mtxdistfile_free()' frees storage allocated for a distributed
 * Matrix Market file.
 */
void mtxdistfile_free(
    struct mtxdistfile * mtxdistfile);

/**
 * `mtxdistfile_copy()' copies a distributed Matrix Market file.
 */
int mtxdistfile_copy(
    struct mtxdistfile * dst,
    const struct mtxdistfile * src);

/*
 * Convert to and from (non-distributed) Matrix Market format
 */

/**
 * `mtxdistfile_from_mtxfile()' partitions and distributes rows of a
 * distributed Matrix Market file from an MPI root process to other
 * processes in a communicator.
 *
 * This function performs collective communication and therefore
 * requires every process in the communicator to perform matching
 * calls to `mtxdistfile_from_mtxfile()'.
 *
 * `row_partition' must be a partitioning of the rows of the matrix or
 * vector represented by `src'.
 */
int mtxdistfile_distribute_rows(
    struct mtxdistfile * dst,
    struct mtxfile * src,
    const struct mtx_partition * row_partition,
    int root,
    MPI_Comm comm,
    struct mtxmpierror * mpierror);

/*
 * I/O functions
 */

/**
 * `mtxdistfile_read()' reads a Matrix Market file from the given path
 * and distributes the rows of the underlying matrix or vector among
 * MPI processes in a communicator.
 *
 * The `precision' argument specifies which precision to use for
 * storing matrix or vector values.
 *
 * If `path' is `-', then standard input is used.
 *
 * If an error code is returned, then `lines_read' and `bytes_read'
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the Matrix Market file.
 *
 * Only the designated root process will read from the specified
 * stream.  The file is read in a buffered manner to avoid reading the
 * entire file into the memory of the root process, which would
 * severely limit the size of files that could be read.  That is, a
 * buffer of at most `bufsize' bytes is allocated, and data is read
 * into the buffer until it is full.  The data is then distributed
 * among processes before the buffer is cleared and filled with data
 * from the stream once more.  This continues until an error occurs or
 * until end-of-file.
 *
 * Note that for a matrix (or vector) in array format, `bufsize' must
 * be at least large enough to fit one row of matrix (or vector) data
 * per MPI process in the communicator.
 *
 * This function performs collective communication and therefore
 * requires every process in the communicator to perform matching
 * calls to the function.
 */
int mtxdistfile_read(
    struct mtxdistfile * mtxdistfile,
    enum mtx_precision precision,
    enum mtx_partition_type row_partition_type,
    const char * path,
    int * lines_read,
    int64_t * bytes_read,
    size_t line_max,
    char * linebuf,
    size_t bufsize,
    int root,
    MPI_Comm comm,
    struct mtxmpierror * mpierror);

/**
 * `mtxdistfile_fread()' reads a Matrix Market file from a stream and
 * distributes the rows of the underlying matrix or vector among MPI
 * processes in a communicator.
 *
 * `precision' is used to determine the precision to use for storing
 * the values of matrix or vector entries.
 *
 * If an error code is returned, then `lines_read' and `bytes_read'
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the Matrix Market file.
 *
 * Only the designated root process will read from the specified
 * stream.  The file is read in a buffered manner to avoid reading the
 * entire file into the memory of the root process, which would
 * severely limit the size of files that could be read.  That is, a
 * buffer of at most `bufsize' bytes is allocated, and data is read
 * into the buffer until it is full.  The data is then distributed
 * among processes before the buffer is cleared and filled with data
 * from the stream once more.  This continues until an error occurs or
 * until end-of-file.
 *
 * Note that for a matrix (or vector) in array format, `bufsize' must
 * be at least large enough to fit one row of matrix (or vector) data
 * per MPI process in the communicator.
 *
 * This function performs collective communication and therefore
 * requires every process in the communicator to perform matching
 * calls to the function.
 */
int mtxdistfile_fread(
    struct mtxdistfile * mtxdistfile,
    enum mtx_precision precision,
    enum mtx_partition_type row_partition_type,
    FILE * f,
    int * lines_read,
    int64_t * bytes_read,
    size_t line_max,
    char * linebuf,
    size_t bufsize,
    int root,
    MPI_Comm comm,
    struct mtxmpierror * mpierror);

/**
 * `mtxdistfile_write()' writes a distributed Matrix Market file to
 * the given path.  The file may optionally be compressed by gzip.
 *
 * If `path' is `-', then standard output is used.
 *
 * If `format' is `NULL', then the format specifier '%d' is used to
 * print integers and '%f' is used to print floating point
 * numbers. Otherwise, the given format string is used when printing
 * numerical values.
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
 * This function performs collective communication and therefore
 * requires every process in the communicator to perform matching
 * calls to the function.
 */
int mtxdistfile_write(
    const struct mtxdistfile * mtxdistfile,
    const char * path,
    bool gzip,
    const char * format,
    int64_t * bytes_written);

/**
 * `mtxdistfile_fwrite()' writes a distributed Matrix Market file to
 * the specified stream on each process.
 *
 * If `format' is `NULL', then the format specifier '%d' is used to
 * print integers and '%f' is used to print floating point
 * numbers. Otherwise, the given format string is used when printing
 * numerical values.
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
 * If `sequential' is true, then output is performed in sequence by
 * MPI processes in the communicator.  This is useful, for example,
 * when writing to standard output.  In this case, we want to ensure
 * that the processes write their data in the correct order without
 * interfering with each other.
 *
 * This function performs collective communication and therefore
 * requires every process in the communicator to perform matching
 * calls to the function.
 */
int mtxdistfile_fwrite(
    const struct mtxdistfile * mtxdistfile,
    FILE * f,
    const char * format,
    int64_t * bytes_written,
    bool sequential);
#endif

#endif
