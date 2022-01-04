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
 * Last modified: 2021-09-18
 *
 * Data structures for distributed vectors.
 */

#ifndef LIBMTX_VECTOR_DISTRIBUTED_H
#define LIBMTX_VECTOR_DISTRIBUTED_H

#include <libmtx/libmtx-config.h>

#ifdef LIBMTX_HAVE_MPI
#include <libmtx/mtx/precision.h>
#include <libmtx/util/field.h>
#include <libmtx/util/partition.h>

#include <mpi.h>

#ifdef LIBMTX_HAVE_LIBZ
#include <zlib.h>
#endif

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

struct mtxmpierror;
struct mtx_partition;

/**
 * `mtxvector_distributed' represents a vector in distributed format.
 */
struct mtxvector_distributed
{
    MPI_Comm comm;
    struct mtx_partition partition;
    struct mtxvector_array interior;
    struct mtxvector_array interior_halo;
    struct mtxvector_coordinate exterior_halo;
};

/*
 * Memory management
 */

/**
 * `mtxvector_distributed_free()' frees storage allocated for a
 * vector.
 */
void mtxvector_distributed_free(
    struct mtxvector_distributed * vector);

/**
 * `mtxvector_distributed_copy()' copies a vector.
 */
int mtxvector_distributed_copy(
    struct mtxvector_distributed * dst,
    const struct mtxvector_distributed * src);

/*
 * Vector distributed formats
 */

/**
 * `mtxvector_distributed_alloc()' allocates a vector in distributed
 * format.
 */
int mtxvector_distributed_alloc(
    struct mtxvector_distributed * vector,
    enum mtx_field_ field,
    enum mtxprecision precision,
    int size,
    int64_t num_nonzeros);

/**
 * `mtxvector_distributed_init_real_single()' allocates and
 * initialises a vector in distributed format with real, single
 * precision coefficients.
 */
int mtxvector_distributed_init_real_single(
    struct mtxvector_distributed * vector,
    int size,
    int64_t num_nonzeros,
    const float * data);

/**
 * `mtxvector_distributed_init_real_double()' allocates and
 * initialises a vector in distributed format with real, double
 * precision coefficients.
 */
int mtxvector_distributed_init_real_double(
    struct mtxvector_distributed * vector,
    int size,
    int64_t num_nonzeros,
    const double * data);

/**
 * `mtxvector_distributed_init_complex_single()' allocates and
 * initialises a vector in distributed format with complex, single
 * precision coefficients.
 */
int mtxvector_distributed_init_complex_single(
    struct mtxvector_distributed * vector,
    int size,
    int64_t num_nonzeros,
    const float (* data)[2]);

/**
 * `mtxvector_distributed_init_complex_double()' allocates and
 * initialises a vector in distributed format with complex, double
 * precision coefficients.
 */
int mtxvector_distributed_init_complex_double(
    struct mtxvector_distributed * vector,
    int size,
    int64_t num_nonzeros,
    const double (* data)[2]);

/**
 * `mtxvector_distributed_init_integer_single()' allocates and
 * initialises a vector in distributed format with integer, single
 * precision coefficients.
 */
int mtxvector_distributed_init_integer_single(
    struct mtxvector_distributed * vector,
    int size,
    int64_t num_nonzeros,
    const int32_t * data);

/**
 * `mtxvector_distributed_init_integer_double()' allocates and
 * initialises a vector in distributed format with integer, double
 * precision coefficients.
 */
int mtxvector_distributed_init_integer_double(
    struct mtxvector_distributed * vector,
    int size,
    int64_t num_nonzeros,
    const int64_t * data);

/*
 * Convert to and from Matrix Market format
 */

/**
 * `mtxvector_distributed_from_mtxfile()' converts a vector in Matrix
 * Market format to a vector.
 */
int mtxvector_distributed_from_mtxfile(
    struct mtxvector_distributed * vector,
    const struct mtxfile * mtxfile);

/**
 * `mtxvector_distributed_to_mtxfile()' converts a vector to a vector
 * in Matrix Market format.
 */
int mtxvector_distributed_to_mtxfile(
    const struct mtxvector_distributed * vector,
    struct mtxfile * mtxfile);

/*
 * I/O functions
 */

/**
 * `mtxvector_distributed_read()' reads a vector from a Matrix Market
 * file.  The file may optionally be compressed by gzip.
 *
 * The `precision' argument specifies which precision to use for
 * storing matrix or vector values.
 *
 * If `path' is `-', then standard input is used.
 *
 * If an error code is returned, then `lines_read' and `bytes_read'
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the vector.
 */
int mtxvector_distributed_read(
    struct mtxvector_distributed * vector,
    enum mtxprecision precision,
    const char * path,
    bool gzip,
    int * lines_read,
    int64_t * bytes_read);

/**
 * `mtxvector_distributed_fread()' reads a vector from a stream in
 * Matrix Market format.
 *
 * `precision' is used to determine the precision to use for storing
 * the values of matrix or vector entries.
 *
 * If an error code is returned, then `lines_read' and `bytes_read'
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the vector.
 */
int mtxvector_distributed_fread(
    struct mtxvector_distributed * vector,
    enum mtxprecision precision,
    FILE * f,
    int * lines_read,
    int64_t * bytes_read,
    size_t line_max,
    char * linebuf);

#ifdef LIBMTX_HAVE_LIBZ
/**
 * `mtxvector_distributed_gzread()' reads a vector from a
 * gzip-compressed stream.
 *
 * `precision' is used to determine the precision to use for storing
 * the values of matrix or vector entries.
 *
 * If an error code is returned, then `lines_read' and `bytes_read'
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the vector.
 */
int mtxvector_distributed_gzread(
    struct mtxvector_distributed * vector,
    enum mtxprecision precision,
    gzFile f,
    int * lines_read,
    int64_t * bytes_read,
    size_t line_max,
    char * linebuf);
#endif

/**
 * `mtxvector_distributed_write()' writes a vector to a Matrix Market
 * file. The file may optionally be compressed by gzip.
 *
 * If `path' is `-', then standard output is used.
 *
 * If `format' is `NULL', then the format specifier '%d' is used to
 * print integers and '%g' is used to print floating point
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
 */
int mtxvector_distributed_write(
    const struct mtxvector_distributed * vector,
    const char * path,
    bool gzip,
    const char * format,
    int64_t * bytes_written);

/**
 * `mtxvector_distributed_fwrite()' writes a vector to a stream.
 *
 * If `format' is `NULL', then the format specifier '%d' is used to
 * print integers and '%g' is used to print floating point
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
 */
int mtxvector_distributed_fwrite(
    const struct mtxvector_distributed * vector,
    FILE * f,
    const char * format,
    int64_t * bytes_written);

#ifdef LIBMTX_HAVE_LIBZ
/**
 * `mtxvector_distributed_gzwrite()' writes a vector to a
 * gzip-compressed stream.
 *
 * If `format' is `NULL', then the format specifier '%d' is used to
 * print integers and '%g' is used to print floating point
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
 */
int mtxvector_distributed_gzwrite(
    const struct mtxvector_distributed * vector,
    gzFile f,
    const char * format,
    int64_t * bytes_written);
#endif

/*
 * Partitioning
 */

/**
 * `mtxvector_distributed_partition_rows()' partitions and reorders
 * data lines of a vector according to the given row partitioning.
 *
 * The array `data_lines_per_part_ptr' must contain at least enough
 * storage for `row_partition->num_parts+1' values of type `int64_t'.
 * If successful, the `p'-th value of `data_lines_per_part_ptr' is an
 * offset to the first data line belonging to the `p'-th part of the
 * partition, while the final value of the array points to one place
 * beyond the final data line.
 *
 * If it is not `NULL', the array `row_parts' must contain enough
 * storage to hold one `int' for each data line. (The number of data
 * lines is obtained by calling `mtxvector_size_num_data_lines()'). On
 * a successful return, the `k'-th entry in the array specifies the
 * part number that was assigned to the `k'-th data line.
 */
int mtxvector_distributed_partition_rows(
    struct mtxvector_distributed * vector,
    const struct mtx_partition * row_partition,
    int64_t * data_lines_per_part_ptr,
    int * row_parts);

/**
 * `mtxvector_distributed_init_from_row_partition()' creates a vector
 * from a subset of the rows of another vector.
 *
 * The array `data_lines_per_part_ptr' should have been obtained
 * previously by calling `mtxvector_distributed_partition_rows'.
 */
int mtxvector_distributed_init_from_row_partition(
    struct mtxvector_distributed * dst,
    const struct mtxvector_distributed * src,
    const struct mtx_partition * row_partition,
    int64_t * data_lines_per_part_ptr,
    int part);

/*
 * MPI functions
 */

/**
 * `mtxvector_distributed_send()' sends a vector to another MPI
 * process.
 *
 * This is analogous to `MPI_Send()' and requires the receiving
 * process to perform a matching call to
 * `mtxvector_distributed_recv()'.
 */
int mtxvector_distributed_send(
    const struct mtxvector_distributed * vector,
    int dest,
    int tag,
    MPI_Comm comm,
    struct mtxmpierror * mpierror);

/**
 * `mtxvector_distributed_recv()' receives a vector from another MPI
 * process.
 *
 * This is analogous to `MPI_Recv()' and requires the sending process
 * to perform a matching call to `mtxvector_distributed_send()'.
 */
int mtxvector_distributed_recv(
    struct mtxvector_distributed * vector,
    int source,
    int tag,
    MPI_Comm comm,
    struct mtxmpierror * mpierror);

/**
 * `mtxvector_distributed_bcast()' broadcasts a vector from an MPI
 * root process to other processes in a communicator.
 *
 * This is analogous to `MPI_Bcast()' and requires every process in
 * the communicator to perform matching calls to
 * `mtxvector_distributed_bcast()'.
 */
int mtxvector_distributed_bcast(
    struct mtxvector_distributed * vector,
    int root,
    MPI_Comm comm,
    struct mtxmpierror * mpierror);

/**
 * `mtxvector_distributed_scatterv()' scatters a vector from an MPI
 * root process to other processes in a communicator.
 *
 * This is analogous to `MPI_Scatterv()' and requires every process in
 * the communicator to perform matching calls to
 * `mtxvector_distributed_scatterv()'.
 *
 * For a matrix in `array' format, entire rows are scattered, which
 * means that the send and receive counts must be multiples of the
 * number of matrix columns.
 */
int mtxvector_distributed_scatterv(
    const struct mtxvector_distributed * sendmtxvector,
    int * sendcounts,
    int * displs,
    struct mtxvector_distributed * recvmtxvector,
    int recvcount,
    int root,
    MPI_Comm comm,
    struct mtxmpierror * mpierror);

/**
 * `mtxvector_distributed_distribute_rows()' partitions and
 * distributes rows of a vector from an MPI root process to other
 * processes in a communicator.
 *
 * This function performs collective communication and therefore
 * requires every process in the communicator to perform matching
 * calls to `mtxvector_distributed_distribute_rows()'.
 *
 * `row_partition' must be a partitioning of the rows of the matrix or
 * vector represented by `src'.
 */
int mtxvector_distributed_distribute_rows(
    struct mtxvector_distributed * dst,
    struct mtxvector_distributed * src,
    const struct mtx_partition * row_partition,
    int root,
    MPI_Comm comm,
    struct mtxmpierror * mpierror);

/**
 * `mtxvector_distributed_fread_distribute_rows()' reads a vector from
 * a stream and distributes the rows of the underlying matrix or
 * vector among MPI processes in a communicator.
 *
 * `precision' is used to determine the precision to use for storing
 * the values of matrix or vector entries.
 *
 * If an error code is returned, then `lines_read' and `bytes_read'
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the vector.
 *
 * For a matrix or vector in array format, `bufsize' must be at least
 * large enough to fit one row per MPI process in the communicator.
 */
int mtxvector_distributed_fread_distribute_rows(
    struct mtxvector_distributed * vector,
    FILE * f,
    int * lines_read,
    int64_t * bytes_read,
    size_t line_max,
    char * linebuf,
    enum mtxprecision precision,
    enum mtx_partition_type row_partition_type,
    size_t bufsize,
    int root,
    MPI_Comm comm,
    struct mtxmpierror * mpierror);
#endif

#endif
