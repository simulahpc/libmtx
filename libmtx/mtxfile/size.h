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
 * Last modified: 2021-09-01
 *
 * Matrix Market size lines.
 */

#ifndef LIBMTX_MTXFILE_SIZE_H
#define LIBMTX_MTXFILE_SIZE_H

#include <libmtx/libmtx-config.h>

#include <libmtx/mtxfile/header.h>

#ifdef LIBMTX_HAVE_MPI
#include <mpi.h>
#endif

#ifdef LIBMTX_HAVE_LIBZ
#include <zlib.h>
#endif

#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>

struct mtxmpierror;

/**
 * `mtxfile_size' represents a size line of a Matrix Market file.
 */
struct mtxfile_size
{
    /**
     * `num_rows' is the number of rows in the matrix or vector.
     */
    int num_rows;

    /**
     * `num_columns' is the number of columns in the matrix if
     * `object' is `matrix'. Otherwise, if `object' is `vector', then
     * `num_columns' is equal to `-1'.
     */
    int num_columns;

    /**
     * `num_nonzeros' is the number of nonzero matrix or vector
     * entries for a sparse matrix or vector.  This only includes
     * entries that are stored explicitly, and not those that are
     * implicitly, for example, due to symmetry.
     *
     * If `format' is `array', then `num_nonzeros' is set to `-1', and
     * it is not used.
     */
    int64_t num_nonzeros;
};

/**
 * `mtxfile_parse_size()' parses a string containing the size line for
 * a file in Matrix Market format.
 *
 * If `endptr' is not `NULL', then the address stored in `endptr'
 * points to the first character beyond the characters that were
 * consumed during parsing.
 *
 * On success, `mtxfile_parse_size()' returns `MTX_SUCCESS' and the
 * fields of `size' will be set accordingly.  Otherwise, an
 * appropriate error code is returned.
 */
int mtxfile_parse_size(
    struct mtxfile_size * size,
    int64_t * bytes_read,
    const char ** endptr,
    const char * s,
    enum mtxfile_object object,
    enum mtxfile_format format);

/**
 * `mtxfile_size_copy()' copies a size line.
 */
int mtxfile_size_copy(
    struct mtxfile_size * dst,
    const struct mtxfile_size * src);

/**
 * `mtxfile_size_cat()' updates a size line to match with the
 * concatenation of two Matrix Market files.
 */
int mtxfile_size_cat(
    struct mtxfile_size * dst,
    const struct mtxfile_size * src,
    enum mtxfile_object object,
    enum mtxfile_format format);

/**
 * `mtxfile_size_num_data_lines()' computes the number of data lines
 * that are required in a Matrix Market file with the given size line.
 */
int mtxfile_size_num_data_lines(
    const struct mtxfile_size * size,
    int64_t * num_data_lines);

/*
 * I/O functions
 */

/**
 * `mtxfile_fread_size()` reads a Matrix Market size line from a
 * stream.
 *
 * If an error code is returned, then `lines_read' and `bytes_read'
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the Matrix Market file.
 */
int mtxfile_fread_size(
    struct mtxfile_size * size,
    FILE * f,
    int * lines_read,
    int64_t * bytes_read,
    size_t line_max,
    char * linebuf,
    enum mtxfile_object object,
    enum mtxfile_format format);

#ifdef LIBMTX_HAVE_LIBZ
/**
 * `mtxfile_gzread_size()` reads a Matrix Market size line from a
 * gzip-compressed stream.
 *
 * If an error code is returned, then `lines_read' and `bytes_read'
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the Matrix Market file.
 */
int mtxfile_gzread_size(
    struct mtxfile_size * size,
    gzFile f,
    int * lines_read,
    int64_t * bytes_read,
    size_t line_max,
    char * linebuf,
    enum mtxfile_object object,
    enum mtxfile_format format);
#endif

/**
 * `mtxfile_size_fwrite()' writes the size line of a Matrix Market
 * file to a stream.
 *
 * If it is not `NULL', then the number of bytes written to the stream
 * is returned in `bytes_written'.
 */
int mtxfile_size_fwrite(
    const struct mtxfile_size * size,
    enum mtxfile_object object,
    enum mtxfile_format format,
    FILE * f,
    int64_t * bytes_written);

#ifdef LIBMTX_HAVE_LIBZ
/**
 * `mtxfile_size_gzwrite()' writes the size line of a Matrix Market
 * file to a gzip-compressed stream.
 *
 * If it is not `NULL', then the number of bytes written to the stream
 * is returned in `bytes_written'.
 */
int mtxfile_size_gzwrite(
    const struct mtxfile_size * size,
    enum mtxfile_object object,
    enum mtxfile_format format,
    gzFile f,
    int64_t * bytes_written);
#endif

/*
 * Transpose
 */

/**
 * `mtxfile_size_transpose()' tranposes the size line of a Matrix
 * Market file.
 */
int mtxfile_size_transpose(
    struct mtxfile_size * size);

/*
 * MPI functions
 */

#ifdef LIBMTX_HAVE_MPI
/**
 * `mtxfile_size_send()' sends a Matrix Market size line to another
 * MPI process.
 *
 * This is analogous to `MPI_Send()' and requires the receiving
 * process to perform a matching call to `mtxfile_size_recv()'.
 */
int mtxfile_size_send(
    const struct mtxfile_size * size,
    int dest,
    int tag,
    MPI_Comm comm,
    struct mtxmpierror * mpierror);

/**
 * `mtxfile_size_recv()' receives a Matrix Market size line from
 * another MPI process.
 *
 * This is analogous to `MPI_Recv()' and requires the sending process
 * to perform a matching call to `mtxfile_size_send()'.
 */
int mtxfile_size_recv(
    struct mtxfile_size * size,
    int source,
    int tag,
    MPI_Comm comm,
    struct mtxmpierror * mpierror);

/**
 * `mtxfile_size_bcast()' broadcasts a Matrix Market size line from an
 * MPI root process to other processes in a communicator.
 *
 * This is analogous to `MPI_Bcast()' and requires every process in
 * the communicator to perform matching calls to
 * `mtxfile_size_bcast()'.
 */
int mtxfile_size_bcast(
    struct mtxfile_size * size,
    int root,
    MPI_Comm comm,
    struct mtxmpierror * mpierror);

/**
 * `mtxfile_size_scatterv()' scatters a Matrix Market size line from
 * an MPI root process to other processes in a communicator.
 *
 * This is analogous to `MPI_Scatterv()' and requires every process in
 * the communicator to perform matching calls to
 * `mtxfile_size_scatterv()'.
 */
int mtxfile_size_scatterv(
    const struct mtxfile_size * sendsize,
    struct mtxfile_size * recvsize,
    enum mtxfile_object object,
    enum mtxfile_format format,
    int recvcount,
    int root,
    MPI_Comm comm,
    struct mtxmpierror * mpierror);
#endif

#endif
