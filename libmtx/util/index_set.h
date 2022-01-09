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
 * Last modified: 2022-01-09
 *
 * Index sets.
 */

#ifndef LIBMTX_INDEX_SET_H
#define LIBMTX_INDEX_SET_H

#include <libmtx/libmtx-config.h>

#ifdef LIBMTX_HAVE_MPI
#include <mpi.h>
#endif

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

struct mtxdisterror;

/**
 * ‘mtxidxsettype’ enumerates different kinds of index sets.
 */
enum mtxidxsettype
{
    mtxidxset_interval,      /* contiguous interval of integers */
    mtxidxset_strided,       /* integers separated by a stride */
    mtxidxset_blockstrided,  /* fixed-size sets of integers
                              * separated by a stride */
    mtxidxset_array,         /* index set given by an array */
};

/**
 * ‘mtxidxsettype_str()’ is a string representing the index set type.
 */
const char * mtxidxsettype_str(
    enum mtxidxsettype index_set_type);

/**
 * ‘mtxidxset’ is a data structure for index sets.
 */
struct mtxidxset
{
    /**
     * ‘type’ is the type of index set.
     */
    enum mtxidxsettype type;

    /**
     * ‘size’ is the number of elements in the index set.
     */
    int64_t size;

    /**
     * ‘offset’ is an offset to the first element of the indexed set,
     * if ‘type’ is ‘mtxidxset_interval’, ‘mtxidxset_strided’ or
     * ‘mtxidxset_blockstrided’.  Otherwise, this value is not used.
     */
    int64_t offset;

    /**
     * ‘stride’ is a stride between elements of the indexed set, if
     * ‘type’ is ‘mtxidxset_strided’ or ‘mtxidxset_blockstrided’.
     * Otherwise, this value is not used.
     */
    int stride;

    /**
     * ‘block_size’ is the size of each block, if ‘type’ is
     * ‘mtxidxset_blockstrided’.  Otherwise, this value is not used.
     */
    int block_size;

    /**
     * ‘indices’ is an array containing the indices of the index set,
     * if ‘type’ is ‘mtxidxset_array’.  Otherwise, this value is not
     * used.
     */
    int64_t * indices;
};

/**
 * ‘mtxidxset_free()’ frees resources associated with an index set.
 */
void mtxidxset_free(
    struct mtxidxset * index_set);

/**
 * ‘mtxidxset_init_interval()’ creates an index set of contiguous
 * integers from an interval [a,b).
 */
int mtxidxset_init_interval(
    struct mtxidxset * index_set,
    int64_t a,
    int64_t b);

/**
 * ‘mtxidxset_init_strided()’ creates an index set of strided integers
 * from an interval:
 *
 *   ‘offset,offset+stride,offset+2*stride,...,offset+(size-1)*stride’.
 */
int mtxidxset_init_strided(
    struct mtxidxset * index_set,
    int64_t offset,
    int64_t size,
    int stride);

/**
 * ‘mtxidxset_init_blockstrided()’ creates an index set of fixed-size
 * blocks separated by a stride:
 */
int mtxidxset_init_blockstrided(
    struct mtxidxset * index_set,
    int64_t offset,
    int64_t size,
    int stride,
    int block_size);

/**
 * ‘mtxidxset_init_discrete()’ creates an index set of discrete
 * integer values given by an array.
 */
int mtxidxset_init_discrete(
    struct mtxidxset * index_set,
    int64_t size,
    const int64_t * indices);

/**
 * ‘mtxidxset_contains()’ returns ‘true’ if the given integer is
 * contained in the index set and ‘false’ otherwise.
 */
bool mtxidxset_contains(
    const struct mtxidxset * index_set,
    int64_t n);

/**
 * ‘mtxidxset_read()’ reads an index set from the given path as a
 * Matrix Market file in the form of an integer vector in array
 * format.
 *
 * If ‘path’ is ‘-’, then standard input is used.
 *
 * If an error code is returned, then ‘lines_read’ and ‘bytes_read’
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the Matrix Market file.
 */
int mtxidxset_read(
    struct mtxidxset * index_set,
    const char * path,
    int * lines_read,
    int64_t * bytes_read);

/**
 * ‘mtxidxset_fread()’ reads an index set from a stream as a Matrix
 * Market file in the form of an integer vector in array format.
 *
 * If an error code is returned, then ‘lines_read’ and ‘bytes_read’
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the Matrix Market file.
 */
int mtxidxset_fread(
    struct mtxidxset * index_set,
    FILE * f,
    int * lines_read,
    int64_t * bytes_read,
    size_t line_max,
    char * linebuf);

/**
 * ‘mtxidxset_write()’ writes an index set to the given path as a
 * Matrix Market file in the form of an integer vector in array
 * format.
 *
 * If ‘path’ is ‘-’, then standard output is used.
 *
 * If ‘format’ is not ‘NULL’, then the given format string is used
 * when printing numerical values.  The format specifier must be '%d',
 * and a fixed field width may optionally be specified (e.g., "%3d"),
 * but variable field width (e.g., "%*d"), as well as length modifiers
 * (e.g., "%ld") are not allowed.  If ‘format’ is ‘NULL’, then the
 * format specifier '%d' is used.
 */
int mtxidxset_write(
    const struct mtxidxset * index_set,
    const char * path,
    const char * format,
    int64_t * bytes_written);

/**
 * ‘mtxidxset_fwrite()’ writes an index set to a stream as a Matrix
 * Market file in the form of an integer vector in array format.
 *
 * If ‘format’ is not ‘NULL’, then the given format string is used
 * when printing numerical values.  The format specifier must be '%d',
 * and a fixed field width may optionally be specified (e.g., "%3d"),
 * but variable field width (e.g., "%*d"), as well as length modifiers
 * (e.g., "%ld") are not allowed.  If ‘format’ is ‘NULL’, then the
 * format specifier '%d' is used.
 *
 * If it is not ‘NULL’, then the number of bytes written to the stream
 * is returned in ‘bytes_written’.
 */
int mtxidxset_fwrite(
    const struct mtxidxset * index_set,
    FILE * f,
    const char * format,
    int64_t * bytes_written);

/*
 * MPI functions
 */

#ifdef LIBMTX_HAVE_MPI
/**
 * ‘mtxidxset_send()’ sends an index set to another MPI process.
 *
 * This is analogous to ‘MPI_Send()’ and requires the receiving
 * process to perform a matching call to ‘mtxidxset_recv()’.
 */
int mtxidxset_send(
    const struct mtxidxset * index_set,
    int dest,
    int tag,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxidxset_recv()’ receives an index set from another MPI process.
 *
 * This is analogous to ‘MPI_Recv()’ and requires the sending process
 * to perform a matching call to ‘mtxidxset_send()’.
 */
int mtxidxset_recv(
    struct mtxidxset * index_set,
    int source,
    int tag,
    MPI_Comm comm,
    struct mtxdisterror * disterr);
#endif

#endif
