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
 * Index sets.
 */

#ifndef LIBMTX_INDEX_SET_H
#define LIBMTX_INDEX_SET_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

/**
 * `mtx_index_set_type' enumerates different kinds of index sets.
 */
enum mtx_index_set_type
{
    mtx_index_set_interval, /* contiguous interval of integers */
    mtx_index_set_strided,  /* set of integers separated by a
                             * stride */
    mtx_index_set_block_strided,  /* fixed-size sets of integers
                                   * separated by a stride */
    mtx_index_set_discrete, /* discrete index set given by an array */
};

/**
 * `mtx_index_set_type_str()' is a string representing the index set
 * type.
 */
const char * mtx_index_set_type_str(
    enum mtx_index_set_type index_set_type);

/**
 * `mtx_index_set' is a data structure for index sets.
 */
struct mtx_index_set
{
    /**
     * `type' is the type of index set: `interval'.
     */
    enum mtx_index_set_type type;

    /**
     * `size' is the number of elements in the index set.
     */
    int64_t size;

    /**
     * `offset' is an offset to the first element of the indexed set,
     * if `type' is `mtx_index_set_strided' or
     * `mtx_index_set_block_strided'.  Otherwise, this value is not
     * used.
     */
    int64_t offset;

    /**
     * `stride' is a stride between elements of the indexed set, if
     * `type' is `mtx_index_set_strided' or
     * `mtx_index_set_block_strided'.  Otherwise, this value is not
     * used.
     */
    int stride;

    /**
     * `block_size' is the size of each block, if `type' is
     * `mtx_index_set_block_strided'.  Otherwise, this value is not
     * used.
     */
    int block_size;

    /**
     * `indices' is an array containing the indices of the index set,
     * if `type' is `mtx_index_set_discrete'.  Otherwise, this value
     * is not used.
     */
    int64_t * indices;
};

/**
 * `mtx_index_set_free()' frees resources associated with an index
 * set.
 */
void mtx_index_set_free(
    struct mtx_index_set * index_set);

/**
 * `mtx_index_set_init_interval()' creates an index set of contiguous
 * integers from an interval [a,b).
 */
int mtx_index_set_init_interval(
    struct mtx_index_set * index_set,
    int64_t a,
    int64_t b);

/**
 * `mtx_index_set_init_strided()' creates an index set of strided
 * integers from an interval:
 *
 *   `offset,offset+stride,offset+2*stride,...,offset+(size-1)*stride'.
 */
int mtx_index_set_init_strided(
    struct mtx_index_set * index_set,
    int64_t offset,
    int64_t size,
    int stride);

/**
 * `mtx_index_set_init_block_strided()' creates an index set of
 * fixed-size blocks separated by a stride:
 */
int mtx_index_set_init_block_strided(
    struct mtx_index_set * index_set,
    int64_t offset,
    int64_t size,
    int stride,
    int block_size);

/**
 * `mtx_index_set_init_discrete()' creates an index set of discrete
 * integer values given by an array.
 */
int mtx_index_set_init_discrete(
    struct mtx_index_set * index_set,
    int64_t size,
    const int64_t * indices);

/**
 * `mtx_index_set_contains()' returns `true' if the given integer is
 * contained in the index set and `false' otherwise.
 */
bool mtx_index_set_contains(
    const struct mtx_index_set * index_set,
    int64_t n);

/**
 * `mtx_index_set_read()' reads an index set from the given path as a
 * Matrix Market file in the form of an integer vector in array
 * format.
 *
 * If `path' is `-', then standard input is used.
 *
 * If an error code is returned, then `lines_read' and `bytes_read'
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the Matrix Market file.
 */
int mtx_index_set_read(
    struct mtx_index_set * index_set,
    const char * path,
    int * lines_read,
    int64_t * bytes_read);

/**
 * `mtx_index_set_fread()' reads an index set from a stream as a
 * Matrix Market file in the form of an integer vector in array
 * format.
 *
 * If an error code is returned, then `lines_read' and `bytes_read'
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the Matrix Market file.
 */
int mtx_index_set_fread(
    struct mtx_index_set * index_set,
    FILE * f,
    int * lines_read,
    int64_t * bytes_read,
    size_t line_max,
    char * linebuf);

/**
 * `mtx_index_set_write()' writes an index set to the given path as a
 * Matrix Market file in the form of an integer vector in array
 * format.
 *
 * If `path' is `-', then standard output is used.
 *
 * If `format' is not `NULL', then the given format string is used
 * when printing numerical values.  The format specifier must be '%d',
 * and a fixed field width may optionally be specified (e.g., "%3d"),
 * but variable field width (e.g., "%*d"), as well as length modifiers
 * (e.g., "%ld") are not allowed.  If `format' is `NULL', then the
 * format specifier '%d' is used.
 */
int mtx_index_set_write(
    const struct mtx_index_set * index_set,
    const char * path,
    const char * format,
    int64_t * bytes_written);

/**
 * `mtx_index_set_fwrite()' writes an index set to a stream as a
 * Matrix Market file in the form of an integer vector in array
 * format.
 *
 * If `format' is not `NULL', then the given format string is used
 * when printing numerical values.  The format specifier must be '%d',
 * and a fixed field width may optionally be specified (e.g., "%3d"),
 * but variable field width (e.g., "%*d"), as well as length modifiers
 * (e.g., "%ld") are not allowed.  If `format' is `NULL', then the
 * format specifier '%d' is used.
 *
 * If it is not `NULL', then the number of bytes written to the stream
 * is returned in `bytes_written'.
 */
int mtx_index_set_fwrite(
    const struct mtx_index_set * index_set,
    FILE * f,
    const char * format,
    int64_t * bytes_written);

#endif
