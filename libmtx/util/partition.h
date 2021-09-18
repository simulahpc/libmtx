/* This file is part of libmtx.
 *
 * Copyright (C) 2021 James D. Trotter
 *
 * libmtx is free software: you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * libmtx is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with libmtx.  If not, see
 * <https://www.gnu.org/licenses/>.
 *
 * Authors: James D. Trotter <james@simula.no>
 * Last modified: 2021-09-06
 *
 * Data types and functions for partitioning finite sets.
 */

#ifndef LIBMTX_UTIL_PARTITION_H
#define LIBMTX_UTIL_PARTITION_H

#include <stdint.h>

/**
 * `mtx_partition_type' enumerates different kinds of partitionings.
 */
enum mtx_partition_type
{
    mtx_nonpartitioned,
    mtx_block,          /* contiguous, fixed-size blocks */
    mtx_cyclic,         /* cyclic partitioning */
    mtx_block_cyclic,   /* cyclic partitioning of contiguous,
                         * fixed-size blocks. */
    mtx_unstructured,   /* unstructured partition */
};

/**
 * `mtx_partition_type_str()' is a string representing the partition
 * type.
 */
const char * mtx_partition_type_str(
    enum mtx_partition_type type);

/**
 * `mtx_parse_partition_type()' parses a string to obtain one of the
 * partition types of `enum mtx_partition_type'.
 *
 * `valid_delimiters' is either `NULL', in which case it is ignored,
 * or it is a string of characters considered to be valid delimiters
 * for the parsed string.  That is, if there are any remaining,
 * non-NULL characters after parsing, then then the next character is
 * searched for in `valid_delimiters'.  If the character is found,
 * then the parsing succeeds and the final delimiter character is
 * consumed by the parser. Otherwise, the parsing fails with an error.
 *
 * If `endptr' is not `NULL', then the address stored in `endptr'
 * points to the first character beyond the characters that were
 * consumed during parsing.
 *
 * On success, `mtx_parse_partition_type()' returns `MTX_SUCCESS' and
 * `partition_type' is set according to the parsed string and
 * `bytes_read' is set to the number of bytes that were consumed by
 * the parser.  Otherwise, an error code is returned.
 */
int mtx_parse_partition_type(
    enum mtx_partition_type * partition_type,
    int64_t * bytes_read,
    const char ** endptr,
    const char * s,
    const char * valid_delimiters);

/**
 * `mtx_partition' represents a partitioning of a finite set.
 */
struct mtx_partition
{
    /**
     * `type' is the type of partitioning.
     */
    enum mtx_partition_type type;

    /**
     * `size' is the number of elements in the partitioned set.
     */
    int64_t size;

    /**
     * `num_parts' is the number of parts in the partition.
     */
    int num_parts;

    /**
     * `size_per_part' is an array containing the size of each part of
     * the partition.
     */
    int * size_per_part;

    /**
     * `block_size' is the size of each block, if `type' is
     * `mtx_block_cyclic'. Otherwise, this value is ignored.
     */
    int block_size;

    /**
     * `parts' is an array containing the part number for each element
     * in the partitioned set.
     */
    int * parts;
};

/**
 * `mtx_partition_free()' frees resources associated with a
 * partitioning.
 */
void mtx_partition_free(
    struct mtx_partition * partition);

/**
 * `mtx_partition_init()' initialises a partitioning of a finite set.
 */
int mtx_partition_init(
    struct mtx_partition * partition,
    enum mtx_partition_type type,
    int64_t size,
    int num_parts,
    int block_size,
    const int * parts);

/**
 * `mtx_partition_init_nonpartitioned()' initialises a finite set that
 * is not partitioned.
 */
int mtx_partition_init_nonpartitioned(
    struct mtx_partition * partition,
    int64_t size);

/**
 * `mtx_partition_init_block()' initialises a block partitioning of a
 * finite set.
 */
int mtx_partition_init_block(
    struct mtx_partition * partition,
    int64_t size,
    int num_parts);

/**
 * `mtx_partition_init_cyclic()' initialises a cyclic partitioning of
 * a finite set.
 */
int mtx_partition_init_cyclic(
    struct mtx_partition * partition,
    int64_t size,
    int num_parts);

/**
 * `mtx_partition_init_block_cyclic()' initialises a block-cyclic
 * partitioning of a finite set.
 */
int mtx_partition_init_block_cyclic(
    struct mtx_partition * partition,
    int64_t size,
    int num_parts,
    int block_size);

/**
 * `mtx_partition_init_unstructured()' initialises an unstructured
 * partitioning of a finite set.
 */
int mtx_partition_init_unstructured(
    struct mtx_partition * partition,
    int64_t size,
    int num_parts,
    const int * parts);

/**
 * `mtx_partition_part()' determines which part of a partition that a
 * given element belongs to.
 */
int mtx_partition_part(
    const struct mtx_partition * partition,
    int * p,
    int64_t n);

#endif
