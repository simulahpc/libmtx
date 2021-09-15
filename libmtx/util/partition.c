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

#include <libmtx/error.h>
#include <libmtx/util/partition.h>

#include <errno.h>

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/**
 * `mtx_partition_type_str()' is a string representing the partition
 * type.
 */
const char * mtx_partition_type_str(
    enum mtx_partition_type type)
{
    switch (type) {
    case mtx_nonpartitioned: return "non-partitioned";
    case mtx_block: return "block";
    case mtx_cyclic: return "cyclic";
    case mtx_block_cyclic: return "block-cyclic";
    default: return mtx_strerror(MTX_ERR_INVALID_PARTITION_TYPE);
    }
}

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
    const char * valid_delimiters)
{
    const char * t = s;
    if (strncmp("block", t, strlen("block")) == 0) {
        t += strlen("block");
        *partition_type = mtx_block;
    } else if (strncmp("cyclic", t, strlen("cyclic")) == 0) {
        t += strlen("cyclic");
        *partition_type = mtx_cyclic;
    } else if (strncmp("block-cyclic", t, strlen("block-cyclic")) == 0) {
        t += strlen("block-cyclic");
        *partition_type = mtx_block_cyclic;
    } else {
        return MTX_ERR_INVALID_PARTITION_TYPE;
    }
    if (valid_delimiters && *t != '\0') {
        if (!strchr(valid_delimiters, *t))
            return MTX_ERR_INVALID_PARTITION_TYPE;
        t++;
    }
    if (bytes_read)
        *bytes_read += t-s;
    if (endptr)
        *endptr = t;
    return MTX_SUCCESS;
}

/**
 * `mtx_partition_free()' frees resources associated with a
 * partitioning.
 */
void mtx_partition_free(
    struct mtx_partition * partition)
{
    free(partition->size_per_part);
}

/**
 * `mtx_partition_init()' initialises a partitioning of a finite set.
 */
int mtx_partition_init(
    struct mtx_partition * partition,
    enum mtx_partition_type type,
    int64_t size,
    int num_parts,
    int block_size)
{
    if (type == mtx_nonpartitioned) {
        return mtx_partition_init_nonpartitioned(partition, size);
    } else if (type == mtx_block) {
        return mtx_partition_init_block(partition, size, num_parts);
    } else if (type == mtx_cyclic) {
        return mtx_partition_init_cyclic(partition, size, num_parts);
    } else if (type == mtx_block_cyclic) {
        return mtx_partition_init_block_cyclic(
            partition, size, num_parts, block_size);
    } else {
        return MTX_ERR_INVALID_PARTITION_TYPE;
    }
}

/**
 * `mtx_partition_init_nonpartitioned()' initialises a finite set that
 * is not partitioned.
 */
int mtx_partition_init_nonpartitioned(
    struct mtx_partition * partition,
    int64_t size)
{
    partition->type = mtx_nonpartitioned;
    partition->size = size;
    partition->num_parts = 1;
    partition->size_per_part = malloc(partition->num_parts * sizeof(int));
    if (!partition->size_per_part)
        return MTX_ERR_ERRNO;
    partition->size_per_part[0] = size;
    partition->block_size = -1;
    return MTX_SUCCESS;
}

/**
 * `mtx_partition_init_block()' initialises a block partitioning of a
 * finite set.
 */
int mtx_partition_init_block(
    struct mtx_partition * partition,
    int64_t size,
    int num_parts)
{
    partition->type = mtx_block;
    partition->size = size;
    partition->num_parts = num_parts;
    partition->size_per_part = malloc(num_parts * sizeof(int));
    if (!partition->size_per_part)
        return MTX_ERR_ERRNO;
    for (int p = 0; p < num_parts; p++) {
        partition->size_per_part[p] =
            size / num_parts + (p < (size % num_parts) ? 1 : 0);
    }
    partition->block_size = -1;
    return MTX_SUCCESS;
}

/**
 * `mtx_partition_init_cyclic()' initialises a cyclic partitioning of
 * a finite set.
 */
int mtx_partition_init_cyclic(
    struct mtx_partition * partition,
    int64_t size,
    int num_parts)
{
    partition->type = mtx_cyclic;
    partition->size = size;
    partition->num_parts = num_parts;
    partition->size_per_part = malloc(num_parts * sizeof(int));
    if (!partition->size_per_part)
        return MTX_ERR_ERRNO;
    for (int p = 0; p < num_parts; p++) {
        partition->size_per_part[p] =
            size / num_parts + (p < (size % num_parts) ? 1 : 0);
    }
    partition->block_size = -1;
    return MTX_SUCCESS;
}

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
 * `mtx_partition_part()' determines which part of a partition that a
 * given element belongs to.
 */
int mtx_partition_part(
    const struct mtx_partition * partition,
    int * p,
    int64_t n)
{
    if (n >= partition->size)
        return MTX_ERR_INDEX_OUT_OF_BOUNDS;

    if (partition->type == mtx_block) {
        int64_t size_per_part = partition->size / partition->num_parts;
        int64_t remainder = partition->size % partition->num_parts;
        *p = n / (size_per_part+1);
        if (*p >= remainder)
            *p = remainder + (n - remainder * (size_per_part+1)) / size_per_part;
    } else if (partition->type == mtx_cyclic) {
        *p = n % partition->num_parts;
    } else if (partition->type == mtx_block_cyclic) {
        /* TODO: Not implemented. */
        errno = ENOTSUP;
        return MTX_ERR_ERRNO;
    } else {
        return MTX_ERR_INVALID_PARTITION_TYPE;
    }
    return MTX_SUCCESS;
}
