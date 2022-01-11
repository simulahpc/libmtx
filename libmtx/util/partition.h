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
 * Data types and functions for partitioning finite sets.
 */

#ifndef LIBMTX_UTIL_PARTITION_H
#define LIBMTX_UTIL_PARTITION_H

#include <stdint.h>
#include <stdio.h>

/*
 * Types of partitioning
 */

/**
 * ‘mtxpartitioning’ enumerates different kinds of partitionings.
 */
enum mtxpartitioning
{
    mtx_singleton,    /* singleton partition with only one component */
    mtx_block,        /* contiguous, fixed-size blocks */
    mtx_cyclic,       /* cyclic partition */
    mtx_block_cyclic, /* cyclic partition of fixed-size blocks. */
    mtx_partition,    /* general, user-defined partition */
};

/**
 * ‘mtxpartitioning_str()’ is a string representing the partition
 * type.
 */
const char * mtxpartitioning_str(
    enum mtxpartitioning type);

/**
 * ‘mtxpartitioning_parse()’ parses a string to obtain one of the
 * partition types of ‘enum mtxpartitioning’.
 *
 * ‘valid_delimiters’ is either ‘NULL’, in which case it is ignored,
 * or it is a string of characters considered to be valid delimiters
 * for the parsed string.  That is, if there are any remaining,
 * non-NULL characters after parsing, then then the next character is
 * searched for in ‘valid_delimiters’.  If the character is found,
 * then the parsing succeeds and the final delimiter character is
 * consumed by the parser. Otherwise, the parsing fails with an error.
 *
 * If ‘endptr’ is not ‘NULL’, then the address stored in ‘endptr’
 * points to the first character beyond the characters that were
 * consumed during parsing.
 *
 * On success, ‘mtxpartitioning_parse()’ returns ‘MTX_SUCCESS’ and
 * ‘partition_type’ is set according to the parsed string and
 * ‘bytes_read’ is set to the number of bytes that were consumed by
 * the parser.  Otherwise, an error code is returned.
 */
int mtxpartitioning_parse(
    enum mtxpartitioning * partition_type,
    int64_t * bytes_read,
    const char ** endptr,
    const char * s,
    const char * valid_delimiters);

/*
 * Partitions of finite sets
 */

/**
 * ‘mtxpartition’ represents a partitioning of a finite set.
 */
struct mtxpartition
{
    /**
     * ‘type’ is the type of partitioning.
     */
    enum mtxpartitioning type;

    /**
     * ‘size’ is the number of elements in the partitioned set.
     */
    int64_t size;

    /**
     * ‘num_parts’ is the number of parts in the partition.
     */
    int num_parts;

    /**
     * ‘part_sizes’ is an array containing the number of elements in
     * each part of the partition.
     */
    int64_t * part_sizes;

    /**
     * ‘parts_ptr’ is an array of length ‘num_parts+1’, containing
     * offsets to the first elements of each part in the partition.
     */
    int64_t * parts_ptr;

    /**
     * ‘parts’ is an array is an array of length ‘size’, if ‘type’ is
     * ‘mtx_partition’, containing the part number assigned to each
     * element in the partitioned set.
     *
     * If ‘type’ is not ‘mtx_partition’, then ‘parts’ is set to ‘NULL’
     * and is not used.
     */
    int * parts;

    /**
     * ‘elements_per_part’ is an array of length ‘size’, if ‘type’ is
     * ‘mtx_partition’. The elements belonging to the ‘p’th part are
     * given by ‘elements_per_part[i]’, for ‘i’ in ‘parts_ptr[p],
     * parts_ptr[p]+1, ..., parts_ptr[p+1]-1’.
     *
     * If ‘type’ is not ‘mtx_partition’, then ‘elements_per_part’ is
     * set to ‘NULL’ and is not used.
     */
    int64_t * elements_per_part;
};

/**
 * ‘mtxpartition_free()’ frees resources associated with a
 * partitioning.
 */
void mtxpartition_free(
    struct mtxpartition * partition);

/**
 * ‘mtxpartition_init()’ initialises a partitioning of a finite set.
 */
int mtxpartition_init(
    struct mtxpartition * partition,
    enum mtxpartitioning type,
    int64_t size,
    int num_parts,
    const int64_t * part_sizes,
    int block_size,
    const int * parts);

/**
 * ‘mtxpartition_init_singleton()’ initialises a singleton partition
 * of a finite set.  That is, a partition with only one part, also
 * called the trivial partition.
 */
int mtxpartition_init_singleton(
    struct mtxpartition * partition,
    int64_t size);

/**
 * ‘mtxpartition_init_block()’ initialises a block partitioning of a
 * finite set.
 */
int mtxpartition_init_block(
    struct mtxpartition * partition,
    int64_t size,
    int num_parts,
    const int64_t * part_sizes);

/**
 * ‘mtxpartition_init_cyclic()’ initialises a cyclic partitioning of
 * a finite set.
 */
int mtxpartition_init_cyclic(
    struct mtxpartition * partition,
    int64_t size,
    int num_parts);

/**
 * ‘mtxpartition_init_block_cyclic()’ initialises a block-cyclic
 * partitioning of a finite set.
 */
int mtxpartition_init_block_cyclic(
    struct mtxpartition * partition,
    int64_t size,
    int num_parts,
    int block_size);

/**
 * ‘mtxpartition_init_partition()’ initialises a user-defined
 * partitioning of a finite set.
 */
int mtxpartition_init_partition(
    struct mtxpartition * partition,
    int64_t size,
    int num_parts,
    const int * parts);

/**
 * ‘mtxpartition_assign()’ assigns part numbers to elements of an
 * array according to the partitioning.
 *
 * The arrays ‘elements’ and ‘parts’ must both contain enough storage
 * for ‘size’ values of type ‘int’. If successful, ‘parts’ will
 * contain the part numbers of each element in the ‘elements’ array.
 *
 * If needed, ‘elements’ and ‘parts’ are allowed to point to the same
 * underlying array. The values of ‘elements’ will then be overwritten
 * by the assigned part numbers.
 */
int mtxpartition_assign(
    const struct mtxpartition * partition,
    int64_t size,
    const int * elements,
    int * parts);

/**
 * ‘mtxpartition_globalidx()’ translates from a local numbering of
 * elements within a given part to a global numbering of elements in
 * the partitioned set.
 *
 * The argument ‘part’ denotes the part of the partition for which the
 * local element numbers are given.
 *
 * The arrays ‘localelem’ and ‘globalelem’ must be of length equal to
 * ‘size’. The former is used to specify the local element numbers
 * within the specified part, and must therefore contain values in the
 * range ‘0, 1, ..., partition->part_sizes[part]-1’. If successful,
 * the array ‘globalelem’ will contain the global numbers
 * corresponding to each of the local element numbers in ‘localelem’.
 *
 * If needed, ‘localelem’ and ‘globalelem’ are allowed to point to the
 * same underlying array. The values of ‘localelem’ will then be
 * overwritten by the global element numbers.
 */
int mtxpartition_globalidx(
    const struct mtxpartition * partition,
    int part,
    int64_t size,
    const int * localelem,
    int * globalelem);

/*
 * I/O functions
 *
 * Reading a partition from file:
 *
 *   1. Read the part numbers for each element in the partitioned set
 *      from a single Matrix Market file.
 *
 *   2. Read the global indices of each element of the partitioned set
 *      from Matrix Market files for each part.
 *
 * Writing a partition to file:
 *
 *   1. Write the part numbers for each element in the partitioned set
 *      to a single Matrix Market file.
 *
 *   2. Write the global indices of each element of the partitioned
 *      set to a Matrix Market file for each part.
 */

/**
 * ‘mtxpartition_read_parts()’ reads the part numbers assigned to
 * each element of a partitioned set from the given path.  The path
 * must be to a Matrix Market file in the form of an integer vector in
 * array format.
 *
 * If ‘path’ is ‘-’, then standard input is used.
 *
 * If an error code is returned, then ‘lines_read’ and ‘bytes_read’
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the Matrix Market file.
 */
int mtxpartition_read_parts(
    struct mtxpartition * partition,
    int num_parts,
    const char * path,
    int * lines_read,
    int64_t * bytes_read);

/**
 * ‘mtxpartition_fread_parts()’ reads the part numbers assigned to
 * each element of a partitioned set from a stream formatted as a
 * Matrix Market file.  The Matrix Market file must be in the form of
 * an integer vector in array format.
 *
 * If an error code is returned, then ‘lines_read’ and ‘bytes_read’
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the Matrix Market file.
 */
int mtxpartition_fread_parts(
    struct mtxpartition * partition,
    int num_parts,
    FILE * f,
    int * lines_read,
    int64_t * bytes_read,
    size_t line_max,
    char * linebuf);

/**
 * ‘mtxpartition_fread_indices()’ reads the global indices of
 * elements belonging to a given part of a partitioned set from a
 * stream formatted as a Matrix Market file.  The Matrix Market file
 * must be in the form of an integer vector in array format.
 *
 * If an error code is returned, then ‘lines_read’ and ‘bytes_read’
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the Matrix Market file.
 */
int mtxpartition_fread_indices(
    struct mtxpartition * partition,
    int part,
    FILE * f,
    int * lines_read,
    int64_t * bytes_read,
    size_t line_max,
    char * linebuf);

/**
 * ‘mtxpartition_write_parts()’ writes the part numbers assigned to
 * each element of a partitioned set to the given path.  The file is
 * written as a Matrix Market file in the form of an integer vector in
 * array format.
 *
 * If ‘path’ is ‘-’, then standard output is used.
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
int mtxpartition_write_parts(
    const struct mtxpartition * partition,
    const char * path,
    const char * format,
    int64_t * bytes_written);

/**
 * ‘mtxpartition_fwrite_parts()’ writes the part numbers assigned to
 * each element of a partitioned set to a stream formatted as a Matrix
 * Market file.  The Matrix Market file is written in the form of an
 * integer vector in array format.
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
int mtxpartition_fwrite_parts(
    const struct mtxpartition * partition,
    FILE * f,
    const char * format,
    int64_t * bytes_written);

#endif
