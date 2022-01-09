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

#include <libmtx/util/index_set.h>

#include <stdint.h>

/**
 * ‘mtxpartitioning’ enumerates different kinds of partitionings.
 */
enum mtxpartitioning
{
    mtx_singleton,    /* trivial partition */
    mtx_block,        /* contiguous, fixed-size blocks */
    mtx_cyclic,       /* cyclic partition */
    mtx_block_cyclic, /* cyclic partition of fixed-size blocks. */
    mtx_unstructured, /* unstructured partition */
};

/**
 * ‘mtxpartitioning_str()’ is a string representing the partition
 * type.
 */
const char * mtxpartitioning_str(
    enum mtxpartitioning type);

/**
 * ‘mtx_parse_partition_type()’ parses a string to obtain one of the
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
 * On success, ‘mtx_parse_partition_type()’ returns ‘MTX_SUCCESS’ and
 * ‘partition_type’ is set according to the parsed string and
 * ‘bytes_read’ is set to the number of bytes that were consumed by
 * the parser.  Otherwise, an error code is returned.
 */
int mtx_parse_partition_type(
    enum mtxpartitioning * partition_type,
    int64_t * bytes_read,
    const char ** endptr,
    const char * s,
    const char * valid_delimiters);

/**
 * ‘mtx_partition’ represents a partitioning of a finite set.
 */
struct mtx_partition
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
     * ‘index_sets’ is an array containing an index set for each part,
     * where the pth index set describes the elements of the
     * partitioned set belonging to the pth part of the partition.
     */
    struct mtx_index_set * index_sets;

    /**
     * ‘parts’ is an array containing the part number assigned to each
     * element in the partitioned set, if ‘type’ is
     * ‘mtx_unstructured’.  Otherwise, this value is not used.
     */
    int * parts;
};

/**
 * ‘mtx_partition_free()’ frees resources associated with a
 * partitioning.
 */
void mtx_partition_free(
    struct mtx_partition * partition);

/**
 * ‘mtx_partition_init()’ initialises a partitioning of a finite set.
 */
int mtx_partition_init(
    struct mtx_partition * partition,
    enum mtxpartitioning type,
    int64_t size,
    int num_parts,
    int block_size,
    const int * parts);

/**
 * ‘mtx_partition_init_singleton()’ initialises a singleton partition
 * of a finite set.  That is, a partition with only one part, also
 * called the trivial partition.
 */
int mtx_partition_init_singleton(
    struct mtx_partition * partition,
    int64_t size);

/**
 * ‘mtx_partition_init_block()’ initialises a block partitioning of a
 * finite set.
 */
int mtx_partition_init_block(
    struct mtx_partition * partition,
    int64_t size,
    int num_parts);

/**
 * ‘mtx_partition_init_cyclic()’ initialises a cyclic partitioning of
 * a finite set.
 */
int mtx_partition_init_cyclic(
    struct mtx_partition * partition,
    int64_t size,
    int num_parts);

/**
 * ‘mtx_partition_init_block_cyclic()’ initialises a block-cyclic
 * partitioning of a finite set.
 */
int mtx_partition_init_block_cyclic(
    struct mtx_partition * partition,
    int64_t size,
    int num_parts,
    int block_size);

/**
 * ‘mtx_partition_init_unstructured()’ initialises an unstructured
 * partitioning of a finite set.
 */
int mtx_partition_init_unstructured(
    struct mtx_partition * partition,
    int64_t size,
    int num_parts,
    const int * parts);

/**
 * ‘mtx_partition_part()’ determines which part of a partition that a
 * given element belongs to.
 */
int mtx_partition_part(
    const struct mtx_partition * partition,
    int * p,
    int64_t n);

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
 * ‘mtx_partition_read_parts()’ reads the part numbers assigned to
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
int mtx_partition_read_parts(
    struct mtx_partition * partition,
    int num_parts,
    const char * path,
    int * lines_read,
    int64_t * bytes_read);

/**
 * ‘mtx_partition_fread_parts()’ reads the part numbers assigned to
 * each element of a partitioned set from a stream formatted as a
 * Matrix Market file.  The Matrix Market file must be in the form of
 * an integer vector in array format.
 *
 * If an error code is returned, then ‘lines_read’ and ‘bytes_read’
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the Matrix Market file.
 */
int mtx_partition_fread_parts(
    struct mtx_partition * partition,
    int num_parts,
    FILE * f,
    int * lines_read,
    int64_t * bytes_read,
    size_t line_max,
    char * linebuf);

/**
 * ‘mtx_partition_fread_indices()’ reads the global indices of
 * elements belonging to a given part of a partitioned set from a
 * stream formatted as a Matrix Market file.  The Matrix Market file
 * must be in the form of an integer vector in array format.
 *
 * If an error code is returned, then ‘lines_read’ and ‘bytes_read’
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the Matrix Market file.
 */
int mtx_partition_fread_indices(
    struct mtx_partition * partition,
    int part,
    FILE * f,
    int * lines_read,
    int64_t * bytes_read,
    size_t line_max,
    char * linebuf);

/**
 * ‘mtx_partition_write_parts()’ writes the part numbers assigned to
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
int mtx_partition_write_parts(
    const struct mtx_partition * partition,
    const char * path,
    const char * format,
    int64_t * bytes_written);

/**
 * ‘mtx_partition_fwrite_parts()’ writes the part numbers assigned to
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
int mtx_partition_fwrite_parts(
    const struct mtx_partition * partition,
    FILE * f,
    const char * format,
    int64_t * bytes_written);

/**
 * ‘mtx_partition_write_permutation()’ writes the permutation of a
 * given part of a partitioned set to the given path.  The permutation
 * is represented by an array of global indices of the elements
 * belonging to the given part prior to partitioning.  The file is
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
int mtx_partition_write_permutation(
    const struct mtx_partition * partition,
    int part,
    const char * path,
    const char * format,
    int64_t * bytes_written);

/**
 * ‘mtx_partition_write_permutations()’ writes the permutations for
 * each part of a partitioned set to the given path.  The permutation
 * is represented by an array of global indices of the elements
 * belonging each part prior to partitioning.  The file for each part
 * is written as a Matrix Market file in the form of an integer vector
 * in array format.
 *
 * Each occurrence of '%p' in ‘pathfmt’ is replaced by the number of
 * each part number.
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
int mtx_partition_write_permutations(
    const struct mtx_partition * partition,
    const char * pathfmt,
    const char * format,
    int64_t * bytes_written);

/**
 * ‘mtx_partition_write_permutation()’ writes the permutation of a
 * given part of a partitioned set to a stream as a Matrix Market
 * file.  The permutation is represented by an array of global indices
 * of the elements belonging to the given part prior to partitioning.
 * The file is written as a Matrix Market file in the form of an
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
int mtx_partition_fwrite_permutation(
    const struct mtx_partition * partition,
    int part,
    FILE * f,
    const char * format,
    int64_t * bytes_written);

#endif
