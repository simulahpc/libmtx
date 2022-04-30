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
 * Last modified: 2022-04-30
 *
 * Data types and functions for partitioning finite sets.
 */

#ifndef LIBMTX_UTIL_PARTITION_H
#define LIBMTX_UTIL_PARTITION_H

#include <libmtx/libmtx-config.h>

#ifdef LIBMTX_HAVE_MPI
#include <mpi.h>
#endif

#include <stdint.h>
#include <stdio.h>

struct mtxdisterror;

/*
 * Types of partitioning
 */

/**
 * ‘mtxpartitioning’ enumerates different kinds of partitionings.
 */
enum mtxpartitioning
{
    mtx_singleton,         /* singleton partition with only one component */
    mtx_block,             /* contiguous, fixed-size blocks */
    mtx_cyclic,            /* cyclic partition */
    mtx_block_cyclic,      /* cyclic partition of fixed-size blocks */
    mtx_custom_partition,  /* general, user-defined partition */
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
 * partitioning sets of integers
 */

/**
 * ‘partition_block_int64()’ partitions elements of a set of 64-bit
 * signed integers based on a block partitioning to produce an array
 * of part numbers assigned to each element in the input array.
 *
 * The array to be partitioned, ‘idx’, contains ‘idxsize’ items.
 * Moreover, the user must provide an output array, ‘dstpart’, of size
 * ‘idxsize’, which is used to write the part number assigned to each
 * element of the input array.
 *
 * The set to be partitioned consists of ‘size’ items that are
 * partitioned into ‘num_parts’ contiguous blocks. Furthermore, the
 * array ‘partsizes’ contains ‘num_parts’ integers, specifying the
 * size of each block of the partitioned set.
 */
int partition_block_int64(
    int64_t size,
    int num_parts,
    const int64_t * partsizes,
    int64_t idxsize,
    int idxstride,
    const int64_t * idx,
    int * dstpart);

/**
 * ‘partition_block_cyclic_int64()’ partitions elements of a set of
 * 64-bit signed integers based on a block-cyclic partitioning to
 * produce an array of part numbers assigned to each element in the
 * input array.
 *
 * The array to be partitioned, ‘idx’, contains ‘idxsize’ items.
 * Moreover, the user must provide an output array, ‘dstpart’, of size
 * ‘idxsize’, which is used to write the part number assigned to each
 * element of the input array.
 *
 * The set to be partitioned consists of ‘size’ items arranged in
 * contiguous block of size ‘blksize’, which are then partitioned in a
 * cyclic fashion into ‘num_parts’ parts.
 */
int partition_block_cyclic_int64(
    int64_t size,
    int num_parts,
    int64_t blksize,
    int64_t idxsize,
    int idxstride,
    const int64_t * idx,
    int * dstpart);

/**
 * ‘partition_custom_int64()’ partitions elements of a set of 64-bit
 * signed integers based on a user-defined partitioning to produce an
 * array of part numbers assigned to each element in the input array.
 *
 * The array to be partitioned, ‘idx’, contains ‘idxsize’ items.
 * Moreover, the user must provide an output array, ‘dstpart’, of size
 * ‘idxsize’, which is used to write the part number assigned to each
 * element of the input array.
 *
 * The set to be partitioned consists of ‘size’ items. Moreover,
 * ‘parts’ is an array of length ‘size’, which specifies the part
 * number of each element in the set.
 */
int partition_custom_int64(
    int64_t size,
    int num_parts,
    const int64_t * parts,
    int64_t idxsize,
    int idxstride,
    const int64_t * idx,
    int * dstpart);

/**
 * ‘partition_int64()’ partitions elements of a set of 64-bit signed
 * integers based on a given partitioning to produce an array of part
 * numbers assigned to each element in the input array.
 *
 * The array to be partitioned, ‘idx’, contains ‘idxsize’ items.
 * Moreover, the user must provide an output array, ‘dstpart’, of size
 * ‘idxsize’, which is used to write the part number assigned to each
 * element of the input array.
 *
 * The set to be partitioned consists of ‘size’ items.
 *
 * - If ‘type’ is ‘mtx_block’, then the array ‘partsizes’ contains
 *   ‘num_parts’ integers, specifying the size of each block of the
 *   partitioned set.
 *
 * - If ‘type’ is ‘mtx_block_cyclic’, then items are arranged in
 *   contiguous block of size ‘blksize’, which are then partitioned in
 *   a cyclic fashion into ‘num_parts’ parts.
 *
 * - If ‘type’ is ‘mtx_block_cyclic’, then ‘parts’ is an array of
 *   length ‘size’, which specifies the part number of each element in
 *   the set.
 */
int partition_int64(
    enum mtxpartitioning type,
    int64_t size,
    int num_parts,
    const int64_t * partsizes,
    int64_t blksize,
    const int64_t * parts,
    int64_t idxsize,
    int idxstride,
    const int64_t * idx,
    int * dstpart);

/*
 * distributed partitioning of sets of integers
 */

#ifdef LIBMTX_HAVE_MPI
/**
 * ‘distpartition_block_int64()’ partitions elements of a set of
 * 64-bit signed integers based on a block partitioning to produce an
 * array of part numbers assigned to each element in the input array.
 *
 * The array to be partitioned, ‘idx’, contains ‘idxsize’ items.
 * Moreover, the user must provide an output array, ‘dstpart’, of size
 * ‘idxsize’, which is used to write the part number assigned to each
 * element of the input array.
 *
 * The set to be partitioned consists of ‘size’ items that are
 * partitioned into ‘P’ contiguous blocks, where ‘P’ is the number of
 * processes in the MPI communicator ‘comm’. Furthermore, ‘partsize’
 * is used to specify the size of the block on the current process.
 */
int distpartition_block_int64(
    int64_t size,
    int64_t partsize,
    int64_t idxsize,
    int idxstride,
    const int64_t * idx,
    int * dstpart,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘distpartition_block_cyclic_int64()’ partitions elements of a set
 * of 64-bit signed integers based on a block-cyclic partitioning to
 * produce an array of part numbers assigned to each element in the
 * input array.
 *
 * The array to be partitioned, ‘idx’, contains ‘idxsize’ items.
 * Moreover, the user must provide an output array, ‘dstpart’, of size
 * ‘idxsize’, which is used to write the part number assigned to each
 * element of the input array.
 *
 * The partitioned set consists of ‘size’ items arranged in contiguous
 * block of size ‘blksize’, which are then partitioned in a cyclic
 * fashion into ‘P’ parts, where ‘P’ is the number of processes in the
 * MPI communicator ‘comm’.
 */
int distpartition_block_cyclic_int64(
    int64_t size,
    int64_t blksize,
    int64_t idxsize,
    int idxstride,
    const int64_t * idx,
    int * dstpart,
    MPI_Comm comm,
    struct mtxdisterror * disterr);
#endif

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
     * ‘parts’ is an array of length ‘size’, if ‘type’ is
     * ‘mtx_custom_partition’, containing the part number assigned to
     * each element in the partitioned set.
     *
     * If ‘type’ is not ‘mtx_custom_partition’, then ‘parts’ is set to ‘NULL’
     * and is not used.
     */
    int * parts;

    /**
     * ‘elements_per_part’ is an array of length ‘size’, if ‘type’ is
     * ‘mtx_custom_partition’. The elements belonging to the ‘p’th part are
     * given by ‘elements_per_part[i]’, for ‘i’ in ‘parts_ptr[p],
     * parts_ptr[p]+1, ..., parts_ptr[p+1]-1’.
     *
     * If ‘type’ is not ‘mtx_custom_partition’, then ‘elements_per_part’ is
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
    const int * parts,
    const int64_t * elements_per_part);

/**
 * ‘mtxpartition_copy()’ creates a copy of a partitioning.
 */
int mtxpartition_copy(
    struct mtxpartition * dst,
    const struct mtxpartition * src);

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
 * finite set. Each block is made up of a contiguous set of elements,
 * but blocks may vary in size.
 *
 * If ‘part_sizes’ is ‘NULL’, then the elements are divided into
 * blocks of equal size. Otherwise, ‘part_sizes’ must point to an
 * array of length ‘num_parts’ containing the number of elements in
 * each part. Moreover, the sum of the entries in ‘part_sizes’ must be
 * equal to ‘size’.
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
 * ‘mtxpartition_init_custom()’ initialises a user-defined
 * partitioning of a finite set.
 *
 * In the simplest case, a partition can be created by specifying the
 * part number for each element in the set. Elements remain ordered
 * within each part according to their global element numbers, and
 * thus no additional reordering is performed. To achieve this,
 * ‘part_sizes’ and ‘elements_per_part’ must both be ‘NULL’, and
 * ‘parts’ must point to an array of length ‘size’. Moreover, each
 * entry in ‘parts’ is a non-negative integer less than ‘num_parts’,
 * which assigns a part number to the corresponding global element.
 *
 * Alternatively, a partition may be specified by providing the global
 * element numbers for the elements that make up each part. This
 * method also allows an arbitrary ordering or numbering of elements
 * within each part. Thus, there is no requirement for elements within
 * a part to be ordered according to their global element numbers.
 *
 * To create a custom partition with arbitrary ordering of local
 * elements, ‘part_sizes’ and ‘elements_per_part’ must both be
 * non-‘NULL’. The former must point to an array of size ‘num_parts’,
 * whereas the latter must point to an array of length ‘size’. In this
 * case, ‘parts’ is ignored and may be set to ‘NULL’. Moreover,
 * ‘part_sizes’ must contain non-negative integers that specify the
 * number of elements in each part, and whose sum must be equal to
 * ‘size’. For a given part ‘p’, taking the sum of the ‘p-1’ first
 * integers in ‘part_sizes’, that is, ‘r := part_sizes[0] +
 * part_sizes[1] + ... + part_sizes[p-1]’, gives the location in the
 * array ‘elements_per_part’ of the first element belonging to the pth
 * part. Thus, the pth part of the partition is made up of the
 * elements ‘elements_per_part[r]’, ‘elements_per_part[r+1]’, ...,
 * ‘elements_per_part[r+part_sizes[p]-1]’.
 *
 * As mentioned above, any ordering of elements is allowed within each
 * part. However, some operations, such as halo updates or exchanges
 * can sometimes be carried out more efficiently if certain rules are
 * observed. For example, some reordering steps may be avoided if the
 * elements in each part are already ordered in ascendingly by global
 * element numbers. (In other words, ‘elements_per_part[r] <
 * elements_per_part[r+1] < ... <
 * elements_per_part[r+part_sizes[p]-1]’.)
 */
int mtxpartition_init_custom(
    struct mtxpartition * partition,
    int64_t size,
    int num_parts,
    const int * parts,
    const int64_t * part_sizes,
    const int64_t * elements_per_part);

/**
 * ‘mtxpartition_compare()’ checks if two partitions are the same.
 *
 * ‘result’ must point to an integer, which is used to return the
 * result of the comparison. If ‘a’ and ‘b’ are the same partitioning
 * of the same set, then the integer pointed to by ‘result’ will be
 * set to 0. Otherwise, it is set to some nonzero value.
 */
int mtxpartition_compare(
    const struct mtxpartition * a,
    const struct mtxpartition * b,
    int * result);

/**
 * ‘mtxpartition_assign()’ assigns part numbers to elements of an
 * array according to the partitioning.
 *
 * ‘elements’ must point to an array of length ‘size’ that is used to
 * specify (global) element numbers. If ‘parts’ is not ‘NULL’, then it
 * must also point to an array of length ‘size’, which is then used to
 * store the corresponding part number of each element in the
 * ‘elements’ array.
 *
 * Finally, if ‘localelem’ is not ‘NULL’, then it must point to an
 * array of length ‘size’. For each global element number in the
 * ‘elements’ array, ‘localelem’ is used to store the corresponding
 * local, partwise element number based on the numbering of elements
 * within each part. The ‘elements’ and ‘localelem’ pointers are
 * allowed to point to the same underlying array, in which case the
 * former is overwritten by the latter.
 */
int mtxpartition_assign(
    const struct mtxpartition * partition,
    int64_t size,
    const int64_t * elements,
    int * parts,
    int64_t * localelem);

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
 */
int mtxpartition_globalidx(
    const struct mtxpartition * partition,
    int part,
    int64_t size,
    const int64_t * localelem,
    int64_t * globalelem);

/**
 * ‘mtxpartition_localidx()’ translates from a global numbering of
 * elements in the partitioned set to a local numbering of elements
 * within a given part.
 *
 * The argument ‘part’ denotes the part of the partition for which the
 * local element numbers are obtained.
 *
 * The arrays ‘globalelem’ and ‘localelem’ must be of length equal to
 * ‘size’. The former is used to specify the global element numbers of
 * elements belonging to the specified part. 
 *
 * If successful, the array ‘localelem’ will contain local element
 * numbers in the range ‘0, 1, ..., partition->part_sizes[part]-1’
 * that were obtained by translating from the global element numbers
 * in ‘globalelem’.
 */
int mtxpartition_localidx(
    const struct mtxpartition * partition,
    int part,
    int64_t size,
    const int64_t * globalelem,
    int64_t * localelem);

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
    int64_t * lines_read,
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
    int64_t * lines_read,
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
    int64_t * lines_read,
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
