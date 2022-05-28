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
 * Last modified: 2022-05-28
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
 * ‘partition_cyclic_int64()’ partitions elements of a set of 64-bit
 * signed integers based on a cyclic partitioning to produce an array
 * of part numbers assigned to each element in the input array.
 *
 * The array to be partitioned, ‘idx’, contains ‘idxsize’ items.
 * Moreover, the user must provide an output array, ‘dstpart’, of size
 * ‘idxsize’, which is used to write the part number assigned to each
 * element of the input array.
 *
 * The set to be partitioned consists of ‘size’ items, which are
 * partitioned in a cyclic fashion into ‘num_parts’ parts.
 */
int partition_cyclic_int64(
    int64_t size,
    int num_parts,
    int64_t idxsize,
    int idxstride,
    const int64_t * idx,
    int * dstpart);

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
 * - If ‘type’ is ‘mtx_cyclic’, then items are partitioned in a cyclic
 *   fashion into ‘num_parts’ parts.
 *
 * - If ‘type’ is ‘mtx_block’, then the array ‘partsizes’ contains
 *   ‘num_parts’ integers, specifying the size of each block of the
 *   partitioned set.
 *
 * - If ‘type’ is ‘mtx_block_cyclic’, then items are arranged in
 *   contiguous blocks of size ‘blksize’, which are then partitioned
 *   in a cyclic fashion into ‘num_parts’ parts.
 */
int partition_int64(
    enum mtxpartitioning type,
    int64_t size,
    int num_parts,
    const int64_t * partsizes,
    int64_t blksize,
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

/**
 * ‘distpartition_int64()’ partitions elements of a set of 64-bit
 * signed integers based on a given partitioning to produce an array
 * of part numbers assigned to each element in the input array.
 *
 * The number of parts is equal to the number of processes in the
 * communicator ‘comm’.
 *
 * The array to be partitioned, ‘idx’, contains ‘idxsize’ items.
 * Moreover, the user must provide an output array, ‘dstpart’, of size
 * ‘idxsize’, which is used to write the part number assigned to each
 * element of the input array.
 *
 * The set to be partitioned consists of ‘size’ items.
 *
 * - If ‘type’ is ‘mtx_block’, then ‘partsize’ specifies the size of
 *   the block on the current MPI process.
 *
 * - If ‘type’ is ‘mtx_block_cyclic’, then items are arranged in
 *   contiguous blocks of size ‘blksize’, which are then partitioned
 *   in a cyclic fashion.
 */
int distpartition_int64(
    enum mtxpartitioning type,
    int64_t size,
    int64_t partsize,
    int64_t blksize,
    int64_t idxsize,
    int idxstride,
    const int64_t * idx,
    int * dstpart,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/*
 * The “assumed partition” strategy, a parallel rendezvous protocol
 * algorithm to answer queries about the ownership and location of
 * distributed data.
 *
 * See A.H. Baker, R.D. Falgout and U.M. Yang, “An assumed partition
 * algorithm for determining processor inter-communication,” Parallel
 * Computing, Vol. 32, No. 5–6, June 2006, pp. 319–414.
 */

/**
 * ‘assumedpartition_write()’ sets up the data structures needed to
 * query about ownership and location of distributed data using the
 * assumed partition strategy.
 *
 * The method assumed an underlying one-dimensional array of ‘size’
 * elements distributed among ‘P’ processes, where ‘P’ is the size of
 * the MPI communicator ‘comm’.
 *
 * On a given process, ‘partsize’ is the number of elements it owns
 * and ‘globalidx’ is an array containing the global numbers of those
 * elements. Because ‘globalidx’ may be modified (sorted) by the
 * function, the array ‘perm’ is used to write the permutation that is
 * applied to the elements of ‘globalidx’.
 *
 * The ‘apownerrank’ and ‘apowneridx’ arrays are of length ‘apsize’,
 * which must be at least equal to ‘(size+P-1)/P’. For each element
 * that is assumed to be owned by a given process (according to an
 * equal-sized block partitioning), these arrays are used to write the
 * actual owning rank, as well as the offset to the element among
 * elements of the owning rank.
 */
int assumedpartition_write(
    int64_t size,
    int64_t partsize,
    const int64_t * globalidx,
    int apsize,
    int * apownerrank,
    int * apowneridx,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘assumedpartition_read()’ performs a query to learn the ownership
 * and location of distributed data using the assumed partition
 * strategy.
 *
 * The method assumed an underlying one-dimensional array of ‘size’
 * elements distributed among ‘P’ processes, where ‘P’ is the size of
 * the MPI communicator ‘comm’.
 *
 * The ‘apownerrank’ and ‘apowneridx’ arrays are of length ‘apsize’,
 * which must be at least equal to ‘(size+P-1)/P’. For each element
 * that is assumed to be owned by a given process (according to an
 * equal-sized block partitioning), these arrays specify the actual
 * owning rank, as well as the offset to the element among elements of
 * the owning rank. See also ‘assumedpartition_write’ for how to set
 * up these arrays.
 *
 * On a given process, ‘partsize’ is the number of elements to query
 * for ownership information and ‘globalidx’ is an array containing
 * the global numbers of those elements. Because ‘globalidx’ may be
 * modified (sorted) by the function, the array ‘perm’ is used to
 * write the permutation that is applied to the elements of
 * ‘globalidx’.
 *
 * The ‘ownerrank’ and ‘owneridx’ arrays must also be of length
 * ‘partsize’. They are used to write the actual owning rank, as well
 * as the offset to the element among elements of the owning rank.
 */
int assumedpartition_read(
    int64_t size,
    int apsize,
    const int * apownerrank,
    const int * apowneridx,
    int64_t partsize,
    const int64_t * globalidx,
    int * ownerrank,
    int * owneridx,
    MPI_Comm comm,
    struct mtxdisterror * disterr);
#endif

#endif
