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
 * Last modified: 2022-10-10
 *
 * Data types and functions for partitioning finite sets.
 */

#include <libmtx/util/partition.h>

#include <errno.h>
#include <unistd.h>

#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

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
 *
 * If ‘dstpartsizes’ is not ‘NULL’, then it must be an array of length
 * ‘num_parts’, which is used to store the number of items assigned to
 * each part.
 *
 * Returns ‘0’ if successful, or ‘EINVAL’ if ‘num_parts’ is not a
 * positive integer or if any element in ‘idx’ lies outside of the
 * valid range ‘[0,size)’.
 */
int partition_cyclic_int64(
    int64_t size,
    int num_parts,
    int64_t idxsize,
    int idxstride,
    const int64_t * idx,
    int * dstpart,
    int64_t * dstpartsizes)
{
    if (num_parts <= 0) return EINVAL;
    if (dstpartsizes) { for (int p = 0; p < num_parts; p++) dstpartsizes[p] = 0; }
    for (int64_t i = 0; i < idxsize; i++) {
        int64_t x = *(const int64_t *) ((const unsigned char *) idx+i*idxstride);
        if (x < 0 || x >= size) return EINVAL;
        if (dstpart) dstpart[i] = x % num_parts;
        if (dstpartsizes) dstpartsizes[x % num_parts]++;
    }
    return 0;
}

/**
 * ‘partition_cyclic_int()’ partitions elements of a set of signed
 * integers based on a cyclic partitioning to produce an array of part
 * numbers assigned to each element in the input array.
 *
 * The array to be partitioned, ‘idx’, contains ‘idxsize’ items.
 * Moreover, the user must provide an output array, ‘dstpart’, of size
 * ‘idxsize’, which is used to write the part number assigned to each
 * element of the input array.
 *
 * The set to be partitioned consists of ‘size’ items, which are
 * partitioned in a cyclic fashion into ‘num_parts’ parts.
 *
 * If ‘dstpartsizes’ is not ‘NULL’, then it must be an array of length
 * ‘num_parts’, which is used to store the number of items assigned to
 * each part.
 *
 * Returns ‘0’ if successful, or ‘EINVAL’ if ‘num_parts’ is not a
 * positive integer or if any element in ‘idx’ lies outside of the
 * valid range ‘[0,size)’.
 */
int partition_cyclic_int(
    int size,
    int num_parts,
    int64_t idxsize,
    int idxstride,
    const int * idx,
    int * dstpart,
    int64_t * dstpartsizes)
{
    if (num_parts <= 0) return EINVAL;
    if (dstpartsizes) { for (int p = 0; p < num_parts; p++) dstpartsizes[p] = 0; }
    for (int64_t i = 0; i < idxsize; i++) {
        int x = *(const int *) ((const unsigned char *) idx+i*idxstride);
        if (x < 0 || x >= size) return EINVAL;
        if (dstpart) dstpart[i] = x % num_parts;
        if (dstpartsizes) dstpartsizes[x % num_parts]++;
    }
    return 0;
}

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
 * size of each block of the partitioned set. The sum of the entries
 * in ‘partsizes’ should therefore be equal to ‘size’.
 *
 * If ‘dstpartsizes’ is not ‘NULL’, then it must be an array of length
 * ‘num_parts’, which is used to store the number of items assigned to
 * each part.
 *
 * Returns ‘0’ if successful, or ‘EINVAL’ if ‘num_parts’ is not a
 * positive integer or if any element in ‘idx’ lies outside of the
 * valid range ‘[0,size)’.
 */
int partition_block_int64(
    int64_t size,
    int num_parts,
    const int64_t * partsizes,
    int64_t idxsize,
    int idxstride,
    const int64_t * idx,
    int * dstpart,
    int64_t * dstpartsizes)
{
    if (dstpartsizes) { for (int p = 0; p < num_parts; p++) dstpartsizes[p] = 0; }
    if (partsizes) {
        for (int64_t i = 0; i < idxsize; i++) {
            int64_t x = *(const int64_t *) ((const unsigned char *) idx+i*idxstride);
            if (x < 0 || x >= size) return EINVAL;
            int p = 0;
            while (p < num_parts && x >= partsizes[p])
                x -= partsizes[p++];
            if (p >= num_parts) return EINVAL;
            if (dstpart) dstpart[i] = p;
            if (dstpartsizes) dstpartsizes[p]++;
        }
    } else {
        if (num_parts <= 0) return EINVAL;
        int64_t blksize = (size+num_parts-1) / num_parts;
        if (blksize <= 0) return EINVAL;
        for (int64_t i = 0; i < idxsize; i++) {
            int64_t x = *(const int64_t *) ((const unsigned char *) idx+i*idxstride);
            if (x < 0 || x >= size) return EINVAL;
            if (dstpart) dstpart[i] = x / blksize;
            if (dstpartsizes) dstpartsizes[x / blksize]++;
        }
    }
    return 0;
}

/**
 * ‘partition_block_int()’ partitions elements of a set of signed
 * integers based on a block partitioning to produce an array of part
 * numbers assigned to each element in the input array.
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
 *
 * If ‘dstpartsizes’ is not ‘NULL’, then it must be an array of length
 * ‘num_parts’, which is used to store the number of items assigned to
 * each part.
 *
 * Returns ‘0’ if successful, or ‘EINVAL’ if ‘num_parts’ is not a
 * positive integer or if any element in ‘idx’ lies outside of the
 * valid range ‘[0,size)’.
 */
int partition_block_int(
    int size,
    int num_parts,
    const int * partsizes,
    int64_t idxsize,
    int idxstride,
    const int * idx,
    int * dstpart,
    int64_t * dstpartsizes)
{
    if (dstpartsizes) { for (int p = 0; p < num_parts; p++) dstpartsizes[p] = 0; }
    if (partsizes) {
        for (int64_t i = 0; i < idxsize; i++) {
            int x = *(const int *) ((const unsigned char *) idx+i*idxstride);
            if (x < 0 || x >= size) return EINVAL;
            int p = 0;
            while (p < num_parts && x >= partsizes[p])
                x -= partsizes[p++];
            if (p >= num_parts) return EINVAL;
            if (dstpart) dstpart[i] = p;
            if (dstpartsizes) dstpartsizes[p]++;
        }
    } else {
        if (num_parts <= 0) return EINVAL;
        int blksize = (size+num_parts-1) / num_parts;
        if (blksize <= 0) return EINVAL;
        for (int64_t i = 0; i < idxsize; i++) {
            int x = *(const int *) ((const unsigned char *) idx+i*idxstride);
            if (x < 0 || x >= size) return EINVAL;
            if (dstpart) dstpart[i] = x / blksize;
            if (dstpartsizes) dstpartsizes[x / blksize]++;
        }
    }
    return 0;
}

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
 *
 * If ‘dstpartsizes’ is not ‘NULL’, then it must be an array of length
 * ‘num_parts’, which is used to store the number of items assigned to
 * each part.
 *
 * Returns ‘0’ if successful, or ‘EINVAL’ if ‘num_parts’ is not a
 * positive integer or if any element in ‘idx’ lies outside of the
 * valid range ‘[0,size)’.
 */
int partition_block_cyclic_int64(
    int64_t size,
    int num_parts,
    int64_t blksize,
    int64_t idxsize,
    int idxstride,
    const int64_t * idx,
    int * dstpart,
    int64_t * dstpartsizes)
{
    if (num_parts <= 0 || blksize <= 0) return EINVAL;
    if (dstpartsizes) { for (int p = 0; p < num_parts; p++) dstpartsizes[p] = 0; }
    for (int64_t i = 0; i < idxsize; i++) {
        int64_t x = *(const int64_t *) ((const unsigned char *) idx+i*idxstride);
        if (x < 0 || x >= size) return EINVAL;
        if (dstpart) dstpart[i] = (x / blksize) % num_parts;
        if (dstpartsizes) dstpartsizes[(x / blksize) % num_parts]++;
    }
    return 0;
}

/**
 * ‘partition_block_cyclic_int()’ partitions elements of a set of
 * signed integers based on a block-cyclic partitioning to produce an
 * array of part numbers assigned to each element in the input array.
 *
 * The array to be partitioned, ‘idx’, contains ‘idxsize’ items.
 * Moreover, the user must provide an output array, ‘dstpart’, of size
 * ‘idxsize’, which is used to write the part number assigned to each
 * element of the input array.
 *
 * The set to be partitioned consists of ‘size’ items arranged in
 * contiguous block of size ‘blksize’, which are then partitioned in a
 * cyclic fashion into ‘num_parts’ parts.
 *
 * If ‘dstpartsizes’ is not ‘NULL’, then it must be an array of length
 * ‘num_parts’, which is used to store the number of items assigned to
 * each part.
 *
 * Returns ‘0’ if successful, or ‘EINVAL’ if ‘num_parts’ is not a
 * positive integer or if any element in ‘idx’ lies outside of the
 * valid range ‘[0,size)’.
 */
int partition_block_cyclic_int(
    int size,
    int num_parts,
    int blksize,
    int64_t idxsize,
    int idxstride,
    const int * idx,
    int * dstpart,
    int64_t * dstpartsizes)
{
    if (num_parts <= 0 || blksize <= 0) return EINVAL;
    if (dstpartsizes) { for (int p = 0; p < num_parts; p++) dstpartsizes[p] = 0; }
    for (int64_t i = 0; i < idxsize; i++) {
        int x = *(const int *) ((const unsigned char *) idx+i*idxstride);
        if (x < 0 || x >= size) return EINVAL;
        if (dstpart) dstpart[i] = (x / blksize) % num_parts;
        if (dstpartsizes) dstpartsizes[(x / blksize) % num_parts]++;
    }
    return 0;
}

/**
 * ‘partition_custom_int64()’ partitions elements of a set of 64-bit
 * signed integers based on a custom, user-defined partitioning to
 * produce an array of part numbers assigned to each element in the
 * input array.
 *
 * The array to be partitioned, ‘idx’, contains ‘idxsize’ items.
 * Moreover, the user must provide an output array, ‘dstpart’, of size
 * ‘idxsize’, which is used to write the part number assigned to each
 * element of the input array.
 *
 * The set to be partitioned consists of ‘size’ items. Moreover, the
 * array ‘parts’ must be of length ‘size’, and it is used to specify
 * the part number for each element ‘0,1,...,size-1’. The values must
 * therefore be in the range ‘[0,num_parts)’.
 *
 * If ‘dstpartsizes’ is not ‘NULL’, then it must be an array of length
 * ‘num_parts’, which is used to store the number of items assigned to
 * each part.
 *
 * Returns ‘0’ if successful, or ‘EINVAL’ if ‘num_parts’ is not a
 * positive integer or if any element in ‘idx’ lies outside of the
 * valid range ‘[0,size)’.
 */
int partition_custom_int64(
    int64_t size,
    int num_parts,
    const int * parts,
    int64_t idxsize,
    int idxstride,
    const int64_t * idx,
    int * dstpart,
    int64_t * dstpartsizes)
{
    if (num_parts <= 0) return EINVAL;
    if (dstpartsizes) { for (int p = 0; p < num_parts; p++) dstpartsizes[p] = 0; }
    for (int64_t i = 0; i < idxsize; i++) {
        int64_t x = *(const int64_t *) ((const unsigned char *) idx+i*idxstride);
        if (x < 0 || x >= size) return EINVAL;
        if (dstpart) dstpart[i] = parts[x];
        if (dstpartsizes) dstpartsizes[parts[x]]++;
    }
    return 0;
}

/**
 * ‘partition_custom_int()’ partitions elements of a set of signed
 * integers based on a custom, user-defined partitioning to produce an
 * array of part numbers assigned to each element in the input array.
 *
 * The array to be partitioned, ‘idx’, contains ‘idxsize’ items.
 * Moreover, the user must provide an output array, ‘dstpart’, of size
 * ‘idxsize’, which is used to write the part number assigned to each
 * element of the input array.
 *
 * The set to be partitioned consists of ‘size’ items. Moreover, the
 * array ‘parts’ must be of length ‘size’, and it is used to specify
 * the part number for each element ‘0,1,...,size-1’. The values must
 * therefore be in the range ‘[0,num_parts)’.
 *
 * If ‘dstpartsizes’ is not ‘NULL’, then it must be an array of length
 * ‘num_parts’, which is used to store the number of items assigned to
 * each part.
 *
 * Returns ‘0’ if successful, or ‘EINVAL’ if ‘num_parts’ is not a
 * positive integer or if any element in ‘idx’ lies outside of the
 * valid range ‘[0,size)’.
 */
int partition_custom_int(
    int size,
    int num_parts,
    const int * parts,
    int64_t idxsize,
    int idxstride,
    const int * idx,
    int * dstpart,
    int64_t * dstpartsizes)
{
    if (num_parts <= 0) return EINVAL;
    if (dstpartsizes) { for (int p = 0; p < num_parts; p++) dstpartsizes[p] = 0; }
    for (int64_t i = 0; i < idxsize; i++) {
        int x = *(const int *) ((const unsigned char *) idx+i*idxstride);
        if (x < 0 || x >= size) return EINVAL;
        if (dstpart) dstpart[i] = parts[x];
        if (dstpartsizes) dstpartsizes[parts[x]]++;
    }
    return 0;
}

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
 * If ‘dstpartsizes’ is not ‘NULL’, then it must be an array of length
 * ‘num_parts’, which is used to store the number of items assigned to
 * each part.
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
 *
 * - If ‘type’ is ‘mtx_custom_partition’, then the array ‘parts’ must
 *   be of length ‘size’ and should contain the part number (i.e., an
 *   integer in the range ‘[0,num_parts)’) for each element.
 *
 * Returns ‘0’ if successful, or ‘EINVAL’ if ‘type’ is not a valid
 * partitioning type, ‘num_parts’ is not a positive integer or if any
 * element in ‘idx’ lies outside of the valid range ‘[0,size)’.
 */
int partition_int64(
    enum mtxpartitioning type,
    int64_t size,
    int num_parts,
    const int64_t * partsizes,
    int64_t blksize,
    const int * parts,
    int64_t idxsize,
    int idxstride,
    const int64_t * idx,
    int * dstpart,
    int64_t * dstpartsizes)
{
    if (type == mtx_cyclic) {
        return partition_cyclic_int64(
            size, num_parts, idxsize, idxstride, idx,
            dstpart, dstpartsizes);
    } else if (type == mtx_block) {
        return partition_block_int64(
            size, num_parts, partsizes, idxsize, idxstride, idx,
            dstpart, dstpartsizes);
    } else if (type == mtx_block_cyclic) {
        return partition_block_cyclic_int64(
            size, num_parts, blksize, idxsize, idxstride, idx,
            dstpart, dstpartsizes);
    } else if (type == mtx_custom_partition) {
        return partition_custom_int64(
            size, num_parts, parts, idxsize, idxstride, idx,
            dstpart, dstpartsizes);
    } else { return EINVAL; }
}

/**
 * ‘partition_int()’ partitions elements of a set of signed integers
 * based on a given partitioning to produce an array of part numbers
 * assigned to each element in the input array.
 *
 * The array to be partitioned, ‘idx’, contains ‘idxsize’ items.
 * Moreover, the user must provide an output array, ‘dstpart’, of size
 * ‘idxsize’, which is used to write the part number assigned to each
 * element of the input array.
 *
 * If ‘dstpartsizes’ is not ‘NULL’, then it must be an array of length
 * ‘num_parts’, which is used to store the number of items assigned to
 * each part.
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
 *
 * - If ‘type’ is ‘mtx_custom_partition’, then the array ‘parts’ must
 *   be of length ‘size’ and should contain the part number (i.e., an
 *   integer in the range ‘[0,num_parts)’) for each element.
 *
 * Returns ‘0’ if successful, or ‘EINVAL’ if ‘type’ is not a valid
 * partitioning type, ‘num_parts’ is not a positive integer or if any
 * element in ‘idx’ lies outside of the valid range ‘[0,size)’.
 */
int partition_int(
    enum mtxpartitioning type,
    int size,
    int num_parts,
    const int * partsizes,
    int blksize,
    const int * parts,
    int64_t idxsize,
    int idxstride,
    const int * idx,
    int * dstpart,
    int64_t * dstpartsizes)
{
    if (type == mtx_cyclic) {
        return partition_cyclic_int(
            size, num_parts, idxsize, idxstride, idx,
            dstpart, dstpartsizes);
    } else if (type == mtx_block) {
        return partition_block_int(
            size, num_parts, partsizes, idxsize, idxstride, idx,
            dstpart, dstpartsizes);
    } else if (type == mtx_block_cyclic) {
        return partition_block_cyclic_int(
            size, num_parts, blksize, idxsize, idxstride, idx,
            dstpart, dstpartsizes);
    } else if (type == mtx_custom_partition) {
        return partition_custom_int(
            size, num_parts, parts, idxsize, idxstride, idx,
            dstpart, dstpartsizes);
    } else { return EINVAL; }
}
