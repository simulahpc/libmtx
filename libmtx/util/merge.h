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
 * Last modified: 2022-04-26
 *
 * Merge operations on arrays.
 */

#ifndef LIBMTX_UTIL_MERGE_H
#define LIBMTX_UTIL_MERGE_H

#include <libmtx/libmtx-config.h>

#include <stdint.h>

/*
 * stream compaction
 */

/**
 * ‘compact_sorted_int32()’ performs a stream compaction on a sorted
 * array of 32-bit signed integers to produce a sorted output array of
 * unique elements from the input array.
 *
 * The array to be compacted, ‘a’, contains ‘asize’ items. The array
 * must be sorted in ascending order and values may appear more than
 * once.
 *
 * If ‘b’ is ‘NULL’, then no output is written. However, ‘bsize’ is
 * used to indicate the number of items that would have been written
 * if an output array were provided.
 *
 * Otherwise, the user must provide an output array ‘b’ with enough
 * storage to hold the compacted results. Moreover, the user must
 * specify the allocated size of the output array with the value
 * pointed to by ‘bsize’. On success, the value returned in ‘bsize’
 * indicates the number of items that were written to the output
 * array.
 *
 * If ‘dstidx’ is not ‘NULL’, then it must point to an array of length
 * ‘asize’. For a given item in the input array ‘a’ the corresponding
 * value in ‘dstidx’ indicates the offset to the same item in the
 * output array ‘b’.
 */
int compact_sorted_int32(
    int64_t * bsize,
    int32_t * b,
    int64_t asize,
    const int32_t * a,
    int64_t * dstidx);

/**
 * ‘compact_sorted_int64()’ performs a stream compaction on a sorted
 * array of 64-bit signed integers to produce a sorted output array of
 * unique elements from the input array.
 *
 * The array to be compacted, ‘a’, contains ‘asize’ items. The array
 * must be sorted in ascending order and values may appear more than
 * once.
 *
 * If ‘b’ is ‘NULL’, then no output is written. However, ‘bsize’ is
 * used to indicate the number of items that would have been written
 * if an output array were provided.
 *
 * Otherwise, the user must provide an output array ‘b’ with enough
 * storage to hold the compacted results. Moreover, the user must
 * specify the allocated size of the output array with the value
 * pointed to by ‘bsize’. On success, the value returned in ‘bsize’
 * indicates the number of items that were written to the output
 * array.
 *
 * If ‘dstidx’ is not ‘NULL’, then it must point to an array of length
 * ‘asize’. For a given item in the input array ‘a’ the corresponding
 * value in ‘dstidx’ indicates the offset to the same item in the
 * output array ‘b’.
 */
int compact_sorted_int64(
    int64_t * bsize,
    int64_t * b,
    int64_t asize,
    const int64_t * a,
    int64_t * dstidx);

/**
 * ‘compact_sorted_int()’ performs a stream compaction on a sorted
 * array of signed integers to produce a sorted output array of unique
 * elements from the input array.
 *
 * The array to be compacted, ‘a’, contains ‘asize’ items. The array
 * must be sorted in ascending order and values may appear more than
 * once.
 *
 * If ‘b’ is ‘NULL’, then no output is written. However, ‘bsize’ is
 * used to indicate the number of items that would have been written
 * if an output array were provided.
 *
 * Otherwise, the user must provide an output array ‘b’ with enough
 * storage to hold the compacted results. Moreover, the user must
 * specify the allocated size of the output array with the value
 * pointed to by ‘bsize’. On success, the value returned in ‘bsize’
 * indicates the number of items that were written to the output
 * array.
 *
 * If ‘dstidx’ is not ‘NULL’, then it must point to an array of length
 * ‘asize’. For a given item in the input array ‘a’ the corresponding
 * value in ‘dstidx’ indicates the offset to the same item in the
 * output array ‘b’.
 */
int compact_sorted_int(
    int64_t * bsize,
    int * b,
    int64_t asize,
    const int * a,
    int64_t * dstidx);

/**
 * ‘compact_sorted_int32_pair()’ performs a stream compaction on a
 * sorted array of pairs of 32-bit signed integers to produce a sorted
 * output array of unique elements from the input array.
 *
 * The array to be compacted, ‘a’, contains ‘asize’ items. The array
 * must be sorted in ascending, lexicographic order and values may
 * appear more than once.
 *
 * If ‘b’ is ‘NULL’, then no output is written. However, ‘bsize’ is
 * used to indicate the number of items (pairs) that would have been
 * written if an output array were provided.
 *
 * Otherwise, the user must provide an output array ‘b’ with enough
 * storage to hold the compacted results. Moreover, the user must
 * specify the allocated size of the output array with the value
 * pointed to by ‘bsize’. On success, the value returned in ‘bsize’
 * indicates the number of items that were written to the output
 * array.
 *
 * If ‘dstidx’ is not ‘NULL’, then it must point to an array of length
 * ‘asize’. For a given item (pair) in the input array ‘a’ the
 * corresponding value in ‘dstidx’ indicates the offset to the same
 * item (pair) in the output array ‘b’.
 */
int compact_sorted_int32_pair(
    int64_t * bsize,
    int32_t (* b)[2],
    int64_t asize,
    const int32_t (* a)[2],
    int64_t * dstidx);

/**
 * ‘compact_sorted_int64_pair()’ performs a stream compaction on a
 * sorted array of pairs of 64-bit signed integers to produce a sorted
 * output array of unique elements from the input array.
 *
 * The array to be compacted, ‘a’, contains ‘asize’ items. The array
 * must be sorted in ascending, lexicographic order and values may
 * appear more than once.
 *
 * If ‘b’ is ‘NULL’, then no output is written. However, ‘bsize’ is
 * used to indicate the number of items (pairs) that would have been
 * written if an output array were provided.
 *
 * Otherwise, the user must provide an output array ‘b’ with enough
 * storage to hold the compacted results. Moreover, the user must
 * specify the allocated size of the output array with the value
 * pointed to by ‘bsize’. On success, the value returned in ‘bsize’
 * indicates the number of items that were written to the output
 * array.
 *
 * If ‘dstidx’ is not ‘NULL’, then it must point to an array of length
 * ‘asize’. For a given item (pair) in the input array ‘a’ the
 * corresponding value in ‘dstidx’ indicates the offset to the same
 * item (pair) in the output array ‘b’.
 */
int compact_sorted_int64_pair(
    int64_t * bsize,
    int64_t (* b)[2],
    int64_t asize,
    const int64_t (* a)[2],
    int64_t * dstidx);


/**
 * ‘compact_sorted_int_pair()’ performs a stream compaction on a
 * sorted array of pairs of signed integers to produce a sorted output
 * array of unique elements from the input array.
 *
 * The array to be compacted, ‘a’, contains ‘asize’ items. The array
 * must be sorted in ascending, lexicographic order and values may
 * appear more than once.
 *
 * If ‘b’ is ‘NULL’, then no output is written. However, ‘bsize’ is
 * used to indicate the number of items (pairs) that would have been
 * written if an output array were provided.
 *
 * Otherwise, the user must provide an output array ‘b’ with enough
 * storage to hold the compacted results. Moreover, the user must
 * specify the allocated size of the output array with the value
 * pointed to by ‘bsize’. On success, the value returned in ‘bsize’
 * indicates the number of items that were written to the output
 * array.
 *
 * If ‘dstidx’ is not ‘NULL’, then it must point to an array of length
 * ‘asize’. For a given item (pair) in the input array ‘a’ the
 * corresponding value in ‘dstidx’ indicates the offset to the same
 * item (pair) in the output array ‘b’.
 */
int compact_sorted_int_pair(
    int64_t * bsize,
    int (* b)[2],
    int64_t asize,
    const int (* a)[2],
    int64_t * dstidx);

/**
 * ‘compact_unsorted_int32()’ performs a stream compaction on a sorted
 * array of 32-bit signed integers to produce a sorted output array of
 * unique elements from the input array.
 *
 * The array to be compacted, ‘a’, contains ‘asize’ items. The arrays
 * need not be sorted beforehand, but it will be sorted if the
 * function returns successfully. Duplicate values are allowed in the
 * input array.
 *
 * If ‘b’ is ‘NULL’, then no output is written. However, ‘bsize’ is
 * used to indicate the number of items that would have been written
 * if an output array were provided.
 *
 * Otherwise, the user must provide an output array ‘b’ with enough
 * storage to hold the compacted results. (The input and output arrays
 * may be the same.) Moreover, the user must specify the allocated
 * size of the output array with the value pointed to by ‘bsize’. On
 * success, the value returned in ‘bsize’ indicates the number of
 * items that were written to the output array. The output will be
 * sorted in ascending order.
 *
 * If ‘dstidx’ is not ‘NULL’, then it must point to an array of length
 * ‘asize’. For a given item in the input array ‘a’ the corresponding
 * value in ‘dstidx’ indicates the offset to the same item in the
 * output array ‘b’.
 */
int compact_unsorted_int32(
    int64_t * bsize,
    int32_t * b,
    int64_t asize,
    int32_t * a,
    int64_t * perm,
    int64_t * dstidx);

/**
 * ‘compact_unsorted_int64()’ performs a stream compaction on a sorted
 * array of 64-bit signed integers to produce a sorted output array of
 * unique elements from the input array.
 *
 * The array to be compacted, ‘a’, contains ‘asize’ items. The arrays
 * need not be sorted beforehand, but it will be sorted if the
 * function returns successfully. Duplicate values are allowed in the
 * input array.
 *
 * If ‘b’ is ‘NULL’, then no output is written. However, ‘bsize’ is
 * used to indicate the number of items that would have been written
 * if an output array were provided.
 *
 * Otherwise, the user must provide an output array ‘b’ with enough
 * storage to hold the compacted results. (The input and output arrays
 * may be the same.) Moreover, the user must specify the allocated
 * size of the output array with the value pointed to by ‘bsize’. On
 * success, the value returned in ‘bsize’ indicates the number of
 * items that were written to the output array. The output will be
 * sorted in ascending order.
 *
 * If ‘dstidx’ is not ‘NULL’, then it must point to an array of length
 * ‘asize’. For a given item in the input array ‘a’ the corresponding
 * value in ‘dstidx’ indicates the offset to the same item in the
 * output array ‘b’.
 */
int compact_unsorted_int64(
    int64_t * bsize,
    int64_t * b,
    int64_t asize,
    int64_t * a,
    int64_t * perm,
    int64_t * dstidx);

/**
 * ‘compact_unsorted_int()’ performs a stream compaction on a sorted
 * array of signed integers to produce a sorted output array of
 * unique elements from the input array.
 *
 * The array to be compacted, ‘a’, contains ‘asize’ items. The arrays
 * need not be sorted beforehand, but it will be sorted if the
 * function returns successfully. Duplicate values are allowed in the
 * input array.
 *
 * If ‘b’ is ‘NULL’, then no output is written. However, ‘bsize’ is
 * used to indicate the number of items that would have been written
 * if an output array were provided.
 *
 * Otherwise, the user must provide an output array ‘b’ with enough
 * storage to hold the compacted results. (The input and output arrays
 * may be the same.) Moreover, the user must specify the allocated
 * size of the output array with the value pointed to by ‘bsize’. On
 * success, the value returned in ‘bsize’ indicates the number of
 * items that were written to the output array. The output will be
 * sorted in ascending order.
 *
 * If ‘dstidx’ is not ‘NULL’, then it must point to an array of length
 * ‘asize’. For a given item in the input array ‘a’ the corresponding
 * value in ‘dstidx’ indicates the offset to the same item in the
 * output array ‘b’.
 */
int compact_unsorted_int(
    int64_t * bsize,
    int * b,
    int64_t asize,
    int * a,
    int64_t * perm,
    int64_t * dstidx);

/**
 * ‘compact_unsorted_int32_pair()’ performs a stream compaction on a
 * sorted array of pairs of 32-bit signed integers to produce a sorted
 * output array of unique elements from the input array.
 *
 * The array to be compacted, ‘a’, contains ‘asize’ items (pairs). The
 * arrays need not be sorted beforehand, but it will be sorted if the
 * function returns successfully. Duplicate values are allowed in the
 * input array.
 *
 * If ‘b’ is ‘NULL’, then no output is written. However, ‘bsize’ is
 * used to indicate the number of items (pairs) that would have been
 * written if an output array were provided.
 *
 * Otherwise, the user must provide an output array ‘b’ with enough
 * storage to hold the compacted results. (The input and output arrays
 * may be the same.) Moreover, the user must specify the allocated
 * size of the output array with the value pointed to by ‘bsize’. On
 * success, the value returned in ‘bsize’ indicates the number of
 * items (pairs) that were written to the output array. The output
 * will be sorted in ascending, lexicographic order.
 *
 * If ‘dstidx’ is not ‘NULL’, then it must point to an array of length
 * ‘asize’. For a given item (pair) in the input array ‘a’ the
 * corresponding value in ‘dstidx’ indicates the offset to the same
 * item (pair) in the output array ‘b’.
 */
int compact_unsorted_int32_pair(
    int64_t * bsize,
    int32_t (* b)[2],
    int64_t asize,
    int32_t (* a)[2],
    int64_t * perm,
    int64_t * dstidx);

/**
 * ‘compact_unsorted_int32_pair()’ performs a stream compaction on a
 * sorted array of pairs of 32-bit signed integers to produce a sorted
 * output array of unique elements from the input array.
 *
 * The array to be compacted, ‘a’, contains ‘asize’ items (pairs). The
 * arrays need not be sorted beforehand, but it will be sorted if the
 * function returns successfully. Duplicate values are allowed in the
 * input array.
 *
 * If ‘b’ is ‘NULL’, then no output is written. However, ‘bsize’ is
 * used to indicate the number of items (pairs) that would have been
 * written if an output array were provided.
 *
 * Otherwise, the user must provide an output array ‘b’ with enough
 * storage to hold the compacted results. (The input and output arrays
 * may be the same.) Moreover, the user must specify the allocated
 * size of the output array with the value pointed to by ‘bsize’. On
 * success, the value returned in ‘bsize’ indicates the number of
 * items (pairs) that were written to the output array. The output
 * will be sorted in ascending, lexicographic order.
 *
 * If ‘dstidx’ is not ‘NULL’, then it must point to an array of length
 * ‘asize’. For a given item (pair) in the input array ‘a’ the
 * corresponding value in ‘dstidx’ indicates the offset to the same
 * item (pair) in the output array ‘b’.
 */
int compact_unsorted_int32_pair(
    int64_t * bsize,
    int32_t (* b)[2],
    int64_t asize,
    int32_t (* a)[2],
    int64_t * perm,
    int64_t * dstidx);

/**
 * ‘compact_unsorted_int64_pair()’ performs a stream compaction on a
 * sorted array of pairs of 64-bit signed integers to produce a sorted
 * output array of unique elements from the input array.
 *
 * The array to be compacted, ‘a’, contains ‘asize’ items (pairs). The
 * arrays need not be sorted beforehand, but it will be sorted if the
 * function returns successfully. Duplicate values are allowed in the
 * input array.
 *
 * If ‘b’ is ‘NULL’, then no output is written. However, ‘bsize’ is
 * used to indicate the number of items (pairs) that would have been
 * written if an output array were provided.
 *
 * Otherwise, the user must provide an output array ‘b’ with enough
 * storage to hold the compacted results. (The input and output arrays
 * may be the same.) Moreover, the user must specify the allocated
 * size of the output array with the value pointed to by ‘bsize’. On
 * success, the value returned in ‘bsize’ indicates the number of
 * items (pairs) that were written to the output array. The output
 * will be sorted in ascending, lexicographic order.
 *
 * If ‘dstidx’ is not ‘NULL’, then it must point to an array of length
 * ‘asize’. For a given item (pair) in the input array ‘a’ the
 * corresponding value in ‘dstidx’ indicates the offset to the same
 * item (pair) in the output array ‘b’.
 */
int compact_unsorted_int64_pair(
    int64_t * bsize,
    int64_t (* b)[2],
    int64_t asize,
    int64_t (* a)[2],
    int64_t * perm,
    int64_t * dstidx);

/**
 * ‘compact_unsorted_int_pair()’ performs a stream compaction on a
 * sorted array of pairs of signed integers to produce a sorted output
 * array of unique elements from the input array.
 *
 * The array to be compacted, ‘a’, contains ‘asize’ items (pairs). The
 * arrays need not be sorted beforehand, but it will be sorted if the
 * function returns successfully. Duplicate values are allowed in the
 * input array.
 *
 * If ‘b’ is ‘NULL’, then no output is written. However, ‘bsize’ is
 * used to indicate the number of items (pairs) that would have been
 * written if an output array were provided.
 *
 * Otherwise, the user must provide an output array ‘b’ with enough
 * storage to hold the compacted results. (The input and output arrays
 * may be the same.) Moreover, the user must specify the allocated
 * size of the output array with the value pointed to by ‘bsize’. On
 * success, the value returned in ‘bsize’ indicates the number of
 * items (pairs) that were written to the output array. The output
 * will be sorted in ascending, lexicographic order.
 *
 * If ‘dstidx’ is not ‘NULL’, then it must point to an array of length
 * ‘asize’. For a given item (pair) in the input array ‘a’ the
 * corresponding value in ‘dstidx’ indicates the offset to the same
 * item (pair) in the output array ‘b’.
 */
int compact_unsorted_int_pair(
    int64_t * bsize,
    int (* b)[2],
    int64_t asize,
    int (* a)[2],
    int64_t * perm,
    int64_t * dstidx);

/*
 * merge sorted arrays
 */

/**
 * ‘merge_sorted_int32()’ merges two sorted arrays of 32-bit signed
 * integers to produce a sorted output array containing the elements
 * of both input arrays.
 *
 * The two arrays to be merged, ‘a’ and ‘b’, contain ‘asize’ and
 * ‘bsize’ items, respectively. The arrays must be sorted in ascending
 * order and values may appear more than once.
 *
 * The output array ‘c’ must have enough storage to hold the merged
 * results, which consists of the items in the input arrays.
 */
int merge_sorted_int32(
    int64_t csize,
    int32_t * c,
    int64_t asize,
    const int32_t * a,
    int64_t bsize,
    const int32_t * b);

/**
 * ‘merge_sorted_int64()’ merges two sorted arrays of 64-bit signed
 * integers to produce a sorted output array containing the elements
 * of both input arrays.
 *
 * The two arrays to be merged, ‘a’ and ‘b’, contain ‘asize’ and
 * ‘bsize’ items, respectively. The arrays must be sorted in ascending
 * order and values may appear more than once.
 *
 * The output array ‘c’ must have enough storage to hold the merged
 * results, which consists of the items in the input arrays.
 */
int merge_sorted_int64(
    int64_t csize,
    int64_t * c,
    int64_t asize,
    const int64_t * a,
    int64_t bsize,
    const int64_t * b);

/**
 * ‘merge_sorted_int()’ merges two sorted arrays of signed integers to
 * produce a sorted output array containing the elements of both input
 * arrays.
 *
 * The two arrays to be merged, ‘a’ and ‘b’, contain ‘asize’ and
 * ‘bsize’ items, respectively. The arrays must be sorted in ascending
 * order and values may appear more than once.
 *
 * The output array ‘c’ must have enough storage to hold the merged
 * results, which consists of the items in the input arrays.
 */
int merge_sorted_int(
    int64_t csize,
    int * c,
    int64_t asize,
    const int * a,
    int64_t bsize,
    const int * b);

/*
 * set union operations on sorted arrays of unique values (i.e., no
 * duplicates).
 */

/**
 * ‘setunion_sorted_unique_int32()’ merges two sorted arrays of 32-bit
 * signed integers based on a set union operation.
 *
 * The two arrays to be merged, ‘a’ and ‘b’, contain ‘asize’ and
 * ‘bsize’ items, respectively. Each array must be sorted in ascending
 * order and duplicate values are not allowed.
 *
 * If ‘c’ is ‘NULL’, then no output is written. However, ‘csize’ is
 * used to indicate the number of items that would have been written
 * if an output array were provided.
 *
 * Otherwise, the user must provide an output array ‘c’ with enough
 * storage to hold the merged results. Moreover, the user must specify
 * the allocated size of the output array with the value pointed to by
 * ‘csize’. On success, the value returned in ‘csize’ indicates the
 * number of items that were written to the output array.
 */
int setunion_sorted_unique_int32(
    int64_t * csize,
    int32_t * c,
    int64_t asize,
    const int32_t * a,
    int64_t bsize,
    const int32_t * b);

/**
 * ‘setunion_sorted_unique_int64()’ merges two sorted arrays of 64-bit
 * signed integers based on a set union operation.
 *
 * The two arrays to be merged, ‘a’ and ‘b’, contain ‘asize’ and
 * ‘bsize’ items, respectively. Each array must be sorted in ascending
 * order and duplicate values are not allowed.
 *
 * If ‘c’ is ‘NULL’, then no output is written. However, ‘csize’ is
 * used to indicate the number of items that would have been written
 * if an output array were provided.
 *
 * Otherwise, the user must provide an output array ‘c’ with enough
 * storage to hold the merged results. Moreover, the user must specify
 * the allocated size of the output array with the value pointed to by
 * ‘csize’. On success, the value returned in ‘csize’ indicates the
 * number of items that were written to the output array.
 */
int setunion_sorted_unique_int64(
    int64_t * csize,
    int64_t * c,
    int64_t asize,
    const int64_t * a,
    int64_t bsize,
    const int64_t * b);

/**
 * ‘setunion_sorted_unique_int()’ merges two sorted arrays of signed
 * integers based on a set union operation.
 *
 * The two arrays to be merged, ‘a’ and ‘b’, contain ‘asize’ and
 * ‘bsize’ items, respectively. Each array must be sorted in ascending
 * order and duplicate values are not allowed.
 *
 * If ‘c’ is ‘NULL’, then no output is written. However, ‘csize’ is
 * used to indicate the number of items that would have been written
 * if an output array were provided.
 *
 * Otherwise, the user must provide an output array ‘c’ with enough
 * storage to hold the merged results. Moreover, the user must specify
 * the allocated size of the output array with the value pointed to by
 * ‘csize’. On success, the value returned in ‘csize’ indicates the
 * number of items that were written to the output array.
 */
int setunion_sorted_unique_int(
    int64_t * csize,
    int * c,
    int64_t asize,
    const int * a,
    int64_t bsize,
    const int * b);

/*
 * set union on sorted arrays, possibly containing non-unique values
 * (i.e., duplicates are allowed).
 */

/**
 * ‘setunion_sorted_nonunique_int32()’ merges two sorted arrays of
 * 32-bit signed integers based on a set union operation.
 *
 * The two arrays to be merged, ‘a’ and ‘b’, contain ‘asize’ and
 * ‘bsize’ items, respectively. Each array must be sorted in ascending
 * order and duplicate values are allowed.
 *
 * If ‘c’ is ‘NULL’, then no output is written. However, ‘csize’ is
 * used to indicate the number of items that would have been written
 * if an output array were provided.
 *
 * Otherwise, the user must provide an output array ‘c’ with enough
 * storage to hold the merged results. Moreover, the user must specify
 * the allocated size of the output array with the value pointed to by
 * ‘csize’. On success, the value returned in ‘csize’ indicates the
 * number of items that were written to the output array.
 */
int setunion_sorted_nonunique_int32(
    int64_t * csize,
    int32_t * c,
    int64_t asize,
    const int32_t * a,
    int64_t bsize,
    const int32_t * b);

/**
 * ‘setunion_sorted_nonunique_int64()’ merges two sorted arrays of
 * 64-bit signed integers based on a set union operation.
 *
 * The two arrays to be merged, ‘a’ and ‘b’, contain ‘asize’ and
 * ‘bsize’ items, respectively. Each array must be sorted in ascending
 * order and duplicate values are allowed.
 *
 * If ‘c’ is ‘NULL’, then no output is written. However, ‘csize’ is
 * used to indicate the number of items that would have been written
 * if an output array were provided.
 *
 * Otherwise, the user must provide an output array ‘c’ with enough
 * storage to hold the merged results. Moreover, the user must specify
 * the allocated size of the output array with the value pointed to by
 * ‘csize’. On success, the value returned in ‘csize’ indicates the
 * number of items that were written to the output array.
 */
int setunion_sorted_nonunique_int64(
    int64_t * csize,
    int64_t * c,
    int64_t asize,
    const int64_t * a,
    int64_t bsize,
    const int64_t * b);

/**
 * ‘setunion_sorted_nonunique_int()’ merges two sorted arrays of
 * signed integers based on a set union operation.
 *
 * The two arrays to be merged, ‘a’ and ‘b’, contain ‘asize’ and
 * ‘bsize’ items, respectively. Each array must be sorted in ascending
 * order and duplicate values are allowed.
 *
 * If ‘c’ is ‘NULL’, then no output is written. However, ‘csize’ is
 * used to indicate the number of items that would have been written
 * if an output array were provided.
 *
 * Otherwise, the user must provide an output array ‘c’ with enough
 * storage to hold the merged results. Moreover, the user must specify
 * the allocated size of the output array with the value pointed to by
 * ‘csize’. On success, the value returned in ‘csize’ indicates the
 * number of items that were written to the output array.
 */
int setunion_sorted_nonunique_int(
    int64_t * csize,
    int * c,
    int64_t asize,
    const int * a,
    int64_t bsize,
    const int * b);

/*
 * set union operations on unsorted arrays of unique values (i.e., no
 * duplicates)
 */

/**
 * ‘setunion_unsorted_unique_int32()’ merges two arrays of 32-bit
 * signed integers based on a set union operation.
 *
 * The two arrays to be merged, ‘a’ and ‘b’, contain ‘asize’ and
 * ‘bsize’ items, respectively. The arrays need not be sorted
 * beforehand, but they will be sorted if the function returns
 * successfully. Duplicate values are not allowed in the input arrays.
 *
 * If ‘c’ is ‘NULL’, then no output is written. However, ‘csize’ is
 * used to indicate the number of items that would have been written
 * if an output array were provided.
 *
 * Otherwise, the user must provide an output array ‘c’ with enough
 * storage to hold the merged results. Moreover, the user must specify
 * the allocated size of the output array with the value pointed to by
 * ‘csize’. On success, the value returned in ‘csize’ indicates the
 * number of items that were written to the output array.
 */
int setunion_unsorted_unique_int32(
    int64_t * csize,
    int32_t * c,
    int64_t asize,
    int32_t * a,
    int64_t bsize,
    int32_t * b);

/**
 * ‘setunion_unsorted_unique_int64()’ merges two arrays of 64-bit
 * signed integers based on a set union operation.
 *
 * The two arrays to be merged, ‘a’ and ‘b’, contain ‘asize’ and
 * ‘bsize’ items, respectively. The arrays need not be sorted
 * beforehand, but they will be sorted if the function returns
 * successfully. Duplicate values are not allowed in the input arrays.
 *
 * If ‘c’ is ‘NULL’, then no output is written. However, ‘csize’ is
 * used to indicate the number of items that would have been written
 * if an output array were provided.
 *
 * Otherwise, the user must provide an output array ‘c’ with enough
 * storage to hold the merged results. Moreover, the user must specify
 * the allocated size of the output array with the value pointed to by
 * ‘csize’. On success, the value returned in ‘csize’ indicates the
 * number of items that were written to the output array.
 */
int setunion_unsorted_unique_int64(
    int64_t * csize,
    int64_t * c,
    int64_t asize,
    int64_t * a,
    int64_t bsize,
    int64_t * b);

/**
 * ‘setunion_unsorted_unique_int()’ merges two arrays of signed
 * integers based on a set union operation.
 *
 * The two arrays to be merged, ‘a’ and ‘b’, contain ‘asize’ and
 * ‘bsize’ items, respectively. The arrays need not be sorted
 * beforehand, but they will be sorted if the function returns
 * successfully. Duplicate values are not allowed in the input arrays.
 *
 * If ‘c’ is ‘NULL’, then no output is written. However, ‘csize’ is
 * used to indicate the number of items that would have been written
 * if an output array were provided.
 *
 * Otherwise, the user must provide an output array ‘c’ with enough
 * storage to hold the merged results. Moreover, the user must specify
 * the allocated size of the output array with the value pointed to by
 * ‘csize’. On success, the value returned in ‘csize’ indicates the
 * number of items that were written to the output array.
 */
int setunion_unsorted_unique_int(
    int64_t * csize,
    int * c,
    int64_t asize,
    int * a,
    int64_t bsize,
    int * b);

/*
 * set union on unsorted arrays, possibly containing non-unique values
 * (i.e., duplicates are allowed)
 */

/**
 * ‘setunion_unsorted_nonunique_int32()’ merges two arrays of 32-bit
 * signed integers based on a set union operation.
 *
 * The two arrays to be merged, ‘a’ and ‘b’, contain ‘asize’ and
 * ‘bsize’ items, respectively. The arrays need not be sorted
 * beforehand, but they will be sorted if the function returns
 * successfully. Duplicate values are allowed in the input arrays.
 *
 * If ‘c’ is ‘NULL’, then no output is written. However, ‘csize’ is
 * used to indicate the number of items that would have been written
 * if an output array were provided.
 *
 * Otherwise, the user must provide an output array ‘c’ with enough
 * storage to hold the merged results. Moreover, the user must specify
 * the allocated size of the output array with the value pointed to by
 * ‘csize’. On success, the value returned in ‘csize’ indicates the
 * number of items that were written to the output array.
 */
int setunion_unsorted_nonunique_int32(
    int64_t * csize,
    int32_t * c,
    int64_t asize,
    int32_t * a,
    int64_t bsize,
    int32_t * b);

/**
 * ‘setunion_unsorted_nonunique_int64()’ merges two arrays of 64-bit
 * signed integers based on a set union operation.
 *
 * The two arrays to be merged, ‘a’ and ‘b’, contain ‘asize’ and
 * ‘bsize’ items, respectively. The arrays need not be sorted
 * beforehand, but they will be sorted if the function returns
 * successfully. Duplicate values are allowed in the input arrays.
 *
 * If ‘c’ is ‘NULL’, then no output is written. However, ‘csize’ is
 * used to indicate the number of items that would have been written
 * if an output array were provided.
 *
 * Otherwise, the user must provide an output array ‘c’ with enough
 * storage to hold the merged results. Moreover, the user must specify
 * the allocated size of the output array with the value pointed to by
 * ‘csize’. On success, the value returned in ‘csize’ indicates the
 * number of items that were written to the output array.
 */
int setunion_unsorted_nonunique_int64(
    int64_t * csize,
    int64_t * c,
    int64_t asize,
    int64_t * a,
    int64_t bsize,
    int64_t * b);

/**
 * ‘setunion_unsorted_nonunique_int()’ merges two arrays of signed
 * integers based on a set union operation.
 *
 * The two arrays to be merged, ‘a’ and ‘b’, contain ‘asize’ and
 * ‘bsize’ items, respectively. The arrays need not be sorted
 * beforehand, but they will be sorted if the function returns
 * successfully. Duplicate values are allowed in the input arrays.
 *
 * If ‘c’ is ‘NULL’, then no output is written. However, ‘csize’ is
 * used to indicate the number of items that would have been written
 * if an output array were provided.
 *
 * Otherwise, the user must provide an output array ‘c’ with enough
 * storage to hold the merged results. Moreover, the user must specify
 * the allocated size of the output array with the value pointed to by
 * ‘csize’. On success, the value returned in ‘csize’ indicates the
 * number of items that were written to the output array.
 */
int setunion_unsorted_nonunique_int(
    int64_t * csize,
    int * c,
    int64_t asize,
    int * a,
    int64_t bsize,
    int * b);

/*
 * set intersection operations on sorted arrays of unique values
 * (i.e., no duplicates)
 */

/**
 * ‘setintersection_sorted_unique_int32()’ merges two sorted arrays of
 * 32-bit signed integers based on a set intersection operation.
 *
 * The two arrays to be merged, ‘a’ and ‘b’, contain ‘asize’ and
 * ‘bsize’ items, respectively. Each array must be sorted in ascending
 * order and duplicate values are not allowed.
 *
 * If ‘c’ is ‘NULL’, then no output is written. However, ‘csize’ is
 * used to indicate the number of items that would have been written
 * if an output array were provided.
 *
 * Otherwise, the user must provide an output array ‘c’ with enough
 * storage to hold the merged results. Moreover, the user must specify
 * the allocated size of the output array with the value pointed to by
 * ‘csize’. On success, the value returned in ‘csize’ indicates the
 * number of items that were written to the output array.
 */
int setintersection_sorted_unique_int32(
    int64_t * csize,
    int32_t * c,
    int64_t asize,
    const int32_t * a,
    int64_t bsize,
    const int32_t * b);

/**
 * ‘setintersection_sorted_unique_int64()’ merges two sorted arrays of
 * 64-bit signed integers based on a set intersection operation.
 *
 * The two arrays to be merged, ‘a’ and ‘b’, contain ‘asize’ and
 * ‘bsize’ items, respectively. Each array must be sorted in ascending
 * order and duplicate values are not allowed.
 *
 * If ‘c’ is ‘NULL’, then no output is written. However, ‘csize’ is
 * used to indicate the number of items that would have been written
 * if an output array were provided.
 *
 * Otherwise, the user must provide an output array ‘c’ with enough
 * storage to hold the merged results. Moreover, the user must specify
 * the allocated size of the output array with the value pointed to by
 * ‘csize’. On success, the value returned in ‘csize’ indicates the
 * number of items that were written to the output array.
 */
int setintersection_sorted_unique_int64(
    int64_t * csize,
    int64_t * c,
    int64_t asize,
    const int64_t * a,
    int64_t bsize,
    const int64_t * b);

/**
 * ‘setintersection_sorted_unique_int()’ merges two sorted arrays of
 * signed integers based on a set intersection operation.
 *
 * The two arrays to be merged, ‘a’ and ‘b’, contain ‘asize’ and
 * ‘bsize’ items, respectively. Each array must be sorted in ascending
 * order and duplicate values are not allowed.
 *
 * If ‘c’ is ‘NULL’, then no output is written. However, ‘csize’ is
 * used to indicate the number of items that would have been written
 * if an output array were provided.
 *
 * Otherwise, the user must provide an output array ‘c’ with enough
 * storage to hold the merged results. Moreover, the user must specify
 * the allocated size of the output array with the value pointed to by
 * ‘csize’. On success, the value returned in ‘csize’ indicates the
 * number of items that were written to the output array.
 */
int setintersection_sorted_unique_int(
    int64_t * csize,
    int * c,
    int64_t asize,
    const int * a,
    int64_t bsize,
    const int * b);

/*
 * set intersection on sorted arrays, possibly containing non-unique
 * values (i.e., duplicates are allowed)
 */

/**
 * ‘setintersection_sorted_nonunique_int32()’ merges two sorted arrays
 * of 32-bit signed integers based on a set intersection operation.
 *
 * The two arrays to be merged, ‘a’ and ‘b’, contain ‘asize’ and
 * ‘bsize’ items, respectively. Each array must be sorted in ascending
 * order and duplicate values are allowed.
 *
 * If ‘c’ is ‘NULL’, then no output is written. However, ‘csize’ is
 * used to indicate the number of items that would have been written
 * if an output array were provided.
 *
 * Otherwise, the user must provide an output array ‘c’ with enough
 * storage to hold the merged results. Moreover, the user must specify
 * the allocated size of the output array with the value pointed to by
 * ‘csize’. On success, the value returned in ‘csize’ indicates the
 * number of items that were written to the output array.
 */
int setintersection_sorted_nonunique_int32(
    int64_t * csize,
    int32_t * c,
    int64_t asize,
    const int32_t * a,
    int64_t bsize,
    const int32_t * b);

/**
 * ‘setintersection_sorted_nonunique_int64()’ merges two sorted arrays
 * of 64-bit signed integers based on a set intersection operation.
 *
 * The two arrays to be merged, ‘a’ and ‘b’, contain ‘asize’ and
 * ‘bsize’ items, respectively. Each array must be sorted in ascending
 * order and duplicate values are allowed.
 *
 * If ‘c’ is ‘NULL’, then no output is written. However, ‘csize’ is
 * used to indicate the number of items that would have been written
 * if an output array were provided.
 *
 * Otherwise, the user must provide an output array ‘c’ with enough
 * storage to hold the merged results. Moreover, the user must specify
 * the allocated size of the output array with the value pointed to by
 * ‘csize’. On success, the value returned in ‘csize’ indicates the
 * number of items that were written to the output array.
 */
int setintersection_sorted_nonunique_int64(
    int64_t * csize,
    int64_t * c,
    int64_t asize,
    const int64_t * a,
    int64_t bsize,
    const int64_t * b);

/**
 * ‘setintersection_sorted_nonunique_int()’ merges two sorted arrays
 * of signed integers based on a set intersection operation.
 *
 * The two arrays to be merged, ‘a’ and ‘b’, contain ‘asize’ and
 * ‘bsize’ items, respectively. Each array must be sorted in ascending
 * order and duplicate values are allowed.
 *
 * If ‘c’ is ‘NULL’, then no output is written. However, ‘csize’ is
 * used to indicate the number of items that would have been written
 * if an output array were provided.
 *
 * Otherwise, the user must provide an output array ‘c’ with enough
 * storage to hold the merged results. Moreover, the user must specify
 * the allocated size of the output array with the value pointed to by
 * ‘csize’. On success, the value returned in ‘csize’ indicates the
 * number of items that were written to the output array.
 */
int setintersection_sorted_nonunique_int(
    int64_t * csize,
    int * c,
    int64_t asize,
    const int * a,
    int64_t bsize,
    const int * b);

/*
 * set intersection operations on unsorted arrays of unique values
 * (i.e., no duplicates)
 */

/**
 * ‘setintersection_unsorted_unique_int32()’ merges two arrays of
 * 32-bit signed integers based on a set intersection operation.
 *
 * The two arrays to be merged, ‘a’ and ‘b’, contain ‘asize’ and
 * ‘bsize’ items, respectively. The arrays need not be sorted
 * beforehand, but they will be sorted if the function returns
 * successfully. Duplicate values are not allowed in the input arrays.
 *
 * If ‘c’ is ‘NULL’, then no output is written. However, ‘csize’ is
 * used to indicate the number of items that would have been written
 * if an output array were provided.
 *
 * Otherwise, the user must provide an output array ‘c’ with enough
 * storage to hold the merged results. Moreover, the user must specify
 * the allocated size of the output array with the value pointed to by
 * ‘csize’. On success, the value returned in ‘csize’ indicates the
 * number of items that were written to the output array.
 */
int setintersection_unsorted_unique_int32(
    int64_t * csize,
    int32_t * c,
    int64_t asize,
    int32_t * a,
    int64_t bsize,
    int32_t * b);

/**
 * ‘setintersection_unsorted_unique_int64()’ merges two arrays of
 * 64-bit signed integers based on a set intersection operation.
 *
 * The two arrays to be merged, ‘a’ and ‘b’, contain ‘asize’ and
 * ‘bsize’ items, respectively. The arrays need not be sorted
 * beforehand, but they will be sorted if the function returns
 * successfully. Duplicate values are not allowed in the input arrays.
 *
 * If ‘c’ is ‘NULL’, then no output is written. However, ‘csize’ is
 * used to indicate the number of items that would have been written
 * if an output array were provided.
 *
 * Otherwise, the user must provide an output array ‘c’ with enough
 * storage to hold the merged results. Moreover, the user must specify
 * the allocated size of the output array with the value pointed to by
 * ‘csize’. On success, the value returned in ‘csize’ indicates the
 * number of items that were written to the output array.
 */
int setintersection_unsorted_unique_int64(
    int64_t * csize,
    int64_t * c,
    int64_t asize,
    int64_t * a,
    int64_t bsize,
    int64_t * b);

/**
 * ‘setintersection_unsorted_unique_int()’ merges two arrays of signed
 * integers based on a set intersection operation.
 *
 * The two arrays to be merged, ‘a’ and ‘b’, contain ‘asize’ and
 * ‘bsize’ items, respectively. The arrays need not be sorted
 * beforehand, but they will be sorted if the function returns
 * successfully. Duplicate values are not allowed in the input arrays.
 *
 * If ‘c’ is ‘NULL’, then no output is written. However, ‘csize’ is
 * used to indicate the number of items that would have been written
 * if an output array were provided.
 *
 * Otherwise, the user must provide an output array ‘c’ with enough
 * storage to hold the merged results. Moreover, the user must specify
 * the allocated size of the output array with the value pointed to by
 * ‘csize’. On success, the value returned in ‘csize’ indicates the
 * number of items that were written to the output array.
 */
int setintersection_unsorted_unique_int(
    int64_t * csize,
    int * c,
    int64_t asize,
    int * a,
    int64_t bsize,
    int * b);

/*
 * set intersection on unsorted arrays, possibly containing non-unique
 * values (i.e., duplicates are allowed)
 */

/**
 * ‘setintersection_unsorted_nonunique_int32()’ merges two arrays of
 * 32-bit signed integers based on a set intersection operation.
 *
 * The two arrays to be merged, ‘a’ and ‘b’, contain ‘asize’ and
 * ‘bsize’ items, respectively. The arrays need not be sorted
 * beforehand, but they will be sorted if the function returns
 * successfully. Duplicate values are allowed in the input arrays.
 *
 * If ‘c’ is ‘NULL’, then no output is written. However, ‘csize’ is
 * used to indicate the number of items that would have been written
 * if an output array were provided.
 *
 * Otherwise, the user must provide an output array ‘c’ with enough
 * storage to hold the merged results. Moreover, the user must specify
 * the allocated size of the output array with the value pointed to by
 * ‘csize’. On success, the value returned in ‘csize’ indicates the
 * number of items that were written to the output array.
 */
int setintersection_unsorted_nonunique_int32(
    int64_t * csize,
    int32_t * c,
    int64_t asize,
    int32_t * a,
    int64_t bsize,
    int32_t * b);

/**
 * ‘setintersection_unsorted_nonunique_int64()’ merges two arrays of
 * 64-bit signed integers based on a set intersection operation.
 *
 * The two arrays to be merged, ‘a’ and ‘b’, contain ‘asize’ and
 * ‘bsize’ items, respectively. The arrays need not be sorted
 * beforehand, but they will be sorted if the function returns
 * successfully. Duplicate values are allowed in the input arrays.
 *
 * If ‘c’ is ‘NULL’, then no output is written. However, ‘csize’ is
 * used to indicate the number of items that would have been written
 * if an output array were provided.
 *
 * Otherwise, the user must provide an output array ‘c’ with enough
 * storage to hold the merged results. Moreover, the user must specify
 * the allocated size of the output array with the value pointed to by
 * ‘csize’. On success, the value returned in ‘csize’ indicates the
 * number of items that were written to the output array.
 */
int setintersection_unsorted_nonunique_int64(
    int64_t * csize,
    int64_t * c,
    int64_t asize,
    int64_t * a,
    int64_t bsize,
    int64_t * b);

/**
 * ‘setintersection_unsorted_nonunique_int()’ merges two arrays of
 * signed integers based on a set intersection operation.
 *
 * The two arrays to be merged, ‘a’ and ‘b’, contain ‘asize’ and
 * ‘bsize’ items, respectively. The arrays need not be sorted
 * beforehand, but they will be sorted if the function returns
 * successfully. Duplicate values are allowed in the input arrays.
 *
 * If ‘c’ is ‘NULL’, then no output is written. However, ‘csize’ is
 * used to indicate the number of items that would have been written
 * if an output array were provided.
 *
 * Otherwise, the user must provide an output array ‘c’ with enough
 * storage to hold the merged results. Moreover, the user must specify
 * the allocated size of the output array with the value pointed to by
 * ‘csize’. On success, the value returned in ‘csize’ indicates the
 * number of items that were written to the output array.
 */
int setintersection_unsorted_nonunique_int(
    int64_t * csize,
    int * c,
    int64_t asize,
    int * a,
    int64_t bsize,
    int * b);

/*
 * set difference operations on sorted arrays of unique values (i.e.,
 * no duplicates)
 */

/**
 * ‘setdifference_sorted_unique_int32()’ merges two sorted arrays of
 * 32-bit signed integers based on a set difference operation.
 *
 * The two arrays to be merged, ‘a’ and ‘b’, contain ‘asize’ and
 * ‘bsize’ items, respectively. Each array must be sorted in ascending
 * order and duplicate values are not allowed.
 *
 * If ‘c’ is ‘NULL’, then no output is written. However, ‘csize’ is
 * used to indicate the number of items that would have been written
 * if an output array were provided.
 *
 * Otherwise, the user must provide an output array ‘c’ with enough
 * storage to hold the merged results. Moreover, the user must specify
 * the allocated size of the output array with the value pointed to by
 * ‘csize’. On success, the value returned in ‘csize’ indicates the
 * number of items that were written to the output array.
 */
int setdifference_sorted_unique_int32(
    int64_t * csize,
    int32_t * c,
    int64_t asize,
    const int32_t * a,
    int64_t bsize,
    const int32_t * b);

/**
 * ‘setdifference_sorted_unique_int64()’ merges two sorted arrays of
 * 64-bit signed integers based on a set difference operation.
 *
 * The two arrays to be merged, ‘a’ and ‘b’, contain ‘asize’ and
 * ‘bsize’ items, respectively. Each array must be sorted in ascending
 * order and duplicate values are not allowed.
 *
 * If ‘c’ is ‘NULL’, then no output is written. However, ‘csize’ is
 * used to indicate the number of items that would have been written
 * if an output array were provided.
 *
 * Otherwise, the user must provide an output array ‘c’ with enough
 * storage to hold the merged results. Moreover, the user must specify
 * the allocated size of the output array with the value pointed to by
 * ‘csize’. On success, the value returned in ‘csize’ indicates the
 * number of items that were written to the output array.
 */
int setdifference_sorted_unique_int64(
    int64_t * csize,
    int64_t * c,
    int64_t asize,
    const int64_t * a,
    int64_t bsize,
    const int64_t * b);

/**
 * ‘setdifference_sorted_unique_int()’ merges two sorted arrays of
 * signed integers based on a set difference operation.
 *
 * The two arrays to be merged, ‘a’ and ‘b’, contain ‘asize’ and
 * ‘bsize’ items, respectively. Each array must be sorted in ascending
 * order and duplicate values are not allowed.
 *
 * If ‘c’ is ‘NULL’, then no output is written. However, ‘csize’ is
 * used to indicate the number of items that would have been written
 * if an output array were provided.
 *
 * Otherwise, the user must provide an output array ‘c’ with enough
 * storage to hold the merged results. Moreover, the user must specify
 * the allocated size of the output array with the value pointed to by
 * ‘csize’. On success, the value returned in ‘csize’ indicates the
 * number of items that were written to the output array.
 */
int setdifference_sorted_unique_int(
    int64_t * csize,
    int * c,
    int64_t asize,
    const int * a,
    int64_t bsize,
    const int * b);

/*
 * set difference on sorted arrays, possibly containing non-unique values
 * (i.e., duplicates are allowed)
 */

/**
 * ‘setdifference_sorted_nonunique_int32()’ merges two sorted arrays
 * of 32-bit signed integers based on a set difference operation.
 *
 * The two arrays to be merged, ‘a’ and ‘b’, contain ‘asize’ and
 * ‘bsize’ items, respectively. Each array must be sorted in ascending
 * order and duplicate values are allowed.
 *
 * If ‘c’ is ‘NULL’, then no output is written. However, ‘csize’ is
 * used to indicate the number of items that would have been written
 * if an output array were provided.
 *
 * Otherwise, the user must provide an output array ‘c’ with enough
 * storage to hold the merged results. Moreover, the user must specify
 * the allocated size of the output array with the value pointed to by
 * ‘csize’. On success, the value returned in ‘csize’ indicates the
 * number of items that were written to the output array.
 */
int setdifference_sorted_nonunique_int32(
    int64_t * csize,
    int32_t * c,
    int64_t asize,
    const int32_t * a,
    int64_t bsize,
    const int32_t * b);

/**
 * ‘setdifference_sorted_nonunique_int64()’ merges two sorted arrays
 * of 64-bit signed integers based on a set difference operation.
 *
 * The two arrays to be merged, ‘a’ and ‘b’, contain ‘asize’ and
 * ‘bsize’ items, respectively. Each array must be sorted in ascending
 * order and duplicate values are allowed.
 *
 * If ‘c’ is ‘NULL’, then no output is written. However, ‘csize’ is
 * used to indicate the number of items that would have been written
 * if an output array were provided.
 *
 * Otherwise, the user must provide an output array ‘c’ with enough
 * storage to hold the merged results. Moreover, the user must specify
 * the allocated size of the output array with the value pointed to by
 * ‘csize’. On success, the value returned in ‘csize’ indicates the
 * number of items that were written to the output array.
 */
int setdifference_sorted_nonunique_int64(
    int64_t * csize,
    int64_t * c,
    int64_t asize,
    const int64_t * a,
    int64_t bsize,
    const int64_t * b);

/**
 * ‘setdifference_sorted_nonunique_int()’ merges two sorted arrays of
 * signed integers based on a set difference operation.
 *
 * The two arrays to be merged, ‘a’ and ‘b’, contain ‘asize’ and
 * ‘bsize’ items, respectively. Each array must be sorted in ascending
 * order and duplicate values are allowed.
 *
 * If ‘c’ is ‘NULL’, then no output is written. However, ‘csize’ is
 * used to indicate the number of items that would have been written
 * if an output array were provided.
 *
 * Otherwise, the user must provide an output array ‘c’ with enough
 * storage to hold the merged results. Moreover, the user must specify
 * the allocated size of the output array with the value pointed to by
 * ‘csize’. On success, the value returned in ‘csize’ indicates the
 * number of items that were written to the output array.
 */
int setdifference_sorted_nonunique_int(
    int64_t * csize,
    int * c,
    int64_t asize,
    const int * a,
    int64_t bsize,
    const int * b);

/*
 * set difference operations on unsorted arrays of unique values
 * (i.e., no duplicates)
 */

/**
 * ‘setdifference_unsorted_unique_int32()’ merges two arrays of 32-bit
 * signed integers based on a set difference operation.
 *
 * The two arrays to be merged, ‘a’ and ‘b’, contain ‘asize’ and
 * ‘bsize’ items, respectively. The arrays need not be sorted
 * beforehand, but they will be sorted if the function returns
 * successfully. Duplicate values are not allowed in the input arrays.
 *
 * If ‘c’ is ‘NULL’, then no output is written. However, ‘csize’ is
 * used to indicate the number of items that would have been written
 * if an output array were provided.
 *
 * Otherwise, the user must provide an output array ‘c’ with enough
 * storage to hold the merged results. Moreover, the user must specify
 * the allocated size of the output array with the value pointed to by
 * ‘csize’. On success, the value returned in ‘csize’ indicates the
 * number of items that were written to the output array.
 */
int setdifference_unsorted_unique_int32(
    int64_t * csize,
    int32_t * c,
    int64_t asize,
    int32_t * a,
    int64_t bsize,
    int32_t * b);

/**
 * ‘setdifference_unsorted_unique_int64()’ merges two arrays of 64-bit
 * signed integers based on a set difference operation.
 *
 * The two arrays to be merged, ‘a’ and ‘b’, contain ‘asize’ and
 * ‘bsize’ items, respectively. The arrays need not be sorted
 * beforehand, but they will be sorted if the function returns
 * successfully. Duplicate values are not allowed in the input arrays.
 *
 * If ‘c’ is ‘NULL’, then no output is written. However, ‘csize’ is
 * used to indicate the number of items that would have been written
 * if an output array were provided.
 *
 * Otherwise, the user must provide an output array ‘c’ with enough
 * storage to hold the merged results. Moreover, the user must specify
 * the allocated size of the output array with the value pointed to by
 * ‘csize’. On success, the value returned in ‘csize’ indicates the
 * number of items that were written to the output array.
 */
int setdifference_unsorted_unique_int64(
    int64_t * csize,
    int64_t * c,
    int64_t asize,
    int64_t * a,
    int64_t bsize,
    int64_t * b);

/**
 * ‘setdifference_unsorted_unique_int()’ merges two arrays of signed
 * integers based on a set difference operation.
 *
 * The two arrays to be merged, ‘a’ and ‘b’, contain ‘asize’ and
 * ‘bsize’ items, respectively. The arrays need not be sorted
 * beforehand, but they will be sorted if the function returns
 * successfully. Duplicate values are not allowed in the input arrays.
 *
 * If ‘c’ is ‘NULL’, then no output is written. However, ‘csize’ is
 * used to indicate the number of items that would have been written
 * if an output array were provided.
 *
 * Otherwise, the user must provide an output array ‘c’ with enough
 * storage to hold the merged results. Moreover, the user must specify
 * the allocated size of the output array with the value pointed to by
 * ‘csize’. On success, the value returned in ‘csize’ indicates the
 * number of items that were written to the output array.
 */
int setdifference_unsorted_unique_int(
    int64_t * csize,
    int * c,
    int64_t asize,
    int * a,
    int64_t bsize,
    int * b);

/*
 * set difference on unsorted arrays, possibly containing non-unique
 * values (i.e., duplicates are allowed)
 */

/**
 * ‘setdifference_unsorted_nonunique_int32()’ merges two arrays of
 * 32-bit signed integers based on a set difference operation.
 *
 * The two arrays to be merged, ‘a’ and ‘b’, contain ‘asize’ and
 * ‘bsize’ items, respectively. The arrays need not be sorted
 * beforehand, but they will be sorted if the function returns
 * successfully. Duplicate values are allowed in the input arrays.
 *
 * If ‘c’ is ‘NULL’, then no output is written. However, ‘csize’ is
 * used to indicate the number of items that would have been written
 * if an output array were provided.
 *
 * Otherwise, the user must provide an output array ‘c’ with enough
 * storage to hold the merged results. Moreover, the user must specify
 * the allocated size of the output array with the value pointed to by
 * ‘csize’. On success, the value returned in ‘csize’ indicates the
 * number of items that were written to the output array.
 */
int setdifference_unsorted_nonunique_int32(
    int64_t * csize,
    int32_t * c,
    int64_t asize,
    int32_t * a,
    int64_t bsize,
    int32_t * b);

/**
 * ‘setdifference_unsorted_nonunique_int64()’ merges two arrays of
 * 64-bit signed integers based on a set difference operation.
 *
 * The two arrays to be merged, ‘a’ and ‘b’, contain ‘asize’ and
 * ‘bsize’ items, respectively. The arrays need not be sorted
 * beforehand, but they will be sorted if the function returns
 * successfully. Duplicate values are allowed in the input arrays.
 *
 * If ‘c’ is ‘NULL’, then no output is written. However, ‘csize’ is
 * used to indicate the number of items that would have been written
 * if an output array were provided.
 *
 * Otherwise, the user must provide an output array ‘c’ with enough
 * storage to hold the merged results. Moreover, the user must specify
 * the allocated size of the output array with the value pointed to by
 * ‘csize’. On success, the value returned in ‘csize’ indicates the
 * number of items that were written to the output array.
 */
int setdifference_unsorted_nonunique_int64(
    int64_t * csize,
    int64_t * c,
    int64_t asize,
    int64_t * a,
    int64_t bsize,
    int64_t * b);

/**
 * ‘setdifference_unsorted_nonunique_int()’ merges two arrays of
 * signed integers based on a set difference operation.
 *
 * The two arrays to be merged, ‘a’ and ‘b’, contain ‘asize’ and
 * ‘bsize’ items, respectively. The arrays need not be sorted
 * beforehand, but they will be sorted if the function returns
 * successfully. Duplicate values are allowed in the input arrays.
 *
 * If ‘c’ is ‘NULL’, then no output is written. However, ‘csize’ is
 * used to indicate the number of items that would have been written
 * if an output array were provided.
 *
 * Otherwise, the user must provide an output array ‘c’ with enough
 * storage to hold the merged results. Moreover, the user must specify
 * the allocated size of the output array with the value pointed to by
 * ‘csize’. On success, the value returned in ‘csize’ indicates the
 * number of items that were written to the output array.
 */
int setdifference_unsorted_nonunique_int(
    int64_t * csize,
    int * c,
    int64_t asize,
    int * a,
    int64_t bsize,
    int * b);

#endif
