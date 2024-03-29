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
 * Last modified: 2022-10-09
 *
 * Sorting.
 */

#ifndef LIBMTX_UTIL_SORT_H
#define LIBMTX_UTIL_SORT_H

#include <libmtx/libmtx-config.h>

#include <stdint.h>

/*
 * counting sort
 */

/**
 * ‘counting_sort_uint8()’ sorts an array of 8-bit integer keys using
 * a stable counting sort algorithm based on sorting the keys into 256
 * separate buckets.
 *
 * The keys to sort are provided in the array ‘keys’, where each key
 * is separated by a stride (in bytes) of ‘srcstride’. Therefore, the
 * ‘keys’ array must hold enough storage for ‘size’ values of type
 * ‘uint8_t’, while also taking into account the specified stride.
 *
 * If ‘sorted_keys’ is ‘NULL’, then this argument is ignored and the
 * sorted keys are not provided. Otherwise, it must point to an array
 * that holds enough storage for ‘size’ values of type ‘uint8_t’,
 * while also taking into account the stride, ‘dststride’, which is
 * given in bytes. On success, the array will contain the keys in
 * sorted order. Note that the arrays ‘keys’ and ‘sorted_keys’ must
 * not overlap.
 *
 * If ‘perm’ is ‘NULL’, then this argument is ignored and a sorting
 * permutation is not computed. Otherwise, it must point to an array
 * that holds enough storage for ‘size’ values of type ‘int64_t’. On
 * success, this array will contain the sorting permutation.
 *
 * If ‘bucketptr’ is ‘NULL’, then the function will internally
 * allocate a temporary array of 257 ‘int64_t’ values. Otherwise, the
 * pointer must point to an array with enough room for 257 ‘int64_t’
 * values. The array is needed for storing offsets to the first item
 * in each of the 256 buckets.
 */
int counting_sort_uint8(
    int64_t size,
    int srcstride,
    const uint8_t * keys,
    int dststride,
    uint8_t * sorted_keys,
    int64_t * perm,
    int64_t bucketptr[257]);

/**
 * ‘counting_sort_uint16()’ sorts an array of 16-bit integer keys
 * using a stable counting sort algorithm based on sorting the keys
 * into 65536 separate buckets.
 *
 * The keys to sort are provided in the array ‘keys’, where each key
 * is separated by a stride (in bytes) of ‘srcstride’. Therefore, the
 * ‘keys’ array must hold enough storage for ‘size’ values of type
 * ‘uint16_t’, while also taking into account the specified stride.
 *
 * If ‘sorted_keys’ is ‘NULL’, then this argument is ignored and the
 * sorted keys are not provided. Otherwise, it must point to an array
 * that holds enough storage for ‘size’ values of type ‘uint16_t’,
 * while also taking into account the stride, ‘dststride’, which is
 * given in bytes. On success, the array will contain the keys in
 * sorted order. Note that the arrays ‘keys’ and ‘sorted_keys’ must
 * not overlap.
 *
 * If ‘perm’ is ‘NULL’, then this argument is ignored and a sorting
 * permutation is not computed. Otherwise, it must point to an array
 * that holds enough storage for ‘size’ values of type ‘int64_t’. On
 * success, this array will contain the sorting permutation.
 *
 * If ‘bucketptr’ is ‘NULL’, then the function will internally
 * allocate a temporary array of 65537 ‘int64_t’ values. Otherwise,
 * the pointer must point to an array with enough room for 65537
 * ‘int64_t’ values. The array is needed for storing offsets to the
 * first item in each of the 65536 buckets.
 */
int counting_sort_uint16(
    int64_t size,
    int srcstride,
    const uint16_t * keys,
    int dststride,
    uint16_t * sorted_keys,
    int64_t * perm,
    int64_t bucketptr[65537]);

/*
 * radix sort for unsigned integers
 */

/**
 * ‘radix_sort_uint32()’ sorts an array of 32-bit unsigned integers in
 * ascending order using a radix sort algorithm.
 *
 * The number of keys to sort is given by ‘size’, and the unsorted,
 * integer keys are given in the array ‘keys’. On success, the same
 * array will contain the keys in sorted order.
 *
 * If ‘perm’ is ‘NULL’, then this argument is ignored and a sorting
 * permutation is not computed. Otherwise, it must point to an array
 * that holds enough storage for ‘size’ values of type ‘int64_t’. On
 * success, this array will contain the sorting permutation, mapping
 * the locations of the original, unsorted keys to their new locations
 * in the sorted array.
 */
int radix_sort_uint32(
    int64_t size,
    uint32_t * keys,
    int64_t * perm);

/**
 * ‘radix_sort_uint64()’ sorts an array of 64-bit unsigned integers in
 * ascending order using a radix sort algorithm.
 *
 * The number of keys to sort is given by ‘size’, and the unsorted,
 * integer keys are given in the array ‘keys’. On success, the same
 * array will contain the keys in sorted order.
 *
 * The value ‘stride’ is used to specify a stride (in bytes), which is
 * used when accessing elements of the ‘keys’ array. This is useful
 * for cases where the keys are not necessarily stored contiguously in
 * memory.
 *
 * If ‘perm’ is ‘NULL’, then this argument is ignored and a sorting
 * permutation is not computed. Otherwise, it must point to an array
 * that holds enough storage for ‘size’ values of type ‘int64_t’. On
 * success, this array will contain the sorting permutation, mapping
 * the locations of the original, unsorted keys to their new locations
 * in the sorted array.
 */
int radix_sort_uint64(
    int64_t size,
    int stride,
    uint64_t * keys,
    int64_t * perm);

/*
 * radix sort for signed integers
 */

/**
 * ‘radix_sort_int32()’ sorts an array of 32-bit (signed) integers in
 * ascending order using a radix sort algorithm.
 *
 * The number of keys to sort is given by ‘size’, and the unsorted,
 * integer keys are given in the array ‘keys’. On success, the same
 * array will contain the keys in sorted order.
 *
 * If ‘perm’ is ‘NULL’, then this argument is ignored and a sorting
 * permutation is not computed. Otherwise, it must point to an array
 * that holds enough storage for ‘size’ values of type ‘int64_t’. On
 * success, this array will contain the sorting permutation, mapping
 * the locations of the original, unsorted keys to their new locations
 * in the sorted array.
 */
int radix_sort_int32(
    int64_t size,
    int32_t * keys,
    int64_t * perm);

/**
 * ‘radix_sort_int64()’ sorts an array of 64-bit (signed) integers in
 * ascending order using a radix sort algorithm.
 *
 * The number of keys to sort is given by ‘size’, and the unsorted,
 * integer keys are given in the array ‘keys’. On success, the same
 * array will contain the keys in sorted order.
 *
 * The value ‘stride’ is used to specify a stride (in bytes), which is
 * used when accessing elements of the ‘keys’ array. This is useful
 * for cases where the keys are not necessarily stored contiguously in
 * memory.
 *
 * If ‘perm’ is ‘NULL’, then this argument is ignored and a sorting
 * permutation is not computed. Otherwise, it must point to an array
 * that holds enough storage for ‘size’ values of type ‘int64_t’. On
 * success, this array will contain the sorting permutation, mapping
 * the locations of the original, unsorted keys to their new locations
 * in the sorted array.
 */
int radix_sort_int64(
    int64_t size,
    int stride,
    int64_t * keys,
    int64_t * perm);

/**
 * ‘radix_sort_int()’ sorts an array of (signed) integers in ascending
 * order using a radix sort algorithm.
 *
 * The number of keys to sort is given by ‘size’, and the unsorted,
 * integer keys are given in the array ‘keys’. On success, the same
 * array will contain the keys in sorted order.
 *
 * If ‘perm’ is ‘NULL’, then this argument is ignored and a sorting
 * permutation is not computed. Otherwise, it must point to an array
 * that holds enough storage for ‘size’ values of type ‘int64_t’. On
 * success, this array will contain the sorting permutation, mapping
 * the locations of the original, unsorted keys to their new locations
 * in the sorted array.
 */
int radix_sort_int(
    int64_t size,
    int * keys,
    int64_t * perm);

/*
 * radix sort for pairs of integers
 */

/**
 * ‘radix_sort_uint32_pair()’ sorts pairs of 32-bit unsigned integers
 * in ascending, lexicographic order using a radix sort algorithm.
 *
 * The number of pairs to sort is given by ‘size’, and the unsorted,
 * integer keys are given in the arrays ‘a’ and ‘b’. On success, the
 * same arrays will contain the keys in sorted order.
 *
 * The values ‘astride’ and ‘bstride’ may be used to specify strides
 * (in bytes) that are used when accessing the keys in ‘a’ and ‘b’,
 * respectively. This is useful for cases where the keys are not
 * necessarily stored contiguously in memory.
 *
 * If ‘perm’ is ‘NULL’, then this argument is ignored and a sorting
 * permutation is not computed. Otherwise, it must point to an array
 * of length ‘size’. On success, this array will contain the sorting
 * permutation, mapping the locations of the original, unsorted keys
 * to their new locations in the sorted array.
 */
int radix_sort_uint32_pair(
    int64_t size,
    int astride,
    uint32_t * a,
    int bstride,
    uint32_t * b,
    int64_t * perm);

/**
 * ‘radix_sort_uint64_pair()’ sorts pairs of 64-bit unsigned integers
 * in ascending, lexicographic order using a radix sort algorithm.
 *
 * The number of pairs to sort is given by ‘size’, and the unsorted,
 * integer keys are given in the arrays ‘a’ and ‘b’. On success, the
 * same arrays will contain the keys in sorted order.
 *
 * The values ‘astride’ and ‘bstride’ may be used to specify strides
 * (in bytes) that are used when accessing the keys in ‘a’ and ‘b’,
 * respectively. This is useful for cases where the keys are not
 * necessarily stored contiguously in memory.
 *
 * If ‘perm’ is ‘NULL’, then this argument is ignored and a sorting
 * permutation is not computed. Otherwise, it must point to an array
 * of length ‘size’. On success, this array will contain the sorting
 * permutation, mapping the locations of the original, unsorted keys
 * to their new locations in the sorted array.
 */
int radix_sort_uint64_pair(
    int64_t size,
    int astride,
    uint64_t * a,
    int bstride,
    uint64_t * b,
    int64_t * perm);

/**
 * ‘radix_sort_int32_pair()’ sorts pairs of 32-bit signed integers
 * in ascending, lexicographic order using a radix sort algorithm.
 *
 * The number of pairs to sort is given by ‘size’, and the unsorted,
 * integer keys are given in the arrays ‘a’ and ‘b’. On success, the
 * same arrays will contain the keys in sorted order.
 *
 * The values ‘astride’ and ‘bstride’ may be used to specify strides
 * (in bytes) that are used when accessing the keys in ‘a’ and ‘b’,
 * respectively. This is useful for cases where the keys are not
 * necessarily stored contiguously in memory.
 *
 * If ‘perm’ is ‘NULL’, then this argument is ignored and a sorting
 * permutation is not computed. Otherwise, it must point to an array
 * of length ‘size’. On success, this array will contain the sorting
 * permutation, mapping the locations of the original, unsorted keys
 * to their new locations in the sorted array.
 */
int radix_sort_int32_pair(
    int64_t size,
    int astride,
    int32_t * a,
    int bstride,
    int32_t * b,
    int64_t * perm);

/**
 * ‘radix_sort_int64_pair()’ sorts pairs of 64-bit signed integers
 * in ascending, lexicographic order using a radix sort algorithm.
 *
 * The number of pairs to sort is given by ‘size’, and the unsorted,
 * integer keys are given in the arrays ‘a’ and ‘b’. On success, the
 * same arrays will contain the keys in sorted order.
 *
 * The values ‘astride’ and ‘bstride’ may be used to specify strides
 * (in bytes) that are used when accessing the keys in ‘a’ and ‘b’,
 * respectively. This is useful for cases where the keys are not
 * necessarily stored contiguously in memory.
 *
 * If ‘perm’ is ‘NULL’, then this argument is ignored and a sorting
 * permutation is not computed. Otherwise, it must point to an array
 * of length ‘size’. On success, this array will contain the sorting
 * permutation, mapping the locations of the original, unsorted keys
 * to their new locations in the sorted array.
 */
int radix_sort_int64_pair(
    int64_t size,
    int astride,
    int64_t * a,
    int bstride,
    int64_t * b,
    int64_t * perm);

/**
 * ‘radix_sort_int_pair()’ sorts pairs of signed integers in
 * ascending, lexicographic order using a radix sort algorithm.
 *
 * The number of pairs to sort is given by ‘size’, and the unsorted,
 * integer keys are given in the arrays ‘a’ and ‘b’. On success, the
 * same arrays will contain the keys in sorted order.
 *
 * The values ‘astride’ and ‘bstride’ may be used to specify strides
 * (in bytes) that are used when accessing the keys in ‘a’ and ‘b’,
 * respectively. This is useful for cases where the keys are not
 * necessarily stored contiguously in memory.
 *
 * If ‘perm’ is ‘NULL’, then this argument is ignored and a sorting
 * permutation is not computed. Otherwise, it must point to an array
 * of length ‘size’. On success, this array will contain the sorting
 * permutation, mapping the locations of the original, unsorted keys
 * to their new locations in the sorted array.
 */
int radix_sort_int_pair(
    int64_t size,
    int astride,
    int * a,
    int bstride,
    int * b,
    int64_t * perm);

#endif
