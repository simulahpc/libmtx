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
 * Merge operations on arrays.
 */

#include "libmtx/util/merge.h"
#include "libmtx/util/sort.h"

#include <errno.h>

#include <stddef.h>
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
 *
 * Returns ‘0’ if successful, or ‘EINVAL’ if the output array is not
 * large enough.
 */
int compact_sorted_int32(
    int64_t * bsize,
    int32_t * b,
    int64_t asize,
    const int32_t * a,
    int64_t * dstidx)
{
    int64_t i = 0, k = 0;
    if (b && dstidx) {
        while (i < asize) {
            if (k >= *bsize) return EINVAL;
            dstidx[i] = k; b[k++] = a[i];
            for (i++; i < asize && a[i] == a[i-1]; i++) { dstidx[i] = k-1; }
        }
    } else if (b) {
        while (i < asize) {
            if (k >= *bsize) return EINVAL;
            b[k++] = a[i];
            for (i++; i < asize && a[i] == a[i-1]; i++) {}
        }
    } else if (dstidx) {
        while (i < asize) {
            dstidx[i] = k; k++;
            for (i++; i < asize && a[i] == a[i-1]; i++) { dstidx[i] = k-1; }
        }
    } else {
        while (i < asize) {
            k++;
            for (i++; i < asize && a[i] == a[i-1]; i++) {}
        }
    }
    *bsize = k;
    return 0;
}

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
 *
 * Returns ‘0’ if successful, or ‘EINVAL’ if the output array is not
 * large enough.
 */
int compact_sorted_int64(
    int64_t * bsize,
    int64_t * b,
    int64_t asize,
    const int64_t * a,
    int64_t * dstidx)
{
    int64_t i = 0, k = 0;
    if (b && dstidx) {
        while (i < asize) {
            if (k >= *bsize) return EINVAL;
            dstidx[i] = k; b[k++] = a[i];
            for (i++; i < asize && a[i] == a[i-1]; i++) { dstidx[i] = k-1; }
        }
    } else if (b) {
        while (i < asize) {
            if (k >= *bsize) return EINVAL;
            b[k++] = a[i];
            for (i++; i < asize && a[i] == a[i-1]; i++) {}
        }
    } else if (dstidx) {
        while (i < asize) {
            dstidx[i] = k; k++;
            for (i++; i < asize && a[i] == a[i-1]; i++) { dstidx[i] = k-1; }
        }
    } else {
        while (i < asize) {
            k++;
            for (i++; i < asize && a[i] == a[i-1]; i++) {}
        }
    }
    *bsize = k;
    return 0;
}

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
 *
 * Returns ‘0’ if successful, or ‘EINVAL’ if the output array is not
 * large enough.
 */
int compact_sorted_int(
    int64_t * bsize,
    int * b,
    int64_t asize,
    const int * a,
    int64_t * dstidx)
{
    int64_t i = 0, k = 0;
    if (b && dstidx) {
        while (i < asize) {
            if (k >= *bsize) return EINVAL;
            dstidx[i] = k; b[k++] = a[i];
            for (i++; i < asize && a[i] == a[i-1]; i++) { dstidx[i] = k-1; }
        }
    } else if (b) {
        while (i < asize) {
            if (k >= *bsize) return EINVAL;
            b[k++] = a[i];
            for (i++; i < asize && a[i] == a[i-1]; i++) {}
        }
    } else if (dstidx) {
        while (i < asize) {
            dstidx[i] = k; k++;
            for (i++; i < asize && a[i] == a[i-1]; i++) { dstidx[i] = k-1; }
        }
    } else {
        while (i < asize) {
            k++;
            for (i++; i < asize && a[i] == a[i-1]; i++) {}
        }
    }
    *bsize = k;
    return 0;
}

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
 *
 * Returns ‘0’ if successful, or ‘EINVAL’ if the output array is not
 * large enough.
 */
int compact_sorted_int32_pair(
    int64_t * bsize,
    int32_t (* b)[2],
    int64_t asize,
    const int32_t (* a)[2],
    int64_t * dstidx)
{
    int64_t i = 0, k = 0;
    if (b && dstidx) {
        while (i < asize) {
            if (k >= *bsize) return EINVAL;
            dstidx[i] = k; b[k][0] = a[i][0]; b[k][1] = a[i][1]; k++;
            for (i++; i < asize && a[i][0] == a[i-1][0] && a[i][1] == a[i-1][1]; i++) { dstidx[i] = k-1; }
        }
    } else if (b) {
        while (i < asize) {
            if (k >= *bsize) return EINVAL;
            b[k][0] = a[i][0]; b[k][1] = a[i][1]; k++;
            for (i++; i < asize && a[i][0] == a[i-1][0] && a[i][1] == a[i-1][1]; i++) {}
        }
    } else if (dstidx) {
        while (i < asize) {
            dstidx[i] = k; k++;
            for (i++; i < asize && a[i][0] == a[i-1][0] && a[i][1] == a[i-1][1]; i++) { dstidx[i] = k-1; }
        }
    } else {
        while (i < asize) {
            k++;
            for (i++; i < asize && a[i][0] == a[i-1][0] && a[i][1] == a[i-1][1]; i++) {}
        }
    }
    *bsize = k;
    return 0;
}

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
 *
 * Returns ‘0’ if successful, or ‘EINVAL’ if the output array is not
 * large enough.
 */
int compact_sorted_int64_pair(
    int64_t * bsize,
    int64_t (* b)[2],
    int64_t asize,
    const int64_t (* a)[2],
    int64_t * dstidx)
{
    int64_t i = 0, k = 0;
    if (b && dstidx) {
        while (i < asize) {
            if (k >= *bsize) return EINVAL;
            dstidx[i] = k; b[k][0] = a[i][0]; b[k][1] = a[i][1]; k++;
            for (i++; i < asize && a[i][0] == a[i-1][0] && a[i][1] == a[i-1][1]; i++) { dstidx[i] = k-1; }
        }
    } else if (b) {
        while (i < asize) {
            if (k >= *bsize) return EINVAL;
            b[k][0] = a[i][0]; b[k][1] = a[i][1]; k++;
            for (i++; i < asize && a[i][0] == a[i-1][0] && a[i][1] == a[i-1][1]; i++) {}
        }
    } else if (dstidx) {
        while (i < asize) {
            dstidx[i] = k; k++;
            for (i++; i < asize && a[i][0] == a[i-1][0] && a[i][1] == a[i-1][1]; i++) { dstidx[i] = k-1; }
        }
    } else {
        while (i < asize) {
            k++;
            for (i++; i < asize && a[i][0] == a[i-1][0] && a[i][1] == a[i-1][1]; i++) {}
        }
    }
    *bsize = k;
    return 0;
}

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
 *
 * Returns ‘0’ if successful, or ‘EINVAL’ if the output array is not
 * large enough.
 */
int compact_sorted_int_pair(
    int64_t * bsize,
    int (* b)[2],
    int64_t asize,
    const int (* a)[2],
    int64_t * dstidx)
{
    int64_t i = 0, k = 0;
    if (b && dstidx) {
        while (i < asize) {
            if (k >= *bsize) return EINVAL;
            dstidx[i] = k; b[k][0] = a[i][0]; b[k][1] = a[i][1]; k++;
            for (i++; i < asize && a[i][0] == a[i-1][0] && a[i][1] == a[i-1][1]; i++) { dstidx[i] = k-1; }
        }
    } else if (b) {
        while (i < asize) {
            if (k >= *bsize) return EINVAL;
            b[k][0] = a[i][0]; b[k][1] = a[i][1]; k++;
            for (i++; i < asize && a[i][0] == a[i-1][0] && a[i][1] == a[i-1][1]; i++) {}
        }
    } else if (dstidx) {
        while (i < asize) {
            dstidx[i] = k; k++;
            for (i++; i < asize && a[i][0] == a[i-1][0] && a[i][1] == a[i-1][1]; i++) { dstidx[i] = k-1; }
        }
    } else {
        while (i < asize) {
            k++;
            for (i++; i < asize && a[i][0] == a[i-1][0] && a[i][1] == a[i-1][1]; i++) {}
        }
    }
    *bsize = k;
    return 0;
}

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
 *
 * Returns ‘0’ if successful, or ‘EINVAL’ if the output array is not
 * large enough.
 */
int compact_unsorted_int32(
    int64_t * bsize,
    int32_t * b,
    int64_t asize,
    int32_t * a,
    int64_t * perm,
    int64_t * dstidx)
{
    int err = radix_sort_int32(asize, a, perm);
    if (err) return err;
    return compact_sorted_int32(bsize, b, asize, a, dstidx);
}

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
 *
 * Returns ‘0’ if successful, or ‘EINVAL’ if the output array is not
 * large enough.
 */
int compact_unsorted_int64(
    int64_t * bsize,
    int64_t * b,
    int64_t asize,
    int64_t * a,
    int64_t * perm,
    int64_t * dstidx)
{
    int err = radix_sort_int64(asize, sizeof(*a), a, perm);
    if (err) return err;
    return compact_sorted_int64(bsize, b, asize, a, dstidx);
}

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
 *
 * Returns ‘0’ if successful, or ‘EINVAL’ if the output array is not
 * large enough.
 */
int compact_unsorted_int(
    int64_t * bsize,
    int * b,
    int64_t asize,
    int * a,
    int64_t * perm,
    int64_t * dstidx)
{
    int err = radix_sort_int(asize, a, perm);
    if (err) return err;
    return compact_sorted_int(bsize, b, asize, a, dstidx);
}

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
 *
 * Returns ‘0’ if successful, or ‘EINVAL’ if the output array is not
 * large enough.
 */
int compact_unsorted_int32_pair(
    int64_t * bsize,
    int32_t (* b)[2],
    int64_t asize,
    int32_t (* a)[2],
    int64_t * perm,
    int64_t * dstidx)
{
    int err = radix_sort_int32_pair(
        asize, sizeof(*a), &a[0][0], sizeof(*a), &a[0][1], perm);
    if (err) return err;
    return compact_sorted_int32_pair(bsize, b, asize, a, dstidx);
}

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
 *
 * Returns ‘0’ if successful, or ‘EINVAL’ if the output array is not
 * large enough.
 */
int compact_unsorted_int64_pair(
    int64_t * bsize,
    int64_t (* b)[2],
    int64_t asize,
    int64_t (* a)[2],
    int64_t * perm,
    int64_t * dstidx)
{
    int err = radix_sort_int64_pair(
        asize, sizeof(*a), &a[0][0], sizeof(*a), &a[0][1], perm);
    if (err) return err;
    return compact_sorted_int64_pair(bsize, b, asize, a, dstidx);
}

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
 *
 * Returns ‘0’ if successful, or ‘EINVAL’ if the output array is not
 * large enough.
 */
int compact_unsorted_int_pair(
    int64_t * bsize,
    int (* b)[2],
    int64_t asize,
    int (* a)[2],
    int64_t * perm,
    int64_t * dstidx)
{
    int err = radix_sort_int_pair(
        asize, sizeof(*a), &a[0][0], sizeof(*a), &a[0][1], perm);
    if (err) return err;
    return compact_sorted_int_pair(bsize, b, asize, a, dstidx);
}

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
 *
 * Returns ‘0’ if successful, or ‘EINVAL’ if the output array is not
 * large enough.
 */
int merge_sorted_int32(
    int64_t csize,
    int32_t * c,
    int64_t asize,
    const int32_t * a,
    int64_t * adstidx,
    int64_t bsize,
    const int32_t * b,
    int64_t * bdstidx)
{
    if (csize < asize + bsize) return EINVAL;
    if (adstidx && bdstidx) {
        int64_t i = 0, j = 0, k = 0;
        while (i < asize && j < bsize) {
            if (a[i] < b[j]) { adstidx[i] = k; c[k++] = a[i++]; }
            else if (a[i] > b[j]) { bdstidx[j] = k; c[k++] = b[j++]; }
            else { adstidx[i] = bdstidx[j] = k; c[k++] = a[i++]; }
        }
        while (i < asize) { adstidx[i] = k; c[k++] = a[i++]; }
        while (j < bsize) { bdstidx[j] = k; c[k++] = b[j++]; }
    } else {
        int64_t i = 0, j = 0, k = 0;
        while (i < asize && j < bsize) {
            if (a[i] < b[j]) c[k++] = a[i++];
            else if (a[i] > b[j]) c[k++] = b[j++];
            else { c[k++] = a[i++]; }
        }
        while (i < asize) c[k++] = a[i++];
        while (j < bsize) c[k++] = b[j++];
    }
    return 0;
}

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
 *
 * Returns ‘0’ if successful, or ‘EINVAL’ if the output array is not
 * large enough.
 */
int merge_sorted_int64(
    int64_t csize,
    int64_t * c,
    int64_t asize,
    const int64_t * a,
    int64_t * adstidx,
    int64_t bsize,
    const int64_t * b,
    int64_t * bdstidx)
{
    if (csize < asize + bsize) return EINVAL;
    if (adstidx && bdstidx) {
        int64_t i = 0, j = 0, k = 0;
        while (i < asize && j < bsize) {
            if (a[i] < b[j]) { adstidx[i] = k; c[k++] = a[i++]; }
            else if (a[i] > b[j]) { bdstidx[j] = k; c[k++] = b[j++]; }
            else { adstidx[i] = bdstidx[j] = k; c[k++] = a[i++]; }
        }
        while (i < asize) { adstidx[i] = k; c[k++] = a[i++]; }
        while (j < bsize) { bdstidx[j] = k; c[k++] = b[j++]; }
    } else {
        int64_t i = 0, j = 0, k = 0;
        while (i < asize && j < bsize) {
            if (a[i] < b[j]) c[k++] = a[i++];
            else if (a[i] > b[j]) c[k++] = b[j++];
            else { c[k++] = a[i++]; }
        }
        while (i < asize) c[k++] = a[i++];
        while (j < bsize) c[k++] = b[j++];
    }
    return 0;
}

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
 *
 * Returns ‘0’ if successful, or ‘EINVAL’ if the output array is not
 * large enough.
 */
int merge_sorted_int(
    int64_t csize,
    int * c,
    int64_t asize,
    const int * a,
    int64_t * adstidx,
    int64_t bsize,
    const int * b,
    int64_t * bdstidx)
{
    if (csize < asize + bsize) return EINVAL;
    if (adstidx && bdstidx) {
        int64_t i = 0, j = 0, k = 0;
        while (i < asize && j < bsize) {
            if (a[i] < b[j]) { adstidx[i] = k; c[k++] = a[i++]; }
            else if (a[i] > b[j]) { bdstidx[j] = k; c[k++] = b[j++]; }
            else { adstidx[i] = bdstidx[j] = k; c[k++] = a[i++]; }
        }
        while (i < asize) { adstidx[i] = k; c[k++] = a[i++]; }
        while (j < bsize) { bdstidx[j] = k; c[k++] = b[j++]; }
    } else {
        int64_t i = 0, j = 0, k = 0;
        while (i < asize && j < bsize) {
            if (a[i] < b[j]) c[k++] = a[i++];
            else if (a[i] > b[j]) c[k++] = b[j++];
            else { c[k++] = a[i++]; }
        }
        while (i < asize) c[k++] = a[i++];
        while (j < bsize) c[k++] = b[j++];
    }
    return 0;
}

/*
 * set union operations on sorted arrays of unique values (i.e., no
 * duplicates)
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
 *
 * Returns ‘0’ if successful, or ‘EINVAL’ if the output array is not
 * large enough.
 */
int setunion_sorted_unique_int32(
    int64_t * csize,
    int32_t * c,
    int64_t asize,
    const int32_t * a,
    int64_t * adstidx,
    int64_t bsize,
    const int32_t * b,
    int64_t * bdstidx)
{
    if (c && adstidx && bdstidx) {
        int64_t i = 0, j = 0, k = 0;
        while (i < asize && j < bsize) {
            if (k >= *csize) return EINVAL;
            if (a[i] < b[j]) { adstidx[i] = k; c[k++] = a[i++]; }
            else if (a[i] > b[j]) { bdstidx[j] = k; c[k++] = b[j++]; }
            else { adstidx[i] = k; bdstidx[j] = k; c[k++] = a[i++]; j++; }
        }
        if (asize - i >= *csize) return EINVAL;
        while (i < asize) {
            if (k >= *csize) return EINVAL;
            adstidx[i] = k; c[k++] = a[i++];
        }
        if (bsize - j >= *csize) return EINVAL;
        while (j < bsize) {
            if (k >= *csize) return EINVAL;
            bdstidx[i] = k; c[k++] = b[j++];
        }
        *csize = k;
    } else if (c) {
        int64_t i = 0, j = 0, k = 0;
        while (i < asize && j < bsize) {
            if (k >= *csize) return EINVAL;
            if (a[i] < b[j]) { c[k++] = a[i++]; }
            else if (a[i] > b[j]) { c[k++] = b[j++]; }
            else { c[k++] = a[i++]; j++; }
        }
        if (asize - i >= *csize) return EINVAL;
        while (i < asize) {
            if (k >= *csize) return EINVAL;
            c[k++] = a[i++];
        }
        if (bsize - j >= *csize) return EINVAL;
        while (j < bsize) {
            if (k >= *csize) return EINVAL;
            c[k++] = b[j++];
        }
        *csize = k;
    } else {
        int64_t i = 0, j = 0, k = 0;
        while (i < asize && j < bsize) {
            if (a[i] < b[j]) { k++; i++; }
            else if (a[i] > b[j]) { k++; j++; }
            else { k++; i++; j++; }
        }
        while (i < asize) { k++; i++; }
        while (j < bsize) { k++; j++; }
        *csize = k;
    }
    return 0;
}

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
 *
 * Returns ‘0’ if successful, or ‘EINVAL’ if the output array is not
 * large enough.
 */
int setunion_sorted_unique_int64(
    int64_t * csize,
    int64_t * c,
    int64_t asize,
    const int64_t * a,
    int64_t * adstidx,
    int64_t bsize,
    const int64_t * b,
    int64_t * bdstidx)
{
    if (c && adstidx && bdstidx) {
        int64_t i = 0, j = 0, k = 0;
        while (i < asize && j < bsize) {
            if (k >= *csize) return EINVAL;
            if (a[i] < b[j]) { adstidx[i] = k; c[k++] = a[i++]; }
            else if (a[i] > b[j]) { bdstidx[j] = k; c[k++] = b[j++]; }
            else { adstidx[i] = k; bdstidx[j] = k; c[k++] = a[i++]; j++; }
        }
        if (asize - i >= *csize) return EINVAL;
        while (i < asize) {
            if (k >= *csize) return EINVAL;
            adstidx[i] = k; c[k++] = a[i++];
        }
        if (bsize - j >= *csize) return EINVAL;
        while (j < bsize) {
            if (k >= *csize) return EINVAL;
            bdstidx[i] = k; c[k++] = b[j++];
        }
        *csize = k;
    } else if (c) {
        int64_t i = 0, j = 0, k = 0;
        while (i < asize && j < bsize) {
            if (k >= *csize) return EINVAL;
            if (a[i] < b[j]) { c[k++] = a[i++]; }
            else if (a[i] > b[j]) { c[k++] = b[j++]; }
            else { c[k++] = a[i++]; j++; }
        }
        if (asize - i >= *csize) return EINVAL;
        while (i < asize) {
            if (k >= *csize) return EINVAL;
            c[k++] = a[i++];
        }
        if (bsize - j >= *csize) return EINVAL;
        while (j < bsize) {
            if (k >= *csize) return EINVAL;
            c[k++] = b[j++];
        }
        *csize = k;
    } else {
        int64_t i = 0, j = 0, k = 0;
        while (i < asize && j < bsize) {
            if (a[i] < b[j]) { k++; i++; }
            else if (a[i] > b[j]) { k++; j++; }
            else { k++; i++; j++; }
        }
        while (i < asize) { k++; i++; }
        while (j < bsize) { k++; j++; }
        *csize = k;
    }
    return 0;
}

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
 *
 * Returns ‘0’ if successful, or ‘EINVAL’ if the output array is not
 * large enough.
 */
int setunion_sorted_unique_int(
    int64_t * csize,
    int * c,
    int64_t asize,
    const int * a,
    int64_t * adstidx,
    int64_t bsize,
    const int * b,
    int64_t * bdstidx)
{
    if (c && adstidx && bdstidx) {
        int64_t i = 0, j = 0, k = 0;
        while (i < asize && j < bsize) {
            if (k >= *csize) return EINVAL;
            if (a[i] < b[j]) { adstidx[i] = k; c[k++] = a[i++]; }
            else if (a[i] > b[j]) { bdstidx[j] = k; c[k++] = b[j++]; }
            else { adstidx[i] = k; bdstidx[j] = k; c[k++] = a[i++]; j++; }
        }
        if (asize - i >= *csize) return EINVAL;
        while (i < asize) {
            if (k >= *csize) return EINVAL;
            adstidx[i] = k; c[k++] = a[i++];
        }
        if (bsize - j >= *csize) return EINVAL;
        while (j < bsize) {
            if (k >= *csize) return EINVAL;
            bdstidx[i] = k; c[k++] = b[j++];
        }
        *csize = k;
    } else if (c) {
        int64_t i = 0, j = 0, k = 0;
        while (i < asize && j < bsize) {
            if (k >= *csize) return EINVAL;
            if (a[i] < b[j]) { c[k++] = a[i++]; }
            else if (a[i] > b[j]) { c[k++] = b[j++]; }
            else { c[k++] = a[i++]; j++; }
        }
        if (asize - i >= *csize) return EINVAL;
        while (i < asize) {
            if (k >= *csize) return EINVAL;
            c[k++] = a[i++];
        }
        if (bsize - j >= *csize) return EINVAL;
        while (j < bsize) {
            if (k >= *csize) return EINVAL;
            c[k++] = b[j++];
        }
        *csize = k;
    } else {
        int64_t i = 0, j = 0, k = 0;
        while (i < asize && j < bsize) {
            if (a[i] < b[j]) { k++; i++; }
            else if (a[i] > b[j]) { k++; j++; }
            else { k++; i++; j++; }
        }
        while (i < asize) { k++; i++; }
        while (j < bsize) { k++; j++; }
        *csize = k;
    }
    return 0;
}

/*
 * set union on sorted arrays, possibly containing non-unique values
 * (i.e., duplicates are allowed)
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
 *
 * Returns ‘0’ if successful, or ‘EINVAL’ if the output array is not
 * large enough.
 */
int setunion_sorted_nonunique_int32(
    int64_t * csize,
    int32_t * c,
    int64_t asize,
    const int32_t * a,
    int64_t * adstidx,
    int64_t bsize,
    const int32_t * b,
    int64_t * bdstidx)
{
    if (c && adstidx && bdstidx) {
        int64_t i = 0, j = 0, k = 0;
        while (i < asize && j < bsize) {
            if (k >= *csize) return EINVAL;
            if (a[i] < b[j]) {
                adstidx[i] = k; c[k++] = a[i];
                for (i++; i < asize && a[i] == a[i-1]; i++) { adstidx[i] = adstidx[i-1]; }
            } else if (a[i] > b[j]) {
                bdstidx[j] = k; c[k++] = b[j];
                for (j++; j < bsize && b[j] == b[j-1]; j++) { bdstidx[j] = bdstidx[j-1]; }
            } else {
                adstidx[i] = bdstidx[j] = k; c[k++] = a[i];
                for (i++; i < asize && a[i] == a[i-1]; i++) { adstidx[i] = adstidx[i-1]; }
                for (j++; j < bsize && b[j] == b[j-1]; j++) { bdstidx[j] = bdstidx[j-1]; }
            }
        }
        if (asize - i >= *csize) return EINVAL;
        while (i < asize) {
            if (k >= *csize) return EINVAL;
            adstidx[i] = k; c[k++] = a[i];
            for (i++; i < asize && a[i] == a[i-1]; i++) { adstidx[i] = adstidx[i-1]; }
        }
        if (bsize - j >= *csize) return EINVAL;
        while (j < bsize) {
            if (k >= *csize) return EINVAL;
            bdstidx[j] = k; c[k++] = b[j];
            for (j++; j < bsize && b[j] == b[j-1]; j++) { bdstidx[j] = bdstidx[j-1]; }
        }
        *csize = k;
    } else if (c) {
        int64_t i = 0, j = 0, k = 0;
        while (i < asize && j < bsize) {
            if (k >= *csize) return EINVAL;
            if (a[i] < b[j]) {
                c[k++] = a[i];
                for (i++; i < asize && a[i] == a[i-1]; i++) {}
            } else if (a[i] > b[j]) {
                c[k++] = b[j];
                for (j++; j < bsize && b[j] == b[j-1]; j++) {}
            } else {
                c[k++] = a[i];
                for (i++; i < asize && a[i] == a[i-1]; i++) {}
                for (j++; j < bsize && b[j] == b[j-1]; j++) {}
            }
        }
        if (asize - i >= *csize) return EINVAL;
        while (i < asize) {
            if (k >= *csize) return EINVAL;
            c[k++] = a[i];
            for (i++; i < asize && a[i] == a[i-1]; i++) {}
        }
        if (bsize - j >= *csize) return EINVAL;
        while (j < bsize) {
            if (k >= *csize) return EINVAL;
            c[k++] = b[j];
            for (j++; j < bsize && b[j] == b[j-1]; j++) {}
        }
        *csize = k;
    } else {
        int64_t i = 0, j = 0, k = 0;
        while (i < asize && j < bsize) {
            if (a[i] < b[j]) {
                k++;
                for (i++; i < asize && a[i] == a[i-1]; i++) {}
            } else if (a[i] > b[j]) {
                k++;
                for (j++; j < bsize && b[j] == b[j-1]; j++) {}
            } else {
                k++;
                for (i++; i < asize && a[i] == a[i-1]; i++) {}
                for (j++; j < bsize && b[j] == b[j-1]; j++) {}
            }
        }
        while (i < asize) { k++; for (i++; i < asize && a[i] == a[i-1]; i++) {} }
        while (j < bsize) { k++; for (j++; j < bsize && b[j] == b[j-1]; j++) {} }
        *csize = k;
    }
    return 0;
}

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
 *
 * Returns ‘0’ if successful, or ‘EINVAL’ if the output array is not
 * large enough.
 */
int setunion_sorted_nonunique_int64(
    int64_t * csize,
    int64_t * c,
    int64_t asize,
    const int64_t * a,
    int64_t * adstidx,
    int64_t bsize,
    const int64_t * b,
    int64_t * bdstidx)
{
    if (c && adstidx && bdstidx) {
        int64_t i = 0, j = 0, k = 0;
        while (i < asize && j < bsize) {
            if (k >= *csize) return EINVAL;
            if (a[i] < b[j]) {
                adstidx[i] = k; c[k++] = a[i];
                for (i++; i < asize && a[i] == a[i-1]; i++) { adstidx[i] = adstidx[i-1]; }
            } else if (a[i] > b[j]) {
                bdstidx[j] = k; c[k++] = b[j];
                for (j++; j < bsize && b[j] == b[j-1]; j++) { bdstidx[j] = bdstidx[j-1]; }
            } else {
                adstidx[i] = bdstidx[j] = k; c[k++] = a[i];
                for (i++; i < asize && a[i] == a[i-1]; i++) { adstidx[i] = adstidx[i-1]; }
                for (j++; j < bsize && b[j] == b[j-1]; j++) { bdstidx[j] = bdstidx[j-1]; }
            }
        }
        if (asize - i >= *csize) return EINVAL;
        while (i < asize) {
            if (k >= *csize) return EINVAL;
            adstidx[i] = k; c[k++] = a[i];
            for (i++; i < asize && a[i] == a[i-1]; i++) { adstidx[i] = adstidx[i-1]; }
        }
        if (bsize - j >= *csize) return EINVAL;
        while (j < bsize) {
            if (k >= *csize) return EINVAL;
            bdstidx[j] = k; c[k++] = b[j];
            for (j++; j < bsize && b[j] == b[j-1]; j++) { bdstidx[j] = bdstidx[j-1]; }
        }
        *csize = k;
    } else if (c) {
        int64_t i = 0, j = 0, k = 0;
        while (i < asize && j < bsize) {
            if (k >= *csize) return EINVAL;
            if (a[i] < b[j]) {
                c[k++] = a[i];
                for (i++; i < asize && a[i] == a[i-1]; i++) {}
            } else if (a[i] > b[j]) {
                c[k++] = b[j];
                for (j++; j < bsize && b[j] == b[j-1]; j++) {}
            } else {
                c[k++] = a[i];
                for (i++; i < asize && a[i] == a[i-1]; i++) {}
                for (j++; j < bsize && b[j] == b[j-1]; j++) {}
            }
        }
        if (asize - i >= *csize) return EINVAL;
        while (i < asize) {
            if (k >= *csize) return EINVAL;
            c[k++] = a[i];
            for (i++; i < asize && a[i] == a[i-1]; i++) {}
        }
        if (bsize - j >= *csize) return EINVAL;
        while (j < bsize) {
            if (k >= *csize) return EINVAL;
            c[k++] = b[j];
            for (j++; j < bsize && b[j] == b[j-1]; j++) {}
        }
        *csize = k;
    } else {
        int64_t i = 0, j = 0, k = 0;
        while (i < asize && j < bsize) {
            if (a[i] < b[j]) {
                k++;
                for (i++; i < asize && a[i] == a[i-1]; i++) {}
            } else if (a[i] > b[j]) {
                k++;
                for (j++; j < bsize && b[j] == b[j-1]; j++) {}
            } else {
                k++;
                for (i++; i < asize && a[i] == a[i-1]; i++) {}
                for (j++; j < bsize && b[j] == b[j-1]; j++) {}
            }
        }
        while (i < asize) { k++; for (i++; i < asize && a[i] == a[i-1]; i++) {} }
        while (j < bsize) { k++; for (j++; j < bsize && b[j] == b[j-1]; j++) {} }
        *csize = k;
    }
    return 0;
}

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
 *
 * Returns ‘0’ if successful, or ‘EINVAL’ if the output array is not
 * large enough.
 */
int setunion_sorted_nonunique_int(
    int64_t * csize,
    int * c,
    int64_t asize,
    const int * a,
    int64_t * adstidx,
    int64_t bsize,
    const int * b,
    int64_t * bdstidx)
{
    if (c && adstidx && bdstidx) {
        int64_t i = 0, j = 0, k = 0;
        while (i < asize && j < bsize) {
            if (k >= *csize) return EINVAL;
            if (a[i] < b[j]) {
                adstidx[i] = k; c[k++] = a[i];
                for (i++; i < asize && a[i] == a[i-1]; i++) { adstidx[i] = adstidx[i-1]; }
            } else if (a[i] > b[j]) {
                bdstidx[j] = k; c[k++] = b[j];
                for (j++; j < bsize && b[j] == b[j-1]; j++) { bdstidx[j] = bdstidx[j-1]; }
            } else {
                adstidx[i] = bdstidx[j] = k; c[k++] = a[i];
                for (i++; i < asize && a[i] == a[i-1]; i++) { adstidx[i] = adstidx[i-1]; }
                for (j++; j < bsize && b[j] == b[j-1]; j++) { bdstidx[j] = bdstidx[j-1]; }
            }
        }
        if (asize - i >= *csize) return EINVAL;
        while (i < asize) {
            if (k >= *csize) return EINVAL;
            adstidx[i] = k; c[k++] = a[i];
            for (i++; i < asize && a[i] == a[i-1]; i++) { adstidx[i] = adstidx[i-1]; }
        }
        if (bsize - j >= *csize) return EINVAL;
        while (j < bsize) {
            if (k >= *csize) return EINVAL;
            bdstidx[j] = k; c[k++] = b[j];
            for (j++; j < bsize && b[j] == b[j-1]; j++) { bdstidx[j] = bdstidx[j-1]; }
        }
        *csize = k;
    } else if (c) {
        int64_t i = 0, j = 0, k = 0;
        while (i < asize && j < bsize) {
            if (k >= *csize) return EINVAL;
            if (a[i] < b[j]) {
                c[k++] = a[i];
                for (i++; i < asize && a[i] == a[i-1]; i++) {}
            } else if (a[i] > b[j]) {
                c[k++] = b[j];
                for (j++; j < bsize && b[j] == b[j-1]; j++) {}
            } else {
                c[k++] = a[i];
                for (i++; i < asize && a[i] == a[i-1]; i++) {}
                for (j++; j < bsize && b[j] == b[j-1]; j++) {}
            }
        }
        if (asize - i >= *csize) return EINVAL;
        while (i < asize) {
            if (k >= *csize) return EINVAL;
            c[k++] = a[i];
            for (i++; i < asize && a[i] == a[i-1]; i++) {}
        }
        if (bsize - j >= *csize) return EINVAL;
        while (j < bsize) {
            if (k >= *csize) return EINVAL;
            c[k++] = b[j];
            for (j++; j < bsize && b[j] == b[j-1]; j++) {}
        }
        *csize = k;
    } else {
        int64_t i = 0, j = 0, k = 0;
        while (i < asize && j < bsize) {
            if (a[i] < b[j]) {
                k++;
                for (i++; i < asize && a[i] == a[i-1]; i++) {}
            } else if (a[i] > b[j]) {
                k++;
                for (j++; j < bsize && b[j] == b[j-1]; j++) {}
            } else {
                k++;
                for (i++; i < asize && a[i] == a[i-1]; i++) {}
                for (j++; j < bsize && b[j] == b[j-1]; j++) {}
            }
        }
        while (i < asize) { k++; for (i++; i < asize && a[i] == a[i-1]; i++) {} }
        while (j < bsize) { k++; for (j++; j < bsize && b[j] == b[j-1]; j++) {} }
        *csize = k;
    }
    return 0;
}

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
 *
 * Returns ‘0’ if successful, or ‘EINVAL’ if the output array is not
 * large enough.
 */
int setunion_unsorted_unique_int32(
    int64_t * csize,
    int32_t * c,
    int64_t asize,
    int32_t * a,
    int64_t * aperm,
    int64_t * adstidx,
    int64_t bsize,
    int32_t * b,
    int64_t * bperm,
    int64_t * bdstidx)
{
    int err = radix_sort_int32(asize, a, aperm);
    if (err) return err;
    err = radix_sort_int32(bsize, b, bperm);
    if (err) return err;
    return setunion_sorted_unique_int32(
        csize, c, asize, a, adstidx, bsize, b, bdstidx);
}

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
 *
 * Returns ‘0’ if successful, or ‘EINVAL’ if the output array is not
 * large enough.
 */
int setunion_unsorted_unique_int64(
    int64_t * csize,
    int64_t * c,
    int64_t asize,
    int64_t * a,
    int64_t * aperm,
    int64_t * adstidx,
    int64_t bsize,
    int64_t * b,
    int64_t * bperm,
    int64_t * bdstidx)
{
    int err = radix_sort_int64(asize, sizeof(*a), a, aperm);
    if (err) return err;
    err = radix_sort_int64(bsize, sizeof(*b), b, bperm);
    if (err) return err;
    return setunion_sorted_unique_int64(
        csize, c, asize, a, adstidx, bsize, b, bdstidx);
}

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
 *
 * Returns ‘0’ if successful, or ‘EINVAL’ if the output array is not
 * large enough.
 */
int setunion_unsorted_unique_int(
    int64_t * csize,
    int * c,
    int64_t asize,
    int * a,
    int64_t * aperm,
    int64_t * adstidx,
    int64_t bsize,
    int * b,
    int64_t * bperm,
    int64_t * bdstidx)
{
    int err = radix_sort_int(asize, a, aperm);
    if (err) return err;
    err = radix_sort_int(bsize, b, bperm);
    if (err) return err;
    return setunion_sorted_unique_int(
        csize, c, asize, a, adstidx, bsize, b, bdstidx);
}

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
 *
 * Returns ‘0’ if successful, or ‘EINVAL’ if the output array is not
 * large enough.
 */
int setunion_unsorted_nonunique_int32(
    int64_t * csize,
    int32_t * c,
    int64_t asize,
    int32_t * a,
    int64_t * aperm,
    int64_t * adstidx,
    int64_t bsize,
    int32_t * b,
    int64_t * bperm,
    int64_t * bdstidx)
{
    int err = radix_sort_int32(asize, a, aperm);
    if (err) return err;
    err = radix_sort_int32(bsize, b, bperm);
    if (err) return err;
    return setunion_sorted_nonunique_int32(
        csize, c, asize, a, adstidx, bsize, b, bdstidx);
}

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
 *
 * Returns ‘0’ if successful, or ‘EINVAL’ if the output array is not
 * large enough.
 */
int setunion_unsorted_nonunique_int64(
    int64_t * csize,
    int64_t * c,
    int64_t asize,
    int64_t * a,
    int64_t * aperm,
    int64_t * adstidx,
    int64_t bsize,
    int64_t * b,
    int64_t * bperm,
    int64_t * bdstidx)
{
    int err = radix_sort_int64(asize, sizeof(*a), a, aperm);
    if (err) return err;
    err = radix_sort_int64(bsize, sizeof(*b), b, bperm);
    if (err) return err;
    return setunion_sorted_nonunique_int64(
        csize, c, asize, a, adstidx, bsize, b, bdstidx);
}

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
 *
 * Returns ‘0’ if successful, or ‘EINVAL’ if the output array is not
 * large enough.
 */
int setunion_unsorted_nonunique_int(
    int64_t * csize,
    int * c,
    int64_t asize,
    int * a,
    int64_t * aperm,
    int64_t * adstidx,
    int64_t bsize,
    int * b,
    int64_t * bperm,
    int64_t * bdstidx)
{
    int err = radix_sort_int(asize, a, aperm);
    if (err) return err;
    err = radix_sort_int(bsize, b, bperm);
    if (err) return err;
    return setunion_sorted_nonunique_int(
        csize, c, asize, a, adstidx, bsize, b, bdstidx);
}

/*
 * set intersection operations on sorted arrays of unique values (i.e., no
 * duplicates)
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
 *
 * Returns ‘0’ if successful, or ‘EINVAL’ if the output array is not
 * large enough.
 */
int setintersection_sorted_unique_int32(
    int64_t * csize,
    int32_t * c,
    int64_t asize,
    const int32_t * a,
    int64_t * adstidx,
    int64_t bsize,
    const int32_t * b,
    int64_t * bdstidx)
{
    if (c && adstidx && bdstidx) {
        int64_t i = 0, j = 0, k = 0;
        while (i < asize && j < bsize) {
            if (a[i] < b[j]) { adstidx[i] = -1; i++; }
            else if (a[i] > b[j]) { bdstidx[j] = -1; j++; }
            else if (k < *csize) { adstidx[i] = bdstidx[j] = k; c[k++] = a[i]; i++; j++; }
            else { return EINVAL; }
        }
        *csize = k;
        while (i < asize) { adstidx[i++] = -1; }
        while (j < bsize) { bdstidx[j++] = -1; }
    } else if (c) {
        int64_t i = 0, j = 0, k = 0;
        while (i < asize && j < bsize) {
            if (a[i] < b[j]) { i++; }
            else if (a[i] > b[j]) { j++; }
            else if (k < *csize) { c[k++] = a[i]; i++; j++; }
            else { return EINVAL; }
        }
        *csize = k;
    } else {
        int64_t i = 0, j = 0, k = 0;
        while (i < asize && j < bsize) {
            if (a[i] < b[j]) { i++; }
            else if (a[i] > b[j]) { j++; }
            else { k++; i++; j++; }
        }
        *csize = k;
    }
    return 0;
}

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
 *
 * Returns ‘0’ if successful, or ‘EINVAL’ if the output array is not
 * large enough.
 */
int setintersection_sorted_unique_int64(
    int64_t * csize,
    int64_t * c,
    int64_t asize,
    const int64_t * a,
    int64_t * adstidx,
    int64_t bsize,
    const int64_t * b,
    int64_t * bdstidx)
{
    if (c && adstidx && bdstidx) {
        int64_t i = 0, j = 0, k = 0;
        while (i < asize && j < bsize) {
            if (a[i] < b[j]) { adstidx[i] = -1; i++; }
            else if (a[i] > b[j]) { bdstidx[j] = -1; j++; }
            else if (k < *csize) { adstidx[i] = bdstidx[j] = k; c[k++] = a[i]; i++; j++; }
            else { return EINVAL; }
        }
        *csize = k;
        while (i < asize) { adstidx[i++] = -1; }
        while (j < bsize) { bdstidx[j++] = -1; }
    } else if (c) {
        int64_t i = 0, j = 0, k = 0;
        while (i < asize && j < bsize) {
            if (a[i] < b[j]) { i++; }
            else if (a[i] > b[j]) { j++; }
            else if (k < *csize) { c[k++] = a[i]; i++; j++; }
            else { return EINVAL; }
        }
        *csize = k;
    } else {
        int64_t i = 0, j = 0, k = 0;
        while (i < asize && j < bsize) {
            if (a[i] < b[j]) { i++; }
            else if (a[i] > b[j]) { j++; }
            else { k++; i++; j++; }
        }
        *csize = k;
    }
    return 0;
}

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
 *
 * Returns ‘0’ if successful, or ‘EINVAL’ if the output array is not
 * large enough.
 */
int setintersection_sorted_unique_int(
    int64_t * csize,
    int * c,
    int64_t asize,
    const int * a,
    int64_t * adstidx,
    int64_t bsize,
    const int * b,
    int64_t * bdstidx)
{
    if (c && adstidx && bdstidx) {
        int64_t i = 0, j = 0, k = 0;
        while (i < asize && j < bsize) {
            if (a[i] < b[j]) { adstidx[i] = -1; i++; }
            else if (a[i] > b[j]) { bdstidx[j] = -1; j++; }
            else if (k < *csize) { adstidx[i] = bdstidx[j] = k; c[k++] = a[i]; i++; j++; }
            else { return EINVAL; }
        }
        *csize = k;
        while (i < asize) { adstidx[i++] = -1; }
        while (j < bsize) { bdstidx[j++] = -1; }
    } else if (c) {
        int64_t i = 0, j = 0, k = 0;
        while (i < asize && j < bsize) {
            if (a[i] < b[j]) { i++; }
            else if (a[i] > b[j]) { j++; }
            else if (k < *csize) { c[k++] = a[i]; i++; j++; }
            else { return EINVAL; }
        }
        *csize = k;
    } else {
        int64_t i = 0, j = 0, k = 0;
        while (i < asize && j < bsize) {
            if (a[i] < b[j]) { i++; }
            else if (a[i] > b[j]) { j++; }
            else { k++; i++; j++; }
        }
        *csize = k;
    }
    return 0;
}

/*
 * set intersection on sorted arrays, possibly containing non-unique values
 * (i.e., duplicates are allowed)
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
 *
 * Returns ‘0’ if successful, or ‘EINVAL’ if the output array is not
 * large enough.
 */
int setintersection_sorted_nonunique_int32(
    int64_t * csize,
    int32_t * c,
    int64_t asize,
    const int32_t * a,
    int64_t * adstidx,
    int64_t bsize,
    const int32_t * b,
    int64_t * bdstidx)
{
    if (c && adstidx && bdstidx) {
        int64_t i = 0, j = 0, k = 0;
        while (i < asize && j < bsize) {
            if (a[i] < b[j]) {
                adstidx[i] = -1;
                for (i++; i < asize && a[i] == a[i-1]; i++) { adstidx[i] = -1; }
            } else if (a[i] > b[j]) {
                bdstidx[j] = -1;
                for (j++; j < bsize && b[j] == b[j-1]; j++) { bdstidx[j] = -1; }
            } else if (k < *csize) {
                adstidx[i] = bdstidx[j] = k;
                c[k] = a[i];
                for (i++; i < asize && a[i] == a[i-1]; i++) { adstidx[i] = k; }
                for (j++; j < bsize && b[j] == b[j-1]; j++) { bdstidx[j] = k; }
                k++;
            } else { return EINVAL; }
        }
        *csize = k;
        while (i < asize) { adstidx[i++] = -1; }
        while (j < bsize) { bdstidx[j++] = -1; }
    } else if (c) {
        int64_t i = 0, j = 0, k = 0;
        while (i < asize && j < bsize) {
            if (a[i] < b[j]) {
                for (i++; i < asize && a[i] == a[i-1]; i++) {}
            } else if (a[i] > b[j]) {
                for (j++; j < bsize && b[j] == b[j-1]; j++) {}
            } else if (k < *csize) {
                c[k++] = a[i];
                for (i++; i < asize && a[i] == a[i-1]; i++) {}
                for (j++; j < bsize && b[j] == b[j-1]; j++) {}
            } else { return EINVAL; }
        }
        *csize = k;
    } else {
        int64_t i = 0, j = 0, k = 0;
        while (i < asize && j < bsize) {
            if (a[i] < b[j]) {
                for (i++; i < asize && a[i] == a[i-1]; i++) {}
            } else if (a[i] > b[j]) {
                for (j++; j < bsize && b[j] == b[j-1]; j++) {}
            } else {
                k++;
                for (i++; i < asize && a[i] == a[i-1]; i++) {}
                for (j++; j < bsize && b[j] == b[j-1]; j++) {}
            }
        }
        *csize = k;
    }
    return 0;
}

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
 *
 * Returns ‘0’ if successful, or ‘EINVAL’ if the output array is not
 * large enough.
 */
int setintersection_sorted_nonunique_int64(
    int64_t * csize,
    int64_t * c,
    int64_t asize,
    const int64_t * a,
    int64_t * adstidx,
    int64_t bsize,
    const int64_t * b,
    int64_t * bdstidx)
{
    if (c && adstidx && bdstidx) {
        int64_t i = 0, j = 0, k = 0;
        while (i < asize && j < bsize) {
            if (a[i] < b[j]) {
                adstidx[i] = -1;
                for (i++; i < asize && a[i] == a[i-1]; i++) { adstidx[i] = -1; }
            } else if (a[i] > b[j]) {
                bdstidx[j] = -1;
                for (j++; j < bsize && b[j] == b[j-1]; j++) { bdstidx[j] = -1; }
            } else if (k < *csize) {
                adstidx[i] = bdstidx[j] = k;
                c[k] = a[i];
                for (i++; i < asize && a[i] == a[i-1]; i++) { adstidx[i] = k; }
                for (j++; j < bsize && b[j] == b[j-1]; j++) { bdstidx[j] = k; }
                k++;
            } else { return EINVAL; }
        }
        *csize = k;
        while (i < asize) { adstidx[i++] = -1; }
        while (j < bsize) { bdstidx[j++] = -1; }
    } else if (c) {
        int64_t i = 0, j = 0, k = 0;
        while (i < asize && j < bsize) {
            if (a[i] < b[j]) {
                for (i++; i < asize && a[i] == a[i-1]; i++) {}
            } else if (a[i] > b[j]) {
                for (j++; j < bsize && b[j] == b[j-1]; j++) {}
            } else if (k < *csize) {
                c[k++] = a[i];
                for (i++; i < asize && a[i] == a[i-1]; i++) {}
                for (j++; j < bsize && b[j] == b[j-1]; j++) {}
            } else { return EINVAL; }
        }
        *csize = k;
    } else {
        int64_t i = 0, j = 0, k = 0;
        while (i < asize && j < bsize) {
            if (a[i] < b[j]) {
                for (i++; i < asize && a[i] == a[i-1]; i++) {}
            } else if (a[i] > b[j]) {
                for (j++; j < bsize && b[j] == b[j-1]; j++) {}
            } else {
                k++;
                for (i++; i < asize && a[i] == a[i-1]; i++) {}
                for (j++; j < bsize && b[j] == b[j-1]; j++) {}
            }
        }
        *csize = k;
    }
    return 0;
}

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
 *
 * Returns ‘0’ if successful, or ‘EINVAL’ if the output array is not
 * large enough.
 */
int setintersection_sorted_nonunique_int(
    int64_t * csize,
    int * c,
    int64_t asize,
    const int * a,
    int64_t * adstidx,
    int64_t bsize,
    const int * b,
    int64_t * bdstidx)
{
    if (c && adstidx && bdstidx) {
        int64_t i = 0, j = 0, k = 0;
        while (i < asize && j < bsize) {
            if (a[i] < b[j]) {
                adstidx[i] = -1;
                for (i++; i < asize && a[i] == a[i-1]; i++) { adstidx[i] = -1; }
            } else if (a[i] > b[j]) {
                bdstidx[j] = -1;
                for (j++; j < bsize && b[j] == b[j-1]; j++) { bdstidx[j] = -1; }
            } else if (k < *csize) {
                adstidx[i] = bdstidx[j] = k;
                c[k] = a[i];
                for (i++; i < asize && a[i] == a[i-1]; i++) { adstidx[i] = k; }
                for (j++; j < bsize && b[j] == b[j-1]; j++) { bdstidx[j] = k; }
                k++;
            } else { return EINVAL; }
        }
        *csize = k;
        while (i < asize) { adstidx[i++] = -1; }
        while (j < bsize) { bdstidx[j++] = -1; }
    } else if (c) {
        int64_t i = 0, j = 0, k = 0;
        while (i < asize && j < bsize) {
            if (a[i] < b[j]) {
                for (i++; i < asize && a[i] == a[i-1]; i++) {}
            } else if (a[i] > b[j]) {
                for (j++; j < bsize && b[j] == b[j-1]; j++) {}
            } else if (k < *csize) {
                c[k++] = a[i];
                for (i++; i < asize && a[i] == a[i-1]; i++) {}
                for (j++; j < bsize && b[j] == b[j-1]; j++) {}
            } else { return EINVAL; }
        }
        *csize = k;
    } else {
        int64_t i = 0, j = 0, k = 0;
        while (i < asize && j < bsize) {
            if (a[i] < b[j]) {
                for (i++; i < asize && a[i] == a[i-1]; i++) {}
            } else if (a[i] > b[j]) {
                for (j++; j < bsize && b[j] == b[j-1]; j++) {}
            } else {
                k++;
                for (i++; i < asize && a[i] == a[i-1]; i++) {}
                for (j++; j < bsize && b[j] == b[j-1]; j++) {}
            }
        }
        *csize = k;
    }
    return 0;
}

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
 *
 * Returns ‘0’ if successful, or ‘EINVAL’ if the output array is not
 * large enough.
 */
int setintersection_unsorted_unique_int32(
    int64_t * csize,
    int32_t * c,
    int64_t asize,
    int32_t * a,
    int64_t * aperm,
    int64_t * adstidx,
    int64_t bsize,
    int32_t * b,
    int64_t * bperm,
    int64_t * bdstidx)
{
    int err = radix_sort_int32(asize, a, aperm);
    if (err) return err;
    err = radix_sort_int32(bsize, b, bperm);
    if (err) return err;
    return setintersection_sorted_unique_int32(
        csize, c, asize, a, adstidx, bsize, b, bdstidx);
}

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
 *
 * Returns ‘0’ if successful, or ‘EINVAL’ if the output array is not
 * large enough.
 */
int setintersection_unsorted_unique_int64(
    int64_t * csize,
    int64_t * c,
    int64_t asize,
    int64_t * a,
    int64_t * aperm,
    int64_t * adstidx,
    int64_t bsize,
    int64_t * b,
    int64_t * bperm,
    int64_t * bdstidx)
{
    int err = radix_sort_int64(asize, sizeof(*a), a, aperm);
    if (err) return err;
    err = radix_sort_int64(bsize, sizeof(*b), b, bperm);
    if (err) return err;
    return setintersection_sorted_unique_int64(
        csize, c, asize, a, adstidx, bsize, b, bdstidx);
}

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
 *
 * Returns ‘0’ if successful, or ‘EINVAL’ if the output array is not
 * large enough.
 */
int setintersection_unsorted_unique_int(
    int64_t * csize,
    int * c,
    int64_t asize,
    int * a,
    int64_t * aperm,
    int64_t * adstidx,
    int64_t bsize,
    int * b,
    int64_t * bperm,
    int64_t * bdstidx)
{
    int err = radix_sort_int(asize, a, aperm);
    if (err) return err;
    err = radix_sort_int(bsize, b, bperm);
    if (err) return err;
    return setintersection_sorted_unique_int(
        csize, c, asize, a, adstidx, bsize, b, bdstidx);
}

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
 *
 * Returns ‘0’ if successful, or ‘EINVAL’ if the output array is not
 * large enough.
 */
int setintersection_unsorted_nonunique_int32(
    int64_t * csize,
    int32_t * c,
    int64_t asize,
    int32_t * a,
    int64_t * aperm,
    int64_t * adstidx,
    int64_t bsize,
    int32_t * b,
    int64_t * bperm,
    int64_t * bdstidx)
{
    int err = radix_sort_int32(asize, a, aperm);
    if (err) return err;
    err = radix_sort_int32(bsize, b, bperm);
    if (err) return err;
    return setintersection_sorted_nonunique_int32(
        csize, c, asize, a, adstidx, bsize, b, bdstidx);
}

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
 *
 * Returns ‘0’ if successful, or ‘EINVAL’ if the output array is not
 * large enough.
 */
int setintersection_unsorted_nonunique_int64(
    int64_t * csize,
    int64_t * c,
    int64_t asize,
    int64_t * a,
    int64_t * aperm,
    int64_t * adstidx,
    int64_t bsize,
    int64_t * b,
    int64_t * bperm,
    int64_t * bdstidx)
{
    int err = radix_sort_int64(asize, sizeof(*a), a, aperm);
    if (err) return err;
    err = radix_sort_int64(bsize, sizeof(*b), b, bperm);
    if (err) return err;
    return setintersection_sorted_nonunique_int64(
        csize, c, asize, a, adstidx, bsize, b, bdstidx);
}

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
 *
 * Returns ‘0’ if successful, or ‘EINVAL’ if the output array is not
 * large enough.
 */
int setintersection_unsorted_nonunique_int(
    int64_t * csize,
    int * c,
    int64_t asize,
    int * a,
    int64_t * aperm,
    int64_t * adstidx,
    int64_t bsize,
    int * b,
    int64_t * bperm,
    int64_t * bdstidx)
{
    int err = radix_sort_int(asize, a, aperm);
    if (err) return err;
    err = radix_sort_int(bsize, b, bperm);
    if (err) return err;
    return setintersection_sorted_nonunique_int(
        csize, c, asize, a, adstidx, bsize, b, bdstidx);
}

/*
 * set difference operations on sorted arrays of unique values (i.e., no
 * duplicates)
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
 *
 * Returns ‘0’ if successful, or ‘EINVAL’ if the output array is not
 * large enough.
 */
int setdifference_sorted_unique_int32(
    int64_t * csize,
    int32_t * c,
    int64_t asize,
    const int32_t * a,
    int64_t bsize,
    const int32_t * b)
{
    if (c) {
        int64_t i = 0, j = 0, k = 0;
        while (i < asize && j < bsize) {
            if (a[i] > b[j]) { j++; }
            else if (a[i] == b[j]) { i++; j++; }
            else if (a[i] < b[j] && k < *csize) { c[k++] = a[i++]; }
            else { return EINVAL; }
        }
        if (k + (asize - i) > *csize) return EINVAL;
        while (i < asize) { c[k++] = a[i++]; }
        *csize = k;
    } else {
        int64_t i = 0, j = 0, k = 0;
        while (i < asize && j < bsize) {
            if (a[i] > b[j]) { j++; }
            else if (a[i] == b[j]) { i++; j++; }
            else if (a[i] < b[j]) { k++; i++; }
        }
        while (i < asize) { k++; i++; }
        *csize = k;
    }
    return 0;
}

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
 *
 * Returns ‘0’ if successful, or ‘EINVAL’ if the output array is not
 * large enough.
 */
int setdifference_sorted_unique_int64(
    int64_t * csize,
    int64_t * c,
    int64_t asize,
    const int64_t * a,
    int64_t bsize,
    const int64_t * b)
{
    if (c) {
        int64_t i = 0, j = 0, k = 0;
        while (i < asize && j < bsize) {
            if (a[i] > b[j]) { j++; }
            else if (a[i] == b[j]) { i++; j++; }
            else if (a[i] < b[j] && k < *csize) { c[k++] = a[i++]; }
            else { return EINVAL; }
        }
        if (k + (asize - i) > *csize) return EINVAL;
        while (i < asize) { c[k++] = a[i++]; }
        *csize = k;
    } else {
        int64_t i = 0, j = 0, k = 0;
        while (i < asize && j < bsize) {
            if (a[i] > b[j]) { j++; }
            else if (a[i] == b[j]) { i++; j++; }
            else if (a[i] < b[j]) { k++; i++; }
        }
        while (i < asize) { k++; i++; }
        *csize = k;
    }
    return 0;
}

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
 *
 * Returns ‘0’ if successful, or ‘EINVAL’ if the output array is not
 * large enough.
 */
int setdifference_sorted_unique_int(
    int64_t * csize,
    int * c,
    int64_t asize,
    const int * a,
    int64_t bsize,
    const int * b)
{
    if (c) {
        int64_t i = 0, j = 0, k = 0;
        while (i < asize && j < bsize) {
            if (a[i] > b[j]) { j++; }
            else if (a[i] == b[j]) { i++; j++; }
            else if (a[i] < b[j] && k < *csize) { c[k++] = a[i++]; }
            else { return EINVAL; }
        }
        if (k + (asize - i) > *csize) return EINVAL;
        while (i < asize) { c[k++] = a[i++]; }
        *csize = k;
    } else {
        int64_t i = 0, j = 0, k = 0;
        while (i < asize && j < bsize) {
            if (a[i] > b[j]) { j++; }
            else if (a[i] == b[j]) { i++; j++; }
            else if (a[i] < b[j]) { k++; i++; }
        }
        while (i < asize) { k++; i++; }
        *csize = k;
    }
    return 0;
}

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
 *
 * Returns ‘0’ if successful, or ‘EINVAL’ if the output array is not
 * large enough.
 */
int setdifference_sorted_nonunique_int32(
    int64_t * csize,
    int32_t * c,
    int64_t asize,
    const int32_t * a,
    int64_t bsize,
    const int32_t * b)
{
    if (c) {
        int64_t i = 0, j = 0, k = 0;
        while (i < asize && j < bsize) {
            if (a[i] > b[j]) {
                for (j++; j < bsize && b[j] == b[j-1]; j++) {}
            } else if (a[i] == b[j]) {
                for (i++; i < asize && a[i] == a[i-1]; i++) {}
                for (j++; j < bsize && b[j] == b[j-1]; j++) {}
            } else if (a[i] < b[j] && k < *csize) {
                c[k++] = a[i];
                for (i++; i < asize && a[i] == a[i-1]; i++) {}
            } else { return EINVAL; }
        }
        while (i < asize) {
            if (k >= *csize) return EINVAL;
            c[k++] = a[i];
            for (i++; i < asize && a[i] == a[i-1]; i++) {}
        }
        *csize = k;
    } else {
        int64_t i = 0, j = 0, k = 0;
        while (i < asize && j < bsize) {
            if (a[i] > b[j]) {
                for (j++; j < bsize && b[j] == b[j-1]; j++) {}
            } else if (a[i] == b[j]) {
                for (i++; i < asize && a[i] == a[i-1]; i++) {}
                for (j++; j < bsize && b[j] == b[j-1]; j++) {}
            } else if (a[i] < b[j]) {
                k++;
                for (i++; i < asize && a[i] == a[i-1]; i++) {}
            }
        }
        while (i < asize) {
            k++;
            for (i++; i < asize && a[i] == a[i-1]; i++) {}
        }
        *csize = k;
    }
    return 0;
}

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
 *
 * Returns ‘0’ if successful, or ‘EINVAL’ if the output array is not
 * large enough.
 */
int setdifference_sorted_nonunique_int64(
    int64_t * csize,
    int64_t * c,
    int64_t asize,
    const int64_t * a,
    int64_t bsize,
    const int64_t * b)
{
    if (c) {
        int64_t i = 0, j = 0, k = 0;
        while (i < asize && j < bsize) {
            if (a[i] > b[j]) {
                for (j++; j < bsize && b[j] == b[j-1]; j++) {}
            } else if (a[i] == b[j]) {
                for (i++; i < asize && a[i] == a[i-1]; i++) {}
                for (j++; j < bsize && b[j] == b[j-1]; j++) {}
            } else if (a[i] < b[j] && k < *csize) {
                c[k++] = a[i];
                for (i++; i < asize && a[i] == a[i-1]; i++) {}
            } else { return EINVAL; }
        }
        while (i < asize) {
            if (k >= *csize) return EINVAL;
            c[k++] = a[i];
            for (i++; i < asize && a[i] == a[i-1]; i++) {}
        }
        *csize = k;
    } else {
        int64_t i = 0, j = 0, k = 0;
        while (i < asize && j < bsize) {
            if (a[i] > b[j]) {
                for (j++; j < bsize && b[j] == b[j-1]; j++) {}
            } else if (a[i] == b[j]) {
                for (i++; i < asize && a[i] == a[i-1]; i++) {}
                for (j++; j < bsize && b[j] == b[j-1]; j++) {}
            } else if (a[i] < b[j]) {
                k++;
                for (i++; i < asize && a[i] == a[i-1]; i++) {}
            }
        }
        while (i < asize) {
            k++;
            for (i++; i < asize && a[i] == a[i-1]; i++) {}
        }
        *csize = k;
    }
    return 0;
}

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
 *
 * Returns ‘0’ if successful, or ‘EINVAL’ if the output array is not
 * large enough.
 */
int setdifference_sorted_nonunique_int(
    int64_t * csize,
    int * c,
    int64_t asize,
    const int * a,
    int64_t bsize,
    const int * b)
{
    if (c) {
        int64_t i = 0, j = 0, k = 0;
        while (i < asize && j < bsize) {
            if (a[i] > b[j]) {
                for (j++; j < bsize && b[j] == b[j-1]; j++) {}
            } else if (a[i] == b[j]) {
                for (i++; i < asize && a[i] == a[i-1]; i++) {}
                for (j++; j < bsize && b[j] == b[j-1]; j++) {}
            } else if (a[i] < b[j] && k < *csize) {
                c[k++] = a[i];
                for (i++; i < asize && a[i] == a[i-1]; i++) {}
            } else { return EINVAL; }
        }
        while (i < asize) {
            if (k >= *csize) return EINVAL;
            c[k++] = a[i];
            for (i++; i < asize && a[i] == a[i-1]; i++) {}
        }
        *csize = k;
    } else {
        int64_t i = 0, j = 0, k = 0;
        while (i < asize && j < bsize) {
            if (a[i] > b[j]) {
                for (j++; j < bsize && b[j] == b[j-1]; j++) {}
            } else if (a[i] == b[j]) {
                for (i++; i < asize && a[i] == a[i-1]; i++) {}
                for (j++; j < bsize && b[j] == b[j-1]; j++) {}
            } else if (a[i] < b[j]) {
                k++;
                for (i++; i < asize && a[i] == a[i-1]; i++) {}
            }
        }
        while (i < asize) {
            k++;
            for (i++; i < asize && a[i] == a[i-1]; i++) {}
        }
        *csize = k;
    }
    return 0;
}

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
 *
 * Returns ‘0’ if successful, or ‘EINVAL’ if the output array is not
 * large enough.
 */
int setdifference_unsorted_unique_int32(
    int64_t * csize,
    int32_t * c,
    int64_t asize,
    int32_t * a,
    int64_t bsize,
    int32_t * b)
{
    int err = radix_sort_int32(asize, a, NULL);
    if (err) return err;
    err = radix_sort_int32(bsize, b, NULL);
    if (err) return err;
    return setdifference_sorted_unique_int32(csize, c, asize, a, bsize, b);
}

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
 *
 * Returns ‘0’ if successful, or ‘EINVAL’ if the output array is not
 * large enough.
 */
int setdifference_unsorted_unique_int64(
    int64_t * csize,
    int64_t * c,
    int64_t asize,
    int64_t * a,
    int64_t bsize,
    int64_t * b)
{
    int err = radix_sort_int64(asize, sizeof(*a), a, NULL);
    if (err) return err;
    err = radix_sort_int64(bsize, sizeof(*b), b, NULL);
    if (err) return err;
    return setdifference_sorted_unique_int64(csize, c, asize, a, bsize, b);
}

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
 *
 * Returns ‘0’ if successful, or ‘EINVAL’ if the output array is not
 * large enough.
 */
int setdifference_unsorted_unique_int(
    int64_t * csize,
    int * c,
    int64_t asize,
    int * a,
    int64_t bsize,
    int * b)
{
    int err = radix_sort_int(asize, a, NULL);
    if (err) return err;
    err = radix_sort_int(bsize, b, NULL);
    if (err) return err;
    return setdifference_sorted_unique_int(csize, c, asize, a, bsize, b);
}

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
 *
 * Returns ‘0’ if successful, or ‘EINVAL’ if the output array is not
 * large enough.
 */
int setdifference_unsorted_nonunique_int32(
    int64_t * csize,
    int32_t * c,
    int64_t asize,
    int32_t * a,
    int64_t bsize,
    int32_t * b)
{
    int err = radix_sort_int32(asize, a, NULL);
    if (err) return err;
    err = radix_sort_int32(bsize, b, NULL);
    if (err) return err;
    return setdifference_sorted_nonunique_int32(csize, c, asize, a, bsize, b);
}

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
 *
 * Returns ‘0’ if successful, or ‘EINVAL’ if the output array is not
 * large enough.
 */
int setdifference_unsorted_nonunique_int64(
    int64_t * csize,
    int64_t * c,
    int64_t asize,
    int64_t * a,
    int64_t bsize,
    int64_t * b)
{
    int err = radix_sort_int64(asize, sizeof(*a), a, NULL);
    if (err) return err;
    err = radix_sort_int64(bsize, sizeof(*b), b, NULL);
    if (err) return err;
    return setdifference_sorted_nonunique_int64(csize, c, asize, a, bsize, b);
}

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
 *
 * Returns ‘0’ if successful, or ‘EINVAL’ if the output array is not
 * large enough.
 */
int setdifference_unsorted_nonunique_int(
    int64_t * csize,
    int * c,
    int64_t asize,
    int * a,
    int64_t bsize,
    int * b)
{
    int err = radix_sort_int(asize, a, NULL);
    if (err) return err;
    err = radix_sort_int(bsize, b, NULL);
    if (err) return err;
    return setdifference_sorted_nonunique_int(csize, c, asize, a, bsize, b);
}
