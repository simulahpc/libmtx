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
 * Merging arrays.
 */

#include "merge.h"

#include <libmtx/libmtx-config.h>

#include <libmtx/error.h>

#include <errno.h>

#include <stdint.h>

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
    const int32_t * b)
{
    if (csize < asize + bsize) return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    int64_t i = 0, j = 0, k = 0;
    while (i < asize && j < bsize) {
        if (a[i] < b[j]) c[k++] = a[i++];
        else if (a[i] > b[j]) c[k++] = b[j++];
        else { c[k++] = a[i++]; }
    }
    while (i < asize) c[k++] = a[i++];
    while (j < bsize) c[k++] = b[j++];
    return MTX_SUCCESS;
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
 */
int merge_sorted_int64(
    int64_t csize,
    int64_t * c,
    int64_t asize,
    const int64_t * a,
    int64_t bsize,
    const int64_t * b)
{
    if (csize < asize + bsize) return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    int64_t i = 0, j = 0, k = 0;
    while (i < asize && j < bsize) {
        if (a[i] < b[j]) c[k++] = a[i++];
        else if (a[i] > b[j]) c[k++] = b[j++];
        else { c[k++] = a[i++]; }
    }
    while (i < asize) c[k++] = a[i++];
    while (j < bsize) c[k++] = b[j++];
    return MTX_SUCCESS;
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
 */
int merge_sorted_int(
    int64_t csize,
    int * c,
    int64_t asize,
    const int * a,
    int64_t bsize,
    const int * b)
{
    if (csize < asize + bsize) return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    int64_t i = 0, j = 0, k = 0;
    while (i < asize && j < bsize) {
        if (a[i] < b[j]) c[k++] = a[i++];
        else if (a[i] > b[j]) c[k++] = b[j++];
        else { c[k++] = a[i++]; }
    }
    while (i < asize) c[k++] = a[i++];
    while (j < bsize) c[k++] = b[j++];
    return MTX_SUCCESS;
}

/**
 * ‘merge_sorted_union_int32()’ merges two sorted arrays of 32-bit
 * signed integers based on a set union operation.
 *
 * The two arrays to be merged, ‘a’ and ‘b’, contain ‘asize’ and
 * ‘bsize’ items, respectively. The arrays must be sorted in ascending
 * order and values may appear more than once.
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
int merge_sorted_union_int32(
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
            if (k >= *csize) return MTX_ERR_INDEX_OUT_OF_BOUNDS;
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
        if (asize - i >= *csize) return MTX_ERR_INDEX_OUT_OF_BOUNDS;
        while (i < asize) {
            if (k >= *csize) return MTX_ERR_INDEX_OUT_OF_BOUNDS;
            c[k++] = a[i];
            for (i++; i < asize && a[i] == a[i-1]; i++) {}
        }
        if (bsize - j >= *csize) return MTX_ERR_INDEX_OUT_OF_BOUNDS;
        while (j < bsize) {
            if (k >= *csize) return MTX_ERR_INDEX_OUT_OF_BOUNDS;
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
    return MTX_SUCCESS;
}

/**
 * ‘merge_sorted_union_int64()’ merges two sorted arrays of 64-bit
 * signed integers based on a set union operation.
 *
 * The two arrays to be merged, ‘a’ and ‘b’, contain ‘asize’ and
 * ‘bsize’ items, respectively. The arrays must be sorted in ascending
 * order and values may appear more than once.
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
int merge_sorted_union_int64(
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
            if (k >= *csize) return MTX_ERR_INDEX_OUT_OF_BOUNDS;
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
        if (asize - i >= *csize) return MTX_ERR_INDEX_OUT_OF_BOUNDS;
        while (i < asize) {
            if (k >= *csize) return MTX_ERR_INDEX_OUT_OF_BOUNDS;
            c[k++] = a[i];
            for (i++; i < asize && a[i] == a[i-1]; i++) {}
        }
        if (bsize - j >= *csize) return MTX_ERR_INDEX_OUT_OF_BOUNDS;
        while (j < bsize) {
            if (k >= *csize) return MTX_ERR_INDEX_OUT_OF_BOUNDS;
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
    return MTX_SUCCESS;
}

/**
 * ‘merge_sorted_union_int()’ merges two sorted arrays of 32-bit
 * signed integers based on a set union operation.
 *
 * The two arrays to be merged, ‘a’ and ‘b’, contain ‘asize’ and
 * ‘bsize’ items, respectively. The arrays must be sorted in ascending
 * order and values may appear more than once.
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
int merge_sorted_union_int(
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
            if (k >= *csize) return MTX_ERR_INDEX_OUT_OF_BOUNDS;
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
        if (asize - i >= *csize) return MTX_ERR_INDEX_OUT_OF_BOUNDS;
        while (i < asize) {
            if (k >= *csize) return MTX_ERR_INDEX_OUT_OF_BOUNDS;
            c[k++] = a[i];
            for (i++; i < asize && a[i] == a[i-1]; i++) {}
        }
        if (bsize - j >= *csize) return MTX_ERR_INDEX_OUT_OF_BOUNDS;
        while (j < bsize) {
            if (k >= *csize) return MTX_ERR_INDEX_OUT_OF_BOUNDS;
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
    return MTX_SUCCESS;
}

/**
 * ‘merge_sorted_intersection_int32()’ merges two sorted arrays of
 * 32-bit signed integers based on a set intersection operation.
 *
 * The two arrays to be merged, ‘a’ and ‘b’, contain ‘asize’ and
 * ‘bsize’ items, respectively. The arrays must be sorted in ascending
 * order and values may appear more than once.
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
int merge_sorted_intersection_int32(
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
            if (a[i] < b[j]) {
                for (i++; i < asize && a[i] == a[i-1]; i++) {}
            } else if (a[i] > b[j]) {
                for (j++; j < bsize && b[j] == b[j-1]; j++) {}
            } else if (k < *csize) {
                c[k++] = a[i];
                for (i++; i < asize && a[i] == a[i-1]; i++) {}
                for (j++; j < bsize && b[j] == b[j-1]; j++) {}
            } else { return MTX_ERR_INDEX_OUT_OF_BOUNDS; }
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
    return MTX_SUCCESS;
}

/**
 * ‘merge_sorted_intersection_int64()’ merges two sorted arrays of
 * 64-bit signed integers based on a set intersection operation.
 *
 * The two arrays to be merged, ‘a’ and ‘b’, contain ‘asize’ and
 * ‘bsize’ items, respectively. The arrays must be sorted in ascending
 * order and values may appear more than once.
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
int merge_sorted_intersection_int64(
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
            if (a[i] < b[j]) {
                for (i++; i < asize && a[i] == a[i-1]; i++) {}
            } else if (a[i] > b[j]) {
                for (j++; j < bsize && b[j] == b[j-1]; j++) {}
            } else if (k < *csize) {
                c[k++] = a[i];
                for (i++; i < asize && a[i] == a[i-1]; i++) {}
                for (j++; j < bsize && b[j] == b[j-1]; j++) {}
            } else { return MTX_ERR_INDEX_OUT_OF_BOUNDS; }
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
    return MTX_SUCCESS;
}

/**
 * ‘merge_sorted_intersection_int()’ merges two sorted arrays of
 * signed integers based on a set intersection operation.
 *
 * The two arrays to be merged, ‘a’ and ‘b’, contain ‘asize’ and
 * ‘bsize’ items, respectively. The arrays must be sorted in ascending
 * order and values may appear more than once.
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
int merge_sorted_intersection_int(
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
            if (a[i] < b[j]) {
                for (i++; i < asize && a[i] == a[i-1]; i++) {}
            } else if (a[i] > b[j]) {
                for (j++; j < bsize && b[j] == b[j-1]; j++) {}
            } else if (k < *csize) {
                c[k++] = a[i];
                for (i++; i < asize && a[i] == a[i-1]; i++) {}
                for (j++; j < bsize && b[j] == b[j-1]; j++) {}
            } else { return MTX_ERR_INDEX_OUT_OF_BOUNDS; }
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
    return MTX_SUCCESS;
}

/**
 * ‘merge_sorted_difference_int32()’ merges two sorted arrays of
 * 32-bit signed integers based on a set difference operation.
 *
 * The two arrays to be merged, ‘a’ and ‘b’, contain ‘asize’ and
 * ‘bsize’ items, respectively. The arrays must be sorted in ascending
 * order and values may appear more than once.
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
int merge_sorted_difference_int32(
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
            } else { return MTX_ERR_INDEX_OUT_OF_BOUNDS; }
        }
        while (i < asize) {
            if (k >= *csize) return MTX_ERR_INDEX_OUT_OF_BOUNDS;
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
    return MTX_SUCCESS;
}

/**
 * ‘merge_sorted_difference_int64()’ merges two sorted arrays of
 * 64-bit signed integers based on a set difference operation.
 *
 * The two arrays to be merged, ‘a’ and ‘b’, contain ‘asize’ and
 * ‘bsize’ items, respectively. The arrays must be sorted in ascending
 * order and values may appear more than once.
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
int merge_sorted_difference_int64(
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
            } else { return MTX_ERR_INDEX_OUT_OF_BOUNDS; }
        }
        while (i < asize) {
            if (k >= *csize) return MTX_ERR_INDEX_OUT_OF_BOUNDS;
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
    return MTX_SUCCESS;
}

/**
 * ‘merge_sorted_difference_int()’ merges two sorted arrays of signed
 * integers based on a set difference operation.
 *
 * The two arrays to be merged, ‘a’ and ‘b’, contain ‘asize’ and
 * ‘bsize’ items, respectively. The arrays must be sorted in ascending
 * order and values may appear more than once.
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
int merge_sorted_difference_int(
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
            } else { return MTX_ERR_INDEX_OUT_OF_BOUNDS; }
        }
        while (i < asize) {
            if (k >= *csize) return MTX_ERR_INDEX_OUT_OF_BOUNDS;
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
    return MTX_SUCCESS;
}
