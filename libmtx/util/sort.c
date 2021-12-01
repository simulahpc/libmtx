/* This file is part of libmtx.
 *
 * Copyright (C) 2021 James D. Trotter
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
 * Last modified: 2021-11-30
 *
 * Sorting.
 */

#include "sort.h"

#include <libmtx/error.h>

#include <errno.h>

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

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
 * If ‘sorting_permutation’ is ‘NULL’, then this argument is ignored
 * and a sorting permutation is not computed. Otherwise, it must point
 * to an array that holds enough storage for ‘size’ values of type
 * ‘int64_t’. On success, this array will contain the sorting
 * permutation.
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
    int64_t * sorting_permutation,
    int64_t bucketptr[257])
{
    bool alloc_bucketptr = !bucketptr;
    if (alloc_bucketptr) {
        bucketptr = malloc(257 * sizeof(int64_t));
        if (!bucketptr)
            return MTX_ERR_ERRNO;
    }

    /* 1. Count the number of keys in each bucket. */
    for (int j = 0; j <= 256; j++)
        bucketptr[j] = 0;
    for (int64_t i = 0; i < size; i++)
        bucketptr[keys[srcstride*i]+1]++;

    /* 2. Compute offset to first key in each bucket. */
    for (int j = 0; j < 256; j++)
        bucketptr[j+1] += bucketptr[j];

    /* 3. Sort the keys into their respective buckets. */
    if (sorted_keys && sorting_permutation) {
        for (int64_t i = 0; i < size; i++) {
            int64_t destidx = bucketptr[keys[srcstride*i]];
            sorted_keys[dststride*destidx] = keys[srcstride*i];
            sorting_permutation[i] = destidx;
            bucketptr[keys[srcstride*i]]++;
        }
    } else if (sorted_keys) {
        for (int64_t i = 0; i < size; i++) {
            int64_t destidx = bucketptr[keys[srcstride*i]];
            sorted_keys[dststride*destidx] = keys[srcstride*i];
            bucketptr[keys[srcstride*i]]++;
        }
    } else if (sorting_permutation) {
        for (int64_t i = 0; i < size; i++) {
            int64_t destidx = bucketptr[keys[srcstride*i]];
            sorting_permutation[i] = destidx;
            bucketptr[keys[srcstride*i]]++;
        }
    } else {
        /* Nothing to be done. */
    }

    /* 4. Adjust the offsets to the first key in each bucket, or free
     * the resources that were allocated temporarily. */
    if (alloc_bucketptr) {
        free(bucketptr);
    } else {
        for (int j = 256; j > 0; j--)
            bucketptr[j] = bucketptr[j-1];
        bucketptr[0] = 0;
    }
    return MTX_SUCCESS;
}

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
 * If ‘sorting_permutation’ is ‘NULL’, then this argument is ignored
 * and a sorting permutation is not computed. Otherwise, it must point
 * to an array that holds enough storage for ‘size’ values of type
 * ‘int64_t’. On success, this array will contain the sorting
 * permutation.
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
    int64_t * sorting_permutation,
    int64_t bucketptr[65537])
{
    bool alloc_bucketptr = !bucketptr;
    if (alloc_bucketptr) {
        bucketptr = malloc(65537 * sizeof(int64_t));
        if (!bucketptr)
            return MTX_ERR_ERRNO;
    }

    /* 1. Count the number of keys in each bucket. */
    for (int j = 0; j <= 65536; j++)
        bucketptr[j] = 0;
    for (int64_t i = 0; i < size; i++) {
        const uint16_t * src = (const uint16_t *) ((uint8_t *) keys + srcstride*i);
        bucketptr[(*src)+1]++;
    }

    /* 2. Compute offset to first key in each bucket. */
    for (int j = 0; j < 65536; j++)
        bucketptr[j+1] += bucketptr[j];

    /* 3. Sort the keys into their respective buckets. */
    if (sorted_keys && sorting_permutation) {
        for (int64_t i = 0; i < size; i++) {
            const uint16_t * src = (const uint16_t *) ((uint8_t *) keys + srcstride*i);
            int64_t destidx = bucketptr[*src]++;
            uint16_t * dest = (uint16_t *)((uint8_t *) sorted_keys+dststride*destidx);
            *dest = *src;
            sorting_permutation[i] = destidx;
        }
    } else if (sorted_keys) {
        for (int64_t i = 0; i < size; i++) {
            const uint16_t * src = (const uint16_t *) ((uint8_t *) keys + srcstride*i);
            int64_t destidx = bucketptr[*src]++;
            uint16_t * dest = (uint16_t *)((uint8_t *) sorted_keys+dststride*destidx);
            *dest = *src;
        }
    } else if (sorting_permutation) {
        for (int64_t i = 0; i < size; i++) {
            const uint16_t * src = (const uint16_t *) ((uint8_t *) keys + srcstride*i);
            int64_t destidx = bucketptr[*src]++;
            sorting_permutation[i] = destidx;
        }
    } else {
        /* Nothing to be done. */
    }

    /* 4. Adjust the offsets to the first key in each bucket, or free
     * the resources that were allocated temporarily. */
    if (alloc_bucketptr) {
        free(bucketptr);
    } else {
        for (int j = 65536; j > 0; j--)
            bucketptr[j] = bucketptr[j-1];
        bucketptr[0] = 0;
    }
    return MTX_SUCCESS;
}

int counting_sort_uint8_uint32(
    int64_t size,
    int stride,
    const uint8_t * keys,
    const uint32_t * values,
    uint32_t * sorted_values,
    int64_t * sorting_permutation,
    int64_t bucketptr[257])
{
    bool alloc_bucketptr = !bucketptr;
    if (alloc_bucketptr) {
        bucketptr = malloc(257 * sizeof(int64_t));
        if (!bucketptr)
            return MTX_ERR_ERRNO;
    }

    /* 1. Count the number of keys in each bucket. */
    for (int j = 0; j <= 256; j++)
        bucketptr[j] = 0;
    for (int64_t i = 0; i < size; i++)
        bucketptr[keys[stride*i]+1]++;

    /* 2. Compute offset to first key in each bucket. */
    for (int j = 0; j < 256; j++)
        bucketptr[j+1] += bucketptr[j];

    /* 3. Sort the keys into their respective buckets. */
    if (values && sorted_values && sorting_permutation) {
        for (int64_t i = 0; i < size; i++) {
            int64_t destidx = bucketptr[keys[stride*i]]++;
            sorted_values[destidx] = values[i];
            sorting_permutation[i] = destidx;
        }
    } else if (values && sorted_values) {
        for (int64_t i = 0; i < size; i++) {
            int64_t destidx = bucketptr[keys[stride*i]]++;
            sorted_values[destidx] = values[i];
        }
    } else if (sorting_permutation) {
        for (int64_t i = 0; i < size; i++) {
            int64_t destidx = bucketptr[keys[stride*i]]++;
            sorting_permutation[i] = destidx;
        }
    } else {
        /* Nothing to be done. */
    }

    /* 4. Adjust the offsets to the first key in each bucket, or free
     * the resources that were allocated temporarily. */
    if (alloc_bucketptr) {
        free(bucketptr);
    } else {
        for (int j = 256; j > 0; j--)
            bucketptr[j] = bucketptr[j-1];
        bucketptr[0] = 0;
    }
    return MTX_SUCCESS;
}

int counting_sort_uint8_values(
    int64_t size,
    int stride,
    const uint8_t * keys,
    int valuesize,
    const void * values,
    void * sorted_values,
    int64_t bucketptr[257])
{
    bool alloc_bucketptr = !bucketptr;
    if (alloc_bucketptr) {
        bucketptr = malloc(257 * sizeof(int64_t));
        if (!bucketptr)
            return MTX_ERR_ERRNO;
    }

    /* 1. Count the number of keys in each bucket. */
    for (int j = 0; j <= 256; j++)
        bucketptr[j] = 0;
    for (int64_t i = 0; i < size; i++)
        bucketptr[keys[stride*i]+1]++;

    /* 2. Compute offset to first key in each bucket. */
    for (int j = 0; j < 256; j++)
        bucketptr[j+1] += bucketptr[j];

    /* 3. Sort the keys into their respective buckets. */
    if (values && sorted_values) {
        for (int64_t i = 0; i < size; i++) {
            int64_t destidx = bucketptr[keys[stride*i]]++;
            memcpy((void *) ((uintptr_t) sorted_values + valuesize * destidx),
                   (const void *) ((uintptr_t) values + valuesize * i),
                   valuesize);
        }
    }

    /* 4. Adjust the offsets to the first key in each bucket, or free
     * the resources that were allocated temporarily. */
    if (alloc_bucketptr) {
        free(bucketptr);
    } else {
        for (int j = 256; j > 0; j--)
            bucketptr[j] = bucketptr[j-1];
        bucketptr[0] = 0;
    }
    return MTX_SUCCESS;
}

/**
 * ‘radix_sort_uint32()’ sorts an array of 32-bit unsigned integers in
 * ascending order using a radix sort algorithm.
 *
 * Note that the keys are internally converted from host to network
 * byte order before sorting to guarantee that the sorting is the same
 * regardless of endianness. The keys are converted back to host order
 * before the function returns.
 *
 * The number of keys to sort is given by ‘size’, and the unsorted,
 * integer keys are given in the array ‘keys’. On success, the same
 * array will contain the keys in sorted order.
 *
 * If ‘sorting_permutation’ is ‘NULL’, then this argument is ignored
 * and a sorting permutation is not computed. Otherwise, it must point
 * to an array that holds enough storage for ‘size’ values of type
 * ‘int64_t’. On success, this array will contain the sorting
 * permutation, mapping the locations of the original, unsorted keys
 * to their new locations in the sorted array.
 */
int radix_sort_uint32(
    int64_t size,
    uint32_t * keys,
    int64_t * sorting_permutation)
{
    int err;
    uint32_t * extra_keys = malloc(size * sizeof(uint32_t));
    if (!extra_keys)
        return MTX_ERR_ERRNO;

    int64_t * bucketptr = malloc(257 * sizeof(int64_t));
    if (!bucketptr) {
        free(extra_keys);
        return MTX_ERR_ERRNO;
    }

    int64_t * extra_sorting_permutation = NULL;
    if (sorting_permutation) {
        extra_sorting_permutation = malloc(size * sizeof(int64_t));
        if (!extra_sorting_permutation) {
            free(bucketptr);
            free(extra_keys);
            return MTX_ERR_ERRNO;
        }
    }

    /* Perform one round of sorting for each digit in a key */
    for (int k = 0; k < 4; k++) {
        /* 1. Count the number of keys in each bucket. */
        for (int j = 0; j <= 256; j++)
            bucketptr[j] = 0;
        for (int64_t i = 0; i < size; i++)
            bucketptr[((keys[i] >> (8*k)) & 0xff)+1]++;

        /* 2. Compute offset to first key in each bucket. */
        for (int j = 0; j < 256; j++)
            bucketptr[j+1] += bucketptr[j];

        /* 3. Sort the keys into their respective buckets. */
        if (sorting_permutation && k == 0) {
            for (int64_t i = 0; i < size; i++) {
                int64_t destidx = bucketptr[((keys[i] >> (8*k)) & 0xff)]++;
                extra_keys[destidx] = keys[i];
                sorting_permutation[destidx] = i;
            }
        } else if (sorting_permutation) {
            for (int64_t i = 0; i < size; i++) {
                int64_t destidx = bucketptr[((keys[i] >> (8*k)) & 0xff)]++;
                extra_keys[destidx] = keys[i];
                extra_sorting_permutation[destidx] = sorting_permutation[i];
            }
        } else {
            for (int64_t i = 0; i < size; i++) {
                int64_t destidx = bucketptr[((keys[i] >> (8*k)) & 0xff)]++;
                extra_keys[destidx] = keys[i];
            }
        }

        /* 4. Copy data needed for the next round. */
        if (sorting_permutation && k == 3) {
            for (int64_t j = 0; j < size; j++) {
                keys[j] = extra_keys[j];
                sorting_permutation[extra_sorting_permutation[j]] = j;
            }
        } else if (sorting_permutation && k > 0) {
            for (int64_t j = 0; j < size; j++) {
                keys[j] = extra_keys[j];
                sorting_permutation[j] = extra_sorting_permutation[j];
            }
        } else {
            for (int64_t j = 0; j < size; j++)
                keys[j] = extra_keys[j];
        }
    }

    if (sorting_permutation)
        free(extra_sorting_permutation);
    free(bucketptr);
    free(extra_keys);
    return MTX_SUCCESS;
}

/**
 * ‘radix_sort_uint64()’ sorts an array of 64-bit unsigned integers in
 * ascending order using a radix sort algorithm.
 *
 * Note that the keys are internally converted from host to network
 * byte order before sorting to guarantee that the sorting is the same
 * regardless of endianness. The keys are converted back to host order
 * before the function returns.
 *
 * The number of keys to sort is given by ‘size’, and the unsorted,
 * integer keys are given in the array ‘keys’. On success, the same
 * array will contain the keys in sorted order.
 *
 * If ‘sorting_permutation’ is ‘NULL’, then this argument is ignored
 * and a sorting permutation is not computed. Otherwise, it must point
 * to an array that holds enough storage for ‘size’ values of type
 * ‘int64_t’. On success, this array will contain the sorting
 * permutation, mapping the locations of the original, unsorted keys
 * to their new locations in the sorted array.
 */
int radix_sort_uint64(
    int64_t size,
    uint64_t * keys,
    int64_t * sorting_permutation)
{
    int err;
    uint64_t * extra_keys = malloc(size * sizeof(uint64_t));
    if (!extra_keys)
        return MTX_ERR_ERRNO;

    int64_t * bucketptr = malloc(257 * sizeof(int64_t));
    if (!bucketptr) {
        free(extra_keys);
        return MTX_ERR_ERRNO;
    }

    int64_t * extra_sorting_permutation = NULL;
    if (sorting_permutation) {
        extra_sorting_permutation = malloc(size * sizeof(int64_t));
        if (!extra_sorting_permutation) {
            free(bucketptr);
            free(extra_keys);
            return MTX_ERR_ERRNO;
        }
    }

    /* Perform one round of sorting for each digit in a key */
    for (int k = 0; k < 8; k++) {
        /* 1. Count the number of keys in each bucket. */
        for (int j = 0; j <= 256; j++)
            bucketptr[j] = 0;
        for (int64_t i = 0; i < size; i++)
            bucketptr[((keys[i] >> (8*k)) & 0xff)+1]++;

        /* 2. Compute offset to first key in each bucket. */
        for (int j = 0; j < 256; j++)
            bucketptr[j+1] += bucketptr[j];

        /* 3. Sort the keys into their respective buckets. */
        if (sorting_permutation && k == 0) {
            for (int64_t i = 0; i < size; i++) {
                int64_t destidx = bucketptr[((keys[i] >> (8*k)) & 0xff)]++;
                extra_keys[destidx] = keys[i];
                sorting_permutation[destidx] = i;
            }
        } else if (sorting_permutation) {
            for (int64_t i = 0; i < size; i++) {
                int64_t destidx = bucketptr[((keys[i] >> (8*k)) & 0xff)]++;
                extra_keys[destidx] = keys[i];
                extra_sorting_permutation[destidx] = sorting_permutation[i];
            }
        } else {
            for (int64_t i = 0; i < size; i++) {
                int64_t destidx = bucketptr[((keys[i] >> (8*k)) & 0xff)]++;
                extra_keys[destidx] = keys[i];
            }
        }

        /* 4. Copy data needed for the next round. */
        if (sorting_permutation && k == 7) {
            for (int64_t j = 0; j < size; j++) {
                keys[j] = extra_keys[j];
                sorting_permutation[extra_sorting_permutation[j]] = j;
            }
        } else if (sorting_permutation && k > 0) {
            for (int64_t j = 0; j < size; j++) {
                keys[j] = extra_keys[j];
                sorting_permutation[j] = extra_sorting_permutation[j];
            }
        } else {
            for (int64_t j = 0; j < size; j++)
                keys[j] = extra_keys[j];
        }
    }

    if (sorting_permutation)
        free(extra_sorting_permutation);
    free(bucketptr);
    free(extra_keys);
    return MTX_SUCCESS;
}
