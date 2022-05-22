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
 * Last modified: 2022-01-20
 *
 * Sorting.
 */

#include "sort.h"

#include <libmtx/libmtx-config.h>

#include <libmtx/error.h>

#ifdef LIBMTX_HAVE_MPI
#include <mpi.h>
#endif

#include <errno.h>

#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
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

static void swap_uint64_t(
    uint64_t ** a,
    uint64_t ** b)
{
    uint64_t * tmp = *a;
    *a = *b;
    *b = tmp;
}

static void swap_int64_t(
    int64_t ** a,
    int64_t ** b)
{
    int64_t * tmp = *a;
    *a = *b;
    *b = tmp;
}

/**
 * ‘radix_sort_uint64()’ sorts an array of 64-bit unsigned integers in
 * ascending order using a radix sort algorithm.
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
    uint64_t * extra_keys = malloc(size * sizeof(uint64_t));
    if (!extra_keys)
        return MTX_ERR_ERRNO;

    /* allocate six buckets to count occurrences and offsets for
     * 11-digit binary numbers. */
    int64_t * bucketptr = malloc(6*2049 * sizeof(int64_t));
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

    /* 1. Count the number of keys in each bucket. */
    for (int k = 0; k < 6; k++) {
        for (int j = 0; j <= 2048; j++) {
            bucketptr[k*2049+j] = 0;
        }
    }
    for (int64_t i = 0; i < size; i++) {
        bucketptr[0*2049+((keys[i] >> (11*0)) & 0x7ff)+1]++;
        bucketptr[1*2049+((keys[i] >> (11*1)) & 0x7ff)+1]++;
        bucketptr[2*2049+((keys[i] >> (11*2)) & 0x7ff)+1]++;
        bucketptr[3*2049+((keys[i] >> (11*3)) & 0x7ff)+1]++;
        bucketptr[4*2049+((keys[i] >> (11*4)) & 0x7ff)+1]++;
        bucketptr[5*2049+((keys[i] >> (11*5)) & 0x7ff)+1]++;
    }

    /* 2. Compute offset to first key in each bucket. */
    for (int k = 0; k < 6; k++) {
        for (int j = 0; j < 2048; j++) {
            bucketptr[k*2049+j+1] += bucketptr[k*2049+j];
        }
    }

    /*
     * 3. Sort in 6 rounds with 11 binary digits treated in each
     * round. Note that pointers to the original and auxiliary arrays
     * of keys (and sorting permutation) are swapped after each round.
     * There is an even number of swaps, so that the sorted keys (and
     * the final sorting permutation) end up in the original array.
     *
     * The choice of using 11 bits in each round is described in the
     * article "Radix Tricks" by Michael Herf, published online in
     * December 2001 at http://stereopsis.com/radix.html.
     */
    if (sorting_permutation) {
        for (int64_t i = 0; i < size; i++) {
            int64_t destidx = bucketptr[0*2049+((keys[i] >> (11*0)) & 0x7ff)]++;
            extra_keys[destidx] = keys[i];
            sorting_permutation[destidx] = i;
        }
        swap_uint64_t(&keys, &extra_keys);
        for (int64_t i = 0; i < size; i++) {
            int64_t destidx = bucketptr[1*2049+((keys[i] >> (11*1)) & 0x7ff)]++;
            extra_keys[destidx] = keys[i];
            extra_sorting_permutation[destidx] = sorting_permutation[i];
        }
        swap_uint64_t(&keys, &extra_keys);
        swap_int64_t(&sorting_permutation, &extra_sorting_permutation);
        for (int64_t i = 0; i < size; i++) {
            int64_t destidx = bucketptr[2*2049+((keys[i] >> (11*2)) & 0x7ff)]++;
            extra_keys[destidx] = keys[i];
            extra_sorting_permutation[destidx] = sorting_permutation[i];
        }
        swap_uint64_t(&keys, &extra_keys);
        swap_int64_t(&sorting_permutation, &extra_sorting_permutation);
        for (int64_t i = 0; i < size; i++) {
            int64_t destidx = bucketptr[3*2049+((keys[i] >> (11*3)) & 0x7ff)]++;
            extra_keys[destidx] = keys[i];
            extra_sorting_permutation[destidx] = sorting_permutation[i];
        }
        swap_uint64_t(&keys, &extra_keys);
        swap_int64_t(&sorting_permutation, &extra_sorting_permutation);
        for (int64_t i = 0; i < size; i++) {
            int64_t destidx = bucketptr[4*2049+((keys[i] >> (11*4)) & 0x7ff)]++;
            extra_keys[destidx] = keys[i];
            extra_sorting_permutation[destidx] = sorting_permutation[i];
        }
        swap_uint64_t(&keys, &extra_keys);
        swap_int64_t(&sorting_permutation, &extra_sorting_permutation);
        for (int64_t i = 0; i < size; i++) {
            int64_t destidx = bucketptr[5*2049+((keys[i] >> (11*5)) & 0x7ff)]++;
            extra_keys[destidx] = keys[i];
            extra_sorting_permutation[destidx] = sorting_permutation[i];
        }
        swap_uint64_t(&keys, &extra_keys);
        for (int64_t j = 0; j < size; j++)
            sorting_permutation[extra_sorting_permutation[j]] = j;
    } else {
        for (int64_t i = 0; i < size; i++) {
            int64_t destidx = bucketptr[0*2049+((keys[i] >> (11*0)) & 0x7ff)]++;
            extra_keys[destidx] = keys[i];
        }
        swap_uint64_t(&keys, &extra_keys);
        for (int64_t i = 0; i < size; i++) {
            int64_t destidx = bucketptr[1*2049+((keys[i] >> (11*1)) & 0x7ff)]++;
            extra_keys[destidx] = keys[i];
        }
        swap_uint64_t(&keys, &extra_keys);
        for (int64_t i = 0; i < size; i++) {
            int64_t destidx = bucketptr[2*2049+((keys[i] >> (11*2)) & 0x7ff)]++;
            extra_keys[destidx] = keys[i];
        }
        swap_uint64_t(&keys, &extra_keys);
        for (int64_t i = 0; i < size; i++) {
            int64_t destidx = bucketptr[3*2049+((keys[i] >> (11*3)) & 0x7ff)]++;
            extra_keys[destidx] = keys[i];
        }
        swap_uint64_t(&keys, &extra_keys);
        for (int64_t i = 0; i < size; i++) {
            int64_t destidx = bucketptr[4*2049+((keys[i] >> (11*4)) & 0x7ff)]++;
            extra_keys[destidx] = keys[i];
        }
        swap_uint64_t(&keys, &extra_keys);
        for (int64_t i = 0; i < size; i++) {
            int64_t destidx = bucketptr[5*2049+((keys[i] >> (11*5)) & 0x7ff)]++;
            extra_keys[destidx] = keys[i];
        }
        swap_uint64_t(&keys, &extra_keys);
    }

    if (sorting_permutation)
        free(extra_sorting_permutation);
    free(bucketptr);
    free(extra_keys);
    return MTX_SUCCESS;
}

/**
 * ‘radix_sort_int32()’ sorts an array of 32-bit (signed) integers in
 * ascending order using a radix sort algorithm.
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
int radix_sort_int32(
    int64_t size,
    int32_t * keys,
    int64_t * sorting_permutation)
{
    int err;
    for (int64_t i = 0; i < size; i++)
        keys[i] ^= INT32_MIN;
    err = radix_sort_uint32(size, (uint32_t *) keys, sorting_permutation);
    for (int64_t i = 0; i < size; i++)
        keys[i] ^= INT32_MIN;
    return err;
}

/**
 * ‘radix_sort_int64()’ sorts an array of 64-bit (signed) integers in
 * ascending order using a radix sort algorithm.
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
int radix_sort_int64(
    int64_t size,
    int64_t * keys,
    int64_t * sorting_permutation)
{
    int err;
    for (int64_t i = 0; i < size; i++)
        keys[i] ^= INT64_MIN;
    err = radix_sort_uint64(size, (uint64_t *) keys, sorting_permutation);
    for (int64_t i = 0; i < size; i++)
        keys[i] ^= INT64_MIN;
    return err;
}

/**
 * ‘radix_sort_int()’ sorts an array of (signed) integers in ascending
 * order using a radix sort algorithm.
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
int radix_sort_int(
    int64_t size,
    int * keys,
    int64_t * sorting_permutation)
{
    if (sizeof(int) == sizeof(int32_t)) {
        return radix_sort_int32(size, (int32_t *) keys, sorting_permutation);
    } else if (sizeof(int) == sizeof(int64_t)) {
        return radix_sort_int64(size, (int64_t *) keys, sorting_permutation);
    } else {
        return MTX_ERR_NOT_SUPPORTED;
    }
}

#ifdef LIBMTX_HAVE_MPI
/**
 * ‘distradix_sort_uint32()’ sorts a distributed array of 32-bit
 * unsigned integers in ascending order using a distributed radix sort
 * algorithm.
 *
 * The number of keys on the current process that need to be sorted is
 * given by ‘size’, and the unsorted, integer keys on the current
 * process are given in the array ‘keys’. On success, the same array
 * will contain ‘size’ keys in a globally sorted order.
 *
 * If ‘sorting_permutation’ is ‘NULL’, then this argument is ignored
 * and a sorting permutation is not computed. Otherwise, it must point
 * to an array that holds enough storage for ‘size’ values of type
 * ‘int64_t’ on each MPI process. On success, this array will contain
 * the sorting permutation, mapping the locations of the original,
 * unsorted keys to their new locations in the sorted array.
 */
int distradix_sort_uint32(
    int64_t size,
    uint32_t * keys,
    int64_t * sorting_permutation,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err;

    /* The current implementation can only sort at most ‘INT_MAX’ keys
     * on each process. */
    if (size > INT_MAX) errno = ERANGE;
    err = size > INT_MAX ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;

    int comm_size;
    disterr->mpierrcode = MPI_Comm_size(comm, &comm_size);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;
    int rank;
    disterr->mpierrcode = MPI_Comm_rank(comm, &rank);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;

    int64_t total_size;
    disterr->mpierrcode = MPI_Allreduce(
        &size, &total_size, 1, MPI_INT64_T, MPI_SUM, comm);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;

    int64_t global_offset = 0;
    disterr->mpierrcode = MPI_Exscan(
        &size, &global_offset, 1, MPI_INT64_T, MPI_SUM, comm);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;

    uint32_t * extra_keys = malloc(size * sizeof(uint32_t));
    err = !extra_keys ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;

    int64_t * bucketptrs = malloc(comm_size * 257 * sizeof(int64_t));
    err = !bucketptrs ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(extra_keys);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    int64_t * bucketptr = &bucketptrs[rank*257];

    int64_t * extra_sorting_permutation = NULL;
    if (sorting_permutation) {
        extra_sorting_permutation = malloc(size * sizeof(int64_t));
        err = !extra_sorting_permutation ? MTX_ERR_ERRNO : MTX_SUCCESS;
        if (mtxdisterror_allreduce(disterr, err)) {
            free(bucketptrs);
            free(extra_keys);
            return MTX_ERR_MPI_COLLECTIVE;
        }
    }

    int * sendrecvbufs = malloc(comm_size * 5 * sizeof(int));
    err = !sendrecvbufs ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        if (extra_sorting_permutation)
            free(extra_sorting_permutation);
        free(bucketptrs);
        free(extra_keys);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    int * sendcounts = &sendrecvbufs[0*comm_size];
    int * senddisps  = &sendrecvbufs[1*comm_size];
    int * recvcounts = &sendrecvbufs[2*comm_size];
    int * recvdisps  = &sendrecvbufs[3*comm_size];
    int * sizes      = &sendrecvbufs[4*comm_size];

    /* Perform one round of sorting for each digit in a key */
    for (int k = 0; k < 4; k++) {
#ifdef DEBUG_DISTRADIXSORT
        for (int p = 0; p < comm_size; p++) {
            if (rank == p) {
                fprintf(stderr, "before round %d, process %d, keys=[", k, rank);
                for (int64_t i = 0; i < size; i++)
                    fprintf(stderr, " %lld (0x%02x)", keys[i], (keys[i] >> (8*k)) & 0xff);
                fprintf(stderr, "]\n");
            }
            MPI_Barrier(comm);
        }
#endif

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
                extra_sorting_permutation[destidx] = global_offset+i;
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

        /* Adjust the offsets to each bucket. */
        for (int j = 256; j > 0; j--)
            bucketptr[j] = bucketptr[j-1];
        bucketptr[0] = 0;

#ifdef DEBUG_DISTRADIXSORT
        for (int p = 0; p < comm_size; p++) {
            if (rank == p) {
                fprintf(stderr, "middle of round %d, process %d, extra_keys=[", k, rank);
                for (int64_t i = 0; i < size; i++)
                    fprintf(stderr, " %lld (0x%02x)", extra_keys[i], (extra_keys[i] >> (8*k)) & 0xff);
                fprintf(stderr, "], extra_sorting_permutation=[");
                for (int64_t i = 0; i < size; i++)
                    fprintf(stderr, " %2lld", extra_sorting_permutation[i]);
                fprintf(stderr, "]\n");
            }
            MPI_Barrier(comm);
        }
#endif

#ifdef DEBUG_DISTRADIXSORT
        for (int p = 0; p < comm_size; p++) {
            if (rank == p) {
                fprintf(stderr, "round %d, process %d, bucketptr=[", k, p);
                for (int j = 0; j <= 256; j++)
                    fprintf(stderr, " %lld", bucketptr[j]);
                fprintf(stderr, "]\n");
            }
            MPI_Barrier(comm);
        }
#endif

        /* 5. Gather all the bucket pointers onto every process. */
        disterr->mpierrcode = MPI_Allgather(
            MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, bucketptrs, 257, MPI_INT64_T, comm);
        err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
        if (mtxdisterror_allreduce(disterr, err)) {
            free(sendrecvbufs);
            if (sorting_permutation)
                free(extra_sorting_permutation);
            free(bucketptrs);
            free(extra_keys);
            return MTX_ERR_MPI_COLLECTIVE;
        }

        /* Calculate the number of keys held by each process. */
        for (int p = 0; p < comm_size; p++) {
            sendcounts[p] = 0;
            recvcounts[p] = 0;
            sizes[p] = bucketptrs[(p+1)*257-1] - bucketptrs[p*257];
        }

        /* Calculate the number of keys in each bucket. */
        for (int p = 0; p < comm_size; p++) {
            for (int j = 255; j >= 0; j--)
                bucketptrs[p*257+j+1] -= bucketptrs[p*257+j];
        }

        /* Find the keys to send and receive for each process. */
        err = MTX_SUCCESS;
        int q = 0;
        for (int j = 0; j <= 256; j++) {
            for (int p = 0; p < comm_size; p++) {
                if (q >= comm_size) {
                    err = MTX_ERR_INDEX_OUT_OF_BOUNDS;
                    break;
                }

                int count =
                    sizes[q] < bucketptrs[p*257+j]
                    ? sizes[q] : bucketptrs[p*257+j];
                sizes[q] -= count;
                bucketptrs[p*257+j] -= count;

                if (rank == p)
                    sendcounts[q] += count;
                if (rank == q)
                    recvcounts[p] += count;

#ifdef DEBUG_DISTRADIXSORT
                if (count > 0 && rank == 0) {
                    fprintf(stderr, "round %d, process %d sends %d key(s) "
                            "from bucket %d to process %d "
                            "(process %d has %d more keys in this bucket, "
                            "process %d has room for %d more keys)\n",
                            k, p, count, j, q, p, bucketptrs[p*257+j],
                            q, sizes[q]);
                }
#endif

                if (err || (sizes[q] == 0 && q == comm_size-1))
                    break;
                else if (sizes[q] == 0) {
                    q++;
                    p--;
                    continue;
                }
            }
            if (sizes[q] == 0 && q == comm_size-1)
                break;
        }
        if (mtxdisterror_allreduce(disterr, err)) {
            free(sendrecvbufs);
            if (sorting_permutation)
                free(extra_sorting_permutation);
            free(bucketptrs);
            free(extra_keys);
            return MTX_ERR_MPI_COLLECTIVE;
        }

        senddisps[0] = 0;
        recvdisps[0] = 0;
        for (int p = 1; p < comm_size; p++) {
            senddisps[p] = senddisps[p-1] + sendcounts[p-1];
            recvdisps[p] = recvdisps[p-1] + recvcounts[p-1];
        }

#ifdef DEBUG_DISTRADIXSORT
        for (int q = 0; q < comm_size; q++) {
            if (rank == q) {
                fprintf(stderr, "round %d, process %d sends ", k, q);
                for (int p = 0; p < comm_size; p++) {
                    fprintf(stderr, "[");
                    for (int j = senddisps[p]; j < senddisps[p+1]; j++)
                        fprintf(stderr, " %d", extra_keys[j]);
                    fprintf(stderr, "] to process %d, ", p);
                }
                fprintf(stderr, "\n");
            }
            MPI_Barrier(comm);
        }
#endif

        /* 6. Redistribute keys among processes. */
        disterr->mpierrcode = MPI_Alltoallv(
            extra_keys, sendcounts, senddisps, MPI_INT32_T,
            keys, recvcounts, recvdisps, MPI_INT32_T, comm);
        err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
        if (mtxdisterror_allreduce(disterr, err)) {
            free(sendrecvbufs);
            if (sorting_permutation)
                free(extra_sorting_permutation);
            free(bucketptrs);
            free(extra_keys);
            return MTX_ERR_MPI_COLLECTIVE;
        }

        /* Also, redistribute the sorting permutation. */
        if (sorting_permutation) {
            disterr->mpierrcode = MPI_Alltoallv(
                extra_sorting_permutation, sendcounts, senddisps, MPI_INT64_T,
                sorting_permutation, recvcounts, recvdisps, MPI_INT64_T, comm);
            err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
            if (mtxdisterror_allreduce(disterr, err)) {
                free(sendrecvbufs);
                if (sorting_permutation)
                    free(extra_sorting_permutation);
                free(bucketptrs);
                free(extra_keys);
                return MTX_ERR_MPI_COLLECTIVE;
            }
        }

#ifdef DEBUG_DISTRADIXSORT
        for (int p = 0; p < comm_size; p++) {
            if (rank == p) {
                fprintf(stderr, "after redistribution in round %d, process %d, keys=[", k, rank);
                for (int64_t i = 0; i < size; i++)
                    fprintf(stderr, " %lld (0x%02x)", keys[i], (keys[i] >> (8*k)) & 0xff);
                fprintf(stderr, "], sorting_permutation=[");
                for (int64_t i = 0; i < size; i++)
                    fprintf(stderr, " %2lld", sorting_permutation[i]);
                fprintf(stderr, "]\n");
            }
            MPI_Barrier(comm);
        }
#endif

        /* 1. Count the number of keys in each bucket. */
        for (int j = 0; j <= 256; j++)
            bucketptr[j] = 0;
        for (int64_t i = 0; i < size; i++)
            bucketptr[((keys[i] >> (8*k)) & 0xff)+1]++;

        /* 2. Compute offset to first key in each bucket. */
        for (int j = 0; j < 256; j++)
            bucketptr[j+1] += bucketptr[j];

        /* 3. Sort the keys into their respective buckets. */
        if (sorting_permutation) {
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
        if (sorting_permutation) {
            for (int64_t j = 0; j < size; j++) {
                keys[j] = extra_keys[j];
                sorting_permutation[j] = extra_sorting_permutation[j];
            }
        } else {
            for (int64_t j = 0; j < size; j++)
                keys[j] = extra_keys[j];
        }

#ifdef DEBUG_DISTRADIXSORT
        for (int p = 0; p < comm_size; p++) {
            if (rank == p) {
                fprintf(stderr, "after round %d, process %d, keys=[", k, rank);
                for (int64_t i = 0; i < size; i++)
                    fprintf(stderr, " %lld (0x%02x)", keys[i], (keys[i] >> (8*k)) & 0xff);
                fprintf(stderr, "], sorting_permutation=[");
                for (int64_t i = 0; i < size; i++)
                    fprintf(stderr, " %2lld", sorting_permutation[i]);
                fprintf(stderr, "]\n");
            }
            MPI_Barrier(comm);
        }
#endif
    }

    free(sendrecvbufs);
    free(bucketptrs);
    free(extra_keys);

    /* Invert the sorting permutation */
    if (sorting_permutation) {
        int64_t * global_offsets = malloc((comm_size+1) * sizeof(int64_t));
        err = !global_offsets ? MTX_ERR_ERRNO : MTX_SUCCESS;
        if (mtxdisterror_allreduce(disterr, err)) {
            free(extra_sorting_permutation);
            return MTX_ERR_MPI_COLLECTIVE;
        }

        global_offsets[rank] = global_offset;
        disterr->mpierrcode = MPI_Allgather(
            MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, global_offsets, 1, MPI_INT64_T, comm);
        err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
        if (mtxdisterror_allreduce(disterr, err)) {
            free(global_offsets);
            free(extra_sorting_permutation);
            return MTX_ERR_MPI_COLLECTIVE;
        }
        global_offsets[comm_size] = total_size;

        MPI_Win window;
        disterr->mpierrcode = MPI_Win_create(
            sorting_permutation, size * sizeof(int64_t),
            sizeof(int64_t), MPI_INFO_NULL, comm, &window);
        err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
        if (mtxdisterror_allreduce(disterr, err)) {
            free(global_offsets);
            free(extra_sorting_permutation);
            return MTX_ERR_MPI_COLLECTIVE;
        }
        MPI_Win_fence(0, window);

#ifdef DEBUG_DISTRADIXSORT
        for (int p = 0; p < comm_size; p++) {
            if (rank == p) {
                fprintf(stderr, "process %d, global_offsets=[", rank);
                for (int q = 0; q < comm_size; q++)
                    fprintf(stderr, " %lld", global_offsets[q]);
                fprintf(stderr, "]\n");
            }
            MPI_Barrier(comm);
        }
        for (int p = 0; p < comm_size; p++) {
            if (rank == p) {
                fprintf(stderr, "process %d, sorting_permutation=[", rank);
                for (int64_t i = 0; i < size; i++)
                    fprintf(stderr, " %lld", sorting_permutation[i]);
                fprintf(stderr, "]\n");
            }
            MPI_Barrier(comm);
        }
#endif

#ifdef DEBUG_DISTRADIXSORT
        for (int p = 0; p < comm_size; p++) {
            if (rank == p) {
                fprintf(stderr, "process %d, extra_sorting_permutation=[", rank);
                for (int64_t i = 0; i < size; i++)
                    fprintf(stderr, " %lld", extra_sorting_permutation[i]);
                fprintf(stderr, "]\n");
            }
            MPI_Barrier(comm);
        }
#endif

        err = MTX_SUCCESS;
        for (int64_t j = 0; j < size; j++) {
            int64_t destidx = global_offset + j;
            int64_t srcidx = extra_sorting_permutation[j];

            int p = 0;
            while (p < comm_size && global_offsets[p+1] <= srcidx)
                p++;

#ifdef DEBUG_DISTRADIXSORT
            fprintf(stderr, "process %d put the value %lld at process %d location %lld (srcidx=%lld, global_offsets[%d]=%lld).\n",
                    rank, destidx, p, srcidx-global_offsets[p], srcidx, p, global_offsets[p]);
#endif
            if (p != rank) {
                disterr->mpierrcode = MPI_Put(
                    &destidx, 1, MPI_INT64_T, p,
                    srcidx-global_offsets[p], 1, MPI_INT64_T, window);
                err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
                if (err)
                    break;
            } else {
                sorting_permutation[srcidx-global_offset] = destidx;
            }
        }
        MPI_Win_fence(0, window);
        MPI_Win_free(&window);
        free(global_offsets);
        free(extra_sorting_permutation);
        if (mtxdisterror_allreduce(disterr, err))
            return MTX_ERR_MPI_COLLECTIVE;

#ifdef DEBUG_DISTRADIXSORT
        for (int p = 0; p < comm_size; p++) {
            if (rank == p) {
                fprintf(stderr, "after inverting sorting permutation, sorting_permutation=[");
                for (int64_t i = 0; i < size; i++)
                    fprintf(stderr, " %2lld", sorting_permutation[i]);
                fprintf(stderr, "]\n");
            }
            MPI_Barrier(comm);
        }
#endif
    }

    return MTX_SUCCESS;
}

/**
 * ‘distradix_sort_uint64()’ sorts a distributed array of 64-bit
 * unsigned integers in ascending order using a distributed radix sort
 * algorithm.
 *
 * The number of keys on the current process that need to be sorted is
 * given by ‘size’, and the unsorted, integer keys on the current
 * process are given in the array ‘keys’. On success, the same array
 * will contain ‘size’ keys in a globally sorted order.
 *
 * If ‘sorting_permutation’ is ‘NULL’, then this argument is ignored
 * and a sorting permutation is not computed. Otherwise, it must point
 * to an array that holds enough storage for ‘size’ values of type
 * ‘int64_t’ on each MPI process. On success, this array will contain
 * the sorting permutation, mapping the locations of the original,
 * unsorted keys to their new locations in the sorted array.
 */
int distradix_sort_uint64(
    int64_t size,
    uint64_t * keys,
    int64_t * sorting_permutation,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err;

    /* The current implementation can only sort at most ‘INT_MAX’ keys
     * on each process. */
    if (size > INT_MAX) errno = ERANGE;
    err = size > INT_MAX ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;

    int comm_size;
    disterr->mpierrcode = MPI_Comm_size(comm, &comm_size);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;
    int rank;
    disterr->mpierrcode = MPI_Comm_rank(comm, &rank);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;

    int64_t total_size;
    disterr->mpierrcode = MPI_Allreduce(
        &size, &total_size, 1, MPI_INT64_T, MPI_SUM, comm);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;

    int64_t global_offset = 0;
    disterr->mpierrcode = MPI_Exscan(
        &size, &global_offset, 1, MPI_INT64_T, MPI_SUM, comm);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;

    uint64_t * extra_keys = malloc(size * sizeof(uint64_t));
    err = !extra_keys ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;

    int64_t * bucketptrs = malloc(comm_size * 257 * sizeof(int64_t));
    err = !bucketptrs ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(extra_keys);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    int64_t * bucketptr = &bucketptrs[rank*257];

    int64_t * extra_sorting_permutation = NULL;
    if (sorting_permutation) {
        extra_sorting_permutation = malloc(size * sizeof(int64_t));
        err = !extra_sorting_permutation ? MTX_ERR_ERRNO : MTX_SUCCESS;
        if (mtxdisterror_allreduce(disterr, err)) {
            free(bucketptrs);
            free(extra_keys);
            return MTX_ERR_MPI_COLLECTIVE;
        }
    }

    int * sendrecvbufs = malloc(comm_size * 5 * sizeof(int));
    err = !sendrecvbufs ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        if (extra_sorting_permutation)
            free(extra_sorting_permutation);
        free(bucketptrs);
        free(extra_keys);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    int * sendcounts = &sendrecvbufs[0*comm_size];
    int * senddisps  = &sendrecvbufs[1*comm_size];
    int * recvcounts = &sendrecvbufs[2*comm_size];
    int * recvdisps  = &sendrecvbufs[3*comm_size];
    int * sizes      = &sendrecvbufs[4*comm_size];

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
                extra_sorting_permutation[destidx] = global_offset+i;
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

        /* Adjust the offsets to each bucket. */
        for (int j = 256; j > 0; j--)
            bucketptr[j] = bucketptr[j-1];
        bucketptr[0] = 0;

        /* 5. Gather all the bucket pointers onto every process. */
        disterr->mpierrcode = MPI_Allgather(
            MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, bucketptrs, 257, MPI_INT64_T, comm);
        err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
        if (mtxdisterror_allreduce(disterr, err)) {
            free(sendrecvbufs);
            if (sorting_permutation)
                free(extra_sorting_permutation);
            free(bucketptrs);
            free(extra_keys);
            return MTX_ERR_MPI_COLLECTIVE;
        }

        /* Calculate the number of keys held by each process. */
        for (int p = 0; p < comm_size; p++) {
            sendcounts[p] = 0;
            recvcounts[p] = 0;
            sizes[p] = bucketptrs[(p+1)*257-1] - bucketptrs[p*257];
        }

        /* Calculate the number of keys in each bucket. */
        for (int p = 0; p < comm_size; p++) {
            for (int j = 255; j >= 0; j--)
                bucketptrs[p*257+j+1] -= bucketptrs[p*257+j];
        }

        /* Find the keys to send and receive for each process. */
        err = MTX_SUCCESS;
        int q = 0;
        for (int j = 0; j <= 256; j++) {
            for (int p = 0; p < comm_size; p++) {
                if (q >= comm_size) {
                    err = MTX_ERR_INDEX_OUT_OF_BOUNDS;
                    break;
                }

                int count =
                    sizes[q] < bucketptrs[p*257+j]
                    ? sizes[q] : bucketptrs[p*257+j];
                sizes[q] -= count;
                bucketptrs[p*257+j] -= count;

                if (rank == p)
                    sendcounts[q] += count;
                if (rank == q)
                    recvcounts[p] += count;

                if (err || (sizes[q] == 0 && q == comm_size-1))
                    break;
                else if (sizes[q] == 0) {
                    q++;
                    p--;
                    continue;
                }
            }
            if (sizes[q] == 0 && q == comm_size-1)
                break;
        }
        if (mtxdisterror_allreduce(disterr, err)) {
            free(sendrecvbufs);
            if (sorting_permutation)
                free(extra_sorting_permutation);
            free(bucketptrs);
            free(extra_keys);
            return MTX_ERR_MPI_COLLECTIVE;
        }

        senddisps[0] = 0;
        recvdisps[0] = 0;
        for (int p = 1; p < comm_size; p++) {
            senddisps[p] = senddisps[p-1] + sendcounts[p-1];
            recvdisps[p] = recvdisps[p-1] + recvcounts[p-1];
        }

        /* 6. Redistribute keys among processes. */
        disterr->mpierrcode = MPI_Alltoallv(
            extra_keys, sendcounts, senddisps, MPI_INT64_T,
            keys, recvcounts, recvdisps, MPI_INT64_T, comm);
        err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
        if (mtxdisterror_allreduce(disterr, err)) {
            free(sendrecvbufs);
            if (sorting_permutation)
                free(extra_sorting_permutation);
            free(bucketptrs);
            free(extra_keys);
            return MTX_ERR_MPI_COLLECTIVE;
        }

        /* Also, redistribute the sorting permutation. */
        if (sorting_permutation) {
            disterr->mpierrcode = MPI_Alltoallv(
                extra_sorting_permutation, sendcounts, senddisps, MPI_INT64_T,
                sorting_permutation, recvcounts, recvdisps, MPI_INT64_T, comm);
            err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
            if (mtxdisterror_allreduce(disterr, err)) {
                free(sendrecvbufs);
                if (sorting_permutation)
                    free(extra_sorting_permutation);
                free(bucketptrs);
                free(extra_keys);
                return MTX_ERR_MPI_COLLECTIVE;
            }
        }

        /* 1. Count the number of keys in each bucket. */
        for (int j = 0; j <= 256; j++)
            bucketptr[j] = 0;
        for (int64_t i = 0; i < size; i++)
            bucketptr[((keys[i] >> (8*k)) & 0xff)+1]++;

        /* 2. Compute offset to first key in each bucket. */
        for (int j = 0; j < 256; j++)
            bucketptr[j+1] += bucketptr[j];

        /* 3. Sort the keys into their respective buckets. */
        if (sorting_permutation) {
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
        if (sorting_permutation) {
            for (int64_t j = 0; j < size; j++) {
                keys[j] = extra_keys[j];
                sorting_permutation[j] = extra_sorting_permutation[j];
            }
        } else {
            for (int64_t j = 0; j < size; j++)
                keys[j] = extra_keys[j];
        }
    }

    free(sendrecvbufs);
    free(bucketptrs);
    free(extra_keys);

    /* Invert the sorting permutation */
    if (sorting_permutation) {
        int64_t * global_offsets = malloc((comm_size+1) * sizeof(int64_t));
        err = !global_offsets ? MTX_ERR_ERRNO : MTX_SUCCESS;
        if (mtxdisterror_allreduce(disterr, err)) {
            free(extra_sorting_permutation);
            return MTX_ERR_MPI_COLLECTIVE;
        }

        global_offsets[rank] = global_offset;
        disterr->mpierrcode = MPI_Allgather(
            MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, global_offsets, 1, MPI_INT64_T, comm);
        err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
        if (mtxdisterror_allreduce(disterr, err)) {
            free(global_offsets);
            free(extra_sorting_permutation);
            return MTX_ERR_MPI_COLLECTIVE;
        }
        global_offsets[comm_size] = total_size;

        MPI_Win window;
        disterr->mpierrcode = MPI_Win_create(
            sorting_permutation, size * sizeof(int64_t),
            sizeof(int64_t), MPI_INFO_NULL, comm, &window);
        err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
        if (mtxdisterror_allreduce(disterr, err)) {
            free(global_offsets);
            free(extra_sorting_permutation);
            return MTX_ERR_MPI_COLLECTIVE;
        }
        MPI_Win_fence(0, window);

        /*
         * Use one-sided MPI communication to write the remote values
         * for inverting the sorting permutation.
         */
        err = MTX_SUCCESS;
        for (int64_t j = 0; j < size; j++) {
            int64_t destidx = global_offset + j;
            int64_t srcidx = extra_sorting_permutation[j];

            int p = 0;
            while (p < comm_size && global_offsets[p+1] <= srcidx)
                p++;

            if (p != rank) {
                disterr->mpierrcode = MPI_Put(
                    &destidx, 1, MPI_INT64_T, p,
                    srcidx-global_offsets[p], 1, MPI_INT64_T, window);
                err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
                if (err)
                    break;
            } else {
                sorting_permutation[srcidx-global_offset] = destidx;
            }
        }
        MPI_Win_fence(0, window);
        MPI_Win_free(&window);
        free(global_offsets);
        free(extra_sorting_permutation);
        if (mtxdisterror_allreduce(disterr, err))
            return MTX_ERR_MPI_COLLECTIVE;
    }

    return MTX_SUCCESS;
}
#endif
