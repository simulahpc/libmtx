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

#include "sort.h"

#include <libmtx/libmtx-config.h>

#include <errno.h>

#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
 * If ‘perm’ is ‘NULL’, then this argument is ignored
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
    int64_t * perm,
    int64_t bucketptr[257])
{
    bool alloc_bucketptr = !bucketptr;
    if (alloc_bucketptr) {
        bucketptr = malloc(257 * sizeof(int64_t));
        if (!bucketptr) return errno;
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
    if (sorted_keys && perm) {
        for (int64_t i = 0; i < size; i++) {
            int64_t destidx = bucketptr[keys[srcstride*i]];
            sorted_keys[dststride*destidx] = keys[srcstride*i];
            perm[i] = destidx;
            bucketptr[keys[srcstride*i]]++;
        }
    } else if (sorted_keys) {
        for (int64_t i = 0; i < size; i++) {
            int64_t destidx = bucketptr[keys[srcstride*i]];
            sorted_keys[dststride*destidx] = keys[srcstride*i];
            bucketptr[keys[srcstride*i]]++;
        }
    } else if (perm) {
        for (int64_t i = 0; i < size; i++) {
            int64_t destidx = bucketptr[keys[srcstride*i]];
            perm[i] = destidx;
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
    return 0;
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
 * If ‘perm’ is ‘NULL’, then this argument is ignored
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
    int64_t * perm,
    int64_t bucketptr[65537])
{
    bool alloc_bucketptr = !bucketptr;
    if (alloc_bucketptr) {
        bucketptr = malloc(65537 * sizeof(int64_t));
        if (!bucketptr) return errno;
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
    if (sorted_keys && perm) {
        for (int64_t i = 0; i < size; i++) {
            const uint16_t * src = (const uint16_t *) ((uint8_t *) keys + srcstride*i);
            int64_t destidx = bucketptr[*src]++;
            uint16_t * dest = (uint16_t *)((uint8_t *) sorted_keys+dststride*destidx);
            *dest = *src;
            perm[i] = destidx;
        }
    } else if (sorted_keys) {
        for (int64_t i = 0; i < size; i++) {
            const uint16_t * src = (const uint16_t *) ((uint8_t *) keys + srcstride*i);
            int64_t destidx = bucketptr[*src]++;
            uint16_t * dest = (uint16_t *)((uint8_t *) sorted_keys+dststride*destidx);
            *dest = *src;
        }
    } else if (perm) {
        for (int64_t i = 0; i < size; i++) {
            const uint16_t * src = (const uint16_t *) ((uint8_t *) keys + srcstride*i);
            int64_t destidx = bucketptr[*src]++;
            perm[i] = destidx;
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
    return 0;
}

int counting_sort_uint8_uint32(
    int64_t size,
    int stride,
    const uint8_t * keys,
    const uint32_t * values,
    uint32_t * sorted_values,
    int64_t * perm,
    int64_t bucketptr[257])
{
    bool alloc_bucketptr = !bucketptr;
    if (alloc_bucketptr) {
        bucketptr = malloc(257 * sizeof(int64_t));
        if (!bucketptr) return errno;
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
    if (values && sorted_values && perm) {
        for (int64_t i = 0; i < size; i++) {
            int64_t destidx = bucketptr[keys[stride*i]]++;
            sorted_values[destidx] = values[i];
            perm[i] = destidx;
        }
    } else if (values && sorted_values) {
        for (int64_t i = 0; i < size; i++) {
            int64_t destidx = bucketptr[keys[stride*i]]++;
            sorted_values[destidx] = values[i];
        }
    } else if (perm) {
        for (int64_t i = 0; i < size; i++) {
            int64_t destidx = bucketptr[keys[stride*i]]++;
            perm[i] = destidx;
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
    return 0;
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
        if (!bucketptr) return errno;
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
    return 0;
}

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
 * If ‘perm’ is ‘NULL’, then this argument is ignored
 * and a sorting permutation is not computed. Otherwise, it must point
 * to an array that holds enough storage for ‘size’ values of type
 * ‘int64_t’. On success, this array will contain the sorting
 * permutation, mapping the locations of the original, unsorted keys
 * to their new locations in the sorted array.
 */
int radix_sort_uint32(
    int64_t size,
    uint32_t * keys,
    int64_t * perm)
{
    uint32_t * extra_keys = malloc(size * sizeof(uint32_t));
    if (!extra_keys) return errno;

    int64_t * bucketptr = malloc(257 * sizeof(int64_t));
    if (!bucketptr) { free(extra_keys); return errno; }

    int64_t * extra_perm = NULL;
    if (perm) {
        extra_perm = malloc(size * sizeof(int64_t));
        if (!extra_perm) {
            free(bucketptr);
            free(extra_keys);
            return errno;
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
        if (perm && k == 0) {
            for (int64_t i = 0; i < size; i++) {
                int64_t destidx = bucketptr[((keys[i] >> (8*k)) & 0xff)]++;
                extra_keys[destidx] = keys[i];
                perm[destidx] = i;
            }
        } else if (perm) {
            for (int64_t i = 0; i < size; i++) {
                int64_t destidx = bucketptr[((keys[i] >> (8*k)) & 0xff)]++;
                extra_keys[destidx] = keys[i];
                extra_perm[destidx] = perm[i];
            }
        } else {
            for (int64_t i = 0; i < size; i++) {
                int64_t destidx = bucketptr[((keys[i] >> (8*k)) & 0xff)]++;
                extra_keys[destidx] = keys[i];
            }
        }

        /* 4. Copy data needed for the next round. */
        if (perm && k == 3) {
            for (int64_t j = 0; j < size; j++) {
                keys[j] = extra_keys[j];
                perm[extra_perm[j]] = j;
            }
        } else if (perm && k > 0) {
            for (int64_t j = 0; j < size; j++) {
                keys[j] = extra_keys[j];
                perm[j] = extra_perm[j];
            }
        } else {
            for (int64_t j = 0; j < size; j++)
                keys[j] = extra_keys[j];
        }
    }

    if (perm)
        free(extra_perm);
    free(bucketptr);
    free(extra_keys);
    return 0;
}

static void swap_int(int * a, int * b) { int tmp = *a; *a = *b; *b = tmp; }
static void swap_int64ptr(int64_t ** a, int64_t ** b) { int64_t * tmp = *a; *a = *b; *b = tmp; }
static void swap_uint32ptr(uint32_t ** a, uint32_t ** b) { uint32_t * tmp = *a; *a = *b; *b = tmp; }
static void swap_uint64ptr(uint64_t ** a, uint64_t ** b) { uint64_t * tmp = *a; *a = *b; *b = tmp; }

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
    int64_t * perm)
{
    uint64_t * tmpkeys = malloc(size * sizeof(uint64_t));
    if (!tmpkeys) return errno;
    int tmpkeysstride = sizeof(*tmpkeys);

    /* allocate six buckets to count occurrences and offsets for
     * 11-digit binary numbers. */
    int64_t * bucketptr = malloc(6*2049 * sizeof(int64_t));
    if (!bucketptr) { free(tmpkeys); return errno; }

    int64_t * tmpperm = NULL;
    if (perm) {
        tmpperm = malloc(size * sizeof(int64_t));
        if (!tmpperm) {
            free(bucketptr);
            free(tmpkeys);
            return errno;
        }
    }

    /* 1. Count the number of keys in each bucket. */
    for (int k = 0; k < 6; k++) {
        for (int j = 0; j <= 2048; j++) {
            bucketptr[k*2049+j] = 0;
        }
    }
    for (int64_t i = 0; i < size; i++) {
        uint64_t x = *(const uint64_t *) ((const char *) keys + i*stride);
        bucketptr[0*2049+((x >> (11*0)) & 0x7ff)+1]++;
        bucketptr[1*2049+((x >> (11*1)) & 0x7ff)+1]++;
        bucketptr[2*2049+((x >> (11*2)) & 0x7ff)+1]++;
        bucketptr[3*2049+((x >> (11*3)) & 0x7ff)+1]++;
        bucketptr[4*2049+((x >> (11*4)) & 0x7ff)+1]++;
        bucketptr[5*2049+((x >> (11*5)) & 0x7ff)+1]++;
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
    if (perm) {
        for (int64_t i = 0; i < size; i++) {
            uint64_t x = *(const uint64_t *) ((const char *) keys + i*stride);
            int64_t dst = bucketptr[0*2049+((x >> (11*0)) & 0x7ff)]++;
            *(uint64_t *) ((char *) tmpkeys + dst*tmpkeysstride) = x;
            perm[dst] = i;
        }
        swap_uint64ptr(&keys, &tmpkeys); swap_int(&stride, &tmpkeysstride);
        for (int64_t i = 0; i < size; i++) {
            uint64_t x = *(const uint64_t *) ((const char *) keys + i*stride);
            int64_t dst = bucketptr[1*2049+((x >> (11*1)) & 0x7ff)]++;
            *(uint64_t *) ((char *) tmpkeys + dst*tmpkeysstride) = x;
            tmpperm[dst] = perm[i];
        }
        swap_uint64ptr(&keys, &tmpkeys); swap_int(&stride, &tmpkeysstride);
        swap_int64ptr(&perm, &tmpperm);
        for (int64_t i = 0; i < size; i++) {
            uint64_t x = *(const uint64_t *) ((const char *) keys + i*stride);
            int64_t dst = bucketptr[2*2049+((x >> (11*2)) & 0x7ff)]++;
            *(uint64_t *) ((char *) tmpkeys + dst*tmpkeysstride) = x;
            tmpperm[dst] = perm[i];
        }
        swap_uint64ptr(&keys, &tmpkeys); swap_int(&stride, &tmpkeysstride);
        swap_int64ptr(&perm, &tmpperm);
        for (int64_t i = 0; i < size; i++) {
            uint64_t x = *(const uint64_t *) ((const char *) keys + i*stride);
            int64_t dst = bucketptr[3*2049+((x >> (11*3)) & 0x7ff)]++;
            *(uint64_t *) ((char *) tmpkeys + dst*tmpkeysstride) = x;
            tmpperm[dst] = perm[i];
        }
        swap_uint64ptr(&keys, &tmpkeys); swap_int(&stride, &tmpkeysstride);
        swap_int64ptr(&perm, &tmpperm);
        for (int64_t i = 0; i < size; i++) {
            uint64_t x = *(const uint64_t *) ((const char *) keys + i*stride);
            int64_t dst = bucketptr[4*2049+((x >> (11*4)) & 0x7ff)]++;
            *(uint64_t *) ((char *) tmpkeys + dst*tmpkeysstride) = x;
            tmpperm[dst] = perm[i];
        }
        swap_uint64ptr(&keys, &tmpkeys); swap_int(&stride, &tmpkeysstride);
        swap_int64ptr(&perm, &tmpperm);
        for (int64_t i = 0; i < size; i++) {
            uint64_t x = *(const uint64_t *) ((const char *) keys + i*stride);
            int64_t dst = bucketptr[5*2049+((x >> (11*5)) & 0x7ff)]++;
            *(uint64_t *) ((char *) tmpkeys + dst*tmpkeysstride) = x;
            tmpperm[dst] = perm[i];
        }
        swap_uint64ptr(&keys, &tmpkeys); swap_int(&stride, &tmpkeysstride);
        for (int64_t j = 0; j < size; j++)
            perm[tmpperm[j]] = j;
    } else {
        for (int64_t i = 0; i < size; i++) {
            uint64_t x = *(const uint64_t *) ((const char *) keys + i*stride);
            int64_t dst = bucketptr[0*2049+((x >> (11*0)) & 0x7ff)]++;
            *(uint64_t *) ((char *) tmpkeys + dst*tmpkeysstride) = x;
        }
        swap_uint64ptr(&keys, &tmpkeys); swap_int(&stride, &tmpkeysstride);
        for (int64_t i = 0; i < size; i++) {
            uint64_t x = *(const uint64_t *) ((const char *) keys + i*stride);
            int64_t dst = bucketptr[1*2049+((x >> (11*1)) & 0x7ff)]++;
            *(uint64_t *) ((char *) tmpkeys + dst*tmpkeysstride) = x;
        }
        swap_uint64ptr(&keys, &tmpkeys); swap_int(&stride, &tmpkeysstride);
        for (int64_t i = 0; i < size; i++) {
            uint64_t x = *(const uint64_t *) ((const char *) keys + i*stride);
            int64_t dst = bucketptr[2*2049+((x >> (11*2)) & 0x7ff)]++;
            *(uint64_t *) ((char *) tmpkeys + dst*tmpkeysstride) = x;
        }
        swap_uint64ptr(&keys, &tmpkeys); swap_int(&stride, &tmpkeysstride);
        for (int64_t i = 0; i < size; i++) {
            uint64_t x = *(const uint64_t *) ((const char *) keys + i*stride);
            int64_t dst = bucketptr[3*2049+((x >> (11*3)) & 0x7ff)]++;
            *(uint64_t *) ((char *) tmpkeys + dst*tmpkeysstride) = x;
        }
        swap_uint64ptr(&keys, &tmpkeys); swap_int(&stride, &tmpkeysstride);
        for (int64_t i = 0; i < size; i++) {
            uint64_t x = *(const uint64_t *) ((const char *) keys + i*stride);
            int64_t dst = bucketptr[4*2049+((x >> (11*4)) & 0x7ff)]++;
            *(uint64_t *) ((char *) tmpkeys + dst*tmpkeysstride) = x;
        }
        swap_uint64ptr(&keys, &tmpkeys); swap_int(&stride, &tmpkeysstride);
        for (int64_t i = 0; i < size; i++) {
            uint64_t x = *(const uint64_t *) ((const char *) keys + i*stride);
            int64_t dst = bucketptr[5*2049+((x >> (11*5)) & 0x7ff)]++;
            *(uint64_t *) ((char *) tmpkeys + dst*tmpkeysstride) = x;
        }
        swap_uint64ptr(&keys, &tmpkeys); swap_int(&stride, &tmpkeysstride);
    }

    if (perm) free(tmpperm);
    free(bucketptr); free(tmpkeys);
    return 0;
}

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
 * If ‘perm’ is ‘NULL’, then this argument is ignored
 * and a sorting permutation is not computed. Otherwise, it must point
 * to an array that holds enough storage for ‘size’ values of type
 * ‘int64_t’. On success, this array will contain the sorting
 * permutation, mapping the locations of the original, unsorted keys
 * to their new locations in the sorted array.
 */
int radix_sort_int32(
    int64_t size,
    int32_t * keys,
    int64_t * perm)
{
    for (int64_t i = 0; i < size; i++)
        keys[i] ^= INT32_MIN;
    int err = radix_sort_uint32(size, (uint32_t *) keys, perm);
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
    int64_t * perm)
{
    for (int64_t i = 0; i < size; i++)
        *(int64_t *)((char *) keys+i*stride) ^= INT64_MIN;
    int err = radix_sort_uint64(size, stride, (uint64_t *) keys, perm);
    for (int64_t i = 0; i < size; i++)
        *(int64_t *)((char *) keys+i*stride) ^= INT64_MIN;
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
 * If ‘perm’ is ‘NULL’, then this argument is ignored
 * and a sorting permutation is not computed. Otherwise, it must point
 * to an array that holds enough storage for ‘size’ values of type
 * ‘int64_t’. On success, this array will contain the sorting
 * permutation, mapping the locations of the original, unsorted keys
 * to their new locations in the sorted array.
 */
int radix_sort_int(
    int64_t size,
    int * keys,
    int64_t * perm)
{
    if (sizeof(int) == sizeof(int32_t)) {
        return radix_sort_int32(size, (int32_t *) keys, perm);
    } else if (sizeof(int) == sizeof(int64_t)) {
        return radix_sort_int64(size, sizeof(*keys), (int64_t *) keys, perm);
    } else { return ENOTSUP; }
}

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
    int64_t * perm)
{
    uint32_t * tmpa = malloc(size * sizeof(uint32_t));
    int tmpastride = sizeof(*tmpa);
    if (!tmpa) return errno;
    uint32_t * tmpb = malloc(size * sizeof(uint32_t));
    if (!tmpb) { free(tmpa); return errno; }
    int tmpbstride = sizeof(*tmpb);

    /* allocate six buckets to count occurrences and offsets for
     * 11-digit binary numbers. */
    int64_t * bucketptr = malloc(6*2049 * sizeof(int64_t));
    if (!bucketptr) { free(tmpb); free(tmpa); return errno; }

    int64_t * tmpperm = NULL;
    if (perm) {
        tmpperm = malloc(size * sizeof(int64_t));
        if (!tmpperm) {
            free(bucketptr); free(tmpb); free(tmpa);
            return errno;
        }
    }

    /* 1. count the number of keys in each bucket */
    for (int k = 0; k < 6; k++) {
        for (int j = 0; j <= 2048; j++) {
            bucketptr[k*2049+j] = 0;
        }
    }
    for (int64_t i = 0; i < size; i++) {
        uint32_t x = *(const uint32_t *) ((const char *) a + i*astride);
        uint32_t y = *(const uint32_t *) ((const char *) b + i*bstride);
        bucketptr[0*2049+((y >> (11*0)) & 0x7ff)+1]++;
        bucketptr[1*2049+((y >> (11*1)) & 0x7ff)+1]++;
        bucketptr[2*2049+((y >> (11*2)) & 0x3ff)+1]++;
        bucketptr[3*2049+((x >> (11*0)) & 0x7ff)+1]++;
        bucketptr[4*2049+((x >> (11*1)) & 0x7ff)+1]++;
        bucketptr[5*2049+((x >> (11*2)) & 0x3ff)+1]++;
    }

    /* 2. compute offset to first key in each bucket */
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
    if (perm) {
        for (int64_t i = 0; i < size; i++) {
            uint32_t x = *(const uint32_t *) ((const char *) a + i*astride);
            uint32_t y = *(const uint32_t *) ((const char *) b + i*bstride);
            int64_t dst = bucketptr[0*2049+((y >> (11*0)) & 0x7ff)]++;
            *(uint32_t *) ((char *) tmpa + dst*tmpastride) = x;
            *(uint32_t *) ((char *) tmpb + dst*tmpbstride) = y;
            perm[dst] = i;
        }
        swap_uint32ptr(&a, &tmpa); swap_int(&astride, &tmpastride);
        swap_uint32ptr(&b, &tmpb); swap_int(&bstride, &tmpbstride);
        for (int64_t i = 0; i < size; i++) {
            uint32_t x = *(const uint32_t *) ((const char *) a + i*astride);
            uint32_t y = *(const uint32_t *) ((const char *) b + i*bstride);
            int64_t dst = bucketptr[1*2049+((y >> (11*1)) & 0x7ff)]++;
            *(uint32_t *) ((char *) tmpa + dst*tmpastride) = x;
            *(uint32_t *) ((char *) tmpb + dst*tmpbstride) = y;
            tmpperm[dst] = perm[i];
        }
        swap_uint32ptr(&a, &tmpa); swap_int(&astride, &tmpastride);
        swap_uint32ptr(&b, &tmpb); swap_int(&bstride, &tmpbstride);
        swap_int64ptr(&perm, &tmpperm);
        for (int64_t i = 0; i < size; i++) {
            uint32_t x = *(const uint32_t *) ((const char *) a + i*astride);
            uint32_t y = *(const uint32_t *) ((const char *) b + i*bstride);
            int64_t dst = bucketptr[2*2049+((y >> (11*2)) & 0x3ff)]++;
            *(uint32_t *) ((char *) tmpa + dst*tmpastride) = x;
            *(uint32_t *) ((char *) tmpb + dst*tmpbstride) = y;
            tmpperm[dst] = perm[i];
        }
        swap_uint32ptr(&a, &tmpa); swap_int(&astride, &tmpastride);
        swap_uint32ptr(&b, &tmpb); swap_int(&bstride, &tmpbstride);
        swap_int64ptr(&perm, &tmpperm);
        for (int64_t i = 0; i < size; i++) {
            uint32_t x = *(const uint32_t *) ((const char *) a + i*astride);
            uint32_t y = *(const uint32_t *) ((const char *) b + i*bstride);
            int64_t dst = bucketptr[3*2049+((x >> (11*0)) & 0x7ff)]++;
            *(uint32_t *) ((char *) tmpa + dst*tmpastride) = x;
            *(uint32_t *) ((char *) tmpb + dst*tmpbstride) = y;
            tmpperm[dst] = perm[i];
        }
        swap_uint32ptr(&a, &tmpa); swap_int(&astride, &tmpastride);
        swap_uint32ptr(&b, &tmpb); swap_int(&bstride, &tmpbstride);
        swap_int64ptr(&perm, &tmpperm);
        for (int64_t i = 0; i < size; i++) {
            uint32_t x = *(const uint32_t *) ((const char *) a + i*astride);
            uint32_t y = *(const uint32_t *) ((const char *) b + i*bstride);
            int64_t dst = bucketptr[4*2049+((x >> (11*1)) & 0x7ff)]++;
            *(uint32_t *) ((char *) tmpa + dst*tmpastride) = x;
            *(uint32_t *) ((char *) tmpb + dst*tmpbstride) = y;
            tmpperm[dst] = perm[i];
        }
        swap_uint32ptr(&a, &tmpa); swap_int(&astride, &tmpastride);
        swap_uint32ptr(&b, &tmpb); swap_int(&bstride, &tmpbstride);
        swap_int64ptr(&perm, &tmpperm);
        for (int64_t i = 0; i < size; i++) {
            uint32_t x = *(const uint32_t *) ((const char *) a + i*astride);
            uint32_t y = *(const uint32_t *) ((const char *) b + i*bstride);
            int64_t dst = bucketptr[5*2049+((x >> (11*2)) & 0x3ff)]++;
            *(uint32_t *) ((char *) tmpa + dst*tmpastride) = x;
            *(uint32_t *) ((char *) tmpb + dst*tmpbstride) = y;
            tmpperm[dst] = perm[i];
        }
        swap_uint32ptr(&a, &tmpa); swap_int(&astride, &tmpastride);
        swap_uint32ptr(&b, &tmpb); swap_int(&bstride, &tmpbstride);
        for (int64_t j = 0; j < size; j++) perm[tmpperm[j]] = j;
    } else {
        for (int64_t i = 0; i < size; i++) {
            uint32_t x = *(const uint32_t *) ((const char *) a + i*astride);
            uint32_t y = *(const uint32_t *) ((const char *) b + i*bstride);
            int64_t dst = bucketptr[0*2049+((y >> (11*0)) & 0x7ff)]++;
            *(uint32_t *) ((char *) tmpa + dst*tmpastride) = x;
            *(uint32_t *) ((char *) tmpb + dst*tmpbstride) = y;
        }
        swap_uint32ptr(&a, &tmpa); swap_int(&astride, &tmpastride);
        swap_uint32ptr(&b, &tmpb); swap_int(&bstride, &tmpbstride);
        for (int64_t i = 0; i < size; i++) {
            uint32_t x = *(const uint32_t *) ((const char *) a + i*astride);
            uint32_t y = *(const uint32_t *) ((const char *) b + i*bstride);
            int64_t dst = bucketptr[1*2049+((y >> (11*1)) & 0x7ff)]++;
            *(uint32_t *) ((char *) tmpa + dst*tmpastride) = x;
            *(uint32_t *) ((char *) tmpb + dst*tmpbstride) = y;
        }
        swap_uint32ptr(&a, &tmpa); swap_int(&astride, &tmpastride);
        swap_uint32ptr(&b, &tmpb); swap_int(&bstride, &tmpbstride);
        for (int64_t i = 0; i < size; i++) {
            uint32_t x = *(const uint32_t *) ((const char *) a + i*astride);
            uint32_t y = *(const uint32_t *) ((const char *) b + i*bstride);
            int64_t dst = bucketptr[2*2049+((y >> (11*2)) & 0x3ff)]++;
            *(uint32_t *) ((char *) tmpa + dst*tmpastride) = x;
            *(uint32_t *) ((char *) tmpb + dst*tmpbstride) = y;
        }
        swap_uint32ptr(&a, &tmpa); swap_int(&astride, &tmpastride);
        swap_uint32ptr(&b, &tmpb); swap_int(&bstride, &tmpbstride);
        for (int64_t i = 0; i < size; i++) {
            uint32_t x = *(const uint32_t *) ((const char *) a + i*astride);
            uint32_t y = *(const uint32_t *) ((const char *) b + i*bstride);
            int64_t dst = bucketptr[3*2049+((x >> (11*0)) & 0x7ff)]++;
            *(uint32_t *) ((char *) tmpa + dst*tmpastride) = x;
            *(uint32_t *) ((char *) tmpb + dst*tmpbstride) = y;
        }
        swap_uint32ptr(&a, &tmpa); swap_int(&astride, &tmpastride);
        swap_uint32ptr(&b, &tmpb); swap_int(&bstride, &tmpbstride);
        for (int64_t i = 0; i < size; i++) {
            uint32_t x = *(const uint32_t *) ((const char *) a + i*astride);
            uint32_t y = *(const uint32_t *) ((const char *) b + i*bstride);
            int64_t dst = bucketptr[4*2049+((x >> (11*1)) & 0x7ff)]++;
            *(uint32_t *) ((char *) tmpa + dst*tmpastride) = x;
            *(uint32_t *) ((char *) tmpb + dst*tmpbstride) = y;
        }
        swap_uint32ptr(&a, &tmpa); swap_int(&astride, &tmpastride);
        swap_uint32ptr(&b, &tmpb); swap_int(&bstride, &tmpbstride);
        for (int64_t i = 0; i < size; i++) {
            uint32_t x = *(const uint32_t *) ((const char *) a + i*astride);
            uint32_t y = *(const uint32_t *) ((const char *) b + i*bstride);
            int64_t dst = bucketptr[5*2049+((x >> (11*2)) & 0x3ff)]++;
            *(uint32_t *) ((char *) tmpa + dst*tmpastride) = x;
            *(uint32_t *) ((char *) tmpb + dst*tmpbstride) = y;
        }
        swap_uint32ptr(&a, &tmpa); swap_int(&astride, &tmpastride);
        swap_uint32ptr(&b, &tmpb); swap_int(&bstride, &tmpbstride);
    }

    if (perm) free(tmpperm);
    free(bucketptr); free(tmpb); free(tmpa);
    return 0;
}

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
    int64_t * perm)
{
    uint64_t * tmpa = malloc(size * sizeof(uint64_t));
    int tmpastride = sizeof(*tmpa);
    if (!tmpa) return errno;
    uint64_t * tmpb = malloc(size * sizeof(uint64_t));
    if (!tmpb) { free(tmpa); return errno; }
    int tmpbstride = sizeof(*tmpb);

    /* allocate twelve buckets to count occurrences and offsets for
     * 11-digit binary numbers. */
    int64_t * bucketptr = malloc(12*2049 * sizeof(int64_t));
    if (!bucketptr) { free(tmpb); free(tmpa); return errno; }

    int64_t * tmpperm = NULL;
    if (perm) {
        tmpperm = malloc(size * sizeof(int64_t));
        if (!tmpperm) {
            free(bucketptr); free(tmpb); free(tmpa);
            return errno;
        }
    }

    /* 1. count the number of keys in each bucket */
    for (int k = 0; k < 12; k++) {
        for (int j = 0; j <= 2048; j++) {
            bucketptr[k*2049+j] = 0;
        }
    }
    for (int64_t i = 0; i < size; i++) {
        uint64_t x = *(const uint64_t *) ((const char *) a + i*astride);
        uint64_t y = *(const uint64_t *) ((const char *) b + i*bstride);
        bucketptr[ 0*2049+((y >> (11*0)) & 0x7ff)+1]++;
        bucketptr[ 1*2049+((y >> (11*1)) & 0x7ff)+1]++;
        bucketptr[ 2*2049+((y >> (11*2)) & 0x7ff)+1]++;
        bucketptr[ 3*2049+((y >> (11*3)) & 0x7ff)+1]++;
        bucketptr[ 4*2049+((y >> (11*4)) & 0x7ff)+1]++;
        bucketptr[ 5*2049+((y >> (11*5)) & 0x1ff)+1]++;
        bucketptr[ 6*2049+((x >> (11*0)) & 0x7ff)+1]++;
        bucketptr[ 7*2049+((x >> (11*1)) & 0x7ff)+1]++;
        bucketptr[ 8*2049+((x >> (11*2)) & 0x7ff)+1]++;
        bucketptr[ 9*2049+((x >> (11*3)) & 0x7ff)+1]++;
        bucketptr[10*2049+((x >> (11*4)) & 0x7ff)+1]++;
        bucketptr[11*2049+((x >> (11*5)) & 0x1ff)+1]++;
    }

    /* 2. compute offset to first key in each bucket */
    for (int k = 0; k < 12; k++) {
        for (int j = 0; j < 2048; j++) {
            bucketptr[k*2049+j+1] += bucketptr[k*2049+j];
        }
    }

    /*
     * 3. Sort in 12 rounds with 11 binary digits treated in each
     * round. Note that pointers to the original and auxiliary arrays
     * of keys (and sorting permutation) are swapped after each round.
     * There is an even number of swaps, so that the sorted keys (and
     * the final sorting permutation) end up in the original array.
     *
     * The choice of using 11 bits in each round is described in the
     * article "Radix Tricks" by Michael Herf, published online in
     * December 2001 at http://stereopsis.com/radix.html.
     */
    if (perm) {
        for (int64_t i = 0; i < size; i++) {
            uint64_t x = *(const uint64_t *) ((const char *) a + i*astride);
            uint64_t y = *(const uint64_t *) ((const char *) b + i*bstride);
            int64_t dst = bucketptr[0*2049+((y >> (11*0)) & 0x7ff)]++;
            *(uint64_t *) ((char *) tmpa + dst*tmpastride) = x;
            *(uint64_t *) ((char *) tmpb + dst*tmpbstride) = y;
            perm[dst] = i;
        }
        swap_uint64ptr(&a, &tmpa); swap_int(&astride, &tmpastride);
        swap_uint64ptr(&b, &tmpb); swap_int(&bstride, &tmpbstride);
        for (int64_t i = 0; i < size; i++) {
            uint64_t x = *(const uint64_t *) ((const char *) a + i*astride);
            uint64_t y = *(const uint64_t *) ((const char *) b + i*bstride);
            int64_t dst = bucketptr[1*2049+((y >> (11*1)) & 0x7ff)]++;
            *(uint64_t *) ((char *) tmpa + dst*tmpastride) = x;
            *(uint64_t *) ((char *) tmpb + dst*tmpbstride) = y;
            tmpperm[dst] = perm[i];
        }
        swap_uint64ptr(&a, &tmpa); swap_int(&astride, &tmpastride);
        swap_uint64ptr(&b, &tmpb); swap_int(&bstride, &tmpbstride);
        swap_int64ptr(&perm, &tmpperm);
        for (int64_t i = 0; i < size; i++) {
            uint64_t x = *(const uint64_t *) ((const char *) a + i*astride);
            uint64_t y = *(const uint64_t *) ((const char *) b + i*bstride);
            int64_t dst = bucketptr[2*2049+((y >> (11*2)) & 0x7ff)]++;
            *(uint64_t *) ((char *) tmpa + dst*tmpastride) = x;
            *(uint64_t *) ((char *) tmpb + dst*tmpbstride) = y;
            tmpperm[dst] = perm[i];
        }
        swap_uint64ptr(&a, &tmpa); swap_int(&astride, &tmpastride);
        swap_uint64ptr(&b, &tmpb); swap_int(&bstride, &tmpbstride);
        swap_int64ptr(&perm, &tmpperm);
        for (int64_t i = 0; i < size; i++) {
            uint64_t x = *(const uint64_t *) ((const char *) a + i*astride);
            uint64_t y = *(const uint64_t *) ((const char *) b + i*bstride);
            int64_t dst = bucketptr[3*2049+((y >> (11*3)) & 0x7ff)]++;
            *(uint64_t *) ((char *) tmpa + dst*tmpastride) = x;
            *(uint64_t *) ((char *) tmpb + dst*tmpbstride) = y;
            tmpperm[dst] = perm[i];
        }
        swap_uint64ptr(&a, &tmpa); swap_int(&astride, &tmpastride);
        swap_uint64ptr(&b, &tmpb); swap_int(&bstride, &tmpbstride);
        swap_int64ptr(&perm, &tmpperm);
        for (int64_t i = 0; i < size; i++) {
            uint64_t x = *(const uint64_t *) ((const char *) a + i*astride);
            uint64_t y = *(const uint64_t *) ((const char *) b + i*bstride);
            int64_t dst = bucketptr[4*2049+((y >> (11*4)) & 0x7ff)]++;
            *(uint64_t *) ((char *) tmpa + dst*tmpastride) = x;
            *(uint64_t *) ((char *) tmpb + dst*tmpbstride) = y;
            tmpperm[dst] = perm[i];
        }
        swap_uint64ptr(&a, &tmpa); swap_int(&astride, &tmpastride);
        swap_uint64ptr(&b, &tmpb); swap_int(&bstride, &tmpbstride);
        swap_int64ptr(&perm, &tmpperm);
        for (int64_t i = 0; i < size; i++) {
            uint64_t x = *(const uint64_t *) ((const char *) a + i*astride);
            uint64_t y = *(const uint64_t *) ((const char *) b + i*bstride);
            int64_t dst = bucketptr[5*2049+((y >> (11*5)) & 0x1ff)]++;
            *(uint64_t *) ((char *) tmpa + dst*tmpastride) = x;
            *(uint64_t *) ((char *) tmpb + dst*tmpbstride) = y;
            tmpperm[dst] = perm[i];
        }
        swap_uint64ptr(&a, &tmpa); swap_int(&astride, &tmpastride);
        swap_uint64ptr(&b, &tmpb); swap_int(&bstride, &tmpbstride);
        swap_int64ptr(&perm, &tmpperm);
        for (int64_t i = 0; i < size; i++) {
            uint64_t x = *(const uint64_t *) ((const char *) a + i*astride);
            uint64_t y = *(const uint64_t *) ((const char *) b + i*bstride);
            int64_t dst = bucketptr[6*2049+((x >> (11*0)) & 0x7ff)]++;
            *(uint64_t *) ((char *) tmpa + dst*tmpastride) = x;
            *(uint64_t *) ((char *) tmpb + dst*tmpbstride) = y;
            tmpperm[dst] = perm[i];
        }
        swap_uint64ptr(&a, &tmpa); swap_int(&astride, &tmpastride);
        swap_uint64ptr(&b, &tmpb); swap_int(&bstride, &tmpbstride);
        swap_int64ptr(&perm, &tmpperm);
        for (int64_t i = 0; i < size; i++) {
            uint64_t x = *(const uint64_t *) ((const char *) a + i*astride);
            uint64_t y = *(const uint64_t *) ((const char *) b + i*bstride);
            int64_t dst = bucketptr[7*2049+((x >> (11*1)) & 0x7ff)]++;
            *(uint64_t *) ((char *) tmpa + dst*tmpastride) = x;
            *(uint64_t *) ((char *) tmpb + dst*tmpbstride) = y;
            tmpperm[dst] = perm[i];
        }
        swap_uint64ptr(&a, &tmpa); swap_int(&astride, &tmpastride);
        swap_uint64ptr(&b, &tmpb); swap_int(&bstride, &tmpbstride);
        swap_int64ptr(&perm, &tmpperm);
        for (int64_t i = 0; i < size; i++) {
            uint64_t x = *(const uint64_t *) ((const char *) a + i*astride);
            uint64_t y = *(const uint64_t *) ((const char *) b + i*bstride);
            int64_t dst = bucketptr[8*2049+((x >> (11*2)) & 0x7ff)]++;
            *(uint64_t *) ((char *) tmpa + dst*tmpastride) = x;
            *(uint64_t *) ((char *) tmpb + dst*tmpbstride) = y;
            tmpperm[dst] = perm[i];
        }
        swap_uint64ptr(&a, &tmpa); swap_int(&astride, &tmpastride);
        swap_uint64ptr(&b, &tmpb); swap_int(&bstride, &tmpbstride);
        swap_int64ptr(&perm, &tmpperm);
        for (int64_t i = 0; i < size; i++) {
            uint64_t x = *(const uint64_t *) ((const char *) a + i*astride);
            uint64_t y = *(const uint64_t *) ((const char *) b + i*bstride);
            int64_t dst = bucketptr[9*2049+((x >> (11*3)) & 0x7ff)]++;
            *(uint64_t *) ((char *) tmpa + dst*tmpastride) = x;
            *(uint64_t *) ((char *) tmpb + dst*tmpbstride) = y;
            tmpperm[dst] = perm[i];
        }
        swap_uint64ptr(&a, &tmpa); swap_int(&astride, &tmpastride);
        swap_uint64ptr(&b, &tmpb); swap_int(&bstride, &tmpbstride);
        swap_int64ptr(&perm, &tmpperm);
        for (int64_t i = 0; i < size; i++) {
            uint64_t x = *(const uint64_t *) ((const char *) a + i*astride);
            uint64_t y = *(const uint64_t *) ((const char *) b + i*bstride);
            int64_t dst = bucketptr[10*2049+((x >> (11*4)) & 0x7ff)]++;
            *(uint64_t *) ((char *) tmpa + dst*tmpastride) = x;
            *(uint64_t *) ((char *) tmpb + dst*tmpbstride) = y;
            tmpperm[dst] = perm[i];
        }
        swap_uint64ptr(&a, &tmpa); swap_int(&astride, &tmpastride);
        swap_uint64ptr(&b, &tmpb); swap_int(&bstride, &tmpbstride);
        swap_int64ptr(&perm, &tmpperm);
        for (int64_t i = 0; i < size; i++) {
            uint64_t x = *(const uint64_t *) ((const char *) a + i*astride);
            uint64_t y = *(const uint64_t *) ((const char *) b + i*bstride);
            int64_t dst = bucketptr[11*2049+((x >> (11*5)) & 0x1ff)]++;
            *(uint64_t *) ((char *) tmpa + dst*tmpastride) = x;
            *(uint64_t *) ((char *) tmpb + dst*tmpbstride) = y;
            tmpperm[dst] = perm[i];
        }
        swap_uint64ptr(&a, &tmpa); swap_int(&astride, &tmpastride);
        swap_uint64ptr(&b, &tmpb); swap_int(&bstride, &tmpbstride);
        for (int64_t j = 0; j < size; j++) perm[tmpperm[j]] = j;
    } else {
        for (int64_t i = 0; i < size; i++) {
            uint64_t x = *(const uint64_t *) ((const char *) a + i*astride);
            uint64_t y = *(const uint64_t *) ((const char *) b + i*bstride);
            int64_t dst = bucketptr[0*2049+((y >> (11*0)) & 0x7ff)]++;
            *(uint64_t *) ((char *) tmpa + dst*tmpastride) = x;
            *(uint64_t *) ((char *) tmpb + dst*tmpbstride) = y;
        }
        swap_uint64ptr(&a, &tmpa); swap_int(&astride, &tmpastride);
        swap_uint64ptr(&b, &tmpb); swap_int(&bstride, &tmpbstride);
        for (int64_t i = 0; i < size; i++) {
            uint64_t x = *(const uint64_t *) ((const char *) a + i*astride);
            uint64_t y = *(const uint64_t *) ((const char *) b + i*bstride);
            int64_t dst = bucketptr[1*2049+((y >> (11*1)) & 0x7ff)]++;
            *(uint64_t *) ((char *) tmpa + dst*tmpastride) = x;
            *(uint64_t *) ((char *) tmpb + dst*tmpbstride) = y;
        }
        swap_uint64ptr(&a, &tmpa); swap_int(&astride, &tmpastride);
        swap_uint64ptr(&b, &tmpb); swap_int(&bstride, &tmpbstride);
        for (int64_t i = 0; i < size; i++) {
            uint64_t x = *(const uint64_t *) ((const char *) a + i*astride);
            uint64_t y = *(const uint64_t *) ((const char *) b + i*bstride);
            int64_t dst = bucketptr[2*2049+((y >> (11*2)) & 0x7ff)]++;
            *(uint64_t *) ((char *) tmpa + dst*tmpastride) = x;
            *(uint64_t *) ((char *) tmpb + dst*tmpbstride) = y;
        }
        swap_uint64ptr(&a, &tmpa); swap_int(&astride, &tmpastride);
        swap_uint64ptr(&b, &tmpb); swap_int(&bstride, &tmpbstride);
        for (int64_t i = 0; i < size; i++) {
            uint64_t x = *(const uint64_t *) ((const char *) a + i*astride);
            uint64_t y = *(const uint64_t *) ((const char *) b + i*bstride);
            int64_t dst = bucketptr[3*2049+((y >> (11*3)) & 0x7ff)]++;
            *(uint64_t *) ((char *) tmpa + dst*tmpastride) = x;
            *(uint64_t *) ((char *) tmpb + dst*tmpbstride) = y;
        }
        swap_uint64ptr(&a, &tmpa); swap_int(&astride, &tmpastride);
        swap_uint64ptr(&b, &tmpb); swap_int(&bstride, &tmpbstride);
        for (int64_t i = 0; i < size; i++) {
            uint64_t x = *(const uint64_t *) ((const char *) a + i*astride);
            uint64_t y = *(const uint64_t *) ((const char *) b + i*bstride);
            int64_t dst = bucketptr[4*2049+((y >> (11*4)) & 0x7ff)]++;
            *(uint64_t *) ((char *) tmpa + dst*tmpastride) = x;
            *(uint64_t *) ((char *) tmpb + dst*tmpbstride) = y;
        }
        swap_uint64ptr(&a, &tmpa); swap_int(&astride, &tmpastride);
        swap_uint64ptr(&b, &tmpb); swap_int(&bstride, &tmpbstride);
        for (int64_t i = 0; i < size; i++) {
            uint64_t x = *(const uint64_t *) ((const char *) a + i*astride);
            uint64_t y = *(const uint64_t *) ((const char *) b + i*bstride);
            int64_t dst = bucketptr[5*2049+((y >> (11*5)) & 0x1ff)]++;
            *(uint64_t *) ((char *) tmpa + dst*tmpastride) = x;
            *(uint64_t *) ((char *) tmpb + dst*tmpbstride) = y;
        }
        swap_uint64ptr(&a, &tmpa); swap_int(&astride, &tmpastride);
        swap_uint64ptr(&b, &tmpb); swap_int(&bstride, &tmpbstride);
        for (int64_t i = 0; i < size; i++) {
            uint64_t x = *(const uint64_t *) ((const char *) a + i*astride);
            uint64_t y = *(const uint64_t *) ((const char *) b + i*bstride);
            int64_t dst = bucketptr[6*2049+((x >> (11*0)) & 0x7ff)]++;
            *(uint64_t *) ((char *) tmpa + dst*tmpastride) = x;
            *(uint64_t *) ((char *) tmpb + dst*tmpbstride) = y;
        }
        swap_uint64ptr(&a, &tmpa); swap_int(&astride, &tmpastride);
        swap_uint64ptr(&b, &tmpb); swap_int(&bstride, &tmpbstride);
        for (int64_t i = 0; i < size; i++) {
            uint64_t x = *(const uint64_t *) ((const char *) a + i*astride);
            uint64_t y = *(const uint64_t *) ((const char *) b + i*bstride);
            int64_t dst = bucketptr[7*2049+((x >> (11*1)) & 0x7ff)]++;
            *(uint64_t *) ((char *) tmpa + dst*tmpastride) = x;
            *(uint64_t *) ((char *) tmpb + dst*tmpbstride) = y;
        }
        swap_uint64ptr(&a, &tmpa); swap_int(&astride, &tmpastride);
        swap_uint64ptr(&b, &tmpb); swap_int(&bstride, &tmpbstride);
        for (int64_t i = 0; i < size; i++) {
            uint64_t x = *(const uint64_t *) ((const char *) a + i*astride);
            uint64_t y = *(const uint64_t *) ((const char *) b + i*bstride);
            int64_t dst = bucketptr[8*2049+((x >> (11*2)) & 0x7ff)]++;
            *(uint64_t *) ((char *) tmpa + dst*tmpastride) = x;
            *(uint64_t *) ((char *) tmpb + dst*tmpbstride) = y;
        }
        swap_uint64ptr(&a, &tmpa); swap_int(&astride, &tmpastride);
        swap_uint64ptr(&b, &tmpb); swap_int(&bstride, &tmpbstride);
        for (int64_t i = 0; i < size; i++) {
            uint64_t x = *(const uint64_t *) ((const char *) a + i*astride);
            uint64_t y = *(const uint64_t *) ((const char *) b + i*bstride);
            int64_t dst = bucketptr[9*2049+((x >> (11*3)) & 0x7ff)]++;
            *(uint64_t *) ((char *) tmpa + dst*tmpastride) = x;
            *(uint64_t *) ((char *) tmpb + dst*tmpbstride) = y;
        }
        swap_uint64ptr(&a, &tmpa); swap_int(&astride, &tmpastride);
        swap_uint64ptr(&b, &tmpb); swap_int(&bstride, &tmpbstride);
        for (int64_t i = 0; i < size; i++) {
            uint64_t x = *(const uint64_t *) ((const char *) a + i*astride);
            uint64_t y = *(const uint64_t *) ((const char *) b + i*bstride);
            int64_t dst = bucketptr[10*2049+((x >> (11*4)) & 0x7ff)]++;
            *(uint64_t *) ((char *) tmpa + dst*tmpastride) = x;
            *(uint64_t *) ((char *) tmpb + dst*tmpbstride) = y;
        }
        swap_uint64ptr(&a, &tmpa); swap_int(&astride, &tmpastride);
        swap_uint64ptr(&b, &tmpb); swap_int(&bstride, &tmpbstride);
        for (int64_t i = 0; i < size; i++) {
            uint64_t x = *(const uint64_t *) ((const char *) a + i*astride);
            uint64_t y = *(const uint64_t *) ((const char *) b + i*bstride);
            int64_t dst = bucketptr[11*2049+((x >> (11*5)) & 0x1ff)]++;
            *(uint64_t *) ((char *) tmpa + dst*tmpastride) = x;
            *(uint64_t *) ((char *) tmpb + dst*tmpbstride) = y;
        }
        swap_uint64ptr(&a, &tmpa); swap_int(&astride, &tmpastride);
        swap_uint64ptr(&b, &tmpb); swap_int(&bstride, &tmpbstride);
    }

    if (perm) free(tmpperm);
    free(bucketptr); free(tmpb); free(tmpa);
    return 0;
}

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
    int64_t * perm)
{
    for (int64_t i = 0; i < size; i++) {
        *(int32_t *)((char *) a+i*astride) ^= INT32_MIN;
        *(int32_t *)((char *) b+i*bstride) ^= INT32_MIN;
    }
    int err = radix_sort_uint32_pair(
        size, astride, (uint32_t *) a, bstride, (uint32_t *) b, perm);
    for (int64_t i = 0; i < size; i++) {
        *(int32_t *)((char *) a+i*astride) ^= INT32_MIN;
        *(int32_t *)((char *) b+i*bstride) ^= INT32_MIN;
    }
    return err;
}

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
    int64_t * perm)
{
    for (int64_t i = 0; i < size; i++) {
        *(int64_t *)((char *) a+i*astride) ^= INT64_MIN;
        *(int64_t *)((char *) b+i*bstride) ^= INT64_MIN;
    }
    int err = radix_sort_uint64_pair(
        size, astride, (uint64_t *) a, bstride, (uint64_t *) b, perm);
    for (int64_t i = 0; i < size; i++) {
        *(int64_t *)((char *) a+i*astride) ^= INT64_MIN;
        *(int64_t *)((char *) b+i*bstride) ^= INT64_MIN;
    }
    return err;
}

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
    int64_t * perm)
{
    if (sizeof(int) == sizeof(int32_t)) {
        return radix_sort_int32_pair(
            size, astride, (int32_t *) a, bstride, b, perm);
    } else if (sizeof(int) == sizeof(int64_t)) {
        return radix_sort_int64_pair(
            size, astride, (int64_t *) a, bstride, (int64_t *) b, perm);
    } else { return ENOTSUP; }
}
