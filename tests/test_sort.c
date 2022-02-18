/* This file is part of Libmtx.
 *
 * Copyright (C) 2021 James D. Trotter
 *
 * Libmtx is free software: you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * Libmtx is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Libmtx.  If not, see
 * <https://www.gnu.org/licenses/>.
 *
 * Authors: James D. Trotter <james@simula.no>
 * Last modified: 2021-30-11
 *
 * Unit tests for sorting functions.
 */

#include <libmtx/error.h>

#include "libmtx/util/sort.h"
#include "test.h"

#include <errno.h>

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * ‘test_counting_sort()’ tests a counting sort algorithm.
 */
int test_counting_sort(void)
{
    {
        int err;
        int64_t size = 5;
        uint8_t keys[5] = {0,255,30,1,2};
        uint8_t sorted_keys[5];
        int64_t permutation[5];
        err = counting_sort_uint8(
            size, sizeof(*keys), keys,
            sizeof(*sorted_keys), sorted_keys,
            permutation, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(  0, sorted_keys[0]);
        TEST_ASSERT_EQ(  1, sorted_keys[1]);
        TEST_ASSERT_EQ(  2, sorted_keys[2]);
        TEST_ASSERT_EQ( 30, sorted_keys[3]);
        TEST_ASSERT_EQ(255, sorted_keys[4]);
        TEST_ASSERT_EQ(0, permutation[0]);
        TEST_ASSERT_EQ(4, permutation[1]);
        TEST_ASSERT_EQ(3, permutation[2]);
        TEST_ASSERT_EQ(1, permutation[3]);
        TEST_ASSERT_EQ(2, permutation[4]);
    }

    {
        int err;
        int64_t size = 5;
        uint16_t keys[5] = {0,255,30,1,2};
        uint16_t sorted_keys[5];
        int64_t permutation[5];
        err = counting_sort_uint16(
            size, sizeof(*keys), keys,
            sizeof(*sorted_keys), sorted_keys,
            permutation, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(  0, sorted_keys[0]);
        TEST_ASSERT_EQ(  1, sorted_keys[1]);
        TEST_ASSERT_EQ(  2, sorted_keys[2]);
        TEST_ASSERT_EQ( 30, sorted_keys[3]);
        TEST_ASSERT_EQ(255, sorted_keys[4]);
        TEST_ASSERT_EQ(0, permutation[0]);
        TEST_ASSERT_EQ(4, permutation[1]);
        TEST_ASSERT_EQ(3, permutation[2]);
        TEST_ASSERT_EQ(1, permutation[3]);
        TEST_ASSERT_EQ(2, permutation[4]);
    }

    return TEST_SUCCESS;
}

#include <time.h>

/**
 * `timespec_duration()` is the duration, in seconds, elapsed between
 * two given time points.
 */
static double timespec_duration(
    struct timespec t0,
    struct timespec t1)
{
    return (t1.tv_sec - t0.tv_sec) +
        (t1.tv_nsec - t0.tv_nsec) * 1e-9;
}

/* A simple, linear congruential random number generator. */
static uint64_t rand_uint64(void)
{
    static uint64_t i = 1;
    return (i = (164603309694725029ull * i) % 14738995463583502973ull);
}

/**
 * ‘test_radix_sort()’ tests a radix sort algorithm.
 */
int test_radix_sort(void)
{
    /*
     * 32-bit unsigned integer keys.
     */

    {
        int err;
        int64_t size = 5;
        uint32_t keys[5] = {0,255,30,1,2};
        int64_t permutation[5];
        err = radix_sort_uint32(size, keys, permutation);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(  0, keys[0]);
        TEST_ASSERT_EQ(  1, keys[1]);
        TEST_ASSERT_EQ(  2, keys[2]);
        TEST_ASSERT_EQ( 30, keys[3]);
        TEST_ASSERT_EQ(255, keys[4]);
        TEST_ASSERT_EQ(0, permutation[0]);
        TEST_ASSERT_EQ(4, permutation[1]);
        TEST_ASSERT_EQ(3, permutation[2]);
        TEST_ASSERT_EQ(1, permutation[3]);
        TEST_ASSERT_EQ(2, permutation[4]);
    }

    {
        int err;
        int64_t size = 5;
        uint32_t keys[5] = {25820, 24732, 1352, 34041, 38784};
        int64_t permutation[5];
        err = radix_sort_uint32(size, keys, permutation);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( 1352, keys[0]);
        TEST_ASSERT_EQ(24732, keys[1]);
        TEST_ASSERT_EQ(25820, keys[2]);
        TEST_ASSERT_EQ(34041, keys[3]);
        TEST_ASSERT_EQ(38784, keys[4]);
        TEST_ASSERT_EQ(2, permutation[0]);
        TEST_ASSERT_EQ(1, permutation[1]);
        TEST_ASSERT_EQ(0, permutation[2]);
        TEST_ASSERT_EQ(3, permutation[3]);
        TEST_ASSERT_EQ(4, permutation[4]);
    }

    {
        int err;
        int64_t size = 100;
        uint32_t keys[100];
        srand(415);
        for (int i = 0; i < size; i++)
            keys[i] = rand() % UINT32_MAX;
        err = radix_sort_uint32(size, keys, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        for (int i = 1; i < size; i++)
            TEST_ASSERT_LE_MSG(keys[i-1], keys[i], "i=%d, keys[i-1]=%"PRIu32", keys[i]=%"PRIu32"",
                               i, keys[i-1], keys[i]);
    }

    {
        int err;
        int64_t size = 1000;
        uint32_t * keys = malloc(size * sizeof(uint32_t));
        TEST_ASSERT_NEQ_MSG(NULL, keys, "%s", strerror(errno));
        srand(415);
        for (int i = 0; i < size; i++)
            keys[i] = rand() % UINT32_MAX;
        int64_t * sorting_permutation = malloc(size * sizeof(int64_t));
        TEST_ASSERT_NEQ_MSG(NULL, sorting_permutation, "%s", strerror(errno));

        struct timespec t0, t1;
        fprintf(stderr, "radix_sort_uint32: ");
        fflush(stderr);
        clock_gettime(CLOCK_MONOTONIC, &t0);

        err = radix_sort_uint32(size, keys, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        clock_gettime(CLOCK_MONOTONIC, &t1);
        fprintf(stderr, "%'.6f seconds (%'.1f MB/s)\n",
                timespec_duration(t0, t1),
                1.0e-6 * (size * sizeof(*keys)) / timespec_duration(t0, t1));

        free(sorting_permutation);
        free(keys);
    }

    /*
     * 64-bit unsigned integer keys.
     */

    {
        int err;
        int64_t size = 5;
        uint64_t keys[5] = {0,255,30,1,2};
        int64_t permutation[5];
        err = radix_sort_uint64(size, keys, permutation);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(  0, keys[0]);
        TEST_ASSERT_EQ(  1, keys[1]);
        TEST_ASSERT_EQ(  2, keys[2]);
        TEST_ASSERT_EQ( 30, keys[3]);
        TEST_ASSERT_EQ(255, keys[4]);
        TEST_ASSERT_EQ(0, permutation[0]);
        TEST_ASSERT_EQ(4, permutation[1]);
        TEST_ASSERT_EQ(3, permutation[2]);
        TEST_ASSERT_EQ(1, permutation[3]);
        TEST_ASSERT_EQ(2, permutation[4]);
    }

    {
        int err;
        int64_t size = 5;
        uint64_t keys[5] = {25820, 24732, 1352, 34041, 38784};
        int64_t permutation[5];
        err = radix_sort_uint64(size, keys, permutation);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( 1352, keys[0]);
        TEST_ASSERT_EQ(24732, keys[1]);
        TEST_ASSERT_EQ(25820, keys[2]);
        TEST_ASSERT_EQ(34041, keys[3]);
        TEST_ASSERT_EQ(38784, keys[4]);
        TEST_ASSERT_EQ(2, permutation[0]);
        TEST_ASSERT_EQ(1, permutation[1]);
        TEST_ASSERT_EQ(0, permutation[2]);
        TEST_ASSERT_EQ(3, permutation[3]);
        TEST_ASSERT_EQ(4, permutation[4]);
    }

    {
        int err;
        int64_t size = 100;
        uint64_t keys[100];
        for (int i = 0; i < size; i++)
            keys[i] = rand_uint64();
        err = radix_sort_uint64(size, keys, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        for (int i = 1; i < size; i++)
            TEST_ASSERT_LE_MSG(keys[i-1], keys[i], "i=%d, keys[i-1]=%"PRIu64", keys[i]=%"PRIu64"",
                               i, keys[i-1], keys[i]);
    }

    {
        int err;
        int64_t size = 1000;
        uint64_t * keys = malloc(size * sizeof(uint64_t));
        TEST_ASSERT_NEQ_MSG(NULL, keys, "%s", strerror(errno));
        for (int i = 0; i < size; i++)
            keys[i] = rand_uint64();
        int64_t * sorting_permutation = malloc(size * sizeof(int64_t));
        TEST_ASSERT_NEQ_MSG(NULL, sorting_permutation, "%s", strerror(errno));

        struct timespec t0, t1;
        fprintf(stderr, "radix_sort_uint64: ");
        fflush(stderr);
        clock_gettime(CLOCK_MONOTONIC, &t0);

        err = radix_sort_uint64(size, keys, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        clock_gettime(CLOCK_MONOTONIC, &t1);
        fprintf(stderr, "%'.6f seconds (%'.1f MB/s)\n",
                timespec_duration(t0, t1),
                1.0e-6 * (size * sizeof(*keys)) / timespec_duration(t0, t1));

        free(sorting_permutation);
        free(keys);
    }

    /*
     * signed integer keys.
     */

    {
        int err;
        int64_t size = 5;
        int keys[5] = {0,255,-30,1,-2};
        int64_t permutation[5];
        err = radix_sort_int(size, keys, permutation);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(-30, keys[0]);
        TEST_ASSERT_EQ( -2, keys[1]);
        TEST_ASSERT_EQ(  0, keys[2]);
        TEST_ASSERT_EQ(  1, keys[3]);
        TEST_ASSERT_EQ(255, keys[4]);
        TEST_ASSERT_EQ(2, permutation[0]);
        TEST_ASSERT_EQ(4, permutation[1]);
        TEST_ASSERT_EQ(0, permutation[2]);
        TEST_ASSERT_EQ(3, permutation[3]);
        TEST_ASSERT_EQ(1, permutation[4]);
    }

    {
        int err;
        int64_t size = 5;
        int keys[5] = {25820, -24732, -1352, 34041, 38784};
        int64_t permutation[5];
        err = radix_sort_int(size, keys, permutation);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(-24732, keys[0]);
        TEST_ASSERT_EQ( -1352, keys[1]);
        TEST_ASSERT_EQ( 25820, keys[2]);
        TEST_ASSERT_EQ( 34041, keys[3]);
        TEST_ASSERT_EQ( 38784, keys[4]);
        TEST_ASSERT_EQ(2, permutation[0]);
        TEST_ASSERT_EQ(0, permutation[1]);
        TEST_ASSERT_EQ(1, permutation[2]);
        TEST_ASSERT_EQ(3, permutation[3]);
        TEST_ASSERT_EQ(4, permutation[4]);
    }

    {
        int err;
        int64_t size = 100;
        int keys[100];
        srand(415);
        for (int i = 0; i < size; i++)
            keys[i] = rand() - RAND_MAX/2;
        err = radix_sort_int(size, keys, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        for (int i = 1; i < size; i++)
            TEST_ASSERT_LE_MSG(
                keys[i-1], keys[i],
                "i=%d, keys[i-1]=%"PRIu32", keys[i]=%"PRIu32"",
                i, keys[i-1], keys[i]);
    }

    {
        int err;
        int64_t size = 1000;
        int * keys = malloc(size * sizeof(int));
        TEST_ASSERT_NEQ_MSG(NULL, keys, "%s", strerror(errno));
        srand(415);
        for (int i = 0; i < size; i++)
            keys[i] = rand() - RAND_MAX/2;
        int64_t * sorting_permutation = malloc(size * sizeof(int64_t));
        TEST_ASSERT_NEQ_MSG(NULL, sorting_permutation, "%s", strerror(errno));

        struct timespec t0, t1;
        fprintf(stderr, "radix_sort_int: ");
        fflush(stderr);
        clock_gettime(CLOCK_MONOTONIC, &t0);

        err = radix_sort_int(size, keys, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        clock_gettime(CLOCK_MONOTONIC, &t1);
        fprintf(stderr, "%'.6f seconds (%'.1f MB/s)\n",
                timespec_duration(t0, t1),
                1.0e-6 * (size * sizeof(*keys)) / timespec_duration(t0, t1));

        free(sorting_permutation);
        free(keys);
    }

    return TEST_SUCCESS;
}

/**
 * ‘main()’ entry point and test driver.
 */
int main(int argc, char * argv[])
{
    TEST_SUITE_BEGIN("Running tests for sorting functions\n");
    TEST_RUN(test_counting_sort);
    TEST_RUN(test_radix_sort);
    TEST_SUITE_END();
    return (TEST_SUITE_STATUS == TEST_SUCCESS) ?
        EXIT_SUCCESS : EXIT_FAILURE;
}
