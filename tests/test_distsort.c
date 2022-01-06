/* This file is part of libmtx.
 *
 * Copyright (C) 2022 James D. Trotter
 *
 * libmtx is free software: you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * libmtx is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with libmtx.  If not, see
 * <https://www.gnu.org/licenses/>.
 *
 * Authors: James D. Trotter <james@simula.no>
 * Last modified: 2022-01-03
 *
 * Unit tests for sorting functions.
 */

#include <libmtx/libmtx-config.h>

#include <libmtx/error.h>

#include "libmtx/util/sort.h"

#include "test.h"

#include <mpi.h>

#include <errno.h>

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

const char * program_invocation_short_name = "test_distsort";

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
 * ‘test_distradix_sort()’ tests a distributed radix sort algorithm.
 */
int test_distradix_sort(void)
{
    int err;
    char mpierrstr[MPI_MAX_ERROR_STRING];
    int mpierrstrlen;
    MPI_Comm comm = MPI_COMM_WORLD;
    int root = 0;

    int comm_size;
    err = MPI_Comm_size(comm, &comm_size);
    if (err) {
        MPI_Error_string(err, mpierrstr, &mpierrstrlen);
        fprintf(stderr, "%s: MPI_Comm_size failed with %s\n",
                program_invocation_short_name, mpierrstr);
        MPI_Abort(comm, EXIT_FAILURE);
    }
    int rank;
    err = MPI_Comm_rank(comm, &rank);
    if (err) {
        MPI_Error_string(err, mpierrstr, &mpierrstrlen);
        fprintf(stderr, "%s: MPI_Comm_rank failed with %s\n",
                program_invocation_short_name, mpierrstr);
        MPI_Abort(comm, EXIT_FAILURE);
    }
    if (comm_size != 2)
        TEST_FAIL_MSG("Expected exactly two MPI processes");

    struct mtxdisterror disterr;
    err = mtxdisterror_alloc(&disterr, comm, NULL);
    if (err)
        MPI_Abort(comm, EXIT_FAILURE);

    /*
     * 32-bit integer keys.
     */

    {
        int err;
        int64_t size = (rank == 0) ? 3 : 2;
        uint32_t * keys = (rank == 0)
            ? ((uint32_t[3]) {0,255,30})
            : ((uint32_t[2]) {1,2});
        int64_t permutation[5];
        err = distradix_sort_uint32(size, keys, permutation, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s",
            err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr)
            : mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(  0, keys[0]);
            TEST_ASSERT_EQ(  1, keys[1]);
            TEST_ASSERT_EQ(  2, keys[2]);
            TEST_ASSERT_EQ(0, permutation[0]);
            TEST_ASSERT_EQ(4, permutation[1]);
            TEST_ASSERT_EQ(3, permutation[2]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ( 30, keys[0]);
            TEST_ASSERT_EQ(255, keys[1]);
            TEST_ASSERT_EQ(1, permutation[0]);
            TEST_ASSERT_EQ(2, permutation[1]);
        }
    }

    {
        int err;
        int64_t size = (rank == 0) ? 3 : 2;
        uint32_t * keys = (rank == 0)
            ? ((uint32_t[3]) {25820, 24732, 1352})
            : ((uint32_t[2]) {34041, 38784});
        int64_t permutation[5];
        err = distradix_sort_uint32(size, keys, permutation, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s",
            err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr)
            : mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ( 1352, keys[0]);
            TEST_ASSERT_EQ(24732, keys[1]);
            TEST_ASSERT_EQ(25820, keys[2]);
            TEST_ASSERT_EQ(2, permutation[0]);
            TEST_ASSERT_EQ(1, permutation[1]);
            TEST_ASSERT_EQ(0, permutation[2]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(34041, keys[0]);
            TEST_ASSERT_EQ(38784, keys[1]);
            TEST_ASSERT_EQ(3, permutation[0]);
            TEST_ASSERT_EQ(4, permutation[1]);
        }
    }

    {
        int err;
        int64_t size = 100;
        uint32_t keys[100];
        srand(415);
        for (int i = 0; i < size; i++)
            keys[i] = rand() % UINT32_MAX;
        err = distradix_sort_uint32(size, keys, NULL, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s",
            err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr)
            : mtxstrerror(err));
        for (int i = 1; i < size; i++)
            TEST_ASSERT_LE_MSG(keys[i-1], keys[i], "i=%d, keys[i-1]=%"PRIu32", keys[i]=%"PRIu32"",
                               i, keys[i-1], keys[i]);

        if (rank == 0) {
            uint32_t key;
            err = MPI_Recv(&key, 1, MPI_UINT32_T, 1, 0, comm, MPI_STATUS_IGNORE);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_LE_MSG(
                keys[size-1], key, "keys[%lld]=%"PRIu32", keys[%lld]=%"PRIu32"",
                size-1, keys[size-1], size, key);
        } else if (rank == 1) {
            err = MPI_Send(&keys[0], 1, MPI_UINT32_T, 0, 0, comm);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        }
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
        if (rank == 0)
            fprintf(stderr, "distradix_sort_uint32: ");
        fflush(stderr);
        clock_gettime(CLOCK_MONOTONIC, &t0);

        err = distradix_sort_uint32(size, keys, NULL, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s",
            err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr)
            : mtxstrerror(err));

        MPI_Barrier(comm);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        if (rank == 0) {
            fprintf(stderr, "%'.6f seconds (%'.1f MB/s)\n",
                    timespec_duration(t0, t1),
                    1.0e-6 * (comm_size * size * sizeof(*keys)) / timespec_duration(t0, t1));
        }

        free(sorting_permutation);
        free(keys);
    }

    /*
     * 64-bit integer keys.
     */

    {
        int err;
        int64_t size = (rank == 0) ? 3 : 2;
        uint64_t * keys = (rank == 0)
            ? ((uint64_t[3]) {0,255,30})
            : ((uint64_t[2]) {1,2});
        int64_t permutation[5];
        err = distradix_sort_uint64(size, keys, permutation, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s",
            err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr)
            : mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(  0, keys[0]);
            TEST_ASSERT_EQ(  1, keys[1]);
            TEST_ASSERT_EQ(  2, keys[2]);
            TEST_ASSERT_EQ(0, permutation[0]);
            TEST_ASSERT_EQ(4, permutation[1]);
            TEST_ASSERT_EQ(3, permutation[2]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ( 30, keys[0]);
            TEST_ASSERT_EQ(255, keys[1]);
            TEST_ASSERT_EQ(1, permutation[0]);
            TEST_ASSERT_EQ(2, permutation[1]);
        }
    }

    {
        int err;
        int64_t size = (rank == 0) ? 3 : 2;
        uint64_t * keys = (rank == 0)
            ? ((uint64_t[3]) {25820, 24732, 1352})
            : ((uint64_t[2]) {34041, 38784});
        int64_t permutation[5];
        err = distradix_sort_uint64(size, keys, permutation, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s",
            err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr)
            : mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ( 1352, keys[0]);
            TEST_ASSERT_EQ(24732, keys[1]);
            TEST_ASSERT_EQ(25820, keys[2]);
            TEST_ASSERT_EQ(2, permutation[0]);
            TEST_ASSERT_EQ(1, permutation[1]);
            TEST_ASSERT_EQ(0, permutation[2]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(34041, keys[0]);
            TEST_ASSERT_EQ(38784, keys[1]);
            TEST_ASSERT_EQ(3, permutation[0]);
            TEST_ASSERT_EQ(4, permutation[1]);
        }
    }

    {
        int err;
        int64_t size = 100;
        uint64_t keys[100];
        for (int i = 0; i < size; i++)
            keys[i] = rand_uint64();
        err = distradix_sort_uint64(size, keys, NULL, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s",
            err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr)
            : mtxstrerror(err));
        for (int i = 1; i < size; i++)
            TEST_ASSERT_LE_MSG(keys[i-1], keys[i], "i=%d, keys[i-1]=%"PRIu64", keys[i]=%"PRIu64"",
                               i, keys[i-1], keys[i]);

        if (rank == 0) {
            uint64_t key;
            err = MPI_Recv(&key, 1, MPI_UINT64_T, 1, 0, comm, MPI_STATUS_IGNORE);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_LE_MSG(
                keys[size-1], key, "keys[%lld]=%"PRIu64", keys[%lld]=%"PRIu64"",
                size-1, keys[size-1], size, key);
        } else if (rank == 1) {
            err = MPI_Send(&keys[0], 1, MPI_UINT64_T, 0, 0, comm);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        }
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
        if (rank == 0)
            fprintf(stderr, "distradix_sort_uint64: ");
        fflush(stderr);
        clock_gettime(CLOCK_MONOTONIC, &t0);

        err = distradix_sort_uint64(size, keys, NULL, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s",
            err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr)
            : mtxstrerror(err));

        MPI_Barrier(comm);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        if (rank == 0) {
            fprintf(stderr, "%'.6f seconds (%'.1f MB/s)\n",
                    timespec_duration(t0, t1),
                    1.0e-6 * (size * sizeof(*keys)) / timespec_duration(t0, t1));
        }

        free(sorting_permutation);
        free(keys);
    }

    mtxdisterror_free(&disterr);
    return TEST_SUCCESS;
}

/**
 * ‘main()’ entry point and test driver.
 */
int main(int argc, char * argv[])
{
    int mpierr;
    char mpierrstr[MPI_MAX_ERROR_STRING];
    int mpierrstrlen;

    /* 1. Initialise MPI. */
    const MPI_Comm mpi_comm = MPI_COMM_WORLD;
    const int mpi_root = 0;
    mpierr = MPI_Init(&argc, &argv);
    if (mpierr) {
        MPI_Error_string(mpierr, mpierrstr, &mpierrstrlen);
        fprintf(stderr, "%s: MPI_Init failed with %s\n",
                program_invocation_short_name, mpierrstr);
        return EXIT_FAILURE;
    }

    /* 2. Run test suite. */
    TEST_SUITE_BEGIN("Running tests for distributed sorting functions\n");
    TEST_RUN(test_distradix_sort);
    TEST_SUITE_END();

    /* 3. Clean up and return. */
    MPI_Finalize();
    return (TEST_SUITE_STATUS == TEST_SUCCESS) ?
        EXIT_SUCCESS : EXIT_FAILURE;
}
