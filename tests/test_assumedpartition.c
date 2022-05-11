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
 * Last modified: 2022-05-11
 *
 * Unit tests for the assumed partition strategy.
 */

#include "test.h"

#include <libmtx/libmtx-config.h>

#include <libmtx/error.h>
#include <libmtx/util/partition.h>

#include <mpi.h>

#include <errno.h>

#include <stdlib.h>

const char * program_invocation_short_name = "test_assumedpartition";

/**
 * ‘test_assumedpartition_write()’ tests creating an assumed partition.
 */
int test_assumedpartition_write(void)
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
    if (err) MPI_Abort(comm, EXIT_FAILURE);

    {
        int size = 12;
        int partsize = rank == 0 ? 2 : 3;
        int64_t * globalidx = rank == 0 ? (int64_t[2]){0,3} : (int64_t[3]){5,6,9};
        const int assumedpartsize = 6; /* (size+comm_size-1) / comm_size */
        int apownerrank[assumedpartsize];
        int apowneridx[assumedpartsize];
        err = assumedpartition_write(
            size, partsize, globalidx, assumedpartsize,
            apownerrank, apowneridx, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));

        if (rank == 0) {
            TEST_ASSERT_EQ(0, apownerrank[0]); TEST_ASSERT_EQ(0, apowneridx[0]);
            /* TEST_ASSERT_EQ(?, apownerrank[1]); TEST_ASSERT_EQ(?, apowneridx[1]); */
            /* TEST_ASSERT_EQ(?, apownerrank[2]); TEST_ASSERT_EQ(?, apowneridx[2]); */
            TEST_ASSERT_EQ(0, apownerrank[3]); TEST_ASSERT_EQ(1, apowneridx[3]);
            /* TEST_ASSERT_EQ(?, apownerrank[4]); TEST_ASSERT_EQ(?, apowneridx[4]); */
            TEST_ASSERT_EQ(1, apownerrank[5]); TEST_ASSERT_EQ(0, apowneridx[5]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(1, apownerrank[0]); TEST_ASSERT_EQ(1, apowneridx[0]);
            /* TEST_ASSERT_EQ(?, apownerrank[1]); TEST_ASSERT_EQ(?, apowneridx[1]); */
            /* TEST_ASSERT_EQ(?, apownerrank[2]); TEST_ASSERT_EQ(?, apowneridx[2]); */
            TEST_ASSERT_EQ(1, apownerrank[3]); TEST_ASSERT_EQ(2, apowneridx[3]);
            /* TEST_ASSERT_EQ(?, apownerrank[4]); TEST_ASSERT_EQ(?, apowneridx[4]); */
            /* TEST_ASSERT_EQ(?, apownerrank[5]); TEST_ASSERT_EQ(?, apowneridx[5]); */
        }
    }

    {
        int size = 12;
        int partsize = rank == 0 ? 2 : 3;
        int64_t * globalidx = rank == 0 ? (int64_t[2]){3,0} : (int64_t[3]){5,9,6};
        const int assumedpartsize = 6; /* (size+comm_size-1) / comm_size */
        int apownerrank[assumedpartsize];
        int apowneridx[assumedpartsize];
        err = assumedpartition_write(
            size, partsize, globalidx, assumedpartsize,
            apownerrank, apowneridx, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));

        if (rank == 0) {
            TEST_ASSERT_EQ(0, apownerrank[0]); TEST_ASSERT_EQ(1, apowneridx[0]);
            /* TEST_ASSERT_EQ(?, apownerrank[1]); TEST_ASSERT_EQ(?, apowneridx[1]); */
            /* TEST_ASSERT_EQ(?, apownerrank[2]); TEST_ASSERT_EQ(?, apowneridx[2]); */
            TEST_ASSERT_EQ(0, apownerrank[3]); TEST_ASSERT_EQ(0, apowneridx[3]);
            /* TEST_ASSERT_EQ(?, apownerrank[4]); TEST_ASSERT_EQ(?, apowneridx[4]); */
            TEST_ASSERT_EQ(1, apownerrank[5]); TEST_ASSERT_EQ(0, apowneridx[5]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(1, apownerrank[0]); TEST_ASSERT_EQ(2, apowneridx[0]);
            /* TEST_ASSERT_EQ(?, apownerrank[1]); TEST_ASSERT_EQ(?, apowneridx[1]); */
            /* TEST_ASSERT_EQ(?, apownerrank[2]); TEST_ASSERT_EQ(?, apowneridx[2]); */
            TEST_ASSERT_EQ(1, apownerrank[3]); TEST_ASSERT_EQ(1, apowneridx[3]);
            /* TEST_ASSERT_EQ(?, apownerrank[4]); TEST_ASSERT_EQ(?, apowneridx[4]); */
            /* TEST_ASSERT_EQ(?, apownerrank[5]); TEST_ASSERT_EQ(?, apowneridx[5]); */
        }
    }

    mtxdisterror_free(&disterr);
    return TEST_SUCCESS;
}

/**
 * ‘test_assumedpartition_read()’ tests querying an assumed partition.
 */
int test_assumedpartition_read(void)
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
    if (err) MPI_Abort(comm, EXIT_FAILURE);

    {
        /* cyclic partitioning */
        int size = 12;
        const int apsize = 6; /* (size+comm_size-1) / comm_size */
        const int * apownerrank = rank == 0
            ? ((int[6]) {0, 1, 0, 1, 0, 1})
            : ((int[6]) {0, 1, 0, 1, 0, 1});
        const int * apowneridx = rank == 0
            ? ((int[6]) {0, 0, 1, 1, 2, 2})
            : ((int[6]) {3, 3, 4, 4, 5, 5});
        const int partsize = rank == 0 ? 2 : 3;
        int64_t * globalidx = rank == 0 ? (int64_t[2]){0,3} : (int64_t[3]){5,6,9};
        int * ownerrank = rank == 0 ? (int[2]){0,0} : (int[3]){0,0,0};
        int * owneridx = rank == 0 ? (int[2]){0,0} : (int[3]){0,0,0};
        err = assumedpartition_read(
            size, apsize, apownerrank, apowneridx,
            partsize, globalidx, ownerrank, owneridx,
            comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));

        if (rank == 0) {
            TEST_ASSERT_EQ(0, ownerrank[0]); TEST_ASSERT_EQ(0, owneridx[0]);
            TEST_ASSERT_EQ(1, ownerrank[1]); TEST_ASSERT_EQ(1, owneridx[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(1, ownerrank[0]); TEST_ASSERT_EQ(2, owneridx[0]);
            TEST_ASSERT_EQ(0, ownerrank[1]); TEST_ASSERT_EQ(3, owneridx[1]);
            TEST_ASSERT_EQ(1, ownerrank[2]); TEST_ASSERT_EQ(4, owneridx[2]);
        }
    }

    {
        /* cyclic partitioning */
        int size = 12;
        const int apsize = 6; /* (size+comm_size-1) / comm_size */
        const int * apownerrank = rank == 0
            ? ((int[6]) {0, 1, 0, 1, 0, 1})
            : ((int[6]) {0, 1, 0, 1, 0, 1});
        const int * apowneridx = rank == 0
            ? ((int[6]) {0, 0, 1, 1, 2, 2})
            : ((int[6]) {3, 3, 4, 4, 5, 5});
        const int partsize = rank == 0 ? 2 : 3;
        int64_t * globalidx = rank == 0 ? (int64_t[2]){3,0} : (int64_t[3]){5,9,6};
        int * ownerrank = rank == 0 ? (int[2]){0,0} : (int[3]){0,0,0};
        int * owneridx = rank == 0 ? (int[2]){0,0} : (int[3]){0,0,0};
        err = assumedpartition_read(
            size, apsize, apownerrank, apowneridx,
            partsize, globalidx, ownerrank, owneridx,
            comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));

        if (rank == 0) {
            TEST_ASSERT_EQ(1, ownerrank[0]); TEST_ASSERT_EQ(1, owneridx[0]);
            TEST_ASSERT_EQ(0, ownerrank[1]); TEST_ASSERT_EQ(0, owneridx[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(1, ownerrank[0]); TEST_ASSERT_EQ(2, owneridx[0]);
            TEST_ASSERT_EQ(1, ownerrank[1]); TEST_ASSERT_EQ(4, owneridx[1]);
            TEST_ASSERT_EQ(0, ownerrank[2]); TEST_ASSERT_EQ(3, owneridx[2]);
        }
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
    const MPI_Comm comm = MPI_COMM_WORLD;
    const int root = 0;
    mpierr = MPI_Init(&argc, &argv);
    if (mpierr) {
        MPI_Error_string(mpierr, mpierrstr, &mpierrstrlen);
        fprintf(stderr, "%s: MPI_Init failed with %s\n",
                program_invocation_short_name, mpierrstr);
        return EXIT_FAILURE;
    }
    mpierr = MPI_Comm_set_errhandler(comm, MPI_ERRORS_RETURN);
    if (mpierr) {
        MPI_Error_string(mpierr, mpierrstr, &mpierrstrlen);
        fprintf(stderr, "%s: MPI_Init failed with %s\n",
                program_invocation_short_name, mpierrstr);
        return EXIT_FAILURE;
    }

    /* 2. Run test suite. */
    TEST_SUITE_BEGIN("Running tests for the assumed partition strategy\n");
    TEST_RUN(test_assumedpartition_write);
    TEST_RUN(test_assumedpartition_read);
    TEST_SUITE_END();

    /* 3. Clean up and return. */
    MPI_Finalize();
    return (TEST_SUITE_STATUS == TEST_SUCCESS) ?
        EXIT_SUCCESS : EXIT_FAILURE;
}
