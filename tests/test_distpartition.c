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
 * Last modified: 2022-04-30
 *
 * Unit tests for distributed partitioning functions.
 */

#include <libmtx/libmtx-config.h>

#include <libmtx/error.h>

#include "libmtx/util/partition.h"
#include "test.h"

#include <errno.h>

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

const char * program_invocation_short_name = "test_distpartition";

/**
 * ‘test_distpartition_block()’ tests block partitioning.
 */
int test_distpartition_block(void)
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
    if (comm_size != 2) TEST_FAIL_MSG("Expected exactly two MPI processes");
    struct mtxdisterror disterr;
    err = mtxdisterror_alloc(&disterr, comm, NULL);
    if (err) MPI_Abort(comm, EXIT_FAILURE);

    {
        int size = 5;
        int64_t partsize = rank == 0 ? 5 : 0;
        int64_t idx[] = {0,2,1,4,3};
        int idxsize = sizeof(idx) / sizeof(*idx);
        int dstpart[5] = {};
        err = distpartition_block_int64(
            size, partsize, idxsize, sizeof(*idx), idx, dstpart, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(0, dstpart[0]);
        TEST_ASSERT_EQ(0, dstpart[1]);
        TEST_ASSERT_EQ(0, dstpart[2]);
        TEST_ASSERT_EQ(0, dstpart[3]);
        TEST_ASSERT_EQ(0, dstpart[4]);
    }
    {
        int size = 5;
        int64_t partsize = rank == 0 ? 4 : 1;
        int64_t idx[] = {0,2,1,4,3};
        int idxsize = sizeof(idx) / sizeof(*idx);
        int dstpart[5] = {};
        err = distpartition_block_int64(
            size, partsize, idxsize, sizeof(*idx), idx, dstpart, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(0, dstpart[0]);
        TEST_ASSERT_EQ(0, dstpart[1]);
        TEST_ASSERT_EQ(0, dstpart[2]);
        TEST_ASSERT_EQ(1, dstpart[3]);
        TEST_ASSERT_EQ(0, dstpart[4]);
    }
    {
        int size = 5;
        int64_t partsize = rank == 0 ? 2 : 3;
        int64_t idx[] = {0,2,1,4,3};
        int idxsize = sizeof(idx) / sizeof(*idx);
        int dstpart[5] = {};
        err = distpartition_block_int64(
            size, partsize, idxsize, sizeof(*idx), idx, dstpart, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(0, dstpart[0]);
        TEST_ASSERT_EQ(1, dstpart[1]);
        TEST_ASSERT_EQ(0, dstpart[2]);
        TEST_ASSERT_EQ(1, dstpart[3]);
        TEST_ASSERT_EQ(1, dstpart[4]);
    }
    mtxdisterror_free(&disterr);
    return TEST_SUCCESS;
}

/**
 * ‘test_distpartition_block_cyclic()’ tests block-cyclic
 * partitioning.
 */
int test_distpartition_block_cyclic(void)
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
    if (comm_size != 2) TEST_FAIL_MSG("Expected exactly two MPI processes");
    struct mtxdisterror disterr;
    err = mtxdisterror_alloc(&disterr, comm, NULL);
    if (err) MPI_Abort(comm, EXIT_FAILURE);

    {
        int size = 5;
        int blksize = 5;
        int64_t idx[] = {0,2,1,4,3};
        int idxsize = sizeof(idx) / sizeof(*idx);
        int dstpart[5] = {};
        err = distpartition_block_cyclic_int64(
            size, blksize, idxsize, sizeof(*idx), idx, dstpart, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(0, dstpart[0]);
        TEST_ASSERT_EQ(0, dstpart[1]);
        TEST_ASSERT_EQ(0, dstpart[2]);
        TEST_ASSERT_EQ(0, dstpart[3]);
        TEST_ASSERT_EQ(0, dstpart[4]);
    }
    {
        int size = 5;
        int blksize = 1;
        int64_t idx[] = {0,2,1,4,3};
        int idxsize = sizeof(idx) / sizeof(*idx);
        int dstpart[5] = {};
        err = distpartition_block_cyclic_int64(
            size, blksize, idxsize, sizeof(*idx), idx, dstpart, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(0, dstpart[0]);
        TEST_ASSERT_EQ(0, dstpart[1]);
        TEST_ASSERT_EQ(1, dstpart[2]);
        TEST_ASSERT_EQ(0, dstpart[3]);
        TEST_ASSERT_EQ(1, dstpart[4]);
    }
    {
        int size = 5;
        int blksize = 3;
        int64_t idx[] = {0,2,1,4,3};
        int idxsize = sizeof(idx) / sizeof(*idx);
        int dstpart[5] = {};
        err = distpartition_block_cyclic_int64(
            size, blksize, idxsize, sizeof(*idx), idx, dstpart, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(0, dstpart[0]);
        TEST_ASSERT_EQ(0, dstpart[1]);
        TEST_ASSERT_EQ(0, dstpart[2]);
        TEST_ASSERT_EQ(1, dstpart[3]);
        TEST_ASSERT_EQ(1, dstpart[4]);
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
    const MPI_Comm mpi_comm = MPI_COMM_WORLD;
    const int mpi_root = 0;
    mpierr = MPI_Init(&argc, &argv);
    if (mpierr) {
        MPI_Error_string(mpierr, mpierrstr, &mpierrstrlen);
        fprintf(stderr, "%s: MPI_Init failed with %s\n",
                program_invocation_short_name, mpierrstr);
        return EXIT_FAILURE;
    }

    TEST_SUITE_BEGIN("Running tests for distributed partitioning of sets\n");
    TEST_RUN(test_distpartition_block);
    TEST_RUN(test_distpartition_block_cyclic);
    TEST_SUITE_END();
    MPI_Finalize();
    return (TEST_SUITE_STATUS == TEST_SUCCESS) ?
        EXIT_SUCCESS : EXIT_FAILURE;
}
