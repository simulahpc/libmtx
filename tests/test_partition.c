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
 * Last modified: 2022-05-28
 *
 * Unit tests for partitioning functions.
 */

#include <libmtx/error.h>

#include "libmtx/util/partition.h"
#include "test.h"

#include <errno.h>

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * ‘test_partition_block()’ tests block partitioning.
 */
int test_partition_block(void)
{
    int err;
    {
        int size = 5;
        int num_parts = 1;
        int64_t partsizes[] = {5};
        int64_t idx[] = {0,2,1,4,3};
        int idxsize = sizeof(idx) / sizeof(*idx);
        int dstpart[5] = {};
        err = partition_block_int64(
            size, num_parts, partsizes, idxsize, sizeof(*idx), idx, dstpart);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(0, dstpart[0]);
        TEST_ASSERT_EQ(0, dstpart[1]);
        TEST_ASSERT_EQ(0, dstpart[2]);
        TEST_ASSERT_EQ(0, dstpart[3]);
        TEST_ASSERT_EQ(0, dstpart[4]);
    }
    {
        int size = 5;
        int num_parts = 2;
        int64_t partsizes[] = {5,0};
        int64_t idx[] = {0,2,1,4,3};
        int idxsize = sizeof(idx) / sizeof(*idx);
        int dstpart[5] = {};
        err = partition_block_int64(
            size, num_parts, partsizes, idxsize, sizeof(*idx), idx, dstpart);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(0, dstpart[0]);
        TEST_ASSERT_EQ(0, dstpart[1]);
        TEST_ASSERT_EQ(0, dstpart[2]);
        TEST_ASSERT_EQ(0, dstpart[3]);
        TEST_ASSERT_EQ(0, dstpart[4]);
    }
    {
        int size = 5;
        int num_parts = 2;
        int64_t partsizes[] = {4,1};
        int64_t idx[] = {0,2,1,4,3};
        int idxsize = sizeof(idx) / sizeof(*idx);
        int dstpart[5] = {};
        err = partition_block_int64(
            size, num_parts, partsizes, idxsize, sizeof(*idx), idx, dstpart);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(0, dstpart[0]);
        TEST_ASSERT_EQ(0, dstpart[1]);
        TEST_ASSERT_EQ(0, dstpart[2]);
        TEST_ASSERT_EQ(1, dstpart[3]);
        TEST_ASSERT_EQ(0, dstpart[4]);
    }
    {
        int size = 5;
        int num_parts = 2;
        int64_t partsizes[] = {2,3};
        int64_t idx[] = {0,2,1,4,3};
        int idxsize = sizeof(idx) / sizeof(*idx);
        int dstpart[5] = {};
        err = partition_block_int64(
            size, num_parts, partsizes, idxsize, sizeof(*idx), idx, dstpart);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(0, dstpart[0]);
        TEST_ASSERT_EQ(1, dstpart[1]);
        TEST_ASSERT_EQ(0, dstpart[2]);
        TEST_ASSERT_EQ(1, dstpart[3]);
        TEST_ASSERT_EQ(1, dstpart[4]);
    }
    {
        int size = 5;
        int num_parts = 3;
        int64_t partsizes[] = {1,3,1};
        int64_t idx[] = {0,2,1,4,3};
        int idxsize = sizeof(idx) / sizeof(*idx);
        int dstpart[5] = {};
        err = partition_block_int64(
            size, num_parts, partsizes, idxsize, sizeof(*idx), idx, dstpart);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(0, dstpart[0]);
        TEST_ASSERT_EQ(1, dstpart[1]);
        TEST_ASSERT_EQ(1, dstpart[2]);
        TEST_ASSERT_EQ(2, dstpart[3]);
        TEST_ASSERT_EQ(1, dstpart[4]);
    }
    return TEST_SUCCESS;
}

/**
 * ‘test_partition_block_cyclic()’ tests block-cyclic partitioning.
 */
int test_partition_block_cyclic(void)
{
    int err;
    {
        int size = 5;
        int num_parts = 1;
        int blksize = 5;
        int64_t idx[] = {0,2,1,4,3};
        int idxsize = sizeof(idx) / sizeof(*idx);
        int dstpart[5] = {};
        err = partition_block_cyclic_int64(
            size, num_parts, blksize, idxsize, sizeof(*idx), idx, dstpart);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(0, dstpart[0]);
        TEST_ASSERT_EQ(0, dstpart[1]);
        TEST_ASSERT_EQ(0, dstpart[2]);
        TEST_ASSERT_EQ(0, dstpart[3]);
        TEST_ASSERT_EQ(0, dstpart[4]);
    }
    {
        int size = 5;
        int num_parts = 2;
        int blksize = 1;
        int64_t idx[] = {0,2,1,4,3};
        int idxsize = sizeof(idx) / sizeof(*idx);
        int dstpart[5] = {};
        err = partition_block_cyclic_int64(
            size, num_parts, blksize, idxsize, sizeof(*idx), idx, dstpart);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(0, dstpart[0]);
        TEST_ASSERT_EQ(0, dstpart[1]);
        TEST_ASSERT_EQ(1, dstpart[2]);
        TEST_ASSERT_EQ(0, dstpart[3]);
        TEST_ASSERT_EQ(1, dstpart[4]);
    }
    {
        int size = 5;
        int num_parts = 2;
        int blksize = 2;
        int64_t idx[] = {0,2,1,4,3};
        int idxsize = sizeof(idx) / sizeof(*idx);
        int dstpart[5] = {};
        err = partition_block_cyclic_int64(
            size, num_parts, blksize, idxsize, sizeof(*idx), idx, dstpart);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(0, dstpart[0]);
        TEST_ASSERT_EQ(1, dstpart[1]);
        TEST_ASSERT_EQ(0, dstpart[2]);
        TEST_ASSERT_EQ(0, dstpart[3]);
        TEST_ASSERT_EQ(1, dstpart[4]);
    }
    {
        int size = 5;
        int num_parts = 3;
        int blksize = 1;
        int64_t idx[] = {0,2,1,4,3};
        int idxsize = sizeof(idx) / sizeof(*idx);
        int dstpart[5] = {};
        err = partition_block_cyclic_int64(
            size, num_parts, blksize, idxsize, sizeof(*idx), idx, dstpart);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(0, dstpart[0]);
        TEST_ASSERT_EQ(2, dstpart[1]);
        TEST_ASSERT_EQ(1, dstpart[2]);
        TEST_ASSERT_EQ(1, dstpart[3]);
        TEST_ASSERT_EQ(0, dstpart[4]);
    }
    {
        int size = 5;
        int num_parts = 3;
        int blksize = 2;
        int64_t idx[] = {0,2,1,4,3};
        int idxsize = sizeof(idx) / sizeof(*idx);
        int dstpart[5] = {};
        err = partition_block_cyclic_int64(
            size, num_parts, blksize, idxsize, sizeof(*idx), idx, dstpart);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(0, dstpart[0]);
        TEST_ASSERT_EQ(1, dstpart[1]);
        TEST_ASSERT_EQ(0, dstpart[2]);
        TEST_ASSERT_EQ(2, dstpart[3]);
        TEST_ASSERT_EQ(1, dstpart[4]);
    }
    return TEST_SUCCESS;
}

/**
 * ‘main()’ entry point and test driver.
 */
int main(int argc, char * argv[])
{
    TEST_SUITE_BEGIN("Running tests for partitioning\n");
    TEST_RUN(test_partition_block);
    TEST_RUN(test_partition_block_cyclic);
    TEST_SUITE_END();
    return (TEST_SUITE_STATUS == TEST_SUCCESS) ?
        EXIT_SUCCESS : EXIT_FAILURE;
}
