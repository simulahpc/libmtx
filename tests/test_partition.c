/* This file is part of libmtx.
 *
 * Copyright (C) 2022 James D. Trotter
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
 * Last modified: 2022-01-10
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
 * ‘test_mtxpartition_singleton()’ tests singleton partitioning.
 */
int test_mtxpartition_singleton(void)
{
    int err;

    {
        int size = 5;
        int num_parts = 1;
        struct mtxpartition partition;
        err = mtxpartition_init(
            &partition, mtx_singleton, size, num_parts, NULL, 0, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        {
            int64_t globalelem[5] = {0,1,2,3,4};
            int parts[5] = {};
            err = mtxpartition_assign(&partition, size, globalelem, parts);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(0, parts[0]);
            TEST_ASSERT_EQ(0, parts[1]);
            TEST_ASSERT_EQ(0, parts[2]);
            TEST_ASSERT_EQ(0, parts[3]);
            TEST_ASSERT_EQ(0, parts[4]);
        }
        {
            int64_t localelem[5] = {0,1,2,3,4};
            int64_t globalelem[5] = {};
            err = mtxpartition_globalidx(
                &partition, 0, size, localelem, globalelem);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(0, globalelem[0]);
            TEST_ASSERT_EQ(1, globalelem[1]);
            TEST_ASSERT_EQ(2, globalelem[2]);
            TEST_ASSERT_EQ(3, globalelem[3]);
            TEST_ASSERT_EQ(4, globalelem[4]);
        }
        {
            int64_t globalelem[5] = {0,1,2,3,4};
            int64_t localelem[5] = {};
            err = mtxpartition_localidx(
                &partition, 0, size, globalelem, localelem);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(0, localelem[0]);
            TEST_ASSERT_EQ(1, localelem[1]);
            TEST_ASSERT_EQ(2, localelem[2]);
            TEST_ASSERT_EQ(3, localelem[3]);
            TEST_ASSERT_EQ(4, localelem[4]);
        }
        mtxpartition_free(&partition);
    }
    {
        int size = 5;
        int num_parts = 2;
        struct mtxpartition partition;
        err = mtxpartition_init(
            &partition, mtx_singleton, size, num_parts, NULL, 0, NULL);
        TEST_ASSERT_EQ_MSG(
            MTX_ERR_INVALID_PARTITION_TYPE, err, "%s", mtxstrerror(err));
    }

    return TEST_SUCCESS;
}

/**
 * ‘test_mtxpartition_block()’ tests block partitioning.
 */
int test_mtxpartition_block(void)
{
    int err;

    {
        int size = 5;
        int num_parts = 1;
        struct mtxpartition partition;
        err = mtxpartition_init(
            &partition, mtx_block, size, num_parts, NULL, 0, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        {
            int64_t globalelem[5] = {0,1,2,3,4};
            int parts[5] = {};
            err = mtxpartition_assign(&partition, size, globalelem, parts);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(0, parts[0]);
            TEST_ASSERT_EQ(0, parts[1]);
            TEST_ASSERT_EQ(0, parts[2]);
            TEST_ASSERT_EQ(0, parts[3]);
            TEST_ASSERT_EQ(0, parts[4]);
        }
        {
            int64_t localelem[5] = {0,1,2,3,4};
            int64_t globalelem[5] = {};
            err = mtxpartition_globalidx(
                &partition, 0, size, localelem, globalelem);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(0, globalelem[0]);
            TEST_ASSERT_EQ(1, globalelem[1]);
            TEST_ASSERT_EQ(2, globalelem[2]);
            TEST_ASSERT_EQ(3, globalelem[3]);
            TEST_ASSERT_EQ(4, globalelem[4]);
        }
        {
            int64_t globalelem[5] = {0,1,2,3,4};
            int64_t localelem[5] = {};
            err = mtxpartition_localidx(
                &partition, 0, size, globalelem, localelem);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(0, localelem[0]);
            TEST_ASSERT_EQ(1, localelem[1]);
            TEST_ASSERT_EQ(2, localelem[2]);
            TEST_ASSERT_EQ(3, localelem[3]);
            TEST_ASSERT_EQ(4, localelem[4]);
        }
        mtxpartition_free(&partition);
    }
    {
        int size = 5;
        int num_parts = 2;
        struct mtxpartition partition;
        err = mtxpartition_init(
            &partition, mtx_block, size, num_parts, NULL, 0, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        {
            int64_t globalelem[5] = {0,1,2,3,4};
            int parts[5] = {};
            err = mtxpartition_assign(&partition, size, globalelem, parts);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(0, parts[0]);
            TEST_ASSERT_EQ(0, parts[1]);
            TEST_ASSERT_EQ(0, parts[2]);
            TEST_ASSERT_EQ(1, parts[3]);
            TEST_ASSERT_EQ(1, parts[4]);
        }
        {
            int part = 0;
            int64_t localelem[3] = {0,1,2};
            int64_t globalelem[3] = {};
            err = mtxpartition_globalidx(
                &partition, part, partition.part_sizes[part],
                localelem, globalelem);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(0, globalelem[0]);
            TEST_ASSERT_EQ(1, globalelem[1]);
            TEST_ASSERT_EQ(2, globalelem[2]);
        }
        {
            int part = 1;
            int64_t localelem[2] = {0,1};
            int64_t globalelem[2] = {};
            err = mtxpartition_globalidx(
                &partition, part, partition.part_sizes[part],
                localelem, globalelem);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(3, globalelem[0]);
            TEST_ASSERT_EQ(4, globalelem[1]);
        }
        {
            int part = 0;
            int64_t globalelem[3] = {0,1,2};
            int64_t localelem[3] = {};
            err = mtxpartition_localidx(
                &partition, part, partition.part_sizes[part],
                globalelem, localelem);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(0, localelem[0]);
            TEST_ASSERT_EQ(1, localelem[1]);
            TEST_ASSERT_EQ(2, localelem[2]);
        }
        {
            int part = 1;
            int64_t globalelem[2] = {3,4};
            int64_t localelem[2] = {};
            err = mtxpartition_localidx(
                &partition, part, partition.part_sizes[part],
                globalelem, localelem);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(0, localelem[0]);
            TEST_ASSERT_EQ(1, localelem[1]);
        }
        mtxpartition_free(&partition);
    }
    {
        int size = 5;
        int num_parts = 3;
        struct mtxpartition partition;
        err = mtxpartition_init(
            &partition, mtx_block, size, num_parts, NULL, 0, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        {
            int64_t globalelem[5] = {0,1,2,3,4};
            int parts[5] = {};
            err = mtxpartition_assign(&partition, size, globalelem, parts);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(0, parts[0]);
            TEST_ASSERT_EQ(0, parts[1]);
            TEST_ASSERT_EQ(1, parts[2]);
            TEST_ASSERT_EQ(1, parts[3]);
            TEST_ASSERT_EQ(2, parts[4]);
        }
        {
            int part = 0;
            int64_t localelem[2] = {0,1};
            int64_t globalelem[2] = {};
            err = mtxpartition_globalidx(
                &partition, part, partition.part_sizes[part],
                localelem, globalelem);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(0, globalelem[0]);
            TEST_ASSERT_EQ(1, globalelem[1]);
        }
        {
            int part = 1;
            int64_t localelem[2] = {0,1};
            int64_t globalelem[2] = {};
            err = mtxpartition_globalidx(
                &partition, part, partition.part_sizes[part],
                localelem, globalelem);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(2, globalelem[0]);
            TEST_ASSERT_EQ(3, globalelem[1]);
        }
        {
            int part = 2;
            int64_t localelem[1] = {0};
            int64_t globalelem[1] = {};
            err = mtxpartition_globalidx(
                &partition, part, partition.part_sizes[part],
                localelem, globalelem);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(4, globalelem[0]);
        }
        {
            int part = 0;
            int64_t globalelem[2] = {0,1};
            int64_t localelem[2] = {};
            err = mtxpartition_localidx(
                &partition, part, partition.part_sizes[part],
                globalelem, localelem);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(0, localelem[0]);
            TEST_ASSERT_EQ(1, localelem[1]);
        }
        {
            int part = 1;
            int64_t globalelem[2] = {2,3};
            int64_t localelem[2] = {};
            err = mtxpartition_localidx(
                &partition, part, partition.part_sizes[part],
                globalelem, localelem);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(0, localelem[0]);
            TEST_ASSERT_EQ(1, localelem[1]);
        }
        {
            int part = 2;
            int64_t globalelem[1] = {4};
            int64_t localelem[1] = {};
            err = mtxpartition_localidx(
                &partition, part, partition.part_sizes[part],
                globalelem, localelem);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(0, localelem[0]);
        }
        mtxpartition_free(&partition);
    }
    {
        int size = 5;
        int num_parts = 4;
        struct mtxpartition partition;
        err = mtxpartition_init(
            &partition, mtx_block, size, num_parts, NULL, 0, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        {
            int64_t globalelem[5] = {0,1,2,3,4};
            int parts[5] = {};
            err = mtxpartition_assign(&partition, size, globalelem, parts);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(0, parts[0]);
            TEST_ASSERT_EQ(0, parts[1]);
            TEST_ASSERT_EQ(1, parts[2]);
            TEST_ASSERT_EQ(2, parts[3]);
            TEST_ASSERT_EQ(3, parts[4]);
        }
        {
            int part = 0;
            int64_t localelem[2] = {0,1};
            int64_t globalelem[2] = {};
            err = mtxpartition_globalidx(
                &partition, part, partition.part_sizes[part],
                localelem, globalelem);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(0, globalelem[0]);
            TEST_ASSERT_EQ(1, globalelem[1]);
        }
        {
            int part = 1;
            int64_t localelem[1] = {0};
            int64_t globalelem[1] = {};
            err = mtxpartition_globalidx(
                &partition, part, partition.part_sizes[part],
                localelem, globalelem);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(2, globalelem[0]);
        }
        {
            int part = 2;
            int64_t localelem[1] = {0};
            int64_t globalelem[1] = {};
            err = mtxpartition_globalidx(
                &partition, part, partition.part_sizes[part],
                localelem, globalelem);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(3, globalelem[0]);
        }
        {
            int part = 3;
            int64_t localelem[1] = {0};
            int64_t globalelem[1] = {};
            err = mtxpartition_globalidx(
                &partition, part, partition.part_sizes[part],
                localelem, globalelem);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(4, globalelem[0]);
        }
        {
            int part = 0;
            int64_t globalelem[2] = {0,1};
            int64_t localelem[2] = {};
            err = mtxpartition_localidx(
                &partition, part, partition.part_sizes[part],
                globalelem, localelem);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(0, localelem[0]);
            TEST_ASSERT_EQ(1, localelem[1]);
        }
        {
            int part = 1;
            int64_t globalelem[1] = {2};
            int64_t localelem[1] = {};
            err = mtxpartition_localidx(
                &partition, part, partition.part_sizes[part],
                globalelem, localelem);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(0, localelem[0]);
        }
        {
            int part = 2;
            int64_t globalelem[1] = {3};
            int64_t localelem[1] = {};
            err = mtxpartition_localidx(
                &partition, part, partition.part_sizes[part],
                globalelem, localelem);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(0, localelem[0]);
        }
        {
            int part = 3;
            int64_t globalelem[1] = {4};
            int64_t localelem[1] = {};
            err = mtxpartition_localidx(
                &partition, part, partition.part_sizes[part],
                globalelem, localelem);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(0, localelem[0]);
        }
        mtxpartition_free(&partition);
    }
    {
        int size = 5;
        int num_parts = 5;
        struct mtxpartition partition;
        err = mtxpartition_init(
            &partition, mtx_block, size, num_parts, NULL, 0, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        int64_t globalelem[5] = {0,1,2,3,4};
        int parts[5] = {};
        err = mtxpartition_assign(&partition, size, globalelem, parts);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(0, parts[0]);
        TEST_ASSERT_EQ(1, parts[1]);
        TEST_ASSERT_EQ(2, parts[2]);
        TEST_ASSERT_EQ(3, parts[3]);
        TEST_ASSERT_EQ(4, parts[4]);
        {
            int part = 0;
            int64_t localelem[1] = {0};
            int64_t globalelem[1] = {};
            err = mtxpartition_globalidx(
                &partition, part, partition.part_sizes[part],
                localelem, globalelem);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(0, globalelem[0]);
        }
        {
            int part = 1;
            int64_t localelem[1] = {0};
            int64_t globalelem[1] = {};
            err = mtxpartition_globalidx(
                &partition, part, partition.part_sizes[part],
                localelem, globalelem);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(1, globalelem[0]);
        }
        {
            int part = 2;
            int64_t localelem[1] = {0};
            int64_t globalelem[1] = {};
            err = mtxpartition_globalidx(
                &partition, part, partition.part_sizes[part],
                localelem, globalelem);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(2, globalelem[0]);
        }
        {
            int part = 3;
            int64_t localelem[1] = {0};
            int64_t globalelem[1] = {};
            err = mtxpartition_globalidx(
                &partition, part, partition.part_sizes[part],
                localelem, globalelem);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(3, globalelem[0]);
        }
        {
            int part = 4;
            int64_t localelem[1] = {0};
            int64_t globalelem[1] = {};
            err = mtxpartition_globalidx(
                &partition, part, partition.part_sizes[part],
                localelem, globalelem);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(4, globalelem[0]);
        }
        {
            int part = 0;
            int64_t globalelem[1] = {0};
            int64_t localelem[1] = {};
            err = mtxpartition_localidx(
                &partition, part, partition.part_sizes[part],
                globalelem, localelem);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(0, localelem[0]);
        }
        {
            int part = 1;
            int64_t globalelem[1] = {1};
            int64_t localelem[1] = {};
            err = mtxpartition_localidx(
                &partition, part, partition.part_sizes[part],
                globalelem, localelem);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(0, localelem[0]);
        }
        {
            int part = 2;
            int64_t globalelem[1] = {2};
            int64_t localelem[1] = {};
            err = mtxpartition_localidx(
                &partition, part, partition.part_sizes[part],
                globalelem, localelem);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(0, localelem[0]);
        }
        {
            int part = 3;
            int64_t globalelem[1] = {3};
            int64_t localelem[1] = {};
            err = mtxpartition_localidx(
                &partition, part, partition.part_sizes[part],
                globalelem, localelem);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(0, localelem[0]);
        }
        {
            int part = 4;
            int64_t globalelem[1] = {4};
            int64_t localelem[1] = {};
            err = mtxpartition_localidx(
                &partition, part, partition.part_sizes[part],
                globalelem, localelem);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(0, localelem[0]);
        }
        mtxpartition_free(&partition);
    }
    {
        int size = 5;
        int num_parts = 6;
        struct mtxpartition partition;
        err = mtxpartition_init(
            &partition, mtx_block, size, num_parts, NULL, 0, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        int64_t globalelem[5] = {0,1,2,3,4};
        int parts[5] = {};
        err = mtxpartition_assign(&partition, size, globalelem, parts);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(0, parts[0]);
        TEST_ASSERT_EQ(1, parts[1]);
        TEST_ASSERT_EQ(2, parts[2]);
        TEST_ASSERT_EQ(3, parts[3]);
        TEST_ASSERT_EQ(4, parts[4]);
        {
            int part = 0;
            int64_t localelem[1] = {0};
            int64_t globalelem[1] = {};
            err = mtxpartition_globalidx(
                &partition, part, partition.part_sizes[part],
                localelem, globalelem);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(0, globalelem[0]);
        }
        {
            int part = 1;
            int64_t localelem[1] = {0};
            int64_t globalelem[1] = {};
            err = mtxpartition_globalidx(
                &partition, part, partition.part_sizes[part],
                localelem, globalelem);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(1, globalelem[0]);
        }
        {
            int part = 2;
            int64_t localelem[1] = {0};
            int64_t globalelem[1] = {};
            err = mtxpartition_globalidx(
                &partition, part, partition.part_sizes[part],
                localelem, globalelem);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(2, globalelem[0]);
        }
        {
            int part = 3;
            int64_t localelem[1] = {0};
            int64_t globalelem[1] = {};
            err = mtxpartition_globalidx(
                &partition, part, partition.part_sizes[part],
                localelem, globalelem);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(3, globalelem[0]);
        }
        {
            int part = 4;
            int64_t localelem[1] = {0};
            int64_t globalelem[1] = {};
            err = mtxpartition_globalidx(
                &partition, part, partition.part_sizes[part],
                localelem, globalelem);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(4, globalelem[0]);
        }
        {
            int part = 0;
            int64_t globalelem[1] = {0};
            int64_t localelem[1] = {};
            err = mtxpartition_localidx(
                &partition, part, partition.part_sizes[part],
                globalelem, localelem);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(0, localelem[0]);
        }
        {
            int part = 1;
            int64_t globalelem[1] = {1};
            int64_t localelem[1] = {};
            err = mtxpartition_localidx(
                &partition, part, partition.part_sizes[part],
                globalelem, localelem);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(0, localelem[0]);
        }
        {
            int part = 2;
            int64_t globalelem[1] = {2};
            int64_t localelem[1] = {};
            err = mtxpartition_localidx(
                &partition, part, partition.part_sizes[part],
                globalelem, localelem);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(0, localelem[0]);
        }
        {
            int part = 3;
            int64_t globalelem[1] = {3};
            int64_t localelem[1] = {};
            err = mtxpartition_localidx(
                &partition, part, partition.part_sizes[part],
                globalelem, localelem);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(0, localelem[0]);
        }
        {
            int part = 4;
            int64_t globalelem[1] = {4};
            int64_t localelem[1] = {};
            err = mtxpartition_localidx(
                &partition, part, partition.part_sizes[part],
                globalelem, localelem);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(0, localelem[0]);
        }
        mtxpartition_free(&partition);
    }

    return TEST_SUCCESS;
}

/**
 * ‘test_mtxpartition_cyclic()’ tests cyclic partitioning.
 */
int test_mtxpartition_cyclic(void)
{
    int err;
    {
        int size = 5;
        int num_parts = 1;
        struct mtxpartition partition;
        err = mtxpartition_init(
            &partition, mtx_cyclic, size, num_parts, NULL, 0, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        int64_t globalelem[5] = {0,1,2,3,4};
        int parts[5] = {};
        err = mtxpartition_assign(&partition, size, globalelem, parts);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(0, parts[0]);
        TEST_ASSERT_EQ(0, parts[1]);
        TEST_ASSERT_EQ(0, parts[2]);
        TEST_ASSERT_EQ(0, parts[3]);
        TEST_ASSERT_EQ(0, parts[4]);
        {
            int64_t localelem[5] = {0,1,2,3,4};
            int64_t globalelem[5] = {};
            err = mtxpartition_globalidx(
                &partition, 0, size, localelem, globalelem);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(0, globalelem[0]);
            TEST_ASSERT_EQ(1, globalelem[1]);
            TEST_ASSERT_EQ(2, globalelem[2]);
            TEST_ASSERT_EQ(3, globalelem[3]);
            TEST_ASSERT_EQ(4, globalelem[4]);
        }
        mtxpartition_free(&partition);
    }
    {
        int size = 5;
        int num_parts = 2;
        struct mtxpartition partition;
        err = mtxpartition_init(
            &partition, mtx_cyclic, size, num_parts, NULL, 0, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        int64_t globalelem[5] = {0,1,2,3,4};
        int parts[5] = {};
        err = mtxpartition_assign(&partition, size, globalelem, parts);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(0, parts[0]);
        TEST_ASSERT_EQ(1, parts[1]);
        TEST_ASSERT_EQ(0, parts[2]);
        TEST_ASSERT_EQ(1, parts[3]);
        TEST_ASSERT_EQ(0, parts[4]);
        {
            int part = 0;
            int64_t localelem[3] = {0,1,2};
            int64_t globalelem[3] = {};
            err = mtxpartition_globalidx(
                &partition, part, partition.part_sizes[part],
                localelem, globalelem);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(0, globalelem[0]);
            TEST_ASSERT_EQ(2, globalelem[1]);
            TEST_ASSERT_EQ(4, globalelem[2]);
        }
        {
            int part = 1;
            int64_t localelem[2] = {0,1};
            int64_t globalelem[2] = {};
            err = mtxpartition_globalidx(
                &partition, part, partition.part_sizes[part],
                localelem, globalelem);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(1, globalelem[0]);
            TEST_ASSERT_EQ(3, globalelem[1]);
        }
        mtxpartition_free(&partition);
    }
    {
        int size = 5;
        int num_parts = 3;
        struct mtxpartition partition;
        err = mtxpartition_init(
            &partition, mtx_cyclic, size, num_parts, NULL, 0, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        int64_t globalelem[5] = {0,1,2,3,4};
        int parts[5] = {};
        err = mtxpartition_assign(&partition, size, globalelem, parts);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(0, parts[0]);
        TEST_ASSERT_EQ(1, parts[1]);
        TEST_ASSERT_EQ(2, parts[2]);
        TEST_ASSERT_EQ(0, parts[3]);
        TEST_ASSERT_EQ(1, parts[4]);
        {
            int part = 0;
            int64_t localelem[2] = {0,1};
            int64_t globalelem[2] = {};
            err = mtxpartition_globalidx(
                &partition, part, partition.part_sizes[part],
                localelem, globalelem);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(0, globalelem[0]);
            TEST_ASSERT_EQ(3, globalelem[1]);
        }
        {
            int part = 1;
            int64_t localelem[2] = {0,1};
            int64_t globalelem[2] = {};
            err = mtxpartition_globalidx(
                &partition, part, partition.part_sizes[part],
                localelem, globalelem);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(1, globalelem[0]);
            TEST_ASSERT_EQ(4, globalelem[1]);
        }
        {
            int part = 2;
            int64_t localelem[1] = {0};
            int64_t globalelem[1] = {};
            err = mtxpartition_globalidx(
                &partition, part, partition.part_sizes[part],
                localelem, globalelem);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(2, globalelem[0]);
        }
        mtxpartition_free(&partition);
    }
    {
        int size = 5;
        int num_parts = 4;
        struct mtxpartition partition;
        err = mtxpartition_init(
            &partition, mtx_cyclic, size, num_parts, NULL, 0, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        int64_t globalelem[5] = {0,1,2,3,4};
        int parts[5] = {};
        err = mtxpartition_assign(&partition, size, globalelem, parts);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(0, parts[0]);
        TEST_ASSERT_EQ(1, parts[1]);
        TEST_ASSERT_EQ(2, parts[2]);
        TEST_ASSERT_EQ(3, parts[3]);
        TEST_ASSERT_EQ(0, parts[4]);
        {
            int part = 0;
            int64_t localelem[2] = {0,1};
            int64_t globalelem[2] = {};
            err = mtxpartition_globalidx(
                &partition, part, partition.part_sizes[part],
                localelem, globalelem);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(0, globalelem[0]);
            TEST_ASSERT_EQ(4, globalelem[1]);
        }
        {
            int part = 1;
            int64_t localelem[1] = {0};
            int64_t globalelem[1] = {};
            err = mtxpartition_globalidx(
                &partition, part, partition.part_sizes[part],
                localelem, globalelem);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(1, globalelem[0]);
        }
        {
            int part = 2;
            int64_t localelem[1] = {0};
            int64_t globalelem[1] = {};
            err = mtxpartition_globalidx(
                &partition, part, partition.part_sizes[part],
                localelem, globalelem);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(2, globalelem[0]);
        }
        {
            int part = 3;
            int64_t localelem[1] = {0};
            int64_t globalelem[1] = {};
            err = mtxpartition_globalidx(
                &partition, part, partition.part_sizes[part],
                localelem, globalelem);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(3, globalelem[0]);
        }
        mtxpartition_free(&partition);
    }
    {
        int size = 5;
        int num_parts = 5;
        struct mtxpartition partition;
        err = mtxpartition_init(
            &partition, mtx_cyclic, size, num_parts, NULL, 0, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        int64_t globalelem[5] = {0,1,2,3,4};
        int parts[5] = {};
        err = mtxpartition_assign(&partition, size, globalelem, parts);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(0, parts[0]);
        TEST_ASSERT_EQ(1, parts[1]);
        TEST_ASSERT_EQ(2, parts[2]);
        TEST_ASSERT_EQ(3, parts[3]);
        TEST_ASSERT_EQ(4, parts[4]);
        {
            int part = 0;
            int64_t localelem[1] = {0};
            int64_t globalelem[1] = {};
            err = mtxpartition_globalidx(
                &partition, part, partition.part_sizes[part],
                localelem, globalelem);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(0, globalelem[0]);
        }
        {
            int part = 1;
            int64_t localelem[1] = {0};
            int64_t globalelem[1] = {};
            err = mtxpartition_globalidx(
                &partition, part, partition.part_sizes[part],
                localelem, globalelem);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(1, globalelem[0]);
        }
        {
            int part = 2;
            int64_t localelem[1] = {0};
            int64_t globalelem[1] = {};
            err = mtxpartition_globalidx(
                &partition, part, partition.part_sizes[part],
                localelem, globalelem);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(2, globalelem[0]);
        }
        {
            int part = 3;
            int64_t localelem[1] = {0};
            int64_t globalelem[1] = {};
            err = mtxpartition_globalidx(
                &partition, part, partition.part_sizes[part],
                localelem, globalelem);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(3, globalelem[0]);
        }
        {
            int part = 4;
            int64_t localelem[1] = {0};
            int64_t globalelem[1] = {};
            err = mtxpartition_globalidx(
                &partition, part, partition.part_sizes[part],
                localelem, globalelem);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(4, globalelem[0]);
        }
        mtxpartition_free(&partition);
    }
    {
        int size = 5;
        int num_parts = 6;
        struct mtxpartition partition;
        err = mtxpartition_init(
            &partition, mtx_cyclic, size, num_parts, NULL, 0, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        int64_t globalelem[5] = {0,1,2,3,4};
        int parts[5] = {};
        err = mtxpartition_assign(&partition, size, globalelem, parts);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(0, parts[0]);
        TEST_ASSERT_EQ(1, parts[1]);
        TEST_ASSERT_EQ(2, parts[2]);
        TEST_ASSERT_EQ(3, parts[3]);
        TEST_ASSERT_EQ(4, parts[4]);
        {
            int part = 0;
            int64_t localelem[1] = {0};
            int64_t globalelem[1] = {};
            err = mtxpartition_globalidx(
                &partition, part, partition.part_sizes[part],
                localelem, globalelem);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(0, globalelem[0]);
        }
        {
            int part = 1;
            int64_t localelem[1] = {0};
            int64_t globalelem[1] = {};
            err = mtxpartition_globalidx(
                &partition, part, partition.part_sizes[part],
                localelem, globalelem);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(1, globalelem[0]);
        }
        {
            int part = 2;
            int64_t localelem[1] = {0};
            int64_t globalelem[1] = {};
            err = mtxpartition_globalidx(
                &partition, part, partition.part_sizes[part],
                localelem, globalelem);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(2, globalelem[0]);
        }
        {
            int part = 3;
            int64_t localelem[1] = {0};
            int64_t globalelem[1] = {};
            err = mtxpartition_globalidx(
                &partition, part, partition.part_sizes[part],
                localelem, globalelem);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(3, globalelem[0]);
        }
        {
            int part = 4;
            int64_t localelem[1] = {0};
            int64_t globalelem[1] = {};
            err = mtxpartition_globalidx(
                &partition, part, partition.part_sizes[part],
                localelem, globalelem);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(4, globalelem[0]);
        }
        mtxpartition_free(&partition);
    }

    return TEST_SUCCESS;
}

/**
 * ‘main()’ entry point and test driver.
 */
int main(int argc, char * argv[])
{
    TEST_SUITE_BEGIN("Running tests for partitioning functions\n");
    TEST_RUN(test_mtxpartition_singleton);
    TEST_RUN(test_mtxpartition_block);
    TEST_RUN(test_mtxpartition_cyclic);
    TEST_SUITE_END();
    return (TEST_SUITE_STATUS == TEST_SUCCESS) ?
        EXIT_SUCCESS : EXIT_FAILURE;
}
