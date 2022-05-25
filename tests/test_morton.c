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
 * Last modified: 2022-05-24
 *
 * Unit tests for Morton Z-order codes.
 */

#include <libmtx/error.h>

#include "libmtx/util/morton.h"
#include "test.h"

#include <errno.h>

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/* A simple, linear congruential random number generator. */
static uint64_t rand_uint64(void)
{
    static uint64_t i = 1;
    return (i = (164603309694725029ull * i) % 14738995463583502973ull);
}

/**
 * ‘test_morton2d_from_cartesian()’ tests converting from Cartesian to
 * 2D Morton Z-order.
 */
int test_morton2d_from_cartesian(void)
{
    /* 32-bit unsigned integers */
    {
        int size = 16;
        uint32_t x[16] = {0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3};
        uint32_t y[16] = {0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3};
        uint64_t z[16] = {};
        int err = morton2d_from_cartesian_uint32(
            size, sizeof(*x), x, sizeof(*y), y, sizeof(*z), z);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( 0, z[ 0]);
        TEST_ASSERT_EQ( 1, z[ 1]);
        TEST_ASSERT_EQ( 4, z[ 2]);
        TEST_ASSERT_EQ( 5, z[ 3]);
        TEST_ASSERT_EQ( 2, z[ 4]);
        TEST_ASSERT_EQ( 3, z[ 5]);
        TEST_ASSERT_EQ( 6, z[ 6]);
        TEST_ASSERT_EQ( 7, z[ 7]);
        TEST_ASSERT_EQ( 8, z[ 8]);
        TEST_ASSERT_EQ( 9, z[ 9]);
        TEST_ASSERT_EQ(12, z[10]);
        TEST_ASSERT_EQ(13, z[11]);
        TEST_ASSERT_EQ(10, z[12]);
        TEST_ASSERT_EQ(11, z[13]);
        TEST_ASSERT_EQ(14, z[14]);
        TEST_ASSERT_EQ(15, z[15]);
    }

    /* 64-bit unsigned integers */
    {
        int size = 16;
        uint64_t x[16] = {0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3};
        uint64_t y[16] = {0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3};
        uint64_t z0[16] = {};
        uint64_t z1[16] = {};
        int err = morton2d_from_cartesian_uint64(
            size, sizeof(*x), x, sizeof(*y), y, sizeof(*z0), z0, sizeof(*z1), z1);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( 0, z0[ 0]); TEST_ASSERT_EQ( 0, z1[ 0]);
        TEST_ASSERT_EQ( 0, z0[ 1]); TEST_ASSERT_EQ( 1, z1[ 1]);
        TEST_ASSERT_EQ( 0, z0[ 2]); TEST_ASSERT_EQ( 4, z1[ 2]);
        TEST_ASSERT_EQ( 0, z0[ 3]); TEST_ASSERT_EQ( 5, z1[ 3]);
        TEST_ASSERT_EQ( 0, z0[ 4]); TEST_ASSERT_EQ( 2, z1[ 4]);
        TEST_ASSERT_EQ( 0, z0[ 5]); TEST_ASSERT_EQ( 3, z1[ 5]);
        TEST_ASSERT_EQ( 0, z0[ 6]); TEST_ASSERT_EQ( 6, z1[ 6]);
        TEST_ASSERT_EQ( 0, z0[ 7]); TEST_ASSERT_EQ( 7, z1[ 7]);
        TEST_ASSERT_EQ( 0, z0[ 8]); TEST_ASSERT_EQ( 8, z1[ 8]);
        TEST_ASSERT_EQ( 0, z0[ 9]); TEST_ASSERT_EQ( 9, z1[ 9]);
        TEST_ASSERT_EQ( 0, z0[10]); TEST_ASSERT_EQ(12, z1[10]);
        TEST_ASSERT_EQ( 0, z0[11]); TEST_ASSERT_EQ(13, z1[11]);
        TEST_ASSERT_EQ( 0, z0[12]); TEST_ASSERT_EQ(10, z1[12]);
        TEST_ASSERT_EQ( 0, z0[13]); TEST_ASSERT_EQ(11, z1[13]);
        TEST_ASSERT_EQ( 0, z0[14]); TEST_ASSERT_EQ(14, z1[14]);
        TEST_ASSERT_EQ( 0, z0[15]); TEST_ASSERT_EQ(15, z1[15]);
    }
    return TEST_SUCCESS;
}

/**
 * ‘test_morton2d_to_cartesian()’ tests converting to Cartesian from
 * 2D Morton Z-order.
 */
int test_morton2d_to_cartesian(void)
{
    /* 32-bit unsigned integers */
    {
        int size = 16;
        uint32_t x[16] = {};
        uint32_t y[16] = {};
        uint64_t z[16] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
        int err = morton2d_to_cartesian_uint32(
            size, sizeof(*z), z, sizeof(*x), x, sizeof(*y), y);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(0, x[ 0]); TEST_ASSERT_EQ(0, y[ 0]);
        TEST_ASSERT_EQ(0, x[ 1]); TEST_ASSERT_EQ(1, y[ 1]);
        TEST_ASSERT_EQ(1, x[ 2]); TEST_ASSERT_EQ(0, y[ 2]);
        TEST_ASSERT_EQ(1, x[ 3]); TEST_ASSERT_EQ(1, y[ 3]);
        TEST_ASSERT_EQ(0, x[ 4]); TEST_ASSERT_EQ(2, y[ 4]);
        TEST_ASSERT_EQ(0, x[ 5]); TEST_ASSERT_EQ(3, y[ 5]);
        TEST_ASSERT_EQ(1, x[ 6]); TEST_ASSERT_EQ(2, y[ 6]);
        TEST_ASSERT_EQ(1, x[ 7]); TEST_ASSERT_EQ(3, y[ 7]);
        TEST_ASSERT_EQ(2, x[ 8]); TEST_ASSERT_EQ(0, y[ 8]);
        TEST_ASSERT_EQ(2, x[ 9]); TEST_ASSERT_EQ(1, y[ 9]);
        TEST_ASSERT_EQ(3, x[10]); TEST_ASSERT_EQ(0, y[10]);
        TEST_ASSERT_EQ(3, x[11]); TEST_ASSERT_EQ(1, y[11]);
        TEST_ASSERT_EQ(2, x[12]); TEST_ASSERT_EQ(2, y[12]);
        TEST_ASSERT_EQ(2, x[13]); TEST_ASSERT_EQ(3, y[13]);
        TEST_ASSERT_EQ(3, x[14]); TEST_ASSERT_EQ(2, y[14]);
        TEST_ASSERT_EQ(3, x[15]); TEST_ASSERT_EQ(3, y[15]);
    }
#if 0
    /* 64-bit unsigned integers */
    {
        int size = 16;
        uint64_t x[16] = {};
        uint64_t y[16] = {};
        uint64_t z[16][2] = {
            {0,0},{0,1},{0,2},{0,3},{0,4},{0,5},{0,6},{0,7},{0,8},{0,9},{0,10},{0,11},{0,12},{0,13},{0,14},{0,15}};
        int err = morton2d_to_cartesian_uint64(
            size, sizeof(*z), z, sizeof(*x), x, sizeof(*y), y);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(0, x[ 0]); TEST_ASSERT_EQ(0, y[ 0]);
        TEST_ASSERT_EQ(0, x[ 1]); TEST_ASSERT_EQ(1, y[ 1]);
        TEST_ASSERT_EQ(1, x[ 2]); TEST_ASSERT_EQ(0, y[ 2]);
        TEST_ASSERT_EQ(1, x[ 3]); TEST_ASSERT_EQ(1, y[ 3]);
        TEST_ASSERT_EQ(0, x[ 4]); TEST_ASSERT_EQ(2, y[ 4]);
        TEST_ASSERT_EQ(0, x[ 5]); TEST_ASSERT_EQ(3, y[ 5]);
        TEST_ASSERT_EQ(1, x[ 6]); TEST_ASSERT_EQ(2, y[ 6]);
        TEST_ASSERT_EQ(1, x[ 7]); TEST_ASSERT_EQ(3, y[ 7]);
        TEST_ASSERT_EQ(2, x[ 8]); TEST_ASSERT_EQ(0, y[ 8]);
        TEST_ASSERT_EQ(2, x[ 9]); TEST_ASSERT_EQ(1, y[ 9]);
        TEST_ASSERT_EQ(3, x[10]); TEST_ASSERT_EQ(0, y[10]);
        TEST_ASSERT_EQ(3, x[11]); TEST_ASSERT_EQ(1, y[11]);
        TEST_ASSERT_EQ(2, x[12]); TEST_ASSERT_EQ(2, y[12]);
        TEST_ASSERT_EQ(2, x[13]); TEST_ASSERT_EQ(3, y[13]);
        TEST_ASSERT_EQ(3, x[14]); TEST_ASSERT_EQ(2, y[14]);
        TEST_ASSERT_EQ(3, x[15]); TEST_ASSERT_EQ(3, y[15]);
    }

    {
        int size = 100;
        uint64_t xin[100] = {};
        uint64_t yin[100] = {};
        uint64_t z[100][2] = {};
        uint64_t xout[100] = {};
        uint64_t yout[100] = {};
        for (int i = 0; i < size; i++) {
            xin[i] = rand_uint64();
            yin[i] = rand_uint64();
        }
        int err = morton2d_from_cartesian_uint64(
            size, sizeof(*xin), xin, sizeof(*yin), yin,
            sizeof(*z), &z[0][0], sizeof(*z) &z[0][1]);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = morton2d_to_cartesian_uint64(
            size, sizeof(*z), z, sizeof(*xout), xout, sizeof(*yout), yout);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        for (int i = 0; i < size; i++) {
            TEST_ASSERT_MSG(
                xin[i] == xout[i] && yin[i] == yout[i],
                "i=%d, xin[i]=%"PRIu64", xout[i]=%"PRIu64"",
                "yin[i]=%"PRIu64", yout[i]=%"PRIu64,
                i, xin[i], xout[i], yin[i], yout[i]);
        }
    }

#endif
    return TEST_SUCCESS;
}

/**
 * ‘main()’ entry point and test driver.
 */
int main(int argc, char * argv[])
{
    TEST_SUITE_BEGIN("Running tests for Morton Z-order codes\n");
    TEST_RUN(test_morton2d_from_cartesian);
    TEST_RUN(test_morton2d_to_cartesian);
    TEST_SUITE_END();
    return (TEST_SUITE_STATUS == TEST_SUCCESS) ?
        EXIT_SUCCESS : EXIT_FAILURE;
}
