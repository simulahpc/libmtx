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
 * Last modified: 2022-04-26
 *
 * Unit tests for merging functions.
 */

#include <libmtx/error.h>

#include "libmtx/util/merge.h"
#include "test.h"

#include <errno.h>

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * ‘test_merge_sorted()’ tests merging two sorted arrays.
 */
int test_merge_sorted(void)
{
    /* 32-bit signed integers */
    {
        int err;
        int64_t asize = 8;
        int32_t a[8] = {0,2,4,6,8,10,12,14};
        int64_t bsize = 9;
        int32_t b[9] = {0,1,2,4,5,6,8,9,12};
        int64_t csize = 17;
        int32_t c[17] = {};
        err = merge_sorted_int32(csize, c, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( 0, c[ 0]);
        TEST_ASSERT_EQ( 0, c[ 1]);
        TEST_ASSERT_EQ( 1, c[ 2]);
        TEST_ASSERT_EQ( 2, c[ 3]);
        TEST_ASSERT_EQ( 2, c[ 4]);
        TEST_ASSERT_EQ( 4, c[ 5]);
        TEST_ASSERT_EQ( 4, c[ 6]);
        TEST_ASSERT_EQ( 5, c[ 7]);
        TEST_ASSERT_EQ( 6, c[ 8]);
        TEST_ASSERT_EQ( 6, c[ 9]);
        TEST_ASSERT_EQ( 8, c[10]);
        TEST_ASSERT_EQ( 8, c[11]);
        TEST_ASSERT_EQ( 9, c[12]);
        TEST_ASSERT_EQ(10, c[13]);
        TEST_ASSERT_EQ(12, c[14]);
        TEST_ASSERT_EQ(12, c[15]);
        TEST_ASSERT_EQ(14, c[16]);
    }
    {
        int err;
        int64_t asize = 9;
        int32_t a[ 9] = {0,2,4,6,6,8,10,12,14};
        int64_t bsize = 10;
        int32_t b[10] = {0,0,1,2,4,5,6,8,9,12};
        int64_t csize = 19;
        int32_t c[19] = {};
        err = merge_sorted_int32(csize, c, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( 0, c[ 0]);
        TEST_ASSERT_EQ( 0, c[ 1]);
        TEST_ASSERT_EQ( 0, c[ 2]);
        TEST_ASSERT_EQ( 1, c[ 3]);
        TEST_ASSERT_EQ( 2, c[ 4]);
        TEST_ASSERT_EQ( 2, c[ 5]);
        TEST_ASSERT_EQ( 4, c[ 6]);
        TEST_ASSERT_EQ( 4, c[ 7]);
        TEST_ASSERT_EQ( 5, c[ 8]);
        TEST_ASSERT_EQ( 6, c[ 9]);
        TEST_ASSERT_EQ( 6, c[10]);
        TEST_ASSERT_EQ( 6, c[11]);
        TEST_ASSERT_EQ( 8, c[12]);
        TEST_ASSERT_EQ( 8, c[13]);
        TEST_ASSERT_EQ( 9, c[14]);
        TEST_ASSERT_EQ(10, c[15]);
        TEST_ASSERT_EQ(12, c[16]);
        TEST_ASSERT_EQ(12, c[17]);
        TEST_ASSERT_EQ(14, c[18]);
    }

    /* 64-bit signed integers */
    {
        int err;
        int64_t asize = 8;
        int64_t a[8] = {0,2,4,6,8,10,12,14};
        int64_t bsize = 9;
        int64_t b[9] = {0,1,2,4,5,6,8,9,12};
        int64_t csize = 17;
        int64_t c[17] = {};
        err = merge_sorted_int64(csize, c, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( 0, c[ 0]);
        TEST_ASSERT_EQ( 0, c[ 1]);
        TEST_ASSERT_EQ( 1, c[ 2]);
        TEST_ASSERT_EQ( 2, c[ 3]);
        TEST_ASSERT_EQ( 2, c[ 4]);
        TEST_ASSERT_EQ( 4, c[ 5]);
        TEST_ASSERT_EQ( 4, c[ 6]);
        TEST_ASSERT_EQ( 5, c[ 7]);
        TEST_ASSERT_EQ( 6, c[ 8]);
        TEST_ASSERT_EQ( 6, c[ 9]);
        TEST_ASSERT_EQ( 8, c[10]);
        TEST_ASSERT_EQ( 8, c[11]);
        TEST_ASSERT_EQ( 9, c[12]);
        TEST_ASSERT_EQ(10, c[13]);
        TEST_ASSERT_EQ(12, c[14]);
        TEST_ASSERT_EQ(12, c[15]);
        TEST_ASSERT_EQ(14, c[16]);
    }
    {
        int err;
        int64_t asize = 9;
        int64_t a[ 9] = {0,2,4,6,6,8,10,12,14};
        int64_t bsize = 10;
        int64_t b[10] = {0,0,1,2,4,5,6,8,9,12};
        int64_t csize = 19;
        int64_t c[19] = {};
        err = merge_sorted_int64(csize, c, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( 0, c[ 0]);
        TEST_ASSERT_EQ( 0, c[ 1]);
        TEST_ASSERT_EQ( 0, c[ 2]);
        TEST_ASSERT_EQ( 1, c[ 3]);
        TEST_ASSERT_EQ( 2, c[ 4]);
        TEST_ASSERT_EQ( 2, c[ 5]);
        TEST_ASSERT_EQ( 4, c[ 6]);
        TEST_ASSERT_EQ( 4, c[ 7]);
        TEST_ASSERT_EQ( 5, c[ 8]);
        TEST_ASSERT_EQ( 6, c[ 9]);
        TEST_ASSERT_EQ( 6, c[10]);
        TEST_ASSERT_EQ( 6, c[11]);
        TEST_ASSERT_EQ( 8, c[12]);
        TEST_ASSERT_EQ( 8, c[13]);
        TEST_ASSERT_EQ( 9, c[14]);
        TEST_ASSERT_EQ(10, c[15]);
        TEST_ASSERT_EQ(12, c[16]);
        TEST_ASSERT_EQ(12, c[17]);
        TEST_ASSERT_EQ(14, c[18]);
    }

    /* signed integers */
    {
        int err;
        int64_t asize = 8;
        int a[8] = {0,2,4,6,8,10,12,14};
        int64_t bsize = 9;
        int b[9] = {0,1,2,4,5,6,8,9,12};
        int64_t csize = 17;
        int c[17] = {};
        err = merge_sorted_int(csize, c, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( 0, c[ 0]);
        TEST_ASSERT_EQ( 0, c[ 1]);
        TEST_ASSERT_EQ( 1, c[ 2]);
        TEST_ASSERT_EQ( 2, c[ 3]);
        TEST_ASSERT_EQ( 2, c[ 4]);
        TEST_ASSERT_EQ( 4, c[ 5]);
        TEST_ASSERT_EQ( 4, c[ 6]);
        TEST_ASSERT_EQ( 5, c[ 7]);
        TEST_ASSERT_EQ( 6, c[ 8]);
        TEST_ASSERT_EQ( 6, c[ 9]);
        TEST_ASSERT_EQ( 8, c[10]);
        TEST_ASSERT_EQ( 8, c[11]);
        TEST_ASSERT_EQ( 9, c[12]);
        TEST_ASSERT_EQ(10, c[13]);
        TEST_ASSERT_EQ(12, c[14]);
        TEST_ASSERT_EQ(12, c[15]);
        TEST_ASSERT_EQ(14, c[16]);
    }
    {
        int err;
        int64_t asize = 9;
        int a[ 9] = {0,2,4,6,6,8,10,12,14};
        int64_t bsize = 10;
        int b[10] = {0,0,1,2,4,5,6,8,9,12};
        int64_t csize = 19;
        int c[19] = {};
        err = merge_sorted_int(csize, c, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( 0, c[ 0]);
        TEST_ASSERT_EQ( 0, c[ 1]);
        TEST_ASSERT_EQ( 0, c[ 2]);
        TEST_ASSERT_EQ( 1, c[ 3]);
        TEST_ASSERT_EQ( 2, c[ 4]);
        TEST_ASSERT_EQ( 2, c[ 5]);
        TEST_ASSERT_EQ( 4, c[ 6]);
        TEST_ASSERT_EQ( 4, c[ 7]);
        TEST_ASSERT_EQ( 5, c[ 8]);
        TEST_ASSERT_EQ( 6, c[ 9]);
        TEST_ASSERT_EQ( 6, c[10]);
        TEST_ASSERT_EQ( 6, c[11]);
        TEST_ASSERT_EQ( 8, c[12]);
        TEST_ASSERT_EQ( 8, c[13]);
        TEST_ASSERT_EQ( 9, c[14]);
        TEST_ASSERT_EQ(10, c[15]);
        TEST_ASSERT_EQ(12, c[16]);
        TEST_ASSERT_EQ(12, c[17]);
        TEST_ASSERT_EQ(14, c[18]);
    }
    return TEST_SUCCESS;
}

/**
 * ‘test_setunion_sorted_unique()’ tests merging two sorted arrays of
 * unique values (i.e., no duplicates), based on a set union
 * operation.
 */
int test_setunion_sorted_unique(void)
{
    /* 32-bit signed integers */
    {
        int err;
        int64_t asize = 8;
        int32_t a[8] = {0,2,4,6,8,10,12,14};
        int64_t bsize = 9;
        int32_t b[9] = {0,1,2,4,5,6,8,9,12};
        int64_t csize = 17;
        int32_t c[17] = {};
        err = setunion_sorted_unique_int32(&csize, NULL, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(11, csize);
        err = setunion_sorted_unique_int32(&csize, c, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(11, csize);
        TEST_ASSERT_EQ( 0, c[0]);
        TEST_ASSERT_EQ( 1, c[1]);
        TEST_ASSERT_EQ( 2, c[2]);
        TEST_ASSERT_EQ( 4, c[3]);
        TEST_ASSERT_EQ( 5, c[4]);
        TEST_ASSERT_EQ( 6, c[5]);
        TEST_ASSERT_EQ( 8, c[6]);
        TEST_ASSERT_EQ( 9, c[7]);
        TEST_ASSERT_EQ(10, c[8]);
        TEST_ASSERT_EQ(12, c[9]);
        TEST_ASSERT_EQ(14, c[10]);
    }

    /* 64-bit signed integers */
    {
        int err;
        int64_t asize = 8;
        int64_t a[8] = {0,2,4,6,8,10,12,14};
        int64_t bsize = 9;
        int64_t b[9] = {0,1,2,4,5,6,8,9,12};
        int64_t csize = 17;
        int64_t c[17] = {};
        err = setunion_sorted_unique_int64(&csize, NULL, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(11, csize);
        err = setunion_sorted_unique_int64(&csize, c, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(11, csize);
        TEST_ASSERT_EQ( 0, c[0]);
        TEST_ASSERT_EQ( 1, c[1]);
        TEST_ASSERT_EQ( 2, c[2]);
        TEST_ASSERT_EQ( 4, c[3]);
        TEST_ASSERT_EQ( 5, c[4]);
        TEST_ASSERT_EQ( 6, c[5]);
        TEST_ASSERT_EQ( 8, c[6]);
        TEST_ASSERT_EQ( 9, c[7]);
        TEST_ASSERT_EQ(10, c[8]);
        TEST_ASSERT_EQ(12, c[9]);
        TEST_ASSERT_EQ(14, c[10]);
    }

    /* signed integers */
    {
        int err;
        int64_t asize = 8;
        int a[8] = {0,2,4,6,8,10,12,14};
        int64_t bsize = 9;
        int b[9] = {0,1,2,4,5,6,8,9,12};
        int64_t csize = 17;
        int c[17] = {};
        err = setunion_sorted_unique_int(&csize, NULL, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(11, csize);
        err = setunion_sorted_unique_int(&csize, c, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(11, csize);
        TEST_ASSERT_EQ( 0, c[0]);
        TEST_ASSERT_EQ( 1, c[1]);
        TEST_ASSERT_EQ( 2, c[2]);
        TEST_ASSERT_EQ( 4, c[3]);
        TEST_ASSERT_EQ( 5, c[4]);
        TEST_ASSERT_EQ( 6, c[5]);
        TEST_ASSERT_EQ( 8, c[6]);
        TEST_ASSERT_EQ( 9, c[7]);
        TEST_ASSERT_EQ(10, c[8]);
        TEST_ASSERT_EQ(12, c[9]);
        TEST_ASSERT_EQ(14, c[10]);
    }
    return TEST_SUCCESS;
}

/**
 * ‘test_setunion_sorted_nonunique()’ tests merging two sorted arrays,
 * possibly containing non-unique (duplicate) values, based on a set
 * union operation.
 */
int test_setunion_sorted_nonunique(void)
{
    /* 32-bit signed integers */
    {
        int err;
        int64_t asize = 8;
        int32_t a[8] = {0,2,4,6,8,10,12,14};
        int64_t bsize = 9;
        int32_t b[9] = {0,1,2,4,5,6,8,9,12};
        int64_t csize = 17;
        int32_t c[17] = {};
        err = setunion_sorted_nonunique_int32(&csize, NULL, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(11, csize);
        err = setunion_sorted_nonunique_int32(&csize, c, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(11, csize);
        TEST_ASSERT_EQ( 0, c[0]);
        TEST_ASSERT_EQ( 1, c[1]);
        TEST_ASSERT_EQ( 2, c[2]);
        TEST_ASSERT_EQ( 4, c[3]);
        TEST_ASSERT_EQ( 5, c[4]);
        TEST_ASSERT_EQ( 6, c[5]);
        TEST_ASSERT_EQ( 8, c[6]);
        TEST_ASSERT_EQ( 9, c[7]);
        TEST_ASSERT_EQ(10, c[8]);
        TEST_ASSERT_EQ(12, c[9]);
        TEST_ASSERT_EQ(14, c[10]);
    }
    {
        int err;
        int64_t asize = 11;
        int32_t a[11] = {0,2,2,2,4,6,8,10,10,12,14};
        int64_t bsize = 13;
        int32_t b[13] = {0,0,1,2,2,4,4,5,6,8,9,9,12};
        int64_t csize = 24;
        int32_t c[24] = {};
        err = setunion_sorted_nonunique_int32(&csize, NULL, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(11, csize);
        err = setunion_sorted_nonunique_int32(&csize, c, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(11, csize);
        TEST_ASSERT_EQ( 0, c[0]);
        TEST_ASSERT_EQ( 1, c[1]);
        TEST_ASSERT_EQ( 2, c[2]);
        TEST_ASSERT_EQ( 4, c[3]);
        TEST_ASSERT_EQ( 5, c[4]);
        TEST_ASSERT_EQ( 6, c[5]);
        TEST_ASSERT_EQ( 8, c[6]);
        TEST_ASSERT_EQ( 9, c[7]);
        TEST_ASSERT_EQ(10, c[8]);
        TEST_ASSERT_EQ(12, c[9]);
        TEST_ASSERT_EQ(14, c[10]);
    }

    /* 64-bit signed integers */
    {
        int err;
        int64_t asize = 8;
        int64_t a[8] = {0,2,4,6,8,10,12,14};
        int64_t bsize = 9;
        int64_t b[9] = {0,1,2,4,5,6,8,9,12};
        int64_t csize = 17;
        int64_t c[17] = {};
        err = setunion_sorted_nonunique_int64(&csize, NULL, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(11, csize);
        err = setunion_sorted_nonunique_int64(&csize, c, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(11, csize);
        TEST_ASSERT_EQ( 0, c[0]);
        TEST_ASSERT_EQ( 1, c[1]);
        TEST_ASSERT_EQ( 2, c[2]);
        TEST_ASSERT_EQ( 4, c[3]);
        TEST_ASSERT_EQ( 5, c[4]);
        TEST_ASSERT_EQ( 6, c[5]);
        TEST_ASSERT_EQ( 8, c[6]);
        TEST_ASSERT_EQ( 9, c[7]);
        TEST_ASSERT_EQ(10, c[8]);
        TEST_ASSERT_EQ(12, c[9]);
        TEST_ASSERT_EQ(14, c[10]);
    }
    {
        int err;
        int64_t asize = 11;
        int64_t a[11] = {0,2,2,2,4,6,8,10,10,12,14};
        int64_t bsize = 13;
        int64_t b[13] = {0,0,1,2,2,4,4,5,6,8,9,9,12};
        int64_t csize = 24;
        int64_t c[24] = {};
        err = setunion_sorted_nonunique_int64(&csize, NULL, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(11, csize);
        err = setunion_sorted_nonunique_int64(&csize, c, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(11, csize);
        TEST_ASSERT_EQ( 0, c[0]);
        TEST_ASSERT_EQ( 1, c[1]);
        TEST_ASSERT_EQ( 2, c[2]);
        TEST_ASSERT_EQ( 4, c[3]);
        TEST_ASSERT_EQ( 5, c[4]);
        TEST_ASSERT_EQ( 6, c[5]);
        TEST_ASSERT_EQ( 8, c[6]);
        TEST_ASSERT_EQ( 9, c[7]);
        TEST_ASSERT_EQ(10, c[8]);
        TEST_ASSERT_EQ(12, c[9]);
        TEST_ASSERT_EQ(14, c[10]);
    }

    /* signed integers */
    {
        int err;
        int64_t asize = 8;
        int a[8] = {0,2,4,6,8,10,12,14};
        int64_t bsize = 9;
        int b[9] = {0,1,2,4,5,6,8,9,12};
        int64_t csize = 17;
        int c[17] = {};
        err = setunion_sorted_nonunique_int(&csize, NULL, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(11, csize);
        err = setunion_sorted_nonunique_int(&csize, c, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(11, csize);
        TEST_ASSERT_EQ( 0, c[0]);
        TEST_ASSERT_EQ( 1, c[1]);
        TEST_ASSERT_EQ( 2, c[2]);
        TEST_ASSERT_EQ( 4, c[3]);
        TEST_ASSERT_EQ( 5, c[4]);
        TEST_ASSERT_EQ( 6, c[5]);
        TEST_ASSERT_EQ( 8, c[6]);
        TEST_ASSERT_EQ( 9, c[7]);
        TEST_ASSERT_EQ(10, c[8]);
        TEST_ASSERT_EQ(12, c[9]);
        TEST_ASSERT_EQ(14, c[10]);
    }
    {
        int err;
        int64_t asize = 11;
        int a[11] = {0,2,2,2,4,6,8,10,10,12,14};
        int64_t bsize = 13;
        int b[13] = {0,0,1,2,2,4,4,5,6,8,9,9,12};
        int64_t csize = 24;
        int c[24] = {};
        err = setunion_sorted_nonunique_int(&csize, NULL, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(11, csize);
        err = setunion_sorted_nonunique_int(&csize, c, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(11, csize);
        TEST_ASSERT_EQ( 0, c[0]);
        TEST_ASSERT_EQ( 1, c[1]);
        TEST_ASSERT_EQ( 2, c[2]);
        TEST_ASSERT_EQ( 4, c[3]);
        TEST_ASSERT_EQ( 5, c[4]);
        TEST_ASSERT_EQ( 6, c[5]);
        TEST_ASSERT_EQ( 8, c[6]);
        TEST_ASSERT_EQ( 9, c[7]);
        TEST_ASSERT_EQ(10, c[8]);
        TEST_ASSERT_EQ(12, c[9]);
        TEST_ASSERT_EQ(14, c[10]);
    }
    return TEST_SUCCESS;
}

/**
 * ‘test_setunion_unsorted_unique()’ tests merging two unsorted arrays
 * of unique values (i.e., no duplicates), based on a set union
 * operation.
 */
int test_setunion_unsorted_unique(void)
{
    /* 32-bit signed integers */
    {
        int err;
        int64_t asize = 8;
        int32_t a[8] = {10,2,14,6,8,0,4,12};
        int64_t bsize = 9;
        int32_t b[9] = {0,1,4,2,5,6,12,9,8};
        int64_t csize = 17;
        int32_t c[17] = {};
        err = setunion_unsorted_unique_int32(&csize, NULL, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(11, csize);
        err = setunion_unsorted_unique_int32(&csize, c, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(11, csize);
        TEST_ASSERT_EQ( 0, c[0]);
        TEST_ASSERT_EQ( 1, c[1]);
        TEST_ASSERT_EQ( 2, c[2]);
        TEST_ASSERT_EQ( 4, c[3]);
        TEST_ASSERT_EQ( 5, c[4]);
        TEST_ASSERT_EQ( 6, c[5]);
        TEST_ASSERT_EQ( 8, c[6]);
        TEST_ASSERT_EQ( 9, c[7]);
        TEST_ASSERT_EQ(10, c[8]);
        TEST_ASSERT_EQ(12, c[9]);
        TEST_ASSERT_EQ(14, c[10]);
    }

    /* 64-bit signed integers */
    {
        int err;
        int64_t asize = 8;
        int64_t a[8] = {10,2,14,6,8,0,4,12};
        int64_t bsize = 9;
        int64_t b[9] = {0,1,4,2,5,6,12,9,8};
        int64_t csize = 17;
        int64_t c[17] = {};
        err = setunion_unsorted_unique_int64(&csize, NULL, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(11, csize);
        err = setunion_unsorted_unique_int64(&csize, c, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(11, csize);
        TEST_ASSERT_EQ( 0, c[0]);
        TEST_ASSERT_EQ( 1, c[1]);
        TEST_ASSERT_EQ( 2, c[2]);
        TEST_ASSERT_EQ( 4, c[3]);
        TEST_ASSERT_EQ( 5, c[4]);
        TEST_ASSERT_EQ( 6, c[5]);
        TEST_ASSERT_EQ( 8, c[6]);
        TEST_ASSERT_EQ( 9, c[7]);
        TEST_ASSERT_EQ(10, c[8]);
        TEST_ASSERT_EQ(12, c[9]);
        TEST_ASSERT_EQ(14, c[10]);
    }

    /* signed integers */
    {
        int err;
        int64_t asize = 8;
        int a[8] = {10,2,14,6,8,0,4,12};
        int64_t bsize = 9;
        int b[9] = {0,1,4,2,5,6,12,9,8};
        int64_t csize = 17;
        int c[17] = {};
        err = setunion_unsorted_unique_int(&csize, NULL, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(11, csize);
        err = setunion_unsorted_unique_int(&csize, c, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(11, csize);
        TEST_ASSERT_EQ( 0, c[0]);
        TEST_ASSERT_EQ( 1, c[1]);
        TEST_ASSERT_EQ( 2, c[2]);
        TEST_ASSERT_EQ( 4, c[3]);
        TEST_ASSERT_EQ( 5, c[4]);
        TEST_ASSERT_EQ( 6, c[5]);
        TEST_ASSERT_EQ( 8, c[6]);
        TEST_ASSERT_EQ( 9, c[7]);
        TEST_ASSERT_EQ(10, c[8]);
        TEST_ASSERT_EQ(12, c[9]);
        TEST_ASSERT_EQ(14, c[10]);
    }
    return TEST_SUCCESS;
}

/**
 * ‘test_setunion_unsorted_nonunique()’ tests merging two unsorted
 * arrays, possibly containing non-unique (duplicate) values, based on
 * a set union operation.
 */
int test_setunion_unsorted_nonunique(void)
{
    /* 32-bit signed integers */
    {
        int err;
        int64_t asize = 8;
        int32_t a[8] = {10,2,14,6,8,0,4,12};
        int64_t bsize = 9;
        int32_t b[9] = {0,1,4,2,5,6,12,9,8};
        int64_t csize = 17;
        int32_t c[17] = {};
        err = setunion_unsorted_nonunique_int32(&csize, NULL, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(11, csize);
        err = setunion_unsorted_nonunique_int32(&csize, c, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(11, csize);
        TEST_ASSERT_EQ( 0, c[0]);
        TEST_ASSERT_EQ( 1, c[1]);
        TEST_ASSERT_EQ( 2, c[2]);
        TEST_ASSERT_EQ( 4, c[3]);
        TEST_ASSERT_EQ( 5, c[4]);
        TEST_ASSERT_EQ( 6, c[5]);
        TEST_ASSERT_EQ( 8, c[6]);
        TEST_ASSERT_EQ( 9, c[7]);
        TEST_ASSERT_EQ(10, c[8]);
        TEST_ASSERT_EQ(12, c[9]);
        TEST_ASSERT_EQ(14, c[10]);
    }
    {
        int err;
        int64_t asize = 11;
        int32_t a[11] = {2,0,2,2,4,8,6,10,14,12,10};
        int64_t bsize = 13;
        int32_t b[13] = {4,0,1,2,2,4,0,5,9,8,9,6,12};
        int64_t csize = 24;
        int32_t c[24] = {};
        err = setunion_unsorted_nonunique_int32(&csize, NULL, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(11, csize);
        err = setunion_unsorted_nonunique_int32(&csize, c, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(11, csize);
        TEST_ASSERT_EQ( 0, c[0]);
        TEST_ASSERT_EQ( 1, c[1]);
        TEST_ASSERT_EQ( 2, c[2]);
        TEST_ASSERT_EQ( 4, c[3]);
        TEST_ASSERT_EQ( 5, c[4]);
        TEST_ASSERT_EQ( 6, c[5]);
        TEST_ASSERT_EQ( 8, c[6]);
        TEST_ASSERT_EQ( 9, c[7]);
        TEST_ASSERT_EQ(10, c[8]);
        TEST_ASSERT_EQ(12, c[9]);
        TEST_ASSERT_EQ(14, c[10]);
    }

    /* 64-bit signed integers */
    {
        int err;
        int64_t asize = 8;
        int64_t a[8] = {10,2,14,6,8,0,4,12};
        int64_t bsize = 9;
        int64_t b[9] = {0,1,4,2,5,6,12,9,8};
        int64_t csize = 17;
        int64_t c[17] = {};
        err = setunion_unsorted_nonunique_int64(&csize, NULL, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(11, csize);
        err = setunion_unsorted_nonunique_int64(&csize, c, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(11, csize);
        TEST_ASSERT_EQ( 0, c[0]);
        TEST_ASSERT_EQ( 1, c[1]);
        TEST_ASSERT_EQ( 2, c[2]);
        TEST_ASSERT_EQ( 4, c[3]);
        TEST_ASSERT_EQ( 5, c[4]);
        TEST_ASSERT_EQ( 6, c[5]);
        TEST_ASSERT_EQ( 8, c[6]);
        TEST_ASSERT_EQ( 9, c[7]);
        TEST_ASSERT_EQ(10, c[8]);
        TEST_ASSERT_EQ(12, c[9]);
        TEST_ASSERT_EQ(14, c[10]);
    }
    {
        int err;
        int64_t asize = 11;
        int64_t a[11] = {2,0,2,2,4,8,6,10,14,12,10};
        int64_t bsize = 13;
        int64_t b[13] = {4,0,1,2,2,4,0,5,9,8,9,6,12};
        int64_t csize = 24;
        int64_t c[24] = {};
        err = setunion_unsorted_nonunique_int64(&csize, NULL, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(11, csize);
        err = setunion_unsorted_nonunique_int64(&csize, c, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(11, csize);
        TEST_ASSERT_EQ( 0, c[0]);
        TEST_ASSERT_EQ( 1, c[1]);
        TEST_ASSERT_EQ( 2, c[2]);
        TEST_ASSERT_EQ( 4, c[3]);
        TEST_ASSERT_EQ( 5, c[4]);
        TEST_ASSERT_EQ( 6, c[5]);
        TEST_ASSERT_EQ( 8, c[6]);
        TEST_ASSERT_EQ( 9, c[7]);
        TEST_ASSERT_EQ(10, c[8]);
        TEST_ASSERT_EQ(12, c[9]);
        TEST_ASSERT_EQ(14, c[10]);
    }

    /* signed integers */
    {
        int err;
        int64_t asize = 8;
        int a[8] = {10,2,14,6,8,0,4,12};
        int64_t bsize = 9;
        int b[9] = {0,1,4,2,5,6,12,9,8};
        int64_t csize = 17;
        int c[17] = {};
        err = setunion_unsorted_nonunique_int(&csize, NULL, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(11, csize);
        err = setunion_unsorted_nonunique_int(&csize, c, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(11, csize);
        TEST_ASSERT_EQ( 0, c[0]);
        TEST_ASSERT_EQ( 1, c[1]);
        TEST_ASSERT_EQ( 2, c[2]);
        TEST_ASSERT_EQ( 4, c[3]);
        TEST_ASSERT_EQ( 5, c[4]);
        TEST_ASSERT_EQ( 6, c[5]);
        TEST_ASSERT_EQ( 8, c[6]);
        TEST_ASSERT_EQ( 9, c[7]);
        TEST_ASSERT_EQ(10, c[8]);
        TEST_ASSERT_EQ(12, c[9]);
        TEST_ASSERT_EQ(14, c[10]);
    }
    {
        int err;
        int64_t asize = 11;
        int a[11] = {2,0,2,2,4,8,6,10,14,12,10};
        int64_t bsize = 13;
        int b[13] = {4,0,1,2,2,4,0,5,9,8,9,6,12};
        int64_t csize = 24;
        int c[24] = {};
        err = setunion_unsorted_nonunique_int(&csize, NULL, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(11, csize);
        err = setunion_unsorted_nonunique_int(&csize, c, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(11, csize);
        TEST_ASSERT_EQ( 0, c[0]);
        TEST_ASSERT_EQ( 1, c[1]);
        TEST_ASSERT_EQ( 2, c[2]);
        TEST_ASSERT_EQ( 4, c[3]);
        TEST_ASSERT_EQ( 5, c[4]);
        TEST_ASSERT_EQ( 6, c[5]);
        TEST_ASSERT_EQ( 8, c[6]);
        TEST_ASSERT_EQ( 9, c[7]);
        TEST_ASSERT_EQ(10, c[8]);
        TEST_ASSERT_EQ(12, c[9]);
        TEST_ASSERT_EQ(14, c[10]);
    }
    return TEST_SUCCESS;
}

/**
 * ‘test_setintersection_sorted_unique()’ tests merging two sorted
 * arrays of unique values (i.e., no duplicates), based on a set
 * intersection operation.
 */
int test_setintersection_sorted_unique(void)
{
    /* 32-bit signed integers */
    {
        int err;
        int64_t asize = 8;
        int32_t a[8] = {0,2,4,6,8,10,12,14};
        int64_t bsize = 9;
        int32_t b[9] = {0,1,2,4,5,6,8,9,12};
        int64_t csize = 17;
        int32_t c[17] = {};
        err = setintersection_sorted_unique_int32(&csize, NULL, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( 6, csize);
        err = setintersection_sorted_unique_int32(&csize, c, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( 6, csize);
        TEST_ASSERT_EQ( 0, c[0]);
        TEST_ASSERT_EQ( 2, c[1]);
        TEST_ASSERT_EQ( 4, c[2]);
        TEST_ASSERT_EQ( 6, c[3]);
        TEST_ASSERT_EQ( 8, c[4]);
        TEST_ASSERT_EQ(12, c[5]);
    }

    /* 64-bit signed integers */
    {
        int err;
        int64_t asize = 8;
        int64_t a[8] = {0,2,4,6,8,10,12,14};
        int64_t bsize = 9;
        int64_t b[9] = {0,1,2,4,5,6,8,9,12};
        int64_t csize = 17;
        int64_t c[17] = {};
        err = setintersection_sorted_unique_int64(&csize, NULL, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( 6, csize);
        err = setintersection_sorted_unique_int64(&csize, c, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( 6, csize);
        TEST_ASSERT_EQ( 0, c[0]);
        TEST_ASSERT_EQ( 2, c[1]);
        TEST_ASSERT_EQ( 4, c[2]);
        TEST_ASSERT_EQ( 6, c[3]);
        TEST_ASSERT_EQ( 8, c[4]);
        TEST_ASSERT_EQ(12, c[5]);
    }

    /* signed integers */
    {
        int err;
        int64_t asize = 8;
        int a[8] = {0,2,4,6,8,10,12,14};
        int64_t bsize = 9;
        int b[9] = {0,1,2,4,5,6,8,9,12};
        int64_t csize = 17;
        int c[17] = {};
        err = setintersection_sorted_unique_int(&csize, NULL, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( 6, csize);
        err = setintersection_sorted_unique_int(&csize, c, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( 6, csize);
        TEST_ASSERT_EQ( 0, c[0]);
        TEST_ASSERT_EQ( 2, c[1]);
        TEST_ASSERT_EQ( 4, c[2]);
        TEST_ASSERT_EQ( 6, c[3]);
        TEST_ASSERT_EQ( 8, c[4]);
        TEST_ASSERT_EQ(12, c[5]);
    }
    return TEST_SUCCESS;
}

/**
 * ‘test_setintersection_sorted_nonunique()’ tests merging two sorted
 * arrays, possibly containing non-unique (duplicate) values, based on
 * a set intersection operation.
 */
int test_setintersection_sorted_nonunique(void)
{
    /* 32-bit signed integers */
    {
        int err;
        int64_t asize = 8;
        int32_t a[8] = {0,2,4,6,8,10,12,14};
        int64_t bsize = 9;
        int32_t b[9] = {0,1,2,4,5,6,8,9,12};
        int64_t csize = 17;
        int32_t c[17] = {};
        err = setintersection_sorted_nonunique_int32(&csize, NULL, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( 6, csize);
        err = setintersection_sorted_nonunique_int32(&csize, c, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( 6, csize);
        TEST_ASSERT_EQ( 0, c[0]);
        TEST_ASSERT_EQ( 2, c[1]);
        TEST_ASSERT_EQ( 4, c[2]);
        TEST_ASSERT_EQ( 6, c[3]);
        TEST_ASSERT_EQ( 8, c[4]);
        TEST_ASSERT_EQ(12, c[5]);
    }
    {
        int err;
        int64_t asize = 11;
        int32_t a[11] = {0,2,2,2,4,6,8,10,10,12,14};
        int64_t bsize = 13;
        int32_t b[13] = {0,0,1,2,2,4,4,5,6,8,9,9,12};
        int64_t csize = 24;
        int32_t c[24] = {};
        err = setintersection_sorted_nonunique_int32(&csize, NULL, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(6, csize);
        err = setintersection_sorted_nonunique_int32(&csize, c, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(6, csize);
        TEST_ASSERT_EQ( 0, c[0]);
        TEST_ASSERT_EQ( 2, c[1]);
        TEST_ASSERT_EQ( 4, c[2]);
        TEST_ASSERT_EQ( 6, c[3]);
        TEST_ASSERT_EQ( 8, c[4]);
        TEST_ASSERT_EQ(12, c[5]);
    }

    /* 64-bit signed integers */
    {
        int err;
        int64_t asize = 8;
        int64_t a[8] = {0,2,4,6,8,10,12,14};
        int64_t bsize = 9;
        int64_t b[9] = {0,1,2,4,5,6,8,9,12};
        int64_t csize = 17;
        int64_t c[17] = {};
        err = setintersection_sorted_nonunique_int64(&csize, NULL, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( 6, csize);
        err = setintersection_sorted_nonunique_int64(&csize, c, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( 6, csize);
        TEST_ASSERT_EQ( 0, c[0]);
        TEST_ASSERT_EQ( 2, c[1]);
        TEST_ASSERT_EQ( 4, c[2]);
        TEST_ASSERT_EQ( 6, c[3]);
        TEST_ASSERT_EQ( 8, c[4]);
        TEST_ASSERT_EQ(12, c[5]);
    }
    {
        int err;
        int64_t asize = 11;
        int64_t a[11] = {0,2,2,2,4,6,8,10,10,12,14};
        int64_t bsize = 13;
        int64_t b[13] = {0,0,1,2,2,4,4,5,6,8,9,9,12};
        int64_t csize = 24;
        int64_t c[24] = {};
        err = setintersection_sorted_nonunique_int64(&csize, NULL, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(6, csize);
        err = setintersection_sorted_nonunique_int64(&csize, c, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(6, csize);
        TEST_ASSERT_EQ( 0, c[0]);
        TEST_ASSERT_EQ( 2, c[1]);
        TEST_ASSERT_EQ( 4, c[2]);
        TEST_ASSERT_EQ( 6, c[3]);
        TEST_ASSERT_EQ( 8, c[4]);
        TEST_ASSERT_EQ(12, c[5]);
    }

    /* signed integers */
    {
        int err;
        int64_t asize = 8;
        int a[8] = {0,2,4,6,8,10,12,14};
        int64_t bsize = 9;
        int b[9] = {0,1,2,4,5,6,8,9,12};
        int64_t csize = 17;
        int c[17] = {};
        err = setintersection_sorted_nonunique_int(&csize, NULL, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( 6, csize);
        err = setintersection_sorted_nonunique_int(&csize, c, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( 6, csize);
        TEST_ASSERT_EQ( 0, c[0]);
        TEST_ASSERT_EQ( 2, c[1]);
        TEST_ASSERT_EQ( 4, c[2]);
        TEST_ASSERT_EQ( 6, c[3]);
        TEST_ASSERT_EQ( 8, c[4]);
        TEST_ASSERT_EQ(12, c[5]);
    }
    {
        int err;
        int64_t asize = 11;
        int a[11] = {0,2,2,2,4,6,8,10,10,12,14};
        int64_t bsize = 13;
        int b[13] = {0,0,1,2,2,4,4,5,6,8,9,9,12};
        int64_t csize = 24;
        int c[24] = {};
        err = setintersection_sorted_nonunique_int(&csize, NULL, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(6, csize);
        err = setintersection_sorted_nonunique_int(&csize, c, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(6, csize);
        TEST_ASSERT_EQ( 0, c[0]);
        TEST_ASSERT_EQ( 2, c[1]);
        TEST_ASSERT_EQ( 4, c[2]);
        TEST_ASSERT_EQ( 6, c[3]);
        TEST_ASSERT_EQ( 8, c[4]);
        TEST_ASSERT_EQ(12, c[5]);
    }
    return TEST_SUCCESS;
}

/**
 * ‘test_setintersection_unsorted_unique()’ tests merging two unsorted
 * arrays of unique values (i.e., no duplicates), based on a set
 * intersection operation.
 */
int test_setintersection_unsorted_unique(void)
{
    /* 32-bit signed integers */
    {
        int err;
        int64_t asize = 8;
        int32_t a[8] = {10,2,14,6,8,0,4,12};
        int64_t bsize = 9;
        int32_t b[9] = {0,1,4,2,5,6,12,9,8};
        int64_t csize = 17;
        int32_t c[17] = {};
        err = setintersection_unsorted_unique_int32(&csize, NULL, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( 6, csize);
        err = setintersection_unsorted_unique_int32(&csize, c, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( 6, csize);
        TEST_ASSERT_EQ( 0, c[0]);
        TEST_ASSERT_EQ( 2, c[1]);
        TEST_ASSERT_EQ( 4, c[2]);
        TEST_ASSERT_EQ( 6, c[3]);
        TEST_ASSERT_EQ( 8, c[4]);
        TEST_ASSERT_EQ(12, c[5]);
    }

    /* 64-bit signed integers */
    {
        int err;
        int64_t asize = 8;
        int64_t a[8] = {10,2,14,6,8,0,4,12};
        int64_t bsize = 9;
        int64_t b[9] = {0,1,4,2,5,6,12,9,8};
        int64_t csize = 17;
        int64_t c[17] = {};
        err = setintersection_unsorted_unique_int64(&csize, NULL, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( 6, csize);
        err = setintersection_unsorted_unique_int64(&csize, c, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( 6, csize);
        TEST_ASSERT_EQ( 0, c[0]);
        TEST_ASSERT_EQ( 2, c[1]);
        TEST_ASSERT_EQ( 4, c[2]);
        TEST_ASSERT_EQ( 6, c[3]);
        TEST_ASSERT_EQ( 8, c[4]);
        TEST_ASSERT_EQ(12, c[5]);
    }

    /* signed integers */
    {
        int err;
        int64_t asize = 8;
        int a[8] = {10,2,14,6,8,0,4,12};
        int64_t bsize = 9;
        int b[9] = {0,1,4,2,5,6,12,9,8};
        int64_t csize = 17;
        int c[17] = {};
        err = setintersection_unsorted_unique_int(&csize, NULL, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( 6, csize);
        err = setintersection_unsorted_unique_int(&csize, c, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( 6, csize);
        TEST_ASSERT_EQ( 0, c[0]);
        TEST_ASSERT_EQ( 2, c[1]);
        TEST_ASSERT_EQ( 4, c[2]);
        TEST_ASSERT_EQ( 6, c[3]);
        TEST_ASSERT_EQ( 8, c[4]);
        TEST_ASSERT_EQ(12, c[5]);
    }
    return TEST_SUCCESS;
}

/**
 * ‘test_setintersection_unsorted_nonunique()’ tests merging two
 * unsorted arrays, possibly containing non-unique (duplicate) values,
 * based on a set intersection operation.
 */
int test_setintersection_unsorted_nonunique(void)
{
    /* 32-bit signed integers */
    {
        int err;
        int64_t asize = 8;
        int32_t a[8] = {10,2,14,6,8,0,4,12};
        int64_t bsize = 9;
        int32_t b[9] = {0,1,4,2,5,6,12,9,8};
        int64_t csize = 17;
        int32_t c[17] = {};
        err = setintersection_unsorted_nonunique_int32(&csize, NULL, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( 6, csize);
        err = setintersection_unsorted_nonunique_int32(&csize, c, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( 6, csize);
        TEST_ASSERT_EQ( 0, c[0]);
        TEST_ASSERT_EQ( 2, c[1]);
        TEST_ASSERT_EQ( 4, c[2]);
        TEST_ASSERT_EQ( 6, c[3]);
        TEST_ASSERT_EQ( 8, c[4]);
        TEST_ASSERT_EQ(12, c[5]);
    }
    {
        int err;
        int64_t asize = 11;
        int32_t a[11] = {2,0,2,2,4,8,6,10,14,12,10};
        int64_t bsize = 13;
        int32_t b[13] = {4,0,1,2,2,4,0,5,9,8,9,6,12};
        int64_t csize = 24;
        int32_t c[24] = {};
        err = setintersection_unsorted_nonunique_int32(&csize, NULL, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( 6, csize);
        err = setintersection_unsorted_nonunique_int32(&csize, c, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( 6, csize);
        TEST_ASSERT_EQ( 0, c[0]);
        TEST_ASSERT_EQ( 2, c[1]);
        TEST_ASSERT_EQ( 4, c[2]);
        TEST_ASSERT_EQ( 6, c[3]);
        TEST_ASSERT_EQ( 8, c[4]);
        TEST_ASSERT_EQ(12, c[5]);
    }

    /* 64-bit signed integers */
    {
        int err;
        int64_t asize = 8;
        int64_t a[8] = {10,2,14,6,8,0,4,12};
        int64_t bsize = 9;
        int64_t b[9] = {0,1,4,2,5,6,12,9,8};
        int64_t csize = 17;
        int64_t c[17] = {};
        err = setintersection_unsorted_nonunique_int64(&csize, NULL, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( 6, csize);
        err = setintersection_unsorted_nonunique_int64(&csize, c, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( 6, csize);
        TEST_ASSERT_EQ( 0, c[0]);
        TEST_ASSERT_EQ( 2, c[1]);
        TEST_ASSERT_EQ( 4, c[2]);
        TEST_ASSERT_EQ( 6, c[3]);
        TEST_ASSERT_EQ( 8, c[4]);
        TEST_ASSERT_EQ(12, c[5]);
    }
    {
        int err;
        int64_t asize = 11;
        int64_t a[11] = {2,0,2,2,4,8,6,10,14,12,10};
        int64_t bsize = 13;
        int64_t b[13] = {4,0,1,2,2,4,0,5,9,8,9,6,12};
        int64_t csize = 24;
        int64_t c[24] = {};
        err = setintersection_unsorted_nonunique_int64(&csize, NULL, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( 6, csize);
        err = setintersection_unsorted_nonunique_int64(&csize, c, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( 6, csize);
        TEST_ASSERT_EQ( 0, c[0]);
        TEST_ASSERT_EQ( 2, c[1]);
        TEST_ASSERT_EQ( 4, c[2]);
        TEST_ASSERT_EQ( 6, c[3]);
        TEST_ASSERT_EQ( 8, c[4]);
        TEST_ASSERT_EQ(12, c[5]);
    }

    /* signed integers */
    {
        int err;
        int64_t asize = 8;
        int a[8] = {10,2,14,6,8,0,4,12};
        int64_t bsize = 9;
        int b[9] = {0,1,4,2,5,6,12,9,8};
        int64_t csize = 17;
        int c[17] = {};
        err = setintersection_unsorted_nonunique_int(&csize, NULL, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( 6, csize);
        err = setintersection_unsorted_nonunique_int(&csize, c, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( 6, csize);
        TEST_ASSERT_EQ( 0, c[0]);
        TEST_ASSERT_EQ( 2, c[1]);
        TEST_ASSERT_EQ( 4, c[2]);
        TEST_ASSERT_EQ( 6, c[3]);
        TEST_ASSERT_EQ( 8, c[4]);
        TEST_ASSERT_EQ(12, c[5]);
    }
    {
        int err;
        int64_t asize = 11;
        int a[11] = {2,0,2,2,4,8,6,10,14,12,10};
        int64_t bsize = 13;
        int b[13] = {4,0,1,2,2,4,0,5,9,8,9,6,12};
        int64_t csize = 24;
        int c[24] = {};
        err = setintersection_unsorted_nonunique_int(&csize, NULL, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( 6, csize);
        err = setintersection_unsorted_nonunique_int(&csize, c, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( 6, csize);
        TEST_ASSERT_EQ( 0, c[0]);
        TEST_ASSERT_EQ( 2, c[1]);
        TEST_ASSERT_EQ( 4, c[2]);
        TEST_ASSERT_EQ( 6, c[3]);
        TEST_ASSERT_EQ( 8, c[4]);
        TEST_ASSERT_EQ(12, c[5]);
    }
    return TEST_SUCCESS;
}

/**
 * ‘test_setdifference_sorted_unique()’ tests merging two sorted
 * arrays of unique values (i.e., no duplicates), based on a set
 * difference operation.
 */
int test_setdifference_sorted_unique(void)
{
    /* 32-bit signed integers */
    {
        int err;
        int64_t asize = 8;
        int32_t a[8] = {0,2,4,6,8,10,12,14};
        int64_t bsize = 9;
        int32_t b[9] = {0,1,2,4,5,6,8,9,12};
        int64_t csize = 17;
        int32_t c[17] = {};
        err = setdifference_sorted_unique_int32(&csize, NULL, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( 2, csize);
        err = setdifference_sorted_unique_int32(&csize, c, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( 2, csize);
        TEST_ASSERT_EQ(10, c[0]);
        TEST_ASSERT_EQ(14, c[1]);
    }

    /* 64-bit signed integers */
    {
        int err;
        int64_t asize = 8;
        int64_t a[8] = {0,2,4,6,8,10,12,14};
        int64_t bsize = 9;
        int64_t b[9] = {0,1,2,4,5,6,8,9,12};
        int64_t csize = 17;
        int64_t c[17] = {};
        err = setdifference_sorted_unique_int64(&csize, NULL, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( 2, csize);
        err = setdifference_sorted_unique_int64(&csize, c, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( 2, csize);
        TEST_ASSERT_EQ(10, c[0]);
        TEST_ASSERT_EQ(14, c[1]);
    }

    /* signed integers */
    {
        int err;
        int64_t asize = 8;
        int a[8] = {0,2,4,6,8,10,12,14};
        int64_t bsize = 9;
        int b[9] = {0,1,2,4,5,6,8,9,12};
        int64_t csize = 17;
        int c[17] = {};
        err = setdifference_sorted_unique_int(&csize, NULL, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( 2, csize);
        err = setdifference_sorted_unique_int(&csize, c, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( 2, csize);
        TEST_ASSERT_EQ(10, c[0]);
        TEST_ASSERT_EQ(14, c[1]);
    }
    return TEST_SUCCESS;
}

/**
 * ‘test_setdifference_sorted_nonunique()’ tests merging two sorted
 * arrays, possibly containing non-unique (duplicate) values, based on
 * a set difference operation.
 */
int test_setdifference_sorted_nonunique(void)
{
    /* 32-bit signed integers */
    {
        int err;
        int64_t asize = 8;
        int32_t a[8] = {0,2,4,6,8,10,12,14};
        int64_t bsize = 9;
        int32_t b[9] = {0,1,2,4,5,6,8,9,12};
        int64_t csize = 17;
        int32_t c[17] = {};
        err = setdifference_sorted_nonunique_int32(&csize, NULL, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( 2, csize);
        err = setdifference_sorted_nonunique_int32(&csize, c, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( 2, csize);
        TEST_ASSERT_EQ(10, c[0]);
        TEST_ASSERT_EQ(14, c[1]);
    }
    {
        int err;
        int64_t asize = 11;
        int32_t a[11] = {0,2,2,2,4,6,8,10,10,12,14};
        int64_t bsize = 13;
        int32_t b[13] = {0,0,1,2,2,4,4,5,6,8,9,9,12};
        int64_t csize = 24;
        int32_t c[24] = {};
        err = setdifference_sorted_nonunique_int32(&csize, NULL, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( 2, csize);
        err = setdifference_sorted_nonunique_int32(&csize, c, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( 2, csize);
        TEST_ASSERT_EQ(10, c[0]);
        TEST_ASSERT_EQ(14, c[1]);
    }

    /* 64-bit signed integers */
    {
        int err;
        int64_t asize = 8;
        int64_t a[8] = {0,2,4,6,8,10,12,14};
        int64_t bsize = 9;
        int64_t b[9] = {0,1,2,4,5,6,8,9,12};
        int64_t csize = 17;
        int64_t c[17] = {};
        err = setdifference_sorted_nonunique_int64(&csize, NULL, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( 2, csize);
        err = setdifference_sorted_nonunique_int64(&csize, c, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( 2, csize);
        TEST_ASSERT_EQ(10, c[0]);
        TEST_ASSERT_EQ(14, c[1]);
    }
    {
        int err;
        int64_t asize = 11;
        int64_t a[11] = {0,2,2,2,4,6,8,10,10,12,14};
        int64_t bsize = 13;
        int64_t b[13] = {0,0,1,2,2,4,4,5,6,8,9,9,12};
        int64_t csize = 24;
        int64_t c[24] = {};
        err = setdifference_sorted_nonunique_int64(&csize, NULL, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( 2, csize);
        err = setdifference_sorted_nonunique_int64(&csize, c, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( 2, csize);
        TEST_ASSERT_EQ(10, c[0]);
        TEST_ASSERT_EQ(14, c[1]);
    }

    /* signed integers */
    {
        int err;
        int64_t asize = 8;
        int a[8] = {0,2,4,6,8,10,12,14};
        int64_t bsize = 9;
        int b[9] = {0,1,2,4,5,6,8,9,12};
        int64_t csize = 17;
        int c[17] = {};
        err = setdifference_sorted_nonunique_int(&csize, NULL, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( 2, csize);
        err = setdifference_sorted_nonunique_int(&csize, c, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( 2, csize);
        TEST_ASSERT_EQ(10, c[0]);
        TEST_ASSERT_EQ(14, c[1]);
    }
    {
        int err;
        int64_t asize = 11;
        int a[11] = {0,2,2,2,4,6,8,10,10,12,14};
        int64_t bsize = 13;
        int b[13] = {0,0,1,2,2,4,4,5,6,8,9,9,12};
        int64_t csize = 24;
        int c[24] = {};
        err = setdifference_sorted_nonunique_int(&csize, NULL, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( 2, csize);
        err = setdifference_sorted_nonunique_int(&csize, c, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( 2, csize);
        TEST_ASSERT_EQ(10, c[0]);
        TEST_ASSERT_EQ(14, c[1]);
    }
    return TEST_SUCCESS;
}

/**
 * ‘test_setdifference_unsorted_unique()’ tests merging two unsorted arrays
 * of unique values (i.e., no duplicates), based on a set difference
 * operation.
 */
int test_setdifference_unsorted_unique(void)
{
    /* 32-bit signed integers */
    {
        int err;
        int64_t asize = 8;
        int32_t a[8] = {10,2,14,6,8,0,4,12};
        int64_t bsize = 9;
        int32_t b[9] = {0,1,4,2,5,6,12,9,8};
        int64_t csize = 17;
        int32_t c[17] = {};
        err = setdifference_unsorted_unique_int32(&csize, NULL, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( 2, csize);
        err = setdifference_unsorted_unique_int32(&csize, c, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( 2, csize);
        TEST_ASSERT_EQ(10, c[0]);
        TEST_ASSERT_EQ(14, c[1]);
    }

    /* 64-bit signed integers */
    {
        int err;
        int64_t asize = 8;
        int64_t a[8] = {10,2,14,6,8,0,4,12};
        int64_t bsize = 9;
        int64_t b[9] = {0,1,4,2,5,6,12,9,8};
        int64_t csize = 17;
        int64_t c[17] = {};
        err = setdifference_unsorted_unique_int64(&csize, NULL, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( 2, csize);
        err = setdifference_unsorted_unique_int64(&csize, c, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( 2, csize);
        TEST_ASSERT_EQ(10, c[0]);
        TEST_ASSERT_EQ(14, c[1]);
    }

    /* signed integers */
    {
        int err;
        int64_t asize = 8;
        int a[8] = {10,2,14,6,8,0,4,12};
        int64_t bsize = 9;
        int b[9] = {0,1,4,2,5,6,12,9,8};
        int64_t csize = 17;
        int c[17] = {};
        err = setdifference_unsorted_unique_int(&csize, NULL, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( 2, csize);
        err = setdifference_unsorted_unique_int(&csize, c, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( 2, csize);
        TEST_ASSERT_EQ(10, c[0]);
        TEST_ASSERT_EQ(14, c[1]);
    }
    return TEST_SUCCESS;
}

/**
 * ‘test_setdifference_unsorted_nonunique()’ tests merging two unsorted
 * arrays, possibly containing non-unique (duplicate) values, based on
 * a set difference operation.
 */
int test_setdifference_unsorted_nonunique(void)
{
    /* 32-bit signed integers */
    {
        int err;
        int64_t asize = 8;
        int32_t a[8] = {10,2,14,6,8,0,4,12};
        int64_t bsize = 9;
        int32_t b[9] = {0,1,4,2,5,6,12,9,8};
        int64_t csize = 17;
        int32_t c[17] = {};
        err = setdifference_unsorted_nonunique_int32(&csize, NULL, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( 2, csize);
        err = setdifference_unsorted_nonunique_int32(&csize, c, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( 2, csize);
        TEST_ASSERT_EQ(10, c[0]);
        TEST_ASSERT_EQ(14, c[1]);
    }
    {
        int err;
        int64_t asize = 11;
        int32_t a[11] = {2,0,2,2,4,8,6,10,14,12,10};
        int64_t bsize = 13;
        int32_t b[13] = {4,0,1,2,2,4,0,5,9,8,9,6,12};
        int64_t csize = 24;
        int32_t c[24] = {};
        err = setdifference_unsorted_nonunique_int32(&csize, NULL, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( 2, csize);
        err = setdifference_unsorted_nonunique_int32(&csize, c, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( 2, csize);
        TEST_ASSERT_EQ(10, c[0]);
        TEST_ASSERT_EQ(14, c[1]);
    }

    /* 64-bit signed integers */
    {
        int err;
        int64_t asize = 8;
        int64_t a[8] = {10,2,14,6,8,0,4,12};
        int64_t bsize = 9;
        int64_t b[9] = {0,1,4,2,5,6,12,9,8};
        int64_t csize = 17;
        int64_t c[17] = {};
        err = setdifference_unsorted_nonunique_int64(&csize, NULL, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( 2, csize);
        err = setdifference_unsorted_nonunique_int64(&csize, c, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( 2, csize);
        TEST_ASSERT_EQ(10, c[0]);
        TEST_ASSERT_EQ(14, c[1]);
    }
    {
        int err;
        int64_t asize = 11;
        int64_t a[11] = {2,0,2,2,4,8,6,10,14,12,10};
        int64_t bsize = 13;
        int64_t b[13] = {4,0,1,2,2,4,0,5,9,8,9,6,12};
        int64_t csize = 24;
        int64_t c[24] = {};
        err = setdifference_unsorted_nonunique_int64(&csize, NULL, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( 2, csize);
        err = setdifference_unsorted_nonunique_int64(&csize, c, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( 2, csize);
        TEST_ASSERT_EQ(10, c[0]);
        TEST_ASSERT_EQ(14, c[1]);
    }

    /* signed integers */
    {
        int err;
        int64_t asize = 8;
        int a[8] = {10,2,14,6,8,0,4,12};
        int64_t bsize = 9;
        int b[9] = {0,1,4,2,5,6,12,9,8};
        int64_t csize = 17;
        int c[17] = {};
        err = setdifference_unsorted_nonunique_int(&csize, NULL, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( 2, csize);
        err = setdifference_unsorted_nonunique_int(&csize, c, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( 2, csize);
        TEST_ASSERT_EQ(10, c[0]);
        TEST_ASSERT_EQ(14, c[1]);
    }
    {
        int err;
        int64_t asize = 11;
        int a[11] = {2,0,2,2,4,8,6,10,14,12,10};
        int64_t bsize = 13;
        int b[13] = {4,0,1,2,2,4,0,5,9,8,9,6,12};
        int64_t csize = 24;
        int c[24] = {};
        err = setdifference_unsorted_nonunique_int(&csize, NULL, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( 2, csize);
        err = setdifference_unsorted_nonunique_int(&csize, c, asize, a, bsize, b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( 2, csize);
        TEST_ASSERT_EQ(10, c[0]);
        TEST_ASSERT_EQ(14, c[1]);
    }
    return TEST_SUCCESS;
}

/**
 * ‘main()’ entry point and test driver.
 */
int main(int argc, char * argv[])
{
    TEST_SUITE_BEGIN("Running tests for merging functions\n");
    TEST_RUN(test_merge_sorted);
    TEST_RUN(test_setunion_sorted_unique);
    TEST_RUN(test_setunion_sorted_nonunique);
    TEST_RUN(test_setunion_unsorted_unique);
    TEST_RUN(test_setunion_unsorted_nonunique);
    TEST_RUN(test_setintersection_sorted_unique);
    TEST_RUN(test_setintersection_sorted_nonunique);
    TEST_RUN(test_setintersection_unsorted_unique);
    TEST_RUN(test_setintersection_unsorted_nonunique);
    TEST_RUN(test_setdifference_sorted_unique);
    TEST_RUN(test_setdifference_sorted_nonunique);
    TEST_RUN(test_setdifference_unsorted_unique);
    TEST_RUN(test_setdifference_unsorted_nonunique);
    TEST_SUITE_END();
    return (TEST_SUITE_STATUS == TEST_SUCCESS) ?
        EXIT_SUCCESS : EXIT_FAILURE;
}
