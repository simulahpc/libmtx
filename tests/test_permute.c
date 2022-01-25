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
 * Last modified: 2022-01-25
 *
 * Unit tests for permutations.
 */

#include <libmtx/error.h>

#include "libmtx/util/permute.h"
#include "test.h"

#include <errno.h>

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * ‘test_mtxpermutation()’ tests permuting arrays.
 */
int test_mtxpermutation(void)
{
    int err;
    {
        int values[] = {0,255,30,1,2};
        int64_t size = sizeof(values) / sizeof(*values);
        int64_t perm[] = {0,1,2,3,4};
        struct mtxpermutation permutation;
        err = mtxpermutation_init(&permutation, size, perm);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxpermutation_permute_int(&permutation, size, values);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(  0, values[0]);
        TEST_ASSERT_EQ(255, values[1]);
        TEST_ASSERT_EQ( 30, values[2]);
        TEST_ASSERT_EQ(  1, values[3]);
        TEST_ASSERT_EQ(  2, values[4]);
        mtxpermutation_free(&permutation);
    }
    {
        int values[] = {0,255,30,1,2};
        int64_t size = sizeof(values) / sizeof(*values);
        int64_t perm[] = {1,0,2,3,4};
        struct mtxpermutation permutation;
        err = mtxpermutation_init(&permutation, size, perm);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxpermutation_permute_int(&permutation, size, values);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(255, values[0]);
        TEST_ASSERT_EQ(  0, values[1]);
        TEST_ASSERT_EQ( 30, values[2]);
        TEST_ASSERT_EQ(  1, values[3]);
        TEST_ASSERT_EQ(  2, values[4]);
        mtxpermutation_free(&permutation);
    }
    {
        int values[] = {0,255,30,1,2};
        int64_t size = sizeof(values) / sizeof(*values);
        int64_t perm[] = {1,0,3,4,2};
        struct mtxpermutation permutation;
        err = mtxpermutation_init(&permutation, size, perm);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxpermutation_permute_int(&permutation, size, values);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(255, values[0]);
        TEST_ASSERT_EQ(  0, values[1]);
        TEST_ASSERT_EQ(  2, values[2]);
        TEST_ASSERT_EQ( 30, values[3]);
        TEST_ASSERT_EQ(  1, values[4]);
        mtxpermutation_free(&permutation);
    }
    return TEST_SUCCESS;
}

/**
 * ‘test_mtxpermutation_invert()’ tests inverting permutations.
 */
int test_mtxpermutation_invert(void)
{
    int err;
    {
        int values[] = {0,255,30,1,2};
        int64_t size = sizeof(values) / sizeof(*values);
        int64_t perm[] = {0,1,2,3,4};
        struct mtxpermutation permutation;
        err = mtxpermutation_init(&permutation, size, perm);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxpermutation_invert(&permutation);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxpermutation_permute_int(&permutation, size, values);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(  0, values[0]);
        TEST_ASSERT_EQ(255, values[1]);
        TEST_ASSERT_EQ( 30, values[2]);
        TEST_ASSERT_EQ(  1, values[3]);
        TEST_ASSERT_EQ(  2, values[4]);
        mtxpermutation_free(&permutation);
    }
    {
        int values[] = {0,255,30,1,2};
        int64_t size = sizeof(values) / sizeof(*values);
        int64_t perm[] = {1,0,2,3,4};
        struct mtxpermutation permutation;
        err = mtxpermutation_init(&permutation, size, perm);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxpermutation_invert(&permutation);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxpermutation_permute_int(&permutation, size, values);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(255, values[0]);
        TEST_ASSERT_EQ(  0, values[1]);
        TEST_ASSERT_EQ( 30, values[2]);
        TEST_ASSERT_EQ(  1, values[3]);
        TEST_ASSERT_EQ(  2, values[4]);
        mtxpermutation_free(&permutation);
    }
    {
        int values[] = {0,255,30,1,2};
        int64_t size = sizeof(values) / sizeof(*values);
        int64_t perm[] = {1,2,3,4,0};
        struct mtxpermutation permutation;
        err = mtxpermutation_init(&permutation, size, perm);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxpermutation_invert(&permutation);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxpermutation_permute_int(&permutation, size, values);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(255, values[0]);
        TEST_ASSERT_EQ( 30, values[1]);
        TEST_ASSERT_EQ(  1, values[2]);
        TEST_ASSERT_EQ(  2, values[3]);
        TEST_ASSERT_EQ(  0, values[4]);
        mtxpermutation_free(&permutation);
    }
    return TEST_SUCCESS;
}

/**
 * ‘test_mtxpermutation_compose()’ tests composing permutations.
 */
int test_mtxpermutation_compose(void)
{
    int err;
    {
        int values[] = {0,255,30,1,2};
        int64_t size = sizeof(values) / sizeof(*values);
        int64_t aperm[] = {0,1,2,3,4};
        struct mtxpermutation a;
        err = mtxpermutation_init(&a, size, aperm);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        int64_t bperm[] = {0,1,2,3,4};
        struct mtxpermutation b;
        err = mtxpermutation_init(&b, size, bperm);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxpermutation c;
        err = mtxpermutation_compose(&c, &a, &b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxpermutation_permute_int(&c, size, values);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(  0, values[0]);
        TEST_ASSERT_EQ(255, values[1]);
        TEST_ASSERT_EQ( 30, values[2]);
        TEST_ASSERT_EQ(  1, values[3]);
        TEST_ASSERT_EQ(  2, values[4]);
        mtxpermutation_free(&c);
        mtxpermutation_free(&b);
        mtxpermutation_free(&a);
    }
    {
        int values[] = {0,255,30,1,2};
        int64_t size = sizeof(values) / sizeof(*values);
        int64_t aperm[] = {1,0,2,3,4};
        struct mtxpermutation a;
        err = mtxpermutation_init(&a, size, aperm);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        int64_t bperm[] = {4,3,2,1,0};
        struct mtxpermutation b;
        err = mtxpermutation_init(&b, size, bperm);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxpermutation c;
        err = mtxpermutation_compose(&c, &a, &b);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxpermutation_permute_int(&c, size, values);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(  2, values[0]);
        TEST_ASSERT_EQ(  1, values[1]);
        TEST_ASSERT_EQ( 30, values[2]);
        TEST_ASSERT_EQ(  0, values[3]);
        TEST_ASSERT_EQ(255, values[4]);
        mtxpermutation_free(&c);
        mtxpermutation_free(&b);
        mtxpermutation_free(&a);
    }
    return TEST_SUCCESS;
}

/**
 * ‘test_permute()’ tests permuting arrays.
 */
int test_permute(void)
{
    int err;
    {
        int values[] = {0,255,30,1,2};
        int64_t size = sizeof(values) / sizeof(*values);
        int64_t perm[] = {0,1,2,3,4};
        err = permute_int(size, perm, values);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(  0, values[0]);
        TEST_ASSERT_EQ(255, values[1]);
        TEST_ASSERT_EQ( 30, values[2]);
        TEST_ASSERT_EQ(  1, values[3]);
        TEST_ASSERT_EQ(  2, values[4]);
    }
    {
        int values[] = {0,255,30,1,2};
        int64_t size = sizeof(values) / sizeof(*values);
        int64_t perm[] = {1,0,2,3,4};
        err = permute_int(size, perm, values);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(255, values[0]);
        TEST_ASSERT_EQ(  0, values[1]);
        TEST_ASSERT_EQ( 30, values[2]);
        TEST_ASSERT_EQ(  1, values[3]);
        TEST_ASSERT_EQ(  2, values[4]);
    }
    {
        int values[] = {0,255,30,1,2};
        int64_t size = sizeof(values) / sizeof(*values);
        int64_t perm[] = {1,0,3,4,2};
        err = permute_int(size, perm, values);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(255, values[0]);
        TEST_ASSERT_EQ(  0, values[1]);
        TEST_ASSERT_EQ(  2, values[2]);
        TEST_ASSERT_EQ( 30, values[3]);
        TEST_ASSERT_EQ(  1, values[4]);
    }
    return TEST_SUCCESS;
}

/**
 * ‘main()’ entry point and test driver.
 */
int main(int argc, char * argv[])
{
    TEST_SUITE_BEGIN("Running tests for permutations\n");
    TEST_RUN(test_mtxpermutation);
    TEST_RUN(test_mtxpermutation_invert);
    TEST_RUN(test_mtxpermutation_compose);
    TEST_RUN(test_permute);
    TEST_SUITE_END();
    return (TEST_SUITE_STATUS == TEST_SUCCESS) ?
        EXIT_SUCCESS : EXIT_FAILURE;
}
