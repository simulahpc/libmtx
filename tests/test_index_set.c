/* This file is part of libmtx.
 *
 * Copyright (C) 2021 James D. Trotter
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
 * Last modified: 2021-08-09
 *
 * Unit tests for index sets.
 */

#include "test.h"

#include <libmtx/error.h>
#include <libmtx/util/index_set.h>

#include <stdbool.h>
#include <stdlib.h>

/**
 * `test_index_set_interval()` tests index sets of contiguous integers
 * from a half-open interval.
 */
int test_index_set_interval(void)
{
    struct mtxidxset index_set;
    TEST_ASSERT_EQ(MTX_SUCCESS, mtxidxset_init_interval(&index_set, 0, 4));
    int size;
    TEST_ASSERT_EQ(4, index_set.size);
    TEST_ASSERT(mtxidxset_contains(&index_set, 0));
    TEST_ASSERT(mtxidxset_contains(&index_set, 1));
    TEST_ASSERT(mtxidxset_contains(&index_set, 2));
    TEST_ASSERT(mtxidxset_contains(&index_set, 3));
    TEST_ASSERT_FALSE(mtxidxset_contains(&index_set, 4));
    TEST_ASSERT_FALSE(mtxidxset_contains(&index_set, -1));
    mtxidxset_free(&index_set);
    return TEST_SUCCESS;
}

/**
 * `main()' entry point and test driver.
 */
int main(int argc, char * argv[])
{
    TEST_SUITE_BEGIN("Running tests for index sets\n");
    TEST_RUN(test_index_set_interval);
    TEST_SUITE_END();
    return (TEST_SUITE_STATUS == TEST_SUCCESS) ?
        EXIT_SUCCESS : EXIT_FAILURE;
}
