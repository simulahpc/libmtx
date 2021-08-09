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
 * Unit tests for string parsing functions.
 */

#include "test.h"

#include "libmtx/util/parse.h"

#include <errno.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * `test_parse_int32()` tests parsing strings of 32-bit integers.
 */
int test_parse_int32(void)
{
    {
        int32_t number;
        const char * s = "";
        TEST_ASSERT_EQ(EINVAL, parse_int32(s, NULL, &number, NULL));
    }

    {
        int32_t number;
        const char * s = "0";
        const char * s_end;
        TEST_ASSERT_EQ(0, parse_int32(s, NULL, &number, &s_end));
        TEST_ASSERT_EQ(0, number);
        TEST_ASSERT_EQ(s_end, s+1);
    }

    {
        int32_t number;
        const char * s = "1";
        const char * s_end;
        TEST_ASSERT_EQ(0, parse_int32(s, NULL, &number, &s_end));
        TEST_ASSERT_EQ(1, number);
        TEST_ASSERT_EQ(s_end, s+1);
    }

    {
        int32_t number;
        const char * s = "1";
        TEST_ASSERT_EQ(0, parse_int32(s, NULL, &number, NULL));
        TEST_ASSERT_EQ(1, number);
    }

    {
        int32_t number;
        const char * s = "-1";
        TEST_ASSERT_EQ(0, parse_int32(s, NULL, &number, NULL));
        TEST_ASSERT_EQ(-1, number);
    }

    {
        int32_t number;
        const char * s = "42";
        const char * s_end;
        TEST_ASSERT_EQ(0, parse_int32(s, NULL, &number, &s_end));
        TEST_ASSERT_EQ(42, number);
        TEST_ASSERT_EQ(s_end, s+2);
    }

    {
        /* Parse INT32_MAX, which is 2^31-1. */
        int32_t number;
        const char * s = "2147483647";
        const char * s_end;
        TEST_ASSERT_EQ(0, parse_int32(s, NULL, &number, &s_end));
        TEST_ASSERT_EQ(INT32_MAX, number);
        TEST_ASSERT_EQ(s_end, s+10);
    }

    {
        /* Parse INT32_MIN, which is -2^31-1. */
        int32_t number;
        const char * s = "-2147483648";
        const char * s_end;
        TEST_ASSERT_EQ(0, parse_int32(s, NULL, &number, &s_end));
        TEST_ASSERT_EQ(INT32_MIN, number);
        TEST_ASSERT_EQ(s_end, s+11);
    }

    {
        /* Parse a number larger than INT32_MAX. */
        int32_t number;
        const char * s = "2147483648";
        TEST_ASSERT_EQ(ERANGE, parse_int32(s, NULL, &number, NULL));
    }

    {
        /* Parse a number smaller than INT32_MIN. */
        int32_t number;
        const char * s = "-2147483649";
        TEST_ASSERT_EQ(ERANGE, parse_int32(s, NULL, &number, NULL));
    }

    /* Parse strings containing numbers that may end with a delimiter,
     * such as a comma. */
    {
        int32_t number;
        const char * s = "42";
        const char * end_chars = "";
        const char * s_end;
        TEST_ASSERT_EQ(0, parse_int32(s, end_chars, &number, &s_end));
        TEST_ASSERT_EQ(42, number);
        TEST_ASSERT_EQ(s_end, s+2);
    }

    {
        int32_t number;
        const char * s = "42,";
        const char * end_chars = "";
        const char * s_end;
        TEST_ASSERT_EQ(EINVAL, parse_int32(s, end_chars, &number, &s_end));
    }

    {
        int32_t number;
        const char * s = "42,";
        const char * end_chars = ",";
        const char * s_end;
        TEST_ASSERT_EQ(0, parse_int32(s, end_chars, &number, &s_end));
        TEST_ASSERT_EQ(42, number);
        TEST_ASSERT_EQ(s_end, s+3);
    }

    {
        int32_t number;
        const char * s = "42;";
        const char * end_chars = ",";
        const char * s_end;
        TEST_ASSERT_EQ(EINVAL, parse_int32(s, end_chars, &number, &s_end));
    }

    return TEST_SUCCESS;
}

/**
 * `test_parse_int64()` tests parsing strings of 64-bit integers.
 */
int test_parse_int64()
{
    {
        int64_t number;
        const char * s = "";
        TEST_ASSERT_EQ(EINVAL, parse_int64(s, NULL, &number, NULL));
    }

    {
        int64_t number;
        const char * s = "0";
        const char * s_end;
        TEST_ASSERT_EQ(0, parse_int64(s, NULL, &number, &s_end));
        TEST_ASSERT_EQ(0, number);
        TEST_ASSERT_EQ(s_end, s+1);
    }

    {
        int64_t number;
        const char * s = "1";
        const char * s_end;
        TEST_ASSERT_EQ(0, parse_int64(s, NULL, &number, &s_end));
        TEST_ASSERT_EQ(1, number);
        TEST_ASSERT_EQ(s_end, s+1);
    }

    {
        int64_t number;
        const char * s = "1";
        TEST_ASSERT_EQ(0, parse_int64(s, NULL, &number, NULL));
        TEST_ASSERT_EQ(1, number);
    }

    {
        int64_t number;
        const char * s = "-1";
        TEST_ASSERT_EQ(0, parse_int64(s, NULL, &number, NULL));
        TEST_ASSERT_EQ(-1, number);
    }

    {
        int64_t number;
        const char * s = "42";
        const char * s_end;
        TEST_ASSERT_EQ(0, parse_int64(s, NULL, &number, &s_end));
        TEST_ASSERT_EQ(42, number);
        TEST_ASSERT_EQ(s_end, s+2);
    }

    /* Test 32-bit integer range. */

    {
        /* Parse INT32_MAX, which is 2^31-1. */
        int64_t number;
        const char * s = "2147483647";
        const char * s_end;
        TEST_ASSERT_EQ(0, parse_int64(s, NULL, &number, &s_end));
        TEST_ASSERT_EQ(INT32_MAX, number);
        TEST_ASSERT_EQ(s_end, s+10);
    }

    {
        /* Parse INT32_MIN, which is -2^31-1. */
        int64_t number;
        const char * s = "-2147483648";
        const char * s_end;
        TEST_ASSERT_EQ(0, parse_int64(s, NULL, &number, &s_end));
        TEST_ASSERT_EQ(INT32_MIN, number);
        TEST_ASSERT_EQ(s_end, s+11);
    }

    {
        /* Parse a number larger than INT32_MAX. */
        int64_t number;
        const char * s = "2147483648";
        const char * s_end;
        TEST_ASSERT_EQ(0, parse_int64(s, NULL, &number, &s_end));
        TEST_ASSERT_EQ(2147483648LL, number);
        TEST_ASSERT_EQ(s_end, s+10);
    }

    {
        /* Parse a number smaller than INT32_MIN. */
        int64_t number;
        const char * s = "-2147483649";
        const char * s_end;
        TEST_ASSERT_EQ(0, parse_int64(s, NULL, &number, &s_end));
        TEST_ASSERT_EQ(-2147483649LL, number);
        TEST_ASSERT_EQ(s_end, s+11);
    }

    /* Test 64-bit integer range. */

    {
        /* Parse INT64_MAX, which is 2^63-1. */
        int64_t number;
        const char * s = "9223372036854775807";
        const char * s_end;
        TEST_ASSERT_EQ(0, parse_int64(s, NULL, &number, &s_end));
        TEST_ASSERT_EQ(INT64_MAX, number);
        TEST_ASSERT_EQ(s_end, s+19);
    }

    {
        /* Parse INT64_MIN, which is -2^63-1. */
        int64_t number;
        const char * s = "-9223372036854775808";
        const char * s_end;
        TEST_ASSERT_EQ(0, parse_int64(s, NULL, &number, &s_end));
        TEST_ASSERT_EQ(INT64_MIN, number);
        TEST_ASSERT_EQ(s_end, s+20);
    }

    {
        /* Parse a number larger than INT64_MAX. */
        int64_t number;
        const char * s = "9223372036854775808";
        TEST_ASSERT_EQ(ERANGE, parse_int64(s, NULL, &number, NULL));
    }

    {
        /* Parse a number smaller than INT64_MIN. */
        int64_t number;
        const char * s = "-9223372036854775809";
        TEST_ASSERT_EQ(ERANGE, parse_int64(s, NULL, &number, NULL));
    }

    /* Test parsing strings that contain numbers which may end with a
     * delimiter, such as a comma. */

    {
        int64_t number;
        const char * s = "42";
        const char * end_chars = "";
        const char * s_end;
        TEST_ASSERT_EQ(0, parse_int64(s, end_chars, &number, &s_end));
        TEST_ASSERT_EQ(42, number);
        TEST_ASSERT_EQ(s_end, s+2);
    }

    {
        int64_t number;
        const char * s = "42,";
        const char * end_chars = "";
        const char * s_end;
        TEST_ASSERT_EQ(EINVAL, parse_int64(s, end_chars, &number, &s_end));
    }

    {
        int64_t number;
        const char * s = "42,";
        const char * end_chars = ",";
        const char * s_end;
        TEST_ASSERT_EQ(0, parse_int64(s, end_chars, &number, &s_end));
        TEST_ASSERT_EQ(42, number);
        TEST_ASSERT_EQ(s_end, s+3);
    }

    {
        int64_t number;
        const char * s = "42;";
        const char * end_chars = ",";
        const char * s_end;
        TEST_ASSERT_EQ(EINVAL, parse_int64(s, end_chars, &number, &s_end));
    }

    return TEST_SUCCESS;
}

/**
 * `main()' entry point and test driver.
 */
int main(int argc, char * argv[])
{
    TEST_SUITE_BEGIN("Running tests for string parsing functions\n");
    TEST_RUN(test_parse_int32);
    TEST_RUN(test_parse_int64);
    TEST_SUITE_END();
    return (TEST_SUITE_STATUS == TEST_SUCCESS) ?
        EXIT_SUCCESS : EXIT_FAILURE;
}
