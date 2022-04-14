/* This file is part of Libmtx.
 *
 * Copyright (C) 2022 James D. Trotter
 *
 * Libmtx is free software: you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * Libmtx is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Libmtx.  If not, see
 * <https://www.gnu.org/licenses/>.
 *
 * Authors: James D. Trotter <james@simula.no>
 * Last modified: 2022-04-14
 *
 * Unit tests for string parsing functions.
 */

#include "test.h"

#include <libmtx/error.h>

#include "libmtx/util/parse.h"

#include <errno.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * ‘test_parse_int32()’ tests parsing strings of 32-bit integers.
 */
int test_parse_int32(void)
{
    {
        int32_t x;
        const char * s = "";
        TEST_ASSERT_EQ(MTX_SUCCESS, parse_int32(&x, s, NULL, NULL));
    }
    {
        int32_t x;
        const char * s = "0";
        char * endptr;
        int64_t bytes_read = 0;
        TEST_ASSERT_EQ(MTX_SUCCESS, parse_int32(&x, s, &endptr, &bytes_read));
        TEST_ASSERT_EQ(0, x);
        TEST_ASSERT_EQ(endptr, s+1);
        TEST_ASSERT_EQ(bytes_read, 1);
    }
    {
        int32_t x;
        const char * s = "1";
        char * endptr;
        int64_t bytes_read = 0;
        TEST_ASSERT_EQ(MTX_SUCCESS, parse_int32(&x, s, &endptr, &bytes_read));
        TEST_ASSERT_EQ(1, x);
        TEST_ASSERT_EQ(endptr, s+1);
        TEST_ASSERT_EQ(bytes_read, 1);
    }
    {
        int32_t x;
        const char * s = "1";
        TEST_ASSERT_EQ(MTX_SUCCESS, parse_int32(&x, s, NULL, NULL));
        TEST_ASSERT_EQ(1, x);
    }
    {
        int32_t x;
        const char * s = "-1";
        TEST_ASSERT_EQ(MTX_SUCCESS, parse_int32(&x, s, NULL, NULL));
        TEST_ASSERT_EQ(-1, x);
    }
    {
        int32_t x;
        const char * s = "42";
        char * endptr;
        int64_t bytes_read = 0;
        TEST_ASSERT_EQ(MTX_SUCCESS, parse_int32(&x, s, &endptr, &bytes_read));
        TEST_ASSERT_EQ(42, x);
        TEST_ASSERT_EQ(endptr, s+2);
        TEST_ASSERT_EQ(bytes_read, 2);
    }
    {
        /* Parse INT32_MAX, which is 2^31-1. */
        int32_t x;
        const char * s = "2147483647";
        char * endptr;
        int64_t bytes_read = 0;
        TEST_ASSERT_EQ(MTX_SUCCESS, parse_int32(&x, s, &endptr, &bytes_read));
        TEST_ASSERT_EQ(INT32_MAX, x);
        TEST_ASSERT_EQ(endptr, s+10);
        TEST_ASSERT_EQ(bytes_read, 10);
    }
    {
        /* Parse INT32_MIN, which is -2^31-1. */
        int32_t x;
        const char * s = "-2147483648";
        char * endptr;
        int64_t bytes_read = 0;
        TEST_ASSERT_EQ(MTX_SUCCESS, parse_int32(&x, s, &endptr, &bytes_read));
        TEST_ASSERT_EQ(INT32_MIN, x);
        TEST_ASSERT_EQ(endptr, s+11);
        TEST_ASSERT_EQ(bytes_read, 11);
    }
    {
        /* Parse a number larger than INT32_MAX. */
        int32_t x;
        const char * s = "2147483648";
        TEST_ASSERT_EQ(MTX_ERR_ERRNO, parse_int32(&x, s, NULL, NULL));
        TEST_ASSERT_EQ(ERANGE, errno);
    }
    {
        /* Parse a number smaller than INT32_MIN. */
        int32_t x;
        const char * s = "-2147483649";
        TEST_ASSERT_EQ(MTX_ERR_ERRNO, parse_int32(&x, s, NULL, NULL));
        TEST_ASSERT_EQ(ERANGE, errno);
    }
    {
        /* Test parsing strings of numbers ending with a delimiter,
         * such as a comma. */
        int32_t x;
        const char * s = "42,";
        char * endptr;
        int64_t bytes_read = 0;
        TEST_ASSERT_EQ(MTX_SUCCESS, parse_int32(&x, s, &endptr, &bytes_read));
        TEST_ASSERT_EQ(42, x);
        TEST_ASSERT_EQ(endptr, s+2);
        TEST_ASSERT_EQ(bytes_read, 2);
    }
    return TEST_SUCCESS;
}

/**
 * ‘test_parse_int64()’ tests parsing strings of 64-bit integers.
 */
int test_parse_int64()
{
    {
        int64_t x;
        const char * s = "";
        TEST_ASSERT_EQ(MTX_SUCCESS, parse_int64(&x, s, NULL, NULL));
    }
    {
        int64_t x;
        const char * s = "0";
        char * endptr;
        int64_t bytes_read = 0;
        TEST_ASSERT_EQ(MTX_SUCCESS, parse_int64(&x, s, &endptr, &bytes_read));
        TEST_ASSERT_EQ(0, x);
        TEST_ASSERT_EQ(endptr, s+1);
        TEST_ASSERT_EQ(bytes_read, 1);
    }
    {
        int64_t x;
        const char * s = "1";
        char * endptr;
        int64_t bytes_read = 0;
        TEST_ASSERT_EQ(MTX_SUCCESS, parse_int64(&x, s, &endptr, &bytes_read));
        TEST_ASSERT_EQ(1, x);
        TEST_ASSERT_EQ(endptr, s+1);
        TEST_ASSERT_EQ(bytes_read, 1);
    }
    {
        int64_t x;
        const char * s = "1";
        TEST_ASSERT_EQ(MTX_SUCCESS, parse_int64(&x, s, NULL, NULL));
        TEST_ASSERT_EQ(1, x);
    }
    {
        int64_t x;
        const char * s = "-1";
        TEST_ASSERT_EQ(MTX_SUCCESS, parse_int64(&x, s, NULL, NULL));
        TEST_ASSERT_EQ(-1, x);
    }
    {
        int64_t x;
        const char * s = "42";
        char * endptr;
        int64_t bytes_read = 0;
        TEST_ASSERT_EQ(MTX_SUCCESS, parse_int64(&x, s, &endptr, &bytes_read));
        TEST_ASSERT_EQ(42, x);
        TEST_ASSERT_EQ(endptr, s+2);
        TEST_ASSERT_EQ(bytes_read, 2);
    }
    {
        /* Parse INT32_MAX, which is 2^31-1. */
        int64_t x;
        const char * s = "2147483647";
        char * endptr;
        int64_t bytes_read = 0;
        TEST_ASSERT_EQ(MTX_SUCCESS, parse_int64(&x, s, &endptr, &bytes_read));
        TEST_ASSERT_EQ(INT32_MAX, x);
        TEST_ASSERT_EQ(endptr, s+10);
        TEST_ASSERT_EQ(bytes_read, 10);
    }
    {
        /* Parse INT32_MIN, which is -2^31-1. */
        int64_t x;
        const char * s = "-2147483648";
        char * endptr;
        int64_t bytes_read = 0;
        TEST_ASSERT_EQ(MTX_SUCCESS, parse_int64(&x, s, &endptr, &bytes_read));
        TEST_ASSERT_EQ(INT32_MIN, x);
        TEST_ASSERT_EQ(endptr, s+11);
        TEST_ASSERT_EQ(bytes_read, 11);
    }
    {
        /* Parse a number larger than INT32_MAX. */
        int64_t x;
        const char * s = "2147483648";
        char * endptr;
        int64_t bytes_read = 0;
        TEST_ASSERT_EQ(MTX_SUCCESS, parse_int64(&x, s, &endptr, &bytes_read));
        TEST_ASSERT_EQ(2147483648LL, x);
        TEST_ASSERT_EQ(endptr, s+10);
        TEST_ASSERT_EQ(bytes_read, 10);
    }
    {
        /* Parse a number smaller than INT32_MIN. */
        int64_t x;
        const char * s = "-2147483649";
        char * endptr;
        int64_t bytes_read = 0;
        TEST_ASSERT_EQ(MTX_SUCCESS, parse_int64(&x, s, &endptr, &bytes_read));
        TEST_ASSERT_EQ(-2147483649LL, x);
        TEST_ASSERT_EQ(endptr, s+11);
        TEST_ASSERT_EQ(bytes_read, 11);
    }
    {
        /* Parse INT64_MAX, which is 2^63-1. */
        int64_t x;
        const char * s = "9223372036854775807";
        char * endptr;
        int64_t bytes_read = 0;
        TEST_ASSERT_EQ(MTX_SUCCESS, parse_int64(&x, s, &endptr, &bytes_read));
        TEST_ASSERT_EQ(INT64_MAX, x);
        TEST_ASSERT_EQ(endptr, s+19);
        TEST_ASSERT_EQ(bytes_read, 19);
    }
    {
        /* Parse INT64_MIN, which is -2^63-1. */
        int64_t x;
        const char * s = "-9223372036854775808";
        char * endptr;
        int64_t bytes_read = 0;
        TEST_ASSERT_EQ(MTX_SUCCESS, parse_int64(&x, s, &endptr, &bytes_read));
        TEST_ASSERT_EQ(INT64_MIN, x);
        TEST_ASSERT_EQ(endptr, s+20);
        TEST_ASSERT_EQ(bytes_read, 20);
    }
    {
        /* Parse a number larger than INT64_MAX. */
        int64_t x;
        const char * s = "9223372036854775808";
        TEST_ASSERT_EQ(MTX_ERR_ERRNO, parse_int64(&x, s, NULL, NULL));
        TEST_ASSERT_EQ(ERANGE, errno);
    }
    {
        /* Parse a number smaller than INT64_MIN. */
        int64_t x;
        const char * s = "-9223372036854775809";
        TEST_ASSERT_EQ(MTX_ERR_ERRNO, parse_int64(&x, s, NULL, NULL));
        TEST_ASSERT_EQ(ERANGE, errno);
    }
    {
        /* Test parsing strings of numbers ending with a delimiter,
         * such as a comma. */
        int64_t x;
        const char * s = "42,";
        char * endptr;
        int64_t bytes_read = 0;
        TEST_ASSERT_EQ(MTX_SUCCESS, parse_int64(&x, s, &endptr, &bytes_read));
        TEST_ASSERT_EQ(42, x);
        TEST_ASSERT_EQ(endptr, s+2);
        TEST_ASSERT_EQ(bytes_read, 2);
    }
    return TEST_SUCCESS;
}

/**
 * ‘test_parse_int32_ex()’ tests parsing strings of 32-bit integers.
 */
int test_parse_int32_ex(void)
{
    {
        int32_t number;
        const char * s = "";
        TEST_ASSERT_EQ(EINVAL, parse_int32_ex(s, NULL, &number, NULL));
    }

    {
        int32_t number;
        const char * s = "0";
        const char * s_end;
        TEST_ASSERT_EQ(0, parse_int32_ex(s, NULL, &number, &s_end));
        TEST_ASSERT_EQ(0, number);
        TEST_ASSERT_EQ(s_end, s+1);
    }

    {
        int32_t number;
        const char * s = "1";
        const char * s_end;
        TEST_ASSERT_EQ(0, parse_int32_ex(s, NULL, &number, &s_end));
        TEST_ASSERT_EQ(1, number);
        TEST_ASSERT_EQ(s_end, s+1);
    }

    {
        int32_t number;
        const char * s = "1";
        TEST_ASSERT_EQ(0, parse_int32_ex(s, NULL, &number, NULL));
        TEST_ASSERT_EQ(1, number);
    }

    {
        int32_t number;
        const char * s = "-1";
        TEST_ASSERT_EQ(0, parse_int32_ex(s, NULL, &number, NULL));
        TEST_ASSERT_EQ(-1, number);
    }

    {
        int32_t number;
        const char * s = "42";
        const char * s_end;
        TEST_ASSERT_EQ(0, parse_int32_ex(s, NULL, &number, &s_end));
        TEST_ASSERT_EQ(42, number);
        TEST_ASSERT_EQ(s_end, s+2);
    }

    {
        /* Parse INT32_MAX, which is 2^31-1. */
        int32_t number;
        const char * s = "2147483647";
        const char * s_end;
        TEST_ASSERT_EQ(0, parse_int32_ex(s, NULL, &number, &s_end));
        TEST_ASSERT_EQ(INT32_MAX, number);
        TEST_ASSERT_EQ(s_end, s+10);
    }

    {
        /* Parse INT32_MIN, which is -2^31-1. */
        int32_t number;
        const char * s = "-2147483648";
        const char * s_end;
        TEST_ASSERT_EQ(0, parse_int32_ex(s, NULL, &number, &s_end));
        TEST_ASSERT_EQ(INT32_MIN, number);
        TEST_ASSERT_EQ(s_end, s+11);
    }

    {
        /* Parse a number larger than INT32_MAX. */
        int32_t number;
        const char * s = "2147483648";
        TEST_ASSERT_EQ(ERANGE, parse_int32_ex(s, NULL, &number, NULL));
    }

    {
        /* Parse a number smaller than INT32_MIN. */
        int32_t number;
        const char * s = "-2147483649";
        TEST_ASSERT_EQ(ERANGE, parse_int32_ex(s, NULL, &number, NULL));
    }

    /* Parse strings containing numbers that may end with a delimiter,
     * such as a comma. */
    {
        int32_t number;
        const char * s = "42";
        const char * end_chars = "";
        const char * s_end;
        TEST_ASSERT_EQ(0, parse_int32_ex(s, end_chars, &number, &s_end));
        TEST_ASSERT_EQ(42, number);
        TEST_ASSERT_EQ(s_end, s+2);
    }

    {
        int32_t number;
        const char * s = "42,";
        const char * end_chars = "";
        const char * s_end;
        TEST_ASSERT_EQ(EINVAL, parse_int32_ex(s, end_chars, &number, &s_end));
    }

    {
        int32_t number;
        const char * s = "42,";
        const char * end_chars = ",";
        const char * s_end;
        TEST_ASSERT_EQ(0, parse_int32_ex(s, end_chars, &number, &s_end));
        TEST_ASSERT_EQ(42, number);
        TEST_ASSERT_EQ(s_end, s+3);
    }

    {
        int32_t number;
        const char * s = "42;";
        const char * end_chars = ",";
        const char * s_end;
        TEST_ASSERT_EQ(EINVAL, parse_int32_ex(s, end_chars, &number, &s_end));
    }

    return TEST_SUCCESS;
}

/**
 * ‘test_parse_int64_ex()’ tests parsing strings of 64-bit integers.
 */
int test_parse_int64_ex()
{
    {
        int64_t number;
        const char * s = "";
        TEST_ASSERT_EQ(EINVAL, parse_int64_ex(s, NULL, &number, NULL));
    }

    {
        int64_t number;
        const char * s = "0";
        const char * s_end;
        TEST_ASSERT_EQ(0, parse_int64_ex(s, NULL, &number, &s_end));
        TEST_ASSERT_EQ(0, number);
        TEST_ASSERT_EQ(s_end, s+1);
    }

    {
        int64_t number;
        const char * s = "1";
        const char * s_end;
        TEST_ASSERT_EQ(0, parse_int64_ex(s, NULL, &number, &s_end));
        TEST_ASSERT_EQ(1, number);
        TEST_ASSERT_EQ(s_end, s+1);
    }

    {
        int64_t number;
        const char * s = "1";
        TEST_ASSERT_EQ(0, parse_int64_ex(s, NULL, &number, NULL));
        TEST_ASSERT_EQ(1, number);
    }

    {
        int64_t number;
        const char * s = "-1";
        TEST_ASSERT_EQ(0, parse_int64_ex(s, NULL, &number, NULL));
        TEST_ASSERT_EQ(-1, number);
    }

    {
        int64_t number;
        const char * s = "42";
        const char * s_end;
        TEST_ASSERT_EQ(0, parse_int64_ex(s, NULL, &number, &s_end));
        TEST_ASSERT_EQ(42, number);
        TEST_ASSERT_EQ(s_end, s+2);
    }

    /* Test 32-bit integer range. */

    {
        /* Parse INT32_MAX, which is 2^31-1. */
        int64_t number;
        const char * s = "2147483647";
        const char * s_end;
        TEST_ASSERT_EQ(0, parse_int64_ex(s, NULL, &number, &s_end));
        TEST_ASSERT_EQ(INT32_MAX, number);
        TEST_ASSERT_EQ(s_end, s+10);
    }

    {
        /* Parse INT32_MIN, which is -2^31-1. */
        int64_t number;
        const char * s = "-2147483648";
        const char * s_end;
        TEST_ASSERT_EQ(0, parse_int64_ex(s, NULL, &number, &s_end));
        TEST_ASSERT_EQ(INT32_MIN, number);
        TEST_ASSERT_EQ(s_end, s+11);
    }

    {
        /* Parse a number larger than INT32_MAX. */
        int64_t number;
        const char * s = "2147483648";
        const char * s_end;
        TEST_ASSERT_EQ(0, parse_int64_ex(s, NULL, &number, &s_end));
        TEST_ASSERT_EQ(2147483648LL, number);
        TEST_ASSERT_EQ(s_end, s+10);
    }

    {
        /* Parse a number smaller than INT32_MIN. */
        int64_t number;
        const char * s = "-2147483649";
        const char * s_end;
        TEST_ASSERT_EQ(0, parse_int64_ex(s, NULL, &number, &s_end));
        TEST_ASSERT_EQ(-2147483649LL, number);
        TEST_ASSERT_EQ(s_end, s+11);
    }

    /* Test 64-bit integer range. */

    {
        /* Parse INT64_MAX, which is 2^63-1. */
        int64_t number;
        const char * s = "9223372036854775807";
        const char * s_end;
        TEST_ASSERT_EQ(0, parse_int64_ex(s, NULL, &number, &s_end));
        TEST_ASSERT_EQ(INT64_MAX, number);
        TEST_ASSERT_EQ(s_end, s+19);
    }

    {
        /* Parse INT64_MIN, which is -2^63-1. */
        int64_t number;
        const char * s = "-9223372036854775808";
        const char * s_end;
        TEST_ASSERT_EQ(0, parse_int64_ex(s, NULL, &number, &s_end));
        TEST_ASSERT_EQ(INT64_MIN, number);
        TEST_ASSERT_EQ(s_end, s+20);
    }

    {
        /* Parse a number larger than INT64_MAX. */
        int64_t number;
        const char * s = "9223372036854775808";
        TEST_ASSERT_EQ(ERANGE, parse_int64_ex(s, NULL, &number, NULL));
    }

    {
        /* Parse a number smaller than INT64_MIN. */
        int64_t number;
        const char * s = "-9223372036854775809";
        TEST_ASSERT_EQ(ERANGE, parse_int64_ex(s, NULL, &number, NULL));
    }

    /* Test parsing strings that contain numbers which may end with a
     * delimiter, such as a comma. */

    {
        int64_t number;
        const char * s = "42";
        const char * end_chars = "";
        const char * s_end;
        TEST_ASSERT_EQ(0, parse_int64_ex(s, end_chars, &number, &s_end));
        TEST_ASSERT_EQ(42, number);
        TEST_ASSERT_EQ(s_end, s+2);
    }

    {
        int64_t number;
        const char * s = "42,";
        const char * end_chars = "";
        const char * s_end;
        TEST_ASSERT_EQ(EINVAL, parse_int64_ex(s, end_chars, &number, &s_end));
    }

    {
        int64_t number;
        const char * s = "42,";
        const char * end_chars = ",";
        const char * s_end;
        TEST_ASSERT_EQ(0, parse_int64_ex(s, end_chars, &number, &s_end));
        TEST_ASSERT_EQ(42, number);
        TEST_ASSERT_EQ(s_end, s+3);
    }

    {
        int64_t number;
        const char * s = "42;";
        const char * end_chars = ",";
        const char * s_end;
        TEST_ASSERT_EQ(EINVAL, parse_int64_ex(s, end_chars, &number, &s_end));
    }

    return TEST_SUCCESS;
}

/**
 * ‘main()’ entry point and test driver.
 */
int main(int argc, char * argv[])
{
    TEST_SUITE_BEGIN("Running tests for string parsing functions\n");
    TEST_RUN(test_parse_int32);
    TEST_RUN(test_parse_int64);
    TEST_RUN(test_parse_int32_ex);
    TEST_RUN(test_parse_int64_ex);
    TEST_SUITE_END();
    return (TEST_SUITE_STATUS == TEST_SUCCESS) ?
        EXIT_SUCCESS : EXIT_FAILURE;
}
