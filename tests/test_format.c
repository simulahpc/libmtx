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
 * Last modified: 2021-06-29
 *
 * Unit tests for string printing functions.
 */

#include "test.h"

#include "matrixmarket/format.h"

#include <errno.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * `test_format_specifier_str()` tests converting format specifiers to
 * strings.
 */
int test_format_specifier_str(void)
{
    {
        struct format_specifier format = {
            .flags = 0,
            .width = format_specifier_width_none,
            .precision = format_specifier_precision_none,
            .length = format_specifier_length_none,
            .specifier = format_specifier_d };
        char * format_str = format_specifier_str(format);
        TEST_ASSERT_STREQ("%d", format_str);
        free(format_str);
    }

    {
        struct format_specifier format = {
            .flags = 0,
            .width = format_specifier_width_none,
            .precision = format_specifier_precision_none,
            .length = format_specifier_length_none,
            .specifier = format_specifier_f };
        char * format_str = format_specifier_str(format);
        TEST_ASSERT_STREQ("%f", format_str);
        free(format_str);
    }

    {
        struct format_specifier format = {
            .flags = 0,
            .width = format_specifier_width_none,
            .precision = format_specifier_precision_none,
            .length = format_specifier_length_L,
            .specifier = format_specifier_g };
        char * format_str = format_specifier_str(format);
        TEST_ASSERT_STREQ("%Lg", format_str);
        free(format_str);
    }

    {
        struct format_specifier format = {
            .flags = 0,
            .width = format_specifier_width_none,
            .precision = 2,
            .length = format_specifier_length_none,
            .specifier = format_specifier_f };
        char * format_str = format_specifier_str(format);
        TEST_ASSERT_STREQ("%.2f", format_str);
        free(format_str);
    }

    {
        struct format_specifier format = {
            .flags = 0,
            .width = 3,
            .precision = format_specifier_precision_none,
            .length = format_specifier_length_none,
            .specifier = format_specifier_f };
        char * format_str = format_specifier_str(format);
        TEST_ASSERT_STREQ("%3f", format_str);
        free(format_str);
    }

    {
        struct format_specifier format = {
            .flags = format_specifier_flags_minus,
            .width = 3,
            .precision = format_specifier_precision_none,
            .length = format_specifier_length_none,
            .specifier = format_specifier_f };
        char * format_str = format_specifier_str(format);
        TEST_ASSERT_STREQ("%-3f", format_str);
        free(format_str);
    }

    {
        struct format_specifier format = {
            .flags = format_specifier_flags_minus | format_specifier_flags_space,
            .width = 4,
            .precision = format_specifier_precision_none,
            .length = format_specifier_length_none,
            .specifier = format_specifier_f };
        char * format_str = format_specifier_str(format);
        TEST_ASSERT_STREQ("%- 4f", format_str);
        free(format_str);
    }

    {
        struct format_specifier format = {
            .flags = 0,
            .width = 4,
            .precision = 1,
            .length = format_specifier_length_none,
            .specifier = format_specifier_f };
        char * format_str = format_specifier_str(format);
        TEST_ASSERT_STREQ("%4.1f", format_str);
        free(format_str);
    }

    return TEST_SUCCESS;
}

/**
 * `test_parse_format_specifier()` tests converting a string to a
 * format specification.
 */
int test_parse_format_specifier(void)
{
    {
        struct format_specifier format;
        int err = parse_format_specifier("", &format, NULL);
        TEST_ASSERT_EQ_MSG(EINVAL, err, "%s", strerror(err));
    }

    {
        struct format_specifier format;
        int err = parse_format_specifier("%", &format, NULL);
        TEST_ASSERT_EQ_MSG(EINVAL, err, "%s", strerror(err));
    }

    {
        struct format_specifier format;
        int err = parse_format_specifier("%d", &format, NULL);
        TEST_ASSERT_EQ_MSG(0, err, "%s", strerror(err));
        TEST_ASSERT_EQ(0, format.flags);
        TEST_ASSERT_EQ(format_specifier_width_none, format.width);
        TEST_ASSERT_EQ(format_specifier_precision_none, format.precision);
        TEST_ASSERT_EQ(format_specifier_length_none, format.length);
        TEST_ASSERT_EQ(format_specifier_d, format.specifier);
    }

    {
        struct format_specifier format;
        int err = parse_format_specifier("%f", &format, NULL);
        TEST_ASSERT_EQ_MSG(0, err, "%s", strerror(err));
        TEST_ASSERT_EQ(0, format.flags);
        TEST_ASSERT_EQ(format_specifier_width_none, format.width);
        TEST_ASSERT_EQ(format_specifier_precision_none, format.precision);
        TEST_ASSERT_EQ(format_specifier_length_none, format.length);
        TEST_ASSERT_EQ(format_specifier_f, format.specifier);
    }

    {
        struct format_specifier format;
        int err = parse_format_specifier("%+f", &format, NULL);
        TEST_ASSERT_EQ_MSG(0, err, "%s", strerror(err));
        TEST_ASSERT_EQ(format_specifier_flags_plus, format.flags);
        TEST_ASSERT_EQ(format_specifier_width_none, format.width);
        TEST_ASSERT_EQ(format_specifier_precision_none, format.precision);
        TEST_ASSERT_EQ(format_specifier_length_none, format.length);
        TEST_ASSERT_EQ(format_specifier_f, format.specifier);
    }

    {
        struct format_specifier format;
        int err = parse_format_specifier("%+ f", &format, NULL);
        TEST_ASSERT_EQ_MSG(0, err, "%s", strerror(err));
        TEST_ASSERT_EQ(format_specifier_flags_plus | format_specifier_flags_space, format.flags);
        TEST_ASSERT_EQ(format_specifier_width_none, format.width);
        TEST_ASSERT_EQ(format_specifier_precision_none, format.precision);
        TEST_ASSERT_EQ(format_specifier_length_none, format.length);
        TEST_ASSERT_EQ(format_specifier_f, format.specifier);
    }

    {
        struct format_specifier format;
        int err = parse_format_specifier("%2f", &format, NULL);
        TEST_ASSERT_EQ_MSG(0, err, "%s", strerror(err));
        TEST_ASSERT_EQ(0, format.flags);
        TEST_ASSERT_EQ(2, format.width);
        TEST_ASSERT_EQ(format_specifier_precision_none, format.precision);
        TEST_ASSERT_EQ(format_specifier_length_none, format.length);
        TEST_ASSERT_EQ(format_specifier_f, format.specifier);
    }

    {
        struct format_specifier format;
        int err = parse_format_specifier("%.1f", &format, NULL);
        TEST_ASSERT_EQ_MSG(0, err, "%s", strerror(err));
        TEST_ASSERT_EQ(0, format.flags);
        TEST_ASSERT_EQ(format_specifier_width_none, format.width);
        TEST_ASSERT_EQ(1, format.precision);
        TEST_ASSERT_EQ(format_specifier_length_none, format.length);
        TEST_ASSERT_EQ(format_specifier_f, format.specifier);
    }

    {
        struct format_specifier format;
        int err = parse_format_specifier("%.*f", &format, NULL);
        TEST_ASSERT_EQ_MSG(0, err, "%s", strerror(err));
        TEST_ASSERT_EQ(0, format.flags);
        TEST_ASSERT_EQ(format_specifier_width_none, format.width);
        TEST_ASSERT_EQ(format_specifier_precision_star, format.precision);
        TEST_ASSERT_EQ(format_specifier_length_none, format.length);
        TEST_ASSERT_EQ(format_specifier_f, format.specifier);
    }

    {
        struct format_specifier format;
        int err = parse_format_specifier("%Lf", &format, NULL);
        TEST_ASSERT_EQ_MSG(0, err, "%s", strerror(err));
        TEST_ASSERT_EQ(0, format.flags);
        TEST_ASSERT_EQ(format_specifier_width_none, format.width);
        TEST_ASSERT_EQ(format_specifier_precision_none, format.precision);
        TEST_ASSERT_EQ(format_specifier_length_L, format.length);
        TEST_ASSERT_EQ(format_specifier_f, format.specifier);
    }

    return TEST_SUCCESS;
}

/**
 * `main()' entry point and test driver.
 */
int main(int argc, char * argv[])
{
    TEST_SUITE_BEGIN("Running tests for string formatting\n");
    TEST_RUN(test_format_specifier_str);
    TEST_RUN(test_parse_format_specifier);
    TEST_SUITE_END();
    return (TEST_SUITE_STATUS == TEST_SUCCESS) ?
        EXIT_SUCCESS : EXIT_FAILURE;
}
