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
 * Last modified: 2022-10-10
 *
 * Unit tests for string printing functions.
 */

#include "test.h"

#include "libmtx/util/fmtspec.h"

#include <errno.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * `test_fmtspecstr()` tests converting format specifiers to
 * strings.
 */
int test_fmtspecstr(void)
{
    {
        struct fmtspec format = {
            .flags = 0,
            .width = fmtspec_width_none,
            .precision = fmtspec_precision_none,
            .length = fmtspec_length_none,
            .specifier = fmtspec_d };
        char * format_str = fmtspecstr(format);
        TEST_ASSERT_STREQ("%d", format_str);
        free(format_str);
    }

    {
        struct fmtspec format = {
            .flags = 0,
            .width = fmtspec_width_none,
            .precision = fmtspec_precision_none,
            .length = fmtspec_length_none,
            .specifier = fmtspec_f };
        char * format_str = fmtspecstr(format);
        TEST_ASSERT_STREQ("%f", format_str);
        free(format_str);
    }

    {
        struct fmtspec format = {
            .flags = 0,
            .width = fmtspec_width_none,
            .precision = fmtspec_precision_none,
            .length = fmtspec_length_L,
            .specifier = fmtspec_g };
        char * format_str = fmtspecstr(format);
        TEST_ASSERT_STREQ("%Lg", format_str);
        free(format_str);
    }

    {
        struct fmtspec format = {
            .flags = 0,
            .width = fmtspec_width_none,
            .precision = 2,
            .length = fmtspec_length_none,
            .specifier = fmtspec_f };
        char * format_str = fmtspecstr(format);
        TEST_ASSERT_STREQ("%.2f", format_str);
        free(format_str);
    }

    {
        struct fmtspec format = {
            .flags = 0,
            .width = 3,
            .precision = fmtspec_precision_none,
            .length = fmtspec_length_none,
            .specifier = fmtspec_f };
        char * format_str = fmtspecstr(format);
        TEST_ASSERT_STREQ("%3f", format_str);
        free(format_str);
    }

    {
        struct fmtspec format = {
            .flags = fmtspec_flags_minus,
            .width = 3,
            .precision = fmtspec_precision_none,
            .length = fmtspec_length_none,
            .specifier = fmtspec_f };
        char * format_str = fmtspecstr(format);
        TEST_ASSERT_STREQ("%-3f", format_str);
        free(format_str);
    }

    {
        struct fmtspec format = {
            .flags = fmtspec_flags_minus | fmtspec_flags_space,
            .width = 4,
            .precision = fmtspec_precision_none,
            .length = fmtspec_length_none,
            .specifier = fmtspec_f };
        char * format_str = fmtspecstr(format);
        TEST_ASSERT_STREQ("%- 4f", format_str);
        free(format_str);
    }

    {
        struct fmtspec format = {
            .flags = 0,
            .width = 4,
            .precision = 1,
            .length = fmtspec_length_none,
            .specifier = fmtspec_f };
        char * format_str = fmtspecstr(format);
        TEST_ASSERT_STREQ("%4.1f", format_str);
        free(format_str);
    }

    return TEST_SUCCESS;
}

/**
 * `test_parse_fmtspec()` tests converting a string to a
 * format specification.
 */
int test_parse_fmtspec(void)
{
    {
        struct fmtspec format;
        int err = parse_fmtspec("", &format, NULL);
        TEST_ASSERT_EQ_MSG(EINVAL, err, "%s", strerror(err));
    }

    {
        struct fmtspec format;
        int err = parse_fmtspec("%", &format, NULL);
        TEST_ASSERT_EQ_MSG(EINVAL, err, "%s", strerror(err));
    }

    {
        struct fmtspec format;
        int err = parse_fmtspec("%d", &format, NULL);
        TEST_ASSERT_EQ_MSG(0, err, "%s", strerror(err));
        TEST_ASSERT_EQ(0, format.flags);
        TEST_ASSERT_EQ(fmtspec_width_none, format.width);
        TEST_ASSERT_EQ(fmtspec_precision_none, format.precision);
        TEST_ASSERT_EQ(fmtspec_length_none, format.length);
        TEST_ASSERT_EQ(fmtspec_d, format.specifier);
    }

    {
        struct fmtspec format;
        int err = parse_fmtspec("%f", &format, NULL);
        TEST_ASSERT_EQ_MSG(0, err, "%s", strerror(err));
        TEST_ASSERT_EQ(0, format.flags);
        TEST_ASSERT_EQ(fmtspec_width_none, format.width);
        TEST_ASSERT_EQ(fmtspec_precision_none, format.precision);
        TEST_ASSERT_EQ(fmtspec_length_none, format.length);
        TEST_ASSERT_EQ(fmtspec_f, format.specifier);
    }

    {
        struct fmtspec format;
        int err = parse_fmtspec("%+f", &format, NULL);
        TEST_ASSERT_EQ_MSG(0, err, "%s", strerror(err));
        TEST_ASSERT_EQ(fmtspec_flags_plus, format.flags);
        TEST_ASSERT_EQ(fmtspec_width_none, format.width);
        TEST_ASSERT_EQ(fmtspec_precision_none, format.precision);
        TEST_ASSERT_EQ(fmtspec_length_none, format.length);
        TEST_ASSERT_EQ(fmtspec_f, format.specifier);
    }

    {
        struct fmtspec format;
        int err = parse_fmtspec("%+ f", &format, NULL);
        TEST_ASSERT_EQ_MSG(0, err, "%s", strerror(err));
        TEST_ASSERT_EQ(fmtspec_flags_plus | fmtspec_flags_space, format.flags);
        TEST_ASSERT_EQ(fmtspec_width_none, format.width);
        TEST_ASSERT_EQ(fmtspec_precision_none, format.precision);
        TEST_ASSERT_EQ(fmtspec_length_none, format.length);
        TEST_ASSERT_EQ(fmtspec_f, format.specifier);
    }

    {
        struct fmtspec format;
        int err = parse_fmtspec("%2f", &format, NULL);
        TEST_ASSERT_EQ_MSG(0, err, "%s", strerror(err));
        TEST_ASSERT_EQ(0, format.flags);
        TEST_ASSERT_EQ(2, format.width);
        TEST_ASSERT_EQ(fmtspec_precision_none, format.precision);
        TEST_ASSERT_EQ(fmtspec_length_none, format.length);
        TEST_ASSERT_EQ(fmtspec_f, format.specifier);
    }

    {
        struct fmtspec format;
        int err = parse_fmtspec("%.1f", &format, NULL);
        TEST_ASSERT_EQ_MSG(0, err, "%s", strerror(err));
        TEST_ASSERT_EQ(0, format.flags);
        TEST_ASSERT_EQ(fmtspec_width_none, format.width);
        TEST_ASSERT_EQ(1, format.precision);
        TEST_ASSERT_EQ(fmtspec_length_none, format.length);
        TEST_ASSERT_EQ(fmtspec_f, format.specifier);
    }

    {
        struct fmtspec format;
        int err = parse_fmtspec("%.*f", &format, NULL);
        TEST_ASSERT_EQ_MSG(0, err, "%s", strerror(err));
        TEST_ASSERT_EQ(0, format.flags);
        TEST_ASSERT_EQ(fmtspec_width_none, format.width);
        TEST_ASSERT_EQ(fmtspec_precision_star, format.precision);
        TEST_ASSERT_EQ(fmtspec_length_none, format.length);
        TEST_ASSERT_EQ(fmtspec_f, format.specifier);
    }

    {
        struct fmtspec format;
        int err = parse_fmtspec("%Lf", &format, NULL);
        TEST_ASSERT_EQ_MSG(0, err, "%s", strerror(err));
        TEST_ASSERT_EQ(0, format.flags);
        TEST_ASSERT_EQ(fmtspec_width_none, format.width);
        TEST_ASSERT_EQ(fmtspec_precision_none, format.precision);
        TEST_ASSERT_EQ(fmtspec_length_L, format.length);
        TEST_ASSERT_EQ(fmtspec_f, format.specifier);
    }

    return TEST_SUCCESS;
}

/**
 * `main()' entry point and test driver.
 */
int main(int argc, char * argv[])
{
    TEST_SUITE_BEGIN("Running tests for string formatting\n");
    TEST_RUN(test_fmtspecstr);
    TEST_RUN(test_parse_fmtspec);
    TEST_SUITE_END();
    return (TEST_SUITE_STATUS == TEST_SUCCESS) ?
        EXIT_SUCCESS : EXIT_FAILURE;
}
