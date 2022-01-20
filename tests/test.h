/* This file is part of Libmtx.
 *
 * Copyright (C) 2021 James D. Trotter
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
 * Last modified: 2021-08-09
 *
 * Unit test utilities.
 */

#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

/*
 * Constants.
 */

#define TEST_SUCCESS 0
#define TEST_FAILURE 1

/*
 * Test suite macros.
 */

#define TEST_SUITE_BEGIN(description)           \
    int err;                                    \
    int num_tests = 0;                          \
    int num_tests_passed = 0;                   \
    int num_tests_failed = 0;                   \
    fprintf(stdout, description);               \
    for (int i = 0; i < 60; i++)                \
        fputc('=', stdout);                     \
    fputc('\n', stdout);

#define TEST_SUITE_END()                        \
    for (int i = 0; i < 60; i++)                \
        fputc('-', stdout);                     \
    fputc('\n', stdout);                        \
    fprintf(stdout, "Summary: Ran %d tests, "   \
            "%d passed, %d failed\n",           \
            num_tests, num_tests_passed,        \
            num_tests_failed);                  \
    for (int i = 0; i < 60; i++)                \
        fputc('=', stdout);                     \
    fputc('\n', stdout);                        \

#define TEST_SUITE_STATUS                       \
    ((num_tests_failed > 0) ?                   \
     TEST_FAILURE : TEST_SUCCESS)

/*
 * Test macros.
 */

#define TEST_RUN(test)                          \
    num_tests++;                                \
    err = test();                               \
    if (err)                                    \
        num_tests_failed++;                     \
    else {                                      \
        fprintf(stdout, "PASS:%s\n", #test);    \
        num_tests_passed++;                     \
    }

/*
 * Test assertion macros.
 */

#define TEST_FAIL()                                     \
    {                                                   \
        fprintf(stdout, "FAIL:%s:%s:%d\n",              \
                __FUNCTION__, __FILE__, __LINE__);      \
        return TEST_FAILURE;                            \
    }

#define TEST_FAIL_MSG(msg, ...)                         \
    {                                                   \
        fprintf(stdout, "FAIL:%s:%s:%d: "msg"\n",       \
                __FUNCTION__, __FILE__, __LINE__,       \
                ##__VA_ARGS__);                         \
        return TEST_FAILURE;                            \
    }

#define TEST_ASSERT(condition)                          \
    if (!(condition)) {                                 \
        fprintf(stdout, "FAIL:%s:%s:%d: "               \
                "Assertion failed: %s\n",               \
                __FUNCTION__, __FILE__, __LINE__,       \
                #condition);                            \
        return TEST_FAILURE;                            \
    }

#define TEST_ASSERT_FALSE(condition)                    \
    if ((condition)) {                                  \
        fprintf(stdout, "FAIL:%s:%s:%d: "               \
                "Assertion failed: !(%s)\n",            \
                __FUNCTION__, __FILE__, __LINE__,       \
                #condition);                            \
        return TEST_FAILURE;                            \
    }

#define TEST_ASSERT_EQ(expected, actual)                \
    if ((expected) != (actual)) {                       \
        fprintf(stdout, "FAIL:%s:%s:%d: "               \
                "Assertion failed: %s != %s\n",         \
                __FUNCTION__, __FILE__, __LINE__,       \
                #expected, #actual);                    \
        return TEST_FAILURE;                            \
    }

#define TEST_ASSERT_EQ_MSG(expected, actual, msg, ...)  \
    if ((expected) != (actual)) {                       \
        fprintf(stdout, "FAIL:%s:%s:%d: "               \
                "Assertion failed: %s != %s ("msg")\n", \
                __FUNCTION__, __FILE__, __LINE__,       \
                #expected, #actual, ##__VA_ARGS__);     \
        return TEST_FAILURE;                            \
    }

#define TEST_ASSERT_NEQ(expected, actual)               \
    if ((expected) == (actual)) {                       \
        fprintf(stdout, "FAIL:%s:%s:%d: "               \
                "Assertion failed: %s == %s\n",         \
                __FUNCTION__, __FILE__, __LINE__,       \
                #expected, #actual);                    \
        return TEST_FAILURE;                            \
    }

#define TEST_ASSERT_NEQ_MSG(expected, actual, msg, ...) \
    if ((expected) == (actual)) {                       \
        fprintf(stdout, "FAIL:%s:%s:%d: "               \
                "Assertion failed: %s == %s ("msg")\n", \
                __FUNCTION__, __FILE__, __LINE__,       \
                #expected, #actual, ##__VA_ARGS__);     \
        return TEST_FAILURE;                            \
    }

#define TEST_ASSERT_LE(expected, actual)                \
    if (!((expected) <= (actual))) {                    \
        fprintf(stdout, "FAIL:%s:%s:%d: "               \
                "Assertion failed: %s <= %s\n",         \
                __FUNCTION__, __FILE__, __LINE__,       \
                #expected, #actual);                    \
        return TEST_FAILURE;                            \
    }

#define TEST_ASSERT_LE_MSG(expected, actual, msg, ...)  \
    if (!((expected) <= (actual))) {                    \
        fprintf(stdout, "FAIL:%s:%s:%d: "               \
                "Assertion failed: %s <= %s ("msg")\n", \
                __FUNCTION__, __FILE__, __LINE__,       \
                #expected, #actual, ##__VA_ARGS__);     \
        return TEST_FAILURE;                            \
    }

/*
 * Test assertions for strings.
 */

#define TEST_ASSERT_STREQ(expected, actual)             \
    if (strcmp((expected), (actual)) != 0) {            \
        fprintf(stdout, "FAIL:%s:%s:%d: "               \
                "Assertion failed: %s != %s\n",         \
                __FUNCTION__, __FILE__, __LINE__,       \
                #expected, #actual);                    \
        return TEST_FAILURE;                            \
    }

#define TEST_ASSERT_STREQ_MSG(expected, actual, msg, ...)       \
    if (strcmp((expected), (actual)) != 0) {                    \
        fprintf(stdout, "FAIL:%s:%s:%d: "                       \
                "Assertion failed: %s != %s ("msg")\n",         \
                __FUNCTION__, __FILE__, __LINE__,               \
                #expected, #actual, ##__VA_ARGS__);             \
        return TEST_FAILURE;                                    \
    }

/*
 * Test assertions for floating point numbers.
 */

/**
 * `float_nearly_equal()' compares two single precision floating point
 * numbers for equality using a given tolerance.
 *
 * If either `a' or `b' is `NaN', then `false' is returned.  If `a'
 * and `b' are infinities of equal sign, then `true' is returned.
 *
 * If `a' or `b' are zero or sub-normal (i.e., less than `FLT_MIN'),
 * then the absolute difference `|a-b|/FLT_MIN' is compared to
 * `epsilon'.  If `a' and `b' are near zero (`|a|+|b|' is smaller than
 * `FLT_EPSILON'), then their absolute difference, `|a-b|', is
 * compared to `epsilon'.  Otherwise, the relative difference
 * `|a-b|/(|a|+|b|)' is used (or, `|a-b|/FLT_MAX' if
 * `|a|+|b|>FLT_MAX').
 */
static inline bool float_nearly_equal(
    float a,
    float b,
    float epsilon)
{
    float a_abs = fabsf(a);
    float b_abs = fabsf(b);
    float diff_abs = fabsf(a - b);
    if (a == b) {
        return true;
    } else if (a == 0 || b == 0 || (a_abs + b_abs) < FLT_MIN) {
        return diff_abs < (epsilon * FLT_MIN);
    } else if (a_abs + b_abs < DBL_EPSILON) {
        return diff_abs < epsilon;
    } else {
        return diff_abs / fminf(a_abs + b_abs, FLT_MAX) < epsilon;
    }
}

#define TEST_ASSERT_FLOAT_NEAR(expected, actual, epsilon)       \
    if (!float_nearly_equal((expected), (actual), (epsilon))) { \
        fprintf(stdout, "FAIL:%s:%s:%d: "                       \
                "Assertion failed: %s != %s\n",                 \
                __FUNCTION__, __FILE__, __LINE__,               \
                #expected, #actual);                            \
        return TEST_FAILURE;                                    \
    }

#define TEST_ASSERT_FLOAT_NEAR_MSG(                             \
    expected, actual, epsilon, msg, ...)                        \
    if (!float_nearly_equal((expected), (actual), (epsilon))) { \
        fprintf(stdout, "FAIL:%s:%s:%d: "                       \
                "Assertion failed: %s != %s ("msg")\n",         \
                __FUNCTION__, __FILE__, __LINE__,               \
                #expected, #actual, ##__VA_ARGS__);             \
        return TEST_FAILURE;                                    \
    }

/**
 * `double_nearly_equal()' compares two double precision floating
 * point numbers for equality using a given tolerance.
 *
 * If either `a' or `b' is `NaN', then `false' is returned.  If `a'
 * and `b' are infinities of equal sign, then `true' is returned.
 *
 * If `a' or `b' are zero or sub-normal (i.e., less than `DBL_MIN'),
 * then the absolute difference `|a-b|/DBL_MIN' is compared to
 * `epsilon'.  If `a' and `b' are near zero (`|a|+|b|' is smaller than
 * `DBL_EPSILON'), then their absolute difference, `|a-b|', is
 * compared to `epsilon'.  Otherwise, the relative difference
 * `|a-b|/(|a|+|b|)' is used (or, `|a-b|/DBL_MAX' if
 * `|a|+|b|>DBL_MAX').
 */
static inline bool double_nearly_equal(
    double a,
    double b,
    double epsilon)
{
    double a_abs = fabs(a);
    double b_abs = fabs(b);
    double diff_abs = fabs(a - b);
    if (a == b) {
        return true;
    } else if (a == 0 || b == 0 || (a_abs + b_abs) < DBL_MIN) {
        return diff_abs < (epsilon * DBL_MIN);
    } else if (a_abs + b_abs < DBL_EPSILON) {
        return diff_abs < epsilon;
    } else {
        return diff_abs / fmin(a_abs + b_abs, DBL_MAX) < epsilon;
    }
}

#define TEST_ASSERT_DOUBLE_NEAR(expected, actual, epsilon)       \
    if (!double_nearly_equal((expected), (actual), (epsilon))) { \
        fprintf(stdout, "FAIL:%s:%s:%d: "                        \
                "Assertion failed: %s != %s\n",                  \
                __FUNCTION__, __FILE__, __LINE__,                \
                #expected, #actual);                             \
        return TEST_FAILURE;                                     \
    }

#define TEST_ASSERT_DOUBLE_NEAR_MSG(                             \
    expected, actual, epsilon, msg, ...)                         \
    if (!double_nearly_equal((expected), (actual), (epsilon))) { \
        fprintf(stdout, "FAIL:%s:%s:%d: "                        \
                "Assertion failed: %s != %s ("msg")\n",          \
                __FUNCTION__, __FILE__, __LINE__,                \
                #expected, #actual, ##__VA_ARGS__);              \
        return TEST_FAILURE;                                     \
    }
