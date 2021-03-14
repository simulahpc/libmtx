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
 * Last modified: 2021-06-18
 *
 * Unit test utilities.
 */

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
