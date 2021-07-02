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
 * Unit tests for vectors in Matrix Market format.
 */

#include "test.h"

#include <matrixmarket/error.h>
#include <matrixmarket/mtx.h>
#include <matrixmarket/vector_array.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * `test_mtx_init_vector_array_real()` tests creating a real vector in the Matrix
 * Market format.
 */
int test_mtx_init_vector_array_real(void)
{
    int err;
    struct mtx vector;
    int num_comment_lines = 1;
    const char * comment_lines[] = {"a comment"};
    float data[] = {1.0f, 2.0f, 3.0f};
    int size = sizeof(data) / sizeof(*data);
    err = mtx_init_vector_array_real(
        &vector, num_comment_lines, comment_lines, size, data);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
    TEST_ASSERT_EQ(mtx_vector, vector.object);
    TEST_ASSERT_EQ(mtx_array, vector.format);
    TEST_ASSERT_EQ(mtx_real, vector.field);
    TEST_ASSERT_EQ(mtx_general, vector.symmetry);
    TEST_ASSERT_EQ(1, vector.num_comment_lines);
    TEST_ASSERT_STREQ("a comment", vector.comment_lines[0]);
    TEST_ASSERT_EQ(3, vector.num_rows);
    TEST_ASSERT_EQ(-1, vector.num_columns);
    TEST_ASSERT_EQ(3, vector.num_nonzeros);
    TEST_ASSERT_EQ(3, vector.size);
    TEST_ASSERT_EQ(1.0f, ((const float *) vector.data)[0]);
    TEST_ASSERT_EQ(2.0f, ((const float *) vector.data)[1]);
    TEST_ASSERT_EQ(3.0f, ((const float *) vector.data)[2]);
    if (!err)
        mtx_free(&vector);
    return TEST_SUCCESS;
}

/**
 * `test_mtx_init_vector_array_double()` tests creating a double vector in the
 * Matrix Market format.
 */
int test_mtx_init_vector_array_double(void)
{
    int err;
    struct mtx vector;
    int num_comment_lines = 1;
    const char * comment_lines[] = {"a comment"};
    double data[] = {1.0, 2.0, 3.0};
    int size = sizeof(data) / sizeof(*data);
    err = mtx_init_vector_array_double(
        &vector, num_comment_lines, comment_lines, size, data);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
    TEST_ASSERT_EQ(mtx_vector, vector.object);
    TEST_ASSERT_EQ(mtx_array, vector.format);
    TEST_ASSERT_EQ(mtx_double, vector.field);
    TEST_ASSERT_EQ(mtx_general, vector.symmetry);
    TEST_ASSERT_EQ(1, vector.num_comment_lines);
    TEST_ASSERT_STREQ("a comment", vector.comment_lines[0]);
    TEST_ASSERT_EQ(3, vector.num_rows);
    TEST_ASSERT_EQ(-1, vector.num_columns);
    TEST_ASSERT_EQ(3, vector.num_nonzeros);
    TEST_ASSERT_EQ(3, vector.size);
    TEST_ASSERT_EQ(1.0f, ((const double *) vector.data)[0]);
    TEST_ASSERT_EQ(2.0f, ((const double *) vector.data)[1]);
    TEST_ASSERT_EQ(3.0f, ((const double *) vector.data)[2]);
    if (!err)
        mtx_free(&vector);
    return TEST_SUCCESS;
}

/**
 * `test_mtx_init_vector_array_complex()` tests creating a complex vector in the
 * Matrix Market format.
 */
int test_mtx_init_vector_array_complex(void)
{
    int err;
    struct mtx vector;
    int num_comment_lines = 1;
    const char * comment_lines[] = {"a comment"};
    float data[] = {1.0f, 2.0, 3.0f, 4.0f};
    int size = sizeof(data) / (2*sizeof(*data));
    err = mtx_init_vector_array_complex(
        &vector, num_comment_lines, comment_lines, size, data);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
    TEST_ASSERT_EQ(mtx_vector, vector.object);
    TEST_ASSERT_EQ(mtx_array, vector.format);
    TEST_ASSERT_EQ(mtx_complex, vector.field);
    TEST_ASSERT_EQ(mtx_general, vector.symmetry);
    TEST_ASSERT_EQ(1, vector.num_comment_lines);
    TEST_ASSERT_STREQ("a comment", vector.comment_lines[0]);
    TEST_ASSERT_EQ(2, vector.num_rows);
    TEST_ASSERT_EQ(-1, vector.num_columns);
    TEST_ASSERT_EQ(2, vector.num_nonzeros);
    TEST_ASSERT_EQ(2, vector.size);
    TEST_ASSERT_EQ(1.0f, ((const float *) vector.data)[0]);
    TEST_ASSERT_EQ(2.0f, ((const float *) vector.data)[1]);
    TEST_ASSERT_EQ(3.0f, ((const float *) vector.data)[2]);
    TEST_ASSERT_EQ(4.0f, ((const float *) vector.data)[3]);
    if (!err)
        mtx_free(&vector);
    return TEST_SUCCESS;
}

/**
 * `test_mtx_init_vector_array_integer()` tests creating an integer vector in the
 * Matrix Market format.
 */
int test_mtx_init_vector_array_integer(void)
{
    int err;
    struct mtx vector;
    int num_comment_lines = 1;
    const char * comment_lines[] = {"a comment"};
    int data[] = {1, 2, 3};
    int size = sizeof(data) / sizeof(*data);
    err = mtx_init_vector_array_integer(
        &vector, num_comment_lines, comment_lines, size, data);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
    TEST_ASSERT_EQ(mtx_vector, vector.object);
    TEST_ASSERT_EQ(mtx_array, vector.format);
    TEST_ASSERT_EQ(mtx_integer, vector.field);
    TEST_ASSERT_EQ(mtx_general, vector.symmetry);
    TEST_ASSERT_EQ(1, vector.num_comment_lines);
    TEST_ASSERT_STREQ("a comment", vector.comment_lines[0]);
    TEST_ASSERT_EQ(3, vector.num_rows);
    TEST_ASSERT_EQ(-1, vector.num_columns);
    TEST_ASSERT_EQ(3, vector.num_nonzeros);
    TEST_ASSERT_EQ(3, vector.size);
    TEST_ASSERT_EQ(1, ((const int *) vector.data)[0]);
    TEST_ASSERT_EQ(2, ((const int *) vector.data)[1]);
    TEST_ASSERT_EQ(3, ((const int *) vector.data)[2]);
    if (!err)
        mtx_free(&vector);
    return TEST_SUCCESS;
}

/**
 * `main()' entry point and test driver.
 */
int main(int argc, char * argv[])
{
    TEST_SUITE_BEGIN("Running tests for vectors in Matrix Market format.\n");
    TEST_RUN(test_mtx_init_vector_array_real);
    TEST_RUN(test_mtx_init_vector_array_double);
    TEST_RUN(test_mtx_init_vector_array_complex);
    TEST_RUN(test_mtx_init_vector_array_integer);
    TEST_SUITE_END();
    return (TEST_SUITE_STATUS == TEST_SUCCESS) ?
        EXIT_SUCCESS : EXIT_FAILURE;
}
