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
 * Unit tests for dense matrices in Matrix Market format.
 */

#include "test.h"

#include <libmtx/error.h>
#include <libmtx/matrix/array/array.h>
#include <libmtx/mtx.h>
#include <libmtx/header.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * `test_mtx_init_matrix_array_real()` tests creating dense matrices with
 * real coefficients in the Matrix Market format.
 */
int test_mtx_init_matrix_array_real(void)
{
    int err;
    struct mtx mtx;
    int num_comment_lines = 1;
    const char * comment_lines[] = {"% a comment"};
    int num_rows = 2;
    int num_columns = 2;
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    err = mtx_init_matrix_array_real(
        &mtx, mtx_general,
        mtx_nontriangular, mtx_row_major,
        num_comment_lines, comment_lines,
        num_rows, num_columns, data);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
    TEST_ASSERT_EQ(mtx_matrix, mtx.object);
    TEST_ASSERT_EQ(mtx_array, mtx.format);
    TEST_ASSERT_EQ(mtx_real, mtx.field);
    TEST_ASSERT_EQ(mtx_general, mtx.symmetry);
    TEST_ASSERT_EQ(mtx_nontriangular, mtx.triangle);
    TEST_ASSERT_EQ(mtx_row_major, mtx.sorting);
    TEST_ASSERT_EQ(mtx_unordered, mtx.ordering);
    TEST_ASSERT_EQ(mtx_assembled, mtx.assembly);
    TEST_ASSERT_EQ(1, mtx.num_comment_lines);
    TEST_ASSERT_STREQ("% a comment", mtx.comment_lines[0]);
    TEST_ASSERT_EQ(2, mtx.num_rows);
    TEST_ASSERT_EQ(2, mtx.num_columns);
    TEST_ASSERT_EQ(4, mtx.num_nonzeros);
    TEST_ASSERT_EQ(4, mtx.size);
    TEST_ASSERT_EQ(sizeof(float), mtx.nonzero_size);
    TEST_ASSERT_EQ(1.0f, ((const float *) mtx.data)[0]);
    TEST_ASSERT_EQ(2.0f, ((const float *) mtx.data)[1]);
    TEST_ASSERT_EQ(3.0f, ((const float *) mtx.data)[2]);
    TEST_ASSERT_EQ(4.0f, ((const float *) mtx.data)[3]);
    if (!err)
        mtx_free(&mtx);
    return TEST_SUCCESS;
}

/**
 * `test_mtx_init_matrix_array_real_symmetric()` tests creating symmetric,
 * dense matrices with real coefficients in the Matrix Market format.
 */
int test_mtx_init_matrix_array_real_symmetric(void)
{
    int err;
    struct mtx mtx;
    int num_comment_lines = 1;
    const char * comment_lines[] = {"% a comment"};
    int num_rows = 2;
    int num_columns = 2;
    float data[] = {1.0f, 3.0f, 4.0f};
    err = mtx_init_matrix_array_real(
        &mtx, mtx_symmetric,
        mtx_lower_triangular, mtx_row_major,
        num_comment_lines, comment_lines,
        num_rows, num_columns, data);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
    TEST_ASSERT_EQ(mtx_matrix, mtx.object);
    TEST_ASSERT_EQ(mtx_array, mtx.format);
    TEST_ASSERT_EQ(mtx_real, mtx.field);
    TEST_ASSERT_EQ(mtx_symmetric, mtx.symmetry);
    TEST_ASSERT_EQ(mtx_lower_triangular, mtx.triangle);
    TEST_ASSERT_EQ(mtx_row_major, mtx.sorting);
    TEST_ASSERT_EQ(mtx_unordered, mtx.ordering);
    TEST_ASSERT_EQ(mtx_assembled, mtx.assembly);
    TEST_ASSERT_EQ(1, mtx.num_comment_lines);
    TEST_ASSERT_STREQ("% a comment", mtx.comment_lines[0]);
    TEST_ASSERT_EQ(2, mtx.num_rows);
    TEST_ASSERT_EQ(2, mtx.num_columns);
    TEST_ASSERT_EQ(4, mtx.num_nonzeros);
    TEST_ASSERT_EQ(3, mtx.size);
    TEST_ASSERT_EQ(sizeof(float), mtx.nonzero_size);
    TEST_ASSERT_EQ(1.0f, ((const float *) mtx.data)[0]);
    TEST_ASSERT_EQ(3.0f, ((const float *) mtx.data)[1]);
    TEST_ASSERT_EQ(4.0f, ((const float *) mtx.data)[2]);
    if (!err)
        mtx_free(&mtx);
    return TEST_SUCCESS;
}

/**
 * `test_mtx_init_matrix_array_real_skew_symmetric()` tests creating
 * skew-symmetric, dense matrices with real coefficients in the Matrix
 * Market format.
 */
int test_mtx_init_matrix_array_real_skew_symmetric(void)
{
    int err;
    struct mtx mtx;
    int num_comment_lines = 1;
    const char * comment_lines[] = {"% a comment"};
    int num_rows = 3;
    int num_columns = 3;
    float data[] = {1.0f, 2.0f, 4.0f};
    err = mtx_init_matrix_array_real(
        &mtx, mtx_skew_symmetric,
        mtx_lower_triangular, mtx_row_major,
        num_comment_lines, comment_lines,
        num_rows, num_columns, data);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
    TEST_ASSERT_EQ(mtx_matrix, mtx.object);
    TEST_ASSERT_EQ(mtx_array, mtx.format);
    TEST_ASSERT_EQ(mtx_real, mtx.field);
    TEST_ASSERT_EQ(mtx_skew_symmetric, mtx.symmetry);
    TEST_ASSERT_EQ(mtx_lower_triangular, mtx.triangle);
    TEST_ASSERT_EQ(mtx_row_major, mtx.sorting);
    TEST_ASSERT_EQ(mtx_unordered, mtx.ordering);
    TEST_ASSERT_EQ(mtx_assembled, mtx.assembly);
    TEST_ASSERT_EQ(1, mtx.num_comment_lines);
    TEST_ASSERT_STREQ("% a comment", mtx.comment_lines[0]);
    TEST_ASSERT_EQ(3, mtx.num_rows);
    TEST_ASSERT_EQ(3, mtx.num_columns);
    TEST_ASSERT_EQ(9, mtx.num_nonzeros);
    TEST_ASSERT_EQ(3, mtx.size);
    TEST_ASSERT_EQ(sizeof(float), mtx.nonzero_size);
    TEST_ASSERT_EQ(1.0f, ((const float *) mtx.data)[0]);
    TEST_ASSERT_EQ(2.0f, ((const float *) mtx.data)[1]);
    TEST_ASSERT_EQ(4.0f, ((const float *) mtx.data)[2]);
    if (!err)
        mtx_free(&mtx);
    return TEST_SUCCESS;
}

/**
 * `test_mtx_init_matrix_array_double()` tests creating dense matrices with
 * double-precision real coefficients in the Matrix Market format.
 */
int test_mtx_init_matrix_array_double(void)
{
    int err;
    struct mtx mtx;
    int num_comment_lines = 1;
    const char * comment_lines[] = {"% a comment"};
    int num_rows = 2;
    int num_columns = 2;
    double data[] = {1.0, 2.0, 3.0, 4.0};
    err = mtx_init_matrix_array_double(
        &mtx, mtx_general,
        mtx_nontriangular, mtx_row_major,
        num_comment_lines, comment_lines,
        num_rows, num_columns, data);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
    TEST_ASSERT_EQ(mtx_matrix, mtx.object);
    TEST_ASSERT_EQ(mtx_array, mtx.format);
    TEST_ASSERT_EQ(mtx_double, mtx.field);
    TEST_ASSERT_EQ(mtx_general, mtx.symmetry);
    TEST_ASSERT_EQ(mtx_nontriangular, mtx.triangle);
    TEST_ASSERT_EQ(mtx_row_major, mtx.sorting);
    TEST_ASSERT_EQ(mtx_unordered, mtx.ordering);
    TEST_ASSERT_EQ(mtx_assembled, mtx.assembly);
    TEST_ASSERT_EQ(1, mtx.num_comment_lines);
    TEST_ASSERT_STREQ("% a comment", mtx.comment_lines[0]);
    TEST_ASSERT_EQ(2, mtx.num_rows);
    TEST_ASSERT_EQ(2, mtx.num_columns);
    TEST_ASSERT_EQ(4, mtx.num_nonzeros);
    TEST_ASSERT_EQ(4, mtx.size);
    TEST_ASSERT_EQ(sizeof(double), mtx.nonzero_size);
    TEST_ASSERT_EQ(1.0, ((const double *) mtx.data)[0]);
    TEST_ASSERT_EQ(2.0, ((const double *) mtx.data)[1]);
    TEST_ASSERT_EQ(3.0, ((const double *) mtx.data)[2]);
    TEST_ASSERT_EQ(4.0, ((const double *) mtx.data)[3]);
    if (!err)
        mtx_free(&mtx);
    return TEST_SUCCESS;
}

/**
 * `test_mtx_init_matrix_array_complex()` tests creating dense matrices with
 * single-precision complex coefficients in the Matrix Market format.
 */
int test_mtx_init_matrix_array_complex(void)
{
    int err;
    struct mtx mtx;
    int num_comment_lines = 1;
    const char * comment_lines[] = {"% a comment"};
    int num_rows = 2;
    int num_columns = 2;
    float data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    err = mtx_init_matrix_array_complex(
        &mtx, mtx_general,
        mtx_nontriangular, mtx_row_major,
        num_comment_lines, comment_lines,
        num_rows, num_columns, data);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
    TEST_ASSERT_EQ(mtx_matrix, mtx.object);
    TEST_ASSERT_EQ(mtx_array, mtx.format);
    TEST_ASSERT_EQ(mtx_complex, mtx.field);
    TEST_ASSERT_EQ(mtx_general, mtx.symmetry);
    TEST_ASSERT_EQ(mtx_nontriangular, mtx.triangle);
    TEST_ASSERT_EQ(mtx_row_major, mtx.sorting);
    TEST_ASSERT_EQ(mtx_unordered, mtx.ordering);
    TEST_ASSERT_EQ(mtx_assembled, mtx.assembly);
    TEST_ASSERT_EQ(1, mtx.num_comment_lines);
    TEST_ASSERT_STREQ("% a comment", mtx.comment_lines[0]);
    TEST_ASSERT_EQ(2, mtx.num_rows);
    TEST_ASSERT_EQ(2, mtx.num_columns);
    TEST_ASSERT_EQ(4, mtx.num_nonzeros);
    TEST_ASSERT_EQ(4, mtx.size);
    TEST_ASSERT_EQ(2*sizeof(float), mtx.nonzero_size);
    TEST_ASSERT_EQ(1.0, ((const float *) mtx.data)[0]);
    TEST_ASSERT_EQ(2.0, ((const float *) mtx.data)[1]);
    TEST_ASSERT_EQ(3.0, ((const float *) mtx.data)[2]);
    TEST_ASSERT_EQ(4.0, ((const float *) mtx.data)[3]);
    TEST_ASSERT_EQ(5.0, ((const float *) mtx.data)[4]);
    TEST_ASSERT_EQ(6.0, ((const float *) mtx.data)[5]);
    TEST_ASSERT_EQ(7.0, ((const float *) mtx.data)[6]);
    TEST_ASSERT_EQ(8.0, ((const float *) mtx.data)[7]);
    if (!err)
        mtx_free(&mtx);
    return TEST_SUCCESS;
}

/**
 * `test_mtx_init_matrix_array_integer()` tests creating dense matrices with
 * integer coefficients in the Matrix Market format.
 */
int test_mtx_init_matrix_array_integer(void)
{
    int err;
    struct mtx mtx;
    int num_comment_lines = 1;
    const char * comment_lines[] = {"% a comment"};
    int num_rows = 2;
    int num_columns = 2;
    int data[] = {1, 2, 3, 4};
    err = mtx_init_matrix_array_integer(
        &mtx, mtx_general,
        mtx_nontriangular, mtx_row_major,
        num_comment_lines, comment_lines,
        num_rows, num_columns, data);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
    TEST_ASSERT_EQ(mtx_matrix, mtx.object);
    TEST_ASSERT_EQ(mtx_array, mtx.format);
    TEST_ASSERT_EQ(mtx_integer, mtx.field);
    TEST_ASSERT_EQ(mtx_general, mtx.symmetry);
    TEST_ASSERT_EQ(mtx_nontriangular, mtx.triangle);
    TEST_ASSERT_EQ(mtx_row_major, mtx.sorting);
    TEST_ASSERT_EQ(mtx_unordered, mtx.ordering);
    TEST_ASSERT_EQ(mtx_assembled, mtx.assembly);
    TEST_ASSERT_EQ(1, mtx.num_comment_lines);
    TEST_ASSERT_STREQ("% a comment", mtx.comment_lines[0]);
    TEST_ASSERT_EQ(2, mtx.num_rows);
    TEST_ASSERT_EQ(2, mtx.num_columns);
    TEST_ASSERT_EQ(4, mtx.num_nonzeros);
    TEST_ASSERT_EQ(4, mtx.size);
    TEST_ASSERT_EQ(sizeof(int), mtx.nonzero_size);
    TEST_ASSERT_EQ(1, ((const int *) mtx.data)[0]);
    TEST_ASSERT_EQ(2, ((const int *) mtx.data)[1]);
    TEST_ASSERT_EQ(3, ((const int *) mtx.data)[2]);
    TEST_ASSERT_EQ(4, ((const int *) mtx.data)[3]);
    if (!err)
        mtx_free(&mtx);
    return TEST_SUCCESS;
}

/**
 * `main()' entry point and test driver.
 */
int main(int argc, char * argv[])
{
    TEST_SUITE_BEGIN(
        "Running tests for dense matrices in Matrix Market format.\n");
    TEST_RUN(test_mtx_init_matrix_array_real);
    TEST_RUN(test_mtx_init_matrix_array_real_symmetric);
    TEST_RUN(test_mtx_init_matrix_array_real_skew_symmetric);
    TEST_RUN(test_mtx_init_matrix_array_double);
    TEST_RUN(test_mtx_init_matrix_array_complex);
    TEST_RUN(test_mtx_init_matrix_array_integer);
    TEST_SUITE_END();
    return (TEST_SUITE_STATUS == TEST_SUCCESS) ?
        EXIT_SUCCESS : EXIT_FAILURE;
}
