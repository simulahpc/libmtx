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
#include <libmtx/matrix/array.h>
#include <libmtx/mtx/mtx.h>
#include <libmtx/mtx/header.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * `test_mtx_init_matrix_array_real_single()` tests creating dense
 * matrices with single precision, real coefficients in the Matrix
 * Market format.
 */
int test_mtx_init_matrix_array_real_single(void)
{
    int err;
    struct mtx mtx;
    int num_comment_lines = 1;
    const char * comment_lines[] = {"% a comment\n"};
    int num_rows = 2;
    int num_columns = 2;
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    size_t size = sizeof(data) / sizeof(*data);
    err = mtx_init_matrix_array_real_single(
        &mtx, mtx_general,
        mtx_nontriangular, mtx_row_major,
        num_comment_lines, comment_lines,
        num_rows, num_columns, size, data);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
    TEST_ASSERT_EQ(mtx_matrix, mtx.object);
    TEST_ASSERT_EQ(mtx_array, mtx.format);
    TEST_ASSERT_EQ(mtx_real, mtx.field);
    TEST_ASSERT_EQ(mtx_general, mtx.symmetry);
    TEST_ASSERT_EQ(1, mtx.num_comment_lines);
    TEST_ASSERT_STREQ("% a comment\n", mtx.comment_lines[0]);
    TEST_ASSERT_EQ(2, mtx.num_rows);
    TEST_ASSERT_EQ(2, mtx.num_columns);
    TEST_ASSERT_EQ(-1, mtx.num_nonzeros);

    const struct mtx_matrix_array_data * matrix_array =
        &mtx.storage.matrix_array;
    TEST_ASSERT_EQ(mtx_real, matrix_array->field);
    TEST_ASSERT_EQ(mtx_single, matrix_array->precision);
    TEST_ASSERT_EQ(mtx_general, matrix_array->symmetry);
    TEST_ASSERT_EQ(mtx_nontriangular, matrix_array->triangle);
    TEST_ASSERT_EQ(mtx_row_major, matrix_array->sorting);
    TEST_ASSERT_EQ(2, matrix_array->num_rows);
    TEST_ASSERT_EQ(2, matrix_array->num_columns);
    TEST_ASSERT_EQ(4, matrix_array->size);
    const float * mtxdata = matrix_array->data.real_single;
    TEST_ASSERT_EQ(1.0f, mtxdata[0]);
    TEST_ASSERT_EQ(2.0f, mtxdata[1]);
    TEST_ASSERT_EQ(3.0f, mtxdata[2]);
    TEST_ASSERT_EQ(4.0f, mtxdata[3]);
    mtx_free(&mtx);
    return TEST_SUCCESS;
}

/**
 * `test_mtx_init_matrix_array_real_single_symmetric()` tests creating
 * symmetric, dense matrices with real coefficients in the Matrix
 * Market format.
 */
int test_mtx_init_matrix_array_real_single_symmetric(void)
{
    int err;
    struct mtx mtx;
    int num_comment_lines = 0;
    const char * comment_lines[] = {};
    int num_rows = 2;
    int num_columns = 2;
    float data[] = {1.0f, 3.0f, 4.0f};
    size_t size = sizeof(data) / sizeof(*data);
    err = mtx_init_matrix_array_real_single(
        &mtx, mtx_symmetric,
        mtx_lower_triangular, mtx_row_major,
        num_comment_lines, comment_lines,
        num_rows, num_columns, size, data);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
    TEST_ASSERT_EQ(mtx_matrix, mtx.object);
    TEST_ASSERT_EQ(mtx_array, mtx.format);
    TEST_ASSERT_EQ(mtx_real, mtx.field);
    TEST_ASSERT_EQ(mtx_symmetric, mtx.symmetry);
    TEST_ASSERT_EQ(2, mtx.num_rows);
    TEST_ASSERT_EQ(2, mtx.num_columns);
    TEST_ASSERT_EQ(-1, mtx.num_nonzeros);

    const struct mtx_matrix_array_data * matrix_array =
        &mtx.storage.matrix_array;
    TEST_ASSERT_EQ(mtx_real, matrix_array->field);
    TEST_ASSERT_EQ(mtx_single, matrix_array->precision);
    TEST_ASSERT_EQ(mtx_symmetric, matrix_array->symmetry);
    TEST_ASSERT_EQ(mtx_lower_triangular, matrix_array->triangle);
    TEST_ASSERT_EQ(mtx_row_major, matrix_array->sorting);
    TEST_ASSERT_EQ(2, matrix_array->num_rows);
    TEST_ASSERT_EQ(2, matrix_array->num_columns);
    TEST_ASSERT_EQ(3, matrix_array->size);
    const float * mtxdata = matrix_array->data.real_single;
    TEST_ASSERT_EQ(1.0f, mtxdata[0]);
    TEST_ASSERT_EQ(3.0f, mtxdata[1]);
    TEST_ASSERT_EQ(4.0f, mtxdata[2]);
    mtx_free(&mtx);
    return TEST_SUCCESS;
}

/**
 * `test_mtx_init_matrix_array_real_single_skew_symmetric()` tests
 * creating skew-symmetric, dense matrices with real coefficients in
 * the Matrix Market format.
 */
int test_mtx_init_matrix_array_real_single_skew_symmetric(void)
{
    int err;
    struct mtx mtx;
    int num_comment_lines = 0;
    const char * comment_lines[] = {};
    int num_rows = 3;
    int num_columns = 3;
    float data[] = {1.0f, 2.0f, 4.0f};
    size_t size = sizeof(data) / sizeof(*data);
    err = mtx_init_matrix_array_real_single(
        &mtx, mtx_skew_symmetric,
        mtx_strict_lower_triangular, mtx_row_major,
        num_comment_lines, comment_lines,
        num_rows, num_columns, size, data);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
    TEST_ASSERT_EQ(mtx_matrix, mtx.object);
    TEST_ASSERT_EQ(mtx_array, mtx.format);
    TEST_ASSERT_EQ(mtx_real, mtx.field);
    TEST_ASSERT_EQ(mtx_skew_symmetric, mtx.symmetry);
    TEST_ASSERT_EQ(3, mtx.num_rows);
    TEST_ASSERT_EQ(3, mtx.num_columns);
    TEST_ASSERT_EQ(-1, mtx.num_nonzeros);

    const struct mtx_matrix_array_data * matrix_array =
        &mtx.storage.matrix_array;
    TEST_ASSERT_EQ(mtx_real, matrix_array->field);
    TEST_ASSERT_EQ(mtx_single, matrix_array->precision);
    TEST_ASSERT_EQ(mtx_skew_symmetric, matrix_array->symmetry);
    TEST_ASSERT_EQ(mtx_strict_lower_triangular, matrix_array->triangle);
    TEST_ASSERT_EQ(mtx_row_major, matrix_array->sorting);
    TEST_ASSERT_EQ(3, matrix_array->num_rows);
    TEST_ASSERT_EQ(3, matrix_array->num_columns);
    TEST_ASSERT_EQ(3, matrix_array->size);
    const float * mtxdata = matrix_array->data.real_single;
    TEST_ASSERT_EQ(1.0f, mtxdata[0]);
    TEST_ASSERT_EQ(2.0f, mtxdata[1]);
    TEST_ASSERT_EQ(4.0f, mtxdata[2]);
    mtx_free(&mtx);
    return TEST_SUCCESS;
}

/**
 * `test_mtx_init_matrix_array_real_double()` tests creating dense
 * matrices with double precision, real coefficients in the Matrix
 * Market format.
 */
int test_mtx_init_matrix_array_real_double(void)
{
    int err;
    struct mtx mtx;
    int num_comment_lines = 0;
    const char * comment_lines[] = {};
    int num_rows = 2;
    int num_columns = 2;
    double data[] = {1.0, 2.0, 3.0, 4.0};
    size_t size = sizeof(data) / sizeof(*data);
    err = mtx_init_matrix_array_real_double(
        &mtx, mtx_general, mtx_nontriangular, mtx_row_major,
        num_comment_lines, comment_lines,
        num_rows, num_columns, size, data);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
    TEST_ASSERT_EQ(mtx_matrix, mtx.object);
    TEST_ASSERT_EQ(mtx_array, mtx.format);
    TEST_ASSERT_EQ(mtx_real, mtx.field);
    TEST_ASSERT_EQ(mtx_general, mtx.symmetry);
    TEST_ASSERT_EQ(2, mtx.num_rows);
    TEST_ASSERT_EQ(2, mtx.num_columns);
    TEST_ASSERT_EQ(-1, mtx.num_nonzeros);

    const struct mtx_matrix_array_data * matrix_array =
        &mtx.storage.matrix_array;
    TEST_ASSERT_EQ(mtx_real, matrix_array->field);
    TEST_ASSERT_EQ(mtx_double, matrix_array->precision);
    TEST_ASSERT_EQ(mtx_general, matrix_array->symmetry);
    TEST_ASSERT_EQ(mtx_nontriangular, matrix_array->triangle);
    TEST_ASSERT_EQ(mtx_row_major, matrix_array->sorting);
    TEST_ASSERT_EQ(2, matrix_array->num_rows);
    TEST_ASSERT_EQ(2, matrix_array->num_columns);
    TEST_ASSERT_EQ(4, matrix_array->size);
    const double * mtxdata = matrix_array->data.real_double;
    TEST_ASSERT_EQ(1.0, mtxdata[0]);
    TEST_ASSERT_EQ(2.0, mtxdata[1]);
    TEST_ASSERT_EQ(3.0, mtxdata[2]);
    TEST_ASSERT_EQ(4.0, mtxdata[3]);
    mtx_free(&mtx);
    return TEST_SUCCESS;
}

/**
 * `test_mtx_init_matrix_array_complex_single()` tests creating dense
 * matrices with single precision, complex coefficients in the Matrix
 * Market format.
 */
int test_mtx_init_matrix_array_complex_single(void)
{
    int err;
    struct mtx mtx;
    int num_comment_lines = 0;
    const char * comment_lines[] = {};
    int num_rows = 2;
    int num_columns = 2;
    float data[][2] = {{1.0,2.0}, {3.0,4.0}, {5.0,6.0}, {7.0,8.0}};
    size_t size = sizeof(data) / sizeof(*data);
    err = mtx_init_matrix_array_complex_single(
        &mtx, mtx_general, mtx_nontriangular, mtx_row_major,
        num_comment_lines, comment_lines,
        num_rows, num_columns, size, data);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
    TEST_ASSERT_EQ(mtx_matrix, mtx.object);
    TEST_ASSERT_EQ(mtx_array, mtx.format);
    TEST_ASSERT_EQ(mtx_complex, mtx.field);
    TEST_ASSERT_EQ(mtx_general, mtx.symmetry);
    TEST_ASSERT_EQ(2, mtx.num_rows);
    TEST_ASSERT_EQ(2, mtx.num_columns);
    TEST_ASSERT_EQ(-1, mtx.num_nonzeros);

    const struct mtx_matrix_array_data * matrix_array =
        &mtx.storage.matrix_array;
    TEST_ASSERT_EQ(mtx_complex, matrix_array->field);
    TEST_ASSERT_EQ(mtx_single, matrix_array->precision);
    TEST_ASSERT_EQ(mtx_general, matrix_array->symmetry);
    TEST_ASSERT_EQ(mtx_nontriangular, matrix_array->triangle);
    TEST_ASSERT_EQ(mtx_row_major, matrix_array->sorting);
    TEST_ASSERT_EQ(2, matrix_array->num_rows);
    TEST_ASSERT_EQ(2, matrix_array->num_columns);
    TEST_ASSERT_EQ(4, matrix_array->size);
    const float (* mtxdata)[2] = matrix_array->data.complex_single;
    TEST_ASSERT_EQ(1.0, mtxdata[0][0]); TEST_ASSERT_EQ(2.0, mtxdata[0][1]);
    TEST_ASSERT_EQ(3.0, mtxdata[1][0]); TEST_ASSERT_EQ(4.0, mtxdata[1][1]);
    TEST_ASSERT_EQ(5.0, mtxdata[2][0]); TEST_ASSERT_EQ(6.0, mtxdata[2][1]);
    TEST_ASSERT_EQ(7.0, mtxdata[3][0]); TEST_ASSERT_EQ(8.0, mtxdata[3][1]);
    mtx_free(&mtx);
    return TEST_SUCCESS;
}

/**
 * `test_mtx_init_matrix_array_integer_single()` tests creating dense
 * matrices with single precision, integer coefficients in the Matrix
 * Market format.
 */
int test_mtx_init_matrix_array_integer_single(void)
{
    int err;
    struct mtx mtx;
    int num_comment_lines = 0;
    const char * comment_lines[] = {};
    int num_rows = 2;
    int num_columns = 2;
    int32_t data[] = {1, 2, 3, 4};
    size_t size = sizeof(data) / sizeof(*data);
    err = mtx_init_matrix_array_integer_single(
        &mtx, mtx_general, mtx_nontriangular, mtx_row_major,
        num_comment_lines, comment_lines,
        num_rows, num_columns, size, data);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
    TEST_ASSERT_EQ(mtx_matrix, mtx.object);
    TEST_ASSERT_EQ(mtx_array, mtx.format);
    TEST_ASSERT_EQ(mtx_integer, mtx.field);
    TEST_ASSERT_EQ(mtx_general, mtx.symmetry);
    TEST_ASSERT_EQ(2, mtx.num_rows);
    TEST_ASSERT_EQ(2, mtx.num_columns);
    TEST_ASSERT_EQ(-1, mtx.num_nonzeros);

    const struct mtx_matrix_array_data * matrix_array =
        &mtx.storage.matrix_array;
    TEST_ASSERT_EQ(mtx_integer, matrix_array->field);
    TEST_ASSERT_EQ(mtx_single, matrix_array->precision);
    TEST_ASSERT_EQ(mtx_general, matrix_array->symmetry);
    TEST_ASSERT_EQ(mtx_nontriangular, matrix_array->triangle);
    TEST_ASSERT_EQ(mtx_row_major, matrix_array->sorting);
    TEST_ASSERT_EQ(2, matrix_array->num_rows);
    TEST_ASSERT_EQ(2, matrix_array->num_columns);
    TEST_ASSERT_EQ(4, matrix_array->size);
    const int32_t * mtxdata = matrix_array->data.integer_single;
    TEST_ASSERT_EQ(1, mtxdata[0]);
    TEST_ASSERT_EQ(2, mtxdata[1]);
    TEST_ASSERT_EQ(3, mtxdata[2]);
    TEST_ASSERT_EQ(4, mtxdata[3]);
    mtx_free(&mtx);
    return TEST_SUCCESS;
}

/**
 * `test_mtx_set_zero_matrix_array_real_single()` tests zeroing a
 * dense, real matrix in Matrix Market format.
 */
int test_mtx_set_zero_matrix_array_real_single(void)
{
    int err;
    struct mtx mtx;
    int num_comment_lines = 0;
    const char * comment_lines[] = {};
    int num_rows = 2;
    int num_columns = 2;
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    size_t size = sizeof(data) / sizeof(*data);
    err = mtx_init_matrix_array_real_single(
        &mtx, mtx_general, mtx_nontriangular, mtx_row_major,
        num_comment_lines, comment_lines,
        num_rows, num_columns, size, data);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

    err = mtx_set_zero(&mtx);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
    const struct mtx_matrix_array_data * matrix_array =
        &mtx.storage.matrix_array;
    TEST_ASSERT_EQ(mtx_real, matrix_array->field);
    TEST_ASSERT_EQ(mtx_single, matrix_array->precision);
    TEST_ASSERT_EQ(mtx_general, matrix_array->symmetry);
    TEST_ASSERT_EQ(mtx_nontriangular, matrix_array->triangle);
    TEST_ASSERT_EQ(mtx_row_major, matrix_array->sorting);
    TEST_ASSERT_EQ(2, matrix_array->num_rows);
    TEST_ASSERT_EQ(2, matrix_array->num_columns);
    TEST_ASSERT_EQ(4, matrix_array->size);
    const float * mtxdata = matrix_array->data.real_single;
    TEST_ASSERT_EQ(0.0f, mtxdata[0]);
    TEST_ASSERT_EQ(0.0f, mtxdata[1]);
    TEST_ASSERT_EQ(0.0f, mtxdata[2]);
    TEST_ASSERT_EQ(0.0f, mtxdata[3]);
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
    TEST_RUN(test_mtx_init_matrix_array_real_single);
    TEST_RUN(test_mtx_init_matrix_array_real_single_symmetric);
    TEST_RUN(test_mtx_init_matrix_array_real_single_skew_symmetric);
    TEST_RUN(test_mtx_init_matrix_array_real_double);
    TEST_RUN(test_mtx_init_matrix_array_complex_single);
    TEST_RUN(test_mtx_init_matrix_array_integer_single);
    TEST_RUN(test_mtx_set_zero_matrix_array_real_single);
    TEST_SUITE_END();
    return (TEST_SUITE_STATUS == TEST_SUCCESS) ?
        EXIT_SUCCESS : EXIT_FAILURE;
}
