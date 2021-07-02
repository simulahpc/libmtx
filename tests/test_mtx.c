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
 * Unit tests for Matrix Market I/O.
 */

#include "test.h"

#include <matrixmarket/error.h>
#include <matrixmarket/io.h>
#include <matrixmarket/matrix.h>
#include <matrixmarket/matrix_coordinate.h>
#include <matrixmarket/mtx.h>

#include <errno.h>
#include <unistd.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * `test_mtx_copy()' tests copying an `mtx' object.
 */
int test_mtx_copy(void)
{
    int err;

    /* Create a sparse matrix. */
    struct mtx srcmtx;
    int num_comment_lines = 1;
    const char * comment_lines[] = {"a comment"};
    int num_rows = 4;
    int num_columns = 4;
    int64_t size = 6;
    const struct mtx_matrix_coordinate_real data[] = {
        {1,1,1.0f}, {1,4,2.0f},
        {2,2,3.0f},
        {3,3,4.0f},
        {4,1,5.0f}, {4,4,6.0f}};
    err = mtx_init_matrix_coordinate_real(
        &srcmtx, mtx_general, mtx_unsorted, mtx_unordered, mtx_unassembled,
        num_comment_lines, comment_lines,
        num_rows, num_columns, size, data);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));

    /* Make a copy and verify the copied contents. */
    struct mtx destmtx;
    err = mtx_copy(&destmtx, &srcmtx);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
    TEST_ASSERT_EQ(mtx_matrix, destmtx.object);
    TEST_ASSERT_EQ(mtx_coordinate, destmtx.format);
    TEST_ASSERT_EQ(mtx_real, destmtx.field);
    TEST_ASSERT_EQ(mtx_general, destmtx.symmetry);
    TEST_ASSERT_EQ(mtx_unsorted, destmtx.sorting);
    TEST_ASSERT_EQ(mtx_unordered, destmtx.ordering);
    TEST_ASSERT_EQ(mtx_unassembled, destmtx.assembly);
    TEST_ASSERT_EQ(1, destmtx.num_comment_lines);
    TEST_ASSERT_STREQ("a comment", destmtx.comment_lines[0]);
    TEST_ASSERT_EQ(4, destmtx.num_rows);
    TEST_ASSERT_EQ(4, destmtx.num_columns);
    TEST_ASSERT_EQ(6, destmtx.num_nonzeros);
    TEST_ASSERT_EQ(6, destmtx.size);
    TEST_ASSERT_EQ(sizeof(struct mtx_matrix_coordinate_real), destmtx.nonzero_size);
    TEST_ASSERT_EQ(   1, ((const struct mtx_matrix_coordinate_real *) destmtx.data)[0].i);
    TEST_ASSERT_EQ(   1, ((const struct mtx_matrix_coordinate_real *) destmtx.data)[0].j);
    TEST_ASSERT_EQ(1.0f, ((const struct mtx_matrix_coordinate_real *) destmtx.data)[0].a);
    TEST_ASSERT_EQ(   1, ((const struct mtx_matrix_coordinate_real *) destmtx.data)[1].i);
    TEST_ASSERT_EQ(   4, ((const struct mtx_matrix_coordinate_real *) destmtx.data)[1].j);
    TEST_ASSERT_EQ(2.0f, ((const struct mtx_matrix_coordinate_real *) destmtx.data)[1].a);
    TEST_ASSERT_EQ(   2, ((const struct mtx_matrix_coordinate_real *) destmtx.data)[2].i);
    TEST_ASSERT_EQ(   2, ((const struct mtx_matrix_coordinate_real *) destmtx.data)[2].j);
    TEST_ASSERT_EQ(3.0f, ((const struct mtx_matrix_coordinate_real *) destmtx.data)[2].a);
    TEST_ASSERT_EQ(   3, ((const struct mtx_matrix_coordinate_real *) destmtx.data)[3].i);
    TEST_ASSERT_EQ(   3, ((const struct mtx_matrix_coordinate_real *) destmtx.data)[3].j);
    TEST_ASSERT_EQ(4.0f, ((const struct mtx_matrix_coordinate_real *) destmtx.data)[3].a);
    TEST_ASSERT_EQ(   4, ((const struct mtx_matrix_coordinate_real *) destmtx.data)[4].i);
    TEST_ASSERT_EQ(   1, ((const struct mtx_matrix_coordinate_real *) destmtx.data)[4].j);
    TEST_ASSERT_EQ(5.0f, ((const struct mtx_matrix_coordinate_real *) destmtx.data)[4].a);
    TEST_ASSERT_EQ(   4, ((const struct mtx_matrix_coordinate_real *) destmtx.data)[5].i);
    TEST_ASSERT_EQ(   4, ((const struct mtx_matrix_coordinate_real *) destmtx.data)[5].j);
    TEST_ASSERT_EQ(6.0f, ((const struct mtx_matrix_coordinate_real *) destmtx.data)[5].a);
    mtx_free(&destmtx);
    mtx_free(&srcmtx);
    return MTX_SUCCESS;
}

/**
 * `test_mtx_matrix_size_per_row()' tests counting the number of
 * matrix entries in each row of a matrix.
 */
int test_mtx_matrix_size_per_row(void)
{
    int err;

    /* Create a sparse matrix. */
    struct mtx mtx;
    int num_comment_lines = 1;
    const char * comment_lines[] = {"a comment"};
    int num_rows = 4;
    int num_columns = 4;
    int64_t size = 6;
    const struct mtx_matrix_coordinate_real data[] = {
        {1,1,1.0f}, {1,4,2.0f},
        {2,2,3.0f},
        {3,3,4.0f},
        {4,1,5.0f}, {4,4,6.0f}};
    err = mtx_init_matrix_coordinate_real(
        &mtx, mtx_general, mtx_unsorted, mtx_unordered, mtx_unassembled,
        num_comment_lines, comment_lines,
        num_rows, num_columns, size, data);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));

    /* Compute and check the size of each row. */
    int size_per_row[4];
    err = mtx_matrix_size_per_row(&mtx, size_per_row);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
    TEST_ASSERT_EQ(2, size_per_row[0]);
    TEST_ASSERT_EQ(1, size_per_row[1]);
    TEST_ASSERT_EQ(1, size_per_row[2]);
    TEST_ASSERT_EQ(2, size_per_row[3]);
    mtx_free(&mtx);
    return MTX_SUCCESS;
}

/**
 * `test_mtx_matrix_num_nonzeros()' tests computing the number of nonzeros
 * of a matrix or vector.
 */
int test_mtx_matrix_num_nonzeros(void)
{
    /* Dense matrices. */
    {
        int num_rows = 3;
        int num_columns = 3;
        int64_t size = 9;
        const float data[] = {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f};
        int64_t num_nonzeros;
        int err = mtx_matrix_num_nonzeros(
            mtx_matrix, mtx_array, mtx_real, mtx_general,
            num_rows, num_columns, size, data, &num_nonzeros);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9, num_nonzeros);
    }
    {
        int num_rows = 3;
        int num_columns = 3;
        int64_t size = 6;
        const float data[] = {0.f, 1.f, 2.f, 3.f, 4.f, 5.f};
        int64_t num_nonzeros;
        int err = mtx_matrix_num_nonzeros(
            mtx_matrix, mtx_array, mtx_real, mtx_symmetric,
            num_rows, num_columns, size, data, &num_nonzeros);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9, num_nonzeros);
    }
    {
        int num_rows = 3;
        int num_columns = 3;
        int64_t size = 3;
        const float data[] = {0.f, 1.f, 2.f};
        int64_t num_nonzeros;
        int err = mtx_matrix_num_nonzeros(
            mtx_matrix, mtx_array, mtx_real, mtx_skew_symmetric,
            num_rows, num_columns, size, data, &num_nonzeros);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9, num_nonzeros);
    }
    {
        int num_rows = 3;
        int num_columns = 3;
        int64_t size = 6;
        const float data[] = {0.f,0.f, 1.f,1.f, 2.f,2.f, 3.f,3.f, 4.f,4.f, 5.f,5.f};
        int64_t num_nonzeros;
        int err = mtx_matrix_num_nonzeros(
            mtx_matrix, mtx_array, mtx_complex, mtx_hermitian,
            num_rows, num_columns, size, data, &num_nonzeros);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9, num_nonzeros);
    }

    /* Sparse matrices. */
    {
        int num_rows = 3;
        int num_columns = 3;
        int64_t size = 5;
        const struct mtx_matrix_coordinate_real data[] = {
            {1,1,1.f}, {1,3,2.f}, {2,2,3.f}, {3,1,2.f}, {3,3,4.f}};
        int64_t num_nonzeros;
        int err = mtx_matrix_num_nonzeros(
            mtx_matrix, mtx_coordinate, mtx_real, mtx_general,
            num_rows, num_columns, size, data, &num_nonzeros);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(5, num_nonzeros);
    }
    {
        int num_rows = 3;
        int num_columns = 3;
        int64_t size = 4;
        const struct mtx_matrix_coordinate_real data[] = {
            {1,1,1.f}, {1,3,2.f}, {2,2,3.f}, {3,3,4.f}};
        int64_t num_nonzeros;
        int err = mtx_matrix_num_nonzeros(
            mtx_matrix, mtx_coordinate, mtx_real, mtx_symmetric,
            num_rows, num_columns, size, data, &num_nonzeros);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(5, num_nonzeros);
    }
    {
        int num_rows = 3;
        int num_columns = 3;
        int64_t size = 1;
        const struct mtx_matrix_coordinate_real data[] = {
            {1,3,2.f}};
        int64_t num_nonzeros;
        int err = mtx_matrix_num_nonzeros(
            mtx_matrix, mtx_coordinate, mtx_real, mtx_skew_symmetric,
            num_rows, num_columns, size, data, &num_nonzeros);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(2, num_nonzeros);
    }

    return TEST_SUCCESS;
}

/**
 * `main()' entry point and test driver.
 */
int main(int argc, char * argv[])
{
    TEST_SUITE_BEGIN("Running tests for Matrix Market objects\n");
    TEST_RUN(test_mtx_copy);
    TEST_RUN(test_mtx_matrix_size_per_row);
    TEST_RUN(test_mtx_matrix_num_nonzeros);
    TEST_SUITE_END();
    return (TEST_SUITE_STATUS == TEST_SUCCESS) ?
        EXIT_SUCCESS : EXIT_FAILURE;
}
