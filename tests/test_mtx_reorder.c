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
 * Last modified: 2021-07-28
 *
 * Unit tests for reordering sparse matrices.
 */

#include "test.h"

#include <matrixmarket/error.h>
#include <matrixmarket/io.h>
#include <matrixmarket/matrix.h>
#include <matrixmarket/matrix_array.h>
#include <matrixmarket/matrix_coordinate.h>
#include <matrixmarket/mtx.h>
#include <matrixmarket/reorder.h>
#include <matrixmarket/vector_array.h>
#include <matrixmarket/vector_coordinate.h>

#include <errno.h>
#include <unistd.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * `test_mtx_permute_vector_array_real()` tests permuting the elements
 * of a dense, real vector in the Matrix Market format.
 */
int test_mtx_permute_vector_array_real(void)
{
    int err;
    struct mtx vector;
    int num_comment_lines = 0;
    const char * comment_lines[] = {};
    float data[] = {1.0f, 2.0f, 3.0f};
    int size = sizeof(data) / sizeof(*data);
    err = mtx_init_vector_array_real(
        &vector, num_comment_lines, comment_lines, size, data);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));

    const int permutation[] = {2, 1, 3};
    err = mtx_permute_vector(&vector, permutation);
    TEST_ASSERT_EQ(3, vector.size);
    TEST_ASSERT_EQ(2.0f, ((const float *) vector.data)[0]);
    TEST_ASSERT_EQ(1.0f, ((const float *) vector.data)[1]);
    TEST_ASSERT_EQ(3.0f, ((const float *) vector.data)[2]);
    mtx_free(&vector);
    return TEST_SUCCESS;
}

/**
 * `test_mtx_permute_vector_coordinate_real()` tests permuting the
 * elements of a sparse, real vector in the Matrix Market format.
 */
int test_mtx_permute_vector_coordinate_real(void)
{
    int err;
    struct mtx vector;
    int num_comment_lines = 0;
    const char * comment_lines[] = {};
    int num_rows = 4;
    struct mtx_vector_coordinate_real data[] = {
        {1, 1.0f}, {2, 2.0f}, {4, 4.0f}};
    int size = sizeof(data) / sizeof(*data);
    err = mtx_init_vector_coordinate_real(
        &vector, mtx_unsorted, mtx_unordered, mtx_unassembled,
        num_comment_lines, comment_lines, num_rows, size, data);

    const int permutation[] = {2, 1, 4, 3};
    err = mtx_permute_vector(&vector, permutation);
    TEST_ASSERT_EQ(3, vector.size);
    const struct mtx_vector_coordinate_real * mtxdata =
        (const struct mtx_vector_coordinate_real *) vector.data;
    TEST_ASSERT_EQ(2, mtxdata[0].i); TEST_ASSERT_EQ(1.0f, mtxdata[0].a);
    TEST_ASSERT_EQ(1, mtxdata[1].i); TEST_ASSERT_EQ(2.0f, mtxdata[1].a);
    TEST_ASSERT_EQ(3, mtxdata[2].i); TEST_ASSERT_EQ(4.0f, mtxdata[2].a);
    mtx_free(&vector);
    return TEST_SUCCESS;
}

/**
 * `test_mtx_permute_matrix_array_real()` tests permuting the elements
 * of a dense, real matrix in the Matrix Market format.
 */
int test_mtx_permute_matrix_array_real(void)
{
    /* Permute rows. */
    {
        int err;
        struct mtx mtx;
        int num_comment_lines = 0;
        const char * comment_lines[] = {};
        int num_rows = 3;
        int num_columns = 3;
        float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
        err = mtx_init_matrix_array_real(
            &mtx, num_comment_lines, comment_lines,
            mtx_general, mtx_row_major, num_rows, num_columns, data);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));

        const int row_permutation[] = {2, 1, 3};
        const int * column_permutation = NULL;
        err = mtx_permute_matrix(&mtx, row_permutation, column_permutation);
        TEST_ASSERT_EQ(9, mtx.size);
        const float * mtxdata = (const float *) mtx.data;
        TEST_ASSERT_EQ(1.0f, mtxdata[3]);
        TEST_ASSERT_EQ(2.0f, mtxdata[4]);
        TEST_ASSERT_EQ(3.0f, mtxdata[5]);
        TEST_ASSERT_EQ(4.0f, mtxdata[0]);
        TEST_ASSERT_EQ(5.0f, mtxdata[1]);
        TEST_ASSERT_EQ(6.0f, mtxdata[2]);
        TEST_ASSERT_EQ(7.0f, mtxdata[6]);
        TEST_ASSERT_EQ(8.0f, mtxdata[7]);
        TEST_ASSERT_EQ(9.0f, mtxdata[8]);
        mtx_free(&mtx);
    }

    /* Permute columns. */
    {
        int err;
        struct mtx mtx;
        int num_comment_lines = 0;
        const char * comment_lines[] = {};
        int num_rows = 3;
        int num_columns = 3;
        float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
        err = mtx_init_matrix_array_real(
            &mtx, num_comment_lines, comment_lines,
            mtx_general, mtx_row_major, num_rows, num_columns, data);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));

        const int * row_permutation = NULL;
        const int column_permutation[] = {2, 1, 3};
        err = mtx_permute_matrix(&mtx, row_permutation, column_permutation);
        TEST_ASSERT_EQ(9, mtx.size);
        const float * mtxdata = (const float *) mtx.data;
        TEST_ASSERT_EQ(1.0f, mtxdata[1]);
        TEST_ASSERT_EQ(2.0f, mtxdata[0]);
        TEST_ASSERT_EQ(3.0f, mtxdata[2]);
        TEST_ASSERT_EQ(4.0f, mtxdata[4]);
        TEST_ASSERT_EQ(5.0f, mtxdata[3]);
        TEST_ASSERT_EQ(6.0f, mtxdata[5]);
        TEST_ASSERT_EQ(7.0f, mtxdata[7]);
        TEST_ASSERT_EQ(8.0f, mtxdata[6]);
        TEST_ASSERT_EQ(9.0f, mtxdata[8]);
        mtx_free(&mtx);
    }

    /* Permute rows and columns. */
    {
        int err;
        struct mtx mtx;
        int num_comment_lines = 0;
        const char * comment_lines[] = {};
        int num_rows = 3;
        int num_columns = 3;
        float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
        err = mtx_init_matrix_array_real(
            &mtx, num_comment_lines, comment_lines,
            mtx_general, mtx_row_major, num_rows, num_columns, data);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));

        const int row_permutation[] = {2, 1, 3};
        const int column_permutation[] = {2, 1, 3};
        err = mtx_permute_matrix(&mtx, row_permutation, column_permutation);
        TEST_ASSERT_EQ(9, mtx.size);
        const float * mtxdata = (const float *) mtx.data;
        TEST_ASSERT_EQ(1.0f, mtxdata[4]);
        TEST_ASSERT_EQ(2.0f, mtxdata[3]);
        TEST_ASSERT_EQ(3.0f, mtxdata[5]);
        TEST_ASSERT_EQ(4.0f, mtxdata[1]);
        TEST_ASSERT_EQ(5.0f, mtxdata[0]);
        TEST_ASSERT_EQ(6.0f, mtxdata[2]);
        TEST_ASSERT_EQ(7.0f, mtxdata[7]);
        TEST_ASSERT_EQ(8.0f, mtxdata[6]);
        TEST_ASSERT_EQ(9.0f, mtxdata[8]);
        mtx_free(&mtx);
    }
    return TEST_SUCCESS;
}

/**
 * `test_mtx_permute_matrix_coordinate_real()` tests permuting the elements
 * of a dense, real matrix in the Matrix Market format.
 */
int test_mtx_permute_matrix_coordinate_real(void)
{
    /* Permute rows. */
    {
        int err;
        struct mtx mtx;
        int num_comment_lines = 0;
        const char * comment_lines[] = {};
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

        const int row_permutation[] = {2, 1, 4, 3};
        const int * column_permutation = NULL;
        err = mtx_permute_matrix(&mtx, row_permutation, column_permutation);
        TEST_ASSERT_EQ(6, mtx.size);
        const struct mtx_matrix_coordinate_real * mtxdata =
            (const struct mtx_matrix_coordinate_real *) mtx.data;
        TEST_ASSERT_EQ(   2, mtxdata[0].i); TEST_ASSERT_EQ(   1, mtxdata[0].j);
        TEST_ASSERT_EQ(1.0f, mtxdata[0].a);
        TEST_ASSERT_EQ(   2, mtxdata[1].i); TEST_ASSERT_EQ(   4, mtxdata[1].j);
        TEST_ASSERT_EQ(2.0f, mtxdata[1].a);
        TEST_ASSERT_EQ(   1, mtxdata[2].i); TEST_ASSERT_EQ(   2, mtxdata[2].j);
        TEST_ASSERT_EQ(3.0f, mtxdata[2].a);
        TEST_ASSERT_EQ(   4, mtxdata[3].i); TEST_ASSERT_EQ(   3, mtxdata[3].j);
        TEST_ASSERT_EQ(4.0f, mtxdata[3].a);
        TEST_ASSERT_EQ(   3, mtxdata[4].i); TEST_ASSERT_EQ(   1, mtxdata[4].j);
        TEST_ASSERT_EQ(5.0f, mtxdata[4].a);
        TEST_ASSERT_EQ(   3, mtxdata[5].i); TEST_ASSERT_EQ(   4, mtxdata[5].j);
        TEST_ASSERT_EQ(6.0f, mtxdata[5].a);
        mtx_free(&mtx);
    }

    /* Permute columns. */
    {
        int err;
        struct mtx mtx;
        int num_comment_lines = 0;
        const char * comment_lines[] = {};
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

        const int * row_permutation = NULL;
        const int column_permutation[] = {2, 1, 4, 3};
        err = mtx_permute_matrix(&mtx, row_permutation, column_permutation);
        TEST_ASSERT_EQ(6, mtx.size);
        const struct mtx_matrix_coordinate_real * mtxdata =
            (const struct mtx_matrix_coordinate_real *) mtx.data;
        TEST_ASSERT_EQ(   1, mtxdata[0].i); TEST_ASSERT_EQ(   2, mtxdata[0].j);
        TEST_ASSERT_EQ(1.0f, mtxdata[0].a);
        TEST_ASSERT_EQ(   1, mtxdata[1].i); TEST_ASSERT_EQ(   3, mtxdata[1].j);
        TEST_ASSERT_EQ(2.0f, mtxdata[1].a);
        TEST_ASSERT_EQ(   2, mtxdata[2].i); TEST_ASSERT_EQ(   1, mtxdata[2].j);
        TEST_ASSERT_EQ(3.0f, mtxdata[2].a);
        TEST_ASSERT_EQ(   3, mtxdata[3].i); TEST_ASSERT_EQ(   4, mtxdata[3].j);
        TEST_ASSERT_EQ(4.0f, mtxdata[3].a);
        TEST_ASSERT_EQ(   4, mtxdata[4].i); TEST_ASSERT_EQ(   2, mtxdata[4].j);
        TEST_ASSERT_EQ(5.0f, mtxdata[4].a);
        TEST_ASSERT_EQ(   4, mtxdata[5].i); TEST_ASSERT_EQ(   3, mtxdata[5].j);
        TEST_ASSERT_EQ(6.0f, mtxdata[5].a);
        mtx_free(&mtx);
    }

    /* Permute rows and columns. */
    {
        int err;
        struct mtx mtx;
        int num_comment_lines = 0;
        const char * comment_lines[] = {};
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

        const int row_permutation[] = {2, 1, 4, 3};
        const int column_permutation[] = {2, 1, 4, 3};
        err = mtx_permute_matrix(&mtx, row_permutation, column_permutation);
        TEST_ASSERT_EQ(6, mtx.size);
        const struct mtx_matrix_coordinate_real * mtxdata =
            (const struct mtx_matrix_coordinate_real *) mtx.data;
        TEST_ASSERT_EQ(   2, mtxdata[0].i); TEST_ASSERT_EQ(   2, mtxdata[0].j);
        TEST_ASSERT_EQ(1.0f, mtxdata[0].a);
        TEST_ASSERT_EQ(   2, mtxdata[1].i); TEST_ASSERT_EQ(   3, mtxdata[1].j);
        TEST_ASSERT_EQ(2.0f, mtxdata[1].a);
        TEST_ASSERT_EQ(   1, mtxdata[2].i); TEST_ASSERT_EQ(   1, mtxdata[2].j);
        TEST_ASSERT_EQ(3.0f, mtxdata[2].a);
        TEST_ASSERT_EQ(   4, mtxdata[3].i); TEST_ASSERT_EQ(   4, mtxdata[3].j);
        TEST_ASSERT_EQ(4.0f, mtxdata[3].a);
        TEST_ASSERT_EQ(   3, mtxdata[4].i); TEST_ASSERT_EQ(   2, mtxdata[4].j);
        TEST_ASSERT_EQ(5.0f, mtxdata[4].a);
        TEST_ASSERT_EQ(   3, mtxdata[5].i); TEST_ASSERT_EQ(   3, mtxdata[5].j);
        TEST_ASSERT_EQ(6.0f, mtxdata[5].a);
        mtx_free(&mtx);
    }
    return TEST_SUCCESS;
}

/**
 * `main()' entry point and test driver.
 */
int main(int argc, char * argv[])
{
    TEST_SUITE_BEGIN("Running tests for reordering sparse matrices\n");
    TEST_RUN(test_mtx_permute_vector_array_real);
    TEST_RUN(test_mtx_permute_vector_coordinate_real);
    TEST_RUN(test_mtx_permute_matrix_array_real);
    TEST_RUN(test_mtx_permute_matrix_coordinate_real);
    TEST_SUITE_END();
    return (TEST_SUITE_STATUS == TEST_SUCCESS) ?
        EXIT_SUCCESS : EXIT_FAILURE;
}
