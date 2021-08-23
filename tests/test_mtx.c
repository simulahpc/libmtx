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
 * Unit tests for Matrix Market I/O.
 */

#include "test.h"

#include <libmtx/error.h>
#include <libmtx/matrix/array.h>
#include <libmtx/matrix/array/data.h>
#include <libmtx/matrix/coordinate.h>
#include <libmtx/matrix/coordinate/data.h>
#include <libmtx/mtx/mtx.h>
#include <libmtx/vector/array.h>
#include <libmtx/vector/coordinate.h>

#include <errno.h>
#include <unistd.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * `test_mtx_copy_init()' tests copying an `mtx' object.
 */
int test_mtx_copy_init(void)
{
    int err;

    /* Create a sparse matrix. */
    struct mtx srcmtx;
    int num_comment_lines = 1;
    const char * comment_lines[] = {"% a comment\n"};
    int num_rows = 4;
    int num_columns = 4;
    int64_t size = 6;
    const struct mtx_matrix_coordinate_real_single data[] = {
        {1,1,1.0f}, {1,4,2.0f},
        {2,2,3.0f},
        {3,3,4.0f},
        {4,1,5.0f}, {4,4,6.0f}};
    err = mtx_init_matrix_coordinate_real_single(
        &srcmtx, mtx_general, mtx_nontriangular,
        mtx_unsorted, mtx_unassembled,
        num_comment_lines, comment_lines,
        num_rows, num_columns, size, data);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));

    /* Make a copy and verify the copied contents. */
    struct mtx destmtx;
    err = mtx_copy_init(&destmtx, &srcmtx);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
    TEST_ASSERT_EQ(mtx_matrix, destmtx.object);
    TEST_ASSERT_EQ(mtx_coordinate, destmtx.format);
    TEST_ASSERT_EQ(mtx_real, destmtx.field);
    TEST_ASSERT_EQ(mtx_general, destmtx.symmetry);
    TEST_ASSERT_EQ(1, destmtx.num_comment_lines);
    TEST_ASSERT_STREQ("% a comment\n", destmtx.comment_lines[0]);
    TEST_ASSERT_EQ(4, destmtx.num_rows);
    TEST_ASSERT_EQ(4, destmtx.num_columns);
    TEST_ASSERT_EQ(6, destmtx.num_nonzeros);
    const struct mtx_matrix_coordinate_data * matrix_coordinate =
        &destmtx.storage.matrix_coordinate;
    TEST_ASSERT_EQ(mtx_real, matrix_coordinate->field);
    TEST_ASSERT_EQ(mtx_single, matrix_coordinate->precision);
    TEST_ASSERT_EQ(mtx_nontriangular, matrix_coordinate->triangle);
    TEST_ASSERT_EQ(mtx_unsorted, matrix_coordinate->sorting);
    TEST_ASSERT_EQ(mtx_unassembled, matrix_coordinate->assembly);
    TEST_ASSERT_EQ(6, matrix_coordinate->size);
    const struct mtx_matrix_coordinate_real_single * mtxdata =
        matrix_coordinate->data.real_single;
    TEST_ASSERT_EQ(   1, mtxdata[0].i); TEST_ASSERT_EQ(   1, mtxdata[0].j);
    TEST_ASSERT_EQ(1.0f, mtxdata[0].a);
    TEST_ASSERT_EQ(   1, mtxdata[1].i); TEST_ASSERT_EQ(   4, mtxdata[1].j);
    TEST_ASSERT_EQ(2.0f, mtxdata[1].a);
    TEST_ASSERT_EQ(   2, mtxdata[2].i); TEST_ASSERT_EQ(   2, mtxdata[2].j);
    TEST_ASSERT_EQ(3.0f, mtxdata[2].a);
    TEST_ASSERT_EQ(   3, mtxdata[3].i); TEST_ASSERT_EQ(   3, mtxdata[3].j);
    TEST_ASSERT_EQ(4.0f, mtxdata[3].a);
    TEST_ASSERT_EQ(   4, mtxdata[4].i); TEST_ASSERT_EQ(   1, mtxdata[4].j);
    TEST_ASSERT_EQ(5.0f, mtxdata[4].a);
    TEST_ASSERT_EQ(   4, mtxdata[5].i); TEST_ASSERT_EQ(   4, mtxdata[5].j);
    TEST_ASSERT_EQ(6.0f, mtxdata[5].a);
    mtx_free(&destmtx);
    mtx_free(&srcmtx);
    return TEST_SUCCESS;
}

/**
 * `test_mtx_set_comment_lines()' tests setting comment lines for a
 * matrix or vector.
 */
int test_mtx_set_comment_lines(void)
{
    {
        struct mtx mtx;
        mtx.num_comment_lines = 0;
        mtx.comment_lines = NULL;
        int num_comment_lines = 1;
        const char * comment_lines[] = {"% comment\n"};
        int err = mtx_set_comment_lines(
            &mtx, num_comment_lines, comment_lines);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_STREQ("% comment\n", mtx.comment_lines[0]);
        for (int i = 0; i < mtx.num_comment_lines; i++)
            free(mtx.comment_lines[i]);
        free(mtx.comment_lines);
    }

    {
        struct mtx mtx;
        mtx.num_comment_lines = 0;
        mtx.comment_lines = NULL;
        int num_comment_lines = 1;
        const char * comment_lines[] = {"invalid comment"};
        int err = mtx_set_comment_lines(
            &mtx, num_comment_lines, comment_lines);
        TEST_ASSERT_EQ(MTX_ERR_INVALID_MTX_COMMENT, err);
        for (int i = 0; i < mtx.num_comment_lines; i++)
            free(mtx.comment_lines[i]);
        free(mtx.comment_lines);
    }

    {
        struct mtx mtx;
        mtx.num_comment_lines = 0;
        mtx.comment_lines = NULL;
        int num_comment_lines = 1;
        const char * comment_lines[] = {"% invalid comment"};
        int err = mtx_set_comment_lines(
            &mtx, num_comment_lines, comment_lines);
        TEST_ASSERT_EQ(MTX_ERR_INVALID_MTX_COMMENT, err);
        for (int i = 0; i < mtx.num_comment_lines; i++)
            free(mtx.comment_lines[i]);
        free(mtx.comment_lines);
    }

    {
        struct mtx mtx;
        mtx.num_comment_lines = 0;
        mtx.comment_lines = NULL;
        int num_comment_lines = 1;
        const char * comment_lines[] = {"% first comment\n"};
        int err = mtx_set_comment_lines(
            &mtx, num_comment_lines, comment_lines);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        err = mtx_add_comment_line(&mtx, "% second comment\n");
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_STREQ("% first comment\n", mtx.comment_lines[0]);
        TEST_ASSERT_STREQ("% second comment\n", mtx.comment_lines[1]);
        for (int i = 0; i < mtx.num_comment_lines; i++)
            free(mtx.comment_lines[i]);
        free(mtx.comment_lines);
    }

    {
        struct mtx mtx;
        mtx.num_comment_lines = 0;
        mtx.comment_lines = NULL;
        int err = mtx_add_comment_line_printf(
            &mtx, "%% %d%s %s\n", 1, "st", "comment");
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_STREQ_MSG(
            "% 1st comment\n", mtx.comment_lines[0], "%s", mtx.comment_lines[0]);
        for (int i = 0; i < mtx.num_comment_lines; i++)
            free(mtx.comment_lines[i]);
        free(mtx.comment_lines);
    }

    return TEST_SUCCESS;
}

/**
 * `test_mtx_set_zero_matrix_coordinate_real_single()` tests zeroing a
 * dense, real matrix in Matrix Market format.
 */
int test_mtx_set_zero_matrix_coordinate_real_single(void)
{
    int err;
    struct mtx mtx;
    int num_comment_lines = 0;
    const char * comment_lines[] = {};
    int num_rows = 4;
    int num_columns = 4;
    struct mtx_matrix_coordinate_real_single data[] = {
        {1, 1, 1.0f}, {2, 3, 2.0f}, {4, 2, 4.0f}};
    int size = sizeof(data) / sizeof(*data);
    err = mtx_init_matrix_coordinate_real_single(
        &mtx, mtx_general, mtx_nontriangular,
        mtx_unsorted, mtx_unassembled,
        num_comment_lines, comment_lines,
        num_rows, num_columns, size, data);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));

    err = mtx_set_zero(&mtx);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
    const struct mtx_matrix_coordinate_data * matrix_coordinate =
        &mtx.storage.matrix_coordinate;
    TEST_ASSERT_EQ(mtx_real, matrix_coordinate->field);
    TEST_ASSERT_EQ(mtx_single, matrix_coordinate->precision);
    TEST_ASSERT_EQ(mtx_unsorted, matrix_coordinate->sorting);
    TEST_ASSERT_EQ(mtx_unassembled, matrix_coordinate->assembly);
    TEST_ASSERT_EQ(3, matrix_coordinate->size);
    const struct mtx_matrix_coordinate_real_single * mtxdata =
        matrix_coordinate->data.real_single;
    TEST_ASSERT_EQ(0.0f, mtxdata[0].a);
    TEST_ASSERT_EQ(0.0f, mtxdata[1].a);
    TEST_ASSERT_EQ(0.0f, mtxdata[2].a);
    mtx_free(&mtx);
    return TEST_SUCCESS;
}

/**
 * `test_mtx_set_zero_vector_array_real_single()` tests zeroing a
 * dense, real vector in Matrix Market format.
 */
int test_mtx_set_zero_vector_array_real_single(void)
{
    int err;
    struct mtx mtx;
    int num_comment_lines = 0;
    const char * comment_lines[] = {};
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    int size = sizeof(data) / sizeof(*data);
    err = mtx_init_vector_array_real_single(
        &mtx, num_comment_lines, comment_lines, size, data);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));

    err = mtx_set_zero(&mtx);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
    const struct mtx_vector_array_data * vector_array =
        &mtx.storage.vector_array;
    TEST_ASSERT_EQ(mtx_real, vector_array->field);
    TEST_ASSERT_EQ(mtx_single, vector_array->precision);
    TEST_ASSERT_EQ(4, vector_array->size);
    const float * mtxdata = vector_array->data.real_single;
    TEST_ASSERT_EQ(0.0f, mtxdata[0]);
    TEST_ASSERT_EQ(0.0f, mtxdata[1]);
    TEST_ASSERT_EQ(0.0f, mtxdata[2]);
    TEST_ASSERT_EQ(0.0f, mtxdata[3]);
    mtx_free(&mtx);
    return TEST_SUCCESS;
}

/**
 * `test_mtx_set_zero_vector_coordinate_real_single()` tests zeroing a
 * dense, real vector in Matrix Market format.
 */
int test_mtx_set_zero_vector_coordinate_real_single(void)
{
    int err;
    struct mtx mtx;
    int num_comment_lines = 0;
    const char * comment_lines[] = {};
    int num_rows = 4;
    struct mtx_vector_coordinate_real_single data[] = {
        {1, 1.0f}, {2, 2.0f}, {4, 4.0f}};
    int size = sizeof(data) / sizeof(*data);
    err = mtx_init_vector_coordinate_real_single(
        &mtx, mtx_unsorted, mtx_unassembled,
        num_comment_lines, comment_lines, num_rows, size, data);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));

    err = mtx_set_zero(&mtx);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
    const struct mtx_vector_coordinate_data * vector_coordinate =
        &mtx.storage.vector_coordinate;
    TEST_ASSERT_EQ(mtx_real, vector_coordinate->field);
    TEST_ASSERT_EQ(mtx_single, vector_coordinate->precision);
    TEST_ASSERT_EQ(mtx_unsorted, vector_coordinate->sorting);
    TEST_ASSERT_EQ(mtx_unassembled, vector_coordinate->assembly);
    TEST_ASSERT_EQ(3, vector_coordinate->size);
    const struct mtx_vector_coordinate_real_single * mtxdata =
        vector_coordinate->data.real_single;
    TEST_ASSERT_EQ(0.0f, mtxdata[0].a);
    TEST_ASSERT_EQ(0.0f, mtxdata[1].a);
    TEST_ASSERT_EQ(0.0f, mtxdata[2].a);
    mtx_free(&mtx);
    return TEST_SUCCESS;
}

/**
 * `main()' entry point and test driver.
 */
int main(int argc, char * argv[])
{
    TEST_SUITE_BEGIN("Running tests for Matrix Market objects\n");
    TEST_RUN(test_mtx_copy_init);
    TEST_RUN(test_mtx_set_comment_lines);
    TEST_RUN(test_mtx_set_zero_matrix_coordinate_real_single);
    TEST_RUN(test_mtx_set_zero_vector_array_real_single);
    TEST_RUN(test_mtx_set_zero_vector_coordinate_real_single);
    TEST_SUITE_END();
    return (TEST_SUITE_STATUS == TEST_SUCCESS) ?
        EXIT_SUCCESS : EXIT_FAILURE;
}
