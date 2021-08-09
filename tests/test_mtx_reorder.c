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
 * Unit tests for reordering sparse matrices.
 */

#include "test.h"

#include <libmtx/error.h>
#include <libmtx/io.h>
#include <libmtx/matrix.h>
#include <libmtx/matrix_array.h>
#include <libmtx/matrix_coordinate.h>
#include <libmtx/mtx.h>
#include <libmtx/reorder.h>
#include <libmtx/vector_array.h>
#include <libmtx/vector_coordinate.h>

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
            &mtx, mtx_general, mtx_nontriangular, mtx_row_major,
            num_comment_lines, comment_lines,
            num_rows, num_columns, data);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));

        const int row_permutation[] = {2, 1, 3};
        const int * column_permutation = NULL;
        err = mtx_permute_matrix(&mtx, row_permutation, column_permutation);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
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
            &mtx, mtx_general, mtx_nontriangular, mtx_row_major,
            num_comment_lines, comment_lines,
            num_rows, num_columns, data);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));

        const int * row_permutation = NULL;
        const int column_permutation[] = {2, 1, 3};
        err = mtx_permute_matrix(&mtx, row_permutation, column_permutation);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
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
            &mtx, mtx_general, mtx_nontriangular, mtx_row_major,
            num_comment_lines, comment_lines,
            num_rows, num_columns, data);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));

        const int row_permutation[] = {2, 1, 3};
        const int column_permutation[] = {2, 1, 3};
        err = mtx_permute_matrix(&mtx, row_permutation, column_permutation);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
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
            &mtx, mtx_general, mtx_nontriangular,
            mtx_unsorted, mtx_unordered, mtx_unassembled,
            num_comment_lines, comment_lines,
            num_rows, num_columns, size, data);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));

        const int row_permutation[] = {2, 1, 4, 3};
        const int * column_permutation = NULL;
        err = mtx_permute_matrix(&mtx, row_permutation, column_permutation);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
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
            &mtx, mtx_general, mtx_nontriangular,
            mtx_unsorted, mtx_unordered, mtx_unassembled,
            num_comment_lines, comment_lines,
            num_rows, num_columns, size, data);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));

        const int * row_permutation = NULL;
        const int column_permutation[] = {2, 1, 4, 3};
        err = mtx_permute_matrix(&mtx, row_permutation, column_permutation);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
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
            &mtx, mtx_general, mtx_nontriangular,
            mtx_unsorted, mtx_unordered, mtx_unassembled,
            num_comment_lines, comment_lines,
            num_rows, num_columns, size, data);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));

        const int row_permutation[] = {2, 1, 4, 3};
        const int column_permutation[] = {2, 1, 4, 3};
        err = mtx_permute_matrix(&mtx, row_permutation, column_permutation);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
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

    /* Permute rows and columns.    */
    /*                              */
    /*    5--7--6           7--8--9 */
    /*    |  | /            |  | /  */
    /* 4--8--2     -->   3--5--6    */
    /* |  |  |           |  |  |    */
    /* 9--1--3           1--2--4    */
    {
        int err;
        struct mtx mtx;
        int num_comment_lines = 0;
        const char * comment_lines[] = {};
        int num_rows = 9;
        int num_columns = 9;
        int64_t size = 24;
        const struct mtx_matrix_coordinate_pattern data[] = {
            {1,3}, {1,8}, {1,9},
            {2,3}, {2,6}, {2,7}, {2,8},
            {3,1}, {3,2},
            {4,8}, {4,9},
            {5,7}, {5,8},
            {6,2}, {6,7},
            {7,2}, {7,5}, {7,6},
            {8,1}, {8,2}, {8,4}, {8,5},
            {9,1}, {9,4}};
        err = mtx_init_matrix_coordinate_pattern(
            &mtx, mtx_general, mtx_nontriangular,
            mtx_unsorted, mtx_unordered, mtx_unassembled,
            num_comment_lines, comment_lines,
            num_rows, num_columns, size, data);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));

        const int row_permutation[] = {2,6,4,3,7,9,8,5,1};
        err = mtx_permute_matrix(&mtx, row_permutation, row_permutation);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        err = mtx_sort(&mtx, mtx_row_major);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(24, mtx.size);
        const struct mtx_matrix_coordinate_pattern * mtxdata =
            (const struct mtx_matrix_coordinate_pattern *) mtx.data;
        TEST_ASSERT_EQ(1, mtxdata[ 0].i); TEST_ASSERT_EQ(2, mtxdata[ 0].j);
        TEST_ASSERT_EQ(1, mtxdata[ 1].i); TEST_ASSERT_EQ(3, mtxdata[ 1].j);
        TEST_ASSERT_EQ(2, mtxdata[ 2].i); TEST_ASSERT_EQ(1, mtxdata[ 2].j);
        TEST_ASSERT_EQ(2, mtxdata[ 3].i); TEST_ASSERT_EQ(4, mtxdata[ 3].j);
        TEST_ASSERT_EQ(2, mtxdata[ 4].i); TEST_ASSERT_EQ(5, mtxdata[ 4].j);
        TEST_ASSERT_EQ(3, mtxdata[ 5].i); TEST_ASSERT_EQ(1, mtxdata[ 5].j);
        TEST_ASSERT_EQ(3, mtxdata[ 6].i); TEST_ASSERT_EQ(5, mtxdata[ 6].j);
        TEST_ASSERT_EQ(4, mtxdata[ 7].i); TEST_ASSERT_EQ(2, mtxdata[ 7].j);
        TEST_ASSERT_EQ(4, mtxdata[ 8].i); TEST_ASSERT_EQ(6, mtxdata[ 8].j);
        TEST_ASSERT_EQ(5, mtxdata[ 9].i); TEST_ASSERT_EQ(2, mtxdata[ 9].j);
        TEST_ASSERT_EQ(5, mtxdata[10].i); TEST_ASSERT_EQ(3, mtxdata[10].j);
        TEST_ASSERT_EQ(5, mtxdata[11].i); TEST_ASSERT_EQ(6, mtxdata[11].j);
        TEST_ASSERT_EQ(5, mtxdata[12].i); TEST_ASSERT_EQ(7, mtxdata[12].j);
        TEST_ASSERT_EQ(6, mtxdata[13].i); TEST_ASSERT_EQ(4, mtxdata[13].j);
        TEST_ASSERT_EQ(6, mtxdata[14].i); TEST_ASSERT_EQ(5, mtxdata[14].j);
        TEST_ASSERT_EQ(6, mtxdata[15].i); TEST_ASSERT_EQ(8, mtxdata[15].j);
        TEST_ASSERT_EQ(6, mtxdata[16].i); TEST_ASSERT_EQ(9, mtxdata[16].j);
        TEST_ASSERT_EQ(7, mtxdata[17].i); TEST_ASSERT_EQ(5, mtxdata[17].j);
        TEST_ASSERT_EQ(7, mtxdata[18].i); TEST_ASSERT_EQ(8, mtxdata[18].j);
        TEST_ASSERT_EQ(8, mtxdata[19].i); TEST_ASSERT_EQ(6, mtxdata[19].j);
        TEST_ASSERT_EQ(8, mtxdata[20].i); TEST_ASSERT_EQ(7, mtxdata[20].j);
        TEST_ASSERT_EQ(8, mtxdata[21].i); TEST_ASSERT_EQ(9, mtxdata[21].j);
        TEST_ASSERT_EQ(9, mtxdata[22].i); TEST_ASSERT_EQ(6, mtxdata[22].j);
        TEST_ASSERT_EQ(9, mtxdata[23].i); TEST_ASSERT_EQ(8, mtxdata[23].j);
        mtx_free(&mtx);
    }
    return TEST_SUCCESS;
}

/**
 * `test_mtx_matrix_reorder_rcm_coordinate_real()' tests reordering the
 * nonzeros of a real matrix in coordinate format.
 */
int test_mtx_matrix_reorder_rcm_coordinate_real(void)
{
    int err;

    /* Create matrix. */
    struct mtx mtx;
    int num_comment_lines = 0;
    const char * comment_lines[] = {};
    int num_rows = 5;
    int num_columns = 5;
    int64_t size = 10;
    const struct mtx_matrix_coordinate_real data[] = {
        {1,2, 1.0f},
        {1,3, 2.0f},
        {2,1, 3.0f},
        {2,4, 4.0f},
        {3,1, 5.0f},
        {3,4, 6.0f},
        {3,5, 7.0f},
        {4,2, 8.0f},
        {4,3, 9.0f},
        {5,3,10.0f}};
    err = mtx_init_matrix_coordinate_real(
        &mtx, mtx_general, mtx_nontriangular,
        mtx_row_major, mtx_unordered, mtx_unassembled,
        num_comment_lines, comment_lines,
        num_rows, num_columns, size, data);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));

    /* Reorder the matrix and verify the results. */
    int starting_row = 1;
    int * permutation;
    err = mtx_matrix_reorder_rcm(&mtx, &permutation, starting_row);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));

    TEST_ASSERT_EQ(5, permutation[0]);
    TEST_ASSERT_EQ(3, permutation[1]);
    TEST_ASSERT_EQ(4, permutation[2]);
    TEST_ASSERT_EQ(2, permutation[3]);
    TEST_ASSERT_EQ(1, permutation[4]);
    free(permutation);

    TEST_ASSERT_EQ(mtx_matrix, mtx.object);
    TEST_ASSERT_EQ(mtx_coordinate, mtx.format);
    TEST_ASSERT_EQ(mtx_real, mtx.field);
    TEST_ASSERT_EQ(mtx_general, mtx.symmetry);
    TEST_ASSERT_EQ(mtx_nontriangular, mtx.triangle);
    TEST_ASSERT_EQ(mtx_unsorted, mtx.sorting);
    TEST_ASSERT_EQ(mtx_rcm, mtx.ordering);
    TEST_ASSERT_EQ(mtx_unassembled, mtx.assembly);
    TEST_ASSERT_EQ(0, mtx.num_comment_lines);
    TEST_ASSERT_EQ(5, mtx.num_rows);
    TEST_ASSERT_EQ(5, mtx.num_columns);
    TEST_ASSERT_EQ(-1, mtx.num_nonzeros);
    TEST_ASSERT_EQ(10, mtx.size);
    TEST_ASSERT_EQ(sizeof(struct mtx_matrix_coordinate_real), mtx.nonzero_size);
    const struct mtx_matrix_coordinate_real * mtxdata =
        (const struct mtx_matrix_coordinate_real *) mtx.data;
    TEST_ASSERT_EQ(    5, mtxdata[0].i); TEST_ASSERT_EQ(3, mtxdata[0].j);
    TEST_ASSERT_EQ( 1.0f, mtxdata[0].a);
    TEST_ASSERT_EQ(    5, mtxdata[1].i); TEST_ASSERT_EQ(4, mtxdata[1].j);
    TEST_ASSERT_EQ( 2.0f, mtxdata[1].a);
    TEST_ASSERT_EQ(    3, mtxdata[2].i); TEST_ASSERT_EQ(5, mtxdata[2].j);
    TEST_ASSERT_EQ( 3.0f, mtxdata[2].a);
    TEST_ASSERT_EQ(    3, mtxdata[3].i); TEST_ASSERT_EQ(2, mtxdata[3].j);
    TEST_ASSERT_EQ( 4.0f, mtxdata[3].a);
    TEST_ASSERT_EQ(    4, mtxdata[4].i); TEST_ASSERT_EQ(5, mtxdata[4].j);
    TEST_ASSERT_EQ( 5.0f, mtxdata[4].a);
    TEST_ASSERT_EQ(    4, mtxdata[5].i); TEST_ASSERT_EQ(2, mtxdata[5].j);
    TEST_ASSERT_EQ( 6.0f, mtxdata[5].a);
    TEST_ASSERT_EQ(    4, mtxdata[6].i); TEST_ASSERT_EQ(1, mtxdata[6].j);
    TEST_ASSERT_EQ( 7.0f, mtxdata[6].a);
    TEST_ASSERT_EQ(    2, mtxdata[7].i); TEST_ASSERT_EQ(3, mtxdata[7].j);
    TEST_ASSERT_EQ( 8.0f, mtxdata[7].a);
    TEST_ASSERT_EQ(    2, mtxdata[8].i); TEST_ASSERT_EQ(4, mtxdata[8].j);
    TEST_ASSERT_EQ( 9.0f, mtxdata[8].a);
    TEST_ASSERT_EQ(    1, mtxdata[9].i); TEST_ASSERT_EQ(4, mtxdata[9].j);
    TEST_ASSERT_EQ(10.0f, mtxdata[9].a);
    mtx_free(&mtx);
    return MTX_SUCCESS;
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
    TEST_RUN(test_mtx_matrix_reorder_rcm_coordinate_real);
    TEST_SUITE_END();
    return (TEST_SUITE_STATUS == TEST_SUCCESS) ?
        EXIT_SUCCESS : EXIT_FAILURE;
}
