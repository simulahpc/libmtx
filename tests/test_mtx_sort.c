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
 * Last modified: 2021-08-06
 *
 * Unit tests for sorting Matrix Market objects.
 */

#include "test.h"

#include <matrixmarket/error.h>
#include <matrixmarket/mtx.h>
#include <matrixmarket/io.h>
#include <matrixmarket/matrix_coordinate.h>

#include <errno.h>
#include <unistd.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * `test_mtx_sort_matrix_coordinate_real()' tests sorting the
 * nonzeros of a real matrix in coordinate format.
 */
int test_mtx_sort_matrix_coordinate_real(void)
{
    int err;

    /* Create matrix. */
    struct mtx mtx;
    int num_comment_lines = 1;
    const char * comment_lines[] = {"a comment"};
    int num_rows = 4;
    int num_columns = 4;
    int64_t size = 6;
    const struct mtx_matrix_coordinate_real data[] = {
        {3,3,4.0f},
        {1,4,2.0f},
        {4,1,5.0f},
        {1,1,1.0f},
        {2,2,3.0f},
        {4,4,6.0f}};
    err = mtx_init_matrix_coordinate_real(
        &mtx, mtx_general, mtx_nontriangular,
        mtx_unsorted, mtx_unordered, mtx_unassembled,
        num_comment_lines, comment_lines,
        num_rows, num_columns, size, data);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));

    /* Sort the nonzeros and verify the results. */
    enum mtx_sorting sorting = mtx_row_major;
    err = mtx_sort(&mtx, sorting);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
    TEST_ASSERT_EQ(mtx_matrix, mtx.object);
    TEST_ASSERT_EQ(mtx_coordinate, mtx.format);
    TEST_ASSERT_EQ(mtx_real, mtx.field);
    TEST_ASSERT_EQ(mtx_general, mtx.symmetry);
    TEST_ASSERT_EQ(mtx_nontriangular, mtx.triangle);
    TEST_ASSERT_EQ(mtx_row_major, mtx.sorting);
    TEST_ASSERT_EQ(mtx_unordered, mtx.ordering);
    TEST_ASSERT_EQ(mtx_unassembled, mtx.assembly);
    TEST_ASSERT_EQ(1, mtx.num_comment_lines);
    TEST_ASSERT_STREQ("a comment", mtx.comment_lines[0]);
    TEST_ASSERT_EQ(4, mtx.num_rows);
    TEST_ASSERT_EQ(4, mtx.num_columns);
    TEST_ASSERT_EQ(6, mtx.num_nonzeros);
    TEST_ASSERT_EQ(6, mtx.size);
    TEST_ASSERT_EQ(sizeof(struct mtx_matrix_coordinate_real), mtx.nonzero_size);
    const struct mtx_matrix_coordinate_real * mtxdata =
        (const struct mtx_matrix_coordinate_real *) mtx.data;
    TEST_ASSERT_EQ(   1, mtxdata[0].i);
    TEST_ASSERT_EQ(   1, mtxdata[0].j);
    TEST_ASSERT_EQ(1.0f, mtxdata[0].a);
    TEST_ASSERT_EQ(   1, mtxdata[1].i);
    TEST_ASSERT_EQ(   4, mtxdata[1].j);
    TEST_ASSERT_EQ(2.0f, mtxdata[1].a);
    TEST_ASSERT_EQ(   2, mtxdata[2].i);
    TEST_ASSERT_EQ(   2, mtxdata[2].j);
    TEST_ASSERT_EQ(3.0f, mtxdata[2].a);
    TEST_ASSERT_EQ(   3, mtxdata[3].i);
    TEST_ASSERT_EQ(   3, mtxdata[3].j);
    TEST_ASSERT_EQ(4.0f, mtxdata[3].a);
    TEST_ASSERT_EQ(   4, mtxdata[4].i);
    TEST_ASSERT_EQ(   1, mtxdata[4].j);
    TEST_ASSERT_EQ(5.0f, mtxdata[4].a);
    TEST_ASSERT_EQ(   4, mtxdata[5].i);
    TEST_ASSERT_EQ(   4, mtxdata[5].j);
    TEST_ASSERT_EQ(6.0f, mtxdata[5].a);
    mtx_free(&mtx);
    return MTX_SUCCESS;
}

/**
 * `test_mtx_sort_matrix_coordinate_double()' tests sorting the
 * nonzeros of a double matrix in coordinate format.
 */
int test_mtx_sort_matrix_coordinate_double(void)
{
    int err;

    /* Create matrix. */
    struct mtx mtx;
    int num_comment_lines = 1;
    const char * comment_lines[] = {"a comment"};
    int num_rows = 4;
    int num_columns = 4;
    int64_t size = 6;
    const struct mtx_matrix_coordinate_double data[] = {
        {3,3,4.0f},
        {1,4,2.0f},
        {4,1,5.0f},
        {1,1,1.0f},
        {2,2,3.0f},
        {4,4,6.0f}};
    err = mtx_init_matrix_coordinate_double(
        &mtx, mtx_general, mtx_nontriangular,
        mtx_unsorted, mtx_unordered, mtx_unassembled,
        num_comment_lines, comment_lines,
        num_rows, num_columns, size, data);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));

    /* Sort the nonzeros and verify the results. */
    enum mtx_sorting sorting = mtx_row_major;
    err = mtx_sort(&mtx, sorting);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
    TEST_ASSERT_EQ(mtx_matrix, mtx.object);
    TEST_ASSERT_EQ(mtx_coordinate, mtx.format);
    TEST_ASSERT_EQ(mtx_double, mtx.field);
    TEST_ASSERT_EQ(mtx_general, mtx.symmetry);
    TEST_ASSERT_EQ(mtx_nontriangular, mtx.symmetry);
    TEST_ASSERT_EQ(mtx_row_major, mtx.sorting);
    TEST_ASSERT_EQ(mtx_unordered, mtx.ordering);
    TEST_ASSERT_EQ(mtx_unassembled, mtx.assembly);
    TEST_ASSERT_EQ(1, mtx.num_comment_lines);
    TEST_ASSERT_STREQ("a comment", mtx.comment_lines[0]);
    TEST_ASSERT_EQ(4, mtx.num_rows);
    TEST_ASSERT_EQ(4, mtx.num_columns);
    TEST_ASSERT_EQ(6, mtx.num_nonzeros);
    TEST_ASSERT_EQ(6, mtx.size);
    TEST_ASSERT_EQ(sizeof(struct mtx_matrix_coordinate_double), mtx.nonzero_size);
    const struct mtx_matrix_coordinate_double * mtxdata =
        (const struct mtx_matrix_coordinate_double *) mtx.data;
    TEST_ASSERT_EQ(   1, mtxdata[0].i);
    TEST_ASSERT_EQ(   1, mtxdata[0].j);
    TEST_ASSERT_EQ(1.0f, mtxdata[0].a);
    TEST_ASSERT_EQ(   1, mtxdata[1].i);
    TEST_ASSERT_EQ(   4, mtxdata[1].j);
    TEST_ASSERT_EQ(2.0f, mtxdata[1].a);
    TEST_ASSERT_EQ(   2, mtxdata[2].i);
    TEST_ASSERT_EQ(   2, mtxdata[2].j);
    TEST_ASSERT_EQ(3.0f, mtxdata[2].a);
    TEST_ASSERT_EQ(   3, mtxdata[3].i);
    TEST_ASSERT_EQ(   3, mtxdata[3].j);
    TEST_ASSERT_EQ(4.0f, mtxdata[3].a);
    TEST_ASSERT_EQ(   4, mtxdata[4].i);
    TEST_ASSERT_EQ(   1, mtxdata[4].j);
    TEST_ASSERT_EQ(5.0f, mtxdata[4].a);
    TEST_ASSERT_EQ(   4, mtxdata[5].i);
    TEST_ASSERT_EQ(   4, mtxdata[5].j);
    TEST_ASSERT_EQ(6.0f, mtxdata[5].a);
    mtx_free(&mtx);
    return MTX_SUCCESS;
}

/**
 * `test_mtx_sort_matrix_coordinate_complex()' tests sorting the
 * nonzeros of a complex matrix in coordinate format.
 */
int test_mtx_sort_matrix_coordinate_complex(void)
{
    int err;

    /* Create matrix. */
    struct mtx mtx;
    int num_comment_lines = 1;
    const char * comment_lines[] = {"a comment"};
    int num_rows = 4;
    int num_columns = 4;
    int64_t size = 6;
    const struct mtx_matrix_coordinate_complex data[] = {
        {3,3,4.0f,4.0f},
        {1,4,2.0f,2.0f},
        {4,1,5.0f,5.0f},
        {1,1,1.0f,1.0f},
        {2,2,3.0f,3.0f},
        {4,4,6.0f,6.0f}};
    err = mtx_init_matrix_coordinate_complex(
        &mtx, mtx_general, mtx_nontriangular,
        mtx_unsorted, mtx_unordered, mtx_unassembled,
        num_comment_lines, comment_lines,
        num_rows, num_columns, size, data);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));

    /* Sort the nonzeros and verify the results. */
    enum mtx_sorting sorting = mtx_row_major;
    err = mtx_sort(&mtx, sorting);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
    TEST_ASSERT_EQ(mtx_matrix, mtx.object);
    TEST_ASSERT_EQ(mtx_coordinate, mtx.format);
    TEST_ASSERT_EQ(mtx_complex, mtx.field);
    TEST_ASSERT_EQ(mtx_general, mtx.symmetry);
    TEST_ASSERT_EQ(mtx_nontriangular, mtx.symmetry);
    TEST_ASSERT_EQ(mtx_row_major, mtx.sorting);
    TEST_ASSERT_EQ(mtx_unordered, mtx.ordering);
    TEST_ASSERT_EQ(mtx_unassembled, mtx.assembly);
    TEST_ASSERT_EQ(1, mtx.num_comment_lines);
    TEST_ASSERT_STREQ("a comment", mtx.comment_lines[0]);
    TEST_ASSERT_EQ(4, mtx.num_rows);
    TEST_ASSERT_EQ(4, mtx.num_columns);
    TEST_ASSERT_EQ(6, mtx.num_nonzeros);
    TEST_ASSERT_EQ(6, mtx.size);
    TEST_ASSERT_EQ(sizeof(struct mtx_matrix_coordinate_complex), mtx.nonzero_size);
    const struct mtx_matrix_coordinate_complex * mtxdata =
        (const struct mtx_matrix_coordinate_complex *) mtx.data;
    TEST_ASSERT_EQ(   1, mtxdata[0].i);
    TEST_ASSERT_EQ(   1, mtxdata[0].j);
    TEST_ASSERT_EQ(1.0f, mtxdata[0].a);
    TEST_ASSERT_EQ(1.0f, mtxdata[0].b);
    TEST_ASSERT_EQ(   1, mtxdata[1].i);
    TEST_ASSERT_EQ(   4, mtxdata[1].j);
    TEST_ASSERT_EQ(2.0f, mtxdata[1].a);
    TEST_ASSERT_EQ(2.0f, mtxdata[1].b);
    TEST_ASSERT_EQ(   2, mtxdata[2].i);
    TEST_ASSERT_EQ(   2, mtxdata[2].j);
    TEST_ASSERT_EQ(3.0f, mtxdata[2].a);
    TEST_ASSERT_EQ(3.0f, mtxdata[2].b);
    TEST_ASSERT_EQ(   3, mtxdata[3].i);
    TEST_ASSERT_EQ(   3, mtxdata[3].j);
    TEST_ASSERT_EQ(4.0f, mtxdata[3].a);
    TEST_ASSERT_EQ(4.0f, mtxdata[3].b);
    TEST_ASSERT_EQ(   4, mtxdata[4].i);
    TEST_ASSERT_EQ(   1, mtxdata[4].j);
    TEST_ASSERT_EQ(5.0f, mtxdata[4].a);
    TEST_ASSERT_EQ(5.0f, mtxdata[4].b);
    TEST_ASSERT_EQ(   4, mtxdata[5].i);
    TEST_ASSERT_EQ(   4, mtxdata[5].j);
    TEST_ASSERT_EQ(6.0f, mtxdata[5].a);
    TEST_ASSERT_EQ(6.0f, mtxdata[5].b);
    mtx_free(&mtx);
    return MTX_SUCCESS;
}

/**
 * `test_mtx_sort_matrix_coordinate_integer()' tests sorting the
 * nonzeros of a integer matrix in coordinate format.
 */
int test_mtx_sort_matrix_coordinate_integer(void)
{
    int err;

    /* Create matrix. */
    struct mtx mtx;
    int num_comment_lines = 1;
    const char * comment_lines[] = {"a comment"};
    int num_rows = 4;
    int num_columns = 4;
    int64_t size = 6;
    const struct mtx_matrix_coordinate_integer data[] = {
        {3,3,4.0f},
        {1,4,2.0f},
        {4,1,5.0f},
        {1,1,1.0f},
        {2,2,3.0f},
        {4,4,6.0f}};
    err = mtx_init_matrix_coordinate_integer(
        &mtx, mtx_general, mtx_nontriangular,
        mtx_unsorted, mtx_unordered, mtx_unassembled,
        num_comment_lines, comment_lines,
        num_rows, num_columns, size, data);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));

    /* Sort the nonzeros and verify the results. */
    enum mtx_sorting sorting = mtx_row_major;
    err = mtx_sort(&mtx, sorting);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
    TEST_ASSERT_EQ(mtx_matrix, mtx.object);
    TEST_ASSERT_EQ(mtx_coordinate, mtx.format);
    TEST_ASSERT_EQ(mtx_integer, mtx.field);
    TEST_ASSERT_EQ(mtx_general, mtx.symmetry);
    TEST_ASSERT_EQ(mtx_nontriangular, mtx.symmetry);
    TEST_ASSERT_EQ(mtx_row_major, mtx.sorting);
    TEST_ASSERT_EQ(mtx_unordered, mtx.ordering);
    TEST_ASSERT_EQ(mtx_unassembled, mtx.assembly);
    TEST_ASSERT_EQ(1, mtx.num_comment_lines);
    TEST_ASSERT_STREQ("a comment", mtx.comment_lines[0]);
    TEST_ASSERT_EQ(4, mtx.num_rows);
    TEST_ASSERT_EQ(4, mtx.num_columns);
    TEST_ASSERT_EQ(6, mtx.num_nonzeros);
    TEST_ASSERT_EQ(6, mtx.size);
    TEST_ASSERT_EQ(sizeof(struct mtx_matrix_coordinate_integer), mtx.nonzero_size);
    const struct mtx_matrix_coordinate_integer * mtxdata =
        (const struct mtx_matrix_coordinate_integer *) mtx.data;
    TEST_ASSERT_EQ(1, mtxdata[0].i);
    TEST_ASSERT_EQ(1, mtxdata[0].j);
    TEST_ASSERT_EQ(1, mtxdata[0].a);
    TEST_ASSERT_EQ(1, mtxdata[1].i);
    TEST_ASSERT_EQ(4, mtxdata[1].j);
    TEST_ASSERT_EQ(2, mtxdata[1].a);
    TEST_ASSERT_EQ(2, mtxdata[2].i);
    TEST_ASSERT_EQ(2, mtxdata[2].j);
    TEST_ASSERT_EQ(3, mtxdata[2].a);
    TEST_ASSERT_EQ(3, mtxdata[3].i);
    TEST_ASSERT_EQ(3, mtxdata[3].j);
    TEST_ASSERT_EQ(4, mtxdata[3].a);
    TEST_ASSERT_EQ(4, mtxdata[4].i);
    TEST_ASSERT_EQ(1, mtxdata[4].j);
    TEST_ASSERT_EQ(5, mtxdata[4].a);
    TEST_ASSERT_EQ(4, mtxdata[5].i);
    TEST_ASSERT_EQ(4, mtxdata[5].j);
    TEST_ASSERT_EQ(6, mtxdata[5].a);
    mtx_free(&mtx);
    return MTX_SUCCESS;
}

/**
 * `test_mtx_sort_matrix_coordinate_pattern()' tests sorting the
 * nonzeros of a pattern matrix in coordinate format.
 */
int test_mtx_sort_matrix_coordinate_pattern(void)
{
    int err;

    /* Create matrix. */
    struct mtx mtx;
    int num_comment_lines = 1;
    const char * comment_lines[] = {"a comment"};
    int num_rows = 4;
    int num_columns = 4;
    int64_t size = 6;
    const struct mtx_matrix_coordinate_pattern data[] = {
        {3,3},
        {1,4},
        {4,1},
        {1,1},
        {2,2},
        {4,4}};
    err = mtx_init_matrix_coordinate_pattern(
        &mtx, mtx_general, mtx_nontriangular,
        mtx_unsorted, mtx_unordered, mtx_unassembled,
        num_comment_lines, comment_lines,
        num_rows, num_columns, size, data);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));

    /* Sort the nonzeros and verify the results. */
    enum mtx_sorting sorting = mtx_row_major;
    err = mtx_sort(&mtx, sorting);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
    TEST_ASSERT_EQ(mtx_matrix, mtx.object);
    TEST_ASSERT_EQ(mtx_coordinate, mtx.format);
    TEST_ASSERT_EQ(mtx_pattern, mtx.field);
    TEST_ASSERT_EQ(mtx_general, mtx.symmetry);
    TEST_ASSERT_EQ(mtx_nontriangular, mtx.symmetry);
    TEST_ASSERT_EQ(mtx_row_major, mtx.sorting);
    TEST_ASSERT_EQ(mtx_unordered, mtx.ordering);
    TEST_ASSERT_EQ(mtx_unassembled, mtx.assembly);
    TEST_ASSERT_EQ(1, mtx.num_comment_lines);
    TEST_ASSERT_STREQ("a comment", mtx.comment_lines[0]);
    TEST_ASSERT_EQ(4, mtx.num_rows);
    TEST_ASSERT_EQ(4, mtx.num_columns);
    TEST_ASSERT_EQ(6, mtx.num_nonzeros);
    TEST_ASSERT_EQ(6, mtx.size);
    TEST_ASSERT_EQ(sizeof(struct mtx_matrix_coordinate_pattern), mtx.nonzero_size);
    const struct mtx_matrix_coordinate_pattern * mtxdata =
        (const struct mtx_matrix_coordinate_pattern *) mtx.data;
    TEST_ASSERT_EQ(   1, mtxdata[0].i);
    TEST_ASSERT_EQ(   1, mtxdata[0].j);
    TEST_ASSERT_EQ(   1, mtxdata[1].i);
    TEST_ASSERT_EQ(   4, mtxdata[1].j);
    TEST_ASSERT_EQ(   2, mtxdata[2].i);
    TEST_ASSERT_EQ(   2, mtxdata[2].j);
    TEST_ASSERT_EQ(   3, mtxdata[3].i);
    TEST_ASSERT_EQ(   3, mtxdata[3].j);
    TEST_ASSERT_EQ(   4, mtxdata[4].i);
    TEST_ASSERT_EQ(   1, mtxdata[4].j);
    TEST_ASSERT_EQ(   4, mtxdata[5].i);
    TEST_ASSERT_EQ(   4, mtxdata[5].j);
    mtx_free(&mtx);
    return MTX_SUCCESS;
}

/**
 * `test_mtx_sort_matrix_column_major_coordinate_double()' tests
 * sorting the nonzeros of a double matrix in coordinate format in
 * column major order.
 */
int test_mtx_sort_matrix_column_major_coordinate_double(void)
{
    int err;

    /* Create matrix. */
    struct mtx mtx;
    int num_comment_lines = 1;
    const char * comment_lines[] = {"a comment"};
    int num_rows = 4;
    int num_columns = 4;
    int64_t size = 6;
    const struct mtx_matrix_coordinate_double data[] = {
        {3,3,4.0f},
        {1,4,2.0f},
        {4,1,5.0f},
        {1,1,1.0f},
        {2,2,3.0f},
        {4,4,6.0f}};
    err = mtx_init_matrix_coordinate_double(
        &mtx, mtx_general, mtx_nontriangular,
        mtx_unsorted, mtx_unordered, mtx_unassembled,
        num_comment_lines, comment_lines,
        num_rows, num_columns, size, data);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));

    /* Sort the nonzeros and verify the results. */
    enum mtx_sorting sorting = mtx_column_major;
    err = mtx_sort(&mtx, sorting);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
    TEST_ASSERT_EQ(mtx_matrix, mtx.object);
    TEST_ASSERT_EQ(mtx_coordinate, mtx.format);
    TEST_ASSERT_EQ(mtx_double, mtx.field);
    TEST_ASSERT_EQ(mtx_general, mtx.symmetry);
    TEST_ASSERT_EQ(mtx_nontriangular, mtx.symmetry);
    TEST_ASSERT_EQ(mtx_column_major, mtx.sorting);
    TEST_ASSERT_EQ(mtx_unordered, mtx.ordering);
    TEST_ASSERT_EQ(mtx_unassembled, mtx.assembly);
    TEST_ASSERT_EQ(1, mtx.num_comment_lines);
    TEST_ASSERT_STREQ("a comment", mtx.comment_lines[0]);
    TEST_ASSERT_EQ(4, mtx.num_rows);
    TEST_ASSERT_EQ(4, mtx.num_columns);
    TEST_ASSERT_EQ(6, mtx.num_nonzeros);
    TEST_ASSERT_EQ(6, mtx.size);
    TEST_ASSERT_EQ(sizeof(struct mtx_matrix_coordinate_double), mtx.nonzero_size);
    const struct mtx_matrix_coordinate_double * mtxdata =
        (const struct mtx_matrix_coordinate_double *) mtx.data;
    TEST_ASSERT_EQ(   1, mtxdata[0].i);
    TEST_ASSERT_EQ(   1, mtxdata[0].j);
    TEST_ASSERT_EQ(1.0f, mtxdata[0].a);
    TEST_ASSERT_EQ(   4, mtxdata[1].i);
    TEST_ASSERT_EQ(   1, mtxdata[1].j);
    TEST_ASSERT_EQ(5.0f, mtxdata[1].a);
    TEST_ASSERT_EQ(   2, mtxdata[2].i);
    TEST_ASSERT_EQ(   2, mtxdata[2].j);
    TEST_ASSERT_EQ(3.0f, mtxdata[2].a);
    TEST_ASSERT_EQ(   3, mtxdata[3].i);
    TEST_ASSERT_EQ(   3, mtxdata[3].j);
    TEST_ASSERT_EQ(4.0f, mtxdata[3].a);
    TEST_ASSERT_EQ(   1, mtxdata[4].i);
    TEST_ASSERT_EQ(   4, mtxdata[4].j);
    TEST_ASSERT_EQ(2.0f, mtxdata[4].a);
    TEST_ASSERT_EQ(   4, mtxdata[5].i);
    TEST_ASSERT_EQ(   4, mtxdata[5].j);
    TEST_ASSERT_EQ(6.0f, mtxdata[5].a);
    mtx_free(&mtx);
    return MTX_SUCCESS;
}

/**
 * `main()' entry point and test driver.
 */
int main(int argc, char * argv[])
{
    TEST_SUITE_BEGIN("Running tests for Matrix Market objects\n");
    TEST_RUN(test_mtx_sort_matrix_coordinate_real);
    TEST_RUN(test_mtx_sort_matrix_coordinate_double);
    TEST_RUN(test_mtx_sort_matrix_coordinate_complex);
    TEST_RUN(test_mtx_sort_matrix_coordinate_integer);
    TEST_RUN(test_mtx_sort_matrix_coordinate_pattern);
    TEST_RUN(test_mtx_sort_matrix_column_major_coordinate_double);
    TEST_SUITE_END();
    return (TEST_SUITE_STATUS == TEST_SUCCESS) ?
        EXIT_SUCCESS : EXIT_FAILURE;
}
