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
 * Unit tests for sparse matrices in Matrix Market format.
 */

#include "test.h"

#include <libmtx/error.h>
#include <libmtx/mtx/matrix.h>
#include <libmtx/matrix/coordinate.h>
#include <libmtx/mtx/mtx.h>
#include <libmtx/mtx/header.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * `test_mtx_init_matrix_coordinate_real_general()` tests creating non-symmetric
 * sparse matrices with real, single-precision coefficients in the
 * Matrix Market format.
 */
int test_mtx_init_matrix_coordinate_real_general(void)
{
    int err;
    struct mtx matrix;
    int num_comment_lines = 1;
    const char * comment_lines[] = {"% a comment"};
    int num_rows = 4;
    int num_columns = 4;
    int64_t size = 6;
    const struct mtx_matrix_coordinate_real data[] = {
        {1,1,1.0f}, {1,4,2.0f},
        {2,2,3.0f},
        {3,3,4.0f},
        {4,1,5.0f}, {4,4,6.0f}};
    err = mtx_init_matrix_coordinate_real(
        &matrix, mtx_general, mtx_nontriangular,
        mtx_unsorted, mtx_unordered, mtx_unassembled,
        num_comment_lines, comment_lines,
        num_rows, num_columns, size, data);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
    TEST_ASSERT_EQ(mtx_matrix, matrix.object);
    TEST_ASSERT_EQ(mtx_coordinate, matrix.format);
    TEST_ASSERT_EQ(mtx_real, matrix.field);
    TEST_ASSERT_EQ(mtx_general, matrix.symmetry);
    TEST_ASSERT_EQ(mtx_nontriangular, matrix.triangle);
    TEST_ASSERT_EQ(mtx_unsorted, matrix.sorting);
    TEST_ASSERT_EQ(mtx_unordered, matrix.ordering);
    TEST_ASSERT_EQ(mtx_unassembled, matrix.assembly);
    TEST_ASSERT_EQ(1, matrix.num_comment_lines);
    TEST_ASSERT_STREQ("% a comment", matrix.comment_lines[0]);
    TEST_ASSERT_EQ(4, matrix.num_rows);
    TEST_ASSERT_EQ(4, matrix.num_columns);
    TEST_ASSERT_EQ(-1, matrix.num_nonzeros);
    TEST_ASSERT_EQ(6, matrix.size);
    TEST_ASSERT_EQ(sizeof(struct mtx_matrix_coordinate_real), matrix.nonzero_size);
    const struct mtx_matrix_coordinate_real * mtxdata =
        (const struct mtx_matrix_coordinate_real *) matrix.data;
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
    if (!err)
        mtx_free(&matrix);
    return TEST_SUCCESS;
}

/**
 * `test_mtx_init_matrix_coordinate_real_symmetric()` tests creating symmetric
 * sparse matrices with real, single-precision coefficients in the
 * Matrix Market format.
 */
int test_mtx_init_matrix_coordinate_real_symmetric(void)
{
    int err;
    struct mtx matrix;
    int num_comment_lines = 1;
    const char * comment_lines[] = {"% a comment"};
    int num_rows = 4;
    int num_columns = 4;
    int64_t size = 5;
    const struct mtx_matrix_coordinate_real data[] = {
        {1,1,1.0f},
        {2,2,2.0f},
        {3,3,3.0f},
        {4,1,4.0f}, {4,4,5.0f}};
    err = mtx_init_matrix_coordinate_real(
        &matrix, mtx_symmetric, mtx_lower_triangular,
        mtx_unsorted, mtx_unordered, mtx_unassembled,
        num_comment_lines, comment_lines,
        num_rows, num_columns, size, data);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
    TEST_ASSERT_EQ(mtx_matrix, matrix.object);
    TEST_ASSERT_EQ(mtx_coordinate, matrix.format);
    TEST_ASSERT_EQ(mtx_real, matrix.field);
    TEST_ASSERT_EQ(mtx_symmetric, matrix.symmetry);
    TEST_ASSERT_EQ(mtx_lower_triangular, matrix.triangle);
    TEST_ASSERT_EQ(mtx_unsorted, matrix.sorting);
    TEST_ASSERT_EQ(mtx_unordered, matrix.ordering);
    TEST_ASSERT_EQ(mtx_unassembled, matrix.assembly);
    TEST_ASSERT_EQ(1, matrix.num_comment_lines);
    TEST_ASSERT_STREQ("% a comment", matrix.comment_lines[0]);
    TEST_ASSERT_EQ(4, matrix.num_rows);
    TEST_ASSERT_EQ(4, matrix.num_columns);
    TEST_ASSERT_EQ(-1, matrix.num_nonzeros);
    TEST_ASSERT_EQ(5, matrix.size);
    TEST_ASSERT_EQ(sizeof(struct mtx_matrix_coordinate_real), matrix.nonzero_size);
    const struct mtx_matrix_coordinate_real * mtxdata =
        (const struct mtx_matrix_coordinate_real *) matrix.data;
    TEST_ASSERT_EQ(   1, mtxdata[0].i); TEST_ASSERT_EQ(   1, mtxdata[0].j);
    TEST_ASSERT_EQ(1.0f, mtxdata[0].a);
    TEST_ASSERT_EQ(   2, mtxdata[1].i); TEST_ASSERT_EQ(   2, mtxdata[1].j);
    TEST_ASSERT_EQ(2.0f, mtxdata[1].a);
    TEST_ASSERT_EQ(   3, mtxdata[2].i); TEST_ASSERT_EQ(   3, mtxdata[2].j);
    TEST_ASSERT_EQ(3.0f, mtxdata[2].a);
    TEST_ASSERT_EQ(   4, mtxdata[3].i); TEST_ASSERT_EQ(   1, mtxdata[3].j);
    TEST_ASSERT_EQ(4.0f, mtxdata[3].a);
    TEST_ASSERT_EQ(   4, mtxdata[4].i); TEST_ASSERT_EQ(   4, mtxdata[4].j);
    TEST_ASSERT_EQ(5.0f, mtxdata[4].a);
    if (!err)
        mtx_free(&matrix);
    return TEST_SUCCESS;
}

/**
 * `test_mtx_init_matrix_coordinate_double_general()` tests creating
 * non-symmetric sparse matrices with real, double-precision
 * coefficients in the Matrix Market format.
 */
int test_mtx_init_matrix_coordinate_double_general(void)
{
    int err;
    struct mtx matrix;
    int num_comment_lines = 1;
    const char * comment_lines[] = {"% a comment"};
    int num_rows = 4;
    int num_columns = 4;
    int64_t size = 6;
    const struct mtx_matrix_coordinate_double data[] = {
        {1,1,1.0}, {1,4,2.0},
        {2,2,3.0},
        {3,3,4.0},
        {4,1,5.0}, {4,4,6.0}};
    err = mtx_init_matrix_coordinate_double(
        &matrix, mtx_general, mtx_nontriangular,
        mtx_unsorted, mtx_unordered, mtx_unassembled,
        num_comment_lines, comment_lines,
        num_rows, num_columns, size, data);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
    TEST_ASSERT_EQ(mtx_matrix, matrix.object);
    TEST_ASSERT_EQ(mtx_coordinate, matrix.format);
    TEST_ASSERT_EQ(mtx_double, matrix.field);
    TEST_ASSERT_EQ(mtx_general, matrix.symmetry);
    TEST_ASSERT_EQ(mtx_nontriangular, matrix.triangle);
    TEST_ASSERT_EQ(mtx_unsorted, matrix.sorting);
    TEST_ASSERT_EQ(mtx_unordered, matrix.ordering);
    TEST_ASSERT_EQ(mtx_unassembled, matrix.assembly);
    TEST_ASSERT_EQ(1, matrix.num_comment_lines);
    TEST_ASSERT_STREQ("% a comment", matrix.comment_lines[0]);
    TEST_ASSERT_EQ(4, matrix.num_rows);
    TEST_ASSERT_EQ(4, matrix.num_columns);
    TEST_ASSERT_EQ(-1, matrix.num_nonzeros);
    TEST_ASSERT_EQ(6, matrix.size);
    TEST_ASSERT_EQ(sizeof(struct mtx_matrix_coordinate_double), matrix.nonzero_size);
    const struct mtx_matrix_coordinate_double * mtxdata =
        (const struct mtx_matrix_coordinate_double *) matrix.data;
    TEST_ASSERT_EQ(  1, mtxdata[0].i); TEST_ASSERT_EQ(  1, mtxdata[0].j);
    TEST_ASSERT_EQ(1.0, mtxdata[0].a);
    TEST_ASSERT_EQ(  1, mtxdata[1].i); TEST_ASSERT_EQ(  4, mtxdata[1].j);
    TEST_ASSERT_EQ(2.0, mtxdata[1].a);
    TEST_ASSERT_EQ(  2, mtxdata[2].i); TEST_ASSERT_EQ(  2, mtxdata[2].j);
    TEST_ASSERT_EQ(3.0, mtxdata[2].a);
    TEST_ASSERT_EQ(  3, mtxdata[3].i); TEST_ASSERT_EQ(  3, mtxdata[3].j);
    TEST_ASSERT_EQ(4.0, mtxdata[3].a);
    TEST_ASSERT_EQ(  4, mtxdata[4].i); TEST_ASSERT_EQ(  1, mtxdata[4].j);
    TEST_ASSERT_EQ(5.0, mtxdata[4].a);
    TEST_ASSERT_EQ(  4, mtxdata[5].i); TEST_ASSERT_EQ(  4, mtxdata[5].j);
    TEST_ASSERT_EQ(6.0, mtxdata[5].a);
    if (!err)
        mtx_free(&matrix);
    return TEST_SUCCESS;
}

/**
 * `test_mtx_init_matrix_coordinate_complex_general()` tests creating
 * non-symmetric sparse matrices with complex, single-precision
 * coefficients in the Matrix Market format.
 */
int test_mtx_init_matrix_coordinate_complex_general(void)
{
    int err;
    struct mtx matrix;
    int num_comment_lines = 1;
    const char * comment_lines[] = {"% a comment"};
    int num_rows = 4;
    int num_columns = 4;
    int64_t size = 6;
    const struct mtx_matrix_coordinate_complex data[] = {
        {1,1,1.0f,-1.0f}, {1,4,2.0f,-2.0f},
        {2,2,3.0f,-3.0f},
        {3,3,4.0f,-4.0f},
        {4,1,5.0f,-5.0f}, {4,4,6.0f,-6.0f}};
    err = mtx_init_matrix_coordinate_complex(
        &matrix, mtx_general, mtx_nontriangular,
        mtx_unsorted, mtx_unordered, mtx_unassembled,
        num_comment_lines, comment_lines,
        num_rows, num_columns, size, data);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
    TEST_ASSERT_EQ(mtx_matrix, matrix.object);
    TEST_ASSERT_EQ(mtx_coordinate, matrix.format);
    TEST_ASSERT_EQ(mtx_complex, matrix.field);
    TEST_ASSERT_EQ(mtx_general, matrix.symmetry);
    TEST_ASSERT_EQ(mtx_nontriangular, matrix.triangle);
    TEST_ASSERT_EQ(mtx_unsorted, matrix.sorting);
    TEST_ASSERT_EQ(mtx_unordered, matrix.ordering);
    TEST_ASSERT_EQ(mtx_unassembled, matrix.assembly);
    TEST_ASSERT_EQ(1, matrix.num_comment_lines);
    TEST_ASSERT_STREQ("% a comment", matrix.comment_lines[0]);
    TEST_ASSERT_EQ(4, matrix.num_rows);
    TEST_ASSERT_EQ(4, matrix.num_columns);
    TEST_ASSERT_EQ(-1, matrix.num_nonzeros);
    TEST_ASSERT_EQ(6, matrix.size);
    TEST_ASSERT_EQ(sizeof(struct mtx_matrix_coordinate_complex), matrix.nonzero_size);
    const struct mtx_matrix_coordinate_complex * mtxdata =
        (const struct mtx_matrix_coordinate_complex *) matrix.data;
    TEST_ASSERT_EQ(    1, mtxdata[0].i); TEST_ASSERT_EQ(    1, mtxdata[0].j);
    TEST_ASSERT_EQ( 1.0f, mtxdata[0].a); TEST_ASSERT_EQ(-1.0f, mtxdata[0].b);
    TEST_ASSERT_EQ(    1, mtxdata[1].i); TEST_ASSERT_EQ(    4, mtxdata[1].j);
    TEST_ASSERT_EQ( 2.0f, mtxdata[1].a); TEST_ASSERT_EQ(-2.0f, mtxdata[1].b);
    TEST_ASSERT_EQ(    2, mtxdata[2].i); TEST_ASSERT_EQ(    2, mtxdata[2].j);
    TEST_ASSERT_EQ( 3.0f, mtxdata[2].a); TEST_ASSERT_EQ(-3.0f, mtxdata[2].b);
    TEST_ASSERT_EQ(    3, mtxdata[3].i); TEST_ASSERT_EQ(    3, mtxdata[3].j);
    TEST_ASSERT_EQ( 4.0f, mtxdata[3].a); TEST_ASSERT_EQ(-4.0f, mtxdata[3].b);
    TEST_ASSERT_EQ(    4, mtxdata[4].i); TEST_ASSERT_EQ(    1, mtxdata[4].j);
    TEST_ASSERT_EQ( 5.0f, mtxdata[4].a); TEST_ASSERT_EQ(-5.0f, mtxdata[4].b);
    TEST_ASSERT_EQ(    4, mtxdata[5].i); TEST_ASSERT_EQ(    4, mtxdata[5].j);
    TEST_ASSERT_EQ( 6.0f, mtxdata[5].a); TEST_ASSERT_EQ(-6.0f, mtxdata[5].b);
    if (!err)
        mtx_free(&matrix);
    return TEST_SUCCESS;
}

/**
 * `test_mtx_init_matrix_coordinate_integer_general()` tests creating non-symmetric
 * sparse matrices with integer, single-precision coefficients in the
 * Matrix Market format.
 */
int test_mtx_init_matrix_coordinate_integer_general(void)
{
    int err;
    struct mtx matrix;
    int num_comment_lines = 1;
    const char * comment_lines[] = {"% a comment"};
    int num_rows = 4;
    int num_columns = 4;
    int64_t size = 6;
    const struct mtx_matrix_coordinate_integer data[] = {
        {1,1,1}, {1,4,2},
        {2,2,3},
        {3,3,4},
        {4,1,5}, {4,4,6}};
    err = mtx_init_matrix_coordinate_integer(
        &matrix, mtx_general, mtx_nontriangular,
        mtx_unsorted, mtx_unordered, mtx_unassembled,
        num_comment_lines, comment_lines,
        num_rows, num_columns, size, data);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
    TEST_ASSERT_EQ(mtx_matrix, matrix.object);
    TEST_ASSERT_EQ(mtx_coordinate, matrix.format);
    TEST_ASSERT_EQ(mtx_integer, matrix.field);
    TEST_ASSERT_EQ(mtx_general, matrix.symmetry);
    TEST_ASSERT_EQ(mtx_nontriangular, matrix.triangle);
    TEST_ASSERT_EQ(mtx_unsorted, matrix.sorting);
    TEST_ASSERT_EQ(mtx_unordered, matrix.ordering);
    TEST_ASSERT_EQ(mtx_unassembled, matrix.assembly);
    TEST_ASSERT_EQ(1, matrix.num_comment_lines);
    TEST_ASSERT_STREQ("% a comment", matrix.comment_lines[0]);
    TEST_ASSERT_EQ(4, matrix.num_rows);
    TEST_ASSERT_EQ(4, matrix.num_columns);
    TEST_ASSERT_EQ(-1, matrix.num_nonzeros);
    TEST_ASSERT_EQ(6, matrix.size);
    TEST_ASSERT_EQ(sizeof(struct mtx_matrix_coordinate_integer), matrix.nonzero_size);
    const struct mtx_matrix_coordinate_integer * mtxdata =
        (const struct mtx_matrix_coordinate_integer *) matrix.data;
    TEST_ASSERT_EQ(1, mtxdata[0].i); TEST_ASSERT_EQ(1, mtxdata[0].j);
    TEST_ASSERT_EQ(1, mtxdata[0].a);
    TEST_ASSERT_EQ(1, mtxdata[1].i); TEST_ASSERT_EQ(4, mtxdata[1].j);
    TEST_ASSERT_EQ(2, mtxdata[1].a);
    TEST_ASSERT_EQ(2, mtxdata[2].i); TEST_ASSERT_EQ(2, mtxdata[2].j);
    TEST_ASSERT_EQ(3, mtxdata[2].a);
    TEST_ASSERT_EQ(3, mtxdata[3].i); TEST_ASSERT_EQ(3, mtxdata[3].j);
    TEST_ASSERT_EQ(4, mtxdata[3].a);
    TEST_ASSERT_EQ(4, mtxdata[4].i); TEST_ASSERT_EQ(1, mtxdata[4].j);
    TEST_ASSERT_EQ(5, mtxdata[4].a);
    TEST_ASSERT_EQ(4, mtxdata[5].i); TEST_ASSERT_EQ(4, mtxdata[5].j);
    TEST_ASSERT_EQ(6, mtxdata[5].a);
    if (!err)
        mtx_free(&matrix);
    return TEST_SUCCESS;
}

/**
 * `test_mtx_init_matrix_coordinate_pattern_general()` tests creating
 * non-symmetric sparse matrices with boolean coefficients in the
 * Matrix Market format.
 */
int test_mtx_init_matrix_coordinate_pattern_general(void)
{
    int err;
    struct mtx matrix;
    int num_comment_lines = 1;
    const char * comment_lines[] = {"% a comment"};
    int num_rows = 4;
    int num_columns = 4;
    int64_t size = 6;
    const struct mtx_matrix_coordinate_pattern data[] = {
        {1,1}, {1,4},
        {2,2},
        {3,3},
        {4,1}, {4,4}};
    err = mtx_init_matrix_coordinate_pattern(
        &matrix, mtx_general, mtx_nontriangular,
        mtx_unsorted, mtx_unordered, mtx_unassembled,
        num_comment_lines, comment_lines,
        num_rows, num_columns, size, data);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
    TEST_ASSERT_EQ(mtx_matrix, matrix.object);
    TEST_ASSERT_EQ(mtx_coordinate, matrix.format);
    TEST_ASSERT_EQ(mtx_pattern, matrix.field);
    TEST_ASSERT_EQ(mtx_general, matrix.symmetry);
    TEST_ASSERT_EQ(mtx_nontriangular, matrix.triangle);
    TEST_ASSERT_EQ(mtx_unsorted, matrix.sorting);
    TEST_ASSERT_EQ(mtx_unordered, matrix.ordering);
    TEST_ASSERT_EQ(mtx_unassembled, matrix.assembly);
    TEST_ASSERT_EQ(1, matrix.num_comment_lines);
    TEST_ASSERT_STREQ("% a comment", matrix.comment_lines[0]);
    TEST_ASSERT_EQ(4, matrix.num_rows);
    TEST_ASSERT_EQ(4, matrix.num_columns);
    TEST_ASSERT_EQ(-1, matrix.num_nonzeros);
    TEST_ASSERT_EQ(6, matrix.size);
    TEST_ASSERT_EQ(sizeof(struct mtx_matrix_coordinate_pattern), matrix.nonzero_size);
    const struct mtx_matrix_coordinate_pattern * mtxdata =
        (const struct mtx_matrix_coordinate_pattern *) matrix.data;
    TEST_ASSERT_EQ(1, mtxdata[0].i); TEST_ASSERT_EQ(1, mtxdata[0].j);
    TEST_ASSERT_EQ(1, mtxdata[1].i); TEST_ASSERT_EQ(4, mtxdata[1].j);
    TEST_ASSERT_EQ(2, mtxdata[2].i); TEST_ASSERT_EQ(2, mtxdata[2].j);
    TEST_ASSERT_EQ(3, mtxdata[3].i); TEST_ASSERT_EQ(3, mtxdata[3].j);
    TEST_ASSERT_EQ(4, mtxdata[4].i); TEST_ASSERT_EQ(1, mtxdata[4].j);
    TEST_ASSERT_EQ(4, mtxdata[5].i); TEST_ASSERT_EQ(4, mtxdata[5].j);
    if (!err)
        mtx_free(&matrix);
    return TEST_SUCCESS;
}

/**
 * `test_mtx_matrix_row_ptr_coordinate_real()` tests computing row
 * pointers of a sparse matrix with real, single-precision
 * coefficients in the Matrix Market format.
 */
int test_mtx_matrix_row_ptr_coordinate_real(void)
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

    int64_t * row_ptr = malloc((mtx.num_rows+1) * sizeof(int64_t));
    TEST_ASSERT(row_ptr);
    err = mtx_matrix_row_ptr(&mtx, row_ptr);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
    TEST_ASSERT_EQ(0, row_ptr[0]);
    TEST_ASSERT_EQ(2, row_ptr[1]);
    TEST_ASSERT_EQ(3, row_ptr[2]);
    TEST_ASSERT_EQ(4, row_ptr[3]);
    TEST_ASSERT_EQ(6, row_ptr[4]);
    free(row_ptr);
    mtx_free(&mtx);
    return TEST_SUCCESS;
}

/**
 * `main()' entry point and test driver.
 */
int main(int argc, char * argv[])
{
    TEST_SUITE_BEGIN(
        "Running tests for sparse matrices in Matrix Market format.\n");
    TEST_RUN(test_mtx_init_matrix_coordinate_real_general);
    TEST_RUN(test_mtx_init_matrix_coordinate_real_symmetric);
    TEST_RUN(test_mtx_init_matrix_coordinate_double_general);
    TEST_RUN(test_mtx_init_matrix_coordinate_complex_general);
    TEST_RUN(test_mtx_init_matrix_coordinate_integer_general);
    TEST_RUN(test_mtx_init_matrix_coordinate_pattern_general);
    TEST_RUN(test_mtx_matrix_row_ptr_coordinate_real);
    TEST_SUITE_END();
    return (TEST_SUITE_STATUS == TEST_SUCCESS) ?
        EXIT_SUCCESS : EXIT_FAILURE;
}
