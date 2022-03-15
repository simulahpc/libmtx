/* This file is part of Libmtx.
 *
 * Copyright (C) 2021 James D. Trotter
 *
 * Libmtx is free software: you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * Libmtx is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Libmtx.  If not, see
 * <https://www.gnu.org/licenses/>.
 *
 * Authors: James D. Trotter <james@simula.no>
 * Last modified: 2021-08-09
 *
 * Unit tests for transposing sparse matrices.
 */

#include "test.h"

#include <libmtx/error.h>
#include <libmtx/matrix/coordinate.h>
#include <libmtx/mtx/mtx.h>
#include <libmtx/mtx/header.h>
#include <libmtx/mtx/transpose.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * `test_mtx_transpose_matrix_coordinate_real_single_general()` tests
 * transposing non-symmetric sparse matrices with real,
 * single-precision coefficients in the Matrix Market format.
 */
int test_mtx_transpose_matrix_coordinate_real_single_general(void)
{
    int err;
    struct mtx mtx;
    int num_comment_lines = 0;
    const char * comment_lines[] = {};
    int num_rows = 4;
    int num_columns = 4;
    const struct mtx_matrix_coordinate_real_single data[] = {
        {1,1,1.0f}, {1,4,2.0f},
        {2,2,3.0f},
        {3,3,4.0f},
        {4,1,5.0f}, {4,4,6.0f}};
    size_t size = sizeof(data) / sizeof(*data);
    err = mtx_init_matrix_coordinate_real_single(
        &mtx, mtx_general_, mtx_nontriangular, mtx_unsorted, mtx_unassembled,
        num_comment_lines, comment_lines,
        num_rows, num_columns, size, data);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
    err = mtx_transpose(&mtx);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
    TEST_ASSERT_EQ(mtx_matrix, mtx.object);
    TEST_ASSERT_EQ(mtx_coordinate, mtx.format);
    TEST_ASSERT_EQ(mtx_real, mtx.field);
    TEST_ASSERT_EQ(mtx_general_, mtx.symmetry);
    TEST_ASSERT_EQ(4, mtx.num_rows);
    TEST_ASSERT_EQ(4, mtx.num_columns);
    TEST_ASSERT_EQ(6, mtx.num_nonzeros);

    const struct mtx_matrix_coordinate_data * matrix_coordinate =
        &mtx.storage.matrix_coordinate;
    TEST_ASSERT_EQ(mtx_real, matrix_coordinate->field);
    TEST_ASSERT_EQ(mtx_single, matrix_coordinate->precision);
    TEST_ASSERT_EQ(mtx_unsorted, matrix_coordinate->sorting);
    TEST_ASSERT_EQ(mtx_unassembled, matrix_coordinate->assembly);
    TEST_ASSERT_EQ(4, matrix_coordinate->num_rows);
    TEST_ASSERT_EQ(4, matrix_coordinate->num_columns);
    TEST_ASSERT_EQ(6, matrix_coordinate->size);
    const struct mtx_matrix_coordinate_real_single * mtxdata =
        matrix_coordinate->data.real_single;
    TEST_ASSERT_EQ(   1, mtxdata[0].i); TEST_ASSERT_EQ(   1, mtxdata[0].j);
    TEST_ASSERT_EQ(1.0f, mtxdata[0].a);
    TEST_ASSERT_EQ(   4, mtxdata[1].i); TEST_ASSERT_EQ(   1, mtxdata[1].j);
    TEST_ASSERT_EQ(2.0f, mtxdata[1].a);
    TEST_ASSERT_EQ(   2, mtxdata[2].i); TEST_ASSERT_EQ(   2, mtxdata[2].j);
    TEST_ASSERT_EQ(3.0f, mtxdata[2].a);
    TEST_ASSERT_EQ(   3, mtxdata[3].i); TEST_ASSERT_EQ(   3, mtxdata[3].j);
    TEST_ASSERT_EQ(4.0f, mtxdata[3].a);
    TEST_ASSERT_EQ(   1, mtxdata[4].i); TEST_ASSERT_EQ(   4, mtxdata[4].j);
    TEST_ASSERT_EQ(5.0f, mtxdata[4].a);
    TEST_ASSERT_EQ(   4, mtxdata[5].i); TEST_ASSERT_EQ(   4, mtxdata[5].j);
    TEST_ASSERT_EQ(6.0f, mtxdata[5].a);
    mtx_free(&mtx);
    return TEST_SUCCESS;
}

/**
 * `test_mtx_transpose_matrix_coordinate_real_single_symmetric()` tests
 * transposing symmetric sparse matrices with real, single-precision
 * coefficients in the Matrix Market format.
 */
int test_mtx_transpose_matrix_coordinate_real_single_symmetric(void)
{
    int err;
    struct mtx mtx;
    int num_comment_lines = 0;
    const char * comment_lines[] = {};
    int num_rows = 4;
    int num_columns = 4;
    const struct mtx_matrix_coordinate_real_single data[] = {
        {1,1,1.0f},
        {2,2,2.0f},
        {3,3,3.0f},
        {4,1,4.0f}, {4,4,5.0f}};
    size_t size = sizeof(data) / sizeof(*data);
    err = mtx_init_matrix_coordinate_real_single(
        &mtx, mtx_symmetric_, mtx_nontriangular, mtx_unsorted, mtx_unassembled,
        num_comment_lines, comment_lines,
        num_rows, num_columns, size, data);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
    err = mtx_transpose(&mtx);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
    TEST_ASSERT_EQ(mtx_matrix, mtx.object);
    TEST_ASSERT_EQ(mtx_coordinate, mtx.format);
    TEST_ASSERT_EQ(mtx_real, mtx.field);
    TEST_ASSERT_EQ(mtx_symmetric_, mtx.symmetry);
    TEST_ASSERT_EQ(4, mtx.num_rows);
    TEST_ASSERT_EQ(4, mtx.num_columns);
    TEST_ASSERT_EQ(5, mtx.num_nonzeros);

    const struct mtx_matrix_coordinate_data * matrix_coordinate =
        &mtx.storage.matrix_coordinate;
    TEST_ASSERT_EQ(mtx_real, matrix_coordinate->field);
    TEST_ASSERT_EQ(mtx_single, matrix_coordinate->precision);
    TEST_ASSERT_EQ(mtx_unsorted, matrix_coordinate->sorting);
    TEST_ASSERT_EQ(mtx_unassembled, matrix_coordinate->assembly);
    TEST_ASSERT_EQ(4, matrix_coordinate->num_rows);
    TEST_ASSERT_EQ(4, matrix_coordinate->num_columns);
    TEST_ASSERT_EQ(5, matrix_coordinate->size);
    const struct mtx_matrix_coordinate_real_single * mtxdata =
        matrix_coordinate->data.real_single;
    TEST_ASSERT_EQ(   1, mtxdata[0].i); TEST_ASSERT_EQ(   1, mtxdata[0].j);
    TEST_ASSERT_EQ(1.0f, mtxdata[0].a);
    TEST_ASSERT_EQ(   2, mtxdata[1].i); TEST_ASSERT_EQ(   2, mtxdata[1].j);
    TEST_ASSERT_EQ(2.0f, mtxdata[1].a);
    TEST_ASSERT_EQ(   3, mtxdata[2].i); TEST_ASSERT_EQ(   3, mtxdata[2].j);
    TEST_ASSERT_EQ(3.0f, mtxdata[2].a);
    TEST_ASSERT_EQ(   1, mtxdata[3].i); TEST_ASSERT_EQ(   4, mtxdata[3].j);
    TEST_ASSERT_EQ(4.0f, mtxdata[3].a);
    TEST_ASSERT_EQ(   4, mtxdata[4].i); TEST_ASSERT_EQ(   4, mtxdata[4].j);
    TEST_ASSERT_EQ(5.0f, mtxdata[4].a);
    mtx_free(&mtx);
    return TEST_SUCCESS;
}

/**
 * `test_mtx_transpose_matrix_coordinate_real_double_general()` tests
 * transposing non-symmetric sparse matrices with real,
 * double-precision coefficients in the Matrix Market format.
 */
int test_mtx_transpose_matrix_coordinate_real_double_general(void)
{
    int err;
    struct mtx mtx;
    int num_comment_lines = 0;
    const char * comment_lines[] = {};
    int num_rows = 4;
    int num_columns = 4;
    const struct mtx_matrix_coordinate_real_double data[] = {
        {1,1,1.0}, {1,4,2.0},
        {2,2,3.0},
        {3,3,4.0},
        {4,1,5.0}, {4,4,6.0}};
    size_t size = sizeof(data) / sizeof(*data);
    err = mtx_init_matrix_coordinate_real_double(
        &mtx, mtx_general_, mtx_nontriangular, mtx_unsorted, mtx_unassembled,
        num_comment_lines, comment_lines,
        num_rows, num_columns, size, data);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
    err = mtx_transpose(&mtx);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
    TEST_ASSERT_EQ(mtx_matrix, mtx.object);
    TEST_ASSERT_EQ(mtx_coordinate, mtx.format);
    TEST_ASSERT_EQ(mtx_real, mtx.field);
    TEST_ASSERT_EQ(mtx_general_, mtx.symmetry);
    TEST_ASSERT_EQ(4, mtx.num_rows);
    TEST_ASSERT_EQ(4, mtx.num_columns);
    TEST_ASSERT_EQ(6, mtx.num_nonzeros);

    const struct mtx_matrix_coordinate_data * matrix_coordinate =
        &mtx.storage.matrix_coordinate;
    TEST_ASSERT_EQ(mtx_real, matrix_coordinate->field);
    TEST_ASSERT_EQ(mtx_double, matrix_coordinate->precision);
    TEST_ASSERT_EQ(mtx_unsorted, matrix_coordinate->sorting);
    TEST_ASSERT_EQ(mtx_unassembled, matrix_coordinate->assembly);
    TEST_ASSERT_EQ(4, matrix_coordinate->num_rows);
    TEST_ASSERT_EQ(4, matrix_coordinate->num_columns);
    TEST_ASSERT_EQ(6, matrix_coordinate->size);
    const struct mtx_matrix_coordinate_real_double * mtxdata =
        matrix_coordinate->data.real_double;
    TEST_ASSERT_EQ(  1, mtxdata[0].i); TEST_ASSERT_EQ(  1, mtxdata[0].j);
    TEST_ASSERT_EQ(1.0, mtxdata[0].a);
    TEST_ASSERT_EQ(  4, mtxdata[1].i); TEST_ASSERT_EQ(  1, mtxdata[1].j);
    TEST_ASSERT_EQ(2.0, mtxdata[1].a);
    TEST_ASSERT_EQ(  2, mtxdata[2].i); TEST_ASSERT_EQ(  2, mtxdata[2].j);
    TEST_ASSERT_EQ(3.0, mtxdata[2].a);
    TEST_ASSERT_EQ(  3, mtxdata[3].i); TEST_ASSERT_EQ(  3, mtxdata[3].j);
    TEST_ASSERT_EQ(4.0, mtxdata[3].a);
    TEST_ASSERT_EQ(  1, mtxdata[4].i); TEST_ASSERT_EQ(  4, mtxdata[4].j);
    TEST_ASSERT_EQ(5.0, mtxdata[4].a);
    TEST_ASSERT_EQ(  4, mtxdata[5].i); TEST_ASSERT_EQ(  4, mtxdata[5].j);
    TEST_ASSERT_EQ(6.0, mtxdata[5].a);
    mtx_free(&mtx);
    return TEST_SUCCESS;
}

/**
 * `test_mtx_transpose_matrix_coordinate_complex_single_general()`
 * tests transposing non-symmetric sparse matrices with real,
 * single-precision coefficients in the Matrix Market format.
 */
int test_mtx_transpose_matrix_coordinate_complex_single_general(void)
{
    int err;
    struct mtx mtx;
    int num_comment_lines = 0;
    const char * comment_lines[] = {};
    int num_rows = 4;
    int num_columns = 4;
    const struct mtx_matrix_coordinate_complex_single data[] = {
        {1,1,1.0,-1.0}, {1,4,2.0,-2.0},
        {2,2,3.0,-3.0},
        {3,3,4.0,-4.0},
        {4,1,5.0,-5.0}, {4,4,6.0,-6.0}};
    size_t size = sizeof(data) / sizeof(*data);
    err = mtx_init_matrix_coordinate_complex_single(
        &mtx, mtx_general_, mtx_nontriangular, mtx_unsorted, mtx_unassembled,
        num_comment_lines, comment_lines,
        num_rows, num_columns, size, data);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
    err = mtx_transpose(&mtx);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
    TEST_ASSERT_EQ(mtx_matrix, mtx.object);
    TEST_ASSERT_EQ(mtx_coordinate, mtx.format);
    TEST_ASSERT_EQ(mtx_complex, mtx.field);
    TEST_ASSERT_EQ(mtx_general_, mtx.symmetry);
    TEST_ASSERT_EQ(4, mtx.num_rows);
    TEST_ASSERT_EQ(4, mtx.num_columns);
    TEST_ASSERT_EQ(6, mtx.num_nonzeros);

    const struct mtx_matrix_coordinate_data * matrix_coordinate =
        &mtx.storage.matrix_coordinate;
    TEST_ASSERT_EQ(mtx_complex, matrix_coordinate->field);
    TEST_ASSERT_EQ(mtx_single, matrix_coordinate->precision);
    TEST_ASSERT_EQ(mtx_unsorted, matrix_coordinate->sorting);
    TEST_ASSERT_EQ(mtx_unassembled, matrix_coordinate->assembly);
    TEST_ASSERT_EQ(4, matrix_coordinate->num_rows);
    TEST_ASSERT_EQ(4, matrix_coordinate->num_columns);
    TEST_ASSERT_EQ(6, matrix_coordinate->size);
    const struct mtx_matrix_coordinate_complex_single * mtxdata =
        matrix_coordinate->data.complex_single;
    TEST_ASSERT_EQ(   1, mtxdata[0].i); TEST_ASSERT_EQ(   1, mtxdata[0].j);
    TEST_ASSERT_EQ( 1.0, mtxdata[0].a[0]); TEST_ASSERT_EQ(-1.0, mtxdata[0].a[1]);
    TEST_ASSERT_EQ(   4, mtxdata[1].i); TEST_ASSERT_EQ(   1, mtxdata[1].j);
    TEST_ASSERT_EQ( 2.0, mtxdata[1].a[0]); TEST_ASSERT_EQ(-2.0, mtxdata[1].a[1]);
    TEST_ASSERT_EQ(   2, mtxdata[2].i); TEST_ASSERT_EQ(   2, mtxdata[2].j);
    TEST_ASSERT_EQ( 3.0, mtxdata[2].a[0]); TEST_ASSERT_EQ(-3.0, mtxdata[2].a[1]);
    TEST_ASSERT_EQ(   3, mtxdata[3].i); TEST_ASSERT_EQ(   3, mtxdata[3].j);
    TEST_ASSERT_EQ( 4.0, mtxdata[3].a[0]); TEST_ASSERT_EQ(-4.0, mtxdata[3].a[1]);
    TEST_ASSERT_EQ(   1, mtxdata[4].i); TEST_ASSERT_EQ(   4, mtxdata[4].j);
    TEST_ASSERT_EQ( 5.0, mtxdata[4].a[0]); TEST_ASSERT_EQ(-5.0, mtxdata[4].a[1]);
    TEST_ASSERT_EQ(   4, mtxdata[5].i); TEST_ASSERT_EQ(   4, mtxdata[5].j);
    TEST_ASSERT_EQ( 6.0, mtxdata[5].a[0]); TEST_ASSERT_EQ(-6.0, mtxdata[5].a[1]);
    mtx_free(&mtx);
    return TEST_SUCCESS;
}

/**
 * `test_mtx_transpose_matrix_coordinate_integer_single_general()`
 * tests transposing non-symmetric sparse matrices with single
 * precision, integer coefficients in the Matrix Market format.
 */
int test_mtx_transpose_matrix_coordinate_integer_single_general(void)
{
    int err;
    struct mtx mtx;
    int num_comment_lines = 0;
    const char * comment_lines[] = {};
    int num_rows = 4;
    int num_columns = 4;
    const struct mtx_matrix_coordinate_integer_single data[] = {
        {1,1,1}, {1,4,2},
        {2,2,3},
        {3,3,4},
        {4,1,5}, {4,4,6}};
    size_t size = sizeof(data) / sizeof(*data);
    err = mtx_init_matrix_coordinate_integer_single(
        &mtx, mtx_general_, mtx_nontriangular, mtx_unsorted, mtx_unassembled,
        num_comment_lines, comment_lines,
        num_rows, num_columns, size, data);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
    err = mtx_transpose(&mtx);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
    TEST_ASSERT_EQ(mtx_matrix, mtx.object);
    TEST_ASSERT_EQ(mtx_coordinate, mtx.format);
    TEST_ASSERT_EQ(mtx_integer, mtx.field);
    TEST_ASSERT_EQ(mtx_general_, mtx.symmetry);
    TEST_ASSERT_EQ(4, mtx.num_rows);
    TEST_ASSERT_EQ(4, mtx.num_columns);
    TEST_ASSERT_EQ(6, mtx.num_nonzeros);

    const struct mtx_matrix_coordinate_data * matrix_coordinate =
        &mtx.storage.matrix_coordinate;
    TEST_ASSERT_EQ(mtx_integer, matrix_coordinate->field);
    TEST_ASSERT_EQ(mtx_single, matrix_coordinate->precision);
    TEST_ASSERT_EQ(mtx_unsorted, matrix_coordinate->sorting);
    TEST_ASSERT_EQ(mtx_unassembled, matrix_coordinate->assembly);
    TEST_ASSERT_EQ(4, matrix_coordinate->num_rows);
    TEST_ASSERT_EQ(4, matrix_coordinate->num_columns);
    TEST_ASSERT_EQ(6, matrix_coordinate->size);
    const struct mtx_matrix_coordinate_integer_single * mtxdata =
        matrix_coordinate->data.integer_single;
    TEST_ASSERT_EQ(1, mtxdata[0].i); TEST_ASSERT_EQ(1, mtxdata[0].j);
    TEST_ASSERT_EQ(1, mtxdata[0].a);
    TEST_ASSERT_EQ(4, mtxdata[1].i); TEST_ASSERT_EQ(1, mtxdata[1].j);
    TEST_ASSERT_EQ(2, mtxdata[1].a);
    TEST_ASSERT_EQ(2, mtxdata[2].i); TEST_ASSERT_EQ(2, mtxdata[2].j);
    TEST_ASSERT_EQ(3, mtxdata[2].a);
    TEST_ASSERT_EQ(3, mtxdata[3].i); TEST_ASSERT_EQ(3, mtxdata[3].j);
    TEST_ASSERT_EQ(4, mtxdata[3].a);
    TEST_ASSERT_EQ(1, mtxdata[4].i); TEST_ASSERT_EQ(4, mtxdata[4].j);
    TEST_ASSERT_EQ(5, mtxdata[4].a);
    TEST_ASSERT_EQ(4, mtxdata[5].i); TEST_ASSERT_EQ(4, mtxdata[5].j);
    TEST_ASSERT_EQ(6, mtxdata[5].a);
    mtx_free(&mtx);
    return TEST_SUCCESS;
}

/**
 * `test_mtx_transpose_matrix_coordinate_pattern_general()` tests
 * transposing non-symmetric sparse matrices with boolean coefficients
 * in the Matrix Market format.
 */
int test_mtx_transpose_matrix_coordinate_pattern_general(void)
{
    int err;
    struct mtx mtx;
    int num_comment_lines = 0;
    const char * comment_lines[] = {};
    int num_rows = 4;
    int num_columns = 4;
    const struct mtx_matrix_coordinate_pattern data[] = {
        {1,1}, {1,4},
        {2,2},
        {3,3},
        {4,1}, {4,4}};
    size_t size = sizeof(data) / sizeof(*data);
    err = mtx_init_matrix_coordinate_pattern(
        &mtx, mtx_general_, mtx_nontriangular, mtx_unsorted, mtx_unassembled,
        num_comment_lines, comment_lines,
        num_rows, num_columns, size, data);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
    err = mtx_transpose(&mtx);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
    TEST_ASSERT_EQ(mtx_matrix, mtx.object);
    TEST_ASSERT_EQ(mtx_coordinate, mtx.format);
    TEST_ASSERT_EQ(mtx_pattern, mtx.field);
    TEST_ASSERT_EQ(mtx_general_, mtx.symmetry);
    TEST_ASSERT_EQ(4, mtx.num_rows);
    TEST_ASSERT_EQ(4, mtx.num_columns);
    TEST_ASSERT_EQ(6, mtx.num_nonzeros);

    const struct mtx_matrix_coordinate_data * matrix_coordinate =
        &mtx.storage.matrix_coordinate;
    TEST_ASSERT_EQ(mtx_pattern, matrix_coordinate->field);
    TEST_ASSERT_EQ(mtx_single, matrix_coordinate->precision);
    TEST_ASSERT_EQ(mtx_unsorted, matrix_coordinate->sorting);
    TEST_ASSERT_EQ(mtx_unassembled, matrix_coordinate->assembly);
    TEST_ASSERT_EQ(4, matrix_coordinate->num_rows);
    TEST_ASSERT_EQ(4, matrix_coordinate->num_columns);
    TEST_ASSERT_EQ(6, matrix_coordinate->size);
    const struct mtx_matrix_coordinate_pattern * mtxdata =
        matrix_coordinate->data.pattern;
    TEST_ASSERT_EQ(1, mtxdata[0].i); TEST_ASSERT_EQ(1, mtxdata[0].j);
    TEST_ASSERT_EQ(4, mtxdata[1].i); TEST_ASSERT_EQ(1, mtxdata[1].j);
    TEST_ASSERT_EQ(2, mtxdata[2].i); TEST_ASSERT_EQ(2, mtxdata[2].j);
    TEST_ASSERT_EQ(3, mtxdata[3].i); TEST_ASSERT_EQ(3, mtxdata[3].j);
    TEST_ASSERT_EQ(1, mtxdata[4].i); TEST_ASSERT_EQ(4, mtxdata[4].j);
    TEST_ASSERT_EQ(4, mtxdata[5].i); TEST_ASSERT_EQ(4, mtxdata[5].j);
    mtx_free(&mtx);
    return TEST_SUCCESS;
}

/**
 * `main()' entry point and test driver.
 */
int main(int argc, char * argv[])
{
    TEST_SUITE_BEGIN(
        "Running tests for transposing sparse matrices in Matrix Market format.\n");
    TEST_RUN(test_mtx_transpose_matrix_coordinate_real_single_general);
    TEST_RUN(test_mtx_transpose_matrix_coordinate_real_single_symmetric);
    TEST_RUN(test_mtx_transpose_matrix_coordinate_real_double_general);
    TEST_RUN(test_mtx_transpose_matrix_coordinate_complex_single_general);
    TEST_RUN(test_mtx_transpose_matrix_coordinate_integer_single_general);
    TEST_RUN(test_mtx_transpose_matrix_coordinate_pattern_general);
    TEST_SUITE_END();
    return (TEST_SUITE_STATUS == TEST_SUCCESS) ?
        EXIT_SUCCESS : EXIT_FAILURE;
}
