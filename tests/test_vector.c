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
 * Unit tests for vectors in Matrix Market format.
 */

#include "test.h"

#include <libmtx/error.h>
#include <libmtx/mtx/mtx.h>
#include <libmtx/vector/array.h>
#include <libmtx/vector/coordinate.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * `test_mtx_alloc_vector_array_real_single()` tests allocating a
 * single precision, real vector in array format.
 */
int test_mtx_alloc_vector_array_real_single(void)
{
    int err;
    struct mtx mtx;
    int num_comment_lines = 1;
    const char * comment_lines[] = {"% a comment\n"};
    float data[] = {1.0f, 2.0f, 3.0f};
    int size = sizeof(data) / sizeof(*data);
    err = mtx_alloc_vector_array(
        &mtx, mtx_real, mtx_single, num_comment_lines, comment_lines, size);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
    TEST_ASSERT_EQ(mtx_vector, mtx.object);
    TEST_ASSERT_EQ(mtx_array, mtx.format);
    TEST_ASSERT_EQ(mtx_real, mtx.field);
    TEST_ASSERT_EQ(mtx_general, mtx.symmetry);
    TEST_ASSERT_EQ(1, mtx.num_comment_lines);
    TEST_ASSERT_STREQ("% a comment\n", mtx.comment_lines[0]);
    TEST_ASSERT_EQ(3, mtx.num_rows);
    TEST_ASSERT_EQ(-1, mtx.num_columns);
    TEST_ASSERT_EQ(-1, mtx.num_nonzeros);

    const struct mtx_vector_array_data * vector_array =
        &mtx.storage.vector_array;
    TEST_ASSERT_EQ(mtx_real, vector_array->field);
    TEST_ASSERT_EQ(mtx_single, vector_array->precision);
    TEST_ASSERT_EQ(3, vector_array->size);
    const float * mtxdata = vector_array->data.real_single;
    TEST_ASSERT_NEQ(NULL, mtxdata);
    mtx_free(&mtx);
    return TEST_SUCCESS;
}

/**
 * `test_mtx_init_vector_array_real_single()` tests creating a single
 * precision, real vector in array format.
 */
int test_mtx_init_vector_array_real_single(void)
{
    int err;
    struct mtx mtx;
    int num_comment_lines = 1;
    const char * comment_lines[] = {"% a comment\n"};
    float data[] = {1.0f, 2.0f, 3.0f};
    int size = sizeof(data) / sizeof(*data);
    err = mtx_init_vector_array_real_single(
        &mtx, num_comment_lines, comment_lines, size, data);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
    TEST_ASSERT_EQ(mtx_vector, mtx.object);
    TEST_ASSERT_EQ(mtx_array, mtx.format);
    TEST_ASSERT_EQ(mtx_real, mtx.field);
    TEST_ASSERT_EQ(mtx_general, mtx.symmetry);
    TEST_ASSERT_EQ(1, mtx.num_comment_lines);
    TEST_ASSERT_STREQ("% a comment\n", mtx.comment_lines[0]);
    TEST_ASSERT_EQ(3, mtx.num_rows);
    TEST_ASSERT_EQ(-1, mtx.num_columns);
    TEST_ASSERT_EQ(-1, mtx.num_nonzeros);

    const struct mtx_vector_array_data * vector_array =
        &mtx.storage.vector_array;
    TEST_ASSERT_EQ(mtx_real, vector_array->field);
    TEST_ASSERT_EQ(mtx_single, vector_array->precision);
    TEST_ASSERT_EQ(3, vector_array->size);
    const float * mtxdata = vector_array->data.real_single;
    TEST_ASSERT_EQ(1.0f, mtxdata[0]);
    TEST_ASSERT_EQ(2.0f, mtxdata[1]);
    TEST_ASSERT_EQ(3.0f, mtxdata[2]);
    mtx_free(&mtx);
    return TEST_SUCCESS;
}

/**
 * `test_mtx_init_vector_array_real_double()` tests creating a double
 * vector in the Matrix Market format.
 */
int test_mtx_init_vector_array_real_double(void)
{
    int err;
    struct mtx mtx;
    int num_comment_lines = 0;
    const char * comment_lines[] = {};
    double data[] = {1.0, 2.0, 3.0};
    int size = sizeof(data) / sizeof(*data);
    err = mtx_init_vector_array_real_double(
        &mtx, num_comment_lines, comment_lines, size, data);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
    TEST_ASSERT_EQ(mtx_vector, mtx.object);
    TEST_ASSERT_EQ(mtx_array, mtx.format);
    TEST_ASSERT_EQ(mtx_real, mtx.field);
    TEST_ASSERT_EQ(mtx_general, mtx.symmetry);
    TEST_ASSERT_EQ(3, mtx.num_rows);
    TEST_ASSERT_EQ(-1, mtx.num_columns);
    TEST_ASSERT_EQ(-1, mtx.num_nonzeros);

    const struct mtx_vector_array_data * vector_array =
        &mtx.storage.vector_array;
    TEST_ASSERT_EQ(mtx_real, vector_array->field);
    TEST_ASSERT_EQ(mtx_double, vector_array->precision);
    TEST_ASSERT_EQ(3, vector_array->size);
    const double * mtxdata = vector_array->data.real_double;
    TEST_ASSERT_EQ(1.0, mtxdata[0]);
    TEST_ASSERT_EQ(2.0, mtxdata[1]);
    TEST_ASSERT_EQ(3.0, mtxdata[2]);
    mtx_free(&mtx);
    return TEST_SUCCESS;
}

/**
 * `test_mtx_init_vector_array_complex_single()' tests creating a
 * complex vector in the Matrix Market format.
 */
int test_mtx_init_vector_array_complex_single(void)
{
    int err;
    struct mtx mtx;
    int num_comment_lines = 0;
    const char * comment_lines[] = {};
    float data[][2] = {{1.0f,2.0}, {3.0f,4.0f}};
    int size = sizeof(data) / sizeof(*data);
    err = mtx_init_vector_array_complex_single(
        &mtx, num_comment_lines, comment_lines, size, data);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
    TEST_ASSERT_EQ(mtx_vector, mtx.object);
    TEST_ASSERT_EQ(mtx_array, mtx.format);
    TEST_ASSERT_EQ(mtx_complex, mtx.field);
    TEST_ASSERT_EQ(mtx_general, mtx.symmetry);
    TEST_ASSERT_EQ(2, mtx.num_rows);
    TEST_ASSERT_EQ(-1, mtx.num_columns);
    TEST_ASSERT_EQ(-1, mtx.num_nonzeros);

    const struct mtx_vector_array_data * vector_array =
        &mtx.storage.vector_array;
    TEST_ASSERT_EQ(mtx_complex, vector_array->field);
    TEST_ASSERT_EQ(mtx_single, vector_array->precision);
    TEST_ASSERT_EQ(2, vector_array->size);
    const float (* mtxdata)[2] = vector_array->data.complex_single;
    TEST_ASSERT_EQ(1.0, mtxdata[0][0]); TEST_ASSERT_EQ(2.0, mtxdata[0][1]);
    TEST_ASSERT_EQ(3.0, mtxdata[1][0]); TEST_ASSERT_EQ(4.0, mtxdata[1][1]);
    mtx_free(&mtx);
    return TEST_SUCCESS;
}

/**
 * `test_mtx_init_vector_array_integer_single()` tests creating an
 * integer vector in the Matrix Market format.
 */
int test_mtx_init_vector_array_integer_single(void)
{
    int err;
    struct mtx mtx;
    int num_comment_lines = 0;
    const char * comment_lines[] = {};
    int32_t data[] = {1, 2, 3};
    int size = sizeof(data) / sizeof(*data);
    err = mtx_init_vector_array_integer_single(
        &mtx, num_comment_lines, comment_lines, size, data);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
    TEST_ASSERT_EQ(mtx_vector, mtx.object);
    TEST_ASSERT_EQ(mtx_array, mtx.format);
    TEST_ASSERT_EQ(mtx_integer, mtx.field);
    TEST_ASSERT_EQ(mtx_general, mtx.symmetry);
    TEST_ASSERT_EQ(3, mtx.num_rows);
    TEST_ASSERT_EQ(-1, mtx.num_columns);
    TEST_ASSERT_EQ(-1, mtx.num_nonzeros);

    const struct mtx_vector_array_data * vector_array =
        &mtx.storage.vector_array;
    TEST_ASSERT_EQ(mtx_integer, vector_array->field);
    TEST_ASSERT_EQ(mtx_single, vector_array->precision);
    TEST_ASSERT_EQ(3, vector_array->size);
    const int32_t * mtxdata = vector_array->data.integer_single;
    TEST_ASSERT_EQ(1, mtxdata[0]);
    TEST_ASSERT_EQ(2, mtxdata[1]);
    TEST_ASSERT_EQ(3, mtxdata[2]);
    mtx_free(&mtx);
    return TEST_SUCCESS;
}

/**
 * `test_mtx_alloc_vector_coordinate_real_single()` tests allocating a
 * single precision, real vector in coordinate format.
 */
int test_mtx_alloc_vector_coordinate_real_single(void)
{
    int err;
    struct mtx mtx;
    int num_comment_lines = 0;
    const char * comment_lines[] = {};
    int num_rows = 4;
    int size = 3;
    err = mtx_alloc_vector_coordinate(
        &mtx, mtx_real, mtx_single,
        num_comment_lines, comment_lines,
        num_rows, size);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
    TEST_ASSERT_EQ(mtx_vector, mtx.object);
    TEST_ASSERT_EQ(mtx_coordinate, mtx.format);
    TEST_ASSERT_EQ(mtx_real, mtx.field);
    TEST_ASSERT_EQ(mtx_general, mtx.symmetry);
    TEST_ASSERT_EQ(4, mtx.num_rows);
    TEST_ASSERT_EQ(-1, mtx.num_columns);
    TEST_ASSERT_EQ(3, mtx.num_nonzeros);

    const struct mtx_vector_coordinate_data * vector_coordinate =
        &mtx.storage.vector_coordinate;
    TEST_ASSERT_EQ(mtx_real, vector_coordinate->field);
    TEST_ASSERT_EQ(mtx_single, vector_coordinate->precision);
    TEST_ASSERT_EQ(mtx_unsorted, vector_coordinate->sorting);
    TEST_ASSERT_EQ(mtx_unassembled, vector_coordinate->assembly);
    TEST_ASSERT_EQ(4, vector_coordinate->num_rows);
    TEST_ASSERT_EQ(-1, vector_coordinate->num_columns);
    TEST_ASSERT_EQ(3, vector_coordinate->size);
    const struct mtx_vector_coordinate_real_single * mtxdata =
        vector_coordinate->data.real_single;
    TEST_ASSERT_NEQ(NULL, mtxdata);
    mtx_free(&mtx);
    return TEST_SUCCESS;
}

/**
 * `test_mtx_init_vector_coordinate_real_single()` tests allocating
 * and initialising a single precision, real vector in coordinate
 * format.
 */
int test_mtx_init_vector_coordinate_real_single(void)
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
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
    TEST_ASSERT_EQ(mtx_vector, mtx.object);
    TEST_ASSERT_EQ(mtx_coordinate, mtx.format);
    TEST_ASSERT_EQ(mtx_real, mtx.field);
    TEST_ASSERT_EQ(mtx_general, mtx.symmetry);
    TEST_ASSERT_EQ(4, mtx.num_rows);
    TEST_ASSERT_EQ(-1, mtx.num_columns);
    TEST_ASSERT_EQ(3, mtx.num_nonzeros);

    const struct mtx_vector_coordinate_data * vector_coordinate =
        &mtx.storage.vector_coordinate;
    TEST_ASSERT_EQ(mtx_real, vector_coordinate->field);
    TEST_ASSERT_EQ(mtx_single, vector_coordinate->precision);
    TEST_ASSERT_EQ(mtx_unsorted, vector_coordinate->sorting);
    TEST_ASSERT_EQ(mtx_unassembled, vector_coordinate->assembly);
    TEST_ASSERT_EQ(4, vector_coordinate->num_rows);
    TEST_ASSERT_EQ(-1, vector_coordinate->num_columns);
    TEST_ASSERT_EQ(3, vector_coordinate->size);
    const struct mtx_vector_coordinate_real_single * mtxdata =
        vector_coordinate->data.real_single;
    TEST_ASSERT_EQ(1, mtxdata[0].i); TEST_ASSERT_EQ(1.0f, mtxdata[0].a);
    TEST_ASSERT_EQ(2, mtxdata[1].i); TEST_ASSERT_EQ(2.0f, mtxdata[1].a);
    TEST_ASSERT_EQ(4, mtxdata[2].i); TEST_ASSERT_EQ(4.0f, mtxdata[2].a);
    mtx_free(&mtx);
    return TEST_SUCCESS;
}

/**
 * `test_mtx_init_vector_coordinate_real_double()' tests creating a
 * double precision, real vector in coordinate format.
 */
int test_mtx_init_vector_coordinate_real_double(void)
{
    int err;
    struct mtx mtx;
    int num_comment_lines = 0;
    const char * comment_lines[] = {};
    int num_rows = 4;
    struct mtx_vector_coordinate_real_double data[] = {
        {1, 1.0}, {2, 2.0}, {4, 4.0}};
    int size = sizeof(data) / sizeof(*data);
    err = mtx_init_vector_coordinate_real_double(
        &mtx, mtx_unsorted, mtx_unassembled,
        num_comment_lines, comment_lines, num_rows, size, data);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
    TEST_ASSERT_EQ(mtx_vector, mtx.object);
    TEST_ASSERT_EQ(mtx_coordinate, mtx.format);
    TEST_ASSERT_EQ(mtx_real, mtx.field);
    TEST_ASSERT_EQ(mtx_general, mtx.symmetry);
    TEST_ASSERT_EQ(4, mtx.num_rows);
    TEST_ASSERT_EQ(-1, mtx.num_columns);
    TEST_ASSERT_EQ(3, mtx.num_nonzeros);
    const struct mtx_vector_coordinate_data * vector_coordinate =
        &mtx.storage.vector_coordinate;
    TEST_ASSERT_EQ(mtx_real, vector_coordinate->field);
    TEST_ASSERT_EQ(mtx_double, vector_coordinate->precision);
    TEST_ASSERT_EQ(mtx_unsorted, vector_coordinate->sorting);
    TEST_ASSERT_EQ(mtx_unassembled, vector_coordinate->assembly);
    TEST_ASSERT_EQ(4, vector_coordinate->num_rows);
    TEST_ASSERT_EQ(-1, vector_coordinate->num_columns);
    TEST_ASSERT_EQ(3, vector_coordinate->size);
    const struct mtx_vector_coordinate_real_double * mtxdata =
        vector_coordinate->data.real_double;
    TEST_ASSERT_EQ(1, mtxdata[0].i); TEST_ASSERT_EQ(1.0, mtxdata[0].a);
    TEST_ASSERT_EQ(2, mtxdata[1].i); TEST_ASSERT_EQ(2.0, mtxdata[1].a);
    TEST_ASSERT_EQ(4, mtxdata[2].i); TEST_ASSERT_EQ(4.0, mtxdata[2].a);
    mtx_free(&mtx);
    return TEST_SUCCESS;
}

/**
 * `test_mtx_init_vector_coordinate_complex_single()` tests creating a
 * single precision, complex vector in coordinate format.
 */
int test_mtx_init_vector_coordinate_complex_single(void)
{
    int err;
    struct mtx mtx;
    int num_comment_lines = 0;
    const char * comment_lines[] = {};
    int num_rows = 4;
    struct mtx_vector_coordinate_complex_single data[] = {
        {1, 1.0f, 6.0f}, {2, 2.0f, 7.0f}, {4, 4.0f, 8.0f}};
    int size = sizeof(data) / sizeof(*data);
    err = mtx_init_vector_coordinate_complex_single(
        &mtx, mtx_unsorted, mtx_unassembled,
        num_comment_lines, comment_lines, num_rows, size, data);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
    TEST_ASSERT_EQ(mtx_vector, mtx.object);
    TEST_ASSERT_EQ(mtx_coordinate, mtx.format);
    TEST_ASSERT_EQ(mtx_complex, mtx.field);
    TEST_ASSERT_EQ(mtx_general, mtx.symmetry);
    TEST_ASSERT_EQ(4, mtx.num_rows);
    TEST_ASSERT_EQ(-1, mtx.num_columns);
    TEST_ASSERT_EQ(3, mtx.num_nonzeros);

    const struct mtx_vector_coordinate_data * vector_coordinate =
        &mtx.storage.vector_coordinate;
    TEST_ASSERT_EQ(mtx_complex, vector_coordinate->field);
    TEST_ASSERT_EQ(mtx_single, vector_coordinate->precision);
    TEST_ASSERT_EQ(mtx_unsorted, vector_coordinate->sorting);
    TEST_ASSERT_EQ(mtx_unassembled, vector_coordinate->assembly);
    TEST_ASSERT_EQ(4, vector_coordinate->num_rows);
    TEST_ASSERT_EQ(-1, vector_coordinate->num_columns);
    TEST_ASSERT_EQ(3, vector_coordinate->size);
    const struct mtx_vector_coordinate_complex_single * mtxdata =
        vector_coordinate->data.complex_single;
    TEST_ASSERT_EQ(1, mtxdata[0].i); TEST_ASSERT_EQ(1.0f, mtxdata[0].a[0]); TEST_ASSERT_EQ(6.0f, mtxdata[0].a[1]);
    TEST_ASSERT_EQ(2, mtxdata[1].i); TEST_ASSERT_EQ(2.0f, mtxdata[1].a[0]); TEST_ASSERT_EQ(7.0f, mtxdata[1].a[1]);
    TEST_ASSERT_EQ(4, mtxdata[2].i); TEST_ASSERT_EQ(4.0f, mtxdata[2].a[0]); TEST_ASSERT_EQ(8.0f, mtxdata[2].a[1]);
    mtx_free(&mtx);
    return TEST_SUCCESS;
}

/**
 * `test_mtx_init_vector_coordinate_integer_single()` tests creating a
 * single precision, integer vector in coordinate format.
 */
int test_mtx_init_vector_coordinate_integer_single(void)
{
    int err;
    struct mtx mtx;
    int num_comment_lines = 0;
    const char * comment_lines[] = {};
    int num_rows = 4;
    struct mtx_vector_coordinate_integer_single data[] = {
        {1, 3}, {2, 2}, {4, 4}};
    int size = sizeof(data) / sizeof(*data);
    err = mtx_init_vector_coordinate_integer_single(
        &mtx, mtx_unsorted, mtx_unassembled,
        num_comment_lines, comment_lines, num_rows, size, data);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
    TEST_ASSERT_EQ(mtx_vector, mtx.object);
    TEST_ASSERT_EQ(mtx_coordinate, mtx.format);
    TEST_ASSERT_EQ(mtx_integer, mtx.field);
    TEST_ASSERT_EQ(mtx_general, mtx.symmetry);
    TEST_ASSERT_EQ(4, mtx.num_rows);
    TEST_ASSERT_EQ(-1, mtx.num_columns);
    TEST_ASSERT_EQ(3, mtx.num_nonzeros);

    const struct mtx_vector_coordinate_data * vector_coordinate =
        &mtx.storage.vector_coordinate;
    TEST_ASSERT_EQ(mtx_integer, vector_coordinate->field);
    TEST_ASSERT_EQ(mtx_single, vector_coordinate->precision);
    TEST_ASSERT_EQ(mtx_unsorted, vector_coordinate->sorting);
    TEST_ASSERT_EQ(mtx_unassembled, vector_coordinate->assembly);
    TEST_ASSERT_EQ(4, vector_coordinate->num_rows);
    TEST_ASSERT_EQ(-1, vector_coordinate->num_columns);
    TEST_ASSERT_EQ(3, vector_coordinate->size);
    const struct mtx_vector_coordinate_integer_single * mtxdata =
        vector_coordinate->data.integer_single;
    TEST_ASSERT_EQ(1, mtxdata[0].i); TEST_ASSERT_EQ(3, mtxdata[0].a);
    TEST_ASSERT_EQ(2, mtxdata[1].i); TEST_ASSERT_EQ(2, mtxdata[1].a);
    TEST_ASSERT_EQ(4, mtxdata[2].i); TEST_ASSERT_EQ(4, mtxdata[2].a);
    mtx_free(&mtx);
    return TEST_SUCCESS;
}

/**
 * `test_mtx_init_vector_coordinate_pattern()` tests creating a
 * pattern vector in coordinate format.
 */
int test_mtx_init_vector_coordinate_pattern(void)
{
    int err;
    struct mtx mtx;
    int num_comment_lines = 0;
    const char * comment_lines[] = {};
    int num_rows = 4;
    struct mtx_vector_coordinate_pattern data[] = {{1}, {2}, {4}};
    int size = sizeof(data) / sizeof(*data);
    err = mtx_init_vector_coordinate_pattern(
        &mtx, mtx_unsorted, mtx_unassembled,
        num_comment_lines, comment_lines, num_rows, size, data);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
    TEST_ASSERT_EQ(mtx_vector, mtx.object);
    TEST_ASSERT_EQ(mtx_coordinate, mtx.format);
    TEST_ASSERT_EQ(mtx_pattern, mtx.field);
    TEST_ASSERT_EQ(mtx_general, mtx.symmetry);
    TEST_ASSERT_EQ(4, mtx.num_rows);
    TEST_ASSERT_EQ(-1, mtx.num_columns);
    TEST_ASSERT_EQ(3, mtx.num_nonzeros);

    const struct mtx_vector_coordinate_data * vector_coordinate =
        &mtx.storage.vector_coordinate;
    TEST_ASSERT_EQ(mtx_pattern, vector_coordinate->field);
    TEST_ASSERT_EQ(mtx_single, vector_coordinate->precision);
    TEST_ASSERT_EQ(mtx_unsorted, vector_coordinate->sorting);
    TEST_ASSERT_EQ(mtx_unassembled, vector_coordinate->assembly);
    TEST_ASSERT_EQ(4, vector_coordinate->num_rows);
    TEST_ASSERT_EQ(-1, vector_coordinate->num_columns);
    TEST_ASSERT_EQ(3, vector_coordinate->size);
    const struct mtx_vector_coordinate_pattern * mtxdata =
        vector_coordinate->data.pattern;
    TEST_ASSERT_EQ(1, mtxdata[0].i);
    TEST_ASSERT_EQ(2, mtxdata[1].i);
    TEST_ASSERT_EQ(4, mtxdata[2].i);
    mtx_free(&mtx);
    return TEST_SUCCESS;
}

/**
 * `main()' entry point and test driver.
 */
int main(int argc, char * argv[])
{
    TEST_SUITE_BEGIN("Running tests for vectors in Matrix Market format.\n");
    TEST_RUN(test_mtx_alloc_vector_array_real_single);
    TEST_RUN(test_mtx_init_vector_array_real_single);
    TEST_RUN(test_mtx_init_vector_array_real_double);
    TEST_RUN(test_mtx_init_vector_array_complex_single);
    TEST_RUN(test_mtx_init_vector_array_integer_single);
    TEST_RUN(test_mtx_init_vector_coordinate_real_single);
    TEST_RUN(test_mtx_init_vector_coordinate_real_double);
    TEST_RUN(test_mtx_init_vector_coordinate_complex_single);
    TEST_RUN(test_mtx_init_vector_coordinate_integer_single);
    TEST_RUN(test_mtx_init_vector_coordinate_pattern);
    TEST_SUITE_END();
    return (TEST_SUITE_STATUS == TEST_SUCCESS) ?
        EXIT_SUCCESS : EXIT_FAILURE;
}
