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
 * Unit tests for extracting submatrices.
 */

#include "test.h"

#include <libmtx/error.h>
#include <libmtx/matrix/coordinate.h>
#include <libmtx/mtx/header.h>
#include <libmtx/mtx/mtx.h>
#include <libmtx/mtx/submatrix.h>
#include <libmtx/util/index_set.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * `test_mtx_matrix_submatrix_coordinate_real_general()` tests
 * transposing non-symmetric sparse matrices with real,
 * single-precision coefficients in the Matrix Market format.
 */
int test_mtx_matrix_submatrix_coordinate_real_general(void)
{
    int err;

    /* Create a sparse matrix. */
    struct mtx mtx;
    int num_comment_lines = 0;
    const char * comment_lines[] = {};
    int num_rows = 4;
    int num_columns = 4;
    int64_t size = 6;
    const struct mtx_matrix_coordinate_real_single data[] = {
        {1,1,1.0f}, {1,4,2.0f},
        {2,2,3.0f},
        {3,3,4.0f},
        {4,1,5.0f}, {4,4,6.0f}};
    err = mtx_init_matrix_coordinate_real_single(
        &mtx, mtx_general,
        mtx_nontriangular, mtx_unsorted, mtx_unassembled,
        num_comment_lines, comment_lines,
        num_rows, num_columns, size, data);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));

    /* Extract a submatrix. */
    struct mtx_index_set rows;
    mtx_index_set_init_interval(&rows, 1, 3);
    struct mtx_index_set columns;
    mtx_index_set_init_interval(&columns, 1, 5);
    struct mtx submtx;
    err = mtx_matrix_submatrix(&submtx, &mtx, &rows, &columns);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
    TEST_ASSERT_EQ(mtx_matrix, submtx.object);
    TEST_ASSERT_EQ(mtx_coordinate, submtx.format);
    TEST_ASSERT_EQ(mtx_real, submtx.field);
    TEST_ASSERT_EQ(mtx_general, submtx.symmetry);
    TEST_ASSERT_EQ(4, submtx.num_rows);
    TEST_ASSERT_EQ(4, submtx.num_columns);
    TEST_ASSERT_EQ(3, submtx.num_nonzeros);

    const struct mtx_matrix_coordinate_data * matrix_coordinate =
        &submtx.storage.matrix_coordinate;
    TEST_ASSERT_EQ(mtx_real, matrix_coordinate->field);
    TEST_ASSERT_EQ(mtx_single, matrix_coordinate->precision);
    TEST_ASSERT_EQ(mtx_general, matrix_coordinate->symmetry);
    TEST_ASSERT_EQ(mtx_nontriangular, matrix_coordinate->triangle);
    TEST_ASSERT_EQ(mtx_unsorted, matrix_coordinate->sorting);
    TEST_ASSERT_EQ(mtx_unassembled, matrix_coordinate->assembly);
    TEST_ASSERT_EQ(4, matrix_coordinate->num_rows);
    TEST_ASSERT_EQ(4, matrix_coordinate->num_columns);
    TEST_ASSERT_EQ(3, matrix_coordinate->size);
    const struct mtx_matrix_coordinate_real_single * mtxdata =
        matrix_coordinate->data.real_single;
    TEST_ASSERT_EQ(   1, mtxdata[0].i); TEST_ASSERT_EQ(   1, mtxdata[0].j);
    TEST_ASSERT_EQ(1.0f, mtxdata[0].a);
    TEST_ASSERT_EQ(   1, mtxdata[1].i); TEST_ASSERT_EQ(   4, mtxdata[1].j);
    TEST_ASSERT_EQ(2.0f, mtxdata[1].a);
    TEST_ASSERT_EQ(   2, mtxdata[2].i); TEST_ASSERT_EQ(   2, mtxdata[2].j);
    TEST_ASSERT_EQ(3.0f, mtxdata[2].a);
    mtx_free(&submtx);
    mtx_free(&mtx);
    return TEST_SUCCESS;
}

/**
 * `main()' entry point and test driver.
 */
int main(int argc, char * argv[])
{
    TEST_SUITE_BEGIN(
        "Running tests for extracting submatrices.\n");
    TEST_RUN(test_mtx_matrix_submatrix_coordinate_real_general);
    TEST_SUITE_END();
    return (TEST_SUITE_STATUS == TEST_SUCCESS) ?
        EXIT_SUCCESS : EXIT_FAILURE;
}
