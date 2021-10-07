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
 * Last modified: 2021-09-20
 *
 * Unit tests for matrices.
 */

#include "test.h"

#include <libmtx/error.h>
#include <libmtx/matrix/matrix.h>
#include <libmtx/mtxfile/mtxfile.h>

#include <errno.h>
#include <unistd.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * `test_mtxmatrix_from_mtxfile()' tests converting Matrix Market
 *  files to matrices.
 */
int test_mtxmatrix_from_mtxfile(void)
{
    int err;

    /*
     * Array formats
     */

    {
        int num_rows = 3;
        int num_columns = 3;
        const float mtxdata[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
        int64_t num_nonzeros = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile mtxfile;
        err = mtxfile_init_matrix_array_real_single(
            &mtxfile, mtxfile_general, num_rows, num_columns, mtxdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        struct mtxmatrix x;
        err = mtxmatrix_from_mtxfile(&x, &mtxfile, mtxmatrix_auto);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(mtxmatrix_array, x.type);
        const struct mtxmatrix_array * x_ = &x.storage.array;
        TEST_ASSERT_EQ(mtx_field_real, x_->field);
        TEST_ASSERT_EQ(mtx_single, x_->precision);
        TEST_ASSERT_EQ(3, x_->num_rows);
        TEST_ASSERT_EQ(3, x_->num_columns);
        TEST_ASSERT_EQ(9, x_->size);
        TEST_ASSERT_EQ(x_->data.real_single[0], 1.0f);
        TEST_ASSERT_EQ(x_->data.real_single[1], 2.0f);
        TEST_ASSERT_EQ(x_->data.real_single[2], 3.0f);
        TEST_ASSERT_EQ(x_->data.real_single[3], 4.0f);
        TEST_ASSERT_EQ(x_->data.real_single[4], 5.0f);
        TEST_ASSERT_EQ(x_->data.real_single[5], 6.0f);
        TEST_ASSERT_EQ(x_->data.real_single[6], 7.0f);
        TEST_ASSERT_EQ(x_->data.real_single[7], 8.0f);
        TEST_ASSERT_EQ(x_->data.real_single[8], 9.0f);
        mtxmatrix_free(&x);
        mtxfile_free(&mtxfile);
    }

    {
        int num_rows = 3;
        int num_columns = 3;
        const double mtxdata[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
        int64_t num_nonzeros = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile mtxfile;
        err = mtxfile_init_matrix_array_real_double(
            &mtxfile, mtxfile_general, num_rows, num_columns, mtxdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        struct mtxmatrix x;
        err = mtxmatrix_from_mtxfile(&x, &mtxfile, mtxmatrix_auto);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(mtxmatrix_array, x.type);
        const struct mtxmatrix_array * x_ = &x.storage.array;
        TEST_ASSERT_EQ(mtx_field_real, x_->field);
        TEST_ASSERT_EQ(mtx_double, x_->precision);
        TEST_ASSERT_EQ(3, x_->num_rows);
        TEST_ASSERT_EQ(3, x_->num_columns);
        TEST_ASSERT_EQ(9, x_->size);
        TEST_ASSERT_EQ(x_->data.real_double[0], 1.0);
        TEST_ASSERT_EQ(x_->data.real_double[1], 2.0);
        TEST_ASSERT_EQ(x_->data.real_double[2], 3.0);
        TEST_ASSERT_EQ(x_->data.real_double[3], 4.0);
        TEST_ASSERT_EQ(x_->data.real_double[4], 5.0);
        TEST_ASSERT_EQ(x_->data.real_double[5], 6.0);
        TEST_ASSERT_EQ(x_->data.real_double[6], 7.0);
        TEST_ASSERT_EQ(x_->data.real_double[7], 8.0);
        TEST_ASSERT_EQ(x_->data.real_double[8], 9.0);
        mtxmatrix_free(&x);
        mtxfile_free(&mtxfile);
    }

    {
        int num_rows = 2;
        int num_columns = 2;
        const float mtxdata[][2] = {{1.0f,2.0f}, {3.0f,4.0f}, {5.0f,6.0f}, {7.0f,8.0f}};
        int64_t num_nonzeros = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile mtxfile;
        err = mtxfile_init_matrix_array_complex_single(
            &mtxfile, mtxfile_general, num_rows, num_columns, mtxdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        struct mtxmatrix x;
        err = mtxmatrix_from_mtxfile(&x, &mtxfile, mtxmatrix_auto);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(mtxmatrix_array, x.type);
        const struct mtxmatrix_array * x_ = &x.storage.array;
        TEST_ASSERT_EQ(mtx_field_complex, x_->field);
        TEST_ASSERT_EQ(mtx_single, x_->precision);
        TEST_ASSERT_EQ(2, x_->num_rows);
        TEST_ASSERT_EQ(2, x_->num_columns);
        TEST_ASSERT_EQ(4, x_->size);
        TEST_ASSERT_EQ(x_->data.complex_single[0][0], 1.0f);
        TEST_ASSERT_EQ(x_->data.complex_single[0][1], 2.0f);
        TEST_ASSERT_EQ(x_->data.complex_single[1][0], 3.0f);
        TEST_ASSERT_EQ(x_->data.complex_single[1][1], 4.0f);
        TEST_ASSERT_EQ(x_->data.complex_single[2][0], 5.0f);
        TEST_ASSERT_EQ(x_->data.complex_single[2][1], 6.0f);
        TEST_ASSERT_EQ(x_->data.complex_single[3][0], 7.0f);
        TEST_ASSERT_EQ(x_->data.complex_single[3][1], 8.0f);
        mtxmatrix_free(&x);
        mtxfile_free(&mtxfile);
    }

    {
        int num_rows = 2;
        int num_columns = 2;
        const double mtxdata[][2] = {{1.0,2.0}, {3.0,4.0}, {5.0,6.0}, {7.0,8.0}};
        int64_t num_nonzeros = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile mtxfile;
        err = mtxfile_init_matrix_array_complex_double(
            &mtxfile, mtxfile_general, num_rows, num_columns, mtxdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        struct mtxmatrix x;
        err = mtxmatrix_from_mtxfile(&x, &mtxfile, mtxmatrix_auto);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(mtxmatrix_array, x.type);
        const struct mtxmatrix_array * x_ = &x.storage.array;
        TEST_ASSERT_EQ(mtx_field_complex, x_->field);
        TEST_ASSERT_EQ(mtx_double, x_->precision);
        TEST_ASSERT_EQ(2, x_->num_rows);
        TEST_ASSERT_EQ(2, x_->num_columns);
        TEST_ASSERT_EQ(4, x_->size);
        TEST_ASSERT_EQ(x_->data.complex_double[0][0], 1.0);
        TEST_ASSERT_EQ(x_->data.complex_double[0][1], 2.0);
        TEST_ASSERT_EQ(x_->data.complex_double[1][0], 3.0);
        TEST_ASSERT_EQ(x_->data.complex_double[1][1], 4.0);
        TEST_ASSERT_EQ(x_->data.complex_double[2][0], 5.0);
        TEST_ASSERT_EQ(x_->data.complex_double[2][1], 6.0);
        TEST_ASSERT_EQ(x_->data.complex_double[3][0], 7.0);
        TEST_ASSERT_EQ(x_->data.complex_double[3][1], 8.0);
        mtxmatrix_free(&x);
        mtxfile_free(&mtxfile);
    }

    {
        int num_rows = 3;
        int num_columns = 3;
        const int32_t mtxdata[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
        int64_t num_nonzeros = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile mtxfile;
        err = mtxfile_init_matrix_array_integer_single(
            &mtxfile, mtxfile_general, num_rows, num_columns, mtxdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        struct mtxmatrix x;
        err = mtxmatrix_from_mtxfile(&x, &mtxfile, mtxmatrix_auto);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(mtxmatrix_array, x.type);
        const struct mtxmatrix_array * x_ = &x.storage.array;
        TEST_ASSERT_EQ(mtx_field_integer, x_->field);
        TEST_ASSERT_EQ(mtx_single, x_->precision);
        TEST_ASSERT_EQ(3, x_->num_rows);
        TEST_ASSERT_EQ(3, x_->num_columns);
        TEST_ASSERT_EQ(9, x_->size);
        TEST_ASSERT_EQ(x_->data.integer_single[0], 1);
        TEST_ASSERT_EQ(x_->data.integer_single[1], 2);
        TEST_ASSERT_EQ(x_->data.integer_single[2], 3);
        TEST_ASSERT_EQ(x_->data.integer_single[3], 4);
        TEST_ASSERT_EQ(x_->data.integer_single[4], 5);
        TEST_ASSERT_EQ(x_->data.integer_single[5], 6);
        TEST_ASSERT_EQ(x_->data.integer_single[6], 7);
        TEST_ASSERT_EQ(x_->data.integer_single[7], 8);
        TEST_ASSERT_EQ(x_->data.integer_single[8], 9);
        mtxmatrix_free(&x);
        mtxfile_free(&mtxfile);
    }

    {
        int num_rows = 3;
        int num_columns = 3;
        const int64_t mtxdata[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
        int64_t num_nonzeros = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile mtxfile;
        err = mtxfile_init_matrix_array_integer_double(
            &mtxfile, mtxfile_general, num_rows, num_columns, mtxdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        struct mtxmatrix x;
        err = mtxmatrix_from_mtxfile(&x, &mtxfile, mtxmatrix_auto);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(mtxmatrix_array, x.type);
        const struct mtxmatrix_array * x_ = &x.storage.array;
        TEST_ASSERT_EQ(mtx_field_integer, x_->field);
        TEST_ASSERT_EQ(mtx_double, x_->precision);
        TEST_ASSERT_EQ(3, x_->num_rows);
        TEST_ASSERT_EQ(3, x_->num_columns);
        TEST_ASSERT_EQ(9, x_->size);
        TEST_ASSERT_EQ(x_->data.integer_double[0], 1);
        TEST_ASSERT_EQ(x_->data.integer_double[1], 2);
        TEST_ASSERT_EQ(x_->data.integer_double[2], 3);
        TEST_ASSERT_EQ(x_->data.integer_double[3], 4);
        TEST_ASSERT_EQ(x_->data.integer_double[4], 5);
        TEST_ASSERT_EQ(x_->data.integer_double[5], 6);
        TEST_ASSERT_EQ(x_->data.integer_double[6], 7);
        TEST_ASSERT_EQ(x_->data.integer_double[7], 8);
        TEST_ASSERT_EQ(x_->data.integer_double[8], 9);
        mtxmatrix_free(&x);
        mtxfile_free(&mtxfile);
    }

    /*
     * Coordinate formats
     */

    {
        int num_rows = 3;
        int num_columns = 3;
        struct mtxfile_matrix_coordinate_real_single mtxdata[] = {
            {1,1,1.0f}, {1,3,3.0f}, {2,1,4.0f}, {3,3,9.0f}};
        int64_t num_nonzeros = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile mtxfile;
        err = mtxfile_init_matrix_coordinate_real_single(
            &mtxfile, mtxfile_general, num_rows, num_columns, num_nonzeros, mtxdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        struct mtxmatrix x;
        err = mtxmatrix_from_mtxfile(&x, &mtxfile, mtxmatrix_auto);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(mtxmatrix_coordinate, x.type);
        const struct mtxmatrix_coordinate * x_ = &x.storage.coordinate;
        TEST_ASSERT_EQ(mtx_field_real, x_->field);
        TEST_ASSERT_EQ(mtx_single, x_->precision);
        TEST_ASSERT_EQ(3, x_->num_rows);
        TEST_ASSERT_EQ(3, x_->num_columns);
        TEST_ASSERT_EQ(4, x_->num_nonzeros);
        TEST_ASSERT_EQ(x_->rowidx[0], 1); TEST_ASSERT_EQ(x_->colidx[0], 1);
        TEST_ASSERT_EQ(x_->rowidx[1], 1); TEST_ASSERT_EQ(x_->colidx[1], 3);
        TEST_ASSERT_EQ(x_->rowidx[2], 2); TEST_ASSERT_EQ(x_->colidx[2], 1);
        TEST_ASSERT_EQ(x_->rowidx[3], 3); TEST_ASSERT_EQ(x_->colidx[3], 3);
        TEST_ASSERT_EQ(x_->data.real_single[0], 1.0f);
        TEST_ASSERT_EQ(x_->data.real_single[1], 3.0f);
        TEST_ASSERT_EQ(x_->data.real_single[2], 4.0f);
        TEST_ASSERT_EQ(x_->data.real_single[3], 9.0f);
        mtxmatrix_free(&x);
        mtxfile_free(&mtxfile);
    }

    {
        int num_rows = 3;
        int num_columns = 3;
        struct mtxfile_matrix_coordinate_real_double mtxdata[] = {
            {1,1,1.0}, {1,3,3.0}, {2,1,4.0}, {3,3,9.0}};
        int64_t num_nonzeros = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile mtxfile;
        err = mtxfile_init_matrix_coordinate_real_double(
            &mtxfile, mtxfile_general, num_rows, num_columns, num_nonzeros, mtxdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        struct mtxmatrix x;
        err = mtxmatrix_from_mtxfile(&x, &mtxfile, mtxmatrix_auto);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(mtxmatrix_coordinate, x.type);
        const struct mtxmatrix_coordinate * x_ = &x.storage.coordinate;
        TEST_ASSERT_EQ(mtx_field_real, x_->field);
        TEST_ASSERT_EQ(mtx_double, x_->precision);
        TEST_ASSERT_EQ(3, x_->num_rows);
        TEST_ASSERT_EQ(3, x_->num_columns);
        TEST_ASSERT_EQ(4, x_->num_nonzeros);
        TEST_ASSERT_EQ(x_->rowidx[0], 1); TEST_ASSERT_EQ(x_->colidx[0], 1);
        TEST_ASSERT_EQ(x_->rowidx[1], 1); TEST_ASSERT_EQ(x_->colidx[1], 3);
        TEST_ASSERT_EQ(x_->rowidx[2], 2); TEST_ASSERT_EQ(x_->colidx[2], 1);
        TEST_ASSERT_EQ(x_->rowidx[3], 3); TEST_ASSERT_EQ(x_->colidx[3], 3);
        TEST_ASSERT_EQ(x_->data.real_double[0], 1.0);
        TEST_ASSERT_EQ(x_->data.real_double[1], 3.0);
        TEST_ASSERT_EQ(x_->data.real_double[2], 4.0);
        TEST_ASSERT_EQ(x_->data.real_double[3], 9.0);
        mtxmatrix_free(&x);
        mtxfile_free(&mtxfile);
    }

    {
        int num_rows = 3;
        int num_columns = 3;
        struct mtxfile_matrix_coordinate_complex_single mtxdata[] = {
            {1,1,{1.0f,-1.0f}}, {1,3,{3.0f,-3.0f}},
            {2,1,{4.0f,-4.0f}}, {3,3,{9.0f,-9.0f}}};
        int64_t num_nonzeros = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile mtxfile;
        err = mtxfile_init_matrix_coordinate_complex_single(
            &mtxfile, mtxfile_general, num_rows, num_columns, num_nonzeros, mtxdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        struct mtxmatrix x;
        err = mtxmatrix_from_mtxfile(&x, &mtxfile, mtxmatrix_auto);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(mtxmatrix_coordinate, x.type);
        const struct mtxmatrix_coordinate * x_ = &x.storage.coordinate;
        TEST_ASSERT_EQ(mtx_field_complex, x_->field);
        TEST_ASSERT_EQ(mtx_single, x_->precision);
        TEST_ASSERT_EQ(3, x_->num_rows);
        TEST_ASSERT_EQ(3, x_->num_columns);
        TEST_ASSERT_EQ(4, x_->num_nonzeros);
        TEST_ASSERT_EQ(x_->rowidx[0], 1); TEST_ASSERT_EQ(x_->colidx[0], 1);
        TEST_ASSERT_EQ(x_->rowidx[1], 1); TEST_ASSERT_EQ(x_->colidx[1], 3);
        TEST_ASSERT_EQ(x_->rowidx[2], 2); TEST_ASSERT_EQ(x_->colidx[2], 1);
        TEST_ASSERT_EQ(x_->rowidx[3], 3); TEST_ASSERT_EQ(x_->colidx[3], 3);
        TEST_ASSERT_EQ(x_->data.complex_single[0][0], 1.0f);
        TEST_ASSERT_EQ(x_->data.complex_single[0][1],-1.0f);
        TEST_ASSERT_EQ(x_->data.complex_single[1][0], 3.0f);
        TEST_ASSERT_EQ(x_->data.complex_single[1][1],-3.0f);
        TEST_ASSERT_EQ(x_->data.complex_single[2][0], 4.0f);
        TEST_ASSERT_EQ(x_->data.complex_single[2][1],-4.0f);
        TEST_ASSERT_EQ(x_->data.complex_single[3][0], 9.0f);
        TEST_ASSERT_EQ(x_->data.complex_single[3][1],-9.0f);
        mtxmatrix_free(&x);
        mtxfile_free(&mtxfile);
    }

    {
        int num_rows = 3;
        int num_columns = 3;
        struct mtxfile_matrix_coordinate_complex_double mtxdata[] = {
            {1,1,{1.0,-1.0}}, {1,3,{3.0,-3.0}},
            {2,1,{4.0,-4.0}}, {3,3,{9.0,-9.0}}};
        int64_t num_nonzeros = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile mtxfile;
        err = mtxfile_init_matrix_coordinate_complex_double(
            &mtxfile, mtxfile_general, num_rows, num_columns, num_nonzeros, mtxdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        struct mtxmatrix x;
        err = mtxmatrix_from_mtxfile(&x, &mtxfile, mtxmatrix_auto);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(mtxmatrix_coordinate, x.type);
        const struct mtxmatrix_coordinate * x_ = &x.storage.coordinate;
        TEST_ASSERT_EQ(mtx_field_complex, x_->field);
        TEST_ASSERT_EQ(mtx_double, x_->precision);
        TEST_ASSERT_EQ(3, x_->num_rows);
        TEST_ASSERT_EQ(3, x_->num_columns);
        TEST_ASSERT_EQ(4, x_->num_nonzeros);
        TEST_ASSERT_EQ(x_->rowidx[0], 1); TEST_ASSERT_EQ(x_->colidx[0], 1);
        TEST_ASSERT_EQ(x_->rowidx[1], 1); TEST_ASSERT_EQ(x_->colidx[1], 3);
        TEST_ASSERT_EQ(x_->rowidx[2], 2); TEST_ASSERT_EQ(x_->colidx[2], 1);
        TEST_ASSERT_EQ(x_->rowidx[3], 3); TEST_ASSERT_EQ(x_->colidx[3], 3);
        TEST_ASSERT_EQ(x_->data.complex_double[0][0], 1.0);
        TEST_ASSERT_EQ(x_->data.complex_double[0][1],-1.0);
        TEST_ASSERT_EQ(x_->data.complex_double[1][0], 3.0);
        TEST_ASSERT_EQ(x_->data.complex_double[1][1],-3.0);
        TEST_ASSERT_EQ(x_->data.complex_double[2][0], 4.0);
        TEST_ASSERT_EQ(x_->data.complex_double[2][1],-4.0);
        TEST_ASSERT_EQ(x_->data.complex_double[3][0], 9.0);
        TEST_ASSERT_EQ(x_->data.complex_double[3][1],-9.0);
        mtxmatrix_free(&x);
        mtxfile_free(&mtxfile);
    }

    {
        int num_rows = 3;
        int num_columns = 3;
        struct mtxfile_matrix_coordinate_integer_single mtxdata[] = {
            {1,1,1}, {1,3,3}, {2,1,4}, {3,3,9}};
        int64_t num_nonzeros = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile mtxfile;
        err = mtxfile_init_matrix_coordinate_integer_single(
            &mtxfile, mtxfile_general, num_rows, num_columns, num_nonzeros, mtxdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        struct mtxmatrix x;
        err = mtxmatrix_from_mtxfile(&x, &mtxfile, mtxmatrix_auto);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(mtxmatrix_coordinate, x.type);
        const struct mtxmatrix_coordinate * x_ = &x.storage.coordinate;
        TEST_ASSERT_EQ(mtx_field_integer, x_->field);
        TEST_ASSERT_EQ(mtx_single, x_->precision);
        TEST_ASSERT_EQ(3, x_->num_rows);
        TEST_ASSERT_EQ(3, x_->num_columns);
        TEST_ASSERT_EQ(4, x_->num_nonzeros);
        TEST_ASSERT_EQ(x_->rowidx[0], 1); TEST_ASSERT_EQ(x_->colidx[0], 1);
        TEST_ASSERT_EQ(x_->rowidx[1], 1); TEST_ASSERT_EQ(x_->colidx[1], 3);
        TEST_ASSERT_EQ(x_->rowidx[2], 2); TEST_ASSERT_EQ(x_->colidx[2], 1);
        TEST_ASSERT_EQ(x_->rowidx[3], 3); TEST_ASSERT_EQ(x_->colidx[3], 3);
        TEST_ASSERT_EQ(x_->data.integer_single[0], 1);
        TEST_ASSERT_EQ(x_->data.integer_single[1], 3);
        TEST_ASSERT_EQ(x_->data.integer_single[2], 4);
        TEST_ASSERT_EQ(x_->data.integer_single[3], 9);
        mtxmatrix_free(&x);
        mtxfile_free(&mtxfile);
    }

    {
        int num_rows = 3;
        int num_columns = 3;
        struct mtxfile_matrix_coordinate_integer_double mtxdata[] = {
            {1,1,1}, {1,3,3}, {2,1,4}, {3,3,9}};
        int64_t num_nonzeros = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile mtxfile;
        err = mtxfile_init_matrix_coordinate_integer_double(
            &mtxfile, mtxfile_general, num_rows, num_columns, num_nonzeros, mtxdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        struct mtxmatrix x;
        err = mtxmatrix_from_mtxfile(&x, &mtxfile, mtxmatrix_auto);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(mtxmatrix_coordinate, x.type);
        const struct mtxmatrix_coordinate * x_ = &x.storage.coordinate;
        TEST_ASSERT_EQ(mtx_field_integer, x_->field);
        TEST_ASSERT_EQ(mtx_double, x_->precision);
        TEST_ASSERT_EQ(3, x_->num_rows);
        TEST_ASSERT_EQ(3, x_->num_columns);
        TEST_ASSERT_EQ(4, x_->num_nonzeros);
        TEST_ASSERT_EQ(x_->rowidx[0], 1); TEST_ASSERT_EQ(x_->colidx[0], 1);
        TEST_ASSERT_EQ(x_->rowidx[1], 1); TEST_ASSERT_EQ(x_->colidx[1], 3);
        TEST_ASSERT_EQ(x_->rowidx[2], 2); TEST_ASSERT_EQ(x_->colidx[2], 1);
        TEST_ASSERT_EQ(x_->rowidx[3], 3); TEST_ASSERT_EQ(x_->colidx[3], 3);
        TEST_ASSERT_EQ(x_->data.integer_double[0], 1);
        TEST_ASSERT_EQ(x_->data.integer_double[1], 3);
        TEST_ASSERT_EQ(x_->data.integer_double[2], 4);
        TEST_ASSERT_EQ(x_->data.integer_double[3], 9);
        mtxmatrix_free(&x);
        mtxfile_free(&mtxfile);
    }

    {
        int num_rows = 3;
        int num_columns = 3;
        struct mtxfile_matrix_coordinate_pattern mtxdata[] = {
            {1,1}, {1,3}, {2,1}, {3,3}};
        int64_t num_nonzeros = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile mtxfile;
        err = mtxfile_init_matrix_coordinate_pattern(
            &mtxfile, mtxfile_general, num_rows, num_columns, num_nonzeros, mtxdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        struct mtxmatrix x;
        err = mtxmatrix_from_mtxfile(&x, &mtxfile, mtxmatrix_auto);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(mtxmatrix_coordinate, x.type);
        const struct mtxmatrix_coordinate * x_ = &x.storage.coordinate;
        TEST_ASSERT_EQ(mtx_field_pattern, x_->field);
        TEST_ASSERT_EQ(3, x_->num_rows);
        TEST_ASSERT_EQ(3, x_->num_columns);
        TEST_ASSERT_EQ(4, x_->num_nonzeros);
        TEST_ASSERT_EQ(x_->rowidx[0], 1); TEST_ASSERT_EQ(x_->colidx[0], 1);
        TEST_ASSERT_EQ(x_->rowidx[1], 1); TEST_ASSERT_EQ(x_->colidx[1], 3);
        TEST_ASSERT_EQ(x_->rowidx[2], 2); TEST_ASSERT_EQ(x_->colidx[2], 1);
        TEST_ASSERT_EQ(x_->rowidx[3], 3); TEST_ASSERT_EQ(x_->colidx[3], 3);
        mtxmatrix_free(&x);
        mtxfile_free(&mtxfile);
    }
    return TEST_SUCCESS;
}

/**
 * `test_mtxmatrix_to_mtxfile()' tests converting matrices to Matrix
 * Market files.
 */
int test_mtxmatrix_to_mtxfile(void)
{
    int err;

    /*
     * Array formats
     */

    {
        struct mtxmatrix A;
        int num_rows = 3;
        int num_columns = 3;
        float Adata[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0, 8.0f, 9.0f};
        int Asize = sizeof(Adata) / sizeof(*Adata);
        err = mtxmatrix_init_array_real_single(&A, num_rows, num_columns, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        struct mtxfile mtxfile;
        err = mtxmatrix_to_mtxfile(&A, &mtxfile);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(mtxfile_matrix, mtxfile.header.object);
        TEST_ASSERT_EQ(mtxfile_array, mtxfile.header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtxfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxfile.header.symmetry);
        TEST_ASSERT_EQ(num_rows, mtxfile.size.num_rows);
        TEST_ASSERT_EQ(num_columns, mtxfile.size.num_columns);
        TEST_ASSERT_EQ(-1, mtxfile.size.num_nonzeros);
        TEST_ASSERT_EQ(mtx_single, mtxfile.precision);
        const float * data = mtxfile.data.array_real_single;
        for (int64_t k = 0; k < Asize; k++)
            TEST_ASSERT_EQ(Adata[k], data[k]);
        mtxfile_free(&mtxfile);
        mtxmatrix_free(&A);
    }

    {
        struct mtxmatrix A;
        int num_rows = 3;
        int num_columns = 3;
        double Adata[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
        int Asize = sizeof(Adata) / sizeof(*Adata);
        err = mtxmatrix_init_array_real_double(&A, num_rows, num_columns, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        struct mtxfile mtxfile;
        err = mtxmatrix_to_mtxfile(&A, &mtxfile);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(mtxfile_matrix, mtxfile.header.object);
        TEST_ASSERT_EQ(mtxfile_array, mtxfile.header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtxfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxfile.header.symmetry);
        TEST_ASSERT_EQ(num_rows, mtxfile.size.num_rows);
        TEST_ASSERT_EQ(num_columns, mtxfile.size.num_columns);
        TEST_ASSERT_EQ(-1, mtxfile.size.num_nonzeros);
        TEST_ASSERT_EQ(mtx_double, mtxfile.precision);
        const double * data = mtxfile.data.array_real_double;
        for (int64_t k = 0; k < Asize; k++)
            TEST_ASSERT_EQ(Adata[k], data[k]);
        mtxfile_free(&mtxfile);
        mtxmatrix_free(&A);
    }

    {
        struct mtxmatrix A;
        int num_rows = 2;
        int num_columns = 2;
        float Adata[][2] = {{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}, {7.0, 8.0f}};
        int Asize = sizeof(Adata) / sizeof(*Adata);
        err = mtxmatrix_init_array_complex_single(&A, num_rows, num_columns, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        struct mtxfile mtxfile;
        err = mtxmatrix_to_mtxfile(&A, &mtxfile);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(mtxfile_matrix, mtxfile.header.object);
        TEST_ASSERT_EQ(mtxfile_array, mtxfile.header.format);
        TEST_ASSERT_EQ(mtxfile_complex, mtxfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxfile.header.symmetry);
        TEST_ASSERT_EQ(num_rows, mtxfile.size.num_rows);
        TEST_ASSERT_EQ(num_columns, mtxfile.size.num_columns);
        TEST_ASSERT_EQ(-1, mtxfile.size.num_nonzeros);
        TEST_ASSERT_EQ(mtx_single, mtxfile.precision);
        const float (* data)[2] = mtxfile.data.array_complex_single;
        for (int64_t k = 0; k < Asize; k++) {
            TEST_ASSERT_EQ(Adata[k][0], data[k][0]);
            TEST_ASSERT_EQ(Adata[k][1], data[k][1]);
        }
        mtxfile_free(&mtxfile);
        mtxmatrix_free(&A);
    }

    {
        struct mtxmatrix A;
        int num_rows = 2;
        int num_columns = 2;
        double Adata[][2] = {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}, {7.0, 8.0}};
        int Asize = sizeof(Adata) / sizeof(*Adata);
        err = mtxmatrix_init_array_complex_double(&A, num_rows, num_columns, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        struct mtxfile mtxfile;
        err = mtxmatrix_to_mtxfile(&A, &mtxfile);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(mtxfile_matrix, mtxfile.header.object);
        TEST_ASSERT_EQ(mtxfile_array, mtxfile.header.format);
        TEST_ASSERT_EQ(mtxfile_complex, mtxfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxfile.header.symmetry);
        TEST_ASSERT_EQ(num_rows, mtxfile.size.num_rows);
        TEST_ASSERT_EQ(num_columns, mtxfile.size.num_columns);
        TEST_ASSERT_EQ(-1, mtxfile.size.num_nonzeros);
        TEST_ASSERT_EQ(mtx_double, mtxfile.precision);
        const double (* data)[2] = mtxfile.data.array_complex_double;
        for (int64_t k = 0; k < Asize; k++) {
            TEST_ASSERT_EQ(Adata[k][0], data[k][0]);
            TEST_ASSERT_EQ(Adata[k][1], data[k][1]);
        }
        mtxfile_free(&mtxfile);
        mtxmatrix_free(&A);
    }

    {
        struct mtxmatrix A;
        int num_rows = 3;
        int num_columns = 3;
        int32_t Adata[] = {1, 2, 3, 4, 5, 6, 7.0, 8, 9};
        int Asize = sizeof(Adata) / sizeof(*Adata);
        err = mtxmatrix_init_array_integer_single(&A, num_rows, num_columns, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        struct mtxfile mtxfile;
        err = mtxmatrix_to_mtxfile(&A, &mtxfile);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(mtxfile_matrix, mtxfile.header.object);
        TEST_ASSERT_EQ(mtxfile_array, mtxfile.header.format);
        TEST_ASSERT_EQ(mtxfile_integer, mtxfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxfile.header.symmetry);
        TEST_ASSERT_EQ(num_rows, mtxfile.size.num_rows);
        TEST_ASSERT_EQ(num_columns, mtxfile.size.num_columns);
        TEST_ASSERT_EQ(-1, mtxfile.size.num_nonzeros);
        TEST_ASSERT_EQ(mtx_single, mtxfile.precision);
        const int32_t * data = mtxfile.data.array_integer_single;
        for (int64_t k = 0; k < Asize; k++)
            TEST_ASSERT_EQ(Adata[k], data[k]);
        mtxfile_free(&mtxfile);
        mtxmatrix_free(&A);
    }

    {
        struct mtxmatrix A;
        int num_rows = 3;
        int num_columns = 3;
        int64_t Adata[] = {1, 2, 3, 4, 5, 6, 7.0, 8, 9};
        int Asize = sizeof(Adata) / sizeof(*Adata);
        err = mtxmatrix_init_array_integer_double(&A, num_rows, num_columns, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        struct mtxfile mtxfile;
        err = mtxmatrix_to_mtxfile(&A, &mtxfile);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(mtxfile_matrix, mtxfile.header.object);
        TEST_ASSERT_EQ(mtxfile_array, mtxfile.header.format);
        TEST_ASSERT_EQ(mtxfile_integer, mtxfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxfile.header.symmetry);
        TEST_ASSERT_EQ(num_rows, mtxfile.size.num_rows);
        TEST_ASSERT_EQ(num_columns, mtxfile.size.num_columns);
        TEST_ASSERT_EQ(-1, mtxfile.size.num_nonzeros);
        TEST_ASSERT_EQ(mtx_double, mtxfile.precision);
        const int64_t * data = mtxfile.data.array_integer_double;
        for (int64_t k = 0; k < Asize; k++)
            TEST_ASSERT_EQ(Adata[k], data[k]);
        mtxfile_free(&mtxfile);
        mtxmatrix_free(&A);
    }

    /*
     * Coordinate formats
     */

    {
        int num_rows = 3;
        int num_columns = 3;
        int nnz = 4;
        int rowidx[] = {1, 1, 2, 3};
        int colidx[] = {1, 3, 1, 3};
        float Adata[] = {1.0f, 3.0f, 4.0f, 9.0f};
        struct mtxmatrix A;
        err = mtxmatrix_init_coordinate_real_single(
            &A, num_rows, num_columns, nnz, rowidx, colidx, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        struct mtxfile mtxfile;
        err = mtxmatrix_to_mtxfile(&A, &mtxfile);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(mtxfile_matrix, mtxfile.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxfile.header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtxfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxfile.header.symmetry);
        TEST_ASSERT_EQ(num_rows, mtxfile.size.num_rows);
        TEST_ASSERT_EQ(num_columns, mtxfile.size.num_columns);
        TEST_ASSERT_EQ(nnz, mtxfile.size.num_nonzeros);
        TEST_ASSERT_EQ(mtx_single, mtxfile.precision);
        const struct mtxfile_matrix_coordinate_real_single * data =
            mtxfile.data.matrix_coordinate_real_single;
        for (int64_t k = 0; k < nnz; k++) {
            TEST_ASSERT_EQ(rowidx[k], data[k].i);
            TEST_ASSERT_EQ(colidx[k], data[k].j);
            TEST_ASSERT_EQ(Adata[k], data[k].a);
        }
        mtxfile_free(&mtxfile);
        mtxmatrix_free(&A);
    }

    {
        int num_rows = 3;
        int num_columns = 3;
        int nnz = 4;
        int rowidx[] = {1, 1, 2, 3};
        int colidx[] = {1, 3, 1, 3};
        double Adata[] = {1.0, 3.0, 4.0, 9.0};
        struct mtxmatrix A;
        err = mtxmatrix_init_coordinate_real_double(
            &A, num_rows, num_columns, nnz, rowidx, colidx, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        struct mtxfile mtxfile;
        err = mtxmatrix_to_mtxfile(&A, &mtxfile);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(mtxfile_matrix, mtxfile.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxfile.header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtxfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxfile.header.symmetry);
        TEST_ASSERT_EQ(num_rows, mtxfile.size.num_rows);
        TEST_ASSERT_EQ(num_columns, mtxfile.size.num_columns);
        TEST_ASSERT_EQ(nnz, mtxfile.size.num_nonzeros);
        TEST_ASSERT_EQ(mtx_double, mtxfile.precision);
        const struct mtxfile_matrix_coordinate_real_double * data =
            mtxfile.data.matrix_coordinate_real_double;
        for (int64_t k = 0; k < nnz; k++) {
            TEST_ASSERT_EQ(rowidx[k], data[k].i);
            TEST_ASSERT_EQ(colidx[k], data[k].j);
            TEST_ASSERT_EQ(Adata[k], data[k].a);
        }
        mtxfile_free(&mtxfile);
        mtxmatrix_free(&A);
    }

    {
        int num_rows = 3;
        int num_columns = 3;
        int nnz = 4;
        int rowidx[] = {1, 1, 2, 3};
        int colidx[] = {1, 3, 1, 3};
        float Adata[][2] = {{1.0f,-1.0f}, {3.0f,-3.0f}, {4.0f,-4.0f}, {9.0f,-9.0f}};
        struct mtxmatrix A;
        err = mtxmatrix_init_coordinate_complex_single(
            &A, num_rows, num_columns, nnz, rowidx, colidx, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        struct mtxfile mtxfile;
        err = mtxmatrix_to_mtxfile(&A, &mtxfile);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(mtxfile_matrix, mtxfile.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxfile.header.format);
        TEST_ASSERT_EQ(mtxfile_complex, mtxfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxfile.header.symmetry);
        TEST_ASSERT_EQ(num_rows, mtxfile.size.num_rows);
        TEST_ASSERT_EQ(num_columns, mtxfile.size.num_columns);
        TEST_ASSERT_EQ(nnz, mtxfile.size.num_nonzeros);
        TEST_ASSERT_EQ(mtx_single, mtxfile.precision);
        const struct mtxfile_matrix_coordinate_complex_single * data =
            mtxfile.data.matrix_coordinate_complex_single;
        for (int64_t k = 0; k < nnz; k++) {
            TEST_ASSERT_EQ(rowidx[k], data[k].i);
            TEST_ASSERT_EQ(colidx[k], data[k].j);
            TEST_ASSERT_EQ(Adata[k][0], data[k].a[0]);
            TEST_ASSERT_EQ(Adata[k][1], data[k].a[1]);
        }
        mtxfile_free(&mtxfile);
        mtxmatrix_free(&A);
    }

    {
        int num_rows = 3;
        int num_columns = 3;
        int nnz = 4;
        int rowidx[] = {1, 1, 2, 3};
        int colidx[] = {1, 3, 1, 3};
        double Adata[][2] = {{1.0,-1.0}, {3.0,-3.0}, {4.0,-4.0}, {9.0,-9.0}};
        struct mtxmatrix A;
        err = mtxmatrix_init_coordinate_complex_double(
            &A, num_rows, num_columns, nnz, rowidx, colidx, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        struct mtxfile mtxfile;
        err = mtxmatrix_to_mtxfile(&A, &mtxfile);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(mtxfile_matrix, mtxfile.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxfile.header.format);
        TEST_ASSERT_EQ(mtxfile_complex, mtxfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxfile.header.symmetry);
        TEST_ASSERT_EQ(num_rows, mtxfile.size.num_rows);
        TEST_ASSERT_EQ(num_columns, mtxfile.size.num_columns);
        TEST_ASSERT_EQ(nnz, mtxfile.size.num_nonzeros);
        TEST_ASSERT_EQ(mtx_double, mtxfile.precision);
        const struct mtxfile_matrix_coordinate_complex_double * data =
            mtxfile.data.matrix_coordinate_complex_double;
        for (int64_t k = 0; k < nnz; k++) {
            TEST_ASSERT_EQ(rowidx[k], data[k].i);
            TEST_ASSERT_EQ(colidx[k], data[k].j);
            TEST_ASSERT_EQ(Adata[k][0], data[k].a[0]);
            TEST_ASSERT_EQ(Adata[k][1], data[k].a[1]);
        }
        mtxfile_free(&mtxfile);
        mtxmatrix_free(&A);
    }

    {
        int num_rows = 3;
        int num_columns = 3;
        int nnz = 4;
        int rowidx[] = {1, 1, 2, 3};
        int colidx[] = {1, 3, 1, 3};
        int32_t Adata[] = {1, 3, 4, 9};
        struct mtxmatrix A;
        err = mtxmatrix_init_coordinate_integer_single(
            &A, num_rows, num_columns, nnz, rowidx, colidx, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        struct mtxfile mtxfile;
        err = mtxmatrix_to_mtxfile(&A, &mtxfile);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(mtxfile_matrix, mtxfile.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxfile.header.format);
        TEST_ASSERT_EQ(mtxfile_integer, mtxfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxfile.header.symmetry);
        TEST_ASSERT_EQ(num_rows, mtxfile.size.num_rows);
        TEST_ASSERT_EQ(num_columns, mtxfile.size.num_columns);
        TEST_ASSERT_EQ(nnz, mtxfile.size.num_nonzeros);
        TEST_ASSERT_EQ(mtx_single, mtxfile.precision);
        const struct mtxfile_matrix_coordinate_integer_single * data =
            mtxfile.data.matrix_coordinate_integer_single;
        for (int64_t k = 0; k < nnz; k++) {
            TEST_ASSERT_EQ(rowidx[k], data[k].i);
            TEST_ASSERT_EQ(colidx[k], data[k].j);
            TEST_ASSERT_EQ(Adata[k], data[k].a);
        }
        mtxfile_free(&mtxfile);
        mtxmatrix_free(&A);
    }

    {
        int num_rows = 3;
        int num_columns = 3;
        int nnz = 4;
        int rowidx[] = {1, 1, 2, 3};
        int colidx[] = {1, 3, 1, 3};
        int64_t Adata[] = {1, 3, 4, 9};
        struct mtxmatrix A;
        err = mtxmatrix_init_coordinate_integer_double(
            &A, num_rows, num_columns, nnz, rowidx, colidx, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        struct mtxfile mtxfile;
        err = mtxmatrix_to_mtxfile(&A, &mtxfile);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(mtxfile_matrix, mtxfile.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxfile.header.format);
        TEST_ASSERT_EQ(mtxfile_integer, mtxfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxfile.header.symmetry);
        TEST_ASSERT_EQ(num_rows, mtxfile.size.num_rows);
        TEST_ASSERT_EQ(num_columns, mtxfile.size.num_columns);
        TEST_ASSERT_EQ(nnz, mtxfile.size.num_nonzeros);
        TEST_ASSERT_EQ(mtx_double, mtxfile.precision);
        const struct mtxfile_matrix_coordinate_integer_double * data =
            mtxfile.data.matrix_coordinate_integer_double;
        for (int64_t k = 0; k < nnz; k++) {
            TEST_ASSERT_EQ(rowidx[k], data[k].i);
            TEST_ASSERT_EQ(colidx[k], data[k].j);
            TEST_ASSERT_EQ(Adata[k], data[k].a);
        }
        mtxfile_free(&mtxfile);
        mtxmatrix_free(&A);
    }

    {
        int num_rows = 3;
        int num_columns = 3;
        int nnz = 4;
        int rowidx[] = {1, 1, 2, 3};
        int colidx[] = {1, 3, 1, 3};
        struct mtxmatrix A;
        err = mtxmatrix_init_coordinate_pattern(
            &A, num_rows, num_columns, nnz, rowidx, colidx);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        struct mtxfile mtxfile;
        err = mtxmatrix_to_mtxfile(&A, &mtxfile);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(mtxfile_matrix, mtxfile.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxfile.header.format);
        TEST_ASSERT_EQ(mtxfile_pattern, mtxfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxfile.header.symmetry);
        TEST_ASSERT_EQ(num_rows, mtxfile.size.num_rows);
        TEST_ASSERT_EQ(num_columns, mtxfile.size.num_columns);
        TEST_ASSERT_EQ(nnz, mtxfile.size.num_nonzeros);
        const struct mtxfile_matrix_coordinate_pattern * data =
            mtxfile.data.matrix_coordinate_pattern;
        for (int64_t k = 0; k < nnz; k++) {
            TEST_ASSERT_EQ(rowidx[k], data[k].i);
            TEST_ASSERT_EQ(colidx[k], data[k].j);
        }
        mtxfile_free(&mtxfile);
        mtxmatrix_free(&A);
    }
    return TEST_SUCCESS;
}

/**
 * `main()' entry point and test driver.
 */
int main(int argc, char * argv[])
{
    TEST_SUITE_BEGIN("Running tests for matrices\n");
    TEST_RUN(test_mtxmatrix_from_mtxfile);
    TEST_RUN(test_mtxmatrix_to_mtxfile);
    TEST_SUITE_END();
    return (TEST_SUITE_STATUS == TEST_SUCCESS) ?
        EXIT_SUCCESS : EXIT_FAILURE;
}