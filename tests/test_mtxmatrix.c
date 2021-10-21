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
#include <libmtx/vector/vector.h>
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
        TEST_ASSERT_EQ(x_->rowidx[0], 0); TEST_ASSERT_EQ(x_->colidx[0], 0);
        TEST_ASSERT_EQ(x_->rowidx[1], 0); TEST_ASSERT_EQ(x_->colidx[1], 2);
        TEST_ASSERT_EQ(x_->rowidx[2], 1); TEST_ASSERT_EQ(x_->colidx[2], 0);
        TEST_ASSERT_EQ(x_->rowidx[3], 2); TEST_ASSERT_EQ(x_->colidx[3], 2);
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
        struct mtxfile_matrix_coordinate_real_single mtxdata[] = {
            {1,1,1.0f}, {1,3,3.0f}, {2,1,4.0f}, {3,3,9.0f}};
        int64_t num_nonzeros = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile mtxfile;
        err = mtxfile_init_matrix_coordinate_real_single(
            &mtxfile, mtxfile_symmetric, num_rows, num_columns, num_nonzeros, mtxdata);
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
        TEST_ASSERT_EQ(6, x_->num_nonzeros);
        TEST_ASSERT_EQ(x_->rowidx[0], 0); TEST_ASSERT_EQ(x_->colidx[0], 0);
        TEST_ASSERT_EQ(x_->rowidx[1], 0); TEST_ASSERT_EQ(x_->colidx[1], 2);
        TEST_ASSERT_EQ(x_->rowidx[2], 1); TEST_ASSERT_EQ(x_->colidx[2], 0);
        TEST_ASSERT_EQ(x_->rowidx[3], 2); TEST_ASSERT_EQ(x_->colidx[3], 2);
        TEST_ASSERT_EQ(x_->rowidx[4], 2); TEST_ASSERT_EQ(x_->colidx[4], 0);
        TEST_ASSERT_EQ(x_->rowidx[5], 0); TEST_ASSERT_EQ(x_->colidx[5], 1);
        TEST_ASSERT_EQ(x_->data.real_single[0], 1.0f);
        TEST_ASSERT_EQ(x_->data.real_single[1], 3.0f);
        TEST_ASSERT_EQ(x_->data.real_single[2], 4.0f);
        TEST_ASSERT_EQ(x_->data.real_single[3], 9.0f);
        TEST_ASSERT_EQ(x_->data.real_single[4], 3.0f);
        TEST_ASSERT_EQ(x_->data.real_single[5], 4.0f);
        mtxmatrix_free(&x);
        mtxfile_free(&mtxfile);
    }
    {
        int num_rows = 3;
        int num_columns = 3;
        struct mtxfile_matrix_coordinate_real_single mtxdata[] = {
            {1,1,1.0f}, {1,3,3.0f}, {2,1,4.0f}, {3,3,9.0f}};
        int64_t num_nonzeros = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile mtxfile;
        err = mtxfile_init_matrix_coordinate_real_single(
            &mtxfile, mtxfile_skew_symmetric, num_rows, num_columns, num_nonzeros,
            mtxdata);
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
        TEST_ASSERT_EQ(6, x_->num_nonzeros);
        TEST_ASSERT_EQ(x_->rowidx[0], 0); TEST_ASSERT_EQ(x_->colidx[0], 0);
        TEST_ASSERT_EQ(x_->rowidx[1], 0); TEST_ASSERT_EQ(x_->colidx[1], 2);
        TEST_ASSERT_EQ(x_->rowidx[2], 1); TEST_ASSERT_EQ(x_->colidx[2], 0);
        TEST_ASSERT_EQ(x_->rowidx[3], 2); TEST_ASSERT_EQ(x_->colidx[3], 2);
        TEST_ASSERT_EQ(x_->rowidx[4], 2); TEST_ASSERT_EQ(x_->colidx[4], 0);
        TEST_ASSERT_EQ(x_->rowidx[5], 0); TEST_ASSERT_EQ(x_->colidx[5], 1);
        TEST_ASSERT_EQ(x_->data.real_single[0], 1.0f);
        TEST_ASSERT_EQ(x_->data.real_single[1], 3.0f);
        TEST_ASSERT_EQ(x_->data.real_single[2], 4.0f);
        TEST_ASSERT_EQ(x_->data.real_single[3], 9.0f);
        TEST_ASSERT_EQ(x_->data.real_single[4], -3.0f);
        TEST_ASSERT_EQ(x_->data.real_single[5], -4.0f);
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
        TEST_ASSERT_EQ(x_->rowidx[0], 0); TEST_ASSERT_EQ(x_->colidx[0], 0);
        TEST_ASSERT_EQ(x_->rowidx[1], 0); TEST_ASSERT_EQ(x_->colidx[1], 2);
        TEST_ASSERT_EQ(x_->rowidx[2], 1); TEST_ASSERT_EQ(x_->colidx[2], 0);
        TEST_ASSERT_EQ(x_->rowidx[3], 2); TEST_ASSERT_EQ(x_->colidx[3], 2);
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
        struct mtxfile_matrix_coordinate_real_double mtxdata[] = {
            {1,1,1.0}, {1,3,3.0}, {2,1,4.0}, {3,3,9.0}};
        int64_t num_nonzeros = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile mtxfile;
        err = mtxfile_init_matrix_coordinate_real_double(
            &mtxfile, mtxfile_symmetric, num_rows, num_columns, num_nonzeros, mtxdata);
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
        TEST_ASSERT_EQ(6, x_->num_nonzeros);
        TEST_ASSERT_EQ(x_->rowidx[0], 0); TEST_ASSERT_EQ(x_->colidx[0], 0);
        TEST_ASSERT_EQ(x_->rowidx[1], 0); TEST_ASSERT_EQ(x_->colidx[1], 2);
        TEST_ASSERT_EQ(x_->rowidx[2], 1); TEST_ASSERT_EQ(x_->colidx[2], 0);
        TEST_ASSERT_EQ(x_->rowidx[3], 2); TEST_ASSERT_EQ(x_->colidx[3], 2);
        TEST_ASSERT_EQ(x_->rowidx[4], 2); TEST_ASSERT_EQ(x_->colidx[4], 0);
        TEST_ASSERT_EQ(x_->rowidx[5], 0); TEST_ASSERT_EQ(x_->colidx[5], 1);
        TEST_ASSERT_EQ(x_->data.real_double[0], 1.0);
        TEST_ASSERT_EQ(x_->data.real_double[1], 3.0);
        TEST_ASSERT_EQ(x_->data.real_double[2], 4.0);
        TEST_ASSERT_EQ(x_->data.real_double[3], 9.0);
        TEST_ASSERT_EQ(x_->data.real_double[4], 3.0);
        TEST_ASSERT_EQ(x_->data.real_double[5], 4.0);
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
            &mtxfile, mtxfile_skew_symmetric, num_rows, num_columns, num_nonzeros,
            mtxdata);
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
        TEST_ASSERT_EQ(6, x_->num_nonzeros);
        TEST_ASSERT_EQ(x_->rowidx[0], 0); TEST_ASSERT_EQ(x_->colidx[0], 0);
        TEST_ASSERT_EQ(x_->rowidx[1], 0); TEST_ASSERT_EQ(x_->colidx[1], 2);
        TEST_ASSERT_EQ(x_->rowidx[2], 1); TEST_ASSERT_EQ(x_->colidx[2], 0);
        TEST_ASSERT_EQ(x_->rowidx[3], 2); TEST_ASSERT_EQ(x_->colidx[3], 2);
        TEST_ASSERT_EQ(x_->rowidx[4], 2); TEST_ASSERT_EQ(x_->colidx[4], 0);
        TEST_ASSERT_EQ(x_->rowidx[5], 0); TEST_ASSERT_EQ(x_->colidx[5], 1);
        TEST_ASSERT_EQ(x_->data.real_double[0], 1.0);
        TEST_ASSERT_EQ(x_->data.real_double[1], 3.0);
        TEST_ASSERT_EQ(x_->data.real_double[2], 4.0);
        TEST_ASSERT_EQ(x_->data.real_double[3], 9.0);
        TEST_ASSERT_EQ(x_->data.real_double[4], -3.0);
        TEST_ASSERT_EQ(x_->data.real_double[5], -4.0);
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
        TEST_ASSERT_EQ(x_->rowidx[0], 0); TEST_ASSERT_EQ(x_->colidx[0], 0);
        TEST_ASSERT_EQ(x_->rowidx[1], 0); TEST_ASSERT_EQ(x_->colidx[1], 2);
        TEST_ASSERT_EQ(x_->rowidx[2], 1); TEST_ASSERT_EQ(x_->colidx[2], 0);
        TEST_ASSERT_EQ(x_->rowidx[3], 2); TEST_ASSERT_EQ(x_->colidx[3], 2);
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
        struct mtxfile_matrix_coordinate_complex_single mtxdata[] = {
            {1,1,{1.0f,-1.0f}}, {1,3,{3.0f,-3.0f}},
            {2,1,{4.0f,-4.0f}}, {3,3,{9.0f,-9.0f}}};
        int64_t num_nonzeros = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile mtxfile;
        err = mtxfile_init_matrix_coordinate_complex_single(
            &mtxfile, mtxfile_symmetric, num_rows, num_columns, num_nonzeros, mtxdata);
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
        TEST_ASSERT_EQ(6, x_->num_nonzeros);
        TEST_ASSERT_EQ(x_->rowidx[0], 0); TEST_ASSERT_EQ(x_->colidx[0], 0);
        TEST_ASSERT_EQ(x_->rowidx[1], 0); TEST_ASSERT_EQ(x_->colidx[1], 2);
        TEST_ASSERT_EQ(x_->rowidx[2], 1); TEST_ASSERT_EQ(x_->colidx[2], 0);
        TEST_ASSERT_EQ(x_->rowidx[3], 2); TEST_ASSERT_EQ(x_->colidx[3], 2);
        TEST_ASSERT_EQ(x_->rowidx[4], 2); TEST_ASSERT_EQ(x_->colidx[4], 0);
        TEST_ASSERT_EQ(x_->rowidx[5], 0); TEST_ASSERT_EQ(x_->colidx[5], 1);
        TEST_ASSERT_EQ(x_->data.complex_single[0][0], 1.0f);
        TEST_ASSERT_EQ(x_->data.complex_single[0][1],-1.0f);
        TEST_ASSERT_EQ(x_->data.complex_single[1][0], 3.0f);
        TEST_ASSERT_EQ(x_->data.complex_single[1][1],-3.0f);
        TEST_ASSERT_EQ(x_->data.complex_single[2][0], 4.0f);
        TEST_ASSERT_EQ(x_->data.complex_single[2][1],-4.0f);
        TEST_ASSERT_EQ(x_->data.complex_single[3][0], 9.0f);
        TEST_ASSERT_EQ(x_->data.complex_single[3][1],-9.0f);
        TEST_ASSERT_EQ(x_->data.complex_single[4][0], 3.0f);
        TEST_ASSERT_EQ(x_->data.complex_single[4][1],-3.0f);
        TEST_ASSERT_EQ(x_->data.complex_single[5][0], 4.0f);
        TEST_ASSERT_EQ(x_->data.complex_single[5][1],-4.0f);
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
            &mtxfile, mtxfile_skew_symmetric, num_rows, num_columns, num_nonzeros,
            mtxdata);
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
        TEST_ASSERT_EQ(6, x_->num_nonzeros);
        TEST_ASSERT_EQ(x_->rowidx[0], 0); TEST_ASSERT_EQ(x_->colidx[0], 0);
        TEST_ASSERT_EQ(x_->rowidx[1], 0); TEST_ASSERT_EQ(x_->colidx[1], 2);
        TEST_ASSERT_EQ(x_->rowidx[2], 1); TEST_ASSERT_EQ(x_->colidx[2], 0);
        TEST_ASSERT_EQ(x_->rowidx[3], 2); TEST_ASSERT_EQ(x_->colidx[3], 2);
        TEST_ASSERT_EQ(x_->rowidx[4], 2); TEST_ASSERT_EQ(x_->colidx[4], 0);
        TEST_ASSERT_EQ(x_->rowidx[5], 0); TEST_ASSERT_EQ(x_->colidx[5], 1);
        TEST_ASSERT_EQ(x_->data.complex_single[0][0], 1.0f);
        TEST_ASSERT_EQ(x_->data.complex_single[0][1],-1.0f);
        TEST_ASSERT_EQ(x_->data.complex_single[1][0], 3.0f);
        TEST_ASSERT_EQ(x_->data.complex_single[1][1],-3.0f);
        TEST_ASSERT_EQ(x_->data.complex_single[2][0], 4.0f);
        TEST_ASSERT_EQ(x_->data.complex_single[2][1],-4.0f);
        TEST_ASSERT_EQ(x_->data.complex_single[3][0], 9.0f);
        TEST_ASSERT_EQ(x_->data.complex_single[3][1],-9.0f);
        TEST_ASSERT_EQ(x_->data.complex_single[4][0],-3.0f);
        TEST_ASSERT_EQ(x_->data.complex_single[4][1], 3.0f);
        TEST_ASSERT_EQ(x_->data.complex_single[5][0],-4.0f);
        TEST_ASSERT_EQ(x_->data.complex_single[5][1], 4.0f);
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
            &mtxfile, mtxfile_hermitian, num_rows, num_columns, num_nonzeros, mtxdata);
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
        TEST_ASSERT_EQ(6, x_->num_nonzeros);
        TEST_ASSERT_EQ(x_->rowidx[0], 0); TEST_ASSERT_EQ(x_->colidx[0], 0);
        TEST_ASSERT_EQ(x_->rowidx[1], 0); TEST_ASSERT_EQ(x_->colidx[1], 2);
        TEST_ASSERT_EQ(x_->rowidx[2], 1); TEST_ASSERT_EQ(x_->colidx[2], 0);
        TEST_ASSERT_EQ(x_->rowidx[3], 2); TEST_ASSERT_EQ(x_->colidx[3], 2);
        TEST_ASSERT_EQ(x_->rowidx[4], 2); TEST_ASSERT_EQ(x_->colidx[4], 0);
        TEST_ASSERT_EQ(x_->rowidx[5], 0); TEST_ASSERT_EQ(x_->colidx[5], 1);
        TEST_ASSERT_EQ(x_->data.complex_single[0][0], 1.0f);
        TEST_ASSERT_EQ(x_->data.complex_single[0][1],-1.0f);
        TEST_ASSERT_EQ(x_->data.complex_single[1][0], 3.0f);
        TEST_ASSERT_EQ(x_->data.complex_single[1][1],-3.0f);
        TEST_ASSERT_EQ(x_->data.complex_single[2][0], 4.0f);
        TEST_ASSERT_EQ(x_->data.complex_single[2][1],-4.0f);
        TEST_ASSERT_EQ(x_->data.complex_single[3][0], 9.0f);
        TEST_ASSERT_EQ(x_->data.complex_single[3][1],-9.0f);
        TEST_ASSERT_EQ(x_->data.complex_single[4][0], 3.0f);
        TEST_ASSERT_EQ(x_->data.complex_single[4][1], 3.0f);
        TEST_ASSERT_EQ(x_->data.complex_single[5][0], 4.0f);
        TEST_ASSERT_EQ(x_->data.complex_single[5][1], 4.0f);
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
        TEST_ASSERT_EQ(x_->rowidx[0], 0); TEST_ASSERT_EQ(x_->colidx[0], 0);
        TEST_ASSERT_EQ(x_->rowidx[1], 0); TEST_ASSERT_EQ(x_->colidx[1], 2);
        TEST_ASSERT_EQ(x_->rowidx[2], 1); TEST_ASSERT_EQ(x_->colidx[2], 0);
        TEST_ASSERT_EQ(x_->rowidx[3], 2); TEST_ASSERT_EQ(x_->colidx[3], 2);
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
        TEST_ASSERT_EQ(x_->rowidx[0], 0); TEST_ASSERT_EQ(x_->colidx[0], 0);
        TEST_ASSERT_EQ(x_->rowidx[1], 0); TEST_ASSERT_EQ(x_->colidx[1], 2);
        TEST_ASSERT_EQ(x_->rowidx[2], 1); TEST_ASSERT_EQ(x_->colidx[2], 0);
        TEST_ASSERT_EQ(x_->rowidx[3], 2); TEST_ASSERT_EQ(x_->colidx[3], 2);
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
        struct mtxfile_matrix_coordinate_integer_single mtxdata[] = {
            {1,1,1}, {1,3,3}, {2,1,4}, {3,3,9}};
        int64_t num_nonzeros = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile mtxfile;
        err = mtxfile_init_matrix_coordinate_integer_single(
            &mtxfile, mtxfile_symmetric, num_rows, num_columns, num_nonzeros, mtxdata);
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
        TEST_ASSERT_EQ(6, x_->num_nonzeros);
        TEST_ASSERT_EQ(x_->rowidx[0], 0); TEST_ASSERT_EQ(x_->colidx[0], 0);
        TEST_ASSERT_EQ(x_->rowidx[1], 0); TEST_ASSERT_EQ(x_->colidx[1], 2);
        TEST_ASSERT_EQ(x_->rowidx[2], 1); TEST_ASSERT_EQ(x_->colidx[2], 0);
        TEST_ASSERT_EQ(x_->rowidx[3], 2); TEST_ASSERT_EQ(x_->colidx[3], 2);
        TEST_ASSERT_EQ(x_->rowidx[4], 2); TEST_ASSERT_EQ(x_->colidx[4], 0);
        TEST_ASSERT_EQ(x_->rowidx[5], 0); TEST_ASSERT_EQ(x_->colidx[5], 1);
        TEST_ASSERT_EQ(x_->data.integer_single[0], 1);
        TEST_ASSERT_EQ(x_->data.integer_single[1], 3);
        TEST_ASSERT_EQ(x_->data.integer_single[2], 4);
        TEST_ASSERT_EQ(x_->data.integer_single[3], 9);
        TEST_ASSERT_EQ(x_->data.integer_single[4], 3);
        TEST_ASSERT_EQ(x_->data.integer_single[5], 4);
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
            &mtxfile, mtxfile_skew_symmetric, num_rows, num_columns, num_nonzeros,
            mtxdata);
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
        TEST_ASSERT_EQ(6, x_->num_nonzeros);
        TEST_ASSERT_EQ(x_->rowidx[0], 0); TEST_ASSERT_EQ(x_->colidx[0], 0);
        TEST_ASSERT_EQ(x_->rowidx[1], 0); TEST_ASSERT_EQ(x_->colidx[1], 2);
        TEST_ASSERT_EQ(x_->rowidx[2], 1); TEST_ASSERT_EQ(x_->colidx[2], 0);
        TEST_ASSERT_EQ(x_->rowidx[3], 2); TEST_ASSERT_EQ(x_->colidx[3], 2);
        TEST_ASSERT_EQ(x_->rowidx[4], 2); TEST_ASSERT_EQ(x_->colidx[4], 0);
        TEST_ASSERT_EQ(x_->rowidx[5], 0); TEST_ASSERT_EQ(x_->colidx[5], 1);
        TEST_ASSERT_EQ(x_->data.integer_single[0], 1);
        TEST_ASSERT_EQ(x_->data.integer_single[1], 3);
        TEST_ASSERT_EQ(x_->data.integer_single[2], 4);
        TEST_ASSERT_EQ(x_->data.integer_single[3], 9);
        TEST_ASSERT_EQ(x_->data.integer_single[4], -3);
        TEST_ASSERT_EQ(x_->data.integer_single[5], -4);
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
        TEST_ASSERT_EQ(x_->rowidx[0], 0); TEST_ASSERT_EQ(x_->colidx[0], 0);
        TEST_ASSERT_EQ(x_->rowidx[1], 0); TEST_ASSERT_EQ(x_->colidx[1], 2);
        TEST_ASSERT_EQ(x_->rowidx[2], 1); TEST_ASSERT_EQ(x_->colidx[2], 0);
        TEST_ASSERT_EQ(x_->rowidx[3], 2); TEST_ASSERT_EQ(x_->colidx[3], 2);
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
        struct mtxfile_matrix_coordinate_integer_double mtxdata[] = {
            {1,1,1}, {1,3,3}, {2,1,4}, {3,3,9}};
        int64_t num_nonzeros = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile mtxfile;
        err = mtxfile_init_matrix_coordinate_integer_double(
            &mtxfile, mtxfile_symmetric, num_rows, num_columns, num_nonzeros, mtxdata);
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
        TEST_ASSERT_EQ(6, x_->num_nonzeros);
        TEST_ASSERT_EQ(x_->rowidx[0], 0); TEST_ASSERT_EQ(x_->colidx[0], 0);
        TEST_ASSERT_EQ(x_->rowidx[1], 0); TEST_ASSERT_EQ(x_->colidx[1], 2);
        TEST_ASSERT_EQ(x_->rowidx[2], 1); TEST_ASSERT_EQ(x_->colidx[2], 0);
        TEST_ASSERT_EQ(x_->rowidx[3], 2); TEST_ASSERT_EQ(x_->colidx[3], 2);
        TEST_ASSERT_EQ(x_->rowidx[4], 2); TEST_ASSERT_EQ(x_->colidx[4], 0);
        TEST_ASSERT_EQ(x_->rowidx[5], 0); TEST_ASSERT_EQ(x_->colidx[5], 1);
        TEST_ASSERT_EQ(x_->data.integer_double[0], 1);
        TEST_ASSERT_EQ(x_->data.integer_double[1], 3);
        TEST_ASSERT_EQ(x_->data.integer_double[2], 4);
        TEST_ASSERT_EQ(x_->data.integer_double[3], 9);
        TEST_ASSERT_EQ(x_->data.integer_double[4], 3);
        TEST_ASSERT_EQ(x_->data.integer_double[5], 4);
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
            &mtxfile, mtxfile_skew_symmetric, num_rows, num_columns, num_nonzeros,
            mtxdata);
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
        TEST_ASSERT_EQ(6, x_->num_nonzeros);
        TEST_ASSERT_EQ(x_->rowidx[0], 0); TEST_ASSERT_EQ(x_->colidx[0], 0);
        TEST_ASSERT_EQ(x_->rowidx[1], 0); TEST_ASSERT_EQ(x_->colidx[1], 2);
        TEST_ASSERT_EQ(x_->rowidx[2], 1); TEST_ASSERT_EQ(x_->colidx[2], 0);
        TEST_ASSERT_EQ(x_->rowidx[3], 2); TEST_ASSERT_EQ(x_->colidx[3], 2);
        TEST_ASSERT_EQ(x_->rowidx[4], 2); TEST_ASSERT_EQ(x_->colidx[4], 0);
        TEST_ASSERT_EQ(x_->rowidx[5], 0); TEST_ASSERT_EQ(x_->colidx[5], 1);
        TEST_ASSERT_EQ(x_->data.integer_double[0], 1);
        TEST_ASSERT_EQ(x_->data.integer_double[1], 3);
        TEST_ASSERT_EQ(x_->data.integer_double[2], 4);
        TEST_ASSERT_EQ(x_->data.integer_double[3], 9);
        TEST_ASSERT_EQ(x_->data.integer_double[4], -3);
        TEST_ASSERT_EQ(x_->data.integer_double[5], -4);
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
        TEST_ASSERT_EQ(x_->rowidx[0], 0); TEST_ASSERT_EQ(x_->colidx[0], 0);
        TEST_ASSERT_EQ(x_->rowidx[1], 0); TEST_ASSERT_EQ(x_->colidx[1], 2);
        TEST_ASSERT_EQ(x_->rowidx[2], 1); TEST_ASSERT_EQ(x_->colidx[2], 0);
        TEST_ASSERT_EQ(x_->rowidx[3], 2); TEST_ASSERT_EQ(x_->colidx[3], 2);
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
            &mtxfile, mtxfile_symmetric, num_rows, num_columns, num_nonzeros, mtxdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        struct mtxmatrix x;
        err = mtxmatrix_from_mtxfile(&x, &mtxfile, mtxmatrix_auto);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(mtxmatrix_coordinate, x.type);
        const struct mtxmatrix_coordinate * x_ = &x.storage.coordinate;
        TEST_ASSERT_EQ(mtx_field_pattern, x_->field);
        TEST_ASSERT_EQ(3, x_->num_rows);
        TEST_ASSERT_EQ(3, x_->num_columns);
        TEST_ASSERT_EQ(6, x_->num_nonzeros);
        TEST_ASSERT_EQ(x_->rowidx[0], 0); TEST_ASSERT_EQ(x_->colidx[0], 0);
        TEST_ASSERT_EQ(x_->rowidx[1], 0); TEST_ASSERT_EQ(x_->colidx[1], 2);
        TEST_ASSERT_EQ(x_->rowidx[2], 1); TEST_ASSERT_EQ(x_->colidx[2], 0);
        TEST_ASSERT_EQ(x_->rowidx[3], 2); TEST_ASSERT_EQ(x_->colidx[3], 2);
        TEST_ASSERT_EQ(x_->rowidx[4], 2); TEST_ASSERT_EQ(x_->colidx[4], 0);
        TEST_ASSERT_EQ(x_->rowidx[5], 0); TEST_ASSERT_EQ(x_->colidx[5], 1);
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
            TEST_ASSERT_EQ(rowidx[k]+1, data[k].i);
            TEST_ASSERT_EQ(colidx[k]+1, data[k].j);
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
            TEST_ASSERT_EQ(rowidx[k]+1, data[k].i);
            TEST_ASSERT_EQ(colidx[k]+1, data[k].j);
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
            TEST_ASSERT_EQ(rowidx[k]+1, data[k].i);
            TEST_ASSERT_EQ(colidx[k]+1, data[k].j);
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
            TEST_ASSERT_EQ(rowidx[k]+1, data[k].i);
            TEST_ASSERT_EQ(colidx[k]+1, data[k].j);
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
            TEST_ASSERT_EQ(rowidx[k]+1, data[k].i);
            TEST_ASSERT_EQ(colidx[k]+1, data[k].j);
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
            TEST_ASSERT_EQ(rowidx[k]+1, data[k].i);
            TEST_ASSERT_EQ(colidx[k]+1, data[k].j);
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
            TEST_ASSERT_EQ(rowidx[k]+1, data[k].i);
            TEST_ASSERT_EQ(colidx[k]+1, data[k].j);
        }
        mtxfile_free(&mtxfile);
        mtxmatrix_free(&A);
    }
    return TEST_SUCCESS;
}

/**
 * `test_mtxmatrix_gemv_array()` tests computing matrix-vector
 * products for matrices in array format.
 */
int test_mtxmatrix_gemv_array(void)
{
    int err;

    /*
     * For real or integer matrices, calculate
     *
     *   ⎡ 1 2 3⎤  ⎡ 3⎤     ⎡ 1⎤  ⎡ 20⎤  ⎡ 3⎤  ⎡ 23⎤
     * 2*⎢ 4 5 6⎥ *⎢ 2⎥ + 3*⎢ 0⎥ =⎢ 56⎥ +⎢ 0⎥ =⎢ 56⎥,
     *   ⎣ 7 8 9⎦  ⎣ 1⎦     ⎣ 2⎦  ⎣ 92⎦  ⎣ 6⎦  ⎣ 98⎦
     *
     * and
     *
     *   ⎡ 1 4 7⎤  ⎡ 3⎤     ⎡ 1⎤  ⎡ 36⎤  ⎡ 3⎤  ⎡ 39⎤
     * 2*⎢ 2 5 8⎥ *⎢ 2⎥ + 3*⎢ 0⎥ =⎢ 48⎥ +⎢ 0⎥ =⎢ 48⎥.
     *   ⎣ 3 6 9⎦  ⎣ 1⎦     ⎣ 2⎦  ⎣ 60⎦  ⎣ 6⎦  ⎣ 66⎦
     *
     * For complex matrices, calculate
     *
     *   ⎡ 1+2i 3+4i⎤  ⎡ 3+1i⎤     ⎡ 1+0i⎤  ⎡-8+34i⎤  ⎡ 3   ⎤  ⎡-5+34i⎤
     * 2*⎢          ⎥ *⎢     ⎥ + 3*⎢     ⎥ =⎢      ⎥ +⎢     ⎥ =⎢      ⎥,
     *   ⎣ 5+6i 7+8i⎦  ⎣ 1+2i⎦     ⎣ 2+2i⎦  ⎣ 0+90i⎦  ⎣ 6+6i⎦  ⎣ 6+96i⎦
     *
     * and
     *
     *   ⎡ 1+2i 5+6i⎤  ⎡ 3+1i⎤     ⎡ 1+0i⎤  ⎡-12+46i⎤  ⎡ 3   ⎤  ⎡-9+46i⎤
     * 2*⎢          ⎥ *⎢     ⎥ + 3*⎢     ⎥ =⎢       ⎥ +⎢     ⎥ =⎢      ⎥,
     *   ⎣ 3+4i 7+8i⎦  ⎣ 1+2i⎦     ⎣ 2+2i⎦  ⎣ -8+74i⎦  ⎣ 6+6i⎦  ⎣-2+80i⎦
     *
     * and
     *
     *   ⎡ 1-2i 5-6i⎤  ⎡ 3+1i⎤     ⎡ 1+0i⎤  ⎡ 44-2i⎤  ⎡ 3   ⎤  ⎡ 47-2i⎤
     * 2*⎢          ⎥ *⎢     ⎥ + 3*⎢     ⎥ =⎢      ⎥ +⎢     ⎥ =⎢      ⎥.
     *   ⎣ 3-4i 7-8i⎦  ⎣ 1+2i⎦     ⎣ 2+2i⎦  ⎣ 72-6i⎦  ⎣ 6+6i⎦  ⎣ 78   ⎦
     *
     * and
     *
     *    ⎡ 1+2i 3+4i⎤  ⎡ 3+1i⎤          ⎡ 1+0i⎤  ⎡-34-8i⎤  ⎡ 3+1i⎤  ⎡-31-7i⎤
     * 2i*⎢          ⎥ *⎢     ⎥ + (3+1i)*⎢     ⎥ =⎢      ⎥ +⎢     ⎥ =⎢      ⎥,
     *    ⎣ 5+6i 7+8i⎦  ⎣ 1+2i⎦          ⎣ 2+2i⎦  ⎣-90   ⎦  ⎣ 4+8i⎦  ⎣-86+8i⎦
     *
     * and
     *
     *    ⎡ 1+2i 5+6i⎤  ⎡ 3+1i⎤          ⎡ 1+0i⎤  ⎡-46-12i⎤  ⎡ 3+1i⎤  ⎡-43-11i⎤
     * 2i*⎢          ⎥ *⎢     ⎥ + (3+1i)*⎢     ⎥ =⎢       ⎥ +⎢     ⎥ =⎢       ⎥,
     *    ⎣ 3+4i 7+8i⎦  ⎣ 1+2i⎦          ⎣ 2+2i⎦  ⎣-74- 8i⎦  ⎣ 4+8i⎦  ⎣-70    ⎦
     *
     * and
     *
     *    ⎡ 1-2i 5-6i⎤  ⎡ 3+1i⎤          ⎡ 1+0i⎤  ⎡ 2+44i⎤  ⎡ 3+1i⎤  ⎡  5+45i⎤
     * 2i*⎢          ⎥ *⎢     ⎥ + (3+1i)*⎢     ⎥ =⎢      ⎥ +⎢     ⎥ =⎢       ⎥.
     *    ⎣ 3-4i 7-8i⎦  ⎣ 1+2i⎦          ⎣ 2+2i⎦  ⎣ 6+72i⎦  ⎣ 4+8i⎦  ⎣ 10+80i⎦
     */

    /*
     * Real matrices
     */

    {
        struct mtxmatrix A;
        struct mtxvector x;
        struct mtxvector y;
        int num_rows = 3;
        int num_columns = 3;
        float Adata[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0, 8.0f, 9.0f};
        float xdata[] = {3.0f, 2.0f, 1.0f};
        float ydata[] = {1.0f, 0.0f, 2.0f};
        err = mtxmatrix_init_array_real_single(&A, num_rows, num_columns, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        err = mtxvector_init_array_real_single(&x, num_columns, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        {
            err = mtxvector_init_array_real_single(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            err = mtxmatrix_sgemv(mtx_notrans, 2.0f, &A, &x, 3.0f, &y);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
            TEST_ASSERT_EQ(mtx_field_real, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(3, y_->size);
            TEST_ASSERT_EQ(y_->data.real_single[0], 23.0f);
            TEST_ASSERT_EQ(y_->data.real_single[1], 56.0f);
            TEST_ASSERT_EQ(y_->data.real_single[2], 98.0f);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_array_real_single(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            err = mtxmatrix_sgemv(mtx_trans, 2.0f, &A, &x, 3.0f, &y);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
            TEST_ASSERT_EQ(mtx_field_real, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(3, y_->size);
            TEST_ASSERT_EQ(y_->data.real_single[0], 39.0f);
            TEST_ASSERT_EQ(y_->data.real_single[1], 48.0f);
            TEST_ASSERT_EQ(y_->data.real_single[2], 66.0f);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_array_real_single(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            err = mtxmatrix_dgemv(mtx_notrans, 2.0, &A, &x, 3.0, &y);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
            TEST_ASSERT_EQ(mtx_field_real, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(3, y_->size);
            TEST_ASSERT_EQ(y_->data.real_single[0], 23.0f);
            TEST_ASSERT_EQ(y_->data.real_single[1], 56.0f);
            TEST_ASSERT_EQ(y_->data.real_single[2], 98.0f);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_array_real_single(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            err = mtxmatrix_dgemv(mtx_trans, 2.0, &A, &x, 3.0, &y);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
            TEST_ASSERT_EQ(mtx_field_real, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(3, y_->size);
            TEST_ASSERT_EQ(y_->data.real_single[0], 39.0f);
            TEST_ASSERT_EQ(y_->data.real_single[1], 48.0f);
            TEST_ASSERT_EQ(y_->data.real_single[2], 66.0f);
            mtxvector_free(&y);
        }
        mtxvector_free(&x);
        mtxmatrix_free(&A);
    }

    {
        struct mtxmatrix A;
        struct mtxvector x;
        struct mtxvector y;
        int num_rows = 3;
        int num_columns = 3;
        double Adata[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
        double xdata[] = {3.0, 2.0, 1.0};
        double ydata[] = {1.0, 0.0, 2.0};
        err = mtxmatrix_init_array_real_double(&A, num_rows, num_columns, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        err = mtxvector_init_array_real_double(&x, num_columns, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        {
            err = mtxvector_init_array_real_double(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            err = mtxmatrix_sgemv(mtx_notrans, 2.0f, &A, &x, 3.0f, &y);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
            TEST_ASSERT_EQ(mtx_field_real, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(3, y_->size);
            TEST_ASSERT_EQ(y_->data.real_double[0], 23.0);
            TEST_ASSERT_EQ(y_->data.real_double[1], 56.0);
            TEST_ASSERT_EQ(y_->data.real_double[2], 98.0);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_array_real_double(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            err = mtxmatrix_sgemv(mtx_trans, 2.0f, &A, &x, 3.0f, &y);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
            TEST_ASSERT_EQ(mtx_field_real, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(3, y_->size);
            TEST_ASSERT_EQ(y_->data.real_double[0], 39.0);
            TEST_ASSERT_EQ(y_->data.real_double[1], 48.0);
            TEST_ASSERT_EQ(y_->data.real_double[2], 66.0);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_array_real_double(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            err = mtxmatrix_dgemv(mtx_notrans, 2.0, &A, &x, 3.0, &y);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
            TEST_ASSERT_EQ(mtx_field_real, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(3, y_->size);
            TEST_ASSERT_EQ(y_->data.real_double[0], 23.0);
            TEST_ASSERT_EQ(y_->data.real_double[1], 56.0);
            TEST_ASSERT_EQ(y_->data.real_double[2], 98.0);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_array_real_double(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            err = mtxmatrix_dgemv(mtx_trans, 2.0, &A, &x, 3.0, &y);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
            TEST_ASSERT_EQ(mtx_field_real, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(3, y_->size);
            TEST_ASSERT_EQ(y_->data.real_double[0], 39.0);
            TEST_ASSERT_EQ(y_->data.real_double[1], 48.0);
            TEST_ASSERT_EQ(y_->data.real_double[2], 66.0);
            mtxvector_free(&y);
        }
        mtxvector_free(&x);
        mtxmatrix_free(&A);
    }

    /*
     * Complex matrices
     */

    {
        struct mtxmatrix A;
        struct mtxvector x;
        struct mtxvector y;
        int num_rows = 2;
        int num_columns = 2;
        float Adata[][2] = {{1.0f,2.0f}, {3.0f,4.0f}, {5.0f,6.0f}, {7.0,8.0f}};
        float xdata[][2] = {{3.0f,1.0f}, {1.0f,2.0f}};
        float ydata[][2] = {{1.0f,0.0f}, {2.0f,2.0f}};
        err = mtxmatrix_init_array_complex_single(&A, num_rows, num_columns, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        err = mtxvector_init_array_complex_single(&x, num_columns, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        {
            err = mtxvector_init_array_complex_single(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            err = mtxmatrix_sgemv(mtx_notrans, 2.0f, &A, &x, 3.0f, &y);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_single[0][0], -5.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[0][1], 34.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][0],  6.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1], 96.0f);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_array_complex_single(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            err = mtxmatrix_sgemv(mtx_trans, 2.0f, &A, &x, 3.0f, &y);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_single[0][0], -9.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[0][1], 46.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][0], -2.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1], 80.0f);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_array_complex_single(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            err = mtxmatrix_sgemv(mtx_conjtrans, 2.0f, &A, &x, 3.0f, &y);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_single[0][0], 47.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[0][1], -2.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][0], 78.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1],  0.0f);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_array_complex_single(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            err = mtxmatrix_dgemv(mtx_notrans, 2.0, &A, &x, 3.0, &y);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_single[0][0], -5.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[0][1], 34.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][0],  6.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1], 96.0f);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_array_complex_single(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            err = mtxmatrix_dgemv(mtx_trans, 2.0, &A, &x, 3.0, &y);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_single[0][0], -9.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[0][1], 46.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][0], -2.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1], 80.0f);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_array_complex_single(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            err = mtxmatrix_dgemv(mtx_conjtrans, 2.0, &A, &x, 3.0, &y);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_single[0][0], 47.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[0][1], -2.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][0], 78.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1],  0.0f);
            mtxvector_free(&y);
        }

        float calpha[2] = {0.0f, 2.0f};
        float cbeta[2]  = {3.0f, 1.0f};
        {
            err = mtxvector_init_array_complex_single(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            err = mtxmatrix_cgemv(mtx_notrans, calpha, &A, &x, cbeta, &y);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_single[0][0], -31.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[0][1],  -7.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][0], -86.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1],   8.0f);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_array_complex_single(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            err = mtxmatrix_cgemv(mtx_trans, calpha, &A, &x, cbeta, &y);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_single[0][0], -43.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[0][1], -11.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][0], -70.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1],   0.0f);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_array_complex_single(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            err = mtxmatrix_cgemv(mtx_conjtrans, calpha, &A, &x, cbeta, &y);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_single[0][0],  5.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[0][1], 45.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][0], 10.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1], 80.0f);
            mtxvector_free(&y);
        }

        double zalpha[2] = {0.0, 2.0};
        double zbeta[2]  = {3.0, 1.0};
        {
            err = mtxvector_init_array_complex_single(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            err = mtxmatrix_zgemv(mtx_notrans, zalpha, &A, &x, zbeta, &y);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_single[0][0], -31.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[0][1],  -7.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][0], -86.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1],   8.0f);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_array_complex_single(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            err = mtxmatrix_zgemv(mtx_trans, zalpha, &A, &x, zbeta, &y);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_single[0][0], -43.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[0][1], -11.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][0], -70.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1],   0.0f);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_array_complex_single(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            err = mtxmatrix_zgemv(mtx_conjtrans, zalpha, &A, &x, zbeta, &y);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_single[0][0],  5.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[0][1], 45.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][0], 10.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1], 80.0f);
            mtxvector_free(&y);
        }

        mtxvector_free(&x);
        mtxmatrix_free(&A);
    }

    {
        struct mtxmatrix A;
        struct mtxvector x;
        struct mtxvector y;
        int num_rows = 2;
        int num_columns = 2;
        double Adata[][2] = {{1.0,2.0}, {3.0,4.0}, {5.0,6.0}, {7.0,8.0}};
        double xdata[][2] = {{3.0,1.0}, {1.0,2.0}};
        double ydata[][2] = {{1.0,0.0}, {2.0,2.0}};
        err = mtxmatrix_init_array_complex_double(&A, num_rows, num_columns, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        err = mtxvector_init_array_complex_double(&x, num_columns, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        {
            err = mtxvector_init_array_complex_double(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            err = mtxmatrix_sgemv(mtx_notrans, 2.0f, &A, &x, 3.0f, &y);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_double[0][0], -5.0);
            TEST_ASSERT_EQ(y_->data.complex_double[0][1], 34.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][0],  6.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][1], 96.0);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_array_complex_double(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            err = mtxmatrix_sgemv(mtx_trans, 2.0f, &A, &x, 3.0f, &y);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_double[0][0], -9.0);
            TEST_ASSERT_EQ(y_->data.complex_double[0][1], 46.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][0], -2.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][1], 80.0);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_array_complex_double(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            err = mtxmatrix_sgemv(mtx_conjtrans, 2.0f, &A, &x, 3.0f, &y);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_double[0][0], 47.0);
            TEST_ASSERT_EQ(y_->data.complex_double[0][1], -2.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][0], 78.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][1],  0.0);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_array_complex_double(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            err = mtxmatrix_dgemv(mtx_notrans, 2.0, &A, &x, 3.0, &y);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_double[0][0], -5.0);
            TEST_ASSERT_EQ(y_->data.complex_double[0][1], 34.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][0],  6.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][1], 96.0);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_array_complex_double(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            err = mtxmatrix_dgemv(mtx_trans, 2.0, &A, &x, 3.0, &y);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_double[0][0], -9.0);
            TEST_ASSERT_EQ(y_->data.complex_double[0][1], 46.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][0], -2.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][1], 80.0);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_array_complex_double(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            err = mtxmatrix_dgemv(mtx_conjtrans, 2.0, &A, &x, 3.0, &y);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_double[0][0], 47.0);
            TEST_ASSERT_EQ(y_->data.complex_double[0][1], -2.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][0], 78.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][1],  0.0);
            mtxvector_free(&y);
        }

        float calpha[2] = {0.0f, 2.0f};
        float cbeta[2]  = {3.0f, 1.0f};
        {
            err = mtxvector_init_array_complex_double(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            err = mtxmatrix_cgemv(mtx_notrans, calpha, &A, &x, cbeta, &y);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_double[0][0], -31.0);
            TEST_ASSERT_EQ(y_->data.complex_double[0][1],  -7.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][0], -86.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][1],   8.0);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_array_complex_double(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            err = mtxmatrix_cgemv(mtx_trans, calpha, &A, &x, cbeta, &y);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_double[0][0], -43.0);
            TEST_ASSERT_EQ(y_->data.complex_double[0][1], -11.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][0], -70.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][1],   0.0);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_array_complex_double(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            err = mtxmatrix_cgemv(mtx_conjtrans, calpha, &A, &x, cbeta, &y);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_double[0][0],  5.0);
            TEST_ASSERT_EQ(y_->data.complex_double[0][1], 45.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][0], 10.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][1], 80.0);
            mtxvector_free(&y);
        }

        double zalpha[2] = {0.0, 2.0};
        double zbeta[2]  = {3.0, 1.0};
        {
            err = mtxvector_init_array_complex_double(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            err = mtxmatrix_zgemv(mtx_notrans, zalpha, &A, &x, zbeta, &y);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_double[0][0], -31.0);
            TEST_ASSERT_EQ(y_->data.complex_double[0][1],  -7.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][0], -86.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][1],   8.0);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_array_complex_double(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            err = mtxmatrix_zgemv(mtx_trans, zalpha, &A, &x, zbeta, &y);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_double[0][0], -43.0);
            TEST_ASSERT_EQ(y_->data.complex_double[0][1], -11.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][0], -70.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][1],   0.0);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_array_complex_double(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            err = mtxmatrix_zgemv(mtx_conjtrans, zalpha, &A, &x, zbeta, &y);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_double[0][0],  5.0);
            TEST_ASSERT_EQ(y_->data.complex_double[0][1], 45.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][0], 10.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][1], 80.0);
            mtxvector_free(&y);
        }

        mtxvector_free(&x);
        mtxmatrix_free(&A);
    }

    /*
     * Integer matrices
     */

    {
        struct mtxmatrix A;
        struct mtxvector x;
        struct mtxvector y;
        int num_rows = 3;
        int num_columns = 3;
        int32_t Adata[] = {1, 2, 3, 4, 5, 6, 7.0, 8, 9};
        int32_t xdata[] = {3, 2, 1};
        int32_t ydata[] = {1, 0, 2};
        err = mtxmatrix_init_array_integer_single(&A, num_rows, num_columns, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        err = mtxvector_init_array_integer_single(&x, num_columns, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        {
            err = mtxvector_init_array_integer_single(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            err = mtxmatrix_sgemv(mtx_notrans, 2.0f, &A, &x, 3.0f, &y);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
            TEST_ASSERT_EQ(mtx_field_integer, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(3, y_->size);
            TEST_ASSERT_EQ(y_->data.integer_single[0], 23);
            TEST_ASSERT_EQ(y_->data.integer_single[1], 56);
            TEST_ASSERT_EQ(y_->data.integer_single[2], 98);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_array_integer_single(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            err = mtxmatrix_sgemv(mtx_trans, 2.0f, &A, &x, 3.0f, &y);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
            TEST_ASSERT_EQ(mtx_field_integer, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(3, y_->size);
            TEST_ASSERT_EQ(y_->data.integer_single[0], 39);
            TEST_ASSERT_EQ(y_->data.integer_single[1], 48);
            TEST_ASSERT_EQ(y_->data.integer_single[2], 66);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_array_integer_single(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            err = mtxmatrix_dgemv(mtx_notrans, 2.0, &A, &x, 3.0, &y);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
            TEST_ASSERT_EQ(mtx_field_integer, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(3, y_->size);
            TEST_ASSERT_EQ(y_->data.integer_single[0], 23);
            TEST_ASSERT_EQ(y_->data.integer_single[1], 56);
            TEST_ASSERT_EQ(y_->data.integer_single[2], 98);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_array_integer_single(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            err = mtxmatrix_dgemv(mtx_trans, 2.0, &A, &x, 3.0, &y);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
            TEST_ASSERT_EQ(mtx_field_integer, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(3, y_->size);
            TEST_ASSERT_EQ(y_->data.integer_single[0], 39);
            TEST_ASSERT_EQ(y_->data.integer_single[1], 48);
            TEST_ASSERT_EQ(y_->data.integer_single[2], 66);
            mtxvector_free(&y);
        }
        mtxvector_free(&x);
        mtxmatrix_free(&A);
    }

    {
        struct mtxmatrix A;
        struct mtxvector x;
        struct mtxvector y;
        int num_rows = 3;
        int num_columns = 3;
        int64_t Adata[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
        int64_t xdata[] = {3, 2, 1};
        int64_t ydata[] = {1, 0, 2};
        err = mtxmatrix_init_array_integer_double(&A, num_rows, num_columns, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        err = mtxvector_init_array_integer_double(&x, num_columns, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        {
            err = mtxvector_init_array_integer_double(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            err = mtxmatrix_sgemv(mtx_notrans, 2.0f, &A, &x, 3.0f, &y);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
            TEST_ASSERT_EQ(mtx_field_integer, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(3, y_->size);
            TEST_ASSERT_EQ(y_->data.integer_double[0], 23);
            TEST_ASSERT_EQ(y_->data.integer_double[1], 56);
            TEST_ASSERT_EQ(y_->data.integer_double[2], 98);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_array_integer_double(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            err = mtxmatrix_sgemv(mtx_trans, 2.0f, &A, &x, 3.0f, &y);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
            TEST_ASSERT_EQ(mtx_field_integer, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(3, y_->size);
            TEST_ASSERT_EQ(y_->data.integer_double[0], 39);
            TEST_ASSERT_EQ(y_->data.integer_double[1], 48);
            TEST_ASSERT_EQ(y_->data.integer_double[2], 66);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_array_integer_double(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            err = mtxmatrix_dgemv(mtx_notrans, 2.0, &A, &x, 3.0, &y);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
            TEST_ASSERT_EQ(mtx_field_integer, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(3, y_->size);
            TEST_ASSERT_EQ(y_->data.integer_double[0], 23);
            TEST_ASSERT_EQ(y_->data.integer_double[1], 56);
            TEST_ASSERT_EQ(y_->data.integer_double[2], 98);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_array_integer_double(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            err = mtxmatrix_dgemv(mtx_trans, 2.0, &A, &x, 3.0, &y);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
            TEST_ASSERT_EQ(mtx_field_integer, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(3, y_->size);
            TEST_ASSERT_EQ(y_->data.integer_double[0], 39);
            TEST_ASSERT_EQ(y_->data.integer_double[1], 48);
            TEST_ASSERT_EQ(y_->data.integer_double[2], 66);
            mtxvector_free(&y);
        }
        mtxvector_free(&x);
        mtxmatrix_free(&A);
    }

    return TEST_SUCCESS;
}

/**
 * `test_mtxmatrix_gemv_coordinate()` tests computing matrix-vector
 * products for matrices in coordinate format.
 */
int test_mtxmatrix_gemv_coordinate(void)
{
    int err;

    /*
     * For real and integer matrices, calculate:
     *
     * 1. sgemv/dgemv, notrans, beta=1.
     *
     *     ⎡ 1 0 3⎤  ⎡ 3⎤  ⎡ 1⎤  ⎡ 12⎤  ⎡ 1⎤  ⎡ 13⎤
     *   2*⎢ 4 5 0⎥ *⎢ 2⎥ +⎢ 0⎥ =⎢ 44⎥ +⎢ 0⎥ =⎢ 44⎥
     *     ⎣ 0 0 9⎦  ⎣ 1⎦  ⎣ 2⎦  ⎣ 18⎦  ⎣ 2⎦  ⎣ 20⎦
     *
     * 2. sgemv/dgemv, notrans, beta=3.
     *
     *     ⎡ 1 0 3⎤  ⎡ 3⎤     ⎡ 1⎤  ⎡ 12⎤  ⎡ 3⎤  ⎡ 15⎤
     *   2*⎢ 4 5 0⎥ *⎢ 2⎥ + 3*⎢ 0⎥ =⎢ 44⎥ +⎢ 0⎥ =⎢ 44⎥
     *     ⎣ 0 0 9⎦  ⎣ 1⎦     ⎣ 2⎦  ⎣ 18⎦  ⎣ 6⎦  ⎣ 24⎦
     *
     * 3. sgemv/dgemv, trans, beta=3.
     *
     *     ⎡ 1 4 0⎤  ⎡ 3⎤     ⎡ 1⎤  ⎡ 22⎤  ⎡ 3⎤  ⎡ 25⎤
     *   2*⎢ 0 5 0⎥ *⎢ 2⎥ + 3*⎢ 0⎥ =⎢ 20⎥ +⎢ 0⎥ =⎢ 20⎥
     *     ⎣ 3 0 9⎦  ⎣ 1⎦     ⎣ 2⎦  ⎣ 36⎦  ⎣ 6⎦  ⎣ 42⎦
     *
     *
     * For complex matrices, calculate:
     *
     * 1. sgemv/dgemv, notrans, beta=1.
     *
     *     ⎡ 1+2i 3+4i⎤  ⎡ 3+1i⎤  ⎡ 1+0i⎤  ⎡ -8+34i⎤  ⎡ 1   ⎤  ⎡ -7+34i⎤
     *   2*⎢          ⎥ *⎢     ⎥ +⎢     ⎥ =⎢       ⎥ +⎢     ⎥ =⎢       ⎥
     *     ⎣    0 7+8i⎦  ⎣ 1+2i⎦  ⎣ 2+2i⎦  ⎣-18+44i⎦  ⎣ 2+2i⎦  ⎣-16+46i⎦
     *
     * 2. sgemv/dgemv, notrans, beta=3.
     *
     *     ⎡ 1+2i 3+4i⎤  ⎡ 3+1i⎤     ⎡ 1+0i⎤  ⎡ -8+34i⎤  ⎡ 3   ⎤  ⎡ -5+34i⎤
     *   2*⎢          ⎥ *⎢     ⎥ + 3*⎢     ⎥ =⎢       ⎥ +⎢     ⎥ =⎢       ⎥
     *     ⎣    0 7+8i⎦  ⎣ 1+2i⎦     ⎣ 2+2i⎦  ⎣-18+44i⎦  ⎣ 6+6i⎦  ⎣-12+50i⎦
     *
     * 3. sgemv/dgemv, trans, beta=3.
     *
     *     ⎡ 1+2i    0⎤  ⎡ 3+1i⎤     ⎡ 1+0i⎤  ⎡  2+14i⎤  ⎡ 3   ⎤  ⎡ 5+14i⎤
     *   2*⎢          ⎥ *⎢     ⎥ + 3*⎢     ⎥ =⎢       ⎥ +⎢     ⎥ =⎢      ⎥
     *     ⎣ 3+4i 7+8i⎦  ⎣ 1+2i⎦     ⎣ 2+2i⎦  ⎣ -8+74i⎦  ⎣ 6+6i⎦  ⎣-2+80i⎦
     *
     * 4. sgemv/dgemv, conjtrans, beta=3.
     *
     *     ⎡ 1-2i    0⎤  ⎡ 3+1i⎤     ⎡ 1+0i⎤  ⎡ 10-10i⎤  ⎡ 3   ⎤  ⎡ 13-10i⎤
     *   2*⎢          ⎥ *⎢     ⎥ + 3*⎢     ⎥ =⎢       ⎥ +⎢     ⎥ =⎢       ⎥
     *     ⎣ 3-4i 7-8i⎦  ⎣ 1+2i⎦     ⎣ 2+2i⎦  ⎣ 72- 6i⎦  ⎣ 6+6i⎦  ⎣ 78    ⎦
     *
     * 5. cgemv/zgemv, notrans, beta=3+1i.
     *
     *     ⎡ 1+2i 3+4i⎤  ⎡ 3+1i⎤          ⎡ 1+0i⎤  ⎡-34- 8i⎤  ⎡ 3+1i⎤  ⎡-31- 7i⎤
     *  2i*⎢          ⎥ *⎢     ⎥ + (3+1i)*⎢     ⎥ =⎢       ⎥ +⎢     ⎥ =⎢       ⎥
     *     ⎣    0 7+8i⎦  ⎣ 1+2i⎦          ⎣ 2+2i⎦  ⎣-44-18i⎦  ⎣ 4+8i⎦  ⎣-40-10i⎦
     *
     * 6. cgemv/zgemv, trans, beta=3+1i.
     *
     *     ⎡ 1+2i    0⎤  ⎡ 3+1i⎤          ⎡ 1+0i⎤  ⎡-14+2i⎤  ⎡ 3+1i⎤  ⎡-11+3i⎤
     *  2i*⎢          ⎥ *⎢     ⎥ + (3+1i)*⎢     ⎥ =⎢      ⎥ +⎢     ⎥ =⎢      ⎥
     *     ⎣ 3+4i 7+8i⎦  ⎣ 1+2i⎦          ⎣ 2+2i⎦  ⎣-74-8i⎦  ⎣ 4+8i⎦  ⎣-70   ⎦
     *
     * 7. cgemv/zgemv, conjtrans, beta=3+1i.
     *
     *     ⎡ 1-2i    0⎤  ⎡ 3+1i⎤          ⎡ 1+0i⎤  ⎡ 10+10i⎤  ⎡ 3+1i⎤  ⎡13+11i⎤
     *  2i*⎢          ⎥ *⎢     ⎥ + (3+1i)*⎢     ⎥ =⎢       ⎥ +⎢     ⎥ =⎢      ⎥
     *     ⎣ 3-4i 7-8i⎦  ⎣ 1+2i⎦          ⎣ 2+2i⎦  ⎣  6+72i⎦  ⎣ 4+8i⎦  ⎣10+80i⎦
     *
     *
     * For binary (pattern) matrices, calculate:
     *
     * 1. sgemv/dgemv, notrans, beta=1.
     *
     *     ⎡ 1 0 1⎤  ⎡ 3⎤  ⎡ 1⎤  ⎡  8⎤  ⎡ 1⎤  ⎡  9⎤
     *   2*⎢ 1 1 0⎥ *⎢ 2⎥ +⎢ 0⎥ =⎢ 10⎥ +⎢ 0⎥ =⎢ 10⎥
     *     ⎣ 0 0 1⎦  ⎣ 1⎦  ⎣ 2⎦  ⎣  2⎦  ⎣ 2⎦  ⎣  4⎦
     *
     * 2. sgemv/dgemv, notrans, beta=3.
     *
     *     ⎡ 1 0 1⎤  ⎡ 3⎤     ⎡ 1⎤  ⎡  8⎤  ⎡ 3⎤  ⎡ 11⎤
     *   2*⎢ 1 1 0⎥ *⎢ 2⎥ + 3*⎢ 0⎥ =⎢ 10⎥ +⎢ 0⎥ =⎢ 10⎥
     *     ⎣ 0 0 1⎦  ⎣ 1⎦     ⎣ 2⎦  ⎣  2⎦  ⎣ 6⎦  ⎣  8⎦
     *
     * 3. sgemv/dgemv, trans, beta=3.
     *
     *     ⎡ 1 1 0⎤  ⎡ 3⎤     ⎡ 1⎤  ⎡ 10⎤  ⎡ 3⎤  ⎡ 13⎤
     *   2*⎢ 0 1 0⎥ *⎢ 2⎥ + 3*⎢ 0⎥ =⎢  4⎥ +⎢ 0⎥ =⎢  4⎥
     *     ⎣ 1 0 1⎦  ⎣ 1⎦     ⎣ 2⎦  ⎣  8⎦  ⎣ 6⎦  ⎣ 14⎦
     */

    /*
     * Real matrices
     */

    {
        struct mtxmatrix A;
        struct mtxvector x;
        struct mtxvector y;
        int num_rows = 3;
        int num_columns = 3;
        int num_nonzeros = 5;
        int Arowidx[] = {0, 0, 1, 1, 2};
        int Acolidx[] = {0, 2, 0, 1, 2};
        float Adata[] = {1.0f, 3.0f, 4.0f, 5.0f, 9.0f};
        float xdata[] = {3.0f, 2.0f, 1.0f};
        float ydata[] = {1.0f, 0.0f, 2.0f};
        err = mtxmatrix_init_coordinate_real_single(
            &A, num_rows, num_columns, num_nonzeros, Arowidx, Acolidx, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        err = mtxvector_init_array_real_single(&x, num_columns, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        {
            err = mtxvector_init_array_real_single(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            err = mtxmatrix_sgemv(mtx_notrans, 2.0f, &A, &x, 1.0f, &y);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
            TEST_ASSERT_EQ(mtx_field_real, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(3, y_->size);
            TEST_ASSERT_EQ(y_->data.real_single[0], 13.0f);
            TEST_ASSERT_EQ(y_->data.real_single[1], 44.0f);
            TEST_ASSERT_EQ(y_->data.real_single[2], 20.0f);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_array_real_single(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            err = mtxmatrix_sgemv(mtx_notrans, 2.0f, &A, &x, 3.0f, &y);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
            TEST_ASSERT_EQ(mtx_field_real, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(3, y_->size);
            TEST_ASSERT_EQ(y_->data.real_single[0], 15.0f);
            TEST_ASSERT_EQ(y_->data.real_single[1], 44.0f);
            TEST_ASSERT_EQ(y_->data.real_single[2], 24.0f);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_array_real_single(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            err = mtxmatrix_sgemv(mtx_trans, 2.0f, &A, &x, 3.0f, &y);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
            TEST_ASSERT_EQ(mtx_field_real, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(3, y_->size);
            TEST_ASSERT_EQ(y_->data.real_single[0], 25.0f);
            TEST_ASSERT_EQ(y_->data.real_single[1], 20.0f);
            TEST_ASSERT_EQ(y_->data.real_single[2], 42.0f);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_array_real_single(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            err = mtxmatrix_dgemv(mtx_notrans, 2.0, &A, &x, 3.0, &y);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
            TEST_ASSERT_EQ(mtx_field_real, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(3, y_->size);
            TEST_ASSERT_EQ(y_->data.real_single[0], 15.0f);
            TEST_ASSERT_EQ(y_->data.real_single[1], 44.0f);
            TEST_ASSERT_EQ(y_->data.real_single[2], 24.0f);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_array_real_single(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            err = mtxmatrix_dgemv(mtx_trans, 2.0, &A, &x, 3.0, &y);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
            TEST_ASSERT_EQ(mtx_field_real, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(3, y_->size);
            TEST_ASSERT_EQ(y_->data.real_single[0], 25.0f);
            TEST_ASSERT_EQ(y_->data.real_single[1], 20.0f);
            TEST_ASSERT_EQ(y_->data.real_single[2], 42.0f);
            mtxvector_free(&y);
        }
        mtxvector_free(&x);
        mtxmatrix_free(&A);
    }

    {
        struct mtxmatrix A;
        struct mtxvector x;
        struct mtxvector y;
        int num_rows = 3;
        int num_columns = 3;
        int num_nonzeros = 5;
        int Arowidx[] = {0, 0, 1, 1, 2};
        int Acolidx[] = {0, 2, 0, 1, 2};
        double Adata[] = {1.0, 3.0, 4.0, 5.0, 9.0};
        double xdata[] = {3.0, 2.0, 1.0};
        double ydata[] = {1.0, 0.0, 2.0};
        err = mtxmatrix_init_coordinate_real_double(
            &A, num_rows, num_columns, num_nonzeros, Arowidx, Acolidx, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        err = mtxvector_init_array_real_double(&x, num_columns, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        {
            err = mtxvector_init_array_real_double(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            err = mtxmatrix_sgemv(mtx_notrans, 2.0f, &A, &x, 1.0f, &y);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
            TEST_ASSERT_EQ(mtx_field_real, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(3, y_->size);
            TEST_ASSERT_EQ(y_->data.real_double[0], 13.0);
            TEST_ASSERT_EQ(y_->data.real_double[1], 44.0);
            TEST_ASSERT_EQ(y_->data.real_double[2], 20.0);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_array_real_double(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            err = mtxmatrix_sgemv(mtx_notrans, 2.0f, &A, &x, 3.0f, &y);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
            TEST_ASSERT_EQ(mtx_field_real, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(3, y_->size);
            TEST_ASSERT_EQ(y_->data.real_double[0], 15.0);
            TEST_ASSERT_EQ(y_->data.real_double[1], 44.0);
            TEST_ASSERT_EQ(y_->data.real_double[2], 24.0);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_array_real_double(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            err = mtxmatrix_sgemv(mtx_trans, 2.0f, &A, &x, 3.0f, &y);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
            TEST_ASSERT_EQ(mtx_field_real, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(3, y_->size);
            TEST_ASSERT_EQ(y_->data.real_double[0], 25.0);
            TEST_ASSERT_EQ(y_->data.real_double[1], 20.0);
            TEST_ASSERT_EQ(y_->data.real_double[2], 42.0);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_array_real_double(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            err = mtxmatrix_dgemv(mtx_notrans, 2.0, &A, &x, 3.0, &y);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
            TEST_ASSERT_EQ(mtx_field_real, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(3, y_->size);
            TEST_ASSERT_EQ(y_->data.real_double[0], 15.0);
            TEST_ASSERT_EQ(y_->data.real_double[1], 44.0);
            TEST_ASSERT_EQ(y_->data.real_double[2], 24.0);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_array_real_double(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            err = mtxmatrix_dgemv(mtx_trans, 2.0, &A, &x, 3.0, &y);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
            TEST_ASSERT_EQ(mtx_field_real, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(3, y_->size);
            TEST_ASSERT_EQ(y_->data.real_double[0], 25.0);
            TEST_ASSERT_EQ(y_->data.real_double[1], 20.0);
            TEST_ASSERT_EQ(y_->data.real_double[2], 42.0);
            mtxvector_free(&y);
        }
        mtxvector_free(&x);
        mtxmatrix_free(&A);
    }

    /*
     * Complex matrices
     */

    {
        struct mtxmatrix A;
        struct mtxvector x;
        struct mtxvector y;
        int num_rows = 2;
        int num_columns = 2;
        int num_nonzeros = 3;
        int Arowidx[] = {0, 0, 1};
        int Acolidx[] = {0, 1, 1};
        float Adata[][2] = {{1.0f,2.0f}, {3.0f,4.0f}, {7.0,8.0f}};
        float xdata[][2] = {{3.0f,1.0f}, {1.0f,2.0f}};
        float ydata[][2] = {{1.0f,0.0f}, {2.0f,2.0f}};
        err = mtxmatrix_init_coordinate_complex_single(
            &A, num_rows, num_columns, num_nonzeros, Arowidx, Acolidx, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        err = mtxvector_init_array_complex_single(&x, num_columns, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        {
            err = mtxvector_init_array_complex_single(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            err = mtxmatrix_sgemv(mtx_notrans, 2.0f, &A, &x, 1.0f, &y);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_single[0][0], -7.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[0][1], 34.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][0],-16.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1], 46.0f);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_array_complex_single(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            err = mtxmatrix_sgemv(mtx_notrans, 2.0f, &A, &x, 3.0f, &y);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_single[0][0], -5.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[0][1], 34.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][0],-12.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1], 50.0f);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_array_complex_single(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            err = mtxmatrix_sgemv(mtx_trans, 2.0f, &A, &x, 3.0f, &y);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_single[0][0],  5.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[0][1], 14.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][0], -2.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1], 80.0f);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_array_complex_single(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            err = mtxmatrix_sgemv(mtx_conjtrans, 2.0f, &A, &x, 3.0f, &y);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_single[0][0], 13.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[0][1],-10.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][0], 78.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1],  0.0f);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_array_complex_single(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            err = mtxmatrix_dgemv(mtx_notrans, 2.0, &A, &x, 3.0, &y);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_single[0][0], -5.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[0][1], 34.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][0],-12.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1], 50.0f);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_array_complex_single(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            err = mtxmatrix_dgemv(mtx_trans, 2.0, &A, &x, 3.0, &y);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_single[0][0],  5.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[0][1], 14.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][0], -2.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1], 80.0f);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_array_complex_single(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            err = mtxmatrix_dgemv(mtx_conjtrans, 2.0, &A, &x, 3.0, &y);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_single[0][0], 13.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[0][1],-10.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][0], 78.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1],  0.0f);
            mtxvector_free(&y);
        }

        float calpha[2] = {0.0f, 2.0f};
        float cbeta[2]  = {3.0f, 1.0f};
        {
            err = mtxvector_init_array_complex_single(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            err = mtxmatrix_cgemv(mtx_notrans, calpha, &A, &x, cbeta, &y);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_single[0][0],-31.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[0][1], -7.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][0],-40.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1],-10.0f);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_array_complex_single(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            err = mtxmatrix_cgemv(mtx_trans, calpha, &A, &x, cbeta, &y);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_single[0][0],-11.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[0][1],  3.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][0],-70.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1],  0.0f);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_array_complex_single(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            err = mtxmatrix_cgemv(mtx_conjtrans, calpha, &A, &x, cbeta, &y);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_single[0][0], 13.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[0][1], 11.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][0], 10.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1], 80.0f);
            mtxvector_free(&y);
        }

        double zalpha[2] = {0.0, 2.0};
        double zbeta[2]  = {3.0, 1.0};
        {
            err = mtxvector_init_array_complex_single(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            err = mtxmatrix_zgemv(mtx_notrans, zalpha, &A, &x, zbeta, &y);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_single[0][0], -31.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[0][1],  -7.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][0], -40.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1], -10.0f);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_array_complex_single(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            err = mtxmatrix_zgemv(mtx_trans, zalpha, &A, &x, zbeta, &y);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_single[0][0], -11.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[0][1],   3.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][0], -70.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1],   0.0f);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_array_complex_single(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            err = mtxmatrix_zgemv(mtx_conjtrans, zalpha, &A, &x, zbeta, &y);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_single[0][0], 13.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[0][1], 11.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][0], 10.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1], 80.0f);
            mtxvector_free(&y);
        }

        mtxvector_free(&x);
        mtxmatrix_free(&A);
    }

    {
        struct mtxmatrix A;
        struct mtxvector x;
        struct mtxvector y;
        int num_rows = 2;
        int num_columns = 2;
        int num_nonzeros = 3;
        int Arowidx[] = {0, 0, 1};
        int Acolidx[] = {0, 1, 1};
        double Adata[][2] = {{1.0,2.0}, {3.0,4.0}, {7.0,8.0}};
        double xdata[][2] = {{3.0,1.0}, {1.0,2.0}};
        double ydata[][2] = {{1.0,0.0}, {2.0,2.0}};
        err = mtxmatrix_init_coordinate_complex_double(
            &A, num_rows, num_columns, num_nonzeros, Arowidx, Acolidx, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        err = mtxvector_init_array_complex_double(&x, num_columns, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        {
            err = mtxvector_init_array_complex_double(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            err = mtxmatrix_sgemv(mtx_notrans, 2.0f, &A, &x, 1.0f, &y);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_double[0][0], -7.0);
            TEST_ASSERT_EQ(y_->data.complex_double[0][1], 34.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][0],-16.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][1], 46.0);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_array_complex_double(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            err = mtxmatrix_sgemv(mtx_notrans, 2.0f, &A, &x, 3.0f, &y);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_double[0][0],  -5.0);
            TEST_ASSERT_EQ(y_->data.complex_double[0][1],  34.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][0], -12.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][1],  50.0);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_array_complex_double(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            err = mtxmatrix_sgemv(mtx_trans, 2.0f, &A, &x, 3.0f, &y);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_double[0][0],  5.0);
            TEST_ASSERT_EQ(y_->data.complex_double[0][1], 14.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][0], -2.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][1], 80.0);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_array_complex_double(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            err = mtxmatrix_sgemv(mtx_conjtrans, 2.0f, &A, &x, 3.0f, &y);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_double[0][0], 13.0);
            TEST_ASSERT_EQ(y_->data.complex_double[0][1],-10.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][0], 78.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][1],  0.0);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_array_complex_double(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            err = mtxmatrix_dgemv(mtx_notrans, 2.0, &A, &x, 3.0, &y);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_double[0][0], -5.0);
            TEST_ASSERT_EQ(y_->data.complex_double[0][1], 34.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][0],-12.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][1], 50.0);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_array_complex_double(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            err = mtxmatrix_dgemv(mtx_trans, 2.0, &A, &x, 3.0, &y);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_double[0][0],  5.0);
            TEST_ASSERT_EQ(y_->data.complex_double[0][1], 14.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][0], -2.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][1], 80.0);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_array_complex_double(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            err = mtxmatrix_dgemv(mtx_conjtrans, 2.0, &A, &x, 3.0, &y);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_double[0][0], 13.0);
            TEST_ASSERT_EQ(y_->data.complex_double[0][1],-10.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][0], 78.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][1],  0.0);
            mtxvector_free(&y);
        }
        mtxvector_free(&x);
        mtxmatrix_free(&A);
    }

    /*
     * Integer matrices
     */

    {
        struct mtxmatrix A;
        struct mtxvector x;
        struct mtxvector y;
        int num_rows = 3;
        int num_columns = 3;
        int num_nonzeros = 5;
        int Arowidx[] = {0, 0, 1, 1, 2};
        int Acolidx[] = {0, 2, 0, 1, 2};
        int32_t Adata[] = {1, 3, 4, 5, 9};
        int32_t xdata[] = {3, 2, 1};
        int32_t ydata[] = {1, 0, 2};
        err = mtxmatrix_init_coordinate_integer_single(
            &A, num_rows, num_columns, num_nonzeros, Arowidx, Acolidx, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        err = mtxvector_init_array_integer_single(&x, num_columns, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        {
            err = mtxvector_init_array_integer_single(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            err = mtxmatrix_sgemv(mtx_notrans, 2.0f, &A, &x, 3.0f, &y);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
            TEST_ASSERT_EQ(mtx_field_integer, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(3, y_->size);
            TEST_ASSERT_EQ(y_->data.integer_single[0], 15);
            TEST_ASSERT_EQ(y_->data.integer_single[1], 44);
            TEST_ASSERT_EQ(y_->data.integer_single[2], 24);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_array_integer_single(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            err = mtxmatrix_sgemv(mtx_trans, 2.0f, &A, &x, 3.0f, &y);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
            TEST_ASSERT_EQ(mtx_field_integer, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(3, y_->size);
            TEST_ASSERT_EQ(y_->data.integer_single[0], 25);
            TEST_ASSERT_EQ(y_->data.integer_single[1], 20);
            TEST_ASSERT_EQ(y_->data.integer_single[2], 42);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_array_integer_single(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            err = mtxmatrix_dgemv(mtx_notrans, 2.0, &A, &x, 3.0, &y);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
            TEST_ASSERT_EQ(mtx_field_integer, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(3, y_->size);
            TEST_ASSERT_EQ(y_->data.integer_single[0], 15);
            TEST_ASSERT_EQ(y_->data.integer_single[1], 44);
            TEST_ASSERT_EQ(y_->data.integer_single[2], 24);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_array_integer_single(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            err = mtxmatrix_dgemv(mtx_trans, 2.0, &A, &x, 3.0, &y);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
            TEST_ASSERT_EQ(mtx_field_integer, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(3, y_->size);
            TEST_ASSERT_EQ(y_->data.integer_single[0], 25);
            TEST_ASSERT_EQ(y_->data.integer_single[1], 20);
            TEST_ASSERT_EQ(y_->data.integer_single[2], 42);
            mtxvector_free(&y);
        }
        mtxvector_free(&x);
        mtxmatrix_free(&A);
    }

    {
        struct mtxmatrix A;
        struct mtxvector x;
        struct mtxvector y;
        int num_rows = 3;
        int num_columns = 3;
        int num_nonzeros = 5;
        int Arowidx[] = {0, 0, 1, 1, 2};
        int Acolidx[] = {0, 2, 0, 1, 2};
        int64_t Adata[] = {1, 3, 4, 5, 9};
        int64_t xdata[] = {3, 2, 1};
        int64_t ydata[] = {1, 0, 2};
        err = mtxmatrix_init_coordinate_integer_double(
            &A, num_rows, num_columns, num_nonzeros, Arowidx, Acolidx, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        err = mtxvector_init_array_integer_double(&x, num_columns, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        {
            err = mtxvector_init_array_integer_double(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            err = mtxmatrix_sgemv(mtx_notrans, 2.0f, &A, &x, 3.0f, &y);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
            TEST_ASSERT_EQ(mtx_field_integer, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(3, y_->size);
            TEST_ASSERT_EQ(y_->data.integer_double[0], 15);
            TEST_ASSERT_EQ(y_->data.integer_double[1], 44);
            TEST_ASSERT_EQ(y_->data.integer_double[2], 24);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_array_integer_double(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            err = mtxmatrix_sgemv(mtx_trans, 2.0f, &A, &x, 3.0f, &y);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
            TEST_ASSERT_EQ(mtx_field_integer, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(3, y_->size);
            TEST_ASSERT_EQ(y_->data.integer_double[0], 25);
            TEST_ASSERT_EQ(y_->data.integer_double[1], 20);
            TEST_ASSERT_EQ(y_->data.integer_double[2], 42);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_array_integer_double(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            err = mtxmatrix_dgemv(mtx_notrans, 2.0, &A, &x, 3.0, &y);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
            TEST_ASSERT_EQ(mtx_field_integer, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(3, y_->size);
            TEST_ASSERT_EQ(y_->data.integer_double[0], 15);
            TEST_ASSERT_EQ(y_->data.integer_double[1], 44);
            TEST_ASSERT_EQ(y_->data.integer_double[2], 24);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_array_integer_double(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            err = mtxmatrix_dgemv(mtx_trans, 2.0, &A, &x, 3.0, &y);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
            TEST_ASSERT_EQ(mtx_field_integer, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(3, y_->size);
            TEST_ASSERT_EQ(y_->data.integer_double[0], 25);
            TEST_ASSERT_EQ(y_->data.integer_double[1], 20);
            TEST_ASSERT_EQ(y_->data.integer_double[2], 42);
            mtxvector_free(&y);
        }
        mtxvector_free(&x);
        mtxmatrix_free(&A);
    }

    /*
     * Binary (pattern) matrices
     */

    {
        struct mtxmatrix A;
        struct mtxvector x;
        struct mtxvector y;
        int num_rows = 3;
        int num_columns = 3;
        int num_nonzeros = 5;
        int Arowidx[] = {0, 0, 1, 1, 2};
        int Acolidx[] = {0, 2, 0, 1, 2};
        err = mtxmatrix_init_coordinate_pattern(
            &A, num_rows, num_columns, num_nonzeros, Arowidx, Acolidx);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        {
            float xdata[] = {3, 2, 1}, ydata[] = {1, 0, 2};
            err = mtxvector_init_array_real_single(&x, num_columns, xdata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            err = mtxvector_init_array_real_single(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            err = mtxmatrix_sgemv(mtx_notrans, 2.0f, &A, &x, 3.0f, &y);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
            TEST_ASSERT_EQ(mtx_field_real, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(3, y_->size);
            TEST_ASSERT_EQ(y_->data.real_single[0], 11.0f);
            TEST_ASSERT_EQ(y_->data.real_single[1], 10.0f);
            TEST_ASSERT_EQ(y_->data.real_single[2],  8.0f);
            mtxvector_free(&y);
            mtxvector_free(&x);
        }
        {
            double xdata[] = {3, 2, 1}, ydata[] = {1, 0, 2};
            err = mtxvector_init_array_real_double(&x, num_columns, xdata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            err = mtxvector_init_array_real_double(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            err = mtxmatrix_sgemv(mtx_notrans, 2.0f, &A, &x, 3.0f, &y);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
            TEST_ASSERT_EQ(mtx_field_real, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(3, y_->size);
            TEST_ASSERT_EQ(y_->data.real_double[0], 11.0);
            TEST_ASSERT_EQ(y_->data.real_double[1], 10.0);
            TEST_ASSERT_EQ(y_->data.real_double[2],  8.0);
            mtxvector_free(&y);
            mtxvector_free(&x);
        }
        {
            int32_t xdata[] = {3, 2, 1}, ydata[] = {1, 0, 2};
            err = mtxvector_init_array_integer_single(&x, num_columns, xdata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            err = mtxvector_init_array_integer_single(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            err = mtxmatrix_sgemv(mtx_notrans, 2.0f, &A, &x, 3.0f, &y);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
            TEST_ASSERT_EQ(mtx_field_integer, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(3, y_->size);
            TEST_ASSERT_EQ(y_->data.integer_single[0], 11);
            TEST_ASSERT_EQ(y_->data.integer_single[1], 10);
            TEST_ASSERT_EQ(y_->data.integer_single[2],  8);
            mtxvector_free(&y);
            mtxvector_free(&x);
        }
        {
            int32_t xdata[] = {3, 2, 1}, ydata[] = {1, 0, 2};
            err = mtxvector_init_array_integer_single(&x, num_columns, xdata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            err = mtxvector_init_array_integer_single(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            err = mtxmatrix_sgemv(mtx_trans, 2.0f, &A, &x, 3.0f, &y);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
            TEST_ASSERT_EQ(mtx_field_integer, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(3, y_->size);
            TEST_ASSERT_EQ(y_->data.integer_single[0], 13);
            TEST_ASSERT_EQ(y_->data.integer_single[1],  4);
            TEST_ASSERT_EQ(y_->data.integer_single[2], 14);
            mtxvector_free(&y);
            mtxvector_free(&x);
        }
        {
            int32_t xdata[] = {3, 2, 1}, ydata[] = {1, 0, 2};
            err = mtxvector_init_array_integer_single(&x, num_columns, xdata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            err = mtxvector_init_array_integer_single(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            err = mtxmatrix_dgemv(mtx_notrans, 2.0, &A, &x, 3.0, &y);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
            TEST_ASSERT_EQ(mtx_field_integer, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(3, y_->size);
            TEST_ASSERT_EQ(y_->data.integer_single[0], 11);
            TEST_ASSERT_EQ(y_->data.integer_single[1], 10);
            TEST_ASSERT_EQ(y_->data.integer_single[2],  8);
            mtxvector_free(&y);
            mtxvector_free(&x);
        }
        {
            int32_t xdata[] = {3, 2, 1}, ydata[] = {1, 0, 2};
            err = mtxvector_init_array_integer_single(&x, num_columns, xdata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            err = mtxvector_init_array_integer_single(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            err = mtxmatrix_dgemv(mtx_trans, 2.0, &A, &x, 3.0, &y);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
            TEST_ASSERT_EQ(mtx_field_integer, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(3, y_->size);
            TEST_ASSERT_EQ(y_->data.integer_single[0], 13);
            TEST_ASSERT_EQ(y_->data.integer_single[1],  4);
            TEST_ASSERT_EQ(y_->data.integer_single[2], 14);
            mtxvector_free(&y);
            mtxvector_free(&x);
        }
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
    TEST_RUN(test_mtxmatrix_gemv_array);
    TEST_RUN(test_mtxmatrix_gemv_coordinate);
    TEST_SUITE_END();
    return (TEST_SUITE_STATUS == TEST_SUCCESS) ?
        EXIT_SUCCESS : EXIT_FAILURE;
}
