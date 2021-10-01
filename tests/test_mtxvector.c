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
 * Unit tests for Matrix Market files.
 */

#include "test.h"

#include <libmtx/error.h>
#include <libmtx/vector/vector.h>
#include <libmtx/mtxfile/mtxfile.h>

#include <errno.h>
#include <unistd.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * `test_mtxvector_from_mtxfile()' tests converting Matrix Market
 *  files to vectors.
 */
int test_mtxvector_from_mtxfile(void)
{
    int err;

    /*
     * Array formats
     */

    {
        int num_rows = 3;
        const float mtxdata[] = {3.0f, 4.0f, 5.0f};
        int64_t num_nonzeros = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile mtxfile;
        err = mtxfile_init_vector_array_real_single(&mtxfile, num_rows, mtxdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        struct mtxvector x;
        err = mtxvector_from_mtxfile(&x, &mtxfile, mtxvector_auto);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(mtxvector_array, x.type);
        const struct mtxvector_array * x_ = &x.storage.array;
        TEST_ASSERT_EQ(mtx_field_real, x_->field);
        TEST_ASSERT_EQ(mtx_single, x_->precision);
        TEST_ASSERT_EQ(3, x_->size);
        TEST_ASSERT_EQ(x_->data.real_single[0], 3.0f);
        TEST_ASSERT_EQ(x_->data.real_single[1], 4.0f);
        TEST_ASSERT_EQ(x_->data.real_single[2], 5.0f);
        mtxvector_free(&x);
        mtxfile_free(&mtxfile);
    }

    {
        int num_rows = 3;
        const double mtxdata[] = {3.0, 4.0, 5.0};
        int64_t num_nonzeros = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile mtxfile;
        err = mtxfile_init_vector_array_real_double(&mtxfile, num_rows, mtxdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        struct mtxvector x;
        err = mtxvector_from_mtxfile(&x, &mtxfile, mtxvector_auto);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(mtxvector_array, x.type);
        const struct mtxvector_array * x_ = &x.storage.array;
        TEST_ASSERT_EQ(mtx_field_real, x_->field);
        TEST_ASSERT_EQ(mtx_double, x_->precision);
        TEST_ASSERT_EQ(3, x_->size);
        TEST_ASSERT_EQ(x_->data.real_double[0], 3.0);
        TEST_ASSERT_EQ(x_->data.real_double[1], 4.0);
        TEST_ASSERT_EQ(x_->data.real_double[2], 5.0);
        mtxvector_free(&x);
        mtxfile_free(&mtxfile);
    }

    {
        int num_rows = 3;
        const float mtxdata[][2] = {{3.0f, 4.0f}, {5.0f, 6.0f}, {7.0f, 8.0f}};
        int64_t num_nonzeros = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile mtxfile;
        err = mtxfile_init_vector_array_complex_single(&mtxfile, num_rows, mtxdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        struct mtxvector x;
        err = mtxvector_from_mtxfile(&x, &mtxfile, mtxvector_auto);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(mtxvector_array, x.type);
        const struct mtxvector_array * x_ = &x.storage.array;
        TEST_ASSERT_EQ(mtx_field_complex, x_->field);
        TEST_ASSERT_EQ(mtx_single, x_->precision);
        TEST_ASSERT_EQ(3, x_->size);
        TEST_ASSERT_EQ(x_->data.complex_single[0][0], 3.0f);
        TEST_ASSERT_EQ(x_->data.complex_single[0][1], 4.0f);
        TEST_ASSERT_EQ(x_->data.complex_single[1][0], 5.0f);
        TEST_ASSERT_EQ(x_->data.complex_single[1][1], 6.0f);
        TEST_ASSERT_EQ(x_->data.complex_single[2][0], 7.0f);
        TEST_ASSERT_EQ(x_->data.complex_single[2][1], 8.0f);
        mtxvector_free(&x);
        mtxfile_free(&mtxfile);
    }

    {
        int num_rows = 3;
        const double mtxdata[][2] = {{3.0, 4.0}, {5.0, 6.0}, {7.0, 8.0}};
        int64_t num_nonzeros = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile mtxfile;
        err = mtxfile_init_vector_array_complex_double(&mtxfile, num_rows, mtxdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        struct mtxvector x;
        err = mtxvector_from_mtxfile(&x, &mtxfile, mtxvector_auto);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(mtxvector_array, x.type);
        const struct mtxvector_array * x_ = &x.storage.array;
        TEST_ASSERT_EQ(mtx_field_complex, x_->field);
        TEST_ASSERT_EQ(mtx_double, x_->precision);
        TEST_ASSERT_EQ(3, x_->size);
        TEST_ASSERT_EQ(x_->data.complex_double[0][0], 3.0f);
        TEST_ASSERT_EQ(x_->data.complex_double[0][1], 4.0f);
        TEST_ASSERT_EQ(x_->data.complex_double[1][0], 5.0f);
        TEST_ASSERT_EQ(x_->data.complex_double[1][1], 6.0f);
        TEST_ASSERT_EQ(x_->data.complex_double[2][0], 7.0f);
        TEST_ASSERT_EQ(x_->data.complex_double[2][1], 8.0f);
        mtxvector_free(&x);
        mtxfile_free(&mtxfile);
    }

    {
        int num_rows = 3;
        const int32_t mtxdata[] = {3, 4, 5};
        int64_t num_nonzeros = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile mtxfile;
        err = mtxfile_init_vector_array_integer_single(&mtxfile, num_rows, mtxdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        struct mtxvector x;
        err = mtxvector_from_mtxfile(&x, &mtxfile, mtxvector_auto);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(mtxvector_array, x.type);
        const struct mtxvector_array * x_ = &x.storage.array;
        TEST_ASSERT_EQ(mtx_field_integer, x_->field);
        TEST_ASSERT_EQ(mtx_single, x_->precision);
        TEST_ASSERT_EQ(3, x_->size);
        TEST_ASSERT_EQ(x_->data.integer_single[0], 3);
        TEST_ASSERT_EQ(x_->data.integer_single[1], 4);
        TEST_ASSERT_EQ(x_->data.integer_single[2], 5);
        mtxvector_free(&x);
        mtxfile_free(&mtxfile);
    }

    {
        int num_rows = 3;
        const int64_t mtxdata[] = {3, 4, 5};
        int64_t num_nonzeros = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile mtxfile;
        err = mtxfile_init_vector_array_integer_double(&mtxfile, num_rows, mtxdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        struct mtxvector x;
        err = mtxvector_from_mtxfile(&x, &mtxfile, mtxvector_auto);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(mtxvector_array, x.type);
        const struct mtxvector_array * x_ = &x.storage.array;
        TEST_ASSERT_EQ(mtx_field_integer, x_->field);
        TEST_ASSERT_EQ(mtx_double, x_->precision);
        TEST_ASSERT_EQ(3, x_->size);
        TEST_ASSERT_EQ(x_->data.integer_double[0], 3);
        TEST_ASSERT_EQ(x_->data.integer_double[1], 4);
        TEST_ASSERT_EQ(x_->data.integer_double[2], 5);
        mtxvector_free(&x);
        mtxfile_free(&mtxfile);
    }

    /*
     * Coordinate formats
     */

    {
        int num_rows = 4;
        struct mtxfile_vector_coordinate_real_single mtxdata[] = {
            {1, 1.0f}, {2, 2.0f}, {4, 4.0f}};
        int64_t num_nonzeros = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile mtxfile;
        err = mtxfile_init_vector_coordinate_real_single(
            &mtxfile, num_rows, num_nonzeros, mtxdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        struct mtxvector x;
        err = mtxvector_from_mtxfile(&x, &mtxfile, mtxvector_auto);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(mtxvector_coordinate, x.type);
        const struct mtxvector_coordinate * x_ = &x.storage.coordinate;
        TEST_ASSERT_EQ(mtx_field_real, x_->field);
        TEST_ASSERT_EQ(mtx_single, x_->precision);
        TEST_ASSERT_EQ(4, x_->size);
        TEST_ASSERT_EQ(3, x_->num_nonzeros);
        TEST_ASSERT_EQ(x_->indices[0], 1);
        TEST_ASSERT_EQ(x_->indices[1], 2);
        TEST_ASSERT_EQ(x_->indices[2], 4);
        TEST_ASSERT_EQ(x_->data.real_single[0], 1.0f);
        TEST_ASSERT_EQ(x_->data.real_single[1], 2.0f);
        TEST_ASSERT_EQ(x_->data.real_single[2], 4.0f);
        mtxvector_free(&x);
        mtxfile_free(&mtxfile);
    }

    {
        int num_rows = 4;
        struct mtxfile_vector_coordinate_real_double mtxdata[] = {
            {1, 1.0}, {2, 2.0}, {4, 4.0}};
        int64_t num_nonzeros = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile mtxfile;
        err = mtxfile_init_vector_coordinate_real_double(
            &mtxfile, num_rows, num_nonzeros, mtxdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        struct mtxvector x;
        err = mtxvector_from_mtxfile(&x, &mtxfile, mtxvector_auto);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(mtxvector_coordinate, x.type);
        const struct mtxvector_coordinate * x_ = &x.storage.coordinate;
        TEST_ASSERT_EQ(mtx_field_real, x_->field);
        TEST_ASSERT_EQ(mtx_double, x_->precision);
        TEST_ASSERT_EQ(4, x_->size);
        TEST_ASSERT_EQ(3, x_->num_nonzeros);
        TEST_ASSERT_EQ(x_->indices[0], 1);
        TEST_ASSERT_EQ(x_->indices[1], 2);
        TEST_ASSERT_EQ(x_->indices[2], 4);
        TEST_ASSERT_EQ(x_->data.real_double[0], 1.0);
        TEST_ASSERT_EQ(x_->data.real_double[1], 2.0);
        TEST_ASSERT_EQ(x_->data.real_double[2], 4.0);
        mtxvector_free(&x);
        mtxfile_free(&mtxfile);
    }

    {
        int num_rows = 4;
        struct mtxfile_vector_coordinate_complex_single mtxdata[] = {
            {1,1.0f,-1.0f}, {2,2.0f,-2.0f}, {4,4.0f,-4.0f}};
        int64_t num_nonzeros = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile mtxfile;
        err = mtxfile_init_vector_coordinate_complex_single(
            &mtxfile, num_rows, num_nonzeros, mtxdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        struct mtxvector x;
        err = mtxvector_from_mtxfile(&x, &mtxfile, mtxvector_auto);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(mtxvector_coordinate, x.type);
        const struct mtxvector_coordinate * x_ = &x.storage.coordinate;
        TEST_ASSERT_EQ(mtx_field_complex, x_->field);
        TEST_ASSERT_EQ(mtx_single, x_->precision);
        TEST_ASSERT_EQ(4, x_->size);
        TEST_ASSERT_EQ(3, x_->num_nonzeros);
        TEST_ASSERT_EQ(x_->indices[0], 1);
        TEST_ASSERT_EQ(x_->indices[1], 2);
        TEST_ASSERT_EQ(x_->indices[2], 4);
        TEST_ASSERT_EQ(x_->data.complex_single[0][0], 1.0f);
        TEST_ASSERT_EQ(x_->data.complex_single[0][1], -1.0f);
        TEST_ASSERT_EQ(x_->data.complex_single[1][0], 2.0f);
        TEST_ASSERT_EQ(x_->data.complex_single[1][1], -2.0f);
        TEST_ASSERT_EQ(x_->data.complex_single[2][0], 4.0f);
        TEST_ASSERT_EQ(x_->data.complex_single[2][1], -4.0f);
        mtxvector_free(&x);
        mtxfile_free(&mtxfile);
    }

    {
        int num_rows = 4;
        struct mtxfile_vector_coordinate_complex_double mtxdata[] = {
            {1,1.0,-1.0}, {2,2.0,-2.0}, {4,4.0,-4.0}};
        int64_t num_nonzeros = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile mtxfile;
        err = mtxfile_init_vector_coordinate_complex_double(
            &mtxfile, num_rows, num_nonzeros, mtxdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        struct mtxvector x;
        err = mtxvector_from_mtxfile(&x, &mtxfile, mtxvector_auto);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(mtxvector_coordinate, x.type);
        const struct mtxvector_coordinate * x_ = &x.storage.coordinate;
        TEST_ASSERT_EQ(mtx_field_complex, x_->field);
        TEST_ASSERT_EQ(mtx_double, x_->precision);
        TEST_ASSERT_EQ(4, x_->size);
        TEST_ASSERT_EQ(3, x_->num_nonzeros);
        TEST_ASSERT_EQ(x_->indices[0], 1);
        TEST_ASSERT_EQ(x_->indices[1], 2);
        TEST_ASSERT_EQ(x_->indices[2], 4);
        TEST_ASSERT_EQ(x_->data.complex_double[0][0], 1.0);
        TEST_ASSERT_EQ(x_->data.complex_double[0][1], -1.0);
        TEST_ASSERT_EQ(x_->data.complex_double[1][0], 2.0);
        TEST_ASSERT_EQ(x_->data.complex_double[1][1], -2.0);
        TEST_ASSERT_EQ(x_->data.complex_double[2][0], 4.0);
        TEST_ASSERT_EQ(x_->data.complex_double[2][1], -4.0);
        mtxvector_free(&x);
        mtxfile_free(&mtxfile);
    }

    {
        int num_rows = 4;
        struct mtxfile_vector_coordinate_integer_single mtxdata[] = {
            {1, 1}, {2, 2}, {4, 4}};
        int64_t num_nonzeros = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile mtxfile;
        err = mtxfile_init_vector_coordinate_integer_single(
            &mtxfile, num_rows, num_nonzeros, mtxdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        struct mtxvector x;
        err = mtxvector_from_mtxfile(&x, &mtxfile, mtxvector_auto);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(mtxvector_coordinate, x.type);
        const struct mtxvector_coordinate * x_ = &x.storage.coordinate;
        TEST_ASSERT_EQ(mtx_field_integer, x_->field);
        TEST_ASSERT_EQ(mtx_single, x_->precision);
        TEST_ASSERT_EQ(4, x_->size);
        TEST_ASSERT_EQ(3, x_->num_nonzeros);
        TEST_ASSERT_EQ(x_->indices[0], 1);
        TEST_ASSERT_EQ(x_->indices[1], 2);
        TEST_ASSERT_EQ(x_->indices[2], 4);
        TEST_ASSERT_EQ(x_->data.integer_single[0], 1);
        TEST_ASSERT_EQ(x_->data.integer_single[1], 2);
        TEST_ASSERT_EQ(x_->data.integer_single[2], 4);
        mtxvector_free(&x);
        mtxfile_free(&mtxfile);
    }

    {
        int num_rows = 4;
        struct mtxfile_vector_coordinate_integer_double mtxdata[] = {
            {1,1}, {2,2}, {4,4}};
        int64_t num_nonzeros = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile mtxfile;
        err = mtxfile_init_vector_coordinate_integer_double(
            &mtxfile, num_rows, num_nonzeros, mtxdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        struct mtxvector x;
        err = mtxvector_from_mtxfile(&x, &mtxfile, mtxvector_auto);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(mtxvector_coordinate, x.type);
        const struct mtxvector_coordinate * x_ = &x.storage.coordinate;
        TEST_ASSERT_EQ(mtx_field_integer, x_->field);
        TEST_ASSERT_EQ(mtx_double, x_->precision);
        TEST_ASSERT_EQ(4, x_->size);
        TEST_ASSERT_EQ(3, x_->num_nonzeros);
        TEST_ASSERT_EQ(x_->indices[0], 1);
        TEST_ASSERT_EQ(x_->indices[1], 2);
        TEST_ASSERT_EQ(x_->indices[2], 4);
        TEST_ASSERT_EQ(x_->data.integer_double[0], 1);
        TEST_ASSERT_EQ(x_->data.integer_double[1], 2);
        TEST_ASSERT_EQ(x_->data.integer_double[2], 4);
        mtxvector_free(&x);
        mtxfile_free(&mtxfile);
    }

    {
        int num_rows = 4;
        struct mtxfile_vector_coordinate_pattern mtxdata[] = {{1}, {2}, {4}};
        int64_t num_nonzeros = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile mtxfile;
        err = mtxfile_init_vector_coordinate_pattern(
            &mtxfile, num_rows, num_nonzeros, mtxdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        struct mtxvector x;
        err = mtxvector_from_mtxfile(&x, &mtxfile, mtxvector_auto);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(mtxvector_coordinate, x.type);
        const struct mtxvector_coordinate * x_ = &x.storage.coordinate;
        TEST_ASSERT_EQ(mtx_field_pattern, x_->field);
        TEST_ASSERT_EQ(4, x_->size);
        TEST_ASSERT_EQ(3, x_->num_nonzeros);
        TEST_ASSERT_EQ(x_->indices[0], 1);
        TEST_ASSERT_EQ(x_->indices[1], 2);
        TEST_ASSERT_EQ(x_->indices[2], 4);
        mtxvector_free(&x);
        mtxfile_free(&mtxfile);
    }
    return TEST_SUCCESS;
}

/**
 * `test_mtxvector_dot()' tests computing the dot products of pairs of
 * vectors.
 */
int test_mtxvector_dot(void)
{
    int err;

    /*
     * Array formats
     */

    {
        struct mtxvector x;
        float xdata[] = {1.0f, 1.0f, 1.0f, 2.0f, 3.0f};
        int xsize = sizeof(xdata) / sizeof(*xdata);
        err = mtxvector_init_array_real_single(&x, xsize, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        struct mtxvector y;
        float ydata[] = {3.0f, 2.0f, 1.0f, 0.0f, 1.0f};
        int ysize = sizeof(ydata) / sizeof(*ydata);
        err = mtxvector_init_array_real_single(&y, ysize, ydata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        float sdot;
        err = mtxvector_sdot(&x, &y, &sdot);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0f, sdot);
        double ddot;
        err = mtxvector_ddot(&x, &y, &ddot);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0, ddot);
        float cdotu[2];
        err = mtxvector_cdotu(&x, &y, &cdotu);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0f, cdotu[0]); TEST_ASSERT_EQ(0.0f, cdotu[1]);
        float cdotc[2];
        err = mtxvector_cdotc(&x, &y, &cdotc);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0f, cdotc[0]); TEST_ASSERT_EQ(0.0f, cdotc[1]);
        double zdotu[2];
        err = mtxvector_zdotu(&x, &y, &zdotu);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0, zdotu[0]); TEST_ASSERT_EQ(0.0, zdotu[1]);
        double zdotc[2];
        err = mtxvector_zdotc(&x, &y, &zdotc);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0, zdotc[0]); TEST_ASSERT_EQ(0.0, zdotc[1]);
        mtxvector_free(&y);
        mtxvector_free(&x);
    }

    {
        struct mtxvector x;
        double xdata[] = {1.0, 1.0, 1.0, 2.0, 3.0};
        int xsize = sizeof(xdata) / sizeof(*xdata);
        err = mtxvector_init_array_real_double(&x, xsize, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        struct mtxvector y;
        double ydata[] = {3.0, 2.0, 1.0, 0.0, 1.0};
        int ysize = sizeof(ydata) / sizeof(*ydata);
        err = mtxvector_init_array_real_double(&y, ysize, ydata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        float sdot;
        err = mtxvector_sdot(&x, &y, &sdot);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0f, sdot);
        double ddot;
        err = mtxvector_ddot(&x, &y, &ddot);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0, ddot);
        float cdotu[2];
        err = mtxvector_cdotu(&x, &y, &cdotu);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0f, cdotu[0]); TEST_ASSERT_EQ(0.0f, cdotu[1]);
        float cdotc[2];
        err = mtxvector_cdotc(&x, &y, &cdotc);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0f, cdotc[0]); TEST_ASSERT_EQ(0.0f, cdotc[1]);
        double zdotu[2];
        err = mtxvector_zdotu(&x, &y, &zdotu);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0, zdotu[0]); TEST_ASSERT_EQ(0.0, zdotu[1]);
        double zdotc[2];
        err = mtxvector_zdotc(&x, &y, &zdotc);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0, zdotc[0]); TEST_ASSERT_EQ(0.0, zdotc[1]);
        mtxvector_free(&y);
        mtxvector_free(&x);
    }

    {
        struct mtxvector x;
        float xdata[][2] = {{1.0f, 1.0f}, {1.0f, 2.0f}, {3.0f, 0.0f}};
        int xsize = sizeof(xdata) / sizeof(*xdata);
        err = mtxvector_init_array_complex_single(&x, xsize, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        struct mtxvector y;
        float ydata[][2] = {{3.0f, 2.0f}, {1.0f, 0.0f}, {1.0f, 0.0f}};
        int ysize = sizeof(ydata) / sizeof(*ydata);
        err = mtxvector_init_array_complex_single(&y, ysize, ydata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        float sdot;
        err = mtxvector_sdot(&x, &y, &sdot);
        TEST_ASSERT_EQ_MSG(MTX_ERR_INVALID_FIELD, err, "%s", mtx_strerror(err));
        double ddot;
        err = mtxvector_ddot(&x, &y, &ddot);
        TEST_ASSERT_EQ_MSG(MTX_ERR_INVALID_FIELD, err, "%s", mtx_strerror(err));
        float cdotu[2];
        err = mtxvector_cdotu(&x, &y, &cdotu);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(5.0f, cdotu[0]); TEST_ASSERT_EQ(7.0f, cdotu[1]);
        float cdotc[2];
        err = mtxvector_cdotc(&x, &y, &cdotc);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0f, cdotc[0]); TEST_ASSERT_EQ(-3.0f, cdotc[1]);
        double zdotu[2];
        err = mtxvector_zdotu(&x, &y, &zdotu);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(5.0, zdotu[0]); TEST_ASSERT_EQ(7.0, zdotu[1]);
        double zdotc[2];
        err = mtxvector_zdotc(&x, &y, &zdotc);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0, zdotc[0]); TEST_ASSERT_EQ(-3.0, zdotc[1]);
        mtxvector_free(&y);
        mtxvector_free(&x);
    }

    {
        struct mtxvector x;
        double xdata[][2] = {{1.0, 1.0}, {1.0, 2.0}, {3.0, 0.0}};
        int xsize = sizeof(xdata) / sizeof(*xdata);
        err = mtxvector_init_array_complex_double(&x, xsize, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        struct mtxvector y;
        double ydata[][2] = {{3.0, 2.0}, {1.0, 0.0}, {1.0, 0.0}};
        int ysize = sizeof(ydata) / sizeof(*ydata);
        err = mtxvector_init_array_complex_double(&y, ysize, ydata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        float sdot;
        err = mtxvector_sdot(&x, &y, &sdot);
        TEST_ASSERT_EQ_MSG(MTX_ERR_INVALID_FIELD, err, "%s", mtx_strerror(err));
        double ddot;
        err = mtxvector_ddot(&x, &y, &ddot);
        TEST_ASSERT_EQ_MSG(MTX_ERR_INVALID_FIELD, err, "%s", mtx_strerror(err));
        float cdotu[2];
        err = mtxvector_cdotu(&x, &y, &cdotu);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(5.0f, cdotu[0]); TEST_ASSERT_EQ(7.0f, cdotu[1]);
        float cdotc[2];
        err = mtxvector_cdotc(&x, &y, &cdotc);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0f, cdotc[0]); TEST_ASSERT_EQ(-3.0f, cdotc[1]);
        double zdotu[2];
        err = mtxvector_zdotu(&x, &y, &zdotu);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(5.0, zdotu[0]); TEST_ASSERT_EQ(7.0, zdotu[1]);
        double zdotc[2];
        err = mtxvector_zdotc(&x, &y, &zdotc);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0, zdotc[0]); TEST_ASSERT_EQ(-3.0, zdotc[1]);
        mtxvector_free(&y);
        mtxvector_free(&x);
    }

    {
        struct mtxvector x;
        int32_t xdata[] = {1, 1, 1, 2, 3};
        int xsize = sizeof(xdata) / sizeof(*xdata);
        err = mtxvector_init_array_integer_single(&x, xsize, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        struct mtxvector y;
        int32_t ydata[] = {3, 2, 1, 0, 1};
        int ysize = sizeof(ydata) / sizeof(*ydata);
        err = mtxvector_init_array_integer_single(&y, ysize, ydata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        float sdot;
        err = mtxvector_sdot(&x, &y, &sdot);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0f, sdot);
        double ddot;
        err = mtxvector_ddot(&x, &y, &ddot);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0, ddot);
        float cdotu[2];
        err = mtxvector_cdotu(&x, &y, &cdotu);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0f, cdotu[0]); TEST_ASSERT_EQ(0.0f, cdotu[1]);
        float cdotc[2];
        err = mtxvector_cdotc(&x, &y, &cdotc);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0f, cdotc[0]); TEST_ASSERT_EQ(0.0f, cdotc[1]);
        double zdotu[2];
        err = mtxvector_zdotu(&x, &y, &zdotu);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0, zdotu[0]); TEST_ASSERT_EQ(0.0, zdotu[1]);
        double zdotc[2];
        err = mtxvector_zdotc(&x, &y, &zdotc);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0, zdotc[0]); TEST_ASSERT_EQ(0.0, zdotc[1]);
        mtxvector_free(&y);
        mtxvector_free(&x);
    }

    {
        struct mtxvector x;
        int64_t xdata[] = {1, 1, 1, 2, 3};
        int xsize = sizeof(xdata) / sizeof(*xdata);
        err = mtxvector_init_array_integer_double(&x, xsize, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        struct mtxvector y;
        int64_t ydata[] = {3, 2, 1, 0, 1};
        int ysize = sizeof(ydata) / sizeof(*ydata);
        err = mtxvector_init_array_integer_double(&y, ysize, ydata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        float sdot;
        err = mtxvector_sdot(&x, &y, &sdot);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0f, sdot);
        double ddot;
        err = mtxvector_ddot(&x, &y, &ddot);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0, ddot);
        float cdotu[2];
        err = mtxvector_cdotu(&x, &y, &cdotu);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0f, cdotu[0]); TEST_ASSERT_EQ(0.0f, cdotu[1]);
        float cdotc[2];
        err = mtxvector_cdotc(&x, &y, &cdotc);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0f, cdotc[0]); TEST_ASSERT_EQ(0.0f, cdotc[1]);
        double zdotu[2];
        err = mtxvector_zdotu(&x, &y, &zdotu);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0, zdotu[0]); TEST_ASSERT_EQ(0.0, zdotu[1]);
        double zdotc[2];
        err = mtxvector_zdotc(&x, &y, &zdotc);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0, zdotc[0]); TEST_ASSERT_EQ(0.0, zdotc[1]);
        mtxvector_free(&y);
        mtxvector_free(&x);
    }

    /*
     * Coordinate formats
     */

    {
        int size = 12;
        int nnz = 5;
        int idx[] = {1, 3, 5, 7, 9};
        struct mtxvector x;
        float xdata[] = {1.0f, 1.0f, 1.0f, 2.0f, 3.0f};
        err = mtxvector_init_coordinate_real_single(&x, size, nnz, idx, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        struct mtxvector y;
        float ydata[] = {3.0f, 2.0f, 1.0f, 0.0f, 1.0f};
        err = mtxvector_init_coordinate_real_single(&y, size, nnz, idx, ydata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        float sdot;
        err = mtxvector_sdot(&x, &y, &sdot);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0f, sdot);
        double ddot;
        err = mtxvector_ddot(&x, &y, &ddot);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0, ddot);
        float cdotu[2];
        err = mtxvector_cdotu(&x, &y, &cdotu);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0f, cdotu[0]); TEST_ASSERT_EQ(0.0f, cdotu[1]);
        float cdotc[2];
        err = mtxvector_cdotc(&x, &y, &cdotc);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0f, cdotc[0]); TEST_ASSERT_EQ(0.0f, cdotc[1]);
        double zdotu[2];
        err = mtxvector_zdotu(&x, &y, &zdotu);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0, zdotu[0]); TEST_ASSERT_EQ(0.0, zdotu[1]);
        double zdotc[2];
        err = mtxvector_zdotc(&x, &y, &zdotc);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0, zdotc[0]); TEST_ASSERT_EQ(0.0, zdotc[1]);
        mtxvector_free(&y);
        mtxvector_free(&x);
    }

    {
        int size = 12;
        int nnz = 5;
        int idx[] = {1, 3, 5, 7, 9};
        struct mtxvector x;
        double xdata[] = {1.0, 1.0, 1.0, 2.0, 3.0};
        err = mtxvector_init_coordinate_real_double(&x, size, nnz, idx, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        struct mtxvector y;
        double ydata[] = {3.0, 2.0, 1.0, 0.0, 1.0};
        err = mtxvector_init_coordinate_real_double(&y, size, nnz, idx, ydata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        float sdot;
        err = mtxvector_sdot(&x, &y, &sdot);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0f, sdot);
        double ddot;
        err = mtxvector_ddot(&x, &y, &ddot);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0, ddot);
        float cdotu[2];
        err = mtxvector_cdotu(&x, &y, &cdotu);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0f, cdotu[0]); TEST_ASSERT_EQ(0.0f, cdotu[1]);
        float cdotc[2];
        err = mtxvector_cdotc(&x, &y, &cdotc);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0f, cdotc[0]); TEST_ASSERT_EQ(0.0f, cdotc[1]);
        double zdotu[2];
        err = mtxvector_zdotu(&x, &y, &zdotu);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0, zdotu[0]); TEST_ASSERT_EQ(0.0, zdotu[1]);
        double zdotc[2];
        err = mtxvector_zdotc(&x, &y, &zdotc);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0, zdotc[0]); TEST_ASSERT_EQ(0.0, zdotc[1]);
        mtxvector_free(&y);
        mtxvector_free(&x);
    }

    {
        int size = 12;
        int nnz = 3;
        int idx[] = {1, 3, 5};
        struct mtxvector x;
        float xdata[][2] = {{1.0f, 1.0f}, {1.0f, 2.0f}, {3.0f, 0.0f}};
        err = mtxvector_init_coordinate_complex_single(&x, size, nnz, idx, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        struct mtxvector y;
        float ydata[][2] = {{3.0f, 2.0f}, {1.0f, 0.0f}, {1.0f, 0.0f}};
        err = mtxvector_init_coordinate_complex_single(&y, size, nnz, idx, ydata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        float sdot;
        err = mtxvector_sdot(&x, &y, &sdot);
        TEST_ASSERT_EQ_MSG(MTX_ERR_INVALID_FIELD, err, "%s", mtx_strerror(err));
        double ddot;
        err = mtxvector_ddot(&x, &y, &ddot);
        TEST_ASSERT_EQ_MSG(MTX_ERR_INVALID_FIELD, err, "%s", mtx_strerror(err));
        float cdotu[2];
        err = mtxvector_cdotu(&x, &y, &cdotu);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(5.0f, cdotu[0]); TEST_ASSERT_EQ(7.0f, cdotu[1]);
        float cdotc[2];
        err = mtxvector_cdotc(&x, &y, &cdotc);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0f, cdotc[0]); TEST_ASSERT_EQ(-3.0f, cdotc[1]);
        double zdotu[2];
        err = mtxvector_zdotu(&x, &y, &zdotu);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(5.0, zdotu[0]); TEST_ASSERT_EQ(7.0, zdotu[1]);
        double zdotc[2];
        err = mtxvector_zdotc(&x, &y, &zdotc);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0, zdotc[0]); TEST_ASSERT_EQ(-3.0, zdotc[1]);
        mtxvector_free(&y);
        mtxvector_free(&x);
    }

    {
        int size = 12;
        int nnz = 3;
        int idx[] = {1, 3, 5};
        struct mtxvector x;
        double xdata[][2] = {{1.0, 1.0}, {1.0, 2.0}, {3.0, 0.0}};
        err = mtxvector_init_coordinate_complex_double(&x, size, nnz, idx, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        struct mtxvector y;
        double ydata[][2] = {{3.0, 2.0}, {1.0, 0.0}, {1.0, 0.0}};
        err = mtxvector_init_coordinate_complex_double(&y, size, nnz, idx, ydata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        float sdot;
        err = mtxvector_sdot(&x, &y, &sdot);
        TEST_ASSERT_EQ_MSG(MTX_ERR_INVALID_FIELD, err, "%s", mtx_strerror(err));
        double ddot;
        err = mtxvector_ddot(&x, &y, &ddot);
        TEST_ASSERT_EQ_MSG(MTX_ERR_INVALID_FIELD, err, "%s", mtx_strerror(err));
        float cdotu[2];
        err = mtxvector_cdotu(&x, &y, &cdotu);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(5.0f, cdotu[0]); TEST_ASSERT_EQ(7.0f, cdotu[1]);
        float cdotc[2];
        err = mtxvector_cdotc(&x, &y, &cdotc);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0f, cdotc[0]); TEST_ASSERT_EQ(-3.0f, cdotc[1]);
        double zdotu[2];
        err = mtxvector_zdotu(&x, &y, &zdotu);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(5.0, zdotu[0]); TEST_ASSERT_EQ(7.0, zdotu[1]);
        double zdotc[2];
        err = mtxvector_zdotc(&x, &y, &zdotc);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0, zdotc[0]); TEST_ASSERT_EQ(-3.0, zdotc[1]);
        mtxvector_free(&y);
        mtxvector_free(&x);
    }

    {
        int size = 12;
        int nnz = 5;
        int idx[] = {1, 3, 5, 7, 9};
        struct mtxvector x;
        int32_t xdata[] = {1, 1, 1, 2, 3};
        err = mtxvector_init_coordinate_integer_single(&x, size, nnz, idx, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        struct mtxvector y;
        int32_t ydata[] = {3, 2, 1, 0, 1};
        err = mtxvector_init_coordinate_integer_single(&y, size, nnz, idx, ydata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        float sdot;
        err = mtxvector_sdot(&x, &y, &sdot);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0f, sdot);
        double ddot;
        err = mtxvector_ddot(&x, &y, &ddot);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0, ddot);
        float cdotu[2];
        err = mtxvector_cdotu(&x, &y, &cdotu);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0f, cdotu[0]); TEST_ASSERT_EQ(0.0f, cdotu[1]);
        float cdotc[2];
        err = mtxvector_cdotc(&x, &y, &cdotc);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0f, cdotc[0]); TEST_ASSERT_EQ(0.0f, cdotc[1]);
        double zdotu[2];
        err = mtxvector_zdotu(&x, &y, &zdotu);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0, zdotu[0]); TEST_ASSERT_EQ(0.0, zdotu[1]);
        double zdotc[2];
        err = mtxvector_zdotc(&x, &y, &zdotc);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0, zdotc[0]); TEST_ASSERT_EQ(0.0, zdotc[1]);
        mtxvector_free(&y);
        mtxvector_free(&x);
    }

    {
        int size = 12;
        int nnz = 5;
        int idx[] = {1, 3, 5, 7, 9};
        int64_t xdata[] = {1, 1, 1, 2, 3};
        struct mtxvector x;
        err = mtxvector_init_coordinate_integer_double(&x, size, nnz, idx, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        struct mtxvector y;
        int64_t ydata[] = {3, 2, 1, 0, 1};
        err = mtxvector_init_coordinate_integer_double(&y, size, nnz, idx, ydata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        float sdot;
        err = mtxvector_sdot(&x, &y, &sdot);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0f, sdot);
        double ddot;
        err = mtxvector_ddot(&x, &y, &ddot);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0, ddot);
        float cdotu[2];
        err = mtxvector_cdotu(&x, &y, &cdotu);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0f, cdotu[0]); TEST_ASSERT_EQ(0.0f, cdotu[1]);
        float cdotc[2];
        err = mtxvector_cdotc(&x, &y, &cdotc);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0f, cdotc[0]); TEST_ASSERT_EQ(0.0f, cdotc[1]);
        double zdotu[2];
        err = mtxvector_zdotu(&x, &y, &zdotu);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0, zdotu[0]); TEST_ASSERT_EQ(0.0, zdotu[1]);
        double zdotc[2];
        err = mtxvector_zdotc(&x, &y, &zdotc);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0, zdotc[0]); TEST_ASSERT_EQ(0.0, zdotc[1]);
        mtxvector_free(&y);
        mtxvector_free(&x);
    }
    return TEST_SUCCESS;
}

/**
 * `test_mtxvector_nrm2()' tests computing the Euclidean norm of
 * vectors.
 */
int test_mtxvector_nrm2(void)
{
    int err;

    /*
     * Array formats
     */

    {
        struct mtxvector x;
        float data[] = {1.0f, 1.0f, 1.0f, 2.0f, 3.0f};
        int size = sizeof(data) / sizeof(*data);
        err = mtxvector_init_array_real_single(&x, size, data);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        float snrm2;
        err = mtxvector_snrm2(&x, &snrm2);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(4.0f, snrm2);
        double dnrm2;
        err = mtxvector_dnrm2(&x, &dnrm2);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(4.0, dnrm2);
        mtxvector_free(&x);
    }

    {
        struct mtxvector x;
        double data[] = {1.0, 1.0, 1.0, 2.0, 3.0};
        int size = sizeof(data) / sizeof(*data);
        err = mtxvector_init_array_real_double(&x, size, data);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        float snrm2;
        err = mtxvector_snrm2(&x, &snrm2);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(4.0f, snrm2);
        double dnrm2;
        err = mtxvector_dnrm2(&x, &dnrm2);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(4.0, dnrm2);
        mtxvector_free(&x);
    }

    {
        struct mtxvector x;
        float data[][2] = {{1.0f,1.0f}, {1.0f,2.0f}, {3.0f,0.0f}};
        int size = sizeof(data) / sizeof(*data);
        err = mtxvector_init_array_complex_single(&x, size, data);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        float snrm2;
        err = mtxvector_snrm2(&x, &snrm2);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(4.0f, snrm2);
        double dnrm2;
        err = mtxvector_dnrm2(&x, &dnrm2);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(4.0, dnrm2);
        mtxvector_free(&x);
    }

    {
        struct mtxvector x;
        double data[][2] = {{1.0,1.0}, {1.0,2.0}, {3.0,0.0}};
        int size = sizeof(data) / sizeof(*data);
        err = mtxvector_init_array_complex_double(&x, size, data);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        float snrm2;
        err = mtxvector_snrm2(&x, &snrm2);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(4.0f, snrm2);
        double dnrm2;
        err = mtxvector_dnrm2(&x, &dnrm2);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(4.0, dnrm2);
        mtxvector_free(&x);
    }

    /*
     * Coordinate formats
     */

    {
        struct mtxvector x;
        int size = 12;
        int indices[] = {0, 3, 5, 6, 9};
        float data[] = {1.0f, 1.0f, 1.0f, 2.0f, 3.0f};
        int num_nonzeros = sizeof(data) / sizeof(*data);
        err = mtxvector_init_coordinate_real_single(
            &x, size, num_nonzeros, indices, data);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        float snrm2;
        err = mtxvector_snrm2(&x, &snrm2);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(4.0f, snrm2);
        double dnrm2;
        err = mtxvector_dnrm2(&x, &dnrm2);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(4.0, dnrm2);
        mtxvector_free(&x);
    }

    {
        struct mtxvector x;
        int size = 12;
        int indices[] = {0, 3, 5, 6, 9};
        double data[] = {1.0, 1.0, 1.0, 2.0, 3.0};
        int num_nonzeros = sizeof(data) / sizeof(*data);
        err = mtxvector_init_coordinate_real_double(
            &x, size, num_nonzeros, indices, data);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        float snrm2;
        err = mtxvector_snrm2(&x, &snrm2);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(4.0f, snrm2);
        double dnrm2;
        err = mtxvector_dnrm2(&x, &dnrm2);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(4.0, dnrm2);
        mtxvector_free(&x);
    }

    {
        struct mtxvector x;
        int size = 12;
        int indices[] = {0, 3, 5, 6, 9};
        float data[][2] = {{1.0f,1.0f}, {1.0f,2.0f}, {3.0f,0.0f}};
        int64_t num_nonzeros = sizeof(data) / sizeof(*data);
        err = mtxvector_init_coordinate_complex_single(
            &x, size, num_nonzeros, indices, data);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        float snrm2;
        err = mtxvector_snrm2(&x, &snrm2);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(4.0f, snrm2);
        double dnrm2;
        err = mtxvector_dnrm2(&x, &dnrm2);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(4.0, dnrm2);
        mtxvector_free(&x);
    }

    {
        struct mtxvector x;
        int size = 12;
        int indices[] = {0, 3, 5, 6, 9};
        double data[][2] = {{1.0,1.0}, {1.0,2.0}, {3.0,0.0}};
        int64_t num_nonzeros = sizeof(data) / sizeof(*data);
        err = mtxvector_init_coordinate_complex_double(
            &x, size, num_nonzeros, indices, data);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        float snrm2;
        err = mtxvector_snrm2(&x, &snrm2);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(4.0f, snrm2);
        double dnrm2;
        err = mtxvector_dnrm2(&x, &dnrm2);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(4.0, dnrm2);
        mtxvector_free(&x);
    }
    return TEST_SUCCESS;
}

/**
 * `main()' entry point and test driver.
 */
int main(int argc, char * argv[])
{
    TEST_SUITE_BEGIN("Running tests for vectors\n");
    TEST_RUN(test_mtxvector_from_mtxfile);
    TEST_RUN(test_mtxvector_dot);
    TEST_RUN(test_mtxvector_nrm2);
    TEST_SUITE_END();
    return (TEST_SUITE_STATUS == TEST_SUCCESS) ?
        EXIT_SUCCESS : EXIT_FAILURE;
}
