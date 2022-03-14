/* This file is part of Libmtx.
 *
 * Copyright (C) 2022 James D. Trotter
 *
 * Libmtx is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Libmtx is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Libmtx.  If not, see <https://www.gnu.org/licenses/>.
 *
 * Authors: James D. Trotter <james@simula.no>
 * Last modified: 2022-03-14
 *
 * Unit tests for vectors.
 */

#include "test.h"

#include <libmtx/error.h>
#include <libmtx/mtxfile/mtxfile.h>
#include <libmtx/util/partition.h>
#include <libmtx/vector/vector.h>

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
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxvector x;
        err = mtxvector_from_mtxfile(&x, &mtxfile, mtxvector_auto);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxvector x;
        err = mtxvector_from_mtxfile(&x, &mtxfile, mtxvector_auto);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxvector x;
        err = mtxvector_from_mtxfile(&x, &mtxfile, mtxvector_auto);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxvector x;
        err = mtxvector_from_mtxfile(&x, &mtxfile, mtxvector_auto);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxvector x;
        err = mtxvector_from_mtxfile(&x, &mtxfile, mtxvector_auto);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxvector x;
        err = mtxvector_from_mtxfile(&x, &mtxfile, mtxvector_auto);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxvector x;
        err = mtxvector_from_mtxfile(&x, &mtxfile, mtxvector_auto);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxvector_coordinate, x.type);
        const struct mtxvector_coordinate * x_ = &x.storage.coordinate;
        TEST_ASSERT_EQ(mtx_field_real, x_->field);
        TEST_ASSERT_EQ(mtx_single, x_->precision);
        TEST_ASSERT_EQ(4, x_->size);
        TEST_ASSERT_EQ(3, x_->num_nonzeros);
        TEST_ASSERT_EQ(x_->indices[0], 0);
        TEST_ASSERT_EQ(x_->indices[1], 1);
        TEST_ASSERT_EQ(x_->indices[2], 3);
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
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxvector x;
        err = mtxvector_from_mtxfile(&x, &mtxfile, mtxvector_auto);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxvector_coordinate, x.type);
        const struct mtxvector_coordinate * x_ = &x.storage.coordinate;
        TEST_ASSERT_EQ(mtx_field_real, x_->field);
        TEST_ASSERT_EQ(mtx_double, x_->precision);
        TEST_ASSERT_EQ(4, x_->size);
        TEST_ASSERT_EQ(3, x_->num_nonzeros);
        TEST_ASSERT_EQ(x_->indices[0], 0);
        TEST_ASSERT_EQ(x_->indices[1], 1);
        TEST_ASSERT_EQ(x_->indices[2], 3);
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
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxvector x;
        err = mtxvector_from_mtxfile(&x, &mtxfile, mtxvector_auto);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxvector_coordinate, x.type);
        const struct mtxvector_coordinate * x_ = &x.storage.coordinate;
        TEST_ASSERT_EQ(mtx_field_complex, x_->field);
        TEST_ASSERT_EQ(mtx_single, x_->precision);
        TEST_ASSERT_EQ(4, x_->size);
        TEST_ASSERT_EQ(3, x_->num_nonzeros);
        TEST_ASSERT_EQ(x_->indices[0], 0);
        TEST_ASSERT_EQ(x_->indices[1], 1);
        TEST_ASSERT_EQ(x_->indices[2], 3);
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
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxvector x;
        err = mtxvector_from_mtxfile(&x, &mtxfile, mtxvector_auto);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxvector_coordinate, x.type);
        const struct mtxvector_coordinate * x_ = &x.storage.coordinate;
        TEST_ASSERT_EQ(mtx_field_complex, x_->field);
        TEST_ASSERT_EQ(mtx_double, x_->precision);
        TEST_ASSERT_EQ(4, x_->size);
        TEST_ASSERT_EQ(3, x_->num_nonzeros);
        TEST_ASSERT_EQ(x_->indices[0], 0);
        TEST_ASSERT_EQ(x_->indices[1], 1);
        TEST_ASSERT_EQ(x_->indices[2], 3);
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
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxvector x;
        err = mtxvector_from_mtxfile(&x, &mtxfile, mtxvector_auto);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxvector_coordinate, x.type);
        const struct mtxvector_coordinate * x_ = &x.storage.coordinate;
        TEST_ASSERT_EQ(mtx_field_integer, x_->field);
        TEST_ASSERT_EQ(mtx_single, x_->precision);
        TEST_ASSERT_EQ(4, x_->size);
        TEST_ASSERT_EQ(3, x_->num_nonzeros);
        TEST_ASSERT_EQ(x_->indices[0], 0);
        TEST_ASSERT_EQ(x_->indices[1], 1);
        TEST_ASSERT_EQ(x_->indices[2], 3);
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
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxvector x;
        err = mtxvector_from_mtxfile(&x, &mtxfile, mtxvector_auto);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxvector_coordinate, x.type);
        const struct mtxvector_coordinate * x_ = &x.storage.coordinate;
        TEST_ASSERT_EQ(mtx_field_integer, x_->field);
        TEST_ASSERT_EQ(mtx_double, x_->precision);
        TEST_ASSERT_EQ(4, x_->size);
        TEST_ASSERT_EQ(3, x_->num_nonzeros);
        TEST_ASSERT_EQ(x_->indices[0], 0);
        TEST_ASSERT_EQ(x_->indices[1], 1);
        TEST_ASSERT_EQ(x_->indices[2], 3);
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
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxvector x;
        err = mtxvector_from_mtxfile(&x, &mtxfile, mtxvector_auto);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxvector_coordinate, x.type);
        const struct mtxvector_coordinate * x_ = &x.storage.coordinate;
        TEST_ASSERT_EQ(mtx_field_pattern, x_->field);
        TEST_ASSERT_EQ(4, x_->size);
        TEST_ASSERT_EQ(3, x_->num_nonzeros);
        TEST_ASSERT_EQ(x_->indices[0], 0);
        TEST_ASSERT_EQ(x_->indices[1], 1);
        TEST_ASSERT_EQ(x_->indices[2], 3);
        mtxvector_free(&x);
        mtxfile_free(&mtxfile);
    }
    return TEST_SUCCESS;
}

/**
 * `test_mtxvector_to_mtxfile()' tests converting vectors to Matrix
 * Market files.
 */
int test_mtxvector_to_mtxfile(void)
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
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxfile mtxfile;
        err = mtxvector_to_mtxfile(&mtxfile, &x, mtxfile_array);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_vector, mtxfile.header.object);
        TEST_ASSERT_EQ(mtxfile_array, mtxfile.header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtxfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxfile.header.symmetry);
        TEST_ASSERT_EQ(xsize, mtxfile.size.num_rows);
        TEST_ASSERT_EQ(-1, mtxfile.size.num_columns);
        TEST_ASSERT_EQ(-1, mtxfile.size.num_nonzeros);
        TEST_ASSERT_EQ(mtx_single, mtxfile.precision);
        const float * data = mtxfile.data.array_real_single;
        for (int64_t k = 0; k < xsize; k++)
            TEST_ASSERT_EQ(xdata[k], data[k]);
        mtxfile_free(&mtxfile);
        mtxvector_free(&x);
    }

    {
        struct mtxvector x;
        double xdata[] = {1.0, 1.0, 1.0, 2.0, 3.0};
        int xsize = sizeof(xdata) / sizeof(*xdata);
        err = mtxvector_init_array_real_double(&x, xsize, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxfile mtxfile;
        err = mtxvector_to_mtxfile(&mtxfile, &x, mtxfile_array);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_vector, mtxfile.header.object);
        TEST_ASSERT_EQ(mtxfile_array, mtxfile.header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtxfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxfile.header.symmetry);
        TEST_ASSERT_EQ(xsize, mtxfile.size.num_rows);
        TEST_ASSERT_EQ(-1, mtxfile.size.num_columns);
        TEST_ASSERT_EQ(-1, mtxfile.size.num_nonzeros);
        TEST_ASSERT_EQ(mtx_double, mtxfile.precision);
        const double * data = mtxfile.data.array_real_double;
        for (int64_t k = 0; k < xsize; k++)
            TEST_ASSERT_EQ(xdata[k], data[k]);
        mtxfile_free(&mtxfile);
        mtxvector_free(&x);
    }

    {
        struct mtxvector x;
        float xdata[][2] = {{1.0f, 1.0f}, {1.0f, 2.0f}, {3.0f, 0.0f}};
        int xsize = sizeof(xdata) / sizeof(*xdata);
        err = mtxvector_init_array_complex_single(&x, xsize, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxfile mtxfile;
        err = mtxvector_to_mtxfile(&mtxfile, &x, mtxfile_array);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_vector, mtxfile.header.object);
        TEST_ASSERT_EQ(mtxfile_array, mtxfile.header.format);
        TEST_ASSERT_EQ(mtxfile_complex, mtxfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxfile.header.symmetry);
        TEST_ASSERT_EQ(xsize, mtxfile.size.num_rows);
        TEST_ASSERT_EQ(-1, mtxfile.size.num_columns);
        TEST_ASSERT_EQ(-1, mtxfile.size.num_nonzeros);
        TEST_ASSERT_EQ(mtx_single, mtxfile.precision);
        const float (* data)[2] = mtxfile.data.array_complex_single;
        for (int64_t k = 0; k < xsize; k++) {
            TEST_ASSERT_EQ(xdata[k][0], data[k][0]);
            TEST_ASSERT_EQ(xdata[k][1], data[k][1]);
        }
        mtxfile_free(&mtxfile);
        mtxvector_free(&x);
    }

    {
        struct mtxvector x;
        double xdata[][2] = {{1.0, 1.0}, {1.0, 2.0}, {3.0, 0.0}};
        int xsize = sizeof(xdata) / sizeof(*xdata);
        err = mtxvector_init_array_complex_double(&x, xsize, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxfile mtxfile;
        err = mtxvector_to_mtxfile(&mtxfile, &x, mtxfile_array);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_vector, mtxfile.header.object);
        TEST_ASSERT_EQ(mtxfile_array, mtxfile.header.format);
        TEST_ASSERT_EQ(mtxfile_complex, mtxfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxfile.header.symmetry);
        TEST_ASSERT_EQ(xsize, mtxfile.size.num_rows);
        TEST_ASSERT_EQ(-1, mtxfile.size.num_columns);
        TEST_ASSERT_EQ(-1, mtxfile.size.num_nonzeros);
        TEST_ASSERT_EQ(mtx_double, mtxfile.precision);
        const double (* data)[2] = mtxfile.data.array_complex_double;
        for (int64_t k = 0; k < xsize; k++) {
            TEST_ASSERT_EQ(xdata[k][0], data[k][0]);
            TEST_ASSERT_EQ(xdata[k][1], data[k][1]);
        }
        mtxfile_free(&mtxfile);
        mtxvector_free(&x);
    }

    {
        struct mtxvector x;
        int32_t xdata[] = {1, 1, 1, 2, 3};
        int xsize = sizeof(xdata) / sizeof(*xdata);
        err = mtxvector_init_array_integer_single(&x, xsize, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxfile mtxfile;
        err = mtxvector_to_mtxfile(&mtxfile, &x, mtxfile_array);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_vector, mtxfile.header.object);
        TEST_ASSERT_EQ(mtxfile_array, mtxfile.header.format);
        TEST_ASSERT_EQ(mtxfile_integer, mtxfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxfile.header.symmetry);
        TEST_ASSERT_EQ(xsize, mtxfile.size.num_rows);
        TEST_ASSERT_EQ(-1, mtxfile.size.num_columns);
        TEST_ASSERT_EQ(-1, mtxfile.size.num_nonzeros);
        TEST_ASSERT_EQ(mtx_single, mtxfile.precision);
        const int32_t * data = mtxfile.data.array_integer_single;
        for (int64_t k = 0; k < xsize; k++)
            TEST_ASSERT_EQ(xdata[k], data[k]);
        mtxfile_free(&mtxfile);
        mtxvector_free(&x);
    }

    {
        struct mtxvector x;
        int64_t xdata[] = {1, 1, 1, 2, 3};
        int xsize = sizeof(xdata) / sizeof(*xdata);
        err = mtxvector_init_array_integer_double(&x, xsize, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxfile mtxfile;
        err = mtxvector_to_mtxfile(&mtxfile, &x, mtxfile_array);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_vector, mtxfile.header.object);
        TEST_ASSERT_EQ(mtxfile_array, mtxfile.header.format);
        TEST_ASSERT_EQ(mtxfile_integer, mtxfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxfile.header.symmetry);
        TEST_ASSERT_EQ(xsize, mtxfile.size.num_rows);
        TEST_ASSERT_EQ(-1, mtxfile.size.num_columns);
        TEST_ASSERT_EQ(-1, mtxfile.size.num_nonzeros);
        TEST_ASSERT_EQ(mtx_double, mtxfile.precision);
        const int64_t * data = mtxfile.data.array_integer_double;
        for (int64_t k = 0; k < xsize; k++)
            TEST_ASSERT_EQ(xdata[k], data[k]);
        mtxfile_free(&mtxfile);
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
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxfile mtxfile;
        err = mtxvector_to_mtxfile(&mtxfile, &x, mtxfile_coordinate);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_vector, mtxfile.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxfile.header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtxfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxfile.header.symmetry);
        TEST_ASSERT_EQ(size, mtxfile.size.num_rows);
        TEST_ASSERT_EQ(-1, mtxfile.size.num_columns);
        TEST_ASSERT_EQ(nnz, mtxfile.size.num_nonzeros);
        TEST_ASSERT_EQ(mtx_single, mtxfile.precision);
        const struct mtxfile_vector_coordinate_real_single * data =
            mtxfile.data.vector_coordinate_real_single;
        for (int64_t k = 0; k < nnz; k++) {
            TEST_ASSERT_EQ(idx[k]+1, data[k].i);
            TEST_ASSERT_EQ(xdata[k], data[k].a);
        }
        mtxfile_free(&mtxfile);
        mtxvector_free(&x);
    }

    {
        int size = 12;
        int nnz = 5;
        int idx[] = {1, 3, 5, 7, 9};
        struct mtxvector x;
        double xdata[] = {1.0, 1.0, 1.0, 2.0, 3.0};
        int xsize = sizeof(xdata) / sizeof(*xdata);
        err = mtxvector_init_coordinate_real_double(&x, size, nnz, idx, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxfile mtxfile;
        err = mtxvector_to_mtxfile(&mtxfile, &x, mtxfile_coordinate);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_vector, mtxfile.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxfile.header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtxfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxfile.header.symmetry);
        TEST_ASSERT_EQ(size, mtxfile.size.num_rows);
        TEST_ASSERT_EQ(-1, mtxfile.size.num_columns);
        TEST_ASSERT_EQ(nnz, mtxfile.size.num_nonzeros);
        TEST_ASSERT_EQ(mtx_double, mtxfile.precision);
        const struct mtxfile_vector_coordinate_real_double * data =
            mtxfile.data.vector_coordinate_real_double;
        for (int64_t k = 0; k < nnz; k++) {
            TEST_ASSERT_EQ(idx[k]+1, data[k].i);
            TEST_ASSERT_EQ(xdata[k], data[k].a);
        }
        mtxfile_free(&mtxfile);
        mtxvector_free(&x);
    }

    {
        int size = 12;
        int nnz = 3;
        int idx[] = {1, 3, 5, 7, 9};
        struct mtxvector x;
        float xdata[][2] = {{1.0f, 1.0f}, {1.0f, 2.0f}, {3.0f, 0.0f}};
        int xsize = sizeof(xdata) / sizeof(*xdata);
        err = mtxvector_init_coordinate_complex_single(&x, size, nnz, idx, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxfile mtxfile;
        err = mtxvector_to_mtxfile(&mtxfile, &x, mtxfile_coordinate);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_vector, mtxfile.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxfile.header.format);
        TEST_ASSERT_EQ(mtxfile_complex, mtxfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxfile.header.symmetry);
        TEST_ASSERT_EQ(size, mtxfile.size.num_rows);
        TEST_ASSERT_EQ(-1, mtxfile.size.num_columns);
        TEST_ASSERT_EQ(nnz, mtxfile.size.num_nonzeros);
        TEST_ASSERT_EQ(mtx_single, mtxfile.precision);
        const struct mtxfile_vector_coordinate_complex_single * data =
            mtxfile.data.vector_coordinate_complex_single;
        for (int64_t k = 0; k < nnz; k++) {
            TEST_ASSERT_EQ(idx[k]+1, data[k].i);
            TEST_ASSERT_EQ(xdata[k][0], data[k].a[0]);
            TEST_ASSERT_EQ(xdata[k][1], data[k].a[1]);
        }
        mtxfile_free(&mtxfile);
        mtxvector_free(&x);
    }

    {
        int size = 12;
        int nnz = 3;
        int idx[] = {1, 3, 5, 7, 9};
        struct mtxvector x;
        double xdata[][2] = {{1.0, 1.0}, {1.0, 2.0}, {3.0, 0.0}};
        int xsize = sizeof(xdata) / sizeof(*xdata);
        err = mtxvector_init_coordinate_complex_double(&x, size, nnz, idx, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxfile mtxfile;
        err = mtxvector_to_mtxfile(&mtxfile, &x, mtxfile_coordinate);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_vector, mtxfile.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxfile.header.format);
        TEST_ASSERT_EQ(mtxfile_complex, mtxfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxfile.header.symmetry);
        TEST_ASSERT_EQ(size, mtxfile.size.num_rows);
        TEST_ASSERT_EQ(-1, mtxfile.size.num_columns);
        TEST_ASSERT_EQ(nnz, mtxfile.size.num_nonzeros);
        TEST_ASSERT_EQ(mtx_double, mtxfile.precision);
        const struct mtxfile_vector_coordinate_complex_double * data =
            mtxfile.data.vector_coordinate_complex_double;
        for (int64_t k = 0; k < nnz; k++) {
            TEST_ASSERT_EQ(idx[k]+1, data[k].i);
            TEST_ASSERT_EQ(xdata[k][0], data[k].a[0]);
            TEST_ASSERT_EQ(xdata[k][1], data[k].a[1]);
        }
        mtxfile_free(&mtxfile);
        mtxvector_free(&x);
    }

    {
        int size = 12;
        int nnz = 5;
        int idx[] = {1, 3, 5, 7, 9};
        struct mtxvector x;
        int32_t xdata[] = {1, 1, 1, 2, 3};
        int xsize = sizeof(xdata) / sizeof(*xdata);
        err = mtxvector_init_coordinate_integer_single(&x, size, nnz, idx, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxfile mtxfile;
        err = mtxvector_to_mtxfile(&mtxfile, &x, mtxfile_coordinate);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_vector, mtxfile.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxfile.header.format);
        TEST_ASSERT_EQ(mtxfile_integer, mtxfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxfile.header.symmetry);
        TEST_ASSERT_EQ(size, mtxfile.size.num_rows);
        TEST_ASSERT_EQ(-1, mtxfile.size.num_columns);
        TEST_ASSERT_EQ(nnz, mtxfile.size.num_nonzeros);
        TEST_ASSERT_EQ(mtx_single, mtxfile.precision);
        const struct mtxfile_vector_coordinate_integer_single * data =
            mtxfile.data.vector_coordinate_integer_single;
        for (int64_t k = 0; k < nnz; k++) {
            TEST_ASSERT_EQ(idx[k]+1, data[k].i);
            TEST_ASSERT_EQ(xdata[k], data[k].a);
        }
        mtxfile_free(&mtxfile);
        mtxvector_free(&x);
    }

    {
        int size = 12;
        int nnz = 5;
        int idx[] = {1, 3, 5, 7, 9};
        struct mtxvector x;
        int64_t xdata[] = {1, 1, 1, 2, 3};
        int xsize = sizeof(xdata) / sizeof(*xdata);
        err = mtxvector_init_coordinate_integer_double(&x, size, nnz, idx, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxfile mtxfile;
        err = mtxvector_to_mtxfile(&mtxfile, &x, mtxfile_coordinate);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_vector, mtxfile.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxfile.header.format);
        TEST_ASSERT_EQ(mtxfile_integer, mtxfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxfile.header.symmetry);
        TEST_ASSERT_EQ(size, mtxfile.size.num_rows);
        TEST_ASSERT_EQ(-1, mtxfile.size.num_columns);
        TEST_ASSERT_EQ(nnz, mtxfile.size.num_nonzeros);
        TEST_ASSERT_EQ(mtx_double, mtxfile.precision);
        const struct mtxfile_vector_coordinate_integer_double * data =
            mtxfile.data.vector_coordinate_integer_double;
        for (int64_t k = 0; k < nnz; k++) {
            TEST_ASSERT_EQ(idx[k]+1, data[k].i);
            TEST_ASSERT_EQ(xdata[k], data[k].a);
        }
        mtxfile_free(&mtxfile);
        mtxvector_free(&x);
    }
    return TEST_SUCCESS;
}

/**
 * ‘test_mtxvector_partition()’ tests partitioning vectors.
 */
int test_mtxvector_partition(void)
{
    int err;

    {
        struct mtxvector src;
        float srcdata[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0, 8.0f, 9.0f};
        int srcsize = sizeof(srcdata) / sizeof(*srcdata);
        err = mtxvector_init_array_real_single(&src, srcsize, srcdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        int num_parts = 2;
        enum mtxpartitioning parttype = mtx_block;
        struct mtxpartition part;
        err = mtxpartition_init(
            &part, parttype, srcsize, num_parts, NULL, 0, NULL, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        struct mtxvector dsts[num_parts];
        err = mtxvector_partition(dsts, &src, &part);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        {
            TEST_ASSERT_EQ(mtxvector_array, dsts[0].type);
            const struct mtxvector_array * x = &dsts[0].storage.array;
            TEST_ASSERT_EQ(mtx_field_real, x->field);
            TEST_ASSERT_EQ(mtx_single, x->precision);
            TEST_ASSERT_EQ(5, x->size);
            TEST_ASSERT_EQ(x->data.real_single[0], 1.0f);
            TEST_ASSERT_EQ(x->data.real_single[1], 2.0f);
            TEST_ASSERT_EQ(x->data.real_single[2], 3.0f);
            TEST_ASSERT_EQ(x->data.real_single[3], 4.0f);
            TEST_ASSERT_EQ(x->data.real_single[4], 5.0f);
            mtxvector_free(&dsts[0]);
        }
        {
            TEST_ASSERT_EQ(mtxvector_array, dsts[1].type);
            const struct mtxvector_array * x = &dsts[1].storage.array;
            TEST_ASSERT_EQ(mtx_field_real, x->field);
            TEST_ASSERT_EQ(mtx_single, x->precision);
            TEST_ASSERT_EQ(4, x->size);
            TEST_ASSERT_EQ(x->data.real_single[0], 6.0f);
            TEST_ASSERT_EQ(x->data.real_single[1], 7.0f);
            TEST_ASSERT_EQ(x->data.real_single[2], 8.0f);
            TEST_ASSERT_EQ(x->data.real_single[3], 9.0f);
            mtxvector_free(&dsts[1]);
        }
        mtxpartition_free(&part);
        mtxvector_free(&src);
    }

    {
        int srcsize = 12;
        int srcnnz = 5;
        struct mtxvector src;
        int srcidx[] = {0, 2, 4, 6, 8};
        int64_t srcdata[] = {1, 3, 5, 7, 9};
        err = mtxvector_init_coordinate_integer_double(
            &src, srcsize, srcnnz, srcidx, srcdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        int num_parts = 2;
        enum mtxpartitioning parttype = mtx_block;
        struct mtxpartition part;
        err = mtxpartition_init(
            &part, parttype, srcsize, num_parts, NULL, 0, NULL, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        struct mtxvector dsts[num_parts];
        err = mtxvector_partition(dsts, &src, &part);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        {
            TEST_ASSERT_EQ(mtxvector_coordinate, dsts[0].type);
            const struct mtxvector_coordinate * x = &dsts[0].storage.coordinate;
            TEST_ASSERT_EQ(mtx_field_integer, x->field);
            TEST_ASSERT_EQ(mtx_double, x->precision);
            TEST_ASSERT_EQ(6, x->size);
            TEST_ASSERT_EQ(3, x->num_nonzeros);
            TEST_ASSERT_EQ(x->indices[0], 0);
            TEST_ASSERT_EQ(x->indices[1], 2);
            TEST_ASSERT_EQ(x->indices[2], 4);
            TEST_ASSERT_EQ(x->data.integer_double[0], 1);
            TEST_ASSERT_EQ(x->data.integer_double[1], 3);
            TEST_ASSERT_EQ(x->data.integer_double[2], 5);
            mtxvector_free(&dsts[0]);
        }
        {
            TEST_ASSERT_EQ(mtxvector_coordinate, dsts[1].type);
            const struct mtxvector_coordinate * x = &dsts[1].storage.coordinate;
            TEST_ASSERT_EQ(mtx_field_integer, x->field);
            TEST_ASSERT_EQ(mtx_double, x->precision);
            TEST_ASSERT_EQ(6, x->size);
            TEST_ASSERT_EQ(2, x->num_nonzeros);
            TEST_ASSERT_EQ(x->indices[0], 0);
            TEST_ASSERT_EQ(x->indices[1], 2);
            TEST_ASSERT_EQ(x->data.integer_double[0], 7);
            TEST_ASSERT_EQ(x->data.integer_double[1], 9);
            mtxvector_free(&dsts[1]);
        }
        mtxpartition_free(&part);
        mtxvector_free(&src);
    }
    return TEST_SUCCESS;
}

/**
 * ‘test_mtxvector_join()’ tests joining vectors.
 */
int test_mtxvector_join(void)
{
    int err;

    {
        struct mtxvector srcs[2];
        int num_rows[] = {6, 3};
        float srcdata0[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        float srcdata1[] = {7.0, 8.0f, 9.0f};
        err = mtxvector_init_array_real_single(&srcs[0], num_rows[0], srcdata0);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_init_array_real_single(&srcs[1], num_rows[1], srcdata1);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        int num_parts = 2;
        int64_t partsizes[] = {6, 3};
        enum mtxpartitioning parttype = mtx_block;
        struct mtxpartition part;
        err = mtxpartition_init(
            &part, parttype, num_rows[0]+num_rows[1],
            num_parts, partsizes, 0, NULL, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        struct mtxvector dst;
        err = mtxvector_join(&dst, srcs, &part);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxvector_array, dst.type);
        const struct mtxvector_array * A = &dst.storage.array;
        TEST_ASSERT_EQ(mtx_field_real, A->field);
        TEST_ASSERT_EQ(mtx_single, A->precision);
        TEST_ASSERT_EQ(9, A->size);
        TEST_ASSERT_EQ(A->data.real_single[0], 1.0f);
        TEST_ASSERT_EQ(A->data.real_single[1], 2.0f);
        TEST_ASSERT_EQ(A->data.real_single[2], 3.0f);
        TEST_ASSERT_EQ(A->data.real_single[3], 4.0f);
        TEST_ASSERT_EQ(A->data.real_single[4], 5.0f);
        TEST_ASSERT_EQ(A->data.real_single[5], 6.0f);
        TEST_ASSERT_EQ(A->data.real_single[6], 7.0f);
        TEST_ASSERT_EQ(A->data.real_single[7], 8.0f);
        TEST_ASSERT_EQ(A->data.real_single[8], 9.0f);
        mtxvector_free(&dst);
        mtxpartition_free(&part);
        mtxvector_free(&srcs[1]);
        mtxvector_free(&srcs[0]);
    }

    {
        struct mtxvector srcs[2];
        int num_rows[] = {9, 3};
        int nnz[] = {6, 3};
        int idx0[] = {0, 1, 2, 3, 4, 5};
        int idx1[] = {0, 1, 2};
        err = mtxvector_init_coordinate_pattern(
            &srcs[0], num_rows[0], nnz[0], idx0);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_init_coordinate_pattern(
            &srcs[1], num_rows[1], nnz[1], idx1);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        int num_row_parts = 2;
        int64_t partsizes[] = {9, 3};
        enum mtxpartitioning parttype = mtx_block;
        struct mtxpartition part;
        err = mtxpartition_init(
            &part, parttype, num_rows[0]+num_rows[1], num_row_parts,
            partsizes, 0, NULL, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        struct mtxvector dst;
        err = mtxvector_join(&dst, srcs, &part);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxvector_coordinate, dst.type);
        const struct mtxvector_coordinate * A = &dst.storage.coordinate;
        TEST_ASSERT_EQ(mtx_field_pattern, A->field);
        TEST_ASSERT_EQ(12, A->size);
        TEST_ASSERT_EQ(9, A->num_nonzeros);
        TEST_ASSERT_EQ(A->indices[0], 0);
        TEST_ASSERT_EQ(A->indices[1], 1);
        TEST_ASSERT_EQ(A->indices[2], 2);
        TEST_ASSERT_EQ(A->indices[3], 3);
        TEST_ASSERT_EQ(A->indices[4], 4);
        TEST_ASSERT_EQ(A->indices[5], 5);
        TEST_ASSERT_EQ(A->indices[6], 9);
        TEST_ASSERT_EQ(A->indices[7],10);
        TEST_ASSERT_EQ(A->indices[8],11);
        mtxvector_free(&dst);
        mtxpartition_free(&part);
        mtxvector_free(&srcs[1]);
        mtxvector_free(&srcs[0]);
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
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxvector y;
        float ydata[] = {3.0f, 2.0f, 1.0f, 0.0f, 1.0f};
        int ysize = sizeof(ydata) / sizeof(*ydata);
        err = mtxvector_init_array_real_single(&y, ysize, ydata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        float sdot;
        err = mtxvector_sdot(&x, &y, &sdot, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, sdot);
        double ddot;
        err = mtxvector_ddot(&x, &y, &ddot, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, ddot);
        float cdotu[2];
        err = mtxvector_cdotu(&x, &y, &cdotu, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, cdotu[0]); TEST_ASSERT_EQ(0.0f, cdotu[1]);
        float cdotc[2];
        err = mtxvector_cdotc(&x, &y, &cdotc, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, cdotc[0]); TEST_ASSERT_EQ(0.0f, cdotc[1]);
        double zdotu[2];
        err = mtxvector_zdotu(&x, &y, &zdotu, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, zdotu[0]); TEST_ASSERT_EQ(0.0, zdotu[1]);
        double zdotc[2];
        err = mtxvector_zdotc(&x, &y, &zdotc, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, zdotc[0]); TEST_ASSERT_EQ(0.0, zdotc[1]);
        mtxvector_free(&y);
        mtxvector_free(&x);
    }

    {
        struct mtxvector x;
        double xdata[] = {1.0, 1.0, 1.0, 2.0, 3.0};
        int xsize = sizeof(xdata) / sizeof(*xdata);
        err = mtxvector_init_array_real_double(&x, xsize, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxvector y;
        double ydata[] = {3.0, 2.0, 1.0, 0.0, 1.0};
        int ysize = sizeof(ydata) / sizeof(*ydata);
        err = mtxvector_init_array_real_double(&y, ysize, ydata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        float sdot;
        err = mtxvector_sdot(&x, &y, &sdot, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, sdot);
        double ddot;
        err = mtxvector_ddot(&x, &y, &ddot, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, ddot);
        float cdotu[2];
        err = mtxvector_cdotu(&x, &y, &cdotu, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, cdotu[0]); TEST_ASSERT_EQ(0.0f, cdotu[1]);
        float cdotc[2];
        err = mtxvector_cdotc(&x, &y, &cdotc, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, cdotc[0]); TEST_ASSERT_EQ(0.0f, cdotc[1]);
        double zdotu[2];
        err = mtxvector_zdotu(&x, &y, &zdotu, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, zdotu[0]); TEST_ASSERT_EQ(0.0, zdotu[1]);
        double zdotc[2];
        err = mtxvector_zdotc(&x, &y, &zdotc, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, zdotc[0]); TEST_ASSERT_EQ(0.0, zdotc[1]);
        mtxvector_free(&y);
        mtxvector_free(&x);
    }

    {
        struct mtxvector x;
        float xdata[][2] = {{1.0f, 1.0f}, {1.0f, 2.0f}, {3.0f, 0.0f}};
        int xsize = sizeof(xdata) / sizeof(*xdata);
        err = mtxvector_init_array_complex_single(&x, xsize, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxvector y;
        float ydata[][2] = {{3.0f, 2.0f}, {1.0f, 0.0f}, {1.0f, 0.0f}};
        int ysize = sizeof(ydata) / sizeof(*ydata);
        err = mtxvector_init_array_complex_single(&y, ysize, ydata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        float sdot;
        err = mtxvector_sdot(&x, &y, &sdot, NULL);
        TEST_ASSERT_EQ_MSG(MTX_ERR_INVALID_FIELD, err, "%s", mtxstrerror(err));
        double ddot;
        err = mtxvector_ddot(&x, &y, &ddot, NULL);
        TEST_ASSERT_EQ_MSG(MTX_ERR_INVALID_FIELD, err, "%s", mtxstrerror(err));
        float cdotu[2];
        err = mtxvector_cdotu(&x, &y, &cdotu, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(5.0f, cdotu[0]); TEST_ASSERT_EQ(7.0f, cdotu[1]);
        float cdotc[2];
        err = mtxvector_cdotc(&x, &y, &cdotc, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, cdotc[0]); TEST_ASSERT_EQ(-3.0f, cdotc[1]);
        double zdotu[2];
        err = mtxvector_zdotu(&x, &y, &zdotu, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(5.0, zdotu[0]); TEST_ASSERT_EQ(7.0, zdotu[1]);
        double zdotc[2];
        err = mtxvector_zdotc(&x, &y, &zdotc, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, zdotc[0]); TEST_ASSERT_EQ(-3.0, zdotc[1]);
        mtxvector_free(&y);
        mtxvector_free(&x);
    }

    {
        struct mtxvector x;
        double xdata[][2] = {{1.0, 1.0}, {1.0, 2.0}, {3.0, 0.0}};
        int xsize = sizeof(xdata) / sizeof(*xdata);
        err = mtxvector_init_array_complex_double(&x, xsize, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxvector y;
        double ydata[][2] = {{3.0, 2.0}, {1.0, 0.0}, {1.0, 0.0}};
        int ysize = sizeof(ydata) / sizeof(*ydata);
        err = mtxvector_init_array_complex_double(&y, ysize, ydata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        float sdot;
        err = mtxvector_sdot(&x, &y, &sdot, NULL);
        TEST_ASSERT_EQ_MSG(MTX_ERR_INVALID_FIELD, err, "%s", mtxstrerror(err));
        double ddot;
        err = mtxvector_ddot(&x, &y, &ddot, NULL);
        TEST_ASSERT_EQ_MSG(MTX_ERR_INVALID_FIELD, err, "%s", mtxstrerror(err));
        float cdotu[2];
        err = mtxvector_cdotu(&x, &y, &cdotu, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(5.0f, cdotu[0]); TEST_ASSERT_EQ(7.0f, cdotu[1]);
        float cdotc[2];
        err = mtxvector_cdotc(&x, &y, &cdotc, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, cdotc[0]); TEST_ASSERT_EQ(-3.0f, cdotc[1]);
        double zdotu[2];
        err = mtxvector_zdotu(&x, &y, &zdotu, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(5.0, zdotu[0]); TEST_ASSERT_EQ(7.0, zdotu[1]);
        double zdotc[2];
        err = mtxvector_zdotc(&x, &y, &zdotc, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, zdotc[0]); TEST_ASSERT_EQ(-3.0, zdotc[1]);
        mtxvector_free(&y);
        mtxvector_free(&x);
    }

    {
        struct mtxvector x;
        int32_t xdata[] = {1, 1, 1, 2, 3};
        int xsize = sizeof(xdata) / sizeof(*xdata);
        err = mtxvector_init_array_integer_single(&x, xsize, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxvector y;
        int32_t ydata[] = {3, 2, 1, 0, 1};
        int ysize = sizeof(ydata) / sizeof(*ydata);
        err = mtxvector_init_array_integer_single(&y, ysize, ydata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        float sdot;
        err = mtxvector_sdot(&x, &y, &sdot, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, sdot);
        double ddot;
        err = mtxvector_ddot(&x, &y, &ddot, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, ddot);
        float cdotu[2];
        err = mtxvector_cdotu(&x, &y, &cdotu, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, cdotu[0]); TEST_ASSERT_EQ(0.0f, cdotu[1]);
        float cdotc[2];
        err = mtxvector_cdotc(&x, &y, &cdotc, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, cdotc[0]); TEST_ASSERT_EQ(0.0f, cdotc[1]);
        double zdotu[2];
        err = mtxvector_zdotu(&x, &y, &zdotu, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, zdotu[0]); TEST_ASSERT_EQ(0.0, zdotu[1]);
        double zdotc[2];
        err = mtxvector_zdotc(&x, &y, &zdotc, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, zdotc[0]); TEST_ASSERT_EQ(0.0, zdotc[1]);
        mtxvector_free(&y);
        mtxvector_free(&x);
    }

    {
        struct mtxvector x;
        int64_t xdata[] = {1, 1, 1, 2, 3};
        int xsize = sizeof(xdata) / sizeof(*xdata);
        err = mtxvector_init_array_integer_double(&x, xsize, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxvector y;
        int64_t ydata[] = {3, 2, 1, 0, 1};
        int ysize = sizeof(ydata) / sizeof(*ydata);
        err = mtxvector_init_array_integer_double(&y, ysize, ydata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        float sdot;
        err = mtxvector_sdot(&x, &y, &sdot, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, sdot);
        double ddot;
        err = mtxvector_ddot(&x, &y, &ddot, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, ddot);
        float cdotu[2];
        err = mtxvector_cdotu(&x, &y, &cdotu, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, cdotu[0]); TEST_ASSERT_EQ(0.0f, cdotu[1]);
        float cdotc[2];
        err = mtxvector_cdotc(&x, &y, &cdotc, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, cdotc[0]); TEST_ASSERT_EQ(0.0f, cdotc[1]);
        double zdotu[2];
        err = mtxvector_zdotu(&x, &y, &zdotu, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, zdotu[0]); TEST_ASSERT_EQ(0.0, zdotu[1]);
        double zdotc[2];
        err = mtxvector_zdotc(&x, &y, &zdotc, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxvector y;
        float ydata[] = {3.0f, 2.0f, 1.0f, 0.0f, 1.0f};
        err = mtxvector_init_coordinate_real_single(&y, size, nnz, idx, ydata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        float sdot;
        err = mtxvector_sdot(&x, &y, &sdot, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, sdot);
        double ddot;
        err = mtxvector_ddot(&x, &y, &ddot, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, ddot);
        float cdotu[2];
        err = mtxvector_cdotu(&x, &y, &cdotu, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, cdotu[0]); TEST_ASSERT_EQ(0.0f, cdotu[1]);
        float cdotc[2];
        err = mtxvector_cdotc(&x, &y, &cdotc, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, cdotc[0]); TEST_ASSERT_EQ(0.0f, cdotc[1]);
        double zdotu[2];
        err = mtxvector_zdotu(&x, &y, &zdotu, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, zdotu[0]); TEST_ASSERT_EQ(0.0, zdotu[1]);
        double zdotc[2];
        err = mtxvector_zdotc(&x, &y, &zdotc, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxvector y;
        double ydata[] = {3.0, 2.0, 1.0, 0.0, 1.0};
        err = mtxvector_init_coordinate_real_double(&y, size, nnz, idx, ydata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        float sdot;
        err = mtxvector_sdot(&x, &y, &sdot, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, sdot);
        double ddot;
        err = mtxvector_ddot(&x, &y, &ddot, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, ddot);
        float cdotu[2];
        err = mtxvector_cdotu(&x, &y, &cdotu, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, cdotu[0]); TEST_ASSERT_EQ(0.0f, cdotu[1]);
        float cdotc[2];
        err = mtxvector_cdotc(&x, &y, &cdotc, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, cdotc[0]); TEST_ASSERT_EQ(0.0f, cdotc[1]);
        double zdotu[2];
        err = mtxvector_zdotu(&x, &y, &zdotu, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, zdotu[0]); TEST_ASSERT_EQ(0.0, zdotu[1]);
        double zdotc[2];
        err = mtxvector_zdotc(&x, &y, &zdotc, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxvector y;
        float ydata[][2] = {{3.0f, 2.0f}, {1.0f, 0.0f}, {1.0f, 0.0f}};
        err = mtxvector_init_coordinate_complex_single(&y, size, nnz, idx, ydata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        float sdot;
        err = mtxvector_sdot(&x, &y, &sdot, NULL);
        TEST_ASSERT_EQ_MSG(MTX_ERR_INVALID_FIELD, err, "%s", mtxstrerror(err));
        double ddot;
        err = mtxvector_ddot(&x, &y, &ddot, NULL);
        TEST_ASSERT_EQ_MSG(MTX_ERR_INVALID_FIELD, err, "%s", mtxstrerror(err));
        float cdotu[2];
        err = mtxvector_cdotu(&x, &y, &cdotu, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(5.0f, cdotu[0]); TEST_ASSERT_EQ(7.0f, cdotu[1]);
        float cdotc[2];
        err = mtxvector_cdotc(&x, &y, &cdotc, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, cdotc[0]); TEST_ASSERT_EQ(-3.0f, cdotc[1]);
        double zdotu[2];
        err = mtxvector_zdotu(&x, &y, &zdotu, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(5.0, zdotu[0]); TEST_ASSERT_EQ(7.0, zdotu[1]);
        double zdotc[2];
        err = mtxvector_zdotc(&x, &y, &zdotc, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxvector y;
        double ydata[][2] = {{3.0, 2.0}, {1.0, 0.0}, {1.0, 0.0}};
        err = mtxvector_init_coordinate_complex_double(&y, size, nnz, idx, ydata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        float sdot;
        err = mtxvector_sdot(&x, &y, &sdot, NULL);
        TEST_ASSERT_EQ_MSG(MTX_ERR_INVALID_FIELD, err, "%s", mtxstrerror(err));
        double ddot;
        err = mtxvector_ddot(&x, &y, &ddot, NULL);
        TEST_ASSERT_EQ_MSG(MTX_ERR_INVALID_FIELD, err, "%s", mtxstrerror(err));
        float cdotu[2];
        err = mtxvector_cdotu(&x, &y, &cdotu, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(5.0f, cdotu[0]); TEST_ASSERT_EQ(7.0f, cdotu[1]);
        float cdotc[2];
        err = mtxvector_cdotc(&x, &y, &cdotc, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, cdotc[0]); TEST_ASSERT_EQ(-3.0f, cdotc[1]);
        double zdotu[2];
        err = mtxvector_zdotu(&x, &y, &zdotu, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(5.0, zdotu[0]); TEST_ASSERT_EQ(7.0, zdotu[1]);
        double zdotc[2];
        err = mtxvector_zdotc(&x, &y, &zdotc, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxvector y;
        int32_t ydata[] = {3, 2, 1, 0, 1};
        err = mtxvector_init_coordinate_integer_single(&y, size, nnz, idx, ydata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        float sdot;
        err = mtxvector_sdot(&x, &y, &sdot, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, sdot);
        double ddot;
        err = mtxvector_ddot(&x, &y, &ddot, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, ddot);
        float cdotu[2];
        err = mtxvector_cdotu(&x, &y, &cdotu, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, cdotu[0]); TEST_ASSERT_EQ(0.0f, cdotu[1]);
        float cdotc[2];
        err = mtxvector_cdotc(&x, &y, &cdotc, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, cdotc[0]); TEST_ASSERT_EQ(0.0f, cdotc[1]);
        double zdotu[2];
        err = mtxvector_zdotu(&x, &y, &zdotu, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, zdotu[0]); TEST_ASSERT_EQ(0.0, zdotu[1]);
        double zdotc[2];
        err = mtxvector_zdotc(&x, &y, &zdotc, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxvector y;
        int64_t ydata[] = {3, 2, 1, 0, 1};
        err = mtxvector_init_coordinate_integer_double(&y, size, nnz, idx, ydata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        float sdot;
        err = mtxvector_sdot(&x, &y, &sdot, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, sdot);
        double ddot;
        err = mtxvector_ddot(&x, &y, &ddot, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, ddot);
        float cdotu[2];
        err = mtxvector_cdotu(&x, &y, &cdotu, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, cdotu[0]); TEST_ASSERT_EQ(0.0f, cdotu[1]);
        float cdotc[2];
        err = mtxvector_cdotc(&x, &y, &cdotc, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, cdotc[0]); TEST_ASSERT_EQ(0.0f, cdotc[1]);
        double zdotu[2];
        err = mtxvector_zdotu(&x, &y, &zdotu, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, zdotu[0]); TEST_ASSERT_EQ(0.0, zdotu[1]);
        double zdotc[2];
        err = mtxvector_zdotc(&x, &y, &zdotc, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        float snrm2;
        err = mtxvector_snrm2(&x, &snrm2, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(4.0f, snrm2);
        double dnrm2;
        err = mtxvector_dnrm2(&x, &dnrm2, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(4.0, dnrm2);
        mtxvector_free(&x);
    }

    {
        struct mtxvector x;
        double data[] = {1.0, 1.0, 1.0, 2.0, 3.0};
        int size = sizeof(data) / sizeof(*data);
        err = mtxvector_init_array_real_double(&x, size, data);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        float snrm2;
        err = mtxvector_snrm2(&x, &snrm2, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(4.0f, snrm2);
        double dnrm2;
        err = mtxvector_dnrm2(&x, &dnrm2, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(4.0, dnrm2);
        mtxvector_free(&x);
    }

    {
        struct mtxvector x;
        float data[][2] = {{1.0f,1.0f}, {1.0f,2.0f}, {3.0f,0.0f}};
        int size = sizeof(data) / sizeof(*data);
        err = mtxvector_init_array_complex_single(&x, size, data);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        float snrm2;
        err = mtxvector_snrm2(&x, &snrm2, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(4.0f, snrm2);
        double dnrm2;
        err = mtxvector_dnrm2(&x, &dnrm2, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(4.0, dnrm2);
        mtxvector_free(&x);
    }

    {
        struct mtxvector x;
        double data[][2] = {{1.0,1.0}, {1.0,2.0}, {3.0,0.0}};
        int size = sizeof(data) / sizeof(*data);
        err = mtxvector_init_array_complex_double(&x, size, data);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        float snrm2;
        err = mtxvector_snrm2(&x, &snrm2, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(4.0f, snrm2);
        double dnrm2;
        err = mtxvector_dnrm2(&x, &dnrm2, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        float snrm2;
        err = mtxvector_snrm2(&x, &snrm2, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(4.0f, snrm2);
        double dnrm2;
        err = mtxvector_dnrm2(&x, &dnrm2, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        float snrm2;
        err = mtxvector_snrm2(&x, &snrm2, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(4.0f, snrm2);
        double dnrm2;
        err = mtxvector_dnrm2(&x, &dnrm2, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        float snrm2;
        err = mtxvector_snrm2(&x, &snrm2, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(4.0f, snrm2);
        double dnrm2;
        err = mtxvector_dnrm2(&x, &dnrm2, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        float snrm2;
        err = mtxvector_snrm2(&x, &snrm2, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(4.0f, snrm2);
        double dnrm2;
        err = mtxvector_dnrm2(&x, &dnrm2, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(4.0, dnrm2);
        mtxvector_free(&x);
    }
    return TEST_SUCCESS;
}

/**
 * `test_mtxvector_asum()' tests computing the sum of absolute values
 * of vectors.
 */
int test_mtxvector_asum(void)
{
    int err;

    /*
     * Array formats
     */

    {
        struct mtxvector x;
        float data[] = {-1.0f, 1.0f, 1.0f, 2.0f, 3.0f};
        int size = sizeof(data) / sizeof(*data);
        err = mtxvector_init_array_real_single(&x, size, data);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        float sasum;
        err = mtxvector_sasum(&x, &sasum, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(8.0f, sasum);
        double dasum;
        err = mtxvector_dasum(&x, &dasum, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(8.0, dasum);
        mtxvector_free(&x);
    }

    {
        struct mtxvector x;
        double data[] = {-1.0, 1.0, 1.0, 2.0, 3.0};
        int size = sizeof(data) / sizeof(*data);
        err = mtxvector_init_array_real_double(&x, size, data);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        float sasum;
        err = mtxvector_sasum(&x, &sasum, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(8.0f, sasum);
        double dasum;
        err = mtxvector_dasum(&x, &dasum, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(8.0, dasum);
        mtxvector_free(&x);
    }

    {
        struct mtxvector x;
        float data[][2] = {{-1.0f,-1.0f}, {1.0f,2.0f}, {3.0f,0.0f}};
        int size = sizeof(data) / sizeof(*data);
        err = mtxvector_init_array_complex_single(&x, size, data);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        float sasum;
        err = mtxvector_sasum(&x, &sasum, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(8.0f, sasum);
        double dasum;
        err = mtxvector_dasum(&x, &dasum, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(8.0, dasum);
        mtxvector_free(&x);
    }

    {
        struct mtxvector x;
        double data[][2] = {{-1.0,-1.0}, {1.0,2.0}, {3.0,0.0}};
        int size = sizeof(data) / sizeof(*data);
        err = mtxvector_init_array_complex_double(&x, size, data);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        float sasum;
        err = mtxvector_sasum(&x, &sasum, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(8.0f, sasum);
        double dasum;
        err = mtxvector_dasum(&x, &dasum, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(8.0, dasum);
        mtxvector_free(&x);
    }

    /*
     * Coordinate formats
     */

    {
        struct mtxvector x;
        int size = 12;
        int indices[] = {0, 3, 5, 6, 9};
        float data[] = {-1.0f, 1.0f, 1.0f, 2.0f, 3.0f};
        int num_nonzeros = sizeof(data) / sizeof(*data);
        err = mtxvector_init_coordinate_real_single(
            &x, size, num_nonzeros, indices, data);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        float sasum;
        err = mtxvector_sasum(&x, &sasum, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(8.0f, sasum);
        double dasum;
        err = mtxvector_dasum(&x, &dasum, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(8.0, dasum);
        mtxvector_free(&x);
    }

    {
        struct mtxvector x;
        int size = 12;
        int indices[] = {0, 3, 5, 6, 9};
        double data[] = {-1.0, 1.0, 1.0, 2.0, 3.0};
        int num_nonzeros = sizeof(data) / sizeof(*data);
        err = mtxvector_init_coordinate_real_double(
            &x, size, num_nonzeros, indices, data);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        float sasum;
        err = mtxvector_sasum(&x, &sasum, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(8.0f, sasum);
        double dasum;
        err = mtxvector_dasum(&x, &dasum, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(8.0, dasum);
        mtxvector_free(&x);
    }

    {
        struct mtxvector x;
        int size = 12;
        int indices[] = {0, 3, 5, 6, 9};
        float data[][2] = {{-1.0f,-1.0f}, {1.0f,2.0f}, {3.0f,0.0f}};
        int64_t num_nonzeros = sizeof(data) / sizeof(*data);
        err = mtxvector_init_coordinate_complex_single(
            &x, size, num_nonzeros, indices, data);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        float sasum;
        err = mtxvector_sasum(&x, &sasum, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(8.0f, sasum);
        double dasum;
        err = mtxvector_dasum(&x, &dasum, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(8.0, dasum);
        mtxvector_free(&x);
    }

    {
        struct mtxvector x;
        int size = 12;
        int indices[] = {0, 3, 5, 6, 9};
        double data[][2] = {{-1.0,-1.0}, {1.0,2.0}, {3.0,0.0}};
        int64_t num_nonzeros = sizeof(data) / sizeof(*data);
        err = mtxvector_init_coordinate_complex_double(
            &x, size, num_nonzeros, indices, data);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        float sasum;
        err = mtxvector_sasum(&x, &sasum, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(8.0f, sasum);
        double dasum;
        err = mtxvector_dasum(&x, &dasum, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(8.0, dasum);
        mtxvector_free(&x);
    }
    return TEST_SUCCESS;
}

/**
 * `test_mtxvector_iamax()' tests computing the sum of absolute values
 * of vectors.
 */
int test_mtxvector_iamax(void)
{
    int err;

    /*
     * Array formats
     */

    {
        struct mtxvector x;
        float data[] = {-1.0f, 1.0f, 3.0f, 2.0f, 3.0f};
        int size = sizeof(data) / sizeof(*data);
        err = mtxvector_init_array_real_single(&x, size, data);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        int iamax;
        err = mtxvector_iamax(&x, &iamax);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(2, iamax);
        mtxvector_free(&x);
    }

    {
        struct mtxvector x;
        double data[] = {-1.0, 1.0, 3.0, 2.0, 3.0};
        int size = sizeof(data) / sizeof(*data);
        err = mtxvector_init_array_real_double(&x, size, data);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        int iamax;
        err = mtxvector_iamax(&x, &iamax);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(2, iamax);
        mtxvector_free(&x);
    }

    {
        struct mtxvector x;
        float data[][2] = {{-1.0f,-1.0f}, {1.0f,2.0f}, {3.0f,0.0f}};
        int size = sizeof(data) / sizeof(*data);
        err = mtxvector_init_array_complex_single(&x, size, data);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        int iamax;
        err = mtxvector_iamax(&x, &iamax);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(1, iamax);
        mtxvector_free(&x);
    }

    {
        struct mtxvector x;
        double data[][2] = {{-1.0,-1.0}, {1.0,2.0}, {3.0,0.0}};
        int size = sizeof(data) / sizeof(*data);
        err = mtxvector_init_array_complex_double(&x, size, data);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        int iamax;
        err = mtxvector_iamax(&x, &iamax);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(1, iamax);
        mtxvector_free(&x);
    }

    /*
     * Coordinate formats
     */

    {
        struct mtxvector x;
        int size = 12;
        int indices[] = {0, 3, 5, 6, 9};
        float data[] = {-1.0f, 1.0f, 3.0f, 2.0f, 3.0f};
        int num_nonzeros = sizeof(data) / sizeof(*data);
        err = mtxvector_init_coordinate_real_single(
            &x, size, num_nonzeros, indices, data);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        int iamax;
        err = mtxvector_iamax(&x, &iamax);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(2, iamax);
        mtxvector_free(&x);
    }

    {
        struct mtxvector x;
        int size = 12;
        int indices[] = {0, 3, 5, 6, 9};
        double data[] = {-1.0, 1.0, 3.0, 2.0, 3.0};
        int num_nonzeros = sizeof(data) / sizeof(*data);
        err = mtxvector_init_coordinate_real_double(
            &x, size, num_nonzeros, indices, data);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        int iamax;
        err = mtxvector_iamax(&x, &iamax);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(2, iamax);
        mtxvector_free(&x);
    }

    {
        struct mtxvector x;
        int size = 12;
        int indices[] = {0, 3, 5, 6, 9};
        float data[][2] = {{-1.0f,-1.0f}, {1.0f,2.0f}, {3.0f,0.0f}};
        int64_t num_nonzeros = sizeof(data) / sizeof(*data);
        err = mtxvector_init_coordinate_complex_single(
            &x, size, num_nonzeros, indices, data);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        int iamax;
        err = mtxvector_iamax(&x, &iamax);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(1, iamax);
        mtxvector_free(&x);
    }

    {
        struct mtxvector x;
        int size = 12;
        int indices[] = {0, 3, 5, 6, 9};
        double data[][2] = {{-1.0,-1.0}, {1.0,2.0}, {3.0,0.0}};
        int64_t num_nonzeros = sizeof(data) / sizeof(*data);
        err = mtxvector_init_coordinate_complex_double(
            &x, size, num_nonzeros, indices, data);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        int iamax;
        err = mtxvector_iamax(&x, &iamax);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(1, iamax);
        mtxvector_free(&x);
    }
    return TEST_SUCCESS;
}

/**
 * `test_mtxvector_scal()' tests scaling vectors by a constant.
 */
int test_mtxvector_scal(void)
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
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_sscal(2.0f, &x, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(2.0f, x.storage.array.data.real_single[0]);
        TEST_ASSERT_EQ(2.0f, x.storage.array.data.real_single[1]);
        TEST_ASSERT_EQ(2.0f, x.storage.array.data.real_single[2]);
        TEST_ASSERT_EQ(4.0f, x.storage.array.data.real_single[3]);
        TEST_ASSERT_EQ(6.0f, x.storage.array.data.real_single[4]);
        err = mtxvector_dscal(2.0, &x, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(4.0f, x.storage.array.data.real_single[0]);
        TEST_ASSERT_EQ(4.0f, x.storage.array.data.real_single[1]);
        TEST_ASSERT_EQ(4.0f, x.storage.array.data.real_single[2]);
        TEST_ASSERT_EQ(8.0f, x.storage.array.data.real_single[3]);
        TEST_ASSERT_EQ(12.0f, x.storage.array.data.real_single[4]);
        mtxvector_free(&x);
    }

    {
        struct mtxvector x;
        double data[] = {1.0, 1.0, 1.0, 2.0, 3.0};
        int size = sizeof(data) / sizeof(*data);
        err = mtxvector_init_array_real_double(&x, size, data);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_sscal(2.0f, &x, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(2.0, x.storage.array.data.real_double[0]);
        TEST_ASSERT_EQ(2.0, x.storage.array.data.real_double[1]);
        TEST_ASSERT_EQ(2.0, x.storage.array.data.real_double[2]);
        TEST_ASSERT_EQ(4.0, x.storage.array.data.real_double[3]);
        TEST_ASSERT_EQ(6.0, x.storage.array.data.real_double[4]);
        err = mtxvector_dscal(2.0, &x, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(4.0, x.storage.array.data.real_double[0]);
        TEST_ASSERT_EQ(4.0, x.storage.array.data.real_double[1]);
        TEST_ASSERT_EQ(4.0, x.storage.array.data.real_double[2]);
        TEST_ASSERT_EQ(8.0, x.storage.array.data.real_double[3]);
        TEST_ASSERT_EQ(12.0, x.storage.array.data.real_double[4]);
        mtxvector_free(&x);
    }

    {
        struct mtxvector x;
        float data[][2] = {{1.0f,1.0f}, {1.0f,2.0f}, {3.0f,0.0f}};
        int size = sizeof(data) / sizeof(*data);
        err = mtxvector_init_array_complex_single(&x, size, data);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_sscal(2.0f, &x, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(2.0f, x.storage.array.data.complex_single[0][0]);
        TEST_ASSERT_EQ(2.0f, x.storage.array.data.complex_single[0][1]);
        TEST_ASSERT_EQ(2.0f, x.storage.array.data.complex_single[1][0]);
        TEST_ASSERT_EQ(4.0f, x.storage.array.data.complex_single[1][1]);
        TEST_ASSERT_EQ(6.0f, x.storage.array.data.complex_single[2][0]);
        TEST_ASSERT_EQ(0.0f, x.storage.array.data.complex_single[2][1]);
        err = mtxvector_dscal(2.0, &x, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(4.0f, x.storage.array.data.complex_single[0][0]);
        TEST_ASSERT_EQ(4.0f, x.storage.array.data.complex_single[0][1]);
        TEST_ASSERT_EQ(4.0f, x.storage.array.data.complex_single[1][0]);
        TEST_ASSERT_EQ(8.0f, x.storage.array.data.complex_single[1][1]);
        TEST_ASSERT_EQ(12.0f, x.storage.array.data.complex_single[2][0]);
        TEST_ASSERT_EQ(0.0f, x.storage.array.data.complex_single[2][1]);
        float as[2] = {2, 3};
        err = mtxvector_cscal(as, &x, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( -4.0f, x.storage.array.data.complex_single[0][0]);
        TEST_ASSERT_EQ( 20.0f, x.storage.array.data.complex_single[0][1]);
        TEST_ASSERT_EQ(-16.0f, x.storage.array.data.complex_single[1][0]);
        TEST_ASSERT_EQ( 28.0f, x.storage.array.data.complex_single[1][1]);
        TEST_ASSERT_EQ( 24.0f, x.storage.array.data.complex_single[2][0]);
        TEST_ASSERT_EQ( 36.0f, x.storage.array.data.complex_single[2][1]);
        double ad[2] = {2, 3};
        err = mtxvector_zscal(ad, &x, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( -68.0f, x.storage.array.data.complex_single[0][0]);
        TEST_ASSERT_EQ(  28.0f, x.storage.array.data.complex_single[0][1]);
        TEST_ASSERT_EQ(-116.0f, x.storage.array.data.complex_single[1][0]);
        TEST_ASSERT_EQ(   8.0f, x.storage.array.data.complex_single[1][1]);
        TEST_ASSERT_EQ( -60.0f, x.storage.array.data.complex_single[2][0]);
        TEST_ASSERT_EQ( 144.0f, x.storage.array.data.complex_single[2][1]);
        mtxvector_free(&x);
    }

    {
        struct mtxvector x;
        double data[][2] = {{1.0,1.0}, {1.0,2.0}, {3.0,0.0}};
        int size = sizeof(data) / sizeof(*data);
        err = mtxvector_init_array_complex_double(&x, size, data);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_sscal(2.0f, &x, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(2.0, x.storage.array.data.complex_double[0][0]);
        TEST_ASSERT_EQ(2.0, x.storage.array.data.complex_double[0][1]);
        TEST_ASSERT_EQ(2.0, x.storage.array.data.complex_double[1][0]);
        TEST_ASSERT_EQ(4.0, x.storage.array.data.complex_double[1][1]);
        TEST_ASSERT_EQ(6.0, x.storage.array.data.complex_double[2][0]);
        TEST_ASSERT_EQ(0.0, x.storage.array.data.complex_double[2][1]);
        err = mtxvector_dscal(2.0, &x, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(4.0, x.storage.array.data.complex_double[0][0]);
        TEST_ASSERT_EQ(4.0, x.storage.array.data.complex_double[0][1]);
        TEST_ASSERT_EQ(4.0, x.storage.array.data.complex_double[1][0]);
        TEST_ASSERT_EQ(8.0, x.storage.array.data.complex_double[1][1]);
        TEST_ASSERT_EQ(12.0, x.storage.array.data.complex_double[2][0]);
        TEST_ASSERT_EQ(0.0, x.storage.array.data.complex_double[2][1]);
        float as[2] = {2, 3};
        err = mtxvector_cscal(as, &x, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( -4.0, x.storage.array.data.complex_double[0][0]);
        TEST_ASSERT_EQ( 20.0, x.storage.array.data.complex_double[0][1]);
        TEST_ASSERT_EQ(-16.0, x.storage.array.data.complex_double[1][0]);
        TEST_ASSERT_EQ( 28.0, x.storage.array.data.complex_double[1][1]);
        TEST_ASSERT_EQ( 24.0, x.storage.array.data.complex_double[2][0]);
        TEST_ASSERT_EQ( 36.0, x.storage.array.data.complex_double[2][1]);
        double ad[2] = {2, 3};
        err = mtxvector_zscal(ad, &x, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( -68.0, x.storage.array.data.complex_double[0][0]);
        TEST_ASSERT_EQ(  28.0, x.storage.array.data.complex_double[0][1]);
        TEST_ASSERT_EQ(-116.0, x.storage.array.data.complex_double[1][0]);
        TEST_ASSERT_EQ(   8.0, x.storage.array.data.complex_double[1][1]);
        TEST_ASSERT_EQ( -60.0, x.storage.array.data.complex_double[2][0]);
        TEST_ASSERT_EQ( 144.0, x.storage.array.data.complex_double[2][1]);
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
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_sscal(2.0f, &x, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(2.0f, x.storage.coordinate.data.real_single[0]);
        TEST_ASSERT_EQ(2.0f, x.storage.coordinate.data.real_single[1]);
        TEST_ASSERT_EQ(2.0f, x.storage.coordinate.data.real_single[2]);
        TEST_ASSERT_EQ(4.0f, x.storage.coordinate.data.real_single[3]);
        TEST_ASSERT_EQ(6.0f, x.storage.coordinate.data.real_single[4]);
        err = mtxvector_dscal(2.0, &x, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(4.0f, x.storage.coordinate.data.real_single[0]);
        TEST_ASSERT_EQ(4.0f, x.storage.coordinate.data.real_single[1]);
        TEST_ASSERT_EQ(4.0f, x.storage.coordinate.data.real_single[2]);
        TEST_ASSERT_EQ(8.0f, x.storage.coordinate.data.real_single[3]);
        TEST_ASSERT_EQ(12.0f, x.storage.coordinate.data.real_single[4]);
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
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_sscal(2.0f, &x, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(2.0f, x.storage.coordinate.data.real_double[0]);
        TEST_ASSERT_EQ(2.0f, x.storage.coordinate.data.real_double[1]);
        TEST_ASSERT_EQ(2.0f, x.storage.coordinate.data.real_double[2]);
        TEST_ASSERT_EQ(4.0f, x.storage.coordinate.data.real_double[3]);
        TEST_ASSERT_EQ(6.0f, x.storage.coordinate.data.real_double[4]);
        err = mtxvector_dscal(2.0, &x, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(4.0f, x.storage.coordinate.data.real_double[0]);
        TEST_ASSERT_EQ(4.0f, x.storage.coordinate.data.real_double[1]);
        TEST_ASSERT_EQ(4.0f, x.storage.coordinate.data.real_double[2]);
        TEST_ASSERT_EQ(8.0f, x.storage.coordinate.data.real_double[3]);
        TEST_ASSERT_EQ(12.0f, x.storage.coordinate.data.real_double[4]);
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
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_sscal(2.0f, &x, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(2.0f, x.storage.coordinate.data.complex_single[0][0]);
        TEST_ASSERT_EQ(2.0f, x.storage.coordinate.data.complex_single[0][1]);
        TEST_ASSERT_EQ(2.0f, x.storage.coordinate.data.complex_single[1][0]);
        TEST_ASSERT_EQ(4.0f, x.storage.coordinate.data.complex_single[1][1]);
        TEST_ASSERT_EQ(6.0f, x.storage.coordinate.data.complex_single[2][0]);
        TEST_ASSERT_EQ(0.0f, x.storage.coordinate.data.complex_single[2][1]);
        err = mtxvector_dscal(2.0, &x, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(4.0f, x.storage.coordinate.data.complex_single[0][0]);
        TEST_ASSERT_EQ(4.0f, x.storage.coordinate.data.complex_single[0][1]);
        TEST_ASSERT_EQ(4.0f, x.storage.coordinate.data.complex_single[1][0]);
        TEST_ASSERT_EQ(8.0f, x.storage.coordinate.data.complex_single[1][1]);
        TEST_ASSERT_EQ(12.0f, x.storage.coordinate.data.complex_single[2][0]);
        TEST_ASSERT_EQ(0.0f, x.storage.coordinate.data.complex_single[2][1]);
        float as[2] = {2, 3};
        err = mtxvector_cscal(as, &x, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( -4.0f, x.storage.coordinate.data.complex_single[0][0]);
        TEST_ASSERT_EQ( 20.0f, x.storage.coordinate.data.complex_single[0][1]);
        TEST_ASSERT_EQ(-16.0f, x.storage.coordinate.data.complex_single[1][0]);
        TEST_ASSERT_EQ( 28.0f, x.storage.coordinate.data.complex_single[1][1]);
        TEST_ASSERT_EQ( 24.0f, x.storage.coordinate.data.complex_single[2][0]);
        TEST_ASSERT_EQ( 36.0f, x.storage.coordinate.data.complex_single[2][1]);
        double ad[2] = {2, 3};
        err = mtxvector_zscal(ad, &x, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( -68.0f, x.storage.coordinate.data.complex_single[0][0]);
        TEST_ASSERT_EQ(  28.0f, x.storage.coordinate.data.complex_single[0][1]);
        TEST_ASSERT_EQ(-116.0f, x.storage.coordinate.data.complex_single[1][0]);
        TEST_ASSERT_EQ(   8.0f, x.storage.coordinate.data.complex_single[1][1]);
        TEST_ASSERT_EQ( -60.0f, x.storage.coordinate.data.complex_single[2][0]);
        TEST_ASSERT_EQ( 144.0f, x.storage.coordinate.data.complex_single[2][1]);
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
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_sscal(2.0f, &x, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(2.0f, x.storage.coordinate.data.complex_double[0][0]);
        TEST_ASSERT_EQ(2.0f, x.storage.coordinate.data.complex_double[0][1]);
        TEST_ASSERT_EQ(2.0f, x.storage.coordinate.data.complex_double[1][0]);
        TEST_ASSERT_EQ(4.0f, x.storage.coordinate.data.complex_double[1][1]);
        TEST_ASSERT_EQ(6.0f, x.storage.coordinate.data.complex_double[2][0]);
        TEST_ASSERT_EQ(0.0f, x.storage.coordinate.data.complex_double[2][1]);
        err = mtxvector_dscal(2.0, &x, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(4.0f, x.storage.coordinate.data.complex_double[0][0]);
        TEST_ASSERT_EQ(4.0f, x.storage.coordinate.data.complex_double[0][1]);
        TEST_ASSERT_EQ(4.0f, x.storage.coordinate.data.complex_double[1][0]);
        TEST_ASSERT_EQ(8.0f, x.storage.coordinate.data.complex_double[1][1]);
        TEST_ASSERT_EQ(12.0f, x.storage.coordinate.data.complex_double[2][0]);
        TEST_ASSERT_EQ(0.0f, x.storage.coordinate.data.complex_double[2][1]);
        float as[2] = {2, 3};
        err = mtxvector_cscal(as, &x, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( -4.0, x.storage.coordinate.data.complex_double[0][0]);
        TEST_ASSERT_EQ( 20.0, x.storage.coordinate.data.complex_double[0][1]);
        TEST_ASSERT_EQ(-16.0, x.storage.coordinate.data.complex_double[1][0]);
        TEST_ASSERT_EQ( 28.0, x.storage.coordinate.data.complex_double[1][1]);
        TEST_ASSERT_EQ( 24.0, x.storage.coordinate.data.complex_double[2][0]);
        TEST_ASSERT_EQ( 36.0, x.storage.coordinate.data.complex_double[2][1]);
        double ad[2] = {2, 3};
        err = mtxvector_zscal(ad, &x, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( -68.0, x.storage.coordinate.data.complex_double[0][0]);
        TEST_ASSERT_EQ(  28.0, x.storage.coordinate.data.complex_double[0][1]);
        TEST_ASSERT_EQ(-116.0, x.storage.coordinate.data.complex_double[1][0]);
        TEST_ASSERT_EQ(   8.0, x.storage.coordinate.data.complex_double[1][1]);
        TEST_ASSERT_EQ( -60.0, x.storage.coordinate.data.complex_double[2][0]);
        TEST_ASSERT_EQ( 144.0, x.storage.coordinate.data.complex_double[2][1]);
        mtxvector_free(&x);
    }
    return TEST_SUCCESS;
}

/**
 * `test_mtxvector_axpy()' tests multiplying a vector by a constant
 * and adding the result to another vector.
 */
int test_mtxvector_axpy(void)
{
    int err;

    /*
     * Array formats
     */

    {
        struct mtxvector x;
        struct mtxvector y;
        float xdata[] = {1.0f, 1.0f, 1.0f, 2.0f, 3.0f};
        float ydata[] = {2.0f, 1.0f, 0.0f, 2.0f, 1.0f};
        int size = sizeof(xdata) / sizeof(*xdata);
        err = mtxvector_init_array_real_single(&x, size, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_init_array_real_single(&y, size, ydata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_saxpy(2.0f, &x, &y, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(4.0f, y.storage.array.data.real_single[0]);
        TEST_ASSERT_EQ(3.0f, y.storage.array.data.real_single[1]);
        TEST_ASSERT_EQ(2.0f, y.storage.array.data.real_single[2]);
        TEST_ASSERT_EQ(6.0f, y.storage.array.data.real_single[3]);
        TEST_ASSERT_EQ(7.0f, y.storage.array.data.real_single[4]);
        err = mtxvector_daxpy(2.0, &x, &y, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(6.0f, y.storage.array.data.real_single[0]);
        TEST_ASSERT_EQ(5.0f, y.storage.array.data.real_single[1]);
        TEST_ASSERT_EQ(4.0f, y.storage.array.data.real_single[2]);
        TEST_ASSERT_EQ(10.0f, y.storage.array.data.real_single[3]);
        TEST_ASSERT_EQ(13.0f, y.storage.array.data.real_single[4]);
        err = mtxvector_saypx(2.0f, &y, &x, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(13.0f, y.storage.array.data.real_single[0]);
        TEST_ASSERT_EQ(11.0f, y.storage.array.data.real_single[1]);
        TEST_ASSERT_EQ(9.0f, y.storage.array.data.real_single[2]);
        TEST_ASSERT_EQ(22.0f, y.storage.array.data.real_single[3]);
        TEST_ASSERT_EQ(29.0f, y.storage.array.data.real_single[4]);
        err = mtxvector_daypx(2.0, &y, &x, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(27.0f, y.storage.array.data.real_single[0]);
        TEST_ASSERT_EQ(23.0f, y.storage.array.data.real_single[1]);
        TEST_ASSERT_EQ(19.0f, y.storage.array.data.real_single[2]);
        TEST_ASSERT_EQ(46.0f, y.storage.array.data.real_single[3]);
        TEST_ASSERT_EQ(61.0f, y.storage.array.data.real_single[4]);
        mtxvector_free(&y);
        mtxvector_free(&x);
    }

    {
        struct mtxvector x;
        struct mtxvector y;
        double xdata[] = {1.0, 1.0, 1.0, 2.0, 3.0};
        double ydata[] = {2.0, 1.0, 0.0, 2.0, 1.0};
        int size = sizeof(xdata) / sizeof(*xdata);
        err = mtxvector_init_array_real_double(&x, size, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_init_array_real_double(&y, size, ydata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_saxpy(2.0f, &x, &y, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(4.0, y.storage.array.data.real_double[0]);
        TEST_ASSERT_EQ(3.0, y.storage.array.data.real_double[1]);
        TEST_ASSERT_EQ(2.0, y.storage.array.data.real_double[2]);
        TEST_ASSERT_EQ(6.0, y.storage.array.data.real_double[3]);
        TEST_ASSERT_EQ(7.0, y.storage.array.data.real_double[4]);
        err = mtxvector_daxpy(2.0, &x, &y, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(6.0, y.storage.array.data.real_double[0]);
        TEST_ASSERT_EQ(5.0, y.storage.array.data.real_double[1]);
        TEST_ASSERT_EQ(4.0, y.storage.array.data.real_double[2]);
        TEST_ASSERT_EQ(10.0, y.storage.array.data.real_double[3]);
        TEST_ASSERT_EQ(13.0, y.storage.array.data.real_double[4]);
        err = mtxvector_saypx(2.0f, &y, &x, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(13.0, y.storage.array.data.real_double[0]);
        TEST_ASSERT_EQ(11.0, y.storage.array.data.real_double[1]);
        TEST_ASSERT_EQ(9.0, y.storage.array.data.real_double[2]);
        TEST_ASSERT_EQ(22.0, y.storage.array.data.real_double[3]);
        TEST_ASSERT_EQ(29.0, y.storage.array.data.real_double[4]);
        err = mtxvector_daypx(2.0, &y, &x, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(27.0, y.storage.array.data.real_double[0]);
        TEST_ASSERT_EQ(23.0, y.storage.array.data.real_double[1]);
        TEST_ASSERT_EQ(19.0, y.storage.array.data.real_double[2]);
        TEST_ASSERT_EQ(46.0, y.storage.array.data.real_double[3]);
        TEST_ASSERT_EQ(61.0, y.storage.array.data.real_double[4]);
        mtxvector_free(&y);
        mtxvector_free(&x);
    }

    {
        struct mtxvector x;
        struct mtxvector y;
        float xdata[][2] = {{1.0f,1.0f}, {1.0f,2.0f}, {3.0f,0.0f}};
        float ydata[][2] = {{2.0f,1.0f}, {0.0f,2.0f}, {1.0f,0.0f}};
        int size = sizeof(xdata) / sizeof(*xdata);
        err = mtxvector_init_array_complex_single(&x, size, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_init_array_complex_single(&y, size, ydata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_saxpy(2.0f, &x, &y, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(4.0f, y.storage.array.data.complex_single[0][0]);
        TEST_ASSERT_EQ(3.0f, y.storage.array.data.complex_single[0][1]);
        TEST_ASSERT_EQ(2.0f, y.storage.array.data.complex_single[1][0]);
        TEST_ASSERT_EQ(6.0f, y.storage.array.data.complex_single[1][1]);
        TEST_ASSERT_EQ(7.0f, y.storage.array.data.complex_single[2][0]);
        TEST_ASSERT_EQ(0.0f, y.storage.array.data.complex_single[2][1]);
        err = mtxvector_daxpy(2.0, &x, &y, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(6.0f, y.storage.array.data.complex_single[0][0]);
        TEST_ASSERT_EQ(5.0f, y.storage.array.data.complex_single[0][1]);
        TEST_ASSERT_EQ(4.0f, y.storage.array.data.complex_single[1][0]);
        TEST_ASSERT_EQ(10.0f, y.storage.array.data.complex_single[1][1]);
        TEST_ASSERT_EQ(13.0f, y.storage.array.data.complex_single[2][0]);
        TEST_ASSERT_EQ(0.0f, y.storage.array.data.complex_single[2][1]);
        err = mtxvector_saypx(2.0f, &y, &x, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(13.0f, y.storage.array.data.complex_single[0][0]);
        TEST_ASSERT_EQ(11.0f, y.storage.array.data.complex_single[0][1]);
        TEST_ASSERT_EQ(9.0f, y.storage.array.data.complex_single[1][0]);
        TEST_ASSERT_EQ(22.0f, y.storage.array.data.complex_single[1][1]);
        TEST_ASSERT_EQ(29.0f, y.storage.array.data.complex_single[2][0]);
        TEST_ASSERT_EQ(0.0f, y.storage.array.data.complex_single[2][1]);
        err = mtxvector_daypx(2.0, &y, &x, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(27.0f, y.storage.array.data.complex_single[0][0]);
        TEST_ASSERT_EQ(23.0f, y.storage.array.data.complex_single[0][1]);
        TEST_ASSERT_EQ(19.0f, y.storage.array.data.complex_single[1][0]);
        TEST_ASSERT_EQ(46.0f, y.storage.array.data.complex_single[1][1]);
        TEST_ASSERT_EQ(61.0f, y.storage.array.data.complex_single[2][0]);
        TEST_ASSERT_EQ(0.0f, y.storage.array.data.complex_single[2][1]);
        mtxvector_free(&y);
        mtxvector_free(&x);
    }

    {
        struct mtxvector x;
        struct mtxvector y;
        double xdata[][2] = {{1.0,1.0}, {1.0,2.0}, {3.0,0.0}};
        double ydata[][2] = {{2.0,1.0}, {0.0,2.0}, {1.0,0.0}};
        int size = sizeof(xdata) / sizeof(*xdata);
        err = mtxvector_init_array_complex_double(&x, size, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_init_array_complex_double(&y, size, ydata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_saxpy(2.0f, &x, &y, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(4.0, y.storage.array.data.complex_double[0][0]);
        TEST_ASSERT_EQ(3.0, y.storage.array.data.complex_double[0][1]);
        TEST_ASSERT_EQ(2.0, y.storage.array.data.complex_double[1][0]);
        TEST_ASSERT_EQ(6.0, y.storage.array.data.complex_double[1][1]);
        TEST_ASSERT_EQ(7.0, y.storage.array.data.complex_double[2][0]);
        TEST_ASSERT_EQ(0.0, y.storage.array.data.complex_double[2][1]);
        err = mtxvector_daxpy(2.0, &x, &y, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(6.0, y.storage.array.data.complex_double[0][0]);
        TEST_ASSERT_EQ(5.0, y.storage.array.data.complex_double[0][1]);
        TEST_ASSERT_EQ(4.0, y.storage.array.data.complex_double[1][0]);
        TEST_ASSERT_EQ(10.0, y.storage.array.data.complex_double[1][1]);
        TEST_ASSERT_EQ(13.0, y.storage.array.data.complex_double[2][0]);
        TEST_ASSERT_EQ(0.0, y.storage.array.data.complex_double[2][1]);
        err = mtxvector_saypx(2.0f, &y, &x, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(13.0, y.storage.array.data.complex_double[0][0]);
        TEST_ASSERT_EQ(11.0, y.storage.array.data.complex_double[0][1]);
        TEST_ASSERT_EQ(9.0, y.storage.array.data.complex_double[1][0]);
        TEST_ASSERT_EQ(22.0, y.storage.array.data.complex_double[1][1]);
        TEST_ASSERT_EQ(29.0, y.storage.array.data.complex_double[2][0]);
        TEST_ASSERT_EQ(0.0, y.storage.array.data.complex_double[2][1]);
        err = mtxvector_daypx(2.0, &y, &x, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(27.0, y.storage.array.data.complex_double[0][0]);
        TEST_ASSERT_EQ(23.0, y.storage.array.data.complex_double[0][1]);
        TEST_ASSERT_EQ(19.0, y.storage.array.data.complex_double[1][0]);
        TEST_ASSERT_EQ(46.0, y.storage.array.data.complex_double[1][1]);
        TEST_ASSERT_EQ(61.0, y.storage.array.data.complex_double[2][0]);
        TEST_ASSERT_EQ(0.0, y.storage.array.data.complex_double[2][1]);
        mtxvector_free(&y);
        mtxvector_free(&x);
    }

    /*
     * Coordinate formats
     */

    {
        struct mtxvector x;
        struct mtxvector y;
        int size = 12;
        int indices[] = {0, 3, 5, 6, 9};
        float xdata[] = {1.0f, 1.0f, 1.0f, 2.0f, 3.0f};
        float ydata[] = {2.0f, 1.0f, 0.0f, 2.0f, 1.0f};
        int num_nonzeros = sizeof(xdata) / sizeof(*xdata);
        err = mtxvector_init_coordinate_real_single(
            &x, size, num_nonzeros, indices, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_init_coordinate_real_single(
            &y, size, num_nonzeros, indices, ydata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_saxpy(2.0f, &x, &y, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(4.0f, y.storage.coordinate.data.real_single[0]);
        TEST_ASSERT_EQ(3.0f, y.storage.coordinate.data.real_single[1]);
        TEST_ASSERT_EQ(2.0f, y.storage.coordinate.data.real_single[2]);
        TEST_ASSERT_EQ(6.0f, y.storage.coordinate.data.real_single[3]);
        TEST_ASSERT_EQ(7.0f, y.storage.coordinate.data.real_single[4]);
        err = mtxvector_daxpy(2.0, &x, &y, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(6.0f, y.storage.coordinate.data.real_single[0]);
        TEST_ASSERT_EQ(5.0f, y.storage.coordinate.data.real_single[1]);
        TEST_ASSERT_EQ(4.0f, y.storage.coordinate.data.real_single[2]);
        TEST_ASSERT_EQ(10.0f, y.storage.coordinate.data.real_single[3]);
        TEST_ASSERT_EQ(13.0f, y.storage.coordinate.data.real_single[4]);
        err = mtxvector_saypx(2.0f, &y, &x, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(13.0f, y.storage.coordinate.data.real_single[0]);
        TEST_ASSERT_EQ(11.0f, y.storage.coordinate.data.real_single[1]);
        TEST_ASSERT_EQ(9.0f, y.storage.coordinate.data.real_single[2]);
        TEST_ASSERT_EQ(22.0f, y.storage.coordinate.data.real_single[3]);
        TEST_ASSERT_EQ(29.0f, y.storage.coordinate.data.real_single[4]);
        err = mtxvector_daypx(2.0, &y, &x, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(27.0f, y.storage.coordinate.data.real_single[0]);
        TEST_ASSERT_EQ(23.0f, y.storage.coordinate.data.real_single[1]);
        TEST_ASSERT_EQ(19.0f, y.storage.coordinate.data.real_single[2]);
        TEST_ASSERT_EQ(46.0f, y.storage.coordinate.data.real_single[3]);
        TEST_ASSERT_EQ(61.0f, y.storage.coordinate.data.real_single[4]);
        mtxvector_free(&y);
        mtxvector_free(&x);
    }

    {
        struct mtxvector x;
        struct mtxvector y;
        int size = 12;
        int indices[] = {0, 3, 5, 6, 9};
        double xdata[] = {1.0, 1.0, 1.0, 2.0, 3.0};
        double ydata[] = {2.0, 1.0, 0.0, 2.0, 1.0};
        int num_nonzeros = sizeof(xdata) / sizeof(*xdata);
        err = mtxvector_init_coordinate_real_double(
            &x, size, num_nonzeros, indices, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_init_coordinate_real_double(
            &y, size, num_nonzeros, indices, ydata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_saxpy(2.0f, &x, &y, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(4.0, y.storage.coordinate.data.real_double[0]);
        TEST_ASSERT_EQ(3.0, y.storage.coordinate.data.real_double[1]);
        TEST_ASSERT_EQ(2.0, y.storage.coordinate.data.real_double[2]);
        TEST_ASSERT_EQ(6.0, y.storage.coordinate.data.real_double[3]);
        TEST_ASSERT_EQ(7.0, y.storage.coordinate.data.real_double[4]);
        err = mtxvector_daxpy(2.0, &x, &y, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(6.0, y.storage.coordinate.data.real_double[0]);
        TEST_ASSERT_EQ(5.0, y.storage.coordinate.data.real_double[1]);
        TEST_ASSERT_EQ(4.0, y.storage.coordinate.data.real_double[2]);
        TEST_ASSERT_EQ(10.0, y.storage.coordinate.data.real_double[3]);
        TEST_ASSERT_EQ(13.0, y.storage.coordinate.data.real_double[4]);
        err = mtxvector_saypx(2.0f, &y, &x, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(13.0, y.storage.coordinate.data.real_double[0]);
        TEST_ASSERT_EQ(11.0, y.storage.coordinate.data.real_double[1]);
        TEST_ASSERT_EQ(9.0, y.storage.coordinate.data.real_double[2]);
        TEST_ASSERT_EQ(22.0, y.storage.coordinate.data.real_double[3]);
        TEST_ASSERT_EQ(29.0, y.storage.coordinate.data.real_double[4]);
        err = mtxvector_daypx(2.0, &y, &x, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(27.0, y.storage.coordinate.data.real_double[0]);
        TEST_ASSERT_EQ(23.0, y.storage.coordinate.data.real_double[1]);
        TEST_ASSERT_EQ(19.0, y.storage.coordinate.data.real_double[2]);
        TEST_ASSERT_EQ(46.0, y.storage.coordinate.data.real_double[3]);
        TEST_ASSERT_EQ(61.0, y.storage.coordinate.data.real_double[4]);
        mtxvector_free(&y);
        mtxvector_free(&x);
    }

    {
        struct mtxvector x;
        struct mtxvector y;
        int size = 12;
        int indices[] = {0, 3, 5, 6, 9};
        float xdata[][2] = {{1.0f,1.0f}, {1.0f,2.0f}, {3.0f,0.0f}};
        float ydata[][2] = {{2.0f,1.0f}, {0.0f,2.0f}, {1.0f,0.0f}};
        int64_t num_nonzeros = sizeof(xdata) / sizeof(*xdata);
        err = mtxvector_init_coordinate_complex_single(
            &x, size, num_nonzeros, indices, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_init_coordinate_complex_single(
            &y, size, num_nonzeros, indices, ydata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_saxpy(2.0f, &x, &y, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(4.0f, y.storage.coordinate.data.complex_single[0][0]);
        TEST_ASSERT_EQ(3.0f, y.storage.coordinate.data.complex_single[0][1]);
        TEST_ASSERT_EQ(2.0f, y.storage.coordinate.data.complex_single[1][0]);
        TEST_ASSERT_EQ(6.0f, y.storage.coordinate.data.complex_single[1][1]);
        TEST_ASSERT_EQ(7.0f, y.storage.coordinate.data.complex_single[2][0]);
        TEST_ASSERT_EQ(0.0f, y.storage.coordinate.data.complex_single[2][1]);
        err = mtxvector_daxpy(2.0, &x, &y, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(6.0f, y.storage.coordinate.data.complex_single[0][0]);
        TEST_ASSERT_EQ(5.0f, y.storage.coordinate.data.complex_single[0][1]);
        TEST_ASSERT_EQ(4.0f, y.storage.coordinate.data.complex_single[1][0]);
        TEST_ASSERT_EQ(10.0f, y.storage.coordinate.data.complex_single[1][1]);
        TEST_ASSERT_EQ(13.0f, y.storage.coordinate.data.complex_single[2][0]);
        TEST_ASSERT_EQ(0.0f, y.storage.coordinate.data.complex_single[2][1]);
        err = mtxvector_saypx(2.0f, &y, &x, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(13.0f, y.storage.coordinate.data.complex_single[0][0]);
        TEST_ASSERT_EQ(11.0f, y.storage.coordinate.data.complex_single[0][1]);
        TEST_ASSERT_EQ(9.0f, y.storage.coordinate.data.complex_single[1][0]);
        TEST_ASSERT_EQ(22.0f, y.storage.coordinate.data.complex_single[1][1]);
        TEST_ASSERT_EQ(29.0f, y.storage.coordinate.data.complex_single[2][0]);
        TEST_ASSERT_EQ(0.0f, y.storage.coordinate.data.complex_single[2][1]);
        err = mtxvector_daypx(2.0, &y, &x, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(27.0f, y.storage.coordinate.data.complex_single[0][0]);
        TEST_ASSERT_EQ(23.0f, y.storage.coordinate.data.complex_single[0][1]);
        TEST_ASSERT_EQ(19.0f, y.storage.coordinate.data.complex_single[1][0]);
        TEST_ASSERT_EQ(46.0f, y.storage.coordinate.data.complex_single[1][1]);
        TEST_ASSERT_EQ(61.0f, y.storage.coordinate.data.complex_single[2][0]);
        TEST_ASSERT_EQ(0.0f, y.storage.coordinate.data.complex_single[2][1]);
        mtxvector_free(&y);
        mtxvector_free(&x);
    }

    {
        struct mtxvector x;
        struct mtxvector y;
        int size = 12;
        int indices[] = {0, 3, 5, 6, 9};
        double xdata[][2] = {{1.0,1.0}, {1.0,2.0}, {3.0,0.0}};
        double ydata[][2] = {{2.0,1.0}, {0.0,2.0}, {1.0,0.0}};
        int64_t num_nonzeros = sizeof(xdata) / sizeof(*xdata);
        err = mtxvector_init_coordinate_complex_double(
            &x, size, num_nonzeros, indices, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_init_coordinate_complex_double(
            &y, size, num_nonzeros, indices, ydata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_saxpy(2.0f, &x, &y, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(4.0, y.storage.coordinate.data.complex_double[0][0]);
        TEST_ASSERT_EQ(3.0, y.storage.coordinate.data.complex_double[0][1]);
        TEST_ASSERT_EQ(2.0, y.storage.coordinate.data.complex_double[1][0]);
        TEST_ASSERT_EQ(6.0, y.storage.coordinate.data.complex_double[1][1]);
        TEST_ASSERT_EQ(7.0, y.storage.coordinate.data.complex_double[2][0]);
        TEST_ASSERT_EQ(0.0, y.storage.coordinate.data.complex_double[2][1]);
        err = mtxvector_daxpy(2.0, &x, &y, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(6.0, y.storage.coordinate.data.complex_double[0][0]);
        TEST_ASSERT_EQ(5.0, y.storage.coordinate.data.complex_double[0][1]);
        TEST_ASSERT_EQ(4.0, y.storage.coordinate.data.complex_double[1][0]);
        TEST_ASSERT_EQ(10.0, y.storage.coordinate.data.complex_double[1][1]);
        TEST_ASSERT_EQ(13.0, y.storage.coordinate.data.complex_double[2][0]);
        TEST_ASSERT_EQ(0.0, y.storage.coordinate.data.complex_double[2][1]);
        err = mtxvector_saypx(2.0f, &y, &x, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(13.0, y.storage.coordinate.data.complex_double[0][0]);
        TEST_ASSERT_EQ(11.0, y.storage.coordinate.data.complex_double[0][1]);
        TEST_ASSERT_EQ(9.0, y.storage.coordinate.data.complex_double[1][0]);
        TEST_ASSERT_EQ(22.0, y.storage.coordinate.data.complex_double[1][1]);
        TEST_ASSERT_EQ(29.0, y.storage.coordinate.data.complex_double[2][0]);
        TEST_ASSERT_EQ(0.0, y.storage.coordinate.data.complex_double[2][1]);
        err = mtxvector_daypx(2.0, &y, &x, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(27.0, y.storage.coordinate.data.complex_double[0][0]);
        TEST_ASSERT_EQ(23.0, y.storage.coordinate.data.complex_double[0][1]);
        TEST_ASSERT_EQ(19.0, y.storage.coordinate.data.complex_double[1][0]);
        TEST_ASSERT_EQ(46.0, y.storage.coordinate.data.complex_double[1][1]);
        TEST_ASSERT_EQ(61.0, y.storage.coordinate.data.complex_double[2][0]);
        TEST_ASSERT_EQ(0.0, y.storage.coordinate.data.complex_double[2][1]);
        mtxvector_free(&y);
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
    TEST_RUN(test_mtxvector_to_mtxfile);
    TEST_RUN(test_mtxvector_partition);
    TEST_RUN(test_mtxvector_join);
    TEST_RUN(test_mtxvector_dot);
    TEST_RUN(test_mtxvector_nrm2);
    TEST_RUN(test_mtxvector_asum);
    TEST_RUN(test_mtxvector_iamax);
    TEST_RUN(test_mtxvector_scal);
    TEST_RUN(test_mtxvector_axpy);
    TEST_SUITE_END();
    return (TEST_SUITE_STATUS == TEST_SUCCESS) ?
        EXIT_SUCCESS : EXIT_FAILURE;
}
