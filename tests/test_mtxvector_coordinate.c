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
 * Last modified: 2022-03-23
 *
 * Unit tests for sparse vectors in coordinate format.
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
 * ‘test_mtxvector_coordinate_from_mtxfile()’ tests converting Matrix
 *  Market files to vectors.
 */
int test_mtxvector_coordinate_from_mtxfile(void)
{
    int err;
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
        TEST_ASSERT_EQ(4, x_->num_entries);
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
        TEST_ASSERT_EQ(4, x_->num_entries);
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
        TEST_ASSERT_EQ(4, x_->num_entries);
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
        TEST_ASSERT_EQ(4, x_->num_entries);
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
        TEST_ASSERT_EQ(4, x_->num_entries);
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
        TEST_ASSERT_EQ(4, x_->num_entries);
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
        TEST_ASSERT_EQ(4, x_->num_entries);
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
 * ‘test_mtxvector_coordinate_to_mtxfile()’ tests converting vectors
 * to Matrix Market files.
 */
int test_mtxvector_coordinate_to_mtxfile(void)
{
    int err;
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
 * ‘test_mtxvector_coordinate_partition()’ tests partitioning vectors.
 */
int test_mtxvector_coordinate_partition(void)
{
    int err;

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
            TEST_ASSERT_EQ(6, x->num_entries);
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
            TEST_ASSERT_EQ(6, x->num_entries);
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
 * ‘test_mtxvector_coordinate_join()’ tests joining vectors.
 */
int test_mtxvector_coordinate_join(void)
{
    int err;

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
        TEST_ASSERT_EQ(12, A->num_entries);
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
 * ‘test_mtxvector_coordinate_swap()’ tests swapping values of two
 * vectors.
 */
int test_mtxvector_coordinate_swap(void)
{
    int err;
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
        err = mtxvector_swap(&x, &y);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(2.0f, x.storage.coordinate.data.real_single[0]);
        TEST_ASSERT_EQ(1.0f, x.storage.coordinate.data.real_single[1]);
        TEST_ASSERT_EQ(0.0f, x.storage.coordinate.data.real_single[2]);
        TEST_ASSERT_EQ(2.0f, x.storage.coordinate.data.real_single[3]);
        TEST_ASSERT_EQ(1.0f, x.storage.coordinate.data.real_single[4]);
        TEST_ASSERT_EQ(1.0f, y.storage.coordinate.data.real_single[0]);
        TEST_ASSERT_EQ(1.0f, y.storage.coordinate.data.real_single[1]);
        TEST_ASSERT_EQ(1.0f, y.storage.coordinate.data.real_single[2]);
        TEST_ASSERT_EQ(2.0f, y.storage.coordinate.data.real_single[3]);
        TEST_ASSERT_EQ(3.0f, y.storage.coordinate.data.real_single[4]);
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
        err = mtxvector_swap(&x, &y);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(2.0, x.storage.coordinate.data.real_double[0]);
        TEST_ASSERT_EQ(1.0, x.storage.coordinate.data.real_double[1]);
        TEST_ASSERT_EQ(0.0, x.storage.coordinate.data.real_double[2]);
        TEST_ASSERT_EQ(2.0, x.storage.coordinate.data.real_double[3]);
        TEST_ASSERT_EQ(1.0, x.storage.coordinate.data.real_double[4]);
        TEST_ASSERT_EQ(1.0, y.storage.coordinate.data.real_double[0]);
        TEST_ASSERT_EQ(1.0, y.storage.coordinate.data.real_double[1]);
        TEST_ASSERT_EQ(1.0, y.storage.coordinate.data.real_double[2]);
        TEST_ASSERT_EQ(2.0, y.storage.coordinate.data.real_double[3]);
        TEST_ASSERT_EQ(3.0, y.storage.coordinate.data.real_double[4]);
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
        err = mtxvector_swap(&x, &y);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(2.0f, x.storage.coordinate.data.complex_single[0][0]);
        TEST_ASSERT_EQ(1.0f, x.storage.coordinate.data.complex_single[0][1]);
        TEST_ASSERT_EQ(0.0f, x.storage.coordinate.data.complex_single[1][0]);
        TEST_ASSERT_EQ(2.0f, x.storage.coordinate.data.complex_single[1][1]);
        TEST_ASSERT_EQ(1.0f, x.storage.coordinate.data.complex_single[2][0]);
        TEST_ASSERT_EQ(0.0f, x.storage.coordinate.data.complex_single[2][1]);
        TEST_ASSERT_EQ(1.0f, y.storage.coordinate.data.complex_single[0][0]);
        TEST_ASSERT_EQ(1.0f, y.storage.coordinate.data.complex_single[0][1]);
        TEST_ASSERT_EQ(1.0f, y.storage.coordinate.data.complex_single[1][0]);
        TEST_ASSERT_EQ(2.0f, y.storage.coordinate.data.complex_single[1][1]);
        TEST_ASSERT_EQ(3.0f, y.storage.coordinate.data.complex_single[2][0]);
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
        err = mtxvector_swap(&x, &y);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(2.0, x.storage.coordinate.data.complex_double[0][0]);
        TEST_ASSERT_EQ(1.0, x.storage.coordinate.data.complex_double[0][1]);
        TEST_ASSERT_EQ(0.0, x.storage.coordinate.data.complex_double[1][0]);
        TEST_ASSERT_EQ(2.0, x.storage.coordinate.data.complex_double[1][1]);
        TEST_ASSERT_EQ(1.0, x.storage.coordinate.data.complex_double[2][0]);
        TEST_ASSERT_EQ(0.0, x.storage.coordinate.data.complex_double[2][1]);
        TEST_ASSERT_EQ(1.0, y.storage.coordinate.data.complex_double[0][0]);
        TEST_ASSERT_EQ(1.0, y.storage.coordinate.data.complex_double[0][1]);
        TEST_ASSERT_EQ(1.0, y.storage.coordinate.data.complex_double[1][0]);
        TEST_ASSERT_EQ(2.0, y.storage.coordinate.data.complex_double[1][1]);
        TEST_ASSERT_EQ(3.0, y.storage.coordinate.data.complex_double[2][0]);
        TEST_ASSERT_EQ(0.0, y.storage.coordinate.data.complex_double[2][1]);
        mtxvector_free(&y);
        mtxvector_free(&x);
    }

    /* swap vectors with non-matching sparsity patterns */

    {
        struct mtxvector x;
        struct mtxvector y;
        int size = 12;
        int xidx[] = {0, 3, 5, 6, 9};
        float xdata[] = {1.0f, 1.0f, 1.0f, 2.0f, 3.0f};
        int yidx[] = {0, 3, 6, 7, 9};
        float ydata[] = {2.0f, 1.0f, 0.0f, 2.0f, 1.0f};
        int num_nonzeros = sizeof(xdata) / sizeof(*xdata);
        err = mtxvector_init_coordinate_real_single(
            &x, size, num_nonzeros, xidx, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_init_coordinate_real_single(
            &y, size, num_nonzeros, yidx, ydata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_swap(&x, &y);
        TEST_ASSERT_EQ_MSG(MTX_ERR_INCOMPATIBLE_PATTERN, err, "%s", mtxstrerror(err));
        mtxvector_free(&y);
        mtxvector_free(&x);
    }
    return TEST_SUCCESS;
}

/**
 * ‘test_mtxvector_coordinate_copy()’ tests copying values from one
 * vector to another.
 */
int test_mtxvector_coordinate_copy(void)
{
    int err;
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
        err = mtxvector_copy(&y, &x);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(1.0f, y.storage.coordinate.data.real_single[0]);
        TEST_ASSERT_EQ(1.0f, y.storage.coordinate.data.real_single[1]);
        TEST_ASSERT_EQ(1.0f, y.storage.coordinate.data.real_single[2]);
        TEST_ASSERT_EQ(2.0f, y.storage.coordinate.data.real_single[3]);
        TEST_ASSERT_EQ(3.0f, y.storage.coordinate.data.real_single[4]);
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
        err = mtxvector_copy(&y, &x);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(1.0, y.storage.coordinate.data.real_double[0]);
        TEST_ASSERT_EQ(1.0, y.storage.coordinate.data.real_double[1]);
        TEST_ASSERT_EQ(1.0, y.storage.coordinate.data.real_double[2]);
        TEST_ASSERT_EQ(2.0, y.storage.coordinate.data.real_double[3]);
        TEST_ASSERT_EQ(3.0, y.storage.coordinate.data.real_double[4]);
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
        err = mtxvector_copy(&y, &x);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(1.0f, y.storage.coordinate.data.complex_single[0][0]);
        TEST_ASSERT_EQ(1.0f, y.storage.coordinate.data.complex_single[0][1]);
        TEST_ASSERT_EQ(1.0f, y.storage.coordinate.data.complex_single[1][0]);
        TEST_ASSERT_EQ(2.0f, y.storage.coordinate.data.complex_single[1][1]);
        TEST_ASSERT_EQ(3.0f, y.storage.coordinate.data.complex_single[2][0]);
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
        err = mtxvector_copy(&y, &x);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(1.0, y.storage.coordinate.data.complex_double[0][0]);
        TEST_ASSERT_EQ(1.0, y.storage.coordinate.data.complex_double[0][1]);
        TEST_ASSERT_EQ(1.0, y.storage.coordinate.data.complex_double[1][0]);
        TEST_ASSERT_EQ(2.0, y.storage.coordinate.data.complex_double[1][1]);
        TEST_ASSERT_EQ(3.0, y.storage.coordinate.data.complex_double[2][0]);
        TEST_ASSERT_EQ(0.0, y.storage.coordinate.data.complex_double[2][1]);
        mtxvector_free(&y);
        mtxvector_free(&x);
    }

    /* swap vectors with non-matching sparsity patterns */

    {
        struct mtxvector x;
        struct mtxvector y;
        int size = 12;
        int xidx[] = {0, 3, 5, 6, 9};
        float xdata[] = {1.0f, 1.0f, 1.0f, 2.0f, 3.0f};
        int yidx[] = {0, 3, 6, 7, 9};
        float ydata[] = {2.0f, 1.0f, 0.0f, 2.0f, 1.0f};
        int num_nonzeros = sizeof(xdata) / sizeof(*xdata);
        err = mtxvector_init_coordinate_real_single(
            &x, size, num_nonzeros, xidx, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_init_coordinate_real_single(
            &y, size, num_nonzeros, yidx, ydata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_copy(&y, &x);
        TEST_ASSERT_EQ_MSG(MTX_ERR_INCOMPATIBLE_PATTERN, err, "%s", mtxstrerror(err));
        mtxvector_free(&y);
        mtxvector_free(&x);
    }
    return TEST_SUCCESS;
}

/**
 * ‘test_mtxvector_coordinate_dot()’ tests computing the dot products
 * of pairs of vectors.
 */
int test_mtxvector_coordinate_dot(void)
{
    int err;

    /*
     * dot products of sparse vectors
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
        float sdot = 0.0f;
        err = mtxvector_sdot(&x, &y, &sdot, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, sdot);
        double ddot = 0.0;
        err = mtxvector_ddot(&x, &y, &ddot, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, ddot);
        float cdotu[2] = {0.0f, 0.0f};
        err = mtxvector_cdotu(&x, &y, &cdotu, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, cdotu[0]); TEST_ASSERT_EQ(0.0f, cdotu[1]);
        float cdotc[2] = {0.0f, 0.0f};
        err = mtxvector_cdotc(&x, &y, &cdotc, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, cdotc[0]); TEST_ASSERT_EQ(0.0f, cdotc[1]);
        double zdotu[2] = {0.0, 0.0};
        err = mtxvector_zdotu(&x, &y, &zdotu, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, zdotu[0]); TEST_ASSERT_EQ(0.0, zdotu[1]);
        double zdotc[2] = {0.0, 0.0};
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
        float sdot = 0.0f;
        err = mtxvector_sdot(&x, &y, &sdot, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, sdot);
        double ddot = 0.0;
        err = mtxvector_ddot(&x, &y, &ddot, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, ddot);
        float cdotu[2] = {0.0f, 0.0f};
        err = mtxvector_cdotu(&x, &y, &cdotu, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, cdotu[0]); TEST_ASSERT_EQ(0.0f, cdotu[1]);
        float cdotc[2] = {0.0f, 0.0f};
        err = mtxvector_cdotc(&x, &y, &cdotc, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, cdotc[0]); TEST_ASSERT_EQ(0.0f, cdotc[1]);
        double zdotu[2] = {0.0, 0.0};
        err = mtxvector_zdotu(&x, &y, &zdotu, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, zdotu[0]); TEST_ASSERT_EQ(0.0, zdotu[1]);
        double zdotc[2] = {0.0, 0.0};
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
        float sdot = 0.0f;
        err = mtxvector_sdot(&x, &y, &sdot, NULL);
        TEST_ASSERT_EQ_MSG(MTX_ERR_INVALID_FIELD, err, "%s", mtxstrerror(err));
        double ddot = 0.0;
        err = mtxvector_ddot(&x, &y, &ddot, NULL);
        TEST_ASSERT_EQ_MSG(MTX_ERR_INVALID_FIELD, err, "%s", mtxstrerror(err));
        float cdotu[2] = {0.0f, 0.0f};
        err = mtxvector_cdotu(&x, &y, &cdotu, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(5.0f, cdotu[0]); TEST_ASSERT_EQ(7.0f, cdotu[1]);
        float cdotc[2] = {0.0f, 0.0f};
        err = mtxvector_cdotc(&x, &y, &cdotc, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, cdotc[0]); TEST_ASSERT_EQ(-3.0f, cdotc[1]);
        double zdotu[2] = {0.0, 0.0};
        err = mtxvector_zdotu(&x, &y, &zdotu, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(5.0, zdotu[0]); TEST_ASSERT_EQ(7.0, zdotu[1]);
        double zdotc[2] = {0.0, 0.0};
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
        float sdot = 0.0f;
        err = mtxvector_sdot(&x, &y, &sdot, NULL);
        TEST_ASSERT_EQ_MSG(MTX_ERR_INVALID_FIELD, err, "%s", mtxstrerror(err));
        double ddot = 0.0;
        err = mtxvector_ddot(&x, &y, &ddot, NULL);
        TEST_ASSERT_EQ_MSG(MTX_ERR_INVALID_FIELD, err, "%s", mtxstrerror(err));
        float cdotu[2] = {0.0f, 0.0f};
        err = mtxvector_cdotu(&x, &y, &cdotu, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(5.0f, cdotu[0]); TEST_ASSERT_EQ(7.0f, cdotu[1]);
        float cdotc[2] = {0.0f, 0.0f};
        err = mtxvector_cdotc(&x, &y, &cdotc, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, cdotc[0]); TEST_ASSERT_EQ(-3.0f, cdotc[1]);
        double zdotu[2] = {0.0, 0.0};
        err = mtxvector_zdotu(&x, &y, &zdotu, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(5.0, zdotu[0]); TEST_ASSERT_EQ(7.0, zdotu[1]);
        double zdotc[2] = {0.0, 0.0};
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
        float sdot = 0.0f;
        err = mtxvector_sdot(&x, &y, &sdot, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, sdot);
        double ddot = 0.0;
        err = mtxvector_ddot(&x, &y, &ddot, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, ddot);
        float cdotu[2] = {0.0f, 0.0f};
        err = mtxvector_cdotu(&x, &y, &cdotu, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, cdotu[0]); TEST_ASSERT_EQ(0.0f, cdotu[1]);
        float cdotc[2] = {0.0f, 0.0f};
        err = mtxvector_cdotc(&x, &y, &cdotc, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, cdotc[0]); TEST_ASSERT_EQ(0.0f, cdotc[1]);
        double zdotu[2] = {0.0, 0.0};
        err = mtxvector_zdotu(&x, &y, &zdotu, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, zdotu[0]); TEST_ASSERT_EQ(0.0, zdotu[1]);
        double zdotc[2] = {0.0, 0.0};
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
        float sdot = 0.0f;
        err = mtxvector_sdot(&x, &y, &sdot, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, sdot);
        double ddot = 0.0;
        err = mtxvector_ddot(&x, &y, &ddot, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, ddot);
        float cdotu[2] = {0.0f, 0.0f};
        err = mtxvector_cdotu(&x, &y, &cdotu, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, cdotu[0]); TEST_ASSERT_EQ(0.0f, cdotu[1]);
        float cdotc[2] = {0.0f, 0.0f};
        err = mtxvector_cdotc(&x, &y, &cdotc, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, cdotc[0]); TEST_ASSERT_EQ(0.0f, cdotc[1]);
        double zdotu[2] = {0.0, 0.0};
        err = mtxvector_zdotu(&x, &y, &zdotu, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, zdotu[0]); TEST_ASSERT_EQ(0.0, zdotu[1]);
        double zdotc[2] = {0.0, 0.0};
        err = mtxvector_zdotc(&x, &y, &zdotc, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, zdotc[0]); TEST_ASSERT_EQ(0.0, zdotc[1]);
        mtxvector_free(&y);
        mtxvector_free(&x);
    }

    /*
     * dot products of a sparse and a dense vector
     */

    {
        int size = 12;
        int nnz = 5;
        int idx[] = {1, 3, 5, 7, 9};
        struct mtxvector x;
        float xdata[] = {3.0f, 2.0f, 1.0f, 0.0f, 1.0f};
        err = mtxvector_init_coordinate_real_single(&x, size, nnz, idx, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxvector y;
        float ydata[] = {0, 1.0f, 0, 1.0f, 0, 1.0f, 0, 2.0f, 0, 3.0f, 0, 0};
        err = mtxvector_init_array_real_single(&y, size, ydata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        float sdot = 0.0f;
        err = mtxvector_sdot(&x, &y, &sdot, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, sdot);
        double ddot = 0.0;
        err = mtxvector_ddot(&x, &y, &ddot, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, ddot);
        float cdotu[2] = {0.0f, 0.0f};
        err = mtxvector_cdotu(&x, &y, &cdotu, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, cdotu[0]); TEST_ASSERT_EQ(0.0f, cdotu[1]);
        float cdotc[2] = {0.0f, 0.0f};
        err = mtxvector_cdotc(&x, &y, &cdotc, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, cdotc[0]); TEST_ASSERT_EQ(0.0f, cdotc[1]);
        double zdotu[2] = {0.0, 0.0};
        err = mtxvector_zdotu(&x, &y, &zdotu, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, zdotu[0]); TEST_ASSERT_EQ(0.0, zdotu[1]);
        double zdotc[2] = {0.0, 0.0};
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
        double xdata[] = {3.0, 2.0, 1.0, 0.0, 1.0};
        err = mtxvector_init_coordinate_real_double(&x, size, nnz, idx, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxvector y;
        double ydata[] = {0, 1.0, 0, 1.0, 0, 1.0, 0, 2.0, 0, 3.0, 0, 0};
        err = mtxvector_init_array_real_double(&y, size, ydata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        float sdot = 0.0f;
        err = mtxvector_sdot(&x, &y, &sdot, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, sdot);
        double ddot = 0.0;
        err = mtxvector_ddot(&x, &y, &ddot, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, ddot);
        float cdotu[2] = {0.0f, 0.0f};
        err = mtxvector_cdotu(&x, &y, &cdotu, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, cdotu[0]); TEST_ASSERT_EQ(0.0f, cdotu[1]);
        float cdotc[2] = {0.0f, 0.0f};
        err = mtxvector_cdotc(&x, &y, &cdotc, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, cdotc[0]); TEST_ASSERT_EQ(0.0f, cdotc[1]);
        double zdotu[2] = {0.0, 0.0};
        err = mtxvector_zdotu(&x, &y, &zdotu, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, zdotu[0]); TEST_ASSERT_EQ(0.0, zdotu[1]);
        double zdotc[2] = {0.0, 0.0};
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
        float xdata[][2] = {{3.0f, 2.0f}, {1.0f, 0.0f}, {1.0f, 0.0f}};
        err = mtxvector_init_coordinate_complex_single(&x, size, nnz, idx, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxvector y;
        float ydata[][2] = {{0,0}, {1.0f, 1.0f}, {0,0}, {1.0f, 2.0f}, {0,0}, {3.0f, 0.0f}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0}};
        err = mtxvector_init_array_complex_single(&y, size, ydata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        float sdot = 0.0f;
        err = mtxvector_sdot(&x, &y, &sdot, NULL);
        TEST_ASSERT_EQ_MSG(MTX_ERR_INVALID_FIELD, err, "%s", mtxstrerror(err));
        double ddot = 0.0;
        err = mtxvector_ddot(&x, &y, &ddot, NULL);
        TEST_ASSERT_EQ_MSG(MTX_ERR_INVALID_FIELD, err, "%s", mtxstrerror(err));
        float cdotu[2] = {0.0f, 0.0f};
        err = mtxvector_cdotu(&x, &y, &cdotu, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(5.0f, cdotu[0]); TEST_ASSERT_EQ(7.0f, cdotu[1]);
        float cdotc[2] = {0.0f, 0.0f};
        err = mtxvector_cdotc(&x, &y, &cdotc, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, cdotc[0]); TEST_ASSERT_EQ(-3.0f, cdotc[1]);
        double zdotu[2] = {0.0, 0.0};
        err = mtxvector_zdotu(&x, &y, &zdotu, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(5.0, zdotu[0]); TEST_ASSERT_EQ(7.0, zdotu[1]);
        double zdotc[2] = {0.0, 0.0};
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
        double xdata[][2] = {{3.0, 2.0}, {1.0, 0.0}, {1.0, 0.0}};
        err = mtxvector_init_coordinate_complex_double(&x, size, nnz, idx, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxvector y;
        double ydata[][2] = {{0,0}, {1.0, 1.0}, {0,0}, {1.0, 2.0}, {0,0}, {3.0, 0.0}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0}};
        err = mtxvector_init_array_complex_double(&y, size, ydata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        float sdot = 0.0f;
        err = mtxvector_sdot(&x, &y, &sdot, NULL);
        TEST_ASSERT_EQ_MSG(MTX_ERR_INVALID_FIELD, err, "%s", mtxstrerror(err));
        double ddot = 0.0;
        err = mtxvector_ddot(&x, &y, &ddot, NULL);
        TEST_ASSERT_EQ_MSG(MTX_ERR_INVALID_FIELD, err, "%s", mtxstrerror(err));
        float cdotu[2] = {0.0f, 0.0f};
        err = mtxvector_cdotu(&x, &y, &cdotu, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(5.0f, cdotu[0]); TEST_ASSERT_EQ(7.0f, cdotu[1]);
        float cdotc[2] = {0.0f, 0.0f};
        err = mtxvector_cdotc(&x, &y, &cdotc, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, cdotc[0]); TEST_ASSERT_EQ(-3.0f, cdotc[1]);
        double zdotu[2] = {0.0, 0.0};
        err = mtxvector_zdotu(&x, &y, &zdotu, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(5.0, zdotu[0]); TEST_ASSERT_EQ(7.0, zdotu[1]);
        double zdotc[2] = {0.0, 0.0};
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
        int32_t xdata[] = {3, 2, 1, 0, 1};
        err = mtxvector_init_coordinate_integer_single(&x, size, nnz, idx, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxvector y;
        int32_t ydata[] = {0, 1, 0, 1, 0, 1, 0, 2, 0, 3, 0, 0};
        err = mtxvector_init_array_integer_single(&y, size, ydata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        float sdot = 0.0f;
        err = mtxvector_sdot(&x, &y, &sdot, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, sdot);
        double ddot = 0.0;
        err = mtxvector_ddot(&x, &y, &ddot, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, ddot);
        float cdotu[2] = {0.0f, 0.0f};
        err = mtxvector_cdotu(&x, &y, &cdotu, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, cdotu[0]); TEST_ASSERT_EQ(0.0f, cdotu[1]);
        float cdotc[2] = {0.0f, 0.0f};
        err = mtxvector_cdotc(&x, &y, &cdotc, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, cdotc[0]); TEST_ASSERT_EQ(0.0f, cdotc[1]);
        double zdotu[2] = {0.0, 0.0};
        err = mtxvector_zdotu(&x, &y, &zdotu, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, zdotu[0]); TEST_ASSERT_EQ(0.0, zdotu[1]);
        double zdotc[2] = {0.0, 0.0};
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
        int64_t xdata[] = {3, 2, 1, 0, 1};
        err = mtxvector_init_coordinate_integer_double(&x, size, nnz, idx, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxvector y;
        int64_t ydata[] = {0, 1, 0, 1, 0, 1, 0, 2, 0, 3, 0, 0};
        err = mtxvector_init_array_integer_double(&y, size, ydata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        float sdot = 0.0f;
        err = mtxvector_sdot(&x, &y, &sdot, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, sdot);
        double ddot = 0.0;
        err = mtxvector_ddot(&x, &y, &ddot, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, ddot);
        float cdotu[2] = {0.0f, 0.0f};
        err = mtxvector_cdotu(&x, &y, &cdotu, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, cdotu[0]); TEST_ASSERT_EQ(0.0f, cdotu[1]);
        float cdotc[2] = {0.0f, 0.0f};
        err = mtxvector_cdotc(&x, &y, &cdotc, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, cdotc[0]); TEST_ASSERT_EQ(0.0f, cdotc[1]);
        double zdotu[2] = {0.0, 0.0};
        err = mtxvector_zdotu(&x, &y, &zdotu, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, zdotu[0]); TEST_ASSERT_EQ(0.0, zdotu[1]);
        double zdotc[2] = {0.0, 0.0};
        err = mtxvector_zdotc(&x, &y, &zdotc, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, zdotc[0]); TEST_ASSERT_EQ(0.0, zdotc[1]);
        mtxvector_free(&y);
        mtxvector_free(&x);
    }
    return TEST_SUCCESS;
}

/**
 * ‘test_mtxvector_coordinate_nrm2()’ tests computing the Euclidean
 * norm of vectors.
 */
int test_mtxvector_coordinate_nrm2(void)
{
    int err;
    {
        struct mtxvector x;
        int size = 12;
        int indices[] = {0, 3, 5, 6, 9};
        float data[] = {1.0f, 1.0f, 1.0f, 2.0f, 3.0f};
        int num_nonzeros = sizeof(data) / sizeof(*data);
        err = mtxvector_init_coordinate_real_single(
            &x, size, num_nonzeros, indices, data);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        float snrm2 = 0.0f;
        err = mtxvector_snrm2(&x, &snrm2, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(4.0f, snrm2);
        double dnrm2 = 0.0;
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
        float snrm2 = 0.0f;
        err = mtxvector_snrm2(&x, &snrm2, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(4.0f, snrm2);
        double dnrm2 = 0.0;
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
        float snrm2 = 0.0f;
        err = mtxvector_snrm2(&x, &snrm2, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(4.0f, snrm2);
        double dnrm2 = 0.0;
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
        float snrm2 = 0.0f;
        err = mtxvector_snrm2(&x, &snrm2, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(4.0f, snrm2);
        double dnrm2 = 0.0;
        err = mtxvector_dnrm2(&x, &dnrm2, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(4.0, dnrm2);
        mtxvector_free(&x);
    }
    return TEST_SUCCESS;
}

/**
 * ‘test_mtxvector_coordinate_asum()’ tests computing the sum of
 * absolute values of vectors.
 */
int test_mtxvector_coordinate_asum(void)
{
    int err;
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
 * ‘test_mtxvector_coordinate_iamax()’ tests computing the sum of
 * absolute values of vectors.
 */
int test_mtxvector_coordinate_iamax(void)
{
    int err;
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
 * ‘test_mtxvector_coordinate_scal()’ tests scaling vectors by a
 * constant.
 */
int test_mtxvector_coordinate_scal(void)
{
    int err;
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
 * ‘test_mtxvector_coordinate_axpy()’ tests multiplying a vector by a
 * constant and adding the result to another vector.
 */
int test_mtxvector_coordinate_axpy(void)
{
    int err;

    /*
     * Multiply a sparse vector by a constant and add the result to
     * another sparse vector.
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
 * ‘test_mtxvector_coordinate_usga()’ tests gathering values from a
 * vector into a sparse vector.
 */
int test_mtxvector_coordinate_usga(void)
{
    int err;

    /*
     * gather nonzero values from a vector into a sparse vector
     */

    {
        struct mtxvector x;
        struct mtxvector y;
        int size = 12;
        int indices[] = {0, 3, 5, 6, 9};
        float xdata[] = {1.0f, 1.0f, 1.0f, 2.0f, 3.0f};
        float ydata[] = {2.0f, 0, 0, 1.0f, 0, 0.0f, 2.0f, 0, 0, 1.0f, 0, 0};
        int num_nonzeros = sizeof(xdata) / sizeof(*xdata);
        err = mtxvector_init_coordinate_real_single(
            &x, size, num_nonzeros, indices, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_init_array_real_single(&y, size, ydata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_usga(&y, &x);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(2.0f, x.storage.coordinate.data.real_single[ 0]);
        TEST_ASSERT_EQ(1.0f, x.storage.coordinate.data.real_single[ 1]);
        TEST_ASSERT_EQ(0.0f, x.storage.coordinate.data.real_single[ 2]);
        TEST_ASSERT_EQ(2.0f, x.storage.coordinate.data.real_single[ 3]);
        TEST_ASSERT_EQ(1.0f, x.storage.coordinate.data.real_single[ 4]);
        mtxvector_free(&y);
        mtxvector_free(&x);
    }
    {
        struct mtxvector x;
        struct mtxvector y;
        int size = 12;
        int indices[] = {0, 3, 5, 6, 9};
        double xdata[] = {1.0, 1.0, 1.0, 2.0, 3.0};
        double ydata[] = {2.0, 0, 0, 1.0, 0, 0.0, 2.0, 0, 0, 1.0, 0, 0};
        int num_nonzeros = sizeof(xdata) / sizeof(*xdata);
        err = mtxvector_init_coordinate_real_double(
            &x, size, num_nonzeros, indices, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_init_array_real_double(&y, size, ydata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_usga(&y, &x);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(2.0, x.storage.coordinate.data.real_double[ 0]);
        TEST_ASSERT_EQ(1.0, x.storage.coordinate.data.real_double[ 1]);
        TEST_ASSERT_EQ(0.0, x.storage.coordinate.data.real_double[ 2]);
        TEST_ASSERT_EQ(2.0, x.storage.coordinate.data.real_double[ 3]);
        TEST_ASSERT_EQ(1.0, x.storage.coordinate.data.real_double[ 4]);
        mtxvector_free(&y);
        mtxvector_free(&x);
    }
    {
        struct mtxvector x;
        struct mtxvector y;
        int size = 6;
        int indices[] = {0, 3, 5};
        float xdata[][2] = {{1.0f,1.0f}, {1.0f,2.0f}, {3.0f,0.0f}};
        float ydata[][2] = {{2.0f,1.0f}, {0,0}, {0,0}, {0.0f,2.0f}, {0,0}, {1.0f,0.0f}};
        int64_t num_nonzeros = sizeof(xdata) / sizeof(*xdata);
        err = mtxvector_init_coordinate_complex_single(
            &x, size, num_nonzeros, indices, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_init_array_complex_single(&y, size, ydata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_usga(&y, &x);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(2.0f, x.storage.coordinate.data.complex_single[0][0]);
        TEST_ASSERT_EQ(1.0f, x.storage.coordinate.data.complex_single[0][1]);
        TEST_ASSERT_EQ(0.0f, x.storage.coordinate.data.complex_single[1][0]);
        TEST_ASSERT_EQ(2.0f, x.storage.coordinate.data.complex_single[1][1]);
        TEST_ASSERT_EQ(1.0f, x.storage.coordinate.data.complex_single[2][0]);
        TEST_ASSERT_EQ(0.0f, x.storage.coordinate.data.complex_single[2][1]);
        mtxvector_free(&y);
        mtxvector_free(&x);
    }
    {
        struct mtxvector x;
        struct mtxvector y;
        int size = 6;
        int indices[] = {0, 3, 5};
        double xdata[][2] = {{1.0,1.0}, {1.0,2.0}, {3.0,0.0}};
        double ydata[][2] = {{2.0,1.0}, {0,0}, {0,0}, {0.0,2.0}, {0,0}, {1.0,0.0}};
        int64_t num_nonzeros = sizeof(xdata) / sizeof(*xdata);
        err = mtxvector_init_coordinate_complex_double(
            &x, size, num_nonzeros, indices, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_init_array_complex_double(&y, size, ydata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_usga(&y, &x);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(2.0, x.storage.coordinate.data.complex_double[0][0]);
        TEST_ASSERT_EQ(1.0, x.storage.coordinate.data.complex_double[0][1]);
        TEST_ASSERT_EQ(0.0, x.storage.coordinate.data.complex_double[1][0]);
        TEST_ASSERT_EQ(2.0, x.storage.coordinate.data.complex_double[1][1]);
        TEST_ASSERT_EQ(1.0, x.storage.coordinate.data.complex_double[2][0]);
        TEST_ASSERT_EQ(0.0, x.storage.coordinate.data.complex_double[2][1]);
        mtxvector_free(&y);
        mtxvector_free(&x);
    }
    return TEST_SUCCESS;
}

/**
 * ‘main()’ entry point and test driver.
 */
int main(int argc, char * argv[])
{
    TEST_SUITE_BEGIN("Running tests for vectors\n");
    TEST_RUN(test_mtxvector_coordinate_from_mtxfile);
    TEST_RUN(test_mtxvector_coordinate_to_mtxfile);
    TEST_RUN(test_mtxvector_coordinate_partition);
    TEST_RUN(test_mtxvector_coordinate_join);
    TEST_RUN(test_mtxvector_coordinate_swap);
    TEST_RUN(test_mtxvector_coordinate_copy);
    TEST_RUN(test_mtxvector_coordinate_dot);
    TEST_RUN(test_mtxvector_coordinate_nrm2);
    TEST_RUN(test_mtxvector_coordinate_asum);
    TEST_RUN(test_mtxvector_coordinate_iamax);
    TEST_RUN(test_mtxvector_coordinate_scal);
    TEST_RUN(test_mtxvector_coordinate_axpy);
    TEST_RUN(test_mtxvector_coordinate_usga);
    TEST_SUITE_END();
    return (TEST_SUITE_STATUS == TEST_SUCCESS) ?
        EXIT_SUCCESS : EXIT_FAILURE;
}
