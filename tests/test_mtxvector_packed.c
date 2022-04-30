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
 * Last modified: 2022-04-28
 *
 * Unit tests for sparse vectors in packed storage format.
 */

#include "test.h"

#include <libmtx/error.h>
#include <libmtx/mtxfile/mtxfile.h>
#include <libmtx/vector/base.h>
#include <libmtx/vector/packed.h>
#include <libmtx/vector/vector.h>

#include <errno.h>
#include <unistd.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * ‘test_mtxvector_packed_from_mtxfile()’ tests converting Matrix
 *  Market files to vectors.
 */
int test_mtxvector_packed_from_mtxfile(void)
{
    int err;
    {
        int num_rows = 3;
        const float mtxdata[] = {3.0f, 4.0f, 5.0f};
        int64_t num_nonzeros = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile mtxfile;
        err = mtxfile_init_vector_array_real_single(&mtxfile, num_rows, mtxdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxvector_packed xpacked;
        err = mtxvector_packed_from_mtxfile(&xpacked, &mtxfile, mtxvector_base);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(3, xpacked.size);
        TEST_ASSERT_EQ(3, xpacked.num_nonzeros);
        TEST_ASSERT_EQ(NULL, xpacked.idx);
        TEST_ASSERT_EQ(mtxvector_base, xpacked.x.type);
        const struct mtxvector_base * x = &xpacked.x.storage.base;
        TEST_ASSERT_EQ(mtx_field_real, x->field);
        TEST_ASSERT_EQ(mtx_single, x->precision);
        TEST_ASSERT_EQ(3, x->size);
        TEST_ASSERT_EQ(x->data.real_single[0], 3.0f);
        TEST_ASSERT_EQ(x->data.real_single[1], 4.0f);
        TEST_ASSERT_EQ(x->data.real_single[2], 5.0f);
        mtxvector_packed_free(&xpacked);
        mtxfile_free(&mtxfile);
    }
    {
        int size = 4;
        struct mtxfile_vector_coordinate_real_single mtxdata[] = {
            {1, 1.0f}, {2, 2.0f}, {4, 4.0f}};
        int64_t num_nonzeros = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile mtxfile;
        err = mtxfile_init_vector_coordinate_real_single(
            &mtxfile, size, num_nonzeros, mtxdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxvector_packed xpacked;
        err = mtxvector_packed_from_mtxfile(&xpacked, &mtxfile, mtxvector_base);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(4, xpacked.size);
        TEST_ASSERT_EQ(3, xpacked.num_nonzeros);
        TEST_ASSERT_EQ(xpacked.idx[0], 0);
        TEST_ASSERT_EQ(xpacked.idx[1], 1);
        TEST_ASSERT_EQ(xpacked.idx[2], 3);
        TEST_ASSERT_EQ(mtxvector_base, xpacked.x.type);
        const struct mtxvector_base * x = &xpacked.x.storage.base;
        TEST_ASSERT_EQ(mtx_field_real, x->field);
        TEST_ASSERT_EQ(mtx_single, x->precision);
        TEST_ASSERT_EQ(3, x->size);
        TEST_ASSERT_EQ(x->data.real_single[0], 1.0f);
        TEST_ASSERT_EQ(x->data.real_single[1], 2.0f);
        TEST_ASSERT_EQ(x->data.real_single[2], 4.0f);
        mtxvector_packed_free(&xpacked);
        mtxfile_free(&mtxfile);
    }
    {
        int size = 4;
        struct mtxfile_vector_coordinate_real_double mtxdata[] = {
            {1, 1.0}, {2, 2.0}, {4, 4.0}};
        int64_t num_nonzeros = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile mtxfile;
        err = mtxfile_init_vector_coordinate_real_double(
            &mtxfile, size, num_nonzeros, mtxdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxvector_packed xpacked;
        err = mtxvector_packed_from_mtxfile(&xpacked, &mtxfile, mtxvector_base);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(4, xpacked.size);
        TEST_ASSERT_EQ(3, xpacked.num_nonzeros);
        TEST_ASSERT_EQ(xpacked.idx[0], 0);
        TEST_ASSERT_EQ(xpacked.idx[1], 1);
        TEST_ASSERT_EQ(xpacked.idx[2], 3);
        TEST_ASSERT_EQ(mtxvector_base, xpacked.x.type);
        const struct mtxvector_base * x = &xpacked.x.storage.base;
        TEST_ASSERT_EQ(mtx_field_real, x->field);
        TEST_ASSERT_EQ(mtx_double, x->precision);
        TEST_ASSERT_EQ(3, x->size);
        TEST_ASSERT_EQ(x->data.real_double[0], 1.0);
        TEST_ASSERT_EQ(x->data.real_double[1], 2.0);
        TEST_ASSERT_EQ(x->data.real_double[2], 4.0);
        mtxvector_packed_free(&xpacked);
        mtxfile_free(&mtxfile);
    }
    {
        int size = 4;
        struct mtxfile_vector_coordinate_complex_single mtxdata[] = {
            {1,1.0f,-1.0f}, {2,2.0f,-2.0f}, {4,4.0f,-4.0f}};
        int64_t num_nonzeros = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile mtxfile;
        err = mtxfile_init_vector_coordinate_complex_single(
            &mtxfile, size, num_nonzeros, mtxdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxvector_packed xpacked;
        err = mtxvector_packed_from_mtxfile(&xpacked, &mtxfile, mtxvector_base);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(4, xpacked.size);
        TEST_ASSERT_EQ(3, xpacked.num_nonzeros);
        TEST_ASSERT_EQ(xpacked.idx[0], 0);
        TEST_ASSERT_EQ(xpacked.idx[1], 1);
        TEST_ASSERT_EQ(xpacked.idx[2], 3);
        TEST_ASSERT_EQ(mtxvector_base, xpacked.x.type);
        const struct mtxvector_base * x = &xpacked.x.storage.base;
        TEST_ASSERT_EQ(mtx_field_complex, x->field);
        TEST_ASSERT_EQ(mtx_single, x->precision);
        TEST_ASSERT_EQ(3, x->size);
        TEST_ASSERT_EQ(x->data.complex_single[0][0], 1.0f);
        TEST_ASSERT_EQ(x->data.complex_single[0][1], -1.0f);
        TEST_ASSERT_EQ(x->data.complex_single[1][0], 2.0f);
        TEST_ASSERT_EQ(x->data.complex_single[1][1], -2.0f);
        TEST_ASSERT_EQ(x->data.complex_single[2][0], 4.0f);
        TEST_ASSERT_EQ(x->data.complex_single[2][1], -4.0f);
        mtxvector_packed_free(&xpacked);
        mtxfile_free(&mtxfile);
    }
    {
        int size = 4;
        struct mtxfile_vector_coordinate_complex_double mtxdata[] = {
            {1,1.0,-1.0}, {2,2.0,-2.0}, {4,4.0,-4.0}};
        int64_t num_nonzeros = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile mtxfile;
        err = mtxfile_init_vector_coordinate_complex_double(
            &mtxfile, size, num_nonzeros, mtxdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxvector_packed xpacked;
        err = mtxvector_packed_from_mtxfile(&xpacked, &mtxfile, mtxvector_base);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(4, xpacked.size);
        TEST_ASSERT_EQ(3, xpacked.num_nonzeros);
        TEST_ASSERT_EQ(xpacked.idx[0], 0);
        TEST_ASSERT_EQ(xpacked.idx[1], 1);
        TEST_ASSERT_EQ(xpacked.idx[2], 3);
        TEST_ASSERT_EQ(mtxvector_base, xpacked.x.type);
        const struct mtxvector_base * x = &xpacked.x.storage.base;
        TEST_ASSERT_EQ(mtx_field_complex, x->field);
        TEST_ASSERT_EQ(mtx_double, x->precision);
        TEST_ASSERT_EQ(3, x->size);
        TEST_ASSERT_EQ(x->data.complex_double[0][0], 1.0);
        TEST_ASSERT_EQ(x->data.complex_double[0][1], -1.0);
        TEST_ASSERT_EQ(x->data.complex_double[1][0], 2.0);
        TEST_ASSERT_EQ(x->data.complex_double[1][1], -2.0);
        TEST_ASSERT_EQ(x->data.complex_double[2][0], 4.0);
        TEST_ASSERT_EQ(x->data.complex_double[2][1], -4.0);
        mtxvector_packed_free(&xpacked);
        mtxfile_free(&mtxfile);
    }
    {
        int size = 4;
        struct mtxfile_vector_coordinate_integer_single mtxdata[] = {
            {1, 1}, {2, 2}, {4, 4}};
        int64_t num_nonzeros = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile mtxfile;
        err = mtxfile_init_vector_coordinate_integer_single(
            &mtxfile, size, num_nonzeros, mtxdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxvector_packed xpacked;
        err = mtxvector_packed_from_mtxfile(&xpacked, &mtxfile, mtxvector_base);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(4, xpacked.size);
        TEST_ASSERT_EQ(3, xpacked.num_nonzeros);
        TEST_ASSERT_EQ(xpacked.idx[0], 0);
        TEST_ASSERT_EQ(xpacked.idx[1], 1);
        TEST_ASSERT_EQ(xpacked.idx[2], 3);
        TEST_ASSERT_EQ(mtxvector_base, xpacked.x.type);
        const struct mtxvector_base * x = &xpacked.x.storage.base;
        TEST_ASSERT_EQ(mtx_field_integer, x->field);
        TEST_ASSERT_EQ(mtx_single, x->precision);
        TEST_ASSERT_EQ(3, x->size);
        TEST_ASSERT_EQ(x->data.integer_single[0], 1);
        TEST_ASSERT_EQ(x->data.integer_single[1], 2);
        TEST_ASSERT_EQ(x->data.integer_single[2], 4);
        mtxvector_packed_free(&xpacked);
        mtxfile_free(&mtxfile);
    }
    {
        int size = 4;
        struct mtxfile_vector_coordinate_integer_double mtxdata[] = {
            {1,1}, {2,2}, {4,4}};
        int64_t num_nonzeros = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile mtxfile;
        err = mtxfile_init_vector_coordinate_integer_double(
            &mtxfile, size, num_nonzeros, mtxdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxvector_packed xpacked;
        err = mtxvector_packed_from_mtxfile(&xpacked, &mtxfile, mtxvector_base);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(4, xpacked.size);
        TEST_ASSERT_EQ(3, xpacked.num_nonzeros);
        TEST_ASSERT_EQ(xpacked.idx[0], 0);
        TEST_ASSERT_EQ(xpacked.idx[1], 1);
        TEST_ASSERT_EQ(xpacked.idx[2], 3);
        TEST_ASSERT_EQ(mtxvector_base, xpacked.x.type);
        const struct mtxvector_base * x = &xpacked.x.storage.base;
        TEST_ASSERT_EQ(mtx_field_integer, x->field);
        TEST_ASSERT_EQ(mtx_double, x->precision);
        TEST_ASSERT_EQ(3, x->size);
        TEST_ASSERT_EQ(x->data.integer_double[0], 1);
        TEST_ASSERT_EQ(x->data.integer_double[1], 2);
        TEST_ASSERT_EQ(x->data.integer_double[2], 4);
        mtxvector_packed_free(&xpacked);
        mtxfile_free(&mtxfile);
    }
    {
        int size = 4;
        struct mtxfile_vector_coordinate_pattern mtxdata[] = {{1}, {2}, {4}};
        int64_t num_nonzeros = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile mtxfile;
        err = mtxfile_init_vector_coordinate_pattern(
            &mtxfile, size, num_nonzeros, mtxdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxvector_packed xpacked;
        err = mtxvector_packed_from_mtxfile(&xpacked, &mtxfile, mtxvector_base);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(4, xpacked.size);
        TEST_ASSERT_EQ(3, xpacked.num_nonzeros);
        TEST_ASSERT_EQ(xpacked.idx[0], 0);
        TEST_ASSERT_EQ(xpacked.idx[1], 1);
        TEST_ASSERT_EQ(xpacked.idx[2], 3);
        TEST_ASSERT_EQ(mtxvector_base, xpacked.x.type);
        const struct mtxvector_base * x = &xpacked.x.storage.base;
        TEST_ASSERT_EQ(mtx_field_pattern, x->field);
        TEST_ASSERT_EQ(3, x->size);
        mtxvector_packed_free(&xpacked);
        mtxfile_free(&mtxfile);
    }
    return TEST_SUCCESS;
}

/**
 * ‘test_mtxvector_packed_to_mtxfile()’ tests converting vectors
 * to Matrix Market files.
 */
int test_mtxvector_packed_to_mtxfile(void)
{
    int err;
    {
        int size = 5;
        int nnz = 5;
        struct mtxvector_packed x;
        float xdata[] = {1.0f, 1.0f, 1.0f, 2.0f, 3.0f};
        err = mtxvector_packed_init_real_single(
            &x, mtxvector_base, size, nnz, NULL, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxfile mtxfile;
        err = mtxvector_packed_to_mtxfile(&mtxfile, &x, mtxfile_array);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_vector, mtxfile.header.object);
        TEST_ASSERT_EQ(mtxfile_array, mtxfile.header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtxfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxfile.header.symmetry);
        TEST_ASSERT_EQ(size, mtxfile.size.num_rows);
        TEST_ASSERT_EQ(-1, mtxfile.size.num_columns);
        TEST_ASSERT_EQ(-1, mtxfile.size.num_nonzeros);
        TEST_ASSERT_EQ(mtx_single, mtxfile.precision);
        const float * data = mtxfile.data.array_real_single;
        for (int64_t k = 0; k < size; k++)
            TEST_ASSERT_EQ(xdata[k], data[k]);
        mtxfile_free(&mtxfile);
        mtxvector_packed_free(&x);
    }
    {
        int size = 5;
        int nnz = 5;
        struct mtxvector_packed x;
        float xdata[] = {1.0f, 1.0f, 1.0f, 2.0f, 3.0f};
        err = mtxvector_packed_init_real_single(
            &x, mtxvector_base, size, nnz, NULL, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxfile mtxfile;
        err = mtxvector_packed_to_mtxfile(&mtxfile, &x, mtxfile_coordinate);
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
            TEST_ASSERT_EQ(k+1, data[k].i);
            TEST_ASSERT_EQ(xdata[k], data[k].a);
        }
        mtxfile_free(&mtxfile);
        mtxvector_packed_free(&x);
    }
    {
        int size = 12;
        int nnz = 5;
        int64_t idx[] = {1, 3, 5, 7, 9};
        struct mtxvector_packed x;
        float xdata[] = {1.0f, 1.0f, 1.0f, 2.0f, 3.0f};
        err = mtxvector_packed_init_real_single(
            &x, mtxvector_base, size, nnz, idx, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxfile mtxfile;
        err = mtxvector_packed_to_mtxfile(&mtxfile, &x, mtxfile_coordinate);
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
            TEST_ASSERT_EQ_MSG(idx[k]+1, data[k].i,);
            TEST_ASSERT_EQ(xdata[k], data[k].a);
        }
        mtxfile_free(&mtxfile);
        mtxvector_packed_free(&x);
    }
    {
        int size = 12;
        int nnz = 5;
        int64_t idx[] = {1, 3, 5, 7, 9};
        struct mtxvector_packed x;
        double xdata[] = {1.0, 1.0, 1.0, 2.0, 3.0};
        int xsize = sizeof(xdata) / sizeof(*xdata);
        err = mtxvector_packed_init_real_double(
            &x, mtxvector_base, size, nnz, idx, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxfile mtxfile;
        err = mtxvector_packed_to_mtxfile(&mtxfile, &x, mtxfile_coordinate);
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
        mtxvector_packed_free(&x);
    }
    {
        int size = 12;
        int nnz = 3;
        int64_t idx[] = {1, 3, 5, 7, 9};
        struct mtxvector_packed x;
        float xdata[][2] = {{1.0f, 1.0f}, {1.0f, 2.0f}, {3.0f, 0.0f}};
        int xsize = sizeof(xdata) / sizeof(*xdata);
        err = mtxvector_packed_init_complex_single(
            &x, mtxvector_base, size, nnz, idx, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxfile mtxfile;
        err = mtxvector_packed_to_mtxfile(&mtxfile, &x, mtxfile_coordinate);
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
        mtxvector_packed_free(&x);
    }
    {
        int size = 12;
        int nnz = 3;
        int64_t idx[] = {1, 3, 5, 7, 9};
        struct mtxvector_packed x;
        double xdata[][2] = {{1.0, 1.0}, {1.0, 2.0}, {3.0, 0.0}};
        int xsize = sizeof(xdata) / sizeof(*xdata);
        err = mtxvector_packed_init_complex_double(
            &x, mtxvector_base, size, nnz, idx, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxfile mtxfile;
        err = mtxvector_packed_to_mtxfile(&mtxfile, &x, mtxfile_coordinate);
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
        mtxvector_packed_free(&x);
    }
    {
        int size = 12;
        int nnz = 5;
        int64_t idx[] = {1, 3, 5, 7, 9};
        struct mtxvector_packed x;
        int32_t xdata[] = {1, 1, 1, 2, 3};
        int xsize = sizeof(xdata) / sizeof(*xdata);
        err = mtxvector_packed_init_integer_single(
            &x, mtxvector_base, size, nnz, idx, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxfile mtxfile;
        err = mtxvector_packed_to_mtxfile(&mtxfile, &x, mtxfile_coordinate);
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
        mtxvector_packed_free(&x);
    }
    {
        int size = 12;
        int nnz = 5;
        int64_t idx[] = {1, 3, 5, 7, 9};
        struct mtxvector_packed x;
        int64_t xdata[] = {1, 1, 1, 2, 3};
        int xsize = sizeof(xdata) / sizeof(*xdata);
        err = mtxvector_packed_init_integer_double(
            &x, mtxvector_base, size, nnz, idx, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxfile mtxfile;
        err = mtxvector_packed_to_mtxfile(&mtxfile, &x, mtxfile_coordinate);
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
        mtxvector_packed_free(&x);
    }
    return TEST_SUCCESS;
}

/**
 * ‘test_mtxvector_packed_swap()’ tests swapping values of two
 * vectors.
 */
int test_mtxvector_packed_swap(void)
{
    int err;
    {
        struct mtxvector_packed x;
        struct mtxvector_packed y;
        int size = 12;
        int64_t xidx[] = {0, 3, 5, 6, 9};
        float xdata[] = {1.0f, 1.0f, 1.0f, 2.0f, 3.0f};
        int64_t yidx[] = {1, 2, 4, 6, 9};
        float ydata[] = {2.0f, 1.0f, 0.0f, 2.0f, 1.0f};
        int num_nonzeros = sizeof(xdata) / sizeof(*xdata);
        err = mtxvector_packed_init_real_single(
            &x, mtxvector_base, size, num_nonzeros, xidx, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_packed_init_real_single(
            &y, mtxvector_base, size, num_nonzeros, yidx, ydata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_packed_swap(&x, &y);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(12, x.size);
        TEST_ASSERT_EQ(5, x.num_nonzeros);
        TEST_ASSERT_EQ(x.idx[0], 1);
        TEST_ASSERT_EQ(x.idx[1], 2);
        TEST_ASSERT_EQ(x.idx[2], 4);
        TEST_ASSERT_EQ(x.idx[3], 6);
        TEST_ASSERT_EQ(x.idx[4], 9);
        TEST_ASSERT_EQ(12, y.size);
        TEST_ASSERT_EQ(5, y.num_nonzeros);
        TEST_ASSERT_EQ(y.idx[0], 0);
        TEST_ASSERT_EQ(y.idx[1], 3);
        TEST_ASSERT_EQ(y.idx[2], 5);
        TEST_ASSERT_EQ(y.idx[3], 6);
        TEST_ASSERT_EQ(y.idx[4], 9);
        TEST_ASSERT_EQ(mtxvector_base, x.x.type);
        TEST_ASSERT_EQ(mtx_field_real, x.x.storage.base.field);
        TEST_ASSERT_EQ(mtx_single, x.x.storage.base.precision);
        TEST_ASSERT_EQ(5, x.x.storage.base.size);
        TEST_ASSERT_EQ(2.0f, x.x.storage.base.data.real_single[0]);
        TEST_ASSERT_EQ(1.0f, x.x.storage.base.data.real_single[1]);
        TEST_ASSERT_EQ(0.0f, x.x.storage.base.data.real_single[2]);
        TEST_ASSERT_EQ(2.0f, x.x.storage.base.data.real_single[3]);
        TEST_ASSERT_EQ(1.0f, x.x.storage.base.data.real_single[4]);
        TEST_ASSERT_EQ(mtxvector_base, y.x.type);
        TEST_ASSERT_EQ(mtx_field_real, y.x.storage.base.field);
        TEST_ASSERT_EQ(mtx_single, y.x.storage.base.precision);
        TEST_ASSERT_EQ(5, y.x.storage.base.size);
        TEST_ASSERT_EQ(1.0f, y.x.storage.base.data.real_single[0]);
        TEST_ASSERT_EQ(1.0f, y.x.storage.base.data.real_single[1]);
        TEST_ASSERT_EQ(1.0f, y.x.storage.base.data.real_single[2]);
        TEST_ASSERT_EQ(2.0f, y.x.storage.base.data.real_single[3]);
        TEST_ASSERT_EQ(3.0f, y.x.storage.base.data.real_single[4]);
        mtxvector_packed_free(&y);
        mtxvector_packed_free(&x);
    }
    {
        struct mtxvector_packed x;
        struct mtxvector_packed y;
        int size = 12;
        int64_t xidx[] = {0, 3, 5, 6, 9};
        double xdata[] = {1.0, 1.0, 1.0, 2.0, 3.0};
        int64_t yidx[] = {1, 2, 4, 6, 9};
        double ydata[] = {2.0, 1.0, 0.0, 2.0, 1.0};
        int num_nonzeros = sizeof(xdata) / sizeof(*xdata);
        err = mtxvector_packed_init_real_double(
            &x, mtxvector_base, size, num_nonzeros, xidx, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_packed_init_real_double(
            &y, mtxvector_base, size, num_nonzeros, yidx, ydata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_packed_swap(&x, &y);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(12, x.size);
        TEST_ASSERT_EQ(5, x.num_nonzeros);
        TEST_ASSERT_EQ(x.idx[0], 1);
        TEST_ASSERT_EQ(x.idx[1], 2);
        TEST_ASSERT_EQ(x.idx[2], 4);
        TEST_ASSERT_EQ(x.idx[3], 6);
        TEST_ASSERT_EQ(x.idx[4], 9);
        TEST_ASSERT_EQ(12, y.size);
        TEST_ASSERT_EQ(5, y.num_nonzeros);
        TEST_ASSERT_EQ(y.idx[0], 0);
        TEST_ASSERT_EQ(y.idx[1], 3);
        TEST_ASSERT_EQ(y.idx[2], 5);
        TEST_ASSERT_EQ(y.idx[3], 6);
        TEST_ASSERT_EQ(y.idx[4], 9);
        TEST_ASSERT_EQ(mtxvector_base, x.x.type);
        TEST_ASSERT_EQ(mtx_field_real, x.x.storage.base.field);
        TEST_ASSERT_EQ(mtx_double, x.x.storage.base.precision);
        TEST_ASSERT_EQ(5, x.x.storage.base.size);
        TEST_ASSERT_EQ(2.0, x.x.storage.base.data.real_double[0]);
        TEST_ASSERT_EQ(1.0, x.x.storage.base.data.real_double[1]);
        TEST_ASSERT_EQ(0.0, x.x.storage.base.data.real_double[2]);
        TEST_ASSERT_EQ(2.0, x.x.storage.base.data.real_double[3]);
        TEST_ASSERT_EQ(1.0, x.x.storage.base.data.real_double[4]);
        TEST_ASSERT_EQ(mtxvector_base, y.x.type);
        TEST_ASSERT_EQ(mtx_field_real, y.x.storage.base.field);
        TEST_ASSERT_EQ(mtx_double, y.x.storage.base.precision);
        TEST_ASSERT_EQ(5, y.x.storage.base.size);
        TEST_ASSERT_EQ(1.0, y.x.storage.base.data.real_double[0]);
        TEST_ASSERT_EQ(1.0, y.x.storage.base.data.real_double[1]);
        TEST_ASSERT_EQ(1.0, y.x.storage.base.data.real_double[2]);
        TEST_ASSERT_EQ(2.0, y.x.storage.base.data.real_double[3]);
        TEST_ASSERT_EQ(3.0, y.x.storage.base.data.real_double[4]);
        mtxvector_packed_free(&y);
        mtxvector_packed_free(&x);
    }
    {
        struct mtxvector_packed x;
        struct mtxvector_packed y;
        int size = 12;
        int64_t xidx[] = {0, 3, 5};
        float xdata[][2] = {{1.0f,1.0f}, {1.0f,2.0f}, {3.0f,0.0f}};
        int64_t yidx[] = {1, 2, 4};
        float ydata[][2] = {{2.0f,1.0f}, {0.0f,2.0f}, {1.0f,0.0f}};
        int64_t num_nonzeros = sizeof(xdata) / sizeof(*xdata);
        err = mtxvector_packed_init_complex_single(
            &x, mtxvector_base, size, num_nonzeros, xidx, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_packed_init_complex_single(
            &y, mtxvector_base, size, num_nonzeros, yidx, ydata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_packed_swap(&x, &y);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(12, x.size);
        TEST_ASSERT_EQ(3, x.num_nonzeros);
        TEST_ASSERT_EQ(x.idx[0], 1);
        TEST_ASSERT_EQ(x.idx[1], 2);
        TEST_ASSERT_EQ(x.idx[2], 4);
        TEST_ASSERT_EQ(12, y.size);
        TEST_ASSERT_EQ(3, y.num_nonzeros);
        TEST_ASSERT_EQ(y.idx[0], 0);
        TEST_ASSERT_EQ(y.idx[1], 3);
        TEST_ASSERT_EQ(y.idx[2], 5);
        TEST_ASSERT_EQ(mtxvector_base, x.x.type);
        TEST_ASSERT_EQ(mtx_field_complex, x.x.storage.base.field);
        TEST_ASSERT_EQ(mtx_single, x.x.storage.base.precision);
        TEST_ASSERT_EQ(3, x.x.storage.base.size);
        TEST_ASSERT_EQ(2.0f, x.x.storage.base.data.complex_single[0][0]);
        TEST_ASSERT_EQ(1.0f, x.x.storage.base.data.complex_single[0][1]);
        TEST_ASSERT_EQ(0.0f, x.x.storage.base.data.complex_single[1][0]);
        TEST_ASSERT_EQ(2.0f, x.x.storage.base.data.complex_single[1][1]);
        TEST_ASSERT_EQ(1.0f, x.x.storage.base.data.complex_single[2][0]);
        TEST_ASSERT_EQ(0.0f, x.x.storage.base.data.complex_single[2][1]);
        TEST_ASSERT_EQ(mtxvector_base, y.x.type);
        TEST_ASSERT_EQ(mtx_field_complex, y.x.storage.base.field);
        TEST_ASSERT_EQ(mtx_single, y.x.storage.base.precision);
        TEST_ASSERT_EQ(3, y.x.storage.base.size);
        TEST_ASSERT_EQ(1.0f, y.x.storage.base.data.complex_single[0][0]);
        TEST_ASSERT_EQ(1.0f, y.x.storage.base.data.complex_single[0][1]);
        TEST_ASSERT_EQ(1.0f, y.x.storage.base.data.complex_single[1][0]);
        TEST_ASSERT_EQ(2.0f, y.x.storage.base.data.complex_single[1][1]);
        TEST_ASSERT_EQ(3.0f, y.x.storage.base.data.complex_single[2][0]);
        TEST_ASSERT_EQ(0.0f, y.x.storage.base.data.complex_single[2][1]);
        mtxvector_packed_free(&y);
        mtxvector_packed_free(&x);
    }
    {
        struct mtxvector_packed x;
        struct mtxvector_packed y;
        int size = 12;
        int64_t xidx[] = {0, 3, 5};
        double xdata[][2] = {{1.0,1.0}, {1.0,2.0}, {3.0,0.0}};
        int64_t yidx[] = {1, 2, 4};
        double ydata[][2] = {{2.0,1.0}, {0.0,2.0}, {1.0,0.0}};
        int64_t num_nonzeros = sizeof(xdata) / sizeof(*xdata);
        err = mtxvector_packed_init_complex_double(
            &x, mtxvector_base, size, num_nonzeros, xidx, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_packed_init_complex_double(
            &y, mtxvector_base, size, num_nonzeros, yidx, ydata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_packed_swap(&x, &y);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(12, x.size);
        TEST_ASSERT_EQ(3, x.num_nonzeros);
        TEST_ASSERT_EQ(x.idx[0], 1);
        TEST_ASSERT_EQ(x.idx[1], 2);
        TEST_ASSERT_EQ(x.idx[2], 4);
        TEST_ASSERT_EQ(12, y.size);
        TEST_ASSERT_EQ(3, y.num_nonzeros);
        TEST_ASSERT_EQ(y.idx[0], 0);
        TEST_ASSERT_EQ(y.idx[1], 3);
        TEST_ASSERT_EQ(y.idx[2], 5);
        TEST_ASSERT_EQ(mtxvector_base, x.x.type);
        TEST_ASSERT_EQ(mtx_field_complex, x.x.storage.base.field);
        TEST_ASSERT_EQ(mtx_double, x.x.storage.base.precision);
        TEST_ASSERT_EQ(3, x.x.storage.base.size);
        TEST_ASSERT_EQ(2.0, x.x.storage.base.data.complex_double[0][0]);
        TEST_ASSERT_EQ(1.0, x.x.storage.base.data.complex_double[0][1]);
        TEST_ASSERT_EQ(0.0, x.x.storage.base.data.complex_double[1][0]);
        TEST_ASSERT_EQ(2.0, x.x.storage.base.data.complex_double[1][1]);
        TEST_ASSERT_EQ(1.0, x.x.storage.base.data.complex_double[2][0]);
        TEST_ASSERT_EQ(0.0, x.x.storage.base.data.complex_double[2][1]);
        TEST_ASSERT_EQ(mtxvector_base, y.x.type);
        TEST_ASSERT_EQ(mtx_field_complex, y.x.storage.base.field);
        TEST_ASSERT_EQ(mtx_double, y.x.storage.base.precision);
        TEST_ASSERT_EQ(3, y.x.storage.base.size);
        TEST_ASSERT_EQ(1.0, y.x.storage.base.data.complex_double[0][0]);
        TEST_ASSERT_EQ(1.0, y.x.storage.base.data.complex_double[0][1]);
        TEST_ASSERT_EQ(1.0, y.x.storage.base.data.complex_double[1][0]);
        TEST_ASSERT_EQ(2.0, y.x.storage.base.data.complex_double[1][1]);
        TEST_ASSERT_EQ(3.0, y.x.storage.base.data.complex_double[2][0]);
        TEST_ASSERT_EQ(0.0, y.x.storage.base.data.complex_double[2][1]);
        mtxvector_packed_free(&y);
        mtxvector_packed_free(&x);
    }
    return TEST_SUCCESS;
}

/**
 * ‘test_mtxvector_packed_copy()’ tests copying values from one vector
 * to another.
 */
int test_mtxvector_packed_copy(void)
{
    int err;
    {
        struct mtxvector_packed x;
        struct mtxvector_packed y;
        int size = 12;
        int64_t xidx[] = {0, 3, 5, 6, 9};
        float xdata[] = {1.0f, 1.0f, 1.0f, 2.0f, 3.0f};
        int64_t yidx[] = {1, 2, 4, 6, 9};
        float ydata[] = {2.0f, 1.0f, 0.0f, 2.0f, 1.0f};
        int num_nonzeros = sizeof(xdata) / sizeof(*xdata);
        err = mtxvector_packed_init_real_single(
            &x, mtxvector_base, size, num_nonzeros, xidx, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_packed_init_real_single(
            &y, mtxvector_base, size, num_nonzeros, yidx, ydata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_packed_copy(&y, &x);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(12, y.size);
        TEST_ASSERT_EQ(5, y.num_nonzeros);
        TEST_ASSERT_EQ(y.idx[0], 0);
        TEST_ASSERT_EQ(y.idx[1], 3);
        TEST_ASSERT_EQ(y.idx[2], 5);
        TEST_ASSERT_EQ(y.idx[3], 6);
        TEST_ASSERT_EQ(y.idx[4], 9);
        TEST_ASSERT_EQ(mtxvector_base, y.x.type);
        TEST_ASSERT_EQ(mtx_field_real, y.x.storage.base.field);
        TEST_ASSERT_EQ(mtx_single, y.x.storage.base.precision);
        TEST_ASSERT_EQ(5, y.x.storage.base.size);
        TEST_ASSERT_EQ(1.0f, y.x.storage.base.data.real_single[0]);
        TEST_ASSERT_EQ(1.0f, y.x.storage.base.data.real_single[1]);
        TEST_ASSERT_EQ(1.0f, y.x.storage.base.data.real_single[2]);
        TEST_ASSERT_EQ(2.0f, y.x.storage.base.data.real_single[3]);
        TEST_ASSERT_EQ(3.0f, y.x.storage.base.data.real_single[4]);
        mtxvector_packed_free(&y);
        mtxvector_packed_free(&x);
    }
    {
        struct mtxvector_packed x;
        struct mtxvector_packed y;
        int size = 12;
        int64_t xidx[] = {0, 3, 5, 6, 9};
        double xdata[] = {1.0, 1.0, 1.0, 2.0, 3.0};
        int64_t yidx[] = {1, 2, 4, 6, 9};
        double ydata[] = {2.0, 1.0, 0.0, 2.0, 1.0};
        int num_nonzeros = sizeof(xdata) / sizeof(*xdata);
        err = mtxvector_packed_init_real_double(
            &x, mtxvector_base, size, num_nonzeros, xidx, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_packed_init_real_double(
            &y, mtxvector_base, size, num_nonzeros, yidx, ydata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_packed_copy(&y, &x);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(12, y.size);
        TEST_ASSERT_EQ(5, y.num_nonzeros);
        TEST_ASSERT_EQ(y.idx[0], 0);
        TEST_ASSERT_EQ(y.idx[1], 3);
        TEST_ASSERT_EQ(y.idx[2], 5);
        TEST_ASSERT_EQ(y.idx[3], 6);
        TEST_ASSERT_EQ(y.idx[4], 9);
        TEST_ASSERT_EQ(mtxvector_base, y.x.type);
        TEST_ASSERT_EQ(mtx_field_real, y.x.storage.base.field);
        TEST_ASSERT_EQ(mtx_double, y.x.storage.base.precision);
        TEST_ASSERT_EQ(5, y.x.storage.base.size);
        TEST_ASSERT_EQ(1.0, y.x.storage.base.data.real_double[0]);
        TEST_ASSERT_EQ(1.0, y.x.storage.base.data.real_double[1]);
        TEST_ASSERT_EQ(1.0, y.x.storage.base.data.real_double[2]);
        TEST_ASSERT_EQ(2.0, y.x.storage.base.data.real_double[3]);
        TEST_ASSERT_EQ(3.0, y.x.storage.base.data.real_double[4]);
        mtxvector_packed_free(&y);
        mtxvector_packed_free(&x);
    }
    {
        struct mtxvector_packed x;
        struct mtxvector_packed y;
        int size = 12;
        int64_t xidx[] = {0, 3, 5};
        float xdata[][2] = {{1.0f,1.0f}, {1.0f,2.0f}, {3.0f,0.0f}};
        int64_t yidx[] = {1, 2, 4};
        float ydata[][2] = {{2.0f,1.0f}, {0.0f,2.0f}, {1.0f,0.0f}};
        int64_t num_nonzeros = sizeof(xdata) / sizeof(*xdata);
        err = mtxvector_packed_init_complex_single(
            &x, mtxvector_base, size, num_nonzeros, xidx, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_packed_init_complex_single(
            &y, mtxvector_base, size, num_nonzeros, yidx, ydata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_packed_copy(&y, &x);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(12, y.size);
        TEST_ASSERT_EQ(3, y.num_nonzeros);
        TEST_ASSERT_EQ(y.idx[0], 0);
        TEST_ASSERT_EQ(y.idx[1], 3);
        TEST_ASSERT_EQ(y.idx[2], 5);
        TEST_ASSERT_EQ(mtxvector_base, y.x.type);
        TEST_ASSERT_EQ(mtx_field_complex, y.x.storage.base.field);
        TEST_ASSERT_EQ(mtx_single, y.x.storage.base.precision);
        TEST_ASSERT_EQ(3, y.x.storage.base.size);
        TEST_ASSERT_EQ(1.0f, y.x.storage.base.data.complex_single[0][0]);
        TEST_ASSERT_EQ(1.0f, y.x.storage.base.data.complex_single[0][1]);
        TEST_ASSERT_EQ(1.0f, y.x.storage.base.data.complex_single[1][0]);
        TEST_ASSERT_EQ(2.0f, y.x.storage.base.data.complex_single[1][1]);
        TEST_ASSERT_EQ(3.0f, y.x.storage.base.data.complex_single[2][0]);
        TEST_ASSERT_EQ(0.0f, y.x.storage.base.data.complex_single[2][1]);
        mtxvector_packed_free(&y);
        mtxvector_packed_free(&x);
    }
    {
        struct mtxvector_packed x;
        struct mtxvector_packed y;
        int size = 12;
        int64_t xidx[] = {0, 3, 5};
        double xdata[][2] = {{1.0,1.0}, {1.0,2.0}, {3.0,0.0}};
        int64_t yidx[] = {1, 2, 4};
        double ydata[][2] = {{2.0,1.0}, {0.0,2.0}, {1.0,0.0}};
        int64_t num_nonzeros = sizeof(xdata) / sizeof(*xdata);
        err = mtxvector_packed_init_complex_double(
            &x, mtxvector_base, size, num_nonzeros, xidx, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_packed_init_complex_double(
            &y, mtxvector_base, size, num_nonzeros, yidx, ydata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_packed_copy(&y, &x);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(12, y.size);
        TEST_ASSERT_EQ(3, y.num_nonzeros);
        TEST_ASSERT_EQ(y.idx[0], 0);
        TEST_ASSERT_EQ(y.idx[1], 3);
        TEST_ASSERT_EQ(y.idx[2], 5);
        TEST_ASSERT_EQ(mtxvector_base, y.x.type);
        TEST_ASSERT_EQ(mtx_field_complex, y.x.storage.base.field);
        TEST_ASSERT_EQ(mtx_double, y.x.storage.base.precision);
        TEST_ASSERT_EQ(3, y.x.storage.base.size);
        TEST_ASSERT_EQ(1.0, y.x.storage.base.data.complex_double[0][0]);
        TEST_ASSERT_EQ(1.0, y.x.storage.base.data.complex_double[0][1]);
        TEST_ASSERT_EQ(1.0, y.x.storage.base.data.complex_double[1][0]);
        TEST_ASSERT_EQ(2.0, y.x.storage.base.data.complex_double[1][1]);
        TEST_ASSERT_EQ(3.0, y.x.storage.base.data.complex_double[2][0]);
        TEST_ASSERT_EQ(0.0, y.x.storage.base.data.complex_double[2][1]);
        mtxvector_packed_free(&y);
        mtxvector_packed_free(&x);
    }
    return TEST_SUCCESS;
}

/**
 * ‘test_mtxvector_packed_scal()’ tests scaling vectors by a constant.
 */
int test_mtxvector_packed_scal(void)
{
    int err;
    {
        struct mtxvector_packed x;
        int size = 12;
        int64_t idx[] = {0, 3, 5, 6, 9};
        float data[] = {1.0f, 1.0f, 1.0f, 2.0f, 3.0f};
        int num_nonzeros = sizeof(data) / sizeof(*data);
        err = mtxvector_packed_init_real_single(
            &x, mtxvector_base, size, num_nonzeros, idx, data);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_packed_sscal(2.0f, &x, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(2.0f, x.x.storage.base.data.real_single[0]);
        TEST_ASSERT_EQ(2.0f, x.x.storage.base.data.real_single[1]);
        TEST_ASSERT_EQ(2.0f, x.x.storage.base.data.real_single[2]);
        TEST_ASSERT_EQ(4.0f, x.x.storage.base.data.real_single[3]);
        TEST_ASSERT_EQ(6.0f, x.x.storage.base.data.real_single[4]);
        err = mtxvector_packed_dscal(2.0, &x, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( 4.0f, x.x.storage.base.data.real_single[0]);
        TEST_ASSERT_EQ( 4.0f, x.x.storage.base.data.real_single[1]);
        TEST_ASSERT_EQ( 4.0f, x.x.storage.base.data.real_single[2]);
        TEST_ASSERT_EQ( 8.0f, x.x.storage.base.data.real_single[3]);
        TEST_ASSERT_EQ(12.0f, x.x.storage.base.data.real_single[4]);
        mtxvector_packed_free(&x);
    }
    {
        struct mtxvector_packed x;
        int size = 12;
        int64_t idx[] = {0, 3, 5, 6, 9};
        double data[] = {1.0, 1.0, 1.0, 2.0, 3.0};
        int num_nonzeros = sizeof(data) / sizeof(*data);
        err = mtxvector_packed_init_real_double(
            &x, mtxvector_base, size, num_nonzeros, idx, data);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_packed_sscal(2.0f, &x, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(2.0f, x.x.storage.base.data.real_double[0]);
        TEST_ASSERT_EQ(2.0f, x.x.storage.base.data.real_double[1]);
        TEST_ASSERT_EQ(2.0f, x.x.storage.base.data.real_double[2]);
        TEST_ASSERT_EQ(4.0f, x.x.storage.base.data.real_double[3]);
        TEST_ASSERT_EQ(6.0f, x.x.storage.base.data.real_double[4]);
        err = mtxvector_packed_dscal(2.0, &x, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( 4.0f, x.x.storage.base.data.real_double[0]);
        TEST_ASSERT_EQ( 4.0f, x.x.storage.base.data.real_double[1]);
        TEST_ASSERT_EQ( 4.0f, x.x.storage.base.data.real_double[2]);
        TEST_ASSERT_EQ( 8.0f, x.x.storage.base.data.real_double[3]);
        TEST_ASSERT_EQ(12.0f, x.x.storage.base.data.real_double[4]);
        mtxvector_packed_free(&x);
    }
    {
        struct mtxvector_packed x;
        int size = 12;
        int64_t idx[] = {0, 3, 5, 6, 9};
        float data[][2] = {{1.0f,1.0f}, {1.0f,2.0f}, {3.0f,0.0f}};
        int64_t num_nonzeros = sizeof(data) / sizeof(*data);
        err = mtxvector_packed_init_complex_single(
            &x, mtxvector_base, size, num_nonzeros, idx, data);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_packed_sscal(2.0f, &x, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(2.0f, x.x.storage.base.data.complex_single[0][0]);
        TEST_ASSERT_EQ(2.0f, x.x.storage.base.data.complex_single[0][1]);
        TEST_ASSERT_EQ(2.0f, x.x.storage.base.data.complex_single[1][0]);
        TEST_ASSERT_EQ(4.0f, x.x.storage.base.data.complex_single[1][1]);
        TEST_ASSERT_EQ(6.0f, x.x.storage.base.data.complex_single[2][0]);
        TEST_ASSERT_EQ(0.0f, x.x.storage.base.data.complex_single[2][1]);
        err = mtxvector_packed_dscal(2.0, &x, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( 4.0f, x.x.storage.base.data.complex_single[0][0]);
        TEST_ASSERT_EQ( 4.0f, x.x.storage.base.data.complex_single[0][1]);
        TEST_ASSERT_EQ( 4.0f, x.x.storage.base.data.complex_single[1][0]);
        TEST_ASSERT_EQ( 8.0f, x.x.storage.base.data.complex_single[1][1]);
        TEST_ASSERT_EQ(12.0f, x.x.storage.base.data.complex_single[2][0]);
        TEST_ASSERT_EQ( 0.0f, x.x.storage.base.data.complex_single[2][1]);
        float as[2] = {2, 3};
        err = mtxvector_packed_cscal(as, &x, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( -4.0f, x.x.storage.base.data.complex_single[0][0]);
        TEST_ASSERT_EQ( 20.0f, x.x.storage.base.data.complex_single[0][1]);
        TEST_ASSERT_EQ(-16.0f, x.x.storage.base.data.complex_single[1][0]);
        TEST_ASSERT_EQ( 28.0f, x.x.storage.base.data.complex_single[1][1]);
        TEST_ASSERT_EQ( 24.0f, x.x.storage.base.data.complex_single[2][0]);
        TEST_ASSERT_EQ( 36.0f, x.x.storage.base.data.complex_single[2][1]);
        double ad[2] = {2, 3};
        err = mtxvector_packed_zscal(ad, &x, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( -68.0f, x.x.storage.base.data.complex_single[0][0]);
        TEST_ASSERT_EQ(  28.0f, x.x.storage.base.data.complex_single[0][1]);
        TEST_ASSERT_EQ(-116.0f, x.x.storage.base.data.complex_single[1][0]);
        TEST_ASSERT_EQ(   8.0f, x.x.storage.base.data.complex_single[1][1]);
        TEST_ASSERT_EQ( -60.0f, x.x.storage.base.data.complex_single[2][0]);
        TEST_ASSERT_EQ( 144.0f, x.x.storage.base.data.complex_single[2][1]);
        mtxvector_packed_free(&x);
    }
    {
        struct mtxvector_packed x;
        int size = 12;
        int64_t idx[] = {0, 3, 5, 6, 9};
        double data[][2] = {{1.0,1.0}, {1.0,2.0}, {3.0,0.0}};
        int64_t num_nonzeros = sizeof(data) / sizeof(*data);
        err = mtxvector_packed_init_complex_double(
            &x, mtxvector_base, size, num_nonzeros, idx, data);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_packed_sscal(2.0f, &x, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(2.0f, x.x.storage.base.data.complex_double[0][0]);
        TEST_ASSERT_EQ(2.0f, x.x.storage.base.data.complex_double[0][1]);
        TEST_ASSERT_EQ(2.0f, x.x.storage.base.data.complex_double[1][0]);
        TEST_ASSERT_EQ(4.0f, x.x.storage.base.data.complex_double[1][1]);
        TEST_ASSERT_EQ(6.0f, x.x.storage.base.data.complex_double[2][0]);
        TEST_ASSERT_EQ(0.0f, x.x.storage.base.data.complex_double[2][1]);
        err = mtxvector_packed_dscal(2.0, &x, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( 4.0f, x.x.storage.base.data.complex_double[0][0]);
        TEST_ASSERT_EQ( 4.0f, x.x.storage.base.data.complex_double[0][1]);
        TEST_ASSERT_EQ( 4.0f, x.x.storage.base.data.complex_double[1][0]);
        TEST_ASSERT_EQ( 8.0f, x.x.storage.base.data.complex_double[1][1]);
        TEST_ASSERT_EQ(12.0f, x.x.storage.base.data.complex_double[2][0]);
        TEST_ASSERT_EQ( 0.0f, x.x.storage.base.data.complex_double[2][1]);
        float as[2] = {2, 3};
        err = mtxvector_packed_cscal(as, &x, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( -4.0, x.x.storage.base.data.complex_double[0][0]);
        TEST_ASSERT_EQ( 20.0, x.x.storage.base.data.complex_double[0][1]);
        TEST_ASSERT_EQ(-16.0, x.x.storage.base.data.complex_double[1][0]);
        TEST_ASSERT_EQ( 28.0, x.x.storage.base.data.complex_double[1][1]);
        TEST_ASSERT_EQ( 24.0, x.x.storage.base.data.complex_double[2][0]);
        TEST_ASSERT_EQ( 36.0, x.x.storage.base.data.complex_double[2][1]);
        double ad[2] = {2, 3};
        err = mtxvector_packed_zscal(ad, &x, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( -68.0, x.x.storage.base.data.complex_double[0][0]);
        TEST_ASSERT_EQ(  28.0, x.x.storage.base.data.complex_double[0][1]);
        TEST_ASSERT_EQ(-116.0, x.x.storage.base.data.complex_double[1][0]);
        TEST_ASSERT_EQ(   8.0, x.x.storage.base.data.complex_double[1][1]);
        TEST_ASSERT_EQ( -60.0, x.x.storage.base.data.complex_double[2][0]);
        TEST_ASSERT_EQ( 144.0, x.x.storage.base.data.complex_double[2][1]);
        mtxvector_packed_free(&x);
    }
    return TEST_SUCCESS;
}

/**
 * ‘test_mtxvector_packed_axpy()’ tests multiplying a vector by a
 * constant and adding the result to another vector.
 */
int test_mtxvector_packed_axpy(void)
{
    int err;
    {
        struct mtxvector_packed x;
        struct mtxvector_packed y;
        int size = 12;
        int64_t idx[] = {0, 3, 5, 6, 9};
        float xdata[] = {1.0f, 1.0f, 1.0f, 2.0f, 3.0f};
        float ydata[] = {2.0f, 1.0f, 0.0f, 2.0f, 1.0f};
        int num_nonzeros = sizeof(xdata) / sizeof(*xdata);
        err = mtxvector_packed_init_real_single(
            &x, mtxvector_base, size, num_nonzeros, idx, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_packed_init_real_single(
            &y, mtxvector_base, size, num_nonzeros, idx, ydata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_packed_saxpy(2.0f, &x, &y, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( 4.0f, y.x.storage.base.data.real_single[0]);
        TEST_ASSERT_EQ( 3.0f, y.x.storage.base.data.real_single[1]);
        TEST_ASSERT_EQ( 2.0f, y.x.storage.base.data.real_single[2]);
        TEST_ASSERT_EQ( 6.0f, y.x.storage.base.data.real_single[3]);
        TEST_ASSERT_EQ( 7.0f, y.x.storage.base.data.real_single[4]);
        err = mtxvector_packed_daxpy(2.0, &x, &y, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( 6.0f, y.x.storage.base.data.real_single[0]);
        TEST_ASSERT_EQ( 5.0f, y.x.storage.base.data.real_single[1]);
        TEST_ASSERT_EQ( 4.0f, y.x.storage.base.data.real_single[2]);
        TEST_ASSERT_EQ(10.0f, y.x.storage.base.data.real_single[3]);
        TEST_ASSERT_EQ(13.0f, y.x.storage.base.data.real_single[4]);
        err = mtxvector_packed_saypx(2.0f, &y, &x, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(13.0f, y.x.storage.base.data.real_single[0]);
        TEST_ASSERT_EQ(11.0f, y.x.storage.base.data.real_single[1]);
        TEST_ASSERT_EQ( 9.0f, y.x.storage.base.data.real_single[2]);
        TEST_ASSERT_EQ(22.0f, y.x.storage.base.data.real_single[3]);
        TEST_ASSERT_EQ(29.0f, y.x.storage.base.data.real_single[4]);
        err = mtxvector_packed_daypx(2.0, &y, &x, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(27.0f, y.x.storage.base.data.real_single[0]);
        TEST_ASSERT_EQ(23.0f, y.x.storage.base.data.real_single[1]);
        TEST_ASSERT_EQ(19.0f, y.x.storage.base.data.real_single[2]);
        TEST_ASSERT_EQ(46.0f, y.x.storage.base.data.real_single[3]);
        TEST_ASSERT_EQ(61.0f, y.x.storage.base.data.real_single[4]);
        mtxvector_packed_free(&y);
        mtxvector_packed_free(&x);
    }
    {
        struct mtxvector_packed x;
        struct mtxvector_packed y;
        int size = 12;
        int64_t idx[] = {0, 3, 5, 6, 9};
        double xdata[] = {1.0, 1.0, 1.0, 2.0, 3.0};
        double ydata[] = {2.0, 1.0, 0.0, 2.0, 1.0};
        int num_nonzeros = sizeof(xdata) / sizeof(*xdata);
        err = mtxvector_packed_init_real_double(
            &x, mtxvector_base, size, num_nonzeros, idx, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_packed_init_real_double(
            &y, mtxvector_base, size, num_nonzeros, idx, ydata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_packed_saxpy(2.0f, &x, &y, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( 4.0, y.x.storage.base.data.real_double[0]);
        TEST_ASSERT_EQ( 3.0, y.x.storage.base.data.real_double[1]);
        TEST_ASSERT_EQ( 2.0, y.x.storage.base.data.real_double[2]);
        TEST_ASSERT_EQ( 6.0, y.x.storage.base.data.real_double[3]);
        TEST_ASSERT_EQ( 7.0, y.x.storage.base.data.real_double[4]);
        err = mtxvector_packed_daxpy(2.0, &x, &y, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( 6.0, y.x.storage.base.data.real_double[0]);
        TEST_ASSERT_EQ( 5.0, y.x.storage.base.data.real_double[1]);
        TEST_ASSERT_EQ( 4.0, y.x.storage.base.data.real_double[2]);
        TEST_ASSERT_EQ(10.0, y.x.storage.base.data.real_double[3]);
        TEST_ASSERT_EQ(13.0, y.x.storage.base.data.real_double[4]);
        err = mtxvector_packed_saypx(2.0f, &y, &x, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(13.0, y.x.storage.base.data.real_double[0]);
        TEST_ASSERT_EQ(11.0, y.x.storage.base.data.real_double[1]);
        TEST_ASSERT_EQ( 9.0, y.x.storage.base.data.real_double[2]);
        TEST_ASSERT_EQ(22.0, y.x.storage.base.data.real_double[3]);
        TEST_ASSERT_EQ(29.0, y.x.storage.base.data.real_double[4]);
        err = mtxvector_packed_daypx(2.0, &y, &x, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(27.0, y.x.storage.base.data.real_double[0]);
        TEST_ASSERT_EQ(23.0, y.x.storage.base.data.real_double[1]);
        TEST_ASSERT_EQ(19.0, y.x.storage.base.data.real_double[2]);
        TEST_ASSERT_EQ(46.0, y.x.storage.base.data.real_double[3]);
        TEST_ASSERT_EQ(61.0, y.x.storage.base.data.real_double[4]);
        mtxvector_packed_free(&y);
        mtxvector_packed_free(&x);
    }
    {
        struct mtxvector_packed x;
        struct mtxvector_packed y;
        int size = 12;
        int64_t idx[] = {0, 3, 5, 6, 9};
        float xdata[][2] = {{1.0f,1.0f}, {1.0f,2.0f}, {3.0f,0.0f}};
        float ydata[][2] = {{2.0f,1.0f}, {0.0f,2.0f}, {1.0f,0.0f}};
        int64_t num_nonzeros = sizeof(xdata) / sizeof(*xdata);
        err = mtxvector_packed_init_complex_single(
            &x, mtxvector_base, size, num_nonzeros, idx, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_packed_init_complex_single(
            &y, mtxvector_base, size, num_nonzeros, idx, ydata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_packed_saxpy(2.0f, &x, &y, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(4.0f, y.x.storage.base.data.complex_single[0][0]);
        TEST_ASSERT_EQ(3.0f, y.x.storage.base.data.complex_single[0][1]);
        TEST_ASSERT_EQ(2.0f, y.x.storage.base.data.complex_single[1][0]);
        TEST_ASSERT_EQ(6.0f, y.x.storage.base.data.complex_single[1][1]);
        TEST_ASSERT_EQ(7.0f, y.x.storage.base.data.complex_single[2][0]);
        TEST_ASSERT_EQ(0.0f, y.x.storage.base.data.complex_single[2][1]);
        err = mtxvector_packed_daxpy(2.0, &x, &y, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( 6.0f, y.x.storage.base.data.complex_single[0][0]);
        TEST_ASSERT_EQ( 5.0f, y.x.storage.base.data.complex_single[0][1]);
        TEST_ASSERT_EQ( 4.0f, y.x.storage.base.data.complex_single[1][0]);
        TEST_ASSERT_EQ(10.0f, y.x.storage.base.data.complex_single[1][1]);
        TEST_ASSERT_EQ(13.0f, y.x.storage.base.data.complex_single[2][0]);
        TEST_ASSERT_EQ( 0.0f, y.x.storage.base.data.complex_single[2][1]);
        err = mtxvector_packed_saypx(2.0f, &y, &x, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(13.0f, y.x.storage.base.data.complex_single[0][0]);
        TEST_ASSERT_EQ(11.0f, y.x.storage.base.data.complex_single[0][1]);
        TEST_ASSERT_EQ( 9.0f, y.x.storage.base.data.complex_single[1][0]);
        TEST_ASSERT_EQ(22.0f, y.x.storage.base.data.complex_single[1][1]);
        TEST_ASSERT_EQ(29.0f, y.x.storage.base.data.complex_single[2][0]);
        TEST_ASSERT_EQ( 0.0f, y.x.storage.base.data.complex_single[2][1]);
        err = mtxvector_packed_daypx(2.0, &y, &x, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(27.0f, y.x.storage.base.data.complex_single[0][0]);
        TEST_ASSERT_EQ(23.0f, y.x.storage.base.data.complex_single[0][1]);
        TEST_ASSERT_EQ(19.0f, y.x.storage.base.data.complex_single[1][0]);
        TEST_ASSERT_EQ(46.0f, y.x.storage.base.data.complex_single[1][1]);
        TEST_ASSERT_EQ(61.0f, y.x.storage.base.data.complex_single[2][0]);
        TEST_ASSERT_EQ( 0.0f, y.x.storage.base.data.complex_single[2][1]);
        mtxvector_packed_free(&y);
        mtxvector_packed_free(&x);
    }
    {
        struct mtxvector_packed x;
        struct mtxvector_packed y;
        int size = 12;
        int64_t idx[] = {0, 3, 5, 6, 9};
        double xdata[][2] = {{1.0,1.0}, {1.0,2.0}, {3.0,0.0}};
        double ydata[][2] = {{2.0,1.0}, {0.0,2.0}, {1.0,0.0}};
        int64_t num_nonzeros = sizeof(xdata) / sizeof(*xdata);
        err = mtxvector_packed_init_complex_double(
            &x, mtxvector_base, size, num_nonzeros, idx, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_packed_init_complex_double(
            &y, mtxvector_base, size, num_nonzeros, idx, ydata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_packed_saxpy(2.0f, &x, &y, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(4.0, y.x.storage.base.data.complex_double[0][0]);
        TEST_ASSERT_EQ(3.0, y.x.storage.base.data.complex_double[0][1]);
        TEST_ASSERT_EQ(2.0, y.x.storage.base.data.complex_double[1][0]);
        TEST_ASSERT_EQ(6.0, y.x.storage.base.data.complex_double[1][1]);
        TEST_ASSERT_EQ(7.0, y.x.storage.base.data.complex_double[2][0]);
        TEST_ASSERT_EQ(0.0, y.x.storage.base.data.complex_double[2][1]);
        err = mtxvector_packed_daxpy(2.0, &x, &y, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ( 6.0, y.x.storage.base.data.complex_double[0][0]);
        TEST_ASSERT_EQ( 5.0, y.x.storage.base.data.complex_double[0][1]);
        TEST_ASSERT_EQ( 4.0, y.x.storage.base.data.complex_double[1][0]);
        TEST_ASSERT_EQ(10.0, y.x.storage.base.data.complex_double[1][1]);
        TEST_ASSERT_EQ(13.0, y.x.storage.base.data.complex_double[2][0]);
        TEST_ASSERT_EQ( 0.0, y.x.storage.base.data.complex_double[2][1]);
        err = mtxvector_packed_saypx(2.0f, &y, &x, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(13.0, y.x.storage.base.data.complex_double[0][0]);
        TEST_ASSERT_EQ(11.0, y.x.storage.base.data.complex_double[0][1]);
        TEST_ASSERT_EQ( 9.0, y.x.storage.base.data.complex_double[1][0]);
        TEST_ASSERT_EQ(22.0, y.x.storage.base.data.complex_double[1][1]);
        TEST_ASSERT_EQ(29.0, y.x.storage.base.data.complex_double[2][0]);
        TEST_ASSERT_EQ( 0.0, y.x.storage.base.data.complex_double[2][1]);
        err = mtxvector_packed_daypx(2.0, &y, &x, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(27.0, y.x.storage.base.data.complex_double[0][0]);
        TEST_ASSERT_EQ(23.0, y.x.storage.base.data.complex_double[0][1]);
        TEST_ASSERT_EQ(19.0, y.x.storage.base.data.complex_double[1][0]);
        TEST_ASSERT_EQ(46.0, y.x.storage.base.data.complex_double[1][1]);
        TEST_ASSERT_EQ(61.0, y.x.storage.base.data.complex_double[2][0]);
        TEST_ASSERT_EQ( 0.0, y.x.storage.base.data.complex_double[2][1]);
        mtxvector_packed_free(&y);
        mtxvector_packed_free(&x);
    }
    return TEST_SUCCESS;
}

/**
 * ‘test_mtxvector_packed_dot()’ tests computing the dot products of
 * pairs of vectors.
 */
int test_mtxvector_packed_dot(void)
{
    int err;
    {
        int size = 12;
        int nnz = 5;
        int64_t idx[] = {1, 3, 5, 7, 9};
        struct mtxvector_packed x;
        float xdata[] = {1.0f, 1.0f, 1.0f, 2.0f, 3.0f};
        err = mtxvector_packed_init_real_single(&x, mtxvector_base, size, nnz, idx, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxvector_packed y;
        float ydata[] = {3.0f, 2.0f, 1.0f, 0.0f, 1.0f};
        err = mtxvector_packed_init_real_single(&y, mtxvector_base, size, nnz, idx, ydata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        float sdot;
        err = mtxvector_packed_sdot(&x, &y, &sdot, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, sdot);
        double ddot;
        err = mtxvector_packed_ddot(&x, &y, &ddot, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, ddot);
        float cdotu[2];
        err = mtxvector_packed_cdotu(&x, &y, &cdotu, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, cdotu[0]); TEST_ASSERT_EQ(0.0f, cdotu[1]);
        float cdotc[2];
        err = mtxvector_packed_cdotc(&x, &y, &cdotc, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, cdotc[0]); TEST_ASSERT_EQ(0.0f, cdotc[1]);
        double zdotu[2];
        err = mtxvector_packed_zdotu(&x, &y, &zdotu, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, zdotu[0]); TEST_ASSERT_EQ(0.0, zdotu[1]);
        double zdotc[2];
        err = mtxvector_packed_zdotc(&x, &y, &zdotc, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, zdotc[0]); TEST_ASSERT_EQ(0.0, zdotc[1]);
        mtxvector_packed_free(&y);
        mtxvector_packed_free(&x);
    }
    {
        int size = 12;
        int nnz = 5;
        int64_t idx[] = {1, 3, 5, 7, 9};
        struct mtxvector_packed x;
        double xdata[] = {1.0, 1.0, 1.0, 2.0, 3.0};
        err = mtxvector_packed_init_real_double(&x, mtxvector_base, size, nnz, idx, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxvector_packed y;
        double ydata[] = {3.0, 2.0, 1.0, 0.0, 1.0};
        err = mtxvector_packed_init_real_double(&y, mtxvector_base, size, nnz, idx, ydata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        float sdot;
        err = mtxvector_packed_sdot(&x, &y, &sdot, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, sdot);
        double ddot;
        err = mtxvector_packed_ddot(&x, &y, &ddot, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, ddot);
        float cdotu[2];
        err = mtxvector_packed_cdotu(&x, &y, &cdotu, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, cdotu[0]); TEST_ASSERT_EQ(0.0f, cdotu[1]);
        float cdotc[2];
        err = mtxvector_packed_cdotc(&x, &y, &cdotc, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, cdotc[0]); TEST_ASSERT_EQ(0.0f, cdotc[1]);
        double zdotu[2];
        err = mtxvector_packed_zdotu(&x, &y, &zdotu, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, zdotu[0]); TEST_ASSERT_EQ(0.0, zdotu[1]);
        double zdotc[2];
        err = mtxvector_packed_zdotc(&x, &y, &zdotc, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, zdotc[0]); TEST_ASSERT_EQ(0.0, zdotc[1]);
        mtxvector_packed_free(&y);
        mtxvector_packed_free(&x);
    }
    {
        int size = 12;
        int nnz = 3;
        int64_t idx[] = {1, 3, 5};
        struct mtxvector_packed x;
        float xdata[][2] = {{1.0f, 1.0f}, {1.0f, 2.0f}, {3.0f, 0.0f}};
        err = mtxvector_packed_init_complex_single(&x, mtxvector_base, size, nnz, idx, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxvector_packed y;
        float ydata[][2] = {{3.0f, 2.0f}, {1.0f, 0.0f}, {1.0f, 0.0f}};
        err = mtxvector_packed_init_complex_single(&y, mtxvector_base, size, nnz, idx, ydata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        float sdot;
        err = mtxvector_packed_sdot(&x, &y, &sdot, NULL);
        TEST_ASSERT_EQ_MSG(MTX_ERR_INVALID_FIELD, err, "%s", mtxstrerror(err));
        double ddot;
        err = mtxvector_packed_ddot(&x, &y, &ddot, NULL);
        TEST_ASSERT_EQ_MSG(MTX_ERR_INVALID_FIELD, err, "%s", mtxstrerror(err));
        float cdotu[2];
        err = mtxvector_packed_cdotu(&x, &y, &cdotu, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(5.0f, cdotu[0]); TEST_ASSERT_EQ(7.0f, cdotu[1]);
        float cdotc[2];
        err = mtxvector_packed_cdotc(&x, &y, &cdotc, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, cdotc[0]); TEST_ASSERT_EQ(-3.0f, cdotc[1]);
        double zdotu[2];
        err = mtxvector_packed_zdotu(&x, &y, &zdotu, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(5.0, zdotu[0]); TEST_ASSERT_EQ(7.0, zdotu[1]);
        double zdotc[2];
        err = mtxvector_packed_zdotc(&x, &y, &zdotc, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, zdotc[0]); TEST_ASSERT_EQ(-3.0, zdotc[1]);
        mtxvector_packed_free(&y);
        mtxvector_packed_free(&x);
    }
    {
        int size = 12;
        int nnz = 3;
        int64_t idx[] = {1, 3, 5};
        struct mtxvector_packed x;
        double xdata[][2] = {{1.0, 1.0}, {1.0, 2.0}, {3.0, 0.0}};
        err = mtxvector_packed_init_complex_double(&x, mtxvector_base, size, nnz, idx, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxvector_packed y;
        double ydata[][2] = {{3.0, 2.0}, {1.0, 0.0}, {1.0, 0.0}};
        err = mtxvector_packed_init_complex_double(&y, mtxvector_base, size, nnz, idx, ydata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        float sdot;
        err = mtxvector_packed_sdot(&x, &y, &sdot, NULL);
        TEST_ASSERT_EQ_MSG(MTX_ERR_INVALID_FIELD, err, "%s", mtxstrerror(err));
        double ddot;
        err = mtxvector_packed_ddot(&x, &y, &ddot, NULL);
        TEST_ASSERT_EQ_MSG(MTX_ERR_INVALID_FIELD, err, "%s", mtxstrerror(err));
        float cdotu[2];
        err = mtxvector_packed_cdotu(&x, &y, &cdotu, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(5.0f, cdotu[0]); TEST_ASSERT_EQ(7.0f, cdotu[1]);
        float cdotc[2];
        err = mtxvector_packed_cdotc(&x, &y, &cdotc, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, cdotc[0]); TEST_ASSERT_EQ(-3.0f, cdotc[1]);
        double zdotu[2];
        err = mtxvector_packed_zdotu(&x, &y, &zdotu, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(5.0, zdotu[0]); TEST_ASSERT_EQ(7.0, zdotu[1]);
        double zdotc[2];
        err = mtxvector_packed_zdotc(&x, &y, &zdotc, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, zdotc[0]); TEST_ASSERT_EQ(-3.0, zdotc[1]);
        mtxvector_packed_free(&y);
        mtxvector_packed_free(&x);
    }
    {
        int size = 12;
        int nnz = 5;
        int64_t idx[] = {1, 3, 5, 7, 9};
        struct mtxvector_packed x;
        int32_t xdata[] = {1, 1, 1, 2, 3};
        err = mtxvector_packed_init_integer_single(&x, mtxvector_base, size, nnz, idx, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxvector_packed y;
        int32_t ydata[] = {3, 2, 1, 0, 1};
        err = mtxvector_packed_init_integer_single(&y, mtxvector_base, size, nnz, idx, ydata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        float sdot;
        err = mtxvector_packed_sdot(&x, &y, &sdot, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, sdot);
        double ddot;
        err = mtxvector_packed_ddot(&x, &y, &ddot, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, ddot);
        float cdotu[2];
        err = mtxvector_packed_cdotu(&x, &y, &cdotu, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, cdotu[0]); TEST_ASSERT_EQ(0.0f, cdotu[1]);
        float cdotc[2];
        err = mtxvector_packed_cdotc(&x, &y, &cdotc, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, cdotc[0]); TEST_ASSERT_EQ(0.0f, cdotc[1]);
        double zdotu[2];
        err = mtxvector_packed_zdotu(&x, &y, &zdotu, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, zdotu[0]); TEST_ASSERT_EQ(0.0, zdotu[1]);
        double zdotc[2];
        err = mtxvector_packed_zdotc(&x, &y, &zdotc, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, zdotc[0]); TEST_ASSERT_EQ(0.0, zdotc[1]);
        mtxvector_packed_free(&y);
        mtxvector_packed_free(&x);
    }
    {
        int size = 12;
        int nnz = 5;
        int64_t idx[] = {1, 3, 5, 7, 9};
        int64_t xdata[] = {1, 1, 1, 2, 3};
        struct mtxvector_packed x;
        err = mtxvector_packed_init_integer_double(&x, mtxvector_base, size, nnz, idx, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxvector_packed y;
        int64_t ydata[] = {3, 2, 1, 0, 1};
        err = mtxvector_packed_init_integer_double(&y, mtxvector_base, size, nnz, idx, ydata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        float sdot;
        err = mtxvector_packed_sdot(&x, &y, &sdot, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, sdot);
        double ddot;
        err = mtxvector_packed_ddot(&x, &y, &ddot, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, ddot);
        float cdotu[2];
        err = mtxvector_packed_cdotu(&x, &y, &cdotu, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, cdotu[0]); TEST_ASSERT_EQ(0.0f, cdotu[1]);
        float cdotc[2];
        err = mtxvector_packed_cdotc(&x, &y, &cdotc, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, cdotc[0]); TEST_ASSERT_EQ(0.0f, cdotc[1]);
        double zdotu[2];
        err = mtxvector_packed_zdotu(&x, &y, &zdotu, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, zdotu[0]); TEST_ASSERT_EQ(0.0, zdotu[1]);
        double zdotc[2];
        err = mtxvector_packed_zdotc(&x, &y, &zdotc, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, zdotc[0]); TEST_ASSERT_EQ(0.0, zdotc[1]);
        mtxvector_packed_free(&y);
        mtxvector_packed_free(&x);
    }
    return TEST_SUCCESS;
}

/**
 * ‘test_mtxvector_packed_nrm2()’ tests computing the Euclidean norm
 * of vectors.
 */
int test_mtxvector_packed_nrm2(void)
{
    int err;
    {
        struct mtxvector_packed x;
        int size = 12;
        int64_t idx[] = {0, 3, 5, 6, 9};
        float data[] = {1.0f, 1.0f, 1.0f, 2.0f, 3.0f};
        int num_nonzeros = sizeof(data) / sizeof(*data);
        err = mtxvector_packed_init_real_single(
            &x, mtxvector_base, size, num_nonzeros, idx, data);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        float snrm2;
        err = mtxvector_packed_snrm2(&x, &snrm2, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(4.0f, snrm2);
        double dnrm2;
        err = mtxvector_packed_dnrm2(&x, &dnrm2, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(4.0, dnrm2);
        mtxvector_packed_free(&x);
    }
    {
        struct mtxvector_packed x;
        int size = 12;
        int64_t idx[] = {0, 3, 5, 6, 9};
        double data[] = {1.0, 1.0, 1.0, 2.0, 3.0};
        int num_nonzeros = sizeof(data) / sizeof(*data);
        err = mtxvector_packed_init_real_double(
            &x, mtxvector_base, size, num_nonzeros, idx, data);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        float snrm2;
        err = mtxvector_packed_snrm2(&x, &snrm2, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(4.0f, snrm2);
        double dnrm2;
        err = mtxvector_packed_dnrm2(&x, &dnrm2, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(4.0, dnrm2);
        mtxvector_packed_free(&x);
    }
    {
        struct mtxvector_packed x;
        int size = 12;
        int64_t idx[] = {0, 3, 5, 6, 9};
        float data[][2] = {{1.0f,1.0f}, {1.0f,2.0f}, {3.0f,0.0f}};
        int64_t num_nonzeros = sizeof(data) / sizeof(*data);
        err = mtxvector_packed_init_complex_single(
            &x, mtxvector_base, size, num_nonzeros, idx, data);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        float snrm2;
        err = mtxvector_packed_snrm2(&x, &snrm2, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(4.0f, snrm2);
        double dnrm2;
        err = mtxvector_packed_dnrm2(&x, &dnrm2, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(4.0, dnrm2);
        mtxvector_packed_free(&x);
    }
    {
        struct mtxvector_packed x;
        int size = 12;
        int64_t idx[] = {0, 3, 5, 6, 9};
        double data[][2] = {{1.0,1.0}, {1.0,2.0}, {3.0,0.0}};
        int64_t num_nonzeros = sizeof(data) / sizeof(*data);
        err = mtxvector_packed_init_complex_double(
            &x, mtxvector_base, size, num_nonzeros, idx, data);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        float snrm2;
        err = mtxvector_packed_snrm2(&x, &snrm2, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(4.0f, snrm2);
        double dnrm2;
        err = mtxvector_packed_dnrm2(&x, &dnrm2, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(4.0, dnrm2);
        mtxvector_packed_free(&x);
    }
    return TEST_SUCCESS;
}

/**
 * ‘test_mtxvector_packed_asum()’ tests computing the sum of
 * absolute values of vectors.
 */
int test_mtxvector_packed_asum(void)
{
    int err;
    {
        struct mtxvector_packed x;
        int size = 12;
        int64_t idx[] = {0, 3, 5, 6, 9};
        float data[] = {-1.0f, 1.0f, 1.0f, 2.0f, 3.0f};
        int num_nonzeros = sizeof(data) / sizeof(*data);
        err = mtxvector_packed_init_real_single(
            &x, mtxvector_base, size, num_nonzeros, idx, data);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        float sasum;
        err = mtxvector_packed_sasum(&x, &sasum, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(8.0f, sasum);
        double dasum;
        err = mtxvector_packed_dasum(&x, &dasum, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(8.0, dasum);
        mtxvector_packed_free(&x);
    }
    {
        struct mtxvector_packed x;
        int size = 12;
        int64_t idx[] = {0, 3, 5, 6, 9};
        double data[] = {-1.0, 1.0, 1.0, 2.0, 3.0};
        int num_nonzeros = sizeof(data) / sizeof(*data);
        err = mtxvector_packed_init_real_double(
            &x, mtxvector_base, size, num_nonzeros, idx, data);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        float sasum;
        err = mtxvector_packed_sasum(&x, &sasum, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(8.0f, sasum);
        double dasum;
        err = mtxvector_packed_dasum(&x, &dasum, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(8.0, dasum);
        mtxvector_packed_free(&x);
    }
    {
        struct mtxvector_packed x;
        int size = 12;
        int64_t idx[] = {0, 3, 5, 6, 9};
        float data[][2] = {{-1.0f,-1.0f}, {1.0f,2.0f}, {3.0f,0.0f}};
        int64_t num_nonzeros = sizeof(data) / sizeof(*data);
        err = mtxvector_packed_init_complex_single(
            &x, mtxvector_base, size, num_nonzeros, idx, data);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        float sasum;
        err = mtxvector_packed_sasum(&x, &sasum, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(8.0f, sasum);
        double dasum;
        err = mtxvector_packed_dasum(&x, &dasum, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(8.0, dasum);
        mtxvector_packed_free(&x);
    }
    {
        struct mtxvector_packed x;
        int size = 12;
        int64_t idx[] = {0, 3, 5, 6, 9};
        double data[][2] = {{-1.0,-1.0}, {1.0,2.0}, {3.0,0.0}};
        int64_t num_nonzeros = sizeof(data) / sizeof(*data);
        err = mtxvector_packed_init_complex_double(
            &x, mtxvector_base, size, num_nonzeros, idx, data);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        float sasum;
        err = mtxvector_packed_sasum(&x, &sasum, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(8.0f, sasum);
        double dasum;
        err = mtxvector_packed_dasum(&x, &dasum, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(8.0, dasum);
        mtxvector_packed_free(&x);
    }
    return TEST_SUCCESS;
}

/**
 * ‘test_mtxvector_packed_iamax()’ tests computing the sum of
 * absolute values of vectors.
 */
int test_mtxvector_packed_iamax(void)
{
    int err;
    {
        struct mtxvector_packed x;
        int size = 12;
        int64_t idx[] = {0, 3, 5, 6, 9};
        float data[] = {-1.0f, 1.0f, 3.0f, 2.0f, 3.0f};
        int num_nonzeros = sizeof(data) / sizeof(*data);
        err = mtxvector_packed_init_real_single(
            &x, mtxvector_base, size, num_nonzeros, idx, data);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        int iamax;
        err = mtxvector_packed_iamax(&x, &iamax);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(2, iamax);
        mtxvector_packed_free(&x);
    }
    {
        struct mtxvector_packed x;
        int size = 12;
        int64_t idx[] = {0, 3, 5, 6, 9};
        double data[] = {-1.0, 1.0, 3.0, 2.0, 3.0};
        int num_nonzeros = sizeof(data) / sizeof(*data);
        err = mtxvector_packed_init_real_double(
            &x, mtxvector_base, size, num_nonzeros, idx, data);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        int iamax;
        err = mtxvector_packed_iamax(&x, &iamax);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(2, iamax);
        mtxvector_packed_free(&x);
    }
    {
        struct mtxvector_packed x;
        int size = 12;
        int64_t idx[] = {0, 3, 5, 6, 9};
        float data[][2] = {{-1.0f,-1.0f}, {1.0f,2.0f}, {3.0f,0.0f}};
        int64_t num_nonzeros = sizeof(data) / sizeof(*data);
        err = mtxvector_packed_init_complex_single(
            &x, mtxvector_base, size, num_nonzeros, idx, data);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        int iamax;
        err = mtxvector_packed_iamax(&x, &iamax);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(1, iamax);
        mtxvector_packed_free(&x);
    }
    {
        struct mtxvector_packed x;
        int size = 12;
        int64_t idx[] = {0, 3, 5, 6, 9};
        double data[][2] = {{-1.0,-1.0}, {1.0,2.0}, {3.0,0.0}};
        int64_t num_nonzeros = sizeof(data) / sizeof(*data);
        err = mtxvector_packed_init_complex_double(
            &x, mtxvector_base, size, num_nonzeros, idx, data);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        int iamax;
        err = mtxvector_packed_iamax(&x, &iamax);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(1, iamax);
        mtxvector_packed_free(&x);
    }
    return TEST_SUCCESS;
}

/**
 * ‘main()’ entry point and test driver.
 */
int main(int argc, char * argv[])
{
    TEST_SUITE_BEGIN("Running tests for sparse vectors in packed storage format\n");
    TEST_RUN(test_mtxvector_packed_from_mtxfile);
    TEST_RUN(test_mtxvector_packed_to_mtxfile);
    TEST_RUN(test_mtxvector_packed_swap);
    TEST_RUN(test_mtxvector_packed_copy);
    TEST_RUN(test_mtxvector_packed_scal);
    TEST_RUN(test_mtxvector_packed_axpy);
    TEST_RUN(test_mtxvector_packed_dot);
    TEST_RUN(test_mtxvector_packed_nrm2);
    TEST_RUN(test_mtxvector_packed_asum);
    TEST_RUN(test_mtxvector_packed_iamax);
    TEST_SUITE_END();
    return (TEST_SUITE_STATUS == TEST_SUCCESS) ?
        EXIT_SUCCESS : EXIT_FAILURE;
}
