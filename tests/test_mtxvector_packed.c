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
 * Last modified: 2022-04-09
 *
 * Unit tests for sparse vectors in packed storage format.
 */

#include "test.h"

#include <libmtx/error.h>
#include <libmtx/mtxfile/mtxfile.h>
#include <libmtx/util/partition.h>
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
 * ‘main()’ entry point and test driver.
 */
int main(int argc, char * argv[])
{
    TEST_SUITE_BEGIN("Running tests for sparse vectors in packed storage format\n");
    TEST_RUN(test_mtxvector_packed_from_mtxfile);
    TEST_RUN(test_mtxvector_packed_to_mtxfile);
    TEST_SUITE_END();
    return (TEST_SUITE_STATUS == TEST_SUCCESS) ?
        EXIT_SUCCESS : EXIT_FAILURE;
}
