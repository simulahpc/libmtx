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
 * Last modified: 2022-05-04
 *
 * Unit tests for dense matrices.
 */

#include "test.h"

#include <libmtx/error.h>
#include <libmtx/matrix/dense.h>
#include <libmtx/matrix/matrix.h>
#include <libmtx/mtxfile/mtxfile.h>
#include <libmtx/vector/vector.h>

#include <errno.h>
#include <unistd.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * ‘test_mtxmatrix_dense_from_mtxfile()’ tests converting Matrix
 * Market files to matrices.
 */
int test_mtxmatrix_dense_from_mtxfile(void)
{
    int err;

    /* array formats */

    {
        int num_rows = 3;
        int num_columns = 3;
        const float mtxdata[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
        int64_t num_nonzeros = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile mtxfile;
        err = mtxfile_init_matrix_array_real_single(
            &mtxfile, mtxfile_general, num_rows, num_columns, mtxdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxmatrix x;
        err = mtxmatrix_from_mtxfile(&x, mtxmatrix_dense, &mtxfile);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxmatrix_dense, x.type);
        const struct mtxmatrix_dense * x_ = &x.storage.dense;
        TEST_ASSERT_EQ(mtx_unsymmetric, x_->symmetry);
        TEST_ASSERT_EQ(3, x_->num_rows);
        TEST_ASSERT_EQ(3, x_->num_columns);
        TEST_ASSERT_EQ(9, x_->num_nonzeros);
        TEST_ASSERT_EQ(9, x_->size);
        TEST_ASSERT_EQ(mtx_field_real, x_->a.field);
        TEST_ASSERT_EQ(mtx_single, x_->a.precision);
        TEST_ASSERT_EQ(x_->a.data.real_single[0], 1.0f);
        TEST_ASSERT_EQ(x_->a.data.real_single[1], 2.0f);
        TEST_ASSERT_EQ(x_->a.data.real_single[2], 3.0f);
        TEST_ASSERT_EQ(x_->a.data.real_single[3], 4.0f);
        TEST_ASSERT_EQ(x_->a.data.real_single[4], 5.0f);
        TEST_ASSERT_EQ(x_->a.data.real_single[5], 6.0f);
        TEST_ASSERT_EQ(x_->a.data.real_single[6], 7.0f);
        TEST_ASSERT_EQ(x_->a.data.real_single[7], 8.0f);
        TEST_ASSERT_EQ(x_->a.data.real_single[8], 9.0f);
        mtxmatrix_free(&x);
        mtxfile_free(&mtxfile);
    }
    {
        int num_rows = 3;
        int num_columns = 3;
        const float mtxdata[] = {1.0f, 2.0f, 3.0f, /*4.0f,*/ 5.0f, 6.0f, /*7.0f, 8.0f,*/ 9.0f};
        int64_t num_nonzeros = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile mtxfile;
        err = mtxfile_init_matrix_array_real_single(
            &mtxfile, mtxfile_symmetric, num_rows, num_columns, mtxdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxmatrix x;
        err = mtxmatrix_from_mtxfile(&x, mtxmatrix_dense, &mtxfile);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxmatrix_dense, x.type);
        const struct mtxmatrix_dense * x_ = &x.storage.dense;
        TEST_ASSERT_EQ(mtx_symmetric, x_->symmetry);
        TEST_ASSERT_EQ(3, x_->num_rows);
        TEST_ASSERT_EQ(3, x_->num_columns);
        TEST_ASSERT_EQ(9, x_->num_nonzeros);
        TEST_ASSERT_EQ(6, x_->size);
        TEST_ASSERT_EQ(mtx_field_real, x_->a.field);
        TEST_ASSERT_EQ(mtx_single, x_->a.precision);
        TEST_ASSERT_EQ(x_->a.data.real_single[0], 1.0f);
        TEST_ASSERT_EQ(x_->a.data.real_single[1], 2.0f);
        TEST_ASSERT_EQ(x_->a.data.real_single[2], 3.0f);
        TEST_ASSERT_EQ(x_->a.data.real_single[3], 5.0f);
        TEST_ASSERT_EQ(x_->a.data.real_single[4], 6.0f);
        TEST_ASSERT_EQ(x_->a.data.real_single[5], 9.0f);
        mtxmatrix_free(&x);
        mtxfile_free(&mtxfile);
    }
    {
        int num_rows = 3;
        int num_columns = 3;
        const float mtxdata[] = {/*1.0f,*/ 2.0f, 3.0f, /*4.0f, 5.0f,*/ 6.0f, /*7.0f, 8.0f, 9.0f*/};
        int64_t num_nonzeros = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile mtxfile;
        err = mtxfile_init_matrix_array_real_single(
            &mtxfile, mtxfile_skew_symmetric, num_rows, num_columns, mtxdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxmatrix x;
        err = mtxmatrix_from_mtxfile(&x, mtxmatrix_dense, &mtxfile);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxmatrix_dense, x.type);
        const struct mtxmatrix_dense * x_ = &x.storage.dense;
        TEST_ASSERT_EQ(mtx_skew_symmetric, x_->symmetry);
        TEST_ASSERT_EQ(3, x_->num_rows);
        TEST_ASSERT_EQ(3, x_->num_columns);
        TEST_ASSERT_EQ(6, x_->num_nonzeros);
        TEST_ASSERT_EQ(3, x_->size);
        TEST_ASSERT_EQ(mtx_field_real, x_->a.field);
        TEST_ASSERT_EQ(mtx_single, x_->a.precision);
        TEST_ASSERT_EQ(x_->a.data.real_single[0], 2.0f);
        TEST_ASSERT_EQ(x_->a.data.real_single[1], 3.0f);
        TEST_ASSERT_EQ(x_->a.data.real_single[2], 6.0f);
        mtxmatrix_free(&x);
        mtxfile_free(&mtxfile);
    }

    /* coordinate formats */

    {
        int num_rows = 3;
        int num_columns = 3;
        struct mtxfile_matrix_coordinate_real_single mtxdata[] = {
            {1,1,1.0f}, {1,3,3.0f}, {2,1,4.0f}, {3,3,9.0f}};
        int64_t size = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile mtxfile;
        err = mtxfile_init_matrix_coordinate_real_single(
            &mtxfile, mtxfile_general, num_rows, num_columns, size, mtxdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxmatrix x;
        err = mtxmatrix_from_mtxfile(&x, mtxmatrix_dense, &mtxfile);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxmatrix_dense, x.type);
        const struct mtxmatrix_dense * x_ = &x.storage.dense;
        TEST_ASSERT_EQ(mtx_unsymmetric, x_->symmetry);
        TEST_ASSERT_EQ(3, x_->num_rows);
        TEST_ASSERT_EQ(3, x_->num_columns);
        TEST_ASSERT_EQ(9, x_->num_nonzeros);
        TEST_ASSERT_EQ(9, x_->size);
        TEST_ASSERT_EQ(mtx_field_real, x_->a.field);
        TEST_ASSERT_EQ(mtx_single, x_->a.precision);
        TEST_ASSERT_EQ(x_->a.data.real_single[0], 1.0f);
        TEST_ASSERT_EQ(x_->a.data.real_single[1], 0.0f);
        TEST_ASSERT_EQ(x_->a.data.real_single[2], 3.0f);
        TEST_ASSERT_EQ(x_->a.data.real_single[3], 4.0f);
        TEST_ASSERT_EQ(x_->a.data.real_single[4], 0.0f);
        TEST_ASSERT_EQ(x_->a.data.real_single[5], 0.0f);
        TEST_ASSERT_EQ(x_->a.data.real_single[6], 0.0f);
        TEST_ASSERT_EQ(x_->a.data.real_single[7], 0.0f);
        TEST_ASSERT_EQ(x_->a.data.real_single[8], 9.0f);
        mtxmatrix_free(&x);
        mtxfile_free(&mtxfile);
    }
    {
        int num_rows = 3;
        int num_columns = 3;
        struct mtxfile_matrix_coordinate_real_single mtxdata[] = {
            {1,1,1.0f}, {1,3,3.0f}, {2,1,4.0f}, {3,3,9.0f}};
        int64_t size = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile mtxfile;
        err = mtxfile_init_matrix_coordinate_real_single(
            &mtxfile, mtxfile_symmetric, num_rows, num_columns, size, mtxdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxmatrix x;
        err = mtxmatrix_from_mtxfile(&x, mtxmatrix_dense, &mtxfile);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxmatrix_dense, x.type);
        const struct mtxmatrix_dense * x_ = &x.storage.dense;
        TEST_ASSERT_EQ(mtx_symmetric, x_->symmetry);
        TEST_ASSERT_EQ(3, x_->num_rows);
        TEST_ASSERT_EQ(3, x_->num_columns);
        TEST_ASSERT_EQ(9, x_->num_nonzeros);
        TEST_ASSERT_EQ(6, x_->size);
        TEST_ASSERT_EQ(mtx_field_real, x_->a.field);
        TEST_ASSERT_EQ(mtx_single, x_->a.precision);
        TEST_ASSERT_EQ(x_->a.data.real_single[0], 1.0f);
        TEST_ASSERT_EQ(x_->a.data.real_single[1], 4.0f);
        TEST_ASSERT_EQ(x_->a.data.real_single[2], 3.0f);
        TEST_ASSERT_EQ(x_->a.data.real_single[3], 0.0f);
        TEST_ASSERT_EQ(x_->a.data.real_single[4], 0.0f);
        TEST_ASSERT_EQ(x_->a.data.real_single[5], 9.0f);
        mtxmatrix_free(&x);
        mtxfile_free(&mtxfile);
    }
    {
        int num_rows = 3;
        int num_columns = 3;
        struct mtxfile_matrix_coordinate_real_single mtxdata[] = {
            {1,3,3.0f}, {2,1,4.0f}, {3,2,6.0f}};
        int64_t size = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile mtxfile;
        err = mtxfile_init_matrix_coordinate_real_single(
            &mtxfile, mtxfile_skew_symmetric, num_rows, num_columns, size,
            mtxdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxmatrix x;
        err = mtxmatrix_from_mtxfile(&x, mtxmatrix_dense, &mtxfile);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxmatrix_dense, x.type);
        const struct mtxmatrix_dense * x_ = &x.storage.dense;
        TEST_ASSERT_EQ(mtx_skew_symmetric, x_->symmetry);
        TEST_ASSERT_EQ(3, x_->num_rows);
        TEST_ASSERT_EQ(3, x_->num_columns);
        TEST_ASSERT_EQ(6, x_->num_nonzeros);
        TEST_ASSERT_EQ(3, x_->size);
        TEST_ASSERT_EQ(mtx_field_real, x_->a.field);
        TEST_ASSERT_EQ(mtx_single, x_->a.precision);
        TEST_ASSERT_EQ(x_->a.data.real_single[0], 4.0f);
        TEST_ASSERT_EQ(x_->a.data.real_single[1], 3.0f);
        TEST_ASSERT_EQ(x_->a.data.real_single[2], 6.0f);
        mtxmatrix_free(&x);
        mtxfile_free(&mtxfile);
    }
    {
        int num_rows = 3;
        int num_columns = 3;
        struct mtxfile_matrix_coordinate_real_double mtxdata[] = {
            {1,1,1.0}, {1,3,3.0}, {2,1,4.0}, {3,3,9.0}};
        int64_t size = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile mtxfile;
        err = mtxfile_init_matrix_coordinate_real_double(
            &mtxfile, mtxfile_general, num_rows, num_columns, size, mtxdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxmatrix x;
        err = mtxmatrix_from_mtxfile(&x, mtxmatrix_dense, &mtxfile);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxmatrix_dense, x.type);
        const struct mtxmatrix_dense * x_ = &x.storage.dense;
        TEST_ASSERT_EQ(mtx_unsymmetric, x_->symmetry);
        TEST_ASSERT_EQ(3, x_->num_rows);
        TEST_ASSERT_EQ(3, x_->num_columns);
        TEST_ASSERT_EQ(9, x_->num_nonzeros);
        TEST_ASSERT_EQ(9, x_->size);
        TEST_ASSERT_EQ(mtx_field_real, x_->a.field);
        TEST_ASSERT_EQ(mtx_double, x_->a.precision);
        TEST_ASSERT_EQ(x_->a.data.real_double[0], 1.0);
        TEST_ASSERT_EQ(x_->a.data.real_double[1], 0.0);
        TEST_ASSERT_EQ(x_->a.data.real_double[2], 3.0);
        TEST_ASSERT_EQ(x_->a.data.real_double[3], 4.0);
        TEST_ASSERT_EQ(x_->a.data.real_double[4], 0.0);
        TEST_ASSERT_EQ(x_->a.data.real_double[5], 0.0);
        TEST_ASSERT_EQ(x_->a.data.real_double[6], 0.0);
        TEST_ASSERT_EQ(x_->a.data.real_double[7], 0.0);
        TEST_ASSERT_EQ(x_->a.data.real_double[8], 9.0);
        mtxmatrix_free(&x);
        mtxfile_free(&mtxfile);
    }
    {
        int num_rows = 3;
        int num_columns = 3;
        struct mtxfile_matrix_coordinate_complex_single mtxdata[] = {
            {1,1,{1.0f,-1.0f}}, {1,3,{3.0f,-3.0f}},
            {2,1,{4.0f,-4.0f}}, {3,3,{9.0f,-9.0f}}};
        int64_t size = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile mtxfile;
        err = mtxfile_init_matrix_coordinate_complex_single(
            &mtxfile, mtxfile_general, num_rows, num_columns, size, mtxdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxmatrix x;
        err = mtxmatrix_from_mtxfile(&x, mtxmatrix_dense, &mtxfile);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxmatrix_dense, x.type);
        const struct mtxmatrix_dense * x_ = &x.storage.dense;
        TEST_ASSERT_EQ(mtx_unsymmetric, x_->symmetry);
        TEST_ASSERT_EQ(3, x_->num_rows);
        TEST_ASSERT_EQ(3, x_->num_columns);
        TEST_ASSERT_EQ(9, x_->num_nonzeros);
        TEST_ASSERT_EQ(9, x_->size);
        TEST_ASSERT_EQ(mtx_field_complex, x_->a.field);
        TEST_ASSERT_EQ(mtx_single, x_->a.precision);
        TEST_ASSERT_EQ(x_->a.data.complex_single[0][0], 1.0f);
        TEST_ASSERT_EQ(x_->a.data.complex_single[0][1],-1.0f);
        TEST_ASSERT_EQ(x_->a.data.complex_single[2][0], 3.0f);
        TEST_ASSERT_EQ(x_->a.data.complex_single[2][1],-3.0f);
        TEST_ASSERT_EQ(x_->a.data.complex_single[3][0], 4.0f);
        TEST_ASSERT_EQ(x_->a.data.complex_single[3][1],-4.0f);
        TEST_ASSERT_EQ(x_->a.data.complex_single[8][0], 9.0f);
        TEST_ASSERT_EQ(x_->a.data.complex_single[8][1],-9.0f);
        mtxmatrix_free(&x);
        mtxfile_free(&mtxfile);
    }
    {
        int num_rows = 3;
        int num_columns = 3;
        struct mtxfile_matrix_coordinate_complex_single mtxdata[] = {
            {1,1,{1.0f,-1.0f}}, {1,3,{3.0f,-3.0f}},
            {2,1,{4.0f,-4.0f}}, {3,3,{9.0f,-9.0f}}};
        int64_t size = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile mtxfile;
        err = mtxfile_init_matrix_coordinate_complex_single(
            &mtxfile, mtxfile_symmetric, num_rows, num_columns, size, mtxdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxmatrix x;
        err = mtxmatrix_from_mtxfile(&x, mtxmatrix_dense, &mtxfile);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxmatrix_dense, x.type);
        const struct mtxmatrix_dense * x_ = &x.storage.dense;
        TEST_ASSERT_EQ(mtx_symmetric, x_->symmetry);
        TEST_ASSERT_EQ(3, x_->num_rows);
        TEST_ASSERT_EQ(3, x_->num_columns);
        TEST_ASSERT_EQ(9, x_->num_nonzeros);
        TEST_ASSERT_EQ(6, x_->size);
        TEST_ASSERT_EQ(mtx_field_complex, x_->a.field);
        TEST_ASSERT_EQ(mtx_single, x_->a.precision);
        TEST_ASSERT_EQ(x_->a.data.complex_single[0][0], 1.0f);
        TEST_ASSERT_EQ(x_->a.data.complex_single[0][1],-1.0f);
        TEST_ASSERT_EQ(x_->a.data.complex_single[2][0], 3.0f);
        TEST_ASSERT_EQ(x_->a.data.complex_single[2][1],-3.0f);
        TEST_ASSERT_EQ(x_->a.data.complex_single[1][0], 4.0f);
        TEST_ASSERT_EQ(x_->a.data.complex_single[1][1],-4.0f);
        TEST_ASSERT_EQ(x_->a.data.complex_single[5][0], 9.0f);
        TEST_ASSERT_EQ(x_->a.data.complex_single[5][1],-9.0f);
        mtxmatrix_free(&x);
        mtxfile_free(&mtxfile);
    }
    {
        int num_rows = 3;
        int num_columns = 3;
        struct mtxfile_matrix_coordinate_complex_single mtxdata[] = {
            {1,2,{2.0f,-2.0f}}, {1,3,{3.0f,-3.0f}}, {2,3,{6.0f,-6.0f}}};
        int64_t size = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile mtxfile;
        err = mtxfile_init_matrix_coordinate_complex_single(
            &mtxfile, mtxfile_skew_symmetric, num_rows, num_columns, size,
            mtxdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxmatrix x;
        err = mtxmatrix_from_mtxfile(&x, mtxmatrix_dense, &mtxfile);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxmatrix_dense, x.type);
        const struct mtxmatrix_dense * x_ = &x.storage.dense;
        TEST_ASSERT_EQ(mtx_skew_symmetric, x_->symmetry);
        TEST_ASSERT_EQ(3, x_->num_rows);
        TEST_ASSERT_EQ(3, x_->num_columns);
        TEST_ASSERT_EQ(6, x_->num_nonzeros);
        TEST_ASSERT_EQ(3, x_->size);
        TEST_ASSERT_EQ(mtx_field_complex, x_->a.field);
        TEST_ASSERT_EQ(mtx_single, x_->a.precision);
        TEST_ASSERT_EQ(x_->a.data.complex_single[0][0], 2.0f);
        TEST_ASSERT_EQ(x_->a.data.complex_single[0][1],-2.0f);
        TEST_ASSERT_EQ(x_->a.data.complex_single[1][0], 3.0f);
        TEST_ASSERT_EQ(x_->a.data.complex_single[1][1],-3.0f);
        TEST_ASSERT_EQ(x_->a.data.complex_single[2][0], 6.0f);
        TEST_ASSERT_EQ(x_->a.data.complex_single[2][1],-6.0f);
        mtxmatrix_free(&x);
        mtxfile_free(&mtxfile);
    }
    {
        int num_rows = 3;
        int num_columns = 3;
        struct mtxfile_matrix_coordinate_complex_single mtxdata[] = {
            {1,1,{1.0f,-1.0f}}, {1,3,{3.0f,-3.0f}},
            {2,1,{4.0f,-4.0f}}, {3,3,{9.0f,-9.0f}}};
        int64_t size = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile mtxfile;
        err = mtxfile_init_matrix_coordinate_complex_single(
            &mtxfile, mtxfile_hermitian, num_rows, num_columns, size, mtxdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxmatrix x;
        err = mtxmatrix_from_mtxfile(&x, mtxmatrix_dense, &mtxfile);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxmatrix_dense, x.type);
        const struct mtxmatrix_dense * x_ = &x.storage.dense;
        TEST_ASSERT_EQ(mtx_hermitian, x_->symmetry);
        TEST_ASSERT_EQ(3, x_->num_rows);
        TEST_ASSERT_EQ(3, x_->num_columns);
        TEST_ASSERT_EQ(9, x_->num_nonzeros);
        TEST_ASSERT_EQ(6, x_->size);
        TEST_ASSERT_EQ(mtx_field_complex, x_->a.field);
        TEST_ASSERT_EQ(mtx_single, x_->a.precision);
        TEST_ASSERT_EQ(x_->a.data.complex_single[0][0], 1.0f);
        TEST_ASSERT_EQ(x_->a.data.complex_single[0][1],-1.0f);
        TEST_ASSERT_EQ(x_->a.data.complex_single[2][0], 3.0f);
        TEST_ASSERT_EQ(x_->a.data.complex_single[2][1],-3.0f);
        TEST_ASSERT_EQ(x_->a.data.complex_single[1][0], 4.0f);
        TEST_ASSERT_EQ(x_->a.data.complex_single[1][1],-4.0f);
        TEST_ASSERT_EQ(x_->a.data.complex_single[5][0], 9.0f);
        TEST_ASSERT_EQ(x_->a.data.complex_single[5][1],-9.0f);
        mtxmatrix_free(&x);
        mtxfile_free(&mtxfile);
    }
    {
        int num_rows = 3;
        int num_columns = 3;
        struct mtxfile_matrix_coordinate_complex_double mtxdata[] = {
            {1,1,{1.0,-1.0}}, {1,3,{3.0,-3.0}},
            {2,1,{4.0,-4.0}}, {3,3,{9.0,-9.0}}};
        int64_t size = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile mtxfile;
        err = mtxfile_init_matrix_coordinate_complex_double(
            &mtxfile, mtxfile_general, num_rows, num_columns, size, mtxdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxmatrix x;
        err = mtxmatrix_from_mtxfile(&x, mtxmatrix_dense, &mtxfile);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxmatrix_dense, x.type);
        const struct mtxmatrix_dense * x_ = &x.storage.dense;
        TEST_ASSERT_EQ(mtx_unsymmetric, x_->symmetry);
        TEST_ASSERT_EQ(3, x_->num_rows);
        TEST_ASSERT_EQ(3, x_->num_columns);
        TEST_ASSERT_EQ(9, x_->num_nonzeros);
        TEST_ASSERT_EQ(9, x_->size);
        TEST_ASSERT_EQ(mtx_field_complex, x_->a.field);
        TEST_ASSERT_EQ(mtx_double, x_->a.precision);
        TEST_ASSERT_EQ(x_->a.data.complex_double[0][0], 1.0);
        TEST_ASSERT_EQ(x_->a.data.complex_double[0][1],-1.0);
        TEST_ASSERT_EQ(x_->a.data.complex_double[2][0], 3.0);
        TEST_ASSERT_EQ(x_->a.data.complex_double[2][1],-3.0);
        TEST_ASSERT_EQ(x_->a.data.complex_double[3][0], 4.0);
        TEST_ASSERT_EQ(x_->a.data.complex_double[3][1],-4.0);
        TEST_ASSERT_EQ(x_->a.data.complex_double[8][0], 9.0);
        TEST_ASSERT_EQ(x_->a.data.complex_double[8][1],-9.0);
        mtxmatrix_free(&x);
        mtxfile_free(&mtxfile);
    }
    {
        int num_rows = 3;
        int num_columns = 3;
        struct mtxfile_matrix_coordinate_integer_single mtxdata[] = {
            {1,1,1}, {1,3,3}, {2,1,4}, {3,3,9}};
        int64_t size = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile mtxfile;
        err = mtxfile_init_matrix_coordinate_integer_single(
            &mtxfile, mtxfile_general, num_rows, num_columns, size, mtxdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxmatrix x;
        err = mtxmatrix_from_mtxfile(&x, mtxmatrix_dense, &mtxfile);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxmatrix_dense, x.type);
        const struct mtxmatrix_dense * x_ = &x.storage.dense;
        TEST_ASSERT_EQ(mtx_unsymmetric, x_->symmetry);
        TEST_ASSERT_EQ(3, x_->num_rows);
        TEST_ASSERT_EQ(3, x_->num_columns);
        TEST_ASSERT_EQ(9, x_->num_nonzeros);
        TEST_ASSERT_EQ(9, x_->size);
        TEST_ASSERT_EQ(mtx_field_integer, x_->a.field);
        TEST_ASSERT_EQ(mtx_single, x_->a.precision);
        TEST_ASSERT_EQ(x_->a.data.integer_single[0], 1);
        TEST_ASSERT_EQ(x_->a.data.integer_single[2], 3);
        TEST_ASSERT_EQ(x_->a.data.integer_single[3], 4);
        TEST_ASSERT_EQ(x_->a.data.integer_single[8], 9);
        mtxmatrix_free(&x);
        mtxfile_free(&mtxfile);
    }
    {
        int num_rows = 3;
        int num_columns = 3;
        struct mtxfile_matrix_coordinate_integer_single mtxdata[] = {
            {1,1,1}, {1,3,3}, {2,1,4}, {3,3,9}};
        int64_t size = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile mtxfile;
        err = mtxfile_init_matrix_coordinate_integer_single(
            &mtxfile, mtxfile_symmetric, num_rows, num_columns, size, mtxdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxmatrix x;
        err = mtxmatrix_from_mtxfile(&x, mtxmatrix_dense, &mtxfile);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxmatrix_dense, x.type);
        const struct mtxmatrix_dense * x_ = &x.storage.dense;
        TEST_ASSERT_EQ(mtx_symmetric, x_->symmetry);
        TEST_ASSERT_EQ(3, x_->num_rows);
        TEST_ASSERT_EQ(3, x_->num_columns);
        TEST_ASSERT_EQ(9, x_->num_nonzeros);
        TEST_ASSERT_EQ(6, x_->size);
        TEST_ASSERT_EQ(mtx_field_integer, x_->a.field);
        TEST_ASSERT_EQ(mtx_single, x_->a.precision);
        TEST_ASSERT_EQ(x_->a.data.integer_single[0], 1);
        TEST_ASSERT_EQ(x_->a.data.integer_single[2], 3);
        TEST_ASSERT_EQ(x_->a.data.integer_single[1], 4);
        TEST_ASSERT_EQ(x_->a.data.integer_single[5], 9);
        mtxmatrix_free(&x);
        mtxfile_free(&mtxfile);
    }
    {
        int num_rows = 3;
        int num_columns = 3;
        struct mtxfile_matrix_coordinate_integer_double mtxdata[] = {
            {1,1,1}, {1,3,3}, {2,1,4}, {3,3,9}};
        int64_t size = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile mtxfile;
        err = mtxfile_init_matrix_coordinate_integer_double(
            &mtxfile, mtxfile_general, num_rows, num_columns, size, mtxdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxmatrix x;
        err = mtxmatrix_from_mtxfile(&x, mtxmatrix_dense, &mtxfile);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxmatrix_dense, x.type);
        const struct mtxmatrix_dense * x_ = &x.storage.dense;
        TEST_ASSERT_EQ(mtx_unsymmetric, x_->symmetry);
        TEST_ASSERT_EQ(3, x_->num_rows);
        TEST_ASSERT_EQ(3, x_->num_columns);
        TEST_ASSERT_EQ(9, x_->num_nonzeros);
        TEST_ASSERT_EQ(9, x_->size);
        TEST_ASSERT_EQ(mtx_field_integer, x_->a.field);
        TEST_ASSERT_EQ(mtx_double, x_->a.precision);
        TEST_ASSERT_EQ(x_->a.data.integer_double[0], 1);
        TEST_ASSERT_EQ(x_->a.data.integer_double[2], 3);
        TEST_ASSERT_EQ(x_->a.data.integer_double[3], 4);
        TEST_ASSERT_EQ(x_->a.data.integer_double[8], 9);
        mtxmatrix_free(&x);
        mtxfile_free(&mtxfile);
    }
    return TEST_SUCCESS;
}

/**
 * ‘test_mtxmatrix_dense_to_mtxfile()’ tests converting matrices
 * to Matrix Market files.
 */
int test_mtxmatrix_dense_to_mtxfile(void)
{
    int err;

    {
        int num_rows = 3;
        int num_columns = 3;
        int nnz = 4;
        int rowidx[] = {0, 0, 1, 2};
        int colidx[] = {0, 2, 0, 2};
        float Adata[] = {1.0f, 3.0f, 4.0f, 9.0f};
        struct mtxmatrix A;
        err = mtxmatrix_init_entries_real_single(
            &A, mtxmatrix_dense, mtx_unsymmetric, num_rows, num_columns, nnz, rowidx, colidx, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxfile mtxfile;
        err = mtxmatrix_to_mtxfile(&mtxfile, &A, 0, NULL, 0, NULL, mtxfile_array);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_matrix, mtxfile.header.object);
        TEST_ASSERT_EQ(mtxfile_array, mtxfile.header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtxfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxfile.header.symmetry);
        TEST_ASSERT_EQ(num_rows, mtxfile.size.num_rows);
        TEST_ASSERT_EQ(num_columns, mtxfile.size.num_columns);
        TEST_ASSERT_EQ(-1, mtxfile.size.num_nonzeros);
        TEST_ASSERT_EQ(mtx_single, mtxfile.precision);
        const float * data = mtxfile.data.array_real_single;
        TEST_ASSERT_EQ(1.0f, data[0]);
        TEST_ASSERT_EQ(0.0f, data[1]);
        TEST_ASSERT_EQ(3.0f, data[2]);
        TEST_ASSERT_EQ(4.0f, data[3]);
        TEST_ASSERT_EQ(0.0f, data[4]);
        TEST_ASSERT_EQ(0.0f, data[5]);
        TEST_ASSERT_EQ(0.0f, data[6]);
        TEST_ASSERT_EQ(0.0f, data[7]);
        TEST_ASSERT_EQ(9.0f, data[8]);
        mtxfile_free(&mtxfile);
        mtxmatrix_free(&A);
    }
    {
        int num_rows = 3;
        int num_columns = 3;
        int nnz = 4;
        int rowidx[] = {0, 0, 1, 2};
        int colidx[] = {0, 2, 0, 2};
        float Adata[] = {1.0f, 3.0f, 4.0f, 9.0f};
        struct mtxmatrix A;
        err = mtxmatrix_init_entries_real_single(
            &A, mtxmatrix_dense, mtx_unsymmetric, num_rows, num_columns, nnz, rowidx, colidx, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxfile mtxfile;
        err = mtxmatrix_to_mtxfile(&mtxfile, &A, 0, NULL, 0, NULL, mtxfile_coordinate);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_matrix, mtxfile.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxfile.header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtxfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxfile.header.symmetry);
        TEST_ASSERT_EQ(num_rows, mtxfile.size.num_rows);
        TEST_ASSERT_EQ(num_columns, mtxfile.size.num_columns);
        TEST_ASSERT_EQ(9, mtxfile.size.num_nonzeros);
        TEST_ASSERT_EQ(mtx_single, mtxfile.precision);
        const struct mtxfile_matrix_coordinate_real_single * data =
            mtxfile.data.matrix_coordinate_real_single;
        TEST_ASSERT_EQ(1, data[0].i); TEST_ASSERT_EQ(1, data[0].j); TEST_ASSERT_EQ(1.0f, data[0].a);
        TEST_ASSERT_EQ(1, data[1].i); TEST_ASSERT_EQ(2, data[1].j); TEST_ASSERT_EQ(0.0f, data[1].a);
        TEST_ASSERT_EQ(1, data[2].i); TEST_ASSERT_EQ(3, data[2].j); TEST_ASSERT_EQ(3.0f, data[2].a);
        TEST_ASSERT_EQ(2, data[3].i); TEST_ASSERT_EQ(1, data[3].j); TEST_ASSERT_EQ(4.0f, data[3].a);
        TEST_ASSERT_EQ(2, data[4].i); TEST_ASSERT_EQ(2, data[4].j); TEST_ASSERT_EQ(0.0f, data[4].a);
        TEST_ASSERT_EQ(2, data[5].i); TEST_ASSERT_EQ(3, data[5].j); TEST_ASSERT_EQ(0.0f, data[5].a);
        TEST_ASSERT_EQ(3, data[6].i); TEST_ASSERT_EQ(1, data[6].j); TEST_ASSERT_EQ(0.0f, data[6].a);
        TEST_ASSERT_EQ(3, data[7].i); TEST_ASSERT_EQ(2, data[7].j); TEST_ASSERT_EQ(0.0f, data[7].a);
        TEST_ASSERT_EQ(3, data[8].i); TEST_ASSERT_EQ(3, data[8].j); TEST_ASSERT_EQ(9.0f, data[8].a);
        mtxfile_free(&mtxfile);
        mtxmatrix_free(&A);
    }
    {
        int num_rows = 3;
        int num_columns = 3;
        int nnz = 4;
        int rowidx[] = {0, 0, 1, 2};
        int colidx[] = {0, 2, 2, 2};
        float Adata[] = {1.0f, 3.0f, 6.0f, 9.0f};
        struct mtxmatrix A;
        err = mtxmatrix_init_entries_real_single(
            &A, mtxmatrix_dense, mtx_symmetric, num_rows, num_columns, nnz, rowidx, colidx, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxfile mtxfile;
        err = mtxmatrix_to_mtxfile(&mtxfile, &A, 0, NULL, 0, NULL, mtxfile_coordinate);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_matrix, mtxfile.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxfile.header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtxfile.header.field);
        TEST_ASSERT_EQ(mtxfile_symmetric, mtxfile.header.symmetry);
        TEST_ASSERT_EQ(num_rows, mtxfile.size.num_rows);
        TEST_ASSERT_EQ(num_columns, mtxfile.size.num_columns);
        TEST_ASSERT_EQ(6, mtxfile.size.num_nonzeros);
        TEST_ASSERT_EQ(mtx_single, mtxfile.precision);
        const struct mtxfile_matrix_coordinate_real_single * data =
            mtxfile.data.matrix_coordinate_real_single;
        TEST_ASSERT_EQ(1, data[0].i); TEST_ASSERT_EQ(1, data[0].j); TEST_ASSERT_EQ(1.0f, data[0].a);
        TEST_ASSERT_EQ(1, data[1].i); TEST_ASSERT_EQ(2, data[1].j); TEST_ASSERT_EQ(0.0f, data[1].a);
        TEST_ASSERT_EQ(1, data[2].i); TEST_ASSERT_EQ(3, data[2].j); TEST_ASSERT_EQ(3.0f, data[2].a);
        TEST_ASSERT_EQ(2, data[3].i); TEST_ASSERT_EQ(2, data[3].j); TEST_ASSERT_EQ(0.0f, data[3].a);
        TEST_ASSERT_EQ(2, data[4].i); TEST_ASSERT_EQ(3, data[4].j); TEST_ASSERT_EQ(6.0f, data[4].a);
        TEST_ASSERT_EQ(3, data[5].i); TEST_ASSERT_EQ(3, data[5].j); TEST_ASSERT_EQ(9.0f, data[5].a);
        mtxfile_free(&mtxfile);
        mtxmatrix_free(&A);
    }
    {
        int num_rows = 3;
        int num_columns = 3;
        int nnz = 3;
        int rowidx[] = {0, 0, 1};
        int colidx[] = {1, 2, 2};
        float Adata[] = {2.0f, 3.0f, 6.0f};
        struct mtxmatrix A;
        err = mtxmatrix_init_entries_real_single(
            &A, mtxmatrix_dense, mtx_skew_symmetric, num_rows, num_columns, nnz, rowidx, colidx, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxfile mtxfile;
        err = mtxmatrix_to_mtxfile(&mtxfile, &A, 0, NULL, 0, NULL, mtxfile_coordinate);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_matrix, mtxfile.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxfile.header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtxfile.header.field);
        TEST_ASSERT_EQ(mtxfile_skew_symmetric, mtxfile.header.symmetry);
        TEST_ASSERT_EQ(num_rows, mtxfile.size.num_rows);
        TEST_ASSERT_EQ(num_columns, mtxfile.size.num_columns);
        TEST_ASSERT_EQ(3, mtxfile.size.num_nonzeros);
        TEST_ASSERT_EQ(mtx_single, mtxfile.precision);
        const struct mtxfile_matrix_coordinate_real_single * data =
            mtxfile.data.matrix_coordinate_real_single;
        TEST_ASSERT_EQ(1, data[0].i); TEST_ASSERT_EQ(2, data[0].j); TEST_ASSERT_EQ(2.0f, data[0].a);
        TEST_ASSERT_EQ(1, data[1].i); TEST_ASSERT_EQ(3, data[1].j); TEST_ASSERT_EQ(3.0f, data[1].a);
        TEST_ASSERT_EQ(2, data[2].i); TEST_ASSERT_EQ(3, data[2].j); TEST_ASSERT_EQ(6.0f, data[2].a);
        mtxfile_free(&mtxfile);
        mtxmatrix_free(&A);
    }
    {
        int num_rows = 3;
        int num_columns = 3;
        int nnz = 4;
        int rowidx[] = {0, 0, 1, 2};
        int colidx[] = {0, 2, 0, 2};
        double Adata[] = {1.0, 3.0, 4.0, 9.0};
        struct mtxmatrix A;
        err = mtxmatrix_init_entries_real_double(
            &A, mtxmatrix_dense, mtx_unsymmetric, num_rows, num_columns, nnz, rowidx, colidx, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxfile mtxfile;
        err = mtxmatrix_to_mtxfile(&mtxfile, &A, 0, NULL, 0, NULL, mtxfile_coordinate);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_matrix, mtxfile.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxfile.header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtxfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxfile.header.symmetry);
        TEST_ASSERT_EQ(num_rows, mtxfile.size.num_rows);
        TEST_ASSERT_EQ(num_columns, mtxfile.size.num_columns);
        TEST_ASSERT_EQ(9, mtxfile.size.num_nonzeros);
        TEST_ASSERT_EQ(mtx_double, mtxfile.precision);
        const struct mtxfile_matrix_coordinate_real_double * data =
            mtxfile.data.matrix_coordinate_real_double;
        TEST_ASSERT_EQ(1, data[0].i); TEST_ASSERT_EQ(1, data[0].j); TEST_ASSERT_EQ(1.0, data[0].a);
        TEST_ASSERT_EQ(1, data[1].i); TEST_ASSERT_EQ(2, data[1].j); TEST_ASSERT_EQ(0.0, data[1].a);
        TEST_ASSERT_EQ(1, data[2].i); TEST_ASSERT_EQ(3, data[2].j); TEST_ASSERT_EQ(3.0, data[2].a);
        TEST_ASSERT_EQ(2, data[3].i); TEST_ASSERT_EQ(1, data[3].j); TEST_ASSERT_EQ(4.0, data[3].a);
        TEST_ASSERT_EQ(2, data[4].i); TEST_ASSERT_EQ(2, data[4].j); TEST_ASSERT_EQ(0.0, data[4].a);
        TEST_ASSERT_EQ(2, data[5].i); TEST_ASSERT_EQ(3, data[5].j); TEST_ASSERT_EQ(0.0, data[5].a);
        TEST_ASSERT_EQ(3, data[6].i); TEST_ASSERT_EQ(1, data[6].j); TEST_ASSERT_EQ(0.0, data[6].a);
        TEST_ASSERT_EQ(3, data[7].i); TEST_ASSERT_EQ(2, data[7].j); TEST_ASSERT_EQ(0.0, data[7].a);
        TEST_ASSERT_EQ(3, data[8].i); TEST_ASSERT_EQ(3, data[8].j); TEST_ASSERT_EQ(9.0, data[8].a);
        mtxfile_free(&mtxfile);
        mtxmatrix_free(&A);
    }
    {
        int num_rows = 3;
        int num_columns = 3;
        int nnz = 4;
        int rowidx[] = {0, 0, 1, 2};
        int colidx[] = {0, 2, 0, 2};
        float Adata[][2] = {{1.0f,-1.0f}, {3.0f,-3.0f}, {4.0f,-4.0f}, {9.0f,-9.0f}};
        struct mtxmatrix A;
        err = mtxmatrix_init_entries_complex_single(
            &A, mtxmatrix_dense, mtx_unsymmetric, num_rows, num_columns, nnz, rowidx, colidx, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxfile mtxfile;
        err = mtxmatrix_to_mtxfile(&mtxfile, &A, 0, NULL, 0, NULL, mtxfile_coordinate);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_matrix, mtxfile.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxfile.header.format);
        TEST_ASSERT_EQ(mtxfile_complex, mtxfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxfile.header.symmetry);
        TEST_ASSERT_EQ(num_rows, mtxfile.size.num_rows);
        TEST_ASSERT_EQ(num_columns, mtxfile.size.num_columns);
        TEST_ASSERT_EQ(9, mtxfile.size.num_nonzeros);
        TEST_ASSERT_EQ(mtx_single, mtxfile.precision);
        const struct mtxfile_matrix_coordinate_complex_single * data =
            mtxfile.data.matrix_coordinate_complex_single;
        TEST_ASSERT_EQ(1, data[0].i); TEST_ASSERT_EQ(1, data[0].j); TEST_ASSERT_EQ(1.0f, data[0].a[0]); TEST_ASSERT_EQ(-1.0f, data[0].a[1]);
        TEST_ASSERT_EQ(1, data[1].i); TEST_ASSERT_EQ(2, data[1].j); TEST_ASSERT_EQ(0.0f, data[1].a[0]); TEST_ASSERT_EQ(0.0f, data[1].a[1]);
        TEST_ASSERT_EQ(1, data[2].i); TEST_ASSERT_EQ(3, data[2].j); TEST_ASSERT_EQ(3.0f, data[2].a[0]); TEST_ASSERT_EQ(-3.0f, data[2].a[1]);
        TEST_ASSERT_EQ(2, data[3].i); TEST_ASSERT_EQ(1, data[3].j); TEST_ASSERT_EQ(4.0f, data[3].a[0]); TEST_ASSERT_EQ(-4.0f, data[3].a[1]);
        TEST_ASSERT_EQ(2, data[4].i); TEST_ASSERT_EQ(2, data[4].j); TEST_ASSERT_EQ(0.0f, data[4].a[0]); TEST_ASSERT_EQ(0.0f, data[4].a[1]);
        TEST_ASSERT_EQ(2, data[5].i); TEST_ASSERT_EQ(3, data[5].j); TEST_ASSERT_EQ(0.0f, data[5].a[0]); TEST_ASSERT_EQ(0.0f, data[5].a[1]);
        TEST_ASSERT_EQ(3, data[6].i); TEST_ASSERT_EQ(1, data[6].j); TEST_ASSERT_EQ(0.0f, data[6].a[0]); TEST_ASSERT_EQ(0.0f, data[6].a[1]);
        TEST_ASSERT_EQ(3, data[7].i); TEST_ASSERT_EQ(2, data[7].j); TEST_ASSERT_EQ(0.0f, data[7].a[0]); TEST_ASSERT_EQ(0.0f, data[7].a[1]);
        TEST_ASSERT_EQ(3, data[8].i); TEST_ASSERT_EQ(3, data[8].j); TEST_ASSERT_EQ(9.0f, data[8].a[0]); TEST_ASSERT_EQ(-9.0f, data[8].a[1]);
        mtxfile_free(&mtxfile);
        mtxmatrix_free(&A);
    }
    {
        int num_rows = 3;
        int num_columns = 3;
        int nnz = 4;
        int rowidx[] = {0, 0, 1, 2};
        int colidx[] = {1, 2, 2, 2};
        float Adata[][2] = {{2.0f,-2.0f}, {3.0f,-3.0f}, {6.0f,-6.0f}, {9.0f,-9.0f}};
        struct mtxmatrix A;
        err = mtxmatrix_init_entries_complex_single(
            &A, mtxmatrix_dense, mtx_hermitian, num_rows, num_columns, nnz, rowidx, colidx, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxfile mtxfile;
        err = mtxmatrix_to_mtxfile(&mtxfile, &A, 0, NULL, 0, NULL, mtxfile_coordinate);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_matrix, mtxfile.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxfile.header.format);
        TEST_ASSERT_EQ(mtxfile_complex, mtxfile.header.field);
        TEST_ASSERT_EQ(mtxfile_hermitian, mtxfile.header.symmetry);
        TEST_ASSERT_EQ(num_rows, mtxfile.size.num_rows);
        TEST_ASSERT_EQ(num_columns, mtxfile.size.num_columns);
        TEST_ASSERT_EQ(6, mtxfile.size.num_nonzeros);
        TEST_ASSERT_EQ(mtx_single, mtxfile.precision);
        const struct mtxfile_matrix_coordinate_complex_single * data =
            mtxfile.data.matrix_coordinate_complex_single;
        TEST_ASSERT_EQ(1, data[0].i); TEST_ASSERT_EQ(1, data[0].j); TEST_ASSERT_EQ(0.0f, data[0].a[0]); TEST_ASSERT_EQ(0.0f, data[0].a[1]);
        TEST_ASSERT_EQ(1, data[1].i); TEST_ASSERT_EQ(2, data[1].j); TEST_ASSERT_EQ(2.0f, data[1].a[0]); TEST_ASSERT_EQ(-2.0f, data[1].a[1]);
        TEST_ASSERT_EQ(1, data[2].i); TEST_ASSERT_EQ(3, data[2].j); TEST_ASSERT_EQ(3.0f, data[2].a[0]); TEST_ASSERT_EQ(-3.0f, data[2].a[1]);
        TEST_ASSERT_EQ(2, data[3].i); TEST_ASSERT_EQ(2, data[3].j); TEST_ASSERT_EQ(0.0f, data[3].a[0]); TEST_ASSERT_EQ(0.0f, data[3].a[1]);
        TEST_ASSERT_EQ(2, data[4].i); TEST_ASSERT_EQ(3, data[4].j); TEST_ASSERT_EQ(6.0f, data[4].a[0]); TEST_ASSERT_EQ(-6.0f, data[4].a[1]);
        TEST_ASSERT_EQ(3, data[5].i); TEST_ASSERT_EQ(3, data[5].j); TEST_ASSERT_EQ(9.0f, data[5].a[0]); TEST_ASSERT_EQ(-9.0f, data[5].a[1]);
        mtxfile_free(&mtxfile);
        mtxmatrix_free(&A);
    }
    {
        int num_rows = 3;
        int num_columns = 3;
        int nnz = 4;
        int rowidx[] = {0, 0, 1, 2};
        int colidx[] = {0, 2, 0, 2};
        int32_t Adata[] = {1, 3, 4, 9};
        struct mtxmatrix A;
        err = mtxmatrix_init_entries_integer_single(
            &A, mtxmatrix_dense, mtx_unsymmetric, num_rows, num_columns, nnz, rowidx, colidx, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxfile mtxfile;
        err = mtxmatrix_to_mtxfile(&mtxfile, &A, 0, NULL, 0, NULL, mtxfile_coordinate);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_matrix, mtxfile.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxfile.header.format);
        TEST_ASSERT_EQ(mtxfile_integer, mtxfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxfile.header.symmetry);
        TEST_ASSERT_EQ(num_rows, mtxfile.size.num_rows);
        TEST_ASSERT_EQ(num_columns, mtxfile.size.num_columns);
        TEST_ASSERT_EQ(9, mtxfile.size.num_nonzeros);
        TEST_ASSERT_EQ(mtx_single, mtxfile.precision);
        const struct mtxfile_matrix_coordinate_integer_single * data =
            mtxfile.data.matrix_coordinate_integer_single;
        TEST_ASSERT_EQ(1, data[0].i); TEST_ASSERT_EQ(1, data[0].j); TEST_ASSERT_EQ(1, data[0].a);
        TEST_ASSERT_EQ(1, data[1].i); TEST_ASSERT_EQ(2, data[1].j); TEST_ASSERT_EQ(0, data[1].a);
        TEST_ASSERT_EQ(1, data[2].i); TEST_ASSERT_EQ(3, data[2].j); TEST_ASSERT_EQ(3, data[2].a);
        TEST_ASSERT_EQ(2, data[3].i); TEST_ASSERT_EQ(1, data[3].j); TEST_ASSERT_EQ(4, data[3].a);
        TEST_ASSERT_EQ(2, data[4].i); TEST_ASSERT_EQ(2, data[4].j); TEST_ASSERT_EQ(0, data[4].a);
        TEST_ASSERT_EQ(2, data[5].i); TEST_ASSERT_EQ(3, data[5].j); TEST_ASSERT_EQ(0, data[5].a);
        TEST_ASSERT_EQ(3, data[6].i); TEST_ASSERT_EQ(1, data[6].j); TEST_ASSERT_EQ(0, data[6].a);
        TEST_ASSERT_EQ(3, data[7].i); TEST_ASSERT_EQ(2, data[7].j); TEST_ASSERT_EQ(0, data[7].a);
        TEST_ASSERT_EQ(3, data[8].i); TEST_ASSERT_EQ(3, data[8].j); TEST_ASSERT_EQ(9, data[8].a);
        mtxfile_free(&mtxfile);
        mtxmatrix_free(&A);
    }
    {
        int num_rows = 3;
        int num_columns = 3;
        int nnz = 4;
        int rowidx[] = {0, 0, 1, 2};
        int colidx[] = {0, 2, 0, 2};
        int64_t Adata[] = {1, 3, 4, 9};
        struct mtxmatrix A;
        err = mtxmatrix_init_entries_integer_double(
            &A, mtxmatrix_dense, mtx_unsymmetric, num_rows, num_columns, nnz, rowidx, colidx, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxfile mtxfile;
        err = mtxmatrix_to_mtxfile(&mtxfile, &A, 0, NULL, 0, NULL, mtxfile_coordinate);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_matrix, mtxfile.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxfile.header.format);
        TEST_ASSERT_EQ(mtxfile_integer, mtxfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxfile.header.symmetry);
        TEST_ASSERT_EQ(num_rows, mtxfile.size.num_rows);
        TEST_ASSERT_EQ(num_columns, mtxfile.size.num_columns);
        TEST_ASSERT_EQ(9, mtxfile.size.num_nonzeros);
        TEST_ASSERT_EQ(mtx_double, mtxfile.precision);
        const struct mtxfile_matrix_coordinate_integer_double * data =
            mtxfile.data.matrix_coordinate_integer_double;
        TEST_ASSERT_EQ(1, data[0].i); TEST_ASSERT_EQ(1, data[0].j); TEST_ASSERT_EQ(1, data[0].a);
        TEST_ASSERT_EQ(1, data[1].i); TEST_ASSERT_EQ(2, data[1].j); TEST_ASSERT_EQ(0, data[1].a);
        TEST_ASSERT_EQ(1, data[2].i); TEST_ASSERT_EQ(3, data[2].j); TEST_ASSERT_EQ(3, data[2].a);
        TEST_ASSERT_EQ(2, data[3].i); TEST_ASSERT_EQ(1, data[3].j); TEST_ASSERT_EQ(4, data[3].a);
        TEST_ASSERT_EQ(2, data[4].i); TEST_ASSERT_EQ(2, data[4].j); TEST_ASSERT_EQ(0, data[4].a);
        TEST_ASSERT_EQ(2, data[5].i); TEST_ASSERT_EQ(3, data[5].j); TEST_ASSERT_EQ(0, data[5].a);
        TEST_ASSERT_EQ(3, data[6].i); TEST_ASSERT_EQ(1, data[6].j); TEST_ASSERT_EQ(0, data[6].a);
        TEST_ASSERT_EQ(3, data[7].i); TEST_ASSERT_EQ(2, data[7].j); TEST_ASSERT_EQ(0, data[7].a);
        TEST_ASSERT_EQ(3, data[8].i); TEST_ASSERT_EQ(3, data[8].j); TEST_ASSERT_EQ(9, data[8].a);
        mtxfile_free(&mtxfile);
        mtxmatrix_free(&A);
    }
    return TEST_SUCCESS;
}

/**
 * ‘test_mtxmatrix_dense_gemv()’ tests computing matrix-vector
 * products for matrices in coordinate format.
 */
int test_mtxmatrix_dense_gemv(void)
{
    int err;

    /*
     * a) For unsymmetric real or integer matrices, calculate
     *
     *   ⎡ 1 2 0⎤  ⎡ 3⎤     ⎡ 1⎤  ⎡ 14⎤  ⎡ 3⎤  ⎡ 17⎤
     * 2*⎢ 4 5 6⎥ *⎢ 2⎥ + 3*⎢ 0⎥ =⎢ 56⎥ +⎢ 0⎥ =⎢ 56⎥,
     *   ⎣ 7 8 9⎦  ⎣ 1⎦     ⎣ 2⎦  ⎣ 92⎦  ⎣ 6⎦  ⎣ 98⎦
     *
     * and the transposed product
     *
     *   ⎡ 1 4 7⎤  ⎡ 3⎤     ⎡ 1⎤  ⎡ 36⎤  ⎡ 3⎤  ⎡ 39⎤
     * 2*⎢ 2 5 8⎥ *⎢ 2⎥ + 3*⎢ 0⎥ =⎢ 48⎥ +⎢ 0⎥ =⎢ 48⎥.
     *   ⎣ 0 6 9⎦  ⎣ 1⎦     ⎣ 2⎦  ⎣ 42⎦  ⎣ 6⎦  ⎣ 48⎦
     *
     * b) For symmetric real or integer matrices, calculate
     *
     *   ⎡ 1 2 0⎤  ⎡ 3⎤     ⎡ 1⎤  ⎡ 14⎤  ⎡ 3⎤  ⎡ 17⎤
     * 2*⎢ 2 5 6⎥ *⎢ 2⎥ + 3*⎢ 0⎥ =⎢ 44⎥ +⎢ 0⎥ =⎢ 44⎥,
     *   ⎣ 0 6 9⎦  ⎣ 1⎦     ⎣ 2⎦  ⎣ 42⎦  ⎣ 6⎦  ⎣ 48⎦
     *
     * c) For unsymmetric complex matrices, calculate
     *
     *   ⎡ 1+2i 3+4i⎤  ⎡ 3+1i⎤     ⎡ 1+0i⎤  ⎡-8+34i⎤  ⎡ 3   ⎤  ⎡-5+34i⎤
     * 2*⎢          ⎥ *⎢     ⎥ + 3*⎢     ⎥ =⎢      ⎥ +⎢     ⎥ =⎢      ⎥,
     *   ⎣ 5+6i 7+8i⎦  ⎣ 1+2i⎦     ⎣ 2+2i⎦  ⎣ 0+90i⎦  ⎣ 6+6i⎦  ⎣ 6+96i⎦
     *
     * and the transposed product
     *
     *   ⎡ 1+2i 5+6i⎤  ⎡ 3+1i⎤     ⎡ 1+0i⎤  ⎡-12+46i⎤  ⎡ 3   ⎤  ⎡-9+46i⎤
     * 2*⎢          ⎥ *⎢     ⎥ + 3*⎢     ⎥ =⎢       ⎥ +⎢     ⎥ =⎢      ⎥,
     *   ⎣ 3+4i 7+8i⎦  ⎣ 1+2i⎦     ⎣ 2+2i⎦  ⎣ -8+74i⎦  ⎣ 6+6i⎦  ⎣-2+80i⎦
     *
     * and the conjugate transposed product
     *
     *   ⎡ 1-2i 5-6i⎤  ⎡ 3+1i⎤     ⎡ 1+0i⎤  ⎡ 44-2i⎤  ⎡ 3   ⎤  ⎡ 47-2i⎤
     * 2*⎢          ⎥ *⎢     ⎥ + 3*⎢     ⎥ =⎢      ⎥ +⎢     ⎥ =⎢      ⎥,
     *   ⎣ 3-4i 7-8i⎦  ⎣ 1+2i⎦     ⎣ 2+2i⎦  ⎣ 72-6i⎦  ⎣ 6+6i⎦  ⎣ 78   ⎦
     *
     * and the product with complex coefficients (cgemv/zgemv)
     *
     *    ⎡ 1+2i 3+4i⎤  ⎡ 3+1i⎤          ⎡ 1+0i⎤  ⎡-34-8i⎤  ⎡ 3+1i⎤  ⎡-31-7i⎤
     * 2i*⎢          ⎥ *⎢     ⎥ + (3+1i)*⎢     ⎥ =⎢      ⎥ +⎢     ⎥ =⎢      ⎥,
     *    ⎣ 5+6i 7+8i⎦  ⎣ 1+2i⎦          ⎣ 2+2i⎦  ⎣-90   ⎦  ⎣ 4+8i⎦  ⎣-86+8i⎦
     *
     * and the transposed product with complex coefficients (cgemv/zgemv)
     *
     *    ⎡ 1+2i 5+6i⎤  ⎡ 3+1i⎤          ⎡ 1+0i⎤  ⎡-46-12i⎤  ⎡ 3+1i⎤  ⎡-43-11i⎤
     * 2i*⎢          ⎥ *⎢     ⎥ + (3+1i)*⎢     ⎥ =⎢       ⎥ +⎢     ⎥ =⎢       ⎥,
     *    ⎣ 3+4i 7+8i⎦  ⎣ 1+2i⎦          ⎣ 2+2i⎦  ⎣-74- 8i⎦  ⎣ 4+8i⎦  ⎣-70    ⎦
     *
     * and the conjugate transposed product with complex coefficients (cgemv/zgemv)
     *
     *    ⎡ 1-2i 5-6i⎤  ⎡ 3+1i⎤          ⎡ 1+0i⎤  ⎡ 2+44i⎤  ⎡ 3+1i⎤  ⎡  5+45i⎤
     * 2i*⎢          ⎥ *⎢     ⎥ + (3+1i)*⎢     ⎥ =⎢      ⎥ +⎢     ⎥ =⎢       ⎥.
     *    ⎣ 3-4i 7-8i⎦  ⎣ 1+2i⎦          ⎣ 2+2i⎦  ⎣ 6+72i⎦  ⎣ 4+8i⎦  ⎣ 10+80i⎦
     *
     * d) For symmetric complex matrices, calculate
     *
     *   ⎡ 1+2i 3+4i⎤  ⎡ 3+1i⎤     ⎡ 1+0i⎤  ⎡-8+34i⎤  ⎡ 3   ⎤  ⎡-5+34i⎤
     * 2*⎢          ⎥ *⎢     ⎥ + 3*⎢     ⎥ =⎢      ⎥ +⎢     ⎥ =⎢      ⎥.
     *   ⎣ 3+4i 7+8i⎦  ⎣ 1+2i⎦     ⎣ 2+2i⎦  ⎣-8+74i⎦  ⎣ 6+6i⎦  ⎣-2+80i⎦
     *
     * and the conjugate transposed product
     *
     *   ⎡ 1-2i 3-4i⎤  ⎡ 3+1i⎤     ⎡ 1+0i⎤  ⎡ 32-6i⎤  ⎡ 3   ⎤  ⎡ 35-6i⎤
     * 2*⎢          ⎥ *⎢     ⎥ + 3*⎢     ⎥ =⎢      ⎥ +⎢     ⎥ =⎢      ⎥.
     *   ⎣ 3-4i 7-8i⎦  ⎣ 1+2i⎦     ⎣ 2+2i⎦  ⎣ 72-6i⎦  ⎣ 6+6i⎦  ⎣ 78+0i⎦
     *
     * and the product with complex coefficients (cgemv/zgemv)
     *
     *    ⎡ 1+2i 3+4i⎤  ⎡ 3+1i⎤          ⎡ 1+0i⎤  ⎡-34-8i⎤  ⎡ 3+1i⎤  ⎡-31-7i⎤
     * 2i*⎢          ⎥ *⎢     ⎥ + (3+1i)*⎢     ⎥ =⎢      ⎥ +⎢     ⎥ =⎢      ⎥.
     *    ⎣ 3+4i 7+8i⎦  ⎣ 1+2i⎦          ⎣ 2+2i⎦  ⎣-74-8i⎦  ⎣ 4+8i⎦  ⎣-70+0i⎦
     *
     * and the conjugate transposed product with complex coefficients (cgemv/zgemv)
     *
     *    ⎡ 1-2i 3-4i⎤  ⎡ 3+1i⎤          ⎡ 1+0i⎤  ⎡ 6+32i⎤  ⎡ 3+1i⎤  ⎡ 9+33i⎤
     * 2i*⎢          ⎥ *⎢     ⎥ + (3+1i)*⎢     ⎥ =⎢      ⎥ +⎢     ⎥ =⎢      ⎥.
     *    ⎣ 3-4i 7-8i⎦  ⎣ 1+2i⎦          ⎣ 2+2i⎦  ⎣ 6+72i⎦  ⎣ 4+8i⎦  ⎣10+80i⎦
     *
     * e) for Hermitian complex matrices, calculate
     *
     *   ⎡ 1+0i 3+4i⎤  ⎡ 3+1i⎤     ⎡ 1+0i⎤  ⎡-4+22i⎤  ⎡ 3   ⎤  ⎡-1+22i⎤
     * 2*⎢          ⎥ *⎢     ⎥ + 3*⎢     ⎥ =⎢      ⎥ +⎢     ⎥ =⎢      ⎥.
     *   ⎣ 3-4i 7+0i⎦  ⎣ 1+2i⎦     ⎣ 2+2i⎦  ⎣40+10i⎦  ⎣ 6+6i⎦  ⎣46+16i⎦
     *
     * and the transposed product
     *
     *   ⎡ 1+0i 3-4i⎤  ⎡ 3+1i⎤     ⎡ 1+0i⎤  ⎡  28+6i⎤  ⎡ 3   ⎤  ⎡  31+6i⎤
     * 2*⎢          ⎥ *⎢     ⎥ + 3*⎢     ⎥ =⎢       ⎥ +⎢     ⎥ =⎢       ⎥.
     *   ⎣ 3+4i 7+0i⎦  ⎣ 1+2i⎦     ⎣ 2+2i⎦  ⎣ 24+58i⎦  ⎣ 6+6i⎦  ⎣ 30+64i⎦
     *
     * and the product with complex coefficients (cgemv/zgemv)
     *
     *    ⎡ 1+0i 3+4i⎤  ⎡ 3+1i⎤          ⎡ 1+0i⎤  ⎡ -22-4i⎤  ⎡ 3+1i⎤  ⎡-19-3i⎤
     * 2i*⎢          ⎥ *⎢     ⎥ + (3+1i)*⎢     ⎥ =⎢       ⎥ +⎢     ⎥ =⎢      ⎥.
     *    ⎣ 3-4i 7+0i⎦  ⎣ 1+2i⎦          ⎣ 2+2i⎦  ⎣-10+40i⎦  ⎣ 4+8i⎦  ⎣-6+48i⎦
     *
     * and the transposed product with complex coefficients (cgemv/zgemv)
     *
     *    ⎡ 1+0i 3-4i⎤  ⎡ 3+1i⎤          ⎡ 1+0i⎤  ⎡ -6+28i⎤  ⎡ 3+1i⎤  ⎡ -3+29i⎤
     * 2i*⎢          ⎥ *⎢     ⎥ + (3+1i)*⎢     ⎥ =⎢       ⎥ +⎢     ⎥ =⎢       ⎥.
     *    ⎣ 3+4i 7+0i⎦  ⎣ 1+2i⎦          ⎣ 2+2i⎦  ⎣-58+24i⎦  ⎣ 4+8i⎦  ⎣-54+32i⎦
     *
     */

    /*
     * Real, single precision, unsymmetric matrices
     */

    {
        struct mtxmatrix A;
        struct mtxvector x;
        struct mtxvector y;
        int num_rows = 3;
        int num_columns = 3;
        int num_nonzeros = 8;
        int Arowidx[] = {0, 0, 1, 1, 1, 2, 2, 2};
        int Acolidx[] = {0, 1, 0, 1, 2, 0, 1, 2};
        float Adata[] = {1.0f, 2.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
        float xdata[] = {3.0f, 2.0f, 1.0f};
        float ydata[] = {1.0f, 0.0f, 2.0f};
        err = mtxmatrix_init_entries_real_single(
            &A, mtxmatrix_dense, mtx_unsymmetric, num_rows, num_columns, num_nonzeros, Arowidx, Acolidx, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_init_real_single(&x, mtxvector_base, num_columns, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        {
            err = mtxvector_init_real_single(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemv(mtx_notrans, 2.0f, &A, &x, 3.0f, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
            TEST_ASSERT_EQ(mtx_field_real, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(3, y_->size);
            TEST_ASSERT_EQ(y_->data.real_single[0], 17.0f);
            TEST_ASSERT_EQ(y_->data.real_single[1], 56.0f);
            TEST_ASSERT_EQ(y_->data.real_single[2], 98.0f);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_real_single(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemv(mtx_trans, 2.0f, &A, &x, 3.0f, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
            TEST_ASSERT_EQ(mtx_field_real, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(3, y_->size);
            TEST_ASSERT_EQ(y_->data.real_single[0], 39.0f);
            TEST_ASSERT_EQ(y_->data.real_single[1], 48.0f);
            TEST_ASSERT_EQ(y_->data.real_single[2], 48.0f);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_real_single(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dgemv(mtx_notrans, 2.0, &A, &x, 3.0, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
            TEST_ASSERT_EQ(mtx_field_real, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(3, y_->size);
            TEST_ASSERT_EQ(y_->data.real_single[0], 17.0f);
            TEST_ASSERT_EQ(y_->data.real_single[1], 56.0f);
            TEST_ASSERT_EQ(y_->data.real_single[2], 98.0f);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_real_single(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dgemv(mtx_trans, 2.0, &A, &x, 3.0, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
            TEST_ASSERT_EQ(mtx_field_real, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(3, y_->size);
            TEST_ASSERT_EQ(y_->data.real_single[0], 39.0f);
            TEST_ASSERT_EQ(y_->data.real_single[1], 48.0f);
            TEST_ASSERT_EQ(y_->data.real_single[2], 48.0f);
            mtxvector_free(&y);
        }
        mtxvector_free(&x);
        mtxmatrix_free(&A);
    }

    /*
     * Real, single precision, symmetric matrices
     */

    {
        struct mtxmatrix A;
        struct mtxvector x;
        struct mtxvector y;
        int num_rows = 3;
        int num_columns = 3;
        int num_nonzeros = 5;
        int Arowidx[] = {0, 0, 1, 1, 2};
        int Acolidx[] = {0, 1, 1, 2, 2};
        float Adata[] = {1.0f, 2.0f, 5.0f, 6.0f, 9.0f};
        float xdata[] = {3.0f, 2.0f, 1.0f};
        float ydata[] = {1.0f, 0.0f, 2.0f};
        err = mtxmatrix_init_entries_real_single(
            &A, mtxmatrix_dense, mtx_symmetric, num_rows, num_columns, num_nonzeros, Arowidx, Acolidx, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_init_real_single(&x, mtxvector_base, num_columns, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        {
            err = mtxvector_init_real_single(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemv(mtx_notrans, 2.0f, &A, &x, 3.0f, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
            TEST_ASSERT_EQ(mtx_field_real, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(3, y_->size);
            TEST_ASSERT_EQ(y_->data.real_single[0], 17.0f);
            TEST_ASSERT_EQ(y_->data.real_single[1], 44.0f);
            TEST_ASSERT_EQ(y_->data.real_single[2], 48.0f);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_real_single(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemv(mtx_trans, 2.0f, &A, &x, 3.0f, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
            TEST_ASSERT_EQ(mtx_field_real, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(3, y_->size);
            TEST_ASSERT_EQ(y_->data.real_single[0], 17.0f);
            TEST_ASSERT_EQ(y_->data.real_single[1], 44.0f);
            TEST_ASSERT_EQ(y_->data.real_single[2], 48.0f);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_real_single(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dgemv(mtx_notrans, 2.0, &A, &x, 3.0, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
            TEST_ASSERT_EQ(mtx_field_real, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(3, y_->size);
            TEST_ASSERT_EQ(y_->data.real_single[0], 17.0f);
            TEST_ASSERT_EQ(y_->data.real_single[1], 44.0f);
            TEST_ASSERT_EQ(y_->data.real_single[2], 48.0f);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_real_single(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dgemv(mtx_trans, 2.0, &A, &x, 3.0, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
            TEST_ASSERT_EQ(mtx_field_real, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(3, y_->size);
            TEST_ASSERT_EQ(y_->data.real_single[0], 17.0f);
            TEST_ASSERT_EQ(y_->data.real_single[1], 44.0f);
            TEST_ASSERT_EQ(y_->data.real_single[2], 48.0f);
            mtxvector_free(&y);
        }
        mtxvector_free(&x);
        mtxmatrix_free(&A);
    }

    /*
     * Real, double precision, unsymmetric matrices
     */

    {
        struct mtxmatrix A;
        struct mtxvector x;
        struct mtxvector y;
        int num_rows = 3;
        int num_columns = 3;
        int num_nonzeros = 8;
        int Arowidx[] = {0, 0, 1, 1, 1, 2, 2, 2};
        int Acolidx[] = {0, 1, 0, 1, 2, 0, 1, 2};
        double Adata[] = {1.0, 2.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
        double xdata[] = {3.0, 2.0, 1.0};
        double ydata[] = {1.0, 0.0, 2.0};
        err = mtxmatrix_init_entries_real_double(
            &A, mtxmatrix_dense, mtx_unsymmetric, num_rows, num_columns, num_nonzeros, Arowidx, Acolidx, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_init_real_double(&x, mtxvector_base, num_columns, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        {
            err = mtxvector_init_real_double(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemv(mtx_notrans, 2.0f, &A, &x, 3.0f, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
            TEST_ASSERT_EQ(mtx_field_real, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(3, y_->size);
            TEST_ASSERT_EQ(y_->data.real_double[0], 17.0);
            TEST_ASSERT_EQ(y_->data.real_double[1], 56.0);
            TEST_ASSERT_EQ(y_->data.real_double[2], 98.0);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_real_double(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemv(mtx_trans, 2.0f, &A, &x, 3.0f, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
            TEST_ASSERT_EQ(mtx_field_real, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(3, y_->size);
            TEST_ASSERT_EQ(y_->data.real_double[0], 39.0);
            TEST_ASSERT_EQ(y_->data.real_double[1], 48.0);
            TEST_ASSERT_EQ(y_->data.real_double[2], 48.0);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_real_double(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dgemv(mtx_notrans, 2.0, &A, &x, 3.0, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
            TEST_ASSERT_EQ(mtx_field_real, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(3, y_->size);
            TEST_ASSERT_EQ(y_->data.real_double[0], 17.0);
            TEST_ASSERT_EQ(y_->data.real_double[1], 56.0);
            TEST_ASSERT_EQ(y_->data.real_double[2], 98.0);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_real_double(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dgemv(mtx_trans, 2.0, &A, &x, 3.0, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
            TEST_ASSERT_EQ(mtx_field_real, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(3, y_->size);
            TEST_ASSERT_EQ(y_->data.real_double[0], 39.0);
            TEST_ASSERT_EQ(y_->data.real_double[1], 48.0);
            TEST_ASSERT_EQ(y_->data.real_double[2], 48.0);
            mtxvector_free(&y);
        }
        mtxvector_free(&x);
        mtxmatrix_free(&A);
    }

    /*
     * Real, double precision, symmetric matrices
     */

    {
        struct mtxmatrix A;
        struct mtxvector x;
        struct mtxvector y;
        int num_rows = 3;
        int num_columns = 3;
        int num_nonzeros = 5;
        int Arowidx[] = {0, 0, 1, 1, 2};
        int Acolidx[] = {0, 1, 1, 2, 2};
        double Adata[] = {1.0, 2.0, 5.0, 6.0, 9.0};
        double xdata[] = {3.0, 2.0, 1.0};
        double ydata[] = {1.0, 0.0, 2.0};
        err = mtxmatrix_init_entries_real_double(
            &A, mtxmatrix_dense, mtx_symmetric, num_rows, num_columns, num_nonzeros, Arowidx, Acolidx, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_init_real_double(&x, mtxvector_base, num_columns, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        {
            err = mtxvector_init_real_double(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemv(mtx_notrans, 2.0f, &A, &x, 3.0f, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
            TEST_ASSERT_EQ(mtx_field_real, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(3, y_->size);
            TEST_ASSERT_EQ(y_->data.real_double[0], 17.0);
            TEST_ASSERT_EQ(y_->data.real_double[1], 44.0);
            TEST_ASSERT_EQ(y_->data.real_double[2], 48.0);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_real_double(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemv(mtx_trans, 2.0f, &A, &x, 3.0f, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
            TEST_ASSERT_EQ(mtx_field_real, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(3, y_->size);
            TEST_ASSERT_EQ(y_->data.real_double[0], 17.0);
            TEST_ASSERT_EQ(y_->data.real_double[1], 44.0);
            TEST_ASSERT_EQ(y_->data.real_double[2], 48.0);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_real_double(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dgemv(mtx_notrans, 2.0, &A, &x, 3.0, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
            TEST_ASSERT_EQ(mtx_field_real, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(3, y_->size);
            TEST_ASSERT_EQ(y_->data.real_double[0], 17.0);
            TEST_ASSERT_EQ(y_->data.real_double[1], 44.0);
            TEST_ASSERT_EQ(y_->data.real_double[2], 48.0);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_real_double(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dgemv(mtx_trans, 2.0, &A, &x, 3.0, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
            TEST_ASSERT_EQ(mtx_field_real, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(3, y_->size);
            TEST_ASSERT_EQ(y_->data.real_double[0], 17.0);
            TEST_ASSERT_EQ(y_->data.real_double[1], 44.0);
            TEST_ASSERT_EQ(y_->data.real_double[2], 48.0);
            mtxvector_free(&y);
        }
        mtxvector_free(&x);
        mtxmatrix_free(&A);
    }

    /*
     * Complex, single precision, unsymmetric matrices
     */

    {
        struct mtxmatrix A;
        struct mtxvector x;
        struct mtxvector y;
        int num_rows = 2;
        int num_columns = 2;
        int num_nonzeros = 4;
        int Arowidx[] = {0, 0, 1, 1};
        int Acolidx[] = {0, 1, 0, 1};
        float Adata[][2] = {{1.0f,2.0f}, {3.0f,4.0f}, {5.0f,6.0f}, {7.0f,8.0f}};
        float xdata[][2] = {{3.0f,1.0f}, {1.0f,2.0f}};
        float ydata[][2] = {{1.0f,0.0f}, {2.0f,2.0f}};
        err = mtxmatrix_init_entries_complex_single(
            &A, mtxmatrix_dense, mtx_unsymmetric, num_rows, num_columns, num_nonzeros, Arowidx, Acolidx, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_init_complex_single(&x, mtxvector_base, num_columns, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        {
            err = mtxvector_init_complex_single(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemv(mtx_notrans, 2.0f, &A, &x, 3.0f, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
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
            err = mtxvector_init_complex_single(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemv(mtx_trans, 2.0f, &A, &x, 3.0f, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
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
            err = mtxvector_init_complex_single(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemv(mtx_conjtrans, 2.0f, &A, &x, 3.0f, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
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
            err = mtxvector_init_complex_single(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dgemv(mtx_notrans, 2.0, &A, &x, 3.0, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
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
            err = mtxvector_init_complex_single(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dgemv(mtx_trans, 2.0, &A, &x, 3.0, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
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
            err = mtxvector_init_complex_single(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dgemv(mtx_conjtrans, 2.0, &A, &x, 3.0, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
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
            err = mtxvector_init_complex_single(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_cgemv(mtx_notrans, calpha, &A, &x, cbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
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
            err = mtxvector_init_complex_single(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_cgemv(mtx_trans, calpha, &A, &x, cbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
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
            err = mtxvector_init_complex_single(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_cgemv(mtx_conjtrans, calpha, &A, &x, cbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
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
            err = mtxvector_init_complex_single(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_zgemv(mtx_notrans, zalpha, &A, &x, zbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
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
            err = mtxvector_init_complex_single(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_zgemv(mtx_trans, zalpha, &A, &x, zbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
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
            err = mtxvector_init_complex_single(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_zgemv(mtx_conjtrans, zalpha, &A, &x, zbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
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

    /*
     * Complex, single precision, symmetric matrices
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
        float Adata[][2] = {{1.0f,2.0f}, {3.0f,4.0f}, {7.0f,8.0f}};
        float xdata[][2] = {{3.0f,1.0f}, {1.0f,2.0f}};
        float ydata[][2] = {{1.0f,0.0f}, {2.0f,2.0f}};
        err = mtxmatrix_init_entries_complex_single(
            &A, mtxmatrix_dense, mtx_symmetric, num_rows, num_columns, num_nonzeros, Arowidx, Acolidx, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_init_complex_single(&x, mtxvector_base, num_columns, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        {
            err = mtxvector_init_complex_single(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemv(mtx_notrans, 2.0f, &A, &x, 3.0f, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_single[0][0], -5.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[0][1], 34.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][0], -2.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1], 80.0f);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_complex_single(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemv(mtx_trans, 2.0f, &A, &x, 3.0f, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_single[0][0], -5.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[0][1], 34.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][0], -2.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1], 80.0f);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_complex_single(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemv(mtx_conjtrans, 2.0f, &A, &x, 3.0f, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_single[0][0], 35.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[0][1], -6.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][0], 78.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1],  0.0f);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_complex_single(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dgemv(mtx_notrans, 2.0, &A, &x, 3.0, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_single[0][0], -5.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[0][1], 34.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][0], -2.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1], 80.0f);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_complex_single(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dgemv(mtx_trans, 2.0, &A, &x, 3.0, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_single[0][0], -5.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[0][1], 34.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][0], -2.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1], 80.0f);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_complex_single(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dgemv(mtx_conjtrans, 2.0, &A, &x, 3.0, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_single[0][0], 35.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[0][1], -6.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][0], 78.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1],  0.0f);
            mtxvector_free(&y);
        }
        float calpha[2] = {0.0f, 2.0f};
        float cbeta[2]  = {3.0f, 1.0f};
        {
            err = mtxvector_init_complex_single(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_cgemv(mtx_notrans, calpha, &A, &x, cbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_single[0][0], -31.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[0][1],  -7.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][0], -70.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1],   0.0f);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_complex_single(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_cgemv(mtx_trans, calpha, &A, &x, cbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_single[0][0], -31.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[0][1],  -7.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][0], -70.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1],   0.0f);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_complex_single(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_cgemv(mtx_conjtrans, calpha, &A, &x, cbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_single[0][0],  9.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[0][1], 33.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][0], 10.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1], 80.0f);
            mtxvector_free(&y);
        }
        double zalpha[2] = {0.0, 2.0};
        double zbeta[2]  = {3.0, 1.0};
        {
            err = mtxvector_init_complex_single(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_zgemv(mtx_notrans, zalpha, &A, &x, zbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_single[0][0], -31.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[0][1],  -7.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][0], -70.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1],   0.0f);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_complex_single(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_zgemv(mtx_trans, zalpha, &A, &x, zbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_single[0][0], -31.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[0][1],  -7.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][0], -70.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1],   0.0f);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_complex_single(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_zgemv(mtx_conjtrans, zalpha, &A, &x, zbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_single[0][0],  9.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[0][1], 33.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][0], 10.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1], 80.0f);
            mtxvector_free(&y);
        }
        mtxvector_free(&x);
        mtxmatrix_free(&A);
    }

    /*
     * Complex, single precision, hermitian matrices
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
        float Adata[][2] = {{1.0f,0.0f}, {3.0f,4.0f}, {7.0f,0.0f}};
        float xdata[][2] = {{3.0f,1.0f}, {1.0f,2.0f}};
        float ydata[][2] = {{1.0f,0.0f}, {2.0f,2.0f}};
        err = mtxmatrix_init_entries_complex_single(
            &A, mtxmatrix_dense, mtx_hermitian, num_rows, num_columns, num_nonzeros, Arowidx, Acolidx, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_init_complex_single(&x, mtxvector_base, num_columns, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        {
            err = mtxvector_init_complex_single(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemv(mtx_notrans, 2.0f, &A, &x, 3.0f, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_single[0][0], -1.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[0][1], 22.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][0], 46.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1], 16.0f);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_complex_single(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemv(mtx_trans, 2.0f, &A, &x, 3.0f, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_single[0][0], 31.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[0][1],  6.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][0], 30.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1], 64.0f);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_complex_single(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemv(mtx_conjtrans, 2.0f, &A, &x, 3.0f, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_single[0][0], -1.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[0][1], 22.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][0], 46.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1], 16.0f);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_complex_single(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dgemv(mtx_notrans, 2.0, &A, &x, 3.0, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_single[0][0], -1.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[0][1], 22.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][0], 46.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1], 16.0f);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_complex_single(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dgemv(mtx_trans, 2.0, &A, &x, 3.0, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_single[0][0], 31.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[0][1],  6.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][0], 30.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1], 64.0f);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_complex_single(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dgemv(mtx_conjtrans, 2.0, &A, &x, 3.0, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_single[0][0], -1.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[0][1], 22.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][0], 46.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1], 16.0f);
            mtxvector_free(&y);
        }
        float calpha[2] = {0.0f, 2.0f};
        float cbeta[2]  = {3.0f, 1.0f};
        {
            err = mtxvector_init_complex_single(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_cgemv(mtx_notrans, calpha, &A, &x, cbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_single[0][0], -19.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[0][1],  -3.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][0],  -6.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1],  48.0f);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_complex_single(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_cgemv(mtx_trans, calpha, &A, &x, cbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_single[0][0],  -3.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[0][1],  29.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][0], -54.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1],  32.0f);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_complex_single(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_cgemv(mtx_conjtrans, calpha, &A, &x, cbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_single[0][0],-19.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[0][1], -3.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][0], -6.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1], 48.0f);
            mtxvector_free(&y);
        }
        double zalpha[2] = {0.0, 2.0};
        double zbeta[2]  = {3.0, 1.0};
        {
            err = mtxvector_init_complex_single(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_zgemv(mtx_notrans, zalpha, &A, &x, zbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_single[0][0], -19.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[0][1],  -3.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][0],  -6.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1],  48.0f);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_complex_single(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_zgemv(mtx_trans, zalpha, &A, &x, zbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_single[0][0],  -3.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[0][1],  29.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][0], -54.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1],  32.0f);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_complex_single(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_zgemv(mtx_conjtrans, zalpha, &A, &x, zbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_single[0][0], -19.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[0][1],  -3.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][0],  -6.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1],  48.0f);
            mtxvector_free(&y);
        }
        mtxvector_free(&x);
        mtxmatrix_free(&A);
    }

    /*
     * Complex, double precision, unsymmetric matrices
     */

    {
        struct mtxmatrix A;
        struct mtxvector x;
        struct mtxvector y;
        int num_rows = 2;
        int num_columns = 2;
        int num_nonzeros = 4;
        int Arowidx[] = {0, 0, 1, 1};
        int Acolidx[] = {0, 1, 0, 1};
        double Adata[][2] = {{1.0,2.0}, {3.0,4.0}, {5.0,6.0}, {7.0,8.0}};
        double xdata[][2] = {{3.0,1.0}, {1.0,2.0}};
        double ydata[][2] = {{1.0,0.0}, {2.0,2.0}};
        err = mtxmatrix_init_entries_complex_double(
            &A, mtxmatrix_dense, mtx_unsymmetric, num_rows, num_columns, num_nonzeros, Arowidx, Acolidx, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_init_complex_double(&x, mtxvector_base, num_columns, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        {
            err = mtxvector_init_complex_double(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemv(mtx_notrans, 2.0f, &A, &x, 3.0f, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
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
            err = mtxvector_init_complex_double(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemv(mtx_trans, 2.0f, &A, &x, 3.0f, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
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
            err = mtxvector_init_complex_double(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemv(mtx_conjtrans, 2.0f, &A, &x, 3.0f, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
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
            err = mtxvector_init_complex_double(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dgemv(mtx_notrans, 2.0, &A, &x, 3.0, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
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
            err = mtxvector_init_complex_double(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dgemv(mtx_trans, 2.0, &A, &x, 3.0, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
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
            err = mtxvector_init_complex_double(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dgemv(mtx_conjtrans, 2.0, &A, &x, 3.0, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
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
            err = mtxvector_init_complex_double(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_cgemv(mtx_notrans, calpha, &A, &x, cbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
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
            err = mtxvector_init_complex_double(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_cgemv(mtx_trans, calpha, &A, &x, cbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
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
            err = mtxvector_init_complex_double(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_cgemv(mtx_conjtrans, calpha, &A, &x, cbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
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
            err = mtxvector_init_complex_double(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_zgemv(mtx_notrans, zalpha, &A, &x, zbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
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
            err = mtxvector_init_complex_double(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_zgemv(mtx_trans, zalpha, &A, &x, zbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
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
            err = mtxvector_init_complex_double(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_zgemv(mtx_conjtrans, zalpha, &A, &x, zbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
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
     * Complex, double precision, symmetric matrices
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
        double Adata[][2] = {{1.0,2.0}, {3.0,4.0}, {7.0,8.0}};
        double xdata[][2] = {{3.0,1.0}, {1.0,2.0}};
        double ydata[][2] = {{1.0,0.0}, {2.0,2.0}};
        err = mtxmatrix_init_entries_complex_double(
            &A, mtxmatrix_dense, mtx_symmetric, num_rows, num_columns, num_nonzeros, Arowidx, Acolidx, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_init_complex_double(&x, mtxvector_base, num_columns, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        {
            err = mtxvector_init_complex_double(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemv(mtx_notrans, 2.0f, &A, &x, 3.0f, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_double[0][0], -5.0);
            TEST_ASSERT_EQ(y_->data.complex_double[0][1], 34.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][0], -2.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][1], 80.0);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_complex_double(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemv(mtx_trans, 2.0f, &A, &x, 3.0f, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_double[0][0], -5.0);
            TEST_ASSERT_EQ(y_->data.complex_double[0][1], 34.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][0], -2.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][1], 80.0);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_complex_double(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemv(mtx_conjtrans, 2.0f, &A, &x, 3.0f, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_double[0][0], 35.0);
            TEST_ASSERT_EQ(y_->data.complex_double[0][1], -6.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][0], 78.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][1],  0.0);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_complex_double(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dgemv(mtx_notrans, 2.0, &A, &x, 3.0, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_double[0][0], -5.0);
            TEST_ASSERT_EQ(y_->data.complex_double[0][1], 34.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][0], -2.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][1], 80.0);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_complex_double(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dgemv(mtx_trans, 2.0, &A, &x, 3.0, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_double[0][0], -5.0);
            TEST_ASSERT_EQ(y_->data.complex_double[0][1], 34.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][0], -2.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][1], 80.0);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_complex_double(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dgemv(mtx_conjtrans, 2.0, &A, &x, 3.0, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_double[0][0], 35.0);
            TEST_ASSERT_EQ(y_->data.complex_double[0][1], -6.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][0], 78.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][1],  0.0);
            mtxvector_free(&y);
        }
        float calpha[2] = {0.0f, 2.0f};
        float cbeta[2]  = {3.0f, 1.0f};
        {
            err = mtxvector_init_complex_double(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_cgemv(mtx_notrans, calpha, &A, &x, cbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_double[0][0], -31.0);
            TEST_ASSERT_EQ(y_->data.complex_double[0][1],  -7.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][0], -70.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][1],   0.0);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_complex_double(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_cgemv(mtx_trans, calpha, &A, &x, cbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_double[0][0], -31.0);
            TEST_ASSERT_EQ(y_->data.complex_double[0][1],  -7.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][0], -70.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][1],   0.0);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_complex_double(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_cgemv(mtx_conjtrans, calpha, &A, &x, cbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_double[0][0],  9.0);
            TEST_ASSERT_EQ(y_->data.complex_double[0][1], 33.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][0], 10.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][1], 80.0);
            mtxvector_free(&y);
        }
        double zalpha[2] = {0.0, 2.0};
        double zbeta[2]  = {3.0, 1.0};
        {
            err = mtxvector_init_complex_double(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_zgemv(mtx_notrans, zalpha, &A, &x, zbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_double[0][0], -31.0);
            TEST_ASSERT_EQ(y_->data.complex_double[0][1],  -7.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][0], -70.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][1],   0.0);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_complex_double(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_zgemv(mtx_trans, zalpha, &A, &x, zbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_double[0][0], -31.0);
            TEST_ASSERT_EQ(y_->data.complex_double[0][1],  -7.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][0], -70.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][1],   0.0);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_complex_double(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_zgemv(mtx_conjtrans, zalpha, &A, &x, zbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_double[0][0],  9.0);
            TEST_ASSERT_EQ(y_->data.complex_double[0][1], 33.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][0], 10.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][1], 80.0);
            mtxvector_free(&y);
        }
        mtxvector_free(&x);
        mtxmatrix_free(&A);
    }

    /*
     * Complex, double precision, hermitian matrices
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
        double Adata[][2] = {{1.0,0.0}, {3.0,4.0}, {7.0,0.0}};
        double xdata[][2] = {{3.0,1.0}, {1.0,2.0}};
        double ydata[][2] = {{1.0,0.0}, {2.0,2.0}};
        err = mtxmatrix_init_entries_complex_double(
            &A, mtxmatrix_dense, mtx_hermitian, num_rows, num_columns, num_nonzeros, Arowidx, Acolidx, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_init_complex_double(&x, mtxvector_base, num_columns, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        {
            err = mtxvector_init_complex_double(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemv(mtx_notrans, 2.0f, &A, &x, 3.0f, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_double[0][0], -1.0);
            TEST_ASSERT_EQ(y_->data.complex_double[0][1], 22.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][0], 46.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][1], 16.0);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_complex_double(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemv(mtx_trans, 2.0f, &A, &x, 3.0f, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_double[0][0], 31.0);
            TEST_ASSERT_EQ(y_->data.complex_double[0][1],  6.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][0], 30.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][1], 64.0);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_complex_double(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemv(mtx_conjtrans, 2.0f, &A, &x, 3.0f, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_double[0][0], -1.0);
            TEST_ASSERT_EQ(y_->data.complex_double[0][1], 22.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][0], 46.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][1], 16.0);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_complex_double(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dgemv(mtx_notrans, 2.0, &A, &x, 3.0, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_double[0][0], -1.0);
            TEST_ASSERT_EQ(y_->data.complex_double[0][1], 22.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][0], 46.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][1], 16.0);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_complex_double(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dgemv(mtx_trans, 2.0, &A, &x, 3.0, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_double[0][0], 31.0);
            TEST_ASSERT_EQ(y_->data.complex_double[0][1],  6.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][0], 30.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][1], 64.0);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_complex_double(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dgemv(mtx_conjtrans, 2.0, &A, &x, 3.0, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_double[0][0], -1.0);
            TEST_ASSERT_EQ(y_->data.complex_double[0][1], 22.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][0], 46.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][1], 16.0);
            mtxvector_free(&y);
        }
        float calpha[2] = {0.0f, 2.0f};
        float cbeta[2]  = {3.0f, 1.0f};
        {
            err = mtxvector_init_complex_double(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_cgemv(mtx_notrans, calpha, &A, &x, cbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_double[0][0], -19.0);
            TEST_ASSERT_EQ(y_->data.complex_double[0][1],  -3.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][0],  -6.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][1],  48.0);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_complex_double(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_cgemv(mtx_trans, calpha, &A, &x, cbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_double[0][0],  -3.0);
            TEST_ASSERT_EQ(y_->data.complex_double[0][1],  29.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][0], -54.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][1],  32.0);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_complex_double(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_cgemv(mtx_conjtrans, calpha, &A, &x, cbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_double[0][0],-19.0);
            TEST_ASSERT_EQ(y_->data.complex_double[0][1], -3.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][0], -6.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][1], 48.0);
            mtxvector_free(&y);
        }
        double zalpha[2] = {0.0, 2.0};
        double zbeta[2]  = {3.0, 1.0};
        {
            err = mtxvector_init_complex_double(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_zgemv(mtx_notrans, zalpha, &A, &x, zbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_double[0][0], -19.0);
            TEST_ASSERT_EQ(y_->data.complex_double[0][1],  -3.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][0],  -6.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][1],  48.0);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_complex_double(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_zgemv(mtx_trans, zalpha, &A, &x, zbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_double[0][0],  -3.0);
            TEST_ASSERT_EQ(y_->data.complex_double[0][1],  29.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][0], -54.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][1],  32.0);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_complex_double(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_zgemv(mtx_conjtrans, zalpha, &A, &x, zbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_double[0][0], -19.0);
            TEST_ASSERT_EQ(y_->data.complex_double[0][1],  -3.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][0],  -6.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][1],  48.0);
            mtxvector_free(&y);
        }
        mtxvector_free(&x);
        mtxmatrix_free(&A);
    }

    /*
     * Integer, single precision, unsymmetric matrices
     */

    {
        struct mtxmatrix A;
        struct mtxvector x;
        struct mtxvector y;
        int num_rows = 3;
        int num_columns = 3;
        int num_nonzeros = 9;
        int Arowidx[] = {0, 0, 0, 1, 1, 1, 2, 2, 2};
        int Acolidx[] = {0, 1, 2, 0, 1, 2, 0, 1, 2};
        int32_t Adata[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
        int32_t xdata[] = {3, 2, 1};
        int32_t ydata[] = {1, 0, 2};
        err = mtxmatrix_init_entries_integer_single(
            &A, mtxmatrix_dense, mtx_unsymmetric, num_rows, num_columns, num_nonzeros, Arowidx, Acolidx, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_init_integer_single(&x, mtxvector_base, num_columns, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        {
            err = mtxvector_init_integer_single(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemv(mtx_notrans, 2.0f, &A, &x, 3.0f, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
            TEST_ASSERT_EQ(mtx_field_integer, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(3, y_->size);
            TEST_ASSERT_EQ(y_->data.integer_single[0], 23);
            TEST_ASSERT_EQ(y_->data.integer_single[1], 56);
            TEST_ASSERT_EQ(y_->data.integer_single[2], 98);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_integer_single(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemv(mtx_trans, 2.0f, &A, &x, 3.0f, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
            TEST_ASSERT_EQ(mtx_field_integer, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(3, y_->size);
            TEST_ASSERT_EQ(y_->data.integer_single[0], 39);
            TEST_ASSERT_EQ(y_->data.integer_single[1], 48);
            TEST_ASSERT_EQ(y_->data.integer_single[2], 66);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_integer_single(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dgemv(mtx_notrans, 2.0, &A, &x, 3.0, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
            TEST_ASSERT_EQ(mtx_field_integer, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(3, y_->size);
            TEST_ASSERT_EQ(y_->data.integer_single[0], 23);
            TEST_ASSERT_EQ(y_->data.integer_single[1], 56);
            TEST_ASSERT_EQ(y_->data.integer_single[2], 98);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_integer_single(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dgemv(mtx_trans, 2.0, &A, &x, 3.0, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
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

    /*
     * Integer, single precision, symmetric matrices
     */

    {
        struct mtxmatrix A;
        struct mtxvector x;
        struct mtxvector y;
        int num_rows = 3;
        int num_columns = 3;
        int num_nonzeros = 6;
        int Arowidx[] = {0, 0, 0, 1, 1, 2};
        int Acolidx[] = {0, 1, 2, 1, 2, 2};
        int32_t Adata[] = {1, 2, 3, 5, 6, 9};
        int32_t xdata[] = {3, 2, 1};
        int32_t ydata[] = {1, 0, 2};
        err = mtxmatrix_init_entries_integer_single(
            &A, mtxmatrix_dense, mtx_symmetric, num_rows, num_columns, num_nonzeros, Arowidx, Acolidx, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_init_integer_single(&x, mtxvector_base, num_columns, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        {
            err = mtxvector_init_integer_single(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemv(mtx_notrans, 2.0f, &A, &x, 3.0f, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
            TEST_ASSERT_EQ(mtx_field_integer, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(3, y_->size);
            TEST_ASSERT_EQ(y_->data.integer_single[0], 23);
            TEST_ASSERT_EQ(y_->data.integer_single[1], 44);
            TEST_ASSERT_EQ(y_->data.integer_single[2], 66);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_integer_single(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemv(mtx_trans, 2.0f, &A, &x, 3.0f, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
            TEST_ASSERT_EQ(mtx_field_integer, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(3, y_->size);
            TEST_ASSERT_EQ(y_->data.integer_single[0], 23);
            TEST_ASSERT_EQ(y_->data.integer_single[1], 44);
            TEST_ASSERT_EQ(y_->data.integer_single[2], 66);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_integer_single(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dgemv(mtx_notrans, 2.0, &A, &x, 3.0, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
            TEST_ASSERT_EQ(mtx_field_integer, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(3, y_->size);
            TEST_ASSERT_EQ(y_->data.integer_single[0], 23);
            TEST_ASSERT_EQ(y_->data.integer_single[1], 44);
            TEST_ASSERT_EQ(y_->data.integer_single[2], 66);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_integer_single(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dgemv(mtx_trans, 2.0, &A, &x, 3.0, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
            TEST_ASSERT_EQ(mtx_field_integer, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(3, y_->size);
            TEST_ASSERT_EQ(y_->data.integer_single[0], 23);
            TEST_ASSERT_EQ(y_->data.integer_single[1], 44);
            TEST_ASSERT_EQ(y_->data.integer_single[2], 66);
            mtxvector_free(&y);
        }
        mtxvector_free(&x);
        mtxmatrix_free(&A);
    }

    /*
     * Integer, double precision, unsymmetric matrices
     */

    {
        struct mtxmatrix A;
        struct mtxvector x;
        struct mtxvector y;
        int num_rows = 3;
        int num_columns = 3;
        int num_nonzeros = 9;
        int Arowidx[] = {0, 0, 0, 1, 1, 1, 2, 2, 2};
        int Acolidx[] = {0, 1, 2, 0, 1, 2, 0, 1, 2};
        int64_t Adata[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
        int64_t xdata[] = {3, 2, 1};
        int64_t ydata[] = {1, 0, 2};
        err = mtxmatrix_init_entries_integer_double(
            &A, mtxmatrix_dense, mtx_unsymmetric, num_rows, num_columns, num_nonzeros, Arowidx, Acolidx, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_init_integer_double(&x, mtxvector_base, num_columns, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        {
            err = mtxvector_init_integer_double(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemv(mtx_notrans, 2.0f, &A, &x, 3.0f, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
            TEST_ASSERT_EQ(mtx_field_integer, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(3, y_->size);
            TEST_ASSERT_EQ(y_->data.integer_double[0], 23);
            TEST_ASSERT_EQ(y_->data.integer_double[1], 56);
            TEST_ASSERT_EQ(y_->data.integer_double[2], 98);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_integer_double(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemv(mtx_trans, 2.0f, &A, &x, 3.0f, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
            TEST_ASSERT_EQ(mtx_field_integer, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(3, y_->size);
            TEST_ASSERT_EQ(y_->data.integer_double[0], 39);
            TEST_ASSERT_EQ(y_->data.integer_double[1], 48);
            TEST_ASSERT_EQ(y_->data.integer_double[2], 66);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_integer_double(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dgemv(mtx_notrans, 2.0, &A, &x, 3.0, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
            TEST_ASSERT_EQ(mtx_field_integer, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(3, y_->size);
            TEST_ASSERT_EQ(y_->data.integer_double[0], 23);
            TEST_ASSERT_EQ(y_->data.integer_double[1], 56);
            TEST_ASSERT_EQ(y_->data.integer_double[2], 98);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_integer_double(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dgemv(mtx_trans, 2.0, &A, &x, 3.0, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
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

    /*
     * Integer, double precision, symmetric matrices
     */

    {
        struct mtxmatrix A;
        struct mtxvector x;
        struct mtxvector y;
        int num_rows = 3;
        int num_columns = 3;
        int num_nonzeros = 6;
        int Arowidx[] = {0, 0, 0, 1, 1, 2};
        int Acolidx[] = {0, 1, 2, 1, 2, 2};
        int64_t Adata[] = {1, 2, 3, 5, 6, 9};
        int64_t xdata[] = {3, 2, 1};
        int64_t ydata[] = {1, 0, 2};
        err = mtxmatrix_init_entries_integer_double(
            &A, mtxmatrix_dense, mtx_symmetric, num_rows, num_columns, num_nonzeros, Arowidx, Acolidx, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_init_integer_double(&x, mtxvector_base, num_columns, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        {
            err = mtxvector_init_integer_double(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemv(mtx_notrans, 2.0f, &A, &x, 3.0f, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
            TEST_ASSERT_EQ(mtx_field_integer, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(3, y_->size);
            TEST_ASSERT_EQ(y_->data.integer_double[0], 23);
            TEST_ASSERT_EQ(y_->data.integer_double[1], 44);
            TEST_ASSERT_EQ(y_->data.integer_double[2], 66);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_integer_double(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemv(mtx_trans, 2.0f, &A, &x, 3.0f, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
            TEST_ASSERT_EQ(mtx_field_integer, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(3, y_->size);
            TEST_ASSERT_EQ(y_->data.integer_double[0], 23);
            TEST_ASSERT_EQ(y_->data.integer_double[1], 44);
            TEST_ASSERT_EQ(y_->data.integer_double[2], 66);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_integer_double(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dgemv(mtx_notrans, 2.0, &A, &x, 3.0, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
            TEST_ASSERT_EQ(mtx_field_integer, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(3, y_->size);
            TEST_ASSERT_EQ(y_->data.integer_double[0], 23);
            TEST_ASSERT_EQ(y_->data.integer_double[1], 44);
            TEST_ASSERT_EQ(y_->data.integer_double[2], 66);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_integer_double(&y, mtxvector_base, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dgemv(mtx_trans, 2.0, &A, &x, 3.0, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.type);
            const struct mtxvector_base * y_ = &y.storage.base;
            TEST_ASSERT_EQ(mtx_field_integer, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(3, y_->size);
            TEST_ASSERT_EQ(y_->data.integer_double[0], 23);
            TEST_ASSERT_EQ(y_->data.integer_double[1], 44);
            TEST_ASSERT_EQ(y_->data.integer_double[2], 66);
            mtxvector_free(&y);
        }
        mtxvector_free(&x);
        mtxmatrix_free(&A);
    }
    return TEST_SUCCESS;
}
/**
 * ‘main()’ entry point and test driver.
 */
int main(int argc, char * argv[])
{
    TEST_SUITE_BEGIN("Running tests for dense matrices\n");
    TEST_RUN(test_mtxmatrix_dense_from_mtxfile);
    TEST_RUN(test_mtxmatrix_dense_to_mtxfile);
    TEST_RUN(test_mtxmatrix_dense_gemv);
    TEST_SUITE_END();
    return (TEST_SUITE_STATUS == TEST_SUCCESS) ?
        EXIT_SUCCESS : EXIT_FAILURE;
}
