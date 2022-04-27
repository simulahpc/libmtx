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
 * Last modified: 2022-04-27
 *
 * Unit tests for sparse matrices in coordinate format.
 */

#include "test.h"

#include <libmtx/error.h>
#include <libmtx/matrix/matrix.h>
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
 * ‘test_mtxmatrix_coordinate_from_mtxfile()’ tests converting Matrix
 * Market files to matrices.
 */
int test_mtxmatrix_coordinate_from_mtxfile(void)
{
    int err;

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
        err = mtxmatrix_from_mtxfile(&x, mtxmatrix_auto, &mtxfile);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxmatrix_coordinate, x.type);
        const struct mtxmatrix_coordinate * x_ = &x.storage.coordinate;
        TEST_ASSERT_EQ(mtx_unsymmetric, x_->symmetry);
        TEST_ASSERT_EQ(3, x_->num_rows);
        TEST_ASSERT_EQ(3, x_->num_columns);
        TEST_ASSERT_EQ(4, x_->num_nonzeros);
        TEST_ASSERT_EQ(4, x_->size);
        TEST_ASSERT_EQ(x_->rowidx[0], 0); TEST_ASSERT_EQ(x_->colidx[0], 0);
        TEST_ASSERT_EQ(x_->rowidx[1], 0); TEST_ASSERT_EQ(x_->colidx[1], 2);
        TEST_ASSERT_EQ(x_->rowidx[2], 1); TEST_ASSERT_EQ(x_->colidx[2], 0);
        TEST_ASSERT_EQ(x_->rowidx[3], 2); TEST_ASSERT_EQ(x_->colidx[3], 2);
        TEST_ASSERT_EQ(mtx_field_real, x_->a.field);
        TEST_ASSERT_EQ(mtx_single, x_->a.precision);
        TEST_ASSERT_EQ(x_->a.data.real_single[0], 1.0f);
        TEST_ASSERT_EQ(x_->a.data.real_single[1], 3.0f);
        TEST_ASSERT_EQ(x_->a.data.real_single[2], 4.0f);
        TEST_ASSERT_EQ(x_->a.data.real_single[3], 9.0f);
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
        err = mtxmatrix_from_mtxfile(&x, mtxmatrix_auto, &mtxfile);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxmatrix_coordinate, x.type);
        const struct mtxmatrix_coordinate * x_ = &x.storage.coordinate;
        TEST_ASSERT_EQ(mtx_symmetric, x_->symmetry);
        TEST_ASSERT_EQ(3, x_->num_rows);
        TEST_ASSERT_EQ(3, x_->num_columns);
        TEST_ASSERT_EQ(6, x_->num_nonzeros);
        TEST_ASSERT_EQ(4, x_->size);
        TEST_ASSERT_EQ(x_->rowidx[0], 0); TEST_ASSERT_EQ(x_->colidx[0], 0);
        TEST_ASSERT_EQ(x_->rowidx[1], 0); TEST_ASSERT_EQ(x_->colidx[1], 2);
        TEST_ASSERT_EQ(x_->rowidx[2], 1); TEST_ASSERT_EQ(x_->colidx[2], 0);
        TEST_ASSERT_EQ(x_->rowidx[3], 2); TEST_ASSERT_EQ(x_->colidx[3], 2);
        TEST_ASSERT_EQ(mtx_field_real, x_->a.field);
        TEST_ASSERT_EQ(mtx_single, x_->a.precision);
        TEST_ASSERT_EQ(x_->a.data.real_single[0], 1.0f);
        TEST_ASSERT_EQ(x_->a.data.real_single[1], 3.0f);
        TEST_ASSERT_EQ(x_->a.data.real_single[2], 4.0f);
        TEST_ASSERT_EQ(x_->a.data.real_single[3], 9.0f);
        mtxmatrix_free(&x);
        mtxfile_free(&mtxfile);
    }
    {
        int num_rows = 3;
        int num_columns = 3;
        struct mtxfile_matrix_coordinate_real_single mtxdata[] = {
            {1,3,3.0f}, {2,1,4.0f}, {3,2,8.0f}};
        int64_t size = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile mtxfile;
        err = mtxfile_init_matrix_coordinate_real_single(
            &mtxfile, mtxfile_skew_symmetric, num_rows, num_columns, size,
            mtxdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxmatrix x;
        err = mtxmatrix_from_mtxfile(&x, mtxmatrix_auto, &mtxfile);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxmatrix_coordinate, x.type);
        const struct mtxmatrix_coordinate * x_ = &x.storage.coordinate;
        TEST_ASSERT_EQ(mtx_skew_symmetric, x_->symmetry);
        TEST_ASSERT_EQ(3, x_->num_rows);
        TEST_ASSERT_EQ(3, x_->num_columns);
        TEST_ASSERT_EQ(6, x_->num_nonzeros);
        TEST_ASSERT_EQ(3, x_->size);
        TEST_ASSERT_EQ(x_->rowidx[0], 0); TEST_ASSERT_EQ(x_->colidx[0], 2);
        TEST_ASSERT_EQ(x_->rowidx[1], 1); TEST_ASSERT_EQ(x_->colidx[1], 0);
        TEST_ASSERT_EQ(x_->rowidx[2], 2); TEST_ASSERT_EQ(x_->colidx[2], 1);
        TEST_ASSERT_EQ(mtx_field_real, x_->a.field);
        TEST_ASSERT_EQ(mtx_single, x_->a.precision);
        TEST_ASSERT_EQ(x_->a.data.real_single[0], 3.0f);
        TEST_ASSERT_EQ(x_->a.data.real_single[1], 4.0f);
        TEST_ASSERT_EQ(x_->a.data.real_single[2], 8.0f);
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
        err = mtxmatrix_from_mtxfile(&x, mtxmatrix_auto, &mtxfile);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxmatrix_coordinate, x.type);
        const struct mtxmatrix_coordinate * x_ = &x.storage.coordinate;
        TEST_ASSERT_EQ(mtx_unsymmetric, x_->symmetry);
        TEST_ASSERT_EQ(3, x_->num_rows);
        TEST_ASSERT_EQ(3, x_->num_columns);
        TEST_ASSERT_EQ(4, x_->num_nonzeros);
        TEST_ASSERT_EQ(4, x_->size);
        TEST_ASSERT_EQ(x_->rowidx[0], 0); TEST_ASSERT_EQ(x_->colidx[0], 0);
        TEST_ASSERT_EQ(x_->rowidx[1], 0); TEST_ASSERT_EQ(x_->colidx[1], 2);
        TEST_ASSERT_EQ(x_->rowidx[2], 1); TEST_ASSERT_EQ(x_->colidx[2], 0);
        TEST_ASSERT_EQ(x_->rowidx[3], 2); TEST_ASSERT_EQ(x_->colidx[3], 2);
        TEST_ASSERT_EQ(mtx_field_real, x_->a.field);
        TEST_ASSERT_EQ(mtx_double, x_->a.precision);
        TEST_ASSERT_EQ(x_->a.data.real_double[0], 1.0);
        TEST_ASSERT_EQ(x_->a.data.real_double[1], 3.0);
        TEST_ASSERT_EQ(x_->a.data.real_double[2], 4.0);
        TEST_ASSERT_EQ(x_->a.data.real_double[3], 9.0);
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
            &mtxfile, mtxfile_symmetric, num_rows, num_columns, size, mtxdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxmatrix x;
        err = mtxmatrix_from_mtxfile(&x, mtxmatrix_auto, &mtxfile);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxmatrix_coordinate, x.type);
        const struct mtxmatrix_coordinate * x_ = &x.storage.coordinate;
        TEST_ASSERT_EQ(mtx_symmetric, x_->symmetry);
        TEST_ASSERT_EQ(3, x_->num_rows);
        TEST_ASSERT_EQ(3, x_->num_columns);
        TEST_ASSERT_EQ(6, x_->num_nonzeros);
        TEST_ASSERT_EQ(4, x_->size);
        TEST_ASSERT_EQ(x_->rowidx[0], 0); TEST_ASSERT_EQ(x_->colidx[0], 0);
        TEST_ASSERT_EQ(x_->rowidx[1], 0); TEST_ASSERT_EQ(x_->colidx[1], 2);
        TEST_ASSERT_EQ(x_->rowidx[2], 1); TEST_ASSERT_EQ(x_->colidx[2], 0);
        TEST_ASSERT_EQ(x_->rowidx[3], 2); TEST_ASSERT_EQ(x_->colidx[3], 2);
        TEST_ASSERT_EQ(mtx_field_real, x_->a.field);
        TEST_ASSERT_EQ(mtx_double, x_->a.precision);
        TEST_ASSERT_EQ(x_->a.data.real_double[0], 1.0);
        TEST_ASSERT_EQ(x_->a.data.real_double[1], 3.0);
        TEST_ASSERT_EQ(x_->a.data.real_double[2], 4.0);
        TEST_ASSERT_EQ(x_->a.data.real_double[3], 9.0);
        mtxmatrix_free(&x);
        mtxfile_free(&mtxfile);
    }
    {
        int num_rows = 3;
        int num_columns = 3;
        struct mtxfile_matrix_coordinate_real_double mtxdata[] = {
            {1,3,3.0}, {2,1,4.0}, {3,2,8.0}};
        int64_t size = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile mtxfile;
        err = mtxfile_init_matrix_coordinate_real_double(
            &mtxfile, mtxfile_skew_symmetric, num_rows, num_columns, size,
            mtxdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxmatrix x;
        err = mtxmatrix_from_mtxfile(&x, mtxmatrix_auto, &mtxfile);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxmatrix_coordinate, x.type);
        const struct mtxmatrix_coordinate * x_ = &x.storage.coordinate;
        TEST_ASSERT_EQ(mtx_skew_symmetric, x_->symmetry);
        TEST_ASSERT_EQ(3, x_->num_rows);
        TEST_ASSERT_EQ(3, x_->num_columns);
        TEST_ASSERT_EQ(6, x_->num_nonzeros);
        TEST_ASSERT_EQ(3, x_->size);
        TEST_ASSERT_EQ(x_->rowidx[0], 0); TEST_ASSERT_EQ(x_->colidx[0], 2);
        TEST_ASSERT_EQ(x_->rowidx[1], 1); TEST_ASSERT_EQ(x_->colidx[1], 0);
        TEST_ASSERT_EQ(x_->rowidx[2], 2); TEST_ASSERT_EQ(x_->colidx[2], 1);
        TEST_ASSERT_EQ(mtx_field_real, x_->a.field);
        TEST_ASSERT_EQ(mtx_double, x_->a.precision);
        TEST_ASSERT_EQ(x_->a.data.real_double[0], 3.0);
        TEST_ASSERT_EQ(x_->a.data.real_double[1], 4.0);
        TEST_ASSERT_EQ(x_->a.data.real_double[2], 8.0);
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
        err = mtxmatrix_from_mtxfile(&x, mtxmatrix_auto, &mtxfile);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxmatrix_coordinate, x.type);
        const struct mtxmatrix_coordinate * x_ = &x.storage.coordinate;
        TEST_ASSERT_EQ(mtx_unsymmetric, x_->symmetry);
        TEST_ASSERT_EQ(3, x_->num_rows);
        TEST_ASSERT_EQ(3, x_->num_columns);
        TEST_ASSERT_EQ(4, x_->num_nonzeros);
        TEST_ASSERT_EQ(4, x_->size);
        TEST_ASSERT_EQ(x_->rowidx[0], 0); TEST_ASSERT_EQ(x_->colidx[0], 0);
        TEST_ASSERT_EQ(x_->rowidx[1], 0); TEST_ASSERT_EQ(x_->colidx[1], 2);
        TEST_ASSERT_EQ(x_->rowidx[2], 1); TEST_ASSERT_EQ(x_->colidx[2], 0);
        TEST_ASSERT_EQ(x_->rowidx[3], 2); TEST_ASSERT_EQ(x_->colidx[3], 2);
        TEST_ASSERT_EQ(mtx_field_complex, x_->a.field);
        TEST_ASSERT_EQ(mtx_single, x_->a.precision);
        TEST_ASSERT_EQ(x_->a.data.complex_single[0][0], 1.0f);
        TEST_ASSERT_EQ(x_->a.data.complex_single[0][1],-1.0f);
        TEST_ASSERT_EQ(x_->a.data.complex_single[1][0], 3.0f);
        TEST_ASSERT_EQ(x_->a.data.complex_single[1][1],-3.0f);
        TEST_ASSERT_EQ(x_->a.data.complex_single[2][0], 4.0f);
        TEST_ASSERT_EQ(x_->a.data.complex_single[2][1],-4.0f);
        TEST_ASSERT_EQ(x_->a.data.complex_single[3][0], 9.0f);
        TEST_ASSERT_EQ(x_->a.data.complex_single[3][1],-9.0f);
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
        err = mtxmatrix_from_mtxfile(&x, mtxmatrix_auto, &mtxfile);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxmatrix_coordinate, x.type);
        const struct mtxmatrix_coordinate * x_ = &x.storage.coordinate;
        TEST_ASSERT_EQ(mtx_symmetric, x_->symmetry);
        TEST_ASSERT_EQ(3, x_->num_rows);
        TEST_ASSERT_EQ(3, x_->num_columns);
        TEST_ASSERT_EQ(6, x_->num_nonzeros);
        TEST_ASSERT_EQ(4, x_->size);
        TEST_ASSERT_EQ(x_->rowidx[0], 0); TEST_ASSERT_EQ(x_->colidx[0], 0);
        TEST_ASSERT_EQ(x_->rowidx[1], 0); TEST_ASSERT_EQ(x_->colidx[1], 2);
        TEST_ASSERT_EQ(x_->rowidx[2], 1); TEST_ASSERT_EQ(x_->colidx[2], 0);
        TEST_ASSERT_EQ(x_->rowidx[3], 2); TEST_ASSERT_EQ(x_->colidx[3], 2);
        TEST_ASSERT_EQ(mtx_field_complex, x_->a.field);
        TEST_ASSERT_EQ(mtx_single, x_->a.precision);
        TEST_ASSERT_EQ(x_->a.data.complex_single[0][0], 1.0f);
        TEST_ASSERT_EQ(x_->a.data.complex_single[0][1],-1.0f);
        TEST_ASSERT_EQ(x_->a.data.complex_single[1][0], 3.0f);
        TEST_ASSERT_EQ(x_->a.data.complex_single[1][1],-3.0f);
        TEST_ASSERT_EQ(x_->a.data.complex_single[2][0], 4.0f);
        TEST_ASSERT_EQ(x_->a.data.complex_single[2][1],-4.0f);
        TEST_ASSERT_EQ(x_->a.data.complex_single[3][0], 9.0f);
        TEST_ASSERT_EQ(x_->a.data.complex_single[3][1],-9.0f);
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
        err = mtxmatrix_from_mtxfile(&x, mtxmatrix_auto, &mtxfile);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxmatrix_coordinate, x.type);
        const struct mtxmatrix_coordinate * x_ = &x.storage.coordinate;
        TEST_ASSERT_EQ(mtx_skew_symmetric, x_->symmetry);
        TEST_ASSERT_EQ(3, x_->num_rows);
        TEST_ASSERT_EQ(3, x_->num_columns);
        TEST_ASSERT_EQ(6, x_->num_nonzeros);
        TEST_ASSERT_EQ(3, x_->size);
        TEST_ASSERT_EQ(x_->rowidx[0], 0); TEST_ASSERT_EQ(x_->colidx[0], 1);
        TEST_ASSERT_EQ(x_->rowidx[1], 0); TEST_ASSERT_EQ(x_->colidx[1], 2);
        TEST_ASSERT_EQ(x_->rowidx[2], 1); TEST_ASSERT_EQ(x_->colidx[2], 2);
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
        err = mtxmatrix_from_mtxfile(&x, mtxmatrix_auto, &mtxfile);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxmatrix_coordinate, x.type);
        const struct mtxmatrix_coordinate * x_ = &x.storage.coordinate;
        TEST_ASSERT_EQ(mtx_hermitian, x_->symmetry);
        TEST_ASSERT_EQ(3, x_->num_rows);
        TEST_ASSERT_EQ(3, x_->num_columns);
        TEST_ASSERT_EQ(6, x_->num_nonzeros);
        TEST_ASSERT_EQ(4, x_->size);
        TEST_ASSERT_EQ(x_->rowidx[0], 0); TEST_ASSERT_EQ(x_->colidx[0], 0);
        TEST_ASSERT_EQ(x_->rowidx[1], 0); TEST_ASSERT_EQ(x_->colidx[1], 2);
        TEST_ASSERT_EQ(x_->rowidx[2], 1); TEST_ASSERT_EQ(x_->colidx[2], 0);
        TEST_ASSERT_EQ(x_->rowidx[3], 2); TEST_ASSERT_EQ(x_->colidx[3], 2);
        TEST_ASSERT_EQ(mtx_field_complex, x_->a.field);
        TEST_ASSERT_EQ(mtx_single, x_->a.precision);
        TEST_ASSERT_EQ(x_->a.data.complex_single[0][0], 1.0f);
        TEST_ASSERT_EQ(x_->a.data.complex_single[0][1],-1.0f);
        TEST_ASSERT_EQ(x_->a.data.complex_single[1][0], 3.0f);
        TEST_ASSERT_EQ(x_->a.data.complex_single[1][1],-3.0f);
        TEST_ASSERT_EQ(x_->a.data.complex_single[2][0], 4.0f);
        TEST_ASSERT_EQ(x_->a.data.complex_single[2][1],-4.0f);
        TEST_ASSERT_EQ(x_->a.data.complex_single[3][0], 9.0f);
        TEST_ASSERT_EQ(x_->a.data.complex_single[3][1],-9.0f);
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
        err = mtxmatrix_from_mtxfile(&x, mtxmatrix_auto, &mtxfile);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxmatrix_coordinate, x.type);
        const struct mtxmatrix_coordinate * x_ = &x.storage.coordinate;
        TEST_ASSERT_EQ(mtx_unsymmetric, x_->symmetry);
        TEST_ASSERT_EQ(3, x_->num_rows);
        TEST_ASSERT_EQ(3, x_->num_columns);
        TEST_ASSERT_EQ(4, x_->num_nonzeros);
        TEST_ASSERT_EQ(4, x_->size);
        TEST_ASSERT_EQ(x_->rowidx[0], 0); TEST_ASSERT_EQ(x_->colidx[0], 0);
        TEST_ASSERT_EQ(x_->rowidx[1], 0); TEST_ASSERT_EQ(x_->colidx[1], 2);
        TEST_ASSERT_EQ(x_->rowidx[2], 1); TEST_ASSERT_EQ(x_->colidx[2], 0);
        TEST_ASSERT_EQ(x_->rowidx[3], 2); TEST_ASSERT_EQ(x_->colidx[3], 2);
        TEST_ASSERT_EQ(mtx_field_complex, x_->a.field);
        TEST_ASSERT_EQ(mtx_double, x_->a.precision);
        TEST_ASSERT_EQ(x_->a.data.complex_double[0][0], 1.0);
        TEST_ASSERT_EQ(x_->a.data.complex_double[0][1],-1.0);
        TEST_ASSERT_EQ(x_->a.data.complex_double[1][0], 3.0);
        TEST_ASSERT_EQ(x_->a.data.complex_double[1][1],-3.0);
        TEST_ASSERT_EQ(x_->a.data.complex_double[2][0], 4.0);
        TEST_ASSERT_EQ(x_->a.data.complex_double[2][1],-4.0);
        TEST_ASSERT_EQ(x_->a.data.complex_double[3][0], 9.0);
        TEST_ASSERT_EQ(x_->a.data.complex_double[3][1],-9.0);
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
        err = mtxmatrix_from_mtxfile(&x, mtxmatrix_auto, &mtxfile);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxmatrix_coordinate, x.type);
        const struct mtxmatrix_coordinate * x_ = &x.storage.coordinate;
        TEST_ASSERT_EQ(mtx_unsymmetric, x_->symmetry);
        TEST_ASSERT_EQ(3, x_->num_rows);
        TEST_ASSERT_EQ(3, x_->num_columns);
        TEST_ASSERT_EQ(4, x_->num_nonzeros);
        TEST_ASSERT_EQ(4, x_->size);
        TEST_ASSERT_EQ(x_->rowidx[0], 0); TEST_ASSERT_EQ(x_->colidx[0], 0);
        TEST_ASSERT_EQ(x_->rowidx[1], 0); TEST_ASSERT_EQ(x_->colidx[1], 2);
        TEST_ASSERT_EQ(x_->rowidx[2], 1); TEST_ASSERT_EQ(x_->colidx[2], 0);
        TEST_ASSERT_EQ(x_->rowidx[3], 2); TEST_ASSERT_EQ(x_->colidx[3], 2);
        TEST_ASSERT_EQ(mtx_field_integer, x_->a.field);
        TEST_ASSERT_EQ(mtx_single, x_->a.precision);
        TEST_ASSERT_EQ(x_->a.data.integer_single[0], 1);
        TEST_ASSERT_EQ(x_->a.data.integer_single[1], 3);
        TEST_ASSERT_EQ(x_->a.data.integer_single[2], 4);
        TEST_ASSERT_EQ(x_->a.data.integer_single[3], 9);
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
        err = mtxmatrix_from_mtxfile(&x, mtxmatrix_auto, &mtxfile);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxmatrix_coordinate, x.type);
        const struct mtxmatrix_coordinate * x_ = &x.storage.coordinate;
        TEST_ASSERT_EQ(mtx_symmetric, x_->symmetry);
        TEST_ASSERT_EQ(3, x_->num_rows);
        TEST_ASSERT_EQ(3, x_->num_columns);
        TEST_ASSERT_EQ(6, x_->num_nonzeros);
        TEST_ASSERT_EQ(4, x_->size);
        TEST_ASSERT_EQ(x_->rowidx[0], 0); TEST_ASSERT_EQ(x_->colidx[0], 0);
        TEST_ASSERT_EQ(x_->rowidx[1], 0); TEST_ASSERT_EQ(x_->colidx[1], 2);
        TEST_ASSERT_EQ(x_->rowidx[2], 1); TEST_ASSERT_EQ(x_->colidx[2], 0);
        TEST_ASSERT_EQ(x_->rowidx[3], 2); TEST_ASSERT_EQ(x_->colidx[3], 2);
        TEST_ASSERT_EQ(mtx_field_integer, x_->a.field);
        TEST_ASSERT_EQ(mtx_single, x_->a.precision);
        TEST_ASSERT_EQ(x_->a.data.integer_single[0], 1);
        TEST_ASSERT_EQ(x_->a.data.integer_single[1], 3);
        TEST_ASSERT_EQ(x_->a.data.integer_single[2], 4);
        TEST_ASSERT_EQ(x_->a.data.integer_single[3], 9);
        mtxmatrix_free(&x);
        mtxfile_free(&mtxfile);
    }
    {
        int num_rows = 3;
        int num_columns = 3;
        struct mtxfile_matrix_coordinate_integer_single mtxdata[] = {
            {1,2,2}, {1,3,3}, {2,3,6}};
        int64_t size = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile mtxfile;
        err = mtxfile_init_matrix_coordinate_integer_single(
            &mtxfile, mtxfile_skew_symmetric, num_rows, num_columns, size,
            mtxdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxmatrix x;
        err = mtxmatrix_from_mtxfile(&x, mtxmatrix_auto, &mtxfile);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxmatrix_coordinate, x.type);
        const struct mtxmatrix_coordinate * x_ = &x.storage.coordinate;
        TEST_ASSERT_EQ(mtx_skew_symmetric, x_->symmetry);
        TEST_ASSERT_EQ(3, x_->num_rows);
        TEST_ASSERT_EQ(3, x_->num_columns);
        TEST_ASSERT_EQ(6, x_->num_nonzeros);
        TEST_ASSERT_EQ(3, x_->size);
        TEST_ASSERT_EQ(x_->rowidx[0], 0); TEST_ASSERT_EQ(x_->colidx[0], 1);
        TEST_ASSERT_EQ(x_->rowidx[1], 0); TEST_ASSERT_EQ(x_->colidx[1], 2);
        TEST_ASSERT_EQ(x_->rowidx[2], 1); TEST_ASSERT_EQ(x_->colidx[2], 2);
        TEST_ASSERT_EQ(mtx_field_integer, x_->a.field);
        TEST_ASSERT_EQ(mtx_single, x_->a.precision);
        TEST_ASSERT_EQ(x_->a.data.integer_single[0], 2);
        TEST_ASSERT_EQ(x_->a.data.integer_single[1], 3);
        TEST_ASSERT_EQ(x_->a.data.integer_single[2], 6);
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
        err = mtxmatrix_from_mtxfile(&x, mtxmatrix_auto, &mtxfile);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxmatrix_coordinate, x.type);
        const struct mtxmatrix_coordinate * x_ = &x.storage.coordinate;
        TEST_ASSERT_EQ(mtx_unsymmetric, x_->symmetry);
        TEST_ASSERT_EQ(3, x_->num_rows);
        TEST_ASSERT_EQ(3, x_->num_columns);
        TEST_ASSERT_EQ(4, x_->num_nonzeros);
        TEST_ASSERT_EQ(4, x_->size);
        TEST_ASSERT_EQ(x_->rowidx[0], 0); TEST_ASSERT_EQ(x_->colidx[0], 0);
        TEST_ASSERT_EQ(x_->rowidx[1], 0); TEST_ASSERT_EQ(x_->colidx[1], 2);
        TEST_ASSERT_EQ(x_->rowidx[2], 1); TEST_ASSERT_EQ(x_->colidx[2], 0);
        TEST_ASSERT_EQ(x_->rowidx[3], 2); TEST_ASSERT_EQ(x_->colidx[3], 2);
        TEST_ASSERT_EQ(mtx_field_integer, x_->a.field);
        TEST_ASSERT_EQ(mtx_double, x_->a.precision);
        TEST_ASSERT_EQ(x_->a.data.integer_double[0], 1);
        TEST_ASSERT_EQ(x_->a.data.integer_double[1], 3);
        TEST_ASSERT_EQ(x_->a.data.integer_double[2], 4);
        TEST_ASSERT_EQ(x_->a.data.integer_double[3], 9);
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
            &mtxfile, mtxfile_symmetric, num_rows, num_columns, size, mtxdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxmatrix x;
        err = mtxmatrix_from_mtxfile(&x, mtxmatrix_auto, &mtxfile);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxmatrix_coordinate, x.type);
        const struct mtxmatrix_coordinate * x_ = &x.storage.coordinate;
        TEST_ASSERT_EQ(mtx_symmetric, x_->symmetry);
        TEST_ASSERT_EQ(3, x_->num_rows);
        TEST_ASSERT_EQ(3, x_->num_columns);
        TEST_ASSERT_EQ(6, x_->num_nonzeros);
        TEST_ASSERT_EQ(4, x_->size);
        TEST_ASSERT_EQ(x_->rowidx[0], 0); TEST_ASSERT_EQ(x_->colidx[0], 0);
        TEST_ASSERT_EQ(x_->rowidx[1], 0); TEST_ASSERT_EQ(x_->colidx[1], 2);
        TEST_ASSERT_EQ(x_->rowidx[2], 1); TEST_ASSERT_EQ(x_->colidx[2], 0);
        TEST_ASSERT_EQ(x_->rowidx[3], 2); TEST_ASSERT_EQ(x_->colidx[3], 2);
        TEST_ASSERT_EQ(mtx_field_integer, x_->a.field);
        TEST_ASSERT_EQ(mtx_double, x_->a.precision);
        TEST_ASSERT_EQ(x_->a.data.integer_double[0], 1);
        TEST_ASSERT_EQ(x_->a.data.integer_double[1], 3);
        TEST_ASSERT_EQ(x_->a.data.integer_double[2], 4);
        TEST_ASSERT_EQ(x_->a.data.integer_double[3], 9);
        mtxmatrix_free(&x);
        mtxfile_free(&mtxfile);
    }
    {
        int num_rows = 3;
        int num_columns = 3;
        struct mtxfile_matrix_coordinate_integer_double mtxdata[] = {
            {1,2,2}, {1,3,3}, {2,3,6}};
        int64_t size = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile mtxfile;
        err = mtxfile_init_matrix_coordinate_integer_double(
            &mtxfile, mtxfile_skew_symmetric, num_rows, num_columns, size,
            mtxdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxmatrix x;
        err = mtxmatrix_from_mtxfile(&x, mtxmatrix_auto, &mtxfile);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxmatrix_coordinate, x.type);
        const struct mtxmatrix_coordinate * x_ = &x.storage.coordinate;
        TEST_ASSERT_EQ(mtx_skew_symmetric, x_->symmetry);
        TEST_ASSERT_EQ(3, x_->num_rows);
        TEST_ASSERT_EQ(3, x_->num_columns);
        TEST_ASSERT_EQ(6, x_->num_nonzeros);
        TEST_ASSERT_EQ(3, x_->size);
        TEST_ASSERT_EQ(x_->rowidx[0], 0); TEST_ASSERT_EQ(x_->colidx[0], 1);
        TEST_ASSERT_EQ(x_->rowidx[1], 0); TEST_ASSERT_EQ(x_->colidx[1], 2);
        TEST_ASSERT_EQ(x_->rowidx[2], 1); TEST_ASSERT_EQ(x_->colidx[2], 2);
        TEST_ASSERT_EQ(mtx_field_integer, x_->a.field);
        TEST_ASSERT_EQ(mtx_double, x_->a.precision);
        TEST_ASSERT_EQ(x_->a.data.integer_double[0], 2);
        TEST_ASSERT_EQ(x_->a.data.integer_double[1], 3);
        TEST_ASSERT_EQ(x_->a.data.integer_double[2], 6);
        mtxmatrix_free(&x);
        mtxfile_free(&mtxfile);
    }

    {
        int num_rows = 3;
        int num_columns = 3;
        struct mtxfile_matrix_coordinate_pattern mtxdata[] = {
            {1,1}, {1,3}, {2,1}, {3,3}};
        int64_t size = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile mtxfile;
        err = mtxfile_init_matrix_coordinate_pattern(
            &mtxfile, mtxfile_general, num_rows, num_columns, size, mtxdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxmatrix x;
        err = mtxmatrix_from_mtxfile(&x, mtxmatrix_auto, &mtxfile);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxmatrix_coordinate, x.type);
        const struct mtxmatrix_coordinate * x_ = &x.storage.coordinate;
        TEST_ASSERT_EQ(3, x_->num_rows);
        TEST_ASSERT_EQ(3, x_->num_columns);
        TEST_ASSERT_EQ(4, x_->num_nonzeros);
        TEST_ASSERT_EQ(4, x_->size);
        TEST_ASSERT_EQ(x_->rowidx[0], 0); TEST_ASSERT_EQ(x_->colidx[0], 0);
        TEST_ASSERT_EQ(x_->rowidx[1], 0); TEST_ASSERT_EQ(x_->colidx[1], 2);
        TEST_ASSERT_EQ(x_->rowidx[2], 1); TEST_ASSERT_EQ(x_->colidx[2], 0);
        TEST_ASSERT_EQ(x_->rowidx[3], 2); TEST_ASSERT_EQ(x_->colidx[3], 2);
        TEST_ASSERT_EQ(mtx_field_pattern, x_->a.field);
        mtxmatrix_free(&x);
        mtxfile_free(&mtxfile);
    }
    {
        int num_rows = 3;
        int num_columns = 3;
        struct mtxfile_matrix_coordinate_pattern mtxdata[] = {
            {1,1}, {1,3}, {2,1}, {3,3}};
        int64_t size = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile mtxfile;
        err = mtxfile_init_matrix_coordinate_pattern(
            &mtxfile, mtxfile_symmetric, num_rows, num_columns, size, mtxdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxmatrix x;
        err = mtxmatrix_from_mtxfile(&x, mtxmatrix_auto, &mtxfile);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxmatrix_coordinate, x.type);
        const struct mtxmatrix_coordinate * x_ = &x.storage.coordinate;
        TEST_ASSERT_EQ(3, x_->num_rows);
        TEST_ASSERT_EQ(3, x_->num_columns);
        TEST_ASSERT_EQ(6, x_->num_nonzeros);
        TEST_ASSERT_EQ(4, x_->size);
        TEST_ASSERT_EQ(x_->rowidx[0], 0); TEST_ASSERT_EQ(x_->colidx[0], 0);
        TEST_ASSERT_EQ(x_->rowidx[1], 0); TEST_ASSERT_EQ(x_->colidx[1], 2);
        TEST_ASSERT_EQ(x_->rowidx[2], 1); TEST_ASSERT_EQ(x_->colidx[2], 0);
        TEST_ASSERT_EQ(x_->rowidx[3], 2); TEST_ASSERT_EQ(x_->colidx[3], 2);
        TEST_ASSERT_EQ(mtx_field_pattern, x_->a.field);
        mtxmatrix_free(&x);
        mtxfile_free(&mtxfile);
    }
    return TEST_SUCCESS;
}

/**
 * ‘test_mtxmatrix_coordinate_to_mtxfile()’ tests converting matrices
 * to Matrix Market files.
 */
int test_mtxmatrix_coordinate_to_mtxfile(void)
{
    int err;

    {
        int num_rows = 3;
        int num_columns = 3;
        int nnz = 4;
        int rowidx[] = {1, 1, 2, 3};
        int colidx[] = {1, 3, 1, 3};
        float Adata[] = {1.0f, 3.0f, 4.0f, 9.0f};
        struct mtxmatrix A;
        err = mtxmatrix_init_coordinate_real_single(
            &A, mtx_unsymmetric, num_rows, num_columns, nnz, rowidx, colidx, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxfile mtxfile;
        err = mtxmatrix_to_mtxfile(&mtxfile, &A, mtxfile_coordinate);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
        int colidx[] = {1, 3, 3, 3};
        float Adata[] = {1.0f, 3.0f, 6.0f, 9.0f};
        struct mtxmatrix A;
        err = mtxmatrix_init_coordinate_real_single(
            &A, mtx_symmetric, num_rows, num_columns, nnz, rowidx, colidx, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxfile mtxfile;
        err = mtxmatrix_to_mtxfile(&mtxfile, &A, mtxfile_coordinate);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_matrix, mtxfile.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxfile.header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtxfile.header.field);
        TEST_ASSERT_EQ(mtxfile_symmetric, mtxfile.header.symmetry);
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
        int nnz = 3;
        int rowidx[] = {1, 1, 2};
        int colidx[] = {2, 3, 3};
        float Adata[] = {2.0f, 3.0f, 6.0f};
        struct mtxmatrix A;
        err = mtxmatrix_init_coordinate_real_single(
            &A, mtx_skew_symmetric, num_rows, num_columns, nnz, rowidx, colidx, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxfile mtxfile;
        err = mtxmatrix_to_mtxfile(&mtxfile, &A, mtxfile_coordinate);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_matrix, mtxfile.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxfile.header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtxfile.header.field);
        TEST_ASSERT_EQ(mtxfile_skew_symmetric, mtxfile.header.symmetry);
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
            &A, mtx_unsymmetric, num_rows, num_columns, nnz, rowidx, colidx, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxfile mtxfile;
        err = mtxmatrix_to_mtxfile(&mtxfile, &A, mtxfile_coordinate);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
            &A, mtx_unsymmetric, num_rows, num_columns, nnz, rowidx, colidx, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxfile mtxfile;
        err = mtxmatrix_to_mtxfile(&mtxfile, &A, mtxfile_coordinate);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
        int colidx[] = {2, 3, 3, 3};
        float Adata[][2] = {{2.0f,-2.0f}, {3.0f,-3.0f}, {6.0f,-6.0f}, {9.0f,-9.0f}};
        struct mtxmatrix A;
        err = mtxmatrix_init_coordinate_complex_single(
            &A, mtx_hermitian, num_rows, num_columns, nnz, rowidx, colidx, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxfile mtxfile;
        err = mtxmatrix_to_mtxfile(&mtxfile, &A, mtxfile_coordinate);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_matrix, mtxfile.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxfile.header.format);
        TEST_ASSERT_EQ(mtxfile_complex, mtxfile.header.field);
        TEST_ASSERT_EQ(mtxfile_hermitian, mtxfile.header.symmetry);
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
            &A, mtx_unsymmetric, num_rows, num_columns, nnz, rowidx, colidx, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxfile mtxfile;
        err = mtxmatrix_to_mtxfile(&mtxfile, &A, mtxfile_coordinate);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
            &A, mtx_unsymmetric, num_rows, num_columns, nnz, rowidx, colidx, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxfile mtxfile;
        err = mtxmatrix_to_mtxfile(&mtxfile, &A, mtxfile_coordinate);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
            &A, mtx_unsymmetric, num_rows, num_columns, nnz, rowidx, colidx, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxfile mtxfile;
        err = mtxmatrix_to_mtxfile(&mtxfile, &A, mtxfile_coordinate);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
            &A, mtx_unsymmetric, num_rows, num_columns, nnz, rowidx, colidx);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxfile mtxfile;
        err = mtxmatrix_to_mtxfile(&mtxfile, &A, mtxfile_coordinate);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
 * ‘test_mtxmatrix_coordinate_partition()’ tests partitioning matrices
 * in coordinate format.
 */
int test_mtxmatrix_coordinate_partition(void)
{
    int err;

    {
        struct mtxmatrix src;
        int num_rows = 3;
        int num_columns = 3;
        int nnz = 9;
        int rowidx[] = {0, 0, 0, 1, 1, 1, 2, 2, 2};
        int colidx[] = {0, 1, 2, 0, 1, 2, 0, 1, 2};
        err = mtxmatrix_init_coordinate_pattern(
            &src, mtx_unsymmetric, num_rows, num_columns, nnz, rowidx, colidx);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        int num_row_parts = 2;
        enum mtxpartitioning rowpart_type = mtx_block;
        struct mtxpartition rowpart;
        err = mtxpartition_init(
            &rowpart, rowpart_type, num_rows, num_row_parts, NULL, 0, NULL, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        const int num_parts = num_row_parts;
        struct mtxmatrix dsts[num_parts];
        err = mtxmatrix_partition(dsts, &src, &rowpart, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        {
            TEST_ASSERT_EQ(mtxmatrix_coordinate, dsts[0].type);
            const struct mtxmatrix_coordinate * A = &dsts[0].storage.coordinate;
            TEST_ASSERT_EQ(2, A->num_rows);
            TEST_ASSERT_EQ(3, A->num_columns);
            TEST_ASSERT_EQ(6, A->num_nonzeros);
            TEST_ASSERT_EQ(A->rowidx[0], 0); TEST_ASSERT_EQ(A->colidx[0], 0);
            TEST_ASSERT_EQ(A->rowidx[1], 0); TEST_ASSERT_EQ(A->colidx[1], 1);
            TEST_ASSERT_EQ(A->rowidx[2], 0); TEST_ASSERT_EQ(A->colidx[2], 2);
            TEST_ASSERT_EQ(A->rowidx[3], 1); TEST_ASSERT_EQ(A->colidx[3], 0);
            TEST_ASSERT_EQ(A->rowidx[4], 1); TEST_ASSERT_EQ(A->colidx[4], 1);
            TEST_ASSERT_EQ(A->rowidx[5], 1); TEST_ASSERT_EQ(A->colidx[5], 2);
            TEST_ASSERT_EQ(mtx_field_pattern, A->a.field);
            mtxmatrix_free(&dsts[0]);
        }
        {
            TEST_ASSERT_EQ(mtxmatrix_coordinate, dsts[1].type);
            const struct mtxmatrix_coordinate * A = &dsts[1].storage.coordinate;
            TEST_ASSERT_EQ(1, A->num_rows);
            TEST_ASSERT_EQ(3, A->num_columns);
            TEST_ASSERT_EQ(3, A->num_nonzeros);
            TEST_ASSERT_EQ(A->rowidx[0], 0); TEST_ASSERT_EQ(A->colidx[0], 0);
            TEST_ASSERT_EQ(A->rowidx[1], 0); TEST_ASSERT_EQ(A->colidx[1], 1);
            TEST_ASSERT_EQ(A->rowidx[2], 0); TEST_ASSERT_EQ(A->colidx[2], 2);
            TEST_ASSERT_EQ(mtx_field_pattern, A->a.field);
            mtxmatrix_free(&dsts[1]);
        }
        mtxpartition_free(&rowpart);
        mtxmatrix_free(&src);
    }
    return TEST_SUCCESS;
}

/**
 * ‘test_mtxmatrix_coordinate_join()’ tests joining matrices in
 * coordinate format.
 */
int test_mtxmatrix_coordinate_join(void)
{
    int err;

    {
        struct mtxmatrix srcs[2];
        int num_rows[] = {2, 1};
        int num_columns[] = {3, 3};
        int nnz[] = {6, 3};
        int rowidx0[] = {0, 0, 0, 1, 1, 1};
        int colidx0[] = {0, 1, 2, 0, 1, 2};
        int rowidx1[] = {0, 0, 0};
        int colidx1[] = {0, 1, 2};
        err = mtxmatrix_init_coordinate_pattern(
            &srcs[0], mtx_unsymmetric, num_rows[0], num_columns[0], nnz[0], rowidx0, colidx0);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxmatrix_init_coordinate_pattern(
            &srcs[1], mtx_unsymmetric, num_rows[1], num_columns[1], nnz[1], rowidx1, colidx1);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        int num_row_parts = 2;
        enum mtxpartitioning rowpart_type = mtx_block;
        struct mtxpartition rowpart;
        err = mtxpartition_init(
            &rowpart, rowpart_type, num_rows[0]+num_rows[1], num_row_parts, NULL, 0, NULL, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        struct mtxmatrix dst;
        err = mtxmatrix_join(&dst, srcs, &rowpart, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxmatrix_coordinate, dst.type);
        const struct mtxmatrix_coordinate * A = &dst.storage.coordinate;
        TEST_ASSERT_EQ(3, A->num_rows);
        TEST_ASSERT_EQ(3, A->num_columns);
        TEST_ASSERT_EQ(9, A->num_nonzeros);
        TEST_ASSERT_EQ(A->rowidx[0], 0); TEST_ASSERT_EQ(A->colidx[0], 0);
        TEST_ASSERT_EQ(A->rowidx[1], 0); TEST_ASSERT_EQ(A->colidx[1], 1);
        TEST_ASSERT_EQ(A->rowidx[2], 0); TEST_ASSERT_EQ(A->colidx[2], 2);
        TEST_ASSERT_EQ(A->rowidx[3], 1); TEST_ASSERT_EQ(A->colidx[3], 0);
        TEST_ASSERT_EQ(A->rowidx[4], 1); TEST_ASSERT_EQ(A->colidx[4], 1);
        TEST_ASSERT_EQ(A->rowidx[5], 1); TEST_ASSERT_EQ(A->colidx[5], 2);
        TEST_ASSERT_EQ(A->rowidx[6], 2); TEST_ASSERT_EQ(A->colidx[6], 0);
        TEST_ASSERT_EQ(A->rowidx[7], 2); TEST_ASSERT_EQ(A->colidx[7], 1);
        TEST_ASSERT_EQ(A->rowidx[8], 2); TEST_ASSERT_EQ(A->colidx[8], 2);
        TEST_ASSERT_EQ(mtx_field_pattern, A->a.field);
        mtxmatrix_free(&dst);
        mtxpartition_free(&rowpart);
        mtxmatrix_free(&srcs[1]);
        mtxmatrix_free(&srcs[0]);
    }
    return TEST_SUCCESS;
}

/**
 * ‘test_mtxmatrix_coordinate_gemv()’ tests computing matrix-vector
 * products for matrices in coordinate format.
 */
int test_mtxmatrix_coordinate_gemv(void)
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
        err = mtxmatrix_init_coordinate_real_single(
            &A, mtx_unsymmetric, num_rows, num_columns, num_nonzeros, Arowidx, Acolidx, Adata);
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
        err = mtxmatrix_init_coordinate_real_single(
            &A, mtx_symmetric, num_rows, num_columns, num_nonzeros, Arowidx, Acolidx, Adata);
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
        err = mtxmatrix_init_coordinate_real_double(
            &A, mtx_unsymmetric, num_rows, num_columns, num_nonzeros, Arowidx, Acolidx, Adata);
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
        err = mtxmatrix_init_coordinate_real_double(
            &A, mtx_symmetric, num_rows, num_columns, num_nonzeros, Arowidx, Acolidx, Adata);
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
        err = mtxmatrix_init_coordinate_complex_single(
            &A, mtx_unsymmetric, num_rows, num_columns, num_nonzeros, Arowidx, Acolidx, Adata);
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
        err = mtxmatrix_init_coordinate_complex_single(
            &A, mtx_symmetric, num_rows, num_columns, num_nonzeros, Arowidx, Acolidx, Adata);
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
        err = mtxmatrix_init_coordinate_complex_single(
            &A, mtx_hermitian, num_rows, num_columns, num_nonzeros, Arowidx, Acolidx, Adata);
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
        err = mtxmatrix_init_coordinate_complex_double(
            &A, mtx_unsymmetric, num_rows, num_columns, num_nonzeros, Arowidx, Acolidx, Adata);
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
        err = mtxmatrix_init_coordinate_complex_double(
            &A, mtx_symmetric, num_rows, num_columns, num_nonzeros, Arowidx, Acolidx, Adata);
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
        err = mtxmatrix_init_coordinate_complex_double(
            &A, mtx_hermitian, num_rows, num_columns, num_nonzeros, Arowidx, Acolidx, Adata);
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
        err = mtxmatrix_init_coordinate_integer_single(
            &A, mtx_unsymmetric, num_rows, num_columns, num_nonzeros, Arowidx, Acolidx, Adata);
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
        err = mtxmatrix_init_coordinate_integer_single(
            &A, mtx_symmetric, num_rows, num_columns, num_nonzeros, Arowidx, Acolidx, Adata);
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
        err = mtxmatrix_init_coordinate_integer_double(
            &A, mtx_unsymmetric, num_rows, num_columns, num_nonzeros, Arowidx, Acolidx, Adata);
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
        err = mtxmatrix_init_coordinate_integer_double(
            &A, mtx_symmetric, num_rows, num_columns, num_nonzeros, Arowidx, Acolidx, Adata);
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
    TEST_SUITE_BEGIN("Running tests for coordinate matrices\n");
    TEST_RUN(test_mtxmatrix_coordinate_from_mtxfile);
    TEST_RUN(test_mtxmatrix_coordinate_to_mtxfile);
    TEST_RUN(test_mtxmatrix_coordinate_partition);
    TEST_RUN(test_mtxmatrix_coordinate_join);
    TEST_RUN(test_mtxmatrix_coordinate_gemv);
    TEST_SUITE_END();
    return (TEST_SUITE_STATUS == TEST_SUCCESS) ?
        EXIT_SUCCESS : EXIT_FAILURE;
}
