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
 * Last modified: 2022-03-19
 *
 * Unit tests for sparse matrices in CSR format.
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

/*
 * TODO: The following tests need to be updated to handle symmetric
 * matrices:
 *
 * 1. test_mtxmatrix_csr_from_mtxfile
 * 2. test_mtxmatrix_csr_to_mtxfile
 * 3. test_mtxmatrix_csr_partition
 * 4. test_mtxmatrix_csr_join
 *
 * In addition, test_mtxmatrix_csr_gemv needs to be updated to
 * incorporate all the cases from test_mtxmatrix_array_gemv in
 * test_mtxmatrix_array.c.
 */

/**
 * ‘test_mtxmatrix_csr_from_mtxfile()’ tests converting Matrix Market
 * files to matrices.
 */
int test_mtxmatrix_csr_from_mtxfile(void)
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
        err = mtxmatrix_from_mtxfile(&x, &mtxfile, mtxmatrix_csr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxmatrix_csr, x.type);
        const struct mtxmatrix_csr * x_ = &x.storage.csr;
        TEST_ASSERT_EQ(mtx_field_real, x_->field);
        TEST_ASSERT_EQ(mtx_single, x_->precision);
        TEST_ASSERT_EQ(mtx_unsymmetric, x_->symmetry);
        TEST_ASSERT_EQ(3, x_->num_rows);
        TEST_ASSERT_EQ(3, x_->num_columns);
        TEST_ASSERT_EQ(9, x_->num_entries);
        TEST_ASSERT_EQ(4, x_->num_nonzeros);
        TEST_ASSERT_EQ(4, x_->size);
        TEST_ASSERT_EQ(x_->rowptr[0], 0);
        TEST_ASSERT_EQ(x_->rowptr[1], 2);
        TEST_ASSERT_EQ(x_->rowptr[2], 3);
        TEST_ASSERT_EQ(x_->rowptr[3], 4);
        TEST_ASSERT_EQ(x_->colidx[0], 0);
        TEST_ASSERT_EQ(x_->colidx[1], 2);
        TEST_ASSERT_EQ(x_->colidx[2], 0);
        TEST_ASSERT_EQ(x_->colidx[3], 2);
        TEST_ASSERT_EQ(x_->data.real_single[0], 1.0f);
        TEST_ASSERT_EQ(x_->data.real_single[1], 3.0f);
        TEST_ASSERT_EQ(x_->data.real_single[2], 4.0f);
        TEST_ASSERT_EQ(x_->data.real_single[3], 9.0f);
        mtxmatrix_free(&x);
        mtxfile_free(&mtxfile);
    }
    /* { */
    /*     int num_rows = 3; */
    /*     int num_columns = 3; */
    /*     struct mtxfile_matrix_coordinate_real_single mtxdata[] = { */
    /*         {1,1,1.0f}, {1,3,3.0f}, {2,1,4.0f}, {3,3,9.0f}}; */
    /*     int64_t size = sizeof(mtxdata) / sizeof(*mtxdata); */
    /*     struct mtxfile mtxfile; */
    /*     err = mtxfile_init_matrix_coordinate_real_single( */
    /*         &mtxfile, mtxfile_symmetric, num_rows, num_columns, size, mtxdata); */
    /*     TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err)); */
    /*     struct mtxmatrix x; */
    /*     err = mtxmatrix_from_mtxfile(&x, &mtxfile, mtxmatrix_auto); */
    /*     TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err)); */
    /*     TEST_ASSERT_EQ(mtxmatrix_coordinate, x.type); */
    /*     const struct mtxmatrix_coordinate * x_ = &x.storage.coordinate; */
    /*     TEST_ASSERT_EQ(mtx_field_real, x_->field); */
    /*     TEST_ASSERT_EQ(mtx_single, x_->precision); */
    /*     TEST_ASSERT_EQ(mtx_symmetric, x_->symmetry); */
    /*     TEST_ASSERT_EQ(3, x_->num_rows); */
    /*     TEST_ASSERT_EQ(3, x_->num_columns); */
    /*     TEST_ASSERT_EQ(6, x_->num_nonzeros); */
    /*     TEST_ASSERT_EQ(4, x_->size); */
    /*     TEST_ASSERT_EQ(x_->rowidx[0], 0); TEST_ASSERT_EQ(x_->colidx[0], 0); */
    /*     TEST_ASSERT_EQ(x_->rowidx[1], 0); TEST_ASSERT_EQ(x_->colidx[1], 2); */
    /*     TEST_ASSERT_EQ(x_->rowidx[2], 1); TEST_ASSERT_EQ(x_->colidx[2], 0); */
    /*     TEST_ASSERT_EQ(x_->rowidx[3], 2); TEST_ASSERT_EQ(x_->colidx[3], 2); */
    /*     TEST_ASSERT_EQ(x_->data.real_single[0], 1.0f); */
    /*     TEST_ASSERT_EQ(x_->data.real_single[1], 3.0f); */
    /*     TEST_ASSERT_EQ(x_->data.real_single[2], 4.0f); */
    /*     TEST_ASSERT_EQ(x_->data.real_single[3], 9.0f); */
    /*     mtxmatrix_free(&x); */
    /*     mtxfile_free(&mtxfile); */
    /* } */
    /* { */
    /*     int num_rows = 3; */
    /*     int num_columns = 3; */
    /*     struct mtxfile_matrix_coordinate_real_single mtxdata[] = { */
    /*         {1,3,3.0f}, {2,1,4.0f}, {3,2,8.0f}}; */
    /*     int64_t size = sizeof(mtxdata) / sizeof(*mtxdata); */
    /*     struct mtxfile mtxfile; */
    /*     err = mtxfile_init_matrix_coordinate_real_single( */
    /*         &mtxfile, mtxfile_skew_symmetric, num_rows, num_columns, size, */
    /*         mtxdata); */
    /*     TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err)); */
    /*     struct mtxmatrix x; */
    /*     err = mtxmatrix_from_mtxfile(&x, &mtxfile, mtxmatrix_auto); */
    /*     TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err)); */
    /*     TEST_ASSERT_EQ(mtxmatrix_coordinate, x.type); */
    /*     const struct mtxmatrix_coordinate * x_ = &x.storage.coordinate; */
    /*     TEST_ASSERT_EQ(mtx_field_real, x_->field); */
    /*     TEST_ASSERT_EQ(mtx_single, x_->precision); */
    /*     TEST_ASSERT_EQ(mtx_skew_symmetric, x_->symmetry); */
    /*     TEST_ASSERT_EQ(3, x_->num_rows); */
    /*     TEST_ASSERT_EQ(3, x_->num_columns); */
    /*     TEST_ASSERT_EQ(6, x_->num_nonzeros); */
    /*     TEST_ASSERT_EQ(3, x_->size); */
    /*     TEST_ASSERT_EQ(x_->rowidx[0], 0); TEST_ASSERT_EQ(x_->colidx[0], 2); */
    /*     TEST_ASSERT_EQ(x_->rowidx[1], 1); TEST_ASSERT_EQ(x_->colidx[1], 0); */
    /*     TEST_ASSERT_EQ(x_->rowidx[2], 2); TEST_ASSERT_EQ(x_->colidx[2], 1); */
    /*     TEST_ASSERT_EQ(x_->data.real_single[0], 3.0f); */
    /*     TEST_ASSERT_EQ(x_->data.real_single[1], 4.0f); */
    /*     TEST_ASSERT_EQ(x_->data.real_single[2], 8.0f); */
    /*     mtxmatrix_free(&x); */
    /*     mtxfile_free(&mtxfile); */
    /* } */

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
        err = mtxmatrix_from_mtxfile(&x, &mtxfile, mtxmatrix_auto);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxmatrix_coordinate, x.type);
        const struct mtxmatrix_coordinate * x_ = &x.storage.coordinate;
        TEST_ASSERT_EQ(mtx_field_real, x_->field);
        TEST_ASSERT_EQ(mtx_double, x_->precision);
        TEST_ASSERT_EQ(mtx_unsymmetric, x_->symmetry);
        TEST_ASSERT_EQ(3, x_->num_rows);
        TEST_ASSERT_EQ(3, x_->num_columns);
        TEST_ASSERT_EQ(4, x_->num_nonzeros);
        TEST_ASSERT_EQ(4, x_->size);
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
    /* { */
    /*     int num_rows = 3; */
    /*     int num_columns = 3; */
    /*     struct mtxfile_matrix_coordinate_real_double mtxdata[] = { */
    /*         {1,1,1.0}, {1,3,3.0}, {2,1,4.0}, {3,3,9.0}}; */
    /*     int64_t size = sizeof(mtxdata) / sizeof(*mtxdata); */
    /*     struct mtxfile mtxfile; */
    /*     err = mtxfile_init_matrix_coordinate_real_double( */
    /*         &mtxfile, mtxfile_symmetric, num_rows, num_columns, size, mtxdata); */
    /*     TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err)); */
    /*     struct mtxmatrix x; */
    /*     err = mtxmatrix_from_mtxfile(&x, &mtxfile, mtxmatrix_auto); */
    /*     TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err)); */
    /*     TEST_ASSERT_EQ(mtxmatrix_coordinate, x.type); */
    /*     const struct mtxmatrix_coordinate * x_ = &x.storage.coordinate; */
    /*     TEST_ASSERT_EQ(mtx_field_real, x_->field); */
    /*     TEST_ASSERT_EQ(mtx_double, x_->precision); */
    /*     TEST_ASSERT_EQ(mtx_symmetric, x_->symmetry); */
    /*     TEST_ASSERT_EQ(3, x_->num_rows); */
    /*     TEST_ASSERT_EQ(3, x_->num_columns); */
    /*     TEST_ASSERT_EQ(6, x_->num_nonzeros); */
    /*     TEST_ASSERT_EQ(4, x_->size); */
    /*     TEST_ASSERT_EQ(x_->rowidx[0], 0); TEST_ASSERT_EQ(x_->colidx[0], 0); */
    /*     TEST_ASSERT_EQ(x_->rowidx[1], 0); TEST_ASSERT_EQ(x_->colidx[1], 2); */
    /*     TEST_ASSERT_EQ(x_->rowidx[2], 1); TEST_ASSERT_EQ(x_->colidx[2], 0); */
    /*     TEST_ASSERT_EQ(x_->rowidx[3], 2); TEST_ASSERT_EQ(x_->colidx[3], 2); */
    /*     TEST_ASSERT_EQ(x_->data.real_double[0], 1.0); */
    /*     TEST_ASSERT_EQ(x_->data.real_double[1], 3.0); */
    /*     TEST_ASSERT_EQ(x_->data.real_double[2], 4.0); */
    /*     TEST_ASSERT_EQ(x_->data.real_double[3], 9.0); */
    /*     mtxmatrix_free(&x); */
    /*     mtxfile_free(&mtxfile); */
    /* } */
    /* { */
    /*     int num_rows = 3; */
    /*     int num_columns = 3; */
    /*     struct mtxfile_matrix_coordinate_real_double mtxdata[] = { */
    /*         {1,3,3.0}, {2,1,4.0}, {3,2,8.0}}; */
    /*     int64_t size = sizeof(mtxdata) / sizeof(*mtxdata); */
    /*     struct mtxfile mtxfile; */
    /*     err = mtxfile_init_matrix_coordinate_real_double( */
    /*         &mtxfile, mtxfile_skew_symmetric, num_rows, num_columns, size, */
    /*         mtxdata); */
    /*     TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err)); */
    /*     struct mtxmatrix x; */
    /*     err = mtxmatrix_from_mtxfile(&x, &mtxfile, mtxmatrix_auto); */
    /*     TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err)); */
    /*     TEST_ASSERT_EQ(mtxmatrix_coordinate, x.type); */
    /*     const struct mtxmatrix_coordinate * x_ = &x.storage.coordinate; */
    /*     TEST_ASSERT_EQ(mtx_field_real, x_->field); */
    /*     TEST_ASSERT_EQ(mtx_double, x_->precision); */
    /*     TEST_ASSERT_EQ(mtx_skew_symmetric, x_->symmetry); */
    /*     TEST_ASSERT_EQ(3, x_->num_rows); */
    /*     TEST_ASSERT_EQ(3, x_->num_columns); */
    /*     TEST_ASSERT_EQ(6, x_->num_nonzeros); */
    /*     TEST_ASSERT_EQ(3, x_->size); */
    /*     TEST_ASSERT_EQ(x_->rowidx[0], 0); TEST_ASSERT_EQ(x_->colidx[0], 2); */
    /*     TEST_ASSERT_EQ(x_->rowidx[1], 1); TEST_ASSERT_EQ(x_->colidx[1], 0); */
    /*     TEST_ASSERT_EQ(x_->rowidx[2], 2); TEST_ASSERT_EQ(x_->colidx[2], 1); */
    /*     TEST_ASSERT_EQ(x_->data.real_double[0], 3.0); */
    /*     TEST_ASSERT_EQ(x_->data.real_double[1], 4.0); */
    /*     TEST_ASSERT_EQ(x_->data.real_double[2], 8.0); */
    /*     mtxmatrix_free(&x); */
    /*     mtxfile_free(&mtxfile); */
    /* } */

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
        err = mtxmatrix_from_mtxfile(&x, &mtxfile, mtxmatrix_auto);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxmatrix_coordinate, x.type);
        const struct mtxmatrix_coordinate * x_ = &x.storage.coordinate;
        TEST_ASSERT_EQ(mtx_field_complex, x_->field);
        TEST_ASSERT_EQ(mtx_single, x_->precision);
        TEST_ASSERT_EQ(mtx_unsymmetric, x_->symmetry);
        TEST_ASSERT_EQ(3, x_->num_rows);
        TEST_ASSERT_EQ(3, x_->num_columns);
        TEST_ASSERT_EQ(4, x_->num_nonzeros);
        TEST_ASSERT_EQ(4, x_->size);
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
    /* { */
    /*     int num_rows = 3; */
    /*     int num_columns = 3; */
    /*     struct mtxfile_matrix_coordinate_complex_single mtxdata[] = { */
    /*         {1,1,{1.0f,-1.0f}}, {1,3,{3.0f,-3.0f}}, */
    /*         {2,1,{4.0f,-4.0f}}, {3,3,{9.0f,-9.0f}}}; */
    /*     int64_t size = sizeof(mtxdata) / sizeof(*mtxdata); */
    /*     struct mtxfile mtxfile; */
    /*     err = mtxfile_init_matrix_coordinate_complex_single( */
    /*         &mtxfile, mtxfile_symmetric, num_rows, num_columns, size, mtxdata); */
    /*     TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err)); */
    /*     struct mtxmatrix x; */
    /*     err = mtxmatrix_from_mtxfile(&x, &mtxfile, mtxmatrix_auto); */
    /*     TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err)); */
    /*     TEST_ASSERT_EQ(mtxmatrix_coordinate, x.type); */
    /*     const struct mtxmatrix_coordinate * x_ = &x.storage.coordinate; */
    /*     TEST_ASSERT_EQ(mtx_field_complex, x_->field); */
    /*     TEST_ASSERT_EQ(mtx_single, x_->precision); */
    /*     TEST_ASSERT_EQ(mtx_symmetric, x_->symmetry); */
    /*     TEST_ASSERT_EQ(3, x_->num_rows); */
    /*     TEST_ASSERT_EQ(3, x_->num_columns); */
    /*     TEST_ASSERT_EQ(6, x_->num_nonzeros); */
    /*     TEST_ASSERT_EQ(4, x_->size); */
    /*     TEST_ASSERT_EQ(x_->rowidx[0], 0); TEST_ASSERT_EQ(x_->colidx[0], 0); */
    /*     TEST_ASSERT_EQ(x_->rowidx[1], 0); TEST_ASSERT_EQ(x_->colidx[1], 2); */
    /*     TEST_ASSERT_EQ(x_->rowidx[2], 1); TEST_ASSERT_EQ(x_->colidx[2], 0); */
    /*     TEST_ASSERT_EQ(x_->rowidx[3], 2); TEST_ASSERT_EQ(x_->colidx[3], 2); */
    /*     TEST_ASSERT_EQ(x_->data.complex_single[0][0], 1.0f); */
    /*     TEST_ASSERT_EQ(x_->data.complex_single[0][1],-1.0f); */
    /*     TEST_ASSERT_EQ(x_->data.complex_single[1][0], 3.0f); */
    /*     TEST_ASSERT_EQ(x_->data.complex_single[1][1],-3.0f); */
    /*     TEST_ASSERT_EQ(x_->data.complex_single[2][0], 4.0f); */
    /*     TEST_ASSERT_EQ(x_->data.complex_single[2][1],-4.0f); */
    /*     TEST_ASSERT_EQ(x_->data.complex_single[3][0], 9.0f); */
    /*     TEST_ASSERT_EQ(x_->data.complex_single[3][1],-9.0f); */
    /*     mtxmatrix_free(&x); */
    /*     mtxfile_free(&mtxfile); */
    /* } */
    /* { */
    /*     int num_rows = 3; */
    /*     int num_columns = 3; */
    /*     struct mtxfile_matrix_coordinate_complex_single mtxdata[] = { */
    /*         {1,2,{2.0f,-2.0f}}, {1,3,{3.0f,-3.0f}}, {2,3,{6.0f,-6.0f}}}; */
    /*     int64_t size = sizeof(mtxdata) / sizeof(*mtxdata); */
    /*     struct mtxfile mtxfile; */
    /*     err = mtxfile_init_matrix_coordinate_complex_single( */
    /*         &mtxfile, mtxfile_skew_symmetric, num_rows, num_columns, size, */
    /*         mtxdata); */
    /*     TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err)); */
    /*     struct mtxmatrix x; */
    /*     err = mtxmatrix_from_mtxfile(&x, &mtxfile, mtxmatrix_auto); */
    /*     TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err)); */
    /*     TEST_ASSERT_EQ(mtxmatrix_coordinate, x.type); */
    /*     const struct mtxmatrix_coordinate * x_ = &x.storage.coordinate; */
    /*     TEST_ASSERT_EQ(mtx_field_complex, x_->field); */
    /*     TEST_ASSERT_EQ(mtx_single, x_->precision); */
    /*     TEST_ASSERT_EQ(mtx_skew_symmetric, x_->symmetry); */
    /*     TEST_ASSERT_EQ(3, x_->num_rows); */
    /*     TEST_ASSERT_EQ(3, x_->num_columns); */
    /*     TEST_ASSERT_EQ(6, x_->num_nonzeros); */
    /*     TEST_ASSERT_EQ(3, x_->size); */
    /*     TEST_ASSERT_EQ(x_->rowidx[0], 0); TEST_ASSERT_EQ(x_->colidx[0], 1); */
    /*     TEST_ASSERT_EQ(x_->rowidx[1], 0); TEST_ASSERT_EQ(x_->colidx[1], 2); */
    /*     TEST_ASSERT_EQ(x_->rowidx[2], 1); TEST_ASSERT_EQ(x_->colidx[2], 2); */
    /*     TEST_ASSERT_EQ(x_->data.complex_single[0][0], 2.0f); */
    /*     TEST_ASSERT_EQ(x_->data.complex_single[0][1],-2.0f); */
    /*     TEST_ASSERT_EQ(x_->data.complex_single[1][0], 3.0f); */
    /*     TEST_ASSERT_EQ(x_->data.complex_single[1][1],-3.0f); */
    /*     TEST_ASSERT_EQ(x_->data.complex_single[2][0], 6.0f); */
    /*     TEST_ASSERT_EQ(x_->data.complex_single[2][1],-6.0f); */
    /*     mtxmatrix_free(&x); */
    /*     mtxfile_free(&mtxfile); */
    /* } */
    /* { */
    /*     int num_rows = 3; */
    /*     int num_columns = 3; */
    /*     struct mtxfile_matrix_coordinate_complex_single mtxdata[] = { */
    /*         {1,1,{1.0f,-1.0f}}, {1,3,{3.0f,-3.0f}}, */
    /*         {2,1,{4.0f,-4.0f}}, {3,3,{9.0f,-9.0f}}}; */
    /*     int64_t size = sizeof(mtxdata) / sizeof(*mtxdata); */
    /*     struct mtxfile mtxfile; */
    /*     err = mtxfile_init_matrix_coordinate_complex_single( */
    /*         &mtxfile, mtxfile_hermitian, num_rows, num_columns, size, mtxdata); */
    /*     TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err)); */
    /*     struct mtxmatrix x; */
    /*     err = mtxmatrix_from_mtxfile(&x, &mtxfile, mtxmatrix_auto); */
    /*     TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err)); */
    /*     TEST_ASSERT_EQ(mtxmatrix_coordinate, x.type); */
    /*     const struct mtxmatrix_coordinate * x_ = &x.storage.coordinate; */
    /*     TEST_ASSERT_EQ(mtx_field_complex, x_->field); */
    /*     TEST_ASSERT_EQ(mtx_single, x_->precision); */
    /*     TEST_ASSERT_EQ(mtx_hermitian, x_->symmetry); */
    /*     TEST_ASSERT_EQ(3, x_->num_rows); */
    /*     TEST_ASSERT_EQ(3, x_->num_columns); */
    /*     TEST_ASSERT_EQ(6, x_->num_nonzeros); */
    /*     TEST_ASSERT_EQ(4, x_->size); */
    /*     TEST_ASSERT_EQ(x_->rowidx[0], 0); TEST_ASSERT_EQ(x_->colidx[0], 0); */
    /*     TEST_ASSERT_EQ(x_->rowidx[1], 0); TEST_ASSERT_EQ(x_->colidx[1], 2); */
    /*     TEST_ASSERT_EQ(x_->rowidx[2], 1); TEST_ASSERT_EQ(x_->colidx[2], 0); */
    /*     TEST_ASSERT_EQ(x_->rowidx[3], 2); TEST_ASSERT_EQ(x_->colidx[3], 2); */
    /*     TEST_ASSERT_EQ(x_->data.complex_single[0][0], 1.0f); */
    /*     TEST_ASSERT_EQ(x_->data.complex_single[0][1],-1.0f); */
    /*     TEST_ASSERT_EQ(x_->data.complex_single[1][0], 3.0f); */
    /*     TEST_ASSERT_EQ(x_->data.complex_single[1][1],-3.0f); */
    /*     TEST_ASSERT_EQ(x_->data.complex_single[2][0], 4.0f); */
    /*     TEST_ASSERT_EQ(x_->data.complex_single[2][1],-4.0f); */
    /*     TEST_ASSERT_EQ(x_->data.complex_single[3][0], 9.0f); */
    /*     TEST_ASSERT_EQ(x_->data.complex_single[3][1],-9.0f); */
    /*     mtxmatrix_free(&x); */
    /*     mtxfile_free(&mtxfile); */
    /* } */

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
        err = mtxmatrix_from_mtxfile(&x, &mtxfile, mtxmatrix_auto);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxmatrix_coordinate, x.type);
        const struct mtxmatrix_coordinate * x_ = &x.storage.coordinate;
        TEST_ASSERT_EQ(mtx_field_complex, x_->field);
        TEST_ASSERT_EQ(mtx_double, x_->precision);
        TEST_ASSERT_EQ(mtx_unsymmetric, x_->symmetry);
        TEST_ASSERT_EQ(3, x_->num_rows);
        TEST_ASSERT_EQ(3, x_->num_columns);
        TEST_ASSERT_EQ(4, x_->num_nonzeros);
        TEST_ASSERT_EQ(4, x_->size);
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
        int64_t size = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile mtxfile;
        err = mtxfile_init_matrix_coordinate_integer_single(
            &mtxfile, mtxfile_general, num_rows, num_columns, size, mtxdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxmatrix x;
        err = mtxmatrix_from_mtxfile(&x, &mtxfile, mtxmatrix_auto);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxmatrix_coordinate, x.type);
        const struct mtxmatrix_coordinate * x_ = &x.storage.coordinate;
        TEST_ASSERT_EQ(mtx_field_integer, x_->field);
        TEST_ASSERT_EQ(mtx_single, x_->precision);
        TEST_ASSERT_EQ(mtx_unsymmetric, x_->symmetry);
        TEST_ASSERT_EQ(3, x_->num_rows);
        TEST_ASSERT_EQ(3, x_->num_columns);
        TEST_ASSERT_EQ(4, x_->num_nonzeros);
        TEST_ASSERT_EQ(4, x_->size);
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
    /* { */
    /*     int num_rows = 3; */
    /*     int num_columns = 3; */
    /*     struct mtxfile_matrix_coordinate_integer_single mtxdata[] = { */
    /*         {1,1,1}, {1,3,3}, {2,1,4}, {3,3,9}}; */
    /*     int64_t size = sizeof(mtxdata) / sizeof(*mtxdata); */
    /*     struct mtxfile mtxfile; */
    /*     err = mtxfile_init_matrix_coordinate_integer_single( */
    /*         &mtxfile, mtxfile_symmetric, num_rows, num_columns, size, mtxdata); */
    /*     TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err)); */
    /*     struct mtxmatrix x; */
    /*     err = mtxmatrix_from_mtxfile(&x, &mtxfile, mtxmatrix_auto); */
    /*     TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err)); */
    /*     TEST_ASSERT_EQ(mtxmatrix_coordinate, x.type); */
    /*     const struct mtxmatrix_coordinate * x_ = &x.storage.coordinate; */
    /*     TEST_ASSERT_EQ(mtx_field_integer, x_->field); */
    /*     TEST_ASSERT_EQ(mtx_single, x_->precision); */
    /*     TEST_ASSERT_EQ(mtx_symmetric, x_->symmetry); */
    /*     TEST_ASSERT_EQ(3, x_->num_rows); */
    /*     TEST_ASSERT_EQ(3, x_->num_columns); */
    /*     TEST_ASSERT_EQ(6, x_->num_nonzeros); */
    /*     TEST_ASSERT_EQ(4, x_->size); */
    /*     TEST_ASSERT_EQ(x_->rowidx[0], 0); TEST_ASSERT_EQ(x_->colidx[0], 0); */
    /*     TEST_ASSERT_EQ(x_->rowidx[1], 0); TEST_ASSERT_EQ(x_->colidx[1], 2); */
    /*     TEST_ASSERT_EQ(x_->rowidx[2], 1); TEST_ASSERT_EQ(x_->colidx[2], 0); */
    /*     TEST_ASSERT_EQ(x_->rowidx[3], 2); TEST_ASSERT_EQ(x_->colidx[3], 2); */
    /*     TEST_ASSERT_EQ(x_->data.integer_single[0], 1); */
    /*     TEST_ASSERT_EQ(x_->data.integer_single[1], 3); */
    /*     TEST_ASSERT_EQ(x_->data.integer_single[2], 4); */
    /*     TEST_ASSERT_EQ(x_->data.integer_single[3], 9); */
    /*     mtxmatrix_free(&x); */
    /*     mtxfile_free(&mtxfile); */
    /* } */
    /* { */
    /*     int num_rows = 3; */
    /*     int num_columns = 3; */
    /*     struct mtxfile_matrix_coordinate_integer_single mtxdata[] = { */
    /*         {1,2,2}, {1,3,3}, {2,3,6}}; */
    /*     int64_t size = sizeof(mtxdata) / sizeof(*mtxdata); */
    /*     struct mtxfile mtxfile; */
    /*     err = mtxfile_init_matrix_coordinate_integer_single( */
    /*         &mtxfile, mtxfile_skew_symmetric, num_rows, num_columns, size, */
    /*         mtxdata); */
    /*     TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err)); */
    /*     struct mtxmatrix x; */
    /*     err = mtxmatrix_from_mtxfile(&x, &mtxfile, mtxmatrix_auto); */
    /*     TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err)); */
    /*     TEST_ASSERT_EQ(mtxmatrix_coordinate, x.type); */
    /*     const struct mtxmatrix_coordinate * x_ = &x.storage.coordinate; */
    /*     TEST_ASSERT_EQ(mtx_field_integer, x_->field); */
    /*     TEST_ASSERT_EQ(mtx_single, x_->precision); */
    /*     TEST_ASSERT_EQ(mtx_skew_symmetric, x_->symmetry); */
    /*     TEST_ASSERT_EQ(3, x_->num_rows); */
    /*     TEST_ASSERT_EQ(3, x_->num_columns); */
    /*     TEST_ASSERT_EQ(6, x_->num_nonzeros); */
    /*     TEST_ASSERT_EQ(3, x_->size); */
    /*     TEST_ASSERT_EQ(x_->rowidx[0], 0); TEST_ASSERT_EQ(x_->colidx[0], 1); */
    /*     TEST_ASSERT_EQ(x_->rowidx[1], 0); TEST_ASSERT_EQ(x_->colidx[1], 2); */
    /*     TEST_ASSERT_EQ(x_->rowidx[2], 1); TEST_ASSERT_EQ(x_->colidx[2], 2); */
    /*     TEST_ASSERT_EQ(x_->data.integer_single[0], 2); */
    /*     TEST_ASSERT_EQ(x_->data.integer_single[1], 3); */
    /*     TEST_ASSERT_EQ(x_->data.integer_single[2], 6); */
    /*     mtxmatrix_free(&x); */
    /*     mtxfile_free(&mtxfile); */
    /* } */

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
        err = mtxmatrix_from_mtxfile(&x, &mtxfile, mtxmatrix_auto);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxmatrix_coordinate, x.type);
        const struct mtxmatrix_coordinate * x_ = &x.storage.coordinate;
        TEST_ASSERT_EQ(mtx_field_integer, x_->field);
        TEST_ASSERT_EQ(mtx_double, x_->precision);
        TEST_ASSERT_EQ(mtx_unsymmetric, x_->symmetry);
        TEST_ASSERT_EQ(3, x_->num_rows);
        TEST_ASSERT_EQ(3, x_->num_columns);
        TEST_ASSERT_EQ(4, x_->num_nonzeros);
        TEST_ASSERT_EQ(4, x_->size);
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
    /* { */
    /*     int num_rows = 3; */
    /*     int num_columns = 3; */
    /*     struct mtxfile_matrix_coordinate_integer_double mtxdata[] = { */
    /*         {1,1,1}, {1,3,3}, {2,1,4}, {3,3,9}}; */
    /*     int64_t size = sizeof(mtxdata) / sizeof(*mtxdata); */
    /*     struct mtxfile mtxfile; */
    /*     err = mtxfile_init_matrix_coordinate_integer_double( */
    /*         &mtxfile, mtxfile_symmetric, num_rows, num_columns, size, mtxdata); */
    /*     TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err)); */
    /*     struct mtxmatrix x; */
    /*     err = mtxmatrix_from_mtxfile(&x, &mtxfile, mtxmatrix_auto); */
    /*     TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err)); */
    /*     TEST_ASSERT_EQ(mtxmatrix_coordinate, x.type); */
    /*     const struct mtxmatrix_coordinate * x_ = &x.storage.coordinate; */
    /*     TEST_ASSERT_EQ(mtx_field_integer, x_->field); */
    /*     TEST_ASSERT_EQ(mtx_double, x_->precision); */
    /*     TEST_ASSERT_EQ(mtx_symmetric, x_->symmetry); */
    /*     TEST_ASSERT_EQ(3, x_->num_rows); */
    /*     TEST_ASSERT_EQ(3, x_->num_columns); */
    /*     TEST_ASSERT_EQ(6, x_->num_nonzeros); */
    /*     TEST_ASSERT_EQ(4, x_->size); */
    /*     TEST_ASSERT_EQ(x_->rowidx[0], 0); TEST_ASSERT_EQ(x_->colidx[0], 0); */
    /*     TEST_ASSERT_EQ(x_->rowidx[1], 0); TEST_ASSERT_EQ(x_->colidx[1], 2); */
    /*     TEST_ASSERT_EQ(x_->rowidx[2], 1); TEST_ASSERT_EQ(x_->colidx[2], 0); */
    /*     TEST_ASSERT_EQ(x_->rowidx[3], 2); TEST_ASSERT_EQ(x_->colidx[3], 2); */
    /*     TEST_ASSERT_EQ(x_->data.integer_double[0], 1); */
    /*     TEST_ASSERT_EQ(x_->data.integer_double[1], 3); */
    /*     TEST_ASSERT_EQ(x_->data.integer_double[2], 4); */
    /*     TEST_ASSERT_EQ(x_->data.integer_double[3], 9); */
    /*     mtxmatrix_free(&x); */
    /*     mtxfile_free(&mtxfile); */
    /* } */
    /* { */
    /*     int num_rows = 3; */
    /*     int num_columns = 3; */
    /*     struct mtxfile_matrix_coordinate_integer_double mtxdata[] = { */
    /*         {1,2,2}, {1,3,3}, {2,3,6}}; */
    /*     int64_t size = sizeof(mtxdata) / sizeof(*mtxdata); */
    /*     struct mtxfile mtxfile; */
    /*     err = mtxfile_init_matrix_coordinate_integer_double( */
    /*         &mtxfile, mtxfile_skew_symmetric, num_rows, num_columns, size, */
    /*         mtxdata); */
    /*     TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err)); */
    /*     struct mtxmatrix x; */
    /*     err = mtxmatrix_from_mtxfile(&x, &mtxfile, mtxmatrix_auto); */
    /*     TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err)); */
    /*     TEST_ASSERT_EQ(mtxmatrix_coordinate, x.type); */
    /*     const struct mtxmatrix_coordinate * x_ = &x.storage.coordinate; */
    /*     TEST_ASSERT_EQ(mtx_field_integer, x_->field); */
    /*     TEST_ASSERT_EQ(mtx_double, x_->precision); */
    /*     TEST_ASSERT_EQ(mtx_skew_symmetric, x_->symmetry); */
    /*     TEST_ASSERT_EQ(3, x_->num_rows); */
    /*     TEST_ASSERT_EQ(3, x_->num_columns); */
    /*     TEST_ASSERT_EQ(6, x_->num_nonzeros); */
    /*     TEST_ASSERT_EQ(3, x_->size); */
    /*     TEST_ASSERT_EQ(x_->rowidx[0], 0); TEST_ASSERT_EQ(x_->colidx[0], 1); */
    /*     TEST_ASSERT_EQ(x_->rowidx[1], 0); TEST_ASSERT_EQ(x_->colidx[1], 2); */
    /*     TEST_ASSERT_EQ(x_->rowidx[2], 1); TEST_ASSERT_EQ(x_->colidx[2], 2); */
    /*     TEST_ASSERT_EQ(x_->data.integer_double[0], 2); */
    /*     TEST_ASSERT_EQ(x_->data.integer_double[1], 3); */
    /*     TEST_ASSERT_EQ(x_->data.integer_double[2], 6); */
    /*     mtxmatrix_free(&x); */
    /*     mtxfile_free(&mtxfile); */
    /* } */

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
        err = mtxmatrix_from_mtxfile(&x, &mtxfile, mtxmatrix_auto);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxmatrix_coordinate, x.type);
        const struct mtxmatrix_coordinate * x_ = &x.storage.coordinate;
        TEST_ASSERT_EQ(mtx_field_pattern, x_->field);
        TEST_ASSERT_EQ(3, x_->num_rows);
        TEST_ASSERT_EQ(3, x_->num_columns);
        TEST_ASSERT_EQ(4, x_->num_nonzeros);
        TEST_ASSERT_EQ(4, x_->size);
        TEST_ASSERT_EQ(x_->rowidx[0], 0); TEST_ASSERT_EQ(x_->colidx[0], 0);
        TEST_ASSERT_EQ(x_->rowidx[1], 0); TEST_ASSERT_EQ(x_->colidx[1], 2);
        TEST_ASSERT_EQ(x_->rowidx[2], 1); TEST_ASSERT_EQ(x_->colidx[2], 0);
        TEST_ASSERT_EQ(x_->rowidx[3], 2); TEST_ASSERT_EQ(x_->colidx[3], 2);
        mtxmatrix_free(&x);
        mtxfile_free(&mtxfile);
    }
    /* { */
    /*     int num_rows = 3; */
    /*     int num_columns = 3; */
    /*     struct mtxfile_matrix_coordinate_pattern mtxdata[] = { */
    /*         {1,1}, {1,3}, {2,1}, {3,3}}; */
    /*     int64_t size = sizeof(mtxdata) / sizeof(*mtxdata); */
    /*     struct mtxfile mtxfile; */
    /*     err = mtxfile_init_matrix_coordinate_pattern( */
    /*         &mtxfile, mtxfile_symmetric, num_rows, num_columns, size, mtxdata); */
    /*     TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err)); */
    /*     struct mtxmatrix x; */
    /*     err = mtxmatrix_from_mtxfile(&x, &mtxfile, mtxmatrix_auto); */
    /*     TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err)); */
    /*     TEST_ASSERT_EQ(mtxmatrix_coordinate, x.type); */
    /*     const struct mtxmatrix_coordinate * x_ = &x.storage.coordinate; */
    /*     TEST_ASSERT_EQ(mtx_field_pattern, x_->field); */
    /*     TEST_ASSERT_EQ(3, x_->num_rows); */
    /*     TEST_ASSERT_EQ(3, x_->num_columns); */
    /*     TEST_ASSERT_EQ(6, x_->num_nonzeros); */
    /*     TEST_ASSERT_EQ(4, x_->size); */
    /*     TEST_ASSERT_EQ(x_->rowidx[0], 0); TEST_ASSERT_EQ(x_->colidx[0], 0); */
    /*     TEST_ASSERT_EQ(x_->rowidx[1], 0); TEST_ASSERT_EQ(x_->colidx[1], 2); */
    /*     TEST_ASSERT_EQ(x_->rowidx[2], 1); TEST_ASSERT_EQ(x_->colidx[2], 0); */
    /*     TEST_ASSERT_EQ(x_->rowidx[3], 2); TEST_ASSERT_EQ(x_->colidx[3], 2); */
    /*     mtxmatrix_free(&x); */
    /*     mtxfile_free(&mtxfile); */
    /* } */
    return TEST_SUCCESS;
}

/**
 * ‘test_mtxmatrix_csr_to_mtxfile()’ tests converting matrices to
 * Matrix Market files.
 */
int test_mtxmatrix_csr_to_mtxfile(void)
{
    int err;

    {
        int num_rows = 3;
        int num_columns = 3;
        int nnz = 4;
        int64_t rowptr[] = {0, 2, 3, 4};
        int colidx[] = {1, 3, 1, 3};
        float Adata[] = {1.0f, 3.0f, 4.0f, 9.0f};
        struct mtxmatrix A;
        err = mtxmatrix_init_csr_real_single(
            &A, mtx_unsymmetric, num_rows, num_columns, rowptr, colidx, Adata);
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
        for (int i = 0; i < num_rows; i++) {
            for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++) {
                TEST_ASSERT_EQ(i+1, data[k].i);
                TEST_ASSERT_EQ(colidx[k]+1, data[k].j);
                TEST_ASSERT_EQ(Adata[k], data[k].a);
            }
        }
        mtxfile_free(&mtxfile);
        mtxmatrix_free(&A);
    }
    /* { */
    /*     int num_rows = 3; */
    /*     int num_columns = 3; */
    /*     int nnz = 4; */
    /*     int rowidx[] = {1, 1, 2, 3}; */
    /*     int colidx[] = {1, 3, 3, 3}; */
    /*     float Adata[] = {1.0f, 3.0f, 6.0f, 9.0f}; */
    /*     struct mtxmatrix A; */
    /*     err = mtxmatrix_init_coordinate_real_single( */
    /*         &A, mtx_symmetric, num_rows, num_columns, nnz, rowidx, colidx, Adata); */
    /*     TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err)); */
    /*     struct mtxfile mtxfile; */
    /*     err = mtxmatrix_to_mtxfile(&mtxfile, &A, mtxfile_coordinate); */
    /*     TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err)); */
    /*     TEST_ASSERT_EQ(mtxfile_matrix, mtxfile.header.object); */
    /*     TEST_ASSERT_EQ(mtxfile_coordinate, mtxfile.header.format); */
    /*     TEST_ASSERT_EQ(mtxfile_real, mtxfile.header.field); */
    /*     TEST_ASSERT_EQ(mtxfile_symmetric, mtxfile.header.symmetry); */
    /*     TEST_ASSERT_EQ(num_rows, mtxfile.size.num_rows); */
    /*     TEST_ASSERT_EQ(num_columns, mtxfile.size.num_columns); */
    /*     TEST_ASSERT_EQ(nnz, mtxfile.size.num_nonzeros); */
    /*     TEST_ASSERT_EQ(mtx_single, mtxfile.precision); */
    /*     const struct mtxfile_matrix_coordinate_real_single * data = */
    /*         mtxfile.data.matrix_coordinate_real_single; */
    /*     for (int64_t k = 0; k < nnz; k++) { */
    /*         TEST_ASSERT_EQ(rowidx[k]+1, data[k].i); */
    /*         TEST_ASSERT_EQ(colidx[k]+1, data[k].j); */
    /*         TEST_ASSERT_EQ(Adata[k], data[k].a); */
    /*     } */
    /*     mtxfile_free(&mtxfile); */
    /*     mtxmatrix_free(&A); */
    /* } */
    /* { */
    /*     int num_rows = 3; */
    /*     int num_columns = 3; */
    /*     int nnz = 3; */
    /*     int rowidx[] = {1, 1, 2}; */
    /*     int colidx[] = {2, 3, 3}; */
    /*     float Adata[] = {2.0f, 3.0f, 6.0f}; */
    /*     struct mtxmatrix A; */
    /*     err = mtxmatrix_init_coordinate_real_single( */
    /*         &A, mtx_skew_symmetric, num_rows, num_columns, nnz, rowidx, colidx, Adata); */
    /*     TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err)); */
    /*     struct mtxfile mtxfile; */
    /*     err = mtxmatrix_to_mtxfile(&mtxfile, &A, mtxfile_coordinate); */
    /*     TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err)); */
    /*     TEST_ASSERT_EQ(mtxfile_matrix, mtxfile.header.object); */
    /*     TEST_ASSERT_EQ(mtxfile_coordinate, mtxfile.header.format); */
    /*     TEST_ASSERT_EQ(mtxfile_real, mtxfile.header.field); */
    /*     TEST_ASSERT_EQ(mtxfile_skew_symmetric, mtxfile.header.symmetry); */
    /*     TEST_ASSERT_EQ(num_rows, mtxfile.size.num_rows); */
    /*     TEST_ASSERT_EQ(num_columns, mtxfile.size.num_columns); */
    /*     TEST_ASSERT_EQ(nnz, mtxfile.size.num_nonzeros); */
    /*     TEST_ASSERT_EQ(mtx_single, mtxfile.precision); */
    /*     const struct mtxfile_matrix_coordinate_real_single * data = */
    /*         mtxfile.data.matrix_coordinate_real_single; */
    /*     for (int64_t k = 0; k < nnz; k++) { */
    /*         TEST_ASSERT_EQ(rowidx[k]+1, data[k].i); */
    /*         TEST_ASSERT_EQ(colidx[k]+1, data[k].j); */
    /*         TEST_ASSERT_EQ(Adata[k], data[k].a); */
    /*     } */
    /*     mtxfile_free(&mtxfile); */
    /*     mtxmatrix_free(&A); */
    /* } */

    {
        int num_rows = 3;
        int num_columns = 3;
        int nnz = 4;
        int64_t rowptr[] = {0, 2, 3, 4};
        int colidx[] = {1, 3, 1, 3};
        double Adata[] = {1.0, 3.0, 4.0, 9.0};
        struct mtxmatrix A;
        err = mtxmatrix_init_csr_real_double(
            &A, mtx_unsymmetric, num_rows, num_columns, rowptr, colidx, Adata);
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
        for (int i = 0; i < num_rows; i++) {
            for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++) {
                TEST_ASSERT_EQ(i+1, data[k].i);
                TEST_ASSERT_EQ(colidx[k]+1, data[k].j);
                TEST_ASSERT_EQ(Adata[k], data[k].a);
            }
        }
        mtxfile_free(&mtxfile);
        mtxmatrix_free(&A);
    }

    {
        int num_rows = 3;
        int num_columns = 3;
        int nnz = 4;
        int64_t rowptr[] = {0, 2, 3, 4};
        int colidx[] = {1, 3, 1, 3};
        float Adata[][2] = {{1.0f,-1.0f}, {3.0f,-3.0f}, {4.0f,-4.0f}, {9.0f,-9.0f}};
        struct mtxmatrix A;
        err = mtxmatrix_init_csr_complex_single(
            &A, mtx_unsymmetric, num_rows, num_columns, rowptr, colidx, Adata);
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
        for (int i = 0; i < num_rows; i++) {
            for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++) {
                TEST_ASSERT_EQ(i+1, data[k].i);
                TEST_ASSERT_EQ(colidx[k]+1, data[k].j);
                TEST_ASSERT_EQ(Adata[k][0], data[k].a[0]);
                TEST_ASSERT_EQ(Adata[k][1], data[k].a[1]);
            }
        }
        mtxfile_free(&mtxfile);
        mtxmatrix_free(&A);
    }

    /* { */
    /*     int num_rows = 3; */
    /*     int num_columns = 3; */
    /*     int nnz = 4; */
    /*     int rowidx[] = {1, 1, 2, 3}; */
    /*     int colidx[] = {2, 3, 3, 3}; */
    /*     float Adata[][2] = {{2.0f,-2.0f}, {3.0f,-3.0f}, {6.0f,-6.0f}, {9.0f,-9.0f}}; */
    /*     struct mtxmatrix A; */
    /*     err = mtxmatrix_init_coordinate_complex_single( */
    /*         &A, mtx_hermitian, num_rows, num_columns, nnz, rowidx, colidx, Adata); */
    /*     TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err)); */
    /*     struct mtxfile mtxfile; */
    /*     err = mtxmatrix_to_mtxfile(&mtxfile, &A, mtxfile_coordinate); */
    /*     TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err)); */
    /*     TEST_ASSERT_EQ(mtxfile_matrix, mtxfile.header.object); */
    /*     TEST_ASSERT_EQ(mtxfile_coordinate, mtxfile.header.format); */
    /*     TEST_ASSERT_EQ(mtxfile_complex, mtxfile.header.field); */
    /*     TEST_ASSERT_EQ(mtxfile_hermitian, mtxfile.header.symmetry); */
    /*     TEST_ASSERT_EQ(num_rows, mtxfile.size.num_rows); */
    /*     TEST_ASSERT_EQ(num_columns, mtxfile.size.num_columns); */
    /*     TEST_ASSERT_EQ(nnz, mtxfile.size.num_nonzeros); */
    /*     TEST_ASSERT_EQ(mtx_single, mtxfile.precision); */
    /*     const struct mtxfile_matrix_coordinate_complex_single * data = */
    /*         mtxfile.data.matrix_coordinate_complex_single; */
    /*     for (int64_t k = 0; k < nnz; k++) { */
    /*         TEST_ASSERT_EQ(rowidx[k]+1, data[k].i); */
    /*         TEST_ASSERT_EQ(colidx[k]+1, data[k].j); */
    /*         TEST_ASSERT_EQ(Adata[k][0], data[k].a[0]); */
    /*         TEST_ASSERT_EQ(Adata[k][1], data[k].a[1]); */
    /*     } */
    /*     mtxfile_free(&mtxfile); */
    /*     mtxmatrix_free(&A); */
    /* } */

    {
        int num_rows = 3;
        int num_columns = 3;
        int nnz = 4;
        int64_t rowptr[] = {0, 2, 3, 4};
        int colidx[] = {1, 3, 1, 3};
        double Adata[][2] = {{1.0,-1.0}, {3.0,-3.0}, {4.0,-4.0}, {9.0,-9.0}};
        struct mtxmatrix A;
        err = mtxmatrix_init_csr_complex_double(
            &A, mtx_unsymmetric, num_rows, num_columns, rowptr, colidx, Adata);
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
        for (int i = 0; i < num_rows; i++) {
            for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++) {
                TEST_ASSERT_EQ(i+1, data[k].i);
                TEST_ASSERT_EQ(colidx[k]+1, data[k].j);
                TEST_ASSERT_EQ(Adata[k][0], data[k].a[0]);
                TEST_ASSERT_EQ(Adata[k][1], data[k].a[1]);
            }
        }
        mtxfile_free(&mtxfile);
        mtxmatrix_free(&A);
    }

    {
        int num_rows = 3;
        int num_columns = 3;
        int nnz = 4;
        int64_t rowptr[] = {0, 2, 3, 4};
        int colidx[] = {1, 3, 1, 3};
        int32_t Adata[] = {1, 3, 4, 9};
        struct mtxmatrix A;
        err = mtxmatrix_init_csr_integer_single(
            &A, mtx_unsymmetric, num_rows, num_columns, rowptr, colidx, Adata);
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
        for (int i = 0; i < num_rows; i++) {
            for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++) {
                TEST_ASSERT_EQ(i+1, data[k].i);
                TEST_ASSERT_EQ(colidx[k]+1, data[k].j);
                TEST_ASSERT_EQ(Adata[k], data[k].a);
            }
        }
        mtxfile_free(&mtxfile);
        mtxmatrix_free(&A);
    }

    {
        int num_rows = 3;
        int num_columns = 3;
        int nnz = 4;
        int64_t rowptr[] = {0, 2, 3, 4};
        int colidx[] = {1, 3, 1, 3};
        int64_t Adata[] = {1, 3, 4, 9};
        struct mtxmatrix A;
        err = mtxmatrix_init_csr_integer_double(
            &A, mtx_unsymmetric, num_rows, num_columns, rowptr, colidx, Adata);
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
        for (int i = 0; i < num_rows; i++) {
            for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++) {
                TEST_ASSERT_EQ(i+1, data[k].i);
                TEST_ASSERT_EQ(colidx[k]+1, data[k].j);
                TEST_ASSERT_EQ(Adata[k], data[k].a);
            }
        }
        mtxfile_free(&mtxfile);
        mtxmatrix_free(&A);
    }

    {
        int num_rows = 3;
        int num_columns = 3;
        int nnz = 4;
        int64_t rowptr[] = {0, 2, 3, 4};
        int colidx[] = {1, 3, 1, 3};
        struct mtxmatrix A;
        err = mtxmatrix_init_csr_pattern(
            &A, mtx_unsymmetric, num_rows, num_columns, rowptr, colidx);
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
        for (int i = 0; i < num_rows; i++) {
            for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++) {
                TEST_ASSERT_EQ(i+1, data[k].i);
                TEST_ASSERT_EQ(colidx[k]+1, data[k].j);
            }
        }
        mtxfile_free(&mtxfile);
        mtxmatrix_free(&A);
    }
    return TEST_SUCCESS;
}

/**
 * ‘test_mtxmatrix_csr_partition()’ tests partitioning matrices in
 * coordinate format.
 */
int test_mtxmatrix_csr_partition(void)
{
    int err;

    {
        struct mtxmatrix src;
        int num_rows = 3;
        int num_columns = 3;
        int nnz = 9;
        int64_t rowptr[] = {0, 3, 6, 9};
        int colidx[] = {0, 1, 2, 0, 1, 2, 0, 1, 2};
        err = mtxmatrix_init_csr_pattern(
            &src, mtx_unsymmetric, num_rows, num_columns, rowptr, colidx);
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
            TEST_ASSERT_EQ(mtxmatrix_csr, dsts[0].type);
            const struct mtxmatrix_csr * A = &dsts[0].storage.csr;
            TEST_ASSERT_EQ(mtx_field_pattern, A->field);
            TEST_ASSERT_EQ(2, A->num_rows);
            TEST_ASSERT_EQ(3, A->num_columns);
            TEST_ASSERT_EQ(6, A->num_nonzeros);
            TEST_ASSERT_EQ(6, A->size);
            TEST_ASSERT_EQ(A->rowptr[0], 0);
            TEST_ASSERT_EQ(A->rowptr[1], 3);
            TEST_ASSERT_EQ(A->rowptr[2], 6);
            TEST_ASSERT_EQ(A->colidx[0], 0);
            TEST_ASSERT_EQ(A->colidx[1], 1);
            TEST_ASSERT_EQ(A->colidx[2], 2);
            TEST_ASSERT_EQ(A->colidx[3], 0);
            TEST_ASSERT_EQ(A->colidx[4], 1);
            TEST_ASSERT_EQ(A->colidx[5], 2);
            mtxmatrix_free(&dsts[0]);
        }
        {
            TEST_ASSERT_EQ(mtxmatrix_csr, dsts[1].type);
            const struct mtxmatrix_csr * A = &dsts[1].storage.csr;
            TEST_ASSERT_EQ(mtx_field_pattern, A->field);
            TEST_ASSERT_EQ(1, A->num_rows);
            TEST_ASSERT_EQ(3, A->num_columns);
            TEST_ASSERT_EQ(3, A->num_nonzeros);
            TEST_ASSERT_EQ(3, A->size);
            TEST_ASSERT_EQ(A->rowptr[0], 0);
            TEST_ASSERT_EQ(A->rowptr[1], 3);
            TEST_ASSERT_EQ(A->colidx[0], 0);
            TEST_ASSERT_EQ(A->colidx[1], 1);
            TEST_ASSERT_EQ(A->colidx[2], 2);
            mtxmatrix_free(&dsts[1]);
        }
        mtxpartition_free(&rowpart);
        mtxmatrix_free(&src);
    }
    return TEST_SUCCESS;
}

/**
 * ‘test_mtxmatrix_csr_join()’ tests joining matrices in coordinate
 * format.
 */
int test_mtxmatrix_csr_join(void)
{
    int err;

    {
        struct mtxmatrix srcs[2];
        int num_rows[] = {2, 1};
        int num_columns[] = {3, 3};
        int nnz[] = {6, 3};
        int64_t rowptr0[] = {0, 3, 6};
        int colidx0[] = {0, 1, 2, 0, 1, 2};
        int64_t rowptr1[] = {0, 3};
        int colidx1[] = {0, 1, 2};
        err = mtxmatrix_init_csr_pattern(
            &srcs[0], mtx_unsymmetric, num_rows[0], num_columns[0], rowptr0, colidx0);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxmatrix_init_csr_pattern(
            &srcs[1], mtx_unsymmetric, num_rows[1], num_columns[1], rowptr1, colidx1);
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
        TEST_ASSERT_EQ(mtxmatrix_csr, dst.type);
        const struct mtxmatrix_csr * A = &dst.storage.csr;
        TEST_ASSERT_EQ(mtx_field_pattern, A->field);
        TEST_ASSERT_EQ(3, A->num_rows);
        TEST_ASSERT_EQ(3, A->num_columns);
        TEST_ASSERT_EQ(9, A->num_nonzeros);
        TEST_ASSERT_EQ(A->rowptr[0], 0);
        TEST_ASSERT_EQ(A->rowptr[1], 3);
        TEST_ASSERT_EQ(A->rowptr[2], 6);
        TEST_ASSERT_EQ(A->rowptr[3], 9);
        TEST_ASSERT_EQ(A->colidx[0], 0);
        TEST_ASSERT_EQ(A->colidx[1], 1);
        TEST_ASSERT_EQ(A->colidx[2], 2);
        TEST_ASSERT_EQ(A->colidx[3], 0);
        TEST_ASSERT_EQ(A->colidx[4], 1);
        TEST_ASSERT_EQ(A->colidx[5], 2);
        TEST_ASSERT_EQ(A->colidx[6], 0);
        TEST_ASSERT_EQ(A->colidx[7], 1);
        TEST_ASSERT_EQ(A->colidx[8], 2);
        mtxmatrix_free(&dst);
        mtxpartition_free(&rowpart);
        mtxmatrix_free(&srcs[1]);
        mtxmatrix_free(&srcs[0]);
    }
    return TEST_SUCCESS;
}

/**
 * ‘test_mtxmatrix_csr_gemv()’ tests computing matrix-vector products
 * for matrices in CSR format.
 */
int test_mtxmatrix_csr_gemv(void)
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
        /* int num_nonzeros = 5; */
        /* int Arowidx[] = {0, 0, 1, 1, 2}; */
        int64_t Arowptr[] = {0, 2, 4, 5};
        int Acolidx[] = {0, 2, 0, 1, 2};
        float Adata[] = {1.0f, 3.0f, 4.0f, 5.0f, 9.0f};
        float xdata[] = {3.0f, 2.0f, 1.0f};
        float ydata[] = {1.0f, 0.0f, 2.0f};
        err = mtxmatrix_init_csr_real_single(
            &A, mtx_unsymmetric, num_rows, num_columns, Arowptr, Acolidx, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_init_array_real_single(&x, num_columns, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        {
            err = mtxvector_init_array_real_single(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemv(mtx_notrans, 2.0f, &A, &x, 1.0f, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemv(mtx_notrans, 2.0f, &A, &x, 3.0f, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
        /* { */
        /*     err = mtxvector_init_array_real_single(&y, num_rows, ydata); */
        /*     TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err)); */
        /*     err = mtxmatrix_sgemv(mtx_trans, 2.0f, &A, &x, 3.0f, &y, NULL); */
        /*     TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err)); */
        /*     TEST_ASSERT_EQ(mtxvector_array, y.type); */
        /*     const struct mtxvector_array * y_ = &y.storage.array; */
        /*     TEST_ASSERT_EQ(mtx_field_real, y_->field); */
        /*     TEST_ASSERT_EQ(mtx_single, y_->precision); */
        /*     TEST_ASSERT_EQ(3, y_->size); */
        /*     TEST_ASSERT_EQ(y_->data.real_single[0], 25.0f); */
        /*     TEST_ASSERT_EQ(y_->data.real_single[1], 20.0f); */
        /*     TEST_ASSERT_EQ(y_->data.real_single[2], 42.0f); */
        /*     mtxvector_free(&y); */
        /* } */
        {
            err = mtxvector_init_array_real_single(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dgemv(mtx_notrans, 2.0, &A, &x, 3.0, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
        /* { */
        /*     err = mtxvector_init_array_real_single(&y, num_rows, ydata); */
        /*     TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err)); */
        /*     err = mtxmatrix_dgemv(mtx_trans, 2.0, &A, &x, 3.0, &y, NULL); */
        /*     TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err)); */
        /*     TEST_ASSERT_EQ(mtxvector_array, y.type); */
        /*     const struct mtxvector_array * y_ = &y.storage.array; */
        /*     TEST_ASSERT_EQ(mtx_field_real, y_->field); */
        /*     TEST_ASSERT_EQ(mtx_single, y_->precision); */
        /*     TEST_ASSERT_EQ(3, y_->size); */
        /*     TEST_ASSERT_EQ(y_->data.real_single[0], 25.0f); */
        /*     TEST_ASSERT_EQ(y_->data.real_single[1], 20.0f); */
        /*     TEST_ASSERT_EQ(y_->data.real_single[2], 42.0f); */
        /*     mtxvector_free(&y); */
        /* } */
        mtxvector_free(&x);
        mtxmatrix_free(&A);
    }

    {
        struct mtxmatrix A;
        struct mtxvector x;
        struct mtxvector y;
        int num_rows = 3;
        int num_columns = 3;
        /* int num_nonzeros = 5; */
        /* int Arowidx[] = {0, 0, 1, 1, 2}; */
        int64_t Arowptr[] = {0, 2, 4, 5};
        int Acolidx[] = {0, 2, 0, 1, 2};
        double Adata[] = {1.0, 3.0, 4.0, 5.0, 9.0};
        double xdata[] = {3.0, 2.0, 1.0};
        double ydata[] = {1.0, 0.0, 2.0};
        err = mtxmatrix_init_csr_real_double(
            &A, mtx_unsymmetric, num_rows, num_columns, Arowptr, Acolidx, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_init_array_real_double(&x, num_columns, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        {
            err = mtxvector_init_array_real_double(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemv(mtx_notrans, 2.0f, &A, &x, 1.0f, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemv(mtx_notrans, 2.0f, &A, &x, 3.0f, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
        /* { */
        /*     err = mtxvector_init_array_real_double(&y, num_rows, ydata); */
        /*     TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err)); */
        /*     err = mtxmatrix_sgemv(mtx_trans, 2.0f, &A, &x, 3.0f, &y, NULL); */
        /*     TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err)); */
        /*     TEST_ASSERT_EQ(mtxvector_array, y.type); */
        /*     const struct mtxvector_array * y_ = &y.storage.array; */
        /*     TEST_ASSERT_EQ(mtx_field_real, y_->field); */
        /*     TEST_ASSERT_EQ(mtx_double, y_->precision); */
        /*     TEST_ASSERT_EQ(3, y_->size); */
        /*     TEST_ASSERT_EQ(y_->data.real_double[0], 25.0); */
        /*     TEST_ASSERT_EQ(y_->data.real_double[1], 20.0); */
        /*     TEST_ASSERT_EQ(y_->data.real_double[2], 42.0); */
        /*     mtxvector_free(&y); */
        /* } */
        {
            err = mtxvector_init_array_real_double(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dgemv(mtx_notrans, 2.0, &A, &x, 3.0, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
        /* { */
        /*     err = mtxvector_init_array_real_double(&y, num_rows, ydata); */
        /*     TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err)); */
        /*     err = mtxmatrix_dgemv(mtx_trans, 2.0, &A, &x, 3.0, &y, NULL); */
        /*     TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err)); */
        /*     TEST_ASSERT_EQ(mtxvector_array, y.type); */
        /*     const struct mtxvector_array * y_ = &y.storage.array; */
        /*     TEST_ASSERT_EQ(mtx_field_real, y_->field); */
        /*     TEST_ASSERT_EQ(mtx_double, y_->precision); */
        /*     TEST_ASSERT_EQ(3, y_->size); */
        /*     TEST_ASSERT_EQ(y_->data.real_double[0], 25.0); */
        /*     TEST_ASSERT_EQ(y_->data.real_double[1], 20.0); */
        /*     TEST_ASSERT_EQ(y_->data.real_double[2], 42.0); */
        /*     mtxvector_free(&y); */
        /* } */
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
        /* int num_nonzeros = 3; */
        /* int Arowidx[] = {0, 0, 1}; */
        int64_t Arowptr[] = {0, 2, 3};
        int Acolidx[] = {0, 1, 1};
        float Adata[][2] = {{1.0f,2.0f}, {3.0f,4.0f}, {7.0f,8.0f}};
        float xdata[][2] = {{3.0f,1.0f}, {1.0f,2.0f}};
        float ydata[][2] = {{1.0f,0.0f}, {2.0f,2.0f}};
        err = mtxmatrix_init_csr_complex_single(
            &A, mtx_unsymmetric, num_rows, num_columns, Arowptr, Acolidx, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_init_array_complex_single(&x, num_columns, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        {
            err = mtxvector_init_array_complex_single(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemv(mtx_notrans, 2.0f, &A, &x, 1.0f, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemv(mtx_notrans, 2.0f, &A, &x, 3.0f, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
        /* { */
        /*     err = mtxvector_init_array_complex_single(&y, num_rows, ydata); */
        /*     TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err)); */
        /*     err = mtxmatrix_sgemv(mtx_trans, 2.0f, &A, &x, 3.0f, &y, NULL); */
        /*     TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err)); */
        /*     TEST_ASSERT_EQ(mtxvector_array, y.type); */
        /*     const struct mtxvector_array * y_ = &y.storage.array; */
        /*     TEST_ASSERT_EQ(mtx_field_complex, y_->field); */
        /*     TEST_ASSERT_EQ(mtx_single, y_->precision); */
        /*     TEST_ASSERT_EQ(2, y_->size); */
        /*     TEST_ASSERT_EQ(y_->data.complex_single[0][0],  5.0f); */
        /*     TEST_ASSERT_EQ(y_->data.complex_single[0][1], 14.0f); */
        /*     TEST_ASSERT_EQ(y_->data.complex_single[1][0], -2.0f); */
        /*     TEST_ASSERT_EQ(y_->data.complex_single[1][1], 80.0f); */
        /*     mtxvector_free(&y); */
        /* } */
        /* { */
        /*     err = mtxvector_init_array_complex_single(&y, num_rows, ydata); */
        /*     TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err)); */
        /*     err = mtxmatrix_sgemv(mtx_conjtrans, 2.0f, &A, &x, 3.0f, &y, NULL); */
        /*     TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err)); */
        /*     TEST_ASSERT_EQ(mtxvector_array, y.type); */
        /*     const struct mtxvector_array * y_ = &y.storage.array; */
        /*     TEST_ASSERT_EQ(mtx_field_complex, y_->field); */
        /*     TEST_ASSERT_EQ(mtx_single, y_->precision); */
        /*     TEST_ASSERT_EQ(2, y_->size); */
        /*     TEST_ASSERT_EQ(y_->data.complex_single[0][0], 13.0f); */
        /*     TEST_ASSERT_EQ(y_->data.complex_single[0][1],-10.0f); */
        /*     TEST_ASSERT_EQ(y_->data.complex_single[1][0], 78.0f); */
        /*     TEST_ASSERT_EQ(y_->data.complex_single[1][1],  0.0f); */
        /*     mtxvector_free(&y); */
        /* } */
        {
            err = mtxvector_init_array_complex_single(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dgemv(mtx_notrans, 2.0, &A, &x, 3.0, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
        /* { */
        /*     err = mtxvector_init_array_complex_single(&y, num_rows, ydata); */
        /*     TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err)); */
        /*     err = mtxmatrix_dgemv(mtx_trans, 2.0, &A, &x, 3.0, &y, NULL); */
        /*     TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err)); */
        /*     TEST_ASSERT_EQ(mtxvector_array, y.type); */
        /*     const struct mtxvector_array * y_ = &y.storage.array; */
        /*     TEST_ASSERT_EQ(mtx_field_complex, y_->field); */
        /*     TEST_ASSERT_EQ(mtx_single, y_->precision); */
        /*     TEST_ASSERT_EQ(2, y_->size); */
        /*     TEST_ASSERT_EQ(y_->data.complex_single[0][0],  5.0f); */
        /*     TEST_ASSERT_EQ(y_->data.complex_single[0][1], 14.0f); */
        /*     TEST_ASSERT_EQ(y_->data.complex_single[1][0], -2.0f); */
        /*     TEST_ASSERT_EQ(y_->data.complex_single[1][1], 80.0f); */
        /*     mtxvector_free(&y); */
        /* } */
        /* { */
        /*     err = mtxvector_init_array_complex_single(&y, num_rows, ydata); */
        /*     TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err)); */
        /*     err = mtxmatrix_dgemv(mtx_conjtrans, 2.0, &A, &x, 3.0, &y, NULL); */
        /*     TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err)); */
        /*     TEST_ASSERT_EQ(mtxvector_array, y.type); */
        /*     const struct mtxvector_array * y_ = &y.storage.array; */
        /*     TEST_ASSERT_EQ(mtx_field_complex, y_->field); */
        /*     TEST_ASSERT_EQ(mtx_single, y_->precision); */
        /*     TEST_ASSERT_EQ(2, y_->size); */
        /*     TEST_ASSERT_EQ(y_->data.complex_single[0][0], 13.0f); */
        /*     TEST_ASSERT_EQ(y_->data.complex_single[0][1],-10.0f); */
        /*     TEST_ASSERT_EQ(y_->data.complex_single[1][0], 78.0f); */
        /*     TEST_ASSERT_EQ(y_->data.complex_single[1][1],  0.0f); */
        /*     mtxvector_free(&y); */
        /* } */

        float calpha[2] = {0.0f, 2.0f};
        float cbeta[2]  = {3.0f, 1.0f};
        {
            err = mtxvector_init_array_complex_single(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_cgemv(mtx_notrans, calpha, &A, &x, cbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
        /* { */
        /*     err = mtxvector_init_array_complex_single(&y, num_rows, ydata); */
        /*     TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err)); */
        /*     err = mtxmatrix_cgemv(mtx_trans, calpha, &A, &x, cbeta, &y, NULL); */
        /*     TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err)); */
        /*     TEST_ASSERT_EQ(mtxvector_array, y.type); */
        /*     const struct mtxvector_array * y_ = &y.storage.array; */
        /*     TEST_ASSERT_EQ(mtx_field_complex, y_->field); */
        /*     TEST_ASSERT_EQ(mtx_single, y_->precision); */
        /*     TEST_ASSERT_EQ(2, y_->size); */
        /*     TEST_ASSERT_EQ(y_->data.complex_single[0][0],-11.0f); */
        /*     TEST_ASSERT_EQ(y_->data.complex_single[0][1],  3.0f); */
        /*     TEST_ASSERT_EQ(y_->data.complex_single[1][0],-70.0f); */
        /*     TEST_ASSERT_EQ(y_->data.complex_single[1][1],  0.0f); */
        /*     mtxvector_free(&y); */
        /* } */
        /* { */
        /*     err = mtxvector_init_array_complex_single(&y, num_rows, ydata); */
        /*     TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err)); */
        /*     err = mtxmatrix_cgemv(mtx_conjtrans, calpha, &A, &x, cbeta, &y, NULL); */
        /*     TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err)); */
        /*     TEST_ASSERT_EQ(mtxvector_array, y.type); */
        /*     const struct mtxvector_array * y_ = &y.storage.array; */
        /*     TEST_ASSERT_EQ(mtx_field_complex, y_->field); */
        /*     TEST_ASSERT_EQ(mtx_single, y_->precision); */
        /*     TEST_ASSERT_EQ(2, y_->size); */
        /*     TEST_ASSERT_EQ(y_->data.complex_single[0][0], 13.0f); */
        /*     TEST_ASSERT_EQ(y_->data.complex_single[0][1], 11.0f); */
        /*     TEST_ASSERT_EQ(y_->data.complex_single[1][0], 10.0f); */
        /*     TEST_ASSERT_EQ(y_->data.complex_single[1][1], 80.0f); */
        /*     mtxvector_free(&y); */
        /* } */

        double zalpha[2] = {0.0, 2.0};
        double zbeta[2]  = {3.0, 1.0};
        {
            err = mtxvector_init_array_complex_single(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_zgemv(mtx_notrans, zalpha, &A, &x, zbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
        /* { */
        /*     err = mtxvector_init_array_complex_single(&y, num_rows, ydata); */
        /*     TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err)); */
        /*     err = mtxmatrix_zgemv(mtx_trans, zalpha, &A, &x, zbeta, &y, NULL); */
        /*     TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err)); */
        /*     TEST_ASSERT_EQ(mtxvector_array, y.type); */
        /*     const struct mtxvector_array * y_ = &y.storage.array; */
        /*     TEST_ASSERT_EQ(mtx_field_complex, y_->field); */
        /*     TEST_ASSERT_EQ(mtx_single, y_->precision); */
        /*     TEST_ASSERT_EQ(2, y_->size); */
        /*     TEST_ASSERT_EQ(y_->data.complex_single[0][0], -11.0f); */
        /*     TEST_ASSERT_EQ(y_->data.complex_single[0][1],   3.0f); */
        /*     TEST_ASSERT_EQ(y_->data.complex_single[1][0], -70.0f); */
        /*     TEST_ASSERT_EQ(y_->data.complex_single[1][1],   0.0f); */
        /*     mtxvector_free(&y); */
        /* } */
        /* { */
        /*     err = mtxvector_init_array_complex_single(&y, num_rows, ydata); */
        /*     TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err)); */
        /*     err = mtxmatrix_zgemv(mtx_conjtrans, zalpha, &A, &x, zbeta, &y, NULL); */
        /*     TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err)); */
        /*     TEST_ASSERT_EQ(mtxvector_array, y.type); */
        /*     const struct mtxvector_array * y_ = &y.storage.array; */
        /*     TEST_ASSERT_EQ(mtx_field_complex, y_->field); */
        /*     TEST_ASSERT_EQ(mtx_single, y_->precision); */
        /*     TEST_ASSERT_EQ(2, y_->size); */
        /*     TEST_ASSERT_EQ(y_->data.complex_single[0][0], 13.0f); */
        /*     TEST_ASSERT_EQ(y_->data.complex_single[0][1], 11.0f); */
        /*     TEST_ASSERT_EQ(y_->data.complex_single[1][0], 10.0f); */
        /*     TEST_ASSERT_EQ(y_->data.complex_single[1][1], 80.0f); */
        /*     mtxvector_free(&y); */
        /* } */

        mtxvector_free(&x);
        mtxmatrix_free(&A);
    }

    {
        struct mtxmatrix A;
        struct mtxvector x;
        struct mtxvector y;
        int num_rows = 2;
        int num_columns = 2;
        /* int num_nonzeros = 3; */
        /* int Arowidx[] = {0, 0, 1}; */
        int64_t Arowptr[] = {0, 2, 3};
        int Acolidx[] = {0, 1, 1};
        double Adata[][2] = {{1.0,2.0}, {3.0,4.0}, {7.0,8.0}};
        double xdata[][2] = {{3.0,1.0}, {1.0,2.0}};
        double ydata[][2] = {{1.0,0.0}, {2.0,2.0}};
        err = mtxmatrix_init_csr_complex_double(
            &A, mtx_unsymmetric, num_rows, num_columns, Arowptr, Acolidx, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_init_array_complex_double(&x, num_columns, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        {
            err = mtxvector_init_array_complex_double(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemv(mtx_notrans, 2.0f, &A, &x, 1.0f, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemv(mtx_notrans, 2.0f, &A, &x, 3.0f, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
        /* { */
        /*     err = mtxvector_init_array_complex_double(&y, num_rows, ydata); */
        /*     TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err)); */
        /*     err = mtxmatrix_sgemv(mtx_trans, 2.0f, &A, &x, 3.0f, &y, NULL); */
        /*     TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err)); */
        /*     TEST_ASSERT_EQ(mtxvector_array, y.type); */
        /*     const struct mtxvector_array * y_ = &y.storage.array; */
        /*     TEST_ASSERT_EQ(mtx_field_complex, y_->field); */
        /*     TEST_ASSERT_EQ(mtx_double, y_->precision); */
        /*     TEST_ASSERT_EQ(2, y_->size); */
        /*     TEST_ASSERT_EQ(y_->data.complex_double[0][0],  5.0); */
        /*     TEST_ASSERT_EQ(y_->data.complex_double[0][1], 14.0); */
        /*     TEST_ASSERT_EQ(y_->data.complex_double[1][0], -2.0); */
        /*     TEST_ASSERT_EQ(y_->data.complex_double[1][1], 80.0); */
        /*     mtxvector_free(&y); */
        /* } */
        /* { */
        /*     err = mtxvector_init_array_complex_double(&y, num_rows, ydata); */
        /*     TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err)); */
        /*     err = mtxmatrix_sgemv(mtx_conjtrans, 2.0f, &A, &x, 3.0f, &y, NULL); */
        /*     TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err)); */
        /*     TEST_ASSERT_EQ(mtxvector_array, y.type); */
        /*     const struct mtxvector_array * y_ = &y.storage.array; */
        /*     TEST_ASSERT_EQ(mtx_field_complex, y_->field); */
        /*     TEST_ASSERT_EQ(mtx_double, y_->precision); */
        /*     TEST_ASSERT_EQ(2, y_->size); */
        /*     TEST_ASSERT_EQ(y_->data.complex_double[0][0], 13.0); */
        /*     TEST_ASSERT_EQ(y_->data.complex_double[0][1],-10.0); */
        /*     TEST_ASSERT_EQ(y_->data.complex_double[1][0], 78.0); */
        /*     TEST_ASSERT_EQ(y_->data.complex_double[1][1],  0.0); */
        /*     mtxvector_free(&y); */
        /* } */
        {
            err = mtxvector_init_array_complex_double(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dgemv(mtx_notrans, 2.0, &A, &x, 3.0, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
        /* { */
        /*     err = mtxvector_init_array_complex_double(&y, num_rows, ydata); */
        /*     TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err)); */
        /*     err = mtxmatrix_dgemv(mtx_trans, 2.0, &A, &x, 3.0, &y, NULL); */
        /*     TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err)); */
        /*     TEST_ASSERT_EQ(mtxvector_array, y.type); */
        /*     const struct mtxvector_array * y_ = &y.storage.array; */
        /*     TEST_ASSERT_EQ(mtx_field_complex, y_->field); */
        /*     TEST_ASSERT_EQ(mtx_double, y_->precision); */
        /*     TEST_ASSERT_EQ(2, y_->size); */
        /*     TEST_ASSERT_EQ(y_->data.complex_double[0][0],  5.0); */
        /*     TEST_ASSERT_EQ(y_->data.complex_double[0][1], 14.0); */
        /*     TEST_ASSERT_EQ(y_->data.complex_double[1][0], -2.0); */
        /*     TEST_ASSERT_EQ(y_->data.complex_double[1][1], 80.0); */
        /*     mtxvector_free(&y); */
        /* } */
        /* { */
        /*     err = mtxvector_init_array_complex_double(&y, num_rows, ydata); */
        /*     TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err)); */
        /*     err = mtxmatrix_dgemv(mtx_conjtrans, 2.0, &A, &x, 3.0, &y, NULL); */
        /*     TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err)); */
        /*     TEST_ASSERT_EQ(mtxvector_array, y.type); */
        /*     const struct mtxvector_array * y_ = &y.storage.array; */
        /*     TEST_ASSERT_EQ(mtx_field_complex, y_->field); */
        /*     TEST_ASSERT_EQ(mtx_double, y_->precision); */
        /*     TEST_ASSERT_EQ(2, y_->size); */
        /*     TEST_ASSERT_EQ(y_->data.complex_double[0][0], 13.0); */
        /*     TEST_ASSERT_EQ(y_->data.complex_double[0][1],-10.0); */
        /*     TEST_ASSERT_EQ(y_->data.complex_double[1][0], 78.0); */
        /*     TEST_ASSERT_EQ(y_->data.complex_double[1][1],  0.0); */
        /*     mtxvector_free(&y); */
        /* } */
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
        /* int num_nonzeros = 5; */
        /* int Arowidx[] = {0, 0, 1, 1, 2}; */
        int64_t Arowptr[] = {0, 2, 4, 5};
        int Acolidx[] = {0, 2, 0, 1, 2};
        int32_t Adata[] = {1, 3, 4, 5, 9};
        int32_t xdata[] = {3, 2, 1};
        int32_t ydata[] = {1, 0, 2};
        err = mtxmatrix_init_csr_integer_single(
            &A, mtx_unsymmetric, num_rows, num_columns, Arowptr, Acolidx, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_init_array_integer_single(&x, num_columns, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        {
            err = mtxvector_init_array_integer_single(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemv(mtx_notrans, 2.0f, &A, &x, 3.0f, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
        /* { */
        /*     err = mtxvector_init_array_integer_single(&y, num_rows, ydata); */
        /*     TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err)); */
        /*     err = mtxmatrix_sgemv(mtx_trans, 2.0f, &A, &x, 3.0f, &y, NULL); */
        /*     TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err)); */
        /*     TEST_ASSERT_EQ(mtxvector_array, y.type); */
        /*     const struct mtxvector_array * y_ = &y.storage.array; */
        /*     TEST_ASSERT_EQ(mtx_field_integer, y_->field); */
        /*     TEST_ASSERT_EQ(mtx_single, y_->precision); */
        /*     TEST_ASSERT_EQ(3, y_->size); */
        /*     TEST_ASSERT_EQ(y_->data.integer_single[0], 25); */
        /*     TEST_ASSERT_EQ(y_->data.integer_single[1], 20); */
        /*     TEST_ASSERT_EQ(y_->data.integer_single[2], 42); */
        /*     mtxvector_free(&y); */
        /* } */
        {
            err = mtxvector_init_array_integer_single(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dgemv(mtx_notrans, 2.0, &A, &x, 3.0, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
        /* { */
        /*     err = mtxvector_init_array_integer_single(&y, num_rows, ydata); */
        /*     TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err)); */
        /*     err = mtxmatrix_dgemv(mtx_trans, 2.0, &A, &x, 3.0, &y, NULL); */
        /*     TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err)); */
        /*     TEST_ASSERT_EQ(mtxvector_array, y.type); */
        /*     const struct mtxvector_array * y_ = &y.storage.array; */
        /*     TEST_ASSERT_EQ(mtx_field_integer, y_->field); */
        /*     TEST_ASSERT_EQ(mtx_single, y_->precision); */
        /*     TEST_ASSERT_EQ(3, y_->size); */
        /*     TEST_ASSERT_EQ(y_->data.integer_single[0], 25); */
        /*     TEST_ASSERT_EQ(y_->data.integer_single[1], 20); */
        /*     TEST_ASSERT_EQ(y_->data.integer_single[2], 42); */
        /*     mtxvector_free(&y); */
        /* } */
        mtxvector_free(&x);
        mtxmatrix_free(&A);
    }

    {
        struct mtxmatrix A;
        struct mtxvector x;
        struct mtxvector y;
        int num_rows = 3;
        int num_columns = 3;
        /* int num_nonzeros = 5; */
        /* int Arowidx[] = {0, 0, 1, 1, 2}; */
        int64_t Arowptr[] = {0, 2, 4, 5};
        int Acolidx[] = {0, 2, 0, 1, 2};
        int64_t Adata[] = {1, 3, 4, 5, 9};
        int64_t xdata[] = {3, 2, 1};
        int64_t ydata[] = {1, 0, 2};
        err = mtxmatrix_init_csr_integer_double(
            &A, mtx_unsymmetric, num_rows, num_columns, Arowptr, Acolidx, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_init_array_integer_double(&x, num_columns, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        {
            err = mtxvector_init_array_integer_double(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemv(mtx_notrans, 2.0f, &A, &x, 3.0f, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
        /* { */
        /*     err = mtxvector_init_array_integer_double(&y, num_rows, ydata); */
        /*     TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err)); */
        /*     err = mtxmatrix_sgemv(mtx_trans, 2.0f, &A, &x, 3.0f, &y, NULL); */
        /*     TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err)); */
        /*     TEST_ASSERT_EQ(mtxvector_array, y.type); */
        /*     const struct mtxvector_array * y_ = &y.storage.array; */
        /*     TEST_ASSERT_EQ(mtx_field_integer, y_->field); */
        /*     TEST_ASSERT_EQ(mtx_double, y_->precision); */
        /*     TEST_ASSERT_EQ(3, y_->size); */
        /*     TEST_ASSERT_EQ(y_->data.integer_double[0], 25); */
        /*     TEST_ASSERT_EQ(y_->data.integer_double[1], 20); */
        /*     TEST_ASSERT_EQ(y_->data.integer_double[2], 42); */
        /*     mtxvector_free(&y); */
        /* } */
        {
            err = mtxvector_init_array_integer_double(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dgemv(mtx_notrans, 2.0, &A, &x, 3.0, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
        /* { */
        /*     err = mtxvector_init_array_integer_double(&y, num_rows, ydata); */
        /*     TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err)); */
        /*     err = mtxmatrix_dgemv(mtx_trans, 2.0, &A, &x, 3.0, &y, NULL); */
        /*     TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err)); */
        /*     TEST_ASSERT_EQ(mtxvector_array, y.type); */
        /*     const struct mtxvector_array * y_ = &y.storage.array; */
        /*     TEST_ASSERT_EQ(mtx_field_integer, y_->field); */
        /*     TEST_ASSERT_EQ(mtx_double, y_->precision); */
        /*     TEST_ASSERT_EQ(3, y_->size); */
        /*     TEST_ASSERT_EQ(y_->data.integer_double[0], 25); */
        /*     TEST_ASSERT_EQ(y_->data.integer_double[1], 20); */
        /*     TEST_ASSERT_EQ(y_->data.integer_double[2], 42); */
        /*     mtxvector_free(&y); */
        /* } */
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
        /* int num_nonzeros = 5; */
        /* int Arowidx[] = {0, 0, 1, 1, 2}; */
        int64_t Arowptr[] = {0, 2, 4, 5};
        int Acolidx[] = {0, 2, 0, 1, 2};
        err = mtxmatrix_init_csr_pattern(
            &A, mtx_unsymmetric, num_rows, num_columns, Arowptr, Acolidx);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        {
            float xdata[] = {3, 2, 1}, ydata[] = {1, 0, 2};
            err = mtxvector_init_array_real_single(&x, num_columns, xdata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxvector_init_array_real_single(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemv(mtx_notrans, 2.0f, &A, &x, 3.0f, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxvector_init_array_real_double(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemv(mtx_notrans, 2.0f, &A, &x, 3.0f, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxvector_init_array_integer_single(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemv(mtx_notrans, 2.0f, &A, &x, 3.0f, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
        /* { */
        /*     int32_t xdata[] = {3, 2, 1}, ydata[] = {1, 0, 2}; */
        /*     err = mtxvector_init_array_integer_single(&x, num_columns, xdata); */
        /*     TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err)); */
        /*     err = mtxvector_init_array_integer_single(&y, num_rows, ydata); */
        /*     TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err)); */
        /*     err = mtxmatrix_sgemv(mtx_trans, 2.0f, &A, &x, 3.0f, &y, NULL); */
        /*     TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err)); */
        /*     TEST_ASSERT_EQ(mtxvector_array, y.type); */
        /*     const struct mtxvector_array * y_ = &y.storage.array; */
        /*     TEST_ASSERT_EQ(mtx_field_integer, y_->field); */
        /*     TEST_ASSERT_EQ(mtx_single, y_->precision); */
        /*     TEST_ASSERT_EQ(3, y_->size); */
        /*     TEST_ASSERT_EQ(y_->data.integer_single[0], 13); */
        /*     TEST_ASSERT_EQ(y_->data.integer_single[1],  4); */
        /*     TEST_ASSERT_EQ(y_->data.integer_single[2], 14); */
        /*     mtxvector_free(&y); */
        /*     mtxvector_free(&x); */
        /* } */
        {
            int32_t xdata[] = {3, 2, 1}, ydata[] = {1, 0, 2};
            err = mtxvector_init_array_integer_single(&x, num_columns, xdata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxvector_init_array_integer_single(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dgemv(mtx_notrans, 2.0, &A, &x, 3.0, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
        /* { */
        /*     int32_t xdata[] = {3, 2, 1}, ydata[] = {1, 0, 2}; */
        /*     err = mtxvector_init_array_integer_single(&x, num_columns, xdata); */
        /*     TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err)); */
        /*     err = mtxvector_init_array_integer_single(&y, num_rows, ydata); */
        /*     TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err)); */
        /*     err = mtxmatrix_dgemv(mtx_trans, 2.0, &A, &x, 3.0, &y, NULL); */
        /*     TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err)); */
        /*     TEST_ASSERT_EQ(mtxvector_array, y.type); */
        /*     const struct mtxvector_array * y_ = &y.storage.array; */
        /*     TEST_ASSERT_EQ(mtx_field_integer, y_->field); */
        /*     TEST_ASSERT_EQ(mtx_single, y_->precision); */
        /*     TEST_ASSERT_EQ(3, y_->size); */
        /*     TEST_ASSERT_EQ(y_->data.integer_single[0], 13); */
        /*     TEST_ASSERT_EQ(y_->data.integer_single[1],  4); */
        /*     TEST_ASSERT_EQ(y_->data.integer_single[2], 14); */
        /*     mtxvector_free(&y); */
        /*     mtxvector_free(&x); */
        /* } */
        mtxmatrix_free(&A);
    }

    return TEST_SUCCESS;
}

/**
 * ‘main()’ entry point and test driver.
 */
int main(int argc, char * argv[])
{
    TEST_SUITE_BEGIN("Running tests for CSR matrices\n");
    TEST_RUN(test_mtxmatrix_csr_from_mtxfile);
    TEST_RUN(test_mtxmatrix_csr_to_mtxfile);
    TEST_RUN(test_mtxmatrix_csr_partition);
    TEST_RUN(test_mtxmatrix_csr_join);
    TEST_RUN(test_mtxmatrix_csr_gemv);
    TEST_SUITE_END();
    return (TEST_SUITE_STATUS == TEST_SUCCESS) ?
        EXIT_SUCCESS : EXIT_FAILURE;
}
