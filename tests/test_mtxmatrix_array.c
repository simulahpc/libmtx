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
 * Unit tests for (dense) matrices in array format.
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
 * ‘test_mtxmatrix_array_from_mtxfile()’ tests converting Matrix
 * Market files to matrices.
 */
int test_mtxmatrix_array_from_mtxfile(void)
{
    int err;

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
        err = mtxmatrix_from_mtxfile(&x, mtxmatrix_auto, &mtxfile);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxmatrix_array, x.type);
        const struct mtxmatrix_array * x_ = &x.storage.array;
        TEST_ASSERT_EQ(mtx_field_real, x_->field);
        TEST_ASSERT_EQ(mtx_single, x_->precision);
        TEST_ASSERT_EQ(mtx_unsymmetric, x_->symmetry);
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
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxmatrix x;
        err = mtxmatrix_from_mtxfile(&x, mtxmatrix_auto, &mtxfile);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxmatrix_array, x.type);
        const struct mtxmatrix_array * x_ = &x.storage.array;
        TEST_ASSERT_EQ(mtx_field_real, x_->field);
        TEST_ASSERT_EQ(mtx_double, x_->precision);
        TEST_ASSERT_EQ(mtx_unsymmetric, x_->symmetry);
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
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxmatrix x;
        err = mtxmatrix_from_mtxfile(&x, mtxmatrix_auto, &mtxfile);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxmatrix_array, x.type);
        const struct mtxmatrix_array * x_ = &x.storage.array;
        TEST_ASSERT_EQ(mtx_field_complex, x_->field);
        TEST_ASSERT_EQ(mtx_single, x_->precision);
        TEST_ASSERT_EQ(mtx_unsymmetric, x_->symmetry);
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
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxmatrix x;
        err = mtxmatrix_from_mtxfile(&x, mtxmatrix_auto, &mtxfile);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxmatrix_array, x.type);
        const struct mtxmatrix_array * x_ = &x.storage.array;
        TEST_ASSERT_EQ(mtx_field_complex, x_->field);
        TEST_ASSERT_EQ(mtx_double, x_->precision);
        TEST_ASSERT_EQ(mtx_unsymmetric, x_->symmetry);
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
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxmatrix x;
        err = mtxmatrix_from_mtxfile(&x, mtxmatrix_auto, &mtxfile);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxmatrix_array, x.type);
        const struct mtxmatrix_array * x_ = &x.storage.array;
        TEST_ASSERT_EQ(mtx_field_integer, x_->field);
        TEST_ASSERT_EQ(mtx_single, x_->precision);
        TEST_ASSERT_EQ(mtx_unsymmetric, x_->symmetry);
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
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxmatrix x;
        err = mtxmatrix_from_mtxfile(&x, mtxmatrix_auto, &mtxfile);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxmatrix_array, x.type);
        const struct mtxmatrix_array * x_ = &x.storage.array;
        TEST_ASSERT_EQ(mtx_field_integer, x_->field);
        TEST_ASSERT_EQ(mtx_double, x_->precision);
        TEST_ASSERT_EQ(mtx_unsymmetric, x_->symmetry);
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
    return TEST_SUCCESS;
}

/**
 * ‘test_mtxmatrix_array_to_mtxfile()’ tests converting matrices to
 * Matrix Market files.
 */
int test_mtxmatrix_array_to_mtxfile(void)
{
    int err;

    {
        struct mtxmatrix A;
        int num_rows = 3;
        int num_columns = 3;
        float Adata[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
        int Asize = sizeof(Adata) / sizeof(*Adata);
        err = mtxmatrix_init_array_real_single(
            &A, mtx_unsymmetric, num_rows, num_columns, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxfile mtxfile;
        err = mtxmatrix_to_mtxfile(&mtxfile, &A, mtxfile_array);
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
        for (int64_t k = 0; k < Asize; k++)
            TEST_ASSERT_EQ(Adata[k], data[k]);
        mtxfile_free(&mtxfile);
        mtxmatrix_free(&A);
    }

    {
        struct mtxmatrix A;
        int num_rows = 3;
        int num_columns = 3;
        float Adata[] = {1.0f, 2.0f, 3.0f, 5.0f, 6.0f, 9.0f};
        int Asize = sizeof(Adata) / sizeof(*Adata);
        err = mtxmatrix_init_array_real_single(
            &A, mtx_symmetric, num_rows, num_columns, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxfile mtxfile;
        err = mtxmatrix_to_mtxfile(&mtxfile, &A, mtxfile_array);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_matrix, mtxfile.header.object);
        TEST_ASSERT_EQ(mtxfile_array, mtxfile.header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtxfile.header.field);
        TEST_ASSERT_EQ(mtxfile_symmetric, mtxfile.header.symmetry);
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
        float Adata[] = {2.0f, 3.0f, 6.0f};
        int Asize = sizeof(Adata) / sizeof(*Adata);
        err = mtxmatrix_init_array_real_single(
            &A, mtx_skew_symmetric, num_rows, num_columns, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxfile mtxfile;
        err = mtxmatrix_to_mtxfile(&mtxfile, &A, mtxfile_array);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_matrix, mtxfile.header.object);
        TEST_ASSERT_EQ(mtxfile_array, mtxfile.header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtxfile.header.field);
        TEST_ASSERT_EQ(mtxfile_skew_symmetric, mtxfile.header.symmetry);
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
        err = mtxmatrix_init_array_real_double(
            &A, mtx_unsymmetric, num_rows, num_columns, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxfile mtxfile;
        err = mtxmatrix_to_mtxfile(&mtxfile, &A, mtxfile_array);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
        float Adata[][2] = {{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}, {7.0f, 8.0f}};
        int Asize = sizeof(Adata) / sizeof(*Adata);
        err = mtxmatrix_init_array_complex_single(
            &A, mtx_unsymmetric, num_rows, num_columns, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxfile mtxfile;
        err = mtxmatrix_to_mtxfile(&mtxfile, &A, mtxfile_array);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
        err = mtxmatrix_init_array_complex_double(
            &A, mtx_unsymmetric, num_rows, num_columns, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxfile mtxfile;
        err = mtxmatrix_to_mtxfile(&mtxfile, &A, mtxfile_array);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
        int32_t Adata[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
        int Asize = sizeof(Adata) / sizeof(*Adata);
        err = mtxmatrix_init_array_integer_single(
            &A, mtx_unsymmetric, num_rows, num_columns, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxfile mtxfile;
        err = mtxmatrix_to_mtxfile(&mtxfile, &A, mtxfile_array);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
        int64_t Adata[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
        int Asize = sizeof(Adata) / sizeof(*Adata);
        err = mtxmatrix_init_array_integer_double(
            &A, mtx_unsymmetric, num_rows, num_columns, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxfile mtxfile;
        err = mtxmatrix_to_mtxfile(&mtxfile, &A, mtxfile_array);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
    return TEST_SUCCESS;
}

/**
 * ‘test_mtxmatrix_array_partition()’ tests partitioning matrices in
 * array format.
 */
int test_mtxmatrix_array_partition(void)
{
    int err;

    {
        struct mtxmatrix src;
        int num_rows = 3;
        int num_columns = 3;
        float srcdata[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
        err = mtxmatrix_init_array_real_single(
            &src, mtx_unsymmetric, num_rows, num_columns, srcdata);
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
            TEST_ASSERT_EQ(mtxmatrix_array, dsts[0].type);
            const struct mtxmatrix_array * A = &dsts[0].storage.array;
            TEST_ASSERT_EQ(mtx_field_real, A->field);
            TEST_ASSERT_EQ(mtx_single, A->precision);
            TEST_ASSERT_EQ(2, A->num_rows);
            TEST_ASSERT_EQ(3, A->num_columns);
            TEST_ASSERT_EQ(6, A->size);
            TEST_ASSERT_EQ(A->data.real_single[0], 1.0f);
            TEST_ASSERT_EQ(A->data.real_single[1], 2.0f);
            TEST_ASSERT_EQ(A->data.real_single[2], 3.0f);
            TEST_ASSERT_EQ(A->data.real_single[3], 4.0f);
            TEST_ASSERT_EQ(A->data.real_single[4], 5.0f);
            TEST_ASSERT_EQ(A->data.real_single[5], 6.0f);
            mtxmatrix_free(&dsts[0]);
        }
        {
            TEST_ASSERT_EQ(mtxmatrix_array, dsts[1].type);
            const struct mtxmatrix_array * A = &dsts[1].storage.array;
            TEST_ASSERT_EQ(mtx_field_real, A->field);
            TEST_ASSERT_EQ(mtx_single, A->precision);
            TEST_ASSERT_EQ(1, A->num_rows);
            TEST_ASSERT_EQ(3, A->num_columns);
            TEST_ASSERT_EQ(3, A->size);
            TEST_ASSERT_EQ(A->data.real_single[0], 7.0f);
            TEST_ASSERT_EQ(A->data.real_single[1], 8.0f);
            TEST_ASSERT_EQ(A->data.real_single[2], 9.0f);
            mtxmatrix_free(&dsts[1]);
        }
        mtxpartition_free(&rowpart);
        mtxmatrix_free(&src);
    }
    return TEST_SUCCESS;
}

/**
 * ‘test_mtxmatrix_array_join()’ tests joining matrices in array
 * format.
 */
int test_mtxmatrix_array_join(void)
{
    int err;

    {
        struct mtxmatrix srcs[2];
        int num_rows[] = {2, 1};
        int num_columns[] = {3, 3};
        float srcdata0[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        float srcdata1[] = {7.0f, 8.0f, 9.0f};
        err = mtxmatrix_init_array_real_single(
            &srcs[0], mtx_unsymmetric, num_rows[0], num_columns[0], srcdata0);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxmatrix_init_array_real_single(
            &srcs[1], mtx_unsymmetric, num_rows[1], num_columns[1], srcdata1);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        int num_row_parts = 2;
        enum mtxpartitioning rowpart_type = mtx_block;
        struct mtxpartition rowpart;
        err = mtxpartition_init(
            &rowpart, rowpart_type, num_rows[0]+num_rows[1], num_row_parts, NULL, 0, NULL, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        const int num_parts = num_row_parts;
        struct mtxmatrix dst;
        err = mtxmatrix_join(&dst, srcs, &rowpart, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxmatrix_array, dst.type);
        const struct mtxmatrix_array * A = &dst.storage.array;
        TEST_ASSERT_EQ(mtx_field_real, A->field);
        TEST_ASSERT_EQ(mtx_single, A->precision);
        TEST_ASSERT_EQ(3, A->num_rows);
        TEST_ASSERT_EQ(3, A->num_columns);
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
        mtxmatrix_free(&dst);
        mtxpartition_free(&rowpart);
        mtxmatrix_free(&srcs[1]);
        mtxmatrix_free(&srcs[0]);
    }
    return TEST_SUCCESS;
}

/**
 * ‘test_mtxmatrix_array_gemv()’ tests computing matrix-vector
 * products for matrices in array format.
 */
int test_mtxmatrix_array_gemv(void)
{
    int err;

    /*
     * a) For unsymmetric real or integer matrices, calculate
     *
     *   ⎡ 1 2 3⎤  ⎡ 3⎤     ⎡ 1⎤  ⎡ 20⎤  ⎡ 3⎤  ⎡ 23⎤
     * 2*⎢ 4 5 6⎥ *⎢ 2⎥ + 3*⎢ 0⎥ =⎢ 56⎥ +⎢ 0⎥ =⎢ 56⎥,
     *   ⎣ 7 8 9⎦  ⎣ 1⎦     ⎣ 2⎦  ⎣ 92⎦  ⎣ 6⎦  ⎣ 98⎦
     *
     * and the transposed product
     *
     *   ⎡ 1 4 7⎤  ⎡ 3⎤     ⎡ 1⎤  ⎡ 36⎤  ⎡ 3⎤  ⎡ 39⎤
     * 2*⎢ 2 5 8⎥ *⎢ 2⎥ + 3*⎢ 0⎥ =⎢ 48⎥ +⎢ 0⎥ =⎢ 48⎥.
     *   ⎣ 3 6 9⎦  ⎣ 1⎦     ⎣ 2⎦  ⎣ 60⎦  ⎣ 6⎦  ⎣ 66⎦
     *
     * b) For symmetric real or integer matrices, calculate
     *
     *   ⎡ 1 2 3⎤  ⎡ 3⎤     ⎡ 1⎤  ⎡ 20⎤  ⎡ 3⎤  ⎡ 23⎤
     * 2*⎢ 2 5 6⎥ *⎢ 2⎥ + 3*⎢ 0⎥ =⎢ 44⎥ +⎢ 0⎥ =⎢ 44⎥,
     *   ⎣ 3 6 9⎦  ⎣ 1⎦     ⎣ 2⎦  ⎣ 60⎦  ⎣ 6⎦  ⎣ 66⎦
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
        float Adata[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
        float xdata[] = {3.0f, 2.0f, 1.0f};
        float ydata[] = {1.0f, 0.0f, 2.0f};
        err = mtxmatrix_init_array_real_single(
            &A, mtx_unsymmetric, num_rows, num_columns, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_init_array_real_single(&x, num_columns, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
            TEST_ASSERT_EQ(y_->data.real_single[0], 23.0f);
            TEST_ASSERT_EQ(y_->data.real_single[1], 56.0f);
            TEST_ASSERT_EQ(y_->data.real_single[2], 98.0f);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_array_real_single(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemv(mtx_trans, 2.0f, &A, &x, 3.0f, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dgemv(mtx_notrans, 2.0, &A, &x, 3.0, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dgemv(mtx_trans, 2.0, &A, &x, 3.0, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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

    /*
     * Real, single precision, symmetric matrices
     */

    {
        struct mtxmatrix A;
        struct mtxvector x;
        struct mtxvector y;
        int num_rows = 3;
        int num_columns = 3;
        float Adata[] = {1.0f, 2.0f, 3.0f, 5.0f, 6.0f, 9.0f};
        float xdata[] = {3.0f, 2.0f, 1.0f};
        float ydata[] = {1.0f, 0.0f, 2.0f};
        err = mtxmatrix_init_array_real_single(
            &A, mtx_symmetric, num_rows, num_columns, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_init_array_real_single(&x, num_columns, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
            TEST_ASSERT_EQ(y_->data.real_single[0], 23.0f);
            TEST_ASSERT_EQ(y_->data.real_single[1], 44.0f);
            TEST_ASSERT_EQ(y_->data.real_single[2], 66.0f);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_array_real_single(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemv(mtx_trans, 2.0f, &A, &x, 3.0f, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
            TEST_ASSERT_EQ(mtx_field_real, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(3, y_->size);
            TEST_ASSERT_EQ(y_->data.real_single[0], 23.0f);
            TEST_ASSERT_EQ(y_->data.real_single[1], 44.0f);
            TEST_ASSERT_EQ(y_->data.real_single[2], 66.0f);
            mtxvector_free(&y);
        }
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
            TEST_ASSERT_EQ(y_->data.real_single[0], 23.0f);
            TEST_ASSERT_EQ(y_->data.real_single[1], 44.0f);
            TEST_ASSERT_EQ(y_->data.real_single[2], 66.0f);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_array_real_single(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dgemv(mtx_trans, 2.0, &A, &x, 3.0, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
            TEST_ASSERT_EQ(mtx_field_real, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(3, y_->size);
            TEST_ASSERT_EQ(y_->data.real_single[0], 23.0f);
            TEST_ASSERT_EQ(y_->data.real_single[1], 44.0f);
            TEST_ASSERT_EQ(y_->data.real_single[2], 66.0f);
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
        double Adata[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
        double xdata[] = {3.0, 2.0, 1.0};
        double ydata[] = {1.0, 0.0, 2.0};
        err = mtxmatrix_init_array_real_double(
            &A, mtx_unsymmetric, num_rows, num_columns, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_init_array_real_double(&x, num_columns, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
            TEST_ASSERT_EQ(y_->data.real_double[0], 23.0);
            TEST_ASSERT_EQ(y_->data.real_double[1], 56.0);
            TEST_ASSERT_EQ(y_->data.real_double[2], 98.0);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_array_real_double(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemv(mtx_trans, 2.0f, &A, &x, 3.0f, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dgemv(mtx_notrans, 2.0, &A, &x, 3.0, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dgemv(mtx_trans, 2.0, &A, &x, 3.0, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
     * Real, double precision, symmetric matrices
     */

    {
        struct mtxmatrix A;
        struct mtxvector x;
        struct mtxvector y;
        int num_rows = 3;
        int num_columns = 3;
        double Adata[] = {1.0, 2.0, 3.0, 5.0, 6.0, 9.0};
        double xdata[] = {3.0, 2.0, 1.0};
        double ydata[] = {1.0, 0.0, 2.0};
        err = mtxmatrix_init_array_real_double(
            &A, mtx_symmetric, num_rows, num_columns, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_init_array_real_double(&x, num_columns, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
            TEST_ASSERT_EQ(y_->data.real_double[0], 23.0);
            TEST_ASSERT_EQ(y_->data.real_double[1], 44.0);
            TEST_ASSERT_EQ(y_->data.real_double[2], 66.0);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_array_real_double(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemv(mtx_trans, 2.0f, &A, &x, 3.0f, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
            TEST_ASSERT_EQ(mtx_field_real, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(3, y_->size);
            TEST_ASSERT_EQ(y_->data.real_double[0], 23.0);
            TEST_ASSERT_EQ(y_->data.real_double[1], 44.0);
            TEST_ASSERT_EQ(y_->data.real_double[2], 66.0);
            mtxvector_free(&y);
        }
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
            TEST_ASSERT_EQ(y_->data.real_double[0], 23.0);
            TEST_ASSERT_EQ(y_->data.real_double[1], 44.0);
            TEST_ASSERT_EQ(y_->data.real_double[2], 66.0);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_array_real_double(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dgemv(mtx_trans, 2.0, &A, &x, 3.0, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
            TEST_ASSERT_EQ(mtx_field_real, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(3, y_->size);
            TEST_ASSERT_EQ(y_->data.real_double[0], 23.0);
            TEST_ASSERT_EQ(y_->data.real_double[1], 44.0);
            TEST_ASSERT_EQ(y_->data.real_double[2], 66.0);
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
        float Adata[][2] = {{1.0f,2.0f}, {3.0f,4.0f}, {5.0f,6.0f}, {7.0f,8.0f}};
        float xdata[][2] = {{3.0f,1.0f}, {1.0f,2.0f}};
        float ydata[][2] = {{1.0f,0.0f}, {2.0f,2.0f}};
        err = mtxmatrix_init_array_complex_single(
            &A, mtx_unsymmetric, num_rows, num_columns, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_init_array_complex_single(&x, num_columns, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
            TEST_ASSERT_EQ(y_->data.complex_single[1][0],  6.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1], 96.0f);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_array_complex_single(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemv(mtx_trans, 2.0f, &A, &x, 3.0f, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemv(mtx_conjtrans, 2.0f, &A, &x, 3.0f, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
            TEST_ASSERT_EQ(y_->data.complex_single[1][0],  6.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1], 96.0f);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_array_complex_single(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dgemv(mtx_trans, 2.0, &A, &x, 3.0, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dgemv(mtx_conjtrans, 2.0, &A, &x, 3.0, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_cgemv(mtx_notrans, calpha, &A, &x, cbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_cgemv(mtx_trans, calpha, &A, &x, cbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_cgemv(mtx_conjtrans, calpha, &A, &x, cbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
            TEST_ASSERT_EQ(y_->data.complex_single[1][0], -86.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1],   8.0f);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_array_complex_single(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_zgemv(mtx_trans, zalpha, &A, &x, zbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_zgemv(mtx_conjtrans, zalpha, &A, &x, zbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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

    /*
     * Complex, single precision, symmetric matrices
     */

    {
        struct mtxmatrix A;
        struct mtxvector x;
        struct mtxvector y;
        int num_rows = 2;
        int num_columns = 2;
        float Adata[][2] = {{1.0f,2.0f}, {3.0f,4.0f}, {7.0f,8.0f}};
        float xdata[][2] = {{3.0f,1.0f}, {1.0f,2.0f}};
        float ydata[][2] = {{1.0f,0.0f}, {2.0f,2.0f}};
        err = mtxmatrix_init_array_complex_single(
            &A, mtx_symmetric, num_rows, num_columns, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_init_array_complex_single(&x, num_columns, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
            TEST_ASSERT_EQ(y_->data.complex_single[1][0], -2.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1], 80.0f);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_array_complex_single(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemv(mtx_trans, 2.0f, &A, &x, 3.0f, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
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
            err = mtxvector_init_array_complex_single(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemv(mtx_conjtrans, 2.0f, &A, &x, 3.0f, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
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
            TEST_ASSERT_EQ(y_->data.complex_single[1][0], -2.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1], 80.0f);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_array_complex_single(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dgemv(mtx_trans, 2.0, &A, &x, 3.0, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
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
            err = mtxvector_init_array_complex_single(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dgemv(mtx_conjtrans, 2.0, &A, &x, 3.0, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
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
            err = mtxvector_init_array_complex_single(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_cgemv(mtx_notrans, calpha, &A, &x, cbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
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
            err = mtxvector_init_array_complex_single(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_cgemv(mtx_trans, calpha, &A, &x, cbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
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
            err = mtxvector_init_array_complex_single(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_cgemv(mtx_conjtrans, calpha, &A, &x, cbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
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
            TEST_ASSERT_EQ(y_->data.complex_single[1][0], -70.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1],   0.0f);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_array_complex_single(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_zgemv(mtx_trans, zalpha, &A, &x, zbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
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
            err = mtxvector_init_array_complex_single(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_zgemv(mtx_conjtrans, zalpha, &A, &x, zbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
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
        float Adata[][2] = {{1.0f,0.0f}, {3.0f,4.0f}, {7.0f,0.0f}};
        float xdata[][2] = {{3.0f,1.0f}, {1.0f,2.0f}};
        float ydata[][2] = {{1.0f,0.0f}, {2.0f,2.0f}};
        err = mtxmatrix_init_array_complex_single(
            &A, mtx_hermitian, num_rows, num_columns, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_init_array_complex_single(&x, num_columns, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
            TEST_ASSERT_EQ(y_->data.complex_single[0][0], -1.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[0][1], 22.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][0], 46.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1], 16.0f);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_array_complex_single(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemv(mtx_trans, 2.0f, &A, &x, 3.0f, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
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
            err = mtxvector_init_array_complex_single(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemv(mtx_conjtrans, 2.0f, &A, &x, 3.0f, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
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
            err = mtxvector_init_array_complex_single(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dgemv(mtx_notrans, 2.0, &A, &x, 3.0, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
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
            err = mtxvector_init_array_complex_single(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dgemv(mtx_trans, 2.0, &A, &x, 3.0, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
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
            err = mtxvector_init_array_complex_single(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dgemv(mtx_conjtrans, 2.0, &A, &x, 3.0, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
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
            err = mtxvector_init_array_complex_single(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_cgemv(mtx_notrans, calpha, &A, &x, cbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
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
            err = mtxvector_init_array_complex_single(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_cgemv(mtx_trans, calpha, &A, &x, cbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
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
            err = mtxvector_init_array_complex_single(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_cgemv(mtx_conjtrans, calpha, &A, &x, cbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
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
            err = mtxvector_init_array_complex_single(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_zgemv(mtx_notrans, zalpha, &A, &x, zbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
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
            err = mtxvector_init_array_complex_single(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_zgemv(mtx_trans, zalpha, &A, &x, zbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
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
            err = mtxvector_init_array_complex_single(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_zgemv(mtx_conjtrans, zalpha, &A, &x, zbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
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
        double Adata[][2] = {{1.0,2.0}, {3.0,4.0}, {5.0,6.0}, {7.0,8.0}};
        double xdata[][2] = {{3.0,1.0}, {1.0,2.0}};
        double ydata[][2] = {{1.0,0.0}, {2.0,2.0}};
        err = mtxmatrix_init_array_complex_double(
            &A, mtx_unsymmetric, num_rows, num_columns, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_init_array_complex_double(&x, num_columns, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
            TEST_ASSERT_EQ(y_->data.complex_double[0][0], -5.0);
            TEST_ASSERT_EQ(y_->data.complex_double[0][1], 34.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][0],  6.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][1], 96.0);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_array_complex_double(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemv(mtx_trans, 2.0f, &A, &x, 3.0f, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemv(mtx_conjtrans, 2.0f, &A, &x, 3.0f, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
            TEST_ASSERT_EQ(y_->data.complex_double[1][0],  6.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][1], 96.0);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_array_complex_double(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dgemv(mtx_trans, 2.0, &A, &x, 3.0, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dgemv(mtx_conjtrans, 2.0, &A, &x, 3.0, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_cgemv(mtx_notrans, calpha, &A, &x, cbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_cgemv(mtx_trans, calpha, &A, &x, cbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_cgemv(mtx_conjtrans, calpha, &A, &x, cbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_zgemv(mtx_notrans, zalpha, &A, &x, zbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_zgemv(mtx_trans, zalpha, &A, &x, zbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_zgemv(mtx_conjtrans, zalpha, &A, &x, zbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
     * Complex, double precision, symmetric matrices
     */

    {
        struct mtxmatrix A;
        struct mtxvector x;
        struct mtxvector y;
        int num_rows = 2;
        int num_columns = 2;
        double Adata[][2] = {{1.0,2.0}, {3.0,4.0}, {7.0,8.0}};
        double xdata[][2] = {{3.0,1.0}, {1.0,2.0}};
        double ydata[][2] = {{1.0,0.0}, {2.0,2.0}};
        err = mtxmatrix_init_array_complex_double(
            &A, mtx_symmetric, num_rows, num_columns, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_init_array_complex_double(&x, num_columns, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
            TEST_ASSERT_EQ(y_->data.complex_double[0][0], -5.0);
            TEST_ASSERT_EQ(y_->data.complex_double[0][1], 34.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][0], -2.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][1], 80.0);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_array_complex_double(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemv(mtx_trans, 2.0f, &A, &x, 3.0f, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
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
            err = mtxvector_init_array_complex_double(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemv(mtx_conjtrans, 2.0f, &A, &x, 3.0f, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
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
            TEST_ASSERT_EQ(y_->data.complex_double[1][0], -2.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][1], 80.0);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_array_complex_double(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dgemv(mtx_trans, 2.0, &A, &x, 3.0, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
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
            err = mtxvector_init_array_complex_double(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dgemv(mtx_conjtrans, 2.0, &A, &x, 3.0, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
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
            err = mtxvector_init_array_complex_double(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_cgemv(mtx_notrans, calpha, &A, &x, cbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
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
            err = mtxvector_init_array_complex_double(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_cgemv(mtx_trans, calpha, &A, &x, cbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
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
            err = mtxvector_init_array_complex_double(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_cgemv(mtx_conjtrans, calpha, &A, &x, cbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
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
            err = mtxvector_init_array_complex_double(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_zgemv(mtx_notrans, zalpha, &A, &x, zbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
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
            err = mtxvector_init_array_complex_double(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_zgemv(mtx_trans, zalpha, &A, &x, zbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
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
            err = mtxvector_init_array_complex_double(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_zgemv(mtx_conjtrans, zalpha, &A, &x, zbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
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
        double Adata[][2] = {{1.0,0.0}, {3.0,4.0}, {7.0,0.0}};
        double xdata[][2] = {{3.0,1.0}, {1.0,2.0}};
        double ydata[][2] = {{1.0,0.0}, {2.0,2.0}};
        err = mtxmatrix_init_array_complex_double(
            &A, mtx_hermitian, num_rows, num_columns, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_init_array_complex_double(&x, num_columns, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
            TEST_ASSERT_EQ(y_->data.complex_double[0][0], -1.0);
            TEST_ASSERT_EQ(y_->data.complex_double[0][1], 22.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][0], 46.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][1], 16.0);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_array_complex_double(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemv(mtx_trans, 2.0f, &A, &x, 3.0f, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
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
            err = mtxvector_init_array_complex_double(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemv(mtx_conjtrans, 2.0f, &A, &x, 3.0f, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
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
            err = mtxvector_init_array_complex_double(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dgemv(mtx_notrans, 2.0, &A, &x, 3.0, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
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
            err = mtxvector_init_array_complex_double(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dgemv(mtx_trans, 2.0, &A, &x, 3.0, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
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
            err = mtxvector_init_array_complex_double(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dgemv(mtx_conjtrans, 2.0, &A, &x, 3.0, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
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
            err = mtxvector_init_array_complex_double(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_cgemv(mtx_notrans, calpha, &A, &x, cbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
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
            err = mtxvector_init_array_complex_double(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_cgemv(mtx_trans, calpha, &A, &x, cbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
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
            err = mtxvector_init_array_complex_double(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_cgemv(mtx_conjtrans, calpha, &A, &x, cbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
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
            err = mtxvector_init_array_complex_double(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_zgemv(mtx_notrans, zalpha, &A, &x, zbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
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
            err = mtxvector_init_array_complex_double(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_zgemv(mtx_trans, zalpha, &A, &x, zbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
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
            err = mtxvector_init_array_complex_double(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_zgemv(mtx_conjtrans, zalpha, &A, &x, zbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
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
        int32_t Adata[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
        int32_t xdata[] = {3, 2, 1};
        int32_t ydata[] = {1, 0, 2};
        err = mtxmatrix_init_array_integer_single(
            &A, mtx_unsymmetric, num_rows, num_columns, Adata);
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
            TEST_ASSERT_EQ(y_->data.integer_single[0], 23);
            TEST_ASSERT_EQ(y_->data.integer_single[1], 56);
            TEST_ASSERT_EQ(y_->data.integer_single[2], 98);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_array_integer_single(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemv(mtx_trans, 2.0f, &A, &x, 3.0f, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dgemv(mtx_notrans, 2.0, &A, &x, 3.0, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dgemv(mtx_trans, 2.0, &A, &x, 3.0, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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

    /*
     * Integer, single precision, symmetric matrices
     */

    {
        struct mtxmatrix A;
        struct mtxvector x;
        struct mtxvector y;
        int num_rows = 3;
        int num_columns = 3;
        int32_t Adata[] = {1, 2, 3, 5, 6, 9};
        int32_t xdata[] = {3, 2, 1};
        int32_t ydata[] = {1, 0, 2};
        err = mtxmatrix_init_array_integer_single(
            &A, mtx_symmetric, num_rows, num_columns, Adata);
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
            TEST_ASSERT_EQ(y_->data.integer_single[0], 23);
            TEST_ASSERT_EQ(y_->data.integer_single[1], 44);
            TEST_ASSERT_EQ(y_->data.integer_single[2], 66);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_array_integer_single(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemv(mtx_trans, 2.0f, &A, &x, 3.0f, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
            TEST_ASSERT_EQ(mtx_field_integer, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(3, y_->size);
            TEST_ASSERT_EQ(y_->data.integer_single[0], 23);
            TEST_ASSERT_EQ(y_->data.integer_single[1], 44);
            TEST_ASSERT_EQ(y_->data.integer_single[2], 66);
            mtxvector_free(&y);
        }
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
            TEST_ASSERT_EQ(y_->data.integer_single[0], 23);
            TEST_ASSERT_EQ(y_->data.integer_single[1], 44);
            TEST_ASSERT_EQ(y_->data.integer_single[2], 66);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_array_integer_single(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dgemv(mtx_trans, 2.0, &A, &x, 3.0, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
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
        int64_t Adata[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
        int64_t xdata[] = {3, 2, 1};
        int64_t ydata[] = {1, 0, 2};
        err = mtxmatrix_init_array_integer_double(
            &A, mtx_unsymmetric, num_rows, num_columns, Adata);
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
            TEST_ASSERT_EQ(y_->data.integer_double[0], 23);
            TEST_ASSERT_EQ(y_->data.integer_double[1], 56);
            TEST_ASSERT_EQ(y_->data.integer_double[2], 98);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_array_integer_double(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemv(mtx_trans, 2.0f, &A, &x, 3.0f, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dgemv(mtx_notrans, 2.0, &A, &x, 3.0, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dgemv(mtx_trans, 2.0, &A, &x, 3.0, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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

    /*
     * Integer, double precision, symmetric matrices
     */

    {
        struct mtxmatrix A;
        struct mtxvector x;
        struct mtxvector y;
        int num_rows = 3;
        int num_columns = 3;
        int64_t Adata[] = {1, 2, 3, 5, 6, 9};
        int64_t xdata[] = {3, 2, 1};
        int64_t ydata[] = {1, 0, 2};
        err = mtxmatrix_init_array_integer_double(
            &A, mtx_symmetric, num_rows, num_columns, Adata);
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
            TEST_ASSERT_EQ(y_->data.integer_double[0], 23);
            TEST_ASSERT_EQ(y_->data.integer_double[1], 44);
            TEST_ASSERT_EQ(y_->data.integer_double[2], 66);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_array_integer_double(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemv(mtx_trans, 2.0f, &A, &x, 3.0f, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
            TEST_ASSERT_EQ(mtx_field_integer, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(3, y_->size);
            TEST_ASSERT_EQ(y_->data.integer_double[0], 23);
            TEST_ASSERT_EQ(y_->data.integer_double[1], 44);
            TEST_ASSERT_EQ(y_->data.integer_double[2], 66);
            mtxvector_free(&y);
        }
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
            TEST_ASSERT_EQ(y_->data.integer_double[0], 23);
            TEST_ASSERT_EQ(y_->data.integer_double[1], 44);
            TEST_ASSERT_EQ(y_->data.integer_double[2], 66);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_array_integer_double(&y, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dgemv(mtx_trans, 2.0, &A, &x, 3.0, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.type);
            const struct mtxvector_array * y_ = &y.storage.array;
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
    TEST_SUITE_BEGIN("Running tests for array (dense) matrices\n");
    TEST_RUN(test_mtxmatrix_array_from_mtxfile);
    TEST_RUN(test_mtxmatrix_array_to_mtxfile);
    TEST_RUN(test_mtxmatrix_array_partition);
    TEST_RUN(test_mtxmatrix_array_join);
    TEST_RUN(test_mtxmatrix_array_gemv);
    TEST_SUITE_END();
    return (TEST_SUITE_STATUS == TEST_SUCCESS) ?
        EXIT_SUCCESS : EXIT_FAILURE;
}
