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
 * TODO: The following tests need to be converted from coordinate
 * matrices to CSR matrices:
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
    return TEST_FAILURE;
}

/**
 * ‘test_mtxmatrix_csr_to_mtxfile()’ tests converting matrices to
 * Matrix Market files.
 */
int test_mtxmatrix_csr_to_mtxfile(void)
{
    return TEST_FAILURE;
}

/**
 * ‘test_mtxmatrix_csr_partition()’ tests partitioning matrices in
 * coordinate format.
 */
int test_mtxmatrix_csr_partition(void)
{
    return TEST_FAILURE;
}

/**
 * ‘test_mtxmatrix_csr_join()’ tests joining matrices in coordinate
 * format.
 */
int test_mtxmatrix_csr_join(void)
{
    return TEST_FAILURE;
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
            &A, num_rows, num_columns, Arowptr, Acolidx, Adata);
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
            &A, num_rows, num_columns, Arowptr, Acolidx, Adata);
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
            &A, num_rows, num_columns, Arowptr, Acolidx, Adata);
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
            &A, num_rows, num_columns, Arowptr, Acolidx, Adata);
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
            &A, num_rows, num_columns, Arowptr, Acolidx, Adata);
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
            &A, num_rows, num_columns, Arowptr, Acolidx, Adata);
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
            &A, num_rows, num_columns, Arowptr, Acolidx);
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
    /* TEST_RUN(test_mtxmatrix_csr_from_mtxfile); */
    /* TEST_RUN(test_mtxmatrix_csr_to_mtxfile); */
    /* TEST_RUN(test_mtxmatrix_csr_partition); */
    /* TEST_RUN(test_mtxmatrix_csr_join); */
    TEST_RUN(test_mtxmatrix_csr_gemv);
    TEST_SUITE_END();
    return (TEST_SUITE_STATUS == TEST_SUCCESS) ?
        EXIT_SUCCESS : EXIT_FAILURE;
}
