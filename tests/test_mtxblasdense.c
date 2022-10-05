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
 * Last modified: 2022-10-05
 *
 * Unit tests for dense matrices with BLAS-accelerated operations.
 */

#include "test.h"

#include <libmtx/error.h>
#include <libmtx/linalg/blas/dense.h>
#include <libmtx/linalg/local/matrix.h>
#include <libmtx/mtxfile/mtxfile.h>
#include <libmtx/linalg/local/vector.h>

#include <errno.h>
#include <unistd.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * ‘test_mtxblasdense_gemv()’ tests computing matrix-vector
 * products for matrices in coordinate format.
 */
int test_mtxblasdense_gemv(void)
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
            &A, mtxblasdense, mtx_unsymmetric, num_rows, num_columns, num_nonzeros, Arowidx, Acolidx, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_init_real_single(&x, mtxbasevector, num_columns, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        {
            err = mtxvector_init_real_single(&y, mtxbasevector, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemv(mtx_notrans, 2.0f, &A, &x, 3.0f, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxbasevector, y.type);
            const struct mtxbasevector * y_ = &y.storage.base;
            TEST_ASSERT_EQ(mtx_field_real, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(3, y_->size);
            TEST_ASSERT_EQ(y_->data.real_single[0], 17.0f);
            TEST_ASSERT_EQ(y_->data.real_single[1], 56.0f);
            TEST_ASSERT_EQ(y_->data.real_single[2], 98.0f);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_real_single(&y, mtxbasevector, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemv(mtx_trans, 2.0f, &A, &x, 3.0f, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxbasevector, y.type);
            const struct mtxbasevector * y_ = &y.storage.base;
            TEST_ASSERT_EQ(mtx_field_real, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(3, y_->size);
            TEST_ASSERT_EQ(y_->data.real_single[0], 39.0f);
            TEST_ASSERT_EQ(y_->data.real_single[1], 48.0f);
            TEST_ASSERT_EQ(y_->data.real_single[2], 48.0f);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_real_single(&y, mtxbasevector, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dgemv(mtx_notrans, 2.0, &A, &x, 3.0, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxbasevector, y.type);
            const struct mtxbasevector * y_ = &y.storage.base;
            TEST_ASSERT_EQ(mtx_field_real, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(3, y_->size);
            TEST_ASSERT_EQ(y_->data.real_single[0], 17.0f);
            TEST_ASSERT_EQ(y_->data.real_single[1], 56.0f);
            TEST_ASSERT_EQ(y_->data.real_single[2], 98.0f);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_real_single(&y, mtxbasevector, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dgemv(mtx_trans, 2.0, &A, &x, 3.0, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxbasevector, y.type);
            const struct mtxbasevector * y_ = &y.storage.base;
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
            &A, mtxblasdense, mtx_symmetric, num_rows, num_columns, num_nonzeros, Arowidx, Acolidx, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_init_real_single(&x, mtxbasevector, num_columns, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        {
            err = mtxvector_init_real_single(&y, mtxbasevector, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemv(mtx_notrans, 2.0f, &A, &x, 3.0f, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxbasevector, y.type);
            const struct mtxbasevector * y_ = &y.storage.base;
            TEST_ASSERT_EQ(mtx_field_real, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(3, y_->size);
            TEST_ASSERT_EQ(y_->data.real_single[0], 17.0f);
            TEST_ASSERT_EQ(y_->data.real_single[1], 44.0f);
            TEST_ASSERT_EQ(y_->data.real_single[2], 48.0f);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_real_single(&y, mtxbasevector, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemv(mtx_trans, 2.0f, &A, &x, 3.0f, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxbasevector, y.type);
            const struct mtxbasevector * y_ = &y.storage.base;
            TEST_ASSERT_EQ(mtx_field_real, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(3, y_->size);
            TEST_ASSERT_EQ(y_->data.real_single[0], 17.0f);
            TEST_ASSERT_EQ(y_->data.real_single[1], 44.0f);
            TEST_ASSERT_EQ(y_->data.real_single[2], 48.0f);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_real_single(&y, mtxbasevector, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dgemv(mtx_notrans, 2.0, &A, &x, 3.0, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxbasevector, y.type);
            const struct mtxbasevector * y_ = &y.storage.base;
            TEST_ASSERT_EQ(mtx_field_real, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(3, y_->size);
            TEST_ASSERT_EQ(y_->data.real_single[0], 17.0f);
            TEST_ASSERT_EQ(y_->data.real_single[1], 44.0f);
            TEST_ASSERT_EQ(y_->data.real_single[2], 48.0f);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_real_single(&y, mtxbasevector, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dgemv(mtx_trans, 2.0, &A, &x, 3.0, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxbasevector, y.type);
            const struct mtxbasevector * y_ = &y.storage.base;
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
            &A, mtxblasdense, mtx_unsymmetric, num_rows, num_columns, num_nonzeros, Arowidx, Acolidx, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_init_real_double(&x, mtxbasevector, num_columns, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        {
            err = mtxvector_init_real_double(&y, mtxbasevector, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemv(mtx_notrans, 2.0f, &A, &x, 3.0f, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxbasevector, y.type);
            const struct mtxbasevector * y_ = &y.storage.base;
            TEST_ASSERT_EQ(mtx_field_real, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(3, y_->size);
            TEST_ASSERT_EQ(y_->data.real_double[0], 17.0);
            TEST_ASSERT_EQ(y_->data.real_double[1], 56.0);
            TEST_ASSERT_EQ(y_->data.real_double[2], 98.0);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_real_double(&y, mtxbasevector, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemv(mtx_trans, 2.0f, &A, &x, 3.0f, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxbasevector, y.type);
            const struct mtxbasevector * y_ = &y.storage.base;
            TEST_ASSERT_EQ(mtx_field_real, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(3, y_->size);
            TEST_ASSERT_EQ(y_->data.real_double[0], 39.0);
            TEST_ASSERT_EQ(y_->data.real_double[1], 48.0);
            TEST_ASSERT_EQ(y_->data.real_double[2], 48.0);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_real_double(&y, mtxbasevector, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dgemv(mtx_notrans, 2.0, &A, &x, 3.0, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxbasevector, y.type);
            const struct mtxbasevector * y_ = &y.storage.base;
            TEST_ASSERT_EQ(mtx_field_real, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(3, y_->size);
            TEST_ASSERT_EQ(y_->data.real_double[0], 17.0);
            TEST_ASSERT_EQ(y_->data.real_double[1], 56.0);
            TEST_ASSERT_EQ(y_->data.real_double[2], 98.0);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_real_double(&y, mtxbasevector, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dgemv(mtx_trans, 2.0, &A, &x, 3.0, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxbasevector, y.type);
            const struct mtxbasevector * y_ = &y.storage.base;
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
            &A, mtxblasdense, mtx_symmetric, num_rows, num_columns, num_nonzeros, Arowidx, Acolidx, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_init_real_double(&x, mtxbasevector, num_columns, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        {
            err = mtxvector_init_real_double(&y, mtxbasevector, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemv(mtx_notrans, 2.0f, &A, &x, 3.0f, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxbasevector, y.type);
            const struct mtxbasevector * y_ = &y.storage.base;
            TEST_ASSERT_EQ(mtx_field_real, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(3, y_->size);
            TEST_ASSERT_EQ(y_->data.real_double[0], 17.0);
            TEST_ASSERT_EQ(y_->data.real_double[1], 44.0);
            TEST_ASSERT_EQ(y_->data.real_double[2], 48.0);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_real_double(&y, mtxbasevector, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemv(mtx_trans, 2.0f, &A, &x, 3.0f, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxbasevector, y.type);
            const struct mtxbasevector * y_ = &y.storage.base;
            TEST_ASSERT_EQ(mtx_field_real, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(3, y_->size);
            TEST_ASSERT_EQ(y_->data.real_double[0], 17.0);
            TEST_ASSERT_EQ(y_->data.real_double[1], 44.0);
            TEST_ASSERT_EQ(y_->data.real_double[2], 48.0);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_real_double(&y, mtxbasevector, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dgemv(mtx_notrans, 2.0, &A, &x, 3.0, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxbasevector, y.type);
            const struct mtxbasevector * y_ = &y.storage.base;
            TEST_ASSERT_EQ(mtx_field_real, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(3, y_->size);
            TEST_ASSERT_EQ(y_->data.real_double[0], 17.0);
            TEST_ASSERT_EQ(y_->data.real_double[1], 44.0);
            TEST_ASSERT_EQ(y_->data.real_double[2], 48.0);
            mtxvector_free(&y);
        }
        {
            err = mtxvector_init_real_double(&y, mtxbasevector, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dgemv(mtx_trans, 2.0, &A, &x, 3.0, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxbasevector, y.type);
            const struct mtxbasevector * y_ = &y.storage.base;
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
            &A, mtxblasdense, mtx_unsymmetric, num_rows, num_columns, num_nonzeros, Arowidx, Acolidx, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_init_complex_single(&x, mtxbasevector, num_columns, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        {
            err = mtxvector_init_complex_single(&y, mtxbasevector, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemv(mtx_notrans, 2.0f, &A, &x, 3.0f, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxbasevector, y.type);
            const struct mtxbasevector * y_ = &y.storage.base;
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
            err = mtxvector_init_complex_single(&y, mtxbasevector, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemv(mtx_trans, 2.0f, &A, &x, 3.0f, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxbasevector, y.type);
            const struct mtxbasevector * y_ = &y.storage.base;
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
            err = mtxvector_init_complex_single(&y, mtxbasevector, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemv(mtx_conjtrans, 2.0f, &A, &x, 3.0f, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxbasevector, y.type);
            const struct mtxbasevector * y_ = &y.storage.base;
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
            err = mtxvector_init_complex_single(&y, mtxbasevector, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dgemv(mtx_notrans, 2.0, &A, &x, 3.0, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxbasevector, y.type);
            const struct mtxbasevector * y_ = &y.storage.base;
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
            err = mtxvector_init_complex_single(&y, mtxbasevector, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dgemv(mtx_trans, 2.0, &A, &x, 3.0, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxbasevector, y.type);
            const struct mtxbasevector * y_ = &y.storage.base;
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
            err = mtxvector_init_complex_single(&y, mtxbasevector, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dgemv(mtx_conjtrans, 2.0, &A, &x, 3.0, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxbasevector, y.type);
            const struct mtxbasevector * y_ = &y.storage.base;
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
            err = mtxvector_init_complex_single(&y, mtxbasevector, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_cgemv(mtx_notrans, calpha, &A, &x, cbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxbasevector, y.type);
            const struct mtxbasevector * y_ = &y.storage.base;
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
            err = mtxvector_init_complex_single(&y, mtxbasevector, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_cgemv(mtx_trans, calpha, &A, &x, cbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxbasevector, y.type);
            const struct mtxbasevector * y_ = &y.storage.base;
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
            err = mtxvector_init_complex_single(&y, mtxbasevector, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_cgemv(mtx_conjtrans, calpha, &A, &x, cbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxbasevector, y.type);
            const struct mtxbasevector * y_ = &y.storage.base;
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
            err = mtxvector_init_complex_single(&y, mtxbasevector, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_zgemv(mtx_notrans, zalpha, &A, &x, zbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxbasevector, y.type);
            const struct mtxbasevector * y_ = &y.storage.base;
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
            err = mtxvector_init_complex_single(&y, mtxbasevector, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_zgemv(mtx_trans, zalpha, &A, &x, zbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxbasevector, y.type);
            const struct mtxbasevector * y_ = &y.storage.base;
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
            err = mtxvector_init_complex_single(&y, mtxbasevector, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_zgemv(mtx_conjtrans, zalpha, &A, &x, zbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxbasevector, y.type);
            const struct mtxbasevector * y_ = &y.storage.base;
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
            &A, mtxblasdense, mtx_hermitian, num_rows, num_columns, num_nonzeros, Arowidx, Acolidx, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_init_complex_single(&x, mtxbasevector, num_columns, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        {
            err = mtxvector_init_complex_single(&y, mtxbasevector, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemv(mtx_notrans, 2.0f, &A, &x, 3.0f, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxbasevector, y.type);
            const struct mtxbasevector * y_ = &y.storage.base;
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
            err = mtxvector_init_complex_single(&y, mtxbasevector, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemv(mtx_conjtrans, 2.0f, &A, &x, 3.0f, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxbasevector, y.type);
            const struct mtxbasevector * y_ = &y.storage.base;
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
            err = mtxvector_init_complex_single(&y, mtxbasevector, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dgemv(mtx_notrans, 2.0, &A, &x, 3.0, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxbasevector, y.type);
            const struct mtxbasevector * y_ = &y.storage.base;
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
            err = mtxvector_init_complex_single(&y, mtxbasevector, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dgemv(mtx_conjtrans, 2.0, &A, &x, 3.0, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxbasevector, y.type);
            const struct mtxbasevector * y_ = &y.storage.base;
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
            err = mtxvector_init_complex_single(&y, mtxbasevector, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_cgemv(mtx_notrans, calpha, &A, &x, cbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxbasevector, y.type);
            const struct mtxbasevector * y_ = &y.storage.base;
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
            err = mtxvector_init_complex_single(&y, mtxbasevector, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_cgemv(mtx_conjtrans, calpha, &A, &x, cbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxbasevector, y.type);
            const struct mtxbasevector * y_ = &y.storage.base;
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
            err = mtxvector_init_complex_single(&y, mtxbasevector, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_zgemv(mtx_notrans, zalpha, &A, &x, zbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxbasevector, y.type);
            const struct mtxbasevector * y_ = &y.storage.base;
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
            err = mtxvector_init_complex_single(&y, mtxbasevector, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_zgemv(mtx_conjtrans, zalpha, &A, &x, zbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxbasevector, y.type);
            const struct mtxbasevector * y_ = &y.storage.base;
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
            &A, mtxblasdense, mtx_unsymmetric, num_rows, num_columns, num_nonzeros, Arowidx, Acolidx, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_init_complex_double(&x, mtxbasevector, num_columns, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        {
            err = mtxvector_init_complex_double(&y, mtxbasevector, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemv(mtx_notrans, 2.0f, &A, &x, 3.0f, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxbasevector, y.type);
            const struct mtxbasevector * y_ = &y.storage.base;
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
            err = mtxvector_init_complex_double(&y, mtxbasevector, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemv(mtx_trans, 2.0f, &A, &x, 3.0f, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxbasevector, y.type);
            const struct mtxbasevector * y_ = &y.storage.base;
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
            err = mtxvector_init_complex_double(&y, mtxbasevector, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemv(mtx_conjtrans, 2.0f, &A, &x, 3.0f, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxbasevector, y.type);
            const struct mtxbasevector * y_ = &y.storage.base;
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
            err = mtxvector_init_complex_double(&y, mtxbasevector, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dgemv(mtx_notrans, 2.0, &A, &x, 3.0, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxbasevector, y.type);
            const struct mtxbasevector * y_ = &y.storage.base;
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
            err = mtxvector_init_complex_double(&y, mtxbasevector, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dgemv(mtx_trans, 2.0, &A, &x, 3.0, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxbasevector, y.type);
            const struct mtxbasevector * y_ = &y.storage.base;
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
            err = mtxvector_init_complex_double(&y, mtxbasevector, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dgemv(mtx_conjtrans, 2.0, &A, &x, 3.0, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxbasevector, y.type);
            const struct mtxbasevector * y_ = &y.storage.base;
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
            err = mtxvector_init_complex_double(&y, mtxbasevector, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_cgemv(mtx_notrans, calpha, &A, &x, cbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxbasevector, y.type);
            const struct mtxbasevector * y_ = &y.storage.base;
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
            err = mtxvector_init_complex_double(&y, mtxbasevector, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_cgemv(mtx_trans, calpha, &A, &x, cbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxbasevector, y.type);
            const struct mtxbasevector * y_ = &y.storage.base;
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
            err = mtxvector_init_complex_double(&y, mtxbasevector, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_cgemv(mtx_conjtrans, calpha, &A, &x, cbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxbasevector, y.type);
            const struct mtxbasevector * y_ = &y.storage.base;
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
            err = mtxvector_init_complex_double(&y, mtxbasevector, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_zgemv(mtx_notrans, zalpha, &A, &x, zbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxbasevector, y.type);
            const struct mtxbasevector * y_ = &y.storage.base;
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
            err = mtxvector_init_complex_double(&y, mtxbasevector, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_zgemv(mtx_trans, zalpha, &A, &x, zbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxbasevector, y.type);
            const struct mtxbasevector * y_ = &y.storage.base;
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
            err = mtxvector_init_complex_double(&y, mtxbasevector, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_zgemv(mtx_conjtrans, zalpha, &A, &x, zbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxbasevector, y.type);
            const struct mtxbasevector * y_ = &y.storage.base;
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
            &A, mtxblasdense, mtx_hermitian, num_rows, num_columns, num_nonzeros, Arowidx, Acolidx, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_init_complex_double(&x, mtxbasevector, num_columns, xdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        {
            err = mtxvector_init_complex_double(&y, mtxbasevector, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemv(mtx_notrans, 2.0f, &A, &x, 3.0f, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxbasevector, y.type);
            const struct mtxbasevector * y_ = &y.storage.base;
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
            err = mtxvector_init_complex_double(&y, mtxbasevector, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemv(mtx_conjtrans, 2.0f, &A, &x, 3.0f, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxbasevector, y.type);
            const struct mtxbasevector * y_ = &y.storage.base;
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
            err = mtxvector_init_complex_double(&y, mtxbasevector, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dgemv(mtx_notrans, 2.0, &A, &x, 3.0, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxbasevector, y.type);
            const struct mtxbasevector * y_ = &y.storage.base;
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
            err = mtxvector_init_complex_double(&y, mtxbasevector, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dgemv(mtx_conjtrans, 2.0, &A, &x, 3.0, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxbasevector, y.type);
            const struct mtxbasevector * y_ = &y.storage.base;
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
            err = mtxvector_init_complex_double(&y, mtxbasevector, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_cgemv(mtx_notrans, calpha, &A, &x, cbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxbasevector, y.type);
            const struct mtxbasevector * y_ = &y.storage.base;
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
            err = mtxvector_init_complex_double(&y, mtxbasevector, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_cgemv(mtx_conjtrans, calpha, &A, &x, cbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxbasevector, y.type);
            const struct mtxbasevector * y_ = &y.storage.base;
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
            err = mtxvector_init_complex_double(&y, mtxbasevector, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_zgemv(mtx_notrans, zalpha, &A, &x, zbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxbasevector, y.type);
            const struct mtxbasevector * y_ = &y.storage.base;
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
            err = mtxvector_init_complex_double(&y, mtxbasevector, num_rows, ydata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_zgemv(mtx_conjtrans, zalpha, &A, &x, zbeta, &y, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxbasevector, y.type);
            const struct mtxbasevector * y_ = &y.storage.base;
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
    return TEST_SUCCESS;
}

/**
 * ‘test_mtxblasdense_gemm()’ tests computing matrix-matrix products
 * for dense matrices.
 */
int test_mtxblasdense_gemm(void)
{
    int err;

    /*
     * a) For unsymmetric real or integer matrices, calculate
     *
     *   ⎡ 1 2 0⎤  ⎡ 3 2 2⎤     ⎡ 1 2 0⎤  ⎡ 14  8  4⎤  ⎡ 3 6 0⎤  ⎡ 17 14  4⎤
     * 2*⎢ 4 5 6⎥ *⎢ 2 1 0⎥ + 3*⎢ 0 1 2⎥ =⎢ 56 26 16⎥ +⎢ 0 3 6⎥ =⎢ 56 29 22⎥,
     *   ⎣ 7 8 9⎦  ⎣ 1 0 0⎦     ⎣ 2 0 1⎦  ⎣ 92 44 28⎦  ⎣ 6 0 3⎦  ⎣ 98 44 31⎦
     *
     * and the transposed products
     *
     *   ⎡ 1 2 0⎤  ⎡ 3 2 1⎤     ⎡ 1 2 0⎤  ⎡ 14  8  4⎤  ⎡ 3 6 0⎤  ⎡ 17 14  2⎤
     * 2*⎢ 4 5 6⎥ *⎢ 2 1 0⎥ + 3*⎢ 0 1 2⎥ =⎢ 56 26 16⎥ +⎢ 0 3 6⎥ =⎢ 68 29 14⎥,
     *   ⎣ 7 8 9⎦  ⎣ 2 0 0⎦     ⎣ 2 0 1⎦  ⎣ 92 44 28⎦  ⎣ 6 0 3⎦  ⎣116 44 17⎦
     *
     *   ⎡ 1 4 7⎤  ⎡ 3 2 2⎤     ⎡ 1 2 0⎤  ⎡ 36 12  4⎤  ⎡ 3 6 0⎤  ⎡ 39 18  4⎤
     * 2*⎢ 2 5 8⎥ *⎢ 2 1 0⎥ + 3*⎢ 0 1 2⎥ =⎢ 48 18  8⎥ +⎢ 0 3 6⎥ =⎢ 48 21 14⎥,
     *   ⎣ 0 6 9⎦  ⎣ 1 0 0⎦     ⎣ 2 0 1⎦  ⎣ 42 12  0⎦  ⎣ 6 0 3⎦  ⎣ 48 12  3⎦
     *
     *   ⎡ 1 4 7⎤  ⎡ 3 2 1⎤     ⎡ 1 2 0⎤  ⎡ 36 12  4⎤  ⎡ 3 6 0⎤  ⎡ 53 18  2⎤
     * 2*⎢ 2 5 8⎥ *⎢ 2 1 0⎥ + 3*⎢ 0 1 2⎥ =⎢ 48 18  8⎥ +⎢ 0 3 6⎥ =⎢ 64 21 10⎥,
     *   ⎣ 0 6 9⎦  ⎣ 2 0 0⎦     ⎣ 2 0 1⎦  ⎣ 42 12  0⎦  ⎣ 6 0 3⎦  ⎣ 66 12  3⎦
     */

    /* Real, single precision, unsymmetric matrices */
    {
        struct mtxmatrix A;
        struct mtxmatrix B;
        struct mtxmatrix C;
        int M = 3, N = 3, K = 3;
        int Annz = 8;
        int Arowidx[] = {0, 0, 1, 1, 1, 2, 2, 2};
        int Acolidx[] = {0, 1, 0, 1, 2, 0, 1, 2};
        float Adata[] = {1.0f, 2.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
        int Bnnz = 6;
        int Browidx[] = {0, 0, 0, 1, 1, 2};
        int Bcolidx[] = {0, 1, 2, 0, 1, 0};
        float Bdata[] = {3.0f, 2.0f, 2.0f, 2.0f, 1.0f, 1.0f};
        int Cnnz = 6;
        int Crowidx[] = {0, 0, 1, 1, 2, 2};
        int Ccolidx[] = {0, 1, 1, 2, 0, 2};
        float Cdata[] = {1.0f, 2.0f, 1.0f, 2.0f, 2.0f, 1.0f};
        err = mtxmatrix_init_entries_real_single(
            &A, mtxblasdense, mtx_unsymmetric, M, N, Annz, Arowidx, Acolidx, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxmatrix_init_entries_real_single(
            &B, mtxblasdense, mtx_unsymmetric, N, K, Bnnz, Browidx, Bcolidx, Bdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        {
            err = mtxmatrix_init_entries_real_single(
                &C, mtxblasdense, mtx_unsymmetric, M, K, Cnnz, Crowidx, Ccolidx, Cdata);
            err = mtxmatrix_sgemm(
                mtx_notrans, mtx_notrans, 2.0f, &A, &B, 3.0f, &C, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxblasdense, C.type);
            const struct mtxbasevector * Ca = &C.storage.dense.a;
            TEST_ASSERT_EQ(mtx_field_real, Ca->field);
            TEST_ASSERT_EQ(mtx_single, Ca->precision);
            TEST_ASSERT_EQ(9, Ca->size);
            TEST_ASSERT_EQ(Ca->data.real_single[0], 17.0f);
            TEST_ASSERT_EQ(Ca->data.real_single[1], 14.0f);
            TEST_ASSERT_EQ(Ca->data.real_single[2],  4.0f);
            TEST_ASSERT_EQ(Ca->data.real_single[3], 56.0f);
            TEST_ASSERT_EQ(Ca->data.real_single[4], 29.0f);
            TEST_ASSERT_EQ(Ca->data.real_single[5], 22.0f);
            TEST_ASSERT_EQ(Ca->data.real_single[6], 98.0f);
            TEST_ASSERT_EQ(Ca->data.real_single[7], 44.0f);
            TEST_ASSERT_EQ(Ca->data.real_single[8], 31.0f);
            mtxmatrix_free(&C);
        }
        {
            err = mtxmatrix_init_entries_real_single(
                &C, mtxblasdense, mtx_unsymmetric, M, K, Cnnz, Crowidx, Ccolidx, Cdata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemm(mtx_notrans, mtx_trans, 2.0f, &A, &B, 3.0f, &C, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxblasdense, C.type);
            const struct mtxbasevector * Ca = &C.storage.dense.a;
            TEST_ASSERT_EQ(mtx_field_real, Ca->field);
            TEST_ASSERT_EQ(mtx_single, Ca->precision);
            TEST_ASSERT_EQ(9, Ca->size);
            TEST_ASSERT_EQ(Ca->data.real_single[0],  17.0f);
            TEST_ASSERT_EQ(Ca->data.real_single[1],  14.0f);
            TEST_ASSERT_EQ(Ca->data.real_single[2],   2.0f);
            TEST_ASSERT_EQ(Ca->data.real_single[3],  68.0f);
            TEST_ASSERT_EQ(Ca->data.real_single[4],  29.0f);
            TEST_ASSERT_EQ(Ca->data.real_single[5],  14.0f);
            TEST_ASSERT_EQ(Ca->data.real_single[6], 116.0f);
            TEST_ASSERT_EQ(Ca->data.real_single[7],  44.0f);
            TEST_ASSERT_EQ(Ca->data.real_single[8],  17.0f);
            mtxmatrix_free(&C);
        }
        {
            err = mtxmatrix_init_entries_real_single(
                &C, mtxblasdense, mtx_unsymmetric, M, K, Cnnz, Crowidx, Ccolidx, Cdata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemm(mtx_trans, mtx_notrans, 2.0f, &A, &B, 3.0f, &C, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxblasdense, C.type);
            const struct mtxbasevector * Ca = &C.storage.dense.a;
            TEST_ASSERT_EQ(mtx_field_real, Ca->field);
            TEST_ASSERT_EQ(mtx_single, Ca->precision);
            TEST_ASSERT_EQ(9, Ca->size);
            TEST_ASSERT_EQ(Ca->data.real_single[0], 39.0f);
            TEST_ASSERT_EQ(Ca->data.real_single[1], 18.0f);
            TEST_ASSERT_EQ(Ca->data.real_single[2],  4.0f);
            TEST_ASSERT_EQ(Ca->data.real_single[3], 48.0f);
            TEST_ASSERT_EQ(Ca->data.real_single[4], 21.0f);
            TEST_ASSERT_EQ(Ca->data.real_single[5], 14.0f);
            TEST_ASSERT_EQ(Ca->data.real_single[6], 48.0f);
            TEST_ASSERT_EQ(Ca->data.real_single[7], 12.0f);
            TEST_ASSERT_EQ(Ca->data.real_single[8],  3.0f);
            mtxmatrix_free(&C);
        }
        {
            err = mtxmatrix_init_entries_real_single(
                &C, mtxblasdense, mtx_unsymmetric, M, K, Cnnz, Crowidx, Ccolidx, Cdata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemm(mtx_trans, mtx_trans, 2.0f, &A, &B, 3.0f, &C, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxblasdense, C.type);
            const struct mtxbasevector * Ca = &C.storage.dense.a;
            TEST_ASSERT_EQ(mtx_field_real, Ca->field);
            TEST_ASSERT_EQ(mtx_single, Ca->precision);
            TEST_ASSERT_EQ(9, Ca->size);
            TEST_ASSERT_EQ(Ca->data.real_single[0], 53.0f);
            TEST_ASSERT_EQ(Ca->data.real_single[1], 18.0f);
            TEST_ASSERT_EQ(Ca->data.real_single[2],  2.0f);
            TEST_ASSERT_EQ(Ca->data.real_single[3], 64.0f);
            TEST_ASSERT_EQ(Ca->data.real_single[4], 21.0f);
            TEST_ASSERT_EQ(Ca->data.real_single[5], 10.0f);
            TEST_ASSERT_EQ(Ca->data.real_single[6], 66.0f);
            TEST_ASSERT_EQ(Ca->data.real_single[7], 12.0f);
            TEST_ASSERT_EQ(Ca->data.real_single[8],  3.0f);
            mtxmatrix_free(&C);
        }
        mtxmatrix_free(&B);
        mtxmatrix_free(&A);
    }

    /* Real, double precision, unsymmetric matrices */
    {
        struct mtxmatrix A;
        struct mtxmatrix B;
        struct mtxmatrix C;
        int M = 3, N = 3, K = 3;
        int Annz = 8;
        int Arowidx[] = {0, 0, 1, 1, 1, 2, 2, 2};
        int Acolidx[] = {0, 1, 0, 1, 2, 0, 1, 2};
        double Adata[] = {1.0, 2.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
        int Bnnz = 6;
        int Browidx[] = {0, 0, 0, 1, 1, 2};
        int Bcolidx[] = {0, 1, 2, 0, 1, 0};
        double Bdata[] = {3.0, 2.0, 2.0, 2.0, 1.0, 1.0};
        int Cnnz = 6;
        int Crowidx[] = {0, 0, 1, 1, 2, 2};
        int Ccolidx[] = {0, 1, 1, 2, 0, 2};
        double Cdata[] = {1.0, 2.0, 1.0, 2.0, 2.0, 1.0};
        err = mtxmatrix_init_entries_real_double(
            &A, mtxblasdense, mtx_unsymmetric, M, N, Annz, Arowidx, Acolidx, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxmatrix_init_entries_real_double(
            &B, mtxblasdense, mtx_unsymmetric, N, K, Bnnz, Browidx, Bcolidx, Bdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        {
            err = mtxmatrix_init_entries_real_double(
                &C, mtxblasdense, mtx_unsymmetric, M, K, Cnnz, Crowidx, Ccolidx, Cdata);
            err = mtxmatrix_sgemm(
                mtx_notrans, mtx_notrans, 2.0f, &A, &B, 3.0f, &C, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxblasdense, C.type);
            const struct mtxbasevector * Ca = &C.storage.dense.a;
            TEST_ASSERT_EQ(mtx_field_real, Ca->field);
            TEST_ASSERT_EQ(mtx_double, Ca->precision);
            TEST_ASSERT_EQ(9, Ca->size);
            TEST_ASSERT_EQ(Ca->data.real_double[0], 17.0);
            TEST_ASSERT_EQ(Ca->data.real_double[1], 14.0);
            TEST_ASSERT_EQ(Ca->data.real_double[2],  4.0);
            TEST_ASSERT_EQ(Ca->data.real_double[3], 56.0);
            TEST_ASSERT_EQ(Ca->data.real_double[4], 29.0);
            TEST_ASSERT_EQ(Ca->data.real_double[5], 22.0);
            TEST_ASSERT_EQ(Ca->data.real_double[6], 98.0);
            TEST_ASSERT_EQ(Ca->data.real_double[7], 44.0);
            TEST_ASSERT_EQ(Ca->data.real_double[8], 31.0);
            mtxmatrix_free(&C);
        }
        {
            err = mtxmatrix_init_entries_real_double(
                &C, mtxblasdense, mtx_unsymmetric, M, K, Cnnz, Crowidx, Ccolidx, Cdata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemm(mtx_notrans, mtx_trans, 2.0f, &A, &B, 3.0f, &C, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxblasdense, C.type);
            const struct mtxbasevector * Ca = &C.storage.dense.a;
            TEST_ASSERT_EQ(mtx_field_real, Ca->field);
            TEST_ASSERT_EQ(mtx_double, Ca->precision);
            TEST_ASSERT_EQ(9, Ca->size);
            TEST_ASSERT_EQ(Ca->data.real_double[0],  17.0);
            TEST_ASSERT_EQ(Ca->data.real_double[1],  14.0);
            TEST_ASSERT_EQ(Ca->data.real_double[2],   2.0);
            TEST_ASSERT_EQ(Ca->data.real_double[3],  68.0);
            TEST_ASSERT_EQ(Ca->data.real_double[4],  29.0);
            TEST_ASSERT_EQ(Ca->data.real_double[5],  14.0);
            TEST_ASSERT_EQ(Ca->data.real_double[6], 116.0);
            TEST_ASSERT_EQ(Ca->data.real_double[7],  44.0);
            TEST_ASSERT_EQ(Ca->data.real_double[8],  17.0);
            mtxmatrix_free(&C);
        }
        {
            err = mtxmatrix_init_entries_real_double(
                &C, mtxblasdense, mtx_unsymmetric, M, K, Cnnz, Crowidx, Ccolidx, Cdata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemm(mtx_trans, mtx_notrans, 2.0f, &A, &B, 3.0f, &C, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxblasdense, C.type);
            const struct mtxbasevector * Ca = &C.storage.dense.a;
            TEST_ASSERT_EQ(mtx_field_real, Ca->field);
            TEST_ASSERT_EQ(mtx_double, Ca->precision);
            TEST_ASSERT_EQ(9, Ca->size);
            TEST_ASSERT_EQ(Ca->data.real_double[0], 39.0);
            TEST_ASSERT_EQ(Ca->data.real_double[1], 18.0);
            TEST_ASSERT_EQ(Ca->data.real_double[2],  4.0);
            TEST_ASSERT_EQ(Ca->data.real_double[3], 48.0);
            TEST_ASSERT_EQ(Ca->data.real_double[4], 21.0);
            TEST_ASSERT_EQ(Ca->data.real_double[5], 14.0);
            TEST_ASSERT_EQ(Ca->data.real_double[6], 48.0);
            TEST_ASSERT_EQ(Ca->data.real_double[7], 12.0);
            TEST_ASSERT_EQ(Ca->data.real_double[8],  3.0);
            mtxmatrix_free(&C);
        }
        {
            err = mtxmatrix_init_entries_real_double(
                &C, mtxblasdense, mtx_unsymmetric, M, K, Cnnz, Crowidx, Ccolidx, Cdata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemm(mtx_trans, mtx_trans, 2.0f, &A, &B, 3.0f, &C, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxblasdense, C.type);
            const struct mtxbasevector * Ca = &C.storage.dense.a;
            TEST_ASSERT_EQ(mtx_field_real, Ca->field);
            TEST_ASSERT_EQ(mtx_double, Ca->precision);
            TEST_ASSERT_EQ(9, Ca->size);
            TEST_ASSERT_EQ(Ca->data.real_double[0], 53.0);
            TEST_ASSERT_EQ(Ca->data.real_double[1], 18.0);
            TEST_ASSERT_EQ(Ca->data.real_double[2],  2.0);
            TEST_ASSERT_EQ(Ca->data.real_double[3], 64.0);
            TEST_ASSERT_EQ(Ca->data.real_double[4], 21.0);
            TEST_ASSERT_EQ(Ca->data.real_double[5], 10.0);
            TEST_ASSERT_EQ(Ca->data.real_double[6], 66.0);
            TEST_ASSERT_EQ(Ca->data.real_double[7], 12.0);
            TEST_ASSERT_EQ(Ca->data.real_double[8],  3.0);
            mtxmatrix_free(&C);
        }
        mtxmatrix_free(&B);
        mtxmatrix_free(&A);
    }

    /* Integer, single precision, unsymmetric matrices */
    {
        struct mtxmatrix A;
        struct mtxmatrix B;
        struct mtxmatrix C;
        int M = 3, N = 3, K = 3;
        int Annz = 8;
        int Arowidx[] = {0, 0, 1, 1, 1, 2, 2, 2};
        int Acolidx[] = {0, 1, 0, 1, 2, 0, 1, 2};
        int32_t Adata[] = {1, 2, 4, 5, 6, 7, 8, 9};
        int Bnnz = 6;
        int Browidx[] = {0, 0, 0, 1, 1, 2};
        int Bcolidx[] = {0, 1, 2, 0, 1, 0};
        int32_t Bdata[] = {3, 2, 2, 2, 1, 1};
        int Cnnz = 6;
        int Crowidx[] = {0, 0, 1, 1, 2, 2};
        int Ccolidx[] = {0, 1, 1, 2, 0, 2};
        int32_t Cdata[] = {1, 2, 1, 2, 2, 1};
        err = mtxmatrix_init_entries_integer_single(
            &A, mtxblasdense, mtx_unsymmetric, M, N, Annz, Arowidx, Acolidx, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxmatrix_init_entries_integer_single(
            &B, mtxblasdense, mtx_unsymmetric, N, K, Bnnz, Browidx, Bcolidx, Bdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        {
            err = mtxmatrix_init_entries_integer_single(
                &C, mtxblasdense, mtx_unsymmetric, M, K, Cnnz, Crowidx, Ccolidx, Cdata);
            err = mtxmatrix_sgemm(
                mtx_notrans, mtx_notrans, 2.0f, &A, &B, 3.0f, &C, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxblasdense, C.type);
            const struct mtxbasevector * Ca = &C.storage.dense.a;
            TEST_ASSERT_EQ(mtx_field_integer, Ca->field);
            TEST_ASSERT_EQ(mtx_single, Ca->precision);
            TEST_ASSERT_EQ(9, Ca->size);
            TEST_ASSERT_EQ(Ca->data.integer_single[0], 17);
            TEST_ASSERT_EQ(Ca->data.integer_single[1], 14);
            TEST_ASSERT_EQ(Ca->data.integer_single[2],  4);
            TEST_ASSERT_EQ(Ca->data.integer_single[3], 56);
            TEST_ASSERT_EQ(Ca->data.integer_single[4], 29);
            TEST_ASSERT_EQ(Ca->data.integer_single[5], 22);
            TEST_ASSERT_EQ(Ca->data.integer_single[6], 98);
            TEST_ASSERT_EQ(Ca->data.integer_single[7], 44);
            TEST_ASSERT_EQ(Ca->data.integer_single[8], 31);
            mtxmatrix_free(&C);
        }
        {
            err = mtxmatrix_init_entries_integer_single(
                &C, mtxblasdense, mtx_unsymmetric, M, K, Cnnz, Crowidx, Ccolidx, Cdata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemm(mtx_notrans, mtx_trans, 2.0f, &A, &B, 3.0f, &C, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxblasdense, C.type);
            const struct mtxbasevector * Ca = &C.storage.dense.a;
            TEST_ASSERT_EQ(mtx_field_integer, Ca->field);
            TEST_ASSERT_EQ(mtx_single, Ca->precision);
            TEST_ASSERT_EQ(9, Ca->size);
            TEST_ASSERT_EQ(Ca->data.integer_single[0],  17);
            TEST_ASSERT_EQ(Ca->data.integer_single[1],  14);
            TEST_ASSERT_EQ(Ca->data.integer_single[2],   2);
            TEST_ASSERT_EQ(Ca->data.integer_single[3],  68);
            TEST_ASSERT_EQ(Ca->data.integer_single[4],  29);
            TEST_ASSERT_EQ(Ca->data.integer_single[5],  14);
            TEST_ASSERT_EQ(Ca->data.integer_single[6], 116);
            TEST_ASSERT_EQ(Ca->data.integer_single[7],  44);
            TEST_ASSERT_EQ(Ca->data.integer_single[8],  17);
            mtxmatrix_free(&C);
        }
        {
            err = mtxmatrix_init_entries_integer_single(
                &C, mtxblasdense, mtx_unsymmetric, M, K, Cnnz, Crowidx, Ccolidx, Cdata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemm(mtx_trans, mtx_notrans, 2.0f, &A, &B, 3.0f, &C, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxblasdense, C.type);
            const struct mtxbasevector * Ca = &C.storage.dense.a;
            TEST_ASSERT_EQ(mtx_field_integer, Ca->field);
            TEST_ASSERT_EQ(mtx_single, Ca->precision);
            TEST_ASSERT_EQ(9, Ca->size);
            TEST_ASSERT_EQ(Ca->data.integer_single[0], 39);
            TEST_ASSERT_EQ(Ca->data.integer_single[1], 18);
            TEST_ASSERT_EQ(Ca->data.integer_single[2],  4);
            TEST_ASSERT_EQ(Ca->data.integer_single[3], 48);
            TEST_ASSERT_EQ(Ca->data.integer_single[4], 21);
            TEST_ASSERT_EQ(Ca->data.integer_single[5], 14);
            TEST_ASSERT_EQ(Ca->data.integer_single[6], 48);
            TEST_ASSERT_EQ(Ca->data.integer_single[7], 12);
            TEST_ASSERT_EQ(Ca->data.integer_single[8],  3);
            mtxmatrix_free(&C);
        }
        {
            err = mtxmatrix_init_entries_integer_single(
                &C, mtxblasdense, mtx_unsymmetric, M, K, Cnnz, Crowidx, Ccolidx, Cdata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemm(mtx_trans, mtx_trans, 2.0f, &A, &B, 3.0f, &C, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxblasdense, C.type);
            const struct mtxbasevector * Ca = &C.storage.dense.a;
            TEST_ASSERT_EQ(mtx_field_integer, Ca->field);
            TEST_ASSERT_EQ(mtx_single, Ca->precision);
            TEST_ASSERT_EQ(9, Ca->size);
            TEST_ASSERT_EQ(Ca->data.integer_single[0], 53);
            TEST_ASSERT_EQ(Ca->data.integer_single[1], 18);
            TEST_ASSERT_EQ(Ca->data.integer_single[2],  2);
            TEST_ASSERT_EQ(Ca->data.integer_single[3], 64);
            TEST_ASSERT_EQ(Ca->data.integer_single[4], 21);
            TEST_ASSERT_EQ(Ca->data.integer_single[5], 10);
            TEST_ASSERT_EQ(Ca->data.integer_single[6], 66);
            TEST_ASSERT_EQ(Ca->data.integer_single[7], 12);
            TEST_ASSERT_EQ(Ca->data.integer_single[8],  3);
            mtxmatrix_free(&C);
        }
        mtxmatrix_free(&B);
        mtxmatrix_free(&A);
    }

    /* Integer, double precision, unsymmetric matrices */
    {
        struct mtxmatrix A;
        struct mtxmatrix B;
        struct mtxmatrix C;
        int M = 3, N = 3, K = 3;
        int Annz = 8;
        int Arowidx[] = {0, 0, 1, 1, 1, 2, 2, 2};
        int Acolidx[] = {0, 1, 0, 1, 2, 0, 1, 2};
        int64_t Adata[] = {1, 2, 4, 5, 6, 7, 8, 9};
        int Bnnz = 6;
        int Browidx[] = {0, 0, 0, 1, 1, 2};
        int Bcolidx[] = {0, 1, 2, 0, 1, 0};
        int64_t Bdata[] = {3, 2, 2, 2, 1, 1};
        int Cnnz = 6;
        int Crowidx[] = {0, 0, 1, 1, 2, 2};
        int Ccolidx[] = {0, 1, 1, 2, 0, 2};
        int64_t Cdata[] = {1, 2, 1, 2, 2, 1};
        err = mtxmatrix_init_entries_integer_double(
            &A, mtxblasdense, mtx_unsymmetric, M, N, Annz, Arowidx, Acolidx, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxmatrix_init_entries_integer_double(
            &B, mtxblasdense, mtx_unsymmetric, N, K, Bnnz, Browidx, Bcolidx, Bdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        {
            err = mtxmatrix_init_entries_integer_double(
                &C, mtxblasdense, mtx_unsymmetric, M, K, Cnnz, Crowidx, Ccolidx, Cdata);
            err = mtxmatrix_sgemm(
                mtx_notrans, mtx_notrans, 2.0f, &A, &B, 3.0f, &C, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxblasdense, C.type);
            const struct mtxbasevector * Ca = &C.storage.dense.a;
            TEST_ASSERT_EQ(mtx_field_integer, Ca->field);
            TEST_ASSERT_EQ(mtx_double, Ca->precision);
            TEST_ASSERT_EQ(9, Ca->size);
            TEST_ASSERT_EQ(Ca->data.integer_double[0], 17);
            TEST_ASSERT_EQ(Ca->data.integer_double[1], 14);
            TEST_ASSERT_EQ(Ca->data.integer_double[2],  4);
            TEST_ASSERT_EQ(Ca->data.integer_double[3], 56);
            TEST_ASSERT_EQ(Ca->data.integer_double[4], 29);
            TEST_ASSERT_EQ(Ca->data.integer_double[5], 22);
            TEST_ASSERT_EQ(Ca->data.integer_double[6], 98);
            TEST_ASSERT_EQ(Ca->data.integer_double[7], 44);
            TEST_ASSERT_EQ(Ca->data.integer_double[8], 31);
            mtxmatrix_free(&C);
        }
        {
            err = mtxmatrix_init_entries_integer_double(
                &C, mtxblasdense, mtx_unsymmetric, M, K, Cnnz, Crowidx, Ccolidx, Cdata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemm(mtx_notrans, mtx_trans, 2.0f, &A, &B, 3.0f, &C, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxblasdense, C.type);
            const struct mtxbasevector * Ca = &C.storage.dense.a;
            TEST_ASSERT_EQ(mtx_field_integer, Ca->field);
            TEST_ASSERT_EQ(mtx_double, Ca->precision);
            TEST_ASSERT_EQ(9, Ca->size);
            TEST_ASSERT_EQ(Ca->data.integer_double[0],  17);
            TEST_ASSERT_EQ(Ca->data.integer_double[1],  14);
            TEST_ASSERT_EQ(Ca->data.integer_double[2],   2);
            TEST_ASSERT_EQ(Ca->data.integer_double[3],  68);
            TEST_ASSERT_EQ(Ca->data.integer_double[4],  29);
            TEST_ASSERT_EQ(Ca->data.integer_double[5],  14);
            TEST_ASSERT_EQ(Ca->data.integer_double[6], 116);
            TEST_ASSERT_EQ(Ca->data.integer_double[7],  44);
            TEST_ASSERT_EQ(Ca->data.integer_double[8],  17);
            mtxmatrix_free(&C);
        }
        {
            err = mtxmatrix_init_entries_integer_double(
                &C, mtxblasdense, mtx_unsymmetric, M, K, Cnnz, Crowidx, Ccolidx, Cdata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemm(mtx_trans, mtx_notrans, 2.0f, &A, &B, 3.0f, &C, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxblasdense, C.type);
            const struct mtxbasevector * Ca = &C.storage.dense.a;
            TEST_ASSERT_EQ(mtx_field_integer, Ca->field);
            TEST_ASSERT_EQ(mtx_double, Ca->precision);
            TEST_ASSERT_EQ(9, Ca->size);
            TEST_ASSERT_EQ(Ca->data.integer_double[0], 39);
            TEST_ASSERT_EQ(Ca->data.integer_double[1], 18);
            TEST_ASSERT_EQ(Ca->data.integer_double[2],  4);
            TEST_ASSERT_EQ(Ca->data.integer_double[3], 48);
            TEST_ASSERT_EQ(Ca->data.integer_double[4], 21);
            TEST_ASSERT_EQ(Ca->data.integer_double[5], 14);
            TEST_ASSERT_EQ(Ca->data.integer_double[6], 48);
            TEST_ASSERT_EQ(Ca->data.integer_double[7], 12);
            TEST_ASSERT_EQ(Ca->data.integer_double[8],  3);
            mtxmatrix_free(&C);
        }
        {
            err = mtxmatrix_init_entries_integer_double(
                &C, mtxblasdense, mtx_unsymmetric, M, K, Cnnz, Crowidx, Ccolidx, Cdata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemm(mtx_trans, mtx_trans, 2.0f, &A, &B, 3.0f, &C, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxblasdense, C.type);
            const struct mtxbasevector * Ca = &C.storage.dense.a;
            TEST_ASSERT_EQ(mtx_field_integer, Ca->field);
            TEST_ASSERT_EQ(mtx_double, Ca->precision);
            TEST_ASSERT_EQ(9, Ca->size);
            TEST_ASSERT_EQ(Ca->data.integer_double[0], 53);
            TEST_ASSERT_EQ(Ca->data.integer_double[1], 18);
            TEST_ASSERT_EQ(Ca->data.integer_double[2],  2);
            TEST_ASSERT_EQ(Ca->data.integer_double[3], 64);
            TEST_ASSERT_EQ(Ca->data.integer_double[4], 21);
            TEST_ASSERT_EQ(Ca->data.integer_double[5], 10);
            TEST_ASSERT_EQ(Ca->data.integer_double[6], 66);
            TEST_ASSERT_EQ(Ca->data.integer_double[7], 12);
            TEST_ASSERT_EQ(Ca->data.integer_double[8],  3);
            mtxmatrix_free(&C);
        }
        mtxmatrix_free(&B);
        mtxmatrix_free(&A);
    }

    /*
     * b) For unsymmetric complex matrices, calculate
     *
     *   ⎡ 1+2i 3+4i⎤  ⎡ 3+1i 0+1i⎤     ⎡ 1+0i 2+2i⎤  ⎡-5+34i  8+16i⎤
     * 2*⎢          ⎥ *⎢          ⎥ + 3*⎢          ⎥ =⎢             ⎥,
     *   ⎣ 5+6i 7+8i⎦  ⎣ 1+2i 1+0i⎦     ⎣ 2+2i 0+1i⎦  ⎣ 6+96i  2+29i⎦
     *
     * and the transposed products
     *
     *   ⎡ 1+2i 3+4i⎤  ⎡ 3+1i 1+2i⎤     ⎡ 1+0i 2+2i⎤  ⎡-3+20i  6+22i⎤
     * 2*⎢          ⎥ *⎢          ⎥ + 3*⎢          ⎥ =⎢             ⎥,
     *   ⎣ 5+6i 7+8i⎦  ⎣ 0+1i 1+0i⎦     ⎣ 2+2i 0+1i⎦  ⎣ 8+66i    51i⎦
     *
     *   ⎡ 1+2i 5+6i⎤  ⎡ 3+1i 0+1i⎤     ⎡ 1+0i 2+2i⎤  ⎡-9+46i 12+20i⎤
     * 2*⎢          ⎥ *⎢          ⎥ + 3*⎢          ⎥ =⎢             ⎥,
     *   ⎣ 3+4i 7+8i⎦  ⎣ 1+2i 1+0i⎦     ⎣ 2+2i 0+1i⎦  ⎣-2+80i  6+25i⎦
     *
     *   ⎡ 1+2i 5+6i⎤  ⎡ 3+1i 1+2i⎤     ⎡ 1+0i 2+2i⎤  ⎡-7+24i 10+26i⎤
     * 2*⎢          ⎥ *⎢          ⎥ + 3*⎢          ⎥ =⎢             ⎥,
     *   ⎣ 3+4i 7+8i⎦  ⎣ 0+1i 1+0i⎦     ⎣ 2+2i 0+1i⎦  ⎣   50i  4+39i⎦
     *
     * and the conjugate transposed product
     *
     *   ⎡ 1-2i 5-6i⎤  ⎡ 3+1i 0+1i⎤     ⎡ 1+0i 2+2i⎤  ⎡ 47-2i 20-4i⎤
     * 2*⎢          ⎥ *⎢          ⎥ + 3*⎢          ⎥ =⎢            ⎥,
     *   ⎣ 3-4i 7-8i⎦  ⎣ 1+2i 1+0i⎦     ⎣ 2+2i 0+1i⎦  ⎣    78 22-7i⎦
     */

    /* Complex, single precision, unsymmetric matrices */
    {
        struct mtxmatrix A;
        struct mtxmatrix B;
        struct mtxmatrix C;
        int M = 2, N = 2, K = 2;
        int Annz = 4;
        int Arowidx[] = {0, 0, 1, 1};
        int Acolidx[] = {0, 1, 0, 1};
        float Adata[][2] = {{1.0f,2.0f}, {3.0f,4.0f}, {5.0f,6.0f}, {7.0f,8.0f}};
        int Bnnz = 4;
        int Browidx[] = {0, 0, 1, 1};
        int Bcolidx[] = {0, 1, 0, 1};
        float Bdata[][2] = {{3.0f,1.0f}, {0.0f,1.0f}, {1.0f,2.0f}, {1.0f,0.0f}};
        int Cnnz = 4;
        int Crowidx[] = {0, 0, 1, 1};
        int Ccolidx[] = {0, 1, 0, 1};
        float Cdata[][2] = {{1.0f,0.0f}, {2.0f,2.0f}, {2.0f,2.0f}, {0.0f,1.0f}};
        err = mtxmatrix_init_entries_complex_single(
            &A, mtxblasdense, mtx_unsymmetric, M, N, Annz, Arowidx, Acolidx, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxmatrix_init_entries_complex_single(
            &B, mtxblasdense, mtx_unsymmetric, N, K, Bnnz, Browidx, Bcolidx, Bdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        {
            err = mtxmatrix_init_entries_complex_single(
                &C, mtxblasdense, mtx_unsymmetric, M, K, Cnnz, Crowidx, Ccolidx, Cdata);
            err = mtxmatrix_sgemm(
                mtx_notrans, mtx_notrans, 2.0f, &A, &B, 3.0f, &C, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxblasdense, C.type);
            const struct mtxbasevector * Ca = &C.storage.dense.a;
            TEST_ASSERT_EQ(mtx_field_complex, Ca->field);
            TEST_ASSERT_EQ(mtx_single, Ca->precision);
            TEST_ASSERT_EQ(4, Ca->size);
            TEST_ASSERT_EQ(Ca->data.complex_single[0][0], -5.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[0][1], 34.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[1][0],  8.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[1][1], 16.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[2][0],  6.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[2][1], 96.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[3][0],  2.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[3][1], 29.0f);
            mtxmatrix_free(&C);
        }
        {
            err = mtxmatrix_init_entries_complex_single(
                &C, mtxblasdense, mtx_unsymmetric, M, K, Cnnz, Crowidx, Ccolidx, Cdata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemm(mtx_notrans, mtx_trans, 2.0f, &A, &B, 3.0f, &C, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxblasdense, C.type);
            const struct mtxbasevector * Ca = &C.storage.dense.a;
            TEST_ASSERT_EQ(mtx_field_complex, Ca->field);
            TEST_ASSERT_EQ(mtx_single, Ca->precision);
            TEST_ASSERT_EQ(4, Ca->size);
            TEST_ASSERT_EQ(Ca->data.complex_single[0][0], -3.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[0][1], 20.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[1][0],  6.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[1][1], 22.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[2][0],  8.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[2][1], 66.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[3][0],  0.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[3][1], 51.0f);
            mtxmatrix_free(&C);
        }
        {
            err = mtxmatrix_init_entries_complex_single(
                &C, mtxblasdense, mtx_unsymmetric, M, K, Cnnz, Crowidx, Ccolidx, Cdata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemm(mtx_trans, mtx_notrans, 2.0f, &A, &B, 3.0f, &C, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxblasdense, C.type);
            const struct mtxbasevector * Ca = &C.storage.dense.a;
            TEST_ASSERT_EQ(mtx_field_complex, Ca->field);
            TEST_ASSERT_EQ(mtx_single, Ca->precision);
            TEST_ASSERT_EQ(4, Ca->size);
            TEST_ASSERT_EQ(Ca->data.complex_single[0][0], -9.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[0][1], 46.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[1][0], 12.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[1][1], 20.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[2][0], -2.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[2][1], 80.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[3][0],  6.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[3][1], 25.0f);
            mtxmatrix_free(&C);
        }
        {
            err = mtxmatrix_init_entries_complex_single(
                &C, mtxblasdense, mtx_unsymmetric, M, K, Cnnz, Crowidx, Ccolidx, Cdata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemm(mtx_trans, mtx_trans, 2.0f, &A, &B, 3.0f, &C, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxblasdense, C.type);
            const struct mtxbasevector * Ca = &C.storage.dense.a;
            TEST_ASSERT_EQ(mtx_field_complex, Ca->field);
            TEST_ASSERT_EQ(mtx_single, Ca->precision);
            TEST_ASSERT_EQ(4, Ca->size);
            TEST_ASSERT_EQ(Ca->data.complex_single[0][0], -7.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[0][1], 24.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[1][0], 10.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[1][1], 26.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[2][0],  0.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[2][1], 50.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[3][0],  4.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[3][1], 39.0f);
            mtxmatrix_free(&C);
        }
        {
            err = mtxmatrix_init_entries_complex_single(
                &C, mtxblasdense, mtx_unsymmetric, M, K, Cnnz, Crowidx, Ccolidx, Cdata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemm(mtx_conjtrans, mtx_notrans, 2.0f, &A, &B, 3.0f, &C, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxblasdense, C.type);
            const struct mtxbasevector * Ca = &C.storage.dense.a;
            TEST_ASSERT_EQ(mtx_field_complex, Ca->field);
            TEST_ASSERT_EQ(mtx_single, Ca->precision);
            TEST_ASSERT_EQ(4, Ca->size);
            TEST_ASSERT_EQ(Ca->data.complex_single[0][0], 47.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[0][1], -2.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[1][0], 20.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[1][1], -4.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[2][0], 78.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[2][1],  0.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[3][0], 22.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[3][1], -7.0f);
            mtxmatrix_free(&C);
        }
        mtxmatrix_free(&B);
        mtxmatrix_free(&A);
    }

    /* Complex, double precision, unsymmetric matrices */
    {
        struct mtxmatrix A;
        struct mtxmatrix B;
        struct mtxmatrix C;
        int M = 2, N = 2, K = 2;
        int Annz = 4;
        int Arowidx[] = {0, 0, 1, 1};
        int Acolidx[] = {0, 1, 0, 1};
        double Adata[][2] = {{1.0,2.0}, {3.0,4.0}, {5.0,6.0}, {7.0,8.0}};
        int Bnnz = 4;
        int Browidx[] = {0, 0, 1, 1};
        int Bcolidx[] = {0, 1, 0, 1};
        double Bdata[][2] = {{3.0,1.0}, {0.0,1.0}, {1.0,2.0}, {1.0,0.0}};
        int Cnnz = 4;
        int Crowidx[] = {0, 0, 1, 1};
        int Ccolidx[] = {0, 1, 0, 1};
        double Cdata[][2] = {{1.0,0.0}, {2.0,2.0}, {2.0,2.0}, {0.0,1.0}};
        err = mtxmatrix_init_entries_complex_double(
            &A, mtxblasdense, mtx_unsymmetric, M, N, Annz, Arowidx, Acolidx, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxmatrix_init_entries_complex_double(
            &B, mtxblasdense, mtx_unsymmetric, N, K, Bnnz, Browidx, Bcolidx, Bdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        {
            err = mtxmatrix_init_entries_complex_double(
                &C, mtxblasdense, mtx_unsymmetric, M, K, Cnnz, Crowidx, Ccolidx, Cdata);
            err = mtxmatrix_sgemm(
                mtx_notrans, mtx_notrans, 2.0f, &A, &B, 3.0f, &C, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxblasdense, C.type);
            const struct mtxbasevector * Ca = &C.storage.dense.a;
            TEST_ASSERT_EQ(mtx_field_complex, Ca->field);
            TEST_ASSERT_EQ(mtx_double, Ca->precision);
            TEST_ASSERT_EQ(4, Ca->size);
            TEST_ASSERT_EQ(Ca->data.complex_double[0][0], -5.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[0][1], 34.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[1][0],  8.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[1][1], 16.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[2][0],  6.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[2][1], 96.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[3][0],  2.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[3][1], 29.0);
            mtxmatrix_free(&C);
        }
        {
            err = mtxmatrix_init_entries_complex_double(
                &C, mtxblasdense, mtx_unsymmetric, M, K, Cnnz, Crowidx, Ccolidx, Cdata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemm(mtx_notrans, mtx_trans, 2.0f, &A, &B, 3.0f, &C, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxblasdense, C.type);
            const struct mtxbasevector * Ca = &C.storage.dense.a;
            TEST_ASSERT_EQ(mtx_field_complex, Ca->field);
            TEST_ASSERT_EQ(mtx_double, Ca->precision);
            TEST_ASSERT_EQ(4, Ca->size);
            TEST_ASSERT_EQ(Ca->data.complex_double[0][0], -3.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[0][1], 20.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[1][0],  6.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[1][1], 22.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[2][0],  8.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[2][1], 66.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[3][0],  0.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[3][1], 51.0);
            mtxmatrix_free(&C);
        }
        {
            err = mtxmatrix_init_entries_complex_double(
                &C, mtxblasdense, mtx_unsymmetric, M, K, Cnnz, Crowidx, Ccolidx, Cdata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemm(mtx_trans, mtx_notrans, 2.0f, &A, &B, 3.0f, &C, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxblasdense, C.type);
            const struct mtxbasevector * Ca = &C.storage.dense.a;
            TEST_ASSERT_EQ(mtx_field_complex, Ca->field);
            TEST_ASSERT_EQ(mtx_double, Ca->precision);
            TEST_ASSERT_EQ(4, Ca->size);
            TEST_ASSERT_EQ(Ca->data.complex_double[0][0], -9.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[0][1], 46.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[1][0], 12.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[1][1], 20.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[2][0], -2.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[2][1], 80.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[3][0],  6.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[3][1], 25.0);
            mtxmatrix_free(&C);
        }
        {
            err = mtxmatrix_init_entries_complex_double(
                &C, mtxblasdense, mtx_unsymmetric, M, K, Cnnz, Crowidx, Ccolidx, Cdata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemm(mtx_trans, mtx_trans, 2.0f, &A, &B, 3.0f, &C, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxblasdense, C.type);
            const struct mtxbasevector * Ca = &C.storage.dense.a;
            TEST_ASSERT_EQ(mtx_field_complex, Ca->field);
            TEST_ASSERT_EQ(mtx_double, Ca->precision);
            TEST_ASSERT_EQ(4, Ca->size);
            TEST_ASSERT_EQ(Ca->data.complex_double[0][0], -7.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[0][1], 24.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[1][0], 10.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[1][1], 26.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[2][0],  0.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[2][1], 50.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[3][0],  4.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[3][1], 39.0);
            mtxmatrix_free(&C);
        }
        {
            err = mtxmatrix_init_entries_complex_double(
                &C, mtxblasdense, mtx_unsymmetric, M, K, Cnnz, Crowidx, Ccolidx, Cdata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemm(mtx_conjtrans, mtx_notrans, 2.0f, &A, &B, 3.0f, &C, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxblasdense, C.type);
            const struct mtxbasevector * Ca = &C.storage.dense.a;
            TEST_ASSERT_EQ(mtx_field_complex, Ca->field);
            TEST_ASSERT_EQ(mtx_double, Ca->precision);
            TEST_ASSERT_EQ(4, Ca->size);
            TEST_ASSERT_EQ(Ca->data.complex_double[0][0], 47.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[0][1], -2.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[1][0], 20.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[1][1], -4.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[2][0], 78.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[2][1],  0.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[3][0], 22.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[3][1], -7.0);
            mtxmatrix_free(&C);
        }
        mtxmatrix_free(&B);
        mtxmatrix_free(&A);
    }

    /*
     * c) For symmetric real or integer matrices, calculate
     *
     *   ⎡ 1 2 0⎤  ⎡ 3 2 2⎤     ⎡ 1 2 0⎤  ⎡ 14  8  4⎤  ⎡ 3 6 0⎤  ⎡ 17 14  4⎤
     * 2*⎢ 2 5 6⎥ *⎢ 2 1 0⎥ + 3*⎢ 0 1 2⎥ =⎢ 56 26 16⎥ +⎢ 0 3 6⎥ =⎢ 44 21 14⎥,
     *   ⎣ 0 6 9⎦  ⎣ 1 0 0⎦     ⎣ 2 0 1⎦  ⎣ 92 44 28⎦  ⎣ 6 0 3⎦  ⎣ 48 12  3⎦
     */

    /* Real, single precision, symmetric matrices */
    {
        struct mtxmatrix A;
        struct mtxmatrix B;
        struct mtxmatrix C;
        int M = 3, N = 3, K = 3;
        int Annz = 5;
        int Arowidx[] = {0, 0, 1, 1, 2};
        int Acolidx[] = {0, 1, 1, 2, 2};
        float Adata[] = {1.0f, 2.0f, 5.0f, 6.0f, 9.0f};
        int Bnnz = 6;
        int Browidx[] = {0, 0, 0, 1, 1, 2};
        int Bcolidx[] = {0, 1, 2, 0, 1, 0};
        float Bdata[] = {3.0f, 2.0f, 2.0f, 2.0f, 1.0f, 1.0f};
        int Cnnz = 6;
        int Crowidx[] = {0, 0, 1, 1, 2, 2};
        int Ccolidx[] = {0, 1, 1, 2, 0, 2};
        float Cdata[] = {1.0f, 2.0f, 1.0f, 2.0f, 2.0f, 1.0f};
        err = mtxmatrix_init_entries_real_single(
            &A, mtxblasdense, mtx_symmetric, M, N, Annz, Arowidx, Acolidx, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxmatrix_init_entries_real_single(
            &B, mtxblasdense, mtx_unsymmetric, N, K, Bnnz, Browidx, Bcolidx, Bdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        {
            err = mtxmatrix_init_entries_real_single(
                &C, mtxblasdense, mtx_unsymmetric, M, K, Cnnz, Crowidx, Ccolidx, Cdata);
            err = mtxmatrix_sgemm(
                mtx_notrans, mtx_notrans, 2.0f, &A, &B, 3.0f, &C, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxblasdense, C.type);
            const struct mtxbasevector * Ca = &C.storage.dense.a;
            TEST_ASSERT_EQ(mtx_field_real, Ca->field);
            TEST_ASSERT_EQ(mtx_single, Ca->precision);
            TEST_ASSERT_EQ(9, Ca->size);
            TEST_ASSERT_EQ(Ca->data.real_single[0], 17.0f);
            TEST_ASSERT_EQ(Ca->data.real_single[1], 14.0f);
            TEST_ASSERT_EQ(Ca->data.real_single[2],  4.0f);
            TEST_ASSERT_EQ(Ca->data.real_single[3], 44.0f);
            TEST_ASSERT_EQ(Ca->data.real_single[4], 21.0f);
            TEST_ASSERT_EQ(Ca->data.real_single[5], 14.0f);
            TEST_ASSERT_EQ(Ca->data.real_single[6], 48.0f);
            TEST_ASSERT_EQ(Ca->data.real_single[7], 12.0f);
            TEST_ASSERT_EQ(Ca->data.real_single[8],  3.0f);
            mtxmatrix_free(&C);
        }
        {
            err = mtxmatrix_init_entries_real_single(
                &C, mtxblasdense, mtx_unsymmetric, M, K, Cnnz, Crowidx, Ccolidx, Cdata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemm(mtx_notrans, mtx_trans, 2.0f, &A, &B, 3.0f, &C, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxblasdense, C.type);
            const struct mtxbasevector * Ca = &C.storage.dense.a;
            TEST_ASSERT_EQ(mtx_field_real, Ca->field);
            TEST_ASSERT_EQ(mtx_single, Ca->precision);
            TEST_ASSERT_EQ(9, Ca->size);
            TEST_ASSERT_EQ(Ca->data.real_single[0], 17.0f);
            TEST_ASSERT_EQ(Ca->data.real_single[1], 14.0f);
            TEST_ASSERT_EQ(Ca->data.real_single[2],  2.0f);
            TEST_ASSERT_EQ(Ca->data.real_single[3], 56.0f);
            TEST_ASSERT_EQ(Ca->data.real_single[4], 21.0f);
            TEST_ASSERT_EQ(Ca->data.real_single[5], 10.0f);
            TEST_ASSERT_EQ(Ca->data.real_single[6], 66.0f);
            TEST_ASSERT_EQ(Ca->data.real_single[7], 12.0f);
            TEST_ASSERT_EQ(Ca->data.real_single[8],  3.0f);
            mtxmatrix_free(&C);
        }
        {
            err = mtxmatrix_init_entries_real_single(
                &C, mtxblasdense, mtx_unsymmetric, M, K, Cnnz, Crowidx, Ccolidx, Cdata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemm(mtx_trans, mtx_notrans, 2.0f, &A, &B, 3.0f, &C, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxblasdense, C.type);
            const struct mtxbasevector * Ca = &C.storage.dense.a;
            TEST_ASSERT_EQ(mtx_field_real, Ca->field);
            TEST_ASSERT_EQ(mtx_single, Ca->precision);
            TEST_ASSERT_EQ(9, Ca->size);
            TEST_ASSERT_EQ(Ca->data.real_single[0], 17.0f);
            TEST_ASSERT_EQ(Ca->data.real_single[1], 14.0f);
            TEST_ASSERT_EQ(Ca->data.real_single[2],  4.0f);
            TEST_ASSERT_EQ(Ca->data.real_single[3], 44.0f);
            TEST_ASSERT_EQ(Ca->data.real_single[4], 21.0f);
            TEST_ASSERT_EQ(Ca->data.real_single[5], 14.0f);
            TEST_ASSERT_EQ(Ca->data.real_single[6], 48.0f);
            TEST_ASSERT_EQ(Ca->data.real_single[7], 12.0f);
            TEST_ASSERT_EQ(Ca->data.real_single[8],  3.0f);
            mtxmatrix_free(&C);
        }
        {
            err = mtxmatrix_init_entries_real_single(
                &C, mtxblasdense, mtx_unsymmetric, M, K, Cnnz, Crowidx, Ccolidx, Cdata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemm(mtx_trans, mtx_trans, 2.0f, &A, &B, 3.0f, &C, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxblasdense, C.type);
            const struct mtxbasevector * Ca = &C.storage.dense.a;
            TEST_ASSERT_EQ(mtx_field_real, Ca->field);
            TEST_ASSERT_EQ(mtx_single, Ca->precision);
            TEST_ASSERT_EQ(9, Ca->size);
            TEST_ASSERT_EQ(Ca->data.real_single[0], 17.0f);
            TEST_ASSERT_EQ(Ca->data.real_single[1], 14.0f);
            TEST_ASSERT_EQ(Ca->data.real_single[2],  2.0f);
            TEST_ASSERT_EQ(Ca->data.real_single[3], 56.0f);
            TEST_ASSERT_EQ(Ca->data.real_single[4], 21.0f);
            TEST_ASSERT_EQ(Ca->data.real_single[5], 10.0f);
            TEST_ASSERT_EQ(Ca->data.real_single[6], 66.0f);
            TEST_ASSERT_EQ(Ca->data.real_single[7], 12.0f);
            TEST_ASSERT_EQ(Ca->data.real_single[8],  3.0f);
            mtxmatrix_free(&C);
        }
        mtxmatrix_free(&B);
        mtxmatrix_free(&A);
    }

    /* Real, double precision, symmetric matrices */
    {
        struct mtxmatrix A;
        struct mtxmatrix B;
        struct mtxmatrix C;
        int M = 3, N = 3, K = 3;
        int Annz = 5;
        int Arowidx[] = {0, 0, 1, 1, 2};
        int Acolidx[] = {0, 1, 1, 2, 2};
        double Adata[] = {1.0, 2.0, 5.0, 6.0, 9.0};
        int Bnnz = 6;
        int Browidx[] = {0, 0, 0, 1, 1, 2};
        int Bcolidx[] = {0, 1, 2, 0, 1, 0};
        double Bdata[] = {3.0, 2.0, 2.0, 2.0, 1.0, 1.0};
        int Cnnz = 6;
        int Crowidx[] = {0, 0, 1, 1, 2, 2};
        int Ccolidx[] = {0, 1, 1, 2, 0, 2};
        double Cdata[] = {1.0, 2.0, 1.0, 2.0, 2.0, 1.0};
        err = mtxmatrix_init_entries_real_double(
            &A, mtxblasdense, mtx_symmetric, M, N, Annz, Arowidx, Acolidx, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxmatrix_init_entries_real_double(
            &B, mtxblasdense, mtx_unsymmetric, N, K, Bnnz, Browidx, Bcolidx, Bdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        {
            err = mtxmatrix_init_entries_real_double(
                &C, mtxblasdense, mtx_unsymmetric, M, K, Cnnz, Crowidx, Ccolidx, Cdata);
            err = mtxmatrix_sgemm(
                mtx_notrans, mtx_notrans, 2.0f, &A, &B, 3.0f, &C, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxblasdense, C.type);
            const struct mtxbasevector * Ca = &C.storage.dense.a;
            TEST_ASSERT_EQ(mtx_field_real, Ca->field);
            TEST_ASSERT_EQ(mtx_double, Ca->precision);
            TEST_ASSERT_EQ(9, Ca->size);
            TEST_ASSERT_EQ(Ca->data.real_double[0], 17.0);
            TEST_ASSERT_EQ(Ca->data.real_double[1], 14.0);
            TEST_ASSERT_EQ(Ca->data.real_double[2],  4.0);
            TEST_ASSERT_EQ(Ca->data.real_double[3], 44.0);
            TEST_ASSERT_EQ(Ca->data.real_double[4], 21.0);
            TEST_ASSERT_EQ(Ca->data.real_double[5], 14.0);
            TEST_ASSERT_EQ(Ca->data.real_double[6], 48.0);
            TEST_ASSERT_EQ(Ca->data.real_double[7], 12.0);
            TEST_ASSERT_EQ(Ca->data.real_double[8],  3.0);
            mtxmatrix_free(&C);
        }
        {
            err = mtxmatrix_init_entries_real_double(
                &C, mtxblasdense, mtx_unsymmetric, M, K, Cnnz, Crowidx, Ccolidx, Cdata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemm(mtx_notrans, mtx_trans, 2.0f, &A, &B, 3.0f, &C, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxblasdense, C.type);
            const struct mtxbasevector * Ca = &C.storage.dense.a;
            TEST_ASSERT_EQ(mtx_field_real, Ca->field);
            TEST_ASSERT_EQ(mtx_double, Ca->precision);
            TEST_ASSERT_EQ(9, Ca->size);
            TEST_ASSERT_EQ(Ca->data.real_double[0], 17.0);
            TEST_ASSERT_EQ(Ca->data.real_double[1], 14.0);
            TEST_ASSERT_EQ(Ca->data.real_double[2],  2.0);
            TEST_ASSERT_EQ(Ca->data.real_double[3], 56.0);
            TEST_ASSERT_EQ(Ca->data.real_double[4], 21.0);
            TEST_ASSERT_EQ(Ca->data.real_double[5], 10.0);
            TEST_ASSERT_EQ(Ca->data.real_double[6], 66.0);
            TEST_ASSERT_EQ(Ca->data.real_double[7], 12.0);
            TEST_ASSERT_EQ(Ca->data.real_double[8],  3.0);
            mtxmatrix_free(&C);
        }
        {
            err = mtxmatrix_init_entries_real_double(
                &C, mtxblasdense, mtx_unsymmetric, M, K, Cnnz, Crowidx, Ccolidx, Cdata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemm(mtx_trans, mtx_notrans, 2.0f, &A, &B, 3.0f, &C, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxblasdense, C.type);
            const struct mtxbasevector * Ca = &C.storage.dense.a;
            TEST_ASSERT_EQ(mtx_field_real, Ca->field);
            TEST_ASSERT_EQ(mtx_double, Ca->precision);
            TEST_ASSERT_EQ(9, Ca->size);
            TEST_ASSERT_EQ(Ca->data.real_double[0], 17.0);
            TEST_ASSERT_EQ(Ca->data.real_double[1], 14.0);
            TEST_ASSERT_EQ(Ca->data.real_double[2],  4.0);
            TEST_ASSERT_EQ(Ca->data.real_double[3], 44.0);
            TEST_ASSERT_EQ(Ca->data.real_double[4], 21.0);
            TEST_ASSERT_EQ(Ca->data.real_double[5], 14.0);
            TEST_ASSERT_EQ(Ca->data.real_double[6], 48.0);
            TEST_ASSERT_EQ(Ca->data.real_double[7], 12.0);
            TEST_ASSERT_EQ(Ca->data.real_double[8],  3.0);
            mtxmatrix_free(&C);
        }
        {
            err = mtxmatrix_init_entries_real_double(
                &C, mtxblasdense, mtx_unsymmetric, M, K, Cnnz, Crowidx, Ccolidx, Cdata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemm(mtx_trans, mtx_trans, 2.0f, &A, &B, 3.0f, &C, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxblasdense, C.type);
            const struct mtxbasevector * Ca = &C.storage.dense.a;
            TEST_ASSERT_EQ(mtx_field_real, Ca->field);
            TEST_ASSERT_EQ(mtx_double, Ca->precision);
            TEST_ASSERT_EQ(9, Ca->size);
            TEST_ASSERT_EQ(Ca->data.real_double[0], 17.0);
            TEST_ASSERT_EQ(Ca->data.real_double[1], 14.0);
            TEST_ASSERT_EQ(Ca->data.real_double[2],  2.0);
            TEST_ASSERT_EQ(Ca->data.real_double[3], 56.0);
            TEST_ASSERT_EQ(Ca->data.real_double[4], 21.0);
            TEST_ASSERT_EQ(Ca->data.real_double[5], 10.0);
            TEST_ASSERT_EQ(Ca->data.real_double[6], 66.0);
            TEST_ASSERT_EQ(Ca->data.real_double[7], 12.0);
            TEST_ASSERT_EQ(Ca->data.real_double[8],  3.0);
            mtxmatrix_free(&C);
        }
        mtxmatrix_free(&B);
        mtxmatrix_free(&A);
    }

    /* Integer, single precision, symmetric matrices */
    {
        struct mtxmatrix A;
        struct mtxmatrix B;
        struct mtxmatrix C;
        int M = 3, N = 3, K = 3;
        int Annz = 5;
        int Arowidx[] = {0, 0, 1, 1, 2};
        int Acolidx[] = {0, 1, 1, 2, 2};
        int32_t Adata[] = {1.0f, 2.0f, 5.0f, 6.0f, 9.0f};
        int Bnnz = 6;
        int Browidx[] = {0, 0, 0, 1, 1, 2};
        int Bcolidx[] = {0, 1, 2, 0, 1, 0};
        int32_t Bdata[] = {3.0f, 2.0f, 2.0f, 2.0f, 1.0f, 1.0f};
        int Cnnz = 6;
        int Crowidx[] = {0, 0, 1, 1, 2, 2};
        int Ccolidx[] = {0, 1, 1, 2, 0, 2};
        int32_t Cdata[] = {1.0f, 2.0f, 1.0f, 2.0f, 2.0f, 1.0f};
        err = mtxmatrix_init_entries_integer_single(
            &A, mtxblasdense, mtx_symmetric, M, N, Annz, Arowidx, Acolidx, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxmatrix_init_entries_integer_single(
            &B, mtxblasdense, mtx_unsymmetric, N, K, Bnnz, Browidx, Bcolidx, Bdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        {
            err = mtxmatrix_init_entries_integer_single(
                &C, mtxblasdense, mtx_unsymmetric, M, K, Cnnz, Crowidx, Ccolidx, Cdata);
            err = mtxmatrix_sgemm(
                mtx_notrans, mtx_notrans, 2.0f, &A, &B, 3.0f, &C, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxblasdense, C.type);
            const struct mtxbasevector * Ca = &C.storage.dense.a;
            TEST_ASSERT_EQ(mtx_field_integer, Ca->field);
            TEST_ASSERT_EQ(mtx_single, Ca->precision);
            TEST_ASSERT_EQ(9, Ca->size);
            TEST_ASSERT_EQ(Ca->data.integer_single[0], 17.0f);
            TEST_ASSERT_EQ(Ca->data.integer_single[1], 14.0f);
            TEST_ASSERT_EQ(Ca->data.integer_single[2],  4.0f);
            TEST_ASSERT_EQ(Ca->data.integer_single[3], 44.0f);
            TEST_ASSERT_EQ(Ca->data.integer_single[4], 21.0f);
            TEST_ASSERT_EQ(Ca->data.integer_single[5], 14.0f);
            TEST_ASSERT_EQ(Ca->data.integer_single[6], 48.0f);
            TEST_ASSERT_EQ(Ca->data.integer_single[7], 12.0f);
            TEST_ASSERT_EQ(Ca->data.integer_single[8],  3.0f);
            mtxmatrix_free(&C);
        }
        {
            err = mtxmatrix_init_entries_integer_single(
                &C, mtxblasdense, mtx_unsymmetric, M, K, Cnnz, Crowidx, Ccolidx, Cdata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemm(mtx_notrans, mtx_trans, 2.0f, &A, &B, 3.0f, &C, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxblasdense, C.type);
            const struct mtxbasevector * Ca = &C.storage.dense.a;
            TEST_ASSERT_EQ(mtx_field_integer, Ca->field);
            TEST_ASSERT_EQ(mtx_single, Ca->precision);
            TEST_ASSERT_EQ(9, Ca->size);
            TEST_ASSERT_EQ(Ca->data.integer_single[0], 17.0f);
            TEST_ASSERT_EQ(Ca->data.integer_single[1], 14.0f);
            TEST_ASSERT_EQ(Ca->data.integer_single[2],  2.0f);
            TEST_ASSERT_EQ(Ca->data.integer_single[3], 56.0f);
            TEST_ASSERT_EQ(Ca->data.integer_single[4], 21.0f);
            TEST_ASSERT_EQ(Ca->data.integer_single[5], 10.0f);
            TEST_ASSERT_EQ(Ca->data.integer_single[6], 66.0f);
            TEST_ASSERT_EQ(Ca->data.integer_single[7], 12.0f);
            TEST_ASSERT_EQ(Ca->data.integer_single[8],  3.0f);
            mtxmatrix_free(&C);
        }
        {
            err = mtxmatrix_init_entries_integer_single(
                &C, mtxblasdense, mtx_unsymmetric, M, K, Cnnz, Crowidx, Ccolidx, Cdata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemm(mtx_trans, mtx_notrans, 2.0f, &A, &B, 3.0f, &C, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxblasdense, C.type);
            const struct mtxbasevector * Ca = &C.storage.dense.a;
            TEST_ASSERT_EQ(mtx_field_integer, Ca->field);
            TEST_ASSERT_EQ(mtx_single, Ca->precision);
            TEST_ASSERT_EQ(9, Ca->size);
            TEST_ASSERT_EQ(Ca->data.integer_single[0], 17.0f);
            TEST_ASSERT_EQ(Ca->data.integer_single[1], 14.0f);
            TEST_ASSERT_EQ(Ca->data.integer_single[2],  4.0f);
            TEST_ASSERT_EQ(Ca->data.integer_single[3], 44.0f);
            TEST_ASSERT_EQ(Ca->data.integer_single[4], 21.0f);
            TEST_ASSERT_EQ(Ca->data.integer_single[5], 14.0f);
            TEST_ASSERT_EQ(Ca->data.integer_single[6], 48.0f);
            TEST_ASSERT_EQ(Ca->data.integer_single[7], 12.0f);
            TEST_ASSERT_EQ(Ca->data.integer_single[8],  3.0f);
            mtxmatrix_free(&C);
        }
        {
            err = mtxmatrix_init_entries_integer_single(
                &C, mtxblasdense, mtx_unsymmetric, M, K, Cnnz, Crowidx, Ccolidx, Cdata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemm(mtx_trans, mtx_trans, 2.0f, &A, &B, 3.0f, &C, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxblasdense, C.type);
            const struct mtxbasevector * Ca = &C.storage.dense.a;
            TEST_ASSERT_EQ(mtx_field_integer, Ca->field);
            TEST_ASSERT_EQ(mtx_single, Ca->precision);
            TEST_ASSERT_EQ(9, Ca->size);
            TEST_ASSERT_EQ(Ca->data.integer_single[0], 17.0f);
            TEST_ASSERT_EQ(Ca->data.integer_single[1], 14.0f);
            TEST_ASSERT_EQ(Ca->data.integer_single[2],  2.0f);
            TEST_ASSERT_EQ(Ca->data.integer_single[3], 56.0f);
            TEST_ASSERT_EQ(Ca->data.integer_single[4], 21.0f);
            TEST_ASSERT_EQ(Ca->data.integer_single[5], 10.0f);
            TEST_ASSERT_EQ(Ca->data.integer_single[6], 66.0f);
            TEST_ASSERT_EQ(Ca->data.integer_single[7], 12.0f);
            TEST_ASSERT_EQ(Ca->data.integer_single[8],  3.0f);
            mtxmatrix_free(&C);
        }
        mtxmatrix_free(&B);
        mtxmatrix_free(&A);
    }

    /* Integer, double precision, symmetric matrices */
    {
        struct mtxmatrix A;
        struct mtxmatrix B;
        struct mtxmatrix C;
        int M = 3, N = 3, K = 3;
        int Annz = 5;
        int Arowidx[] = {0, 0, 1, 1, 2};
        int Acolidx[] = {0, 1, 1, 2, 2};
        int64_t Adata[] = {1.0f, 2.0f, 5.0f, 6.0f, 9.0f};
        int Bnnz = 6;
        int Browidx[] = {0, 0, 0, 1, 1, 2};
        int Bcolidx[] = {0, 1, 2, 0, 1, 0};
        int64_t Bdata[] = {3.0f, 2.0f, 2.0f, 2.0f, 1.0f, 1.0f};
        int Cnnz = 6;
        int Crowidx[] = {0, 0, 1, 1, 2, 2};
        int Ccolidx[] = {0, 1, 1, 2, 0, 2};
        int64_t Cdata[] = {1.0f, 2.0f, 1.0f, 2.0f, 2.0f, 1.0f};
        err = mtxmatrix_init_entries_integer_double(
            &A, mtxblasdense, mtx_symmetric, M, N, Annz, Arowidx, Acolidx, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxmatrix_init_entries_integer_double(
            &B, mtxblasdense, mtx_unsymmetric, N, K, Bnnz, Browidx, Bcolidx, Bdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        {
            err = mtxmatrix_init_entries_integer_double(
                &C, mtxblasdense, mtx_unsymmetric, M, K, Cnnz, Crowidx, Ccolidx, Cdata);
            err = mtxmatrix_sgemm(
                mtx_notrans, mtx_notrans, 2.0f, &A, &B, 3.0f, &C, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxblasdense, C.type);
            const struct mtxbasevector * Ca = &C.storage.dense.a;
            TEST_ASSERT_EQ(mtx_field_integer, Ca->field);
            TEST_ASSERT_EQ(mtx_double, Ca->precision);
            TEST_ASSERT_EQ(9, Ca->size);
            TEST_ASSERT_EQ(Ca->data.integer_double[0], 17.0f);
            TEST_ASSERT_EQ(Ca->data.integer_double[1], 14.0f);
            TEST_ASSERT_EQ(Ca->data.integer_double[2],  4.0f);
            TEST_ASSERT_EQ(Ca->data.integer_double[3], 44.0f);
            TEST_ASSERT_EQ(Ca->data.integer_double[4], 21.0f);
            TEST_ASSERT_EQ(Ca->data.integer_double[5], 14.0f);
            TEST_ASSERT_EQ(Ca->data.integer_double[6], 48.0f);
            TEST_ASSERT_EQ(Ca->data.integer_double[7], 12.0f);
            TEST_ASSERT_EQ(Ca->data.integer_double[8],  3.0f);
            mtxmatrix_free(&C);
        }
        {
            err = mtxmatrix_init_entries_integer_double(
                &C, mtxblasdense, mtx_unsymmetric, M, K, Cnnz, Crowidx, Ccolidx, Cdata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemm(mtx_notrans, mtx_trans, 2.0f, &A, &B, 3.0f, &C, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxblasdense, C.type);
            const struct mtxbasevector * Ca = &C.storage.dense.a;
            TEST_ASSERT_EQ(mtx_field_integer, Ca->field);
            TEST_ASSERT_EQ(mtx_double, Ca->precision);
            TEST_ASSERT_EQ(9, Ca->size);
            TEST_ASSERT_EQ(Ca->data.integer_double[0], 17.0f);
            TEST_ASSERT_EQ(Ca->data.integer_double[1], 14.0f);
            TEST_ASSERT_EQ(Ca->data.integer_double[2],  2.0f);
            TEST_ASSERT_EQ(Ca->data.integer_double[3], 56.0f);
            TEST_ASSERT_EQ(Ca->data.integer_double[4], 21.0f);
            TEST_ASSERT_EQ(Ca->data.integer_double[5], 10.0f);
            TEST_ASSERT_EQ(Ca->data.integer_double[6], 66.0f);
            TEST_ASSERT_EQ(Ca->data.integer_double[7], 12.0f);
            TEST_ASSERT_EQ(Ca->data.integer_double[8],  3.0f);
            mtxmatrix_free(&C);
        }
        {
            err = mtxmatrix_init_entries_integer_double(
                &C, mtxblasdense, mtx_unsymmetric, M, K, Cnnz, Crowidx, Ccolidx, Cdata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemm(mtx_trans, mtx_notrans, 2.0f, &A, &B, 3.0f, &C, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxblasdense, C.type);
            const struct mtxbasevector * Ca = &C.storage.dense.a;
            TEST_ASSERT_EQ(mtx_field_integer, Ca->field);
            TEST_ASSERT_EQ(mtx_double, Ca->precision);
            TEST_ASSERT_EQ(9, Ca->size);
            TEST_ASSERT_EQ(Ca->data.integer_double[0], 17.0f);
            TEST_ASSERT_EQ(Ca->data.integer_double[1], 14.0f);
            TEST_ASSERT_EQ(Ca->data.integer_double[2],  4.0f);
            TEST_ASSERT_EQ(Ca->data.integer_double[3], 44.0f);
            TEST_ASSERT_EQ(Ca->data.integer_double[4], 21.0f);
            TEST_ASSERT_EQ(Ca->data.integer_double[5], 14.0f);
            TEST_ASSERT_EQ(Ca->data.integer_double[6], 48.0f);
            TEST_ASSERT_EQ(Ca->data.integer_double[7], 12.0f);
            TEST_ASSERT_EQ(Ca->data.integer_double[8],  3.0f);
            mtxmatrix_free(&C);
        }
        {
            err = mtxmatrix_init_entries_integer_double(
                &C, mtxblasdense, mtx_unsymmetric, M, K, Cnnz, Crowidx, Ccolidx, Cdata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemm(mtx_trans, mtx_trans, 2.0f, &A, &B, 3.0f, &C, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxblasdense, C.type);
            const struct mtxbasevector * Ca = &C.storage.dense.a;
            TEST_ASSERT_EQ(mtx_field_integer, Ca->field);
            TEST_ASSERT_EQ(mtx_double, Ca->precision);
            TEST_ASSERT_EQ(9, Ca->size);
            TEST_ASSERT_EQ(Ca->data.integer_double[0], 17.0f);
            TEST_ASSERT_EQ(Ca->data.integer_double[1], 14.0f);
            TEST_ASSERT_EQ(Ca->data.integer_double[2],  2.0f);
            TEST_ASSERT_EQ(Ca->data.integer_double[3], 56.0f);
            TEST_ASSERT_EQ(Ca->data.integer_double[4], 21.0f);
            TEST_ASSERT_EQ(Ca->data.integer_double[5], 10.0f);
            TEST_ASSERT_EQ(Ca->data.integer_double[6], 66.0f);
            TEST_ASSERT_EQ(Ca->data.integer_double[7], 12.0f);
            TEST_ASSERT_EQ(Ca->data.integer_double[8],  3.0f);
            mtxmatrix_free(&C);
        }
        mtxmatrix_free(&B);
        mtxmatrix_free(&A);
    }

    /*
     * d) For symmetric complex matrices, calculate
     *
     *   ⎡ 1+2i 3+4i⎤  ⎡ 3+1i 0+1i⎤     ⎡ 1+0i 2+2i⎤  ⎡-5+34i  8+16i⎤
     * 2*⎢          ⎥ *⎢          ⎥ + 3*⎢          ⎥ =⎢             ⎥,
     *   ⎣ 3+4i 7+8i⎦  ⎣ 1+2i 1+0i⎦     ⎣ 2+2i 0+1i⎦  ⎣-2+80i  6+25i⎦
     *
     * and the transposed and conjugate transposed products
     *
     *   ⎡ 1+2i 3+4i⎤  ⎡ 3+1i 1+2i⎤     ⎡ 1+0i 2+2i⎤  ⎡-3+20i  6+22i⎤
     * 2*⎢          ⎥ *⎢          ⎥ + 3*⎢          ⎥ =⎢             ⎥,
     *   ⎣ 3+4i 7+8i⎦  ⎣ 0+1i 1+0i⎦     ⎣ 2+2i 0+1i⎦  ⎣   50i  4+39i⎦
     *
     *   ⎡ 1-2i 3-4i⎤  ⎡ 3+1i 1+2i⎤     ⎡ 1+0i 2+2i⎤  ⎡-35-6i  16   ⎤
     * 2*⎢          ⎥ *⎢          ⎥ + 3*⎢          ⎥ =⎢             ⎥,
     *   ⎣ 3-4i 7-8i⎦  ⎣ 0+1i 1+0i⎦     ⎣ 2+2i 0+1i⎦  ⎣ 78     22-7i⎦
     */

    /* Complex, single precision, symmetric matrices */
    {
        struct mtxmatrix A;
        struct mtxmatrix B;
        struct mtxmatrix C;
        int M = 2, N = 2, K = 2;
        int Annz = 3;
        int Arowidx[] = {0, 0, 1};
        int Acolidx[] = {0, 1, 1};
        float Adata[][2] = {{1.0f,2.0f}, {3.0f,4.0f}, {7.0f,8.0f}};
        int Bnnz = 4;
        int Browidx[] = {0, 0, 1, 1};
        int Bcolidx[] = {0, 1, 0, 1};
        float Bdata[][2] = {{3.0f,1.0f}, {0.0f,1.0f}, {1.0f,2.0f}, {1.0f,0.0f}};
        int Cnnz = 4;
        int Crowidx[] = {0, 0, 1, 1};
        int Ccolidx[] = {0, 1, 0, 1};
        float Cdata[][2] = {{1.0f,0.0f}, {2.0f,2.0f}, {2.0f,2.0f}, {0.0f,1.0f}};
        err = mtxmatrix_init_entries_complex_single(
            &A, mtxblasdense, mtx_symmetric, M, N, Annz, Arowidx, Acolidx, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxmatrix_init_entries_complex_single(
            &B, mtxblasdense, mtx_unsymmetric, N, K, Bnnz, Browidx, Bcolidx, Bdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        {
            err = mtxmatrix_init_entries_complex_single(
                &C, mtxblasdense, mtx_unsymmetric, M, K, Cnnz, Crowidx, Ccolidx, Cdata);
            err = mtxmatrix_sgemm(
                mtx_notrans, mtx_notrans, 2.0f, &A, &B, 3.0f, &C, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxblasdense, C.type);
            const struct mtxbasevector * Ca = &C.storage.dense.a;
            TEST_ASSERT_EQ(mtx_field_complex, Ca->field);
            TEST_ASSERT_EQ(mtx_single, Ca->precision);
            TEST_ASSERT_EQ(4, Ca->size);
            TEST_ASSERT_EQ(Ca->data.complex_single[0][0], -5.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[0][1], 34.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[1][0],  8.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[1][1], 16.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[2][0], -2.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[2][1], 80.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[3][0],  6.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[3][1], 25.0f);
            mtxmatrix_free(&C);
        }
        {
            err = mtxmatrix_init_entries_complex_single(
                &C, mtxblasdense, mtx_unsymmetric, M, K, Cnnz, Crowidx, Ccolidx, Cdata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemm(mtx_notrans, mtx_trans, 2.0f, &A, &B, 3.0f, &C, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxblasdense, C.type);
            const struct mtxbasevector * Ca = &C.storage.dense.a;
            TEST_ASSERT_EQ(mtx_field_complex, Ca->field);
            TEST_ASSERT_EQ(mtx_single, Ca->precision);
            TEST_ASSERT_EQ(4, Ca->size);
            TEST_ASSERT_EQ(Ca->data.complex_single[0][0], -3.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[0][1], 20.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[1][0],  6.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[1][1], 22.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[2][0],  0.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[2][1], 50.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[3][0],  4.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[3][1], 39.0f);
            mtxmatrix_free(&C);
        }
        {
            err = mtxmatrix_init_entries_complex_single(
                &C, mtxblasdense, mtx_unsymmetric, M, K, Cnnz, Crowidx, Ccolidx, Cdata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemm(mtx_trans, mtx_notrans, 2.0f, &A, &B, 3.0f, &C, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxblasdense, C.type);
            const struct mtxbasevector * Ca = &C.storage.dense.a;
            TEST_ASSERT_EQ(mtx_field_complex, Ca->field);
            TEST_ASSERT_EQ(mtx_single, Ca->precision);
            TEST_ASSERT_EQ(4, Ca->size);
            TEST_ASSERT_EQ(Ca->data.complex_single[0][0], -5.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[0][1], 34.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[1][0],  8.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[1][1], 16.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[2][0], -2.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[2][1], 80.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[3][0],  6.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[3][1], 25.0f);
            mtxmatrix_free(&C);
        }
        {
            err = mtxmatrix_init_entries_complex_single(
                &C, mtxblasdense, mtx_unsymmetric, M, K, Cnnz, Crowidx, Ccolidx, Cdata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemm(mtx_trans, mtx_trans, 2.0f, &A, &B, 3.0f, &C, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxblasdense, C.type);
            const struct mtxbasevector * Ca = &C.storage.dense.a;
            TEST_ASSERT_EQ(mtx_field_complex, Ca->field);
            TEST_ASSERT_EQ(mtx_single, Ca->precision);
            TEST_ASSERT_EQ(4, Ca->size);
            TEST_ASSERT_EQ(Ca->data.complex_single[0][0], -3.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[0][1], 20.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[1][0],  6.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[1][1], 22.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[2][0],  0.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[2][1], 50.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[3][0],  4.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[3][1], 39.0f);
            mtxmatrix_free(&C);
        }
        {
            err = mtxmatrix_init_entries_complex_single(
                &C, mtxblasdense, mtx_unsymmetric, M, K, Cnnz, Crowidx, Ccolidx, Cdata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemm(mtx_conjtrans, mtx_notrans, 2.0f, &A, &B, 3.0f, &C, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxblasdense, C.type);
            const struct mtxbasevector * Ca = &C.storage.dense.a;
            TEST_ASSERT_EQ(mtx_field_complex, Ca->field);
            TEST_ASSERT_EQ(mtx_single, Ca->precision);
            TEST_ASSERT_EQ(4, Ca->size);
            TEST_ASSERT_EQ(Ca->data.complex_single[0][0], 35.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[0][1], -6.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[1][0], 16.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[1][1],  0.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[2][0], 78.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[2][1],  0.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[3][0], 22.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[3][1], -7.0f);
            mtxmatrix_free(&C);
        }
        mtxmatrix_free(&B);
        mtxmatrix_free(&A);
    }

    /* Complex, double precision, symmetric matrices */
    {
        struct mtxmatrix A;
        struct mtxmatrix B;
        struct mtxmatrix C;
        int M = 2, N = 2, K = 2;
        int Annz = 3;
        int Arowidx[] = {0, 0, 1};
        int Acolidx[] = {0, 1, 1};
        double Adata[][2] = {{1.0,2.0}, {3.0,4.0}, {7.0,8.0}};
        int Bnnz = 4;
        int Browidx[] = {0, 0, 1, 1};
        int Bcolidx[] = {0, 1, 0, 1};
        double Bdata[][2] = {{3.0,1.0}, {0.0,1.0}, {1.0,2.0}, {1.0,0.0}};
        int Cnnz = 4;
        int Crowidx[] = {0, 0, 1, 1};
        int Ccolidx[] = {0, 1, 0, 1};
        double Cdata[][2] = {{1.0,0.0}, {2.0,2.0}, {2.0,2.0}, {0.0,1.0}};
        err = mtxmatrix_init_entries_complex_double(
            &A, mtxblasdense, mtx_symmetric, M, N, Annz, Arowidx, Acolidx, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxmatrix_init_entries_complex_double(
            &B, mtxblasdense, mtx_unsymmetric, N, K, Bnnz, Browidx, Bcolidx, Bdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        {
            err = mtxmatrix_init_entries_complex_double(
                &C, mtxblasdense, mtx_unsymmetric, M, K, Cnnz, Crowidx, Ccolidx, Cdata);
            err = mtxmatrix_sgemm(
                mtx_notrans, mtx_notrans, 2.0f, &A, &B, 3.0f, &C, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxblasdense, C.type);
            const struct mtxbasevector * Ca = &C.storage.dense.a;
            TEST_ASSERT_EQ(mtx_field_complex, Ca->field);
            TEST_ASSERT_EQ(mtx_double, Ca->precision);
            TEST_ASSERT_EQ(4, Ca->size);
            TEST_ASSERT_EQ(Ca->data.complex_double[0][0], -5.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[0][1], 34.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[1][0],  8.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[1][1], 16.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[2][0], -2.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[2][1], 80.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[3][0],  6.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[3][1], 25.0);
            mtxmatrix_free(&C);
        }
        {
            err = mtxmatrix_init_entries_complex_double(
                &C, mtxblasdense, mtx_unsymmetric, M, K, Cnnz, Crowidx, Ccolidx, Cdata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemm(mtx_notrans, mtx_trans, 2.0f, &A, &B, 3.0f, &C, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxblasdense, C.type);
            const struct mtxbasevector * Ca = &C.storage.dense.a;
            TEST_ASSERT_EQ(mtx_field_complex, Ca->field);
            TEST_ASSERT_EQ(mtx_double, Ca->precision);
            TEST_ASSERT_EQ(4, Ca->size);
            TEST_ASSERT_EQ(Ca->data.complex_double[0][0], -3.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[0][1], 20.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[1][0],  6.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[1][1], 22.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[2][0],  0.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[2][1], 50.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[3][0],  4.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[3][1], 39.0);
            mtxmatrix_free(&C);
        }
        {
            err = mtxmatrix_init_entries_complex_double(
                &C, mtxblasdense, mtx_unsymmetric, M, K, Cnnz, Crowidx, Ccolidx, Cdata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemm(mtx_trans, mtx_notrans, 2.0f, &A, &B, 3.0f, &C, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxblasdense, C.type);
            const struct mtxbasevector * Ca = &C.storage.dense.a;
            TEST_ASSERT_EQ(mtx_field_complex, Ca->field);
            TEST_ASSERT_EQ(mtx_double, Ca->precision);
            TEST_ASSERT_EQ(4, Ca->size);
            TEST_ASSERT_EQ(Ca->data.complex_double[0][0], -5.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[0][1], 34.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[1][0],  8.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[1][1], 16.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[2][0], -2.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[2][1], 80.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[3][0],  6.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[3][1], 25.0);
            mtxmatrix_free(&C);
        }
        {
            err = mtxmatrix_init_entries_complex_double(
                &C, mtxblasdense, mtx_unsymmetric, M, K, Cnnz, Crowidx, Ccolidx, Cdata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemm(mtx_trans, mtx_trans, 2.0f, &A, &B, 3.0f, &C, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxblasdense, C.type);
            const struct mtxbasevector * Ca = &C.storage.dense.a;
            TEST_ASSERT_EQ(mtx_field_complex, Ca->field);
            TEST_ASSERT_EQ(mtx_double, Ca->precision);
            TEST_ASSERT_EQ(4, Ca->size);
            TEST_ASSERT_EQ(Ca->data.complex_double[0][0], -3.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[0][1], 20.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[1][0],  6.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[1][1], 22.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[2][0],  0.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[2][1], 50.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[3][0],  4.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[3][1], 39.0);
            mtxmatrix_free(&C);
        }
        {
            err = mtxmatrix_init_entries_complex_double(
                &C, mtxblasdense, mtx_unsymmetric, M, K, Cnnz, Crowidx, Ccolidx, Cdata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemm(mtx_conjtrans, mtx_notrans, 2.0f, &A, &B, 3.0f, &C, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxblasdense, C.type);
            const struct mtxbasevector * Ca = &C.storage.dense.a;
            TEST_ASSERT_EQ(mtx_field_complex, Ca->field);
            TEST_ASSERT_EQ(mtx_double, Ca->precision);
            TEST_ASSERT_EQ(4, Ca->size);
            TEST_ASSERT_EQ(Ca->data.complex_double[0][0], 35.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[0][1], -6.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[1][0], 16.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[1][1],  0.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[2][0], 78.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[2][1],  0.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[3][0], 22.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[3][1], -7.0);
            mtxmatrix_free(&C);
        }
        mtxmatrix_free(&B);
        mtxmatrix_free(&A);
    }

    /*
     * e) For Hermitian complex matrices, calculate
     *
     *   ⎡ 1    3+4i⎤  ⎡ 3+1i 0+1i⎤     ⎡ 1+0i 2+2i⎤  ⎡-1+22i 12+16i⎤
     * 2*⎢          ⎥ *⎢          ⎥ + 3*⎢          ⎥ =⎢             ⎥,
     *   ⎣ 3-4i 7   ⎦  ⎣ 1+2i 1+0i⎦     ⎣ 2+2i 0+1i⎦  ⎣46+16i 22+9i ⎦
     *
     * and the transposed products
     *
     *   ⎡ 1    3+4i⎤  ⎡ 3+1i 1+2i⎤     ⎡ 1+0i 2+2i⎤  ⎡ 1+8i  14+18i⎤
     * 2*⎢          ⎥ *⎢          ⎥ + 3*⎢          ⎥ =⎢             ⎥,
     *   ⎣ 3-4i 7   ⎦  ⎣ 0+1i 1+0i⎦     ⎣ 2+2i 0+1i⎦  ⎣32+2i   36+7i⎦
     *
     *   ⎡ 1    3-4i⎤  ⎡ 3+1i 0+1i⎤     ⎡ 1+0i 2+2i⎤  ⎡31+6i   12   ⎤
     * 2*⎢          ⎥ *⎢          ⎥ + 3*⎢          ⎥ =⎢             ⎥,
     *   ⎣ 3+4i 7   ⎦  ⎣ 1+2i 1+0i⎦     ⎣ 2+2i 0+1i⎦  ⎣30+64i  6+9i ⎦
     *
     *   ⎡ 1    3-4i⎤  ⎡ 3+1i 1+2i⎤     ⎡ 1+0i 2+2i⎤  ⎡17+8i   14+2i⎤
     * 2*⎢          ⎥ *⎢          ⎥ + 3*⎢          ⎥ =⎢             ⎥,
     *   ⎣ 3+4i 7   ⎦  ⎣ 0+1i 1+0i⎦     ⎣ 2+2i 0+1i⎦  ⎣16+50i  4+23i⎦
     */

    /* Complex, single precision, Hermitian matrices */
    {
        struct mtxmatrix A;
        struct mtxmatrix B;
        struct mtxmatrix C;
        int M = 2, N = 2, K = 2;
        int Annz = 3;
        int Arowidx[] = {0, 0, 1};
        int Acolidx[] = {0, 1, 1};
        float Adata[][2] = {{1.0f,0.0f}, {3.0f,4.0f}, {7.0f,0.0f}};
        int Bnnz = 4;
        int Browidx[] = {0, 0, 1, 1};
        int Bcolidx[] = {0, 1, 0, 1};
        float Bdata[][2] = {{3.0f,1.0f}, {0.0f,1.0f}, {1.0f,2.0f}, {1.0f,0.0f}};
        int Cnnz = 4;
        int Crowidx[] = {0, 0, 1, 1};
        int Ccolidx[] = {0, 1, 0, 1};
        float Cdata[][2] = {{1.0f,0.0f}, {2.0f,2.0f}, {2.0f,2.0f}, {0.0f,1.0f}};
        err = mtxmatrix_init_entries_complex_single(
            &A, mtxblasdense, mtx_hermitian, M, N, Annz, Arowidx, Acolidx, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxmatrix_init_entries_complex_single(
            &B, mtxblasdense, mtx_unsymmetric, N, K, Bnnz, Browidx, Bcolidx, Bdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        {
            err = mtxmatrix_init_entries_complex_single(
                &C, mtxblasdense, mtx_unsymmetric, M, K, Cnnz, Crowidx, Ccolidx, Cdata);
            err = mtxmatrix_sgemm(
                mtx_notrans, mtx_notrans, 2.0f, &A, &B, 3.0f, &C, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxblasdense, C.type);
            const struct mtxbasevector * Ca = &C.storage.dense.a;
            TEST_ASSERT_EQ(mtx_field_complex, Ca->field);
            TEST_ASSERT_EQ(mtx_single, Ca->precision);
            TEST_ASSERT_EQ(4, Ca->size);
            TEST_ASSERT_EQ(Ca->data.complex_single[0][0], -1.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[0][1], 22.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[1][0], 12.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[1][1], 16.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[2][0], 46.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[2][1], 16.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[3][0], 22.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[3][1],  9.0f);
            mtxmatrix_free(&C);
        }
        {
            err = mtxmatrix_init_entries_complex_single(
                &C, mtxblasdense, mtx_unsymmetric, M, K, Cnnz, Crowidx, Ccolidx, Cdata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemm(mtx_notrans, mtx_trans, 2.0f, &A, &B, 3.0f, &C, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxblasdense, C.type);
            const struct mtxbasevector * Ca = &C.storage.dense.a;
            TEST_ASSERT_EQ(mtx_field_complex, Ca->field);
            TEST_ASSERT_EQ(mtx_single, Ca->precision);
            TEST_ASSERT_EQ(4, Ca->size);
            TEST_ASSERT_EQ(Ca->data.complex_single[0][0],  1.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[0][1],  8.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[1][0], 14.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[1][1], 18.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[2][0], 32.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[2][1],  2.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[3][0], 36.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[3][1],  7.0f);
            mtxmatrix_free(&C);
        }
        {
            err = mtxmatrix_init_entries_complex_single(
                &C, mtxblasdense, mtx_unsymmetric, M, K, Cnnz, Crowidx, Ccolidx, Cdata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemm(mtx_trans, mtx_notrans, 2.0f, &A, &B, 3.0f, &C, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxblasdense, C.type);
            const struct mtxbasevector * Ca = &C.storage.dense.a;
            TEST_ASSERT_EQ(mtx_field_complex, Ca->field);
            TEST_ASSERT_EQ(mtx_single, Ca->precision);
            TEST_ASSERT_EQ(4, Ca->size);
            TEST_ASSERT_EQ(Ca->data.complex_single[0][0], 31.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[0][1],  6.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[1][0], 12.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[1][1],  0.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[2][0], 30.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[2][1], 64.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[3][0],  6.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[3][1],  9.0f);
            mtxmatrix_free(&C);
        }
        {
            err = mtxmatrix_init_entries_complex_single(
                &C, mtxblasdense, mtx_unsymmetric, M, K, Cnnz, Crowidx, Ccolidx, Cdata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemm(mtx_trans, mtx_trans, 2.0f, &A, &B, 3.0f, &C, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxblasdense, C.type);
            const struct mtxbasevector * Ca = &C.storage.dense.a;
            TEST_ASSERT_EQ(mtx_field_complex, Ca->field);
            TEST_ASSERT_EQ(mtx_single, Ca->precision);
            TEST_ASSERT_EQ(4, Ca->size);
            TEST_ASSERT_EQ(Ca->data.complex_single[0][0], 17.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[0][1],  8.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[1][0], 14.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[1][1],  2.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[2][0], 16.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[2][1], 50.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[3][0],  4.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[3][1], 23.0f);
            mtxmatrix_free(&C);
        }
        {
            err = mtxmatrix_init_entries_complex_single(
                &C, mtxblasdense, mtx_unsymmetric, M, K, Cnnz, Crowidx, Ccolidx, Cdata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemm(mtx_conjtrans, mtx_notrans, 2.0f, &A, &B, 3.0f, &C, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxblasdense, C.type);
            const struct mtxbasevector * Ca = &C.storage.dense.a;
            TEST_ASSERT_EQ(mtx_field_complex, Ca->field);
            TEST_ASSERT_EQ(mtx_single, Ca->precision);
            TEST_ASSERT_EQ(4, Ca->size);
            TEST_ASSERT_EQ(Ca->data.complex_single[0][0], -1.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[0][1], 22.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[1][0], 12.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[1][1], 16.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[2][0], 46.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[2][1], 16.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[3][0], 22.0f);
            TEST_ASSERT_EQ(Ca->data.complex_single[3][1],  9.0f);
            mtxmatrix_free(&C);
        }
        mtxmatrix_free(&B);
        mtxmatrix_free(&A);
    }

    /* Complex, double precision, Hermitian matrices */
    {
        struct mtxmatrix A;
        struct mtxmatrix B;
        struct mtxmatrix C;
        int M = 2, N = 2, K = 2;
        int Annz = 3;
        int Arowidx[] = {0, 0, 1};
        int Acolidx[] = {0, 1, 1};
        double Adata[][2] = {{1.0,0.0}, {3.0,4.0}, {7.0,0.0}};
        int Bnnz = 4;
        int Browidx[] = {0, 0, 1, 1};
        int Bcolidx[] = {0, 1, 0, 1};
        double Bdata[][2] = {{3.0,1.0}, {0.0,1.0}, {1.0,2.0}, {1.0,0.0}};
        int Cnnz = 4;
        int Crowidx[] = {0, 0, 1, 1};
        int Ccolidx[] = {0, 1, 0, 1};
        double Cdata[][2] = {{1.0,0.0}, {2.0,2.0}, {2.0,2.0}, {0.0,1.0}};
        err = mtxmatrix_init_entries_complex_double(
            &A, mtxblasdense, mtx_hermitian, M, N, Annz, Arowidx, Acolidx, Adata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxmatrix_init_entries_complex_double(
            &B, mtxblasdense, mtx_unsymmetric, N, K, Bnnz, Browidx, Bcolidx, Bdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        {
            err = mtxmatrix_init_entries_complex_double(
                &C, mtxblasdense, mtx_unsymmetric, M, K, Cnnz, Crowidx, Ccolidx, Cdata);
            err = mtxmatrix_sgemm(
                mtx_notrans, mtx_notrans, 2.0f, &A, &B, 3.0f, &C, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxblasdense, C.type);
            const struct mtxbasevector * Ca = &C.storage.dense.a;
            TEST_ASSERT_EQ(mtx_field_complex, Ca->field);
            TEST_ASSERT_EQ(mtx_double, Ca->precision);
            TEST_ASSERT_EQ(4, Ca->size);
            TEST_ASSERT_EQ(Ca->data.complex_double[0][0], -1.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[0][1], 22.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[1][0], 12.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[1][1], 16.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[2][0], 46.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[2][1], 16.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[3][0], 22.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[3][1],  9.0);
            mtxmatrix_free(&C);
        }
        {
            err = mtxmatrix_init_entries_complex_double(
                &C, mtxblasdense, mtx_unsymmetric, M, K, Cnnz, Crowidx, Ccolidx, Cdata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemm(mtx_notrans, mtx_trans, 2.0f, &A, &B, 3.0f, &C, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxblasdense, C.type);
            const struct mtxbasevector * Ca = &C.storage.dense.a;
            TEST_ASSERT_EQ(mtx_field_complex, Ca->field);
            TEST_ASSERT_EQ(mtx_double, Ca->precision);
            TEST_ASSERT_EQ(4, Ca->size);
            TEST_ASSERT_EQ(Ca->data.complex_double[0][0],  1.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[0][1],  8.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[1][0], 14.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[1][1], 18.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[2][0], 32.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[2][1],  2.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[3][0], 36.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[3][1],  7.0);
            mtxmatrix_free(&C);
        }
        {
            err = mtxmatrix_init_entries_complex_double(
                &C, mtxblasdense, mtx_unsymmetric, M, K, Cnnz, Crowidx, Ccolidx, Cdata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemm(mtx_trans, mtx_notrans, 2.0f, &A, &B, 3.0f, &C, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxblasdense, C.type);
            const struct mtxbasevector * Ca = &C.storage.dense.a;
            TEST_ASSERT_EQ(mtx_field_complex, Ca->field);
            TEST_ASSERT_EQ(mtx_double, Ca->precision);
            TEST_ASSERT_EQ(4, Ca->size);
            TEST_ASSERT_EQ(Ca->data.complex_double[0][0], 31.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[0][1],  6.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[1][0], 12.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[1][1],  0.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[2][0], 30.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[2][1], 64.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[3][0],  6.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[3][1],  9.0);
            mtxmatrix_free(&C);
        }
        {
            err = mtxmatrix_init_entries_complex_double(
                &C, mtxblasdense, mtx_unsymmetric, M, K, Cnnz, Crowidx, Ccolidx, Cdata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemm(mtx_trans, mtx_trans, 2.0f, &A, &B, 3.0f, &C, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxblasdense, C.type);
            const struct mtxbasevector * Ca = &C.storage.dense.a;
            TEST_ASSERT_EQ(mtx_field_complex, Ca->field);
            TEST_ASSERT_EQ(mtx_double, Ca->precision);
            TEST_ASSERT_EQ(4, Ca->size);
            TEST_ASSERT_EQ(Ca->data.complex_double[0][0], 17.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[0][1],  8.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[1][0], 14.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[1][1],  2.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[2][0], 16.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[2][1], 50.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[3][0],  4.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[3][1], 23.0);
            mtxmatrix_free(&C);
        }
        {
            err = mtxmatrix_init_entries_complex_double(
                &C, mtxblasdense, mtx_unsymmetric, M, K, Cnnz, Crowidx, Ccolidx, Cdata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_sgemm(mtx_conjtrans, mtx_notrans, 2.0f, &A, &B, 3.0f, &C, NULL);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxblasdense, C.type);
            const struct mtxbasevector * Ca = &C.storage.dense.a;
            TEST_ASSERT_EQ(mtx_field_complex, Ca->field);
            TEST_ASSERT_EQ(mtx_double, Ca->precision);
            TEST_ASSERT_EQ(4, Ca->size);
            TEST_ASSERT_EQ(Ca->data.complex_double[0][0], -1.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[0][1], 22.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[1][0], 12.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[1][1], 16.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[2][0], 46.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[2][1], 16.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[3][0], 22.0);
            TEST_ASSERT_EQ(Ca->data.complex_double[3][1],  9.0);
            mtxmatrix_free(&C);
        }
        mtxmatrix_free(&B);
        mtxmatrix_free(&A);
    }
    return TEST_SUCCESS;
}

/**
 * ‘main()’ entry point and test driver.
 */
int main(int argc, char * argv[])
{
    TEST_SUITE_BEGIN("Running tests for dense matrices with BLAS-acclerated operations\n");
    TEST_RUN(test_mtxblasdense_gemv);
    TEST_RUN(test_mtxblasdense_gemm);
    TEST_SUITE_END();
    return (TEST_SUITE_STATUS == TEST_SUCCESS) ?
        EXIT_SUCCESS : EXIT_FAILURE;
}
