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
 * Last modified: 2022-02-26
 *
 * Unit tests for distributed matrix-vector multiplication.
 */

#include "test.h"

#include <libmtx/libmtx-config.h>

#include <libmtx/error.h>
#include <libmtx/matrix/distmatrix.h>
#include <libmtx/mtxfile/mtxdistfile.h>
#include <libmtx/mtxfile/mtxfile.h>
#include <libmtx/util/partition.h>
#include <libmtx/vector/distvector.h>

#include <mpi.h>

#include <errno.h>

#include <stdlib.h>

const char * program_invocation_short_name = "test_mtxdistmatrixgemv";

/**
 * ‘test_mtxdistmatrixgemv_array()’ tests computing matrix-vector
 * products for matrices in array format.
 */
int test_mtxdistmatrixgemv_array(void)
{
    int err;
    char mpierrstr[MPI_MAX_ERROR_STRING];
    int mpierrstrlen;
    MPI_Comm comm = MPI_COMM_WORLD;
    int root = 0;

    int comm_size;
    err = MPI_Comm_size(comm, &comm_size);
    if (err) {
        MPI_Error_string(err, mpierrstr, &mpierrstrlen);
        fprintf(stderr, "%s: MPI_Comm_size failed with %s\n",
                program_invocation_short_name, mpierrstr);
        MPI_Abort(comm, EXIT_FAILURE);
    }
    int rank;
    err = MPI_Comm_rank(comm, &rank);
    if (err) {
        MPI_Error_string(err, mpierrstr, &mpierrstrlen);
        fprintf(stderr, "%s: MPI_Comm_rank failed with %s\n",
                program_invocation_short_name, mpierrstr);
        MPI_Abort(comm, EXIT_FAILURE);
    }
    if (comm_size != 2)
        TEST_FAIL_MSG("Expected exactly two MPI processes");

    struct mtxdisterror disterr;
    err = mtxdisterror_alloc(&disterr, comm, NULL);
    if (err)
        MPI_Abort(comm, EXIT_FAILURE);

    /*
     * For real or integer matrices, calculate
     *
     *   ⎡ 1 2 3⎤  ⎡ 3⎤     ⎡ 1⎤  ⎡ 20⎤  ⎡ 3⎤  ⎡ 23⎤
     * 2*⎢ 4 5 6⎥ *⎢ 2⎥ + 3*⎢ 0⎥ =⎢ 56⎥ +⎢ 0⎥ =⎢ 56⎥,
     *   ⎣ 7 8 9⎦  ⎣ 1⎦     ⎣ 2⎦  ⎣ 92⎦  ⎣ 6⎦  ⎣ 98⎦
     *
     * and
     *
     *   ⎡ 1 4 7⎤  ⎡ 3⎤     ⎡ 1⎤  ⎡ 36⎤  ⎡ 3⎤  ⎡ 39⎤
     * 2*⎢ 2 5 8⎥ *⎢ 2⎥ + 3*⎢ 0⎥ =⎢ 48⎥ +⎢ 0⎥ =⎢ 48⎥.
     *   ⎣ 3 6 9⎦  ⎣ 1⎦     ⎣ 2⎦  ⎣ 60⎦  ⎣ 6⎦  ⎣ 66⎦
     *
     * For complex matrices, calculate
     *
     *   ⎡ 1+2i 3+4i⎤  ⎡ 3+1i⎤     ⎡ 1+0i⎤  ⎡-8+34i⎤  ⎡ 3   ⎤  ⎡-5+34i⎤
     * 2*⎢          ⎥ *⎢     ⎥ + 3*⎢     ⎥ =⎢      ⎥ +⎢     ⎥ =⎢      ⎥,
     *   ⎣ 5+6i 7+8i⎦  ⎣ 1+2i⎦     ⎣ 2+2i⎦  ⎣ 0+90i⎦  ⎣ 6+6i⎦  ⎣ 6+96i⎦
     *
     * and
     *
     *   ⎡ 1+2i 5+6i⎤  ⎡ 3+1i⎤     ⎡ 1+0i⎤  ⎡-12+46i⎤  ⎡ 3   ⎤  ⎡-9+46i⎤
     * 2*⎢          ⎥ *⎢     ⎥ + 3*⎢     ⎥ =⎢       ⎥ +⎢     ⎥ =⎢      ⎥,
     *   ⎣ 3+4i 7+8i⎦  ⎣ 1+2i⎦     ⎣ 2+2i⎦  ⎣ -8+74i⎦  ⎣ 6+6i⎦  ⎣-2+80i⎦
     *
     * and
     *
     *   ⎡ 1-2i 5-6i⎤  ⎡ 3+1i⎤     ⎡ 1+0i⎤  ⎡ 44-2i⎤  ⎡ 3   ⎤  ⎡ 47-2i⎤
     * 2*⎢          ⎥ *⎢     ⎥ + 3*⎢     ⎥ =⎢      ⎥ +⎢     ⎥ =⎢      ⎥.
     *   ⎣ 3-4i 7-8i⎦  ⎣ 1+2i⎦     ⎣ 2+2i⎦  ⎣ 72-6i⎦  ⎣ 6+6i⎦  ⎣ 78   ⎦
     *
     * and
     *
     *    ⎡ 1+2i 3+4i⎤  ⎡ 3+1i⎤          ⎡ 1+0i⎤  ⎡-34-8i⎤  ⎡ 3+1i⎤  ⎡-31-7i⎤
     * 2i*⎢          ⎥ *⎢     ⎥ + (3+1i)*⎢     ⎥ =⎢      ⎥ +⎢     ⎥ =⎢      ⎥,
     *    ⎣ 5+6i 7+8i⎦  ⎣ 1+2i⎦          ⎣ 2+2i⎦  ⎣-90   ⎦  ⎣ 4+8i⎦  ⎣-86+8i⎦
     *
     * and
     *
     *    ⎡ 1+2i 5+6i⎤  ⎡ 3+1i⎤          ⎡ 1+0i⎤  ⎡-46-12i⎤  ⎡ 3+1i⎤  ⎡-43-11i⎤
     * 2i*⎢          ⎥ *⎢     ⎥ + (3+1i)*⎢     ⎥ =⎢       ⎥ +⎢     ⎥ =⎢       ⎥,
     *    ⎣ 3+4i 7+8i⎦  ⎣ 1+2i⎦          ⎣ 2+2i⎦  ⎣-74- 8i⎦  ⎣ 4+8i⎦  ⎣-70    ⎦
     *
     * and
     *
     *    ⎡ 1-2i 5-6i⎤  ⎡ 3+1i⎤          ⎡ 1+0i⎤  ⎡ 2+44i⎤  ⎡ 3+1i⎤  ⎡  5+45i⎤
     * 2i*⎢          ⎥ *⎢     ⎥ + (3+1i)*⎢     ⎥ =⎢      ⎥ +⎢     ⎥ =⎢       ⎥.
     *    ⎣ 3-4i 7-8i⎦  ⎣ 1+2i⎦          ⎣ 2+2i⎦  ⎣ 6+72i⎦  ⎣ 4+8i⎦  ⎣ 10+80i⎦
     */

    /*
     * Real matrices
     */

    {
        struct mtxdistmatrix A;
        struct mtxdistvector x;
        struct mtxdistvector y;
        int num_local_rows = rank == 0 ? 2 : 1;
        int num_local_columns = 3;
        float * Adata = rank == 0
            ? ((float[]) {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f})
            : ((float[]) {7.0f, 8.0f, 9.0f});
        float * xdata = (float[]) {3.0f, 2.0f, 1.0f};
        float * ydata = rank == 0
            ? ((float[]) {1.0f, 0.0f})
            : ((float[]) {2.0f});

        struct mtxpartition rowpart;
        err = mtxpartition_init_block(&rowpart, 3, comm_size, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxpartition colpart;
        err = mtxpartition_init_singleton(&colpart, 3);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxdistmatrix_init_array_real_single(
            &A, num_local_rows, num_local_columns,
            Adata, &rowpart, &colpart, comm, 0, 0, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));

        err = mtxdistvector_init_array_real_single(
            &x, num_local_columns, xdata, NULL, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        mtxpartition_free(&colpart);
        mtxpartition_free(&rowpart);
        {
            err = mtxdistvector_init_array_real_single(
                &y, num_local_rows, ydata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistmatrix_sgemv(
                mtx_notrans, 2.0f, &A, &x, 3.0f, &y, &disterr);
            TEST_ASSERT_EQ_MSG(
                MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
                ? mtxdisterror_description(&disterr) : mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.interior.type);
            const struct mtxvector_array * y_ = &y.interior.storage.array;
            TEST_ASSERT_EQ(mtx_field_real, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(2, y_->size);
                TEST_ASSERT_EQ(y_->data.real_single[0], 23.0f);
                TEST_ASSERT_EQ(y_->data.real_single[1], 56.0f);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.real_single[0], 98.0f);
            }
            mtxdistvector_free(&y);
        }
        mtxdistvector_free(&x);
        mtxdistmatrix_free(&A);
    }

    {
        struct mtxdistmatrix A;
        struct mtxdistvector x;
        struct mtxdistvector y;
        int num_local_rows = rank == 0 ? 2 : 1;
        int num_local_columns = 3;
        float * Adata = rank == 0
            ? ((float[]) {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f})
            : ((float[]) {7.0f, 8.0f, 9.0f});
        float * xdata = (float[]) {3.0f, 2.0f, 1.0f};
        float * ydata = rank == 0
            ? ((float[]) {1.0f, 0.0f})
            : ((float[]) {2.0f});
        err = mtxdistmatrix_init_array_real_single(
            &A, num_local_rows, num_local_columns,
            Adata, NULL, NULL, comm, 0, 0, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxdistvector_init_array_real_single(
            &x, num_local_columns, xdata, NULL, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        {
            err = mtxdistvector_init_array_real_single(
                &y, num_local_rows, ydata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistmatrix_sgemv(
                mtx_notrans, 2.0f, &A, &x, 3.0f, &y, &disterr);
            TEST_ASSERT_EQ_MSG(
                MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
                ? mtxdisterror_description(&disterr) : mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.interior.type);
            const struct mtxvector_array * y_ = &y.interior.storage.array;
            TEST_ASSERT_EQ(mtx_field_real, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(2, y_->size);
                TEST_ASSERT_EQ(y_->data.real_single[0], 23.0f);
                TEST_ASSERT_EQ(y_->data.real_single[1], 56.0f);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.real_single[0], 98.0f);
            }
            mtxdistvector_free(&y);
        }
#if 0
        {
            err = mtxdistvector_init_array_real_single(
                &y, num_local_rows, ydata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistmatrix_sgemv(
                mtx_trans, 2.0f, &A, &x, 3.0f, &y, &disterr);
            TEST_ASSERT_EQ_MSG(
                MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
                ? mtxdisterror_description(&disterr) : mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.interior.type);
            const struct mtxvector_array * y_ = &y.interior.storage.array;
            TEST_ASSERT_EQ(mtx_field_real, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(2, y_->size);
                TEST_ASSERT_EQ(y_->data.real_single[0], 39.0f);
                TEST_ASSERT_EQ(y_->data.real_single[1], 48.0f);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.real_single[0], 66.0f);
            }
            mtxdistvector_free(&y);
        }
#endif
        {
            err = mtxdistvector_init_array_real_single(
                &y, num_local_rows, ydata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistmatrix_dgemv(mtx_notrans, 2.0, &A, &x, 3.0, &y, &disterr);
            TEST_ASSERT_EQ_MSG(
                MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
                ? mtxdisterror_description(&disterr) : mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.interior.type);
            const struct mtxvector_array * y_ = &y.interior.storage.array;
            TEST_ASSERT_EQ(mtx_field_real, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(2, y_->size);
                TEST_ASSERT_EQ(y_->data.real_single[0], 23.0f);
                TEST_ASSERT_EQ(y_->data.real_single[1], 56.0f);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.real_single[0], 98.0f);
            }
            mtxdistvector_free(&y);
        }
#if 0
        {
            err = mtxdistvector_init_array_real_single(
                &y, num_local_rows, ydata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistmatrix_dgemv(mtx_trans, 2.0, &A, &x, 3.0, &y, &disterr);
            TEST_ASSERT_EQ_MSG(
                MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
                ? mtxdisterror_description(&disterr) : mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.interior.type);
            const struct mtxvector_array * y_ = &y.interior.storage.array;
            TEST_ASSERT_EQ(mtx_field_real, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(2, y_->size);
                TEST_ASSERT_EQ(y_->data.real_single[0], 39.0f);
                TEST_ASSERT_EQ(y_->data.real_single[1], 48.0f);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.real_single[0], 66.0f);
            }
            mtxdistvector_free(&y);
        }
#endif
        mtxdistvector_free(&x);
        mtxdistmatrix_free(&A);
    }

    {
        struct mtxdistmatrix A;
        struct mtxdistvector x;
        struct mtxdistvector y;
        int num_local_rows = rank == 0 ? 2 : 1;
        int num_local_columns = 3;
        double * Adata = rank == 0
            ? ((double[]) {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f})
            : ((double[]) {7.0f, 8.0f, 9.0f});
        double * xdata = (double[]) {3.0f, 2.0f, 1.0f};
        double * ydata = rank == 0
            ? ((double[]) {1.0f, 0.0f})
            : ((double[]) {2.0f});
        err = mtxdistmatrix_init_array_real_double(
            &A, num_local_rows, num_local_columns,
            Adata, NULL, NULL, comm, 0, 0, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxdistvector_init_array_real_double(
            &x, num_local_columns, xdata, NULL, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        {
            err = mtxdistvector_init_array_real_double(
                &y, num_local_rows, ydata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistmatrix_sgemv(
                mtx_notrans, 2.0f, &A, &x, 3.0f, &y, &disterr);
            TEST_ASSERT_EQ_MSG(
                MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
                ? mtxdisterror_description(&disterr) : mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.interior.type);
            const struct mtxvector_array * y_ = &y.interior.storage.array;
            TEST_ASSERT_EQ(mtx_field_real, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(2, y_->size);
                TEST_ASSERT_EQ(y_->data.real_double[0], 23.0);
                TEST_ASSERT_EQ(y_->data.real_double[1], 56.0);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.real_double[0], 98.0);
            }
            mtxdistvector_free(&y);
        }
#if 0
        {
            err = mtxdistvector_init_array_real_double(
                &y, num_local_rows, ydata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistmatrix_sgemv(
                mtx_trans, 2.0f, &A, &x, 3.0f, &y, &disterr);
            TEST_ASSERT_EQ_MSG(
                MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
                ? mtxdisterror_description(&disterr) : mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.interior.type);
            const struct mtxvector_array * y_ = &y.interior.storage.array;
            TEST_ASSERT_EQ(mtx_field_real, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(2, y_->size);
                TEST_ASSERT_EQ(y_->data.real_double[0], 39.0);
                TEST_ASSERT_EQ(y_->data.real_double[1], 48.0);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.real_double[0], 66.0);
            }
            mtxdistvector_free(&y);
        }
#endif
        {
            err = mtxdistvector_init_array_real_double(
                &y, num_local_rows, ydata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistmatrix_dgemv(
                mtx_notrans, 2.0, &A, &x, 3.0, &y, &disterr);
            TEST_ASSERT_EQ_MSG(
                MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
                ? mtxdisterror_description(&disterr) : mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.interior.type);
            const struct mtxvector_array * y_ = &y.interior.storage.array;
            TEST_ASSERT_EQ(mtx_field_real, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(2, y_->size);
                TEST_ASSERT_EQ(y_->data.real_double[0], 23.0);
                TEST_ASSERT_EQ(y_->data.real_double[1], 56.0);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.real_double[0], 98.0);
            }
            mtxdistvector_free(&y);
        }
#if 0
        {
            err = mtxdistvector_init_array_real_double(
                &y, num_local_rows, ydata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistmatrix_dgemv(mtx_trans, 2.0, &A, &x, 3.0, &y);
            TEST_ASSERT_EQ_MSG(
                MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
                ? mtxdisterror_description(&disterr) : mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.interior.type);
            const struct mtxvector_array * y_ = &y.interior.storage.array;
            TEST_ASSERT_EQ(mtx_field_real, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(2, y_->size);
                TEST_ASSERT_EQ(y_->data.real_double[0], 39.0);
                TEST_ASSERT_EQ(y_->data.real_double[1], 48.0);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.real_double[0], 66.0);
            }
            mtxdistvector_free(&y);
        }
#endif
        mtxdistvector_free(&x);
        mtxdistmatrix_free(&A);
    }

    /*
     * Complex matrices
     */
    {
        struct mtxdistmatrix A;
        struct mtxdistvector x;
        struct mtxdistvector y;
        int num_local_rows = rank == 0 ? 1 : 1;
        int num_local_columns = 2;
        float (* Adata)[2] = rank == 0
            ? ((float[][2]) {{1.0f,2.0f}, {3.0f,4.0f}})
            : ((float[][2]) {{5.0f,6.0f}, {7.0f,8.0f}});
        float (* xdata)[2] = (float[][2]) {{3.0f,1.0f}, {1.0f,2.0f}};
        float (* ydata)[2] = rank == 0
            ? ((float[][2]) {{1.0f,0.0f}})
            : ((float[][2]) {{2.0f,2.0f}});
        err = mtxdistmatrix_init_array_complex_single(
            &A, num_local_rows, num_local_columns,
            Adata, NULL, NULL, comm, 0, 0, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxdistvector_init_array_complex_single(
            &x, num_local_columns, xdata, NULL, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        {
            err = mtxdistvector_init_array_complex_single(
                &y, num_local_rows, ydata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistmatrix_sgemv(
                mtx_notrans, 2.0f, &A, &x, 3.0f, &y, &disterr);
            TEST_ASSERT_EQ_MSG(
                MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
                ? mtxdisterror_description(&disterr) : mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.interior.type);
            const struct mtxvector_array * y_ = &y.interior.storage.array;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.complex_single[0][0], -5.0f);
                TEST_ASSERT_EQ(y_->data.complex_single[0][1], 34.0f);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.complex_single[0][0],  6.0f);
                TEST_ASSERT_EQ(y_->data.complex_single[0][1], 96.0f);
            }
            mtxdistvector_free(&y);
        }
#if 0
        {
            err = mtxdistvector_init_array_complex_single(
                &y, num_local_rows, ydata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistmatrix_sgemv(
                mtx_trans, 2.0f, &A, &x, 3.0f, &y, &disterr);
            TEST_ASSERT_EQ_MSG(
                MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
                ? mtxdisterror_description(&disterr) : mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.interior.type);
            const struct mtxvector_array * y_ = &y.interior.storage.array;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.complex_single[0][0], -9.0f);
                TEST_ASSERT_EQ(y_->data.complex_single[0][1], 46.0f);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.complex_single[0][0], -2.0f);
                TEST_ASSERT_EQ(y_->data.complex_single[0][1], 80.0f);
            }
            mtxdistvector_free(&y);
        }
        {
            err = mtxdistvector_init_array_complex_single(
                &y, num_local_rows, ydata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistmatrix_sgemv(
                mtx_conjtrans, 2.0f, &A, &x, 3.0f, &y, &disterr);
            TEST_ASSERT_EQ_MSG(
                MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
                ? mtxdisterror_description(&disterr) : mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.interior.type);
            const struct mtxvector_array * y_ = &y.interior.storage.array;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.complex_single[0][0], 47.0f);
                TEST_ASSERT_EQ(y_->data.complex_single[0][1], -2.0f);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.complex_single[0][0], 78.0f);
                TEST_ASSERT_EQ(y_->data.complex_single[0][1],  0.0f);
            }
            mtxdistvector_free(&y);
        }
#endif
        {
            err = mtxdistvector_init_array_complex_single(
                &y, num_local_rows, ydata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistmatrix_dgemv(
                mtx_notrans, 2.0, &A, &x, 3.0, &y, &disterr);
            TEST_ASSERT_EQ_MSG(
                MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
                ? mtxdisterror_description(&disterr) : mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.interior.type);
            const struct mtxvector_array * y_ = &y.interior.storage.array;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.complex_single[0][0], -5.0f);
                TEST_ASSERT_EQ(y_->data.complex_single[0][1], 34.0f);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.complex_single[0][0],  6.0f);
                TEST_ASSERT_EQ(y_->data.complex_single[0][1], 96.0f);
            }
            mtxdistvector_free(&y);
        }
#if 0
        {
            err = mtxdistvector_init_array_complex_single(
                &y, num_local_rows, ydata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistmatrix_dgemv(
                mtx_trans, 2.0, &A, &x, 3.0, &y, &disterr);
            TEST_ASSERT_EQ_MSG(
                MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
                ? mtxdisterror_description(&disterr) : mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.interior.type);
            const struct mtxvector_array * y_ = &y.interior.storage.array;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.complex_single[0][0], -9.0f);
                TEST_ASSERT_EQ(y_->data.complex_single[0][1], 46.0f);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.complex_single[0][0], -2.0f);
                TEST_ASSERT_EQ(y_->data.complex_single[0][1], 80.0f);
            }
            mtxdistvector_free(&y);
        }
        {
            err = mtxdistvector_init_array_complex_single(
                &y, num_local_rows, ydata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistmatrix_dgemv(
                mtx_conjtrans, 2.0, &A, &x, 3.0, &y, &disterr);
            TEST_ASSERT_EQ_MSG(
                MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
                ? mtxdisterror_description(&disterr) : mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.interior.type);
            const struct mtxvector_array * y_ = &y.interior.storage.array;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.complex_single[0][0], 47.0f);
                TEST_ASSERT_EQ(y_->data.complex_single[0][1], -2.0f);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.complex_single[0][0], 78.0f);
                TEST_ASSERT_EQ(y_->data.complex_single[0][1],  0.0f);
            }
            mtxdistvector_free(&y);
        }
#endif

        float calpha[2] = {0.0f, 2.0f};
        float cbeta[2]  = {3.0f, 1.0f};
        {
            err = mtxdistvector_init_array_complex_single(
                &y, num_local_rows, ydata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistmatrix_cgemv(
                mtx_notrans, calpha, &A, &x, cbeta, &y, &disterr);
            TEST_ASSERT_EQ_MSG(
                MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
                ? mtxdisterror_description(&disterr) : mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.interior.type);
            const struct mtxvector_array * y_ = &y.interior.storage.array;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.complex_single[0][0], -31.0f);
                TEST_ASSERT_EQ(y_->data.complex_single[0][1],  -7.0f);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.complex_single[0][0], -86.0f);
                TEST_ASSERT_EQ(y_->data.complex_single[0][1],   8.0f);
            }
            mtxdistvector_free(&y);
        }
#if 0
        {
            err = mtxdistvector_init_array_complex_single(
                &y, num_local_rows, ydata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistmatrix_cgemv(
                mtx_trans, calpha, &A, &x, cbeta, &y, &disterr);
            TEST_ASSERT_EQ_MSG(
                MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
                ? mtxdisterror_description(&disterr) : mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.interior.type);
            const struct mtxvector_array * y_ = &y.interior.storage.array;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.complex_single[0][0], -43.0f);
                TEST_ASSERT_EQ(y_->data.complex_single[0][1], -11.0f);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.complex_single[0][0], -70.0f);
                TEST_ASSERT_EQ(y_->data.complex_single[0][1],   0.0f);
            }
            mtxdistvector_free(&y);
        }
        {
            err = mtxdistvector_init_array_complex_single(
                &y, num_local_rows, ydata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistmatrix_cgemv(
                mtx_conjtrans, calpha, &A, &x, cbeta, &y, &disterr);
            TEST_ASSERT_EQ_MSG(
                MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
                ? mtxdisterror_description(&disterr) : mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.interior.type);
            const struct mtxvector_array * y_ = &y.interior.storage.array;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.complex_single[0][0],  5.0f);
                TEST_ASSERT_EQ(y_->data.complex_single[0][1], 45.0f);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.complex_single[0][0], 10.0f);
                TEST_ASSERT_EQ(y_->data.complex_single[0][1], 80.0f);
            }
            mtxdistvector_free(&y);
        }
#endif

        double zalpha[2] = {0.0, 2.0};
        double zbeta[2]  = {3.0, 1.0};
        {
            err = mtxdistvector_init_array_complex_single(
                &y, num_local_rows, ydata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistmatrix_zgemv(
                mtx_notrans, zalpha, &A, &x, zbeta, &y, &disterr);
            TEST_ASSERT_EQ_MSG(
                MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
                ? mtxdisterror_description(&disterr) : mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.interior.type);
            const struct mtxvector_array * y_ = &y.interior.storage.array;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.complex_single[0][0], -31.0f);
                TEST_ASSERT_EQ(y_->data.complex_single[0][1],  -7.0f);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.complex_single[0][0], -86.0f);
                TEST_ASSERT_EQ(y_->data.complex_single[0][1],   8.0f);
            }
            mtxdistvector_free(&y);
        }
#if 0
        {
            err = mtxdistvector_init_array_complex_single(
                &y, num_local_rows, ydata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistmatrix_zgemv(
                mtx_trans, zalpha, &A, &x, zbeta, &y, &disterr);
            TEST_ASSERT_EQ_MSG(
                MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
                ? mtxdisterror_description(&disterr) : mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.interior.type);
            const struct mtxvector_array * y_ = &y.interior.storage.array;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.complex_single[0][0], -43.0f);
                TEST_ASSERT_EQ(y_->data.complex_single[0][1], -11.0f);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.complex_single[0][0], -70.0f);
                TEST_ASSERT_EQ(y_->data.complex_single[0][1],   0.0f);
            }
            mtxdistvector_free(&y);
        }
        {
            err = mtxdistvector_init_array_complex_single(
                &y, num_local_rows, ydata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistmatrix_zgemv(
                mtx_conjtrans, zalpha, &A, &x, zbeta, &y, &disterr);
            TEST_ASSERT_EQ_MSG(
                MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
                ? mtxdisterror_description(&disterr) : mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.interior.type);
            const struct mtxvector_array * y_ = &y.interior.storage.array;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.complex_single[0][0],  5.0f);
                TEST_ASSERT_EQ(y_->data.complex_single[0][1], 45.0f);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.complex_single[0][0], 10.0f);
                TEST_ASSERT_EQ(y_->data.complex_single[0][1], 80.0f);
            }
            mtxdistvector_free(&y);
        }
#endif
        mtxdistvector_free(&x);
        mtxdistmatrix_free(&A);
    }

    {
        struct mtxdistmatrix A;
        struct mtxdistvector x;
        struct mtxdistvector y;
        int num_local_rows = rank == 0 ? 1 : 1;
        int num_local_columns = 2;
        double (* Adata)[2] = rank == 0
            ? ((double[][2]) {{1.0,2.0}, {3.0,4.0}})
            : ((double[][2]) {{5.0,6.0}, {7.0,8.0}});
        double (* xdata)[2] = (double[][2]) {{3.0,1.0}, {1.0,2.0}};
        double (* ydata)[2] = rank == 0
            ? ((double[][2]) {{1.0,0.0}})
            : ((double[][2]) {{2.0,2.0}});
        err = mtxdistmatrix_init_array_complex_double(
            &A, num_local_rows, num_local_columns,
            Adata, NULL, NULL, comm, 0, 0, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxdistvector_init_array_complex_double(
            &x, num_local_columns, xdata, NULL, comm, &disterr);
        {
            err = mtxdistvector_init_array_complex_double(
                &y, num_local_rows, ydata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistmatrix_sgemv(
                mtx_notrans, 2.0f, &A, &x, 3.0f, &y, &disterr);
            TEST_ASSERT_EQ_MSG(
                MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
                ? mtxdisterror_description(&disterr) : mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.interior.type);
            const struct mtxvector_array * y_ = &y.interior.storage.array;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.complex_double[0][0], -5.0);
                TEST_ASSERT_EQ(y_->data.complex_double[0][1], 34.0);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.complex_double[0][0],  6.0);
                TEST_ASSERT_EQ(y_->data.complex_double[0][1], 96.0);
            }
            mtxdistvector_free(&y);
        }
#if 0
        {
            err = mtxdistvector_init_array_complex_double(
                &y, num_local_rows, ydata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistmatrix_sgemv(
                mtx_trans, 2.0f, &A, &x, 3.0f, &y, &disterr);
            TEST_ASSERT_EQ_MSG(
                MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
                ? mtxdisterror_description(&disterr) : mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.interior.type);
            const struct mtxvector_array * y_ = &y.interior.storage.array;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.complex_double[0][0], -9.0);
                TEST_ASSERT_EQ(y_->data.complex_double[0][1], 46.0);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.complex_double[0][0], -2.0);
                TEST_ASSERT_EQ(y_->data.complex_double[0][1], 80.0);
            }
            mtxdistvector_free(&y);
        }
        {
            err = mtxdistvector_init_array_complex_double(
                &y, num_local_rows, ydata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistmatrix_sgemv(
                mtx_conjtrans, 2.0f, &A, &x, 3.0f, &y, &disterr);
            TEST_ASSERT_EQ_MSG(
                MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
                ? mtxdisterror_description(&disterr) : mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.interior.type);
            const struct mtxvector_array * y_ = &y.interior.storage.array;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.complex_double[0][0], 47.0);
                TEST_ASSERT_EQ(y_->data.complex_double[0][1], -2.0);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.complex_double[0][0], 78.0);
                TEST_ASSERT_EQ(y_->data.complex_double[0][1],  0.0);
            }
            mtxdistvector_free(&y);
        }
#endif
        {
            err = mtxdistvector_init_array_complex_double(
                &y, num_local_rows, ydata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistmatrix_dgemv(
                mtx_notrans, 2.0, &A, &x, 3.0, &y, &disterr);
            TEST_ASSERT_EQ_MSG(
                MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
                ? mtxdisterror_description(&disterr) : mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.interior.type);
            const struct mtxvector_array * y_ = &y.interior.storage.array;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.complex_double[0][0], -5.0);
                TEST_ASSERT_EQ(y_->data.complex_double[0][1], 34.0);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.complex_double[0][0],  6.0);
                TEST_ASSERT_EQ(y_->data.complex_double[0][1], 96.0);
            }
            mtxdistvector_free(&y);
        }
#if 0
        {
            err = mtxdistvector_init_array_complex_double(
                &y, num_local_rows, ydata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistmatrix_dgemv(
                mtx_trans, 2.0, &A, &x, 3.0, &y, &disterr);
            TEST_ASSERT_EQ_MSG(
                MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
                ? mtxdisterror_description(&disterr) : mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.interior.type);
            const struct mtxvector_array * y_ = &y.interior.storage.array;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.complex_double[0][0], -9.0);
                TEST_ASSERT_EQ(y_->data.complex_double[0][1], 46.0);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.complex_double[0][0], -2.0);
                TEST_ASSERT_EQ(y_->data.complex_double[0][1], 80.0);
            }
            mtxdistvector_free(&y);
        }
        {
            err = mtxdistvector_init_array_complex_double(
                &y, num_local_rows, ydata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistmatrix_dgemv(
                mtx_conjtrans, 2.0, &A, &x, 3.0, &y, &disterr);
            TEST_ASSERT_EQ_MSG(
                MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
                ? mtxdisterror_description(&disterr) : mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.interior.type);
            const struct mtxvector_array * y_ = &y.interior.storage.array;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.complex_double[0][0], 47.0);
                TEST_ASSERT_EQ(y_->data.complex_double[0][1], -2.0);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.complex_double[0][0], 78.0);
                TEST_ASSERT_EQ(y_->data.complex_double[0][1],  0.0);
            }
            mtxdistvector_free(&y);
        }
#endif

        float calpha[2] = {0.0f, 2.0f};
        float cbeta[2]  = {3.0f, 1.0f};
        {
            err = mtxdistvector_init_array_complex_double(
                &y, num_local_rows, ydata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistmatrix_cgemv(
                mtx_notrans, calpha, &A, &x, cbeta, &y, &disterr);
            TEST_ASSERT_EQ_MSG(
                MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
                ? mtxdisterror_description(&disterr) : mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.interior.type);
            const struct mtxvector_array * y_ = &y.interior.storage.array;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.complex_double[0][0], -31.0);
                TEST_ASSERT_EQ(y_->data.complex_double[0][1],  -7.0);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.complex_double[0][0], -86.0);
                TEST_ASSERT_EQ(y_->data.complex_double[0][1],   8.0);
            }
            mtxdistvector_free(&y);
        }
#if 0
        {
            err = mtxdistvector_init_array_complex_double(
                &y, num_local_rows, ydata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistmatrix_cgemv(
                mtx_trans, calpha, &A, &x, cbeta, &y, &disterr);
            TEST_ASSERT_EQ_MSG(
                MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
                ? mtxdisterror_description(&disterr) : mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.interior.type);
            const struct mtxvector_array * y_ = &y.interior.storage.array;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.complex_double[0][0], -43.0);
                TEST_ASSERT_EQ(y_->data.complex_double[0][1], -11.0);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.complex_double[0][0], -70.0);
                TEST_ASSERT_EQ(y_->data.complex_double[0][1],   0.0);
            }
            mtxdistvector_free(&y);
        }
        {
            err = mtxdistvector_init_array_complex_double(
                &y, num_local_rows, ydata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistmatrix_cgemv(
                mtx_conjtrans, calpha, &A, &x, cbeta, &y, &disterr);
            TEST_ASSERT_EQ_MSG(
                MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
                ? mtxdisterror_description(&disterr) : mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.interior.type);
            const struct mtxvector_array * y_ = &y.interior.storage.array;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.complex_double[0][0],  5.0);
                TEST_ASSERT_EQ(y_->data.complex_double[0][1], 45.0);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.complex_double[0][0], 10.0);
                TEST_ASSERT_EQ(y_->data.complex_double[0][1], 80.0);
            }
            mtxdistvector_free(&y);
        }
#endif

        double zalpha[2] = {0.0, 2.0};
        double zbeta[2]  = {3.0, 1.0};
        {
            err = mtxdistvector_init_array_complex_double(
                &y, num_local_rows, ydata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistmatrix_zgemv(
                mtx_notrans, zalpha, &A, &x, zbeta, &y, &disterr);
            TEST_ASSERT_EQ_MSG(
                MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
                ? mtxdisterror_description(&disterr) : mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.interior.type);
            const struct mtxvector_array * y_ = &y.interior.storage.array;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.complex_double[0][0], -31.0);
                TEST_ASSERT_EQ(y_->data.complex_double[0][1],  -7.0);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.complex_double[0][0], -86.0);
                TEST_ASSERT_EQ(y_->data.complex_double[0][1],   8.0);
            }
            mtxdistvector_free(&y);
        }
#if 0
        {
            err = mtxdistvector_init_array_complex_double(
                &y, num_local_rows, ydata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistmatrix_zgemv(
                mtx_trans, zalpha, &A, &x, zbeta, &y, &disterr);
            TEST_ASSERT_EQ_MSG(
                MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
                ? mtxdisterror_description(&disterr) : mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.interior.type);
            const struct mtxvector_array * y_ = &y.interior.storage.array;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.complex_double[0][0], -43.0);
                TEST_ASSERT_EQ(y_->data.complex_double[0][1], -11.0);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.complex_double[0][0], -70.0);
                TEST_ASSERT_EQ(y_->data.complex_double[0][1],   0.0);
            }
            mtxdistvector_free(&y);
        }
        {
            err = mtxdistvector_init_array_complex_double(
                &y, num_local_rows, ydata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistmatrix_zgemv(
                mtx_conjtrans, zalpha, &A, &x, zbeta, &y, &disterr);
            TEST_ASSERT_EQ_MSG(
                MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
                ? mtxdisterror_description(&disterr) : mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.interior.type);
            const struct mtxvector_array * y_ = &y.interior.storage.array;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.complex_double[0][0],  5.0);
                TEST_ASSERT_EQ(y_->data.complex_double[0][1], 45.0);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.complex_double[0][0], 10.0);
                TEST_ASSERT_EQ(y_->data.complex_double[0][1], 80.0);
            }
            mtxdistvector_free(&y);
        }
#endif
        mtxdistvector_free(&x);
        mtxdistmatrix_free(&A);
    }

    /*
     * Integer matrices
     */

    {
        struct mtxdistmatrix A;
        struct mtxdistvector x;
        struct mtxdistvector y;
        int num_local_rows = rank == 0 ? 2 : 1;
        int num_local_columns = 3;
        int32_t * Adata = rank == 0
            ? ((int32_t[]) {1, 2, 3, 4, 5, 6})
            : ((int32_t[]) {7, 8, 9});
        int32_t * xdata = (int32_t[]) {3, 2, 1};
        int32_t * ydata = rank == 0 ? ((int32_t[]) {1, 0}) : ((int32_t[]) {2});
        err = mtxdistmatrix_init_array_integer_single(
            &A, num_local_rows, num_local_columns,
            Adata, NULL, NULL, comm, 0, 0, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxdistvector_init_array_integer_single(
            &x, num_local_columns, xdata, NULL, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        {
            err = mtxdistvector_init_array_integer_single(
                &y, num_local_rows, ydata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistmatrix_sgemv(
                mtx_notrans, 2.0f, &A, &x, 3.0f, &y, &disterr);
            TEST_ASSERT_EQ_MSG(
                MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
                ? mtxdisterror_description(&disterr) : mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.interior.type);
            const struct mtxvector_array * y_ = &y.interior.storage.array;
            TEST_ASSERT_EQ(mtx_field_integer, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(2, y_->size);
                TEST_ASSERT_EQ(y_->data.integer_single[0], 23);
                TEST_ASSERT_EQ(y_->data.integer_single[1], 56);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.integer_single[0], 98);
            }
            mtxdistvector_free(&y);
        }
#if 0
        {
            err = mtxdistvector_init_array_integer_single(
                &y, num_local_rows, ydata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistmatrix_sgemv(
                mtx_trans, 2.0f, &A, &x, 3.0f, &y, &disterr);
            TEST_ASSERT_EQ_MSG(
                MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
                ? mtxdisterror_description(&disterr) : mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.interior.type);
            const struct mtxvector_array * y_ = &y.interior.storage.array;
            TEST_ASSERT_EQ(mtx_field_integer, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(2, y_->size);
                TEST_ASSERT_EQ(y_->data.integer_single[0], 39);
                TEST_ASSERT_EQ(y_->data.integer_single[1], 48);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.integer_single[0], 66);
            }
            mtxdistvector_free(&y);
        }
#endif
        {
            err = mtxdistvector_init_array_integer_single(
                &y, num_local_rows, ydata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistmatrix_dgemv(
                mtx_notrans, 2.0, &A, &x, 3.0, &y, &disterr);
            TEST_ASSERT_EQ_MSG(
                MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
                ? mtxdisterror_description(&disterr) : mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.interior.type);
            const struct mtxvector_array * y_ = &y.interior.storage.array;
            TEST_ASSERT_EQ(mtx_field_integer, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(2, y_->size);
                TEST_ASSERT_EQ(y_->data.integer_single[0], 23);
                TEST_ASSERT_EQ(y_->data.integer_single[1], 56);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.integer_single[0], 98);
            }
            mtxdistvector_free(&y);
        }
#if 0
        {
            err = mtxdistvector_init_array_integer_single(
                &y, num_local_rows, ydata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistmatrix_dgemv(
                mtx_trans, 2.0, &A, &x, 3.0, &y, &disterr);
            TEST_ASSERT_EQ_MSG(
                MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
                ? mtxdisterror_description(&disterr) : mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.interior.type);
            const struct mtxvector_array * y_ = &y.interior.storage.array;
            TEST_ASSERT_EQ(mtx_field_integer, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(2, y_->size);
                TEST_ASSERT_EQ(y_->data.integer_single[0], 39);
                TEST_ASSERT_EQ(y_->data.integer_single[1], 48);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.integer_single[0], 66);
            }
            mtxdistvector_free(&y);
        }
#endif
        mtxdistvector_free(&x);
        mtxdistmatrix_free(&A);
    }

    {
        struct mtxdistmatrix A;
        struct mtxdistvector x;
        struct mtxdistvector y;
        int num_local_rows = rank == 0 ? 2 : 1;
        int num_local_columns = 3;
        int64_t * Adata = rank == 0
            ? ((int64_t[]) {1, 2, 3, 4, 5, 6})
            : ((int64_t[]) {7, 8, 9});
        int64_t * xdata = (int64_t[]) {3, 2, 1};
        int64_t * ydata = rank == 0 ? ((int64_t[]) {1, 0}) : ((int64_t[]) {2});
        err = mtxdistmatrix_init_array_integer_double(
            &A, num_local_rows, num_local_columns,
            Adata, NULL, NULL, comm, 0, 0, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxdistvector_init_array_integer_double(
            &x, num_local_columns, xdata, NULL, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        {
            err = mtxdistvector_init_array_integer_double(
                &y, num_local_rows, ydata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistmatrix_sgemv(
                mtx_notrans, 2.0f, &A, &x, 3.0f, &y, &disterr);
            TEST_ASSERT_EQ_MSG(
                MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
                ? mtxdisterror_description(&disterr) : mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.interior.type);
            const struct mtxvector_array * y_ = &y.interior.storage.array;
            TEST_ASSERT_EQ(mtx_field_integer, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(2, y_->size);
                TEST_ASSERT_EQ(y_->data.integer_double[0], 23);
                TEST_ASSERT_EQ(y_->data.integer_double[1], 56);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.integer_double[0], 98);
            }
            mtxdistvector_free(&y);
        }
#if 0
        {
            err = mtxdistvector_init_array_integer_double(
                &y, num_local_rows, ydata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistmatrix_sgemv(
                mtx_trans, 2.0f, &A, &x, 3.0f, &y, &disterr);
            TEST_ASSERT_EQ_MSG(
                MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
                ? mtxdisterror_description(&disterr) : mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.interior.type);
            const struct mtxvector_array * y_ = &y.interior.storage.array;
            TEST_ASSERT_EQ(mtx_field_integer, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(2, y_->size);
                TEST_ASSERT_EQ(y_->data.integer_double[0], 39);
                TEST_ASSERT_EQ(y_->data.integer_double[1], 48);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.integer_double[0], 66);
            }
            mtxdistvector_free(&y);
        }
#endif
        {
            err = mtxdistvector_init_array_integer_double(
                &y, num_local_rows, ydata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistmatrix_dgemv(
                mtx_notrans, 2.0, &A, &x, 3.0, &y, &disterr);
            TEST_ASSERT_EQ_MSG(
                MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
                ? mtxdisterror_description(&disterr) : mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.interior.type);
            const struct mtxvector_array * y_ = &y.interior.storage.array;
            TEST_ASSERT_EQ(mtx_field_integer, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(2, y_->size);
                TEST_ASSERT_EQ(y_->data.integer_double[0], 23);
                TEST_ASSERT_EQ(y_->data.integer_double[1], 56);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.integer_double[0], 98);
            }
            mtxdistvector_free(&y);
        }
#if 0
        {
            err = mtxdistvector_init_array_integer_double(
                &y, num_local_rows, ydata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistmatrix_dgemv(
                mtx_trans, 2.0, &A, &x, 3.0, &y, &disterr);
            TEST_ASSERT_EQ_MSG(
                MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
                ? mtxdisterror_description(&disterr) : mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.interior.type);
            const struct mtxvector_array * y_ = &y.interior.storage.array;
            TEST_ASSERT_EQ(mtx_field_integer, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(2, y_->size);
                TEST_ASSERT_EQ(y_->data.integer_double[0], 39);
                TEST_ASSERT_EQ(y_->data.integer_double[1], 48);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.integer_double[0], 66);
            }
            mtxdistvector_free(&y);
        }
#endif
        mtxdistvector_free(&x);
        mtxdistmatrix_free(&A);
    }
    mtxdisterror_free(&disterr);
    return TEST_SUCCESS;
}

/**
 * ‘test_mtxdistmatrixgemv_coordinate()’ tests computing matrix-vector
 * products for matrices in coordinate format.
 */
int test_mtxdistmatrixgemv_coordinate(void)
{
    int err;
    char mpierrstr[MPI_MAX_ERROR_STRING];
    int mpierrstrlen;
    MPI_Comm comm = MPI_COMM_WORLD;
    int root = 0;

    int comm_size;
    err = MPI_Comm_size(comm, &comm_size);
    if (err) {
        MPI_Error_string(err, mpierrstr, &mpierrstrlen);
        fprintf(stderr, "%s: MPI_Comm_size failed with %s\n",
                program_invocation_short_name, mpierrstr);
        MPI_Abort(comm, EXIT_FAILURE);
    }
    int rank;
    err = MPI_Comm_rank(comm, &rank);
    if (err) {
        MPI_Error_string(err, mpierrstr, &mpierrstrlen);
        fprintf(stderr, "%s: MPI_Comm_rank failed with %s\n",
                program_invocation_short_name, mpierrstr);
        MPI_Abort(comm, EXIT_FAILURE);
    }
    if (comm_size != 2)
        TEST_FAIL_MSG("Expected exactly two MPI processes");

    struct mtxdisterror disterr;
    err = mtxdisterror_alloc(&disterr, comm, NULL);
    if (err)
        MPI_Abort(comm, EXIT_FAILURE);

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
        struct mtxdistmatrix A;
        struct mtxdistvector x;
        struct mtxdistvector y;
        int num_local_rows = rank == 0 ? 2 : 1;
        int num_local_columns = 3;
        int num_local_nonzeros = rank == 0 ? 4 : 1;
        int * Arowidx = rank == 0 ? ((int[]) {0, 0, 1, 1}) : ((int[]) {0});
        int * Acolidx = rank == 0 ? ((int[]) {0, 2, 0, 1}) : ((int[]) {2});
        float * Adata = rank == 0 ? ((float[]) {1.0f, 3.0f, 4.0f, 5.0f}) : ((float[]) {9.0f});
        float * xdata = (float[]) {3.0f, 2.0f, 1.0f};
        float * ydata = rank == 0 ? ((float[]) {1.0f, 0.0f}) : ((float[]) {2.0f});
        err = mtxdistmatrix_init_coordinate_real_single(
            &A, num_local_rows, num_local_columns, num_local_nonzeros,
            Arowidx, Acolidx, Adata, NULL, NULL, comm, 0, 0, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxdistvector_init_array_real_single(
            &x, num_local_columns, xdata, NULL, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        {
            err = mtxdistvector_init_array_real_single(
                &y, num_local_rows, ydata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistmatrix_sgemv(mtx_notrans, 2.0f, &A, &x, 1.0f, &y, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.interior.type);
            const struct mtxvector_array * y_ = &y.interior.storage.array;
            TEST_ASSERT_EQ(mtx_field_real, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(2, y_->size);
                TEST_ASSERT_EQ(y_->data.real_single[0], 13.0f);
                TEST_ASSERT_EQ(y_->data.real_single[1], 44.0f);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.real_single[0], 20.0f);
            }
            mtxdistvector_free(&y);
        }
        {
            err = mtxdistvector_init_array_real_single(
                &y, num_local_rows, ydata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistmatrix_sgemv(mtx_notrans, 2.0f, &A, &x, 3.0f, &y, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.interior.type);
            const struct mtxvector_array * y_ = &y.interior.storage.array;
            TEST_ASSERT_EQ(mtx_field_real, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(2, y_->size);
                TEST_ASSERT_EQ(y_->data.real_single[0], 15.0f);
                TEST_ASSERT_EQ(y_->data.real_single[1], 44.0f);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.real_single[0], 24.0f);
            }
            mtxdistvector_free(&y);
        }
#if 0
        {
            err = mtxdistvector_init_array_real_single(
                &y, num_local_rows, ydata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistmatrix_sgemv(mtx_trans, 2.0f, &A, &x, 3.0f, &y, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.interior.type);
            const struct mtxvector_array * y_ = &y.interior.storage.array;
            TEST_ASSERT_EQ(mtx_field_real, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(2, y_->size);
                TEST_ASSERT_EQ(y_->data.real_single[0], 25.0f);
                TEST_ASSERT_EQ(y_->data.real_single[1], 20.0f);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.real_single[0], 42.0f);
            }
            mtxdistvector_free(&y);
        }
#endif
        {
            err = mtxdistvector_init_array_real_single(
                &y, num_local_rows, ydata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistmatrix_dgemv(mtx_notrans, 2.0, &A, &x, 3.0, &y, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.interior.type);
            const struct mtxvector_array * y_ = &y.interior.storage.array;
            TEST_ASSERT_EQ(mtx_field_real, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(2, y_->size);
                TEST_ASSERT_EQ(y_->data.real_single[0], 15.0f);
                TEST_ASSERT_EQ(y_->data.real_single[1], 44.0f);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.real_single[0], 24.0f);
            }
            mtxdistvector_free(&y);
        }
#if 0
        {
            err = mtxdistvector_init_array_real_single(
                &y, num_local_rows, ydata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistmatrix_dgemv(mtx_trans, 2.0, &A, &x, 3.0, &y, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.interior.type);
            const struct mtxvector_array * y_ = &y.interior.storage.array;
            TEST_ASSERT_EQ(mtx_field_real, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(2, y_->size);
                TEST_ASSERT_EQ(y_->data.real_single[0], 25.0f);
                TEST_ASSERT_EQ(y_->data.real_single[1], 20.0f);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.real_single[0], 42.0f);
            }
            mtxdistvector_free(&y);
        }
#endif
        mtxdistvector_free(&x);
        mtxdistmatrix_free(&A);
    }

    {
        struct mtxdistmatrix A;
        struct mtxdistvector x;
        struct mtxdistvector y;
        int num_local_rows = rank == 0 ? 2 : 1;
        int num_local_columns = 3;
        int num_local_nonzeros = rank == 0 ? 4 : 1;
        int * Arowidx = rank == 0 ? ((int[]) {0, 0, 1, 1}) : ((int[]) {0});
        int * Acolidx = rank == 0 ? ((int[]) {0, 2, 0, 1}) : ((int[]) {2});
        double * Adata = rank == 0 ? ((double[]) {1.0, 3.0, 4.0, 5.0}) : ((double[]) {9.0});
        double * xdata = (double[]) {3.0, 2.0, 1.0};
        double * ydata = rank == 0 ? ((double[]) {1.0, 0.0}) : ((double[]) {2.0});
        err = mtxdistmatrix_init_coordinate_real_double(
            &A, num_local_rows, num_local_columns, num_local_nonzeros,
            Arowidx, Acolidx, Adata, NULL, NULL, comm, 0, 0, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxdistvector_init_array_real_double(
            &x, num_local_columns, xdata, NULL, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        {
            err = mtxdistvector_init_array_real_double(
                &y, num_local_rows, ydata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistmatrix_sgemv(mtx_notrans, 2.0f, &A, &x, 1.0f, &y, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.interior.type);
            const struct mtxvector_array * y_ = &y.interior.storage.array;
            TEST_ASSERT_EQ(mtx_field_real, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(2, y_->size);
                TEST_ASSERT_EQ(y_->data.real_double[0], 13.0);
                TEST_ASSERT_EQ(y_->data.real_double[1], 44.0);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.real_double[0], 20.0);
            }
            mtxdistvector_free(&y);
        }
        {
            err = mtxdistvector_init_array_real_double(
                &y, num_local_rows, ydata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistmatrix_sgemv(mtx_notrans, 2.0f, &A, &x, 3.0f, &y, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.interior.type);
            const struct mtxvector_array * y_ = &y.interior.storage.array;
            TEST_ASSERT_EQ(mtx_field_real, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(2, y_->size);
                TEST_ASSERT_EQ(y_->data.real_double[0], 15.0);
                TEST_ASSERT_EQ(y_->data.real_double[1], 44.0);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.real_double[0], 24.0);
            }
            mtxdistvector_free(&y);
        }
#if 0
        {
            err = mtxdistvector_init_array_real_double(
                &y, num_local_rows, ydata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistmatrix_sgemv(mtx_trans, 2.0f, &A, &x, 3.0f, &y, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.interior.type);
            const struct mtxvector_array * y_ = &y.interior.storage.array;
            TEST_ASSERT_EQ(mtx_field_real, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(2, y_->size);
                TEST_ASSERT_EQ(y_->data.real_double[0], 25.0);
                TEST_ASSERT_EQ(y_->data.real_double[1], 20.0);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.real_double[0], 42.0);
            }
            mtxdistvector_free(&y);
        }
#endif
        {
            err = mtxdistvector_init_array_real_double(
                &y, num_local_rows, ydata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistmatrix_dgemv(mtx_notrans, 2.0, &A, &x, 3.0, &y, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.interior.type);
            const struct mtxvector_array * y_ = &y.interior.storage.array;
            TEST_ASSERT_EQ(mtx_field_real, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(2, y_->size);
                TEST_ASSERT_EQ(y_->data.real_double[0], 15.0);
                TEST_ASSERT_EQ(y_->data.real_double[1], 44.0);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.real_double[0], 24.0);
            }
            mtxdistvector_free(&y);
        }
#if 0
        {
            err = mtxdistvector_init_array_real_double(
                &y, num_local_rows, ydata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistmatrix_dgemv(mtx_trans, 2.0, &A, &x, 3.0, &y, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.interior.type);
            const struct mtxvector_array * y_ = &y.interior.storage.array;
            TEST_ASSERT_EQ(mtx_field_real, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(2, y_->size);
                TEST_ASSERT_EQ(y_->data.real_double[0], 25.0);
                TEST_ASSERT_EQ(y_->data.real_double[1], 20.0);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.real_double[0], 42.0);
            }
            mtxdistvector_free(&y);
        }
#endif
        mtxdistvector_free(&x);
        mtxdistmatrix_free(&A);
    }

    /*
     * Complex matrices
     */

    {
        struct mtxdistmatrix A;
        struct mtxdistvector x;
        struct mtxdistvector y;
        int num_local_rows = rank == 0 ? 1 : 1;
        int num_local_columns = 2;
        int num_local_nonzeros = rank == 0 ? 2 : 1;
        int * Arowidx = rank == 0 ? ((int[]) {0, 0}) : ((int[]) {0});
        int * Acolidx = rank == 0 ? ((int[]) {0, 1}) : ((int[]) {2});
        float (* Adata)[2] = rank == 0 ? ((float[][2]) {{1.0f,2.0f}, {3.0f,4.0f}}) : ((float[][2]) {{7.0f,8.0f}});
        float (* xdata)[2] = (float[][2]) {{3.0f,1.0f}, {1.0f,2.0f}};
        float (* ydata)[2] = rank == 0 ? ((float[][2]) {{1.0f, 0.0f}}) : ((float[][2]) {{2.0f,2.0f}});
        err = mtxdistmatrix_init_coordinate_complex_single(
            &A, num_local_rows, num_local_columns, num_local_nonzeros,
            Arowidx, Acolidx, Adata, NULL, NULL, comm, 0, 0, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxdistvector_init_array_complex_single(
            &x, num_local_columns, xdata, NULL, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        {
            err = mtxdistvector_init_array_complex_single(
                &y, num_local_rows, ydata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistmatrix_sgemv(mtx_notrans, 2.0f, &A, &x, 1.0f, &y, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.interior.type);
            const struct mtxvector_array * y_ = &y.interior.storage.array;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.complex_single[0][0], -7.0f);
                TEST_ASSERT_EQ(y_->data.complex_single[0][1], 34.0f);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.complex_single[0][0],-16.0f);
                TEST_ASSERT_EQ(y_->data.complex_single[0][1], 46.0f);
            }
            mtxdistvector_free(&y);
        }
        {
            err = mtxdistvector_init_array_complex_single(
                &y, num_local_rows, ydata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistmatrix_sgemv(mtx_notrans, 2.0f, &A, &x, 3.0f, &y, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.interior.type);
            const struct mtxvector_array * y_ = &y.interior.storage.array;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.complex_single[0][0], -5.0f);
                TEST_ASSERT_EQ(y_->data.complex_single[0][1], 34.0f);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.complex_single[0][0],-12.0f);
                TEST_ASSERT_EQ(y_->data.complex_single[0][1], 50.0f);
            }
            mtxdistvector_free(&y);
        }
#if 0
        {
            err = mtxdistvector_init_array_complex_single(
                &y, num_local_rows, ydata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistmatrix_sgemv(mtx_trans, 2.0f, &A, &x, 3.0f, &y, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.interior.type);
            const struct mtxvector_array * y_ = &y.interior.storage.array;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.complex_single[0][0],  5.0f);
                TEST_ASSERT_EQ(y_->data.complex_single[0][1], 14.0f);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.complex_single[0][0], -2.0f);
                TEST_ASSERT_EQ(y_->data.complex_single[0][1], 80.0f);
            }
            mtxdistvector_free(&y);
        }
#endif
        {
            err = mtxdistvector_init_array_complex_single(
                &y, num_local_rows, ydata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistmatrix_sgemv(mtx_conjtrans, 2.0f, &A, &x, 3.0f, &y, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.interior.type);
            const struct mtxvector_array * y_ = &y.interior.storage.array;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.complex_single[0][0], 13.0f);
                TEST_ASSERT_EQ(y_->data.complex_single[0][1],-10.0f);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.complex_single[0][0], 78.0f);
                TEST_ASSERT_EQ(y_->data.complex_single[0][1],  0.0f);
            }
            mtxdistvector_free(&y);
        }
        {
            err = mtxdistvector_init_array_complex_single(
                &y, num_local_rows, ydata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistmatrix_dgemv(mtx_notrans, 2.0, &A, &x, 3.0, &y, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.interior.type);
            const struct mtxvector_array * y_ = &y.interior.storage.array;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.complex_single[0][0], -5.0f);
                TEST_ASSERT_EQ(y_->data.complex_single[0][1], 34.0f);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.complex_single[0][0],-12.0f);
                TEST_ASSERT_EQ(y_->data.complex_single[0][1], 50.0f);
            }
            mtxdistvector_free(&y);
        }
#if 0
        {
            err = mtxdistvector_init_array_complex_single(
                &y, num_local_rows, ydata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistmatrix_dgemv(mtx_trans, 2.0, &A, &x, 3.0, &y, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.interior.type);
            const struct mtxvector_array * y_ = &y.interior.storage.array;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.complex_single[0][0],  5.0f);
                TEST_ASSERT_EQ(y_->data.complex_single[0][1], 14.0f);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.complex_single[0][0], -2.0f);
                TEST_ASSERT_EQ(y_->data.complex_single[0][1], 80.0f);
            }
            mtxdistvector_free(&y);
        }
#endif
        {
            err = mtxdistvector_init_array_complex_single(
                &y, num_local_rows, ydata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistmatrix_dgemv(mtx_conjtrans, 2.0, &A, &x, 3.0, &y, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.interior.type);
            const struct mtxvector_array * y_ = &y.interior.storage.array;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.complex_single[0][0], 13.0f);
                TEST_ASSERT_EQ(y_->data.complex_single[0][1],-10.0f);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.complex_single[0][0], 78.0f);
                TEST_ASSERT_EQ(y_->data.complex_single[0][1],  0.0f);
            }
            mtxdistvector_free(&y);
        }

        float calpha[2] = {0.0f, 2.0f};
        float cbeta[2]  = {3.0f, 1.0f};
        {
            err = mtxdistvector_init_array_complex_single(
                &y, num_local_rows, ydata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistmatrix_cgemv(mtx_notrans, calpha, &A, &x, cbeta, &y, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.interior.type);
            const struct mtxvector_array * y_ = &y.interior.storage.array;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.complex_single[0][0],-31.0f);
                TEST_ASSERT_EQ(y_->data.complex_single[0][1], -7.0f);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.complex_single[0][0],-40.0f);
                TEST_ASSERT_EQ(y_->data.complex_single[0][1],-10.0f);
            }
            mtxdistvector_free(&y);
        }
#if 0
        {
            err = mtxdistvector_init_array_complex_single(
                &y, num_local_rows, ydata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistmatrix_cgemv(mtx_trans, calpha, &A, &x, cbeta, &y, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.interior.type);
            const struct mtxvector_array * y_ = &y.interior.storage.array;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.complex_single[0][0],-11.0f);
                TEST_ASSERT_EQ(y_->data.complex_single[0][1],  3.0f);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.complex_single[0][0],-70.0f);
                TEST_ASSERT_EQ(y_->data.complex_single[0][1],  0.0f);
            }
            mtxdistvector_free(&y);
        }
        {
            err = mtxdistvector_init_array_complex_single(
                &y, num_local_rows, ydata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistmatrix_cgemv(mtx_conjtrans, calpha, &A, &x, cbeta, &y, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.interior.type);
            const struct mtxvector_array * y_ = &y.interior.storage.array;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.complex_single[0][0], 13.0f);
                TEST_ASSERT_EQ(y_->data.complex_single[0][1], 11.0f);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.complex_single[0][0], 10.0f);
                TEST_ASSERT_EQ(y_->data.complex_single[0][1], 80.0f);
            }
            mtxdistvector_free(&y);
        }
#endif

        double zalpha[2] = {0.0, 2.0};
        double zbeta[2]  = {3.0, 1.0};
        {
            err = mtxdistvector_init_array_complex_single(
                &y, num_local_rows, ydata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistmatrix_zgemv(mtx_notrans, zalpha, &A, &x, zbeta, &y, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.interior.type);
            const struct mtxvector_array * y_ = &y.interior.storage.array;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.complex_single[0][0], -31.0f);
                TEST_ASSERT_EQ(y_->data.complex_single[0][1],  -7.0f);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.complex_single[0][0], -40.0f);
                TEST_ASSERT_EQ(y_->data.complex_single[0][1], -10.0f);
            }
            mtxdistvector_free(&y);
        }
#if 0
        {
            err = mtxdistvector_init_array_complex_single(
                &y, num_local_rows, ydata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistmatrix_zgemv(mtx_trans, zalpha, &A, &x, zbeta, &y, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.interior.type);
            const struct mtxvector_array * y_ = &y.interior.storage.array;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.complex_single[0][0], -11.0f);
                TEST_ASSERT_EQ(y_->data.complex_single[0][1],   3.0f);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.complex_single[0][0], -70.0f);
                TEST_ASSERT_EQ(y_->data.complex_single[0][1],   0.0f);
            }
            mtxdistvector_free(&y);
        }
        {
            err = mtxdistvector_init_array_complex_single(
                &y, num_local_rows, ydata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistmatrix_zgemv(mtx_conjtrans, zalpha, &A, &x, zbeta, &y, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.interior.type);
            const struct mtxvector_array * y_ = &y.interior.storage.array;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.complex_single[0][0], 13.0f);
                TEST_ASSERT_EQ(y_->data.complex_single[0][1], 11.0f);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.complex_single[0][0], 10.0f);
                TEST_ASSERT_EQ(y_->data.complex_single[0][1], 80.0f);
            }
            mtxdistvector_free(&y);
        }
#endif
        mtxdistvector_free(&x);
        mtxdistmatrix_free(&A);
    }

    {
        struct mtxdistmatrix A;
        struct mtxdistvector x;
        struct mtxdistvector y;
        int num_local_rows = rank == 0 ? 1 : 1;
        int num_local_columns = 2;
        int num_local_nonzeros = rank == 0 ? 2 : 1;
        int * Arowidx = rank == 0 ? ((int[]) {0, 0}) : ((int[]) {0});
        int * Acolidx = rank == 0 ? ((int[]) {0, 1}) : ((int[]) {2});
        double (* Adata)[2] = rank == 0 ? ((double[][2]) {{1.0,2.0}, {3.0,4.0}}) : ((double[][2]) {{7.0,8.0}});
        double (* xdata)[2] = (double[][2]) {{3.0,1.0}, {1.0,2.0}};
        double (* ydata)[2] = rank == 0 ? ((double[][2]) {{1.0, 0.0}}) : ((double[][2]) {{2.0,2.0}});
        err = mtxdistmatrix_init_coordinate_complex_double(
            &A, num_local_rows, num_local_columns, num_local_nonzeros,
            Arowidx, Acolidx, Adata, NULL, NULL, comm, 0, 0, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxdistvector_init_array_complex_double(
            &x, num_local_columns, xdata, NULL, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        {
            err = mtxdistvector_init_array_complex_double(
                &y, num_local_rows, ydata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistmatrix_sgemv(mtx_notrans, 2.0f, &A, &x, 1.0f, &y, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.interior.type);
            const struct mtxvector_array * y_ = &y.interior.storage.array;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.complex_double[0][0], -7.0);
                TEST_ASSERT_EQ(y_->data.complex_double[0][1], 34.0);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.complex_double[0][0],-16.0);
                TEST_ASSERT_EQ(y_->data.complex_double[0][1], 46.0);
            }
            mtxdistvector_free(&y);
        }
        {
            err = mtxdistvector_init_array_complex_double(
                &y, num_local_rows, ydata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistmatrix_sgemv(mtx_notrans, 2.0f, &A, &x, 3.0f, &y, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.interior.type);
            const struct mtxvector_array * y_ = &y.interior.storage.array;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.complex_double[0][0],  -5.0);
                TEST_ASSERT_EQ(y_->data.complex_double[0][1],  34.0);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.complex_double[0][0], -12.0);
                TEST_ASSERT_EQ(y_->data.complex_double[0][1],  50.0);
            }
            mtxdistvector_free(&y);
        }
#if 0
        {
            err = mtxdistvector_init_array_complex_double(
                &y, num_local_rows, ydata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistmatrix_sgemv(mtx_trans, 2.0f, &A, &x, 3.0f, &y, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.interior.type);
            const struct mtxvector_array * y_ = &y.interior.storage.array;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.complex_double[0][0],  5.0);
                TEST_ASSERT_EQ(y_->data.complex_double[0][1], 14.0);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.complex_double[0][0], -2.0);
                TEST_ASSERT_EQ(y_->data.complex_double[0][1], 80.0);
            }
            mtxdistvector_free(&y);
        }
        {
            err = mtxdistvector_init_array_complex_double(
                &y, num_local_rows, ydata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistmatrix_sgemv(mtx_conjtrans, 2.0f, &A, &x, 3.0f, &y, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.interior.type);
            const struct mtxvector_array * y_ = &y.interior.storage.array;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.complex_double[0][0], 13.0);
                TEST_ASSERT_EQ(y_->data.complex_double[0][1],-10.0);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.complex_double[0][0], 78.0);
                TEST_ASSERT_EQ(y_->data.complex_double[0][1],  0.0);
            }
            mtxdistvector_free(&y);
        }
#endif
        {
            err = mtxdistvector_init_array_complex_double(
                &y, num_local_rows, ydata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistmatrix_dgemv(mtx_notrans, 2.0, &A, &x, 3.0, &y, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.interior.type);
            const struct mtxvector_array * y_ = &y.interior.storage.array;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.complex_double[0][0], -5.0);
                TEST_ASSERT_EQ(y_->data.complex_double[0][1], 34.0);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.complex_double[0][0],-12.0);
                TEST_ASSERT_EQ(y_->data.complex_double[0][1], 50.0);
            }
            mtxdistvector_free(&y);
        }
#if 0
        {
            err = mtxdistvector_init_array_complex_double(
                &y, num_local_rows, ydata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistmatrix_dgemv(mtx_trans, 2.0, &A, &x, 3.0, &y, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.interior.type);
            const struct mtxvector_array * y_ = &y.interior.storage.array;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.complex_double[0][0],  5.0);
                TEST_ASSERT_EQ(y_->data.complex_double[0][1], 14.0);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.complex_double[0][0], -2.0);
                TEST_ASSERT_EQ(y_->data.complex_double[0][1], 80.0);
            }
            mtxdistvector_free(&y);
        }
        {
            err = mtxdistvector_init_array_complex_double(
                &y, num_local_rows, ydata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistmatrix_dgemv(mtx_conjtrans, 2.0, &A, &x, 3.0, &y, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.interior.type);
            const struct mtxvector_array * y_ = &y.interior.storage.array;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.complex_double[0][0], 13.0);
                TEST_ASSERT_EQ(y_->data.complex_double[0][1],-10.0);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.complex_double[0][0], 78.0);
                TEST_ASSERT_EQ(y_->data.complex_double[0][1],  0.0);
            }
            mtxdistvector_free(&y);
        }
#endif
        mtxdistvector_free(&x);
        mtxdistmatrix_free(&A);
    }

    /*
     * Integer matrices
     */

    {
        struct mtxdistmatrix A;
        struct mtxdistvector x;
        struct mtxdistvector y;
        int num_local_rows = rank == 0 ? 2 : 1;
        int num_local_columns = 3;
        int num_local_nonzeros = rank == 0 ? 4 : 1;
        int * Arowidx = rank == 0 ? ((int[]) {0, 0, 1, 1}) : ((int[]) {0});
        int * Acolidx = rank == 0 ? ((int[]) {0, 2, 0, 1}) : ((int[]) {2});
        int32_t * Adata = rank == 0 ? ((int32_t[]) {1, 3, 4, 5}) : ((int32_t[]) {9});
        int32_t * xdata = (int32_t[]) {3, 2, 1};
        int32_t * ydata = rank == 0 ? ((int32_t[]) {1, 0}) : ((int32_t[]) {2});
        err = mtxdistmatrix_init_coordinate_integer_single(
            &A, num_local_rows, num_local_columns, num_local_nonzeros,
            Arowidx, Acolidx, Adata, NULL, NULL, comm, 0, 0, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxdistvector_init_array_integer_single(
            &x, num_local_columns, xdata, NULL, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        {
            err = mtxdistvector_init_array_integer_single(
                &y, num_local_rows, ydata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistmatrix_sgemv(mtx_notrans, 2.0f, &A, &x, 3.0f, &y, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.interior.type);
            const struct mtxvector_array * y_ = &y.interior.storage.array;
            TEST_ASSERT_EQ(mtx_field_integer, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(2, y_->size);
                TEST_ASSERT_EQ(y_->data.integer_single[0], 15);
                TEST_ASSERT_EQ(y_->data.integer_single[1], 44);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.integer_single[0], 24);
            }
            mtxdistvector_free(&y);
        }
#if 0
        {
            err = mtxdistvector_init_array_integer_single(
                &y, num_local_rows, ydata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistmatrix_sgemv(mtx_trans, 2.0f, &A, &x, 3.0f, &y, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.interior.type);
            const struct mtxvector_array * y_ = &y.interior.storage.array;
            TEST_ASSERT_EQ(mtx_field_integer, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(2, y_->size);
                TEST_ASSERT_EQ(y_->data.integer_single[0], 25);
                TEST_ASSERT_EQ(y_->data.integer_single[1], 20);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.integer_single[0], 42);
            }
            mtxdistvector_free(&y);
        }
#endif
        {
            err = mtxdistvector_init_array_integer_single(
                &y, num_local_rows, ydata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistmatrix_dgemv(mtx_notrans, 2.0, &A, &x, 3.0, &y, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.interior.type);
            const struct mtxvector_array * y_ = &y.interior.storage.array;
            TEST_ASSERT_EQ(mtx_field_integer, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(2, y_->size);
                TEST_ASSERT_EQ(y_->data.integer_single[0], 15);
                TEST_ASSERT_EQ(y_->data.integer_single[1], 44);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.integer_single[0], 24);
            }
            mtxdistvector_free(&y);
        }
#if 0
        {
            err = mtxdistvector_init_array_integer_single(
                &y, num_local_rows, ydata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistmatrix_dgemv(mtx_trans, 2.0, &A, &x, 3.0, &y, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.interior.type);
            const struct mtxvector_array * y_ = &y.interior.storage.array;
            TEST_ASSERT_EQ(mtx_field_integer, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(2, y_->size);
                TEST_ASSERT_EQ(y_->data.integer_single[0], 25);
                TEST_ASSERT_EQ(y_->data.integer_single[1], 20);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.integer_single[0], 42);
            }
            mtxdistvector_free(&y);
        }
#endif
        mtxdistvector_free(&x);
        mtxdistmatrix_free(&A);
    }

    {
        struct mtxdistmatrix A;
        struct mtxdistvector x;
        struct mtxdistvector y;
        int num_local_rows = rank == 0 ? 2 : 1;
        int num_local_columns = 3;
        int num_local_nonzeros = rank == 0 ? 4 : 1;
        int * Arowidx = rank == 0 ? ((int[]) {0, 0, 1, 1}) : ((int[]) {0});
        int * Acolidx = rank == 0 ? ((int[]) {0, 2, 0, 1}) : ((int[]) {2});
        int64_t * Adata = rank == 0 ? ((int64_t[]) {1, 3, 4, 5}) : ((int64_t[]) {9});
        int64_t * xdata = (int64_t[]) {3, 2, 1};
        int64_t * ydata = rank == 0 ? ((int64_t[]) {1, 0}) : ((int64_t[]) {2});
        err = mtxdistmatrix_init_coordinate_integer_double(
            &A, num_local_rows, num_local_columns, num_local_nonzeros,
            Arowidx, Acolidx, Adata, NULL, NULL, comm, 0, 0, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxdistvector_init_array_integer_double(
            &x, num_local_columns, xdata, NULL, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        {
            err = mtxdistvector_init_array_integer_double(
                &y, num_local_rows, ydata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistmatrix_sgemv(mtx_notrans, 2.0f, &A, &x, 3.0f, &y, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.interior.type);
            const struct mtxvector_array * y_ = &y.interior.storage.array;
            TEST_ASSERT_EQ(mtx_field_integer, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(2, y_->size);
                TEST_ASSERT_EQ(y_->data.integer_double[0], 15);
                TEST_ASSERT_EQ(y_->data.integer_double[1], 44);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.integer_double[0], 24);
            }
            mtxdistvector_free(&y);
        }
#if 0
        {
            err = mtxdistvector_init_array_integer_double(
                &y, num_local_rows, ydata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistmatrix_sgemv(mtx_trans, 2.0f, &A, &x, 3.0f, &y, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.interior.type);
            const struct mtxvector_array * y_ = &y.interior.storage.array;
            TEST_ASSERT_EQ(mtx_field_integer, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(2, y_->size);
                TEST_ASSERT_EQ(y_->data.integer_double[0], 25);
                TEST_ASSERT_EQ(y_->data.integer_double[1], 20);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.integer_double[0], 42);
            }
            mtxdistvector_free(&y);
        }
#endif
        {
            err = mtxdistvector_init_array_integer_double(
                &y, num_local_rows, ydata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistmatrix_dgemv(mtx_notrans, 2.0, &A, &x, 3.0, &y, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.interior.type);
            const struct mtxvector_array * y_ = &y.interior.storage.array;
            TEST_ASSERT_EQ(mtx_field_integer, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(2, y_->size);
                TEST_ASSERT_EQ(y_->data.integer_double[0], 15);
                TEST_ASSERT_EQ(y_->data.integer_double[1], 44);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.integer_double[0], 24);
            }
            mtxdistvector_free(&y);
        }
#if 0
        {
            err = mtxdistvector_init_array_integer_double(
                &y, num_local_rows, ydata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistmatrix_dgemv(mtx_trans, 2.0, &A, &x, 3.0, &y, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.interior.type);
            const struct mtxvector_array * y_ = &y.interior.storage.array;
            TEST_ASSERT_EQ(mtx_field_integer, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(2, y_->size);
                TEST_ASSERT_EQ(y_->data.integer_double[0], 25);
                TEST_ASSERT_EQ(y_->data.integer_double[1], 20);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.integer_double[0], 42);
            }
            mtxdistvector_free(&y);
        }
#endif
        mtxdistvector_free(&x);
        mtxdistmatrix_free(&A);
    }

    /*
     * Binary (pattern) matrices
     */

    {
        struct mtxdistmatrix A;
        struct mtxdistvector x;
        struct mtxdistvector y;
        int num_local_rows = rank == 0 ? 2 : 1;
        int num_local_columns = 3;
        int num_local_nonzeros = rank == 0 ? 4 : 1;
        int * Arowidx = rank == 0 ? ((int[]) {0, 0, 1, 1}) : ((int[]) {0});
        int * Acolidx = rank == 0 ? ((int[]) {0, 2, 0, 1}) : ((int[]) {2});
        err = mtxdistmatrix_init_coordinate_pattern(
            &A, num_local_rows, num_local_columns, num_local_nonzeros,
            Arowidx, Acolidx, NULL, NULL, comm, 0, 0, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        {
            float * xdata = (float[]) {3, 2, 1};
            float * ydata = rank == 0 ? ((float[]) {1, 0}) : ((float[]) {2});
            err = mtxdistvector_init_array_real_single(
                &x, num_local_columns, xdata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistvector_init_array_real_single(
                &y, num_local_rows, ydata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistmatrix_sgemv(mtx_notrans, 2.0f, &A, &x, 3.0f, &y, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.interior.type);
            const struct mtxvector_array * y_ = &y.interior.storage.array;
            TEST_ASSERT_EQ(mtx_field_real, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(2, y_->size);
                TEST_ASSERT_EQ(y_->data.real_single[0], 11.0f);
                TEST_ASSERT_EQ(y_->data.real_single[1], 10.0f);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.real_single[0],  8.0f);
            }
            mtxdistvector_free(&y);
            mtxdistvector_free(&x);
        }
        {
            double * xdata = (double[]) {3, 2, 1};
            double * ydata = rank == 0 ? ((double[]) {1, 0}) : ((double[]) {2});
            err = mtxdistvector_init_array_real_double(
                &x, num_local_columns, xdata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistvector_init_array_real_double(
                &y, num_local_rows, ydata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistmatrix_sgemv(mtx_notrans, 2.0f, &A, &x, 3.0f, &y, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.interior.type);
            const struct mtxvector_array * y_ = &y.interior.storage.array;
            TEST_ASSERT_EQ(mtx_field_real, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(2, y_->size);
                TEST_ASSERT_EQ(y_->data.real_double[0], 11.0);
                TEST_ASSERT_EQ(y_->data.real_double[1], 10.0);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.real_double[0],  8.0);
            }
            mtxdistvector_free(&y);
            mtxdistvector_free(&x);
        }
        {
            int32_t * xdata = (int32_t[]) {3, 2, 1};
            int32_t * ydata = rank == 0 ? ((int32_t[]) {1, 0}) : ((int32_t[]) {2});
            err = mtxdistvector_init_array_integer_single(
                &x, num_local_columns, xdata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistvector_init_array_integer_single(
                &y, num_local_rows, ydata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistmatrix_sgemv(mtx_notrans, 2.0f, &A, &x, 3.0f, &y, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.interior.type);
            const struct mtxvector_array * y_ = &y.interior.storage.array;
            TEST_ASSERT_EQ(mtx_field_integer, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(2, y_->size);
                TEST_ASSERT_EQ(y_->data.integer_single[0], 11);
                TEST_ASSERT_EQ(y_->data.integer_single[1], 10);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.integer_single[0],  8);
            }
            mtxdistvector_free(&y);
            mtxdistvector_free(&x);
        }
        {
            int32_t * xdata = (int32_t[]) {3, 2, 1};
            int32_t * ydata = rank == 0 ? ((int32_t[]) {1, 0}) : ((int32_t[]) {2});
            err = mtxdistvector_init_array_integer_single(
                &x, num_local_columns, xdata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistvector_init_array_integer_single(
                &y, num_local_rows, ydata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistmatrix_sgemv(mtx_trans, 2.0f, &A, &x, 3.0f, &y, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.interior.type);
            const struct mtxvector_array * y_ = &y.interior.storage.array;
            TEST_ASSERT_EQ(mtx_field_integer, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(2, y_->size);
                TEST_ASSERT_EQ(y_->data.integer_single[0], 13);
                TEST_ASSERT_EQ(y_->data.integer_single[1],  4);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.integer_single[0], 14);
            }
            mtxdistvector_free(&y);
            mtxdistvector_free(&x);
        }
        {
            int32_t * xdata = (int32_t[]) {3, 2, 1};
            int32_t * ydata = rank == 0 ? ((int32_t[]) {1, 0}) : ((int32_t[]) {2});
            err = mtxdistvector_init_array_integer_single(
                &x, num_local_columns, xdata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistvector_init_array_integer_single(
                &y, num_local_rows, ydata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistmatrix_dgemv(mtx_notrans, 2.0, &A, &x, 3.0, &y, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.interior.type);
            const struct mtxvector_array * y_ = &y.interior.storage.array;
            TEST_ASSERT_EQ(mtx_field_integer, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(2, y_->size);
                TEST_ASSERT_EQ(y_->data.integer_single[0], 11);
                TEST_ASSERT_EQ(y_->data.integer_single[1], 10);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.integer_single[0],  8);
            }
            mtxdistvector_free(&y);
            mtxdistvector_free(&x);
        }
        {
            int32_t * xdata = (int32_t[]) {3, 2, 1};
            int32_t * ydata = rank == 0 ? ((int32_t[]) {1, 0}) : ((int32_t[]) {2});
            err = mtxdistvector_init_array_integer_single(
                &x, num_local_columns, xdata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistvector_init_array_integer_single(
                &y, num_local_rows, ydata, NULL, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxdistmatrix_dgemv(mtx_trans, 2.0, &A, &x, 3.0, &y, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_array, y.interior.type);
            const struct mtxvector_array * y_ = &y.interior.storage.array;
            TEST_ASSERT_EQ(mtx_field_integer, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(2, y_->size);
                TEST_ASSERT_EQ(y_->data.integer_single[0], 13);
                TEST_ASSERT_EQ(y_->data.integer_single[1],  4);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.integer_single[0], 14);
            }
            mtxdistvector_free(&y);
            mtxdistvector_free(&x);
        }
        mtxdistmatrix_free(&A);
    }
    mtxdisterror_free(&disterr);
    return TEST_SUCCESS;
}

/**
 * ‘main()’ entry point and test driver.
 */
int main(int argc, char * argv[])
{
    int mpierr;
    char mpierrstr[MPI_MAX_ERROR_STRING];
    int mpierrstrlen;

    /* 1. Initialise MPI. */
    const MPI_Comm mpi_comm = MPI_COMM_WORLD;
    const int mpi_root = 0;
    mpierr = MPI_Init(&argc, &argv);
    if (mpierr) {
        MPI_Error_string(mpierr, mpierrstr, &mpierrstrlen);
        fprintf(stderr, "%s: MPI_Init failed with %s\n",
                program_invocation_short_name, mpierrstr);
        return EXIT_FAILURE;
    }

    /* 2. Run test suite. */
    TEST_SUITE_BEGIN("Running tests for distributed matrix-vector multiplication\n");
    TEST_RUN(test_mtxdistmatrixgemv_array);
    TEST_RUN(test_mtxdistmatrixgemv_coordinate);
    TEST_SUITE_END();

    /* 3. Clean up and return. */
    MPI_Finalize();
    return (TEST_SUITE_STATUS == TEST_SUCCESS) ?
        EXIT_SUCCESS : EXIT_FAILURE;
}
