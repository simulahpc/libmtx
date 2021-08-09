/* This file is part of libmtx.
 *
 * Copyright (C) 2021 James D. Trotter
 *
 * libmtx is free software: you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * libmtx is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with libmtx.  If not, see
 * <https://www.gnu.org/licenses/>.
 *
 * Authors: James D. Trotter <james@simula.no>
 * Last modified: 2021-08-09
 *
 * Unit tests for BLAS operations with Matrix Market objects.
 */

#include "test.h"

#include <libmtx/error.h>
#include <libmtx/mtx/blas.h>
#include <libmtx/matrix/array/array.h>
#include <libmtx/matrix/coordinate/coordinate.h>
#include <libmtx/mtx.h>
#include <libmtx/vector/array/array.h>

#include <errno.h>

#include <stdlib.h>

/**
 * `test_mtx_sscal_vector_array()` tests scaling a dense vector by
 * a single precision floating-point scalar.
 */
int test_mtx_sscal_vector_array(void)
{
    int err;
    struct mtx x;
    float data[] = {1.0f, 2.0f, 3.0f};
    int size = sizeof(data) / sizeof(*data);
    err = mtx_init_vector_array_real(&x, 0, NULL, size, data);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
    float a = 2.0f;
    err = mtx_sscal(a, &x);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
    TEST_ASSERT_EQ(3, x.size);
    TEST_ASSERT_EQ(2.0f, ((const float *) x.data)[0]);
    TEST_ASSERT_EQ(4.0f, ((const float *) x.data)[1]);
    TEST_ASSERT_EQ(6.0f, ((const float *) x.data)[2]);
    mtx_free(&x);
    return TEST_SUCCESS;
}

/**
 * `test_mtx_dscal_vector_array()` tests scaling a dense vector by
 * a double precision floating-point scalar.
 */
int test_mtx_dscal_vector_array(void)
{
    int err;
    struct mtx x;
    double data[] = {1.0, 2.0, 3.0};
    int size = sizeof(data) / sizeof(*data);
    err = mtx_init_vector_array_double(&x, 0, NULL, size, data);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
    double a = 2.0;
    err = mtx_dscal(a, &x);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
    TEST_ASSERT_EQ(3, x.size);
    TEST_ASSERT_EQ(2.0, ((const double *) x.data)[0]);
    TEST_ASSERT_EQ(4.0, ((const double *) x.data)[1]);
    TEST_ASSERT_EQ(6.0, ((const double *) x.data)[2]);
    mtx_free(&x);
    return TEST_SUCCESS;
}

/**
 * `test_mtx_saxpy_vector_array()` tests adding two dense vectors
 * of single precision floating-point values.
 */
int test_mtx_saxpy_vector_array(void)
{
    int err;
    struct mtx x, y;
    float data[] = {1.0f, 2.0f, 3.0f};
    int size = sizeof(data) / sizeof(*data);
    err = mtx_init_vector_array_real(&x, 0, NULL, size, data);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
    err = mtx_init_vector_array_real(&y, 0, NULL, size, data);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
    float a = 2.0f;
    err = mtx_saxpy(a, &x, &y);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
    TEST_ASSERT_EQ(3, y.size);
    TEST_ASSERT_EQ(3.0f, ((const float *) y.data)[0]);
    TEST_ASSERT_EQ(6.0f, ((const float *) y.data)[1]);
    TEST_ASSERT_EQ(9.0f, ((const float *) y.data)[2]);
    mtx_free(&y);
    mtx_free(&x);
    return TEST_SUCCESS;
}

/**
 * `test_mtx_daxpy_vector_array()` tests adding two dense vectors
 * of double precision floating-point values.
 */
int test_mtx_daxpy_vector_array(void)
{
    int err;
    struct mtx x, y;
    double data[] = {1.0, 2.0, 3.0};
    int size = sizeof(data) / sizeof(*data);
    err = mtx_init_vector_array_double(&x, 0, NULL, size, data);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
    err = mtx_init_vector_array_double(&y, 0, NULL, size, data);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
    double a = 2.0;
    err = mtx_daxpy(a, &x, &y);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
    TEST_ASSERT_EQ(3, y.size);
    TEST_ASSERT_EQ(3.0, ((const double *) y.data)[0]);
    TEST_ASSERT_EQ(6.0, ((const double *) y.data)[1]);
    TEST_ASSERT_EQ(9.0, ((const double *) y.data)[2]);
    mtx_free(&y);
    mtx_free(&x);
    return TEST_SUCCESS;
}

/**
 * `test_mtx_sdot_vector_array()` tests computing the dot product
 * of two dense vectors of single precision floating-point values.
 */
int test_mtx_sdot_vector_array(void)
{
    int err;
    struct mtx x, y;
    float data[] = {1.0f, 2.0f, 3.0f};
    int size = sizeof(data) / sizeof(*data);
    err = mtx_init_vector_array_real(&x, 0, NULL, size, data);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
    err = mtx_init_vector_array_real(&y, 0, NULL, size, data);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
    float dot;
    err = mtx_sdot(&x, &y, &dot);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
    TEST_ASSERT_EQ(14.0f, dot);
    mtx_free(&y);
    mtx_free(&x);
    return TEST_SUCCESS;
}

/**
 * `test_mtx_ddot_vector_array()` tests computing the dot product
 * of two dense vectors of double precision floating-point values.
 */
int test_mtx_ddot_vector_array(void)
{
    int err;
    struct mtx x, y;
    double data[] = {1.0, 2.0, 3.0};
    int size = sizeof(data) / sizeof(*data);
    err = mtx_init_vector_array_double(&x, 0, NULL, size, data);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
    err = mtx_init_vector_array_double(&y, 0, NULL, size, data);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
    double dot;
    err = mtx_ddot(&x, &y, &dot);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
    TEST_ASSERT_EQ(14.0, dot);
    mtx_free(&y);
    mtx_free(&x);
    return TEST_SUCCESS;
}

/**
 * `test_mtx_snrm2_vector_array()` tests computing the Euclidean
 * norm of a dense vector of single precision floating-point values.
 */
int test_mtx_snrm2_vector_array(void)
{
    int err;
    struct mtx x;
    float data[] = {1.0f, 1.0f, 1.0f, 2.0f, 3.0f};
    int size = sizeof(data) / sizeof(*data);
    err = mtx_init_vector_array_real(&x, 0, NULL, size, data);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
    float nrm2;
    err = mtx_snrm2(&x, &nrm2);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
    TEST_ASSERT_EQ(4.0f, nrm2);
    mtx_free(&x);
    return TEST_SUCCESS;
}

/**
 * `test_mtx_dnrm2_vector_array()` tests computing the Euclidean
 * norm of a dense vector of double precision floating-point values.
 */
int test_mtx_dnrm2_vector_array(void)
{
    int err;
    struct mtx x;
    double data[] = {1.0, 1.0, 1.0, 2.0, 3.0};
    int size = sizeof(data) / sizeof(*data);
    err = mtx_init_vector_array_double(&x, 0, NULL, size, data);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
    double nrm2;
    err = mtx_dnrm2(&x, &nrm2);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
    TEST_ASSERT_EQ(4.0, nrm2);
    mtx_free(&x);
    return TEST_SUCCESS;
}

/**
 * `test_mtx_sgemv_array()` tests computing a dense matrix-vector
 * product with single precision floating-point values.
 */
int test_mtx_sgemv_array(void)
{
    int err;

    /* Create matrix */
    struct mtx A;
    int num_rows = 2;
    int num_columns = 2;
    float A_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    err = mtx_init_matrix_array_real(
        &A, mtx_general, mtx_nontriangular, mtx_row_major,
        0, NULL, num_rows, num_columns, A_data);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));

    /* Create vectors */
    struct mtx x;
    float xdata[] = {3.0f, 2.0f};
    err = mtx_init_vector_array_real(&x, 0, NULL, sizeof(xdata)/sizeof(*xdata), xdata);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
    struct mtx y;
    float ydata[] = {1.0f, 0.0f};
    err = mtx_init_vector_array_real(&y, 0, NULL, sizeof(ydata)/sizeof(*ydata), ydata);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));

    /* Multiply */
    float alpha = 2.0f;
    float beta = 3.0f;
    err = mtx_sgemv(alpha, &A, &x, beta, &y);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
    TEST_ASSERT_EQ(2, y.size);
    TEST_ASSERT_EQ(17.0f, ((const float *) y.data)[0]);
    TEST_ASSERT_EQ(34.0f, ((const float *) y.data)[1]);
    mtx_free(&y);
    mtx_free(&x);
    mtx_free(&A);
    return TEST_SUCCESS;
}

/**
 * `test_mtx_dgemv_array()` tests computing a dense matrix-vector
 * product with double precision floating-point values.
 */
int test_mtx_dgemv_array(void)
{
    int err;

    /* Create matrix */
    struct mtx A;
    int num_rows = 2;
    int num_columns = 2;
    double A_data[] = {1.0, 2.0, 3.0, 4.0};
    err = mtx_init_matrix_array_double(
        &A, mtx_general, mtx_nontriangular, mtx_row_major,
        0, NULL, num_rows, num_columns, A_data);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));

    /* Create vectors */
    struct mtx x;
    double xdata[] = {3.0, 2.0};
    err = mtx_init_vector_array_double(&x, 0, NULL, sizeof(xdata)/sizeof(*xdata), xdata);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
    struct mtx y;
    double ydata[] = {1.0, 0.0};
    err = mtx_init_vector_array_double(&y, 0, NULL, sizeof(ydata)/sizeof(*ydata), ydata);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));

    /* Multiply */
    double alpha = 2.0;
    double beta = 3.0;
    err = mtx_dgemv(alpha, &A, &x, beta, &y);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
    TEST_ASSERT_EQ(2, y.size);
    TEST_ASSERT_EQ(17.0, ((const double *) y.data)[0]);
    TEST_ASSERT_EQ(34.0, ((const double *) y.data)[1]);
    mtx_free(&y);
    mtx_free(&x);
    mtx_free(&A);
    return TEST_SUCCESS;
}

/**
 * `test_mtx_sgemv_coordinate()` tests computing a sparse
 * matrix-vector product with single precision floating-point values.
 */
int test_mtx_sgemv_coordinate(void)
{
    int err;

    /* Create matrix */
    struct mtx A;
    int num_comment_lines = 0;
    const char * comment_lines[] = {};
    int num_rows = 2;
    int num_columns = 2;
    int64_t size = 3;
    const struct mtx_matrix_coordinate_real Adata[] = {
        {1,1,1.0f}, {1,2,2.0f}, {2,2,3.0f}};
    err = mtx_init_matrix_coordinate_real(
        &A, mtx_general, mtx_nontriangular, mtx_unsorted,
        mtx_unordered, mtx_unassembled,
        num_comment_lines, comment_lines,
        num_rows, num_columns, size, Adata);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));

    /* Create vectors */
    struct mtx x;
    float xdata[] = {3.0f, 2.0f};
    err = mtx_init_vector_array_real(&x, 0, NULL, sizeof(xdata)/sizeof(*xdata), xdata);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
    struct mtx y;
    float ydata[] = {1.0f, 0.0f};
    err = mtx_init_vector_array_real(&y, 0, NULL, sizeof(ydata)/sizeof(*ydata), ydata);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));

    /* Multiply */
    float alpha = 2.0f;
    float beta = 1.0f;
    err = mtx_sgemv(alpha, &A, &x, beta, &y);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
    TEST_ASSERT_EQ(2, y.size);
    TEST_ASSERT_EQ(15.0f, ((const float *) y.data)[0]);
    TEST_ASSERT_EQ(12.0f, ((const float *) y.data)[1]);
    mtx_free(&y);
    mtx_free(&x);
    mtx_free(&A);
    return TEST_SUCCESS;
}

/**
 * `main()' entry point and test driver.
 */
int main(int argc, char * argv[])
{
    TEST_SUITE_BEGIN("Running tests for Matrix Market BLAS operations\n");
    TEST_RUN(test_mtx_sscal_vector_array);
    TEST_RUN(test_mtx_dscal_vector_array);
    TEST_RUN(test_mtx_saxpy_vector_array);
    TEST_RUN(test_mtx_daxpy_vector_array);
    TEST_RUN(test_mtx_sdot_vector_array);
    TEST_RUN(test_mtx_ddot_vector_array);
    TEST_RUN(test_mtx_snrm2_vector_array);
    TEST_RUN(test_mtx_dnrm2_vector_array);
    TEST_RUN(test_mtx_sgemv_array);
    TEST_RUN(test_mtx_dgemv_array);
    TEST_RUN(test_mtx_sgemv_coordinate);
    TEST_SUITE_END();
    return (TEST_SUITE_STATUS == TEST_SUCCESS) ?
        EXIT_SUCCESS : EXIT_FAILURE;
}
