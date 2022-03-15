/* This file is part of Libmtx.
 *
 * Copyright (C) 2021 James D. Trotter
 *
 * Libmtx is free software: you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * Libmtx is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Libmtx.  If not, see
 * <https://www.gnu.org/licenses/>.
 *
 * Authors: James D. Trotter <james@simula.no>
 * Last modified: 2021-08-20
 *
 * Unit tests for an iterative linear solver based on the conjugate
 * gradient method.
 */

#include "test.h"

#include <libmtx/error.h>
#include <libmtx/matrix/array.h>
#include <libmtx/matrix/coordinate.h>
#include <libmtx/mtx/blas.h>
#include <libmtx/mtx/cg.h>
#include <libmtx/mtx/mtx.h>
#include <libmtx/vector/array.h>

#include <errno.h>

#include <stdlib.h>

#define N 100

/**
 * `test_mtx_dcg_array_real_double()' tests solving a symmetric,
 * positive definite linear system using the conjugate gradient
 * algorithm, with double precision, real matrix in array format.
 */
int test_mtx_dcg_array_real_double(void)
{
    int err;

    /* Create matrix */
    struct mtx A;
    int num_rows = 2;
    int num_columns = 2;
    double A_data[] = {2.0, -1.0, -1.0, 2.0};
    size_t size = sizeof(A_data) / sizeof(*A_data);
    err = mtx_init_matrix_array_real_double(
        &A, mtx_general_, mtx_nontriangular, mtx_row_major,
        0, NULL, num_rows, num_columns, size, A_data);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

    /* Create right-hand side and solution vectors */
    struct mtx b;
    double bdata[] = {1.0, 0.0};
    err = mtx_init_vector_array_real_double(
        &b, 0, NULL, sizeof(bdata)/sizeof(*bdata), bdata);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

    struct mtx x;
    double xdata[] = {0.0, 0.0};
    err = mtx_init_vector_array_real_double(
        &x, 0, NULL, sizeof(xdata)/sizeof(*xdata), xdata);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

    /* Solve */
    double atol = 1e-15;
    double rtol = 1e-15;
    int max_iterations = 10;
    int num_iterations;
    double b_nrm2;
    double r_nrm2;
    err = mtx_dcg(
        &A, &x, &b,
        atol, rtol,
        max_iterations, &num_iterations,
        &b_nrm2, &r_nrm2, NULL);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

    const struct mtx_vector_array_data * vector_array =
        &x.storage.vector_array;
    TEST_ASSERT_EQ(mtx_real, vector_array->field);
    TEST_ASSERT_EQ(mtx_double, vector_array->precision);
    TEST_ASSERT_EQ(2, vector_array->size);
    const double * mtxdata = vector_array->data.real_double;
    TEST_ASSERT_DOUBLE_NEAR_MSG(
        2.0/3.0, mtxdata[0], 1e-12, "mtxdata[0]=%.15g", mtxdata[0]);
    TEST_ASSERT_DOUBLE_NEAR_MSG(
        1.0/3.0, mtxdata[1], 1e-12, "mtxdata[1]=%.15g", mtxdata[1]);
    mtx_free(&b);
    mtx_free(&x);
    mtx_free(&A);
    return TEST_SUCCESS;
}

/**
 * `test_mtx_dcg_poisson_array_real_double()' tests solving a
 * symmetric, positive definite linear system for the 1D Poisson
 * equation with homogeneous Dirichlet boundary conditions, using the
 * conjugate gradient algorithm, with double precision, real matrix in
 * array format.
 */
int test_mtx_dcg_poisson_array_real_double(void)
{
    int err;

    /* Create matrix */
    struct mtx A;
    int num_rows = N-1;
    int num_columns = N-1;
    double A_data[(N-1)*(N-1)] = {};
    size_t size = sizeof(A_data) / sizeof(*A_data);
    double h = 1.0 / (double) N;

    if (num_rows > 0 && num_columns > 0)
        A_data[0] = 2.0;
    if (num_rows > 0 && num_columns > 1)
        A_data[1] = -1.0;
    for (int i = 1, j = 1; i < num_rows-1 && j < num_columns-1; i++, j++) {
        A_data[i*num_columns+(j-1)] = -1.0;
        A_data[i*num_columns+j] = 2.0;
        A_data[i*num_columns+(j+1)] = -1.0;
    }
    if (num_rows > 1 && num_columns > 1) {
        A_data[(num_rows-1)*num_columns+(num_columns-2)] = -1.0;
        A_data[(num_rows-1)*num_columns+(num_columns-1)] = 2.0;
    }

    err = mtx_init_matrix_array_real_double(
        &A, mtx_general_, mtx_nontriangular, mtx_row_major,
        0, NULL, num_rows, num_columns, size, A_data);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

    /* Create right-hand side and solution vectors */
    struct mtx b;
    double bdata[N-1] = {};
    for (int i = 0; i < num_rows; i++)
        bdata[i] = h*h * sin(2.0*M_PI*(i+1)*h);

    err = mtx_init_vector_array_real_double(
        &b, 0, NULL, sizeof(bdata)/sizeof(*bdata), bdata);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

    struct mtx x;
    double xdata[N-1] = {};
    err = mtx_init_vector_array_real_double(
        &x, 0, NULL, sizeof(xdata)/sizeof(*xdata), xdata);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

    /* Solve */
    double atol = 1e-15;
    double rtol = 1e-15;
    int max_iterations = N;
    int num_iterations;
    double b_nrm2;
    double r_nrm2;
    err = mtx_dcg(
        &A, &x, &b,
        atol, rtol,
        max_iterations, &num_iterations,
        &b_nrm2, &r_nrm2, NULL);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

    const struct mtx_vector_array_data * vector_array =
        &x.storage.vector_array;
    TEST_ASSERT_EQ(mtx_real, vector_array->field);
    TEST_ASSERT_EQ(mtx_double, vector_array->precision);
    TEST_ASSERT_EQ(num_rows, vector_array->size);
    const double * mtxdata = vector_array->data.real_double;

#if 0
    fprintf(stderr, "N=%d, h=%g\n", N, h);
    fprintf(stderr, "% 6s", "A");
    for (int j = 0; j < num_columns; j++)
        fprintf(stderr, " % 4d", j+1);
    fprintf(stderr, " | %8s | %3s | %8s | %8s\n", "b", "x0", "x", "exact");
    for (int i = 0; i < num_rows; i++) {
        fprintf(stderr, "% 6d", i+1);
        for (int j = 0; j < num_columns; j++)
            fprintf(stderr, " %4.1g", A_data[i*num_columns+j]);
        fprintf(stderr, " | %8.3g | %3.0g | %8.3g | %8.3g\n", bdata[i], xdata[i], mtxdata[i], sin(2.0*M_PI*(i+1)*h)/(4.0*M_PI*M_PI));
    }
#endif

    /* Check the error in the max norm. */
    for (int i = 0; i < num_rows; i++) {
        TEST_ASSERT_DOUBLE_NEAR_MSG(
            sin(2.0*M_PI*(i+1)*h)/(4.0*M_PI*M_PI), mtxdata[i], 1e-3,
            "i=%d, sin(2.0*M_PI*i*h)/(4.0*M_PI*M_PI)=%.15g, mtxdata[i]=%.15g",
            i, sin(2.0*M_PI*(i+1)*h)/(4.0*M_PI*M_PI), mtxdata[i]);
    }

    mtx_free(&b);
    mtx_free(&x);
    mtx_free(&A);
    return TEST_SUCCESS;
}

/**
 * `test_mtx_dcg_coordinate_real_double()' tests solving a symmetric,
 * positive definite linear system using the conjugate gradient
 * algorithm, with double precision, real matrix in coordinate format.
 */
int test_mtx_dcg_coordinate_real_double(void)
{
    int err;

    /* Create matrix */
    struct mtx A;
    int num_rows = 2;
    int num_columns = 2;
    struct mtx_matrix_coordinate_real_double A_data[] = {
        {1,1,2.0}, {1,2,-1.0}, {2,1,-1.0}, {2,2,2.0}};
    size_t size = sizeof(A_data) / sizeof(*A_data);
    err = mtx_init_matrix_coordinate_real_double(
        &A, mtx_general_, mtx_nontriangular, mtx_row_major, mtx_unassembled,
        0, NULL, num_rows, num_columns, size, A_data);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

    /* Create right-hand side and solution vectors */
    struct mtx b;
    double bdata[] = {1.0, 0.0};
    err = mtx_init_vector_array_real_double(
        &b, 0, NULL, sizeof(bdata)/sizeof(*bdata), bdata);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

    struct mtx x;
    double xdata[] = {0.0, 0.0};
    err = mtx_init_vector_array_real_double(
        &x, 0, NULL, sizeof(xdata)/sizeof(*xdata), xdata);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

    /* Solve */
    double atol = 1e-15;
    double rtol = 1e-15;
    int max_iterations = 10;
    int num_iterations;
    double b_nrm2;
    double r_nrm2;
    err = mtx_dcg(
        &A, &x, &b,
        atol, rtol,
        max_iterations, &num_iterations,
        &b_nrm2, &r_nrm2, NULL);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

    const struct mtx_vector_array_data * vector_array =
        &x.storage.vector_array;
    TEST_ASSERT_EQ(mtx_real, vector_array->field);
    TEST_ASSERT_EQ(mtx_double, vector_array->precision);
    TEST_ASSERT_EQ(2, vector_array->size);
    const double * mtxdata = vector_array->data.real_double;
    TEST_ASSERT_DOUBLE_NEAR_MSG(
        2.0/3.0, mtxdata[0], 1e-12, "mtxdata[0]=%.15g", mtxdata[0]);
    TEST_ASSERT_DOUBLE_NEAR_MSG(
        1.0/3.0, mtxdata[1], 1e-12, "mtxdata[1]=%.15g", mtxdata[1]);
    mtx_free(&b);
    mtx_free(&x);
    mtx_free(&A);
    return TEST_SUCCESS;
}

/**
 * `test_mtx_dcg_poisson_coordinate_real_double()' tests solving a
 * symmetric, positive definite linear system for the 1D Poisson
 * equation with homogeneous Dirichlet boundary conditions, using the
 * conjugate gradient algorithm, with double precision, real matrix in
 * coordinate format.
 */
int test_mtx_dcg_poisson_coordinate_real_double(void)
{
    int err;

    /* Create matrix */
    struct mtx A;
    int num_rows = N-1;
    int num_columns = N-1;
    struct mtx_matrix_coordinate_real_double A_data[2+3*(N-3)+2] = {};
    size_t size = sizeof(A_data) / sizeof(*A_data);
    double h = 1.0 / (double) N;

    if (num_rows < 2 || num_columns < 2)
        TEST_FAIL_MSG("Expected num_rows >= 2 and num_columns >= 2");

    int k = 0;
    A_data[k].i = 1; A_data[k].j = 1; A_data[k].a = 2.0; k++;
    A_data[k].i = 1; A_data[k].j = 2; A_data[k].a = -1.0; k++;
    for (int i = 1, j = 1; i < num_rows-1 && j < num_columns-1; i++, j++) {
        A_data[k].i = i+1; A_data[k].j = j;   A_data[k].a = -1.0; k++;
        A_data[k].i = i+1; A_data[k].j = j+1; A_data[k].a =  2.0; k++;
        A_data[k].i = i+1; A_data[k].j = j+2; A_data[k].a = -1.0; k++;
    }
    A_data[k].i = num_rows; A_data[k].j = num_rows-1; A_data[k].a = -1.0; k++;
    A_data[k].i = num_rows; A_data[k].j = num_rows;   A_data[k].a =  2.0; k++;

    err = mtx_init_matrix_coordinate_real_double(
        &A, mtx_general_, mtx_nontriangular, mtx_row_major, mtx_assembled,
        0, NULL, num_rows, num_columns, size, A_data);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

    /* Create right-hand side and solution vectors */
    struct mtx b;
    double bdata[N-1] = {};
    for (int i = 0; i < num_rows; i++)
        bdata[i] = h*h * sin(2.0*M_PI*(i+1)*h);

    err = mtx_init_vector_array_real_double(
        &b, 0, NULL, sizeof(bdata)/sizeof(*bdata), bdata);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

    struct mtx x;
    double xdata[N-1] = {};
    err = mtx_init_vector_array_real_double(
        &x, 0, NULL, sizeof(xdata)/sizeof(*xdata), xdata);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

    /* Solve */
    double atol = 1e-15;
    double rtol = 1e-15;
    int max_iterations = N;
    int num_iterations;
    double b_nrm2;
    double r_nrm2;
    err = mtx_dcg(
        &A, &x, &b,
        atol, rtol,
        max_iterations, &num_iterations,
        &b_nrm2, &r_nrm2, NULL);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

    const struct mtx_vector_array_data * vector_array =
        &x.storage.vector_array;
    TEST_ASSERT_EQ(mtx_real, vector_array->field);
    TEST_ASSERT_EQ(mtx_double, vector_array->precision);
    TEST_ASSERT_EQ(num_rows, vector_array->size);
    const double * mtxdata = vector_array->data.real_double;

#if 0
    fprintf(stderr, "N=%d, h=%g\n", N, h);
    fprintf(stderr, "% 6s", "A");
    for (int j = 0; j < num_columns; j++)
        fprintf(stderr, " % 4d", j+1);
    fprintf(stderr, " | %8s | %3s | %8s | %8s\n", "b", "x0", "x", "exact");
    for (int i = 0; i < num_rows; i++) {
        fprintf(stderr, "% 6d", i+1);
        for (int j = 0; j < num_columns; j++)
            fprintf(stderr, " %4.1g", A_data[i*num_columns+j]);
        fprintf(stderr, " | %8.3g | %3.0g | %8.3g | %8.3g\n", bdata[i], xdata[i], mtxdata[i], sin(2.0*M_PI*(i+1)*h)/(4.0*M_PI*M_PI));
    }
#endif

    /* Check the error in the max norm. */
    for (int i = 0; i < num_rows; i++) {
        TEST_ASSERT_DOUBLE_NEAR_MSG(
            sin(2.0*M_PI*(i+1)*h)/(4.0*M_PI*M_PI), mtxdata[i], 1e-3,
            "i=%d, sin(2.0*M_PI*i*h)/(4.0*M_PI*M_PI)=%.15g, mtxdata[i]=%.15g",
            i, sin(2.0*M_PI*(i+1)*h)/(4.0*M_PI*M_PI), mtxdata[i]);
    }

    mtx_free(&b);
    mtx_free(&x);
    mtx_free(&A);
    return TEST_SUCCESS;
}

/**
 * `main()' entry point and test driver.
 */
int main(int argc, char * argv[])
{
    TEST_SUITE_BEGIN("Running tests for conjugate gradient solver\n");
    TEST_RUN(test_mtx_dcg_array_real_double);
    TEST_RUN(test_mtx_dcg_poisson_array_real_double);
    TEST_RUN(test_mtx_dcg_coordinate_real_double);
    TEST_RUN(test_mtx_dcg_poisson_coordinate_real_double);
    TEST_SUITE_END();
    return (TEST_SUITE_STATUS == TEST_SUCCESS) ?
        EXIT_SUCCESS : EXIT_FAILURE;
}
