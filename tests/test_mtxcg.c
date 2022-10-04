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
 * Last modified: 2022-05-22
 *
 * Unit tests for an iterative linear solver based on the conjugate
 * gradient method.
 */

#include "test.h"

#include <libmtx/error.h>
#include <libmtx/linalg/local/matrix.h>
#include <libmtx/linalg/local/vector.h>
#include <libmtx/solver/cg.h>

#include <errno.h>

#include <stdlib.h>

#define N 100

/**
 * ‘test_mtxcg_2x2()’ tests solving a 2-by-2 symmetric, positive
 * definite linear system using the conjugate gradient algorithm.
 */
int test_mtxcg_2x2(void)
{
    int err;

    /* create matrix */
    struct mtxmatrix A;
    int num_rows = 2;
    int num_columns = 2;
    int rowidx[] = {0, 0, 1, 1};
    int colidx[] = {0, 1, 0, 1};
    double Adata[] = {2.0, -1.0, -1.0, 2.0};
    size_t size = sizeof(Adata) / sizeof(*Adata);
    err = mtxmatrix_init_entries_real_double(
        &A, mtxbasecoo, mtx_unsymmetric,
        num_rows, num_columns, size, rowidx, colidx, Adata);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

    /* create right-hand side and solution vectors */
    struct mtxvector b;
    double bdata[] = {1.0, 0.0};
    err = mtxvector_init_real_double(
        &b, mtxbasevector, sizeof(bdata)/sizeof(*bdata), bdata);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

    struct mtxvector x;
    double xdata[] = {0.0, 0.0};
    err = mtxvector_init_real_double(
        &x, mtxbasevector, sizeof(xdata)/sizeof(*xdata), xdata);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

    /* solve */
    struct mtxcg cg;
    err = mtxcg_init(&cg, &A, mtxbasevector);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

    double atol = 1e-15;
    double rtol = 1e-15;
    int max_iterations = 10;
    int num_iterations;
    double b_nrm2;
    double r_nrm2;
    err = mtxcg_solve(
        &cg, &b, &x, atol, rtol, max_iterations, false,
        &num_iterations, &b_nrm2, &r_nrm2, NULL, NULL);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

    const struct mtxbasevector * xbase = &x.storage.base;
    TEST_ASSERT_EQ(mtx_field_real, xbase->field);
    TEST_ASSERT_EQ(mtx_double, xbase->precision);
    TEST_ASSERT_EQ(2, xbase->size);
    const double * mtxdata = xbase->data.real_double;
    TEST_ASSERT_DOUBLE_NEAR_MSG(
        2.0/3.0, mtxdata[0], 1e-12, "mtxdata[0]=%.15g", mtxdata[0]);
    TEST_ASSERT_DOUBLE_NEAR_MSG(
        1.0/3.0, mtxdata[1], 1e-12, "mtxdata[1]=%.15g", mtxdata[1]);
    mtxcg_free(&cg);
    mtxvector_free(&b);
    mtxvector_free(&x);
    mtxmatrix_free(&A);
    return TEST_SUCCESS;
}

/**
 * ‘test_mtxcg_poisson()’ tests solving a symmetric, positive definite
 * linear system for the 1D Poisson equation with homogeneous
 * Dirichlet boundary conditions, using the conjugate gradient
 * algorithm.
 */
int test_mtxcg_poisson(void)
{
    int err;

    /* create matrix */
    struct mtxmatrix A;
    int num_rows = N-1;
    int num_columns = N-1;
    double Adata[2+3*(N-3)+2] = {};
    int rowidx[2+3*(N-3)+2] = {};
    int colidx[2+3*(N-3)+2] = {};

    size_t size = sizeof(Adata) / sizeof(*Adata);
    double h = 1.0 / (double) N;

    if (num_rows < 2 || num_columns < 2)
        TEST_FAIL_MSG("Expected num_rows >= 2 and num_columns >= 2");

    int k = 0;
    rowidx[k] = 0; colidx[k] = 0; Adata[k] = 2.0; k++;
    rowidx[k] = 0; colidx[k] = 1; Adata[k] = -1.0; k++;
    for (int i = 1, j = 1; i < num_rows-1 && j < num_columns-1; i++, j++) {
        rowidx[k] = i; colidx[k] = j-1; Adata[k] = -1.0; k++;
        rowidx[k] = i; colidx[k] = j;   Adata[k] =  2.0; k++;
        rowidx[k] = i; colidx[k] = j+1; Adata[k] = -1.0; k++;
    }
    rowidx[k] = num_rows-1; colidx[k] = num_rows-2; Adata[k] = -1.0; k++;
    rowidx[k] = num_rows-1; colidx[k] = num_rows-1; Adata[k] =  2.0; k++;

    err = mtxmatrix_init_entries_real_double(
        &A, mtxbasecoo, mtx_unsymmetric,
        num_rows, num_columns, size, rowidx, colidx, Adata);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

    /* create right-hand side and solution vectors */
    struct mtxvector b;
    double bdata[N-1] = {};
    for (int i = 0; i < num_rows; i++)
        bdata[i] = h*h * sin(2.0*M_PI*(i+1)*h);

    err = mtxvector_init_real_double(
        &b, mtxbasevector, sizeof(bdata)/sizeof(*bdata), bdata);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

    struct mtxvector x;
    double xdata[N-1] = {};
    err = mtxvector_init_real_double(
        &x, mtxbasevector, sizeof(xdata)/sizeof(*xdata), xdata);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

    /* solve */
    struct mtxcg cg;
    err = mtxcg_init(&cg, &A, mtxbasevector);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

    double atol = 1e-15;
    double rtol = 1e-15;
    int max_iterations = N;
    int num_iterations;
    double b_nrm2;
    double r_nrm2;
    err = mtxcg_solve(
        &cg, &b, &x, atol, rtol, max_iterations, false,
        &num_iterations, &b_nrm2, &r_nrm2, NULL, NULL);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

    const struct mtxbasevector * xbase = &x.storage.base;
    TEST_ASSERT_EQ(mtx_field_real, xbase->field);
    TEST_ASSERT_EQ(mtx_double, xbase->precision);
    TEST_ASSERT_EQ(num_rows, xbase->size);

#if 0
    fprintf(stderr, "N=%d, h=%g\n", N, h);
    fprintf(stderr, "% 6s", "A");
    for (int j = 0; j < num_columns; j++)
        fprintf(stderr, " % 4d", j+1);
    fprintf(stderr, " | %8s | %3s | %8s | %8s\n", "b", "x0", "x", "exact");
    for (int i = 0; i < num_rows; i++) {
        fprintf(stderr, "% 6d", i+1);
        for (int j = 0; j < num_columns; j++)
            fprintf(stderr, " %4.1g", Adata[i*num_columns+j]);
        fprintf(stderr, " | %8.3g | %3.0g | %8.3g | %8.3g\n", bdata[i], xdata[i], xbase->data.real_double[i], sin(2.0*M_PI*(i+1)*h)/(4.0*M_PI*M_PI));
    }
#endif

    /* Check the error in the max norm. */
    for (int i = 0; i < num_rows; i++) {
        TEST_ASSERT_DOUBLE_NEAR_MSG(
            sin(2.0*M_PI*(i+1)*h)/(4.0*M_PI*M_PI), xbase->data.real_double[i], 1e-3,
            "i=%d, sin(2.0*M_PI*i*h)/(4.0*M_PI*M_PI)=%.15g, xbase->data.real_double[i]=%.15g",
            i, sin(2.0*M_PI*(i+1)*h)/(4.0*M_PI*M_PI), xbase->data.real_double[i]);
    }

    mtxcg_free(&cg);
    mtxvector_free(&b);
    mtxvector_free(&x);
    mtxmatrix_free(&A);
    return TEST_SUCCESS;
}

/**
 * ‘main()’ entry point and test driver.
 */
int main(int argc, char * argv[])
{
    TEST_SUITE_BEGIN("Running tests for conjugate gradient solver\n");
    TEST_RUN(test_mtxcg_2x2);
    TEST_RUN(test_mtxcg_poisson);
    TEST_SUITE_END();
    return (TEST_SUITE_STATUS == TEST_SUCCESS) ?
        EXIT_SUCCESS : EXIT_FAILURE;
}
