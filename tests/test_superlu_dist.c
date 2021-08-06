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
 * Last modified: 2021-08-03
 *
 * Unit tests for direct solution of linear systems of equations based
 * on SuperLU_DIST.
 */

#include "test.h"

#include <matrixmarket/blas.h>
#include <matrixmarket/error.h>
#include <matrixmarket/superlu_dist.h>
#include <matrixmarket/matrix_coordinate.h>
#include <matrixmarket/mtx.h>
#include <matrixmarket/vector_array.h>

#include <mpi.h>

#include <errno.h>

#include <stdlib.h>

const char * program_invocation_short_name = "test_superlu_dist";

/**
 * `test_mtx_superlu_dist_solve_global_coordinate_double()` tests
 * solving a linear system `Ax=b' for a matrix in coordinate format
 * with real, double precision floating point values. The matrix and
 * right-hand side is replicated globally on every MPI process.
 */
int test_mtx_superlu_dist_solve_global_coordinate_double(void)
{
    int err;

    /* Create matrix */
    struct mtx A;
    int num_comment_lines = 0;
    const char * comment_lines[] = {};
    int num_rows = 2;
    int num_columns = 2;
    const struct mtx_matrix_coordinate_double Adata[] = {
        {1,1,2.0}, {2,1,3.0}, {1,2,-5.0}, {2,2,1.0}};
    int64_t size = sizeof(Adata) / sizeof(*Adata);
    err = mtx_init_matrix_coordinate_double(
        &A, mtx_general, mtx_nontriangular, mtx_column_major,
        mtx_unordered, mtx_unassembled,
        num_comment_lines, comment_lines,
        num_rows, num_columns, size, Adata);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));

    /* Create vectors */
    struct mtx b;
    double bdata[] = {15.0, 31.0};
    err = mtx_init_vector_array_double(&b, 0, NULL, sizeof(bdata)/sizeof(*bdata), bdata);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
    struct mtx x;
    double xdata[] = {0.0, 0.0};
    err = mtx_init_vector_array_double(&x, 0, NULL, sizeof(xdata)/sizeof(*xdata), xdata);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));

    /* Solve */
    MPI_Comm comm = MPI_COMM_WORLD;
    int num_process_rows = 1;
    int num_process_columns = 1;
    int verbose = 1;
    int mpierrcode;
    char mpierrstr[MPI_MAX_ERROR_STRING];
    err = mtx_superlu_dist_solve_global(
        &A, &b, &x, verbose, stderr, &mpierrcode,
        comm, 0, num_process_rows, num_process_columns,
        mtx_superlu_dist_fact_DOFACT,
        true, false,
        mtx_superlu_dist_colperm_MMD_AT_PLUS_A,
        mtx_superlu_dist_rowperm_LargeDiag_MC64,
        false,
        mtx_superlu_dist_iterrefine_DOUBLE,
        mtx_superlu_dist_trans_NOTRANS,
        false, false, true, 10, false, false);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror_mpi(err, mpierrcode, mpierrstr));
    TEST_ASSERT_EQ(2, x.size);
    TEST_ASSERT_DOUBLE_NEAR_MSG(
        10.0, ((const double *) x.data)[0], 1e-15,
        "x.data[0]=%.17f", ((const double *) x.data)[0]);
    TEST_ASSERT_DOUBLE_NEAR_MSG(
        1.0, ((const double *) x.data)[1], 1e-15,
        "x.data[1]=%.17f", ((const double *) x.data)[1]);
    mtx_free(&x);
    mtx_free(&b);
    mtx_free(&A);
    return TEST_SUCCESS;
}

/**
 * `test_mtx_superlu_dist_solve_distributed_coordinate_double()` tests
 * solving a linear system `Ax=b' for a matrix in coordinate format
 * with real, double precision floating point values. The matrix and
 * right-hand side are distributed across multiple MPI processes.
 */
int test_mtx_superlu_dist_solve_distributed_coordinate_double(void)
{
    int err;
    int mpierr;
    char mpierrstr[MPI_MAX_ERROR_STRING];
    int mpierrstrlen;

    /* Get the size of the MPI communicator. */
    MPI_Comm comm = MPI_COMM_WORLD;
    int comm_size;
    mpierr = MPI_Comm_size(comm, &comm_size);
    if (mpierr) {
        MPI_Error_string(mpierr, mpierrstr, &mpierrstrlen);
        fprintf(stderr, "%s: MPI_Comm_size failed with %s\n",
                program_invocation_short_name, mpierrstr);
        MPI_Abort(comm, EXIT_FAILURE);
    }

    /* Get the MPI rank of the current process. */
    int rank;
    err = MPI_Comm_rank(comm, &rank);
    if (mpierr) {
        MPI_Error_string(err, mpierrstr, &mpierrstrlen);
        fprintf(stderr, "%s: MPI_Comm_rank failed with %s\n",
                program_invocation_short_name, mpierrstr);
        MPI_Abort(comm, EXIT_FAILURE);
    }

    /* Create a local, sparse matrix on each MPI process. */
    int num_rows;
    int num_columns;
    int64_t size;
    struct mtx_matrix_coordinate_double Adata[7];
    if (comm_size == 1) {
        num_rows = 3;
        num_columns = 3;
        size = 7;
        Adata[0].i = 1; Adata[0].j = 1; Adata[0].a = 1.0;
        Adata[1].i = 1; Adata[1].j = 2; Adata[1].a = 2.0;
        Adata[2].i = 2; Adata[2].j = 1; Adata[2].a = 2.0;
        Adata[3].i = 2; Adata[3].j = 2; Adata[3].a = 1.0;
        Adata[4].i = 2; Adata[4].j = 3; Adata[4].a = 3.0;
        Adata[5].i = 3; Adata[5].j = 2; Adata[5].a = 3.0;
        Adata[6].i = 3; Adata[6].j = 3; Adata[6].a = 1.0;
    } else if (comm_size == 2) {
        if (rank == 0) {
            num_rows = 2;
            num_columns = 3;
            size = 5;
            Adata[0].i = 1; Adata[0].j = 1; Adata[0].a = 1.0;
            Adata[1].i = 1; Adata[1].j = 2; Adata[1].a = 2.0;
            Adata[2].i = 2; Adata[2].j = 1; Adata[2].a = 2.0;
            Adata[3].i = 2; Adata[3].j = 2; Adata[3].a = 1.0;
            Adata[4].i = 2; Adata[4].j = 3; Adata[4].a = 3.0;
        } else {
            num_rows = 1;
            num_columns = 3;
            size = 2;
            Adata[0].i = 3; Adata[0].j = 2; Adata[0].a = 3.0;
            Adata[1].i = 3; Adata[1].j = 3; Adata[1].a = 1.0;
        }
    } else {
        TEST_FAIL_MSG("Expected one or two MPI processes");
    }

    int num_comment_lines = 0;
    const char * comment_lines[] = {};
    struct mtx A;
    err = mtx_init_matrix_coordinate_double(
        &A, mtx_general, mtx_nontriangular,
        mtx_row_major, mtx_unordered, mtx_assembled,
        num_comment_lines, comment_lines,
        num_rows, num_columns, size, Adata);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));

    /* Create vectors */
    int64_t bsize;
    int64_t xsize;
    double bdata[3];
    double xdata[3];
    if (comm_size == 1) {
        bsize = 3; bdata[0] = 1.0; bdata[1] = 1.0; bdata[2] = 1.0;
        xsize = 3; xdata[0] = 0.0; xdata[1] = 0.0; xdata[2] = 0.0;
    } else if (comm_size == 2) {
        if (rank == 0) {
            bsize = 2; bdata[0] = 1.0; bdata[1] = 1.0;
            xsize = 2; xdata[0] = 0.0; xdata[1] = 0.0;
        } else {
            bsize = 1; bdata[0] = 1.0;
            xsize = 1; xdata[0] = 0.0;
        }
    } else {
        TEST_FAIL_MSG("Expected one or two MPI processes");
    }

    struct mtx b;
    err = mtx_init_vector_array_double(&b, 0, NULL, bsize, bdata);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
    struct mtx x;
    err = mtx_init_vector_array_double(&x, 0, NULL, xsize, xdata);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));

    /* Solve */
    int num_process_rows = 1;
    int num_process_columns = comm_size;
    int verbose = 1;
    err = mtx_superlu_dist_solve_distributed(
        &A, &b, &x, verbose, stderr, &mpierr,
        comm, 0, num_process_rows, num_process_columns,
        mtx_superlu_dist_fact_DOFACT,
        true, false,
        mtx_superlu_dist_colperm_MMD_AT_PLUS_A,
        mtx_superlu_dist_rowperm_LargeDiag_MC64,
        false,
        mtx_superlu_dist_iterrefine_DOUBLE,
        mtx_superlu_dist_trans_NOTRANS,
        false, false, true, 10, false, false);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror_mpi(err, mpierr, mpierrstr));

    if (comm_size == 1) {
        TEST_ASSERT_EQ(3, x.size);
        TEST_ASSERT_DOUBLE_NEAR_MSG(
            1.0/3.0, ((const double *) x.data)[0], 1e-15,
            "x.data[0]=%.17f", ((const double *) x.data)[0]);
        TEST_ASSERT_DOUBLE_NEAR_MSG(
            1.0/3.0, ((const double *) x.data)[1], 1e-15,
            "x.data[1]=%.17f", ((const double *) x.data)[1]);
        TEST_ASSERT_DOUBLE_NEAR_MSG(
            1.0/3.0, ((const double *) x.data)[2], 1e-15,
            "x.data[2]=%.17f", ((const double *) x.data)[2]);
    } else if (comm_size == 2) {
        if (rank == 0) {
            TEST_ASSERT_EQ(2, x.size);
            TEST_ASSERT_DOUBLE_NEAR_MSG(
                1.0/3.0, ((const double *) x.data)[0], 1e-15,
                "x.data[0]=%.17f", ((const double *) x.data)[0]);
            TEST_ASSERT_DOUBLE_NEAR_MSG(
                1.0/3.0, ((const double *) x.data)[1], 1e-15,
                "x.data[1]=%.17f", ((const double *) x.data)[1]);
        } else {
            TEST_ASSERT_EQ(1, x.size);
            TEST_ASSERT_DOUBLE_NEAR_MSG(
                1.0/3.0, ((const double *) x.data)[0], 1e-15,
                "x.data[0]=%.17f", ((const double *) x.data)[0]);
        }
    } else {
        TEST_FAIL_MSG("Expected one or two MPI processes");
    }

    mtx_free(&x);
    mtx_free(&b);
    mtx_free(&A);
    return TEST_SUCCESS;
}

/**
 * `main()' entry point and test driver.
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
    mpierr = MPI_Barrier(mpi_comm);
    if (mpierr) {
        MPI_Error_string(mpierr, mpierrstr, &mpierrstrlen);
        fprintf(stderr, "%s: MPI_Barrier failed with %s\n",
                program_invocation_short_name, mpierrstr);
        MPI_Abort(mpi_comm, EXIT_FAILURE);
    }

    /* 2. Run test suite. */
    TEST_SUITE_BEGIN("Running tests for direct solver based on SuperLU_DIST\n");
    TEST_RUN(test_mtx_superlu_dist_solve_global_coordinate_double);
    TEST_RUN(test_mtx_superlu_dist_solve_distributed_coordinate_double);
    TEST_SUITE_END();

    /* 3. Clean up and return. */
    MPI_Finalize();
    return (TEST_SUITE_STATUS == TEST_SUCCESS) ?
        EXIT_SUCCESS : EXIT_FAILURE;
}
