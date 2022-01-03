/* This file is part of libmtx.
 *
 * Copyright (C) 2022 James D. Trotter
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
 * Last modified: 2022-01-03
 *
 * Unit tests for MPI communication routines for matrices and vectors
 * in Matrix Market format.
 */

#include "test.h"

#include <libmtx/libmtx-config.h>

#include <libmtx/error.h>
#include <libmtx/util/partition.h>
#include <libmtx/mtxfile/mtxfile.h>

#include <mpi.h>

#include <errno.h>

#include <stdlib.h>

const char * program_invocation_short_name = "test_mtxfile_mpi";

/**
 * `test_mtxfile_sendrecv()` tests sending a matrix from one MPI
 * process and receiving it at another.
 */
int test_mtxfile_sendrecv(void)
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
    mpierr = MPI_Comm_rank(comm, &rank);
    if (mpierr) {
        MPI_Error_string(err, mpierrstr, &mpierrstrlen);
        fprintf(stderr, "%s: MPI_Comm_rank failed with %s\n",
                program_invocation_short_name, mpierrstr);
        MPI_Abort(comm, EXIT_FAILURE);
    }

    /* Create a matrix on the root process. */
    int num_rows = 4;
    int num_columns = 4;
    const struct mtxfile_matrix_coordinate_real_single data[] = {
        {1, 1, 1.0f},
        {2, 2, 2.0f},
        {3, 3, 3.0f},
        {4, 4, 4.0f}};
    int64_t num_nonzeros = sizeof(data) / sizeof(*data);
    struct mtxfile srcmtx;
    if (rank == 0) {
        err = mtxfile_init_matrix_coordinate_real_single(
            &srcmtx, mtxfile_general, num_rows, num_columns, num_nonzeros, data);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        err = mtxfile_comments_write(&srcmtx.comments, "% A comment line\n");
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
    }

    /* Send the matrix from the root process to another process. */
    struct mtxmpierror mpierror;
    err = mtxmpierror_alloc(&mpierror, comm, NULL);
    if (err)
        MPI_Abort(comm, EXIT_FAILURE);

    struct mtxfile destmtx;
    if (comm_size > 1) {
        if (rank == 0) {
            err = mtxfile_send(&srcmtx, 1, 0, comm, &mpierror);
            if (err)
                mtxfile_free(&srcmtx);
            TEST_ASSERT_EQ(MTX_SUCCESS, err);
        } else if (rank == 1) {
            err = mtxfile_recv(&destmtx, 0, 0, comm, &mpierror);
            TEST_ASSERT_EQ(MTX_SUCCESS, err);
        }
    } else {
        mtxfile_free(&srcmtx);
        TEST_FAIL_MSG("Expected at least two MPI processes");
    }

    /* Check the received matrix. */
    if (rank == 1) {
        TEST_ASSERT_EQ(mtxfile_matrix, destmtx.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, destmtx.header.format);
        TEST_ASSERT_EQ(mtxfile_real, destmtx.header.field);
        TEST_ASSERT_EQ(mtxfile_general, destmtx.header.symmetry);
        TEST_ASSERT_NEQ(NULL, destmtx.comments.root);
        TEST_ASSERT_NEQ(NULL, destmtx.comments.root->comment_line);
        TEST_ASSERT_STREQ("% A comment line\n", destmtx.comments.root->comment_line);
        TEST_ASSERT_EQ(mtx_single, destmtx.precision);
        TEST_ASSERT_EQ(4, destmtx.size.num_rows);
        TEST_ASSERT_EQ(4, destmtx.size.num_columns);
        TEST_ASSERT_EQ(4, destmtx.size.num_nonzeros);

        const struct mtxfile_matrix_coordinate_real_single * data =
            destmtx.data.matrix_coordinate_real_single;
        TEST_ASSERT_EQ(   1, data[0].i); TEST_ASSERT_EQ(   1, data[0].j);
        TEST_ASSERT_EQ(1.0f, data[0].a);
        TEST_ASSERT_EQ(   2, data[1].i); TEST_ASSERT_EQ(   2, data[1].j);
        TEST_ASSERT_EQ(2.0f, data[1].a);
        TEST_ASSERT_EQ(   3, data[2].i); TEST_ASSERT_EQ(   3, data[2].j);
        TEST_ASSERT_EQ(3.0f, data[2].a);
        TEST_ASSERT_EQ(   4, data[3].i); TEST_ASSERT_EQ(   4, data[3].j);
        TEST_ASSERT_EQ(4.0f, data[3].a);
        mtxfile_free(&destmtx);
    } else if (rank == 0) {
        mtxfile_free(&srcmtx);
    }
    return TEST_SUCCESS;
}

/**
 * `test_mtxfile_bcast()` tests broadcasting a matrix from an MPI root
 * process to all other MPI processes in a communicator.
 */
int test_mtxfile_bcast(void)
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
    mpierr = MPI_Comm_rank(comm, &rank);
    if (mpierr) {
        MPI_Error_string(err, mpierrstr, &mpierrstrlen);
        fprintf(stderr, "%s: MPI_Comm_rank failed with %s\n",
                program_invocation_short_name, mpierrstr);
        MPI_Abort(comm, EXIT_FAILURE);
    }

    /* Create a matrix on the root process. */
    int num_rows = 4;
    int num_columns = 4;
    const struct mtxfile_matrix_coordinate_real_double srcdata[] = {
        {1, 1, 1.0},
        {2, 2, 2.0},
        {3, 3, 3.0},
        {4, 4, 4.0}};
    int64_t num_nonzeros = sizeof(srcdata) / sizeof(*srcdata);
    struct mtxfile mtx;
    if (rank == 0) {
        err = mtxfile_init_matrix_coordinate_real_double(
            &mtx, mtxfile_general, num_rows, num_columns, num_nonzeros, srcdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        err = mtxfile_comments_write(&mtx.comments, "% A comment line\n");
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
    }

    /* Broadcast the matrix from the root process. */
    struct mtxmpierror mpierror;
    err = mtxmpierror_alloc(&mpierror, comm, NULL);
    if (err)
        MPI_Abort(comm, EXIT_FAILURE);
    if (comm_size < 2) {
        mtxfile_free(&mtx);
        TEST_FAIL_MSG("Expected at least two MPI processes");
    }
    err = mtxfile_bcast(&mtx, 0, comm, &mpierror);
    if (err)
        mtxfile_free(&mtx);
    TEST_ASSERT_EQ(MTX_SUCCESS, err);

    /* Check the received matrix. */
    TEST_ASSERT_EQ(mtxfile_matrix, mtx.header.object);
    TEST_ASSERT_EQ(mtxfile_coordinate, mtx.header.format);
    TEST_ASSERT_EQ(mtxfile_real, mtx.header.field);
    TEST_ASSERT_EQ(mtxfile_general, mtx.header.symmetry);
    TEST_ASSERT_NEQ(NULL, mtx.comments.root);
    TEST_ASSERT_NEQ(NULL, mtx.comments.root->comment_line);
    TEST_ASSERT_STREQ("% A comment line\n", mtx.comments.root->comment_line);
    TEST_ASSERT_EQ(mtx_double, mtx.precision);
    TEST_ASSERT_EQ(4, mtx.size.num_rows);
    TEST_ASSERT_EQ(4, mtx.size.num_columns);
    TEST_ASSERT_EQ(4, mtx.size.num_nonzeros);

    const struct mtxfile_matrix_coordinate_real_double * data =
        mtx.data.matrix_coordinate_real_double;
    TEST_ASSERT_EQ(  1, data[0].i); TEST_ASSERT_EQ(   1, data[0].j);
    TEST_ASSERT_EQ(1.0, data[0].a);
    TEST_ASSERT_EQ(  2, data[1].i); TEST_ASSERT_EQ(   2, data[1].j);
    TEST_ASSERT_EQ(2.0, data[1].a);
    TEST_ASSERT_EQ(  3, data[2].i); TEST_ASSERT_EQ(   3, data[2].j);
    TEST_ASSERT_EQ(3.0, data[2].a);
    TEST_ASSERT_EQ(  4, data[3].i); TEST_ASSERT_EQ(   4, data[3].j);
    TEST_ASSERT_EQ(4.0, data[3].a);
    mtxfile_free(&mtx);
    return TEST_SUCCESS;
}

/**
 * `test_mtxfile_gather()' tests gathering Matrix Market files onto an
 * MPI root process from other MPI processes in a communicator.
 */
int test_mtxfile_gather(void)
{
    int err;
    int mpierr;
    char mpierrstr[MPI_MAX_ERROR_STRING];
    int mpierrstrlen;
    int root = 0;

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
    if (comm_size != 2)
        TEST_FAIL_MSG("Expected exactly two MPI processes");

    /* Get the MPI rank of the current process. */
    int rank;
    mpierr = MPI_Comm_rank(comm, &rank);
    if (mpierr) {
        MPI_Error_string(err, mpierrstr, &mpierrstrlen);
        fprintf(stderr, "%s: MPI_Comm_rank failed with %s\n",
                program_invocation_short_name, mpierrstr);
        MPI_Abort(comm, EXIT_FAILURE);
    }

    struct mtxmpierror mpierror;
    err = mtxmpierror_alloc(&mpierror, comm, NULL);
    if (err)
        MPI_Abort(comm, EXIT_FAILURE);

    /*
     * Array formats
     */

    {
        int num_rows = (rank == 0) ? 2 : 1;
        int num_columns = 3;
        const double * srcdata = (rank == 0)
            ? ((const double[6]) {1.0, 2.0, 3.0, 4.0, 5.0, 6.0})
            : ((const double[3]) {7.0, 8.0, 9.0});
        int64_t num_nonzeros = (rank == 0) ? 6 : 3;
        struct mtxfile srcmtx;
        err = mtxfile_init_matrix_array_real_double(
            &srcmtx, mtxfile_general, num_rows, num_columns, srcdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));

        struct mtxfile dstmtxs[2];
        err = mtxfile_gather(&srcmtx, dstmtxs, root, comm, &mpierror);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s",
            err == MTX_ERR_MPI_COLLECTIVE
            ? mtxmpierror_description(&mpierror)
            : mtx_strerror(err));
        if (rank == root) {
            TEST_ASSERT_EQ(mtxfile_matrix, dstmtxs[0].header.object);
            TEST_ASSERT_EQ(mtxfile_array, dstmtxs[0].header.format);
            TEST_ASSERT_EQ(mtxfile_real, dstmtxs[0].header.field);
            TEST_ASSERT_EQ(mtxfile_general, dstmtxs[0].header.symmetry);
            TEST_ASSERT_EQ(mtx_double, dstmtxs[0].precision);
            TEST_ASSERT_EQ(2, dstmtxs[0].size.num_rows);
            TEST_ASSERT_EQ(3, dstmtxs[0].size.num_columns);
            TEST_ASSERT_EQ(-1, dstmtxs[0].size.num_nonzeros);
            const double * data0 = dstmtxs[0].data.array_real_double;
            TEST_ASSERT_EQ(1.0, data0[0]);
            TEST_ASSERT_EQ(2.0, data0[1]);
            TEST_ASSERT_EQ(3.0, data0[2]);
            TEST_ASSERT_EQ(4.0, data0[3]);
            TEST_ASSERT_EQ(5.0, data0[4]);
            TEST_ASSERT_EQ(6.0, data0[5]);
            TEST_ASSERT_EQ(mtxfile_matrix, dstmtxs[1].header.object);
            TEST_ASSERT_EQ(mtxfile_array, dstmtxs[1].header.format);
            TEST_ASSERT_EQ(mtxfile_real, dstmtxs[1].header.field);
            TEST_ASSERT_EQ(mtxfile_general, dstmtxs[1].header.symmetry);
            TEST_ASSERT_EQ(mtx_double, dstmtxs[1].precision);
            TEST_ASSERT_EQ(1, dstmtxs[1].size.num_rows);
            TEST_ASSERT_EQ(3, dstmtxs[1].size.num_columns);
            TEST_ASSERT_EQ(-1, dstmtxs[1].size.num_nonzeros);
            const double * data1 = dstmtxs[1].data.array_real_double;
            TEST_ASSERT_EQ(7.0, data1[0]);
            TEST_ASSERT_EQ(8.0, data1[1]);
            TEST_ASSERT_EQ(9.0, data1[2]);
            mtxfile_free(&dstmtxs[0]);
            mtxfile_free(&dstmtxs[1]);
        }
        mtxfile_free(&srcmtx);
    }

    {
        int num_rows = (rank == 0) ? 2 : 3;
        const double * srcdata = (rank == 0)
            ? ((const double[2]) {1.0, 2.0})
            : ((const double[3]) {3.0, 4.0, 5.0});
        int64_t num_nonzeros = (rank == 0) ? 2 : 3;
        struct mtxfile srcmtx;
        err = mtxfile_init_vector_array_real_double(&srcmtx, num_rows, srcdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));

        struct mtxfile dstmtxs[2];
        err = mtxfile_gather(&srcmtx, dstmtxs, root, comm, &mpierror);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s",
            err == MTX_ERR_MPI_COLLECTIVE
            ? mtxmpierror_description(&mpierror)
            : mtx_strerror(err));
        if (rank == root) {
            TEST_ASSERT_EQ(mtxfile_vector, dstmtxs[0].header.object);
            TEST_ASSERT_EQ(mtxfile_array, dstmtxs[0].header.format);
            TEST_ASSERT_EQ(mtxfile_real, dstmtxs[0].header.field);
            TEST_ASSERT_EQ(mtxfile_general, dstmtxs[0].header.symmetry);
            TEST_ASSERT_EQ(mtx_double, dstmtxs[0].precision);
            TEST_ASSERT_EQ(2, dstmtxs[0].size.num_rows);
            TEST_ASSERT_EQ(-1, dstmtxs[0].size.num_columns);
            TEST_ASSERT_EQ(-1, dstmtxs[0].size.num_nonzeros);
            const double * data0 = dstmtxs[0].data.array_real_double;
            TEST_ASSERT_EQ(1.0, data0[0]);
            TEST_ASSERT_EQ(2.0, data0[1]);
            TEST_ASSERT_EQ(mtxfile_vector, dstmtxs[1].header.object);
            TEST_ASSERT_EQ(mtxfile_array, dstmtxs[1].header.format);
            TEST_ASSERT_EQ(mtxfile_real, dstmtxs[1].header.field);
            TEST_ASSERT_EQ(mtxfile_general, dstmtxs[1].header.symmetry);
            TEST_ASSERT_EQ(mtx_double, dstmtxs[1].precision);
            TEST_ASSERT_EQ(3, dstmtxs[1].size.num_rows);
            TEST_ASSERT_EQ(-1, dstmtxs[1].size.num_columns);
            TEST_ASSERT_EQ(-1, dstmtxs[1].size.num_nonzeros);
            const double * data1 = dstmtxs[1].data.array_real_double;
            TEST_ASSERT_EQ(3.0, data1[0]);
            TEST_ASSERT_EQ(4.0, data1[1]);
            TEST_ASSERT_EQ(5.0, data1[2]);
            mtxfile_free(&dstmtxs[0]);
            mtxfile_free(&dstmtxs[1]);
        }
        mtxfile_free(&srcmtx);
    }

    /*
     * Matrix coordinate formats
     */

    {
        int num_rows = 4;
        int num_columns = 4;
        const struct mtxfile_matrix_coordinate_real_double * srcdata = (rank == 0)
            ? ((const struct mtxfile_matrix_coordinate_real_double[2])
                {{1, 1, 1.0}, {2, 2, 2.0}})
            : ((const struct mtxfile_matrix_coordinate_real_double[2])
                {{3, 3, 3.0}, {4, 4, 4.0}});
        int64_t num_nonzeros = (rank == 0) ? 2 : 2;
        struct mtxfile srcmtx;
        err = mtxfile_init_matrix_coordinate_real_double(
            &srcmtx, mtxfile_general, num_rows, num_columns, num_nonzeros, srcdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));

        struct mtxfile dstmtxs[2];
        err = mtxfile_gather(&srcmtx, dstmtxs, root, comm, &mpierror);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s",
            err == MTX_ERR_MPI_COLLECTIVE
            ? mtxmpierror_description(&mpierror)
            : mtx_strerror(err));
        if (rank == root) {
            TEST_ASSERT_EQ(mtxfile_matrix, dstmtxs[0].header.object);
            TEST_ASSERT_EQ(mtxfile_coordinate, dstmtxs[0].header.format);
            TEST_ASSERT_EQ(mtxfile_real, dstmtxs[0].header.field);
            TEST_ASSERT_EQ(mtxfile_general, dstmtxs[0].header.symmetry);
            TEST_ASSERT_EQ(mtx_double, dstmtxs[0].precision);
            TEST_ASSERT_EQ(4, dstmtxs[0].size.num_rows);
            TEST_ASSERT_EQ(4, dstmtxs[0].size.num_columns);
            TEST_ASSERT_EQ(2, dstmtxs[0].size.num_nonzeros);
            const struct mtxfile_matrix_coordinate_real_double * data0 =
                dstmtxs[0].data.matrix_coordinate_real_double;
            TEST_ASSERT_EQ(  1, data0[0].i); TEST_ASSERT_EQ(   1, data0[0].j);
            TEST_ASSERT_EQ(1.0, data0[0].a);
            TEST_ASSERT_EQ(  2, data0[1].i); TEST_ASSERT_EQ(   2, data0[1].j);
            TEST_ASSERT_EQ(2.0, data0[1].a);
            TEST_ASSERT_EQ(mtxfile_matrix, dstmtxs[1].header.object);
            TEST_ASSERT_EQ(mtxfile_coordinate, dstmtxs[1].header.format);
            TEST_ASSERT_EQ(mtxfile_real, dstmtxs[1].header.field);
            TEST_ASSERT_EQ(mtxfile_general, dstmtxs[1].header.symmetry);
            TEST_ASSERT_EQ(mtx_double, dstmtxs[1].precision);
            TEST_ASSERT_EQ(4, dstmtxs[1].size.num_rows);
            TEST_ASSERT_EQ(4, dstmtxs[1].size.num_columns);
            TEST_ASSERT_EQ(2, dstmtxs[1].size.num_nonzeros);
            const struct mtxfile_matrix_coordinate_real_double * data1 =
                dstmtxs[1].data.matrix_coordinate_real_double;
            TEST_ASSERT_EQ(  3, data1[0].i); TEST_ASSERT_EQ(   3, data1[0].j);
            TEST_ASSERT_EQ(3.0, data1[0].a);
            TEST_ASSERT_EQ(  4, data1[1].i); TEST_ASSERT_EQ(   4, data1[1].j);
            TEST_ASSERT_EQ(4.0, data1[1].a);
            mtxfile_free(&dstmtxs[0]);
            mtxfile_free(&dstmtxs[1]);
        }
        mtxfile_free(&srcmtx);
    }
    return TEST_SUCCESS;
}

/**
 * `test_mtxfile_allgather()' tests gathering Matrix Market files onto
 * every MPI process from other MPI processes in a communicator.
 */
int test_mtxfile_allgather(void)
{
    int err;
    int mpierr;
    char mpierrstr[MPI_MAX_ERROR_STRING];
    int mpierrstrlen;
    int root = 0;

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
    if (comm_size != 2)
        TEST_FAIL_MSG("Expected exactly two MPI processes");

    /* Get the MPI rank of the current process. */
    int rank;
    mpierr = MPI_Comm_rank(comm, &rank);
    if (mpierr) {
        MPI_Error_string(err, mpierrstr, &mpierrstrlen);
        fprintf(stderr, "%s: MPI_Comm_rank failed with %s\n",
                program_invocation_short_name, mpierrstr);
        MPI_Abort(comm, EXIT_FAILURE);
    }

    struct mtxmpierror mpierror;
    err = mtxmpierror_alloc(&mpierror, comm, NULL);
    if (err)
        MPI_Abort(comm, EXIT_FAILURE);

    /*
     * Array formats
     */

    {
        int num_rows = (rank == 0) ? 2 : 1;
        int num_columns = 3;
        const double * srcdata = (rank == 0)
            ? ((const double[6]) {1.0, 2.0, 3.0, 4.0, 5.0, 6.0})
            : ((const double[3]) {7.0, 8.0, 9.0});
        int64_t num_nonzeros = (rank == 0) ? 6 : 3;
        struct mtxfile srcmtx;
        err = mtxfile_init_matrix_array_real_double(
            &srcmtx, mtxfile_general, num_rows, num_columns, srcdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));

        struct mtxfile dstmtxs[2];
        err = mtxfile_allgather(&srcmtx, dstmtxs, comm, &mpierror);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s",
            err == MTX_ERR_MPI_COLLECTIVE
            ? mtxmpierror_description(&mpierror)
            : mtx_strerror(err));
        TEST_ASSERT_EQ(mtxfile_matrix, dstmtxs[0].header.object);
        TEST_ASSERT_EQ(mtxfile_array, dstmtxs[0].header.format);
        TEST_ASSERT_EQ(mtxfile_real, dstmtxs[0].header.field);
        TEST_ASSERT_EQ(mtxfile_general, dstmtxs[0].header.symmetry);
        TEST_ASSERT_EQ(mtx_double, dstmtxs[0].precision);
        TEST_ASSERT_EQ(2, dstmtxs[0].size.num_rows);
        TEST_ASSERT_EQ(3, dstmtxs[0].size.num_columns);
        TEST_ASSERT_EQ(-1, dstmtxs[0].size.num_nonzeros);
        const double * data0 = dstmtxs[0].data.array_real_double;
        TEST_ASSERT_EQ(1.0, data0[0]);
        TEST_ASSERT_EQ(2.0, data0[1]);
        TEST_ASSERT_EQ(3.0, data0[2]);
        TEST_ASSERT_EQ(4.0, data0[3]);
        TEST_ASSERT_EQ(5.0, data0[4]);
        TEST_ASSERT_EQ(6.0, data0[5]);
        TEST_ASSERT_EQ(mtxfile_matrix, dstmtxs[1].header.object);
        TEST_ASSERT_EQ(mtxfile_array, dstmtxs[1].header.format);
        TEST_ASSERT_EQ(mtxfile_real, dstmtxs[1].header.field);
        TEST_ASSERT_EQ(mtxfile_general, dstmtxs[1].header.symmetry);
        TEST_ASSERT_EQ(mtx_double, dstmtxs[1].precision);
        TEST_ASSERT_EQ(1, dstmtxs[1].size.num_rows);
        TEST_ASSERT_EQ(3, dstmtxs[1].size.num_columns);
        TEST_ASSERT_EQ(-1, dstmtxs[1].size.num_nonzeros);
        const double * data1 = dstmtxs[1].data.array_real_double;
        TEST_ASSERT_EQ(7.0, data1[0]);
        TEST_ASSERT_EQ(8.0, data1[1]);
        TEST_ASSERT_EQ(9.0, data1[2]);
        mtxfile_free(&dstmtxs[0]);
        mtxfile_free(&dstmtxs[1]);
        mtxfile_free(&srcmtx);
    }

    {
        int num_rows = (rank == 0) ? 2 : 3;
        const double * srcdata = (rank == 0)
            ? ((const double[2]) {1.0, 2.0})
            : ((const double[3]) {3.0, 4.0, 5.0});
        int64_t num_nonzeros = (rank == 0) ? 2 : 3;
        struct mtxfile srcmtx;
        err = mtxfile_init_vector_array_real_double(&srcmtx, num_rows, srcdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));

        struct mtxfile dstmtxs[2];
        err = mtxfile_allgather(&srcmtx, dstmtxs, comm, &mpierror);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s",
            err == MTX_ERR_MPI_COLLECTIVE
            ? mtxmpierror_description(&mpierror)
            : mtx_strerror(err));
        TEST_ASSERT_EQ(mtxfile_vector, dstmtxs[0].header.object);
        TEST_ASSERT_EQ(mtxfile_array, dstmtxs[0].header.format);
        TEST_ASSERT_EQ(mtxfile_real, dstmtxs[0].header.field);
        TEST_ASSERT_EQ(mtxfile_general, dstmtxs[0].header.symmetry);
        TEST_ASSERT_EQ(mtx_double, dstmtxs[0].precision);
        TEST_ASSERT_EQ(2, dstmtxs[0].size.num_rows);
        TEST_ASSERT_EQ(-1, dstmtxs[0].size.num_columns);
        TEST_ASSERT_EQ(-1, dstmtxs[0].size.num_nonzeros);
        const double * data0 = dstmtxs[0].data.array_real_double;
        TEST_ASSERT_EQ(1.0, data0[0]);
        TEST_ASSERT_EQ(2.0, data0[1]);
        TEST_ASSERT_EQ(mtxfile_vector, dstmtxs[1].header.object);
        TEST_ASSERT_EQ(mtxfile_array, dstmtxs[1].header.format);
        TEST_ASSERT_EQ(mtxfile_real, dstmtxs[1].header.field);
        TEST_ASSERT_EQ(mtxfile_general, dstmtxs[1].header.symmetry);
        TEST_ASSERT_EQ(mtx_double, dstmtxs[1].precision);
        TEST_ASSERT_EQ(3, dstmtxs[1].size.num_rows);
        TEST_ASSERT_EQ(-1, dstmtxs[1].size.num_columns);
        TEST_ASSERT_EQ(-1, dstmtxs[1].size.num_nonzeros);
        const double * data1 = dstmtxs[1].data.array_real_double;
        TEST_ASSERT_EQ(3.0, data1[0]);
        TEST_ASSERT_EQ(4.0, data1[1]);
        TEST_ASSERT_EQ(5.0, data1[2]);
        mtxfile_free(&dstmtxs[0]);
        mtxfile_free(&dstmtxs[1]);
        mtxfile_free(&srcmtx);
    }

    /*
     * Matrix coordinate formats
     */

    {
        int num_rows = 4;
        int num_columns = 4;
        const struct mtxfile_matrix_coordinate_real_double * srcdata = (rank == 0)
            ? ((const struct mtxfile_matrix_coordinate_real_double[2])
                {{1, 1, 1.0}, {2, 2, 2.0}})
            : ((const struct mtxfile_matrix_coordinate_real_double[2])
                {{3, 3, 3.0}, {4, 4, 4.0}});
        int64_t num_nonzeros = (rank == 0) ? 2 : 2;
        struct mtxfile srcmtx;
        err = mtxfile_init_matrix_coordinate_real_double(
            &srcmtx, mtxfile_general, num_rows, num_columns, num_nonzeros, srcdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));

        struct mtxfile dstmtxs[2];
        err = mtxfile_allgather(&srcmtx, dstmtxs, comm, &mpierror);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s",
            err == MTX_ERR_MPI_COLLECTIVE
            ? mtxmpierror_description(&mpierror)
            : mtx_strerror(err));
        TEST_ASSERT_EQ(mtxfile_matrix, dstmtxs[0].header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, dstmtxs[0].header.format);
        TEST_ASSERT_EQ(mtxfile_real, dstmtxs[0].header.field);
        TEST_ASSERT_EQ(mtxfile_general, dstmtxs[0].header.symmetry);
        TEST_ASSERT_EQ(mtx_double, dstmtxs[0].precision);
        TEST_ASSERT_EQ(4, dstmtxs[0].size.num_rows);
        TEST_ASSERT_EQ(4, dstmtxs[0].size.num_columns);
        TEST_ASSERT_EQ(2, dstmtxs[0].size.num_nonzeros);
        const struct mtxfile_matrix_coordinate_real_double * data0 =
            dstmtxs[0].data.matrix_coordinate_real_double;
        TEST_ASSERT_EQ(  1, data0[0].i); TEST_ASSERT_EQ(   1, data0[0].j);
        TEST_ASSERT_EQ(1.0, data0[0].a);
        TEST_ASSERT_EQ(  2, data0[1].i); TEST_ASSERT_EQ(   2, data0[1].j);
        TEST_ASSERT_EQ(2.0, data0[1].a);
        TEST_ASSERT_EQ(mtxfile_matrix, dstmtxs[1].header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, dstmtxs[1].header.format);
        TEST_ASSERT_EQ(mtxfile_real, dstmtxs[1].header.field);
        TEST_ASSERT_EQ(mtxfile_general, dstmtxs[1].header.symmetry);
        TEST_ASSERT_EQ(mtx_double, dstmtxs[1].precision);
        TEST_ASSERT_EQ(4, dstmtxs[1].size.num_rows);
        TEST_ASSERT_EQ(4, dstmtxs[1].size.num_columns);
        TEST_ASSERT_EQ(2, dstmtxs[1].size.num_nonzeros);
        const struct mtxfile_matrix_coordinate_real_double * data1 =
            dstmtxs[1].data.matrix_coordinate_real_double;
        TEST_ASSERT_EQ(  3, data1[0].i); TEST_ASSERT_EQ(   3, data1[0].j);
        TEST_ASSERT_EQ(3.0, data1[0].a);
        TEST_ASSERT_EQ(  4, data1[1].i); TEST_ASSERT_EQ(   4, data1[1].j);
        TEST_ASSERT_EQ(4.0, data1[1].a);
        mtxfile_free(&dstmtxs[0]);
        mtxfile_free(&dstmtxs[1]);
        mtxfile_free(&srcmtx);
    }
    return TEST_SUCCESS;
}

/**
 * `test_mtxfile_scatter()' tests scattering Matrix Market files from
 * an MPI root process to all other MPI processes in a communicator.
 */
int test_mtxfile_scatter(void)
{
    int err;
    int mpierr;
    char mpierrstr[MPI_MAX_ERROR_STRING];
    int mpierrstrlen;
    int root = 0;

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
    if (comm_size != 2)
        TEST_FAIL_MSG("Expected exactly two MPI processes");

    /* Get the MPI rank of the current process. */
    int rank;
    mpierr = MPI_Comm_rank(comm, &rank);
    if (mpierr) {
        MPI_Error_string(err, mpierrstr, &mpierrstrlen);
        fprintf(stderr, "%s: MPI_Comm_rank failed with %s\n",
                program_invocation_short_name, mpierrstr);
        MPI_Abort(comm, EXIT_FAILURE);
    }

    struct mtxmpierror mpierror;
    err = mtxmpierror_alloc(&mpierror, comm, NULL);
    if (err)
        MPI_Abort(comm, EXIT_FAILURE);

    /*
     * Array formats
     */

    {
        int num_rows[] = {2, 1};
        int num_columns = 3;
        const double srcdata0[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        const double srcdata1[] = {7.0, 8.0, 9.0};
        int64_t num_nonzeros[] = {6, 3};
        struct mtxfile srcmtxs[2];
        err = mtxfile_init_matrix_array_real_double(
            &srcmtxs[0], mtxfile_general, num_rows[0], num_columns, srcdata0);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        err = mtxfile_init_matrix_array_real_double(
            &srcmtxs[1], mtxfile_general, num_rows[1], num_columns, srcdata1);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));

        struct mtxfile dstmtx;
        err = mtxfile_scatter(srcmtxs, &dstmtx, root, comm, &mpierror);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s",
            err == MTX_ERR_MPI_COLLECTIVE
            ? mtxmpierror_description(&mpierror)
            : mtx_strerror(err));
        TEST_ASSERT_EQ(mtxfile_matrix, dstmtx.header.object);
        TEST_ASSERT_EQ(mtxfile_array, dstmtx.header.format);
        TEST_ASSERT_EQ(mtxfile_real, dstmtx.header.field);
        TEST_ASSERT_EQ(mtxfile_general, dstmtx.header.symmetry);
        TEST_ASSERT_EQ(mtx_double, dstmtx.precision);
        TEST_ASSERT_EQ(rank == root ? 2 : 1, dstmtx.size.num_rows);
        TEST_ASSERT_EQ(3, dstmtx.size.num_columns);
        TEST_ASSERT_EQ(-1, dstmtx.size.num_nonzeros);
        const double * data = dstmtx.data.array_real_double;
        if (rank == root) {
            TEST_ASSERT_EQ(1.0, data[0]);
            TEST_ASSERT_EQ(2.0, data[1]);
            TEST_ASSERT_EQ(3.0, data[2]);
            TEST_ASSERT_EQ(4.0, data[3]);
            TEST_ASSERT_EQ(5.0, data[4]);
            TEST_ASSERT_EQ(6.0, data[5]);
        } else {
            TEST_ASSERT_EQ(7.0, data[0]);
            TEST_ASSERT_EQ(8.0, data[1]);
            TEST_ASSERT_EQ(9.0, data[2]);
        }
        mtxfile_free(&dstmtx);
        mtxfile_free(&srcmtxs[0]);
        mtxfile_free(&srcmtxs[1]);
    }

    /*
     * Matrix coordinate formats
     */

    {
        int num_rows[] = {4, 4};
        int num_columns[] = {4, 4};
        const struct mtxfile_matrix_coordinate_real_double srcdata0[] = {
            {1,1,1.0}, {2,2,2.0}};
        const struct mtxfile_matrix_coordinate_real_double srcdata1[] = {
            {3,3,3.0}, {4,4,4.0}};
        int64_t num_nonzeros[] = {2, 2};
        struct mtxfile srcmtxs[2];
        err = mtxfile_init_matrix_coordinate_real_double(
            &srcmtxs[0], mtxfile_general, num_rows[0], num_columns[0],
            num_nonzeros[0], srcdata0);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        err = mtxfile_init_matrix_coordinate_real_double(
            &srcmtxs[1], mtxfile_general, num_rows[1], num_columns[1],
            num_nonzeros[1], srcdata1);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));

        struct mtxfile dstmtx;
        err = mtxfile_scatter(srcmtxs, &dstmtx, root, comm, &mpierror);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s",
            err == MTX_ERR_MPI_COLLECTIVE
            ? mtxmpierror_description(&mpierror)
            : mtx_strerror(err));
        TEST_ASSERT_EQ(mtxfile_matrix, dstmtx.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, dstmtx.header.format);
        TEST_ASSERT_EQ(mtxfile_real, dstmtx.header.field);
        TEST_ASSERT_EQ(mtxfile_general, dstmtx.header.symmetry);
        TEST_ASSERT_EQ(mtx_double, dstmtx.precision);
        TEST_ASSERT_EQ(4, dstmtx.size.num_rows);
        TEST_ASSERT_EQ(4, dstmtx.size.num_columns);
        TEST_ASSERT_EQ(2, dstmtx.size.num_nonzeros);
        const struct mtxfile_matrix_coordinate_real_double * data =
            dstmtx.data.matrix_coordinate_real_double;
        if (rank == root) {
            TEST_ASSERT_EQ(1, data[0].i); TEST_ASSERT_EQ(1, data[0].j);
            TEST_ASSERT_EQ(1.0, data[0].a);
            TEST_ASSERT_EQ(2, data[1].i); TEST_ASSERT_EQ(2, data[1].j);
            TEST_ASSERT_EQ(2.0, data[1].a);
        } else {
            TEST_ASSERT_EQ(3, data[0].i); TEST_ASSERT_EQ(3, data[0].j);
            TEST_ASSERT_EQ(3.0, data[0].a);
            TEST_ASSERT_EQ(4, data[1].i); TEST_ASSERT_EQ(4, data[1].j);
            TEST_ASSERT_EQ(4.0, data[1].a);
        }
        mtxfile_free(&dstmtx);
        mtxfile_free(&srcmtxs[0]);
        mtxfile_free(&srcmtxs[1]);
    }
    return TEST_SUCCESS;
}

/**
 * `test_mtxfile_alltoall()' tests all-to-all exchanges of Matrix
 * Market files among MPI processes in a communicator.
 */
int test_mtxfile_alltoall(void)
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
    if (comm_size != 2)
        TEST_FAIL_MSG("Expected exactly two MPI processes");

    /* Get the MPI rank of the current process. */
    int rank;
    mpierr = MPI_Comm_rank(comm, &rank);
    if (mpierr) {
        MPI_Error_string(err, mpierrstr, &mpierrstrlen);
        fprintf(stderr, "%s: MPI_Comm_rank failed with %s\n",
                program_invocation_short_name, mpierrstr);
        MPI_Abort(comm, EXIT_FAILURE);
    }

    struct mtxmpierror mpierror;
    err = mtxmpierror_alloc(&mpierror, comm, NULL);
    if (err)
        MPI_Abort(comm, EXIT_FAILURE);

    /*
     * Array formats
     */

    {
        int num_rows[] = {1, 1};
        int num_columns = 3;
        const double * srcdata0 = (rank == 0)
            ? ((const double[3]) {1.0, 2.0, 3.0})
            : ((const double[3]) {9.0, 8.0, 7.0});
        const double * srcdata1 = (rank == 0)
            ? ((const double[3]) {4.0, 5.0, 6.0})
            : ((const double[3]) {6.0, 5.0, 4.0});
        int64_t num_nonzeros[] = {3, 3};
        struct mtxfile srcmtxs[2];
        err = mtxfile_init_matrix_array_real_double(
            &srcmtxs[0], mtxfile_general, num_rows[0], num_columns, srcdata0);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        err = mtxfile_init_matrix_array_real_double(
            &srcmtxs[1], mtxfile_general, num_rows[1], num_columns, srcdata1);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));

        struct mtxfile dstmtxs[2];
        err = mtxfile_alltoall(srcmtxs, dstmtxs, comm, &mpierror);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s",
            err == MTX_ERR_MPI_COLLECTIVE
            ? mtxmpierror_description(&mpierror)
            : mtx_strerror(err));

        TEST_ASSERT_EQ(mtxfile_matrix, dstmtxs[0].header.object);
        TEST_ASSERT_EQ(mtxfile_array, dstmtxs[0].header.format);
        TEST_ASSERT_EQ(mtxfile_real, dstmtxs[0].header.field);
        TEST_ASSERT_EQ(mtxfile_general, dstmtxs[0].header.symmetry);
        TEST_ASSERT_EQ(mtx_double, dstmtxs[0].precision);
        TEST_ASSERT_EQ(1, dstmtxs[0].size.num_rows);
        TEST_ASSERT_EQ(3, dstmtxs[0].size.num_columns);
        TEST_ASSERT_EQ(-1, dstmtxs[0].size.num_nonzeros);
        const double * data0 = dstmtxs[0].data.array_real_double;
        if (rank == 0) {
            TEST_ASSERT_EQ(1.0, data0[0]);
            TEST_ASSERT_EQ(2.0, data0[1]);
            TEST_ASSERT_EQ(3.0, data0[2]);
        } else {
            TEST_ASSERT_EQ(4.0, data0[0]);
            TEST_ASSERT_EQ(5.0, data0[1]);
            TEST_ASSERT_EQ(6.0, data0[2]);
        }
        TEST_ASSERT_EQ(mtxfile_matrix, dstmtxs[1].header.object);
        TEST_ASSERT_EQ(mtxfile_array, dstmtxs[1].header.format);
        TEST_ASSERT_EQ(mtxfile_real, dstmtxs[1].header.field);
        TEST_ASSERT_EQ(mtxfile_general, dstmtxs[1].header.symmetry);
        TEST_ASSERT_EQ(mtx_double, dstmtxs[1].precision);
        TEST_ASSERT_EQ(1, dstmtxs[1].size.num_rows);
        TEST_ASSERT_EQ(3, dstmtxs[1].size.num_columns);
        TEST_ASSERT_EQ(-1, dstmtxs[1].size.num_nonzeros);
        const double * data1 = dstmtxs[1].data.array_real_double;
        if (rank == 0) {
            TEST_ASSERT_EQ(9.0, data1[0]);
            TEST_ASSERT_EQ(8.0, data1[1]);
            TEST_ASSERT_EQ(7.0, data1[2]);
        } else {
            TEST_ASSERT_EQ(6.0, data1[0]);
            TEST_ASSERT_EQ(5.0, data1[1]);
            TEST_ASSERT_EQ(4.0, data1[2]);
        }
        mtxfile_free(&dstmtxs[0]);
        mtxfile_free(&dstmtxs[1]);
        mtxfile_free(&srcmtxs[0]);
        mtxfile_free(&srcmtxs[1]);
    }

    /*
     * Matrix coordinate formats
     */

    {
        int num_rows = 4;
        int num_columns = 4;
        const struct mtxfile_matrix_coordinate_real_double * srcdata0 = (rank == 0)
            ? ((const struct mtxfile_matrix_coordinate_real_double[3])
                {{1,1,1.0}, {1,2,2.0}, {1,3,3.0}})
            : ((const struct mtxfile_matrix_coordinate_real_double[3])
                {{3,3,9.0}, {3,2,8.0}, {3,1,7.0}});
        const struct mtxfile_matrix_coordinate_real_double * srcdata1 = (rank == 0)
            ? ((const struct mtxfile_matrix_coordinate_real_double[3])
                {{2,1,4.0}, {2,2,5.0}, {2,3,6.0}})
            : ((const struct mtxfile_matrix_coordinate_real_double[3])
                {{2,3,6.0}, {2,2,5.0}, {2,1,4.0}});
        int64_t num_nonzeros = 3;
        struct mtxfile srcmtxs[2];
        err = mtxfile_init_matrix_coordinate_real_double(
            &srcmtxs[0], mtxfile_general, num_rows, num_columns,
            num_nonzeros, srcdata0);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        err = mtxfile_init_matrix_coordinate_real_double(
            &srcmtxs[1], mtxfile_general, num_rows, num_columns,
            num_nonzeros, srcdata1);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));

        struct mtxfile dstmtxs[2];
        err = mtxfile_alltoall(srcmtxs, dstmtxs, comm, &mpierror);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s",
            err == MTX_ERR_MPI_COLLECTIVE
            ? mtxmpierror_description(&mpierror)
            : mtx_strerror(err));

        TEST_ASSERT_EQ(mtxfile_matrix, dstmtxs[0].header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, dstmtxs[0].header.format);
        TEST_ASSERT_EQ(mtxfile_real, dstmtxs[0].header.field);
        TEST_ASSERT_EQ(mtxfile_general, dstmtxs[0].header.symmetry);
        TEST_ASSERT_EQ(mtx_double, dstmtxs[0].precision);
        TEST_ASSERT_EQ(4, dstmtxs[0].size.num_rows);
        TEST_ASSERT_EQ(4, dstmtxs[0].size.num_columns);
        TEST_ASSERT_EQ(3, dstmtxs[0].size.num_nonzeros);
        const struct mtxfile_matrix_coordinate_real_double * data0 =
            dstmtxs[0].data.matrix_coordinate_real_double;
        if (rank == 0) {
            TEST_ASSERT_EQ(1, data0[0].i); TEST_ASSERT_EQ(1, data0[0].j);
            TEST_ASSERT_EQ(1.0, data0[0].a);
            TEST_ASSERT_EQ(1, data0[1].i); TEST_ASSERT_EQ(2, data0[1].j);
            TEST_ASSERT_EQ(2.0, data0[1].a);
            TEST_ASSERT_EQ(1, data0[2].i); TEST_ASSERT_EQ(3, data0[2].j);
            TEST_ASSERT_EQ(3.0, data0[2].a);
        } else {
            TEST_ASSERT_EQ(2, data0[0].i); TEST_ASSERT_EQ(1, data0[0].j);
            TEST_ASSERT_EQ(4.0, data0[0].a);
            TEST_ASSERT_EQ(2, data0[1].i); TEST_ASSERT_EQ(2, data0[1].j);
            TEST_ASSERT_EQ(5.0, data0[1].a);
            TEST_ASSERT_EQ(2, data0[2].i); TEST_ASSERT_EQ(3, data0[2].j);
            TEST_ASSERT_EQ(6.0, data0[2].a);
        }
        TEST_ASSERT_EQ(mtxfile_matrix, dstmtxs[1].header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, dstmtxs[1].header.format);
        TEST_ASSERT_EQ(mtxfile_real, dstmtxs[1].header.field);
        TEST_ASSERT_EQ(mtxfile_general, dstmtxs[1].header.symmetry);
        TEST_ASSERT_EQ(mtx_double, dstmtxs[1].precision);
        TEST_ASSERT_EQ(4, dstmtxs[1].size.num_rows);
        TEST_ASSERT_EQ(4, dstmtxs[1].size.num_columns);
        TEST_ASSERT_EQ(3, dstmtxs[1].size.num_nonzeros);
        const struct mtxfile_matrix_coordinate_real_double * data1 =
            dstmtxs[1].data.matrix_coordinate_real_double;
        if (rank == 0) {
            TEST_ASSERT_EQ(3, data1[0].i); TEST_ASSERT_EQ(3, data1[0].j);
            TEST_ASSERT_EQ(9.0, data1[0].a);
            TEST_ASSERT_EQ(3, data1[1].i); TEST_ASSERT_EQ(2, data1[1].j);
            TEST_ASSERT_EQ(8.0, data1[1].a);
            TEST_ASSERT_EQ(3, data1[2].i); TEST_ASSERT_EQ(1, data1[2].j);
            TEST_ASSERT_EQ(7.0, data1[2].a);
        } else {
            TEST_ASSERT_EQ(2, data1[0].i); TEST_ASSERT_EQ(3, data1[0].j);
            TEST_ASSERT_EQ(6.0, data1[0].a);
            TEST_ASSERT_EQ(2, data1[1].i); TEST_ASSERT_EQ(2, data1[1].j);
            TEST_ASSERT_EQ(5.0, data1[1].a);
            TEST_ASSERT_EQ(2, data1[2].i); TEST_ASSERT_EQ(1, data1[2].j);
            TEST_ASSERT_EQ(4.0, data1[2].a);
        }
        mtxfile_free(&dstmtxs[0]);
        mtxfile_free(&dstmtxs[1]);
        mtxfile_free(&srcmtxs[0]);
        mtxfile_free(&srcmtxs[1]);
    }
    return TEST_SUCCESS;
}

/**
 * `test_mtxfile_scatterv()' tests scattering a Matrix Market file
 * from an MPI root process to all other MPI processes in a
 * communicator.
 */
int test_mtxfile_scatterv(void)
{
    int err;
    int mpierr;
    char mpierrstr[MPI_MAX_ERROR_STRING];
    int mpierrstrlen;
    int root = 0;

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
    if (comm_size != 2)
        TEST_FAIL_MSG("Expected exactly two MPI processes");

    /* Get the MPI rank of the current process. */
    int rank;
    mpierr = MPI_Comm_rank(comm, &rank);
    if (mpierr) {
        MPI_Error_string(err, mpierrstr, &mpierrstrlen);
        fprintf(stderr, "%s: MPI_Comm_rank failed with %s\n",
                program_invocation_short_name, mpierrstr);
        MPI_Abort(comm, EXIT_FAILURE);
    }

    struct mtxmpierror mpierror;
    err = mtxmpierror_alloc(&mpierror, comm, NULL);
    if (err)
        MPI_Abort(comm, EXIT_FAILURE);

    /*
     * Array formats
     */

    {
        int num_rows = 3;
        int num_columns = 3;
        const double srcdata[] = {
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
        int64_t num_nonzeros = sizeof(srcdata) / sizeof(*srcdata);
        struct mtxfile srcmtx;
        if (rank == root) {
            err = mtxfile_init_matrix_array_real_double(
                &srcmtx, mtxfile_general, num_rows, num_columns, srcdata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        }

        struct mtxfile dstmtx;
        const int sendcounts[2] = {3, 6};
        const int displs[2] = {0, 3};
        int recvcount = rank == root ? 3 : 6;
        err = mtxfile_scatterv(
            &srcmtx, sendcounts, displs, &dstmtx, recvcount,
            root, comm, &mpierror);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s",
            err == MTX_ERR_MPI_COLLECTIVE
            ? mtxmpierror_description(&mpierror)
            : mtx_strerror(err));
        TEST_ASSERT_EQ(mtxfile_matrix, dstmtx.header.object);
        TEST_ASSERT_EQ(mtxfile_array, dstmtx.header.format);
        TEST_ASSERT_EQ(mtxfile_real, dstmtx.header.field);
        TEST_ASSERT_EQ(mtxfile_general, dstmtx.header.symmetry);
        TEST_ASSERT_EQ(mtx_double, dstmtx.precision);
        TEST_ASSERT_EQ(rank == root ? 1 : 2, dstmtx.size.num_rows);
        TEST_ASSERT_EQ(3, dstmtx.size.num_columns);
        TEST_ASSERT_EQ(-1, dstmtx.size.num_nonzeros);
        if (rank == 0) {
            const double * data = dstmtx.data.array_real_double;
            TEST_ASSERT_EQ(1.0, data[0]);
            TEST_ASSERT_EQ(2.0, data[1]);
            TEST_ASSERT_EQ(3.0, data[2]);
        } else if (rank == 1) {
            const double * data = dstmtx.data.array_real_double;
            TEST_ASSERT_EQ(4.0, data[0]);
            TEST_ASSERT_EQ(5.0, data[1]);
            TEST_ASSERT_EQ(6.0, data[2]);
            TEST_ASSERT_EQ(7.0, data[3]);
            TEST_ASSERT_EQ(8.0, data[4]);
            TEST_ASSERT_EQ(9.0, data[5]);
        }
        mtxfile_free(&dstmtx);
        if (rank == root)
            mtxfile_free(&srcmtx);
    }

    {
        int num_rows = 4;
        const double srcdata[] = {1.0, 2.0, 3.0, 4.0};
        int64_t num_nonzeros = sizeof(srcdata) / sizeof(*srcdata);
        struct mtxfile srcmtx;
        if (rank == root) {
            err = mtxfile_init_vector_array_real_double(&srcmtx, num_rows, srcdata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        }

        struct mtxfile dstmtx;
        const int sendcounts[2] = {1, 3};
        const int displs[2] = {0, 1};
        int recvcount = rank == root ? 1 : 3;
        err = mtxfile_scatterv(
            &srcmtx, sendcounts, displs, &dstmtx, recvcount,
            root, comm, &mpierror);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s",
            err == MTX_ERR_MPI_COLLECTIVE
            ? mtxmpierror_description(&mpierror)
            : mtx_strerror(err));
        TEST_ASSERT_EQ(mtxfile_vector, dstmtx.header.object);
        TEST_ASSERT_EQ(mtxfile_array, dstmtx.header.format);
        TEST_ASSERT_EQ(mtxfile_real, dstmtx.header.field);
        TEST_ASSERT_EQ(mtxfile_general, dstmtx.header.symmetry);
        TEST_ASSERT_EQ(mtx_double, dstmtx.precision);
        TEST_ASSERT_EQ(rank == root ? 1 : 3, dstmtx.size.num_rows);
        TEST_ASSERT_EQ(-1, dstmtx.size.num_columns);
        TEST_ASSERT_EQ(-1, dstmtx.size.num_nonzeros);
        if (rank == 0) {
            const double * data = dstmtx.data.array_real_double;
            TEST_ASSERT_EQ(1.0, data[0]);
        } else if (rank == 1) {
            const double * data = dstmtx.data.array_real_double;
            TEST_ASSERT_EQ(2.0, data[0]);
            TEST_ASSERT_EQ(3.0, data[1]);
            TEST_ASSERT_EQ(4.0, data[2]);
        }
        mtxfile_free(&dstmtx);
        if (rank == root)
            mtxfile_free(&srcmtx);
    }

    /*
     * Matrix coordinate formats
     */

    {
        int num_rows = 4;
        int num_columns = 4;
        const struct mtxfile_matrix_coordinate_real_double srcdata[] = {
            {1, 1, 1.0}, {2, 2, 2.0}, {3, 3, 3.0}, {4, 4, 4.0}};
        int64_t num_nonzeros = sizeof(srcdata) / sizeof(*srcdata);
        struct mtxfile srcmtx;
        if (rank == root) {
            err = mtxfile_init_matrix_coordinate_real_double(
                &srcmtx, mtxfile_general, num_rows, num_columns, num_nonzeros, srcdata);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        }

        struct mtxfile dstmtx;
        const int sendcounts[2] = {1, 3};
        const int displs[2] = {0, 1};
        int recvcount = rank == root ? 1 : 3;
        err = mtxfile_scatterv(
            &srcmtx, sendcounts, displs, &dstmtx, recvcount,
            root, comm, &mpierror);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s",
            err == MTX_ERR_MPI_COLLECTIVE
            ? mtxmpierror_description(&mpierror)
            : mtx_strerror(err));
        TEST_ASSERT_EQ(mtxfile_matrix, dstmtx.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, dstmtx.header.format);
        TEST_ASSERT_EQ(mtxfile_real, dstmtx.header.field);
        TEST_ASSERT_EQ(mtxfile_general, dstmtx.header.symmetry);
        TEST_ASSERT_EQ(mtx_double, dstmtx.precision);
        TEST_ASSERT_EQ(4, dstmtx.size.num_rows);
        TEST_ASSERT_EQ(4, dstmtx.size.num_columns);
        TEST_ASSERT_EQ(rank == root ? 1 : 3, dstmtx.size.num_nonzeros);
        if (rank == 0) {
            const struct mtxfile_matrix_coordinate_real_double * data =
                dstmtx.data.matrix_coordinate_real_double;
            TEST_ASSERT_EQ(  1, data[0].i); TEST_ASSERT_EQ(   1, data[0].j);
            TEST_ASSERT_EQ(1.0, data[0].a);
        } else if (rank == 1) {
            const struct mtxfile_matrix_coordinate_real_double * data =
                dstmtx.data.matrix_coordinate_real_double;
            TEST_ASSERT_EQ(  2, data[0].i); TEST_ASSERT_EQ(   2, data[0].j);
            TEST_ASSERT_EQ(2.0, data[0].a);
            TEST_ASSERT_EQ(  3, data[1].i); TEST_ASSERT_EQ(   3, data[1].j);
            TEST_ASSERT_EQ(3.0, data[1].a);
            TEST_ASSERT_EQ(  4, data[2].i); TEST_ASSERT_EQ(   4, data[2].j);
            TEST_ASSERT_EQ(4.0, data[2].a);
        }
        mtxfile_free(&dstmtx);
        if (rank == root)
            mtxfile_free(&srcmtx);
    }

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

    /* 2. Run test suite. */
    TEST_SUITE_BEGIN("Running tests for distributed Matrix Market files\n");
    TEST_RUN(test_mtxfile_sendrecv);
    TEST_RUN(test_mtxfile_bcast);
    TEST_RUN(test_mtxfile_scatterv);
    TEST_RUN(test_mtxfile_gather);
    TEST_RUN(test_mtxfile_allgather);
    TEST_RUN(test_mtxfile_scatter);
    TEST_RUN(test_mtxfile_alltoall);
    TEST_SUITE_END();

    /* 3. Clean up and return. */
    MPI_Finalize();
    return (TEST_SUITE_STATUS == TEST_SUCCESS) ?
        EXIT_SUCCESS : EXIT_FAILURE;
}
