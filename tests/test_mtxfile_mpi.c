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
 * Unit tests for MPI communication routines for matrices and vectors
 * in Matrix Market format.
 */

#include "test.h"

#include <libmtx/libmtx-config.h>

#include <libmtx/error.h>
#include <libmtx/mtxfile/coordinate.h>
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
    err = mtxmpierror_alloc(&mpierror, comm);
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
    err = mtxmpierror_alloc(&mpierror, comm);
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
 * `test_mtxfile_data_scatterv()` tests scattering data lines from an
 * MPI root process to all other MPI processes in a communicator.
 */
int test_mtxfile_data_scatterv(void)
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

    /* Create a matrix on the root process. */
    int num_rows = 4;
    int num_columns = 4;
    const struct mtxfile_matrix_coordinate_real_double srcdata[] = {
        {1, 1, 1.0},
        {2, 2, 2.0},
        {3, 3, 3.0},
        {4, 4, 4.0}};
    int64_t num_nonzeros = sizeof(srcdata) / sizeof(*srcdata);
    struct mtxfile srcmtx;
    if (rank == root) {
        err = mtxfile_init_matrix_coordinate_real_double(
            &srcmtx, mtxfile_general, num_rows, num_columns, num_nonzeros, srcdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
    }

    /* Alloc a destination buffer. */
    union mtxfile_data dst;
    int recvcount = rank == root ? 2 : 2;
    err = mtxfile_data_alloc(
        &dst, mtxfile_matrix, mtxfile_coordinate, mtxfile_real, mtx_double,
        recvcount);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));

    /* Scatter the data lines from the root process. */
    struct mtxmpierror mpierror;
    err = mtxmpierror_alloc(&mpierror, comm);
    if (err)
        MPI_Abort(comm, EXIT_FAILURE);

    int sendcounts[2] = {2, 2};
    int displs[2] = {0, 2};
    err = mtxfile_data_scatterv(
        &srcmtx.data, mtxfile_matrix, mtxfile_coordinate, mtxfile_real, mtx_double,
        0, sendcounts, displs, &dst, 0, recvcount, root, comm, &mpierror);
    TEST_ASSERT_EQ(MTX_SUCCESS, err);

    /* Check the scattered matrix values. */
    if (rank == 0) {
        const struct mtxfile_matrix_coordinate_real_double * data =
            dst.matrix_coordinate_real_double;
        TEST_ASSERT_EQ(  1, data[0].i); TEST_ASSERT_EQ(   1, data[0].j);
        TEST_ASSERT_EQ(1.0, data[0].a);
        TEST_ASSERT_EQ(  2, data[1].i); TEST_ASSERT_EQ(   2, data[1].j);
        TEST_ASSERT_EQ(2.0, data[1].a);
    } else if (rank == 1) {
        const struct mtxfile_matrix_coordinate_real_double * data =
            dst.matrix_coordinate_real_double;
        TEST_ASSERT_EQ(  3, data[0].i); TEST_ASSERT_EQ(   3, data[0].j);
        TEST_ASSERT_EQ(3.0, data[0].a);
        TEST_ASSERT_EQ(  4, data[1].i); TEST_ASSERT_EQ(   4, data[1].j);
        TEST_ASSERT_EQ(4.0, data[1].a);
    }

    mtxfile_data_free(
        &dst, mtxfile_matrix, mtxfile_coordinate, mtxfile_real, mtx_double);
    if (rank == root)
        mtxfile_free(&srcmtx);
    return TEST_SUCCESS;
}

#if 0
/**
 * `test_mtxfile_matrix_coordinate_gather()` tests gathering a
 * distributed sparse matrix onto a single MPI root process.
 */
int test_mtxfile_matrix_coordinate_gather(void)
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

    /* Create a local, sparse matrix on each MPI process. */
    int num_rows = 4;
    int num_columns = 4;
    int64_t size;
    struct mtx_matrix_coordinate_real_single data[4];
    if (comm_size == 1) {
        size = 4;
        data[0].i = 1; data[0].j = 1; data[0].a = 1.0f;
        data[1].i = 2; data[1].j = 2; data[1].a = 2.0f;
        data[2].i = 3; data[2].j = 3; data[2].a = 3.0f;
        data[3].i = 4; data[3].j = 4; data[3].a = 4.0f;
    } else if (comm_size == 2) {
        size = 2;
        if (rank == 0) {
            data[0].i = 1; data[0].j = 1; data[0].a = 1.0f;
            data[1].i = 2; data[1].j = 2; data[1].a = 2.0f;
        } else {
            data[0].i = 3; data[0].j = 3; data[0].a = 3.0f;
            data[1].i = 4; data[1].j = 4; data[1].a = 4.0f;
        }
    } else {
        TEST_FAIL_MSG("Expected one or two MPI processes");
    }

    int num_comment_lines = 1;
    const char * comment_lines[] = { "% a comment\n" };
    struct mtx srcmtx;
    err = mtx_init_matrix_coordinate_real_single(
        &srcmtx, mtx_general, mtx_nontriangular,
        mtx_unsorted, mtx_assembled,
        num_comment_lines, comment_lines,
        num_rows, num_columns, size, data);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));

    /* Gather the distributed sparse matrix onto the root process. */
    int root = 0;
    struct mtx dstmtx;
    err = mtx_matrix_coordinate_gather(
        &dstmtx, &srcmtx, comm, root, &mpierr);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s",
                       mtx_strerror_mpi(err, mpierr, mpierrstr));
    mtx_free(&srcmtx);

    /* Check the gathered matrix on the MPI root process. */
    if (rank == 0) {
        TEST_ASSERT_EQ(mtx_matrix, dstmtx.object);
        TEST_ASSERT_EQ(mtx_coordinate, dstmtx.format);
        TEST_ASSERT_EQ(mtx_real, dstmtx.field);
        TEST_ASSERT_EQ(mtx_general, dstmtx.symmetry);
        TEST_ASSERT_EQ(1, dstmtx.num_comment_lines);
        TEST_ASSERT_STREQ("% a comment\n", dstmtx.comment_lines[0]);
        TEST_ASSERT_EQ(4, dstmtx.num_rows);
        TEST_ASSERT_EQ(4, dstmtx.num_columns);
        TEST_ASSERT_EQ(4, dstmtx.num_nonzeros);

        const struct mtx_matrix_coordinate_data * matrix_coordinate =
            &dstmtx.storage.matrix_coordinate;
        TEST_ASSERT_EQ(mtx_real, matrix_coordinate->field);
        TEST_ASSERT_EQ(mtx_single, matrix_coordinate->precision);
        TEST_ASSERT_EQ(mtx_general, matrix_coordinate->symmetry);
        TEST_ASSERT_EQ(mtx_nontriangular, matrix_coordinate->triangle);
        TEST_ASSERT_EQ(mtx_unsorted, matrix_coordinate->sorting);
        TEST_ASSERT_EQ(mtx_unassembled, matrix_coordinate->assembly);
        TEST_ASSERT_EQ(4, matrix_coordinate->num_rows);
        TEST_ASSERT_EQ(4, matrix_coordinate->num_columns);
        TEST_ASSERT_EQ(4, matrix_coordinate->size);
        const struct mtx_matrix_coordinate_real_single * dstmtxdata =
            matrix_coordinate->data.real_single;
        TEST_ASSERT_EQ(   1, dstmtxdata[0].i); TEST_ASSERT_EQ(   1, dstmtxdata[0].j);
        TEST_ASSERT_EQ(1.0f, dstmtxdata[0].a);
        TEST_ASSERT_EQ(   2, dstmtxdata[1].i); TEST_ASSERT_EQ(   2, dstmtxdata[1].j);
        TEST_ASSERT_EQ(2.0f, dstmtxdata[1].a);
        TEST_ASSERT_EQ(   3, dstmtxdata[2].i); TEST_ASSERT_EQ(   3, dstmtxdata[2].j);
        TEST_ASSERT_EQ(3.0f, dstmtxdata[2].a);
        TEST_ASSERT_EQ(   4, dstmtxdata[3].i); TEST_ASSERT_EQ(   4, dstmtxdata[3].j);
        TEST_ASSERT_EQ(4.0f, dstmtxdata[3].a);
        mtx_free(&dstmtx);
    }
    return TEST_SUCCESS;
}

/**
 * `test_mtxfile_matrix_coordinate_scatter()` tests scattering a
 * sparse matrix fom a single MPI root process to a group of
 * processes.
 */
int test_mtxfile_matrix_coordinate_scatter(void)
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

    /* Create a sparse matrix on the MPI root process (rank 0). */
    int root = 0;
    int num_comment_lines = 1;
    const char * comment_lines[] = {"% a comment\n"};
    int num_rows = 4;
    int num_columns = 4;
    int64_t size = 4;
    const struct mtx_matrix_coordinate_real_single data[] = {
        {1, 1, 1.0f},
        {2, 2, 2.0f},
        {3, 3, 3.0f},
        {4, 4, 4.0f},
    };
    struct mtx srcmtx;
    if (rank == root) {
        err = mtx_init_matrix_coordinate_real_single(
            &srcmtx, mtx_general, mtx_nontriangular,
            mtx_unsorted, mtx_unassembled,
            num_comment_lines, comment_lines,
            num_rows, num_columns, size, data);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
    }

    /* Determine the rows and columns owned by the current process. */
    struct mtx_index_set rows[2];
    struct mtx_index_set columns[2];
    if (comm_size == 1) {
        err = mtx_index_set_init_interval(&rows[0], 0, num_rows+1);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        err = mtx_index_set_init_interval(&columns[0], 0, num_columns+1);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
    } else if (comm_size == 2) {
        {
            int rank = 0;
            int start_row = 1+rank*((num_rows+(comm_size-1))/comm_size);
            int end_row = 1+(rank+1)*((num_rows+(comm_size-1))/comm_size);
            err = mtx_index_set_init_interval(&rows[0], start_row, end_row);
            TEST_ASSERT_EQ(MTX_SUCCESS, err);
            err = mtx_index_set_init_interval(&columns[0], 0, num_columns+1);
            TEST_ASSERT_EQ(MTX_SUCCESS, err);
        }
        {
            int rank = 1;
            int start_row = 1+rank*((num_rows+(comm_size-1))/comm_size);
            int end_row = 1+(rank+1)*((num_rows+(comm_size-1))/comm_size);
            err = mtx_index_set_init_interval(&rows[1], start_row, end_row);
            TEST_ASSERT_EQ(MTX_SUCCESS, err);
            err = mtx_index_set_init_interval(&columns[1], 0, num_columns+1);
            TEST_ASSERT_EQ(MTX_SUCCESS, err);
        }
    } else {
        TEST_FAIL_MSG("Expected one or two MPI processes");
    }

    /* Scatter the matrix from the root process to the process group. */
    struct mtx dstmtx;
    err = mtx_matrix_coordinate_scatter(
        &dstmtx, &srcmtx, rows, columns, comm, root, &mpierr);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s",
                       mtx_strerror_mpi(err, mpierr, mpierrstr));
    if (rank == root)
        mtx_free(&srcmtx);

    /* Check the local matrix on each MPI process. */
    TEST_ASSERT_EQ(mtx_matrix, dstmtx.object);
    TEST_ASSERT_EQ(mtx_coordinate, dstmtx.format);
    TEST_ASSERT_EQ(mtx_real, dstmtx.field);
    TEST_ASSERT_EQ(mtx_general, dstmtx.symmetry);
    TEST_ASSERT_EQ(1, dstmtx.num_comment_lines);
    TEST_ASSERT_STREQ("% a comment\n", dstmtx.comment_lines[0]);
    TEST_ASSERT_EQ(4, dstmtx.num_rows);
    TEST_ASSERT_EQ(4, dstmtx.num_columns);
    if (comm_size == 1) {
        const struct mtx_matrix_coordinate_data * matrix_coordinate =
            &dstmtx.storage.matrix_coordinate;
        TEST_ASSERT_EQ(mtx_real, matrix_coordinate->field);
        TEST_ASSERT_EQ(mtx_single, matrix_coordinate->precision);
        TEST_ASSERT_EQ(mtx_general, matrix_coordinate->symmetry);
        TEST_ASSERT_EQ(mtx_nontriangular, matrix_coordinate->triangle);
        TEST_ASSERT_EQ(mtx_unsorted, matrix_coordinate->sorting);
        TEST_ASSERT_EQ(mtx_unassembled, matrix_coordinate->assembly);
        TEST_ASSERT_EQ(4, matrix_coordinate->num_rows);
        TEST_ASSERT_EQ(4, matrix_coordinate->num_columns);
        TEST_ASSERT_EQ(4, matrix_coordinate->size);
        const struct mtx_matrix_coordinate_real_single * destmtxdata =
            matrix_coordinate->data.real_single;
        TEST_ASSERT_EQ(   1, destmtxdata[0].i); TEST_ASSERT_EQ(   1, destmtxdata[0].j);
        TEST_ASSERT_EQ(1.0f, destmtxdata[0].a);
        TEST_ASSERT_EQ(   2, destmtxdata[1].i); TEST_ASSERT_EQ(   2, destmtxdata[1].j);
        TEST_ASSERT_EQ(2.0f, destmtxdata[1].a);
        TEST_ASSERT_EQ(   3, destmtxdata[2].i); TEST_ASSERT_EQ(   3, destmtxdata[2].j);
        TEST_ASSERT_EQ(3.0f, destmtxdata[2].a);
        TEST_ASSERT_EQ(   4, destmtxdata[3].i); TEST_ASSERT_EQ(   4, destmtxdata[3].j);
        TEST_ASSERT_EQ(4.0f, destmtxdata[3].a);
    } else if (comm_size == 2) {
        if (rank == 0) {
            const struct mtx_matrix_coordinate_data * matrix_coordinate =
                &dstmtx.storage.matrix_coordinate;
            TEST_ASSERT_EQ(mtx_real, matrix_coordinate->field);
            TEST_ASSERT_EQ(mtx_single, matrix_coordinate->precision);
            TEST_ASSERT_EQ(mtx_general, matrix_coordinate->symmetry);
            TEST_ASSERT_EQ(mtx_nontriangular, matrix_coordinate->triangle);
            TEST_ASSERT_EQ(mtx_unsorted, matrix_coordinate->sorting);
            TEST_ASSERT_EQ(mtx_unassembled, matrix_coordinate->assembly);
            TEST_ASSERT_EQ(4, matrix_coordinate->num_rows);
            TEST_ASSERT_EQ(4, matrix_coordinate->num_columns);
            TEST_ASSERT_EQ(2, matrix_coordinate->size);
            const struct mtx_matrix_coordinate_real_single * dstmtxdata =
                matrix_coordinate->data.real_single;
            TEST_ASSERT_EQ(   1, dstmtxdata[0].i);
            TEST_ASSERT_EQ(   1, dstmtxdata[0].j);
            TEST_ASSERT_EQ(1.0f, dstmtxdata[0].a);
            TEST_ASSERT_EQ(   2, dstmtxdata[1].i);
            TEST_ASSERT_EQ(   2, dstmtxdata[1].j);
            TEST_ASSERT_EQ(2.0f, dstmtxdata[1].a);
        } else if (rank == 1) {
            const struct mtx_matrix_coordinate_data * matrix_coordinate =
                &dstmtx.storage.matrix_coordinate;
            TEST_ASSERT_EQ(mtx_real, matrix_coordinate->field);
            TEST_ASSERT_EQ(mtx_single, matrix_coordinate->precision);
            TEST_ASSERT_EQ(mtx_general, matrix_coordinate->symmetry);
            TEST_ASSERT_EQ(mtx_nontriangular, matrix_coordinate->triangle);
            TEST_ASSERT_EQ(mtx_unsorted, matrix_coordinate->sorting);
            TEST_ASSERT_EQ(mtx_unassembled, matrix_coordinate->assembly);
            TEST_ASSERT_EQ(4, matrix_coordinate->num_rows);
            TEST_ASSERT_EQ(4, matrix_coordinate->num_columns);
            TEST_ASSERT_EQ(2, matrix_coordinate->size);
            const struct mtx_matrix_coordinate_real_single * dstmtxdata =
                matrix_coordinate->data.real_single;
            TEST_ASSERT_EQ(   3, dstmtxdata[0].i);
            TEST_ASSERT_EQ(   3, dstmtxdata[0].j);
            TEST_ASSERT_EQ(3.0f, dstmtxdata[0].a);
            TEST_ASSERT_EQ(   4, dstmtxdata[1].i);
            TEST_ASSERT_EQ(   4, dstmtxdata[1].j);
            TEST_ASSERT_EQ(4.0f, dstmtxdata[1].a);
        }
    } else {
        TEST_FAIL_MSG("Expected one or two MPI processes");
    }
    mtx_free(&dstmtx);
    return TEST_SUCCESS;
}
#endif

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
    TEST_RUN(test_mtxfile_data_scatterv);
#if 0
    TEST_RUN(test_mtxfile_matrix_coordinate_gather);
    TEST_RUN(test_mtxfile_matrix_coordinate_scatter);
#endif
    TEST_SUITE_END();

    /* 3. Clean up and return. */
    MPI_Finalize();
    return (TEST_SUITE_STATUS == TEST_SUCCESS) ?
        EXIT_SUCCESS : EXIT_FAILURE;
}
