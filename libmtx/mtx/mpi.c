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
 * Functions for communicating objects in Matrix Market format between
 * processes using MPI.
 */

#include <libmtx/libmtx-config.h>

#ifdef LIBMTX_HAVE_MPI
#include <libmtx/error.h>
#include <libmtx/matrix/coordinate.h>
#include <libmtx/matrix/coordinate/mpi.h>
#include <libmtx/mtx/header.h>
#include <libmtx/mtx/mpi.h>
#include <libmtx/mtx/mtx.h>
#include <libmtx/mtx/submatrix.h>
#include <libmtx/util/index_set.h>
#include <libmtx/vector/coordinate.h>

#include <mpi.h>

#include <errno.h>

#include <stdlib.h>
#include <string.h>

/**
 * `mtx_send_header()' sends the header information of `struct mtx' to
 * another MPI process.
 *
 * This is analogous to `MPI_Send()' and requires the receiving
 * process to perform a matching call to `mtx_recv_header()'.
 */
static int mtx_send_header(
    const struct mtx * mtx,
    int dest,
    int tag,
    MPI_Comm comm,
    int * mpierrcode)
{
    *mpierrcode = MPI_Send(&mtx->object, 1, MPI_INT, dest, tag, comm);
    if (*mpierrcode)
        return MTX_ERR_MPI;
    *mpierrcode = MPI_Send(&mtx->format, 1, MPI_INT, dest, tag, comm);
    if (*mpierrcode)
        return MTX_ERR_MPI;
    *mpierrcode = MPI_Send(&mtx->field, 1, MPI_INT, dest, tag, comm);
    if (*mpierrcode)
        return MTX_ERR_MPI;
    *mpierrcode = MPI_Send(&mtx->symmetry, 1, MPI_INT, dest, tag, comm);
    if (*mpierrcode)
        return MTX_ERR_MPI;
    return MTX_SUCCESS;
}

/**
 * `mtx_send_comments()' sends the comments section of `struct
 * mtx' to another MPI process.
 *
 * This is analogous to `MPI_Send()' and requires the receiving
 * process to perform a matching call to `mtx_recv_comments()'.
 */
static int mtx_send_comments(
    const struct mtx * mtx,
    int dest,
    int tag,
    MPI_Comm comm,
    int * mpierrcode)
{
    *mpierrcode = MPI_SUCCESS;

    /* Send the number of comment lines. */
    *mpierrcode = MPI_Send(&mtx->num_comment_lines, 1, MPI_INT, dest, tag, comm);
    if (*mpierrcode)
        return MTX_ERR_MPI;

    /* Send each comment line. */
    for (int i = 0; i < mtx->num_comment_lines; i++) {
        int n = strlen(mtx->comment_lines[i]);
        *mpierrcode = MPI_Send(&n, 1, MPI_INT, dest, tag, comm);
        if (*mpierrcode)
            return MTX_ERR_MPI;

        *mpierrcode = MPI_Send(
            mtx->comment_lines[i], n, MPI_CHAR, dest, tag, comm);
        if (*mpierrcode)
            return MTX_ERR_MPI;
    }

    return MTX_SUCCESS;
}

/**
 * `mtx_send_size()' sends the size information of `struct mtx'
 * to another MPI process.
 *
 * This is analogous to `MPI_Send()' and requires the receiving
 * process to perform a matching call to `mtx_recv_size()'.
 */
static int mtx_send_size(
    const struct mtx * mtx,
    int dest,
    int tag,
    MPI_Comm comm,
    int * mpierrcode)
{
    *mpierrcode = MPI_SUCCESS;

    /* Receive number of rows, columns and nonzeros. */
    *mpierrcode = MPI_Send(
        &mtx->num_rows, 1, MPI_INT, dest, tag, comm);
    if (*mpierrcode)
        return MTX_ERR_MPI;
    *mpierrcode = MPI_Send(
        &mtx->num_columns, 1, MPI_INT, dest, tag, comm);
    if (*mpierrcode)
        return MTX_ERR_MPI;
    *mpierrcode = MPI_Send(
        &mtx->num_nonzeros, 1, MPI_INT64_T, dest, tag, comm);
    if (*mpierrcode)
        return MTX_ERR_MPI;
    return MTX_SUCCESS;
}

/**
 * `mtx_send_data()' sends the nonzero data of `struct mtx' to
 * another MPI process.
 *
 * This is analogous to `MPI_Send()' and requires the receiving
 * process to perform a matching call to `mtx_recv_data()'.
 */
static int mtx_send_data(
    const struct mtx * mtx,
    int dest,
    int tag,
    MPI_Comm comm,
    int * mpierrcode)
{
    int err;

    /* Send the nonzero data. */
    if (mtx->object == mtx_matrix) {
        if (mtx->format == mtx_coordinate) {
            const struct mtx_matrix_coordinate_data * matrix_coordinate =
                &mtx->storage.matrix_coordinate;
            return mtx_matrix_coordinate_send(
                matrix_coordinate,
                dest, tag, comm, mpierrcode);
        } else {
            return MTX_ERR_INVALID_MTX_FORMAT;
        }
    } else {
        return MTX_ERR_INVALID_MTX_OBJECT;
    }
}

/**
 * `mtx_send()' sends a `struct mtx' to another MPI process.
 *
 * This is analogous to `MPI_Send()' and requires the receiving
 * process to perform a matching call to `mtx_recv()'.
 */
int mtx_send(
    const struct mtx * mtx,
    int dest,
    int tag,
    MPI_Comm comm,
    int * mpierrcode)
{
    int err;
    *mpierrcode = MPI_SUCCESS;

    /* Send the Matrix Market header. */
    err = mtx_send_header(mtx, dest, tag, comm, mpierrcode);
    if (err)
        return err;

    /* Send Matrix Market comments. */
    err = mtx_send_comments(mtx, dest, tag, comm, mpierrcode);
    if (err)
        return err;

    /* Send Matrix Market size information. */
    err = mtx_send_size(mtx, dest, tag, comm, mpierrcode);
    if (err)
        return err;

    /* Send Matrix Market data. */
    err = mtx_send_data(mtx, dest, tag, comm, mpierrcode);
    if (err)
        return err;

    return MTX_SUCCESS;
}

/**
 * `mtx_recv_header()' receives the header information of `struct
 * mtx' from another MPI process.
 *
 * This is analogous to `MPI_Recv()' and requires the sending process
 * to perform a matching call to `mtx_send_header()'.
 */
static int mtx_recv_header(
    struct mtx * mtx,
    int source,
    int tag,
    MPI_Comm comm,
    int * mpierrcode)
{
    *mpierrcode = MPI_Recv(
        &mtx->object, 1, MPI_INT, source, tag, comm, MPI_STATUS_IGNORE);
    if (*mpierrcode)
        return MTX_ERR_MPI;
    *mpierrcode = MPI_Recv(
        &mtx->format, 1, MPI_INT, source, tag, comm, MPI_STATUS_IGNORE);
    if (*mpierrcode)
        return MTX_ERR_MPI;
    *mpierrcode = MPI_Recv(
        &mtx->field, 1, MPI_INT, source, tag, comm, MPI_STATUS_IGNORE);
    if (*mpierrcode)
        return MTX_ERR_MPI;
    *mpierrcode = MPI_Recv(
        &mtx->symmetry, 1, MPI_INT, source, tag, comm, MPI_STATUS_IGNORE);
    if (*mpierrcode)
        return MTX_ERR_MPI;
    return MTX_SUCCESS;
}

/**
 * `mtx_recv_comments()' receives the comments section of `struct
 * mtx' from another MPI process.
 *
 * This is analogous to `MPI_Recv()' and requires the sending process
 * to perform a matching call to `mtx_send_comments()'.
 */
static int mtx_recv_comments(
    struct mtx * mtx,
    int dest,
    int tag,
    MPI_Comm comm,
    int * mpierrcode)
{
    *mpierrcode = MPI_SUCCESS;

    /* Receive the number of comment lines. */
    *mpierrcode = MPI_Recv(
        &mtx->num_comment_lines, 1, MPI_INT, dest, tag, comm, MPI_STATUS_IGNORE);
    if (*mpierrcode)
        return MTX_ERR_MPI;

    /* Allocate storage for comment lines. */
    mtx->comment_lines = malloc(mtx->num_comment_lines * sizeof(char *));
    if (!mtx->comment_lines)
        return MTX_ERR_ERRNO;

    /* Receive each comment line. */
    for (int i = 0; i < mtx->num_comment_lines; i++) {
        int n;
        *mpierrcode = MPI_Recv(&n, 1, MPI_INT, dest, tag, comm, MPI_STATUS_IGNORE);
        if (*mpierrcode) {
            for (int j = i-1; j >= 0; j--)
                free(mtx->comment_lines[j]);
            free(mtx->comment_lines);
            return MTX_ERR_MPI;
        }

        /* Allocate storage for the comment line. */
        mtx->comment_lines[i] = malloc((n+1) * sizeof(char));
        if (!mtx->comment_lines[i]) {
            for (int j = i-1; j >= 0; j--)
                free(mtx->comment_lines[j]);
            free(mtx->comment_lines);
            return MTX_ERR_ERRNO;
        }

        /* Receive the comment line. */
        *mpierrcode = MPI_Recv(
            mtx->comment_lines[i], n, MPI_CHAR, dest, tag, comm, MPI_STATUS_IGNORE);
        if (*mpierrcode) {
            for (int j = i; j >= 0; j--)
                free(mtx->comment_lines[j]);
            free(mtx->comment_lines);
            return MTX_ERR_MPI;
        }
        mtx->comment_lines[i][n] = '\0';
    }

    return MTX_SUCCESS;
}

/**
 * `mtx_recv_size()' receives the size information of `struct
 * mtx' from another MPI process.
 *
 * This is analogous to `MPI_Recv()' and requires the sending process
 * to perform a matching call to `mtx_send_size()'.
 */
static int mtx_recv_size(
    struct mtx * mtx,
    int source,
    int tag,
    MPI_Comm comm,
    int * mpierrcode)
{
    *mpierrcode = MPI_SUCCESS;

    /* Receive number of rows, columns and nonzeros. */
    *mpierrcode = MPI_Recv(
        &mtx->num_rows, 1, MPI_INT, source, tag, comm, MPI_STATUS_IGNORE);
    if (*mpierrcode)
        return MTX_ERR_MPI;
    *mpierrcode = MPI_Recv(
        &mtx->num_columns, 1, MPI_INT, source, tag, comm, MPI_STATUS_IGNORE);
    if (*mpierrcode)
        return MTX_ERR_MPI;
    *mpierrcode = MPI_Recv(
        &mtx->num_nonzeros, 1, MPI_INT64_T, source, tag, comm, MPI_STATUS_IGNORE);
    if (*mpierrcode)
        return MTX_ERR_MPI;
    return MTX_SUCCESS;
}

/**
 * `mtx_recv_data()' receives the nonzero data of `struct mtx'
 * from another MPI process.
 *
 * This is analogous to `MPI_Recv()' and requires the sending process
 * to perform a matching call to `mtx_send_data()'.
 */
static int mtx_recv_data(
    struct mtx * mtx,
    int source,
    int tag,
    MPI_Comm comm,
    int * mpierrcode)
{
    int err;
    if (mtx->object == mtx_matrix) {
        if (mtx->format == mtx_coordinate) {
            struct mtx_matrix_coordinate_data * matrix_coordinate =
                &mtx->storage.matrix_coordinate;
            return mtx_matrix_coordinate_recv(
                matrix_coordinate,
                source, tag, comm, mpierrcode);
        } else {
            return MTX_ERR_INVALID_MTX_FORMAT;
        }
    } else {
        return MTX_ERR_INVALID_MTX_OBJECT;
    }
}

/**
 * `mtx_recv()' receives a `struct mtx' from another MPI
 * process.
 *
 * This is analogous to `MPI_Recv()' and requires the sending
 * process to perform a matching call to `mtx_send()'.
 */
int mtx_recv(
    struct mtx * mtx,
    int source,
    int tag,
    MPI_Comm comm,
    int * mpierrcode)
{
    int err;
    *mpierrcode = MPI_SUCCESS;

    /* Receive the Matrix Market header. */
    err = mtx_recv_header(mtx, source, tag, comm, mpierrcode);
    if (err)
        return err;

    /* Receive Matrix Market comments. */
    err = mtx_recv_comments(mtx, source, tag, comm, mpierrcode);
    if (err)
        return err;

    /* Receive Matrix Market size information. */
    err = mtx_recv_size(mtx, source, tag, comm, mpierrcode);
    if (err) {
        for (int j = mtx->num_comment_lines-1; j >= 0; j--)
            free(mtx->comment_lines[j]);
        free(mtx->comment_lines);
        return err;
    }

    /* Receive Matrix Market data. */
    err = mtx_recv_data(mtx, source, tag, comm, mpierrcode);
    if (err) {
        for (int j = mtx->num_comment_lines-1; j >= 0; j--)
            free(mtx->comment_lines[j]);
        free(mtx->comment_lines);
        return err;
    }

    return MTX_SUCCESS;
}

/**
 * `mtx_bcast_header()' broadcasts the header section of `struct mtx'
 * to other MPI processes in a communicator.
 *
 * This is analogous to `MPI_Bcast()' and requires all processes in
 * the communicator to collectively call `mtx_bcast_header()'.
 */
static int mtx_bcast_header(
    struct mtx * mtx,
    int root,
    MPI_Comm comm,
    int * mpierrcode)
{
    *mpierrcode = MPI_Bcast(&mtx->object, 1, MPI_INT, root, comm);
    if (*mpierrcode)
        return MTX_ERR_MPI;
    *mpierrcode = MPI_Bcast(&mtx->format, 1, MPI_INT, root, comm);
    if (*mpierrcode)
        return MTX_ERR_MPI;
    *mpierrcode = MPI_Bcast(&mtx->field, 1, MPI_INT, root, comm);
    if (*mpierrcode)
        return MTX_ERR_MPI;
    *mpierrcode = MPI_Bcast(&mtx->symmetry, 1, MPI_INT, root, comm);
    if (*mpierrcode)
        return MTX_ERR_MPI;
    return MTX_SUCCESS;
}

/**
 * `mtx_bcast_comments()' broadcasts the comments section of `struct
 * mtx' to other MPI processes in a communicator.
 *
 * This is analogous to `MPI_Bcast()' and requires all processes in
 * the communicator to collectively call `mtx_bcast_comments()'.
 */
static int mtx_bcast_comments(
    struct mtx * mtx,
    int root,
    MPI_Comm comm,
    int * mpierrcode)
{
    /* Get the MPI rank of the current process. */
    int rank;
    *mpierrcode = MPI_Comm_rank(comm, &rank);
    if (*mpierrcode)
        return MTX_ERR_MPI;

    /* Broadcast the number of comment lines. */
    *mpierrcode = MPI_Bcast(&mtx->num_comment_lines, 1, MPI_INT, root, comm);
    if (*mpierrcode)
        return MTX_ERR_MPI;

    /* Allocate storage for comment lines. */
    if (rank != root) {
        mtx->comment_lines = malloc(mtx->num_comment_lines * sizeof(char *));
        if (!mtx->comment_lines)
            return MTX_ERR_ERRNO;
    }

    /* Broadcast each comment line. */
    for (int i = 0; i < mtx->num_comment_lines; i++) {
        int n = rank == root ? strlen(mtx->comment_lines[i]) : 0;
        *mpierrcode = MPI_Bcast(&n, 1, MPI_INT, root, comm);
        if (*mpierrcode)
            return MTX_ERR_MPI;

        /* Allocate storage for the comment line. */
        if (rank != root) {
            mtx->comment_lines[i] = malloc((n+1) * sizeof(char));
            if (!mtx->comment_lines[i]) {
                for (int j = i-1; j >= 0; j--)
                    free(mtx->comment_lines[j]);
                free(mtx->comment_lines);
                return MTX_ERR_ERRNO;
            }
        }

        *mpierrcode = MPI_Bcast(
            mtx->comment_lines[i], n, MPI_CHAR, root, comm);
        if (*mpierrcode)
            return MTX_ERR_MPI;
        if (rank != root)
            mtx->comment_lines[i][n] = '\0';
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_bcast_size()' broadcasts the size section of `struct mtx' to
 * other MPI processes in a communicator.
 *
 * This is analogous to `MPI_Bcast()' and requires all processes in
 * the communicator to collectively call `mtx_bcast_size()'.
 */
static int mtx_bcast_size(
    struct mtx * mtx,
    int root,
    MPI_Comm comm,
    int * mpierrcode)
{
    *mpierrcode = MPI_Bcast(
        &mtx->num_rows, 1, MPI_INT, root, comm);
    if (*mpierrcode)
        return MTX_ERR_MPI;
    *mpierrcode = MPI_Bcast(
        &mtx->num_columns, 1, MPI_INT, root, comm);
    if (*mpierrcode)
        return MTX_ERR_MPI;
    *mpierrcode = MPI_Bcast(
        &mtx->num_nonzeros, 1, MPI_INT64_T, root, comm);
    if (*mpierrcode)
        return MTX_ERR_MPI;
    return MTX_SUCCESS;
}

/**
 * `mtx_bcast_data()' broadcasts the data section of `struct mtx' to
 * other MPI processes in a communicator.
 *
 * This is analogous to `MPI_Bcast()' and requires all processes in
 * the communicator to collectively call `mtx_bcast_data()'.
 */
static int mtx_bcast_data(
    struct mtx * mtx,
    int root,
    MPI_Comm comm,
    int * mpierrcode)
{
    int err;
    if (mtx->object == mtx_matrix) {
        if (mtx->format == mtx_coordinate) {
            struct mtx_matrix_coordinate_data * matrix_coordinate =
                &mtx->storage.matrix_coordinate;
            return mtx_matrix_coordinate_bcast(
                matrix_coordinate, root, comm, mpierrcode);
        } else {
            return MTX_ERR_INVALID_MTX_FORMAT;
        }
    } else {
        return MTX_ERR_INVALID_MTX_OBJECT;
    }
}

/**
 * `mtx_bcast()' broadcasts a `struct mtx' from an MPI root
 * process to other processes in a communicator.
 *
 * This is analogous to `MPI_Bcast()' and requires every process in
 * the communicator to perform matching calls to `mtx_bcast()'.
 */
int mtx_bcast(
    struct mtx * mtx,
    int root,
    MPI_Comm comm,
    int * mpierrcode)
{
    int err;
    err = mtx_bcast_header(mtx, root, comm, mpierrcode);
    if (err)
        return err;
    err = mtx_bcast_comments(mtx, root, comm, mpierrcode);
    if (err)
        return err;
    err = mtx_bcast_size(mtx, root, comm, mpierrcode);
    if (err)
        return err;
    err = mtx_bcast_data(mtx, root, comm, mpierrcode);
    if (err)
        return err;
    return MTX_SUCCESS;
}

/**
 * `mtx_matrix_coordinate_gather()` gathers a distributed Matrix
 * Market object representing a sparse (coordinate) matrix onto a
 * single MPI root process.
 */
int mtx_matrix_coordinate_gather(
    struct mtx * dstmtx,
    const struct mtx * srcmtx,
    MPI_Comm comm,
    int root,
    int * mpierrcode)
{
    int err;
    if (srcmtx->object != mtx_matrix)
        return MTX_ERR_INVALID_MTX_OBJECT;
    if (srcmtx->format != mtx_coordinate)
        return MTX_ERR_INVALID_MTX_FORMAT;

    /* Get the size of the MPI communicator. */
    int comm_size;
    *mpierrcode = MPI_Comm_size(comm, &comm_size);
    if (*mpierrcode)
        return MTX_ERR_MPI;

    /* Get the MPI rank of the current process. */
    int rank;
    *mpierrcode = MPI_Comm_rank(comm, &rank);
    if (*mpierrcode)
        return MTX_ERR_MPI;

    /* Allocate storage for gathering matrices on the root process. */
    struct mtx * tmpmtxs;
    if (rank == root) {
        tmpmtxs = malloc(sizeof(struct mtx) * comm_size);
        if (!tmpmtxs)
            return MTX_ERR_ERRNO;
    }

    /* Gather matrices from each MPI process. */
    for (int p = 0; p < comm_size; p++) {
        if (rank == root) {
            if (p == root) {
                /* For the root process, there is no need for any
                 * communication; just copy the matrix. */
                err = mtx_copy_init(&tmpmtxs[p], srcmtx);
                if (err) {
                    for (int q = p-1; q >= 0; q--)
                        mtx_free(&tmpmtxs[q]);
                    free(tmpmtxs);
                    return err;
                }
            } else {
                /* Receive the matrix at the root process. */
                err = mtx_recv(&tmpmtxs[p], p, 0, comm, mpierrcode);
                if (err) {
                    for (int q = p-1; q >= 0; q--)
                        mtx_free(&tmpmtxs[q]);
                    free(tmpmtxs);
                    return err;
                }
            }
        } else if (p != root) {
            /* Send the matrix to the root process. */
            err = mtx_send(srcmtx, root, 0, comm, mpierrcode);
            if (err)
                return err;
        }
    }

    if (rank != root)
        return MTX_SUCCESS;

    /* Get the Matrix Market object type. */
    enum mtx_object object;
    if (comm_size > 0)
        object = tmpmtxs[0].object;
    for (int p = 1; p < comm_size; p++) {
        if (object != tmpmtxs[p].object) {
            for (int p = 0; p < comm_size; p++)
                mtx_free(&tmpmtxs[p]);
            free(tmpmtxs);
            return MTX_ERR_INVALID_MTX_OBJECT;
        }
    }

    /* Get the Matrix Market format. */
    enum mtx_format format;
    if (comm_size > 0)
        format = tmpmtxs[0].format;
    for (int p = 1; p < comm_size; p++) {
        if (format != tmpmtxs[p].format) {
            for (int p = 0; p < comm_size; p++)
                mtx_free(&tmpmtxs[p]);
            free(tmpmtxs);
            return MTX_ERR_INVALID_MTX_FORMAT;
        }
    }

    /* Get the Matrix Market field. */
    enum mtx_field field;
    if (comm_size > 0)
        field = tmpmtxs[0].field;
    for (int p = 1; p < comm_size; p++) {
        if (field != tmpmtxs[p].field) {
            for (int p = 0; p < comm_size; p++)
                mtx_free(&tmpmtxs[p]);
            free(tmpmtxs);
            return MTX_ERR_INVALID_MTX_FIELD;
        }
    }

    /* Get the Matrix Market symmetry. */
    enum mtx_symmetry symmetry;
    if (comm_size > 0)
        symmetry = tmpmtxs[0].symmetry;
    for (int p = 1; p < comm_size; p++) {
        if (symmetry != tmpmtxs[p].symmetry) {
            for (int p = 0; p < comm_size; p++)
                mtx_free(&tmpmtxs[p]);
            free(tmpmtxs);
            return MTX_ERR_INVALID_MTX_SYMMETRY;
        }
    }

    /* Get the number of comment lines. */
    int num_comment_lines;
    if (comm_size > 0)
        num_comment_lines = tmpmtxs[0].num_comment_lines;
    for (int p = 1; p < comm_size; p++) {
        if (tmpmtxs[0].num_comment_lines != tmpmtxs[p].num_comment_lines) {
            num_comment_lines += tmpmtxs[p].num_comment_lines;
            continue;
        }

        for (int i = 0; i < tmpmtxs[p].num_comment_lines; i++) {
            int cmp = strcmp(
                tmpmtxs[0].comment_lines[i],
                tmpmtxs[p].comment_lines[i]);
            if (cmp != 0) {
                num_comment_lines += tmpmtxs[p].num_comment_lines;
                break;
            }
        }
    }

    /* Allocate storage for comment lines. */
    char ** comment_lines;
    comment_lines = malloc(num_comment_lines * sizeof(char *));
    if (!comment_lines) {
        for (int p = 0; p < comm_size; p++)
            mtx_free(&tmpmtxs[p]);
        free(tmpmtxs);
        return MTX_ERR_ERRNO;
    }

    /* Get the comment lines. */
    if (comm_size > 0) {
        for (int i = 0; i < tmpmtxs[0].num_comment_lines; i++)
            comment_lines[i] = strdup(tmpmtxs[0].comment_lines[i]);
        num_comment_lines = tmpmtxs[0].num_comment_lines;
    }
    for (int p = 1; p < comm_size; p++) {
        if (tmpmtxs[0].num_comment_lines != tmpmtxs[p].num_comment_lines) {
            for (int i = 0; i < tmpmtxs[p].num_comment_lines; i++)
                comment_lines[i] = strdup(tmpmtxs[p].comment_lines[i]);
            num_comment_lines += tmpmtxs[p].num_comment_lines;
            continue;
        }

        for (int i = 0; i < tmpmtxs[p].num_comment_lines; i++) {
            int cmp = strcmp(
                tmpmtxs[0].comment_lines[i],
                tmpmtxs[p].comment_lines[i]);
            if (cmp != 0) {
                for (int j = 0; j < tmpmtxs[p].num_comment_lines; j++)
                    comment_lines[j] = strdup(tmpmtxs[p].comment_lines[j]);
                num_comment_lines += tmpmtxs[p].num_comment_lines;
                break;
            }
        }
    }

    /* Get the number of rows. */
    int num_rows;
    if (comm_size > 0)
        num_rows = tmpmtxs[0].num_rows;
    for (int p = 1; p < comm_size; p++) {
        if (num_rows != tmpmtxs[p].num_rows) {
            for (int i = 0; i < num_comment_lines; i++)
                free(comment_lines[i]);
            free(comment_lines);
            for (int p = 0; p < comm_size; p++)
                mtx_free(&tmpmtxs[p]);
            free(tmpmtxs);
            return MTX_ERR_INVALID_MTX_SIZE;
        }
    }

    /* Get the number of columns. */
    int num_columns;
    if (comm_size > 0)
        num_columns = tmpmtxs[0].num_columns;
    for (int p = 1; p < comm_size; p++) {
        if (num_columns != tmpmtxs[p].num_columns) {
            for (int i = 0; i < num_comment_lines; i++)
                free(comment_lines[i]);
            free(comment_lines);
            for (int p = 0; p < comm_size; p++)
                mtx_free(&tmpmtxs[p]);
            free(tmpmtxs);
            return MTX_ERR_INVALID_MTX_SIZE;
        }
    }

    /* Get the number of stored nonzeros. */
    int64_t num_nonzeros = 0;
    for (int p = 0; p < comm_size; p++)
        num_nonzeros += tmpmtxs[p].num_nonzeros;

    /* Get the precision. */
    enum mtxprecision precision;
    if (comm_size > 0)
        precision = tmpmtxs[0].storage.matrix_coordinate.precision;
    for (int p = 1; p < comm_size; p++) {
        if (precision != tmpmtxs[p].storage.matrix_coordinate.precision) {
            for (int p = 0; p < comm_size; p++)
                mtx_free(&tmpmtxs[p]);
            free(tmpmtxs);
            return MTX_ERR_INVALID_PRECISION;
        }
    }

    /* Allocate storage for the gathered matrix. */
    err = mtx_alloc_matrix_coordinate(
        dstmtx, field, precision, symmetry,
        num_comment_lines, (const char **) comment_lines,
        num_rows, num_columns, num_nonzeros);
    if (err) {
        for (int i = 0; i < num_comment_lines; i++)
            free(comment_lines[i]);
        free(comment_lines);
        for (int p = 0; p < comm_size; p++)
            mtx_free(&tmpmtxs[p]);
        free(tmpmtxs);
        return err;
    }
    for (int i = 0; i < num_comment_lines; i++)
        free(comment_lines[i]);
    free(comment_lines);

    /* Copy the matrix nonzeros. */
    if (field == mtx_real) {
        if (precision == mtx_single) {
            struct mtx_matrix_coordinate_real_single * dstdata =
                dstmtx->storage.matrix_coordinate.data.real_single;
            int64_t k = 0;
            for (int p = 0; p < comm_size; p++) {
                int64_t size = tmpmtxs[p].storage.matrix_coordinate.size;
                const struct mtx_matrix_coordinate_real_single * srcdata =
                    tmpmtxs[p].storage.matrix_coordinate.data.real_single;
                memcpy(&dstdata[k], srcdata, size * sizeof(*srcdata));
                k += size;
            }
        } else if (precision == mtx_double) {
            struct mtx_matrix_coordinate_real_double * dstdata =
                dstmtx->storage.matrix_coordinate.data.real_double;
            int64_t k = 0;
            for (int p = 0; p < comm_size; p++) {
                int64_t size = tmpmtxs[p].storage.matrix_coordinate.size;
                const struct mtx_matrix_coordinate_real_double * srcdata =
                    tmpmtxs[p].storage.matrix_coordinate.data.real_double;
                memcpy(&dstdata[k], srcdata, size * sizeof(*srcdata));
                k += size;
            }
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (field == mtx_complex) {
        if (precision == mtx_single) {
            struct mtx_matrix_coordinate_complex_single * dstdata =
                dstmtx->storage.matrix_coordinate.data.complex_single;
            int64_t k = 0;
            for (int p = 0; p < comm_size; p++) {
                int64_t size = tmpmtxs[p].storage.matrix_coordinate.size;
                const struct mtx_matrix_coordinate_complex_single * srcdata =
                    tmpmtxs[p].storage.matrix_coordinate.data.complex_single;
                memcpy(&dstdata[k], srcdata, size * sizeof(*srcdata));
                k += size;
            }
        } else if (precision == mtx_double) {
            struct mtx_matrix_coordinate_complex_double * dstdata =
                dstmtx->storage.matrix_coordinate.data.complex_double;
            int64_t k = 0;
            for (int p = 0; p < comm_size; p++) {
                int64_t size = tmpmtxs[p].storage.matrix_coordinate.size;
                const struct mtx_matrix_coordinate_complex_double * srcdata =
                    tmpmtxs[p].storage.matrix_coordinate.data.complex_double;
                memcpy(&dstdata[k], srcdata, size * sizeof(*srcdata));
                k += size;
            }
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (field == mtx_integer) {
        if (precision == mtx_single) {
            struct mtx_matrix_coordinate_integer_single * dstdata =
                dstmtx->storage.matrix_coordinate.data.integer_single;
            int64_t k = 0;
            for (int p = 0; p < comm_size; p++) {
                int64_t size = tmpmtxs[p].storage.matrix_coordinate.size;
                const struct mtx_matrix_coordinate_integer_single * srcdata =
                    tmpmtxs[p].storage.matrix_coordinate.data.integer_single;
                memcpy(&dstdata[k], srcdata, size * sizeof(*srcdata));
                k += size;
            }
        } else if (precision == mtx_double) {
            struct mtx_matrix_coordinate_integer_double * dstdata =
                dstmtx->storage.matrix_coordinate.data.integer_double;
            int64_t k = 0;
            for (int p = 0; p < comm_size; p++) {
                int64_t size = tmpmtxs[p].storage.matrix_coordinate.size;
                const struct mtx_matrix_coordinate_integer_double * srcdata =
                    tmpmtxs[p].storage.matrix_coordinate.data.integer_double;
                memcpy(&dstdata[k], srcdata, size * sizeof(*srcdata));
                k += size;
            }
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (field == mtx_pattern) {
          struct mtx_matrix_coordinate_pattern * dstdata =
              dstmtx->storage.matrix_coordinate.data.pattern;
          int64_t k = 0;
          for (int p = 0; p < comm_size; p++) {
              int64_t size = tmpmtxs[p].storage.matrix_coordinate.size;
              const struct mtx_matrix_coordinate_pattern * srcdata =
                  tmpmtxs[p].storage.matrix_coordinate.data.pattern;
              memcpy(&dstdata[k], srcdata, size * sizeof(*srcdata));
              k += size;
          }
    } else {
        return MTX_ERR_INVALID_MTX_FIELD;
    }

    /* Free the temporarily gathered matrices. */
    for (int p = 0; p < comm_size; p++)
        mtx_free(&tmpmtxs[p]);
    free(tmpmtxs);
    return MTX_SUCCESS;
}

/**
 * `mtx_matrix_coordinate_scatter()` scatters a Matrix Market
 * object representing a sparse (coordinate) matrix from a root
 * process to a group of MPI processes.
 */
int mtx_matrix_coordinate_scatter(
    struct mtx * dstmtx,
    const struct mtx * srcmtx,
    const struct mtxidxset * row_sets,
    const struct mtxidxset * column_sets,
    MPI_Comm comm,
    int root,
    int * mpierrcode)
{
    int err;

    /* Get the size of the MPI communicator. */
    int comm_size;
    *mpierrcode = MPI_Comm_size(comm, &comm_size);
    if (*mpierrcode)
        return MTX_ERR_MPI;

    /* Get the MPI rank of the current process. */
    int rank;
    *mpierrcode = MPI_Comm_rank(comm, &rank);
    if (*mpierrcode)
        return MTX_ERR_MPI;

    /* Scatter submatrices to each MPI process. */
    for (int p = 0; p < comm_size; p++) {
        /* Obtain the submatrix for the current MPI process. */
        if (rank == root) {
            if (p == root) {
                /* For the root process, there is no need for any
                 * communication; just fetch the submatrix. */
                err = mtx_matrix_submatrix(dstmtx, srcmtx, &row_sets[p], &column_sets[p]);
                if (err)
                    return err;

            } else {
                /* Otherwise, fetch the submatrix and send it to the
                 * MPI process that will own it. */
                struct mtx submtx;
                err = mtx_matrix_submatrix(&submtx, srcmtx, &row_sets[p], &column_sets[p]);
                if (err)
                    return err;

                /* Send the submatrix to the MPI process that will own it. */
                err = mtx_send(&submtx, p, 0, comm, mpierrcode);
                if (err) {
                    mtx_free(&submtx);
                    return err;
                }

                /* Free the submatrix. */
                mtx_free(&submtx);
            }

        } else {
            /* Receive the submatrix at the MPI process that will own it. */
            if (p == rank) {
                err = mtx_recv(dstmtx, root, 0, comm, mpierrcode);
                if (err)
                    return err;
            }
        }
    }
    return MTX_SUCCESS;
}

#endif
