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
#include <libmtx/index_set.h>
#include <libmtx/mtx/header.h>
#include <libmtx/mtx/matrix.h>
#include <libmtx/matrix/coordinate/coordinate.h>
#include <libmtx/vector/coordinate.h>
#include <libmtx/mtx/mtx.h>
#include <libmtx/mtx/mpi.h>

#include <mpi.h>

#include <errno.h>

#include <stdlib.h>
#include <string.h>

/**
 * `mtx_partitioning_str()` is a string representing the partitioning
 * type.
 */
const char * mtx_partitioning_str(
    enum mtx_partitioning partitioning)
{
    switch (partitioning) {
    case mtx_partition: return "partition";
    case mtx_cover: return "cover";
    default: return "unknown";
    }
}

/**
 * `mtx_datatype()' creates a custom MPI data type for sending or
 * receiving Matrix Market nonzero data for a given matrix or vector.
 *
 * The user is responsible for calling `MPI_Type_free()' on the
 * returned datatype.
 */
int mtx_datatype(
    const struct mtx * mtx,
    MPI_Datatype * datatype,
    int * mpierrcode)
{
    /* Configure a custom data type based on the format and field. */
    int num_elements;
    int block_lengths[4];
    MPI_Datatype element_types[4];
    MPI_Aint element_offsets[4];
    if (mtx->object == mtx_matrix) {
        if (mtx->format == mtx_coordinate) {
            if (mtx->field == mtx_real) {
                num_elements = 3;
                element_types[0] = MPI_INT;
                block_lengths[0] = 1;
                element_offsets[0] = offsetof(struct mtx_matrix_coordinate_real, i);
                element_types[1] = MPI_INT;
                block_lengths[1] = 1;
                element_offsets[1] = offsetof(struct mtx_matrix_coordinate_real, j);
                element_types[2] = MPI_FLOAT;
                block_lengths[2] = 1;
                element_offsets[2] = offsetof(struct mtx_matrix_coordinate_real, a);
            } else if (mtx->field == mtx_double) {
                num_elements = 3;
                element_types[0] = MPI_INT;
                block_lengths[0] = 1;
                element_offsets[0] = offsetof(struct mtx_matrix_coordinate_double, i);
                element_types[1] = MPI_INT;
                block_lengths[1] = 1;
                element_offsets[1] = offsetof(struct mtx_matrix_coordinate_double, j);
                element_types[2] = MPI_DOUBLE;
                block_lengths[2] = 1;
                element_offsets[2] = offsetof(struct mtx_matrix_coordinate_double, a);
            } else if (mtx->field == mtx_complex) {
                num_elements = 4;
                element_types[0] = MPI_INT;
                block_lengths[0] = 1;
                element_offsets[0] = offsetof(struct mtx_matrix_coordinate_complex, i);
                element_types[1] = MPI_INT;
                block_lengths[1] = 1;
                element_offsets[1] = offsetof(struct mtx_matrix_coordinate_complex, j);
                element_types[2] = MPI_FLOAT;
                block_lengths[2] = 1;
                element_offsets[2] = offsetof(struct mtx_matrix_coordinate_complex, a);
                element_types[3] = MPI_FLOAT;
                block_lengths[3] = 1;
                element_offsets[3] = offsetof(struct mtx_matrix_coordinate_complex, b);
            } else if (mtx->field == mtx_integer) {
                num_elements = 3;
                element_types[0] = MPI_INT;
                block_lengths[0] = 1;
                element_offsets[0] = offsetof(struct mtx_matrix_coordinate_integer, i);
                element_types[1] = MPI_INT;
                block_lengths[1] = 1;
                element_offsets[1] = offsetof(struct mtx_matrix_coordinate_integer, j);
                element_types[2] = MPI_INT;
                block_lengths[2] = 1;
                element_offsets[2] = offsetof(struct mtx_matrix_coordinate_integer, a);
            } else if (mtx->field == mtx_pattern) {
                num_elements = 2;
                element_types[0] = MPI_INT;
                block_lengths[0] = 1;
                element_offsets[0] = offsetof(struct mtx_matrix_coordinate_pattern, i);
                element_types[1] = MPI_INT;
                block_lengths[1] = 1;
                element_offsets[1] = offsetof(struct mtx_matrix_coordinate_pattern, j);
            } else {
                return MTX_ERR_INVALID_MTX_FIELD;
            }
        } else if (mtx->format == mtx_array) {
            if (mtx->field == mtx_real) {
                num_elements = 1;
                element_types[0] = MPI_FLOAT;
                block_lengths[0] = 1;
                element_offsets[0] = 0;
            } else if (mtx->field == mtx_double) {
                num_elements = 1;
                element_types[0] = MPI_DOUBLE;
                block_lengths[0] = 1;
                element_offsets[0] = 0;
            } else if (mtx->field == mtx_complex) {
                num_elements = 2;
                element_types[0] = MPI_FLOAT;
                block_lengths[0] = 1;
                element_offsets[0] = 0;
                element_types[1] = MPI_FLOAT;
                block_lengths[1] = 1;
                element_offsets[1] = sizeof(MPI_FLOAT);
            } else if (mtx->field == mtx_integer) {
                num_elements = 1;
                element_types[0] = MPI_INT;
                block_lengths[0] = 1;
                element_offsets[0] = 0;
            } else {
                return MTX_ERR_INVALID_MTX_FIELD;
            }
        } else {
            return MTX_ERR_INVALID_MTX_FORMAT;
        }
    } else if (mtx->object == mtx_vector) {
        if (mtx->format == mtx_coordinate) {
            if (mtx->field == mtx_real) {
                num_elements = 2;
                element_types[0] = MPI_INT;
                block_lengths[0] = 1;
                element_offsets[0] = offsetof(struct mtx_vector_coordinate_real, i);
                element_types[1] = MPI_FLOAT;
                block_lengths[1] = 1;
                element_offsets[1] = offsetof(struct mtx_vector_coordinate_real, a);
            } else if (mtx->field == mtx_double) {
                num_elements = 3;
                element_types[0] = MPI_INT;
                block_lengths[0] = 1;
                element_offsets[0] = offsetof(struct mtx_vector_coordinate_double, i);
                element_types[1] = MPI_DOUBLE;
                block_lengths[1] = 1;
                element_offsets[1] = offsetof(struct mtx_vector_coordinate_double, a);
            } else if (mtx->field == mtx_complex) {
                num_elements = 4;
                element_types[0] = MPI_INT;
                block_lengths[0] = 1;
                element_offsets[0] = offsetof(struct mtx_vector_coordinate_complex, i);
                element_types[1] = MPI_FLOAT;
                block_lengths[1] = 1;
                element_offsets[1] = offsetof(struct mtx_vector_coordinate_complex, a);
                element_types[2] = MPI_FLOAT;
                block_lengths[2] = 1;
                element_offsets[2] = offsetof(struct mtx_vector_coordinate_complex, b);
            } else if (mtx->field == mtx_integer) {
                num_elements = 3;
                element_types[0] = MPI_INT;
                block_lengths[0] = 1;
                element_offsets[0] = offsetof(struct mtx_vector_coordinate_integer, i);
                element_types[1] = MPI_INT;
                block_lengths[1] = 1;
                element_offsets[1] = offsetof(struct mtx_vector_coordinate_integer, a);
            } else if (mtx->field == mtx_pattern) {
                num_elements = 1;
                element_types[0] = MPI_INT;
                block_lengths[0] = 1;
                element_offsets[0] = offsetof(struct mtx_vector_coordinate_pattern, i);
            } else {
                return MTX_ERR_INVALID_MTX_FIELD;
            }
        } else if (mtx->format == mtx_array) {
            if (mtx->field == mtx_real) {
                num_elements = 1;
                element_types[0] = MPI_FLOAT;
                block_lengths[0] = 1;
                element_offsets[0] = 0;
            } else if (mtx->field == mtx_double) {
                num_elements = 1;
                element_types[0] = MPI_DOUBLE;
                block_lengths[0] = 1;
                element_offsets[0] = 0;
            } else if (mtx->field == mtx_complex) {
                num_elements = 2;
                element_types[0] = MPI_FLOAT;
                block_lengths[0] = 1;
                element_offsets[0] = 0;
                element_types[1] = MPI_FLOAT;
                block_lengths[1] = 1;
                element_offsets[1] = sizeof(MPI_FLOAT);
            } else if (mtx->field == mtx_integer) {
                num_elements = 1;
                element_types[0] = MPI_INT;
                block_lengths[0] = 1;
                element_offsets[0] = 0;
            } else {
                return MTX_ERR_INVALID_MTX_FIELD;
            }
        } else {
            return MTX_ERR_INVALID_MTX_FORMAT;
        }
    } else {
        return MTX_ERR_INVALID_MTX_OBJECT;
    }

    /* Create an MPI data type for receiving nonzero data. */
    MPI_Datatype tmp_datatype;
    *mpierrcode = MPI_Type_create_struct(
        num_elements, block_lengths, element_offsets,
        element_types, &tmp_datatype);
    if (*mpierrcode)
        return MTX_ERR_MPI;

    /* Enable sending an array of the custom data type. */
    MPI_Aint lb, extent;
    *mpierrcode = MPI_Type_get_extent(tmp_datatype, &lb, &extent);
    if (*mpierrcode) {
        MPI_Type_free(&tmp_datatype);
        return MTX_ERR_MPI;
    }
    *mpierrcode = MPI_Type_create_resized(tmp_datatype, lb, extent, datatype);
    if (*mpierrcode) {
        MPI_Type_free(&tmp_datatype);
        return MTX_ERR_MPI;
    }
    *mpierrcode = MPI_Type_commit(datatype);
    if (*mpierrcode) {
        MPI_Type_free(datatype);
        MPI_Type_free(&tmp_datatype);
        return MTX_ERR_MPI;
    }

    MPI_Type_free(&tmp_datatype);
    return MTX_SUCCESS;
}

/**
 * `mtx_send_header()' sends the header information of `struct
 * mtx' to another MPI process.
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
    *mpierrcode = MPI_Send(&mtx->triangle, 1, MPI_INT, dest, tag, comm);
    if (*mpierrcode)
        return MTX_ERR_MPI;
    *mpierrcode = MPI_Send(&mtx->sorting, 1, MPI_INT, dest, tag, comm);
    if (*mpierrcode)
        return MTX_ERR_MPI;
    *mpierrcode = MPI_Send(&mtx->ordering, 1, MPI_INT, dest, tag, comm);
    if (*mpierrcode)
        return MTX_ERR_MPI;
    *mpierrcode = MPI_Send(&mtx->assembly, 1, MPI_INT, dest, tag, comm);
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

    /* Receive the number of elements and the size each element in the
     * data array. */
    *mpierrcode = MPI_Send(
        &mtx->size, 1, MPI_INT64_T, dest, tag, comm);
    if (*mpierrcode)
        return MTX_ERR_MPI;
    *mpierrcode = MPI_Send(
        &mtx->nonzero_size, 1, MPI_INT, dest, tag, comm);
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
    /* Get the data type. */
    MPI_Datatype datatype;
    int err = mtx_datatype(mtx, &datatype, mpierrcode);
    if (err)
        return err;

    /* Send the nonzero data. */
    *mpierrcode = MPI_Send(
        mtx->data, mtx->size, datatype, dest, 0, comm);
    if (*mpierrcode) {
        MPI_Type_free(&datatype);
        return MTX_ERR_MPI;
    }

    MPI_Type_free(&datatype);
    return MTX_SUCCESS;
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
    *mpierrcode = MPI_Recv(
        &mtx->triangle, 1, MPI_INT, source, tag, comm, MPI_STATUS_IGNORE);
    if (*mpierrcode)
        return MTX_ERR_MPI;
    *mpierrcode = MPI_Recv(
        &mtx->sorting, 1, MPI_INT, source, tag, comm, MPI_STATUS_IGNORE);
    if (*mpierrcode)
        return MTX_ERR_MPI;
    *mpierrcode = MPI_Recv(
        &mtx->ordering, 1, MPI_INT, source, tag, comm, MPI_STATUS_IGNORE);
    if (*mpierrcode)
        return MTX_ERR_MPI;
    *mpierrcode = MPI_Recv(
        &mtx->assembly, 1, MPI_INT, source, tag, comm, MPI_STATUS_IGNORE);
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

    /* Receive the number of elements and the size each element in the
     * data array. */
    *mpierrcode = MPI_Recv(
        &mtx->size, 1, MPI_INT64_T, source, tag, comm, MPI_STATUS_IGNORE);
    if (*mpierrcode)
        return MTX_ERR_MPI;
    *mpierrcode = MPI_Recv(
        &mtx->nonzero_size, 1, MPI_INT, source, tag, comm, MPI_STATUS_IGNORE);
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

    /* Allocate storage for the Matrix Market data. */
    mtx->data = malloc(mtx->size * mtx->nonzero_size);
    if (!mtx->data)
        return MTX_ERR_ERRNO;

    /* Get the MPI data type. */
    MPI_Datatype datatype;
    err = mtx_datatype(mtx, &datatype, mpierrcode);
    if (err) {
        free(mtx->data);
        return err;
    }

    /* Receive the nonzero data. */
    *mpierrcode = MPI_Recv(
        mtx->data, mtx->size, datatype, source, 0, comm,
        MPI_STATUS_IGNORE);
    if (*mpierrcode) {
        MPI_Type_free(&datatype);
        free(mtx->data);
        return MTX_ERR_MPI;
    }

    MPI_Type_free(&datatype);
    return MTX_SUCCESS;
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
    *mpierrcode = MPI_Bcast(&mtx->triangle, 1, MPI_INT, root, comm);
    if (*mpierrcode)
        return MTX_ERR_MPI;
    *mpierrcode = MPI_Bcast(&mtx->sorting, 1, MPI_INT, root, comm);
    if (*mpierrcode)
        return MTX_ERR_MPI;
    *mpierrcode = MPI_Bcast(&mtx->ordering, 1, MPI_INT, root, comm);
    if (*mpierrcode)
        return MTX_ERR_MPI;
    *mpierrcode = MPI_Bcast(&mtx->assembly, 1, MPI_INT, root, comm);
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
    *mpierrcode = MPI_Bcast(
        &mtx->size, 1, MPI_INT64_T, root, comm);
    if (*mpierrcode)
        return MTX_ERR_MPI;
    *mpierrcode = MPI_Bcast(
        &mtx->nonzero_size, 1, MPI_INT, root, comm);
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
    /* Get the MPI rank of the current process. */
    int rank;
    *mpierrcode = MPI_Comm_rank(comm, &rank);
    if (*mpierrcode)
        return MTX_ERR_MPI;

    /* Allocate storage for the Matrix Market data. */
    if (rank != root) {
        mtx->data = malloc(mtx->size * mtx->nonzero_size);
        if (!mtx->data)
            return MTX_ERR_ERRNO;
    }

    /* Get the data type. */
    MPI_Datatype datatype;
    int err = mtx_datatype(mtx, &datatype, mpierrcode);
    if (err) {
        if (rank != root)
            free(mtx->data);
        return err;
    }

    /* Broadcast the data. */
    *mpierrcode = MPI_Bcast(
        mtx->data, mtx->size, datatype, root, comm);
    if (*mpierrcode) {
        MPI_Type_free(&datatype);
        if (rank != root)
            free(mtx->data);
        return MTX_ERR_MPI;
    }
    MPI_Type_free(&datatype);
    return MTX_SUCCESS;
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
    enum mtx_partitioning partitioning,
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
                err = mtx_copy(&tmpmtxs[p], srcmtx);
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
    if (comm_size > 0)
        dstmtx->object = tmpmtxs[0].object;
    for (int p = 1; p < comm_size; p++) {
        if (dstmtx->object != tmpmtxs[p].object) {
            for (int p = 0; p < comm_size; p++)
                mtx_free(&tmpmtxs[p]);
            free(tmpmtxs);
            return MTX_ERR_INVALID_MTX_OBJECT;
        }
    }

    /* Get the Matrix Market format. */
    if (comm_size > 0)
        dstmtx->format = tmpmtxs[0].format;
    for (int p = 1; p < comm_size; p++) {
        if (dstmtx->format != tmpmtxs[p].format) {
            for (int p = 0; p < comm_size; p++)
                mtx_free(&tmpmtxs[p]);
            free(tmpmtxs);
            return MTX_ERR_INVALID_MTX_FORMAT;
        }
    }

    /* Get the Matrix Market field. */
    if (comm_size > 0)
        dstmtx->field = tmpmtxs[0].field;
    for (int p = 1; p < comm_size; p++) {
        if (dstmtx->field != tmpmtxs[p].field) {
            for (int p = 0; p < comm_size; p++)
                mtx_free(&tmpmtxs[p]);
            free(tmpmtxs);
            return MTX_ERR_INVALID_MTX_FIELD;
        }
    }

    /* Get the Matrix Market symmetry. */
    if (comm_size > 0)
        dstmtx->symmetry = tmpmtxs[0].symmetry;
    for (int p = 1; p < comm_size; p++) {
        if (dstmtx->symmetry != tmpmtxs[p].symmetry) {
            for (int p = 0; p < comm_size; p++)
                mtx_free(&tmpmtxs[p]);
            free(tmpmtxs);
            return MTX_ERR_INVALID_MTX_SYMMETRY;
        }
    }

    if (comm_size > 0)
        dstmtx->triangle = tmpmtxs[0].triangle;
    for (int p = 1; p < comm_size; p++) {
        if (dstmtx->triangle != tmpmtxs[p].triangle) {
            for (int p = 0; p < comm_size; p++)
                mtx_free(&tmpmtxs[p]);
            free(tmpmtxs);
            return MTX_ERR_INVALID_MTX_TRIANGLE;
        }
    }

    /* Get the Matrix Market sorting. */
    if (dstmtx->format == mtx_array) {
        if (comm_size > 0)
            dstmtx->sorting = tmpmtxs[0].sorting;
        for (int p = 1; p < comm_size; p++) {
            if (dstmtx->sorting != tmpmtxs[p].sorting) {
                for (int p = 0; p < comm_size; p++)
                    mtx_free(&tmpmtxs[p]);
                free(tmpmtxs);
                return MTX_ERR_INVALID_MTX_SORTING;
            }
        }
    } else if (dstmtx->format == mtx_coordinate) {
        dstmtx->sorting = mtx_unsorted;
    } else {
        for (int p = 0; p < comm_size; p++)
            mtx_free(&tmpmtxs[p]);
        free(tmpmtxs);
        return MTX_ERR_INVALID_MTX_FORMAT;
    }

    /* Assume that the gathered matrix is unordered. */
    dstmtx->ordering = mtx_unordered;

    /* Get the Matrix Market assembly. */
    dstmtx->assembly = mtx_assembled;
    if (partitioning != mtx_partition) {
        dstmtx->assembly = mtx_unassembled;
    } else {
        for (int p = 1; p < comm_size; p++) {
            if (tmpmtxs[p].assembly != mtx_assembled) {
                dstmtx->assembly = mtx_unassembled;
            }
        }
    }

    /* Get the number of comment lines. */
    if (comm_size > 0)
        dstmtx->num_comment_lines = tmpmtxs[0].num_comment_lines;
    for (int p = 1; p < comm_size; p++) {
        if (tmpmtxs[0].num_comment_lines != tmpmtxs[p].num_comment_lines) {
            dstmtx->num_comment_lines += tmpmtxs[p].num_comment_lines;
            continue;
        }

        for (int i = 0; i < tmpmtxs[p].num_comment_lines; i++) {
            int cmp = strcmp(
                tmpmtxs[0].comment_lines[i],
                tmpmtxs[p].comment_lines[i]);
            if (cmp != 0) {
                dstmtx->num_comment_lines += tmpmtxs[p].num_comment_lines;
                break;
            }
        }
    }

    /* Allocate storage for comment lines. */
    dstmtx->comment_lines = malloc(dstmtx->num_comment_lines * sizeof(char *));
    if (!dstmtx->comment_lines) {
        for (int p = 0; p < comm_size; p++)
            mtx_free(&tmpmtxs[p]);
        free(tmpmtxs);
        return MTX_ERR_ERRNO;
    }

    /* Get the comment lines. */
    if (comm_size > 0) {
        for (int i = 0; i < tmpmtxs[0].num_comment_lines; i++)
            dstmtx->comment_lines[i] = strdup(tmpmtxs[0].comment_lines[i]);
        dstmtx->num_comment_lines = tmpmtxs[0].num_comment_lines;
    }
    for (int p = 1; p < comm_size; p++) {
        if (tmpmtxs[0].num_comment_lines != tmpmtxs[p].num_comment_lines) {
            for (int i = 0; i < tmpmtxs[p].num_comment_lines; i++)
                dstmtx->comment_lines[i] = strdup(tmpmtxs[p].comment_lines[i]);
            dstmtx->num_comment_lines += tmpmtxs[p].num_comment_lines;
            continue;
        }

        for (int i = 0; i < tmpmtxs[p].num_comment_lines; i++) {
            int cmp = strcmp(
                tmpmtxs[0].comment_lines[i],
                tmpmtxs[p].comment_lines[i]);
            if (cmp != 0) {
                for (int j = 0; j < tmpmtxs[p].num_comment_lines; j++)
                    dstmtx->comment_lines[j] = strdup(tmpmtxs[p].comment_lines[j]);
                dstmtx->num_comment_lines += tmpmtxs[p].num_comment_lines;
                break;
            }
        }
    }

    /* Get the number of rows. */
    if (comm_size > 0)
        dstmtx->num_rows = tmpmtxs[0].num_rows;
    for (int p = 1; p < comm_size; p++) {
        if (dstmtx->num_rows != tmpmtxs[p].num_rows) {
            for (int i = 0; i < dstmtx->num_comment_lines; i++)
                free(dstmtx->comment_lines[i]);
            free(dstmtx->comment_lines);
            for (int p = 0; p < comm_size; p++)
                mtx_free(&tmpmtxs[p]);
            free(tmpmtxs);
            return MTX_ERR_INVALID_MTX_SIZE;
        }
    }

    /* Get the number of columns. */
    if (comm_size > 0)
        dstmtx->num_columns = tmpmtxs[0].num_columns;
    for (int p = 1; p < comm_size; p++) {
        if (dstmtx->num_columns != tmpmtxs[p].num_columns) {
            for (int i = 0; i < dstmtx->num_comment_lines; i++)
                free(dstmtx->comment_lines[i]);
            free(dstmtx->comment_lines);
            for (int p = 0; p < comm_size; p++)
                mtx_free(&tmpmtxs[p]);
            free(tmpmtxs);
            return MTX_ERR_INVALID_MTX_SIZE;
        }
    }

    /* Get the number of stored nonzeros. */
    dstmtx->size = 0;
    for (int p = 0; p < comm_size; p++)
        dstmtx->size += tmpmtxs[p].size;

    /* Get the size of nonzeros. */
    if (comm_size > 0)
        dstmtx->nonzero_size = tmpmtxs[0].nonzero_size;
    for (int p = 1; p < comm_size; p++) {
        if (dstmtx->nonzero_size != tmpmtxs[p].nonzero_size) {
            for (int i = 0; i < dstmtx->num_comment_lines; i++)
                free(dstmtx->comment_lines[i]);
            free(dstmtx->comment_lines);
            for (int p = 0; p < comm_size; p++)
                mtx_free(&tmpmtxs[p]);
            free(tmpmtxs);
            return MTX_ERR_INVALID_MTX_SIZE;
        }
    }

    /* Allocate storage for matrix nonzeros. */
    dstmtx->data = malloc(dstmtx->size * dstmtx->nonzero_size);
    if (!dstmtx->data) {
        for (int i = 0; i < dstmtx->num_comment_lines; i++)
            free(dstmtx->comment_lines[i]);
        free(dstmtx->comment_lines);
        for (int p = 0; p < comm_size; p++)
            mtx_free(&tmpmtxs[p]);
        free(tmpmtxs);
        return MTX_ERR_ERRNO;
    }

    /* Get the matrix nonzeros. */
    dstmtx->size = 0;
    for (int p = 0; p < comm_size; p++) {
        memcpy(
            dstmtx->data + dstmtx->size * dstmtx->nonzero_size,
            tmpmtxs[p].data,
            tmpmtxs[p].size * tmpmtxs[p].nonzero_size);
        dstmtx->size += tmpmtxs[p].size;
    }

    /* Free the temporarily gathered matrices. */
    for (int p = 0; p < comm_size; p++)
        mtx_free(&tmpmtxs[p]);
    free(tmpmtxs);

    /* Calculate the number of nonzeros. */
    err = mtx_matrix_num_nonzeros(
        dstmtx->object, dstmtx->format, dstmtx->field, dstmtx->symmetry,
        dstmtx->num_rows, dstmtx->num_columns, dstmtx->size, dstmtx->data,
        &dstmtx->num_nonzeros);
    if (err) {
        for (int i = 0; i < dstmtx->num_comment_lines; i++)
            free(dstmtx->comment_lines[i]);
        free(dstmtx->comment_lines);
        return err;
    }

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
    const struct mtx_index_set * row_sets,
    const struct mtx_index_set * column_sets,
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
                err = mtx_matrix_submatrix(srcmtx, &row_sets[p], &column_sets[p], dstmtx);
                if (err)
                    return err;

            } else {
                /* Otherwise, fetch the submatrix and send it to the
                 * MPI process that will own it. */
                struct mtx submtx;
                err = mtx_matrix_submatrix(srcmtx, &row_sets[p], &column_sets[p], &submtx);
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
