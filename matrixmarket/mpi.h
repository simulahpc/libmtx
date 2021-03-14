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
 * Last modified: 2021-06-18
 *
 * Functions for communicating objects in Matrix Market format between
 * processes using MPI.
 */

#ifndef MATRIXMARKET_MPI_H
#define MATRIXMARKET_MPI_H

#ifndef HAVE_MPI
#error "Please build with MPI support"
#else
#include <matrixmarket/header.h>
#include <matrixmarket/mtx.h>

#include <mpi.h>

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
    int * mpierrcode);

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
    int * mpierrcode);

/**
 * `mtx_recv()' receives a `struct mtx' from another MPI process.
 *
 * This is analogous to `MPI_Recv()' and requires the sending
 * process to perform a matching call to `mtx_send()'.
 */
int mtx_recv(
    struct mtx * mtx,
    int source,
    int tag,
    MPI_Comm comm,
    int * mpierrcode);

/**
 * `mtx_bcast()' broadcasts a `struct mtx' from an MPI root process to
 * other processes in a communicator.
 *
 * This is analogous to `MPI_Bcast()' and requires every process in
 * the communicator to perform matching calls to `mtx_bcast()'.
 */
int mtx_bcast(
    struct mtx * mtx,
    int root,
    MPI_Comm comm,
    int * mpierrcode);

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
    int * mpierrcode);

/**
 * `mtx_matrix_coordinate_scatter()` scatters a Matrix Market object
 * representing a sparse (coordinate) matrix from a root process to a
 * group of MPI processes.
 */
int mtx_matrix_coordinate_scatter(
    struct mtx * dstmtx,
    const struct mtx * srcmtx,
    const struct mtx_index_set * row_sets,
    const struct mtx_index_set * column_sets,
    MPI_Comm comm,
    int root,
    int * mpierrcode);
#endif

#endif
