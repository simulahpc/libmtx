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
 * Last modified: 2021-08-19
 *
 * Communicating matrices in array format between processes using MPI.
 */

#ifndef LIBMTX_MATRIX_ARRAY_MPI_H
#define LIBMTX_MATRIX_ARRAY_MPI_H

#include <libmtx/libmtx-config.h>

#ifdef LIBMTX_HAVE_MPI
#include <mpi.h>

struct mtx_matrix_array_data;

/**
 * `mtx_matrix_array_send()' sends a `struct mtx_matrix_array_data' to
 * another MPI process.
 *
 * This is analogous to `MPI_Send()' and requires the receiving
 * process to perform a matching call to `mtx_matrix_array_recv()'.
 */
int mtx_matrix_array_send(
    const struct mtx_matrix_array_data * matrix_array,
    int dest,
    int tag,
    MPI_Comm comm,
    int * mpierrcode);

/**
 * `mtx_matrix_array_recv()' receives a `struct mtx_matrix_array_data'
 * from another MPI process.
 *
 * This is analogous to `MPI_Recv()' and requires the sending process
 * to perform a matching call to `mtx_matrix_array_send()'.
 */
int mtx_matrix_array_recv(
    struct mtx_matrix_array_data * matrix_array,
    int source,
    int tag,
    MPI_Comm comm,
    int * mpierrcode);

/**
 * `mtx_matrix_array_bcast()' broadcasts a `struct
 * mtx_matrix_array_data' from an MPI root process to other processes
 * in a communicator.
 *
 * This is analogous to `MPI_Bcast()' and requires every process in
 * the communicator to perform matching calls to
 * `mtx_matrix_array_bcast()'.
 */
int mtx_matrix_array_bcast(
    struct mtx_matrix_array_data * matrix_array,
    int root,
    MPI_Comm comm,
    int * mpierrcode);
#endif

#endif
