/* This file is part of libmtx.
 *
 * Copyright (C) 2021 James D. Trotter
 *
 * libmtx is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * libmtx is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with libmtx.  If not, see <https://www.gnu.org/licenses/>.
 *
 * Authors: James D. Trotter <james@simula.no>
 * Last modified: 2021-12-10
 *
 * Data lines in distributed Matrix Market files.
 */

#ifndef LIBMTX_MTXDISTFILE_DATA_H
#define LIBMTX_MTXDISTFILE_DATA_H

#include "config.h"

#include <libmtx/libmtx-config.h>

#include <libmtx/error.h>
#include <libmtx/mtx/precision.h>
#include <libmtx/mtxdistfile/data.h>
#include <libmtx/mtxfile/data.h>
#include <libmtx/mtxfile/header.h>
#include <libmtx/mtxfile/size.h>

#include <libmtx/util/sort.h>

#ifdef LIBMTX_HAVE_MPI
#include <mpi.h>
#endif

#include <errno.h>

#include <stdbool.h>
#include <stdlib.h>
#include <stddef.h>

/*
 * Sorting
 */

/**
 * ‘mtxdistfiledata_permute()’ permutes the order of data lines in a
 * distributed Matrix Market file according to a given permutation.
 * The data lines are redistributed among the participating processes.
 *
 * ‘size’ is the number of data lines in the ‘data’ array on the
 * current MPI process. ‘perm’ is a permutation mapping the data lines
 * held by the current process to their new global indices.
 */
int mtxdistfiledata_permute(
    union mtxfiledata * data,
    enum mtxfile_object object,
    enum mtxfile_format format,
    enum mtxfile_field field,
    enum mtxprecision precision,
    int num_rows,
    int num_columns,
    int64_t size,
    int64_t * perm,
    MPI_Comm comm,
    struct mtxmpierror * mpierror)
{
    int err;

    /* The current implementation can only sort at most ‘INT_MAX’ keys
     * on each process due to the use of ‘MPI_Alltoallv’. */
    if (size > INT_MAX) errno = ERANGE;
    err = size > INT_MAX ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;

    size_t element_size;
    err = mtxfiledata_size_per_element(
        data, object, format, field, precision, &element_size);
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    void * baseptr;
    err = mtxfiledata_dataptr(
        data, object, format, field, precision, &baseptr, 0);
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;

    int comm_size;
    mpierror->mpierrcode = MPI_Comm_size(comm, &comm_size);
    err = mpierror->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    int rank;
    mpierror->mpierrcode = MPI_Comm_rank(comm, &rank);
    err = mpierror->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;

    /* 1. Compute the total number of distributed data lines. */
    int64_t total_size;
    mpierror->mpierrcode = MPI_Allreduce(
        &size, &total_size, 1, MPI_INT64_T, MPI_SUM, comm);
    err = mpierror->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;

    /* Compute the offsets to the first data line of each process. */
    int64_t global_offset = 0;
    mpierror->mpierrcode = MPI_Exscan(
        &size, &global_offset, 1, MPI_INT64_T, MPI_SUM, comm);
    err = mpierror->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    int64_t * global_offsets = malloc((comm_size+1) * sizeof(int64_t));
    err = !global_offsets ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    global_offsets[rank] = global_offset;
    mpierror->mpierrcode = MPI_Allgather(
        MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, global_offsets, 1, MPI_INT64_T, comm);
    err = mpierror->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err)) {
        free(global_offsets);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    global_offsets[comm_size] = total_size;

    /* Perform some bounds checking on the permutation. */
    err = MTX_SUCCESS;
    for (int64_t k = 0; k < size; k++) {
        if (perm[k] <= 0 || perm[k] > total_size) {
            err = MTX_ERR_INDEX_OUT_OF_BOUNDS;
            break;
        }
    }
    if (mtxmpierror_allreduce(mpierror, err)) {
        free(global_offsets);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    /* 2. Count the number of data lines to send to each process. */
    int * sendcounts = malloc(4*comm_size * sizeof(int));
    err = !sendcounts ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err)) {
        free(global_offsets);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    int * senddispls = &sendcounts[comm_size];
    int * recvcounts = &senddispls[comm_size];
    int * recvdispls = &recvcounts[comm_size];

    for (int p = 0; p < comm_size; p++)
        sendcounts[p] = 0;
    for (int64_t k = 0; k < size; k++) {
        int64_t destidx = perm[k]-1;
        int p = 0;
        while (p < comm_size && global_offsets[p+1] <= destidx)
            p++;
        sendcounts[p]++;
    }
    senddispls[0] = 0;
    for (int p = 1; p < comm_size; p++)
        senddispls[p] = senddispls[p-1] + sendcounts[p-1];

    int64_t * srcdispls = malloc(size * sizeof(int64_t));
    err = !srcdispls ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err)) {
        free(sendcounts);
        free(global_offsets);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    int64_t * sendperm = malloc(size * sizeof(int64_t));
    err = !sendperm ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err)) {
        free(srcdispls);
        free(sendcounts);
        free(global_offsets);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    /* 3. Copy data to the send buffer. */
    for (int p = 0; p < comm_size; p++)
        sendcounts[p] = 0;
    for (int64_t k = 0; k < size; k++) {
        int64_t destidx = perm[k]-1;
        int p = 0;
        while (p < comm_size && global_offsets[p+1] <= destidx)
            p++;
        srcdispls[senddispls[p]+sendcounts[p]] = k;
        sendperm[senddispls[p]+sendcounts[p]] = perm[k];
        sendcounts[p]++;
    }
    free(global_offsets);

    union mtxfiledata senddata;
    err = mtxfiledata_alloc(
        &senddata, object, format, field, precision, size);
    if (mtxmpierror_allreduce(mpierror, err)) {
        free(sendperm);
        free(srcdispls);
        free(sendcounts);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    err = mtxfiledata_copy_gather(
        &senddata, data, object, format, field, precision,
        size, 0, srcdispls);
    if (mtxmpierror_allreduce(mpierror, err)) {
        free(sendperm);
        free(srcdispls);
        free(sendcounts);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    free(srcdispls);

    /* 4. Obtain ‘recvcounts’ from other processes. */
    mpierror->mpierrcode = MPI_Alltoall(
        sendcounts, 1, MPI_INT, recvcounts, 1, MPI_INT, comm);
    err = mpierror->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err)) {
        free(sendperm);
        free(sendcounts);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    recvdispls[0] = 0;
    for (int p = 1; p < comm_size; p++)
        recvdispls[p] = recvdispls[p-1] + recvcounts[p-1];

    /* 5. Exchange the data between processes. */
    err = mtxfiledata_alltoallv(
        &senddata, object, format, field, precision,
        0, sendcounts, senddispls,
        data, 0, recvcounts, recvdispls, comm, mpierror);
    if (err) {
        free(sendperm);
        free(sendcounts);
        return err;
    }

    mpierror->mpierrcode = MPI_Alltoallv(
        sendperm, sendcounts, senddispls, MPI_INT64_T,
        perm, recvcounts, recvdispls, MPI_INT64_T, comm);
    err = mpierror->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err)) {
        free(sendperm);
        free(sendcounts);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    free(sendcounts);

    /* 6. Permute the received data lines. */
    for (int64_t k = 0; k < size; k++)
        perm[k] -= global_offset;
    err = mtxfiledata_permute(
        data, object, format, field, precision,
        num_rows, num_columns, size, perm);
    if (mtxmpierror_allreduce(mpierror, err)) {
        free(sendperm);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    free(sendperm);
    return MTX_SUCCESS;
}

static int mtxdistfiledata_sort_keys(
    union mtxfiledata * data,
    enum mtxfile_object object,
    enum mtxfile_format format,
    enum mtxfile_field field,
    enum mtxprecision precision,
    int num_rows,
    int num_columns,
    int64_t size,
    int64_t * keys,
    int64_t * sorting_permutation,
    MPI_Comm comm,
    struct mtxmpierror * mpierror)
{
    int err;

    /* 1. Sort the keys and obtain a sorting permutation. */
    bool alloc_sorting_permutation = !sorting_permutation;
    if (alloc_sorting_permutation) {
        sorting_permutation = malloc(size * sizeof(int64_t));
        err = !sorting_permutation ? MTX_ERR_ERRNO : MTX_SUCCESS;
        if (mtxmpierror_allreduce(mpierror, err))
            return MTX_ERR_MPI_COLLECTIVE;
    }
    err = distradix_sort_uint64(
        size, keys, sorting_permutation, comm, mpierror);
    if (err) {
        if (alloc_sorting_permutation)
            free(sorting_permutation);
        return err;
    }

    /* Adjust from 0-based to 1-based indexing. */
    for (int64_t i = 0; i < size; i++)
        sorting_permutation[i]++;

    /* 2. Sort nonzeros according to the sorting permutation. */
    err = mtxdistfiledata_permute(
        data, object, format, field, precision,
        num_rows, num_columns, size, sorting_permutation,
        comm, mpierror);
    if (err) {
        if (alloc_sorting_permutation)
            free(sorting_permutation);
        return err;
    }
    if (alloc_sorting_permutation)
        free(sorting_permutation);
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistfiledata_sort_row_major()’ sorts data lines of a
 * distributed Matrix Market file in row major order.
 *
 * Matrices and vectors in ‘array’ format are already in row major
 * order, which means that nothing is done in this case.
 */
int mtxdistfiledata_sort_row_major(
    union mtxfiledata * data,
    enum mtxfile_object object,
    enum mtxfile_format format,
    enum mtxfile_field field,
    enum mtxprecision precision,
    int num_rows,
    int num_columns,
    int64_t size,
    int64_t * sorting_permutation,
    MPI_Comm comm,
    struct mtxmpierror * mpierror)
{
    int err;
    if (format == mtxfile_array) {
        if (sorting_permutation) {
            int64_t global_offset = 0;
            mpierror->mpierrcode = MPI_Exscan(
                &size, &global_offset, 1, MPI_INT64_T, MPI_SUM, comm);
            err = mpierror->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
            if (mtxmpierror_allreduce(mpierror, err))
                return MTX_ERR_MPI_COLLECTIVE;
            for (int64_t k = 0; k < size; k++)
                sorting_permutation[k] = global_offset+k+1;
        }
        return MTX_SUCCESS;
    } else if (format == mtxfile_coordinate) {
        int64_t * keys = malloc(size * sizeof(int64_t));
        err = !keys ? MTX_ERR_ERRNO : MTX_SUCCESS;
        if (mtxmpierror_allreduce(mpierror, err))
            return MTX_ERR_MPI_COLLECTIVE;
        err = mtxfiledata_sortkey_row_major(
            data, object, format, field, precision,
            num_rows, num_columns, size, keys);
        if (mtxmpierror_allreduce(mpierror, err)) {
            free(keys);
            return MTX_ERR_MPI_COLLECTIVE;
        }

        err = mtxdistfiledata_sort_keys(
            data, object, format, field, precision,
            num_rows, num_columns, size,
            keys, sorting_permutation, comm, mpierror);
        if (err) {
            free(keys);
            return err;
        }
        free(keys);
    } else {
        return MTX_ERR_INVALID_MTX_FORMAT;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistfiledata_sort_column_major()’ sorts data lines of a
 * distributed Matrix Market file in row major order.
 */
int mtxdistfiledata_sort_column_major(
    union mtxfiledata * data,
    enum mtxfile_object object,
    enum mtxfile_format format,
    enum mtxfile_field field,
    enum mtxprecision precision,
    int num_rows,
    int num_columns,
    int64_t size,
    int64_t * sorting_permutation,
    MPI_Comm comm,
    struct mtxmpierror * mpierror)
{
    int err;

    int rank;
    mpierror->mpierrcode = MPI_Comm_rank(comm, &rank);
    err = mpierror->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    int comm_size;
    mpierror->mpierrcode = MPI_Comm_size(comm, &comm_size);
    err = mpierror->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    int64_t rowoffset = num_rows;
    mpierror->mpierrcode = MPI_Exscan(
        MPI_IN_PLACE, &rowoffset, 1, MPI_INT64_T, MPI_SUM, comm);
    err = mpierror->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    if (rank == 0)
        rowoffset = 0;

    int64_t * keys = malloc(size * sizeof(int64_t));
    err = !keys ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;

    if (format == mtxfile_array) {
        for (int i = 0; i < num_rows; i++) {
            for (int j = 0; j < num_columns; j++) {
                int64_t k = i * (int64_t) num_columns + (int64_t) j;
                keys[k] = ((uint64_t) j << 32) | (i + rowoffset);
            }
        }
    } else {
        err = mtxfiledata_sortkey_column_major(
            data, object, format, field, precision,
            num_rows, num_columns, size, keys);
        if (mtxmpierror_allreduce(mpierror, err)) {
            free(keys);
            return MTX_ERR_MPI_COLLECTIVE;
        }
    }

    err = mtxdistfiledata_sort_keys(
        data, object, format, field, precision,
        num_rows, num_columns, size,
        keys, sorting_permutation, comm, mpierror);
    if (err) {
        free(keys);
        return err;
    }
    free(keys);
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistfiledata_sort_morton()’ sorts data lines of a distributed
 * Matrix Market file in row major order.
 */
int mtxdistfiledata_sort_morton(
    union mtxfiledata * data,
    enum mtxfile_object object,
    enum mtxfile_format format,
    enum mtxfile_field field,
    enum mtxprecision precision,
    int num_rows,
    int num_columns,
    int64_t size,
    int64_t * sorting_permutation,
    MPI_Comm comm,
    struct mtxmpierror * mpierror)
{
    int err;
    int64_t * keys = malloc(size * sizeof(int64_t));
    err = !keys ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    err = mtxfiledata_sortkey_morton(
        data, object, format, field, precision,
        num_rows, num_columns, size, keys);
    if (mtxmpierror_allreduce(mpierror, err)) {
        free(keys);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    err = mtxdistfiledata_sort_keys(
        data, object, format, field, precision,
        num_rows, num_columns, size,
        keys, sorting_permutation, comm, mpierror);
    if (err) {
        free(keys);
        return err;
    }
    free(keys);
    return MTX_SUCCESS;
}

#endif
