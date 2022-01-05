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

#include <libmtx/libmtx-config.h>

#include <libmtx/mtxfile/header.h>
#include <libmtx/mtx/precision.h>

#ifdef LIBMTX_HAVE_MPI
#include <mpi.h>

#include <stdarg.h>
#include <stddef.h>

struct mtxmpierror;
struct mtx_partition;
union mtxfiledata;

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
    enum mtxfileobject object,
    enum mtxfileformat format,
    enum mtxfilefield field,
    enum mtxprecision precision,
    int num_rows,
    int num_columns,
    int64_t size,
    int64_t * perm,
    MPI_Comm comm,
    struct mtxmpierror * mpierror);

/**
 * ‘mtxdistfiledata_sort_row_major()’ sorts data lines of a
 * distributed Matrix Market file in row major order.
 *
 * Matrices and vectors in ‘array’ format are already in row major
 * order, which means that nothing is done in this case.
 */
int mtxdistfiledata_sort_row_major(
    union mtxfiledata * data,
    enum mtxfileobject object,
    enum mtxfileformat format,
    enum mtxfilefield field,
    enum mtxprecision precision,
    int num_rows,
    int num_columns,
    int64_t size,
    int64_t * sorting_permutation,
    MPI_Comm comm,
    struct mtxmpierror * mpierror);

/**
 * ‘mtxdistfiledata_sort_column_major()’ sorts data lines of a
 * distributed Matrix Market file in row major order.
 */
int mtxdistfiledata_sort_column_major(
    union mtxfiledata * data,
    enum mtxfileobject object,
    enum mtxfileformat format,
    enum mtxfilefield field,
    enum mtxprecision precision,
    int num_rows,
    int num_columns,
    int64_t size,
    int64_t * sorting_permutation,
    MPI_Comm comm,
    struct mtxmpierror * mpierror);

/**
 * ‘mtxdistfiledata_sort_morton()’ sorts data lines of a distributed
 * Matrix Market file in row major order.
 */
int mtxdistfiledata_sort_morton(
    union mtxfiledata * data,
    enum mtxfileobject object,
    enum mtxfileformat format,
    enum mtxfilefield field,
    enum mtxprecision precision,
    int num_rows,
    int num_columns,
    int64_t size,
    int64_t * sorting_permutation,
    MPI_Comm comm,
    struct mtxmpierror * mpierror);
#endif

#endif
