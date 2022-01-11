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

struct mtxdisterror;
struct mtxpartition;
union mtxfiledata;

/*
 * Extracting row/column pointers and indices
 */

/**
 * ‘mtxdistfiledata_rowcolidx()’ extracts row and/or column indices
 * for a distributed matrix or vector in Matrix Market format.
 *
 * ‘rowidx’ may be ‘NULL’, in which case it is ignored. Otherwise, it
 * must point to an array of length at least equal to ‘size’.  If
 * successful, this array will contain the row index of each data line
 * according to the global numbering of rows and columns of the
 * distributed matrix or vector.
 *
 * Similarly, ‘colidx’ may be ‘NULL’, or it must point to an array of
 * length at least equal to ‘size’, which will be used to store the
 * global column index of each data line.
 *
 * Note that indexing is 1-based, meaning that rows are numbered
 * ‘1,2,...,num_rows’, whereas columns are numbered
 * ‘1,2,...,num_columns’.
 *
 * The arguments ‘rowpart’ and ‘colpart’ must partition the rows and
 * columns, respectively, of the distributed matrix or vector.
 * Therefore, ‘rowpart->size’ must be equal to ‘src->size.num_rows’,
 * and ‘colpart->size’ must be equal to ‘src->size.num_columns’. If
 * ‘rowpart’ or ‘colpart’ is ‘NULL’, then a trivial, singleton
 * partition is used for the rows or columns, respectively.
 */
int mtxdistfiledata_rowcolidx(
    const union mtxfiledata * data,
    enum mtxfileobject object,
    enum mtxfileformat format,
    enum mtxfilefield field,
    enum mtxprecision precision,
    int num_local_rows,
    int num_local_columns,
    int64_t size,
    const struct mtxpartition * rowpart,
    const struct mtxpartition * colpart,
    int * rowidx,
    int * colidx,
    MPI_Comm comm,
    int num_proc_rows,
    int num_proc_cols,
    struct mtxdisterror * disterr);

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
    struct mtxdisterror * disterr);

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
    struct mtxdisterror * disterr);

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
    struct mtxdisterror * disterr);

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
    struct mtxdisterror * disterr);
#endif

/*
 * Partitioning
 */

/**
 * ‘mtxdistfiledata_partition()’ partitions data lines according to
 * given row and column partitions.
 *
 * The array ‘parts’ must contain enough storage for ‘size’ values of
 * type ‘int’. If successful, ‘parts’ will contain the part number of
 * each data line in the partitioning.
 *
 * The partitions ‘rowpart’ or ‘colpart’ are allowed to be ‘NULL’, in
 * which case a trivial, singleton partition is used for the rows or
 * columns, respectively.
 */
int mtxdistfiledata_partition(
    const union mtxfiledata * data,
    enum mtxfileobject object,
    enum mtxfileformat format,
    enum mtxfilefield field,
    enum mtxprecision precision,
    int num_rows,
    int num_columns,
    int64_t size,
    const struct mtxpartition * rowpart,
    const struct mtxpartition * colpart,
    int * parts);

#endif
