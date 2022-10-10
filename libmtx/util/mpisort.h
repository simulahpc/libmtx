/* This file is part of Libmtx.
 *
 * Copyright (C) 2022 James D. Trotter
 *
 * Libmtx is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Libmtx is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Libmtx.  If not, see <https://www.gnu.org/licenses/>.
 *
 * Authors: James D. Trotter <james@simula.no>
 * Last modified: 2022-10-08
 *
 * distributed-memory sorting with MPI.
 */

#ifndef LIBMTX_UTIL_MPISORT_H
#define LIBMTX_UTIL_MPISORT_H

#include <libmtx/libmtx-config.h>

#ifdef LIBMTX_HAVE_MPI
#include <mpi.h>

#include <stdint.h>

struct mtxdisterror;

/*
 * distributed-memory radix sort
 */

/**
 * ‘distradix_sort_uint32()’ sorts a distributed array of 32-bit
 * unsigned integers in ascending order using a distributed radix sort
 * algorithm.
 *
 * The number of keys on the current process that need to be sorted is
 * given by ‘size’, and the unsorted, integer keys on the current
 * process are given in the array ‘keys’. On success, the same array
 * will contain ‘size’ keys in a globally sorted order.
 *
 * If ‘perm’ is ‘NULL’, then this argument is ignored
 * and a sorting permutation is not computed. Otherwise, it must point
 * to an array that holds enough storage for ‘size’ values of type
 * ‘int64_t’ on each MPI process. On success, this array will contain
 * the sorting permutation, mapping the locations of the original,
 * unsorted keys to their new locations in the sorted array.
 */
int distradix_sort_uint32(
    int64_t size,
    uint32_t * keys,
    int64_t * perm,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘distradix_sort_uint64()’ sorts a distributed array of 64-bit
 * unsigned integers in ascending order using a distributed radix sort
 * algorithm.
 *
 * The number of keys on the current process that need to be sorted is
 * given by ‘size’, and the unsorted, integer keys on the current
 * process are given in the array ‘keys’. On success, the same array
 * will contain ‘size’ keys in a globally sorted order.
 *
 * If ‘perm’ is ‘NULL’, then this argument is ignored and a sorting
 * permutation is not computed. Otherwise, it must point to an array
 * that holds enough storage for ‘size’ values of type ‘int64_t’ on
 * each MPI process. On success, this array will contain the sorting
 * permutation, mapping the locations of the original, unsorted keys
 * to their new locations in the sorted array.
 */
int distradix_sort_uint64(
    int64_t size,
    uint64_t * keys,
    int64_t * perm,
    MPI_Comm comm,
    struct mtxdisterror * disterr);
#endif
#endif
