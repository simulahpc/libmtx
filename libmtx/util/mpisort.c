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
 * Last modified: 2022-05-24
 *
 * Sorting.
 */

#include <libmtx/libmtx-config.h>

#include "mpisort.h"
#include <libmtx/error.h>

#ifdef LIBMTX_HAVE_MPI
#include <mpi.h>

#include <errno.h>

#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    struct mtxdisterror * disterr)
{
    int err;

    /* The current implementation can only sort at most ‘INT_MAX’ keys
     * on each process. */
    if (size > INT_MAX) errno = ERANGE;
    err = size > INT_MAX ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;

    int comm_size;
    disterr->mpierrcode = MPI_Comm_size(comm, &comm_size);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;
    int rank;
    disterr->mpierrcode = MPI_Comm_rank(comm, &rank);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;

    int64_t total_size;
    disterr->mpierrcode = MPI_Allreduce(
        &size, &total_size, 1, MPI_INT64_T, MPI_SUM, comm);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;

    int64_t global_offset = 0;
    disterr->mpierrcode = MPI_Exscan(
        &size, &global_offset, 1, MPI_INT64_T, MPI_SUM, comm);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;

    uint32_t * extra_keys = malloc(size * sizeof(uint32_t));
    err = !extra_keys ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;

    int64_t * bucketptrs = malloc(comm_size * 257 * sizeof(int64_t));
    err = !bucketptrs ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(extra_keys);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    int64_t * bucketptr = &bucketptrs[rank*257];

    int64_t * extra_perm = NULL;
    if (perm) {
        extra_perm = malloc(size * sizeof(int64_t));
        err = !extra_perm ? MTX_ERR_ERRNO : MTX_SUCCESS;
        if (mtxdisterror_allreduce(disterr, err)) {
            free(bucketptrs);
            free(extra_keys);
            return MTX_ERR_MPI_COLLECTIVE;
        }
    }

    int * sendrecvbufs = malloc(comm_size * 5 * sizeof(int));
    err = !sendrecvbufs ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        if (extra_perm)
            free(extra_perm);
        free(bucketptrs);
        free(extra_keys);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    int * sendcounts = &sendrecvbufs[0*comm_size];
    int * senddisps  = &sendrecvbufs[1*comm_size];
    int * recvcounts = &sendrecvbufs[2*comm_size];
    int * recvdisps  = &sendrecvbufs[3*comm_size];
    int * sizes      = &sendrecvbufs[4*comm_size];

    /* Perform one round of sorting for each digit in a key */
    for (int k = 0; k < 4; k++) {
#ifdef DEBUG_DISTRADIXSORT
        for (int p = 0; p < comm_size; p++) {
            if (rank == p) {
                fprintf(stderr, "before round %d, process %d, keys=[", k, rank);
                for (int64_t i = 0; i < size; i++)
                    fprintf(stderr, " %lld (0x%02x)", keys[i], (keys[i] >> (8*k)) & 0xff);
                fprintf(stderr, "]\n");
            }
            MPI_Barrier(comm);
        }
#endif

        /* 1. Count the number of keys in each bucket. */
        for (int j = 0; j <= 256; j++)
            bucketptr[j] = 0;
        for (int64_t i = 0; i < size; i++)
            bucketptr[((keys[i] >> (8*k)) & 0xff)+1]++;

        /* 2. Compute offset to first key in each bucket. */
        for (int j = 0; j < 256; j++)
            bucketptr[j+1] += bucketptr[j];

        /* 3. Sort the keys into their respective buckets. */
        if (perm && k == 0) {
            for (int64_t i = 0; i < size; i++) {
                int64_t destidx = bucketptr[((keys[i] >> (8*k)) & 0xff)]++;
                extra_keys[destidx] = keys[i];
                extra_perm[destidx] = global_offset+i;
            }
        } else if (perm) {
            for (int64_t i = 0; i < size; i++) {
                int64_t destidx = bucketptr[((keys[i] >> (8*k)) & 0xff)]++;
                extra_keys[destidx] = keys[i];
                extra_perm[destidx] = perm[i];
            }
        } else {
            for (int64_t i = 0; i < size; i++) {
                int64_t destidx = bucketptr[((keys[i] >> (8*k)) & 0xff)]++;
                extra_keys[destidx] = keys[i];
            }
        }

        /* Adjust the offsets to each bucket. */
        for (int j = 256; j > 0; j--)
            bucketptr[j] = bucketptr[j-1];
        bucketptr[0] = 0;

#ifdef DEBUG_DISTRADIXSORT
        for (int p = 0; p < comm_size; p++) {
            if (rank == p) {
                fprintf(stderr, "middle of round %d, process %d, extra_keys=[", k, rank);
                for (int64_t i = 0; i < size; i++)
                    fprintf(stderr, " %lld (0x%02x)", extra_keys[i], (extra_keys[i] >> (8*k)) & 0xff);
                fprintf(stderr, "], extra_perm=[");
                for (int64_t i = 0; i < size; i++)
                    fprintf(stderr, " %2lld", extra_perm[i]);
                fprintf(stderr, "]\n");
            }
            MPI_Barrier(comm);
        }
#endif

#ifdef DEBUG_DISTRADIXSORT
        for (int p = 0; p < comm_size; p++) {
            if (rank == p) {
                fprintf(stderr, "round %d, process %d, bucketptr=[", k, p);
                for (int j = 0; j <= 256; j++)
                    fprintf(stderr, " %lld", bucketptr[j]);
                fprintf(stderr, "]\n");
            }
            MPI_Barrier(comm);
        }
#endif

        /* 5. Gather all the bucket pointers onto every process. */
        disterr->mpierrcode = MPI_Allgather(
            MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, bucketptrs, 257, MPI_INT64_T, comm);
        err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
        if (mtxdisterror_allreduce(disterr, err)) {
            free(sendrecvbufs);
            if (perm)
                free(extra_perm);
            free(bucketptrs);
            free(extra_keys);
            return MTX_ERR_MPI_COLLECTIVE;
        }

        /* Calculate the number of keys held by each process. */
        for (int p = 0; p < comm_size; p++) {
            sendcounts[p] = 0;
            recvcounts[p] = 0;
            sizes[p] = bucketptrs[(p+1)*257-1] - bucketptrs[p*257];
        }

        /* Calculate the number of keys in each bucket. */
        for (int p = 0; p < comm_size; p++) {
            for (int j = 255; j >= 0; j--)
                bucketptrs[p*257+j+1] -= bucketptrs[p*257+j];
        }

        /* Find the keys to send and receive for each process. */
        err = MTX_SUCCESS;
        int q = 0;
        for (int j = 0; j <= 256; j++) {
            for (int p = 0; p < comm_size; p++) {
                if (q >= comm_size) {
                    err = MTX_ERR_INDEX_OUT_OF_BOUNDS;
                    break;
                }

                int count =
                    sizes[q] < bucketptrs[p*257+j]
                    ? sizes[q] : bucketptrs[p*257+j];
                sizes[q] -= count;
                bucketptrs[p*257+j] -= count;

                if (rank == p)
                    sendcounts[q] += count;
                if (rank == q)
                    recvcounts[p] += count;

#ifdef DEBUG_DISTRADIXSORT
                if (count > 0 && rank == 0) {
                    fprintf(stderr, "round %d, process %d sends %d key(s) "
                            "from bucket %d to process %d "
                            "(process %d has %d more keys in this bucket, "
                            "process %d has room for %d more keys)\n",
                            k, p, count, j, q, p, bucketptrs[p*257+j],
                            q, sizes[q]);
                }
#endif

                if (err || (sizes[q] == 0 && q == comm_size-1))
                    break;
                else if (sizes[q] == 0) {
                    q++;
                    p--;
                    continue;
                }
            }
            if (sizes[q] == 0 && q == comm_size-1)
                break;
        }
        if (mtxdisterror_allreduce(disterr, err)) {
            free(sendrecvbufs);
            if (perm)
                free(extra_perm);
            free(bucketptrs);
            free(extra_keys);
            return MTX_ERR_MPI_COLLECTIVE;
        }

        senddisps[0] = 0;
        recvdisps[0] = 0;
        for (int p = 1; p < comm_size; p++) {
            senddisps[p] = senddisps[p-1] + sendcounts[p-1];
            recvdisps[p] = recvdisps[p-1] + recvcounts[p-1];
        }

#ifdef DEBUG_DISTRADIXSORT
        for (int q = 0; q < comm_size; q++) {
            if (rank == q) {
                fprintf(stderr, "round %d, process %d sends ", k, q);
                for (int p = 0; p < comm_size; p++) {
                    fprintf(stderr, "[");
                    for (int j = senddisps[p]; j < senddisps[p+1]; j++)
                        fprintf(stderr, " %d", extra_keys[j]);
                    fprintf(stderr, "] to process %d, ", p);
                }
                fprintf(stderr, "\n");
            }
            MPI_Barrier(comm);
        }
#endif

        /* 6. Redistribute keys among processes. */
        disterr->mpierrcode = MPI_Alltoallv(
            extra_keys, sendcounts, senddisps, MPI_INT32_T,
            keys, recvcounts, recvdisps, MPI_INT32_T, comm);
        err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
        if (mtxdisterror_allreduce(disterr, err)) {
            free(sendrecvbufs);
            if (perm)
                free(extra_perm);
            free(bucketptrs);
            free(extra_keys);
            return MTX_ERR_MPI_COLLECTIVE;
        }

        /* Also, redistribute the sorting permutation. */
        if (perm) {
            disterr->mpierrcode = MPI_Alltoallv(
                extra_perm, sendcounts, senddisps, MPI_INT64_T,
                perm, recvcounts, recvdisps, MPI_INT64_T, comm);
            err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
            if (mtxdisterror_allreduce(disterr, err)) {
                free(sendrecvbufs);
                if (perm)
                    free(extra_perm);
                free(bucketptrs);
                free(extra_keys);
                return MTX_ERR_MPI_COLLECTIVE;
            }
        }

#ifdef DEBUG_DISTRADIXSORT
        for (int p = 0; p < comm_size; p++) {
            if (rank == p) {
                fprintf(stderr, "after redistribution in round %d, process %d, keys=[", k, rank);
                for (int64_t i = 0; i < size; i++)
                    fprintf(stderr, " %lld (0x%02x)", keys[i], (keys[i] >> (8*k)) & 0xff);
                fprintf(stderr, "], perm=[");
                for (int64_t i = 0; i < size; i++)
                    fprintf(stderr, " %2lld", perm[i]);
                fprintf(stderr, "]\n");
            }
            MPI_Barrier(comm);
        }
#endif

        /* 1. Count the number of keys in each bucket. */
        for (int j = 0; j <= 256; j++)
            bucketptr[j] = 0;
        for (int64_t i = 0; i < size; i++)
            bucketptr[((keys[i] >> (8*k)) & 0xff)+1]++;

        /* 2. Compute offset to first key in each bucket. */
        for (int j = 0; j < 256; j++)
            bucketptr[j+1] += bucketptr[j];

        /* 3. Sort the keys into their respective buckets. */
        if (perm) {
            for (int64_t i = 0; i < size; i++) {
                int64_t destidx = bucketptr[((keys[i] >> (8*k)) & 0xff)]++;
                extra_keys[destidx] = keys[i];
                extra_perm[destidx] = perm[i];
            }
        } else {
            for (int64_t i = 0; i < size; i++) {
                int64_t destidx = bucketptr[((keys[i] >> (8*k)) & 0xff)]++;
                extra_keys[destidx] = keys[i];
            }
        }

        /* 4. Copy data needed for the next round. */
        if (perm) {
            for (int64_t j = 0; j < size; j++) {
                keys[j] = extra_keys[j];
                perm[j] = extra_perm[j];
            }
        } else {
            for (int64_t j = 0; j < size; j++)
                keys[j] = extra_keys[j];
        }

#ifdef DEBUG_DISTRADIXSORT
        for (int p = 0; p < comm_size; p++) {
            if (rank == p) {
                fprintf(stderr, "after round %d, process %d, keys=[", k, rank);
                for (int64_t i = 0; i < size; i++)
                    fprintf(stderr, " %lld (0x%02x)", keys[i], (keys[i] >> (8*k)) & 0xff);
                fprintf(stderr, "], perm=[");
                for (int64_t i = 0; i < size; i++)
                    fprintf(stderr, " %2lld", perm[i]);
                fprintf(stderr, "]\n");
            }
            MPI_Barrier(comm);
        }
#endif
    }

    free(sendrecvbufs);
    free(bucketptrs);
    free(extra_keys);

    /* Invert the sorting permutation */
    if (perm) {
        int64_t * global_offsets = malloc((comm_size+1) * sizeof(int64_t));
        err = !global_offsets ? MTX_ERR_ERRNO : MTX_SUCCESS;
        if (mtxdisterror_allreduce(disterr, err)) {
            free(extra_perm);
            return MTX_ERR_MPI_COLLECTIVE;
        }

        global_offsets[rank] = global_offset;
        disterr->mpierrcode = MPI_Allgather(
            MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, global_offsets, 1, MPI_INT64_T, comm);
        err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
        if (mtxdisterror_allreduce(disterr, err)) {
            free(global_offsets);
            free(extra_perm);
            return MTX_ERR_MPI_COLLECTIVE;
        }
        global_offsets[comm_size] = total_size;

        MPI_Win window;
        disterr->mpierrcode = MPI_Win_create(
            perm, size * sizeof(int64_t),
            sizeof(int64_t), MPI_INFO_NULL, comm, &window);
        err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
        if (mtxdisterror_allreduce(disterr, err)) {
            free(global_offsets);
            free(extra_perm);
            return MTX_ERR_MPI_COLLECTIVE;
        }
        MPI_Win_fence(0, window);

#ifdef DEBUG_DISTRADIXSORT
        for (int p = 0; p < comm_size; p++) {
            if (rank == p) {
                fprintf(stderr, "process %d, global_offsets=[", rank);
                for (int q = 0; q < comm_size; q++)
                    fprintf(stderr, " %lld", global_offsets[q]);
                fprintf(stderr, "]\n");
            }
            MPI_Barrier(comm);
        }
        for (int p = 0; p < comm_size; p++) {
            if (rank == p) {
                fprintf(stderr, "process %d, perm=[", rank);
                for (int64_t i = 0; i < size; i++)
                    fprintf(stderr, " %lld", perm[i]);
                fprintf(stderr, "]\n");
            }
            MPI_Barrier(comm);
        }
#endif

#ifdef DEBUG_DISTRADIXSORT
        for (int p = 0; p < comm_size; p++) {
            if (rank == p) {
                fprintf(stderr, "process %d, extra_perm=[", rank);
                for (int64_t i = 0; i < size; i++)
                    fprintf(stderr, " %lld", extra_perm[i]);
                fprintf(stderr, "]\n");
            }
            MPI_Barrier(comm);
        }
#endif

        err = MTX_SUCCESS;
        for (int64_t j = 0; j < size; j++) {
            int64_t destidx = global_offset + j;
            int64_t srcidx = extra_perm[j];

            int p = 0;
            while (p < comm_size && global_offsets[p+1] <= srcidx)
                p++;

#ifdef DEBUG_DISTRADIXSORT
            fprintf(stderr, "process %d put the value %lld at process %d location %lld (srcidx=%lld, global_offsets[%d]=%lld).\n",
                    rank, destidx, p, srcidx-global_offsets[p], srcidx, p, global_offsets[p]);
#endif
            if (p != rank) {
                disterr->mpierrcode = MPI_Put(
                    &destidx, 1, MPI_INT64_T, p,
                    srcidx-global_offsets[p], 1, MPI_INT64_T, window);
                err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
                if (err)
                    break;
            } else {
                perm[srcidx-global_offset] = destidx;
            }
        }
        MPI_Win_fence(0, window);
        MPI_Win_free(&window);
        free(global_offsets);
        free(extra_perm);
        if (mtxdisterror_allreduce(disterr, err))
            return MTX_ERR_MPI_COLLECTIVE;

#ifdef DEBUG_DISTRADIXSORT
        for (int p = 0; p < comm_size; p++) {
            if (rank == p) {
                fprintf(stderr, "after inverting sorting permutation, perm=[");
                for (int64_t i = 0; i < size; i++)
                    fprintf(stderr, " %2lld", perm[i]);
                fprintf(stderr, "]\n");
            }
            MPI_Barrier(comm);
        }
#endif
    }

    return MTX_SUCCESS;
}

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
 * If ‘perm’ is ‘NULL’, then this argument is ignored
 * and a sorting permutation is not computed. Otherwise, it must point
 * to an array that holds enough storage for ‘size’ values of type
 * ‘int64_t’ on each MPI process. On success, this array will contain
 * the sorting permutation, mapping the locations of the original,
 * unsorted keys to their new locations in the sorted array.
 */
int distradix_sort_uint64(
    int64_t size,
    uint64_t * keys,
    int64_t * perm,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err;

    /* The current implementation can only sort at most ‘INT_MAX’ keys
     * on each process. */
    if (size > INT_MAX) errno = ERANGE;
    err = size > INT_MAX ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;

    int comm_size;
    disterr->mpierrcode = MPI_Comm_size(comm, &comm_size);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;
    int rank;
    disterr->mpierrcode = MPI_Comm_rank(comm, &rank);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;

    int64_t total_size;
    disterr->mpierrcode = MPI_Allreduce(
        &size, &total_size, 1, MPI_INT64_T, MPI_SUM, comm);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;

    int64_t global_offset = 0;
    disterr->mpierrcode = MPI_Exscan(
        &size, &global_offset, 1, MPI_INT64_T, MPI_SUM, comm);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;

    uint64_t * extra_keys = malloc(size * sizeof(uint64_t));
    err = !extra_keys ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;

    int64_t * bucketptrs = malloc(comm_size * 257 * sizeof(int64_t));
    err = !bucketptrs ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(extra_keys);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    int64_t * bucketptr = &bucketptrs[rank*257];

    int64_t * extra_perm = NULL;
    if (perm) {
        extra_perm = malloc(size * sizeof(int64_t));
        err = !extra_perm ? MTX_ERR_ERRNO : MTX_SUCCESS;
        if (mtxdisterror_allreduce(disterr, err)) {
            free(bucketptrs);
            free(extra_keys);
            return MTX_ERR_MPI_COLLECTIVE;
        }
    }

    int * sendrecvbufs = malloc(comm_size * 5 * sizeof(int));
    err = !sendrecvbufs ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        if (extra_perm)
            free(extra_perm);
        free(bucketptrs);
        free(extra_keys);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    int * sendcounts = &sendrecvbufs[0*comm_size];
    int * senddisps  = &sendrecvbufs[1*comm_size];
    int * recvcounts = &sendrecvbufs[2*comm_size];
    int * recvdisps  = &sendrecvbufs[3*comm_size];
    int * sizes      = &sendrecvbufs[4*comm_size];

    /* Perform one round of sorting for each digit in a key */
    for (int k = 0; k < 8; k++) {
        /* 1. Count the number of keys in each bucket. */
        for (int j = 0; j <= 256; j++)
            bucketptr[j] = 0;
        for (int64_t i = 0; i < size; i++)
            bucketptr[((keys[i] >> (8*k)) & 0xff)+1]++;

        /* 2. Compute offset to first key in each bucket. */
        for (int j = 0; j < 256; j++)
            bucketptr[j+1] += bucketptr[j];

        /* 3. Sort the keys into their respective buckets. */
        if (perm && k == 0) {
            for (int64_t i = 0; i < size; i++) {
                int64_t destidx = bucketptr[((keys[i] >> (8*k)) & 0xff)]++;
                extra_keys[destidx] = keys[i];
                extra_perm[destidx] = global_offset+i;
            }
        } else if (perm) {
            for (int64_t i = 0; i < size; i++) {
                int64_t destidx = bucketptr[((keys[i] >> (8*k)) & 0xff)]++;
                extra_keys[destidx] = keys[i];
                extra_perm[destidx] = perm[i];
            }
        } else {
            for (int64_t i = 0; i < size; i++) {
                int64_t destidx = bucketptr[((keys[i] >> (8*k)) & 0xff)]++;
                extra_keys[destidx] = keys[i];
            }
        }

        /* Adjust the offsets to each bucket. */
        for (int j = 256; j > 0; j--)
            bucketptr[j] = bucketptr[j-1];
        bucketptr[0] = 0;

        /* 5. Gather all the bucket pointers onto every process. */
        disterr->mpierrcode = MPI_Allgather(
            MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, bucketptrs, 257, MPI_INT64_T, comm);
        err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
        if (mtxdisterror_allreduce(disterr, err)) {
            free(sendrecvbufs);
            if (perm)
                free(extra_perm);
            free(bucketptrs);
            free(extra_keys);
            return MTX_ERR_MPI_COLLECTIVE;
        }

        /* Calculate the number of keys held by each process. */
        for (int p = 0; p < comm_size; p++) {
            sendcounts[p] = 0;
            recvcounts[p] = 0;
            sizes[p] = bucketptrs[(p+1)*257-1] - bucketptrs[p*257];
        }

        /* Calculate the number of keys in each bucket. */
        for (int p = 0; p < comm_size; p++) {
            for (int j = 255; j >= 0; j--)
                bucketptrs[p*257+j+1] -= bucketptrs[p*257+j];
        }

        /* Find the keys to send and receive for each process. */
        err = MTX_SUCCESS;
        int q = 0;
        for (int j = 0; j <= 256; j++) {
            for (int p = 0; p < comm_size; p++) {
                if (q >= comm_size) {
                    err = MTX_ERR_INDEX_OUT_OF_BOUNDS;
                    break;
                }

                int count =
                    sizes[q] < bucketptrs[p*257+j]
                    ? sizes[q] : bucketptrs[p*257+j];
                sizes[q] -= count;
                bucketptrs[p*257+j] -= count;

                if (rank == p)
                    sendcounts[q] += count;
                if (rank == q)
                    recvcounts[p] += count;

                if (err || (sizes[q] == 0 && q == comm_size-1))
                    break;
                else if (sizes[q] == 0) {
                    q++;
                    p--;
                    continue;
                }
            }
            if (sizes[q] == 0 && q == comm_size-1)
                break;
        }
        if (mtxdisterror_allreduce(disterr, err)) {
            free(sendrecvbufs);
            if (perm)
                free(extra_perm);
            free(bucketptrs);
            free(extra_keys);
            return MTX_ERR_MPI_COLLECTIVE;
        }

        senddisps[0] = 0;
        recvdisps[0] = 0;
        for (int p = 1; p < comm_size; p++) {
            senddisps[p] = senddisps[p-1] + sendcounts[p-1];
            recvdisps[p] = recvdisps[p-1] + recvcounts[p-1];
        }

        /* 6. Redistribute keys among processes. */
        disterr->mpierrcode = MPI_Alltoallv(
            extra_keys, sendcounts, senddisps, MPI_INT64_T,
            keys, recvcounts, recvdisps, MPI_INT64_T, comm);
        err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
        if (mtxdisterror_allreduce(disterr, err)) {
            free(sendrecvbufs);
            if (perm)
                free(extra_perm);
            free(bucketptrs);
            free(extra_keys);
            return MTX_ERR_MPI_COLLECTIVE;
        }

        /* Also, redistribute the sorting permutation. */
        if (perm) {
            disterr->mpierrcode = MPI_Alltoallv(
                extra_perm, sendcounts, senddisps, MPI_INT64_T,
                perm, recvcounts, recvdisps, MPI_INT64_T, comm);
            err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
            if (mtxdisterror_allreduce(disterr, err)) {
                free(sendrecvbufs);
                if (perm)
                    free(extra_perm);
                free(bucketptrs);
                free(extra_keys);
                return MTX_ERR_MPI_COLLECTIVE;
            }
        }

        /* 1. Count the number of keys in each bucket. */
        for (int j = 0; j <= 256; j++)
            bucketptr[j] = 0;
        for (int64_t i = 0; i < size; i++)
            bucketptr[((keys[i] >> (8*k)) & 0xff)+1]++;

        /* 2. Compute offset to first key in each bucket. */
        for (int j = 0; j < 256; j++)
            bucketptr[j+1] += bucketptr[j];

        /* 3. Sort the keys into their respective buckets. */
        if (perm) {
            for (int64_t i = 0; i < size; i++) {
                int64_t destidx = bucketptr[((keys[i] >> (8*k)) & 0xff)]++;
                extra_keys[destidx] = keys[i];
                extra_perm[destidx] = perm[i];
            }
        } else {
            for (int64_t i = 0; i < size; i++) {
                int64_t destidx = bucketptr[((keys[i] >> (8*k)) & 0xff)]++;
                extra_keys[destidx] = keys[i];
            }
        }

        /* 4. Copy data needed for the next round. */
        if (perm) {
            for (int64_t j = 0; j < size; j++) {
                keys[j] = extra_keys[j];
                perm[j] = extra_perm[j];
            }
        } else {
            for (int64_t j = 0; j < size; j++)
                keys[j] = extra_keys[j];
        }
    }

    free(sendrecvbufs);
    free(bucketptrs);
    free(extra_keys);

    /* Invert the sorting permutation */
    if (perm) {
        int64_t * global_offsets = malloc((comm_size+1) * sizeof(int64_t));
        err = !global_offsets ? MTX_ERR_ERRNO : MTX_SUCCESS;
        if (mtxdisterror_allreduce(disterr, err)) {
            free(extra_perm);
            return MTX_ERR_MPI_COLLECTIVE;
        }

        global_offsets[rank] = global_offset;
        disterr->mpierrcode = MPI_Allgather(
            MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, global_offsets, 1, MPI_INT64_T, comm);
        err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
        if (mtxdisterror_allreduce(disterr, err)) {
            free(global_offsets);
            free(extra_perm);
            return MTX_ERR_MPI_COLLECTIVE;
        }
        global_offsets[comm_size] = total_size;

        MPI_Win window;
        disterr->mpierrcode = MPI_Win_create(
            perm, size * sizeof(int64_t),
            sizeof(int64_t), MPI_INFO_NULL, comm, &window);
        err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
        if (mtxdisterror_allreduce(disterr, err)) {
            free(global_offsets);
            free(extra_perm);
            return MTX_ERR_MPI_COLLECTIVE;
        }
        MPI_Win_fence(0, window);

        /*
         * Use one-sided MPI communication to write the remote values
         * for inverting the sorting permutation.
         */
        err = MTX_SUCCESS;
        for (int64_t j = 0; j < size; j++) {
            int64_t destidx = global_offset + j;
            int64_t srcidx = extra_perm[j];

            int p = 0;
            while (p < comm_size && global_offsets[p+1] <= srcidx)
                p++;

            if (p != rank) {
                disterr->mpierrcode = MPI_Put(
                    &destidx, 1, MPI_INT64_T, p,
                    srcidx-global_offsets[p], 1, MPI_INT64_T, window);
                err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
                if (err)
                    break;
            } else {
                perm[srcidx-global_offset] = destidx;
            }
        }
        MPI_Win_fence(0, window);
        MPI_Win_free(&window);
        free(global_offsets);
        free(extra_perm);
        if (mtxdisterror_allreduce(disterr, err))
            return MTX_ERR_MPI_COLLECTIVE;
    }

    return MTX_SUCCESS;
}
#endif
