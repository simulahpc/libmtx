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
 * Last modified: 2022-04-30
 *
 * Data types and functions for partitioning finite sets.
 */

#include <libmtx/error.h>
#include <libmtx/mtxfile/mtxfile.h>
#include <libmtx/util/partition.h>
#include <libmtx/util/sort.h>

#include <errno.h>
#include <unistd.h>

#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/*
 * partitioning sets of integers
 */

/**
 * ‘partition_block_int64()’ partitions elements of a set of 64-bit
 * signed integers based on a block partitioning to produce an array
 * of part numbers assigned to each element in the input array.
 *
 * The array to be partitioned, ‘idx’, contains ‘idxsize’ items.
 * Moreover, the user must provide an output array, ‘dstpart’, of size
 * ‘idxsize’, which is used to write the part number assigned to each
 * element of the input array.
 *
 * The set to be partitioned consists of ‘size’ items that are
 * partitioned into ‘num_parts’ contiguous blocks. Furthermore, the
 * array ‘partsizes’ contains ‘num_parts’ integers, specifying the
 * size of each block of the partitioned set. The sum of the entries
 * in ‘partsizes’ should therefore be equal to ‘size’.
 */
int partition_block_int64(
    int64_t size,
    int num_parts,
    const int64_t * partsizes,
    int64_t idxsize,
    int idxstride,
    const int64_t * idx,
    int * dstpart)
{
    for (int64_t i = 0; i < idxsize; i++) {
        int64_t x = *(const int64_t *) ((const unsigned char *) idx+i*idxstride);
        if (x < 0 || x >= size) return MTX_ERR_INDEX_OUT_OF_BOUNDS;
        int p = 0;
        while (p < num_parts && x >= partsizes[p])
            x -= partsizes[p++];
        if (p >= num_parts) return MTX_ERR_INDEX_OUT_OF_BOUNDS;
        dstpart[i] = p;
    }
    return MTX_SUCCESS;
}

/**
 * ‘partition_block_cyclic_int64()’ partitions elements of a set of
 * 64-bit signed integers based on a block-cyclic partitioning to
 * produce an array of part numbers assigned to each element in the
 * input array.
 *
 * The array to be partitioned, ‘idx’, contains ‘idxsize’ items.
 * Moreover, the user must provide an output array, ‘dstpart’, of size
 * ‘idxsize’, which is used to write the part number assigned to each
 * element of the input array.
 *
 * The set to be partitioned consists of ‘size’ items arranged in
 * contiguous block of size ‘blksize’, which are then partitioned in a
 * cyclic fashion into ‘num_parts’ parts.
 */
int partition_block_cyclic_int64(
    int64_t size,
    int num_parts,
    int64_t blksize,
    int64_t idxsize,
    int idxstride,
    const int64_t * idx,
    int * dstpart)
{
    for (int64_t i = 0; i < idxsize; i++) {
        int64_t x = *(const int64_t *) ((const unsigned char *) idx+i*idxstride);
        if (x < 0 || x >= size) return MTX_ERR_INDEX_OUT_OF_BOUNDS;
        dstpart[i] = (x / blksize) % num_parts;
    }
    return MTX_SUCCESS;
}

/**
 * ‘partition_custom_int64()’ partitions elements of a set of 64-bit
 * signed integers based on a user-defined partitioning to produce an
 * array of part numbers assigned to each element in the input array.
 *
 * The array to be partitioned, ‘idx’, contains ‘idxsize’ items.
 * Moreover, the user must provide an output array, ‘dstpart’, of size
 * ‘idxsize’, which is used to write the part number assigned to each
 * element of the input array.
 *
 * The set to be partitioned consists of ‘size’ items. Moreover,
 * ‘parts’ is an array of length ‘size’, which specifies the part
 * number of each element in the set.
 */
int partition_custom_int64(
    int64_t size,
    int num_parts,
    const int64_t * parts,
    int64_t idxsize,
    int idxstride,
    const int64_t * idx,
    int * dstpart)
{
    for (int64_t i = 0; i < idxsize; i++) {
        int64_t x = *(const int64_t *) ((const unsigned char *) idx+i*idxstride);
        if (x < 0 || x >= size) return MTX_ERR_INDEX_OUT_OF_BOUNDS;
        dstpart[i] = parts[x];
    }
    return MTX_SUCCESS;
}

/**
 * ‘partition_int64()’ partitions elements of a set of 64-bit signed
 * integers based on a given partitioning to produce an array of part
 * numbers assigned to each element in the input array.
 *
 * The array to be partitioned, ‘idx’, contains ‘idxsize’ items.
 * Moreover, the user must provide an output array, ‘dstpart’, of size
 * ‘idxsize’, which is used to write the part number assigned to each
 * element of the input array.
 *
 * The set to be partitioned consists of ‘size’ items.
 *
 * - If ‘type’ is ‘mtx_block’, then the array ‘partsizes’ contains
 *   ‘num_parts’ integers, specifying the size of each block of the
 *   partitioned set.
 *
 * - If ‘type’ is ‘mtx_block_cyclic’, then items are arranged in
 *   contiguous block of size ‘blksize’, which are then partitioned in
 *   a cyclic fashion into ‘num_parts’ parts.
 *
 * - If ‘type’ is ‘mtx_block_cyclic’, then ‘parts’ is an array of
 *   length ‘size’, which specifies the part number of each element in
 *   the set.
 */
int partition_int64(
    enum mtxpartitioning type,
    int64_t size,
    int num_parts,
    const int64_t * partsizes,
    int64_t blksize,
    const int64_t * parts,
    int64_t idxsize,
    int idxstride,
    const int64_t * idx,
    int * dstpart)
{
    if (type == mtx_block) {
        return partition_block_int64(
            size, num_parts, partsizes, idxsize, idxstride, idx, dstpart);
    } else if (type == mtx_block_cyclic) {
        return partition_block_cyclic_int64(
            size, num_parts, blksize, idxsize, idxstride, idx, dstpart);
    } else if (type == mtx_custom_partition) {
        return partition_custom_int64(
            size, num_parts, parts, idxsize, idxstride, idx, dstpart);
    } else { return MTX_ERR_INVALID_PARTITION_TYPE; }
}

/*
 * distributed partitioning of sets of integers
 */

#ifdef LIBMTX_HAVE_MPI
/**
 * ‘distpartition_block_int64()’ partitions elements of a set of
 * 64-bit signed integers based on a block partitioning to produce an
 * array of part numbers assigned to each element in the input array.
 *
 * The array to be partitioned, ‘idx’, contains ‘idxsize’ items.
 * Moreover, the user must provide an output array, ‘dstpart’, of size
 * ‘idxsize’, which is used to write the part number assigned to each
 * element of the input array.
 *
 * The set to be partitioned consists of ‘size’ items that are
 * partitioned into ‘P’ contiguous blocks, where ‘P’ is the number of
 * processes in the MPI communicator ‘comm’. Furthermore, ‘partsize’
 * is used to specify the size of the block on the current process.
 */
int distpartition_block_int64(
    int64_t size,
    int64_t partsize,
    int64_t idxsize,
    int idxstride,
    const int64_t * idx,
    int * dstpart,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int comm_size;
    disterr->mpierrcode = MPI_Comm_size(comm, &comm_size);
    int err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    int64_t * partsizes = malloc(comm_size * sizeof(int64_t));
    err = !partsizes ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    disterr->mpierrcode = MPI_Allgather(
        &partsize, 1, MPI_INT64_T, partsizes, 1, MPI_INT64_T, comm);
    if (mtxdisterror_allreduce(disterr, err)) {
        free(partsizes);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    err = partition_block_int64(
        size, comm_size, partsizes, idxsize, idxstride, idx, dstpart);
    if (mtxdisterror_allreduce(disterr, err)) {
        free(partsizes);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    free(partsizes);
    return MTX_SUCCESS;
}

/**
 * ‘distpartition_block_cyclic_int64()’ partitions elements of a set
 * of 64-bit signed integers based on a block-cyclic partitioning to
 * produce an array of part numbers assigned to each element in the
 * input array.
 *
 * The array to be partitioned, ‘idx’, contains ‘idxsize’ items.
 * Moreover, the user must provide an output array, ‘dstpart’, of size
 * ‘idxsize’, which is used to write the part number assigned to each
 * element of the input array.
 *
 * The partitioned set consists of ‘size’ items arranged in contiguous
 * block of size ‘blksize’, which are then partitioned in a cyclic
 * fashion into ‘P’ parts, where ‘P’ is the number of processes in the
 * MPI communicator ‘comm’.
 */
int distpartition_block_cyclic_int64(
    int64_t size,
    int64_t blksize,
    int64_t idxsize,
    int idxstride,
    const int64_t * idx,
    int * dstpart,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int comm_size;
    disterr->mpierrcode = MPI_Comm_size(comm, &comm_size);
    int err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return partition_block_cyclic_int64(
        size, comm_size, blksize, idxsize, idxstride, idx, dstpart);
}

/*
 * The “assumed partition” strategy, a parallel rendezvous protocol
 * algorithm to answer queries about the ownership and location of
 * distributed data.
 */

/**
 * ‘assumedpartition_write()’ sets up the data structures needed to
 * query about ownership and location of distributed data using the
 * assumed partition strategy.
 *
 * The method assumed an underlying one-dimensional array of ‘size’
 * elements distributed among ‘P’ processes, where ‘P’ is the size of
 * the MPI communicator ‘comm’.
 *
 * On a given process, ‘partsize’ is the number of elements it owns
 * and ‘globalidx’ is an array containing the global numbers of those
 * elements.
 *
 * The ‘ownerrank’ and ‘owneridx’ arrays are of length
 * ‘assumedpartsize’, which must be at least equal to
 * ‘(size+comm_size-1)/comm_size’. For each element that is assumed to
 * be owned by a given process (according to an equal-sized block
 * partitioning), these arrays are used to write the actual owning
 * rank, as well as the offset to the element among elements of the
 * owning rank.
 */
int assumedpartition_write(
    int64_t size,
    int64_t partsize,
    const int64_t * globalidx,
    int apsize,
    int * ownerrank,
    int * owneridx,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int64_t N = size;
    int comm_size;
    disterr->mpierrcode = MPI_Comm_size(comm, &comm_size);
    int err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    int rank;
    disterr->mpierrcode = MPI_Comm_rank(comm, &rank);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;

    int blksize = (N+comm_size-1) / comm_size;
    if (apsize < blksize) err = MTX_ERR_INDEX_OUT_OF_BOUNDS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;

    /*
     * Step 1: On each process, sort the array of global offsets,
     * which places them in ascending order according to the assumed
     * owner's rank.
     */

    int64_t * globalidxsorted = malloc(partsize * sizeof(int64_t));
    err = !globalidxsorted ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    for (int64_t i = 0; i < partsize; i++) globalidxsorted[i] = globalidx[i];
    int64_t * perm = malloc(partsize * sizeof(int64_t));
    err = !perm ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(globalidxsorted);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    err = radix_sort_int64(partsize, globalidxsorted, perm);
    if (mtxdisterror_allreduce(disterr, err)) {
        free(perm); free(globalidxsorted);
        return MTX_ERR_MPI_COLLECTIVE;
    }

#ifdef MTXDEBUG_ASSUMED_PARTITION_WRITE
    for (int p = 0; p < comm_size; p++) {
        if (rank == p) {
            fprintf(stderr, "partsize=%d, globalidx=[", partsize);
            for (int64_t i = 0; i < partsize; i++)
                fprintf(stderr, " %"PRId64, globalidx[i]);
            fprintf(stderr, "], globalidxsorted=[", partsize);
            for (int64_t i = 0; i < partsize; i++)
                fprintf(stderr, " %"PRId64, globalidxsorted[i]);
            fprintf(stderr, "], ");
            fprintf(stderr, "perm=[");
            for (int64_t i = 0; i < partsize; i++)
                fprintf(stderr, " %"PRId64, perm[i]);
            fprintf(stderr, "]\n");
        }
        MPI_Barrier(comm); sleep(1);
    }
#endif

    /*
     * Step 2: Compute the assumed owner and assumed offset of each
     * element that is owned by the current process.
     */

    int * assumedrank = malloc(partsize * sizeof(int));
    err = !assumedrank ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    int * assumedidx = malloc(partsize * sizeof(int));
    err = !assumedidx ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(assumedrank);
        free(perm); free(globalidxsorted);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    for (int64_t i = 0; i < partsize; i++) {
        assumedrank[i] = globalidxsorted[i] / blksize;
        assumedidx[i] = globalidxsorted[i] % blksize;
    }
    free(globalidxsorted);

#ifdef MTXDEBUG_ASSUMED_PARTITION_WRITE
    for (int p = 0; p < comm_size; p++) {
        if (rank == p) {
            fprintf(stderr, "assumedrank=[");
            for (int64_t i = 0; i < partsize; i++)
                fprintf(stderr, " %d", assumedrank[i]);
            fprintf(stderr, "], ");
            fprintf(stderr, "assumedidx=[");
            for (int64_t i = 0; i < partsize; i++)
                fprintf(stderr, " %d", assumedidx[i]);
            fprintf(stderr, "]\n");
        }
        MPI_Barrier(comm); sleep(1);
    }
#endif

    /*
     * Step 3: The owners and offsets of each element are now ready to
     * send to the assumed owners. Next, create a list of processes
     * that the current process will send ownership information to
     * and count how many entries to send to each process.
     */

    int nsendranks = partsize > 0 ? 1 : 0;
    for (int64_t i = 1; i < partsize; i++) {
        if (assumedrank[i-1] != assumedrank[i])
            nsendranks++;
    }
    int * sendranks = malloc(nsendranks * sizeof(int));
    err = !sendranks ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(assumedidx); free(assumedrank);
        free(perm);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    int * sendcounts = malloc(nsendranks * sizeof(int));
    err = !sendcounts ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(sendranks);
        free(assumedidx); free(assumedrank);
        free(perm);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    if (nsendranks > 0) {
        sendranks[0] = assumedrank[0];
        sendcounts[0] = 1;
    }
    for (int64_t i = 1, p = 0; i < partsize; i++) {
        if (assumedrank[i-1] != assumedrank[i]) {
            sendranks[++p] = assumedrank[i];
            sendcounts[p] = 0;
        }
        sendcounts[p]++;
    }
    int * sdispls = malloc((nsendranks+1) * sizeof(int));
    err = !sdispls ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(sendcounts); free(sendranks);
        free(assumedidx); free(assumedrank);
        free(perm);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    sdispls[0] = 0;
    for (int p = 0; p < nsendranks; p++)
        sdispls[p+1] = sdispls[p] + sendcounts[p];

#ifdef MTXDEBUG_ASSUMED_PARTITION_WRITE
    for (int p = 0; p < comm_size; p++) {
        if (rank == p) {
            fprintf(stderr, "nsendranks=%d, sendranks=[", nsendranks);
            for (int64_t i = 0; i < nsendranks; i++)
                fprintf(stderr, " %d", sendranks[i]);
            fprintf(stderr, "], sendcounts=[");
            for (int64_t i = 0; i < nsendranks; i++)
                fprintf(stderr, " %d", sendcounts[i]);
            fprintf(stderr, "], sdispls=[");
            for (int64_t i = 0; i <= nsendranks; i++)
                fprintf(stderr, " %d", sdispls[i]);
            fprintf(stderr, "]\n");
        }
        MPI_Barrier(comm); sleep(1);
    }
#endif

    /*
     * Step 4: Count the number of processes to receive data from.
     * This is achieved using a single counter on each process and
     * one-sided communication, so that every process can atomically
     * update remote counters on other processes, if it needs data
     * from them.
     */

    int nrecvranks = 0;
    MPI_Win window;
    disterr->mpierrcode = MPI_Win_create(
        &nrecvranks, sizeof(int), sizeof(int), MPI_INFO_NULL, comm, &window);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(sdispls); free(sendcounts); free(sendranks);
        free(assumedidx); free(assumedrank);
        free(perm);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    disterr->mpierrcode = MPI_Win_fence(0, window);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    for (int p = 0; p < nsendranks && !err; p++) {
        int one = 1;
        disterr->mpierrcode = MPI_Accumulate(
            &one, 1, MPI_INT, sendranks[p], 0, 1, MPI_INT, MPI_SUM, window);
        err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    }
    if (err) {
        MPI_Win_fence(0, window);
    } else {
        disterr->mpierrcode = MPI_Win_fence(0, window);
        err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    }
    MPI_Win_free(&window);
    if (mtxdisterror_allreduce(disterr, err)) {
        free(sdispls); free(sendcounts); free(sendranks);
        free(assumedidx); free(assumedrank);
        free(perm);
        return MTX_ERR_MPI_COLLECTIVE;
    }

#ifdef MTXDEBUG_ASSUMED_PARTITION_WRITE
    for (int p = 0; p < comm_size; p++) {
        if (rank == p) fprintf(stderr, "nrecvranks=%d\n", nrecvranks);
        MPI_Barrier(comm); sleep(1);
    }
#endif

    /*
     * Step 5: Post non-blocking, wildcard receives for every process
     * that will send data to the current process. Thereafter, the
     * current process transmits the number of elements it needs to
     * send to each process it will later send data to.
     */

    /* TODO: With some additional checks, each process should be able
     * to avoid sending to itself. */

    MPI_Request * req = malloc(nrecvranks * sizeof(MPI_Request));
    err = !req ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(sdispls); free(sendcounts); free(sendranks);
        free(assumedidx); free(assumedrank);
        free(perm);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    int * recvranks = malloc(nrecvranks * sizeof(int));
    err = !recvranks ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(req);
        free(sdispls); free(sendcounts); free(sendranks);
        free(assumedidx); free(assumedrank);
        free(perm);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    int * recvcounts = malloc(nrecvranks * sizeof(int));
    err = !recvcounts ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(recvranks); free(req);
        free(sdispls); free(sendcounts); free(sendranks);
        free(assumedidx); free(assumedrank);
        free(perm);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    for (int p = 0; p < nrecvranks; p++) {
        int mpierrcode = MPI_Irecv(
            &recvcounts[p], 1, MPI_INT, MPI_ANY_SOURCE, 0, comm, &req[p]);
        if (!err) {
            disterr->mpierrcode = mpierrcode;
            err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
        }
    }
    if (mtxdisterror_allreduce(disterr, err)) {
        free(recvcounts); free(recvranks); free(req);
        free(sdispls); free(sendcounts); free(sendranks);
        free(assumedidx); free(assumedrank);
        free(perm);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    for (int p = 0; p < nsendranks; p++) {
        int mpierrcode = MPI_Send(
            &sendcounts[p], 1, MPI_INT, sendranks[p], 0, comm);
        if (!err) {
            disterr->mpierrcode = mpierrcode;
            err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
        }
    }
    if (mtxdisterror_allreduce(disterr, err)) {
        free(recvcounts); free(recvranks); free(req);
        free(sdispls); free(sendcounts); free(sendranks);
        free(assumedidx); free(assumedrank);
        free(perm);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    for (int p = 0; p < nrecvranks; p++) {
        MPI_Status status;
        disterr->mpierrcode = MPI_Wait(&req[p], &status);
        err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
        if (err) break;
        recvranks[p] = status.MPI_SOURCE;
    }
    if (mtxdisterror_allreduce(disterr, err)) {
        free(recvcounts); free(recvranks); free(req);
        free(sdispls); free(sendcounts); free(sendranks);
        free(assumedidx); free(assumedrank);
        free(perm);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    int * rdispls = malloc((nrecvranks+1) * sizeof(int));
    err = !rdispls ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(recvcounts); free(recvranks); free(req);
        free(sdispls); free(sendcounts); free(sendranks);
        free(assumedidx); free(assumedrank);
        free(perm);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    rdispls[0] = 0;
    for (int p = 0; p < nrecvranks; p++)
        rdispls[p+1] = rdispls[p] + recvcounts[p];

#ifdef MTXDEBUG_ASSUMED_PARTITION_WRITE
    for (int p = 0; p < comm_size; p++) {
        if (rank == p) {
            fprintf(stderr, "recvranks=[");
            for (int64_t i = 0; i < nrecvranks; i++)
                fprintf(stderr, " %d", recvranks[i]);
            fprintf(stderr, "], recvcounts=[");
            for (int64_t i = 0; i < nrecvranks; i++)
                fprintf(stderr, " %d", recvcounts[i]);
            fprintf(stderr, "], rdispls=[");
            for (int64_t i = 0; i <= nrecvranks; i++)
                fprintf(stderr, " %d", rdispls[i]);
            fprintf(stderr, "]\n");
        }
        MPI_Barrier(comm); sleep(1);
    }
#endif

    /*
     * Step 6: Send the assumed offset together with the real offset
     * at the owning rank for each element.
     */

    int * assumedidxrecvbuf = malloc(rdispls[nrecvranks] * sizeof(int));
    err = !assumedidxrecvbuf ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(rdispls); free(recvcounts); free(recvranks); free(req);
        free(sdispls); free(sendcounts); free(sendranks);
        free(assumedidx); free(assumedrank);
        free(perm);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    int * owneridxrecvbuf = malloc(rdispls[nrecvranks] * sizeof(int));
    err = !owneridxrecvbuf ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(assumedidxrecvbuf);
        free(rdispls); free(recvcounts); free(recvranks); free(req);
        free(sdispls); free(sendcounts); free(sendranks);
        free(assumedidx); free(assumedrank);
        free(perm);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    int * owneridxsendbuf = malloc(sdispls[nsendranks] * sizeof(int));
    err = !owneridxsendbuf ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(owneridxrecvbuf); free(assumedidxrecvbuf);
        free(rdispls); free(recvcounts); free(recvranks); free(req);
        free(sdispls); free(sendcounts); free(sendranks);
        free(assumedidx); free(assumedrank);
        free(perm);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    for (int64_t i = 0; i < partsize; i++)
        owneridxsendbuf[perm[i]] = i;
    free(perm);

    for (int p = 0; p < nrecvranks; p++) {
        int mpierrcode = MPI_Irecv(
            &assumedidxrecvbuf[rdispls[p]], recvcounts[p], MPI_INT,
            recvranks[p], 1, comm, &req[p]);
        if (!err) {
            disterr->mpierrcode = mpierrcode;
            err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
        }
    }
    if (mtxdisterror_allreduce(disterr, err)) {
        free(owneridxsendbuf); free(owneridxrecvbuf); free(assumedidxrecvbuf);
        free(rdispls); free(recvcounts); free(recvranks); free(req);
        free(sdispls); free(sendcounts); free(sendranks);
        free(assumedidx); free(assumedrank);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    for (int p = 0; p < nsendranks; p++) {
        int mpierrcode = MPI_Send(
            &assumedidx[sdispls[p]], sendcounts[p], MPI_INT,
            sendranks[p], 1, comm);
        if (!err)  {
            disterr->mpierrcode = mpierrcode;
            err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
        }
    }
    if (mtxdisterror_allreduce(disterr, err)) {
        free(owneridxsendbuf); free(owneridxrecvbuf); free(assumedidxrecvbuf);
        free(rdispls); free(recvcounts); free(recvranks); free(req);
        free(sdispls); free(sendcounts); free(sendranks);
        free(assumedidx); free(assumedrank);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    MPI_Waitall(nrecvranks, req, MPI_STATUSES_IGNORE);

    for (int p = 0; p < nrecvranks; p++) {
        int mpierrcode = MPI_Irecv(
            &owneridxrecvbuf[rdispls[p]], recvcounts[p], MPI_INT,
            recvranks[p], 2, comm, &req[p]);
        if (!err) {
            disterr->mpierrcode = mpierrcode;
            err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
        }
    }
    if (mtxdisterror_allreduce(disterr, err)) {
        free(owneridxsendbuf); free(owneridxrecvbuf); free(assumedidxrecvbuf);
        free(rdispls); free(recvcounts); free(recvranks); free(req);
        free(sdispls); free(sendcounts); free(sendranks);
        free(assumedidx); free(assumedrank);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    for (int p = 0; p < nsendranks; p++) {
        int mpierrcode = MPI_Send(
            &owneridxsendbuf[sdispls[p]], sendcounts[p], MPI_INT,
            sendranks[p], 2, comm);
        if (!err)  {
            disterr->mpierrcode = mpierrcode;
            err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
        }
    }
    if (mtxdisterror_allreduce(disterr, err)) {
        free(owneridxsendbuf); free(owneridxrecvbuf); free(assumedidxrecvbuf);
        free(rdispls); free(recvcounts); free(recvranks); free(req);
        free(sdispls); free(sendcounts); free(sendranks);
        free(assumedidx); free(assumedrank);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    MPI_Waitall(nrecvranks, req, MPI_STATUSES_IGNORE);

#ifdef MTXDEBUG_ASSUMED_PARTITION_WRITE
    for (int p = 0; p < comm_size; p++) {
        if (rank == p) {
            fprintf(stderr, "assumedidxrecvbuf=[");
            for (int q = 0; q < nrecvranks; q++) {
                if (q > 0) fprintf(stderr, " |");
                for (int64_t i = rdispls[q]; i < rdispls[q+1]; i++)
                    fprintf(stderr, " %d", assumedidxrecvbuf[i]);
            }
            fprintf(stderr, "], owneridxsendbuf=[");
            for (int q = 0; q < nsendranks; q++) {
                if (q > 0) fprintf(stderr, " |");
                for (int64_t i = sdispls[q]; i < sdispls[q+1]; i++)
                    fprintf(stderr, " %d", owneridxsendbuf[i]);
            }
            fprintf(stderr, "], owneridxrecvbuf=[");
            for (int q = 0; q < nrecvranks; q++) {
                if (q > 0) fprintf(stderr, " |");
                for (int64_t i = rdispls[q]; i < rdispls[q+1]; i++)
                    fprintf(stderr, " %d", owneridxrecvbuf[i]);
            }
            fprintf(stderr, "]\n");
        }
        MPI_Barrier(comm); sleep(1);
    }
#endif

    /*
     * Step 7: Scatter the received data to its final destination.
     */

    for (int i = 0; i < blksize; i++) {
        ownerrank[i] = -1;
        owneridx[i] = -1;
    }

    for (int p = 0; p < nrecvranks; p++) {
        for (int i = rdispls[p]; i < rdispls[p+1]; i++) {
            ownerrank[assumedidxrecvbuf[i]] = recvranks[p];
            owneridx[assumedidxrecvbuf[i]] = owneridxrecvbuf[i];
        }
    }

#ifdef MTXDEBUG_ASSUMED_PARTITION_WRITE
    for (int p = 0; p < comm_size; p++) {
        if (rank == p) {
            fprintf(stderr, "ownerrank=[");
            for (int64_t i = 0; i < blksize; i++)
                fprintf(stderr, " %d", ownerrank[i]);
            fprintf(stderr, "], owneridx=[");
            for (int64_t i = 0; i < blksize; i++)
                fprintf(stderr, " %d", owneridx[i]);
            fprintf(stderr, "]\n");
        }
        MPI_Barrier(comm); sleep(1);
    }
#endif

    /* test for scatter conflicts */
    err = radix_sort_int(rdispls[nrecvranks], assumedidxrecvbuf, NULL);
    if (mtxdisterror_allreduce(disterr, err)) {
        free(owneridxsendbuf); free(owneridxrecvbuf); free(assumedidxrecvbuf);
        free(rdispls); free(recvcounts); free(recvranks); free(req);
        free(sdispls); free(sendcounts); free(sendranks);
        free(assumedidx); free(assumedrank);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    for (int i = 1; i < rdispls[nrecvranks]; i++) {
        if (assumedidxrecvbuf[i-1] == assumedidxrecvbuf[i]) {
            err = MTX_ERR_SCATTER_CONFLICT;
            break;
        }
    }
    if (mtxdisterror_allreduce(disterr, err)) {
        free(owneridxsendbuf); free(owneridxrecvbuf); free(assumedidxrecvbuf);
        free(rdispls); free(recvcounts); free(recvranks); free(req);
        free(sdispls); free(sendcounts); free(sendranks);
        free(assumedidx); free(assumedrank);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    free(owneridxsendbuf); free(owneridxrecvbuf); free(assumedidxrecvbuf);
    free(rdispls); free(recvcounts); free(recvranks); free(req);
    free(sdispls); free(sendcounts); free(sendranks);
    free(assumedidx); free(assumedrank);
    return MTX_SUCCESS;
}

/**
 * ‘assumedpartition_read()’ performs a query to learn the ownership
 * and location of distributed data using the assumed partition
 * strategy.
 *
 * The method assumed an underlying one-dimensional array of ‘size’
 * elements distributed among ‘P’ processes, where ‘P’ is the size of
 * the MPI communicator ‘comm’.
 *
 * The ‘apownerrank’ and ‘apowneridx’ arrays are of length ‘apsize’,
 * which must be at least equal to ‘(size+P-1)/P’. For each element
 * that is assumed to be owned by a given process (according to an
 * equal-sized block partitioning), these arrays specify the actual
 * owning rank, as well as the offset to the element among elements of
 * the owning rank. See also ‘assumedpartition_write’ for how to set
 * up these arrays.
 *
 * On a given process, ‘partsize’ is the number of elements to query
 * for ownership information and ‘globalidx’ is an array containing
 * the global numbers of those elements.
 *
 * The ‘ownerrank’ and ‘owneridx’ arrays must also be of length
 * ‘partsize’. They are used to write the actual owning rank, as well
 * as the offset to the element among elements of the owning rank.
 */
int assumedpartition_read(
    int64_t size,
    int apsize,
    const int * apownerrank,
    const int * apowneridx,
    int64_t partsize,
    const int64_t * globalidx,
    int * ownerrank,
    int * owneridx,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int64_t N = size;
    int comm_size;
    disterr->mpierrcode = MPI_Comm_size(comm, &comm_size);
    int err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    int rank;
    disterr->mpierrcode = MPI_Comm_rank(comm, &rank);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;

    int blksize = (N+comm_size-1) / comm_size;
    int64_t blkstart = rank*blksize;
    if (apsize < blksize) err = MTX_ERR_INDEX_OUT_OF_BOUNDS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;

    /*
     * Step 1: On each process, sort the array of global offsets,
     * which places them in ascending order according to the assumed
     * owner's rank.
     */

    int64_t * globalidxsorted = malloc(partsize * sizeof(int64_t));
    err = !globalidxsorted ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    for (int64_t i = 0; i < partsize; i++) globalidxsorted[i] = globalidx[i];
    int64_t * perm = malloc(partsize * sizeof(int64_t));
    err = !perm ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(globalidxsorted);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    err = radix_sort_int64(partsize, globalidxsorted, perm);
    if (mtxdisterror_allreduce(disterr, err)) {
        free(perm); free(globalidxsorted);
        return MTX_ERR_MPI_COLLECTIVE;
    }

#ifdef MTXDEBUG_ASSUMED_PARTITION_READ
    for (int p = 0; p < comm_size; p++) {
        if (rank == p) {
            fprintf(stderr, "partsize=%d, globalidx=[", partsize);
            for (int64_t i = 0; i < partsize; i++)
                fprintf(stderr, " %"PRId64, globalidx[i]);
            fprintf(stderr, "], globalidxsorted=[");
            for (int64_t i = 0; i < partsize; i++)
                fprintf(stderr, " %"PRId64, globalidxsorted[i]);
            fprintf(stderr, "], perm=[");
            for (int64_t i = 0; i < partsize; i++)
                fprintf(stderr, " %"PRId64, perm[i]);
            fprintf(stderr, "]\n");
        }
        MPI_Barrier(comm); sleep(1);
    }
#endif

    /*
     * Step 2: Compute the assumed owner and assumed offset of each
     * element for which the current process wants to know the
     * ownership.
     */

    int * assumedrank = malloc(partsize * sizeof(int));
    err = !assumedrank ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(perm); free(globalidxsorted);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    int * assumedidx = malloc(partsize * sizeof(int));
    err = !assumedidx ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(assumedrank);
        free(perm); free(globalidxsorted);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    for (int64_t i = 0; i < partsize; i++) {
        assumedrank[i] = globalidxsorted[i] / blksize;
        assumedidx[i] = globalidxsorted[i] % blksize;
    }
    free(globalidxsorted);

#ifdef MTXDEBUG_ASSUMED_PARTITION_READ
    for (int p = 0; p < comm_size; p++) {
        if (rank == p) {
            fprintf(stderr, "assumedrank=[");
            for (int64_t i = 0; i < partsize; i++)
                fprintf(stderr, " %d", assumedrank[i]);
            fprintf(stderr, "], ");
            fprintf(stderr, "assumedidx=[");
            for (int64_t i = 0; i < partsize; i++)
                fprintf(stderr, " %d", assumedidx[i]);
            fprintf(stderr, "]\n");
        }
        MPI_Barrier(comm); sleep(1);
    }
#endif

    /*
     * Step 3: The owners and offsets of each element are now ready to
     * send to the assumed owners. Next, create a list of processes
     * that the current process will request ownership information
     * from and count how many entries to request from each process.
     */

    int nsendranks = partsize > 0 ? 1 : 0;
    for (int64_t i = 1; i < partsize; i++) {
        if (assumedrank[i-1] != assumedrank[i])
            nsendranks++;
    }
    int * sendranks = malloc(nsendranks * sizeof(int));
    err = !sendranks ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(assumedidx); free(assumedrank);
        free(perm);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    int * sendcounts = malloc(nsendranks * sizeof(int));
    err = !sendcounts ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(sendranks);
        free(assumedidx); free(assumedrank);
        free(perm);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    if (nsendranks > 0) {
        sendranks[0] = assumedrank[0];
        sendcounts[0] = 1;
    }
    for (int64_t i = 1, p = 0; i < partsize; i++) {
        if (assumedrank[i-1] != assumedrank[i]) {
            sendranks[++p] = assumedrank[i];
            sendcounts[p] = 0;
        }
        sendcounts[p]++;
    }
    int * sdispls = malloc((nsendranks+1) * sizeof(int));
    err = !sdispls ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(sendcounts); free(sendranks);
        free(assumedidx); free(assumedrank);
        free(perm);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    sdispls[0] = 0;
    for (int p = 0; p < nsendranks; p++)
        sdispls[p+1] = sdispls[p] + sendcounts[p];

#ifdef MTXDEBUG_ASSUMED_PARTITION_READ
    for (int p = 0; p < comm_size; p++) {
        if (rank == p) {
            fprintf(stderr, "nsendranks=%d, sendranks=[", nsendranks);
            for (int64_t i = 0; i < nsendranks; i++)
                fprintf(stderr, " %d", sendranks[i]);
            fprintf(stderr, "], sendcounts=[");
            for (int64_t i = 0; i < nsendranks; i++)
                fprintf(stderr, " %d", sendcounts[i]);
            fprintf(stderr, "], sdispls=[");
            for (int64_t i = 0; i <= nsendranks; i++)
                fprintf(stderr, " %d", sdispls[i]);
            fprintf(stderr, "]\n");
        }
        MPI_Barrier(comm); sleep(1);
    }
#endif

    /*
     * Step 4: Count the number of processes to receive data from.
     * This is achieved using a single counter on each process and
     * one-sided communication, so that every process can atomically
     * update remote counters on other processes, if it needs data
     * from them.
     */

    int nrecvranks = 0;
    MPI_Win window;
    disterr->mpierrcode = MPI_Win_create(
        &nrecvranks, sizeof(int), sizeof(int), MPI_INFO_NULL, comm, &window);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(sdispls); free(sendcounts); free(sendranks);
        free(assumedidx); free(assumedrank);
        free(perm);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    disterr->mpierrcode = MPI_Win_fence(0, window);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    for (int p = 0; p < nsendranks && !err; p++) {
        int one = 1;
        disterr->mpierrcode = MPI_Accumulate(
            &one, 1, MPI_INT, sendranks[p], 0, 1, MPI_INT, MPI_SUM, window);
        err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    }
    if (err) {
        MPI_Win_fence(0, window);
    } else {
        disterr->mpierrcode = MPI_Win_fence(0, window);
        err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    }
    MPI_Win_free(&window);
    if (mtxdisterror_allreduce(disterr, err)) {
        free(sdispls); free(sendcounts); free(sendranks);
        free(assumedidx); free(assumedrank);
        free(perm);
        return MTX_ERR_MPI_COLLECTIVE;
    }

#ifdef MTXDEBUG_ASSUMED_PARTITION_READ
    for (int p = 0; p < comm_size; p++) {
        if (rank == p) fprintf(stderr, "nrecvranks=%d\n", nrecvranks);
        MPI_Barrier(comm); sleep(1);
    }
#endif

    /*
     * Step 5: Post non-blocking, wildcard receives for every process
     * that will send data to the current process. Thereafter, the
     * current process transmits the number of elements it needs to
     * send to each process it will later send data to.
     */

    /* TODO: With some additional checks, each process should be able
     * to avoid sending to itself. */

    MPI_Request * req = malloc(
        (nsendranks > nrecvranks ? nsendranks : nrecvranks) * sizeof(MPI_Request));
    err = !req ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(sdispls); free(sendcounts); free(sendranks);
        free(assumedidx); free(assumedrank);
        free(perm);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    int * recvranks = malloc(nrecvranks * sizeof(int));
    err = !recvranks ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(req);
        free(sdispls); free(sendcounts); free(sendranks);
        free(assumedidx); free(assumedrank);
        free(perm);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    int * recvcounts = malloc(nrecvranks * sizeof(int));
    err = !recvcounts ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(recvranks); free(req);
        free(sdispls); free(sendcounts); free(sendranks);
        free(assumedidx); free(assumedrank);
        free(perm);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    for (int p = 0; p < nrecvranks; p++) {
        int mpierrcode = MPI_Irecv(
            &recvcounts[p], 1, MPI_INT, MPI_ANY_SOURCE, 0, comm, &req[p]);
        if (!err) {
            disterr->mpierrcode = mpierrcode;
            err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
        }
    }
    if (mtxdisterror_allreduce(disterr, err)) {
        free(recvcounts); free(recvranks); free(req);
        free(sdispls); free(sendcounts); free(sendranks);
        free(assumedidx); free(assumedrank);
        free(perm);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    for (int p = 0; p < nsendranks; p++) {
        int mpierrcode = MPI_Send(
            &sendcounts[p], 1, MPI_INT, sendranks[p], 0, comm);
        if (!err) {
            disterr->mpierrcode = mpierrcode;
            err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
        }
    }
    if (mtxdisterror_allreduce(disterr, err)) {
        free(recvcounts); free(recvranks); free(req);
        free(sdispls); free(sendcounts); free(sendranks);
        free(assumedidx); free(assumedrank);
        free(perm);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    for (int p = 0; p < nrecvranks; p++) {
        MPI_Status status;
        disterr->mpierrcode = MPI_Wait(&req[p], &status);
        err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
        if (err) break;
        recvranks[p] = status.MPI_SOURCE;
    }
    if (mtxdisterror_allreduce(disterr, err)) {
        free(recvcounts); free(recvranks); free(req);
        free(sdispls); free(sendcounts); free(sendranks);
        free(assumedidx); free(assumedrank);
        free(perm);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    int * rdispls = malloc((nrecvranks+1) * sizeof(int));
    err = !rdispls ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(recvcounts); free(recvranks); free(req);
        free(sdispls); free(sendcounts); free(sendranks);
        free(assumedidx); free(assumedrank);
        free(perm);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    rdispls[0] = 0;
    for (int p = 0; p < nrecvranks; p++)
        rdispls[p+1] = rdispls[p] + recvcounts[p];

#ifdef MTXDEBUG_ASSUMED_PARTITION_READ
    for (int p = 0; p < comm_size; p++) {
        if (rank == p) {
            fprintf(stderr, "recvranks=[");
            for (int64_t i = 0; i < nrecvranks; i++)
                fprintf(stderr, " %d", recvranks[i]);
            fprintf(stderr, "], recvcounts=[");
            for (int64_t i = 0; i < nrecvranks; i++)
                fprintf(stderr, " %d", recvcounts[i]);
            fprintf(stderr, "], rdispls=[");
            for (int64_t i = 0; i <= nrecvranks; i++)
                fprintf(stderr, " %d", rdispls[i]);
            fprintf(stderr, "]\n");
        }
        MPI_Barrier(comm); sleep(1);
    }
#endif

    /*
     * Step 6: Send the assumed offset of each element.
     */

    int * assumedidxrecvbuf = malloc(rdispls[nrecvranks] * sizeof(int));
    err = !assumedidxrecvbuf ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(rdispls); free(recvcounts); free(recvranks); free(req);
        free(sdispls); free(sendcounts); free(sendranks);
        free(assumedidx); free(assumedrank);
        free(perm);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    for (int p = 0; p < nrecvranks; p++) {
        int mpierrcode = MPI_Irecv(
            &assumedidxrecvbuf[rdispls[p]], recvcounts[p], MPI_INT,
            recvranks[p], 1, comm, &req[p]);
        if (!err) {
            disterr->mpierrcode = mpierrcode;
            err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
        }
    }
    if (mtxdisterror_allreduce(disterr, err)) {
        free(assumedidxrecvbuf);
        free(rdispls); free(recvcounts); free(recvranks); free(req);
        free(sdispls); free(sendcounts); free(sendranks);
        free(assumedidx); free(assumedrank);
        free(perm);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    for (int p = 0; p < nsendranks; p++) {
        int mpierrcode = MPI_Send(
            &assumedidx[sdispls[p]], sendcounts[p], MPI_INT,
            sendranks[p], 1, comm);
        if (!err)  {
            disterr->mpierrcode = mpierrcode;
            err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
        }
    }
    if (mtxdisterror_allreduce(disterr, err)) {
        free(assumedidxrecvbuf);
        free(rdispls); free(recvcounts); free(recvranks); free(req);
        free(sdispls); free(sendcounts); free(sendranks);
        free(assumedidx); free(assumedrank);
        free(perm);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    MPI_Waitall(nrecvranks, req, MPI_STATUSES_IGNORE);

#ifdef MTXDEBUG_ASSUMED_PARTITION_READ
    for (int p = 0; p < comm_size; p++) {
        if (rank == p) {
            fprintf(stderr, "assumedidxrecvbuf=[");
            for (int q = 0; q < nrecvranks; q++) {
                if (q > 0) fprintf(stderr, " |");
                for (int64_t i = rdispls[q]; i < rdispls[q+1]; i++)
                    fprintf(stderr, " %d", assumedidxrecvbuf[i]);
            }
            fprintf(stderr, "]\n");
        }
        MPI_Barrier(comm); sleep(1);
    }
#endif

    /*
     * Step 7: Gather the owning ranks and offsets of each requested
     * element.
     */

    int * ownerranksendbuf = malloc(rdispls[nrecvranks] * sizeof(int));
    err = !ownerranksendbuf ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(assumedidxrecvbuf);
        free(rdispls); free(recvcounts); free(recvranks); free(req);
        free(sdispls); free(sendcounts); free(sendranks);
        free(assumedidx); free(assumedrank);
        free(perm);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    int * owneridxsendbuf = malloc(rdispls[nrecvranks] * sizeof(int));
    err = !owneridxsendbuf ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(ownerranksendbuf); free(assumedidxrecvbuf);
        free(rdispls); free(recvcounts); free(recvranks); free(req);
        free(sdispls); free(sendcounts); free(sendranks);
        free(assumedidx); free(assumedrank);
        free(perm);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    for (int p = 0; p < nrecvranks; p++) {
        for (int64_t i = rdispls[p]; i < rdispls[p+1]; i++) {
            ownerranksendbuf[i] = apownerrank[assumedidxrecvbuf[i]];
            owneridxsendbuf[i] = apowneridx[assumedidxrecvbuf[i]];
        }
    }

#ifdef MTXDEBUG_ASSUMED_PARTITION_READ
    for (int p = 0; p < comm_size; p++) {
        if (rank == p) {
            fprintf(stderr, "ownerranksendbuf=[");
            for (int q = 0; q < nrecvranks; q++) {
                if (q > 0) fprintf(stderr, " |");
                for (int64_t i = rdispls[q]; i < rdispls[q+1]; i++)
                    fprintf(stderr, " %d", ownerranksendbuf[i]);
            }
            fprintf(stderr, "], owneridxsendbuf=[");
            for (int q = 0; q < nrecvranks; q++) {
                if (q > 0) fprintf(stderr, " |");
                for (int64_t i = rdispls[q]; i < rdispls[q+1]; i++)
                    fprintf(stderr, " %d", owneridxsendbuf[i]);
            }
            fprintf(stderr, "]\n");
        }
        MPI_Barrier(comm); sleep(1);
    }
#endif

    /*
     * Step 8: Reply with the owning rank of each requested element,
     * as well as its offset among elements of the owning rank.
     */

    int * ownerrankrecvbuf = malloc(sdispls[nsendranks] * sizeof(int));
    err = !ownerrankrecvbuf ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(owneridxsendbuf); free(ownerranksendbuf); free(assumedidxrecvbuf);
        free(rdispls); free(recvcounts); free(recvranks); free(req);
        free(sdispls); free(sendcounts); free(sendranks);
        free(assumedidx); free(assumedrank);
        free(perm);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    int * owneridxrecvbuf = malloc(sdispls[nsendranks] * sizeof(int));
    err = !owneridxrecvbuf ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(ownerrankrecvbuf);
        free(owneridxsendbuf); free(ownerranksendbuf); free(assumedidxrecvbuf);
        free(rdispls); free(recvcounts); free(recvranks); free(req);
        free(sdispls); free(sendcounts); free(sendranks);
        free(assumedidx); free(assumedrank);
        free(perm);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    for (int p = 0; p < nsendranks; p++) {
        int mpierrcode = MPI_Irecv(
            &ownerrankrecvbuf[sdispls[p]], sendcounts[p], MPI_INT,
            sendranks[p], 2, comm, &req[p]);
        if (!err) {
            disterr->mpierrcode = mpierrcode;
            err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
        }
    }
    if (mtxdisterror_allreduce(disterr, err)) {
        free(owneridxrecvbuf); free(ownerrankrecvbuf);
        free(owneridxsendbuf); free(ownerranksendbuf); free(assumedidxrecvbuf);
        free(rdispls); free(recvcounts); free(recvranks); free(req);
        free(sdispls); free(sendcounts); free(sendranks);
        free(assumedidx); free(assumedrank);
        free(perm);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    for (int p = 0; p < nrecvranks; p++) {
        int mpierrcode = MPI_Send(
            &ownerranksendbuf[rdispls[p]], recvcounts[p], MPI_INT,
            recvranks[p], 2, comm);
        if (!err)  {
            disterr->mpierrcode = mpierrcode;
            err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
        }
    }
    if (mtxdisterror_allreduce(disterr, err)) {
        free(owneridxrecvbuf); free(ownerrankrecvbuf);
        free(owneridxsendbuf); free(ownerranksendbuf); free(assumedidxrecvbuf);
        free(rdispls); free(recvcounts); free(recvranks); free(req);
        free(sdispls); free(sendcounts); free(sendranks);
        free(assumedidx); free(assumedrank);
        free(perm);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    MPI_Waitall(nsendranks, req, MPI_STATUSES_IGNORE);

    for (int p = 0; p < nsendranks; p++) {
        int mpierrcode = MPI_Irecv(
            &owneridxrecvbuf[sdispls[p]], sendcounts[p], MPI_INT,
            sendranks[p], 3, comm, &req[p]);
        if (!err) {
            disterr->mpierrcode = mpierrcode;
            err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
        }
    }
    if (mtxdisterror_allreduce(disterr, err)) {
        free(owneridxrecvbuf); free(ownerrankrecvbuf);
        free(owneridxsendbuf); free(ownerranksendbuf); free(assumedidxrecvbuf);
        free(rdispls); free(recvcounts); free(recvranks); free(req);
        free(sdispls); free(sendcounts); free(sendranks);
        free(assumedidx); free(assumedrank);
        free(perm);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    for (int p = 0; p < nrecvranks; p++) {
        int mpierrcode = MPI_Send(
            &owneridxsendbuf[rdispls[p]], recvcounts[p], MPI_INT,
            recvranks[p], 3, comm);
        if (!err)  {
            disterr->mpierrcode = mpierrcode;
            err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
        }
    }
    if (mtxdisterror_allreduce(disterr, err)) {
        free(owneridxrecvbuf); free(ownerrankrecvbuf);
        free(owneridxsendbuf); free(ownerranksendbuf); free(assumedidxrecvbuf);
        free(rdispls); free(recvcounts); free(recvranks); free(req);
        free(sdispls); free(sendcounts); free(sendranks);
        free(assumedidx); free(assumedrank);
        free(perm);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    MPI_Waitall(nsendranks, req, MPI_STATUSES_IGNORE);

#ifdef MTXDEBUG_ASSUMED_PARTITION_READ
    for (int p = 0; p < comm_size; p++) {
        if (rank == p) {
            fprintf(stderr, "ownerrankrecvbuf=[");
            for (int q = 0; q < nsendranks; q++) {
                if (q > 0) fprintf(stderr, " |");
                for (int64_t i = sdispls[q]; i < sdispls[q+1]; i++)
                    fprintf(stderr, " %d", ownerrankrecvbuf[i]);
            }
            fprintf(stderr, "], owneridxrecvbuf=[");
            for (int q = 0; q < nsendranks; q++) {
                if (q > 0) fprintf(stderr, " |");
                for (int64_t i = sdispls[q]; i < sdispls[q+1]; i++)
                    fprintf(stderr, " %d", owneridxrecvbuf[i]);
            }
            fprintf(stderr, "]\n");
        }
        MPI_Barrier(comm); sleep(1);
    }
#endif

    /*
     * Step 9: Scatter the received data to their final locations.
     */

    for (int64_t i = 0; i < partsize; i++) {
        ownerrank[i] = ownerrankrecvbuf[perm[i]];
        owneridx[i] = owneridxrecvbuf[perm[i]];
    }

#ifdef MTXDEBUG_ASSUMED_PARTITION_READ
    for (int p = 0; p < comm_size; p++) {
        if (rank == p) {
            fprintf(stderr, "ownerrank=[");
            for (int64_t i = 0; i < partsize; i++)
                fprintf(stderr, " %d", ownerrank[i]);
            fprintf(stderr, "], owneridx=[");
            for (int64_t i = 0; i < partsize; i++)
                fprintf(stderr, " %d", owneridx[i]);
            fprintf(stderr, "]\n");
        }
        MPI_Barrier(comm); sleep(1);
    }
#endif

    free(owneridxrecvbuf); free(ownerrankrecvbuf);
    free(owneridxsendbuf); free(ownerranksendbuf); free(assumedidxrecvbuf);
    free(rdispls); free(recvcounts); free(recvranks); free(req);
    free(sdispls); free(sendcounts); free(sendranks);
    free(assumedidx); free(assumedrank);
    free(perm);
    return MTX_SUCCESS;
}
#endif

/*
 * Types of partitioning
 */

/**
 * ‘mtxpartitioning_str()’ is a string representing the partition
 * type.
 */
const char * mtxpartitioning_str(
    enum mtxpartitioning type)
{
    switch (type) {
    case mtx_singleton: return "singleton";
    case mtx_block: return "block";
    case mtx_cyclic: return "cyclic";
    case mtx_block_cyclic: return "block-cyclic";
    case mtx_custom_partition: return "custom";
    default: return mtxstrerror(MTX_ERR_INVALID_PARTITION_TYPE);
    }
}

/**
 * ‘mtxpartitioning_parse()’ parses a string to obtain one of the
 * partition types of ‘enum mtxpartitioning’.
 *
 * ‘valid_delimiters’ is either ‘NULL’, in which case it is ignored,
 * or it is a string of characters considered to be valid delimiters
 * for the parsed string.  That is, if there are any remaining,
 * non-NULL characters after parsing, then then the next character is
 * searched for in ‘valid_delimiters’.  If the character is found,
 * then the parsing succeeds and the final delimiter character is
 * consumed by the parser. Otherwise, the parsing fails with an error.
 *
 * If ‘endptr’ is not ‘NULL’, then the address stored in ‘endptr’
 * points to the first character beyond the characters that were
 * consumed during parsing.
 *
 * On success, ‘mtxpartitioning_parse()’ returns ‘MTX_SUCCESS’ and
 * ‘partition_type’ is set according to the parsed string and
 * ‘bytes_read’ is set to the number of bytes that were consumed by
 * the parser.  Otherwise, an error code is returned.
 */
int mtxpartitioning_parse(
    enum mtxpartitioning * partition_type,
    int64_t * bytes_read,
    const char ** endptr,
    const char * s,
    const char * valid_delimiters)
{
    const char * t = s;
    if (strncmp("singleton", t, strlen("singleton")) == 0) {
        t += strlen("singleton");
        *partition_type = mtx_singleton;
    } else if (strncmp("block-cyclic", t, strlen("block-cyclic")) == 0) {
        t += strlen("block-cyclic");
        *partition_type = mtx_block_cyclic;
    } else if (strncmp("block", t, strlen("block")) == 0) {
        t += strlen("block");
        *partition_type = mtx_block;
    } else if (strncmp("cyclic", t, strlen("cyclic")) == 0) {
        t += strlen("cyclic");
        *partition_type = mtx_cyclic;
    } else if (strncmp("custom", t, strlen("custom")) == 0) {
        t += strlen("custom");
        *partition_type = mtx_custom_partition;
    } else {
        return MTX_ERR_INVALID_PARTITION_TYPE;
    }
    if (valid_delimiters && *t != '\0') {
        if (!strchr(valid_delimiters, *t))
            return MTX_ERR_INVALID_PARTITION_TYPE;
        t++;
    }
    if (bytes_read)
        *bytes_read += t-s;
    if (endptr)
        *endptr = t;
    return MTX_SUCCESS;
}

/*
 * Partitions of finite sets
 */

/**
 * ‘mtxpartition_free()’ frees resources associated with a
 * partitioning.
 */
void mtxpartition_free(
    struct mtxpartition * partition)
{
    free(partition->elements_per_part);
    free(partition->parts);
    free(partition->parts_ptr);
    free(partition->part_sizes);
}

/**
 * ‘mtxpartition_init()’ initialises a partitioning of a finite set.
 */
int mtxpartition_init(
    struct mtxpartition * partition,
    enum mtxpartitioning type,
    int64_t size,
    int num_parts,
    const int64_t * part_sizes,
    int block_size,
    const int * parts,
    const int64_t * elements_per_part)
{
    if (type == mtx_singleton && num_parts == 1) {
        return mtxpartition_init_singleton(partition, size);
    } else if (type == mtx_block) {
        return mtxpartition_init_block(partition, size, num_parts, part_sizes);
    } else if (type == mtx_cyclic) {
        return mtxpartition_init_cyclic(partition, size, num_parts);
    } else if (type == mtx_block_cyclic) {
        return mtxpartition_init_block_cyclic(
            partition, size, num_parts, block_size);
    } else if (type == mtx_custom_partition) {
        return mtxpartition_init_custom(
            partition, size, num_parts, parts, part_sizes, elements_per_part);
    } else {
        return MTX_ERR_INVALID_PARTITION_TYPE;
    }
}

/**
 * ‘mtxpartition_copy()’ creates a copy of a partitioning.
 */
int mtxpartition_copy(
    struct mtxpartition * dst,
    const struct mtxpartition * src)
{
    dst->type = src->type;
    dst->size = src->size;
    dst->num_parts = src->num_parts;
    dst->part_sizes = malloc(dst->num_parts * sizeof(int64_t));
    if (!dst->part_sizes)
        return MTX_ERR_ERRNO;
    for (int p = 0; p < dst->num_parts; p++)
        dst->part_sizes[p] = src->part_sizes[p];
    dst->parts_ptr = malloc((dst->num_parts+1) * sizeof(int64_t));
    if (!dst->parts_ptr) {
        free(dst->part_sizes);
        return MTX_ERR_ERRNO;
    }
    for (int p = 0; p <= dst->num_parts; p++)
        dst->parts_ptr[p] = src->parts_ptr[p];
    if (src->parts) {
        dst->parts = malloc(dst->size * sizeof(int64_t));
        if (!dst->parts) {
            free(dst->parts_ptr);
            free(dst->part_sizes);
            return MTX_ERR_ERRNO;
        }
        for (int64_t i = 0; i < dst->size; i++)
            dst->parts[i] = src->parts[i];
    } else {
        dst->parts = NULL;
    }
    if (src->elements_per_part) {
        dst->elements_per_part = malloc(dst->size * sizeof(int64_t));
        if (!dst->elements_per_part) {
            free(dst->parts);
            free(dst->parts_ptr);
            free(dst->part_sizes);
            return MTX_ERR_ERRNO;
        }
        for (int64_t i = 0; i < dst->size; i++)
            dst->elements_per_part[i] = src->elements_per_part[i];
    } else {
        dst->elements_per_part = NULL;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxpartition_init_singleton()’ initialises a finite set that is
 * not partitioned.
 */
int mtxpartition_init_singleton(
    struct mtxpartition * partition,
    int64_t size)
{
    if (size > INT_MAX) {
        errno = ERANGE;
        return MTX_ERR_ERRNO;
    }
    partition->type = mtx_singleton;
    partition->size = size;
    partition->num_parts = 1;
    partition->part_sizes = malloc(partition->num_parts * sizeof(int64_t));
    if (!partition->part_sizes)
        return MTX_ERR_ERRNO;
    partition->part_sizes[0] = size;
    partition->parts_ptr = malloc((partition->num_parts+1) * sizeof(int64_t));
    if (!partition->parts_ptr) {
        free(partition->part_sizes);
        return MTX_ERR_ERRNO;
    }
    partition->parts_ptr[0] = 0;
    partition->parts_ptr[1] = size;
    partition->parts = NULL;
    partition->elements_per_part = NULL;
    return MTX_SUCCESS;
}

/**
 * ‘mtxpartition_init_block()’ initialises a block partitioning of a
 * finite set. Each block is made up of a contiguous set of elements,
 * but blocks may vary in size.
 *
 * If ‘part_sizes’ is ‘NULL’, then the elements are divided into
 * blocks of equal size. Otherwise, ‘part_sizes’ must point to an
 * array of length ‘num_parts’ containing the number of elements in
 * each part. Moreover, the sum of the entries in ‘part_sizes’ must be
 * equal to ‘size’.
 */
int mtxpartition_init_block(
    struct mtxpartition * partition,
    int64_t size,
    int num_parts,
    const int64_t * part_sizes)
{
    int err;
    if (num_parts <= 0)
        return MTX_ERR_INVALID_PARTITION;
    if (size / num_parts >= INT_MAX) {
        errno = ERANGE;
        return MTX_ERR_ERRNO;
    }
    partition->type = mtx_block;
    partition->size = size;
    partition->num_parts = num_parts;
    partition->part_sizes = malloc(partition->num_parts * sizeof(int64_t));
    if (!partition->part_sizes)
        return MTX_ERR_ERRNO;
    if (part_sizes) {
        for (int p = 0; p < num_parts; p++)
            partition->part_sizes[p] = part_sizes[p];
    } else {
        for (int p = 0; p < num_parts; p++) {
            partition->part_sizes[p] =
                (size / num_parts + (p < (size % num_parts) ? 1 : 0));
        }
    }

    partition->parts_ptr = malloc((partition->num_parts+1) * sizeof(int64_t));
    if (!partition->parts_ptr) {
        free(partition->part_sizes);
        return MTX_ERR_ERRNO;
    }
    partition->parts_ptr[0] = 0;
    for (int p = 0; p < num_parts; p++) {
        partition->parts_ptr[p+1] =
            partition->parts_ptr[p] + partition->part_sizes[p];
    }
    if (partition->parts_ptr[num_parts] != size) {
        free(partition->parts_ptr);
        free(partition->part_sizes);
        return MTX_ERR_INVALID_PARTITION;
    }
    partition->parts = NULL;
    partition->elements_per_part = NULL;
    return MTX_SUCCESS;
}

/**
 * ‘mtxpartition_init_cyclic()’ initialises a cyclic partitioning of
 * a finite set.
 */
int mtxpartition_init_cyclic(
    struct mtxpartition * partition,
    int64_t size,
    int num_parts)
{
    int err;
    if (num_parts <= 0)
        return MTX_ERR_INVALID_PARTITION;
    if (size / num_parts >= INT_MAX) {
        errno = ERANGE;
        return MTX_ERR_ERRNO;
    }

    partition->type = mtx_cyclic;
    partition->size = size;
    partition->num_parts = num_parts;
    partition->part_sizes = malloc(partition->num_parts * sizeof(int64_t));
    if (!partition->part_sizes)
        return MTX_ERR_ERRNO;
    for (int p = 0; p < num_parts; p++) {
        partition->part_sizes[p] =
            (size / num_parts + (p < (size % num_parts) ? 1 : 0));
    }

    partition->parts_ptr = malloc((partition->num_parts+1) * sizeof(int64_t));
    if (!partition->parts_ptr) {
        free(partition->part_sizes);
        return MTX_ERR_ERRNO;
    }
    partition->parts_ptr[0] = 0;
    for (int p = 0; p < num_parts; p++) {
        partition->parts_ptr[p+1] =
            partition->parts_ptr[p] + partition->part_sizes[p];
    }
    if (partition->parts_ptr[num_parts] != size) {
        free(partition->parts_ptr);
        free(partition->part_sizes);
        return MTX_ERR_INVALID_PARTITION;
    }
    partition->parts = NULL;
    partition->elements_per_part = NULL;
    return MTX_SUCCESS;
}

/**
 * ‘mtxpartition_init_block_cyclic()’ initialises a block-cyclic
 * partitioning of a finite set.
 */
int mtxpartition_init_block_cyclic(
    struct mtxpartition * partition,
    int64_t size,
    int num_parts,
    int block_size)
{
    errno = ENOTSUP;
    return MTX_ERR_ERRNO;
}

/**
 * ‘mtxpartition_init_custom()’ initialises a user-defined
 * partitioning of a finite set.
 *
 * In the simplest case, a partition can be created by specifying the
 * part number for each element in the set. Elements remain ordered
 * within each part according to their global element numbers, and
 * thus no additional reordering is performed. To achieve this,
 * ‘part_sizes’ and ‘elements_per_part’ must both be ‘NULL’, and
 * ‘parts’ must point to an array of length ‘size’. Moreover, each
 * entry in ‘parts’ is a non-negative integer less than ‘num_parts’,
 * which assigns a part number to the corresponding global element.
 *
 * Alternatively, a partition may be specified by providing the global
 * element numbers for the elements that make up each part. This
 * method also allows an arbitrary ordering or numbering of elements
 * within each part. Thus, there is no requirement for elements within
 * a part to be ordered according to their global element numbers.
 *
 * To create a custom partition with arbitrary ordering of local
 * elements, ‘part_sizes’ and ‘elements_per_part’ must both be
 * non-‘NULL’. The former must point to an array of size ‘num_parts’,
 * whereas the latter must point to an array of length ‘size’. In this
 * case, ‘parts’ is ignored and may be set to ‘NULL’. Moreover,
 * ‘part_sizes’ must contain non-negative integers that specify the
 * number of elements in each part, and whose sum must be equal to
 * ‘size’. For a given part ‘p’, taking the sum of the ‘p-1’ first
 * integers in ‘part_sizes’, that is, ‘r := part_sizes[0] +
 * part_sizes[1] + ... + part_sizes[p-1]’, gives the location in the
 * array ‘elements_per_part’ of the first element belonging to the pth
 * part. Thus, the pth part of the partition is made up of the
 * elements ‘elements_per_part[r]’, ‘elements_per_part[r+1]’, ...,
 * ‘elements_per_part[r+part_sizes[p]-1]’.
 *
 * As mentioned above, any ordering of elements is allowed within each
 * part. However, some operations, such as halo updates or exchanges
 * can sometimes be carried out more efficiently if certain rules are
 * observed. For example, some reordering steps may be avoided if the
 * elements in each part are already ordered in ascendingly by global
 * element numbers. (In other words, ‘elements_per_part[r] <
 * elements_per_part[r+1] < ... <
 * elements_per_part[r+part_sizes[p]-1]’.)
 */
int mtxpartition_init_custom(
    struct mtxpartition * partition,
    int64_t size,
    int num_parts,
    const int * parts,
    const int64_t * part_sizes,
    const int64_t * elements_per_part)
{
    int err;
    if (num_parts <= 0)
        return MTX_ERR_INVALID_PARTITION;

    if (part_sizes && elements_per_part) {
        int64_t i = 0;
        for (int64_t p = 0; p < num_parts; p++) {
            for (int64_t j = 0; j < part_sizes[p]; j++, i++) {
                if (elements_per_part[i] < 0 || elements_per_part[i] >= size)
                    return MTX_ERR_INDEX_OUT_OF_BOUNDS;
            }
        }
        if (i != size)
            return MTX_ERR_INVALID_PARTITION;
    } else if (parts) {
        for (int64_t i = 0; i < size; i++) {
            if (parts[i] < 0 || parts[i] >= num_parts)
                return MTX_ERR_INDEX_OUT_OF_BOUNDS;
        }
    } else {
        return MTX_ERR_INVALID_PARTITION;
    }

    partition->type = mtx_custom_partition;
    partition->size = size;
    partition->num_parts = num_parts;
    partition->part_sizes = malloc(partition->num_parts * sizeof(int64_t));
    if (!partition->part_sizes)
        return MTX_ERR_ERRNO;

    if (part_sizes && elements_per_part) {
        for (int p = 0; p < num_parts; p++)
            partition->part_sizes[p] = part_sizes[p];
    } else {
        for (int p = 0; p < num_parts; p++)
            partition->part_sizes[p] = 0;
        for (int64_t k = 0; k < size; k++)
            partition->part_sizes[parts[k]]++;
    }

    partition->parts_ptr = malloc((partition->num_parts+1) * sizeof(int64_t));
    if (!partition->parts_ptr) {
        free(partition->part_sizes);
        return MTX_ERR_ERRNO;
    }
    partition->parts_ptr[0] = 0;
    for (int p = 0; p < num_parts; p++) {
        partition->parts_ptr[p+1] =
            partition->parts_ptr[p] + partition->part_sizes[p];
    }
    if (partition->parts_ptr[num_parts] != size) {
        free(partition->parts_ptr);
        free(partition->part_sizes);
        return MTX_ERR_INVALID_PARTITION;
    }

    partition->parts = malloc(partition->size * sizeof(int64_t));
    if (!partition->parts) {
        free(partition->parts_ptr);
        free(partition->part_sizes);
        return MTX_ERR_ERRNO;
    }
    partition->elements_per_part = malloc(partition->size * sizeof(int64_t));
    if (!partition->elements_per_part) {
        free(partition->parts);
        free(partition->parts_ptr);
        free(partition->part_sizes);
        return MTX_ERR_ERRNO;
    }
    if (part_sizes && elements_per_part) {
        for (int64_t i = 0; i < size; i++)
            partition->parts[i] = -1;

        int64_t i = 0;
        for (int64_t p = 0; p < num_parts; p++) {
            for (int64_t j = 0; j < part_sizes[p]; j++, i++) {
                partition->elements_per_part[i] = elements_per_part[i];
                partition->parts[elements_per_part[i]] = p;
            }
        }

        for (int64_t i = 0; i < size; i++) {
            if (partition->parts[i] == -1) {
                free(partition->elements_per_part);
                free(partition->parts);
                free(partition->parts_ptr);
                free(partition->part_sizes);
                return MTX_ERR_INVALID_PARTITION;
            }
        }
    } else {
        for (int64_t i = 0; i < size; i++)
            partition->parts[i] = parts[i];
        for (int64_t i = 0; i < size; i++) {
            int p = parts[i];
            partition->elements_per_part[
                partition->parts_ptr[p]] = i;
            partition->parts_ptr[p]++;
        }
        partition->parts_ptr[0] = 0;
        for (int p = 0; p < num_parts; p++) {
            partition->parts_ptr[p+1] =
                partition->parts_ptr[p] + partition->part_sizes[p];
        }
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxpartition_compare()’ checks if two partitions are the same.
 *
 * ‘result’ must point to an integer, which is used to return the
 * result of the comparison. If ‘a’ and ‘b’ are the same partitioning
 * of the same set, then the integer pointed to by ‘result’ will be
 * set to 0. Otherwise, it is set to some nonzero value.
 */
int mtxpartition_compare(
    const struct mtxpartition * a,
    const struct mtxpartition * b,
    int * result)
{
    if ((a->type != mtx_singleton &&
         a->type != mtx_block &&
         a->type != mtx_cyclic &&
         a->type != mtx_block_cyclic &&
         a->type != mtx_custom_partition) ||
        (b->type != mtx_singleton &&
         b->type != mtx_block &&
         b->type != mtx_cyclic &&
         b->type != mtx_block_cyclic &&
         b->type != mtx_custom_partition))
    {
        *result = 1;
        return MTX_ERR_INVALID_PARTITION_TYPE;
    }

    if (a->type != b->type ||
        a->size != b->size ||
        a->num_parts != b->num_parts)
    {
        *result = 1;
        return MTX_SUCCESS;
    }

    for (int p = 0; p < a->num_parts; p++) {
        if (a->part_sizes[p] != b->part_sizes[p]) {
            *result = 1;
            return MTX_SUCCESS;
        }
    }
    for (int p = 0; p <= a->num_parts; p++) {
        if (a->parts_ptr[p] != b->parts_ptr[p]) {
            *result = 1;
            return MTX_SUCCESS;
        }
    }

    if (a->type == mtx_custom_partition) {
        for (int64_t k = 0; k < a->size; k++) {
            if (a->parts[k] != b->parts[k] ||
                a->elements_per_part[k] != b->elements_per_part[k])
            {
                *result = 1;
                return MTX_SUCCESS;
            }
        }
    }

    *result = 0;
    return MTX_SUCCESS;
}

/**
 * ‘mtxpartition_assign()’ assigns part numbers to elements of an
 * array according to the partitioning.
 *
 * ‘elements’ must point to an array of length ‘size’ that is used to
 * specify (global) element numbers. If ‘parts’ is not ‘NULL’, then it
 * must also point to an array of length ‘size’, which is then used to
 * store the corresponding part number of each element in the
 * ‘elements’ array.
 *
 * Finally, if ‘localelem’ is not ‘NULL’, then it must point to an
 * array of length ‘size’. For each global element number in the
 * ‘elements’ array, ‘localelem’ is used to store the corresponding
 * local, partwise element number based on the numbering of elements
 * within each part. The ‘elements’ and ‘localelem’ pointers are
 * allowed to point to the same underlying array, in which case the
 * former is overwritten by the latter.
 */
int mtxpartition_assign(
    const struct mtxpartition * partition,
    int64_t size,
    const int64_t * elements,
    int * parts,
    int64_t * localelem)
{
    if (partition->type == mtx_singleton) {
        for (int64_t k = 0; k < size; k++) {
            if (elements[k] < 0 || elements[k] >= partition->size)
                return MTX_ERR_INDEX_OUT_OF_BOUNDS;
            if (parts) parts[k] = 0;
            if (localelem) localelem[k] = elements[k];
        }

    } else if (partition->type == mtx_block) {
        for (int64_t k = 0; k < size; k++) {
            if (elements[k] < 0 || elements[k] >= partition->size)
                return MTX_ERR_INDEX_OUT_OF_BOUNDS;
            int64_t n = elements[k];
            int part = -1;
            for (int p = 0; p < partition->num_parts; p++) {
                if (n >= partition->parts_ptr[p] &&
                    n < partition->parts_ptr[p+1]) {
                    part = p;
                    break;
                }
            }
            if (part == -1)
                return MTX_ERR_INVALID_PARTITION;
            if (parts) parts[k] = part;
            if (localelem) localelem[k] = n - partition->parts_ptr[part];
        }

    } else if (partition->type == mtx_cyclic) {
        for (int64_t k = 0; k < size; k++) {
            if (elements[k] < 0 || elements[k] >= partition->size)
                return MTX_ERR_INDEX_OUT_OF_BOUNDS;
            int part = elements[k] % partition->num_parts;
            if (parts) parts[k] = part;
            if (localelem)
                localelem[k] = (elements[k] - part) / partition->num_parts;
        }

    } else if (partition->type == mtx_block_cyclic) {
        /* TODO: Not implemented. */
        errno = ENOTSUP;
        return MTX_ERR_ERRNO;

    } else if (partition->type == mtx_custom_partition) {
        for (int64_t k = 0; k < size; k++) {
            if (elements[k] < 0 || elements[k] >= partition->size)
                return MTX_ERR_INDEX_OUT_OF_BOUNDS;
            int part = partition->parts[elements[k]];
            if (parts) parts[k] = part;
            if (localelem) {
                bool found = false;
                for (int64_t l = 0; l < partition->part_sizes[part]; l++) {
                    int64_t offset = partition->parts_ptr[part];
                    if (partition->elements_per_part[offset+l] == elements[k]) {
                        localelem[k] = l;
                        found = true;
                        break;
                    }
                }
                if (!found)
                    return MTX_ERR_INDEX_OUT_OF_BOUNDS;
            }
        }
    } else {
        return MTX_ERR_INVALID_PARTITION_TYPE;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxpartition_globalidx()’ translates from a local numbering of
 * elements within a given part to a global numbering of elements in
 * the partitioned set.
 *
 * The argument ‘part’ denotes the part of the partition for which the
 * local element numbers are given.
 *
 * The arrays ‘localelem’ and ‘globalelem’ must be of length equal to
 * ‘size’. The former is used to specify the local element numbers
 * within the specified part, and must therefore contain values in the
 * range ‘0, 1, ..., partition->part_sizes[part]-1’. If successful,
 * the array ‘globalelem’ will contain the global numbers
 * corresponding to each of the local element numbers in ‘localelem’.
 *
 * If needed, ‘localelem’ and ‘globalelem’ are allowed to point to the
 * same underlying array. The values of ‘localelem’ will then be
 * overwritten by the global element numbers.
 */
int mtxpartition_globalidx(
    const struct mtxpartition * partition,
    int part,
    int64_t size,
    const int64_t * localelem,
    int64_t * globalelem)
{
    if (part < 0 || part >= partition->num_parts)
        return MTX_ERR_INDEX_OUT_OF_BOUNDS;

    if (partition->type == mtx_singleton) {
        for (int64_t k = 0; k < size; k++) {
            if (localelem[k] < 0 || localelem[k] >= partition->part_sizes[part])
                return MTX_ERR_INDEX_OUT_OF_BOUNDS;
            globalelem[k] = localelem[k];
        }

    } else if (partition->type == mtx_block) {
        for (int64_t k = 0; k < size; k++) {
            if (localelem[k] < 0 || localelem[k] >= partition->part_sizes[part])
                return MTX_ERR_INDEX_OUT_OF_BOUNDS;
            globalelem[k] = partition->parts_ptr[part] + localelem[k];
        }

    } else if (partition->type == mtx_cyclic) {
        for (int64_t k = 0; k < size; k++) {
            if (localelem[k] < 0 || localelem[k] >= partition->part_sizes[part])
                return MTX_ERR_INDEX_OUT_OF_BOUNDS;
            globalelem[k] = part + partition->num_parts * localelem[k];
        }

    } else if (partition->type == mtx_block_cyclic) {
        /* TODO: Not implemented. */
        errno = ENOTSUP;
        return MTX_ERR_ERRNO;

    } else if (partition->type == mtx_custom_partition) {
        for (int64_t k = 0; k < size; k++) {
            if (localelem[k] < 0 || localelem[k] >= partition->part_sizes[part])
                return MTX_ERR_INDEX_OUT_OF_BOUNDS;
            int64_t offset = partition->parts_ptr[part];
            globalelem[k] = partition->elements_per_part[offset + localelem[k]];
        }

    } else {
        return MTX_ERR_INVALID_PARTITION_TYPE;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxpartition_localidx()’ translates from a global numbering of
 * elements in the partitioned set to a local numbering of elements
 * within a given part.
 *
 * The argument ‘part’ denotes the part of the partition for which the
 * local element numbers are obtained.
 *
 * The arrays ‘globalelem’ and ‘localelem’ must be of length equal to
 * ‘size’. The former is used to specify the global element numbers of
 * elements belonging to the specified part. 
 *
 * If successful, the array ‘localelem’ will contain local element
 * numbers in the range ‘0, 1, ..., partition->part_sizes[part]-1’
 * that were obtained by translating from the global element numbers
 * in ‘globalelem’.
 *
 * If needed, ‘globalelem’ and ‘localelem’ are allowed to point to the
 * same underlying array. The values of ‘globalelem’ will then be
 * overwritten by the local element numbers.
 */
int mtxpartition_localidx(
    const struct mtxpartition * partition,
    int part,
    int64_t size,
    const int64_t * globalelem,
    int64_t * localelem)
{
    if (part < 0 || part >= partition->num_parts)
        return MTX_ERR_INDEX_OUT_OF_BOUNDS;

    if (partition->type == mtx_singleton) {
        for (int64_t k = 0; k < size; k++) {
            if (globalelem[k] < 0 || globalelem[k] >= partition->part_sizes[part])
                return MTX_ERR_INDEX_OUT_OF_BOUNDS;
            localelem[k] = globalelem[k];
        }

    } else if (partition->type == mtx_block) {
        for (int64_t k = 0; k < size; k++) {
            if (globalelem[k] < partition->parts_ptr[part] ||
                globalelem[k] >= partition->parts_ptr[part+1])
                return MTX_ERR_INDEX_OUT_OF_BOUNDS;
            localelem[k] = globalelem[k] - partition->parts_ptr[part];
        }

    } else if (partition->type == mtx_cyclic) {
        for (int64_t k = 0; k < size; k++) {
            if (globalelem[k] % partition->num_parts != part)
                return MTX_ERR_INDEX_OUT_OF_BOUNDS;
            localelem[k] = (globalelem[k] - part) / partition->num_parts;
        }

    } else if (partition->type == mtx_block_cyclic) {
        /* TODO: Not implemented. */
        errno = ENOTSUP;
        return MTX_ERR_ERRNO;

    } else if (partition->type == mtx_custom_partition) {
        for (int64_t k = 0; k < size; k++) {
            bool found = false;
            for (int64_t l = 0; l < partition->part_sizes[part]; l++) {
                if (partition->elements_per_part[l] == globalelem[k]) {
                    localelem[k] = l;
                    found = true;
                    break;
                }
            }
            if (!found)
                return MTX_ERR_INDEX_OUT_OF_BOUNDS;
        }

    } else {
        return MTX_ERR_INVALID_PARTITION_TYPE;
    }
    return MTX_SUCCESS;
}

/*
 * I/O functions
 */

/**
 * ‘mtxpartition_read_parts()’ reads the part numbers assigned to
 * each element of a partitioned set from the given path.  The path
 * must be to a Matrix Market file in the form of an integer vector in
 * array format.
 *
 * If ‘path’ is ‘-’, then standard input is used.
 *
 * If an error code is returned, then ‘lines_read’ and ‘bytes_read’
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the Matrix Market file.
 */
int mtxpartition_read_parts(
    struct mtxpartition * partition,
    int num_parts,
    const char * path,
    int64_t * lines_read,
    int64_t * bytes_read)
{
    int err;
    *lines_read = -1;
    *bytes_read = 0;

    FILE * f;
    if (strcmp(path, "-") == 0) {
        int fd = dup(STDIN_FILENO);
        if (fd == -1)
            return MTX_ERR_ERRNO;
        if ((f = fdopen(fd, "r")) == NULL) {
            close(fd);
            return MTX_ERR_ERRNO;
        }
    } else if ((f = fopen(path, "r")) == NULL) {
        return MTX_ERR_ERRNO;
    }
    *lines_read = 0;
    err = mtxpartition_fread_parts(
        partition, num_parts, f, lines_read, bytes_read, 0, NULL);
    if (err) {
        fclose(f);
        return err;
    }
    fclose(f);
    return MTX_SUCCESS;
}

/**
 * ‘mtxpartition_fread_parts()’ reads the part numbers assigned to
 * each element of a partitioned set from a stream formatted as a
 * Matrix Market file.  The Matrix Market file must be in the form of
 * an integer vector in array format.
 *
 * If an error code is returned, then ‘lines_read’ and ‘bytes_read’
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the Matrix Market file.
 */
int mtxpartition_fread_parts(
    struct mtxpartition * partition,
    int num_parts,
    FILE * f,
    int64_t * lines_read,
    int64_t * bytes_read,
    size_t line_max,
    char * linebuf)
{
    int err;
    struct mtxfile mtxfile;
    err = mtxfile_fread(
        &mtxfile, mtx_single, f, lines_read, bytes_read, line_max, linebuf);
    if (err)
        return err;

    if (mtxfile.header.object != mtxfile_vector) {
        mtxfile_free(&mtxfile);
        return MTX_ERR_INVALID_MTX_OBJECT;
    } else if (mtxfile.header.format != mtxfile_array) {
        mtxfile_free(&mtxfile);
        return MTX_ERR_INVALID_MTX_FORMAT;
    } else if (mtxfile.header.field != mtxfile_integer) {
        mtxfile_free(&mtxfile);
        return MTX_ERR_INVALID_MTX_FIELD;
    }

    err = mtxpartition_init_custom(
        partition, mtxfile.size.num_rows, num_parts,
        mtxfile.data.array_integer_single, NULL, NULL);
    if (err) {
        mtxfile_free(&mtxfile);
        return err;
    }
    mtxfile_free(&mtxfile);
    return MTX_SUCCESS;
}

/**
 * ‘mtxpartition_fread_indices()’ reads the global indices of
 * elements belonging to a given part of a partitioned set from a
 * stream formatted as a Matrix Market file.  The Matrix Market file
 * must be in the form of an integer vector in array format.
 *
 * If an error code is returned, then ‘lines_read’ and ‘bytes_read’
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the Matrix Market file.
 */
int mtxpartition_fread_indices(
    struct mtxpartition * partition,
    int part,
    FILE * f,
    int64_t * lines_read,
    int64_t * bytes_read,
    size_t line_max,
    char * linebuf);

/**
 * ‘mtxpartition_write_parts()’ writes the part numbers assigned to
 * each element of a partitioned set to the given path.  The file is
 * written as a Matrix Market file in the form of an integer vector in
 * array format.
 *
 * If ‘path’ is ‘-’, then standard output is used.
 *
 * If ‘format’ is not ‘NULL’, then the given format string is used
 * when printing numerical values.  The format specifier must be '%d',
 * and a fixed field width may optionally be specified (e.g., "%3d"),
 * but variable field width (e.g., "%*d"), as well as length modifiers
 * (e.g., "%ld") are not allowed.  If ‘format’ is ‘NULL’, then the
 * format specifier '%d' is used.
 */
int mtxpartition_write_parts(
    const struct mtxpartition * partition,
    const char * path,
    const char * format,
    int64_t * bytes_written)
{
    int err;
    *bytes_written = 0;

    FILE * f;
    if (strcmp(path, "-") == 0) {
        int fd = dup(STDOUT_FILENO);
        if (fd == -1)
            return MTX_ERR_ERRNO;
        if ((f = fdopen(fd, "w")) == NULL) {
            close(fd);
            return MTX_ERR_ERRNO;
        }
    } else if ((f = fopen(path, "w")) == NULL) {
        return MTX_ERR_ERRNO;
    }
    err = mtxpartition_fwrite_parts(partition, f, format, bytes_written);
    if (err) {
        fclose(f);
        return err;
    }
    fclose(f);
    return MTX_SUCCESS;
}

/**
 * ‘mtxpartition_fwrite_parts()’ writes the part numbers assigned to
 * each element of a partitioned set to a stream formatted as a Matrix
 * Market file.  The Matrix Market file is written in the form of an
 * integer vector in array format.
 *
 * If ‘format’ is not ‘NULL’, then the given format string is used
 * when printing numerical values.  The format specifier must be '%d',
 * and a fixed field width may optionally be specified (e.g., "%3d"),
 * but variable field width (e.g., "%*d"), as well as length modifiers
 * (e.g., "%ld") are not allowed.  If ‘format’ is ‘NULL’, then the
 * format specifier '%d' is used.
 *
 * If it is not ‘NULL’, then the number of bytes written to the stream
 * is returned in ‘bytes_written’.
 */
int mtxpartition_fwrite_parts(
    const struct mtxpartition * partition,
    FILE * f,
    const char * format,
    int64_t * bytes_written)
{
    int err;
    struct mtxfile mtxfile;
    err = mtxfile_alloc_vector_array(
        &mtxfile, mtxfile_integer, mtx_single, partition->size);
    if (err)
        return err;

    int * parts = mtxfile.data.array_integer_single;
    if (partition->type == mtx_singleton) {
        for (int64_t i = 0; i < partition->size; i++)
            parts[i] = 0;
    } else if (partition->type == mtx_block) {
        int64_t size_per_part = partition->size / partition->num_parts;
        int64_t remainder = partition->size % partition->num_parts;
        for (int64_t i = 0; i < partition->size; i++) {
            parts[i] = i / (size_per_part+1);
            if (parts[i] >= remainder) {
                parts[i] = remainder +
                    (i - remainder * (size_per_part+1)) / size_per_part;
            }
        }
    } else if (partition->type == mtx_cyclic) {
        for (int64_t i = 0; i < partition->size; i++)
            parts[i] = i % partition->num_parts;
    } else if (partition->type == mtx_block_cyclic) {
        /* TODO: Not implemented. */
        mtxfile_free(&mtxfile);
        errno = ENOTSUP;
        return MTX_ERR_ERRNO;
    } else if (partition->type == mtx_custom_partition) {
        for (int64_t i = 0; i < partition->size; i++)
            parts[i] = partition->parts[i];
    } else {
        mtxfile_free(&mtxfile);
        return MTX_ERR_INVALID_PARTITION_TYPE;
    }

    err = mtxfile_fwrite(&mtxfile, f, format, bytes_written);
    if (err) {
        mtxfile_free(&mtxfile);
        return err;
    }
    mtxfile_free(&mtxfile);
    return MTX_SUCCESS;
}
