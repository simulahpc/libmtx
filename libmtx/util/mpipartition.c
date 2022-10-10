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
 * Last modified: 2022-10-10
 *
 * partitioning of sets of integers distributed among multiple
 * processes with MPI.
 */

#include <libmtx/error.h>
#include <libmtx/mtxfile/mtxfile.h>
#include <libmtx/util/mpipartition.h>
#include <libmtx/util/sort.h>

#ifdef LIBMTX_HAVE_MPI
#include <mpi.h>
#endif

#include <errno.h>
#include <unistd.h>

#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

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
        size, comm_size, partsizes, idxsize, idxstride, idx, dstpart, NULL);
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
        size, comm_size, blksize, idxsize, idxstride, idx, dstpart, NULL);
}


/**
 * ‘distpartition_int64()’ partitions elements of a set of 64-bit
 * signed integers based on a given partitioning to produce an array
 * of part numbers assigned to each element in the input array.
 *
 * The number of parts is equal to the number of processes in the
 * communicator ‘comm’.
 *
 * The array to be partitioned, ‘idx’, contains ‘idxsize’ items.
 * Moreover, the user must provide an output array, ‘dstpart’, of size
 * ‘idxsize’, which is used to write the part number assigned to each
 * element of the input array.
 *
 * The set to be partitioned consists of ‘size’ items.
 *
 * - If ‘type’ is ‘mtx_block’, then ‘partsize’ specifies the size of
 *   the block on the current MPI process.
 *
 * - If ‘type’ is ‘mtx_block_cyclic’, then items are arranged in
 *   contiguous blocks of size ‘blksize’, which are then partitioned
 *   in a cyclic fashion.
 */
int distpartition_int64(
    enum mtxpartitioning type,
    int64_t size,
    int64_t partsize,
    int64_t blksize,
    int64_t idxsize,
    int idxstride,
    const int64_t * idx,
    int * dstpart,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    if (type == mtx_block) {
        return distpartition_block_int64(
            size, partsize, idxsize, idxstride, idx, dstpart, comm, disterr);
    } else if (type == mtx_block_cyclic) {
        return distpartition_block_cyclic_int64(
            size, blksize, idxsize, idxstride, idx, dstpart, comm, disterr);
    } else { return MTX_ERR_INVALID_PARTITION_TYPE; }
}

/*
 * The “assumed partition” strategy, a parallel rendezvous protocol
 * algorithm to answer queries about the ownership and location of
 * distributed data.
 *
 * See A.H. Baker, R.D. Falgout and U.M. Yang, “An assumed partition
 * algorithm for determining processor inter-communication,” Parallel
 * Computing, Vol. 32, No. 5–6, June 2006, pp. 319–414.
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
    errno = radix_sort_int64(partsize, sizeof(*globalidxsorted), globalidxsorted, perm);
    err = errno ? MTX_ERR_ERRNO : MTX_SUCCESS;
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
    errno = radix_sort_int(rdispls[nrecvranks], assumedidxrecvbuf, NULL);
    err = errno ? MTX_ERR_ERRNO : MTX_SUCCESS;
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
    errno = radix_sort_int64(partsize, sizeof(*globalidxsorted), globalidxsorted, perm);
    err = errno ? MTX_ERR_ERRNO : MTX_SUCCESS;
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
