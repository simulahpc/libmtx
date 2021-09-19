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
 * Last modified: 2021-09-19
 *
 * Data types and functions for partitioning finite sets in
 * distributed memory.
 */

#include <libmtx/libmtx-config.h>

#include <libmtx/error.h>
#include <libmtx/util/distpartition.h>
#include <libmtx/util/index_set.h>
#include <libmtx/util/partition.h>

#ifdef LIBMTX_HAVE_MPI
#include <mpi.h>
#endif

#include <stdint.h>

#ifdef LIBMTX_HAVE_MPI
/**
 * `mtx_distpartition_free()' frees resources associated with a
 * partitioning.
 */
void mtx_distpartition_free(
    struct mtx_distpartition * partition)
{
    mtx_index_set_free(&partition->index_set);
}

/**
 * `mtx_distpartition_init()' initialises a distributed partitioning
 * of a finite set.
 */
int mtx_distpartition_init(
    struct mtx_distpartition * partition,
    enum mtx_partition_type type,
    int64_t size,
    int num_parts,
    int block_size,
    const int * parts,
    MPI_Comm comm,
    int root,
    struct mtxmpierror * mpierror)
{
    if (type == mtx_singleton) {
        return mtx_distpartition_init_singleton(
            partition, size, comm, root, mpierror);
    } else if (type == mtx_block) {
        return mtx_distpartition_init_block(
            partition, size, num_parts, comm, mpierror);
    } else if (type == mtx_cyclic) {
        return mtx_distpartition_init_cyclic(
            partition, size, num_parts, comm, mpierror);
    } else if (type == mtx_block_cyclic) {
        return mtx_distpartition_init_block_cyclic(
            partition, size, num_parts, block_size, comm, mpierror);
    } else if (type == mtx_unstructured) {
        return mtx_distpartition_init_unstructured(
            partition, size, num_parts, parts, comm, mpierror);
    } else {
        return MTX_ERR_INVALID_PARTITION_TYPE;
    }
}

/**
 * `mtx_distpartition_init_singleton()' initialises a distributed
 * singleton partition of a finite set.  That is, a partition with
 * only one part, also called the trivial partition.
 */
int mtx_distpartition_init_singleton(
    struct mtx_distpartition * partition,
    int64_t size,
    MPI_Comm comm,
    int root,
    struct mtxmpierror * mpierror)
{
    int err;
    int rank;
    mpierror->mpierrcode = MPI_Comm_rank(comm, &rank);
    err = mpierror->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    partition->comm = comm;
    partition->type = mtx_singleton;
    partition->size = size;
    partition->num_parts = 1;
    err = mtx_index_set_init_interval(
        &partition->index_set, 0,
        rank == root ? size : 0);
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * `mtx_distpartition_init_block()' initialises a distributed block
 * partitioning of a finite set.
 */
int mtx_distpartition_init_block(
    struct mtx_distpartition * partition,
    int64_t size,
    int num_parts,
    MPI_Comm comm,
    struct mtxmpierror * mpierror)
{
    int err;
    int comm_size;
    mpierror->mpierrcode = MPI_Comm_rank(comm, &comm_size);
    err = mpierror->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    err = comm_size < num_parts ? MTX_ERR_INDEX_OUT_OF_BOUNDS : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    int rank;
    mpierror->mpierrcode = MPI_Comm_rank(comm, &rank);
    err = mpierror->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    partition->comm = comm;
    partition->type = mtx_block;
    partition->size = size;
    partition->num_parts = num_parts;

    int64_t a = 0;
    for (int p = 0; p < rank; p++)
        a += size / num_parts + (p < (size % num_parts) ? 1 : 0);
    int64_t b = a + size / num_parts + (rank < (size % num_parts) ? 1 : 0);
    err = mtx_index_set_init_interval(&partition->index_set, a, b);
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * `mtx_distpartition_init_cyclic()' initialises a distributed cyclic
 * partitioning of a finite set.
 */
int mtx_distpartition_init_cyclic(
    struct mtx_distpartition * partition,
    int64_t size,
    int num_parts,
    MPI_Comm comm,
    struct mtxmpierror * mpierror);

/**
 * `mtx_distpartition_init_block_cyclic()' initialises a distributed
 * block-cyclic partitioning of a finite set.
 */
int mtx_distpartition_init_block_cyclic(
    struct mtx_distpartition * partition,
    int64_t size,
    int num_parts,
    int block_size,
    MPI_Comm comm,
    struct mtxmpierror * mpierror);

/**
 * `mtx_distpartition_init_unstructured()' initialises a distributed,
 * unstructured partitioning of a finite set.
 */
int mtx_distpartition_init_unstructured(
    struct mtx_distpartition * partition,
    int64_t size,
    int num_parts,
    const int * parts,
    MPI_Comm comm,
    struct mtxmpierror * mpierror);

/*
 * I/O functions
 *
 * Reading a distributed-memory partitioning from file:
 *
 *   1. Read the part numbers for each element in the partitioned set
 *      from a single Matrix Market file.
 *
 *   2. Read the global indices of each element of the partitioned set
 *      from Matrix Market files for each part.
 *
 * Writing a partition to file:
 *
 *   1. Write the part numbers for each element in the partitioned set
 *      to a single Matrix Market file.
 *
 *   2. Write the global indices of each element of the partitioned
 *      set to a Matrix Market file for each part.
 */

/**
 * `mtx_distpartition_read_parts()' reads the part numbers assigned to
 * each element of a partitioned set from the given path.  The path
 * must be to a Matrix Market file in the form of an integer vector in
 * array format.
 *
 * If `path' is `-', then standard input is used.
 *
 * If an error code is returned, then `lines_read' and `bytes_read'
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the Matrix Market file.
 */
int mtx_distpartition_read_parts(
    struct mtx_distpartition * partition,
    int num_parts,
    const char * path,
    int * lines_read,
    int64_t * bytes_read);

/**
 * `mtx_distpartition_fread_parts()' reads the part numbers assigned
 * to each element of a partitioned set from a stream formatted as a
 * Matrix Market file.  The Matrix Market file must be in the form of
 * an integer vector in array format.
 *
 * If an error code is returned, then `lines_read' and `bytes_read'
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the Matrix Market file.
 */
int mtx_distpartition_fread_parts(
    struct mtx_distpartition * partition,
    int num_parts,
    FILE * f,
    int * lines_read,
    int64_t * bytes_read,
    size_t line_max,
    char * linebuf);

/**
 * `mtx_distpartition_fread_indices()' reads the global indices of
 * elements belonging to a given part of a partitioned set from a
 * stream formatted as a Matrix Market file.  The Matrix Market file
 * must be in the form of an integer vector in array format.
 *
 * If an error code is returned, then `lines_read' and `bytes_read'
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the Matrix Market file.
 */
int mtx_distpartition_fread_indices(
    struct mtx_distpartition * partition,
    int part,
    FILE * f,
    int * lines_read,
    int64_t * bytes_read,
    size_t line_max,
    char * linebuf);

/**
 * `mtx_distpartition_write_parts()' writes the part numbers assigned
 * to each element of a partitioned set to the given path.  The file
 * is written as a Matrix Market file in the form of an integer vector
 * in array format.
 *
 * If `path' is `-', then standard output is used.
 *
 * If `format' is not `NULL', then the given format string is used
 * when printing numerical values.  The format specifier must be '%d',
 * and a fixed field width may optionally be specified (e.g., "%3d"),
 * but variable field width (e.g., "%*d"), as well as length modifiers
 * (e.g., "%ld") are not allowed.  If `format' is `NULL', then the
 * format specifier '%d' is used.
 */
int mtx_distpartition_write_parts(
    const struct mtx_distpartition * partition,
    const char * path,
    const char * format,
    int64_t * bytes_written);

/**
 * `mtx_distpartition_fwrite_parts()' writes the part numbers assigned
 * to each element of a partitioned set to a stream formatted as a
 * Matrix Market file.  The Matrix Market file is written in the form
 * of an integer vector in array format.
 *
 * If `format' is not `NULL', then the given format string is used
 * when printing numerical values.  The format specifier must be '%d',
 * and a fixed field width may optionally be specified (e.g., "%3d"),
 * but variable field width (e.g., "%*d"), as well as length modifiers
 * (e.g., "%ld") are not allowed.  If `format' is `NULL', then the
 * format specifier '%d' is used.
 *
 * If it is not `NULL', then the number of bytes written to the stream
 * is returned in `bytes_written'.
 */
int mtx_distpartition_fwrite_parts(
    const struct mtx_distpartition * partition,
    FILE * f,
    const char * format,
    int64_t * bytes_written);

/**
 * `mtx_distpartition_write_indices()' writes the global indices of
 * elements belonging to a given part of a partitioned set to the
 * given path.  The file is written as a Matrix Market file in the
 * form of an integer vector in array format.
 *
 * If `path' is `-', then standard output is used.
 *
 * If `format' is not `NULL', then the given format string is used
 * when printing numerical values.  The format specifier must be '%d',
 * and a fixed field width may optionally be specified (e.g., "%3d"),
 * but variable field width (e.g., "%*d"), as well as length modifiers
 * (e.g., "%ld") are not allowed.  If `format' is `NULL', then the
 * format specifier '%d' is used.
 */
int mtx_distpartition_write_indices(
    const struct mtx_distpartition * partition,
    int part,
    const char * path,
    const char * format,
    int64_t * bytes_written);

/**
 * `mtx_distpartition_fwrite_indices()' writes the global indices of
 * elements belonging to a given part of a partitioned set to a stream
 * as a Matrix Market file.  The Matrix Market file is written in the
 * form of an integer vector in array format.
 *
 * If `format' is not `NULL', then the given format string is used
 * when printing numerical values.  The format specifier must be '%d',
 * and a fixed field width may optionally be specified (e.g., "%3d"),
 * but variable field width (e.g., "%*d"), as well as length modifiers
 * (e.g., "%ld") are not allowed.  If `format' is `NULL', then the
 * format specifier '%d' is used.
 *
 * If it is not `NULL', then the number of bytes written to the stream
 * is returned in `bytes_written'.
 */
int mtx_distpartition_fwrite_indices(
    const struct mtx_distpartition * partition,
    int part,
    FILE * f,
    const char * format,
    int64_t * bytes_written);
#endif
