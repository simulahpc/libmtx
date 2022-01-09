/* This file is part of libmtx.
 *
 * Copyright (C) 2022 James D. Trotter
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
 * Last modified: 2022-01-09
 *
 * Data types and functions for partitioning finite sets in
 * distributed memory.
 */

#ifndef LIBMTX_UTIL_DISTPARTITION_H
#define LIBMTX_UTIL_DISTPARTITION_H

#include <libmtx/libmtx-config.h>

#include <libmtx/util/index_set.h>
#include <libmtx/util/partition.h>

#ifdef LIBMTX_HAVE_MPI
#include <mpi.h>
#endif

#include <stdint.h>

#ifdef LIBMTX_HAVE_MPI
/**
 * ‘mtxdistpartition’ is a distributed-memory representation of a
 * partitioning of a finite set.
 */
struct mtxdistpartition
{
    /**
     * ‘comm’ is an MPI communicator for processes among which the
     * partition is distributed.
     */
    MPI_Comm comm;

    /**
     * ‘comm_size’ is the size of the MPI communicator.  This is equal
     * to the number of parts of the row partitioning of the matrix or
     * vector.
     */
    int comm_size;

    /**
     * ‘rank’ is the rank of the current process.
     */
    int rank;

    /**
     * ‘type’ is the type of partitioning.
     */
    enum mtxpartitioning type;

    /**
     * ‘size’ is the number of elements in the partitioned set.
     */
    int64_t size;

    /**
     * ‘num_parts’ is the number of parts in the partition, which must
     * be equal to the size of the MPI communicator.
     */
    int num_parts;

    /**
     * ‘index_set’ is an index set that describes the elements of the
     * partitioned set belonging to the current process.
     */
    struct mtxidxset index_set;
};

/**
 * ‘mtxdistpartition_free()’ frees resources associated with a
 * partitioning.
 */
void mtxdistpartition_free(
    struct mtxdistpartition * partition);

/**
 * ‘mtxdistpartition_init()’ initialises a distributed partitioning of
 * a finite set.
 *
 * This function is a collective operation which requires every
 * process in the communicator to perform matching calls.  In
 * particular, every process in the communicator must provide the same
 * values for ‘type’, ‘size’, ‘num_parts’ and ‘block_size’.
 */
int mtxdistpartition_init(
    struct mtxdistpartition * partition,
    enum mtxpartitioning type,
    int64_t size,
    int num_parts,
    int block_size,
    const int * parts,
    MPI_Comm comm,
    int root,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistpartition_init_singleton()’ initialises a distributed
 * singleton partition of a finite set.  That is, a partition with
 * only one part, also called the trivial partition.
 */
int mtxdistpartition_init_singleton(
    struct mtxdistpartition * partition,
    int64_t size,
    MPI_Comm comm,
    int root,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistpartition_init_block()’ initialises a distributed block
 * partitioning of a finite set.
 */
int mtxdistpartition_init_block(
    struct mtxdistpartition * partition,
    int64_t size,
    int num_parts,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistpartition_init_cyclic()’ initialises a distributed cyclic
 * partitioning of a finite set.
 */
int mtxdistpartition_init_cyclic(
    struct mtxdistpartition * partition,
    int64_t size,
    int num_parts,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistpartition_init_block_cyclic()’ initialises a distributed
 * block-cyclic partitioning of a finite set.
 */
int mtxdistpartition_init_block_cyclic(
    struct mtxdistpartition * partition,
    int64_t size,
    int num_parts,
    int block_size,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxdistpartition_init_partition()’ initialises a distributed,
 * partition partitioning of a finite set.
 */
int mtxdistpartition_init_partition(
    struct mtxdistpartition * partition,
    int64_t size,
    int num_parts,
    const int * parts,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

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
 * ‘mtxdistpartition_read_parts()’ reads the part numbers assigned to
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
int mtxdistpartition_read_parts(
    struct mtxdistpartition * partition,
    int num_parts,
    const char * path,
    int * lines_read,
    int64_t * bytes_read);

/**
 * ‘mtxdistpartition_fread_parts()’ reads the part numbers assigned
 * to each element of a partitioned set from a stream formatted as a
 * Matrix Market file.  The Matrix Market file must be in the form of
 * an integer vector in array format.
 *
 * If an error code is returned, then ‘lines_read’ and ‘bytes_read’
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the Matrix Market file.
 */
int mtxdistpartition_fread_parts(
    struct mtxdistpartition * partition,
    int num_parts,
    FILE * f,
    int * lines_read,
    int64_t * bytes_read,
    size_t line_max,
    char * linebuf);

/**
 * ‘mtxdistpartition_fread_indices()’ reads the global indices of
 * elements belonging to a given part of a partitioned set from a
 * stream formatted as a Matrix Market file.  The Matrix Market file
 * must be in the form of an integer vector in array format.
 *
 * If an error code is returned, then ‘lines_read’ and ‘bytes_read’
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the Matrix Market file.
 */
int mtxdistpartition_fread_indices(
    struct mtxdistpartition * partition,
    int part,
    FILE * f,
    int * lines_read,
    int64_t * bytes_read,
    size_t line_max,
    char * linebuf);

/**
 * ‘mtxdistpartition_write_parts()’ writes the part numbers assigned
 * to each element of a partitioned set to the given path.  The file
 * is written as a Matrix Market file in the form of an integer vector
 * in array format.
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
int mtxdistpartition_write_parts(
    const struct mtxdistpartition * partition,
    const char * path,
    const char * format,
    int64_t * bytes_written);

/**
 * ‘mtxdistpartition_fwrite_parts()’ writes the part numbers assigned
 * to each element of a partitioned set to a stream formatted as a
 * Matrix Market file.  The Matrix Market file is written in the form
 * of an integer vector in array format.
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
int mtxdistpartition_fwrite_parts(
    const struct mtxdistpartition * partition,
    FILE * f,
    const char * format,
    int64_t * bytes_written);

/**
 * ‘mtxdistpartition_write_indices()’ writes the global indices of
 * elements belonging to a given part of a partitioned set to the
 * given path.  The file is written as a Matrix Market file in the
 * form of an integer vector in array format.
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
int mtxdistpartition_write_indices(
    const struct mtxdistpartition * partition,
    int part,
    const char * path,
    const char * format,
    int64_t * bytes_written);

/**
 * ‘mtxdistpartition_fwrite_indices()’ writes the global indices of
 * elements belonging to a given part of a partitioned set to a stream
 * as a Matrix Market file.  The Matrix Market file is written in the
 * form of an integer vector in array format.
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
int mtxdistpartition_fwrite_indices(
    const struct mtxdistpartition * partition,
    int part,
    FILE * f,
    const char * format,
    int64_t * bytes_written);
#endif

#endif
