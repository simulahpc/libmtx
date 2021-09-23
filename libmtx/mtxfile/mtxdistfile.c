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
 * Last modified: 2021-09-22
 *
 * Matrix Market files distributed among multiple processes with MPI
 * for inter-process communication.
 */

#include <libmtx/libmtx-config.h>

#include <libmtx/error.h>
#include <libmtx/mtx/precision.h>
#include <libmtx/mtxfile/comments.h>
#include <libmtx/mtxfile/data.h>
#include <libmtx/mtxfile/header.h>
#include <libmtx/mtxfile/mtxdistfile.h>
#include <libmtx/mtxfile/mtxfile.h>
#include <libmtx/mtxfile/size.h>
#include <libmtx/util/distpartition.h>
#include <libmtx/util/partition.h>

#ifdef LIBMTX_HAVE_MPI
#include <mpi.h>

#include <errno.h>

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Memory management
 */

/**
 * `mtxdistfile_free()' frees storage allocated for a distributed
 * Matrix Market file.
 */
void mtxdistfile_free(
    struct mtxdistfile * mtxdistfile)
{
    mtxdistpartition_free(&mtxdistfile->row_partition);
    mtxfile_comments_free(&mtxdistfile->comments);
    mtxfile_free(&mtxdistfile->mtxfile);
}

/**
 * `mtxdistfile_copy()' copies a distributed Matrix Market file.
 */
int mtxdistfile_copy(
    struct mtxdistfile * dst,
    const struct mtxdistfile * src);

/*
 * Convert to and from (non-distributed) Matrix Market format
 */

/**
 * `mtxdistfile_from_mtxfile()' creates a distributed Matrix Market
 * file from a Matrix Market file stored on a single root process by
 * partitioning and distributing rows of the underlying matrix or
 * vector to other processes in a communicator.
 *
 * This function performs collective communication and therefore
 * requires every process in the communicator to perform matching
 * calls to `mtxdistfile_from_mtxfile()'.
 *
 * `row_partition' must be a partitioning of the rows of the matrix or
 * vector represented by `src'.
 */
int mtxdistfile_from_mtxfile(
    struct mtxdistfile * dst,
    struct mtxfile * src,
    const struct mtx_partition * row_partition,
    int root,
    MPI_Comm comm,
    struct mtxmpierror * mpierror);

/*
 * I/O functions
 */

/**
 * `mtxdistfile_read()' reads a Matrix Market file from the given path
 * and distributes the rows of the underlying matrix or vector among
 * MPI processes in a communicator.
 *
 * The `precision' argument specifies which precision to use for
 * storing matrix or vector values.
 *
 * If `path' is `-', then standard input is used.
 *
 * If an error code is returned, then `lines_read' and `bytes_read'
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the Matrix Market file.
 *
 * Only the designated root process will read from the specified
 * stream.  The file is read in a buffered manner to avoid reading the
 * entire file into the memory of the root process, which would
 * severely limit the size of files that could be read.  That is, a
 * buffer of at most `bufsize' bytes is allocated, and data is read
 * into the buffer until it is full.  The data is then distributed
 * among processes before the buffer is cleared and filled with data
 * from the stream once more.  This continues until an error occurs or
 * until end-of-file.
 *
 * Note that for a matrix (or vector) in array format, `bufsize' must
 * be at least large enough to fit one row of matrix (or vector) data
 * per MPI process in the communicator.
 *
 * This function performs collective communication and therefore
 * requires every process in the communicator to perform matching
 * calls to the function.
 */
int mtxdistfile_read(
    struct mtxdistfile * mtxdistfile,
    enum mtx_precision precision,
    enum mtx_partition_type row_partition_type,
    const char * path,
    int * lines_read,
    int64_t * bytes_read,
    size_t line_max,
    char * linebuf,
    size_t bufsize,
    int root,
    MPI_Comm comm,
    struct mtxmpierror * mpierror)
{
    int err;
    if (lines_read)
        *lines_read = -1;
    if (bytes_read)
        *bytes_read = 0;

    int rank;
    mpierror->mpierrcode = MPI_Comm_rank(comm, &rank);
    err = mpierror->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;

    FILE * f;
    if (rank == root && strcmp(path, "-") == 0) {
        int fd = dup(STDIN_FILENO);
        if (fd == -1) {
            err = MTX_ERR_ERRNO;
        } else if ((f = fdopen(fd, "r")) == NULL) {
            int olderrno = errno;
            close(fd);
            errno = olderrno;
            err = MTX_ERR_ERRNO;
        }
    } else if (rank == root && ((f = fopen(path, "r")) == NULL)) {
        err = MTX_ERR_ERRNO;
    } else {
        err = MTX_SUCCESS;
    }
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;

    if (lines_read)
        *lines_read = 0;
    err = mtxdistfile_fread(
        mtxdistfile, precision, row_partition_type,
        f, lines_read, bytes_read, line_max, linebuf, bufsize,
        root, comm, mpierror);
    if (err) {
        if (rank == root)
            fclose(f);
        return err;
    }
    if (rank == root)
        fclose(f);
    return MTX_SUCCESS;
}

/**
 * `mtxfile_buffer_size()' configures a size line for a Matrix Market
 * file that can be used as a temporary buffer for reading on an MPI
 * root process and distributing to all the processes in a
 * communicator.  The buffer will be no greater than the given buffer
 * size `bufsize' in bytes.
 */
static int mtxfile_buffer_size(
    struct mtxfile_size * size,
    enum mtxfile_object object,
    enum mtxfile_format format,
    enum mtxfile_field field,
    enum mtx_precision precision,
    int num_rows,
    int num_columns,
    int64_t num_nonzeros,
    size_t bufsize,
    MPI_Comm comm,
    int * mpierrcode)
{
    int err;
    size_t size_per_data_line;
    err = mtxfile_data_size_per_element(
        &size_per_data_line, object, format,
        field, precision);
    if (err)
        return err;
    int64_t num_data_lines = bufsize / size_per_data_line;

    if (format == mtxfile_array) {
        int comm_size;
        *mpierrcode = MPI_Comm_size(comm, &comm_size);
        if (*mpierrcode)
            return MTX_ERR_MPI;

        if (object == mtxfile_matrix) {
            /* Round to a multiple of the product of the number of
             * columns and the number of processes, since we need
             * to read the same number of entire rows per process
             * to correctly distribute the data. */
            size->num_rows =
                (num_data_lines / (num_columns*comm_size)) * (num_columns*comm_size);
            if (size->num_rows > num_rows)
                size->num_rows = num_rows;
            size->num_columns = num_columns;
            size->num_nonzeros = -1;
        } else if (object == mtxfile_vector) {
            /* Round to a multiple of the number of processes,
             * since we need to read the same number of lines per
             * process to correctly distribute the data. */
            size->num_rows =
                ((num_data_lines / comm_size) * comm_size < num_rows) ?
                (num_data_lines / comm_size) * comm_size : num_rows;
            size->num_columns = -1;
            size->num_nonzeros = -1;
        } else {
            return MTX_ERR_INVALID_MTX_OBJECT;
        }
        if (size->num_rows <= 0)
            return MTX_ERR_NO_BUFFER_SPACE;
    } else if (format == mtxfile_coordinate) {
        size->num_rows = num_rows;
        size->num_columns = num_columns;
        size->num_nonzeros =
            (num_data_lines < num_nonzeros) ? num_data_lines : num_nonzeros;
        if (size->num_nonzeros <= 0)
            return MTX_ERR_NO_BUFFER_SPACE;
    } else {
        return MTX_ERR_INVALID_MTX_FORMAT;
    }
    return MTX_SUCCESS;
}

/**
 * `mtxdistfile_fread()' reads a Matrix Market file from a stream and
 * distributes the rows of the underlying matrix or vector among MPI
 * processes in a communicator.
 *
 * `precision' is used to determine the precision to use for storing
 * the values of matrix or vector entries.
 *
 * If an error code is returned, then `lines_read' and `bytes_read'
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the Matrix Market file.
 *
 * Only the designated root process will read from the specified
 * stream.  The file is read in a buffered manner to avoid reading the
 * entire file into the memory of the root process, which would
 * severely limit the size of files that could be read.  That is, a
 * buffer of at most `bufsize' bytes is allocated, and data is read
 * into the buffer until it is full.  The data is then distributed
 * among processes before the buffer is cleared and filled with data
 * from the stream once more.  This continues until an error occurs or
 * until end-of-file.
 *
 * Note that for a matrix (or vector) in array format, `bufsize' must
 * be at least large enough to fit one row of matrix (or vector) data
 * per MPI process in the communicator.
 *
 * This function performs collective communication and therefore
 * requires every process in the communicator to perform matching
 * calls to the function.
 */
int mtxdistfile_fread(
    struct mtxdistfile * mtxdistfile,
    enum mtx_precision precision,
    enum mtx_partition_type row_partition_type,
    FILE * f,
    int * lines_read,
    int64_t * bytes_read,
    size_t line_max,
    char * linebuf,
    size_t bufsize,
    int root,
    MPI_Comm comm,
    struct mtxmpierror * mpierror)
{
    int err;

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

    mtxdistfile->comm = comm;
    mtxdistfile->comm_size = comm_size;
    mtxdistfile->rank = rank;

    bool free_linebuf = (rank == root) && !linebuf;
    if (rank == root && !linebuf) {
        line_max = sysconf(_SC_LINE_MAX);
        linebuf = malloc(line_max+1);
    }
    err = (rank == root && !linebuf) ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;

    /* Read the header on the root process and broadcast to others. */
    err = (rank == root) ? mtxfile_fread_header(
        &mtxdistfile->header, f, lines_read, bytes_read, line_max, linebuf)
        : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err)) {
        if (free_linebuf)
            free(linebuf);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    err = mtxfile_header_bcast(&mtxdistfile->header, root, comm, mpierror);
    if (err) {
        if (free_linebuf)
            free(linebuf);
        return err;
    }

    /* Read comments on the root process and broadcast to others. */
    err = (rank == root) ? mtxfile_fread_comments(
        &mtxdistfile->comments, f, lines_read, bytes_read, line_max, linebuf)
        : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err)) {
        if (free_linebuf)
            free(linebuf);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    err = mtxfile_comments_bcast(&mtxdistfile->comments, root, comm, mpierror);
    if (err) {
        if (free_linebuf)
            free(linebuf);
        return err;
    }

    /* Read the size line on the root process and broadcast to others. */
    err = (rank == root) ? mtxfile_fread_size(
        &mtxdistfile->size, f, lines_read, bytes_read, line_max, linebuf,
        mtxdistfile->header.object, mtxdistfile->header.format)
        : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err)) {
        mtxfile_comments_free(&mtxdistfile->comments);
        if (free_linebuf)
            free(linebuf);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    err = mtxfile_size_bcast(&mtxdistfile->size, root, comm, mpierror);
    if (err) {
        mtxfile_comments_free(&mtxdistfile->comments);
        if (free_linebuf)
            free(linebuf);
        return err;
    }
    mtxdistfile->precision = precision;

    /* Partition the rows of the matrix or vector. */
    int block_size = 0;
    const int * parts = NULL;
    err = mtxdistpartition_init(
        &mtxdistfile->row_partition, row_partition_type, mtxdistfile->size.num_rows,
        comm_size, block_size, parts, comm, root, mpierror);
    if (err) {
        mtxfile_comments_free(&mtxdistfile->comments);
        if (free_linebuf)
            free(linebuf);
        return err;
    }

    /* Create initial, empty matrices on every process. */
    struct mtxfile_size initial_size;
    initial_size.num_rows =
        (mtxdistfile->size.num_nonzeros >= 0) ? mtxdistfile->size.num_rows : 0;
    initial_size.num_columns = mtxdistfile->size.num_columns;
    initial_size.num_nonzeros = (mtxdistfile->size.num_nonzeros >= 0) ? 0 : -1;
    err = mtxfile_alloc(
        &mtxdistfile->mtxfile, &mtxdistfile->header, &mtxdistfile->comments,
        &initial_size, mtxdistfile->precision);
    if (err) {
        mtxdistpartition_free(&mtxdistfile->row_partition);
        mtxfile_comments_free(&mtxdistfile->comments);
        if (free_linebuf)
            free(linebuf);
        return err;
    }

    /* Allocate a temporary Matrix Market file to be used as a buffer
     * for reading on the root process. */
    struct mtxfile_size rootmtxsize;
    err = (rank == root)
        ? mtxfile_buffer_size(
            &rootmtxsize, mtxdistfile->header.object, mtxdistfile->header.format,
            mtxdistfile->header.field, precision,
            mtxdistfile->size.num_rows, mtxdistfile->size.num_columns,
            mtxdistfile->size.num_nonzeros,
            bufsize, comm, &mpierror->mpierrcode)
        : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err)) {
        mtxdistfile_free(mtxdistfile);
        if (free_linebuf)
            free(linebuf);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    int64_t num_data_lines_rootmtx;
    err = (rank == root)
        ? mtxfile_size_num_data_lines(&mtxdistfile->size, &num_data_lines_rootmtx)
        : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err)) {
        mtxdistfile_free(mtxdistfile);
        if (free_linebuf)
            free(linebuf);
        return err;
    }
    mpierror->mpierrcode = MPI_Bcast(
        &num_data_lines_rootmtx, 1, MPI_INT64_T, root, comm);
    err = mpierror->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err)) {
        mtxdistfile_free(mtxdistfile);
        if (free_linebuf)
            free(linebuf);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    struct mtxfile rootmtx;
    err = (rank == root) ? mtxfile_alloc(
        &rootmtx, &mtxdistfile->header, &mtxdistfile->comments,
        &rootmtxsize, mtxdistfile->precision)
        : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err)) {
        mtxdistfile_free(mtxdistfile);
        if (free_linebuf)
            free(linebuf);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    /* Determine the total number of data lines. */
    int64_t num_data_lines;
    err = mtxfile_size_num_data_lines(&mtxdistfile->size, &num_data_lines);
    if (mtxmpierror_allreduce(mpierror, err)) {
        if (rank == root)
            mtxfile_free(&rootmtx);
        mtxdistfile_free(mtxdistfile);
        if (free_linebuf)
            free(linebuf);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    /* Create partitioning information on the root process to be used
     * for distributing the rows of the matrix or vector. 
     *
     * TODO: Is this needed, or can we use the distributed
     * partitioning information that was already created earlier?
     */
    struct mtx_partition row_partition;
    err = (rank == root) ? mtx_partition_init(
        &row_partition, row_partition_type,
        mtxdistfile->size.num_rows, comm_size, 0, NULL)
        : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err)) {
        if (rank == root)
            mtxfile_free(&rootmtx);
        mtxdistfile_free(mtxdistfile);
        if (free_linebuf)
            free(linebuf);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    /* Read and distribute data until we reach the end of the file or
     * an error occurs. */
    int64_t num_data_lines_remaining = num_data_lines;
    while (num_data_lines_remaining > 0) {
        int64_t num_data_lines_to_read =
            (num_data_lines_rootmtx < num_data_lines_remaining)
            ? num_data_lines_rootmtx : num_data_lines_remaining;
        if (rank == root && rootmtx.size.num_nonzeros >= 0)
            rootmtx.size.num_nonzeros = num_data_lines_to_read;

        /* Read the next set of data lines on the root process. */
        err = (rank == root) ?
            mtxfile_fread_data(
                &rootmtx.data, f, lines_read, bytes_read, line_max, linebuf,
                rootmtx.header.object, rootmtx.header.format,
                rootmtx.header.field, rootmtx.precision,
                rootmtx.size.num_rows, rootmtx.size.num_columns,
                num_data_lines_to_read)
            : MTX_SUCCESS;
        if (mtxmpierror_allreduce(mpierror, err)) {
            if (rank == root) {
                mtx_partition_free(&row_partition);
                mtxfile_free(&rootmtx);
            }
            mtxdistfile_free(mtxdistfile);
            if (free_linebuf)
                free(linebuf);
            return MTX_ERR_MPI_COLLECTIVE;
        }

        /* Distribute the matrix (or vector) rows. */
        struct mtxfile tmpmtx;
        err = mtxfile_distribute_rows(
            &tmpmtx, &rootmtx, &row_partition, root, comm, mpierror);
        if (err) {
            if (rank == root) {
                mtx_partition_free(&row_partition);
                mtxfile_free(&rootmtx);
            }
            mtxdistfile_free(mtxdistfile);
            if (free_linebuf)
                free(linebuf);
            return err;
        }

        /* Concatenate the newly distributed matrices (or vectors)
         * with the existing ones. */
        err = mtxfile_cat(&mtxdistfile->mtxfile, &tmpmtx);
        if (mtxmpierror_allreduce(mpierror, err)) {
            mtxfile_free(&tmpmtx);
            if (rank == root) {
                mtx_partition_free(&row_partition);
                mtxfile_free(&rootmtx);
            }
            mtxdistfile_free(mtxdistfile);
            if (free_linebuf)
                free(linebuf);
            return MTX_ERR_MPI_COLLECTIVE;
        }

        mtxfile_free(&tmpmtx);
        num_data_lines_remaining -= num_data_lines_to_read;
    }

    if (rank == root) {
        mtx_partition_free(&row_partition);
        mtxfile_free(&rootmtx);
    }
    if (free_linebuf)
        free(linebuf);
    return MTX_SUCCESS;
}

/**
 * `mtxdistfile_write()' writes a distributed Matrix Market file to
 * the given path.  The file may optionally be compressed by gzip.
 *
 * If `path' is `-', then standard output is used.
 *
 * If `format' is `NULL', then the format specifier '%d' is used to
 * print integers and '%f' is used to print floating point
 * numbers. Otherwise, the given format string is used when printing
 * numerical values.
 *
 * The format string follows the conventions of `printf'. If the field
 * is `real', `double' or `complex', then the format specifiers '%e',
 * '%E', '%f', '%F', '%g' or '%G' may be used. If the field is
 * `integer', then the format specifier must be '%d'. The format
 * string is ignored if the field is `pattern'. Field width and
 * precision may be specified (e.g., "%3.1f"), but variable field
 * width and precision (e.g., "%*.*f"), as well as length modifiers
 * (e.g., "%Lf") are not allowed.
 *
 * This function performs collective communication and therefore
 * requires every process in the communicator to perform matching
 * calls to the function.
 */
int mtxdistfile_write(
    const struct mtxdistfile * mtxdistfile,
    const char * path,
    bool gzip,
    const char * format,
    int64_t * bytes_written);

/**
 * `mtxdistfile_fwrite()' writes a distributed Matrix Market file to
 * the specified stream on each process.
 *
 * If `format' is `NULL', then the format specifier '%d' is used to
 * print integers and '%f' is used to print floating point
 * numbers. Otherwise, the given format string is used when printing
 * numerical values.
 *
 * The format string follows the conventions of `printf'. If the field
 * is `real', `double' or `complex', then the format specifiers '%e',
 * '%E', '%f', '%F', '%g' or '%G' may be used. If the field is
 * `integer', then the format specifier must be '%d'. The format
 * string is ignored if the field is `pattern'. Field width and
 * precision may be specified (e.g., "%3.1f"), but variable field
 * width and precision (e.g., "%*.*f"), as well as length modifiers
 * (e.g., "%Lf") are not allowed.
 *
 * If it is not `NULL', then the number of bytes written to the stream
 * is returned in `bytes_written'.
 *
 * If `sequential' is true, then output is performed in sequence by
 * MPI processes in the communicator.  This is useful, for example,
 * when writing to standard output.  In this case, we want to ensure
 * that the processes write their data in the correct order without
 * interfering with each other.
 *
 * This function performs collective communication and therefore
 * requires every process in the communicator to perform matching
 * calls to the function.
 */
int mtxdistfile_fwrite(
    const struct mtxdistfile * mtxdistfile,
    FILE * f,
    const char * format,
    int64_t * bytes_written,
    bool sequential);
#endif
