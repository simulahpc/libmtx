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
 * distributing the data of the underlying matrix or vector among
 * processes in a communicator.
 *
 * This function performs collective communication and therefore
 * requires every process in the communicator to perform matching
 * calls to this function.
 */
int mtxdistfile_from_mtxfile(
    struct mtxdistfile * dst,
    struct mtxfile * src,
    MPI_Comm comm,
    int root,
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

    /* Broadcast the header, comments, size line and precision. */
    err = (rank == root) ? mtxfile_header_copy(
        &dst->header, &src->header) : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    err = mtxfile_header_bcast(&dst->header, root, comm, mpierror);
    if (err)
        return err;
    err = (rank == root) ? mtxfile_comments_copy(
        &dst->comments, &src->comments) : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    err = mtxfile_comments_bcast(&dst->comments, root, comm, mpierror);
    if (err) {
        if (rank == root)
            mtxfile_comments_free(&dst->comments);
        return err;
    }
    err = (rank == root) ? mtxfile_size_copy(
        &dst->size, &src->size) : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err)) {
        mtxfile_comments_free(&dst->comments);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    err = mtxfile_size_bcast(&dst->size, root, comm, mpierror);
    if (err) {
        mtxfile_comments_free(&dst->comments);
        return err;
    }
    if (rank == root)
        dst->precision = src->precision;
    mpierror->mpierrcode = MPI_Bcast(
        &dst->precision, 1, MPI_INT, root, comm);
    err = mpierror->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err)) {
        mtxfile_comments_free(&dst->comments);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    int * sendcounts = (rank == root) ?
        malloc((2*comm_size+1) * sizeof(int)) : NULL;
    err = (rank == root && !sendcounts) ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err)) {
        mtxfile_comments_free(&dst->comments);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    int * displs = (rank == root) ? &sendcounts[comm_size] : NULL;

    /* Find the number of data lines to send and offsets for each
     * part. Note for matrices in array format, we evenly distribute
     * the number of rows, whereas in all other cases we evenly
     * distribute the total number of data lines. */
    err = MTX_SUCCESS;
    if (rank == root) {
        int64_t size;
        if (src->size.num_nonzeros >= 0) {
            size = src->size.num_nonzeros;
            displs[0] = 0;
            for (int p = 0; p < comm_size; p++) {
                displs[p+1] = displs[p] +
                    size / comm_size + (p < (size % comm_size) ? 1 : 0);
                sendcounts[p] = displs[p+1] - displs[p];
            }
        } else if (src->size.num_rows >= 0 && src->size.num_columns >= 0) {
            int64_t num_rows = src->size.num_rows;
            int64_t num_columns = src->size.num_columns;
            displs[0] = 0;
            for (int p = 0; p < comm_size; p++) {
                displs[p+1] = displs[p] +
                    (num_rows / comm_size + (p < (num_rows % comm_size) ? 1 : 0)) *
                    num_columns;
                sendcounts[p] = displs[p+1] - displs[p];
            }
        } else if (src->size.num_rows >= 0) {
            size = src->size.num_rows;
            displs[0] = 0;
            for (int p = 0; p < comm_size; p++) {
                displs[p+1] = displs[p] +
                    size / comm_size + (p < (size % comm_size) ? 1 : 0);
                sendcounts[p] = displs[p+1] - displs[p];
            }
        } else {
            err = MTX_ERR_INVALID_MTX_SIZE;
        }
    }
    if (mtxmpierror_allreduce(mpierror, err)) {
        if (rank == root)
            free(sendcounts);
        mtxfile_comments_free(&dst->comments);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    int recvcount;
    mpierror->mpierrcode = MPI_Scatter(
        sendcounts, 1, MPI_INT, &recvcount, 1, MPI_INT, root, comm);
    err = mpierror->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err)) {
        if (rank == root)
            free(sendcounts);
        mtxfile_comments_free(&dst->comments);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    /* Scatter the Matrix Market file. */
    err = mtxfile_scatterv(
        src, sendcounts, displs, &dst->mtxfile, recvcount, root, comm, mpierror);
    if (err) {
        if (rank == root)
            free(sendcounts);
        mtxfile_comments_free(&dst->comments);
        return err;
    }

    if (rank == root)
        free(sendcounts);
    return MTX_SUCCESS;
}

/*
 * I/O functions
 */

/**
 * `mtxdistfile_read()' reads a Matrix Market file from the given path
 * and distributes the data among MPI processes in a communicator.
 *
 * `precision' is used to determine the precision to use for storing
 * the values of matrix or vector entries.
 *
 * If an error code is returned, then `lines_read' and `bytes_read'
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the Matrix Market file.
 *
 * Only a single root process will read from the specified stream.
 * The data is partitioned into equal-sized parts for each process.
 * For matrices and vectors in coordinate format, the total number of
 * data lines is evenly distributed among processes. Otherwise, the
 * rows are evenly distributed among processes.
 *
 * The file is read one part at a time, which is then sent to the
 * owning process. This avoids reading the entire file into the memory
 * of the root process at once, which would severely limit the size of
 * files that could be read.
 *
 * This function performs collective communication and therefore
 * requires every process in the communicator to perform matching
 * calls to the function.
 */
int mtxdistfile_read(
    struct mtxdistfile * mtxdistfile,
    enum mtx_precision precision,
    const char * path,
    int * lines_read,
    int64_t * bytes_read,
    size_t line_max,
    char * linebuf,
    MPI_Comm comm,
    struct mtxmpierror * mpierror)
{
    int err;
    if (lines_read)
        *lines_read = -1;
    if (bytes_read)
        *bytes_read = 0;

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
    int root = comm_size-1;

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
        mtxdistfile, precision, f, lines_read, bytes_read, line_max, linebuf,
        comm, mpierror);
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
 * `mtxdistfile_fread()' reads a Matrix Market file from a stream and
 * distributes the data among MPI processes in a communicator.
 *
 * `precision' is used to determine the precision to use for storing
 * the values of matrix or vector entries.
 *
 * If an error code is returned, then `lines_read' and `bytes_read'
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the Matrix Market file.
 *
 * Only a single root process will read from the specified stream.
 * The data is partitioned into equal-sized parts for each process.
 * For matrices and vectors in coordinate format, the total number of
 * data lines is evenly distributed among processes. Otherwise, the
 * rows are evenly distributed among processes.
 *
 * The file is read one part at a time, which is then sent to the
 * owning process. This avoids reading the entire file into the memory
 * of the root process at once, which would severely limit the size of
 * files that could be read.
 *
 * This function performs collective communication and therefore
 * requires every process in the communicator to perform matching
 * calls to the function.
 */
int mtxdistfile_fread(
    struct mtxdistfile * mtxdistfile,
    enum mtx_precision precision,
    FILE * f,
    int * lines_read,
    int64_t * bytes_read,
    size_t line_max,
    char * linebuf,
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
    int root = comm_size-1;
    if (comm_size <= 0)
        return MTX_SUCCESS;

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

    /* Partition the data into equal-sized blocks for each process.
     * For matrices and vectors in coordinate format, the total number
     * of data lines is evenly distributed among processes. Otherwise,
     * the number of rows is evenly distributed among processes. */
    struct mtxfile_size * sizes = (rank == root) ?
        malloc(comm_size * sizeof(struct mtxfile_size)) : NULL;
    err = (rank == root && !sizes) ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err)) {
        mtxfile_comments_free(&mtxdistfile->comments);
        if (free_linebuf)
            free(linebuf);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    if (rank == root) {
        if (mtxdistfile->size.num_nonzeros >= 0) {
            int64_t N = mtxdistfile->size.num_nonzeros;
            for (int p = 0; p < comm_size; p++) {
                sizes[p].num_rows = mtxdistfile->size.num_rows;
                sizes[p].num_columns = mtxdistfile->size.num_columns;
                sizes[p].num_nonzeros = N / comm_size + (p < (N % comm_size) ? 1 : 0);
            }
        } else {
            int64_t N = mtxdistfile->size.num_rows;
            for (int p = 0; p < comm_size; p++) {
                sizes[p].num_rows = (N / comm_size + (p < (N % comm_size) ? 1 : 0));
                sizes[p].num_columns = mtxdistfile->size.num_columns;
                sizes[p].num_nonzeros = mtxdistfile->size.num_nonzeros;
            }
        }
    }
    if (mtxmpierror_allreduce(mpierror, err)) {
        if (rank == root)
            free(sizes);
        mtxfile_comments_free(&mtxdistfile->comments);
        if (free_linebuf)
            free(linebuf);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    /* Allocate storage for a Matrix Market file on the root process. */
    err = (rank == root) ? mtxfile_alloc(
        &mtxdistfile->mtxfile, &mtxdistfile->header, &mtxdistfile->comments,
        &sizes[0], mtxdistfile->precision)
        : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err)) {
        if (rank == root)
            free(sizes);
        mtxfile_comments_free(&mtxdistfile->comments);
        if (free_linebuf)
            free(linebuf);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    /* Read each part of the Matrix Market file and send it to the
     * owning process. */
    for (int p = 0; p < comm_size-1; p++) {
        err = (rank == root)
            ? mtxfile_size_copy(&mtxdistfile->mtxfile.size, &sizes[p])
            : MTX_SUCCESS;
        if (mtxmpierror_allreduce(mpierror, err)) {
            if (rank == root || rank < p)
                mtxfile_free(&mtxdistfile->mtxfile);
            if (rank == root)
                free(sizes);
            mtxfile_comments_free(&mtxdistfile->comments);
            if (free_linebuf)
                free(linebuf);
            return MTX_ERR_MPI_COLLECTIVE;
        }

        int64_t num_data_lines;
        err = (rank == root) ? mtxfile_size_num_data_lines(
            &mtxdistfile->mtxfile.size, &num_data_lines) : MTX_SUCCESS;
        if (mtxmpierror_allreduce(mpierror, err)) {
            if (rank == root || rank < p)
                mtxfile_free(&mtxdistfile->mtxfile);
            if (rank == root)
                free(sizes);
            mtxfile_comments_free(&mtxdistfile->comments);
            if (free_linebuf)
                free(linebuf);
            return MTX_ERR_MPI_COLLECTIVE;
        }

        /* Read the next set of data lines on the root process. */
        err = (rank == root) ?
            mtxfile_fread_data(
                &mtxdistfile->mtxfile.data,
                f, lines_read, bytes_read, line_max, linebuf,
                mtxdistfile->mtxfile.header.object, mtxdistfile->mtxfile.header.format,
                mtxdistfile->mtxfile.header.field, mtxdistfile->mtxfile.precision,
                mtxdistfile->mtxfile.size.num_rows,
                mtxdistfile->mtxfile.size.num_columns,
                num_data_lines, 0)
            : MTX_SUCCESS;
        if (mtxmpierror_allreduce(mpierror, err)) {
            if (rank == root || rank < p)
                mtxfile_free(&mtxdistfile->mtxfile);
            if (rank == root)
                free(sizes);
            mtxfile_comments_free(&mtxdistfile->comments);
            if (free_linebuf)
                free(linebuf);
            return MTX_ERR_MPI_COLLECTIVE;
        }

        /* Send to the owning process. */
        if (rank == root) {
            err = mtxfile_send(&mtxdistfile->mtxfile, p, 0, comm, mpierror);
        } else if (rank == p) {
            err = mtxfile_recv(&mtxdistfile->mtxfile, root, 0, comm, mpierror);
        } else {
            err = MTX_SUCCESS;
        }
        if (mtxmpierror_allreduce(mpierror, err)) {
            if (rank == root || rank < p)
                mtxfile_free(&mtxdistfile->mtxfile);
            if (rank == root)
                free(sizes);
            mtxfile_comments_free(&mtxdistfile->comments);
            if (free_linebuf)
                free(linebuf);
            return MTX_ERR_MPI_COLLECTIVE;
        }
    }

    /* Read the final set of data lines on the root process. */
    err = (rank == root)
        ? mtxfile_size_copy(&mtxdistfile->mtxfile.size, &sizes[comm_size-1])
        : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err)) {
        mtxfile_free(&mtxdistfile->mtxfile);
        if (rank == root)
            free(sizes);
        mtxfile_comments_free(&mtxdistfile->comments);
        if (free_linebuf)
            free(linebuf);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    int64_t num_data_lines;
    err = (rank == root) ? mtxfile_size_num_data_lines(
        &mtxdistfile->mtxfile.size, &num_data_lines) : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err)) {
        mtxfile_free(&mtxdistfile->mtxfile);
        if (rank == root)
            free(sizes);
        mtxfile_comments_free(&mtxdistfile->comments);
        if (free_linebuf)
            free(linebuf);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    err = (rank == root) ?
        mtxfile_fread_data(
            &mtxdistfile->mtxfile.data,
            f, lines_read, bytes_read, line_max, linebuf,
            mtxdistfile->mtxfile.header.object, mtxdistfile->mtxfile.header.format,
            mtxdistfile->mtxfile.header.field, mtxdistfile->mtxfile.precision,
            mtxdistfile->mtxfile.size.num_rows, mtxdistfile->mtxfile.size.num_columns,
            num_data_lines, 0)
        : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err)) {
        mtxfile_free(&mtxdistfile->mtxfile);
        if (rank == root)
            free(sizes);
        mtxfile_comments_free(&mtxdistfile->comments);
        if (free_linebuf)
            free(linebuf);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    if (rank == root)
        free(sizes);
    if (free_linebuf)
        free(linebuf);

    mpierror->mpierrcode = MPI_Bcast(lines_read, 1, MPI_INT, root, comm);
    err = mpierror->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err)) {
        mtxdistfile_free(mtxdistfile);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    mpierror->mpierrcode = MPI_Bcast(bytes_read, 1, MPI_INT64_T, root, comm);
    err = mpierror->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err)) {
        mtxdistfile_free(mtxdistfile);
        return MTX_ERR_MPI_COLLECTIVE;
    }
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
