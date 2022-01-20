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
 * Last modified: 2022-01-09
 *
 * Index sets.
 */

#include <libmtx/libmtx-config.h>

#include <libmtx/error.h>
#include <libmtx/mtxfile/mtxfile.h>
#include <libmtx/util/index_set.h>

#include <errno.h>
#include <unistd.h>

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * ‘mtxidxsettype_str()’ is a string representing the index set
 * type.
 */
const char * mtxidxsettype_str(
    enum mtxidxsettype index_set_type)
{
    switch (index_set_type) {
    case mtxidxset_interval: return "interval";
    case mtxidxset_strided: return "strided";
    case mtxidxset_blockstrided: return "block-strided";
    case mtxidxset_array: return "array";
    default: return mtxstrerror(MTX_ERR_INVALID_INDEX_SET_TYPE);
    }
}

/**
 * ‘mtxidxset_free()’ frees resources associated with an index
 * set.
 */
void mtxidxset_free(
    struct mtxidxset * index_set)
{
    free(index_set->indices);
}

/**
 * ‘mtxidxset_init_interval()’ creates an index set of contiguous
 * integers from an interval [a,b).
 */
int mtxidxset_init_interval(
    struct mtxidxset * index_set,
    int64_t a,
    int64_t b)
{
    index_set->type = mtxidxset_interval;
    index_set->size = b - a;
    index_set->offset = a;
    index_set->stride = 0;
    index_set->block_size = 0;
    index_set->indices = NULL;
    return MTX_SUCCESS;
}

/**
 * ‘mtxidxset_init_strided()’ creates an index set of strided
 * integers from an interval:
 *
 *   ‘offset,offset+stride,offset+2*stride,...,offset+(size-1)*stride’.
 */
int mtxidxset_init_strided(
    struct mtxidxset * index_set,
    int64_t offset,
    int64_t size,
    int stride)
{
    index_set->type = mtxidxset_strided;
    index_set->size = size;
    index_set->offset = offset;
    index_set->stride = stride;
    index_set->block_size = 0;
    index_set->indices = NULL;
    return MTX_SUCCESS;
}

/**
 * ‘mtxidxset_init_blockstrided()’ creates an index set of
 * fixed-size blocks separated by a stride:
 */
int mtxidxset_init_blockstrided(
    struct mtxidxset * index_set,
    int64_t offset,
    int64_t size,
    int stride,
    int block_size)
{
    index_set->type = mtxidxset_strided;
    index_set->size = size;
    index_set->offset = offset;
    index_set->stride = stride;
    index_set->block_size = block_size;
    index_set->indices = NULL;
    return MTX_SUCCESS;
}

/**
 * ‘mtxidxset_init_array()’ creates an index set of integer values
 * given by an array.
 */
int mtxidxset_init_array(
    struct mtxidxset * index_set,
    int64_t size,
    const int64_t * indices)
{
    index_set->type = mtxidxset_array;
    index_set->size = size;
    index_set->offset = 0;
    index_set->stride = 0;
    index_set->block_size = 0;
    index_set->indices = malloc(size * sizeof(int64_t));
    if (!index_set->indices)
        return MTX_ERR_ERRNO;
    for (int64_t i = 0; i < size; i++)
        index_set->indices[i] = indices[i];
    return MTX_SUCCESS;
}

/**
 * ‘mtxidxset_contains()‘ returns ‘true’ if the given integer is
 * contained in the index set and ‘false’ otherwise.
 */
bool mtxidxset_contains(
    const struct mtxidxset * index_set,
    int64_t n)
{
    if (index_set->type == mtxidxset_interval) {
        return (n >= index_set->offset) && (n < index_set->offset + index_set->size);
    } else if (index_set->type == mtxidxset_strided) {
        return (n >= index_set->offset)
            && (n < index_set->stride * index_set->size)
            && ((n - index_set->offset) % index_set->stride == 0);
    } else if (index_set->type == mtxidxset_blockstrided) {
        /* TODO: Not implemented. */
        return false;
    } else if (index_set->type == mtxidxset_array) {
        for (int64_t i = 0; i < index_set->size; i++) {
            if (index_set->indices[i] == n)
                return true;
        }
        return false;
    } else {
        return false;
    }
}

/**
 * ‘mtxidxset_read()’ reads an index set from the given path as a
 * Matrix Market file in the form of an integer vector in array
 * format.
 *
 * If ‘path’ is ‘-’, then standard input is used.
 *
 * If an error code is returned, then ‘lines_read’ and ‘bytes_read’
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the Matrix Market file.
 */
int mtxidxset_read(
    struct mtxidxset * index_set,
    const char * path,
    int * lines_read,
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
    err = mtxidxset_fread(
        index_set, f, lines_read, bytes_read, 0, NULL);
    if (err) {
        fclose(f);
        return err;
    }
    fclose(f);
    return MTX_SUCCESS;
}

/**
 * ‘mtxidxset_fread()’ reads an index set from a stream as a Matrix
 * Market file in the form of an integer vector in array format.
 *
 * If an error code is returned, then ‘lines_read’ and ‘bytes_read’
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the Matrix Market file.
 */
int mtxidxset_fread(
    struct mtxidxset * index_set,
    FILE * f,
    int * lines_read,
    int64_t * bytes_read,
    size_t line_max,
    char * linebuf)
{
    int err;
    struct mtxfile mtxfile;
    err = mtxfile_fread(
        &mtxfile, mtx_double, f, lines_read, bytes_read, line_max, linebuf);
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

    err = mtxidxset_init_array(
        index_set, mtxfile.size.num_rows, mtxfile.data.array_integer_double);
    if (err) {
        mtxfile_free(&mtxfile);
        return err;
    }
    mtxfile_free(&mtxfile);
    return MTX_SUCCESS;
}

/**
 * ‘mtxidxset_write()’ writes an index set to the given path as a
 * Matrix Market file in the form of an integer vector in array
 * format.
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
int mtxidxset_write(
    const struct mtxidxset * index_set,
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
    err = mtxidxset_fwrite(index_set, f, format, bytes_written);
    if (err) {
        fclose(f);
        return err;
    }
    fclose(f);
    return MTX_SUCCESS;
}

/**
 * ‘mtxidxset_fwrite()’ writes an index set to a stream as a
 * Matrix Market file in the form of an integer vector in array
 * format.
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
int mtxidxset_fwrite(
    const struct mtxidxset * index_set,
    FILE * f,
    const char * format,
    int64_t * bytes_written)
{
    int err;
    struct mtxfile mtxfile;
    err = mtxfile_alloc_vector_array(
        &mtxfile, mtxfile_integer, mtx_double, index_set->size);
    if (err)
        return err;

    int64_t * indices = mtxfile.data.array_integer_double;
    if (index_set->type == mtxidxset_interval) {
        for (int64_t i = 0; i < index_set->size; i++)
            indices[i] = index_set->offset + i;
    } else if (index_set->type == mtxidxset_strided) {
        for (int64_t i = 0; i < index_set->size; i++)
            indices[i] = index_set->offset + i * index_set->stride;
    } else if (index_set->type == mtxidxset_blockstrided) {
        mtxfile_free(&mtxfile);
        errno = ENOTSUP;
        return MTX_ERR_ERRNO;
    } else if (index_set->type == mtxidxset_array) {
        for (int64_t i = 0; i < index_set->size; i++)
            indices[i] = index_set->indices[i];
    } else {
        mtxfile_free(&mtxfile);
        return MTX_ERR_INVALID_INDEX_SET_TYPE;
    }

    err = mtxfile_fwrite(&mtxfile, f, format, bytes_written);
    if (err) {
        mtxfile_free(&mtxfile);
        return err;
    }
    mtxfile_free(&mtxfile);
    return MTX_SUCCESS;
}

/*
 * MPI functions
 */

#ifdef LIBMTX_HAVE_MPI
/**
 * ‘mtxidxset_send()’ sends an index set to another MPI process.
 *
 * This is analogous to ‘MPI_Send()’ and requires the receiving
 * process to perform a matching call to ‘mtxidxset_recv()’.
 */
int mtxidxset_send(
    const struct mtxidxset * index_set,
    int dest,
    int tag,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    disterr->err = MPI_Send(
        &index_set->type, 1, MPI_INT, dest, tag, comm);
    if (disterr->err)
        return MTX_ERR_MPI;
    disterr->err = MPI_Send(
        &index_set->size, 1, MPI_INT64_T, dest, tag, comm);
    if (disterr->err)
        return MTX_ERR_MPI;
    disterr->err = MPI_Send(
        &index_set->offset, 1, MPI_INT64_T, dest, tag, comm);
    if (disterr->err)
        return MTX_ERR_MPI;
    disterr->err = MPI_Send(
        &index_set->stride, 1, MPI_INT, dest, tag, comm);
    if (disterr->err)
        return MTX_ERR_MPI;
    disterr->err = MPI_Send(
        &index_set->block_size, 1, MPI_INT, dest, tag, comm);
    if (disterr->err)
        return MTX_ERR_MPI;
    if (index_set->type == mtxidxset_array) {
        disterr->err = MPI_Send(
            &index_set->indices, index_set->size, MPI_INT64_T, dest, tag, comm);
        if (disterr->err)
            return MTX_ERR_MPI;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxidxset_recv()’ receives an index set from another MPI
 * process.
 *
 * This is analogous to ‘MPI_Recv()’ and requires the sending process
 * to perform a matching call to ‘mtxidxset_send()’.
 */
int mtxidxset_recv(
    struct mtxidxset * index_set,
    int source,
    int tag,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    disterr->err = MPI_Recv(
        &index_set->type, 1, MPI_INT, source, tag, comm, MPI_STATUS_IGNORE);
    if (disterr->err)
        return MTX_ERR_MPI;
    disterr->err = MPI_Recv(
        &index_set->size, 1, MPI_INT64_T, source, tag, comm, MPI_STATUS_IGNORE);
    if (disterr->err)
        return MTX_ERR_MPI;
    disterr->err = MPI_Recv(
        &index_set->offset, 1, MPI_INT64_T, source, tag, comm, MPI_STATUS_IGNORE);
    if (disterr->err)
        return MTX_ERR_MPI;
    disterr->err = MPI_Recv(
        &index_set->stride, 1, MPI_INT, source, tag, comm, MPI_STATUS_IGNORE);
    if (disterr->err)
        return MTX_ERR_MPI;
    disterr->err = MPI_Recv(
        &index_set->block_size, 1, MPI_INT, source, tag, comm, MPI_STATUS_IGNORE);
    if (disterr->err)
        return MTX_ERR_MPI;
    if (index_set->type == mtxidxset_array) {
        index_set->indices = malloc(index_set->size * sizeof(int64_t));
        if (!index_set->indices)
            return MTX_ERR_ERRNO;
        disterr->err = MPI_Recv(
            &index_set->indices, index_set->size, MPI_INT64_T,
            source, tag, comm, MPI_STATUS_IGNORE);
        if (disterr->err)
            return MTX_ERR_MPI;
    }
    return MTX_SUCCESS;
}
#endif
