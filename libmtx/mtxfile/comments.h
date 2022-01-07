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
 * Last modified: 2022-01-04
 *
 * Matrix Market comment lines.
 */

#ifndef LIBMTX_MTXFILE_COMMENTS_H
#define LIBMTX_MTXFILE_COMMENTS_H

#include <libmtx/libmtx-config.h>

#ifdef LIBMTX_HAVE_MPI
#include <mpi.h>
#endif

#ifdef LIBMTX_HAVE_LIBZ
#include <zlib.h>
#endif

#include <inttypes.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>

struct mtxdisterror;

/**
 * `mtxfilecomment' represents a single comment line of a Matrix
 * Market file as a node in a doubly linked list of comment lines.
 */
struct mtxfilecomment
{
    struct mtxfilecomment * prev;
    struct mtxfilecomment * next;
    char * comment_line;
};

/**
 * `mtxfilecomment_init()' initialises a comment line.
 */
int mtxfilecomment_init(
    struct mtxfilecomment * comment,
    const char * comment_line);

/**
 * `mtxfilecomment_free()' frees storage used for a comment line.
 */
void mtxfilecomment_free(
    struct mtxfilecomment * comment);

/**
 * `mtxfilecomments' represents a section of comment lines from a
 * Matrix Market file, stored as a doubly linked list of comment
 * lines.
 */
struct mtxfilecomments
{
    struct mtxfilecomment * root;
};

/**
 * `mtxfilecomments_init()' initialises an empty list of comment lines.
 */
int mtxfilecomments_init(
    struct mtxfilecomments * comments);

/**
 * `mtxfilecomments_copy()' copies a list of comment lines.
 */
int mtxfilecomments_copy(
    struct mtxfilecomments * dst,
    const struct mtxfilecomments * src);

/**
 * `mtxfilecomments_cat()' concatenates a list of comment lines to
 * another list of comment lines.
 */
int mtxfilecomments_cat(
    struct mtxfilecomments * dst,
    const struct mtxfilecomments * src);

/**
 * `mtxfilecomments_free()' frees storage used for comment lines.
 */
void mtxfilecomments_free(
    struct mtxfilecomments * comment);

/**
 * `mtxfilecomments_write()' appends a comment line to a list of
 * comment lines.
 *
 * the comment line must begin with '%' and end with a newline
 * character, '\n'.
 */
int mtxfilecomments_write(
    struct mtxfilecomments * comment,
    const char * comment_line);

/**
 * `mtxfilecomments_printf()' appends a comment line to a list of
 * comment lines using a printf-like syntax.
 *
 * Note that because `format' is a printf-style format string, where
 * '%' is used to denote a format specifier, then `format' must begin
 * with "%%" to produce the initial '%' character that is required for
 * a comment line.  The `format' string must also end with a newline
 * character, '\n'.
 */
int mtxfilecomments_printf(
    struct mtxfilecomments * comment,
    const char * format, ...);

/*
 * I/O functions
 */

/**
 * `mtxfile_fread_comments()' reads Matrix Market comment lines from a
 * stream.
 *
 * If an error code is returned, then `lines_read' and `bytes_read'
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the Matrix Market file.
 */
int mtxfile_fread_comments(
    struct mtxfilecomments * comments,
    FILE * f,
    int * lines_read,
    int64_t * bytes_read,
    size_t line_max,
    char * linebuf);

#ifdef LIBMTX_HAVE_LIBZ
/**
 * `mtxfile_gzread_comments()' reads Matrix Market comment lines from
 * a gzip-compressed stream.
 *
 * If an error code is returned, then `lines_read' and `bytes_read'
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the Matrix Market file.
 */
int mtxfile_gzread_comments(
    struct mtxfilecomments * comments,
    gzFile f,
    int * lines_read,
    int64_t * bytes_read,
    size_t line_max,
    char * linebuf);
#endif

/**
 * `mtxfilecomments_fputs()' write Matrix Market comment lines to a
 * stream.
 */
int mtxfilecomments_fputs(
    const struct mtxfilecomments * comments,
    FILE * f,
    int64_t * bytes_written);

#ifdef LIBMTX_HAVE_LIBZ
/**
 * `mtxfilecomments_gzputs()' write Matrix Market comment lines to a
 * gzip-compressed stream.
 */
int mtxfilecomments_gzputs(
    const struct mtxfilecomments * comments,
    gzFile f,
    int64_t * bytes_written);
#endif

/*
 * MPI functions
 */

#ifdef LIBMTX_HAVE_MPI
/**
 * `mtxfilecomments_send()' sends Matrix Market comment lines to
 * another MPI process.
 *
 * This is analogous to `MPI_Send()' and requires the receiving
 * process to perform a matching call to `mtxfilecomments_recv()'.
 */
int mtxfilecomments_send(
    const struct mtxfilecomments * comments,
    int dest,
    int tag,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * `mtxfilecomments_recv()' receives Matrix Market comment lines from
 * another MPI process.
 *
 * This is analogous to `MPI_Recv()' and requires the sending process
 * to perform a matching call to `mtxfilecomments_send()'.
 */
int mtxfilecomments_recv(
    struct mtxfilecomments * comments,
    int source,
    int tag,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * `mtxfilecomments_bcast()' broadcasts Matrix Market comment lines
 * from an MPI root process to other processes in a communicator.
 *
 * This is analogous to `MPI_Bcast()' and requires every process in
 * the communicator to perform matching calls to
 * `mtxfilecomments_bcast()'.
 */
int mtxfilecomments_bcast(
    struct mtxfilecomments * comments,
    int root,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * `mtxfilecomments_gather()' gathers Matrix Market comments onto an
 * MPI root process from other processes in a communicator.
 *
 * This is analogous to `MPI_Gather()' and requires every process in
 * the communicator to perform matching calls to
 * `mtxfilecomments_gather()'.
 */
int mtxfilecomments_gather(
    const struct mtxfilecomments * sendcomments,
    struct mtxfilecomments * recvcomments,
    int root,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * `mtxfilecomments_allgather()' gathers Matrix Market comment lines
 * onto every MPI process from other processes in a communicator.
 *
 * This is analogous to `MPI_Allgather()' and requires every process
 * in the communicator to perform matching calls to this function.
 */
int mtxfilecomments_allgather(
    const struct mtxfilecomments * sendcomments,
    struct mtxfilecomments * recvcomments,
    MPI_Comm comm,
    struct mtxdisterror * disterr);
#endif

#endif
