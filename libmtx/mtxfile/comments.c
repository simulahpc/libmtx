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
 * Last modified: 2021-09-01
 *
 * Matrix Market comment lines.
 */

#include <libmtx/libmtx-config.h>

#include <libmtx/error.h>
#include <libmtx/mtxfile/comments.h>

#ifdef LIBMTX_HAVE_MPI
#include <mpi.h>
#endif

#ifdef LIBMTX_HAVE_LIBZ
#include <zlib.h>
#endif

#include <unistd.h>

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * `mtxfile_comment_validate()' validates a comment line.
 *
 * A comment line must be a non-empty, null-terminated string,
 * beginning with '%' and ending with '\n'.  Moreover, no other
 * newline characters are allowed other than the final, ending
 * newline.
 *
 * If the comment line is valid, then `MTX_SUCCESS' is
 * returned. Otherwise, `MTX_ERR_INVALID_MTX_COMMENT' is returned.
 */
static int mtxfile_comment_validate(
    const char * comment_line)
{
    int n = strlen(comment_line);
    if (n <= 1 || comment_line[0] != '%' || comment_line[n-1] != '\n')
        return MTX_ERR_INVALID_MTX_COMMENT;
    if (strchr(comment_line, '\n') != &comment_line[n-1])
        return MTX_ERR_INVALID_MTX_COMMENT;
    return MTX_SUCCESS;
}

/**
 * `mtxfile_comment_init()' initialises a comment line.
 */
int mtxfile_comment_init(
    struct mtxfile_comment * comment,
    const char * comment_line)
{
    int err = mtxfile_comment_validate(comment_line);
    if (err)
        return err;

    comment->prev = NULL;
    comment->next = NULL;
    comment->comment_line = strdup(comment_line);
    if (!comment->comment_line)
        return MTX_ERR_ERRNO;
    return MTX_SUCCESS;
}

/**
 * `mtxfile_comments_init()' initialises an empty list of comment lines.
 */
int mtxfile_comments_init(
    struct mtxfile_comments * comments)
{
    comments->root = NULL;
    return MTX_SUCCESS;
}

/**
 * `mtxfile_comments_copy()' copies a list of comment lines.
 */
int mtxfile_comments_copy(
    struct mtxfile_comments * dst,
    const struct mtxfile_comments * src)
{
    int err = mtxfile_comments_init(dst);
    if (err)
        return err;

    const struct mtxfile_comment * node = src->root;
    while (node) {
        err = mtxfile_comments_write(dst, node->comment_line);
        if (err) {
            mtxfile_comments_free(dst);
            return err;
        }
        node = node->next;
    }
    return MTX_SUCCESS;
}

/**
 * `mtxfile_comments_cat()' concatenates a list of comment lines to
 * another list of comment lines.
 */
int mtxfile_comments_cat(
    struct mtxfile_comments * dst,
    const struct mtxfile_comments * src)
{
    const struct mtxfile_comment * node = src->root;
    while (node) {
        int err = mtxfile_comments_write(dst, node->comment_line);
        if (err)
            return err;
        node = node->next;
    }
    return MTX_SUCCESS;
}

/**
 * `mtxfile_comment_free()' frees storage used for a comment line.
 */
void mtxfile_comment_free(
    struct mtxfile_comment * comment)
{
    free(comment->comment_line);
}

/**
 * `mtxfile_comments_free()' frees storage used for comment lines.
 */
void mtxfile_comments_free(
    struct mtxfile_comments * comments)
{
    if (!comments->root)
        return;
    struct mtxfile_comment * node = comments->root;
    while (node->next)
        node = node->next;
    while (node) {
        struct mtxfile_comment * prev = node->prev;
        mtxfile_comment_free(node);
        free(node);
        node = prev;
    }
}

static int mtxfile_comments_push_back(
    struct mtxfile_comments * comments,
    const char * comment_line)
{
    struct mtxfile_comment * comment = malloc(sizeof(struct mtxfile_comment));
    if (!comment)
        return MTX_ERR_ERRNO;
    int err = mtxfile_comment_init(comment, comment_line);
    if (err) {
        free(comment);
        return err;
    }

    if (comments->root == NULL) {
        comments->root = comment;
    } else {
        struct mtxfile_comment * tail = comments->root;
        while (tail->next)
            tail = tail->next;
        comment->prev = tail;
        tail->next = comment;
    }
    return MTX_SUCCESS;
}

/**
 * `mtxfile_comments_write()' appends a comment line to a list of
 * comment lines.
 *
 * the comment line must begin with '%' and end with a newline
 * character, '\n'.
 */
int mtxfile_comments_write(
    struct mtxfile_comments * comments,
    const char * comment_line)
{
    return mtxfile_comments_push_back(
        comments, comment_line);
}

/**
 * `mtxfile_comments_printf()' appends a comment line to a list of
 * comment lines using a printf-like syntax.
 *
 * Note that because `format' is a printf-style format string, where
 * '%' is used to denote a format specifier, then `format' must begin
 * with "%%" to produce the initial '%' character that is required for
 * a comment line.  The `format' string must also end with a newline
 * character, '\n'.
 */
int mtxfile_comments_printf(
    struct mtxfile_comments * comments,
    const char * format, ...)
{
    int err;
    va_list va;
    va_start(va, format);
    int len = vsnprintf(NULL, 0, format, va);
    va_end(va);
    if (len < 0)
        return MTX_ERR_ERRNO;

    char * s = malloc(len+1);
    if (!s)
        return MTX_ERR_ERRNO;

    va_start(va, format);
    int newlen = vsnprintf(s, len+1, format, va);
    va_end(va);
    if (newlen < 0 || len != newlen) {
        free(s);
        return MTX_ERR_ERRNO;
    }
    s[newlen] = '\0';

    err = mtxfile_comments_write(comments, s);
    free(s);
    return err;
}

/**
 * `freadline()' reads a single line from a stream.
 */
static int freadline(
    char * linebuf,
    size_t line_max,
    FILE * f)
{
    char * s = fgets(linebuf, line_max+1, f);
    if (!s && feof(f))
        return MTX_ERR_EOF;
    else if (!s)
        return MTX_ERR_ERRNO;
    int n = strlen(s);
    if (n > 0 && n == line_max && s[n-1] != '\n')
        return MTX_ERR_LINE_TOO_LONG;
    return MTX_SUCCESS;
}

/**
 * `mtxfile_fread_comments()` reads Matrix Market comment lines from a
 * stream.
 *
 * If an error code is returned, then `lines_read' and `bytes_read'
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the Matrix Market file.
 */
int mtxfile_fread_comments(
    struct mtxfile_comments * comments,
    FILE * f,
    int * lines_read,
    int64_t * bytes_read,
    size_t line_max,
    char * linebuf)
{
    int err;
    err = mtxfile_comments_init(comments);
    if (err)
        return err;

    bool free_linebuf = !linebuf;
    if (!linebuf) {
        line_max = sysconf(_SC_LINE_MAX);
        linebuf = malloc(line_max+1);
        if (!linebuf)
            return MTX_ERR_ERRNO;
    }

    while (true) {
        int c = fgetc(f);
        c = ungetc(c, f);

        /* Stop parsing comments on end-of-file or if the line does
         * not start with '%'. */
        if (c == EOF || c != '%')
            break;

        /* Read the next line as a comment line. */
        err = freadline(linebuf, line_max, f);
        if (err) {
            mtxfile_comments_free(comments);
            if (free_linebuf)
                free(linebuf);
            return err;
        }

        err = mtxfile_comments_write(comments, linebuf);
        if (err) {
            mtxfile_comments_free(comments);
            if (free_linebuf)
                free(linebuf);
            return err;
        }

        if (lines_read)
            (*lines_read)++;
        if (bytes_read)
            (*bytes_read) += strlen(linebuf);
    }

    if (free_linebuf)
        free(linebuf);
    return MTX_SUCCESS;
}

#ifdef LIBMTX_HAVE_LIBZ
/**
 * `gzreadline()' reads a single line from a gzip-compressed stream.
 */
static int gzreadline(
    char * linebuf,
    size_t line_max,
    gzFile f)
{
    char * s = gzgets(f, linebuf, line_max+1);
    if (!s && gzeof(f))
        return MTX_ERR_EOF;
    else if (!s)
        return MTX_ERR_ERRNO;
    int n = strlen(s);
    if (n > 0 && n == line_max && s[n-1] != '\n')
        return MTX_ERR_LINE_TOO_LONG;
    return MTX_SUCCESS;
}

/**
 * `mtxfile_gzread_comments()` reads Matrix Market comment lines from
 * a gzip-compressed stream.
 *
 * If an error code is returned, then `lines_read' and `bytes_read'
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the Matrix Market file.
 */
int mtxfile_gzread_comments(
    struct mtxfile_comments * comments,
    gzFile f,
    int * lines_read,
    int64_t * bytes_read,
    size_t line_max,
    char * linebuf)
{
    int err;
    err = mtxfile_comments_init(comments);
    if (err)
        return err;

    bool free_linebuf = !linebuf;
    if (!linebuf) {
        line_max = sysconf(_SC_LINE_MAX);
        linebuf = malloc(line_max+1);
        if (!linebuf)
            return MTX_ERR_ERRNO;
    }

    while (true) {
        int c = gzgetc(f);
        c = gzungetc(c, f);

        /* Stop parsing comments on end-of-file or if the line does
         * not start with '%'. */
        if (c == EOF || c != '%')
            break;

        /* Read the next line as a comment line. */
        err = gzreadline(linebuf, line_max, f);
        if (err) {
            mtxfile_comments_free(comments);
            if (free_linebuf)
                free(linebuf);
            return err;
        }

        err = mtxfile_comments_write(comments, linebuf);
        if (err) {
            mtxfile_comments_free(comments);
            if (free_linebuf)
                free(linebuf);
            return err;
        }

        if (lines_read)
            (*lines_read)++;
        if (bytes_read)
            (*bytes_read) += strlen(linebuf);
    }

    if (free_linebuf)
        free(linebuf);
    return MTX_SUCCESS;
}
#endif

/**
 * `mtxfile_comments_fputs()` write Matrix Market comment lines to a
 * stream.
 */
int mtxfile_comments_fputs(
    const struct mtxfile_comments * comments,
    FILE * f,
    int64_t * bytes_written)
{
    const struct mtxfile_comment * node = comments->root;
    while (node) {
        if (fputs(node->comment_line, f) == EOF)
            return MTX_ERR_ERRNO;
        if (bytes_written)
            *bytes_written += strlen(node->comment_line);
        node = node->next;
    }
    return MTX_SUCCESS;
}

/*
 * MPI functions
 */

#ifdef LIBMTX_HAVE_MPI
/**
 * `mtxfile_comments_send()' sends Matrix Market comment lines to
 * another MPI process.
 *
 * This is analogous to `MPI_Send()' and requires the receiving
 * process to perform a matching call to `mtxfile_comments_recv()'.
 */
int mtxfile_comments_send(
    const struct mtxfile_comments * comments,
    int dest,
    int tag,
    MPI_Comm comm,
    struct mtxmpierror * mpierror)
{
    int err;
    int num_comments = 0;
    const struct mtxfile_comment * node;
    for (node = comments->root; node; node = node->next)
        num_comments++;

    mpierror->err = MPI_Send(&num_comments, 1, MPI_INT, dest, tag, comm);
    if (mpierror->err)
        return MTX_ERR_MPI;

    for (node = comments->root; node; node = node->next) {
        int len = strlen(node->comment_line);
        mpierror->err = MPI_Send(&len, 1, MPI_INT, dest, tag, comm);
        if (mpierror->err)
            return MTX_ERR_MPI;
        mpierror->err = MPI_Send(node->comment_line, len, MPI_CHAR, dest, tag, comm);
        if (mpierror->err)
            return MTX_ERR_MPI;
    }
    return MTX_SUCCESS;
}

/**
 * `mtxfile_comments_recv()' receives Matrix Market comment lines from
 * another MPI process.
 *
 * This is analogous to `MPI_Recv()' and requires the sending process
 * to perform a matching call to `mtxfile_comments_send()'.
 */
int mtxfile_comments_recv(
    struct mtxfile_comments * comments,
    int source,
    int tag,
    MPI_Comm comm,
    struct mtxmpierror * mpierror)
{
    int err;
    int num_comments;
    mpierror->err = MPI_Recv(
        &num_comments, 1, MPI_INT, source, tag, comm, MPI_STATUS_IGNORE);
    if (mpierror->err)
        return MTX_ERR_MPI;

    for (int i = 0; i < num_comments; i++) {
        int len;
        mpierror->err = MPI_Recv(
            &len, 1, MPI_INT, source, tag, comm, MPI_STATUS_IGNORE);
        if (mpierror->err) {
            mtxfile_comments_free(comments);
            return MTX_ERR_MPI;
        }

        char * comment_line = malloc((len+1) * sizeof(char));
        if (!comment_line) {
            mtxfile_comments_free(comments);
            return MTX_ERR_ERRNO;
        }

        mpierror->err = MPI_Recv(
            comment_line, len, MPI_CHAR, source, tag, comm, MPI_STATUS_IGNORE);
        if (mpierror->err) {
            free(comment_line);
            mtxfile_comments_free(comments);
            return MTX_ERR_MPI;
        }
        comment_line[len] = '\0';

        err = mtxfile_comments_write(comments, comment_line);
        if (err) {
            free(comment_line);
            mtxfile_comments_free(comments);
            return err;
        }
        free(comment_line);
    }
    return MTX_SUCCESS;
}

/**
 * `mtxfile_comments_bcast()' broadcasts Matrix Market comment lines
 * from an MPI root process to other processes in a communicator.
 *
 * This is analogous to `MPI_Bcast()' and requires every process in
 * the communicator to perform matching calls to
 * `mtxfile_comments_bcast()'.
 */
int mtxfile_comments_bcast(
    struct mtxfile_comments * comments,
    int root,
    MPI_Comm comm,
    struct mtxmpierror * mpierror)
{
    int err;
    int rank;
    err = MPI_Comm_rank(comm, &rank);
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;

    int num_comments;
    if (rank == root) {
        num_comments = 0;
        const struct mtxfile_comment * node;
        for (node = comments->root; node; node = node->next)
            num_comments++;
    }

    err = MPI_Bcast(&num_comments, 1, MPI_INT, root, comm);
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;

    const struct mtxfile_comment * node;
    if (rank == root)
        node = comments->root;
    else
        mtxfile_comments_init(comments);
    for (int i = 0; i < num_comments; i++) {

        int len;
        if (rank == root)
            len = strlen(node->comment_line);

        err = MPI_Bcast(&len, 1, MPI_INT, root, comm);
        if (mtxmpierror_allreduce(mpierror, err)) {
            mtxfile_comments_free(comments);
            return MTX_ERR_MPI_COLLECTIVE;
        }

        char * comment_line = NULL;
        if (rank == root)
            comment_line = node->comment_line;
        else
            comment_line = malloc((len+1) * sizeof(char));

        err = !comment_line ? MTX_ERR_ERRNO : MTX_SUCCESS;
        if (mtxmpierror_allreduce(mpierror, err)) {
            if (rank != root && comment_line)
                free(comment_line);
            if (rank != root)
                mtxfile_comments_free(comments);
            return MTX_ERR_MPI_COLLECTIVE;
        }

        err = MPI_Bcast(comment_line, len, MPI_CHAR, root, comm);
        if (mtxmpierror_allreduce(mpierror, err)) {
            if (rank != root) {
                free(comment_line);
                mtxfile_comments_free(comments);
            }
            return MTX_ERR_MPI_COLLECTIVE;
        }
        comment_line[len] = '\0';

        if (rank != root)
            err = mtxfile_comments_write(comments, comment_line);
        if (mtxmpierror_allreduce(mpierror, err)) {
            if (rank != root) {
                free(comment_line);
                mtxfile_comments_free(comments);
            }
            return MTX_ERR_MPI_COLLECTIVE;
        }
        if (rank != root)
            free(comment_line);

        if (rank == root)
            node = node->next;
    }
    return MTX_SUCCESS;
}
#endif
