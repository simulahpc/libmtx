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

#ifndef LIBMTX_MTXFILE_COMMENTS_H
#define LIBMTX_MTXFILE_COMMENTS_H

#include <libmtx/libmtx-config.h>

#ifdef LIBMTX_HAVE_LIBZ
#include <zlib.h>
#endif

#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>

/**
 * `mtxfile_comment' represents a single comment line of a Matrix
 * Market file as a node in a doubly linked list of comment lines.
 */
struct mtxfile_comment
{
    struct mtxfile_comment * prev;
    struct mtxfile_comment * next;
    char * comment_line;
};

/**
 * `mtxfile_comment_init()' initialises a comment line.
 */
int mtxfile_comment_init(
    struct mtxfile_comment * comment,
    const char * comment_line);

/**
 * `mtxfile_comment_free()' frees storage used for a comment line.
 */
void mtxfile_comment_free(
    struct mtxfile_comment * comment);

/**
 * `mtxfile_comments' represents a section of comment lines from a
 * Matrix Market file, stored as a doubly linked list of comment
 * lines.
 */
struct mtxfile_comments
{
    struct mtxfile_comment * root;
};

/**
 * `mtxfile_comments_init()' initialises an empty list of comment lines.
 */
int mtxfile_comments_init(
    struct mtxfile_comments * comments);

/**
 * `mtxfile_comments_copy()' copies a list of comment lines.
 */
int mtxfile_comments_copy(
    struct mtxfile_comments * dst,
    const struct mtxfile_comments * src);

/**
 * `mtxfile_comments_free()' frees storage used for comment lines.
 */
void mtxfile_comments_free(
    struct mtxfile_comments * comment);

/**
 * `mtxfile_comments_write()' appends a comment line to a list of
 * comment lines.
 *
 * the comment line must begin with '%' and end with a newline
 * character, '\n'.
 */
int mtxfile_comments_write(
    struct mtxfile_comments * comment,
    const char * comment_line);

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
    struct mtxfile_comments * comment,
    const char * format, ...);

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
    int * bytes_read,
    size_t line_max,
    char * linebuf);

#ifdef LIBMTX_HAVE_LIBZ
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
    int * bytes_read,
    size_t line_max,
    char * linebuf);
#endif

#endif
