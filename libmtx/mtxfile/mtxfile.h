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
 * Matrix Market files.
 */

#ifndef LIBMTX_MTXFILE_MTXFILE_H
#define LIBMTX_MTXFILE_MTXFILE_H

#include <libmtx/libmtx-config.h>

#include <libmtx/mtx/precision.h>
#include <libmtx/mtxfile/comments.h>
#include <libmtx/mtxfile/data.h>
#include <libmtx/mtxfile/header.h>
#include <libmtx/mtxfile/size.h>

#ifdef LIBMTX_HAVE_LIBZ
#include <zlib.h>
#endif

#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>

/**
 * `mtxfile' represents a file in the Matrix Market file format.
 */
struct mtxfile
{
    /**
     * `header' is the Matrix Market file header.
     */
    struct mtxfile_header header;

    /**
     * `comments' is the Matrix Market comment lines.
     */
    struct mtxfile_comments comments;

    /**
     * `size' is the Matrix Market size line.
     */
    struct mtxfile_size size;

    /**
     * `precision' is the precision used to store the values of the
     * Matrix Market data lines.
     */
    enum mtx_precision precision;

    /**
     * `data' contains the data lines of the Matrix Market file.
     */
    union mtxfile_data data;
};

/**
 * `mtxfile_free()' frees storage allocated for a Matrix Market file.
 */
void mtxfile_free(
    struct mtxfile * mtxfile);

/**
 * `mtxfile_copy()' copies a Matrix Market file.
 */
int mtxfile_copy(
    struct mtxfile * dst,
    const struct mtxfile * src);

/**
 * `mtxfile_fread()` reads a Matrix Market file from a stream.
 *
 * `precision' is used to determine the precision to use for storing
 * the values of matrix or vector entries.
 *
 * If an error code is returned, then `lines_read' and `bytes_read'
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the Matrix Market file.
 */
int mtxfile_fread(
    struct mtxfile * mtxfile,
    FILE * f,
    int * lines_read,
    int * bytes_read,
    size_t line_max,
    char * linebuf,
    enum mtx_precision precision);

#ifdef LIBMTX_HAVE_LIBZ
/**
 * `mtxfile_gzread()` reads a Matrix Market file from a
 * gzip-compressed stream.
 *
 * `precision' is used to determine the precision to use for storing
 * the values of matrix or vector entries.
 *
 * If an error code is returned, then `lines_read' and `bytes_read'
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the Matrix Market file.
 */
int mtxfile_gzread(
    struct mtxfile * mtxfile,
    gzFile f,
    int * lines_read,
    int * bytes_read,
    size_t line_max,
    char * linebuf,
    enum mtx_precision precision);
#endif

#endif
