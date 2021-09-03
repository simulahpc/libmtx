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

#include <libmtx/libmtx-config.h>

#include <libmtx/error.h>
#include <libmtx/mtx/precision.h>
#include <libmtx/mtxfile/comments.h>
#include <libmtx/mtxfile/data.h>
#include <libmtx/mtxfile/header.h>
#include <libmtx/mtxfile/mtxfile.h>
#include <libmtx/mtxfile/size.h>

#ifdef LIBMTX_HAVE_LIBZ
#include <zlib.h>
#endif

#include <errno.h>
#include <unistd.h>

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * `mtxfile_free()' frees storage allocated for a Matrix Market file.
 */
void mtxfile_free(
    struct mtxfile * mtxfile)
{
    mtxfile_data_free(
        &mtxfile->data,
        mtxfile->header.object,
        mtxfile->header.format,
        mtxfile->header.field,
        mtxfile->precision);
    mtxfile_comments_free(&mtxfile->comments);
}

/**
 * `mtxfile_copy()' copies a Matrix Market file.
 */
int mtxfile_copy(
    struct mtxfile * dst,
    const struct mtxfile * src)
{
    int err;
    err = mtxfile_header_copy(&dst->header, &src->header);
    if (err)
        return err;
    err = mtxfile_comments_copy(&dst->comments, &src->comments);
    if (err)
        return err;
    err = mtxfile_size_copy(&dst->size, &src->size);
    if (err) {
        mtxfile_comments_free(&dst->comments);
        return err;
    }

    size_t num_data_lines;
    err = mtxfile_size_num_data_lines(
        &src->size, &num_data_lines);
    if (err) {
        mtxfile_comments_free(&dst->comments);
        return err;
    }

    err = mtxfile_data_copy(
        &dst->data, &src->data,
        src->header.object, src->header.format,
        src->header.field, src->precision, num_data_lines);
    if (err) {
        mtxfile_comments_free(&dst->comments);
        return err;
    }
    return MTX_SUCCESS;
}

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
    enum mtx_precision precision)
{
    int err;
    bool free_linebuf = !linebuf;
    if (!linebuf) {
        line_max = sysconf(_SC_LINE_MAX);
        linebuf = malloc(line_max+1);
        if (!linebuf)
            return MTX_ERR_ERRNO;
    }

    err = mtxfile_fread_header(
        &mtxfile->header, f, lines_read, bytes_read, line_max, linebuf);
    if (err) {
        if (free_linebuf)
            free(linebuf);
        return err;
    }

    err = mtxfile_fread_comments(
        &mtxfile->comments, f, lines_read, bytes_read, line_max, linebuf);
    if (err) {
        if (free_linebuf)
            free(linebuf);
        return err;
    }

    err = mtxfile_fread_size(
        &mtxfile->size, f, lines_read, bytes_read, line_max, linebuf,
        mtxfile->header.object, mtxfile->header.format);
    if (err) {
        mtxfile_comments_free(&mtxfile->comments);
        if (free_linebuf)
            free(linebuf);
        return err;
    }

    size_t num_data_lines;
    err = mtxfile_size_num_data_lines(
        &mtxfile->size, &num_data_lines);
    if (err) {
        mtxfile_comments_free(&mtxfile->comments);
        if (free_linebuf)
            free(linebuf);
        return err;
    }

    err = mtxfile_data_alloc(
        &mtxfile->data,
        mtxfile->header.object,
        mtxfile->header.format,
        mtxfile->header.field,
        precision, num_data_lines);
    if (err) {
        mtxfile_comments_free(&mtxfile->comments);
        if (free_linebuf)
            free(linebuf);
        return err;
    }

    err = mtxfile_fread_data(
        &mtxfile->data, f, lines_read, bytes_read, line_max, linebuf,
        mtxfile->header.object,
        mtxfile->header.format,
        mtxfile->header.field,
        precision,
        mtxfile->size.num_rows,
        mtxfile->size.num_columns,
        num_data_lines);
    if (err) {
        mtxfile_data_free(
            &mtxfile->data, mtxfile->header.object,
            mtxfile->header.format, mtxfile->header.field,
            precision);
        mtxfile_comments_free(&mtxfile->comments);
        if (free_linebuf)
            free(linebuf);
        return err;
    }

    mtxfile->precision = precision;
    if (free_linebuf)
        free(linebuf);
    return MTX_SUCCESS;
}

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
    enum mtx_precision precision)
{
    int err;
    bool free_linebuf = !linebuf;
    if (!linebuf) {
        line_max = sysconf(_SC_LINE_MAX);
        linebuf = malloc(line_max+1);
        if (!linebuf)
            return MTX_ERR_ERRNO;
    }

    err = mtxfile_gzread_header(
        &mtxfile->header, f, lines_read, bytes_read, line_max, linebuf);
    if (err) {
        if (free_linebuf)
            free(linebuf);
        return err;
    }

    err = mtxfile_gzread_comments(
        &mtxfile->comments, f, lines_read, bytes_read, line_max, linebuf);
    if (err) {
        if (free_linebuf)
            free(linebuf);
        return err;
    }

    err = mtxfile_gzread_size(
        &mtxfile->size, f, lines_read, bytes_read, line_max, linebuf,
        mtxfile->header.object, mtxfile->header.format);
    if (err) {
        mtxfile_comments_free(&mtxfile->comments);
        if (free_linebuf)
            free(linebuf);
        return err;
    }

    size_t num_data_lines;
    err = mtxfile_size_num_data_lines(
        &mtxfile->size, &num_data_lines);
    if (err) {
        mtxfile_comments_free(&mtxfile->comments);
        if (free_linebuf)
            free(linebuf);
        return err;
    }

    err = mtxfile_data_alloc(
        &mtxfile->data,
        mtxfile->header.object,
        mtxfile->header.format,
        mtxfile->header.field,
        precision, num_data_lines);
    if (err) {
        mtxfile_comments_free(&mtxfile->comments);
        if (free_linebuf)
            free(linebuf);
        return err;
    }

    err = mtxfile_gzread_data(
        &mtxfile->data, f, lines_read, bytes_read, line_max, linebuf,
        mtxfile->header.object,
        mtxfile->header.format,
        mtxfile->header.field,
        precision,
        mtxfile->size.num_rows,
        mtxfile->size.num_columns,
        num_data_lines);
    if (err) {
        mtxfile_data_free(
            &mtxfile->data, mtxfile->header.object,
            mtxfile->header.format, mtxfile->header.field,
            precision);
        mtxfile_comments_free(&mtxfile->comments);
        if (free_linebuf)
            free(linebuf);
        return err;
    }

    mtxfile->precision = precision;
    if (free_linebuf)
        free(linebuf);
    return MTX_SUCCESS;
}
#endif
