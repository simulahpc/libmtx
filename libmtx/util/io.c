/* This file is part of libmtx.
 *
 * Copyright (C) 2021 James D. Trotter
 *
 * libmtx is free software: you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * libmtx is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with libmtx.  If not, see
 * <https://www.gnu.org/licenses/>.
 *
 * Authors: James D. Trotter <james@simula.no>
 * Last modified: 2021-08-09
 *
 * Functions and abstract data types for input/output using standard C
 * library I/O and libz.
 */

#include "io.h"

#include <libmtx/libmtx-config.h>
#include <libmtx/error.h>

#ifdef LIBMTX_HAVE_LIBZ
#include <zlib.h>
#endif

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * `stream_type' is used to enumerate different kinds of file streams.
 */
enum stream_type
{
    stream_stdio,   /* C standard librario I/O stream */
#ifdef LIBMTX_HAVE_LIBZ
    stream_gz       /* libz gzip-compressed I/O stream */
#endif
};

/**
 * `stream' is used to abstract the underlying I/O stream, so that we
 * can easily use standard C library I/O or libz.
 */
struct stream
{
    enum stream_type type;
    FILE * stdio_f;
#ifdef LIBMTX_HAVE_LIBZ
    gzFile gz_f;
#endif
};

/**
 * `stream_init_stdio()' creates a stream to read from or write to a C
 * standard I/O file stream.
 *
 * On success, a pointer to a newly allocated stream is returned.
 * When they are finished, the user must call `free()' with the
 * pointer that was returned by `stream_init_stdio()' to free the
 * allocated storage.
 *
 * If an error occurs, `NULL' is returned and `errno' is set to
 * indicate an appropriate error code.
 */
struct stream * stream_init_stdio(
    FILE * f)
{
    struct stream * stream = malloc(sizeof(struct stream));
    if (!stream)
        return NULL;
    stream->type = stream_stdio;
    stream->stdio_f = f;
    return stream;
}

#ifdef LIBMTX_HAVE_LIBZ
/**
 * `stream_init_gz()' creates a stream to read from or write to a libz
 * gzip-compressed file stream.
 *
 * On success, a pointer to a newly allocated stream is returned.
 * When they are finished, the user must call `free()' with the
 * pointer that was returned by `stream_init_gz()' to free the
 * allocated storage.
 *
 * If an error occurs, `NULL' is returned and `errno' is set to
 * indicate an appropriate error code.
 */
struct stream * stream_init_gz(
    gzFile f)
{
    struct stream * stream = malloc(sizeof(struct stream));
    if (!stream)
        return NULL;
    stream->type = stream_gz;
    stream->gz_f = f;
    return stream;
}
#endif

int stream_getc(
    const struct stream * stream)
{
    if (stream->type == stream_stdio) {
        FILE * f = stream->stdio_f;
        return fgetc(f);
#ifdef LIBMTX_HAVE_LIBZ
    } else if (stream->type == stream_gz) {
        gzFile f = stream->gz_f;
        return gzgetc(f);
#endif
    } else {
        return MTX_ERR_INVALID_STREAM_TYPE;
    }
}

int stream_ungetc(
    int c,
    const struct stream * stream)
{
    if (stream->type == stream_stdio) {
        FILE * f = stream->stdio_f;
        return ungetc(c, f);
#ifdef LIBMTX_HAVE_LIBZ
    } else if (stream->type == stream_gz) {
        gzFile f = stream->gz_f;
        return gzungetc(c, f);
#endif
    } else {
        return MTX_ERR_INVALID_STREAM_TYPE;
    }
}

int stream_putc(
    int c,
    const struct stream * stream)
{
    if (stream->type == stream_stdio) {
        FILE * f = stream->stdio_f;
        return fputc(c, f);
#ifdef LIBMTX_HAVE_LIBZ
    } else if (stream->type == stream_gz) {
        gzFile f = stream->gz_f;
        return gzputc(f, c);
#endif
    } else {
        return MTX_ERR_INVALID_STREAM_TYPE;
    }
}

int stream_vprintf(
    const struct stream * stream,
    const char * format,
    va_list va)
{
    if (stream->type == stream_stdio) {
        FILE * f = stream->stdio_f;
        return vfprintf(f, format, va);
#ifdef LIBMTX_HAVE_LIBZ
    } else if (stream->type == stream_gz) {
        gzFile f = stream->gz_f;
        return gzvprintf(f, format, va);
#endif
    } else {
        return MTX_ERR_INVALID_STREAM_TYPE;
    }
}

int stream_printf(
    const struct stream * stream,
    const char * format,
    ...)
{
    int err;
    va_list va;
    va_start(va, format);
    err = stream_vprintf(stream, format, va);
    va_end(va);
    return err;
}

/**
 * `stream_read_line()` reads a single line from a stream.
 */
int stream_read_line(
    const struct stream * stream,
    size_t line_max,
    char * linebuf)
{
    if (stream->type == stream_stdio) {
        FILE * f = stream->stdio_f;
        char * s = fgets(linebuf, line_max+1, f);
        if (!s && feof(f))
            return MTX_ERR_EOF;
        else if (!s)
            return MTX_ERR_ERRNO;
        int n = strlen(s);
        if (n > 0 && n == line_max && s[n-1] != '\n')
            return MTX_ERR_LINE_TOO_LONG;
#ifdef LIBMTX_HAVE_LIBZ
    } else if (stream->type == stream_gz) {
        gzFile f = stream->gz_f;
        char * s = gzgets(f, linebuf, line_max+1);
        if (!s && gzeof(f))
            return MTX_ERR_EOF;
        else if (!s)
            return MTX_ERR_ERRNO;
        int n = strlen(s);
        if (n > 0 && n == line_max && s[n-1] != '\n')
            return MTX_ERR_LINE_TOO_LONG;
#endif
    } else {
        return MTX_ERR_INVALID_STREAM_TYPE;
    }
    return MTX_SUCCESS;
}

