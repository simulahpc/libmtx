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
 * Last modified: 2022-01-20
 *
 * Functions and abstract data types for input/output using standard C
 * library I/O and libz.
 */

#ifndef LIBMTX_UTIL_IO_H
#define LIBMTX_UTIL_IO_H

#include <libmtx/libmtx-config.h>

#ifdef LIBMTX_HAVE_LIBZ
#include <zlib.h>
#endif

#include <stdarg.h>
#include <stdio.h>

struct stream;

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
    FILE * f);

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
    gzFile f);
#endif

int stream_getc(
    const struct stream * stream);

int stream_ungetc(
    int c,
    const struct stream * stream);

int stream_putc(
    int c,
    const struct stream * stream);

int stream_vprintf(
    const struct stream * stream,
    const char * format,
    va_list va);

int stream_printf(
    const struct stream * stream,
    const char * format,
    ...);

/**
 * `stream_read_line()` reads a single line from a stream.
 */
int stream_read_line(
    const struct stream * stream,
    size_t line_max,
    char * linebuf);

#endif
