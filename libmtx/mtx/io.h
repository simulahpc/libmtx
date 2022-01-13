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
 * Input/output for Matrix Market objects.
 */

#ifndef LIBMTX_MTX_IO_H
#define LIBMTX_MTX_IO_H

#include <libmtx/libmtx-config.h>

#include <libmtx/precision.h>

#ifdef LIBMTX_HAVE_LIBZ
#include <zlib.h>
#endif

#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>

struct mtx;

/**
 * `mtx_read()' reads a `struct mtx' object from a file in Matrix
 * Market format. The file may optionally be compressed by gzip.
 *
 * The `precision' argument specifies which precision to use for
 * storing matrix or vector values.
 *
 * If `path' is `-', then standard input is used.
 *
 * If an error code is returned, then `line_number' and
 * `column_number' are used to return the line and column at which the
 * error was encountered during the parsing of the Matrix Market file.
 */
int mtx_read(
    struct mtx * mtx,
    enum mtxprecision precision,
    const char * path,
    bool gzip,
    int * line_number,
    int * column_number);

/**
 * `mtx_write()' writes a `struct mtx' object from a file in Matrix
 * Market format. The output may optionally be compressed by gzip.
 *
 * If `path' is `-', then standard output is used.
 *
 * If ‘fmt’ is ‘NULL’, then the format specifier ‘%g’ is used to print
 * floating point numbers with enough digits to ensure correct
 * round-trip conversion from decimal text and back.  Otherwise, the
 * given format string is used to print numerical values.
 *
 * The format string follows the conventions of `printf'. If the field
 * is `real', `double' or `complex', then the format specifiers '%e',
 * '%E', '%f', '%F', '%g' or '%G' may be used. If the field is
 * `integer', then the format specifier must be '%d'. The format
 * string is ignored if the field is `pattern'. Field width and
 * precision may be specified (e.g., "%3.1f"), but variable field
 * width and precision (e.g., "%*.*f"), as well as length modifiers
 * (e.g., "%Lf") are not allowed.
 */
int mtx_write(
    const struct mtx * mtx,
    const char * path,
    bool gzip,
    const char * fmt);

/**
 * `mtx_fread()` reads an object (matrix or vector) from a stream in
 * Matrix Market format.
 *
 * The `precision' argument specifies which precision to use for
 * storing matrix or vector values.
 *
 * If an error code is returned, then `line_number' and
 * `column_number' are used to return the line and column at which the
 * error was encountered during the parsing of the Matrix Market file.
 */
int mtx_fread(
    struct mtx * mtx,
    enum mtxprecision precision,
    FILE * f,
    int * line_number,
    int * column_number);

/**
 * `mtx_fwrite()` writes an object (matrix or vector) to a stream in
 * Matrix Market format.
 *
 * If ‘fmt’ is ‘NULL’, then the format specifier ‘%g’ is used to print
 * floating point numbers with enough digits to ensure correct
 * round-trip conversion from decimal text and back.  Otherwise, the
 * given format string is used to print numerical values.
 *
 * The format string follows the conventions of `printf'. If the field
 * is `real', `double' or `complex', then the format specifiers '%e',
 * '%E', '%f', '%F', '%g' or '%G' may be used. If the field is
 * `integer', then the format specifier must be '%d'. The format
 * string is ignored if the field is `pattern'. Field width and
 * precision may be specified (e.g., "%3.1f"), but variable field
 * width and precision (e.g., "%*.*f"), as well as length modifiers
 * (e.g., "%Lf") are not allowed.
 */
int mtx_fwrite(
    const struct mtx * mtx,
    FILE * f,
    const char * fmt);

#ifdef LIBMTX_HAVE_LIBZ
/**
 * `mtx_gzread()` reads a matrix or vector from a gzip-compressed
 * stream in Matrix Market format.
 *
 * The `precision' argument specifies which precision to use for
 * storing matrix or vector values.
 *
 * If an error code is returned, then `line_number' and
 * `column_number' are used to return the line and column at which the
 * error was encountered during the parsing of the Matrix Market file.
 */
int mtx_gzread(
    struct mtx * mtx,
    enum mtxprecision precision,
    gzFile f,
    int * line_number,
    int * column_number);

/**
 * `mtx_gzwrite()` writes a matrix or vector to a gzip-compressed
 * stream in Matrix Market format.
 *
 * If ‘fmt’ is ‘NULL’, then the format specifier ‘%g’ is used to print
 * floating point numbers with enough digits to ensure correct
 * round-trip conversion from decimal text and back.  Otherwise, the
 * given format string is used to print numerical values.
 *
 * The format string follows the conventions of `printf'. If the field
 * is `real', `double' or `complex', then the format specifiers '%e',
 * '%E', '%f', '%F', '%g' or '%G' may be used. If the field is
 * `integer', then the format specifier must be '%d'. The format
 * string is ignored if the field is `pattern'. Field width and
 * precision may be specified (e.g., "%3.1f"), but variable field
 * width and precision (e.g., "%*.*f"), as well as length modifiers
 * (e.g., "%Lf") are not allowed.
 */
int mtx_gzwrite(
    const struct mtx * mtx,
    gzFile f,
    const char * fmt);
#endif

#endif
