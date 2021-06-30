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
 * Last modified: 2021-06-18
 *
 * Input/output for Matrix Market objects.
 */

#ifndef MATRIXMARKET_IO_H
#define MATRIXMARKET_IO_H

#include <matrixmarket/libmtx-config.h>

#ifdef LIBMTX_HAVE_LIBZ
#include <zlib.h>
#endif

#include <stdio.h>

struct mtx;

/**
 * `mtx_fread()` reads an object (matrix or vector) from a stream in
 * Matrix Market format.
 */
int mtx_fread(
    struct mtx * mtx,
    FILE * f,
    int * line_number,
    int * column_number);

/**
 * `mtx_fwrite()` writes an object (matrix or vector) to a stream in
 * Matrix Market format.
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
 */
int mtx_fwrite(
    const struct mtx * mtx,
    FILE * f,
    const char * format);

#ifdef LIBMTX_HAVE_LIBZ
/**
 * `mtx_gzread()` reads a matrix or vector from a gzip-compressed
 * stream in Matrix Market format.
 */
int mtx_gzread(
    struct mtx * mtx,
    gzFile f,
    int * line_number,
    int * column_number);

/**
 * `mtx_gzwrite()` writes a matrix or vector to a gzip-compressed
 * stream in Matrix Market format.
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
 */
int mtx_gzwrite(
    const struct mtx * mtx,
    gzFile f,
    const char * format);
#endif

#endif
