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

#ifdef HAVE_LIBZ
#include <zlib.h>
#endif

#include <stdio.h>

struct mtx;

/**
 * `mtx_read()` reads an object (matrix or vector) from a stream in
 * Matrix Market format.
 */
int mtx_read(
    struct mtx * mtx,
    FILE * f,
    int * line_number,
    int * column_number);

/**
 * `mtx_write()` writes an object (matrix or vector) to a stream in
 * Matrix Market format.
 */
int mtx_write(
    const struct mtx * mtx,
    FILE * f,
    int field_width,
    int precision);

#ifdef HAVE_LIBZ
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
 */
int mtx_gzwrite(
    const struct mtx * mtx,
    gzFile f,
    int field_width,
    int precision);
#endif

#endif
