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
 * Utility functions for reading and writing Matrix Market files.
 */

#ifndef MATRIXMARKET_IOUTIL_H
#define MATRIXMARKET_IOUTIL_H

#include <stdbool.h>

struct mtx;

/**
 * `read_mtx()' reads a `struct mtx' object from a file in Matrix
 * Market format. The file may optionally be compressed by gzip.
 */
int read_mtx(
    const char * mtx_path,
    bool gzip,
    struct mtx * mtx,
    int verbose,
    int * line_number,
    int * column_number);

/**
 * `write_mtx()' writes a `struct mtx' object from a file in Matrix
 * Market format. The output may optionally be compressed by gzip.
 */
int write_mtx(
    const char * mtx_path,
    bool gzip,
    const struct mtx * mtx,
    const char * format,
    int verbose);

#endif
