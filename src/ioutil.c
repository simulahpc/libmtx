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
 * Utility functions for reading Matrix Market files.
 */

#include <matrixmarket/libmtx-config.h>

#include <matrixmarket/error.h>
#include <matrixmarket/io.h>

#ifdef LIBMTX_HAVE_LIBZ
#include <zlib.h>
#endif

#include <errno.h>

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    int * column_number)
{
    int err;
    *line_number = -1;
    *column_number = -1;

    if (!gzip) {
        FILE * f;
        if (strcmp(mtx_path, "-") == 0) {
            f = stdin;
        } else if ((f = fopen(mtx_path, "r")) == NULL) {
            return MTX_ERR_ERRNO;
        }

        err = mtx_read(mtx, f, line_number, column_number);
        if (err)
            return err;
        fclose(f);
    } else {
#ifdef LIBMTX_HAVE_LIBZ
        gzFile f;
        if (strcmp(mtx_path, "-") == 0) {
            f = gzdopen(STDIN_FILENO, "r");
        } else if ((f = gzopen(mtx_path, "r")) == NULL) {
            return MTX_ERR_ERRNO;
        }

        err = mtx_gzread(mtx, f, line_number, column_number);
        if (err)
            return err;
        gzclose(f);
#else
        errno = ENOTSUP;
        return MTX_ERR_ERRNO;
#endif
    }
    return MTX_SUCCESS;
}

/**
 * `write_mtx()' write a `struct mtx' object from a file in Matrix
 * Market format. The file may optionally be compressed by gzip.
 */
int write_mtx(
    const char * mtx_path,
    bool gzip,
    const struct mtx * mtx,
    const char * format,
    int verbose)
{
    int err;
    if (!gzip) {
        FILE * f;
        if (strcmp(mtx_path, "-") == 0) {
            f = stdout;
        } else if ((f = fopen(mtx_path, "w")) == NULL) {
            return MTX_ERR_ERRNO;
        }

        err = mtx_write(mtx, f, format);
        if (err)
            return err;
        fclose(f);
    } else {
#ifdef LIBMTX_HAVE_LIBZ
        gzFile f;
        if (strcmp(mtx_path, "-") == 0) {
            f = gzdopen(STDOUT_FILENO, "w");
        } else if ((f = gzopen(mtx_path, "w")) == NULL) {
            return MTX_ERR_ERRNO;
        }

        err = mtx_gzwrite(mtx, f, format);
        if (err)
            return err;
        gzclose(f);
#else
        errno = ENOTSUP;
        return MTX_ERR_ERRNO;
#endif
    }
    return MTX_SUCCESS;
}
