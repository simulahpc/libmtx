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
 * Input/output for dense vectors in array format.
 */

#include <libmtx/libmtx-config.h>

#include <libmtx/error.h>
#include <libmtx/mtx/io.h>
#include <libmtx/mtx/mtx.h>
#include <libmtx/vector/array/array.h>

#include "libmtx/util/format.h"
#include "libmtx/util/parse.h"

#include <errno.h>
#include <unistd.h>

#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * `mtx_vector_array_parse_size()` parse a size line from a Matrix
 * Market file for a vector in array format.
 */
int mtx_vector_array_parse_size(
    const char * line,
    int * bytes_read,
    const char ** endptr,
    enum mtx_object object,
    enum mtx_format format,
    enum mtx_field field,
    enum mtx_symmetry symmetry,
    int * num_rows,
    int * num_columns,
    int64_t * num_nonzeros,
    int64_t * size,
    int * nonzero_size)
{
    int err;
    *bytes_read = 0;
    if (object != mtx_vector)
        return MTX_ERR_INVALID_MTX_OBJECT;
    if (format != mtx_array)
        return MTX_ERR_INVALID_MTX_FORMAT;

    /* Parse the number of rows. */
    err = parse_int32(line, "\n", num_rows, endptr);
    if (err == EINVAL) {
        return MTX_ERR_INVALID_MTX_SIZE;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    }
    *bytes_read = (*endptr) - line;

    *num_columns = -1;
    *num_nonzeros = *num_rows;
    *size = *num_nonzeros;
    if (field == mtx_real) {
        *nonzero_size = sizeof(float);
    } else if (field == mtx_double) {
        *nonzero_size = sizeof(double);
    } else if (field == mtx_complex) {
        *nonzero_size = 2*sizeof(float);
    } else if (field == mtx_integer) {
        *nonzero_size = sizeof(int);
    } else {
        return MTX_ERR_INVALID_MTX_FIELD;
    }
    return MTX_SUCCESS;
}
