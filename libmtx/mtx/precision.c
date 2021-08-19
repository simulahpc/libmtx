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
 * Last modified: 2021-08-16
 *
 * Precision of data types used to store matrices and vectors.
 */

#include <libmtx/mtx/precision.h>

/**
 * `mtx_precision_str()' is a string representing the given precision
 * type.
 */
const char * mtx_precision_str(
    enum mtx_precision precision)
{
    switch (precision) {
    case mtx_single: return "single";
    case mtx_double: return "double";
    default: return "unknown";
    }
}
