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
 * Data types for the Matrix Market header.
 */

#include <libmtx/header.h>

/**
 * `mtx_object_str()` is a string representing the Matrix Market object
 * type.
 */
const char * mtx_object_str(
    enum mtx_object object)
{
    switch (object) {
    case mtx_matrix: return "matrix";
    case mtx_vector: return "vector";
    default: return "unknown";
    }
}

/**
 * `mtx_format_str()` is a string representing the Matrix Market format
 * type.
 */
const char * mtx_format_str(
    enum mtx_format format)
{
    switch (format) {
    case mtx_array: return "array";
    case mtx_coordinate: return "coordinate";
    default: return "unknown";
    }
}

/**
 * `mtx_field_str()` is a string representing the Matrix Market field
 * type.
 */
const char * mtx_field_str(
    enum mtx_field field)
{
    switch (field) {
    case mtx_real: return "real";
    case mtx_double: return "double";
    case mtx_complex: return "complex";
    case mtx_integer: return "integer";
    case mtx_pattern: return "pattern";
    default: return "unknown";
    }
}

/**
 * `mtx_symmetry_str()` is a string representing the Matrix Market
 * symmetry type.
 */
const char * mtx_symmetry_str(
    enum mtx_symmetry symmetry)
{
    switch (symmetry) {
    case mtx_general: return "general";
    case mtx_symmetric: return "symmetric";
    case mtx_skew_symmetric: return "skew-symmetric";
    case mtx_hermitian: return "hermitian";
    default: return "unknown";
    }
}
