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

#include <libmtx/mtx/header.h>

#include <libmtx/error.h>

#include <stdlib.h>
#include <string.h>

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

/**
 * `mtx_header_parse()' parses a string containing the header line for
 * a file in Matrix Market format.
 *
 * If `endptr' is not `NULL', then the address stored in `endptr'
 * points to the first character beyond the characters that were
 * consumed during parsing.
 *
 * On success, `mtx_header_parse()` returns `MTX_SUCCESS' and
 * `object', `format', `field' and `symmetry' will be set according to
 * the contents of the parsed Matrix Market header.  Otherwise, an
 * appropriate error code is returned if the input is not a valid
 * Matrix Market header.
 */
int mtx_header_parse(
    const char * s,
    int * bytes_read,
    const char ** endptr,
    enum mtx_object * object,
    enum mtx_format * format,
    enum mtx_field * field,
    enum mtx_symmetry * symmetry)
{
    int err;
    const char * t = s;

    *bytes_read = 0;
    /* Parse the identifier. */
    if (strncmp("%%MatrixMarket", t, strlen("%%MatrixMarket")) != 0)
        return MTX_ERR_INVALID_MTX_HEADER;
    while (*t != '\0' && *t != ' ')
        t++;
    while (*t == ' ')
        t++;
    *bytes_read = t-s;

    /* Parse the object type. */
    if (strncmp("matrix ", t, strlen("matrix ")) == 0) {
        *object = mtx_matrix;
    } else if (strncmp("vector ", t, strlen("vector ")) == 0) {
        *object = mtx_vector;
    } else {
        return MTX_ERR_INVALID_MTX_OBJECT;
    }
    while (*t != '\0' && *t != ' ')
        t++;
    while (*t == ' ')
        t++;
    *bytes_read = t-s;

    /* Parse the format. */
    if (strncmp("array ", t, strlen("array ")) == 0) {
        *format = mtx_array;
    } else if (strncmp("coordinate ", t, strlen("coordinate ")) == 0) {
        *format = mtx_coordinate;
    } else {
        return MTX_ERR_INVALID_MTX_FORMAT;
    }
    while (*t != '\0' && *t != ' ')
        t++;
    while (*t == ' ')
        t++;
    *bytes_read = t-s;

    /* Parse the field type. */
    if (strncmp("real ", t, strlen("real ")) == 0) {
        *field = mtx_real;
    } else if (strncmp("double ", t, strlen("double ")) == 0) {
        *field = mtx_double;
    } else if (strncmp("complex ", t, strlen("complex ")) == 0) {
        *field = mtx_complex;
    } else if (strncmp("integer ", t, strlen("integer ")) == 0) {
        *field = mtx_integer;
    } else if (strncmp("pattern ", t, strlen("pattern ")) == 0) {
        *field = mtx_pattern;
    } else {
        return MTX_ERR_INVALID_MTX_FIELD;
    }
    while (*t != '\0' && *t != ' ')
        t++;
    while (*t == ' ')
        t++;
    *bytes_read = t-s;

    /* Parse the symmetry type. */
    if (strcmp("general", t) == 0 || strcmp("general\n", t) == 0) {
        *symmetry = mtx_general;
    } else if (strcmp("symmetric", t) == 0 || strcmp("symmetric\n", t) == 0) {
        *symmetry = mtx_symmetric;
    } else if (strcmp("skew-symmetric", t) == 0 ||
               strcmp("skew-symmetric\n", t) == 0)
    {
        *symmetry = mtx_skew_symmetric;
    } else if (strcmp("hermitian", t) == 0 || strcmp("hermitian\n", t) == 0 ||
               strcmp("Hermitian", t) == 0 || strcmp("Hermitian\n", t) == 0)
    {
        *symmetry = mtx_hermitian;
    } else {
        return MTX_ERR_INVALID_MTX_SYMMETRY;
    }
    while (*t != '\0' && *t != '\n')
        t++;
    *bytes_read = t-s;

    if (endptr)
        *endptr = t;
    return MTX_SUCCESS;
}
