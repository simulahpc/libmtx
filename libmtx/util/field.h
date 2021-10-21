/* This file is part of libmtx.
 *
 * Copyright (C) 2021 James D. Trotter
 *
 * libmtx is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * libmtx is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with libmtx.  If not, see <https://www.gnu.org/licenses/>.
 *
 * Authors: James D. Trotter <james@simula.no>
 * Last modified: 2021-09-19
 *
 * Fields for values of matrices and vectors.
 */

#ifndef LIBMTX_UTIL_FIELD_H
#define LIBMTX_UTIL_FIELD_H

#include <stdint.h>

/**
 * `mtx_field' is used to enumerate fields for representing matrix and
 * vector values.
 */
enum mtx_field_
{
    mtx_field_auto,    /* automatic selection of field */
    mtx_field_real,    /* real, floating-point coefficients */
    mtx_field_complex, /* complex, floating point coefficients */
    mtx_field_integer, /* integer coefficients */
    mtx_field_pattern  /* boolean coefficients (sparsity pattern) */
};

/**
 * `mtx_field_str()' is a string representing the field type.
 */
const char * mtx_field_str_(
    enum mtx_field_ field);

/**
 * `mtx_field_parse()' parses a string to obtain one of the field
 * types of `enum mtx_field'.
 *
 * `valid_delimiters' is either `NULL', in which case it is ignored,
 * or it is a string of characters considered to be valid delimiters
 * for the parsed string.  That is, if there are any remaining,
 * non-NULL characters after parsing, then then the next character is
 * searched for in `valid_delimiters'.  If the character is found,
 * then the parsing succeeds and the final delimiter character is
 * consumed by the parser. Otherwise, the parsing fails with an error.
 *
 * If `endptr' is not `NULL', then the address stored in `endptr'
 * points to the first character beyond the characters that were
 * consumed during parsing.
 *
 * On success, `mtx_field_parse()' returns `MTX_SUCCESS' and `field'
 * is set according to the parsed string and `bytes_read' is set to
 * the number of bytes that were consumed by the parser.  Otherwise,
 * an error code is returned.
 */
int mtx_field_parse(
    enum mtx_field_ * field,
    int64_t * bytes_read,
    const char ** endptr,
    const char * s,
    const char * valid_delimiters);

#endif
