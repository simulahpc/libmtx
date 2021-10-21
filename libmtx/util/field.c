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

#include <libmtx/error.h>
#include <libmtx/util/field.h>

#include <stdint.h>
#include <string.h>

/**
 * `mtx_field_str()' is a string representing the field type.
 */
const char * mtx_field_str_(
    enum mtx_field_ field)
{
    switch (field) {
    case mtx_field_auto: return "auto";
    case mtx_field_real: return "real";
    case mtx_field_complex: return "complex";
    case mtx_field_integer: return "integer";
    case mtx_field_pattern: return "pattern";
    default: return mtx_strerror(MTX_ERR_INVALID_FIELD);
    }
}

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
    const char * valid_delimiters)
{
    const char * t = s;
    if (strncmp("auto", t, strlen("auto")) == 0) {
        t += strlen("auto");
        *field = mtx_field_auto;
    } else if (strncmp("real", t, strlen("real")) == 0) {
        t += strlen("real");
        *field = mtx_field_real;
    } else if (strncmp("complex", t, strlen("complex")) == 0) {
        t += strlen("complex");
        *field = mtx_field_complex;
    } else if (strncmp("integer", t, strlen("integer")) == 0) {
        t += strlen("integer");
        *field = mtx_field_integer;
    } else if (strncmp("pattern", t, strlen("pattern")) == 0) {
        t += strlen("pattern");
        *field = mtx_field_pattern;
    } else {
        return MTX_ERR_INVALID_FIELD;
    }
    if (valid_delimiters && *t != '\0') {
        if (!strchr(valid_delimiters, *t))
            return MTX_ERR_INVALID_FIELD;
        t++;
    }
    if (bytes_read)
        *bytes_read += t-s;
    if (endptr)
        *endptr = t;
    return MTX_SUCCESS;
}
