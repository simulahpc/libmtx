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
 * Last modified: 2021-10-08
 *
 * Precision of data types used to store matrices and vectors.
 */

#include <libmtx/error.h>
#include <libmtx/precision.h>

#include <stddef.h>
#include <string.h>

/**
 * `mtxprecision_str()' is a string representing the given precision
 * type.
 */
const char * mtxprecision_str(
    enum mtxprecision precision)
{
    switch (precision) {
    case mtx_single: return "single";
    case mtx_double: return "double";
    default: return mtxstrerror(MTX_ERR_INVALID_PRECISION);
    }
}

/**
 * `mtxprecision_parse()' parses a string to obtain one of the
 * precision types of `enum mtxprecision'.
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
 * On success, `mtxprecision_parse()' returns `MTX_SUCCESS' and
 * `precision' is set according to the parsed string and `bytes_read'
 * is set to the number of bytes that were consumed by the parser.
 * Otherwise, an error code is returned.
 */
int mtxprecision_parse(
    enum mtxprecision * precision,
    int64_t * bytes_read,
    const char ** endptr,
    const char * s,
    const char * valid_delimiters)
{
    const char * t = s;
    if (strncmp("single", t, strlen("single")) == 0) {
        t += strlen("single");
        *precision = mtx_single;
    } else if (strncmp("double", t, strlen("double")) == 0) {
        t += strlen("double");
        *precision = mtx_double;
    } else {
        return MTX_ERR_INVALID_PRECISION;
    }
    if (valid_delimiters && *t != '\0') {
        if (!strchr(valid_delimiters, *t))
            return MTX_ERR_INVALID_PRECISION;
        t++;
    }
    if (bytes_read)
        *bytes_read += t-s;
    if (endptr)
        *endptr = t;
    return MTX_SUCCESS;
}
