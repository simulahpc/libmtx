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
 * Last modified: 2021-10-07
 *
 * Different ways of transposing matrices and vectors.
 */

#include <libmtx/error.h>
#include <libmtx/util/transpose.h>

#include <stdint.h>
#include <string.h>

/**
 * `mtx_trans_type_str()' is a string representing the transpose type.
 */
const char * mtx_trans_type_str_(
    enum mtx_trans_type transpose)
{
    switch (transpose) {
    case mtx_notrans: return "notrans";
    case mtx_trans: return "trans";
    case mtx_conjtrans: return "conjtrans";
    default: return mtxstrerror(MTX_ERR_INVALID_TRANS_TYPE);
    }
}

/**
 * `mtx_trans_type_parse()' parses a string to obtain one of the
 * transpose types of `enum mtx_trans_type'.
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
 * On success, `mtx_trans_type_parse()' returns `MTX_SUCCESS' and
 * `transpose' is set according to the parsed string and `bytes_read'
 * is set to the number of bytes that were consumed by the parser.
 * Otherwise, an error code is returned.
 */
int mtx_trans_type_parse(
    enum mtx_trans_type * transpose,
    int64_t * bytes_read,
    const char ** endptr,
    const char * s,
    const char * valid_delimiters)
{
    const char * t = s;
    if (strncmp("notrans", t, strlen("notrans")) == 0) {
        t += strlen("notrans");
        *transpose = mtx_notrans;
    } else if (strncmp("trans", t, strlen("trans")) == 0) {
        t += strlen("trans");
        *transpose = mtx_trans;
    } else if (strncmp("conjtrans", t, strlen("conjtrans")) == 0) {
        t += strlen("conjtrans");
        *transpose = mtx_conjtrans;
    } else {
        return MTX_ERR_INVALID_TRANS_TYPE;
    }
    if (valid_delimiters && *t != '\0') {
        if (!strchr(valid_delimiters, *t))
            return MTX_ERR_INVALID_TRANS_TYPE;
        t++;
    }
    if (bytes_read)
        *bytes_read += t-s;
    if (endptr)
        *endptr = t;
    return MTX_SUCCESS;
}
