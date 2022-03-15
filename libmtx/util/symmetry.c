/* This file is part of Libmtx.
 *
 * Copyright (C) 2022 James D. Trotter
 *
 * Libmtx is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Libmtx is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Libmtx.  If not, see <https://www.gnu.org/licenses/>.
 *
 * Authors: James D. Trotter <james@simula.no>
 * Last modified: 2022-03-15
 *
 * Symmetry properties of matrices.
 */

#include <libmtx/error.h>
#include <libmtx/util/symmetry.h>

#include <stdint.h>
#include <string.h>

/**
 * ‘mtxsymmetry_str()’ is a string representing the symmetry type.
 */
const char * mtxsymmetry_str(
    enum mtxsymmetry symmetry)
{
    switch (symmetry) {
    case mtx_unsymmetric: return "unsymmetric";
    case mtx_symmetric: return "symmetric";
    case mtx_skew_symmetric: return "skew-symmetric";
    case mtx_hermitian: return "hermitian";
    default: return mtxstrerror(MTX_ERR_INVALID_TRANSPOSITION);
    }
}

/**
 * ‘mtxsymmetry_parse()’ parses a string to obtain one of the symmetry
 * types of ‘enum mtxsymmetry’.
 *
 * ‘valid_delimiters’ is either ‘NULL’, in which case it is ignored,
 * or it is a string of characters considered to be valid delimiters
 * for the parsed string.  That is, if there are any remaining,
 * non-NULL characters after parsing, then then the next character is
 * searched for in ‘valid_delimiters’.  If the character is found,
 * then the parsing succeeds and the final delimiter character is
 * consumed by the parser. Otherwise, the parsing fails with an error.
 *
 * If ‘endptr’ is not ‘NULL’, then the address stored in ‘endptr’
 * points to the first character beyond the characters that were
 * consumed during parsing.
 *
 * On success, ‘mtxsymmetry_parse()’ returns ‘MTX_SUCCESS’ and
 * ‘symmetry’ is set according to the parsed string and ‘bytes_read’
 * is set to the number of bytes that were consumed by the parser.
 * Otherwise, an error code is returned.
 */
int mtxsymmetry_parse(
    enum mtxsymmetry * symmetry,
    int64_t * bytes_read,
    const char ** endptr,
    const char * s,
    const char * valid_delimiters)
{
    const char * t = s;
    if (strncmp("unsymmetric", t, strlen("unsymmetric")) == 0) {
        t += strlen("unsymmetric");
        *symmetry = mtx_unsymmetric;
    } else if (strncmp("symmetric", t, strlen("symmetric")) == 0) {
        t += strlen("symmetric");
        *symmetry = mtx_symmetric;
    } else if (strncmp("skew-symmetric", t, strlen("skew-symmetric")) == 0) {
        t += strlen("skew-symmetric");
        *symmetry = mtx_skew_symmetric;
    } else if (strncmp("hermitian", t, strlen("hermitian")) == 0) {
        t += strlen("hermitian");
        *symmetry = mtx_hermitian;
    } else {
        return MTX_ERR_INVALID_TRANSPOSITION;
    }
    if (valid_delimiters && *t != '\0') {
        if (!strchr(valid_delimiters, *t))
            return MTX_ERR_INVALID_TRANSPOSITION;
        t++;
    }
    if (bytes_read)
        *bytes_read += t-s;
    if (endptr)
        *endptr = t;
    return MTX_SUCCESS;
}
