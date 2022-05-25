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
 * Last modified: 2022-05-22
 *
 * Matrix partitionings.
 */

#include <libmtx/error.h>
#include <libmtx/matrix/partition.h>

#include <stdint.h>
#include <string.h>

/**
 * ‘mtxmatrixparttype_str()’ is a string representing the matrix
 * partitioning type.
 */
const char * mtxmatrixparttype_str(
    enum mtxmatrixparttype matrixparttype)
{
    switch (matrixparttype) {
    case mtx_matrixparttype_nonzeros: return "nonzeros";
    case mtx_matrixparttype_rows: return "rows";
    case mtx_matrixparttype_columns: return "columns";
    case mtx_matrixparttype_2d: return "2d";
    case mtx_matrixparttype_metis: return "metis";
    default: return mtxstrerror(MTX_ERR_INVALID_MATRIXPARTTYPE);
    }
}

/**
 * ‘mtxmatrixparttype_parse()’ parses a string to obtain one of the
 * matrix partitioning types of ‘enum mtxmatrixparttype’.
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
 * On success, ‘mtxmatrixparttype_parse()’ returns ‘MTX_SUCCESS’ and
 * ‘matrixparttype’ is set according to the parsed string and
 * ‘bytes_read’ is set to the number of bytes that were consumed by
 * the parser.  Otherwise, an error code is returned.
 */
int mtxmatrixparttype_parse(
    enum mtxmatrixparttype * matrixparttype,
    int64_t * bytes_read,
    const char ** endptr,
    const char * s,
    const char * valid_delimiters)
{
    const char * t = s;
    if (strncmp("nonzeros", t, strlen("nonzeros")) == 0) {
        t += strlen("nonzeros");
        *matrixparttype = mtx_matrixparttype_nonzeros;
    } else if (strncmp("rows", t, strlen("rows")) == 0) {
        t += strlen("rows");
        *matrixparttype = mtx_matrixparttype_rows;
    } else if (strncmp("columns", t, strlen("columns")) == 0) {
        t += strlen("columns");
        *matrixparttype = mtx_matrixparttype_columns;
    } else if (strncmp("2d", t, strlen("2d")) == 0) {
        t += strlen("2d");
        *matrixparttype = mtx_matrixparttype_2d;
    } else if (strncmp("metis", t, strlen("metis")) == 0) {
        t += strlen("metis");
        *matrixparttype = mtx_matrixparttype_metis;
    } else { return MTX_ERR_INVALID_MATRIXPARTTYPE; }
    if (valid_delimiters && *t != '\0') {
        if (!strchr(valid_delimiters, *t))
            return MTX_ERR_INVALID_MATRIXPARTTYPE;
        t++;
    }
    if (bytes_read) *bytes_read += t-s;
    if (endptr) *endptr = t;
    return MTX_SUCCESS;
}
