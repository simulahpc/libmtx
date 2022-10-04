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
 * Last modified: 2022-10-03
 *
 * Different ways of overlapping communication with computation for
 * distributed matrix-vector multiply.
 */

#ifndef LIBMTX_LINALG_GEMVOVERLAP_H
#define LIBMTX_LINALG_GEMVOVERLAP_H

#include <stdint.h>

/**
 * ‘mtxgemvoverlap’ is used to enumerate different ways of overlapping
 * communication with computation when performing distributed
 * matrix-vector multiplication.
 */
enum mtxgemvoverlap
{
    mtxgemvoverlap_none,    /* no overlap */
    mtxgemvoverlap_irecv,   /* overlap using non-blocking receive (MPI_Irecv) */
};

/**
 * ‘mtxgemvoverlap_str()’ is a string representing the overlap type.
 */
const char * mtxgemvoverlap_str(
    enum mtxgemvoverlap overlap);

/**
 * ‘mtxgemvoverlap_parse()’ parses a string to obtain one of the
 * overlap types of ‘enum mtxgemvoverlap’.
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
 * On success, ‘mtxgemvoverlap_parse()’ returns ‘MTX_SUCCESS’ and
 * ‘overlap’ is set according to the parsed string and ‘bytes_read’ is
 * set to the number of bytes that were consumed by the parser.
 * Otherwise, an error code is returned.
 */
int mtxgemvoverlap_parse(
    enum mtxgemvoverlap * overlap,
    int64_t * bytes_read,
    const char ** endptr,
    const char * s,
    const char * valid_delimiters);

#endif
