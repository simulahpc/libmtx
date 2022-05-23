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
 * Last modified: 2022-04-14
 *
 * String parsing.
 */

#ifndef LIBMTX_UTIL_PARSE_H
#define LIBMTX_UTIL_PARSE_H

#include <stdint.h>

/**
 * ‘parse_int()’ parses a string to produce a number that may be
 * represented as an integer.
 *
 * The number is parsed using ‘strtoll()’, following the conventions
 * documented in the man page for that function. In addition, some
 * further error checking is performed to ensure that the number is
 * parsed correctly. The parsed number is stored in ‘x’.
 *
 * If ‘endptr’ is not ‘NULL’, the address stored in ‘endptr’ points to
 * the first character beyond the characters that were consumed during
 * parsing.
 *
 * On success, ‘MTX_SUCCESS’ is returned. Otherwise, if the input
 * contained invalid characters, errno is set to ‘EINVAL’ and
 * ‘MTX_ERR_ERRNO’ is returned. If the resulting number cannot be
 * represented as a signed integer, errno is set to ‘ERANGE’ and
 * ‘MTX_ERR_ERRNO’ is returned.
 */
int parse_int(
    int * x,
    const char * s,
    char ** endptr,
    int64_t * bytes_read);

/**
 * ‘parse_int32()’ parses a string to produce a number that may be
 * represented as a 32-bit integer.
 *
 * The number is parsed using ‘strtoll()’, following the conventions
 * documented in the man page for that function. In addition, some
 * further error checking is performed to ensure that the number is
 * parsed correctly. The parsed number is stored in ‘x’.
 *
 * If ‘endptr’ is not ‘NULL’, the address stored in ‘endptr’ points to
 * the first character beyond the characters that were consumed during
 * parsing.
 *
 * On success, ‘MTX_SUCCESS’ is returned. Otherwise, if the input
 * contained invalid characters, errno is set to ‘EINVAL’ and
 * ‘MTX_ERR_ERRNO’ is returned. If the resulting number cannot be
 * represented as a signed 32-bit integer, errno is set to ‘ERANGE’
 * and ‘MTX_ERR_ERRNO’ is returned.
 */
int parse_int32(
    int32_t * x,
    const char * s,
    char ** endptr,
    int64_t * bytes_read);

/**
 * ‘parse_int64()’ parses a string to produce a number that may be
 * represented as a 64-bit integer.
 *
 * The number is parsed using ‘strtoll()’, following the conventions
 * documented in the man page for that function. In addition, some
 * further error checking is performed to ensure that the number is
 * parsed correctly. The parsed number is stored in ‘x’.
 *
 * If ‘endptr’ is not ‘NULL’, the address stored in ‘endptr’ points to
 * the first character beyond the characters that were consumed during
 * parsing.
 *
 * On success, ‘MTX_SUCCESS’ is returned. Otherwise, if the input
 * contained invalid characters, errno is set to ‘EINVAL’ and
 * ‘MTX_ERR_ERRNO’ is returned. If the resulting number cannot be
 * represented as a signed 64-bit integer, errno is set to ‘ERANGE’
 * and ‘MTX_ERR_ERRNO’ is returned.
 */
int parse_int64(
    int64_t * x,
    const char * s,
    char ** endptr,
    int64_t * bytes_read);

/**
 * ‘parse_float()’ parses a string to produce a number that may be
 * represented as ‘float’.
 *
 * The number is parsed using ‘strtof()’, following the conventions
 * documented in the man page for that function.  In addition, some
 * further error checking is performed to ensure that the number is
 * parsed correctly.  The parsed number is stored in ‘number’.
 *
 * If ‘endptr’ is not ‘NULL’, the address stored in ‘endptr’ points to
 * the first character beyond the characters that were consumed during
 * parsing.
 *
 * On success, ‘MTX_SUCCESS’ is returned. Otherwise, if the input
 * contained invalid characters, errno is set to ‘EINVAL’ and
 * ‘MTX_ERR_ERRNO’ is returned. If the resulting number cannot be
 * represented as a float, errno is set to ‘ERANGE’ and
 * ‘MTX_ERR_ERRNO’ is returned.
 */
int parse_float(
    float * x,
    const char * s,
    char ** endptr,
    int64_t * bytes_read);

/**
 * ‘parse_double()’ parses a string to produce a number that may be
 * represented as ‘double’.
 *
 * The number is parsed using ‘strtod()’, following the conventions
 * documented in the man page for that function.  In addition, some
 * further error checking is performed to ensure that the number is
 * parsed correctly.  The parsed number is stored in ‘number’.
 *
 * If ‘endptr’ is not ‘NULL’, the address stored in ‘endptr’ points to
 * the first character beyond the characters that were consumed during
 * parsing.
 *
 * On success, ‘MTX_SUCCESS’ is returned. Otherwise, if the input
 * contained invalid characters, errno is set to ‘EINVAL’ and
 * ‘MTX_ERR_ERRNO’ is returned. If the resulting number cannot be
 * represented as a double, errno is set to ‘ERANGE’ and
 * ‘MTX_ERR_ERRNO’ is returned.
 */
int parse_double(
    double * x,
    const char * s,
    char ** endptr,
    int64_t * bytes_read);

/**
 * ‘parse_int32_ex()’ parses a string to produce a number that may be
 * represented as a 32-bit integer.
 *
 * The number is parsed using ‘strtoll()’, following the conventions
 * documented in the man page for that function.  In addition, some
 * further error checking is performed to ensure that the number is
 * parsed correctly.  The parsed number is stored in ‘number’.
 *
 * ‘valid_delimiters’ is either ‘NULL’, in which case it is ignored,
 * or, it may contain a string of characters that constitute valid
 * delimiters for the parsed string.  That is, after parsing a number,
 * if there are any remaining, unconsumed characters in the string,
 * ‘parse_int32_ex()’ checks if the next character is found in the
 * string ‘valid_delimiters’.  If the character is not found, then the
 * string is judged to be invalid, and ‘EINVAL’ is returned.
 * Otherwise, the final, delimiter character is consumed by the
 * parser.
 *
 * If ‘endptr’ is not ‘NULL’, the address stored in ‘endptr’ points to
 * the first character beyond the characters that were consumed during
 * parsing.
 *
 * On success, ‘parse_int32_ex()’ returns ‘0’.  Otherwise, if the input
 * contained invalid characters, ‘parse_int32_ex()’ returns ‘EINVAL’.  If
 * the resulting number cannot be represented as a signed 32-bit
 * integer, ‘parse_int32_ex()’ returns ‘ERANGE’.
 */
int parse_int32_ex(
    const char * s,
    const char * valid_delimiters,
    int32_t * out_number,
    const char ** endptr);

/**
 * ‘parse_int32_hex()’ parses a hexadecimal number string that may be
 * represented as a 32-bit integer.
 *
 * The number is parsed using ‘strtoll()’, following the conventions
 * documented in the man page for that function.  In addition, some
 * further error checking is performed to ensure that the number is
 * parsed correctly.  The parsed number is stored in ‘number’.
 *
 * ‘valid_delimiters’ is either ‘NULL’, in which case it is ignored,
 * or, it may contain a string of characters that constitute valid
 * delimiters for the parsed string.  That is, after parsing a number,
 * if there are any remaining, unconsumed characters in the string,
 * ‘parse_int32_hex()’ checks if the next character is found in the
 * string ‘valid_delimiters’.  If the character is not found, then the
 * string is judged to be invalid, and ‘EINVAL’ is returned.
 * Otherwise, the final, delimiter character is consumed by the
 * parser.
 *
 * If ‘endptr’ is not ‘NULL’, the address stored in ‘endptr’ points to
 * the first character beyond the characters that were consumed during
 * parsing.
 *
 * On success, ‘parse_int32_hex()’ returns ‘0’.  Otherwise, if the
 * input contained invalid characters, ‘parse_int32_hex()’ returns
 * ‘EINVAL’.  If the resulting number cannot be represented as a
 * signed 32-bit integer, ‘parse_int32_hex()’ returns ‘ERANGE’.
 */
int parse_int32_hex(
    const char * s,
    const char * valid_delimiters,
    int32_t * out_number,
    const char ** endptr);

/**
 * ‘parse_int64_ex()’ parses a string to produce a number that may be
 * represented as a 64-bit integer.
 *
 * The number is parsed using ‘strtoll()’, following the conventions
 * documented in the man page for that function.  In addition, some
 * further error checking is performed to ensure that the number is
 * parsed correctly.  The parsed number is stored in ‘number’.
 *
 * ‘valid_delimiters’ is either ‘NULL’, in which case it is ignored,
 * or, it may contain a string of characters that constitute valid
 * delimiters for the parsed string.  That is, after parsing a number,
 * if there are any remaining, unconsumed characters in the string,
 * ‘parse_int64_ex()’ checks if the next character is found in the string
 * ‘valid_delimiters’.  If the character is not found, then the string
 * is judged to be invalid, and ‘EINVAL’ is returned.  Otherwise, the
 * final, delimiter character is consumed by the parser.
 *
 * If ‘endptr’ is not ‘NULL’, the address stored in ‘endptr’ points to
 * the first character beyond the characters that were consumed during
 * parsing.
 *
 * On success, ‘parse_int64_ex()’ returns ‘0’.  Otherwise, if the input
 * contained invalid characters, ‘parse_int64_ex()’ returns ‘EINVAL’.  If
 * the resulting number cannot be represented as a signed 64-bit
 * integer, ‘parse_int64_ex()’ returns ‘ERANGE’.
 */
int parse_int64_ex(
    const char * s,
    const char * valid_delimiters,
    int64_t * number,
    const char ** endptr);

/**
 * ‘parse_float_ex()’ parses a string to produce a number that may be
 * represented as ‘float’.
 *
 * The number is parsed using ‘strtof()’, following the conventions
 * documented in the man page for that function.  In addition, some
 * further error checking is performed to ensure that the number is
 * parsed correctly.  The parsed number is stored in ‘number’.
 *
 * ‘valid_delimiters’ is either ‘NULL’, in which case it is ignored,
 * or, it may contain a string of characters that constitute valid
 * delimiters for the parsed string.  That is, after parsing a number,
 * if there are any remaining, unconsumed characters in the string,
 * ‘parse_float_ex()’ checks if the next character is found in the string
 * ‘valid_delimiters’.  If the character is not found, then the string
 * is judged to be invalid, and ‘EINVAL’ is returned.  Otherwise, the
 * final, delimiter character is consumed by the parser.
 *
 * If ‘endptr’ is not ‘NULL’, the address stored in ‘endptr’ points to
 * the first character beyond the characters that were consumed during
 * parsing.
 *
 * On success, ‘parse_float_ex()’ returns ‘0’.  Otherwise, if the input
 * contained invalid characters, ‘parse_float_ex()’ returns ‘EINVAL’.
 */
int parse_float_ex(
    const char * s,
    const char * valid_delimiters,
    float * number,
    const char ** endptr);

/**
 * ‘parse_double_ex()’ parses a string to produce a number that may be
 * represented as ‘double’.
 *
 * The number is parsed using ‘strtod()’, following the conventions
 * documented in the man page for that function.  In addition, some
 * further error checking is performed to ensure that the number is
 * parsed correctly.  The parsed number is stored in ‘number’.
 *
 * ‘valid_delimiters’ is either ‘NULL’, in which case it is ignored,
 * or, it may contain a string of characters that constitute valid
 * delimiters for the parsed string.  That is, after parsing a number,
 * if there are any remaining, unconsumed characters in the string,
 * ‘parse_double_ex()’ checks if the next character is found in the string
 * ‘valid_delimiters’.  If the character is not found, then the string
 * is judged to be invalid, and ‘EINVAL’ is returned.  Otherwise, the
 * final, delimiter character is consumed by the parser.
 *
 * If ‘endptr’ is not ‘NULL’, the address stored in ‘endptr’ points to
 * the first character beyond the characters that were consumed during
 * parsing.
 *
 * On success, ‘parse_double_ex()’ returns ‘0’.  Otherwise, if the input
 * contained invalid characters, ‘parse_double_ex()’ returns ‘EINVAL’.
 */
int parse_double_ex(
    const char * s,
    const char * valid_delimiters,
    double * number,
    const char ** endptr);

#endif
