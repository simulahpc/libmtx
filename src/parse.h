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
 * Last modified: 2022-10-08
 *
 * String parsing.
 */

#ifndef PARSE_H
#define PARSE_H

#include <stdint.h>

enum mtxprecision;
enum mtxfileobject;
enum mtxfileformat;
enum mtxfilefield;
enum mtxfilesymmetry;
enum mtxfilesorting;
enum mtxfileordering;
enum mtxpartitioning;
enum mtxmatrixtype;
enum mtxvectortype;
enum mtxtransposition;
enum mtxgemvoverlap;
enum mtxmatrixparttype;

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
 * ‘parse_int32_hex()’ parses a hexadecimal number string to produce a
 * number that may be represented as a 32-bit integer.
 *
 * The number is parsed using ‘strtoll()’, following the conventions
 * documented in the man page for that function.  In addition, some
 * further error checking is performed to ensure that the number is
 * parsed correctly.  The parsed number is stored in ‘x’.
 *
 * If ‘endptr’ is not ‘NULL’, the address stored in ‘endptr’ points to
 * the first character beyond the characters that were consumed during
 * parsing.
 *
 * On success, ‘0’ is returned. Otherwise, if the input contained
 * invalid characters, ‘EINVAL’ is returned, or if the resulting
 * number cannot be represented as a signed, 32-bit integer, ‘ERANGE’
 * is returned.
 */
int parse_int32_hex(
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
 * ‘parse_mtxfileobject()’ parses a string to obtain a value of type
 * ‘enum mtxfileobject’.
 *
 * Valid strings are: ‘matrix’ and ‘vector’.
 *
 * If ‘endptr’ is not ‘NULL’, then the address stored in ‘endptr’
 * points to the first character beyond the characters that were
 * consumed during parsing. Also, if ‘bytes_read’ is not ‘NULL’, then
 * it is set to the number of bytes that were consumed by the parser.
 *
 * On success, ‘0’ is returned. Otherwise, if the input contained
 * invalid characters, ‘EINVAL’ is returned.
 */
int parse_mtxfileobject(
    enum mtxfileobject * fileobject,
    const char * s,
    char ** endptr,
    int64_t * bytes_read);

/**
 * ‘parse_mtxfileformat()’ parses a string to obtain a value of type
 * ‘enum mtxfileformat’.
 *
 * Valid strings are: ‘array’ and ‘coordinate’.
 *
 * If ‘endptr’ is not ‘NULL’, then the address stored in ‘endptr’
 * points to the first character beyond the characters that were
 * consumed during parsing. Also, if ‘bytes_read’ is not ‘NULL’, then
 * it is set to the number of bytes that were consumed by the parser.
 *
 * On success, ‘0’ is returned. Otherwise, if the input contained
 * invalid characters, ‘EINVAL’ is returned.
 */
int parse_mtxfileformat(
    enum mtxfileformat * fileformat,
    const char * s,
    char ** endptr,
    int64_t * bytes_read);

/**
 * ‘parse_mtxfilefield()’ parses a string to obtain a value of type
 * ‘enum mtxfilefield’.
 *
 * Valid strings are: ‘real’, ‘complex’, ‘integer’ and ‘pattern’.
 *
 * If ‘endptr’ is not ‘NULL’, then the address stored in ‘endptr’
 * points to the first character beyond the characters that were
 * consumed during parsing. Also, if ‘bytes_read’ is not ‘NULL’, then
 * it is set to the number of bytes that were consumed by the parser.
 *
 * On success, ‘0’ is returned. Otherwise, if the input contained
 * invalid characters, ‘EINVAL’ is returned.
 */
int parse_mtxfilefield(
    enum mtxfilefield * filefield,
    const char * s,
    char ** endptr,
    int64_t * bytes_read);

/**
 * ‘parse_mtxfilesymmetry()’ parses a string to obtain a value of type
 * ‘enum mtxfilesymmetry’.
 *
 * Valid strings are: ‘general’, ‘symmetric’, ‘skew-symmetric’ and
 * ‘hermitian’.
 *
 * If ‘endptr’ is not ‘NULL’, then the address stored in ‘endptr’
 * points to the first character beyond the characters that were
 * consumed during parsing. Also, if ‘bytes_read’ is not ‘NULL’, then
 * it is set to the number of bytes that were consumed by the parser.
 *
 * On success, ‘0’ is returned. Otherwise, if the input contained
 * invalid characters, ‘EINVAL’ is returned.
 */
int parse_mtxfilesymmetry(
    enum mtxfilesymmetry * filesymmetry,
    const char * s,
    char ** endptr,
    int64_t * bytes_read);

/**
 * ‘parse_mtxfilesorting()’ parses a string to obtain a value of type
 * ‘enum mtxfilesorting’.
 *
 * Valid strings are: ‘unsorted’, ‘permute’, ‘row-major’,
 * ‘column-major’ and ‘morton’.
 *
 * If ‘endptr’ is not ‘NULL’, then the address stored in ‘endptr’
 * points to the first character beyond the characters that were
 * consumed during parsing. Also, if ‘bytes_read’ is not ‘NULL’, then
 * it is set to the number of bytes that were consumed by the parser.
 *
 * On success, ‘0’ is returned. Otherwise, if the input contained
 * invalid characters, ‘EINVAL’ is returned.
 */
int parse_mtxfilesorting(
    enum mtxfilesorting * filesorting,
    const char * s,
    char ** endptr,
    int64_t * bytes_read);

/**
 * ‘parse_mtxfileordering()’ parses a string to obtain a value of type
 * ‘enum mtxfileordering’.
 *
 * Valid strings are: ‘default’, ‘custom’, ‘rcm’ and ‘nd’.
 *
 * If ‘endptr’ is not ‘NULL’, then the address stored in ‘endptr’
 * points to the first character beyond the characters that were
 * consumed during parsing. Also, if ‘bytes_read’ is not ‘NULL’, then
 * it is set to the number of bytes that were consumed by the parser.
 *
 * On success, ‘0’ is returned. Otherwise, if the input contained
 * invalid characters, ‘EINVAL’ is returned.
 */
int parse_mtxfileordering(
    enum mtxfileordering * fileordering,
    const char * s,
    char ** endptr,
    int64_t * bytes_read);

/**
 * ‘parse_mtxprecision()’ parses a string to obtain a value of type
 * ‘enum mtxprecision’.
 *
 * Valid strings are: ‘single’ and ‘double’.
 *
 * If ‘endptr’ is not ‘NULL’, then the address stored in ‘endptr’
 * points to the first character beyond the characters that were
 * consumed during parsing. Also, if ‘bytes_read’ is not ‘NULL’, then
 * it is set to the number of bytes that were consumed by the parser.
 *
 * On success, ‘0’ is returned. Otherwise, if the input contained
 * invalid characters, ‘EINVAL’ is returned.
 */
int parse_mtxprecision(
    enum mtxprecision * precision,
    const char * s,
    char ** endptr,
    int64_t * bytes_read);

/**
 * ‘parse_mtxpartitioning()’ parses a string to obtain a value of type
 * ‘enum mtxpartitioning’.
 *
 * Valid strings are: ‘block’, ‘block-cyclic’, ‘cyclic’ and ‘custom’.
 *
 * If ‘endptr’ is not ‘NULL’, then the address stored in ‘endptr’
 * points to the first character beyond the characters that were
 * consumed during parsing. Also, if ‘bytes_read’ is not ‘NULL’, then
 * it is set to the number of bytes that were consumed by the parser.
 *
 * On success, ‘0’ is returned. Otherwise, if the input contained
 * invalid characters, ‘EINVAL’ is returned.
 */
int parse_mtxpartitioning(
    enum mtxpartitioning * partitioning,
    const char * s,
    char ** endptr,
    int64_t * bytes_read);

/**
 * ‘parse_mtxtransposition()’ parses a string to obtain a value of type
 * ‘enum mtxtransposition’.
 *
 * Valid strings are: ‘notrans’, ‘trans’ and ‘conjtrans’.
 *
 * If ‘endptr’ is not ‘NULL’, then the address stored in ‘endptr’
 * points to the first character beyond the characters that were
 * consumed during parsing. Also, if ‘bytes_read’ is not ‘NULL’, then
 * it is set to the number of bytes that were consumed by the parser.
 *
 * On success, ‘0’ is returned. Otherwise, if the input contained
 * invalid characters, ‘EINVAL’ is returned.
 */
int parse_mtxtransposition(
    enum mtxtransposition * transposition,
    const char * s,
    char ** endptr,
    int64_t * bytes_read);

/**
 * ‘parse_mtxvectortype()’ parses a string to obtain a value of type
 * ‘enum mtxvectortype’.
 *
 * Valid strings are: ‘base’, ‘blas’, ‘null’ and ‘omp’.
 *
 * If ‘endptr’ is not ‘NULL’, then the address stored in ‘endptr’
 * points to the first character beyond the characters that were
 * consumed during parsing. Also, if ‘bytes_read’ is not ‘NULL’, then
 * it is set to the number of bytes that were consumed by the parser.
 *
 * On success, ‘0’ is returned. Otherwise, if the input contained
 * invalid characters, ‘EINVAL’ is returned.
 */
int parse_mtxvectortype(
    enum mtxvectortype * vectortype,
    const char * s,
    char ** endptr,
    int64_t * bytes_read);

/**
 * ‘parse_mtxmatrixtype()’ parses a string to obtain a value of type
 * ‘enum mtxmatrixtype’.
 *
 * Valid strings are: ‘blas’, ‘coo’, ‘csr’, ‘dense’, ‘nullcoo’ and
 * ‘ompcsr’.
 *
 * If ‘endptr’ is not ‘NULL’, then the address stored in ‘endptr’
 * points to the first character beyond the characters that were
 * consumed during parsing. Also, if ‘bytes_read’ is not ‘NULL’, then
 * it is set to the number of bytes that were consumed by the parser.
 *
 * On success, ‘0’ is returned. Otherwise, if the input contained
 * invalid characters, ‘EINVAL’ is returned.
 */
int parse_mtxmatrixtype(
    enum mtxmatrixtype * matrixtype,
    const char * s,
    char ** endptr,
    int64_t * bytes_read);

/**
 * ‘parse_mtxgemvoverlap()’ parses a string to obtain a value of type
 * ‘enum mtxgemvoverlap’.
 *
 * Valid strings are: ‘none’ and ‘irecv’.
 *
 * If ‘endptr’ is not ‘NULL’, then the address stored in ‘endptr’
 * points to the first character beyond the characters that were
 * consumed during parsing. Also, if ‘bytes_read’ is not ‘NULL’, then
 * it is set to the number of bytes that were consumed by the parser.
 *
 * On success, ‘0’ is returned. Otherwise, if the input contained
 * invalid characters, ‘EINVAL’ is returned.
 */
int parse_mtxgemvoverlap(
    enum mtxgemvoverlap * gemvoverlap,
    const char * s,
    char ** endptr,
    int64_t * bytes_read);

/**
 * ‘parse_mtxmatrixparttype()’ parses a string to obtain a value of type
 * ‘enum mtxmatrixparttype’.
 *
 * Valid strings are: ‘nonzeros’, ‘rows’, ‘columns’, ‘2d’ and ‘metis’.
 *
 * If ‘endptr’ is not ‘NULL’, then the address stored in ‘endptr’
 * points to the first character beyond the characters that were
 * consumed during parsing. Also, if ‘bytes_read’ is not ‘NULL’, then
 * it is set to the number of bytes that were consumed by the parser.
 *
 * On success, ‘0’ is returned. Otherwise, if the input contained
 * invalid characters, ‘EINVAL’ is returned.
 */
int parse_mtxmatrixparttype(
    enum mtxmatrixparttype * matrixparttype,
    const char * s,
    char ** endptr,
    int64_t * bytes_read);

#endif
