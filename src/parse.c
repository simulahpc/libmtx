/* This file is part of Libmtx.
 *
 * Copyright (C) 2023 James D. Trotter
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
 * Last modified: 2023-03-24
 *
 * String parsing.
 */

#include "parse.h"

#include <libmtx/mtxfile/header.h>
#include <libmtx/mtxfile/mtxfile.h>
#include <libmtx/linalg/precision.h>
#include <libmtx/linalg/local/matrix.h>
#include <libmtx/linalg/local/vector.h>
#include <libmtx/linalg/gemvoverlap.h>
#include <libmtx/linalg/partition.h>
#include <libmtx/util/partition.h>

#include <errno.h>

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/**
 * ‘parse_long_long_int()’ parses a string to produce a number that
 * may be represented with the type ‘long long int’.
 */
static int parse_long_long_int(
    long long int * outnumber,
    int base,
    const char * s,
    char ** outendptr,
    int64_t * bytes_read)
{
    errno = 0;
    char * endptr;
    long long int number = strtoll(s, &endptr, base);
    if ((errno == ERANGE && (number == LLONG_MAX || number == LLONG_MIN)) ||
        (errno != 0 && number == 0))
        return errno;
    if (s == endptr) return EINVAL;
    if (outendptr) *outendptr = endptr;
    if (bytes_read) *bytes_read += endptr - s;
    *outnumber = number;
    return 0;
}

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
 * On success, ‘0’ is returned. Otherwise, if the input contained
 * invalid characters, ‘EINVAL’ is returned, or if the resulting
 * number cannot be represented as a signed integer, ‘ERANGE’ is
 * returned.
 */
int parse_int(
    int * x,
    const char * s,
    char ** endptr,
    int64_t * bytes_read)
{
    long long int y;
    int err = parse_long_long_int(&y, 10, s, endptr, bytes_read);
    if (err) return err;
    if (y < INT_MIN || y > INT_MAX) return ERANGE;
    *x = y;
    return 0;
}

/**
 * ‘parse_int32()’ parses a string to produce a number that may be
 * represented as a 32-bit integer.
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
int parse_int32(
    int32_t * x,
    const char * s,
    char ** endptr,
    int64_t * bytes_read)
{
    long long int y;
    int err = parse_long_long_int(&y, 10, s, endptr, bytes_read);
    if (err) return err;
    if (y < INT32_MIN || y > INT32_MAX) return ERANGE;
    *x = y;
    return 0;
}

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
    int64_t * bytes_read)
{
    long long int y;
    int err = parse_long_long_int(&y, 16, s, endptr, bytes_read);
    if (err) return err;
    if (y < INT32_MIN || y > INT32_MAX) return ERANGE;
    *x = y;
    return 0;
}

/**
 * ‘parse_int64()’ parses a string to produce a number that may be
 * represented as a 64-bit integer.
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
 * number cannot be represented as a signed, 64-bit integer, ‘ERANGE’
 * is returned.
 */
int parse_int64(
    int64_t * x,
    const char * s,
    char ** endptr,
    int64_t * bytes_read)
{
    long long int y;
    int err = parse_long_long_int(&y, 10, s, endptr, bytes_read);
    if (err) return err;
    if (y < INT64_MIN || y > INT64_MAX) return ERANGE;
    *x = y;
    return 0;
}

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
 * On success, ‘0’ is returned. Otherwise, if the input contained
 * invalid characters, ‘EINVAL’ is returned, or if the resulting
 * number cannot be represented as a single precision float, ‘ERANGE’
 * is returned.
 */
int parse_float(
    float * x,
    const char * s,
    char ** outendptr,
    int64_t * bytes_read)
{
    errno = 0;
    char * endptr;
    *x = strtof(s, &endptr);
    if ((errno == ERANGE && (*x == HUGE_VALF || *x == -HUGE_VALF)) ||
        (errno != 0 && x == 0)) {
        return errno;
    }
    if (outendptr) *outendptr = endptr;
    if (bytes_read) *bytes_read += endptr - s;
    return 0;
}

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
 * On success, ‘0’ is returned. Otherwise, if the input contained
 * invalid characters, ‘EINVAL’ is returned, or if the resulting
 * number cannot be represented as a double precision float, ‘ERANGE’
 * is returned.
 */
int parse_double(
    double * x,
    const char * s,
    char ** outendptr,
    int64_t * bytes_read)
{
    errno = 0;
    char * endptr;
    *x = strtod(s, &endptr);
    if ((errno == ERANGE && (*x == HUGE_VAL || *x == -HUGE_VAL)) ||
        (errno != 0 && x == 0)) {
        return errno;
    }
    if (outendptr) *outendptr = endptr;
    if (bytes_read) *bytes_read += endptr - s;
    return 0;
}

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
    int64_t * bytes_read)
{
    const char * t = s;
    if (strncmp("matrix", t, strlen("matrix")) == 0) {
        t += strlen("matrix"); *fileobject = mtxfile_matrix;
    } else if (strncmp("vector", t, strlen("vector")) == 0) {
        t += strlen("vector"); *fileobject = mtxfile_vector;
    } else { return EINVAL; }
    if (bytes_read) *bytes_read += t-s;
    if (endptr) *endptr = (char *) t;
    return 0;
}

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
    int64_t * bytes_read)
{
    const char * t = s;
    if (strncmp("array", t, strlen("array")) == 0) {
        t += strlen("array"); *fileformat = mtxfile_array;
    } else if (strncmp("coordinate", t, strlen("coordinate")) == 0) {
        t += strlen("coordinate"); *fileformat = mtxfile_coordinate;
    } else { return EINVAL; }
    if (bytes_read) *bytes_read += t-s;
    if (endptr) *endptr = (char *) t;
    return 0;
}

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
    int64_t * bytes_read)
{
    const char * t = s;
    if (strncmp("real", t, strlen("real")) == 0) {
        t += strlen("real"); *filefield = mtxfile_real;
    } else if (strncmp("complex", t, strlen("complex")) == 0) {
        t += strlen("complex"); *filefield = mtxfile_complex;
    } else if (strncmp("integer", t, strlen("integer")) == 0) {
        t += strlen("integer"); *filefield = mtxfile_integer;
    } else if (strncmp("pattern", t, strlen("pattern")) == 0) {
        t += strlen("pattern"); *filefield = mtxfile_pattern;
    } else { return EINVAL; }
    if (bytes_read) *bytes_read += t-s;
    if (endptr) *endptr = (char *) t;
    return 0;
}

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
    int64_t * bytes_read)
{
    const char * t = s;
    if (strncmp("general", t, strlen("general")) == 0) {
        t += strlen("general"); *filesymmetry = mtxfile_general;
    } else if (strncmp("symmetric", t, strlen("symmetric")) == 0) {
        t += strlen("symmetric"); *filesymmetry = mtxfile_symmetric;
    } else if (strncmp("skew-symmetric", t, strlen("skew-symmetric")) == 0) {
        t += strlen("skew-symmetric"); *filesymmetry = mtxfile_skew_symmetric;
    } else if (strncmp("hermitian", t, strlen("hermitian")) == 0) {
        t += strlen("hermitian"); *filesymmetry = mtxfile_hermitian;
    } else { return EINVAL; }
    if (bytes_read) *bytes_read += t-s;
    if (endptr) *endptr = (char *) t;
    return 0;
}

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
    int64_t * bytes_read)
{
    const char * t = s;
    if (strncmp("unsorted", t, strlen("unsorted")) == 0) {
        t += strlen("unsorted"); *filesorting = mtxfile_unsorted;
    } else if (strncmp("permutation", t, strlen("permutation")) == 0) {
        t += strlen("permutation"); *filesorting = mtxfile_permutation;
    } else if (strncmp("row-major", t, strlen("row-major")) == 0) {
        t += strlen("row-major"); *filesorting = mtxfile_row_major;
    } else if (strncmp("column-major", t, strlen("column-major")) == 0) {
        t += strlen("column-major"); *filesorting = mtxfile_column_major;
    } else if (strncmp("morton", t, strlen("morton")) == 0) {
        t += strlen("morton"); *filesorting = mtxfile_morton;
    } else { return EINVAL; }
    if (bytes_read) *bytes_read += t-s;
    if (endptr) *endptr = (char *) t;
    return 0;
}

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
    int64_t * bytes_read)
{
    const char * t = s;
    if (strncmp("default", t, strlen("default")) == 0) {
        t += strlen("default"); *fileordering = mtxfile_default_order;
    } else if (strncmp("custom", t, strlen("custom")) == 0) {
        t += strlen("custom"); *fileordering = mtxfile_custom_order;
    } else if (strncmp("rcm", t, strlen("rcm")) == 0) {
        t += strlen("rcm"); *fileordering = mtxfile_rcm;
    } else if (strncmp("nd", t, strlen("nd")) == 0) {
        t += strlen("nd"); *fileordering = mtxfile_nd;
    } else if (strncmp("metis", t, strlen("metis")) == 0) {
        t += strlen("metis"); *fileordering = mtxfile_metis;
    } else { return EINVAL; }
    if (bytes_read) *bytes_read += t-s;
    if (endptr) *endptr = (char *) t;
    return 0;
}

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
    int64_t * bytes_read)
{
    const char * t = s;
    if (strncmp("single", t, strlen("single")) == 0) {
        t += strlen("single"); *precision = mtx_single;
    } else if (strncmp("double", t, strlen("double")) == 0) {
        t += strlen("double"); *precision = mtx_double;
    } else { return EINVAL; }
    if (bytes_read) *bytes_read += t-s;
    if (endptr) *endptr = (char *) t;
    return 0;
}

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
    int64_t * bytes_read)
{
    const char * t = s;
    if (strncmp("block-cyclic", t, strlen("block-cyclic")) == 0) {
        t += strlen("block-cyclic"); *partitioning = mtx_block_cyclic;
    } else if (strncmp("block", t, strlen("block")) == 0) {
        t += strlen("block"); *partitioning = mtx_block;
    } else if (strncmp("cyclic", t, strlen("cyclic")) == 0) {
        t += strlen("cyclic"); *partitioning = mtx_cyclic;
    } else if (strncmp("custom", t, strlen("custom")) == 0) {
        t += strlen("custom"); *partitioning = mtx_custom_partition;
    } else { return EINVAL; }
    if (bytes_read) *bytes_read += t-s;
    if (endptr) *endptr = (char *) t;
    return 0;
}

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
    int64_t * bytes_read)
{
    const char * t = s;
    if (strncmp("notrans", t, strlen("notrans")) == 0) {
        t += strlen("notrans"); *transposition = mtx_notrans;
    } else if (strncmp("trans", t, strlen("trans")) == 0) {
        t += strlen("trans"); *transposition = mtx_trans;
    } else if (strncmp("conjtrans", t, strlen("conjtrans")) == 0) {
        t += strlen("conjtrans"); *transposition = mtx_conjtrans;
    } else { return EINVAL; }
    if (bytes_read) *bytes_read += t-s;
    if (endptr) *endptr = (char *) t;
    return 0;
}

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
    int64_t * bytes_read)
{
    const char * t = s;
    if (strncmp("base", t, strlen("base")) == 0) {
        t += strlen("base"); *vectortype = mtxbasevector;
    } else if (strncmp("blas", t, strlen("blas")) == 0) {
        t += strlen("blas"); *vectortype = mtxblasvector;
    } else if (strncmp("null", t, strlen("null")) == 0) {
        t += strlen("null"); *vectortype = mtxnullvector;
    } else if (strncmp("omp", t, strlen("omp")) == 0) {
        t += strlen("omp"); *vectortype = mtxompvector;
    } else { return EINVAL; }
    if (bytes_read) *bytes_read += t-s;
    if (endptr) *endptr = (char *) t;
    return 0;
}

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
    int64_t * bytes_read)
{
    const char * t = s;
    if (strncmp("blas", t, strlen("blas")) == 0) {
        t += strlen("blas"); *matrixtype = mtxblasdense;
    } else if (strncmp("coo", t, strlen("coo")) == 0) {
        t += strlen("coo"); *matrixtype = mtxbasecoo;
    } else if (strncmp("csr", t, strlen("csr")) == 0) {
        t += strlen("csr"); *matrixtype = mtxbasecsr;
    } else if (strncmp("dense", t, strlen("dense")) == 0) {
        t += strlen("dense"); *matrixtype = mtxbasedense;
    } else if (strncmp("nullcoo", t, strlen("nullcoo")) == 0) {
        t += strlen("nullcoo"); *matrixtype = mtxnullcoo;
    } else if (strncmp("ompcsr", t, strlen("ompcsr")) == 0) {
        t += strlen("ompcsr"); *matrixtype = mtxompcsr;
    } else { return EINVAL; }
    if (bytes_read) *bytes_read += t-s;
    if (endptr) *endptr = (char *) t;
    return 0;
}

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
    int64_t * bytes_read)
{
    const char * t = s;
    if (strncmp("none", t, strlen("none")) == 0) {
        t += strlen("none"); *gemvoverlap = mtxgemvoverlap_none;
    } else if (strncmp("irecv", t, strlen("irecv")) == 0) {
        t += strlen("irecv"); *gemvoverlap = mtxgemvoverlap_irecv;
    } else { return EINVAL; }
    if (bytes_read) *bytes_read += t-s;
    if (endptr) *endptr = (char *) t;
    return 0;
}

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
    int64_t * bytes_read)
{
    const char * t = s;
    if (strncmp("nonzeros", t, strlen("nonzeros")) == 0) {
        t += strlen("nonzeros"); *matrixparttype = mtx_matrixparttype_nonzeros;
    } else if (strncmp("rows", t, strlen("rows")) == 0) {
        t += strlen("rows"); *matrixparttype = mtx_matrixparttype_rows;
    } else if (strncmp("columns", t, strlen("columns")) == 0) {
        t += strlen("columns"); *matrixparttype = mtx_matrixparttype_columns;
    } else if (strncmp("2d", t, strlen("2d")) == 0) {
        t += strlen("2d"); *matrixparttype = mtx_matrixparttype_2d;
    } else if (strncmp("metis", t, strlen("metis")) == 0) {
        t += strlen("metis"); *matrixparttype = mtx_matrixparttype_metis;
    } else { return EINVAL; }
    if (bytes_read) *bytes_read += t-s;
    if (endptr) *endptr = (char *) t;
    return 0;
}
