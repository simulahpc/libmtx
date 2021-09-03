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
 * Last modified: 2021-09-01
 *
 * Matrix Market file headers.
 */

#ifndef LIBMTX_MTXFILE_HEADER_H
#define LIBMTX_MTXFILE_HEADER_H

#include <libmtx/libmtx-config.h>

#ifdef LIBMTX_HAVE_LIBZ
#include <zlib.h>
#endif

#include <stddef.h>
#include <stdio.h>

/*
 * Matrix Market header data types.
 */

/**
 * `mtxfile_object' is used to enumerate different kinds of Matrix
 * Market objects.
 */
enum mtxfile_object
{
    mtxfile_matrix,
    mtxfile_vector
};

/**
 * `mtxfile_format' is used to enumerate different kinds of Matrix
 * Market formats.
 */
enum mtxfile_format
{
    mtxfile_array,     /* array format for dense matrices and vectors */
    mtxfile_coordinate /* coordinate format for sparse matrices and vectors */
};

/**
 * `mtxfile_field' is used to enumerate different kinds of fields for
 * matrix values in Matrix Market files.
 */
enum mtxfile_field
{
    mtxfile_real,    /* real, floating-point coefficients */
    mtxfile_complex, /* complex, floating point coefficients */
    mtxfile_integer, /* integer coefficients */
    mtxfile_pattern  /* boolean coefficients (sparsity pattern) */
};

/**
 * `mtxfile_symmetry' is used to enumerate different kinds of symmetry
 * for matrices in Matrix Market format.
 */
enum mtxfile_symmetry
{
    mtxfile_general,        /* general, non-symmetric matrix */
    mtxfile_symmetric,      /* symmetric matrix */
    mtxfile_skew_symmetric, /* skew-symmetric matrix */
    mtxfile_hermitian       /* Hermitian matrix */
};

/*
 * Formatting and parsing of Matrix Market header data types.
 */

/**
 * `mtxfile_object_str()' is a string representing the Matrix Market
 * object type.
 */
const char * mtxfile_object_str(
    enum mtxfile_object object);

/**
 * `mtxfile_parse_object()' parses a string containing the `object' of
 * a Matrix Market file format header.
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
 * On success, `mtxfile_parse_object()' returns `MTX_SUCCESS' and
 * `object' is set according to the parsed string and `bytes_read' is
 * set to the number of bytes that were consumed by the parser.
 * Otherwise, an error code is returned.
 */
int mtxfile_parse_object(
    enum mtxfile_object * object,
    int * bytes_read,
    const char ** endptr,
    const char * s,
    const char * valid_delimiters);

/**
 * `mtxfile_format_str()' is a string representing the Matrix Market
 * format type.
 */
const char * mtxfile_format_str(
    enum mtxfile_format format);

/**
 * `mtxfile_parse_format()' parses a string containing the `format' of
 * a Matrix Market file format header.
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
 * On success, `mtxfile_parse_format()' returns `MTX_SUCCESS' and
 * `format' is set according to the parsed string and `bytes_read' is
 * set to the number of bytes that were consumed by the parser.
 * Otherwise, an error code is returned.
 */
int mtxfile_parse_format(
    enum mtxfile_format * format,
    int * bytes_read,
    const char ** endptr,
    const char * s,
    const char * valid_delimiters);

/**
 * `mtxfile_field_str()' is a string representing the Matrix Market
 * field type.
 */
const char * mtxfile_field_str(
    enum mtxfile_field field);

/**
 * `mtxfile_parse_field()' parses a string containing the `field' of a
 * Matrix Market file format header.
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
 * On success, `mtxfile_parse_field()' returns `MTX_SUCCESS' and
 * `field' is set according to the parsed string and `bytes_read' is
 * set to the number of bytes that were consumed by the parser.
 * Otherwise, an error code is returned.
 */
int mtxfile_parse_field(
    enum mtxfile_field * field,
    int * bytes_read,
    const char ** endptr,
    const char * s,
    const char * valid_delimiters);

/**
 * `mtxfile_symmetry_str()' is a string representing the Matrix Market
 * symmetry type.
 */
const char * mtxfile_symmetry_str(
    enum mtxfile_symmetry symmetry);

/**
 * `mtxfile_parse_symmetry()' parses a string containing the
 * `symmetry' of a Matrix Market file format header.
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
 * On success, `mtxfile_parse_symmetry()' returns `MTX_SUCCESS' and
 * `symmetry' is set according to the parsed string and `bytes_read'
 * is set to the number of bytes that were consumed by the parser.
 * Otherwise, an error code is returned.
 */
int mtxfile_parse_symmetry(
    enum mtxfile_symmetry * symmetry,
    int * bytes_read,
    const char ** endptr,
    const char * s,
    const char * valid_delimiters);

/*
 * Matrix Market header.
 */

/**
 * `mtxfile_header' represents the header line of a Matrix Market
 * file.
 */
struct mtxfile_header
{
    /**
     * `object' is the type of Matrix Market object: `matrix' or
     * `vector'.
     */
    enum mtxfile_object object;

    /**
     * `format' is the matrix format: `coordinate' or `array'.
     */
    enum mtxfile_format format;

    /**
     * `field' is the matrix field: `real', `complex', `integer' or
     * `pattern'.
     */
    enum mtxfile_field field;

    /**
     * `symmetry' is the matrix symmetry: `general', `symmetric',
     * `skew-symmetric', or `hermitian'.
     *
     * Note that if `symmetry' is `symmetric', `skew-symmetric' or
     * `hermitian', then the matrix must be square, so that `num_rows'
     * is equal to `num_columns'.
     */
    enum mtxfile_symmetry symmetry;
};

/**
 * `mtxfile_parse_header()' parses a string containing the header line
 * for a file in Matrix Market format.
 *
 * If `endptr' is not `NULL', then the address stored in `endptr'
 * points to the first character beyond the characters that were
 * consumed during parsing.
 *
 * On success, `mtxfile_parse_header()' returns `MTX_SUCCESS' and the
 * `object', `format', `field' and `symmetry' fields of the header
 * will be set according to the contents of the parsed Matrix Market
 * header.  Otherwise, an appropriate error code is returned if the
 * input is not a valid Matrix Market header.
 */
int mtxfile_parse_header(
    struct mtxfile_header * header,
    int * bytes_read,
    const char ** endptr,
    const char * s);

/**
 * `mtxfile_header_copy()' copies a Matrix Market header.
 */
int mtxfile_header_copy(
    struct mtxfile_header * dst,
    const struct mtxfile_header * src);

/**
 * `mtxfile_fread_header()` reads a Matrix Market header from a
 * stream.
 *
 * If an error code is returned, then `lines_read' and `bytes_read'
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the Matrix Market file.
 */
int mtxfile_fread_header(
    struct mtxfile_header * header,
    FILE * f,
    int * lines_read,
    int * bytes_read,
    size_t line_max,
    char * linebuf);

#ifdef LIBMTX_HAVE_LIBZ
/**
 * `mtxfile_gzread_header()` reads a Matrix Market header from a
 * gzip-compressed stream.
 *
 * If an error code is returned, then `lines_read' and `bytes_read'
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the Matrix Market file.
 */
int mtxfile_gzread_header(
    struct mtxfile_header * header,
    gzFile f,
    int * lines_read,
    int * bytes_read,
    size_t line_max,
    char * linebuf);
#endif

#endif
