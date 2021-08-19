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
 * Last modified: 2021-08-19
 *
 * Data types for the Matrix Market header.
 */

#ifndef LIBMTX_MTX_HEADER_H
#define LIBMTX_MTX_HEADER_H

#include <stdint.h>

/*
 * Data types for Matrix Market headers.
 */

/**
 * `mtx_object' is used to enumerate different kinds of Matrix Market
 * objects.
 */
enum mtx_object
{
    mtx_matrix,
    mtx_vector
};

/**
 * `mtx_object_str()' is a string representing the Matrix Market
 * object type.
 */
const char * mtx_object_str(
    enum mtx_object object);

/**
 * `mtx_format' is used to enumerate different kinds of Matrix Market
 * formats.
 */
enum mtx_format
{
    mtx_array,     /* array of dense matrix values */
    mtx_coordinate /* coordinate format of sparse matrix values */
};

/**
 * `mtx_format_str()' is a string representing the Matrix Market
 * format type.
 */
const char * mtx_format_str(
    enum mtx_format format);

/**
 * `mtx_field' is used to enumerate different kinds of fields for
 * matrix values in Matrix Market files.
 */
enum mtx_field
{
    mtx_real,    /* real, floating-point coefficients */
    mtx_complex, /* complex, floating point coefficients */
    mtx_integer, /* integer coefficients */
    mtx_pattern  /* boolean coefficients (sparsity pattern) */
};

/**
 * `mtx_field_str()' is a string representing the Matrix Market field
 * type.
 */
const char * mtx_field_str(
    enum mtx_field field);

/**
 * `mtx_symmetry' is used to enumerate different kinds of symmetry for
 * matrices in Matrix Market format.
 */
enum mtx_symmetry
{
    mtx_general,        /* general, non-symmetric matrix */
    mtx_symmetric,      /* symmetric matrix */
    mtx_skew_symmetric, /* skew-symmetric matrix */
    mtx_hermitian       /* Hermitian matrix */
};

/**
 * `mtx_symmetry_str()' is a string representing the Matrix Market
 * symmetry type.
 */
const char * mtx_symmetry_str(
    enum mtx_symmetry symmetry);

/*
 * Matrix Market header.
 */

/**
 * `mtx_header' represents the header line of a Matrix Market file.
 */
struct mtx_header
{
    /**
     * `object' is the type of Matrix Market object: `matrix' or
     * `vector'.
     */
    enum mtx_object object;

    /**
     * `format' is the matrix format: `coordinate' or `array'.
     */
    enum mtx_format format;

    /**
     * `field' is the matrix field: `real', `complex', `integer' or
     * `pattern'.
     */
    enum mtx_field field;

    /**
     * `symmetry' is the matrix symmetry: `general', `symmetric',
     * `skew-symmetric', or `hermitian'.
     *
     * Note that if `symmetry' is `symmetric', `skew-symmetric' or
     * `hermitian', then the matrix must be square, so that `num_rows'
     * is equal to `num_columns'.
     */
    enum mtx_symmetry symmetry;
};

/**
 * `mtx_header_parse()' parses a string containing the header line for
 * a file in Matrix Market format.
 *
 * If `endptr' is not `NULL', then the address stored in `endptr'
 * points to the first character beyond the characters that were
 * consumed during parsing.
 *
 * On success, `mtx_header_parse()' returns `MTX_SUCCESS' and the
 * `object', `format', `field' and `symmetry' fields of the header
 * will be set according to the contents of the parsed Matrix Market
 * header.  Otherwise, an appropriate error code is returned if the
 * input is not a valid Matrix Market header.
 */
int mtx_header_parse(
    struct mtx_header * header,
    const char * line,
    int * bytes_read,
    const char ** endptr);

/*
 * Matrix Market comment lines.
 */

/**
 * `mtx_comments' represents the comment lines of a Matrix Market
 * file.
 */
struct mtx_comments
{
    /**
     * `num_comment_lines' is the number of comment lines.
     */
    int num_comment_lines;

    /**
     * `comment_lines' is an array containing comment lines.
     */
    char ** comment_lines;
};

/**
 * `mtx_comments_alloc()' allocates storage for comment lines.
 */
int mtx_comments_alloc(
    struct mtx_comments * comments,
    int num_comment_lines,
    int * len);

/**
 * `mtx_comments_init()' allocates storage for comment lines and
 * copies contents from the given array of strings.
 *
 * Note that each string in `comment_lines' must begin with '%'.
 */
int mtx_comments_init(
    struct mtx_comments * comments,
    int num_comment_lines,
    const char ** comment_lines);

/**
 * `mtx_comments_free()` frees storage used for comment lines.
 */
void mtx_comments_free(
    struct mtx_comments * comments);

/**
 * `mtx_comments_copy()' copies the given comment lines.
 */
int mtx_comments_copy(
    struct mtx_comments * dst,
    const struct mtx_comments * src);

/*
 * Matrix Market size line. 
 */

/**
 * `mtx_size' represents the size line of a Matrix Market file.
 */
struct mtx_size
{
    /**
     * `num_rows' is the number of rows in the matrix or vector.
     */
    int num_rows;

    /**
     * `num_columns' is the number of columns in the matrix if
     * `object' is `matrix'. Otherwise, if `object' is `vector', then
     * `num_columns' is equal to `-1'.
     */
    int num_columns;

    /**
     * `num_nonzeros' is the number of nonzero matrix or vector
     * entries for a sparse matrix or vector.  This only includes
     * entries that are stored explicitly, and not those that are
     * implicitly, for example, due to symmetry.
     *
     * If `format' is `array', then `num_nonzeros' is set to `-1', and
     * it is not used.
     */
    int64_t num_nonzeros;
};

/**
 * `mtx_size_parse()' parses a string containing the size line for a
 * file in Matrix Market format.  Note that the `object' and `format'
 * fields from the Matrix Market header are required, since the format
 * of the size line depends on them.
 *
 * If `endptr' is not `NULL', then the address stored in `endptr'
 * points to the first character beyond the characters that were
 * consumed during parsing.
 *
 * On success, `mtx_size_parse()' returns `MTX_SUCCESS' and the
 * `num_rows', `num_columns' and `num_nonzeros' fields of the size
 * line will be set according to the parsed contents.  Otherwise, an
 * appropriate error code is returned if the input is not a valid
 * Matrix Market size line.
 */
int mtx_size_parse(
    struct mtx_size * size,
    enum mtx_object object,
    enum mtx_format format,
    const char * line,
    int * bytes_read,
    const char ** endptr);

#endif
