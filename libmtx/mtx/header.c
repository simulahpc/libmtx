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
 * Last modified: 2021-08-09
 *
 * Data types for the Matrix Market header.
 */

#include <libmtx/mtx/header.h>

#include <libmtx/error.h>

#include "../util/parse.h"

#include <errno.h>

#include <stdlib.h>
#include <string.h>

/*
 * Data types for Matrix Market headers.
 */

/**
 * `mtx_object_str()` is a string representing the Matrix Market object
 * type.
 */
const char * mtx_object_str(
    enum mtx_object object)
{
    switch (object) {
    case mtx_matrix: return "matrix";
    case mtx_vector: return "vector";
    default: return "unknown";
    }
}

/**
 * `mtx_format_str()` is a string representing the Matrix Market format
 * type.
 */
const char * mtx_format_str(
    enum mtx_format format)
{
    switch (format) {
    case mtx_array: return "array";
    case mtx_coordinate: return "coordinate";
    default: return "unknown";
    }
}

/**
 * `mtx_field_str()` is a string representing the Matrix Market field
 * type.
 */
const char * mtx_field_str(
    enum mtx_field field)
{
    switch (field) {
    case mtx_real: return "real";
    case mtx_complex: return "complex";
    case mtx_integer: return "integer";
    case mtx_pattern: return "pattern";
    default: return "unknown";
    }
}

/**
 * `mtx_symmetry_str()` is a string representing the Matrix Market
 * symmetry type.
 */
const char * mtx_symmetry_str(
    enum mtx_symmetry symmetry)
{
    switch (symmetry) {
    case mtx_general: return "general";
    case mtx_symmetric: return "symmetric";
    case mtx_skew_symmetric: return "skew-symmetric";
    case mtx_hermitian: return "hermitian";
    default: return "unknown";
    }
}

/*
 * Matrix Market header.
 */

/**
 * `mtx_header_parse()' parses a string containing the header line for
 * a file in Matrix Market format.
 *
 * If `endptr' is not `NULL', then the address stored in `endptr'
 * points to the first character beyond the characters that were
 * consumed during parsing.
 *
 * On success, `mtx_header_parse()` returns `MTX_SUCCESS' and
 * `object', `format', `field' and `symmetry' will be set according to
 * the contents of the parsed Matrix Market header.  Otherwise, an
 * appropriate error code is returned if the input is not a valid
 * Matrix Market header.
 */
int mtx_header_parse(
    struct mtx_header * header,
    const char * s,
    int * bytes_read,
    const char ** endptr)
{
    int err;
    const char * t = s;

    *bytes_read = 0;
    /* Parse the identifier. */
    if (strncmp("%%MatrixMarket", t, strlen("%%MatrixMarket")) != 0)
        return MTX_ERR_INVALID_MTX_HEADER;
    while (*t != '\0' && *t != ' ')
        t++;
    while (*t == ' ')
        t++;
    *bytes_read = t-s;

    /* Parse the object type. */
    if (strncmp("matrix ", t, strlen("matrix ")) == 0) {
        header->object = mtx_matrix;
    } else if (strncmp("vector ", t, strlen("vector ")) == 0) {
        header->object = mtx_vector;
    } else {
        return MTX_ERR_INVALID_MTX_OBJECT;
    }
    while (*t != '\0' && *t != ' ')
        t++;
    while (*t == ' ')
        t++;
    *bytes_read = t-s;

    /* Parse the format. */
    if (strncmp("array ", t, strlen("array ")) == 0) {
        header->format = mtx_array;
    } else if (strncmp("coordinate ", t, strlen("coordinate ")) == 0) {
        header->format = mtx_coordinate;
    } else {
        return MTX_ERR_INVALID_MTX_FORMAT;
    }
    while (*t != '\0' && *t != ' ')
        t++;
    while (*t == ' ')
        t++;
    *bytes_read = t-s;

    /* Parse the field type. */
    if (strncmp("real ", t, strlen("real ")) == 0) {
        header->field = mtx_real;
    } else if (strncmp("complex ", t, strlen("complex ")) == 0) {
        header->field = mtx_complex;
    } else if (strncmp("integer ", t, strlen("integer ")) == 0) {
        header->field = mtx_integer;
    } else if (strncmp("pattern ", t, strlen("pattern ")) == 0) {
        header->field = mtx_pattern;
    } else {
        return MTX_ERR_INVALID_MTX_FIELD;
    }
    while (*t != '\0' && *t != ' ')
        t++;
    while (*t == ' ')
        t++;
    *bytes_read = t-s;

    /* Parse the symmetry type. */
    if (strcmp("general", t) == 0 || strcmp("general\n", t) == 0) {
        header->symmetry = mtx_general;
    } else if (strcmp("symmetric", t) == 0 || strcmp("symmetric\n", t) == 0) {
        header->symmetry = mtx_symmetric;
    } else if (strcmp("skew-symmetric", t) == 0 ||
               strcmp("skew-symmetric\n", t) == 0)
    {
        header->symmetry = mtx_skew_symmetric;
    } else if (strcmp("hermitian", t) == 0 || strcmp("hermitian\n", t) == 0 ||
               strcmp("Hermitian", t) == 0 || strcmp("Hermitian\n", t) == 0)
    {
        header->symmetry = mtx_hermitian;
    } else {
        return MTX_ERR_INVALID_MTX_SYMMETRY;
    }
    while (*t != '\0' && *t != '\n')
        t++;
    *bytes_read = t-s;

    if (endptr)
        *endptr = t;
    return MTX_SUCCESS;
}


/*
 * Matrix Market comment lines.
 */

/**
 * `mtx_comments_alloc()' allocates storage for comment lines.
 */
int mtx_comments_alloc(
    struct mtx_comments * comments,
    int num_comment_lines,
    int * len)
{
    comments->comment_lines = malloc(num_comment_lines * sizeof(char *));
    if (!comments->comment_lines)
        return MTX_ERR_ERRNO;

    for (int i = 0; i < num_comment_lines; i++) {
        comments->comment_lines[i] = malloc(len[i] * sizeof(char));
        if (!comments->comment_lines[i]) {
            for (int j = i-1; j >= 0; j--)
                free(comments->comment_lines[j]);
            free(comments->comment_lines);
            return MTX_ERR_ERRNO;
        }
    }
    comments->num_comment_lines = num_comment_lines;
    return MTX_SUCCESS;
}

/**
 * `mtx_comments_init()' allocates storage for comment lines and
 * copies contents from the given array of strings.
 *
 * Note that each string in `comment_lines' must begin with '%'.
 */
int mtx_comments_init(
    struct mtx_comments * comments,
    int  num_comment_lines,
    const char ** comment_lines)
{
    /* Copy the given comment lines. */
    comments->comment_lines = malloc(num_comment_lines * sizeof(char *));
    if (!comments->comment_lines)
        return MTX_ERR_ERRNO;
    for (int i = 0; i < num_comment_lines; i++) {
        if (strlen(comment_lines[i]) <= 0 || comment_lines[i][0] != '%') {
            for (int j = i-1; j >= 0; j--)
                free(comments->comment_lines[j]);
            free(comments->comment_lines);
            return MTX_ERR_INVALID_MTX_COMMENT;
        }

        comments->comment_lines[i] = strdup(comment_lines[i]);
        if (!comments->comment_lines[i]) {
            for (int j = i-1; j >= 0; j--)
                free(comments->comment_lines[j]);
            free(comments->comment_lines);
            return MTX_ERR_ERRNO;
        }
    }
    comments->num_comment_lines = num_comment_lines;
    return MTX_SUCCESS;
}

/**
 * `mtx_comments_free()` frees storage used for comment lines.
 */
void mtx_comments_free(
    struct mtx_comments * comments)
{
    for (int i = 0; i < comments->num_comment_lines; i++)
        free(comments->comment_lines[i]);
    free(comments->comment_lines);
}

/**
 * `mtx_comments_copy()' copies the given comment lines.
 */
int mtx_comments_copy(
    struct mtx_comments * dst,
    const struct mtx_comments * src)
{
    return mtx_comments_init(
        dst, src->num_comment_lines,
        (const char **) src->comment_lines);
}

/*
 * Matrix Market size line.
 */

/**
 * `mtx_size_parse_matrix_array()' parse a size line from a Matrix
 * Market file for a matrix in array format.
 */
static int mtx_size_parse_matrix_array(
    struct mtx_size * size,
    const char * line,
    int * bytes_read,
    const char ** endptr)
{
    int err;
    *bytes_read = 0;

    /* Parse the number of rows. */
    err = parse_int32(line, " ", &size->num_rows, endptr);
    if (err == EINVAL) {
        return MTX_ERR_INVALID_MTX_SIZE;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    }
    *bytes_read = *endptr - line;

    /* Parse the number of columns. */
    err = parse_int32(*endptr, "\n", &size->num_columns, endptr);
    if (err == EINVAL) {
        return MTX_ERR_INVALID_MTX_SIZE;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    }
    *bytes_read = *endptr - line;

    size->num_nonzeros = -1;
    return MTX_SUCCESS;
}

/**
 * `mtx_size_parse_matrix_coordinate()' parse a size line from a
 * Matrix Market file for a matrix in coordinate format.
 */
static int mtx_size_parse_matrix_coordinate(
    struct mtx_size * size,
    const char * line,
    int * bytes_read,
    const char ** endptr)
{
    int err;
    *bytes_read = 0;

    /* Parse the number of rows. */
    err = parse_int32(line, " ", &size->num_rows, endptr);
    if (err == EINVAL) {
        return MTX_ERR_INVALID_MTX_SIZE;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    }
    *bytes_read = (*endptr) - line;

    /* Parse the number of columns. */
    err = parse_int32(*endptr, " ", &size->num_columns, endptr);
    if (err == EINVAL) {
        return MTX_ERR_INVALID_MTX_SIZE;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    }
    *bytes_read = *endptr - line;

    /* Parse the number of stored nonzeros. */
    err = parse_int64(*endptr, "\n", &size->num_nonzeros, endptr);
    if (err == EINVAL) {
        return MTX_ERR_INVALID_MTX_SIZE;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    }
    *bytes_read = *endptr - line;
    return MTX_SUCCESS;
}

/**
 * `mtx_size_parse_vector_array()` parse a size line from a Matrix
 * Market file for a vector in array format.
 */
int mtx_size_parse_vector_array(
    struct mtx_size * size,
    const char * line,
    int * bytes_read,
    const char ** endptr)
{
    /* Parse the number of rows. */
    *bytes_read = 0;
    int err = parse_int32(line, "\n", &size->num_rows, endptr);
    if (err == EINVAL) {
        return MTX_ERR_INVALID_MTX_SIZE;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    }
    *bytes_read = *endptr - line;

    size->num_columns = -1;
    size->num_nonzeros = -1;
    return MTX_SUCCESS;
}

/**
 * `mtx_size_parse_vector_coordinate()` parses a size line from a
 * Matrix Market file for a vector in coordinate format.
 */
int mtx_size_parse_vector_coordinate(
    struct mtx_size * size,
    const char * line,
    int * bytes_read,
    const char ** endptr)
{
    int err;
    *bytes_read = 0;

    /* Parse the number of rows. */
    err = parse_int32(line, " ", &size->num_rows, endptr);
    if (err == EINVAL) {
        return MTX_ERR_INVALID_MTX_SIZE;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    }
    *bytes_read = *endptr - line;

    size->num_columns = -1;

    /* Parse the number of stored nonzeros. */
    err = parse_int64(*endptr, "\n", &size->num_nonzeros, endptr);
    if (err == EINVAL) {
        return MTX_ERR_INVALID_MTX_SIZE;
    } else if (err) {
        errno = err;
        return MTX_ERR_ERRNO;
    }
    *bytes_read = *endptr - line;
    return MTX_SUCCESS;
}

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
    const char ** endptr)
{
    int err;
    const char * t = line;
    if (object == mtx_matrix) {
        if (format == mtx_array) {
            err = mtx_size_parse_matrix_array(
                size, line, bytes_read, &t);
            if (err)
                return err;
        } else if (format == mtx_coordinate) {
            err = mtx_size_parse_matrix_coordinate(
                size, line, bytes_read, &t);
            if (err)
                return err;
        } else {
            return MTX_ERR_INVALID_MTX_FORMAT;
        }
    } else if (object == mtx_vector) {
        if (format == mtx_array) {
            err = mtx_size_parse_vector_array(
                size, line, bytes_read, &t);
            if (err)
                return err;
        } else if (format == mtx_coordinate) {
            err = mtx_size_parse_vector_coordinate(
                size, line, bytes_read, &t);
            if (err)
                return err;
        } else {
            return MTX_ERR_INVALID_MTX_FORMAT;
        }
    } else {
        return MTX_ERR_INVALID_MTX_OBJECT;
    }
    if (endptr)
        *endptr = t;
    return MTX_SUCCESS;
}
