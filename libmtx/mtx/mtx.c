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
 * Data structures for representing objects in Matrix Market format.
 */

#include <libmtx/mtx/mtx.h>

#include <libmtx/error.h>
#include <libmtx/mtx/header.h>
#include <libmtx/matrix/array.h>
#include <libmtx/matrix/coordinate.h>
#include <libmtx/vector/array.h>
#include <libmtx/vector/coordinate.h>

#include <errno.h>

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void mtx_free_data(
    struct mtx * mtx)
{
    if (mtx->object == mtx_matrix) {
        if (mtx->format == mtx_array) {
            mtx_matrix_array_data_free(
                &mtx->storage.matrix_array);
        } else if (mtx->format == mtx_coordinate) {
            mtx_matrix_coordinate_data_free(
                &mtx->storage.matrix_coordinate);
        }
    } else if (mtx->object == mtx_vector) {
        if (mtx->format == mtx_array) {
            mtx_vector_array_data_free(
                &mtx->storage.vector_array);
        } else if (mtx->format == mtx_coordinate) {
            mtx_vector_coordinate_data_free(
                &mtx->storage.vector_coordinate);
        }
    }
}

/**
 * `mtx_free()` frees resources associated with a Matrix Market
 * object.
 */
void mtx_free(
    struct mtx * mtx)
{
    mtx_free_data(mtx);
    for (int i = 0; i < mtx->num_comment_lines; i++)
        free(mtx->comment_lines[i]);
    free(mtx->comment_lines);
}

static int mtx_copy_alloc_data(
    struct mtx * dst,
    const struct mtx * src)
{
    if (src->object == mtx_matrix) {
        if (src->format == mtx_array) {
            return mtx_matrix_array_data_copy_alloc(
                &dst->storage.matrix_array,
                &src->storage.matrix_array);
        } else if (src->format == mtx_coordinate) {
            return mtx_matrix_coordinate_data_copy_alloc(
                &dst->storage.matrix_coordinate,
                &src->storage.matrix_coordinate);
        } else {
            return MTX_ERR_INVALID_MTX_FORMAT;
        }
    } else if (src->object == mtx_vector) {
        if (src->format == mtx_array) {
            return mtx_vector_array_data_copy_alloc(
                &dst->storage.vector_array,
                &src->storage.vector_array);
        } else if (src->format == mtx_coordinate) {
            return mtx_vector_coordinate_data_copy_alloc(
                &dst->storage.vector_coordinate,
                &src->storage.vector_coordinate);
        } else {
            return MTX_ERR_INVALID_MTX_FORMAT;
        }
    } else {
        return MTX_ERR_INVALID_MTX_OBJECT;
    }
}

/**
 * `mtx_copy_alloc()' allocates a copy of a matrix or vector without
 * copying the matrix or vector values.
 */
int mtx_copy_alloc(
    struct mtx * dst,
    const struct mtx * src)
{
    int err;
    dst->object = src->object;
    dst->format = src->format;
    dst->field = src->field;
    dst->symmetry = src->symmetry;
    dst->num_comment_lines = src->num_comment_lines;
    dst->comment_lines = malloc(dst->num_comment_lines * sizeof(char *));
    if (!dst->comment_lines)
        return MTX_ERR_ERRNO;
    for (int i = 0; i < dst->num_comment_lines; i++) {
        dst->comment_lines[i] = strdup(src->comment_lines[i]);
        if (!dst->comment_lines[i]) {
            for (int j = i-1; j >= 0; j--)
                free(dst->comment_lines[j]);
            free(dst->comment_lines);
            return MTX_ERR_ERRNO;
        }
    }
    dst->num_rows = src->num_rows;
    dst->num_columns = src->num_columns;
    dst->num_nonzeros = src->num_nonzeros;

    err = mtx_copy_alloc_data(dst, src);
    if (err) {
        for (int i = 0; i < dst->num_comment_lines; i++)
            free(dst->comment_lines[i]);
        free(dst->comment_lines);
        return err;
    }
    return MTX_SUCCESS;
}

static int mtx_copy_init_data(
    struct mtx * dst,
    const struct mtx * src)
{
    if (src->object == mtx_matrix) {
        if (src->format == mtx_array) {
            return mtx_matrix_array_data_copy_init(
                &dst->storage.matrix_array,
                &src->storage.matrix_array);
        } else if (src->format == mtx_coordinate) {
            return mtx_matrix_coordinate_data_copy_init(
                &dst->storage.matrix_coordinate,
                &src->storage.matrix_coordinate);
        } else {
            return MTX_ERR_INVALID_MTX_FORMAT;
        }
    } else if (src->object == mtx_vector) {
        if (src->format == mtx_array) {
            return mtx_vector_array_data_copy_init(
                &dst->storage.vector_array,
                &src->storage.vector_array);
        } else if (src->format == mtx_coordinate) {
            return mtx_vector_coordinate_data_copy_init(
                &dst->storage.vector_coordinate,
                &src->storage.vector_coordinate);
        } else {
            return MTX_ERR_INVALID_MTX_FORMAT;
        }
    } else {
        return MTX_ERR_INVALID_MTX_OBJECT;
    }
}

/**
 * `mtx_copy_init()' creates a copy of a matrix or vector and also
 * copies the matrix or vector values.
 */
int mtx_copy_init(
    struct mtx * dst,
    const struct mtx * src)
{
    int err;
    dst->object = src->object;
    dst->format = src->format;
    dst->field = src->field;
    dst->symmetry = src->symmetry;
    dst->num_comment_lines = src->num_comment_lines;
    dst->comment_lines = malloc(dst->num_comment_lines * sizeof(char *));
    if (!dst->comment_lines)
        return MTX_ERR_ERRNO;
    for (int i = 0; i < dst->num_comment_lines; i++) {
        dst->comment_lines[i] = strdup(src->comment_lines[i]);
        if (!dst->comment_lines[i]) {
            for (int j = i-1; j >= 0; j--)
                free(dst->comment_lines[j]);
            free(dst->comment_lines);
            return MTX_ERR_ERRNO;
        }
    }
    dst->num_rows = src->num_rows;
    dst->num_columns = src->num_columns;
    dst->num_nonzeros = src->num_nonzeros;

    err = mtx_copy_init_data(dst, src);
    if (err) {
        for (int i = 0; i < dst->num_comment_lines; i++)
            free(dst->comment_lines[i]);
        free(dst->comment_lines);
        return err;
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_set_comment_lines()' copies comment lines to a Matrix Market
 * object.
 *
 * Any storage associated with existing comment lines is freed.
 */
int mtx_set_comment_lines(
    struct mtx * mtx,
    int  num_comment_lines,
    const char ** comment_lines)
{
    /* Copy the given comment lines. */
    char ** comment_lines_copy = malloc(num_comment_lines * sizeof(char *));
    if (!comment_lines_copy)
        return MTX_ERR_ERRNO;
    for (int i = 0; i < num_comment_lines; i++) {
        int err = mtx_comments_validate(comment_lines[i]);
        if (err) {
            for (int j = i-1; j >= 0; j--)
                free(comment_lines_copy[j]);
            free(comment_lines_copy);
            return err;
        }

        comment_lines_copy[i] = strdup(comment_lines[i]);
        if (!comment_lines_copy[i]) {
            for (int j = i-1; j >= 0; j--)
                free(comment_lines_copy[j]);
            free(comment_lines_copy);
            return MTX_ERR_ERRNO;
        }
    }

    /* Free existing comment lines. */
    if (mtx->comment_lines) {
        for (int i = 0; i < mtx->num_comment_lines; i++)
            free(mtx->comment_lines[i]);
        free(mtx->comment_lines);
    }

    mtx->num_comment_lines = num_comment_lines;
    mtx->comment_lines = comment_lines_copy;
    return MTX_SUCCESS;
}

/**
 * `mtx_add_comment_line()' appends another comment line to a Matrix
 * Market object.
 */
int mtx_add_comment_line(
    struct mtx * mtx,
    const char * comment_line)
{
    int err = mtx_comments_validate(comment_line);
    if (err)
        return err;

    /* Copy the existing comment lines. */
    char ** comment_lines = malloc((mtx->num_comment_lines+1) * sizeof(char *));
    if (!comment_lines)
        return MTX_ERR_ERRNO;
    for (int i = 0; i < mtx->num_comment_lines; i++) {
        comment_lines[i] = strdup(mtx->comment_lines[i]);
        if (!comment_lines[i]) {
            for (int j = i-1; j >= 0; j--)
                free(comment_lines[j]);
            free(comment_lines);
            return MTX_ERR_ERRNO;
        }
    }

    /* Append the new line. */
    comment_lines[mtx->num_comment_lines] = strdup(comment_line);
    if (!comment_lines[mtx->num_comment_lines]) {
        for (int j = mtx->num_comment_lines-1; j >= 0; j--)
            free(comment_lines[j]);
        free(comment_lines);
        return MTX_ERR_ERRNO;
    }

    /* Replace the existing comment lines. */
    for (int i = 0; i < mtx->num_comment_lines; i++)
        free(mtx->comment_lines[i]);
    free(mtx->comment_lines);

    mtx->num_comment_lines = mtx->num_comment_lines+1;
    mtx->comment_lines = comment_lines;
    return MTX_SUCCESS;
}

/**
 * `mtx_add_comment_line_printf()' appends another comment line to a
 * Matrix Market object using a printf-like syntax.
 *
 * Note that because `format' is a printf-style format string, where
 * '%' is used to denote a format specifier, then `format' must begin
 * with "%%" to produce the initial '%' character that is required for
 * a comment line.  The `format' string must also end with a newline
 * character, '\n'.
 */
int mtx_add_comment_line_printf(
    struct mtx * mtx,
    const char * format,
    ...)
{
    int err;
    int n = strlen(format);
    if (n <= 0 || format[0] != '%')
        return MTX_ERR_INVALID_MTX_COMMENT;
    err = mtx_comments_validate(&format[1]);
    if (err)
        return err;

    va_list va;
    va_start(va, format);
    int len = vsnprintf(NULL, 0, format, va);
    va_end(va);
    if (len < 0)
        return MTX_ERR_ERRNO;

    char * s = malloc(len+1);
    if (!s)
        return MTX_ERR_ERRNO;

    va_start(va, format);
    int newlen = vsnprintf(s, len+1, format, va);
    va_end(va);
    if (newlen < 0 || len != newlen) {
        free(s);
        return MTX_ERR_ERRNO;
    }
    s[newlen] = '\0';

    err = mtx_add_comment_line(mtx, s);
    free(s);
    return err;
}

/**
 * `mtx_set_zero()' zeroes a matrix or vector.
 */
int mtx_set_zero(
    struct mtx * mtx)
{
    if (mtx->object == mtx_matrix) {
        if (mtx->format == mtx_array) {
            return mtx_matrix_array_set_zero(mtx);
        } else if (mtx->format == mtx_coordinate) {
            return mtx_matrix_coordinate_set_zero(mtx);
        } else {
            return MTX_ERR_INVALID_MTX_FORMAT;
        }
    } else if (mtx->object == mtx_vector) {
        if (mtx->format == mtx_array) {
            return mtx_vector_array_set_zero(mtx);
        } else if (mtx->format == mtx_coordinate) {
            return mtx_vector_coordinate_set_zero(mtx);
        } else {
            return MTX_ERR_INVALID_MTX_FORMAT;
        }
    } else {
        return MTX_ERR_INVALID_MTX_OBJECT;
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_set_constant_real_single()' sets every (nonzero) value of a
 * matrix or vector equal to a constant, single precision floating
 * point number.
 */
int mtx_set_constant_real_single(
    struct mtx * mtx,
    float a)
{
    if (mtx->object == mtx_matrix) {
        if (mtx->format == mtx_array) {
            return mtx_matrix_array_set_constant_real_single(mtx, a);
        } else if (mtx->format == mtx_coordinate) {
            return mtx_matrix_coordinate_set_constant_real_single(mtx, a);
        } else {
            return MTX_ERR_INVALID_MTX_FORMAT;
        }
    } else if (mtx->object == mtx_vector) {
        if (mtx->format == mtx_array) {
            return mtx_vector_array_set_constant_real_single(mtx, a);
        } else if (mtx->format == mtx_coordinate) {
            return mtx_vector_coordinate_set_constant_real_single(mtx, a);
        } else {
            return MTX_ERR_INVALID_MTX_FORMAT;
        }
    } else {
        return MTX_ERR_INVALID_MTX_OBJECT;
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_set_constant_real_double()' sets every (nonzero) value of a
 * matrix or vector equal to a constant, double precision floating
 * point number.
 */
int mtx_set_constant_real_double(
    struct mtx * mtx,
    double a)
{
    if (mtx->object == mtx_matrix) {
        if (mtx->format == mtx_array) {
            return mtx_matrix_array_set_constant_real_double(mtx, a);
        } else if (mtx->format == mtx_coordinate) {
            return mtx_matrix_coordinate_set_constant_real_double(mtx, a);
        } else {
            return MTX_ERR_INVALID_MTX_FORMAT;
        }
    } else if (mtx->object == mtx_vector) {
        if (mtx->format == mtx_array) {
            return mtx_vector_array_set_constant_real_double(mtx, a);
        } else if (mtx->format == mtx_coordinate) {
            return mtx_vector_coordinate_set_constant_real_double(mtx, a);
        } else {
            return MTX_ERR_INVALID_MTX_FORMAT;
        }
    } else {
        return MTX_ERR_INVALID_MTX_OBJECT;
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_set_constant_complex_single()' sets every (nonzero) value of a
 * matrix or vector equal to a constant, single precision floating
 * point complex number.
 */
int mtx_set_constant_complex_single(
    struct mtx * mtx,
    float a[2])
{
    if (mtx->object == mtx_matrix) {
        if (mtx->format == mtx_array) {
            return mtx_matrix_array_set_constant_complex_single(mtx, a);
        } else if (mtx->format == mtx_coordinate) {
            return mtx_matrix_coordinate_set_constant_complex_single(mtx, a);
        } else {
            return MTX_ERR_INVALID_MTX_FORMAT;
        }
    } else if (mtx->object == mtx_vector) {
        if (mtx->format == mtx_array) {
            return mtx_vector_array_set_constant_complex_single(mtx, a);
        } else if (mtx->format == mtx_coordinate) {
            return mtx_vector_coordinate_set_constant_complex_single(mtx, a);
        } else {
            return MTX_ERR_INVALID_MTX_FORMAT;
        }
    } else {
        return MTX_ERR_INVALID_MTX_OBJECT;
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_set_constant_integer_single()' sets every (nonzero) value of a
 * matrix or vector equal to a constant integer.
 */
int mtx_set_constant_integer_single(
    struct mtx * mtx,
    int32_t a)
{
    if (mtx->object == mtx_matrix) {
        if (mtx->format == mtx_array) {
            return mtx_matrix_array_set_constant_integer_single(mtx, a);
        } else if (mtx->format == mtx_coordinate) {
            return mtx_matrix_coordinate_set_constant_integer_single(mtx, a);
        } else {
            return MTX_ERR_INVALID_MTX_FORMAT;
        }
    } else if (mtx->object == mtx_vector) {
        if (mtx->format == mtx_array) {
            return mtx_vector_array_set_constant_integer_single(mtx, a);
        } else if (mtx->format == mtx_coordinate) {
            return mtx_vector_coordinate_set_constant_integer_single(mtx, a);
        } else {
            return MTX_ERR_INVALID_MTX_FORMAT;
        }
    } else {
        return MTX_ERR_INVALID_MTX_OBJECT;
    }
    return MTX_SUCCESS;
}
