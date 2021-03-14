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
 * Last modified: 2021-06-18
 *
 * Sparse matrices in Matrix Market format.
 */

#include <matrixmarket/error.h>
#include <matrixmarket/matrix_coordinate.h>
#include <matrixmarket/mtx.h>
#include <matrixmarket/header.h>

#include <errno.h>

#include <stdlib.h>
#include <string.h>

/**
 * `mtx_matrix_coordinate_num_nonzeros()` computes the number of matrix
 * nonzeros, including those that are not stored explicitly due to
 * symmetry.
 */
int mtx_matrix_coordinate_num_nonzeros(
    enum mtx_field field,
    enum mtx_symmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const void * data,
    int64_t * num_nonzeros)
{
    /*
     * Compute the total number of nonzeros, while ensuring that a
     * symmetric, skew-symmetric or Hermitian matrix is also square.
     */
    if (symmetry == mtx_general) {
        *num_nonzeros = size;

    } else if (symmetry == mtx_symmetric ||
               symmetry == mtx_hermitian)
    {
        if (num_rows != num_columns)
            return MTX_ERR_INVALID_MTX_SIZE;
        if (__builtin_mul_overflow(
                2, size, num_nonzeros))
        {
            errno = EOVERFLOW;
            return MTX_ERR_ERRNO;
        }

        /*
         * Subtract the number of nonzeros on the main diagonal to
         * avoid counting them twice.
         */
        int64_t num_diagonal_nonzeros;
        int err = mtx_matrix_coordinate_num_diagonal_nonzeros(
            field, size, data,
            &num_diagonal_nonzeros);
        if (err)
            return err;
        *num_nonzeros -= num_diagonal_nonzeros;

    } else if (symmetry == mtx_skew_symmetric) {
        if (num_rows != num_columns)
            return MTX_ERR_INVALID_MTX_SIZE;
        if (__builtin_mul_overflow(
                2, size, num_nonzeros))
        {
            errno = EOVERFLOW;
            return MTX_ERR_ERRNO;
        }

    } else {
        errno = EINVAL;
        return MTX_ERR_ERRNO;
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_matrix_coordinate_num_diagonal_nonzeros()` counts the number of
 * nonzeros on the main diagonal of a sparse matrix in the Matrix
 * Market format.
 */
int mtx_matrix_coordinate_num_diagonal_nonzeros(
    enum mtx_field field,
    int64_t size,
    const void * data,
    int64_t * num_diagonal_nonzeros)
{
    *num_diagonal_nonzeros = 0;
    if (field == mtx_real) {
        const struct mtx_matrix_coordinate_real * a =
            (const struct mtx_matrix_coordinate_real *) data;
        for (int64_t k = 0; k < size; k++) {
            if (a[k].i == a[k].j)
                (*num_diagonal_nonzeros)++;
        }
    } else if (field == mtx_double) {
        const struct mtx_matrix_coordinate_double * a =
            (const struct mtx_matrix_coordinate_double *) data;
        for (int64_t k = 0; k < size; k++) {
            if (a[k].i == a[k].j)
                (*num_diagonal_nonzeros)++;
        }
    } else if (field == mtx_complex) {
        const struct mtx_matrix_coordinate_complex * a =
            (const struct mtx_matrix_coordinate_complex *) data;
        for (int64_t k = 0; k < size; k++) {
            if (a[k].i == a[k].j)
                (*num_diagonal_nonzeros)++;
        }
    } else if (field == mtx_integer) {
        const struct mtx_matrix_coordinate_integer * a =
            (const struct mtx_matrix_coordinate_integer *) data;
        for (int64_t k = 0; k < size; k++) {
            if (a[k].i == a[k].j)
                (*num_diagonal_nonzeros)++;
        }
    } else if (field == mtx_pattern) {
        const struct mtx_matrix_coordinate_pattern * a =
            (const struct mtx_matrix_coordinate_pattern *) data;
        for (int64_t k = 0; k < size; k++) {
            if (a[k].i == a[k].j)
                (*num_diagonal_nonzeros)++;
        }
    } else {
        errno = EINVAL;
        return MTX_ERR_ERRNO;
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_init_matrix_coordinate_real()` creates a sparse matrix with real,
 * single-precision floating point coefficients.
 */
int mtx_init_matrix_coordinate_real(
    struct mtx * matrix,
    enum mtx_symmetry symmetry,
    enum mtx_sorting sorting,
    enum mtx_ordering ordering,
    enum mtx_assembly assembly,
    int num_comment_lines,
    const char ** comment_lines,
    int num_rows,
    int num_columns,
    int64_t size,
    const struct mtx_matrix_coordinate_real * data)
{
    matrix->object = mtx_matrix;
    matrix->format = mtx_coordinate;
    matrix->field = mtx_real;
    matrix->symmetry = symmetry;
    matrix->sorting = sorting;
    matrix->ordering = ordering;
    matrix->assembly = assembly;

    /* Allocate storage for and copy comment lines. */
    matrix->num_comment_lines = num_comment_lines;
    matrix->comment_lines = malloc(
        num_comment_lines * sizeof(char *));
    if (!matrix->comment_lines)
        return MTX_ERR_ERRNO;
    for (int i = 0; i < num_comment_lines; i++)
        matrix->comment_lines[i] = strdup(comment_lines[i]);

    /* Compute the matrix size. */
    matrix->num_rows = num_rows;
    matrix->num_columns = num_columns;
    matrix->size = size;
    int err = mtx_matrix_coordinate_num_nonzeros(
        matrix->field, symmetry, num_rows, num_columns,
        size, data, &matrix->num_nonzeros);
    if (err) {
        for (int i = 0; i < num_comment_lines; i++)
            free(matrix->comment_lines[i]);
        free(matrix->comment_lines);
        return err;
    }

    /* Allocate storage for and copy matrix data. */
    matrix->nonzero_size = sizeof(struct mtx_matrix_coordinate_real);
    matrix->data = malloc(matrix->size * matrix->nonzero_size);
    if (!matrix->data) {
        for (int i = 0; i < num_comment_lines; i++)
            free(matrix->comment_lines[i]);
        free(matrix->comment_lines);
        return MTX_ERR_ERRNO;
    }
    for (int64_t i = 0; i < matrix->size; i++)
        ((struct mtx_matrix_coordinate_real *) matrix->data)[i] = data[i];
    return MTX_SUCCESS;
}

/**
 * `mtx_init_matrix_coordinate_double()` creates a sparse matrix with real,
 * double-precision floating point coefficients.
 */
int mtx_init_matrix_coordinate_double(
    struct mtx * matrix,
    enum mtx_symmetry symmetry,
    enum mtx_sorting sorting,
    enum mtx_ordering ordering,
    enum mtx_assembly assembly,
    int num_comment_lines,
    const char ** comment_lines,
    int num_rows,
    int num_columns,
    int64_t size,
    const struct mtx_matrix_coordinate_double * data)
{
    matrix->object = mtx_matrix;
    matrix->format = mtx_coordinate;
    matrix->field = mtx_double;
    matrix->symmetry = symmetry;
    matrix->sorting = sorting;
    matrix->ordering = ordering;
    matrix->assembly = assembly;

    /* Allocate storage for and copy comment lines. */
    matrix->num_comment_lines = num_comment_lines;
    matrix->comment_lines = malloc(
        num_comment_lines * sizeof(char *));
    if (!matrix->comment_lines)
        return MTX_ERR_ERRNO;
    for (int i = 0; i < num_comment_lines; i++)
        matrix->comment_lines[i] = strdup(comment_lines[i]);

    /* Compute the matrix size. */
    matrix->num_rows = num_rows;
    matrix->num_columns = num_columns;
    matrix->size = size;
    int err = mtx_matrix_coordinate_num_nonzeros(
        matrix->field, symmetry, num_rows, num_columns,
        size, data, &matrix->num_nonzeros);
    if (err) {
        for (int i = 0; i < num_comment_lines; i++)
            free(matrix->comment_lines[i]);
        free(matrix->comment_lines);
        return err;
    }

    /* Allocate storage for and copy matrix data. */
    matrix->nonzero_size = sizeof(struct mtx_matrix_coordinate_double);
    matrix->data = malloc(matrix->size * matrix->nonzero_size);
    if (!matrix->data) {
        for (int i = 0; i < num_comment_lines; i++)
            free(matrix->comment_lines[i]);
        free(matrix->comment_lines);
        return MTX_ERR_ERRNO;
    }
    for (int64_t i = 0; i < matrix->size; i++)
        ((struct mtx_matrix_coordinate_double *) matrix->data)[i] = data[i];
    return MTX_SUCCESS;
}

/**
 * `mtx_init_matrix_coordinate_complex()` creates a sparse matrix with complex,
 * single-precision floating point coefficients.
 */
int mtx_init_matrix_coordinate_complex(
    struct mtx * matrix,
    enum mtx_symmetry symmetry,
    enum mtx_sorting sorting,
    enum mtx_ordering ordering,
    enum mtx_assembly assembly,
    int num_comment_lines,
    const char ** comment_lines,
    int num_rows,
    int num_columns,
    int64_t size,
    const struct mtx_matrix_coordinate_complex * data)
{
    matrix->object = mtx_matrix;
    matrix->format = mtx_coordinate;
    matrix->field = mtx_complex;
    matrix->symmetry = symmetry;
    matrix->sorting = sorting;
    matrix->ordering = ordering;
    matrix->assembly = assembly;

    /* Allocate storage for and copy comment lines. */
    matrix->num_comment_lines = num_comment_lines;
    matrix->comment_lines = malloc(
        num_comment_lines * sizeof(char *));
    if (!matrix->comment_lines)
        return MTX_ERR_ERRNO;
    for (int i = 0; i < num_comment_lines; i++)
        matrix->comment_lines[i] = strdup(comment_lines[i]);

    /* Compute the matrix size. */
    matrix->num_rows = num_rows;
    matrix->num_columns = num_columns;
    matrix->size = size;
    int err = mtx_matrix_coordinate_num_nonzeros(
        matrix->field, symmetry, num_rows, num_columns,
        size, data, &matrix->num_nonzeros);
    if (err) {
        for (int i = 0; i < num_comment_lines; i++)
            free(matrix->comment_lines[i]);
        free(matrix->comment_lines);
        return err;
    }

    /* Allocate storage for and copy matrix data. */
    matrix->nonzero_size = sizeof(struct mtx_matrix_coordinate_complex);
    matrix->data = malloc(matrix->size * matrix->nonzero_size);
    if (!matrix->data) {
        for (int i = 0; i < num_comment_lines; i++)
            free(matrix->comment_lines[i]);
        free(matrix->comment_lines);
        return MTX_ERR_ERRNO;
    }
    for (int64_t i = 0; i < matrix->size; i++)
        ((struct mtx_matrix_coordinate_complex *) matrix->data)[i] = data[i];
    return MTX_SUCCESS;
}

/**
 * `mtx_init_matrix_coordinate_integer()` creates a sparse matrix with integer
 * coefficients.
 */
int mtx_init_matrix_coordinate_integer(
    struct mtx * matrix,
    enum mtx_symmetry symmetry,
    enum mtx_sorting sorting,
    enum mtx_ordering ordering,
    enum mtx_assembly assembly,
    int num_comment_lines,
    const char ** comment_lines,
    int num_rows,
    int num_columns,
    int64_t size,
    const struct mtx_matrix_coordinate_integer * data)
{
    matrix->object = mtx_matrix;
    matrix->format = mtx_coordinate;
    matrix->field = mtx_integer;
    matrix->symmetry = symmetry;
    matrix->sorting = sorting;
    matrix->ordering = ordering;
    matrix->assembly = assembly;

    /* Allocate storage for and copy comment lines. */
    matrix->num_comment_lines = num_comment_lines;
    matrix->comment_lines = malloc(
        num_comment_lines * sizeof(char *));
    if (!matrix->comment_lines)
        return MTX_ERR_ERRNO;
    for (int i = 0; i < num_comment_lines; i++)
        matrix->comment_lines[i] = strdup(comment_lines[i]);

    /* Compute the matrix size. */
    matrix->num_rows = num_rows;
    matrix->num_columns = num_columns;
    matrix->size = size;
    int err = mtx_matrix_coordinate_num_nonzeros(
        matrix->field, symmetry, num_rows, num_columns,
        size, data, &matrix->num_nonzeros);
    if (err) {
        for (int i = 0; i < num_comment_lines; i++)
            free(matrix->comment_lines[i]);
        free(matrix->comment_lines);
        return err;
    }

    /* Allocate storage for and copy matrix data. */
    matrix->nonzero_size = sizeof(struct mtx_matrix_coordinate_integer);
    matrix->data = malloc(matrix->size * matrix->nonzero_size);
    if (!matrix->data) {
        for (int i = 0; i < num_comment_lines; i++)
            free(matrix->comment_lines[i]);
        free(matrix->comment_lines);
        return MTX_ERR_ERRNO;
    }
    for (int64_t i = 0; i < matrix->size; i++)
        ((struct mtx_matrix_coordinate_integer *) matrix->data)[i] = data[i];
    return MTX_SUCCESS;
}

/**
 * `mtx_init_matrix_coordinate_pattern()` creates a sparse matrix with boolean
 * coefficients.
 */
int mtx_init_matrix_coordinate_pattern(
    struct mtx * matrix,
    enum mtx_symmetry symmetry,
    enum mtx_sorting sorting,
    enum mtx_ordering ordering,
    enum mtx_assembly assembly,
    int num_comment_lines,
    const char ** comment_lines,
    int num_rows,
    int num_columns,
    int64_t size,
    const struct mtx_matrix_coordinate_pattern * data)
{
    matrix->object = mtx_matrix;
    matrix->format = mtx_coordinate;
    matrix->field = mtx_pattern;
    matrix->symmetry = symmetry;
    matrix->sorting = sorting;
    matrix->ordering = ordering;
    matrix->assembly = assembly;

    /* Allocate storage for and copy comment lines. */
    matrix->num_comment_lines = num_comment_lines;
    matrix->comment_lines = malloc(
        num_comment_lines * sizeof(char *));
    if (!matrix->comment_lines)
        return MTX_ERR_ERRNO;
    for (int i = 0; i < num_comment_lines; i++)
        matrix->comment_lines[i] = strdup(comment_lines[i]);

    /* Compute the matrix size. */
    matrix->num_rows = num_rows;
    matrix->num_columns = num_columns;
    matrix->size = size;
    int err = mtx_matrix_coordinate_num_nonzeros(
        matrix->field, symmetry, num_rows, num_columns,
        size, data, &matrix->num_nonzeros);
    if (err) {
        for (int i = 0; i < num_comment_lines; i++)
            free(matrix->comment_lines[i]);
        free(matrix->comment_lines);
        return err;
    }

    /* Allocate storage for and copy matrix data. */
    matrix->nonzero_size = sizeof(struct mtx_matrix_coordinate_pattern);
    matrix->data = malloc(matrix->size * matrix->nonzero_size);
    if (!matrix->data) {
        for (int i = 0; i < num_comment_lines; i++)
            free(matrix->comment_lines[i]);
        free(matrix->comment_lines);
        return MTX_ERR_ERRNO;
    }
    for (int64_t i = 0; i < matrix->size; i++)
        ((struct mtx_matrix_coordinate_pattern *) matrix->data)[i] = data[i];
    return MTX_SUCCESS;
}
