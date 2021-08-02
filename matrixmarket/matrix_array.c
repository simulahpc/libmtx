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
 * Last modified: 2021-08-02
 *
 * Dense matrices in Matrix Market format.
 */

#include <matrixmarket/error.h>
#include <matrixmarket/matrix_array.h>
#include <matrixmarket/mtx.h>
#include <matrixmarket/header.h>

#include <errno.h>

#include <stdlib.h>
#include <string.h>

/**
 * `mtx_matrix_array_num_nonzeros()` computes the number of matrix
 * nonzeros, including those not explicitly stored due to symmetry.
 */
int mtx_matrix_array_num_nonzeros(
    int num_rows,
    int num_columns,
    int64_t * num_nonzeros)
{
    if (__builtin_mul_overflow(
            num_rows, num_columns, num_nonzeros))
    {
        errno = EOVERFLOW;
        return MTX_ERR_ERRNO;
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_matrix_array_size()` computes the number of matrix nonzeros,
 * excluding those that are not stored explicitly due to symmetry.
 */
int mtx_matrix_array_size(
    enum mtx_symmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t * size)
{
    /* Compute the number of nonzeros. */
    int64_t num_nonzeros;
    int err = mtx_matrix_array_num_nonzeros(
        num_rows, num_columns, &num_nonzeros);
    if (err)
        return err;

    /*
     * Compute the number of stored nonzeros, while ensuring that a
     * symmetric, skew-symmetric or Hermitian matrix is also square.
     */
    if (symmetry == mtx_general) {
        *size = num_nonzeros;

    } else if (symmetry == mtx_symmetric ||
               symmetry == mtx_hermitian)
    {
        if (num_rows != num_columns)
            return MTX_ERR_INVALID_MTX_SIZE;
        if (__builtin_mul_overflow(
                num_rows+1, num_columns,
                size))
        {
            errno = EOVERFLOW;
            return MTX_ERR_ERRNO;
        }
        *size /= 2;

    } else if (symmetry == mtx_skew_symmetric) {
        if (num_rows != num_columns)
            return MTX_ERR_INVALID_MTX_SIZE;
        if (__builtin_mul_overflow(
                num_rows+1, num_columns,
                size))
        {
            errno = EOVERFLOW;
            return MTX_ERR_ERRNO;
        }
        *size /= 2;
        *size = num_nonzeros - *size;

    } else {
        return MTX_ERR_INVALID_MTX_SYMMETRY;
    }

    return MTX_SUCCESS;
}

/**
 * `mtx_init_matrix_array_real()` creates a dense matrix with real,
 * single-precision floating point coefficients.
 *
 * ´sorting' must be `mtx_row_major' or `mtx_column_major'.
 */
int mtx_init_matrix_array_real(
    struct mtx * matrix,
    int num_comment_lines,
    const char ** comment_lines,
    enum mtx_symmetry symmetry,
    enum mtx_sorting sorting,
    int num_rows,
    int num_columns,
    const float * data)
{
    int err;

    if (sorting != mtx_row_major &&
        sorting != mtx_column_major)
    {
        return MTX_ERR_INVALID_MTX_SORTING;
    }

    matrix->object = mtx_matrix;
    matrix->format = mtx_array;
    matrix->field = mtx_real;
    matrix->symmetry = symmetry;
    matrix->sorting = sorting;
    matrix->ordering = mtx_unordered;
    matrix->assembly = mtx_assembled;

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
    err = mtx_matrix_array_num_nonzeros(
        num_rows, num_columns, &matrix->num_nonzeros);
    if (err) {
        for (int i = 0; i < num_comment_lines; i++)
            free(matrix->comment_lines[i]);
        free(matrix->comment_lines);
        return err;
    }
    err = mtx_matrix_array_size(
        symmetry, num_rows, num_columns, &matrix->size);
    if (err) {
        for (int i = 0; i < num_comment_lines; i++)
            free(matrix->comment_lines[i]);
        free(matrix->comment_lines);
        return err;
    }

    /* Allocate storage for and copy matrix data. */
    matrix->nonzero_size = sizeof(float);
    matrix->data = malloc(matrix->size * matrix->nonzero_size);
    if (!matrix->data) {
        for (int i = 0; i < num_comment_lines; i++)
            free(matrix->comment_lines[i]);
        free(matrix->comment_lines);
        return MTX_ERR_ERRNO;
    }
    for (int64_t i = 0; i < matrix->size; i++)
        ((float *) matrix->data)[i] = data[i];

    return MTX_SUCCESS;
}

/**
 * `mtx_init_matrix_array_double()` creates a dense matrix with real,
 * double-precision floating point coefficients.
 *
 * ´sorting' must be `mtx_row_major' or `mtx_column_major'.
 */
int mtx_init_matrix_array_double(
    struct mtx * matrix,
    int num_comment_lines,
    const char ** comment_lines,
    enum mtx_symmetry symmetry,
    enum mtx_sorting sorting,
    int num_rows,
    int num_columns,
    const double * data)
{
    int err;

    if (sorting != mtx_row_major &&
        sorting != mtx_column_major)
    {
        return MTX_ERR_INVALID_MTX_SORTING;
    }

    matrix->object = mtx_matrix;
    matrix->format = mtx_array;
    matrix->field = mtx_double;
    matrix->symmetry = symmetry;
    matrix->sorting = sorting;
    matrix->ordering = mtx_unordered;
    matrix->assembly = mtx_assembled;

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
    err = mtx_matrix_array_num_nonzeros(
        num_rows, num_columns, &matrix->num_nonzeros);
    if (err) {
        for (int i = 0; i < num_comment_lines; i++)
            free(matrix->comment_lines[i]);
        free(matrix->comment_lines);
        return err;
    }
    err = mtx_matrix_array_size(
        symmetry, num_rows, num_columns, &matrix->size);
    if (err) {
        for (int i = 0; i < num_comment_lines; i++)
            free(matrix->comment_lines[i]);
        free(matrix->comment_lines);
        return err;
    }

    /* Allocate storage for and copy matrix data. */
    matrix->nonzero_size = sizeof(double);
    matrix->data = malloc(matrix->size * matrix->nonzero_size);
    if (!matrix->data) {
        for (int i = 0; i < num_comment_lines; i++)
            free(matrix->comment_lines[i]);
        free(matrix->comment_lines);
        return MTX_ERR_ERRNO;
    }
    for (int64_t i = 0; i < matrix->size; i++)
        ((double *) matrix->data)[i] = data[i];

    return MTX_SUCCESS;
}

/**
 * `mtx_init_matrix_array_complex()` creates a dense matrix with complex,
 * single-precision floating point coefficients.
 *
 * ´sorting' must be `mtx_row_major' or `mtx_column_major'.
 */
int mtx_init_matrix_array_complex(
    struct mtx * matrix,
    int num_comment_lines,
    const char ** comment_lines,
    enum mtx_symmetry symmetry,
    enum mtx_sorting sorting,
    int num_rows,
    int num_columns,
    const float * data)
{
    int err;

    if (sorting != mtx_row_major &&
        sorting != mtx_column_major)
    {
        return MTX_ERR_INVALID_MTX_SORTING;
    }

    matrix->object = mtx_matrix;
    matrix->format = mtx_array;
    matrix->field = mtx_complex;
    matrix->symmetry = symmetry;
    matrix->sorting = sorting;
    matrix->ordering = mtx_unordered;
    matrix->assembly = mtx_assembled;

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
    err = mtx_matrix_array_num_nonzeros(
        num_rows, num_columns, &matrix->num_nonzeros);
    if (err) {
        for (int i = 0; i < num_comment_lines; i++)
            free(matrix->comment_lines[i]);
        free(matrix->comment_lines);
        return err;
    }
    err = mtx_matrix_array_size(
        symmetry, num_rows, num_columns, &matrix->size);
    if (err) {
        for (int i = 0; i < num_comment_lines; i++)
            free(matrix->comment_lines[i]);
        free(matrix->comment_lines);
        return err;
    }

    /* Allocate storage for and copy matrix data. */
    matrix->nonzero_size = 2*sizeof(float);
    matrix->data = malloc(matrix->size * matrix->nonzero_size);
    if (!matrix->data) {
        for (int i = 0; i < num_comment_lines; i++)
            free(matrix->comment_lines[i]);
        free(matrix->comment_lines);
        return MTX_ERR_ERRNO;
    }
    for (int64_t i = 0; i < matrix->size; i++) {
        ((float *) matrix->data)[2*i+0] = data[2*i+0];
        ((float *) matrix->data)[2*i+1] = data[2*i+1];
    }

    return MTX_SUCCESS;
}

/**
 * `mtx_init_matrix_array_integer()` creates a dense matrix with integer
 * coefficients.
 *
 * ´sorting' must be `mtx_row_major' or `mtx_column_major'.
 */
int mtx_init_matrix_array_integer(
    struct mtx * matrix,
    int num_comment_lines,
    const char ** comment_lines,
    enum mtx_symmetry symmetry,
    enum mtx_sorting sorting,
    int num_rows,
    int num_columns,
    const int * data)
{
    int err;

    if (sorting != mtx_row_major &&
        sorting != mtx_column_major)
    {
        return MTX_ERR_INVALID_MTX_SORTING;
    }

    matrix->object = mtx_matrix;
    matrix->format = mtx_array;
    matrix->field = mtx_integer;
    matrix->symmetry = symmetry;
    matrix->sorting = sorting;
    matrix->ordering = mtx_unordered;
    matrix->assembly = mtx_assembled;

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
    err = mtx_matrix_array_num_nonzeros(
        num_rows, num_columns, &matrix->num_nonzeros);
    if (err) {
        for (int i = 0; i < num_comment_lines; i++)
            free(matrix->comment_lines[i]);
        free(matrix->comment_lines);
        return err;
    }
    err = mtx_matrix_array_size(
        symmetry, num_rows, num_columns, &matrix->size);
    if (err) {
        for (int i = 0; i < num_comment_lines; i++)
            free(matrix->comment_lines[i]);
        free(matrix->comment_lines);
        return err;
    }

    /* Allocate storage for and copy matrix data. */
    matrix->nonzero_size = sizeof(int);
    matrix->data = malloc(matrix->size * matrix->nonzero_size);
    if (!matrix->data) {
        for (int i = 0; i < num_comment_lines; i++)
            free(matrix->comment_lines[i]);
        free(matrix->comment_lines);
        return MTX_ERR_ERRNO;
    }
    for (int64_t i = 0; i < matrix->size; i++)
        ((int *) matrix->data)[i] = data[i];

    return MTX_SUCCESS;
}

/**
 * `mtx_matrix_array_set_zero()' zeroes a matrix in array format.
 */
int mtx_matrix_array_set_zero(
    struct mtx * mtx)
{
    if (mtx->object != mtx_matrix)
        return MTX_ERR_INVALID_MTX_OBJECT;
    if (mtx->format != mtx_array)
        return MTX_ERR_INVALID_MTX_FORMAT;

    if (mtx->field == mtx_real) {
        float * data = (float *) mtx->data;
        for (int64_t k = 0; k < mtx->size; k++)
            data[k] = 0;
    } else if (mtx->field == mtx_double) {
        double * data = (double *) mtx->data;
        for (int64_t k = 0; k < mtx->size; k++)
            data[k] = 0;
    } else if (mtx->field == mtx_complex) {
        float * data = (float *) mtx->data;
        for (int64_t k = 0; k < mtx->size; k++) {
            data[2*k+0] = 0;
            data[2*k+1] = 0;
        }
    } else if (mtx->field == mtx_integer) {
        int * data = (int *) mtx->data;
        for (int64_t k = 0; k < mtx->size; k++)
            data[k] = 0;
    } else {
        return MTX_ERR_INVALID_MTX_FIELD;
    }
    return MTX_SUCCESS;
}
