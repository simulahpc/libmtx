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
 * Dense matrices in Matrix Market array format.
 */

#include <libmtx/matrix/array/array.h>

#include <libmtx/error.h>
#include <libmtx/mtx.h>
#include <libmtx/mtx/header.h>
#include <libmtx/mtx/sort.h>
#include <libmtx/mtx/triangle.h>

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
    enum mtx_triangle triangle,
    int num_rows,
    int num_columns,
    int64_t * size)
{
    /*
     * Compute the number of stored nonzeros, while ensuring that a
     * symmetric, skew-symmetric or Hermitian matrix is also square.
     */
    if (symmetry == mtx_general) {
        if (triangle != mtx_nontriangular)
            return MTX_ERR_INVALID_MTX_TRIANGLE;

        if (__builtin_mul_overflow(
                num_rows, num_columns, size))
        {
            errno = EOVERFLOW;
            return MTX_ERR_ERRNO;
        }

    } else if (symmetry == mtx_symmetric ||
               symmetry == mtx_hermitian)
    {
        if (triangle == mtx_lower_triangular) {
            if (num_rows <= num_columns) {
                if (__builtin_mul_overflow(
                        num_rows, num_rows+1, size))
                {
                    errno = EOVERFLOW;
                    return MTX_ERR_ERRNO;
                }
                *size /= 2;
            } else {
                int a;
                if (__builtin_mul_overflow(
                        num_columns, num_columns+1, &a))
                {
                    errno = EOVERFLOW;
                    return MTX_ERR_ERRNO;
                }
                a /= 2;

                int b;
                if (__builtin_mul_overflow(
                        num_rows-num_columns, num_columns, &b))
                {
                    errno = EOVERFLOW;
                    return MTX_ERR_ERRNO;
                }
                *size = a + b;
            }
        } else if (triangle == mtx_upper_triangular) {
            if (num_columns <= num_rows) {
                if (__builtin_mul_overflow(
                        num_columns, num_columns+1, size))
                {
                    errno = EOVERFLOW;
                    return MTX_ERR_ERRNO;
                }
                *size /= 2;
            } else {
                int a;
                if (__builtin_mul_overflow(
                        num_rows, num_rows+1, &a))
                {
                    errno = EOVERFLOW;
                    return MTX_ERR_ERRNO;
                }
                a /= 2;

                int b;
                if (__builtin_mul_overflow(
                        num_columns-num_rows, num_rows, &b))
                {
                    errno = EOVERFLOW;
                    return MTX_ERR_ERRNO;
                }
                *size = a + b;
            }
        } else {
            return MTX_ERR_INVALID_MTX_TRIANGLE;
        }

    } else if (symmetry == mtx_skew_symmetric) {
        if (triangle == mtx_lower_triangular) {
            if (num_rows <= num_columns) {
                if (__builtin_mul_overflow(
                        num_rows, num_rows-1, size))
                {
                    errno = EOVERFLOW;
                    return MTX_ERR_ERRNO;
                }
                *size /= 2;
            } else {
                int a;
                if (__builtin_mul_overflow(
                        num_columns, num_columns-1, &a))
                {
                    errno = EOVERFLOW;
                    return MTX_ERR_ERRNO;
                }
                a /= 2;

                int b;
                if (__builtin_mul_overflow(
                        num_rows-num_columns, num_columns, &b))
                {
                    errno = EOVERFLOW;
                    return MTX_ERR_ERRNO;
                }
                *size = a + b;
            }
        } else if (triangle == mtx_upper_triangular) {
            if (num_columns <= num_rows) {
                if (__builtin_mul_overflow(
                        num_columns, num_columns-1, size))
                {
                    errno = EOVERFLOW;
                    return MTX_ERR_ERRNO;
                }
                *size /= 2;
            } else {
                int a;
                if (__builtin_mul_overflow(
                        num_rows, num_rows-1, &a))
                {
                    errno = EOVERFLOW;
                    return MTX_ERR_ERRNO;
                }
                a /= 2;

                int b;
                if (__builtin_mul_overflow(
                        num_columns-num_rows, num_rows, &b))
                {
                    errno = EOVERFLOW;
                    return MTX_ERR_ERRNO;
                }
                *size = a + b;
            }
        } else {
            return MTX_ERR_INVALID_MTX_TRIANGLE;
        }

    } else {
        return MTX_ERR_INVALID_MTX_SYMMETRY;
    }
    return MTX_SUCCESS;
}

/*
 * Dense matrix allocation.
 */

static int mtx_alloc_matrix_array_field(
    struct mtx * mtx,
    enum mtx_field field,
    enum mtx_symmetry symmetry,
    enum mtx_triangle triangle,
    enum mtx_sorting sorting,
    int num_comment_lines,
    const char ** comment_lines,
    int num_rows,
    int num_columns,
    int nonzero_size)
{
    int err;
    if (sorting != mtx_row_major &&
        sorting != mtx_column_major)
    {
        return MTX_ERR_INVALID_MTX_SORTING;
    }

    mtx->object = mtx_matrix;
    mtx->format = mtx_array;
    mtx->field = field;
    mtx->symmetry = symmetry;
    mtx->triangle = triangle;
    mtx->sorting = sorting;
    mtx->ordering = mtx_unordered;
    mtx->assembly = mtx_assembled;

    mtx->num_comment_lines = 0;
    mtx->comment_lines = NULL;
    err = mtx_set_comment_lines(mtx, num_comment_lines, comment_lines);
    if (err)
        return err;

    mtx->num_rows = num_rows;
    mtx->num_columns = num_columns;
    err = mtx_matrix_array_num_nonzeros(
        num_rows, num_columns, &mtx->num_nonzeros);
    if (err) {
        for (int i = 0; i < num_comment_lines; i++)
            free(mtx->comment_lines[i]);
        free(mtx->comment_lines);
        return err;
    }
    err = mtx_matrix_array_size(
        symmetry, triangle, num_rows, num_columns, &mtx->size);
    if (err) {
        for (int i = 0; i < num_comment_lines; i++)
            free(mtx->comment_lines[i]);
        free(mtx->comment_lines);
        return err;
    }

    mtx->nonzero_size = nonzero_size;
    mtx->data = malloc(mtx->size * mtx->nonzero_size);
    if (!mtx->data) {
        for (int i = 0; i < num_comment_lines; i++)
            free(mtx->comment_lines[i]);
        free(mtx->comment_lines);
        return MTX_ERR_ERRNO;
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_alloc_matrix_array_real()` allocates a dense matrix with real,
 * single-precision floating point coefficients.
 *
 * If `symmetry' is `mtx_symmetric', `mtx_skew_symmetric' or
 * `mtx_hermitian', then `triangle' must be either
 * `mtx_lower_triangular' or `mtx_upper_triangular' to indicate which
 * triangle of the matrix is stored in `data'.  Otherwise, if
 * `symmetry' is `mtx_general', then `triangle' must be
 * `mtx_nontriangular'.
 *
 * ´sorting' must be `mtx_row_major' or `mtx_column_major'.
 */
int mtx_alloc_matrix_array_real(
    struct mtx * mtx,
    enum mtx_symmetry symmetry,
    enum mtx_triangle triangle,
    enum mtx_sorting sorting,
    int num_comment_lines,
    const char ** comment_lines,
    int num_rows,
    int num_columns)
{
    return mtx_alloc_matrix_array_field(
        mtx, mtx_real, symmetry, triangle, sorting,
        num_comment_lines, comment_lines,
        num_rows, num_columns, sizeof(float));
}

/**
 * `mtx_alloc_matrix_array_double()` allocates a dense matrix with
 * real, double-precision floating point coefficients.
 *
 * If `symmetry' is `mtx_symmetric', `mtx_skew_symmetric' or
 * `mtx_hermitian', then `triangle' must be either
 * `mtx_lower_triangular' or `mtx_upper_triangular' to indicate which
 * triangle of the matrix is stored in `data'.  Otherwise, if
 * `symmetry' is `mtx_general', then `triangle' must be
 * `mtx_nontriangular'.
 *
 * ´sorting' must be `mtx_row_major' or `mtx_column_major'.
 */
int mtx_alloc_matrix_array_double(
    struct mtx * mtx,
    enum mtx_symmetry symmetry,
    enum mtx_triangle triangle,
    enum mtx_sorting sorting,
    int num_comment_lines,
    const char ** comment_lines,
    int num_rows,
    int num_columns)
{
    return mtx_alloc_matrix_array_field(
        mtx, mtx_double, symmetry, triangle, sorting,
        num_comment_lines, comment_lines,
        num_rows, num_columns, sizeof(double));
}

/**
 * `mtx_alloc_matrix_array_complex()` allocates a dense matrix with
 * complex, single-precision floating point coefficients.
 *
 * If `symmetry' is `mtx_symmetric', `mtx_skew_symmetric' or
 * `mtx_hermitian', then `triangle' must be either
 * `mtx_lower_triangular' or `mtx_upper_triangular' to indicate which
 * triangle of the matrix is stored in `data'.  Otherwise, if
 * `symmetry' is `mtx_general', then `triangle' must be
 * `mtx_nontriangular'.
 *
 * ´sorting' must be `mtx_row_major' or `mtx_column_major'.
 */
int mtx_alloc_matrix_array_complex(
    struct mtx * mtx,
    enum mtx_symmetry symmetry,
    enum mtx_triangle triangle,
    enum mtx_sorting sorting,
    int num_comment_lines,
    const char ** comment_lines,
    int num_rows,
    int num_columns)
{
    return mtx_alloc_matrix_array_field(
        mtx, mtx_complex, symmetry, triangle, sorting,
        num_comment_lines, comment_lines,
        num_rows, num_columns, 2*sizeof(float));
}

/**
 * `mtx_alloc_matrix_array_integer()` allocates a dense matrix with
 * integer coefficients.
 *
 * If `symmetry' is `mtx_symmetric', `mtx_skew_symmetric' or
 * `mtx_hermitian', then `triangle' must be either
 * `mtx_lower_triangular' or `mtx_upper_triangular' to indicate which
 * triangle of the matrix is stored in `data'.  Otherwise, if
 * `symmetry' is `mtx_general', then `triangle' must be
 * `mtx_nontriangular'.
 *
 * ´sorting' must be `mtx_row_major' or `mtx_column_major'.
 */
int mtx_alloc_matrix_array_integer(
    struct mtx * mtx,
    enum mtx_symmetry symmetry,
    enum mtx_triangle triangle,
    enum mtx_sorting sorting,
    int num_comment_lines,
    const char ** comment_lines,
    int num_rows,
    int num_columns)
{
    return mtx_alloc_matrix_array_field(
        mtx, mtx_integer, symmetry, triangle, sorting,
        num_comment_lines, comment_lines,
        num_rows, num_columns, sizeof(int));
}

/*
 * Dense matrix creation.
 */

/**
 * `mtx_init_matrix_array_real()` creates a dense matrix with real,
 * single-precision floating point coefficients.
 *
 * If `symmetry' is `symmetric', `skew-symmetric' or `hermitian', then
 * `triangle' must be either `lower-triangular' or `upper-triangular'
 * to indicate which triangle of the matrix is stored in `data'.
 * Otherwise, if `symmetry' is `general', then `triangle' must be
 * `nontriangular'.
 *
 * ´sorting' must be `mtx_row_major' or `mtx_column_major'.
 */
int mtx_init_matrix_array_real(
    struct mtx * mtx,
    enum mtx_symmetry symmetry,
    enum mtx_triangle triangle,
    enum mtx_sorting sorting,
    int num_comment_lines,
    const char ** comment_lines,
    int num_rows,
    int num_columns,
    const float * data)
{
    int err = mtx_alloc_matrix_array_real(
        mtx, symmetry, triangle, sorting,
        num_comment_lines, comment_lines,
        num_rows, num_columns);
    if (err)
        return err;
    for (int64_t i = 0; i < mtx->size; i++)
        ((float *) mtx->data)[i] = data[i];
    return MTX_SUCCESS;
}

/**
 * `mtx_init_matrix_array_double()` creates a dense matrix with real,
 * double-precision floating point coefficients.
 *
 * If `symmetry' is `symmetric', `skew-symmetric' or `hermitian', then
 * `triangle' must be either `lower-triangular' or `upper-triangular'
 * to indicate which triangle of the matrix is stored in `data'.
 * Otherwise, if `symmetry' is `general', then `triangle' must be
 * `nontriangular'.
 *
 * ´sorting' must be `mtx_row_major' or `mtx_column_major'.
 */
int mtx_init_matrix_array_double(
    struct mtx * mtx,
    enum mtx_symmetry symmetry,
    enum mtx_triangle triangle,
    enum mtx_sorting sorting,
    int num_comment_lines,
    const char ** comment_lines,
    int num_rows,
    int num_columns,
    const double * data)
{
    int err = mtx_alloc_matrix_array_double(
        mtx, symmetry, triangle, sorting,
        num_comment_lines, comment_lines,
        num_rows, num_columns);
    if (err)
        return err;
    for (int64_t i = 0; i < mtx->size; i++)
        ((double *) mtx->data)[i] = data[i];
    return MTX_SUCCESS;
}

/**
 * `mtx_init_matrix_array_complex()` creates a dense matrix with complex,
 * single-precision floating point coefficients.
 *
 * If `symmetry' is `symmetric', `skew-symmetric' or `hermitian', then
 * `triangle' must be either `lower-triangular' or `upper-triangular'
 * to indicate which triangle of the matrix is stored in `data'.
 * Otherwise, if `symmetry' is `general', then `triangle' must be
 * `nontriangular'.
 *
 * ´sorting' must be `mtx_row_major' or `mtx_column_major'.
 */
int mtx_init_matrix_array_complex(
    struct mtx * mtx,
    enum mtx_symmetry symmetry,
    enum mtx_triangle triangle,
    enum mtx_sorting sorting,
    int num_comment_lines,
    const char ** comment_lines,
    int num_rows,
    int num_columns,
    const float * data)
{
    int err = mtx_alloc_matrix_array_complex(
        mtx, symmetry, triangle, sorting,
        num_comment_lines, comment_lines,
        num_rows, num_columns);
    if (err)
        return err;
    for (int64_t i = 0; i < mtx->size; i++) {
        ((float *) mtx->data)[2*i+0] = data[2*i+0];
        ((float *) mtx->data)[2*i+1] = data[2*i+1];
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_init_matrix_array_integer()` creates a dense matrix with
 * integer coefficients.
 *
 * If `symmetry' is `symmetric', `skew-symmetric' or `hermitian', then
 * `triangle' must be either `lower-triangular' or `upper-triangular'
 * to indicate which triangle of the matrix is stored in `data'.
 * Otherwise, if `symmetry' is `general', then `triangle' must be
 * `nontriangular'.
 *
 * ´sorting' must be `mtx_row_major' or `mtx_column_major'.
 */
int mtx_init_matrix_array_integer(
    struct mtx * mtx,
    enum mtx_symmetry symmetry,
    enum mtx_triangle triangle,
    enum mtx_sorting sorting,
    int num_comment_lines,
    const char ** comment_lines,
    int num_rows,
    int num_columns,
    const int * data)
{
    int err = mtx_alloc_matrix_array_integer(
        mtx, symmetry, triangle, sorting,
        num_comment_lines, comment_lines,
        num_rows, num_columns);
    if (err)
        return err;
    for (int64_t i = 0; i < mtx->size; i++)
        ((int *) mtx->data)[i] = data[i];
    return MTX_SUCCESS;
}

/*
 * Other dense matrix functions.
 */

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

/**
 * `mtx_matrix_array_set_constant_real()' sets every value of a matrix
 * equal to a constant, single precision floating point number.
 */
int mtx_matrix_array_set_constant_real(
    struct mtx * mtx,
    float a)
{
    if (mtx->object != mtx_matrix)
        return MTX_ERR_INVALID_MTX_OBJECT;
    if (mtx->format != mtx_array)
        return MTX_ERR_INVALID_MTX_FORMAT;
    if (mtx->field != mtx_real)
        return MTX_ERR_INVALID_MTX_FIELD;

    float * data = (float *) mtx->data;
    for (int64_t k = 0; k < mtx->size; k++)
        data[k] = a;
    return MTX_SUCCESS;
}

/**
 * `mtx_matrix_array_set_constant_double()' sets every value of a
 * matrix equal to a constant, double precision floating point number.
 */
int mtx_matrix_array_set_constant_double(
    struct mtx * mtx,
    double a)
{
    if (mtx->object != mtx_matrix)
        return MTX_ERR_INVALID_MTX_OBJECT;
    if (mtx->format != mtx_array)
        return MTX_ERR_INVALID_MTX_FORMAT;
    if (mtx->field != mtx_double)
        return MTX_ERR_INVALID_MTX_FIELD;

    double * data = (double *) mtx->data;
    for (int64_t k = 0; k < mtx->size; k++)
        data[k] = a;
    return MTX_SUCCESS;
}

/**
 * `mtx_matrix_array_set_constant_complex()' sets every value of a
 * matrix equal to a constant, single precision floating point complex
 * number.
 */
int mtx_matrix_array_set_constant_complex(
    struct mtx * mtx,
    float a,
    float b)
{
    if (mtx->object != mtx_matrix)
        return MTX_ERR_INVALID_MTX_OBJECT;
    if (mtx->format != mtx_array)
        return MTX_ERR_INVALID_MTX_FORMAT;
    if (mtx->field != mtx_complex)
        return MTX_ERR_INVALID_MTX_FIELD;

    float * data = (float *) mtx->data;
    for (int64_t k = 0; k < mtx->size; k++) {
        data[2*k+0] = a;
        data[2*k+1] = b;
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_matrix_array_set_constant_integer()' sets every value of a
 * matrix equal to a constant integer.
 */
int mtx_matrix_array_set_constant_integer(
    struct mtx * mtx,
    int a)
{
    if (mtx->object != mtx_matrix)
        return MTX_ERR_INVALID_MTX_OBJECT;
    if (mtx->format != mtx_array)
        return MTX_ERR_INVALID_MTX_FORMAT;
    if (mtx->field != mtx_complex)
        return MTX_ERR_INVALID_MTX_FIELD;

    int * data = (int *) mtx->data;
    for (int64_t k = 0; k < mtx->size; k++)
        data[k] = a;
    return MTX_SUCCESS;
}
