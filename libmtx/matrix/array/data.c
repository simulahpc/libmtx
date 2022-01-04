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
 * Last modified: 2021-08-16
 *
 * Data structures for matrices in array format.
 */

#include <libmtx/matrix/array/data.h>

#include <libmtx/error.h>
#include <libmtx/mtx/header.h>
#include <libmtx/mtx/precision.h>
#include <libmtx/mtx/sort.h>
#include <libmtx/mtx/triangle.h>

#include <errno.h>

#include <stdint.h>
#include <stdlib.h>

/**
 * `mtx_matrix_array_data_free()' frees resources associated with a
 * Matrix Market object.
 */
void mtx_matrix_array_data_free(
    struct mtx_matrix_array_data * mtxdata)
{
    if (mtxdata->field == mtx_real) {
        if (mtxdata->precision == mtx_single) {
            free(mtxdata->data.real_single);
        } else if (mtxdata->precision == mtx_double) {
            free(mtxdata->data.real_double);
        }
    } else if (mtxdata->field == mtx_complex) {
        if (mtxdata->precision == mtx_single) {
            free(mtxdata->data.complex_single);
        } else if (mtxdata->precision == mtx_double) {
            free(mtxdata->data.complex_double);
        }
    } else if (mtxdata->field == mtx_integer) {
        if (mtxdata->precision == mtx_single) {
            free(mtxdata->data.integer_single);
        } else if (mtxdata->precision == mtx_double) {
            free(mtxdata->data.integer_double);
        }
    }
}

static int mtx_matrix_array_data_size(
    enum mtx_triangle triangle,
    int num_rows,
    int num_columns,
    int64_t * size)
{
    if (triangle == mtx_nontriangular) {
        if (__builtin_mul_overflow(num_rows, num_columns, size)) {
            errno = EOVERFLOW;
            return MTX_ERR_ERRNO;
        }

    } else if (triangle == mtx_lower_triangular) {
        if (num_rows <= num_columns) {
            if (num_rows >= (INT64_MAX-1) ||
                __builtin_mul_overflow(num_rows, (num_rows+1), size))
            {
                errno = EOVERFLOW;
                return MTX_ERR_ERRNO;
            }
            *size /= 2;
        } else {
            int64_t upper;
            if (num_columns >= (INT64_MAX-1) ||
                __builtin_mul_overflow(num_columns, (num_columns+1), &upper))
            {
                errno = EOVERFLOW;
                return MTX_ERR_ERRNO;
            }
            upper /= 2;
            int64_t lower;
            if (__builtin_mul_overflow((num_rows-num_columns), num_columns, &lower)) {
                errno = EOVERFLOW;
                return MTX_ERR_ERRNO;
            }
            if (__builtin_add_overflow(lower, upper, size)) {
                errno = EOVERFLOW;
                return MTX_ERR_ERRNO;
            }
        }

    } else if (triangle == mtx_upper_triangular) {
        if (num_columns <= num_rows) {
            if (num_columns >= (INT64_MAX-1) ||
                __builtin_mul_overflow(num_columns, (num_columns+1), size))
            {
                errno = EOVERFLOW;
                return MTX_ERR_ERRNO;
            }
            *size /= 2;
        } else {
            int64_t left;
            if (num_rows >= (INT64_MAX-1) ||
                __builtin_mul_overflow(num_rows, (num_rows+1), &left))
            {
                errno = EOVERFLOW;
                return MTX_ERR_ERRNO;
            }
            left /= 2;
            int64_t right;
            if (__builtin_mul_overflow((num_columns-num_rows), num_rows, &right)) {
                errno = EOVERFLOW;
                return MTX_ERR_ERRNO;
            }
            if (__builtin_add_overflow(right, left, size)) {
                errno = EOVERFLOW;
                return MTX_ERR_ERRNO;
            }
        }

    } else if (triangle == mtx_strict_lower_triangular) {
        if (num_rows <= num_columns) {
            if (__builtin_mul_overflow(num_rows, (num_rows-1), size)) {
                errno = EOVERFLOW;
                return MTX_ERR_ERRNO;
            }
            *size /= 2;
        } else {
            int64_t upper;
            if (__builtin_mul_overflow(num_columns, (num_columns-1), &upper)) {
                errno = EOVERFLOW;
                return MTX_ERR_ERRNO;
            }
            upper /= 2;
            int64_t lower;
            if (__builtin_mul_overflow((num_rows-num_columns), num_columns, &lower)) {
                errno = EOVERFLOW;
                return MTX_ERR_ERRNO;
            }
            if (__builtin_add_overflow(lower, upper, size)) {
                errno = EOVERFLOW;
                return MTX_ERR_ERRNO;
            }
        }

    } else if (triangle == mtx_strict_upper_triangular) {
        if (num_columns <= num_rows) {
            if (__builtin_mul_overflow(num_columns, (num_columns-1), size)) {
                errno = EOVERFLOW;
                return MTX_ERR_ERRNO;
            }
            *size /= 2;
        } else {
            int64_t left;
            if (__builtin_mul_overflow(num_rows, (num_rows-1), &left)) {
                errno = EOVERFLOW;
                return MTX_ERR_ERRNO;
            }
            left /= 2;
            int64_t right;
            if (__builtin_mul_overflow((num_columns-num_rows), num_rows, &right)) {
                errno = EOVERFLOW;
                return MTX_ERR_ERRNO;
            }
            if (__builtin_add_overflow(right, left, size)) {
                errno = EOVERFLOW;
                return MTX_ERR_ERRNO;
            }
        }

    } else {
        return MTX_ERR_INVALID_MTX_TRIANGLE;
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_matrix_array_data_alloc()' allocates data for a matrix in
 * array format.
 */
int mtx_matrix_array_data_alloc(
    struct mtx_matrix_array_data * mtxdata,
    enum mtx_field field,
    enum mtxprecision precision,
    enum mtx_symmetry symmetry,
    enum mtx_triangle triangle,
    int num_rows,
    int num_columns)
{
    int err;
    if ((symmetry == mtx_symmetric ||
         symmetry == mtx_hermitian) &&
        triangle != mtx_lower_triangular &&
        triangle != mtx_upper_triangular)
        return MTX_ERR_INVALID_MTX_TRIANGLE;
    if (symmetry == mtx_skew_symmetric &&
        triangle != mtx_strict_lower_triangular &&
        triangle != mtx_strict_upper_triangular)
        return MTX_ERR_INVALID_MTX_TRIANGLE;

    int64_t size;
    err = mtx_matrix_array_data_size(
        triangle, num_rows, num_columns, &size);
    if (err)
        return err;

    if (field == mtx_real) {
        if (precision == mtx_single) {
            mtxdata->data.real_single = malloc(size * sizeof(float));
            if (!mtxdata->data.real_single)
                return MTX_ERR_ERRNO;
        } else if (precision == mtx_double) {
            mtxdata->data.real_double = malloc(size * sizeof(double));
            if (!mtxdata->data.real_double)
                return MTX_ERR_ERRNO;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (field == mtx_complex) {
        if (precision == mtx_single) {
            mtxdata->data.complex_single = malloc(size * sizeof(float[2]));
            if (!mtxdata->data.complex_single)
                return MTX_ERR_ERRNO;
        } else if (precision == mtx_double) {
            mtxdata->data.complex_double = malloc(size * sizeof(double[2]));
            if (!mtxdata->data.complex_double)
                return MTX_ERR_ERRNO;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (field == mtx_integer) {
        if (precision == mtx_single) {
            mtxdata->data.integer_single = malloc(size * sizeof(int32_t));
            if (!mtxdata->data.integer_single)
                return MTX_ERR_ERRNO;
        } else if (precision == mtx_double) {
            mtxdata->data.integer_double = malloc(size * sizeof(int64_t));
            if (!mtxdata->data.integer_double)
                return MTX_ERR_ERRNO;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else {
        return MTX_ERR_INVALID_MTX_FIELD;
    }

    mtxdata->field = field;
    mtxdata->symmetry = symmetry;
    mtxdata->precision = precision;
    mtxdata->triangle = triangle;
    mtxdata->sorting = mtx_row_major;
    mtxdata->num_rows = num_rows;
    mtxdata->num_columns = num_columns;
    mtxdata->size = size;
    return MTX_SUCCESS;
}

/*
 * Array matrix allocation and initialisation.
 */

/**
 * `mtx_matrix_array_data_init_real_single()' creates data for a
 * matrix with real, single-precision floating point coefficients.
 */
int mtx_matrix_array_data_init_real_single(
    struct mtx_matrix_array_data * mtxdata,
    enum mtx_symmetry symmetry,
    enum mtx_triangle triangle,
    int num_rows,
    int num_columns,
    int64_t size,
    const float * data)
{
    int err = mtx_matrix_array_data_alloc(
        mtxdata, mtx_real, mtx_single,
        symmetry, triangle,
        num_rows, num_columns);
    if (err)
        return err;
    if (size < mtxdata->size) {
        mtx_matrix_array_data_free(mtxdata);
        return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    }
    for (int64_t i = 0; i < mtxdata->size; i++)
        mtxdata->data.real_single[i] = data[i];
    return MTX_SUCCESS;
}

/**
 * `mtx_matrix_array_data_init_real_double()' creates data for a
 * matrix with real, double-precision floating point coefficients.
 */
int mtx_matrix_array_data_init_real_double(
    struct mtx_matrix_array_data * mtxdata,
    enum mtx_symmetry symmetry,
    enum mtx_triangle triangle,
    int num_rows,
    int num_columns,
    int64_t size,
    const double * data)
{
    int err = mtx_matrix_array_data_alloc(
        mtxdata, mtx_real, mtx_double,
        symmetry, triangle,
        num_rows, num_columns);
    if (err)
        return err;
    if (size < mtxdata->size) {
        mtx_matrix_array_data_free(mtxdata);
        return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    }
    for (int64_t i = 0; i < mtxdata->size; i++)
        mtxdata->data.real_double[i] = data[i];
    return MTX_SUCCESS;
}

/**
 * `mtx_matrix_array_data_init_complex_single()' creates data for a
 * matrix with complex, single-precision floating point coefficients.
 */
int mtx_matrix_array_data_init_complex_single(
    struct mtx_matrix_array_data * mtxdata,
    enum mtx_symmetry symmetry,
    enum mtx_triangle triangle,
    int num_rows,
    int num_columns,
    int64_t size,
    const float (* data)[2])
{
    int err = mtx_matrix_array_data_alloc(
        mtxdata, mtx_complex, mtx_single,
        symmetry, triangle,
        num_rows, num_columns);
    if (err)
        return err;
    if (size < mtxdata->size) {
        mtx_matrix_array_data_free(mtxdata);
        return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    }
    for (int64_t i = 0; i < mtxdata->size; i++) {
        mtxdata->data.complex_single[i][0] = data[i][0];
        mtxdata->data.complex_single[i][1] = data[i][1];
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_matrix_array_data_init_complex_double()' creates data for a
 * matrix with complex, double-precision floating point coefficients.
 */
int mtx_matrix_array_data_init_complex_double(
    struct mtx_matrix_array_data * mtxdata,
    enum mtx_symmetry symmetry,
    enum mtx_triangle triangle,
    int num_rows,
    int num_columns,
    int64_t size,
    const double (* data)[2])
{
    int err = mtx_matrix_array_data_alloc(
        mtxdata, mtx_complex, mtx_double,
        symmetry, triangle,
        num_rows, num_columns);
    if (err)
        return err;
    if (size < mtxdata->size) {
        mtx_matrix_array_data_free(mtxdata);
        return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    }
    for (int64_t i = 0; i < mtxdata->size; i++) {
        mtxdata->data.complex_double[i][0] = data[i][0];
        mtxdata->data.complex_double[i][1] = data[i][1];
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_matrix_array_data_init_integer_single()' creates data for a
 * matrix with integer, single-precision coefficients.
 */
int mtx_matrix_array_data_init_integer_single(
    struct mtx_matrix_array_data * mtxdata,
    enum mtx_symmetry symmetry,
    enum mtx_triangle triangle,
    int num_rows,
    int num_columns,
    int64_t size,
    const int32_t * data)
{
    int err = mtx_matrix_array_data_alloc(
        mtxdata, mtx_integer, mtx_single,
        symmetry, triangle,
        num_rows, num_columns);
    if (err)
        return err;
    if (size < mtxdata->size) {
        mtx_matrix_array_data_free(mtxdata);
        return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    }
    for (int64_t i = 0; i < mtxdata->size; i++)
        mtxdata->data.integer_single[i] = data[i];
    return MTX_SUCCESS;
}

/**
 * `mtx_matrix_array_data_init_integer_double()' creates data for a
 * matrix with integer, double-precision coefficients.
 */
int mtx_matrix_array_data_init_integer_double(
    struct mtx_matrix_array_data * mtxdata,
    enum mtx_symmetry symmetry,
    enum mtx_triangle triangle,
    int num_rows,
    int num_columns,
    int64_t size,
    const int64_t * data)
{
    int err = mtx_matrix_array_data_alloc(
        mtxdata, mtx_integer, mtx_double,
        symmetry, triangle,
        num_rows, num_columns);
    if (err)
        return err;
    if (size < mtxdata->size) {
        mtx_matrix_array_data_free(mtxdata);
        return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    }
    for (int64_t i = 0; i < mtxdata->size; i++)
        mtxdata->data.integer_double[i] = data[i];
    return MTX_SUCCESS;
}

/**
 * `mtx_matrix_array_data_copy_alloc()' allocates a copy of a matrix
 * without copying the matrix values.
 */
int mtx_matrix_array_data_copy_alloc(
    struct mtx_matrix_array_data * dst,
    const struct mtx_matrix_array_data * src)
{
    return mtx_matrix_array_data_alloc(
        dst, src->field, src->precision,
        src->symmetry, src->triangle,
        src->num_rows, src->num_columns);
}

/**
 * `mtx_matrix_array_data_copy_init()' creates a copy of a matrix and
 * also copies matrix values.
 */
int mtx_matrix_array_data_copy_init(
    struct mtx_matrix_array_data * dst,
    const struct mtx_matrix_array_data * src)
{
    if (src->field == mtx_real) {
        if (src->precision == mtx_single) {
            return mtx_matrix_array_data_init_real_single(
                dst, src->symmetry, src->triangle,
                src->num_rows, src->num_columns,
                src->size, src->data.real_single);
        } else if (src->precision == mtx_double) {
            return mtx_matrix_array_data_init_real_double(
                dst, src->symmetry, src->triangle,
                src->num_rows, src->num_columns,
                src->size, src->data.real_double);
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (src->field == mtx_complex) {
        if (src->precision == mtx_single) {
            return mtx_matrix_array_data_init_complex_single(
                dst, src->symmetry, src->triangle,
                src->num_rows, src->num_columns,
                src->size, src->data.complex_single);
        } else if (src->precision == mtx_double) {
            return mtx_matrix_array_data_init_complex_double(
                dst, src->symmetry, src->triangle,
                src->num_rows, src->num_columns,
                src->size, src->data.complex_double);
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (src->field == mtx_integer) {
        if (src->precision == mtx_single) {
            return mtx_matrix_array_data_init_integer_single(
                dst, src->symmetry, src->triangle,
                src->num_rows, src->num_columns,
                src->size, src->data.integer_single);
        } else if (src->precision == mtx_double) {
            return mtx_matrix_array_data_init_integer_double(
                dst, src->symmetry, src->triangle,
                src->num_rows, src->num_columns,
                src->size, src->data.integer_double);
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else {
        return MTX_ERR_INVALID_MTX_FIELD;
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_matrix_array_data_set_zero()' zeroes a matrix or matrix.
 */
int mtx_matrix_array_data_set_zero(
    struct mtx_matrix_array_data * mtxdata)
{
    if (mtxdata->field == mtx_real) {
        if (mtxdata->precision == mtx_single) {
            for (int64_t k = 0; k < mtxdata->size; k++)
                mtxdata->data.real_single[k] = 0;
        } else if (mtxdata->precision == mtx_double) {
            for (int64_t k = 0; k < mtxdata->size; k++)
                mtxdata->data.real_double[k] = 0;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (mtxdata->field == mtx_complex) {
        if (mtxdata->precision == mtx_single) {
            for (int64_t k = 0; k < mtxdata->size; k++) {
                mtxdata->data.complex_single[k][0] = 0;
                mtxdata->data.complex_single[k][1] = 0;
            }
        } else if (mtxdata->precision == mtx_double) {
            for (int64_t k = 0; k < mtxdata->size; k++) {
                mtxdata->data.complex_double[k][0] = 0;
                mtxdata->data.complex_double[k][1] = 0;
            }
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (mtxdata->field == mtx_integer) {
        if (mtxdata->precision == mtx_single) {
            for (int64_t k = 0; k < mtxdata->size; k++)
                mtxdata->data.integer_single[k] = 0;
        } else if (mtxdata->precision == mtx_double) {
            for (int64_t k = 0; k < mtxdata->size; k++)
                mtxdata->data.integer_double[k] = 0;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else {
        return MTX_ERR_INVALID_MTX_FIELD;
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_matrix_array_data_set_constant_real_single()' sets every value
 * of a matrix equal to a constant, single precision floating point
 * number.
 */
int mtx_matrix_array_data_set_constant_real_single(
    struct mtx_matrix_array_data * mtxdata,
    float a)
{
    if (mtxdata->field != mtx_real)
        return MTX_ERR_INVALID_MTX_FIELD;
    if (mtxdata->precision == mtx_single) {
        for (int64_t k = 0; k < mtxdata->size; k++)
            mtxdata->data.real_single[k] = a;
    } else if (mtxdata->precision == mtx_double) {
        for (int64_t k = 0; k < mtxdata->size; k++)
            mtxdata->data.real_double[k] = a;
    } else {
        return MTX_ERR_INVALID_PRECISION;
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_matrix_array_data_set_constant_real_double()' sets every value
 * of a matrix equal to a constant, double precision floating point
 * number.
 */
int mtx_matrix_array_data_set_constant_real_double(
    struct mtx_matrix_array_data * mtxdata,
    double a)
{
    if (mtxdata->field != mtx_real)
        return MTX_ERR_INVALID_MTX_FIELD;
    if (mtxdata->precision == mtx_single) {
        for (int64_t k = 0; k < mtxdata->size; k++)
            mtxdata->data.real_single[k] = a;
    } else if (mtxdata->precision == mtx_double) {
        for (int64_t k = 0; k < mtxdata->size; k++)
            mtxdata->data.real_double[k] = a;
    } else {
        return MTX_ERR_INVALID_PRECISION;
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_matrix_array_data_set_constant_complex_single()' sets every
 * value of a matrix equal to a constant, single precision floating
 * point complex number.
 */
int mtx_matrix_array_data_set_constant_complex_single(
    struct mtx_matrix_array_data * mtxdata,
    float a[2])
{
    if (mtxdata->field != mtx_complex)
        return MTX_ERR_INVALID_MTX_FIELD;
    if (mtxdata->precision == mtx_single) {
        for (int64_t k = 0; k < mtxdata->size; k++) {
            mtxdata->data.complex_single[k][0] = a[0];
            mtxdata->data.complex_single[k][1] = a[1];
        }
    } else if (mtxdata->precision == mtx_double) {
        for (int64_t k = 0; k < mtxdata->size; k++) {
            mtxdata->data.complex_double[k][0] = a[0];
            mtxdata->data.complex_double[k][1] = a[1];
        }
    } else {
        return MTX_ERR_INVALID_PRECISION;
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_matrix_array_data_set_constant_complex_double()' sets every
 * value of a matrix equal to a constant, double precision floating
 * point complex number.
 */
int mtx_matrix_array_data_set_constant_complex_double(
    struct mtx_matrix_array_data * mtxdata,
    double a[2])
{
    if (mtxdata->field != mtx_complex)
        return MTX_ERR_INVALID_MTX_FIELD;
    if (mtxdata->precision == mtx_single) {
        for (int64_t k = 0; k < mtxdata->size; k++) {
            mtxdata->data.complex_single[k][0] = a[0];
            mtxdata->data.complex_single[k][1] = a[1];
        }
    } else if (mtxdata->precision == mtx_double) {
        for (int64_t k = 0; k < mtxdata->size; k++) {
            mtxdata->data.complex_double[k][0] = a[0];
            mtxdata->data.complex_double[k][1] = a[1];
        }
    } else {
        return MTX_ERR_INVALID_PRECISION;
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_matrix_array_data_set_constant_integer_single()' sets every
 * value of a matrix equal to a constant integer.
 */
int mtx_matrix_array_data_set_constant_integer_single(
    struct mtx_matrix_array_data * mtxdata,
    int32_t a)
{
    if (mtxdata->field != mtx_integer)
        return MTX_ERR_INVALID_MTX_FIELD;
    if (mtxdata->precision == mtx_single) {
        for (int64_t k = 0; k < mtxdata->size; k++)
            mtxdata->data.integer_single[k] = a;
    } else if (mtxdata->precision == mtx_double) {
        for (int64_t k = 0; k < mtxdata->size; k++)
            mtxdata->data.integer_double[k] = a;
    } else {
        return MTX_ERR_INVALID_PRECISION;
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_matrix_array_data_set_constant_integer_double()' sets every
 * value of a matrix equal to a constant integer.
 */
int mtx_matrix_array_data_set_constant_integer_double(
    struct mtx_matrix_array_data * mtxdata,
    int64_t a)
{
    if (mtxdata->field != mtx_integer)
        return MTX_ERR_INVALID_MTX_FIELD;
    if (mtxdata->precision == mtx_single) {
        for (int64_t k = 0; k < mtxdata->size; k++)
            mtxdata->data.integer_single[k] = a;
    } else if (mtxdata->precision == mtx_double) {
        for (int64_t k = 0; k < mtxdata->size; k++)
            mtxdata->data.integer_double[k] = a;
    } else {
        return MTX_ERR_INVALID_PRECISION;
    }
    return MTX_SUCCESS;
}
