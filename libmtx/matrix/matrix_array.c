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
 * Last modified: 2021-10-05
 *
 * Data structures for matrices in array format.
 */

#include <libmtx/libmtx-config.h>

#include <libmtx/error.h>
#include <libmtx/mtx/precision.h>
#include <libmtx/mtxfile/mtxfile.h>
#include <libmtx/util/field.h>
#include <libmtx/matrix/matrix_array.h>

#ifdef LIBMTX_HAVE_BLAS
#include <cblas.h>
#endif

#include <errno.h>

#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Memory management
 */

/**
 * `mtxmatrix_array_free()' frees storage allocated for a matrix.
 */
void mtxmatrix_array_free(
    struct mtxmatrix_array * matrix)
{
    if (matrix->field == mtx_field_real) {
        if (matrix->precision == mtx_single) {
            free(matrix->data.real_single);
        } else if (matrix->precision == mtx_double) {
            free(matrix->data.real_double);
        }
    } else if (matrix->field == mtx_field_complex) {
        if (matrix->precision == mtx_single) {
            free(matrix->data.complex_single);
        } else if (matrix->precision == mtx_double) {
            free(matrix->data.complex_double);
        }
    } else if (matrix->field == mtx_field_integer) {
        if (matrix->precision == mtx_single) {
            free(matrix->data.integer_single);
        } else if (matrix->precision == mtx_double) {
            free(matrix->data.integer_double);
        }
    }
}

/**
 * `mtxmatrix_array_alloc_copy()' allocates a copy of a matrix without
 * initialising the values.
 */
int mtxmatrix_array_alloc_copy(
    struct mtxmatrix_array * dst,
    const struct mtxmatrix_array * src);

/**
 * `mtxmatrix_array_init_copy()' allocates a copy of a matrix and also
 * copies the values.
 */
int mtxmatrix_array_init_copy(
    struct mtxmatrix_array * dst,
    const struct mtxmatrix_array * src);

/*
 * Matrix array formats
 */

/**
 * `mtxmatrix_array_alloc()' allocates a matrix in array format.
 */
int mtxmatrix_array_alloc(
    struct mtxmatrix_array * matrix,
    enum mtx_field_ field,
    enum mtx_precision precision,
    int num_rows,
    int num_columns)
{
    int64_t size;
    if (__builtin_mul_overflow(num_rows, num_columns, &size)) {
        errno = EOVERFLOW;
        return MTX_ERR_ERRNO;
    }

    if (field == mtx_field_real) {
        if (precision == mtx_single) {
            matrix->data.real_single =
                malloc(size * sizeof(*matrix->data.real_single));
            if (!matrix->data.real_single)
                return MTX_ERR_ERRNO;
        } else if (precision == mtx_double) {
            matrix->data.real_double =
                malloc(size * sizeof(*matrix->data.real_double));
            if (!matrix->data.real_double)
                return MTX_ERR_ERRNO;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (field == mtx_field_complex) {
        if (precision == mtx_single) {
            matrix->data.complex_single =
                malloc(size * sizeof(*matrix->data.complex_single));
            if (!matrix->data.complex_single)
                return MTX_ERR_ERRNO;
        } else if (precision == mtx_double) {
            matrix->data.complex_double =
                malloc(size * sizeof(*matrix->data.complex_double));
            if (!matrix->data.complex_double)
                return MTX_ERR_ERRNO;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (field == mtx_field_integer) {
        if (precision == mtx_single) {
            matrix->data.integer_single =
                malloc(size * sizeof(*matrix->data.integer_single));
            if (!matrix->data.integer_single)
                return MTX_ERR_ERRNO;
        } else if (precision == mtx_double) {
            matrix->data.integer_double =
                malloc(size * sizeof(*matrix->data.integer_double));
            if (!matrix->data.integer_double)
                return MTX_ERR_ERRNO;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else {
        return MTX_ERR_INVALID_FIELD;
    }
    matrix->field = field;
    matrix->precision = precision;
    matrix->num_rows = num_rows;
    matrix->num_columns = num_columns;
    matrix->size = size;
    return MTX_SUCCESS;
}

/**
 * `mtxmatrix_array_init_real_single()' allocates and initialises a
 * matrix in array format with real, single precision coefficients.
 */
int mtxmatrix_array_init_real_single(
    struct mtxmatrix_array * matrix,
    int num_rows,
    int num_columns,
    const float * data)
{
    int err = mtxmatrix_array_alloc(
        matrix, mtx_field_real, mtx_single, num_rows, num_columns);
    if (err)
        return err;
    for (int64_t k = 0; k < matrix->size; k++)
        matrix->data.real_single[k] = data[k];
    return MTX_SUCCESS;
}

/**
 * `mtxmatrix_array_init_real_double()' allocates and initialises a
 * matrix in array format with real, double precision coefficients.
 */
int mtxmatrix_array_init_real_double(
    struct mtxmatrix_array * matrix,
    int num_rows,
    int num_columns,
    const double * data)
{
    int err = mtxmatrix_array_alloc(
        matrix, mtx_field_real, mtx_double, num_rows, num_columns);
    if (err)
        return err;
    for (int64_t k = 0; k < matrix->size; k++)
        matrix->data.real_double[k] = data[k];
    return MTX_SUCCESS;
}

/**
 * `mtxmatrix_array_init_complex_single()' allocates and initialises a
 * matrix in array format with complex, single precision coefficients.
 */
int mtxmatrix_array_init_complex_single(
    struct mtxmatrix_array * matrix,
    int num_rows,
    int num_columns,
    const float (* data)[2])
{
    int err = mtxmatrix_array_alloc(
        matrix, mtx_field_complex, mtx_single, num_rows, num_columns);
    if (err)
        return err;
    for (int64_t k = 0; k < matrix->size; k++) {
        matrix->data.complex_single[k][0] = data[k][0];
        matrix->data.complex_single[k][1] = data[k][1];
    }
    return MTX_SUCCESS;
}

/**
 * `mtxmatrix_array_init_complex_double()' allocates and initialises a
 * matrix in array format with complex, double precision coefficients.
 */
int mtxmatrix_array_init_complex_double(
    struct mtxmatrix_array * matrix,
    int num_rows,
    int num_columns,
    const double (* data)[2])
{
    int err = mtxmatrix_array_alloc(
        matrix, mtx_field_complex, mtx_double, num_rows, num_columns);
    if (err)
        return err;
    for (int64_t k = 0; k < matrix->size; k++) {
        matrix->data.complex_double[k][0] = data[k][0];
        matrix->data.complex_double[k][1] = data[k][1];
    }
    return MTX_SUCCESS;
}

/**
 * `mtxmatrix_array_init_integer_single()' allocates and initialises a
 * matrix in array format with integer, single precision coefficients.
 */
int mtxmatrix_array_init_integer_single(
    struct mtxmatrix_array * matrix,
    int num_rows,
    int num_columns,
    const int32_t * data)
{
    int err = mtxmatrix_array_alloc(
        matrix, mtx_field_integer, mtx_single, num_rows, num_columns);
    if (err)
        return err;
    for (int64_t k = 0; k < matrix->size; k++)
        matrix->data.integer_single[k] = data[k];
    return MTX_SUCCESS;
}

/**
 * `mtxmatrix_array_init_integer_double()' allocates and initialises a
 * matrix in array format with integer, double precision coefficients.
 */
int mtxmatrix_array_init_integer_double(
    struct mtxmatrix_array * matrix,
    int num_rows,
    int num_columns,
    const int64_t * data)
{
    int err = mtxmatrix_array_alloc(
        matrix, mtx_field_integer, mtx_double, num_rows, num_columns);
    if (err)
        return err;
    for (int64_t k = 0; k < matrix->size; k++)
        matrix->data.integer_double[k] = data[k];
    return MTX_SUCCESS;
}

/*
 * Convert to and from Matrix Market format
 */

/**
 * `mtxmatrix_array_from_mtxfile()' converts a matrix in Matrix Market
 * format to a matrix.
 */
int mtxmatrix_array_from_mtxfile(
    struct mtxmatrix_array * matrix,
    const struct mtxfile * mtxfile)
{
    if (mtxfile->header.object != mtxfile_matrix)
        return MTX_ERR_INCOMPATIBLE_MTX_OBJECT;

    /* TODO: If needed, we could convert from coordinate to array. */
    if (mtxfile->header.format != mtxfile_array)
        return MTX_ERR_INCOMPATIBLE_MTX_FORMAT;

    if (mtxfile->header.field == mtxfile_real) {
        if (mtxfile->precision == mtx_single) {
            return mtxmatrix_array_init_real_single(
                matrix, mtxfile->size.num_rows, mtxfile->size.num_columns,
                mtxfile->data.array_real_single);
        } else if (mtxfile->precision == mtx_double) {
            return mtxmatrix_array_init_real_double(
                matrix, mtxfile->size.num_rows, mtxfile->size.num_columns,
                mtxfile->data.array_real_double);
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (mtxfile->header.field == mtxfile_complex) {
        if (mtxfile->precision == mtx_single) {
            return mtxmatrix_array_init_complex_single(
                matrix, mtxfile->size.num_rows, mtxfile->size.num_columns,
                mtxfile->data.array_complex_single);
        } else if (mtxfile->precision == mtx_double) {
            return mtxmatrix_array_init_complex_double(
                matrix, mtxfile->size.num_rows, mtxfile->size.num_columns,
                mtxfile->data.array_complex_double);
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (mtxfile->header.field == mtxfile_integer) {
        if (mtxfile->precision == mtx_single) {
            return mtxmatrix_array_init_integer_single(
                matrix, mtxfile->size.num_rows, mtxfile->size.num_columns,
                mtxfile->data.array_integer_single);
        } else if (mtxfile->precision == mtx_double) {
            return mtxmatrix_array_init_integer_double(
                matrix, mtxfile->size.num_rows, mtxfile->size.num_columns,
                mtxfile->data.array_integer_double);
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (mtxfile->header.field == mtxfile_pattern) {
        return MTX_ERR_INCOMPATIBLE_MTX_FIELD;
    } else {
        return MTX_ERR_INVALID_MTX_FIELD;
    }
    return MTX_SUCCESS;
}

/**
 * `mtxmatrix_array_to_mtxfile()' converts a matrix to a matrix in
 * Matrix Market format.
 */
int mtxmatrix_array_to_mtxfile(
    const struct mtxmatrix_array * matrix,
    struct mtxfile * mtxfile)
{
    if (matrix->field == mtx_field_real) {
        if (matrix->precision == mtx_single) {
            return mtxfile_init_matrix_array_real_single(
                mtxfile, mtxfile_general, matrix->num_rows, matrix->num_columns,
                matrix->data.real_single);
        } else if (matrix->precision == mtx_double) {
            return mtxfile_init_matrix_array_real_double(
                mtxfile, mtxfile_general, matrix->num_rows, matrix->num_columns,
                matrix->data.real_double);
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (matrix->field == mtx_field_complex) {
        if (matrix->precision == mtx_single) {
            return mtxfile_init_matrix_array_complex_single(
                mtxfile, mtxfile_general, matrix->num_rows, matrix->num_columns,
                matrix->data.complex_single);
        } else if (matrix->precision == mtx_double) {
            return mtxfile_init_matrix_array_complex_double(
                mtxfile, mtxfile_general, matrix->num_rows, matrix->num_columns,
                matrix->data.complex_double);
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (matrix->field == mtx_field_integer) {
        if (matrix->precision == mtx_single) {
            return mtxfile_init_matrix_array_integer_single(
                mtxfile, mtxfile_general, matrix->num_rows, matrix->num_columns,
                matrix->data.integer_single);
        } else if (matrix->precision == mtx_double) {
            return mtxfile_init_matrix_array_integer_double(
                mtxfile, mtxfile_general, matrix->num_rows, matrix->num_columns,
                matrix->data.integer_double);
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (matrix->field == mtx_field_pattern) {
        return MTX_ERR_INCOMPATIBLE_FIELD;
    } else {
        return MTX_ERR_INVALID_FIELD;
    }
}
