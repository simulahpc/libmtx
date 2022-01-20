/* This file is part of Libmtx.
 *
 * Copyright (C) 2021 James D. Trotter
 *
 * Libmtx is free software: you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * Libmtx is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Libmtx.  If not, see
 * <https://www.gnu.org/licenses/>.
 *
 * Authors: James D. Trotter <james@simula.no>
 * Last modified: 2021-08-16
 *
 * Data structures for matrices in coordinate format.
 */

#include <libmtx/matrix/coordinate/data.h>

#include <libmtx/error.h>
#include <libmtx/mtx/assembly.h>
#include <libmtx/mtx/header.h>
#include <libmtx/precision.h>
#include <libmtx/mtx/sort.h>
#include <libmtx/util/cuthill_mckee.h>
#include <libmtx/util/index_set.h>

#include <errno.h>

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

/**
 * `mtx_matrix_coordinate_data_free()' frees resources associated with
 * the matrix data in coordinate format.
 */
void mtx_matrix_coordinate_data_free(
    struct mtx_matrix_coordinate_data * mtxdata)
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
    } else if (mtxdata->field == mtx_pattern) {
        free(mtxdata->data.pattern);
    }
}

/**
 * `mtx_matrix_coordinate_data_alloc()' allocates data for a matrix in
 * coordinate format.
 */
int mtx_matrix_coordinate_data_alloc(
    struct mtx_matrix_coordinate_data * mtxdata,
    enum mtx_field field,
    enum mtxprecision precision,
    int num_rows,
    int num_columns,
    int64_t size)
{
    if (field == mtx_real) {
        if (precision == mtx_single) {
            mtxdata->data.real_single = malloc(
                size * sizeof(struct mtx_matrix_coordinate_real_single));
            if (!mtxdata->data.real_single)
                return MTX_ERR_ERRNO;
        } else if (precision == mtx_double) {
            mtxdata->data.real_double = malloc(
                size * sizeof(struct mtx_matrix_coordinate_real_double));
            if (!mtxdata->data.real_double)
                return MTX_ERR_ERRNO;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (field == mtx_complex) {
        if (precision == mtx_single) {
            mtxdata->data.complex_single = malloc(
                size * sizeof(struct mtx_matrix_coordinate_complex_single));
            if (!mtxdata->data.complex_single)
                return MTX_ERR_ERRNO;
        } else if (precision == mtx_double) {
            mtxdata->data.complex_double = malloc(
                size * sizeof(struct mtx_matrix_coordinate_complex_double));
            if (!mtxdata->data.complex_double)
                return MTX_ERR_ERRNO;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (field == mtx_integer) {
        if (precision == mtx_single) {
            mtxdata->data.integer_single = malloc(
                size * sizeof(struct mtx_matrix_coordinate_integer_single));
            if (!mtxdata->data.integer_single)
                return MTX_ERR_ERRNO;
        } else if (precision == mtx_double) {
            mtxdata->data.integer_double = malloc(
                size * sizeof(struct mtx_matrix_coordinate_integer_double));
            if (!mtxdata->data.integer_double)
                return MTX_ERR_ERRNO;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (field == mtx_pattern) {
        mtxdata->data.pattern = malloc(
            size * sizeof(struct mtx_matrix_coordinate_pattern));
        if (!mtxdata->data.pattern)
            return MTX_ERR_ERRNO;
    } else {
        return MTX_ERR_INVALID_MTX_FIELD;
    }

    mtxdata->field = field;
    mtxdata->precision = precision;
    mtxdata->symmetry = mtx_general;
    mtxdata->triangle = mtx_nontriangular;
    mtxdata->sorting = mtx_unsorted;
    mtxdata->assembly = mtx_unassembled;
    mtxdata->num_rows = num_rows;
    mtxdata->num_columns = num_columns;
    mtxdata->size = size;
    return MTX_SUCCESS;
}

/*
 * Coordinate matrix allocation and initialisation.
 */

/**
 * `mtx_matrix_coordinate_data_init_real_single()' creates data for a
 * matrix with real, single-precision floating point coefficients.
 */
int mtx_matrix_coordinate_data_init_real_single(
    struct mtx_matrix_coordinate_data * mtxdata,
    enum mtx_symmetry symmetry,
    enum mtx_triangle triangle,
    enum mtx_sorting sorting,
    enum mtx_assembly assembly,
    int num_rows,
    int num_columns,
    int64_t size,
    const struct mtx_matrix_coordinate_real_single * data)
{
    int err = mtx_matrix_coordinate_data_alloc(
        mtxdata, mtx_real, mtx_single,
        num_rows, num_columns, size);
    if (err)
        return err;
    mtxdata->symmetry = symmetry;
    mtxdata->triangle = triangle;
    mtxdata->sorting = sorting;
    mtxdata->assembly = assembly;
    for (int64_t i = 0; i < size; i++)
        mtxdata->data.real_single[i] = data[i];
    return MTX_SUCCESS;
}

/**
 * `mtx_matrix_coordinate_data_init_real_double()' creates data for a
 * matrix with real, double-precision floating point coefficients.
 */
int mtx_matrix_coordinate_data_init_real_double(
    struct mtx_matrix_coordinate_data * mtxdata,
    enum mtx_symmetry symmetry,
    enum mtx_triangle triangle,
    enum mtx_sorting sorting,
    enum mtx_assembly assembly,
    int num_rows,
    int num_columns,
    int64_t size,
    const struct mtx_matrix_coordinate_real_double * data)
{
    int err = mtx_matrix_coordinate_data_alloc(
        mtxdata, mtx_real, mtx_double,
        num_rows, num_columns, size);
    if (err)
        return err;
    mtxdata->symmetry = symmetry;
    mtxdata->triangle = triangle;
    mtxdata->sorting = sorting;
    mtxdata->assembly = assembly;
    for (int64_t i = 0; i < size; i++)
        mtxdata->data.real_double[i] = data[i];
    return MTX_SUCCESS;
}

/**
 * `mtx_matrix_coordinate_data_init_complex_single()' creates data for
 * a matrix with complex, single-precision floating point
 * coefficients.
 */
int mtx_matrix_coordinate_data_init_complex_single(
    struct mtx_matrix_coordinate_data * mtxdata,
    enum mtx_symmetry symmetry,
    enum mtx_triangle triangle,
    enum mtx_sorting sorting,
    enum mtx_assembly assembly,
    int num_rows,
    int num_columns,
    int64_t size,
    const struct mtx_matrix_coordinate_complex_single * data)
{
    int err = mtx_matrix_coordinate_data_alloc(
        mtxdata, mtx_complex, mtx_single,
        num_rows, num_columns, size);
    if (err)
        return err;
    mtxdata->symmetry = symmetry;
    mtxdata->triangle = triangle;
    mtxdata->sorting = sorting;
    mtxdata->assembly = assembly;
    for (int64_t i = 0; i < size; i++)
        mtxdata->data.complex_single[i] = data[i];
    return MTX_SUCCESS;
}

/**
 * `mtx_matrix_coordinate_data_init_complex_double()' creates data for
 * a matrix with complex, double-precision floating point
 * coefficients.
 */
int mtx_matrix_coordinate_data_init_complex_double(
    struct mtx_matrix_coordinate_data * mtxdata,
    enum mtx_symmetry symmetry,
    enum mtx_triangle triangle,
    enum mtx_sorting sorting,
    enum mtx_assembly assembly,
    int num_rows,
    int num_columns,
    int64_t size,
    const struct mtx_matrix_coordinate_complex_double * data)
{
    int err = mtx_matrix_coordinate_data_alloc(
        mtxdata, mtx_complex, mtx_double,
        num_rows, num_columns, size);
    if (err)
        return err;
    mtxdata->symmetry = symmetry;
    mtxdata->triangle = triangle;
    mtxdata->sorting = sorting;
    mtxdata->assembly = assembly;
    for (int64_t i = 0; i < size; i++)
        mtxdata->data.complex_double[i] = data[i];
    return MTX_SUCCESS;
}

/**
 * `mtx_matrix_coordinate_data_init_integer_single()' creates data for
 * a matrix with integer, single-precision coefficients.
 */
int mtx_matrix_coordinate_data_init_integer_single(
    struct mtx_matrix_coordinate_data * mtxdata,
    enum mtx_symmetry symmetry,
    enum mtx_triangle triangle,
    enum mtx_sorting sorting,
    enum mtx_assembly assembly,
    int num_rows,
    int num_columns,
    int64_t size,
    const struct mtx_matrix_coordinate_integer_single * data)
{
    int err = mtx_matrix_coordinate_data_alloc(
        mtxdata, mtx_integer, mtx_single,
        num_rows, num_columns, size);
    if (err)
        return err;
    mtxdata->symmetry = symmetry;
    mtxdata->triangle = triangle;
    mtxdata->sorting = sorting;
    mtxdata->assembly = assembly;
    for (int64_t i = 0; i < size; i++)
        mtxdata->data.integer_single[i] = data[i];
    return MTX_SUCCESS;
}

/**
 * `mtx_matrix_coordinate_data_init_integer_double()' creates data for
 * a matrix with integer, double-precision coefficients.
 */
int mtx_matrix_coordinate_data_init_integer_double(
    struct mtx_matrix_coordinate_data * mtxdata,
    enum mtx_symmetry symmetry,
    enum mtx_triangle triangle,
    enum mtx_sorting sorting,
    enum mtx_assembly assembly,
    int num_rows,
    int num_columns,
    int64_t size,
    const struct mtx_matrix_coordinate_integer_double * data)
{
    int err = mtx_matrix_coordinate_data_alloc(
        mtxdata, mtx_integer, mtx_double,
        num_rows, num_columns, size);
    if (err)
        return err;
    mtxdata->symmetry = symmetry;
    mtxdata->triangle = triangle;
    mtxdata->sorting = sorting;
    mtxdata->assembly = assembly;
    for (int64_t i = 0; i < size; i++)
        mtxdata->data.integer_double[i] = data[i];
    return MTX_SUCCESS;
}

/**
 * `mtx_matrix_coordinate_data_init_pattern()' creates data for a
 * matrix with boolean (pattern) coefficients.
 */
int mtx_matrix_coordinate_data_init_pattern(
    struct mtx_matrix_coordinate_data * mtxdata,
    enum mtx_symmetry symmetry,
    enum mtx_triangle triangle,
    enum mtx_sorting sorting,
    enum mtx_assembly assembly,
    int num_rows,
    int num_columns,
    int64_t size,
    const struct mtx_matrix_coordinate_pattern * data)
{
    int err = mtx_matrix_coordinate_data_alloc(
        mtxdata, mtx_pattern, mtx_single,
        num_rows, num_columns, size);
    if (err)
        return err;
    mtxdata->symmetry = symmetry;
    mtxdata->triangle = triangle;
    mtxdata->sorting = sorting;
    mtxdata->assembly = assembly;
    for (int64_t i = 0; i < size; i++)
        mtxdata->data.pattern[i] = data[i];
    return MTX_SUCCESS;
}

/**
 * `mtx_matrix_coordinate_data_copy_alloc()' allocates a copy of a
 * matrix without copying the matrix values.
 */
int mtx_matrix_coordinate_data_copy_alloc(
    struct mtx_matrix_coordinate_data * dst,
    const struct mtx_matrix_coordinate_data * src)
{
    int err = mtx_matrix_coordinate_data_alloc(
        dst, src->field, src->precision,
        src->num_rows, src->num_columns, src->size);
    if (err)
        return err;
    dst->symmetry = src->symmetry;
    dst->triangle = src->triangle;
    dst->sorting = src->sorting;
    dst->assembly = src->assembly;
    return MTX_SUCCESS;
}

/**
 * `mtx_matrix_coordinate_data_copy_init()' creates a copy of a matrix
 * and also copies matrix values.
 */
int mtx_matrix_coordinate_data_copy_init(
    struct mtx_matrix_coordinate_data * dst,
    const struct mtx_matrix_coordinate_data * src)
{
    if (src->field == mtx_real) {
        if (src->precision == mtx_single) {
            return mtx_matrix_coordinate_data_init_real_single(
                dst, src->symmetry, src->triangle,
                src->sorting, src->assembly,
                src->num_rows, src->num_columns, src->size,
                src->data.real_single);
        } else if (src->precision == mtx_double) {
            return mtx_matrix_coordinate_data_init_real_double(
                dst, src->symmetry, src->triangle,
                src->sorting, src->assembly,
                src->num_rows, src->num_columns, src->size,
                src->data.real_double);
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (src->field == mtx_complex) {
        if (src->precision == mtx_single) {
            return mtx_matrix_coordinate_data_init_complex_single(
                dst, src->symmetry, src->triangle,
                src->sorting, src->assembly,
                src->num_rows, src->num_columns, src->size,
                src->data.complex_single);
        } else if (src->precision == mtx_double) {
            return mtx_matrix_coordinate_data_init_complex_double(
                dst, src->symmetry, src->triangle,
                src->sorting, src->assembly,
                src->num_rows, src->num_columns, src->size,
                src->data.complex_double);
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (src->field == mtx_integer) {
        if (src->precision == mtx_single) {
            return mtx_matrix_coordinate_data_init_integer_single(
                dst, src->symmetry, src->triangle,
                src->sorting, src->assembly,
                src->num_rows, src->num_columns, src->size,
                src->data.integer_single);
        } else if (src->precision == mtx_double) {
            return mtx_matrix_coordinate_data_init_integer_double(
                dst, src->symmetry, src->triangle,
                src->sorting, src->assembly,
                src->num_rows, src->num_columns, src->size,
                src->data.integer_double);
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (src->field == mtx_pattern) {
        return mtx_matrix_coordinate_data_init_pattern(
            dst, src->symmetry, src->triangle,
            src->sorting, src->assembly,
            src->num_rows, src->num_columns, src->size,
            src->data.pattern);
    } else {
        return MTX_ERR_INVALID_MTX_FIELD;
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_matrix_coordinate_data_set_zero()' zeroes a matrix.
 */
int mtx_matrix_coordinate_data_set_zero(
    struct mtx_matrix_coordinate_data * mtxdata)
{
    if (mtxdata->field == mtx_real) {
        if (mtxdata->precision == mtx_single) {
            for (int64_t k = 0; k < mtxdata->size; k++)
                mtxdata->data.real_single[k].a = 0;
        } else if (mtxdata->precision == mtx_double) {
            for (int64_t k = 0; k < mtxdata->size; k++)
                mtxdata->data.real_double[k].a = 0;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (mtxdata->field == mtx_complex) {
        if (mtxdata->precision == mtx_single) {
            for (int64_t k = 0; k < mtxdata->size; k++) {
                mtxdata->data.complex_single[k].a[0] = 0;
                mtxdata->data.complex_single[k].a[1] = 0;
            }
        } else if (mtxdata->precision == mtx_double) {
            for (int64_t k = 0; k < mtxdata->size; k++) {
                mtxdata->data.complex_double[k].a[0] = 0;
                mtxdata->data.complex_double[k].a[1] = 0;
            }
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (mtxdata->field == mtx_integer) {
        if (mtxdata->precision == mtx_single) {
            for (int64_t k = 0; k < mtxdata->size; k++)
                mtxdata->data.integer_single[k].a = 0;
        } else if (mtxdata->precision == mtx_double) {
            for (int64_t k = 0; k < mtxdata->size; k++)
                mtxdata->data.integer_double[k].a = 0;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else {
        return MTX_ERR_INVALID_MTX_FIELD;
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_matrix_coordinate_data_set_constant_real_single()' sets every
 * (nonzero) value of a matrix equal to a constant, single precision
 * floating point number.
 */
int mtx_matrix_coordinate_data_set_constant_real_single(
    struct mtx_matrix_coordinate_data * mtxdata,
    float a)
{
    if (mtxdata->field != mtx_real)
        return MTX_ERR_INVALID_MTX_FIELD;
    if (mtxdata->precision != mtx_single)
        return MTX_ERR_INVALID_PRECISION;
    for (int64_t k = 0; k < mtxdata->size; k++)
        mtxdata->data.real_single[k].a = 0;
    return MTX_SUCCESS;
}

/**
 * `mtx_matrix_coordinate_data_set_constant_real_double()' sets every
 * (nonzero) value of a matrix equal to a constant, double precision
 * floating point number.
 */
int mtx_matrix_coordinate_data_set_constant_real_double(
    struct mtx_matrix_coordinate_data * mtxdata,
    double a)
{
    if (mtxdata->field != mtx_real)
        return MTX_ERR_INVALID_MTX_FIELD;
    if (mtxdata->precision != mtx_double)
        return MTX_ERR_INVALID_PRECISION;
    for (int64_t k = 0; k < mtxdata->size; k++)
        mtxdata->data.real_double[k].a = a;
    return MTX_SUCCESS;
}

/**
 * `mtx_matrix_coordinate_data_set_constant_complex_single()' sets
 * every (nonzero) value of a matrix equal to a constant, single
 * precision floating point complex number.
 */
int mtx_matrix_coordinate_data_set_constant_complex_single(
    struct mtx_matrix_coordinate_data * mtxdata,
    float a[2])
{
    if (mtxdata->field != mtx_complex)
        return MTX_ERR_INVALID_MTX_FIELD;
    if (mtxdata->precision != mtx_single)
        return MTX_ERR_INVALID_PRECISION;
    for (int64_t k = 0; k < mtxdata->size; k++) {
        mtxdata->data.complex_single[k].a[0] = a[0];
        mtxdata->data.complex_single[k].a[1] = a[1];
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_matrix_coordinate_data_set_constant_complex_double()' sets
 * every (nonzero) value of a matrix equal to a constant, double
 * precision floating point complex number.
 */
int mtx_matrix_coordinate_data_set_constant_complex_double(
    struct mtx_matrix_coordinate_data * mtxdata,
    double a[2])
{
    if (mtxdata->field != mtx_complex)
        return MTX_ERR_INVALID_MTX_FIELD;
    if (mtxdata->precision != mtx_double)
        return MTX_ERR_INVALID_PRECISION;
    for (int64_t k = 0; k < mtxdata->size; k++) {
        mtxdata->data.complex_double[k].a[0] = a[0];
        mtxdata->data.complex_double[k].a[1] = a[1];
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_matrix_coordinate_data_set_constant_integer_single()' sets
 * every (nonzero) value of a matrix equal to a constant integer.
 */
int mtx_matrix_coordinate_data_set_constant_integer_single(
    struct mtx_matrix_coordinate_data * mtxdata,
    int32_t a)
{
    if (mtxdata->field != mtx_integer)
        return MTX_ERR_INVALID_MTX_FIELD;
    if (mtxdata->precision != mtx_single)
        return MTX_ERR_INVALID_PRECISION;
    for (int64_t k = 0; k < mtxdata->size; k++)
        mtxdata->data.integer_single[k].a = a;
    return MTX_SUCCESS;

}

/**
 * `mtx_matrix_coordinate_data_set_constant_integer_double()' sets
 * every (nonzero) value of a matrix equal to a constant integer.
 */
int mtx_matrix_coordinate_data_set_constant_integer_double(
    struct mtx_matrix_coordinate_data * mtxdata,
    int64_t a)
{
    if (mtxdata->field != mtx_integer)
        return MTX_ERR_INVALID_MTX_FIELD;
    if (mtxdata->precision != mtx_double)
        return MTX_ERR_INVALID_PRECISION;
    for (int64_t k = 0; k < mtxdata->size; k++)
        mtxdata->data.integer_double[k].a = a;
    return MTX_SUCCESS;
}

/*
 * Other functions.
 */

/**
 * `mtx_matrix_coordinate_data_size_per_row()' counts the number of
 * entries stored for each row of a matrix.
 *
 * The array `size_per_row' must point to an array containing enough
 * storage for `num_rows' values of type `int'.
 */
int mtx_matrix_coordinate_data_size_per_row(
    const struct mtx_matrix_coordinate_data * mtxdata,
    int num_rows,
    int * size_per_row)
{
    for (int i = 0; i < num_rows; i++)
        size_per_row[i] = 0;

    if (mtxdata->field == mtx_real) {
        if (mtxdata->precision == mtx_single) {
            const struct mtx_matrix_coordinate_real_single * data =
                mtxdata->data.real_single;
            for (int64_t k = 0; k < mtxdata->size; k++)
                size_per_row[data[k].i-1]++;
        } else if (mtxdata->precision == mtx_double) {
            const struct mtx_matrix_coordinate_real_double * data =
                mtxdata->data.real_double;
            for (int64_t k = 0; k < mtxdata->size; k++)
                size_per_row[data[k].i-1]++;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (mtxdata->field == mtx_complex) {
        if (mtxdata->precision == mtx_single) {
            const struct mtx_matrix_coordinate_complex_single * data =
                mtxdata->data.complex_single;
            for (int64_t k = 0; k < mtxdata->size; k++)
                size_per_row[data[k].i-1]++;
        } else if (mtxdata->precision == mtx_double) {
            const struct mtx_matrix_coordinate_complex_double * data =
                mtxdata->data.complex_double;
            for (int64_t k = 0; k < mtxdata->size; k++)
                size_per_row[data[k].i-1]++;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (mtxdata->field == mtx_integer) {
        if (mtxdata->precision == mtx_single) {
            const struct mtx_matrix_coordinate_integer_single * data =
                mtxdata->data.integer_single;
            for (int64_t k = 0; k < mtxdata->size; k++)
                size_per_row[data[k].i-1]++;
        } else if (mtxdata->precision == mtx_double) {
            const struct mtx_matrix_coordinate_integer_double * data =
                mtxdata->data.integer_double;
            for (int64_t k = 0; k < mtxdata->size; k++)
                size_per_row[data[k].i-1]++;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (mtxdata->field == mtx_pattern) {
        const struct mtx_matrix_coordinate_pattern * data =
            mtxdata->data.pattern;
        for (int64_t k = 0; k < mtxdata->size; k++)
            size_per_row[data[k].i-1]++;
    } else {
        return MTX_ERR_INVALID_MTX_FIELD;
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_matrix_coordinate_data_diagonals_per_row()' counts for each
 * row of a matrix the number of nonzero entries on the diagonal that
 * are stored.  If the matrix is not in an assembled state, then the
 * count will also count any duplicate diagonal entries.
 *
 * The array `diagonals_per_row' must point to an array containing
 * enough storage for `mtxdata->num_rows' values of type `int'.
 */
int mtx_matrix_coordinate_data_diagonals_per_row(
    const struct mtx_matrix_coordinate_data * mtxdata,
    int num_rows,
    int * diagonals_per_row)
{
    for (int i = 0; i < num_rows; i++)
        diagonals_per_row[i] = 0;

    if (mtxdata->field == mtx_real) {
        if (mtxdata->precision == mtx_single) {
            const struct mtx_matrix_coordinate_real_single * data =
                mtxdata->data.real_single;
            for (int64_t k = 0; k < mtxdata->size; k++) {
                if (data[k].i == data[k].j)
                    diagonals_per_row[data[k].i-1]++;
            }
        } else if (mtxdata->precision == mtx_double) {
            const struct mtx_matrix_coordinate_real_double * data =
                mtxdata->data.real_double;
            for (int64_t k = 0; k < mtxdata->size; k++) {
                if (data[k].i == data[k].j)
                    diagonals_per_row[data[k].i-1]++;
            }
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (mtxdata->field == mtx_complex) {
        if (mtxdata->precision == mtx_single) {
            const struct mtx_matrix_coordinate_complex_single * data =
                mtxdata->data.complex_single;
            for (int64_t k = 0; k < mtxdata->size; k++) {
                if (data[k].i == data[k].j)
                    diagonals_per_row[data[k].i-1]++;
            }
        } else if (mtxdata->precision == mtx_double) {
            const struct mtx_matrix_coordinate_complex_double * data =
                mtxdata->data.complex_double;
            for (int64_t k = 0; k < mtxdata->size; k++) {
                if (data[k].i == data[k].j)
                    diagonals_per_row[data[k].i-1]++;
            }
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (mtxdata->field == mtx_integer) {
        if (mtxdata->precision == mtx_single) {
            const struct mtx_matrix_coordinate_integer_single * data =
                mtxdata->data.integer_single;
            for (int64_t k = 0; k < mtxdata->size; k++) {
                if (data[k].i == data[k].j)
                    diagonals_per_row[data[k].i-1]++;
            }
        } else if (mtxdata->precision == mtx_double) {
            const struct mtx_matrix_coordinate_integer_double * data =
                mtxdata->data.integer_double;
            for (int64_t k = 0; k < mtxdata->size; k++) {
                if (data[k].i == data[k].j)
                    diagonals_per_row[data[k].i-1]++;
            }
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (mtxdata->field == mtx_pattern) {
        const struct mtx_matrix_coordinate_pattern * data =
            mtxdata->data.pattern;
        for (int64_t k = 0; k < mtxdata->size; k++) {
            if (data[k].i == data[k].j)
                diagonals_per_row[data[k].i-1]++;
        }
    } else {
        return MTX_ERR_INVALID_MTX_FIELD;
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_matrix_coordinate_data_column_ptr()' computes column pointers
 * of a matrix.
 *
 * The array `column_ptr' must point to an array containing enough
 * storage for `mtxdata->num_columns+1' values of type `int64_t'.
 *
 * The matrix is not required to be sorted in column major order.  If
 * the matrix is sorted in column major order, then the `i'-th entry
 * of `column_ptr' is the location of the first nonzero in the
 * `mtxdata->data' array that belongs to the `i+1'-th column of the
 * matrix, for `i=0,1,...,num_columns-1'.  The final entry of
 * `column_ptr' indicates the position one place beyond the last entry
 * in `mtxdata->data'.
 */
int mtx_matrix_coordinate_data_column_ptr(
    const struct mtx_matrix_coordinate_data * mtxdata,
    int size,
    int64_t * column_ptr)
{
    if (size <= mtxdata->num_columns)
        return MTX_ERR_INVALID_MTX_SIZE;

    /* Count the number of entries in each column. */
    for (int j = 0; j <= mtxdata->num_columns; j++)
        column_ptr[j] = 0;

    if (mtxdata->field == mtx_real) {
        if (mtxdata->precision == mtx_single) {
            const struct mtx_matrix_coordinate_real_single * data =
                mtxdata->data.real_single;
            for (int64_t k = 0; k < mtxdata->size; k++)
                column_ptr[data[k].j]++;
        } else if (mtxdata->precision == mtx_double) {
            const struct mtx_matrix_coordinate_real_double * data =
                mtxdata->data.real_double;
            for (int64_t k = 0; k < mtxdata->size; k++)
                column_ptr[data[k].j]++;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (mtxdata->field == mtx_complex) {
        if (mtxdata->precision == mtx_single) {
            const struct mtx_matrix_coordinate_complex_single * data =
                mtxdata->data.complex_single;
            for (int64_t k = 0; k < mtxdata->size; k++)
                column_ptr[data[k].j]++;
        } else if (mtxdata->precision == mtx_double) {
            const struct mtx_matrix_coordinate_complex_double * data =
                mtxdata->data.complex_double;
            for (int64_t k = 0; k < mtxdata->size; k++)
                column_ptr[data[k].j]++;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (mtxdata->field == mtx_integer) {
        if (mtxdata->precision == mtx_single) {
            const struct mtx_matrix_coordinate_integer_single * data =
                mtxdata->data.integer_single;
            for (int64_t k = 0; k < mtxdata->size; k++)
                column_ptr[data[k].j]++;
        } else if (mtxdata->precision == mtx_double) {
            const struct mtx_matrix_coordinate_integer_double * data =
                mtxdata->data.integer_double;
            for (int64_t k = 0; k < mtxdata->size; k++)
                column_ptr[data[k].j]++;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (mtxdata->field == mtx_pattern) {
        const struct mtx_matrix_coordinate_pattern * data =
            mtxdata->data.pattern;
        for (int64_t k = 0; k < mtxdata->size; k++)
            column_ptr[data[k].j]++;
    } else {
        return MTX_ERR_INVALID_MTX_FIELD;
    }

    /* Compute the prefix sum of the column lengths. */
    for (int j = 1; j <= mtxdata->num_columns; j++)
        column_ptr[j] += column_ptr[j-1];
    return MTX_SUCCESS;
}

/**
 * `mtx_matrix_coordinate_data_row_ptr()' computes row pointers of a
 * matrix.
 *
 * The array `row_ptr' must point to an array containing enough
 * storage for `mtxdata->num_rows+1' values of type `int64_t'.
 *
 * The matrix is not required to be sorted in row major order.  If the
 * matrix is sorted in row major order, then the `i'-th entry of
 * `row_ptr' is the location of the first nonzero in the
 * `mtxdata->data' array that belongs to the `i+1'-th row of the
 * matrix, for `i=0,1,...,num_rows-1'. The final entry of `row_ptr'
 * indicates the position one place beyond the last entry in
 * `mtxdata->data'.
 */
int mtx_matrix_coordinate_data_row_ptr(
    const struct mtx_matrix_coordinate_data * mtxdata,
    int size,
    int64_t * row_ptr)
{
    if (size <= mtxdata->num_rows)
        return MTX_ERR_INVALID_MTX_SIZE;

    /* Count the number of entries in each row. */
    for (int i = 0; i <= mtxdata->num_rows; i++)
        row_ptr[i] = 0;

    if (mtxdata->field == mtx_real) {
        if (mtxdata->precision == mtx_single) {
            const struct mtx_matrix_coordinate_real_single * data =
                mtxdata->data.real_single;
            for (int64_t k = 0; k < mtxdata->size; k++)
                row_ptr[data[k].i]++;
        } else if (mtxdata->precision == mtx_double) {
            const struct mtx_matrix_coordinate_real_double * data =
                mtxdata->data.real_double;
            for (int64_t k = 0; k < mtxdata->size; k++)
                row_ptr[data[k].i]++;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (mtxdata->field == mtx_complex) {
        if (mtxdata->precision == mtx_single) {
            const struct mtx_matrix_coordinate_complex_single * data =
                mtxdata->data.complex_single;
            for (int64_t k = 0; k < mtxdata->size; k++)
                row_ptr[data[k].i]++;
        } else if (mtxdata->precision == mtx_double) {
            const struct mtx_matrix_coordinate_complex_double * data =
                mtxdata->data.complex_double;
            for (int64_t k = 0; k < mtxdata->size; k++)
                row_ptr[data[k].i]++;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (mtxdata->field == mtx_integer) {
        if (mtxdata->precision == mtx_single) {
            const struct mtx_matrix_coordinate_integer_single * data =
                mtxdata->data.integer_single;
            for (int64_t k = 0; k < mtxdata->size; k++)
                row_ptr[data[k].i]++;
        } else if (mtxdata->precision == mtx_double) {
            const struct mtx_matrix_coordinate_integer_double * data =
                mtxdata->data.integer_double;
            for (int64_t k = 0; k < mtxdata->size; k++)
                row_ptr[data[k].i]++;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (mtxdata->field == mtx_pattern) {
        const struct mtx_matrix_coordinate_pattern * data =
            mtxdata->data.pattern;
        for (int64_t k = 0; k < mtxdata->size; k++)
            row_ptr[data[k].i]++;
    } else {
        return MTX_ERR_INVALID_MTX_FIELD;
    }

    /* Compute the prefix sum of the row lengths. */
    for (int i = 1; i <= mtxdata->num_rows; i++)
        row_ptr[i] += row_ptr[i-1];
    return MTX_SUCCESS;
}

/**
 * `mtx_matrix_coordinate_data_column_indices()' extracts the column
 * indices of a matrix to a separate array.
 *
 * The array `column_indices' must point to an array containing enough
 * storage for `size' values of type `int'.
 */
int mtx_matrix_coordinate_data_column_indices(
    const struct mtx_matrix_coordinate_data * mtxdata,
    int64_t size,
    int * column_indices)
{
    if (size < mtxdata->size)
        return MTX_ERR_INVALID_MTX_SIZE;

    if (mtxdata->field == mtx_real) {
        if (mtxdata->precision == mtx_single) {
            const struct mtx_matrix_coordinate_real_single * data =
                mtxdata->data.real_single;
            for (int64_t k = 0; k < mtxdata->size; k++)
                column_indices[k] = data[k].j;
        } else if (mtxdata->precision == mtx_double) {
            const struct mtx_matrix_coordinate_real_double * data =
                mtxdata->data.real_double;
            for (int64_t k = 0; k < mtxdata->size; k++)
                column_indices[k] = data[k].j;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (mtxdata->field == mtx_complex) {
        if (mtxdata->precision == mtx_single) {
            const struct mtx_matrix_coordinate_complex_single * data =
                mtxdata->data.complex_single;
            for (int64_t k = 0; k < mtxdata->size; k++)
                column_indices[k] = data[k].j;
        } else if (mtxdata->precision == mtx_double) {
            const struct mtx_matrix_coordinate_complex_double * data =
                mtxdata->data.complex_double;
            for (int64_t k = 0; k < mtxdata->size; k++)
                column_indices[k] = data[k].j;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (mtxdata->field == mtx_integer) {
        if (mtxdata->precision == mtx_single) {
            const struct mtx_matrix_coordinate_integer_single * data =
                mtxdata->data.integer_single;
            for (int64_t k = 0; k < mtxdata->size; k++)
                column_indices[k] = data[k].j;
        } else if (mtxdata->precision == mtx_double) {
            const struct mtx_matrix_coordinate_integer_double * data =
                mtxdata->data.integer_double;
            for (int64_t k = 0; k < mtxdata->size; k++)
                column_indices[k] = data[k].j;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (mtxdata->field == mtx_pattern) {
        const struct mtx_matrix_coordinate_pattern * data =
            mtxdata->data.pattern;
        for (int64_t k = 0; k < mtxdata->size; k++)
            column_indices[k] = data[k].j;
    } else {
        return MTX_ERR_INVALID_MTX_FIELD;
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_matrix_coordinate_data_row_indices()' extracts the row indices
 * of a matrix to a separate array.
 *
 * The array `row_indices' must point to an array containing enough
 * storage for `size' values of type `int'.
 */
int mtx_matrix_coordinate_data_row_indices(
    const struct mtx_matrix_coordinate_data * mtxdata,
    int64_t size,
    int * row_indices)
{
    if (size < mtxdata->size)
        return MTX_ERR_INVALID_MTX_SIZE;

    if (mtxdata->field == mtx_real) {
        if (mtxdata->precision == mtx_single) {
            const struct mtx_matrix_coordinate_real_single * data =
                mtxdata->data.real_single;
            for (int64_t k = 0; k < mtxdata->size; k++)
                row_indices[k] = data[k].i;
        } else if (mtxdata->precision == mtx_double) {
            const struct mtx_matrix_coordinate_real_double * data =
                mtxdata->data.real_double;
            for (int64_t k = 0; k < mtxdata->size; k++)
                row_indices[k] = data[k].i;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (mtxdata->field == mtx_complex) {
        if (mtxdata->precision == mtx_single) {
            const struct mtx_matrix_coordinate_complex_single * data =
                mtxdata->data.complex_single;
            for (int64_t k = 0; k < mtxdata->size; k++)
                row_indices[k] = data[k].i;
        } else if (mtxdata->precision == mtx_double) {
            const struct mtx_matrix_coordinate_complex_double * data =
                mtxdata->data.complex_double;
            for (int64_t k = 0; k < mtxdata->size; k++)
                row_indices[k] = data[k].i;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (mtxdata->field == mtx_integer) {
        if (mtxdata->precision == mtx_single) {
            const struct mtx_matrix_coordinate_integer_single * data =
                mtxdata->data.integer_single;
            for (int64_t k = 0; k < mtxdata->size; k++)
                row_indices[k] = data[k].i;
        } else if (mtxdata->precision == mtx_double) {
            const struct mtx_matrix_coordinate_integer_double * data =
                mtxdata->data.integer_double;
            for (int64_t k = 0; k < mtxdata->size; k++)
                row_indices[k] = data[k].i;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (mtxdata->field == mtx_pattern) {
        const struct mtx_matrix_coordinate_pattern * data =
            mtxdata->data.pattern;
        for (int64_t k = 0; k < mtxdata->size; k++)
            row_indices[k] = data[k].i;
    } else {
        return MTX_ERR_INVALID_MTX_FIELD;
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_matrix_coordinate_data_transpose()' transposes a coordinate
 * matrix.
 */
int mtx_matrix_coordinate_data_transpose(
    struct mtx_matrix_coordinate_data * mtxdata)
{
    if (mtxdata->field == mtx_real) {
        if (mtxdata->precision == mtx_single) {
            struct mtx_matrix_coordinate_real_single * data =
                mtxdata->data.real_single;
            for (int64_t k = 0; k < mtxdata->size; k++) {
                int i = data[k].i;
                data[k].i = data[k].j;
                data[k].j = i;
            }
        } else if (mtxdata->precision == mtx_double) {
            struct mtx_matrix_coordinate_real_double * data =
                mtxdata->data.real_double;
            for (int64_t k = 0; k < mtxdata->size; k++) {
                int i = data[k].i;
                data[k].i = data[k].j;
                data[k].j = i;
            }
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (mtxdata->field == mtx_complex) {
        if (mtxdata->precision == mtx_single) {
            struct mtx_matrix_coordinate_complex_single * data =
                mtxdata->data.complex_single;
            for (int64_t k = 0; k < mtxdata->size; k++) {
                int i = data[k].i;
                data[k].i = data[k].j;
                data[k].j = i;
            }
        } else if (mtxdata->precision == mtx_double) {
            struct mtx_matrix_coordinate_complex_double * data =
                mtxdata->data.complex_double;
            for (int64_t k = 0; k < mtxdata->size; k++) {
                int i = data[k].i;
                data[k].i = data[k].j;
                data[k].j = i;
            }
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (mtxdata->field == mtx_integer) {
        if (mtxdata->precision == mtx_single) {
            struct mtx_matrix_coordinate_integer_single * data =
                mtxdata->data.integer_single;
            for (int64_t k = 0; k < mtxdata->size; k++) {
                int i = data[k].i;
                data[k].i = data[k].j;
                data[k].j = i;
            }
        } else if (mtxdata->precision == mtx_double) {
            struct mtx_matrix_coordinate_integer_double * data =
                mtxdata->data.integer_double;
            for (int64_t k = 0; k < mtxdata->size; k++) {
                int i = data[k].i;
                data[k].i = data[k].j;
                data[k].j = i;
            }
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (mtxdata->field == mtx_pattern) {
        struct mtx_matrix_coordinate_pattern * data =
            mtxdata->data.pattern;
            for (int64_t k = 0; k < mtxdata->size; k++) {
                int i = data[k].i;
                data[k].i = data[k].j;
                data[k].j = i;
            }
    } else {
        return MTX_ERR_INVALID_MTX_FIELD;
    }

    int num_rows = mtxdata->num_rows;
    mtxdata->num_rows = mtxdata->num_columns;
    mtxdata->num_columns = num_rows;
    return MTX_SUCCESS;
}

/**
 * `mtx_matrix_coordinate_data_submatrix()' obtains a submatrix
 * consisting of the given rows and columns.
 */
int mtx_matrix_coordinate_data_submatrix(
    struct mtx_matrix_coordinate_data * submtx,
    const struct mtx_matrix_coordinate_data * mtx,
    const struct mtxidxset * rows,
    const struct mtxidxset * columns)
{
    int err;

    /* Count the number of nonzeros in the submatrix. */
    int64_t num_nonzeros = 0;
    if (mtx->field == mtx_real) {
        if (mtx->precision == mtx_single) {
            const struct mtx_matrix_coordinate_real_single * src =
                mtx->data.real_single;
            for (int k = 0; k < mtx->size; k++) {
                bool has_row = mtxidxset_contains(rows, src[k].i);
                bool has_column = mtxidxset_contains(columns, src[k].j);
                if (has_row && has_column)
                    num_nonzeros++;
            }
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else {
        return MTX_ERR_INVALID_MTX_FIELD;
    }

    /* Allocate storage for data. */
    err = mtx_matrix_coordinate_data_alloc(
        submtx, mtx->field, mtx->precision,
        mtx->num_rows, mtx->num_columns, num_nonzeros);
    if (err)
        return err;

    submtx->symmetry = mtx->symmetry;
    submtx->triangle = mtx->triangle;
    submtx->sorting = mtx->sorting;
    submtx->assembly = mtx->assembly;

    /* Copy nonzeros that belong to the submatrix. */
    if (mtx->field == mtx_real) {
        if (mtx->precision == mtx_single) {
            const struct mtx_matrix_coordinate_real_single * src =
                mtx->data.real_single;
            struct mtx_matrix_coordinate_real_single * dst =
                submtx->data.real_single;
            int64_t l = 0;
            for (int k = 0; k < mtx->size; k++) {
                bool has_row = mtxidxset_contains(rows, src[k].i);
                bool has_column = mtxidxset_contains(columns, src[k].j);
                if (has_row && has_column) {
                    dst[l] = src[k];
                    l++;
                }
            }
        } else {
            mtx_matrix_coordinate_data_free(submtx);
            return MTX_ERR_INVALID_PRECISION;
        }
    } else {
        mtx_matrix_coordinate_data_free(submtx);
        return MTX_ERR_INVALID_MTX_FIELD;
    }
    return MTX_SUCCESS;
}
