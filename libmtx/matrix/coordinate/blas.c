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
 * BLAS operations for sparse matrices in coordinate format.
 */

#include <libmtx/matrix/coordinate/blas.h>

#include <libmtx/error.h>
#include <libmtx/mtx/mtx.h>
#include <libmtx/matrix/coordinate/coordinate.h>

#include <errno.h>

#include <math.h>

struct mtx;

/**
 * `mtx_matrix_coordinate_sscal()' scales a matrix by a single
 * precision floating-point scalar, `x = a*x'.
 */
int mtx_matrix_coordinate_sscal(
    float a,
    struct mtx * x)
{
    if (x->object != mtx_matrix)
        return MTX_ERR_INVALID_MTX_OBJECT;
    if (x->format != mtx_coordinate)
        return MTX_ERR_INVALID_MTX_FORMAT;
    if (x->field != mtx_real)
        return MTX_ERR_INVALID_MTX_FIELD;

    struct mtx_matrix_coordinate_real * xdata =
        (struct mtx_matrix_coordinate_real *) x->data;
    for (int64_t i = 0; i < x->size; i++)
        xdata[i].a *= a;
    return MTX_SUCCESS;
}

/**
 * `mtx_matrix_coordinate_dscal()' scales a matrix by a double
 * precision floating-point scalar, `x = a*x'.
 */
int mtx_matrix_coordinate_dscal(
    double a,
    struct mtx * x)
{
    if (x->object != mtx_matrix)
        return MTX_ERR_INVALID_MTX_OBJECT;
    if (x->format != mtx_coordinate)
        return MTX_ERR_INVALID_MTX_FORMAT;
    if (x->field != mtx_double)
        return MTX_ERR_INVALID_MTX_FIELD;

    struct mtx_matrix_coordinate_double * xdata =
        (struct mtx_matrix_coordinate_double *) x->data;
    for (int64_t i = 0; i < x->size; i++)
        xdata[i].a *= a;
    return MTX_SUCCESS;
}

/**
 * `mtx_matrix_coordinate_saxpy()' adds two matrices of single
 * precision floating-point values, `y = a*x + y'.
 */
int mtx_matrix_coordinate_saxpy(
    float a,
    const struct mtx * x,
    struct mtx * y)
{
    if (x->object != mtx_matrix || y->object != mtx_matrix)
        return MTX_ERR_INVALID_MTX_OBJECT;
    if (x->format != mtx_coordinate || y->format != mtx_coordinate)
        return MTX_ERR_INVALID_MTX_FORMAT;
    if (x->field != mtx_real || y->field != mtx_real)
        return MTX_ERR_INVALID_MTX_FIELD;
    if (x->size != y->size)
        return MTX_ERR_INVALID_MTX_SIZE;
    if (x->sorting != y->sorting)
        return MTX_ERR_INVALID_MTX_SORTING;

    /* TODO: Implement matrix addition for sparse matrices. */
    errno = ENOTSUP;
    return MTX_ERR_ERRNO;
}

/**
 * `mtx_matrix_coordinate_daxpy()' adds two matrices of double
 * precision floating-point values, `y = a*x + y'.
 */
int mtx_matrix_coordinate_daxpy(
    double a,
    const struct mtx * x,
    struct mtx * y)
{
    if (x->object != mtx_matrix || y->object != mtx_matrix)
        return MTX_ERR_INVALID_MTX_OBJECT;
    if (x->format != mtx_coordinate || y->format != mtx_coordinate)
        return MTX_ERR_INVALID_MTX_FORMAT;
    if (x->field != mtx_double || y->field != mtx_double)
        return MTX_ERR_INVALID_MTX_FIELD;
    if (x->size != y->size)
        return MTX_ERR_INVALID_MTX_SIZE;
    if (x->sorting != y->sorting)
        return MTX_ERR_INVALID_MTX_SORTING;

    /* TODO: Implement matrix addition for sparse matrices. */
    errno = ENOTSUP;
    return MTX_ERR_ERRNO;
}

/**
 * `mtx_matrix_coordinate_sdot()' computes the Euclidean dot product
 * of two matrices of single precision floating-point values.
 */
int mtx_matrix_coordinate_sdot(
    const struct mtx * x,
    const struct mtx * y,
    float * dot)
{
    if (x->object != mtx_matrix || y->object != mtx_matrix)
        return MTX_ERR_INVALID_MTX_OBJECT;
    if (x->format != mtx_coordinate || y->format != mtx_coordinate)
        return MTX_ERR_INVALID_MTX_FORMAT;
    if (x->field != mtx_real || y->field != mtx_real)
        return MTX_ERR_INVALID_MTX_FIELD;
    if (x->size != y->size)
        return MTX_ERR_INVALID_MTX_SIZE;
    if (x->sorting != y->sorting)
        return MTX_ERR_INVALID_MTX_SORTING;

    /* TODO: Implement dot product for sparse matrices. */
    errno = ENOTSUP;
    return MTX_ERR_ERRNO;
}

/**
 * `mtx_matrix_coordinate_ddot()' computes the Euclidean dot product
 * of two matrices of double precision floating-point values.
 */
int mtx_matrix_coordinate_ddot(
    const struct mtx * x,
    const struct mtx * y,
    double * dot)
{
    if (x->object != mtx_matrix || y->object != mtx_matrix)
        return MTX_ERR_INVALID_MTX_OBJECT;
    if (x->format != mtx_coordinate || y->format != mtx_coordinate)
        return MTX_ERR_INVALID_MTX_FORMAT;
    if (x->field != mtx_double || y->field != mtx_double)
        return MTX_ERR_INVALID_MTX_FIELD;
    if (x->size != y->size)
        return MTX_ERR_INVALID_MTX_SIZE;
    if (x->sorting != y->sorting)
        return MTX_ERR_INVALID_MTX_SORTING;

    /* TODO: Implement dot product for sparse matrices. */
    errno = ENOTSUP;
    return MTX_ERR_ERRNO;
}

/**
 * `mtx_matrix_coordinate_snrm2()' computes the Euclidean norm of a
 * matrix of single precision floating-point values.
 */
int mtx_matrix_coordinate_snrm2(
    const struct mtx * x,
    float * nrm2)
{
    if (x->object != mtx_matrix)
        return MTX_ERR_INVALID_MTX_OBJECT;
    if (x->format != mtx_coordinate)
        return MTX_ERR_INVALID_MTX_FORMAT;
    if (x->field != mtx_real)
        return MTX_ERR_INVALID_MTX_FIELD;

    /* TODO: Implement dot product for sparse matrices. */
    errno = ENOTSUP;
    return MTX_ERR_ERRNO;
}

/**
 * `mtx_matrix_coordinate_dnrm2()' computes the Euclidean norm of a
 * matrix of double precision floating-point values.
 */
int mtx_matrix_coordinate_dnrm2(
    const struct mtx * x,
    double * nrm2)
{
    if (x->object != mtx_matrix)
        return MTX_ERR_INVALID_MTX_OBJECT;
    if (x->format != mtx_coordinate)
        return MTX_ERR_INVALID_MTX_FORMAT;
    if (x->field != mtx_double)
        return MTX_ERR_INVALID_MTX_FIELD;

    /* TODO: Implement dot product for sparse matrices. */
    errno = ENOTSUP;
    return MTX_ERR_ERRNO;
}
