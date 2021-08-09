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
 * Level 1 BLAS operations for dense vectors in array format.
 */

#include <libmtx/vector/array/blas.h>

#include <libmtx/error.h>
#include <libmtx/mtx/mtx.h>
#include <libmtx/vector/array/array.h>

#ifdef LIBMTX_HAVE_BLAS
#include <cblas.h>
#endif

#include <errno.h>

#include <math.h>

struct mtx;

/**
 * `mtx_vector_array_sscal()' scales a vector (or matrix) by a single
 * precision floating-point scalar, `x = a*x'.
 */
int mtx_vector_array_sscal(
    float a,
    struct mtx * x)
{
    if (x->object != mtx_vector)
        return MTX_ERR_INVALID_MTX_OBJECT;
    if (x->format != mtx_array)
        return MTX_ERR_INVALID_MTX_FORMAT;
    if (x->field != mtx_real)
        return MTX_ERR_INVALID_MTX_FIELD;

    float * xdata = (float *) x->data;
#ifdef LIBMTX_HAVE_BLAS
    cblas_sscal(x->size, a, xdata, 1);
#else
    for (int64_t i = 0; i < x->size; i++)
        xdata[i] *= a;
#endif
    return MTX_SUCCESS;
}

/**
 * `mtx_vector_array_dscal()' scales a vector (or matrix) by a double
 * precision floating-point scalar, `x = a*x'.
 */
int mtx_vector_array_dscal(
    double a,
    struct mtx * x)
{
    if (x->object != mtx_vector)
        return MTX_ERR_INVALID_MTX_OBJECT;
    if (x->format != mtx_array)
        return MTX_ERR_INVALID_MTX_FORMAT;
    if (x->field != mtx_double)
        return MTX_ERR_INVALID_MTX_FIELD;

    double * xdata = (double *) x->data;
#ifdef LIBMTX_HAVE_BLAS
    cblas_dscal(x->size, a, xdata, 1);
#else
    for (int64_t i = 0; i < x->size; i++)
        xdata[i] *= a;
#endif
    return MTX_SUCCESS;
}

/**
 * `mtx_vector_array_saxpy()' adds two vectors (or matrices) of single
 * precision floating-point values, `y = a*x + y'.
 */
int mtx_vector_array_saxpy(
    float a,
    const struct mtx * x,
    struct mtx * y)
{
    if (x->object != mtx_vector || y->object != mtx_vector)
        return MTX_ERR_INVALID_MTX_OBJECT;
    if (x->format != mtx_array || y->format != mtx_array)
        return MTX_ERR_INVALID_MTX_FORMAT;
    if (x->field != mtx_real || y->field != mtx_real)
        return MTX_ERR_INVALID_MTX_FIELD;
    if (x->size != y->size)
        return MTX_ERR_INVALID_MTX_SIZE;
    if (x->sorting != y->sorting)
        return MTX_ERR_INVALID_MTX_SORTING;

    const float * xdata = (const float *) x->data;
    float * ydata = (float *) y->data;
#ifdef LIBMTX_HAVE_BLAS
    cblas_saxpy(x->size, a, xdata, 1, ydata, 1);
#else
    for (int64_t i = 0; i < x->size; i++)
        ydata[i] += a*xdata[i];
#endif
    return MTX_SUCCESS;
}

/**
 * `mtx_vector_array_daxpy()' adds two vectors (or matrices) of double
 * precision floating-point values, `y = a*x + y'.
 */
int mtx_vector_array_daxpy(
    double a,
    const struct mtx * x,
    struct mtx * y)
{
    if (x->object != mtx_vector || y->object != mtx_vector)
        return MTX_ERR_INVALID_MTX_OBJECT;
    if (x->format != mtx_array || y->format != mtx_array)
        return MTX_ERR_INVALID_MTX_FORMAT;
    if (x->field != mtx_double || y->field != mtx_double)
        return MTX_ERR_INVALID_MTX_FIELD;
    if (x->size != y->size)
        return MTX_ERR_INVALID_MTX_SIZE;
    if (x->sorting != y->sorting)
        return MTX_ERR_INVALID_MTX_SORTING;

    const double * xdata = (const double *) x->data;
    double * ydata = (double *) y->data;
#ifdef LIBMTX_HAVE_BLAS
    cblas_daxpy(x->size, a, xdata, 1, ydata, 1);
#else
    for (int64_t i = 0; i < x->size; i++)
        ydata[i] += a*xdata[i];
#endif
    return MTX_SUCCESS;
}

/**
 * `mtx_vector_array_sdot()' computes the Euclidean dot product of two
 * vectors (or Frobenius inner product of two matrices) of single
 * precision floating-point values.
 */
int mtx_vector_array_sdot(
    const struct mtx * x,
    const struct mtx * y,
    float * dot)
{
    if (x->object != mtx_vector || y->object != mtx_vector)
        return MTX_ERR_INVALID_MTX_OBJECT;
    if (x->format != mtx_array || y->format != mtx_array)
        return MTX_ERR_INVALID_MTX_FORMAT;
    if (x->field != mtx_real || y->field != mtx_real)
        return MTX_ERR_INVALID_MTX_FIELD;
    if (x->size != y->size)
        return MTX_ERR_INVALID_MTX_SIZE;
    if (x->sorting != y->sorting)
        return MTX_ERR_INVALID_MTX_SORTING;

    const float * xdata = (const float *) x->data;
    const float * ydata = (const float *) y->data;
#ifdef LIBMTX_HAVE_BLAS
    *dot = cblas_sdot(x->size, xdata, 1, ydata, 1);
#else
    for (int64_t i = 0; i < x->size; i++)
        *dot += xdata[i]*ydata[i];
#endif
    return MTX_SUCCESS;
}

/**
 * `mtx_vector_array_ddot()' computes the Euclidean dot product of two
 * vectors (or Frobenius inner product of two matrices) of double
 * precision floating-point values.
 */
int mtx_vector_array_ddot(
    const struct mtx * x,
    const struct mtx * y,
    double * dot)
{
    if (x->object != mtx_vector || y->object != mtx_vector)
        return MTX_ERR_INVALID_MTX_OBJECT;
    if (x->format != mtx_array || y->format != mtx_array)
        return MTX_ERR_INVALID_MTX_FORMAT;
    if (x->field != mtx_double || y->field != mtx_double)
        return MTX_ERR_INVALID_MTX_FIELD;
    if (x->size != y->size)
        return MTX_ERR_INVALID_MTX_SIZE;
    if (x->sorting != y->sorting)
        return MTX_ERR_INVALID_MTX_SORTING;

    const double * xdata = (const double *) x->data;
    const double * ydata = (const double *) y->data;
#ifdef LIBMTX_HAVE_BLAS
    *dot = cblas_ddot(x->size, xdata, 1, ydata, 1);
#else
    for (int64_t i = 0; i < x->size; i++)
        *dot += xdata[i]*ydata[i];
#endif
    return MTX_SUCCESS;
}

/**
 * `mtx_vector_array_snrm2()' computes the Euclidean norm of a vector
 * (or Frobenius norm of a matrix) of single precision floating-point
 * values.
 */
int mtx_vector_array_snrm2(
    const struct mtx * x,
    float * nrm2)
{
    if (x->object != mtx_vector)
        return MTX_ERR_INVALID_MTX_OBJECT;
    if (x->format != mtx_array)
        return MTX_ERR_INVALID_MTX_FORMAT;
    if (x->field != mtx_real)
        return MTX_ERR_INVALID_MTX_FIELD;

    const float * xdata = (const float *) x->data;
#ifdef LIBMTX_HAVE_BLAS
    *nrm2 = cblas_snrm2(x->size, xdata, 1);
#else
    for (int64_t i = 0; i < x->size; i++)
        *nrm2 += xdata[i]*xdata[i];
    *nrm2 = sqrtf(*nrm2);
#endif
    return MTX_SUCCESS;
}

/**
 * `mtx_vector_array_dnrm2()' computes the Euclidean norm of a vector
 * (or Frobenius norm of a matrix) of double precision floating-point
 * values.
 */
int mtx_vector_array_dnrm2(
    const struct mtx * x,
    double * nrm2)
{
    if (x->object != mtx_vector)
        return MTX_ERR_INVALID_MTX_OBJECT;
    if (x->format != mtx_array)
        return MTX_ERR_INVALID_MTX_FORMAT;
    if (x->field != mtx_double)
        return MTX_ERR_INVALID_MTX_FIELD;

    const double * xdata = (const double *) x->data;
#ifdef LIBMTX_HAVE_BLAS
    *nrm2 = cblas_dnrm2(x->size, xdata, 1);
#else
    for (int64_t i = 0; i < x->size; i++)
        *nrm2 += xdata[i]*xdata[i];
    *nrm2 = sqrt(*nrm2);
#endif
    return MTX_SUCCESS;
}
