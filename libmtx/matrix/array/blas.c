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
 * BLAS operations for dense matrices in array format.
 */

#include <libmtx/matrix/array/blas.h>

#include <libmtx/error.h>
#include <libmtx/mtx/mtx.h>
#include <libmtx/matrix/array.h>

#ifdef LIBMTX_HAVE_BLAS
#include <cblas.h>
#endif

#include <errno.h>

#include <math.h>

struct mtx;

/*
 * Level 1 BLAS operations.
 */

/**
 * `mtx_matrix_array_sscal()' scales a matrix by a single precision
 * floating-point scalar, `x = a*x'.
 */
int mtx_matrix_array_sscal(
    float a,
    struct mtx * x)
{
    if (x->object != mtx_matrix)
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
 * `mtx_matrix_array_dscal()' scales a matrix by a double precision
 * floating-point scalar, `x = a*x'.
 */
int mtx_matrix_array_dscal(
    double a,
    struct mtx * x)
{
    if (x->object != mtx_matrix)
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
 * `mtx_matrix_array_saxpy()' adds two matrices of single precision
 * floating-point values, `y = a*x + y'.
 */
int mtx_matrix_array_saxpy(
    float a,
    const struct mtx * x,
    struct mtx * y)
{
    if (x->object != mtx_matrix || y->object != mtx_matrix)
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
 * `mtx_matrix_array_daxpy()' adds two matrices of double precision
 * floating-point values, `y = a*x + y'.
 */
int mtx_matrix_array_daxpy(
    double a,
    const struct mtx * x,
    struct mtx * y)
{
    if (x->object != mtx_matrix || y->object != mtx_matrix)
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
 * `mtx_matrix_array_sdot()' computes the Frobenius inner product of
 * two matrices of single precision floating-point values.
 */
int mtx_matrix_array_sdot(
    const struct mtx * x,
    const struct mtx * y,
    float * dot)
{
    if (x->object != mtx_matrix || y->object != mtx_matrix)
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
 * `mtx_matrix_array_ddot()' computes the Frobenius inner product of
 * two matrices of double precision floating-point values.
 */
int mtx_matrix_array_ddot(
    const struct mtx * x,
    const struct mtx * y,
    double * dot)
{
    if (x->object != mtx_matrix || y->object != mtx_matrix)
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
 * `mtx_matrix_array_snrm2()' computes the Frobenius norm of a matrix
 * of single precision floating-point values.
 */
int mtx_matrix_array_snrm2(
    const struct mtx * x,
    float * nrm2)
{
    if (x->object != mtx_matrix)
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
 * `mtx_matrix_array_dnrm2()' computes the Frobenius norm of a matrix
 * of double precision floating-point values.
 */
int mtx_matrix_array_dnrm2(
    const struct mtx * x,
    double * nrm2)
{
    if (x->object != mtx_matrix)
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

/*
 * Level 2 BLAS operations.
 */

/**
 * `mtx_matrix_array_sgemv()' computes the product of a matrix and a
 * vector of single precision floating-point values, `y = alpha*A*x +
 * beta*y'.
 */
int mtx_matrix_array_sgemv(
    float alpha,
    const struct mtx * A,
    const struct mtx * x,
    float beta,
    struct mtx * y)
{
    if (A->object != mtx_matrix ||
        x->object != mtx_vector ||
        y->object != mtx_vector)
        return MTX_ERR_INVALID_MTX_OBJECT;
    if (A->field != mtx_real ||
        x->field != mtx_real ||
        y->field != mtx_real)
        return MTX_ERR_INVALID_MTX_FIELD;
    if (A->symmetry != mtx_general)
        return MTX_ERR_INVALID_MTX_SYMMETRY;
    if (A->num_rows != y->num_rows ||
        A->num_columns != x->num_rows)
        return MTX_ERR_INVALID_MTX_SIZE;
    if (A->format != mtx_array ||
        x->format != mtx_array ||
        y->format != mtx_array)
        return MTX_ERR_INVALID_MTX_FORMAT;
    if (A->sorting != mtx_row_major)
        return MTX_ERR_INVALID_MTX_SORTING;

    const float * Adata = (const float *) A->data;
    const float * xdata = (const float *) x->data;
    float * ydata = (float *) y->data;
#ifdef LIBMTX_HAVE_BLAS
    cblas_sgemv(
        CblasRowMajor, CblasNoTrans, A->num_rows, A->num_columns,
        alpha, Adata, A->num_columns, xdata, 1, beta, ydata, 1);
#else
    for (int i = 0; i < A->num_rows; i++) {
        float z = 0.0f;
        for (int j = 0; j < A->num_columns; j++)
            z += alpha*Adata[i*A->num_columns+j]*xdata[j];
        ydata[i] = z + beta*ydata[i];
    }
#endif
    return MTX_SUCCESS;
}

/**
 * `mtx_matrix_array_dgemv()' computes the product of a matrix and a
 * vector of single precision floating-point values, `y = alpha*A*x +
 * beta*y'.
 */
int mtx_matrix_array_dgemv(
    double alpha,
    const struct mtx * A,
    const struct mtx * x,
    double beta,
    struct mtx * y)
{
    if (A->object != mtx_matrix ||
        x->object != mtx_vector ||
        y->object != mtx_vector)
        return MTX_ERR_INVALID_MTX_OBJECT;
    if (A->field != mtx_double ||
        x->field != mtx_double ||
        y->field != mtx_double)
        return MTX_ERR_INVALID_MTX_FIELD;
    if (A->symmetry != mtx_general)
        return MTX_ERR_INVALID_MTX_SYMMETRY;
    if (A->num_rows != y->num_rows ||
        A->num_columns != x->num_rows)
        return MTX_ERR_INVALID_MTX_SIZE;
    if (A->format != mtx_array &&
        x->format != mtx_array &&
        y->format != mtx_array)
        return MTX_ERR_INVALID_MTX_FORMAT;

    if (A->sorting != mtx_row_major)
        return MTX_ERR_INVALID_MTX_SORTING;
    const double * Adata = (const double *) A->data;
    const double * xdata = (const double *) x->data;
    double * ydata = (double *) y->data;
#ifdef LIBMTX_HAVE_BLAS
    cblas_dgemv(
        CblasRowMajor, CblasNoTrans, A->num_rows, A->num_columns,
        alpha, Adata, A->num_columns, xdata, 1, beta, ydata, 1);
#else
    for (int i = 0; i < A->num_rows; i++) {
        double z = 0.0;
        for (int j = 0; j < A->num_columns; j++)
            z += alpha*Adata[i*A->num_columns+j]*xdata[j];
        ydata[i] = z + beta*ydata[i];
    }
#endif
    return MTX_SUCCESS;
}
