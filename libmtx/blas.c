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
 * BLAS operations for matrices and vectors in Matrix Market format.
 */

#include <libmtx/libmtx-config.h>

#include <libmtx/error.h>
#include <libmtx/blas.h>
#include <libmtx/matrix_coordinate.h>
#include <libmtx/mtx.h>
#include <libmtx/vector_array.h>
#include <libmtx/vector_coordinate.h>

#ifdef LIBMTX_HAVE_BLAS
#include <cblas.h>
#endif

#include <errno.h>

#include <math.h>

/*
 * Level 1 BLAS operations.
 */

/**
 * `mtx_sscal()' scales a vector (or matrix) by a single precision
 * floating-point scalar, `x = a*x'.
 */
int mtx_sscal(
    float a,
    struct mtx * x)
{
    if (x->field != mtx_real)
        return MTX_ERR_INVALID_MTX_FIELD;

    if (x->format == mtx_array) {
        float * xdata = (float *) x->data;
#ifdef LIBMTX_HAVE_BLAS
        cblas_sscal(x->size, a, xdata, 1);
#else
        for (int64_t i = 0; i < x->size; i++)
            xdata[i] *= a;
#endif
    } else if (x->format == mtx_coordinate) {
        if (x->object == mtx_matrix) {
            struct mtx_matrix_coordinate_real * xdata =
                (struct mtx_matrix_coordinate_real *) x->data;
            for (int64_t i = 0; i < x->size; i++)
                xdata[i].a *= a;
        } else if (x->object == mtx_vector) {
            struct mtx_vector_coordinate_real * xdata =
                (struct mtx_vector_coordinate_real *) x->data;
            for (int64_t i = 0; i < x->size; i++)
                xdata[i].a *= a;
        } else {
            return MTX_ERR_INVALID_MTX_OBJECT;
        }
    } else {
        return MTX_ERR_INVALID_MTX_FORMAT;
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_dscal()' scales a vector (or matrix) by a double precision
 * floating-point scalar, `x = a*x'.
 */
int mtx_dscal(
    double a,
    struct mtx * x)
{
    if (x->field != mtx_double)
        return MTX_ERR_INVALID_MTX_FIELD;

    if (x->format == mtx_array) {
        double * xdata = (double *) x->data;
#ifdef LIBMTX_HAVE_BLAS
        cblas_dscal(x->size, a, xdata, 1);
#else
        for (int64_t i = 0; i < x->size; i++)
            xdata[i] *= a;
#endif
    } else if (x->format == mtx_coordinate) {
        if (x->object == mtx_matrix) {
            struct mtx_matrix_coordinate_double * xdata =
                (struct mtx_matrix_coordinate_double *) x->data;
            for (int64_t i = 0; i < x->size; i++)
                xdata[i].a *= a;
        } else if (x->object == mtx_vector) {
            struct mtx_vector_coordinate_double * xdata =
                (struct mtx_vector_coordinate_double *) x->data;
            for (int64_t i = 0; i < x->size; i++)
                xdata[i].a *= a;
        } else {
            return MTX_ERR_INVALID_MTX_OBJECT;
        }
    } else {
        return MTX_ERR_INVALID_MTX_FORMAT;
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_saxpy()' adds two vectors (or matrices) of single precision
 * floating-point values, `y = a*x + y'.
 */
int mtx_saxpy(
    float a,
    const struct mtx * x,
    struct mtx * y)
{
    if (x->field != mtx_real || y->field != mtx_real)
        return MTX_ERR_INVALID_MTX_FIELD;
    if (x->num_rows != y->num_rows || x->num_columns != y->num_columns)
        return MTX_ERR_INVALID_MTX_SIZE;

    if (x->format == mtx_array &&
        y->format == mtx_array)
    {
        if (x->size != y->size)
            return MTX_ERR_INVALID_MTX_SIZE;
        if (x->symmetry != y->symmetry)
            return MTX_ERR_INVALID_MTX_SYMMETRY;
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
    } else if (x->format == mtx_coordinate &&
               y->format == mtx_coordinate)
    {
        /* TODO: Implement vector addition for sparse vectors. */
        errno = ENOTSUP;
        return MTX_ERR_ERRNO;
    } else {
        return MTX_ERR_INVALID_MTX_FORMAT;
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_daxpy()' adds two vectors (or matrices) of double precision
 * floating-point values, `y = a*x + y'.
 */
int mtx_daxpy(
    double a,
    const struct mtx * x,
    struct mtx * y)
{
    if (x->field != mtx_double || y->field != mtx_double)
        return MTX_ERR_INVALID_MTX_FIELD;
    if (x->num_rows != y->num_rows || x->num_columns != y->num_columns)
        return MTX_ERR_INVALID_MTX_SIZE;

    if (x->format == mtx_array &&
        y->format == mtx_array)
    {
        if (x->size != y->size)
            return MTX_ERR_INVALID_MTX_SIZE;
        if (x->symmetry != y->symmetry)
            return MTX_ERR_INVALID_MTX_SYMMETRY;
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
    } else if (x->format == mtx_coordinate &&
               y->format == mtx_coordinate)
    {
        /* TODO: Implement vector addition for sparse vectors. */
        errno = ENOTSUP;
        return MTX_ERR_ERRNO;
    } else {
        return MTX_ERR_INVALID_MTX_FORMAT;
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_sdot()' computes the Euclidean dot product of two vectors (or
 * Frobenius inner product of two matrices) of single precision
 * floating-point values.
 */
int mtx_sdot(
    const struct mtx * x,
    const struct mtx * y,
    float * dot)
{
    if (x->field != mtx_real || y->field != mtx_real)
        return MTX_ERR_INVALID_MTX_FIELD;
    if (x->num_rows != y->num_rows ||
        x->num_columns != y->num_columns)
        return MTX_ERR_INVALID_MTX_SIZE;

    if (x->format == mtx_array &&
        y->format == mtx_array)
    {
        if (x->size != y->size)
            return MTX_ERR_INVALID_MTX_SIZE;
        if (x->symmetry != y->symmetry)
            return MTX_ERR_INVALID_MTX_SYMMETRY;
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
    } else if (x->format == mtx_coordinate &&
               y->format == mtx_coordinate)
    {
        /* TODO: Implement dot product for sparse vectors. */
        errno = ENOTSUP;
        return MTX_ERR_ERRNO;
    } else {
        return MTX_ERR_INVALID_MTX_FORMAT;
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_ddot()' computes the Euclidean dot product of two vectors (or
 * Frobenius inner product of two matrices) of double precision
 * floating-point values.
 */
int mtx_ddot(
    const struct mtx * x,
    const struct mtx * y,
    double * dot)
{
    if (x->field != mtx_double || y->field != mtx_double)
        return MTX_ERR_INVALID_MTX_FIELD;
    if (x->num_rows != y->num_rows ||
        x->num_columns != y->num_columns)
        return MTX_ERR_INVALID_MTX_SIZE;

    if (x->format == mtx_array &&
        y->format == mtx_array)
    {
        if (x->size != y->size)
            return MTX_ERR_INVALID_MTX_SIZE;
        if (x->symmetry != y->symmetry)
            return MTX_ERR_INVALID_MTX_SYMMETRY;
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
    } else if (x->format == mtx_coordinate &&
               y->format == mtx_coordinate)
    {
        /* TODO: Implement dot product for sparse vectors. */
        errno = ENOTSUP;
        return MTX_ERR_ERRNO;
    } else {
        return MTX_ERR_INVALID_MTX_FORMAT;
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_snrm2()' computes the Euclidean norm of a vector (or Frobenius
 * norm of a matrix) of single precision floating-point values.
 */
int mtx_snrm2(
    const struct mtx * x,
    float * nrm2)
{
    if (x->field != mtx_real)
        return MTX_ERR_INVALID_MTX_FIELD;

    if (x->format == mtx_array) {
        if (x->symmetry != mtx_general)
            return MTX_ERR_INVALID_MTX_SYMMETRY;
        const float * xdata = (const float *) x->data;
#ifdef LIBMTX_HAVE_BLAS
        *nrm2 = cblas_snrm2(x->size, xdata, 1);
#else
        for (int64_t i = 0; i < x->size; i++)
            *nrm2 += xdata[i]*xdata[i];
        *nrm2 = sqrtf(*nrm2);
#endif
    } else if (x->format == mtx_coordinate) {
        /* TODO: Implement Euclidean norm for sparse vectors. */
        errno = ENOTSUP;
        return MTX_ERR_ERRNO;
    } else {
        return MTX_ERR_INVALID_MTX_FORMAT;
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_dnrm2()' computes the Euclidean norm of a vector (or Frobenius
 * norm of a matrix) of double precision floating-point values.
 */
int mtx_dnrm2(
    const struct mtx * x,
    double * nrm2)
{
    if (x->field != mtx_double)
        return MTX_ERR_INVALID_MTX_FIELD;

    if (x->format == mtx_array) {
        if (x->symmetry != mtx_general)
            return MTX_ERR_INVALID_MTX_SYMMETRY;
        const double * xdata = (const double *) x->data;
#ifdef LIBMTX_HAVE_BLAS
        *nrm2 = cblas_dnrm2(x->size, xdata, 1);
#else
        for (int64_t i = 0; i < x->size; i++)
            *nrm2 += xdata[i]*xdata[i];
        *nrm2 = sqrt(*nrm2);
#endif
    } else if (x->format == mtx_coordinate) {
        /* TODO: Implement Euclidean norm for sparse vectors. */
        errno = ENOTSUP;
        return MTX_ERR_ERRNO;
    } else {
        return MTX_ERR_INVALID_MTX_FORMAT;
    }
    return MTX_SUCCESS;
}

/*
 * Level 2 BLAS operations.
 */

/**
 * `mtx_sgemv()' computes the product of a matrix and a vector of
 * single precision floating-point values, `y = alpha*A*x + beta*y'.
 */
int mtx_sgemv(
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

    if (A->format == mtx_array &&
        x->format == mtx_array &&
        y->format == mtx_array)
    {
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
    } else if (A->format == mtx_coordinate &&
               x->format == mtx_array &&
               y->format == mtx_array)
    {
        /* TODO: The naive algorithm below only works if `beta' is
         * equal to one. Otherwise, some intermediate storage will be
         * needed for the values of the matrix-vector product, and a
         * final vector addition must be performed. */
        if (beta != 1.0) {
            errno = ENOTSUP;
            return MTX_ERR_ERRNO;
        }
        const struct mtx_matrix_coordinate_real * Adata =
            (const struct mtx_matrix_coordinate_real *) A->data;
        const float * xdata = (const float *) x->data;
        float * ydata = (float *) y->data;
        for (int64_t k = 0; k < A->size; k++)
            ydata[Adata[k].i-1] += alpha*Adata[k].a*xdata[Adata[k].j-1];
    } else {
        return MTX_ERR_INVALID_MTX_FORMAT;
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_dgemv()' computes the product of a matrix and a vector of
 * single precision floating-point values, `y = alpha*A*x + beta*y'.
 */
int mtx_dgemv(
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

    if (A->format == mtx_array &&
        x->format == mtx_array &&
        y->format == mtx_array)
    {
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
    } else if (A->format == mtx_coordinate &&
               x->format == mtx_array &&
               y->format == mtx_array)
    {
        /* TODO: The naive algorithm below only works if `beta' is
         * equal to one. Otherwise, some intermediate storage will be
         * needed for the values of the matrix-vector product, and a
         * final vector addition must be performed. */
        if (beta != 1.0) {
            errno = ENOTSUP;
            return MTX_ERR_ERRNO;
        }
        const struct mtx_matrix_coordinate_double * Adata =
            (const struct mtx_matrix_coordinate_double *) A->data;
        const double * xdata = (const double *) x->data;
        double * ydata = (double *) y->data;
        for (int64_t k = 0; k < A->size; k++)
            ydata[Adata[k].i-1] += alpha*Adata[k].a*xdata[Adata[k].j-1];
        return MTX_SUCCESS;
    } else {
        return MTX_ERR_INVALID_MTX_FORMAT;
    }
    return MTX_SUCCESS;
}
