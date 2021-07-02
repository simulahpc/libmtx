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
 * Last modified: 2021-07-02
 *
 * BLAS operations for matrices and vectors in Matrix Market format.
 */

#include <matrixmarket/libmtx-config.h>

#include <matrixmarket/error.h>
#include <matrixmarket/blas.h>
#include <matrixmarket/matrix_coordinate.h>
#include <matrixmarket/mtx.h>
#include <matrixmarket/vector_array.h>
#include <matrixmarket/vector_coordinate.h>

#ifdef LIBMTX_HAVE_BLAS
#include <cblas.h>
#endif

#include <errno.h>

#include <math.h>

/*
 * Level 1 BLAS operations.
 */

/**
 * `mtx_sscal()' scales a vector by a single precision
 * floating-point scalar, `x = a*x'.
 */
int mtx_sscal(
    float a,
    struct mtx * x)
{
    if (x->object != mtx_vector || x->field != mtx_real) {
        errno = EINVAL;
        return MTX_ERR_ERRNO;
    }

    if (x->format == mtx_array) {
        float * xdata = (float *) x->data;
#ifdef LIBMTX_HAVE_BLAS
        cblas_sscal(x->size, a, xdata, 1);
        return MTX_SUCCESS;
#else
        for (int64_t i = 0; i < x->size; i++)
            xdata[i] *= a;
        return MTX_SUCCESS;
#endif
    } else if (x->format == mtx_coordinate) {
        struct mtx_vector_coordinate_real * xdata =
            (struct mtx_vector_coordinate_real *) x->data;
        for (int64_t i = 0; i < x->size; i++)
            xdata[i].a *= a;
        return MTX_SUCCESS;
    }
    return MTX_ERR_INVALID_MTX_FORMAT;
}

/**
 * `mtx_dscal()' scales a vector by a double precision
 * floating-point scalar, `x = a*x'.
 */
int mtx_dscal(
    double a,
    struct mtx * x)
{
    if (x->object != mtx_vector || x->field != mtx_double) {
        errno = EINVAL;
        return MTX_ERR_ERRNO;
    }

    if (x->format == mtx_array) {
        double * xdata = (double *) x->data;
#ifdef LIBMTX_HAVE_BLAS
        cblas_dscal(x->size, a, xdata, 1);
        return MTX_SUCCESS;
#else
        for (int64_t i = 0; i < x->size; i++)
            xdata[i] *= a;
        return MTX_SUCCESS;
#endif
    } else if (x->format == mtx_coordinate) {
        struct mtx_vector_coordinate_double * xdata =
            (struct mtx_vector_coordinate_double *) x->data;
        for (int64_t i = 0; i < x->size; i++)
            xdata[i].a *= a;
        return MTX_SUCCESS;
    }
    return MTX_ERR_INVALID_MTX_FORMAT;
}

/**
 * `mtx_saxpy()' adds two vectors of single precision
 * floating-point values, `y = a*x + y'.
 */
int mtx_saxpy(
    float a,
    const struct mtx * x,
    struct mtx * y)
{
    if (x->object != mtx_vector || x->field != mtx_real ||
        y->object != mtx_vector || y->field != mtx_real ||
        x->num_rows != y->num_rows)
    {
        errno = EINVAL;
        return MTX_ERR_ERRNO;
    }

    if (x->format == mtx_array &&
        y->format == mtx_array)
    {
        if (x->num_rows != x->size ||
            y->num_rows != y->size)
        {
            errno = EINVAL;
            return MTX_ERR_ERRNO;
        }
        const float * xdata = (const float *) x->data;
        float * ydata = (float *) y->data;
#ifdef LIBMTX_HAVE_BLAS
        cblas_saxpy(x->size, a, xdata, 1, ydata, 1);
        return MTX_SUCCESS;
#else
        for (int64_t i = 0; i < x->size; i++)
            ydata[i] += a*xdata[i];
        return MTX_SUCCESS;
#endif
    } else if (x->format == mtx_coordinate &&
               y->format == mtx_coordinate)
    {
        /* TODO: Implement vector addition for sparse vectors. */
        errno = ENOTSUP;
        return MTX_ERR_ERRNO;
    }
    return MTX_ERR_INVALID_MTX_FORMAT;
}

/**
 * `mtx_daxpy()' adds two vectors of double precision
 * floating-point values, `y = a*x + y'.
 */
int mtx_daxpy(
    double a,
    const struct mtx * x,
    struct mtx * y)
{
    if (x->object != mtx_vector || x->field != mtx_double ||
        y->object != mtx_vector || y->field != mtx_double ||
        x->num_rows != y->num_rows)
    {
        errno = EINVAL;
        return MTX_ERR_ERRNO;
    }

    if (x->format == mtx_array &&
        y->format == mtx_array)
    {
        if (x->num_rows != x->size ||
            y->num_rows != y->size)
        {
            errno = EINVAL;
            return MTX_ERR_ERRNO;
        }
        const double * xdata = (const double *) x->data;
        double * ydata = (double *) y->data;
#ifdef LIBMTX_HAVE_BLAS
        cblas_daxpy(x->size, a, xdata, 1, ydata, 1);
        return MTX_SUCCESS;
#else
        for (int64_t i = 0; i < x->size; i++)
            ydata[i] += a*xdata[i];
        return MTX_SUCCESS;
#endif
    } else if (x->format == mtx_coordinate &&
               y->format == mtx_coordinate)
    {
        /* TODO: Implement vector addition for sparse vectors. */
        errno = ENOTSUP;
        return MTX_ERR_ERRNO;
    }
    return MTX_ERR_INVALID_MTX_FORMAT;
}

/**
 * `mtx_sdot()' computes the dot product of two dense vectors of
 * single precision floating-point values.
 */
int mtx_sdot(
    const struct mtx * x,
    const struct mtx * y,
    float * dot)
{
    if (x->object != mtx_vector || x->field != mtx_real ||
        y->object != mtx_vector || y->field != mtx_real ||
        x->num_rows != y->num_rows)
    {
        errno = EINVAL;
        return MTX_ERR_ERRNO;
    }

    if (x->format == mtx_array &&
        y->format == mtx_array)
    {
        if (x->num_rows != x->size ||
            y->num_rows != y->size)
        {
            errno = EINVAL;
            return MTX_ERR_ERRNO;
        }
        const float * xdata = (const float *) x->data;
        const float * ydata = (const float *) y->data;
#ifdef LIBMTX_HAVE_BLAS
        *dot = cblas_sdot(x->size, xdata, 1, ydata, 1);
        return MTX_SUCCESS;
#else
        for (int64_t i = 0; i < x->size; i++)
            *dot += xdata[i]*ydata[i];
        return MTX_SUCCESS;
#endif
    } else if (x->format == mtx_coordinate &&
               y->format == mtx_coordinate)
    {
        /* TODO: Implement dot product for sparse vectors. */
        errno = ENOTSUP;
        return MTX_ERR_ERRNO;
    }
    return MTX_ERR_INVALID_MTX_FORMAT;
}

/**
 * `mtx_ddot()' computes the dot product of two dense vectors of
 * double precision floating-point values.
 */
int mtx_ddot(
    const struct mtx * x,
    const struct mtx * y,
    double * dot)
{
    if (x->object != mtx_vector || x->field != mtx_double ||
        y->object != mtx_vector || y->field != mtx_double ||
        x->num_rows != y->num_rows)
    {
        errno = EINVAL;
        return MTX_ERR_ERRNO;
    }

    if (x->format == mtx_array &&
        y->format == mtx_array)
    {
        if (x->num_rows != x->size ||
            y->num_rows != y->size)
        {
            errno = EINVAL;
            return MTX_ERR_ERRNO;
        }
        const double * xdata = (const double *) x->data;
        const double * ydata = (const double *) y->data;
#ifdef LIBMTX_HAVE_BLAS
        *dot = cblas_ddot(x->size, xdata, 1, ydata, 1);
        return MTX_SUCCESS;
#else
        for (int64_t i = 0; i < x->size; i++)
            *dot += xdata[i]*ydata[i];
        return MTX_SUCCESS;
#endif
    } else if (x->format == mtx_coordinate &&
               y->format == mtx_coordinate)
    {
        /* TODO: Implement dot product for sparse vectors. */
        errno = ENOTSUP;
        return MTX_ERR_ERRNO;
    }
    return MTX_ERR_INVALID_MTX_FORMAT;
}

/**
 * `mtx_snrm2()' computes the Euclidean norm of a vector of single
 * precision floating-point values.
 */
int mtx_snrm2(
    const struct mtx * x,
    float * nrm2)
{
    if (x->object != mtx_vector || x->field != mtx_real) {
        errno = EINVAL;
        return MTX_ERR_ERRNO;
    }

    if (x->format == mtx_array) {
        if (x->num_rows != x->size) {
            errno = EINVAL;
            return MTX_ERR_ERRNO;
        }
        const float * xdata = (const float *) x->data;
#ifdef LIBMTX_HAVE_BLAS
        *nrm2 = cblas_snrm2(x->size, xdata, 1);
        return MTX_SUCCESS;
#else
        for (int64_t i = 0; i < x->size; i++)
            *nrm2 += xdata[i]*xdata[i];
        *nrm2 = sqrtf(*nrm2);
        return MTX_SUCCESS;
#endif
    } else if (x->format == mtx_coordinate) {
        /* TODO: Implement Euclidean norm for sparse vectors. */
        errno = ENOTSUP;
        return MTX_ERR_ERRNO;
    }
    return MTX_ERR_INVALID_MTX_FORMAT;
}

/**
 * `mtx_dnrm2()' computes the Euclidean norm of a vector of double
 * precision floating-point values.
 */
int mtx_dnrm2(
    const struct mtx * x,
    double * nrm2)
{
    if (x->object != mtx_vector || x->field != mtx_double) {
        errno = EINVAL;
        return MTX_ERR_ERRNO;
    }

    if (x->format == mtx_array) {
        if (x->num_rows != x->size) {
            errno = EINVAL;
            return MTX_ERR_ERRNO;
        }
        const double * xdata = (const double *) x->data;
#ifdef LIBMTX_HAVE_BLAS
        *nrm2 = cblas_dnrm2(x->size, xdata, 1);
        return MTX_SUCCESS;
#else
        for (int64_t i = 0; i < x->size; i++)
            *nrm2 += xdata[i]*xdata[i];
        *nrm2 = sqrt(*nrm2);
        return MTX_SUCCESS;
#endif
    } else if (x->format == mtx_coordinate) {
        /* TODO: Implement Euclidean norm for sparse vectors. */
        errno = ENOTSUP;
        return MTX_ERR_ERRNO;
    }
    return MTX_ERR_INVALID_MTX_FORMAT;
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
    if (A->object != mtx_matrix || A->field != mtx_real ||
        A->symmetry != mtx_general ||
        x->object != mtx_vector || x->field != mtx_real ||
        y->object != mtx_vector || y->field != mtx_real ||
        A->num_rows != y->num_rows ||
        A->num_columns != x->num_rows)
    {
        errno = EINVAL;
        return MTX_ERR_ERRNO;
    }

    if (A->format == mtx_array &&
        x->format == mtx_array &&
        y->format == mtx_array)
    {
        if (A->sorting != mtx_row_major ||
            x->num_rows != x->size ||
            y->num_rows != y->size)
        {
            errno = EINVAL;
            return MTX_ERR_ERRNO;
        }
        const float * Adata = (const float *) A->data;
        const float * xdata = (const float *) x->data;
        float * ydata = (float *) y->data;
#ifdef LIBMTX_HAVE_BLAS
        cblas_sgemv(
            CblasRowMajor, CblasNoTrans, A->num_rows, A->num_columns,
            alpha, Adata, A->num_columns, xdata, 1, beta, ydata, 1);
        return MTX_SUCCESS;
#else
        for (int i = 0; i < A->num_rows; i++) {
            float z = 0.0f;
            for (int j = 0; j < A->num_columns; j++)
                z += alpha*Adata[i*A->num_columns+j]*xdata[j];
            ydata[i] = z + beta*ydata[i];
        }
        return MTX_SUCCESS;
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
        return MTX_SUCCESS;
    }
    return MTX_ERR_INVALID_MTX_FORMAT;
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
    if (A->object != mtx_matrix || A->field != mtx_double ||
        A->symmetry != mtx_general ||
        x->object != mtx_vector || x->field != mtx_double ||
        y->object != mtx_vector || y->field != mtx_double ||
        A->num_rows != y->num_rows ||
        A->num_columns != x->num_rows)
    {
        errno = EINVAL;
        return MTX_ERR_ERRNO;
    }

    if (A->format == mtx_array &&
        x->format == mtx_array &&
        y->format == mtx_array)
    {
        if (A->sorting != mtx_row_major ||
            x->num_rows != x->size ||
            y->num_rows != y->size)
        {
            errno = EINVAL;
            return MTX_ERR_ERRNO;
        }
        const double * Adata = (const double *) A->data;
        const double * xdata = (const double *) x->data;
        double * ydata = (double *) y->data;
#ifdef LIBMTX_HAVE_BLAS
        cblas_dgemv(
            CblasRowMajor, CblasNoTrans, A->num_rows, A->num_columns,
            alpha, Adata, A->num_columns, xdata, 1, beta, ydata, 1);
        return MTX_SUCCESS;
#else
        for (int i = 0; i < A->num_rows; i++) {
            double z = 0.0;
            for (int j = 0; j < A->num_columns; j++)
                z += alpha*Adata[i*A->num_columns+j]*xdata[j];
            ydata[i] = z + beta*ydata[i];
        }
        return MTX_SUCCESS;
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
    }
    return MTX_ERR_INVALID_MTX_FORMAT;
}
