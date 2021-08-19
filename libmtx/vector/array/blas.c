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
 * Level 1 BLAS operations for vectors in array format.
 */

#include <libmtx/error.h>
#include <libmtx/vector/array.h>
#include <libmtx/vector/array/blas.h>
#include <libmtx/vector/array/data.h>

#ifdef LIBMTX_HAVE_BLAS
#include <cblas.h>
#endif

#include <errno.h>

#include <math.h>

/**
 * `mtx_vector_array_sscal()' scales a vector by a single precision
 * floating-point scalar, `x = a*x'.
 */
int mtx_vector_array_sscal(
    float a,
    struct mtx_vector_array_data * x)
{
    if (x->field == mtx_real) {
        if (x->precision == mtx_single) {
            float * xdata = x->data.real_single;
#ifdef LIBMTX_HAVE_BLAS
            cblas_sscal(x->size, a, xdata, 1);
#else
            for (int64_t k = 0; k < x->size; k++)
                xdata[k] *= a;
#endif
        } else if (x->precision == mtx_double) {
            double * xdata = x->data.real_double;
#ifdef LIBMTX_HAVE_BLAS
            cblas_dscal(x->size, a, xdata, 1);
#else
            for (int64_t k = 0; k < x->size; k++)
                xdata[k] *= a;
#endif
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else {
        return MTX_ERR_INVALID_MTX_FIELD;
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_vector_array_dscal()' scales a vector by a double precision
 * floating-point scalar, `x = a*x'.
 */
int mtx_vector_array_dscal(
    double a,
    struct mtx_vector_array_data * x)
{
    if (x->field == mtx_real) {
        if (x->precision == mtx_single) {
            float * xdata = x->data.real_single;
#ifdef LIBMTX_HAVE_BLAS
            cblas_sscal(x->size, a, xdata, 1);
#else
            for (int64_t k = 0; k < x->size; k++)
                xdata[k] *= a;
#endif
        } else if (x->precision == mtx_double) {
            double * xdata = x->data.real_double;
#ifdef LIBMTX_HAVE_BLAS
            cblas_dscal(x->size, a, xdata, 1);
#else
            for (int64_t k = 0; k < x->size; k++)
                xdata[k] *= a;
#endif
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else {
        return MTX_ERR_INVALID_MTX_FIELD;
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_vector_array_saxpy()' adds two vectors of single precision
 * floating-point values, `y = a*x + y'.
 */
int mtx_vector_array_saxpy(
    float a,
    const struct mtx_vector_array_data * x,
    struct mtx_vector_array_data * y)
{
    if (x->size != y->size)
        return MTX_ERR_INVALID_MTX_SIZE;

    if (x->field == mtx_real && y->field == mtx_real) {
        if (x->precision == mtx_single && y->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            float * ydata = y->data.real_single;
#ifdef LIBMTX_HAVE_BLAS
            cblas_saxpy(x->size, a, xdata, 1, ydata, 1);
#else
            for (int64_t k = 0; k < x->size; k++)
                ydata[k] += a*xdata[k];
#endif
        } else if (x->precision == mtx_double && y->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            double * ydata = y->data.real_double;
#ifdef LIBMTX_HAVE_BLAS
            cblas_daxpy(x->size, a, xdata, 1, ydata, 1);
#else
            for (int64_t k = 0; k < x->size; k++)
                ydata[k] += a*xdata[k];
#endif
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else {
        return MTX_ERR_INVALID_MTX_FIELD;
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_vector_array_daxpy()' adds two vectors of double precision
 * floating-point values, `y = a*x + y'.
 */
int mtx_vector_array_daxpy(
    double a,
    const struct mtx_vector_array_data * x,
    struct mtx_vector_array_data * y)
{
    if (x->size != y->size)
        return MTX_ERR_INVALID_MTX_SIZE;

    if (x->field == mtx_real && y->field == mtx_real) {
        if (x->precision == mtx_single && y->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            float * ydata = y->data.real_single;
#ifdef LIBMTX_HAVE_BLAS
            cblas_saxpy(x->size, a, xdata, 1, ydata, 1);
#else
            for (int64_t k = 0; k < x->size; k++)
                ydata[k] += a*xdata[k];
#endif
        } else if (x->precision == mtx_double && y->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            double * ydata = y->data.real_double;
#ifdef LIBMTX_HAVE_BLAS
            cblas_daxpy(x->size, a, xdata, 1, ydata, 1);
#else
            for (int64_t k = 0; k < x->size; k++)
                ydata[k] += a*xdata[k];
#endif
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else {
        return MTX_ERR_INVALID_MTX_FIELD;
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_vector_array_sdot()' computes the Euclidean dot product of two
 * vectors (or Frobenius inner product of two matrices) of single
 * precision floating-point values.
 */
int mtx_vector_array_sdot(
    const struct mtx_vector_array_data * x,
    const struct mtx_vector_array_data * y,
    float * dot)
{
    if (x->size != y->size)
        return MTX_ERR_INVALID_MTX_SIZE;

    if (x->field == mtx_real && y->field == mtx_real) {
        if (x->precision == mtx_single && y->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            const float * ydata = y->data.real_single;
#ifdef LIBMTX_HAVE_BLAS
            *dot = cblas_sdot(x->size, xdata, 1, ydata, 1);
#else
            *dot = 0;
            for (int64_t k = 0; k < x->size; k++)
                *dot += xdata[k]*ydata[k];
#endif
        } else if (x->precision == mtx_double && y->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            const double * ydata = y->data.real_double;
#ifdef LIBMTX_HAVE_BLAS
            *dot = cblas_ddot(x->size, xdata, 1, ydata, 1);
#else
            *dot = 0;
            for (int64_t k = 0; k < x->size; k++)
                *dot += xdata[k]*ydata[k];
#endif
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else {
        return MTX_ERR_INVALID_MTX_FIELD;
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_vector_array_ddot()' computes the Euclidean dot product of two
 * vectors (or Frobenius inner product of two matrices) of double
 * precision floating-point values.
 */
int mtx_vector_array_ddot(
    const struct mtx_vector_array_data * x,
    const struct mtx_vector_array_data * y,
    double * dot)
{
    if (x->size != y->size)
        return MTX_ERR_INVALID_MTX_SIZE;

    if (x->field == mtx_real && y->field == mtx_real) {
        if (x->precision == mtx_single && y->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            const float * ydata = y->data.real_single;
#ifdef LIBMTX_HAVE_BLAS
            *dot = cblas_sdot(x->size, xdata, 1, ydata, 1);
#else
            *dot = 0;
            for (int64_t k = 0; k < x->size; k++)
                *dot += xdata[k]*ydata[k];
#endif
        } else if (x->precision == mtx_double && y->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            const double * ydata = y->data.real_double;
#ifdef LIBMTX_HAVE_BLAS
            *dot = cblas_ddot(x->size, xdata, 1, ydata, 1);
#else
            *dot = 0;
            for (int64_t k = 0; k < x->size; k++)
                *dot += xdata[k]*ydata[k];
#endif
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else {
        return MTX_ERR_INVALID_MTX_FIELD;
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_vector_array_snrm2()' computes the Euclidean norm of a vector
 * (or Frobenius norm of a matrix) of single precision floating-point
 * values.
 */
int mtx_vector_array_snrm2(
    const struct mtx_vector_array_data * x,
    float * nrm2)
{
    if (x->field == mtx_real) {
        if (x->precision == mtx_single) {
            const float * xdata = x->data.real_single;
#ifdef LIBMTX_HAVE_BLAS
            *nrm2 = cblas_snrm2(x->size, xdata, 1);
#else
            *nrm2 = 0;
            for (int64_t k = 0; k < x->size; k++)
                *nrm2 += xdata[k]*xdata[k];
            *nrm2 = sqrtf(*nrm2);
#endif
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
#ifdef LIBMTX_HAVE_BLAS
            *nrm2 = cblas_dnrm2(x->size, xdata, 1);
#else
            *nrm2 = 0;
            for (int64_t k = 0; k < x->size; k++)
                *nrm2 += xdata[k]*xdata[k];
            *nrm2 = sqrtf(*nrm2);
#endif
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else {
        return MTX_ERR_INVALID_MTX_FIELD;
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_vector_array_dnrm2()' computes the Euclidean norm of a vector
 * (or Frobenius norm of a matrix) of double precision floating-point
 * values.
 */
int mtx_vector_array_dnrm2(
    const struct mtx_vector_array_data * x,
    double * nrm2)
{
    if (x->field == mtx_real) {
        if (x->precision == mtx_single) {
            const float * xdata = x->data.real_single;
#ifdef LIBMTX_HAVE_BLAS
            *nrm2 = cblas_snrm2(x->size, xdata, 1);
#else
            *nrm2 = 0;
            for (int64_t k = 0; k < x->size; k++)
                *nrm2 += xdata[k]*xdata[k];
            *nrm2 = sqrt(*nrm2);
#endif
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
#ifdef LIBMTX_HAVE_BLAS
            *nrm2 = cblas_dnrm2(x->size, xdata, 1);
#else
            *nrm2 = 0;
            for (int64_t k = 0; k < x->size; k++)
                *nrm2 += xdata[k]*xdata[k];
            *nrm2 = sqrt(*nrm2);
#endif
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else {
        return MTX_ERR_INVALID_MTX_FIELD;
    }
    return MTX_SUCCESS;
}
