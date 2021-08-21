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

#include <libmtx/error.h>
#include <libmtx/matrix/coordinate.h>
#include <libmtx/matrix/coordinate/blas.h>
#include <libmtx/matrix/coordinate/data.h>
#include <libmtx/vector/array/data.h>

#include <errno.h>

#include <math.h>

/*
 * Level 1 BLAS operations.
 */

/**
 * `mtx_matrix_coordinate_copy()' copies values of a matrix, `y = x'.
 */
int mtx_matrix_coordinate_copy(
    struct mtx_matrix_coordinate_data * y,
    const struct mtx_matrix_coordinate_data * x)
{
    /* TODO: Implement copying of matrices in coordinate format. */
    errno = ENOTSUP;
    return MTX_ERR_ERRNO;
}

/**
 * `mtx_matrix_coordinate_sscal()' scales a matrix by a single
 * precision floating-point scalar, `x = a*x'.
 */
int mtx_matrix_coordinate_sscal(
    float a,
    struct mtx_matrix_coordinate_data * x)
{
    if (x->field == mtx_real) {
        if (x->precision == mtx_single) {
            struct mtx_matrix_coordinate_real_single * xdata =
                x->data.real_single;
            for (int64_t k = 0; k < x->size; k++)
                xdata[k].a *= a;
        } else if (x->precision == mtx_double) {
            struct mtx_matrix_coordinate_real_double * xdata =
                x->data.real_double;
            for (int64_t k = 0; k < x->size; k++)
                xdata[k].a *= a;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_complex) {
        if (x->precision == mtx_single) {
            struct mtx_matrix_coordinate_complex_single * xdata =
                x->data.complex_single;
            for (int64_t k = 0; k < x->size; k++) {
                xdata[k].a[0] *= a;
                xdata[k].a[1] *= a;
            }
        } else if (x->precision == mtx_double) {
            struct mtx_matrix_coordinate_complex_double * xdata =
                x->data.complex_double;
            for (int64_t k = 0; k < x->size; k++) {
                xdata[k].a[0] *= a;
                xdata[k].a[1] *= a;
            }
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else {
        return MTX_ERR_INVALID_MTX_FIELD;
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_matrix_coordinate_dscal()' scales a matrix by a double
 * precision floating-point scalar, `x = a*x'.
 */
int mtx_matrix_coordinate_dscal(
    double a,
    struct mtx_matrix_coordinate_data * x)
{
    if (x->field == mtx_real) {
        if (x->precision == mtx_single) {
            struct mtx_matrix_coordinate_real_single * xdata =
                x->data.real_single;
            for (int64_t k = 0; k < x->size; k++)
                xdata[k].a *= a;
        } else if (x->precision == mtx_double) {
            struct mtx_matrix_coordinate_real_double * xdata =
                x->data.real_double;
            for (int64_t k = 0; k < x->size; k++)
                xdata[k].a *= a;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_complex) {
        if (x->precision == mtx_single) {
            struct mtx_matrix_coordinate_complex_single * xdata =
                x->data.complex_single;
            for (int64_t k = 0; k < x->size; k++) {
                xdata[k].a[0] *= a;
                xdata[k].a[1] *= a;
            }
        } else if (x->precision == mtx_double) {
            struct mtx_matrix_coordinate_complex_double * xdata =
                x->data.complex_double;
            for (int64_t k = 0; k < x->size; k++) {
                xdata[k].a[0] *= a;
                xdata[k].a[1] *= a;
            }
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else {
        return MTX_ERR_INVALID_MTX_FIELD;
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_matrix_coordinate_saxpy()' adds two matrices of single
 * precision floating-point values, `y = a*x + y'.
 */
int mtx_matrix_coordinate_saxpy(
    float a,
    const struct mtx_matrix_coordinate_data * x,
    struct mtx_matrix_coordinate_data * y)
{
    /* TODO: Implement addition of matrices in coordinate format. */
    errno = ENOTSUP;
    return MTX_ERR_ERRNO;
}

/**
 * `mtx_matrix_coordinate_daxpy()' adds two matrices of double
 * precision floating-point values, `y = a*x + y'.
 */
int mtx_matrix_coordinate_daxpy(
    double a,
    const struct mtx_matrix_coordinate_data * x,
    struct mtx_matrix_coordinate_data * y)
{
    /* TODO: Implement addition of matrices in coordinate format. */
    errno = ENOTSUP;
    return MTX_ERR_ERRNO;
}

/**
 * `mtx_matrix_coordinate_sdot()' computes the Frobenius inner product
 * of two matrices in single precision floating point.
 */
int mtx_matrix_coordinate_sdot(
    const struct mtx_matrix_coordinate_data * x,
    const struct mtx_matrix_coordinate_data * y,
    float * dot)
{
    /* TODO: Implement dot product for matrices in coordinate format. */
    errno = ENOTSUP;
    return MTX_ERR_ERRNO;
}

/**
 * `mtx_matrix_coordinate_ddot()' computes the Frobenius inner product
 * of two matrices in double precision floating point.
 */
int mtx_matrix_coordinate_ddot(
    const struct mtx_matrix_coordinate_data * x,
    const struct mtx_matrix_coordinate_data * y,
    double * dot)
{
    /* TODO: Implement dot product for matrices in coordinate format. */
    errno = ENOTSUP;
    return MTX_ERR_ERRNO;
}

/**
 * `mtx_matrix_coordinate_snrm2()' computes the Frobenius norm of a
 * matrix in single precision floating point.
 */
int mtx_matrix_coordinate_snrm2(
    const struct mtx_matrix_coordinate_data * x,
    float * nrm2)
{
    if (x->symmetry != mtx_general)
        return MTX_ERR_INVALID_MTX_SYMMETRY;

    if (x->field == mtx_real) {
        if (x->precision == mtx_single) {
            struct mtx_matrix_coordinate_real_single * xdata =
                x->data.real_single;
            *nrm2 = 0;
            for (int64_t k = 0; k < x->size; k++)
                *nrm2 += xdata[k].a*xdata[k].a;
        } else if (x->precision == mtx_double) {
            struct mtx_matrix_coordinate_real_double * xdata =
                x->data.real_double;
            *nrm2 = 0;
            for (int64_t k = 0; k < x->size; k++)
                *nrm2 += xdata[k].a*xdata[k].a;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_complex) {
        if (x->precision == mtx_single) {
            struct mtx_matrix_coordinate_complex_single * xdata =
                x->data.complex_single;
            *nrm2 = 0;
            for (int64_t k = 0; k < x->size; k++)
                *nrm2 += xdata[k].a[0]*xdata[k].a[0] + xdata[k].a[1]*xdata[k].a[1];
        } else if (x->precision == mtx_double) {
            struct mtx_matrix_coordinate_complex_double * xdata =
                x->data.complex_double;
            *nrm2 = 0;
            for (int64_t k = 0; k < x->size; k++)
                *nrm2 += xdata[k].a[0]*xdata[k].a[0] + xdata[k].a[1]*xdata[k].a[1];
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else {
        return MTX_ERR_INVALID_MTX_FIELD;
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_matrix_coordinate_dnrm2()' computes the Frobenius norm of a
 * matrix in double precision floating point.
 */
int mtx_matrix_coordinate_dnrm2(
    const struct mtx_matrix_coordinate_data * x,
    double * nrm2)
{
    if (x->symmetry != mtx_general)
        return MTX_ERR_INVALID_MTX_SYMMETRY;

    if (x->field == mtx_real) {
        if (x->precision == mtx_single) {
            struct mtx_matrix_coordinate_real_single * xdata =
                x->data.real_single;
            *nrm2 = 0;
            for (int64_t k = 0; k < x->size; k++)
                *nrm2 += xdata[k].a*xdata[k].a;
        } else if (x->precision == mtx_double) {
            struct mtx_matrix_coordinate_real_double * xdata =
                x->data.real_double;
            *nrm2 = 0;
            for (int64_t k = 0; k < x->size; k++)
                *nrm2 += xdata[k].a*xdata[k].a;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_complex) {
        if (x->precision == mtx_single) {
            struct mtx_matrix_coordinate_complex_single * xdata =
                x->data.complex_single;
            *nrm2 = 0;
            for (int64_t k = 0; k < x->size; k++)
                *nrm2 += xdata[k].a[0]*xdata[k].a[0] + xdata[k].a[1]*xdata[k].a[1];
        } else if (x->precision == mtx_double) {
            struct mtx_matrix_coordinate_complex_double * xdata =
                x->data.complex_double;
            *nrm2 = 0;
            for (int64_t k = 0; k < x->size; k++)
                *nrm2 += xdata[k].a[0]*xdata[k].a[0] + xdata[k].a[1]*xdata[k].a[1];
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else {
        return MTX_ERR_INVALID_MTX_FIELD;
    }
    return MTX_SUCCESS;
}

/*
 * Level 2 BLAS operations.
 */

/**
 * `mtx_matrix_coordinate_sgemv()' computes the product of a matrix
 * and a vector of single precision floating-point values, `y =
 * alpha*A*x + beta*y'.
 */
int mtx_matrix_coordinate_sgemv(
    float alpha,
    const struct mtx_matrix_coordinate_data * A,
    const struct mtx_vector_array_data * x,
    float beta,
    struct mtx_vector_array_data * y)
{
    if (A->symmetry != mtx_general)
        return MTX_ERR_INVALID_MTX_SYMMETRY;
    if (A->num_rows != y->size ||
        A->num_columns != x->size)
        return MTX_ERR_INVALID_MTX_SIZE;

    /* TODO: The naive algorithm below only works if `beta' is
     * equal to one. Otherwise, some intermediate storage will be
     * needed for the values of the matrix-vector product, and a
     * final vector addition must be performed. */
    if (beta != 1.0) {
        errno = ENOTSUP;
        return MTX_ERR_ERRNO;
    }

    if (A->field == mtx_real &&
        x->field == mtx_real &&
        y->field == mtx_real)
    {
        if (A->precision == mtx_single &&
            x->precision == mtx_single &&
            y->precision == mtx_single)
        {
            const struct mtx_matrix_coordinate_real_single * Adata =
                A->data.real_single;
            const float * xdata = x->data.real_single;
            float * ydata = y->data.real_single;
            for (int64_t k = 0; k < A->size; k++)
                ydata[Adata[k].i-1] += alpha*Adata[k].a*xdata[Adata[k].j-1];
        } else if (A->precision == mtx_double &&
                   x->precision == mtx_double &&
                   y->precision == mtx_double)
        {
            const struct mtx_matrix_coordinate_real_double * Adata =
                A->data.real_double;
            const double * xdata = x->data.real_double;
            double * ydata = y->data.real_double;
            for (int64_t k = 0; k < A->size; k++)
                ydata[Adata[k].i-1] += alpha*Adata[k].a*xdata[Adata[k].j-1];
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else {
        return MTX_ERR_INVALID_MTX_FIELD;
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_matrix_coordinate_dgemv()' computes the product of a matrix
 * and a vector of single precision floating-point values, `y =
 * alpha*A*x + beta*y'.
 */
int mtx_matrix_coordinate_dgemv(
    double alpha,
    const struct mtx_matrix_coordinate_data * A,
    const struct mtx_vector_array_data * x,
    double beta,
    struct mtx_vector_array_data * y)
{
    if (A->symmetry != mtx_general)
        return MTX_ERR_INVALID_MTX_SYMMETRY;
    if (A->num_rows != y->size ||
        A->num_columns != x->size)
        return MTX_ERR_INVALID_MTX_SIZE;

    /* TODO: The naive algorithm below only works if `beta' is
     * equal to one. Otherwise, some intermediate storage will be
     * needed for the values of the matrix-vector product, and a
     * final vector addition must be performed. */
    if (beta != 1.0) {
        errno = ENOTSUP;
        return MTX_ERR_ERRNO;
    }

    if (A->field == mtx_real &&
        x->field == mtx_real &&
        y->field == mtx_real)
    {
        if (A->precision == mtx_single &&
            x->precision == mtx_single &&
            y->precision == mtx_single)
        {
            const struct mtx_matrix_coordinate_real_single * Adata =
                A->data.real_single;
            const float * xdata = x->data.real_single;
            float * ydata = y->data.real_single;
            for (int64_t k = 0; k < A->size; k++)
                ydata[Adata[k].i-1] += alpha*Adata[k].a*xdata[Adata[k].j-1];
        } else if (A->precision == mtx_double &&
                   x->precision == mtx_double &&
                   y->precision == mtx_double)
        {
            const struct mtx_matrix_coordinate_real_double * Adata =
                A->data.real_double;
            const double * xdata = x->data.real_double;
            double * ydata = y->data.real_double;
            for (int64_t k = 0; k < A->size; k++)
                ydata[Adata[k].i-1] += alpha*Adata[k].a*xdata[Adata[k].j-1];
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else {
        return MTX_ERR_INVALID_MTX_FIELD;
    }
    return MTX_SUCCESS;
}
