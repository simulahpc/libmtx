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
 * Level 1 BLAS operations for sparse vectors in coordinate format.
 */

#include <libmtx/error.h>
#include <libmtx/vector/coordinate.h>
#include <libmtx/vector/coordinate/blas.h>
#include <libmtx/vector/coordinate/data.h>

#include <errno.h>

#include <math.h>

/**
 * `mtx_vector_coordinate_sscal()' scales a vector by a single
 * precision floating-point scalar, `x = a*x'.
 */
int mtx_vector_coordinate_sscal(
    float a,
    struct mtx_vector_coordinate_data * x)
{
    if (x->field == mtx_real) {
        if (x->precision == mtx_single) {
            struct mtx_vector_coordinate_real_single * xdata = x->data.real_single;
            for (int64_t k = 0; k < x->size; k++)
                xdata[k].a *= a;
        } else if (x->precision == mtx_double) {
            struct mtx_vector_coordinate_real_double * xdata = x->data.real_double;
            for (int64_t k = 0; k < x->size; k++)
                xdata[k].a *= a;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_complex) {
        if (x->precision == mtx_single) {
            struct mtx_vector_coordinate_complex_single * xdata =
                x->data.complex_single;
            for (int64_t k = 0; k < x->size; k++) {
                xdata[k].a[0] *= a;
                xdata[k].a[1] *= a;
            }
        } else if (x->precision == mtx_double) {
            struct mtx_vector_coordinate_complex_double * xdata =
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
 * `mtx_vector_coordinate_dscal()' scales a vector by a double
 * precision floating-point scalar, `x = a*x'.
 */
int mtx_vector_coordinate_dscal(
    double a,
    struct mtx_vector_coordinate_data * x)
{
    if (x->field == mtx_real) {
        if (x->precision == mtx_single) {
            struct mtx_vector_coordinate_real_single * xdata =
                x->data.real_single;
            for (int64_t k = 0; k < x->size; k++)
                xdata[k].a *= a;
        } else if (x->precision == mtx_double) {
            struct mtx_vector_coordinate_real_double * xdata =
                x->data.real_double;
            for (int64_t k = 0; k < x->size; k++)
                xdata[k].a *= a;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_complex) {
        if (x->precision == mtx_single) {
            struct mtx_vector_coordinate_complex_single * xdata =
                x->data.complex_single;
            for (int64_t k = 0; k < x->size; k++) {
                xdata[k].a[0] *= a;
                xdata[k].a[1] *= a;
            }
        } else if (x->precision == mtx_double) {
            struct mtx_vector_coordinate_complex_double * xdata =
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
 * `mtx_vector_coordinate_saxpy()' adds two vectors of single
 * precision floating-point values, `y = a*x + y'.
 */
int mtx_vector_coordinate_saxpy(
    float a,
    const struct mtx_vector_coordinate_data * x,
    struct mtx_vector_coordinate_data * y)
{
    /* TODO: Implement vector addition for sparse vectors. */
    errno = ENOTSUP;
    return MTX_ERR_ERRNO;
}

/**
 * `mtx_vector_coordinate_daxpy()' adds two vectors of double
 * precision floating-point values, `y = a*x + y'.
 */
int mtx_vector_coordinate_daxpy(
    double a,
    const struct mtx_vector_coordinate_data * x,
    struct mtx_vector_coordinate_data * y)
{
    /* TODO: Implement vector addition for sparse vectors. */
    errno = ENOTSUP;
    return MTX_ERR_ERRNO;
}

/**
 * `mtx_vector_coordinate_sdot()' computes the Euclidean dot product
 * of two vectors of single precision floating-point values.
 */
int mtx_vector_coordinate_sdot(
    const struct mtx_vector_coordinate_data * x,
    const struct mtx_vector_coordinate_data * y,
    float * dot)
{
    /* TODO: Implement dot product for sparse vectors. */
    errno = ENOTSUP;
    return MTX_ERR_ERRNO;
}

/**
 * `mtx_vector_coordinate_ddot()' computes the Euclidean dot product
 * of two vectors of double precision floating-point values.
 */
int mtx_vector_coordinate_ddot(
    const struct mtx_vector_coordinate_data * x,
    const struct mtx_vector_coordinate_data * y,
    double * dot)
{
    /* TODO: Implement dot product for sparse vectors. */
    errno = ENOTSUP;
    return MTX_ERR_ERRNO;
}

/**
 * `mtx_vector_coordinate_snrm2()' computes the Euclidean norm of a
 * vector of single precision floating-point values.
 */
int mtx_vector_coordinate_snrm2(
    const struct mtx_vector_coordinate_data * x,
    float * nrm2)
{
    if (x->field == mtx_real) {
        if (x->precision == mtx_single) {
            struct mtx_vector_coordinate_real_single * xdata =
                x->data.real_single;
            *nrm2 = 0;
            for (int64_t k = 0; k < x->size; k++)
                *nrm2 += xdata[k].a*xdata[k].a;
        } else if (x->precision == mtx_double) {
            struct mtx_vector_coordinate_real_double * xdata =
                x->data.real_double;
            *nrm2 = 0;
            for (int64_t k = 0; k < x->size; k++)
                *nrm2 += xdata[k].a*xdata[k].a;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_complex) {
        if (x->precision == mtx_single) {
            struct mtx_vector_coordinate_complex_single * xdata =
                x->data.complex_single;
            *nrm2 = 0;
            for (int64_t k = 0; k < x->size; k++)
                *nrm2 += xdata[k].a[0]*xdata[k].a[0] + xdata[k].a[1]*xdata[k].a[1];
        } else if (x->precision == mtx_double) {
            struct mtx_vector_coordinate_complex_double * xdata =
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
 * `mtx_vector_coordinate_dnrm2()' computes the Euclidean norm of a
 * vector of double precision floating-point values.
 */
int mtx_vector_coordinate_dnrm2(
    const struct mtx_vector_coordinate_data * x,
    double * nrm2)
{
    if (x->field == mtx_real) {
        if (x->precision == mtx_single) {
            struct mtx_vector_coordinate_real_single * xdata =
                x->data.real_single;
            *nrm2 = 0;
            for (int64_t k = 0; k < x->size; k++)
                *nrm2 += xdata[k].a*xdata[k].a;
        } else if (x->precision == mtx_double) {
            struct mtx_vector_coordinate_real_double * xdata =
                x->data.real_double;
            *nrm2 = 0;
            for (int64_t k = 0; k < x->size; k++)
                *nrm2 += xdata[k].a*xdata[k].a;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_complex) {
        if (x->precision == mtx_single) {
            struct mtx_vector_coordinate_complex_single * xdata =
                x->data.complex_single;
            *nrm2 = 0;
            for (int64_t k = 0; k < x->size; k++)
                *nrm2 += xdata[k].a[0]*xdata[k].a[0] + xdata[k].a[1]*xdata[k].a[1];
        } else if (x->precision == mtx_double) {
            struct mtx_vector_coordinate_complex_double * xdata =
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
