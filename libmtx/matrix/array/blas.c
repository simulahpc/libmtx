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
 * Last modified: 2021-08-09
 *
 * BLAS operations for matrices in array format.
 */

#include <libmtx/libmtx-config.h>

#include <libmtx/error.h>
#include <libmtx/matrix/array.h>
#include <libmtx/matrix/array/blas.h>
#include <libmtx/matrix/array/data.h>
#include <libmtx/vector/array/data.h>

#ifdef LIBMTX_HAVE_BLAS
#include <cblas.h>
#endif

#include <errno.h>

#include <math.h>

/*
 * Level 1 BLAS operations.
 */

/**
 * `mtx_matrix_array_copy()' copies values of a matrix, `y = x'.
 */
int mtx_matrix_array_copy(
    struct mtx_matrix_array_data * y,
    const struct mtx_matrix_array_data * x)
{
    if (x->symmetry != y->symmetry)
        return MTX_ERR_INVALID_MTX_SYMMETRY;
    if (x->triangle != y->triangle)
        return MTX_ERR_INVALID_MTX_TRIANGLE;
    if (x->sorting != y->sorting)
        return MTX_ERR_INVALID_MTX_SORTING;
    if (x->num_rows != y->num_rows ||
        x->num_columns != y->num_columns ||
        x->size != y->size)
        return MTX_ERR_INVALID_MTX_SIZE;

    if (x->field == mtx_real && y->field == mtx_real) {
        if (x->precision == mtx_single && y->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            float * ydata = y->data.real_single;
#ifdef LIBMTX_HAVE_BLAS
            cblas_scopy(x->size, xdata, 1, ydata, 1);
#else
            for (int64_t k = 0; k < x->size; k++)
                ydata[k] = xdata[k];
#endif
        } else if (x->precision == mtx_double && y->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            double * ydata = y->data.real_double;
#ifdef LIBMTX_HAVE_BLAS
            cblas_dcopy(x->size, xdata, 1, ydata, 1);
#else
            for (int64_t k = 0; k < x->size; k++)
                ydata[k] = xdata[k];
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
 * `mtx_matrix_array_sscal()' scales a matrix by a single precision
 * floating-point scalar, `x = a*x'.
 */
int mtx_matrix_array_sscal(
    float a,
    struct mtx_matrix_array_data * x)
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
 * `mtx_matrix_array_dscal()' scales a matrix by a double precision
 * floating-point scalar, `x = a*x'.
 */
int mtx_matrix_array_dscal(
    double a,
    struct mtx_matrix_array_data * x)
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
 * `mtx_matrix_array_saxpy()' adds two matrices of single precision
 * floating-point values, `y = a*x + y'.
 */
int mtx_matrix_array_saxpy(
    float a,
    const struct mtx_matrix_array_data * x,
    struct mtx_matrix_array_data * y)
{
    if (x->symmetry != y->symmetry)
        return MTX_ERR_INVALID_MTX_SYMMETRY;
    if (x->triangle != y->triangle)
        return MTX_ERR_INVALID_MTX_TRIANGLE;
    if (x->sorting != y->sorting)
        return MTX_ERR_INVALID_MTX_SORTING;
    if (x->num_rows != y->num_rows ||
        x->num_columns != y->num_columns ||
        x->size != y->size)
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
 * `mtx_matrix_array_daxpy()' adds two matrices of double precision
 * floating-point values, `y = a*x + y'.
 */
int mtx_matrix_array_daxpy(
    double a,
    const struct mtx_matrix_array_data * x,
    struct mtx_matrix_array_data * y)
{
    if (x->symmetry != y->symmetry)
        return MTX_ERR_INVALID_MTX_SYMMETRY;
    if (x->triangle != y->triangle)
        return MTX_ERR_INVALID_MTX_TRIANGLE;
    if (x->sorting != y->sorting)
        return MTX_ERR_INVALID_MTX_SORTING;
    if (x->num_rows != y->num_rows ||
        x->num_columns != y->num_columns ||
        x->size != y->size)
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
 * `mtx_matrix_array_saypx()' adds two matrices of single precision
 * floating-point values, `y = a*x + y'.
 */
int mtx_matrix_array_saypx(
    float a,
    struct mtx_matrix_array_data * y,
    const struct mtx_matrix_array_data * x)
{
    if (x->symmetry != y->symmetry)
        return MTX_ERR_INVALID_MTX_SYMMETRY;
    if (x->triangle != y->triangle)
        return MTX_ERR_INVALID_MTX_TRIANGLE;
    if (x->sorting != y->sorting)
        return MTX_ERR_INVALID_MTX_SORTING;
    if (x->num_rows != y->num_rows ||
        x->num_columns != y->num_columns ||
        x->size != y->size)
        return MTX_ERR_INVALID_MTX_SIZE;

    if (x->field == mtx_real && y->field == mtx_real) {
        if (x->precision == mtx_single && y->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            float * ydata = y->data.real_single;
            for (int64_t k = 0; k < x->size; k++)
                ydata[k] = a*ydata[k]+xdata[k];
        } else if (x->precision == mtx_double && y->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            double * ydata = y->data.real_double;
            for (int64_t k = 0; k < x->size; k++)
                ydata[k] = a*ydata[k]+xdata[k];
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else {
        return MTX_ERR_INVALID_MTX_FIELD;
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_matrix_array_daypx()' adds two matrices of double precision
 * floating-point values, `y = a*y + x'.
 */
int mtx_matrix_array_daypx(
    double a,
    struct mtx_matrix_array_data * y,
    const struct mtx_matrix_array_data * x)
{
    if (x->symmetry != y->symmetry)
        return MTX_ERR_INVALID_MTX_SYMMETRY;
    if (x->triangle != y->triangle)
        return MTX_ERR_INVALID_MTX_TRIANGLE;
    if (x->sorting != y->sorting)
        return MTX_ERR_INVALID_MTX_SORTING;
    if (x->num_rows != y->num_rows ||
        x->num_columns != y->num_columns ||
        x->size != y->size)
        return MTX_ERR_INVALID_MTX_SIZE;

    if (x->field == mtx_real && y->field == mtx_real) {
        if (x->precision == mtx_single && y->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            float * ydata = y->data.real_single;
            for (int64_t k = 0; k < x->size; k++)
                ydata[k] = a*ydata[k]+xdata[k];
        } else if (x->precision == mtx_double && y->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            double * ydata = y->data.real_double;
            for (int64_t k = 0; k < x->size; k++)
                ydata[k] = a*ydata[k]+xdata[k];
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else {
        return MTX_ERR_INVALID_MTX_FIELD;
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_matrix_array_sdot()' computes the Frobenius inner product of
 * two matrices in single precision floating point.
 */
int mtx_matrix_array_sdot(
    const struct mtx_matrix_array_data * x,
    const struct mtx_matrix_array_data * y,
    float * dot)
{
    if (x->symmetry != y->symmetry)
        return MTX_ERR_INVALID_MTX_SYMMETRY;
    if (x->triangle != y->triangle)
        return MTX_ERR_INVALID_MTX_TRIANGLE;
    if (x->sorting != y->sorting)
        return MTX_ERR_INVALID_MTX_SORTING;
    if (x->num_rows != y->num_rows ||
        x->num_columns != y->num_columns ||
        x->size != y->size)
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
 * `mtx_matrix_array_ddot()' computes the Frobenius inner product of
 * two matrices in double precision floating point.
 */
int mtx_matrix_array_ddot(
    const struct mtx_matrix_array_data * x,
    const struct mtx_matrix_array_data * y,
    double * dot)
{
    if (x->symmetry != y->symmetry)
        return MTX_ERR_INVALID_MTX_SYMMETRY;
    if (x->triangle != y->triangle)
        return MTX_ERR_INVALID_MTX_TRIANGLE;
    if (x->sorting != y->sorting)
        return MTX_ERR_INVALID_MTX_SORTING;
    if (x->num_rows != y->num_rows ||
        x->num_columns != y->num_columns ||
        x->size != y->size)
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
 * `mtx_matrix_array_snrm2()' computes the Frobenius norm of a matrix
 * in single precision floating point.
 */
int mtx_matrix_array_snrm2(
    const struct mtx_matrix_array_data * x,
    float * nrm2)
{
    if (x->symmetry != mtx_general)
        return MTX_ERR_INVALID_MTX_SYMMETRY;

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
 * `mtx_matrix_array_dnrm2()' computes the Frobenius norm of a matrix
 * in double precision floating point.
 */
int mtx_matrix_array_dnrm2(
    const struct mtx_matrix_array_data * x,
    double * nrm2)
{
    if (x->symmetry != mtx_general)
        return MTX_ERR_INVALID_MTX_SYMMETRY;

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
    const struct mtx_matrix_array_data * A,
    const struct mtx_vector_array_data * x,
    float beta,
    struct mtx_vector_array_data * y)
{
    if (A->symmetry != mtx_general)
        return MTX_ERR_INVALID_MTX_SYMMETRY;
    if (A->triangle != mtx_nontriangular)
        return MTX_ERR_INVALID_MTX_TRIANGLE;
    if (A->sorting != mtx_row_major)
        return MTX_ERR_INVALID_MTX_SORTING;
    if (A->num_rows != y->size ||
        A->num_columns != x->size)
        return MTX_ERR_INVALID_MTX_SIZE;

    if (A->field == mtx_real &&
        x->field == mtx_real &&
        y->field == mtx_real)
    {
        if (A->precision == mtx_single &&
            x->precision == mtx_single &&
            y->precision == mtx_single)
        {
            const float * Adata = A->data.real_single;
            const float * xdata = x->data.real_single;
            float * ydata = y->data.real_single;
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
        } else if (A->precision == mtx_double &&
            x->precision == mtx_double &&
            y->precision == mtx_double)
        {
            const double * Adata = A->data.real_double;
            const double * xdata = x->data.real_double;
            double * ydata = y->data.real_double;
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
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else {
        return MTX_ERR_INVALID_MTX_FIELD;
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_matrix_array_dgemv()' computes the product of a matrix and a
 * vector of single precision floating-point values, `y = alpha*A*x +
 * beta*y'.
 */
int mtx_matrix_array_dgemv(
    double alpha,
    const struct mtx_matrix_array_data * A,
    const struct mtx_vector_array_data * x,
    double beta,
    struct mtx_vector_array_data * y)
{
    if (A->symmetry != mtx_general)
        return MTX_ERR_INVALID_MTX_SYMMETRY;
    if (A->triangle != mtx_nontriangular)
        return MTX_ERR_INVALID_MTX_TRIANGLE;
    if (A->sorting != mtx_row_major)
        return MTX_ERR_INVALID_MTX_SORTING;
    if (A->num_rows != y->size ||
        A->num_columns != x->size)
        return MTX_ERR_INVALID_MTX_SIZE;

    if (A->field == mtx_real &&
        x->field == mtx_real &&
        y->field == mtx_real)
    {
        if (A->precision == mtx_single &&
            x->precision == mtx_single &&
            y->precision == mtx_single)
        {
            const float * Adata = A->data.real_single;
            const float * xdata = x->data.real_single;
            float * ydata = y->data.real_single;
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
        } else if (A->precision == mtx_double &&
            x->precision == mtx_double &&
            y->precision == mtx_double)
        {
            const double * Adata = A->data.real_double;
            const double * xdata = x->data.real_double;
            double * ydata = y->data.real_double;
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
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else {
        return MTX_ERR_INVALID_MTX_FIELD;
    }
    return MTX_SUCCESS;
}
