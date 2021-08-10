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
#include <libmtx/mtx/blas.h>
#include <libmtx/matrix/array/blas.h>
#include <libmtx/matrix/coordinate/coordinate.h>
#include <libmtx/matrix/coordinate/blas.h>
#include <libmtx/mtx/mtx.h>
#include <libmtx/vector/array/array.h>
#include <libmtx/vector/array/blas.h>
#include <libmtx/vector/coordinate/blas.h>
#include <libmtx/vector/coordinate/coordinate.h>

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
    if (x->object == mtx_matrix) {
        if (x->format == mtx_array) {
            return mtx_matrix_array_sscal(a, x);
        } else if (x->format == mtx_coordinate) {
            return mtx_matrix_coordinate_sscal(a, x);
        } else {
            return MTX_ERR_INVALID_MTX_FORMAT;
        }
    } else if (x->object == mtx_vector) {
        if (x->format == mtx_array) {
            return mtx_vector_array_sscal(a, x);
        } else if (x->format == mtx_coordinate) {
            return mtx_vector_coordinate_sscal(a, x);
        } else {
            return MTX_ERR_INVALID_MTX_FORMAT;
        }
    } else {
        return MTX_ERR_INVALID_MTX_OBJECT;
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
    if (x->object == mtx_matrix) {
        if (x->format == mtx_array) {
            return mtx_matrix_array_dscal(a, x);
        } else if (x->format == mtx_coordinate) {
            return mtx_matrix_coordinate_dscal(a, x);
        } else {
            return MTX_ERR_INVALID_MTX_FORMAT;
        }
    } else if (x->object == mtx_vector) {
        if (x->format == mtx_array) {
            return mtx_vector_array_dscal(a, x);
        } else if (x->format == mtx_coordinate) {
            return mtx_vector_coordinate_dscal(a, x);
        } else {
            return MTX_ERR_INVALID_MTX_FORMAT;
        }
    } else {
        return MTX_ERR_INVALID_MTX_OBJECT;
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
    if (x->object == mtx_matrix && y->object == mtx_matrix) {
        if (x->format == mtx_array && y->format == mtx_array) {
            return mtx_matrix_array_saxpy(a, x, y);
        } else if (x->format == mtx_coordinate && y->format == mtx_coordinate) {
            return mtx_matrix_coordinate_saxpy(a, x, y);
        } else {
            return MTX_ERR_INVALID_MTX_FORMAT;
        }
    } else if (x->object == mtx_vector && y->object == mtx_vector) {
        if (x->format == mtx_array && y->format == mtx_array) {
            return mtx_vector_array_saxpy(a, x, y);
        } else if (x->format == mtx_coordinate && y->format == mtx_coordinate) {
            return mtx_vector_coordinate_saxpy(a, x, y);
        } else {
            return MTX_ERR_INVALID_MTX_FORMAT;
        }
    } else {
        return MTX_ERR_INVALID_MTX_OBJECT;
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
    if (x->object == mtx_matrix && y->object == mtx_matrix) {
        if (x->format == mtx_array && y->format == mtx_array) {
            return mtx_matrix_array_daxpy(a, x, y);
        } else if (x->format == mtx_coordinate && y->format == mtx_coordinate) {
            return mtx_matrix_coordinate_daxpy(a, x, y);
        } else {
            return MTX_ERR_INVALID_MTX_FORMAT;
        }
    } else if (x->object == mtx_vector && y->object == mtx_vector) {
        if (x->format == mtx_array && y->format == mtx_array) {
            return mtx_vector_array_daxpy(a, x, y);
        } else if (x->format == mtx_coordinate && y->format == mtx_coordinate) {
            return mtx_vector_coordinate_daxpy(a, x, y);
        } else {
            return MTX_ERR_INVALID_MTX_FORMAT;
        }
    } else {
        return MTX_ERR_INVALID_MTX_OBJECT;
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
    if (x->object == mtx_matrix && y->object == mtx_matrix) {
        if (x->format == mtx_array && y->format == mtx_array) {
            return mtx_matrix_array_sdot(x, y, dot);
        } else if (x->format == mtx_coordinate && y->format == mtx_coordinate) {
            return mtx_matrix_coordinate_sdot(x, y, dot);
        } else {
            return MTX_ERR_INVALID_MTX_FORMAT;
        }
    } else if (x->object == mtx_vector && y->object == mtx_vector) {
        if (x->format == mtx_array && y->format == mtx_array) {
            return mtx_vector_array_sdot(x, y, dot);
        } else if (x->format == mtx_coordinate && y->format == mtx_coordinate) {
            return mtx_vector_coordinate_sdot(x, y, dot);
        } else {
            return MTX_ERR_INVALID_MTX_FORMAT;
        }
    } else {
        return MTX_ERR_INVALID_MTX_OBJECT;
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
    if (x->object == mtx_matrix && y->object == mtx_matrix) {
        if (x->format == mtx_array && y->format == mtx_array) {
            return mtx_matrix_array_ddot(x, y, dot);
        } else if (x->format == mtx_coordinate && y->format == mtx_coordinate) {
            return mtx_matrix_coordinate_ddot(x, y, dot);
        } else {
            return MTX_ERR_INVALID_MTX_FORMAT;
        }
    } else if (x->object == mtx_vector && y->object == mtx_vector) {
        if (x->format == mtx_array && y->format == mtx_array) {
            return mtx_vector_array_ddot(x, y, dot);
        } else if (x->format == mtx_coordinate && y->format == mtx_coordinate) {
            return mtx_vector_coordinate_ddot(x, y, dot);
        } else {
            return MTX_ERR_INVALID_MTX_FORMAT;
        }
    } else {
        return MTX_ERR_INVALID_MTX_OBJECT;
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
    if (x->object == mtx_matrix) {
        if (x->format == mtx_array) {
            return mtx_matrix_array_snrm2(x, nrm2);
        } else if (x->format == mtx_coordinate) {
            return mtx_matrix_coordinate_snrm2(x, nrm2);
        } else {
            return MTX_ERR_INVALID_MTX_FORMAT;
        }
    } else if (x->object == mtx_vector) {
        if (x->format == mtx_array) {
            return mtx_vector_array_snrm2(x, nrm2);
        } else if (x->format == mtx_coordinate) {
            return mtx_vector_coordinate_snrm2(x, nrm2);
        } else {
            return MTX_ERR_INVALID_MTX_FORMAT;
        }
    } else {
        return MTX_ERR_INVALID_MTX_OBJECT;
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
    if (x->object == mtx_matrix) {
        if (x->format == mtx_array) {
            return mtx_matrix_array_dnrm2(x, nrm2);
        } else if (x->format == mtx_coordinate) {
            return mtx_matrix_coordinate_dnrm2(x, nrm2);
        } else {
            return MTX_ERR_INVALID_MTX_FORMAT;
        }
    } else if (x->object == mtx_vector) {
        if (x->format == mtx_array) {
            return mtx_vector_array_dnrm2(x, nrm2);
        } else if (x->format == mtx_coordinate) {
            return mtx_vector_coordinate_dnrm2(x, nrm2);
        } else {
            return MTX_ERR_INVALID_MTX_FORMAT;
        }
    } else {
        return MTX_ERR_INVALID_MTX_OBJECT;
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
