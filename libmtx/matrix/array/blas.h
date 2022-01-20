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

#ifndef LIBMTX_MATRIX_ARRAY_BLAS_H
#define LIBMTX_MATRIX_ARRAY_BLAS_H

struct mtx_matrix_array_data;
struct mtx_vector_array_data;

/*
 * Level 1 BLAS operations.
 */

/**
 * `mtx_matrix_array_copy()' copies values of a matrix, `y = x'.
 */
int mtx_matrix_array_copy(
    struct mtx_matrix_array_data * y,
    const struct mtx_matrix_array_data * x);

/**
 * `mtx_matrix_array_sscal()' scales a matrix by a single precision
 * floating-point scalar, `x = a*x'.
 */
int mtx_matrix_array_sscal(
    float a,
    struct mtx_matrix_array_data * x);

/**
 * `mtx_matrix_array_dscal()' scales a matrix by a double precision
 * floating-point scalar, `x = a*x'.
 */
int mtx_matrix_array_dscal(
    double a,
    struct mtx_matrix_array_data * x);

/**
 * `mtx_matrix_array_saxpy()' adds two matrices of single precision
 * floating-point values, `y = a*x + y'.
 */
int mtx_matrix_array_saxpy(
    float a,
    const struct mtx_matrix_array_data * x,
    struct mtx_matrix_array_data * y);

/**
 * `mtx_matrix_array_daxpy()' adds two matrices of double precision
 * floating-point values, `y = a*x + y'.
 */
int mtx_matrix_array_daxpy(
    double a,
    const struct mtx_matrix_array_data * x,
    struct mtx_matrix_array_data * y);

/**
 * `mtx_matrix_array_saypx()' adds two matrices of single precision
 * floating-point values, `y = a*y + x'.
 */
int mtx_matrix_array_saypx(
    float a,
    struct mtx_matrix_array_data * y,
    const struct mtx_matrix_array_data * x);

/**
 * `mtx_matrix_array_daypx()' adds two matrices of double precision
 * floating-point values, `y = a*y + x'.
 */
int mtx_matrix_array_daypx(
    double a,
    struct mtx_matrix_array_data * y,
    const struct mtx_matrix_array_data * x);

/**
 * `mtx_matrix_array_sdot()' computes the Frobenius inner product of
 * two matrices in single precision floating point.
 */
int mtx_matrix_array_sdot(
    const struct mtx_matrix_array_data * x,
    const struct mtx_matrix_array_data * y,
    float * dot);

/**
 * `mtx_matrix_array_ddot()' computes the Frobenius inner product of
 * two matrices in double precision floating point.
 */
int mtx_matrix_array_ddot(
    const struct mtx_matrix_array_data * x,
    const struct mtx_matrix_array_data * y,
    double * dot);

/**
 * `mtx_matrix_array_snrm2()' computes the Frobenius norm of a matrix
 * in single precision floating point.
 */
int mtx_matrix_array_snrm2(
    const struct mtx_matrix_array_data * x,
    float * nrm2);

/**
 * `mtx_matrix_array_dnrm2()' computes the Frobenius norm of a matrix
 * in double precision floating point.
 */
int mtx_matrix_array_dnrm2(
    const struct mtx_matrix_array_data * x,
    double * nrm2);

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
    struct mtx_vector_array_data * y);

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
    struct mtx_vector_array_data * y);

#endif
