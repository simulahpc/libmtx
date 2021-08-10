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

#ifndef LIBMTX_MATRIX_ARRAY_BLAS_H
#define LIBMTX_MATRIX_ARRAY_BLAS_H

struct mtx;

/*
 * Level 2 BLAS operations.
 */

/**
 * `mtx_matrix_array_sscal()' scales a matrix by a single precision
 * floating-point scalar, `x = a*x'.
 */
int mtx_matrix_array_sscal(
    float a,
    struct mtx * x);

/**
 * `mtx_matrix_array_dscal()' scales a matrix by a double precision
 * floating-point scalar, `x = a*x'.
 */
int mtx_matrix_array_dscal(
    double a,
    struct mtx * x);

/**
 * `mtx_matrix_array_saxpy()' adds two matrices of single precision
 * floating-point values, `y = a*x + y'.
 */
int mtx_matrix_array_saxpy(
    float a,
    const struct mtx * x,
    struct mtx * y);

/**
 * `mtx_matrix_array_daxpy()' adds two matrices of double precision
 * floating-point values, `y = a*x + y'.
 */
int mtx_matrix_array_daxpy(
    double a,
    const struct mtx * x,
    struct mtx * y);

/**
 * `mtx_matrix_array_sdot()' computes the Frobenius inner product of
 * two matrices of single precision floating-point values.
 */
int mtx_matrix_array_sdot(
    const struct mtx * x,
    const struct mtx * y,
    float * dot);

/**
 * `mtx_matrix_array_ddot()' computes the Frobenius inner product of
 * two matrices of double precision floating-point values.
 */
int mtx_matrix_array_ddot(
    const struct mtx * x,
    const struct mtx * y,
    double * dot);

/**
 * `mtx_matrix_array_snrm2()' computes the Frobenius norm of a matrix
 * of single precision floating-point values.
 */
int mtx_matrix_array_snrm2(
    const struct mtx * x,
    float * nrm2);

/**
 * `mtx_matrix_array_dnrm2()' computes the Frobenius norm of a matrix
 * of double precision floating-point values.
 */
int mtx_matrix_array_dnrm2(
    const struct mtx * x,
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
    const struct mtx * A,
    const struct mtx * x,
    float beta,
    struct mtx * y);

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
    struct mtx * y);

#endif