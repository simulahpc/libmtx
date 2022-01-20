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
 * BLAS operations for matrices and vectors in Matrix Market format.
 */

#ifndef LIBMTX_MTX_BLAS_H
#define LIBMTX_MTX_BLAS_H

struct mtx;

/*
 * Level 1 BLAS operations.
 */

/**
 * `mtx_copy()' copies the values of a vector (or matrix), `y = x'.
 */
int mtx_copy(
    struct mtx * y,
    const struct mtx * x);

/**
 * `mtx_sscal()' scales a vector (or matrix) by a single precision
 * floating-point scalar, `x = a*x'.
 */
int mtx_sscal(
    float a,
    struct mtx * x);

/**
 * `mtx_dscal()' scales a vector (or matrix) by a double precision
 * floating-point scalar, `x = a*x'.
 */
int mtx_dscal(
    double a,
    struct mtx * x);

/**
 * `mtx_saxpy()' adds two vectors (or matrices) of single precision
 * floating-point values, `y = a*x + y'.
 */
int mtx_saxpy(
    float a,
    const struct mtx * x,
    struct mtx * y);

/**
 * `mtx_daxpy()' adds two vectors (or matrices) of double precision
 * floating-point values, `y = a*x + y'.
 */
int mtx_daxpy(
    double a,
    const struct mtx * x,
    struct mtx * y);

/**
 * `mtx_saypx()' adds two vectors (or matrices) of single precision
 * floating-point values, `y = a*y + x'.
 */
int mtx_saypx(
    float a,
    struct mtx * y,
    const struct mtx * x);

/**
 * `mtx_daypx()' adds two vectors (or matrices) of double precision
 * floating-point values, `y = a*y + x'.
 */
int mtx_daypx(
    double a,
    struct mtx * y,
    const struct mtx * x);

/**
 * `mtx_sdot()' computes the Euclidean dot product of two vectors (or
 * Frobenius inner product of two matrices) of single precision
 * floating-point values.
 */
int mtx_sdot(
    const struct mtx * x,
    const struct mtx * y,
    float * dot);

/**
 * `mtx_ddot()' computes the Euclidean dot product of two vectors (or
 * Frobenius inner product of two matrices) of double precision
 * floating-point values.
 */
int mtx_ddot(
    const struct mtx * x,
    const struct mtx * y,
    double * dot);

/**
 * `mtx_snrm2()' computes the Euclidean norm of a vector (or Frobenius
 * norm of a matrix) of single precision floating-point values.
 */
int mtx_snrm2(
    const struct mtx * x,
    float * nrm2);

/**
 * `mtx_dnrm2()' computes the Euclidean norm of a vector (or Frobenius
 * norm of a matrix) of double precision floating-point values.
 */
int mtx_dnrm2(
    const struct mtx * x,
    double * nrm2);

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
    struct mtx * y);

/**
 * `mtx_dgemv()' computes the product of a matrix and a vector of
 * single precision floating-point values, `y = alpha*A*x + beta*y'.
 */
int mtx_dgemv(
    double alpha,
    const struct mtx * A,
    const struct mtx * x,
    double beta,
    struct mtx * y);

#endif
