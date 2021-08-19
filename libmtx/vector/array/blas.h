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

#ifndef LIBMTX_VECTOR_ARRAY_BLAS_H
#define LIBMTX_VECTOR_ARRAY_BLAS_H

struct mtx_vector_array_data;

/**
 * `mtx_vector_array_sscal()' scales a vector by a single precision
 * floating-point scalar, `x = a*x'.
 */
int mtx_vector_array_sscal(
    float a,
    struct mtx_vector_array_data * x);

/**
 * `mtx_vector_array_dscal()' scales a vector by a double precision
 * floating-point scalar, `x = a*x'.
 */
int mtx_vector_array_dscal(
    double a,
    struct mtx_vector_array_data * x);

/**
 * `mtx_vector_array_saxpy()' adds two vectors of single precision
 * floating-point values, `y = a*x + y'.
 */
int mtx_vector_array_saxpy(
    float a,
    const struct mtx_vector_array_data * x,
    struct mtx_vector_array_data * y);

/**
 * `mtx_vector_array_daxpy()' adds two vectors of double precision
 * floating-point values, `y = a*x + y'.
 */
int mtx_vector_array_daxpy(
    double a,
    const struct mtx_vector_array_data * x,
    struct mtx_vector_array_data * y);

/**
 * `mtx_vector_array_sdot()' computes the Euclidean dot product of two
 * vectors (or Frobenius inner product of two matrices) of single
 * precision floating-point values.
 */
int mtx_vector_array_sdot(
    const struct mtx_vector_array_data * x,
    const struct mtx_vector_array_data * y,
    float * dot);

/**
 * `mtx_vector_array_ddot()' computes the Euclidean dot product of two
 * vectors (or Frobenius inner product of two matrices) of double
 * precision floating-point values.
 */
int mtx_vector_array_ddot(
    const struct mtx_vector_array_data * x,
    const struct mtx_vector_array_data * y,
    double * dot);

/**
 * `mtx_vector_array_snrm2()' computes the Euclidean norm of a vector
 * (or Frobenius norm of a matrix) of single precision floating-point
 * values.
 */
int mtx_vector_array_snrm2(
    const struct mtx_vector_array_data * x,
    float * nrm2);

/**
 * `mtx_vector_array_dnrm2()' computes the Euclidean norm of a vector
 * (or Frobenius norm of a matrix) of double precision floating-point
 * values.
 */
int mtx_vector_array_dnrm2(
    const struct mtx_vector_array_data * x,
    double * nrm2);

#endif
