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

#ifndef LIBMTX_VECTOR_COORDINATE_BLAS_H
#define LIBMTX_VECTOR_COORDINATE_BLAS_H

struct mtx_vector_coordinate_data;

/**
 * `mtx_vector_coordinate_copy()' copies values of a vector, `y = x'.
 */
int mtx_vector_coordinate_copy(
    struct mtx_vector_coordinate_data * y,
    const struct mtx_vector_coordinate_data * x);

/**
 * `mtx_vector_coordinate_sscal()' scales a vector by a single
 * precision floating-point scalar, `x = a*x'.
 */
int mtx_vector_coordinate_sscal(
    float a,
    struct mtx_vector_coordinate_data * x);

/**
 * `mtx_vector_coordinate_dscal()' scales a vector by a double
 * precision floating-point scalar, `x = a*x'.
 */
int mtx_vector_coordinate_dscal(
    double a,
    struct mtx_vector_coordinate_data * x);

/**
 * `mtx_vector_coordinate_saxpy()' adds two vectors of single
 * precision floating-point values, `y = a*x + y'.
 */
int mtx_vector_coordinate_saxpy(
    float a,
    const struct mtx_vector_coordinate_data * x,
    struct mtx_vector_coordinate_data * y);

/**
 * `mtx_vector_coordinate_daxpy()' adds two vectors of double
 * precision floating-point values, `y = a*x + y'.
 */
int mtx_vector_coordinate_daxpy(
    double a,
    const struct mtx_vector_coordinate_data * x,
    struct mtx_vector_coordinate_data * y);

/**
 * `mtx_vector_coordinate_saypx()' adds two vectors of single
 * precision floating-point values, `y = a*y + x'.
 */
int mtx_vector_coordinate_saypx(
    float a,
    struct mtx_vector_coordinate_data * y,
    const struct mtx_vector_coordinate_data * x);

/**
 * `mtx_vector_coordinate_daypx()' adds two vectors of double
 * precision floating-point values, `y = a*y + x'.
 */
int mtx_vector_coordinate_daypx(
    double a,
    struct mtx_vector_coordinate_data * y,
    const struct mtx_vector_coordinate_data * x);

/**
 * `mtx_vector_coordinate_sdot()' computes the Euclidean dot product
 * of two vectors of single precision floating-point values.
 */
int mtx_vector_coordinate_sdot(
    const struct mtx_vector_coordinate_data * x,
    const struct mtx_vector_coordinate_data * y,
    float * dot);

/**
 * `mtx_vector_coordinate_ddot()' computes the Euclidean dot product
 * of two vectors of double precision floating-point values.
 */
int mtx_vector_coordinate_ddot(
    const struct mtx_vector_coordinate_data * x,
    const struct mtx_vector_coordinate_data * y,
    double * dot);

/**
 * `mtx_vector_coordinate_snrm2()' computes the Euclidean norm of a
 * vector of single precision floating-point values.
 */
int mtx_vector_coordinate_snrm2(
    const struct mtx_vector_coordinate_data * x,
    float * nrm2);

/**
 * `mtx_vector_coordinate_dnrm2()' computes the Euclidean norm of a
 * vector of double precision floating-point values.
 */
int mtx_vector_coordinate_dnrm2(
    const struct mtx_vector_coordinate_data * x,
    double * nrm2);

#endif
