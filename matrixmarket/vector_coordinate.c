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
 * Last modified: 2021-06-18
 *
 * Sparse vectors in Matrix Market format.
 */

#include <matrixmarket/error.h>
#include <matrixmarket/vector_coordinate.h>

#include <errno.h>

/**
 * `mtx_init_vector_coordinate_real()` creates a sparse vector with
 * real, single-precision floating point coefficients.
 */
int mtx_init_vector_coordinate_real(
    struct mtx * mtx,
    int num_comment_lines,
    const char ** comment_lines,
    int size,
    const struct mtx_vector_coordinate_real * data)
{
    errno = ENOTSUP;
    return MTX_ERR_ERRNO;
}

/**
 * `mtx_init_vector_coordinate_double()` creates a sparse vector
 * with real, double-precision floating point coefficients.
 */
int mtx_init_vector_coordinate_double(
    struct mtx * mtx,
    int num_comment_lines,
    const char ** comment_lines,
    int size,
    const struct mtx_vector_coordinate_double * data)
{
    errno = ENOTSUP;
    return MTX_ERR_ERRNO;
}

/**
 * `mtx_init_vector_coordinate_complex()` creates a sparse vector
 * with complex, single-precision floating point coefficients.
 */
int mtx_init_vector_coordinate_complex(
    struct mtx * mtx,
    int num_comment_lines,
    const char ** comment_lines,
    int size,
    const struct mtx_vector_coordinate_complex * data)
{
    errno = ENOTSUP;
    return MTX_ERR_ERRNO;
}

/**
 * `mtx_init_vector_coordinate_integer()` creates a sparse vector
 * with integer coefficients.
 */
int mtx_init_vector_coordinate_integer(
    struct mtx * mtx,
    int num_comment_lines,
    const char ** comment_lines,
    int size,
    const struct mtx_vector_coordinate_integer * data)
{
    errno = ENOTSUP;
    return MTX_ERR_ERRNO;
}

/**
 * `mtx_init_vector_coordinate_pattern()` creates a sparse vector
 * with boolean coefficients.
 */
int mtx_init_vector_coordinate_pattern(
    struct mtx * mtx,
    int num_comment_lines,
    const char ** comment_lines,
    int size,
    const struct mtx_vector_coordinate_pattern * data)
{
    errno = ENOTSUP;
    return MTX_ERR_ERRNO;
}
