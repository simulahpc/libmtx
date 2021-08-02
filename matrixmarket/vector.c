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
 * Last modified: 2021-08-02
 *
 * Various operations for vectors in the Matrix Market format.
 */

#include <matrixmarket/error.h>
#include <matrixmarket/header.h>
#include <matrixmarket/mtx.h>
#include <matrixmarket/vector_array.h>
#include <matrixmarket/vector_coordinate.h>

/**
 * `mtx_vector_set_zero()' zeroes a vector.
 */
int mtx_vector_set_zero(
    struct mtx * mtx)
{
    if (mtx->object != mtx_vector)
        return MTX_ERR_INVALID_MTX_OBJECT;

    if (mtx->format == mtx_array) {
        return mtx_vector_array_set_zero(mtx);
    } else if (mtx->format == mtx_coordinate) {
        return mtx_vector_coordinate_set_zero(mtx);
    } else {
        return MTX_ERR_INVALID_MTX_FORMAT;
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_vector_set_constant_real()' sets every (nonzero) value of a
 * vector equal to a constant, single precision floating point number.
 */
int mtx_vector_set_constant_real(
    struct mtx * mtx,
    float a)
{
    if (mtx->object != mtx_vector)
        return MTX_ERR_INVALID_MTX_OBJECT;

    if (mtx->format == mtx_array) {
        return mtx_vector_array_set_constant_real(mtx, a);
    } else if (mtx->format == mtx_coordinate) {
        return mtx_vector_coordinate_set_constant_real(mtx, a);
    } else {
        return MTX_ERR_INVALID_MTX_FORMAT;
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_vector_set_constant_double()' sets every (nonzero) value of a
 * vector equal to a constant, double precision floating point number.
 */
int mtx_vector_set_constant_double(
    struct mtx * mtx,
    double a)
{
    if (mtx->object != mtx_vector)
        return MTX_ERR_INVALID_MTX_OBJECT;

    if (mtx->format == mtx_array) {
        return mtx_vector_array_set_constant_double(mtx, a);
    } else if (mtx->format == mtx_coordinate) {
        return mtx_vector_coordinate_set_constant_double(mtx, a);
    } else {
        return MTX_ERR_INVALID_MTX_FORMAT;
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_vector_set_constant_complex()' sets every (nonzero) value of a
 * vector equal to a constant, single precision floating point complex
 * number.
 */
int mtx_vector_set_constant_complex(
    struct mtx * mtx,
    float a,
    float b)
{
    if (mtx->object != mtx_vector)
        return MTX_ERR_INVALID_MTX_OBJECT;

    if (mtx->format == mtx_array) {
        return mtx_vector_array_set_constant_complex(mtx, a, b);
    } else if (mtx->format == mtx_coordinate) {
        return mtx_vector_coordinate_set_constant_complex(mtx, a, b);
    } else {
        return MTX_ERR_INVALID_MTX_FORMAT;
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_vector_set_constant_integer()' sets every (nonzero) value of a
 * vector equal to a constant integer.
 */
int mtx_vector_set_constant_integer(
    struct mtx * mtx,
    int a)
{
    if (mtx->object != mtx_vector)
        return MTX_ERR_INVALID_MTX_OBJECT;

    if (mtx->format == mtx_array) {
        return mtx_vector_array_set_constant_integer(mtx, a);
    } else if (mtx->format == mtx_coordinate) {
        return mtx_vector_coordinate_set_constant_integer(mtx, a);
    } else {
        return MTX_ERR_INVALID_MTX_FORMAT;
    }
    return MTX_SUCCESS;
}
