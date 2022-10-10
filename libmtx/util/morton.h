/* This file is part of Libmtx.
 *
 * Copyright (C) 2022 James D. Trotter
 *
 * Libmtx is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Libmtx is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Libmtx.  If not, see <https://www.gnu.org/licenses/>.
 *
 * Authors: James D. Trotter <james@simula.no>
 * Last modified: 2022-10-10
 *
 * Morton Z-ordering.
 */

#ifndef LIBMTX_UTIL_MORTON_H
#define LIBMTX_UTIL_MORTON_H

#include <stdint.h>

/**
 * ‘morton2d_from_cartesian_uint32()’ converts from Cartesian to 2D
 * Morton Z-order for 32-bit unsigned integers.
 *
 * The arrays ‘x’, ‘y’ and ‘z’ must be of length ‘size’. The arrays
 * ‘x’ and ‘y’ specify the Cartesian coordinates of each point,
 * whereas ‘z’ is used to output the corresponding Morton code.
 */
void morton2d_from_cartesian_uint32(
    int64_t size,
    int xstride,
    const uint32_t * x,
    int ystride,
    const uint32_t * y,
    int zstride,
    uint64_t * z);

/**
 * ‘morton2d_from_cartesian_uint64()’ converts from Cartesian to 2D
 * Morton Z-order for 64-bit unsigned integers.
 *
 * The arrays ‘x’, ‘y’, ‘z0’ and ‘z1’ must be of length ‘size’. The
 * arrays ‘x’ and ‘y’ specify the Cartesian coordinates of each point,
 * whereas ‘z0’ and ‘z1’ are used to output the corresponding Morton
 * code.
 */
void morton2d_from_cartesian_uint64(
    int64_t size,
    int xstride,
    const uint64_t * x,
    int ystride,
    const uint64_t * y,
    int z0stride,
    uint64_t * z0,
    int z1stride,
    uint64_t * z1);

/**
 * ‘morton2d_to_cartesian_uint32()’ converts to Cartesian coordinates
 * from 2D Morton Z-order for 32-bit unsigned integers.
 *
 * The arrays ‘x’, ‘y’ and ‘z’ must be of length ‘size’. The array ‘z’
 * is used to specify the Morton code of each point, whereas ‘x’ and
 * ‘y’ are used to output the cooresponding Cartesian coordinates.
 */
void morton2d_to_cartesian_uint32(
    int64_t size,
    int zstride,
    const uint64_t * z,
    int xstride,
    uint32_t * x,
    int ystride,
    uint32_t * y);

/**
 * ‘morton2d_to_cartesian_uint64()’ converts to Cartesian coordinates
 * from 2D Morton Z-order for 64-bit unsigned integers.
 *
 * The arrays ‘x’, ‘y’, ‘z0’ and ‘z1’ must be of length ‘size’. The
 * arrays ‘z0’ and ‘z1’ are used to specify the Morton code of each
 * point, whereas ‘x’ and ‘y’ are used to output the cooresponding
 * Cartesian coordinates.
 */
void morton2d_to_cartesian_uint64(
    int64_t size,
    int z0stride,
    const uint64_t * z0,
    int z1stride,
    const uint64_t * z1,
    int xstride,
    uint64_t * x,
    int ystride,
    uint64_t * y);

#endif
