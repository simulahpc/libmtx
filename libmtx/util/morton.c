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
 * Last modified: 2022-05-24
 *
 * Morton Z-ordering.
 */

#include <libmtx/libmtx-config.h>

#include <libmtx/error.h>
#include <libmtx/util/morton.h>

#include <stdint.h>

#if defined(HAVE_IMMINTRIN_H) && defined(HAVE_BMI2_INSTRUCTIONS) && defined(LIBMTX_USE_BMI2)
#include <immintrin.h>
#else
static inline uint32_t _pdep_u32(uint32_t val, uint32_t mask)
{
    uint32_t res = 0;
    for (uint32_t bb = 1; mask; bb += bb) {
        if (val & bb)
            res |= mask & -mask;
        mask &= mask - 1;
    }
    return res;
}

static inline uint64_t _pdep_u64(uint64_t val, uint64_t mask)
{
    uint64_t res = 0;
    for (uint64_t bb = 1; mask; bb += bb) {
        if (val & bb)
            res |= mask & -mask;
        mask &= mask - 1;
    }
    return res;
}

static inline uint64_t _pext_u64(uint64_t val, uint64_t mask)
{
    uint64_t res = 0;
    int m = 0, k = 0;
    for (uint64_t bb = 1; m < 64; bb += bb, m++) {
        if (mask & bb) {
            res |= (((val & bb) >> m) << k); k++;
        }
    }
    return res;
}
#endif

/**
 * ‘morton2d_from_cartesian_uint32()’ converts from Cartesian
 * coordinates to 2D Morton Z-order for 32-bit unsigned integers.
 *
 * The arrays ‘x’, ‘x’, ‘z0’ and ‘z1’ must be of length ‘size’. The
 * arrays ‘x’ and ‘y’ specify the Cartesian coordinates of each point,
 * whereas ‘z’ is used to output the corresponding Morton code.
 */
int morton2d_from_cartesian_uint32(
    int64_t size,
    int xstride,
    const uint32_t * x,
    int ystride,
    const uint32_t * y,
    int zstride,
    uint64_t * z)
{
    for (int64_t i = 0; i < size; i++) {
        uint32_t xi = *(const uint32_t *) ((const char *) x + i*xstride);
        uint32_t yi = *(const uint32_t *) ((const char *) y + i*ystride);
        uint64_t * zi = (uint64_t *) ((char *) z + i*zstride);
        *zi = _pdep_u64(xi, 0xaaaaaaaaaaaaaaaaULL) |
            _pdep_u64(yi, 0x5555555555555555ULL);
    }
    return MTX_SUCCESS;
}

/**
 * ‘morton2d_from_cartesian_uint64()’ converts from Cartesian to 2D
 * Morton Z-order for 64-bit unsigned integers.
 *
 * The arrays ‘x’, ‘y’, ‘z0’ and ‘z1’ must be of length ‘size’. The
 * arrays ‘x’ and ‘y’ specify the Cartesian coordinates of each point,
 * whereas ‘z0’ and ‘z1’ are used to output the corresponding Morton
 * code.
 */
int morton2d_from_cartesian_uint64(
    int64_t size,
    int xstride,
    const uint64_t * x,
    int ystride,
    const uint64_t * y,
    int z0stride,
    uint64_t * z0,
    int z1stride,
    uint64_t * z1)
{
    for (int64_t i = 0; i < size; i++) {
        uint64_t xi = *(const uint64_t *) ((const char *) x + i*xstride);
        uint64_t yi = *(const uint64_t *) ((const char *) y + i*ystride);
        uint64_t * z0i = (uint64_t *) ((char *) z0 + i*z0stride);
        uint64_t * z1i = (uint64_t *) ((char *) z1 + i*z1stride);
        *z1i = _pdep_u64(xi, 0xaaaaaaaaaaaaaaaaULL) |
            _pdep_u64(yi, 0x5555555555555555ULL);
        *z0i = _pdep_u64(xi >> 32, 0xaaaaaaaaaaaaaaaaULL) |
            _pdep_u64(yi >> 32, 0x5555555555555555ULL);
    }
    return MTX_SUCCESS;
}

/**
 * ‘morton2d_to_cartesian_uint32()’ converts to Cartesian coordinates
 * from 2D Morton Z-order for 32-bit unsigned integers.
 *
 * The arrays ‘x’, ‘y’ and ‘z’ must be of length ‘size’. The array ‘z’
 * is used to specify the Morton code of each point, whereas ‘x’ and
 * ‘y’ are used to output the cooresponding Cartesian coordinates.
 */
int morton2d_to_cartesian_uint32(
    int64_t size,
    int zstride,
    const uint64_t * z,
    int xstride,
    uint32_t * x,
    int ystride,
    uint32_t * y)
{
    for (int64_t i = 0; i < size; i++) {
        uint32_t * xi = (uint32_t *) ((char *) x + i*xstride);
        uint32_t * yi = (uint32_t *) ((char *) y + i*ystride);
        uint64_t zi = *(const uint64_t *) ((const char *) z + i*zstride);
        *xi = _pext_u64(zi, 0xaaaaaaaaaaaaaaaaULL);
        *yi = _pext_u64(zi, 0x5555555555555555ULL);
    }
    return MTX_SUCCESS;
}

/**
 * ‘morton2d_to_cartesian_uint64()’ converts to Cartesian coordinates
 * from 2D Morton Z-order for 64-bit unsigned integers.
 *
 * The arrays ‘x’, ‘y’, ‘z0’ and ‘z1’ must be of length ‘size’. The
 * arrays ‘z0’ and ‘z1’ are used to specify the Morton code of each
 * point, whereas ‘x’ and ‘y’ are used to output the cooresponding
 * Cartesian coordinates.
 */
int morton2d_to_cartesian_uint64(
    int64_t size,
    int z0stride,
    const uint64_t * z0,
    int z1stride,
    const uint64_t * z1,
    int xstride,
    uint64_t * x,
    int ystride,
    uint64_t * y)
{
    for (int64_t i = 0; i < size; i++) {
        uint64_t * xi = (uint64_t *) ((char *) x + i*xstride);
        uint64_t * yi = (uint64_t *) ((char *) y + i*ystride);
        uint64_t z0i = *(const uint64_t *) ((const char *) z0 + i*z0stride);
        uint64_t z1i = *(const uint64_t *) ((const char *) z1 + i*z1stride);
        *xi = (_pext_u64(z0i, 0xaaaaaaaaaaaaaaaaULL) << 32)
            | _pext_u64(z1i, 0xaaaaaaaaaaaaaaaaULL);
        *yi = (_pext_u64(z0i, 0x5555555555555555ULL) << 32)
            | _pext_u64(z1i, 0x5555555555555555ULL);
    }
    return MTX_SUCCESS;
}
