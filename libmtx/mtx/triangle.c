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
 * Triangular (or trapezoidal) properties of matrices.
 */

#include <libmtx/mtx/triangle.h>

/**
 * `mtx_triangle_str()' is a string representing the given triangle
 * type.
 */
const char * mtx_triangle_str(
    enum mtx_triangle triangle)
{
    switch (triangle) {
    case mtx_nontriangular: return "nontriangular";
    case mtx_lower_triangular: return "lower-triangular";
    case mtx_upper_triangular: return "upper-triangular";
    case mtx_diagonal: return "diagonal";
    /* case mtx_unit_lower_triangular: return "unit-lower-triangular"; */
    /* case mtx_unit_upper_triangular: return "unit-upper-triangular"; */
    /* case mtx_unit_diagonal: return "unit-diagonal"; */
    case mtx_strict_lower_triangular: return "strict-lower-triangular";
    case mtx_strict_upper_triangular: return "strict-upper-triangular";
    /* case mtx_zero: return "zero"; */
    default: return "unknown";
    }
}
