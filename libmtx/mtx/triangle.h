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

#ifndef LIBMTX_MTX_TRIANGLE_H
#define LIBMTX_MTX_TRIANGLE_H

/**
 * `mtx_triangle' is used to enumerate matrix properties related to
 * whether or not matrices are upper or lower triangular. Note that
 * the term triangular is still used even for non-square matrices,
 * where the term trapezoidal would be more accurate.
 */
enum mtx_triangle {
    mtx_nontriangular,           /* nonzero above, below or on main diagonal */
    mtx_lower_triangular,        /* zero above main diagonal */
    mtx_upper_triangular,        /* zero below main diagonal */
    mtx_diagonal,                /* zero above and below main diagonal */
    /* mtx_unit_lower_triangular,   /\* one on main diagonal and zero above *\/ */
    /* mtx_unit_upper_triangular,   /\* one on main diagonal and zero below *\/ */
    /* mtx_unit_diagonal,           /\* one on main diagonal, zero above and below *\/ */
    mtx_strict_lower_triangular, /* zero on or above main diagonal */
    mtx_strict_upper_triangular, /* zero on or below main diagonal */
    /* mtx_zero,                    /\* zero on, above and below main diagonal *\/ */
};

/**
 * `mtx_triangle_str()' is a string representing the given triangle
 * type.
 */
const char * mtx_triangle_str(
    enum mtx_triangle triangle);

#endif
