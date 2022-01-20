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
 * Sorting dense matrices in array format.
 */

#ifndef LIBMTX_MTX_MATRIX_ARRAY_SORT_H
#define LIBMTX_MTX_MATRIX_ARRAY_SORT_H

#include <libmtx/mtx/sort.h>

struct mtx;

/**
 * `mtx_matrix_array_sort()' sorts the entries of a matrix in array
 * format in a given order.
 */
int mtx_matrix_array_sort(
    struct mtx * mtx,
    enum mtx_sorting sorting);

#endif
