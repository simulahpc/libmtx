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
 * Sorting matrices and vectors.
 */

#ifndef LIBMTX_MTX_SORT_H
#define LIBMTX_MTX_SORT_H

struct mtx;

/**
 * `mtx_sorting` is used to enumerate different ways of sorting the
 * entries of vectors and matrices.
 */
enum mtx_sorting
{
    mtx_unsorted,       /* unsorted matrix or vector entries */
    mtx_row_major,      /* row major ordering */
    mtx_column_major,   /* column major ordering */
};

/**
 * `mtx_sorting_str()` is a string representing the sorting type.
 */
const char * mtx_sorting_str(
    enum mtx_sorting sorting);

/**
 * `mtx_sort()' sorts matrix or vector entries in a given order.
 */
int mtx_sort(
    struct mtx * mtx,
    enum mtx_sorting sorting);

#endif
