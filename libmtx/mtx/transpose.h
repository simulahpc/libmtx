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
 * Transpose and complex conjugate of matrices and vectors.
 */

#ifndef LIBMTX_MTX_TRANSPOSE_H
#define LIBMTX_MTX_TRANSPOSE_H

struct mtx;

/**
 * `mtx_transposition' is used to enumerate different ways of
 * transposing vectors and matrices.
 */
enum mtx_transposition
{
    mtx_nontransposed,        /* original, non-transposed */
    mtx_transposed,           /* transpose */
    mtx_conjugated,           /* complex conjugate */
    mtx_conjugate_transposed, /* conjugate transpose */
};

/**
 * `mtx_transposition_str()' is a string representing the
 * transposition type.
 */
const char * mtx_transposition_str(
    enum mtx_transposition transposition);

/**
 * `mtx_transpose()' transposes a matrix or vector.
 */
int mtx_transpose(
    struct mtx * mtx);

#endif
