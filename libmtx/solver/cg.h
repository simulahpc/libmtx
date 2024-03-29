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
 * Last modified: 2022-10-08
 *
 * Conjugate gradient (CG) algorithm.
 */

#ifndef LIBMTX_SOLVER_CG_H
#define LIBMTX_SOLVER_CG_H

#include <libmtx/libmtx-config.h>

#include <libmtx/linalg/precision.h>
#include <libmtx/linalg/field.h>
#include <libmtx/linalg/local/vector.h>

#include <stdbool.h>

struct mtxmatrix;

/**
 * ‘mtxcg’ represents a solver based on the conjugate gradient method.
 */
struct mtxcg
{
    const struct mtxmatrix * A;
    struct mtxvector r;
    struct mtxvector p;
    struct mtxvector t;
    bool started;
};

/**
 * ‘mtxcg_free()’ frees storage allocated for a matrix.
 */
LIBMTX_API void mtxcg_free(
    struct mtxcg * cg);

/**
 * ‘mtxcg_init()’ sets up a conjugate gradient solver.
 */
LIBMTX_API int mtxcg_init(
    struct mtxcg * cg,
    const struct mtxmatrix * A,
    enum mtxvectortype vectortype);

/**
 * ‘mtxcg_solve()’ solves a linear system using CG.
 */
LIBMTX_API int mtxcg_solve(
    struct mtxcg * cg,
    const struct mtxvector * b,
    struct mtxvector * x,
    double atol,
    double rtol,
    int max_iterations,
    bool recompute_residual,
    int * num_iterations,
    double * b_nrm2,
    double * r_nrm2,
    double * r0_nrm2,
    int64_t * num_flops);

#endif
