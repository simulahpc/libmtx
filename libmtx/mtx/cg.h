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
 * Last modified: 2021-08-20
 *
 * Iterative solver based on the conjugate gradient method.
 */

#ifndef LIBMTX_MTX_CG_H
#define LIBMTX_MTX_CG_H

#include <stdbool.h>

struct mtx;
struct mtx_scg_workspace;
struct mtx_dcg_workspace;

/*
 * Single precision floating point version.
 */

/**
 * `mtx_scg()' uses the conjugate gradient method to (approximately)
 * solve the linear system `Ax=b'.
 *
 * Convergence is declared if the 2-norm of the residual, `b-Ax', falls
 * below the absolute tolerance `atol', or if the 2-norm of the
 * residual divided by the 2-norm of the right-hand side `b' falls
 * below the relative tolerance `rtol'.
 *
 * In any case, the iteration is stopped on convergence, or if the
 * number of iterations reaches `max_iterations'.  If the method
 * converged, then `MTX_SUCCESS' is returned, and
 * `MTX_ERROR_NOT_CONVERGED' is returned otherwise.  Furthermore,
 * `b_nrm2' and `r_nrm2' are set to the 2-norm of the right-hand side
 * and residual, respectively.
 *
 * If `workspace' is not `NULL', then it must point to a previously
 * allocated workspace for the conjugate gradient algorithm that was
 * created with a call to `mtx_scg_workspace_alloc'.
 */
int mtx_scg(
    const struct mtx * A,
    struct mtx * x,
    const struct mtx * b,
    float atol,
    float rtol,
    int max_iterations,
    int * num_iterations,
    float * b_nrm2,
    float * r_nrm2,
    struct mtx_scg_workspace * workspace);

/**
 * `mtx_scg_workspace_alloc()' allocates workspace to be used by
 * `mtx_scg'.
 *
 * After allocating memory for auxiliary vectors needed during the
 * conjugate gradient algorithm, `mtx_scg_workspace_alloc' calls
 * `mtx_scg_restart' with the provided `A', `b' and `x0' arguments.
 */
int mtx_scg_workspace_alloc(
    struct mtx_scg_workspace ** workspace,
    const struct mtx * A,
    const struct mtx * b,
    const struct mtx * x0);

/**
 * `mtx_scg_workspace_free()' frees allocated workspace used by
 * `mtx_scg()'.
 */
int mtx_scg_workspace_free(
    struct mtx_scg_workspace * workspace);

/**
 * `mtx_scg_restart()' restarts the conjugate gradient algorithm by
 * initialising a workspace to be used by `mtx_scg()' for a given
 * matrix `A', right-hand side vector `b' and initial guess `x0'.
 *
 * The workspace must already have been allocated by calling
 * `mtx_scg_workspace_alloc()' for a matrix and right-hand side of the
 * same dimensions and type as `A' and `b'.
 */
int mtx_scg_restart(
    struct mtx_scg_workspace * workspace,
    const struct mtx * A,
    const struct mtx * b,
    const struct mtx * x0);

/*
 * Double precision floating point version.
 */

/**
 * `mtx_dcg()' uses the conjugate gradient method to (approximately)
 * solve the linear system `Ax=b'.
 *
 * The matrix `A' should be symmetric and positive definite.
 *
 * Convergence is declared if the 2-norm of the residual, `b-Ax', falls
 * below the absolute tolerance `atol', or if the 2-norm of the
 * residual divided by the 2-norm of the right-hand side `b' falls
 * below the relative tolerance `rtol'.
 *
 * In any case, the iteration is stopped on convergence, or if the
 * number of iterations reaches `max_iterations'.  If the method
 * converged, then `MTX_SUCCESS' is returned, and
 * `MTX_ERROR_NOT_CONVERGED' is returned otherwise.  Furthermore,
 * `b_nrm2' and `r_nrm2' are set to the 2-norm of the right-hand side
 * and residual, respectively.
 *
 * If `workspace' is not `NULL', then it must point to a previously
 * allocated workspace for the conjugate gradient algorithm that was
 * created with a call to `mtx_dcg_workspace_alloc'.
 */
int mtx_dcg(
    const struct mtx * A,
    struct mtx * x,
    const struct mtx * b,
    double atol,
    double rtol,
    int max_iterations,
    int * num_iterations,
    double * b_nrm2,
    double * r_nrm2,
    struct mtx_dcg_workspace * workspace);

/**
 * `mtx_dcg_workspace_alloc()' allocates workspace to be used by
 * `mtx_dcg()'.
 *
 * After allocating memory for auxiliary vectors needed during the
 * conjugate gradient algorithm, `mtx_dcg_workspace_alloc' calls
 * `mtx_dcg_restart' with the provided `A', `b' and `x0' arguments.
 */
int mtx_dcg_workspace_alloc(
    struct mtx_dcg_workspace ** workspace,
    const struct mtx * A,
    const struct mtx * b,
    const struct mtx * x0);

/**
 * `mtx_dcg_workspace_free()' frees allocated workspace used by
 * `mtx_dcg'.
 */
int mtx_dcg_workspace_free(
    struct mtx_dcg_workspace * workspace);

/**
 * `mtx_dcg_restart()' restarts the conjugate gradient algorithm by
 * initialising a workspace to be used by `mtx_dcg()' for a given
 * matrix `A', right-hand side vector `b' and initial guess `x0'.
 *
 * The workspace must already have been allocated by calling
 * `mtx_dcg_workspace_alloc()' for a matrix and right-hand side of the
 * same dimensions and type as `A' and `b'.
 */
int mtx_dcg_restart(
    struct mtx_dcg_workspace * workspace,
    const struct mtx * A,
    const struct mtx * b,
    const struct mtx * x0);

#endif
