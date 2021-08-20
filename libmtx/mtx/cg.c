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

#include <libmtx/mtx/cg.h>

#include <libmtx/error.h>
#include <libmtx/mtx/blas.h>
#include <libmtx/mtx/mtx.h>

#include <math.h>
#include <stdbool.h>
#include <stdlib.h>

/*
 * Single precision floating point version.
 */

/**
 * `mtx_scg_workspace' is used to hold working memory needed by the
 * conjugate gradient algorithm in `mtx_scg()'.
 */
struct mtx_scg_workspace
{
    float b_nrm2;
    struct mtx r;
    struct mtx p;
    struct mtx t;
};

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
    const struct mtx * x0)
{
    int err;
    *workspace = malloc(sizeof(struct mtx_scg_workspace));
    if (!*workspace)
        return MTX_ERR_ERRNO;

    err = mtx_copy_alloc(&(*workspace)->r, b);
    if (err) {
        free(*workspace);
        return err;
    }

    err = mtx_copy_alloc(&(*workspace)->p, b);
    if (err) {
        mtx_free(&(*workspace)->r);
        free(*workspace);
        return err;
    }

    err = mtx_copy_alloc(&(*workspace)->t, x0);
    if (err) {
        mtx_free(&(*workspace)->p);
        mtx_free(&(*workspace)->r);
        free(*workspace);
        return err;
    }

    err = mtx_scg_restart(*workspace, A, b, x0);
    if (err) {
        mtx_free(&(*workspace)->t);
        mtx_free(&(*workspace)->p);
        mtx_free(&(*workspace)->r);
        free(*workspace);
        return err;
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_scg_workspace_free()' frees allocated workspace used by
 * `mtx_scg()'.
 */
int mtx_scg_workspace_free(
    struct mtx_scg_workspace * workspace)
{
    mtx_free(&workspace->t);
    mtx_free(&workspace->p);
    mtx_free(&workspace->r);
    free(workspace);
}

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
    const struct mtx * x0)
{
    int err;
    float * b_nrm2 = &workspace->b_nrm2;
    struct mtx * r = &workspace->r;
    struct mtx * p = &workspace->p;
    struct mtx * t = &workspace->t;

    *b_nrm2 = INFINITY;
    err = mtx_snrm2(b, b_nrm2);
    if (err)
        return err;

    // r0 = b - A*x0
    err = mtx_copy(r, b);
    if (err)
        return err;
    err = mtx_sgemv(-1.0, A, x0, 1.0, r);
    if (err)
        return err;

    // p = r
    err = mtx_copy(p, r);
    if (err)
        return err;
    return MTX_SUCCESS;
}

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
 * created with a call to `mtx_dcg_workspace_alloc'.
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
    struct mtx_scg_workspace * workspace)
{
    int err;
    if (num_iterations)
        *num_iterations = 0;
    *b_nrm2 = INFINITY;
    *r_nrm2 = INFINITY;

    bool alloc_workspace = (workspace == NULL);
    if (alloc_workspace) {
        err = mtx_scg_workspace_alloc(&workspace, A, b, x);
        if (err)
            return MTX_ERR_ERRNO;
    }
    struct mtx * r = &workspace->r;
    struct mtx * p = &workspace->p;
    struct mtx * t = &workspace->t;
    *b_nrm2 = workspace->b_nrm2;
    rtol *= *b_nrm2;

    float r_nrm2_sqr;
    err = mtx_sdot(r, r, &r_nrm2_sqr);
    if (err) {
        if (alloc_workspace)
            mtx_scg_workspace_free(workspace);
        return err;
    }
    *r_nrm2 = sqrt(r_nrm2_sqr);

    if (max_iterations <= 0) {
        if (alloc_workspace)
            mtx_scg_workspace_free(workspace);
        if (*r_nrm2 <= rtol || *r_nrm2 <= atol)
            return MTX_SUCCESS;
        return MTX_ERR_NOT_CONVERGED;
    }

    for (int k = 0; k < max_iterations; k++) {
        // t = A*p
        err = mtx_sgemv(1.0, A, p, 0.0, t);
        if (err) {
            if (alloc_workspace)
                mtx_scg_workspace_free(workspace);
            return err;
        }

        // alpha = (r,r) / (p,A*p)
        float alpha;
        err = mtx_sdot(p, t, &alpha);
        if (err) {
            if (alloc_workspace)
                mtx_scg_workspace_free(workspace);
            return err;
        }
        alpha = r_nrm2_sqr / alpha;

        // x = alpha*p + x
        err = mtx_saxpy(alpha, p, x);
        if (err) {
            if (alloc_workspace)
                mtx_scg_workspace_free(workspace);
            return err;
        }

        // r = -alpha*t + r
        err = mtx_saxpy(-alpha, t, r);
        if (err) {
            if (alloc_workspace)
                mtx_scg_workspace_free(workspace);
            return err;
        }

        // beta = (r,r) / (r_prev,r_prev)
        float beta = r_nrm2_sqr;
        err = mtx_sdot(r, r, &r_nrm2_sqr);
        if (err) {
            if (alloc_workspace)
                mtx_scg_workspace_free(workspace);
            return err;
        }
        beta = r_nrm2_sqr / beta;

        *r_nrm2 = sqrt(r_nrm2_sqr);
        if (num_iterations)
            (*num_iterations)++;
        if (*r_nrm2 <= rtol || *r_nrm2 <= atol) {
            if (alloc_workspace)
                mtx_scg_workspace_free(workspace);
            return MTX_SUCCESS;
        }

        // p = beta*p + r
        err = mtx_saypx(beta, p, r);
        if (err) {
            if (alloc_workspace)
                mtx_scg_workspace_free(workspace);
            return err;
        }
    }

    if (alloc_workspace)
        mtx_scg_workspace_free(workspace);
    return MTX_ERR_NOT_CONVERGED;
}

/*
 * Double precision floating point version.
 */

/**
 * `mtx_dcg_workspace' is used to hold working memory needed by the
 * conjugate gradient algorithm in `mtx_dcg()'.
 */
struct mtx_dcg_workspace
{
    double b_nrm2;
    struct mtx r;
    struct mtx p;
    struct mtx t;
};

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
    const struct mtx * x0)
{
    int err;
    *workspace = malloc(sizeof(struct mtx_dcg_workspace));
    if (!*workspace)
        return MTX_ERR_ERRNO;

    err = mtx_copy_alloc(&(*workspace)->r, b);
    if (err) {
        free(*workspace);
        return err;
    }

    err = mtx_copy_alloc(&(*workspace)->p, b);
    if (err) {
        mtx_free(&(*workspace)->r);
        free(*workspace);
        return err;
    }

    err = mtx_copy_alloc(&(*workspace)->t, x0);
    if (err) {
        mtx_free(&(*workspace)->p);
        mtx_free(&(*workspace)->r);
        free(*workspace);
        return err;
    }

    err = mtx_dcg_restart(*workspace, A, b, x0);
    if (err) {
        mtx_free(&(*workspace)->t);
        mtx_free(&(*workspace)->p);
        mtx_free(&(*workspace)->r);
        free(*workspace);
        return err;
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_dcg_workspace_free()' frees allocated workspace used by
 * `mtx_dcg'.
 */
int mtx_dcg_workspace_free(
    struct mtx_dcg_workspace * workspace)
{
    mtx_free(&workspace->t);
    mtx_free(&workspace->p);
    mtx_free(&workspace->r);
    free(workspace);
}

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
    const struct mtx * x0)
{
    int err;
    double * b_nrm2 = &workspace->b_nrm2;
    struct mtx * r = &workspace->r;
    struct mtx * p = &workspace->p;
    struct mtx * t = &workspace->t;

    *b_nrm2 = INFINITY;
    err = mtx_dnrm2(b, b_nrm2);
    if (err)
        return err;

    // r0 = b - A*x0
    err = mtx_copy(r, b);
    if (err)
        return err;
    err = mtx_dgemv(-1.0, A, x0, 1.0, r);
    if (err)
        return err;

    // p = r
    err = mtx_copy(p, r);
    if (err)
        return err;
    return MTX_SUCCESS;
}

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
    struct mtx_dcg_workspace * workspace)
{
    int err;
    if (num_iterations)
        *num_iterations = 0;
    *b_nrm2 = INFINITY;
    *r_nrm2 = INFINITY;

    bool alloc_workspace = (workspace == NULL);
    if (alloc_workspace) {
        err = mtx_dcg_workspace_alloc(&workspace, A, b, x);
        if (err)
            return MTX_ERR_ERRNO;
    }
    struct mtx * r = &workspace->r;
    struct mtx * p = &workspace->p;
    struct mtx * t = &workspace->t;
    *b_nrm2 = workspace->b_nrm2;
    rtol *= *b_nrm2;

    double r_nrm2_sqr;
    err = mtx_ddot(r, r, &r_nrm2_sqr);
    if (err) {
        if (alloc_workspace)
            mtx_dcg_workspace_free(workspace);
        return err;
    }
    *r_nrm2 = sqrt(r_nrm2_sqr);

    if (max_iterations <= 0) {
        if (alloc_workspace)
            mtx_dcg_workspace_free(workspace);
        if (*r_nrm2 <= rtol || *r_nrm2 <= atol)
            return MTX_SUCCESS;
        return MTX_ERR_NOT_CONVERGED;
    }

    for (int k = 0; k < max_iterations; k++) {
        // t = A*p
        err = mtx_dgemv(1.0, A, p, 0.0, t);
        if (err) {
            if (alloc_workspace)
                mtx_dcg_workspace_free(workspace);
            return err;
        }

        // alpha = (r,r) / (p,A*p)
        double alpha;
        err = mtx_ddot(p, t, &alpha);
        if (err) {
            if (alloc_workspace)
                mtx_dcg_workspace_free(workspace);
            return err;
        }
        alpha = r_nrm2_sqr / alpha;

        // x = alpha*p + x
        err = mtx_daxpy(alpha, p, x);
        if (err) {
            if (alloc_workspace)
                mtx_dcg_workspace_free(workspace);
            return err;
        }

        // r = -alpha*t + r
        err = mtx_daxpy(-alpha, t, r);
        if (err) {
            if (alloc_workspace)
                mtx_dcg_workspace_free(workspace);
            return err;
        }

        // beta = (r,r) / (r_prev,r_prev)
        double beta = r_nrm2_sqr;
        err = mtx_ddot(r, r, &r_nrm2_sqr);
        if (err) {
            if (alloc_workspace)
                mtx_dcg_workspace_free(workspace);
            return err;
        }
        beta = r_nrm2_sqr / beta;

        *r_nrm2 = sqrt(r_nrm2_sqr);
        if (num_iterations)
            (*num_iterations)++;
        if (*r_nrm2 <= rtol || *r_nrm2 <= atol) {
            if (alloc_workspace)
                mtx_dcg_workspace_free(workspace);
            return MTX_SUCCESS;
        }

        // p = beta*p + r
        err = mtx_daypx(beta, p, r);
        if (err) {
            if (alloc_workspace)
                mtx_dcg_workspace_free(workspace);
            return err;
        }
    }

    if (alloc_workspace)
        mtx_dcg_workspace_free(workspace);
    return MTX_ERR_NOT_CONVERGED;
}
