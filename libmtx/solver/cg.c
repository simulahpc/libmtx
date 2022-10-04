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
 * Last modified: 2022-05-19
 *
 * Conjugate gradient (CG) algorithm.
 */

#include <libmtx/libmtx-config.h>

#include <libmtx/error.h>
#include <libmtx/linalg/precision.h>
#include <libmtx/linalg/field.h>
#include <libmtx/solver/cg.h>
#include <libmtx/linalg/local/matrix.h>
#include <libmtx/linalg/local/vector.h>

#include <math.h>

/**
 * ‘mtxcg_free()’ frees storage allocated for a matrix.
 */
void mtxcg_free(
    struct mtxcg * cg)
{
    mtxvector_free(&cg->t);
    mtxvector_free(&cg->p);
    mtxvector_free(&cg->r);
}

/**
 * ‘mtxcg_init()’ sets up a conjugate gradient solver.
 */
int mtxcg_init(
    struct mtxcg * cg,
    const struct mtxmatrix * A,
    enum mtxvectortype vectortype)
{
    cg->A = A;
    int err = mtxmatrix_alloc_row_vector(A, &cg->r, vectortype);
    if (err) return err;
    err = mtxvector_alloc_copy(&cg->p, &cg->r);
    if (err) { mtxvector_free(&cg->r); return err; }
    err = mtxmatrix_alloc_column_vector(A, &cg->t, vectortype);
    if (err) { mtxvector_free(&cg->p); mtxvector_free(&cg->r); return err; }
    cg->started = false;
    return MTX_SUCCESS;
}

/**
 * ‘mtxcg_solve()’ solves a linear system using CG.
 */
int mtxcg_solve(
    struct mtxcg * cg,
    const struct mtxvector * b,
    struct mtxvector * x,
    double atol,
    double rtol,
    int max_iterations,
    bool recompute_residual,
    int * num_iterations,
    double * out_b_nrm2,
    double * out_r_nrm2,
    double * out_r0_nrm2,
    int64_t * num_flops)
{
    const struct mtxmatrix * A = cg->A;
    struct mtxvector * r = &cg->r;
    struct mtxvector * p = &cg->p;
    struct mtxvector * t = &cg->t;

    if (num_iterations) *num_iterations = 0;
    if (out_b_nrm2) *out_b_nrm2 = INFINITY;
    if (out_r0_nrm2) *out_r0_nrm2 = INFINITY;
    if (out_r_nrm2) *out_r_nrm2 = INFINITY;

    double b_nrm2;
    int err = mtxvector_dnrm2(b, &b_nrm2, num_flops);
    if (err) return err;
    if (out_b_nrm2) *out_b_nrm2 = b_nrm2;
    rtol *= b_nrm2;

    // r = b - A*x, and p = r
    if (!cg->started || recompute_residual) {
        err = mtxvector_copy(r, b); if (err) return err;
        err = mtxmatrix_dgemv(mtx_notrans, -1.0, A, x, 1.0, r, num_flops);
        if (err) return err;
        err = mtxvector_copy(p, r); if (err) return err;
        cg->started = true;
    }

    double r_nrm2_sqr;
    err = mtxvector_ddot(r, r, &r_nrm2_sqr, num_flops);
    if (err) return err;

    double r_nrm2;
    r_nrm2 = sqrt(r_nrm2_sqr);
    if (out_r_nrm2) *out_r_nrm2 = r_nrm2;
    if (out_r0_nrm2) *out_r0_nrm2 = r_nrm2;

    if (max_iterations <= 0 && (r_nrm2 <= rtol || r_nrm2 <= atol)) return MTX_SUCCESS;
    else if (max_iterations <= 0) return MTX_ERR_NOT_CONVERGED;

    for (int k = 0; k < max_iterations; k++) {
        // t = A*p
        err = mtxmatrix_dgemv(mtx_notrans, 1.0, A, p, 0.0, t, num_flops);
        if (err) return err;

        // alpha = (r,r) / (p,A*p)
        double alpha;
        err = mtxvector_ddot(p, t, &alpha, num_flops);
        if (err) return err;
        alpha = r_nrm2_sqr / alpha;

        // x = alpha*p + x
        err = mtxvector_daxpy(alpha, p, x, num_flops);
        if (err) return err;

        // r = -alpha*t + r
        err = mtxvector_daxpy(-alpha, t, r, num_flops);
        if (err) return err;

        // beta = (r,r) / (r_prev,r_prev)
        double beta = r_nrm2_sqr;
        err = mtxvector_ddot(r, r, &r_nrm2_sqr, num_flops);
        if (err) return err;
        beta = r_nrm2_sqr / beta;

        // check for convergence
        r_nrm2 = sqrt(r_nrm2_sqr);
        if (out_r_nrm2) *out_r_nrm2 = r_nrm2;
        if (num_iterations) (*num_iterations)++;
        if (r_nrm2 <= rtol || r_nrm2 <= atol) return MTX_SUCCESS;

        // p = beta*p + r
        err = mtxvector_daypx(beta, p, r, num_flops);
        if (err) return err;
    }

    return MTX_ERR_NOT_CONVERGED;
}
