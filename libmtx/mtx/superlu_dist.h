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
 * Direct solvers for linear systems of equations for matrices and
 * vectors in Matrix Market format.
 */

#ifndef LIBMTX_MTX_SUPERLU_DIST_H
#define LIBMTX_MTX_SUPERLU_DIST_H

#include <libmtx/libmtx-config.h>

#ifdef LIBMTX_HAVE_MPI
#include <mpi.h>

#include <stdbool.h>
#include <stdio.h>

struct mtx;

/**
 * `mtx_superlu_dist_fact' is used to select whether or not to reuse
 * factorisations, row or column permutations when using SuperLU_DIST.
 */
enum mtx_superlu_dist_fact
{
    mtx_superlu_dist_fact_DOFACT,
    mtx_superlu_dist_fact_SamePattern,
    mtx_superlu_dist_fact_SamePattern_SameRowPerm,
    mtx_superlu_dist_fact_FACTORED,
};

/**
 * `mtx_superlu_dist_fact_str()` is a string representing a value of
 * type `mtx_superlu_dist_fact'.
 */
const char * mtx_superlu_dist_fact_str(
    enum mtx_superlu_dist_fact fact);

/**
 * `mtx_superlu_dist_fact_parse()' parses a string to obtain a value
 * of type `mtx_superlu_dist_fact'.
 *
 * If `endptr' is not `NULL', the address stored in `endptr' points to
 * the first character beyond the characters that were consumed during
 * parsing.
 *
 * `valid_delimiters' is either `NULL', in which case it is ignored,
 * or, it may contain a string of characters that constitute valid
 * delimiters for the parsed string.  That is, after parsing a value,
 * if there are any remaining, unconsumed characters in the string,
 * the next character in the string is checked to see if it is found
 * in `valid_delimiters'.  If the character is not found, then the
 * string is judged to be invalid, `errno' is set to `EINVAL' and
 * `MTX_ERR_ERRNO' is returned.  Otherwise, the final delimiter
 * character is consumed by the parser.
 *
 * On success, `MTX_SUCCESS' is returned.  Otherwise, if the input
 * contained invalid characters, `errno' is set to `EINVAL' and
 * `MTX_ERR_ERRNO' is returned.
 */
int mtx_superlu_dist_fact_parse(
    const char * s,
    enum mtx_superlu_dist_fact * fact,
    const char ** endptr,
    const char * valid_delimiters);

/**
 * `mtx_superlu_dist_colperm' is used to select a column permutation
 * when using SuperLU_DIST.
 */
enum mtx_superlu_dist_colperm
{
    mtx_superlu_dist_colperm_NATURAL,
    mtx_superlu_dist_colperm_METIS_AT_PLUS_A,
    mtx_superlu_dist_colperm_PARMETIS,
    mtx_superlu_dist_colperm_MMD_ATA,
    mtx_superlu_dist_colperm_MMD_AT_PLUS_A,
    mtx_superlu_dist_colperm_COLAMD,
    mtx_superlu_dist_colperm_MY_PERMC,
};

/**
 * `mtx_superlu_dist_colperm_str()` is a string representing a value
 * of type `mtx_superlu_dist_colperm'.
 */
const char * mtx_superlu_dist_colperm_str(
    enum mtx_superlu_dist_colperm colperm);

/**
 * `mtx_superlu_dist_colperm_parse()' parses a string to obtain a
 * value of type `mtx_superlu_dist_colperm'.
 *
 * If `endptr' is not `NULL', the address stored in `endptr' points to
 * the first character beyond the characters that were consumed during
 * parsing.
 *
 * `valid_delimiters' is either `NULL', in which case it is ignored,
 * or, it may contain a string of characters that constitute valid
 * delimiters for the parsed string.  That is, after parsing a value,
 * if there are any remaining, unconsumed characters in the string,
 * the next character in the string is checked to see if it is found
 * in `valid_delimiters'.  If the character is not found, then the
 * string is judged to be invalid, `errno' is set to `EINVAL' and
 * `MTX_ERR_ERRNO' is returned.  Otherwise, the final delimiter
 * character is consumed by the parser.
 *
 * On success, `MTX_SUCCESS' is returned.  Otherwise, if the input
 * contained invalid characters, `errno' is set to `EINVAL' and
 * `MTX_ERR_ERRNO' is returned.
 */
int mtx_superlu_dist_colperm_parse(
    const char * s,
    enum mtx_superlu_dist_colperm * colperm,
    const char ** endptr,
    const char * valid_delimiters);

/**
 * `mtx_superlu_dist_rowperm' is used to select a row permutation when
 * using SuperLU_DIST.
 */
enum mtx_superlu_dist_rowperm
{
    mtx_superlu_dist_rowperm_NO,
    mtx_superlu_dist_rowperm_LargeDiag_MC64,
    mtx_superlu_dist_rowperm_LargeDiag_AWPM,
    mtx_superlu_dist_rowperm_MY_PERMR,
};

/**
 * `mtx_superlu_dist_rowperm_str()` is a string representing a value
 * of type `mtx_superlu_dist_rowperm'.
 */
const char * mtx_superlu_dist_rowperm_str(
    enum mtx_superlu_dist_rowperm rowperm);

/**
 * `mtx_superlu_dist_rowperm_parse()' parses a string to obtain a
 * value of type `mtx_superlu_dist_rowperm'.
 *
 * If `endptr' is not `NULL', the address stored in `endptr' points to
 * the first character beyond the characters that were consumed during
 * parsing.
 *
 * `valid_delimiters' is either `NULL', in which case it is ignored,
 * or, it may contain a string of characters that constitute valid
 * delimiters for the parsed string.  That is, after parsing a value,
 * if there are any remaining, unconsumed characters in the string,
 * the next character in the string is checked to see if it is found
 * in `valid_delimiters'.  If the character is not found, then the
 * string is judged to be invalid, `errno' is set to `EINVAL' and
 * `MTX_ERR_ERRNO' is returned.  Otherwise, the final delimiter
 * character is consumed by the parser.
 *
 * On success, `MTX_SUCCESS' is returned.  Otherwise, if the input
 * contained invalid characters, `errno' is set to `EINVAL' and
 * `MTX_ERR_ERRNO' is returned.
 */
int mtx_superlu_dist_rowperm_parse(
    const char * s,
    enum mtx_superlu_dist_rowperm * rowperm,
    const char ** endptr,
    const char * valid_delimiters);

/**
 * `mtx_superlu_dist_iterrefine' is used to select an iterative
 * refinement strategy when using SuperLU_DIST.
 */
enum mtx_superlu_dist_iterrefine
{
    mtx_superlu_dist_iterrefine_NO,
    mtx_superlu_dist_iterrefine_SINGLE,
    mtx_superlu_dist_iterrefine_DOUBLE,
};

/**
 * `mtx_superlu_dist_iterrefine_str()` is a string representing a
 * value of type `mtx_superlu_dist_iterrefine'.
 */
const char * mtx_superlu_dist_iterrefine_str(
    enum mtx_superlu_dist_iterrefine iterrefine);

/**
 * `mtx_superlu_dist_iterrefine_parse()' parses a string to obtain a
 * value of type `mtx_superlu_dist_iterrefine'.
 *
 * If `endptr' is not `NULL', the address stored in `endptr' points to
 * the first character beyond the characters that were consumed during
 * parsing.
 *
 * `valid_delimiters' is either `NULL', in which case it is ignored,
 * or, it may contain a string of characters that constitute valid
 * delimiters for the parsed string.  That is, after parsing a value,
 * if there are any remaining, unconsumed characters in the string,
 * the next character in the string is checked to see if it is found
 * in `valid_delimiters'.  If the character is not found, then the
 * string is judged to be invalid, `errno' is set to `EINVAL' and
 * `MTX_ERR_ERRNO' is returned.  Otherwise, the final delimiter
 * character is consumed by the parser.
 *
 * On success, `MTX_SUCCESS' is returned.  Otherwise, if the input
 * contained invalid characters, `errno' is set to `EINVAL' and
 * `MTX_ERR_ERRNO' is returned.
 */
int mtx_superlu_dist_iterrefine_parse(
    const char * s,
    enum mtx_superlu_dist_iterrefine * iterrefine,
    const char ** endptr,
    const char * valid_delimiters);

/**
 * `mtx_superlu_dist_trans' is used to select whether to solve a
 * transposed system when using SuperLU_DIST.
 */
enum mtx_superlu_dist_trans
{
    mtx_superlu_dist_trans_NOTRANS,
    mtx_superlu_dist_trans_TRANS,
    mtx_superlu_dist_trans_CONJ,
};

/**
 * `mtx_superlu_dist_trans_str()` is a string representing a value of
 * type `mtx_superlu_dist_trans'.
 */
const char * mtx_superlu_dist_trans_str(
    enum mtx_superlu_dist_trans trans);

/**
 * `mtx_superlu_dist_trans_parse()' parses a string to obtain a value
 * of type `mtx_superlu_dist_trans'.
 *
 * If `endptr' is not `NULL', the address stored in `endptr' points to
 * the first character beyond the characters that were consumed during
 * parsing.
 *
 * `valid_delimiters' is either `NULL', in which case it is ignored,
 * or, it may contain a string of characters that constitute valid
 * delimiters for the parsed string.  That is, after parsing a value,
 * if there are any remaining, unconsumed characters in the string,
 * the next character in the string is checked to see if it is found
 * in `valid_delimiters'.  If the character is not found, then the
 * string is judged to be invalid, `errno' is set to `EINVAL' and
 * `MTX_ERR_ERRNO' is returned.  Otherwise, the final delimiter
 * character is consumed by the parser.
 *
 * On success, `MTX_SUCCESS' is returned.  Otherwise, if the input
 * contained invalid characters, `errno' is set to `EINVAL' and
 * `MTX_ERR_ERRNO' is returned.
 */
int mtx_superlu_dist_trans_parse(
    const char * s,
    enum mtx_superlu_dist_trans * trans,
    const char ** endptr,
    const char * valid_delimiters);

/**
 * `mtx_superlu_dist_solve_global()' solves the linear system `Ax=b'
 * using an LU factorisation-based direct linear solver from
 * SuperLU_DIST. The matrix and right-hand side are replicated
 * globally on every MPI process in the communicator `comm'.
 *
 * Note that MPI must have been initialised prior to calling
 * `mtx_superlu_dist_solve_global()'. SuperLU_DIST uses a 2D process
 * grid with dimensions given by `num_process_rows' and
 * `num_process_columns'.  These processes must belong to the MPI
 * communicator `comm'.
 *
 * The matrix `A' must already be sorted in row-major order.
 *
 * Most of the remaining arguments relate to various SuperLU_DIST
 * options that control how the linear system is solved.  Note that
 * these arguments are named just like the corresponding options in
 * SuperLU_DIST (using Camel case). The relevant options are described
 * in the SuperLU User Guide (Li et. al. (2018)).
 *
 * The following values are normally used as defaults by SuperLU_DIST:
 *   - Fact = DOFACT
 *   - Equil = YES
 *   - ParSymbFact = NO
 *   - ColPerm = METIS_AT_PLUS_A if ParMETIS is available, or
 *               MMD_AT_PLUS_A otherwise
 *   - RowPerm = LargeDiag_MC64
 *   - ReplaceTinyPivot = NO
 *   - IterRefine = SLU_DOUBLE
 *   - Trans = NOTRANS
 *   - SolveInitialized = NO
 *   - RefineInitialized = NO
 *   - PrintStat = YES
 *   - num_lookaheads = 10
 *   - lookahead_etree = NO
 *   - SymPattern = NO
 *
 * Based on the above defaults, most users will wish to use
 * `mtx_superlu_dist_solve_global()' as follows:
 *
 *   int err = mtx_superlu_dist_solve_global(
 *       A, b, x, verbose, stderr, MPI_COMM_WORLD,
 *       0, num_process_rows, num_process_cols,
 *       mtx_superlu_dist_fact_DOFACT,
 *       true, false,
 *       mtx_superlu_dist_colperm_MMD_AT_PLUS_A,
 *       mtx_superlu_dist_rowperm_LargeDiag_MC64,
 *       false,
 *       mtx_superlu_dist_iterrefine_DOUBLE,
 *       mtx_superlu_dist_trans_NOTRANS,
 *       false, false, true, 10, false, false);
 *
 *
 * References:
 *
 *   Xiaoye S. Li, James W. Demmel, John R. Gilbert, Laura Grigori,
 *   Piyush Sao, Meiyue Shao and Ichitaro Yamazaki. "SuperLU Users'
 *   Guide". June 2018.
 */
int mtx_superlu_dist_solve_global(
    const struct mtx * A,
    const struct mtx * b,
    struct mtx * x,
    int verbose,
    FILE * f,
    int * mpierr,
    MPI_Comm comm,
    int root,
    int num_process_rows,
    int num_process_columns,
    enum mtx_superlu_dist_fact Fact,
    bool Equil,
    bool ParSymbFact,
    enum mtx_superlu_dist_colperm ColPerm,
    enum mtx_superlu_dist_rowperm RowPerm,
    bool ReplaceTinyPivot,
    enum mtx_superlu_dist_iterrefine IterRefine,
    enum mtx_superlu_dist_trans Trans,
    bool SolveInitialized,
    bool RefineInitialized,
    bool PrintStat,
    int num_lookaheads,
    bool lookahead_etree,
    bool SymPattern);

/**
 * `mtx_superlu_dist_solve_distributed()' solves the linear system
 * `Ax=b' using an LU factorisation-based direct linear solver from
 * SuperLU_DIST. The matrix and right-hand side are distributed across
 * MPI processes in the communicator `comm', such that each process
 * owns a block of consecutive rows of `A' and `b'.
 *
 * Note that MPI must have been initialised prior to calling
 * `mtx_superlu_dist_solve_distributed()'. SuperLU_DIST uses a 2D
 * process grid with dimensions given by `num_process_rows' and
 * `num_process_columns'.  These processes must belong to the MPI
 * communicator `comm'.
 *
 * The matrix `A' must already be sorted in row-major order.
 *
 * Most of the remaining arguments relate to various SuperLU_DIST
 * options that control how the linear system is solved.  Note that
 * these arguments are named just like the corresponding options in
 * SuperLU_DIST (using Camel case). The relevant options are described
 * in the SuperLU User Guide (Li et. al. (2018)).
 *
 * The following values are normally used as defaults by SuperLU_DIST:
 *   - Fact = DOFACT
 *   - Equil = YES
 *   - ParSymbFact = NO
 *   - ColPerm = METIS_AT_PLUS_A if ParMETIS is available, or
 *               MMD_AT_PLUS_A otherwise
 *   - RowPerm = LargeDiag_MC64
 *   - ReplaceTinyPivot = NO
 *   - IterRefine = SLU_DOUBLE
 *   - Trans = NOTRANS
 *   - SolveInitialized = NO
 *   - RefineInitialized = NO
 *   - PrintStat = YES
 *   - num_lookaheads = 10
 *   - lookahead_etree = NO
 *   - SymPattern = NO
 *
 * Based on the above defaults, most users will wish to use
 * `mtx_superlu_dist_solve_distributed()' as follows:
 *
 *   int err = mtx_superlu_dist_solve_distributed(
 *       A, b, x, verbose, stderr, MPI_COMM_WORLD,
 *       0, num_process_rows, num_process_cols,
 *       mtx_superlu_dist_fact_DOFACT,
 *       true, false,
 *       mtx_superlu_dist_colperm_MMD_AT_PLUS_A,
 *       mtx_superlu_dist_rowperm_LargeDiag_MC64,
 *       false,
 *       mtx_superlu_dist_iterrefine_DOUBLE,
 *       mtx_superlu_dist_trans_NOTRANS,
 *       false, false, true, 10, false, false);
 *
 *
 * References:
 *
 *   Xiaoye S. Li, James W. Demmel, John R. Gilbert, Laura Grigori,
 *   Piyush Sao, Meiyue Shao and Ichitaro Yamazaki. "SuperLU Users'
 *   Guide". June 2018.
 */
int mtx_superlu_dist_solve_distributed(
    const struct mtx * A,
    const struct mtx * b,
    struct mtx * x,
    int verbose,
    FILE * f,
    int * mpierr,
    MPI_Comm comm,
    int root,
    int num_process_rows,
    int num_process_columns,
    enum mtx_superlu_dist_fact Fact,
    bool Equil,
    bool ParSymbFact,
    enum mtx_superlu_dist_colperm ColPerm,
    enum mtx_superlu_dist_rowperm RowPerm,
    bool ReplaceTinyPivot,
    enum mtx_superlu_dist_iterrefine IterRefine,
    enum mtx_superlu_dist_trans Trans,
    bool SolveInitialized,
    bool RefineInitialized,
    bool PrintStat,
    int num_lookaheads,
    bool lookahead_etree,
    bool SymPattern);
#endif

#endif
