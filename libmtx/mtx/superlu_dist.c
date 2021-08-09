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

#include <libmtx/libmtx-config.h>

#ifdef LIBMTX_HAVE_MPI
#include <libmtx/error.h>
#include <libmtx/matrix/matrix.h>
#include <libmtx/mtx/mtx.h>
#include <libmtx/mtx/superlu_dist.h>

#include <superlu_defs.h>
#include <superlu_ddefs.h>

#include <mpi.h>

#include <errno.h>
#include <fcntl.h>
#include <unistd.h>

#include <stdbool.h>
#include <stdio.h>

/**
 * `mtx_superlu_dist_fact_str()` is a string representing a value of
 * type `mtx_superlu_dist_fact'.
 */
const char * mtx_superlu_dist_fact_str(
    enum mtx_superlu_dist_fact fact)
{
    switch (fact) {
    case mtx_superlu_dist_fact_DOFACT: return "DOFACT";
    case mtx_superlu_dist_fact_SamePattern: return "SamePattern";
    case mtx_superlu_dist_fact_SamePattern_SameRowPerm: return "SamePattern_SameRowPerm";
    case mtx_superlu_dist_fact_FACTORED: return "FACTORED";
    default: return "unknown";
    }
}

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
 * On success, `parse_mtx_superlu_dist_fact()' returns `MTX_SUCCESS'.
 * Otherwise, if the input contained invalid characters, `errno' is
 * set to `EINVAL' and `MTX_ERR_ERRNO' is returned.
 */
int mtx_superlu_dist_fact_parse(
    const char * s,
    enum mtx_superlu_dist_fact * fact,
    const char ** endptr,
    const char * valid_delimiters)
{
    const char * s_end;
    if (strcmp(s, "DOFACT") == 0) {
        *fact = mtx_superlu_dist_fact_DOFACT;
        s_end = s + strlen("DOFACT");
    } else if (strcmp(s, "SamePattern") == 0) {
        *fact = mtx_superlu_dist_fact_SamePattern;
        s_end = s + strlen("SamePattern");
    } else if (strcmp(s, "SamePattern_SameRowPerm") == 0) {
        *fact = mtx_superlu_dist_fact_SamePattern_SameRowPerm;
        s_end = s + strlen("SamePattern_SameRowPerm");
    } else if (strcmp(s, "FACTORED") == 0) {
        *fact = mtx_superlu_dist_fact_FACTORED;
        s_end = s + strlen("FACTORED");
    } else {
        errno = EINVAL;
        return MTX_ERR_ERRNO;
    }

    /* Check for a valid delimiter following the parsed string. */
    if (valid_delimiters && s_end && *s_end != '\0') {
        if (!strchr(valid_delimiters, *s_end)) {
            errno = EINVAL;
            return MTX_ERR_ERRNO;
        }
        s_end++;
    }
    if (endptr)
        *endptr = s_end;
    return MTX_SUCCESS;
}

/**
 * `mtx_superlu_dist_colperm_str()` is a string representing a value
 * of type `mtx_superlu_dist_colperm'.
 */
const char * mtx_superlu_dist_colperm_str(
    enum mtx_superlu_dist_colperm colperm)
{
    switch (colperm) {
    case mtx_superlu_dist_colperm_NATURAL: return "NATURAL";
    case mtx_superlu_dist_colperm_METIS_AT_PLUS_A: return "METIS_AT_PLUS_A";
    case mtx_superlu_dist_colperm_PARMETIS: return "PARMETIS";
    case mtx_superlu_dist_colperm_MMD_ATA: return "MMD_ATA";
    case mtx_superlu_dist_colperm_MMD_AT_PLUS_A: return "MMD_AT_PLUS_A";
    case mtx_superlu_dist_colperm_COLAMD: return "COLAMD";
    case mtx_superlu_dist_colperm_MY_PERMC: return "MY_PERMC";
    default: return "unknown";
    }
}

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
 * On success, `parse_mtx_superlu_dist_colperm()' returns
 * `MTX_SUCCESS'.  Otherwise, if the input contained invalid
 * characters, `errno' is set to `EINVAL' and `MTX_ERR_ERRNO' is
 * returned.
 */
int mtx_superlu_dist_colperm_parse(
    const char * s,
    enum mtx_superlu_dist_colperm * colperm,
    const char ** endptr,
    const char * valid_delimiters)
{
    const char * s_end;
    if (strcmp(s, "NATURAL") == 0) {
        *colperm = mtx_superlu_dist_colperm_NATURAL;
        s_end = s + strlen("NATURAL");
    } else if (strcmp(s, "METIS_AT_PLUS_A") == 0) {
        *colperm = mtx_superlu_dist_colperm_METIS_AT_PLUS_A;
        s_end = s + strlen("METIS_AT_PLUS_A");
    } else if (strcmp(s, "PARMETIS") == 0) {
        *colperm = mtx_superlu_dist_colperm_PARMETIS;
        s_end = s + strlen("PARMETIS");
    } else if (strcmp(s, "MMD_ATA") == 0) {
        *colperm = mtx_superlu_dist_colperm_MMD_ATA;
        s_end = s + strlen("MMD_ATA");
    } else if (strcmp(s, "MMD_AT_PLUS_A") == 0) {
        *colperm = mtx_superlu_dist_colperm_MMD_AT_PLUS_A;
        s_end = s + strlen("MMD_AT_PLUS_A");
    } else if (strcmp(s, "COLAMD") == 0) {
        *colperm = mtx_superlu_dist_colperm_COLAMD;
        s_end = s + strlen("COLAMD");
    } else if (strcmp(s, "MY_PERMC") == 0) {
        *colperm = mtx_superlu_dist_colperm_MY_PERMC;
        s_end = s + strlen("MY_PERMC");
    } else {
        errno = EINVAL;
        return MTX_ERR_ERRNO;
    }

    /* Check for a valid delimiter following the parsed string. */
    if (valid_delimiters && s_end && *s_end != '\0') {
        if (!strchr(valid_delimiters, *s_end)) {
            errno = EINVAL;
            return MTX_ERR_ERRNO;
        }
        s_end++;
    }
    if (endptr)
        *endptr = s_end;
    return MTX_SUCCESS;
}

/**
 * `mtx_superlu_dist_rowperm_str()` is a string representing a value
 * of type `mtx_superlu_dist_rowperm'.
 */
const char * mtx_superlu_dist_rowperm_str(
    enum mtx_superlu_dist_rowperm rowperm)
{
    switch (rowperm) {
    case mtx_superlu_dist_rowperm_NO: return "NO";
    case mtx_superlu_dist_rowperm_LargeDiag_MC64: return "LargeDiag_MC64";
    case mtx_superlu_dist_rowperm_LargeDiag_AWPM: return "LargeDiag_AWPM";
    case mtx_superlu_dist_rowperm_MY_PERMR: return "MY_PERMR";
    default: return "unknown";
    }
}

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
 * On success, `parse_mtx_superlu_dist_rowperm()' returns
 * `MTX_SUCCESS'.  Otherwise, if the input contained invalid
 * characters, `errno' is set to `EINVAL' and `MTX_ERR_ERRNO' is
 * returned.
 */
int mtx_superlu_dist_rowperm_parse(
    const char * s,
    enum mtx_superlu_dist_rowperm * rowperm,
    const char ** endptr,
    const char * valid_delimiters)
{
    const char * s_end;
    if (strcmp(s, "NO") == 0) {
        *rowperm = mtx_superlu_dist_rowperm_NO;
        s_end = s + strlen("NO");
    } else if (strcmp(s, "LargeDiag_MC64") == 0) {
        *rowperm = mtx_superlu_dist_rowperm_LargeDiag_MC64;
        s_end = s + strlen("LargeDiag_MC64");
    } else if (strcmp(s, "LargeDiag_AWPM") == 0) {
        *rowperm = mtx_superlu_dist_rowperm_LargeDiag_AWPM;
        s_end = s + strlen("LargeDiag_AWPM");
    } else if (strcmp(s, "MY_PERMR") == 0) {
        *rowperm = mtx_superlu_dist_rowperm_MY_PERMR;
        s_end = s + strlen("MY_PERMR");
    } else {
        errno = EINVAL;
        return MTX_ERR_ERRNO;
    }

    /* Check for a valid delimiter following the parsed string. */
    if (valid_delimiters && s_end && *s_end != '\0') {
        if (!strchr(valid_delimiters, *s_end)) {
            errno = EINVAL;
            return MTX_ERR_ERRNO;
        }
        s_end++;
    }
    if (endptr)
        *endptr = s_end;
    return MTX_SUCCESS;
}

/**
 * `mtx_superlu_dist_iterrefine_str()` is a string representing a
 * value of type `mtx_superlu_dist_iterrefine'.
 */
const char * mtx_superlu_dist_iterrefine_str(
    enum mtx_superlu_dist_iterrefine iterrefine)
{
    switch (iterrefine) {
    case mtx_superlu_dist_iterrefine_NO: return "NO";
    case mtx_superlu_dist_iterrefine_SINGLE: return "SINGLE";
    case mtx_superlu_dist_iterrefine_DOUBLE: return "DOUBLE";
    default: return "unknown";
    }
}

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
 * On success, `parse_mtx_superlu_dist_iterrefine()' returns
 * `MTX_SUCCESS'.  Otherwise, if the input contained invalid
 * characters, `errno' is set to `EINVAL' and `MTX_ERR_ERRNO' is
 * returned.
 */
int mtx_superlu_dist_iterrefine_parse(
    const char * s,
    enum mtx_superlu_dist_iterrefine * iterrefine,
    const char ** endptr,
    const char * valid_delimiters)
{
    const char * s_end;
    if (strcmp(s, "NO") == 0) {
        *iterrefine = mtx_superlu_dist_iterrefine_NO;
        s_end = s + strlen("NO");
    } else if (strcmp(s, "SINGLE") == 0) {
        *iterrefine = mtx_superlu_dist_iterrefine_SINGLE;
        s_end = s + strlen("SINGLE");
    } else if (strcmp(s, "DOUBLE") == 0) {
        *iterrefine = mtx_superlu_dist_iterrefine_DOUBLE;
        s_end = s + strlen("DOUBLE");
    } else {
        errno = EINVAL;
        return MTX_ERR_ERRNO;
    }

    /* Check for a valid delimiter following the parsed string. */
    if (valid_delimiters && s_end && *s_end != '\0') {
        if (!strchr(valid_delimiters, *s_end)) {
            errno = EINVAL;
            return MTX_ERR_ERRNO;
        }
        s_end++;
    }
    if (endptr)
        *endptr = s_end;
    return MTX_SUCCESS;
}

/**
 * `mtx_superlu_dist_trans_str()` is a string representing a value of
 * type `mtx_superlu_dist_trans'.
 */
const char * mtx_superlu_dist_trans_str(
    enum mtx_superlu_dist_trans trans)
{
    switch (trans) {
    case mtx_superlu_dist_trans_NOTRANS: return "NOTRANS";
    case mtx_superlu_dist_trans_TRANS: return "TRANS";
    case mtx_superlu_dist_trans_CONJ: return "CONJ";
    default: return "unknown";
    }
}

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
 * On success, `parse_mtx_superlu_dist_trans()' returns `MTX_SUCCESS'.
 * Otherwise, if the input contained invalid characters, `errno' is
 * set to `EINVAL' and `MTX_ERR_ERRNO' is returned.
 */
int mtx_superlu_dist_trans_parse(
    const char * s,
    enum mtx_superlu_dist_trans * trans,
    const char ** endptr,
    const char * valid_delimiters)
{
    const char * s_end;
    if (strcmp(s, "NOTRANS") == 0) {
        *trans = mtx_superlu_dist_trans_NOTRANS;
        s_end = s + strlen("NOTRANS");
    } else if (strcmp(s, "TRANS") == 0) {
        *trans = mtx_superlu_dist_trans_TRANS;
        s_end = s + strlen("TRANS");
    } else if (strcmp(s, "CONJ") == 0) {
        *trans = mtx_superlu_dist_trans_CONJ;
        s_end = s + strlen("CONJ");
    } else {
        errno = EINVAL;
        return MTX_ERR_ERRNO;
    }

    /* Check for a valid delimiter following the parsed string. */
    if (valid_delimiters && s_end && *s_end != '\0') {
        if (!strchr(valid_delimiters, *s_end)) {
            errno = EINVAL;
            return MTX_ERR_ERRNO;
        }
        s_end++;
    }
    if (endptr)
        *endptr = s_end;
    return MTX_SUCCESS;
}

/**
 * `superlu_dist_options_set()' configures options for SuperLU_DIST
 * based on the given settings.
 */
static int superlu_dist_options_set(
    superlu_dist_options_t * options,
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
    bool SymPattern)
{
    set_default_options_dist(options);

    if (Fact == mtx_superlu_dist_fact_DOFACT) {
        options->Fact = DOFACT;
    } else if (Fact == mtx_superlu_dist_fact_SamePattern) {
        options->Fact = SamePattern;
    } else if (Fact == mtx_superlu_dist_fact_SamePattern_SameRowPerm) {
        options->Fact = SamePattern_SameRowPerm;
    } else if (Fact == mtx_superlu_dist_fact_FACTORED) {
        options->Fact = FACTORED;
    } else {
        errno = EINVAL;
        return MTX_ERR_ERRNO;
    }

    options->Equil = Equil ? YES : NO;
    options->ParSymbFact = ParSymbFact ? YES : NO;

    if (ColPerm == mtx_superlu_dist_colperm_NATURAL) {
        options->ColPerm = NATURAL;
    } else if (ColPerm == mtx_superlu_dist_colperm_METIS_AT_PLUS_A && !ParSymbFact) {
        options->ColPerm = METIS_AT_PLUS_A;
    } else if (ColPerm == mtx_superlu_dist_colperm_PARMETIS) {
        options->ColPerm = PARMETIS;
    } else if (ColPerm == mtx_superlu_dist_colperm_MMD_ATA && !ParSymbFact) {
        options->ColPerm = MMD_ATA;
    } else if (ColPerm == mtx_superlu_dist_colperm_MMD_AT_PLUS_A && !ParSymbFact) {
        options->ColPerm = MMD_AT_PLUS_A;
    } else if (ColPerm == mtx_superlu_dist_colperm_COLAMD && !ParSymbFact) {
        options->ColPerm = COLAMD;
    } else if (ColPerm == mtx_superlu_dist_colperm_MY_PERMC) {
        options->ColPerm = MY_PERMC;

        /* TODO: Enable user-specified column permutation. */
        errno = EINVAL;
        return MTX_ERR_ERRNO;
    } else {
        errno = EINVAL;
        return MTX_ERR_ERRNO;
    }

    if (RowPerm == mtx_superlu_dist_rowperm_NO) {
        options->RowPerm = NO;
    } else if (RowPerm == mtx_superlu_dist_rowperm_LargeDiag_MC64) {
        options->RowPerm = LargeDiag_MC64;
    } else if (RowPerm == mtx_superlu_dist_rowperm_LargeDiag_AWPM) {
        options->RowPerm = LargeDiag_AWPM;
    } else if (RowPerm == mtx_superlu_dist_rowperm_MY_PERMR) {
        options->RowPerm = MY_PERMR;

        /* TODO: Enable user-specified row permutation. */
        errno = EINVAL;
        return MTX_ERR_ERRNO;
    } else {
        errno = EINVAL;
        return MTX_ERR_ERRNO;
    }

    options->ReplaceTinyPivot = ReplaceTinyPivot ? YES : NO;

    if (IterRefine == mtx_superlu_dist_iterrefine_NO) {
        options->IterRefine = NO;
    } else if (IterRefine == mtx_superlu_dist_iterrefine_SINGLE) {
        options->IterRefine = SLU_SINGLE;
    } else if (IterRefine == mtx_superlu_dist_iterrefine_DOUBLE) {
        options->IterRefine = SLU_DOUBLE;
    } else {
        errno = EINVAL;
        return MTX_ERR_ERRNO;
    }

    if (Trans == mtx_superlu_dist_trans_NOTRANS) {
        options->Trans = NOTRANS;
    } else if (Trans == mtx_superlu_dist_trans_TRANS) {
        options->Trans = TRANS;
    } else if (Trans == mtx_superlu_dist_trans_CONJ) {
        options->Trans = CONJ;
    } else {
        errno = EINVAL;
        return MTX_ERR_ERRNO;
    }

    options->SolveInitialized = SolveInitialized ? YES : NO;
    options->RefineInitialized = RefineInitialized ? YES : NO;
    options->PrintStat = PrintStat ? YES : NO;
    options->num_lookaheads = num_lookaheads;
    options->lookahead_etree = lookahead_etree ? YES : NO;
    options->SymPattern = SymPattern ? YES : NO;
    return MTX_SUCCESS;
}

/**
 * `mtx_matrix_indices_to_csc()' converts the index information of a
 * sparse matrix in coordinate format to an array of column pointers
 * and another array of row indices, as in the compressed sparse
 * column (CSC) format.
 */
static int mtx_matrix_indices_to_csc(
    const struct mtx * A,
    int32_t ** out_column_ptr,
    int ** out_row_indices)
{
    int err;
    if (A->object != mtx_matrix)
        return MTX_ERR_INVALID_MTX_OBJECT;
    if (A->format != mtx_coordinate)
        return MTX_ERR_INVALID_MTX_FORMAT;
    if (A->sorting != mtx_column_major)
        return MTX_ERR_INVALID_MTX_SORTING;

    /* SuperLU_DIST only support 32-bit integer column pointers. */
    if (A->size > INT32_MAX) {
        errno = ERANGE;
        return MTX_ERR_ERRNO;
    }

    /* Compute column pointers. */
    int64_t * column_ptr64 = malloc((A->num_columns+1) * sizeof(int64_t));
    if (!column_ptr64)
        return MTX_ERR_ERRNO;
    err = mtx_matrix_column_ptr(A, column_ptr64);
    if (err) {
        free(column_ptr64);
        return err;
    }

    /* Copy column pointers to 32-bit integers, as required by SuperLU_DIST. */
    int32_t * column_ptr = malloc((A->num_columns+1) * sizeof(int32_t));
    if (!column_ptr) {
        free(column_ptr64);
        return MTX_ERR_ERRNO;
    }
    for (int i = 0; i <= A->num_columns; i++) {
        if (column_ptr64[i] > INT32_MAX) {
            free(column_ptr);
            free(column_ptr64);
            errno = ERANGE;
            return MTX_ERR_ERRNO;
        }

        column_ptr[i] = (int32_t) column_ptr64[i];
    }
    free(column_ptr64);

    /* Extract column indices. */
    int * row_indices = malloc(A->size * sizeof(int));
    if (!row_indices) {
        free(column_ptr);
        return MTX_ERR_ERRNO;
    }
    err = mtx_matrix_row_indices(A, row_indices);
    if (err) {
        free(row_indices);
        free(column_ptr);
        return err;
    }
    /* Subtract one to obtain zero-based indices. */
    for (int64_t k = 0; k < A->size; k++)
        row_indices[k]--;

    *out_column_ptr = column_ptr;
    *out_row_indices = row_indices;
    return MTX_SUCCESS;
}

/**
 * `mtx_matrix_to_csc_coordinate_double()' converts a sparse matrix in
 * coordinate format to an array of row pointers, another array of
 * column indices, and a third array of double precision floating
 * point values, as in the compressed sparse row (CSC) format.
 */
static int mtx_matrix_to_csc_coordinate_double(
    const struct mtx * A,
    int32_t ** out_column_ptr,
    int ** out_row_indices,
    double ** out_values)
{
    int err;
    if (A->object != mtx_matrix)
        return MTX_ERR_INVALID_MTX_OBJECT;
    if (A->format != mtx_coordinate)
        return MTX_ERR_INVALID_MTX_FORMAT;
    if (A->sorting != mtx_column_major)
        return MTX_ERR_INVALID_MTX_SORTING;

    err = mtx_matrix_indices_to_csc(
        A, out_column_ptr, out_row_indices);
    if (err)
        return err;

    /* Extract nonzero values. */
    double * values = malloc(A->size * sizeof(double));
    if (!values) {
        free(*out_row_indices);
        free(*out_column_ptr);
        return MTX_ERR_ERRNO;
    }
    err = mtx_matrix_data_double(A, values);
    if (err) {
        free(values);
        free(*out_row_indices);
        free(*out_column_ptr);
        return err;
    }
    *out_values = values;
    return MTX_SUCCESS;
}

/**
 * `mtx_matrix_indices_to_csr()' converts the index information of a
 * sparse matrix in coordinate format to an array of row pointers and
 * another array of column indices, as in the compressed sparse row
 * (CSR) format.
 */
static int mtx_matrix_indices_to_csr(
    const struct mtx * A,
    int32_t ** out_row_ptr,
    int ** out_column_indices)
{
    int err;
    if (A->object != mtx_matrix)
        return MTX_ERR_INVALID_MTX_OBJECT;
    if (A->format != mtx_coordinate)
        return MTX_ERR_INVALID_MTX_FORMAT;
    if (A->sorting != mtx_row_major)
        return MTX_ERR_INVALID_MTX_SORTING;

    /* SuperLU_DIST only support 32-bit integer row pointers. */
    if (A->size > INT32_MAX) {
        errno = ERANGE;
        return MTX_ERR_ERRNO;
    }

    /* Compute row pointers. */
    int64_t * row_ptr64 = malloc((A->num_rows+1) * sizeof(int64_t));
    if (!row_ptr64)
        return MTX_ERR_ERRNO;
    err = mtx_matrix_row_ptr(A, row_ptr64);
    if (err) {
        free(row_ptr64);
        return err;
    }

    /* Copy row pointers to 32-bit integers, as required by SuperLU_DIST. */
    int32_t * row_ptr = malloc((A->num_rows+1) * sizeof(int32_t));
    if (!row_ptr) {
        free(row_ptr64);
        return MTX_ERR_ERRNO;
    }
    for (int i = 0; i <= A->num_rows; i++) {
        if (row_ptr64[i] > INT32_MAX) {
            free(row_ptr);
            free(row_ptr64);
            errno = ERANGE;
            return MTX_ERR_ERRNO;
        }

        row_ptr[i] = (int32_t) row_ptr64[i];
    }
    free(row_ptr64);

    /* Extract column indices. */
    int * column_indices = malloc(A->size * sizeof(int));
    if (!column_indices) {
        free(row_ptr);
        return MTX_ERR_ERRNO;
    }
    err = mtx_matrix_column_indices(A, column_indices);
    if (err) {
        free(column_indices);
        free(row_ptr);
        return err;
    }
    /* Subtract one to obtain zero-based indices. */
    for (int64_t k = 0; k < A->size; k++)
        column_indices[k]--;

    *out_row_ptr = row_ptr;
    *out_column_indices = column_indices;
    return MTX_SUCCESS;
}

/**
 * `mtx_matrix_to_csr_coordinate_double()' converts a sparse matrix in
 * coordinate format to an array of row pointers, another array of
 * column indices, and a third array of double precision floating
 * point values, as in the compressed sparse row (CSR) format.
 */
static int mtx_matrix_to_csr_coordinate_double(
    const struct mtx * A,
    int32_t ** out_row_ptr,
    int ** out_column_indices,
    double ** out_values)
{
    int err;
    if (A->object != mtx_matrix)
        return MTX_ERR_INVALID_MTX_OBJECT;
    if (A->format != mtx_coordinate)
        return MTX_ERR_INVALID_MTX_FORMAT;
    if (A->sorting != mtx_row_major)
        return MTX_ERR_INVALID_MTX_SORTING;

    err = mtx_matrix_indices_to_csr(
        A, out_row_ptr, out_column_indices);
    if (err)
        return err;

    /* Extract nonzero values. */
    double * values = malloc(A->size * sizeof(double));
    if (!values) {
        free(*out_column_indices);
        free(*out_row_ptr);
        return MTX_ERR_ERRNO;
    }
    err = mtx_matrix_data_double(A, values);
    if (err) {
        free(values);
        free(*out_column_indices);
        free(*out_row_ptr);
        return err;
    }
    *out_values = values;
    return MTX_SUCCESS;
}

/**
 * `mtx_superlu_dist_solve_global_double()' solves the linear system
 * `Ax=b' using an LU factorisation-based direct linear solver from
 * SuperLU_DIST, where `A' is a real matrix and `x' and `b' are real
 * vectors of double precision floating point values.
 */
static int mtx_superlu_dist_solve_global_double(
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
    superlu_dist_options_t * options)
{
#ifdef LIBMTX_HAVE_SUPERLU_DIST
    int err;
    if (A->object != mtx_matrix ||
        b->object != mtx_vector ||
        x->object != mtx_vector)
        return MTX_ERR_INVALID_MTX_OBJECT;
    if (A->format != mtx_coordinate ||
        b->format != mtx_array ||
        x->format != mtx_array)
        return MTX_ERR_INVALID_MTX_FORMAT;
    if (A->field != mtx_double ||
        b->field != mtx_double ||
        x->field != mtx_double)
        return MTX_ERR_INVALID_MTX_FIELD;
    if (A->symmetry != mtx_general)
        return MTX_ERR_INVALID_MTX_SYMMETRY;
    if (A->num_rows != A->num_columns ||
        A->num_rows != b->num_rows ||
        A->num_columns != x->num_rows ||
        b->size != x->size)
        return MTX_ERR_INVALID_MTX_SIZE;
    if (A->sorting != mtx_column_major)
        return MTX_ERR_INVALID_MTX_SORTING;

    /* SuperLU_DIST only supports 32-bit integer row or column
     * pointers. */
    if (A->size > INT32_MAX) {
        errno = ERANGE;
        return MTX_ERR_ERRNO;
    }

    int comm_size;
    *mpierr = MPI_Comm_size(comm, &comm_size);
    if (*mpierr)
        return MTX_ERR_MPI;
    int rank;
    *mpierr = MPI_Comm_rank(comm, &rank);
    if (*mpierr)
        return MTX_ERR_MPI;
    if (comm_size != num_process_rows * num_process_columns)
        return MTX_ERR_SUPERLU_DIST_GRID_SIZE;

    /*
     * Convert the matrix to a compressed sparse row (CSR) format,
     * which requires the matrix to already be sorted in row major
     * order.  Then we compute row pointers and extract column indices
     * and nonzero values into separate arrays.
     *
     * Note that SuperLU_DIST takes ownership of the matrix data, so
     * there is no need to free the arrays `col_ptr', `row_indices'
     * and `values' after calling `dCreate_CompRowLoc_Matrix_dist'.
     * The underlying storage is instead freed with
     * `Destroy_CompRowLoc_Matrix_dist'.
     */
    int32_t * col_ptr;
    int * row_indices;
    double * values;
    err = mtx_matrix_to_csc_coordinate_double(
        A, &col_ptr, &row_indices, &values);
    if (err)
        return err;

    /*
     * SuperLU_DIST uses a single array for the right-hand side and
     * solution vector, where the latter overwrites the former.
     * Therefore, we copy the values from `b' to `x'.
     */
    memmove(x->data, b->data, x->size * x->nonzero_size);

    if (rank == root && verbose) {
        print_options_dist(options);
        print_sp_ienv_dist(options);
    }

    /* Initialize SuperLU process grid. */
    gridinfo_t grid;
    superlu_gridinit(
        comm, num_process_rows,
        num_process_columns, &grid);

    /* Set up the matrix. */
    SuperMatrix super_A;
    Stype_t stype = SLU_NC; /* compressed sparse column format */
    Dtype_t dtype = SLU_D;  /* double precision floating point */
    Mtype_t mtype = SLU_GE; /* general, unsymmetric matrix. */
    dCreate_CompCol_Matrix_dist(
        &super_A, A->num_rows, A->num_columns, A->size,
        values, row_indices, col_ptr,
        stype, dtype, mtype);

    /* Set up data structures for LU factorisation and solver
     * statistics. */
    dScalePermstruct_t scale_perm;
    dLUstruct_t LU;
    SuperLUStat_t stat;
    dScalePermstructInit(
        A->num_rows, A->num_columns, &scale_perm);
    dLUstructInit(A->num_columns, &LU);
    PStatInit(&stat);

    /* Solve the linear system. */
    int num_right_hand_sides = 1;
    double berr;
    pdgssvx_ABglobal(
        options, &super_A, &scale_perm,
        (double *) x->data, x->size, num_right_hand_sides,
        &grid, &LU, &berr, &stat, &err);

    if (verbose)
        PStatPrint(options, &stat, &grid);

    /* Clean up. */
    PStatFree(&stat);
    dDestroy_LU(A->num_columns, &grid, &LU);
    dLUstructFree(&LU);
    dScalePermstructFree(&scale_perm);
    Destroy_CompCol_Matrix_dist(&super_A);
    superlu_gridexit(&grid);
    return MTX_SUCCESS;
#else
    return MTX_ERR_SUPERLU_DIST_NEEDED;
#endif
}

/**
 * `mtx_superlu_dist_solve_distributed_double()' solves the linear
 * system `Ax=b' using an LU factorisation-based direct linear solver
 * from SuperLU_DIST, where `A' is a real matrix and `x' and `b' are
 * real vectors of double precision floating point values.
 */
static int mtx_superlu_dist_solve_distributed_double(
    const struct mtx * A,
    const struct mtx * b,
    struct mtx * x,
    int verbose,
    FILE * f,
    MPI_Comm comm,
    int root,
    int num_process_rows,
    int num_process_columns,
    superlu_dist_options_t * options)
{
#ifdef LIBMTX_HAVE_SUPERLU_DIST
    int err;
    if (A->object != mtx_matrix ||
        b->object != mtx_vector ||
        x->object != mtx_vector)
        return MTX_ERR_INVALID_MTX_OBJECT;
    if (A->format != mtx_coordinate ||
        b->format != mtx_array ||
        x->format != mtx_array)
        return MTX_ERR_INVALID_MTX_FORMAT;
    if (A->field != mtx_double ||
        b->field != mtx_double ||
        x->field != mtx_double)
        return MTX_ERR_INVALID_MTX_FIELD;
    if (A->symmetry != mtx_general)
        return MTX_ERR_INVALID_MTX_SYMMETRY;
    if (A->num_rows != A->num_columns ||
        A->num_rows != b->num_rows ||
        A->num_columns != x->num_rows ||
        b->size != x->size)
        return MTX_ERR_INVALID_MTX_SIZE;
    if (A->sorting != mtx_row_major)
        return MTX_ERR_INVALID_MTX_SORTING;

    /* SuperLU_DIST only support 32-bit integer row pointers. */
    if (A->size > INT32_MAX) {
        errno = ERANGE;
        return MTX_ERR_ERRNO;
    }

    /*
     * Convert the matrix to a compressed sparse row (CSR) format,
     * which requires the matrix to already be sorted in row major
     * order.  Then we compute row pointers and extract column indices
     * and nonzero values into separate arrays.
     *
     * Note that SuperLU_DIST takes ownership of the matrix data, so
     * there is no need to free the arrays `row_ptr', `column_indices'
     * and `values' after calling `dCreate_CompRowLoc_Matrix_dist'.
     * The underlying storage is instead freed with
     * `Destroy_CompRowLoc_Matrix_dist'.
     */
    int32_t * row_ptr;
    int * column_indices;
    double * values;
    err = mtx_matrix_to_csr_coordinate_double(
        A, &row_ptr, &column_indices, &values);
    if (err)
        return err;

    /*
     * SuperLU_DIST uses a single array for the right-hand side and
     * solution vector, where the latter overwrites the former.
     * Therefore, we copy the values from `b' to `x'.
     */
    memmove(x->data, b->data, x->size * x->nonzero_size);

    if (verbose) {
        print_options_dist(options);
        print_sp_ienv_dist(options);
    }

    /* Find the first and last row */

    /* Set up the matrix. */
    SuperMatrix super_A;
    int32_t num_local_rows = A->num_rows;
    int32_t first_local_row = 0;
    Stype_t stype = SLU_NR_loc;
    Dtype_t dtype = SLU_D;  /* double precision floating point */
    Mtype_t mtype = SLU_GE; /* general, unsymmetric matrix. */
    dCreate_CompRowLoc_Matrix_dist(
        &super_A, A->num_rows, A->num_columns, A->size,
        num_local_rows, first_local_row,
        values, column_indices, row_ptr,
        stype, dtype, mtype);

    /* Initialize SuperLU process grid. */
    gridinfo_t grid;
    superlu_gridinit(
        comm, num_process_rows,
        num_process_columns, &grid);

    /* Set up data structures for LU factorisation and solver
     * statistics. */
    dScalePermstruct_t scale_perm;
    dLUstruct_t LU;
    SuperLUStat_t stat;
    dScalePermstructInit(
        A->num_rows, A->num_columns, &scale_perm);
    dLUstructInit(A->num_columns, &LU);
    PStatInit(&stat);

    /* Solve the linear system. */
    dSOLVEstruct_t solve;
    int num_right_hand_sides = 1;
    double berr;
    int status;
    pdgssvx(options, &super_A, &scale_perm,
            (double *) x->data, x->size, num_right_hand_sides,
            &grid, &LU, &solve, &berr, &stat, &err);

    if (verbose)
        PStatPrint(options, &stat, &grid);

    /* Clean up. */
    PStatFree(&stat);
    if (options->SolveInitialized)
        dSolveFinalize(options, &solve);
    dDestroy_LU(A->num_columns, &grid, &LU);
    dLUstructFree(&LU);
    dScalePermstructFree(&scale_perm);
    Destroy_CompRowLoc_Matrix_dist(&super_A);
    return MTX_SUCCESS;
#else
    return MTX_ERR_SUPERLU_DIST_NEEDED;
#endif
}

/**
 * `redirect_stdout()' redirects standard output to the given file
 * descriptor `fd'. This is done in a way such that standard output
 * can later be restored by calling `restore_stdout()'.
 */
static int redirect_stdout(
    int fd,
    int * stdout_copy)
{
    if (fflush(stdout) == EOF)
        return MTX_ERR_ERRNO;
    *stdout_copy = dup(STDOUT_FILENO);
    if (*stdout_copy == -1)
        return MTX_ERR_ERRNO;
    if (dup2(fd, STDOUT_FILENO) == -1) {
        int err = errno;
        close(*stdout_copy);
        errno = err;
        return MTX_ERR_ERRNO;
    }
    return MTX_SUCCESS;
}

/**
 * `restore_stdout()' restores standard output in cases where it was
 * previously redirected to a different file descriptor using
 * `redirect_stdout()'.
 */
static int restore_stdout(
    int fd, int stdout_copy)
{
    if (fflush(stdout) == EOF)
        return MTX_ERR_ERRNO;
    if (dup2(stdout_copy, STDOUT_FILENO) == -1)
        return MTX_ERR_ERRNO;
    if (close(stdout_copy) == -1)
        return MTX_ERR_ERRNO;
    return MTX_SUCCESS;
}

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
 *       num_process_rows, num_process_cols,
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
    bool SymPattern)
{
    int err;


    int mpi_initialized;
    MPI_Initialized(&mpi_initialized);
    if (!mpi_initialized)
        return MTX_ERR_MPI_NOT_INITIALIZED;

    int comm_size;
    *mpierr = MPI_Comm_size(comm, &comm_size);
    if (*mpierr)
        return MTX_ERR_MPI;
    if (comm_size != num_process_rows * num_process_columns)
        return MTX_ERR_SUPERLU_DIST_GRID_SIZE;

    /*
     * SuperLU_DIST will sometimes write to standard output, which
     * will pollute the output of our program. Since there is no easy
     * way to disable the output from SuperLU_DIST, we need to
     * temporarily redirect `stdout' to another file descriptor, and
     * then reinstate `stdout' when we are done.
     */
    int fd = verbose ? fileno(f) : open("/dev/null", O_WRONLY);
    if (fd == -1)
        return MTX_ERR_ERRNO;
    int stdout_copy;
    err = redirect_stdout(fd, &stdout_copy);
    if (err) {
        if (!verbose)
            close(fd);
        return err;
    }

    superlu_dist_options_t options;
    err = superlu_dist_options_set(
        &options, Fact, Equil, ParSymbFact,
        ColPerm, RowPerm, ReplaceTinyPivot, IterRefine,
        Trans, SolveInitialized, RefineInitialized,
        PrintStat, num_lookaheads, lookahead_etree,
        SymPattern);
    if (err) {
        int olderrno = errno;
        restore_stdout(fd, stdout_copy);
        if (!verbose)
            close(fd);
        errno = olderrno;
        return err;
    }

    if (A->field == mtx_double) {
        err = mtx_superlu_dist_solve_global_double(
            A, b, x, verbose, f, mpierr, comm, root,
            num_process_rows, num_process_columns,
            &options);
        if (err) {
            int olderrno = errno;
            restore_stdout(fd, stdout_copy);
            if (!verbose)
                close(fd);
            errno = olderrno;
            return err;
        }
    } else {
        return MTX_ERR_INVALID_MTX_FIELD;
    }

    /* Restore standard output. */
    err = restore_stdout(fd, stdout_copy);
    if (err)
        return err;
    return MTX_SUCCESS;
}

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
    bool SymPattern)
{
    int err;

    int mpi_initialized;
    MPI_Initialized(&mpi_initialized);
    if (!mpi_initialized)
        return MTX_ERR_MPI_NOT_INITIALIZED;

    int comm_size;
    *mpierr = MPI_Comm_size(comm, &comm_size);
    if (*mpierr)
        return MTX_ERR_MPI;
    if (comm_size < num_process_rows * num_process_columns) {
        *mpierr = MPI_ERR_COMM;
        return MTX_ERR_MPI;
    }

    /*
     * SuperLU_DIST will sometimes write to standard output, which
     * will pollute the output of our program. Since there is no easy
     * way to disable the output from SuperLU_DIST, we need to
     * temporarily redirect `stdout' to another file descriptor, and
     * then reinstate `stdout' when we are done.
     */
    int fd = verbose ? fileno(f) : open("/dev/null", O_WRONLY);
    if (fd == -1)
        return MTX_ERR_ERRNO;
    int stdout_copy;
    err = redirect_stdout(fd, &stdout_copy);
    if (err) {
        if (!verbose)
            close(fd);
        return err;
    }

    superlu_dist_options_t options;
    err = superlu_dist_options_set(
        &options, Fact, Equil, ParSymbFact,
        ColPerm, RowPerm, ReplaceTinyPivot, IterRefine,
        Trans, SolveInitialized, RefineInitialized,
        PrintStat, num_lookaheads, lookahead_etree,
        SymPattern);
    if (err) {
        int olderrno = errno;
        restore_stdout(fd, stdout_copy);
        if (!verbose)
            close(fd);
        errno = olderrno;
        return err;
    }

    if (A->field == mtx_double) {
        err = mtx_superlu_dist_solve_distributed_double(
            A, b, x, verbose, f, comm, root,
            num_process_rows, num_process_columns,
            &options);
        if (err) {
            int olderrno = errno;
            restore_stdout(fd, stdout_copy);
            if (!verbose)
                close(fd);
            errno = olderrno;
            return err;
        }
    } else {
        return MTX_ERR_INVALID_MTX_FIELD;
    }

    /* Restore standard output. */
    err = restore_stdout(fd, stdout_copy);
    if (err)
        return err;
    return MTX_SUCCESS;
}
#endif
