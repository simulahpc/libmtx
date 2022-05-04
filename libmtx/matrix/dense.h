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
 * Last modified: 2022-05-04
 *
 * Data structures for dense matrices.
 */

#ifndef LIBMTX_MATRIX_DENSE_H
#define LIBMTX_MATRIX_DENSE_H

#include <libmtx/libmtx-config.h>

#include <libmtx/precision.h>
#include <libmtx/field.h>
#include <libmtx/util/symmetry.h>
#include <libmtx/util/transpose.h>
#include <libmtx/vector/base.h>
#include <libmtx/vector/vector.h>

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

struct mtxfile;
struct mtxmatrix;
struct mtxpartition;

/**
 * ‘mtxmatrix_dense’ represents a dense matrix with entries stored in
 * row major order.
 */
struct mtxmatrix_dense
{
    /**
     * ‘symmetry’ is the matrix symmetry: ‘unsymmetric’, ‘symmetric’,
     * ‘skew-symmetric’ or ‘hermitian’.
     */
    enum mtxsymmetry symmetry;

    /**
     * ‘num_rows’ is the number of matrix rows.
     */
    int num_rows;

    /**
     * ‘num_columns’ is the number of matrix columns.
     */
    int num_columns;

    /**
     * ‘num_entries’ is the total number of (zero and nonzero) matrix
     * entries, which is therefore equal to ‘num_rows*num_columns’.
     */
    int64_t num_entries;

    /**
     * ‘num_nonzeros’ is the number of nonzero matrix entries,
     *  including those represented implicitly due to symmetry.
     */
    int64_t num_nonzeros;

    /**
     * ‘size’ is the number of explicitly stored matrix entries.
     */
    int64_t size;

    /**
     * ‘a’ is a vector storing the underlying nonzero matrix entries.
     */
    struct mtxvector_base a;
};

/*
 * memory management
 */

/**
 * ‘mtxmatrix_dense_free()’ frees storage allocated for a matrix.
 */
void mtxmatrix_dense_free(
    struct mtxmatrix_dense * A);

/**
 * ‘mtxmatrix_dense_alloc_copy()’ allocates a copy of a matrix without
 * initialising the values.
 */
int mtxmatrix_dense_alloc_copy(
    struct mtxmatrix_dense * dst,
    const struct mtxmatrix_dense * src);

/**
 * ‘mtxmatrix_dense_init_copy()’ allocates a copy of a matrix and also
 * copies the values.
 */
int mtxmatrix_dense_init_copy(
    struct mtxmatrix_dense * dst,
    const struct mtxmatrix_dense * src);

/*
 * initialise matrices from entrywise data in coordinate format
 */

/**
 * ‘mtxmatrix_dense_alloc_entries()’ allocates a matrix from entrywise
 * data in coordinate format.
 */
int mtxmatrix_dense_alloc_entries(
    struct mtxmatrix_dense * A,
    enum mtxfield field,
    enum mtxprecision precision,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    int idxstride,
    int idxbase,
    const int * rowidx,
    const int * colidx);

/**
 * ‘mtxmatrix_dense_init_entries_real_single()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * real, single precision coefficients.
 */
int mtxmatrix_dense_init_entries_real_single(
    struct mtxmatrix_dense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx,
    const float * data);

/**
 * ‘mtxmatrix_dense_init_entries_real_double()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * real, double precision coefficients.
 */
int mtxmatrix_dense_init_entries_real_double(
    struct mtxmatrix_dense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx,
    const double * data);

/**
 * ‘mtxmatrix_dense_init_entries_complex_single()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * complex, single precision coefficients.
 */
int mtxmatrix_dense_init_entries_complex_single(
    struct mtxmatrix_dense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx,
    const float (* data)[2]);

/**
 * ‘mtxmatrix_dense_init_entries_complex_double()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * complex, double precision coefficients.
 */
int mtxmatrix_dense_init_entries_complex_double(
    struct mtxmatrix_dense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx,
    const double (* data)[2]);

/**
 * ‘mtxmatrix_dense_init_entries_integer_single()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * integer, single precision coefficients.
 */
int mtxmatrix_dense_init_entries_integer_single(
    struct mtxmatrix_dense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx,
    const int32_t * data);

/**
 * ‘mtxmatrix_dense_init_entries_integer_double()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * integer, double precision coefficients.
 */
int mtxmatrix_dense_init_entries_integer_double(
    struct mtxmatrix_dense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx,
    const int64_t * data);

/**
 * ‘mtxmatrix_dense_init_entries_pattern()’ allocates and initialises
 * a matrix from entrywise data in coordinate format with boolean
 * coefficients.
 */
int mtxmatrix_dense_init_entries_pattern(
    struct mtxmatrix_dense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx);

/*
 * initialise matrices from entrywise data in coordinate format with
 * specified strides
 */

/**
 * ‘mtxmatrix_dense_init_entries_strided_real_single()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * real, single precision coefficients.
 */
int mtxmatrix_dense_init_entries_strided_real_single(
    struct mtxmatrix_dense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    int idxstride,
    int idxbase,
    const int * rowidx,
    const int * colidx,
    int datastride,
    const float * data);

/**
 * ‘mtxmatrix_dense_init_entries_strided_real_double()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * real, double precision coefficients.
 */
int mtxmatrix_dense_init_entries_strided_real_double(
    struct mtxmatrix_dense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    int idxstride,
    int idxbase,
    const int * rowidx,
    const int * colidx,
    int datastride,
    const double * data);

/**
 * ‘mtxmatrix_dense_init_entries_strided_complex_single()’ allocates
 * and initialises a matrix from entrywise data in coordinate format
 * with complex, single precision coefficients.
 */
int mtxmatrix_dense_init_entries_strided_complex_single(
    struct mtxmatrix_dense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    int idxstride,
    int idxbase,
    const int * rowidx,
    const int * colidx,
    int datastride,
    const float (* data)[2]);

/**
 * ‘mtxmatrix_dense_init_entries_strided_complex_double()’ allocates
 * and initialises a matrix from entrywise data in coordinate format
 * with complex, double precision coefficients.
 */
int mtxmatrix_dense_init_entries_strided_complex_double(
    struct mtxmatrix_dense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    int idxstride,
    int idxbase,
    const int * rowidx,
    const int * colidx,
    int datastride,
    const double (* data)[2]);

/**
 * ‘mtxmatrix_dense_init_entries_strided_integer_single()’ allocates
 * and initialises a matrix from entrywise data in coordinate format
 * with integer, single precision coefficients.
 */
int mtxmatrix_dense_init_entries_strided_integer_single(
    struct mtxmatrix_dense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    int idxstride,
    int idxbase,
    const int * rowidx,
    const int * colidx,
    int datastride,
    const int32_t * data);

/**
 * ‘mtxmatrix_dense_init_entries_strided_integer_double()’ allocates
 * and initialises a matrix from entrywise data in coordinate format
 * with integer, double precision coefficients.
 */
int mtxmatrix_dense_init_entries_strided_integer_double(
    struct mtxmatrix_dense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    int idxstride,
    int idxbase,
    const int * rowidx,
    const int * colidx,
    int datastride,
    const int64_t * data);

/**
 * ‘mtxmatrix_dense_init_entries_strided_pattern()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * boolean coefficients.
 */
int mtxmatrix_dense_init_entries_strided_pattern(
    struct mtxmatrix_dense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    int idxstride,
    int idxbase,
    const int * rowidx,
    const int * colidx);

/*
 * initialise matrices from row-wise data in compressed row format
 */

/**
 * ‘mtxmatrix_dense_alloc_rows()’ allocates a matrix from row-wise
 * data in compressed row format.
 */
int mtxmatrix_dense_alloc_rows(
    struct mtxmatrix_dense * A,
    enum mtxfield field,
    enum mtxprecision precision,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx);

/**
 * ‘mtxmatrix_dense_init_rows_real_single()’ allocates and initialises
 * a matrix from row-wise data in compressed row format with real,
 * single precision coefficients.
 */
int mtxmatrix_dense_init_rows_real_single(
    struct mtxmatrix_dense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const float * data);

/**
 * ‘mtxmatrix_dense_init_rows_real_double()’ allocates and initialises
 * a matrix from row-wise data in compressed row format with real,
 * double precision coefficients.
 */
int mtxmatrix_dense_init_rows_real_double(
    struct mtxmatrix_dense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const double * data);

/**
 * ‘mtxmatrix_dense_init_rows_complex_single()’ allocates and
 * initialises a matrix from row-wise data in compressed row format
 * with complex, single precision coefficients.
 */
int mtxmatrix_dense_init_rows_complex_single(
    struct mtxmatrix_dense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const float (* data)[2]);

/**
 * ‘mtxmatrix_dense_init_rows_complex_double()’ allocates and
 * initialises a matrix from row-wise data in compressed row format
 * with complex, double precision coefficients.
 */
int mtxmatrix_dense_init_rows_complex_double(
    struct mtxmatrix_dense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const double (* data)[2]);

/**
 * ‘mtxmatrix_dense_init_rows_integer_single()’ allocates and
 * initialises a matrix from row-wise data in compressed row format
 * with integer, single precision coefficients.
 */
int mtxmatrix_dense_init_rows_integer_single(
    struct mtxmatrix_dense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const int32_t * data);

/**
 * ‘mtxmatrix_dense_init_rows_integer_double()’ allocates and
 * initialises a matrix from row-wise data in compressed row format
 * with integer, double precision coefficients.
 */
int mtxmatrix_dense_init_rows_integer_double(
    struct mtxmatrix_dense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const int64_t * data);

/**
 * ‘mtxmatrix_dense_init_rows_pattern()’ allocates and initialises a
 * matrix from row-wise data in compressed row format with boolean
 * coefficients.
 */
int mtxmatrix_dense_init_rows_pattern(
    struct mtxmatrix_dense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx);

/*
 * initialise matrices from column-wise data in compressed column
 * format
 */

/**
 * ‘mtxmatrix_dense_alloc_columns()’ allocates a matrix from
 * column-wise data in compressed column format.
 */
int mtxmatrix_dense_alloc_columns(
    struct mtxmatrix_dense * A,
    enum mtxfield field,
    enum mtxprecision precision,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx);

/**
 * ‘mtxmatrix_dense_init_columns_real_single()’ allocates and
 * initialises a matrix from column-wise data in compressed column
 * format with real, single precision coefficients.
 */
int mtxmatrix_dense_init_columns_real_single(
    struct mtxmatrix_dense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx,
    const float * data);

/**
 * ‘mtxmatrix_dense_init_columns_real_double()’ allocates and
 * initialises a matrix from column-wise data in compressed column
 * format with real, double precision coefficients.
 */
int mtxmatrix_dense_init_columns_real_double(
    struct mtxmatrix_dense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx,
    const double * data);

/**
 * ‘mtxmatrix_dense_init_columns_complex_single()’ allocates and
 * initialises a matrix from column-wise data in compressed column
 * format with complex, single precision coefficients.
 */
int mtxmatrix_dense_init_columns_complex_single(
    struct mtxmatrix_dense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx,
    const float (* data)[2]);

/**
 * ‘mtxmatrix_dense_init_columns_complex_double()’ allocates and
 * initialises a matrix from column-wise data in compressed column
 * format with complex, double precision coefficients.
 */
int mtxmatrix_dense_init_columns_complex_double(
    struct mtxmatrix_dense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx,
    const double (* data)[2]);

/**
 * ‘mtxmatrix_dense_init_columns_integer_single()’ allocates and
 * initialises a matrix from column-wise data in compressed column
 * format with integer, single precision coefficients.
 */
int mtxmatrix_dense_init_columns_integer_single(
    struct mtxmatrix_dense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx,
    const int32_t * data);

/**
 * ‘mtxmatrix_dense_init_columns_integer_double()’ allocates and
 * initialises a matrix from column-wise data in compressed column
 * format with integer, double precision coefficients.
 */
int mtxmatrix_dense_init_columns_integer_double(
    struct mtxmatrix_dense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx,
    const int64_t * data);

/**
 * ‘mtxmatrix_dense_init_columns_pattern()’ allocates and initialises
 * a matrix from column-wise data in compressed column format with
 * boolean coefficients.
 */
int mtxmatrix_dense_init_columns_pattern(
    struct mtxmatrix_dense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx);

/*
 * initialise matrices from a list of dense cliques
 */

/**
 * ‘mtxmatrix_dense_alloc_cliques()’ allocates a matrix from a list of
 * dense cliques.
 */
int mtxmatrix_dense_alloc_cliques(
    struct mtxmatrix_dense * A,
    enum mtxfield field,
    enum mtxprecision precision,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_cliques,
    const int64_t * cliqueptr,
    const int * rowidx,
    const int * colidx);

/**
 * ‘mtxmatrix_dense_init_cliques_real_single()’ allocates and
 * initialises a matrix from a list of dense cliques with real, single
 * precision coefficients.
 */
int mtxmatrix_dense_init_cliques_real_single(
    struct mtxmatrix_dense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_cliques,
    const int64_t * cliqueptr,
    const int * rowidx,
    const int * colidx,
    const float * data);

/**
 * ‘mtxmatrix_dense_init_cliques_real_double()’ allocates and
 * initialises a matrix from a list of dense cliques with real, double
 * precision coefficients.
 */
int mtxmatrix_dense_init_cliques_real_double(
    struct mtxmatrix_dense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_cliques,
    const int64_t * cliqueptr,
    const int * rowidx,
    const int * colidx,
    const double * data);

/**
 * ‘mtxmatrix_dense_init_cliques_complex_single()’ allocates and
 * initialises a matrix from a list of dense cliques with complex,
 * single precision coefficients.
 */
int mtxmatrix_dense_init_cliques_complex_single(
    struct mtxmatrix_dense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_cliques,
    const int64_t * cliqueptr,
    const int * rowidx,
    const int * colidx,
    const float (* data)[2]);

/**
 * ‘mtxmatrix_dense_init_cliques_complex_double()’ allocates and
 * initialises a matrix from a list of dense cliques with complex,
 * double precision coefficients.
 */
int mtxmatrix_dense_init_cliques_complex_double(
    struct mtxmatrix_dense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_cliques,
    const int64_t * cliqueptr,
    const int * rowidx,
    const int * colidx,
    const double (* data)[2]);

/**
 * ‘mtxmatrix_dense_init_cliques_integer_single()’ allocates and
 * initialises a matrix from a list of dense cliques with integer,
 * single precision coefficients.
 */
int mtxmatrix_dense_init_cliques_integer_single(
    struct mtxmatrix_dense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_cliques,
    const int64_t * cliqueptr,
    const int * rowidx,
    const int * colidx,
    const int32_t * data);

/**
 * ‘mtxmatrix_dense_init_cliques_integer_double()’ allocates and
 * initialises a matrix from a list of dense cliques with integer,
 * double precision coefficients.
 */
int mtxmatrix_dense_init_cliques_integer_double(
    struct mtxmatrix_dense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_cliques,
    const int64_t * cliqueptr,
    const int * rowidx,
    const int * colidx,
    const int64_t * data);

/**
 * ‘mtxmatrix_dense_init_cliques_pattern()’ allocates and initialises
 * a matrix from a list of dense cliques with boolean coefficients.
 */
int mtxmatrix_dense_init_cliques_pattern(
    struct mtxmatrix_dense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_cliques,
    const int64_t * cliqueptr,
    const int * rowidx,
    const int * colidx);

/*
 * modifying values
 */

/**
 * ‘mtxmatrix_dense_setzero()’ sets every value of a matrix to zero.
 */
int mtxmatrix_dense_setzero(
    struct mtxmatrix_dense * A);

/**
 * ‘mtxmatrix_dense_set_real_single()’ sets values of a matrix based
 * on an array of single precision floating point numbers.
 */
int mtxmatrix_dense_set_real_single(
    struct mtxmatrix_dense * A,
    int64_t size,
    int stride,
    const float * a);

/**
 * ‘mtxmatrix_dense_set_real_double()’ sets values of a matrix based
 * on an array of double precision floating point numbers.
 */
int mtxmatrix_dense_set_real_double(
    struct mtxmatrix_dense * A,
    int64_t size,
    int stride,
    const double * a);

/**
 * ‘mtxmatrix_dense_set_complex_single()’ sets values of a matrix
 * based on an array of single precision floating point complex
 * numbers.
 */
int mtxmatrix_dense_set_complex_single(
    struct mtxmatrix_dense * A,
    int64_t size,
    int stride,
    const float (*a)[2]);

/**
 * ‘mtxmatrix_dense_set_complex_double()’ sets values of a matrix
 * based on an array of double precision floating point complex
 * numbers.
 */
int mtxmatrix_dense_set_complex_double(
    struct mtxmatrix_dense * A,
    int64_t size,
    int stride,
    const double (*a)[2]);

/**
 * ‘mtxmatrix_dense_set_integer_single()’ sets values of a matrix
 * based on an array of integers.
 */
int mtxmatrix_dense_set_integer_single(
    struct mtxmatrix_dense * A,
    int64_t size,
    int stride,
    const int32_t * a);

/**
 * ‘mtxmatrix_dense_set_integer_double()’ sets values of a matrix
 * based on an array of integers.
 */
int mtxmatrix_dense_set_integer_double(
    struct mtxmatrix_dense * A,
    int64_t size,
    int stride,
    const int64_t * a);

/*
 * row and column vectors
 */

/**
 * ‘mtxmatrix_dense_alloc_row_vector()’ allocates a row vector for a
 * given matrix, where a row vector is a vector whose length equal to
 * a single row of the matrix.
 */
int mtxmatrix_dense_alloc_row_vector(
    const struct mtxmatrix_dense * A,
    struct mtxvector * x,
    enum mtxvectortype vectortype);

/**
 * ‘mtxmatrix_dense_alloc_column_vector()’ allocates a column vector
 * for a given matrix, where a column vector is a vector whose length
 * equal to a single column of the matrix.
 */
int mtxmatrix_dense_alloc_column_vector(
    const struct mtxmatrix_dense * A,
    struct mtxvector * y,
    enum mtxvectortype vectortype);

/*
 * convert to and from Matrix Market format
 */

/**
 * ‘mtxmatrix_dense_from_mtxfile()’ converts a matrix from Matrix
 * Market format.
 */
int mtxmatrix_dense_from_mtxfile(
    struct mtxmatrix_dense * A,
    const struct mtxfile * mtxfile);

/**
 * ‘mtxmatrix_dense_to_mtxfile()’ converts a matrix to Matrix Market
 * format.
 */
int mtxmatrix_dense_to_mtxfile(
    struct mtxfile * mtxfile,
    const struct mtxmatrix_dense * A,
    int64_t num_rows,
    const int64_t * rowidx,
    int64_t num_columns,
    const int64_t * colidx,
    enum mtxfileformat mtxfmt);

/*
 * Level 1 BLAS operations
 */

/**
 * ‘mtxmatrix_dense_swap()’ swaps values of two matrices,
 * simultaneously performing ‘y <- x’ and ‘x <- y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_dense_swap(
    struct mtxmatrix_dense * x,
    struct mtxmatrix_dense * y);

/**
 * ‘mtxmatrix_dense_copy()’ copies values of a matrix, ‘y = x’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_dense_copy(
    struct mtxmatrix_dense * y,
    const struct mtxmatrix_dense * x);

/**
 * ‘mtxmatrix_dense_sscal()’ scales a matrix by a single precision
 * floating point scalar, ‘x = a*x’.
 */
int mtxmatrix_dense_sscal(
    float a,
    struct mtxmatrix_dense * x,
    int64_t * num_flops);

/**
 * ‘mtxmatrix_dense_dscal()’ scales a matrix by a double precision
 * floating point scalar, ‘x = a*x’.
 */
int mtxmatrix_dense_dscal(
    double a,
    struct mtxmatrix_dense * x,
    int64_t * num_flops);

/**
 * ‘mtxmatrix_dense_cscal()’ scales a matrix by a complex, single
 * precision floating point scalar, ‘x = (a+b*i)*x’.
 */
int mtxmatrix_dense_cscal(
    float a[2],
    struct mtxmatrix_dense * x,
    int64_t * num_flops);

/**
 * ‘mtxmatrix_dense_zscal()’ scales a matrix by a complex, double
 * precision floating point scalar, ‘x = (a+b*i)*x’.
 */
int mtxmatrix_dense_zscal(
    double a[2],
    struct mtxmatrix_dense * x,
    int64_t * num_flops);

/**
 * ‘mtxmatrix_dense_saxpy()’ adds a matrix to another one multiplied
 * by a single precision floating point value, ‘y = a*x + y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_dense_saxpy(
    float a,
    const struct mtxmatrix_dense * x,
    struct mtxmatrix_dense * y,
    int64_t * num_flops);

/**
 * ‘mtxmatrix_dense_daxpy()’ adds a matrix to another one multiplied
 * by a double precision floating point value, ‘y = a*x + y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_dense_daxpy(
    double a,
    const struct mtxmatrix_dense * x,
    struct mtxmatrix_dense * y,
    int64_t * num_flops);

/**
 * ‘mtxmatrix_dense_saypx()’ multiplies a matrix by a single precision
 * floating point scalar and adds another matrix, ‘y = a*y + x’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_dense_saypx(
    float a,
    struct mtxmatrix_dense * y,
    const struct mtxmatrix_dense * x,
    int64_t * num_flops);

/**
 * ‘mtxmatrix_dense_daypx()’ multiplies a matrix by a double precision
 * floating point scalar and adds another matrix, ‘y = a*y + x’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_dense_daypx(
    double a,
    struct mtxmatrix_dense * y,
    const struct mtxmatrix_dense * x,
    int64_t * num_flops);

/**
 * ‘mtxmatrix_dense_sdot()’ computes the Frobenius inner product of
 * two matrices in single precision floating point.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_dense_sdot(
    const struct mtxmatrix_dense * x,
    const struct mtxmatrix_dense * y,
    float * dot,
    int64_t * num_flops);

/**
 * ‘mtxmatrix_dense_ddot()’ computes the Frobenius inner product of
 * two matrices in double precision floating point.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_dense_ddot(
    const struct mtxmatrix_dense * x,
    const struct mtxmatrix_dense * y,
    double * dot,
    int64_t * num_flops);

/**
 * ‘mtxmatrix_dense_cdotu()’ computes the product of the transpose of
 * a complex row matrix with another complex row matrix in single
 * precision floating point, ‘dot := x^T*y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_dense_cdotu(
    const struct mtxmatrix_dense * x,
    const struct mtxmatrix_dense * y,
    float (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxmatrix_dense_zdotu()’ computes the product of the transpose of
 * a complex row matrix with another complex row matrix in double
 * precision floating point, ‘dot := x^T*y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_dense_zdotu(
    const struct mtxmatrix_dense * x,
    const struct mtxmatrix_dense * y,
    double (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxmatrix_dense_cdotc()’ computes the Frobenius inner product of
 * two complex matrices in single precision floating point, ‘dot :=
 * x^H*y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_dense_cdotc(
    const struct mtxmatrix_dense * x,
    const struct mtxmatrix_dense * y,
    float (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxmatrix_dense_zdotc()’ computes the Frobenius inner product of
 * two complex matrices in double precision floating point, ‘dot :=
 * x^H*y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_dense_zdotc(
    const struct mtxmatrix_dense * x,
    const struct mtxmatrix_dense * y,
    double (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxmatrix_dense_snrm2()’ computes the Frobenius norm of a matrix
 * in single precision floating point.
 */
int mtxmatrix_dense_snrm2(
    const struct mtxmatrix_dense * x,
    float * nrm2,
    int64_t * num_flops);

/**
 * ‘mtxmatrix_dense_dnrm2()’ computes the Frobenius norm of a matrix
 * in double precision floating point.
 */
int mtxmatrix_dense_dnrm2(
    const struct mtxmatrix_dense * x,
    double * nrm2,
    int64_t * num_flops);

/**
 * ‘mtxmatrix_dense_sasum()’ computes the sum of absolute values
 * (1-norm) of a matrix in single precision floating point.  If the
 * matrix is complex-valued, then the sum of the absolute values of
 * the real and imaginary parts is computed.
 */
int mtxmatrix_dense_sasum(
    const struct mtxmatrix_dense * x,
    float * asum,
    int64_t * num_flops);

/**
 * ‘mtxmatrix_dense_dasum()’ computes the sum of absolute values
 * (1-norm) of a matrix in double precision floating point.  If the
 * matrix is complex-valued, then the sum of the absolute values of
 * the real and imaginary parts is computed.
 */
int mtxmatrix_dense_dasum(
    const struct mtxmatrix_dense * x,
    double * asum,
    int64_t * num_flops);

/**
 * ‘mtxmatrix_dense_iamax()’ finds the index of the first element
 * having the maximum absolute value.  If the matrix is
 * complex-valued, then the index points to the first element having
 * the maximum sum of the absolute values of the real and imaginary
 * parts.
 */
int mtxmatrix_dense_iamax(
    const struct mtxmatrix_dense * x,
    int * iamax);

/*
 * Level 2 BLAS operations (matrix-vector)
 */

/**
 * ‘mtxmatrix_dense_sgemv()’ multiplies a matrix ‘A’ or its transpose
 * ‘A'’ by a real scalar ‘alpha’ (‘α’) and a vector ‘x’, before adding
 * the result to another vector ‘y’ multiplied by another real scalar
 * ‘beta’ (‘β’). That is, ‘y = α*A*x + β*y’ or ‘y = α*A'*x + β*y’.
 *
 * The scalars ‘alpha’ and ‘beta’ are given as single precision
 * floating point numbers.
 *
 * The vectors ‘x’ and ‘y’ must be of ‘array’ type, and they must both
 * have the same field and precision.  If the field of the matrix ‘A’
 * is ‘real’, ‘integer’ or ‘complex’, then the vectors must have the
 * same field.  Otherwise, if the matrix field is ‘pattern’, then ‘x’
 * and ‘y’ are allowed to be ‘real’, ‘integer’ or ‘complex’, but they
 * must both have the same field.
 *
 * Moreover, if ‘trans’ is ‘mtx_notrans’, then the size of ‘x’ must
 * equal the number of columns of ‘A’ and the size of ‘y’ must equal
 * the number of rows of ‘A’. if ‘trans’ is ‘mtx_trans’ or
 * ‘mtx_conjtrans’, then the size of ‘x’ must equal the number of rows
 * of ‘A’ and the size of ‘y’ must equal the number of columns of ‘A’.
 */
int mtxmatrix_dense_sgemv(
    enum mtxtransposition trans,
    float alpha,
    const struct mtxmatrix_dense * A,
    const struct mtxvector * x,
    float beta,
    struct mtxvector * y,
    int64_t * num_flops);

/**
 * ‘mtxmatrix_dense_dgemv()’ multiplies a matrix ‘A’ or its transpose
 * ‘A'’ by a real scalar ‘alpha’ (‘α’) and a vector ‘x’, before adding
 * the result to another vector ‘y’ multiplied by another scalar real
 * ‘beta’ (‘β’).  That is, ‘y = α*A*x + β*y’ or ‘y = α*A'*x + β*y’.
 *
 * The scalars ‘alpha’ and ‘beta’ are given as double precision
 * floating point numbers.
 *
 * The vectors ‘x’ and ‘y’ must be of ‘array’ type, and they must both
 * have the same field and precision.  If the field of the matrix ‘A’
 * is ‘real’, ‘integer’ or ‘complex’, then the vectors must have the
 * same field.  Otherwise, if the matrix field is ‘pattern’, then ‘x’
 * and ‘y’ are allowed to be ‘real’, ‘integer’ or ‘complex’, but they
 * must both have the same field.
 *
 * Moreover, if ‘trans’ is ‘mtx_notrans’, then the size of ‘x’ must
 * equal the number of columns of ‘A’ and the size of ‘y’ must equal
 * the number of rows of ‘A’. if ‘trans’ is ‘mtx_trans’ or
 * ‘mtx_conjtrans’, then the size of ‘x’ must equal the number of rows
 * of ‘A’ and the size of ‘y’ must equal the number of columns of ‘A’.
 */
int mtxmatrix_dense_dgemv(
    enum mtxtransposition trans,
    double alpha,
    const struct mtxmatrix_dense * A,
    const struct mtxvector * x,
    double beta,
    struct mtxvector * y,
    int64_t * num_flops);

/**
 * ‘mtxmatrix_dense_cgemv()’ multiplies a complex-valued matrix ‘A’,
 * its transpose ‘A'’ or its conjugate transpose ‘Aᴴ’ by a complex
 * scalar ‘alpha’ (‘α’) and a vector ‘x’, before adding the result to
 * another vector ‘y’ multiplied by another complex scalar ‘beta’
 * (‘β’).  That is, ‘y = α*A*x + β*y’, ‘y = α*A'*x + β*y’ or ‘y =
 * α*Aᴴ*x + β*y’.
 *
 * The scalars ‘alpha’ and ‘beta’ are given as single precision
 * floating point numbers.
 *
 * The vectors ‘x’ and ‘y’ must be of ‘array’ type, and they must both
 * have the same field and precision as the matrix ‘A’.
 *
 * Moreover, if ‘trans’ is ‘mtx_notrans’, then the size of ‘x’ must
 * equal the number of columns of ‘A’ and the size of ‘y’ must equal
 * the number of rows of ‘A’. if ‘trans’ is ‘mtx_trans’ or
 * ‘mtx_conjtrans’, then the size of ‘x’ must equal the number of rows
 * of ‘A’ and the size of ‘y’ must equal the number of columns of ‘A’.
 */
int mtxmatrix_dense_cgemv(
    enum mtxtransposition trans,
    float alpha[2],
    const struct mtxmatrix_dense * A,
    const struct mtxvector * x,
    float beta[2],
    struct mtxvector * y,
    int64_t * num_flops);

/**
 * ‘mtxmatrix_dense_zgemv()’ multiplies a complex-valued matrix ‘A’,
 * its transpose ‘A'’ or its conjugate transpose ‘Aᴴ’ by a complex
 * scalar ‘alpha’ (‘α’) and a vector ‘x’, before adding the result to
 * another vector ‘y’ multiplied by another complex scalar ‘beta’
 * (‘β’).  That is, ‘y = α*A*x + β*y’, ‘y = α*A'*x + β*y’ or ‘y =
 * α*Aᴴ*x + β*y’.
 *
 * The scalars ‘alpha’ and ‘beta’ are given as double precision
 * floating point numbers.
 *
 * The vectors ‘x’ and ‘y’ must be of ‘array’ type, and they must both
 * have the same field and precision as the matrix ‘A’.
 *
 * Moreover, if ‘trans’ is ‘mtx_notrans’, then the size of ‘x’ must
 * equal the number of columns of ‘A’ and the size of ‘y’ must equal
 * the number of rows of ‘A’. if ‘trans’ is ‘mtx_trans’ or
 * ‘mtx_conjtrans’, then the size of ‘x’ must equal the number of rows
 * of ‘A’ and the size of ‘y’ must equal the number of columns of ‘A’.
 */
int mtxmatrix_dense_zgemv(
    enum mtxtransposition trans,
    double alpha[2],
    const struct mtxmatrix_dense * A,
    const struct mtxvector * x,
    double beta[2],
    struct mtxvector * y,
    int64_t * num_flops);

#endif
