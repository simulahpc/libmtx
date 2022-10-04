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
 * Last modified: 2022-10-03
 *
 * Dense matrices with BLAS-accelerated operations.
 */

#ifndef LIBMTX_LINALG_BLAS_DENSE_H
#define LIBMTX_LINALG_BLAS_DENSE_H

#include <libmtx/libmtx-config.h>

#include <libmtx/linalg/symmetry.h>
#include <libmtx/linalg/transpose.h>
#include <libmtx/linalg/base/vector.h>
#include <libmtx/linalg/field.h>
#include <libmtx/linalg/precision.h>
#include <libmtx/linalg/local/vector.h>

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>

struct mtxfile;
struct mtxmatrix;

/**
 * ‘mtxblasdense’ represents a dense matrix with entries stored in
 * row major order.
 */
struct mtxblasdense
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
    struct mtxblasvector a;
};

/*
 * matrix properties
 */

/**
 * ‘mtxblasdense_field()’ gets the field of a matrix.
 */
enum mtxfield mtxblasdense_field(const struct mtxblasdense * A);

/**
 * ‘mtxblasdense_precision()’ gets the precision of a matrix.
 */
enum mtxprecision mtxblasdense_precision(const struct mtxblasdense * A);

/**
 * ‘mtxblasdense_symmetry()’ gets the symmetry of a matrix.
 */
enum mtxsymmetry mtxblasdense_symmetry(const struct mtxblasdense * A);

/**
 * ‘mtxblasdense_num_rows()’ gets the number of matrix rows.
 */
int mtxblasdense_num_rows(const struct mtxblasdense * A);

/**
 * ‘mtxblasdense_num_columns()’ gets the number of matrix columns.
 */
int mtxblasdense_num_columns(const struct mtxblasdense * A);

/**
 * ‘mtxblasdense_num_nonzeros()’ gets the number of the number of
 *  nonzero matrix entries, including those represented implicitly due
 *  to symmetry.
 */
int64_t mtxblasdense_num_nonzeros(const struct mtxblasdense * A);

/**
 * ‘mtxblasdense_size()’ gets the number of explicitly stored
 * nonzeros of a matrix.
 */
int64_t mtxblasdense_size(const struct mtxblasdense * A);

/**
 * ‘mtxblasdense_rowcolidx()’ gets the row and column indices of the
 * explicitly stored matrix nonzeros.
 *
 * The arguments ‘rowidx’ and ‘colidx’ may be ‘NULL’ or must point to
 * an arrays of length ‘size’.
 */
int mtxblasdense_rowcolidx(
    const struct mtxblasdense * A,
    int64_t size,
    int * rowidx,
    int * colidx);

/*
 * memory management
 */

/**
 * ‘mtxblasdense_free()’ frees storage allocated for a matrix.
 */
void mtxblasdense_free(
    struct mtxblasdense * A);

/**
 * ‘mtxblasdense_alloc_copy()’ allocates a copy of a matrix without
 * initialising the values.
 */
int mtxblasdense_alloc_copy(
    struct mtxblasdense * dst,
    const struct mtxblasdense * src);

/**
 * ‘mtxblasdense_init_copy()’ allocates a copy of a matrix and also
 * copies the values.
 */
int mtxblasdense_init_copy(
    struct mtxblasdense * dst,
    const struct mtxblasdense * src);

/*
 * initialise matrices from entrywise data in coordinate format
 */

/**
 * ‘mtxblasdense_alloc_entries()’ allocates a matrix from entrywise
 * data in coordinate format.
 */
int mtxblasdense_alloc_entries(
    struct mtxblasdense * A,
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
 * ‘mtxblasdense_init_entries_real_single()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * real, single precision coefficients.
 */
int mtxblasdense_init_entries_real_single(
    struct mtxblasdense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx,
    const float * data);

/**
 * ‘mtxblasdense_init_entries_real_double()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * real, double precision coefficients.
 */
int mtxblasdense_init_entries_real_double(
    struct mtxblasdense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx,
    const double * data);

/**
 * ‘mtxblasdense_init_entries_complex_single()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * complex, single precision coefficients.
 */
int mtxblasdense_init_entries_complex_single(
    struct mtxblasdense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx,
    const float (* data)[2]);

/**
 * ‘mtxblasdense_init_entries_complex_double()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * complex, double precision coefficients.
 */
int mtxblasdense_init_entries_complex_double(
    struct mtxblasdense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx,
    const double (* data)[2]);

/**
 * ‘mtxblasdense_init_entries_integer_single()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * integer, single precision coefficients.
 */
int mtxblasdense_init_entries_integer_single(
    struct mtxblasdense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx,
    const int32_t * data);

/**
 * ‘mtxblasdense_init_entries_integer_double()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * integer, double precision coefficients.
 */
int mtxblasdense_init_entries_integer_double(
    struct mtxblasdense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx,
    const int64_t * data);

/**
 * ‘mtxblasdense_init_entries_pattern()’ allocates and initialises
 * a matrix from entrywise data in coordinate format with boolean
 * coefficients.
 */
int mtxblasdense_init_entries_pattern(
    struct mtxblasdense * A,
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
 * ‘mtxblasdense_init_entries_strided_real_single()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * real, single precision coefficients.
 */
int mtxblasdense_init_entries_strided_real_single(
    struct mtxblasdense * A,
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
 * ‘mtxblasdense_init_entries_strided_real_double()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * real, double precision coefficients.
 */
int mtxblasdense_init_entries_strided_real_double(
    struct mtxblasdense * A,
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
 * ‘mtxblasdense_init_entries_strided_complex_single()’ allocates
 * and initialises a matrix from entrywise data in coordinate format
 * with complex, single precision coefficients.
 */
int mtxblasdense_init_entries_strided_complex_single(
    struct mtxblasdense * A,
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
 * ‘mtxblasdense_init_entries_strided_complex_double()’ allocates
 * and initialises a matrix from entrywise data in coordinate format
 * with complex, double precision coefficients.
 */
int mtxblasdense_init_entries_strided_complex_double(
    struct mtxblasdense * A,
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
 * ‘mtxblasdense_init_entries_strided_integer_single()’ allocates
 * and initialises a matrix from entrywise data in coordinate format
 * with integer, single precision coefficients.
 */
int mtxblasdense_init_entries_strided_integer_single(
    struct mtxblasdense * A,
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
 * ‘mtxblasdense_init_entries_strided_integer_double()’ allocates
 * and initialises a matrix from entrywise data in coordinate format
 * with integer, double precision coefficients.
 */
int mtxblasdense_init_entries_strided_integer_double(
    struct mtxblasdense * A,
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
 * ‘mtxblasdense_init_entries_strided_pattern()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * boolean coefficients.
 */
int mtxblasdense_init_entries_strided_pattern(
    struct mtxblasdense * A,
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
 * ‘mtxblasdense_alloc_rows()’ allocates a matrix from row-wise
 * data in compressed row format.
 */
int mtxblasdense_alloc_rows(
    struct mtxblasdense * A,
    enum mtxfield field,
    enum mtxprecision precision,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx);

/**
 * ‘mtxblasdense_init_rows_real_single()’ allocates and initialises
 * a matrix from row-wise data in compressed row format with real,
 * single precision coefficients.
 */
int mtxblasdense_init_rows_real_single(
    struct mtxblasdense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const float * data);

/**
 * ‘mtxblasdense_init_rows_real_double()’ allocates and initialises
 * a matrix from row-wise data in compressed row format with real,
 * double precision coefficients.
 */
int mtxblasdense_init_rows_real_double(
    struct mtxblasdense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const double * data);

/**
 * ‘mtxblasdense_init_rows_complex_single()’ allocates and
 * initialises a matrix from row-wise data in compressed row format
 * with complex, single precision coefficients.
 */
int mtxblasdense_init_rows_complex_single(
    struct mtxblasdense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const float (* data)[2]);

/**
 * ‘mtxblasdense_init_rows_complex_double()’ allocates and
 * initialises a matrix from row-wise data in compressed row format
 * with complex, double precision coefficients.
 */
int mtxblasdense_init_rows_complex_double(
    struct mtxblasdense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const double (* data)[2]);

/**
 * ‘mtxblasdense_init_rows_integer_single()’ allocates and
 * initialises a matrix from row-wise data in compressed row format
 * with integer, single precision coefficients.
 */
int mtxblasdense_init_rows_integer_single(
    struct mtxblasdense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const int32_t * data);

/**
 * ‘mtxblasdense_init_rows_integer_double()’ allocates and
 * initialises a matrix from row-wise data in compressed row format
 * with integer, double precision coefficients.
 */
int mtxblasdense_init_rows_integer_double(
    struct mtxblasdense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const int64_t * data);

/**
 * ‘mtxblasdense_init_rows_pattern()’ allocates and initialises a
 * matrix from row-wise data in compressed row format with boolean
 * coefficients.
 */
int mtxblasdense_init_rows_pattern(
    struct mtxblasdense * A,
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
 * ‘mtxblasdense_alloc_columns()’ allocates a matrix from
 * column-wise data in compressed column format.
 */
int mtxblasdense_alloc_columns(
    struct mtxblasdense * A,
    enum mtxfield field,
    enum mtxprecision precision,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx);

/**
 * ‘mtxblasdense_init_columns_real_single()’ allocates and
 * initialises a matrix from column-wise data in compressed column
 * format with real, single precision coefficients.
 */
int mtxblasdense_init_columns_real_single(
    struct mtxblasdense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx,
    const float * data);

/**
 * ‘mtxblasdense_init_columns_real_double()’ allocates and
 * initialises a matrix from column-wise data in compressed column
 * format with real, double precision coefficients.
 */
int mtxblasdense_init_columns_real_double(
    struct mtxblasdense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx,
    const double * data);

/**
 * ‘mtxblasdense_init_columns_complex_single()’ allocates and
 * initialises a matrix from column-wise data in compressed column
 * format with complex, single precision coefficients.
 */
int mtxblasdense_init_columns_complex_single(
    struct mtxblasdense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx,
    const float (* data)[2]);

/**
 * ‘mtxblasdense_init_columns_complex_double()’ allocates and
 * initialises a matrix from column-wise data in compressed column
 * format with complex, double precision coefficients.
 */
int mtxblasdense_init_columns_complex_double(
    struct mtxblasdense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx,
    const double (* data)[2]);

/**
 * ‘mtxblasdense_init_columns_integer_single()’ allocates and
 * initialises a matrix from column-wise data in compressed column
 * format with integer, single precision coefficients.
 */
int mtxblasdense_init_columns_integer_single(
    struct mtxblasdense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx,
    const int32_t * data);

/**
 * ‘mtxblasdense_init_columns_integer_double()’ allocates and
 * initialises a matrix from column-wise data in compressed column
 * format with integer, double precision coefficients.
 */
int mtxblasdense_init_columns_integer_double(
    struct mtxblasdense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx,
    const int64_t * data);

/**
 * ‘mtxblasdense_init_columns_pattern()’ allocates and initialises
 * a matrix from column-wise data in compressed column format with
 * boolean coefficients.
 */
int mtxblasdense_init_columns_pattern(
    struct mtxblasdense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx);

/*
 * initialise matrices from a list of dense cliques
 */

/**
 * ‘mtxblasdense_alloc_cliques()’ allocates a matrix from a list of
 * dense cliques.
 */
int mtxblasdense_alloc_cliques(
    struct mtxblasdense * A,
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
 * ‘mtxblasdense_init_cliques_real_single()’ allocates and
 * initialises a matrix from a list of dense cliques with real, single
 * precision coefficients.
 */
int mtxblasdense_init_cliques_real_single(
    struct mtxblasdense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_cliques,
    const int64_t * cliqueptr,
    const int * rowidx,
    const int * colidx,
    const float * data);

/**
 * ‘mtxblasdense_init_cliques_real_double()’ allocates and
 * initialises a matrix from a list of dense cliques with real, double
 * precision coefficients.
 */
int mtxblasdense_init_cliques_real_double(
    struct mtxblasdense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_cliques,
    const int64_t * cliqueptr,
    const int * rowidx,
    const int * colidx,
    const double * data);

/**
 * ‘mtxblasdense_init_cliques_complex_single()’ allocates and
 * initialises a matrix from a list of dense cliques with complex,
 * single precision coefficients.
 */
int mtxblasdense_init_cliques_complex_single(
    struct mtxblasdense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_cliques,
    const int64_t * cliqueptr,
    const int * rowidx,
    const int * colidx,
    const float (* data)[2]);

/**
 * ‘mtxblasdense_init_cliques_complex_double()’ allocates and
 * initialises a matrix from a list of dense cliques with complex,
 * double precision coefficients.
 */
int mtxblasdense_init_cliques_complex_double(
    struct mtxblasdense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_cliques,
    const int64_t * cliqueptr,
    const int * rowidx,
    const int * colidx,
    const double (* data)[2]);

/**
 * ‘mtxblasdense_init_cliques_integer_single()’ allocates and
 * initialises a matrix from a list of dense cliques with integer,
 * single precision coefficients.
 */
int mtxblasdense_init_cliques_integer_single(
    struct mtxblasdense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_cliques,
    const int64_t * cliqueptr,
    const int * rowidx,
    const int * colidx,
    const int32_t * data);

/**
 * ‘mtxblasdense_init_cliques_integer_double()’ allocates and
 * initialises a matrix from a list of dense cliques with integer,
 * double precision coefficients.
 */
int mtxblasdense_init_cliques_integer_double(
    struct mtxblasdense * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_cliques,
    const int64_t * cliqueptr,
    const int * rowidx,
    const int * colidx,
    const int64_t * data);

/**
 * ‘mtxblasdense_init_cliques_pattern()’ allocates and initialises
 * a matrix from a list of dense cliques with boolean coefficients.
 */
int mtxblasdense_init_cliques_pattern(
    struct mtxblasdense * A,
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
 * ‘mtxblasdense_setzero()’ sets every value of a matrix to zero.
 */
int mtxblasdense_setzero(
    struct mtxblasdense * A);

/**
 * ‘mtxblasdense_set_real_single()’ sets values of a matrix based
 * on an array of single precision floating point numbers.
 */
int mtxblasdense_set_real_single(
    struct mtxblasdense * A,
    int64_t size,
    int stride,
    const float * a);

/**
 * ‘mtxblasdense_set_real_double()’ sets values of a matrix based
 * on an array of double precision floating point numbers.
 */
int mtxblasdense_set_real_double(
    struct mtxblasdense * A,
    int64_t size,
    int stride,
    const double * a);

/**
 * ‘mtxblasdense_set_complex_single()’ sets values of a matrix
 * based on an array of single precision floating point complex
 * numbers.
 */
int mtxblasdense_set_complex_single(
    struct mtxblasdense * A,
    int64_t size,
    int stride,
    const float (*a)[2]);

/**
 * ‘mtxblasdense_set_complex_double()’ sets values of a matrix
 * based on an array of double precision floating point complex
 * numbers.
 */
int mtxblasdense_set_complex_double(
    struct mtxblasdense * A,
    int64_t size,
    int stride,
    const double (*a)[2]);

/**
 * ‘mtxblasdense_set_integer_single()’ sets values of a matrix
 * based on an array of integers.
 */
int mtxblasdense_set_integer_single(
    struct mtxblasdense * A,
    int64_t size,
    int stride,
    const int32_t * a);

/**
 * ‘mtxblasdense_set_integer_double()’ sets values of a matrix
 * based on an array of integers.
 */
int mtxblasdense_set_integer_double(
    struct mtxblasdense * A,
    int64_t size,
    int stride,
    const int64_t * a);

/*
 * row and column vectors
 */

/**
 * ‘mtxblasdense_alloc_row_vector()’ allocates a row vector for a
 * given matrix, where a row vector is a vector whose length equal to
 * a single row of the matrix.
 */
int mtxblasdense_alloc_row_vector(
    const struct mtxblasdense * A,
    struct mtxvector * x,
    enum mtxvectortype vectortype);

/**
 * ‘mtxblasdense_alloc_column_vector()’ allocates a column vector
 * for a given matrix, where a column vector is a vector whose length
 * equal to a single column of the matrix.
 */
int mtxblasdense_alloc_column_vector(
    const struct mtxblasdense * A,
    struct mtxvector * y,
    enum mtxvectortype vectortype);

/*
 * convert to and from Matrix Market format
 */

/**
 * ‘mtxblasdense_from_mtxfile()’ converts a matrix from Matrix
 * Market format.
 */
int mtxblasdense_from_mtxfile(
    struct mtxblasdense * A,
    const struct mtxfile * mtxfile);

/**
 * ‘mtxblasdense_to_mtxfile()’ converts a matrix to Matrix Market
 * format.
 */
int mtxblasdense_to_mtxfile(
    struct mtxfile * mtxfile,
    const struct mtxblasdense * A,
    int64_t num_rows,
    const int64_t * rowidx,
    int64_t num_columns,
    const int64_t * colidx,
    enum mtxfileformat mtxfmt);

/*
 * Level 1 BLAS operations
 */

/**
 * ‘mtxblasdense_swap()’ swaps values of two matrices,
 * simultaneously performing ‘y <- x’ and ‘x <- y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxblasdense_swap(
    struct mtxblasdense * x,
    struct mtxblasdense * y);

/**
 * ‘mtxblasdense_copy()’ copies values of a matrix, ‘y = x’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxblasdense_copy(
    struct mtxblasdense * y,
    const struct mtxblasdense * x);

/**
 * ‘mtxblasdense_sscal()’ scales a matrix by a single precision
 * floating point scalar, ‘x = a*x’.
 */
int mtxblasdense_sscal(
    float a,
    struct mtxblasdense * x,
    int64_t * num_flops);

/**
 * ‘mtxblasdense_dscal()’ scales a matrix by a double precision
 * floating point scalar, ‘x = a*x’.
 */
int mtxblasdense_dscal(
    double a,
    struct mtxblasdense * x,
    int64_t * num_flops);

/**
 * ‘mtxblasdense_cscal()’ scales a matrix by a complex, single
 * precision floating point scalar, ‘x = (a+b*i)*x’.
 */
int mtxblasdense_cscal(
    float a[2],
    struct mtxblasdense * x,
    int64_t * num_flops);

/**
 * ‘mtxblasdense_zscal()’ scales a matrix by a complex, double
 * precision floating point scalar, ‘x = (a+b*i)*x’.
 */
int mtxblasdense_zscal(
    double a[2],
    struct mtxblasdense * x,
    int64_t * num_flops);

/**
 * ‘mtxblasdense_saxpy()’ adds a matrix to another one multiplied
 * by a single precision floating point value, ‘y = a*x + y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxblasdense_saxpy(
    float a,
    const struct mtxblasdense * x,
    struct mtxblasdense * y,
    int64_t * num_flops);

/**
 * ‘mtxblasdense_daxpy()’ adds a matrix to another one multiplied
 * by a double precision floating point value, ‘y = a*x + y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxblasdense_daxpy(
    double a,
    const struct mtxblasdense * x,
    struct mtxblasdense * y,
    int64_t * num_flops);

/**
 * ‘mtxblasdense_saypx()’ multiplies a matrix by a single precision
 * floating point scalar and adds another matrix, ‘y = a*y + x’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxblasdense_saypx(
    float a,
    struct mtxblasdense * y,
    const struct mtxblasdense * x,
    int64_t * num_flops);

/**
 * ‘mtxblasdense_daypx()’ multiplies a matrix by a double precision
 * floating point scalar and adds another matrix, ‘y = a*y + x’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxblasdense_daypx(
    double a,
    struct mtxblasdense * y,
    const struct mtxblasdense * x,
    int64_t * num_flops);

/**
 * ‘mtxblasdense_sdot()’ computes the Frobenius inner product of
 * two matrices in single precision floating point.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxblasdense_sdot(
    const struct mtxblasdense * x,
    const struct mtxblasdense * y,
    float * dot,
    int64_t * num_flops);

/**
 * ‘mtxblasdense_ddot()’ computes the Frobenius inner product of
 * two matrices in double precision floating point.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxblasdense_ddot(
    const struct mtxblasdense * x,
    const struct mtxblasdense * y,
    double * dot,
    int64_t * num_flops);

/**
 * ‘mtxblasdense_cdotu()’ computes the product of the transpose of
 * a complex row matrix with another complex row matrix in single
 * precision floating point, ‘dot := x^T*y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxblasdense_cdotu(
    const struct mtxblasdense * x,
    const struct mtxblasdense * y,
    float (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxblasdense_zdotu()’ computes the product of the transpose of
 * a complex row matrix with another complex row matrix in double
 * precision floating point, ‘dot := x^T*y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxblasdense_zdotu(
    const struct mtxblasdense * x,
    const struct mtxblasdense * y,
    double (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxblasdense_cdotc()’ computes the Frobenius inner product of
 * two complex matrices in single precision floating point, ‘dot :=
 * x^H*y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxblasdense_cdotc(
    const struct mtxblasdense * x,
    const struct mtxblasdense * y,
    float (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxblasdense_zdotc()’ computes the Frobenius inner product of
 * two complex matrices in double precision floating point, ‘dot :=
 * x^H*y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxblasdense_zdotc(
    const struct mtxblasdense * x,
    const struct mtxblasdense * y,
    double (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxblasdense_snrm2()’ computes the Frobenius norm of a matrix
 * in single precision floating point.
 */
int mtxblasdense_snrm2(
    const struct mtxblasdense * x,
    float * nrm2,
    int64_t * num_flops);

/**
 * ‘mtxblasdense_dnrm2()’ computes the Frobenius norm of a matrix
 * in double precision floating point.
 */
int mtxblasdense_dnrm2(
    const struct mtxblasdense * x,
    double * nrm2,
    int64_t * num_flops);

/**
 * ‘mtxblasdense_sasum()’ computes the sum of absolute values
 * (1-norm) of a matrix in single precision floating point.  If the
 * matrix is complex-valued, then the sum of the absolute values of
 * the real and imaginary parts is computed.
 */
int mtxblasdense_sasum(
    const struct mtxblasdense * x,
    float * asum,
    int64_t * num_flops);

/**
 * ‘mtxblasdense_dasum()’ computes the sum of absolute values
 * (1-norm) of a matrix in double precision floating point.  If the
 * matrix is complex-valued, then the sum of the absolute values of
 * the real and imaginary parts is computed.
 */
int mtxblasdense_dasum(
    const struct mtxblasdense * x,
    double * asum,
    int64_t * num_flops);

/**
 * ‘mtxblasdense_iamax()’ finds the index of the first element
 * having the maximum absolute value.  If the matrix is
 * complex-valued, then the index points to the first element having
 * the maximum sum of the absolute values of the real and imaginary
 * parts.
 */
int mtxblasdense_iamax(
    const struct mtxblasdense * x,
    int * iamax);

/*
 * Level 2 BLAS operations (matrix-vector)
 */

/**
 * ‘mtxblasdense_sgemv()’ multiplies a matrix ‘A’ or its transpose
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
int mtxblasdense_sgemv(
    enum mtxtransposition trans,
    float alpha,
    const struct mtxblasdense * A,
    const struct mtxvector * x,
    float beta,
    struct mtxvector * y,
    int64_t * num_flops);

/**
 * ‘mtxblasdense_dgemv()’ multiplies a matrix ‘A’ or its transpose
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
int mtxblasdense_dgemv(
    enum mtxtransposition trans,
    double alpha,
    const struct mtxblasdense * A,
    const struct mtxvector * x,
    double beta,
    struct mtxvector * y,
    int64_t * num_flops);

/**
 * ‘mtxblasdense_cgemv()’ multiplies a complex-valued matrix ‘A’,
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
int mtxblasdense_cgemv(
    enum mtxtransposition trans,
    float alpha[2],
    const struct mtxblasdense * A,
    const struct mtxvector * x,
    float beta[2],
    struct mtxvector * y,
    int64_t * num_flops);

/**
 * ‘mtxblasdense_zgemv()’ multiplies a complex-valued matrix ‘A’,
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
int mtxblasdense_zgemv(
    enum mtxtransposition trans,
    double alpha[2],
    const struct mtxblasdense * A,
    const struct mtxvector * x,
    double beta[2],
    struct mtxvector * y,
    int64_t * num_flops);

#endif
