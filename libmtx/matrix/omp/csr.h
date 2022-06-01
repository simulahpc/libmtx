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
 * Last modified: 2022-05-28
 *
 * Matrices in CSR format with OpenMP shared-memory parallelism.
 */

#ifndef LIBMTX_MATRIX_OMP_CSR_H
#define LIBMTX_MATRIX_OMP_CSR_H

#include <libmtx/libmtx-config.h>

#include <libmtx/vector/precision.h>
#include <libmtx/vector/field.h>
#include <libmtx/matrix/symmetry.h>
#include <libmtx/matrix/transpose.h>
#include <libmtx/vector/omp.h>
#include <libmtx/vector/packed.h>
#include <libmtx/vector/vector.h>

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

struct mtxfile;
struct mtxmatrix;

/**
 * ‘mtxmatrix_ompcsr’ represents a matrix in compressed sparse row (CSR)
 * format.
 */
struct mtxmatrix_ompcsr
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
     * ‘rowptr’ is an array containing row pointers. Since nonzeros
     * are arranged rowwise, the entries of this array indicate the
     * position of the first nonzero of each row. There is also an
     * additional, final entry that is equal to the total number of
     * nonzeros.
     */
    int64_t * rowptr;

    /**
     * ‘colidx’ is an array containing the column indices of nonzero
     * matrix entries.  Note that column indices are 0-based, unlike
     * the Matrix Market format, where indices are 1-based.
     */
    int * colidx;

    /**
     * ‘a’ is a vector storing the underlying nonzero matrix entries.
     */
    struct mtxvector_omp a;

    /**
     * ‘diag’ is a vector storing the diagonal nonzero matrix entries.
     */
    struct mtxvector_packed diag;
};

/**
 * ‘mtxmatrix_ompcsr_field()’ gets the field of a matrix.
 */
enum mtxfield mtxmatrix_ompcsr_field(
    const struct mtxmatrix_ompcsr * A);

/**
 * ‘mtxmatrix_ompcsr_precision()’ gets the precision of a matrix.
 */
enum mtxprecision mtxmatrix_ompcsr_precision(
    const struct mtxmatrix_ompcsr * A);

/*
 * memory management
 */

/**
 * ‘mtxmatrix_ompcsr_free()’ frees storage allocated for a matrix.
 */
void mtxmatrix_ompcsr_free(
    struct mtxmatrix_ompcsr * A);

/**
 * ‘mtxmatrix_ompcsr_alloc_copy()’ allocates a copy of a matrix without
 * initialising the values.
 */
int mtxmatrix_ompcsr_alloc_copy(
    struct mtxmatrix_ompcsr * dst,
    const struct mtxmatrix_ompcsr * src);

/**
 * ‘mtxmatrix_ompcsr_init_copy()’ allocates a copy of a matrix and also
 * copies the values.
 */
int mtxmatrix_ompcsr_init_copy(
    struct mtxmatrix_ompcsr * dst,
    const struct mtxmatrix_ompcsr * src);

/*
 * initialise matrices from entrywise data in coordinate format
 */

/**
 * ‘mtxmatrix_ompcsr_alloc_entries()’ allocates a matrix from entrywise
 * data in coordinate format.
 *
 * If it is not ‘NULL’, then ‘perm’ must point to an array of length
 * ‘size’. Because the sparse matrix storage may internally reorder
 * the specified nonzero entries, this array is used to store the
 * permutation applied to the specified nonzero entries.
 */
int mtxmatrix_ompcsr_alloc_entries(
    struct mtxmatrix_ompcsr * A,
    enum mtxfield field,
    enum mtxprecision precision,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    int idxstride,
    int idxbase,
    const int * rowidx,
    const int * colidx,
    int64_t * perm);

/**
 * ‘mtxmatrix_ompcsr_init_entries_real_single()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * real, single precision coefficients.
 */
int mtxmatrix_ompcsr_init_entries_real_single(
    struct mtxmatrix_ompcsr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx,
    const float * data);

/**
 * ‘mtxmatrix_ompcsr_init_entries_real_double()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * real, double precision coefficients.
 */
int mtxmatrix_ompcsr_init_entries_real_double(
    struct mtxmatrix_ompcsr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx,
    const double * data);

/**
 * ‘mtxmatrix_ompcsr_init_entries_complex_single()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * complex, single precision coefficients.
 */
int mtxmatrix_ompcsr_init_entries_complex_single(
    struct mtxmatrix_ompcsr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx,
    const float (* data)[2]);

/**
 * ‘mtxmatrix_ompcsr_init_entries_complex_double()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * complex, double precision coefficients.
 */
int mtxmatrix_ompcsr_init_entries_complex_double(
    struct mtxmatrix_ompcsr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx,
    const double (* data)[2]);

/**
 * ‘mtxmatrix_ompcsr_init_entries_integer_single()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * integer, single precision coefficients.
 */
int mtxmatrix_ompcsr_init_entries_integer_single(
    struct mtxmatrix_ompcsr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx,
    const int32_t * data);

/**
 * ‘mtxmatrix_ompcsr_init_entries_integer_double()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * integer, double precision coefficients.
 */
int mtxmatrix_ompcsr_init_entries_integer_double(
    struct mtxmatrix_ompcsr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx,
    const int64_t * data);

/**
 * ‘mtxmatrix_ompcsr_init_entries_pattern()’ allocates and initialises a
 * matrix from entrywise data in coordinate format with boolean
 * coefficients.
 */
int mtxmatrix_ompcsr_init_entries_pattern(
    struct mtxmatrix_ompcsr * A,
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
 * ‘mtxmatrix_ompcsr_init_entries_strided_real_single()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * real, single precision coefficients.
 */
int mtxmatrix_ompcsr_init_entries_strided_real_single(
    struct mtxmatrix_ompcsr * A,
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
 * ‘mtxmatrix_ompcsr_init_entries_strided_real_double()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * real, double precision coefficients.
 */
int mtxmatrix_ompcsr_init_entries_strided_real_double(
    struct mtxmatrix_ompcsr * A,
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
 * ‘mtxmatrix_ompcsr_init_entries_strided_complex_single()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * complex, single precision coefficients.
 */
int mtxmatrix_ompcsr_init_entries_strided_complex_single(
    struct mtxmatrix_ompcsr * A,
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
 * ‘mtxmatrix_ompcsr_init_entries_strided_complex_double()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * complex, double precision coefficients.
 */
int mtxmatrix_ompcsr_init_entries_strided_complex_double(
    struct mtxmatrix_ompcsr * A,
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
 * ‘mtxmatrix_ompcsr_init_entries_strided_integer_single()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * integer, single precision coefficients.
 */
int mtxmatrix_ompcsr_init_entries_strided_integer_single(
    struct mtxmatrix_ompcsr * A,
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
 * ‘mtxmatrix_ompcsr_init_entries_strided_integer_double()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * integer, double precision coefficients.
 */
int mtxmatrix_ompcsr_init_entries_strided_integer_double(
    struct mtxmatrix_ompcsr * A,
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
 * ‘mtxmatrix_ompcsr_init_entries_strided_pattern()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * boolean coefficients.
 */
int mtxmatrix_ompcsr_init_entries_strided_pattern(
    struct mtxmatrix_ompcsr * A,
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
 * ‘mtxmatrix_ompcsr_alloc_rows()’ allocates a matrix from row-wise data
 * in compressed row format.
 */
int mtxmatrix_ompcsr_alloc_rows(
    struct mtxmatrix_ompcsr * A,
    enum mtxfield field,
    enum mtxprecision precision,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx);

/**
 * ‘mtxmatrix_ompcsr_init_rows_real_single()’ allocates and initialises a
 * matrix from row-wise data in compressed row format with real,
 * single precision coefficients.
 */
int mtxmatrix_ompcsr_init_rows_real_single(
    struct mtxmatrix_ompcsr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const float * data);

/**
 * ‘mtxmatrix_ompcsr_init_rows_real_double()’ allocates and initialises a
 * matrix from row-wise data in compressed row format with real,
 * double precision coefficients.
 */
int mtxmatrix_ompcsr_init_rows_real_double(
    struct mtxmatrix_ompcsr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const double * data);

/**
 * ‘mtxmatrix_ompcsr_init_rows_complex_single()’ allocates and
 * initialises a matrix from row-wise data in compressed row format
 * with complex, single precision coefficients.
 */
int mtxmatrix_ompcsr_init_rows_complex_single(
    struct mtxmatrix_ompcsr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const float (* data)[2]);

/**
 * ‘mtxmatrix_ompcsr_init_rows_complex_double()’ allocates and
 * initialises a matrix from row-wise data in compressed row format
 * with complex, double precision coefficients.
 */
int mtxmatrix_ompcsr_init_rows_complex_double(
    struct mtxmatrix_ompcsr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const double (* data)[2]);

/**
 * ‘mtxmatrix_ompcsr_init_rows_integer_single()’ allocates and
 * initialises a matrix from row-wise data in compressed row format
 * with integer, single precision coefficients.
 */
int mtxmatrix_ompcsr_init_rows_integer_single(
    struct mtxmatrix_ompcsr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const int32_t * data);

/**
 * ‘mtxmatrix_ompcsr_init_rows_integer_double()’ allocates and
 * initialises a matrix from row-wise data in compressed row format
 * with integer, double precision coefficients.
 */
int mtxmatrix_ompcsr_init_rows_integer_double(
    struct mtxmatrix_ompcsr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const int64_t * data);

/**
 * ‘mtxmatrix_ompcsr_init_rows_pattern()’ allocates and initialises a
 * matrix from row-wise data in compressed row format with boolean
 * coefficients.
 */
int mtxmatrix_ompcsr_init_rows_pattern(
    struct mtxmatrix_ompcsr * A,
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
 * ‘mtxmatrix_ompcsr_alloc_columns()’ allocates a matrix from column-wise
 * data in compressed column format.
 */
int mtxmatrix_ompcsr_alloc_columns(
    struct mtxmatrix_ompcsr * A,
    enum mtxfield field,
    enum mtxprecision precision,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx);

/**
 * ‘mtxmatrix_ompcsr_init_columns_real_single()’ allocates and
 * initialises a matrix from column-wise data in compressed column
 * format with real, single precision coefficients.
 */
int mtxmatrix_ompcsr_init_columns_real_single(
    struct mtxmatrix_ompcsr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx,
    const float * data);

/**
 * ‘mtxmatrix_ompcsr_init_columns_real_double()’ allocates and
 * initialises a matrix from column-wise data in compressed column
 * format with real, double precision coefficients.
 */
int mtxmatrix_ompcsr_init_columns_real_double(
    struct mtxmatrix_ompcsr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx,
    const double * data);

/**
 * ‘mtxmatrix_ompcsr_init_columns_complex_single()’ allocates and
 * initialises a matrix from column-wise data in compressed column
 * format with complex, single precision coefficients.
 */
int mtxmatrix_ompcsr_init_columns_complex_single(
    struct mtxmatrix_ompcsr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx,
    const float (* data)[2]);

/**
 * ‘mtxmatrix_ompcsr_init_columns_complex_double()’ allocates and
 * initialises a matrix from column-wise data in compressed column
 * format with complex, double precision coefficients.
 */
int mtxmatrix_ompcsr_init_columns_complex_double(
    struct mtxmatrix_ompcsr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx,
    const double (* data)[2]);

/**
 * ‘mtxmatrix_ompcsr_init_columns_integer_single()’ allocates and
 * initialises a matrix from column-wise data in compressed column
 * format with integer, single precision coefficients.
 */
int mtxmatrix_ompcsr_init_columns_integer_single(
    struct mtxmatrix_ompcsr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx,
    const int32_t * data);

/**
 * ‘mtxmatrix_ompcsr_init_columns_integer_double()’ allocates and
 * initialises a matrix from column-wise data in compressed column
 * format with integer, double precision coefficients.
 */
int mtxmatrix_ompcsr_init_columns_integer_double(
    struct mtxmatrix_ompcsr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx,
    const int64_t * data);

/**
 * ‘mtxmatrix_ompcsr_init_columns_pattern()’ allocates and initialises a
 * matrix from column-wise data in compressed column format with
 * boolean coefficients.
 */
int mtxmatrix_ompcsr_init_columns_pattern(
    struct mtxmatrix_ompcsr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx);

/*
 * initialise matrices from a list of dense cliques
 */

/**
 * ‘mtxmatrix_ompcsr_alloc_cliques()’ allocates a matrix from a list of
 * dense cliques.
 */
int mtxmatrix_ompcsr_alloc_cliques(
    struct mtxmatrix_ompcsr * A,
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
 * ‘mtxmatrix_ompcsr_init_cliques_real_single()’ allocates and
 * initialises a matrix from a list of dense cliques with real, single
 * precision coefficients.
 */
int mtxmatrix_ompcsr_init_cliques_real_single(
    struct mtxmatrix_ompcsr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_cliques,
    const int64_t * cliqueptr,
    const int * rowidx,
    const int * colidx,
    const float * data);

/**
 * ‘mtxmatrix_ompcsr_init_cliques_real_double()’ allocates and
 * initialises a matrix from a list of dense cliques with real, double
 * precision coefficients.
 */
int mtxmatrix_ompcsr_init_cliques_real_double(
    struct mtxmatrix_ompcsr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_cliques,
    const int64_t * cliqueptr,
    const int * rowidx,
    const int * colidx,
    const double * data);

/**
 * ‘mtxmatrix_ompcsr_init_cliques_complex_single()’ allocates and
 * initialises a matrix from a list of dense cliques with complex,
 * single precision coefficients.
 */
int mtxmatrix_ompcsr_init_cliques_complex_single(
    struct mtxmatrix_ompcsr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_cliques,
    const int64_t * cliqueptr,
    const int * rowidx,
    const int * colidx,
    const float (* data)[2]);

/**
 * ‘mtxmatrix_ompcsr_init_cliques_complex_double()’ allocates and
 * initialises a matrix from a list of dense cliques with complex,
 * double precision coefficients.
 */
int mtxmatrix_ompcsr_init_cliques_complex_double(
    struct mtxmatrix_ompcsr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_cliques,
    const int64_t * cliqueptr,
    const int * rowidx,
    const int * colidx,
    const double (* data)[2]);

/**
 * ‘mtxmatrix_ompcsr_init_cliques_integer_single()’ allocates and
 * initialises a matrix from a list of dense cliques with integer,
 * single precision coefficients.
 */
int mtxmatrix_ompcsr_init_cliques_integer_single(
    struct mtxmatrix_ompcsr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_cliques,
    const int64_t * cliqueptr,
    const int * rowidx,
    const int * colidx,
    const int32_t * data);

/**
 * ‘mtxmatrix_ompcsr_init_cliques_integer_double()’ allocates and
 * initialises a matrix from a list of dense cliques with integer,
 * double precision coefficients.
 */
int mtxmatrix_ompcsr_init_cliques_integer_double(
    struct mtxmatrix_ompcsr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_cliques,
    const int64_t * cliqueptr,
    const int * rowidx,
    const int * colidx,
    const int64_t * data);

/**
 * ‘mtxmatrix_ompcsr_init_cliques_pattern()’ allocates and initialises a
 * matrix from a list of dense cliques with boolean coefficients.
 */
int mtxmatrix_ompcsr_init_cliques_pattern(
    struct mtxmatrix_ompcsr * A,
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
 * ‘mtxmatrix_ompcsr_setzero()’ sets every value of a matrix to zero.
 */
int mtxmatrix_ompcsr_setzero(
    struct mtxmatrix_ompcsr * A);

/**
 * ‘mtxmatrix_ompcsr_set_real_single()’ sets values of a matrix based on
 * an array of single precision floating point numbers.
 */
int mtxmatrix_ompcsr_set_real_single(
    struct mtxmatrix_ompcsr * A,
    int64_t size,
    int stride,
    const float * a);

/**
 * ‘mtxmatrix_ompcsr_set_real_double()’ sets values of a matrix based on
 * an array of double precision floating point numbers.
 */
int mtxmatrix_ompcsr_set_real_double(
    struct mtxmatrix_ompcsr * A,
    int64_t size,
    int stride,
    const double * a);

/**
 * ‘mtxmatrix_ompcsr_set_complex_single()’ sets values of a matrix based
 * on an array of single precision floating point complex numbers.
 */
int mtxmatrix_ompcsr_set_complex_single(
    struct mtxmatrix_ompcsr * A,
    int64_t size,
    int stride,
    const float (*a)[2]);

/**
 * ‘mtxmatrix_ompcsr_set_complex_double()’ sets values of a matrix based
 * on an array of double precision floating point complex numbers.
 */
int mtxmatrix_ompcsr_set_complex_double(
    struct mtxmatrix_ompcsr * A,
    int64_t size,
    int stride,
    const double (*a)[2]);

/**
 * ‘mtxmatrix_ompcsr_set_integer_single()’ sets values of a matrix based
 * on an array of integers.
 */
int mtxmatrix_ompcsr_set_integer_single(
    struct mtxmatrix_ompcsr * A,
    int64_t size,
    int stride,
    const int32_t * a);

/**
 * ‘mtxmatrix_ompcsr_set_integer_double()’ sets values of a matrix based
 * on an array of integers.
 */
int mtxmatrix_ompcsr_set_integer_double(
    struct mtxmatrix_ompcsr * A,
    int64_t size,
    int stride,
    const int64_t * a);

/*
 * row and column vectors
 */

/**
 * ‘mtxmatrix_ompcsr_alloc_row_vector()’ allocates a row vector for a
 * given matrix, where a row vector is a vector whose length equal to
 * a single row of the matrix.
 */
int mtxmatrix_ompcsr_alloc_row_vector(
    const struct mtxmatrix_ompcsr * A,
    struct mtxvector * x,
    enum mtxvectortype vectortype);

/**
 * ‘mtxmatrix_ompcsr_alloc_column_vector()’ allocates a column vector for
 * a given matrix, where a column vector is a vector whose length
 * equal to a single column of the matrix.
 */
int mtxmatrix_ompcsr_alloc_column_vector(
    const struct mtxmatrix_ompcsr * A,
    struct mtxvector * y,
    enum mtxvectortype vectortype);

/*
 * convert to and from Matrix Market format
 */

/**
 * ‘mtxmatrix_ompcsr_from_mtxfile()’ converts a matrix from Matrix Market
 * format.
 */
int mtxmatrix_ompcsr_from_mtxfile(
    struct mtxmatrix_ompcsr * A,
    const struct mtxfile * mtxfile);

/**
 * ‘mtxmatrix_ompcsr_to_mtxfile()’ converts a matrix to Matrix Market
 *  format.
 */
int mtxmatrix_ompcsr_to_mtxfile(
    struct mtxfile * mtxfile,
    const struct mtxmatrix_ompcsr * A,
    int64_t num_rows,
    const int64_t * rowidx,
    int64_t num_columns,
    const int64_t * colidx,
    enum mtxfileformat mtxfmt);

/*
 * partitioning
 */

/**
 * ‘mtxmatrix_ompcsr_partition_rowwise()’ partitions the entries of a
 * matrix rowwise.
 *
 * See ‘partition_int()’ for an explanation of the meaning of the
 * arguments ‘parttype’, ‘num_parts’, ‘partsizes’, ‘blksize’ and
 * ‘parts’.
 *
 * The length of the array ‘dstpart’ must be at least equal to the
 * number of (nonzero) matrix entries (which can be obtained by
 * calling ‘mtxmatrix_size()’). If successful, ‘dstpart’ is used to
 * store the part numbers assigned to the matrix nonzeros.
 *
 * If ‘dstpartsizes’ is not ‘NULL’, then it must be an array of length
 * ‘num_parts’, which is used to store the number of nonzeros assigned
 * to each part.
 */
int mtxmatrix_ompcsr_partition_rowwise(
    const struct mtxmatrix_ompcsr * A,
    enum mtxpartitioning parttype,
    int num_parts,
    const int * partsizes,
    int blksize,
    const int * parts,
    int * dstpart,
    int64_t * dstpartsizes);

/**
 * ‘mtxmatrix_ompcsr_partition_columnwise()’ partitions the entries of
 * a matrix columnwise.
 *
 * See ‘partition_int()’ for an explanation of the meaning of the
 * arguments ‘parttype’, ‘num_parts’, ‘partsizes’, ‘blksize’ and
 * ‘parts’.
 *
 * The length of the array ‘dstpart’ must be at least equal to the
 * number of (nonzero) matrix entries (which can be obtained by
 * calling ‘mtxmatrix_size()’). If successful, ‘dstpart’ is used to
 * store the part numbers assigned to the matrix nonzeros.
 *
 * If ‘dstpartsizes’ is not ‘NULL’, then it must be an array of length
 * ‘num_parts’, which is used to store the number of nonzeros assigned
 * to each part.
 */
int mtxmatrix_ompcsr_partition_columnwise(
    const struct mtxmatrix_ompcsr * A,
    enum mtxpartitioning parttype,
    int num_parts,
    const int * partsizes,
    int blksize,
    const int * parts,
    int * dstpart,
    int64_t * dstpartsizes);

/**
 * ‘mtxmatrix_ompcsr_split()’ splits a matrix into multiple matrices
 * according to a given assignment of parts to each nonzero matrix
 * element.
 *
 * The partitioning of the nonzero matrix elements is specified by the
 * array ‘parts’. The length of the ‘parts’ array is given by ‘size’,
 * which must match the number of explicitly stored nonzero matrix
 * entries in ‘src’. Each entry in the ‘parts’ array is an integer in
 * the range ‘[0, num_parts)’ designating the part to which the
 * corresponding matrix nonzero belongs.
 *
 * The argument ‘dsts’ is an array of ‘num_parts’ pointers to objects
 * of type ‘struct mtxmatrix_ompcsr’. If successful, then ‘dsts[p]’
 * points to a matrix consisting of elements from ‘src’ that belong to
 * the ‘p’th part, as designated by the ‘parts’ array.
 *
 * The caller is responsible for calling ‘mtxmatrix_ompcsr_free()’ to
 * free storage allocated for each matrix in the ‘dsts’ array.
 */
int mtxmatrix_ompcsr_split(
    int num_parts,
    struct mtxmatrix_ompcsr ** dsts,
    const struct mtxmatrix_ompcsr * src,
    int64_t size,
    int * parts);

/*
 * Level 1 BLAS operations
 */

/**
 * ‘mtxmatrix_ompcsr_swap()’ swaps values of two matrices, simultaneously
 * performing ‘y <- x’ and ‘x <- y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_ompcsr_swap(
    struct mtxmatrix_ompcsr * x,
    struct mtxmatrix_ompcsr * y);

/**
 * ‘mtxmatrix_ompcsr_copy()’ copies values of a matrix, ‘y = x’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_ompcsr_copy(
    struct mtxmatrix_ompcsr * y,
    const struct mtxmatrix_ompcsr * x);

/**
 * ‘mtxmatrix_ompcsr_sscal()’ scales a matrix by a single precision
 * floating point scalar, ‘x = a*x’.
 */
int mtxmatrix_ompcsr_sscal(
    float a,
    struct mtxmatrix_ompcsr * x,
    int64_t * num_flops);

/**
 * ‘mtxmatrix_ompcsr_dscal()’ scales a matrix by a double precision
 * floating point scalar, ‘x = a*x’.
 */
int mtxmatrix_ompcsr_dscal(
    double a,
    struct mtxmatrix_ompcsr * x,
    int64_t * num_flops);

/**
 * ‘mtxmatrix_ompcsr_cscal()’ scales a matrix by a complex, single
 * precision floating point scalar, ‘x = (a+b*i)*x’.
 */
int mtxmatrix_ompcsr_cscal(
    float a[2],
    struct mtxmatrix_ompcsr * x,
    int64_t * num_flops);

/**
 * ‘mtxmatrix_ompcsr_zscal()’ scales a matrix by a complex, double
 * precision floating point scalar, ‘x = (a+b*i)*x’.
 */
int mtxmatrix_ompcsr_zscal(
    double a[2],
    struct mtxmatrix_ompcsr * x,
    int64_t * num_flops);

/**
 * ‘mtxmatrix_ompcsr_saxpy()’ adds a matrix to another one multiplied by
 * a single precision floating point value, ‘y = a*x + y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_ompcsr_saxpy(
    float a,
    const struct mtxmatrix_ompcsr * x,
    struct mtxmatrix_ompcsr * y,
    int64_t * num_flops);

/**
 * ‘mtxmatrix_ompcsr_daxpy()’ adds a matrix to another one multiplied by
 * a double precision floating point value, ‘y = a*x + y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_ompcsr_daxpy(
    double a,
    const struct mtxmatrix_ompcsr * x,
    struct mtxmatrix_ompcsr * y,
    int64_t * num_flops);

/**
 * ‘mtxmatrix_ompcsr_saypx()’ multiplies a matrix by a single precision
 * floating point scalar and adds another matrix, ‘y = a*y + x’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_ompcsr_saypx(
    float a,
    struct mtxmatrix_ompcsr * y,
    const struct mtxmatrix_ompcsr * x,
    int64_t * num_flops);

/**
 * ‘mtxmatrix_ompcsr_daypx()’ multiplies a matrix by a double precision
 * floating point scalar and adds another matrix, ‘y = a*y + x’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_ompcsr_daypx(
    double a,
    struct mtxmatrix_ompcsr * y,
    const struct mtxmatrix_ompcsr * x,
    int64_t * num_flops);

/**
 * ‘mtxmatrix_ompcsr_sdot()’ computes the Frobenius inner product of two
 * matrices in single precision floating point.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_ompcsr_sdot(
    const struct mtxmatrix_ompcsr * x,
    const struct mtxmatrix_ompcsr * y,
    float * dot,
    int64_t * num_flops);

/**
 * ‘mtxmatrix_ompcsr_ddot()’ computes the Frobenius inner product of two
 * matrices in double precision floating point.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_ompcsr_ddot(
    const struct mtxmatrix_ompcsr * x,
    const struct mtxmatrix_ompcsr * y,
    double * dot,
    int64_t * num_flops);

/**
 * ‘mtxmatrix_ompcsr_cdotu()’ computes the product of the transpose of a
 * complex row matrix with another complex row matrix in single
 * precision floating point, ‘dot := x^T*y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_ompcsr_cdotu(
    const struct mtxmatrix_ompcsr * x,
    const struct mtxmatrix_ompcsr * y,
    float (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxmatrix_ompcsr_zdotu()’ computes the product of the transpose of a
 * complex row matrix with another complex row matrix in double
 * precision floating point, ‘dot := x^T*y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_ompcsr_zdotu(
    const struct mtxmatrix_ompcsr * x,
    const struct mtxmatrix_ompcsr * y,
    double (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxmatrix_ompcsr_cdotc()’ computes the Frobenius inner product of two
 * complex matrices in single precision floating point, ‘dot :=
 * x^H*y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_ompcsr_cdotc(
    const struct mtxmatrix_ompcsr * x,
    const struct mtxmatrix_ompcsr * y,
    float (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxmatrix_ompcsr_zdotc()’ computes the Frobenius inner product of two
 * complex matrices in double precision floating point, ‘dot :=
 * x^H*y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_ompcsr_zdotc(
    const struct mtxmatrix_ompcsr * x,
    const struct mtxmatrix_ompcsr * y,
    double (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxmatrix_ompcsr_snrm2()’ computes the Frobenius norm of a matrix in
 * single precision floating point.
 */
int mtxmatrix_ompcsr_snrm2(
    const struct mtxmatrix_ompcsr * x,
    float * nrm2,
    int64_t * num_flops);

/**
 * ‘mtxmatrix_ompcsr_dnrm2()’ computes the Frobenius norm of a matrix in
 * double precision floating point.
 */
int mtxmatrix_ompcsr_dnrm2(
    const struct mtxmatrix_ompcsr * x,
    double * nrm2,
    int64_t * num_flops);

/**
 * ‘mtxmatrix_ompcsr_sasum()’ computes the sum of absolute values
 * (1-norm) of a matrix in single precision floating point.  If the
 * matrix is complex-valued, then the sum of the absolute values of
 * the real and imaginary parts is computed.
 */
int mtxmatrix_ompcsr_sasum(
    const struct mtxmatrix_ompcsr * x,
    float * asum,
    int64_t * num_flops);

/**
 * ‘mtxmatrix_ompcsr_dasum()’ computes the sum of absolute values
 * (1-norm) of a matrix in double precision floating point.  If the
 * matrix is complex-valued, then the sum of the absolute values of
 * the real and imaginary parts is computed.
 */
int mtxmatrix_ompcsr_dasum(
    const struct mtxmatrix_ompcsr * x,
    double * asum,
    int64_t * num_flops);

/**
 * ‘mtxmatrix_ompcsr_iamax()’ finds the index of the first element having
 * the maximum absolute value.  If the matrix is complex-valued, then
 * the index points to the first element having the maximum sum of the
 * absolute values of the real and imaginary parts.
 */
int mtxmatrix_ompcsr_iamax(
    const struct mtxmatrix_ompcsr * x,
    int * iamax);

/*
 * Level 2 BLAS operations (matrix-vector)
 */

/**
 * ‘mtxmatrix_ompcsr_sgemv()’ multiplies a matrix ‘A’ or its transpose
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
 *
 * For now, the only case that is parallelised with OpenMP is
 * multiplication with non-transposed and unsymmetric matrices.
 */
int mtxmatrix_ompcsr_sgemv(
    enum mtxtransposition trans,
    float alpha,
    const struct mtxmatrix_ompcsr * A,
    const struct mtxvector * x,
    float beta,
    struct mtxvector * y,
    int64_t * num_flops);

/**
 * ‘mtxmatrix_ompcsr_dgemv()’ multiplies a matrix ‘A’ or its transpose
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
 *
 * For now, the only case that is parallelised with OpenMP is
 * multiplication with non-transposed and unsymmetric matrices.
 */
int mtxmatrix_ompcsr_dgemv(
    enum mtxtransposition trans,
    double alpha,
    const struct mtxmatrix_ompcsr * A,
    const struct mtxvector * x,
    double beta,
    struct mtxvector * y,
    int64_t * num_flops);

/**
 * ‘mtxmatrix_ompcsr_cgemv()’ multiplies a complex-valued matrix ‘A’, its
 * transpose ‘A'’ or its conjugate transpose ‘Aᴴ’ by a complex scalar
 * ‘alpha’ (‘α’) and a vector ‘x’, before adding the result to another
 * vector ‘y’ multiplied by another complex scalar ‘beta’ (‘β’).  That
 * is, ‘y = α*A*x + β*y’, ‘y = α*A'*x + β*y’ or ‘y = α*Aᴴ*x + β*y’.
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
 *
 * For now, the only case that is parallelised with OpenMP is
 * multiplication with non-transposed and unsymmetric matrices.
 */
int mtxmatrix_ompcsr_cgemv(
    enum mtxtransposition trans,
    float alpha[2],
    const struct mtxmatrix_ompcsr * A,
    const struct mtxvector * x,
    float beta[2],
    struct mtxvector * y,
    int64_t * num_flops);

/**
 * ‘mtxmatrix_ompcsr_zgemv()’ multiplies a complex-valued matrix ‘A’, its
 * transpose ‘A'’ or its conjugate transpose ‘Aᴴ’ by a complex scalar
 * ‘alpha’ (‘α’) and a vector ‘x’, before adding the result to another
 * vector ‘y’ multiplied by another complex scalar ‘beta’ (‘β’).  That
 * is, ‘y = α*A*x + β*y’, ‘y = α*A'*x + β*y’ or ‘y = α*Aᴴ*x + β*y’.
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
 *
 * For now, the only case that is parallelised with OpenMP is
 * multiplication with non-transposed and unsymmetric matrices.
 */
int mtxmatrix_ompcsr_zgemv(
    enum mtxtransposition trans,
    double alpha[2],
    const struct mtxmatrix_ompcsr * A,
    const struct mtxvector * x,
    double beta[2],
    struct mtxvector * y,
    int64_t * num_flops);

#endif
