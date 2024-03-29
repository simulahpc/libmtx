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
 * Matrices in CSR format.
 */

#ifndef LIBMTX_LINALG_BASE_CSR_H
#define LIBMTX_LINALG_BASE_CSR_H

#include <libmtx/libmtx-config.h>

#include <libmtx/linalg/precision.h>
#include <libmtx/linalg/field.h>
#include <libmtx/linalg/symmetry.h>
#include <libmtx/linalg/transpose.h>
#include <libmtx/linalg/base/vector.h>
#include <libmtx/linalg/local/vector.h>

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

struct mtxfile;
struct mtxmatrix;

/**
 * ‘mtxbasecsr’ represents a matrix in compressed sparse row (CSR)
 * format.
 */
struct mtxbasecsr
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
    struct mtxbasevector a;

    /**
     * ‘diag’ is a vector storing the diagonal nonzero matrix entries.
     */
    struct mtxbasevector diag;
};

/*
 * matrix properties
 */

/**
 * ‘mtxbasecsr_field()’ gets the field of a matrix.
 */
enum mtxfield mtxbasecsr_field(const struct mtxbasecsr * A);

/**
 * ‘mtxbasecsr_precision()’ gets the precision of a matrix.
 */
enum mtxprecision mtxbasecsr_precision(const struct mtxbasecsr * A);

/**
 * ‘mtxbasecsr_symmetry()’ gets the symmetry of a matrix.
 */
enum mtxsymmetry mtxbasecsr_symmetry(const struct mtxbasecsr * A);

/**
 * ‘mtxbasecsr_num_rows()’ gets the number of matrix rows.
 */
int mtxbasecsr_num_rows(const struct mtxbasecsr * A);

/**
 * ‘mtxbasecsr_num_columns()’ gets the number of matrix columns.
 */
int mtxbasecsr_num_columns(const struct mtxbasecsr * A);

/**
 * ‘mtxbasecsr_num_nonzeros()’ gets the number of the number of
 *  nonzero matrix entries, including those represented implicitly due
 *  to symmetry.
 */
int64_t mtxbasecsr_num_nonzeros(const struct mtxbasecsr * A);

/**
 * ‘mtxbasecsr_size()’ gets the number of explicitly stored
 * nonzeros of a matrix.
 */
int64_t mtxbasecsr_size(const struct mtxbasecsr * A);

/**
 * ‘mtxbasecsr_rowcolidx()’ gets the row and column indices of the
 * explicitly stored matrix nonzeros.
 *
 * The arguments ‘rowidx’ and ‘colidx’ may be ‘NULL’ or must point to
 * an arrays of length ‘size’.
 */
int mtxbasecsr_rowcolidx(
    const struct mtxbasecsr * A,
    int64_t size,
    int * rowidx,
    int * colidx);

/*
 * memory management
 */

/**
 * ‘mtxbasecsr_free()’ frees storage allocated for a matrix.
 */
void mtxbasecsr_free(
    struct mtxbasecsr * A);

/**
 * ‘mtxbasecsr_alloc_copy()’ allocates a copy of a matrix without
 * initialising the values.
 */
int mtxbasecsr_alloc_copy(
    struct mtxbasecsr * dst,
    const struct mtxbasecsr * src);

/**
 * ‘mtxbasecsr_init_copy()’ allocates a copy of a matrix and also
 * copies the values.
 */
int mtxbasecsr_init_copy(
    struct mtxbasecsr * dst,
    const struct mtxbasecsr * src);

/*
 * initialise matrices from entrywise data in coordinate format
 */

/**
 * ‘mtxbasecsr_alloc_entries()’ allocates a matrix from entrywise
 * data in coordinate format.
 *
 * If it is not ‘NULL’, then ‘perm’ must point to an array of length
 * ‘size’. Because the sparse matrix storage may internally reorder
 * the specified nonzero entries, this array is used to store the
 * permutation applied to the specified nonzero entries.
 */
int mtxbasecsr_alloc_entries(
    struct mtxbasecsr * A,
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
 * ‘mtxbasecsr_init_entries_real_single()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * real, single precision coefficients.
 */
int mtxbasecsr_init_entries_real_single(
    struct mtxbasecsr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx,
    const float * data);

/**
 * ‘mtxbasecsr_init_entries_real_double()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * real, double precision coefficients.
 */
int mtxbasecsr_init_entries_real_double(
    struct mtxbasecsr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx,
    const double * data);

/**
 * ‘mtxbasecsr_init_entries_complex_single()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * complex, single precision coefficients.
 */
int mtxbasecsr_init_entries_complex_single(
    struct mtxbasecsr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx,
    const float (* data)[2]);

/**
 * ‘mtxbasecsr_init_entries_complex_double()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * complex, double precision coefficients.
 */
int mtxbasecsr_init_entries_complex_double(
    struct mtxbasecsr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx,
    const double (* data)[2]);

/**
 * ‘mtxbasecsr_init_entries_integer_single()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * integer, single precision coefficients.
 */
int mtxbasecsr_init_entries_integer_single(
    struct mtxbasecsr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx,
    const int32_t * data);

/**
 * ‘mtxbasecsr_init_entries_integer_double()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * integer, double precision coefficients.
 */
int mtxbasecsr_init_entries_integer_double(
    struct mtxbasecsr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx,
    const int64_t * data);

/**
 * ‘mtxbasecsr_init_entries_pattern()’ allocates and initialises a
 * matrix from entrywise data in coordinate format with boolean
 * coefficients.
 */
int mtxbasecsr_init_entries_pattern(
    struct mtxbasecsr * A,
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
 * ‘mtxbasecsr_init_entries_strided_real_single()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * real, single precision coefficients.
 */
int mtxbasecsr_init_entries_strided_real_single(
    struct mtxbasecsr * A,
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
 * ‘mtxbasecsr_init_entries_strided_real_double()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * real, double precision coefficients.
 */
int mtxbasecsr_init_entries_strided_real_double(
    struct mtxbasecsr * A,
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
 * ‘mtxbasecsr_init_entries_strided_complex_single()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * complex, single precision coefficients.
 */
int mtxbasecsr_init_entries_strided_complex_single(
    struct mtxbasecsr * A,
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
 * ‘mtxbasecsr_init_entries_strided_complex_double()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * complex, double precision coefficients.
 */
int mtxbasecsr_init_entries_strided_complex_double(
    struct mtxbasecsr * A,
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
 * ‘mtxbasecsr_init_entries_strided_integer_single()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * integer, single precision coefficients.
 */
int mtxbasecsr_init_entries_strided_integer_single(
    struct mtxbasecsr * A,
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
 * ‘mtxbasecsr_init_entries_strided_integer_double()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * integer, double precision coefficients.
 */
int mtxbasecsr_init_entries_strided_integer_double(
    struct mtxbasecsr * A,
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
 * ‘mtxbasecsr_init_entries_strided_pattern()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * boolean coefficients.
 */
int mtxbasecsr_init_entries_strided_pattern(
    struct mtxbasecsr * A,
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
 * ‘mtxbasecsr_alloc_rows()’ allocates a matrix from row-wise data
 * in compressed row format.
 */
int mtxbasecsr_alloc_rows(
    struct mtxbasecsr * A,
    enum mtxfield field,
    enum mtxprecision precision,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx);

/**
 * ‘mtxbasecsr_init_rows_real_single()’ allocates and initialises a
 * matrix from row-wise data in compressed row format with real,
 * single precision coefficients.
 */
int mtxbasecsr_init_rows_real_single(
    struct mtxbasecsr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const float * data);

/**
 * ‘mtxbasecsr_init_rows_real_double()’ allocates and initialises a
 * matrix from row-wise data in compressed row format with real,
 * double precision coefficients.
 */
int mtxbasecsr_init_rows_real_double(
    struct mtxbasecsr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const double * data);

/**
 * ‘mtxbasecsr_init_rows_complex_single()’ allocates and
 * initialises a matrix from row-wise data in compressed row format
 * with complex, single precision coefficients.
 */
int mtxbasecsr_init_rows_complex_single(
    struct mtxbasecsr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const float (* data)[2]);

/**
 * ‘mtxbasecsr_init_rows_complex_double()’ allocates and
 * initialises a matrix from row-wise data in compressed row format
 * with complex, double precision coefficients.
 */
int mtxbasecsr_init_rows_complex_double(
    struct mtxbasecsr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const double (* data)[2]);

/**
 * ‘mtxbasecsr_init_rows_integer_single()’ allocates and
 * initialises a matrix from row-wise data in compressed row format
 * with integer, single precision coefficients.
 */
int mtxbasecsr_init_rows_integer_single(
    struct mtxbasecsr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const int32_t * data);

/**
 * ‘mtxbasecsr_init_rows_integer_double()’ allocates and
 * initialises a matrix from row-wise data in compressed row format
 * with integer, double precision coefficients.
 */
int mtxbasecsr_init_rows_integer_double(
    struct mtxbasecsr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const int64_t * data);

/**
 * ‘mtxbasecsr_init_rows_pattern()’ allocates and initialises a
 * matrix from row-wise data in compressed row format with boolean
 * coefficients.
 */
int mtxbasecsr_init_rows_pattern(
    struct mtxbasecsr * A,
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
 * ‘mtxbasecsr_alloc_columns()’ allocates a matrix from column-wise
 * data in compressed column format.
 */
int mtxbasecsr_alloc_columns(
    struct mtxbasecsr * A,
    enum mtxfield field,
    enum mtxprecision precision,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx);

/**
 * ‘mtxbasecsr_init_columns_real_single()’ allocates and
 * initialises a matrix from column-wise data in compressed column
 * format with real, single precision coefficients.
 */
int mtxbasecsr_init_columns_real_single(
    struct mtxbasecsr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx,
    const float * data);

/**
 * ‘mtxbasecsr_init_columns_real_double()’ allocates and
 * initialises a matrix from column-wise data in compressed column
 * format with real, double precision coefficients.
 */
int mtxbasecsr_init_columns_real_double(
    struct mtxbasecsr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx,
    const double * data);

/**
 * ‘mtxbasecsr_init_columns_complex_single()’ allocates and
 * initialises a matrix from column-wise data in compressed column
 * format with complex, single precision coefficients.
 */
int mtxbasecsr_init_columns_complex_single(
    struct mtxbasecsr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx,
    const float (* data)[2]);

/**
 * ‘mtxbasecsr_init_columns_complex_double()’ allocates and
 * initialises a matrix from column-wise data in compressed column
 * format with complex, double precision coefficients.
 */
int mtxbasecsr_init_columns_complex_double(
    struct mtxbasecsr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx,
    const double (* data)[2]);

/**
 * ‘mtxbasecsr_init_columns_integer_single()’ allocates and
 * initialises a matrix from column-wise data in compressed column
 * format with integer, single precision coefficients.
 */
int mtxbasecsr_init_columns_integer_single(
    struct mtxbasecsr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx,
    const int32_t * data);

/**
 * ‘mtxbasecsr_init_columns_integer_double()’ allocates and
 * initialises a matrix from column-wise data in compressed column
 * format with integer, double precision coefficients.
 */
int mtxbasecsr_init_columns_integer_double(
    struct mtxbasecsr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx,
    const int64_t * data);

/**
 * ‘mtxbasecsr_init_columns_pattern()’ allocates and initialises a
 * matrix from column-wise data in compressed column format with
 * boolean coefficients.
 */
int mtxbasecsr_init_columns_pattern(
    struct mtxbasecsr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx);

/*
 * initialise matrices from a list of dense cliques
 */

/**
 * ‘mtxbasecsr_alloc_cliques()’ allocates a matrix from a list of
 * dense cliques.
 */
int mtxbasecsr_alloc_cliques(
    struct mtxbasecsr * A,
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
 * ‘mtxbasecsr_init_cliques_real_single()’ allocates and
 * initialises a matrix from a list of dense cliques with real, single
 * precision coefficients.
 */
int mtxbasecsr_init_cliques_real_single(
    struct mtxbasecsr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_cliques,
    const int64_t * cliqueptr,
    const int * rowidx,
    const int * colidx,
    const float * data);

/**
 * ‘mtxbasecsr_init_cliques_real_double()’ allocates and
 * initialises a matrix from a list of dense cliques with real, double
 * precision coefficients.
 */
int mtxbasecsr_init_cliques_real_double(
    struct mtxbasecsr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_cliques,
    const int64_t * cliqueptr,
    const int * rowidx,
    const int * colidx,
    const double * data);

/**
 * ‘mtxbasecsr_init_cliques_complex_single()’ allocates and
 * initialises a matrix from a list of dense cliques with complex,
 * single precision coefficients.
 */
int mtxbasecsr_init_cliques_complex_single(
    struct mtxbasecsr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_cliques,
    const int64_t * cliqueptr,
    const int * rowidx,
    const int * colidx,
    const float (* data)[2]);

/**
 * ‘mtxbasecsr_init_cliques_complex_double()’ allocates and
 * initialises a matrix from a list of dense cliques with complex,
 * double precision coefficients.
 */
int mtxbasecsr_init_cliques_complex_double(
    struct mtxbasecsr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_cliques,
    const int64_t * cliqueptr,
    const int * rowidx,
    const int * colidx,
    const double (* data)[2]);

/**
 * ‘mtxbasecsr_init_cliques_integer_single()’ allocates and
 * initialises a matrix from a list of dense cliques with integer,
 * single precision coefficients.
 */
int mtxbasecsr_init_cliques_integer_single(
    struct mtxbasecsr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_cliques,
    const int64_t * cliqueptr,
    const int * rowidx,
    const int * colidx,
    const int32_t * data);

/**
 * ‘mtxbasecsr_init_cliques_integer_double()’ allocates and
 * initialises a matrix from a list of dense cliques with integer,
 * double precision coefficients.
 */
int mtxbasecsr_init_cliques_integer_double(
    struct mtxbasecsr * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_cliques,
    const int64_t * cliqueptr,
    const int * rowidx,
    const int * colidx,
    const int64_t * data);

/**
 * ‘mtxbasecsr_init_cliques_pattern()’ allocates and initialises a
 * matrix from a list of dense cliques with boolean coefficients.
 */
int mtxbasecsr_init_cliques_pattern(
    struct mtxbasecsr * A,
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
 * ‘mtxbasecsr_setzero()’ sets every value of a matrix to zero.
 */
int mtxbasecsr_setzero(
    struct mtxbasecsr * A);

/**
 * ‘mtxbasecsr_set_real_single()’ sets values of a matrix based on
 * an array of single precision floating point numbers.
 */
int mtxbasecsr_set_real_single(
    struct mtxbasecsr * A,
    int64_t size,
    int stride,
    const float * a);

/**
 * ‘mtxbasecsr_set_real_double()’ sets values of a matrix based on
 * an array of double precision floating point numbers.
 */
int mtxbasecsr_set_real_double(
    struct mtxbasecsr * A,
    int64_t size,
    int stride,
    const double * a);

/**
 * ‘mtxbasecsr_set_complex_single()’ sets values of a matrix based
 * on an array of single precision floating point complex numbers.
 */
int mtxbasecsr_set_complex_single(
    struct mtxbasecsr * A,
    int64_t size,
    int stride,
    const float (*a)[2]);

/**
 * ‘mtxbasecsr_set_complex_double()’ sets values of a matrix based
 * on an array of double precision floating point complex numbers.
 */
int mtxbasecsr_set_complex_double(
    struct mtxbasecsr * A,
    int64_t size,
    int stride,
    const double (*a)[2]);

/**
 * ‘mtxbasecsr_set_integer_single()’ sets values of a matrix based
 * on an array of integers.
 */
int mtxbasecsr_set_integer_single(
    struct mtxbasecsr * A,
    int64_t size,
    int stride,
    const int32_t * a);

/**
 * ‘mtxbasecsr_set_integer_double()’ sets values of a matrix based
 * on an array of integers.
 */
int mtxbasecsr_set_integer_double(
    struct mtxbasecsr * A,
    int64_t size,
    int stride,
    const int64_t * a);

/*
 * row and column vectors
 */

/**
 * ‘mtxbasecsr_alloc_row_vector()’ allocates a row vector for a
 * given matrix, where a row vector is a vector whose length equal to
 * a single row of the matrix.
 */
int mtxbasecsr_alloc_row_vector(
    const struct mtxbasecsr * A,
    struct mtxvector * x,
    enum mtxvectortype vectortype);

/**
 * ‘mtxbasecsr_alloc_column_vector()’ allocates a column vector for
 * a given matrix, where a column vector is a vector whose length
 * equal to a single column of the matrix.
 */
int mtxbasecsr_alloc_column_vector(
    const struct mtxbasecsr * A,
    struct mtxvector * y,
    enum mtxvectortype vectortype);

/*
 * convert to and from Matrix Market format
 */

/**
 * ‘mtxbasecsr_from_mtxfile()’ converts a matrix from Matrix Market
 * format.
 */
int mtxbasecsr_from_mtxfile(
    struct mtxbasecsr * A,
    const struct mtxfile * mtxfile);

/**
 * ‘mtxbasecsr_to_mtxfile()’ converts a matrix to Matrix Market
 *  format.
 */
int mtxbasecsr_to_mtxfile(
    struct mtxfile * mtxfile,
    const struct mtxbasecsr * A,
    int64_t num_rows,
    const int64_t * rowidx,
    int64_t num_columns,
    const int64_t * colidx,
    enum mtxfileformat mtxfmt);

/*
 * partitioning
 */

/**
 * ‘mtxbasecsr_partition_rowwise()’ partitions the entries of a matrix
 * rowwise.
 *
 * See ‘partition_int()’ for an explanation of the meaning of the
 * arguments ‘parttype’, ‘num_parts’, ‘partsizes’, ‘blksize’ and
 * ‘parts’.
 *
 * The length of the array ‘dstnzpart’ must be at least equal to the
 * number of (nonzero) matrix entries (which can be obtained by
 * calling ‘mtxmatrix_size()’). If successful, ‘dstnzpart’ is used to
 * store the part numbers assigned to the matrix nonzeros.
 *
 * If ‘dstrowpart’ is not ‘NULL’, then it must be an array of length
 * at least equal to the number of matrix rows, which is used to store
 * the part numbers assigned to the rows.
 *
 * If ‘dstnzpartsizes’ is not ‘NULL’, then it must be an array of
 * length ‘num_parts’, which is used to store the number of nonzeros
 * assigned to each part. Similarly, if ‘dstrowpartsizes’ is not
 * ‘NULL’, then it must be an array of length ‘num_parts’, and it is
 * used to store the number of rows assigned to each part
 */
int mtxbasecsr_partition_rowwise(
    const struct mtxbasecsr * A,
    enum mtxpartitioning parttype,
    int num_parts,
    const int * partsizes,
    int blksize,
    const int * parts,
    int * dstnzpart,
    int64_t * dstnzpartsizes,
    int * dstrowpart,
    int64_t * dstrowpartsizes);

/**
 * ‘mtxbasecsr_partition_columnwise()’ partitions the entries of a
 * matrix columnwise.
 *
 * See ‘partition_int()’ for an explanation of the meaning of the
 * arguments ‘parttype’, ‘num_parts’, ‘partsizes’, ‘blksize’ and
 * ‘parts’.
 *
 * The length of the array ‘dstnzpart’ must be at least equal to the
 * number of (nonzero) matrix entries (which can be obtained by
 * calling ‘mtxmatrix_size()’). If successful, ‘dstnzpart’ is used to
 * store the part numbers assigned to the matrix nonzeros.
 *
 * If ‘dstcolpart’ is not ‘NULL’, then it must be an array of length
 * at least equal to the number of matrix columns, which is used to
 * store the part numbers assigned to the columns.
 *
 * If ‘dstnzpartsizes’ is not ‘NULL’, then it must be an array of
 * length ‘num_parts’, which is used to store the number of nonzeros
 * assigned to each part. Similarly, if ‘dstcolpartsizes’ is not
 * ‘NULL’, then it must be an array of length ‘num_parts’, and it is
 * used to store the number of columns assigned to each part
 */
int mtxbasecsr_partition_columnwise(
    const struct mtxbasecsr * A,
    enum mtxpartitioning parttype,
    int num_parts,
    const int * partsizes,
    int blksize,
    const int * parts,
    int * dstnzpart,
    int64_t * dstnzpartsizes,
    int * dstcolpart,
    int64_t * dstcolpartsizes);

/**
 * ‘mtxbasecsr_partition_2d()’ partitions the entries of a matrix in a
 * 2D manner.
 *
 * See ‘partition_int()’ for an explanation of the meaning of the
 * arguments ‘rowparttype’, ‘num_row_parts’, ‘rowpartsizes’,
 * ‘rowblksize’, ‘rowparts’, and so on.
 *
 * The length of the array ‘dstnzpart’ must be at least equal to the
 * number of (nonzero) matrix entries (which can be obtained by
 * calling ‘mtxmatrix_size()’). If successful, ‘dstnzpart’ is used to
 * store the part numbers assigned to the matrix nonzeros.
 *
 * The length of the array ‘dstrowpart’ must be at least equal to the
 * number of matrix rows, and it is used to store the part numbers
 * assigned to the rows. Similarly, the length of ‘dstcolpart’ must be
 * at least equal to the number of matrix columns, and it is used to
 * store the part numbers assigned to the columns.
 *
 * If ‘dstrowpartsizes’ or ‘dstcolpartsizes’ are not ‘NULL’, then they
 * must point to arrays of length ‘num_row_parts’ and ‘num_col_parts’,
 * respectively, which are used to store the number of rows and
 * columns assigned to each part.
 */
int mtxbasecsr_partition_2d(
    const struct mtxbasecsr * A,
    enum mtxpartitioning rowparttype,
    int num_row_parts,
    const int * rowpartsizes,
    int rowblksize,
    const int * rowparts,
    enum mtxpartitioning colparttype,
    int num_col_parts,
    const int * colpartsizes,
    int colblksize,
    const int * colparts,
    int * dstnzpart,
    int64_t * dstnzpartsizes,
    int * dstrowpart,
    int64_t * dstrowpartsizes,
    int * dstcolpart,
    int64_t * dstcolpartsizes);

/**
 * ‘mtxbasecsr_split()’ splits a matrix into multiple matrices
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
 * of type ‘struct mtxbasecsr’. If successful, then ‘dsts[p]’
 * points to a matrix consisting of elements from ‘src’ that belong to
 * the ‘p’th part, as designated by the ‘parts’ array.
 *
 * The caller is responsible for calling ‘mtxbasecsr_free()’ to
 * free storage allocated for each matrix in the ‘dsts’ array.
 */
int mtxbasecsr_split(
    int num_parts,
    struct mtxbasecsr ** dsts,
    const struct mtxbasecsr * src,
    int64_t size,
    int * parts);

/*
 * Level 1 BLAS operations
 */

/**
 * ‘mtxbasecsr_swap()’ swaps values of two matrices, simultaneously
 * performing ‘y <- x’ and ‘x <- y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxbasecsr_swap(
    struct mtxbasecsr * x,
    struct mtxbasecsr * y);

/**
 * ‘mtxbasecsr_copy()’ copies values of a matrix, ‘y = x’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxbasecsr_copy(
    struct mtxbasecsr * y,
    const struct mtxbasecsr * x);

/**
 * ‘mtxbasecsr_sscal()’ scales a matrix by a single precision
 * floating point scalar, ‘x = a*x’.
 */
int mtxbasecsr_sscal(
    float a,
    struct mtxbasecsr * x,
    int64_t * num_flops);

/**
 * ‘mtxbasecsr_dscal()’ scales a matrix by a double precision
 * floating point scalar, ‘x = a*x’.
 */
int mtxbasecsr_dscal(
    double a,
    struct mtxbasecsr * x,
    int64_t * num_flops);

/**
 * ‘mtxbasecsr_cscal()’ scales a matrix by a complex, single
 * precision floating point scalar, ‘x = (a+b*i)*x’.
 */
int mtxbasecsr_cscal(
    float a[2],
    struct mtxbasecsr * x,
    int64_t * num_flops);

/**
 * ‘mtxbasecsr_zscal()’ scales a matrix by a complex, double
 * precision floating point scalar, ‘x = (a+b*i)*x’.
 */
int mtxbasecsr_zscal(
    double a[2],
    struct mtxbasecsr * x,
    int64_t * num_flops);

/**
 * ‘mtxbasecsr_saxpy()’ adds a matrix to another one multiplied by
 * a single precision floating point value, ‘y = a*x + y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxbasecsr_saxpy(
    float a,
    const struct mtxbasecsr * x,
    struct mtxbasecsr * y,
    int64_t * num_flops);

/**
 * ‘mtxbasecsr_daxpy()’ adds a matrix to another one multiplied by
 * a double precision floating point value, ‘y = a*x + y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxbasecsr_daxpy(
    double a,
    const struct mtxbasecsr * x,
    struct mtxbasecsr * y,
    int64_t * num_flops);

/**
 * ‘mtxbasecsr_saypx()’ multiplies a matrix by a single precision
 * floating point scalar and adds another matrix, ‘y = a*y + x’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxbasecsr_saypx(
    float a,
    struct mtxbasecsr * y,
    const struct mtxbasecsr * x,
    int64_t * num_flops);

/**
 * ‘mtxbasecsr_daypx()’ multiplies a matrix by a double precision
 * floating point scalar and adds another matrix, ‘y = a*y + x’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxbasecsr_daypx(
    double a,
    struct mtxbasecsr * y,
    const struct mtxbasecsr * x,
    int64_t * num_flops);

/**
 * ‘mtxbasecsr_sdot()’ computes the Frobenius inner product of two
 * matrices in single precision floating point.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxbasecsr_sdot(
    const struct mtxbasecsr * x,
    const struct mtxbasecsr * y,
    float * dot,
    int64_t * num_flops);

/**
 * ‘mtxbasecsr_ddot()’ computes the Frobenius inner product of two
 * matrices in double precision floating point.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxbasecsr_ddot(
    const struct mtxbasecsr * x,
    const struct mtxbasecsr * y,
    double * dot,
    int64_t * num_flops);

/**
 * ‘mtxbasecsr_cdotu()’ computes the product of the transpose of a
 * complex row matrix with another complex row matrix in single
 * precision floating point, ‘dot := x^T*y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxbasecsr_cdotu(
    const struct mtxbasecsr * x,
    const struct mtxbasecsr * y,
    float (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxbasecsr_zdotu()’ computes the product of the transpose of a
 * complex row matrix with another complex row matrix in double
 * precision floating point, ‘dot := x^T*y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxbasecsr_zdotu(
    const struct mtxbasecsr * x,
    const struct mtxbasecsr * y,
    double (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxbasecsr_cdotc()’ computes the Frobenius inner product of two
 * complex matrices in single precision floating point, ‘dot :=
 * x^H*y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxbasecsr_cdotc(
    const struct mtxbasecsr * x,
    const struct mtxbasecsr * y,
    float (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxbasecsr_zdotc()’ computes the Frobenius inner product of two
 * complex matrices in double precision floating point, ‘dot :=
 * x^H*y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxbasecsr_zdotc(
    const struct mtxbasecsr * x,
    const struct mtxbasecsr * y,
    double (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxbasecsr_snrm2()’ computes the Frobenius norm of a matrix in
 * single precision floating point.
 */
int mtxbasecsr_snrm2(
    const struct mtxbasecsr * x,
    float * nrm2,
    int64_t * num_flops);

/**
 * ‘mtxbasecsr_dnrm2()’ computes the Frobenius norm of a matrix in
 * double precision floating point.
 */
int mtxbasecsr_dnrm2(
    const struct mtxbasecsr * x,
    double * nrm2,
    int64_t * num_flops);

/**
 * ‘mtxbasecsr_sasum()’ computes the sum of absolute values
 * (1-norm) of a matrix in single precision floating point.  If the
 * matrix is complex-valued, then the sum of the absolute values of
 * the real and imaginary parts is computed.
 */
int mtxbasecsr_sasum(
    const struct mtxbasecsr * x,
    float * asum,
    int64_t * num_flops);

/**
 * ‘mtxbasecsr_dasum()’ computes the sum of absolute values
 * (1-norm) of a matrix in double precision floating point.  If the
 * matrix is complex-valued, then the sum of the absolute values of
 * the real and imaginary parts is computed.
 */
int mtxbasecsr_dasum(
    const struct mtxbasecsr * x,
    double * asum,
    int64_t * num_flops);

/**
 * ‘mtxbasecsr_iamax()’ finds the index of the first element having
 * the maximum absolute value.  If the matrix is complex-valued, then
 * the index points to the first element having the maximum sum of the
 * absolute values of the real and imaginary parts.
 */
int mtxbasecsr_iamax(
    const struct mtxbasecsr * x,
    int * iamax);

/*
 * Level 2 BLAS operations (matrix-vector)
 */

/**
 * ‘mtxbasecsr_sgemv()’ multiplies a matrix ‘A’ or its transpose
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
int mtxbasecsr_sgemv(
    enum mtxtransposition trans,
    float alpha,
    const struct mtxbasecsr * A,
    const struct mtxvector * x,
    float beta,
    struct mtxvector * y,
    int64_t * num_flops);

/**
 * ‘mtxbasecsr_dgemv()’ multiplies a matrix ‘A’ or its transpose
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
int mtxbasecsr_dgemv(
    enum mtxtransposition trans,
    double alpha,
    const struct mtxbasecsr * A,
    const struct mtxvector * x,
    double beta,
    struct mtxvector * y,
    int64_t * num_flops);

/**
 * ‘mtxbasecsr_cgemv()’ multiplies a complex-valued matrix ‘A’, its
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
 */
int mtxbasecsr_cgemv(
    enum mtxtransposition trans,
    float alpha[2],
    const struct mtxbasecsr * A,
    const struct mtxvector * x,
    float beta[2],
    struct mtxvector * y,
    int64_t * num_flops);

/**
 * ‘mtxbasecsr_zgemv()’ multiplies a complex-valued matrix ‘A’, its
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
 */
int mtxbasecsr_zgemv(
    enum mtxtransposition trans,
    double alpha[2],
    const struct mtxbasecsr * A,
    const struct mtxvector * x,
    double beta[2],
    struct mtxvector * y,
    int64_t * num_flops);

#endif
