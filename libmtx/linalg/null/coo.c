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
 * Data structures for matrices in coordinate format, where matrix
 * operations do nothing. While this produces incorrect results, it is
 * sometimes useful for the purpose of eliminating matrix operations
 * while debugging or carrying out performance measurements.
 */

#include <libmtx/libmtx-config.h>

#include <libmtx/error.h>
#include <libmtx/linalg/null/coo.h>
#include <libmtx/linalg/base/coo.h>
#include <libmtx/mtxfile/mtxfile.h>
#include <libmtx/linalg/field.h>
#include <libmtx/linalg/precision.h>
#include <libmtx/linalg/local/vector.h>
#include <libmtx/util/partition.h>

#include <stddef.h>
#include <stdlib.h>

/*
 * matrix properties
 */

/**
 * ‘mtxnullcoo_field()’ gets the field of a matrix.
 */
enum mtxfield mtxnullcoo_field(const struct mtxnullcoo * A)
{
    return mtxbasecoo_field(&A->base);
}

/**
 * ‘mtxnullcoo_precision()’ gets the precision of a matrix.
 */
enum mtxprecision mtxnullcoo_precision(const struct mtxnullcoo * A)
{
    return mtxbasecoo_precision(&A->base);
}

/**
 * ‘mtxnullcoo_symmetry()’ gets the symmetry of a matrix.
 */
enum mtxsymmetry mtxnullcoo_symmetry(const struct mtxnullcoo * A)
{
    return mtxbasecoo_symmetry(&A->base);
}

/**
 * ‘mtxnullcoo_num_rows()’ gets the number of matrix rows.
 */
int mtxnullcoo_num_rows(const struct mtxnullcoo * A)
{
    return mtxbasecoo_num_rows(&A->base);
}

/**
 * ‘mtxnullcoo_num_columns()’ gets the number of matrix columns.
 */
int mtxnullcoo_num_columns(const struct mtxnullcoo * A)
{
    return mtxbasecoo_num_columns(&A->base);
}

/**
 * ‘mtxnullcoo_num_nonzeros()’ gets the number of the number of
 *  nonzero matrix entries, including those represented implicitly due
 *  to symmetry.
 */
int64_t mtxnullcoo_num_nonzeros(const struct mtxnullcoo * A)
{
    return mtxbasecoo_num_nonzeros(&A->base);
}

/**
 * ‘mtxnullcoo_size()’ gets the number of explicitly stored
 * nonzeros of a matrix.
 */
int64_t mtxnullcoo_size(const struct mtxnullcoo * A)
{
    return mtxbasecoo_size(&A->base);
}

/**
 * ‘mtxnullcoo_rowcolidx()’ gets the row and column indices of
 * the explicitly stored matrix nonzeros.
 *
 * The arguments ‘rowidx’ and ‘colidx’ may be ‘NULL’ or must point to
 * an arrays of length ‘size’.
 */
int mtxnullcoo_rowcolidx(
    const struct mtxnullcoo * A,
    int64_t size,
    int * rowidx,
    int * colidx)
{
    return mtxbasecoo_rowcolidx(&A->base, size, rowidx, colidx);
}

/*
 * memory management
 */

/**
 * ‘mtxnullcoo_free()’ frees storage allocated for a matrix.
 */
void mtxnullcoo_free(
    struct mtxnullcoo * A)
{
    mtxbasecoo_free(&A->base);
}

/**
 * ‘mtxnullcoo_alloc_copy()’ allocates a copy of a matrix without
 * initialising the values.
 */
int mtxnullcoo_alloc_copy(
    struct mtxnullcoo * dst,
    const struct mtxnullcoo * src)
{
    return mtxbasecoo_alloc_copy(&dst->base, &src->base);
}

/**
 * ‘mtxnullcoo_init_copy()’ allocates a copy of a matrix and also
 * copies the values.
 */
int mtxnullcoo_init_copy(
    struct mtxnullcoo * dst,
    const struct mtxnullcoo * src)
{
    return mtxbasecoo_init_copy(&dst->base, &src->base);
}

/*
 * initialise matrices from entrywise data in coordinate format
 */

/**
 * ‘mtxnullcoo_alloc_entries()’ allocates a matrix from entrywise
 * data in coordinate format.
 */
int mtxnullcoo_alloc_entries(
    struct mtxnullcoo * A,
    enum mtxfield field,
    enum mtxprecision precision,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    int idxstride,
    int idxbase,
    const int * rowidx,
    const int * colidx)
{
    return mtxbasecoo_alloc_entries(
        &A->base, field, precision, symmetry, num_rows, num_columns, size,
        idxstride, idxbase, rowidx, colidx);
}

/**
 * ‘mtxnullcoo_init_entries_real_single()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * real, single precision coefficients.
 */
int mtxnullcoo_init_entries_real_single(
    struct mtxnullcoo * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx,
    const float * data)
{
    return mtxbasecoo_init_entries_real_single(
        &A->base, symmetry, num_rows, num_columns, size, rowidx, colidx, data);
}

/**
 * ‘mtxnullcoo_init_entries_real_double()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * real, double precision coefficients.
 */
int mtxnullcoo_init_entries_real_double(
    struct mtxnullcoo * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx,
    const double * data)
{
    return mtxbasecoo_init_entries_real_double(
        &A->base, symmetry, num_rows, num_columns, size, rowidx, colidx, data);
}

/**
 * ‘mtxnullcoo_init_entries_complex_single()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * complex, single precision coefficients.
 */
int mtxnullcoo_init_entries_complex_single(
    struct mtxnullcoo * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx,
    const float (* data)[2])
{
    return mtxbasecoo_init_entries_complex_single(
        &A->base, symmetry, num_rows, num_columns, size, rowidx, colidx, data);
}

/**
 * ‘mtxnullcoo_init_entries_complex_double()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * complex, double precision coefficients.
 */
int mtxnullcoo_init_entries_complex_double(
    struct mtxnullcoo * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx,
    const double (* data)[2])
{
    return mtxbasecoo_init_entries_complex_double(
        &A->base, symmetry, num_rows, num_columns, size, rowidx, colidx, data);
}

/**
 * ‘mtxnullcoo_init_entries_integer_single()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * integer, single precision coefficients.
 */
int mtxnullcoo_init_entries_integer_single(
    struct mtxnullcoo * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx,
    const int32_t * data)
{
    return mtxbasecoo_init_entries_integer_single(
        &A->base, symmetry, num_rows, num_columns, size, rowidx, colidx, data);
}

/**
 * ‘mtxnullcoo_init_entries_integer_double()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * integer, double precision coefficients.
 */
int mtxnullcoo_init_entries_integer_double(
    struct mtxnullcoo * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx,
    const int64_t * data)
{
    return mtxbasecoo_init_entries_integer_double(
        &A->base, symmetry, num_rows, num_columns, size, rowidx, colidx, data);
}

/**
 * ‘mtxnullcoo_init_entries_pattern()’ allocates and initialises a
 * matrix from entrywise data in coordinate format with boolean
 * coefficients.
 */
int mtxnullcoo_init_entries_pattern(
    struct mtxnullcoo * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx)
{
    return mtxbasecoo_init_entries_pattern(
        &A->base, symmetry, num_rows, num_columns, size, rowidx, colidx);
}

/*
 * initialise matrices from entrywise data in coordinate format with
 * specified strides
 */

/**
 * ‘mtxnullcoo_init_entries_strided_real_single()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * real, single precision coefficients.
 */
int mtxnullcoo_init_entries_strided_real_single(
    struct mtxnullcoo * A,
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
 * ‘mtxnullcoo_init_entries_strided_real_double()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * real, double precision coefficients.
 */
int mtxnullcoo_init_entries_strided_real_double(
    struct mtxnullcoo * A,
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
 * ‘mtxnullcoo_init_entries_strided_complex_single()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * complex, single precision coefficients.
 */
int mtxnullcoo_init_entries_strided_complex_single(
    struct mtxnullcoo * A,
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
 * ‘mtxnullcoo_init_entries_strided_complex_double()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * complex, double precision coefficients.
 */
int mtxnullcoo_init_entries_strided_complex_double(
    struct mtxnullcoo * A,
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
 * ‘mtxnullcoo_init_entries_strided_integer_single()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * integer, single precision coefficients.
 */
int mtxnullcoo_init_entries_strided_integer_single(
    struct mtxnullcoo * A,
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
 * ‘mtxnullcoo_init_entries_strided_integer_double()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * integer, double precision coefficients.
 */
int mtxnullcoo_init_entries_strided_integer_double(
    struct mtxnullcoo * A,
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
 * ‘mtxnullcoo_init_entries_strided_pattern()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * boolean coefficients.
 */
int mtxnullcoo_init_entries_strided_pattern(
    struct mtxnullcoo * A,
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
 * ‘mtxnullcoo_alloc_rows()’ allocates a matrix from row-wise data
 * in compressed row format.
 */
int mtxnullcoo_alloc_rows(
    struct mtxnullcoo * A,
    enum mtxfield field,
    enum mtxprecision precision,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx)
{
    return mtxbasecoo_alloc_rows(
        &A->base, field, precision, symmetry,
        num_rows, num_columns, rowptr, colidx);
}

/**
 * ‘mtxnullcoo_init_rows_real_single()’ allocates and initialises a
 * matrix from row-wise data in compressed row format with real,
 * single precision coefficients.
 */
int mtxnullcoo_init_rows_real_single(
    struct mtxnullcoo * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const float * data)
{
    return mtxbasecoo_init_rows_real_single(
        &A->base, symmetry, num_rows, num_columns, rowptr, colidx, data);
}

/**
 * ‘mtxnullcoo_init_rows_real_double()’ allocates and initialises a
 * matrix from row-wise data in compressed row format with real,
 * double precision coefficients.
 */
int mtxnullcoo_init_rows_real_double(
    struct mtxnullcoo * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const double * data)
{
    return mtxbasecoo_init_rows_real_double(
        &A->base, symmetry, num_rows, num_columns, rowptr, colidx, data);
}

/**
 * ‘mtxnullcoo_init_rows_complex_single()’ allocates and
 * initialises a matrix from row-wise data in compressed row format
 * with complex, single precision coefficients.
 */
int mtxnullcoo_init_rows_complex_single(
    struct mtxnullcoo * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const float (* data)[2])
{
    return mtxbasecoo_init_rows_complex_single(
        &A->base, symmetry, num_rows, num_columns, rowptr, colidx, data);
}

/**
 * ‘mtxnullcoo_init_rows_complex_double()’ allocates and
 * initialises a matrix from row-wise data in compressed row format
 * with complex, double precision coefficients.
 */
int mtxnullcoo_init_rows_complex_double(
    struct mtxnullcoo * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const double (* data)[2])
{
    return mtxbasecoo_init_rows_complex_double(
        &A->base, symmetry, num_rows, num_columns, rowptr, colidx, data);
}

/**
 * ‘mtxnullcoo_init_rows_integer_single()’ allocates and
 * initialises a matrix from row-wise data in compressed row format
 * with integer, single precision coefficients.
 */
int mtxnullcoo_init_rows_integer_single(
    struct mtxnullcoo * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const int32_t * data)
{
    return mtxbasecoo_init_rows_integer_single(
        &A->base, symmetry, num_rows, num_columns, rowptr, colidx, data);
}

/**
 * ‘mtxnullcoo_init_rows_integer_double()’ allocates and
 * initialises a matrix from row-wise data in compressed row format
 * with integer, double precision coefficients.
 */
int mtxnullcoo_init_rows_integer_double(
    struct mtxnullcoo * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const int64_t * data)
{
    return mtxbasecoo_init_rows_integer_double(
        &A->base, symmetry, num_rows, num_columns, rowptr, colidx, data);
}

/**
 * ‘mtxnullcoo_init_rows_pattern()’ allocates and initialises a
 * matrix from row-wise data in compressed row format with boolean
 * coefficients.
 */
int mtxnullcoo_init_rows_pattern(
    struct mtxnullcoo * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx)
{
    return mtxbasecoo_init_rows_pattern(
        &A->base, symmetry, num_rows, num_columns, rowptr, colidx);
}

/*
 * initialise matrices from column-wise data in compressed column
 * format
 */

/**
 * ‘mtxnullcoo_alloc_columns()’ allocates a matrix from column-wise
 * data in compressed column format.
 */
int mtxnullcoo_alloc_columns(
    struct mtxnullcoo * A,
    enum mtxfield field,
    enum mtxprecision precision,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx);

/**
 * ‘mtxnullcoo_init_columns_real_single()’ allocates and
 * initialises a matrix from column-wise data in compressed column
 * format with real, single precision coefficients.
 */
int mtxnullcoo_init_columns_real_single(
    struct mtxnullcoo * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx,
    const float * data);

/**
 * ‘mtxnullcoo_init_columns_real_double()’ allocates and
 * initialises a matrix from column-wise data in compressed column
 * format with real, double precision coefficients.
 */
int mtxnullcoo_init_columns_real_double(
    struct mtxnullcoo * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx,
    const double * data);

/**
 * ‘mtxnullcoo_init_columns_complex_single()’ allocates and
 * initialises a matrix from column-wise data in compressed column
 * format with complex, single precision coefficients.
 */
int mtxnullcoo_init_columns_complex_single(
    struct mtxnullcoo * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx,
    const float (* data)[2]);

/**
 * ‘mtxnullcoo_init_columns_complex_double()’ allocates and
 * initialises a matrix from column-wise data in compressed column
 * format with complex, double precision coefficients.
 */
int mtxnullcoo_init_columns_complex_double(
    struct mtxnullcoo * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx,
    const double (* data)[2]);

/**
 * ‘mtxnullcoo_init_columns_integer_single()’ allocates and
 * initialises a matrix from column-wise data in compressed column
 * format with integer, single precision coefficients.
 */
int mtxnullcoo_init_columns_integer_single(
    struct mtxnullcoo * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx,
    const int32_t * data);

/**
 * ‘mtxnullcoo_init_columns_integer_double()’ allocates and
 * initialises a matrix from column-wise data in compressed column
 * format with integer, double precision coefficients.
 */
int mtxnullcoo_init_columns_integer_double(
    struct mtxnullcoo * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx,
    const int64_t * data);

/**
 * ‘mtxnullcoo_init_columns_pattern()’ allocates and initialises a
 * matrix from column-wise data in compressed column format with
 * boolean coefficients.
 */
int mtxnullcoo_init_columns_pattern(
    struct mtxnullcoo * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx);

/*
 * initialise matrices from a list of dense cliques
 */

/**
 * ‘mtxnullcoo_alloc_cliques()’ allocates a matrix from a list of
 * cliques.
 */
int mtxnullcoo_alloc_cliques(
    struct mtxnullcoo * A,
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
 * ‘mtxnullcoo_init_cliques_real_single()’ allocates and
 * initialises a matrix from a list of cliques with real, single
 * precision coefficients.
 */
int mtxnullcoo_init_cliques_real_single(
    struct mtxnullcoo * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_cliques,
    const int64_t * cliqueptr,
    const int * rowidx,
    const int * colidx,
    const float * data);

/**
 * ‘mtxnullcoo_init_cliques_real_double()’ allocates and
 * initialises a matrix from a list of cliques with real, double
 * precision coefficients.
 */
int mtxnullcoo_init_cliques_real_double(
    struct mtxnullcoo * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_cliques,
    const int64_t * cliqueptr,
    const int * rowidx,
    const int * colidx,
    const double * data);

/**
 * ‘mtxnullcoo_init_cliques_complex_single()’ allocates and
 * initialises a matrix from a list of cliques with complex, single
 * precision coefficients.
 */
int mtxnullcoo_init_cliques_complex_single(
    struct mtxnullcoo * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_cliques,
    const int64_t * cliqueptr,
    const int * rowidx,
    const int * colidx,
    const float (* data)[2]);

/**
 * ‘mtxnullcoo_init_cliques_complex_double()’ allocates and
 * initialises a matrix from a list of cliques with complex, double
 * precision coefficients.
 */
int mtxnullcoo_init_cliques_complex_double(
    struct mtxnullcoo * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_cliques,
    const int64_t * cliqueptr,
    const int * rowidx,
    const int * colidx,
    const double (* data)[2]);

/**
 * ‘mtxnullcoo_init_cliques_integer_single()’ allocates and
 * initialises a matrix from a list of cliques with integer, single
 * precision coefficients.
 */
int mtxnullcoo_init_cliques_integer_single(
    struct mtxnullcoo * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_cliques,
    const int64_t * cliqueptr,
    const int * rowidx,
    const int * colidx,
    const int32_t * data);

/**
 * ‘mtxnullcoo_init_cliques_integer_double()’ allocates and
 * initialises a matrix from a list of cliques with integer, double
 * precision coefficients.
 */
int mtxnullcoo_init_cliques_integer_double(
    struct mtxnullcoo * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_cliques,
    const int64_t * cliqueptr,
    const int * rowidx,
    const int * colidx,
    const int64_t * data);

/**
 * ‘mtxnullcoo_init_cliques_pattern()’ allocates and initialises a
 * matrix from a list of cliques with boolean coefficients.
 */
int mtxnullcoo_init_cliques_pattern(
    struct mtxnullcoo * A,
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
 * ‘mtxnullcoo_setzero()’ sets every value of a matrix to zero.
 */
int mtxnullcoo_setzero(
    struct mtxnullcoo * A)
{
    return mtxbasecoo_setzero(&A->base);
}

/**
 * ‘mtxnullcoo_set_real_single()’ sets values of a matrix based on
 * an array of single precision floating point numbers.
 */
int mtxnullcoo_set_real_single(
    struct mtxnullcoo * A,
    int64_t size,
    int stride,
    const float * a)
{
    return mtxbasecoo_set_real_single(&A->base, size, stride, a);
}

/**
 * ‘mtxnullcoo_set_real_double()’ sets values of a matrix based on
 * an array of double precision floating point numbers.
 */
int mtxnullcoo_set_real_double(
    struct mtxnullcoo * A,
    int64_t size,
    int stride,
    const double * a)
{
    return mtxbasecoo_set_real_double(&A->base, size, stride, a);
}

/**
 * ‘mtxnullcoo_set_complex_single()’ sets values of a matrix based
 * on an array of single precision floating point complex numbers.
 */
int mtxnullcoo_set_complex_single(
    struct mtxnullcoo * A,
    int64_t size,
    int stride,
    const float (*a)[2])
{
    return mtxbasecoo_set_complex_single(&A->base, size, stride, a);
}

/**
 * ‘mtxnullcoo_set_complex_double()’ sets values of a matrix based
 * on an array of double precision floating point complex numbers.
 */
int mtxnullcoo_set_complex_double(
    struct mtxnullcoo * A,
    int64_t size,
    int stride,
    const double (*a)[2])
{
    return mtxbasecoo_set_complex_double(&A->base, size, stride, a);
}

/**
 * ‘mtxnullcoo_set_integer_single()’ sets values of a matrix based
 * on an array of integers.
 */
int mtxnullcoo_set_integer_single(
    struct mtxnullcoo * A,
    int64_t size,
    int stride,
    const int32_t * a)
{
    return mtxbasecoo_set_integer_single(&A->base, size, stride, a);
}

/**
 * ‘mtxnullcoo_set_integer_double()’ sets values of a matrix based
 * on an array of integers.
 */
int mtxnullcoo_set_integer_double(
    struct mtxnullcoo * A,
    int64_t size,
    int stride,
    const int64_t * a)
{
    return mtxbasecoo_set_integer_double(&A->base, size, stride, a);
}

/*
 * row and column vectors
 */

/**
 * ‘mtxnullcoo_alloc_row_vector()’ allocates a row vector for a
 * given matrix, where a row vector is a vector whose length equal to
 * a single row of the matrix.
 */
int mtxnullcoo_alloc_row_vector(
    const struct mtxnullcoo * A,
    struct mtxvector * x,
    enum mtxvectortype vectortype)
{
    return mtxbasecoo_alloc_row_vector(&A->base, x, vectortype);
}

/**
 * ‘mtxnullcoo_alloc_column_vector()’ allocates a column vector for
 * a given matrix, where a column vector is a vector whose length
 * equal to a single column of the matrix.
 */
int mtxnullcoo_alloc_column_vector(
    const struct mtxnullcoo * A,
    struct mtxvector * y,
    enum mtxvectortype vectortype)
{
    return mtxbasecoo_alloc_column_vector(&A->base, y, vectortype);
}

/*
 * convert to and from Matrix Market format
 */

/**
 * ‘mtxnullcoo_from_mtxfile()’ converts a matrix from Matrix Market
 * format.
 */
int mtxnullcoo_from_mtxfile(
    struct mtxnullcoo * A,
    const struct mtxfile * mtxfile)
{
    return mtxbasecoo_from_mtxfile(&A->base, mtxfile);
}

/**
 * ‘mtxnullcoo_to_mtxfile()’ converts a matrix to Matrix Market
 * format.
 */
int mtxnullcoo_to_mtxfile(
    struct mtxfile * mtxfile,
    const struct mtxnullcoo * A,
    int64_t num_rows,
    const int64_t * rowidx,
    int64_t num_columns,
    const int64_t * colidx,
    enum mtxfileformat mtxfmt)
{
    return mtxbasecoo_to_mtxfile(
        mtxfile, &A->base, num_rows, rowidx, num_columns, colidx, mtxfmt);
}

/*
 * partitioning
 */

/**
 * ‘mtxnullcoo_partition_rowwise()’ partitions the entries of a matrix
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
int mtxnullcoo_partition_rowwise(
    const struct mtxnullcoo * A,
    enum mtxpartitioning parttype,
    int num_parts,
    const int * partsizes,
    int blksize,
    const int * parts,
    int * dstnzpart,
    int64_t * dstnzpartsizes,
    int * dstrowpart,
    int64_t * dstrowpartsizes)
{
    return mtxbasecoo_partition_rowwise(
        &A->base, parttype, num_parts, partsizes, blksize, parts,
        dstnzpart, dstnzpartsizes, dstrowpart, dstrowpartsizes);
}

/**
 * ‘mtxnullcoo_partition_columnwise()’ partitions the entries of a
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
int mtxnullcoo_partition_columnwise(
    const struct mtxnullcoo * A,
    enum mtxpartitioning parttype,
    int num_parts,
    const int * partsizes,
    int blksize,
    const int * parts,
    int * dstnzpart,
    int64_t * dstnzpartsizes,
    int * dstcolpart,
    int64_t * dstcolpartsizes)
{
    return mtxbasecoo_partition_columnwise(
        &A->base, parttype, num_parts, partsizes, blksize, parts,
        dstnzpart, dstnzpartsizes, dstcolpart, dstcolpartsizes);
}

/**
 * ‘mtxnullcoo_partition_2d()’ partitions the entries of a matrix in a
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
int mtxnullcoo_partition_2d(
    const struct mtxnullcoo * A,
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
    int64_t * dstcolpartsizes)
{
    return mtxbasecoo_partition_2d(
        &A->base,
        rowparttype, num_row_parts, rowpartsizes, rowblksize, rowparts,
        colparttype, num_col_parts, colpartsizes, colblksize, colparts,
        dstnzpart, dstnzpartsizes,
        dstrowpart, dstrowpartsizes,
        dstcolpart, dstcolpartsizes);
}

/**
 * ‘mtxnullcoo_split()’ splits a matrix into multiple matrices
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
 * of type ‘struct mtxnullcoo’. If successful, then ‘dsts[p]’
 * points to a matrix consisting of elements from ‘src’ that belong to
 * the ‘p’th part, as designated by the ‘parts’ array.
 *
 * The caller is responsible for calling ‘mtxnullcoo_free()’ to
 * free storage allocated for each matrix in the ‘dsts’ array.
 */
int mtxnullcoo_split(
    int num_parts,
    struct mtxnullcoo ** dsts,
    const struct mtxnullcoo * src,
    int64_t size,
    int * parts)
{
    struct mtxbasecoo ** coodsts = malloc(
        num_parts * sizeof(struct mtxbasecoo *));
    if (!coodsts) return MTX_ERR_ERRNO;
    for (int p = 0; p < num_parts; p++) coodsts[p] = &dsts[p]->base;
    int err = mtxbasecoo_split(num_parts, coodsts, &src->base, size, parts);
    free(coodsts);
    return err;
}

/*
 * Level 1 BLAS operations
 */

/**
 * ‘mtxnullcoo_swap()’ swaps values of two matrices, simultaneously
 * performing ‘y <- x’ and ‘x <- y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxnullcoo_swap(
    struct mtxnullcoo * x,
    struct mtxnullcoo * y)
{
    return MTX_SUCCESS;
}

/**
 * ‘mtxnullcoo_copy()’ copies values of a matrix, ‘y = x’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxnullcoo_copy(
    struct mtxnullcoo * y,
    const struct mtxnullcoo * x)
{
    return MTX_SUCCESS;
}

/**
 * ‘mtxnullcoo_sscal()’ scales a matrix by a single precision
 * floating point scalar, ‘x = a*x’.
 */
int mtxnullcoo_sscal(
    float a,
    struct mtxnullcoo * x,
    int64_t * num_flops)
{
    return MTX_SUCCESS;
}

/**
 * ‘mtxnullcoo_dscal()’ scales a matrix by a double precision
 * floating point scalar, ‘x = a*x’.
 */
int mtxnullcoo_dscal(
    double a,
    struct mtxnullcoo * x,
    int64_t * num_flops)
{
    return MTX_SUCCESS;
}

/**
 * ‘mtxnullcoo_cscal()’ scales a matrix by a complex, single
 * precision floating point scalar, ‘x = (a+b*i)*x’.
 */
int mtxnullcoo_cscal(
    float a[2],
    struct mtxnullcoo * x,
    int64_t * num_flops)
{
    return MTX_SUCCESS;
}

/**
 * ‘mtxnullcoo_zscal()’ scales a matrix by a complex, double
 * precision floating point scalar, ‘x = (a+b*i)*x’.
 */
int mtxnullcoo_zscal(
    double a[2],
    struct mtxnullcoo * x,
    int64_t * num_flops)
{
    return MTX_SUCCESS;
}

/**
 * ‘mtxnullcoo_saxpy()’ adds a matrix to another one multiplied by
 * a single precision floating point value, ‘y = a*x + y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxnullcoo_saxpy(
    float a,
    const struct mtxnullcoo * x,
    struct mtxnullcoo * y,
    int64_t * num_flops)
{
    return MTX_SUCCESS;
}

/**
 * ‘mtxnullcoo_daxpy()’ adds a matrix to another one multiplied by
 * a double precision floating point value, ‘y = a*x + y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxnullcoo_daxpy(
    double a,
    const struct mtxnullcoo * x,
    struct mtxnullcoo * y,
    int64_t * num_flops)
{
    return MTX_SUCCESS;
}

/**
 * ‘mtxnullcoo_saypx()’ multiplies a matrix by a single precision
 * floating point scalar and adds another matrix, ‘y = a*y + x’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxnullcoo_saypx(
    float a,
    struct mtxnullcoo * y,
    const struct mtxnullcoo * x,
    int64_t * num_flops)
{
    return MTX_SUCCESS;
}

/**
 * ‘mtxnullcoo_daypx()’ multiplies a matrix by a double precision
 * floating point scalar and adds another matrix, ‘y = a*y + x’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxnullcoo_daypx(
    double a,
    struct mtxnullcoo * y,
    const struct mtxnullcoo * x,
    int64_t * num_flops)
{
    return MTX_SUCCESS;
}

/**
 * ‘mtxnullcoo_sdot()’ computes the Frobenius inner product of two
 * matrices in single precision floating point.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxnullcoo_sdot(
    const struct mtxnullcoo * x,
    const struct mtxnullcoo * y,
    float * dot,
    int64_t * num_flops)
{
    *dot = 0;
    return MTX_SUCCESS;
}

/**
 * ‘mtxnullcoo_ddot()’ computes the Frobenius inner product of two
 * matrices in double precision floating point.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxnullcoo_ddot(
    const struct mtxnullcoo * x,
    const struct mtxnullcoo * y,
    double * dot,
    int64_t * num_flops)
{
    *dot = 0;
    return MTX_SUCCESS;
}

/**
 * ‘mtxnullcoo_cdotu()’ computes the product of the transpose of a
 * complex row matrix with another complex row matrix in single
 * precision floating point, ‘dot := x^T*y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxnullcoo_cdotu(
    const struct mtxnullcoo * x,
    const struct mtxnullcoo * y,
    float (* dot)[2],
    int64_t * num_flops)
{
    (*dot)[0] = (*dot)[1] = 0;
    return MTX_SUCCESS;
}

/**
 * ‘mtxnullcoo_zdotu()’ computes the product of the transpose of a
 * complex row matrix with another complex row matrix in double
 * precision floating point, ‘dot := x^T*y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxnullcoo_zdotu(
    const struct mtxnullcoo * x,
    const struct mtxnullcoo * y,
    double (* dot)[2],
    int64_t * num_flops)
{
    (*dot)[0] = (*dot)[1] = 0;
    return MTX_SUCCESS;
}

/**
 * ‘mtxnullcoo_cdotc()’ computes the Frobenius inner product of two
 * complex matrices in single precision floating point, ‘dot :=
 * x^H*y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxnullcoo_cdotc(
    const struct mtxnullcoo * x,
    const struct mtxnullcoo * y,
    float (* dot)[2],
    int64_t * num_flops)
{
    (*dot)[0] = (*dot)[1] = 0;
    return MTX_SUCCESS;
}

/**
 * ‘mtxnullcoo_zdotc()’ computes the Frobenius inner product of two
 * complex matrices in double precision floating point, ‘dot :=
 * x^H*y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxnullcoo_zdotc(
    const struct mtxnullcoo * x,
    const struct mtxnullcoo * y,
    double (* dot)[2],
    int64_t * num_flops)
{
    (*dot)[0] = (*dot)[1] = 0;
    return MTX_SUCCESS;
}

/**
 * ‘mtxnullcoo_snrm2()’ computes the Frobenius norm of a matrix in
 * single precision floating point.
 */
int mtxnullcoo_snrm2(
    const struct mtxnullcoo * x,
    float * nrm2,
    int64_t * num_flops)
{
    *nrm2 = 0;
    return MTX_SUCCESS;
}

/**
 * ‘mtxnullcoo_dnrm2()’ computes the Frobenius norm of a matrix in
 * double precision floating point.
 */
int mtxnullcoo_dnrm2(
    const struct mtxnullcoo * x,
    double * nrm2,
    int64_t * num_flops)
{
    *nrm2 = 0;
    return MTX_SUCCESS;
}

/**
 * ‘mtxnullcoo_sasum()’ computes the sum of absolute values
 * (1-norm) of a matrix in single precision floating point.  If the
 * matrix is complex-valued, then the sum of the absolute values of
 * the real and imaginary parts is computed.
 */
int mtxnullcoo_sasum(
    const struct mtxnullcoo * x,
    float * asum,
    int64_t * num_flops)
{
    *asum = 0;
    return MTX_SUCCESS;
}

/**
 * ‘mtxnullcoo_dasum()’ computes the sum of absolute values
 * (1-norm) of a matrix in double precision floating point.  If the
 * matrix is complex-valued, then the sum of the absolute values of
 * the real and imaginary parts is computed.
 */
int mtxnullcoo_dasum(
    const struct mtxnullcoo * x,
    double * asum,
    int64_t * num_flops)
{
    *asum = 0;
    return MTX_SUCCESS;
}

/**
 * ‘mtxnullcoo_iamax()’ finds the index of the first element having
 * the maximum absolute value.  If the matrix is complex-valued, then
 * the index points to the first element having the maximum sum of the
 * absolute values of the real and imaginary parts.
 */
int mtxnullcoo_iamax(
    const struct mtxnullcoo * x,
    int * iamax)
{
    *iamax = 0;
    return MTX_SUCCESS;
}

/*
 * Level 2 BLAS operations (matrix-vector)
 */

/**
 * ‘mtxnullcoo_sgemv()’ multiplies a matrix ‘A’ or its transpose
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
int mtxnullcoo_sgemv(
    enum mtxtransposition trans,
    float alpha,
    const struct mtxnullcoo * A,
    const struct mtxvector * x,
    float beta,
    struct mtxvector * y,
    int64_t * num_flops)
{
    return MTX_SUCCESS;
}

/**
 * ‘mtxnullcoo_dgemv()’ multiplies a matrix ‘A’ or its transpose
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
int mtxnullcoo_dgemv(
    enum mtxtransposition trans,
    double alpha,
    const struct mtxnullcoo * A,
    const struct mtxvector * x,
    double beta,
    struct mtxvector * y,
    int64_t * num_flops)
{
    return MTX_SUCCESS;
}

/**
 * ‘mtxnullcoo_cgemv()’ multiplies a complex-valued matrix ‘A’, its
 * transpose ‘A'’ or its conjugate transpose ‘Aᴴ’ by a complex scalar
 * ‘alpha’ (‘α’) and a vector ‘x’, before adding the result to another
 * vector ‘y’ multiplied by another complex scalar ‘beta’ (‘β’).  That
 * is, ‘y = α*A*x + β*y’, ‘y = α*A'*x + β*y’ or ‘y = α*Aᴴ*x + β*y’.
 *
 * The scalars ‘alpha’ and ‘beta’ are given as single precision
 * floating point numbers.
 *
 * The vectors ‘x’ and ‘y’ must be of ‘array’ type, and they must have
 * the same field and precision as the matrix ‘A’. Moreover, if
 * ‘trans’ is ‘mtx_notrans’, then the size of ‘x’ must equal the
 * number of columns of ‘A’ and the size of ‘y’ must equal the number
 * of rows of ‘A’. if ‘trans’ is ‘mtx_trans’ or ‘mtx_conjtrans’, then
 * the size of ‘x’ must equal the number of rows of ‘A’ and the size
 * of ‘y’ must equal the number of columns of ‘A’.
 *
 * For now, the only case that is parallelised with OpenMP is
 * multiplication with non-transposed and unsymmetric matrices.
 */
int mtxnullcoo_cgemv(
    enum mtxtransposition trans,
    float alpha[2],
    const struct mtxnullcoo * A,
    const struct mtxvector * x,
    float beta[2],
    struct mtxvector * y,
    int64_t * num_flops)
{
    return MTX_SUCCESS;
}

/**
 * ‘mtxnullcoo_zgemv()’ multiplies a complex-valued matrix ‘A’, its
 * transpose ‘A'’ or its conjugate transpose ‘Aᴴ’ by a complex scalar
 * ‘alpha’ (‘α’) and a vector ‘x’, before adding the result to another
 * vector ‘y’ multiplied by another complex scalar ‘beta’ (‘β’).  That
 * is, ‘y = α*A*x + β*y’, ‘y = α*A'*x + β*y’ or ‘y = α*Aᴴ*x + β*y’.
 *
 * The scalars ‘alpha’ and ‘beta’ are given as double precision
 * floating point numbers.
 *
 * The vectors ‘x’ and ‘y’ must be of ‘array’ type, and they must have
 * the same field and precision as the matrix ‘A’. Moreover, if
 * ‘trans’ is ‘mtx_notrans’, then the size of ‘x’ must equal the
 * number of columns of ‘A’ and the size of ‘y’ must equal the number
 * of rows of ‘A’. if ‘trans’ is ‘mtx_trans’ or ‘mtx_conjtrans’, then
 * the size of ‘x’ must equal the number of rows of ‘A’ and the size
 * of ‘y’ must equal the number of columns of ‘A’.
 *
 * For now, the only case that is parallelised with OpenMP is
 * multiplication with non-transposed and unsymmetric matrices.
 */
int mtxnullcoo_zgemv(
    enum mtxtransposition trans,
    double alpha[2],
    const struct mtxnullcoo * A,
    const struct mtxvector * x,
    double beta[2],
    struct mtxvector * y,
    int64_t * num_flops)
{
    return MTX_SUCCESS;
}
