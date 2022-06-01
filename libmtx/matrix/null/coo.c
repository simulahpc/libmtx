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
 * Data structures for matrices in coordinate format, where matrix
 * operations do nothing. While this produces incorrect results, it is
 * sometimes useful for the purpose of eliminating matrix operations
 * while debugging or carrying out performance measurements.
 */

#include <libmtx/libmtx-config.h>

#include <libmtx/error.h>
#include <libmtx/matrix/null/coo.h>
#include <libmtx/matrix/base/coo.h>
#include <libmtx/mtxfile/mtxfile.h>
#include <libmtx/vector/field.h>
#include <libmtx/vector/precision.h>
#include <libmtx/vector/vector.h>
#include <libmtx/util/partition.h>

#include <stddef.h>
#include <stdlib.h>

/*
 * matrix properties
 */

/**
 * ‘mtxmatrix_nullcoo_field()’ gets the field of a matrix.
 */
enum mtxfield mtxmatrix_nullcoo_field(const struct mtxmatrix_nullcoo * A)
{
    return mtxmatrix_coo_field(&A->base);
}

/**
 * ‘mtxmatrix_nullcoo_precision()’ gets the precision of a matrix.
 */
enum mtxprecision mtxmatrix_nullcoo_precision(const struct mtxmatrix_nullcoo * A)
{
    return mtxmatrix_coo_precision(&A->base);
}

/**
 * ‘mtxmatrix_nullcoo_symmetry()’ gets the symmetry of a matrix.
 */
enum mtxsymmetry mtxmatrix_nullcoo_symmetry(const struct mtxmatrix_nullcoo * A)
{
    return mtxmatrix_coo_symmetry(&A->base);
}

/**
 * ‘mtxmatrix_nullcoo_num_nonzeros()’ gets the number of the number of
 *  nonzero matrix entries, including those represented implicitly due
 *  to symmetry.
 */
int64_t mtxmatrix_nullcoo_num_nonzeros(const struct mtxmatrix_nullcoo * A)
{
    return mtxmatrix_coo_num_nonzeros(&A->base);
}

/**
 * ‘mtxmatrix_nullcoo_size()’ gets the number of explicitly stored
 * nonzeros of a matrix.
 */
int64_t mtxmatrix_nullcoo_size(const struct mtxmatrix_nullcoo * A)
{
    return mtxmatrix_coo_size(&A->base);
}

/*
 * memory management
 */

/**
 * ‘mtxmatrix_nullcoo_free()’ frees storage allocated for a matrix.
 */
void mtxmatrix_nullcoo_free(
    struct mtxmatrix_nullcoo * A)
{
    mtxmatrix_coo_free(&A->base);
}

/**
 * ‘mtxmatrix_nullcoo_alloc_copy()’ allocates a copy of a matrix without
 * initialising the values.
 */
int mtxmatrix_nullcoo_alloc_copy(
    struct mtxmatrix_nullcoo * dst,
    const struct mtxmatrix_nullcoo * src)
{
    return mtxmatrix_coo_alloc_copy(&dst->base, &src->base);
}

/**
 * ‘mtxmatrix_nullcoo_init_copy()’ allocates a copy of a matrix and also
 * copies the values.
 */
int mtxmatrix_nullcoo_init_copy(
    struct mtxmatrix_nullcoo * dst,
    const struct mtxmatrix_nullcoo * src)
{
    return mtxmatrix_coo_init_copy(&dst->base, &src->base);
}

/*
 * initialise matrices from entrywise data in coordinate format
 */

/**
 * ‘mtxmatrix_nullcoo_alloc_entries()’ allocates a matrix from entrywise
 * data in coordinate format.
 */
int mtxmatrix_nullcoo_alloc_entries(
    struct mtxmatrix_nullcoo * A,
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
    return mtxmatrix_coo_alloc_entries(
        &A->base, field, precision, symmetry, num_rows, num_columns, size,
        idxstride, idxbase, rowidx, colidx);
}

/**
 * ‘mtxmatrix_nullcoo_init_entries_real_single()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * real, single precision coefficients.
 */
int mtxmatrix_nullcoo_init_entries_real_single(
    struct mtxmatrix_nullcoo * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx,
    const float * data)
{
    return mtxmatrix_coo_init_entries_real_single(
        &A->base, symmetry, num_rows, num_columns, size, rowidx, colidx, data);
}

/**
 * ‘mtxmatrix_nullcoo_init_entries_real_double()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * real, double precision coefficients.
 */
int mtxmatrix_nullcoo_init_entries_real_double(
    struct mtxmatrix_nullcoo * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx,
    const double * data)
{
    return mtxmatrix_coo_init_entries_real_double(
        &A->base, symmetry, num_rows, num_columns, size, rowidx, colidx, data);
}

/**
 * ‘mtxmatrix_nullcoo_init_entries_complex_single()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * complex, single precision coefficients.
 */
int mtxmatrix_nullcoo_init_entries_complex_single(
    struct mtxmatrix_nullcoo * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx,
    const float (* data)[2])
{
    return mtxmatrix_coo_init_entries_complex_single(
        &A->base, symmetry, num_rows, num_columns, size, rowidx, colidx, data);
}

/**
 * ‘mtxmatrix_nullcoo_init_entries_complex_double()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * complex, double precision coefficients.
 */
int mtxmatrix_nullcoo_init_entries_complex_double(
    struct mtxmatrix_nullcoo * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx,
    const double (* data)[2])
{
    return mtxmatrix_coo_init_entries_complex_double(
        &A->base, symmetry, num_rows, num_columns, size, rowidx, colidx, data);
}

/**
 * ‘mtxmatrix_nullcoo_init_entries_integer_single()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * integer, single precision coefficients.
 */
int mtxmatrix_nullcoo_init_entries_integer_single(
    struct mtxmatrix_nullcoo * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx,
    const int32_t * data)
{
    return mtxmatrix_coo_init_entries_integer_single(
        &A->base, symmetry, num_rows, num_columns, size, rowidx, colidx, data);
}

/**
 * ‘mtxmatrix_nullcoo_init_entries_integer_double()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * integer, double precision coefficients.
 */
int mtxmatrix_nullcoo_init_entries_integer_double(
    struct mtxmatrix_nullcoo * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx,
    const int64_t * data)
{
    return mtxmatrix_coo_init_entries_integer_double(
        &A->base, symmetry, num_rows, num_columns, size, rowidx, colidx, data);
}

/**
 * ‘mtxmatrix_nullcoo_init_entries_pattern()’ allocates and initialises a
 * matrix from entrywise data in coordinate format with boolean
 * coefficients.
 */
int mtxmatrix_nullcoo_init_entries_pattern(
    struct mtxmatrix_nullcoo * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx)
{
    return mtxmatrix_coo_init_entries_pattern(
        &A->base, symmetry, num_rows, num_columns, size, rowidx, colidx);
}

/*
 * initialise matrices from entrywise data in coordinate format with
 * specified strides
 */

/**
 * ‘mtxmatrix_nullcoo_init_entries_strided_real_single()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * real, single precision coefficients.
 */
int mtxmatrix_nullcoo_init_entries_strided_real_single(
    struct mtxmatrix_nullcoo * A,
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
 * ‘mtxmatrix_nullcoo_init_entries_strided_real_double()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * real, double precision coefficients.
 */
int mtxmatrix_nullcoo_init_entries_strided_real_double(
    struct mtxmatrix_nullcoo * A,
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
 * ‘mtxmatrix_nullcoo_init_entries_strided_complex_single()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * complex, single precision coefficients.
 */
int mtxmatrix_nullcoo_init_entries_strided_complex_single(
    struct mtxmatrix_nullcoo * A,
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
 * ‘mtxmatrix_nullcoo_init_entries_strided_complex_double()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * complex, double precision coefficients.
 */
int mtxmatrix_nullcoo_init_entries_strided_complex_double(
    struct mtxmatrix_nullcoo * A,
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
 * ‘mtxmatrix_nullcoo_init_entries_strided_integer_single()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * integer, single precision coefficients.
 */
int mtxmatrix_nullcoo_init_entries_strided_integer_single(
    struct mtxmatrix_nullcoo * A,
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
 * ‘mtxmatrix_nullcoo_init_entries_strided_integer_double()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * integer, double precision coefficients.
 */
int mtxmatrix_nullcoo_init_entries_strided_integer_double(
    struct mtxmatrix_nullcoo * A,
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
 * ‘mtxmatrix_nullcoo_init_entries_strided_pattern()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * boolean coefficients.
 */
int mtxmatrix_nullcoo_init_entries_strided_pattern(
    struct mtxmatrix_nullcoo * A,
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
 * ‘mtxmatrix_nullcoo_alloc_rows()’ allocates a matrix from row-wise data
 * in compressed row format.
 */
int mtxmatrix_nullcoo_alloc_rows(
    struct mtxmatrix_nullcoo * A,
    enum mtxfield field,
    enum mtxprecision precision,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx)
{
    return mtxmatrix_coo_alloc_rows(
        &A->base, field, precision, symmetry,
        num_rows, num_columns, rowptr, colidx);
}

/**
 * ‘mtxmatrix_nullcoo_init_rows_real_single()’ allocates and initialises a
 * matrix from row-wise data in compressed row format with real,
 * single precision coefficients.
 */
int mtxmatrix_nullcoo_init_rows_real_single(
    struct mtxmatrix_nullcoo * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const float * data)
{
    return mtxmatrix_coo_init_rows_real_single(
        &A->base, symmetry, num_rows, num_columns, rowptr, colidx, data);
}

/**
 * ‘mtxmatrix_nullcoo_init_rows_real_double()’ allocates and initialises a
 * matrix from row-wise data in compressed row format with real,
 * double precision coefficients.
 */
int mtxmatrix_nullcoo_init_rows_real_double(
    struct mtxmatrix_nullcoo * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const double * data)
{
    return mtxmatrix_coo_init_rows_real_double(
        &A->base, symmetry, num_rows, num_columns, rowptr, colidx, data);
}

/**
 * ‘mtxmatrix_nullcoo_init_rows_complex_single()’ allocates and
 * initialises a matrix from row-wise data in compressed row format
 * with complex, single precision coefficients.
 */
int mtxmatrix_nullcoo_init_rows_complex_single(
    struct mtxmatrix_nullcoo * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const float (* data)[2])
{
    return mtxmatrix_coo_init_rows_complex_single(
        &A->base, symmetry, num_rows, num_columns, rowptr, colidx, data);
}

/**
 * ‘mtxmatrix_nullcoo_init_rows_complex_double()’ allocates and
 * initialises a matrix from row-wise data in compressed row format
 * with complex, double precision coefficients.
 */
int mtxmatrix_nullcoo_init_rows_complex_double(
    struct mtxmatrix_nullcoo * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const double (* data)[2])
{
    return mtxmatrix_coo_init_rows_complex_double(
        &A->base, symmetry, num_rows, num_columns, rowptr, colidx, data);
}

/**
 * ‘mtxmatrix_nullcoo_init_rows_integer_single()’ allocates and
 * initialises a matrix from row-wise data in compressed row format
 * with integer, single precision coefficients.
 */
int mtxmatrix_nullcoo_init_rows_integer_single(
    struct mtxmatrix_nullcoo * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const int32_t * data)
{
    return mtxmatrix_coo_init_rows_integer_single(
        &A->base, symmetry, num_rows, num_columns, rowptr, colidx, data);
}

/**
 * ‘mtxmatrix_nullcoo_init_rows_integer_double()’ allocates and
 * initialises a matrix from row-wise data in compressed row format
 * with integer, double precision coefficients.
 */
int mtxmatrix_nullcoo_init_rows_integer_double(
    struct mtxmatrix_nullcoo * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const int64_t * data)
{
    return mtxmatrix_coo_init_rows_integer_double(
        &A->base, symmetry, num_rows, num_columns, rowptr, colidx, data);
}

/**
 * ‘mtxmatrix_nullcoo_init_rows_pattern()’ allocates and initialises a
 * matrix from row-wise data in compressed row format with boolean
 * coefficients.
 */
int mtxmatrix_nullcoo_init_rows_pattern(
    struct mtxmatrix_nullcoo * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx)
{
    return mtxmatrix_coo_init_rows_pattern(
        &A->base, symmetry, num_rows, num_columns, rowptr, colidx);
}

/*
 * initialise matrices from column-wise data in compressed column
 * format
 */

/**
 * ‘mtxmatrix_nullcoo_alloc_columns()’ allocates a matrix from column-wise
 * data in compressed column format.
 */
int mtxmatrix_nullcoo_alloc_columns(
    struct mtxmatrix_nullcoo * A,
    enum mtxfield field,
    enum mtxprecision precision,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx);

/**
 * ‘mtxmatrix_nullcoo_init_columns_real_single()’ allocates and
 * initialises a matrix from column-wise data in compressed column
 * format with real, single precision coefficients.
 */
int mtxmatrix_nullcoo_init_columns_real_single(
    struct mtxmatrix_nullcoo * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx,
    const float * data);

/**
 * ‘mtxmatrix_nullcoo_init_columns_real_double()’ allocates and
 * initialises a matrix from column-wise data in compressed column
 * format with real, double precision coefficients.
 */
int mtxmatrix_nullcoo_init_columns_real_double(
    struct mtxmatrix_nullcoo * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx,
    const double * data);

/**
 * ‘mtxmatrix_nullcoo_init_columns_complex_single()’ allocates and
 * initialises a matrix from column-wise data in compressed column
 * format with complex, single precision coefficients.
 */
int mtxmatrix_nullcoo_init_columns_complex_single(
    struct mtxmatrix_nullcoo * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx,
    const float (* data)[2]);

/**
 * ‘mtxmatrix_nullcoo_init_columns_complex_double()’ allocates and
 * initialises a matrix from column-wise data in compressed column
 * format with complex, double precision coefficients.
 */
int mtxmatrix_nullcoo_init_columns_complex_double(
    struct mtxmatrix_nullcoo * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx,
    const double (* data)[2]);

/**
 * ‘mtxmatrix_nullcoo_init_columns_integer_single()’ allocates and
 * initialises a matrix from column-wise data in compressed column
 * format with integer, single precision coefficients.
 */
int mtxmatrix_nullcoo_init_columns_integer_single(
    struct mtxmatrix_nullcoo * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx,
    const int32_t * data);

/**
 * ‘mtxmatrix_nullcoo_init_columns_integer_double()’ allocates and
 * initialises a matrix from column-wise data in compressed column
 * format with integer, double precision coefficients.
 */
int mtxmatrix_nullcoo_init_columns_integer_double(
    struct mtxmatrix_nullcoo * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx,
    const int64_t * data);

/**
 * ‘mtxmatrix_nullcoo_init_columns_pattern()’ allocates and initialises a
 * matrix from column-wise data in compressed column format with
 * boolean coefficients.
 */
int mtxmatrix_nullcoo_init_columns_pattern(
    struct mtxmatrix_nullcoo * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx);

/*
 * initialise matrices from a list of dense cliques
 */

/**
 * ‘mtxmatrix_nullcoo_alloc_cliques()’ allocates a matrix from a list of
 * cliques.
 */
int mtxmatrix_nullcoo_alloc_cliques(
    struct mtxmatrix_nullcoo * A,
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
 * ‘mtxmatrix_nullcoo_init_cliques_real_single()’ allocates and
 * initialises a matrix from a list of cliques with real, single
 * precision coefficients.
 */
int mtxmatrix_nullcoo_init_cliques_real_single(
    struct mtxmatrix_nullcoo * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_cliques,
    const int64_t * cliqueptr,
    const int * rowidx,
    const int * colidx,
    const float * data);

/**
 * ‘mtxmatrix_nullcoo_init_cliques_real_double()’ allocates and
 * initialises a matrix from a list of cliques with real, double
 * precision coefficients.
 */
int mtxmatrix_nullcoo_init_cliques_real_double(
    struct mtxmatrix_nullcoo * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_cliques,
    const int64_t * cliqueptr,
    const int * rowidx,
    const int * colidx,
    const double * data);

/**
 * ‘mtxmatrix_nullcoo_init_cliques_complex_single()’ allocates and
 * initialises a matrix from a list of cliques with complex, single
 * precision coefficients.
 */
int mtxmatrix_nullcoo_init_cliques_complex_single(
    struct mtxmatrix_nullcoo * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_cliques,
    const int64_t * cliqueptr,
    const int * rowidx,
    const int * colidx,
    const float (* data)[2]);

/**
 * ‘mtxmatrix_nullcoo_init_cliques_complex_double()’ allocates and
 * initialises a matrix from a list of cliques with complex, double
 * precision coefficients.
 */
int mtxmatrix_nullcoo_init_cliques_complex_double(
    struct mtxmatrix_nullcoo * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_cliques,
    const int64_t * cliqueptr,
    const int * rowidx,
    const int * colidx,
    const double (* data)[2]);

/**
 * ‘mtxmatrix_nullcoo_init_cliques_integer_single()’ allocates and
 * initialises a matrix from a list of cliques with integer, single
 * precision coefficients.
 */
int mtxmatrix_nullcoo_init_cliques_integer_single(
    struct mtxmatrix_nullcoo * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_cliques,
    const int64_t * cliqueptr,
    const int * rowidx,
    const int * colidx,
    const int32_t * data);

/**
 * ‘mtxmatrix_nullcoo_init_cliques_integer_double()’ allocates and
 * initialises a matrix from a list of cliques with integer, double
 * precision coefficients.
 */
int mtxmatrix_nullcoo_init_cliques_integer_double(
    struct mtxmatrix_nullcoo * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_cliques,
    const int64_t * cliqueptr,
    const int * rowidx,
    const int * colidx,
    const int64_t * data);

/**
 * ‘mtxmatrix_nullcoo_init_cliques_pattern()’ allocates and initialises a
 * matrix from a list of cliques with boolean coefficients.
 */
int mtxmatrix_nullcoo_init_cliques_pattern(
    struct mtxmatrix_nullcoo * A,
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
 * ‘mtxmatrix_nullcoo_setzero()’ sets every value of a matrix to zero.
 */
int mtxmatrix_nullcoo_setzero(
    struct mtxmatrix_nullcoo * A)
{
    return mtxmatrix_coo_setzero(&A->base);
}

/**
 * ‘mtxmatrix_nullcoo_set_real_single()’ sets values of a matrix based on
 * an array of single precision floating point numbers.
 */
int mtxmatrix_nullcoo_set_real_single(
    struct mtxmatrix_nullcoo * A,
    int64_t size,
    int stride,
    const float * a)
{
    return mtxmatrix_coo_set_real_single(&A->base, size, stride, a);
}

/**
 * ‘mtxmatrix_nullcoo_set_real_double()’ sets values of a matrix based on
 * an array of double precision floating point numbers.
 */
int mtxmatrix_nullcoo_set_real_double(
    struct mtxmatrix_nullcoo * A,
    int64_t size,
    int stride,
    const double * a)
{
    return mtxmatrix_coo_set_real_double(&A->base, size, stride, a);
}

/**
 * ‘mtxmatrix_nullcoo_set_complex_single()’ sets values of a matrix based
 * on an array of single precision floating point complex numbers.
 */
int mtxmatrix_nullcoo_set_complex_single(
    struct mtxmatrix_nullcoo * A,
    int64_t size,
    int stride,
    const float (*a)[2])
{
    return mtxmatrix_coo_set_complex_single(&A->base, size, stride, a);
}

/**
 * ‘mtxmatrix_nullcoo_set_complex_double()’ sets values of a matrix based
 * on an array of double precision floating point complex numbers.
 */
int mtxmatrix_nullcoo_set_complex_double(
    struct mtxmatrix_nullcoo * A,
    int64_t size,
    int stride,
    const double (*a)[2])
{
    return mtxmatrix_coo_set_complex_double(&A->base, size, stride, a);
}

/**
 * ‘mtxmatrix_nullcoo_set_integer_single()’ sets values of a matrix based
 * on an array of integers.
 */
int mtxmatrix_nullcoo_set_integer_single(
    struct mtxmatrix_nullcoo * A,
    int64_t size,
    int stride,
    const int32_t * a)
{
    return mtxmatrix_coo_set_integer_single(&A->base, size, stride, a);
}

/**
 * ‘mtxmatrix_nullcoo_set_integer_double()’ sets values of a matrix based
 * on an array of integers.
 */
int mtxmatrix_nullcoo_set_integer_double(
    struct mtxmatrix_nullcoo * A,
    int64_t size,
    int stride,
    const int64_t * a)
{
    return mtxmatrix_coo_set_integer_double(&A->base, size, stride, a);
}

/*
 * row and column vectors
 */

/**
 * ‘mtxmatrix_nullcoo_alloc_row_vector()’ allocates a row vector for a
 * given matrix, where a row vector is a vector whose length equal to
 * a single row of the matrix.
 */
int mtxmatrix_nullcoo_alloc_row_vector(
    const struct mtxmatrix_nullcoo * A,
    struct mtxvector * x,
    enum mtxvectortype vectortype)
{
    return mtxmatrix_coo_alloc_row_vector(&A->base, x, vectortype);
}

/**
 * ‘mtxmatrix_nullcoo_alloc_column_vector()’ allocates a column vector for
 * a given matrix, where a column vector is a vector whose length
 * equal to a single column of the matrix.
 */
int mtxmatrix_nullcoo_alloc_column_vector(
    const struct mtxmatrix_nullcoo * A,
    struct mtxvector * y,
    enum mtxvectortype vectortype)
{
    return mtxmatrix_coo_alloc_column_vector(&A->base, y, vectortype);
}

/*
 * convert to and from Matrix Market format
 */

/**
 * ‘mtxmatrix_nullcoo_from_mtxfile()’ converts a matrix from Matrix Market
 * format.
 */
int mtxmatrix_nullcoo_from_mtxfile(
    struct mtxmatrix_nullcoo * A,
    const struct mtxfile * mtxfile)
{
    return mtxmatrix_coo_from_mtxfile(&A->base, mtxfile);
}

/**
 * ‘mtxmatrix_nullcoo_to_mtxfile()’ converts a matrix to Matrix Market
 * format.
 */
int mtxmatrix_nullcoo_to_mtxfile(
    struct mtxfile * mtxfile,
    const struct mtxmatrix_nullcoo * A,
    int64_t num_rows,
    const int64_t * rowidx,
    int64_t num_columns,
    const int64_t * colidx,
    enum mtxfileformat mtxfmt)
{
    return mtxmatrix_coo_to_mtxfile(
        mtxfile, &A->base, num_rows, rowidx, num_columns, colidx, mtxfmt);
}

/*
 * partitioning
 */

/**
 * ‘mtxmatrix_nullcoo_partition_rowwise()’ partitions the entries of a
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
int mtxmatrix_nullcoo_partition_rowwise(
    const struct mtxmatrix_nullcoo * A,
    enum mtxpartitioning parttype,
    int num_parts,
    const int * partsizes,
    int blksize,
    const int * parts,
    int * dstpart,
    int64_t * dstpartsizes)
{
    return mtxmatrix_coo_partition_rowwise(
        &A->base, parttype, num_parts, partsizes, blksize, parts,
        dstpart, dstpartsizes);
}

/**
 * ‘mtxmatrix_nullcoo_partition_columnwise()’ partitions the entries
 * of a matrix columnwise.
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
int mtxmatrix_nullcoo_partition_columnwise(
    const struct mtxmatrix_nullcoo * A,
    enum mtxpartitioning parttype,
    int num_parts,
    const int * partsizes,
    int blksize,
    const int * parts,
    int * dstpart,
    int64_t * dstpartsizes)
{
    return mtxmatrix_coo_partition_columnwise(
        &A->base, parttype, num_parts, partsizes, blksize, parts,
        dstpart, dstpartsizes);
}

/**
 * ‘mtxmatrix_nullcoo_split()’ splits a matrix into multiple matrices
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
 * of type ‘struct mtxmatrix_nullcoo’. If successful, then ‘dsts[p]’
 * points to a matrix consisting of elements from ‘src’ that belong to
 * the ‘p’th part, as designated by the ‘parts’ array.
 *
 * The caller is responsible for calling ‘mtxmatrix_nullcoo_free()’ to
 * free storage allocated for each matrix in the ‘dsts’ array.
 */
int mtxmatrix_nullcoo_split(
    int num_parts,
    struct mtxmatrix_nullcoo ** dsts,
    const struct mtxmatrix_nullcoo * src,
    int64_t size,
    int * parts)
{
    struct mtxmatrix_coo ** coodsts = malloc(
        num_parts * sizeof(struct mtxmatrix_coo *));
    if (!coodsts) return MTX_ERR_ERRNO;
    for (int p = 0; p < num_parts; p++) coodsts[p] = &dsts[p]->base;
    int err = mtxmatrix_coo_split(num_parts, coodsts, &src->base, size, parts);
    free(coodsts);
    return err;
}

/*
 * Level 1 BLAS operations
 */

/**
 * ‘mtxmatrix_nullcoo_swap()’ swaps values of two matrices, simultaneously
 * performing ‘y <- x’ and ‘x <- y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_nullcoo_swap(
    struct mtxmatrix_nullcoo * x,
    struct mtxmatrix_nullcoo * y)
{
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_nullcoo_copy()’ copies values of a matrix, ‘y = x’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_nullcoo_copy(
    struct mtxmatrix_nullcoo * y,
    const struct mtxmatrix_nullcoo * x)
{
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_nullcoo_sscal()’ scales a matrix by a single precision
 * floating point scalar, ‘x = a*x’.
 */
int mtxmatrix_nullcoo_sscal(
    float a,
    struct mtxmatrix_nullcoo * x,
    int64_t * num_flops)
{
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_nullcoo_dscal()’ scales a matrix by a double precision
 * floating point scalar, ‘x = a*x’.
 */
int mtxmatrix_nullcoo_dscal(
    double a,
    struct mtxmatrix_nullcoo * x,
    int64_t * num_flops)
{
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_nullcoo_cscal()’ scales a matrix by a complex, single
 * precision floating point scalar, ‘x = (a+b*i)*x’.
 */
int mtxmatrix_nullcoo_cscal(
    float a[2],
    struct mtxmatrix_nullcoo * x,
    int64_t * num_flops)
{
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_nullcoo_zscal()’ scales a matrix by a complex, double
 * precision floating point scalar, ‘x = (a+b*i)*x’.
 */
int mtxmatrix_nullcoo_zscal(
    double a[2],
    struct mtxmatrix_nullcoo * x,
    int64_t * num_flops)
{
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_nullcoo_saxpy()’ adds a matrix to another one multiplied by
 * a single precision floating point value, ‘y = a*x + y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_nullcoo_saxpy(
    float a,
    const struct mtxmatrix_nullcoo * x,
    struct mtxmatrix_nullcoo * y,
    int64_t * num_flops)
{
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_nullcoo_daxpy()’ adds a matrix to another one multiplied by
 * a double precision floating point value, ‘y = a*x + y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_nullcoo_daxpy(
    double a,
    const struct mtxmatrix_nullcoo * x,
    struct mtxmatrix_nullcoo * y,
    int64_t * num_flops)
{
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_nullcoo_saypx()’ multiplies a matrix by a single precision
 * floating point scalar and adds another matrix, ‘y = a*y + x’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_nullcoo_saypx(
    float a,
    struct mtxmatrix_nullcoo * y,
    const struct mtxmatrix_nullcoo * x,
    int64_t * num_flops)
{
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_nullcoo_daypx()’ multiplies a matrix by a double precision
 * floating point scalar and adds another matrix, ‘y = a*y + x’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_nullcoo_daypx(
    double a,
    struct mtxmatrix_nullcoo * y,
    const struct mtxmatrix_nullcoo * x,
    int64_t * num_flops)
{
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_nullcoo_sdot()’ computes the Frobenius inner product of two
 * matrices in single precision floating point.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_nullcoo_sdot(
    const struct mtxmatrix_nullcoo * x,
    const struct mtxmatrix_nullcoo * y,
    float * dot,
    int64_t * num_flops)
{
    *dot = 0;
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_nullcoo_ddot()’ computes the Frobenius inner product of two
 * matrices in double precision floating point.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_nullcoo_ddot(
    const struct mtxmatrix_nullcoo * x,
    const struct mtxmatrix_nullcoo * y,
    double * dot,
    int64_t * num_flops)
{
    *dot = 0;
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_nullcoo_cdotu()’ computes the product of the transpose of a
 * complex row matrix with another complex row matrix in single
 * precision floating point, ‘dot := x^T*y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_nullcoo_cdotu(
    const struct mtxmatrix_nullcoo * x,
    const struct mtxmatrix_nullcoo * y,
    float (* dot)[2],
    int64_t * num_flops)
{
    (*dot)[0] = (*dot)[1] = 0;
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_nullcoo_zdotu()’ computes the product of the transpose of a
 * complex row matrix with another complex row matrix in double
 * precision floating point, ‘dot := x^T*y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_nullcoo_zdotu(
    const struct mtxmatrix_nullcoo * x,
    const struct mtxmatrix_nullcoo * y,
    double (* dot)[2],
    int64_t * num_flops)
{
    (*dot)[0] = (*dot)[1] = 0;
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_nullcoo_cdotc()’ computes the Frobenius inner product of two
 * complex matrices in single precision floating point, ‘dot :=
 * x^H*y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_nullcoo_cdotc(
    const struct mtxmatrix_nullcoo * x,
    const struct mtxmatrix_nullcoo * y,
    float (* dot)[2],
    int64_t * num_flops)
{
    (*dot)[0] = (*dot)[1] = 0;
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_nullcoo_zdotc()’ computes the Frobenius inner product of two
 * complex matrices in double precision floating point, ‘dot :=
 * x^H*y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_nullcoo_zdotc(
    const struct mtxmatrix_nullcoo * x,
    const struct mtxmatrix_nullcoo * y,
    double (* dot)[2],
    int64_t * num_flops)
{
    (*dot)[0] = (*dot)[1] = 0;
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_nullcoo_snrm2()’ computes the Frobenius norm of a matrix in
 * single precision floating point.
 */
int mtxmatrix_nullcoo_snrm2(
    const struct mtxmatrix_nullcoo * x,
    float * nrm2,
    int64_t * num_flops)
{
    *nrm2 = 0;
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_nullcoo_dnrm2()’ computes the Frobenius norm of a matrix in
 * double precision floating point.
 */
int mtxmatrix_nullcoo_dnrm2(
    const struct mtxmatrix_nullcoo * x,
    double * nrm2,
    int64_t * num_flops)
{
    *nrm2 = 0;
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_nullcoo_sasum()’ computes the sum of absolute values
 * (1-norm) of a matrix in single precision floating point.  If the
 * matrix is complex-valued, then the sum of the absolute values of
 * the real and imaginary parts is computed.
 */
int mtxmatrix_nullcoo_sasum(
    const struct mtxmatrix_nullcoo * x,
    float * asum,
    int64_t * num_flops)
{
    *asum = 0;
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_nullcoo_dasum()’ computes the sum of absolute values
 * (1-norm) of a matrix in double precision floating point.  If the
 * matrix is complex-valued, then the sum of the absolute values of
 * the real and imaginary parts is computed.
 */
int mtxmatrix_nullcoo_dasum(
    const struct mtxmatrix_nullcoo * x,
    double * asum,
    int64_t * num_flops)
{
    *asum = 0;
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_nullcoo_iamax()’ finds the index of the first element having
 * the maximum absolute value.  If the matrix is complex-valued, then
 * the index points to the first element having the maximum sum of the
 * absolute values of the real and imaginary parts.
 */
int mtxmatrix_nullcoo_iamax(
    const struct mtxmatrix_nullcoo * x,
    int * iamax)
{
    *iamax = 0;
    return MTX_SUCCESS;
}

/*
 * Level 2 BLAS operations (matrix-vector)
 */

/**
 * ‘mtxmatrix_nullcoo_sgemv()’ multiplies a matrix ‘A’ or its transpose
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
int mtxmatrix_nullcoo_sgemv(
    enum mtxtransposition trans,
    float alpha,
    const struct mtxmatrix_nullcoo * A,
    const struct mtxvector * x,
    float beta,
    struct mtxvector * y,
    int64_t * num_flops)
{
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_nullcoo_dgemv()’ multiplies a matrix ‘A’ or its transpose
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
int mtxmatrix_nullcoo_dgemv(
    enum mtxtransposition trans,
    double alpha,
    const struct mtxmatrix_nullcoo * A,
    const struct mtxvector * x,
    double beta,
    struct mtxvector * y,
    int64_t * num_flops)
{
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_nullcoo_cgemv()’ multiplies a complex-valued matrix ‘A’, its
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
int mtxmatrix_nullcoo_cgemv(
    enum mtxtransposition trans,
    float alpha[2],
    const struct mtxmatrix_nullcoo * A,
    const struct mtxvector * x,
    float beta[2],
    struct mtxvector * y,
    int64_t * num_flops)
{
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_nullcoo_zgemv()’ multiplies a complex-valued matrix ‘A’, its
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
int mtxmatrix_nullcoo_zgemv(
    enum mtxtransposition trans,
    double alpha[2],
    const struct mtxmatrix_nullcoo * A,
    const struct mtxvector * x,
    double beta[2],
    struct mtxvector * y,
    int64_t * num_flops)
{
    return MTX_SUCCESS;
}
