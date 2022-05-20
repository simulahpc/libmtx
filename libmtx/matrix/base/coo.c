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
 * Data structures for matrices in coordinate format.
 */

#include <libmtx/libmtx-config.h>

#include <libmtx/error.h>
#include <libmtx/vector/field.h>
#include <libmtx/matrix/matrix.h>
#include <libmtx/matrix/base/coo.h>
#include <libmtx/mtxfile/mtxfile.h>
#include <libmtx/vector/precision.h>
#include <libmtx/vector/base.h>
#include <libmtx/vector/vector.h>

#include <errno.h>

#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * memory management
 */

/**
 * ‘mtxmatrix_coordinate_free()’ frees storage allocated for a matrix.
 */
void mtxmatrix_coordinate_free(
    struct mtxmatrix_coordinate * A)
{
    mtxvector_base_free(&A->a);
    free(A->colidx);
    free(A->rowidx);
}

/**
 * ‘mtxmatrix_coordinate_alloc_copy()’ allocates a copy of a matrix
 * without initialising the values.
 */
int mtxmatrix_coordinate_alloc_copy(
    struct mtxmatrix_coordinate * dst,
    const struct mtxmatrix_coordinate * src)
{
    return mtxmatrix_coordinate_alloc_entries(
        dst, src->a.field, src->a.precision, src->symmetry,
        src->num_rows, src->num_columns, src->size,
        sizeof(*src->rowidx), 0, src->rowidx, src->colidx);
}

/**
 * ‘mtxmatrix_coordinate_init_copy()’ allocates a copy of a matrix and
 * also copies the values.
 */
int mtxmatrix_coordinate_init_copy(
    struct mtxmatrix_coordinate * dst,
    const struct mtxmatrix_coordinate * src)
{
    int err = mtxmatrix_coordinate_alloc_copy(dst, src);
    if (err) return err;
    return mtxmatrix_coordinate_copy(dst, src);
}

/*
 * initialise matrices from entrywise data in coordinate format
 */

/**
 * ‘mtxmatrix_coordinate_alloc_entries()’ allocates a matrix from
 * entrywise data in coordinate format.
 */
int mtxmatrix_coordinate_alloc_entries(
    struct mtxmatrix_coordinate * A,
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
    A->symmetry = symmetry;
    A->num_rows = num_rows;
    A->num_columns = num_columns;
    if (__builtin_mul_overflow(num_rows, num_columns, &A->num_entries)) {
        errno = EOVERFLOW;
        return MTX_ERR_ERRNO;
    }
    A->num_nonzeros = 0;
    A->size = size;
    A->rowidx = malloc(size * sizeof(int));
    if (!A->rowidx) return MTX_ERR_ERRNO;
    A->colidx = malloc(size * sizeof(int));
    if (!A->colidx) { free(A->rowidx); return MTX_ERR_ERRNO; }
    for (int64_t k = 0; k < size; k++) {
        A->rowidx[k] = *(const int *)((const char *) rowidx+k*idxstride)-idxbase;
        A->colidx[k] = *(const int *)((const char *) colidx+k*idxstride)-idxbase;
        A->num_nonzeros +=
            (symmetry == mtx_unsymmetric || A->rowidx[k] == A->colidx[k]) ? 1 : 2;
    }
    int err = mtxvector_base_alloc(&A->a, field, precision, size);
    if (err) { free(A->colidx); free(A->rowidx); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_coordinate_init_entries_real_single()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * real, single precision coefficients.
 */
int mtxmatrix_coordinate_init_entries_real_single(
    struct mtxmatrix_coordinate * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx,
    const float * data)
{
    int err = mtxmatrix_coordinate_alloc_entries(
        A, mtx_field_real, mtx_single, symmetry, num_rows, num_columns,
        size, sizeof(*rowidx), 0, rowidx, colidx);
    if (err) return err;
    err = mtxvector_base_set_real_single(&A->a, size, sizeof(*data), data);
    if (err) { mtxmatrix_coordinate_free(A); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_coordinate_init_entries_real_double()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * real, double precision coefficients.
 */
int mtxmatrix_coordinate_init_entries_real_double(
    struct mtxmatrix_coordinate * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx,
    const double * data)
{
    int err = mtxmatrix_coordinate_alloc_entries(
        A, mtx_field_real, mtx_double, symmetry, num_rows, num_columns,
        size, sizeof(*rowidx), 0, rowidx, colidx);
    if (err) return err;
    err = mtxvector_base_set_real_double(&A->a, size, sizeof(*data), data);
    if (err) { mtxmatrix_coordinate_free(A); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_coordinate_init_entries_complex_single()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * complex, single precision coefficients.
 */
int mtxmatrix_coordinate_init_entries_complex_single(
    struct mtxmatrix_coordinate * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx,
    const float (* data)[2])
{
    int err = mtxmatrix_coordinate_alloc_entries(
        A, mtx_field_complex, mtx_single, symmetry, num_rows, num_columns,
        size, sizeof(*rowidx), 0, rowidx, colidx);
    if (err) return err;
    err = mtxvector_base_set_complex_single(&A->a, size, sizeof(*data), data);
    if (err) { mtxmatrix_coordinate_free(A); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_coordinate_init_entries_complex_double()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * complex, double precision coefficients.
 */
int mtxmatrix_coordinate_init_entries_complex_double(
    struct mtxmatrix_coordinate * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx,
    const double (* data)[2])
{
    int err = mtxmatrix_coordinate_alloc_entries(
        A, mtx_field_complex, mtx_double, symmetry, num_rows, num_columns,
        size, sizeof(*rowidx), 0, rowidx, colidx);
    if (err) return err;
    err = mtxvector_base_set_complex_double(&A->a, size, sizeof(*data), data);
    if (err) { mtxmatrix_coordinate_free(A); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_coordinate_init_entries_integer_single()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * integer, single precision coefficients.
 */
int mtxmatrix_coordinate_init_entries_integer_single(
    struct mtxmatrix_coordinate * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx,
    const int32_t * data)
{
    int err = mtxmatrix_coordinate_alloc_entries(
        A, mtx_field_integer, mtx_single, symmetry, num_rows, num_columns,
        size, sizeof(*rowidx), 0, rowidx, colidx);
    if (err) return err;
    err = mtxvector_base_set_integer_single(&A->a, size, sizeof(*data), data);
    if (err) { mtxmatrix_coordinate_free(A); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_coordinate_init_entries_integer_double()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * integer, double precision coefficients.
 */
int mtxmatrix_coordinate_init_entries_integer_double(
    struct mtxmatrix_coordinate * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx,
    const int64_t * data)
{
    int err = mtxmatrix_coordinate_alloc_entries(
        A, mtx_field_integer, mtx_double, symmetry, num_rows, num_columns,
        size, sizeof(*rowidx), 0, rowidx, colidx);
    if (err) return err;
    err = mtxvector_base_set_integer_double(&A->a, size, sizeof(*data), data);
    if (err) { mtxmatrix_coordinate_free(A); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_coordinate_init_entries_pattern()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * boolean coefficients.
 */
int mtxmatrix_coordinate_init_entries_pattern(
    struct mtxmatrix_coordinate * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx)
{
    return mtxmatrix_coordinate_alloc_entries(
        A, mtx_field_pattern, mtx_single, symmetry, num_rows, num_columns,
        size, sizeof(*rowidx), 0, rowidx, colidx);
}

/*
 * initialise matrices from entrywise data in coordinate format with
 * specified strides
 */

/**
 * ‘mtxmatrix_coordinate_init_entries_strided_real_single()’ allocates
 * and initialises a matrix from entrywise data in coordinate format
 * with real, single precision coefficients.
 */
int mtxmatrix_coordinate_init_entries_strided_real_single(
    struct mtxmatrix_coordinate * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    int idxstride,
    int idxbase,
    const int * rowidx,
    const int * colidx,
    int datastride,
    const float * data)
{
    int err = mtxmatrix_coordinate_alloc_entries(
        A, mtx_field_real, mtx_single, symmetry, num_rows, num_columns,
        size, idxstride, idxbase, rowidx, colidx);
    if (err) return err;
    err = mtxvector_base_set_real_single(&A->a, size, datastride, data);
    if (err) { mtxmatrix_coordinate_free(A); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_coordinate_init_entries_strided_real_double()’ allocates
 * and initialises a matrix from entrywise data in coordinate format
 * with real, double precision coefficients.
 */
int mtxmatrix_coordinate_init_entries_strided_real_double(
    struct mtxmatrix_coordinate * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    int idxstride,
    int idxbase,
    const int * rowidx,
    const int * colidx,
    int datastride,
    const double * data)
{
    int err = mtxmatrix_coordinate_alloc_entries(
        A, mtx_field_real, mtx_double, symmetry, num_rows, num_columns,
        size, idxstride, idxbase, rowidx, colidx);
    if (err) return err;
    err = mtxvector_base_set_real_double(&A->a, size, datastride, data);
    if (err) { mtxmatrix_coordinate_free(A); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_coordinate_init_entries_strided_complex_single()’
 * allocates and initialises a matrix from entrywise data in
 * coordinate format with complex, single precision coefficients.
 */
int mtxmatrix_coordinate_init_entries_strided_complex_single(
    struct mtxmatrix_coordinate * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    int idxstride,
    int idxbase,
    const int * rowidx,
    const int * colidx,
    int datastride,
    const float (* data)[2])
{
    int err = mtxmatrix_coordinate_alloc_entries(
        A, mtx_field_complex, mtx_single, symmetry, num_rows, num_columns,
        size, idxstride, idxbase, rowidx, colidx);
    if (err) return err;
    err = mtxvector_base_set_complex_single(&A->a, size, datastride, data);
    if (err) { mtxmatrix_coordinate_free(A); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_coordinate_init_entries_strided_complex_double()’
 * allocates and initialises a matrix from entrywise data in
 * coordinate format with complex, double precision coefficients.
 */
int mtxmatrix_coordinate_init_entries_strided_complex_double(
    struct mtxmatrix_coordinate * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    int idxstride,
    int idxbase,
    const int * rowidx,
    const int * colidx,
    int datastride,
    const double (* data)[2])
{
    int err = mtxmatrix_coordinate_alloc_entries(
        A, mtx_field_complex, mtx_double, symmetry, num_rows, num_columns,
        size, idxstride, idxbase, rowidx, colidx);
    if (err) return err;
    err = mtxvector_base_set_complex_double(&A->a, size, datastride, data);
    if (err) { mtxmatrix_coordinate_free(A); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_coordinate_init_entries_strided_integer_single()’
 * allocates and initialises a matrix from entrywise data in
 * coordinate format with integer, single precision coefficients.
 */
int mtxmatrix_coordinate_init_entries_strided_integer_single(
    struct mtxmatrix_coordinate * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    int idxstride,
    int idxbase,
    const int * rowidx,
    const int * colidx,
    int datastride,
    const int32_t * data)
{
    int err = mtxmatrix_coordinate_alloc_entries(
        A, mtx_field_integer, mtx_single, symmetry, num_rows, num_columns,
        size, idxstride, idxbase, rowidx, colidx);
    if (err) return err;
    err = mtxvector_base_set_integer_single(&A->a, size, datastride, data);
    if (err) { mtxmatrix_coordinate_free(A); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_coordinate_init_entries_strided_integer_double()’
 * allocates and initialises a matrix from entrywise data in
 * coordinate format with integer, double precision coefficients.
 */
int mtxmatrix_coordinate_init_entries_strided_integer_double(
    struct mtxmatrix_coordinate * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    int idxstride,
    int idxbase,
    const int * rowidx,
    const int * colidx,
    int datastride,
    const int64_t * data)
{
    int err = mtxmatrix_coordinate_alloc_entries(
        A, mtx_field_integer, mtx_double, symmetry, num_rows, num_columns,
        size, idxstride, idxbase, rowidx, colidx);
    if (err) return err;
    err = mtxvector_base_set_integer_double(&A->a, size, datastride, data);
    if (err) { mtxmatrix_coordinate_free(A); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_coordinate_init_entries_strided_pattern()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * boolean coefficients.
 */
int mtxmatrix_coordinate_init_entries_strided_pattern(
    struct mtxmatrix_coordinate * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    int idxstride,
    int idxbase,
    const int * rowidx,
    const int * colidx)
{
    int err = mtxmatrix_coordinate_alloc_entries(
        A, mtx_field_pattern, mtx_single, symmetry, num_rows, num_columns,
        size, idxstride, idxbase, rowidx, colidx);
    if (err) return err;
    err = mtxvector_base_init_pattern(&A->a, size);
    if (err) { mtxmatrix_coordinate_free(A); return err; }
    return MTX_SUCCESS;
}

/*
 * initialise matrices from row-wise data in compressed row format
 */

/**
 * ‘mtxmatrix_coordinate_alloc_rows()’ allocates a matrix from
 * row-wise data in compressed row format.
 */
int mtxmatrix_coordinate_alloc_rows(
    struct mtxmatrix_coordinate * A,
    enum mtxfield field,
    enum mtxprecision precision,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx);

/**
 * ‘mtxmatrix_coordinate_init_rows_real_single()’ allocates and
 * initialises a matrix from row-wise data in compressed row format
 * with real, single precision coefficients.
 */
int mtxmatrix_coordinate_init_rows_real_single(
    struct mtxmatrix_coordinate * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const float * data);

/**
 * ‘mtxmatrix_coordinate_init_rows_real_double()’ allocates and
 * initialises a matrix from row-wise data in compressed row format
 * with real, double precision coefficients.
 */
int mtxmatrix_coordinate_init_rows_real_double(
    struct mtxmatrix_coordinate * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const double * data);

/**
 * ‘mtxmatrix_coordinate_init_rows_complex_single()’ allocates and
 * initialises a matrix from row-wise data in compressed row format
 * with complex, single precision coefficients.
 */
int mtxmatrix_coordinate_init_rows_complex_single(
    struct mtxmatrix_coordinate * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const float (* data)[2]);

/**
 * ‘mtxmatrix_coordinate_init_rows_complex_double()’ allocates and
 * initialises a matrix from row-wise data in compressed row format
 * with complex, double precision coefficients.
 */
int mtxmatrix_coordinate_init_rows_complex_double(
    struct mtxmatrix_coordinate * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const double (* data)[2]);

/**
 * ‘mtxmatrix_coordinate_init_rows_integer_single()’ allocates and
 * initialises a matrix from row-wise data in compressed row format
 * with integer, single precision coefficients.
 */
int mtxmatrix_coordinate_init_rows_integer_single(
    struct mtxmatrix_coordinate * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const int32_t * data);

/**
 * ‘mtxmatrix_coordinate_init_rows_integer_double()’ allocates and
 * initialises a matrix from row-wise data in compressed row format
 * with integer, double precision coefficients.
 */
int mtxmatrix_coordinate_init_rows_integer_double(
    struct mtxmatrix_coordinate * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const int64_t * data);

/**
 * ‘mtxmatrix_coordinate_init_rows_pattern()’ allocates and
 * initialises a matrix from row-wise data in compressed row format
 * with boolean coefficients.
 */
int mtxmatrix_coordinate_init_rows_pattern(
    struct mtxmatrix_coordinate * A,
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
 * ‘mtxmatrix_coordinate_alloc_columns()’ allocates a matrix from
 * column-wise data in compressed column format.
 */
int mtxmatrix_coordinate_alloc_columns(
    struct mtxmatrix_coordinate * A,
    enum mtxfield field,
    enum mtxprecision precision,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx);

/**
 * ‘mtxmatrix_coordinate_init_columns_real_single()’ allocates and
 * initialises a matrix from column-wise data in compressed column
 * format with real, single precision coefficients.
 */
int mtxmatrix_coordinate_init_columns_real_single(
    struct mtxmatrix_coordinate * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx,
    const float * data);

/**
 * ‘mtxmatrix_coordinate_init_columns_real_double()’ allocates and
 * initialises a matrix from column-wise data in compressed column
 * format with real, double precision coefficients.
 */
int mtxmatrix_coordinate_init_columns_real_double(
    struct mtxmatrix_coordinate * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx,
    const double * data);

/**
 * ‘mtxmatrix_coordinate_init_columns_complex_single()’ allocates and
 * initialises a matrix from column-wise data in compressed column
 * format with complex, single precision coefficients.
 */
int mtxmatrix_coordinate_init_columns_complex_single(
    struct mtxmatrix_coordinate * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx,
    const float (* data)[2]);

/**
 * ‘mtxmatrix_coordinate_init_columns_complex_double()’ allocates and
 * initialises a matrix from column-wise data in compressed column
 * format with complex, double precision coefficients.
 */
int mtxmatrix_coordinate_init_columns_complex_double(
    struct mtxmatrix_coordinate * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx,
    const double (* data)[2]);

/**
 * ‘mtxmatrix_coordinate_init_columns_integer_single()’ allocates and
 * initialises a matrix from column-wise data in compressed column
 * format with integer, single precision coefficients.
 */
int mtxmatrix_coordinate_init_columns_integer_single(
    struct mtxmatrix_coordinate * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx,
    const int32_t * data);

/**
 * ‘mtxmatrix_coordinate_init_columns_integer_double()’ allocates and
 * initialises a matrix from column-wise data in compressed column
 * format with integer, double precision coefficients.
 */
int mtxmatrix_coordinate_init_columns_integer_double(
    struct mtxmatrix_coordinate * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx,
    const int64_t * data);

/**
 * ‘mtxmatrix_coordinate_init_columns_pattern()’ allocates and
 * initialises a matrix from column-wise data in compressed column
 * format with boolean coefficients.
 */
int mtxmatrix_coordinate_init_columns_pattern(
    struct mtxmatrix_coordinate * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx);

/*
 * initialise matrices from a list of dense cliques
 */

/**
 * ‘mtxmatrix_coordinate_alloc_cliques()’ allocates a matrix from a
 * list of cliques.
 */
int mtxmatrix_coordinate_alloc_cliques(
    struct mtxmatrix_coordinate * A,
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
 * ‘mtxmatrix_coordinate_init_cliques_real_single()’ allocates and
 * initialises a matrix from a list of cliques with real, single
 * precision coefficients.
 */
int mtxmatrix_coordinate_init_cliques_real_single(
    struct mtxmatrix_coordinate * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_cliques,
    const int64_t * cliqueptr,
    const int * rowidx,
    const int * colidx,
    const float * data);

/**
 * ‘mtxmatrix_coordinate_init_cliques_real_double()’ allocates and
 * initialises a matrix from a list of cliques with real, double
 * precision coefficients.
 */
int mtxmatrix_coordinate_init_cliques_real_double(
    struct mtxmatrix_coordinate * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_cliques,
    const int64_t * cliqueptr,
    const int * rowidx,
    const int * colidx,
    const double * data);

/**
 * ‘mtxmatrix_coordinate_init_cliques_complex_single()’ allocates and
 * initialises a matrix from a list of cliques with complex, single
 * precision coefficients.
 */
int mtxmatrix_coordinate_init_cliques_complex_single(
    struct mtxmatrix_coordinate * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_cliques,
    const int64_t * cliqueptr,
    const int * rowidx,
    const int * colidx,
    const float (* data)[2]);

/**
 * ‘mtxmatrix_coordinate_init_cliques_complex_double()’ allocates and
 * initialises a matrix from a list of cliques with complex, double
 * precision coefficients.
 */
int mtxmatrix_coordinate_init_cliques_complex_double(
    struct mtxmatrix_coordinate * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_cliques,
    const int64_t * cliqueptr,
    const int * rowidx,
    const int * colidx,
    const double (* data)[2]);

/**
 * ‘mtxmatrix_coordinate_init_cliques_integer_single()’ allocates and
 * initialises a matrix from a list of cliques with integer, single
 * precision coefficients.
 */
int mtxmatrix_coordinate_init_cliques_integer_single(
    struct mtxmatrix_coordinate * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_cliques,
    const int64_t * cliqueptr,
    const int * rowidx,
    const int * colidx,
    const int32_t * data);

/**
 * ‘mtxmatrix_coordinate_init_cliques_integer_double()’ allocates and
 * initialises a matrix from a list of cliques with integer, double
 * precision coefficients.
 */
int mtxmatrix_coordinate_init_cliques_integer_double(
    struct mtxmatrix_coordinate * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_cliques,
    const int64_t * cliqueptr,
    const int * rowidx,
    const int * colidx,
    const int64_t * data);

/**
 * ‘mtxmatrix_coordinate_init_cliques_pattern()’ allocates and
 * initialises a matrix from a list of cliques with boolean
 * coefficients.
 */
int mtxmatrix_coordinate_init_cliques_pattern(
    struct mtxmatrix_coordinate * A,
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
 * ‘mtxmatrix_coordinate_setzero()’ sets every value of a matrix to
 * zero.
 */
int mtxmatrix_coordinate_setzero(
    struct mtxmatrix_coordinate * A)
{
    return mtxvector_base_setzero(&A->a);
}

/**
 * ‘mtxmatrix_coordinate_set_real_single()’ sets values of a matrix
 * based on an array of single precision floating point numbers.
 */
int mtxmatrix_coordinate_set_real_single(
    struct mtxmatrix_coordinate * A,
    int64_t size,
    int stride,
    const float * a)
{
    return mtxvector_base_set_real_single(&A->a, size, stride, a);
}

/**
 * ‘mtxmatrix_coordinate_set_real_double()’ sets values of a matrix
 * based on an array of double precision floating point numbers.
 */
int mtxmatrix_coordinate_set_real_double(
    struct mtxmatrix_coordinate * A,
    int64_t size,
    int stride,
    const double * a)
{
    return mtxvector_base_set_real_double(&A->a, size, stride, a);
}

/**
 * ‘mtxmatrix_coordinate_set_complex_single()’ sets values of a matrix
 * based on an array of single precision floating point complex
 * numbers.
 */
int mtxmatrix_coordinate_set_complex_single(
    struct mtxmatrix_coordinate * A,
    int64_t size,
    int stride,
    const float (*a)[2])
{
    return mtxvector_base_set_complex_single(&A->a, size, stride, a);
}

/**
 * ‘mtxmatrix_coordinate_set_complex_double()’ sets values of a matrix
 * based on an array of double precision floating point complex
 * numbers.
 */
int mtxmatrix_coordinate_set_complex_double(
    struct mtxmatrix_coordinate * A,
    int64_t size,
    int stride,
    const double (*a)[2])
{
    return mtxvector_base_set_complex_double(&A->a, size, stride, a);
}

/**
 * ‘mtxmatrix_coordinate_set_integer_single()’ sets values of a matrix
 * based on an array of integers.
 */
int mtxmatrix_coordinate_set_integer_single(
    struct mtxmatrix_coordinate * A,
    int64_t size,
    int stride,
    const int32_t * a)
{
    return mtxvector_base_set_integer_single(&A->a, size, stride, a);
}

/**
 * ‘mtxmatrix_coordinate_set_integer_double()’ sets values of a matrix
 * based on an array of integers.
 */
int mtxmatrix_coordinate_set_integer_double(
    struct mtxmatrix_coordinate * A,
    int64_t size,
    int stride,
    const int64_t * a)
{
    return mtxvector_base_set_integer_double(&A->a, size, stride, a);
}

/*
 * row and column vectors
 */

/**
 * ‘mtxmatrix_coordinate_alloc_row_vector()’ allocates a row vector
 * for a given matrix, where a row vector is a vector whose length
 * equal to a single row of the matrix.
 */
int mtxmatrix_coordinate_alloc_row_vector(
    const struct mtxmatrix_coordinate * A,
    struct mtxvector * x,
    enum mtxvectortype vectortype)
{
    return mtxvector_alloc(
        x, vectortype, A->a.field, A->a.precision, A->num_columns);
}

/**
 * ‘mtxmatrix_coordinate_alloc_column_vector()’ allocates a column
 * vector for a given matrix, where a column vector is a vector whose
 * length equal to a single column of the matrix.
 */
int mtxmatrix_coordinate_alloc_column_vector(
    const struct mtxmatrix_coordinate * A,
    struct mtxvector * y,
    enum mtxvectortype vectortype)
{
    return mtxvector_alloc(
        y, vectortype, A->a.field, A->a.precision, A->num_rows);
}

/*
 * convert to and from Matrix Market format
 */

/**
 * ‘mtxmatrix_coordinate_from_mtxfile()’ converts a matrix from Matrix
 * Market format.
 */
int mtxmatrix_coordinate_from_mtxfile(
    struct mtxmatrix_coordinate * A,
    const struct mtxfile * mtxfile)
{
    int err;
    if (mtxfile->header.object != mtxfile_matrix)
        return MTX_ERR_INCOMPATIBLE_MTX_OBJECT;

    /* TODO: If needed, we could convert from array to coordinate. */
    if (mtxfile->header.format != mtxfile_coordinate)
        return MTX_ERR_INCOMPATIBLE_MTX_FORMAT;

    enum mtxfield field;
    err = mtxfilefield_to_mtxfield(&field, mtxfile->header.field);
    if (err) return err;
    enum mtxsymmetry symmetry;
    err = mtxfilesymmetry_to_mtxsymmetry(&symmetry, mtxfile->header.symmetry);
    if (err) return err;
    enum mtxprecision precision = mtxfile->precision;

    int num_rows = mtxfile->size.num_rows;
    int num_columns = mtxfile->size.num_columns;
    int64_t size = mtxfile->size.num_nonzeros;
    int64_t num_entries;
    if (__builtin_mul_overflow(num_rows, num_columns, &num_entries)) {
        errno = EOVERFLOW; return MTX_ERR_ERRNO;
    }
    A->rowidx = malloc(size * sizeof(int));
    if (!A->rowidx) return MTX_ERR_ERRNO;
    A->colidx = malloc(size * sizeof(int));
    if (!A->colidx) { free(A->rowidx); return MTX_ERR_ERRNO; }
    A->symmetry = symmetry;
    A->num_rows = num_rows;
    A->num_columns = num_columns;
    A->num_entries = num_entries;
    A->num_nonzeros = 0;
    A->size = size;
    err = mtxvector_base_alloc(&A->a, field, precision, size);
    if (err) { free(A->colidx); free(A->rowidx); return err; }

    if (mtxfile->header.field == mtxfile_real) {
        if (mtxfile->precision == mtx_single) {
            const struct mtxfile_matrix_coordinate_real_single * data =
                mtxfile->data.matrix_coordinate_real_single;
            for (int64_t k = 0; k < size; k++) {
                A->rowidx[k] = data[k].i-1;
                A->colidx[k] = data[k].j-1;
                A->a.data.real_single[k] = data[k].a;
                A->num_nonzeros +=
                    (symmetry == mtx_unsymmetric || data[k].i == data[k].j) ? 1 : 2;
            }
        } else if (mtxfile->precision == mtx_double) {
            const struct mtxfile_matrix_coordinate_real_double * data =
                mtxfile->data.matrix_coordinate_real_double;
            for (int64_t k = 0; k < size; k++) {
                A->rowidx[k] = data[k].i-1;
                A->colidx[k] = data[k].j-1;
                A->a.data.real_double[k] = data[k].a;
                A->num_nonzeros +=
                    (symmetry == mtx_unsymmetric || data[k].i == data[k].j) ? 1 : 2;
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (mtxfile->header.field == mtxfile_complex) {
        if (mtxfile->precision == mtx_single) {
            const struct mtxfile_matrix_coordinate_complex_single * data =
                mtxfile->data.matrix_coordinate_complex_single;
            for (int64_t k = 0; k < size; k++) {
                A->rowidx[k] = data[k].i-1;
                A->colidx[k] = data[k].j-1;
                A->a.data.complex_single[k][0] = data[k].a[0];
                A->a.data.complex_single[k][1] = data[k].a[1];
                A->num_nonzeros +=
                    (symmetry == mtx_unsymmetric || data[k].i == data[k].j) ? 1 : 2;
            }
        } else if (mtxfile->precision == mtx_double) {
            const struct mtxfile_matrix_coordinate_complex_double * data =
                mtxfile->data.matrix_coordinate_complex_double;
            for (int64_t k = 0; k < size; k++) {
                A->rowidx[k] = data[k].i-1;
                A->colidx[k] = data[k].j-1;
                A->a.data.complex_double[k][0] = data[k].a[0];
                A->a.data.complex_double[k][1] = data[k].a[1];
                A->num_nonzeros +=
                    (symmetry == mtx_unsymmetric || data[k].i == data[k].j) ? 1 : 2;
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (mtxfile->header.field == mtxfile_integer) {
        if (mtxfile->precision == mtx_single) {
            const struct mtxfile_matrix_coordinate_integer_single * data =
                mtxfile->data.matrix_coordinate_integer_single;
            for (int64_t k = 0; k < size; k++) {
                A->rowidx[k] = data[k].i-1;
                A->colidx[k] = data[k].j-1;
                A->a.data.integer_single[k] = data[k].a;
                A->num_nonzeros +=
                    (symmetry == mtx_unsymmetric || data[k].i == data[k].j) ? 1 : 2;
            }
        } else if (mtxfile->precision == mtx_double) {
            const struct mtxfile_matrix_coordinate_integer_double * data =
                mtxfile->data.matrix_coordinate_integer_double;
            for (int64_t k = 0; k < size; k++) {
                A->rowidx[k] = data[k].i-1;
                A->colidx[k] = data[k].j-1;
                A->a.data.integer_double[k] = data[k].a;
                A->num_nonzeros +=
                    (symmetry == mtx_unsymmetric || data[k].i == data[k].j) ? 1 : 2;
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (mtxfile->header.field == mtxfile_pattern) {
        const struct mtxfile_matrix_coordinate_pattern * data =
            mtxfile->data.matrix_coordinate_pattern;
        for (int64_t k = 0; k < size; k++) {
            A->rowidx[k] = data[k].i-1;
            A->colidx[k] = data[k].j-1;
            A->num_nonzeros +=
                (symmetry == mtx_unsymmetric || data[k].i == data[k].j) ? 1 : 2;
        }
    } else { return MTX_ERR_INVALID_MTX_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_coordinate_to_mtxfile()’ converts a matrix to Matrix
 * Market format.
 */
int mtxmatrix_coordinate_to_mtxfile(
    struct mtxfile * mtxfile,
    const struct mtxmatrix_coordinate * A,
    int64_t num_rows,
    const int64_t * rowidx,
    int64_t num_columns,
    const int64_t * colidx,
    enum mtxfileformat mtxfmt)
{
    int err;
    if (mtxfmt != mtxfile_coordinate)
        return MTX_ERR_INCOMPATIBLE_MTX_FORMAT;

    enum mtxfilesymmetry symmetry;
    err = mtxfilesymmetry_from_mtxsymmetry(&symmetry, A->symmetry);
    if (err) return err;

    if (A->a.field == mtx_field_real) {
        err = mtxfile_alloc_matrix_coordinate(
            mtxfile, mtxfile_real, symmetry, A->a.precision,
            rowidx ? num_rows : A->num_rows,
            colidx ? num_columns : A->num_columns,
            A->size);
        if (err) return err;
        if (A->a.precision == mtx_single) {
            struct mtxfile_matrix_coordinate_real_single * data =
                mtxfile->data.matrix_coordinate_real_single;
            for (int64_t k = 0; k < A->size; k++) {
                data[k].i = rowidx ? rowidx[A->rowidx[k]]+1 : A->rowidx[k]+1;
                data[k].j = colidx ? colidx[A->colidx[k]]+1 : A->colidx[k]+1;
                data[k].a = A->a.data.real_single[k];
            }
        } else if (A->a.precision == mtx_double) {
            struct mtxfile_matrix_coordinate_real_double * data =
                mtxfile->data.matrix_coordinate_real_double;
            for (int64_t k = 0; k < A->size; k++) {
                data[k].i = rowidx ? rowidx[A->rowidx[k]]+1 : A->rowidx[k]+1;
                data[k].j = colidx ? colidx[A->colidx[k]]+1 : A->colidx[k]+1;
                data[k].a = A->a.data.real_double[k];
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (A->a.field == mtx_field_complex) {
        err = mtxfile_alloc_matrix_coordinate(
            mtxfile, mtxfile_complex, symmetry, A->a.precision,
            A->num_rows, A->num_columns, A->size);
        if (err) return err;
        if (A->a.precision == mtx_single) {
            struct mtxfile_matrix_coordinate_complex_single * data =
                mtxfile->data.matrix_coordinate_complex_single;
            for (int64_t k = 0; k < A->size; k++) {
                data[k].i = rowidx ? rowidx[A->rowidx[k]]+1 : A->rowidx[k]+1;
                data[k].j = colidx ? colidx[A->colidx[k]]+1 : A->colidx[k]+1;
                data[k].a[0] = A->a.data.complex_single[k][0];
                data[k].a[1] = A->a.data.complex_single[k][1];
            }
        } else if (A->a.precision == mtx_double) {
            struct mtxfile_matrix_coordinate_complex_double * data =
                mtxfile->data.matrix_coordinate_complex_double;
            for (int64_t k = 0; k < A->size; k++) {
                data[k].i = rowidx ? rowidx[A->rowidx[k]]+1 : A->rowidx[k]+1;
                data[k].j = colidx ? colidx[A->colidx[k]]+1 : A->colidx[k]+1;
                data[k].a[0] = A->a.data.complex_double[k][0];
                data[k].a[1] = A->a.data.complex_double[k][1];
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (A->a.field == mtx_field_integer) {
        err = mtxfile_alloc_matrix_coordinate(
            mtxfile, mtxfile_integer, symmetry, A->a.precision,
            A->num_rows, A->num_columns, A->size);
        if (err) return err;
        if (A->a.precision == mtx_single) {
            struct mtxfile_matrix_coordinate_integer_single * data =
                mtxfile->data.matrix_coordinate_integer_single;
            for (int64_t k = 0; k < A->size; k++) {
                data[k].i = rowidx ? rowidx[A->rowidx[k]]+1 : A->rowidx[k]+1;
                data[k].j = colidx ? colidx[A->colidx[k]]+1 : A->colidx[k]+1;
                data[k].a = A->a.data.integer_single[k];
            }
        } else if (A->a.precision == mtx_double) {
            struct mtxfile_matrix_coordinate_integer_double * data =
                mtxfile->data.matrix_coordinate_integer_double;
            for (int64_t k = 0; k < A->size; k++) {
                data[k].i = rowidx ? rowidx[A->rowidx[k]]+1 : A->rowidx[k]+1;
                data[k].j = colidx ? colidx[A->colidx[k]]+1 : A->colidx[k]+1;
                data[k].a = A->a.data.integer_double[k];
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (A->a.field == mtx_field_pattern) {
        err = mtxfile_alloc_matrix_coordinate(
            mtxfile, mtxfile_pattern, symmetry, mtx_single,
            A->num_rows, A->num_columns, A->size);
        if (err) return err;
        struct mtxfile_matrix_coordinate_pattern * data =
            mtxfile->data.matrix_coordinate_pattern;
        for (int64_t k = 0; k < A->size; k++) {
            data[k].i = rowidx ? rowidx[A->rowidx[k]]+1 : A->rowidx[k]+1;
            data[k].j = colidx ? colidx[A->colidx[k]]+1 : A->colidx[k]+1;
        }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/*
 * Level 1 BLAS operations
 */

/**
 * ‘mtxmatrix_coordinate_swap()’ swaps values of two matrices,
 * simultaneously performing ‘y <- x’ and ‘x <- y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_coordinate_swap(
    struct mtxmatrix_coordinate * x,
    struct mtxmatrix_coordinate * y)
{
    return mtxvector_base_swap(&x->a, &y->a);
}

/**
 * ‘mtxmatrix_coordinate_copy()’ copies values of a matrix, ‘y = x’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_coordinate_copy(
    struct mtxmatrix_coordinate * y,
    const struct mtxmatrix_coordinate * x)
{
    return mtxvector_base_copy(&y->a, &x->a);
}

/**
 * ‘mtxmatrix_coordinate_sscal()’ scales a matrix by a single
 * precision floating point scalar, ‘x = a*x’.
 */
int mtxmatrix_coordinate_sscal(
    float a,
    struct mtxmatrix_coordinate * x,
    int64_t * num_flops)
{
    return mtxvector_base_sscal(a, &x->a, num_flops);
}

/**
 * ‘mtxmatrix_coordinate_dscal()’ scales a matrix by a double
 * precision floating point scalar, ‘x = a*x’.
 */
int mtxmatrix_coordinate_dscal(
    double a,
    struct mtxmatrix_coordinate * x,
    int64_t * num_flops)
{
    return mtxvector_base_dscal(a, &x->a, num_flops);
}

/**
 * ‘mtxmatrix_coordinate_cscal()’ scales a matrix by a complex, single
 * precision floating point scalar, ‘x = (a+b*i)*x’.
 */
int mtxmatrix_coordinate_cscal(
    float a[2],
    struct mtxmatrix_coordinate * x,
    int64_t * num_flops)
{
    return mtxvector_base_cscal(a, &x->a, num_flops);
}

/**
 * ‘mtxmatrix_coordinate_zscal()’ scales a matrix by a complex, double
 * precision floating point scalar, ‘x = (a+b*i)*x’.
 */
int mtxmatrix_coordinate_zscal(
    double a[2],
    struct mtxmatrix_coordinate * x,
    int64_t * num_flops)
{
    return mtxvector_base_zscal(a, &x->a, num_flops);
}

/**
 * ‘mtxmatrix_coordinate_saxpy()’ adds a matrix to another one
 * multiplied by a single precision floating point value, ‘y = a*x +
 * y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_coordinate_saxpy(
    float a,
    const struct mtxmatrix_coordinate * x,
    struct mtxmatrix_coordinate * y,
    int64_t * num_flops)
{
    return mtxvector_base_saxpy(a, &x->a, &y->a, num_flops);
}

/**
 * ‘mtxmatrix_coordinate_daxpy()’ adds a matrix to another one
 * multiplied by a double precision floating point value, ‘y = a*x +
 * y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_coordinate_daxpy(
    double a,
    const struct mtxmatrix_coordinate * x,
    struct mtxmatrix_coordinate * y,
    int64_t * num_flops)
{
    return mtxvector_base_daxpy(a, &x->a, &y->a, num_flops);
}

/**
 * ‘mtxmatrix_coordinate_saypx()’ multiplies a matrix by a single
 * precision floating point scalar and adds another matrix, ‘y = a*y +
 * x’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_coordinate_saypx(
    float a,
    struct mtxmatrix_coordinate * y,
    const struct mtxmatrix_coordinate * x,
    int64_t * num_flops)
{
    return mtxvector_base_saypx(a, &y->a, &x->a, num_flops);
}

/**
 * ‘mtxmatrix_coordinate_daypx()’ multiplies a matrix by a double
 * precision floating point scalar and adds another matrix, ‘y = a*y +
 * x’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_coordinate_daypx(
    double a,
    struct mtxmatrix_coordinate * y,
    const struct mtxmatrix_coordinate * x,
    int64_t * num_flops)
{
    return mtxvector_base_daypx(a, &y->a, &x->a, num_flops);
}

/**
 * ‘mtxmatrix_coordinate_sdot()’ computes the Frobenius inner product
 * of two matrices in single precision floating point.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_coordinate_sdot(
    const struct mtxmatrix_coordinate * x,
    const struct mtxmatrix_coordinate * y,
    float * dot,
    int64_t * num_flops)
{
    if (x->symmetry == mtx_unsymmetric && y->symmetry == mtx_unsymmetric) {
        return mtxvector_base_sdot(&x->a, &y->a, dot, num_flops);
    } else { return MTX_ERR_INVALID_SYMMETRY; }
}

/**
 * ‘mtxmatrix_coordinate_ddot()’ computes the Frobenius inner product
 * of two matrices in double precision floating point.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_coordinate_ddot(
    const struct mtxmatrix_coordinate * x,
    const struct mtxmatrix_coordinate * y,
    double * dot,
    int64_t * num_flops)
{
    if (x->symmetry == mtx_unsymmetric && y->symmetry == mtx_unsymmetric) {
        return mtxvector_base_ddot(&x->a, &y->a, dot, num_flops);
    } else { return MTX_ERR_INVALID_SYMMETRY; }
}

/**
 * ‘mtxmatrix_coordinate_cdotu()’ computes the product of the
 * transpose of a complex row matrix with another complex row matrix
 * in single precision floating point, ‘dot := x^T*y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_coordinate_cdotu(
    const struct mtxmatrix_coordinate * x,
    const struct mtxmatrix_coordinate * y,
    float (* dot)[2],
    int64_t * num_flops)
{
    if (x->symmetry == mtx_unsymmetric && y->symmetry == mtx_unsymmetric) {
        return mtxvector_base_cdotu(&x->a, &y->a, dot, num_flops);
    } else { return MTX_ERR_INVALID_SYMMETRY; }
}

/**
 * ‘mtxmatrix_coordinate_zdotu()’ computes the product of the
 * transpose of a complex row matrix with another complex row matrix
 * in double precision floating point, ‘dot := x^T*y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_coordinate_zdotu(
    const struct mtxmatrix_coordinate * x,
    const struct mtxmatrix_coordinate * y,
    double (* dot)[2],
    int64_t * num_flops)
{
    if (x->symmetry == mtx_unsymmetric && y->symmetry == mtx_unsymmetric) {
        return mtxvector_base_zdotu(&x->a, &y->a, dot, num_flops);
    } else { return MTX_ERR_INVALID_SYMMETRY; }
}

/**
 * ‘mtxmatrix_coordinate_cdotc()’ computes the Frobenius inner product
 * of two complex matrices in single precision floating point, ‘dot :=
 * x^H*y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_coordinate_cdotc(
    const struct mtxmatrix_coordinate * x,
    const struct mtxmatrix_coordinate * y,
    float (* dot)[2],
    int64_t * num_flops)
{
    if (x->symmetry == mtx_unsymmetric && y->symmetry == mtx_unsymmetric) {
        return mtxvector_base_cdotc(&x->a, &y->a, dot, num_flops);
    } else { return MTX_ERR_INVALID_SYMMETRY; }
}

/**
 * ‘mtxmatrix_coordinate_zdotc()’ computes the Frobenius inner product
 * of two complex matrices in double precision floating point, ‘dot :=
 * x^H*y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_coordinate_zdotc(
    const struct mtxmatrix_coordinate * x,
    const struct mtxmatrix_coordinate * y,
    double (* dot)[2],
    int64_t * num_flops)
{
    if (x->symmetry == mtx_unsymmetric && y->symmetry == mtx_unsymmetric) {
        return mtxvector_base_zdotc(&x->a, &y->a, dot, num_flops);
    } else { return MTX_ERR_INVALID_SYMMETRY; }
}

/**
 * ‘mtxmatrix_coordinate_snrm2()’ computes the Frobenius norm of a
 * matrix in single precision floating point.
 */
int mtxmatrix_coordinate_snrm2(
    const struct mtxmatrix_coordinate * x,
    float * nrm2,
    int64_t * num_flops)
{
    if (x->symmetry == mtx_unsymmetric) {
        return mtxvector_base_snrm2(&x->a, nrm2, num_flops);
    } else { return MTX_ERR_INVALID_SYMMETRY; }
}

/**
 * ‘mtxmatrix_coordinate_dnrm2()’ computes the Frobenius norm of a
 * matrix in double precision floating point.
 */
int mtxmatrix_coordinate_dnrm2(
    const struct mtxmatrix_coordinate * x,
    double * nrm2,
    int64_t * num_flops)
{
    if (x->symmetry == mtx_unsymmetric) {
        return mtxvector_base_dnrm2(&x->a, nrm2, num_flops);
    } else { return MTX_ERR_INVALID_SYMMETRY; }
}

/**
 * ‘mtxmatrix_coordinate_sasum()’ computes the sum of absolute values
 * (1-norm) of a matrix in single precision floating point.  If the
 * matrix is complex-valued, then the sum of the absolute values of
 * the real and imaginary parts is computed.
 */
int mtxmatrix_coordinate_sasum(
    const struct mtxmatrix_coordinate * x,
    float * asum,
    int64_t * num_flops)
{
    if (x->symmetry == mtx_unsymmetric) {
        return mtxvector_base_sasum(&x->a, asum, num_flops);
    } else { return MTX_ERR_INVALID_SYMMETRY; }
}

/**
 * ‘mtxmatrix_coordinate_dasum()’ computes the sum of absolute values
 * (1-norm) of a matrix in double precision floating point.  If the
 * matrix is complex-valued, then the sum of the absolute values of
 * the real and imaginary parts is computed.
 */
int mtxmatrix_coordinate_dasum(
    const struct mtxmatrix_coordinate * x,
    double * asum,
    int64_t * num_flops)
{
    if (x->symmetry == mtx_unsymmetric) {
        return mtxvector_base_dasum(&x->a, asum, num_flops);
    } else { return MTX_ERR_INVALID_SYMMETRY; }
}

/**
 * ‘mtxmatrix_coordinate_iamax()’ finds the index of the first element
 * having the maximum absolute value.  If the matrix is
 * complex-valued, then the index points to the first element having
 * the maximum sum of the absolute values of the real and imaginary
 * parts.
 */
int mtxmatrix_coordinate_iamax(
    const struct mtxmatrix_coordinate * x,
    int * iamax)
{
    return mtxvector_base_iamax(&x->a, iamax);
}

/*
 * Level 2 BLAS operations (matrix-vector)
 */

/**
 * ‘mtxmatrix_coordinate_sgemv()’ multiplies a matrix ‘A’ or its
 * transpose ‘A'’ by a real scalar ‘alpha’ (‘α’) and a vector ‘x’,
 * before adding the result to another vector ‘y’ multiplied by
 * another real scalar ‘beta’ (‘β’). That is, ‘y = α*A*x + β*y’ or ‘y
 * = α*A'*x + β*y’.
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
int mtxmatrix_coordinate_sgemv(
    enum mtxtransposition trans,
    float alpha,
    const struct mtxmatrix_coordinate * A,
    const struct mtxvector * x,
    float beta,
    struct mtxvector * y,
    int64_t * num_flops)
{
    int err;
    const struct mtxvector_base * a = &A->a;
    if (x->type != mtxvector_base || y->type != mtxvector_base)
        return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
    const struct mtxvector_base * x_ = &x->storage.base;
    struct mtxvector_base * y_ = &y->storage.base;
    if (x_->field != a->field || y_->field != a->field)
        return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x_->precision != a->precision || y_->precision != a->precision)
        return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (trans == mtx_notrans) {
        if (A->num_rows != y_->size || A->num_columns != x_->size)
            return MTX_ERR_INCOMPATIBLE_SIZE;
    } else if (trans == mtx_trans || trans == mtx_conjtrans) {
        if (A->num_columns != y_->size || A->num_rows != x_->size)
            return MTX_ERR_INCOMPATIBLE_SIZE;
    }

    if (beta != 1) {
        err = mtxvector_sscal(beta, y, num_flops);
        if (err) return err;
    }

    if (A->symmetry == mtx_unsymmetric) {
        const int * i = trans == mtx_notrans ? A->rowidx : A->colidx;
        const int * j = trans == mtx_notrans ? A->colidx : A->rowidx;
        if (a->field == mtx_field_real) {
            if (a->precision == mtx_single) {
                const float * Adata = a->data.real_single;
                const float * xdata = x_->data.real_single;
                float * ydata = y_->data.real_single;
                for (int64_t k = 0; k < A->size; k++)
                    ydata[i[k]] += alpha*Adata[k]*xdata[j[k]];
                if (num_flops) *num_flops += 3*A->num_nonzeros;
            } else if (a->precision == mtx_double) {
                const double * Adata = a->data.real_double;
                const double * xdata = x_->data.real_double;
                double * ydata = y_->data.real_double;
                for (int64_t k = 0; k < A->size; k++)
                    ydata[i[k]] += alpha*Adata[k]*xdata[j[k]];
                if (num_flops) *num_flops += 3*A->num_nonzeros;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (a->field == mtx_field_complex) {
            if (trans == mtx_notrans || trans == mtx_trans) {
                if (a->precision == mtx_single) {
                    const float (* Adata)[2] = a->data.complex_single;
                    const float (* xdata)[2] = x_->data.complex_single;
                    float (* ydata)[2] = y_->data.complex_single;
                    for (int64_t k = 0; k < A->size; k++) {
                        ydata[i[k]][0] += alpha*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                        ydata[i[k]][1] += alpha*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                    }
                    if (num_flops) *num_flops += 10*A->num_nonzeros;
                } else if (a->precision == mtx_double) {
                    const double (* Adata)[2] = a->data.complex_double;
                    const double (* xdata)[2] = x_->data.complex_double;
                    double (* ydata)[2] = y_->data.complex_double;
                    for (int64_t k = 0; k < A->size; k++) {
                        ydata[i[k]][0] += alpha*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                        ydata[i[k]][1] += alpha*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                    }
                    if (num_flops) *num_flops += 10*A->num_nonzeros;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (trans == mtx_conjtrans) {
                if (a->precision == mtx_single) {
                    const float (* Adata)[2] = a->data.complex_single;
                    const float (* xdata)[2] = x_->data.complex_single;
                    float (* ydata)[2] = y_->data.complex_single;
                    for (int64_t k = 0; k < A->size; k++) {
                        ydata[i[k]][0] += alpha*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]);
                        ydata[i[k]][1] += alpha*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]);
                    }
                    if (num_flops) *num_flops += 10*A->num_nonzeros;
                } else if (a->precision == mtx_double) {
                    const double (* Adata)[2] = a->data.complex_double;
                    const double (* xdata)[2] = x_->data.complex_double;
                    double (* ydata)[2] = y_->data.complex_double;
                    for (int64_t k = 0; k < A->size; k++) {
                        ydata[i[k]][0] += alpha*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]);
                        ydata[i[k]][1] += alpha*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]);
                    }
                    if (num_flops) *num_flops += 10*A->num_nonzeros;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else { return MTX_ERR_INVALID_TRANSPOSITION; }
        } else if (a->field == mtx_field_integer) {
            if (a->precision == mtx_single) {
                const int32_t * Adata = a->data.integer_single;
                const int32_t * xdata = x_->data.integer_single;
                int32_t * ydata = y_->data.integer_single;
                for (int64_t k = 0; k < A->size; k++)
                    ydata[i[k]] += alpha*Adata[k]*xdata[j[k]];
                if (num_flops) *num_flops += 3*A->num_nonzeros;
            } else if (a->precision == mtx_double) {
                const int64_t * Adata = a->data.integer_double;
                const int64_t * xdata = x_->data.integer_double;
                int64_t * ydata = y_->data.integer_double;
                for (int64_t k = 0; k < A->size; k++)
                    ydata[i[k]] += alpha*Adata[k]*xdata[j[k]];
                if (num_flops) *num_flops += 3*A->num_nonzeros;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else { return MTX_ERR_INVALID_FIELD; }
    } else if (A->num_rows == A->num_columns && A->symmetry == mtx_symmetric) {
        const int * i = A->rowidx;
        const int * j = A->colidx;
        if (a->field == mtx_field_real) {
            if (a->precision == mtx_single) {
                const float * Adata = a->data.real_single;
                const float * xdata = x_->data.real_single;
                float * ydata = y_->data.real_single;
                for (int64_t k = 0; k < A->size; k++) {
                    ydata[i[k]] += alpha*Adata[k]*xdata[j[k]];
                    if (i[k] != j[k]) ydata[j[k]] += alpha*Adata[k]*xdata[i[k]];
                }
                if (num_flops) *num_flops += 3*A->num_nonzeros;
            } else if (a->precision == mtx_double) {
                const double * Adata = a->data.real_double;
                const double * xdata = x_->data.real_double;
                double * ydata = y_->data.real_double;
                for (int64_t k = 0; k < A->size; k++) {
                    ydata[i[k]] += alpha*Adata[k]*xdata[j[k]];
                    if (i[k] != j[k]) ydata[j[k]] += alpha*Adata[k]*xdata[i[k]];
                }
                if (num_flops) *num_flops += 3*A->num_nonzeros;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (a->field == mtx_field_complex) {
            if (trans == mtx_notrans || trans == mtx_trans) {
                if (a->precision == mtx_single) {
                    const float (* Adata)[2] = a->data.complex_single;
                    const float (* xdata)[2] = x_->data.complex_single;
                    float (* ydata)[2] = y_->data.complex_single;
                    for (int k = 0; k < A->size; k++) {
                        ydata[i[k]][0] += alpha*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                        ydata[i[k]][1] += alpha*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                        if (i[k] != j[k]) {
                            ydata[j[k]][0] += alpha*(Adata[k][0]*xdata[i[k]][0]-Adata[k][1]*xdata[i[k]][1]);
                            ydata[j[k]][1] += alpha*(Adata[k][0]*xdata[i[k]][1]+Adata[k][1]*xdata[i[k]][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_nonzeros;
                } else if (a->precision == mtx_double) {
                    const double (* Adata)[2] = a->data.complex_double;
                    const double (* xdata)[2] = x_->data.complex_double;
                    double (* ydata)[2] = y_->data.complex_double;
                    for (int k = 0; k < A->size; k++) {
                        ydata[i[k]][0] += alpha*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                        ydata[i[k]][1] += alpha*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                        if (i[k] != j[k]) {
                            ydata[j[k]][0] += alpha*(Adata[k][0]*xdata[i[k]][0]-Adata[k][1]*xdata[i[k]][1]);
                            ydata[j[k]][1] += alpha*(Adata[k][0]*xdata[i[k]][1]+Adata[k][1]*xdata[i[k]][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_nonzeros;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (trans == mtx_conjtrans) {
                if (a->precision == mtx_single) {
                    const float (* Adata)[2] = a->data.complex_single;
                    const float (* xdata)[2] = x_->data.complex_single;
                    float (* ydata)[2] = y_->data.complex_single;
                    for (int k = 0; k < A->size; k++) {
                        ydata[i[k]][0] += alpha*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]);
                        ydata[i[k]][1] += alpha*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]);
                        if (i[k] != j[k]) {
                            ydata[j[k]][0] += alpha*(Adata[k][0]*xdata[i[k]][0]+Adata[k][1]*xdata[i[k]][1]);
                            ydata[j[k]][1] += alpha*(Adata[k][0]*xdata[i[k]][1]-Adata[k][1]*xdata[i[k]][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_nonzeros;
                } else if (a->precision == mtx_double) {
                    const double (* Adata)[2] = a->data.complex_double;
                    const double (* xdata)[2] = x_->data.complex_double;
                    double (* ydata)[2] = y_->data.complex_double;
                    for (int k = 0; k < A->size; k++) {
                        ydata[i[k]][0] += alpha*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]);
                        ydata[i[k]][1] += alpha*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]);
                        if (i[k] != j[k]) {
                            ydata[j[k]][0] += alpha*(Adata[k][0]*xdata[i[k]][0]+Adata[k][1]*xdata[i[k]][1]);
                            ydata[j[k]][1] += alpha*(Adata[k][0]*xdata[i[k]][1]-Adata[k][1]*xdata[i[k]][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_nonzeros;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else { return MTX_ERR_INVALID_TRANSPOSITION; }
        } else if (a->field == mtx_field_integer) {
            if (a->precision == mtx_single) {
                const int32_t * Adata = a->data.integer_single;
                const int32_t * xdata = x_->data.integer_single;
                int32_t * ydata = y_->data.integer_single;
                for (int k = 0; k < A->size; k++) {
                    ydata[i[k]] += alpha*Adata[k]*xdata[j[k]];
                    if (i[k] != j[k]) ydata[j[k]] += alpha*Adata[k]*xdata[i[k]];
                }
                if (num_flops) *num_flops += 3*A->num_nonzeros;
            } else if (a->precision == mtx_double) {
                const int64_t * Adata = a->data.integer_double;
                const int64_t * xdata = x_->data.integer_double;
                int64_t * ydata = y_->data.integer_double;
                for (int k = 0; k < A->size; k++) {
                    ydata[i[k]] += alpha*Adata[k]*xdata[j[k]];
                    if (i[k] != j[k]) ydata[j[k]] += alpha*Adata[k]*xdata[i[k]];
                }
                if (num_flops) *num_flops += 3*A->num_nonzeros;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else { return MTX_ERR_INVALID_FIELD; }
    } else if (A->num_rows == A->num_columns && A->symmetry == mtx_skew_symmetric) {
        /* TODO: allow skew-symmetric matrices */
        return MTX_ERR_INVALID_SYMMETRY;
    } else if (A->num_rows == A->num_columns && A->symmetry == mtx_hermitian) {
        const int * i = A->rowidx;
        const int * j = A->colidx;
        if (a->field == mtx_field_complex) {
            if (trans == mtx_notrans || trans == mtx_conjtrans) {
                if (a->precision == mtx_single) {
                    const float (* Adata)[2] = a->data.complex_single;
                    const float (* xdata)[2] = x_->data.complex_single;
                    float (* ydata)[2] = y_->data.complex_single;
                    for (int k = 0; k < A->size; k++) {
                        ydata[i[k]][0] += alpha*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                        ydata[i[k]][1] += alpha*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                        if (i[k] != j[k]) {
                            ydata[j[k]][0] += alpha*(Adata[k][0]*xdata[i[k]][0]+Adata[k][1]*xdata[i[k]][1]);
                            ydata[j[k]][1] += alpha*(Adata[k][0]*xdata[i[k]][1]-Adata[k][1]*xdata[i[k]][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_nonzeros;
                } else if (a->precision == mtx_double) {
                    const double (* Adata)[2] = a->data.complex_double;
                    const double (* xdata)[2] = x_->data.complex_double;
                    double (* ydata)[2] = y_->data.complex_double;
                    for (int k = 0; k < A->size; k++) {
                        ydata[i[k]][0] += alpha*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                        ydata[i[k]][1] += alpha*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                        if (i[k] != j[k]) {
                            ydata[j[k]][0] += alpha*(Adata[k][0]*xdata[i[k]][0]+Adata[k][1]*xdata[i[k]][1]);
                            ydata[j[k]][1] += alpha*(Adata[k][0]*xdata[i[k]][1]-Adata[k][1]*xdata[i[k]][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_nonzeros;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (trans == mtx_trans) {
                if (a->precision == mtx_single) {
                    const float (* Adata)[2] = a->data.complex_single;
                    const float (* xdata)[2] = x_->data.complex_single;
                    float (* ydata)[2] = y_->data.complex_single;
                    for (int k = 0; k < A->size; k++) {
                        ydata[i[k]][0] += alpha*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]);
                        ydata[i[k]][1] += alpha*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]);
                        if (i[k] != j[k]) {
                            ydata[j[k]][0] += alpha*(Adata[k][0]*xdata[i[k]][0]-Adata[k][1]*xdata[i[k]][1]);
                            ydata[j[k]][1] += alpha*(Adata[k][0]*xdata[i[k]][1]+Adata[k][1]*xdata[i[k]][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_nonzeros;
                } else if (a->precision == mtx_double) {
                    const double (* Adata)[2] = a->data.complex_double;
                    const double (* xdata)[2] = x_->data.complex_double;
                    double (* ydata)[2] = y_->data.complex_double;
                    for (int k = 0; k < A->size; k++) {
                        ydata[i[k]][0] += alpha*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]);
                        ydata[i[k]][1] += alpha*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]);
                        if (i[k] != j[k]) {
                            ydata[j[k]][0] += alpha*(Adata[k][0]*xdata[i[k]][0]-Adata[k][1]*xdata[i[k]][1]);
                            ydata[j[k]][1] += alpha*(Adata[k][0]*xdata[i[k]][1]+Adata[k][1]*xdata[i[k]][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_nonzeros;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else { return MTX_ERR_INVALID_TRANSPOSITION; }
        } else { return MTX_ERR_INVALID_FIELD; }
    } else { return MTX_ERR_INVALID_SYMMETRY; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_coordinate_dgemv()’ multiplies a matrix ‘A’ or its
 * transpose ‘A'’ by a real scalar ‘alpha’ (‘α’) and a vector ‘x’,
 * before adding the result to another vector ‘y’ multiplied by
 * another scalar real ‘beta’ (‘β’).  That is, ‘y = α*A*x + β*y’ or ‘y
 * = α*A'*x + β*y’.
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
int mtxmatrix_coordinate_dgemv(
    enum mtxtransposition trans,
    double alpha,
    const struct mtxmatrix_coordinate * A,
    const struct mtxvector * x,
    double beta,
    struct mtxvector * y,
    int64_t * num_flops)
{
    int err;
    const struct mtxvector_base * a = &A->a;
    if (x->type != mtxvector_base || y->type != mtxvector_base)
        return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
    const struct mtxvector_base * x_ = &x->storage.base;
    struct mtxvector_base * y_ = &y->storage.base;
    if (x_->field != a->field || y_->field != a->field)
        return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x_->precision != a->precision || y_->precision != a->precision)
        return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (trans == mtx_notrans) {
        if (A->num_rows != y_->size || A->num_columns != x_->size)
            return MTX_ERR_INCOMPATIBLE_SIZE;
    } else if (trans == mtx_trans || trans == mtx_conjtrans) {
        if (A->num_columns != y_->size || A->num_rows != x_->size)
            return MTX_ERR_INCOMPATIBLE_SIZE;
    }

    if (beta != 1) {
        err = mtxvector_dscal(beta, y, num_flops);
        if (err) return err;
    }

    if (A->symmetry == mtx_unsymmetric) {
        const int * i = trans == mtx_notrans ? A->rowidx : A->colidx;
        const int * j = trans == mtx_notrans ? A->colidx : A->rowidx;
        if (a->field == mtx_field_real) {
            if (a->precision == mtx_single) {
                const float * Adata = a->data.real_single;
                const float * xdata = x_->data.real_single;
                float * ydata = y_->data.real_single;
                for (int64_t k = 0; k < A->size; k++)
                    ydata[i[k]] += alpha*Adata[k]*xdata[j[k]];
                if (num_flops) *num_flops += 3*A->num_nonzeros;
            } else if (a->precision == mtx_double) {
                const double * Adata = a->data.real_double;
                const double * xdata = x_->data.real_double;
                double * ydata = y_->data.real_double;
                for (int64_t k = 0; k < A->size; k++)
                    ydata[i[k]] += alpha*Adata[k]*xdata[j[k]];
                if (num_flops) *num_flops += 3*A->num_nonzeros;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (a->field == mtx_field_complex) {
            if (trans == mtx_notrans || trans == mtx_trans) {
                if (a->precision == mtx_single) {
                    const float (* Adata)[2] = a->data.complex_single;
                    const float (* xdata)[2] = x_->data.complex_single;
                    float (* ydata)[2] = y_->data.complex_single;
                    for (int64_t k = 0; k < A->size; k++) {
                        ydata[i[k]][0] += alpha*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                        ydata[i[k]][1] += alpha*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                    }
                    if (num_flops) *num_flops += 10*A->num_nonzeros;
                } else if (a->precision == mtx_double) {
                    const double (* Adata)[2] = a->data.complex_double;
                    const double (* xdata)[2] = x_->data.complex_double;
                    double (* ydata)[2] = y_->data.complex_double;
                    for (int64_t k = 0; k < A->size; k++) {
                        ydata[i[k]][0] += alpha*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                        ydata[i[k]][1] += alpha*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                    }
                    if (num_flops) *num_flops += 10*A->num_nonzeros;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (trans == mtx_conjtrans) {
                if (a->precision == mtx_single) {
                    const float (* Adata)[2] = a->data.complex_single;
                    const float (* xdata)[2] = x_->data.complex_single;
                    float (* ydata)[2] = y_->data.complex_single;
                    for (int64_t k = 0; k < A->size; k++) {
                        ydata[i[k]][0] += alpha*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]);
                        ydata[i[k]][1] += alpha*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]);
                    }
                    if (num_flops) *num_flops += 10*A->num_nonzeros;
                } else if (a->precision == mtx_double) {
                    const double (* Adata)[2] = a->data.complex_double;
                    const double (* xdata)[2] = x_->data.complex_double;
                    double (* ydata)[2] = y_->data.complex_double;
                    for (int64_t k = 0; k < A->size; k++) {
                        ydata[i[k]][0] += alpha*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]);
                        ydata[i[k]][1] += alpha*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]);
                    }
                    if (num_flops) *num_flops += 10*A->num_nonzeros;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else { return MTX_ERR_INVALID_TRANSPOSITION; }
        } else if (a->field == mtx_field_integer) {
            if (a->precision == mtx_single) {
                const int32_t * Adata = a->data.integer_single;
                const int32_t * xdata = x_->data.integer_single;
                int32_t * ydata = y_->data.integer_single;
                for (int64_t k = 0; k < A->size; k++)
                    ydata[i[k]] += alpha*Adata[k]*xdata[j[k]];
                if (num_flops) *num_flops += 3*A->num_nonzeros;
            } else if (a->precision == mtx_double) {
                const int64_t * Adata = a->data.integer_double;
                const int64_t * xdata = x_->data.integer_double;
                int64_t * ydata = y_->data.integer_double;
                for (int64_t k = 0; k < A->size; k++)
                    ydata[i[k]] += alpha*Adata[k]*xdata[j[k]];
                if (num_flops) *num_flops += 3*A->num_nonzeros;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else { return MTX_ERR_INVALID_FIELD; }
    } else if (A->num_rows == A->num_columns && A->symmetry == mtx_symmetric) {
        const int * i = A->rowidx;
        const int * j = A->colidx;
        if (a->field == mtx_field_real) {
            if (a->precision == mtx_single) {
                const float * Adata = a->data.real_single;
                const float * xdata = x_->data.real_single;
                float * ydata = y_->data.real_single;
                for (int64_t k = 0; k < A->size; k++) {
                    ydata[i[k]] += alpha*Adata[k]*xdata[j[k]];
                    if (i[k] != j[k]) ydata[j[k]] += alpha*Adata[k]*xdata[i[k]];
                }
                if (num_flops) *num_flops += 3*A->num_nonzeros;
            } else if (a->precision == mtx_double) {
                const double * Adata = a->data.real_double;
                const double * xdata = x_->data.real_double;
                double * ydata = y_->data.real_double;
                for (int64_t k = 0; k < A->size; k++) {
                    ydata[i[k]] += alpha*Adata[k]*xdata[j[k]];
                    if (i[k] != j[k]) ydata[j[k]] += alpha*Adata[k]*xdata[i[k]];
                }
                if (num_flops) *num_flops += 3*A->num_nonzeros;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (a->field == mtx_field_complex) {
            if (trans == mtx_notrans || trans == mtx_trans) {
                if (a->precision == mtx_single) {
                    const float (* Adata)[2] = a->data.complex_single;
                    const float (* xdata)[2] = x_->data.complex_single;
                    float (* ydata)[2] = y_->data.complex_single;
                    for (int k = 0; k < A->size; k++) {
                        ydata[i[k]][0] += alpha*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                        ydata[i[k]][1] += alpha*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                        if (i[k] != j[k]) {
                            ydata[j[k]][0] += alpha*(Adata[k][0]*xdata[i[k]][0]-Adata[k][1]*xdata[i[k]][1]);
                            ydata[j[k]][1] += alpha*(Adata[k][0]*xdata[i[k]][1]+Adata[k][1]*xdata[i[k]][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_nonzeros;
                } else if (a->precision == mtx_double) {
                    const double (* Adata)[2] = a->data.complex_double;
                    const double (* xdata)[2] = x_->data.complex_double;
                    double (* ydata)[2] = y_->data.complex_double;
                    for (int k = 0; k < A->size; k++) {
                        ydata[i[k]][0] += alpha*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                        ydata[i[k]][1] += alpha*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                        if (i[k] != j[k]) {
                            ydata[j[k]][0] += alpha*(Adata[k][0]*xdata[i[k]][0]-Adata[k][1]*xdata[i[k]][1]);
                            ydata[j[k]][1] += alpha*(Adata[k][0]*xdata[i[k]][1]+Adata[k][1]*xdata[i[k]][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_nonzeros;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (trans == mtx_conjtrans) {
                if (a->precision == mtx_single) {
                    const float (* Adata)[2] = a->data.complex_single;
                    const float (* xdata)[2] = x_->data.complex_single;
                    float (* ydata)[2] = y_->data.complex_single;
                    for (int k = 0; k < A->size; k++) {
                        ydata[i[k]][0] += alpha*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]);
                        ydata[i[k]][1] += alpha*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]);
                        if (i[k] != j[k]) {
                            ydata[j[k]][0] += alpha*(Adata[k][0]*xdata[i[k]][0]+Adata[k][1]*xdata[i[k]][1]);
                            ydata[j[k]][1] += alpha*(Adata[k][0]*xdata[i[k]][1]-Adata[k][1]*xdata[i[k]][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_nonzeros;
                } else if (a->precision == mtx_double) {
                    const double (* Adata)[2] = a->data.complex_double;
                    const double (* xdata)[2] = x_->data.complex_double;
                    double (* ydata)[2] = y_->data.complex_double;
                    for (int k = 0; k < A->size; k++) {
                        ydata[i[k]][0] += alpha*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]);
                        ydata[i[k]][1] += alpha*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]);
                        if (i[k] != j[k]) {
                            ydata[j[k]][0] += alpha*(Adata[k][0]*xdata[i[k]][0]+Adata[k][1]*xdata[i[k]][1]);
                            ydata[j[k]][1] += alpha*(Adata[k][0]*xdata[i[k]][1]-Adata[k][1]*xdata[i[k]][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_nonzeros;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else { return MTX_ERR_INVALID_TRANSPOSITION; }
        } else if (a->field == mtx_field_integer) {
            if (a->precision == mtx_single) {
                const int32_t * Adata = a->data.integer_single;
                const int32_t * xdata = x_->data.integer_single;
                int32_t * ydata = y_->data.integer_single;
                for (int k = 0; k < A->size; k++) {
                    ydata[i[k]] += alpha*Adata[k]*xdata[j[k]];
                    if (i[k] != j[k]) ydata[j[k]] += alpha*Adata[k]*xdata[i[k]];
                }
                if (num_flops) *num_flops += 3*A->num_nonzeros;
            } else if (a->precision == mtx_double) {
                const int64_t * Adata = a->data.integer_double;
                const int64_t * xdata = x_->data.integer_double;
                int64_t * ydata = y_->data.integer_double;
                for (int k = 0; k < A->size; k++) {
                    ydata[i[k]] += alpha*Adata[k]*xdata[j[k]];
                    if (i[k] != j[k]) ydata[j[k]] += alpha*Adata[k]*xdata[i[k]];
                }
                if (num_flops) *num_flops += 3*A->num_nonzeros;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else { return MTX_ERR_INVALID_FIELD; }
    } else if (A->num_rows == A->num_columns && A->symmetry == mtx_skew_symmetric) {
        /* TODO: allow skew-symmetric matrices */
        return MTX_ERR_INVALID_SYMMETRY;
    } else if (A->num_rows == A->num_columns && A->symmetry == mtx_hermitian) {
        const int * i = A->rowidx;
        const int * j = A->colidx;
        if (a->field == mtx_field_complex) {
            if (trans == mtx_notrans || trans == mtx_conjtrans) {
                if (a->precision == mtx_single) {
                    const float (* Adata)[2] = a->data.complex_single;
                    const float (* xdata)[2] = x_->data.complex_single;
                    float (* ydata)[2] = y_->data.complex_single;
                    for (int k = 0; k < A->size; k++) {
                        ydata[i[k]][0] += alpha*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                        ydata[i[k]][1] += alpha*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                        if (i[k] != j[k]) {
                            ydata[j[k]][0] += alpha*(Adata[k][0]*xdata[i[k]][0]+Adata[k][1]*xdata[i[k]][1]);
                            ydata[j[k]][1] += alpha*(Adata[k][0]*xdata[i[k]][1]-Adata[k][1]*xdata[i[k]][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_nonzeros;
                } else if (a->precision == mtx_double) {
                    const double (* Adata)[2] = a->data.complex_double;
                    const double (* xdata)[2] = x_->data.complex_double;
                    double (* ydata)[2] = y_->data.complex_double;
                    for (int k = 0; k < A->size; k++) {
                        ydata[i[k]][0] += alpha*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                        ydata[i[k]][1] += alpha*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                        if (i[k] != j[k]) {
                            ydata[j[k]][0] += alpha*(Adata[k][0]*xdata[i[k]][0]+Adata[k][1]*xdata[i[k]][1]);
                            ydata[j[k]][1] += alpha*(Adata[k][0]*xdata[i[k]][1]-Adata[k][1]*xdata[i[k]][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_nonzeros;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (trans == mtx_trans) {
                if (a->precision == mtx_single) {
                    const float (* Adata)[2] = a->data.complex_single;
                    const float (* xdata)[2] = x_->data.complex_single;
                    float (* ydata)[2] = y_->data.complex_single;
                    for (int k = 0; k < A->size; k++) {
                        ydata[i[k]][0] += alpha*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]);
                        ydata[i[k]][1] += alpha*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]);
                        if (i[k] != j[k]) {
                            ydata[j[k]][0] += alpha*(Adata[k][0]*xdata[i[k]][0]-Adata[k][1]*xdata[i[k]][1]);
                            ydata[j[k]][1] += alpha*(Adata[k][0]*xdata[i[k]][1]+Adata[k][1]*xdata[i[k]][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_nonzeros;
                } else if (a->precision == mtx_double) {
                    const double (* Adata)[2] = a->data.complex_double;
                    const double (* xdata)[2] = x_->data.complex_double;
                    double (* ydata)[2] = y_->data.complex_double;
                    for (int k = 0; k < A->size; k++) {
                        ydata[i[k]][0] += alpha*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]);
                        ydata[i[k]][1] += alpha*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]);
                        if (i[k] != j[k]) {
                            ydata[j[k]][0] += alpha*(Adata[k][0]*xdata[i[k]][0]-Adata[k][1]*xdata[i[k]][1]);
                            ydata[j[k]][1] += alpha*(Adata[k][0]*xdata[i[k]][1]+Adata[k][1]*xdata[i[k]][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_nonzeros;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else { return MTX_ERR_INVALID_TRANSPOSITION; }
        } else { return MTX_ERR_INVALID_FIELD; }
    } else { return MTX_ERR_INVALID_SYMMETRY; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_coordinate_cgemv()’ multiplies a complex-valued matrix
 * ‘A’, its transpose ‘A'’ or its conjugate transpose ‘Aᴴ’ by a
 * complex scalar ‘alpha’ (‘α’) and a vector ‘x’, before adding the
 * result to another vector ‘y’ multiplied by another complex scalar
 * ‘beta’ (‘β’).  That is, ‘y = α*A*x + β*y’, ‘y = α*A'*x + β*y’ or ‘y
 * = α*Aᴴ*x + β*y’.
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
 */
int mtxmatrix_coordinate_cgemv(
    enum mtxtransposition trans,
    float alpha[2],
    const struct mtxmatrix_coordinate * A,
    const struct mtxvector * x,
    float beta[2],
    struct mtxvector * y,
    int64_t * num_flops)
{
    int err;
    const struct mtxvector_base * a = &A->a;
    if (x->type != mtxvector_base || y->type != mtxvector_base)
        return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
    const struct mtxvector_base * x_ = &x->storage.base;
    struct mtxvector_base * y_ = &y->storage.base;
    if (x_->field != a->field || y_->field != a->field)
        return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x_->precision != a->precision || y_->precision != a->precision)
        return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (trans == mtx_notrans) {
        if (A->num_rows != y_->size || A->num_columns != x_->size)
            return MTX_ERR_INCOMPATIBLE_SIZE;
    } else if (trans == mtx_trans || trans == mtx_conjtrans) {
        if (A->num_columns != y_->size || A->num_rows != x_->size)
            return MTX_ERR_INCOMPATIBLE_SIZE;
    }
    if (a->field != mtx_field_complex)
        return MTX_ERR_INCOMPATIBLE_FIELD;

    if (beta[0] != 1 || beta[1] != 0) {
        err = mtxvector_cscal(beta, y, num_flops);
        if (err) return err;
    }

    const int * i = A->rowidx;
    const int * j = A->colidx;
    if (A->symmetry == mtx_unsymmetric) {
        if (trans == mtx_notrans) {
            if (a->precision == mtx_single) {
                const float (* Adata)[2] = a->data.complex_single;
                const float (* xdata)[2] = x_->data.complex_single;
                float (* ydata)[2] = y_->data.complex_single;
                for (int k = 0; k < A->size; k++) {
                    ydata[i[k]][0] += alpha[0]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1])-alpha[1]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                    ydata[i[k]][1] += alpha[0]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0])+alpha[1]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                }
                if (num_flops) *num_flops += 20*A->num_nonzeros;
            } else if (a->precision == mtx_double) {
                const double (* Adata)[2] = a->data.complex_double;
                const double (* xdata)[2] = x_->data.complex_double;
                double (* ydata)[2] = y_->data.complex_double;
                for (int k = 0; k < A->size; k++) {
                    ydata[i[k]][0] += alpha[0]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1])-alpha[1]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                    ydata[i[k]][1] += alpha[0]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0])+alpha[1]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                }
                if (num_flops) *num_flops += 20*A->num_nonzeros;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (trans == mtx_trans) {
            if (a->precision == mtx_single) {
                const float (* Adata)[2] = a->data.complex_single;
                const float (* xdata)[2] = x_->data.complex_single;
                float (* ydata)[2] = y_->data.complex_single;
                for (int k = 0; k < A->size; k++) {
                    ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i[k]][0]-Adata[k][1]*xdata[i[k]][1])-alpha[1]*(Adata[k][0]*xdata[i[k]][1]+Adata[k][1]*xdata[i[k]][0]);
                    ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i[k]][1]+Adata[k][1]*xdata[i[k]][0])+alpha[1]*(Adata[k][0]*xdata[i[k]][0]-Adata[k][1]*xdata[i[k]][1]);
                }
                if (num_flops) *num_flops += 20*A->num_nonzeros;
            } else if (a->precision == mtx_double) {
                const double (* Adata)[2] = a->data.complex_double;
                const double (* xdata)[2] = x_->data.complex_double;
                double (* ydata)[2] = y_->data.complex_double;
                for (int k = 0; k < A->size; k++) {
                    ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i[k]][0]-Adata[k][1]*xdata[i[k]][1])-alpha[1]*(Adata[k][0]*xdata[i[k]][1]+Adata[k][1]*xdata[i[k]][0]);
                    ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i[k]][1]+Adata[k][1]*xdata[i[k]][0])+alpha[1]*(Adata[k][0]*xdata[i[k]][0]-Adata[k][1]*xdata[i[k]][1]);
                }
                if (num_flops) *num_flops += 20*A->num_nonzeros;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (trans == mtx_conjtrans) {
            if (a->precision == mtx_single) {
                const float (* Adata)[2] = a->data.complex_single;
                const float (* xdata)[2] = x_->data.complex_single;
                float (* ydata)[2] = y_->data.complex_single;
                for (int k = 0; k < A->size; k++) {
                    ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i[k]][0]+Adata[k][1]*xdata[i[k]][1])-alpha[1]*(Adata[k][0]*xdata[i[k]][1]-Adata[k][1]*xdata[i[k]][0]);
                    ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i[k]][1]-Adata[k][1]*xdata[i[k]][0])+alpha[1]*(Adata[k][0]*xdata[i[k]][0]+Adata[k][1]*xdata[i[k]][1]);
                }
                if (num_flops) *num_flops += 20*A->num_nonzeros;
            } else if (a->precision == mtx_double) {
                const double (* Adata)[2] = a->data.complex_double;
                const double (* xdata)[2] = x_->data.complex_double;
                double (* ydata)[2] = y_->data.complex_double;
                for (int k = 0; k < A->size; k++) {
                    ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i[k]][0]+Adata[k][1]*xdata[i[k]][1])-alpha[1]*(Adata[k][0]*xdata[i[k]][1]-Adata[k][1]*xdata[i[k]][0]);
                    ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i[k]][1]-Adata[k][1]*xdata[i[k]][0])+alpha[1]*(Adata[k][0]*xdata[i[k]][0]+Adata[k][1]*xdata[i[k]][1]);
                }
                if (num_flops) *num_flops += 20*A->num_nonzeros;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else { return MTX_ERR_INVALID_TRANSPOSITION; }
    } else if (A->num_rows == A->num_columns && A->symmetry == mtx_symmetric) {
        if (trans == mtx_notrans || trans == mtx_trans) {
            if (a->precision == mtx_single) {
                const float (* Adata)[2] = a->data.complex_single;
                const float (* xdata)[2] = x_->data.complex_single;
                float (* ydata)[2] = y_->data.complex_single;
                for (int k = 0; k < A->size; k++) {
                    ydata[i[k]][0] += alpha[0]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]) - alpha[1]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                    ydata[i[k]][1] += alpha[0]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]) + alpha[1]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                    if (i[k] != j[k]) {
                        ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i[k]][0]-Adata[k][1]*xdata[i[k]][1]) - alpha[1]*(Adata[k][0]*xdata[i[k]][1]+Adata[k][1]*xdata[i[k]][0]);
                        ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i[k]][1]+Adata[k][1]*xdata[i[k]][0]) + alpha[1]*(Adata[k][0]*xdata[i[k]][0]-Adata[k][1]*xdata[i[k]][1]);
                    }
                }
                if (num_flops) *num_flops += 20*A->num_nonzeros;
            } else if (a->precision == mtx_double) {
                const double (* Adata)[2] = a->data.complex_double;
                const double (* xdata)[2] = x_->data.complex_double;
                double (* ydata)[2] = y_->data.complex_double;
                for (int k = 0; k < A->size; k++) {
                    ydata[i[k]][0] += alpha[0]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]) - alpha[1]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                    ydata[i[k]][1] += alpha[0]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]) + alpha[1]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                    if (i[k] != j[k]) {
                        ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i[k]][0]-Adata[k][1]*xdata[i[k]][1]) - alpha[1]*(Adata[k][0]*xdata[i[k]][1]+Adata[k][1]*xdata[i[k]][0]);
                        ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i[k]][1]+Adata[k][1]*xdata[i[k]][0]) + alpha[1]*(Adata[k][0]*xdata[i[k]][0]-Adata[k][1]*xdata[i[k]][1]);
                    }
                }
                if (num_flops) *num_flops += 20*A->num_nonzeros;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (trans == mtx_conjtrans) {
            if (a->precision == mtx_single) {
                const float (* Adata)[2] = a->data.complex_single;
                const float (* xdata)[2] = x_->data.complex_single;
                float (* ydata)[2] = y_->data.complex_single;
                for (int k = 0; k < A->size; k++) {
                    ydata[i[k]][0] += alpha[0]*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]) - alpha[1]*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]);
                    ydata[i[k]][1] += alpha[0]*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]) + alpha[1]*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]);
                    if (i[k] != j[k]) {
                        ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i[k]][0]+Adata[k][1]*xdata[i[k]][1]) - alpha[1]*(Adata[k][0]*xdata[i[k]][1]-Adata[k][1]*xdata[i[k]][0]);
                        ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i[k]][1]-Adata[k][1]*xdata[i[k]][0]) + alpha[1]*(Adata[k][0]*xdata[i[k]][0]+Adata[k][1]*xdata[i[k]][1]);
                    }
                }
                if (num_flops) *num_flops += 20*A->num_nonzeros;
            } else if (a->precision == mtx_double) {
                const double (* Adata)[2] = a->data.complex_double;
                const double (* xdata)[2] = x_->data.complex_double;
                double (* ydata)[2] = y_->data.complex_double;
                for (int k = 0; k < A->size; k++) {
                    ydata[i[k]][0] += alpha[0]*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]) - alpha[1]*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]);
                    ydata[i[k]][1] += alpha[0]*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]) + alpha[1]*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]);
                    if (i[k] != j[k]) {
                        ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i[k]][0]+Adata[k][1]*xdata[i[k]][1]) - alpha[1]*(Adata[k][0]*xdata[i[k]][1]-Adata[k][1]*xdata[i[k]][0]);
                        ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i[k]][1]-Adata[k][1]*xdata[i[k]][0]) + alpha[1]*(Adata[k][0]*xdata[i[k]][0]+Adata[k][1]*xdata[i[k]][1]);
                    }
                }
                if (num_flops) *num_flops += 20*A->num_nonzeros;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else { return MTX_ERR_INVALID_TRANSPOSITION; }
    } else if (A->num_rows == A->num_columns && A->symmetry == mtx_skew_symmetric) {
        /* TODO: allow skew-symmetric matrices */
        return MTX_ERR_INVALID_SYMMETRY;
    } else if (A->num_rows == A->num_columns && A->symmetry == mtx_hermitian) {
        if (trans == mtx_notrans || trans == mtx_conjtrans) {
            if (a->precision == mtx_single) {
                const float (* Adata)[2] = a->data.complex_single;
                const float (* xdata)[2] = x_->data.complex_single;
                float (* ydata)[2] = y_->data.complex_single;
                for (int k = 0; k < A->size; k++) {
                    ydata[i[k]][0] += alpha[0]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]) - alpha[1]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                    ydata[i[k]][1] += alpha[0]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]) + alpha[1]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                    if (i[k] != j[k]) {
                        ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i[k]][0]+Adata[k][1]*xdata[i[k]][1]) - alpha[1]*(Adata[k][0]*xdata[i[k]][1]-Adata[k][1]*xdata[i[k]][0]);
                        ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i[k]][1]-Adata[k][1]*xdata[i[k]][0]) + alpha[1]*(Adata[k][0]*xdata[i[k]][0]+Adata[k][1]*xdata[i[k]][1]);
                    }
                }
                if (num_flops) *num_flops += 20*A->num_nonzeros;
            } else if (a->precision == mtx_double) {
                const double (* Adata)[2] = a->data.complex_double;
                const double (* xdata)[2] = x_->data.complex_double;
                double (* ydata)[2] = y_->data.complex_double;
                for (int k = 0; k < A->size; k++) {
                    ydata[i[k]][0] += alpha[0]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]) - alpha[1]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                    ydata[i[k]][1] += alpha[0]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]) + alpha[1]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                    if (i[k] != j[k]) {
                        ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i[k]][0]+Adata[k][1]*xdata[i[k]][1]) - alpha[1]*(Adata[k][0]*xdata[i[k]][1]-Adata[k][1]*xdata[i[k]][0]);
                        ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i[k]][1]-Adata[k][1]*xdata[i[k]][0]) + alpha[1]*(Adata[k][0]*xdata[i[k]][0]+Adata[k][1]*xdata[i[k]][1]);
                    }
                }
                if (num_flops) *num_flops += 20*A->num_nonzeros;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (trans == mtx_trans) {
            if (a->precision == mtx_single) {
                const float (* Adata)[2] = a->data.complex_single;
                const float (* xdata)[2] = x_->data.complex_single;
                float (* ydata)[2] = y_->data.complex_single;
                for (int k = 0; k < A->size; k++) {
                    ydata[i[k]][0] += alpha[0]*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]) - alpha[1]*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]);
                    ydata[i[k]][1] += alpha[0]*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]) + alpha[1]*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]);
                    if (i[k] != j[k]) {
                        ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i[k]][0]-Adata[k][1]*xdata[i[k]][1]) - alpha[1]*(Adata[k][0]*xdata[i[k]][1]+Adata[k][1]*xdata[i[k]][0]);
                        ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i[k]][1]+Adata[k][1]*xdata[i[k]][0]) + alpha[1]*(Adata[k][0]*xdata[i[k]][0]-Adata[k][1]*xdata[i[k]][1]);
                    }
                }
                if (num_flops) *num_flops += 20*A->num_nonzeros;
            } else if (a->precision == mtx_double) {
                const double (* Adata)[2] = a->data.complex_double;
                const double (* xdata)[2] = x_->data.complex_double;
                double (* ydata)[2] = y_->data.complex_double;
                for (int k = 0; k < A->size; k++) {
                    ydata[i[k]][0] += alpha[0]*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]) - alpha[1]*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]);
                    ydata[i[k]][1] += alpha[0]*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]) + alpha[1]*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]);
                    if (i[k] != j[k]) {
                        ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i[k]][0]-Adata[k][1]*xdata[i[k]][1]) - alpha[1]*(Adata[k][0]*xdata[i[k]][1]+Adata[k][1]*xdata[i[k]][0]);
                        ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i[k]][1]+Adata[k][1]*xdata[i[k]][0]) + alpha[1]*(Adata[k][0]*xdata[i[k]][0]-Adata[k][1]*xdata[i[k]][1]);
                    }
                }
                if (num_flops) *num_flops += 20*A->num_nonzeros;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else { return MTX_ERR_INVALID_TRANSPOSITION; }
    } else { return MTX_ERR_INVALID_SYMMETRY; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_coordinate_zgemv()’ multiplies a complex-valued matrix
 * ‘A’, its transpose ‘A'’ or its conjugate transpose ‘Aᴴ’ by a
 * complex scalar ‘alpha’ (‘α’) and a vector ‘x’, before adding the
 * result to another vector ‘y’ multiplied by another complex scalar
 * ‘beta’ (‘β’).  That is, ‘y = α*A*x + β*y’, ‘y = α*A'*x + β*y’ or ‘y
 * = α*Aᴴ*x + β*y’.
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
 */
int mtxmatrix_coordinate_zgemv(
    enum mtxtransposition trans,
    double alpha[2],
    const struct mtxmatrix_coordinate * A,
    const struct mtxvector * x,
    double beta[2],
    struct mtxvector * y,
    int64_t * num_flops)
{
    int err;
    const struct mtxvector_base * a = &A->a;
    if (x->type != mtxvector_base || y->type != mtxvector_base)
        return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
    const struct mtxvector_base * x_ = &x->storage.base;
    struct mtxvector_base * y_ = &y->storage.base;
    if (x_->field != a->field || y_->field != a->field)
        return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x_->precision != a->precision || y_->precision != a->precision)
        return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (trans == mtx_notrans) {
        if (A->num_rows != y_->size || A->num_columns != x_->size)
            return MTX_ERR_INCOMPATIBLE_SIZE;
    } else if (trans == mtx_trans || trans == mtx_conjtrans) {
        if (A->num_columns != y_->size || A->num_rows != x_->size)
            return MTX_ERR_INCOMPATIBLE_SIZE;
    }
    if (a->field != mtx_field_complex)
        return MTX_ERR_INCOMPATIBLE_FIELD;

    if (beta[0] != 1 || beta[1] != 0) {
        err = mtxvector_zscal(beta, y, num_flops);
        if (err) return err;
    }

    const int * i = A->rowidx;
    const int * j = A->colidx;
    if (A->symmetry == mtx_unsymmetric) {
        if (trans == mtx_notrans) {
            if (a->precision == mtx_single) {
                const float (* Adata)[2] = a->data.complex_single;
                const float (* xdata)[2] = x_->data.complex_single;
                float (* ydata)[2] = y_->data.complex_single;
                for (int k = 0; k < A->size; k++) {
                    ydata[i[k]][0] += alpha[0]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1])-alpha[1]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                    ydata[i[k]][1] += alpha[0]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0])+alpha[1]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                }
                if (num_flops) *num_flops += 20*A->num_nonzeros;
            } else if (a->precision == mtx_double) {
                const double (* Adata)[2] = a->data.complex_double;
                const double (* xdata)[2] = x_->data.complex_double;
                double (* ydata)[2] = y_->data.complex_double;
                for (int k = 0; k < A->size; k++) {
                    ydata[i[k]][0] += alpha[0]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1])-alpha[1]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                    ydata[i[k]][1] += alpha[0]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0])+alpha[1]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                }
                if (num_flops) *num_flops += 20*A->num_nonzeros;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (trans == mtx_trans) {
            if (a->precision == mtx_single) {
                const float (* Adata)[2] = a->data.complex_single;
                const float (* xdata)[2] = x_->data.complex_single;
                float (* ydata)[2] = y_->data.complex_single;
                for (int k = 0; k < A->size; k++) {
                    ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i[k]][0]-Adata[k][1]*xdata[i[k]][1])-alpha[1]*(Adata[k][0]*xdata[i[k]][1]+Adata[k][1]*xdata[i[k]][0]);
                    ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i[k]][1]+Adata[k][1]*xdata[i[k]][0])+alpha[1]*(Adata[k][0]*xdata[i[k]][0]-Adata[k][1]*xdata[i[k]][1]);
                }
                if (num_flops) *num_flops += 20*A->num_nonzeros;
            } else if (a->precision == mtx_double) {
                const double (* Adata)[2] = a->data.complex_double;
                const double (* xdata)[2] = x_->data.complex_double;
                double (* ydata)[2] = y_->data.complex_double;
                for (int k = 0; k < A->size; k++) {
                    ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i[k]][0]-Adata[k][1]*xdata[i[k]][1])-alpha[1]*(Adata[k][0]*xdata[i[k]][1]+Adata[k][1]*xdata[i[k]][0]);
                    ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i[k]][1]+Adata[k][1]*xdata[i[k]][0])+alpha[1]*(Adata[k][0]*xdata[i[k]][0]-Adata[k][1]*xdata[i[k]][1]);
                }
                if (num_flops) *num_flops += 20*A->num_nonzeros;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (trans == mtx_conjtrans) {
            if (a->precision == mtx_single) {
                const float (* Adata)[2] = a->data.complex_single;
                const float (* xdata)[2] = x_->data.complex_single;
                float (* ydata)[2] = y_->data.complex_single;
                for (int k = 0; k < A->size; k++) {
                    ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i[k]][0]+Adata[k][1]*xdata[i[k]][1])-alpha[1]*(Adata[k][0]*xdata[i[k]][1]-Adata[k][1]*xdata[i[k]][0]);
                    ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i[k]][1]-Adata[k][1]*xdata[i[k]][0])+alpha[1]*(Adata[k][0]*xdata[i[k]][0]+Adata[k][1]*xdata[i[k]][1]);
                }
                if (num_flops) *num_flops += 20*A->num_nonzeros;
            } else if (a->precision == mtx_double) {
                const double (* Adata)[2] = a->data.complex_double;
                const double (* xdata)[2] = x_->data.complex_double;
                double (* ydata)[2] = y_->data.complex_double;
                for (int k = 0; k < A->size; k++) {
                    ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i[k]][0]+Adata[k][1]*xdata[i[k]][1])-alpha[1]*(Adata[k][0]*xdata[i[k]][1]-Adata[k][1]*xdata[i[k]][0]);
                    ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i[k]][1]-Adata[k][1]*xdata[i[k]][0])+alpha[1]*(Adata[k][0]*xdata[i[k]][0]+Adata[k][1]*xdata[i[k]][1]);
                }
                if (num_flops) *num_flops += 20*A->num_nonzeros;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else { return MTX_ERR_INVALID_TRANSPOSITION; }
    } else if (A->num_rows == A->num_columns && A->symmetry == mtx_symmetric) {
        if (trans == mtx_notrans || trans == mtx_trans) {
            if (a->precision == mtx_single) {
                const float (* Adata)[2] = a->data.complex_single;
                const float (* xdata)[2] = x_->data.complex_single;
                float (* ydata)[2] = y_->data.complex_single;
                for (int k = 0; k < A->size; k++) {
                    ydata[i[k]][0] += alpha[0]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]) - alpha[1]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                    ydata[i[k]][1] += alpha[0]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]) + alpha[1]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                    if (i[k] != j[k]) {
                        ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i[k]][0]-Adata[k][1]*xdata[i[k]][1]) - alpha[1]*(Adata[k][0]*xdata[i[k]][1]+Adata[k][1]*xdata[i[k]][0]);
                        ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i[k]][1]+Adata[k][1]*xdata[i[k]][0]) + alpha[1]*(Adata[k][0]*xdata[i[k]][0]-Adata[k][1]*xdata[i[k]][1]);
                    }
                }
                if (num_flops) *num_flops += 20*A->num_nonzeros;
            } else if (a->precision == mtx_double) {
                const double (* Adata)[2] = a->data.complex_double;
                const double (* xdata)[2] = x_->data.complex_double;
                double (* ydata)[2] = y_->data.complex_double;
                for (int k = 0; k < A->size; k++) {
                    ydata[i[k]][0] += alpha[0]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]) - alpha[1]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                    ydata[i[k]][1] += alpha[0]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]) + alpha[1]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                    if (i[k] != j[k]) {
                        ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i[k]][0]-Adata[k][1]*xdata[i[k]][1]) - alpha[1]*(Adata[k][0]*xdata[i[k]][1]+Adata[k][1]*xdata[i[k]][0]);
                        ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i[k]][1]+Adata[k][1]*xdata[i[k]][0]) + alpha[1]*(Adata[k][0]*xdata[i[k]][0]-Adata[k][1]*xdata[i[k]][1]);
                    }
                }
                if (num_flops) *num_flops += 20*A->num_nonzeros;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (trans == mtx_conjtrans) {
            if (a->precision == mtx_single) {
                const float (* Adata)[2] = a->data.complex_single;
                const float (* xdata)[2] = x_->data.complex_single;
                float (* ydata)[2] = y_->data.complex_single;
                for (int k = 0; k < A->size; k++) {
                    ydata[i[k]][0] += alpha[0]*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]) - alpha[1]*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]);
                    ydata[i[k]][1] += alpha[0]*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]) + alpha[1]*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]);
                    if (i[k] != j[k]) {
                        ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i[k]][0]+Adata[k][1]*xdata[i[k]][1]) - alpha[1]*(Adata[k][0]*xdata[i[k]][1]-Adata[k][1]*xdata[i[k]][0]);
                        ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i[k]][1]-Adata[k][1]*xdata[i[k]][0]) + alpha[1]*(Adata[k][0]*xdata[i[k]][0]+Adata[k][1]*xdata[i[k]][1]);
                    }
                }
                if (num_flops) *num_flops += 20*A->num_nonzeros;
            } else if (a->precision == mtx_double) {
                const double (* Adata)[2] = a->data.complex_double;
                const double (* xdata)[2] = x_->data.complex_double;
                double (* ydata)[2] = y_->data.complex_double;
                for (int k = 0; k < A->size; k++) {
                    ydata[i[k]][0] += alpha[0]*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]) - alpha[1]*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]);
                    ydata[i[k]][1] += alpha[0]*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]) + alpha[1]*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]);
                    if (i[k] != j[k]) {
                        ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i[k]][0]+Adata[k][1]*xdata[i[k]][1]) - alpha[1]*(Adata[k][0]*xdata[i[k]][1]-Adata[k][1]*xdata[i[k]][0]);
                        ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i[k]][1]-Adata[k][1]*xdata[i[k]][0]) + alpha[1]*(Adata[k][0]*xdata[i[k]][0]+Adata[k][1]*xdata[i[k]][1]);
                    }
                }
                if (num_flops) *num_flops += 20*A->num_nonzeros;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else { return MTX_ERR_INVALID_TRANSPOSITION; }
    } else if (A->num_rows == A->num_columns && A->symmetry == mtx_skew_symmetric) {
        /* TODO: allow skew-symmetric matrices */
        return MTX_ERR_INVALID_SYMMETRY;
    } else if (A->num_rows == A->num_columns && A->symmetry == mtx_hermitian) {
        if (trans == mtx_notrans || trans == mtx_conjtrans) {
            if (a->precision == mtx_single) {
                const float (* Adata)[2] = a->data.complex_single;
                const float (* xdata)[2] = x_->data.complex_single;
                float (* ydata)[2] = y_->data.complex_single;
                for (int k = 0; k < A->size; k++) {
                    ydata[i[k]][0] += alpha[0]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]) - alpha[1]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                    ydata[i[k]][1] += alpha[0]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]) + alpha[1]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                    if (i[k] != j[k]) {
                        ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i[k]][0]+Adata[k][1]*xdata[i[k]][1]) - alpha[1]*(Adata[k][0]*xdata[i[k]][1]-Adata[k][1]*xdata[i[k]][0]);
                        ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i[k]][1]-Adata[k][1]*xdata[i[k]][0]) + alpha[1]*(Adata[k][0]*xdata[i[k]][0]+Adata[k][1]*xdata[i[k]][1]);
                    }
                }
                if (num_flops) *num_flops += 20*A->num_nonzeros;
            } else if (a->precision == mtx_double) {
                const double (* Adata)[2] = a->data.complex_double;
                const double (* xdata)[2] = x_->data.complex_double;
                double (* ydata)[2] = y_->data.complex_double;
                for (int k = 0; k < A->size; k++) {
                    ydata[i[k]][0] += alpha[0]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]) - alpha[1]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                    ydata[i[k]][1] += alpha[0]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]) + alpha[1]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                    if (i[k] != j[k]) {
                        ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i[k]][0]+Adata[k][1]*xdata[i[k]][1]) - alpha[1]*(Adata[k][0]*xdata[i[k]][1]-Adata[k][1]*xdata[i[k]][0]);
                        ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i[k]][1]-Adata[k][1]*xdata[i[k]][0]) + alpha[1]*(Adata[k][0]*xdata[i[k]][0]+Adata[k][1]*xdata[i[k]][1]);
                    }
                }
                if (num_flops) *num_flops += 20*A->num_nonzeros;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (trans == mtx_trans) {
            if (a->precision == mtx_single) {
                const float (* Adata)[2] = a->data.complex_single;
                const float (* xdata)[2] = x_->data.complex_single;
                float (* ydata)[2] = y_->data.complex_single;
                for (int k = 0; k < A->size; k++) {
                    ydata[i[k]][0] += alpha[0]*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]) - alpha[1]*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]);
                    ydata[i[k]][1] += alpha[0]*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]) + alpha[1]*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]);
                    if (i[k] != j[k]) {
                        ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i[k]][0]-Adata[k][1]*xdata[i[k]][1]) - alpha[1]*(Adata[k][0]*xdata[i[k]][1]+Adata[k][1]*xdata[i[k]][0]);
                        ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i[k]][1]+Adata[k][1]*xdata[i[k]][0]) + alpha[1]*(Adata[k][0]*xdata[i[k]][0]-Adata[k][1]*xdata[i[k]][1]);
                    }
                }
            } else if (a->precision == mtx_double) {
                const double (* Adata)[2] = a->data.complex_double;
                const double (* xdata)[2] = x_->data.complex_double;
                double (* ydata)[2] = y_->data.complex_double;
                for (int k = 0; k < A->size; k++) {
                    ydata[i[k]][0] += alpha[0]*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]) - alpha[1]*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]);
                    ydata[i[k]][1] += alpha[0]*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]) + alpha[1]*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]);
                    if (i[k] != j[k]) {
                        ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i[k]][0]-Adata[k][1]*xdata[i[k]][1]) - alpha[1]*(Adata[k][0]*xdata[i[k]][1]+Adata[k][1]*xdata[i[k]][0]);
                        ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i[k]][1]+Adata[k][1]*xdata[i[k]][0]) + alpha[1]*(Adata[k][0]*xdata[i[k]][0]-Adata[k][1]*xdata[i[k]][1]);
                    }
                }
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else { return MTX_ERR_INVALID_TRANSPOSITION; }
    } else { return MTX_ERR_INVALID_SYMMETRY; }
    return MTX_SUCCESS;
}
