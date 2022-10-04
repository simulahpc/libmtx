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
 * Data structures for matrices in coordinate format.
 */

#include <libmtx/libmtx-config.h>

#include <libmtx/error.h>
#include <libmtx/vector/field.h>
#include <libmtx/matrix/matrix.h>
#include <libmtx/linalg/base/coo.h>
#include <libmtx/mtxfile/mtxfile.h>
#include <libmtx/util/sort.h>
#include <libmtx/linalg/precision.h>
#include <libmtx/linalg/base/vector.h>
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
 * matrix properties
 */

/**
 * ‘mtxmatrix_coo_field()’ gets the field of a matrix.
 */
enum mtxfield mtxmatrix_coo_field(const struct mtxmatrix_coo * A)
{
    return mtxvector_base_field(&A->a);
}

/**
 * ‘mtxmatrix_coo_precision()’ gets the precision of a matrix.
 */
enum mtxprecision mtxmatrix_coo_precision(const struct mtxmatrix_coo * A)
{
    return mtxvector_base_precision(&A->a);
}

/**
 * ‘mtxmatrix_coo_symmetry()’ gets the symmetry of a matrix.
 */
enum mtxsymmetry mtxmatrix_coo_symmetry(const struct mtxmatrix_coo * A)
{
    return A->symmetry;
}

/**
 * ‘mtxmatrix_coo_num_rows()’ gets the number of matrix rows.
 */
int mtxmatrix_coo_num_rows(const struct mtxmatrix_coo * A)
{
    return A->num_rows;
}

/**
 * ‘mtxmatrix_coo_num_columns()’ gets the number of matrix columns.
 */
int mtxmatrix_coo_num_columns(const struct mtxmatrix_coo * A)
{
    return A->num_columns;
}

/**
 * ‘mtxmatrix_coo_num_nonzeros()’ gets the number of the number of
 *  nonzero matrix entries, including those represented implicitly due
 *  to symmetry.
 */
int64_t mtxmatrix_coo_num_nonzeros(const struct mtxmatrix_coo * A)
{
    return A->num_nonzeros;
}

/**
 * ‘mtxmatrix_coo_size()’ gets the number of explicitly stored
 * nonzeros of a matrix.
 */
int64_t mtxmatrix_coo_size(const struct mtxmatrix_coo * A)
{
    return A->size;
}

/**
 * ‘mtxmatrix_coo_rowcolidx()’ gets the row and column indices of the
 * explicitly stored matrix nonzeros.
 *
 * The arguments ‘rowidx’ and ‘colidx’ may be ‘NULL’ or must point to
 * an arrays of length ‘size’.
 */
int mtxmatrix_coo_rowcolidx(
    const struct mtxmatrix_coo * A,
    int64_t size,
    int * rowidx,
    int * colidx)
{
    if (size != A->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (rowidx) { for (int64_t k = 0; k < A->size; k++) rowidx[k] = A->rowidx[k]; }
    if (colidx) { for (int64_t k = 0; k < A->size; k++) colidx[k] = A->colidx[k]; }
    return MTX_SUCCESS;
}

/*
 * memory management
 */

/**
 * ‘mtxmatrix_coo_free()’ frees storage allocated for a matrix.
 */
void mtxmatrix_coo_free(
    struct mtxmatrix_coo * A)
{
    mtxvector_base_free(&A->a);
    free(A->colidx);
    free(A->rowidx);
}

/**
 * ‘mtxmatrix_coo_alloc_copy()’ allocates a copy of a matrix
 * without initialising the values.
 */
int mtxmatrix_coo_alloc_copy(
    struct mtxmatrix_coo * dst,
    const struct mtxmatrix_coo * src)
{
    return mtxmatrix_coo_alloc_entries(
        dst, src->a.field, src->a.precision, src->symmetry,
        src->num_rows, src->num_columns, src->size,
        sizeof(*src->rowidx), 0, src->rowidx, src->colidx);
}

/**
 * ‘mtxmatrix_coo_init_copy()’ allocates a copy of a matrix and
 * also copies the values.
 */
int mtxmatrix_coo_init_copy(
    struct mtxmatrix_coo * dst,
    const struct mtxmatrix_coo * src)
{
    int err = mtxmatrix_coo_alloc_copy(dst, src);
    if (err) return err;
    err = mtxmatrix_coo_copy(dst, src);
    if (err) { mtxmatrix_coo_free(dst); return err; }
    return MTX_SUCCESS;
}

/*
 * initialise matrices from entrywise data in coordinate format
 */

/**
 * ‘mtxmatrix_coo_alloc_entries()’ allocates a matrix from
 * entrywise data in coordinate format.
 */
int mtxmatrix_coo_alloc_entries(
    struct mtxmatrix_coo * A,
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
 * ‘mtxmatrix_coo_init_entries_real_single()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * real, single precision coefficients.
 */
int mtxmatrix_coo_init_entries_real_single(
    struct mtxmatrix_coo * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx,
    const float * data)
{
    int err = mtxmatrix_coo_alloc_entries(
        A, mtx_field_real, mtx_single, symmetry, num_rows, num_columns,
        size, sizeof(*rowidx), 0, rowidx, colidx);
    if (err) return err;
    err = mtxvector_base_set_real_single(&A->a, size, sizeof(*data), data);
    if (err) { mtxmatrix_coo_free(A); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_coo_init_entries_real_double()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * real, double precision coefficients.
 */
int mtxmatrix_coo_init_entries_real_double(
    struct mtxmatrix_coo * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx,
    const double * data)
{
    int err = mtxmatrix_coo_alloc_entries(
        A, mtx_field_real, mtx_double, symmetry, num_rows, num_columns,
        size, sizeof(*rowidx), 0, rowidx, colidx);
    if (err) return err;
    err = mtxvector_base_set_real_double(&A->a, size, sizeof(*data), data);
    if (err) { mtxmatrix_coo_free(A); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_coo_init_entries_complex_single()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * complex, single precision coefficients.
 */
int mtxmatrix_coo_init_entries_complex_single(
    struct mtxmatrix_coo * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx,
    const float (* data)[2])
{
    int err = mtxmatrix_coo_alloc_entries(
        A, mtx_field_complex, mtx_single, symmetry, num_rows, num_columns,
        size, sizeof(*rowidx), 0, rowidx, colidx);
    if (err) return err;
    err = mtxvector_base_set_complex_single(&A->a, size, sizeof(*data), data);
    if (err) { mtxmatrix_coo_free(A); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_coo_init_entries_complex_double()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * complex, double precision coefficients.
 */
int mtxmatrix_coo_init_entries_complex_double(
    struct mtxmatrix_coo * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx,
    const double (* data)[2])
{
    int err = mtxmatrix_coo_alloc_entries(
        A, mtx_field_complex, mtx_double, symmetry, num_rows, num_columns,
        size, sizeof(*rowidx), 0, rowidx, colidx);
    if (err) return err;
    err = mtxvector_base_set_complex_double(&A->a, size, sizeof(*data), data);
    if (err) { mtxmatrix_coo_free(A); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_coo_init_entries_integer_single()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * integer, single precision coefficients.
 */
int mtxmatrix_coo_init_entries_integer_single(
    struct mtxmatrix_coo * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx,
    const int32_t * data)
{
    int err = mtxmatrix_coo_alloc_entries(
        A, mtx_field_integer, mtx_single, symmetry, num_rows, num_columns,
        size, sizeof(*rowidx), 0, rowidx, colidx);
    if (err) return err;
    err = mtxvector_base_set_integer_single(&A->a, size, sizeof(*data), data);
    if (err) { mtxmatrix_coo_free(A); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_coo_init_entries_integer_double()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * integer, double precision coefficients.
 */
int mtxmatrix_coo_init_entries_integer_double(
    struct mtxmatrix_coo * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx,
    const int64_t * data)
{
    int err = mtxmatrix_coo_alloc_entries(
        A, mtx_field_integer, mtx_double, symmetry, num_rows, num_columns,
        size, sizeof(*rowidx), 0, rowidx, colidx);
    if (err) return err;
    err = mtxvector_base_set_integer_double(&A->a, size, sizeof(*data), data);
    if (err) { mtxmatrix_coo_free(A); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_coo_init_entries_pattern()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * boolean coefficients.
 */
int mtxmatrix_coo_init_entries_pattern(
    struct mtxmatrix_coo * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx)
{
    return mtxmatrix_coo_alloc_entries(
        A, mtx_field_pattern, mtx_single, symmetry, num_rows, num_columns,
        size, sizeof(*rowidx), 0, rowidx, colidx);
}

/*
 * initialise matrices from entrywise data in coordinate format with
 * specified strides
 */

/**
 * ‘mtxmatrix_coo_init_entries_strided_real_single()’ allocates
 * and initialises a matrix from entrywise data in coordinate format
 * with real, single precision coefficients.
 */
int mtxmatrix_coo_init_entries_strided_real_single(
    struct mtxmatrix_coo * A,
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
    int err = mtxmatrix_coo_alloc_entries(
        A, mtx_field_real, mtx_single, symmetry, num_rows, num_columns,
        size, idxstride, idxbase, rowidx, colidx);
    if (err) return err;
    err = mtxvector_base_set_real_single(&A->a, size, datastride, data);
    if (err) { mtxmatrix_coo_free(A); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_coo_init_entries_strided_real_double()’ allocates
 * and initialises a matrix from entrywise data in coordinate format
 * with real, double precision coefficients.
 */
int mtxmatrix_coo_init_entries_strided_real_double(
    struct mtxmatrix_coo * A,
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
    int err = mtxmatrix_coo_alloc_entries(
        A, mtx_field_real, mtx_double, symmetry, num_rows, num_columns,
        size, idxstride, idxbase, rowidx, colidx);
    if (err) return err;
    err = mtxvector_base_set_real_double(&A->a, size, datastride, data);
    if (err) { mtxmatrix_coo_free(A); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_coo_init_entries_strided_complex_single()’
 * allocates and initialises a matrix from entrywise data in
 * coordinate format with complex, single precision coefficients.
 */
int mtxmatrix_coo_init_entries_strided_complex_single(
    struct mtxmatrix_coo * A,
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
    int err = mtxmatrix_coo_alloc_entries(
        A, mtx_field_complex, mtx_single, symmetry, num_rows, num_columns,
        size, idxstride, idxbase, rowidx, colidx);
    if (err) return err;
    err = mtxvector_base_set_complex_single(&A->a, size, datastride, data);
    if (err) { mtxmatrix_coo_free(A); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_coo_init_entries_strided_complex_double()’
 * allocates and initialises a matrix from entrywise data in
 * coordinate format with complex, double precision coefficients.
 */
int mtxmatrix_coo_init_entries_strided_complex_double(
    struct mtxmatrix_coo * A,
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
    int err = mtxmatrix_coo_alloc_entries(
        A, mtx_field_complex, mtx_double, symmetry, num_rows, num_columns,
        size, idxstride, idxbase, rowidx, colidx);
    if (err) return err;
    err = mtxvector_base_set_complex_double(&A->a, size, datastride, data);
    if (err) { mtxmatrix_coo_free(A); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_coo_init_entries_strided_integer_single()’
 * allocates and initialises a matrix from entrywise data in
 * coordinate format with integer, single precision coefficients.
 */
int mtxmatrix_coo_init_entries_strided_integer_single(
    struct mtxmatrix_coo * A,
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
    int err = mtxmatrix_coo_alloc_entries(
        A, mtx_field_integer, mtx_single, symmetry, num_rows, num_columns,
        size, idxstride, idxbase, rowidx, colidx);
    if (err) return err;
    err = mtxvector_base_set_integer_single(&A->a, size, datastride, data);
    if (err) { mtxmatrix_coo_free(A); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_coo_init_entries_strided_integer_double()’
 * allocates and initialises a matrix from entrywise data in
 * coordinate format with integer, double precision coefficients.
 */
int mtxmatrix_coo_init_entries_strided_integer_double(
    struct mtxmatrix_coo * A,
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
    int err = mtxmatrix_coo_alloc_entries(
        A, mtx_field_integer, mtx_double, symmetry, num_rows, num_columns,
        size, idxstride, idxbase, rowidx, colidx);
    if (err) return err;
    err = mtxvector_base_set_integer_double(&A->a, size, datastride, data);
    if (err) { mtxmatrix_coo_free(A); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_coo_init_entries_strided_pattern()’ allocates and
 * initialises a matrix from entrywise data in coordinate format with
 * boolean coefficients.
 */
int mtxmatrix_coo_init_entries_strided_pattern(
    struct mtxmatrix_coo * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    int idxstride,
    int idxbase,
    const int * rowidx,
    const int * colidx)
{
    int err = mtxmatrix_coo_alloc_entries(
        A, mtx_field_pattern, mtx_single, symmetry, num_rows, num_columns,
        size, idxstride, idxbase, rowidx, colidx);
    if (err) return err;
    err = mtxvector_base_init_pattern(&A->a, size);
    if (err) { mtxmatrix_coo_free(A); return err; }
    return MTX_SUCCESS;
}

/*
 * initialise matrices from row-wise data in compressed row format
 */

/**
 * ‘mtxmatrix_coo_alloc_rows()’ allocates a matrix from
 * row-wise data in compressed row format.
 */
int mtxmatrix_coo_alloc_rows(
    struct mtxmatrix_coo * A,
    enum mtxfield field,
    enum mtxprecision precision,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
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
    A->size = rowptr[num_rows];
    A->rowidx = malloc(A->size * sizeof(int));
    if (!A->rowidx) return MTX_ERR_ERRNO;
    A->colidx = malloc(A->size * sizeof(int));
    if (!A->colidx) { free(A->rowidx); return MTX_ERR_ERRNO; }
    for (int i = 0; i < num_rows; i++) {
        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++) {
            A->rowidx[k] = i;
            A->colidx[k] = colidx[k];
            A->num_nonzeros +=
                (symmetry == mtx_unsymmetric || A->rowidx[k] == A->colidx[k]) ? 1 : 2;
        }
    }
    int err = mtxvector_base_alloc(&A->a, field, precision, A->size);
    if (err) { free(A->colidx); free(A->rowidx); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_coo_init_rows_real_single()’ allocates and
 * initialises a matrix from row-wise data in compressed row format
 * with real, single precision coefficients.
 */
int mtxmatrix_coo_init_rows_real_single(
    struct mtxmatrix_coo * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const float * data)
{
    int err = mtxmatrix_coo_alloc_rows(
        A, mtx_field_real, mtx_single, symmetry,
        num_rows, num_columns, rowptr, colidx);
    if (err) return err;
    err = mtxvector_base_set_real_single(
        &A->a, rowptr[num_rows], sizeof(*data), data);
    if (err) { mtxmatrix_coo_free(A); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_coo_init_rows_real_double()’ allocates and
 * initialises a matrix from row-wise data in compressed row format
 * with real, double precision coefficients.
 */
int mtxmatrix_coo_init_rows_real_double(
    struct mtxmatrix_coo * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const double * data)
{
    int err = mtxmatrix_coo_alloc_rows(
        A, mtx_field_real, mtx_double, symmetry,
        num_rows, num_columns, rowptr, colidx);
    if (err) return err;
    err = mtxvector_base_set_real_double(
        &A->a, rowptr[num_rows], sizeof(*data), data);
    if (err) { mtxmatrix_coo_free(A); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_coo_init_rows_complex_single()’ allocates and
 * initialises a matrix from row-wise data in compressed row format
 * with complex, single precision coefficients.
 */
int mtxmatrix_coo_init_rows_complex_single(
    struct mtxmatrix_coo * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const float (* data)[2])
{
    int err = mtxmatrix_coo_alloc_rows(
        A, mtx_field_complex, mtx_single, symmetry,
        num_rows, num_columns, rowptr, colidx);
    if (err) return err;
    err = mtxvector_base_set_complex_single(
        &A->a, rowptr[num_rows], sizeof(*data), data);
    if (err) { mtxmatrix_coo_free(A); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_coo_init_rows_complex_double()’ allocates and
 * initialises a matrix from row-wise data in compressed row format
 * with complex, double precision coefficients.
 */
int mtxmatrix_coo_init_rows_complex_double(
    struct mtxmatrix_coo * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const double (* data)[2])
{
    int err = mtxmatrix_coo_alloc_rows(
        A, mtx_field_complex, mtx_double, symmetry,
        num_rows, num_columns, rowptr, colidx);
    if (err) return err;
    err = mtxvector_base_set_complex_double(
        &A->a, rowptr[num_rows], sizeof(*data), data);
    if (err) { mtxmatrix_coo_free(A); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_coo_init_rows_integer_single()’ allocates and
 * initialises a matrix from row-wise data in compressed row format
 * with integer, single precision coefficients.
 */
int mtxmatrix_coo_init_rows_integer_single(
    struct mtxmatrix_coo * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const int32_t * data)
{
    int err = mtxmatrix_coo_alloc_rows(
        A, mtx_field_integer, mtx_single, symmetry,
        num_rows, num_columns, rowptr, colidx);
    if (err) return err;
    err = mtxvector_base_set_integer_single(
        &A->a, rowptr[num_rows], sizeof(*data), data);
    if (err) { mtxmatrix_coo_free(A); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_coo_init_rows_integer_double()’ allocates and
 * initialises a matrix from row-wise data in compressed row format
 * with integer, double precision coefficients.
 */
int mtxmatrix_coo_init_rows_integer_double(
    struct mtxmatrix_coo * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const int64_t * data)
{
    int err = mtxmatrix_coo_alloc_rows(
        A, mtx_field_integer, mtx_double, symmetry,
        num_rows, num_columns, rowptr, colidx);
    if (err) return err;
    err = mtxvector_base_set_integer_double(
        &A->a, rowptr[num_rows], sizeof(*data), data);
    if (err) { mtxmatrix_coo_free(A); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_coo_init_rows_pattern()’ allocates and
 * initialises a matrix from row-wise data in compressed row format
 * with boolean coefficients.
 */
int mtxmatrix_coo_init_rows_pattern(
    struct mtxmatrix_coo * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx)
{
    return mtxmatrix_coo_alloc_rows(
        A, mtx_field_pattern, mtx_single, symmetry,
        num_rows, num_columns, rowptr, colidx);
}

/*
 * initialise matrices from column-wise data in compressed column
 * format
 */

/**
 * ‘mtxmatrix_coo_alloc_columns()’ allocates a matrix from
 * column-wise data in compressed column format.
 */
int mtxmatrix_coo_alloc_columns(
    struct mtxmatrix_coo * A,
    enum mtxfield field,
    enum mtxprecision precision,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx);

/**
 * ‘mtxmatrix_coo_init_columns_real_single()’ allocates and
 * initialises a matrix from column-wise data in compressed column
 * format with real, single precision coefficients.
 */
int mtxmatrix_coo_init_columns_real_single(
    struct mtxmatrix_coo * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx,
    const float * data);

/**
 * ‘mtxmatrix_coo_init_columns_real_double()’ allocates and
 * initialises a matrix from column-wise data in compressed column
 * format with real, double precision coefficients.
 */
int mtxmatrix_coo_init_columns_real_double(
    struct mtxmatrix_coo * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx,
    const double * data);

/**
 * ‘mtxmatrix_coo_init_columns_complex_single()’ allocates and
 * initialises a matrix from column-wise data in compressed column
 * format with complex, single precision coefficients.
 */
int mtxmatrix_coo_init_columns_complex_single(
    struct mtxmatrix_coo * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx,
    const float (* data)[2]);

/**
 * ‘mtxmatrix_coo_init_columns_complex_double()’ allocates and
 * initialises a matrix from column-wise data in compressed column
 * format with complex, double precision coefficients.
 */
int mtxmatrix_coo_init_columns_complex_double(
    struct mtxmatrix_coo * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx,
    const double (* data)[2]);

/**
 * ‘mtxmatrix_coo_init_columns_integer_single()’ allocates and
 * initialises a matrix from column-wise data in compressed column
 * format with integer, single precision coefficients.
 */
int mtxmatrix_coo_init_columns_integer_single(
    struct mtxmatrix_coo * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx,
    const int32_t * data);

/**
 * ‘mtxmatrix_coo_init_columns_integer_double()’ allocates and
 * initialises a matrix from column-wise data in compressed column
 * format with integer, double precision coefficients.
 */
int mtxmatrix_coo_init_columns_integer_double(
    struct mtxmatrix_coo * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx,
    const int64_t * data);

/**
 * ‘mtxmatrix_coo_init_columns_pattern()’ allocates and
 * initialises a matrix from column-wise data in compressed column
 * format with boolean coefficients.
 */
int mtxmatrix_coo_init_columns_pattern(
    struct mtxmatrix_coo * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * colptr,
    const int * rowidx);

/*
 * initialise matrices from a list of dense cliques
 */

/**
 * ‘mtxmatrix_coo_alloc_cliques()’ allocates a matrix from a
 * list of cliques.
 */
int mtxmatrix_coo_alloc_cliques(
    struct mtxmatrix_coo * A,
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
 * ‘mtxmatrix_coo_init_cliques_real_single()’ allocates and
 * initialises a matrix from a list of cliques with real, single
 * precision coefficients.
 */
int mtxmatrix_coo_init_cliques_real_single(
    struct mtxmatrix_coo * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_cliques,
    const int64_t * cliqueptr,
    const int * rowidx,
    const int * colidx,
    const float * data);

/**
 * ‘mtxmatrix_coo_init_cliques_real_double()’ allocates and
 * initialises a matrix from a list of cliques with real, double
 * precision coefficients.
 */
int mtxmatrix_coo_init_cliques_real_double(
    struct mtxmatrix_coo * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_cliques,
    const int64_t * cliqueptr,
    const int * rowidx,
    const int * colidx,
    const double * data);

/**
 * ‘mtxmatrix_coo_init_cliques_complex_single()’ allocates and
 * initialises a matrix from a list of cliques with complex, single
 * precision coefficients.
 */
int mtxmatrix_coo_init_cliques_complex_single(
    struct mtxmatrix_coo * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_cliques,
    const int64_t * cliqueptr,
    const int * rowidx,
    const int * colidx,
    const float (* data)[2]);

/**
 * ‘mtxmatrix_coo_init_cliques_complex_double()’ allocates and
 * initialises a matrix from a list of cliques with complex, double
 * precision coefficients.
 */
int mtxmatrix_coo_init_cliques_complex_double(
    struct mtxmatrix_coo * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_cliques,
    const int64_t * cliqueptr,
    const int * rowidx,
    const int * colidx,
    const double (* data)[2]);

/**
 * ‘mtxmatrix_coo_init_cliques_integer_single()’ allocates and
 * initialises a matrix from a list of cliques with integer, single
 * precision coefficients.
 */
int mtxmatrix_coo_init_cliques_integer_single(
    struct mtxmatrix_coo * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_cliques,
    const int64_t * cliqueptr,
    const int * rowidx,
    const int * colidx,
    const int32_t * data);

/**
 * ‘mtxmatrix_coo_init_cliques_integer_double()’ allocates and
 * initialises a matrix from a list of cliques with integer, double
 * precision coefficients.
 */
int mtxmatrix_coo_init_cliques_integer_double(
    struct mtxmatrix_coo * A,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_cliques,
    const int64_t * cliqueptr,
    const int * rowidx,
    const int * colidx,
    const int64_t * data);

/**
 * ‘mtxmatrix_coo_init_cliques_pattern()’ allocates and
 * initialises a matrix from a list of cliques with boolean
 * coefficients.
 */
int mtxmatrix_coo_init_cliques_pattern(
    struct mtxmatrix_coo * A,
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
 * ‘mtxmatrix_coo_setzero()’ sets every value of a matrix to
 * zero.
 */
int mtxmatrix_coo_setzero(
    struct mtxmatrix_coo * A)
{
    return mtxvector_base_setzero(&A->a);
}

/**
 * ‘mtxmatrix_coo_set_real_single()’ sets values of a matrix
 * based on an array of single precision floating point numbers.
 */
int mtxmatrix_coo_set_real_single(
    struct mtxmatrix_coo * A,
    int64_t size,
    int stride,
    const float * a)
{
    return mtxvector_base_set_real_single(&A->a, size, stride, a);
}

/**
 * ‘mtxmatrix_coo_set_real_double()’ sets values of a matrix
 * based on an array of double precision floating point numbers.
 */
int mtxmatrix_coo_set_real_double(
    struct mtxmatrix_coo * A,
    int64_t size,
    int stride,
    const double * a)
{
    return mtxvector_base_set_real_double(&A->a, size, stride, a);
}

/**
 * ‘mtxmatrix_coo_set_complex_single()’ sets values of a matrix
 * based on an array of single precision floating point complex
 * numbers.
 */
int mtxmatrix_coo_set_complex_single(
    struct mtxmatrix_coo * A,
    int64_t size,
    int stride,
    const float (*a)[2])
{
    return mtxvector_base_set_complex_single(&A->a, size, stride, a);
}

/**
 * ‘mtxmatrix_coo_set_complex_double()’ sets values of a matrix
 * based on an array of double precision floating point complex
 * numbers.
 */
int mtxmatrix_coo_set_complex_double(
    struct mtxmatrix_coo * A,
    int64_t size,
    int stride,
    const double (*a)[2])
{
    return mtxvector_base_set_complex_double(&A->a, size, stride, a);
}

/**
 * ‘mtxmatrix_coo_set_integer_single()’ sets values of a matrix
 * based on an array of integers.
 */
int mtxmatrix_coo_set_integer_single(
    struct mtxmatrix_coo * A,
    int64_t size,
    int stride,
    const int32_t * a)
{
    return mtxvector_base_set_integer_single(&A->a, size, stride, a);
}

/**
 * ‘mtxmatrix_coo_set_integer_double()’ sets values of a matrix
 * based on an array of integers.
 */
int mtxmatrix_coo_set_integer_double(
    struct mtxmatrix_coo * A,
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
 * ‘mtxmatrix_coo_alloc_row_vector()’ allocates a row vector
 * for a given matrix, where a row vector is a vector whose length
 * equal to a single row of the matrix.
 */
int mtxmatrix_coo_alloc_row_vector(
    const struct mtxmatrix_coo * A,
    struct mtxvector * x,
    enum mtxvectortype vectortype)
{
    return mtxvector_alloc(
        x, vectortype, A->a.field, A->a.precision, A->num_columns);
}

/**
 * ‘mtxmatrix_coo_alloc_column_vector()’ allocates a column
 * vector for a given matrix, where a column vector is a vector whose
 * length equal to a single column of the matrix.
 */
int mtxmatrix_coo_alloc_column_vector(
    const struct mtxmatrix_coo * A,
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
 * ‘mtxmatrix_coo_from_mtxfile()’ converts a matrix from Matrix
 * Market format.
 */
int mtxmatrix_coo_from_mtxfile(
    struct mtxmatrix_coo * A,
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
 * ‘mtxmatrix_coo_to_mtxfile()’ converts a matrix to Matrix
 * Market format.
 */
int mtxmatrix_coo_to_mtxfile(
    struct mtxfile * mtxfile,
    const struct mtxmatrix_coo * A,
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
 * partitioning
 */

/**
 * ‘mtxmatrix_coo_partition_rowwise()’ partitions the entries of a matrix
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
int mtxmatrix_coo_partition_rowwise(
    const struct mtxmatrix_coo * A,
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
    if (dstrowpart) {
        for (int i = 0; i < A->num_rows; i++) dstrowpart[i] = i;
        int err = partition_int(
            parttype, A->num_rows, num_parts, partsizes, blksize, parts,
            A->num_rows, sizeof(*dstrowpart), dstrowpart,
            dstrowpart, dstrowpartsizes);
        if (err) return err;
    }
    return partition_int(
        parttype, A->num_rows, num_parts, partsizes, blksize, parts,
        A->size, sizeof(*A->rowidx), A->rowidx, dstnzpart, dstnzpartsizes);
}

/**
 * ‘mtxmatrix_coo_partition_columnwise()’ partitions the entries of a
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
int mtxmatrix_coo_partition_columnwise(
    const struct mtxmatrix_coo * A,
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
    if (dstcolpart) {
        for (int j = 0; j < A->num_columns; j++) dstcolpart[j] = j;
        int err = partition_int(
            parttype, A->num_columns, num_parts, partsizes, blksize, parts,
            A->num_columns, sizeof(*dstcolpart), dstcolpart,
            dstcolpart, dstcolpartsizes);
        if (err) return err;
    }
    return partition_int(
        parttype, A->num_columns, num_parts, partsizes, blksize, parts,
        A->size, sizeof(*A->colidx), A->colidx, dstnzpart, dstnzpartsizes);
}

/**
 * ‘mtxmatrix_coo_partition_2d()’ partitions the entries of a matrix in a
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
 * If ‘dstnzpartsizes’ is not ‘NULL’, then it must point to an array
 * of length ‘num_row_parts’ times ‘num_col_parts’, which is used to
 * store the number of nonzeros assigned to each part.
 *
 * If ‘dstrowpartsizes’ or ‘dstcolpartsizes’ are not ‘NULL’, then they
 * must point to arrays of length ‘num_row_parts’ and ‘num_col_parts’,
 * respectively, which are used to store the number of rows and
 * columns assigned to each part.
 */
int mtxmatrix_coo_partition_2d(
    const struct mtxmatrix_coo * A,
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
    int num_parts = num_row_parts * num_col_parts;
    int * dstnzrowpart = malloc(A->size * sizeof(int));
    if (!dstnzrowpart) return MTX_ERR_ERRNO;
    int err = mtxmatrix_coo_partition_rowwise(
        A, rowparttype, num_row_parts, rowpartsizes, rowblksize, rowparts,
        dstnzrowpart, NULL, dstrowpart, dstrowpartsizes);
    if (err) { free(dstnzrowpart); return err; }
    err = mtxmatrix_coo_partition_columnwise(
        A, colparttype, num_col_parts, colpartsizes, colblksize, colparts,
        dstnzpart, NULL, dstcolpart, dstcolpartsizes);
    if (err) { free(dstnzrowpart); return err; }
    for (int64_t k = 0; k < A->size; k++)
        dstnzpart[k] = dstnzrowpart[k]*num_col_parts + dstnzpart[k];
    if (dstnzpartsizes) {
        for (int p = 0; p < num_parts; p++) dstnzpartsizes[p] = 0;
        for (int64_t k = 0; k < A->size; k++) dstnzpartsizes[dstnzpart[k]]++;
    }
    free(dstnzrowpart);
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_coo_split()’ splits a matrix into multiple matrices
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
 * of type ‘struct mtxmatrix_coo’. If successful, then ‘dsts[p]’
 * points to a matrix consisting of elements from ‘src’ that belong to
 * the ‘p’th part, as designated by the ‘parts’ array.
 *
 * The caller is responsible for calling ‘mtxmatrix_coo_free()’ to
 * free storage allocated for each matrix in the ‘dsts’ array.
 */
int mtxmatrix_coo_split(
    int num_parts,
    struct mtxmatrix_coo ** dsts,
    const struct mtxmatrix_coo * src,
    int64_t size,
    int * parts)
{
    if (size != src->size) return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    bool sorted = true;
    for (int64_t k = 0; k < size; k++) {
        if (parts[k] < 0 || parts[k] >= num_parts)
            return MTX_ERR_INDEX_OUT_OF_BOUNDS;
        if (k > 0 && parts[k-1] > parts[k]) sorted = false;
    }

    /* sort by part number and invert the sorting permutation */
    int64_t * invperm;
    if (sorted) {
        invperm = malloc(size * sizeof(int64_t));
        if (!invperm) return MTX_ERR_ERRNO;
        for (int64_t k = 0; k < size; k++) invperm[k] = k;
    } else {
        int64_t * perm = malloc(size * sizeof(int64_t));
        if (!perm) return MTX_ERR_ERRNO;
        int err = radix_sort_int(size, parts, perm);
        if (err) { free(perm); return err; }
        invperm = malloc(size * sizeof(int64_t));
        if (!invperm) { free(perm); return MTX_ERR_ERRNO; }
        for (int64_t k = 0; k < size; k++) invperm[perm[k]] = k;
        free(perm);
    }

    /*
     * Extract each part by a) counting the number elements in the
     * part, b) allocating storage, and c) gathering row and column
     * offsets, as well as nonzeros, for the part.
     */
    int64_t offset = 0;
    for (int p = 0; p < num_parts; p++) {
        int64_t partsize = 0;
        while (offset+partsize < size && parts[offset+partsize] == p) partsize++;

        dsts[p]->symmetry = src->symmetry;
        dsts[p]->num_rows = src->num_rows;
        dsts[p]->num_columns = src->num_columns;
        dsts[p]->num_entries = src->num_entries;
        dsts[p]->num_nonzeros = 0;
        dsts[p]->size = partsize;
        dsts[p]->rowidx = malloc(partsize * sizeof(int));
        if (!dsts[p]->rowidx) {
            for (int q = p-1; q >= 0; q--) mtxmatrix_coo_free(dsts[q]);
            free(invperm);
            return MTX_ERR_ERRNO;
        }
        dsts[p]->colidx = malloc(partsize * sizeof(int));
        if (!dsts[p]->colidx) {
            free(dsts[p]->rowidx);
            for (int q = p-1; q >= 0; q--) mtxmatrix_coo_free(dsts[q]);
            free(invperm);
            return MTX_ERR_ERRNO;
        }
        for (int64_t k = 0; k < partsize; k++) {
            dsts[p]->rowidx[k] = src->rowidx[invperm[offset+k]];
            dsts[p]->colidx[k] = src->colidx[invperm[offset+k]];
            dsts[p]->num_nonzeros +=
                (dsts[p]->symmetry == mtx_unsymmetric ||
                 dsts[p]->rowidx[k] == dsts[p]->colidx[k]) ? 1 : 2;
        }

        int err = mtxvector_base_alloc_packed(
            &dsts[p]->a, src->a.field, src->a.precision, size, partsize, &invperm[offset]);
        if (err) {
            free(dsts[p]->colidx); free(dsts[p]->rowidx);
            for (int q = p-1; q >= 0; q--) mtxmatrix_coo_free(dsts[q]);
            free(invperm);
            return err;
        }
        err = mtxvector_base_usga(&dsts[p]->a, &src->a);
        if (err) {
            mtxvector_base_free(&dsts[p]->a);
            free(dsts[p]->colidx); free(dsts[p]->rowidx);
            for (int q = p-1; q >= 0; q--) mtxmatrix_coo_free(dsts[q]);
            free(invperm);
            return err;
        }
        offset += partsize;
    }
    free(invperm);
    return MTX_SUCCESS;
}

/*
 * Level 1 BLAS operations
 */

/**
 * ‘mtxmatrix_coo_swap()’ swaps values of two matrices,
 * simultaneously performing ‘y <- x’ and ‘x <- y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_coo_swap(
    struct mtxmatrix_coo * x,
    struct mtxmatrix_coo * y)
{
    return mtxvector_base_swap(&x->a, &y->a);
}

/**
 * ‘mtxmatrix_coo_copy()’ copies values of a matrix, ‘y = x’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_coo_copy(
    struct mtxmatrix_coo * y,
    const struct mtxmatrix_coo * x)
{
    return mtxvector_base_copy(&y->a, &x->a);
}

/**
 * ‘mtxmatrix_coo_sscal()’ scales a matrix by a single
 * precision floating point scalar, ‘x = a*x’.
 */
int mtxmatrix_coo_sscal(
    float a,
    struct mtxmatrix_coo * x,
    int64_t * num_flops)
{
    return mtxvector_base_sscal(a, &x->a, num_flops);
}

/**
 * ‘mtxmatrix_coo_dscal()’ scales a matrix by a double
 * precision floating point scalar, ‘x = a*x’.
 */
int mtxmatrix_coo_dscal(
    double a,
    struct mtxmatrix_coo * x,
    int64_t * num_flops)
{
    return mtxvector_base_dscal(a, &x->a, num_flops);
}

/**
 * ‘mtxmatrix_coo_cscal()’ scales a matrix by a complex, single
 * precision floating point scalar, ‘x = (a+b*i)*x’.
 */
int mtxmatrix_coo_cscal(
    float a[2],
    struct mtxmatrix_coo * x,
    int64_t * num_flops)
{
    return mtxvector_base_cscal(a, &x->a, num_flops);
}

/**
 * ‘mtxmatrix_coo_zscal()’ scales a matrix by a complex, double
 * precision floating point scalar, ‘x = (a+b*i)*x’.
 */
int mtxmatrix_coo_zscal(
    double a[2],
    struct mtxmatrix_coo * x,
    int64_t * num_flops)
{
    return mtxvector_base_zscal(a, &x->a, num_flops);
}

/**
 * ‘mtxmatrix_coo_saxpy()’ adds a matrix to another one
 * multiplied by a single precision floating point value, ‘y = a*x +
 * y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_coo_saxpy(
    float a,
    const struct mtxmatrix_coo * x,
    struct mtxmatrix_coo * y,
    int64_t * num_flops)
{
    return mtxvector_base_saxpy(a, &x->a, &y->a, num_flops);
}

/**
 * ‘mtxmatrix_coo_daxpy()’ adds a matrix to another one
 * multiplied by a double precision floating point value, ‘y = a*x +
 * y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_coo_daxpy(
    double a,
    const struct mtxmatrix_coo * x,
    struct mtxmatrix_coo * y,
    int64_t * num_flops)
{
    return mtxvector_base_daxpy(a, &x->a, &y->a, num_flops);
}

/**
 * ‘mtxmatrix_coo_saypx()’ multiplies a matrix by a single
 * precision floating point scalar and adds another matrix, ‘y = a*y +
 * x’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_coo_saypx(
    float a,
    struct mtxmatrix_coo * y,
    const struct mtxmatrix_coo * x,
    int64_t * num_flops)
{
    return mtxvector_base_saypx(a, &y->a, &x->a, num_flops);
}

/**
 * ‘mtxmatrix_coo_daypx()’ multiplies a matrix by a double
 * precision floating point scalar and adds another matrix, ‘y = a*y +
 * x’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_coo_daypx(
    double a,
    struct mtxmatrix_coo * y,
    const struct mtxmatrix_coo * x,
    int64_t * num_flops)
{
    return mtxvector_base_daypx(a, &y->a, &x->a, num_flops);
}

/**
 * ‘mtxmatrix_coo_sdot()’ computes the Frobenius inner product
 * of two matrices in single precision floating point.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_coo_sdot(
    const struct mtxmatrix_coo * x,
    const struct mtxmatrix_coo * y,
    float * dot,
    int64_t * num_flops)
{
    if (x->symmetry == mtx_unsymmetric && y->symmetry == mtx_unsymmetric) {
        return mtxvector_base_sdot(&x->a, &y->a, dot, num_flops);
    } else { return MTX_ERR_INVALID_SYMMETRY; }
}

/**
 * ‘mtxmatrix_coo_ddot()’ computes the Frobenius inner product
 * of two matrices in double precision floating point.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_coo_ddot(
    const struct mtxmatrix_coo * x,
    const struct mtxmatrix_coo * y,
    double * dot,
    int64_t * num_flops)
{
    if (x->symmetry == mtx_unsymmetric && y->symmetry == mtx_unsymmetric) {
        return mtxvector_base_ddot(&x->a, &y->a, dot, num_flops);
    } else { return MTX_ERR_INVALID_SYMMETRY; }
}

/**
 * ‘mtxmatrix_coo_cdotu()’ computes the product of the
 * transpose of a complex row matrix with another complex row matrix
 * in single precision floating point, ‘dot := x^T*y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_coo_cdotu(
    const struct mtxmatrix_coo * x,
    const struct mtxmatrix_coo * y,
    float (* dot)[2],
    int64_t * num_flops)
{
    if (x->symmetry == mtx_unsymmetric && y->symmetry == mtx_unsymmetric) {
        return mtxvector_base_cdotu(&x->a, &y->a, dot, num_flops);
    } else { return MTX_ERR_INVALID_SYMMETRY; }
}

/**
 * ‘mtxmatrix_coo_zdotu()’ computes the product of the
 * transpose of a complex row matrix with another complex row matrix
 * in double precision floating point, ‘dot := x^T*y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_coo_zdotu(
    const struct mtxmatrix_coo * x,
    const struct mtxmatrix_coo * y,
    double (* dot)[2],
    int64_t * num_flops)
{
    if (x->symmetry == mtx_unsymmetric && y->symmetry == mtx_unsymmetric) {
        return mtxvector_base_zdotu(&x->a, &y->a, dot, num_flops);
    } else { return MTX_ERR_INVALID_SYMMETRY; }
}

/**
 * ‘mtxmatrix_coo_cdotc()’ computes the Frobenius inner product
 * of two complex matrices in single precision floating point, ‘dot :=
 * x^H*y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_coo_cdotc(
    const struct mtxmatrix_coo * x,
    const struct mtxmatrix_coo * y,
    float (* dot)[2],
    int64_t * num_flops)
{
    if (x->symmetry == mtx_unsymmetric && y->symmetry == mtx_unsymmetric) {
        return mtxvector_base_cdotc(&x->a, &y->a, dot, num_flops);
    } else { return MTX_ERR_INVALID_SYMMETRY; }
}

/**
 * ‘mtxmatrix_coo_zdotc()’ computes the Frobenius inner product
 * of two complex matrices in double precision floating point, ‘dot :=
 * x^H*y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size. Moreover, it is assumed that they have the same underlying
 * sparsity pattern, or else the results are undefined.
 */
int mtxmatrix_coo_zdotc(
    const struct mtxmatrix_coo * x,
    const struct mtxmatrix_coo * y,
    double (* dot)[2],
    int64_t * num_flops)
{
    if (x->symmetry == mtx_unsymmetric && y->symmetry == mtx_unsymmetric) {
        return mtxvector_base_zdotc(&x->a, &y->a, dot, num_flops);
    } else { return MTX_ERR_INVALID_SYMMETRY; }
}

/**
 * ‘mtxmatrix_coo_snrm2()’ computes the Frobenius norm of a
 * matrix in single precision floating point.
 */
int mtxmatrix_coo_snrm2(
    const struct mtxmatrix_coo * x,
    float * nrm2,
    int64_t * num_flops)
{
    if (x->symmetry == mtx_unsymmetric) {
        return mtxvector_base_snrm2(&x->a, nrm2, num_flops);
    } else { return MTX_ERR_INVALID_SYMMETRY; }
}

/**
 * ‘mtxmatrix_coo_dnrm2()’ computes the Frobenius norm of a
 * matrix in double precision floating point.
 */
int mtxmatrix_coo_dnrm2(
    const struct mtxmatrix_coo * x,
    double * nrm2,
    int64_t * num_flops)
{
    if (x->symmetry == mtx_unsymmetric) {
        return mtxvector_base_dnrm2(&x->a, nrm2, num_flops);
    } else { return MTX_ERR_INVALID_SYMMETRY; }
}

/**
 * ‘mtxmatrix_coo_sasum()’ computes the sum of absolute values
 * (1-norm) of a matrix in single precision floating point.  If the
 * matrix is complex-valued, then the sum of the absolute values of
 * the real and imaginary parts is computed.
 */
int mtxmatrix_coo_sasum(
    const struct mtxmatrix_coo * x,
    float * asum,
    int64_t * num_flops)
{
    if (x->symmetry == mtx_unsymmetric) {
        return mtxvector_base_sasum(&x->a, asum, num_flops);
    } else { return MTX_ERR_INVALID_SYMMETRY; }
}

/**
 * ‘mtxmatrix_coo_dasum()’ computes the sum of absolute values
 * (1-norm) of a matrix in double precision floating point.  If the
 * matrix is complex-valued, then the sum of the absolute values of
 * the real and imaginary parts is computed.
 */
int mtxmatrix_coo_dasum(
    const struct mtxmatrix_coo * x,
    double * asum,
    int64_t * num_flops)
{
    if (x->symmetry == mtx_unsymmetric) {
        return mtxvector_base_dasum(&x->a, asum, num_flops);
    } else { return MTX_ERR_INVALID_SYMMETRY; }
}

/**
 * ‘mtxmatrix_coo_iamax()’ finds the index of the first element
 * having the maximum absolute value.  If the matrix is
 * complex-valued, then the index points to the first element having
 * the maximum sum of the absolute values of the real and imaginary
 * parts.
 */
int mtxmatrix_coo_iamax(
    const struct mtxmatrix_coo * x,
    int * iamax)
{
    return mtxvector_base_iamax(&x->a, iamax);
}

/*
 * Level 2 BLAS operations (matrix-vector)
 */

/**
 * ‘mtxmatrix_coo_sgemv()’ multiplies a matrix ‘A’ or its
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
int mtxmatrix_coo_sgemv(
    enum mtxtransposition trans,
    float alpha,
    const struct mtxmatrix_coo * A,
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
 * ‘mtxmatrix_coo_dgemv()’ multiplies a matrix ‘A’ or its
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
int mtxmatrix_coo_dgemv(
    enum mtxtransposition trans,
    double alpha,
    const struct mtxmatrix_coo * A,
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
 * ‘mtxmatrix_coo_cgemv()’ multiplies a complex-valued matrix
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
int mtxmatrix_coo_cgemv(
    enum mtxtransposition trans,
    float alpha[2],
    const struct mtxmatrix_coo * A,
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
 * ‘mtxmatrix_coo_zgemv()’ multiplies a complex-valued matrix
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
int mtxmatrix_coo_zgemv(
    enum mtxtransposition trans,
    double alpha[2],
    const struct mtxmatrix_coo * A,
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
