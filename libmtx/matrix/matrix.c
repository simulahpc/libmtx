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
 * Last modified: 2022-05-21
 *
 * Data structures for matrices.
 */

#include <libmtx/libmtx-config.h>

#include <libmtx/error.h>
#include <libmtx/matrix/matrix.h>
#include <libmtx/mtxfile/mtxfile.h>
#include <libmtx/vector/precision.h>
#include <libmtx/util/partition.h>

#ifdef LIBMTX_HAVE_LIBZ
#include <zlib.h>
#endif

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

/*
 * Matrix types
 */

/**
 * ‘mtxmatrixtype_str()’ is a string representing the matrix type.
 */
const char * mtxmatrixtype_str(
    enum mtxmatrixtype type)
{
    switch (type) {
    case mtxmatrix_blas: return "blas";
    case mtxmatrix_coo: return "coo";
    case mtxmatrix_csr: return "csr";
    case mtxmatrix_dense: return "dense";
    case mtxmatrix_ompcsr: return "ompcsr";
    default: return mtxstrerror(MTX_ERR_INVALID_MATRIX_TYPE);
    }
}

/**
 * ‘mtxmatrixtype_parse()’ parses a string to obtain one of the
 * matrix types of ‘enum mtxmatrixtype’.
 *
 * ‘valid_delimiters’ is either ‘NULL’, in which case it is ignored,
 * or it is a string of characters considered to be valid delimiters
 * for the parsed string.  That is, if there are any remaining,
 * non-NULL characters after parsing, then then the next character is
 * searched for in ‘valid_delimiters’.  If the character is found,
 * then the parsing succeeds and the final delimiter character is
 * consumed by the parser. Otherwise, the parsing fails with an error.
 *
 * If ‘endptr’ is not ‘NULL’, then the address stored in ‘endptr’
 * points to the first character beyond the characters that were
 * consumed during parsing.
 *
 * On success, ‘mtxmatrixtype_parse()’ returns ‘MTX_SUCCESS’ and
 * ‘matrix_type’ is set according to the parsed string and
 * ‘bytes_read’ is set to the number of bytes that were consumed by
 * the parser.  Otherwise, an error code is returned.
 */
int mtxmatrixtype_parse(
    enum mtxmatrixtype * matrix_type,
    int64_t * bytes_read,
    const char ** endptr,
    const char * s,
    const char * valid_delimiters)
{
    const char * t = s;
    if (strncmp("blas", t, strlen("blas")) == 0) {
        t += strlen("blas");
        *matrix_type = mtxmatrix_blas;
    } else if (strncmp("coo", t, strlen("coo")) == 0) {
        t += strlen("coo");
        *matrix_type = mtxmatrix_coo;
    } else if (strncmp("csr", t, strlen("csr")) == 0) {
        t += strlen("csr");
        *matrix_type = mtxmatrix_csr;
    } else if (strncmp("dense", t, strlen("dense")) == 0) {
        t += strlen("dense");
        *matrix_type = mtxmatrix_dense;
    } else if (strncmp("ompcsr", t, strlen("ompcsr")) == 0) {
        t += strlen("ompcsr");
        *matrix_type = mtxmatrix_ompcsr;
    } else { return MTX_ERR_INVALID_MATRIX_TYPE; }
    if (valid_delimiters && *t != '\0') {
        if (!strchr(valid_delimiters, *t))
            return MTX_ERR_INVALID_MATRIX_TYPE;
        t++;
    }
    if (bytes_read) *bytes_read += t-s;
    if (endptr) *endptr = t;
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_field()’ gets the field of a matrix.
 */
int mtxmatrix_field(
    const struct mtxmatrix * A,
    enum mtxfield * field)
{
    if (A->type == mtxmatrix_blas) {
        *field = A->storage.blas.a.base.field;
    } else if (A->type == mtxmatrix_coo) {
        *field = A->storage.coordinate.a.field;
    } else if (A->type == mtxmatrix_dense) {
        *field = A->storage.dense.a.field;
    } else if (A->type == mtxmatrix_csr) {
        *field = A->storage.csr.a.field;
    } else if (A->type == mtxmatrix_ompcsr) {
        *field = A->storage.ompcsr.a.base.field;
    } else { return MTX_ERR_INVALID_MATRIX_TYPE; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_precision()’ gets the precision of a matrix.
 */
int mtxmatrix_precision(
    const struct mtxmatrix * A,
    enum mtxprecision * precision)
{
    if (A->type == mtxmatrix_blas) {
        *precision = A->storage.blas.a.base.precision;
    } else if (A->type == mtxmatrix_coo) {
        *precision = A->storage.coordinate.a.precision;
    } else if (A->type == mtxmatrix_dense) {
        *precision = A->storage.dense.a.precision;
    } else if (A->type == mtxmatrix_csr) {
        *precision = A->storage.csr.a.precision;
    } else if (A->type == mtxmatrix_ompcsr) {
        *precision = A->storage.ompcsr.a.base.precision;
    } else { return MTX_ERR_INVALID_MATRIX_TYPE; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_symmetry()’ gets the symmetry of a matrix.
 */
int mtxmatrix_symmetry(
    const struct mtxmatrix * A,
    enum mtxsymmetry * symmetry)
{
    if (A->type == mtxmatrix_blas) {
        *symmetry = A->storage.blas.symmetry;
    } else if (A->type == mtxmatrix_coo) {
        *symmetry = A->storage.coordinate.symmetry;
    } else if (A->type == mtxmatrix_dense) {
        *symmetry = A->storage.dense.symmetry;
    } else if (A->type == mtxmatrix_csr) {
        *symmetry = A->storage.csr.symmetry;
    } else if (A->type == mtxmatrix_ompcsr) {
        *symmetry = A->storage.ompcsr.symmetry;
    } else { return MTX_ERR_INVALID_MATRIX_TYPE; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_num_nonzeros()’ gets the number of the number of nonzero
 *  matrix entries, including those represented implicitly due to
 *  symmetry.
 */
int mtxmatrix_num_nonzeros(
    const struct mtxmatrix * A,
    int64_t * num_nonzeros)
{
    if (A->type == mtxmatrix_blas) {
        *num_nonzeros = A->storage.blas.num_nonzeros;
    } else if (A->type == mtxmatrix_coo) {
        *num_nonzeros = A->storage.coordinate.num_nonzeros;
    } else if (A->type == mtxmatrix_dense) {
        *num_nonzeros = A->storage.dense.num_nonzeros;
    } else if (A->type == mtxmatrix_csr) {
        *num_nonzeros = A->storage.csr.num_nonzeros;
    } else if (A->type == mtxmatrix_ompcsr) {
        *num_nonzeros = A->storage.ompcsr.num_nonzeros;
    } else { return MTX_ERR_INVALID_MATRIX_TYPE; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_size()’ gets the number of explicitly stored nonzeros of
 * a matrix.
 */
int mtxmatrix_size(
    const struct mtxmatrix * A,
    int64_t * size)
{
    if (A->type == mtxmatrix_blas) {
        *size = A->storage.blas.size;
    } else if (A->type == mtxmatrix_coo) {
        *size = A->storage.coordinate.size;
    } else if (A->type == mtxmatrix_dense) {
        *size = A->storage.dense.size;
    } else if (A->type == mtxmatrix_csr) {
        *size = A->storage.csr.size;
    } else if (A->type == mtxmatrix_ompcsr) {
        *size = A->storage.ompcsr.size;
    } else { return MTX_ERR_INVALID_MATRIX_TYPE; }
    return MTX_SUCCESS;
}

/*
 * Memory management
 */

/**
 * ‘mtxmatrix_free()’ frees storage allocated for a matrix.
 */
void mtxmatrix_free(
    struct mtxmatrix * matrix)
{
    if (matrix->type == mtxmatrix_blas) {
        mtxmatrix_blas_free(&matrix->storage.blas);
    } else if (matrix->type == mtxmatrix_coo) {
        mtxmatrix_coo_free(&matrix->storage.coordinate);
    } else if (matrix->type == mtxmatrix_csr) {
        mtxmatrix_csr_free(&matrix->storage.csr);
    } else if (matrix->type == mtxmatrix_dense) {
        mtxmatrix_dense_free(&matrix->storage.dense);
    } else if (matrix->type == mtxmatrix_ompcsr) {
        mtxmatrix_ompcsr_free(&matrix->storage.ompcsr);
    }
}

/**
 * ‘mtxmatrix_alloc_copy()’ allocates a copy of a matrix without
 * initialising the values.
 */
int mtxmatrix_alloc_copy(
    struct mtxmatrix * dst,
    const struct mtxmatrix * src)
{
    if (src->type == mtxmatrix_blas) {
        return mtxmatrix_blas_alloc_copy(
            &dst->storage.blas, &src->storage.blas);
    } else if (src->type == mtxmatrix_coo) {
        return mtxmatrix_coo_alloc_copy(
            &dst->storage.coordinate, &src->storage.coordinate);
    } else if (src->type == mtxmatrix_csr) {
        return mtxmatrix_csr_alloc_copy(
            &dst->storage.csr, &src->storage.csr);
    } else if (src->type == mtxmatrix_dense) {
        return mtxmatrix_dense_alloc_copy(
            &dst->storage.dense, &src->storage.dense);
    } else if (src->type == mtxmatrix_ompcsr) {
        return mtxmatrix_ompcsr_alloc_copy(
            &dst->storage.ompcsr, &src->storage.ompcsr);
    } else { return MTX_ERR_INVALID_MATRIX_TYPE; }
}

/**
 * ‘mtxmatrix_init_copy()’ allocates a copy of a matrix and also
 * copies the values.
 */
int mtxmatrix_init_copy(
    struct mtxmatrix * dst,
    const struct mtxmatrix * src)
{
    if (src->type == mtxmatrix_blas) {
        return mtxmatrix_blas_init_copy(
            &dst->storage.blas, &src->storage.blas);
    } else if (src->type == mtxmatrix_coo) {
        return mtxmatrix_coo_init_copy(
            &dst->storage.coordinate, &src->storage.coordinate);
    } else if (src->type == mtxmatrix_csr) {
        return mtxmatrix_csr_init_copy(
            &dst->storage.csr, &src->storage.csr);
    } else if (src->type == mtxmatrix_dense) {
        return mtxmatrix_dense_init_copy(
            &dst->storage.dense, &src->storage.dense);
    } else if (src->type == mtxmatrix_ompcsr) {
        return mtxmatrix_ompcsr_init_copy(
            &dst->storage.ompcsr, &src->storage.ompcsr);
    } else { return MTX_ERR_INVALID_MATRIX_TYPE; }
}

/*
 * initialise matrices from data in coordinate format
 */

/**
 * ‘mtxmatrix_alloc_entries()’ allocates storage for a matrix based on
 * entrywise data in coordinate format.
 */
int mtxmatrix_alloc_entries(
    struct mtxmatrix * A,
    enum mtxmatrixtype type,
    enum mtxfield field,
    enum mtxprecision precision,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int * rowidx,
    const int * colidx)
{
    if (type == mtxmatrix_blas) {
        A->type = mtxmatrix_blas;
        return mtxmatrix_blas_alloc_entries(
            &A->storage.blas, field, precision, symmetry,
            num_rows, num_columns, num_nonzeros,
            idxstride, idxbase, rowidx, colidx);
    } else if (type == mtxmatrix_coo) {
        A->type = mtxmatrix_coo;
        return mtxmatrix_coo_alloc_entries(
            &A->storage.coordinate, field, precision, symmetry,
            num_rows, num_columns, num_nonzeros,
            idxstride, idxbase, rowidx, colidx);
    } else if (type == mtxmatrix_csr) {
        A->type = mtxmatrix_csr;
        return mtxmatrix_csr_alloc_entries(
            &A->storage.csr, field, precision, symmetry,
            num_rows, num_columns, num_nonzeros,
            idxstride, idxbase, rowidx, colidx, NULL);
    } else if (type == mtxmatrix_dense) {
        A->type = mtxmatrix_dense;
        return mtxmatrix_dense_alloc_entries(
            &A->storage.dense, field, precision, symmetry,
            num_rows, num_columns, num_nonzeros,
            idxstride, idxbase, rowidx, colidx);
    } else if (type == mtxmatrix_ompcsr) {
        A->type = mtxmatrix_ompcsr;
        return mtxmatrix_ompcsr_alloc_entries(
            &A->storage.ompcsr, field, precision, symmetry,
            num_rows, num_columns, num_nonzeros,
            idxstride, idxbase, rowidx, colidx, NULL);
    } else { return MTX_ERR_INVALID_MATRIX_TYPE; }
}

/**
 * ‘mtxmatrix_init_entries_real_single()’ allocates and initialises
 * a matrix from data in coordinate format with real, single precision
 * coefficients.
 */
int mtxmatrix_init_entries_real_single(
    struct mtxmatrix * A,
    enum mtxmatrixtype type,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_nonzeros,
    const int * rowidx,
    const int * colidx,
    const float * data)
{
    if (type == mtxmatrix_blas) {
        A->type = mtxmatrix_blas;
        return mtxmatrix_blas_init_entries_real_single(
            &A->storage.blas, symmetry,
            num_rows, num_columns, num_nonzeros, rowidx, colidx, data);
    } else if (type == mtxmatrix_coo) {
        A->type = mtxmatrix_coo;
        return mtxmatrix_coo_init_entries_real_single(
            &A->storage.coordinate, symmetry,
            num_rows, num_columns, num_nonzeros, rowidx, colidx, data);
    } else if (type == mtxmatrix_csr) {
        A->type = mtxmatrix_csr;
        return mtxmatrix_csr_init_entries_real_single(
            &A->storage.csr, symmetry,
            num_rows, num_columns, num_nonzeros, rowidx, colidx, data);
    } else if (type == mtxmatrix_dense) {
        A->type = mtxmatrix_dense;
        return mtxmatrix_dense_init_entries_real_single(
            &A->storage.dense, symmetry,
            num_rows, num_columns, num_nonzeros, rowidx, colidx, data);
    } else if (type == mtxmatrix_ompcsr) {
        A->type = mtxmatrix_ompcsr;
        return mtxmatrix_ompcsr_init_entries_real_single(
            &A->storage.ompcsr, symmetry,
            num_rows, num_columns, num_nonzeros, rowidx, colidx, data);
    } else { return MTX_ERR_INVALID_MATRIX_TYPE; }
}

/**
 * ‘mtxmatrix_init_entries_real_double()’ allocates and initialises
 * a matrix from data in coordinate format with real, double precision
 * coefficients.
 */
int mtxmatrix_init_entries_real_double(
    struct mtxmatrix * A,
    enum mtxmatrixtype type,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_nonzeros,
    const int * rowidx,
    const int * colidx,
    const double * data)
{
    if (type == mtxmatrix_blas) {
        A->type = mtxmatrix_blas;
        return mtxmatrix_blas_init_entries_real_double(
            &A->storage.blas, symmetry,
            num_rows, num_columns, num_nonzeros, rowidx, colidx, data);
    } else if (type == mtxmatrix_coo) {
        A->type = mtxmatrix_coo;
        return mtxmatrix_coo_init_entries_real_double(
            &A->storage.coordinate, symmetry,
            num_rows, num_columns, num_nonzeros, rowidx, colidx, data);
    } else if (type == mtxmatrix_csr) {
        A->type = mtxmatrix_csr;
        return mtxmatrix_csr_init_entries_real_double(
            &A->storage.csr, symmetry,
            num_rows, num_columns, num_nonzeros, rowidx, colidx, data);
    } else if (type == mtxmatrix_dense) {
        A->type = mtxmatrix_dense;
        return mtxmatrix_dense_init_entries_real_double(
            &A->storage.dense, symmetry,
            num_rows, num_columns, num_nonzeros, rowidx, colidx, data);
    } else if (type == mtxmatrix_ompcsr) {
        A->type = mtxmatrix_ompcsr;
        return mtxmatrix_ompcsr_init_entries_real_double(
            &A->storage.ompcsr, symmetry,
            num_rows, num_columns, num_nonzeros, rowidx, colidx, data);
    } else { return MTX_ERR_INVALID_MATRIX_TYPE; }
}

/**
 * ‘mtxmatrix_init_entries_complex_single()’ allocates and
 * initialises a matrix from data in coordinate format with complex,
 * single precision coefficients.
 */
int mtxmatrix_init_entries_complex_single(
    struct mtxmatrix * A,
    enum mtxmatrixtype type,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_nonzeros,
    const int * rowidx,
    const int * colidx,
    const float (* data)[2])
{
    if (type == mtxmatrix_blas) {
        A->type = mtxmatrix_blas;
        return mtxmatrix_blas_init_entries_complex_single(
            &A->storage.blas, symmetry,
            num_rows, num_columns, num_nonzeros, rowidx, colidx, data);
    } else if (type == mtxmatrix_coo) {
        A->type = mtxmatrix_coo;
        return mtxmatrix_coo_init_entries_complex_single(
            &A->storage.coordinate, symmetry,
            num_rows, num_columns, num_nonzeros, rowidx, colidx, data);
    } else if (type == mtxmatrix_csr) {
        A->type = mtxmatrix_csr;
        return mtxmatrix_csr_init_entries_complex_single(
            &A->storage.csr, symmetry,
            num_rows, num_columns, num_nonzeros, rowidx, colidx, data);
    } else if (type == mtxmatrix_dense) {
        A->type = mtxmatrix_dense;
        return mtxmatrix_dense_init_entries_complex_single(
            &A->storage.dense, symmetry,
            num_rows, num_columns, num_nonzeros, rowidx, colidx, data);
    } else if (type == mtxmatrix_ompcsr) {
        A->type = mtxmatrix_ompcsr;
        return mtxmatrix_ompcsr_init_entries_complex_single(
            &A->storage.ompcsr, symmetry,
            num_rows, num_columns, num_nonzeros, rowidx, colidx, data);
    } else { return MTX_ERR_INVALID_MATRIX_TYPE; }
}

/**
 * ‘mtxmatrix_init_entries_complex_double()’ allocates and
 * initialises a matrix from data in coordinate format with complex,
 * double precision coefficients.
 */
int mtxmatrix_init_entries_complex_double(
    struct mtxmatrix * A,
    enum mtxmatrixtype type,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_nonzeros,
    const int * rowidx,
    const int * colidx,
    const double (* data)[2])
{
    if (type == mtxmatrix_blas) {
        A->type = mtxmatrix_blas;
        return mtxmatrix_blas_init_entries_complex_double(
            &A->storage.blas, symmetry,
            num_rows, num_columns, num_nonzeros, rowidx, colidx, data);
    } else if (type == mtxmatrix_coo) {
        A->type = mtxmatrix_coo;
        return mtxmatrix_coo_init_entries_complex_double(
            &A->storage.coordinate, symmetry,
            num_rows, num_columns, num_nonzeros, rowidx, colidx, data);
    } else if (type == mtxmatrix_csr) {
        A->type = mtxmatrix_csr;
        return mtxmatrix_csr_init_entries_complex_double(
            &A->storage.csr, symmetry,
            num_rows, num_columns, num_nonzeros, rowidx, colidx, data);
    } else if (type == mtxmatrix_dense) {
        A->type = mtxmatrix_dense;
        return mtxmatrix_dense_init_entries_complex_double(
            &A->storage.dense, symmetry,
            num_rows, num_columns, num_nonzeros, rowidx, colidx, data);
    } else if (type == mtxmatrix_ompcsr) {
        A->type = mtxmatrix_ompcsr;
        return mtxmatrix_ompcsr_init_entries_complex_double(
            &A->storage.ompcsr, symmetry,
            num_rows, num_columns, num_nonzeros, rowidx, colidx, data);
    } else { return MTX_ERR_INVALID_MATRIX_TYPE; }
}

/**
 * ‘mtxmatrix_init_entries_integer_single()’ allocates and
 * initialises a matrix from data in coordinate format with integer,
 * single precision coefficients.
 */
int mtxmatrix_init_entries_integer_single(
    struct mtxmatrix * A,
    enum mtxmatrixtype type,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_nonzeros,
    const int * rowidx,
    const int * colidx,
    const int32_t * data)
{
    if (type == mtxmatrix_blas) {
        A->type = mtxmatrix_blas;
        return mtxmatrix_blas_init_entries_integer_single(
            &A->storage.blas, symmetry,
            num_rows, num_columns, num_nonzeros, rowidx, colidx, data);
    } else if (type == mtxmatrix_coo) {
        A->type = mtxmatrix_coo;
        return mtxmatrix_coo_init_entries_integer_single(
            &A->storage.coordinate, symmetry,
            num_rows, num_columns, num_nonzeros, rowidx, colidx, data);
    } else if (type == mtxmatrix_csr) {
        A->type = mtxmatrix_csr;
        return mtxmatrix_csr_init_entries_integer_single(
            &A->storage.csr, symmetry,
            num_rows, num_columns, num_nonzeros, rowidx, colidx, data);
    } else if (type == mtxmatrix_dense) {
        A->type = mtxmatrix_dense;
        return mtxmatrix_dense_init_entries_integer_single(
            &A->storage.dense, symmetry,
            num_rows, num_columns, num_nonzeros, rowidx, colidx, data);
    } else if (type == mtxmatrix_ompcsr) {
        A->type = mtxmatrix_ompcsr;
        return mtxmatrix_ompcsr_init_entries_integer_single(
            &A->storage.ompcsr, symmetry,
            num_rows, num_columns, num_nonzeros, rowidx, colidx, data);
    } else { return MTX_ERR_INVALID_MATRIX_TYPE; }
}

/**
 * ‘mtxmatrix_init_entries_integer_double()’ allocates and
 * initialises a matrix from data in coordinate format with integer,
 * double precision coefficients.
 */
int mtxmatrix_init_entries_integer_double(
    struct mtxmatrix * A,
    enum mtxmatrixtype type,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_nonzeros,
    const int * rowidx,
    const int * colidx,
    const int64_t * data)
{
    if (type == mtxmatrix_blas) {
        A->type = mtxmatrix_blas;
        return mtxmatrix_blas_init_entries_integer_double(
            &A->storage.blas, symmetry,
            num_rows, num_columns, num_nonzeros, rowidx, colidx, data);
    } else if (type == mtxmatrix_coo) {
        A->type = mtxmatrix_coo;
        return mtxmatrix_coo_init_entries_integer_double(
            &A->storage.coordinate, symmetry,
            num_rows, num_columns, num_nonzeros, rowidx, colidx, data);
    } else if (type == mtxmatrix_csr) {
        A->type = mtxmatrix_csr;
        return mtxmatrix_csr_init_entries_integer_double(
            &A->storage.csr, symmetry,
            num_rows, num_columns, num_nonzeros, rowidx, colidx, data);
    } else if (type == mtxmatrix_dense) {
        A->type = mtxmatrix_dense;
        return mtxmatrix_dense_init_entries_integer_double(
            &A->storage.dense, symmetry,
            num_rows, num_columns, num_nonzeros, rowidx, colidx, data);
    } else if (type == mtxmatrix_ompcsr) {
        A->type = mtxmatrix_ompcsr;
        return mtxmatrix_ompcsr_init_entries_integer_double(
            &A->storage.ompcsr, symmetry,
            num_rows, num_columns, num_nonzeros, rowidx, colidx, data);
    } else { return MTX_ERR_INVALID_MATRIX_TYPE; }
}

/**
 * ‘mtxmatrix_init_entries_pattern()’ allocates and initialises a
 * matrix from data in coordinate format with integer, double
 * precision coefficients.
 */
int mtxmatrix_init_entries_pattern(
    struct mtxmatrix * A,
    enum mtxmatrixtype type,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_nonzeros,
    const int * rowidx,
    const int * colidx)
{
    if (type == mtxmatrix_blas) {
        A->type = mtxmatrix_blas;
        return mtxmatrix_blas_init_entries_pattern(
            &A->storage.blas, symmetry,
            num_rows, num_columns, num_nonzeros, rowidx, colidx);
    } else if (type == mtxmatrix_coo) {
        A->type = mtxmatrix_coo;
        return mtxmatrix_coo_init_entries_pattern(
            &A->storage.coordinate, symmetry,
            num_rows, num_columns, num_nonzeros, rowidx, colidx);
    } else if (type == mtxmatrix_csr) {
        A->type = mtxmatrix_csr;
        return mtxmatrix_csr_init_entries_pattern(
            &A->storage.csr, symmetry,
            num_rows, num_columns, num_nonzeros, rowidx, colidx);
    } else if (type == mtxmatrix_dense) {
        A->type = mtxmatrix_dense;
        return mtxmatrix_dense_init_entries_pattern(
            &A->storage.dense, symmetry,
            num_rows, num_columns, num_nonzeros, rowidx, colidx);
    } else if (type == mtxmatrix_ompcsr) {
        A->type = mtxmatrix_ompcsr;
        return mtxmatrix_ompcsr_init_entries_pattern(
            &A->storage.ompcsr, symmetry,
            num_rows, num_columns, num_nonzeros, rowidx, colidx);
    } else { return MTX_ERR_INVALID_MATRIX_TYPE; }
}

/*
 * initialise matrices from strided data in coordinate format
 */

/**
 * ‘mtxmatrix_init_entries_strided_real_single()’ allocates and initialises
 * a matrix from data in coordinate format with real, single precision
 * coefficients.
 */
int mtxmatrix_init_entries_strided_real_single(
    struct mtxmatrix * A,
    enum mtxmatrixtype type,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int * rowidx,
    const int * colidx,
    int datastride,
    const float * data)
{
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_init_entries_strided_real_double()’ allocates and initialises
 * a matrix from data in coordinate format with real, double precision
 * coefficients.
 */
int mtxmatrix_init_entries_strided_real_double(
    struct mtxmatrix * A,
    enum mtxmatrixtype type,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int * rowidx,
    const int * colidx,
    int datastride,
    const double * data)
{
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_init_entries_strided_complex_single()’ allocates and
 * initialises a matrix from data in coordinate format with complex,
 * single precision coefficients.
 */
int mtxmatrix_init_entries_strided_complex_single(
    struct mtxmatrix * A,
    enum mtxmatrixtype type,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int * rowidx,
    const int * colidx,
    int datastride,
    const float (* data)[2])
{
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_init_entries_strided_complex_double()’ allocates and
 * initialises a matrix from data in coordinate format with complex,
 * double precision coefficients.
 */
int mtxmatrix_init_entries_strided_complex_double(
    struct mtxmatrix * A,
    enum mtxmatrixtype type,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int * rowidx,
    const int * colidx,
    int datastride,
    const double (* data)[2])
{
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_init_entries_strided_integer_single()’ allocates and
 * initialises a matrix from data in coordinate format with integer,
 * single precision coefficients.
 */
int mtxmatrix_init_entries_strided_integer_single(
    struct mtxmatrix * A,
    enum mtxmatrixtype type,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int * rowidx,
    const int * colidx,
    int datastride,
    const int32_t * data)
{
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_init_entries_strided_integer_double()’ allocates and
 * initialises a matrix from data in coordinate format with integer,
 * double precision coefficients.
 */
int mtxmatrix_init_entries_strided_integer_double(
    struct mtxmatrix * A,
    enum mtxmatrixtype type,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int * rowidx,
    const int * colidx,
    int datastride,
    const int64_t * data)
{
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_init_entries_strided_pattern()’ allocates and initialises a
 * matrix from data in coordinate format with integer, double
 * precision coefficients.
 */
int mtxmatrix_init_entries_strided_pattern(
    struct mtxmatrix * A,
    enum mtxmatrixtype type,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int * rowidx,
    const int * colidx)
{
    return MTX_SUCCESS;
}

/*
 * modifying values
 */

/**
 * ‘mtxmatrix_setzero()’ sets every value of a matrix to zero.
 */
int mtxmatrix_setzero(
    struct mtxmatrix * A)
{
    if (A->type == mtxmatrix_blas) {
        return mtxmatrix_blas_setzero(&A->storage.blas);
    } else if (A->type == mtxmatrix_coo) {
        return mtxmatrix_coo_setzero(&A->storage.coordinate);
    } else if (A->type == mtxmatrix_csr) {
        return mtxmatrix_csr_setzero(&A->storage.csr);
    } else if (A->type == mtxmatrix_dense) {
        return mtxmatrix_dense_setzero(&A->storage.dense);
    } else if (A->type == mtxmatrix_ompcsr) {
        return mtxmatrix_ompcsr_setzero(&A->storage.ompcsr);
    } else { return MTX_ERR_INVALID_MATRIX_TYPE; }
}

/**
 * ‘mtxmatrix_set_real_single()’ sets values of a matrix based on an
 * array of single precision floating point numbers.
 */
int mtxmatrix_set_real_single(
    struct mtxmatrix * A,
    int64_t size,
    int stride,
    const float * a)
{
    if (A->type == mtxmatrix_blas) {
        return mtxmatrix_blas_set_real_single(
            &A->storage.blas, size, stride, a);
    } else if (A->type == mtxmatrix_coo) {
        return mtxmatrix_coo_set_real_single(
            &A->storage.coordinate, size, stride, a);
    } else if (A->type == mtxmatrix_csr) {
        return mtxmatrix_csr_set_real_single(
            &A->storage.csr, size, stride, a);
    } else if (A->type == mtxmatrix_dense) {
        return mtxmatrix_dense_set_real_single(
            &A->storage.dense, size, stride, a);
    } else if (A->type == mtxmatrix_ompcsr) {
        return mtxmatrix_ompcsr_set_real_single(
            &A->storage.ompcsr, size, stride, a);
    } else { return MTX_ERR_INVALID_MATRIX_TYPE; }
}

/**
 * ‘mtxmatrix_set_real_double()’ sets values of a matrix based on an
 * array of double precision floating point numbers.
 */
int mtxmatrix_set_real_double(
    struct mtxmatrix * A,
    int64_t size,
    int stride,
    const double * a)
{
    if (A->type == mtxmatrix_blas) {
        return mtxmatrix_blas_set_real_double(
            &A->storage.blas, size, stride, a);
    } else if (A->type == mtxmatrix_coo) {
        return mtxmatrix_coo_set_real_double(
            &A->storage.coordinate, size, stride, a);
    } else if (A->type == mtxmatrix_csr) {
        return mtxmatrix_csr_set_real_double(
            &A->storage.csr, size, stride, a);
    } else if (A->type == mtxmatrix_dense) {
        return mtxmatrix_dense_set_real_double(
            &A->storage.dense, size, stride, a);
    } else if (A->type == mtxmatrix_ompcsr) {
        return mtxmatrix_ompcsr_set_real_double(
            &A->storage.ompcsr, size, stride, a);
    } else { return MTX_ERR_INVALID_MATRIX_TYPE; }
}

/**
 * ‘mtxmatrix_set_complex_single()’ sets values of a matrix based on
 * an array of single precision floating point complex numbers.
 */
int mtxmatrix_set_complex_single(
    struct mtxmatrix * A,
    int64_t size,
    int stride,
    const float (*a)[2])
{
    if (A->type == mtxmatrix_blas) {
        return mtxmatrix_blas_set_complex_single(
            &A->storage.blas, size, stride, a);
    } else if (A->type == mtxmatrix_coo) {
        return mtxmatrix_coo_set_complex_single(
            &A->storage.coordinate, size, stride, a);
    } else if (A->type == mtxmatrix_csr) {
        return mtxmatrix_csr_set_complex_single(
            &A->storage.csr, size, stride, a);
    } else if (A->type == mtxmatrix_dense) {
        return mtxmatrix_dense_set_complex_single(
            &A->storage.dense, size, stride, a);
    } else if (A->type == mtxmatrix_ompcsr) {
        return mtxmatrix_ompcsr_set_complex_single(
            &A->storage.ompcsr, size, stride, a);
    } else { return MTX_ERR_INVALID_MATRIX_TYPE; }
}

/**
 * ‘mtxmatrix_set_complex_double()’ sets values of a matrix based on
 * an array of double precision floating point complex numbers.
 */
int mtxmatrix_set_complex_double(
    struct mtxmatrix * A,
    int64_t size,
    int stride,
    const double (*a)[2])
{
    if (A->type == mtxmatrix_blas) {
        return mtxmatrix_blas_set_complex_double(
            &A->storage.blas, size, stride, a);
    } else if (A->type == mtxmatrix_coo) {
        return mtxmatrix_coo_set_complex_double(
            &A->storage.coordinate, size, stride, a);
    } else if (A->type == mtxmatrix_csr) {
        return mtxmatrix_csr_set_complex_double(
            &A->storage.csr, size, stride, a);
    } else if (A->type == mtxmatrix_dense) {
        return mtxmatrix_dense_set_complex_double(
            &A->storage.dense, size, stride, a);
    } else if (A->type == mtxmatrix_ompcsr) {
        return mtxmatrix_ompcsr_set_complex_double(
            &A->storage.ompcsr, size, stride, a);
    } else { return MTX_ERR_INVALID_MATRIX_TYPE; }
}

/**
 * ‘mtxmatrix_set_integer_single()’ sets values of a matrix based on
 * an array of integers.
 */
int mtxmatrix_set_integer_single(
    struct mtxmatrix * A,
    int64_t size,
    int stride,
    const int32_t * a)
{
    if (A->type == mtxmatrix_blas) {
        return mtxmatrix_blas_set_integer_single(
            &A->storage.blas, size, stride, a);
    } else if (A->type == mtxmatrix_coo) {
        return mtxmatrix_coo_set_integer_single(
            &A->storage.coordinate, size, stride, a);
    } else if (A->type == mtxmatrix_csr) {
        return mtxmatrix_csr_set_integer_single(
            &A->storage.csr, size, stride, a);
    } else if (A->type == mtxmatrix_dense) {
        return mtxmatrix_dense_set_integer_single(
            &A->storage.dense, size, stride, a);
    } else if (A->type == mtxmatrix_ompcsr) {
        return mtxmatrix_ompcsr_set_integer_single(
            &A->storage.ompcsr, size, stride, a);
    } else { return MTX_ERR_INVALID_MATRIX_TYPE; }
}

/**
 * ‘mtxmatrix_set_integer_double()’ sets values of a matrix based on
 * an array of integers.
 */
int mtxmatrix_set_integer_double(
    struct mtxmatrix * A,
    int64_t size,
    int stride,
    const int64_t * a)
{
    if (A->type == mtxmatrix_blas) {
        return mtxmatrix_blas_set_integer_double(
            &A->storage.blas, size, stride, a);
    } else if (A->type == mtxmatrix_coo) {
        return mtxmatrix_coo_set_integer_double(
            &A->storage.coordinate, size, stride, a);
    } else if (A->type == mtxmatrix_csr) {
        return mtxmatrix_csr_set_integer_double(
            &A->storage.csr, size, stride, a);
    } else if (A->type == mtxmatrix_dense) {
        return mtxmatrix_dense_set_integer_double(
            &A->storage.dense, size, stride, a);
    } else if (A->type == mtxmatrix_ompcsr) {
        return mtxmatrix_ompcsr_set_integer_double(
            &A->storage.ompcsr, size, stride, a);
    } else { return MTX_ERR_INVALID_MATRIX_TYPE; }
}

/*
 * Row and columns vectors
 */

/**
 * ‘mtxmatrix_alloc_row_vector()’ allocates a row vector for a given
 * matrix, where a row vector is a vector whose length equal to a
 * single row of the matrix.
 */
int mtxmatrix_alloc_row_vector(
    const struct mtxmatrix * matrix,
    struct mtxvector * vector,
    enum mtxvectortype vectortype)
{
    if (matrix->type == mtxmatrix_blas) {
        return mtxmatrix_blas_alloc_row_vector(
            &matrix->storage.blas, vector, vectortype);
    } else if (matrix->type == mtxmatrix_coo) {
        return mtxmatrix_coo_alloc_row_vector(
            &matrix->storage.coordinate, vector, vectortype);
    } else if (matrix->type == mtxmatrix_csr) {
        return mtxmatrix_csr_alloc_row_vector(
            &matrix->storage.csr, vector, vectortype);
    } else if (matrix->type == mtxmatrix_dense) {
        return mtxmatrix_dense_alloc_row_vector(
            &matrix->storage.dense, vector, vectortype);
    } else if (matrix->type == mtxmatrix_ompcsr) {
        return mtxmatrix_ompcsr_alloc_row_vector(
            &matrix->storage.ompcsr, vector, vectortype);
    } else { return MTX_ERR_INVALID_MATRIX_TYPE; }
}

/**
 * ‘mtxmatrix_alloc_column_vector()’ allocates a column vector for a
 * given matrix, where a column vector is a vector whose length equal
 * to a single column of the matrix.
 */
int mtxmatrix_alloc_column_vector(
    const struct mtxmatrix * matrix,
    struct mtxvector * vector,
    enum mtxvectortype vectortype)
{
    if (matrix->type == mtxmatrix_blas) {
        return mtxmatrix_blas_alloc_column_vector(
            &matrix->storage.blas, vector, vectortype);
    } else if (matrix->type == mtxmatrix_coo) {
        return mtxmatrix_coo_alloc_column_vector(
            &matrix->storage.coordinate, vector, vectortype);
    } else if (matrix->type == mtxmatrix_csr) {
        return mtxmatrix_csr_alloc_column_vector(
            &matrix->storage.csr, vector, vectortype);
    } else if (matrix->type == mtxmatrix_dense) {
        return mtxmatrix_dense_alloc_column_vector(
            &matrix->storage.dense, vector, vectortype);
    } else if (matrix->type == mtxmatrix_ompcsr) {
        return mtxmatrix_ompcsr_alloc_column_vector(
            &matrix->storage.ompcsr, vector, vectortype);
    } else { return MTX_ERR_INVALID_MATRIX_TYPE; }
}

/*
 * Convert to and from Matrix Market format
 */

/**
 * ‘mtxmatrix_from_mtxfile()’ converts to a matrix from Matrix Market
 * format.
 */
int mtxmatrix_from_mtxfile(
    struct mtxmatrix * matrix,
    enum mtxmatrixtype type,
    const struct mtxfile * mtxfile)
{
    if (type == mtxmatrix_blas) {
        matrix->type = mtxmatrix_blas;
        return mtxmatrix_blas_from_mtxfile(
            &matrix->storage.blas, mtxfile);
    } else if (type == mtxmatrix_coo) {
        matrix->type = mtxmatrix_coo;
        return mtxmatrix_coo_from_mtxfile(
            &matrix->storage.coordinate, mtxfile);
    } else if (type == mtxmatrix_csr) {
        matrix->type = mtxmatrix_csr;
        return mtxmatrix_csr_from_mtxfile(
            &matrix->storage.csr, mtxfile);
    } else if (type == mtxmatrix_dense) {
        matrix->type = mtxmatrix_dense;
        return mtxmatrix_dense_from_mtxfile(
            &matrix->storage.dense, mtxfile);
    } else if (type == mtxmatrix_ompcsr) {
        matrix->type = mtxmatrix_ompcsr;
        return mtxmatrix_ompcsr_from_mtxfile(
            &matrix->storage.ompcsr, mtxfile);
    } else { return MTX_ERR_INVALID_MATRIX_TYPE; }
}

/**
 * ‘mtxmatrix_to_mtxfile()’ converts a matrix to Matrix Market format.
 */
int mtxmatrix_to_mtxfile(
    struct mtxfile * dst,
    const struct mtxmatrix * src,
    int64_t num_rows,
    const int64_t * rowidx,
    int64_t num_columns,
    const int64_t * colidx,
    enum mtxfileformat mtxfmt)
{
    if (src->type == mtxmatrix_blas) {
        return mtxmatrix_blas_to_mtxfile(
            dst, &src->storage.blas,
            num_rows, rowidx, num_columns, colidx, mtxfmt);
    } else if (src->type == mtxmatrix_coo) {
        return mtxmatrix_coo_to_mtxfile(
            dst, &src->storage.coordinate,
            num_rows, rowidx, num_columns, colidx, mtxfmt);
    } else if (src->type == mtxmatrix_csr) {
        return mtxmatrix_csr_to_mtxfile(
            dst, &src->storage.csr,
            num_rows, rowidx, num_columns, colidx, mtxfmt);
    } else if (src->type == mtxmatrix_dense) {
        return mtxmatrix_dense_to_mtxfile(
            dst, &src->storage.dense,
            num_rows, rowidx, num_columns, colidx, mtxfmt);
    } else if (src->type == mtxmatrix_ompcsr) {
        return mtxmatrix_ompcsr_to_mtxfile(
            dst, &src->storage.ompcsr,
            num_rows, rowidx, num_columns, colidx, mtxfmt);
    } else { return MTX_ERR_INVALID_MATRIX_TYPE; }
}

/*
 * I/O functions
 */

/**
 * ‘mtxmatrix_read()’ reads a matrix from a Matrix Market file.  The
 * file may optionally be compressed by gzip.
 *
 * The ‘precision’ argument specifies which precision to use for
 * storing matrix or matrix values.
 *
 * The ‘type’ argument specifies which format to use for representing
 * the matrix.
 *
 * If ‘path’ is ‘-’, then standard input is used.
 *
 * The file is assumed to be gzip-compressed if ‘gzip’ is ‘true’, and
 * uncompressed otherwise.
 *
 * If an error code is returned, then ‘lines_read’ and ‘bytes_read’
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the matrix.
 */
int mtxmatrix_read(
    struct mtxmatrix * matrix,
    enum mtxprecision precision,
    enum mtxmatrixtype type,
    const char * path,
    bool gzip,
    int64_t * lines_read,
    int64_t * bytes_read)
{
    struct mtxfile mtxfile;
    int err = mtxfile_read(
        &mtxfile, precision, path, gzip, lines_read, bytes_read);
    if (err) return err;
    err = mtxmatrix_from_mtxfile(matrix, type, &mtxfile);
    if (err) {
        mtxfile_free(&mtxfile);
        return err;
    }
    mtxfile_free(&mtxfile);
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_fread()’ reads a matrix from a stream in Matrix Market
 * format.
 *
 * ‘precision’ is used to determine the precision to use for storing
 * the values of matrix or matrix entries.
 *
 * The ‘type’ argument specifies which format to use for representing
 * the matrix.
 *
 * If an error code is returned, then ‘lines_read’ and ‘bytes_read’
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the matrix.
 */
int mtxmatrix_fread(
    struct mtxmatrix * matrix,
    enum mtxprecision precision,
    enum mtxmatrixtype type,
    FILE * f,
    int64_t * lines_read,
    int64_t * bytes_read,
    size_t line_max,
    char * linebuf)
{
    struct mtxfile mtxfile;
    int err = mtxfile_fread(
        &mtxfile, precision, f, lines_read, bytes_read, line_max, linebuf);
    if (err) return err;
    err = mtxmatrix_from_mtxfile(matrix, type, &mtxfile);
    if (err) {
        mtxfile_free(&mtxfile);
        return err;
    }
    mtxfile_free(&mtxfile);
    return MTX_SUCCESS;
}

#ifdef LIBMTX_HAVE_LIBZ
/**
 * ‘mtxmatrix_gzread()’ reads a matrix from a gzip-compressed stream.
 *
 * ‘precision’ is used to determine the precision to use for storing
 * the values of matrix or matrix entries.
 *
 * The ‘type’ argument specifies which format to use for representing
 * the matrix.
 *
 * If an error code is returned, then ‘lines_read’ and ‘bytes_read’
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the matrix.
 */
int mtxmatrix_gzread(
    struct mtxmatrix * matrix,
    enum mtxprecision precision,
    enum mtxmatrixtype type,
    gzFile f,
    int64_t * lines_read,
    int64_t * bytes_read,
    size_t line_max,
    char * linebuf)
{
    struct mtxfile mtxfile;
    int err = mtxfile_gzread(
        &mtxfile, precision, f, lines_read, bytes_read, line_max, linebuf);
    if (err)
        return err;
    err = mtxmatrix_from_mtxfile(matrix, type, &mtxfile);
    if (err) {
        mtxfile_free(&mtxfile);
        return err;
    }
    mtxfile_free(&mtxfile);
    return MTX_SUCCESS;
}
#endif

/**
 * ‘mtxmatrix_write()’ writes a matrix to a Matrix Market file. The
 * file may optionally be compressed by gzip.
 *
 * If ‘path’ is ‘-’, then standard output is used.
 *
 * If ‘fmt’ is ‘NULL’, then the format specifier ‘%g’ is used to print
 * floating point numbers with enough digits to ensure correct
 * round-trip conversion from decimal text and back.  Otherwise, the
 * given format string is used to print numerical values.
 *
 * The format string follows the conventions of ‘printf’. If the field
 * is ‘real’ or ‘complex’, then the format specifiers '%e', '%E',
 * '%f', '%F', '%g' or '%G' may be used. If the field is ‘integer’,
 * then the format specifier must be '%d'. The format string is
 * ignored if the field is ‘pattern’. Field width and precision may be
 * specified (e.g., "%3.1f"), but variable field width and precision
 * (e.g., "%*.*f"), as well as length modifiers (e.g., "%Lf") are not
 * allowed.
 */
int mtxmatrix_write(
    const struct mtxmatrix * matrix,
    int64_t num_rows,
    const int64_t * rowidx,
    int64_t num_columns,
    const int64_t * colidx,
    enum mtxfileformat mtxfmt,
    const char * path,
    bool gzip,
    const char * fmt,
    int64_t * bytes_written)
{
    struct mtxfile mtxfile;
    int err = mtxmatrix_to_mtxfile(
        &mtxfile, matrix, num_rows, rowidx, num_columns, colidx, mtxfmt);
    if (err)
        return err;
    err = mtxfile_write(
        &mtxfile, path, gzip, fmt, bytes_written);
    if (err) {
        mtxfile_free(&mtxfile);
        return err;
    }
    mtxfile_free(&mtxfile);
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_fwrite()’ writes a matrix to a stream.
 *
 * If ‘fmt’ is ‘NULL’, then the format specifier ‘%g’ is used to print
 * floating point numbers with enough digits to ensure correct
 * round-trip conversion from decimal text and back.  Otherwise, the
 * given format string is used to print numerical values.
 *
 * The format string follows the conventions of ‘printf’. If the field
 * is ‘real’ or ‘complex’, then the format specifiers '%e', '%E',
 * '%f', '%F', '%g' or '%G' may be used. If the field is ‘integer’,
 * then the format specifier must be '%d'. The format string is
 * ignored if the field is ‘pattern’. Field width and precision may be
 * specified (e.g., "%3.1f"), but variable field width and precision
 * (e.g., "%*.*f"), as well as length modifiers (e.g., "%Lf") are not
 * allowed.
 *
 * If it is not ‘NULL’, then the number of bytes written to the stream
 * is returned in ‘bytes_written’.
 */
int mtxmatrix_fwrite(
    const struct mtxmatrix * matrix,
    int64_t num_rows,
    const int64_t * rowidx,
    int64_t num_columns,
    const int64_t * colidx,
    enum mtxfileformat mtxfmt,
    FILE * f,
    const char * fmt,
    int64_t * bytes_written)
{
    struct mtxfile mtxfile;
    int err = mtxmatrix_to_mtxfile(
        &mtxfile, matrix, num_rows, rowidx, num_columns, colidx, mtxfmt);
    if (err) return err;
    err = mtxfile_fwrite(&mtxfile, f, fmt, bytes_written);
    if (err) {
        mtxfile_free(&mtxfile);
        return err;
    }
    mtxfile_free(&mtxfile);
    return MTX_SUCCESS;
}

#ifdef LIBMTX_HAVE_LIBZ
/**
 * ‘mtxmatrix_gzwrite()’ writes a matrix to a gzip-compressed stream.
 *
 * If ‘fmt’ is ‘NULL’, then the format specifier ‘%g’ is used to print
 * floating point numbers with enough digits to ensure correct
 * round-trip conversion from decimal text and back.  Otherwise, the
 * given format string is used to print numerical values.
 *
 * The format string follows the conventions of ‘printf’. If the field
 * is ‘real’ or ‘complex’, then the format specifiers '%e', '%E',
 * '%f', '%F', '%g' or '%G' may be used. If the field is ‘integer’,
 * then the format specifier must be '%d'. The format string is
 * ignored if the field is ‘pattern’. Field width and precision may be
 * specified (e.g., "%3.1f"), but variable field width and precision
 * (e.g., "%*.*f"), as well as length modifiers (e.g., "%Lf") are not
 * allowed.
 *
 * If it is not ‘NULL’, then the number of bytes written to the stream
 * is returned in ‘bytes_written’.
 */
int mtxmatrix_gzwrite(
    const struct mtxmatrix * matrix,
    int64_t num_rows,
    const int64_t * rowidx,
    int64_t num_columns,
    const int64_t * colidx,
    enum mtxfileformat mtxfmt,
    gzFile f,
    const char * fmt,
    int64_t * bytes_written)
{
    struct mtxfile mtxfile;
    int err = mtxmatrix_to_mtxfile(
        &mtxfile, matrix, num_rows, rowidx, num_columns, colidx, mtxfmt);
    if (err) return err;
    err = mtxfile_gzwrite(&mtxfile, f, fmt, bytes_written);
    if (err) {
        mtxfile_free(&mtxfile);
        return err;
    }
    mtxfile_free(&mtxfile);
    return MTX_SUCCESS;
}
#endif

/*
 * Level 1 BLAS operations
 */

/**
 * ‘mtxmatrix_swap()’ swaps values of two matrices, simultaneously
 * performing ‘Y <- X’ and ‘X <- Y’.
 */
int mtxmatrix_swap(
    struct mtxmatrix * X,
    struct mtxmatrix * Y)
{
    if (X->type == mtxmatrix_blas && Y->type == mtxmatrix_blas) {
        return mtxmatrix_blas_swap(
            &X->storage.blas, &Y->storage.blas);
    } else if (X->type == mtxmatrix_coo && Y->type == mtxmatrix_coo) {
        return mtxmatrix_coo_swap(
            &X->storage.coordinate, &Y->storage.coordinate);
    } else if (X->type == mtxmatrix_csr && Y->type == mtxmatrix_csr) {
        return mtxmatrix_csr_swap(
            &X->storage.csr, &Y->storage.csr);
    } else if (X->type == mtxmatrix_dense && Y->type == mtxmatrix_dense) {
        return mtxmatrix_dense_swap(
            &X->storage.dense, &Y->storage.dense);
    } else if (X->type == mtxmatrix_ompcsr && Y->type == mtxmatrix_ompcsr) {
        return mtxmatrix_ompcsr_swap(
            &X->storage.ompcsr, &Y->storage.ompcsr);
    } else { return MTX_ERR_INVALID_MATRIX_TYPE; }
}

/**
 * ‘mtxmatrix_copy()’ copies values of a matrix, ‘Y = X’.
 */
int mtxmatrix_copy(
    struct mtxmatrix * Y,
    const struct mtxmatrix * X)
{
    if (X->type == mtxmatrix_blas && Y->type == mtxmatrix_blas) {
        return mtxmatrix_blas_copy(
            &Y->storage.blas, &X->storage.blas);
    } else if (X->type == mtxmatrix_coo && Y->type == mtxmatrix_coo) {
        return mtxmatrix_coo_copy(
            &Y->storage.coordinate, &X->storage.coordinate);
    } else if (X->type == mtxmatrix_csr && Y->type == mtxmatrix_csr) {
        return mtxmatrix_csr_copy(
            &Y->storage.csr, &X->storage.csr);
    } else if (X->type == mtxmatrix_dense && Y->type == mtxmatrix_dense) {
        return mtxmatrix_dense_copy(
            &Y->storage.dense, &X->storage.dense);
    } else if (X->type == mtxmatrix_ompcsr && Y->type == mtxmatrix_ompcsr) {
        return mtxmatrix_ompcsr_copy(
            &Y->storage.ompcsr, &X->storage.ompcsr);
    } else { return MTX_ERR_INVALID_MATRIX_TYPE; }
}

/**
 * ‘mtxmatrix_sscal()’ scales a matrix by a single precision floating
 * point scalar, ‘X = a*X’.
 */
int mtxmatrix_sscal(
    float a,
    struct mtxmatrix * X,
    int64_t * num_flops)
{
    if (X->type == mtxmatrix_blas) {
        return mtxmatrix_blas_sscal(
            a, &X->storage.blas, num_flops);
    } else if (X->type == mtxmatrix_coo) {
        return mtxmatrix_coo_sscal(
            a, &X->storage.coordinate, num_flops);
    } else if (X->type == mtxmatrix_csr) {
        return mtxmatrix_csr_sscal(
            a, &X->storage.csr, num_flops);
    } else if (X->type == mtxmatrix_dense) {
        return mtxmatrix_dense_sscal(
            a, &X->storage.dense, num_flops);
    } else if (X->type == mtxmatrix_ompcsr) {
        return mtxmatrix_ompcsr_sscal(
            a, &X->storage.ompcsr, num_flops);
    } else { return MTX_ERR_INVALID_MATRIX_TYPE; }
}

/**
 * ‘mtxmatrix_dscal()’ scales a matrix by a double precision floating
 * point scalar, ‘X = a*X’.
 */
int mtxmatrix_dscal(
    double a,
    struct mtxmatrix * X,
    int64_t * num_flops)
{
    if (X->type == mtxmatrix_blas) {
        return mtxmatrix_blas_dscal(
            a, &X->storage.blas, num_flops);
    } else if (X->type == mtxmatrix_coo) {
        return mtxmatrix_coo_dscal(
            a, &X->storage.coordinate, num_flops);
    } else if (X->type == mtxmatrix_csr) {
        return mtxmatrix_csr_dscal(
            a, &X->storage.csr, num_flops);
    } else if (X->type == mtxmatrix_dense) {
        return mtxmatrix_dense_dscal(
            a, &X->storage.dense, num_flops);
    } else if (X->type == mtxmatrix_ompcsr) {
        return mtxmatrix_ompcsr_dscal(
            a, &X->storage.ompcsr, num_flops);
    } else { return MTX_ERR_INVALID_MATRIX_TYPE; }
}

/**
 * ‘mtxmatrix_cscal()’ scales a matrix by a complex, single precision
 * floating point scalar, ‘X = (a+b*i)*X’.
 */
int mtxmatrix_cscal(
    float a[2],
    struct mtxmatrix * X,
    int64_t * num_flops)
{
    if (X->type == mtxmatrix_blas) {
        return mtxmatrix_blas_cscal(
            a, &X->storage.blas, num_flops);
    } else if (X->type == mtxmatrix_coo) {
        return mtxmatrix_coo_cscal(
            a, &X->storage.coordinate, num_flops);
    } else if (X->type == mtxmatrix_dense) {
        return mtxmatrix_dense_cscal(
            a, &X->storage.dense, num_flops);
    } else { return MTX_ERR_INVALID_MATRIX_TYPE; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_zscal()’ scales a matrix by a complex, double precision
 * floating point scalar, ‘X = (a+b*i)*X’.
 */
int mtxmatrix_zscal(
    double a[2],
    struct mtxmatrix * X,
    int64_t * num_flops)
{
    if (X->type == mtxmatrix_blas) {
        return mtxmatrix_blas_zscal(
            a, &X->storage.blas, num_flops);
    } else if (X->type == mtxmatrix_coo) {
        return mtxmatrix_coo_zscal(
            a, &X->storage.coordinate, num_flops);
    } else if (X->type == mtxmatrix_dense) {
        return mtxmatrix_dense_zscal(
            a, &X->storage.dense, num_flops);
    } else { return MTX_ERR_INVALID_MATRIX_TYPE; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_saxpy()’ adds a matrix to another matrix multiplied by a
 * single precision floating point value, ‘Y = a*X + Y’.
 */
int mtxmatrix_saxpy(
    float a,
    const struct mtxmatrix * X,
    struct mtxmatrix * Y,
    int64_t * num_flops)
{
    if (X->type == mtxmatrix_blas && Y->type == mtxmatrix_blas) {
        return mtxmatrix_blas_saxpy(
            a, &X->storage.blas, &Y->storage.blas, num_flops);
    } else if (X->type == mtxmatrix_coo && Y->type == mtxmatrix_coo) {
        return mtxmatrix_coo_saxpy(
            a, &X->storage.coordinate, &Y->storage.coordinate, num_flops);
    } else if (X->type == mtxmatrix_csr && Y->type == mtxmatrix_csr) {
        return mtxmatrix_csr_saxpy(
            a, &X->storage.csr, &Y->storage.csr, num_flops);
    } else if (X->type == mtxmatrix_dense && Y->type == mtxmatrix_dense) {
        return mtxmatrix_dense_saxpy(
            a, &X->storage.dense, &Y->storage.dense, num_flops);
    } else if (X->type == mtxmatrix_ompcsr && Y->type == mtxmatrix_ompcsr) {
        return mtxmatrix_ompcsr_saxpy(
            a, &X->storage.ompcsr, &Y->storage.ompcsr, num_flops);
    } else { return MTX_ERR_INVALID_MATRIX_TYPE; }
}

/**
 * ‘mtxmatrix_daxpy()’ adds a matrix to another matrix multiplied by a
 * double precision floating point value, ‘Y = a*X + Y’.
 */
int mtxmatrix_daxpy(
    double a,
    const struct mtxmatrix * X,
    struct mtxmatrix * Y,
    int64_t * num_flops)
{
    if (X->type == mtxmatrix_blas && Y->type == mtxmatrix_blas) {
        return mtxmatrix_blas_daxpy(
            a, &X->storage.blas, &Y->storage.blas, num_flops);
    } else if (X->type == mtxmatrix_coo && Y->type == mtxmatrix_coo) {
        return mtxmatrix_coo_daxpy(
            a, &X->storage.coordinate, &Y->storage.coordinate, num_flops);
    } else if (X->type == mtxmatrix_csr && Y->type == mtxmatrix_csr) {
        return mtxmatrix_csr_daxpy(
            a, &X->storage.csr, &Y->storage.csr, num_flops);
    } else if (X->type == mtxmatrix_dense && Y->type == mtxmatrix_dense) {
        return mtxmatrix_dense_daxpy(
            a, &X->storage.dense, &Y->storage.dense, num_flops);
    } else if (X->type == mtxmatrix_ompcsr && Y->type == mtxmatrix_ompcsr) {
        return mtxmatrix_ompcsr_daxpy(
            a, &X->storage.ompcsr, &Y->storage.ompcsr, num_flops);
    } else { return MTX_ERR_INVALID_MATRIX_TYPE; }
}

/**
 * ‘mtxmatrix_saypx()’ multiplies a matrix by a single precision
 * floating point scalar and adds another matrix, ‘Y = a*Y + X’.
 */
int mtxmatrix_saypx(
    float a,
    struct mtxmatrix * Y,
    const struct mtxmatrix * X,
    int64_t * num_flops)
{
    if (X->type == mtxmatrix_blas && Y->type == mtxmatrix_blas) {
        return mtxmatrix_blas_saypx(
            a, &Y->storage.blas, &X->storage.blas, num_flops);
    } else if (X->type == mtxmatrix_coo && Y->type == mtxmatrix_coo) {
        return mtxmatrix_coo_saypx(
            a, &Y->storage.coordinate, &X->storage.coordinate, num_flops);
    } else if (X->type == mtxmatrix_csr && Y->type == mtxmatrix_csr) {
        return mtxmatrix_csr_saypx(
            a, &Y->storage.csr, &X->storage.csr, num_flops);
    } else if (X->type == mtxmatrix_dense && Y->type == mtxmatrix_dense) {
        return mtxmatrix_dense_saypx(
            a, &Y->storage.dense, &X->storage.dense, num_flops);
    } else if (X->type == mtxmatrix_ompcsr && Y->type == mtxmatrix_ompcsr) {
        return mtxmatrix_ompcsr_saypx(
            a, &Y->storage.ompcsr, &X->storage.ompcsr, num_flops);
    } else { return MTX_ERR_INVALID_MATRIX_TYPE; }
}

/**
 * ‘mtxmatrix_daypx()’ multiplies a matrix by a double precision
 * floating point scalar and adds another matrix, ‘Y = a*Y + X’.
 */
int mtxmatrix_daypx(
    double a,
    struct mtxmatrix * Y,
    const struct mtxmatrix * X,
    int64_t * num_flops)
{
    if (X->type == mtxmatrix_blas && Y->type == mtxmatrix_blas) {
        return mtxmatrix_blas_daypx(
            a, &Y->storage.blas, &X->storage.blas, num_flops);
    } else if (X->type == mtxmatrix_coo && Y->type == mtxmatrix_coo) {
        return mtxmatrix_coo_daypx(
            a, &Y->storage.coordinate, &X->storage.coordinate, num_flops);
    } else if (X->type == mtxmatrix_csr && Y->type == mtxmatrix_csr) {
        return mtxmatrix_csr_daypx(
            a, &Y->storage.csr, &X->storage.csr, num_flops);
    } else if (X->type == mtxmatrix_dense && Y->type == mtxmatrix_dense) {
        return mtxmatrix_dense_daypx(
            a, &Y->storage.dense, &X->storage.dense, num_flops);
    } else if (X->type == mtxmatrix_ompcsr && Y->type == mtxmatrix_ompcsr) {
        return mtxmatrix_ompcsr_daypx(
            a, &Y->storage.ompcsr, &X->storage.ompcsr, num_flops);
    } else { return MTX_ERR_INVALID_MATRIX_TYPE; }
}

/**
 * ‘mtxmatrix_sdot()’ computes the Frobenius dot product of two
 * matrices in single precision floating point.
 */
int mtxmatrix_sdot(
    const struct mtxmatrix * X,
    const struct mtxmatrix * Y,
    float * dot,
    int64_t * num_flops)
{
    if (X->type == mtxmatrix_blas && Y->type == mtxmatrix_blas) {
        return mtxmatrix_blas_sdot(
            &X->storage.blas, &Y->storage.blas, dot, num_flops);
    } else if (X->type == mtxmatrix_coo && Y->type == mtxmatrix_coo) {
        return mtxmatrix_coo_sdot(
            &X->storage.coordinate, &Y->storage.coordinate, dot, num_flops);
    } else if (X->type == mtxmatrix_csr && Y->type == mtxmatrix_csr) {
        return mtxmatrix_csr_sdot(
            &X->storage.csr, &Y->storage.csr, dot, num_flops);
    } else if (X->type == mtxmatrix_dense && Y->type == mtxmatrix_dense) {
        return mtxmatrix_dense_sdot(
            &X->storage.dense, &Y->storage.dense, dot, num_flops);
    } else if (X->type == mtxmatrix_ompcsr && Y->type == mtxmatrix_ompcsr) {
        return mtxmatrix_ompcsr_sdot(
            &X->storage.ompcsr, &Y->storage.ompcsr, dot, num_flops);
    } else { return MTX_ERR_INVALID_MATRIX_TYPE; }
}

/**
 * ‘mtxmatrix_ddot()’ computes the Frobenius dot product of two
 * matrices in double precision floating point.
 */
int mtxmatrix_ddot(
    const struct mtxmatrix * X,
    const struct mtxmatrix * Y,
    double * dot,
    int64_t * num_flops)
{
    if (X->type == mtxmatrix_blas && Y->type == mtxmatrix_blas) {
        return mtxmatrix_blas_ddot(
            &X->storage.blas, &Y->storage.blas, dot, num_flops);
    } else if (X->type == mtxmatrix_coo && Y->type == mtxmatrix_coo) {
        return mtxmatrix_coo_ddot(
            &X->storage.coordinate, &Y->storage.coordinate, dot, num_flops);
    } else if (X->type == mtxmatrix_csr && Y->type == mtxmatrix_csr) {
        return mtxmatrix_csr_ddot(
            &X->storage.csr, &Y->storage.csr, dot, num_flops);
    } else if (X->type == mtxmatrix_dense && Y->type == mtxmatrix_dense) {
        return mtxmatrix_dense_ddot(
            &X->storage.dense, &Y->storage.dense, dot, num_flops);
    } else if (X->type == mtxmatrix_ompcsr && Y->type == mtxmatrix_ompcsr) {
        return mtxmatrix_ompcsr_ddot(
            &X->storage.ompcsr, &Y->storage.ompcsr, dot, num_flops);
    } else { return MTX_ERR_INVALID_MATRIX_TYPE; }
}

/**
 * ‘mtxmatrix_cdotu()’ computes the product of the dot product of the
 * vectorisation of a complex matrix with the vectorisation of another
 * complex matrix in single precision floating point, ‘dot :=
 * vec(X)^T*vec(Y)’, where ‘vec(X)’ is the vectorisation of ‘X’.
 */
int mtxmatrix_cdotu(
    const struct mtxmatrix * X,
    const struct mtxmatrix * Y,
    float (* dot)[2],
    int64_t * num_flops)
{
    if (X->type == mtxmatrix_blas && Y->type == mtxmatrix_blas) {
        return mtxmatrix_blas_cdotu(
            &X->storage.blas, &Y->storage.blas, dot, num_flops);
    } else if (X->type == mtxmatrix_coo && Y->type == mtxmatrix_coo) {
        return mtxmatrix_coo_cdotu(
            &X->storage.coordinate, &Y->storage.coordinate, dot, num_flops);
    } else if (X->type == mtxmatrix_csr && Y->type == mtxmatrix_csr) {
        return mtxmatrix_csr_cdotu(
            &X->storage.csr, &Y->storage.csr, dot, num_flops);
    } else if (X->type == mtxmatrix_dense && Y->type == mtxmatrix_dense) {
        return mtxmatrix_dense_cdotu(
            &X->storage.dense, &Y->storage.dense, dot, num_flops);
    } else if (X->type == mtxmatrix_ompcsr && Y->type == mtxmatrix_ompcsr) {
        return mtxmatrix_ompcsr_cdotu(
            &X->storage.ompcsr, &Y->storage.ompcsr, dot, num_flops);
    } else { return MTX_ERR_INVALID_MATRIX_TYPE; }
}

/**
 * ‘mtxmatrix_zdotu()’ computes the product of the dot product of the
 * vectorisation of a complex matrix with the vectorisation of another
 * complex matrix in double precision floating point, ‘dot :=
 * vec(X)^T*vec(Y)’, where ‘vec(X)’ is the vectorisation of ‘X’.
 */
int mtxmatrix_zdotu(
    const struct mtxmatrix * X,
    const struct mtxmatrix * Y,
    double (* dot)[2],
    int64_t * num_flops)
{
    if (X->type == mtxmatrix_blas && Y->type == mtxmatrix_blas) {
        return mtxmatrix_blas_zdotu(
            &X->storage.blas, &Y->storage.blas, dot, num_flops);
    } else if (X->type == mtxmatrix_coo && Y->type == mtxmatrix_coo) {
        return mtxmatrix_coo_zdotu(
            &X->storage.coordinate, &Y->storage.coordinate, dot, num_flops);
    } else if (X->type == mtxmatrix_csr && Y->type == mtxmatrix_csr) {
        return mtxmatrix_csr_zdotu(
            &X->storage.csr, &Y->storage.csr, dot, num_flops);
    } else if (X->type == mtxmatrix_dense && Y->type == mtxmatrix_dense) {
        return mtxmatrix_dense_zdotu(
            &X->storage.dense, &Y->storage.dense, dot, num_flops);
    } else if (X->type == mtxmatrix_ompcsr && Y->type == mtxmatrix_ompcsr) {
        return mtxmatrix_ompcsr_zdotu(
            &X->storage.ompcsr, &Y->storage.ompcsr, dot, num_flops);
    } else { return MTX_ERR_INVALID_MATRIX_TYPE; }
}

/**
 * ‘mtxmatrix_cdotc()’ computes the Frobenius dot product of two
 * complex matrices in single precision floating point, ‘dot :=
 * vec(X)^H*vec(Y)’, where ‘vec(X)’ is the vectorisation of ‘X’.
 */
int mtxmatrix_cdotc(
    const struct mtxmatrix * X,
    const struct mtxmatrix * Y,
    float (* dot)[2],
    int64_t * num_flops)
{
    if (X->type == mtxmatrix_blas && Y->type == mtxmatrix_blas) {
        return mtxmatrix_blas_cdotc(
            &X->storage.blas, &Y->storage.blas, dot, num_flops);
    } else if (X->type == mtxmatrix_coo && Y->type == mtxmatrix_coo) {
        return mtxmatrix_coo_cdotc(
            &X->storage.coordinate, &Y->storage.coordinate, dot, num_flops);
    } else if (X->type == mtxmatrix_csr && Y->type == mtxmatrix_csr) {
        return mtxmatrix_csr_cdotc(
            &X->storage.csr, &Y->storage.csr, dot, num_flops);
    } else if (X->type == mtxmatrix_dense && Y->type == mtxmatrix_dense) {
        return mtxmatrix_dense_cdotc(
            &X->storage.dense, &Y->storage.dense, dot, num_flops);
    } else if (X->type == mtxmatrix_ompcsr && Y->type == mtxmatrix_ompcsr) {
        return mtxmatrix_ompcsr_cdotc(
            &X->storage.ompcsr, &Y->storage.ompcsr, dot, num_flops);
    } else { return MTX_ERR_INVALID_MATRIX_TYPE; }
}

/**
 * ‘mtxmatrix_zdotc()’ computes the Frobenius dot product of two
 * complex matrices in double precision floating point, ‘dot :=
 * vec(X)^H*vec(Y)’, where ‘vec(X)’ is the vectorisation of ‘X’.
 */
int mtxmatrix_zdotc(
    const struct mtxmatrix * X,
    const struct mtxmatrix * Y,
    double (* dot)[2],
    int64_t * num_flops)
{
    if (X->type == mtxmatrix_blas && Y->type == mtxmatrix_blas) {
        return mtxmatrix_blas_zdotc(
            &X->storage.blas, &Y->storage.blas, dot, num_flops);
    } else if (X->type == mtxmatrix_coo && Y->type == mtxmatrix_coo) {
        return mtxmatrix_coo_zdotc(
            &X->storage.coordinate, &Y->storage.coordinate, dot, num_flops);
    } else if (X->type == mtxmatrix_csr && Y->type == mtxmatrix_csr) {
        return mtxmatrix_csr_zdotc(
            &X->storage.csr, &Y->storage.csr, dot, num_flops);
    } else if (X->type == mtxmatrix_dense && Y->type == mtxmatrix_dense) {
        return mtxmatrix_dense_zdotc(
            &X->storage.dense, &Y->storage.dense, dot, num_flops);
    } else if (X->type == mtxmatrix_ompcsr && Y->type == mtxmatrix_ompcsr) {
        return mtxmatrix_ompcsr_zdotc(
            &X->storage.ompcsr, &Y->storage.ompcsr, dot, num_flops);
    } else { return MTX_ERR_INVALID_MATRIX_TYPE; }
}

/**
 * ‘mtxmatrix_snrm2()’ computes the Frobenius norm of a matrix in
 * single precision floating point.
 */
int mtxmatrix_snrm2(
    const struct mtxmatrix * X,
    float * nrm2,
    int64_t * num_flops)
{
    if (X->type == mtxmatrix_blas) {
        return mtxmatrix_blas_snrm2(&X->storage.blas, nrm2, num_flops);
    } else if (X->type == mtxmatrix_coo) {
        return mtxmatrix_coo_snrm2(&X->storage.coordinate, nrm2, num_flops);
    } else if (X->type == mtxmatrix_csr) {
        return mtxmatrix_csr_snrm2(&X->storage.csr, nrm2, num_flops);
    } else if (X->type == mtxmatrix_dense) {
        return mtxmatrix_dense_snrm2(&X->storage.dense, nrm2, num_flops);
    } else if (X->type == mtxmatrix_ompcsr) {
        return mtxmatrix_ompcsr_snrm2(&X->storage.ompcsr, nrm2, num_flops);
    } else { return MTX_ERR_INVALID_MATRIX_TYPE; }
}

/**
 * ‘mtxmatrix_dnrm2()’ computes the Frobenius norm of a matrix in
 * double precision floating point.
 */
int mtxmatrix_dnrm2(
    const struct mtxmatrix * X,
    double * nrm2,
    int64_t * num_flops)
{
    if (X->type == mtxmatrix_blas) {
        return mtxmatrix_blas_dnrm2(&X->storage.blas, nrm2, num_flops);
    } else if (X->type == mtxmatrix_coo) {
        return mtxmatrix_coo_dnrm2(&X->storage.coordinate, nrm2, num_flops);
    } else if (X->type == mtxmatrix_csr) {
        return mtxmatrix_csr_dnrm2(&X->storage.csr, nrm2, num_flops);
    } else if (X->type == mtxmatrix_dense) {
        return mtxmatrix_dense_dnrm2(&X->storage.dense, nrm2, num_flops);
    } else if (X->type == mtxmatrix_ompcsr) {
        return mtxmatrix_ompcsr_dnrm2(&X->storage.ompcsr, nrm2, num_flops);
    } else { return MTX_ERR_INVALID_MATRIX_TYPE; }
}

/**
 * ‘mtxmatrix_sasum()’ computes the sum of absolute values (1-norm) of
 * a matrix in single precision floating point.  If the matrix is
 * complex-valued, then the sum of the absolute values of the real and
 * imaginary parts is computed.
 */
int mtxmatrix_sasum(
    const struct mtxmatrix * X,
    float * asum,
    int64_t * num_flops)
{
    if (X->type == mtxmatrix_blas) {
        return mtxmatrix_blas_sasum(&X->storage.blas, asum, num_flops);
    } else if (X->type == mtxmatrix_coo) {
        return mtxmatrix_coo_sasum(&X->storage.coordinate, asum, num_flops);
    } else if (X->type == mtxmatrix_csr) {
        return mtxmatrix_csr_sasum(&X->storage.csr, asum, num_flops);
    } else if (X->type == mtxmatrix_dense) {
        return mtxmatrix_dense_sasum(&X->storage.dense, asum, num_flops);
    } else if (X->type == mtxmatrix_ompcsr) {
        return mtxmatrix_ompcsr_sasum(&X->storage.ompcsr, asum, num_flops);
    } else { return MTX_ERR_INVALID_MATRIX_TYPE; }
}

/**
 * ‘mtxmatrix_dasum()’ computes the sum of absolute values (1-norm) of
 * a matrix in double precision floating point.  If the matrix is
 * complex-valued, then the sum of the absolute values of the real and
 * imaginary parts is computed.
 */
int mtxmatrix_dasum(
    const struct mtxmatrix * X,
    double * asum,
    int64_t * num_flops)
{
    if (X->type == mtxmatrix_blas) {
        return mtxmatrix_blas_dasum(&X->storage.blas, asum, num_flops);
    } else if (X->type == mtxmatrix_coo) {
        return mtxmatrix_coo_dasum(&X->storage.coordinate, asum, num_flops);
    } else if (X->type == mtxmatrix_csr) {
        return mtxmatrix_csr_dasum(&X->storage.csr, asum, num_flops);
    } else if (X->type == mtxmatrix_dense) {
        return mtxmatrix_dense_dasum(&X->storage.dense, asum, num_flops);
    } else if (X->type == mtxmatrix_ompcsr) {
        return mtxmatrix_ompcsr_dasum(&X->storage.ompcsr, asum, num_flops);
    } else { return MTX_ERR_INVALID_MATRIX_TYPE; }
}

/**
 * ‘mtxmatrix_iamax()’ finds the index of the first element having the
 * maximum absolute value.  If the matrix is complex-valued, then the
 * index points to the first element having the maximum sum of the
 * absolute values of the real and imaginary parts.
 */
int mtxmatrix_iamax(
    const struct mtxmatrix * X,
    int * iamax)
{
    if (X->type == mtxmatrix_blas) {
        return mtxmatrix_blas_iamax(&X->storage.blas, iamax);
    } else if (X->type == mtxmatrix_coo) {
        return mtxmatrix_coo_iamax(&X->storage.coordinate, iamax);
    } else if (X->type == mtxmatrix_csr) {
        return mtxmatrix_csr_iamax(&X->storage.csr, iamax);
    } else if (X->type == mtxmatrix_dense) {
        return mtxmatrix_dense_iamax(&X->storage.dense, iamax);
    } else if (X->type == mtxmatrix_ompcsr) {
        return mtxmatrix_ompcsr_iamax(&X->storage.ompcsr, iamax);
    } else { return MTX_ERR_INVALID_MATRIX_TYPE; }
}

/*
 * Level 2 BLAS operations
 */

/**
 * ‘mtxmatrix_sgemv()’ multiplies a matrix ‘A’ or its transpose ‘A'’
 * by a real scalar ‘alpha’ (‘α’) and a vector ‘x’, before adding the
 * result to another vector ‘y’ multiplied by another real scalar
 * ‘beta’ (‘β’). That is, ‘y = α*A*x + β*y’ or ‘y = α*A'*x + β*y’.
 *
 * The scalars ‘alpha’ and ‘beta’ are given as single precision
 * floating point numbers.
 */
int mtxmatrix_sgemv(
    enum mtxtransposition trans,
    float alpha,
    const struct mtxmatrix * A,
    const struct mtxvector * x,
    float beta,
    struct mtxvector * y,
    int64_t * num_flops)
{
    if (A->type == mtxmatrix_blas) {
        return mtxmatrix_blas_sgemv(
            trans, alpha, &A->storage.blas, x, beta, y, num_flops);
    } else if (A->type == mtxmatrix_coo) {
        return mtxmatrix_coo_sgemv(
            trans, alpha, &A->storage.coordinate, x, beta, y, num_flops);
    } else if (A->type == mtxmatrix_csr) {
        return mtxmatrix_csr_sgemv(
            trans, alpha, &A->storage.csr, x, beta, y, num_flops);
    } else if (A->type == mtxmatrix_dense) {
        return mtxmatrix_dense_sgemv(
            trans, alpha, &A->storage.dense, x, beta, y, num_flops);
    } else if (A->type == mtxmatrix_ompcsr) {
        return mtxmatrix_ompcsr_sgemv(
            trans, alpha, &A->storage.ompcsr, x, beta, y, num_flops);
    } else { return MTX_ERR_INVALID_MATRIX_TYPE; }
}

/**
 * ‘mtxmatrix_dgemv()’ multiplies a matrix ‘A’ or its transpose ‘A'’
 * by a real scalar ‘alpha’ (‘α’) and a vector ‘x’, before adding the
 * result to another vector ‘y’ multiplied by another scalar real
 * ‘beta’ (‘β’).  That is, ‘y = α*A*x + β*y’ or ‘y = α*A'*x + β*y’.
 *
 * The scalars ‘alpha’ and ‘beta’ are given as double precision
 * floating point numbers.
 */
int mtxmatrix_dgemv(
    enum mtxtransposition trans,
    double alpha,
    const struct mtxmatrix * A,
    const struct mtxvector * x,
    double beta,
    struct mtxvector * y,
    int64_t * num_flops)
{
    if (A->type == mtxmatrix_blas) {
        return mtxmatrix_blas_dgemv(
            trans, alpha, &A->storage.blas, x, beta, y, num_flops);
    } else if (A->type == mtxmatrix_coo) {
        return mtxmatrix_coo_dgemv(
            trans, alpha, &A->storage.coordinate, x, beta, y, num_flops);
    } else if (A->type == mtxmatrix_csr) {
        return mtxmatrix_csr_dgemv(
            trans, alpha, &A->storage.csr, x, beta, y, num_flops);
    } else if (A->type == mtxmatrix_dense) {
        return mtxmatrix_dense_dgemv(
            trans, alpha, &A->storage.dense, x, beta, y, num_flops);
    } else if (A->type == mtxmatrix_ompcsr) {
        return mtxmatrix_ompcsr_dgemv(
            trans, alpha, &A->storage.ompcsr, x, beta, y, num_flops);
    } else { return MTX_ERR_INVALID_MATRIX_TYPE; }
}

/**
 * ‘mtxmatrix_cgemv()’ multiplies a complex-valued matrix ‘A’, its
 * transpose ‘A'’ or its conjugate transpose ‘Aᴴ’ by a complex scalar
 * ‘alpha’ (‘α’) and a vector ‘x’, before adding the result to another
 * vector ‘y’ multiplied by another complex scalar ‘beta’ (‘β’).  That
 * is, ‘y = α*A*x + β*y’, ‘y = α*A'*x + β*y’ or ‘y = α*Aᴴ*x + β*y’.
 *
 * The scalars ‘alpha’ and ‘beta’ are given as single precision
 * floating point numbers.
 */
int mtxmatrix_cgemv(
    enum mtxtransposition trans,
    float alpha[2],
    const struct mtxmatrix * A,
    const struct mtxvector * x,
    float beta[2],
    struct mtxvector * y,
    int64_t * num_flops)
{
    if (A->type == mtxmatrix_blas) {
        return mtxmatrix_blas_cgemv(
            trans, alpha, &A->storage.blas, x, beta, y, num_flops);
    } else if (A->type == mtxmatrix_coo) {
        return mtxmatrix_coo_cgemv(
            trans, alpha, &A->storage.coordinate, x, beta, y, num_flops);
    } else if (A->type == mtxmatrix_csr) {
        return mtxmatrix_csr_cgemv(
            trans, alpha, &A->storage.csr, x, beta, y, num_flops);
    } else if (A->type == mtxmatrix_dense) {
        return mtxmatrix_dense_cgemv(
            trans, alpha, &A->storage.dense, x, beta, y, num_flops);
    } else if (A->type == mtxmatrix_ompcsr) {
        return mtxmatrix_ompcsr_cgemv(
            trans, alpha, &A->storage.ompcsr, x, beta, y, num_flops);
    } else { return MTX_ERR_INVALID_MATRIX_TYPE; }
}

/**
 * ‘mtxmatrix_zgemv()’ multiplies a complex-valued matrix ‘A’, its
 * transpose ‘A'’ or its conjugate transpose ‘Aᴴ’ by a complex scalar
 * ‘alpha’ (‘α’) and a vector ‘x’, before adding the result to another
 * vector ‘y’ multiplied by another complex scalar ‘beta’ (‘β’).  That
 * is, ‘y = α*A*x + β*y’, ‘y = α*A'*x + β*y’ or ‘y = α*Aᴴ*x + β*y’.
 *
 * The scalars ‘alpha’ and ‘beta’ are given as double precision
 * floating point numbers.
 */
int mtxmatrix_zgemv(
    enum mtxtransposition trans,
    double alpha[2],
    const struct mtxmatrix * A,
    const struct mtxvector * x,
    double beta[2],
    struct mtxvector * y,
    int64_t * num_flops)
{
    if (A->type == mtxmatrix_blas) {
        return mtxmatrix_blas_zgemv(
            trans, alpha, &A->storage.blas, x, beta, y, num_flops);
    } else if (A->type == mtxmatrix_coo) {
        return mtxmatrix_coo_zgemv(
            trans, alpha, &A->storage.coordinate, x, beta, y, num_flops);
    } else if (A->type == mtxmatrix_csr) {
        return mtxmatrix_csr_zgemv(
            trans, alpha, &A->storage.csr, x, beta, y, num_flops);
    } else if (A->type == mtxmatrix_dense) {
        return mtxmatrix_dense_zgemv(
            trans, alpha, &A->storage.dense, x, beta, y, num_flops);
    } else if (A->type == mtxmatrix_ompcsr) {
        return mtxmatrix_ompcsr_zgemv(
            trans, alpha, &A->storage.ompcsr, x, beta, y, num_flops);
    } else { return MTX_ERR_INVALID_MATRIX_TYPE; }
}
