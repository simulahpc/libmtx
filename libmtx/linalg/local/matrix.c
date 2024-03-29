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
 * Last modified: 2022-10-05
 *
 * Data structures for matrices.
 */

#include <libmtx/libmtx-config.h>

#include <libmtx/error.h>
#include <libmtx/linalg/local/matrix.h>
#include <libmtx/mtxfile/mtxfile.h>
#include <libmtx/linalg/precision.h>
#include <libmtx/util/partition.h>

#ifdef LIBMTX_HAVE_LIBZ
#include <zlib.h>
#endif

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
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
    case mtxblasdense: return "blas";
    case mtxbasecoo: return "coo";
    case mtxbasecsr: return "csr";
    case mtxbasedense: return "dense";
    case mtxnullcoo: return "nullcoo";
    case mtxompcsr: return "ompcsr";
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
        *matrix_type = mtxblasdense;
    } else if (strncmp("coo", t, strlen("coo")) == 0) {
        t += strlen("coo");
        *matrix_type = mtxbasecoo;
    } else if (strncmp("csr", t, strlen("csr")) == 0) {
        t += strlen("csr");
        *matrix_type = mtxbasecsr;
    } else if (strncmp("dense", t, strlen("dense")) == 0) {
        t += strlen("dense");
        *matrix_type = mtxbasedense;
    } else if (strncmp("nullcoo", t, strlen("nullcoo")) == 0) {
        t += strlen("nullcoo");
        *matrix_type = mtxnullcoo;
    } else if (strncmp("ompcsr", t, strlen("ompcsr")) == 0) {
        t += strlen("ompcsr");
        *matrix_type = mtxompcsr;
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

/*
 * matrix properties
 */

/**
 * ‘mtxmatrix_field()’ gets the field of a matrix.
 */
int mtxmatrix_field(
    const struct mtxmatrix * A,
    enum mtxfield * field)
{
    if (A->type == mtxblasdense) {
#ifdef LIBMTX_HAVE_BLAS
        *field = mtxblasdense_field(&A->storage.blas);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (A->type == mtxbasecoo) {
        *field = mtxbasecoo_field(&A->storage.coo);
    } else if (A->type == mtxbasedense) {
        *field = mtxbasedense_field(&A->storage.dense);
    } else if (A->type == mtxbasecsr) {
        *field = mtxbasecsr_field(&A->storage.csr);
    } else if (A->type == mtxnullcoo) {
        *field = mtxnullcoo_field(&A->storage.nullcoo);
    } else if (A->type == mtxompcsr) {
#ifdef LIBMTX_HAVE_OPENMP
        *field = mtxompcsr_field(&A->storage.ompcsr);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
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
    if (A->type == mtxblasdense) {
#ifdef LIBMTX_HAVE_BLAS
        *precision = mtxblasdense_precision(&A->storage.blas);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (A->type == mtxbasecoo) {
        *precision = mtxbasecoo_precision(&A->storage.coo);
    } else if (A->type == mtxbasedense) {
        *precision = mtxbasedense_precision(&A->storage.dense);
    } else if (A->type == mtxbasecsr) {
        *precision = mtxbasecsr_precision(&A->storage.csr);
    } else if (A->type == mtxnullcoo) {
        *precision = mtxnullcoo_precision(&A->storage.nullcoo);
    } else if (A->type == mtxompcsr) {
#ifdef LIBMTX_HAVE_OPENMP
        *precision = mtxompcsr_precision(&A->storage.ompcsr);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
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
    if (A->type == mtxblasdense) {
#ifdef LIBMTX_HAVE_BLAS
        *symmetry = mtxblasdense_symmetry(&A->storage.blas);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (A->type == mtxbasecoo) {
        *symmetry = mtxbasecoo_symmetry(&A->storage.coo);
    } else if (A->type == mtxbasedense) {
        *symmetry = mtxbasedense_symmetry(&A->storage.dense);
    } else if (A->type == mtxbasecsr) {
        *symmetry = mtxbasecsr_symmetry(&A->storage.csr);
    } else if (A->type == mtxnullcoo) {
        *symmetry = mtxnullcoo_symmetry(&A->storage.nullcoo);
    } else if (A->type == mtxompcsr) {
#ifdef LIBMTX_HAVE_OPENMP
        *symmetry = mtxompcsr_symmetry(&A->storage.ompcsr);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
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
    if (A->type == mtxblasdense) {
#ifdef LIBMTX_HAVE_BLAS
        *num_nonzeros = mtxblasdense_num_nonzeros(&A->storage.blas);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (A->type == mtxbasecoo) {
        *num_nonzeros = mtxbasecoo_num_nonzeros(&A->storage.coo);
    } else if (A->type == mtxbasedense) {
        *num_nonzeros = mtxbasedense_num_nonzeros(&A->storage.dense);
    } else if (A->type == mtxbasecsr) {
        *num_nonzeros = mtxbasecsr_num_nonzeros(&A->storage.csr);
    } else if (A->type == mtxnullcoo) {
        *num_nonzeros = mtxnullcoo_num_nonzeros(&A->storage.nullcoo);
    } else if (A->type == mtxompcsr) {
#ifdef LIBMTX_HAVE_OPENMP
        *num_nonzeros = mtxompcsr_num_nonzeros(&A->storage.ompcsr);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
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
    if (A->type == mtxblasdense) {
#ifdef LIBMTX_HAVE_BLAS
        *size = mtxblasdense_size(&A->storage.blas);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (A->type == mtxbasecoo) {
        *size = mtxbasecoo_size(&A->storage.coo);
    } else if (A->type == mtxbasedense) {
        *size = mtxbasedense_size(&A->storage.dense);
    } else if (A->type == mtxbasecsr) {
        *size = mtxbasecsr_size(&A->storage.csr);
    } else if (A->type == mtxnullcoo) {
        *size = mtxnullcoo_size(&A->storage.nullcoo);
    } else if (A->type == mtxompcsr) {
#ifdef LIBMTX_HAVE_OPENMP
        *size = mtxompcsr_size(&A->storage.ompcsr);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_MATRIX_TYPE; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_rowcolidx()’ gets the row and column indices of the
 * explicitly stored matrix nonzeros.
 *
 * The arguments ‘rowidx’ and ‘colidx’ may be ‘NULL’ or must point to
 * an arrays of length ‘size’.
 */
int mtxmatrix_rowcolidx(
    const struct mtxmatrix * A,
    int64_t size,
    int * rowidx,
    int * colidx)
{
    if (A->type == mtxblasdense) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasdense_rowcolidx(&A->storage.blas, size, rowidx, colidx);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (A->type == mtxbasecoo) {
        return mtxbasecoo_rowcolidx(&A->storage.coo, size, rowidx, colidx);
    } else if (A->type == mtxbasecsr) {
        return mtxbasecsr_rowcolidx(&A->storage.csr, size, rowidx, colidx);
    } else if (A->type == mtxbasedense) {
        return mtxbasedense_rowcolidx(&A->storage.dense, size, rowidx, colidx);
    } else if (A->type == mtxnullcoo) {
        return mtxnullcoo_rowcolidx(&A->storage.nullcoo, size, rowidx, colidx);
    } else if (A->type == mtxompcsr) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompcsr_rowcolidx(&A->storage.ompcsr, size, rowidx, colidx);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_MATRIX_TYPE; }
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
    if (matrix->type == mtxblasdense) {
#ifdef LIBMTX_HAVE_BLAS
        mtxblasdense_free(&matrix->storage.blas);
#endif
    } else if (matrix->type == mtxbasecoo) {
        mtxbasecoo_free(&matrix->storage.coo);
    } else if (matrix->type == mtxbasecsr) {
        mtxbasecsr_free(&matrix->storage.csr);
    } else if (matrix->type == mtxbasedense) {
        mtxbasedense_free(&matrix->storage.dense);
    } else if (matrix->type == mtxnullcoo) {
        mtxnullcoo_free(&matrix->storage.nullcoo);
    } else if (matrix->type == mtxompcsr) {
#ifdef LIBMTX_HAVE_OPENMP
        mtxompcsr_free(&matrix->storage.ompcsr);
#endif
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
    if (src->type == mtxblasdense) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasdense_alloc_copy(
            &dst->storage.blas, &src->storage.blas);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (src->type == mtxbasecoo) {
        return mtxbasecoo_alloc_copy(
            &dst->storage.coo, &src->storage.coo);
    } else if (src->type == mtxbasecsr) {
        return mtxbasecsr_alloc_copy(
            &dst->storage.csr, &src->storage.csr);
    } else if (src->type == mtxbasedense) {
        return mtxbasedense_alloc_copy(
            &dst->storage.dense, &src->storage.dense);
    } else if (src->type == mtxnullcoo) {
        return mtxnullcoo_alloc_copy(
            &dst->storage.nullcoo, &src->storage.nullcoo);
    } else if (src->type == mtxompcsr) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompcsr_alloc_copy(
            &dst->storage.ompcsr, &src->storage.ompcsr);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
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
    if (src->type == mtxblasdense) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasdense_init_copy(
            &dst->storage.blas, &src->storage.blas);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (src->type == mtxbasecoo) {
        return mtxbasecoo_init_copy(
            &dst->storage.coo, &src->storage.coo);
    } else if (src->type == mtxbasecsr) {
        return mtxbasecsr_init_copy(
            &dst->storage.csr, &src->storage.csr);
    } else if (src->type == mtxbasedense) {
        return mtxbasedense_init_copy(
            &dst->storage.dense, &src->storage.dense);
    } else if (src->type == mtxnullcoo) {
        return mtxnullcoo_init_copy(
            &dst->storage.nullcoo, &src->storage.nullcoo);
    } else if (src->type == mtxompcsr) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompcsr_init_copy(
            &dst->storage.ompcsr, &src->storage.ompcsr);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
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
    if (type == mtxblasdense) {
#ifdef LIBMTX_HAVE_BLAS
        A->type = mtxblasdense;
        return mtxblasdense_alloc_entries(
            &A->storage.blas, field, precision, symmetry,
            num_rows, num_columns, num_nonzeros,
            idxstride, idxbase, rowidx, colidx);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (type == mtxbasecoo) {
        A->type = mtxbasecoo;
        return mtxbasecoo_alloc_entries(
            &A->storage.coo, field, precision, symmetry,
            num_rows, num_columns, num_nonzeros,
            idxstride, idxbase, rowidx, colidx);
    } else if (type == mtxbasecsr) {
        A->type = mtxbasecsr;
        return mtxbasecsr_alloc_entries(
            &A->storage.csr, field, precision, symmetry,
            num_rows, num_columns, num_nonzeros,
            idxstride, idxbase, rowidx, colidx, NULL);
    } else if (type == mtxbasedense) {
        A->type = mtxbasedense;
        return mtxbasedense_alloc_entries(
            &A->storage.dense, field, precision, symmetry,
            num_rows, num_columns, num_nonzeros,
            idxstride, idxbase, rowidx, colidx);
    } else if (type == mtxnullcoo) {
        A->type = mtxnullcoo;
        return mtxnullcoo_alloc_entries(
            &A->storage.nullcoo, field, precision, symmetry,
            num_rows, num_columns, num_nonzeros,
            idxstride, idxbase, rowidx, colidx);
    } else if (type == mtxompcsr) {
#ifdef LIBMTX_HAVE_OPENMP
        A->type = mtxompcsr;
        return mtxompcsr_alloc_entries(
            &A->storage.ompcsr, field, precision, symmetry,
            num_rows, num_columns, num_nonzeros,
            idxstride, idxbase, rowidx, colidx, NULL);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
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
    if (type == mtxblasdense) {
#ifdef LIBMTX_HAVE_BLAS
        A->type = mtxblasdense;
        return mtxblasdense_init_entries_real_single(
            &A->storage.blas, symmetry,
            num_rows, num_columns, num_nonzeros, rowidx, colidx, data);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (type == mtxbasecoo) {
        A->type = mtxbasecoo;
        return mtxbasecoo_init_entries_real_single(
            &A->storage.coo, symmetry,
            num_rows, num_columns, num_nonzeros, rowidx, colidx, data);
    } else if (type == mtxbasecsr) {
        A->type = mtxbasecsr;
        return mtxbasecsr_init_entries_real_single(
            &A->storage.csr, symmetry,
            num_rows, num_columns, num_nonzeros, rowidx, colidx, data);
    } else if (type == mtxbasedense) {
        A->type = mtxbasedense;
        return mtxbasedense_init_entries_real_single(
            &A->storage.dense, symmetry,
            num_rows, num_columns, num_nonzeros, rowidx, colidx, data);
    } else if (type == mtxnullcoo) {
        A->type = mtxnullcoo;
        return mtxnullcoo_init_entries_real_single(
            &A->storage.nullcoo, symmetry,
            num_rows, num_columns, num_nonzeros, rowidx, colidx, data);
    } else if (type == mtxompcsr) {
#ifdef LIBMTX_HAVE_OPENMP
        A->type = mtxompcsr;
        return mtxompcsr_init_entries_real_single(
            &A->storage.ompcsr, symmetry,
            num_rows, num_columns, num_nonzeros, rowidx, colidx, data);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
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
    if (type == mtxblasdense) {
#ifdef LIBMTX_HAVE_BLAS
        A->type = mtxblasdense;
        return mtxblasdense_init_entries_real_double(
            &A->storage.blas, symmetry,
            num_rows, num_columns, num_nonzeros, rowidx, colidx, data);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (type == mtxbasecoo) {
        A->type = mtxbasecoo;
        return mtxbasecoo_init_entries_real_double(
            &A->storage.coo, symmetry,
            num_rows, num_columns, num_nonzeros, rowidx, colidx, data);
    } else if (type == mtxbasecsr) {
        A->type = mtxbasecsr;
        return mtxbasecsr_init_entries_real_double(
            &A->storage.csr, symmetry,
            num_rows, num_columns, num_nonzeros, rowidx, colidx, data);
    } else if (type == mtxbasedense) {
        A->type = mtxbasedense;
        return mtxbasedense_init_entries_real_double(
            &A->storage.dense, symmetry,
            num_rows, num_columns, num_nonzeros, rowidx, colidx, data);
    } else if (type == mtxnullcoo) {
        A->type = mtxnullcoo;
        return mtxnullcoo_init_entries_real_double(
            &A->storage.nullcoo, symmetry,
            num_rows, num_columns, num_nonzeros, rowidx, colidx, data);
    } else if (type == mtxompcsr) {
#ifdef LIBMTX_HAVE_OPENMP
        A->type = mtxompcsr;
        return mtxompcsr_init_entries_real_double(
            &A->storage.ompcsr, symmetry,
            num_rows, num_columns, num_nonzeros, rowidx, colidx, data);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
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
    if (type == mtxblasdense) {
#ifdef LIBMTX_HAVE_BLAS
        A->type = mtxblasdense;
        return mtxblasdense_init_entries_complex_single(
            &A->storage.blas, symmetry,
            num_rows, num_columns, num_nonzeros, rowidx, colidx, data);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (type == mtxbasecoo) {
        A->type = mtxbasecoo;
        return mtxbasecoo_init_entries_complex_single(
            &A->storage.coo, symmetry,
            num_rows, num_columns, num_nonzeros, rowidx, colidx, data);
    } else if (type == mtxbasecsr) {
        A->type = mtxbasecsr;
        return mtxbasecsr_init_entries_complex_single(
            &A->storage.csr, symmetry,
            num_rows, num_columns, num_nonzeros, rowidx, colidx, data);
    } else if (type == mtxbasedense) {
        A->type = mtxbasedense;
        return mtxbasedense_init_entries_complex_single(
            &A->storage.dense, symmetry,
            num_rows, num_columns, num_nonzeros, rowidx, colidx, data);
    } else if (type == mtxnullcoo) {
        A->type = mtxnullcoo;
        return mtxnullcoo_init_entries_complex_single(
            &A->storage.nullcoo, symmetry,
            num_rows, num_columns, num_nonzeros, rowidx, colidx, data);
    } else if (type == mtxompcsr) {
#ifdef LIBMTX_HAVE_OPENMP
        A->type = mtxompcsr;
        return mtxompcsr_init_entries_complex_single(
            &A->storage.ompcsr, symmetry,
            num_rows, num_columns, num_nonzeros, rowidx, colidx, data);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
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
    if (type == mtxblasdense) {
#ifdef LIBMTX_HAVE_BLAS
        A->type = mtxblasdense;
        return mtxblasdense_init_entries_complex_double(
            &A->storage.blas, symmetry,
            num_rows, num_columns, num_nonzeros, rowidx, colidx, data);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (type == mtxbasecoo) {
        A->type = mtxbasecoo;
        return mtxbasecoo_init_entries_complex_double(
            &A->storage.coo, symmetry,
            num_rows, num_columns, num_nonzeros, rowidx, colidx, data);
    } else if (type == mtxbasecsr) {
        A->type = mtxbasecsr;
        return mtxbasecsr_init_entries_complex_double(
            &A->storage.csr, symmetry,
            num_rows, num_columns, num_nonzeros, rowidx, colidx, data);
    } else if (type == mtxbasedense) {
        A->type = mtxbasedense;
        return mtxbasedense_init_entries_complex_double(
            &A->storage.dense, symmetry,
            num_rows, num_columns, num_nonzeros, rowidx, colidx, data);
    } else if (type == mtxnullcoo) {
        A->type = mtxnullcoo;
        return mtxnullcoo_init_entries_complex_double(
            &A->storage.nullcoo, symmetry,
            num_rows, num_columns, num_nonzeros, rowidx, colidx, data);
    } else if (type == mtxompcsr) {
#ifdef LIBMTX_HAVE_OPENMP
        A->type = mtxompcsr;
        return mtxompcsr_init_entries_complex_double(
            &A->storage.ompcsr, symmetry,
            num_rows, num_columns, num_nonzeros, rowidx, colidx, data);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
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
    if (type == mtxblasdense) {
#ifdef LIBMTX_HAVE_BLAS
        A->type = mtxblasdense;
        return mtxblasdense_init_entries_integer_single(
            &A->storage.blas, symmetry,
            num_rows, num_columns, num_nonzeros, rowidx, colidx, data);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (type == mtxbasecoo) {
        A->type = mtxbasecoo;
        return mtxbasecoo_init_entries_integer_single(
            &A->storage.coo, symmetry,
            num_rows, num_columns, num_nonzeros, rowidx, colidx, data);
    } else if (type == mtxbasecsr) {
        A->type = mtxbasecsr;
        return mtxbasecsr_init_entries_integer_single(
            &A->storage.csr, symmetry,
            num_rows, num_columns, num_nonzeros, rowidx, colidx, data);
    } else if (type == mtxbasedense) {
        A->type = mtxbasedense;
        return mtxbasedense_init_entries_integer_single(
            &A->storage.dense, symmetry,
            num_rows, num_columns, num_nonzeros, rowidx, colidx, data);
    } else if (type == mtxnullcoo) {
        A->type = mtxnullcoo;
        return mtxnullcoo_init_entries_integer_single(
            &A->storage.nullcoo, symmetry,
            num_rows, num_columns, num_nonzeros, rowidx, colidx, data);
    } else if (type == mtxompcsr) {
#ifdef LIBMTX_HAVE_OPENMP
        A->type = mtxompcsr;
        return mtxompcsr_init_entries_integer_single(
            &A->storage.ompcsr, symmetry,
            num_rows, num_columns, num_nonzeros, rowidx, colidx, data);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
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
    if (type == mtxblasdense) {
#ifdef LIBMTX_HAVE_BLAS
        A->type = mtxblasdense;
        return mtxblasdense_init_entries_integer_double(
            &A->storage.blas, symmetry,
            num_rows, num_columns, num_nonzeros, rowidx, colidx, data);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (type == mtxbasecoo) {
        A->type = mtxbasecoo;
        return mtxbasecoo_init_entries_integer_double(
            &A->storage.coo, symmetry,
            num_rows, num_columns, num_nonzeros, rowidx, colidx, data);
    } else if (type == mtxbasecsr) {
        A->type = mtxbasecsr;
        return mtxbasecsr_init_entries_integer_double(
            &A->storage.csr, symmetry,
            num_rows, num_columns, num_nonzeros, rowidx, colidx, data);
    } else if (type == mtxbasedense) {
        A->type = mtxbasedense;
        return mtxbasedense_init_entries_integer_double(
            &A->storage.dense, symmetry,
            num_rows, num_columns, num_nonzeros, rowidx, colidx, data);
    } else if (type == mtxnullcoo) {
        A->type = mtxnullcoo;
        return mtxnullcoo_init_entries_integer_double(
            &A->storage.nullcoo, symmetry,
            num_rows, num_columns, num_nonzeros, rowidx, colidx, data);
    } else if (type == mtxompcsr) {
#ifdef LIBMTX_HAVE_OPENMP
        A->type = mtxompcsr;
        return mtxompcsr_init_entries_integer_double(
            &A->storage.ompcsr, symmetry,
            num_rows, num_columns, num_nonzeros, rowidx, colidx, data);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
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
    if (type == mtxblasdense) {
#ifdef LIBMTX_HAVE_BLAS
        A->type = mtxblasdense;
        return mtxblasdense_init_entries_pattern(
            &A->storage.blas, symmetry,
            num_rows, num_columns, num_nonzeros, rowidx, colidx);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (type == mtxbasecoo) {
        A->type = mtxbasecoo;
        return mtxbasecoo_init_entries_pattern(
            &A->storage.coo, symmetry,
            num_rows, num_columns, num_nonzeros, rowidx, colidx);
    } else if (type == mtxbasecsr) {
        A->type = mtxbasecsr;
        return mtxbasecsr_init_entries_pattern(
            &A->storage.csr, symmetry,
            num_rows, num_columns, num_nonzeros, rowidx, colidx);
    } else if (type == mtxbasedense) {
        A->type = mtxbasedense;
        return mtxbasedense_init_entries_pattern(
            &A->storage.dense, symmetry,
            num_rows, num_columns, num_nonzeros, rowidx, colidx);
    } else if (type == mtxnullcoo) {
        A->type = mtxnullcoo;
        return mtxnullcoo_init_entries_pattern(
            &A->storage.nullcoo, symmetry,
            num_rows, num_columns, num_nonzeros, rowidx, colidx);
    } else if (type == mtxompcsr) {
#ifdef LIBMTX_HAVE_OPENMP
        A->type = mtxompcsr;
        return mtxompcsr_init_entries_pattern(
            &A->storage.ompcsr, symmetry,
            num_rows, num_columns, num_nonzeros, rowidx, colidx);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
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
    if (A->type == mtxblasdense) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasdense_setzero(&A->storage.blas);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (A->type == mtxbasecoo) {
        return mtxbasecoo_setzero(&A->storage.coo);
    } else if (A->type == mtxbasecsr) {
        return mtxbasecsr_setzero(&A->storage.csr);
    } else if (A->type == mtxbasedense) {
        return mtxbasedense_setzero(&A->storage.dense);
    } else if (A->type == mtxnullcoo) {
        return mtxnullcoo_setzero(&A->storage.nullcoo);
    } else if (A->type == mtxompcsr) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompcsr_setzero(&A->storage.ompcsr);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
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
    if (A->type == mtxblasdense) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasdense_set_real_single(
            &A->storage.blas, size, stride, a);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (A->type == mtxbasecoo) {
        return mtxbasecoo_set_real_single(
            &A->storage.coo, size, stride, a);
    } else if (A->type == mtxbasecsr) {
        return mtxbasecsr_set_real_single(
            &A->storage.csr, size, stride, a);
    } else if (A->type == mtxbasedense) {
        return mtxbasedense_set_real_single(
            &A->storage.dense, size, stride, a);
    } else if (A->type == mtxnullcoo) {
        return mtxnullcoo_set_real_single(
            &A->storage.nullcoo, size, stride, a);
    } else if (A->type == mtxompcsr) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompcsr_set_real_single(
            &A->storage.ompcsr, size, stride, a);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
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
    if (A->type == mtxblasdense) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasdense_set_real_double(
            &A->storage.blas, size, stride, a);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (A->type == mtxbasecoo) {
        return mtxbasecoo_set_real_double(
            &A->storage.coo, size, stride, a);
    } else if (A->type == mtxbasecsr) {
        return mtxbasecsr_set_real_double(
            &A->storage.csr, size, stride, a);
    } else if (A->type == mtxbasedense) {
        return mtxbasedense_set_real_double(
            &A->storage.dense, size, stride, a);
    } else if (A->type == mtxnullcoo) {
        return mtxnullcoo_set_real_double(
            &A->storage.nullcoo, size, stride, a);
    } else if (A->type == mtxompcsr) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompcsr_set_real_double(
            &A->storage.ompcsr, size, stride, a);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
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
    if (A->type == mtxblasdense) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasdense_set_complex_single(
            &A->storage.blas, size, stride, a);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (A->type == mtxbasecoo) {
        return mtxbasecoo_set_complex_single(
            &A->storage.coo, size, stride, a);
    } else if (A->type == mtxbasecsr) {
        return mtxbasecsr_set_complex_single(
            &A->storage.csr, size, stride, a);
    } else if (A->type == mtxbasedense) {
        return mtxbasedense_set_complex_single(
            &A->storage.dense, size, stride, a);
    } else if (A->type == mtxnullcoo) {
        return mtxnullcoo_set_complex_single(
            &A->storage.nullcoo, size, stride, a);
    } else if (A->type == mtxompcsr) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompcsr_set_complex_single(
            &A->storage.ompcsr, size, stride, a);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
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
    if (A->type == mtxblasdense) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasdense_set_complex_double(
            &A->storage.blas, size, stride, a);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (A->type == mtxbasecoo) {
        return mtxbasecoo_set_complex_double(
            &A->storage.coo, size, stride, a);
    } else if (A->type == mtxbasecsr) {
        return mtxbasecsr_set_complex_double(
            &A->storage.csr, size, stride, a);
    } else if (A->type == mtxbasedense) {
        return mtxbasedense_set_complex_double(
            &A->storage.dense, size, stride, a);
    } else if (A->type == mtxnullcoo) {
        return mtxnullcoo_set_complex_double(
            &A->storage.nullcoo, size, stride, a);
    } else if (A->type == mtxompcsr) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompcsr_set_complex_double(
            &A->storage.ompcsr, size, stride, a);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
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
    if (A->type == mtxblasdense) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasdense_set_integer_single(
            &A->storage.blas, size, stride, a);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (A->type == mtxbasecoo) {
        return mtxbasecoo_set_integer_single(
            &A->storage.coo, size, stride, a);
    } else if (A->type == mtxbasecsr) {
        return mtxbasecsr_set_integer_single(
            &A->storage.csr, size, stride, a);
    } else if (A->type == mtxbasedense) {
        return mtxbasedense_set_integer_single(
            &A->storage.dense, size, stride, a);
    } else if (A->type == mtxnullcoo) {
        return mtxnullcoo_set_integer_single(
            &A->storage.nullcoo, size, stride, a);
    } else if (A->type == mtxompcsr) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompcsr_set_integer_single(
            &A->storage.ompcsr, size, stride, a);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
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
    if (A->type == mtxblasdense) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasdense_set_integer_double(
            &A->storage.blas, size, stride, a);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (A->type == mtxbasecoo) {
        return mtxbasecoo_set_integer_double(
            &A->storage.coo, size, stride, a);
    } else if (A->type == mtxbasecsr) {
        return mtxbasecsr_set_integer_double(
            &A->storage.csr, size, stride, a);
    } else if (A->type == mtxbasedense) {
        return mtxbasedense_set_integer_double(
            &A->storage.dense, size, stride, a);
    } else if (A->type == mtxnullcoo) {
        return mtxnullcoo_set_integer_double(
            &A->storage.nullcoo, size, stride, a);
    } else if (A->type == mtxompcsr) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompcsr_set_integer_double(
            &A->storage.ompcsr, size, stride, a);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
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
    if (matrix->type == mtxblasdense) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasdense_alloc_row_vector(
            &matrix->storage.blas, vector, vectortype);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (matrix->type == mtxbasecoo) {
        return mtxbasecoo_alloc_row_vector(
            &matrix->storage.coo, vector, vectortype);
    } else if (matrix->type == mtxbasecsr) {
        return mtxbasecsr_alloc_row_vector(
            &matrix->storage.csr, vector, vectortype);
    } else if (matrix->type == mtxbasedense) {
        return mtxbasedense_alloc_row_vector(
            &matrix->storage.dense, vector, vectortype);
    } else if (matrix->type == mtxnullcoo) {
        return mtxnullcoo_alloc_row_vector(
            &matrix->storage.nullcoo, vector, vectortype);
    } else if (matrix->type == mtxompcsr) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompcsr_alloc_row_vector(
            &matrix->storage.ompcsr, vector, vectortype);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
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
    if (matrix->type == mtxblasdense) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasdense_alloc_column_vector(
            &matrix->storage.blas, vector, vectortype);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (matrix->type == mtxbasecoo) {
        return mtxbasecoo_alloc_column_vector(
            &matrix->storage.coo, vector, vectortype);
    } else if (matrix->type == mtxbasecsr) {
        return mtxbasecsr_alloc_column_vector(
            &matrix->storage.csr, vector, vectortype);
    } else if (matrix->type == mtxbasedense) {
        return mtxbasedense_alloc_column_vector(
            &matrix->storage.dense, vector, vectortype);
    } else if (matrix->type == mtxnullcoo) {
        return mtxnullcoo_alloc_column_vector(
            &matrix->storage.nullcoo, vector, vectortype);
    } else if (matrix->type == mtxompcsr) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompcsr_alloc_column_vector(
            &matrix->storage.ompcsr, vector, vectortype);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
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
    if (type == mtxblasdense) {
#ifdef LIBMTX_HAVE_BLAS
        matrix->type = mtxblasdense;
        return mtxblasdense_from_mtxfile(
            &matrix->storage.blas, mtxfile);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (type == mtxbasecoo) {
        matrix->type = mtxbasecoo;
        return mtxbasecoo_from_mtxfile(
            &matrix->storage.coo, mtxfile);
    } else if (type == mtxbasecsr) {
        matrix->type = mtxbasecsr;
        return mtxbasecsr_from_mtxfile(
            &matrix->storage.csr, mtxfile);
    } else if (type == mtxbasedense) {
        matrix->type = mtxbasedense;
        return mtxbasedense_from_mtxfile(
            &matrix->storage.dense, mtxfile);
    } else if (type == mtxnullcoo) {
        matrix->type = mtxnullcoo;
        return mtxnullcoo_from_mtxfile(
            &matrix->storage.nullcoo, mtxfile);
    } else if (type == mtxompcsr) {
#ifdef LIBMTX_HAVE_OPENMP
        matrix->type = mtxompcsr;
        return mtxompcsr_from_mtxfile(
            &matrix->storage.ompcsr, mtxfile);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
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
    if (src->type == mtxblasdense) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasdense_to_mtxfile(
            dst, &src->storage.blas,
            num_rows, rowidx, num_columns, colidx, mtxfmt);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (src->type == mtxbasecoo) {
        return mtxbasecoo_to_mtxfile(
            dst, &src->storage.coo,
            num_rows, rowidx, num_columns, colidx, mtxfmt);
    } else if (src->type == mtxbasecsr) {
        return mtxbasecsr_to_mtxfile(
            dst, &src->storage.csr,
            num_rows, rowidx, num_columns, colidx, mtxfmt);
    } else if (src->type == mtxbasedense) {
        return mtxbasedense_to_mtxfile(
            dst, &src->storage.dense,
            num_rows, rowidx, num_columns, colidx, mtxfmt);
    } else if (src->type == mtxnullcoo) {
        return mtxnullcoo_to_mtxfile(
            dst, &src->storage.nullcoo,
            num_rows, rowidx, num_columns, colidx, mtxfmt);
    } else if (src->type == mtxompcsr) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompcsr_to_mtxfile(
            dst, &src->storage.ompcsr,
            num_rows, rowidx, num_columns, colidx, mtxfmt);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
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
 * partitioning
 */

/**
 * ‘mtxmatrix_partition_rowwise()’ partitions the entries of a matrix
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
int mtxmatrix_partition_rowwise(
    const struct mtxmatrix * A,
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
    if (A->type == mtxbasecoo) {
        return mtxbasecoo_partition_rowwise(
            &A->storage.coo,
            parttype, num_parts, partsizes, blksize, parts,
            dstnzpart, dstnzpartsizes, dstrowpart, dstrowpartsizes);
    } else if (A->type == mtxbasecsr) {
        return mtxbasecsr_partition_rowwise(
            &A->storage.csr,
            parttype, num_parts, partsizes, blksize, parts,
            dstnzpart, dstnzpartsizes, dstrowpart, dstrowpartsizes);
    } else if (A->type == mtxnullcoo) {
        return mtxnullcoo_partition_rowwise(
            &A->storage.nullcoo,
            parttype, num_parts, partsizes, blksize, parts,
            dstnzpart, dstnzpartsizes, dstrowpart, dstrowpartsizes);
    } else if (A->type == mtxompcsr) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompcsr_partition_rowwise(
            &A->storage.ompcsr,
            parttype, num_parts, partsizes, blksize, parts,
            dstnzpart, dstnzpartsizes, dstrowpart, dstrowpartsizes);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_MATRIX_TYPE; }
}

/**
 * ‘mtxmatrix_partition_columnwise()’ partitions the entries of a
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
int mtxmatrix_partition_columnwise(
    const struct mtxmatrix * A,
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
    if (A->type == mtxbasecoo) {
        return mtxbasecoo_partition_columnwise(
            &A->storage.coo,
            parttype, num_parts, partsizes, blksize, parts,
            dstnzpart, dstnzpartsizes, dstcolpart, dstcolpartsizes);
    } else if (A->type == mtxbasecsr) {
        return mtxbasecsr_partition_columnwise(
            &A->storage.csr,
            parttype, num_parts, partsizes, blksize, parts,
            dstnzpart, dstnzpartsizes, dstcolpart, dstcolpartsizes);
    } else if (A->type == mtxnullcoo) {
        return mtxnullcoo_partition_columnwise(
            &A->storage.nullcoo,
            parttype, num_parts, partsizes, blksize, parts,
            dstnzpart, dstnzpartsizes, dstcolpart, dstcolpartsizes);
    } else if (A->type == mtxompcsr) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompcsr_partition_columnwise(
            &A->storage.ompcsr,
            parttype, num_parts, partsizes, blksize, parts,
            dstnzpart, dstnzpartsizes, dstcolpart, dstcolpartsizes);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_MATRIX_TYPE; }
}

/**
 * ‘mtxmatrix_partition_2d()’ partitions the entries of a matrix in a
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
int mtxmatrix_partition_2d(
    const struct mtxmatrix * A,
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
    if (A->type == mtxbasecoo) {
        return mtxbasecoo_partition_2d(
            &A->storage.coo,
            rowparttype, num_row_parts, rowpartsizes, rowblksize, rowparts,
            colparttype, num_col_parts, colpartsizes, colblksize, colparts,
            dstnzpart, dstnzpartsizes,
            dstrowpart, dstrowpartsizes,
            dstcolpart, dstcolpartsizes);
    } else if (A->type == mtxbasecsr) {
        return mtxbasecsr_partition_2d(
            &A->storage.csr,
            rowparttype, num_row_parts, rowpartsizes, rowblksize, rowparts,
            colparttype, num_col_parts, colpartsizes, colblksize, colparts,
            dstnzpart, dstnzpartsizes,
            dstrowpart, dstrowpartsizes,
            dstcolpart, dstcolpartsizes);
    } else if (A->type == mtxnullcoo) {
        return mtxnullcoo_partition_2d(
            &A->storage.nullcoo,
            rowparttype, num_row_parts, rowpartsizes, rowblksize, rowparts,
            colparttype, num_col_parts, colpartsizes, colblksize, colparts,
            dstnzpart, dstnzpartsizes,
            dstrowpart, dstrowpartsizes,
            dstcolpart, dstcolpartsizes);
    } else if (A->type == mtxompcsr) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompcsr_partition_2d(
            &A->storage.ompcsr,
            rowparttype, num_row_parts, rowpartsizes, rowblksize, rowparts,
            colparttype, num_col_parts, colpartsizes, colblksize, colparts,
            dstnzpart, dstnzpartsizes,
            dstrowpart, dstrowpartsizes,
            dstcolpart, dstcolpartsizes);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_MATRIX_TYPE; }
}

/**
 * ‘mtxmatrix_split()’ splits a matrix into multiple matrices
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
 * of type ‘struct mtxmatrix’. If successful, then ‘dsts[p]’ points to
 * a matrix consisting of elements from ‘src’ that belong to the ‘p’th
 * part, as designated by the ‘parts’ array.
 *
 * The caller is responsible for calling ‘mtxmatrix_free()’ to free
 * storage allocated for each matrix in the ‘dsts’ array.
 */
int mtxmatrix_split(
    int num_parts,
    struct mtxmatrix ** dsts,
    const struct mtxmatrix * src,
    int64_t size,
    int * parts)
{
    if (src->type == mtxbasecoo) {
        struct mtxbasecoo ** coodsts = malloc(
            num_parts * sizeof(struct mtxbasecoo *));
        if (!coodsts) return MTX_ERR_ERRNO;
        for (int p = 0; p < num_parts; p++) {
            dsts[p]->type = mtxbasecoo;
            coodsts[p] = &dsts[p]->storage.coo;
        }
        int err = mtxbasecoo_split(
            num_parts, coodsts, &src->storage.coo, size, parts);
        free(coodsts);
        return err;
    } else if (src->type == mtxbasecsr) {
        struct mtxbasecsr ** csrdsts = malloc(
            num_parts * sizeof(struct mtxbasecsr *));
        if (!csrdsts) return MTX_ERR_ERRNO;
        for (int p = 0; p < num_parts; p++) {
            dsts[p]->type = mtxbasecsr;
            csrdsts[p] = &dsts[p]->storage.csr;
        }
        int err = mtxbasecsr_split(
            num_parts, csrdsts, &src->storage.csr, size, parts);
        free(csrdsts);
        return err;
    } else if (src->type == mtxnullcoo) {
        struct mtxnullcoo ** nullcoodsts = malloc(
            num_parts * sizeof(struct mtxnullcoo *));
        if (!nullcoodsts) return MTX_ERR_ERRNO;
        for (int p = 0; p < num_parts; p++) {
            dsts[p]->type = mtxnullcoo;
            nullcoodsts[p] = &dsts[p]->storage.nullcoo;
        }
        int err = mtxnullcoo_split(
            num_parts, nullcoodsts, &src->storage.nullcoo, size, parts);
        free(nullcoodsts);
        return err;
    } else if (src->type == mtxompcsr) {
#ifdef LIBMTX_HAVE_OPENMP
        struct mtxompcsr ** ompcsrdsts = malloc(
            num_parts * sizeof(struct mtxompcsr *));
        if (!ompcsrdsts) return MTX_ERR_ERRNO;
        for (int p = 0; p < num_parts; p++) {
            dsts[p]->type = mtxompcsr;
            ompcsrdsts[p] = &dsts[p]->storage.ompcsr;
        }
        int err = mtxompcsr_split(
            num_parts, ompcsrdsts, &src->storage.ompcsr, size, parts);
        free(ompcsrdsts);
        return err;
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_MATRIX_TYPE; }
}

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
    if (X->type == mtxblasdense && Y->type == mtxblasdense) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasdense_swap(
            &X->storage.blas, &Y->storage.blas);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (X->type == mtxbasecoo && Y->type == mtxbasecoo) {
        return mtxbasecoo_swap(
            &X->storage.coo, &Y->storage.coo);
    } else if (X->type == mtxbasecsr && Y->type == mtxbasecsr) {
        return mtxbasecsr_swap(
            &X->storage.csr, &Y->storage.csr);
    } else if (X->type == mtxbasedense && Y->type == mtxbasedense) {
        return mtxbasedense_swap(
            &X->storage.dense, &Y->storage.dense);
    } else if (X->type == mtxnullcoo && Y->type == mtxnullcoo) {
        return mtxnullcoo_swap(
            &X->storage.nullcoo, &Y->storage.nullcoo);
    } else if (X->type == mtxompcsr && Y->type == mtxompcsr) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompcsr_swap(
            &X->storage.ompcsr, &Y->storage.ompcsr);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_MATRIX_TYPE; }
}

/**
 * ‘mtxmatrix_copy()’ copies values of a matrix, ‘Y = X’.
 */
int mtxmatrix_copy(
    struct mtxmatrix * Y,
    const struct mtxmatrix * X)
{
    if (X->type == mtxblasdense && Y->type == mtxblasdense) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasdense_copy(
            &Y->storage.blas, &X->storage.blas);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (X->type == mtxbasecoo && Y->type == mtxbasecoo) {
        return mtxbasecoo_copy(
            &Y->storage.coo, &X->storage.coo);
    } else if (X->type == mtxbasecsr && Y->type == mtxbasecsr) {
        return mtxbasecsr_copy(
            &Y->storage.csr, &X->storage.csr);
    } else if (X->type == mtxbasedense && Y->type == mtxbasedense) {
        return mtxbasedense_copy(
            &Y->storage.dense, &X->storage.dense);
    } else if (X->type == mtxnullcoo && Y->type == mtxnullcoo) {
        return mtxnullcoo_copy(
            &Y->storage.nullcoo, &X->storage.nullcoo);
    } else if (X->type == mtxompcsr && Y->type == mtxompcsr) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompcsr_copy(
            &Y->storage.ompcsr, &X->storage.ompcsr);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
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
    if (X->type == mtxblasdense) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasdense_sscal(
            a, &X->storage.blas, num_flops);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (X->type == mtxbasecoo) {
        return mtxbasecoo_sscal(
            a, &X->storage.coo, num_flops);
    } else if (X->type == mtxbasecsr) {
        return mtxbasecsr_sscal(
            a, &X->storage.csr, num_flops);
    } else if (X->type == mtxbasedense) {
        return mtxbasedense_sscal(
            a, &X->storage.dense, num_flops);
    } else if (X->type == mtxnullcoo) {
        return mtxnullcoo_sscal(
            a, &X->storage.nullcoo, num_flops);
    } else if (X->type == mtxompcsr) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompcsr_sscal(
            a, &X->storage.ompcsr, num_flops);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
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
    if (X->type == mtxblasdense) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasdense_dscal(
            a, &X->storage.blas, num_flops);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (X->type == mtxbasecoo) {
        return mtxbasecoo_dscal(
            a, &X->storage.coo, num_flops);
    } else if (X->type == mtxbasecsr) {
        return mtxbasecsr_dscal(
            a, &X->storage.csr, num_flops);
    } else if (X->type == mtxbasedense) {
        return mtxbasedense_dscal(
            a, &X->storage.dense, num_flops);
    } else if (X->type == mtxnullcoo) {
        return mtxnullcoo_dscal(
            a, &X->storage.nullcoo, num_flops);
    } else if (X->type == mtxompcsr) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompcsr_dscal(
            a, &X->storage.ompcsr, num_flops);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
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
    if (X->type == mtxblasdense) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasdense_cscal(
            a, &X->storage.blas, num_flops);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (X->type == mtxbasecoo) {
        return mtxbasecoo_cscal(
            a, &X->storage.coo, num_flops);
    } else if (X->type == mtxbasedense) {
        return mtxbasedense_cscal(
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
    if (X->type == mtxblasdense) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasdense_zscal(
            a, &X->storage.blas, num_flops);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (X->type == mtxbasecoo) {
        return mtxbasecoo_zscal(
            a, &X->storage.coo, num_flops);
    } else if (X->type == mtxbasedense) {
        return mtxbasedense_zscal(
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
    if (X->type == mtxblasdense && Y->type == mtxblasdense) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasdense_saxpy(
            a, &X->storage.blas, &Y->storage.blas, num_flops);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (X->type == mtxbasecoo && Y->type == mtxbasecoo) {
        return mtxbasecoo_saxpy(
            a, &X->storage.coo, &Y->storage.coo, num_flops);
    } else if (X->type == mtxbasecsr && Y->type == mtxbasecsr) {
        return mtxbasecsr_saxpy(
            a, &X->storage.csr, &Y->storage.csr, num_flops);
    } else if (X->type == mtxbasedense && Y->type == mtxbasedense) {
        return mtxbasedense_saxpy(
            a, &X->storage.dense, &Y->storage.dense, num_flops);
    } else if (X->type == mtxnullcoo && Y->type == mtxnullcoo) {
        return mtxnullcoo_saxpy(
            a, &X->storage.nullcoo, &Y->storage.nullcoo, num_flops);
    } else if (X->type == mtxompcsr && Y->type == mtxompcsr) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompcsr_saxpy(
            a, &X->storage.ompcsr, &Y->storage.ompcsr, num_flops);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
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
    if (X->type == mtxblasdense && Y->type == mtxblasdense) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasdense_daxpy(
            a, &X->storage.blas, &Y->storage.blas, num_flops);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (X->type == mtxbasecoo && Y->type == mtxbasecoo) {
        return mtxbasecoo_daxpy(
            a, &X->storage.coo, &Y->storage.coo, num_flops);
    } else if (X->type == mtxbasecsr && Y->type == mtxbasecsr) {
        return mtxbasecsr_daxpy(
            a, &X->storage.csr, &Y->storage.csr, num_flops);
    } else if (X->type == mtxbasedense && Y->type == mtxbasedense) {
        return mtxbasedense_daxpy(
            a, &X->storage.dense, &Y->storage.dense, num_flops);
    } else if (X->type == mtxnullcoo && Y->type == mtxnullcoo) {
        return mtxnullcoo_daxpy(
            a, &X->storage.nullcoo, &Y->storage.nullcoo, num_flops);
    } else if (X->type == mtxompcsr && Y->type == mtxompcsr) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompcsr_daxpy(
            a, &X->storage.ompcsr, &Y->storage.ompcsr, num_flops);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
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
    if (X->type == mtxblasdense && Y->type == mtxblasdense) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasdense_saypx(
            a, &Y->storage.blas, &X->storage.blas, num_flops);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (X->type == mtxbasecoo && Y->type == mtxbasecoo) {
        return mtxbasecoo_saypx(
            a, &Y->storage.coo, &X->storage.coo, num_flops);
    } else if (X->type == mtxbasecsr && Y->type == mtxbasecsr) {
        return mtxbasecsr_saypx(
            a, &Y->storage.csr, &X->storage.csr, num_flops);
    } else if (X->type == mtxbasedense && Y->type == mtxbasedense) {
        return mtxbasedense_saypx(
            a, &Y->storage.dense, &X->storage.dense, num_flops);
    } else if (X->type == mtxnullcoo && Y->type == mtxnullcoo) {
        return mtxnullcoo_saypx(
            a, &Y->storage.nullcoo, &X->storage.nullcoo, num_flops);
    } else if (X->type == mtxompcsr && Y->type == mtxompcsr) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompcsr_saypx(
            a, &Y->storage.ompcsr, &X->storage.ompcsr, num_flops);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
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
    if (X->type == mtxblasdense && Y->type == mtxblasdense) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasdense_daypx(
            a, &Y->storage.blas, &X->storage.blas, num_flops);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (X->type == mtxbasecoo && Y->type == mtxbasecoo) {
        return mtxbasecoo_daypx(
            a, &Y->storage.coo, &X->storage.coo, num_flops);
    } else if (X->type == mtxbasecsr && Y->type == mtxbasecsr) {
        return mtxbasecsr_daypx(
            a, &Y->storage.csr, &X->storage.csr, num_flops);
    } else if (X->type == mtxbasedense && Y->type == mtxbasedense) {
        return mtxbasedense_daypx(
            a, &Y->storage.dense, &X->storage.dense, num_flops);
    } else if (X->type == mtxnullcoo && Y->type == mtxnullcoo) {
        return mtxnullcoo_daypx(
            a, &Y->storage.nullcoo, &X->storage.nullcoo, num_flops);
    } else if (X->type == mtxompcsr && Y->type == mtxompcsr) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompcsr_daypx(
            a, &Y->storage.ompcsr, &X->storage.ompcsr, num_flops);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
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
    if (X->type == mtxblasdense && Y->type == mtxblasdense) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasdense_sdot(
            &X->storage.blas, &Y->storage.blas, dot, num_flops);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (X->type == mtxbasecoo && Y->type == mtxbasecoo) {
        return mtxbasecoo_sdot(
            &X->storage.coo, &Y->storage.coo, dot, num_flops);
    } else if (X->type == mtxbasecsr && Y->type == mtxbasecsr) {
        return mtxbasecsr_sdot(
            &X->storage.csr, &Y->storage.csr, dot, num_flops);
    } else if (X->type == mtxbasedense && Y->type == mtxbasedense) {
        return mtxbasedense_sdot(
            &X->storage.dense, &Y->storage.dense, dot, num_flops);
    } else if (X->type == mtxnullcoo && Y->type == mtxnullcoo) {
        return mtxnullcoo_sdot(
            &X->storage.nullcoo, &Y->storage.nullcoo, dot, num_flops);
    } else if (X->type == mtxompcsr && Y->type == mtxompcsr) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompcsr_sdot(
            &X->storage.ompcsr, &Y->storage.ompcsr, dot, num_flops);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
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
    if (X->type == mtxblasdense && Y->type == mtxblasdense) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasdense_ddot(
            &X->storage.blas, &Y->storage.blas, dot, num_flops);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (X->type == mtxbasecoo && Y->type == mtxbasecoo) {
        return mtxbasecoo_ddot(
            &X->storage.coo, &Y->storage.coo, dot, num_flops);
    } else if (X->type == mtxbasecsr && Y->type == mtxbasecsr) {
        return mtxbasecsr_ddot(
            &X->storage.csr, &Y->storage.csr, dot, num_flops);
    } else if (X->type == mtxbasedense && Y->type == mtxbasedense) {
        return mtxbasedense_ddot(
            &X->storage.dense, &Y->storage.dense, dot, num_flops);
    } else if (X->type == mtxnullcoo && Y->type == mtxnullcoo) {
        return mtxnullcoo_ddot(
            &X->storage.nullcoo, &Y->storage.nullcoo, dot, num_flops);
    } else if (X->type == mtxompcsr && Y->type == mtxompcsr) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompcsr_ddot(
            &X->storage.ompcsr, &Y->storage.ompcsr, dot, num_flops);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
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
    if (X->type == mtxblasdense && Y->type == mtxblasdense) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasdense_cdotu(
            &X->storage.blas, &Y->storage.blas, dot, num_flops);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (X->type == mtxbasecoo && Y->type == mtxbasecoo) {
        return mtxbasecoo_cdotu(
            &X->storage.coo, &Y->storage.coo, dot, num_flops);
    } else if (X->type == mtxbasecsr && Y->type == mtxbasecsr) {
        return mtxbasecsr_cdotu(
            &X->storage.csr, &Y->storage.csr, dot, num_flops);
    } else if (X->type == mtxbasedense && Y->type == mtxbasedense) {
        return mtxbasedense_cdotu(
            &X->storage.dense, &Y->storage.dense, dot, num_flops);
    } else if (X->type == mtxnullcoo && Y->type == mtxnullcoo) {
        return mtxnullcoo_cdotu(
            &X->storage.nullcoo, &Y->storage.nullcoo, dot, num_flops);
    } else if (X->type == mtxompcsr && Y->type == mtxompcsr) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompcsr_cdotu(
            &X->storage.ompcsr, &Y->storage.ompcsr, dot, num_flops);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
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
    if (X->type == mtxblasdense && Y->type == mtxblasdense) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasdense_zdotu(
            &X->storage.blas, &Y->storage.blas, dot, num_flops);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (X->type == mtxbasecoo && Y->type == mtxbasecoo) {
        return mtxbasecoo_zdotu(
            &X->storage.coo, &Y->storage.coo, dot, num_flops);
    } else if (X->type == mtxbasecsr && Y->type == mtxbasecsr) {
        return mtxbasecsr_zdotu(
            &X->storage.csr, &Y->storage.csr, dot, num_flops);
    } else if (X->type == mtxbasedense && Y->type == mtxbasedense) {
        return mtxbasedense_zdotu(
            &X->storage.dense, &Y->storage.dense, dot, num_flops);
    } else if (X->type == mtxnullcoo && Y->type == mtxnullcoo) {
        return mtxnullcoo_zdotu(
            &X->storage.nullcoo, &Y->storage.nullcoo, dot, num_flops);
    } else if (X->type == mtxompcsr && Y->type == mtxompcsr) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompcsr_zdotu(
            &X->storage.ompcsr, &Y->storage.ompcsr, dot, num_flops);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
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
    if (X->type == mtxblasdense && Y->type == mtxblasdense) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasdense_cdotc(
            &X->storage.blas, &Y->storage.blas, dot, num_flops);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (X->type == mtxbasecoo && Y->type == mtxbasecoo) {
        return mtxbasecoo_cdotc(
            &X->storage.coo, &Y->storage.coo, dot, num_flops);
    } else if (X->type == mtxbasecsr && Y->type == mtxbasecsr) {
        return mtxbasecsr_cdotc(
            &X->storage.csr, &Y->storage.csr, dot, num_flops);
    } else if (X->type == mtxbasedense && Y->type == mtxbasedense) {
        return mtxbasedense_cdotc(
            &X->storage.dense, &Y->storage.dense, dot, num_flops);
    } else if (X->type == mtxnullcoo && Y->type == mtxnullcoo) {
        return mtxnullcoo_cdotc(
            &X->storage.nullcoo, &Y->storage.nullcoo, dot, num_flops);
    } else if (X->type == mtxompcsr && Y->type == mtxompcsr) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompcsr_cdotc(
            &X->storage.ompcsr, &Y->storage.ompcsr, dot, num_flops);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
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
    if (X->type == mtxblasdense && Y->type == mtxblasdense) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasdense_zdotc(
            &X->storage.blas, &Y->storage.blas, dot, num_flops);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (X->type == mtxbasecoo && Y->type == mtxbasecoo) {
        return mtxbasecoo_zdotc(
            &X->storage.coo, &Y->storage.coo, dot, num_flops);
    } else if (X->type == mtxbasecsr && Y->type == mtxbasecsr) {
        return mtxbasecsr_zdotc(
            &X->storage.csr, &Y->storage.csr, dot, num_flops);
    } else if (X->type == mtxbasedense && Y->type == mtxbasedense) {
        return mtxbasedense_zdotc(
            &X->storage.dense, &Y->storage.dense, dot, num_flops);
    } else if (X->type == mtxnullcoo && Y->type == mtxnullcoo) {
        return mtxnullcoo_zdotc(
            &X->storage.nullcoo, &Y->storage.nullcoo, dot, num_flops);
    } else if (X->type == mtxompcsr && Y->type == mtxompcsr) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompcsr_zdotc(
            &X->storage.ompcsr, &Y->storage.ompcsr, dot, num_flops);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
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
    if (X->type == mtxblasdense) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasdense_snrm2(&X->storage.blas, nrm2, num_flops);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (X->type == mtxbasecoo) {
        return mtxbasecoo_snrm2(&X->storage.coo, nrm2, num_flops);
    } else if (X->type == mtxbasecsr) {
        return mtxbasecsr_snrm2(&X->storage.csr, nrm2, num_flops);
    } else if (X->type == mtxbasedense) {
        return mtxbasedense_snrm2(&X->storage.dense, nrm2, num_flops);
    } else if (X->type == mtxnullcoo) {
        return mtxnullcoo_snrm2(&X->storage.nullcoo, nrm2, num_flops);
    } else if (X->type == mtxompcsr) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompcsr_snrm2(&X->storage.ompcsr, nrm2, num_flops);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
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
    if (X->type == mtxblasdense) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasdense_dnrm2(&X->storage.blas, nrm2, num_flops);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (X->type == mtxbasecoo) {
        return mtxbasecoo_dnrm2(&X->storage.coo, nrm2, num_flops);
    } else if (X->type == mtxbasecsr) {
        return mtxbasecsr_dnrm2(&X->storage.csr, nrm2, num_flops);
    } else if (X->type == mtxbasedense) {
        return mtxbasedense_dnrm2(&X->storage.dense, nrm2, num_flops);
    } else if (X->type == mtxnullcoo) {
        return mtxnullcoo_dnrm2(&X->storage.nullcoo, nrm2, num_flops);
    } else if (X->type == mtxompcsr) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompcsr_dnrm2(&X->storage.ompcsr, nrm2, num_flops);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
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
    if (X->type == mtxblasdense) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasdense_sasum(&X->storage.blas, asum, num_flops);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (X->type == mtxbasecoo) {
        return mtxbasecoo_sasum(&X->storage.coo, asum, num_flops);
    } else if (X->type == mtxbasecsr) {
        return mtxbasecsr_sasum(&X->storage.csr, asum, num_flops);
    } else if (X->type == mtxbasedense) {
        return mtxbasedense_sasum(&X->storage.dense, asum, num_flops);
    } else if (X->type == mtxnullcoo) {
        return mtxnullcoo_sasum(&X->storage.nullcoo, asum, num_flops);
    } else if (X->type == mtxompcsr) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompcsr_sasum(&X->storage.ompcsr, asum, num_flops);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
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
    if (X->type == mtxblasdense) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasdense_dasum(&X->storage.blas, asum, num_flops);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (X->type == mtxbasecoo) {
        return mtxbasecoo_dasum(&X->storage.coo, asum, num_flops);
    } else if (X->type == mtxbasecsr) {
        return mtxbasecsr_dasum(&X->storage.csr, asum, num_flops);
    } else if (X->type == mtxbasedense) {
        return mtxbasedense_dasum(&X->storage.dense, asum, num_flops);
    } else if (X->type == mtxnullcoo) {
        return mtxnullcoo_dasum(&X->storage.nullcoo, asum, num_flops);
    } else if (X->type == mtxompcsr) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompcsr_dasum(&X->storage.ompcsr, asum, num_flops);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
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
    if (X->type == mtxblasdense) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasdense_iamax(&X->storage.blas, iamax);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (X->type == mtxbasecoo) {
        return mtxbasecoo_iamax(&X->storage.coo, iamax);
    } else if (X->type == mtxbasecsr) {
        return mtxbasecsr_iamax(&X->storage.csr, iamax);
    } else if (X->type == mtxbasedense) {
        return mtxbasedense_iamax(&X->storage.dense, iamax);
    } else if (X->type == mtxnullcoo) {
        return mtxnullcoo_iamax(&X->storage.nullcoo, iamax);
    } else if (X->type == mtxompcsr) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompcsr_iamax(&X->storage.ompcsr, iamax);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
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
    if (A->type == mtxblasdense) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasdense_sgemv(
            trans, alpha, &A->storage.blas, x, beta, y, num_flops);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (A->type == mtxbasecoo) {
        return mtxbasecoo_sgemv(
            trans, alpha, &A->storage.coo, x, beta, y, num_flops);
    } else if (A->type == mtxbasecsr) {
        return mtxbasecsr_sgemv(
            trans, alpha, &A->storage.csr, x, beta, y, num_flops);
    } else if (A->type == mtxbasedense) {
        return mtxbasedense_sgemv(
            trans, alpha, &A->storage.dense, x, beta, y, num_flops);
    } else if (A->type == mtxnullcoo) {
        return mtxnullcoo_sgemv(
            trans, alpha, &A->storage.nullcoo, x, beta, y, num_flops);
    } else if (A->type == mtxompcsr) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompcsr_sgemv(
            trans, alpha, &A->storage.ompcsr, x, beta, y, num_flops);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
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
    if (A->type == mtxblasdense) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasdense_dgemv(
            trans, alpha, &A->storage.blas, x, beta, y, num_flops);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (A->type == mtxbasecoo) {
        return mtxbasecoo_dgemv(
            trans, alpha, &A->storage.coo, x, beta, y, num_flops);
    } else if (A->type == mtxbasecsr) {
        return mtxbasecsr_dgemv(
            trans, alpha, &A->storage.csr, x, beta, y, num_flops);
    } else if (A->type == mtxbasedense) {
        return mtxbasedense_dgemv(
            trans, alpha, &A->storage.dense, x, beta, y, num_flops);
    } else if (A->type == mtxnullcoo) {
        return mtxnullcoo_dgemv(
            trans, alpha, &A->storage.nullcoo, x, beta, y, num_flops);
    } else if (A->type == mtxompcsr) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompcsr_dgemv(
            trans, alpha, &A->storage.ompcsr, x, beta, y, num_flops);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
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
    if (A->type == mtxblasdense) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasdense_cgemv(
            trans, alpha, &A->storage.blas, x, beta, y, num_flops);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (A->type == mtxbasecoo) {
        return mtxbasecoo_cgemv(
            trans, alpha, &A->storage.coo, x, beta, y, num_flops);
    } else if (A->type == mtxbasecsr) {
        return mtxbasecsr_cgemv(
            trans, alpha, &A->storage.csr, x, beta, y, num_flops);
    } else if (A->type == mtxbasedense) {
        return mtxbasedense_cgemv(
            trans, alpha, &A->storage.dense, x, beta, y, num_flops);
    } else if (A->type == mtxnullcoo) {
        return mtxnullcoo_cgemv(
            trans, alpha, &A->storage.nullcoo, x, beta, y, num_flops);
    } else if (A->type == mtxompcsr) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompcsr_cgemv(
            trans, alpha, &A->storage.ompcsr, x, beta, y, num_flops);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
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
    if (A->type == mtxblasdense) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasdense_zgemv(
            trans, alpha, &A->storage.blas, x, beta, y, num_flops);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (A->type == mtxbasecoo) {
        return mtxbasecoo_zgemv(
            trans, alpha, &A->storage.coo, x, beta, y, num_flops);
    } else if (A->type == mtxbasecsr) {
        return mtxbasecsr_zgemv(
            trans, alpha, &A->storage.csr, x, beta, y, num_flops);
    } else if (A->type == mtxbasedense) {
        return mtxbasedense_zgemv(
            trans, alpha, &A->storage.dense, x, beta, y, num_flops);
    } else if (A->type == mtxnullcoo) {
        return mtxnullcoo_zgemv(
            trans, alpha, &A->storage.nullcoo, x, beta, y, num_flops);
    } else if (A->type == mtxompcsr) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompcsr_zgemv(
            trans, alpha, &A->storage.ompcsr, x, beta, y, num_flops);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_MATRIX_TYPE; }
}

/*
 * Level 3 BLAS operations
 */

/**
 * ‘mtxmatrix_sgemm()’ multiplies a matrix ‘A’ (or its transpose ‘A'’)
 * by a real scalar ‘alpha’ (‘α’) and a matrix ‘B’ (or its transpose
 * ‘B'’), before adding the result to another matrix ‘C’ multiplied by
 * a real scalar ‘beta’ (‘β’). That is,
 *
 * ‘C = α*op(A)*op(B) + β*C’, where ‘op(X)=X’ or ‘op(X)=X'’.
 *
 * The scalars ‘alpha’ and ‘beta’ are given as single precision
 * floating point numbers.
 */
int mtxmatrix_sgemm(
    enum mtxtransposition Atrans,
    enum mtxtransposition Btrans,
    float alpha,
    const struct mtxmatrix * A,
    const struct mtxmatrix * B,
    float beta,
    struct mtxmatrix * C,
    int64_t * num_flops)
{
    if (A->type != B->type || A->type != C->type) return MTX_ERR_INCOMPATIBLE_MATRIX_TYPE;
    if (A->type == mtxblasdense) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasdense_sgemm(
            Atrans, Btrans, alpha, &A->storage.blas, &B->storage.blas, beta, &C->storage.blas, num_flops);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    /* } else if (A->type == mtxbasecoo) { */
    /*     return mtxbasecoo_sgemm( */
    /*         Atrans, Btrans, alpha, &A->storage.coo, &B->storage.coo, beta, &C->storage.coo, num_flops); */
    /* } else if (A->type == mtxbasecsr) { */
    /*     return mtxbasecsr_sgemm( */
    /*         Atrans, Btrans, alpha, &A->storage.csr, &B->storage.csr, beta, &C->storage.csr, num_flops); */
    } else if (A->type == mtxbasedense) {
        return mtxbasedense_sgemm(
            Atrans, Btrans, alpha, &A->storage.dense, &B->storage.dense, beta, &C->storage.dense, num_flops);
/*     } else if (A->type == mtxnullcoo) { */
/*         return mtxnullcoo_sgemm( */
/*             Atrans, Btrans, alpha, &A->storage.nullcoo, &B->storage.nullcoo, beta, &C->storage.nullcoo, num_flops); */
/*     } else if (A->type == mtxompcsr) { */
/* #ifdef LIBMTX_HAVE_OPENMP */
/*         return mtxompcsr_sgemm( */
/*             Atrans, Btrans, alpha, &A->storage.ompcsr, &B->storage.ompcsr, beta, &C->storage.ompcsr, num_flops); */
/* #else */
/*         return MTX_ERR_OPENMP_NOT_SUPPORTED; */
/* #endif */
    } else { return MTX_ERR_INVALID_MATRIX_TYPE; }
}

/**
 * ‘mtxmatrix_dgemm()’ multiplies a matrix ‘A’ (or its transpose ‘A'’)
 * by a real scalar ‘alpha’ (‘α’) and a matrix ‘B’ (or its transpose
 * ‘B'’), before adding the result to another matrix ‘C’ multiplied by
 * a real scalar ‘beta’ (‘β’). That is,
 *
 * ‘C = α*op(A)*op(B) + β*C’, where ‘op(X)=X’ or ‘op(X)=X'’.
 *
 * The scalars ‘alpha’ and ‘beta’ are given as double precision
 * floating point numbers.
 */
int mtxmatrix_dgemm(
    enum mtxtransposition Atrans,
    enum mtxtransposition Btrans,
    double alpha,
    const struct mtxmatrix * A,
    const struct mtxmatrix * B,
    double beta,
    struct mtxmatrix * C,
    int64_t * num_flops)
{
    if (A->type != B->type || A->type != C->type) return MTX_ERR_INCOMPATIBLE_MATRIX_TYPE;
    if (A->type == mtxblasdense) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasdense_dgemm(
            Atrans, Btrans, alpha, &A->storage.blas, &B->storage.blas, beta, &C->storage.blas, num_flops);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
/*     } else if (A->type == mtxbasecoo) { */
/*         return mtxbasecoo_dgemm( */
/*             Atrans, Btrans, alpha, &A->storage.coo, &B->storage.coo, beta, &C->storage.coo, num_flops); */
/*     } else if (A->type == mtxbasecsr) { */
/*         return mtxbasecsr_dgemm( */
/*             Atrans, Btrans, alpha, &A->storage.csr, &B->storage.csr, beta, &C->storage.csr, num_flops); */
    } else if (A->type == mtxbasedense) {
        return mtxbasedense_dgemm(
            Atrans, Btrans, alpha, &A->storage.dense, &B->storage.dense, beta, &C->storage.dense, num_flops);
/*     } else if (A->type == mtxnullcoo) { */
/*         return mtxnullcoo_dgemm( */
/*             Atrans, Btrans, alpha, &A->storage.nullcoo, &B->storage.nullcoo, beta, &C->storage.nullcoo, num_flops); */
/*     } else if (A->type == mtxompcsr) { */
/* #ifdef LIBMTX_HAVE_OPENMP */
/*         return mtxompcsr_dgemm( */
/*             Atrans, Btrans, alpha, &A->storage.ompcsr, &B->storage.ompcsr, beta, &C->storage.ompcsr, num_flops); */
/* #else */
/*         return MTX_ERR_OPENMP_NOT_SUPPORTED; */
/* #endif */
    } else { return MTX_ERR_INVALID_MATRIX_TYPE; }
}
