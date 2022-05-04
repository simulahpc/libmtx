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
 * Last modified: 2022-05-02
 *
 * Data structures for matrices.
 */

#include <libmtx/libmtx-config.h>

#include <libmtx/error.h>
#include <libmtx/matrix/matrix.h>
#include <libmtx/mtxfile/mtxfile.h>
#include <libmtx/precision.h>
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
    case mtxmatrix_auto: return "auto";
    case mtxmatrix_array: return "array";
    case mtxmatrix_coordinate: return "coordinate";
    case mtxmatrix_csr: return "csr";
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
    if (strncmp("auto", t, strlen("auto")) == 0) {
        t += strlen("auto");
        *matrix_type = mtxmatrix_auto;
    } else if (strncmp("array", t, strlen("array")) == 0) {
        t += strlen("array");
        *matrix_type = mtxmatrix_array;
    } else if (strncmp("coordinate", t, strlen("coordinate")) == 0) {
        t += strlen("coordinate");
        *matrix_type = mtxmatrix_coordinate;
    } else if (strncmp("csr", t, strlen("csr")) == 0) {
        t += strlen("csr");
        *matrix_type = mtxmatrix_csr;
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
    if (A->type == mtxmatrix_array) {
        *field = A->storage.array.field;
        return MTX_SUCCESS;
    } else if (A->type == mtxmatrix_coordinate) {
        *field = A->storage.coordinate.a.field;
        return MTX_SUCCESS;
    } else if (A->type == mtxmatrix_csr) {
        *field = A->storage.csr.field;
        return MTX_SUCCESS;
    } else { return MTX_ERR_INVALID_MATRIX_TYPE; }
}

/**
 * ‘mtxmatrix_precision()’ gets the precision of a matrix.
 */
int mtxmatrix_precision(
    const struct mtxmatrix * A,
    enum mtxprecision * precision)
{
    if (A->type == mtxmatrix_array) {
        *precision = A->storage.array.precision;
    } else if (A->type == mtxmatrix_coordinate) {
        *precision = A->storage.coordinate.a.precision;
    } else if (A->type == mtxmatrix_csr) {
        *precision = A->storage.csr.precision;
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
    if (A->type == mtxmatrix_array) {
        *symmetry = A->storage.array.symmetry;
    } else if (A->type == mtxmatrix_coordinate) {
        *symmetry = A->storage.coordinate.symmetry;
    } else if (A->type == mtxmatrix_csr) {
        *symmetry = A->storage.csr.symmetry;
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
    if (A->type == mtxmatrix_array) {
        *num_nonzeros = A->storage.array.num_entries;
    } else if (A->type == mtxmatrix_coordinate) {
        *num_nonzeros = A->storage.coordinate.num_nonzeros;
    } else if (A->type == mtxmatrix_csr) {
        *num_nonzeros = A->storage.csr.num_nonzeros;
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
    if (A->type == mtxmatrix_array) {
        *size = A->storage.array.size;
    } else if (A->type == mtxmatrix_coordinate) {
        *size = A->storage.coordinate.size;
    } else if (A->type == mtxmatrix_csr) {
        *size = A->storage.csr.size;
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
    if (matrix->type == mtxmatrix_array) {
        mtxmatrix_array_free(&matrix->storage.array);
    } else if (matrix->type == mtxmatrix_coordinate) {
        mtxmatrix_coordinate_free(&matrix->storage.coordinate);
    } else if (matrix->type == mtxmatrix_csr) {
        mtxmatrix_csr_free(&matrix->storage.csr);
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
    if (src->type == mtxmatrix_array) {
        return mtxmatrix_array_alloc_copy(
            &dst->storage.array, &src->storage.array);
    } else if (src->type == mtxmatrix_coordinate) {
        return mtxmatrix_coordinate_alloc_copy(
            &dst->storage.coordinate, &src->storage.coordinate);
    } else if (src->type == mtxmatrix_csr) {
        return mtxmatrix_csr_alloc_copy(
            &dst->storage.csr, &src->storage.csr);
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
    if (src->type == mtxmatrix_array) {
        return mtxmatrix_array_init_copy(
            &dst->storage.array, &src->storage.array);
    } else if (src->type == mtxmatrix_coordinate) {
        return mtxmatrix_coordinate_init_copy(
            &dst->storage.coordinate, &src->storage.coordinate);
    } else if (src->type == mtxmatrix_csr) {
        return mtxmatrix_csr_init_copy(
            &dst->storage.csr, &src->storage.csr);
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
    if (type == mtxmatrix_array) {
        A->type = mtxmatrix_array;
        return MTX_ERR_INVALID_MATRIX_TYPE;
        /* return mtxmatrix_array_alloc_entries( */
        /*     &A->storage.array, field, precision, symmetry, */
        /*     num_rows, num_columns, num_nonzeros, rowidx, colidx); */
    } else if (type == mtxmatrix_coordinate) {
        A->type = mtxmatrix_coordinate;
        return mtxmatrix_coordinate_alloc_entries(
            &A->storage.coordinate, field, precision, symmetry,
            num_rows, num_columns, num_nonzeros,
            idxstride, idxbase, rowidx, colidx);
    } else if (type == mtxmatrix_csr) {
        A->type = mtxmatrix_csr;
        return MTX_ERR_INVALID_MATRIX_TYPE;
        /* return mtxmatrix_csr_alloc_entries( */
        /*     &A->storage.csr, field, precision, symmetry, */
        /*     num_rows, num_columns, num_nonzeros, rowidx, colidx); */
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
    if (type == mtxmatrix_coordinate) {
        A->type = mtxmatrix_coordinate;
        return mtxmatrix_coordinate_init_entries_real_single(
            &A->storage.coordinate, symmetry,
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
    if (type == mtxmatrix_coordinate) {
        A->type = mtxmatrix_coordinate;
        return mtxmatrix_coordinate_init_entries_real_double(
            &A->storage.coordinate, symmetry,
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
    if (type == mtxmatrix_coordinate) {
        A->type = mtxmatrix_coordinate;
        return mtxmatrix_coordinate_init_entries_complex_single(
            &A->storage.coordinate, symmetry,
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
    if (type == mtxmatrix_coordinate) {
        A->type = mtxmatrix_coordinate;
        return mtxmatrix_coordinate_init_entries_complex_double(
            &A->storage.coordinate, symmetry,
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
    if (type == mtxmatrix_coordinate) {
        A->type = mtxmatrix_coordinate;
        return mtxmatrix_coordinate_init_entries_integer_single(
            &A->storage.coordinate, symmetry,
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
    if (type == mtxmatrix_coordinate) {
        A->type = mtxmatrix_coordinate;
        return mtxmatrix_coordinate_init_entries_integer_double(
            &A->storage.coordinate, symmetry,
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
    if (type == mtxmatrix_coordinate) {
        A->type = mtxmatrix_coordinate;
        return mtxmatrix_coordinate_init_entries_pattern(
            &A->storage.coordinate, symmetry,
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
    int64_t idxstride,
    int idxbase,
    const int * rowidx,
    const int * colidx,
    int64_t datastride,
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
    int64_t idxstride,
    int idxbase,
    const int * rowidx,
    const int * colidx,
    int64_t datastride,
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
    int64_t idxstride,
    int idxbase,
    const int * rowidx,
    const int * colidx,
    int64_t datastride,
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
    int64_t idxstride,
    int idxbase,
    const int * rowidx,
    const int * colidx,
    int64_t datastride,
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
    int64_t idxstride,
    int idxbase,
    const int * rowidx,
    const int * colidx,
    int64_t datastride,
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
    int64_t idxstride,
    int idxbase,
    const int * rowidx,
    const int * colidx,
    int64_t datastride,
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
    int64_t idxstride,
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
    if (A->type == mtxmatrix_array) {
        return MTX_ERR_INVALID_MATRIX_TYPE;
        /* return mtxmatrix_array_setzero(&A->storage.array); */
    } else if (A->type == mtxmatrix_coordinate) {
        return mtxmatrix_coordinate_setzero(&A->storage.coordinate);
    } else if (A->type == mtxmatrix_csr) {
        return MTX_ERR_INVALID_MATRIX_TYPE;
        /* return mtxmatrix_csr_setzero(&A->storage.csr); */
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
    if (A->type == mtxmatrix_array) {
        return MTX_ERR_INVALID_MATRIX_TYPE;
        /* return mtxmatrix_array_set_real_single( */
        /*     &A->storage.array, size, stride, a); */
    } else if (A->type == mtxmatrix_coordinate) {
        return mtxmatrix_coordinate_set_real_single(
            &A->storage.coordinate, size, stride, a);
    } else if (A->type == mtxmatrix_csr) {
        return MTX_ERR_INVALID_MATRIX_TYPE;
        /* return mtxmatrix_csr_set_real_single( */
        /*     &A->storage.csr, size, stride, a); */
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
    if (A->type == mtxmatrix_array) {
        return MTX_ERR_INVALID_MATRIX_TYPE;
        /* return mtxmatrix_array_set_real_double( */
        /*     &A->storage.array, size, stride, a); */
    } else if (A->type == mtxmatrix_coordinate) {
        return mtxmatrix_coordinate_set_real_double(
            &A->storage.coordinate, size, stride, a);
    } else if (A->type == mtxmatrix_csr) {
        return MTX_ERR_INVALID_MATRIX_TYPE;
        /* return mtxmatrix_csr_set_real_double( */
        /*     &A->storage.csr, size, stride, a); */
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
    if (A->type == mtxmatrix_array) {
        return MTX_ERR_INVALID_MATRIX_TYPE;
        /* return mtxmatrix_array_set_complex_single( */
        /*     &A->storage.array, size, stride, a); */
    } else if (A->type == mtxmatrix_coordinate) {
        return mtxmatrix_coordinate_set_complex_single(
            &A->storage.coordinate, size, stride, a);
    } else if (A->type == mtxmatrix_csr) {
        return MTX_ERR_INVALID_MATRIX_TYPE;
        /* return mtxmatrix_csr_set_complex_single( */
        /*     &A->storage.csr, size, stride, a); */
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
    if (A->type == mtxmatrix_array) {
        return MTX_ERR_INVALID_MATRIX_TYPE;
        /* return mtxmatrix_array_set_complex_double( */
        /*     &A->storage.array, size, stride, a); */
    } else if (A->type == mtxmatrix_coordinate) {
        return mtxmatrix_coordinate_set_complex_double(
            &A->storage.coordinate, size, stride, a);
    } else if (A->type == mtxmatrix_csr) {
        return MTX_ERR_INVALID_MATRIX_TYPE;
        /* return mtxmatrix_csr_set_complex_double( */
        /*     &A->storage.csr, size, stride, a); */
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
    if (A->type == mtxmatrix_array) {
        return MTX_ERR_INVALID_MATRIX_TYPE;
        /* return mtxmatrix_array_set_integer_single( */
        /*     &A->storage.array, size, stride, a); */
    } else if (A->type == mtxmatrix_coordinate) {
        return mtxmatrix_coordinate_set_integer_single(
            &A->storage.coordinate, size, stride, a);
    } else if (A->type == mtxmatrix_csr) {
        return MTX_ERR_INVALID_MATRIX_TYPE;
        /* return mtxmatrix_csr_set_integer_single( */
        /*     &A->storage.csr, size, stride, a); */
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
    if (A->type == mtxmatrix_array) {
        return MTX_ERR_INVALID_MATRIX_TYPE;
        /* return mtxmatrix_array_set_integer_double( */
        /*     &A->storage.array, size, stride, a); */
    } else if (A->type == mtxmatrix_coordinate) {
        return mtxmatrix_coordinate_set_integer_double(
            &A->storage.coordinate, size, stride, a);
    } else if (A->type == mtxmatrix_csr) {
        return MTX_ERR_INVALID_MATRIX_TYPE;
        /* return mtxmatrix_csr_set_integer_double( */
        /*     &A->storage.csr, size, stride, a); */
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
    enum mtxvectortype vector_type)
{
    if (matrix->type == mtxmatrix_array) {
        return mtxmatrix_array_alloc_row_vector(
            &matrix->storage.array, vector, vector_type);
    } else if (matrix->type == mtxmatrix_coordinate) {
        return mtxmatrix_coordinate_alloc_row_vector(
            &matrix->storage.coordinate, vector, vector_type);
    } else if (matrix->type == mtxmatrix_csr) {
        return mtxmatrix_csr_alloc_row_vector(
            &matrix->storage.csr, vector, vector_type);
    } else {
        return MTX_ERR_INVALID_MATRIX_TYPE;
    }
}

/**
 * ‘mtxmatrix_alloc_column_vector()’ allocates a column vector for a
 * given matrix, where a column vector is a vector whose length equal
 * to a single column of the matrix.
 */
int mtxmatrix_alloc_column_vector(
    const struct mtxmatrix * matrix,
    struct mtxvector * vector,
    enum mtxvectortype vector_type)
{
    if (matrix->type == mtxmatrix_array) {
        return mtxmatrix_array_alloc_column_vector(
            &matrix->storage.array, vector, vector_type);
    } else if (matrix->type == mtxmatrix_coordinate) {
        return mtxmatrix_coordinate_alloc_column_vector(
            &matrix->storage.coordinate, vector, vector_type);
    } else if (matrix->type == mtxmatrix_csr) {
        return mtxmatrix_csr_alloc_column_vector(
            &matrix->storage.csr, vector, vector_type);
    } else {
        return MTX_ERR_INVALID_MATRIX_TYPE;
    }
}

/*
 * Matrix array formats
 */

/**
 * ‘mtxmatrix_alloc_array()’ allocates a matrix in array
 * format.
 */
int mtxmatrix_alloc_array(
    struct mtxmatrix * matrix,
    enum mtxfield field,
    enum mtxprecision precision,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns)
{
    matrix->type = mtxmatrix_array;
    return mtxmatrix_array_alloc(
        &matrix->storage.array, field, precision, symmetry, num_rows, num_columns);
}

/**
 * ‘mtxmatrix_init_array_real_single()’ allocates and initialises a
 * matrix in array format with real, single precision coefficients.
 */
int mtxmatrix_init_array_real_single(
    struct mtxmatrix * matrix,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const float * data)
{
    matrix->type = mtxmatrix_array;
    return mtxmatrix_array_init_real_single(
        &matrix->storage.array, symmetry, num_rows, num_columns, data);
}

/**
 * ‘mtxmatrix_init_array_real_double()’ allocates and initialises a
 * matrix in array format with real, double precision coefficients.
 */
int mtxmatrix_init_array_real_double(
    struct mtxmatrix * matrix,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const double * data)
{
    matrix->type = mtxmatrix_array;
    return mtxmatrix_array_init_real_double(
        &matrix->storage.array, symmetry, num_rows, num_columns, data);
}

/**
 * ‘mtxmatrix_init_array_complex_single()’ allocates and initialises a
 * matrix in array format with complex, single precision coefficients.
 */
int mtxmatrix_init_array_complex_single(
    struct mtxmatrix * matrix,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const float (* data)[2])
{
    matrix->type = mtxmatrix_array;
    return mtxmatrix_array_init_complex_single(
        &matrix->storage.array, symmetry, num_rows, num_columns, data);
}

/**
 * ‘mtxmatrix_init_array_complex_double()’ allocates and initialises a
 * matrix in array format with complex, double precision coefficients.
 */
int mtxmatrix_init_array_complex_double(
    struct mtxmatrix * matrix,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const double (* data)[2])
{
    matrix->type = mtxmatrix_array;
    return mtxmatrix_array_init_complex_double(
        &matrix->storage.array, symmetry, num_rows, num_columns, data);
}

/**
 * ‘mtxmatrix_init_array_integer_single()’ allocates and initialises a
 * matrix in array format with integer, single precision coefficients.
 */
int mtxmatrix_init_array_integer_single(
    struct mtxmatrix * matrix,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int32_t * data)
{
    matrix->type = mtxmatrix_array;
    return mtxmatrix_array_init_integer_single(
        &matrix->storage.array, symmetry, num_rows, num_columns, data);
}

/**
 * ‘mtxmatrix_init_array_integer_double()’ allocates and initialises a
 * matrix in array format with integer, double precision coefficients.
 */
int mtxmatrix_init_array_integer_double(
    struct mtxmatrix * matrix,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * data)
{
    matrix->type = mtxmatrix_array;
    return mtxmatrix_array_init_integer_double(
        &matrix->storage.array, symmetry, num_rows, num_columns, data);
}

/*
 * Compressed sparse row (CSR)
 */

/**
 * ‘mtxmatrix_alloc_csr()’ allocates a matrix in CSR format.
 */
int mtxmatrix_alloc_csr(
    struct mtxmatrix * matrix,
    enum mtxfield field,
    enum mtxprecision precision,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_nonzeros)
{
    matrix->type = mtxmatrix_csr;
    return mtxmatrix_csr_alloc(
        &matrix->storage.csr,
        field, precision, symmetry, num_rows, num_columns, num_nonzeros);
}

/**
 * ‘mtxmatrix_init_csr_real_single()’ allocates and initialises a
 * matrix in CSR format with real, single precision coefficients.
 */
int mtxmatrix_init_csr_real_single(
    struct mtxmatrix * matrix,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const float * data)
{
    matrix->type = mtxmatrix_csr;
    return mtxmatrix_csr_init_real_single(
        &matrix->storage.csr,
        symmetry, num_rows, num_columns, rowptr, colidx, data);
}

/**
 * ‘mtxmatrix_init_csr_real_double()’ allocates and initialises a
 * matrix in CSR format with real, double precision coefficients.
 */
int mtxmatrix_init_csr_real_double(
    struct mtxmatrix * matrix,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const double * data)
{
    matrix->type = mtxmatrix_csr;
    return mtxmatrix_csr_init_real_double(
        &matrix->storage.csr,
        symmetry, num_rows, num_columns, rowptr, colidx, data);
}

/**
 * ‘mtxmatrix_init_csr_complex_single()’ allocates and initialises a
 * matrix in CSR format with complex, single precision coefficients.
 */
int mtxmatrix_init_csr_complex_single(
    struct mtxmatrix * matrix,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const float (* data)[2])
{
    matrix->type = mtxmatrix_csr;
    return mtxmatrix_csr_init_complex_single(
        &matrix->storage.csr,
        symmetry, num_rows, num_columns, rowptr, colidx, data);
}

/**
 * ‘mtxmatrix_init_csr_complex_double()’ allocates and initialises a
 * matrix in CSR format with complex, double precision coefficients.
 */
int mtxmatrix_init_csr_complex_double(
    struct mtxmatrix * matrix,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const double (* data)[2])
{
    matrix->type = mtxmatrix_csr;
    return mtxmatrix_csr_init_complex_double(
        &matrix->storage.csr,
        symmetry, num_rows, num_columns, rowptr, colidx, data);
}

/**
 * ‘mtxmatrix_init_csr_integer_single()’ allocates and initialises a
 * matrix in CSR format with integer, single precision coefficients.
 */
int mtxmatrix_init_csr_integer_single(
    struct mtxmatrix * matrix,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const int32_t * data)
{
    matrix->type = mtxmatrix_csr;
    return mtxmatrix_csr_init_integer_single(
        &matrix->storage.csr,
        symmetry, num_rows, num_columns, rowptr, colidx, data);
}

/**
 * ‘mtxmatrix_init_csr_integer_double()’ allocates and initialises a
 * matrix in CSR format with integer, double precision coefficients.
 */
int mtxmatrix_init_csr_integer_double(
    struct mtxmatrix * matrix,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx,
    const int64_t * data)
{
    matrix->type = mtxmatrix_csr;
    return mtxmatrix_csr_init_integer_double(
        &matrix->storage.csr,
        symmetry, num_rows, num_columns, rowptr, colidx, data);
}

/**
 * ‘mtxmatrix_init_csr_pattern()’ allocates and initialises a matrix
 * in CSR format with integer, double precision coefficients.
 */
int mtxmatrix_init_csr_pattern(
    struct mtxmatrix * matrix,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * rowptr,
    const int * colidx)
{
    matrix->type = mtxmatrix_csr;
    return mtxmatrix_csr_init_pattern(
        &matrix->storage.csr,
        symmetry,num_rows, num_columns, rowptr, colidx);
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
    if (type == mtxmatrix_auto) {
        if (mtxfile->header.format == mtxfile_array) {
            type = mtxmatrix_array;
        } else if (mtxfile->header.format == mtxfile_coordinate) {
            type = mtxmatrix_coordinate;
        } else { return MTX_ERR_INVALID_MTX_FORMAT; }
    }

    if (type == mtxmatrix_array) {
        matrix->type = mtxmatrix_array;
        return mtxmatrix_array_from_mtxfile(
            &matrix->storage.array, mtxfile);
    } else if (type == mtxmatrix_coordinate) {
        matrix->type = mtxmatrix_coordinate;
        return mtxmatrix_coordinate_from_mtxfile(
            &matrix->storage.coordinate, mtxfile);
    } else if (type == mtxmatrix_csr) {
        matrix->type = mtxmatrix_csr;
        return mtxmatrix_csr_from_mtxfile(
            &matrix->storage.csr, mtxfile);
    } else if (type == mtxmatrix_dense) {
        matrix->type = mtxmatrix_dense;
        return mtxmatrix_dense_from_mtxfile(
            &matrix->storage.dense, mtxfile);
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
    if (src->type == mtxmatrix_array) {
        return mtxmatrix_array_to_mtxfile(
            dst, &src->storage.array, mtxfmt);
    } else if (src->type == mtxmatrix_coordinate) {
        return mtxmatrix_coordinate_to_mtxfile(
            dst, &src->storage.coordinate,
            num_rows, rowidx, num_columns, colidx, mtxfmt);
    } else if (src->type == mtxmatrix_csr) {
        return mtxmatrix_csr_to_mtxfile(
            dst, &src->storage.csr, mtxfmt);
    } else if (src->type == mtxmatrix_dense) {
        return mtxmatrix_dense_to_mtxfile(
            dst, &src->storage.dense,
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
 * the matrix.  If ‘type’ is ‘mtxmatrix_auto’, then the underlying
 * matrix is stored in array format or coordinate format according to
 * the format of the Matrix Market file.  Otherwise, an attempt is
 * made to convert the matrix to the desired type.
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
 * the matrix.  If ‘type’ is ‘mtxmatrix_auto’, then the underlying
 * matrix is stored in array format or coordinate format according to
 * the format of the Matrix Market file.  Otherwise, an attempt is
 * made to convert the matrix to the desired type.
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
 * the matrix.  If ‘type’ is ‘mtxmatrix_auto’, then the underlying
 * matrix is stored in array format or coordinate format according to
 * the format of the Matrix Market file.  Otherwise, an attempt is
 * made to convert the matrix to the desired type.
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
 * Nonzero rows and columns
 */

/**
 * ‘mtxmatrix_nzrows()’ counts the number of nonzero (non-empty)
 * matrix rows, and, optionally, fills an array with the row indices
 * of the nonzero (non-empty) matrix rows.
 *
 * If ‘num_nonzero_rows’ is ‘NULL’, then it is ignored, or else it
 * must point to an integer that is used to store the number of
 * nonzero matrix rows.
 *
 * ‘nonzero_rows’ may be ‘NULL’, in which case it is ignored.
 * Otherwise, it must point to an array of length at least equal to
 * ‘size’. On successful completion, this array contains the row
 * indices of the nonzero matrix rows. Note that ‘size’ must be at
 * least equal to the number of non-zero rows.
 */
int mtxmatrix_nzrows(
    const struct mtxmatrix * matrix,
    int * num_nonzero_rows,
    int size,
    int * nonzero_rows)
{
    if (matrix->type == mtxmatrix_array) {
        return mtxmatrix_array_nzrows(
            &matrix->storage.array, num_nonzero_rows, size, nonzero_rows);
    } else if (matrix->type == mtxmatrix_csr) {
        return mtxmatrix_csr_nzrows(
            &matrix->storage.csr, num_nonzero_rows, size, nonzero_rows);
    } else {
        return MTX_ERR_INVALID_MATRIX_TYPE;
    }
}

/**
 * ‘mtxmatrix_nzcols()’ counts the number of nonzero (non-empty)
 * matrix columns, and, optionally, fills an array with the column
 * indices of the nonzero (non-empty) matrix columns.
 *
 * If ‘num_nonzero_columns’ is ‘NULL’, then it is ignored, or else it
 * must point to an integer that is used to store the number of
 * nonzero matrix columns.
 *
 * ‘nonzero_columns’ may be ‘NULL’, in which case it is ignored.
 * Otherwise, it must point to an array of length at least equal to
 * ‘size’. On successful completion, this array contains the column
 * indices of the nonzero matrix columns. Note that ‘size’ must be at
 * least equal to the number of non-zero columns.
 */
int mtxmatrix_nzcols(
    const struct mtxmatrix * matrix,
    int * num_nonzero_columns,
    int size,
    int * nonzero_columns)
{
    if (matrix->type == mtxmatrix_array) {
        return mtxmatrix_array_nzcols(
            &matrix->storage.array, num_nonzero_columns, size, nonzero_columns);
    } else if (matrix->type == mtxmatrix_csr) {
        return mtxmatrix_csr_nzcols(
            &matrix->storage.csr, num_nonzero_columns, size, nonzero_columns);
    } else {
        return MTX_ERR_INVALID_MATRIX_TYPE;
    }
}

/*
 * Partitioning
 */

/**
 * ‘mtxmatrix_partition()’ partitions a matrix into blocks according
 * to the given row and column partitions.
 *
 * The partitions ‘rowpart’ or ‘colpart’ are allowed to be ‘NULL’, in
 * which case a trivial, singleton partition is used for the rows or
 * columns, respectively.
 *
 * Otherwise, ‘rowpart’ and ‘colpart’ must partition the rows and
 * columns of the matrix ‘src’, respectively. That is, ‘rowpart->size’
 * must be equal to the number of matrix rows, and ‘colpart->size’
 * must be equal to the number of matrix columns.
 *
 * The argument ‘dsts’ is an array that must have enough storage for
 * ‘P*Q’ values of type ‘struct mtxmatrix’, where ‘P’ is the number of
 * row parts, ‘rowpart->num_parts’, and ‘Q’ is the number of column
 * parts, ‘colpart->num_parts’. Note that the ‘r’th part corresponds
 * to a row part ‘p’ and column part ‘q’, such that ‘r=p*Q+q’. Thus,
 * the ‘r’th entry of ‘dsts’ is the submatrix corresponding to the
 * ‘p’th row and ‘q’th column of the 2D partitioning.
 *
 * The user is responsible for freeing storage allocated for each
 * matrix in the ‘dsts’ array.
 */
int mtxmatrix_partition(
    struct mtxmatrix * dsts,
    const struct mtxmatrix * src,
    const struct mtxpartition * rowpart,
    const struct mtxpartition * colpart)
{
    if (src->type == mtxmatrix_array) {
        return mtxmatrix_array_partition(
            dsts, &src->storage.array, rowpart, colpart);
    } else if (src->type == mtxmatrix_csr) {
        return mtxmatrix_csr_partition(
            dsts, &src->storage.csr, rowpart, colpart);
    } else {
        return MTX_ERR_INVALID_MATRIX_TYPE;
    }
}

/**
 * ‘mtxmatrix_join()’ joins together matrices representing compatible
 * blocks of a partitioned matrix to form a larger matrix.
 *
 * The argument ‘srcs’ is logically arranged as a two-dimensional
 * array of size ‘P*Q’, where ‘P’ is the number of row parts
 * (‘rowpart->num_parts’) and ‘Q’ is the number of column parts
 * (‘colpart->num_parts’).  Note that the ‘r’th part corresponds to a
 * row part ‘p’ and column part ‘q’, such that ‘r=p*Q+q’. Thus, the
 * ‘r’th entry of ‘srcs’ is the submatrix corresponding to the ‘p’th
 * row and ‘q’th column of the 2D partitioning.
 *
 * Moreover, the blocks must be compatible, which means that each part
 * in the same block row ‘p’, must have the same number of rows.
 * Similarly, each part in the same block column ‘q’ must have the
 * same number of columns. Finally, for each block column ‘q’, the sum
 * of the number of rows of ‘srcs[p*Q+q]’ for ‘p=0,1,...,P-1’ must be
 * equal to ‘rowpart->size’. Likewise, for each block row ‘p’, the sum
 * of the number of columns of ‘srcs[p*Q+q]’ for ‘q=0,1,...,Q-1’ must
 * be equal to ‘colpart->size’.
 */
int mtxmatrix_join(
    struct mtxmatrix * dst,
    const struct mtxmatrix * srcs,
    const struct mtxpartition * rowpart,
    const struct mtxpartition * colpart)
{
    int num_row_parts = rowpart ? rowpart->num_parts : 1;
    int num_col_parts = colpart ? colpart->num_parts : 1;
    int num_parts = num_row_parts * num_col_parts;
    if (num_parts <= 0)
        return MTX_SUCCESS;
    if (srcs[0].type == mtxmatrix_array) {
        dst->type = mtxmatrix_array;
        return mtxmatrix_array_join(
            &dst->storage.array, srcs, rowpart, colpart);
    } else if (srcs[0].type == mtxmatrix_csr) {
        dst->type = mtxmatrix_csr;
        return mtxmatrix_csr_join(
            &dst->storage.csr, srcs, rowpart, colpart);
    } else {
        return MTX_ERR_INVALID_MATRIX_TYPE;
    }
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
    if (X->type == mtxmatrix_array && Y->type == mtxmatrix_array) {
        return mtxmatrix_array_swap(
            &X->storage.array, &Y->storage.array);
    } else if (X->type == mtxmatrix_coordinate && Y->type == mtxmatrix_coordinate) {
        return mtxmatrix_coordinate_swap(
            &X->storage.coordinate, &Y->storage.coordinate);
    } else if (X->type == mtxmatrix_csr && Y->type == mtxmatrix_csr) {
        return mtxmatrix_csr_swap(
            &X->storage.csr, &Y->storage.csr);
    } else {
        return MTX_ERR_INVALID_MATRIX_TYPE;
    }
}

/**
 * ‘mtxmatrix_copy()’ copies values of a matrix, ‘Y = X’.
 */
int mtxmatrix_copy(
    struct mtxmatrix * Y,
    const struct mtxmatrix * X)
{
    if (X->type == mtxmatrix_array && Y->type == mtxmatrix_array) {
        return mtxmatrix_array_copy(
            &Y->storage.array, &X->storage.array);
    } else if (X->type == mtxmatrix_coordinate && Y->type == mtxmatrix_coordinate) {
        return mtxmatrix_coordinate_copy(
            &Y->storage.coordinate, &X->storage.coordinate);
    } else if (X->type == mtxmatrix_csr && Y->type == mtxmatrix_csr) {
        return mtxmatrix_csr_copy(
            &Y->storage.csr, &X->storage.csr);
    } else {
        return MTX_ERR_INVALID_MATRIX_TYPE;
    }
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
    if (X->type == mtxmatrix_array) {
        return mtxmatrix_array_sscal(
            a, &X->storage.array, num_flops);
    } else if (X->type == mtxmatrix_coordinate) {
        return mtxmatrix_coordinate_sscal(
            a, &X->storage.coordinate, num_flops);
    } else if (X->type == mtxmatrix_csr) {
        return mtxmatrix_csr_sscal(
            a, &X->storage.csr, num_flops);
    } else {
        return MTX_ERR_INVALID_MATRIX_TYPE;
    }
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
    if (X->type == mtxmatrix_array) {
        return mtxmatrix_array_dscal(
            a, &X->storage.array, num_flops);
    } else if (X->type == mtxmatrix_coordinate) {
        return mtxmatrix_coordinate_dscal(
            a, &X->storage.coordinate, num_flops);
    } else if (X->type == mtxmatrix_csr) {
        return mtxmatrix_csr_dscal(
            a, &X->storage.csr, num_flops);
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
    /* if (X->type == mtxmatrix_array) { */
    /*     return mtxmatrix_array_cscal( */
    /*         a, &X->storage.array, num_flops); */
    /* } else if (X->type == mtxmatrix_coordinate) { */
    /*     return mtxmatrix_coordinate_cscal( */
    /*         a, &X->storage.coordinate, num_flops); */
    /* } else if (X->type == mtxmatrix_csr) { */
    /*     return mtxmatrix_csr_cscal( */
    /*         a, &X->storage.csr, num_flops); */
    /* } else { return MTX_ERR_INVALID_MATRIX_TYPE; } */
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
    /* if (X->type == mtxmatrix_array) { */
    /*     return mtxmatrix_array_zscal( */
    /*         a, &X->storage.array, num_flops); */
    /* } else if (X->type == mtxmatrix_coordinate) { */
    /*     return mtxmatrix_coordinate_zscal( */
    /*         a, &X->storage.coordinate, num_flops); */
    /* } else if (X->type == mtxmatrix_csr) { */
    /*     return mtxmatrix_csr_zscal( */
    /*         a, &X->storage.csr, num_flops); */
    /* } else { return MTX_ERR_INVALID_MATRIX_TYPE; } */
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
    if (X->type == mtxmatrix_array && Y->type == mtxmatrix_array) {
        return mtxmatrix_array_saxpy(
            a, &X->storage.array, &Y->storage.array, num_flops);
    } else if (X->type == mtxmatrix_coordinate && Y->type == mtxmatrix_coordinate) {
        return mtxmatrix_coordinate_saxpy(
            a, &X->storage.coordinate, &Y->storage.coordinate, num_flops);
    } else if (X->type == mtxmatrix_csr && Y->type == mtxmatrix_csr) {
        return mtxmatrix_csr_saxpy(
            a, &X->storage.csr, &Y->storage.csr, num_flops);
    } else {
        return MTX_ERR_INVALID_MATRIX_TYPE;
    }
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
    if (X->type == mtxmatrix_array && Y->type == mtxmatrix_array) {
        return mtxmatrix_array_daxpy(
            a, &X->storage.array, &Y->storage.array, num_flops);
    } else if (X->type == mtxmatrix_coordinate && Y->type == mtxmatrix_coordinate) {
        return mtxmatrix_coordinate_daxpy(
            a, &X->storage.coordinate, &Y->storage.coordinate, num_flops);
    } else if (X->type == mtxmatrix_csr && Y->type == mtxmatrix_csr) {
        return mtxmatrix_csr_daxpy(
            a, &X->storage.csr, &Y->storage.csr, num_flops);
    } else {
        return MTX_ERR_INVALID_MATRIX_TYPE;
    }
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
    if (X->type == mtxmatrix_array && Y->type == mtxmatrix_array) {
        return mtxmatrix_array_saypx(
            a, &Y->storage.array, &X->storage.array, num_flops);
    } else if (X->type == mtxmatrix_coordinate && Y->type == mtxmatrix_coordinate) {
        return mtxmatrix_coordinate_saypx(
            a, &Y->storage.coordinate, &X->storage.coordinate, num_flops);
    } else if (X->type == mtxmatrix_csr && Y->type == mtxmatrix_csr) {
        return mtxmatrix_csr_saypx(
            a, &Y->storage.csr, &X->storage.csr, num_flops);
    } else {
        return MTX_ERR_INVALID_MATRIX_TYPE;
    }
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
    if (X->type == mtxmatrix_array && Y->type == mtxmatrix_array) {
        return mtxmatrix_array_daypx(
            a, &Y->storage.array, &X->storage.array, num_flops);
    } else if (X->type == mtxmatrix_coordinate && Y->type == mtxmatrix_coordinate) {
        return mtxmatrix_coordinate_daypx(
            a, &Y->storage.coordinate, &X->storage.coordinate, num_flops);
    } else if (X->type == mtxmatrix_csr && Y->type == mtxmatrix_csr) {
        return mtxmatrix_csr_daypx(
            a, &Y->storage.csr, &X->storage.csr, num_flops);
    } else {
        return MTX_ERR_INVALID_MATRIX_TYPE;
    }
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
    if (X->type == mtxmatrix_array && Y->type == mtxmatrix_array) {
        return mtxmatrix_array_sdot(
            &X->storage.array, &Y->storage.array, dot, num_flops);
    } else if (X->type == mtxmatrix_coordinate && Y->type == mtxmatrix_coordinate) {
        return mtxmatrix_coordinate_sdot(
            &X->storage.coordinate, &Y->storage.coordinate, dot, num_flops);
    } else if (X->type == mtxmatrix_csr && Y->type == mtxmatrix_csr) {
        return mtxmatrix_csr_sdot(
            &X->storage.csr, &Y->storage.csr, dot, num_flops);
    } else {
        return MTX_ERR_INVALID_MATRIX_TYPE;
    }
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
    if (X->type == mtxmatrix_array && Y->type == mtxmatrix_array) {
        return mtxmatrix_array_ddot(
            &X->storage.array, &Y->storage.array, dot, num_flops);
    } else if (X->type == mtxmatrix_coordinate && Y->type == mtxmatrix_coordinate) {
        return mtxmatrix_coordinate_ddot(
            &X->storage.coordinate, &Y->storage.coordinate, dot, num_flops);
    } else if (X->type == mtxmatrix_csr && Y->type == mtxmatrix_csr) {
        return mtxmatrix_csr_ddot(
            &X->storage.csr, &Y->storage.csr, dot, num_flops);
    } else {
        return MTX_ERR_INVALID_MATRIX_TYPE;
    }
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
    if (X->type == mtxmatrix_array && Y->type == mtxmatrix_array) {
        return mtxmatrix_array_cdotu(
            &X->storage.array, &Y->storage.array, dot, num_flops);
    } else if (X->type == mtxmatrix_coordinate && Y->type == mtxmatrix_coordinate) {
        return mtxmatrix_coordinate_cdotu(
            &X->storage.coordinate, &Y->storage.coordinate, dot, num_flops);
    } else if (X->type == mtxmatrix_csr && Y->type == mtxmatrix_csr) {
        return mtxmatrix_csr_cdotu(
            &X->storage.csr, &Y->storage.csr, dot, num_flops);
    } else {
        return MTX_ERR_INVALID_MATRIX_TYPE;
    }
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
    if (X->type == mtxmatrix_array && Y->type == mtxmatrix_array) {
        return mtxmatrix_array_zdotu(
            &X->storage.array, &Y->storage.array, dot, num_flops);
    } else if (X->type == mtxmatrix_coordinate && Y->type == mtxmatrix_coordinate) {
        return mtxmatrix_coordinate_zdotu(
            &X->storage.coordinate, &Y->storage.coordinate, dot, num_flops);
    } else if (X->type == mtxmatrix_csr && Y->type == mtxmatrix_csr) {
        return mtxmatrix_csr_zdotu(
            &X->storage.csr, &Y->storage.csr, dot, num_flops);
    } else {
        return MTX_ERR_INVALID_MATRIX_TYPE;
    }
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
    if (X->type == mtxmatrix_array && Y->type == mtxmatrix_array) {
        return mtxmatrix_array_cdotc(
            &X->storage.array, &Y->storage.array, dot, num_flops);
    } else if (X->type == mtxmatrix_coordinate && Y->type == mtxmatrix_coordinate) {
        return mtxmatrix_coordinate_cdotc(
            &X->storage.coordinate, &Y->storage.coordinate, dot, num_flops);
    } else if (X->type == mtxmatrix_csr && Y->type == mtxmatrix_csr) {
        return mtxmatrix_csr_cdotc(
            &X->storage.csr, &Y->storage.csr, dot, num_flops);
    } else {
        return MTX_ERR_INVALID_MATRIX_TYPE;
    }
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
    if (X->type == mtxmatrix_array && Y->type == mtxmatrix_array) {
        return mtxmatrix_array_zdotc(
            &X->storage.array, &Y->storage.array, dot, num_flops);
    } else if (X->type == mtxmatrix_coordinate && Y->type == mtxmatrix_coordinate) {
        return mtxmatrix_coordinate_zdotc(
            &X->storage.coordinate, &Y->storage.coordinate, dot, num_flops);
    } else if (X->type == mtxmatrix_csr && Y->type == mtxmatrix_csr) {
        return mtxmatrix_csr_zdotc(
            &X->storage.csr, &Y->storage.csr, dot, num_flops);
    } else {
        return MTX_ERR_INVALID_MATRIX_TYPE;
    }
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
    if (X->type == mtxmatrix_array) {
        return mtxmatrix_array_snrm2(&X->storage.array, nrm2, num_flops);
    } else if (X->type == mtxmatrix_coordinate) {
        return mtxmatrix_coordinate_snrm2(&X->storage.coordinate, nrm2, num_flops);
    } else if (X->type == mtxmatrix_csr) {
        return mtxmatrix_csr_snrm2(&X->storage.csr, nrm2, num_flops);
    } else {
        return MTX_ERR_INVALID_MATRIX_TYPE;
    }
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
    if (X->type == mtxmatrix_array) {
        return mtxmatrix_array_dnrm2(&X->storage.array, nrm2, num_flops);
    } else if (X->type == mtxmatrix_coordinate) {
        return mtxmatrix_coordinate_dnrm2(&X->storage.coordinate, nrm2, num_flops);
    } else if (X->type == mtxmatrix_csr) {
        return mtxmatrix_csr_dnrm2(&X->storage.csr, nrm2, num_flops);
    } else {
        return MTX_ERR_INVALID_MATRIX_TYPE;
    }
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
    if (X->type == mtxmatrix_array) {
        return mtxmatrix_array_sasum(&X->storage.array, asum, num_flops);
    } else if (X->type == mtxmatrix_coordinate) {
        return mtxmatrix_coordinate_sasum(&X->storage.coordinate, asum, num_flops);
    } else if (X->type == mtxmatrix_csr) {
        return mtxmatrix_csr_sasum(&X->storage.csr, asum, num_flops);
    } else {
        return MTX_ERR_INVALID_MATRIX_TYPE;
    }
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
    if (X->type == mtxmatrix_array) {
        return mtxmatrix_array_dasum(&X->storage.array, asum, num_flops);
    } else if (X->type == mtxmatrix_coordinate) {
        return mtxmatrix_coordinate_dasum(&X->storage.coordinate, asum, num_flops);
    } else if (X->type == mtxmatrix_csr) {
        return mtxmatrix_csr_dasum(&X->storage.csr, asum, num_flops);
    } else {
        return MTX_ERR_INVALID_MATRIX_TYPE;
    }
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
    if (X->type == mtxmatrix_array) {
        return mtxmatrix_array_iamax(&X->storage.array, iamax);
    } else if (X->type == mtxmatrix_coordinate) {
        return mtxmatrix_coordinate_iamax(&X->storage.coordinate, iamax);
    } else if (X->type == mtxmatrix_csr) {
        return mtxmatrix_csr_iamax(&X->storage.csr, iamax);
    } else {
        return MTX_ERR_INVALID_MATRIX_TYPE;
    }
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
    if (A->type == mtxmatrix_array) {
        return mtxmatrix_array_sgemv(
            trans, alpha, &A->storage.array, x, beta, y, num_flops);
    } else if (A->type == mtxmatrix_coordinate) {
        return mtxmatrix_coordinate_sgemv(
            trans, alpha, &A->storage.coordinate, x, beta, y, num_flops);
    } else if (A->type == mtxmatrix_csr) {
        return mtxmatrix_csr_sgemv(
            trans, alpha, &A->storage.csr, x, beta, y, num_flops);
    } else {
        return MTX_ERR_INVALID_MATRIX_TYPE;
    }
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
    if (A->type == mtxmatrix_array) {
        return mtxmatrix_array_dgemv(
            trans, alpha, &A->storage.array, x, beta, y, num_flops);
    } else if (A->type == mtxmatrix_coordinate) {
        return mtxmatrix_coordinate_dgemv(
            trans, alpha, &A->storage.coordinate, x, beta, y, num_flops);
    } else if (A->type == mtxmatrix_csr) {
        return mtxmatrix_csr_dgemv(
            trans, alpha, &A->storage.csr, x, beta, y, num_flops);
    } else {
        return MTX_ERR_INVALID_MATRIX_TYPE;
    }
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
    if (A->type == mtxmatrix_array) {
        return mtxmatrix_array_cgemv(
            trans, alpha, &A->storage.array, x, beta, y, num_flops);
    } else if (A->type == mtxmatrix_coordinate) {
        return mtxmatrix_coordinate_cgemv(
            trans, alpha, &A->storage.coordinate, x, beta, y, num_flops);
    } else if (A->type == mtxmatrix_csr) {
        return mtxmatrix_csr_cgemv(
            trans, alpha, &A->storage.csr, x, beta, y, num_flops);
    } else {
        return MTX_ERR_INVALID_MATRIX_TYPE;
    }
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
    if (A->type == mtxmatrix_array) {
        return mtxmatrix_array_zgemv(
            trans, alpha, &A->storage.array, x, beta, y, num_flops);
    } else if (A->type == mtxmatrix_coordinate) {
        return mtxmatrix_coordinate_zgemv(
            trans, alpha, &A->storage.coordinate, x, beta, y, num_flops);
    } else if (A->type == mtxmatrix_csr) {
        return mtxmatrix_csr_zgemv(
            trans, alpha, &A->storage.csr, x, beta, y, num_flops);
    } else {
        return MTX_ERR_INVALID_MATRIX_TYPE;
    }
}
